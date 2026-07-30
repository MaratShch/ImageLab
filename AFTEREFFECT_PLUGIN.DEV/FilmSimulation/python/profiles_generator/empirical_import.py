#!/usr/bin/env python3
"""Import an empirical film analysis (the SVEMA64_MY generator format) and
turn it into vetted FilmProfile field changes.

Usage:
    python3 empirical_import.py MEASURED.txt --profile SVEMA_FN_64
    python3 empirical_import.py MEASURED.txt --profile SVEMA_FN_64 --emit-patch

The point of this tool is NOT to copy numbers across. Scan-derived analyses
mix three kinds of value and only one of them belongs in a profile:

  MEASURED     robust to the scanning setup            -> adopt
  CONTAMINATED measured, but folded with scanner state  -> adopt with caveat
  DEFAULT/BOGUS the generator's fallback, or a number    -> reject, keep the
               that fails physics                          profile's own value

Every field is classified with a stated reason, and the verdicts are printed
as a report. --emit-patch additionally prints the Python fragment to paste
into film_profiles.py.

Field-by-field policy (derived from the SVEMA64_MY batch, 290 frames):

dmin_r/g/b      ADOPT (mean) if 0.05..0.45 and channel spread < 0.010 D
                (absolute -- a relative test punishes low densities unfairly).
                Base+fog is the brightest thing on a negative scan and needs
                no exposure reference, so it survives a DSLR rig. Aged stock
                only ever rises, so a value above the old estimate is doubly
                plausible.

gamma_r/g/b     REJECT unless it passes the self-consistency test:
                (dmax - dmin) / gamma = the log-exposure span of the batch's
                straight-line content. For hundreds of ordinary frames that
                span must be roughly 0.5..1.5 logE. A gamma of 3.0 with a
                0.638 density span implies 0.21 logE = 0.7 stops of scene
                across every frame -- impossible, so the number is an
                artefact of the estimator, not a property of the film.
                Measuring real gamma needs a known exposure series (a step
                wedge or bracketed frames of one scene), not found footage.

dmax            NEVER mapped to the curve shoulder. The densest pixel in a
                batch of scenes is a lower bound on what the emulsion can do,
                not the emulsion's Dmax. Reported for information only.

tint_r/g/b      ADOPT into base_tint, normalised to green, if each channel
                is within 0.9..1.1. CONTAMINATED tier: the scanning
                illuminant and camera white balance are folded in. Kept
                separate from silver_tone on purpose -- base yellowing is
                uniform, image-silver tone is density-weighted.

clump_um,       REJECT when they match the generator's documented defaults
clump_gain,     (25.0 / 0.35 / 15.0 / 1.0) -- the generator itself labels the
rms_granularity, block "Empirical defaults". Grain SIZE additionally cannot
anisotropy      be measured below the scan's own resolution: at typical DSLR
                scan pitch one pixel spans more micrometres than a clump.

kind            IGNORE; the generator guesses.
"""
from __future__ import annotations

import argparse
import re
import sys

GENERATOR_GRAIN_DEFAULTS = {
    "clump_um": 25.0, "clump_gain": 0.35,
    "rms_granularity": 15.0, "anisotropy": 1.0,
}

# Plausible log-exposure span of straight-line content across a large batch
# of ordinary scenes. Below the floor the gamma estimate is self-contradictory.
LOGE_SPAN_MIN = 0.45
LOGE_SPAN_MAX = 1.60


def parse(path: str) -> dict:
    """Parse the generator's INI-with-//-comments format."""
    data: dict = {}
    section = ""
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.split("//")[0].strip()
        if not line:
            continue
        m = re.match(r"\[(\w+)\]", line)
        if m:
            section = m.group(1)
            continue
        m = re.match(r"(\w+)\s*=\s*(.+)", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if val.startswith('"'):
                val = val.strip('"')
            else:
                try:
                    val = float(val)
                except ValueError:
                    pass
            data.setdefault(section, {})[key] = val
    return data


def assess(d: dict) -> tuple[list, dict]:
    """Return ([(field, verdict, value, reason)], patch_fields)."""
    out = []
    patch: dict = {}
    cur = d.get("Curves", {})
    tin = d.get("BaseTint", {})
    grn = d.get("GrainSpec", {})

    # ---- dmin -------------------------------------------------------------
    dmins = [cur.get("dmin_" + c) for c in "rgb"]
    dmin_mean = None
    if all(isinstance(v, float) for v in dmins):
        mean = sum(dmins) / 3.0
        dmin_mean = mean
        spread = max(dmins) - min(dmins)   # absolute density spread
        if 0.05 <= mean <= 0.45 and spread < 0.010:
            out.append(("dmin", "ADOPT", round(mean, 4),
                        "base+fog survives a DSLR rig; %.4f D channel spread "
                        "= neutral base" % spread))
            patch["dmin"] = round(mean, 3)
        else:
            out.append(("dmin", "REJECT", mean,
                        "outside 0.05..0.45 or channels disagree by "
                        ">0.010 D (%.4f D)" % spread))

    # ---- gamma: the self-consistency test ----------------------------------
    gammas = [cur.get("gamma_" + c) for c in "rgb"]
    dmaxs = [cur.get("dmax_" + c) for c in "rgb"]
    if all(isinstance(v, float) for v in gammas + dmaxs) and dmin_mean is not None:
        g = sum(gammas) / 3.0
        span = sum(dmaxs) / 3.0 - dmin_mean
        loge = span / g if g > 0 else 0.0
        if LOGE_SPAN_MIN <= loge <= LOGE_SPAN_MAX:
            out.append(("gamma", "ADOPT", round(g, 3),
                        "consistent: batch density span %.3f / gamma %.2f = "
                        "%.2f logE of scene content" % (span, g, loge)))
            patch["gamma"] = round(g, 3)
        else:
            out.append(("gamma", "REJECT", g,
                        "self-contradictory: density span %.3f at gamma %.2f "
                        "implies %.2f logE = %.1f stops of scene across the "
                        "whole batch; real batches span %.2f..%.2f logE. "
                        "Measure gamma with a step wedge, not found footage."
                        % (span, g, loge, loge / 0.301,
                           LOGE_SPAN_MIN, LOGE_SPAN_MAX)))

    # ---- dmax: information only --------------------------------------------
    if all(isinstance(v, float) for v in dmaxs):
        out.append(("dmax", "INFO", round(sum(dmaxs) / 3.0, 4),
                    "densest pixel in the batch = LOWER BOUND on emulsion "
                    "Dmax; never mapped to the curve shoulder"))

    # ---- base tint ----------------------------------------------------------
    tints = [tin.get("tint_" + c) for c in "rgb"]
    if all(isinstance(v, float) for v in tints):
        gnorm = tints[1]
        t = tuple(round(v / gnorm, 3) for v in tints)
        if all(0.9 <= v <= 1.1 for v in t):
            out.append(("base_tint", "ADOPT-CONTAMINATED", t,
                        "plausible base cast, but the scanning illuminant "
                        "and camera WB are folded in -- tier T2, and it stays "
                        "separate from silver_tone (different physics)"))
            patch["base_tint"] = t
        else:
            out.append(("base_tint", "REJECT", t,
                        "cast above 10% -- more likely a WB failure than a base"))

    # ---- grain: default detection -------------------------------------------
    if grn:
        is_default = all(
            abs(float(grn.get(k, v)) - v) < 1e-9
            for k, v in GENERATOR_GRAIN_DEFAULTS.items()
        )
        if is_default:
            out.append(("grain", "REJECT", dict(grn),
                        "matches the generator's documented defaults exactly "
                        "('Empirical defaults' in its own comment); nothing "
                        "was measured. Grain size is also unmeasurable below "
                        "the scan's own resolution."))
        else:
            out.append(("grain", "MANUAL", dict(grn),
                        "does not match generator defaults -- possibly "
                        "measured, but units differ from GrainSpec; review "
                        "by hand, do not auto-adopt"))

    return out, patch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("measured")
    ap.add_argument("--profile", required=True,
                    help="FilmProfile name the batch belongs to")
    ap.add_argument("--emit-patch", action="store_true")
    args = ap.parse_args()

    d = parse(args.measured)
    verdicts, patch = assess(d)

    name = d.get("FilmProfile", {}).get("name", "?")
    frames = "?"
    for raw in open(args.measured, encoding="utf-8", errors="replace"):
        m = re.search(r"Analyzed Frames:\s*(\d+)", raw)
        if m:
            frames = m.group(1)

    print("Empirical batch %s (%s frames) -> profile %s" %
          (name, frames, args.profile))
    print("-" * 72)
    for field, verdict, value, reason in verdicts:
        print("%-10s %-18s %s" % (field, verdict, value))
        print("           %s" % reason)
    print("-" * 72)

    if args.emit_patch and patch:
        print("\n# --- paste into the %s literal in film_profiles.py ---"
              % args.profile)
        if "dmin" in patch or "gamma" in patch:
            print("# ToneCurve(dmin=%s, gamma=%s, ...)  "
                  "# keep toe/shoulder unless separately measured"
                  % (patch.get("dmin", "<keep>"), patch.get("gamma", "<keep>")))
        if "base_tint" in patch:
            print("base_tint=%s," % (patch["base_tint"],))
        print("# gauge variants (16mm/8mm) take the SAME values; only")
        print("# default_format differs -- magnification is derived, not tuned.")

    adopted = sum(1 for _, v, _, _ in verdicts if v.startswith("ADOPT"))
    rejected = sum(1 for _, v, _, _ in verdicts if v == "REJECT")
    print("\n%d adopted, %d rejected" % (adopted, rejected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
