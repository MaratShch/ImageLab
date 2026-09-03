"""One-shot splice of the 1998 Agfa harvest into `film_profiles.py`.

Run once, then delete or keep as the record of what was written. It is NOT an
audit and is NOT registered in `build.py` -- `agfa_1998_curves.py` is the audit
and re-derives every number below from the source on each build.

    python adopt_agfa_1998.py --emit /tmp/agfa98.json [--apply]

Without --apply it prints the patch and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FP = Path(__file__).with_name("film_profiles.py")

SRC98 = (
    "Agfa-Gevaert, «Technical Data PF -- Agfa range of films», 1st edition, "
    "09/1998 -- PDF/PROFILES/AGFA/agfa_films.pdf"
)

# ---------------------------------------------------------------------------
#  Reciprocity, printed p6. times in seconds, stops, and (reversal only) CC.
#  ⚠ Agfa print an interval "1/10 000 - 1" or "1/10 000 - 1/2" for the zero-
#  correction row. Only the LONG end of that interval is stored, because that
#  is the point the correction is stated to hold up to; storing 1e-4 as well
#  would assert a measurement at the short end that the sheet does not make.
# ---------------------------------------------------------------------------
RECIP = {
    "AGFA_OPTIMA_100":   ((1.0, 10.0, 100.0), (0.0, 0.5, 1.5), ()),
    "AGFA_OPTIMA_200":   ((1.0, 10.0, 100.0), (0.0, 1.0, 2.0), ()),
    "AGFA_OPTIMA_400":   ((1.0, 10.0, 100.0), (0.0, 1.0, 2.0), ()),
    "AGFA_PORTRAIT_160": ((1.0, 10.0, 100.0), (0.0, 1.0, 2.0), ()),
    "AGFA_ULTRA_50":     ((1.0, 10.0, 100.0), (0.0, 1.0, 2.0), ()),
    "AGFA_RSX_II_50":    ((1.0, 10.0, 100.0), (0.0, 0.5, 1.0),
                          ("", "CC05B", "CC10B")),
    "AGFA_RSX_II_100":   ((1.0, 10.0, 100.0), (0.0, 0.5, 1.0),
                          ("", "CC05B", "CC10B")),
    "AGFA_RSX_II_200":   ((1.0, 10.0, 100.0), (0.0, 1.0, 2.0),
                          ("", "CC075Y", "CC15Y+CC05C")),
    "AGFA_APX_25":       ((0.5, 1.0, 10.0, 100.0), (0.0, 0.5, 1.0, 2.0), ()),
    "AGFA_APX_100":      ((0.5, 1.0, 10.0, 100.0), (0.0, 1.0, 2.0, 3.0), ()),
    "AGFA_APX_400":      ((0.5, 1.0, 10.0, 100.0), (0.0, 1.0, 2.0, 3.0), ()),
    "AGFA_SCALA_200X":   ((0.5, 1.0, 10.0, 100.0), (0.0, 0.5, 1.0, 2.0), ()),
}

#: ⚠ APX 25 IS THE ONE ROW WHERE TWO AGFA DOCUMENTS DISAGREE. Its own 1995
#: datasheet (agfapanapx25.pdf) prints none / +1 / +1.5 / +2 for the same four
#: times; this 1998 range brochure prints 0 / +0.5 / +1 / +2. The 2004 B&W
#: handbook confirms the 1998 figures for APX 100 and 400 but does not cover
#: APX 25, so nothing breaks the tie. The 1998 values are stored because they
#: are the ones the rest of the table is internally consistent with, and the
#: conflict is recorded in the ParamSource rather than averaged away.
RECIP_NOTE = {
    "AGFA_APX_25": (
        "⚠ TWO AGFA DOCUMENTS DISAGREE ON THIS ROW. agfapanapx25.pdf (08/1995) "
        "prints none / +1 / +1.5 / +2 for 1/10000-0.5 s, 1 s, 10 s, 100 s; "
        "agfa_films.pdf (09/1998) prints 0 / +0.5 / +1 / +2. The 1998 values "
        "are stored. The 2004 B&W handbook confirms 1998 for APX 100 and APX "
        "400 but drops APX 25, so the tie is unbroken. Difference is up to one "
        "full stop at 10 s"),
}

#: Total layer thickness without base, printed beside each column on pp7-10.
COATED_UM = {
    "AGFA_OPTIMA_100": 16.0, "AGFA_OPTIMA_200": 18.0, "AGFA_OPTIMA_400": 19.0,
    "AGFA_PORTRAIT_160": 18.0, "AGFA_ULTRA_50": 27.0,
    "AGFA_RSX_II_50": 25.0, "AGFA_RSX_II_100": 25.0, "AGFA_RSX_II_200": 27.0,
    "AGFA_SCALA_200X": 7.0,
    "AGFA_APX_25": 3.0, "AGFA_APX_100": 7.0, "AGFA_APX_400": 10.0,
}

#: printed name in the emitted JSON -> profile
BY_PROFILE = {
    "AGFA_OPTIMA_100": "AGFACOLOR OPTIMA II 100",
    "AGFA_OPTIMA_200": "AGFACOLOR OPTIMA II 200",
    "AGFA_OPTIMA_400": "AGFACOLOR OPTIMA II 400",
    "AGFA_PORTRAIT_160": "AGFACOLOR PORTRAIT XPS 160",
    "AGFA_ULTRA_50": "AGFACOLOR ULTRA 50",
    "AGFA_RSX_II_50": "AGFACHROME RSX II 50",
    "AGFA_RSX_II_100": "AGFACHROME RSX II 100",
    "AGFA_RSX_II_200": "AGFACHROME RSX II 200",
    "AGFA_SCALA_200X": "AGFA SCALA 200x",
    "AGFA_APX_25": "AGFAPAN APX 25",
    "AGFA_APX_100": "AGFAPAN APX 100",
    "AGFA_APX_400": "AGFAPAN APX 400",
}

#: Developer reference times at 20 C, small tank, printed on p11. Used only to
#: pick the samples stored in ProcessingFamily -- the gammas themselves are
#: read off the curve.
REF_TIME = {
    "AGFA_APX_25":  {"REFINAL": 6.0, "RODINAL 1+25": 6.0, "RODINAL 1+50": 10.0,
                     "RODINAL SPECIAL": 4.0, "STUDIONAL LIQUID": 4.0},
    "AGFA_APX_100": {"REFINAL": 6.0, "RODINAL 1+25": 8.0, "RODINAL 1+50": 17.0,
                     "RODINAL SPECIAL": 4.0, "STUDIONAL LIQUID": 4.0},
    "AGFA_APX_400": {"REFINAL": 6.0, "RODINAL 1+25": 7.0, "RODINAL 1+50": 11.0,
                     "RODINAL SPECIAL": 4.5, "STUDIONAL LIQUID": 4.5},
}

DILUTION = {"REFINAL": "stock", "RODINAL 1+25": "1+25", "RODINAL 1+50": "1+50",
            "RODINAL SPECIAL": "1+15", "STUDIONAL LIQUID": "1+15"}


def spectral_block(lay, indent=12):
    """A SpectralSensitivity literal from the emitted per-layer samples."""
    pad = " " * indent
    def row(key):
        v = lay[key]["log_s"]
        return ", ".join(f"{x:.2f}" for x in v)
    out = [f"{pad}lambda_start_nm=380.0, lambda_step_nm=10.0,"]
    for k, field in (("r", "log_s_r"), ("g", "log_s_g"), ("b", "log_s_b")):
        out.append(f"{pad}{field}=({row(k)}),")
    return "\n".join(out)


def recip_block(profile, indent=8):
    pad = " " * indent
    t, s, cc = RECIP[profile]
    note = RECIP_NOTE.get(profile, "")
    src = (f"{SRC98} p6, 'Reciprocity effect'. Printed as an exposure-reading "
           f"interval against an f-stop correction; only the long end of the "
           f"zero-correction interval is stored")
    if note:
        src += ". " + note
    lines = [f"{pad}reciprocity_table=ReciprocityTable(",
             f"{pad}    times_s={t!r},",
             f"{pad}    stops_correction={s!r},"]
    if cc:
        lines.append(f"{pad}    cc_filters={cc!r},")
    lines.append(f"{pad}    source=({_wrap(src, indent + 12)}),")
    lines.append(f"{pad}),")
    return "\n".join(lines)


def _wrap(text, indent, width=74):
    pad = " " * indent
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width - indent:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return ("\n" + pad).join(f'"{s} "' if i < len(out) - 1 else f'"{s}"'
                             for i, s in enumerate(out))



def procfam_block(profile, rec, indent=8):
    """A ProcessingFamily literal from the traced gamma-time curves."""
    pad = " " * indent
    refs = REF_TIME[profile]
    lines = [f"{pad}# ⚠ TRACED FROM THE GAMMA-TIME PANEL, AND VALIDATED BY A FACT",
             f"{pad}# THE PANEL DOES NOT STATE. Read at each developer's own",
             f"{pad}# reference time from the p11 processing table, all four",
             f"{pad}# curves return gamma 0.65 +/- 0.01. `agfa_bw_manual.pdf`",
             f"{pad}# then says it in words: every speed table in its developer",
             f"{pad}# section is headed 'Film speed (exposure index) (gamma =",
             f"{pad}# 0.65)' and its developing-time tables are indexed by",
             f"{pad}# gamma = 0.55 / 0.65 / 0.75. Agfa specify the whole AGFAPAN",
             f"{pad}# line to gamma 0.65 and the digitisation reproduces it to",
             f"{pad}# one part in sixty-five. That is what licenses these points.",
             f"{pad}# ⚠ THE STORED ToneCurve.gamma IS NOT THIS NUMBER. It is a",
             f"{pad}# softplus MODEL COEFFICIENT; ToneCurve.mid_slope is the",
             f"{pad}# comparable quantity. Do not 'correct' one to the other.",
             f"{pad}# ⚠ RODINAL SPECIAL AND STUDIONAL LIQUID SHARE ONE DRAWN",
             f"{pad}# CURVE -- the panel plots four curves for five printed",
             f"{pad}# names, and the p11 table gives both developers the same",
             f"{pad}# time at every temperature. Both are listed against the",
             f"{pad}# same gammas because that is what the sheet draws.",
             f"{pad}processing_family=ProcessingFamily(",
             f"{pad}    points=("]
    n = 0
    for fam in rec["gamma_time"]:
        pts = {round(t, 2): g for t, g in fam["samples"]}
        # Every SECOND integer minute, plus each developer's own reference
        # time. Denser sampling buys nothing: the curves are smooth beziers
        # and the carrier is interpolated, so the points are there to pin the
        # shape and the reference time, not to reproduce the artwork.
        want = sorted({float(v) for v in pts
                       if abs(v - round(v)) < 0.26 and round(v) % 2 == 0} |
                      {refs[d] for d in fam["developers"]
                       if fam["t_min"] - 0.01 <= refs[d] <= fam["t_max"] + 0.01})
        for dev in fam["developers"]:
            for t in want:
                near = min(pts, key=lambda k: abs(k - t))
                if abs(near - t) > 0.30:
                    continue
                g = pts[near]
                tag = "   # reference time" if abs(t - refs[dev]) < 1e-6 else ""
                lines.append(
                    f"{pad}        DevelopmentPoint(developer={dev!r},"
                    f" dilution={DILUTION[dev]!r},")
                lines.append(
                    f"{pad}                         minutes={t:g}, celsius=20.0,"
                    f" gamma={g:.3f}),{tag}")
                n += 1
    src = (SRC98 + " p10, the Gamma-time curves panel, digitised by "
           "agfa_1998_curves.py; five printed developer names on four drawn "
           "curves, matched by label bounding-box distance. Developing times "
           "and the reference time per developer are from the printed tables "
           "on p11 of the same document. Axis fit residual 0.001 min / 0.0000 "
           "gamma. ⚠ REFINAL and RODINAL 1+25 are 0.0 and 0.2 pt from their "
           "assigned curves on APX 25, effectively a tie; swapping them moves "
           "gamma(6 min) from 0.652 to 0.646")
    lines.append(f"{pad}    ),")
    lines.append(f"{pad}    source=({_wrap(src, indent + 12)}),")
    lines.append(f"{pad}),")
    return "\n".join(lines), n


def push_block(rec, indent=8):
    """Scala's PushSpec, from the five measured density curves plus the table."""
    pad = " " * indent
    fam = rec["push_family"]
    dmax = {k: v["dmax"] for k, v in fam.items()}
    steps = ["Pull 1", "Standard", "Push 1", "Push 2", "Push 3"]
    per = [(dmax[steps[i + 1]] - dmax[steps[i]]) for i in range(len(steps) - 1)]
    mean_per_stop = sum(per) / len(per)
    src = (SRC98 + " p9. The push/pull TABLE gives the speed of each step "
           "(Pull 1 ISO 100/21, Standard 200/24, Push 1 400/27, Push 2 800/30, "
           "Push 3 1600/33) and states the DIRECTIONS only -- contrast "
           "'increasingly steeper' on push and 'flatter' on pull, maximum "
           "density 'decreasing' on push and 'increasing' on pull, granularity "
           "'increasingly coarse-grained' / 'finer'. The MAGNITUDES here are "
           "digitised from the Density curves panel beside it by "
           "agfa_1998_curves.py: D-max " +
           ", ".join(f"{s} {dmax[s]:.3f}" for s in steps) +
           f", i.e. {mean_per_stop:+.3f} D per stop, monotone over all four "
           "intervals. ⚠ agfa_bw_manual.pdf p11 redraws the same family and "
           "returns 3.012 / 2.800 / 2.543 / 2.289 / 2.034 -- the same ordering "
           "and the same spacing, but Standard differs by 0.18 D, so treat the "
           "absolute level as edition-dependent and the per-stop slope as the "
           "measurement")
    return "\n".join([
        f"{pad}# ⚠ MAGNITUDES ARE DIGITISED, DIRECTIONS ARE PRINTED. The sheet's",
        f"{pad}# push/pull table states only that D-max falls on push and rises",
        f"{pad}# on pull; the numbers come from the five drawn curves beside it.",
        f"{pad}# base_fog_penalty_per_stop is left at zero and fog_penalty_stated",
        f"{pad}# FALSE, because all five curves share one D-min of 0.024 -- the",
        f"{pad}# artwork shows no fog penalty at all, which is a statement about",
        f"{pad}# the drawing and not a measurement that pushing is fog-free.",
        f"{pad}push=PushSpec(",
        f"{pad}    max_push_stops=3.0,",
        f"{pad}    max_pull_stops=1.0,",
        f"{pad}    fog_penalty_stated=False,",
        f"{pad}    source=({_wrap(src, indent + 12)}),",
        f"{pad}),",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    ap.add_argument("--apply", action="store_true")
    ns = ap.parse_args()
    data = json.loads(Path(ns.emit).read_text())["films"]
    src = FP.read_text(encoding="utf-8")
    orig = src

    # ---- 1. adjacency on the eight existing Agfa profiles -----------------
    for profile in ("AGFA_OPTIMA_100", "AGFA_OPTIMA_200", "AGFA_OPTIMA_400",
                    "AGFA_PORTRAIT_160", "AGFA_SCALA_200X",
                    "AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400"):
        adj = data[BY_PROFILE[profile]]["sharpness"]["adjacency"]
        src = _set_adjacency(src, profile, adj)

    # ---- 2. coated_um + reciprocity_table on the same eight ---------------
    for profile in ("AGFA_OPTIMA_100", "AGFA_OPTIMA_200", "AGFA_OPTIMA_400",
                    "AGFA_PORTRAIT_160", "AGFA_SCALA_200X",
                    "AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400"):
        src = _insert_kwargs(src, profile, _extra_for(profile, data))

    # ---- 3. gamma-time families on the three APX --------------------------
    for profile in ("AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400"):
        blk, n = procfam_block(profile, data[BY_PROFILE[profile]])
        print(f"  {profile}: {n} development points")
        src = _insert_kwargs(src, profile, blk)

    # ---- 4. Scala push/pull ------------------------------------------------
    src = _insert_kwargs(src, "AGFA_SCALA_200X",
                         push_block(data["AGFA SCALA 200x"]))

    if not ns.apply:
        print("--- dry run, nothing written ---")
        print(f"film_profiles.py {len(orig)} -> {len(src)} chars")
        return 0
    FP.write_text(src, encoding="utf-8")
    print(f"[ok] film_profiles.py {len(orig)} -> {len(src)} chars")
    return 0


def _set_adjacency(src, profile, adj):
    """Rewrite the `adjacency=` argument of one profile's MTFSpec."""
    i = src.index(f'name="{profile}",')
    j = src.index("mtf=MTFSpec(", i)
    k = src.index(")", j)
    seg = src[j:k]
    if "adjacency=" not in seg:
        raise SystemExit(f"{profile}: MTFSpec has no adjacency= to replace")
    new = re.sub(r"adjacency=[-\d.]+", f"adjacency={adj:.4f}", seg, count=1)
    return src[:j] + new + src[k:]


def _extra_for(profile, data):
    esrc = (SRC98 + " pp7-10, the per-film characteristic values printed "
                    "beside each plotted column")
    emulsion = (
        "        # ⚠ COATED THICKNESS IS THE WHOLE EmulsionSpec THIS SOURCE\n"
        "        # SUPPORTS. Agfa print 'Total layer thickness (without base)'\n"
        "        # beside every column and print nothing about crystal size,\n"
        "        # habit, aspect ratio or iodide, so those stay at zero rather\n"
        "        # than being inferred from the thickness.\n"
        "        emulsion=EmulsionSpec(\n"
        f"            coated_um={COATED_UM[profile]},\n"
        f"            source=({_wrap(esrc, 20)}),\n"
        "        ),"
    )
    return "\n".join([emulsion, recip_block(profile)])


def _insert_kwargs(src, profile, text):
    """Insert extra keyword arguments before a profile's `features=` line."""
    i = src.index(f'name="{profile}",')
    m = re.compile(r"^        features=", re.M).search(src, i)
    if m is None:
        raise SystemExit(f"{profile}: no `features=` line to insert before")
    return src[:m.start()] + text + "\n" + src[m.start():]


if __name__ == "__main__":
    sys.exit(main())
