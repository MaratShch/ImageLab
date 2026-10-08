#!/usr/bin/env python3
"""The mono-primary lobe width, and the record of a fit that was withdrawn.

WHAT THIS FILE IS NOW
---------------------
From 2026-09-26 to 2026-09-27 this module FITTED the width of the three
Gaussian lobes that stand in for the sRGB primaries, and it shipped 15.0 nm.
⚠⚠ THAT FIT IS WITHDRAWN. The width is 34.0 nm and it is now derived from the
primary centres alone. This file keeps the objective's construction, the
numbers it produced, and the reasons it turned out not to be an instrument,
because a withdrawn measurement that leaves no trace is how the same mistake
gets made twice.

WHY THE FIT WAS NOT AN INSTRUMENT -- THREE INDEPENDENT REASONS
---------------------------------------------------------------
**1. IT WAS CIRCULAR, AND THIS FILE SAID SO AND DISMISSED IT.** The objective
compared predicted against measured P30-minus-TRI-X differences over twelve
ColorChecker patches. But P30's spectral curve was itself fitted, the day
before, to thirteen probes off the SAME forum frame. The module's own header
read: *"That is not P30's stored spectral curve being wrong -- the curve is
rebuilt from the same measurement and checked to 0.04 decade"*. Two free
parameters were fitted to one set of numbers, one after the other; minimising
over the second measures how much freedom it has to cancel the first one's
misfit, not the quantity it is named after.

**2. THE OBJECTIVE REWARDED DISCARDING DATA.** The residual fell monotonically
as the lobes narrowed -- 0.2158 at 55 nm to a floor of 0.1542 below 15 nm --
and so did the fraction of the emulsion the basis can see:

    sigma 55 nm   basis sees 70 % of a typical mono stock's own sensitivity
    sigma 35 nm   65 %
    sigma 25 nm   62 %
    sigma 15 nm   41 %
    sigma 11 nm   30 %

An objective that improves as the model throws away more of its input is
degenerate. Below about 20 nm the three lobes stop overlapping and there are
wavelengths inside 400-680 nm where the basis has literally zero weight; the
emulsion's response there is discarded in silence.

**3. THE «INDEPENDENT CHECK» WAS NOT INDEPENDENT.** The adoption argued that
the orthochromatic stocks' red weights falling toward zero confirmed the
change, and those stocks contributed no patch to the objective. But narrowing
ANY lobe reduces cross-talk between the bands: an ortho red weight falling is
an arithmetic consequence of the change, not evidence about its size. The
check could not have come out any other way.

⚠ AND IT MISNAMED ONE OF ITS FOUR «ORTHOCHROMATIC» STOCKS.
KODAK_COMMERCIAL_1956 is non-colour-sensitised, not orthochromatic -- its own
description says the wedge spectrogram dies at 510 nm -- so its near-zero
green weight was correct all along and was being read as a success of the
narrowing.

WHAT IT COST BEFORE IT WAS CAUGHT
----------------------------------
The 540 nm green primary lands in the sensitisation DIP that sits between a
silver halide's intrinsic blue lobe and its green sensitiser. Narrow lobes
sample the bottom of that dip instead of averaging across the green band:

    green weight fell on 29 of the 43 derivable stocks, worst -0.203
    KODAK_COMMERCIAL_1956 reached green 0.0011
    FERRANIA_P30 reached (0.076, 0.166, 0.758) -- less green than a
      red-blind film, on a stock its maker sells as panchromatic

The owner found it by rendering a ColorChecker through P30 on the AVX2 engine
and seeing the blue patch come out white.

WHAT REPLACES IT
----------------
Geometry. For the basis to have no hole, adjacent lobes must cross at least at
half maximum; two Gaussians d apart do that at sigma = d / (2 sqrt(2 ln 2)).
The binding spacing is 460 -> 540 nm, so sigma = 80 / 2.3548 = 33.98 -> 34.0.
The class consequences that width is supposed to protect are gated by
`sensitisation_class.py`. This module now asserts only the two things it is
still in a position to know: that the shipping width is the geometric one,
and that the degeneracy above still reproduces, so that nobody re-runs the
sweep and re-adopts its minimum.

Run:  python mono_primary_width.py [--assert]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import film_profiles as fp  # noqa: E402
import film_sim as fs  # noqa: E402

#: The published X-Rite ColorChecker sRGB renderings for the twelve patches the
#: colour-target module probes. Kept because `ferrania_p30_colour_target.py`
#: imports them and still re-measures those twelve differences every build --
#: the differences are a real measurement of a real difference. What they do
#: not support is a spectral curve or a lobe width.
PATCH_SRGB = {
    "purplish blue": (80, 91, 166),
    "blue":          (56, 61, 150),
    "blue flower":   (133, 128, 177),
    "blue sky":      (98, 122, 157),
    "cyan":          (8, 133, 161),
    "bluish cyan":   (103, 189, 170),
    "green":         (70, 148, 73),
    "foliage":       (87, 108, 67),
    "yellow green":  (157, 188, 64),
    "yellow":        (231, 199, 31),
    "orange":        (214, 126, 44),
    "red":           (175, 54, 60),
}

#: The pair the withdrawn measurement was of.
PAIR = ("FERRANIA_P30", "KODAK_TRI_X_400TX")

#: The withdrawn fit, kept as a record. Nothing reads these to decide anything.
WITHDRAWN = dict(width_nm=15.0, rms_at_fit=0.1548, rms_at_55=0.2158,
                 argmin_nm=11.0, withdrawn="2026-09-27")

#: The width the geometry gives, and the tolerance on reproducing it.
HALF_MAX_K = 2.0 * np.sqrt(2.0 * np.log(2.0))
WIDTH_TOL = 0.05

#: The degeneracy, pinned. `capture(sigma)` is the mean fraction of a
#: monochrome stock's own integrated sensitivity that the basis can see.
#: These are the numbers that say the withdrawn objective was rewarding
#: blindness, so they are asserted rather than described.
#: Re-pinned 2026-10-07e: the corpus mean moved 0.005-0.011 when the
#: monochrome stocks of 2026-10-07 (ROLLEI_PAN_25, ROLLEI_SUPERPAN_200,
#: AGFA_AVIPHOT_PAN_400S with its 300-820 nm curve) joined it. Was
#: 0.3046 / 0.4122 / 0.6218 / 0.6470 / 0.6955; still strictly rising.
CAPTURE = ((11.0, 0.2993), (15.0, 0.4050), (25.0, 0.6110),
           (35.0, 0.6365), (55.0, 0.6860))
CAPTURE_TOL = 0.01


def capture(sigma: float) -> float:
    """Mean fraction of a mono stock's own sensitivity the basis integrates."""
    keep = fs._PRIMARY_WIDTH_NM
    try:
        fs._PRIMARY_WIDTH_NM = float(sigma)
        grid = fs.spectral_grid()
        tot = fs._srgb_primary_spd().sum(axis=0)
        out = []
        for p in fp.FILM_PROFILES:
            if not p.is_monochrome:
                continue
            s = fs.layer_sensitivities(p)
            if s is None or s.shape[0] != 1:
                continue
            den = float(np.trapezoid(s[0], grid))
            if den > 0.0:
                out.append(float(np.trapezoid(s[0] * tot, grid))
                           / den / tot.max())
        return float(np.mean(out)) if out else 0.0
    finally:
        fs._PRIMARY_WIDTH_NM = keep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert", dest="hard", action="store_true")
    ns = ap.parse_args(argv)

    centres = sorted(fs._PRIMARY_CENTRES_NM)
    spacing = max(b - a for a, b in zip(centres, centres[1:]))
    need = spacing / HALF_MAX_K
    got = float(fs._PRIMARY_WIDTH_NM)
    bad = 0

    if not ns.hard:
        print("mono-primary lobe width -- GEOMETRY, not a fit "
              "(the %.1f nm fit was withdrawn %s)"
              % (WITHDRAWN["width_nm"], WITHDRAWN["withdrawn"]))
        print("  centres %s nm, widest spacing %.0f nm"
              % (tuple(int(c) for c in centres), spacing))
        print("  half-max crossing needs sigma %.2f nm; shipping %.1f nm"
              % (need, got))
        for s, _ in CAPTURE:
            print("    sigma %5.1f nm   basis sees %.1f %% of a typical "
                  "emulsion" % (s, 100.0 * capture(s)))

    if abs(got - need) > WIDTH_TOL:
        print("  [FAIL] the shipping width %.2f nm is not the geometric one "
              "%.2f nm (centres %s). This width is DERIVED from the centres: "
              "if the centres moved, move it with them; if it has been "
              "refitted to a residual, read this module's header first."
              % (got, need, tuple(int(c) for c in centres)))
        bad += 1

    prev = prev_s = None
    for s, want in CAPTURE:
        c = capture(s)
        if abs(c - want) > CAPTURE_TOL:
            print("  [FAIL] basis capture at sigma %.0f nm is %.4f, pinned "
                  "%.4f" % (s, c, want))
            bad += 1
        if prev is not None and c <= prev:
            print("  [FAIL] basis capture stopped rising with width between "
                  "%.0f and %.0f nm; the degeneracy this module records is no "
                  "longer reproducible and the withdrawal needs re-arguing"
                  % (prev_s, s))
            bad += 1
        prev, prev_s = c, s

    # ⚠ AND THE THING THE WITHDRAWN FIT WAS FITTED TO MUST STAY WITHDRAWN.
    p30 = next((p for p in fp.FILM_PROFILES if p.name == PAIR[0]), None)
    if p30 is None:
        print("  [FAIL] %s is gone from the database" % PAIR[0])
        bad += 1
    elif p30.spectral.log_s_pan:
        print("  [FAIL] %s carries a spectral curve again. It was withdrawn "
              "on 2026-09-27 because no smooth spectral tilt applied to TRI-X "
              "reproduces the twelve patch differences -- forward-model "
              "residual 0.138 decade against the wavelength-space fit's "
              "0.050. A new curve needs a new source, not a new fit to the "
              "old probes." % PAIR[0])
        bad += 1

    if bad:
        return 1
    print("[OK] mono_primary_width.py -- width %.1f nm is the half-max "
          "crossing of centres %s (%.2f nm); the withdrawn %.1f nm fit's "
          "degeneracy still reproduces, basis capture %.0f %% at sigma %.0f "
          "against %.0f %% at %.0f; %s still carries no spectral curve"
          % (got, tuple(int(c) for c in centres), need, WITHDRAWN["width_nm"],
             100 * capture(CAPTURE[0][0]), CAPTURE[0][0],
             100 * capture(CAPTURE[-1][0]), CAPTURE[-1][0], PAIR[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
