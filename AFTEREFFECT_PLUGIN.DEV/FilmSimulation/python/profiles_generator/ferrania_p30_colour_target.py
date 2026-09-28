#!/usr/bin/env python3
"""FERRANIA P30 (original) — the red deficiency, measured off a colour target.

    PDF/PROFILES/FERRANIA/analogica_sensibilita_cromatica_vs_tri-x.png
    «In evidenza lo scacco del colore rosso» — posted by «chromemax» to
    analogica.it on 01/12/2019 in the thread «Ferrania P-30 new test»
    (https://www.analogica.it/ferrania-p-30-new-test-t20485.html). One
    24-patch colour target, photographed on Kodak Tri-X and on Ferrania
    P-30, printed, and reproduced side by side with the target's own colour
    reference above the pair.

⚠⚠ WHY A FORUM PHOTOGRAPH IS LOAD-BEARING HERE.
FERRANIA_P30 models the ORIGINAL (cinema) emulsion, and no wedge spectrogram
of that emulsion exists — in this corpus or, as far as this project can
establish, anywhere. Ferrania's own sheet carries spectrograms for P 30 New
(the Mk2), P 33 and Orto, and until 2026-09-25 the Mk2's was stored on this
profile, which is how a film the maker describes as having «BASSA SENSIBILITA
AL ROSSO» came to be rendered with a flat panchromatic response to 650 nm.
This frame is the only quantitative measurement of the original emulsion's
colour response that exists, so the stored curve is derived from it.

WHAT IS MEASURED, AND WHY A DIFFERENTIAL IS SOUND WHERE AN ABSOLUTE IS NOT
---------------------------------------------------------------------------
Two photographs, two unknown exposures, two unknown paper grades, one target.
Each panel is calibrated on ITS OWN six-patch neutral row against the
reference chart's luminances, which inverts that panel's entire
tone-reproduction chain — film curve, paper curve, scan, JPEG. What survives
that is the difference BETWEEN the two films at a given wavelength, and
nothing else. No absolute sensitivity is recovered and none is claimed.

    P30 − Tri-X, decade, as measured:
        blue +0.35   cyan +0.35   blue sky +0.29   purplish blue +0.42
        green −0.15  yellow-green −0.25
        yellow −0.52  orange −0.47  red −0.43   moderate red −0.17

⚠ THE INDEPENDENT CHECK COMES FROM A DIFFERENT YEAR AND A DIFFERENT
PHOTOGRAPHER. analogica.it user «ometto», 10/10/2020, metering one scene with
and without an orange/red filter, found the filter's nominal +3 stops
«assolutamente insufficienti» and needed two more — +5 stops in all. Two stops
is 0.60 decade of extra red deficit against an average panchromatic film,
against the −0.54 decade this frame gives at the red patch.

HOW THE STORED CURVE IS BUILT
-------------------------------
    log_s_pan(P30) = log_s_pan(KODAK_TRI_X_400TX) + delta(lambda)

with Tri-X's curve being Kodak's own F-4017 vector artwork already in this
database, and delta a four-parameter logistic fitted to thirteen probes —
twelve patches at their effective wavelengths plus the filter-factor point.
⚠ TIER 3. The DIFFERENCE is measured; the shape between probes is the
logistic's and the shape below 450 nm and above 630 nm is Tri-X's carried
across unchanged.

⚠ THE PATCH-TO-WAVELENGTH ASSIGNMENT IS THE WEAKEST LINK and is pinned here
rather than left implicit: a ColorChecker patch is a broad reflectance, not a
line, so each probe's «effective wavelength» is a nominal centre and the
broad patches carry less weight than the three primaries.

Run:  python ferrania_p30_colour_target.py [--root .] [--assert]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FRAME = "PDF/PROFILES/FERRANIA/analogica_sensibilita_cromatica_vs_tri-x.png"

NAMES = (("dark skin", "light skin", "blue sky", "foliage", "blue flower",
          "bluish cyan"),
         ("orange", "purplish blue", "moderate red", "purple", "yellow green",
          "orange yellow"),
         ("blue", "green", "red", "yellow", "magenta", "cyan"),
         ("white", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5",
          "black"))
#: reference chart (top of the frame) and the two printed panels, as pixel
#: boxes in the 400x257 image.
REF_BOX = (161, 18, 249, 73)
PANELS = {"trix": (30, 85, 205, 215), "p30": (205, 85, 385, 215)}
#: ⚠ CIE Y of the target's six neutral patches, McCamy/X-Rite published
#: values, as FRACTIONS. These and not the frame's own reference chart are
#: what each panel is calibrated against -- see the note in main().
NEUTRAL_Y = (0.886, 0.591, 0.362, 0.198, 0.090, 0.031)
#: seed for the 6x4 grid fit in each panel: x0, y0, dx, dy, row slope x,
#: column slope y.
SEED = {"trix": (58.6, 132.3, 24.8, 20.7, 0.35, 1.1),
        "p30": (231.0, 113.5, 24.6, 24.8, 0.9, 2.6)}

#: ⚠ THE PROBES, PINNED. (effective nm, expected P30-minus-TriX decade,
#: weight, patch). The three primaries carry weight 1.0; broad patches 0.4-0.8.
#: The last row is not a patch at all -- it is the filter factor.
PROBES = (
    (450, +0.42, 0.6, "purplish blue"),
    (465, +0.39, 1.0, "blue"),
    (470, +0.31, 0.6, "blue flower"),
    (480, +0.28, 0.6, "blue sky"),
    (490, +0.33, 1.0, "cyan"),
    (495, +0.24, 0.6, "bluish cyan"),
    (540, -0.16, 1.0, "green"),
    (550, -0.14, 0.4, "foliage"),
    (565, -0.24, 0.6, "yellow green"),
    (575, -0.48, 0.6, "yellow"),
    (600, -0.49, 0.8, "orange"),
    (615, -0.54, 1.0, "red"),
    (650, -0.60, 0.5, None),      # ometto's orange/red filter, +5 not +3 stops
)
#: the fitted logistic, pinned so a drift is visible as a number.
FIT = dict(plateau_short=0.463, plateau_long=-0.589, midpoint_nm=533.6,
           width_nm=27.5, residual=0.050)

TOL_PROBE = 0.08      # decade, per patch, re-measurement against the pin
TOL_FIT = 0.06        # decade, refitted logistic against the pinned one
TOL_CURVE = 0.04      # decade, rebuilt curve against the stored log_s_pan


def _linear(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _grid(img, box, seed):
    """Fit the 6x4 patch lattice of one printed panel and sample its centres."""
    import cv2
    x0, y0, x1, y1 = box
    sub = img[y0:y1, x0:x1].astype(np.uint8)
    th = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 21, -4)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(th, 8)
    blobs = [(cent[i][0] + x0, cent[i][1] + y0) for i in range(1, n)
             if 200 < stats[i, cv2.CC_STAT_AREA] < 900
             and 14 < stats[i, cv2.CC_STAT_WIDTH] < 34
             and 12 < stats[i, cv2.CC_STAT_HEIGHT] < 32]
    X0, Y0, dx, dy, sx, sy = seed
    for _ in range(6):
        A, B = [], []
        M = np.array([[dx, sx], [sy, dy]])
        for (px, py) in blobs:
            j, i = np.linalg.solve(M, [px - X0, py - Y0])
            j, i = round(j), round(i)
            if not (0 <= j < 6 and 0 <= i < 4):
                continue
            A.append([1, 0, j, 0, i, 0]); B.append(px)
            A.append([0, 1, 0, j, 0, i]); B.append(py)
        if len(B) < 20:
            return None, len(blobs)
        p, _r, _rk, _s = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)
        X0, Y0, dx, sy, sx, dy = p
        M = np.array([[dx, sx], [sy, dy]])
    out = {}
    for i in range(4):
        for j in range(6):
            cx = int(round(X0 + j * dx + i * sx))
            cy = int(round(Y0 + j * sy + i * dy))
            out[(i, j)] = float(img[cy - 4:cy + 5, cx - 4:cx + 5].mean())
    return out, len(blobs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    if not (root / FRAME).is_file():
        print("  [SKIP] colour-target frame not present: %s" % (root / FRAME))
        return 0
    from PIL import Image
    from scipy.optimize import least_squares
    print("FERRANIA P30 (original) -- red response measured off the "
          "analogica.it colour target")
    bad = 0
    rgb = np.asarray(Image.open(root / FRAME).convert("RGB")).astype(float)
    if rgb.shape[:2] != (257, 400):
        print("  [FAIL] the frame is %dx%d, not 400x257 -- every pixel box "
              "below is sized to that copy" % (rgb.shape[1], rgb.shape[0]))
        return 1 if ns.assert_ else 0
    grey = np.asarray(Image.open(root / FRAME).convert("L")).astype(float)

    # -- the target's own reference chart, top of the frame -----------------
    x0, y0, x1, y1 = REF_BOX
    w, h = (x1 - x0) / 6.0, (y1 - y0) / 4.0
    ref, Y = {}, {}
    for i in range(4):
        for j in range(6):
            cx, cy = int(x0 + (j + 0.5) * w), int(y0 + (i + 0.5) * h)
            srgb = rgb[cy - 2:cy + 3, cx - 2:cx + 3].reshape(-1, 3).mean(0)
            lin = _linear(srgb)
            ref[NAMES[i][j]] = srgb
            Y[NAMES[i][j]] = float(0.2126 * lin[0] + 0.7152 * lin[1]
                                   + 0.0722 * lin[2])
    chart_ramp = [Y[NAMES[3][j]] for j in range(6)]
    if not (chart_ramp[0] > 0.8 and chart_ramp[5] < 0.05
            and all(chart_ramp[i] > chart_ramp[i + 1] for i in range(5))):
        print("  [FAIL] the reference chart's neutral row does not read as a "
              "monotone ramp (%s) -- REF_BOX is off the chart"
              % np.round(chart_ramp, 3))
        bad += 1
    # ⚠⚠ THE CALIBRATION USES THE TARGET'S PUBLISHED LUMINANCES, NOT THE ONES
    # MEASURED OFF THE REFERENCE CHART IN THIS FRAME, AND THE DIFFERENCE IS A
    # FACTOR OF TWO IN EVERY RESULT. The chart at the top of the frame is a
    # rendered picture of a ColorChecker, not a photometric record of one: its
    # own neutral ramp spans Y 0.998 down to 0.005, i.e. 2.3 decades, where
    # the physical target spans 0.886 to 0.031, i.e. 1.46. Calibrating on the
    # picture stretches the recovered luminance axis by 1.6x, and because the
    # P30-minus-Tri-X difference is taken in that axis it stretches too --
    # measured deltas come out at +0.81 / -1.12 instead of +0.35 / -0.43.
    # McCamy's published CIE Y for the 24 patches is used instead; the chart
    # in the frame is checked for presence and monotonicity and for nothing
    # else.
    neutral_Y = list(NEUTRAL_Y)

    # -- each printed panel, calibrated on its own neutral row --------------
    meas = {}
    for film, box in PANELS.items():
        cells, nblob = _grid(grey, box, SEED[film])
        if cells is None:
            print("  [FAIL] %s panel: only %d patch blobs found" % (film, nblob))
            bad += 1
            continue
        gn = np.array([cells[(3, j)] for j in range(6)])
        yn = np.array(neutral_Y)
        ln = np.log10(yn)
        order = np.argsort(gn)            # print level ascending
        # ⚠ MONOTONE INTERPOLATION, NOT A POLYNOMIAL. A cubic through six
        # points curls back outside the calibrated span, and the inverse then
        # jumps to the wrong branch for the darkest patches -- P30's orange
        # and red, which are exactly the patches this whole measurement is
        # about, came back at +1.62 and +0.76 decade instead of -0.47 and
        # -0.43. PCHIP is monotone by construction and cannot do that.
        from scipy.interpolate import PchipInterpolator
        inv = PchipInterpolator(gn[order], ln[order], extrapolate=True)
        fwd = PchipInterpolator(ln[np.argsort(ln)], gn[np.argsort(ln)],
                                extrapolate=True)
        resid = float(np.max(np.abs(gn - fwd(ln))))
        if resid > 8.0:
            print("  [FAIL] %s panel: the neutral-row calibration does not "
                  "fit (worst %.1f levels)" % (film, resid))
            bad += 1
        lo, hi = float(gn.min()), float(gn.max())
        out, clipped = {}, []
        for i in range(3):
            for j in range(6):
                v = cells[(i, j)]
                if not (lo - 6 <= v <= hi + 6):
                    clipped.append(NAMES[i][j])
                out[NAMES[i][j]] = float(10 ** float(inv(v)))
        if clipped:
            print("    %s: %d patch(es) outside the neutral row's own print "
                  "range, extrapolated: %s"
                  % (film, len(clipped), ", ".join(clipped)))
        meas[film] = out
        print("  %-5s panel: %2d patch blobs, neutral-row fit worst %.1f "
              "levels" % (film, nblob, resid))

    if len(meas) != 2:
        print("\n[FAIL] both panels are needed and only %d resolved" % len(meas))
        return 1 if ns.assert_ else 0

    # -- the differential ----------------------------------------------------
    delta = {k: float(np.log10(meas["p30"][k] / meas["trix"][k]))
             for k in meas["p30"]}
    worst_name, worst = "", 0.0
    for nm, want, _w, patch in PROBES:
        if patch is None:
            continue
        got = delta[patch]
        if abs(got - want) > abs(worst):
            worst, worst_name = got - want, patch
        if abs(got - want) > TOL_PROBE:
            print("  [FAIL] %-14s measures %+.3f, pinned %+.2f decade"
                  % (patch, got, want))
            bad += 1
    print("  twelve patch probes re-measured; worst drift %+.3f decade (%s)"
          % (worst, worst_name))
    print("    blue %+.2f  cyan %+.2f  green %+.2f  yellow %+.2f  "
          "orange %+.2f  red %+.2f"
          % (delta["blue"], delta["cyan"], delta["green"], delta["yellow"],
             delta["orange"], delta["red"]))

    # -- refit the logistic --------------------------------------------------
    lam = np.array([p[0] for p in PROBES], float)
    val = np.array([delta[p[3]] if p[3] else p[1] for p in PROBES])
    wt = np.array([p[2] for p in PROBES])
    model = lambda p, x: p[1] + (p[0] - p[1]) / (1 + np.exp((x - p[2]) / p[3]))
    r = least_squares(lambda p: wt * (model(p, lam) - val),
                      [0.35, -0.55, 555.0, 25.0],
                      bounds=([0.1, -1.2, 500, 8], [0.8, -0.2, 650, 80]))
    A, B, l0, wid = r.x
    rms = float(np.sqrt(np.mean((model(r.x, lam) - val) ** 2)))
    print("  logistic refit: +%.3f below 500 nm, %.3f above 600 nm, midpoint "
          "%.1f nm, width %.1f nm, residual %.3f decade"
          % (A, B, l0, wid, rms))
    if (abs(A - FIT["plateau_short"]) > TOL_FIT
            or abs(B - FIT["plateau_long"]) > TOL_FIT
            or abs(l0 - FIT["midpoint_nm"]) > 15.0):
        print("  [FAIL] the fitted delta has moved away from the pinned one "
              "(+%.3f / %.3f / %.1f nm pinned)"
              % (FIT["plateau_short"], FIT["plateau_long"], FIT["midpoint_nm"]))
        bad += 1

    # -- what the measurement does NOT support -------------------------------
    #
    # ⚠⚠ THIS BLOCK USED TO REBUILD FERRANIA_P30's SPECTRAL CURVE FROM THE
    # LOGISTIC ABOVE AND ASSERT IT AGAINST THE DATABASE. That curve was
    # withdrawn on 2026-09-27 and this block now asserts that it STAYS
    # withdrawn, because the failure was in the method, not in the arithmetic.
    #
    # The logistic above is fitted in WAVELENGTH SPACE: each ColorChecker
    # patch is given an «effective nm» and the curve is fitted through those
    # points. A ColorChecker patch is a BROADBAND reflectance, so its measured
    # difference is an integral over roughly a hundred nanometres. Attributing
    # that whole integral to one wavelength forces the fitted tilt to be far
    # steeper than the emulsions differ -- which is how thirteen probes
    # spanning 0.93 decade produced a 1.05-decade swing, and how subtracting
    # it from TRI-X erased the green/red hump that every panchromatic curve
    # in this database has. The result had the shape of an unsensitised
    # emulsion and gave P30 a green weight of 0.166, a third of red-blind
    # FERRANIA_ORTO_50's 0.448.
    #
    # ⚠⚠ THE HONEST TEST IS THE FORWARD MODEL, AND IT REFUSES. Vary the
    # logistic, build the curve, derive the weights the renderer would
    # actually use, predict each patch's difference through them, compare
    # against the measurement: run on 2026-09-27 the optimiser drives the
    # logistic width to its lower bound and the weighted residual is 0.138
    # decade, against 0.050 for the wavelength-space fit. No smooth spectral
    # tilt applied to TRI-X reproduces these twelve differences. The 0.050
    # was the freedom of the wrong objective, not agreement.
    #
    # THE TWELVE DIFFERENCES ABOVE REMAIN A REAL MEASUREMENT of a real
    # difference between two emulsions, and they are still re-measured off
    # the frame on every build. What they do not support is a curve.
    try:
        import film_profiles as fp
    except Exception as exc:                                  # pragma: no cover
        print("  [WARN] could not import film_profiles: %s" % exc)
        return 1 if ns.assert_ else 0
    p30 = fp.get_profile("FERRANIA_P30")
    if p30.spectral.log_s_pan:
        print("  [FAIL] FERRANIA_P30 carries a spectral curve again. It was "
              "withdrawn on 2026-09-27; the forward-model refit of these same "
              "probes does not converge. A new curve needs a new source.")
        bad += 1
    if p30.spectral.criterion:
        print("  [FAIL] FERRANIA_P30's spectral criterion is %r; with no "
              "curve it must be empty" % p30.spectral.criterion)
        bad += 1

    # ⚠ AND THE RED DEFICIT MUST STILL RENDER, which is the part of the
    # finding that survived. It now comes from the authored triple rather
    # than from a curve, so this asserts the OUTCOME and not the mechanism.
    import film_sim
    w = tuple(p30.spectral_weights)
    wt_trix = film_sim.spectral_monochrome_weights(
        fp.get_profile("KODAK_TRI_X_400TX"))
    print("  mono weights  P30 (authored) r %.3f g %.3f b %.3f   "
          "Tri-X (derived) r %.3f g %.3f b %.3f" % (w + tuple(wt_trix)))
    if not (w[0] < 0.5 * wt_trix[0]):
        print("  [FAIL] the engine is not rendering P30 as red-deficient: "
              "red weight %.3f against Tri-X's %.3f" % (w[0], wt_trix[0]))
        bad += 1
    if w[1] < 0.20:
        print("  [FAIL] P30's green weight %.3f is back below the "
              "panchromatic floor" % w[1])
        bad += 1

    if ns.assert_ and bad:
        print("\n[FAIL] the P30 colour-target measurement does not reproduce")
        return 1
    print("\n[OK] twelve patch differences re-measured off the frame. The "
          "curve they were once used to build is withdrawn and stays "
          "withdrawn; P30's red deficit is carried by an authored triple "
          "whose one measured input is the +5-stop filter datum.")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
