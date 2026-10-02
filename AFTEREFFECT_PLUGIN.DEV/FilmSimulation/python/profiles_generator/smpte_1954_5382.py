#!/usr/bin/env python3
"""SMPTE 63(5), November 1954: the dye deposits of EASTMAN COLOR PRINT FILM 5382.

WHAT THIS SOURCE IS
-------------------
`RETRO/sim_smpte-motion-imaging-journal_1954-11_63_5.pdf` -- **Journal of the
SMPTE, Volume 63 Number 5, November 1954**, an Internet Archive microfilm scan.
The paper is **Lovick and White, "Factors in Applying Color Soundtrack
Developers"**, p.189, and its **Figure 2** is captioned:

    "Fig. 2. Spectral density of dye deposits of Eastman Color Print Film,
     Type 5382."

⚠ THE PAPER IS ABOUT SOUNDTRACK CHEMISTRY, NOT ABOUT THE FILM. Its subject is
how to redevelop silver in the soundtrack area of a colour print, and Fig. 2 is
there to make one argument: that "dyes alone in color films at any practical
concentration will not produce sufficient density in the infrared region",
which is why an S-1 phototube needs a silver track. The figure is a means to
that end, and that is exactly why it is useful here -- it was drawn to be read
in the infrared, so it runs to **1000 nm** where a product data sheet would
stop at 700.

WHAT IS TRACED, AND WHY IT MATTERS
-----------------------------------
Three dye curves, spectral density D(lambda) 0.0-1.2 against wavelength
400-1000 millimicrons. Peaks as traced:

    yellow    0.876 at 460 nm
    magenta   0.870 at 553 nm
    cyan      1.116 at 672 nm

⚠ THIS IS THE FIRST SPECTRAL DYE SET ON A `PrintStock` IN THIS CORPUS. 5382 is
the 1954 Eastman Color print stock, i.e. 2383's direct ancestor by nearly fifty
years, and it was already in the database as `EASTMANCOLOR_5382_1953` -- with
no spectral and no dye -- while this figure was being traced. Filling that from
data already in hand is what the set is adopted for.

⚠ THREE THINGS THIS FIGURE DOES NOT SAY, AND NONE OF THEM CAN BE INVENTED:

  1. **No concentration and no reference density.** The caption says "dye
     deposits", not "midscale neutral" and not "density 1.0". The three peaks
     are 0.876 / 0.870 / 1.116, which is neither a neutral (a neutral would
     have the three near-equal by construction) nor a normalised set. What the
     curves fix is each dye's SHAPE and the three peaks' RATIO to each other,
     at whatever deposit the authors used.
  2. **No status, no illuminant, no densitometer.** 1954 predates Status A/M
     entirely.
  3. **No characteristic curve, no speed, no granularity** anywhere in the
     paper. This figure is all of 5382 that it contains.

⚠ SO WHAT IS ADOPTED IS THE SHAPE, AND THE CAVEAT TRAVELS WITH IT. The three
arrays go onto `EASTMANCOLOR_5382_1953.dye_density` on a 410-700 nm / 5 nm grid
(the plot box begins at 406.1 nm, so 400 and 405 are off the page and are not
invented), and the `normalisation` string on that record says in words that the
level is an uncalibrated deposit and must never be rescaled to a stated
density. Nothing else from this paper is adopted, because there is nothing
else: no tone scale, no speed, no granularity.

THE READING, AND WHAT MADE IT AWKWARD
--------------------------------------
The scan is microfilm and the plot has **no grid**: two axis lines and inward
tick marks, nothing else. Calibration therefore comes from the ticks, found by
projection, and is cross-checked against the OCR text layer's own tick-label
positions -- two independent readings of the same axis:

    x   traced ticks 463.5 .. 1903.7 px for 400 .. 1000 nm, 2.3973 px/nm
        OCR label centres agree to 8 px, which is label centring, not error
    y   traced ticks 3699.5 .. 3017.5 px for D 0.0 .. 1.0, 683.4 px/D
        OCR "0.2" centre 3561 against the traced tick at 3564

⚠ THE CURVES CROSS TWICE AND BOTH CROSSINGS ARE REAL. Yellow descending meets
magenta ascending near 500 nm at D 0.40, and magenta descending meets cyan
ascending near 590 nm at D 0.45; at both the two are one ink run, 32 and 36 px
thick against a 5-7 px curve. `dashtrace.trace_predictive`'s merge coast is
what carries the tracks through -- the same mechanism, and the same reason, as
Gevacolor 682 Fig. 8.

⚠ AND THE TAIL MERGES WITH THE AXIS. Past about 800 nm all three curves are
within a line width of D 0, and the drawn x-axis is the only ink there. The
traced tail is therefore reported as "at the axis" and is NOT read as a
measurement of dye density -- which is, ironically, the very point the paper
was making with this figure.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dashtrace as dt  # noqa: E402

SHEET = os.path.join("RETRO",
                     "sim_smpte-motion-imaging-journal_1954-11_63_5.pdf")

SOURCE = ("R. G. Lovick and D. R. White, «Factors in Applying Color Soundtrack "
          "Developers», Journal of the SMPTE 63(5), November 1954, p.189, "
          "Figure 2 -- 'Spectral density of dye deposits of Eastman Color "
          "Print Film, Type 5382'. PDF/PROFILES/RETRO/"
          "sim_smpte-motion-imaging-journal_1954-11_63_5.pdf page 23, "
          "Internet Archive microfilm scan")

FIG2_PAGE = 23
FIG2_DPI = 400

#: Tick centres in pixels of the pinned render, found by projection.
X_TICKS = (463.5, 706.5, 946.0, 1184.5, 1423.5, 1663.7, 1903.7)
X_NM = (400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0)
Y_TICKS = (3699.5, 3564.0, 3422.5, 3284.5, 3152.0, 3017.5)
Y_D = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#: Plot interior in pixels. The left bound sits clear of the y-axis line, which
#: is 160 px of solid ink in its own column and would otherwise be read as a
#: curve at D 1.10 in the first few columns.
BOX_Y = (2860, 3706)
X_LO, X_HI = 480, 1900

#: ⚠ SEEDED AT 460 nm, WHERE ALL THREE ARE SEPARATE. Yellow is at its own peak
#: there and the other two are still low and 0.10 D apart. Every other obvious
#: seed column has at least one pair merged.
SEED_NM = 460.0
SEEDS_D = {"yellow": 0.876, "magenta": 0.139, "cyan": 0.040}

#: ⚠ 55 px, AND IT WAS FOUND BY SWEEPING RATHER THAN REASONED FROM THE INK
#: WIDTH, WHICH IS WORTH SAYING. The merged runs at the two crossings are 32
#: and 36 px thick, so 10 px looked like the natural setting -- treat the
#: tracks as merged once their predictions are closer than a line width. It is
#: not enough: the INK merges well before the PREDICTIONS do, so at 10, 20 and
#: 30 px each track claimed the fat merged run for a stretch first, its slope
#: history absorbed the merged centroid, and magenta died at 502 nm without
#: ever reaching its own 553 nm peak. Sweeping 10 / 20 / 30 / 40 / 55 moves
#: magenta's peak 0.397 at 500 -> 0.397 at 500 -> 0.397 at 500 -> 0.397 at 500
#: -> **0.870 at 553**, which is the printed peak. The coast has to begin while
#: the two are still a whole run apart.
MERGE_PX = 55.0

#: ⚠ THE BRIDGE HAS TO OUTLAST THE CROSSING, AND THE DEFAULT DOES NOT. At 500
#: nm the yellow and magenta curves are one ink run for about sixty columns --
#: they meet at a shallow angle, unlike the near-perpendicular crossings on the
#: Gevacolor plots the default 26 was set for. With 26 both tracks coasted into
#: the merge, ran out of bridge inside it and DIED: yellow stopped at 488 nm
#: and magenta at 502, so magenta never reached its own 553 nm peak and the
#: pinned-peak check caught it. 90 carries both through with margin and does
#: not let a track wander, because the merge coast still forbids either of them
#: to claim ink while they are together.
#: 120, for the same reason and measured the same way: at merge_px 55 the coast
#: is wider, so the second crossing (magenta descending through cyan ascending
#: near 590 nm) needs more bridge than the first. At 90 cyan died at 588 nm; at
#: 120 all three carry through, and 180 and 260 change nothing, so this is
#: inside a plateau rather than balanced on an edge.
MAX_BRIDGE = 120

#: Past this the three curves and the drawn axis are one line width apart and
#: the figure stops being a measurement of anything. Reported, never adopted.
TAIL_NM = 800.0

#: Below this the curve and the drawn x-axis are indistinguishable, so the
#: trace past a dye's own descent into the axis is FOLLOWING THE AXIS, not
#: measuring the dye. Each dye's usable span is reported, and cut, on this.
AXIS_FLOOR_D = 0.02

#: What a re-run must reproduce: (peak density, peak wavelength) per dye.
EXPECTED_PEAKS = {
    "yellow": (0.876, 460.0),
    "magenta": (0.870, 553.0),
    "cyan": (1.116, 672.0),
}
PEAK_D_TOL = 0.02
PEAK_NM_TOL = 6.0


def page_gray(root=".", page=FIG2_PAGE, dpi=FIG2_DPI):
    """The page as float grayscale in [0, 1]; None when the PDF is absent."""
    import pymupdf
    path = os.path.join(root, "PDF", "PROFILES", SHEET)
    if not os.path.isfile(path):
        return None
    doc = pymupdf.open(path)
    pm = doc[page - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width)
    doc.close()
    return a.astype(np.float64) / 255.0


def calibration():
    mx, bx = np.polyfit(np.asarray(X_NM), np.asarray(X_TICKS), 1)
    my, by = np.polyfit(np.asarray(Y_D), np.asarray(Y_TICKS), 1)
    return (lambda x: (x - bx) / mx, lambda y: (y - by) / my,
            lambda nm: mx * nm + bx, lambda d: my * d + by)


def ocr_ticks(root=".", page=FIG2_PAGE, dpi=FIG2_DPI):
    """Tick-label centres from the scan's OWN text layer, in render pixels.

    ⚠ A SECOND, INDEPENDENT READING OF THE SAME AXIS. The projection finds the
    drawn tick marks; this finds where the printed numbers sit. They are
    produced by different parts of the page and by different code, so agreement
    between them is a real check on the calibration rather than a restatement
    of it.
    """
    import pymupdf
    path = os.path.join(root, "PDF", "PROFILES", SHEET)
    if not os.path.isfile(path):
        return {}
    doc = pymupdf.open(path)
    sc = dpi / 72.0
    out = {}
    for w in doc[page - 1].get_text("words"):
        t = w[4].strip()
        cx, cy = (w[0] + w[2]) / 2 * sc, (w[1] + w[3]) / 2 * sc
        if t.isdigit() and 3690 < cy < 3760 and 400 <= int(t) <= 1000:
            out[("x", float(t))] = cx
        elif t in ("0.0", "0.2") and 3050 < cy < 3760:
            out[("y", float(t))] = cy
    doc.close()
    return out


def trace(gray):
    """{dye: [(nm, D), ...]} left to right, plus the merged-tail cut."""
    nm_of_x, D_of_y, x_of_nm, y_of_D = calibration()
    ink = gray < 150 / 255.0
    y0, y1 = BOX_Y
    seed_x = int(round(x_of_nm(SEED_NM)))
    seeds = {k: y_of_D(v) for k, v in SEEDS_D.items()}
    right = dt.trace_predictive(ink, gray, (seed_x, X_HI), y0, y1, seed_x,
                                seeds, direction=+1, max_bridge=MAX_BRIDGE,
                                merge_px=MERGE_PX)
    left = dt.trace_predictive(ink, gray, (X_LO, seed_x), y0, y1, seed_x,
                               seeds, direction=-1, max_bridge=MAX_BRIDGE,
                               merge_px=MERGE_PX)
    out = {}
    for k in seeds:
        pts = dict(left[k])
        pts.update(right[k])
        out[k] = [(nm_of_x(x), D_of_y(y)) for x, y in sorted(pts.items())]
    return out


def usable(curve, floor=AXIS_FLOOR_D):
    """The stretch of a traced dye that is the dye and not the drawn axis.

    ⚠ EVERY TRACK RUNS PAST ITS OWN CURVE. Once a dye has descended to the
    x-axis the only ink in the column is the axis line, the track finds it, and
    the trace continues flat to the right-hand edge -- magenta comes back
    spanning 406-701 nm when the printed magenta curve is gone by about 640.
    Those samples are not wrong so much as meaningless, and leaving them in
    would put a fabricated 0.00x D reading at every wavelength the dye does not
    reach. Cut at the LAST sample above the floor on each side of the peak.
    """
    pk = max(range(len(curve)), key=lambda i: curve[i][1])
    lo = pk
    while lo > 0 and curve[lo - 1][1] > floor:
        lo -= 1
    hi = pk
    while hi < len(curve) - 1 and curve[hi + 1][1] > floor:
        hi += 1
    return curve[lo:hi + 1]


# ---------------------------------------------------------------------------
# THE TWO TAIL PASSES, ADDED 2026-09-03 WHEN THE SET WAS ACTUALLY ADOPTED.
# ---------------------------------------------------------------------------
# ⚠ WHY THEY EXIST, AND IT IS A DEFECT IN THE FIRST READING RATHER THAN A
# REFINEMENT OF IT. `trace()` runs all three dyes as one three-track pass with
# ONE merge_px, and 55 px is the value the MAGENTA peak needed. That same 55 px
# (0.080 D here, 23 nm wide) is far too coarse for the two places where two
# curves pass within a few thousandths of each other down in the 0.02-0.25 D
# grass, and in BOTH of them a track came back holding a frozen value that
# `usable()` then dressed up as a measurement:
#
#   * YELLOW crosses the rising cyan at about 537 nm. The joint track coasted,
#     re-acquired on the DRAWN AXIS, and yellow was reported as ending at
#     526.7 nm still at 0.158 D -- a dye that simply stops mid-descent. The ink
#     is there: an independent column probe reads yellow at 0.092 (540 nm),
#     0.059 (550), 0.034 (560), 0.017 (570), 0.008 (580), 0.004 (590).
#   * MAGENTA AND CYAN CROSS AT ABOUT 430 nm, and the joint pass missed the
#     crossing entirely: magenta was traced as the axis below 455 nm and cyan
#     was HELD at 0.103 from 410-430 and at 0.049 from 435-455. Those two held
#     numbers are what the first emitted 5 nm cyan array interpolated between,
#     so cyan 415-455 in that array was never a reading of anything.
#
# Each tail is therefore re-traced on its own, seeded from a point of the joint
# trace that IS trustworthy, and the result is checked against the raw ink.
#
#: Seed for the leftward magenta/cyan pass -- 465 nm, five nanometres clear of
#: the last column where the joint trace and the ink agree exactly (460 nm:
#: magenta 0.139 traced against 0.140 in the ink, cyan 0.040 against 0.040).
LEFT_SEED_NM = 465.0
#: ⚠ SMALL ENOUGH TO SEE THE 430 nm CROSSING AT ALL. Measured plateau: 8, 9, 10
#: and 11 px give byte-identical tails; at 12 px both tracks die before 435 nm
#: and at 6 px the crossing needs tol0 4.0 to survive. 10 is the centre.
LEFT_MERGE_PX = 10.0
#: Seed for the rightward yellow pass, taken from the joint trace's own last
#: uncontaminated sample before the cyan crossing.
YELLOW_SEED_NM = 523.8
#: Shared tail-pass tolerances. Plateau: max_bridge 12/26/60/120 identical,
#: tol0 3.0 and 4.0 identical; tol0 2.5 and below kills the yellow track at the
#: crossing, which is the only edge in the set.
TAIL_TOL0, TAIL_TOL_GROW, TAIL_MAX_BRIDGE = 3.0, 0.4, 26

#: ⚠ WHAT THE CROSSING ASSIGNMENT MUST REPRODUCE. Read off the raw ink by a
#: column probe that shares no code with the tracker: at each wavelength the
#: dark runs are found, their centres converted to D, and the tick marks
#: (short runs sitting exactly on 0.2 D multiples near the left axis) ignored.
#: Both branches are smooth under this assignment and neither is under the
#: swap, which is the independent argument for it; the tracker's slope
#: prediction through the coasted crossing then chooses the same one.
INK_PROBE = {
    "cyan":    {410: 0.224, 420: 0.174, 440: 0.092, 450: 0.061},
    "magenta": {410: 0.100, 420: 0.118, 440: 0.129, 450: 0.126},
    "yellow":  {550: 0.059, 560: 0.034, 570: 0.017, 580: 0.008},
}
INK_PROBE_TOL = 0.006

#: Below this the assembled dye is at the drawn axis and is stored as 0.0. It
#: is DELIBERATELY tighter than AXIS_FLOOR_D, which cuts a trace; this one only
#: decides where a real descent stops being separable from the axis line.
ZERO_FLOOR_D = 0.005


def _tail(gray, seeds_d, seed_nm, x_range, direction, merge_px=0.0):
    """One extra `trace_predictive` pass, seeded on the joint trace."""
    nm_of_x, D_of_y, x_of_nm, y_of_D = calibration()
    ink = gray < 150 / 255.0
    y0, y1 = BOX_Y
    sx = int(round(x_of_nm(seed_nm)))
    got = dt.trace_predictive(
        ink, gray, x_range, y0, y1, sx,
        {k: y_of_D(v) for k, v in seeds_d.items()}, direction=direction,
        max_bridge=TAIL_MAX_BRIDGE, tol0=TAIL_TOL0, tol_grow=TAIL_TOL_GROW,
        merge_px=merge_px)
    return {k: [(nm_of_x(x), D_of_y(v)) for x, v in sorted(d.items())]
            for k, d in got.items()}


def _at(curve, nm, tol=0.6):
    if not curve:
        return None
    p = min(curve, key=lambda q: abs(q[0] - nm))
    return p[1] if abs(p[0] - nm) <= tol else None


def assemble(gray, curves=None):
    """{dye: [(nm, D)]} -- the joint trace with both tails repaired.

    Each dye keeps the joint trace only over the span where the joint trace was
    verified against the ink, and takes the dedicated pass everywhere else:

        yellow    joint  <= 526.7 nm      tail pass  above it
        magenta   joint  >= 460 nm        tail pass  below it
        cyan      joint  >= 460 nm        tail pass  below it

    The 430 nm crossing itself is left EMPTY by the merge coast -- both tracks
    refuse the ink there -- so the grid point at 430 is an interpolation across
    about three columns, and is marked as such wherever it is adopted.
    """
    if curves is None:
        curves = trace(gray)
    seed_l = {k: _at(curves[k], LEFT_SEED_NM) for k in ("magenta", "cyan")}
    left = _tail(gray, seed_l, LEFT_SEED_NM,
                 (X_LO, int(round(calibration()[2](LEFT_SEED_NM)))), -1,
                 merge_px=LEFT_MERGE_PX)
    seed_y = {"yellow": _at(curves["yellow"], YELLOW_SEED_NM)}
    right = _tail(gray, seed_y, YELLOW_SEED_NM,
                  (int(round(calibration()[2](YELLOW_SEED_NM))), X_HI), +1)

    out = {}
    base_y = usable(curves["yellow"])
    cut_y = base_y[-1][0]
    out["yellow"] = base_y + [p for p in right["yellow"] if p[0] > cut_y]
    for dye in ("magenta", "cyan"):
        out[dye] = ([p for p in left[dye] if p[0] < LEFT_SEED_NM - 5.0]
                    + [p for p in curves[dye] if p[0] >= LEFT_SEED_NM - 5.0])
        out[dye] = usable(out[dye], floor=-1.0)  # keep everything, order it
        out[dye] = sorted(out[dye])
    return out


def adopted(gray, lo=410.0, hi=700.0, step=5.0, curves=None):
    """The assembled set on one shared grid, ready to store.

    ⚠ ZERO IS A READING HERE, NOT A PAD. Past the wavelength where a dye's own
    descent reaches ZERO_FLOOR_D the printed curve and the drawn axis are one
    line, so the figure says "at most 0.005 D" and that is stored as 0.0. The
    alternative -- leaving those wavelengths out -- is not available: the
    schema requires the three dyes to share one grid, and cutting the grid to
    the span all three are non-trivial on would throw away the whole of
    yellow's 410-460 nm rise.
    """
    a = assemble(gray, curves)
    n = int(round((hi - lo) / step)) + 1
    out = {}
    for dye, cur in a.items():
        pk = max(range(len(cur)), key=lambda i: cur[i][1])
        xs = [p[0] for p in cur]
        ys = [p[1] for p in cur]
        vals = []
        for i in range(n):
            g = lo + i * step
            if g < xs[0] - 1e-9 or g > xs[-1] + 1e-9:
                vals.append(0.0)
                continue
            v = float(np.interp(g, xs, ys))
            # past the dye's own descent, the curve IS the axis
            if v < ZERO_FLOOR_D:
                v = 0.0
            vals.append(round(max(v, 0.0), 3))
        # ⚠ once a dye has gone to the floor on a side it stays there: a later
        # non-zero would be another curve's ink, not a second lobe.
        first = int(round((cur[pk][0] - lo) / step))
        first = max(0, min(n - 1, first))
        for i in range(first, n):
            if vals[i] == 0.0:
                vals[i + 1:] = [0.0] * (n - i - 1)
                break
        for i in range(first, -1, -1):
            if vals[i] == 0.0:
                vals[:i] = [0.0] * i
                break
        out[dye] = tuple(vals)
    return out


def resample(curve, lo, hi, step):
    xs = np.asarray([p[0] for p in curve])
    ys = np.asarray([p[1] for p in curve])
    n = int(round((hi - lo) / step)) + 1
    out = []
    for i in range(n):
        g = lo + i * step
        out.append((g, None if (g < xs.min() - 1e-9 or g > xs.max() + 1e-9)
                    else float(np.interp(g, xs, ys))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ns = ap.parse_args(argv)

    print("[i] %s" % SOURCE)
    gray = page_gray(ns.root)
    if gray is None:
        print("  [SKIP] source not present: %s" % SHEET)
        return 0

    bad = 0
    nm_of_x, D_of_y, x_of_nm, y_of_D = calibration()

    # ---- the two independent axis readings --------------------------------
    ocr = ocr_ticks(ns.root)
    dx = [abs(x_of_nm(nm) - px) for (ax, nm), px in ocr.items() if ax == "x"]
    dy = [abs(y_of_D(d) - py) for (ax, d), py in ocr.items() if ax == "y"]
    if dx:
        ok = max(dx) < 12.0
        bad += (not ok)
        print("  [%s] the drawn ticks and the printed labels agree on the "
              "wavelength axis to %.1f px (%.1f nm) over %d labels"
              % ("OK  " if ok else "FAIL", max(dx), max(dx) / 2.3973, len(dx)))
    if dy:
        ok = max(dy) < 14.0
        bad += (not ok)
        print("  [%s] and on the density axis to %.1f px (%.3f D)"
              % ("OK  " if ok else "FAIL", max(dy), max(dy) / 683.43))

    curves = trace(gray)

    print("\n  Traced dye curves -- traced span, then the span that is the "
          "DYE rather than the drawn axis")
    for dye in ("yellow", "magenta", "cyan"):
        c = curves[dye]
        u = usable(c)
        pk = max(c, key=lambda p: p[1])
        print("      %-8s n=%-4d  traced %6.1f..%-6.1f  usable %6.1f..%-6.1f "
              "nm   peak %.3f at %.1f nm"
              % (dye, len(c), c[0][0], c[-1][0], u[0][0], u[-1][0],
                 pk[1], pk[0]))

    for dye, (wd, wnm) in EXPECTED_PEAKS.items():
        pk = max(curves[dye], key=lambda p: p[1])
        ok = abs(pk[1] - wd) < PEAK_D_TOL and abs(pk[0] - wnm) < PEAK_NM_TOL
        bad += (not ok)
        print("    [%s] %-8s peak %.3f at %.1f nm against the pinned "
              "%.3f at %.0f" % ("OK  " if ok else "FAIL", dye,
                                pk[1], pk[0], wd, wnm))

    # ⚠ ORDERING, because three smooth curves can be three smooth WRONG curves.
    order = sorted(("yellow", "magenta", "cyan"),
                   key=lambda d: max(curves[d], key=lambda p: p[1])[0])
    ok = order == ["yellow", "magenta", "cyan"]
    bad += (not ok)
    print("    [%s] the peaks still run yellow < magenta < cyan in wavelength "
          "-- the only thing that names these three curves"
          % ("OK  " if ok else "FAIL"))

    # ⚠ THE SEPARATION AFTER EACH CROSSING is what says the merge coast worked.
    y600 = float(np.interp(600.0, [p[0] for p in curves["yellow"]],
                           [p[1] for p in curves["yellow"]]))
    m460 = float(np.interp(460.0, [p[0] for p in curves["magenta"]],
                           [p[1] for p in curves["magenta"]]))
    ok = y600 < 0.05 and m460 < 0.25
    bad += (not ok)
    print("    [%s] and neither track followed the other out of a crossing: "
          "yellow is %.3f at 600 nm, magenta %.3f at 460 nm"
          % ("OK  " if ok else "FAIL", y600, m460))

    # the infrared tail, which is the paper's own point
    c = curves["cyan"]
    tail = [d for nm, d in c if nm >= 900.0]
    if tail:
        print("    [note] cyan reads %.3f D at 900 nm and beyond -- the "
              "figure exists to show that dye alone gives no infrared "
              "density, and past %.0f nm all three curves are within a line "
              "width of the drawn axis, so nothing there is adopted"
              % (max(tail), TAIL_NM))

    # ---- the two repaired tails, and the ink they must reproduce ----------
    # ⚠ THE CHECK SHARES NO CODE WITH THE TRACKER. INK_PROBE was read off the
    # raw page by finding dark runs per column and taking their centres; if the
    # tail passes ever drift onto the wrong branch of either crossing, these
    # twelve numbers move by tenths, not thousandths.
    adopt = adopted(gray, curves=curves)
    worst, worst_at = 0.0, ""
    for dye, want in INK_PROBE.items():
        for nm, d in want.items():
            got = adopt[dye][int(round((nm - 410.0) / 5.0))]
            if abs(got - d) > worst:
                worst, worst_at = abs(got - d), "%s %.0f nm" % (dye, nm)
    ok = worst <= INK_PROBE_TOL
    bad += (not ok)
    print("    [%s] both repaired tails reproduce the raw ink to %.3f D "
          "(worst: %s) over %d independent column readings"
          % ("OK  " if ok else "FAIL", worst, worst_at,
             sum(len(v) for v in INK_PROBE.values())))

    # ⚠ THE CROSSING ASSIGNMENT ITSELF, stated as the thing that could be
    # wrong: below 430 nm cyan must be ABOVE magenta and above 435 nm below it.
    ok = (adopt["cyan"][0] > adopt["magenta"][0] + 0.10
          and adopt["cyan"][8] < adopt["magenta"][8])
    bad += (not ok)
    print("    [%s] magenta and cyan are on the branches the crossing at "
          "~430 nm assigns them: at 410 nm cyan %.3f over magenta %.3f, at "
          "450 nm cyan %.3f under magenta %.3f"
          % ("OK  " if ok else "FAIL", adopt["cyan"][0], adopt["magenta"][0],
             adopt["cyan"][8], adopt["magenta"][8]))

    # ⚠ AND THE YELLOW TAIL EXISTS AT ALL. Before the repair yellow stopped at
    # 526.7 nm still at 0.158 D; a dye that ends mid-descent is a tracking
    # failure, and this is the guard that says it has not come back.
    last = max(i for i, v in enumerate(adopt["yellow"]) if v > 0.0)
    ok = 410 + 5 * last >= 575 and adopt["yellow"][last] < 0.02
    bad += (not ok)
    print("    [%s] yellow now descends to the axis on its own: last non-zero "
          "%.3f D at %d nm" % ("OK  " if ok else "FAIL",
                               adopt["yellow"][last], 410 + 5 * last))

    if ns.emit:
        print("\n  --- ADOPTED: 410-700 nm, 5 nm, all three on one grid ---")
        for dye in ("yellow", "magenta", "cyan"):
            print("  %s = (%s)" % (dye, ", ".join(
                "%.3f" % v for v in adopt[dye])))
        print("\n  --- 5 nm resample, 400-700 nm ---")
        for dye in ("yellow", "magenta", "cyan"):
            r = resample(usable(curves[dye]), 400, 700, 5)
            print("  %s = (%s)" % (dye, ", ".join(
                "None" if v is None else "%.3f" % v for _g, v in r)))
        print("\n  --- 10 nm resample, 700-1000 nm (the infrared tail) ---")
        for dye in ("cyan",):
            r = resample(usable(curves[dye]), 700, 1000, 10)
            print("  %s = (%s)" % (dye, ", ".join(
                "None" if v is None else "%.3f" % v for _g, v in r)))

    print()
    if bad:
        print("  [FAIL] %d problem(s)" % bad)
        return 1 if ns.do_assert else 0
    print("  [OK  ] EASTMAN COLOR PRINT FILM 5382: three dye deposit spectra "
          "traced 400-1000 nm, calibration confirmed against the scan's own "
          "text layer, both low crossings repaired and pinned to the raw ink; "
          "410-700 nm at 5 nm adopted onto EASTMANCOLOR_5382_1953 as SHAPE "
          "ONLY -- the level is an uncalibrated 1954 dye deposit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
