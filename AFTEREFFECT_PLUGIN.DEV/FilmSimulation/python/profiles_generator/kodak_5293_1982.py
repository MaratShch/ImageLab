#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kennel, Sehlin, Reinking, Spakowsky and Whittier 1982: EASTMAN COLOR
HIGH-SPEED NEGATIVE FILM 5293, the EI 250T emulsion -- every figure traced.

WHAT THIS SOURCE IS
-------------------
**G. L. Kennel, R. C. Sehlin, F. R. Reinking, S. W. Spakowsky and G. L.
Whittier, "Eastman Color High-Speed Negative Film 5293", SMPTE Journal vol. 91
no. 10, October 1982, pp. 922-930** --
`PDF/PROFILES/KODAK/Kennel_Sehlin_etalNeg5293_1982.pdf`, nine pages, a pure
raster scan with NO TEXT LAYER of any kind.

⚠ IT IS TIER T1. All five authors are Eastman Kodak Company, Rochester, and
the paper is Kodak's own product announcement for its own film, carrying the
measurements Kodak's laboratory made on it. It is not a datasheet, and the
distinction matters in one direction only: a paper prints the figure and the
method, where a datasheet prints the figure alone.

⚠⚠ AND IT IS A DIFFERENT FILM FROM THE 5293 THIS DATABASE ALREADY HELD.
Eastman used the code 5293 twice:

    1982   EASTMAN COLOR HIGH-SPEED NEGATIVE 5293    EI 250T   this paper
    1992   EASTMAN EXR 200T 5293                     EI 200T   H-1-5293

They are ten years and one emulsion technology apart -- the 1982 film is built
on conventional polydisperse three-dimensional silver-halide crystals, the
1992 film on EXR tabular grains -- and merging them would average two coatings
that share nothing but a catalogue number. `EASTMAN_EXR_200T_5293` keeps the
1992 sheet; this module supplies `EASTMAN_5293_250T_1982`.

WHAT THE PAPER GIVES, FIGURE BY FIGURE
--------------------------------------
    Fig. 1   layer structure, 10 named layers, top to bottom      transcribed
    Fig. 2   a schematic of turbidity vs thickness                not data
    Fig. 3   microdensitometer line scans, no calibrated axes     refused
    Fig. 4   D-log E, 5293 and 5247, three records each           TRACED
    Fig. 5   spectral sensitivity, 5293 and 5247                  TRACED
    Fig. 6   trilinear exposure densities                         not a curve
    Fig. 7   spectral dye density, C/M/Y, 5293 and 5247           TRACED
    Fig. 8   trilinear print exposure densities                   not a curve
    Fig. 9   a schematic of colour masking                        not data
    Fig. 10  a schematic of the DIR interimage effect             not data
    Fig. 11  CIELAB a*/b* of printed patches                      TRACED
    Fig. 12  D-log E, normal process against push-1               TRACED
    Fig. 13  RMS granularity AND density against log E            TRACED
    Fig. 14  MTF, 5293 and 5247                                   TRACED
    Fig. 15  cyan dye fading against time, four temperatures      TRACED
    Fig. 16  subjective picture quality against exposure index    ordinate bare
    Table 1  history of Eastman colour negative, 1950-1976        transcribed
    Table 2  resolving power at two test-object contrasts         transcribed

⚠ WHAT IS REFUSED AND WHY, so neither is filed later as an oversight.
**Fig. 3** is a microdensitometer trace of three line exposures with no
calibrated axis on either side; it shows that an edge effect exists and
supports no number. **Fig. 16** plots "OVERALL PICTURE QUALITY" against
exposure index with only two reference lines on the ordinate -- "NOTICEABLE
DIFFERENCE IN QUALITY" and "MINIMUM ACCEPTABLE QUALITY" -- and no scale, so
the EI values at which the curves cross those lines are readable and the
quality values are not. Those crossings ARE extracted, as exposure indices,
because they are on the calibrated axis.

⚠⚠ HOW THE TRACES ARE CHECKED, AND IT IS THE PAPER CHECKING ITSELF. Every
figure here plots 5293 BESIDE 5247, and the body text makes three quantitative
claims about that pair which the traces must reproduce or be wrong:

    "one-and-one-third stops (0.40 log E) faster than the current 5247 film"
    "the same contrast and latitude as the other Eastman color negative films"
    "the MTF's and resolving powers of the two films are similar"

The first two are a HORIZONTAL displacement and a SLOPE, which are exactly the
two quantities a mis-calibrated axis gets wrong, and they are independent of
each other. `check_speed_offset` and `check_equal_contrast` assert them; a
tracing or calibration error large enough to matter cannot pass both.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

SHEET = os.path.join("KODAK", "Kennel_Sehlin_etalNeg5293_1982.pdf")

DPI = 400.0
INK = 150            #: 8-bit grey below this is ink. The scan is bilevel-clean.

#: The paper's own stated speed difference, in log exposure. Fig. 4 marks it
#: with a dimensioned arrow and the body text states it twice.
STATED_OFFSET = 0.40
OFFSET_TOL = 0.08    #: how far the traced offset may sit from 0.40 log E.
CONTRAST_TOL = 0.06  #: how far the two films' traced gammas may differ.


# ---------------------------------------------------------------------------
#  Axis calibration, transcribed once per figure
# ---------------------------------------------------------------------------
# ⚠ THE ANCHORS ARE TICK-LABEL CENTROIDS FOUND GEOMETRICALLY AND THEIR VALUES
# ARE TRANSCRIBED BY EYE, which is the arrangement `kodak_1956_trace.py`
# arrived at after an OCR-layer reader failed on thirteen pages of fourteen.
# Here there is no text layer at all, so the question does not even arise: the
# scan carries pixels and nothing else.
#
# Each entry is (page, frame box, y anchors, x anchors, scale kind). An anchor
# is (pixel, value) at 400 dpi, and the scale kind says whether `value` is the
# quantity or its base-10 logarithm.

FIG4 = dict(                                    # D-log E, 5293 vs 5247
    page=3, box=(463, 2097, 145, 1372),
    y=((150, 3.0), (557, 2.0), (961, 1.0)),
    x=((870, 1.0), (1286, 2.0), (1700, 3.0)),
    ylog=False, xlog=False,
)
FIG5 = dict(                                    # spectral sensitivity
    page=3, box=(485, 2097, 1713, 3841),
    y=((1714, 3.0), (2245, 2.0), (2773, 1.0), (3303, 0.0), (3822, -1.0)),
    x=((686, 400.0), (1090, 500.0), (1492, 600.0), (1897, 700.0)),
    ylog=False, xlog=False,
)
FIG7 = dict(                                    # spectral dye density
    page=4, box=(1324, 2950, 1830, 2659),
    y=((1856, 1.50), (1983, 1.25), (2120, 1.00), (2256, 0.75),
       (2392, 0.50), (2526, 0.25), (2651, 0.0)),
    x=((1470, 400.0), (1654, 450.0), (1840, 500.0), (2392, 650.0),
       (2576, 700.0), (2760, 750.0)),
    ylog=False, xlog=False,
)
FIG12 = dict(                                   # normal vs push-1
    page=7, box=(1356, 3010, 966, 2235),
    y=((968, 3.0), (1396, 2.0), (1812, 1.0)),
    x=((1783, 1.0), (2644, 3.0)),
    ylog=False, xlog=False,
)
FIG13 = dict(                                   # granularity and density
    page=7, box=(460, 2858, 2567, 3997),
    y=((2725, 2.0), (3518, 1.0), (3838, 0.6)),
    x=(),                                       # ⚠ the abscissa is bare
    y2=((3022, 20.0), (3240, 10.0), (3319, 8.0), (3415, 6.0),
        (3538, 4.0), (3773, 2.0), (3969, 1.0)),
    ylog=False, y2log=True, xlog=False,
)
FIG14 = dict(                                   # MTF
    page=8, box=(1267, 2919, 672, 1894),
    y=((634, 150.0), (749, 100.0), (849, 70.0), (952, 50.0), (1099, 30.0),
       (1217, 20.0), (1421, 10.0), (1523, 7.0), (1627, 5.0), (1771, 3.0),
       (1872, 2.0)),
    x=((1270, 1.0), (1731, 5.0), (1927, 10.0), (2126, 20.0), (2394, 50.0),
       (2595, 100.0), (2800, 200.0)),
    ylog=True, xlog=True,
)
FIG15 = dict(                                   # cyan dye fade
    page=9, box=(505, 3040, 150, 1443),
    y=((398, 1.0), (654, 0.9), (922, 0.8), (1186, 0.7), (1442, 0.6)),
    # ⚠ TWO SEPARATE LOG PANELS SHARE ONE FRAME. The abscissa runs 1 to 200
    # DAYS in its left half and 1 to 200 YEARS in its right, with no break
    # drawn between them, so one calibration across the whole width would put
    # the accelerated tests and the room-temperature prediction on one scale
    # and be wrong about both.
    x=((506, 1.0), (1023, 10.0), (1545, 100.0)),          # days panel
    x2=((1835, 1.0), (2356, 10.0), (2885, 100.0)),        # years panel
    x_split=1780,
    ylog=False, xlog=True, x2log=True,
)


def _fit(anchors, log):
    """Linear pixel -> value (or -> log10 value) from transcribed anchors."""
    px = np.array([float(p) for p, _v in anchors])
    vv = np.array([np.log10(float(v)) if log else float(v)
                   for _p, v in anchors])
    a, b = np.polyfit(px, vv, 1)
    res = float(np.max(np.abs(a * px + b - vv)))
    if log:
        return (lambda q: float(10 ** (a * q + b))), res
    return (lambda q: float(a * q + b)), res


def axes(fig, key="x"):
    """(mapping, worst anchor residual) for one axis of one figure."""
    return _fit(fig[key], fig.get(key.replace("x", "x").replace("y", "y")
                                  + "log", False) if False
                else fig.get(key + "log", False))


# ---------------------------------------------------------------------------
#  Raster half
# ---------------------------------------------------------------------------

def page_gray(root=".", page=3):
    """One page of the scan as an 8-bit grey array at 400 dpi, or None."""
    import pymupdf
    path = os.path.join(root, "PDF", "PROFILES", SHEET)
    if not os.path.isfile(path):
        return None
    doc = pymupdf.open(path)
    pm = doc[page - 1].get_pixmap(dpi=int(DPI), colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width)
    doc.close()
    return a


def plot_mask(gray, box, inset=(22, 22), erase=()):
    """Ink inside the plot frame, with the frame's own tick marks removed.

    ⚠ THE INSET IS NOT COSMETIC. Every frame in this paper carries tick marks
    that point INWARD twenty pixels or so at 400 dpi, and a tick is a short
    run of ink in exactly the place a curve's end would be. Left in, they are
    read as curve points at the axis value the tick marks.
    """
    x0, x1, y0, y1 = box
    ix, iy = inset
    m = np.zeros(gray.shape, dtype=np.uint8)
    m[y0 + iy:y1 - iy, x0 + ix:x1 - ix] = \
        (gray[y0 + iy:y1 - iy, x0 + ix:x1 - ix] < INK)
    for ex0, ex1, ey0, ey1 in erase:
        m[ey0:ey1, ex0:ex1] = 0
    return m


def split_style(mask, solid_w, dash_area=60, dash_w=15, dash_h=70,
                drop=()):
    """(solid, dashed) ink, separated by connected-component SHAPE.

    ⚠ THE TWO FILMS ARE DRAWN IN DIFFERENT LINE STYLES ON EVERY COMPARATIVE
    FIGURE HERE, and that is the only thing that tells them apart where they
    run close together. A solid curve is one component hundreds of pixels
    wide; a dashed one is a row of components fifty wide. Separating them
    BEFORE any tracing makes a jump from one film to the other impossible
    rather than merely unlikely -- the failure `dashtrace` was written for.
    """
    import cv2
    n, lab, st, _c = cv2.connectedComponentsWithStats(mask, 8)
    solid = np.zeros_like(mask)
    dash = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        # ⚠ A LEGEND OR AN ANNOTATION IS DROPPED AS A WHOLE COMPONENT AND
        # NEVER AS A RECTANGLE OF PIXELS. Fig. 4's dimensioned 0.40 log E
        # arrow sits ON the red 5293 curve; blanking its rectangle first
        # splits an 1160 px component into two of 321 and 640, neither of
        # which passes the solid-width test, and the record then vanishes from
        # the figure with no error raised anywhere. Containment leaves a curve
        # that merely passes through the same rectangle untouched.
        if any(bx0 <= x and x + w <= bx1 and by0 <= y and y + h <= by1
               for bx0, bx1, by0, by1 in drop):
            continue
        if w >= solid_w:
            solid[lab == i] = 1
        elif area >= dash_area and w >= dash_w and h < dash_h:
            dash[lab == i] = 1
    return solid, dash


def runs(mask, col):
    """Ink-run centres in one column."""
    ys = np.flatnonzero(mask[:, col])
    out = []
    for y in ys:
        if out and y - out[-1][-1] <= 3:
            out[-1].append(y)
        else:
            out.append([y])
    return [float(np.mean(r)) for r in out]


def stack(mask, x_from, x_to, n_curves, step=4):
    """Trace `n_curves` NON-CROSSING curves by vertical order per column.

    Used only where the figure itself guarantees the order: Figs 4, 5 and 12
    draw their three records with a constant separation and no intersection
    anywhere in frame. Columns holding a different number of runs are skipped
    rather than guessed at, so a label or an annotation costs samples and
    never costs correctness.
    """
    out = [[] for _ in range(n_curves)]
    for x in range(x_from, x_to, step):
        r = runs(mask, x)
        if len(r) != n_curves:
            continue
        for k in range(n_curves):
            out[k].append((x, r[k]))
    return out


def follow(mask, x_from, x_to, seeds, step=3, win=18, hist=14):
    """Trace curves that DO cross, by exclusive assignment on slope.

    ⚠ EXCLUSIVE, AND THAT IS THE WHOLE POINT ON FIG. 13. Its rising density
    curve and its falling 5293 granularity curve are both solid, cross once
    near log E 1.1, and a nearest-ink follower takes the wrong branch there
    every time -- the granularity trace climbs to RMS 54, which is not a
    number this figure contains. Each track predicts from a linear fit over
    its own last `hist` points and the two are matched to the two runs in
    order of prediction error, so at the crossing the better-predicted track
    keeps the ink and the other waits.
    """
    tracks = [{x_from: s} for s in seeds]
    for x in range(x_from + step, x_to, step):
        r = runs(mask, x)
        if not r:
            continue
        pred = []
        for t in tracks:
            xs = sorted(t)[-hist:]
            if len(xs) >= 4:
                a, b = np.polyfit(xs, [t[q] for q in xs], 1)
                pred.append(a * x + b)
            else:
                pred.append(t[max(t)])
        cand = sorted((abs(r[j] - pred[i]), i, j)
                      for i in range(len(tracks)) for j in range(len(r)))
        ut, ur = set(), set()
        for d, i, j in cand:
            if i in ut or j in ur or d > win:
                continue
            ut.add(i)
            ur.add(j)
            tracks[i][x] = r[j]
    return tracks


def trace_hump(mask, x_seed, y_seed, x_lo, x_hi, step=2, win=55, hist=10):
    """One single-valued curve, traced outward from a column where it is alone.

    ⚠ FIG. 5 IS READ HUMP BY HUMP AND NOT BY A GENERAL TRACER, and the
    reason is a property of the figure rather than of the tracer. Its three
    records overlap PAIRWISE and never all at once -- the blue one is below
    the frame by 540 nm, the green one only appears at 535 and is gone by
    615, the red one starts at 600 -- so there is no column to seed three
    tracks at, and a sweep that lets curves be born fragments instead,
    because both flanks are steeper than the prediction window a crossing
    needs to be narrow. What each record DOES have is a wavelength where it
    is the only ink in frame: 400 nm, 550 nm and 650 nm. Seeded there and
    walked outward, each is unambiguous at every step.
    """
    pts = {x_seed: y_seed}

    def walk(rng):
        for x in rng:
            xs = sorted(pts)[-hist:] if rng.step > 0 else sorted(pts)[:hist]
            if len(xs) >= 4:
                a, b = np.polyfit(xs, [pts[q] for q in xs], 1)
                pred = a * x + b
            else:
                pred = pts[min(pts, key=lambda q: abs(q - x))]
            r = runs(mask, x)
            if not r:
                continue
            c = min(r, key=lambda v: abs(v - pred))
            if abs(c - pred) <= win:
                pts[x] = c

    walk(range(x_seed + step, x_hi, step))
    walk(range(x_seed - step, x_lo, -step))
    return pts


def sweep_tracks(mask, x_lo, x_hi, step=3, win=16, hist=12, gap_kill=12,
                 min_span=200):
    """Every curve in the frame, found by sweeping and letting them be born.

    ⚠ FIG. 5 HAS NO COLUMN WHERE ALL THREE RECORDS EXIST, which is why a
    seeded tracer cannot read it however the seed is chosen. The blue record
    has fallen off the bottom of the frame by 540 nm, the green one only rises
    above it at 535 and is gone by 615, and the red one appears at 600: the
    three overlap pairwise and never together. A curve therefore has to be
    allowed to BEGIN in the middle of the sweep, which is the one thing a
    fixed set of seeds cannot express.

    Assignment stays exclusive, a track that finds nothing for `gap_kill`
    columns is closed, and a run no live track claims starts a new one.
    """
    live, done = [], []
    for x in range(x_lo, x_hi, step):
        r = runs(mask, x)
        pred = []
        for t in live:
            xs = sorted(t["p"])[-hist:]
            if len(xs) >= 4:
                a, b = np.polyfit(xs, [t["p"][q] for q in xs], 1)
                pred.append(a * x + b)
            else:
                pred.append(t["p"][max(t["p"])])
        cand = sorted((abs(r[j] - pred[i]), i, j)
                      for i in range(len(live)) for j in range(len(r)))
        ut, ur = set(), set()
        for d, i, j in cand:
            if i in ut or j in ur or d > win:
                continue
            ut.add(i)
            ur.add(j)
            live[i]["p"][x] = r[j]
            live[i]["m"] = 0
        for i, t in enumerate(live):
            if i not in ut:
                t["m"] += step
        for j, v in enumerate(r):
            if j not in ur:
                live.append({"p": {x: v}, "m": 0})
        keep = []
        for t in live:
            (done if t["m"] > gap_kill else keep).append(t)
        live = keep
    done += live
    return [t["p"] for t in done
            if max(t["p"]) - min(t["p"]) >= min_span]


def follow_both(mask, x_seed, x_lo, x_hi, step=3, win=16, hist=14):
    """Trace every curve through a seed column, LEFT and RIGHT from it.

    ⚠ THE SEED COLUMN IS WHERE THE CURVES ARE ALL PRESENT AND SEPARATE, and
    on a spectral figure that is nowhere near either end. Figs 5 and 7 draw
    three bands that each exist over part of the abscissa and cross their
    neighbours twice; seeded at an edge, a tracer has one curve and no way to
    acquire the others, and seeded at a crossing it cannot tell them apart.
    Seeding in the middle and walking outwards is the only arrangement that
    starts from a column where the answer is unambiguous.

    Assignment is exclusive, as in `follow`: a run belongs to one track.
    """
    seeds = runs(mask, x_seed)
    right = follow(mask, x_seed, x_hi, seeds, step=step, win=win, hist=hist)
    tracks = [dict(t) for t in right]
    # leftward: the same machinery on a mirrored column index
    flip = mask[:, ::-1]
    w = mask.shape[1]
    xs = w - 1 - x_seed
    left = follow(flip, xs, w - 1 - x_lo, seeds, step=step, win=win,
                  hist=hist)
    for t, lt in zip(tracks, left):
        for q, v in lt.items():
            t.setdefault(w - 1 - q, v)
    return tracks


def resample(pts, fx, fy, grid):
    """Pixel track -> (value, value) pairs on a chosen abscissa grid."""
    if isinstance(pts, dict):
        pts = sorted(pts.items())
    xs = np.array([fx(p[0]) for p in pts])
    ys = np.array([fy(p[1]) for p in pts])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    out = []
    for g in grid:
        if g < xs[0] - 1e-9 or g > xs[-1] + 1e-9:
            continue
        out.append((round(float(g), 4), round(float(np.interp(g, xs, ys)), 4)))
    return tuple(out)


# ---------------------------------------------------------------------------
#  Figure 4 -- the characteristic curves
# ---------------------------------------------------------------------------

def trace_fig4(gray):
    """{'5293': {'r','g','b'}, '5247': {...}} as (log E relative, density)."""
    fig = FIG4
    fx, _rx = _fit(fig["x"], False)
    fy, _ry = _fit(fig["y"], False)
    m = plot_mask(gray, fig["box"], inset=(22, 28))
    solid, dash = split_style(m, solid_w=800, drop=(
        (600, 915, 240, 345),        # the 5293 / 5247 legend block
        (790, 995, 1185, 1280),      # the dimensioned 0.40 log E arrow
        (1560, 1950, 280, 1250),     # the B / G / R record letters
    ))
    x0, x1 = fig["box"][0] + 25, fig["box"][1] - 25
    out = {}
    for name, mask, xlim in (("5293", solid, x1), ("5247", dash, x1)):
        tr = stack(mask, x0, xlim, 3, step=4)
        out[name] = {c: [(fx(p[0]), fy(p[1])) for p in t]
                     for c, t in zip("bgr", tr)}
    return out


def straight_gamma(curve, lo, hi):
    """Slope of the straight-line portion, and the worst residual on it."""
    pts = [(x, y) for x, y in curve if lo <= x <= hi]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    a, b = np.polyfit(xs, ys, 1)
    return float(a), float(np.max(np.abs(a * xs + b - ys)))


def _x_at(curve, dv):
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if (y0 - dv) * (y1 - dv) <= 0 and y1 != y0:
            return x0 + (dv - y0) * (x1 - x0) / (y1 - y0)
    return None


def check_speed_offset(fig4):
    """Traced 5247-minus-5293 displacement, per record, on the straight line.

    ⚠ MEASURED WHERE THE CURVES ARE STRAIGHT AND NOWHERE ELSE. A horizontal
    displacement between two curves is only a speed difference where they are
    parallel; taken on the toe it also contains the difference in toe shape,
    which on the blue record of this figure is 0.13 log E of pure curvature.
    """
    out = {}
    for c in "bgr":
        lo = min(y for _x, y in fig4["5293"][c])
        top = min(max(y for _x, y in fig4["5293"][c]),
                  max(y for _x, y in fig4["5247"][c]))
        d0, d1 = lo + 0.55 * (top - lo), lo + 0.85 * (top - lo)
        vals = []
        for dv in np.linspace(d0, d1, 5):
            a = _x_at(fig4["5293"][c], dv)
            b = _x_at(fig4["5247"][c], dv)
            if a is not None and b is not None:
                vals.append(b - a)
        if vals:
            out[c] = float(np.mean(vals))
    return out


def check_equal_contrast(fig4):
    """Traced gamma of each record of each film, over its straight line."""
    out = {}
    for film, lo, hi in (("5293", 1.0, 2.4), ("5247", 1.4, 2.8)):
        out[film] = {c: straight_gamma(fig4[film][c], lo, hi)[0]
                     for c in "bgr"}
    return out


# ---------------------------------------------------------------------------
#  Figure 13 -- granularity against DENSITY, by eliminating the bare abscissa
# ---------------------------------------------------------------------------

def trace_fig13(gray):
    """{'5293': ((D, sigma), ...), '5247': (...)} -- granularity vs density.

    ⚠⚠ THE ABSCISSA OF THIS FIGURE IS UNCALIBRATED AND THE FIGURE IS STILL
    FULLY USABLE, which is worth stating because the obvious reading is that
    it is not. It is captioned "Log E" with no numbers -- so no exposure, no
    speed and no latitude can come off it, exactly as with Hanson & Kisner's
    D-log E plates. But it draws the DENSITY curve in the same frame as the
    two granularity curves, on its own calibrated ordinate, so log E can be
    ELIMINATED between them: at every column the density curve says what
    density that exposure produced and the granularity curve says what
    granularity, and the pair is a point of sigma(D). The uncalibrated axis is
    a parameter, not a loss.

    ⚠ AND sigma(D) IS THE FORM THE ENGINE WANTS. `GrainSpec` holds granularity
    against density because that is what a renderer can use -- it knows the
    density it just computed and needs the noise there. A granularity against
    exposure would have to be pushed through the characteristic curve first,
    which is this same elimination done later and with one more assumption.
    """
    fig = FIG13
    fy, _r1 = _fit(fig["y"], False)
    fg, _r2 = _fit(fig["y2"], True)
    m = plot_mask(gray, fig["box"], inset=(46, 22))
    solid, dash = split_style(m, solid_w=300)
    x0, x1 = fig["box"][0] + 50, fig["box"][1] - 48
    s0 = runs(solid, x0)
    if len(s0) != 2:
        return None
    dens, gran = follow(solid, x0, x1, [s0[1], s0[0]])
    d47 = follow(dash, x0, x1, [runs(dash, x0)[0]])[0]

    def pair(track):
        common = sorted(set(dens) & set(track))
        return [(fy(dens[x]), fg(track[x]) / 1000.0) for x in common]
    return {"5293": pair(gran), "5247": pair(d47),
            "density": [(x, fy(y)) for x, y in sorted(dens.items())]}


# ---------------------------------------------------------------------------
#  Figure 14 -- the MTF
# ---------------------------------------------------------------------------

def trace_fig14(gray):
    """{'5293': ((cycles/mm, response fraction), ...), '5247': (...)}."""
    fig = FIG14
    fx, _rx = _fit(fig["x"], True)
    fy, _ry = _fit(fig["y"], True)
    m = plot_mask(gray, fig["box"], inset=(30, 8))
    solid, dash = split_style(m, solid_w=250, drop=(
        (2350, 2800, 880, 1120),     # the 5247 leader line and its label
        (2150, 2620, 1130, 1360),    # the 5293 leader line and its label
    ))
    x0, x1 = fig["box"][0] + 34, fig["box"][1] - 32
    out = {}
    for name, mask in (("5293", solid), ("5247", dash)):
        pts = []
        for x in range(x0, x1, 3):
            r = runs(mask, x)
            if len(r) != 1:
                continue
            pts.append((fx(x), fy(r[0]) / 100.0))
        out[name] = pts
    return out


def f50(curve):
    """Frequency, cycles/mm, at which a traced MTF falls to 0.50."""
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if (y0 - 0.5) * (y1 - 0.5) <= 0 and y1 != y0:
            return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    return None


# ---------------------------------------------------------------------------
#  Figure 5 -- spectral sensitivity
# ---------------------------------------------------------------------------

def trace_fig5(gray):
    """{'5247': {'r','g','b'}, '5293': {...}} as (nm, log sensitivity).

    ⚠ THE FAMILIES ARE THE OTHER WAY ROUND ON THIS FIGURE, and the legend says
    so: 5247 is the SOLID curve here and 5293 the chain-dotted one, where
    Fig. 4 has 5293 solid. Reading the legend per figure rather than once per
    paper is the whole of the precaution.
    """
    fig = FIG5
    fx, _rx = _fit(fig["x"], False)
    fy, _ry = _fit(fig["y"], False)
    m = plot_mask(gray, fig["box"], inset=(26, 46), erase=(
        (1410, 2100, 1750, 2110),    # the 5247 / 5293 legend block, top right
    ))
    ink = m.copy()

    def px(nm):
        return int(round(686 + (nm - 400.0) * 4.0370))

    # \u26a0 ONE TRACE PER RECORD, AND IT IS BOTH FILMS WHERE THEY COINCIDE.
    # The paper draws 5247 solid and 5293 chain-dotted and states that the two
    # "differ somewhat in both blue and red sensitivities" -- which is to say
    # they are drawn one on top of the other everywhere else, at a separation
    # smaller than the two strokes together. Splitting them by line style
    # recovers the 5293 dashes only on the flanks where they part; what is
    # traced here is therefore the pair, and it is adopted as 5293's with that
    # limitation stated in the provenance rather than hidden by a split that
    # would be arbitrary over most of the abscissa.
    out = {"5293+5247": {}}
    for c, nm0, lo, hi in (("b", 400.0, 352.0, 548.0),
                           ("g", 550.0, 498.0, 618.0),
                           ("r", 650.0, 588.0, 706.0)):
        xs = px(nm0)
        r = runs(ink, xs)
        if not r:
            continue
        t = trace_hump(ink, xs, min(r), px(lo), px(hi))
        out["5293+5247"][c] = sorted((fx(q), fy(v)) for q, v in t.items())
    return out


# ---------------------------------------------------------------------------
#  Figure 7 -- spectral dye density
# ---------------------------------------------------------------------------

def trace_fig7(gray):
    """Normalised dye density-stain against wavelength, upper envelope.

    Returns {'y': ..., 'm': ..., 'c': ...} of (nm, normalised density) for
    5293. ⚠ THE TWO FILMS ARE ALMOST COINCIDENT HERE -- the paper's own words
    are "The curves are similar, except that the cyan dye of 5293 film has its
    peak spectral dye density at a slightly shorter wavelength" -- so the
    dashed 5247 trace is separable only across the cyan flank, and only the
    5293 set is adopted.
    """
    fig = FIG7
    fx, _rx = _fit(fig["x"], False)
    fy, _ry = _fit(fig["y"], False)
    m = plot_mask(gray, fig["box"], inset=(24, 20))
    solid, _dash = split_style(m, solid_w=200)
    x0, x1 = fig["box"][0] + 28, fig["box"][1] - 26
    tracks = follow_both(solid, 2100, x0, x1, win=14)
    tracks = [t for t in tracks if len(t) >= 30]
    named = {}
    for t in tracks:
        pk = max(t.items(), key=lambda kv: -kv[1])[0]
        named[fx(pk)] = sorted((fx(q), fy(v)) for q, v in t.items())
    keys = sorted(named)
    out = {}
    for c, k in zip("ymc", keys):
        out[c] = named[k]
    out["_peaks"] = {c: k for c, k in zip("ymc", keys)}
    return out


# ---------------------------------------------------------------------------
#  Figure 12 -- push-1 against the normal process
# ---------------------------------------------------------------------------

def trace_fig12(gray):
    """{'normal': {'b','g','r'}, 'push1': {...}} as (log E relative, D)."""
    fig = FIG12
    fx, _rx = _fit(fig["x"], False)
    fy, _ry = _fit(fig["y"], False)
    m = plot_mask(gray, fig["box"], inset=(24, 24), erase=(
        (1500, 2450, 1010, 1240),    # the legend block
    ))
    solid, dash = split_style(m, solid_w=700)
    x0, x1 = fig["box"][0] + 28, fig["box"][1] - 26
    out = {}
    for name, mask in (("normal", solid), ("push1", dash)):
        tr = stack(mask, x0, x1, 3, step=4)
        out[name] = {c: [(fx(p[0]), fy(p[1])) for p in t]
                     for c, t in zip("bgr", tr)}
    return out


# ---------------------------------------------------------------------------
#  Figure 15 -- dark fading of the cyan dye
# ---------------------------------------------------------------------------

def trace_fig15(gray):
    """The PREDICTED 24 degC curve, as (years, cyan density from D 1.0).

    ⚠ ONLY THE PREDICTED ROOM-TEMPERATURE CURVE IS ADOPTED, and the four
    measured ones are not, because they are not measurements of storage. The
    solid curves are Arrhenius incubations at 77, 85 and 93 degC on a scale of
    DAYS; they exist to fit the extrapolation and mean nothing as keeping
    times. The dashed curve is that extrapolation, on a scale of YEARS at
    24 degC and 40 % RH, and it is the one quantity here comparable with
    Wilhelm's Table 19.1 figures already in this database.

    ⚠ AND THE FIGURE IS MARKED "PRELIMINARY DATA" IN ITS OWN LEGEND. That word
    is carried into the provenance rather than dropped.
    """
    fig = FIG15
    fy, _ry = _fit(fig["y"], False)
    fx2, _rx = _fit(fig["x2"], True)
    # ⚠ THE SEARCH IS CONFINED TO THE PREDICTED CURVE'S OWN CORNER OF THE
    # FRAME, and that is not a shortcut around the tracer. Everything else in
    # the years half of this figure is lettering -- the ACTUAL / PREDICTED
    # legend with its own sample dashes, the "24 degC" label and the
    # "*PRELIMINARY DATA" footnote -- and a legend's sample dash is, by
    # construction, indistinguishable in shape from the curve it stands for.
    m = plot_mask(gray, fig["box"], inset=(20, 20))
    m[:, :2440] = 0
    m[:380, :] = 0
    m[530:, :] = 0
    _solid, dash = split_style(m, solid_w=400, dash_area=40, dash_w=10,
                               dash_h=60)
    pts = []
    for x in range(2440, min(3010, fig["box"][1] - 22), 3):
        r = runs(dash, x)
        if len(r) == 1:
            pts.append((fx2(x), fy(r[0])))
    return pts


def years_to_ten_percent(fade):
    """Years to a 10 % loss, from the traced 24 degC prediction.

    ⚠ THE SAME LAW `AlgoStorageAge.hpp` ALREADY IMPLEMENTS, so the number this
    returns is directly comparable with the Wilhelm figures beside it:
    f(t) = 1 - 0.9^(t/T) inverted at the traced loss. The paper's curve is
    read at its right-hand end, where the extrapolation is longest and the
    loss largest, because a 4 % loss read at 200 years constrains T far more
    tightly than a 1 % loss read at 20.
    """
    if not fade:
        return None
    t, d = fade[-1]
    # ⚙ THE STARTING DENSITY IS 1.0 BY THE FIGURE'S OWN CONSTRUCTION, not
    # the first traced point. Every curve in Fig. 15 is the decay of a patch
    # read at density 1.0 at t = 0; the leftmost predicted point is already at
    # 15 years and has lost 1 %, so taking it as the origin would understate
    # the loss and overstate the keeping time.
    frac = 1.0 - d / 1.0
    if frac <= 0.0:
        return None
    return float(t * np.log(0.9) / np.log(1.0 - frac))


# ---------------------------------------------------------------------------
#  Transcriptions
# ---------------------------------------------------------------------------

#: Fig. 1, top to bottom. Kodak's own words for each layer.
LAYERS_1982 = (
    "overcoat", "barrier layer",
    "fast yellow", "slow yellow", "yellow filter layer",
    "fast magenta", "slow magenta", "interlayer",
    "fast cyan", "slow cyan",
    "base", "rem-jet backing",
)

#: Table 2, verbatim. lines/mm at the two test-object contrasts.
RESOLVING_1982 = {
    "5293": {"1.6:1": 50.0, "1000:1": 100.0},
    "5247": {"1.6:1": 50.0, "1000:1": 100.0},
}

#: Table 1, verbatim -- (year, type, EI, characteristic).
HISTORY = (
    (1950, "5247", "16 (D)", "Original"),
    (1952, "5248", "25 (T)", "More Blue Speed"),
    (1959, "5250", "50", "Speed Doubled"),
    (1962, "5251", "50", "Finer Grain"),
    (1968, "5254", "100", "Speed Doubled"),
    (1972, "5247", "100", "Finer Grain"),
    (1976, "5247", "100", "Improved Color"),
)

#: Exposure indices the paper states in words, with what it says about each.
EI_STATEMENTS = (
    (250, "the rated exposure index, on the same speed scale that establishes "
          "5247 film as having an exposure index of 100"),
    (400, "two-thirds of a stop under; 'very little loss in quality'"),
    (500, "'might choose either a normal process or a Push-1 process'"),
    (1000, "two stops under; 'the cinematographer would be more satisfied "
           "with a Push-1 process'; excellent quality obtained on night "
           "scenes at this index"),
)

#: Processing, in the paper's own words.
PROCESS_1982 = (
    "Process ECN-2, completely compatible with 5247, 5243 and 5272; no change "
    "in replenishment rate or in any chemical; compatible with both the "
    "ferricyanide and the persulfate bleach. Push-1 is an increase of 40 "
    "seconds to 1 minute over the standard developer time, compensating one "
    "stop of underexposure, and is usually done by slowing the film transport "
    "for the whole process.")


# ---------------------------------------------------------------------------
#  Gate
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    g3 = page_gray(args.root, 3)
    if g3 is None:
        print("[SKIP] kodak_5293_1982.py -- "
              "Kennel_Sehlin_etalNeg5293_1982.pdf not staged")
        return 0
    g7 = page_gray(args.root, 7)
    g8 = page_gray(args.root, 8)
    g9 = page_gray(args.root, 9)

    f4 = trace_fig4(g3)
    off = check_speed_offset(f4)
    gam = check_equal_contrast(f4)
    bad = [c for c, v in off.items() if abs(v - STATED_OFFSET) > OFFSET_TOL]
    dg = {c: abs(gam["5293"][c] - gam["5247"][c]) for c in "bgr"}
    badg = [c for c, v in dg.items() if v > CONTRAST_TOL]

    f5 = trace_fig5(g3)
    g4 = page_gray(args.root, 4)
    f7 = trace_fig7(g4)
    f12 = trace_fig12(g7)
    f13 = trace_fig13(g7)
    f14 = trace_fig14(g8)
    fade = trace_fig15(g9)

    # ⚠ FIG. 12's NORMAL-PROCESS CURVES ARE FIG. 4's 5293 CURVES, drawn a
    # second time five pages later, and the two traces are independent all the
    # way down to their own axis calibrations. Agreement between them tests
    # both calibrations at once and costs nothing, so it is asserted.
    redraw = max(abs(f12["normal"][c][0][1] - f4["5293"][c][0][1])
                 for c in "bgr")
    if redraw > 0.02:
        print("[FAIL] kodak_5293_1982.py -- Figs 4 and 12 draw the same "
              "normally-processed 5293 and the two traces disagree by %.3f "
              "of density at their left end" % redraw)
        return 1

    if bad or badg:
        print("[FAIL] kodak_5293_1982.py -- the traced Fig. 4 does not "
              "reproduce the paper's own statements: offsets %s against a "
              "stated 0.40 log E, gamma differences %s"
              % ({c: round(v, 3) for c, v in off.items()},
                 {c: round(v, 3) for c, v in dg.items()}))
        return 1

    s93 = f13["5293"] if f13 else []
    fifty = f50(f14["5293"]) if f14 and f14["5293"] else None
    tten = years_to_ten_percent(fade)
    pk = f7.get("_peaks", {})
    print("[OK] kodak_5293_1982.py -- Kennel et al. 1982, SMPTE J. 91(10) "
          "922-930, nine raster pages with no text layer, traced: Fig.4 "
          "reproduces Kodak's own stated 0.40 log E speed gain over 5247 to "
          "%.3f and its equal-contrast claim to %.3f of gamma; Fig.12's "
          "normal process redraws Fig.4's 5293 to %.3f of density and gives "
          "push-1 its own three records; Fig.13 %d sigma(D) points by "
          "eliminating a BARE abscissa against the density curve in the same "
          "frame, RMS %.1f at D 1.0 falling to %.1f; Fig.14 MTF 50%% at %.0f "
          "cycles/mm beside a printed resolving power of 50 and 100 "
          "lines/mm; Fig.5 three spectral records %d points; Fig.7 dye peaks "
          "Y %.0f / M %.0f / C %.0f nm; Fig.15 cyan dye predicts %.0f years "
          "to a 10%% loss at 24 degC and 40%% RH"
          % (max(abs(v - STATED_OFFSET) for v in off.values()),
             max(dg.values()), redraw, len(s93),
             1000.0 * np.interp(1.0, [d for d, _s in s93],
                                [s for _d, s in s93]) if s93 else 0.0,
             1000.0 * s93[-1][1] if s93 else 0.0,
             fifty or 0.0,
             sum(len(v) for v in f5["5293+5247"].values()),
             pk.get("y", 0.0), pk.get("m", 0.0), pk.get("c", 0.0),
             tten or 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
