"""Spectral dye density from the FUJI and KONICA datasheet house styles.

`dye_density.py` reads the Kodak H-1 layout and only that layout. It anchors on
a ROTATED y-axis caption, then takes the nearest plot frame to the right of it.
Pointed at these sheets it fails four different ways, and the failures are worth
naming because each one is a property of the sheet rather than a tolerance:

  * FUJI PROVIA 100F, SENSIA 100, VELVIA 50 -- the panel is a RASTER PNG. There
    is no vector path to extract and no text layer inside the plot, so the axis
    has to be calibrated from the drawn gridlines.
  * FUJI PROVIA 400X -- vector, but all three dye traces live in ONE path of 19
    bezier segments as three disconnected subpaths. `dye_density.extract`
    resamples a path as a whole and returns one curve for the three.
  * KONICA CHROME CENTURIA 100, CHROME R100 -- vector, but no rotated caption at
    all ("Density(D)" is drawn horizontally), so nothing anchors the search.
  * KODAK VISION3 200T 5213 -- Kodak's own layout, and it still failed: the
    curve bounding boxes on that page satisfy `frames()`, and three of them sit
    a few hundredths of a point closer to the caption than the real plot frame
    does. `pick()` takes the nearest and gets a curve. THE FIX IS NOT A NEARER
    FRAME, IT IS A FRAME THAT HAS TICK LABELS AGAINST IT -- see `pick_by_ticks`.
    That sheet is handled here rather than in `dye_density.py` only so the
    eleven sheets already asserted there cannot be disturbed; the extraction
    itself is `dye_density`'s, called through.

WHAT THIS MODULE FOUND THAT THE NUMBERS ALONE WOULD NOT HAVE
-------------------------------------------------------------
⚠ TWO OF THE SIX PANELS ARE REUSED DRAWINGS, AND BOTH REUSES WOULD OTHERWISE
HAVE ENTERED THE DATABASE AS INDEPENDENT MEASUREMENTS.

  * PROVIA 100F p6 and SENSIA 100 p5 are the SAME curve set. They are different
    raster images (972x734 against 938x853), on different sheets, and one is
    colour-coded while the other is black -- so they are traced here by two
    unrelated methods, an ink-mask centroid and a slope-predictive tracker.
    Those two methods reproduce each other to rms 0.004-0.010 D, max 0.028.
    ⚠ That agreement is doing double duty: it is the artwork finding AND it is
    the only independent check this module has on its own raster tracer.
  * CHROME CENTURIA 100 p3 and CHROME R100 p3 share their MAGENTA and CYAN
    exactly -- rms 0.00008 and 0.00006 D, which is float noise, not similarity.
    The YELLOW was redrawn (rms 0.017, max 0.049). So the two sheets carry ONE
    magenta and ONE cyan measurement between them and two yellows.

Both pairs are adopted onto both profiles, because the sheets do publish the
data for both products, but the shared origin is written into every `source`
string and asserted here, so a later re-trace cannot quietly turn one
measurement into two. Method rule 18 in spirit: a shared drawing is one sample.

NORMALISATION -- all six panels are family B, "peak_1.0"
--------------------------------------------------------
Every one of them draws three dyes each scaled to unit peak and NO neutral, so
the `Neutral = C+M+Y` identity that validates the Kodak family-A sets cannot be
used. What is checked instead, on every sheet:

  * each traced peak lands within its physical band (yellow absorbs blue,
    magenta green, cyan red) -- `BANDS`;
  * each traced peak is 1.000 within `PEAK_TOL`, because the sheet asserts it;
  * on the raster sheets, INDEPENDENT PASSES SEEDED AT DIFFERENT PLACES MUST
    AGREE where they overlap. This is the check that matters: a tracker that
    swaps identity at a crossing still returns three smooth curves, and nothing
    about the shape gives it away. Disagreement is reported per sheet in
    `EXPECTED` and the largest anywhere is 0.017 D, at the 590 nm cyan/magenta
    crossing on VELVIA where both passes are coasting by construction.

KNOWN LIMIT, stated rather than discovered later
------------------------------------------------
⚠ VELVIA 50's YELLOW REACHES THE DRAWN AXIS AT ABOUT 597 nm and is thereafter
indistinguishable from it: at 595 nm the run is 0.008 D and 6 px thick, at 605
there is no separate run at all. The trace is therefore measured 400-600 and
floored at 0.000 from 610 nm, with `ZERO_FLOOR_D` as the honest error bar (the
line width, in density). The same convention was used for the 1953 Eastmancolor
5382 tails. It is NOT applied to any other sheet here and the guard checks that.

Run:  python fuji_konica_dye.py [--assert] [--sheet provia100f]
Needs numpy + PyMuPDF. --assert exits non-zero if any extraction moves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pymupdf

import dashtrace as dt
import dye_density as dd

GRID = np.arange(400, 701, 10, dtype=float)

#: Physical absorption bands. A peak outside its band is a mis-assignment, not
#: an unusual dye -- no colour film puts its magenta peak in the red.
BANDS = {"yellow": (430, 470), "magenta": (530, 570), "cyan": (640, 700)}

#: The sheets state that each dye is scaled to unit peak, so this is testing
#: the extraction against the sheet's own claim, not choosing a convention.
PEAK_TOL = 0.010

#: VELVIA yellow beyond 600 nm: the sheet draws it ON the axis. This is the
#: printed line width expressed in density, i.e. the error bar on "zero".
ZERO_FLOOR_D = 0.008
#: The last column on which VELVIA's yellow is a separate ink run reads 0.0080 D
#: and is ONE pixel thick, at 598 nm; at 602 nm the panel has two runs, not
#: three. So 600 nm is where the measurement stops, and the fact that the final
#: measured value equals the line width is the evidence that it stops by
#: reaching the axis rather than by the tracker losing it.
VELVIA_FLOOR_FROM_NM = 600.0

#: How many pixels thicker than the BARE rule a gridline band has to be, in a
#: single column, before that column is read as "a curve is lying on the rule
#: here". The reference is the MEDIAN band thickness over the whole width --
#: i.e. the rule measured where nothing else touches it -- and not the detected
#: group height, which includes the pad and is therefore already larger than
#: the ink. Measured on SENSIA's D = 1.0 rule: bare 4 px, 6-8 px where the
#: cyan peak lies on it, so 2 separates them and 3 does not.
BAND_EXCESS_PX = 2

#: Render resolution for the raster panels. 600 dpi puts 4.8-4.9 render pixels
#: on one nanometre, so the 10 nm output grid is oversampled ~48x and the
#: quantisation of the trace is far below the line width.
RASTER_DPI = 600


# ---------------------------------------------------------------- geometry --

def render(pdf: Path, page: int, clip, dpi: int = RASTER_DPI) -> np.ndarray:
    pg = pymupdf.open(pdf)[page - 1]
    pix = pg.get_pixmap(dpi=dpi, clip=pymupdf.Rect(*clip))
    return (np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width, 3).astype(int))


def line_groups(prof, frac: float, grow: float):
    """Index groups of a 1-D coverage profile that read as a drawn rule.

    `frac` finds the core of a gridline; `grow` then walks outward while the
    coverage is still well above what a curve contributes, which is what picks
    up the antialiased shoulder. Leaving the shoulder behind is not cosmetic:
    on VELVIA it left a one-pixel residue that the tracker read as a fourth
    curve sitting exactly on D = 0.50 in every column.
    """
    idx = [i for i, v in enumerate(prof) if v > frac]
    grp = []
    for i in idx:
        if grp and i - grp[-1][-1] <= 2:
            grp[-1].append(i)
        else:
            grp.append([i])
    out = []
    for g in grp:
        a, b = g[0], g[-1]
        while a - 1 >= 0 and prof[a - 1] > grow:
            a -= 1
        while b + 1 < len(prof) and prof[b + 1] > grow:
            b += 1
        out.append(list(range(a, b + 1)))
    return out


def fit_axis(vals, px):
    """(slope, intercept, worst residual) of a least-squares tick fit.

    Same reason as `dye_density._fit_axis`: two points always define a line and
    nothing checks it. Here the residual is also the honest statement of how
    non-linear the SCAN is -- on PROVIA 100F the four wavelength gridlines are
    not evenly spaced on the page (503 / 484 / 495 render pixels per 100 nm),
    which is scanner geometry, and the fit reports it as 1.30 nm rather than
    hiding it in a two-point span.
    """
    v = np.asarray(vals, float)
    p = np.asarray(px, float)
    A = np.vstack([v, np.ones(len(v))]).T
    (m, c), *_ = np.linalg.lstsq(A, p, rcond=None)
    return float(m), float(c), float(np.abs(m * v + c - p).max())


def prep_raster(pdf, page, clip, thr=0.55, glyph=70, frac=0.45, grow=0.12,
                pad=2, dpi=RASTER_DPI):
    """(img, gray, ink, vertical gridline centres, horizontal gridline centres).

    Two things here are not obvious and both were forced by a measured failure:

    ⚠ TEXT IS REMOVED BY CONNECTED COMPONENT, BEFORE the gridlines are cut.
    Cutting the gridlines first severs every curve that crosses one, after
    which no component is big enough to tell a curve from the word "Magenta".

    ⚠ HORIZONTAL GRIDLINES ARE REMOVED COLUMN-CONDITIONALLY, ON TWO TESTS. A
    pixel in a gridline band survives if the same column carries ink just above
    or just below the band -- a curve crossing it -- OR if the band itself is
    thicker in that column than the bare rule, which is a curve lying ON it.
    Deleting the whole band instead destroys any curve running TANGENT to a
    gridline, and that is exactly what every peak-normalised dye peak does at
    D = 1.0. Measured, with the crossing test alone: SENSIA's cyan peak sits on
    the D = 1.0 rule from 653 to 663 nm, the band was cut, and the track died
    at 650 nm -- six of the thirty-one output points lost, at the peak.
    """
    img = render(pdf, page, clip, dpi)
    H, W, _ = img.shape
    gray = img.mean(2) / 255.0
    ink = gray < thr
    dark = img.sum(2) < 200
    vg = line_groups(dark.sum(0) / H, frac, grow)
    hg = line_groups(dark.sum(1) / W, frac, grow)
    lab, info = dt._components(ink)
    for n, (w, h, _cnt) in info.items():
        if w < glyph and h < glyph:
            ink[lab == n] = False
    for g in vg:
        for c in g:
            ink[:, c] = False
    for g in hg:
        a, b = max(g[0] - pad, 0), min(g[-1] + pad, H - 1)
        above = ink[max(a - 3, 0):a, :].any(0) if a > 0 else np.zeros(W, bool)
        below = ink[b + 1:b + 4, :].any(0) if b + 1 < H else np.zeros(W, bool)
        cnt = ink[a:b + 1, :].sum(0)
        thick = cnt >= (np.median(cnt) + BAND_EXCESS_PX)
        ink[a:b + 1, ~(above | below | thick)] = False
    return (img, gray, ink,
            [float(np.mean(g)) for g in vg], [float(np.mean(g)) for g in hg])


# ------------------------------------------------------------ raster: ink ---

#: PROVIA 100F prints its three dyes in their own process colours. Sampled off
#: the page and quantised: yellow (240,240,20), magenta (232,0,140),
#: cyan (0,175,232) against a (231,231,231) ground. The separation is total --
#: the four populations do not overlap at any threshold below 200 in L1 -- so
#: this sheet needs no tracker at all and provides the control for the one the
#: other two sheets do need.
INK_RGB = {"yellow": (240, 240, 20), "magenta": (232, 0, 140),
           "cyan": (0, 175, 232)}
INK_L1_TOL = 150


def trace_by_ink(img, mx, cx, my, cy, x0, x1):
    out = {}
    for name, rgb in INK_RGB.items():
        sel = np.abs(img - np.array(rgb)).sum(2) < INK_L1_TOL
        vals = []
        for lam in GRID:
            col = int(round(mx * lam + cx))
            acc = []
            for dx in (-1, 0, 1):
                c = col + dx
                if c < x0 or c > x1:
                    continue
                rows = np.nonzero(sel[:, c])[0]
                if rows.size:
                    acc.append(float(np.median(rows)))
            if not acc:
                raise RuntimeError(f"{name}: no ink at {lam:.0f} nm")
            vals.append((float(np.mean(acc)) - cy) / my)
        out[name] = np.array(vals)
    return out


# ------------------------------------------------- raster: black tracking ---

#: Tracker settings. `slope_cap` is the one that had to move: the default 2.5
#: px/column is tuned for the Kodak granularity plots and a peak-normalised
#: yellow edge rises at about 1.9 px/column at 600 dpi, which sounds safe until
#: the seed column sits inside a stripped gridline and the track has to bridge
#: it from a zero-slope history. At 8.0 every sheet here gets through and no
#: pass disagrees with another by more than 0.017 D.
TRACK = dict(tol0=6.0, tol_grow=0.8, max_bridge=34, hist=20,
             slope_cap=8.0, merge_px=14.0)

#: ⚠ THE KODAK RASTER PANELS USE THE SAME TRACK SETTINGS, AND A LOOSER SET WAS
#: TRIED AND REJECTED BY MEASUREMENT. Those pages embed a 693x765 PNG rendered
#: here at 600 dpi, so the panel arrives 2x upsampled with every column
#: duplicated and a curve can move 3.6 px between identical pairs of columns.
#: Seeding on a STEEP part of a curve then fails against a zero-slope seed
#: history -- the neutral and yellow tracks die after 1 and 3 columns from a
#: 410 nm seed. Loosening tol0 to 12 fixes that seed and BREAKS the good one:
#: 5203's cyan then wanders off its own peak, coming back 0.297 short of the
#: unit peak the sheet states. The right answer is not a looser tolerance but a
#: seed column where every curve is FLAT -- see KODAK_RASTER.


def trace_black(ink, gray, gray_top, gray_bot, mx, cx, my, cy, passes):
    """Run several seeded passes and merge them, reporting the disagreement.

    `passes` is a list of (seed wavelength, direction, {name: density}).

    ⚠ THE MERGE IS A MEAN AND THE SPREAD IS RETURNED, deliberately. Where two
    passes both reached a column they should agree exactly, and on five of the
    nine track/pass combinations here they agree to 0.0000 D. Averaging a
    disagreement away silently is how a swapped identity survives; reporting it
    is what makes `EXPECTED` able to fail.
    """
    x_lo = int(round(mx * 392 + cx))
    x_hi = int(round(mx * 702 + cx))
    acc, spread = {}, {}
    for lam0, direction, seeds in passes:
        sx = int(round(mx * lam0 + cx))
        tr = dt.trace_predictive(
            ink, gray, (x_lo, x_hi), gray_top, gray_bot, sx,
            {k: v * my + cy for k, v in seeds.items()},
            direction=direction, **TRACK)
        for k, pts in tr.items():
            for x, y in pts.items():
                acc.setdefault(k, {}).setdefault(x, []).append(y)
    out = {}
    for k, cols in acc.items():
        spread[k] = max((max(v) - min(v)) for v in cols.values()) / abs(my)
        xs = np.array(sorted(cols))
        ys = np.array([np.mean(cols[x]) for x in xs])
        lam = (xs - cx) / mx
        val = np.interp(GRID, lam, (ys - cy) / my)
        val[GRID > lam.max()] = np.nan
        val[GRID < lam.min()] = np.nan
        out[k] = val
    return out, spread


# ------------------------------------------------------------ vector paths --

# ------------------------------------------- raster: the Kodak H-1 layout ---

#: The VISION3 50D 5203 and 250D 5207 dye panels are RASTER, and that is the
#: whole reason they were filed as blocked: NotFound.md recorded "their TI-sheet
#: plots are raster" as if raster meant unreadable. Both draw the five-trace
#: family-C set -- Yellow / Magenta / Cyan peak-normalised, plus an as-printed
#: Midscale Neutral and a DASHED Minimum Density -- at ECN-2, D-mins subtracted.
#:
#: ⚠ THESE PANELS DRAW NO INTERIOR GRIDLINES, so the axis cannot be calibrated
#: the way the Fuji ones are. What they do draw is a TICK COMB on each axis, and
#: the frame corners sit on the labelled extremes. `tick_ladder` uses that.
KODAK_X_AXIS = (400.0, 800.0)
KODAK_Y_AXIS = (1.8, -0.2)
KODAK_NY = 11

#: Component geometry for the dash/solid split, at 600 dpi.
#: ⚠ MEASURED, NOT GUESSED. On 5207 the four solid curves fuse into ONE
#: component 1248 x 1010 px because they cross; every other component is 35 px
#: wide or less. The real Minimum-Density dashes are 23-33 px wide and 4-19 px
#: tall; the label glyphs are 20-35 px wide and 20-39 px TALL. Height separates
#: them with a clear gap, and a second width floor drops the speckle.
KODAK_SOLID_MIN_W = 200
KODAK_DASH_MAX_W = 40
KODAK_DASH_MAX_H = 19
KODAK_DASH_MIN_W = 20


def tick_comb(dark, lo, hi, along, thick=10):
    """Centres of the tick stubs along one axis."""
    prof = dark[lo:hi, :].sum(0) if along == "x" else dark[:, lo:hi].sum(1)
    idx = [i for i, v in enumerate(prof) if v >= thick]
    out, cur = [], []
    for i in idx:
        if cur and i - cur[-1] <= 3:
            cur.append(i)
        else:
            if cur:
                out.append(float(np.mean(cur)))
            cur = [i]
    if cur:
        out.append(float(np.mean(cur)))
    return out


def tick_ladder(found, lo_px, hi_px, v_lo, v_hi, n, tol):
    """Match a detected tick comb to the asserted uniform ladder, then fit.

    The ladder is ASSERTED -- n ticks, evenly spaced, extremes at the frame --
    and the fit residual is what tests the assertion. Matching first is what
    makes it robust: on 5203 the comb comes back with twelve entries where the
    axis has eleven, because one label glyph reaches into the tick band, and an
    unmatched least-squares fit over twelve mis-assigns every value above it.
    Measured consequence before the match was added: the y fit degraded to three
    surviving ticks at 5.9 px residual and put the yellow peak at 0.912 on a
    sheet whose own note says the dyes are peak-normalised.
    """
    exp = [lo_px + (hi_px - lo_px) * k / (n - 1) for k in range(n)]
    val = [v_lo + (v_hi - v_lo) * k / (n - 1) for k in range(n)]
    px, vv, miss = [], [], []
    for e, v in zip(exp, val):
        c = min(found, key=lambda f: abs(f - e)) if found else None
        if c is None or abs(c - e) > tol:
            miss.append(v)
            continue
        px.append(c)
        vv.append(v)
    A = np.vstack([np.array(vv), np.ones(len(vv))]).T
    m, c = np.linalg.lstsq(A, np.array(px), rcond=None)[0]
    res = float(np.abs(m * np.array(vv) + c - np.array(px)).max())
    return float(m), float(c), res, len(vv), miss


def _bboxes(lab):
    n = int(lab.max())
    ys, xs = np.nonzero(lab)
    l = lab[ys, xs]
    x0 = np.full(n + 1, 1 << 30)
    x1 = np.full(n + 1, -1)
    y0 = np.full(n + 1, 1 << 30)
    y1 = np.full(n + 1, -1)
    np.minimum.at(x0, l, xs)
    np.maximum.at(x1, l, xs)
    np.minimum.at(y0, l, ys)
    np.maximum.at(y1, l, ys)
    return x0, x1, y0, y1


def prep_kodak_raster(pdf, page, clip, thr=0.55, dpi=RASTER_DPI):
    """(gray, solid ink, dashed ink, frame, x fit, y fit) for a 5203/5207 panel.

    ⚠ THE TICK STUBS ARE REMOVED BY WHERE THEY ARE, NOT BY A MARGIN. They point
    inward from the frame and run to 33 px on 5203's major ticks; a margin wide
    enough to clear them also throws away 400-413 nm of every curve, which is
    exactly the band the cyan tail and the yellow rise live in.
    """
    img = render(pdf, page, clip, dpi)
    H, W, _ = img.shape
    gray = img.mean(2) / 255.0
    ink = gray < thr
    dark = img.sum(2) < 250
    vg = line_groups(dark.sum(0) / H, 0.45, 0.12)
    hg = line_groups(dark.sum(1) / W, 0.45, 0.12)
    L, R = int(np.mean(vg[0])), int(np.mean(vg[-1]))
    T, B = int(np.mean(hg[0])), int(np.mean(hg[-1]))
    xt = tick_comb(dark, B - 22, B - 6, "x")
    yt = tick_comb(dark, L + 6, L + 22, "y")
    fx = tick_ladder(xt, L, R, *KODAK_X_AXIS, len(xt), 6.0)
    fy = tick_ladder(yt, T, B, *KODAK_Y_AXIS, KODAK_NY, 8.0)
    sub = np.zeros_like(ink)
    sub[T + 6:B - 6, L + 6:R - 8] = ink[T + 6:B - 6, L + 6:R - 8]
    lab, info = dt._components(sub)
    X0, X1, Y0, Y1 = _bboxes(lab)
    solid = np.zeros_like(sub)
    dash = np.zeros_like(sub)
    for n, (w, h, _c) in info.items():
        if X1[n] <= L + 36 and h <= 12 and w <= 40:
            continue                                      # y-axis tick stub
        if Y0[n] >= B - 46 and w <= 12 and h <= 46:
            continue                                      # x-axis tick stub
        if w >= KODAK_SOLID_MIN_W:
            solid |= (lab == n)
        elif KODAK_DASH_MIN_W <= w <= KODAK_DASH_MAX_W and h <= KODAK_DASH_MAX_H:
            dash |= (lab == n)
    return gray, solid, dash, (L, R, T, B), fx, fy


def dash_chain(dash, mx, cx, my, cy):
    """Per-column centroids of every surviving dash segment, in (nm, D).

    ⚠ NOT TRACKED, ASSEMBLED. `trace_predictive` cannot follow this curve: the
    dashes that cross a solid curve fuse into it and are lost to the component
    split, which leaves gaps of 37 nm on 5207 and more on 5203, right where the
    D-min plunges. Chaining the surviving segments states what is measured and
    where the holes are, instead of bridging a hole on a fitted slope.
    """
    lab, info = dt._components(dash)
    pts = {}
    for n, (w, _h, _c) in info.items():
        if w < KODAK_DASH_MIN_W:
            continue
        ys, xs = np.nonzero(lab == n)
        for xx in np.unique(xs):
            pts[float((xx - cx) / mx)] = float((ys[xs == xx].mean() - cy) / my)
    if not pts:
        return np.array([]), np.array([])
    k = np.array(sorted(pts))
    return k, np.array([pts[v] for v in k])


def pick_by_ticks(pg, min_ticks=3):
    """Plot frames that actually carry tick labels, largest first.

    `dye_density.pick` takes the frame nearest the axis caption. On the VISION3
    200T sheet the three dye traces each have a bounding box that passes
    `frames()` and starts 0.01-0.03 pt closer to the caption than the real
    frame, so "nearest" returns a curve and the extraction reports "only 0 x
    ticks against the frame". Requiring the ticks in the first place is both
    simpler and the actual definition of a plot frame.
    """
    out = []
    seen = set()
    for p in pg.get_drawings():
        r = p["rect"]
        if not (80 < r.width < 560 and 50 < r.height < 420):
            continue
        key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
        if key in seen:
            continue
        seen.add(key)
        xs, ys = dd.ticks(pg, r)
        if len(xs) >= min_ticks and len(ys) >= min_ticks:
            out.append((r.width * r.height, r, xs, ys))
    out.sort(key=lambda t: -t[0])
    return [(r, xs, ys) for _a, r, xs, ys in out]


def calibration(xs, ys):
    fx = dd._fit_axis(xs)
    fy = dd._fit_axis(ys)
    vx, vy = sorted(xs), sorted(ys)
    cal = (fx[0] * vx[0] + fx[1], vx[0], fx[0] * vx[-1] + fx[1], vx[-1],
           fy[0] * vy[0] + fy[1], vy[0], fy[0] * vy[-1] + fy[1], vy[-1])
    return cal, fx[2], fy[2]


def split_subpaths(items):
    """Break one drawing's item list where the pen lifts.

    PROVIA 400X draws all three dyes as one path: segment 6 starts at
    (341.6, 542.7) while segment 5 ended at (528.8, 547.5). Nothing but that
    discontinuity separates the curves.
    """
    subs, cur, last = [], [], None
    for it in items:
        p0 = it[1]
        if not hasattr(p0, "x"):        # a 're' item: a rectangle, not a stroke
            continue
        if last is not None and (abs(p0.x - last.x) > 0.5
                                 or abs(p0.y - last.y) > 0.5):
            subs.append(cur)
            cur = []
        cur.append(it)
        last = it[-1]
    if cur:
        subs.append(cur)
    return subs


def curves_in_frame(pg, fr, cal, min_seg=4, min_width_frac=0.5, split=False):
    """Resampled curves whose path lies inside `fr` and spans it.

    ⚠ The width test is what rejects the fourth "curve" on both Konica sheets:
    a path peaking at 1.398 D at exactly 600 nm, made of straight segments,
    which is a leader line and not a measurement. It is also caught downstream
    by the unit-peak test, and being caught twice is the point.
    """
    out = []
    for g in pg.get_drawings():
        r = g["rect"]
        n = sum(1 for it in g["items"] if it[0] in ("l", "c"))
        if n < min_seg or r.width < min_width_frac * fr.width:
            continue
        if not (r.x0 >= fr.x0 - 6 and r.x1 <= fr.x1 + 6
                and r.y0 >= fr.y0 - 6 and r.y1 <= fr.y1 + 6):
            continue
        groups = split_subpaths(g["items"]) if split else [g["items"]]
        for sub in groups:
            if sum(1 for it in sub if it[0] in ("l", "c")) < min_seg:
                continue
            y = dd.resample(dd.flatten(sub, 64), cal, GRID)
            if np.isfinite(y).all():
                out.append(y)
    return out


def assign_by_band(curves):
    """{dye: curve} for the three curves that peak in band AND at unit height."""
    got = {}
    for y in curves:
        pk = float(GRID[int(np.nanargmax(y))])
        mv = float(np.nanmax(y))
        if abs(mv - 1.0) > 0.10:
            continue
        for name, (a, b) in BANDS.items():
            if a <= pk <= b:
                got[name] = y
    return got


# ------------------------------------------------------------------ sheets --

#: tag -> (pdf relative to PDF/PROFILES, page, profile name, kind)
SHEETS = {
    "provia100f": ("FUJI/provia_100f_datasheet.pdf", 6,
                   "FUJI_PROVIA_100F", "raster_ink"),
    "sensia100": ("FUJI/sensia_100_datasheet.pdf", 5,
                  "FUJI_SENSIA_100", "raster_black"),
    "velvia50": ("FUJI/velvia_50_datasheet.pdf", 8,
                 "FUJI_VELVIA_50", "raster_black"),
    "provia400x": ("FUJI/Provia_400X_PIB_1007.pdf", 7,
                   "FUJI_PROVIA_400X", "vector_split"),
    "chrocen100": ("KONICA/chrocen100.pdf", 3,
                   "KONICA_CHROME_CENTURIA_100", "vector_multi"),
    "r100": ("KONICA/R100.pdf", 3,
             "KONICA_CHROME_R100", "vector_multi"),
    "5213": ("KODAK/VISION3-200T-Color-Negative-Film-7213-TECHNICAL-DATA.pdf", 5,
             "KODAK_VISION3_200T_5213", "kodak_h1"),
    # -- added 2026-09-04c: the two RASTER VISION3 panels ---------------------
    "5203": ("KODAK/KODAK-VISION3-50D-5203-7203-technical-information.pdf", 4,
             "KODAK_VISION3_50D_5203", "kodak_raster"),
    "5207": ("KODAK/KODAK-VISION3-250D-5207-7207-technical-information.pdf", 4,
             "KODAK_VISION3_250D_5207", "kodak_raster"),
}

#: The 5203 / 5207 panels: clip, the authoritative seed column, and a SECOND
#: seed column used only as a cross-check.
#:
#: ⚠ ONE SEED COLUMN CARRIES ALL FOUR SOLID CURVES AND IT HAS TO BE 450 nm.
#: Two other columns were tried and both mis-assign, in ways worth recording
#: because neither shows up as a rough-looking curve:
#:   * seeded at 690 nm, 5203's CYAN track walks onto the MAGENTA below 470 nm
#:     -- the two cross at about 478 and are 0.04 D apart on either side -- and
#:     returns a "cyan" whose 400 nm value is 0.000 instead of 0.228;
#:   * seeded at 540 nm, the NEUTRAL track leaves by 0.17 D on both sheets.
#: Seeded at 450 nm all four are separated and every peak lands on 1.000, which
#: is the sheet's own stated normalisation and was not fitted.
#: ⚠ The 410 nm cross-check is not decoration: on 5203 it reproduces all four
#: curves to 0.0000 D from a different column, which is the only independent
#: statement available that the 450 nm assignment is the right one.
KODAK_RASTER = {
    "5203": dict(
        clip=(343, 91, 535, 304),
        seed=(450, {"neutral": 1.542, "yellow": 1.002,
                    "cyan": 0.069, "magenta": 0.001}),
        check=(410, {"neutral": 1.210, "yellow": 0.609,
                     "cyan": 0.201, "magenta": 0.000})),
    "5207": dict(
        clip=(341, 91, 536, 304),
        seed=(450, {"neutral": 1.557, "yellow": 1.000,
                    "cyan": 0.055, "magenta": -0.013}),
        check=(650, {"cyan": 0.876, "neutral": 0.760,
                     "magenta": 0.085, "yellow": -0.004})),
}

#: Panel clip in PDF points and the seeded passes, for the raster sheets only.
#: The seed densities are READ OFF the column runs at the seed wavelength, not
#: guessed; each one is reproduced by `--assert` through the peak test.
RASTER = {
    "provia100f": dict(clip=(300, 430, 545, 620)),
    "sensia100": dict(
        clip=(316, 395, 552, 576),
        # The leftward pass from 401 nm exists only to reach the 400 nm grid
        # point: the 400 nm rule is stripped with the other gridlines, so a
        # pass seeded at 401 has nothing to its left and the first output
        # sample would be an extrapolation.
        passes=[(401, +1, {"yellow": 0.566, "cyan": 0.248, "magenta": 0.165}),
                (401, -1, {"yellow": 0.566, "cyan": 0.248, "magenta": 0.165}),
                (560, -1, {"yellow": 0.040, "cyan": 0.310, "magenta": 0.898}),
                (560, +1, {"yellow": 0.040, "cyan": 0.310, "magenta": 0.898})]),
    "velvia50": dict(
        clip=(306, 400, 538, 584),
        # ⚠ FIVE passes, and the last two exist for one reason: cyan and magenta
        # cross at 590 nm and the tracker COASTS through a crossing by design,
        # so a pass seeded left of it cannot say which branch is which on the
        # far side. Seeding at 620 nm, where the two are 0.65 D apart, settles
        # the identity by measurement instead of by slope extrapolation.
        passes=[(392, +1, {"yellow": 0.452, "cyan": 0.211, "magenta": 0.122}),
                (560, -1, {"yellow": 0.059, "cyan": 0.289, "magenta": 0.937}),
                (560, +1, {"yellow": 0.059, "cyan": 0.289, "magenta": 0.937}),
                (620, +1, {"cyan": 0.811, "magenta": 0.162}),
                (620, -1, {"cyan": 0.811, "magenta": 0.162})]),
}

#: Vector frames, chosen once and pinned so a PyMuPDF ordering change cannot
#: silently move the extraction to the neighbouring MTF plot.
VECTOR_FRAME = {
    "provia400x": (333.3, 400.1, 534.5, 547.5),
    "chrocen100": (225.3, 215.9, 381.2, 348.7),
    "r100": (209.7, 213.1, 365.6, 345.8),
    "5213": (77.899, 171.907, 262.194, 356.343),
}

#: tag -> (peak nm per dye, worst |peak - 1.0|, worst inter-pass disagreement D)
#: Recorded 2026-09-04. --assert fails if any of the three moves.
EXPECTED = {
    "provia100f": ((440, 540, 660), 0.0075, 0.0000),
    "sensia100": ((440, 540, 660), 0.0023, 0.0160),
    "velvia50": ((450, 540, 660), 0.0032, 0.0173),
    "provia400x": ((450, 550, 660), 0.0070, 0.0000),
    "chrocen100": ((450, 550, 660), 0.0058, 0.0000),
    "r100": ((450, 550, 660), 0.0057, 0.0000),
    "5213": ((440, 540, 680), 0.0112, 0.0000),
    # ⚠ the "interpass" column for these two is the 410 nm cross-check against
    # the 450 nm extraction, over all four solid curves.
    "5203": ((450, 540, 690), 0.0017, 0.0000),
    "5207": ((450, 540, 680), 0.0004, 0.0195),
}

#: 5203 / 5207: the family-C identity `Neutral - Dmin = k(C+M+Y)` evaluated over
#: the samples where the DASHED D-min survives, and the span it survives over.
#:
#: ⚠ THIS IS A PARTIAL TEST AND IS REPORTED AS ONE. It cannot cover the whole
#: grid because the D-min dashes that cross a solid curve fuse into it, so the
#: chain starts at 439 nm on 5207 and 429 nm on 5203. Over what is left the
#: coefficient spread is 0.151 and 0.145 -- against the 0.15 that
#: `dye_matrix_from_spectra.NEUTRAL_SPREAD_MAX` allows a full panel, and against
#: 5218's 0.307 and 5293's 0.215, which that bound rejected. It is corroboration
#: of the dye assignment, not a licence to store a partial trace.
#:
#: ⚠ AND IT EARNED ITS KEEP BY MOVING WHEN A MIS-ASSIGNMENT WAS FIXED. With
#: 5203's cyan taken from the 690 nm seed -- the seed that walks onto the magenta
#: below 470 nm -- this spread read 0.145. Re-seeding at 450 nm, which the peak
#: test independently prefers, halved it to 0.073. A test that only ever agrees
#: is not a test; this one disagreed with the wrong answer.
KODAK_RASTER_IDENTITY = {
    "5203": (0.073, 429.0, 758.0),
    "5207": (0.153, 439.0, 782.0),
}

#: ⚠ THE SHARED-ARTWORK ASSERTIONS. These are the reason this module cannot be
#: replaced by "just trace the six sheets". (tag A, tag B, dye, worst allowed
#: rms). A rms ABOVE the bound means a drawing that was identical has stopped
#: being identical, i.e. one of the two extractions has changed; the guards in
#: verify.py hold the same pairs against the ADOPTED arrays.
SHARED = [
    ("chrocen100", "r100", "magenta", 0.0005),
    ("chrocen100", "r100", "cyan", 0.0005),
    ("provia100f", "sensia100", "cyan", 0.012),
    ("provia100f", "sensia100", "magenta", 0.012),
    ("provia100f", "sensia100", "yellow", 0.012),
]

#: And the negative control: these pairs must NOT be near-identical, or the
#: assignment has collapsed two different products onto one drawing.
DISTINCT = [
    # ⚠ THE TIGHTEST BOUND IN THIS TABLE, AND DELIBERATELY SO: 5203 and 5207
    # are two speeds of one VISION3 generation on one process, so their dye
    # sets genuinely resemble each other. 0.016 is still 3x the Fuji shared
    # DRAWING and 200x the Konica one, which is the distinction being made.
    ("5203", "5207", 0.012),
    ("provia100f", "provia400x", 0.030),
    ("provia100f", "velvia50", 0.030),
    ("chrocen100", "provia100f", 0.030),
]


# --------------------------------------------- the neutral + D-min panels ---

#: THE SIX SHEETS THAT DRAW NO DYES AT ALL, and the split is by PROCESS rather
#: than by manufacturer: every C-41 / CN-16 / CNK-4 colour NEGATIVE sheet in
#: this corpus draws a "typical densities for a midscale neutral subject and
#: D-min" pair, and every reversal sheet draws Y/M/C. Six more instances of the
#: shape mismatch already recorded for FUJI_SUPER_F125_8532 and 5248, and they
#: go into `d_neutral` + `d_dmin` -- `has_neutral_pair`, never `has_data`.
#:
#: ⚠ `d_dmin` ON A MASKED COLOUR NEGATIVE IS THE ORANGE MASK, MEASURED, and on
#: these six it is the first spectral record of the mask on any Fuji or Konica
#: consumer negative. That is also the physics gate: a mask FALLS towards the
#: red, so `dmin(700) < dmin(400)` must hold, and it is what would catch the
#: pair being stored the wrong way round.
NEUTRAL_SHEETS = {
    "pro400h": ("FUJI/pro_400h_datasheet.pdf", 8,
                "FUJICOLOR_PRO_400H", "vector_grid"),
    "superia400": ("FUJI/superia_xtra400_datasheet.pdf", 6,
                   "FUJICOLOR_SUPERIA_XTRA_400", "vector_pair"),
    "csuper1600": ("KONICA/csuper1600.pdf", 2,
                   "KONICA_CENTURIA_SUPER_1600", "vector_pair"),
    "vx100": ("KONICA/VX100Improved.pdf", 3,
              "KONICA_VX_100", "vector_pair"),
    "imp50": ("KONICA/IMP50.pdf", 3,
              "KONICA_IMPRESA_50", "raster_pair"),
    "ultramax800": ("KODAK/E7024-Ultra_Max_800.pdf", 4,
                    "KODAK_ULTRAMAX_800", "vector_pair"),
}

NEUTRAL_FRAME = {
    "superia400": (334.6, 396.3, 539.9, 544.7),
    "csuper1600": (226.1, 215.8, 382.3, 404.0),
    "vx100": (216.3, 214.4, 372.4, 405.7),
    "ultramax800": (80.3, 102.0, 264.7, 286.4),
    "pro400h": (334.7, 374.4, 533.9, 538.7),
}

#: ⚠ PRO 400H PRINTS ITS AXIS NUMBERS AS OUTLINED VECTOR PATHS, not as text, so
#: there is no tick label anywhere in its text layer and `pick_by_ticks` finds
#: no frame on that page at all. The calibration therefore comes from the drawn
#: GRIDLINES, read off the 8-segment rule path inside the frame.
#: ⚠ The frame bottom is NOT the zero line: fitted on these four horizontals it
#: sits at D = 0.023, which is recorded rather than snapped to zero.
PRO400H_GRID = (
    {400.0: 334.9, 500.0: 397.2, 600.0: 459.8, 700.0: 522.7},
    {2.0: 386.0, 1.5: 424.0, 1.0: 462.6, 0.5: 502.2},
)

#: tag -> (neutral mean, dmin mean, dmin at 400, dmin at 700, worst x, worst y)
#: Recorded 2026-09-04c. --assert fails if any extraction moves.
#: ⚠ AND NONE OF THE SIX SHARES A DRAWING WITH ANOTHER, which is a result and
#: not an absence. Both makers in this batch DID reuse artwork on their REVERSAL
#: sheets -- Konica's Chrome pair share a magenta and a cyan to 0.00008 D, Fuji's
#: PROVIA 100F and SENSIA 100 share a whole panel -- so the question had to be
#: asked here too. The closest pair among these six is CENTURIA SUPER 1600
#: against VX 100 at 0.072 D rms on the neutral, roughly 900x the Konica reuse
#: figure. Six sheets, six measurements.
NEUTRAL_DISTINCT_MIN = 0.02

#: ⚠ IMPRESA 50's 400 nm SAMPLE IS A HELD ENDPOINT, not a reading. Its panel is
#: a scan and `konica_raster.panel_ink` blanks each gridline with a 3 px skirt;
#: the 400 nm gridline IS the frame's left edge, so the first column that
#: carries two runs is at 401.6 nm and the 400 nm sample repeats it. Over that
#: 1.6 nm the two curves move about 0.002 and 0.001 D, far under the line width,
#: which is why the hold is stated rather than the pair refused.
IMP50_HELD_TO_NM = 401.6

NEUTRAL_EXPECTED = {
    "pro400h": (1.374, 0.641, 1.426, 0.193, 0.24, 0.0059),
    "superia400": (1.590, 0.529, 0.895, 0.230, 0.56, 0.37),
    "csuper1600": (1.501, 0.697, 1.311, 0.362, 0.13, 0.23),
    "vx100": (1.500, 0.588, 1.429, 0.246, 0.19, 0.31),
    "imp50": (1.024, 0.466, 0.831, 0.237, 0.0, 0.0),
    "ultramax800": (1.257, 0.561, 0.862, 0.263, 0.01, 1.04),
}


def _pair_from_curves(curves, lo=-0.05, hi=3.0):
    """Pick the neutral and the D-min out of the paths found inside a frame.

    The neutral is the HIGHER of the two everywhere -- it is a midscale subject
    read through the same mask the D-min is -- so mean density names them and no
    label or stroke style is needed. That matters on ULTRAMAX 800, the one sheet
    of the six where the neutral is DASHED and the D-min solid: a style rule
    would have to be inverted for that sheet alone, and a level rule does not.
    """
    keep = [y for y in curves
            if np.isfinite(y).all() and lo <= float(y.min())
            and float(y.max()) <= hi]
    if len(keep) != 2:
        return None, f"{len(keep)} in-range curves inside the frame"
    keep.sort(key=lambda y: -float(np.mean(y)))
    return (keep[0], keep[1]), None


def extract_pair(root: Path, tag: str):
    rel, pgno, prof, kind = NEUTRAL_SHEETS[tag]
    pdf = root / "PDF" / "PROFILES" / rel
    if not pdf.is_file():
        return None, f"source not present: {rel}"

    if kind == "raster_pair":
        # IMPRESA 50 is one 2008x1184 scan and `konica_raster` already maps its
        # frame and gridlines; this reuses that geometry rather than repeating
        # it, so the two readers cannot drift apart.
        import konica_raster as kr
        p = kr.PANELS["imp50_dye"]
        x0, x1, y0, y1 = p["frame"]
        _g, sub, fx, fy = kr.panel_ink(root, "imp50_dye",
                                       masks=[(300, 900, 180, 300)])
        lo, hi = {}, {}
        for x in range(x0 + 1, x1 - 5):
            runs = dt.column_runs(sub, int(x), y0, y1)
            if len(runs) == 2:
                hi[float(fx(x))] = float(fy(min(runs)))
                lo[float(fx(x))] = float(fy(max(runs)))
        if not lo:
            return None, "no column carries exactly two runs"
        ks = np.array(sorted(lo))
        neu = np.interp(GRID, ks, [hi[k] for k in ks])
        dmn = np.interp(GRID, ks, [lo[k] for k in ks])
        span = (float(ks.min()), float(ks.max()))
        res = (0.0, 0.0)
    else:
        pg = pymupdf.open(pdf)[pgno - 1]
        fr = pymupdf.Rect(*NEUTRAL_FRAME[tag])
        if kind == "vector_grid":
            gx, gy = PRO400H_GRID
            fxa, fya = dd._fit_axis(gx), dd._fit_axis(gy)
            vx, vy = sorted(gx), sorted(gy)
            cal = (fxa[0] * vx[0] + fxa[1], vx[0],
                   fxa[0] * vx[-1] + fxa[1], vx[-1],
                   fya[0] * vy[0] + fya[1], vy[0],
                   fya[0] * vy[-1] + fya[1], vy[-1])
            res = (fxa[2] / fxa[0], abs(fya[2] / fya[0]))
        else:
            xs, ys = dd.ticks(pg, fr)
            if len(xs) < 3 or len(ys) < 3:
                return None, f"{len(xs)} x / {len(ys)} y ticks against the frame"
            cal, rx, ry = calibration(xs, ys)
            res = (rx, ry)
        pair, err = _pair_from_curves(curves_in_frame(pg, fr, cal, split=True))
        if pair is None:
            return None, err
        neu, dmn = pair
        span = (400.0, 700.0)
    return dict(tag=tag, profile=prof, neutral=neu, dmin=dmn,
                span=span, axis=res), None


def extract_sheet(root: Path, tag: str):
    rel, pgno, prof, kind = SHEETS[tag]
    pdf = root / "PDF" / "PROFILES" / rel
    if not pdf.is_file():
        return None, f"source not present: {rel}"

    if kind == "kodak_h1":
        pg = pymupdf.open(pdf)[pgno - 1]
        fr = pymupdf.Rect(*VECTOR_FRAME[tag])
        saved = dd.pick
        try:
            dd.pick = lambda _p, _a: fr
            axes = dd.rot_labels(pg)
            if not axes:
                return None, "no rotated axis label"
            r, err = dd.extract(pg, axes[0], GRID)
            if not r:
                return None, err
            sel = dd.pick_dye_set(r[4], GRID)
            if sel is None:
                sel = dd.pick_dye_set_inked(
                    dd.extract_inked(pg, r[0], r[1], GRID), GRID)
        finally:
            dd.pick = saved
        if sel is None:
            return None, "no curve set matched a normalisation family"
        c, m, y, _neu, _dm, _mode, _res = sel
        return dict(tag=tag, profile=prof, cyan=c, magenta=m, yellow=y,
                    spread=0.0, axis=(0.0, 0.0)), None

    if kind == "kodak_raster":
        cfg = KODAK_RASTER[tag]
        gray, solid, dash, (L, R, T, B), fx, fy = prep_kodak_raster(
            pdf, pgno, cfg["clip"])
        if fx[4] or fy[4]:
            return None, f"ticks missing: x {fx[4]} y {fy[4]}"
        mx, cx, my, cy = fx[0], fx[1], fy[0], fy[1]

        def _pass(lam0, seeds):
            sx = int(round(mx * lam0 + cx))
            acc = {}
            for d in (-1, +1):
                tr = dt.trace_predictive(
                    solid, gray, (int(round(mx * 400 + cx)),
                                  int(round(mx * 714 + cx))),
                    T + 5, B - 6, sx,
                    {k: v * my + cy for k, v in seeds.items()},
                    direction=d, **TRACK)
                for k, pts in tr.items():
                    for x, y in pts.items():
                        acc.setdefault(k, {}).setdefault(x, []).append(y)
            out = {}
            for k, cols in acc.items():
                xs = np.array(sorted(cols))
                ys = np.array([np.mean(cols[x]) for x in xs])
                lam = (xs - cx) / mx
                v = np.interp(GRID, lam, (ys - cy) / my)
                v[GRID > lam.max()] = np.nan
                v[GRID < lam.min()] = np.nan
                out[k] = v
            return out

        got = _pass(*cfg["seed"])
        chk = _pass(*cfg["check"])
        need = ("cyan", "magenta", "yellow", "neutral")
        if any(k not in got for k in need):
            return None, f"seed pass returned {sorted(got)}"
        spread = max(float(np.nanmax(np.abs(chk[k] - got[k])))
                     for k in need if k in chk)
        for k in need:
            got[k][~np.isfinite(got[k])] = np.interp(
                GRID[~np.isfinite(got[k])],
                GRID[np.isfinite(got[k])], got[k][np.isfinite(got[k])])
        # The D-min chain is measured but NOT returned for storage; it exists to
        # run the family-C identity over the band where it survives.
        dk, dv = dash_chain(dash, mx, cx, my, cy)
        ident = None
        if dk.size:
            dm = np.interp(GRID, dk, dv)
            ok = (GRID >= dk.min()) & (GRID <= dk.max())
            A = np.vstack([got["cyan"][ok], got["magenta"][ok],
                           got["yellow"][ok]]).T
            kk, *_ = np.linalg.lstsq(A, (got["neutral"] - dm)[ok],
                                     rcond=None)
            ident = (float((kk.max() - kk.min()) / kk.mean()),
                     float(dk.min()), float(dk.max()))
        return dict(tag=tag, profile=prof, cyan=got["cyan"],
                    magenta=got["magenta"], yellow=got["yellow"],
                    spread=spread, axis=(fx[2] / mx, abs(fy[2] / my)),
                    identity=ident), None

    if kind in ("vector_split", "vector_multi"):
        pg = pymupdf.open(pdf)[pgno - 1]
        fr = pymupdf.Rect(*VECTOR_FRAME[tag])
        xs, ys = dd.ticks(pg, fr)
        if len(xs) < 3 or len(ys) < 3:
            return None, f"{len(xs)} x / {len(ys)} y ticks against the frame"
        cal, rx, ry = calibration(xs, ys)
        got = assign_by_band(curves_in_frame(
            pg, fr, cal, split=(kind == "vector_split")))
        if len(got) != 3:
            return None, f"assigned {sorted(got)} instead of three dyes"
        return dict(tag=tag, profile=prof, cyan=got["cyan"],
                    magenta=got["magenta"], yellow=got["yellow"],
                    spread=0.0, axis=(rx, ry)), None

    cfg = RASTER[tag]
    img, gray, ink, vg, hg = prep_raster(pdf, pgno, cfg["clip"])
    if len(vg) < 6 or len(hg) < 4:
        return None, f"{len(vg)} vertical / {len(hg)} horizontal rules found"
    # vg = [frame left, 400, 500, 600, 700, frame right]
    # hg = [frame top, D=1.0, D=0.5, D=0.0]
    mx, cx, rx = fit_axis([400, 500, 600, 700], vg[1:5])
    my, cy, ry = fit_axis([1.0, 0.5, 0.0], hg[1:4])
    if kind == "raster_ink":
        vals = trace_by_ink(img, mx, cx, my, cy, int(vg[0]), int(vg[-1]))
        spread = 0.0
    else:
        vals, sp = trace_black(ink, gray, int(hg[0]) + 5, int(hg[-1]) - 3,
                               mx, cx, my, cy, cfg["passes"])
        spread = max(sp.values())
    if tag == "velvia50":
        y = vals["yellow"]
        y[(GRID >= VELVIA_FLOOR_FROM_NM) & ~np.isfinite(y)] = 0.0
        vals["yellow"] = y
    for k, v in vals.items():
        if not np.isfinite(v).all():
            return None, f"{k} has {int((~np.isfinite(v)).sum())} unmeasured points"
    return dict(tag=tag, profile=prof, cyan=vals["cyan"],
                magenta=vals["magenta"], yellow=vals["yellow"],
                spread=spread, axis=(rx / mx, abs(ry / my))), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--sheet", action="append", choices=sorted(SHEETS))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--emit", action="store_true",
                    help="print the arrays as film_profiles.py tuples")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()
    tags = ns.sheet or list(SHEETS)
    bad = skipped = 0
    got = {}
    print(f"[i] corpus root {root}")
    for tag in tags:
        r, err = extract_sheet(root, tag)
        if r is None:
            if "not present" in (err or ""):
                print(f"  [SKIP] {tag}: {err}")
                skipped += 1
            else:
                print(f"  [FAIL] {tag}: {err}")
                bad += 1
            continue
        got[tag] = r
        pk = tuple(int(GRID[int(np.argmax(r[k]))])
                   for k in ("yellow", "magenta", "cyan"))
        off = max(abs(float(np.max(r[k])) - 1.0)
                  for k in ("yellow", "magenta", "cyan"))
        wpk, woff, wsp = EXPECTED[tag]
        ok = (pk == wpk and off <= woff + 0.0015
              and r["spread"] <= wsp + 0.0015)
        for name, (a, b) in BANDS.items():
            if not a <= int(GRID[int(np.argmax(r[name]))]) <= b:
                ok = False
        print(f"  [{'OK  ' if ok else 'FAIL'}] {tag:11s} {r['profile']:28s} "
              f"peaks {pk} |peak-1|={off:.4f} interpass={r['spread']:.4f} "
              f"axis={r['axis'][0]:.2f}/{r['axis'][1]:.4f}")
        if not ok:
            print(f"         expected peaks {wpk} |peak-1|<={woff} "
                  f"interpass<={wsp}")
            bad += 1

    # ⚠ THE PARTIAL FAMILY-C IDENTITY on 5203 and 5207. `Neutral - Dmin =
    # k(C+M+Y)` with the three k EQUAL is what makes a neutral a neutral, and
    # the coefficients are free, so a small spread is evidence rather than
    # arithmetic. It runs only over the band where the DASHED D-min survives,
    # and the printout says so -- a partial test reported as a whole one would
    # be worse than no test at all.
    for tag, (want, lo, hi) in KODAK_RASTER_IDENTITY.items():
        r = got.get(tag)
        if r is None or not r.get("identity"):
            continue
        sp, glo, ghi = r["identity"]
        ok = (abs(sp - want) <= 0.004 and abs(glo - lo) <= 1.0
              and abs(ghi - hi) <= 1.0)
        print(f"  [{'OK  ' if ok else 'FAIL'}] IDENTITY {tag} k spread {sp:.3f}"
              f" over {glo:.0f}-{ghi:.0f} nm (want {want:.3f},"
              f" {lo:.0f}-{hi:.0f})")
        if not ok:
            bad += 1

    for a, b, dye, tol in SHARED:
        if a not in got or b not in got:
            continue
        rms = float(np.sqrt(np.mean((got[a][dye] - got[b][dye]) ** 2)))
        ok = rms <= tol
        print(f"  [{'OK  ' if ok else 'FAIL'}] SHARED {a}/{b} {dye:8s} "
              f"rms={rms:.5f} (<= {tol})")
        if not ok:
            bad += 1
    for a, b, tol in DISTINCT:
        if a not in got or b not in got:
            continue
        rms = float(np.sqrt(np.mean(np.concatenate(
            [got[a][d] - got[b][d] for d in ("cyan", "magenta", "yellow")])
            ** 2)))
        ok = rms >= tol
        print(f"  [{'OK  ' if ok else 'FAIL'}] DISTINCT {a}/{b} "
              f"rms={rms:.5f} (>= {tol})")
        if not ok:
            bad += 1

    # ---- the six neutral + D-min panels ---------------------------------
    pairs = {}
    for tag in sorted(NEUTRAL_SHEETS):
        r, err = extract_pair(root, tag)
        if r is None:
            if "not present" in (err or ""):
                print(f"  [SKIP] {tag}: {err}")
                skipped += 1
            else:
                print(f"  [FAIL] {tag}: {err}")
                bad += 1
            continue
        pairs[tag] = r
        n, d = r["neutral"], r["dmin"]
        wn, wd, w4, w7, wx, wy = NEUTRAL_EXPECTED[tag]
        # ⚠ THE MASK TEST IS THE ONE THAT MATTERS. d_dmin on a masked colour
        # negative IS the orange mask, so it must fall towards the red, and the
        # neutral must sit above it everywhere. A pair stored the wrong way
        # round passes every other check here.
        falls = float(d[-1]) < float(d[0])
        above = bool((n > d).all())
        ok = (falls and above
              and abs(float(n.mean()) - wn) <= 0.002
              and abs(float(d.mean()) - wd) <= 0.002
              and abs(float(d[0]) - w4) <= 0.002
              and abs(float(d[-1]) - w7) <= 0.002
              and abs(r["axis"][0] - wx) <= 0.02
              and abs(r["axis"][1] - wy) <= 0.02)
        print(f"  [{'OK  ' if ok else 'FAIL'}] PAIR {tag:11s} "
              f"{r['profile']:28s} neutral {n.mean():.3f} dmin {d.mean():.3f} "
              f"mask {d[0]:.3f}->{d[-1]:.3f} "
              f"{'falls' if falls else 'RISES'} "
              f"axis {r['axis'][0]:.2f}/{r['axis'][1]:.4f}")
        if not ok:
            print(f"         expected neutral {wn} dmin {wd} "
                  f"mask {w4}->{w7} axis {wx}/{wy}"
                  + ("" if above else "; NEUTRAL NOT ABOVE DMIN"))
            bad += 1
    # No two of the six may be the same drawing. Both makers reused artwork on
    # their REVERSAL sheets in this same batch, so the question is live.
    names = sorted(pairs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rms = min(float(np.sqrt(np.mean((pairs[a][k] - pairs[b][k]) ** 2)))
                      for k in ("neutral", "dmin"))
            if rms < NEUTRAL_DISTINCT_MIN:
                print(f"  [FAIL] PAIRS {a}/{b} are the same drawing: "
                      f"rms {rms:.5f} < {NEUTRAL_DISTINCT_MIN}")
                bad += 1
    if len(pairs) == len(NEUTRAL_SHEETS):
        worst = min(
            min(float(np.sqrt(np.mean((pairs[a][k] - pairs[b][k]) ** 2)))
                for k in ("neutral", "dmin"))
            for i, a in enumerate(names) for b in names[i + 1:])
        print(f"  [OK  ] all {len(pairs)} neutral pairs are distinct drawings; "
              f"closest {worst:.5f} D rms")

    if ns.emit:
        for tag, r in got.items():
            print(f"\n# {r['profile']}  ({tag})")
            for f, k in (("d_cyan", "cyan"), ("d_magenta", "magenta"),
                         ("d_yellow", "yellow")):
                v = ", ".join(f"{x:.3f}" for x in r[k])
                print(f"{f}=({v}),")
        for tag, r in pairs.items():
            print(f"\n# {r['profile']}  ({tag}) -- neutral pair")
            for f, k in (("d_neutral", "neutral"), ("d_dmin", "dmin")):
                v = ", ".join(f"{x:.3f}" for x in r[k])
                print(f"{f}=({v}),")

    print(f"[i] {len(tags) + len(NEUTRAL_SHEETS) - bad - skipped} ok, "
          f"{bad} bad, {skipped} skipped")
    return 1 if (ns.do_assert and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
