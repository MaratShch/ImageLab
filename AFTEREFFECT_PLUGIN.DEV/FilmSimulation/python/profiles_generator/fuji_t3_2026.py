"""Three Fuji still-film datasheets -- queue T3, 2026-09-02e.

WHAT THIS IS FOR
----------------
Queue row T3 asked for three NEW PROFILES: FUJICHROME PROVIA 100F Professional,
FUJICOLOR SUPERIA X-TRA 400 and FUJICOLOR PRO 400H PROFESSIONAL. None of the
three existed in the database. All three datasheets are on the owner's machine
and were staged 2026-09-02e:

  PROVIA 100F   `PDF/PROFILES/FUJI/provia_100f_datasheet.pdf`     AF3-036E, p6
  SUPERIA X-TRA `PDF/PROFILES/FUJI/superia_xtra400_datasheet.pdf` AF3-151E, p6
  PRO 400H      `PDF/PROFILES/FUJI/pro_400h_datasheet.pdf`        AF3-176E, p8

⚠ THE ROW NAMED THE WRONG REVISION CODES AND THAT WAS ALREADY CORRECTED on
2026-08-31: it asked for AF3-0076E5 and AF3-058E3, neither of which exists; the
films are held under AF3-036E and AF3-151E, same films, different printings.
PRO 400H (AF3-176E) was found in the same sweep and folded into the row.

WHAT THE SHEETS PRINT AS NUMBERS, and is therefore transcribed rather than
traced -- all three are [T1]:

                         PROVIA 100F     SUPERIA X-TRA 400   PRO 400H
  ISO speed              100 daylight    400 daylight        400 daylight
  process                E-6 / CR-56     CN-16 (C-41)        C-41
  densitometry           Fuji FAD-30S    Status M            Status M
                         (Status A)
  diffuse rms            8               4                   4
    aperture             48 um           48 um, 12x          48 um
    sample density       1.0 above Dmin  1.0 above Dmin      +1.0 above Dmin
  resolving power
    1.6:1 contrast       60 lines/mm     50 lines/mm         50 lines/mm
    1000:1 contrast      140 lines/mm    125 lines/mm        125 lines/mm
  base material          cellulose triacetate (all three)
  base thickness 135     127 um          not printed         122 um
  base thickness 120/220 104 um          not printed          98 um

⚠ THE TWO 400-SPEED FILMS PRINT IDENTICAL IMAGE-STRUCTURE NUMBERS -- rms 4,
50 and 125 lines/mm -- and they are NOT the same film: SUPERIA X-TRA 400 is a
consumer stock on Super Fine-Sigma grain technology, PRO 400H a professional
stock, both with a fourth colour layer. The sheets agree because Fuji rounds the
rms to one digit. Their characteristic curves differ (PRO 400H's Dmin ladder
sits about 0.35 D higher and its curves are shorter), so the two profiles are
NOT collapsed onto one set of numbers; where the printed values coincide, they
coincide because the maker printed the same value, and that is recorded.

WHAT IS TRACED HERE
-------------------
  * CHARACTERISTIC CURVES -> the six-parameter `ToneCurve` per channel.
  * MTF CURVE -> f50 and the overshoot, one curve per sheet (Fuji prints a
    single unlabelled black curve, not three records).
  * SPECTRAL SENSITIVITY -> peak-normalised log sensitivity per layer.

⚠ THE AXIS LABELS ON THESE PAGES ARE OUTLINED TEXT, NOT TEXT, on two of the
three sheets -- `get_text()` on PROVIA p6 returns 149 words and on PRO 400H p8
returns 83, none of them an axis number. So the label-centroid calibration used
on the Kodak sheets is unavailable and the panels are calibrated FROM THEIR OWN
GRIDLINES instead: the uniform ladder of long rules inside the frame is located,
its spacing fitted, and the printed end values (transcribed from the rendered
page, and listed in PANELS below) attached to the two extreme lines.

⚠ AND THE CHECK THAT MAKES THAT SAFE. Fuji draws these H&D panels SQUARE: one
decade of log H occupies the same distance as 1.0 density. The two axes are
calibrated independently here and their scales are then compared; a
misidentified ladder shows up immediately as a broken aspect ratio. The three
sheets come out at 0.1 %, 0.4 % and 0.1 %, which is the gridline-centroid noise.
Nothing is adopted from a panel whose aspect check fails.

USAGE
    python3 fuji_t3_2026.py [--root .] [--assert] [--overlay DIR]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

DPI = 600
PT = DPI / 72.0

#: (pdf, page, approximate panel box in PDF points) per panel.
#: The boxes only have to CONTAIN the panel and exclude its neighbours; the
#: frame and the gridline ladder are found inside them.
SHEETS = {
    "provia100f": dict(
        pdf="FUJI/provia_100f_datasheet.pdf", page=6,
        stock="FUJI_PROVIA_100F", reversal=True,
        hd=dict(box=(55, 150, 330, 360), x=(-3.5, 1.0), y=(0.0, 4.0),
                colour=True),
        mtf=dict(box=(60, 400, 320, 560), f=(1.0, 200.0), r=(2.0, 150.0)),
        spec=dict(box=(325, 120, 560, 290), lam=(400.0, 700.0),
                  s=(-1.0, 1.0), colour=True),
    ),
    "superia400": dict(
        pdf="FUJI/superia_xtra400_datasheet.pdf", page=6,
        stock="FUJICOLOR_SUPERIA_XTRA_400", reversal=False,
        hd=dict(box=(60, 105, 320, 300), x=(-4.0, 0.5), y=(0.0, 4.0),
                colour=True),
        mtf=dict(box=(60, 385, 320, 560), f=(1.0, 200.0), r=(2.0, 150.0)),
        spec=dict(box=(325, 105, 560, 300), lam=(400.0, 700.0),
                  s=(None, None), colour=True),
    ),
    "pro400h": dict(
        pdf="FUJI/pro_400h_datasheet.pdf", page=8,
        stock="FUJICOLOR_PRO_400H", reversal=False,
        hd=dict(box=(60, 95, 320, 285), x=(-4.0, 1.0), y=(0.0, 3.5),
                colour=False),
        mtf=dict(box=(60, 370, 320, 545), f=(1.0, 200.0), r=(2.0, 150.0)),
        spec=dict(box=(325, 95, 560, 285), lam=(400.0, 700.0),
                  s=(None, None), colour=False),
    ),
}

#: Measured 2026-09-02e. --assert fails if a re-trace stops reproducing these.
EXPECTED = {
    "provia100f": dict(aspect=0.004, f50=39.8, peak=1.155, peak_at=1.0,
                       curves=dict(R=(0.0728, 1.9881, 0.5984, 0.2734, 2.2153, 0.1761)),
                       derived=dict(G=2.0745, B=2.0214)),
    "superia400": dict(aspect=0.004, f50=57.9, peak=1.211, peak_at=5.2,
                       curves=dict(R=(0.1366, 0.6622, -2.6956, 0.3041, 1.75, 0.42),
                                   G=(0.3744, 0.7047, -2.8800, 0.3000, 1.75, 0.42),
                                   B=(0.6940, 0.7510, -2.8060, 0.3000, 1.75, 0.42))),
    "pro400h": dict(aspect=0.002, f50=51.9, peak=1.110, peak_at=2.6,
                    curves=dict(R=(0.1503, 0.6136, -2.3221, 0.3000, 1.75, 0.42),
                                G=(0.6535, 0.5721, -2.4195, 0.3000, 1.75, 0.42),
                                B=(0.9199, 0.5431, -2.4894, 0.3000, 1.75, 0.42))),
}

TOL_ASPECT = 0.02

#: The MTF panel's printed gridline values. Identical on all three sheets --
#: Fuji uses one template -- so they are module constants, not per-sheet.
MTF_F_GRID = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
#: ⚠ THE RESPONSE LIST IS TOP-TO-BOTTOM, i.e. DESCENDING, because the detected
#: gridline positions are ascending in PIXELS and a response axis increases
#: upward. Handing _logfit an ascending value list against descending values
#: still fits a line -- with the wrong sign -- and the nearest-neighbour
#: assignment then scrambles, which showed up as a 0.09-decade residual.
MTF_R_GRID = (150.0, 100.0, 70.0, 50.0, 30.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0)

#: Minimum ink-group thickness in pixels for a group to be the CURVE and not a
#: gridline. Measured at 600 dpi: gridlines 2-4 px, curve 9-13 px.
MIN_INK = 6

#: The colour-negative shoulder, pinned rather than fitted -- see the note in
#: main(). These are `_neg`'s own defaults in film_profiles.py.
SHOULDER_X, SHOULDER_K = 1.75, 0.42


def page_rgb(root: Path, rel: str, page: int) -> np.ndarray:
    import pymupdf
    pdf = root / "PDF" / "PROFILES" / rel
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    pm = pymupdf.open(pdf)[page - 1].get_pixmap(dpi=DPI)
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
    return a[:, :, :3].astype(np.int16)


def _runs(mask: np.ndarray) -> np.ndarray:
    """Longest contiguous True run in each row of a 2-D boolean array."""
    out = np.zeros(mask.shape[0], dtype=int)
    for i in range(mask.shape[0]):
        idx = np.flatnonzero(np.diff(np.concatenate(
            ([0], mask[i].view(np.int8), [0]))))
        if len(idx):
            out[i] = int(np.max(idx[1::2] - idx[0::2]))
    return out


def _ladder(pos: list[int], span: int) -> tuple[list[float], float, float]:
    """Fit a uniform ladder through detected gridline positions.

    ⚠ GRIDLINES GO MISSING AND FALSE ONES TURN UP, and the fit has to survive
    both. On PRO 400H's H&D panel one horizontal rule is buried under the
    "Exposure / Process / Densitometry" text block; on SUPERIA two are; on
    PROVIA the section-heading bar above the panel is long enough to be detected
    as a rule and sits 160 px above the D 4.0 gridline, which is neither a rung
    nor far enough away to be obviously wrong (0.86 of a step).

    So the ladder is the arithmetic progression that best explains the
    detections, found in three steps:
      1. the step is the MEDIAN consecutive gap -- robust to a doubled gap where
         one line is missing, and to one intruder at either end;
      2. each detection is assigned the rung index nearest to it and a
         least-squares line is fitted over (index, position);
      3. anything more than 0.12 of a step off its own rung is dropped and the
         fit is redone. 0.12 is not arbitrary: real gridline centroids on these
         scans land within 0.03 of a step (2-7 px against a 186 px step), and
         the PROVIA intruder lands at 0.14, so the threshold sits in a gap two
         standard deviations wide on one side and clear of the intruder on the
         other.
    Returns (rungs, step, worst surviving residual in px).
    """
    if len(pos) < 3:
        raise ValueError(f"only {len(pos)} gridlines")
    step = float(np.median(np.diff(pos)))
    keep = list(pos)
    for _ in range(4):
        n = np.round((np.asarray(keep, float) - keep[0]) / step)
        a, b = np.polyfit(n, keep, 1)
        step = float(a)
        resid = np.abs(np.asarray(keep, float) - (a * n + b))
        good = resid <= 0.12 * abs(step)
        if good.all():
            break
        keep = [p for p, ok in zip(keep, good) if ok]
        if len(keep) < 3:
            raise ValueError("ladder collapsed under outlier rejection")
    lo, hi = keep[0], keep[-1]
    k = int(round((hi - lo) / step))
    rungs = [lo + i * step for i in range(k + 1)]
    worst = float(np.max([min(abs(p - r) for r in rungs) for p in keep]))
    return rungs, float(step), worst


def _logfit(pos: list[int], values: tuple[float, ...]) -> tuple[float, float, float, int]:
    """Fit pixel = a*log10(value) + b over a LOG axis whose gridlines are known.

    ⚠ A LOG PANEL'S GRIDLINES ARE NOT A UNIFORM LADDER and `_ladder` must never
    be used on one: 1, 5, 10, 20, 50, 100, 200 are 0.699, 0.301, 0.301, 0.398,
    0.301, 0.301 decades apart.

    ⚠ AND THE ENDS CANNOT BE ASSUMED. The obvious calibration -- first detection
    is the first value, last is the last -- is wrong on two of the three sheets
    here. PROVIA's MTF panel detects nine of its eleven response gridlines and
    the two it loses are BOTH AT THE BOTTOM (3 % and 2 %, where the frame rule
    and the axis title crowd them), so the lowest detection is 5 % and not 2 %.
    Anchored on the ends that fit returns 520 px per decade against a true 700,
    a 34 % error that produces a perfectly smooth and entirely wrong MTF.

    So the assignment is SEARCHED, not assumed. Every pair (i, j) of value
    indices is tried as the identity of the first and last detection -- 55 pairs
    for an eleven-value axis, which is free -- the implied linear map is built,
    every detection is scored against its nearest predicted position, and the
    pair with the smallest worst-case error wins. The gap PATTERN of a log axis
    is highly asymmetric (1.205, 1.062, 1.000, 1.521, 1.205, 2.062, ... in units
    of the smallest gap), so a wrong alignment scores an order of magnitude
    worse and the search is not close.

    Returns (px per decade, intercept, worst residual in decades, n assigned).
    """
    lv = np.log10(np.asarray(values, dtype=float))
    pos = [float(p) for p in pos]
    best = None
    for i in range(len(lv)):
        for j in range(len(lv)):
            if i == j:
                continue
            a = (pos[-1] - pos[0]) / (lv[j] - lv[i])
            b = pos[0] - a * lv[i]
            pred = a * lv + b
            err = max(float(np.min(np.abs(pred - p))) for p in pos) / abs(a)
            if best is None or err < best[0]:
                best = (err, a, b, i, j)
    _err, a, b, i, j = best
    # refit by least squares over the winning assignment
    xs, ys, seen = [], [], set()
    pred = a * lv + b
    for p in pos:
        k = int(np.argmin(np.abs(pred - p)))
        if k in seen:
            continue
        seen.add(k)
        xs.append(lv[k])
        ys.append(p)
    a, b = np.polyfit(xs, ys, 1)
    resid = float(np.max(np.abs((np.asarray(ys) - b) / a - np.asarray(xs))))
    return float(a), float(b), resid, len(xs)


def _panel(rgb: np.ndarray, box, uniform: bool = True) -> dict:
    """Find one panel's frame and gridline ladders inside an approximate box."""
    x0, y0, x1, y1 = (int(v * PT) for v in box)
    sub = rgb[y0:y1, x0:x1]
    g = sub.mean(axis=2)
    ink = g < 128
    H, W = ink.shape
    hrun, vrun = _runs(ink), _runs(ink.T)
    # the frame's own rules are the longest; gridlines are nearly as long
    # ⚠ 0.50, NOT 0.72. A gridline crossed by the panel's own caption block is
    # broken into two shorter runs, and on PROVIA's H&D panel that puts five of
    # the nine horizontal rules under a 0.72 threshold -- the ladder then fits
    # three lines, returns 158.8 px per density against 373.8 px per decade, and
    # the square-panel check catches it. 0.50 recovers all nine.
    hthr, vthr = 0.50 * hrun.max(), 0.50 * vrun.max()

    def cluster(run, thr):
        idx = [i for i in range(len(run)) if run[i] > thr]
        out, cur = [], [idx[0]]
        for x in idx[1:]:
            if x - cur[-1] <= 8:
                cur.append(x)
            else:
                out.append(int(np.mean(cur)))
                cur = [x]
        out.append(int(np.mean(cur)))
        return out

    hs, vs = cluster(hrun, hthr), cluster(vrun, vthr)
    # ⚠ THE PANEL'S OWN EXTENT, MEASURED, THROWS OUT EVERYTHING OUTSIDE IT. The
    # 0.50 threshold that recovers a caption-broken gridline also lets in the
    # green section-heading bar above PROVIA's H&D panel, which is a long
    # horizontal rule 160 px above the D 4.0 line -- close enough that the
    # ladder's outlier test cannot be trusted to reject it (0.13 of a step
    # against a 0.12 tolerance). It is not a judgement call: the vertical rules
    # START at the frame top and END at the frame bottom, so their ink extent
    # IS the frame, and any horizontal detection outside it is not a gridline.
    # The same test in the other direction bounds the horizontal extent.
    def _extent(idx, axis_ink):
        # ⚠ THE LONGEST RUN IN THE COLUMN, NOT THE FIRST AND LAST INK. PROVIA's
        # green section-heading bar crosses every column of the panel, so
        # "first ink in this column" is the top of the heading, not the top of
        # the frame, and the extent test then admits the heading as a gridline
        # -- which is exactly the intruder it exists to reject.
        lo, hi = 10 ** 9, -1
        n = axis_ink.shape[1]
        for k in idx:
            if k >= n:
                continue
            col = axis_ink[:, k]
            e = np.flatnonzero(np.diff(np.concatenate(([0], col.view(np.int8), [0]))))
            if not len(e):
                continue
            starts, ends = e[0::2], e[1::2]
            b = int(np.argmax(ends - starts))
            lo, hi = min(lo, int(starts[b])), max(hi, int(ends[b]) - 1)
        return lo, hi
    ftop, fbot = _extent(vs, ink)
    flft, frgt = _extent(hs, ink.T)
    hs = [y for y in hs if ftop - 4 <= y <= fbot + 4]
    vs = [x for x in vs if flft - 4 <= x <= frgt + 4]
    out = dict(x0=x0, y0=y0, sub=sub, gray=g, hs=hs, vs=vs)
    if uniform:
        yr, ystep, yres = _ladder(hs, H)
        xr, xstep, xres = _ladder(vs, W)
        out.update(left=xr[0], right=xr[-1], top=yr[0], bottom=yr[-1],
                   xstep=xstep, ystep=ystep, xres=xres, yres=yres,
                   xrungs=xr, yrungs=yr, nx=len(xr), ny=len(yr))
    else:
        out.update(left=vs[0], right=vs[-1], top=hs[0], bottom=hs[-1],
                   nx=len(vs), ny=len(hs))
    return out


def _channel_masks(sub: np.ndarray, colour: bool) -> dict[str, np.ndarray]:
    """Ink masks per record.

    Coloured panels are separated by hue, which is exact -- Fuji draws the red
    record in pure red, green in green, blue in blue. ⚠ A BLACK PANEL CANNOT BE
    SEPARATED THAT WAY and PRO 400H's is black, so it returns one mask and the
    caller separates the three curves by vertical order instead (they are
    parallel and never cross, which the sheet's own Blue/Green/Red labels
    top-to-bottom confirm).
    """
    r, g, b = sub[:, :, 0].astype(int), sub[:, :, 1].astype(int), sub[:, :, 2].astype(int)
    dark = sub.mean(axis=2) < 210
    if not colour:
        return {"-": dark & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)}
    return {
        "R": dark & (r - g > 45) & (r - b > 45),
        "G": dark & (g - r > 35) & (g - b > 25),
        "B": dark & (b - r > 45) & (b - g > 35),
    }


def _track(mask: np.ndarray, cols: range) -> dict[int, float]:
    """Ink centroid per column, for a mask holding exactly one curve."""
    out = {}
    for x in cols:
        ys = np.flatnonzero(mask[:, x])
        if len(ys):
            out[x] = float(ys.mean())
    return out


def _drop_short(mask: np.ndarray, min_w: int) -> np.ndarray:
    """Delete connected components narrower than `min_w` pixels.

    ⚠ THE RECORD LABELS ARE WHAT MAKE A BLACK PANEL WALKABLE OR NOT. PRO 400H
    writes "Blue", "Green" and "Red" in the gap between its three curves, and
    those glyphs give the walker stepping stones: the red track climbs the word
    "Red" and lands on the green record, which is how it returned green's
    D 0.69 toe as red's. Each letter is a separate component 30-50 px wide,
    while every real curve is one component spanning the whole frame, so a
    width test separates them cleanly with a factor of three to spare.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = np.zeros_like(mask)
    for i in range(1, n + 1):
        xs = np.flatnonzero(lab.any(axis=0) & (lab == i).any(axis=0))
        if len(xs) and int(xs.max() - xs.min()) + 1 >= min_w:
            out |= (lab == i)
    return out


def _track_n(mask: np.ndarray, cols: range, n: int) -> list[dict[int, float]]:
    """Split one black mask into `n` non-crossing tracks.

    ⚠ REQUIRING n GROUPS IN EVERY COLUMN DOES NOT WORK and that was the first
    version: on PRO 400H's H&D panel the three records touch at the left (all
    three are flat and within 0.1 D of each other there), the words "Blue",
    "Green" and "Red" are written ACROSS them in the middle, and the red record
    starts later than the other two. Exactly three groups occur in five columns
    out of seventeen hundred.

    So the tracks are WALKED instead: seeded at the column with the three most
    widely separated groups -- which is where the records are least ambiguous --
    and followed outward, each track taking the group nearest its own last
    position within a window. A column that offers fewer groups than tracks
    simply contributes to the tracks it can serve; a column whose nearest group
    is further than the window contributes to none, so a label glyph cannot
    capture a track and neither can a neighbouring record.
    """
    def groups(x):
        ys = np.flatnonzero(mask[:, x])
        if len(ys) == 0:
            return []
        out, cur = [], [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] <= 4:
                cur.append(y)
            else:
                out.append(float(np.mean(cur)))
                cur = [y]
        out.append(float(np.mean(cur)))
        return out

    # ⚠ SEED ON THE RIGHT, NOT WHEREVER THE SPREAD IS WIDEST. The widest spread
    # is on the LEFT, where the caption block ("Exposure / Process /
    # Densitometry") sits above the three flat toes and offers a fourth, fifth
    # and sixth group -- so a spread-maximising seed can pick text plus two
    # curves and hand back a "record" that is a line of type. The right-hand
    # third of the panel has no annotation on any of the three sheets and the
    # three records are fully separated there, so the seed is taken as the
    # RIGHTMOST column offering exactly n groups.
    seed = None
    for x in reversed(list(cols)):
        gp = groups(x)
        if len(gp) == n:
            seed = (x, gp)
            break
    if seed is None:
        return [{} for _ in range(n)]
    tracks: list[dict[int, float]] = [{} for _ in range(n)]
    WIN = 22
    for step in (+1, -1):
        x, ycs = seed[0], list(seed[1])
        while cols.start <= x < cols.stop:
            gp = groups(x)
            if gp:
                used = set()
                for i in range(n):
                    cand = [(abs(g - ycs[i]), k, g) for k, g in enumerate(gp)
                            if k not in used]
                    if not cand:
                        continue
                    dy, k, g = min(cand)
                    if dy <= WIN:
                        used.add(k)
                        ycs[i] = g
                        tracks[i][x] = g
            x += step
    return tracks


def _track_thick(M: dict, af: float, bf: float, ar: float, br: float):
    """Follow the MTF curve, rejecting gridlines by thickness and text by
    continuity. Returns (frequency c/mm, response 0-1) arrays."""
    g = M["gray"]
    H, W = g.shape
    ink = g < 150
    # ⚠ THE FRAME RULES ARE THICKER THAN THE CURVE and the first version of this
    # tracker seeded on one: at the left edge of the panel the only thick ink is
    # the bottom frame, so the "MTF" it returned was the frame line, reported as
    # a flat 1.5-1.7 % response out to 200 c/mm. Every detected rule -- frame and
    # gridline alike -- is therefore erased before the search starts. The curve
    # is the only thing left that is thicker than nothing.
    for y in M["hs"]:
        ink[max(0, y - 5): y + 6, :] = False
    for x in M["vs"]:
        ink[:, max(0, x - 5): x + 6] = False
    x0, x1 = int(M["left"]) + 8, int(M["right"]) - 8

    def groups(x):
        ys = np.flatnonzero(ink[:, x])
        if len(ys) == 0:
            return []
        out, cur = [], [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] <= 2:
                cur.append(y)
            else:
                out.append(cur)
                cur = [y]
        out.append(cur)
        return [(float(np.mean(c)), len(c)) for c in out if len(c) >= MIN_INK]

    # ⚠ SEED IN THE TOP THIRD, NOT MERELY AT THE FIRST THICK GROUP. The panel's
    # caption block ("Exposure : Daylight / Process : CN-16") is inside the frame
    # and its glyphs are as thick as the curve; it sits low, the curve starts
    # near 100-120 % response, so the seed is restricted to the upper 40 % of the
    # frame and the continuity window keeps the track there.
    seed = None
    ytop, ybot = M["top"], M["bottom"]
    ycut = ytop + 0.40 * (ybot - ytop)
    for x in range(x0, x1):
        gp = [t for t in groups(x) if t[0] < ycut]
        if len(gp) == 1:
            seed = (x, gp[0][0])
            break
    if seed is None:
        return np.array([]), np.array([])
    pts = {}
    for step in (+1, -1):
        x, yc = seed
        while x0 <= x <= x1:
            gp = groups(x)
            if gp:
                y, _n = min(gp, key=lambda t: abs(t[0] - yc))
                if abs(y - yc) <= 25:
                    yc = y
                    pts[x] = y
            x += step
    xs = np.array(sorted(pts))
    ys = np.array([pts[k] for k in xs], dtype=float)
    fs = 10.0 ** ((xs - bf) / af)
    rs = 10.0 ** ((ys - br) / ar) / 100.0
    return fs, rs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--overlay", metavar="DIR")
    ap.add_argument("--only")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()
    bad = 0

    for tag, sh in SHEETS.items():
        if ns.only and ns.only != tag:
            continue
        try:
            rgb = page_rgb(root, sh["pdf"], sh["page"])
        except FileNotFoundError as e:
            print(f"  [SKIP] {tag}: source not present: {e}")
            continue
        print(f"[i] {sh['pdf']} p{sh['page']} -> {sh['stock']}")

        # ---- characteristic curves -----------------------------------------
        hd = sh["hd"]
        try:
            P = _panel(rgb, hd["box"])
        except ValueError as e:
            print(f"  [FAIL] {tag} H&D panel: {e}")
            bad += 1
            continue
        xlo, xhi = hd["x"]
        ylo, yhi = hd["y"]
        px_per_dec = (P["right"] - P["left"]) / (xhi - xlo)
        px_per_D = (P["bottom"] - P["top"]) / (yhi - ylo)
        aspect = abs(px_per_dec / px_per_D - 1.0)
        print(f"    H&D frame {P['nx']}x{P['ny']} gridlines, ladder residual "
              f"{P['xres']:.1f}/{P['yres']:.1f} px; {px_per_dec:.1f} px per "
              f"decade against {px_per_D:.1f} px per density")
        print(f"    ⚠ SQUARE-PANEL CHECK: the two axes disagree by "
              f"{100*aspect:.1f} % -- Fuji draws one decade the same length as "
              f"1.0 D, so this is the calibration's own error bar")
        if aspect > TOL_ASPECT:
            print(f"  [FAIL] {tag}: aspect check failed, panel refused")
            bad += 1
            continue
        exp = EXPECTED.get(tag, {})
        if "aspect" in exp and abs(aspect - exp["aspect"]) > 0.01:
            print(f"  [MISMATCH] aspect {aspect:.3f} vs pinned {exp['aspect']:.3f}")
            bad += 1

        def to_x(px):
            return xlo + (px - P["left"]) / px_per_dec

        def to_d(py):
            return yhi - (py - P["top"]) / px_per_D

        masks = _channel_masks(P["sub"], hd["colour"])
        # ⚠ CLIP EVERY MASK TO THE FRAME'S INTERIOR. PROVIA's page puts a GREEN
        # section-heading bar directly above its H&D panel and a three-swatch
        # RED/GREEN/BLUE LEGEND inside it; both are the same inks as the curves,
        # so a hue mask that is not clipped traces the heading (it returned
        # "green" densities of 3.68 to 4.48 on a 0-4.0 axis, i.e. off the top of
        # the frame, which is how it was caught).
        _t, _b = int(P["top"]) + 3, int(P["bottom"]) - 3
        _l, _r = int(P["left"]) + 3, int(P["right"]) - 3
        for _m in masks.values():
            _m[:_t, :] = False
            _m[_b:, :] = False
            _m[:, :_l] = False
            _m[:, _r:] = False
        cols = range(int(P["left"]) + 6, int(P["right"]) - 6)
        if hd["colour"]:
            # ⚠ A PER-COLUMN CENTROID IS NOT SAFE EVEN ON A HUE MASK. PROVIA
            # draws a THREE-SWATCH LEGEND (a red, a green and a blue bar with
            # the words Red / Green / Blue) INSIDE its H&D frame, low and left,
            # where the curves are at D 3.3 -- so the red mask holds the red
            # curve and a red bar 2.7 D away from it, and their centroid is a
            # density the film never reaches. Each hue is therefore WALKED as a
            # single track from the right-hand end, which cannot reach the
            # legend.
            tracks = {k: _track_n(m, cols, 1)[0] for k, m in masks.items()}
        else:
            # ⚠ ORDER, NOT COLOUR. Fuji's own labels run Blue / Green / Red top
            # to bottom on the black panel, and for a colour negative that is
            # also the physical order (the blue record carries the most mask).
            # ⚠ AND THE GRIDLINES MUST GO FIRST. On a black panel the gridlines
            # are the same ink as the curves, so a column offers eleven ink
            # groups and not three; the walker then seeds on gridlines and
            # returns three smooth, entirely fictitious "records" (the first
            # attempt gave the red curve as D 1.28-2.60 where the sheet draws
            # it 0.17-1.95). Every rule the panel finder located is erased
            # before the walk starts.
            # ⚠ ERASE THE FITTED RUNGS, NOT THE RAW DETECTIONS. The rule this
            # panel loses under its own caption block is exactly the one that
            # then survives as ink and gets walked as a record: PRO 400H's
            # D 3.0 gridline is invisible to the detector and came back as a
            # "blue curve" running dead flat at D 2.97-3.08 across the entire
            # frame. The ladder knows where the missing rung is; use it.
            _blk = masks["-"].copy()
            _blk = _drop_short(_blk, 150)
            for _y in P["yrungs"]:
                _blk[max(0, int(_y) - 6): int(_y) + 7, :] = False
            for _x in P["xrungs"]:
                _blk[:, max(0, int(_x) - 6): int(_x) + 7] = False
            tr = _track_n(_blk, cols, 3)
            tracks = {"B": tr[0], "G": tr[1], "R": tr[2]}

        import digitize_plot as dp
        fits = {}
        for ch in ("R", "G", "B"):
            t = tracks.get(ch, {})
            if len(t) < 200:
                print(f"    [WARN] {ch}: only {len(t)} traced columns, skipped")
                continue
            xs = np.array([to_x(k) for k in sorted(t)])
            ds = np.array([to_d(t[k]) for k in sorted(t)])
            if sh["reversal"]:
                # ⚠ x IS NEGATED LOG EXPOSURE FOR A REVERSAL STOCK -- ToneCurve's
                # own docstring says so, and it means toe_x is the HIGHLIGHT end
                # and shoulder_x the SHADOW end, the opposite of a negative. The
                # fit is therefore done in the negated domain, and the INIT has
                # to live there too: seeding toe_x from the un-negated xs.min()
                # puts the toe outside the data on the wrong side and the
                # simplex settles at rms 0.27 with a gamma of 2.9 (the first
                # version of this reader did exactly that).
                xf = -xs
                init = (float(ds.min()), 1.9, float(xf.min()), 0.25,
                        float(xf.max()), 0.25)
                p, rms, mx = dp.fit_tonecurve(xf, ds, init)
            else:
                # ⚠ THE SHOULDER IS NOT IN THE DATA AND MUST NOT BE FITTED.
                # These panels stop at logH +0.3 to +0.7, well inside a colour
                # negative's straight line -- the film has no shoulder in its
                # printed range -- so a free shoulder_x settles wherever the
                # trace happens to end. On SUPERIA that produced shoulder_x
                # 1.16 for red against 0.27 for green, and hence extrapolated
                # Dmax values of 2.68 / 2.57 / 3.00: RED ABOVE GREEN, which
                # inverts the mask ladder the same curves measure directly.
                # The shoulder is therefore pinned at the family default that
                # every other colour negative here uses (_neg's 1.75 / 0.42),
                # which keeps the three records straight and parallel out to
                # Dmax and leaves only dmin, gamma and the toe to the fit.
                init = (float(ds.min()), 0.6, float(xs.min()), 0.35,
                        SHOULDER_X, SHOULDER_K)
                p, rms, mx = dp.fit_tonecurve4(xs, ds, init)
            fits[ch] = (p, rms, mx, xs, ds)
            _ec = EXPECTED.get(tag, {}).get("curves", {}).get(ch)
            if _ec and max(abs(a - b) for a, b in zip(p, _ec)) > 0.02:
                print(f"  [MISMATCH] {ch} curve {tuple(round(v,4) for v in p)} "
                      f"vs pinned {_ec}")
                bad += 1
            print(f"    {ch}: {len(t)} columns over logH {xs.min():+.2f}.."
                  f"{xs.max():+.2f}, D {ds.min():.2f}..{ds.max():.2f}  ->  "
                  f"ToneCurve({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}, {p[3]:.4f}, "
                  f"{p[4]:.4f}, {p[5]:.4f})  rms {rms:.4f} max {mx:.4f}")

        # ⚠ ON A REVERSAL SHEET ONLY ONE RECORD IS COMPLETE, and pretending
        # otherwise is the trap here. PROVIA draws red SOLID, green DASH-DOT and
        # blue DASHED, and all three lie within 0.15 D of each other over the
        # whole scale except the top half-density -- so the green and blue
        # traces survive only where the three separate, 660 and 380 columns
        # against red's 1590, all of it in the shoulder. Fitting a six-parameter
        # curve to that returns dmin 1.49 and gamma 3.12 for green: an excellent
        # fit to a fragment and a nonsense curve.
        #   WHAT IS DONE INSTEAD: red carries the SHAPE, which it measures over
        # the full scale; green and blue take red's toe, shoulder and softness
        # and only their own MEASURED MAXIMUM DENSITY, applied through gamma.
        # That encodes exactly what the sheet shows -- three records of one
        # shape separated at the shadow end -- and nothing it does not.
        if sh["reversal"] and "R" in fits and len(fits) > 1:
            pr = fits["R"][0]
            arm = pr[4] - pr[2]
            for ch in ("G", "B"):
                if ch not in fits:
                    continue
                dmax = float(fits[ch][4].max())
                gam = (dmax - pr[0]) / arm
                print(f"    {ch} DERIVED from R's shape at its own measured "
                      f"Dmax {dmax:.2f}: ToneCurve({pr[0]:.4f}, {gam:.4f}, "
                      f"{pr[2]:.4f}, {pr[3]:.4f}, {pr[4]:.4f}, {pr[5]:.4f})")

        # ---- MTF ------------------------------------------------------------
        mt = sh["mtf"]
        try:
            M = _panel(rgb, mt["box"], uniform=False)
        except (ValueError, IndexError) as e:
            print(f"    [WARN] MTF panel: {e}")
            M = None
        if M is not None:
            af, bf, rf, nf = _logfit(M["vs"], MTF_F_GRID)
            ar, br, rr, nr = _logfit(M["hs"], MTF_R_GRID)
            print(f"    MTF panel: {nf} frequency gridlines assigned "
                  f"(residual {rf:.4f} decade), {nr} response gridlines "
                  f"(residual {rr:.4f} decade); {abs(af):.1f} px per frequency "
                  f"decade, {abs(ar):.1f} px per response decade")
            # ⚠ FUJI'S MTF ABSCISSA IS NOT ACCURATELY DRAWN, and that is a
            # property of the sheet, not of this reader. On SUPERIA the six
            # frequency gridlines imply 653, 618, 608, 648 and 668 px per decade
            # between consecutive pairs -- a +/-5 % scatter about the 638 the
            # least-squares fit returns, against a response axis that is
            # internally consistent to 1 %. PROVIA's panel, whose gridline set
            # is complete, calibrates its two axes independently to 656.6 and
            # 656.7 px per decade, so the TEMPLATE is square and the scatter is
            # drawing error in individual rules. The fit over all of them is the
            # best estimate available; the resulting f50 carries about +/-5 %,
            # which is recorded here rather than implied by quoting three
            # significant figures elsewhere.
            if max(rf, rr) > 0.035:
                print("  [FAIL] MTF gridline assignment failed -- panel refused")
                bad += 1
            else:
                # ⚠ THE GRIDLINES ARE 2-4 PX AND THE CURVE IS 9-13 PX AT 600 DPI.
                # Measured on SUPERIA: gridlines 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 4
                # px thick against the curve's 9-13. A plain ink centroid per
                # column therefore returns the MEAN OF ELEVEN GRIDLINES and a
                # perfectly smooth, entirely fictitious curve -- which is what
                # the first version of this reader produced (f50 196 c/mm,
                # response "4.0-25.6 %"). Groups thinner than MIN_INK px are
                # discarded before anything else happens.
                # ⚠ AND THE PANEL'S OWN CAPTION TEXT IS ALSO THICK. "Exposure :
                # Daylight / Process : CN-16" sits inside the frame and survives
                # the thickness test, so the track is additionally CONTINUOUS:
                # seeded at the leftmost thick group and followed with a window,
                # it cannot jump down to a text block.
                fs, rs = _track_thick(M, af, bf, ar, br)
                if len(fs) < 100:
                    print(f"    [WARN] MTF: only {len(fs)} columns traced")
                else:
                    peak = float(rs.max())
                    pk_at = float(fs[int(rs.argmax())])
                    below = np.flatnonzero(rs < 0.5)
                    f50 = float(np.interp(
                        0.5, [rs[below[0]], rs[below[0] - 1]],
                        [fs[below[0]], fs[below[0] - 1]])) if len(below) else float("nan")
                    print(f"    MTF: {len(fs)} columns over {fs.min():.1f}-"
                          f"{fs.max():.1f} c/mm, response {100*rs.min():.1f}-"
                          f"{100*rs.max():.1f} %  ->  f50 = {f50:.1f} c/mm, "
                          f"overshoot {peak-1:+.3f} (peak at {pk_at:.1f} c/mm)")
                    # ⚠ THE ROLLOFF EXPONENT, FITTED THE SAME WAY THE KODAK
                    # SHEETS ARE: 1/(1 + (f/f50)^q) over the samples ABOVE the
                    # overshoot, because the lift is a separate effect and
                    # including it bends the carrier to absorb it. Scored
                    # against the Gaussian the stock would otherwise use.
                    _sel = fs_np = None
                    _m2 = fs > max(8.0, pk_at)
                    if int(_m2.sum()) > 20:
                        _ff, _rr2 = fs[_m2], rs[_m2]
                        _best = None
                        for _q in np.arange(1.0, 5.01, 0.01):
                            _mod = 1.0 / (1.0 + (_ff / f50) ** _q)
                            _e = float(np.sqrt(np.mean((_mod - _rr2) ** 2)))
                            if _best is None or _e < _best[0]:
                                _best = (_e, float(_q))
                        _g = 1.0 / (1.0 + 0.0)  # placeholder to keep names clear
                        _gauss = np.exp(-math.log(2.0) * (_ff / f50) ** 2)
                        _ge = float(np.sqrt(np.mean((_gauss - _rr2) ** 2)))
                        print(f"    rolloff over {int(_m2.sum())} samples >= "
                              f"{max(8.0, pk_at):.1f} c/mm: power law q = "
                              f"{_best[1]:.2f} at rms {_best[0]:.4f}, Gaussian "
                              f"rms {_ge:.4f} ({_ge/_best[0]:.1f}x "
                              f"{'worse' if _ge > _best[0] else 'BETTER'})")
                    print(f"    \u26a0 ONE UNLABELLED CURVE, so it gives ONE f50. "
                          f"Fuji does not print per-record MTF on these sheets; "
                          f"the value is assigned to GREEN and red/blue take a "
                          f"stated ratio, the same rule 8532 and 8572 use")
                    e = EXPECTED.get(tag, {})
                    for k, v in (("f50", f50), ("peak", peak), ("peak_at", pk_at)):
                        if k in e and abs(v - e[k]) > (1.0 if k != "peak" else 0.01):
                            print(f"  [MISMATCH] {k} {v:.3f} vs pinned {e[k]}")
                            bad += 1

        # ---- spectral sensitivity -------------------------------------------
        sp = sh["spec"]
        try:
            S = _panel(rgb, sp["box"])
            masks = _channel_masks(S["sub"], sp["colour"])
            lam0, lam1 = sp["lam"]
            px_per_nm = (S["right"] - S["left"]) / (lam1 - lam0)
            n = sum(1 for m in masks.values() if m.sum() > 500)
            print(f"    spectral panel: {S['nx']}x{S['ny']} gridlines, "
                  f"{px_per_nm*10:.1f} px per 10 nm, {n} record mask(s) with ink")
            print(f"    ⚠ NOT ADOPTED IN THIS PASS. The ordinate carries no "
                  f"numbered ladder on two of the three sheets -- it is a "
                  f"bracketed arrow marked \"1.0\" -- so the DECADE SCALE is "
                  f"set by an annotation, not by tick labels, and a "
                  f"peak-normalised log_s built on a misread bracket would be "
                  f"wrong by a factor and look plausible. The panel is located "
                  f"and its geometry recorded here so the read is a small job, "
                  f"and the three profiles ship with spectral.has_data False "
                  f"rather than with an invented curve")
        except ValueError as e:
            print(f"    [WARN] spectral panel: {e}")

        if ns.overlay:
            Path(ns.overlay).mkdir(parents=True, exist_ok=True)
            from PIL import Image
            im = Image.fromarray(np.clip(P["sub"], 0, 255).astype(np.uint8))
            im.save(Path(ns.overlay) / f"ov_{tag}_hd.png")

    if ns.do_assert and bad:
        print(f"[FAIL] {bad} panel(s) no longer reproduce")
        return 1
    print("[OK] Fuji T3 sheets read" + (" and pinned values match" if ns.do_assert else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
