"""The 1998 Agfa Professional Films sheet, read as vector: 12 films, 4 panels each.

WHAT THIS SOURCE IS, AND WHY IT IS NOT THE SHEET WE ALREADY READ
----------------------------------------------------------------
`AGFA/agfa_films.pdf` -- Agfa-Gevaert, «Technical Data PF: Agfa range of films»,
**Date 09/1998, 1st edition** (printed on p12), 12 pages, PageMaker 6.5 ->
Distiller 3.01. Document metadata: title "Technical data PF", author "Sckaer
(Redaktion)", keywords "PF, professional, APX, Optima, RSX, Scala".

⚠ **`NotFound.md` AND QUEUE G6 RECORD THAT THE FOUR AGFA CANDIDATES ARE "THE
SAME PUBLICATION, TWO OF THEM BYTE-IDENTICAL". THAT IS HALF RIGHT AND THE HALF
THAT IS WRONG COST THIS PROJECT A DOCUMENT.** The md5s:

    agfa_films.pdf          edb3dd175821a6f9f2fd60bd43341bb4   1998, 1st ed
    AGFA stocks.pdf         bf9f0c1a85e42c3d50f60c00a9159690   2004, 4th ed
    FPD1e.pdf               bf9f0c1a85e42c3d50f60c00a9159690   2004, 4th ed
    Datasheet_F_PF_E4.pdf   f693b5626f160a2989b78fd1c78d081c   2004, 4th ed

The byte-identical PAIR is real. `agfa_films.pdf` is a DIFFERENT EDITION five
years older, with a different page count of plotted films, and it is the only
document in the corpus that plots **AGFACOLOR ULTRA 50** and **AGFAPAN APX 25**
at all. `agfa_2004_curves.py` reads the 2004 edition and knows nothing of it.

⚠ A SECOND STANDING CLAIM IS ALSO FALSE. `agfa_2004_curves.py`'s own docstring
says AGFA_OPTIMA_100's spectral set was traced "from a RASTER page of the older
1998 brochure". **This file has ZERO embedded images on all twelve pages** --
`page.get_images()` returns empty everywhere, `get_drawings()` returns 116-172
stroked objects on pp7-10. It was read as raster because nobody checked, which
is the same defect the APX spectral sets were corrected for on 2026-08-17
("superseding the 2026-08-02 visual transcription of the same plot").

WHAT THE 1998 EDITION PLOTS THAT THE 2004 EDITION DOES NOT
-----------------------------------------------------------
    printed p7   Optima II 100 / 200 / 400        (2004 has these, p7)
    printed p8   Portrait XPS 160 / ULTRA 50 / RSX II 50
    printed p9   RSX II 100 / RSX II 200 / SCALA 200x
    printed p10  APX 25 / APX 100 / APX 400

Nine of the twelve are films the 2004 reader never touches, and three of them
-- ULTRA 50, RSX II 50/100/200 -- have no profile in the database at all.

THE LEGEND, AND THE ONE PANEL WHERE IT DOES NOT APPLY
------------------------------------------------------
Colour pages use the same dash key as the Vista sheet and the 2004 sheet, and
in fact use the VISTA dash LENGTHS exactly (3.159/.79), not the 2004 sheet's:

    solid                 `[] 0`                        -> GREEN
    dashed                `[ 3.159 .79 ] 0`             -> BLUE
    dash-dot-dot          `[ 3.159 .79 .79 .79 ] 0`     -> RED

⚠ **p10 HAS NO DASHED STROKES AT ALL.** Every APX panel is monochrome, so there
is one curve per panel and the dash key is meaningless -- except on the
gamma-time panel, which draws FIVE developer curves and draws them ALL SOLID,
in a SINGLE path object. They are separated by `granularity_vector.subpaths`
and then identified by the printed developer names, ranked by curve height.

⚠ **STROKE WIDTHS DIFFER FROM THE 2004 SHEET AND THE CONSTANTS MUST BE
OVERRIDDEN.** 1998: curves 0.787-0.790 pt, frames 0.262-0.263 pt. 2004: 0.85
and 0.283. The ratio is 0.929, i.e. the 1998 pages are laid out at a different
scale, not redrawn. `agfa_2004_curves.CURVE_W`/`FRAME_W` are patched for the
duration of this reader rather than parameterised, so the 2004 audit's own
constants are untouched on disk.

WHAT IS ADOPTABLE AND WHAT IS NOT
---------------------------------
- **Spectral sensitivity**: adoptable. Three layers on colour films, one pan
  curve on APX and SCALA.
- **Characteristic / colour density curves**: adoptable -- dmin and gamma per
  channel from a six-parameter softplus re-fit.
- ⚠ **Spectral density on the NEGATIVE films is NOT three dyes.** The panel
  plots "Medium density" and "Minimum density" -- two aggregate curves. This is
  the schema-v14 NEUTRAL + D-MIN PAIR, not `d_cyan/d_magenta/d_yellow`, and
  `NotFound.md` already warns that the dye-density counter must not be
  "corrected" upward by conflating the two. Read, reported, stored only as the
  pair carrier if the owner approves.
- ✔ **Spectral density on the RSX II REVERSAL films IS three dyes.** Those
  panels print "Yellow / Magenta / Cyan / Visual grey" -- four labelled curves,
  three of them the separated dyes. This is a genuine `SpectralDyeDensity`
  source and the first Agfa one in the corpus.
- ⚠ **Sharpness is a CTF, not an MTF, and this is the G6 question.** Agfa plot
  "Transfer factor (%)" against "Lines (mm)" -- note the 1998 axis says "Lines
  (mm)" where the 2004 axis says "Lines per mm" -- and the curve exceeds 100 %
  at low frequency. f50 is still well defined and the overshoot is the measured
  adjacency. **The APX columns are what settles G6**: the same page prints a
  resolving power (200/150/110 lines/mm at 1000:1) for the same film, so the
  ratio f50 : RP discriminates cycles from half-cycles. Reported, not adopted.
- **Gamma-time**: five developers x three APX films = fifteen curves, the
  `ProcessingFamily` carrier's exact shape. Adoptable.

Run:  python agfa_1998_curves.py --root <corpus> [--assert] [--emit]
Needs numpy + PyMuPDF. SciPy is OPTIONAL: without it the six-parameter re-fit is
skipped and said to be skipped; every other reading still runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pymupdf
except ImportError:                                       # pragma: no cover
    print("[!] pymupdf not installed:  pip install pymupdf")
    raise SystemExit(1)

import agfa_2004_curves as A4
from granularity_vector import subpaths

SHEET = "AGFA/agfa_films.pdf"

SOURCE = ("Agfa-Gevaert, «Technical Data PF -- Agfa range of films», "
          "1st edition, 09/1998 -- PDF/PROFILES/AGFA/agfa_films.pdf. "
          "Vector line art on every page, real text layer, zero embedded "
          "images. NOT the same document as 'AGFA stocks.pdf' / FPD1e.pdf / "
          "Datasheet_F_PF_E4.pdf, which are the 4th edition of 08/2004.")

# ⚠ The 1998 pages are drawn at 0.929 x the 2004 scale. Patch the shared
# reader's width constants for the duration of this module.
A4.CURVE_W, A4.FRAME_W, A4.W_TOL = 0.789, 0.2625, 0.02

Panel = A4.Panel
_fit = A4._fit
_words = A4._words
_axis_labels = A4._axis_labels
curves_in = A4.curves_in


def _frames(page, band, xlo, xhi):
    """Frame rect of FRAME_W width inside a band. Local copy: the 2004 helper
    hard-codes a 0.02 tolerance that the 1998 sheet's 0.262/0.263 pair needs
    widened, and hard-codes `>= ylo` where p9's SCALA column sits 0.3 pt high."""
    ylo, yhi = band
    best = None
    for dr in page.get_drawings():
        if dr["type"] not in ("s", "fs") or not A4._is_ink(dr):
            continue
        if abs((dr.get("width") or 0.0) - A4.FRAME_W) > 0.006:
            continue
        r = dr["rect"]
        if not (ylo - 2 <= r.y0 <= yhi and xlo - 4 <= r.x0 and r.x1 <= xhi + 4):
            continue
        a = r.width * r.height
        if a < 1500.0:
            continue
        if best is None or a > best[0]:
            best = (a, r)
    return None if best is None else best[1]


#: (profile_or_None, printed name, printed page, x band, kind)
#: `kind` selects the panel set: "colour_neg", "reversal", "mono", "bw_rev".
COLUMNS = (
    ("AGFA_OPTIMA_100",   "AGFACOLOR OPTIMA II 100",   7, (44.0, 202.0), "colour_neg"),
    ("AGFA_OPTIMA_200",   "AGFACOLOR OPTIMA II 200",   7, (220.0, 378.5), "colour_neg"),
    ("AGFA_OPTIMA_400",   "AGFACOLOR OPTIMA II 400",   7, (396.0, 554.5), "colour_neg"),
    ("AGFA_PORTRAIT_160", "AGFACOLOR PORTRAIT XPS 160", 8, (27.0, 185.0), "colour_neg"),
    (None,                "AGFACOLOR ULTRA 50",         8, (203.0, 361.5), "colour_neg"),
    (None,                "AGFACHROME RSX II 50",       8, (379.0, 537.5), "reversal"),
    (None,                "AGFACHROME RSX II 100",      9, (44.0, 202.0), "reversal"),
    (None,                "AGFACHROME RSX II 200",      9, (220.0, 378.5), "reversal"),
    ("AGFA_SCALA_200X",   "AGFA SCALA 200x",            9, (396.0, 554.5), "bw_rev"),
    ("AGFA_APX_25",       "AGFAPAN APX 25",            10, (27.0, 185.0), "mono"),
    ("AGFA_APX_100",      "AGFAPAN APX 100",           10, (203.0, 361.5), "mono"),
    ("AGFA_APX_400",      "AGFAPAN APX 400",           10, (379.0, 537.5), "mono"),
)

#: Panel y bands. Constant across pp7-10 for the three regular column layouts.
BANDS = {
    "spectral":  (78.0, 176.0),
    "density":   (225.0, 292.0),     # spectral density | characteristic curve
    "sharpness": (337.0, 427.0),
    "curves":    (474.0, 580.0),     # colour density | gamma-time
}

#: ⚠ SCALA's column does not use them for two of its three panels. It is a
#: monochrome reversal film, so it has NO spectral-density panel, and Agfa did
#: not leave a gap: the sharpness panel moves up into the density slot and the
#: density curves move up into the sharpness slot. Its spectral panel alone
#: stays put. Measured from its own printed labels:
#:      spectral    x ticks 400-700 at y 179.8   -> the normal slot
#:      sharpness   x ticks 2-100  at y 317.3    -> the density slot
#:      curves      x ticks -3..+3 at y 447.3    -> the sharpness slot
#: Handled as a special case rather than by widening the shared bands, which
#: would make the panels ambiguous on the other eleven columns.
SCALA_BANDS = {
    "spectral":  (78.0, 176.0),
    "sharpness": (220.0, 332.0),
    "curves":    (370.0, 462.0),
}

#: The five processing steps Agfa plot on SCALA's density-curve panel.
SCALA_STEPS = ("Pull 1", "Standard", "Push 1", "Push 2", "Push 3")


# ---------------------------------------------------------------------------
#  Panel builders
# ---------------------------------------------------------------------------

#: Half-width of the containment box, in axis units beyond the outermost
#: printed label. ⚠ IT HAS TO BE NON-ZERO: Agfa let curves run past their own
#: end labels on several panels -- the APX 25 gamma-time set starts left of the
#: "2 min" tick and the colour-density red record dips below its frame.
PAD_PT = 12.0


def _fit_robust(pairs, tol, axis):
    """Least squares with iterative worst-point rejection.

    ⚠ TWO REAL DEFECTS ON THIS SHEET NEED IT, AND ONE OF THEM IS SILENT.

    1. **THE CO-LINEARITY CLUSTER CHAINS ACROSS AXES.** `_axis_labels` groups a
       column by walking sorted x with a 4 pt tolerance. On the p10 characteristic
       -curve panel the ordinate reads 3.0/2.0/1.0/0 right-aligned at x 47.2, and
       the abscissa's leading label "-4.0" is centred at x 52.4 -- 5.2 pt from the
       column, which is outside the tolerance, but the ordinate's own "0" sits at
       49.6 because it is one glyph instead of three, and 47.2 -> 49.6 -> 52.4 is
       a chain of two 2.4-2.9 pt steps. The abscissa label joins the ordinate and
       the fit returns 2.22 D of residual on a 3 D axis. It is not a crash: the
       curve still traces, and every density it returns is wrong.

    2. **A SINGLE-GLYPH "0" IS NOT WHERE ITS NEIGHBOURS ARE.** Even with the
       intruder gone, "0" on a right-aligned ordinate or a centred abscissa sits
       1.3-2.4 pt off the line the multi-glyph labels define. That is 0.06 of a
       decade -- harmless alone, but it is the residual that would otherwise mask
       defect 1 in the reported number.

    Rejection stops at three points, so a genuinely broken axis reports a large
    residual rather than being whittled into a two-point fit that cannot fail.
    """
    seen, uniq = set(), []
    for v, t in pairs:
        k = (round(v, 6), round(t, 3))
        if k not in seen:
            seen.add(k)
            uniq.append((v, t))
    kept, dropped = list(uniq), []
    while len(kept) > 3:
        (m, c), res = _fit(kept)
        if res <= tol:
            break
        i = int(np.argmax([abs(m * t + c - v) for v, t in kept]))
        dropped.append(kept.pop(i))
    (m, c), res = _fit(kept)
    return (m, c), res, len(kept), dropped


def _calibrated(page, ws, name, band, xlo, xhi, logx=False, logy=False):
    """Build a Panel from the panel's OWN PRINTED LABELS, not from a frame rect.

    ⚠ THE STROKED FRAME IS THE WRONG REFERENCE ON THIS SHEET AND USING IT LOSES
    CURVES. On p10 the characteristic-curve panel's only 0.263 pt rect spans
    x 64.3-179.7 while the curve itself is drawn from x 54.4, because that rect
    is an inner grid box and the plot area starts a decade to its left. Frame-
    based containment silently returned "no curve" on six of the twelve columns.
    The labels are the axis: they are what the calibration is fitted to anyway,
    so the containment box is derived from them and the frame is not consulted.

    The x row and the y column are separated by `_axis_labels`' co-linearity
    clustering over the whole column window, which is why no per-panel label
    windows are needed here.
    """
    xs = _axis_labels(ws, xlo, xhi, band[0], band[1] + 22.0, "x")
    ys = _axis_labels(ws, xlo, xhi, band[0] - 8.0, band[1] + 8.0, "y")
    if logx:
        xs = [(np.log10(v), t) for v, t in xs if v > 0]
    if logy:
        ys = [(np.log10(v), t) for v, t in ys if v > 0]
    if len(xs) < 3 or len(ys) < 3:
        return None, f"labels x={len(xs)} y={len(ys)}"
    # Tolerance is in AXIS units: 0.02 of a decade on a log axis, else 1.5 % of
    # the labelled span. Both are far below any real defect and far above the
    # sub-point noise of a correctly grouped row.
    xtol = 0.02 if logx else 0.015 * (max(v for v, _ in xs) - min(v for v, _ in xs))
    ytol = 0.02 if logy else 0.015 * (max(v for v, _ in ys) - min(v for v, _ in ys))
    xm, xr, nx, xdrop = _fit_robust(xs, xtol, "x")
    ym, yr, ny, ydrop = _fit_robust(ys, ytol, "y")
    # ⚠ THE PAD IS ONE LABEL INTERVAL, NOT A CONSTANT, BECAUSE AGFA LABEL FEWER
    # TICKS THAN THEY PLOT. The Optima II 400 colour-density abscissa prints
    # -4.0 .. 0 and then draws every record a full decade further, out to +1.0
    # -- 17 pt past the last printed label. A fixed 12 pt pad clipped all three
    # records out of their own panel and the column reported "0 sub-paths".
    px = sorted(t for _, t in xs)
    py = sorted(t for _, t in ys)
    padx = max(PAD_PT, 1.0 * float(np.median(np.diff(px))) if len(px) > 1 else 0)
    pady = max(PAD_PT, 1.0 * float(np.median(np.diff(py))) if len(py) > 1 else 0)
    rect = pymupdf.Rect(px[0] - padx, py[0] - pady, px[-1] + padx, py[-1] + pady)
    p = Panel(name, rect, xm, ym, xr, yr)
    p.nx, p.ny, p.xdrop, p.ydrop = nx, ny, xdrop, ydrop
    return p, None


def _panel(page, ws, name, band, xlo, xhi):
    return _calibrated(page, ws, name, band, xlo, xhi)


def _log_panel(page, ws, name, band, xlo, xhi):
    return _calibrated(page, ws, name, band, xlo, xhi, logx=True, logy=True)


def _is_curve_stroke(dr) -> bool:
    """Data curve, as opposed to a frame, grid line or tick.

    ⚠ **WIDTH DOES NOT WORK ON THIS SHEET AND USING IT LOSES WHOLE PANELS.**
    `agfa_2004_curves` separates data from furniture with one number, 0.85 pt
    against 0.283 pt, and that is correct for the 2004 sheet. The 1998 sheet
    draws each panel at whatever scale the layout gave it:

        p7  Optima curves         curves 0.789   frames 0.263 / 0.526
        p8  Portrait spectral     curves 0.503   frames 0.263 / 0.527
        p10 APX gamma-time        curves 0.789   frames 0.263 / 0.526

    On the Portrait spectral panel the CURVES ARE THINNER THAN ONE OF THE
    FRAMES, so no width threshold can separate them and a fixed 0.789 returned
    an empty panel. Shape does separate them, everywhere: Agfa draw every data
    curve as beziers and every piece of furniture as `re` or straight `l`
    segments. Dashes are accepted too, because the blue and red records on the
    colour panels are dashed by definition and that is already proof of data.
    """
    if not A4._is_ink(dr) or dr["type"] not in ("s", "fs"):
        return False
    items = dr["items"]
    if any(it[0] == "c" for it in items):
        return True
    if _layer_of_local(dr.get("dashes")) in ("b", "r"):
        return True
    # ⚠ SOME RECORDS ARE SOLID POLYLINES WITH NO BEZIER AND NO DASH, and they
    # are the ones a bezier-only test silently drops. The RSX II 50 and RSX II
    # 100 GREEN spectral records are drawn as `l` runs; both columns came back
    # with blue and red only, which looks like a two-layer panel rather than a
    # failure. A grid line is one `l` and a frame is an `re` or four; a data
    # polyline on this sheet is 30+ segments and is not axis-aligned.
    if len(items) < 6 or any(it[0] != "l" for it in items):
        return False
    pts = np.array([[it[1].x, it[1].y, it[2].x, it[2].y] for it in items], float)
    moves = np.abs(pts[:, 2:] - pts[:, :2])
    axis_aligned = ((moves[:, 0] < 0.05) | (moves[:, 1] < 0.05)).mean()
    return bool(axis_aligned < 0.5)


def _layer_of_local(dashes):
    s = str(dashes or "").strip()
    if s in ("", "[] 0", "[ ] 0", "None"):
        return "g"
    if "[" not in s or "]" not in s:
        return None
    return {2: "b", 4: "r"}.get(len(s[s.find("[") + 1:s.find("]")].split()))


def _curves_by_shape(page, panel, min_pts=8):
    """{layer: (x_pt, y_pt)} using shape, not width, to find the data curves."""
    r = panel.rect
    out: dict[str, tuple] = {}
    for dr in page.get_drawings():
        if not _is_curve_stroke(dr):
            continue
        q = dr["rect"]
        if not (q.x0 >= r.x0 - 3 and q.x1 <= r.x1 + 3
                and q.y0 >= r.y0 - 3 and q.y1 <= r.y1 + 11):
            continue
        key = _layer_of_local(dr.get("dashes"))
        if key is None:
            continue
        for sp in subpaths(dr["items"]):
            if len(sp) < min_pts:
                continue
            arr = np.array(sp, float)
            arr = arr[np.argsort(arr[:, 0])]
            arr = arr[np.concatenate(([True], np.diff(arr[:, 0]) > 1e-9))]
            if len(arr) < min_pts:
                continue
            if key in out and len(out[key][0]) >= len(arr):
                continue
            out[key] = (arr[:, 0], arr[:, 1])
    return out


def _one_curve(page, panel):
    """The longest data curve inside a panel, for the monochrome pages."""
    best = None
    for xs, ys in _split_curves(page, panel):
        if best is None or len(xs) > len(best[0]):
            best = (xs, ys)
    return best


def _split_curves(page, panel, min_pts=8):
    """Every disjoint sub-path inside a panel, x-sorted. For the gamma-time
    panel, whose five developer curves share one path object and one dash key."""
    r = panel.rect
    out = []
    for dr in page.get_drawings():
        if not _is_curve_stroke(dr):
            continue
        q = dr["rect"]
        if not (q.x0 >= r.x0 - 3 and q.x1 <= r.x1 + 3
                and q.y0 >= r.y0 - 3 and q.y1 <= r.y1 + 11):
            continue
        for sp in subpaths(dr["items"]):
            if len(sp) < min_pts:
                continue
            arr = np.array(sp, float)
            o = np.argsort(arr[:, 0])
            arr = arr[o]
            keep = np.concatenate(([True], np.diff(arr[:, 0]) > 1e-9))
            arr = arr[keep]
            if len(arr) >= min_pts:
                out.append((arr[:, 0], arr[:, 1]))
    return out


#: The five developers Agfa plot on every APX gamma-time panel, as printed.
DEVELOPERS = ("REFINAL", "RODINAL 1+25", "RODINAL 1+50",
              "RODINAL SPECIAL", "STUDIONAL LIQUID")


def _named_labels(page, panel, names):
    """Bounding boxes of a known set of printed curve labels inside a panel."""
    ws = [(w[4], w[0], w[1], w[2], w[3]) for w in page.get_text("words")]
    r = panel.rect
    inside = [w for w in ws
              if r.x0 - 6 <= w[1] and w[3] <= r.x1 + 30
              and r.y0 - 6 <= w[2] and w[4] <= r.y1 + 6]
    out = {}
    for name in names:
        toks = name.split()
        for i in range(len(inside) - len(toks) + 1):
            run = inside[i:i + len(toks)]
            if [t[0] for t in run] != toks:
                continue
            # one printed label may wrap onto a second line; keep the union
            box = (min(t[1] for t in run), min(t[2] for t in run),
                   max(t[3] for t in run), max(t[4] for t in run))
            if abs(box[3] - box[1]) > 14.0:          # two stacked lines, not one label
                continue
            out[name] = box
            break
    return out


def _assign(labels, segs):
    """Match printed labels to curves, minimising total box distance.

    ⚠ **GREEDY NEAREST-NEIGHBOUR DOUBLE-CLAIMS AND IT DOES IT SILENTLY.** On
    SCALA's density panel the five step names are stacked 5.1 pt apart at almost
    one x, so "Pull 1" and "Standard" are both 1.3 pt from the same curve and
    both 1.3 pt from the next. Greedy gave that curve two names and left the
    neighbour "UNLABELLED", which reads as a tracing failure rather than as the
    tie it is. Solving the whole assignment turns a 0.0 pt tie-break into the
    globally consistent answer, and the result is independently confirmed by the
    sheet's own push/pull table -- "maximum density: decreasing [push] /
    increasing [pull]" -- which requires exactly the D-max ordering it returns.

    Where the counts differ the assignment is still solved over the smaller set
    and the leftovers are reported, because a curve legitimately carries two
    names when two developers share it (RODINAL SPECIAL and STUDIONAL LIQUID).
    """
    import itertools
    names = list(labels)
    cost = np.array([[_box_distance(labels[n], xs, ys) for xs, ys in segs]
                     for n in names], float)
    owners: dict[int, list[str]] = {i: [] for i in range(len(segs))}
    if len(names) <= len(segs) and len(names) <= 8:
        best, bestcost = None, float("inf")
        for perm in itertools.permutations(range(len(segs)), len(names)):
            c = sum(cost[i, perm[i]] for i in range(len(names)))
            if c < bestcost:
                best, bestcost = perm, c
        for i, n in enumerate(names):
            j = best[i]
            alt = sorted(cost[i])
            owners[j].append(f"{n} ({cost[i, j]:.1f}/{alt[1]:.1f} pt)"
                             if len(alt) > 1 else f"{n} ({cost[i, j]:.1f} pt)")
    else:
        for i, n in enumerate(names):
            j = int(np.argmin(cost[i]))
            alt = sorted(cost[i])
            owners[j].append(f"{n} ({alt[0]:.1f}/{alt[1]:.1f} pt)")
    return owners


def _box_distance(box, xs, ys):
    """Shortest distance from a polyline to a label's bounding box, 0 if it enters.

    ⚠ NEITHER EDGE ALONE IDENTIFIES THESE CURVES, WHICH IS WHY THIS IS A BOX AND
    NOT A POINT. Agfa place three of the five APX 25 labels RIGHT-ALIGNED to the
    top end of their curve ("RODINAL SPECIAL" ends at x 93.9, the curve tops out
    at 93.47) and the other two LEFT-ALIGNED to a crossing part-way down
    ("RODINAL 1+25" starts at x 114.3, its curve passes x 112.8 at that height).
    Matching on the left edge assigns REFINAL to the wrong curve by 8.8 pt while
    the right edge misses RODINAL 1+25 by 21 pt. The box touches both.
    """
    x0, y0, x1, y1 = box
    dx = np.clip(xs, x0, x1) - xs
    dy = np.clip(ys, y0, y1) - ys
    return float(np.min(np.hypot(dx, dy)))


#: Films whose columns feed the database, and the panel set each carries.
EMIT_COLOURS = ("AGFACOLOR OPTIMA II 100", "AGFACOLOR OPTIMA II 200",
                "AGFACOLOR OPTIMA II 400", "AGFACOLOR PORTRAIT XPS 160",
                "AGFACOLOR ULTRA 50", "AGFACHROME RSX II 50",
                "AGFACHROME RSX II 100", "AGFACHROME RSX II 200")


def _fit_tone_ms(panel, xy, seeds=None):
    """`agfa_2004_curves.fit_tone` with multiple starts, keeping the best rms.

    ⚠ **A SINGLE SEED FINDS A LOCAL MINIMUM ON ONE RECORD IN TWELVE AND THE
    RESULT IS IN RANGE.** RSX II 100's BLUE record fitted to gamma 1.404 with
    toe -2.574 and shoulder +2.734 -- knees 5.3 decades apart on a curve drawn
    over 5.0 -- at rms 0.139 D, against 0.009-0.014 D for every other record on
    the same page. Nothing about 1.404 looks wrong for a reversal film; only the
    residual gives it away, and only when compared with its own siblings. Three
    seeds spanning the plausible gamma range remove it, and the chosen fit is
    reported with its rms so a future regression is visible.
    """
    best = None
    for g0, tx0, sx0 in (seeds or ((0.65, -2.2, 1.6), (1.9, -0.7, 0.9),
                                   (2.4, -0.5, 0.7))):
        got = _fit_once(panel, xy, g0, tx0, sx0)
        if got is not None and (best is None or got[-1] < best[-1]):
            best = got
    return best


def _fit_once(panel, xy, g0, tx0, sx0):
    try:
        from scipy.optimize import least_squares
    except ImportError:                                   # pragma: no cover
        return None
    le = np.asarray(panel.X(xy[0]), float)
    d = np.asarray(panel.Y(xy[1]), float)
    o = np.argsort(le)
    le, d = le[o], d[o]

    def sp(z, k):
        u = z / k
        return k * np.where(u > 30.0, u, np.log1p(np.exp(np.minimum(u, 30.0))))

    def model(p):
        dmin, g, tx, tk, sx, ratio = p
        return dmin + g * (sp(le - tx, tk) - sp(le - sx, ratio * tk))

    p0 = np.array([float(d.min()), g0, tx0, 0.30, sx0, 1.30])
    lo = np.array([0.0, 0.20, -6.0, 0.05, -2.0, 0.50])
    # ⚠ gamma's ceiling stays at `agfa_2004_curves.fit_tone`'s 2.50, not higher.
    # Raising it to 3.00 to see whether the fit wanted more improved RSX II 50's
    # red rms from 0.013 to 0.011 D -- a fifth of nothing -- and pushed `gamma`
    # to exactly 3.000, i.e. onto the bound and outside the 1.6-2.1 band the
    # ToneCurve docstring gives for colour reversal. A parameter resting on its
    # own limit is not a measurement. The multi-start above is what fixed the
    # one genuinely bad record; the ceiling was never the problem.
    hi = np.array([3.0, 2.50, 2.0, 2.00, 8.0, 1.40])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
    r = least_squares(lambda p: model(p) - d, p0, bounds=(lo, hi), max_nfev=20000)
    dmin, g, tx, tk, sx, ratio = r.x
    rms = float(np.sqrt(np.mean((model(r.x) - d) ** 2)))
    return (float(dmin), float(g), float(tx), float(tk),
            float(sx), float(ratio * tk), rms)


class _FlipX:
    """A Panel view whose abscissa runs backwards.

    ⚠ **REVERSAL CURVES CANNOT BE FITTED IN THE SHEET'S OWN FRAME AND THE
    FAILURE IS SILENT.** `ToneCurve`'s docstring is explicit: for reversal
    stocks `x` is NEGATED log exposure, so density falls as light rises and
    `toe_x` controls the highlight end. Fed the sheet's ascending abscissa,
    `agfa_2004_curves.tone_of` looks for the steepest RISING chord and returns
    ~0.00 for every RSX II and SCALA record, while `fit_tone` drives its
    parameters into the bounds and reports `dmin` 3.0 with `gamma` 2.0-2.5 --
    numbers that are in range for a reversal film and completely wrong. Negating
    the abscissa puts the curve back in the frame the schema defines.
    """

    def __init__(self, panel):
        self._p = panel
        self.name, self.rect = panel.name, panel.rect
        self.xres, self.yres = panel.xres, panel.yres

    def X(self, pt):
        return -self._p.X(pt)

    def Y(self, pt):
        return self._p.Y(pt)


#: Straight-line midpoint the reversal family already uses. Measured across the
#: stored reversal stocks: SCALA 200x +0.08, EKTACHROME 64 +0.08, EKTACHROME
#: 5239/7239 +0.09, DUFAYCOLOR 1937 +0.11. The negative family sits at +0.10,
#: which `agfa_2004_curves.fit_tone` hard-codes; re-anchoring a reversal curve
#: there instead would shift every one by 0.02 for no reason.
REVERSAL_MIDPOINT = 0.08


def tone_reversal(panel, xy):
    """(dmin, gamma, span) and a six-parameter fit for one reversal record."""
    fp_ = _FlipX(panel)
    dmin, chord, span = A4.tone_of(fp_, xy)
    ft = _fit_tone_ms(fp_, xy)
    if ft is not None:
        dmn, g, tx, tk, sx, sk, rms = ft
        shift = REVERSAL_MIDPOINT - 0.5 * (tx + sx)
        ft = (dmn, g, tx + shift, tk, sx + shift, sk, rms)
    return dmin, chord, span, ft


#: Printed curve labels on the Spectral density panel. ⚠ THE TWO FILM CLASSES
#: PLOT DIFFERENT QUANTITIES UNDER THE SAME HEADING, and conflating them is the
#: error `NotFound.md` warns about when it says the dye-density counter "must
#: not be corrected upward". A colour NEGATIVE panel draws two AGGREGATE
#: curves -- the film's total transmission at two exposure levels -- which is
#: the schema-v14 neutral + D-min pair, NOT three dyes. A reversal panel draws
#: the three SEPARATED dyes plus a visual-grey reference, which is a genuine
#: `SpectralDyeDensity`. Same panel title, two different carriers.
DYE_LABELS = {
    "colour_neg": ("Medium density", "Minimum density"),
    "reversal":   ("Yellow", "Magenta", "Cyan", "Visual grey"),
}

#: Where each printed label's samples are stored.
DYE_FIELD = {
    "Medium density": "d_neutral", "Minimum density": "d_dmin",
    "Yellow": "d_yellow", "Magenta": "d_magenta", "Cyan": "d_cyan",
    "Visual grey": "d_neutral",
}


def read_dye(page, ws, xlo, xhi, kind, lam0=400.0, step=10.0, n=31):
    """The Spectral density panel, resolved to named curves and resampled.

    ⚠ THE DASH KEY IS MEANINGLESS ON THIS PANEL and using it silently mislabels
    every dye. Elsewhere on the sheet solid/dashed/dash-dot-dot means
    green/blue/red; here the curves are named in print -- Yellow, Magenta,
    Cyan, Visual grey -- and the naming does not follow the dash convention.
    Read through `_curves_by_shape` the RSX II 50 panel returns "b/g/r" for what
    are actually the yellow, cyan and magenta dyes. The labels are matched by
    bounding-box distance, the same way the gamma-time developers are.
    """
    band = BANDS["density"]
    p, err = _calibrated(page, ws, "dye", band, xlo, xhi)
    if p is None:
        return None, err
    segs = _split_curves(page, p)
    want = DYE_LABELS.get(kind, ())
    labels = _named_labels(page, p, want)
    if not segs or not labels:
        return None, "segs=%d labels=%d" % (len(segs), len(labels))
    owners = _assign(labels, segs)
    grid = lam0 + step * np.arange(n)
    out = {"residual": [round(p.xres, 4), round(p.yres, 4)], "curves": {}}
    for i, (xs, ys) in enumerate(segs):
        if not owners[i]:
            continue
        name = owners[i][0].split(" (")[0]
        lam, dv = p.X(xs), p.Y(ys)
        o = np.argsort(lam)
        lam, dv = lam[o], dv[o]
        vals = [None if (g < lam.min() - 1e-6 or g > lam.max() + 1e-6)
                else float(np.interp(g, lam, dv)) for g in grid]
        out["curves"][name] = {
            "field": DYE_FIELD[name],
            "drawn_nm": [round(float(lam.min()), 1), round(float(lam.max()), 1)],
            "d": [None if v is None else round(max(0.0, v), 4) for v in vals],
            "peak_nm": round(float(grid[int(np.nanargmax([
                -1e9 if v is None else v for v in vals]))]), 1),
        }
    return out, None


def emit(doc, path):
    """Write every adoptable reading to JSON, for the adoption script to splice.

    ⚠ THE SPLICE IS A SEPARATE STEP ON PURPOSE. This reader is an AUDIT: it must
    be able to run against the source and disagree with the database. If it wrote
    `film_profiles.py` directly it could only ever agree with itself, and the
    build's audit stage would be checking the source against a copy of its own
    last output.
    """
    import json
    out = {"source": SOURCE, "films": {}}
    for profile, printed, pageno, (xlo, xhi), kind in COLUMNS:
        pg = doc[pageno - 1]
        ws = _words(pg)
        rec = {"printed": printed, "profile": profile, "page": pageno,
               "kind": kind}
        bands = SCALA_BANDS if kind == "bw_rev" else BANDS

        # -- spectral sensitivity -------------------------------------------
        p, _ = _panel(pg, ws, "spectral", bands["spectral"], xlo, xhi)
        if p is not None:
            if kind in ("colour_neg", "reversal"):
                lay = {}
                for k, xy in _curves_by_shape(pg, p).items():
                    grid, vals, peak, span = A4.sample_spectral(p, xy)
                    lay[k] = {"log_s": [round(float(v), 3) for v in vals],
                              "peak_nm": round(float(grid[int(np.argmax(vals))]), 1),
                              "drawn_nm": [round(span[0], 1), round(span[1], 1)]}
                rec["spectral"] = lay
            else:
                xy = _one_curve(pg, p)
                if xy is not None:
                    grid, vals, peak, span = A4.sample_spectral(p, xy)
                    rec["spectral"] = {"pan": {
                        "log_s": [round(float(v), 3) for v in vals],
                        "drawn_nm": [round(span[0], 1), round(span[1], 1)]}}
            rec["spectral_residual"] = [round(p.xres, 4), round(p.yres, 4)]

        # -- sharpness -------------------------------------------------------
        p, _ = _log_panel(pg, ws, "sharpness", bands["sharpness"], xlo, xhi)
        if p is not None:
            xy = (_one_curve(pg, p) if kind in ("mono", "bw_rev")
                  else _curves_by_shape(pg, p).get("g"))
            if xy is not None:
                peak, f50, fpk = A4.sharpness_of(p, xy)
                rec["sharpness"] = {"peak_pct": round(peak, 1),
                                    "f50_lines_mm": None if f50 is None else round(f50, 1),
                                    "peak_at": round(fpk, 2),
                                    "adjacency": round(peak / 100.0 - 1.0, 4),
                                    "residual": [round(p.xres, 4), round(p.yres, 4)]}

        # -- characteristic / colour density ---------------------------------
        key = "curve" if kind == "mono" else "curves"
        band = bands["curves"] if kind == "bw_rev" else BANDS[
            "density" if kind == "mono" else "curves"]
        p, _ = _panel(pg, ws, key, band, xlo, xhi)
        if p is not None:
            if kind in ("colour_neg", "reversal"):
                lay = {}
                for k, xy in _curves_by_shape(pg, p).items():
                    if kind == "reversal":
                        dmin, g, span, ft = tone_reversal(p, xy)
                    else:
                        dmin, g, span = A4.tone_of(p, xy)
                        ft = A4.fit_tone(p, xy)
                    dv = p.Y(xy[1])
                    lay[k] = {"dmin": round(dmin, 4), "chord_gamma": round(g, 4),
                              "dmax": round(float(np.max(dv)), 4),
                              "logE_span": round(span, 3),
                              "fit": None if ft is None else
                                     [round(v, 4) for v in ft]}
                rec["density"] = lay
            elif kind == "mono":
                xy = _one_curve(pg, p)
                if xy is not None:
                    dmin, g, span = A4.tone_of(p, xy)
                    ft = A4.fit_tone(p, xy)
                    d = p.Y(xy[1])
                    rec["density"] = {"pan": {
                        "dmin": round(dmin, 4), "chord_gamma": round(g, 4),
                        "dmax": round(float(np.max(d)), 4),
                        "logE_span": round(span, 3),
                        "fit": None if ft is None else [round(v, 4) for v in ft]}}
            else:                                     # SCALA push/pull family
                segs = _split_curves(pg, p)
                owners = _assign(_named_labels(pg, p, SCALA_STEPS), segs)
                steps = {}
                for i, (xs, ys) in enumerate(segs):
                    nm = (owners[i][0].split(" (")[0] if owners[i]
                          else f"unlabelled {i}")
                    d = p.Y(ys)
                    dmin, g, span, ft = tone_reversal(p, (xs, ys))
                    steps[nm] = {"dmin": round(dmin, 4),
                                 "dmax": round(float(np.max(d)), 4),
                                 "chord_gamma": round(g, 4),
                                 "fit": None if ft is None else
                                        [round(v, 4) for v in ft]}
                rec["push_family"] = steps
            rec["density_residual"] = [round(p.xres, 4), round(p.yres, 4)]

        # -- spectral density (dye) ------------------------------------------
        if kind in ("colour_neg", "reversal"):
            dye, derr = read_dye(pg, ws, xlo, xhi, kind)
            rec["dye"] = dye if dye is not None else {"error": derr}

        # -- gamma-time -------------------------------------------------------
        if kind == "mono":
            p, _ = _panel(pg, ws, "gamma", BANDS["curves"], xlo, xhi)
            if p is not None:
                segs = _split_curves(pg, p)
                owners = _assign(_named_labels(pg, p, DEVELOPERS), segs)
                fam = []
                for i, (xs, ys) in enumerate(segs):
                    t, g = p.X(xs), p.Y(ys)
                    names = [o.split(" (")[0] for o in owners[i]]
                    fam.append({"developers": names,
                                "t_min": round(float(t.min()), 2),
                                "t_max": round(float(t.max()), 2),
                                "samples": [[round(float(tt), 2),
                                             round(float(np.interp(tt, t, g)), 4)]
                                            for tt in np.arange(
                                                np.ceil(t.min() * 2) / 2,
                                                t.max() + 1e-9, 0.5)]})
                rec["gamma_time"] = fam
                rec["gamma_time_residual"] = [round(p.xres, 4), round(p.yres, 4)]

        out["films"][printed] = rec
    Path(path).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"  [emit] {len(out['films'])} films -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on the printed name")
    ap.add_argument("--emit", default="", help="write every reading to this JSON")
    ns = ap.parse_args()

    pdf = Path(ns.root).resolve() / "PDF" / "PROFILES" / SHEET
    if not pdf.is_file():
        print(f"  [SKIP] source not present: {pdf}")
        return 0

    doc = pymupdf.open(pdf)
    print(f"[i] {SOURCE}\n")

    # Prove the two claims this reader rests on, every run.
    imgs = sum(len(doc[i].get_images(full=True)) for i in range(doc.page_count))
    if imgs:
        print(f"  [FAIL] expected zero embedded images, found {imgs}")
        return 1
    tail = doc[11].get_text()
    if "09/1998" not in tail or "1st edition" not in tail:
        print("  [FAIL] p12 does not print 'Date: 09/1998 / 1st edition'")
        return 1
    print(f"  [OK  ] 12 pages, {imgs} embedded images, p12 prints 09/1998 1st edition\n")

    for profile, printed, pageno, (xlo, xhi), kind in COLUMNS:
        if ns.only and ns.only.lower() not in printed.lower():
            continue
        pg = doc[pageno - 1]
        ws = _words(pg)
        tag = profile or "(no profile)"
        print(f"  {printed}   [{tag}]   printed p{pageno}")
        if kind == "mono":
            _read_mono(pg, ws, xlo, xhi)
        elif kind == "bw_rev":
            _read_scala(pg, ws, xlo, xhi)
        else:
            _read_colour(pg, ws, xlo, xhi, kind)
        print()
    if ns.emit:
        emit(doc, ns.emit)
    return 0


def _read_mono(pg, ws, xlo, xhi):
    """APX column: spectral sensitivity, characteristic curve, sharpness,
    gamma-time."""
    # -- spectral sensitivity ------------------------------------------------
    p, err = _panel(pg, ws, "spectral", BANDS["spectral"], xlo, xhi)
    if p is None:
        print(f"    spectral   [--] {err}")
    else:
        xy = _one_curve(pg, p)
        if xy is None:
            print("    spectral   [--] no curve")
        else:
            lam, lg = p.X(xy[0]), p.Y(xy[1])
            print(f"    spectral   {lam.min():.0f}-{lam.max():.0f} nm, "
                  f"peak {lam[int(np.argmax(lg))]:.0f} nm, {len(lam)} points "
                  f"(residual {p.xres:.2f} nm / {p.yres:.4f} lg)")

    # -- characteristic curve ------------------------------------------------
    p, err = _panel(pg, ws, "curve", BANDS["density"], xlo, xhi)
    if p is None:
        print(f"    curve      [--] {err}")
    else:
        xy = _one_curve(pg, p)
        if xy is None:
            print("    curve      [--] no curve")
        else:
            lx, dd = p.X(xy[0]), p.Y(xy[1])
            print(f"    curve      lgE {lx.min():+.2f}..{lx.max():+.2f}, "
                  f"D {dd.min():.3f}..{dd.max():.3f}, {len(lx)} points "
                  f"(residual {p.xres:.3f} dec / {p.yres:.4f} D)")

    # -- sharpness -----------------------------------------------------------
    p, err = _log_panel(pg, ws, "sharp", BANDS["sharpness"], xlo, xhi)
    if p is None:
        print(f"    sharpness  [--] {err}")
    else:
        xy = _one_curve(pg, p)
        if xy is None:
            print("    sharpness  [--] no curve")
        else:
            f = 10.0 ** p.X(xy[0])
            t = 10.0 ** p.Y(xy[1])
            print(f"    sharpness  peak {t.max():.0f} % at {f[int(np.argmax(t))]:.1f}, "
                  f"f50 {_f50(f, t):.1f} lines/mm "
                  f"(residual {p.xres:.4f}/{p.yres:.4f} dec)")

    # -- gamma-time ----------------------------------------------------------
    p, err = _panel(pg, ws, "gamma", BANDS["curves"], xlo, xhi)
    if p is None:
        print(f"    gamma-time [--] {err}")
    else:
        segs = _split_curves(pg, p)
        labels = _named_labels(pg, p, DEVELOPERS)
        print(f"    gamma-time {len(segs)} curves for {len(labels)} printed "
              f"developer names (residual {p.xres:.3f} min / {p.yres:.4f} gamma)")
        owners = _assign(labels, segs)
        for i, (xs, ys) in enumerate(segs):
            t, g = p.X(xs), p.Y(ys)
            who = " + ".join(owners[i]) or "⚠ UNLABELLED"
            print(f"        t {t.min():5.1f}-{t.max():5.1f} min  "
                  f"gamma {g.min():.3f}-{g.max():.3f}  {len(t):3d} pts  {who}")
            for tt in (4.0, 6.0, 8.0, 10.0, 12.0):
                if t.min() <= tt <= t.max():
                    print(f"            gamma({tt:4.1f} min) = "
                          f"{float(np.interp(tt, t, g)):.3f}")


def _read_scala(pg, ws, xlo, xhi):
    p, err = _panel(pg, ws, "spectral", SCALA_BANDS["spectral"], xlo, xhi)
    if p is None:
        print(f"    spectral   [--] {err}")
    else:
        xy = _one_curve(pg, p)
        if xy is None:
            print("    spectral   [--] no curve")
        else:
            lam, lg = p.X(xy[0]), p.Y(xy[1])
            print(f"    spectral   {lam.min():.0f}-{lam.max():.0f} nm, "
                  f"peak {lam[int(np.argmax(lg))]:.0f} nm, {len(lam)} points")
    p, err = _log_panel(pg, ws, "sharp", SCALA_BANDS["sharpness"], xlo, xhi)
    if p is None:
        print(f"    sharpness  [--] {err}")
    else:
        xy = _one_curve(pg, p)
        if xy is not None:
            f = 10.0 ** p.X(xy[0])
            t = 10.0 ** p.Y(xy[1])
            print(f"    sharpness  peak {t.max():.0f} %, f50 {_f50(f, t):.1f} lines/mm")
    p, err = _panel(pg, ws, "curves", SCALA_BANDS["curves"], xlo, xhi)
    if p is None:
        print(f"    curves     [--] {err}")
    else:
        segs = _split_curves(pg, p)
        labels = _named_labels(pg, p, SCALA_STEPS)
        print(f"    curves     {len(segs)} curves for {len(labels)} printed "
              f"step names (residual {p.xres:.3f} dec / {p.yres:.4f} D)")
        owners = _assign(labels, segs)
        for i, (xs, ys) in enumerate(segs):
            lx, dsy = p.X(xs), p.Y(ys)
            who = " + ".join(owners[i]) or "⚠ UNLABELLED"
            print(f"        lgE {lx.min():+.2f}..{lx.max():+.2f}  "
                  f"D {dsy.min():.3f}..{dsy.max():.3f}  {len(lx):3d} pts  {who}")


def _read_colour(pg, ws, xlo, xhi, kind):
    for key, band, logaxes, label in (
            ("spectral",  BANDS["spectral"],  False, "spectral  "),
            ("density",   BANDS["density"],   False, "spec.dens "),
            ("sharpness", BANDS["sharpness"], True,  "sharpness "),
            ("curves",    BANDS["curves"],    False, "curves    ")):
        make = _log_panel if logaxes else _panel
        p, err = make(pg, ws, key, band, xlo, xhi)
        if p is None:
            print(f"    {label} [--] {err}")
            continue
        got = _curves_by_shape(pg, p)
        if not got:
            segs = _split_curves(pg, p)
            print(f"    {label} no dash-keyed layers; {len(segs)} sub-paths")
            continue
        bits = []
        for lay in ("b", "g", "r"):
            if lay not in got:
                continue
            xs, ys = got[lay]
            if logaxes:
                f, t = 10.0 ** p.X(xs), 10.0 ** p.Y(ys)
                bits.append(f"{lay}: f50 {_f50(f, t):.1f}, peak {t.max():.0f} %")
            else:
                a, b = p.X(xs), p.Y(ys)
                bits.append(f"{lay}: {a.min():.0f}..{a.max():.0f} / "
                            f"{b.min():.2f}..{b.max():.2f} ({len(a)}p)")
        print(f"    {label} " + " | ".join(bits))


def _f50(freq, resp):
    """Last downward crossing of 50 %. ⚠ LAST, not first: the curve rises above
    100 % at low frequency, so a first-crossing search can return the wrong
    branch on a non-monotone response."""
    o = np.argsort(freq)
    f, r = np.asarray(freq)[o], np.asarray(resp)[o]
    idx = np.where((r[:-1] >= 50.0) & (r[1:] < 50.0))[0]
    if not len(idx):
        return float("nan")
    i = idx[-1]
    w = (50.0 - r[i]) / (r[i + 1] - r[i])
    return float(f[i] + w * (f[i + 1] - f[i]))


if __name__ == "__main__":
    sys.exit(main())
