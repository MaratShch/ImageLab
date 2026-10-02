#!/usr/bin/env python3
"""AgfaPhoto «Vista plus 200» and «Vista plus 400» — eight vector panels.

    PDF/PROFILES/AGFA/agfafilms-Vista.pdf
    AgfaPhoto Holding GmbH, «Product Information -- Color Negative Film»,
    Vista plus 200 (pp1-4) and Vista plus 400 (pp5-8), undated.

⚠⚠ **THIS IS NOT AGFA-GEVAERT'S VISTA AND THE SHEET SAYS SO ITSELF.** p4 and p8
carry, verbatim:

    «AgfaPhoto is used under license of Agfa-Gevaert NV & Co. KG or
     Agfa-Gevaert NV. Neither Agfa-Gevaert NV & Co KG nor Agfa-Gevaert NV
     manufacture this product or provide any product warranty or support.»
    «Produced for and distributed by Lupus Imaging & Media GmbH Co. KG»

So `AGFA_VISTA_200` -- Agfa-Gevaert, «Technical Data AF» 06/2000, whose rms 4.3
and f50 47.8 are that company's own measurements -- and «Vista plus 200» are
different products from different factories. **Nothing here is written to that
profile.** The 2026-09-07 Eterna back-out is the precedent: a neighbouring
product's number labelled as this film's is worse than an honest estimate, and
that pair were the same brand five years apart. This pair are not even that.

⚠ AND THE DOCUMENT IS BUILT ON THE FUJI HOUSE TEMPLATE, which is evidence about
who made the film. Five independent tells, all inside the file:

  * process «AP 70/CN-16/C41» -- CN-16 is Fuji's C-41-compatible process name
  * base «Cellulose Triacetate», «122 um (135)» -- the exact material and
    gauge printed on this corpus's Fuji colour negatives
  * «Typical densities for a mid-scale neutral subject and for D-mini.» --
    the Fuji dye-density caption, word for word, D-mini typo included
  * «Sensitivity equals the reciprocal of the exposure (J/cm2) required to
    produce a specified density.» -- the Fuji spectral footnote
  * section numbering 1-14 with the same headings in the same order

That is an INFERENCE about manufacture, drawn from the document and labelled as
one. It is recorded because it bears directly on the question the owner asked --
whether these curves may stand in for Agfa's Vista -- and it says no twice over.

WHAT IS READABLE, AND THE ONE THING THAT MAKES IT HARDER THAN IT LOOKS
----------------------------------------------------------------------
Four panels per film, all VECTOR: characteristic curves (three layers),
spectral sensitivity (three layers), MTF, spectral dye density (mid-scale
neutral and D-min). Every curve is a 1.00 pt stroked path, every gridline
0.30 pt, every frame a single 0.40 pt rect -- a clean three-way separation by
stroke width, which this corpus rarely gets.

⚠⚠ **BUT THE AXIS NUMBERS HAVE NO TEXT LAYER.** `get_text` returns 131 words on
p4 and not one numeral from any ladder: the labels are glyph OUTLINES, drawn as
filled paths. So the 2026-09-06i rule -- «the printed ladder says WHICH values
an axis spans, the drawn frame says WHERE» -- has only its second half
available here. There is no label text to fit, and therefore no independent
check on a mis-set tick.

What replaces it: every calibration below is anchored on the FRAME and then
required to PREDICT the panel's own 0.30 pt GRIDLINES. Nothing in the fit is
told where those lines are, so reproducing them is a free check on the axis
assignment -- and it is the check that caught the characteristic panel's
abscissa, whose frame edge is NOT a labelled decade:

    the MTF ordinate, anchored 2 % at the bottom edge and 150 % at the top,
    predicts all nine printed gridlines to 1.6 pt worst

    the characteristic ABSCISSA is anchored on the GRIDLINES, not the frame:
    the right edge is lg H 0.0 and the left edge is half a division BEYOND
    -4.0, an unlabelled margin. Anchoring x on the frame shifts every exposure
    by 0.25 decade while leaving every density and every gamma untouched --
    the same defect NEOPAN SS's abscissa had, and gamma cannot catch it.

Run:  python3 agfaphoto_vista_plus.py [--root .] [--assert]
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SHEET = "AGFA/agfafilms-Vista.pdf"

SOURCE = ("AgfaPhoto Holding GmbH, «Product Information -- Color Negative "
          "Film», Vista plus 200 and Vista plus 400 -- "
          "PDF/PROFILES/AGFA/agfafilms-Vista.pdf. ⚠ NOT an Agfa-Gevaert "
          "publication: the sheet states that neither Agfa-Gevaert NV & Co KG "
          "nor Agfa-Gevaert NV manufacture the product, and names Lupus "
          "Imaging & Media GmbH Co. KG as producer")

#: (profile, printed name, page index, ISO, tungsten ISO, production number)
FILMS = (
    ("AGFA_VISTA_PLUS_200", "Vista plus 200", 3, 200, 50, "C74"),
    ("AGFA_VISTA_PLUS_400", "Vista plus 400", 7, 400, 100, "H74"),
)

#: Phrases asserted verbatim on every run. If a future edition drops the
#: disclaimer, the ground for keeping these stocks separate from
#: `AGFA_VISTA_200` is gone and the build should say so rather than carry on.
#: ⚠ «Typical densities for a mid-scale neutral subject and for D-mini.» IS
#: NOT IN THIS TUPLE, AND IT IS ON THE PAGE. It sits INSIDE the dye panel as
#: glyph outlines, like every axis numeral on this sheet, so a text search
#: cannot find it -- which is the same fact that forces the frame-anchored
#: calibration below, met here as a false negative instead of a missing axis.
#: It is quoted in the docstring from the RENDER, and the four phrases that do
#: live in the text layer are asserted instead.
MARKERS = (
    "Neither Agfa-Gevaert NV & Co KG nor Agfa-",
    "Produced for and distributed by Lupus Imaging & Media GmbH",
    "AP 70/CN-16/C41",
    "Cellulose Triacetate",
)

CURVE_W, GRID_W, FRAME_W = 1.00, 0.30, 0.40

#: Gridline agreement a calibration must reach, in points. ⚠ SET BY THE PAGE:
#: the MTF ordinate's worst gridline residual is 1.6 pt over a 148 pt axis
#: (1.1 %), and its abscissa's 1.7 pt over 177 pt. 2.5 pt therefore passes
#: every panel on this sheet with margin and fails a quarter-division slip,
#: which on the characteristic abscissa is 5.4 pt.
GRID_TOL_PT = 2.5

#: What this module reads, pinned so a rerun that disagrees FAILS. Set from
#: the traced values, not before them.
EXPECTED = {
    "AGFA_VISTA_PLUS_200": dict(
        dmin=(0.1166, 0.4196, 0.7301), gamma=(0.6187, 0.6492, 0.6708),
        peaks=(472.0, 560.0, 641.0)),
    "AGFA_VISTA_PLUS_400": dict(
        dmin=(0.1208, 0.4528, 0.7351), gamma=(0.6250, 0.6747, 0.7059),
        peaks=(472.0, 558.0, 639.0)),
}

#: The shared MTF drawing, measured. ⚠ TWO FILMS, ONE PATH: 47 points,
#: translated dx +0.142 pt and dy -0.075 pt with spreads of 0.0010 and 0.0020
#: pt -- i.e. the same path object placed twice, to a five-hundredth of a
#: point. Pinned so the refusal is re-earned on every build rather than
#: remembered.
MTF_DUP_SPREAD_TOL_PT = 0.02


class Cal:
    """A panel's calibration. `X`/`Y` are what `fit_tone` calls."""

    def __init__(self, fx, fy, name):
        self._fx, self._fy, self.name = fx, fy, name

    def X(self, px):
        return self._fx(np.asarray(px, dtype=float))

    def Y(self, py):
        return self._fy(np.asarray(py, dtype=float))


def _strokes(page, width, band=None):
    """Every stroked polyline at `width`, as arrays of points."""
    out = []
    for dr in page.get_drawings():
        if dr.get("type") != "s" or abs((dr.get("width") or 0.0) - width) > 0.02:
            continue
        for it in dr["items"]:
            # ⚠⚠ A BEZIER'S CONTROL POINTS ARE NOT ITS CURVE, and reading them
            # as one returned f50 = nan on both MTF panels: that curve is TWO
            # cubic items, eight control points, and the outer controls sit
            # ABOVE the drawn response -- so the extracted "curve" peaked at
            # 1.19 where the ink peaks near 1.10 and never crossed 50 %.
            # Every item of kind "c" is therefore FLATTENED here, which also
            # removes a systematic error from the characteristic and dye
            # traces, whose curvature was being cut across by straight chords.
            if it[0] == "c" and len(it) == 5 and hasattr(it[1], "x"):
                p0, p1, p2, p3 = (np.array([q.x, q.y], dtype=float)
                                  for q in it[1:5])
                t = np.linspace(0.0, 1.0, 24)[:, None]
                b0 = (1 - t) ** 3
                b1 = 3 * t * (1 - t) ** 2
                b2 = 3 * t ** 2 * (1 - t)
                b3 = t ** 3
                pts = [tuple(v) for v in (b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3)]
            else:
                pts = [(q.x, q.y) for q in it[1:] if hasattr(q, "x")]
            # ⚠ A FRAME IS A RECT ITEM, NOT A POLYLINE, and reading only point
            # payloads returned FOUR EMPTY PANELS on the first run: pymupdf
            # emits the 0.40 pt frames as "re" items carrying a Rect, whose
            # corners are x0/y0/x1/y1 and not x/y. Expanded here.
            if not pts:
                for q in it[1:]:
                    if hasattr(q, "x0"):
                        pts += [(q.x0, q.y0), (q.x1, q.y0),
                                (q.x1, q.y1), (q.x0, q.y1)]
            if len(pts) < 2:
                continue
            a = np.array(pts, dtype=float)
            if band is not None:
                x0, x1, y0, y1 = band
                if not (x0 - 6 <= a[:, 0].min() and a[:, 0].max() <= x1 + 6
                        and y0 - 6 <= a[:, 1].min() and a[:, 1].max() <= y1 + 6):
                    continue
            out.append(a)
    return out


def _frames(page):
    """The four plot frames, ordered: char, spec, mtf, dye."""
    fr = []
    for a in _strokes(page, FRAME_W):
        x0, x1 = a[:, 0].min(), a[:, 0].max()
        y0, y1 = a[:, 1].min(), a[:, 1].max()
        if x1 - x0 > 120 and y1 - y0 > 100:
            fr.append((x0, x1, y0, y1))
    # ⚠ ORDER BY ROW THEN COLUMN, not by discovery order: the four panels sit
    # in a 2x2 grid and the PDF emits them in no fixed sequence.
    fr = sorted(set(fr), key=lambda f: (round(f[2] / 100.0), f[0]))
    return fr


def _grid(page, band):
    """(vertical xs, horizontal ys) of a panel's own gridlines."""
    v, h = set(), set()
    for a in _strokes(page, GRID_W, band):
        if abs(a[:, 0].max() - a[:, 0].min()) < 0.5:
            v.add(round(float(a[0, 0]), 2))
        elif abs(a[:, 1].max() - a[:, 1].min()) < 0.5:
            h.add(round(float(a[0, 1]), 2))
    return sorted(v), sorted(h)


def _check(pred, got, tol=GRID_TOL_PT):
    """Worst |predicted - drawn| over a gridline set, in points."""
    if not len(got):
        return 0.0, True
    worst = max(min(abs(p - g) for p in pred) for g in got)
    return worst, worst <= tol


def _curves(page, band, tol=1.0):
    """Data curves in a panel, chained from their drawn segments.

    ⚠⚠ SORTING BY START POINT AND WALKING FORWARD RETURNED NINE CURVES FOR
    THREE. A drawn record is emitted as a run of bezier items, and the run is
    NOT in left-to-right order -- so a chain built by "does the next segment
    start where the last one ended" breaks the moment the emitter jumps back.
    This joins segments at EITHER end, in any order, which is what a polyline
    actually is.
    """
    segs = [a for a in _strokes(page, CURVE_W, band)]
    out = []
    while segs:
        cur = segs.pop(0)
        grew = True
        while grew:
            grew = False
            for k, a in enumerate(segs):
                for cand in (a, a[::-1]):
                    if np.hypot(*(cand[0] - cur[-1])) <= tol:
                        cur = np.vstack([cur, cand[1:]])
                    elif np.hypot(*(cand[-1] - cur[0])) <= tol:
                        cur = np.vstack([cand[:-1], cur])
                    else:
                        continue
                    segs.pop(k)
                    grew = True
                    break
                if grew:
                    break
        out.append(cur)
    return out


def f50_of(f, r):
    """Last downward crossing of 50 %, computed here rather than imported.

    ⚠ `agfa_1998_curves._f50` returned nan on this panel and the reason is a
    convention, not a bug: this response is traced from 2.8 c/mm, where it is
    already ABOVE 100 %, and that helper's search assumes a form these samples
    do not satisfy. Written out so the crossing is explicit.
    """
    below = np.flatnonzero(r < 0.5)
    if not len(below) or below[0] == 0:
        return float("nan")
    k = below[0]
    return float(np.interp(0.5, [r[k], r[k - 1]], [f[k], f[k - 1]]))


def on_10nm(nm, v, lo=380.0, hi=700.0):
    """Resample to this corpus's 10 nm grid; -4.00 where the curve is not drawn.

    ⚠ THE SENTINEL IS NOT A MEASUREMENT. Outside the drawn span the value is
    UNKNOWN, and -4.00 is what this database uses to say so -- the same
    convention SCALA's spectral set carries above 660 nm.
    """
    grid = np.arange(lo, hi + 0.1, 10.0)
    out = np.full(grid.shape, -4.00)
    m = (grid >= nm.min()) & (grid <= nm.max())
    out[m] = np.interp(grid[m], nm, v)
    return grid, out


# ---------------------------------------------------------------------------
#  The four calibrations
# ---------------------------------------------------------------------------

def cal_char(page, fr):
    """Density 0.0-4.0 against lg H, abscissa anchored on the GRIDLINES.

    ⚠⚠ THE FRAME IS NOT THE LADDER IN X, AND THIS IS THE TRAP ON THIS PANEL.
    The nine vertical gridlines are the half-decade divisions -4.0 .. 0.0, and
    the RIGHT frame edge coincides with the last of them (0.0) while the LEFT
    edge sits half a division further out -- an unlabelled margin at -4.5.
    Anchoring x on the frame therefore shifts every exposure by 0.25 decade,
    and because gamma is a ratio of differences NOTHING downstream complains.
    """
    x0, x1, y0, y1 = fr
    v, h = _grid(page, fr)
    if len(v) != 9 or len(h) != 7:
        return None, "characteristic grid is %d x %d, expected 9 x 7" % (len(v), len(h))
    # x: the nine gridlines span -4.0 .. 0.0 in nine equal steps of 0.5
    step = (v[-1] - v[0]) / 8.0
    dev = max(abs((v[i + 1] - v[i]) - step) for i in range(8))
    if dev > 1.0:
        return None, "characteristic abscissa divisions are uneven: %.2f pt" % dev
    def fx(px):
        return -4.0 + (px - v[0]) / step * 0.5
    # y: the FRAME carries 0.0 at the bottom and 4.0 at the top
    def fy(py):
        return 4.0 * (y1 - py) / (y1 - y0)
    pred = [y1 - (y1 - y0) * k / 8.0 for k in range(1, 8)]
    worst, ok = _check(pred, h)
    if not ok:
        return None, "characteristic ordinate misses its gridlines by %.2f pt" % worst
    return Cal(fx, fy, "char"), "x on 9 divisions (%.2f pt each), y frame-anchored, gridlines to %.2f pt" % (step, worst)


def cal_mtf(page, fr):
    """Response 2-150 % against 1-200 cycles/mm, both log, both frame-anchored.

    ⚠ THE ABSCISSA IS LABELLED «Spatial Frequency (cycles/mm)» IN WORDS, where
    every Agfa-Gevaert range sheet says «Lines per mm». Same panel type, later
    brand, and it writes out the unit queue G6 had to infer from the ICO's 1961
    recommendation. Independent support for the answer already reached.
    """
    x0, x1, y0, y1 = fr
    v, h = _grid(page, fr)
    def fx(px):
        return 10.0 ** (np.log10(1.0) + (px - x0) / (x1 - x0) * np.log10(200.0 / 1.0))
    def fy(py):
        return (10.0 ** (np.log10(150.0) + (py - y0) / (y1 - y0)
                         * np.log10(2.0 / 150.0))) / 100.0
    px = lambda f: x0 + (x1 - x0) * np.log10(f / 1.0) / np.log10(200.0)
    py = lambda r: y0 + (y1 - y0) * np.log10(r / 150.0) / np.log10(2.0 / 150.0)
    wx, okx = _check([px(f) for f in (5, 10, 20, 50, 100)], v)
    wy, oky = _check([py(r) for r in (3, 5, 7, 10, 20, 30, 50, 70, 100)], h)
    if not (okx and oky):
        return None, "MTF gridlines miss by %.2f pt x / %.2f pt y" % (wx, wy)
    return Cal(fx, fy, "mtf"), "frame-anchored 1-200 c/mm and 2-150 %%, gridlines to %.2f / %.2f pt" % (wx, wy)


def cal_nm(page, fr, name, span_d=None):
    """400-700 nm from the four gridlines; ordinate depends on the panel."""
    x0, x1, y0, y1 = fr
    v, h = _grid(page, fr)
    if len(v) != 4:
        return None, "%s has %d wavelength gridlines, expected 4" % (name, len(v))
    step = (v[-1] - v[0]) / 3.0
    dev = max(abs((v[i + 1] - v[i]) - step) for i in range(3))
    if dev > 1.5:
        return None, "%s wavelength divisions uneven: %.2f pt" % (name, dev)
    def fx(px):
        return 400.0 + (px - v[0]) / step * 100.0
    if span_d is None:
        # spectral sensitivity: RELATIVE log, and the panel's only ordinate
        # mark is a bracket one decade tall. Stored peak-normalised, so the
        # bracket sets the SCALE and the peak sets the zero.
        if len(h) != 0:
            return None, "%s ordinate unexpectedly has %d gridlines" % (name, len(h))
        def fy(py):
            return -(py - y0) / (y1 - y0)          # 1.0 over the frame height
        return Cal(fx, fy, name), "nm on 4 divisions (%.2f pt each), ordinate RELATIVE" % step
    # dye density: four gridlines at 0.5 D each, 0.0 at the bottom frame edge
    if len(h) != 4:
        return None, "%s has %d density gridlines, expected 4" % (name, len(h))
    # ⚠ SIGN. `h` is sorted ASCENDING in PDF y, which runs DOWNWARD, so the
    # spacing is h[i+1]-h[i] and taking it the other way round put the
    # predicted gridlines 148 pt off -- the height of the panel.
    per = float(np.mean([h[i + 1] - h[i] for i in range(3)]))
    def fy(py):
        return (y1 - py) / per * 0.5
    pred = [y1 - per * k for k in range(1, 5)]
    worst, ok = _check(pred, h)
    if not ok:
        return None, "%s density gridlines miss by %.2f pt" % (name, worst)
    return Cal(fx, fy, name), "nm on 4 divisions, 0.5 D per %.2f pt, gridlines to %.2f pt" % (per, worst)


# ---------------------------------------------------------------------------

def read_page(doc, page_no, verbose=True):
    page = doc[page_no]
    fr = _frames(page)
    if len(fr) != 4:
        return None, "%d plot frames on this page, expected 4" % len(fr)
    char_fr, spec_fr, mtf_fr, dye_fr = fr
    out = {}

    # ---- 11. characteristic curves ----------------------------------------
    cal, why = cal_char(page, char_fr)
    if cal is None:
        return None, why
    if verbose:
        print("     characteristic  %s" % why)
    cs = _curves(page, char_fr)
    if len(cs) != 3:
        return None, "characteristic panel has %d curves, expected 3" % len(cs)
    # ⚠ THE LEGEND IS Blue / Green / Red TOP TO BOTTOM and the records are
    # separated by their D-min, blue highest -- the mask. Assign by the curve's
    # own left-hand density, not by drawing order.
    cs.sort(key=lambda a: cal.Y(a[:, 1]).max(), reverse=True)
    out["char"] = {}
    for lay, a in zip(("b", "g", "r"), cs):
        o = np.argsort(a[:, 0])
        le, dd = cal.X(a[o, 0]), cal.Y(a[o, 1])
        out["char"][lay] = (le, dd)

    # ---- 13. MTF -- READ, AND REFUSED ------------------------------------
    cal_m, why = cal_mtf(page, mtf_fr)
    if cal_m is None:
        return None, why
    if verbose:
        print("     MTF             %s" % why)
    cs = _curves(page, mtf_fr)
    if len(cs) != 1:
        return None, "MTF panel has %d curves, expected 1" % len(cs)
    a = cs[0]
    o = np.argsort(a[:, 0])
    out["mtf"] = (cal_m.X(a[o, 0]), cal_m.Y(a[o, 1]))
    out["mtf_px"] = a

    # ---- 12. spectral sensitivity -----------------------------------------
    cal_s, why = cal_nm(page, spec_fr, "spectral")
    if cal_s is None:
        return None, why
    if verbose:
        print("     spectral        %s" % why)
    cs = _curves(page, spec_fr)
    if len(cs) != 3:
        return None, "spectral panel has %d curves, expected 3" % len(cs)
    cs.sort(key=lambda a: cal_s.X(a[:, 0]).mean())
    out["spec"] = {}
    for lay, a in zip(("b", "g", "r"), cs):
        o = np.argsort(a[:, 0])
        out["spec"][lay] = (cal_s.X(a[o, 0]), cal_s.Y(a[o, 1]))

    # ---- 14. spectral dye density -----------------------------------------
    cal_d, why = cal_nm(page, dye_fr, "dye", span_d=True)
    if cal_d is None:
        return None, why
    if verbose:
        print("     dye density     %s" % why)
    cs = _curves(page, dye_fr)
    if len(cs) != 2:
        return None, "dye panel has %d curves, expected 2" % len(cs)
    # the neutral sits ABOVE the D-min at every wavelength -- the physics gate
    cs.sort(key=lambda a: cal_d.Y(a[:, 1]).mean(), reverse=True)
    out["dye"] = {}
    for who, a in zip(("neutral", "dmin"), cs):
        o = np.argsort(a[:, 0])
        out["dye"][who] = (cal_d.X(a[o, 0]), cal_d.Y(a[o, 1]))
    return out, None


def report(name, printed, got):
    import agfa_1998_curves as G
    print("  -- %s  (%s)" % (name, printed))
    res = {}
    # tone
    for lay in ("r", "g", "b"):
        le, dd = got["char"][lay]
        ft = G._fit_tone_ms(Cal(lambda p: p, lambda p: p, "id"), (le, dd))
        if ft is None:
            print("     %s  lgH %+.2f..%+.2f  D %.3f..%.3f   (no fit -- SciPy absent)"
                  % (lay, le.min(), le.max(), dd.min(), dd.max()))
            res.setdefault("dmin", {})[lay] = float(dd.min())
            continue
        dmin, gam, tx, tk, sx, sk, rms = ft
        res.setdefault("dmin", {})[lay] = dmin
        res.setdefault("gamma", {})[lay] = gam
        print("     %s  dmin %.4f  gamma %.4f  toe %+.3f/%.3f  sh %+.3f/%.3f  rms %.4f D"
              % (lay, dmin, gam, tx, tk, sx, sk, rms))
    # MTF -- reported and REFUSED, see the duplicate check in main()
    f, r = got["mtf"]
    f50 = f50_of(f, r)
    peak = float(r.max())
    res.update(f50=f50, peak=peak)
    print("     MTF  traced %.1f-%.1f c/mm   f50 %.1f   peak %+.4f at %.1f c/mm"
          "   ⚠ SHARED PANEL -- adopted on both, see the check below"
          % (f.min(), f.max(), f50, peak - 1.0, f[int(r.argmax())]))
    # spectral
    pk = {}
    for lay in ("b", "g", "r"):
        nm, v = got["spec"][lay]
        pk[lay] = float(nm[int(np.argmax(v))])
    res["peaks"] = pk
    print("     spectral peaks  B %.0f  G %.0f  R %.0f nm" % (pk["b"], pk["g"], pk["r"]))
    # dye
    nm, dn = got["dye"]["neutral"]
    nm2, dm = got["dye"]["dmin"]
    print("     dye  neutral %.3f..%.3f D over %.0f-%.0f nm;  D-min %.3f..%.3f  "
          "(the ORANGE MASK, measured)" % (dn.min(), dn.max(), nm.min(), nm.max(),
                                           dm.min(), dm.max()))
    lo = float(np.interp(430.0, nm2, dm))
    hi = float(np.interp(680.0, nm2, dm))
    print("     mask falls %.3f -> %.3f D from 430 to 680 nm  [%s]"
          % (lo, hi, "OK" if lo > hi else "FAIL -- a mask must fall to the red"))
    res["mask_falls"] = lo > hi
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve() / "PDF" / "PROFILES"
    path = root / SHEET
    if not path.is_file():
        print("  [SKIP] source not present: %s" % path)
        return 0
    doc = pymupdf.open(str(path))
    print("[i] %s\n" % SOURCE)
    bad = 0

    txt = "\n".join(p.get_text() for p in doc)
    miss = [m for m in MARKERS if m not in txt]
    if miss:
        print("  [FAIL] the sheet no longer carries: %s" % miss)
        bad += 1
    else:
        print("  [OK  ] all %d marker phrases present, including the "
              "disclaimer that Agfa-Gevaert do NOT manufacture this product "
              "and the CN-16 / cellulose-triacetate / «mid-scale neutral» "
              "tells that place the document on the Fuji template"
              % len(MARKERS))

    for profile, printed, page_no, iso, iso_t, prod in FILMS:
        got, err = read_page(doc, page_no, verbose=True)
        if got is None:
            print("  [FAIL] %s: %s" % (profile, err))
            bad += 1
            continue
        res = report(profile, printed, got)
        want = EXPECTED.get(profile)
        if want:
            for lay, k in enumerate(("r", "g", "b")):
                for field, tol in (("dmin", 0.004), ("gamma", 0.004)):
                    have = res.get(field, {}).get(k)
                    if have is None:
                        continue
                    if abs(have - want[field][lay]) > tol:
                        print("     [MISMATCH] %s %s %.4f vs pinned %.4f"
                              % (k, field, have, want[field][lay]))
                        bad += 1
            for lay, k in enumerate(("b", "g", "r")):
                have = res.get("peaks", {}).get(k)
                if have is not None and abs(have - want["peaks"][lay]) > 3.0:
                    print("     [MISMATCH] %s peak %.0f vs pinned %.0f nm"
                          % (k, have, want["peaks"][lay]))
                    bad += 1
        if not res.get("mask_falls", True):
            bad += 1
        print("     printed: ISO %d/%s daylight, %d tungsten via Wratten 80A; "
              "production number %s and above; base cellulose triacetate "
              "122 um (135); process AP 70 / CN-16 / C41"
              % (iso, "24" if iso == 200 else "27", iso_t, prod))
        print()

    # ---- ⚠⚠ THE SHARED MTF DRAWING, RE-EARNED EVERY BUILD ----------------
    # Two films cannot share a MEASURED MTF. `NotFound.md` row 5d refuses
    # AGFAPAN APX 100 and APX 400's f50 on exactly this ground -- one 73-point
    # path translated 175.21 pt, every y offset identical -- and the same
    # signature is on this sheet, tighter still.
    try:
        a = _curves(doc[FILMS[0][2]], _frames(doc[FILMS[0][2]])[2])[0]
        b = _curves(doc[FILMS[1][2]], _frames(doc[FILMS[1][2]])[2])[0]
        if len(a) != len(b):
            print("  [note] the two MTF paths differ in length (%d vs %d), so "
                  "they are NOT one drawing and the refusal below would need "
                  "re-deciding" % (len(a), len(b)))
        else:
            dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
            sx, sy = float(dx.max() - dx.min()), float(dy.max() - dy.min())
            ok = max(sx, sy) <= MTF_DUP_SPREAD_TOL_PT
            print("  -- MTF: ONE DRAWING FOR TWO FILMS  [%s]" % ("OK" if ok else "FAIL"))
            print("     %d points, translated dx %+.3f dy %+.3f pt, spreads "
                  "%.4f / %.4f pt -- the same path object placed twice"
                  % (len(a), float(dx.mean()), float(dy.mean()), sx, sy))
            print("     => ADOPTED ON BOTH STOCKS, 2026-09-07b, by owner "
                  "decision -- and the finding above is reported, not "
                  "withdrawn. At most one of the two can be a measurement of "
                  "its own film and the sheet does not say which; what "
                  "changed is the comparison, because the alternative was a "
                  "class estimate derived from no document. Owner: \"a vendor "
                  "MTF, even shared, is much more better from estimated "
                  "values\". Both profiles record that the curve cannot be "
                  "per-film for both films. Precedent reversed: "
                  "NotFound.md row 5d")
            if not ok:
                print("     [FAIL] the two paths are no longer one drawing")
                return 1
            print("     ⚠ AND THE OTHER THREE PANELS ARE NOT SHARED, which is "
                  "what makes this a finding rather than a broken reader: the "
                  "characteristic, spectral and dye records differ between the "
                  "two films by 8-17 pt of shape and by their point counts.")
            try:
                import film_profiles as _fp
                _bad = []
                for _n in ("AGFA_VISTA_PLUS_200",
                           "AGFA_VISTA_PLUS_400"):
                    _m = _fp.get_profile(_n).mtf
                    if (abs(_m.f50_g - 58.7) > 0.05
                            or abs(_m.mtf_rolloff_q - 2.65) > 1e-9
                            or not _m.mtf_measured):
                        _bad.append("%s stores f50_g %.2f q %.2f measured %s"
                                    % (_n, _m.f50_g, _m.mtf_rolloff_q,
                                       _m.mtf_measured))
                print("     AGAINST THE DATABASE: %s"
                      % ("both stocks carry the shared panel's f50 58.7 and "
                         "q 2.65, flagged measured" if not _bad
                         else "; ".join(_bad)))
                if _bad:
                    return 1
            except Exception as _e:                           # pragma: no cover
                print("     [WARN] could not compare: %s" % _e)
    except Exception as exc:                                  # pragma: no cover
        print("  [WARN] could not run the shared-drawing check: %s" % exc)
    print()

    # ---- the comparison the owner asked for -------------------------------
    try:
        import film_profiles as fp
        v2 = fp.get_profile("AGFA_VISTA_200")
        print("  -- AGFA_VISTA_200, for distance ONLY. Nothing above is written "
              "to it")
        print("     stored: gamma %.4f (%s)  dmin_g %.4f  f50_g %.1f  "
              "adjacency %+.4f  rms %.1f"
              % (v2.curves.g.gamma,
                 {s.param: s.status for s in v2.param_sources}
                 .get("curves.g.gamma", "?"),
                 v2.curves.g.dmin, v2.mtf.f50_g, v2.mtf.adjacency,
                 v2.grain.rms_granularity))
        print("     ⚠ AGFA_VISTA_400 DOES NOT EXIST in this database, so the "
              "400 half of the question has no counterpart to improve")
    except Exception as exc:                                  # pragma: no cover
        print("  [WARN] could not read film_profiles: %s" % exc)

    if ns.assert_ and bad:
        print("\n[FAIL] %d problem(s)" % bad)
        return 1
    print("\n[OK] eight panels read from a sheet that is NOT Agfa-Gevaert's -- "
          "two new stocks' worth of curves, and the reason they must not be "
          "written onto AGFA_VISTA_200 is printed on the sheet itself.")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
