"""The 2003/2004 Agfa Professional Films sheet, pages 8 and 9, read as vector.

WHAT THIS SOURCE IS, AND WHY A THIRD AGFA READER EXISTS
-------------------------------------------------------
`AGFA/AGFA stocks.pdf` -- Agfa-Gevaert, «Technical Data: Agfa Professional
Films», publication **F-PF-E4**, 4th edition, 08/2004. It has been in the
corpus since 2026-08-29 and `agfa_2004_curves.py` reads it -- but only
**pages 6 and 7**, which is Portrait 160 and the three Optima. Pages 8 and 9
carry six more plotted columns and were never touched:

    printed p8   Agfachrome RSX II 50 / 100 / 200      (reversal, 4 panels)
    printed p9   Agfapan APX 100 / APX 400             (mono, 4 panels)
    printed p9   Agfa Scala 200x                       (B&W reversal)

⚠ **THE GERMAN TWIN.** `AGFA/agfa-aERRKF-Datenblatt_F_PF_D4.pdf` is
**F-PF-D4, 4. Auflage, Stand 07/2003** -- the same PageMaker job in German,
distilled 55 minutes before the English one on 2003-07-18. This module asserts
the twin relationship rather than assuming it, by comparing the two files'
drawing coordinates page by page. **pp7-8 are geometrically IDENTICAL** and
pp6/9 differ only in label placement, so the German edition contributes
terminology and printed tables, not curves, and every curve below is read from
the English file so that one document remains the citation of record.

WHY THIS MATTERS: THE 1998 SHEET SAYS SOMETHING DIFFERENT
---------------------------------------------------------
`agfa_1998_curves.py` reads «Technical Data PF», 1st edition 09/1998, which
plots the SAME films five years earlier, and the database's RSX II, APX and
Scala curves all come from it. The two editions do not agree on the printed
resolving power:

    RSX II 50    1998: 125 lines/mm      2003: 135 lines/mm
    RSX II 100   1998: 125               2003: 130
    RSX II 200   1998: 110               2003: 120

with RMS, the 1.6:1 figure and the layer thickness unchanged on all three.
Two readings of the same product line five years apart is exactly the kind of
claim that must not be adopted on one document's word, so **this module's real
job is the cross-edition comparison**: trace the 2003 curves, put them beside
the 1998 ones the database already holds, and let the shapes say whether the
emulsion moved or only the resolving-power measurement did.

⚠ AND ONE COLUMN IS FLAGGED AS A DIFFERENT PRODUCT BY AGFA THEMSELVES.
D4/E4 p1 marks Agfapan APX 400 «* Neue Generation (ab 2003)» / "* new
generation (as of 2003)". Its headline numbers are unchanged -- RMS 14.0,
110 lines/mm, 10 um -- but its processing tables are wholly different from the
1998 ones (REFINAL tray 20 C: 6 min then, 5 min now; RODINAL 1+50: 11 min then,
30 min now). The control that makes that convincing is APX 100, whose tables
are identical across the two editions cell for cell. So the APX 400 column on
p9 is the LATER film and the database's APX 400 curve is the EARLIER one; this
module reports the difference and does not silently merge them.

THE MACHINERY IS THE 1998 READER'S, NOT THE 2004 READER'S
----------------------------------------------------------
`agfa_2004_curves` calibrates from a stroked frame rect and separates data from
furniture by stroke width. Both work on pp6-7 and both fail on pp8-9: the
RSX II 200 spectral panel returns no curve at all under frame containment, and
the RSX II 50 panel's y calibration comes back with a 0.36 lg residual because
Agfa mis-set that panel's bottom label as «- 0.1» where its two siblings print
«-1.0». So this module imports `agfa_1998_curves`, which calibrates from the
panel's OWN PRINTED LABELS with iterative outlier rejection and finds curves by
path SHAPE, and gives it the 2003 sheet's geometry. The mis-set label is
dropped by the robust fit and named in the output rather than silently averaged
in.

Run:  python agfa_2003_curves.py --root <corpus> [--assert]
Needs numpy + PyMuPDF.
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

import agfa_1998_curves as G
import agfa_2004_curves as A4

SHEET = "AGFA/AGFA stocks.pdf"
TWIN = "AGFA/agfa-aERRKF-Datenblatt_F_PF_D4.pdf"

SOURCE = ("Agfa-Gevaert, «Technical Data: Agfa Professional Films», "
          "publication F-PF-E4, 4th edition, 08/2004 -- "
          "PDF/PROFILES/AGFA/AGFA stocks.pdf pp8-9. German twin: "
          "«Technische Daten -- Agfa Professional Filmsortiment», F-PF-D4, "
          "4. Auflage, Stand 07/2003 -- "
          "PDF/PROFILES/AGFA/agfa-aERRKF-Datenblatt_F_PF_D4.pdf.")

# ⚠ THE 2003 SHEET IS DRAWN AT THE 2004 READER'S SCALE, NOT THE 1998 ONE'S.
# `agfa_1998_curves` patches these three module attributes on `agfa_2004_curves`
# at import time for its own 0.929x artwork; they must be put back before any
# panel here is read, or `_is_ink` and the dash keys are applied at the wrong
# scale. Curve detection itself is by SHAPE and does not consult the width, but
# the constants are shared state and are restored so that nothing downstream
# inherits the 1998 values.
A4.CURVE_W, A4.FRAME_W, A4.W_TOL = 0.85, 0.283, 0.10

#: (profile, printed name, printed page, x band, kind)
#: ⚠ `profile` is the stock this column DESCRIBES, which for APX 400 is not the
#: stock it should be adopted into -- see the module docstring.
COLUMNS = (
    ("AGFA_RSX_II_50",  "Agfachrome RSX II 50",  8, (26.0, 187.0), "reversal"),
    ("AGFA_RSX_II_100", "Agfachrome RSX II 100", 8, (202.0, 363.0), "reversal"),
    ("AGFA_RSX_II_200", "Agfachrome RSX II 200", 8, (377.0, 538.0), "reversal"),
    ("AGFA_APX_100",    "Agfapan APX 100",       9, (44.0, 204.0), "mono"),
    ("AGFA_APX_400",    "Agfapan APX 400",       9, (219.0, 379.0), "mono"),
    ("AGFA_SCALA_200X", "Agfa Scala 200x",       9, (395.0, 555.0), "bw_rev"),
)

#: Panel y bands, measured from the sheet's own inner plot boxes. The colour
#: and mono layouts share three of four slots; the fourth differs because the
#: APX gamma-time panel is 15 pt taller than a colour-density panel.
BANDS = {
    "spectral":  (79.0, 175.0),
    "density":   (220.0, 284.0),
    "sharpness": (327.0, 415.0),
    "curves":    (461.0, 549.0),
}
BANDS_MONO = dict(BANDS, curves=(461.0, 564.0))

#: ⚠ SCALA'S SLOT 2 IS TALLER AND ITS SLOT 4 IS TWO STACKED MINI-PLOTS.
#: Unlike the 1998 sheet -- where Scala has no spectral-density panel and Agfa
#: moved the remaining three UP one slot -- the 2003 sheet keeps all four slots
#: and fills slot 2 with the push/pull density family. Slot 4 holds
#: «Gradation/Maximaldichte bei push/pull-Verarbeitung», drawn as a Maximum
#: density box above a Contrast box sharing one ISO abscissa.
SCALA_BANDS = {
    "spectral":  (79.0, 175.0),
    "curves":    (214.0, 289.0),
    "sharpness": (329.0, 416.0),
}

DEVELOPERS = G.DEVELOPERS
SCALA_STEPS = G.SCALA_STEPS


# ---------------------------------------------------------------------------
#  The twin check
# ---------------------------------------------------------------------------

def _geometry(page):
    """Every drawn coordinate on a page, rounded, in document order."""
    out = []
    for dr in page.get_drawings():
        for it in dr["items"]:
            for pt in it[1:]:
                x = getattr(pt, "x", None)
                if x is not None:
                    out.append((round(x, 2), round(pt.y, 2)))
                    continue
                x0 = getattr(pt, "x0", None)
                if x0 is not None:
                    out.append((round(x0, 2), round(pt.y0, 2),
                                round(pt.x1, 2), round(pt.y1, 2)))
    return out


#: The pages that carry plotted film columns. Only these have to match between
#: the two editions; pp1-5 and pp10-12 are prose, tables, a logo and a
#: distributor list, and they legitimately differ -- the German p12 alone draws
#: twice as many points as the English one because its colophon block is set
#: differently. Asserting identity there would be asserting that a translation
#: cannot re-flow, which is not what "twin" has to mean.
PLOT_PAGES = (6, 7, 8, 9)


def twin_report(eng, ger):
    """Compare the English and German files page by page.

    Returns (identical, delta, failures). A PLOT page whose drawing COUNT
    differs is a failure: that would mean the two editions do not share
    artwork and every conclusion below about "the German twin" is void. A
    non-plot page's difference is reported and not counted.
    """
    same, delta, bad = [], [], 0
    if eng.page_count != ger.page_count:
        print(f"  [FAIL] page counts differ: {eng.page_count} vs {ger.page_count}")
        return same, delta, 1
    for i in range(eng.page_count):
        pno = i + 1
        a, b = _geometry(eng[i]), _geometry(ger[i])
        if len(a) != len(b):
            tag = "FAIL" if pno in PLOT_PAGES else "note"
            print(f"  [{tag}] p{pno}: {len(a)} drawn points in English, "
                  f"{len(b)} in German"
                  + ("" if pno in PLOT_PAGES else " -- prose/colophon page, not asserted"))
            bad += 1 if pno in PLOT_PAGES else 0
            continue
        if a == b:
            same.append(pno)
        else:
            d = max(max(abs(x - y) for x, y in zip(u, v)) for u, v in zip(a, b))
            delta.append((pno, d))
    return same, delta, bad


# ---------------------------------------------------------------------------
#  Panel readers
# ---------------------------------------------------------------------------

#: ⚠ ONE PRINTED AXIS LABEL ON THIS SHEET IS WRONG, AND IT IS WRONG BY 0.9 OF A
#: DECADE. The RSX II 50 spectral panel's ordinate reads 2.0 / 1.0 / 0 / «- 0.1»
#: where its two sibling columns on the same page read 2.0 / 1.0 / 0 / -1.0.
#: The sheet proves the typo against itself: those four ticks are printed at
#: y 76.83, 108.63, 140.43 and 172.22 -- three intervals of EXACTLY 31.80 pt --
#: so a uniformly ruled axis puts -1.0 at the fourth tick and «- 0.1» cannot be
#: where it is drawn. Left in, the least-squares fit compresses the whole
#: ordinate and the three records come back spanning -0.14..1.47 lg instead of
#: the -0.96..1.41 the sibling columns give.
#:
#: The label is VETOED, not corrected: the fit is made on the three labels that
#: agree, and the calibration is then required to PREDICT -1.00 at the fourth
#: tick's own y. That turns Agfa's typo into a check instead of a patch, and it
#: is asserted below rather than trusted.
#: (profile, panel) -> (label y centre, tolerance, why). `_words` reports the
#: y CENTRE of each token, so 175.23 is the midpoint of the 172.22-178.24 box.
_VETO = {
    ("AGFA_RSX_II_50", "spectral"): (
        175.23, 1.5,
        "printed «- 0.1» where the uniform 31.80 pt tick pitch and both "
        "sibling columns require -1.0"),
}
#: (profile, panel) -> (y, expected value, tolerance) the vetoed tick must be
#: predicted to carry once the fit is made without it.
_VETO_PREDICT = {("AGFA_RSX_II_50", "spectral"): (175.23, -1.0, 0.06)}


def _panel(pg, ws, name, band, xlo, xhi, logx=False, logy=False, veto=None):
    if veto is not None:
        yc, tol = veto[0], veto[1]
        ws = [w for w in ws if abs(w[2] - yc) > tol]
    return G._calibrated(pg, ws, name, band, xlo, xhi, logx=logx, logy=logy)


def read_reversal(pg, ws, xlo, xhi, profile):
    """One RSX II column: spectral sensitivity, spectral density, sharpness,
    colour density curves."""
    out = {}
    for key, band, log in (("spectral", BANDS["spectral"], False),
                           ("density", BANDS["density"], False),
                           ("sharpness", BANDS["sharpness"], True),
                           ("curves", BANDS["curves"], False)):
        p, err = _panel(pg, ws, key, band, xlo, xhi, logx=log, logy=log,
                        veto=_VETO.get((profile, key)))
        if p is None:
            out[key] = ("err", err)
            continue
        got = G._curves_by_shape(pg, p)
        out[key] = ("ok", p, got)
    return out


def read_mono(pg, ws, xlo, xhi):
    """One APX column: spectral sensitivity, characteristic curve, sharpness,
    gamma-time family."""
    out = {}
    for key, band, log in (("spectral", BANDS_MONO["spectral"], False),
                           ("curve", BANDS_MONO["density"], False),
                           ("sharpness", BANDS_MONO["sharpness"], True)):
        p, err = _panel(pg, ws, key, band, xlo, xhi, logx=log, logy=log)
        out[key] = ("err", err) if p is None else ("ok", p, G._one_curve(pg, p))
    p, err = _panel(pg, ws, "gamma", BANDS_MONO["curves"], xlo, xhi)
    if p is None:
        out["gamma"] = ("err", err)
    else:
        segs = G._split_curves(pg, p)
        labels = G._named_labels(pg, p, DEVELOPERS)
        out["gamma"] = ("ok", p, segs, labels, G._assign(labels, segs))
    return out


def read_scala(pg, ws, xlo, xhi):
    out = {}
    p, err = _panel(pg, ws, "spectral", SCALA_BANDS["spectral"], xlo, xhi)
    out["spectral"] = ("err", err) if p is None else ("ok", p, G._one_curve(pg, p))
    p, err = _panel(pg, ws, "sharp", SCALA_BANDS["sharpness"], xlo, xhi,
                    logx=True, logy=True)
    out["sharpness"] = ("err", err) if p is None else ("ok", p, G._one_curve(pg, p))
    p, err = _panel(pg, ws, "curves", SCALA_BANDS["curves"], xlo, xhi)
    if p is None:
        out["curves"] = ("err", err)
    else:
        segs = G._split_curves(pg, p)
        labels = G._named_labels(pg, p, SCALA_STEPS)
        out["curves"] = ("ok", p, segs, labels, G._assign(labels, segs))
    return out


# ---------------------------------------------------------------------------
#  The cross-edition comparison -- the reason this module exists
# ---------------------------------------------------------------------------

#: printed name on the 1998 sheet -> the same film's 2003 column.
#: ⚠ APX 25 and ULTRA 50 have no 2003 column: Agfa dropped both from the range
#: between the two editions, which is itself the answer to "why is there no
#: newer sheet for them".
EDITION_PAIRS = (
    ("AGFA_RSX_II_50", "AGFACHROME RSX II 50", 8, (379.0, 537.5), "reversal"),
    ("AGFA_RSX_II_100", "AGFACHROME RSX II 100", 9, (44.0, 202.0), "reversal"),
    ("AGFA_RSX_II_200", "AGFACHROME RSX II 200", 9, (220.0, 378.5), "reversal"),
    ("AGFA_APX_100", "AGFAPAN APX 100", 10, (203.0, 361.5), "mono"),
    ("AGFA_APX_400", "AGFAPAN APX 400", 10, (379.0, 537.5), "mono"),
    ("AGFA_SCALA_200X", "AGFA SCALA 200x", 9, (396.0, 554.5), "bw_rev"),
)


def axis_box_bias(page, panel, lo, hi):
    """The ordinate offset a panel's own axis rectangle proves its labels carry.

    ⚠ **A TEXT BOX'S CENTRE IS NOT A DIGIT'S CENTRE, AND THE TWO AGFA EDITIONS
    SET THEIR LABELS AT DIFFERENT POINT SIZES, SO THE ERROR DOES NOT CANCEL.**
    `_axis_labels` takes each label's y as the centre of its bounding box. That
    box runs from the ascender line to the descender line, while the digits sit
    on the baseline, so the optical centre of "2.0" is ABOVE its box centre by
    a fraction of the font size. It is a pure OFFSET -- every label is displaced
    the same way, so the fitted scale is right and only the intercept moves --
    and it is invisible inside one document.

    Between documents it is not invisible. The 1998 sheet sets its ordinate in
    a 7.69 pt box and the 2003 sheet in a 6.01 pt one, and measured against the
    RSX II 100 colour-density panel's own axis rectangle -- which must span
    exactly 0.0 to 4.0 D -- the label fits come out at

        1998   top 3.9808   bottom -0.0199     ->  0.020 D LOW
        2003   top 4.0153   bottom +0.0154     ->  0.015 D HIGH

    a combined 0.035 D, which is essentially the whole 0.038 D by which the two
    editions' traced curves appeared to disagree. Removing it is the difference
    between "Agfa redrew these curves" and "they did not".

    The rectangle is found, not assumed: of every drawn rect inside the panel,
    the one whose mapped extremes come closest to the outermost printed labels
    wins, and it is rejected outright if it misses either end by more than
    0.10 in axis units. Returns (bias, rect, residual) or (0.0, None, None).
    """
    best = None
    for dr in page.get_drawings():
        if dr["type"] not in ("s", "fs") or not A4._is_ink(dr):
            continue
        r = dr["rect"]
        if r.width < 30.0 or r.height < 20.0:
            continue
        if not (r.x0 >= panel.rect.x0 - 40 and r.x1 <= panel.rect.x1 + 40
                and r.y0 >= panel.rect.y0 - 40 and r.y1 <= panel.rect.y1 + 40):
            continue
        top, bot = float(panel.Y(r.y0)), float(panel.Y(r.y1))
        err = 0.5 * ((top - hi) + (bot - lo))
        miss = max(abs(top - hi - err), abs(bot - lo - err))
        if miss > 0.10:
            continue
        if best is None or abs(err) < abs(best[0]) or miss < best[2]:
            best = (err, r, miss)
    if best is None:
        return 0.0, None, None
    return best[0], best[1], best[2]


def _resample(px, py, panel, grid, logx=False):
    """Sample one traced curve's ordinate at `grid` abscissa values."""
    x = panel.X(px)
    y = panel.Y(py)
    if logx:
        x = 10.0 ** x
    o = np.argsort(x)
    x, y = x[o], y[o]
    lo, hi = x.min(), x.max()
    inside = [g for g in grid if lo - 1e-9 <= g <= hi + 1e-9]
    if len(inside) < 3:
        return None, None
    return np.array(inside), np.interp(inside, x, y)


def _delta(a_panel, a_xy, b_panel, b_xy, grid, logx=False,
           bias_a=0.0, bias_b=0.0):
    """(n, rms, max) difference between the same curve in two editions.

    `bias_a` / `bias_b` are the per-panel label offsets from `axis_box_bias`
    and are SUBTRACTED before differencing, so what is compared is each
    edition's drawing against its own printed axis rather than against its own
    label-box centres.
    """
    ga, ya = _resample(a_xy[0], a_xy[1], a_panel, grid, logx)
    gb, yb = _resample(b_xy[0], b_xy[1], b_panel, grid, logx)
    if ga is None or gb is None:
        return None
    common = np.intersect1d(ga, gb)
    if len(common) < 3:
        return None
    ia = np.searchsorted(ga, common)
    ib = np.searchsorted(gb, common)
    d = (ya[ia] - bias_a) - (yb[ib] - bias_b)
    return len(common), float(np.sqrt((d ** 2).mean())), float(np.abs(d).max())


#: profile -> the 2003 page it was traced from, so the de-bias can be measured
#: on the same page the curve came off.
_NEW_PAGE = {c[0]: c[2] for c in COLUMNS}


def compare_editions(old_doc, new_doc, new_traced):
    """Trace the 1998 sheet's copies of the same six columns and diff them.

    ⚠ THIS IS NOT A TRACING CHECK, IT IS A PRODUCT-HISTORY CHECK. Both readers
    are known to work; what is unknown is whether Agfa redrew the curves
    between 09/1998 and 07/2003. Agreement to a few hundredths says the printed
    resolving-power revision was a measurement change on an unchanged emulsion.
    Disagreement says the film moved, and then no 2003 number may be written
    beside a 1998-traced curve without saying so.
    """
    print("\n  == cross-edition comparison, «Technical Data PF» 09/1998 "
          "against F-PF-E4 08/2004")
    lam_grid = np.arange(400.0, 701.0, 10.0)
    out = {}
    for profile, printed, page_no, (xlo, xhi), kind in EDITION_PAIRS:
        pg = old_doc[page_no - 1]
        ws = G._words(pg)
        new = new_traced.get(profile)
        if new is None:
            continue
        npg = new_doc[_NEW_PAGE[profile] - 1]
        print(f"\n     -- {printed}")
        rec = {}

        # -- spectral sensitivity, every layer the column has ---------------
        band = G.SCALA_BANDS["spectral"] if kind == "bw_rev" else G.BANDS["spectral"]
        op, err = G._calibrated(pg, ws, "spectral", band, xlo, xhi)
        if op is None:
            print(f"        spectral   [--] 1998 side: {err}")
        elif kind == "reversal":
            oldc = G._curves_by_shape(pg, op)
            newc = new["spectral"][2] if new["spectral"][0] == "ok" else {}
            ba, _, _ = axis_box_bias(pg, op, -1.0, 2.0)
            bb, _, _ = axis_box_bias(npg, new["spectral"][1], -1.0, 2.0)
            for lay in ("b", "g", "r"):
                if lay in oldc and lay in newc:
                    d = _delta(new["spectral"][1], newc[lay], op, oldc[lay],
                               lam_grid, bias_a=bb, bias_b=ba)
                    if d:
                        print(f"        spectral {lay}  {d[0]:2d} nm samples  "
                              f"rms {d[1]:.4f} lg  max {d[2]:.4f} lg")
                        rec[f"spectral_{lay}"] = d
        else:
            oldxy = G._one_curve(pg, op)
            newst = new["spectral"]
            if oldxy is not None and newst[0] == "ok" and newst[2] is not None:
                ba, _, _ = axis_box_bias(pg, op, -1.0, 2.0)
                bb, _, _ = axis_box_bias(npg, newst[1], -1.0, 2.0)
                d = _delta(newst[1], newst[2], op, oldxy, lam_grid,
                           bias_a=bb, bias_b=ba)
                if d:
                    print(f"        spectral    {d[0]:2d} nm samples  "
                          f"rms {d[1]:.4f} lg  max {d[2]:.4f} lg")
                    rec["spectral"] = d

        # -- the density family ---------------------------------------------
        if kind == "mono":
            op, err = G._calibrated(pg, ws, "curve", G.BANDS["density"], xlo, xhi)
            oldxy = None if op is None else G._one_curve(pg, op)
            newst = new["curve"]
            if oldxy is not None and newst[0] == "ok" and newst[2] is not None:
                grid = np.arange(-3.6, 1.81, 0.10)
                ba, _, _ = axis_box_bias(pg, op, 0.0, 3.0)
                bb, _, _ = axis_box_bias(npg, newst[1], 0.0, 3.0)
                d = _delta(newst[1], newst[2], op, oldxy, grid,
                           bias_a=bb, bias_b=ba)
                if d:
                    print(f"        curve       {d[0]:2d} logE samples  "
                          f"rms {d[1]:.4f} D  max {d[2]:.4f} D")
                    rec["curve"] = d
        elif kind == "reversal":
            op, err = G._calibrated(pg, ws, "curves", G.BANDS["curves"], xlo, xhi)
            if op is not None:
                oldc = G._curves_by_shape(pg, op)
                newc = new["curves"][2] if new["curves"][0] == "ok" else {}
                grid = np.arange(-2.8, 2.81, 0.10)
                ba, _, _ = axis_box_bias(pg, op, 0.0, 4.0)
                bb, _, _ = axis_box_bias(npg, new["curves"][1], 0.0, 4.0)
                for lay in ("b", "g", "r"):
                    if lay in oldc and lay in newc:
                        d = _delta(new["curves"][1], newc[lay], op, oldc[lay],
                                   grid, bias_a=bb, bias_b=ba)
                        if d:
                            print(f"        density {lay}   {d[0]:2d} logE samples  "
                                  f"rms {d[1]:.4f} D  max {d[2]:.4f} D")
                            rec[f"density_{lay}"] = d
        else:
            op, err = G._calibrated(pg, ws, "curves", G.SCALA_BANDS["curves"], xlo, xhi)
            if op is not None and new["curves"][0] == "ok":
                oldsegs = G._split_curves(pg, op)
                oldlab = G._assign(G._named_labels(pg, op, SCALA_STEPS), oldsegs)
                np_, newsegs, _, newown = (new["curves"][1], new["curves"][2],
                                           new["curves"][3], new["curves"][4])
                ba, _, _ = axis_box_bias(pg, op, 0.0, 3.0)
                bb, _, _ = axis_box_bias(npg, np_, 0.0, 3.0)
                oldmax = {}
                for i, (xs, ys) in enumerate(oldsegs):
                    for nm in oldlab[i]:
                        oldmax[nm] = float(op.Y(ys).max()) - ba
                for i, (xs, ys) in enumerate(newsegs):
                    for nm in newown[i]:
                        if nm in oldmax:
                            nv = float(np_.Y(ys).max()) - bb
                            print(f"        Dmax {nm:9s} 1998 {oldmax[nm]:.3f}  "
                                  f"2003 {nv:.3f}  delta {nv - oldmax[nm]:+.3f} D")
                            rec[f"dmax_{nm}"] = (oldmax[nm], nv)

        # -- sharpness --------------------------------------------------------
        band = G.SCALA_BANDS["sharpness"] if kind == "bw_rev" else G.BANDS["sharpness"]
        op, err = G._calibrated(pg, ws, "sharp", band, xlo, xhi, logx=True, logy=True)
        if op is not None:
            if kind == "reversal":
                oldc = G._curves_by_shape(pg, op)
                oldxy = oldc.get("g")
            else:
                oldxy = G._one_curve(pg, op)
            newst = new["sharpness"]
            newxy = None
            if newst[0] == "ok":
                newxy = newst[2].get("g") if kind == "reversal" else newst[2]
            if oldxy is not None and newxy is not None:
                of = G._f50(10.0 ** op.X(oldxy[0]), 10.0 ** op.Y(oldxy[1]))
                nf = G._f50(10.0 ** newst[1].X(newxy[0]), 10.0 ** newst[1].Y(newxy[1]))
                print(f"        f50         1998 {of:.1f}  2003 {nf:.1f}  "
                      f"delta {nf - of:+.1f} lines/mm")
                rec["f50"] = (of, nf)
        out[profile] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()

    root = Path(ns.root).resolve() / "PDF" / "PROFILES"
    eng_p, ger_p = root / SHEET, root / TWIN
    for p in (eng_p, ger_p):
        if not p.is_file():
            print(f"  [SKIP] source not present: {p}")
            return 0
    eng, ger = pymupdf.open(eng_p), pymupdf.open(ger_p)
    print(f"[i] {SOURCE}\n")

    bad = 0

    # ---- the twin check --------------------------------------------------
    same, delta, tb = twin_report(eng, ger)
    bad += tb
    print(f"  [OK  ] twin: {len(same)} of {eng.page_count} pages have "
          f"byte-identical drawing coordinates ({', '.join('p%d' % i for i in same)}); "
          + "; ".join(f"p{i} differs by {d:.2f} pt" for i, d in delta))
    for want in (7, 8):
        if want not in same:
            print(f"  [FAIL] p{want} was expected to be identical in both editions")
            bad += 1

    # ---- the columns -----------------------------------------------------
    traced = {}
    for profile, printed, page_no, (xlo, xhi), kind in COLUMNS:
        pg = eng[page_no - 1]
        ws = G._words(pg)
        print(f"\n  -- p{page_no} {printed}  [{kind}] -> {profile}")
        if kind == "reversal":
            res = read_reversal(pg, ws, xlo, xhi, profile)
            traced[profile] = res
            pred = _VETO_PREDICT.get((profile, "spectral"))
            if pred is not None and res["spectral"][0] == "ok":
                p = res["spectral"][1]
                got = float(p.Y(pred[0]))
                ok = abs(got - pred[1]) <= pred[2]
                print(f"     veto-check the mis-set «- 0.1» tick: the fit made "
                      f"without it predicts {got:+.3f} lg there, against the "
                      f"{pred[1]:+.1f} its siblings print "
                      f"[{'OK' if ok else 'FAIL'}]")
                if not ok:
                    bad += 1
            for key in ("spectral", "density", "sharpness", "curves"):
                st = res[key]
                if st[0] == "err":
                    print(f"     {key:10s} [--] {st[1]}")
                    bad += 1
                    continue
                p, got = st[1], st[2]
                drop = f", dropped {p.xdrop}x/{p.ydrop}y label" if (p.xdrop or p.ydrop) else ""
                bits = []
                for lay in ("b", "g", "r"):
                    if lay not in got:
                        continue
                    xs, ys = got[lay]
                    if key == "sharpness":
                        f, t = 10.0 ** p.X(xs), 10.0 ** p.Y(ys)
                        bits.append(f"{lay}: f50 {G._f50(f, t):.1f}, peak {t.max():.0f} %")
                    elif key == "curves":
                        # ⚠ REVERSAL: the abscissa has to be negated before the
                        # fit or every record returns gamma ~0.00, because
                        # `tone_of` searches for the steepest RISING chord and
                        # these curves fall. See `agfa_1998_curves._FlipX`.
                        dmin, chord, span, ft = G.tone_reversal(p, (xs, ys))
                        if ft is None:
                            bits.append(f"{lay}: dmin {dmin:.3f} chord {chord:.3f} "
                                        f"span {span:.2f} (no fit -- SciPy absent)")
                        else:
                            dmn, gam, tx, tk, sx, sk, rms = ft
                            bits.append(
                                f"{lay}: dmin {dmn:.3f} gamma {gam:.3f} "
                                f"toe {tx:+.3f}/{tk:.3f} sh {sx:+.3f}/{sk:.3f} "
                                f"rms {rms:.4f} D")
                    else:
                        a, b = p.X(xs), p.Y(ys)
                        bits.append(f"{lay}: {a.min():.0f}-{a.max():.0f} nm / "
                                    f"{b.min():.2f}..{b.max():.2f}")
                if not bits:
                    print(f"     {key:10s} [--] no dash-keyed layers")
                    bad += 1
                    continue
                print(f"     {key:10s} res {p.xres:.4f}/{p.yres:.4f}{drop}")
                for b_ in bits:
                    print(f"        {b_}")
        elif kind == "mono":
            res = read_mono(pg, ws, xlo, xhi)
            traced[profile] = res
            for key in ("spectral", "curve", "sharpness"):
                st = res[key]
                if st[0] == "err" or st[2] is None:
                    print(f"     {key:10s} [--] {st[1] if st[0]=='err' else 'no curve'}")
                    bad += 1
                    continue
                p, xy = st[1], st[2]
                if key == "sharpness":
                    f, t = 10.0 ** p.X(xy[0]), 10.0 ** p.Y(xy[1])
                    print(f"     {key:10s} peak {t.max():.0f} % , f50 {G._f50(f, t):.1f} "
                          f"lines/mm  res {p.xres:.4f}/{p.yres:.4f}")
                else:
                    a, b = p.X(xy[0]), p.Y(xy[1])
                    print(f"     {key:10s} {a.min():+.2f}..{a.max():+.2f} / "
                          f"{b.min():.3f}..{b.max():.3f}  {len(a)} pts  "
                          f"res {p.xres:.4f}/{p.yres:.4f}")
            st = res["gamma"]
            if st[0] == "err":
                print(f"     gamma-time [--] {st[1]}")
                bad += 1
            else:
                p, segs, labels, owners = st[1], st[2], st[3], st[4]
                print(f"     gamma-time {len(segs)} curves for {len(labels)} printed "
                      f"names  res {p.xres:.3f} min / {p.yres:.4f} gamma")
                for i, (xs, ys) in enumerate(segs):
                    t, g = p.X(xs), p.Y(ys)
                    who = " + ".join(owners[i]) or "⚠ UNLABELLED"
                    vals = "  ".join(
                        f"g({tt:.0f})={float(np.interp(tt, t, g)):.3f}"
                        for tt in (4.0, 6.0, 8.0, 10.0) if t.min() <= tt <= t.max())
                    print(f"        {t.min():5.1f}-{t.max():5.1f} min  {who:18s} {vals}")
        else:
            res = read_scala(pg, ws, xlo, xhi)
            traced[profile] = res
            for key in ("spectral", "sharpness"):
                st = res[key]
                if st[0] == "err" or st[2] is None:
                    print(f"     {key:10s} [--] {st[1] if st[0]=='err' else 'no curve'}")
                    bad += 1
                    continue
                p, xy = st[1], st[2]
                if key == "sharpness":
                    f, t = 10.0 ** p.X(xy[0]), 10.0 ** p.Y(xy[1])
                    print(f"     {key:10s} peak {t.max():.0f} %, f50 {G._f50(f, t):.1f} "
                          f"lines/mm  res {p.xres:.4f}/{p.yres:.4f}")
                else:
                    a, b = p.X(xy[0]), p.Y(xy[1])
                    print(f"     {key:10s} {a.min():.0f}-{a.max():.0f} nm, peak "
                          f"{a[int(np.argmax(b))]:.0f} nm  res {p.xres:.3f}/{p.yres:.4f}")
            st = res["curves"]
            if st[0] == "err":
                print(f"     curves     [--] {st[1]}")
                bad += 1
            else:
                p, segs, labels, owners = st[1], st[2], st[3], st[4]
                print(f"     curves     {len(segs)} curves for {len(labels)} printed "
                      f"step names  res {p.xres:.3f} dec / {p.yres:.4f} D")
                for i, (xs, ys) in enumerate(segs):
                    lx, dd = p.X(xs), p.Y(ys)
                    who = " + ".join(owners[i]) or "⚠ UNLABELLED"
                    print(f"        lgE {lx.min():+.2f}..{lx.max():+.2f}  "
                          f"D {dd.min():.3f}..{dd.max():.3f}  {who}")

    # ---- the cross-edition comparison ------------------------------------
    old_p = root / G.SHEET
    if old_p.is_file():
        compare_editions(pymupdf.open(old_p), eng, traced)
    else:
        print(f"\n  [note] 1998 sheet absent ({old_p}); no cross-edition check")

    print()
    if bad:
        print(f"  [FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("  [OK  ] every panel on pp8-9 read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
