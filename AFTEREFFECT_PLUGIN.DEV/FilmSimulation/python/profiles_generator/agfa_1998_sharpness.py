#!/usr/bin/env python3
"""The eleven AGFA "Sharpness" panels of Technical Data PF, 2026-09-06h (queue A5).

    PDF/PROFILES/AGFA/agfa_films.pdf -- Agfa-Gevaert, «Technical Data PF»,
    1st edition 09/1998, pages 7-10.

⚠⚠ ELEVEN MEASURED MTF CURVES THAT THIS DATABASE HAD NEVER READ. Every AGFA
stock in the database carried `mtf_measured = False` and a RED estimated f50
triple, while this document has been in the corpus since 2026-09-01 printing a
"Sharpness" panel -- transfer factor against lines per mm, which is an MTF
curve under another name -- for all eleven of them. The 2026-09-01 pass took
this sheet's characteristic curves, spectral sets, granularity and reciprocity
and left the sharpness panels unread.

⚠ THE PANEL IS CALLED "SHARPNESS" AND THE AXES ARE CAPTIONED IN GERMAN HOUSE
STYLE, which is most of why it was missed: "Transfer factor (%)" is modulation
transfer and "Lines (mm)" is cycles per mm. That second equivalence is not an
assumption here -- queue row G6 settled it on the authority of the
International Commission for Optics (Ingelstam 1961, PS&E 5(5) p282): the
German «Linien pro mm» IS cycles per mm, the only halving being a television
line.

⚠ EVERY LADDER IS READ FROM THE PANEL'S OWN TEXT LABELS, never assumed from the
family. The layout is regular to a fraction of a point across all four pages,
which makes it tempting to hardcode one grid and apply it eleven times -- and
that is exactly how a page-offset error becomes eleven wrong stocks.

⚠⚠ AND THE DUPLICATE-ARTWORK CHECK IS MANDATORY, NOT DEFENSIVE. `NotFound.md`
row 5d already records that APX 100 and APX 400 SHARE this panel on this very
sheet: the same 73-point path translated 175.21 pt, every y offset identical to
0.0000. Two different films cannot share a measured MTF, so any pair that comes
back identical is reported and refused rather than stored twice.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PDF = "PDF/PROFILES/AGFA/agfa_films.pdf"

#: (page index, plot-frame x0, film). The sharpness row is the third of four on
#: every page and spans y 339.8..425.4; the three columns are the three films
#: the page's heading names, left to right.
PANELS = [
    (6, 70.4, "AGFA_OPTIMA_100", "AGFACOLOR OPTIMA II 100"),
    (6, 246.7, "AGFA_OPTIMA_200", "AGFACOLOR OPTIMA II 200"),
    (6, 422.8, "AGFA_OPTIMA_400", "AGFACOLOR OPTIMA II 400"),
    (7, 53.4, "AGFA_PORTRAIT_160", "AGFACOLOR PORTRAIT XPS 160"),
    (7, 229.7, "AGFA_ULTRA_50", "AGFACOLOR ULTRA 50"),
    (7, 405.8, "AGFA_RSX_II_50", "AGFACHROME RSX II 50"),
    (8, 70.5, "AGFA_RSX_II_100", "AGFACHROME RSX II 100"),
    (8, 246.7, "AGFA_RSX_II_200", "AGFACHROME RSX II 200"),
    (9, 53.4, "AGFA_APX_25", "AGFAPAN APX 25"),
    (9, 229.7, "AGFA_APX_100", "AGFAPAN APX 100"),
    (9, 405.8, "AGFA_APX_400", "AGFAPAN APX 400"),
]
ROW_Y0, ROW_Y1 = 339.8, 425.4
PLOT_W = 126.4

#: ⚠ SCALA 200x BREAKS THE GRID AND IS THE ONLY ONE THAT DOES. Page 9's third
#: column carries a push/pull table the other eleven do not have, which pushes
#: its four panels up and shrinks them; its sharpness plot sits in the SECOND
#: row band at 227.3-312.5 instead of the third at 339.8-425.4. Reading it on
#: the family grid returns the spectral-density panel, which is a curve, in the
#: right ink, of entirely the wrong quantity.
ODD_PANELS = [
    (8, (423.2, 227.3, 549.2, 312.5), "AGFA_SCALA_200X", "AGFA SCALA 200x"),
]

#: ⚠⚠ THE SAME PANEL EXISTS IN THE 2004 EDITION AND MUST BE COMPARED, NOT
#: IGNORED. `AGFA stocks.pdf` is «Technical Data: Agfa Professional Films»
#: F-PF-E4, 4th edition 08/2004, and it prints its own Sharpness row. This
#: project has already been here once: on 2026-09-01 the RESOLVING POWERS were
#: raised from the 1998 to the 2004 figures on all three RSX II stocks --
#: 125->135, 125->130, 110->120 -- because the older value was not wrong, it was
#: older. If the two editions' CURVES differ the same rule applies to f50, and
#: if they are one drawing then the 1998 trace stands and the agreement is not
#: corroboration. Either way it has to be measured.
PDF_2004 = "PDF/PROFILES/AGFA/AGFA stocks.pdf"
PANELS_2004 = [
    (6, 70.5, "AGFA_OPTIMA_100"), (6, 245.7, "AGFA_OPTIMA_200"),
    (6, 422.2, "AGFA_OPTIMA_400"), (5, 405.1, "AGFA_PORTRAIT_160"),
    (7, 53.1, "AGFA_RSX_II_50"), (7, 229.4, "AGFA_RSX_II_100"),
    (7, 404.4, "AGFA_RSX_II_200"), (8, 70.6, "AGFA_APX_100"),
    (8, 245.9, "AGFA_APX_400"), (8, 422.0, "AGFA_SCALA_200X"),
]
ROW04_Y0, ROW04_Y1 = 327.6, 415.3
#: The 2004 plot frames are ruled at y0 328.743; the band above is the search
#: window, this is the seed `_frame` matches against.
FRAME04_Y0 = 328.743

#: What each edition's panels read, pinned so a rerun that disagrees FAILS.
#:
#: ⚠⚠ REPINNED TWICE ON 2026-09-06i, FOR TWO SEPARATE CALIBRATION DEFECTS, and
#: the superseded numbers are kept here as the record rather than deleted:
#:   * as first published -- frequency ladder fitted through the nudged "100"
#:     tick (`_logfit`): 45.18 / 48.55 / 49.08 Optima, 37.13 Portrait,
#:     44.04 Ultra, 30.00 / 32.67 / 21.62 RSX II, 31.31 Scala, 81.56 APX 25.
#:   * after the rejection but still label-calibrated: 43.74 / 46.96 / 47.46,
#:     36.06, 42.65, 29.23 / 31.79 / 21.18, 30.49, 78.24.
#: The values below are frame-calibrated (`_calibrate`) and sit ~0.5 % above
#: the label-calibrated pass, which is the size of the deliberate difference
#: between this module and `agfa_1998_curves.py`.
EXPECTED: dict[str, tuple[float, float]] = {
    "AGFA_APX_100": (57.96, 2.27),
    "AGFA_APX_25": (78.67, 2.31),
    "AGFA_APX_400": (57.96, 2.27),
    "AGFA_OPTIMA_100": (43.95, 2.96),
    "AGFA_OPTIMA_200": (47.19, 2.62),
    "AGFA_OPTIMA_400": (47.70, 2.86),
    "AGFA_PORTRAIT_160": (36.24, 2.47),
    "AGFA_RSX_II_100": (31.92, 2.08),
    "AGFA_RSX_II_200": (21.26, 2.49),
    "AGFA_RSX_II_50": (29.37, 2.31),
    "AGFA_SCALA_200X": (30.64, 2.14),
    "AGFA_ULTRA_50": (42.86, 3.12),
}

#: The 2004 edition's own readings. ⚠ SEVEN OF THESE TEN NOW REPRODUCE THE 1998
#: FIGURE TO 0.1 %, which is the whole finding: they are one drawing printed
#: twice, and only the three Optima panels were redrawn. Three of these are
#: what the database stores; the rest are the comparison.
EXPECTED_2004: dict[str, tuple[float, float]] = {
    "AGFA_APX_100": (57.95, 2.27),
    "AGFA_APX_400": (57.99, 2.27),
    "AGFA_OPTIMA_100": (43.70, 2.95),
    "AGFA_OPTIMA_200": (47.99, 2.67),
    "AGFA_OPTIMA_400": (47.42, 2.87),
    "AGFA_PORTRAIT_160": (36.22, 2.47),
    "AGFA_RSX_II_100": (31.95, 2.09),
    "AGFA_RSX_II_200": (21.29, 2.49),
    "AGFA_RSX_II_50": (29.36, 2.31),
    "AGFA_SCALA_200X": (30.63, 2.13),
}

#: The adjacency overshoot each adopted panel reads, on the SAME drawing its
#: f50 comes from. ⚠ THESE TWO NUMBERS MUST TRAVEL TOGETHER: they are two
#: readings of one curve, and a stock holding f50 from 2004 and an overshoot
#: from 1998 describes a curve on neither page.
EXPECTED_PEAK: dict[str, float] = {
    "AGFA_APX_25": 0.0549,        # 1998
    "AGFA_OPTIMA_100": 0.1111,    # 2004, redrawn
    "AGFA_OPTIMA_200": 0.1043,    # 2004, redrawn
    "AGFA_OPTIMA_400": 0.0703,    # 2004, redrawn
    "AGFA_PORTRAIT_160": 0.0681,  # 1998
    "AGFA_RSX_II_100": 0.0852,    # 1998
    "AGFA_RSX_II_200": 0.1067,    # 1998
    "AGFA_RSX_II_50": 0.0470,     # 1998
    "AGFA_SCALA_200X": 0.0248,    # 1998
    "AGFA_ULTRA_50": 0.1487,      # 1998
}


def _labels(page, FR):
    """The panel's own tick labels: (frequency ladder, response ladder).

    Frequency labels sit BELOW the frame, response labels to its LEFT. Both are
    read by position rather than by order, because the text layer emits them
    interleaved with the axis captions.
    """
    fx, fy = {}, {}
    for w in page.get_text("words"):
        x0, y0, x1, y1, t = w[0], w[1], w[2], w[3], w[4]
        if not re.fullmatch(r"\d{1,3}", t):
            continue
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if FR.x0 - 4 <= cx <= FR.x1 + 6 and FR.y1 < cy < FR.y1 + 14:
            fx[float(t)] = cx
        if FR.x0 - 20 <= cx < FR.x0 - 1 and FR.y0 - 6 <= cy <= FR.y1 + 6:
            fy[float(t)] = cy
    return fx, fy


#: Rejection tolerance for one tick label, in POINTS on the page. 0.8 pt sits
#: just above the +-0.77 pt sawtooth Agfa's typesetter leaves on a correctly
#: placed ladder, and far below the 3.37 pt the one bad label is out by.
#: ⚠⚠ THE SHEET'S OWN DEFINITION OF WHAT THESE PANELS MEASURE, p4, ASSERTED ON
#: EVERY RUN AND UNPINNED UNTIL 2026-09-07:
#:
#:     «Sharpness -- This is an MTF (Modulation Transfer Function) chart, which
#:      indicates the image sharpness. The higher the transfer factor in %, the
#:      lower the transfer losses are. Reference: -- exposure: daylight --
#:      densitometry: visual filter (Vlambda)»
#:
#: The 2026-09-06j batch found the SAME statement on p4 of the AGFACOLOR Vista
#: sheet and used it to retire a refusal on AGFA_VISTA_200, pinning it there as
#: `agfa_vista_mtf.P4_DEFINITION`. It is on THIS sheet too -- the 1998 edition,
#: the one all twelve adopted panels come from -- and nothing asserted it. The
#: owner noticed. So the primary authority for reading these panels as MTF now
#: has a guard on the document that actually carries the panels, not only on a
#: sister sheet two years later.
#: ⚠ WHAT IT DOES NOT SAY is what a «Linie» counts, which is queue G6's
#: question and stays answered by inference plus the RP*sqrt(rms) scale test.
P4_DEFINITION = "This is an MTF (Modulation Transfer Function) chart"


LABEL_TOL_PT = 0.8

#: A ladder that has had a label rejected must still meet the panel's own frame
#: to this fraction. See `_logfit`.
FRAME_TOL = 0.002

#: Two frame-normalised curves closer than this, in fractions of frame height,
#: are ONE DRAWING. Twice the worst agreement among the eight panels the 2004
#: sheet reprints unchanged (0.0009); fifty times under the two it redrew.
REDRAW_TOL = 0.002


def _logfit(m, frame_span=None):
    """Least squares over the printed tick labels, with worst-point rejection.

    ⚠⚠ THIS WAS A PLAIN `polyfit` UNTIL 2026-09-06i AND IT PUT EVERY f50 THIS
    MODULE PUBLISHED ABOUT 4 % HIGH -- all ten adopted stocks.

    The frequency ladder reads 2 3 5 10 20 30 50 100 and AGFA NUDGE THE LAST
    ONE LEFT so its three glyphs stay inside the column. Even nudged, its box
    overhangs the frame -- x1 181.17 against a frame edge at 179.80 -- and
    centred on its own tick it would overhang by 4.7 pt. So its centre lies
    3.37 pt left of the tick while every other label is within 0.77 pt, and one
    line through all eight tilts to **73.257 pt/decade instead of 74.422**,
    identically on all eleven 1998 panels and all ten 2004 ones.

    ⚠ 1.6 % OF SCALE IS NOT 1.6 % OF f50. Tilting the line moves the intercept
    too, and at the frequency where these curves cross 50 % the two errors add
    instead of cancelling: APX 25 read 81.6 c/mm for a true 78.2, RSX II 200
    22.3 for 21.8. Nothing in the output looked wrong. The residual was printed
    every run as "1.92 pt" and the question nobody asked was 1.92 pt of WHAT --
    0.026 of a decade, on an axis where 0.3 of a decade is a factor of two.

    ⚠⚠ THE REJECTION IS NOT THE SAFEGUARD. THE SECOND WITNESS IS. Dropping the
    worst label until the residual falls is precisely how a genuinely broken
    axis gets whittled down to a fit that cannot fail, so the surviving ladder
    is tested against a quantity it was never fitted to: THE PANEL FRAME. Agfa
    rule the frame from the first labelled tick to the last, so
    `frame_span / log10(f_max / f_min)` measures the same scale independently --
    74.398 against the 74.422 the seven surviving labels give, a 0.03 % meet.
    A ladder that misses the frame by more than `FRAME_TOL` is REPORTED.

    ⚠ WHICH IS WHY THE FIT IS NO LONGER WHAT CALIBRATES THE PANEL -- see
    `_calibrate`. The rule these panels ended up needing is: **the printed
    ladder says WHICH values the axis spans, the drawn frame says WHERE.** Text
    is typeset and gets nudged; ink does not.

    `agfa_1998_curves._fit_robust` has had this rejection since 2026-09-01,
    which is why that module read these same panels correctly all along and why
    the two disagreed by exactly the 4 % above. `--assert` now cross-checks the
    two readers against each other so the pair can never silently drift again.
    """
    kept, dropped = sorted(m), []
    while len(kept) > 4:
        lv = np.log10(np.array(kept, dtype=float))
        px = np.array([m[v] for v in kept], dtype=float)
        c = np.polyfit(lv, px, 1)
        res = np.abs(np.polyval(c, lv) - px)
        if res.max() <= LABEL_TOL_PT:
            break
        i = int(res.argmax())
        dropped.append((kept[i], round(float(res[i]), 2)))
        kept.pop(i)
    lv = np.log10(np.array(kept, dtype=float))
    px = np.array([m[v] for v in kept], dtype=float)
    c = np.polyfit(lv, px, 1)
    r = float(np.abs(np.polyval(c, lv) - px).max())
    # ⚠ THE FRAME SPANS THE WHOLE PRINTED LADDER, NOT THE SURVIVING SUBSET.
    # Agfa rule the frame from the first labelled tick to the last, and the
    # last is exactly the label the rejection just threw away -- so the span
    # has to be divided by the range of `m`, not of `kept`. Dividing by the
    # survivors' range reports every panel 17.7 % out and fails all twelve.
    ferr = None
    if frame_span is not None and len(m) > 1:
        implied = abs(frame_span) / (np.log10(max(m)) - np.log10(min(m)))
        ferr = abs(abs(c[0]) / implied - 1.0)
    return c, r, len(kept), dropped, ferr


#: The plot frame is drawn at 0.52-0.53 pt on both sheets -- twice the 0.26 pt
#: gridline, half again under the 0.79 pt curve.
def _frame(page, x0, y0):
    """The panel's drawn plot frame, or None.

    ⚠⚠ THIS IS THE CALIBRATION REFERENCE AND THE TICK LABELS ARE NOT. See
    `_calibrate` for what that cost before it was true.
    """
    best = None
    for dr in page.get_drawings():
        if dr["type"] not in ("s", "fs"):
            continue
        r = dr["rect"]
        if not (abs(r.x0 - x0) < 3.0 and abs(r.y0 - y0) < 4.0):
            continue
        if r.width < 100.0 or r.height < 70.0:
            continue
        if best is None or r.width * r.height > best[0]:
            best = (r.width * r.height, r)
    return None if best is None else best[1]


def _calibrate(fx, fy, fr):
    """Map page points to (cycles/mm, transfer factor), anchored on the frame.

    ⚠⚠ THE PRINTED LADDER SAYS *WHICH* VALUES THE AXIS SPANS. THE DRAWN FRAME
    SAYS *WHERE*. That split is the whole lesson of 2026-09-06i and it was
    learned twice in one afternoon:

      1. The frequency ladder's "100" is nudged left off its own tick to keep
         three glyphs inside the column -- 3.37 pt, see `_logfit`.
      2. ⚠ THE 2004 EDITION'S RESPONSE LADDER IS SET ~0.9 pt LOW, ALL SIX
         LABELS TOGETHER. Nothing about that looks like a defect: the labels
         are evenly spaced, the fit residual is a healthy 0.52 pt, and the
         slope is within 0.7 % of the 1998 page's. But 0.9 pt on a 73 pt
         decade is 1.029x of transfer, so every 2004 curve read ~3 % high,
         which pushed its 50 % crossing ~3.5 % right.

    ⚠⚠ AND DEFECT 2 MANUFACTURED A FINDING. Read off their own labels, EIGHT of
    the ten films that appear in both editions looked to have "moved more than
    3 % between editions", and this module duly adopted the 2004 figure for
    each of them on the newer-edition precedent. Anchored on their frames
    instead, seven of the nine comparable panels agree to **0.1 % of frame
    height** -- they are ONE DRAWING REPRINTED, not a re-measurement -- and the
    f50s agree to 0.6 %. A calibration error that invents a plausible physical
    story is worse than one that produces nonsense, because nonsense gets
    looked at.

    The frame is ink ruled from the first labelled tick to the last, so the
    labels' own min and max supply the endpoint VALUES and the rect supplies
    the endpoint POSITIONS. On the 1998 page the two agree to 0.16 pt, which is
    what licences the method; the disagreement is reported either way.
    """
    fmin, fmax = min(fx), max(fx)
    rmax, rmin = max(fy), min(fy)          # y0 is the TOP of the frame
    def X(px):
        return fmin * 10.0 ** ((px - fr.x0) / fr.width
                               * np.log10(fmax / fmin))
    def Y(py):
        return rmax * 10.0 ** ((py - fr.y0) / fr.height
                               * np.log10(rmin / rmax)) / 100.0
    return X, Y


def _curve(page, FR):
    """The one thick stroked path inside the plot frame.

    ⚠ THICKNESS IS THE DISCRIMINATOR. The gridlines are 0.26-0.28 pt strokes in
    the SAME ink as the curve, which is 0.79 pt. A reader that takes every
    stroked path inside the frame gets the curve plus four rules and averages
    them into something smooth and entirely fictitious.
    """
    # ⚠ THE CONTAINMENT TEST NEEDS A TOLERANCE AND PORTRAIT XPS 160 IS WHY.
    # Its curve begins at x 53.326 against a frame edge at 53.393 -- 0.067 pt
    # outside, a rounding artefact of the drawing, and a strict `contains`
    # rejected the only real curve on the panel and reported "no thick stroked
    # path". One film silently missing out of eleven.
    OUT = pymupdf.Rect(FR.x0 - 1.5, FR.y0 - 1.5, FR.x1 + 1.5, FR.y1 + 1.5)
    best = None
    for dr in page.get_drawings():
        if not OUT.contains(dr["rect"]):
            continue
        w = dr.get("width")
        if w is None or w < 0.5:
            continue
        n = sum(1 for it in dr["items"] if it[0] == "c")
        if n < 3:
            continue
        if best is None or n > best[0]:
            best = (n, dr)
    if best is None:
        return None, None
    pts = []
    for it in best[1]["items"]:
        if it[0] == "c":
            P = [np.array([q.x, q.y]) for q in it[1:5]]
            for t in np.linspace(0.0, 1.0, 200):
                pts.append(((1 - t) ** 3) * P[0] + 3 * ((1 - t) ** 2) * t * P[1]
                           + 3 * (1 - t) * t * t * P[2] + (t ** 3) * P[3])
        elif it[0] == "l":
            pts.append(np.array([it[1].x, it[1].y]))
            pts.append(np.array([it[2].x, it[2].y]))
    return np.array(pts), best[1]


def read_all(root: Path, verbose=True):
    doc = pymupdf.open(str(root / PDF))
    out, sigs, bad = {}, {}, 0
    jobs = [(pg, pymupdf.Rect(x0, ROW_Y0, x0 + PLOT_W, ROW_Y1), n, l)
            for pg, x0, n, l in PANELS]
    jobs += [(pg, pymupdf.Rect(*fr), n, l) for pg, fr, n, l in ODD_PANELS]
    for pageno, FR, name, label in jobs:
        page = doc[pageno]
        fx, fy = _labels(page, FR)
        if len(fx) < 5 or len(fy) < 4:
            print("  [FAIL] %s: %d frequency and %d response labels found"
                  % (name, len(fx), len(fy)))
            bad += 1
            continue
        fr = _frame(page, FR.x0, FR.y0)
        if fr is None:
            print("  [FAIL] %s: no plot frame found" % name)
            bad += 1
            continue
        cf, rf, nf, fdrop, ferr = _logfit(fx, fr.width)
        cr, rr, nr, rdrop, rerr = _logfit(fy, fr.height)
        # ⚠ THE LADDER IS NOW THE WITNESS AND THE FRAME IS THE CALIBRATION --
        # the reverse of what this module did for its first three hours. A
        # frequency ladder that misses the frame is a panel this reader does not
        # understand, so it is refused rather than read.
        if ferr is not None and ferr > FRAME_TOL:
            print("  [FAIL] %s: frequency ladder misses its own frame by "
                  "%.2f %% after dropping %s" % (name, 100 * ferr, fdrop))
            bad += 1
            continue
        # ⚠ THE RESPONSE LADDER IS REPORTED, NOT ENFORCED, and 2004 is why: its
        # six labels are set 0.9 pt low as a block, which is 1.1 % here and 3 %
        # of transfer. Enforcing would refuse a perfectly good drawing; using
        # the labels to calibrate read every 2004 curve 3 % high. Reporting is
        # the honest middle, and `--assert` pins the numbers it produces.
        X, Y = _calibrate(fx, fy, fr)
        # ⚠ SQUARE CHECK, REPORTED NOT USED. Agfa draws one decade of frequency
        # and one of response at very nearly the same length here; a panel that
        # departs is reported, never corrected.
        square = abs(fr.width / np.log10(max(fx) / min(fx))
                     / (fr.height / np.log10(max(fy) / min(fy))) - 1.0)
        A, dr = _curve(page, FR)
        if A is None:
            print("  [FAIL] %s: no thick stroked path in the plot frame" % name)
            bad += 1
            continue
        # ⚠⚠ THE ARTWORK SIGNATURE IS THE Y PROFILE ALONE, AND THAT MATTERS.
        # A signature over BOTH coordinates missed the 2004 edition's copy of
        # this defect: APX 400's curve there is APX 100's translated 175.17 pt
        # with every y coordinate identical to 0.0000000, but the x offsets
        # drift by 0.096 pt across the path -- a 0.08 % horizontal stretch,
        # invisible in the plot and enough to make an x-inclusive signature
        # unique. What a shared MTF drawing means is that the RESPONSE profile
        # is the same curve, so the response axis is the one to fingerprint.
        yy = A[::20, 1] - A[0, 1]
        sig = tuple(np.round(yy, 4).tolist())
        sigs.setdefault(sig, []).append(name)

        f, r = X(A[:, 0]), Y(A[:, 1])
        o = np.argsort(f)
        f, r = f[o], r[o]
        below = np.flatnonzero(r < 0.5)
        if not len(below) or below[0] == 0:
            print("  [FAIL] %s: the curve never crosses 50 %% inside the panel "
                  "(%.1f-%.1f %%)" % (name, 100 * r.min(), 100 * r.max()))
            bad += 1
            continue
        f50 = float(np.interp(0.5, [r[below[0]], r[below[0] - 1]],
                              [f[below[0]], f[below[0] - 1]]))
        peak, pk_at = float(r.max()), float(f[int(r.argmax())])
        m = f > max(4.0, pk_at)
        q = qe = None
        if int(m.sum()) > 20:
            ff, rr2 = f[m], r[m]
            for cand in np.arange(1.0, 5.001, 0.01):
                e = float(np.sqrt(np.mean(
                    (1.0 / (1.0 + (ff / f50) ** cand) - rr2) ** 2)))
                if qe is None or e < qe:
                    q, qe = float(cand), e
            ge = float(np.sqrt(np.mean(
                (np.exp(-np.log(2.0) * (ff / f50) ** 2) - rr2) ** 2)))
        else:
            ge = float("nan")
        out[name] = dict(f50=round(f50, 2), q=round(q, 2) if q else None,
                         q_rms=qe, gauss_rms=ge, square=square,
                         res_f=rf, res_r=rr, nf=nf, nr=nr,
                         overshoot=round(peak - 1.0, 4),
                         f_lo=round(float(f.min()), 2),
                         f_hi=round(float(f.max()), 2), label=label,
                         fdrop=fdrop, rdrop=rdrop, frame_err=ferr,
                         resp_err=rerr, page=pageno + 1,
                         norm=((A[:, 0] - fr.x0) / fr.width,
                               (A[:, 1] - fr.y0) / fr.height))
        if verbose:
            print("  %-20s p%d  %d/%d ladder rungs, residual %.2f/%.2f pt, "
                  "square %.2f %%, ladder-vs-frame %.2f %% x / %.2f %% y%s"
                  % (name, pageno + 1, nf, nr, rf, rr, square * 100,
                     100 * (ferr or 0.0), 100 * (rerr or 0.0),
                     ("  dropped " + repr(fdrop)) if fdrop else ""))
            print("      f50 %5.1f c/mm, q %s (rms %s vs Gaussian %.4f), "
                  "traced %.1f-%.1f c/mm, peak %+.3f"
                  % (f50, ("%.2f" % q) if q else "n/a",
                     ("%.4f" % qe) if qe else "n/a", ge, f.min(), f.max(),
                     peak - 1.0))

    # ⚠⚠ THE DUPLICATE CHECK. NotFound.md row 5d already records that APX 100
    # and APX 400 share this panel; anything else that comes back identical is
    # a new instance of the same defect and must not be stored as two readings.
    dups = {s: n for s, n in sigs.items() if len(n) > 1}
    return out, dups, bad


def read_2004(root: Path, verbose=True):
    """The 2004 edition's Sharpness row, for the edition comparison.

    ⚠⚠ CALIBRATED ON ITS FRAME, LIKE THE 1998 SIDE, AND THAT IS THE ONLY WAY
    THE COMPARISON MEANS ANYTHING. Read off their own tick labels the two
    editions are not comparable at all: the 2004 response column is set 0.9 pt
    low as a block. See `_calibrate`.
    """
    doc = pymupdf.open(str(root / PDF_2004))
    out = {}
    for pageno, x0, name in PANELS_2004:
        page = doc[pageno]
        FR = pymupdf.Rect(x0, ROW04_Y0, x0 + 127.2, ROW04_Y1)
        fx, fy = _labels(page, FR)
        if len(fx) < 5 or len(fy) < 4:
            print("  [WARN] 2004 %s: %d/%d labels" % (name, len(fx), len(fy)))
            continue
        fr = _frame(page, FR.x0, FRAME04_Y0)
        if fr is None:
            print("  [WARN] 2004 %s: no plot frame found" % name)
            continue
        _cf, _rf, _nf, _fd, ferr = _logfit(fx, fr.width)
        _cr, _rr, _nr, _rd, rerr = _logfit(fy, fr.height)
        if ferr is not None and ferr > FRAME_TOL:
            print("  [WARN] 2004 %s: frequency ladder misses its frame by "
                  "%.2f %%" % (name, 100 * ferr))
            continue
        X, Y = _calibrate(fx, fy, fr)
        A, _dr = _curve(page, FR)
        if A is None:
            print("  [WARN] 2004 %s: no thick stroked path" % name)
            continue
        f, r = X(A[:, 0]), Y(A[:, 1])
        o = np.argsort(f)
        f, r = f[o], r[o]
        below = np.flatnonzero(r < 0.5)
        if not len(below) or below[0] == 0:
            continue
        f50 = float(np.interp(0.5, [r[below[0]], r[below[0] - 1]],
                              [f[below[0]], f[below[0] - 1]]))
        pk_at = float(f[int(r.argmax())])
        m = f > max(4.0, pk_at)
        q = qe = None
        if int(m.sum()) > 20:
            for cand in np.arange(1.0, 5.001, 0.01):
                e = float(np.sqrt(np.mean(
                    (1.0 / (1.0 + (f[m] / f50) ** cand) - r[m]) ** 2)))
                if qe is None or e < qe:
                    q, qe = float(cand), e
        yy = tuple(np.round(A[::20, 1] - A[0, 1], 4).tolist())
        # ⚠ THE OVERSHOOT IS READ HERE TOO, AND IT HAS TO BE. f50 and the
        # adjacency overshoot are two readings of ONE curve; taking f50 from
        # the 2004 drawing while `mtf.adjacency` still held the 1998 drawing's
        # peak described a curve that exists on neither page.
        out[name] = dict(f50=round(f50, 2), q=round(q, 2) if q else None,
                         q_rms=qe, sig=yy, page=pageno + 1, resp_err=rerr,
                         norm=((A[:, 0] - fr.x0) / fr.width,
                               (A[:, 1] - fr.y0) / fr.height),
                         overshoot=round(float(r.max()) - 1.0, 4))
    # ⚠ THE DUPLICATE CHECK RUNS ON THIS EDITION TOO, and it must: the shared
    # APX drawing did NOT go away in 2004.
    seen = {}
    for n, v in out.items():
        seen.setdefault(v["sig"], []).append(n)
    out["_dups"] = [ns for ns in seen.values() if len(ns) > 1]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()

    print("AGFA «Technical Data PF» 09/1998 -- the eleven Sharpness panels")
    print("  ⚠ 'Transfer factor (%) against Lines (mm)' IS an MTF curve; the "
          "German 'Linien pro mm' = cycles per mm on the ICO's authority "
          "(queue G6, Ingelstam 1961)")
    # ⚠⚠ AND THE SHEET SAYS IT ITSELF, ON p4, WHICH NOTHING ASSERTED UNTIL
    # 2026-09-07. The owner read that page and asked why these curves were not
    # in the database -- they are, ten of twelve since 2026-09-06h/i -- but the
    # primary authority for reading them as MTF had a guard only on the sister
    # Vista sheet, not on the document the panels live in. Fixed here.
    try:
        _p4 = pymupdf.open(str(root / PDF))[3].get_text()
    except Exception:                                         # pragma: no cover
        _p4 = None
    if _p4 and P4_DEFINITION in _p4:
        _i = _p4.find("Sharpness")
        print("  [OK  ] p4 defines the chart, in Agfa's own words:")
        print("         %s" % " ".join(_p4[_i:_i + 230].split()))
    elif _p4 is not None:
        print("  [FAIL] p4 no longer defines these panels as an MTF chart -- "
              "the primary authority for every adoption below is missing")
    print("")
    out, dups, bad = read_all(root, verbose=True)

    print("\n  DUPLICATE-ARTWORK CHECK over all %d panels:" % len(out))
    if not dups:
        print("      no two panels share a path signature")
    for _s, names in dups.items():
        print("      ⚠⚠ SHARED ARTWORK: %s -- one drawing, so at most ONE of "
              "them is a measurement of its own film" % " / ".join(names))

    # ---- THE EDITION COMPARISON -------------------------------------------
    print("\n  THE 2004 EDITION PRINTS THIS PANEL TOO -- comparing, because "
          "the resolving powers were already raised 1998 -> 2004 on this "
          "family (2026-09-01):")
    o4 = read_2004(root, verbose=True)
    d4 = o4.pop("_dups", [])
    for _pair in d4:
        print("      ⚠⚠ SHARED ARTWORK IN 2004 AS WELL: %s. The defect did not "
              "go away between editions, so neither film has a measured MTF in "
              "EITHER of them" % " / ".join(_pair))
    # ⚠⚠ THE QUESTION IS "IS THIS THE SAME DRAWING", AND f50 IS THE WRONG WAY
    # TO ASK IT. f50 is a number derived through two calibrations; the drawing
    # is the thing itself. Both editions' curves are already expressed as
    # fractions of their own frame, so they can be laid on top of each other
    # directly, and that comparison is free of every ladder question that made
    # the first three attempts at this section wrong. `REDRAW_TOL` is 0.002 of
    # frame height -- twice the worst agreement among the panels that ARE the
    # same ink (0.0009), fifty times under the two that are not.
    moved, same = [], []
    for name, v4 in sorted(o4.items()):
        if name not in out:
            continue
        u8, w8 = out[name]["norm"]
        u4, w4 = v4["norm"]
        o8i, o4i = np.argsort(u8), np.argsort(u4)
        g = np.linspace(max(u8.min(), u4.min()), min(u8.max(), u4.max()), 300)
        dv = np.interp(g, u4[o4i], w4[o4i]) - np.interp(g, u8[o8i], w8[o8i])
        geo = float(np.max(np.abs(dv)))
        d = 100.0 * abs(v4["f50"] - out[name]["f50"]) / out[name]["f50"]
        flag = "  <-- REDRAWN" if geo > REDRAW_TOL else ""
        print("      %-20s 1998 f50 %5.1f   2004 %5.1f   %4.1f %%   "
              "artwork max|dy| %.4f of frame%s"
              % (name, out[name]["f50"], v4["f50"], d, geo, flag))
        (moved if geo > REDRAW_TOL else same).append((name, geo))
    print("      %d of %d panels are ONE DRAWING REPRINTED (agree to %.4f of "
          "frame height); %d were redrawn: %s"
          % (len(same), len(same) + len(moved),
             max([g for _n, g in same] or [0.0]), len(moved),
             ", ".join(n for n, _g in moved) or "none"))
    if same:
        print("      ⚠ WHERE THE ARTWORK IS THE SAME INK, THE AGREEMENT IS NOT "
              "CORROBORATION -- it is one measurement printed twice, and the "
              "1998 printing is adopted because its ladder is the sound one.")
    redrawn = {n for n, _g in moved}

    # ---- WHAT IS ACTUALLY ADOPTED -----------------------------------------
    # ⚠⚠ THE 1998 EDITION WINS EXCEPT WHERE THE PANEL WAS REDRAWN, WHICH IS THE
    # OPPOSITE OF WHAT THIS MODULE DECIDED THIS MORNING. The reversal is not a
    # change of policy -- the newer-edition precedent still holds where there
    # IS a newer measurement -- it is that eight of the ten panels turned out
    # not to be newer measurements at all. They are the 1998 ink, rescaled into
    # a frame 0.6 % larger, agreeing to 0.0009 of frame height. Between two
    # printings of one drawing the question is which PAGE reads better, and
    # that is 1998: its response ladder meets its frame to 0.16 pt where the
    # 2004 column is set 0.9 pt low as a block (`_calibrate`).
    # ⚠ THE TWO OPTIMA PANELS WERE GENUINELY REDRAWN -- max|dy| 0.0099 and
    # 0.0108 of frame height, an order of magnitude past the other eight -- and
    # those take the 2004 reading on the ordinary precedent.
    # ⚠ APX 25 AND ULTRA 50 ARE NOT IN THE 2004 EDITION and keep their 1998
    # trace, which is the only one that exists.
    # ⚠⚠ APX 100 AND APX 400 ARE NOW ADOPTED FROM THE DRAWING THEY SHARE,
    # 2026-09-07b, BY OWNER DECISION. This line used to read "ADOPTED FROM
    # NEITHER" and the loop below printed REFUSED for both. The shared-artwork
    # finding is unchanged and still printed above -- one drawing, max |dy|
    # 0.0002 of frame height in both editions, so this cannot be a per-film
    # measurement for both films. What changed is what follows from it. The
    # owner: *"If the vendor datasheet provide MTF please don't simply discard
    # this value even 'Two films cannot share a measured MTF'. We haven't
    # additional better source for check this, so please accept and include
    # these values from vendors datasheet - this is much more better from
    # estimated values!"* The refusal had been substituting a class estimate
    # derived from no document, which is further from the film than a shared
    # vendor curve.
    print("\n  ADOPTED:")
    adopted = {}
    for name in sorted(out):
        _shared = any(name in pr for pr in list(dups.values()) + d4)
        if _shared:
            adopted[name] = (out[name]["f50"], out[name]["q"],
                             out[name]["overshoot"])
            print("      %-20s 1998 edition: f50 %.2f, q %.2f, peak %+.4f  "
                  "⚠ SHARED DRAWING -- adopted 2026-09-07b by owner decision, "
                  "and it cannot be a per-film measurement for both films"
                  % (name, *adopted[name]))
            continue
        if name in redrawn:
            adopted[name] = (o4[name]["f50"], o4[name]["q"],
                             o4[name]["overshoot"])
            print("      %-20s 2004 edition: f50 %.2f, q %.2f, peak %+.4f  "
                  "(panel redrawn between editions)" % (name, *adopted[name]))
        else:
            adopted[name] = (out[name]["f50"], out[name]["q"],
                             out[name]["overshoot"])
            print("      %-20s 1998 edition: f50 %.2f, q %.2f, peak %+.4f  (%s)"
                  % (name, *adopted[name],
                     "one drawing, reprinted unchanged in 2004"
                     if name in o4 else "not reprinted in 2004"))

    # ---- ⚠ THE DATABASE MUST SAY WHAT WAS JUST ADOPTED ---------------------
    try:
        import film_profiles as _fp
        drift = []
        for name, (f50, q, pk) in sorted(adopted.items()):
            m = _fp.get_profile(name).mtf
            if (abs(m.f50_g - f50) > 0.6 or abs(m.mtf_rolloff_q - q) > 0.02
                    or abs(m.adjacency - pk) > 0.002):
                drift.append("%s stores (%.2f, %.2f, %.4f) but the panel reads "
                             "(%.2f, %.2f, %.4f)"
                             % (name, m.f50_g, m.mtf_rolloff_q, m.adjacency,
                                f50, q, pk))
        print("\n  AGAINST THE DATABASE: %s"
              % ("all %d agree" % len(adopted) if not drift
                 else "%d DISAGREE -- %s" % (len(drift), "; ".join(drift))))
        if drift:
            bad += len(drift)
    except Exception as _exc:                                # pragma: no cover
        print("\n  [WARN] could not compare against film_profiles: %s" % _exc)

    for name, want in EXPECTED.items():
        if name not in out:
            continue
        got = (out[name]["f50"], out[name]["q"])
        if abs(got[0] - want[0]) > 0.6 or abs((got[1] or 0) - want[1]) > 0.02:
            print("  [MISMATCH] %s (%s) vs pinned %s" % (name, got, want))
            bad += 1
    for name, want in EXPECTED_PEAK.items():
        got = (adopted.get(name) or (None, None, None))[2]
        if got is not None and abs(got - want) > 0.002:
            print("  [MISMATCH] peak %s (%.4f) vs pinned %.4f"
                  % (name, got, want))
            bad += 1
    for name, want in EXPECTED_2004.items():
        if name not in o4:
            continue
        got = (o4[name]["f50"], o4[name]["q"])
        if abs(got[0] - want[0]) > 0.6 or abs((got[1] or 0) - want[1]) > 0.02:
            print("  [MISMATCH] 2004 %s (%s) vs pinned %s" % (name, got, want))
            bad += 1

    # ---- ⚠⚠ THE CROSS-READER CHECK, AND WHY IT EXISTS ---------------------
    # `agfa_1998_curves.py` reads these same twelve 1998 panels for its own
    # purposes, from its own frames, with its own label windows and its own
    # robust fit. For four days the two modules disagreed by 4 % on every one
    # of them and NEITHER NOTICED, because neither had ever been asked to
    # compare. That is the whole defect: not that a fit was wrong, but that a
    # second reading of the same drawing existed and was never consulted.
    # ⚠ 1.0 % IS THE TOLERANCE AND IT IS NOT SLACK. The two modules calibrate
    # DIFFERENTLY ON PURPOSE -- that one anchors on the printed labels with a
    # robust fit, this one on the drawn frame -- and the two answers differ by
    # up to 0.55 % on these panels. That gap is the measurement's own
    # uncertainty made visible, which is worth more than forcing them to agree.
    # What must never recur is the 4 % that went unseen for four days.
    print("\n  CROSS-READER CHECK against agfa_1998_curves.py (the other "
          "module that reads these same twelve panels):")
    other = None
    try:
        import agfa_1998_curves as C
        other = C.collect(pymupdf.open(str(root / PDF)))["films"]
    except Exception as exc:                                  # pragma: no cover
        print("      [WARN] could not run the sister reader: %s" % exc)
    if other:
        worst, seen = 0.0, 0
        for _printed, rec in sorted(other.items()):
            name = (rec or {}).get("profile")
            v = ((rec or {}).get("sharpness") or {}).get("f50_lines_mm")
            if v is None or name not in out:
                continue
            seen += 1
            d = 100.0 * abs(v - out[name]["f50"]) / v
            worst = max(worst, d)
            if d > 1.0:
                print("      [FAIL] %-20s this module %.2f vs "
                      "agfa_1998_curves %.2f  (%.2f %%)"
                      % (name, out[name]["f50"], v, d))
                bad += 1
        print("      %d panels compared, worst disagreement %.2f %%"
              % (seen, worst))
        if seen < 12:
            print("      [FAIL] only %d panels could be compared; the "
                  "cross-check is not doing its job" % seen)
            bad += 1

    if ns.assert_:
        if bad:
            print("\n[FAIL] the AGFA sharpness panels do not reproduce")
            return 1
        print("\n[OK] %d AGFA «Sharpness» panels re-derived from «Technical "
              "Data PF» 09/1998, each calibrated on ITS OWN printed ladder "
              "rather than on the family's regular layout. ⚠ These are the "
              "first measured MTF curves any AGFA stock in this database has "
              "carried: every one of them held a RED estimated f50 triple "
              "while this document sat in the corpus printing the panel. "
              "⚠ The duplicate-artwork check is re-run every build and "
              "reproduces the known APX 100 / APX 400 shared drawing "
              "(NotFound.md row 5d). ⚠ SINCE 2026-09-07b BOTH ARE ADOPTED "
              "FROM IT by owner decision -- a vendor curve shared between two "
              "films beats a class estimate derived from no document -- and "
              "each profile records that the value cannot be a per-film "
              "measurement for both." % len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
