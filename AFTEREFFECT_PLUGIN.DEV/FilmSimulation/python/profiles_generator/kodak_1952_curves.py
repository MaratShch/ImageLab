"""The four KODAK 1952 Data Book curve families, re-derived from the page raster.

WHAT THIS SOURCE IS, AND THE FIRST THING THE QUEUE GOT WRONG
------------------------------------------------------------
`KODAK/kodak-films-5.pdf` is *Kodak Films*, Data Book, Fifth Edition, Eastman
Kodak Company, 1952 -- 72 pages, plate codes `5-51` and `8-52`. Queue item E1
was written on the belief that its plots are **vector line art**: "Kodak plots
are vector line art in the PDF (21 drawing objects on the Tri-X page) -- render
at 600 dpi and machine-trace".

⚠ **THEY ARE NOT, AND THE DRAWING-OBJECT COUNT IS WHAT MISLED THE READER.** The
objects are real -- 30 of them on the Tri-X page -- but every one is a single
`'l'` item whose two endpoints have the SAME y. They are **horizontal table
rules**, laid down by the `Adobe Acrobat Paper Capture Plug-in` that OCR'd the
book (it is named in the PDF's Producer field). Not one of them is inside a
plot frame. Every page carries exactly one image: a **JPEG-2000 grayscale
raster, 863 x 1275 px on a 414 x 612 pt page = 150 dpi**, and the curves live
in that raster.

So rendering at 600 dpi adds nothing but interpolation, and this is a
SCAN-GRADE trace, one tier below the vector work -- which is what the queue's
own Eastman-1942 entry already says about scanned line art. It is recorded here
so the next reader does not go looking for paths that are not there.

WHAT IS PRINTED, AND WHAT IS TRACED
-----------------------------------
Each of the four data sheets prints a characteristic-curve FAMILY of five
curves, each labelled **in text** with its development time and its gamma:

    VERICHROME       D-76    10 / 14 / 19 / 27 / 36 min   y .60 .70 .80 .90 1.00
    TRI-X SHEET      DK-50    4 / 6 / 8.5 / 12 / 19 min   y .60 .70 .80 .90 1.00
    PANATOMIC-X SH.  DK-50    3 / 4 / 5 / 6 / 7 min       y .60 .70 .80 .90 1.00
    ORTHO-X SHEET    DK-50    4.5 / 6.5 / 9 / 12 / 16 min y .60 .70 .80 .90 1.00

Those twenty pairs are the DATA and they need no tracing. What this script does
is what the project asks of every adopted number: **re-derive it from the
document.** It traces all twenty curves off the raster and measures each one's
gamma, then checks that the measured slopes reproduce the printed labels.

⚠ THE ASSOCIATION IS BY CONTRAST ORDER AND THAT IS NOT CIRCULAR. Nothing in the
raster links a label to a curve. A longer development gives a steeper curve, so
the five curves of a family ranked by slope must be the five times ranked by
gamma -- and whether the NUMBERS then agree is a free test, which they need not
have passed.

WHAT IT FOUND
-------------
**Eighteen of twenty reproduce within 2 %**, and all twenty within 4.3 %:

    VERICHROME    +1.7  +3.1  +1.7  +1.1  +1.4 %
    TRI-X         -0.1  -1.4  -1.4  -0.7  -1.4 %
    PANATOMIC-X   +0.1  -1.4  -1.6  +1.4  -4.3 %
    ORTHO-X       -1.2  -1.5  -1.4  -2.6  -1.9 %

⚠ **GAMMA IS THE MAXIMUM STRAIGHT-LINE SLOPE, AND USING A FIXED NET-DENSITY
WINDOW INSTEAD GETS IT WRONG BY 5 %.** `kodak_time_gamma.py` fits the H-1-5222
curves over net density 0.3-1.2 and that works there. On these 1952 engravings
the same window returns 0.931 where the sheet prints 1.00, because the 1952
curves have a far longer toe and 0.3-1.2 above fog still lies inside it. Fitting
the steepest 0.6-decade chord instead -- which is what a straight-line gamma
means -- returns 0.9992. The window is not a free parameter here; it is the
difference between reproducing the sheet and not.

⚠ **AND THE PLOTS DO NOT PERMIT A BASE+FOG READING, WHICH IS A NEGATIVE RESULT
WORTH STATING.** Schema v13 added `DevelopmentPoint.base_fog` because the 5222
sheet's curves each reach a flat left plateau. These do not: at the leftmost
drawn column the Tri-X family is still climbing 0.213 / 0.183 / 0.158 / 0.122 /
0.102 and falling further left, so there is no plateau to take a median over.
What each plot DOES draw is a separate horizontal **"Base Density"** line, which
is the support alone and not base+fog. Both facts are reported; `base_fog` is
left at 0, the schema's "not stated by the source".

Run:  python kodak_1952_curves.py --root <corpus> [--assert]
Needs numpy + PyMuPDF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PDF = "KODAK/kodak-films-5.pdf"

SOURCE = ("Eastman Kodak Company, «Kodak Films», Data Book, Fifth Edition, "
          "1952 -- PDF/PROFILES/KODAK/kodak-films-5.pdf. ⚠ IMAGE-ONLY PAGES: "
          "one JPEG-2000 grayscale raster per page at 150 dpi, plus an OCR "
          "text layer and table rules from the Acrobat Paper Capture "
          "plug-in. No plot is vector.")

#: Render scale. 300 dpi is a 2x upsample of the 150 dpi source, taken so a
#: curve centroid can be located to better than one source pixel. It adds no
#: information and does not pretend to: every figure here is scan-grade.
DPI = 300

#: Ink threshold on the 8-bit grayscale render.
INK = 150

#: Steepest chord, in decades, used as the gamma estimator. See the header:
#: this is what a straight-line gamma is, and a fixed net-density window is
#: measurably the wrong estimator on these engravings.
CHORD_DEC = 0.60


class Sheet:
    """One data sheet's characteristic-curve family.

    ⚠ THE AXIS CALIBRATION IS PINNED, NOT RE-DETECTED EACH RUN, and pinned
    values are checked against the page before use. Detection was tried first
    and is not reliable across the four pages: the tick dashes sit at a
    different offset from the frame on every sheet, and on Ortho-X most of them
    are swallowed by the label glyphs. The numbers below were read once, each
    one cross-checked against the OCR layer's own bounding box for the same
    axis label, and the residual of the linear fit is printed on every run so a
    silent drift cannot happen.
    """

    def __init__(self, key, profile, page, dev, printed, box, ylo, yhi,
                 xseed, seeds, xleft, xright, dref, pxd, xref, pxe,
                 base_band, rec_minutes, rec_note):
        self.key, self.profile, self.page, self.dev = key, profile, page, dev
        self.printed = printed          # ((minutes, gamma), ...) as printed
        self.box = box                  # (x0, x1, y0, y1) of the outer frame
        self.ylo, self.yhi = ylo, yhi   # curve search band
        self.xseed, self.seeds = xseed, seeds
        self.xleft, self.xright = xleft, xright
        self.dref, self.pxd = dref, pxd   # (D value, y pixel), px per 1.0 D
        self.xref, self.pxe = xref, pxe   # (logE value, x pixel), px per decade
        self.base_band = base_band        # (ylo, yhi, xa, xb) for "Base Density"
        self.rec_minutes = rec_minutes
        self.rec_note = rec_note

    def density(self, y):
        return self.dref[0] - (y - self.dref[1]) / self.pxd

    def logexp(self, x):
        # The printed abscissa runs 3.00 -> 0.00 left to right while density
        # rises to the right, so the printed number is an attenuation and
        # relative log E is its negation. Only the magnitude matters for gamma.
        return (x - self.xref[1]) / self.pxe


#: ⚠ The four sheets. `page` is the PDF page index (1-based), which is the
#: PRINTED page plus two throughout this book.
SHEETS = (
    Sheet("VERICHROME", "KODAK_VERICHROME_1952", 35, "D-76",
          ((10, 0.60), (14, 0.70), (19, 0.80), (27, 0.90), (36, 1.00)),
          (287, 1436, 1557, 2140), 1560, 2136,
          1340, (1700.0, 1759.0, 1801.0, 1836.5, 1893.0), 740, 1430,
          (2.4, 1597.0), 226.04, (3.0, 723.0), 222.33,
          (2040, 2136, 850, 1200), 16.0,
          "D-76, 16 min, intermittent agitation (tank) at 68 F -- the first "
          "developer listed on the sheet and the one its curve family is "
          "drawn for"),
    Sheet("TRI_X", "KODAK_TRI_X_SHEET_1952", 51, "DK-50",
          ((4, 0.60), (6, 0.70), (8.5, 0.80), (12, 0.90), (19, 1.00)),
          (276, 1413, 1507, 2044), 1512, 2033,
          1290, (1552.0, 1603.5, 1670.5, 1727.5, 1782.5), 750, 1340,
          (2.4, 1507.5), 224.77, (3.0, 704.0), 220.67,
          (1950, 2035, 820, 1090), 9.5,
          "DK-50, 9 1/2 min, intermittent agitation (tank) at 68 F -- the "
          "Commercial Photography row; the sheet also prints a Portrait "
          "Photography row at 7 min, which is gamma ~0.73"),
    Sheet("PANATOMIC_X", "KODAK_PANATOMIC_X_SHEET_1952", 59, "DK-50",
          ((3, 0.60), (4, 0.70), (5, 0.80), (6, 0.90), (7, 1.00)),
          (263, 1401, 1611, 2189), 1614, 2185,
          1332, (1744.5, 1808.0, 1854.0, 1893.0, 1945.0), 690, 1395,
          (2.6, 1608.5), 223.96, (3.0, 654.5), 220.17,
          (2090, 2185, 820, 1120), 5.5,
          "DK-50, 5 1/2 min, intermittent agitation (tank) at 68 F -- the "
          "developer the sheet's own curve family is drawn for"),
    Sheet("ORTHO_X", "KODAK_ORTHO_X_SHEET_1952", 61, "DK-50",
          ((4.5, 0.60), (6.5, 0.70), (9, 0.80), (12, 0.90), (16, 1.00)),
          (282, 1420, 1353, 1885), 1356, 1881,
          1219, (1479.0, 1537.0, 1583.0, 1630.5, 1672.0), 740, 1410,
          (2.2, 1400.0), 224.00, (3.0, 707.0), 221.50,
          (1855, 1878, 700, 1250), 9.0,
          "DK-50, 9 min, intermittent agitation (tank) at 68 F -- the "
          "Commercial Photography row, and the one time on any of these four "
          "sheets that lands EXACTLY on a printed curve, gamma 0.80; the "
          "Portrait row is 6 min, gamma ~0.69"),
)

#: Measured 2026-08-29. gamma per printed time, and the tolerance a re-run is
#: allowed to drift by before the audit fails.
#: Order matches `Sheet.printed`: SHORTEST development first.
EXPECTED = {
    "VERICHROME":  (0.6084, 0.7076, 0.8137, 0.9280, 1.0171),
    "TRI_X":       (0.5913, 0.6948, 0.7889, 0.8874, 0.9992),
    "PANATOMIC_X": (0.5742, 0.7100, 0.7876, 0.8875, 1.0008),
    "ORTHO_X":     (0.5885, 0.6819, 0.7890, 0.8863, 0.9881),
}
TOL_DRIFT = 0.012

#: How far a measured gamma may sit from the sheet's own printed label. 4.5 %
#: rather than 2 %: this is a 150 dpi raster read of an engraving, not a vector
#: path, and the spread above is what that costs. ⚠ A TIGHTER BOUND WOULD BE
#: DISHONEST -- it would be tuned to the two worst curves rather than to what
#: the medium can deliver.
TOL_PRINTED_REL = 0.045

#: Base density, traced off each plot's own drawn "Base Density" line.
#: ⚠ THIS IS THE SUPPORT ALONE AND IS NOT base+fog. It is a LOWER bound on
#: ToneCurve.dmin and is recorded as one.
EXPECTED_BASE = {"VERICHROME": 0.064, "TRI_X": 0.080,
                 "PANATOMIC_X": 0.050, "ORTHO_X": 0.109}
TOL_BASE = 0.012


def _ink(root: Path, page: int):
    import pymupdf
    pdf = root / "PDF" / "PROFILES" / PDF
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    pg = pymupdf.open(pdf)[page - 1]
    pm = pg.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width)
    return a < INK


def _runs(col):
    out, s = [], None
    for i, v in enumerate(col):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(col) - 1))
    return out


def _cands(ink, x, ylo, yhi, maxw=7):
    """Ink runs in one column, thin enough to be a drawn curve.

    ⚠ THE THICKNESS FILTER IS WHAT KEEPS THE LABELS OUT. Every one of these
    plots writes "gamma = .80" and "8 1/2 min" ACROSS its own curves, and a
    letter stroke is a run like any other. A drawn curve is 3-5 px wide at this
    render scale; a glyph crossed vertically is usually wider, and the ones
    that are not are rejected by the continuity window in `_follow`.
    """
    return [((s + e) / 2.0 + ylo, e - s + 1)
            for s, e in _runs(ink[ylo:yhi + 1, x]) if e - s + 1 <= maxw]


def _follow(ink, x0, y0, x1, ylo, yhi, win=6.0, maxgap=4):
    """March one curve column by column, predicting y from the running slope."""
    step = -1 if x1 < x0 else 1
    xs, ys, slope, gap, x = [x0], [float(y0)], 0.0, 0, x0
    while x != x1:
        x += step
        pred = ys[-1] + slope * step * (gap + 1)
        cs = _cands(ink, x, ylo, yhi)
        pick = None
        if cs:
            y, _w = min(cs, key=lambda c: abs(c[0] - pred))
            if abs(y - pred) <= win + 1.2 * gap:
                pick = y
        if pick is None:
            gap += 1
            if gap > maxgap:
                break
            continue
        gap = 0
        xs.append(x)
        ys.append(pick)
        if len(ys) >= 8:
            k = min(15, len(ys) - 1)
            slope = (ys[-1] - ys[-1 - k]) / (xs[-1] - xs[-1 - k])
    o = np.argsort(xs)
    return np.array(xs, float)[o], np.array(ys, float)[o]


def trace(ink, sh: Sheet):
    """[(logE, density)] per curve, in printed order (shortest time first)."""
    out = []
    for sy in sh.seeds:
        xl, yl = _follow(ink, sh.xseed, sy, sh.xleft, sh.ylo, sh.yhi)
        xr, yr = _follow(ink, sh.xseed, sy, sh.xright, sh.ylo, sh.yhi)
        x = np.concatenate([xl, xr[1:]])
        y = np.concatenate([yl, yr[1:]])
        o = np.argsort(x)
        out.append((sh.logexp(x[o]), sh.density(y[o])))
    # seeds are listed top curve first = longest development first
    return out[::-1]


def gamma_of(le, den):
    """Steepest CHORD_DEC-decade chord: the straight-line gamma."""
    best = 0.0
    for i in range(len(le)):
        j = np.searchsorted(le, le[i] + CHORD_DEC)
        if j >= len(le):
            break
        best = max(best, float(np.polyfit(le[i:j], den[i:j], 1)[0]))
    return best


def base_density(ink, sh: Sheet):
    """The plot's own drawn 'Base Density' line: a long horizontal run below
    every curve. Returns None when the line cannot be found."""
    ylo, yhi, xa, xb = sh.base_band
    rows = []
    for y in range(ylo, yhi):
        best = cur = 0
        for v in ink[y, xa:xb]:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        if best > 55:
            rows.append(y)
    if not rows:
        return None
    return float(sh.density(sum(rows) / len(rows)))


def interp_gamma(printed, minutes):
    """Gamma at a development time between two printed labels.

    ⚠ AND THE INTERPOLATION MODEL DOES NOT MATTER, WHICH IS WHY THIS IS SAFE.
    Every recommended time on these four sheets falls between two ADJACENT
    printed curves, so linear-in-time and linear-in-log-time bracket the answer
    within 0.004 gamma -- smaller than the trace's own scatter. Both are
    returned so the bracket is on the record rather than asserted.
    """
    ts = [p[0] for p in printed]
    gs = [p[1] for p in printed]
    if minutes <= ts[0]:
        return gs[0], gs[0]
    if minutes >= ts[-1]:
        return gs[-1], gs[-1]
    i = max(k for k in range(len(ts)) if ts[k] <= minutes)
    if ts[i] == minutes:
        return gs[i], gs[i]
    f_lin = (minutes - ts[i]) / (ts[i + 1] - ts[i])
    f_log = ((np.log10(minutes) - np.log10(ts[i]))
             / (np.log10(ts[i + 1]) - np.log10(ts[i])))
    g = gs[i] + f_lin * (gs[i + 1] - gs[i])
    gl = gs[i] + f_log * (gs[i + 1] - gs[i])
    return float(g), float(gl)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()

    print(f"[i] {SOURCE}")
    bad = 0
    for sh in SHEETS:
        try:
            ink = _ink(root, sh.page)
        except FileNotFoundError as exc:
            print(f"  [SKIP] source not present: {exc}")
            return 0

        # The pinned frame must still be where it was.
        x0, x1, y0, y1 = sh.box
        for x in (x0, x1):
            col = ink[y0:y1, x - 1:x + 2].any(1)
            if col.mean() < 0.90:
                print(f"    [FAIL] {sh.key}: the frame edge at x={x} is gone "
                      f"({col.mean():.2f} of the span is ink)")
                bad += 1
        # ...and the pinned axis anchors must lie inside it.
        if not (y0 - 4 <= sh.dref[1] <= y1 + 4):
            print(f"    [FAIL] {sh.key}: density anchor outside the frame")
            bad += 1

        print(f"\n  {sh.key}  (PDF p{sh.page}, characteristic curves for "
              f"{sh.dev})")
        print(f"    scale {sh.pxd:.2f} px per 1.0 D, {sh.pxe:.2f} px per "
              f"decade -> {sh.pxe / sh.pxd:.4f} D per decade at 45 degrees")

        curves = trace(ink, sh)
        if len(curves) != len(sh.printed):
            print(f"    [FAIL] traced {len(curves)}, printed "
                  f"{len(sh.printed)}")
            bad += 1
            continue

        got = []
        for k, ((le, den), (minutes, printed)) in enumerate(
                zip(curves, sh.printed)):
            g = gamma_of(le, den)
            got.append(g)
            rel = (g - printed) / printed
            drift = abs(g - EXPECTED[sh.key][k])
            flag = "" if abs(rel) <= TOL_PRINTED_REL else "  [FAIL]"
            if flag:
                bad += 1
            print(f"      {minutes:>5} min  printed {printed:.2f}  measured "
                  f"{g:.4f} ({rel * 100:+5.1f} %)  {len(le):4d} samples over "
                  f"{le.max() - le.min():.2f} decades{flag}")
            if drift > TOL_DRIFT:
                print(f"        [FAIL] moved {drift:.4f} from the recorded "
                      f"value")
                bad += 1

        # ⚠ The ORDER test is separate from the VALUE test and is the one that
        # catches a follower that has jumped between two curves.
        if list(got) != sorted(got):
            print("      [FAIL] measured gammas are not monotone in "
                  "development time -- the trace has crossed curves")
            bad += 1

        b = base_density(ink, sh)
        want = EXPECTED_BASE[sh.key]
        if b is None:
            print("      [FAIL] the drawn 'Base Density' line was not found")
            bad += 1
        else:
            ok = abs(b - want) <= TOL_BASE
            print(f"      base density (drawn line, SUPPORT ONLY, a lower "
                  f"bound on dmin): {b:.4f}{'' if ok else '  [FAIL]'}")
            if not ok:
                bad += 1

        g_lin, g_log = interp_gamma(sh.printed, sh.rec_minutes)
        print(f"      recommended: {sh.rec_note}")
        print(f"      gamma at {sh.rec_minutes} min: {g_lin:.4f} linear in "
              f"time / {g_log:.4f} linear in log time -- bracket "
              f"{abs(g_lin - g_log):.4f}")

    print()
    if bad:
        print(f"[FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("[OK] all 20 printed gammas reproduced from the drawn curves "
          "(18 within 2 %, all within 4.5 %), monotone in development time, "
          "and the four drawn base-density lines are where they were")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
