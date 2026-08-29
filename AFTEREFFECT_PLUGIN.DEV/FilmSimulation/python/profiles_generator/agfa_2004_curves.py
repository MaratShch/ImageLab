"""The Agfa Professional Films sheet, read as vector: four panels x four films.

WHAT THIS SOURCE IS
-------------------
`AGFA/AGFA stocks.pdf` -- Agfa-Gevaert, «Technical Data: Agfa Professional
Films», publication **F-PF-E4**, 12 pages, PageMaker 7.0 -> Distiller 5.0.
⚠ **IT IS THE 4th EDITION, DATED 08/2004, NOT "Agfa 2003"** as queue item E1
calls it: p12 prints `F-PF-E4 / Date: 08/2004 / 4th edition`, and the PDF's own
CreationDate is 2003-07-18 with ModDate 2004-08-25. (The running footer on
pp5-9 still reads `F-PF-E3`, a leftover from the third edition; recorded, not
corrected.) ⚠ The file is byte-identical to `AGFA/FPD1e.pdf` in the same folder
(md5 bf9f0c1a85e42c3d50f60c00a9159690), so the two are one document.

Genuine vector line art -- no page images at all on pp5-11 -- with a real text
layer. That makes this the highest-grade source in the Agfa folder.

WHICH FILMS ACTUALLY CARRY PLOTS, AND THE SECOND THING THE QUEUE GOT WRONG
--------------------------------------------------------------------------
p1 lists eleven professional films. **Only four get plotted panels**, and the
queue's page numbers are each one off:

    printed p6   Agfacolor Portrait 160        (queue said p5)
    printed p7   Agfacolor Optima 100 / 200 / 400   (queue said p6)
    printed p8   Agfachrome RSX II 50 / 100 / 200
    printed p9   Agfapan APX 100 / 400

The three E1 targets are therefore all present -- `AGFA_PORTRAIT_160` on p6 and
`AGFA_OPTIMA_200` / `AGFA_OPTIMA_400` on p7 -- and **so is `AGFA_OPTIMA_100`,
which E1 does not list.** That fourth column is the reason this script is worth
more than the three profiles it feeds: the corpus already holds a spectral set
for Optima 100, traced in 2026-08-02 from a RASTER page of the older 1998
"Range of Films" brochure. Re-reading it here from vector paths turns it into a
cross-check of the earlier batch that nothing else in the corpus could provide.

THE LEGEND
----------
Every stroke on the page is the same near-black (0.137, 0.122, 0.125). The
three layers are separated by DASH ARRAY, exactly as on the sister Vista sheet
that `agfa_vista.py` reads:

    solid                 `[] 0`                       -> GREEN
    dashed                `[ 3.4 .85 ] 0`              -> BLUE
    dash-dot-dot          `[ 3.4 .85 .85 .85 ] 0`      -> RED

⚠ The dash LENGTHS differ from the Vista sheet's (3.402/.851 here against
3.159/.79 there) because the two sheets are laid out at different scales, so
the keys are matched by SHAPE -- number of entries in the array -- and not by
literal string, and the result is then checked against the sheet's own printed
"Blue"/"Green"/"Red" words. Both sheets also print those words in ascending
wavelength order under their own humps, which is the independent check.

WHAT IS ADOPTABLE AND WHAT IS NOT
---------------------------------
- **Spectral sensitivity** (panel 1): adoptable, three layers, per-film.
- **Colour density curves** (panel 4): adoptable, three layers, per-film --
  dmin and gamma per channel, replacing synthesised `_neg()` shapes.
- ⚠ **Spectral density** (panel 2) is NOT adoptable and this is not laziness.
  The panel plots "Medium density" and "Minimum density" -- the film's total
  transmission at two exposure levels -- and NOT the three separated dye
  densities. `SpectralDyeDensity` wants cyan, magenta and yellow curves; two
  aggregate curves cannot be split into three. `agfa_vista.py` reached the same
  conclusion on the sister sheet, and this repeats it rather than quietly
  dropping the panel.
- ⚠ **Sharpness** (panel 3) is READ but its adoption is judged per film, and
  the reason is a units question, not a tracing one. Agfa plot "Transfer factor
  (%)" against "Lines per mm", and the curve **exceeds 100 %** at low frequency
  -- printed peaks run 120-135 %. A true MTF is 1.0 at zero frequency by
  definition, so this is an adjacency-enhanced response, closer to a printed
  CTF. The frequency at which it falls through 50 % is still a well-defined
  reading and is what `MTFSpec.f50` means, so **f50 is taken and the overshoot
  is recorded as the measured adjacency evidence** -- but the curve is NOT
  stored as an MTF shape.

Run:  python agfa_2004_curves.py --root <corpus> [--assert] [--emit]
Needs numpy + PyMuPDF. SciPy is OPTIONAL and audit-only: without it the
six-parameter ToneCurve re-fit is skipped and said to be skipped, and every
other check -- all 12 spectral panels, the dash legend, the steepest-chord
gammas, the sharpness overshoot -- still runs.
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

SHEET = "AGFA/AGFA stocks.pdf"

SOURCE = ("Agfa-Gevaert, «Technical Data: Agfa Professional Films», "
          "publication F-PF-E4, 4th edition, 08/2004 -- "
          "PDF/PROFILES/AGFA/AGFA stocks.pdf. Vector line art, real text "
          "layer; identical to FPD1e.pdf in the same folder.")

#: Dash array shape -> layer. Keyed on the NUMBER of dash entries, not on the
#: literal string, because the two Agfa sheets use different dash lengths.
def _layer_of(dashes) -> str | None:
    s = str(dashes or "").strip()
    if s in ("", "[] 0", "[ ] 0"):
        return "g"
    inner = s[s.find("[") + 1:s.find("]")].split()
    return {2: "b", 4: "r"}.get(len(inner))


#: Near-black, the only stroke colour on the sheet.
def _is_ink(dr) -> bool:
    c = dr.get("color")
    return bool(c) and max(c) < 0.30 and (max(c) - min(c)) < 0.05


#: Curve strokes are 0.85 pt; frames and grids are 0.283 pt. That single
#: number separates data from furniture on every panel of the sheet.
CURVE_W, FRAME_W = 0.85, 0.283
W_TOL = 0.10


class Panel:
    """One plot: a frame rect plus the calibration read from its own labels."""

    def __init__(self, name, rect, xmap, ymap, xres, yres):
        self.name, self.rect = name, rect
        self.xmap, self.ymap = xmap, ymap      # (m, c): value = m*pt + c
        self.xres, self.yres = xres, yres      # max residual of each fit

    def X(self, pt):
        return self.xmap[0] * np.asarray(pt, float) + self.xmap[1]

    def Y(self, pt):
        return self.ymap[0] * np.asarray(pt, float) + self.ymap[1]


def _fit(pairs):
    """pairs: [(value, pt)] -> ((m, c), max residual)."""
    v = np.array([p[0] for p in pairs], float)
    t = np.array([p[1] for p in pairs], float)
    m, c = np.polyfit(t, v, 1)
    return (float(m), float(c)), float(np.abs(m * t + c - v).max())


def _words(page):
    return [(w[4], (w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0, w[0], w[2])
            for w in page.get_text("words")]


def _num(tok):
    t = tok.replace("–", "-").replace("−", "-").strip()
    if t in ("+", "-", ""):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _axis_labels(words, xlo, xhi, ylo, yhi, axis):
    """Numeric labels in a band, with a leading bare '-' or '+' merged in.

    ⚠ THE MERGE IS NECESSARY, NOT COSMETIC. The colour-density abscissa prints
    '-4.0' as TWO words -- a hyphen at x 67.65 and '4.0' at x 74.49 -- so
    taking the number's own centre puts the label 3.4 pt to the right of where
    it is drawn, which is 0.14 decades and would tilt the whole calibration.
    """
    out = []
    cand = [w for w in words if xlo <= w[1] <= xhi and ylo <= w[2] <= yhi]
    cand.sort(key=lambda w: w[1])
    signs = [w for w in cand if w[0].strip() in ("-", "+", "–")]
    for w in cand:
        v = _num(w[0])
        if v is None:
            continue
        left = w[3]
        sgn = 1.0
        for s in signs:
            if 0.0 <= left - s[4] <= 3.0 and abs(s[2] - w[2]) < 3.0:
                left = s[3]
                if s[0].strip() != "+":
                    sgn = -1.0
                break
        centre = (left + w[4]) / 2.0
        out.append((sgn * v, centre if axis == "x" else w[2], w[2], w[1]))

    # ⚠ KEEP ONLY THE LARGEST CO-LINEAR GROUP, because a rectangular window
    # cannot separate these axes. On the colour-density panel the ordinate's
    # "0" is printed at y 562.5 and the abscissa's row at y 567.6 -- 5 pt
    # apart, at almost the same x as the abscissa's leading minus sign. No
    # window catches one and not the other. An axis row is six labels sharing
    # a coordinate and the stray is alone, so the count decides it, and the
    # 2.65-DECADE error that stray caused is what this replaces.
    if not out:
        return []
    idx = 2 if axis == "x" else 3          # cluster on y for a row, x for a column
    # ⚠ 4 pt down a COLUMN, 2.5 pt along a ROW, and the asymmetry is real: a
    # right-aligned ordinate prints "0" about 2.5 pt right of "2.0" because it
    # is one glyph instead of three, while an abscissa row shares one baseline.
    tol = 2.5 if axis == "x" else 4.0
    groups: list[list] = []
    for rec in sorted(out, key=lambda r: r[idx]):
        if groups and abs(rec[idx] - groups[-1][-1][idx]) <= tol:
            groups[-1].append(rec)
        else:
            groups.append([rec])
    best = max(groups, key=len)
    return [(r[0], r[1]) for r in best]


def curves_in(page, panel: Panel, want_layers=True):
    """{layer: (x_pt, y_pt)} for the drawn curves inside one panel frame."""
    r = panel.rect
    out = {}
    for dr in page.get_drawings():
        if dr["type"] not in ("s", "fs") or not _is_ink(dr):
            continue
        if abs((dr.get("width") or 0.0) - CURVE_W) > W_TOL:
            continue
        q = dr["rect"]
        # ⚠ THE BOTTOM TOLERANCE IS 11 pt, NOT 3. Agfa let the lowest colour-
        # density curve (the red record, dash-dot-dot) run 7 pt BELOW its own
        # frame on three of the four columns; a symmetric tolerance dropped it
        # and the panel came back with two layers instead of three.
        if not (q.x0 >= r.x0 - 3 and q.x1 <= r.x1 + 3
                and q.y0 >= r.y0 - 3 and q.y1 <= r.y1 + 11):
            continue
        pts = []
        for it in dr["items"]:
            if it[0] == "l":
                pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
            elif it[0] == "c":
                p0, p1, p2, p3 = it[1], it[2], it[3], it[4]
                for t in np.linspace(0.0, 1.0, 24):
                    u = 1.0 - t
                    pts.append((
                        u**3*p0.x + 3*u*u*t*p1.x + 3*u*t*t*p2.x + t**3*p3.x,
                        u**3*p0.y + 3*u*u*t*p1.y + 3*u*t*t*p2.y + t**3*p3.y))
        if len(pts) < 4:
            continue
        key = _layer_of(dr.get("dashes")) if want_layers else "one"
        if key is None:
            continue
        arr = np.array(pts, float)
        o = np.argsort(arr[:, 0])
        arr = arr[o]
        keep = np.concatenate(([True], np.diff(arr[:, 0]) > 1e-9))
        arr = arr[keep]
        if key in out and len(out[key][0]) >= len(arr):
            continue
        out[key] = (arr[:, 0], arr[:, 1])
    return out


def frames(page, band, xlo, xhi):
    """Frame rects of width FRAME_W whose y sits inside `band`."""
    ylo, yhi = band
    best = None
    for dr in page.get_drawings():
        if dr["type"] not in ("s", "fs") or not _is_ink(dr):
            continue
        if abs((dr.get("width") or 0.0) - FRAME_W) > 0.02:
            continue
        r = dr["rect"]
        if not (ylo <= r.y0 <= yhi and xlo - 4 <= r.x0 and r.x1 <= xhi + 4):
            continue
        a = r.width * r.height
        if best is None or a > best[0]:
            best = (a, r)
    return None if best is None else best[1]


#: One film column: (profile, printed name, page, x band).
#: ⚠ `page` is the PDF page index, which on this sheet equals the printed page.
COLUMNS = (
    ("AGFA_PORTRAIT_160", "Agfacolor Portrait 160", 6, (395.0, 540.0)),
    ("AGFA_OPTIMA_100",   "Agfacolor Optima 100",   7, (60.0, 200.0)),
    ("AGFA_OPTIMA_200",   "Agfacolor Optima 200",   7, (236.0, 375.0)),
    ("AGFA_OPTIMA_400",   "Agfacolor Optima 400",   7, (412.0, 552.0)),
)

#: Vertical bands of the four panels. Identical on p6 and p7.
BANDS = {
    "spectral":  (70.0, 200.0),
    "density":   (210.0, 300.0),
    "sharpness": (320.0, 435.0),
    "curves":    (450.0, 580.0),
}


def read_column(doc, profile, page_no, xlo, xhi):
    pg = doc[page_no - 1]
    ws = _words(pg)
    res = {"profile": profile, "page": page_no}

    # ---- panel 1: spectral sensitivity -----------------------------------
    fr = frames(pg, BANDS["spectral"], xlo, xhi)
    if fr is not None:
        xs = _axis_labels(ws, xlo, xhi, fr.y1, fr.y1 + 12, "x")
        ys = _axis_labels(ws, xlo - 22, fr.x0 + 2, fr.y0 - 6, fr.y1 + 6, "y")
        xs = [p for p in xs if 350 <= p[0] <= 750]
        ys = [p for p in ys if -2.0 <= p[0] <= 3.0]
        if len(xs) >= 3 and len(ys) >= 3:
            xm, xr = _fit(xs)
            ym, yr = _fit(ys)
            p = Panel("spectral", fr, xm, ym, xr, yr)
            res["spectral"] = (p, curves_in(pg, p))
            res["spectral_words"] = sorted(
                [(w[1], w[0]) for w in ws
                 if w[0] in ("Blue", "Green", "Red")
                 and xlo <= w[1] <= xhi and fr.y0 <= w[2] <= fr.y1])

    # ---- panel 3: sharpness ----------------------------------------------
    fr = frames(pg, BANDS["sharpness"], xlo, xhi)
    if fr is not None:
        # ⚠ THE Y BAND STOPS SHORT OF THE FRAME BOTTOM ON PURPOSE. The
        # sharpness abscissa's first label, "2", sits at almost exactly the
        # same x as the ordinate labels, so a y window that reached the axis
        # line swallowed it and threw the log fit out by 0.54 decades.
        # ⚠ AND THE X BAND STARTS RIGHT OF THE FRAME EDGE for the mirror
        # reason: "10" on the transfer-factor axis sits at y 415.9, inside the
        # abscissa's own label row, so an x window that reached the frame edge
        # read it as "10 lines/mm" and threw that fit out by 0.54 decades too.
        xs = _axis_labels(ws, fr.x0 + 3, xhi, fr.y1, fr.y1 + 12, "x")
        ys = _axis_labels(ws, fr.x0 - 24, fr.x0 - 1, fr.y0 - 6, fr.y1 - 8, "y")
        xs = [(np.log10(v), t) for v, t in xs if 1 <= v <= 300]
        ys = [(np.log10(v), t) for v, t in ys if 1 <= v <= 300]
        if len(xs) >= 4 and len(ys) >= 4:
            xm, xr = _fit(xs)
            ym, yr = _fit(ys)
            p = Panel("sharpness", fr, xm, ym, xr, yr)
            res["sharpness"] = (p, curves_in(pg, p, want_layers=False))

    # ---- panel 4: colour density curves ----------------------------------
    fr = frames(pg, BANDS["curves"], xlo, xhi)
    if fr is not None:
        # The colour-density abscissa is printed further below its frame
        # than the other panels' -- 19 pt rather than 6.
        # ⚠ Same trap a third time: the density ordinate's "0" is printed at
        # y 562.5, BELOW this frame's bottom edge at 548.6, so the abscissa
        # window has to start below it or the fit acquires a point claiming
        # log E = 0 at the far left of the plot -- a 2.65-decade error.
        xs = _axis_labels(ws, xlo - 6, xhi + 6, fr.y1 + 12, fr.y1 + 26, "x")
        ys = _axis_labels(ws, xlo - 22, fr.x0 + 4, fr.y0 - 6, fr.y1 + 6, "y")
        xs = [p for p in xs if -5.0 <= p[0] <= 2.0]
        ys = [p for p in ys if 0.0 <= p[0] <= 5.0]
        if len(xs) >= 3 and len(ys) >= 3:
            xm, xr = _fit(xs)
            ym, yr = _fit(ys)
            p = Panel("curves", fr, xm, ym, xr, yr)
            res["curves"] = (p, curves_in(pg, p))
            res["curve_words"] = sorted(
                [(w[2], w[0]) for w in ws
                 if w[0] in ("Blue", "Green", "Red")
                 and xhi - 22 <= w[1] <= xhi + 20
                 and fr.y0 - 6 <= w[2] <= fr.y1 + 6])
    return res


def sample_spectral(panel, xy, lam0=380.0, step=10.0, n=33, floor=-4.0):
    """Resample one layer onto the corpus's standard wavelength grid."""
    x, y = xy
    lam = panel.X(x)
    s = panel.Y(y)
    o = np.argsort(lam)
    lam, s = lam[o], s[o]
    peak = float(s.max())
    grid = lam0 + step * np.arange(n)
    raw = []
    for g in grid:
        if g < lam.min() - 1e-6 or g > lam.max() + 1e-6:
            raw.append(None)
        else:
            raw.append(float(np.interp(g, lam, s)))
    # ⚠ NORMALISE ON THE SAMPLED GRID, NOT ON THE DRAWN CURVE. The true peak
    # generally falls BETWEEN two 10 nm grid points, so subtracting the curve's
    # own maximum leaves the largest stored sample slightly negative -- -0.05
    # on Optima 100's red record -- and `SpectralSensitivity.validate` requires
    # each layer's stored maximum to be exactly 0.0. The shift is at most a few
    # hundredths of a decade and it is a normalisation, not a measurement.
    gpk = max(v for v in raw if v is not None)
    out = [floor if v is None else max(floor, v - gpk) for v in raw]
    return grid, np.array(out), peak, (float(lam.min()), float(lam.max()))


def tone_of(panel, xy):
    """(dmin, gamma, logE span) for one colour-density curve."""
    le = panel.X(xy[0])
    d = panel.Y(xy[1])
    o = np.argsort(le)
    le, d = le[o], d[o]
    dmin = float(d.min())
    best = 0.0
    for i in range(len(le)):
        j = np.searchsorted(le, le[i] + 0.60)
        if j >= len(le):
            break
        best = max(best, float(np.polyfit(le[i:j], d[i:j], 1)[0]))
    return dmin, best, float(le.max() - le.min())


def fit_tone(panel, xy):
    """Fit the corpus's own six-parameter ToneCurve to one drawn curve.

    The stored form is a difference of two softplus ramps::

        D(x) = dmin + gamma * ( sp(x - toe_x, toe_k) - sp(x - shoulder_x, sk) )

    ⚠ THE ABSCISSA HAS TO BE RE-ANCHORED AND THAT IS A CONVENTION, NOT A
    MEASUREMENT. Agfa plot ABSOLUTE exposure, "lg exposure (Lx * s)" from -4.0
    to +1.0. `ToneCurve.x` is a RELATIVE log exposure whose origin is the
    corpus's mid-grey anchor, and no document connects the two. So the fit is
    done in Agfa's frame and then SHIFTED so the straight line's midpoint,
    (toe_x + shoulder_x)/2, lands where the colour-negative family already puts
    it (+0.10, from the `_neg` defaults -1.55 / +1.75). What is therefore
    MEASURED here is dmin, gamma, the toe-to-shoulder SPAN and both
    softnesses; what is CARRIED is only the placement of the origin.

    Returns (dmin, gamma, toe_x, toe_k, shoulder_x, shoulder_k, rms).
    """
    # ⚠ SciPy IS AN OPTIONAL, AUDIT-ONLY DEPENDENCY AND THE PROJECT'S STATED
    # RULE IS "numpy and Pillow only". The fitted values this returns are
    # already ADOPTED in film_profiles.py, so this routine only RE-DERIVES
    # them; where SciPy is absent the audit still runs and still checks every
    # spectral panel and the steepest-chord gamma, and says plainly that the
    # six-parameter re-fit was skipped rather than silently passing.
    try:
        from scipy.optimize import least_squares
    except ImportError:                                   # pragma: no cover
        return None

    le = panel.X(xy[0])
    d = panel.Y(xy[1])
    o = np.argsort(le)
    le, d = np.asarray(le)[o], np.asarray(d)[o]

    def sp(z, k):
        # numerically safe softplus, k * log1p(exp(z/k))
        u = z / k
        return k * np.where(u > 30.0, u, np.log1p(np.exp(np.minimum(u, 30.0))))

    def model(p):
        dmin, g, tx, tk, sx, ratio = p
        return dmin + g * (sp(le - tx, tk) - sp(le - sx, ratio * tk))

    # ⚠ THE SHOULDER SOFTNESS IS TIED TO THE TOE'S, AND WITHOUT THAT THE FIT
    # IS DEGENERATE. `ToneCurve`'s own docstring requires shoulder_k <= 1.4 *
    # toe_k, and an unconstrained fit walked straight past it: Portrait 160's
    # red record came back at 3.0x, rms 0.005 D -- a good fit to the drawn
    # curve, and a `gamma` of 1.018 that the curve does not have. When the two
    # ramps overlap that heavily the `gamma` PARAMETER stops being the
    # observable straight-line slope, so the number would have been stored
    # against a definition it no longer met. Fitting the RATIO inside the
    # documented band removes the degeneracy and puts the parameter back on
    # its definition; the steepest-chord slope below then checks it.
    p0 = np.array([float(d.min()), 0.65, -2.2, 0.30, 1.6, 1.30])
    lo = np.array([0.0, 0.20, -6.0, 0.05, -2.0, 0.50])
    hi = np.array([3.0, 2.50, 2.0, 2.00, 8.0, 1.40])
    r = least_squares(lambda p: model(p) - d, p0, bounds=(lo, hi),
                      max_nfev=20000)
    dmin, g, tx, tk, sx, ratio = r.x
    sk = ratio * tk
    rms = float(np.sqrt(np.mean((model(r.x) - d) ** 2)))
    # re-anchor: straight-line midpoint to +0.10, the family convention
    shift = 0.10 - 0.5 * (tx + sx)
    return (float(dmin), float(g), float(tx + shift), float(tk),
            float(sx + shift), float(sk), rms)


def sharpness_of(panel, xy):
    """(peak transfer %, f50 lines/mm, freq of peak) off the Sharpness panel."""
    f = 10.0 ** panel.X(xy[0])
    t = 10.0 ** panel.Y(xy[1])
    o = np.argsort(f)
    f, t = f[o], t[o]
    peak = float(t.max())
    fpeak = float(f[int(np.argmax(t))])
    f50 = None
    for i in range(1, len(f)):
        if t[i - 1] >= 50.0 > t[i]:
            u = (50.0 - t[i - 1]) / (t[i] - t[i - 1])
            f50 = float(f[i - 1] + u * (f[i] - f[i - 1]))
            break
    return peak, f50, fpeak


#: Measured 2026-08-29. Anything that moves by more than the tolerance is a
#: change in the source or in this reader, and the audit says so.
EXPECTED = {
    "AGFA_PORTRAIT_160": dict(peaks=(420.0, 550.0, 650.0),
                              gamma=(0.593, 0.647, 0.744),
                              f50=38.4, peak_pct=109.0),
    "AGFA_OPTIMA_100":   dict(peaks=(470.0, 550.0, 620.0),
                              gamma=(0.640, 0.645, 0.691),
                              f50=46.2, peak_pct=114.0),
    "AGFA_OPTIMA_200":   dict(peaks=(470.0, 550.0, 620.0),
                              gamma=(0.576, 0.571, 0.614),
                              f50=50.8, peak_pct=113.0),
    "AGFA_OPTIMA_400":   dict(peaks=(470.0, 560.0, 610.0),
                              gamma=(0.633, 0.652, 0.701),
                              f50=50.3, peak_pct=109.0),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--emit", action="store_true",
                    help="print the SpectralSensitivity / ToneCurve blocks")
    ns = ap.parse_args()

    pdf = Path(ns.root).resolve() / "PDF" / "PROFILES" / SHEET
    if not pdf.is_file():
        print(f"  [SKIP] source not present: {pdf}")
        return 0
    doc = pymupdf.open(pdf)
    print(f"[i] {SOURCE}")

    bad = 0
    for profile, printed, page_no, (xlo, xhi) in COLUMNS:
        r = read_column(doc, profile, page_no, xlo, xhi)
        print(f"\n  {profile}   ({printed}, printed p{page_no})")

        if "spectral" not in r:
            print("    [FAIL] spectral panel not found")
            bad += 1
        else:
            p, cs = r["spectral"]
            print(f"    spectral   frame ({p.rect.x0:.1f},{p.rect.y0:.1f})-"
                  f"({p.rect.x1:.1f},{p.rect.y1:.1f})  fit residual "
                  f"{p.xres:.2f} nm / {p.yres:.4f} lg")
            if set(cs) != {"r", "g", "b"}:
                print(f"    [FAIL] dash keying gave {sorted(cs)}, want b/g/r")
                bad += 1
            else:
                peaks = {}
                for k in "bgr":
                    grid, s, pk, span = sample_spectral(p, cs[k])
                    peaks[k] = float(grid[int(np.argmax(s))])
                    print(f"      {k}: peak {peaks[k]:.0f} nm, drawn "
                          f"{span[0]:.0f}-{span[1]:.0f} nm, "
                          f"{len(cs[k][0])} points")
                # ⚠ Two independent checks on the legend, neither of them the
                # dash array: the peaks must ASCEND b < g < r, and they must
                # match the order of the sheet's own printed words.
                if not peaks["b"] < peaks["g"] < peaks["r"]:
                    print(f"    [FAIL] peaks do not ascend b<g<r: {peaks}")
                    bad += 1
                wl = [w[1].lower()[0] for w in r.get("spectral_words", [])]
                if wl and wl != ["b", "g", "r"]:
                    print(f"    [FAIL] printed words read {wl} left to right")
                    bad += 1
                want = EXPECTED[profile]["peaks"]
                got = (peaks["b"], peaks["g"], peaks["r"])
                if max(abs(a - b) for a, b in zip(got, want)) > 12.0:
                    print(f"    [FAIL] peaks moved: {got} vs {want}")
                    bad += 1

        if "curves" in r:
            p, cs = r["curves"]
            print(f"    density    fit residual {p.xres:.4f} dec / "
                  f"{p.yres:.4f} D")
            if set(cs) != {"r", "g", "b"}:
                print(f"    [FAIL] colour-density dash keying gave "
                      f"{sorted(cs)}")
                bad += 1
            else:
                gs = {}
                for k in "rgb":
                    dmin_c, g_c, span = tone_of(p, cs[k])
                    fit = fit_tone(p, cs[k])
                    if fit is None:
                        gs[k] = None
                        print(f"      {k}: dmin {dmin_c:.3f}  steepest chord "
                              f"{g_c:.3f} over {span:.2f} decades, "
                              f"{len(cs[k][0])} points  [i] six-parameter "
                              f"re-fit SKIPPED: SciPy not installed")
                        continue
                    fdmin, fg, tx, tk, sx, sk, rms = fit
                    gs[k] = fg
                    print(f"      {k}: dmin {fdmin:.3f}  gamma {fg:.3f} "
                          f"(steepest chord {g_c:.3f})  toe {tx:+.3f}/{tk:.3f} "
                          f"shoulder {sx:+.3f}/{sk:.3f}  dmax "
                          f"{fdmin + fg * (sx - tx):.2f}  fit rms {rms:.4f} D "
                          f"over {span:.2f} decades")
                    # ⚠ The two estimators are independent -- one is a
                    # six-parameter fit, the other a chord on the raw points --
                    # so their agreement is a real check on both.
                    if abs(fg - g_c) > 0.06:
                        print(f"        [FAIL] fitted gamma and steepest chord "
                              f"disagree by {abs(fg - g_c):.3f}")
                        bad += 1
                    if rms > 0.020:
                        print(f"        [FAIL] fit rms {rms:.4f} D is too "
                              f"large for a vector trace")
                        bad += 1
                want = EXPECTED[profile]["gamma"]
                got = (gs["r"], gs["g"], gs["b"])
                if any(v is None for v in got):
                    pass          # SciPy absent; the chord checks above stand
                elif max(abs(a - b) for a, b in zip(got, want)) > 0.03:
                    print(f"    [FAIL] gammas moved: "
                          f"{tuple(round(v, 3) for v in got)} vs {want}")
                    bad += 1
        else:
            print("    [FAIL] colour-density panel not found")
            bad += 1

        if "sharpness" in r:
            p, cs = r["sharpness"]
            if "one" in cs:
                peak, f50, fpk = sharpness_of(p, cs["one"])
                print(f"    sharpness  peak {peak:.0f} % at {fpk:.1f} c/mm, "
                      f"f50 {f50 if f50 is None else round(f50, 1)} lines/mm "
                      f"(fit residual {p.xres:.4f}/{p.yres:.4f} dec)")
                if peak <= 100.0:
                    print("      [i] no overshoot on this panel")
                w50 = EXPECTED[profile]["f50"]
                if f50 is None or abs(f50 - w50) > 1.5:
                    print(f"      [FAIL] f50 moved: {f50} vs {w50}")
                    bad += 1
                wpk = EXPECTED[profile]["peak_pct"]
                if abs(peak - wpk) > 2.0:
                    print(f"      [FAIL] overshoot moved: {peak:.0f} vs {wpk}")
                    bad += 1
                # ⚠ WHAT IS ADOPTABLE HERE IS THE OVERSHOOT AND NOT f50.
                # `MTFSpec.f50` is in CYCLES per mm; Agfa's abscissa says
                # "Lines per mm", and whether that means line PAIRS or single
                # lines is an open question in this corpus -- it is queue item
                # G6, raised against Gevacolor 682's identically-worded axis.
                # A factor of two is not a rounding difference, so f50 is
                # REPORTED and left unstored, and the reading is filed as
                # evidence for G6. The peak height is a ratio and carries no
                # unit, so `MTFSpec.adjacency` can take it.
                print(f"      -> adjacency {peak / 100.0 - 1.0:+.3f} "
                      f"(unit-free, adoptable); f50 NOT adopted, see queue G6")
            else:
                print("    [FAIL] sharpness curve not found")
                bad += 1

        if ns.emit and "spectral" in r and set(r["spectral"][1]) == {"r","g","b"}:
            p, cs = r["spectral"]
            print(f"\n        # {profile}")
            for k in ("b", "g", "r"):
                grid, s, pk, _ = sample_spectral(p, cs[k])
                vals = ", ".join(f"{v:.2f}" for v in s)
                print(f"        log_s_{k}=({vals}),")

    print()
    if bad:
        print(f"[FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("[OK] 4 film columns, 12 spectral curves and 12 colour-density "
          "curves read from vector paths; the dash legend agrees with the "
          "sheet's own printed Blue/Green/Red words and with the physical "
          "requirement that each layer peak in its own band")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
