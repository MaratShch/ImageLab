"""Fuji NEOPAN SS (135), AF3-411E(N) -- the datasheet that made the profile
possible, and the four measurements it is deliberately NOT joined to.

Queue item N1, 2026-09-02.  FUJIFILM DATA SHEET "NEOPAN SS (135)", Ref. No.
AF3-411E(N) (EIGI-99.3-HB4-8), four pages, supplied by the owner:

    PDF/PROFILES/FUJI/SS35.pdf

WHY THIS DOCUMENT CLOSED A QUESTION THREE OTHERS COULD NOT
-------------------------------------------------------------
`EMULSION_KNOWLEDGE_BASE.md` §23k.8 recorded, on 2026-09-02, that FUJI NEOPAN
could not be profiled: three papers in this corpus measure its GRAIN and
nothing measured its TONE SCALE.  This sheet is the tone scale -- §3 speed,
§7 a full development matrix, §8 a spectral sensitivity curve, §9 a
characteristic-curve family and §10 time-Gbar curves.

⚠ AND IT DOES NOT JOIN UP WITH THE GRAIN, WHICH IS THE POINT OF THIS READER.
Ooue 1959 and Takano 1969 measured a film called Neopan SS.  This sheet is
dated 1999 by its own printer's code.  One trade name, two products, forty
years apart -- the trap this corpus already documents for EASTMAN_5247 (1974
against 1983) and for ILFORD PAN F against PAN F PLUS.  The grain block on
`FUJI_NEOPAN_SS` is therefore a flagged CLASS ESTIMATE from its cubic ISO 100
peers, and the four real measurements stay where they are, attached to no
profile.

WHAT MAKES THE CURVE TRACE SELF-CHECKING
-------------------------------------------
§9 draws five development times and PRINTS THE AVERAGE GRADIENT ON EACH:
4 min Gbar 0.28, 6 min 0.37, 8 min 0.45, 10 min 0.53, 12 min 0.61.  Nothing in
the trace is told those numbers, so reproducing them is a free check on the
axis calibration, the curve identification and the fit at once -- the same
property NEOPAN 1600's three printed Gbar values gave.

⚠ TWO CALIBRATION TRAPS IN THIS PANEL, BOTH FOUND THE HARD WAY:

  * THE FRAME IS NOT THE FIRST LABEL.  The abscissa runs from logH -4.0 at the
    frame to +1.0 at the right edge, and the leftmost PRINTED label is -3.0,
    one gridline pair in from the frame.  Reading the frame as -3.0 shifts
    every exposure by a full decade while leaving every density and every
    slope untouched -- so the Gbar check still passes and nothing complains.
    The calibration here is anchored on the "0.0" label, which is the only one
    without a minus sign and therefore the only one whose glyph centre lands
    on its gridline.
  * THE LABEL CENTROIDS AND THE GRIDLINES DISAGREE on the ordinate, 152.1 px
    against 158.7 px per 0.5 D, because the axis-title glyphs contaminate the
    label band.  The gridlines are right, and the proof is that they make one
    density unit 317.5 px against one exposure decade at 318.4 -- the 1:1
    aspect a sensitometric plot is drawn at.

⚠ AND THE TWO SHALLOWEST CURVES ARE NOT USABLE.  All five converge at the toe,
and below about Gbar 0.4 the follower cannot be kept off its neighbours: the
4 min and 6 min traces reconstruct Gbar at 0.67 and 0.87 of the printed value
against 0.99-1.11 for the other three.  They are traced, reported and excluded
from the fit rather than quietly averaged in -- and the GAP between those two
groups, not a tuned threshold, is what the 12 % gate reads.

WHAT IS ADOPTED
------------------
`FUJI_NEOPAN_SS`, stock 172, appended at frozen id 171 so no existing index
moves.  Curve from the 10 minute member -- the drawn curve nearest the sheet's
own recommendation of 9 1/2 min for Microfine at 20 C and EI 100 -- spectral
sensitivity from §8, processing from §7, speed from §3.

⚠ WHAT IS NOT: the SHOULDER, which the panel never reaches (it stops at D 1.82
with the curve still straight), so Dmax is pinned to a class 2.70 and refitting
at 2.5 or 3.0 moves the fit rms by 0.0002 D.  And the whole IMAGE-STRUCTURE
block, which the sheet does not have -- all four pages searched.

Usage:
    python3 fuji_neopan_ss.py --root . [--assert]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PDF_REL = os.path.join("PDF", "PROFILES", "FUJI", "SS35.pdf")

#: 595 pt wide against the embedded raster is about 8 px/pt at this scale; the
#: clips are in POINTS so the geometry survives a different rasteriser.
SCALE = 8.0

# --- §9 CHARACTERISTIC CURVES, page 3.  Frame x 132.5/1724.5, y 65.5/858.5.
#     ⚠ The abscissa's FRAME is logH -4.0, not the leftmost label: "0.0" sits
#     on gridline x 1405.5 and one decade is 318.4 px (two 159.2 px gridlines).
#     The ordinate is calibrated on the GRIDLINES, not the label centroids --
#     317.5 px per density unit, which is 1:1 with the exposure decade.
F09 = dict(clip=(330, 335, 555, 470), page=2,
           x_zero=1405.5, px_per_decade=318.4,
           y_zero=858.5, px_per_d=317.5,
           grid_dy=158.7, n_grid_y=6,
           x_lo=240, x_hi=1560, y_lo=70, y_hi=856,
           seed_x=1200)
#: The printed average gradients, top curve first.  NOTHING IN THE TRACE IS
#: TOLD THESE -- they are the check.
F09_PRINTED = ((12, 0.61), (10, 0.53), (8, 0.45), (6, 0.37), (4, 0.28))
#: The printed "Base Density" rule, read off the panel at D 0.245.
F09_BASE_D = 0.245
#: Dmax is a class value: the panel never reaches the shoulder.
F09_DMAX = 2.70
#: Fit window, inside the plotted range and clear of the converging toe.
F09_FIT = (-2.35, 0.45)
#: How far a RECONSTRUCTED Gbar may sit from the printed one before the trace
#: is refused.  12 % separates the three usable curves (1.11 / 0.99 / 1.08)
#: from the two the follower cannot keep apart at the toe (0.87 / 0.67), and
#: the gap between those groups is what makes the threshold a reading of the
#: data rather than a tuned number.
F09_GBAR_TOL = 0.12

# --- §8 SPECTRAL SENSITIVITY, page 3.  400 nm at x 349.5, 4.303 px/nm;
#     one log unit is 427.6 px and the ordinate has NO zero, only a bracket.
F08 = dict(clip=(72, 335, 300, 530), page=2,
           x_400=349.5, px_per_nm=4.303,
           y_ref=922.0, px_per_log=427.6,
           grid_y=(66.5, 495.0, 922.0, 1349.5),
           y_lo=300, y_hi=1346, x_lo=268, x_hi=1722)


def page_gray(doc, spec):
    import pymupdf
    p = doc[spec["page"]].get_pixmap(
        matrix=pymupdf.Matrix(SCALE, SCALE),
        clip=pymupdf.Rect(*spec["clip"]),
        colorspace=pymupdf.csGRAY)
    return np.frombuffer(p.samples, dtype=np.uint8).reshape(p.height, p.width)


def _runs(ink, x, lo, hi, grid, max_len=22, guard=4.0):
    """Ink-run centres in one column, gridlines removed.

    ⚠ REMOVING THE GRIDLINES COSTS A SAMPLE WHEREVER A CURVE CROSSES ONE, and
    that is why the walker below coasts per track instead of requiring every
    track to find a candidate at every column.
    """
    out, s = [], None
    for y in range(lo, hi):
        if ink[y, x]:
            if s is None:
                s = y
        elif s is not None:
            if y - s <= max_len:
                m = (s + y - 1) / 2.0
                if not any(abs(m - g) < guard for g in grid):
                    out.append(m)
            s = None
    return out


def _multiwalk(ink, g, grid, x0, ys, step, xlo, xhi):
    """Follow all five curves at once, each coasting on its own miss count.

    ⚠ A JOINT WALKER THAT REQUIRES ALL FIVE TO MATCH DIES AT THE FIRST GRIDLINE
    CROSSING -- one masked sample stalls the whole step.  Each track therefore
    keeps its own miss counter and its own slope-scaled tolerance, and the only
    thing shared between them is mutual exclusion: no two tracks may claim the
    same ink run in one column.
    """
    n = len(ys)
    T = [{x0: y} for y in ys]
    cur, sl, ms = list(ys), [0.0] * n, [0] * n
    x = x0 + step
    while xlo <= x < xhi:
        cand = _runs(ink, x, g["y_lo"], g["y_hi"], grid)
        used = set()
        pred = [cur[i] + sl[i] * step * (1 + ms[i]) for i in range(n)]
        for i in sorted(range(n), key=lambda k: ms[k]):
            tol = (7.0 + 2.5 * abs(sl[i])) * (1 + 0.7 * ms[i])
            best = None
            for k, c in enumerate(cand):
                if k in used:
                    continue
                if abs(c - pred[i]) <= tol and (
                        best is None or abs(c - pred[i]) < abs(cand[best] - pred[i])):
                    best = k
            if best is None:
                ms[i] += 1
            else:
                used.add(best)
                sl[i] = 0.6 * sl[i] + 0.4 * ((cand[best] - cur[i])
                                             / (step * (1 + ms[i])))
                cur[i] = cand[best]
                T[i][x] = cand[best]
                ms[i] = 0
        if all(m > 30 for m in ms):
            break
        x += step
    return T


def trace_fig09(img):
    ink = img < 140
    g = F09
    grid = [g["y_zero"] - k * g["grid_dy"] for k in range(g["n_grid_y"])]
    seed = g["seed_x"]
    ys = _runs(ink, seed, g["y_lo"], g["y_hi"], grid)
    if len(ys) != len(F09_PRINTED):
        raise RuntimeError("seed column found %d curves, expected %d"
                           % (len(ys), len(F09_PRINTED)))
    L = _multiwalk(ink, g, grid, seed, ys, -1, g["x_lo"], seed)
    R = _multiwalk(ink, g, grid, seed, ys, +1, seed, g["x_hi"])
    out = []
    for i in range(len(ys)):
        t = dict(L[i])
        t.update(R[i])
        t[seed] = ys[i]
        xs = sorted(t)
        out.append(np.array(
            [[(x - g["x_zero"]) / g["px_per_decade"],
              (g["y_zero"] - t[x]) / g["px_per_d"]] for x in xs]))
    return out


def _softplus(x, k):
    z = x / k
    return np.where(z > 60.0, x, k * np.log1p(np.exp(np.minimum(z, 60.0))))


def fit_curve(pts):
    """Fit this schema's ToneCurve, with dmin PINNED to the printed base rule
    and Dmax pinned to a class value.

    ⚠ NEITHER PIN IS A CONVENIENCE.  The plotted curves never reach the base
    plateau, so dmin is unconstrained by the trace and a free fit runs it to
    whatever bound it is given; and the panel stops with the curve still
    straight, so the shoulder is unconstrained too.  `shoulder_k` is held at
    the 2*toe_k monotonicity bound this project fits every curve under.
    """
    from scipy.optimize import least_squares
    x, y = pts[:, 0], pts[:, 1]
    o = np.argsort(x)
    x, y = x[o], y[o]
    m = (x >= F09_FIT[0]) & (x <= F09_FIT[1])
    x, y = x[m], y[m]

    def model(p, xx):
        gam, tx, tk, rk = p
        sk = tk * (1.0 + rk)
        sx = tx + (F09_DMAX - F09_BASE_D) / gam
        return F09_BASE_D + gam * (_softplus(xx - tx, tk) - _softplus(xx - sx, sk))

    r = least_squares(lambda p: model(p, x) - y, [0.60, -2.4, 0.30, 0.5],
                      bounds=([0.2, -4.0, 0.05, 0.0], [1.2, 0.5, 2.0, 1.0]))
    gam, tx, tk, rk = r.x
    sk = tk * (1.0 + rk)
    sx = tx + (F09_DMAX - F09_BASE_D) / gam
    res = model(r.x, x) - y
    xs = np.linspace(-4.0, 4.0, 8001)
    ys = model(r.x, xs)
    i = int(np.argmax(ys >= F09_BASE_D + 0.10))
    gbar = (float(np.interp(xs[i] + 2.0, xs, ys)) - ys[i]) / 2.0
    lin = (x >= -1.5) & (x <= -0.7)
    slope = float(np.polyfit(x[lin], y[lin], 1)[0]) if lin.sum() > 20 else float("nan")
    return dict(dmin=F09_BASE_D, gamma=float(gam), toe_x=float(tx),
                toe_k=float(tk), shoulder_x=float(sx), shoulder_k=float(sk),
                rms=float(np.sqrt((res ** 2).mean())),
                max=float(np.abs(res).max()), n=int(len(x)),
                gbar=float(gbar), slope=slope)


def straight_slope(pts, lo=-1.4, hi=-0.6):
    x, y = pts[:, 0], pts[:, 1]
    m = (x >= lo) & (x <= hi)
    if m.sum() < 20:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def gbar_from_trace(pts, span=2.0):
    x, y = pts[:, 0], pts[:, 1]
    o = np.argsort(x)
    x, y = x[o], y[o]
    tgt = F09_BASE_D + 0.10
    if not (y >= tgt).any():
        return float("nan")
    i = int(np.argmax(y >= tgt))
    if x[i] + span > x[-1]:
        return float("nan")
    return (float(np.interp(x[i] + span, x, y)) - y[i]) / span


def trace_fig08(img):
    ink = img < 140
    g = F08
    out = {}
    for nm in range(380, 671, 10):
        x = int(round(g["x_400"] + (nm - 400) * g["px_per_nm"]))
        if not (g["x_lo"] <= x <= g["x_hi"]):
            continue
        r = [v for v in _runs(ink, x, g["y_lo"], g["y_hi"], g["grid_y"],
                              max_len=26)
             if v < g["y_hi"] - 6]
        if len(r) == 1:
            out[nm] = (g["y_ref"] - r[0]) / g["px_per_log"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()
    try:
        import pymupdf
    except ImportError:
        print("[!] pymupdf not installed:  pip install pymupdf")
        return 1
    import film_profiles as fp

    pdf = os.path.join(ns.root, PDF_REL)
    if not os.path.exists(pdf):
        print("[!] missing %s" % pdf)
        return 1
    doc = pymupdf.open(pdf)
    bad = 0

    print("FUJIFILM DATA SHEET \"NEOPAN SS (135)\", Ref. AF3-411E(N)")
    print("  %s" % PDF_REL)

    prof = fp._BY_NAME["FUJI_NEOPAN_SS"]

    # ---- §3 and §4, printed text ----------------------------------------
    txt = "".join(pg.get_text() for pg in doc)
    ok_speed = "ISO 100/21" in txt and "Orthopanchromatic" in txt
    if not ok_speed:
        bad += 1
    print("\n  §3/§4 -- the sheet's own words")
    print("    [%s] «ISO 100/21°» and «Orthopanchromatic» are printed, and the "
          "profile stores EI %d" % ("OK  " if ok_speed else "FAIL",
                                    prof.exposure_index))
    ok_nostruct = not any(k in txt for k in
                          ("RMS GRANULARITY", "RMS Granularity",
                           "RESOLVING POWER", "Resolving Power"))
    if not ok_nostruct:
        bad += 1
    print("    [%s] ⚠ AND THERE IS NO IMAGE-STRUCTURE SECTION -- no rms "
          "granularity, no resolving power, no MTF, no reciprocity, no base "
          "thickness. All four pages searched, which is why the grain block "
          "is a flagged class estimate" % ("OK  " if ok_nostruct else "FAIL"))

    # ---- §9 ---------------------------------------------------------------
    print("\n  §9 CHARACTERISTIC CURVES -- Microfine 20 C, small tank")
    img9 = page_gray(doc, F09)
    curves = trace_fig09(img9)
    rows = []
    for (mins, printed), pts in zip(F09_PRINTED, curves):
        s = straight_slope(pts)
        gb = gbar_from_trace(pts)
        rows.append((mins, printed, s, gb, len(pts)))
        print("    %2d min  printed Gbar %.2f   traced straight slope %.3f "
              "(%.2fx)   reconstructed Gbar %.3f (%.2fx)   %d columns"
              % (mins, printed, s, s / printed, gb, gb / printed, len(pts)))
    good = [r for r in rows if abs(r[3] / r[1] - 1.0) <= F09_GBAR_TOL]
    okgb = len(good) >= 3 and all(m in [r[0] for r in good] for m in (8, 10, 12))
    if not okgb:
        bad += 1
    print("    [%s] ⚠ THE PRINTED GRADIENTS ARE REPRODUCED ON THE THREE "
          "STEEPEST CURVES to within %.0f %%, and nothing in the trace was "
          "told them -- a free check on the axis calibration, the curve "
          "identification and the fit at once"
          % ("OK  " if okgb else "FAIL", 100 * F09_GBAR_TOL))
    okbad = all(abs(r[3] / r[1] - 1.0) > F09_GBAR_TOL for r in rows if r[0] in (4, 6))
    if not okbad:
        bad += 1
    print("    [%s] ⚠ AND THE TWO SHALLOWEST ARE REFUSED, NOT AVERAGED IN: "
          "all five converge at the toe and below Gbar 0.4 the follower "
          "cannot be kept off its neighbours" % ("OK  " if okbad else "FAIL"))
    okmono = all(rows[i][2] > rows[i + 1][2] for i in range(len(rows) - 1))
    if not okmono:
        bad += 1
    print("    [%s] the five traced slopes are monotone in development time, "
          "in the printed order" % ("OK  " if okmono else "FAIL"))

    fit = fit_curve(curves[1])            # the 10 minute curve
    c = prof.curves.g
    # ⚠ 3e-3, NOT AN EXACT MATCH, AND THE REASON IS IN THE TRACE. The walker's
    # left extent depends on where its per-track miss counters give out, which
    # moves by a column or two with the fit window, so the last digit of each
    # fitted parameter is not reproducible to machine precision. 3e-3 is two
    # orders below the fit's own 0.023 D rms -- tight enough to catch a
    # different curve or a different calibration, loose enough not to fire on
    # a one-column difference in where the trace ended.
    okfit = all(abs(getattr(c, k) - fit[k]) < 3e-3 for k in
                ("dmin", "gamma", "toe_x", "toe_k", "shoulder_x", "shoulder_k"))
    if not okfit:
        bad += 1
    print("    fit  ToneCurve(%.4f, %.4f, %.4f, %.4f, %.4f, %.4f)  "
          "rms %.4f D  max %.4f D  n %d"
          % (fit["dmin"], fit["gamma"], fit["toe_x"], fit["toe_k"],
             fit["shoulder_x"], fit["shoulder_k"], fit["rms"], fit["max"],
             fit["n"]))
    print("    [%s] FUJI_NEOPAN_SS carries exactly this curve%s"
          % ("OK  " if okfit else "FAIL", "" if okfit else
             ": stored (%.4f, %.4f, %.4f, %.4f, %.4f, %.4f)"
             % (c.dmin, c.gamma, c.toe_x, c.toe_k, c.shoulder_x, c.shoulder_k)))
    okmodel = abs(fit["gbar"] / 0.53 - 1.0) < F09_GBAR_TOL
    if not okmodel:
        bad += 1
    print("    [%s] the MODEL's Gbar over dlogH 2.0 is %.3f against the "
          "printed 0.53 (%+.1f %%), while its straight-line slope is %.3f -- "
          "both honoured, gamma being the straight-line value"
          % ("OK  " if okmodel else "FAIL", fit["gbar"],
             100 * (fit["gbar"] / 0.53 - 1.0), fit["slope"]))
    print("    [note] ⚠ dmin %.3f is the sheet's printed «Base Density» rule, "
          "not a fitted value -- the plotted curves never reach the plateau. "
          "⚠ AND THE SHOULDER IS NOT MEASURED: the panel stops at D 1.82 with "
          "the curve still straight, so Dmax is pinned at a class %.2f and "
          "refitting at 2.5 or 3.0 moves the rms by 0.0002 D"
          % (F09_BASE_D, F09_DMAX))

    # ---- §8 ---------------------------------------------------------------
    print("\n  §8 SPECTRAL SENSITIVITY -- spectrogram to daylight 5400 K")
    img8 = page_gray(doc, F08)
    sp = trace_fig08(img8)
    peak_nm = max(sp, key=lambda k: sp[k])
    trough_nm = min((k for k in sp if 470 <= k <= 520), key=lambda k: sp[k])
    red_nm = max((k for k in sp if 560 <= k <= 620), key=lambda k: sp[k])
    print("    traced %d samples over %d-%d nm; peak %d nm, trough %d nm, "
          "secondary red lobe %d nm"
          % (len(sp), min(sp), max(sp), peak_nm, trough_nm, red_nm))
    okortho = (400 <= peak_nm <= 430 and 480 <= trough_nm <= 510
               and 570 <= red_nm <= 610 and max(sp) <= 650)
    if not okortho:
        bad += 1
    print("    [%s] ⚠ THE ORTHOPANCHROMATIC SIGNATURE §4 STATES IN WORDS IS "
          "WHAT THE TRACE SHOWS: a blue peak, a green-blue trough, a secondary "
          "red lobe, and the curve leaving the panel past 650 nm rather than "
          "running on into the deep red" % ("OK  " if okortho else "FAIL"))
    stored = prof.spectral.log_s_pan
    okstore = (len(stored) == 29 and abs(max(stored)) < 1e-9
               and abs(stored[3]) < 1e-9)
    if not okstore:
        bad += 1
    print("    [%s] the stored curve is peak-normalised at 410 nm and claims "
          "no absolute level -- the ordinate carries one «1.0» bracket and no "
          "zero" % ("OK  " if okstore else "FAIL"))

    # ---- the four measurements this profile is NOT joined to --------------
    print("\n  ⚠ THE FOUR NEOPAN SS GRANULARITY MEASUREMENTS, AND WHY NONE IS "
          "USED HERE")
    print("    Ooue 1959 Part 2 Fig. 26   Wiener spectrum, Minidol 20 C 10 min")
    print("    Ooue 1959 «23_7» Fig. 7    sigma vs D at a stated 10 um aperture")
    print("    Takano 1969 Fig. 8         Selwyn G at thirteen apertures "
          "-> clump_um %.2f um" % fp._TAKANO_APERTURE_FIT_1969[1][2])
    okage = (prof.grain.rms_granularity == 9.0
             and not any(s.status == "measured"
                         for s in fp._PARAM_SOURCES.get("FUJI_NEOPAN_SS", ())
                         if s.param.startswith("grain")))
    if not okage:
        bad += 1
    print("    [%s] ⚠ THEY MEASURE THE 1959-1969 COATING AND THIS SHEET IS "
          "DATED 1999 BY ITS OWN PRINTER'S CODE. One trade name, two products, "
          "forty years apart -- the trap already on file for EASTMAN_5247 "
          "(1974 against 1983) and ILFORD PAN F against PAN F PLUS. The grain "
          "block is a flagged class estimate from AGFA_APX_100 (9.0) and "
          "KODAK_PLUS_X_125 (9.5)" % ("OK  " if okage else "FAIL"))

    # ---- the index contract ----------------------------------------------
    # ⚠ REWRITTEN 2026-09-02e, AND THE OLD FORM WAS THE WRONG TEST. It asserted
    # that this stock is the LAST profile, which was true on the day it was
    # written and stopped being true the moment queue T3 appended three more --
    # so it failed the build on a database that is entirely correct. The
    # contract was never "last"; it is "this stock's frozen id is 171 and
    # nothing was inserted below it", which is what saves every existing
    # ListBox index and which keeps holding however many stocks are appended
    # after it.
    okidx = (fp.FILM_IDS.get("FUJI_NEOPAN_SS") == 171
             and sorted(fp.FILM_IDS.values()) == list(range(len(fp.FILM_IDS))))
    if not okidx:
        bad += 1
    print("\n  [%s] frozen id %s of %d stocks, and the id space is dense from "
          "0 -- nothing was inserted below it, so no existing ListBox index "
          "moves" % ("OK  " if okidx else "FAIL",
                     fp.FILM_IDS.get("FUJI_NEOPAN_SS"),
                     len(fp.FILM_PROFILES)))

    print()
    if bad:
        print("  [FAIL] %d problem(s)" % bad)
        return 1 if ns.do_assert else 0
    print("  [OK  ] N1: FUJI_NEOPAN_SS added from AF3-411E(N) -- curve, "
          "spectral sensitivity, speed and processing measured; grain and MTF "
          "flagged estimates, and the four same-name granularity measurements "
          "deliberately not joined to it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
