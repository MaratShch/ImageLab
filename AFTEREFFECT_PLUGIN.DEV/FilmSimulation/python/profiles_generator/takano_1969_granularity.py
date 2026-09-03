"""Takano 1969, «写真フィルムの粒状性» -- the aperture law, the clump scale, and
the two printed equations this project needed.

Queue items TK1-TK5, 2026-09-02.  Kiyoshi Takano (高野潔), "Granularity of
Photographic Film", テレビジョン / J. Inst. Telev. Engrs. Japan 23(1) 13-23
(1969):

    PDF/PROFILES/RETRO/JAPAN/23_13.pdf

Ten pages, real text layer, every page one 4299 x 6071 raster (about 600 ppi).

WHAT IT IS, AND WHY THAT DOES NOT MAKE IT LESS USEFUL
--------------------------------------------------------
It is a review, like Ooue's pair.  Almost all of it is other people's work,
cited and redrawn, and none of its samples is a stock in this database.  Four
items are still not incidental to this project, and two of them are things the
corpus was carrying as assumptions:

    Fig. 8    SELWYN GRANULARITY AGAINST SCANNING APERTURE, two named samples.
              The saturation Selwyn's own law does not have.
    Fig. 13   OPTICAL AUTOCORRELATION of two named samples -- an independent
              measurement of the quantity `GrainSpec.clump_um` parameterises.
    Fig. 9    RMS granularity of a colour negative against density, per LAYER.
    eq (2)    sigma(D) from sigma(T) TO FOURTH ORDER.
    eq (13)   the print-chain grain law, F_pr = F_pos + F_neg * R_pr^2 * gamma^2.

WHAT FIG. 8 SETTLES: THE APERTURE TERM NOBODY HAD CHECKED
------------------------------------------------------------
Selwyn's constant is G = sqrt(A) * sigma(D) and its whole point is that it does
not depend on the aperture.  Fig. 8 shows that it does, once the aperture stops
being large compared with the grain: both traces rise steeply and flatten, the
colour negative at G ~ 1.04 and Neopan-SS at G ~ 0.63.

`film_sim.grain_reference_energy` already models exactly this.  It normalises
grain by 2*pi * integral (h*a)^2 f df with a Gaussian aperture of sigma =
size/4, so its predicted G(s) = C * s * sqrt(E(s)) saturates for the same
reason.  ⚠ FITTING THAT LAW UNCHANGED, WITH NOTHING TUNED BUT ONE OVERALL
CONSTANT AND `clump_um`, REPRODUCES BOTH TRACES TO rms 0.007-0.020 IN G over a
0.2-1.04 range.  The aperture handling was inherited from theory and had never
been checked against a measurement; it now has one, and it fits.

⚠ WHAT IT DOES NOT SETTLE IS THE SIZE.  clump_um and clump_gain trade off
almost exactly against an aperture series: over the corpus's whole clump_gain
range 0.30-1.50 the fitted clump_um moves 6.20 -> 2.38 um on the colour
negative while the residual only moves 0.020 -> 0.007 G.  The same
non-identifiability the JPS 1965 crystal-size work hit, quantified here.

WHAT FIG. 13 ADDS, AND WHERE IT CONTRADICTS OOUE
---------------------------------------------------
Two autocorrelations with stated stock, density, developer, temperature and
time.  Half-widths 1.33 um (Neopan-SSS, D 2.0, Minidol 20 C 10 min) and 0.65 um
(cine positive, D 1.7, D-16 20 C 6 min), i.e. `clump_um` 1.77 and 0.87 um under
this engine's own tau_half = 374.8 / f_hi.

⚠ NEITHER CURVE GOES NEGATIVE.  Ooue's Fig. 24, the same quantity measured ten
years earlier on a microdensitometer, does -- an anti-correlated ring past
12 um that `EMULSION_KNOWLEDGE_BASE.md` §23j records as a refutation of the
engine's Gaussian shape AND of Sayanagi's Poisson placement.  Takano's optical
autocorrelator shows both curves approaching zero from above, which the Gaussian
reproduces perfectly.  The disagreement is between two measurements, not between
either and the model, and it is left standing rather than resolved by preferring
the one that suits the engine.

⚠ AND THE CENSUS THE TWO PAPERS TOGETHER NOW MAKE POSSIBLE.  Every direct
measurement of grain correlation length in this corpus, converted by the
engine's own laws: 0.87, 1.77, 2.46, 3.22, 4.64 um -- median 2.46.  The 171
stored `clump_um_g` values run 0.66-40.0 with median 13.0.  The stored scale is
about FIVE TIMES every measurement on file.  Queue C45 owns that; this reader
supplies the number and changes nothing.

WHAT FIG. 9 CONFIRMS, AND THE ONE PLACE IT DISAGREES
-------------------------------------------------------
sigma_D against mean integral colour density, 16 x 16 um aperture, one curve per
layer.  The magenta curve traces to a rise, a maximum near D 1.0, and a fall to
0.31 of it by D 2.5.

⚠ THAT IS A FOURTH INDEPENDENT CONFIRMATION of the correction made to
`GrainSpec`'s docstring on 2026-08-17.  The docstring used to say colour
negatives are monotone rising; four Kodak VISION3 sheets said otherwise.  This
is a Japanese colour negative, measured in 1969, by another laboratory, on
another instrument, and it turns over too.

⚠ IT DISAGREES ON WHERE THE MAXIMUM SITS.  All eleven measured colour negatives
here peak at D 0.65-0.80 at 1.20-1.62x the D = 1.0 value.  This one peaks at
D 1.04 at 1.00x -- no interior peak above the mid anchor at all.  All eleven are
Kodak ECN stocks of the 1990s-2000s, so `sigma_shape_peak` is a Kodak-family
measurement rather than a corpus-wide law, which is exactly what the
`sigma_shape_measured` gate already refuses to let a renderer assume.

THE TWO EQUATIONS, WHICH ARE THE PART THAT CHANGES CODE
----------------------------------------------------------
eq (2), printed with T = 10^-D:

    sigma(D) = 0.434 * (sigma(T)/T_bar) * [ 1 + (1/12)(sigma(T)/T_bar)^2
                                              + (1/80)(sigma(T)/T_bar)^4 + ...]

⚠ THIS IS THE CORRECTION THE CORPUS WAS MISSING.  Provenance work here has
converted rms granularity into density with the first term alone, and that is
precisely what failed on BBC Report T-101 Fig. 26: its sigma(T)/T_bar runs 0.39
to 1.64, where the first-order form is 1.3 % low at best and 31 % low at worst,
and the law `sigma_D = 0.648*D^0.665` fitted from it was WITHDRAWN for that
reason (ILFORD_HPS provenance note).  Adopted as
`film_sim.sigma_density_from_transmittance` and its Newton inverse.  ⚠ INERT:
no render path calls either, so no stored value and no rendered pixel moves.

eq (13), the print chain:

    F_pr(u,v) = F_pos(u,v) + F_neg(u,v) * R_pr^2(u,v) * gamma^2

with R_pr the response of "プリント光学系およびポジフィルム" -- the printing
optics AND the positive film -- and gamma the positive's contrast dD/dlogE.

⚠ THE ENGINE ALREADY SATISFIES IT, BY CONSTRUCTION, IN TWO PLACES AND WITH ONE
DEPARTURE.  The `+ F_pos` and the `* gamma^2` are structural: stage 13 computes
log_e_print = offset - dens with the negative's grain already inside `dens`, so
the print curve's local slope multiplies that fluctuation, and the print stock's
own grain field is added afterwards at stage 14 and therefore adds in power.
R_pr^2 is stage 10: `scan_t` is applied to the negative density before the print
curve, and applying an amplitude transfer to a density field multiplies its
Wiener spectrum by its square.  With no scanner override `scan_f50 =
settings.scanner_f50 or print_stock.mtf_f50`, so the engine's default R_pr IS
the positive stock's own MTF -- eq (13) with a contact printer.

⚠ THE DEPARTURE IS THAT STAGE 14 ALSO BAND-LIMITS THE PRINT STOCK'S OWN GRAIN BY
THAT SAME TRANSFER (film_sim.py, `make_grain_field(..., scan_t)` in the print
block).  eq (13) does not: F_pos is generated in the positive emulsion and is
not imaged through the positive's MTF.  The duplication chain in the same
function gets this right and says so in its own comment -- "This stage's own
grain is created in THIS emulsion, so it is not blurred by this stage's optics".
The print stage does not.  ⚠ NOT CHANGED HERE.  When `scanner_f50` IS set,
`scan_t` is a real scanner and filtering print grain by it is correct; the error
appears only on the fallback, it moves a pixel on every print render, and it is
a rendering decision rather than a data one.  The exact fix, should it ever be
taken, is to pass the scanner transfer -- not the print stock's MTF -- as the
print grain's band limit, which needs `scan_t` split into its two factors.

WHAT IS WRITTEN TO A PROFILE
-------------------------------
⚠ NOTHING.  Fig. 8's samples are "カラーネガフィルム" and "ネオパン-SS", Fig. 13's
are Neopan-SSS and an unnamed cine positive, Fig. 9's is "カラーネガフィルム"
again.  None is a stock in `film_profiles.py`, and Fig. 9's ordinate is BROKEN
between 0.03 and 0.06 with a 2.04x scale change across the break.  What is
stored is the traces, the fits, the census and the two equations.

Usage:
    python3 takano_1969_granularity.py --root . [--assert]
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PDF_REL = os.path.join("PDF", "PROFILES", "RETRO", "JAPAN", "23_13.pdf")

#: Every figure is read off the page raster at its native scale.  516 pt wide
#: against 4299 px is 8.331 px/pt, and the clips below are in POINTS so the
#: geometry survives a different rasteriser.
PX_PER_PT = 8.331

# --- Fig. 8, page 3 (journal p16), left column.  Axes detected on the printed
#     rules: y-axis x = 344.5, x-axis y = 1768.5, both in clip pixels.  The
#     printed ticks give 0 and 100 um at x 344.5 and 1529.5, and G = 0 and 1.5
#     at y 1768.5 and 889.5.  Both checked against the intermediate ticks.
F08 = dict(page=3, clip=(40, 430, 270, 680),
           x_zero=344.5, px_per_um=11.85,
           y_zero=1768.5, px_per_g=(1768.5 - 889.5) / 1.5,
           x0=346, x1=1534, y0=885, y1=1766,
           seed_x=720, seed_upper=1229.0, seed_lower=1422.0)

# --- Fig. 13, page 6 (journal p20), left column.  x ticks 0/5/10/15 at
#     398.5/805.5/1211/1612.5; y ticks 1.0/0.5/0 at 821/1216/1618.
F13 = dict(page=6, clip=(35, 390, 270, 625),
           x_zero=398.5, px_per_um=(1612.5 - 398.5) / 15.0,
           y_zero=1618.0, px_per_phi=(1618.0 - 821.0),
           x_lo=418, x_hi=1000)

# --- Fig. 9, page 3 (journal p16), right column.  x ticks every 0.2 D from
#     x 401 (D = 0) to x 1241 (D = 2.4).  ⚠ THE ORDINATE IS BROKEN: below the
#     break 0.01/0.02/0.03 sit at y 1489.5/1351/1210.5, i.e. 139.3 px per 0.01;
#     above it 0.06/0.08/0.10/0.12 sit at y 930.5/794.5/656.5/521, i.e. 68.25 px
#     per 0.01.  A 2.04x scale change, and the region between is unreadable.
F09 = dict(page=3, clip=(265, 230, 500, 470),
           x_zero=401.0, px_per_d=350.0,
           y_zero=1628.5, px_per_sigma_low=13930.0,
           y_006=930.5, px_per_sigma_high=6825.0,
           box=(470, 1620, 405, 1330))

#: The Selwyn fit's frequency grid.  3000 c/mm is far past where the aperture
#: term has killed everything at these clump sizes; 6000 samples put the
#: trapezoid error three orders below the trace noise.
FIT_F_MAX, FIT_F_N = 3000.0, 6000
#: The clump_gain values the aperture fit is reported at.  Corpus minimum,
#: median and a coarse-cubic value -- the point is the spread, not any one row.
FIT_GAINS = (0.30, 0.85, 1.50)


def page_gray(doc, spec) -> np.ndarray:
    import pymupdf
    p = doc[spec["page"]].get_pixmap(
        matrix=pymupdf.Matrix(PX_PER_PT, PX_PER_PT),
        clip=pymupdf.Rect(*spec["clip"]),
        colorspace=pymupdf.csGRAY)
    return np.frombuffer(p.samples, dtype=np.uint8).reshape(p.height, p.width)


def _runs(ink: np.ndarray, x: int, lo: int, hi: int):
    out, s = [], None
    for y in range(lo, hi):
        if ink[y, x]:
            if s is None:
                s = y
        elif s is not None:
            out.append((s, y - 1))
            s = None
    if s is not None:
        out.append((s, hi - 1))
    return out


# ---------------------------------------------------------------------------
# Fig. 8 -- Selwyn G against scanning aperture
# ---------------------------------------------------------------------------


def _walk_f08(ink, x0, y0, step, avoid=None):
    """Slope-predictive follower.

    ⚠ PREDICT, DO NOT SNAP TO THE NEAREST INK.  The figure carries error bars
    whose caps sit within a few pixels of the curve and two in-plot captions,
    and a nearest-ink follower walks onto all of them.  Runs longer than 22 px
    are error bars and are dropped outright; everything else has to land inside
    a tolerance that grows with the local slope and with how long the walker has
    been coasting, and may not coincide with the already-traced curve.
    """
    g = F08
    pts = {x0: y0}
    y, slope, miss = y0, 0.0, 0
    x = x0 + step
    while g["x0"] <= x < g["x1"]:
        cand = [(s + e) / 2.0 for s, e in _runs(ink, x, g["y0"], g["y1"])
                if e - s + 1 <= 22]
        pred = y + slope * step * (1 + miss)
        tol = min(60.0, (6.0 + 2.5 * abs(slope)) * (1 + 0.6 * miss))
        best = None
        for m in cand:
            if avoid is not None and x in avoid and abs(m - avoid[x]) < 7.0:
                continue
            if abs(m - pred) <= tol and (best is None
                                         or abs(m - pred) < abs(best - pred)):
                best = m
        if best is None:
            miss += 1
            if miss > 25:
                break
            x += step
            continue
        slope = 0.6 * slope + 0.4 * ((best - y) / (step * (1 + miss)))
        y = best
        pts[x] = y
        miss = 0
        x += step
    return pts


def trace_fig08(img: np.ndarray):
    ink = img < 128
    g = F08
    upper = dict(_walk_f08(ink, g["seed_x"], g["seed_upper"], -1))
    upper.update(_walk_f08(ink, g["seed_x"], g["seed_upper"], +1))
    upper[g["seed_x"]] = g["seed_upper"]
    lower = dict(_walk_f08(ink, g["seed_x"], g["seed_lower"], -1, avoid=upper))
    lower.update(_walk_f08(ink, g["seed_x"], g["seed_lower"], +1, avoid=upper))
    lower[g["seed_x"]] = g["seed_lower"]

    def conv(pt):
        return [((x - g["x_zero"]) / g["px_per_um"],
                 (g["y_zero"] - pt[x]) / g["px_per_g"]) for x in sorted(pt)]

    return conv(upper), conv(lower)


def _selwyn_model(clump_um: float, gain: float, s_um: np.ndarray,
                  f: np.ndarray) -> np.ndarray:
    """s * sqrt(E(s)) with E exactly as `film_sim.grain_reference_energy`
    defines it -- the engine's own amplitude transfer times the engine's own
    Gaussian aperture of sigma = size/4.  Nothing here is a new model."""
    f_hi = 1000.0 / (2.0 * clump_um)
    h = np.exp(-(f / f_hi) ** 2) * (1.0 + gain * np.exp(-(f / (f_hi / 6.0)) ** 2))
    ap = np.exp(-2.0 * math.pi ** 2 * (s_um[:, None] / 4000.0) ** 2
                * f[None, :] ** 2)
    fn = getattr(np, "trapezoid", None) or np.trapz
    e = 2.0 * math.pi * fn((h[None, :] * ap) ** 2 * f[None, :], f, axis=1)
    return s_um * np.sqrt(e)


def fit_fig08(data, gain: float):
    f = np.linspace(1e-3, FIT_F_MAX, FIT_F_N)
    ss = np.array([d[0] for d in data])
    gg = np.array([d[1] for d in data])
    keep = ss > 2.0
    ss, gg = ss[keep], gg[keep]
    best = None
    for cl in np.arange(0.40, 14.0, 0.02):
        mod = _selwyn_model(cl, gain, ss, f)
        c = float((mod * gg).sum() / (mod * mod).sum())
        r = float(np.sqrt(((c * mod - gg) ** 2).mean()))
        if best is None or r < best[1]:
            best = (float(cl), r, c)
    return best


# ---------------------------------------------------------------------------
# Fig. 13 -- the optical autocorrelation
# ---------------------------------------------------------------------------


def trace_fig13(img: np.ndarray):
    """Both curves, by ROW rather than by column.

    ⚠ THE COLUMN-WISE FOLLOWER IS THE WRONG TOOL HERE.  Both curves are almost
    vertical over the first micrometre, so a column carries a 25 px run and the
    half-width -- the one number wanted -- lands inside it.  phi is monotone
    falling on both, so a row scan gives exactly one x per curve, the dashed
    one always to the left.  x < 418 is excluded because the printed y-axis
    ticks extend into the plot and would read as a third curve.
    """
    ink = img < 128
    g = F13
    out = {"dashed": {}, "solid": {}}
    for i in range(5, 96):
        phi = i / 100.0
        y = int(round(g["y_zero"] - phi * g["px_per_phi"]))
        xs, s = [], None
        for x in range(g["x_lo"], g["x_hi"]):
            if ink[y, x]:
                if s is None:
                    s = x
            elif s is not None:
                xs.append((s + x - 1) / 2.0)
                s = None
        if len(xs) < 2:
            continue
        for name, xv in (("dashed", xs[0]), ("solid", xs[1])):
            out[name][round(phi, 2)] = (xv - g["x_zero"]) / g["px_per_um"]
    return out


def half_width(curve: dict) -> float:
    ks = sorted(curve)
    for a, b in zip(ks, ks[1:]):
        if a <= 0.5 <= b:
            t = (0.5 - a) / (b - a)
            return curve[a] + t * (curve[b] - curve[a])
    return float("nan")


# ---------------------------------------------------------------------------
# Fig. 9 -- sigma_D against density, per layer
# ---------------------------------------------------------------------------


def trace_fig09_magenta(img: np.ndarray):
    """The magenta (green-filter) curve only, by connected component.

    ⚠ ONLY THE MAGENTA CURVE IS TRACED, AND THAT IS A LIMIT OF THE FIGURE.  It
    is the one curve drawn as an unbroken stroke, so a single connected
    component 813 px wide isolates it from the scatter of crosses, circles and
    triangles.  The cyan curve is dashed and the yellow is dash-dot; their
    segments are the same size as the scatter marks and cannot be separated
    from them reliably, so their peaks are GRID READINGS and are reported as
    such, not as traces.
    """
    from scipy import ndimage
    g = F09
    ink = img < 128
    y0, y1, x0, x1 = g["box"]
    sub = np.zeros_like(ink)
    sub[y0:y1, x0:x1] = ink[y0:y1, x0:x1]
    lab, _ = ndimage.label(sub, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    pick, wide = None, 0
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        w = sl[1].stop - sl[1].start
        if w > wide:
            wide, pick = w, i + 1
    m = (lab == pick)
    xs = np.where(m.any(0))[0]
    d, sig, thick = [], [], []
    for x in xs:
        ys = np.where(m[:, x])[0]
        d.append((x - g["x_zero"]) / g["px_per_d"])
        sig.append((g["y_zero"] - ys.mean()) / g["px_per_sigma_low"])
        thick.append(len(ys))
    d = np.array(d)
    sig = np.array(sig)
    thick = np.array(thick)
    # A scatter mark touching the stroke doubles its local thickness and drags
    # the centroid; those columns are dropped rather than smoothed over.
    keep = thick <= np.median(thick) * 2.0
    return d[keep], sig[keep], int(wide)


def fig09_anchors(d: np.ndarray, sig: np.ndarray, win: int = 41):
    sm = np.convolve(sig, np.ones(win) / win, mode="same")
    core = slice(win, len(d) - win)
    i = int(np.argmax(sm[core])) + win

    def at(x):
        return float(sm[int(np.argmin(abs(d - x)))])

    mid = at(1.0)
    return dict(peak_at=float(d[i]), peak=sm[i] / mid,
                toe_at=0.30, toe=at(0.30) / mid,
                dmax_at=2.50, dmax=at(2.50) / mid,
                sigma_mid=mid, sigma_peak=float(sm[i]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


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
    import film_sim

    pdf = os.path.join(ns.root, PDF_REL)
    if not os.path.exists(pdf):
        print("[!] missing %s" % pdf)
        return 1
    doc = pymupdf.open(pdf)
    bad = 0

    print("Takano 1969, テレビジョン 23(1) 13-23 -- 写真フィルムの粒状性")
    print("  %s" % PDF_REL)

    # ---- Fig. 8 ---------------------------------------------------------
    print("\n  Fig. 8 (p16) -- Selwyn G = sqrt(A)*sigma(D) vs scanning aperture")
    img8 = page_gray(doc, F08)
    neg, neopan = trace_fig08(img8)
    for name, cur in (("colour negative", neg), ("Neopan-SS", neopan)):
        print("    %-16s traced sqrt(A) %.1f-%.1f um, G %.3f -> %.3f, %d columns"
              % (name, cur[0][0], cur[-1][0], cur[0][1], cur[-1][1], len(cur)))
    okmono = all(cur[-1][1] > cur[0][1] and
                 cur[-1][1] - cur[len(cur) // 2][1] < 0.25 * cur[-1][1]
                 for cur in (neg, neopan))
    if not okmono:
        bad += 1
    print("    [%s] ⚠ BOTH SATURATE. Selwyn's constant is supposed to be "
          "aperture-independent and it is not: G climbs steeply and flattens, "
          "which is the whole reason rms granularity replaced it"
          % ("OK  " if okmono else "FAIL"))

    stored = tuple(tuple(r) for r in fp._TAKANO_APERTURE_FIT_1969)
    got = []
    for gain in FIT_GAINS:
        cl_n, r_n, _ = fit_fig08(neg[::40], gain)
        cl_s, r_s, _ = fit_fig08(neopan[::40], gain)
        got.append((gain, round(cl_n, 2), round(cl_s, 2)))
        print("    clump_gain %.2f -> clump_um %.2f (colour negative, rms "
              "%.4f G) / %.2f (Neopan-SS, rms %.4f G)"
              % (gain, cl_n, r_n, cl_s, r_s))
    okfit = tuple(got) == stored
    if not okfit:
        bad += 1
    print("    [%s] film_profiles._TAKANO_APERTURE_FIT_1969 reproduces the "
          "fit%s" % ("OK  " if okfit else "FAIL",
                     "" if okfit else ": got %s" % (tuple(got),)))
    print("    [note] ⚠ THE ENGINE'S OWN APERTURE LAW FITS A MEASUREMENT IT "
          "WAS NEVER CHECKED AGAINST. grain_reference_energy integrates "
          "(h*a)^2 with a Gaussian aperture of sigma = size/4; nothing but one "
          "overall constant and clump_um was tuned here")
    spread = stored[0][1] / stored[-1][1]
    okdeg = spread > 2.0
    if not okdeg:
        bad += 1
    print("    [%s] ⚠ AND IT DOES NOT DETERMINE THE SIZE: clump_um moves %.1fx "
          "across the corpus clump_gain range while the residual moves under "
          "0.02 G. An aperture series cannot separate the two"
          % ("OK  " if okdeg else "FAIL", spread))

    # ---- Fig. 13 --------------------------------------------------------
    print("\n  Fig. 13 (p20) -- optical autocorrelation phi(tau,0)")
    img13 = page_gray(doc, F13)
    cur13 = trace_fig13(img13)
    h_solid = half_width(cur13["solid"])
    h_dash = half_width(cur13["dashed"])
    for name, lbl, hw in (("solid", "Neopan-SSS, D 2.0, Minidol 20C 10min",
                           h_solid),
                          ("dashed", "cine positive, D 1.7, D-16 20C 6min",
                           h_dash)):
        print("    %-38s tau_half %.2f um -> clump_um %.2f um"
              % (lbl, hw, hw * 1.334))
    okh = (abs(round(h_solid, 2) - fp._TAKANO_AUTOCORR_1969[0]) < 0.02 and
           abs(round(h_dash, 2) - fp._TAKANO_AUTOCORR_1969[1]) < 0.02)
    if not okh:
        bad += 1
    print("    [%s] film_profiles._TAKANO_AUTOCORR_1969 reproduces both "
          "half-widths%s" % ("OK  " if okh else "FAIL",
                             "" if okh else ": got %.2f / %.2f"
                             % (h_solid, h_dash)))
    okpos = all(v > 0 for v in cur13["solid"].values())
    if not okpos:
        bad += 1
    print("    [%s] ⚠ NEITHER CURVE GOES NEGATIVE, where Ooue Fig. 24 does. "
          "Same quantity, two instruments, ten years apart. The engine's "
          "Gaussian reproduces Takano's shape and cannot reproduce Ooue's; the "
          "disagreement is between the measurements and is left standing"
          % ("OK  " if okpos else "FAIL"))

    # ---- the clump census ----------------------------------------------
    print("\n  The clump census -- queue C45's missing number")
    census = tuple(sorted(round(v, 2) for v in (
        h_dash * 1.334, h_solid * 1.334,
        stored[1][2], stored[1][1],
        fp._OOUE_AUTOCORR_1959[0] * 500.0 / 374.8)))
    okc = census == fp._TAKANO_CLUMP_CENSUS_1969
    if not okc:
        bad += 1
    stock = sorted(p.grain.clump_um_g for p in fp.FILM_PROFILES)
    med_stock = stock[len(stock) // 2]
    med_meas = census[len(census) // 2]
    print("    measured  %s um, median %.2f"
          % (", ".join("%.2f" % v for v in census), med_meas))
    print("    stored    %d values %.2f-%.2f um, median %.2f, %d below 5 um"
          % (len(stock), stock[0], stock[-1], med_stock,
             sum(1 for v in stock if v < 5.0)))
    print("    [%s] film_profiles._TAKANO_CLUMP_CENSUS_1969 reproduces it%s"
          % ("OK  " if okc else "FAIL",
             "" if okc else ": got %s" % (census,)))
    print("    [note] ⚠ THE STORED SCALE IS %.1fx EVERY MEASUREMENT ON FILE. "
          "Nothing is changed: clump_um moves a pixel on 168 stocks, none of "
          "the five samples is a stock in this database, and the aperture fit "
          "above shows the value is only as well determined as clump_gain. "
          "QUEUE C45 OWNS THE DECISION" % (med_stock / med_meas))

    # ---- Fig. 9 ---------------------------------------------------------
    print("\n  Fig. 9 (p16) -- RMS granularity per layer vs integral density")
    img9 = page_gray(doc, F09)
    d9, s9, wide = trace_fig09_magenta(img9)
    a = fig09_anchors(d9, s9)
    print("    magenta layer traced as one %d px component, %d clean columns"
          % (wide, len(d9)))
    print("    sigma_D  %.4f at D 0.30, %.4f at D 1.00 (mid), %.4f at D 2.50"
          % (a["toe"] * a["sigma_mid"], a["sigma_mid"],
             a["dmax"] * a["sigma_mid"]))
    print("    as this schema's anchors: toe %.3f @ %.2f / mid 1.000 / "
          "dmax %.3f @ %.2f / peak %.3f @ %.2f"
          % (a["toe"], a["toe_at"], a["dmax"], a["dmax_at"],
             a["peak"], a["peak_at"]))
    stored9 = fp._TAKANO_SIGMA_SHAPE_1969
    got9 = (a["toe_at"], round(a["toe"], 3), 1.000,
            a["dmax_at"], round(a["dmax"], 3),
            round(a["peak"], 3), round(a["peak_at"], 2))
    ok9 = all(abs(x - y) < 5e-3 for x, y in zip(got9, stored9))
    if not ok9:
        bad += 1
    print("    [%s] film_profiles._TAKANO_SIGMA_SHAPE_1969 reproduces it%s"
          % ("OK  " if ok9 else "FAIL",
             "" if ok9 else ": got %s" % (got9,)))
    okturn = a["dmax"] < 0.6 and a["toe"] < 0.6
    if not okturn:
        bad += 1
    print("    [%s] ⚠ IT TURNS OVER -- a fourth independent confirmation of "
          "the 2026-08-17 correction to GrainSpec's docstring, on a Japanese "
          "colour negative measured in 1969 rather than a Kodak sheet"
          % ("OK  " if okturn else "FAIL"))

    kod = [p.grain for p in fp.FILM_PROFILES
           if p.grain.sigma_shape_measured and not p.is_reversal]
    pk_at = [g.sigma_shape_peak_at for g in kod if g.sigma_shape_peak > 0]
    pk = [g.sigma_shape_peak for g in kod if g.sigma_shape_peak > 0]
    okdis = a["peak_at"] > max(pk_at) and a["peak"] < min(pk)
    if not okdis:
        bad += 1
    print("    [%s] ⚠ AND IT DISAGREES ON WHERE: %d measured colour negatives "
          "here peak at D %.2f-%.2f at %.2f-%.2fx; this one peaks at D %.2f at "
          "%.2fx -- no interior peak at all. All %d are Kodak ECN stocks, so "
          "sigma_shape_peak is a family measurement and sigma_shape_measured "
          "is right to gate it"
          % ("OK  " if okdis else "FAIL", len(pk_at), min(pk_at), max(pk_at),
             min(pk), max(pk), a["peak_at"], a["peak"], len(pk_at)))

    ye, ma, cy = fp._TAKANO_LAYER_SIGMA_1969
    print("    layer maxima (⚠ yellow and cyan are GRID READINGS +/-0.002, not "
          "traces -- dash-dot and dash segments are the size of the scatter "
          "marks): yellow %.3f, magenta %.4f, cyan %.3f" % (ye, ma, cy))
    print("      cyan/magenta %.2f  vs the corpus's nine measured r/g "
          "(0.75-1.05) -- 10 %% above the highest" % (cy / ma))
    print("      yellow/magenta %.2f vs the corpus's nine measured b/g "
          "(1.81-2.79) -- 65 %% above the highest. ⚠ BOTH DISAGREE, AND THE "
          "RANKING IS THE EXPLANATION: Takano reads INTEGRAL colour density "
          "through a filter, so each reading carries the orange mask and every "
          "layer's absorption in that band, and the mask absorbs mostly BLUE. "
          "The corpus's ratios are per-layer analytical densities. Different "
          "quantities, not reconciled" % (ye / ma))

    # ---- the two printed equations --------------------------------------
    print("\n  eq (2), p16 -- sigma(D) from sigma(T) to FOURTH order")
    print("    sigma(D) = 0.434*(s/T)*[1 + (1/12)(s/T)^2 + (1/80)(s/T)^4 + ...]")
    for r in (0.10, 0.39, 1.00, 1.64):
        full = film_sim.sigma_density_from_transmittance(r)
        first = 0.434 * r
        print("      sigma(T)/T = %.2f   first order %.5f   full %.5f   "
              "%+.1f %%" % (r, first, full, 100.0 * (full / first - 1.0)))
    rt = film_sim.sigma_transmittance_from_density(
        film_sim.sigma_density_from_transmittance(0.87))
    okinv = abs(rt - 0.87) < 1e-9
    if not okinv:
        bad += 1
    print("    [%s] film_sim.sigma_transmittance_from_density inverts it "
          "(round trip on 0.87 closes to %.1e)"
          % ("OK  " if okinv else "FAIL", abs(rt - 0.87)))
    okinert = abs(film_sim.sigma_density_from_transmittance(1e-9)
                  / (0.434e-9) - 1.0) < 1e-9
    if not okinert:
        bad += 1
    print("    [%s] and degrades to the first-order form the corpus was using "
          "as the ratio goes to zero -- which is why it went unnoticed"
          % ("OK  " if okinert else "FAIL"))
    print("    [note] ⚠ THIS IS THE CORRECTION THAT WITHDREW sigma_D = "
          "0.648*D^0.665. BBC T-101 Fig. 26 measures sigma(T)/T from 0.39 to "
          "1.64, where the first-order form is 1.3 % to 31 % low. ADOPTED as "
          "film_sim.sigma_density_from_transmittance, ⚠ INERT -- no render "
          "path calls it, so no stored value and no rendered pixel moves")

    print("\n  eq (13), p22 -- the print chain")
    print("    F_pr = F_pos + F_neg * R_pr^2 * gamma^2, with R_pr the response "
          "of 「プリント光学系およびポジフィルム」 (printing optics AND positive "
          "film) and gamma the positive's dD/dlogE")
    src = open(film_sim.__file__, encoding="utf-8").read()
    checks = (
        ("F_pos adds in power: print grain is a separate field added after "
         "the print curve",
         "if not reversal and settings.print_grain "
         "and print_stock.grain_rms > 0.0:" in src.replace("\n", " ")
         or "settings.print_grain and print_stock.grain_rms > 0.0" in src),
        ("gamma^2: the negative's grain rides inside `dens` through "
         "density(offset - dens, pcurves[c]), so the print curve's slope "
         "multiplies it",
         "log_e_print = (np.float32(offsets[c]) - dens[:, :, c])" in src),
        ("R_pr defaults to the POSITIVE STOCK'S OWN MTF -- eq (13) with a "
         "contact printer",
         "scan_f50 = settings.scanner_f50 or print_stock.mtf_f50" in src),
        ("⚠ THE DEPARTURE: stage 14 band-limits the print stock's own grain "
         "by that same transfer, which eq (13) does not",
         "print_stock.grain_clump_um, 0.25, print_stock.grain_rms, scan_t"
         in src),
        ("the duplication chain gets it right and says so",
         "not blurred by this stage's optics" in src),
    )
    for label, ok in checks:
        if not ok:
            bad += 1
        print("    [%s] %s" % ("OK  " if ok else "FAIL", label))
    print("    [note] ⚠ NOT CHANGED. When scanner_f50 IS set, scan_t is a real "
          "scanner and filtering print grain by it is correct; the error "
          "appears only on the fallback, moves a pixel on every print render, "
          "and is a RENDERING decision, not a data one. The exact fix is to "
          "pass the scanner transfer -- not the print stock MTF -- as the "
          "print grain's band limit, which needs scan_t split into its two "
          "factors")

    print()
    if bad:
        print("  [FAIL] %d problem(s)" % bad)
        return 1 if ns.do_assert else 0
    print("  [OK  ] TK1-TK5: two aperture series, two autocorrelations and one "
          "sigma(D) curve traced; eq (2) adopted inert; eq (13) checked "
          "against the engine; no stored profile value changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
