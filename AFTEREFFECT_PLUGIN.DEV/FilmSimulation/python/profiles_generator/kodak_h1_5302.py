#!/usr/bin/env python3
"""H-1-5302 -- the 5302 / 7302 release positive's own Kodak sheet, read.

WHAT THE DOCUMENT IS
--------------------
KODAK Publication No. H-1-5302, «EASTMAN Fine Grain Release Positive Film
5302 (TM) / 7302 (TM)», TECHNICAL DATA / PRINT FILM, February 1999, Minor
Revision 2/99, CAT 831 2100. Four pages. The manufacturer describing its own
product.

WHY IT MATTERS MORE THAN A FOURTH KODAK SHEET USUALLY WOULD
-----------------------------------------------------------
`KODAK_5302` has been in this database since 2026-08-24 as a [T2] record built
from BBC Report T-101 and BBC Engineering Monograph 54 -- two third-party
research reports from 1963-64. T-101 prints no characteristic curve, no MTF
and no resolving power for any of its six emulsions, so FOUR of the profile's
numbers were estimates with the emulsion class as their only justification.

⚠⚠ ONE OF THOSE ESTIMATES WAS WRONG BY A FACTOR OF 2.2. `mtf_f50` was 85.0
cycle/mm, justified in the file as "a fine-grain 16 mm positive, nothing
more". Panel F010_0025AC on page 3 of this sheet crosses 50 % response at
38.2 cycle/mm. That correction is the single largest this source produced and
it is the reason the module exists rather than a note in a report.

WHAT IS READ, AND WHY THERE IS NO SCANNING ERROR TERM AT ALL
-------------------------------------------------------------
The file is VECTOR. Every curve on page 3 is a PDF path, so the points below
come out of the path data itself; the only error terms are the draughtsman's
and the axis fit's, and both are reported. Three panels:

    F010_0023AC   characteristic curves -- FIVE development times, plus a
                  Time-Gamma inset and a Time-Fog inset in the same frame
    F010_0024AC   spectral sensitivity -- TWO criteria, D = 0.3 and D = 1.0
                  above gross fog
    F010_0025AC   the modulation-transfer curve

THE ONE JUDGEMENT ON THE PAGE, AND HOW IT WAS REMOVED
------------------------------------------------------
Kodak letters the five development times -- 2, 3 1/2, 5, 7 and 9 minutes --
BESIDE the curves rather than on them, and there is no leader line. Two
independent facts fix the mapping and a third fixes which trace to adopt:

  1. Within one emulsion, one developer and one temperature, gamma rises
     monotonically with development time. Sorting the five traces by steepness
     therefore IS sorting them by time. Not in dispute, and decisive.
  2. The Time-Gamma inset, traced from the same frame with no reference to the
     characteristic curves, gives a gamma at each printed time. Read against
     the traces' own 0.3-decade slopes the two agree to a mean of 0.12 of
     gamma; every alternative assignment is worse, reversal by six times.
  3. The PROCESSING table prints NO development time. Its footnote says
     "Develop to the recommended control gamma of 2.4 to 2.6 Status M
     Densitometry (Blue)", and the inset puts 2.40 at 3.18 min, 2.50 at
     3.55 min and 2.60 at 3.99 min -- so the 3 1/2-minute trace is the only
     drawn one inside Kodak's own control band, and the band's CENTRE lands on
     it to within three seconds. That is the trace the profile stores.

⚠ THE SPECTRAL PANEL IS TRUNCATED AND THE TRUNCATION IS KODAK'S. The frame is
ruled from 250 to 750 nm; the curve begins at 400. The peak of a
blue-sensitive emulsion is at or below 400 nm and this document does not show
it, so the stored record is normalised to its 400 nm value and NOT to a peak.
Saying otherwise would claim a maximum the sheet never draws.

⚠ THE TWO SPECTRAL CRITERIA ARE A FREE CROSS-CHECK AND ONLY IF THE ANCHOR IS
KEPT. Going from D = 0.3 to D = 1.0 above gross fog costs 0.7/gamma of log
exposure, so the two drawn curves must be separated by that much -- 0.28 at
the adopted gamma of 2.5, measured 0.254 over 440-500 nm. The check crosses
two panels, because gamma comes from the characteristic frame. It only works
on the RAW axis readings: normalising each curve to its own 400 nm value
removes exactly the separation being tested.

WHAT THE SHEET SETTLES ABOUT GRANULARITY, AND WHAT IT DOES NOT
----------------------------------------------------------------
Kodak prints "Diffuse RMS Granularity 8", read at a net diffuse visual density
of 1.0 through a 48-micrometre aperture. `GrainSpec.rms_granularity` is
defined in `film_profiles.py` as sigma(D)*1000 through a 48 um aperture AT
D = 1.0 -- the same condition, exactly. The stored 4.7 came from T-101's
relative ladder anchored on Monograph 54's absolute Wiener spectrum for HPS,
measured AT D 0.48 ABOVE BASE. So the two numbers are not in conflict; one is
at this field's own density and the other is not, and the manufacturer figure
is adopted for that reason and no other.

⚠⚠ THE 1.70 RATIO IS NOT PROPAGATED. It was measured on a gamma-2.5 POSITIVE,
where D 0.48 is still in the toe and D 1.0 is on the straight line. The other
five T-101 stocks are gamma-0.6 negatives on which both densities are on the
straight line, and the three of them whose net-1.0 granularity is published
elsewhere already land within 12 % of it. Applying 1.70 to them would invent
an effect. `verify.py`'s G-5302-LADDER-NOT-RESCALED fails the build if anyone
ever does.

WHAT THE SHEET DOES NOT SAY
-----------------------------
No grain diameter (so `grain_clump_um` keeps T-101's 0.589, an upper bound),
no specific gravity, no shrinkage, no reciprocity, no dye densities, and no
development TIME in the processing table.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import digitize_plot as DP            # noqa: E402
import film_profiles as FP            # noqa: E402

PDF = (Path(__file__).resolve().parent / "PDF" / "PROFILES" / "KODAK"
       / "EASTMAN Fine Grain Release 5302.pdf")

#: Page index of the plot page. Page 1 is the description, 2 the processing
#: and image-structure tables, 3 the three panels, 4 the Kodak addresses.
PLOT_PAGE = 2

# ---------------------------------------------------------------------------
# AXIS CALIBRATION
#
# Every ladder below is the CENTRE of each printed tick label, taken from the
# PDF's own text-word boxes. The fits are reported with their residuals and
# the module fails if any of them degrades, because a silently re-flowed axis
# is how a whole panel goes wrong without looking wrong.
#
# ⚠ THE CHARACTERISTIC ABSCISSA'S LEFTMOST LABEL IS "-1.0" AND ITS MINUS SIGN
# IS NOT TEXT. It is a drawn line at x 82.1-85.0, which is why the label ladder
# reads "1.0 0.0 1.0 2.0 3.0" and the first entry is assigned -1.0: the four
# gaps are 46.3, 45.4, 47.3 and 45.1 points, uniform to 2 %, so a ladder
# reading +1.0 there would have to be non-monotonic.
# ---------------------------------------------------------------------------
CHAR_D_PX = (51.10, 97.20, 142.75, 189.00, 235.95)
CHAR_D_VAL = (4.0, 3.0, 2.0, 1.0, 0.0)
CHAR_X_PX = (86.25, 132.55, 177.90, 225.20, 270.30)
CHAR_X_VAL = (-1.0, 0.0, 1.0, 2.0, 3.0)

INSET_G_PX = (104.25, 113.30, 123.55, 133.55, 144.30, 154.60, 164.80, 175.90)
INSET_G_VAL = (3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 2.0)
INSET_T_PX = (100.75, 111.45, 122.65, 133.55, 144.40)
INSET_T_VAL = (2.0, 4.0, 6.0, 8.0, 10.0)

MTF_F_PX = (362.95, 384.35, 397.35, 406.25, 413.35, 435.30, 457.25, 486.25,
            508.45, 530.15, 563.85)
MTF_F_VAL = (1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 600)
MTF_R_PX = (202.40, 183.35, 171.50, 157.20, 146.75, 135.55, 116.25, 104.25,
            89.65, 80.15, 68.75, 49.40)
MTF_R_VAL = (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200)

SPEC_Y_PX = (292.10, 329.90, 366.90, 404.20, 441.60)
SPEC_Y_VAL = (1.0, 0.0, -1.0, -2.0, -3.0)
SPEC_X_PX = (87.3, 107.3, 127.4, 147.5, 167.6, 187.6, 207.7, 227.75, 247.8,
             267.9, 287.55)
SPEC_X_VAL = (250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750)

#: Which vector drawing on the page holds what. Indices into
#: `page.get_drawings()`. ⚠ THEY ARE PINNED BECAUSE THE PANEL GEOMETRY IS, and
#: `_check_geometry` re-derives each one's bounding box and refuses a file
#: whose paths have moved -- so a different printing of the sheet fails loudly
#: instead of being traced against the wrong frame.
DRAW_CHAR_A, DRAW_CHAR_B = 4, 5        # 3 traces + 2 traces, width 0.72
DRAW_INSET = 2                         # Time-Gamma, first subpath
DRAW_SPEC = 23                         # 2 criteria, width 0.72
DRAW_MTF = 30                          # 1 curve, width 0.72

EXPECT_BOXES = {
    DRAW_CHAR_A: (114.2, 74.4, 206.4, 231.3),
    DRAW_CHAR_B: (114.2, 79.0, 234.0, 233.6),
    DRAW_INSET: (100.9, 108.9, 144.7, 184.5),
    DRAW_SPEC: (147.4, 317.2, 191.5, 401.3),
    DRAW_MTF: (390.5, 70.9, 521.0, 142.8),
}

#: Development times as Kodak letters them beside the five traces, steepest
#: (longest development) first -- which is the order `_traces` returns.
TIMES_MIN = (9.0, 7.0, 5.0, 3.5, 2.0)

#: Tolerances. Every one of these is a number the module FAILS on, not a
#: number it prints.
AXIS_RESID_TOL = 0.02        # decades / density units / log units
MTF_AXIS_RESID_TOL = 0.015   # log10 of the plotted quantity
FIT_RMS_TOL = 0.020          # density, on the adopted trace
SLOPE_TOL = 0.08             # fitted realised slope vs the drawn slope
INSET_AGREE_TOL = 0.25       # mean |drawn slope - inset gamma| over 5 traces
CRITERION_TOL = 0.06         # log, the 0.7/gamma spectral cross-check
BOX_TOL = 0.5                # points, on every pinned panel bounding box


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _subpaths(g, gap=0.35):
    """Split one PDF drawing into its disconnected polylines.

    ⚠ A SINGLE `s` DRAWING HOLDS SEVERAL CURVES ON THIS PAGE and nothing in
    the path data marks the boundary except the pen jumping: drawing 4 carries
    three of the five characteristic traces and drawing 5 the other two. Split
    on a discontinuity larger than a third of a point and the five come back
    separately.
    """
    out: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for it in g["items"]:
        if it[0] == "l":
            a, b = it[1], it[2]
            if cur and (abs(cur[-1][0] - a.x) > gap
                        or abs(cur[-1][1] - a.y) > gap):
                out.append(cur)
                cur = []
            if not cur:
                cur.append((a.x, a.y))
            cur.append((b.x, b.y))
        else:
            if cur:
                out.append(cur)
                cur = []
    if cur:
        out.append(cur)
    return out


def _fit(px, val):
    """Least-squares straight line through a tick ladder -> (m, c, worst)."""
    a = np.polyfit(np.asarray(px, float), np.asarray(val, float), 1)
    worst = max(abs(float(np.polyval(a, p)) - v) for p, v in zip(px, val))
    return float(a[0]), float(a[1]), worst


def _log_fit(px, val):
    """Same, for an axis whose LABELS are decades of the plotted quantity."""
    a = np.polyfit(np.asarray(px, float),
                   np.asarray([math.log10(v) for v in val], float), 1)
    worst = max(abs(float(np.polyval(a, p)) - math.log10(v))
                for p, v in zip(px, val))
    return float(a[0]), float(a[1]), worst


def _max_secant(pts, span=0.30, tol=0.015):
    """Steepest chord of `span` decades anywhere on a traced curve.

    ⚠ NOT A FITTED GAMMA AND DELIBERATELY SO. This is what the ARTIST DREW,
    measured without any model in between, and it is the quantity the inset is
    compared against. A fitted parameter could not do that job: on a
    six-parameter softplus difference the `gamma` field and the realised slope
    part company on steep traces, which is the whole reason this project
    checks realised slopes rather than stored ones.
    """
    best = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[j][0] - pts[i][0]
            if abs(dx - span) < tol:
                best = max(best, (pts[j][1] - pts[i][1]) / dx)
    return best


def _realised(p6, span=0.30, lo=-1.0, hi=2.6, n=3601):
    xs = np.linspace(lo, hi, n)
    d = DP.softplus_curve(xs, *p6)
    k = int(round(span / (xs[1] - xs[0])))
    return float(np.max((d[k:] - d[:-k]) / (xs[k:] - xs[:-k])))


def _fit_tied(x, d, init, pin):
    """Fit with shoulder_x FREE and shoulder_k TIED to toe_k.

    ⚠ THE TIE IS NOT A CONVENIENCE. `ToneCurve`'s own docstring proves that
    shoulder_k == toe_k is the ONLY setting under which the model is monotone
    everywhere; any inequality puts an extremum either below dmin or above
    dmax. A free six-parameter fit to this same trace reaches rms 0.0105
    against the tied fit's 0.0123 -- it buys 0.002 D by leaving the monotone
    set, and a print curve with a local maximum in it is not a print curve.

    `pin` is the trace's own lowest drawn point. The base+fog plateau is the
    best-measured thing on the panel and an unconstrained optimiser will trade
    it against a steeper gamma; the asymmetric window (0.02 down, 0.03 up) is
    the same one `fit_nine.py` arrived at after a contrast check caught the
    slack.
    """
    lo, hi = max(0.0, pin - 0.02), pin + 0.03

    def loss(p):
        dmin, g, tx, tk, sx = p
        if g <= 0 or tk <= 0.02 or sx <= tx:
            return 1e9
        pen = 0.0
        if dmin < lo:
            pen += 100.0 * (lo - dmin) ** 2
        elif dmin > hi:
            pen += 100.0 * (dmin - hi) ** 2
        r = DP.softplus_curve(x, dmin, g, tx, tk, sx, tk) - d
        return float(np.mean(r * r)) + pen

    p, _ = DP._nelder_mead(loss, np.asarray(init, dtype=np.float64),
                           [0.02, 0.03, 0.08, 0.04, 0.08])
    full = (float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]),
            float(p[3]))
    r = DP.softplus_curve(x, *full) - d
    return full, float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))


INITS = (
    (0.06, 2.5, 0.30, 0.25, 1.60), (0.06, 3.0, 0.50, 0.20, 1.50),
    (0.05, 2.2, 0.20, 0.30, 1.80), (0.08, 2.8, 0.60, 0.15, 1.40),
    (0.05, 3.5, 0.40, 0.12, 1.30), (0.06, 2.0, 0.10, 0.35, 2.00),
    (0.06, 3.2, 0.35, 0.18, 1.45), (0.05, 2.6, 0.25, 0.22, 1.70),
)


# ---------------------------------------------------------------------------
def run(do_assert: bool = True) -> int:
    fail: list[str] = []

    if not PDF.exists():
        print("[SKIP] kodak_h1_5302.py -- %s is not in the corpus" % PDF.name)
        return 0

    doc = pymupdf.open(PDF)
    page = doc[PLOT_PAGE]
    draw = page.get_drawings()

    # -- the panels have not moved ------------------------------------------
    for idx, want in EXPECT_BOXES.items():
        got = tuple(round(float(v), 1) for v in draw[idx]["rect"])
        if max(abs(a - b) for a, b in zip(got, want)) > BOX_TOL:
            fail.append("drawing %d bounding box %s, expected %s -- this is "
                        "not the printing the module was written against"
                        % (idx, got, want))
    if fail:
        print("\n".join(["FAILURES:"] + ["  " + f for f in fail]))
        return 1

    # -- axes ----------------------------------------------------------------
    mD, cD, rD = _fit(CHAR_D_PX, CHAR_D_VAL)
    mE, cE, rE = _fit(CHAR_X_PX, CHAR_X_VAL)
    mG, cG, rG = _fit(INSET_G_PX, INSET_G_VAL)
    mT, cT, rT = _fit(INSET_T_PX, INSET_T_VAL)
    mSy, cSy, rSy = _fit(SPEC_Y_PX, SPEC_Y_VAL)
    mSx, cSx, rSx = _fit(SPEC_X_PX, SPEC_X_VAL)
    mF, cF, rF = _log_fit(MTF_F_PX, MTF_F_VAL)
    mR, cR, rR = _log_fit(MTF_R_PX, MTF_R_VAL)

    for nm, r, tol in (("characteristic density", rD, AXIS_RESID_TOL),
                       ("characteristic log exposure", rE, AXIS_RESID_TOL),
                       ("inset gamma", rG, AXIS_RESID_TOL),
                       ("inset time", rT, 0.05),
                       ("spectral log sensitivity", rSy, AXIS_RESID_TOL),
                       ("MTF frequency", rF, MTF_AXIS_RESID_TOL),
                       ("MTF response", rR, MTF_AXIS_RESID_TOL)):
        if r > tol:
            fail.append("%s axis worst residual %.4f > %.4f" % (nm, r, tol))
    if rSx > 0.8:
        fail.append("spectral wavelength axis worst residual %.2f nm" % rSx)

    print("=== axis fits (worst residual against the printed ticks) ===")
    print("  characteristic   D %.4f   log H %.4f" % (rD, rE))
    print("  inset            gamma %.4f   minutes %.4f" % (rG, rT))
    print("  spectral         log S %.4f   nm %.2f" % (rSy, rSx))
    print("  MTF              log f %.4f   log R %.4f" % (rF, rR))

    # -- the five characteristic traces --------------------------------------
    traces = []
    for gi in (DRAW_CHAR_A, DRAW_CHAR_B):
        for s in _subpaths(draw[gi]):
            traces.append(sorted((mE * x + cE, mD * y + cD) for x, y in s))
    if len(traces) != 5:
        fail.append("expected 5 characteristic traces, split found %d"
                    % len(traces))
        print("\n".join(["FAILURES:"] + ["  " + f for f in fail]))
        return 1
    # Steepest first: the trace that reaches the panel's right-hand densities
    # soonest is the longest development. Nothing else distinguishes them.
    traces.sort(key=lambda p: p[-1][0])

    # -- the Time-Gamma inset, read with no reference to the traces ----------
    inset = sorted((mT * x + cT, mG * y + cG)
                   for x, y in _subpaths(draw[DRAW_INSET])[0])
    it = np.asarray([p[0] for p in inset])
    ig = np.asarray([p[1] for p in inset])

    def gamma_at(t):
        if t < it[0]:
            # ⚠ EXTRAPOLATION, AND ONLY HERE. The drawn inset begins at
            # 2.048 min and the leftmost characteristic trace is lettered
            # "2 min.", so the 2-minute gamma is a linear extension of the
            # inset's first eight samples over three seconds.
            a = np.polyfit(it[:8], ig[:8], 1)
            return float(np.polyval(a, t))
        return float(np.interp(t, it, ig))

    def time_at(g):
        return float(np.interp(g, ig, it))

    drawn = [_max_secant(p) for p in traces]
    inset_g = [gamma_at(t) for t in TIMES_MIN]
    agree = sum(abs(a - b) for a, b in zip(drawn, inset_g)) / 5.0
    alt_up = sum(abs(a - b) for a, b in zip(drawn[:-1], inset_g[1:])) / 4.0
    alt_dn = sum(abs(a - b) for a, b in zip(drawn[1:], inset_g[:-1])) / 4.0
    alt_rev = sum(abs(a - b)
                  for a, b in zip(drawn, inset_g[::-1])) / 5.0

    print("\n=== the five drawn traces against the Time-Gamma inset ===")
    print("  %-8s %10s %10s" % ("minutes", "drawn", "inset"))
    for t, a, b in zip(TIMES_MIN, drawn, inset_g):
        print("  %-8.1f %10.3f %10.3f" % (t, a, b))
    print("  mean |difference| %.3f;  one step later %.3f, one step earlier "
          "%.3f, reversed %.3f" % (agree, alt_up, alt_dn, alt_rev))
    if agree > INSET_AGREE_TOL:
        fail.append("trace-to-inset agreement %.3f > %.3f"
                    % (agree, INSET_AGREE_TOL))
    if not (agree < alt_up and agree < alt_dn and agree < alt_rev):
        fail.append("an ALTERNATIVE label assignment fits better than the "
                    "adopted one -- as-is %.3f, up %.3f, down %.3f, reversed "
                    "%.3f" % (agree, alt_up, alt_dn, alt_rev))

    # -- Kodak's own control band, and which trace falls inside it -----------
    band = (time_at(2.40), time_at(2.60))
    inside = [t for t in TIMES_MIN if band[0] <= t <= band[1]]
    print("\n=== the control gamma band the PROCESSING table names ===")
    print("  gamma 2.40 at %.2f min, 2.50 at %.2f min, 2.60 at %.2f min"
          % (band[0], time_at(2.50), band[1]))
    print("  drawn times inside the band: %s" % (inside,))
    if inside != [3.5]:
        fail.append("the 3 1/2-minute trace is no longer the unique drawn "
                    "trace inside Kodak's 2.4-2.6 control band; got %s"
                    % (inside,))

    # -- fit the adopted trace ------------------------------------------------
    adopted = traces[TIMES_MIN.index(3.5)]
    xs = np.asarray([p[0] for p in adopted])
    ds = np.asarray([p[1] for p in adopted])
    best = None
    for init in INITS:
        p, rms, mx = _fit_tied(xs, ds, init, float(ds.min()))
        if best is None or rms < best[1]:
            best = (p, rms, mx)
    par, rms, mx = best
    rl = _realised(par)
    print("\n=== the adopted 3 1/2-minute trace ===")
    print("  %d points, log H %.3f..%.3f, D %.3f..%.3f"
          % (len(xs), xs.min(), xs.max(), ds.min(), ds.max()))
    print("  ToneCurve(%.4f, %.4f, %.4f, %.4f, %.4f, %.4f)" % par)
    print("  rms %.4f D, worst %.4f D; realised 0.3-decade slope %.3f against "
          "the drawn %.3f" % (rms, mx, rl, drawn[TIMES_MIN.index(3.5)]))
    if rms > FIT_RMS_TOL:
        fail.append("adopted-trace fit rms %.4f > %.4f" % (rms, FIT_RMS_TOL))
    if abs(rl - drawn[TIMES_MIN.index(3.5)]) > SLOPE_TOL:
        fail.append("adopted fit realises slope %.3f against the drawn %.3f"
                    % (rl, drawn[TIMES_MIN.index(3.5)]))

    # -- the MTF ---------------------------------------------------------------
    mtf = sorted((10.0 ** (mF * x + cF), 10.0 ** (mR * y + cR))
                 for x, y in _subpaths(draw[DRAW_MTF])[0])

    def freq_at(resp):
        for a, b in zip(mtf, mtf[1:]):
            if (a[1] - resp) * (b[1] - resp) <= 0 and a[1] != b[1]:
                f = ((math.log10(resp) - math.log10(a[1]))
                     / (math.log10(b[1]) - math.log10(a[1])))
                return 10.0 ** (math.log10(a[0])
                                + f * (math.log10(b[0]) - math.log10(a[0])))
        return None

    def resp_at(freq):
        for a, b in zip(mtf, mtf[1:]):
            if a[0] <= freq <= b[0]:
                t = ((math.log10(freq) - math.log10(a[0]))
                     / (math.log10(b[0]) - math.log10(a[0])))
                return 10.0 ** (math.log10(a[1])
                                + t * (math.log10(b[1]) - math.log10(a[1])))
        return None

    f50 = freq_at(50.0)
    print("\n=== the modulation-transfer curve ===")
    print("  drawn from %.1f to %.0f cycle/mm, %.0f %% down to %.0f %%"
          % (mtf[0][0], mtf[-1][0], mtf[0][1], mtf[-1][1]))
    print("  " + ",  ".join("%g c/mm %.1f %%" % (f, resp_at(f))
                            for f in (5, 10, 20, 50, 100)))
    print("  80 %% at %.1f,  50 %% at %.2f,  30 %% at %.1f,  10 %% at %.1f "
          "cycle/mm" % (freq_at(80.0), f50, freq_at(30.0), freq_at(10.0)))

    # -- the spectral panel ----------------------------------------------------
    spec = []
    for s in _subpaths(draw[DRAW_SPEC]):
        spec.append(sorted((mSx * x + cSx, mSy * y + cSy) for x, y in s))
    if len(spec) != 2:
        fail.append("expected 2 spectral criteria, found %d" % len(spec))
    else:
        # The higher curve is the LOWER density criterion: less density needs
        # less exposure, hence more sensitivity. Kodak's own labels agree --
        # "D=0.3 Above gross fog" is lettered beside the upper one.
        spec.sort(key=lambda q: -q[0][1])
        grid = np.arange(400.0, 501.0, 10.0)
        raw = [np.interp(grid, [p[0] for p in q], [p[1] for p in q])
               for q in spec]
        gap = raw[0][4:] - raw[1][4:]          # 440 nm up
        pred = 0.7 / rl
        print("\n=== the spectral panel, two criteria ===")
        print("  D=0.3 raw at 400 nm %+.3f, D=1.0 raw at 400 nm %+.3f"
              % (raw[0][0], raw[1][0]))
        print("  separation over 440-500 nm mean %.3f log against the "
              "0.7/gamma = %.3f the characteristic panel predicts"
              % (float(gap.mean()), pred))
        if abs(float(gap.mean()) - pred) > CRITERION_TOL:
            fail.append("the two spectral criteria are %.3f apart where the "
                        "adopted gamma predicts %.3f"
                        % (float(gap.mean()), pred))
        norm = [r - r[0] for r in raw]
        print("  stored shape (D=0.3, normalised to 400 nm): %s"
              % [round(float(v), 3) for v in norm[0]])

    # -- against what the database actually holds ------------------------------
    ps = FP.get_print_stock("KODAK_5302")
    print("\n=== against the stored profile ===")
    checks = [
        ("curve", tuple(round(v, 4) for v in par),
         (ps.curves.g.dmin, ps.curves.g.gamma, ps.curves.g.toe_x,
          ps.curves.g.toe_k, ps.curves.g.shoulder_x, ps.curves.g.shoulder_k)),
        ("mtf_f50", round(f50, 1), ps.mtf_f50),
        ("resolving power low/high", (63.0, 125.0),
         (ps.resolving_power_lp_mm_lowc, ps.resolving_power_lp_mm_highc)),
        ("grain_rms", 8.0, ps.grain_rms),
    ]
    for nm, got, stored in checks:
        same = (np.allclose(np.asarray(got, float), np.asarray(stored, float),
                            atol=0.05) if isinstance(got, tuple)
                else abs(float(got) - float(stored)) < 0.05)
        print("  %-26s re-derived %s   stored %s   %s"
              % (nm, got, stored, "ok" if same else "MISMATCH"))
        if not same:
            fail.append("%s: re-derived %s, database holds %s"
                        % (nm, got, stored))

    fam = ps.processing_family.points
    if len(fam) != 5 or [p.minutes for p in fam] != sorted(TIMES_MIN):
        fail.append("the stored processing_family is not the five printed "
                    "times")
    else:
        worst = max(abs(p.gamma - gamma_at(p.minutes)) for p in fam)
        print("  %-26s 5 points, worst gamma drift from the inset %.4f"
              % ("processing_family", worst))
        if worst > 0.01:
            fail.append("a stored family gamma is %.4f off this module's own "
                        "reading of the inset" % worst)

    doc.close()
    if fail and do_assert:
        print("\n".join(["", "FAILURES:"] + ["  " + f for f in fail]))
        return 1
    print("\n[OK] kodak_h1_5302.py -- H-1-5302 page 3, three vector panels. "
          "The five drawn development times and the Time-Gamma inset are read "
          "INDEPENDENTLY and agree to %.3f of gamma, every alternative label "
          "assignment being worse and reversal worse by %.1fx. Kodak's own "
          "control band of 2.4-2.6 spans %.2f-%.2f min, inside which the "
          "3 1/2-minute trace is the ONLY drawn one -- adopted at rms %.4f D "
          "with a realised slope of %.3f against the drawn %.3f. "
          "⚠ mtf_f50 %.1f cycle/mm against the 85.0 ESTIMATE it replaces, a "
          "factor of %.1f and the largest single correction of the batch. "
          "⚠ The two spectral criteria check each other across panels: %.3f "
          "log apart where 0.7/gamma predicts %.3f. Granularity 8 at net "
          "density 1.0 is Kodak's, at this file's own defined condition, and "
          "the 1.70 ratio to the BBC reading at D 0.48 is NOT propagated to "
          "the ladder"
          % (agree, alt_rev / agree, band[0], band[1], rms, rl,
             drawn[TIMES_MIN.index(3.5)], f50, 85.0 / f50,
             float(gap.mean()), pred))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="")     # accepted, unused
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    default=True)
    ns = ap.parse_args(argv)
    return run(ns.do_assert)


if __name__ == "__main__":
    sys.exit(main())
