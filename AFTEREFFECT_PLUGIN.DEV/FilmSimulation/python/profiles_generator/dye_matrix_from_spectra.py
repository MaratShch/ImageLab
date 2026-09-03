"""Dye crosstalk derived from measured dye spectra through ISO 5-3 densitometry.

⚠⚠ READ THIS FIRST: THE RESULT IS CORRECT AND IS **NOT** ADOPTED.

The NINE matrices below describe their dyes accurately. They are the WRONG
QUANTITY for `dye_matrix`, and wiring them in double-counts crosstalk. The
module exists to derive them, to prove they are right, and to assert that they
stay out of the render path until the missing half arrives. `--assert` FAILS if
anyone adopts them. Why, in full, under WHY IT IS NOT ADOPTED below.

WHAT IT LOOKED LIKE IT WOULD REPLACE
------------------------------------
Every colour `dye_matrix` in the database is built by `_dye(k)` from ONE scalar
per stock, which makes it **symmetric by construction**:

    KODAK_VISION2_50D_5201    1.0667  -0.0333  -0.0333
    GEVACOLOR_NEG_682         1.0600  -0.0300  -0.0300

⚠ SYMMETRIC IS THE DEFECT, NOT THE MAGNITUDE. Real dye crosstalk is asymmetric
and always in the same direction: the CYAN dye has large unwanted green and blue
absorption, MAGENTA has large unwanted blue, and YELLOW has almost no unwanted
red. A symmetric matrix cannot express any of it, so today every colour stock
crosstalks the same way and differs only in how much. The schema's own
`SpectralDyeDensity` docstring names the consequence: "a matrix cannot express
an unwanted absorption that peaks off-band, which is exactly what makes
Gevacolor's 550 nm magenta look different from Agfacolor's 540 nm one."

Twelve stocks carry traced three-dye spectral density panels, and nothing on the
render path has ever read one.

WHY IT IS NOT ADOPTED
---------------------
⚠ THE CURVES ARE ALREADY STATUS DENSITIES, SO THE CROSSTALK IS ALREADY IN THEM.
`density_metric` says status M for 74 colour stocks and status A for 23. A
status density is what a densitometer reads from the WHOLE developed film --
unwanted absorptions included. The characteristic curves therefore already carry
the crosstalk, and multiplying those densities by a matrix built from the same
absorptions applies it a second time.

⚠ SO THE NEAR-IDENTITY MATRICES `_dye(k)` PRODUCES ARE STRUCTURALLY RIGHT AND
THIS TABLE IS STRUCTURALLY WRONG, however much better sourced it is. What stage
12 can legitimately hold is the difference between the reader's response and the
densitometry already baked into the curves:

    dye_matrix  =  M_reader . M_status^-1

which is near identity whenever the reader resembles the declared status -- and
near identity is exactly the shape the hand-set scalars produce. They are an
aesthetic stand-in for a real quantity; they are not a placeholder for this one.

⚠ `M_reader` ARRIVED ON 2026-08-31 (queue M1), AND THE ANSWER IT GIVES IS THE
ONE THIS MODULE ARGUED FOR. `KODAK_2383_RELEASE` now carries a traced spectral
sensitivity -- the first print stock in the database to have one -- so
`M_reader . M_status^-1` is computable for every panel here. It lands between
**0.048 and 0.116** of identity, against raw status off-diagonals reaching
**+0.24**: the correction actually owed is four to six times smaller than the
table that kept looking ready to wire in. Measured, not asserted, and pinned in
`EXPECTED_STAGE12`.

⚠ AND IT IS STILL NOT ADOPTED, FOR A NEW REASON THAT REPLACES THE OLD ONE. The
reader 2383 describes is a release PRINT FILM. 164 of the 165 profiles set
`default_print=SCAN_DI`, so their reader is a scanner; the one exception prints
on TECHNICOLOR_IB. **Not one stock in this database is rendered through 2383.**
Storing this matrix would state that a stock's reader is a film it is never
printed on, which is the same substitution refused above wearing better
sourcing. The gap has moved from "no reader response exists anywhere" to "no
stock renders through the reader we now have" -- from acquisition to
configuration, which is a much smaller thing and a differently-shaped one.

⚠ AND `verify.py` CAUGHT THE DOUBLE COUNT BEFORE THE REASONING DID. Adopting the
table made "Agfacolor Neu is much less saturated than a clean reversal stock"
fail -- measured Ektachrome 100D came out nearly as muddy as the 1936 stock that
exists in this file as the muddiness reference. That is what a double count
looks like from outside, and the guard turned a shipped defect into a caught one.

THE DERIVATION
--------------
For each dye j, solve for the AMOUNT `a_j` that produces exactly 1.00 density in
that dye's own analysing band, then read what that same amount produces in the
other two:

    T_j(lambda, a)   = 10 ** (-a * D_j(lambda))
    D_measured(i, a) = -log10( INT R_i T_j dlambda / INT R_i dlambda )
    a_j              : D_measured(j, a_j) = 1.00
    M[i][j]          = D_measured(i, a_j)

`R_i` is the ISO 5-3 status A or status M response, chosen per profile from its
own `density_metric`. The diagonal is 1.00 by construction and the off-diagonals
are "unwanted density at unit useful density" -- which is exactly what
`DyeImpurityRatio` is defined to hold, so this fills an existing representation
from a second kind of evidence rather than inventing one.

⚠ SOLVING FOR THE AMOUNT IS WHAT MAKES THE `peak_1.0` PANELS USABLE. Eight of
the twelve are normalised to unit peak, so their absolute levels are gone. Any
construction that needed those levels would be inventing them -- but `a_j` is
solved, not assumed, so an arbitrary scale on the stored curve is absorbed and
divides out. The remaining four panels, which do carry absolute levels, give the
same matrix either way; that is asserted below.

⚠ AND THE INTEGRAL IS OVER TRANSMITTANCE, NOT DENSITY. A densitometer averages
light and then takes a logarithm; it does not average logarithms. The two differ
whenever the dye's density varies across the band, which for an unwanted
absorption on a steep flank is exactly the case -- averaging density instead
overstates the unwanted term, and does so most where the number matters.

TWO EARLIER ATTEMPTS, BOTH REFUSED
----------------------------------
⚠ **The project's own spectral basis.** Tried first, because using one definition
of "R, G and B" everywhere is worth a lot. Refused: its Gaussian lobes are
sigma = 55 nm, half-power width about 130 nm, against 20-35 nm for the ISO
status responses. The derived matrix came out with a diagonal of 0.58 -- a
reader taking 42 % of its red density from magenta and yellow. That is not a
densitometer, and it would have made colour worse while looking principled.

⚠ **Sampling at each dye's peak.** A point estimate, ratio-based, needing no
standard at all. It was the plan while ISO 5-3 was believed absent from the
corpus; it survives here as `peak_ratio_matrix` and is used only as a
cross-check, because a point sample ignores band shape and over-weights a narrow
unwanted spike that happens to land on another dye's peak.

THE CROSS-CHECK, WHICH IS THE REASON TO TRUST ANY OF IT
-------------------------------------------------------
⚠ AND A SECOND, STRONGER CHECK ARRIVED 2026-08-30 (queue M2b): the sheets' own
MIDSCALE NEUTRAL, which the extractor had been discarding. `Neutral - Dmin =
k(C+M+Y)` must hold with the three k EQUAL, and the coefficients are free, so a
small spread is evidence. It refused three panels on first run -- including
EASTMAN_EXR_200T_5293, which had passed the sign test, the ratio bounds AND the
Soviet cross-check below and had already been adopted. Nine adopted, not twelve.

Four Soviet stocks carry `DyeImpurity` ratios read off MANUFACTURING
SPECIFICATIONS -- a different manufacturer, a different era, and a document
rather than a traced plot. Nothing here is fitted to them, so they are an
independent test of magnitude:

    magenta -> blue     Soviet specification   0.15 .. 0.25

The early emulsions in this set are expected to land in that band and the modern
Kodak cine stocks well below it. That is forty years of dye chemistry, and it is
asserted rather than admired.

UNIT ROW SUMS ARE PRESERVED
---------------------------
⚠ The raw derived matrix has row sums of 1.02 to 1.30 and shipping those would
break a contract that is right. `_dye()`'s docstring and a `verify.py` check both
require unit row sums, because a row sum away from 1 shifts neutral DENSITY as
well as colour: the anchor solve then has to undo the density half, and a
stock's black level ends up depending on its saturation setting. The row sum is
a per-channel density gain and belongs to the curves and to dmin, not here.

Rows are therefore normalised to sum 1. ⚠ THAT IS NOT A REDUCTION IN MAGNITUDE:
the measured crosstalk survives in full in the only axis the matrix is allowed
to act on.

Run:
    python dye_matrix_from_spectra.py            # the derived matrices
    python dye_matrix_from_spectra.py --assert   # non-zero exit on drift
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import film_profiles as fp
from film_profiles import FILM_PROFILES, PRINT_STOCKS
import iso_5_3_status as iso

#: Integration grid. 1 nm over the union of what the status responses and the
#: traced panels cover; the responses are the wider of the two.
GRID = np.arange(380.0, 780.0 + 1e-9, 1.0)

#: Row order is the analysing band, column order is the dye. Both are the
#: renderer's red-green-blue / cyan-magenta-yellow order, and cyan is read in
#: red, magenta in green, yellow in blue.
BAND_OF_ROW = ("r", "g", "b")
DYE_OF_COL = ("c", "m", "y")

#: Where each dye's peak must fall. Wide -- these catch a mis-assigned layer,
#: not a measurement. A "cyan" trace peaking at 540 nm is a magenta trace under
#: the wrong name, the likeliest defect in a hand-assigned three-curve panel.
PEAK_WINDOW_NM = {"c": (620.0, 720.0), "m": (510.0, 580.0), "y": (420.0, 470.0)}

#: The useful density each dye's amount is solved to.
UNIT_DENSITY = 1.0

#: Soviet manufacturing-specification bands, from the four `DyeImpurity` sets
#: already in the database. Reported against every derived stock, as
#: information: they are one manufacturer's tolerances, not a law of dyes, and a
#: stock outside them is a thing to look at rather than a thing that is wrong.
SOVIET_BANDS = {("m", "b"): (0.15, 0.25), ("c", "g"): (0.00, 0.10),
                ("y", "g"): (0.05, 0.18), ("y", "r"): (0.00, 0.06),
                ("m", "r"): (0.00, 0.15), ("c", "b"): (0.00, 0.10)}

#: Stocks whose panel predates modern dye chemistry.
#: ⚠ 2026-09-02, queue G2: GEVACHROME_600 and GEVACHROME_605 join the early set,
#: and they are the cleanest members of it. They are 1968 Agfa-Gevaert reversal
#: stock -- the same maker and the same decade as GEVACOLOR_NEG_682, which was
#: already here -- and their derived magenta-into-blue is 0.2232, inside the
#: Soviet specification band 0.15-0.25 without being fitted to it. ⚠ CLASSING
#: THEM AS "later" ON THE FIRST PASS DROPPED THE EARLY/LATER RATIO TO 1.37 AND
#: FIRED THIS AUDIT, which is the guard working: a 1968 emulsion counted as
#: modern makes forty years of dye chemistry look like twenty.
OLD_STOCKS = ("GEVACOLOR_NEG_682", "KODAK_EKTACHROME_100D_5285",
              "EASTMAN_EKTACHROME_7239", "GEVACHROME_600", "GEVACHROME_605")

#: ⚠ THE HISTORICAL CLAIM, AND IT IS DELIBERATELY A CLAIM ABOUT THE SET RATHER
#: THAN ABOUT ANY ONE STOCK. An earlier version asserted "every modern stock
#: has magenta-into-blue under 0.12" and fired on EXR 200T (0.141) and VISION2
#: 250D (0.150). That threshold was invented here, not documented anywhere, and
#: neither stock is modern in dye terms -- EXR is 1989. What the evidence
#: actually supports is that the early emulsions sit in the specification band
#: and that the set as a whole improved, so that is what is pinned.
OLD_IN_BAND_TOL = (0.7, 1.2)
OLD_OVER_MODERN_MIN = 1.4

#: ⚠ REFUSED, BY NAME, WITH THE REASON KEPT. Not skipped quietly and not
#: averaged in. A derivation that contradicts every other stock AND all four
#: manufacturing specifications is evidence about the SOURCE PANEL, and the
#: useful thing to do with it is record it where the next reader will find it.
#: ⚠ BOTH REFUSALS WERE TAKEN BACK TO THE SOURCE ON 2026-08-30 (queue M2) AND
#: NEITHER IS AN EXTRACTION BUG. The panels were re-opened at the PDF level:
#:
#:   * Each sheet's dye panel is FIVE separate stroke paths, not a raster to be
#:     tracked, so there is no crossing to get wrong. Peak positions assign them
#:     unambiguously -- on 5218, at 380 / 446 / 537 / 680 nm plus the neutral.
#:   * The stored arrays reproduce those paths (5218 magenta 0.302 stored
#:     against 0.276 read straight off the path at 640 nm). `dye_density.py`
#:     is faithful to what it traced.
#:   * 5218's panel is internally consistent: least squares gives
#:     neutral = 0.478 C + 0.556 M + 0.693 Y + 0.949 Dmin at relative rms
#:     0.0061. ⚠ But that identity has four free parameters and will absorb a
#:     contaminated dye, so it is NOT evidence the magenta is right.
#:
#: What disqualifies 5218 is physics, not peer comparison: its traced magenta
#: sits at 0.30 of peak across 640-700 nm and RISES from 0.302 to 0.310 over
#: that span. An absorption band decays away from its peak. Whatever that path
#: is past 620 nm, it is not a magenta dye.
EXPECTED_REFUSALS = {
    "EASTMAN_EXR_50D_5245":
        "cyan reads 0.009 into green while yellow reads 0.030 into red -- "
        "inverted against all eleven other panels and all four Soviet "
        "specifications, which agree cyan-into-green is the large unwanted "
        "term. Its yellow-into-green, 0.157, is also the highest in the set. "
        "The three peaks land in the right windows so the layers are not "
        "misnamed; the traced cyan and yellow shapes need re-reading",
    "KODAK_VISION2_500T_5218":
        "magenta reads 0.363 into red -- more than twice any other panel and "
        "well outside the Soviet specification's 0.00-0.15, and its "
        "yellow-into-green, 0.223, is likewise the highest in the set. Two "
        "unwanted terms both extreme in the same panel is a baseline problem "
        "at the red end of the trace rather than an unusual emulsion. "
        "⚠ CONFIRMED INDEPENDENTLY 2026-08-30 once the sheet's own neutral was "
        "kept: Neutral - Dmin resolves into 0.484 / 0.546 / 0.657 of the three "
        "dyes, a 31 % spread where a neutral demands equality",
    # ⚠ ADDED 2026-08-30 AND IT HAD PASSED EVERYTHING ELSE. 5293 satisfies the
    # sign pattern, the ratio bounds and the Soviet cross-check, and it was one
    # of the ten adopted into `_MEASURED_DYE_MATRIX` on the first pass. Its own
    # midscale neutral refuses it. That is the whole argument for keeping the
    # traces the extractor had been discarding: this defect was invisible to
    # every test that looked only at the three dye curves.
    "EASTMAN_EXR_200T_5293":
        "its midscale neutral resolves into 0.500 / 0.559 / 0.620 of the three "
        "dyes -- a 21 % spread where a visual neutral demands equality, so one "
        "of the three traces carries something that is not its dye. Passed the "
        "sign test, the ratio bounds and the Soviet cross-check; only the "
        "sheet's own neutral catches it",
    # ⚠ ADDED 2026-09-01d, AND THE REFUSAL IS THE EXPECTED CONSEQUENCE OF A
    # CAVEAT ALREADY RECORDED ON THE SOURCE rather than a surprise. Every other
    # entry here is a datasheet panel with a calibrated ordinate. Technicolor's
    # is not: Flueckiger et al. 2018 Fig. 16 has NO ordinate scale, no ticks and
    # no label, so the stored curves are peak-normalised WITH THE BOTTOM AXIS
    # ASSUMED TO BE ZERO ABSORBANCE. Any real baseline above that axis inflates
    # every off-band term, and the off-band terms are exactly what a dye matrix
    # is made of. The trace's own minima sit at 0.09-0.16 of peak, which is the
    # size of the effect. So the matrix is refused while the CURVES are kept:
    # the peak positions, which is what the source validates (460/540/660/720 nm
    # against its own printed list), are unaffected by a baseline offset, and
    # the cross-talk is not.
    "TECHNICOLOR_THREE_STRIP":
        "cyan reads 0.4298 into green, outside the admissible -0.06..0.30. "
        "⚠ NOT A BAD TRACE -- A MISSING ORDINATE. This is the only set here "
        "that is not off a calibrated datasheet panel: Flueckiger et al. 2018 "
        "Fig. 16 carries no ordinate scale, so the curves are stored "
        "peak-normalised with the axis assumed to be zero absorbance, and an "
        "unknown baseline inflates precisely the off-band terms a dye matrix "
        "is built from. The curves stay (their peaks are validated against the "
        "report's own printed peak list); the derived matrix does not. What "
        "would lift the refusal is a Technicolor dye measurement with a stated "
        "absorbance scale",
}

#: Physically admissible range for an unwanted-absorption ratio. Slightly
#: negative is legal -- `DyeImpurityRatio`'s own docstring records LN-8's
#: printed "minus 0.05-0.10" as a real interlayer effect.
RATIO_RANGE = (-0.06, 0.30)

#: Allowed spread in the three coefficients of `Neutral - Dmin = k(C+M+Y)`.
#: 0.15 is family C's own figure, adopted unchanged: on 5201 the unconstrained
#: solution comes out 0.628 / 0.604 / 0.595, a 5 % spread on numbers free to be
#: anything at all.
NEUTRAL_SPREAD_MAX = 0.15


def dye_curves(p):
    """The three traced dyes resampled onto GRID, holding the end values.

    ⚠ HOLDING RATHER THAN ZEROING, AND IT IS NOT NEUTRAL. The panels stop at
    700 nm; status M red runs to 770. Cyan still absorbs strongly out there, so
    zeroing would invent a transparent dye exactly where the red channel is
    still looking and would understate every cyan term. Holding the last value
    overstates it slightly instead -- the safer direction, and small: the status
    M red response above 700 nm is already down to about 1 % of its peak.
    """
    d = p.dye_density
    n = len(d.d_cyan)
    lam = d.lambda_start_nm + d.lambda_step_nm * np.arange(n)
    out = []
    for arr in (d.d_cyan, d.d_magenta, d.d_yellow):
        y = np.asarray(arr, dtype=np.float64)
        out.append(np.interp(GRID, lam, y, left=y[0], right=y[-1]))
    return lam, out


def band_density(resp, dye, amount):
    """Density this band reads from `amount` of this dye. Light, then log."""
    t = np.power(10.0, -amount * dye)
    return -np.log10(max(float(np.trapezoid(resp * t, GRID) /
                               np.trapezoid(resp, GRID)), 1e-300))


def solve_amount(resp, dye, target=UNIT_DENSITY):
    """The amount giving `target` density in this dye's own band. Bisection.

    Monotone in `amount`, so bisection cannot land on the wrong root; and it
    needs no derivative of an integral that is only defined numerically.
    """
    lo, hi = 1e-9, 1.0
    for _ in range(200):
        if band_density(resp, dye, hi) >= target:
            break
        hi *= 2.0
        if hi > 1e9:
            return None
    else:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if band_density(resp, dye, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def status_of(p) -> str:
    """status A or status M, from the profile's own declaration."""
    dm = (p.density_metric or "").strip().lower()
    return dm if dm in iso.STATUS else ""


def raw_matrix(p, cols):
    """M[i][j] = density band i reads from unit-own-density of dye j."""
    st = status_of(p)
    if not st:
        return None, "density_metric %r is neither status_a nor status_m" % (
            p.density_metric,)
    resp = {b: iso.response(st, b, GRID) for b in BAND_OF_ROW}
    m = np.zeros((3, 3))
    for j in range(3):
        own = resp[BAND_OF_ROW[j]]
        a = solve_amount(own, cols[j])
        if a is None:
            return None, "%s dye reaches no useful density in its own band" % (
                DYE_OF_COL[j],)
        for i in range(3):
            m[i, j] = band_density(resp[BAND_OF_ROW[i]], cols[j], a)
    return m, None


def peak_ratio_matrix(lam, cols):
    """The older point-sample estimate, kept only to cross-check the integral."""
    pk = [float(lam[int(np.argmax(np.interp(lam, GRID, c)))]) for c in cols]
    m = np.eye(3)
    for i in range(3):
        for j in range(3):
            own = float(np.interp(pk[j], GRID, cols[j]))
            if own <= 0:
                return None, pk
            m[i, j] = float(np.interp(pk[i], GRID, cols[j])) / own
    return m, pk


def normalise_rows(m):
    s = m.sum(axis=1, keepdims=True)
    if np.any(np.abs(s) < 1e-9):
        return None
    return m / s


#: The print stock whose spectral sensitivity supplies `M_reader` (queue M1).
PRINT_READER = "KODAK_2383_RELEASE"

#: max |M_reader . M_status^-1 - I| per stock, recorded 2026-08-31 (queue M1).
#: ⚠ THESE NUMBERS ARE WHAT THE MODULE WAS WAITING FOR AND THEY VINDICATE ITS
#: REFUSAL. The raw status matrices carry off-diagonals up to +0.24; the
#: quantity stage 12 may legitimately hold lands between 0.048 and 0.116 of
#: identity -- FOUR TO SIX TIMES SMALLER. Adopting the raw table would have
#: applied crosstalk already present in the status curves, at several times the
#: strength of the correction actually owed.
EXPECTED_STAGE12 = {
    # ⚠ GEVAERT, 2026-09-02 (queue G2). ONE VALUE FOR TWO STOCKS, because Bild 4
    # draws one dye set for both Typ 6.00 and Typ 6.05 -- the same one-drawing
    # case as RSX II 50 / 100 below. At 0.0903 they are the HIGHEST in this
    # table, which is what a 1968 dye set read through a modern release print
    # should be.
    "GEVACHROME_600": 0.0903,
    "GEVACHROME_605": 0.0903,
    # ⚠ THE THREE AGFA ENTRIES ADDED 2026-09-01 WERE THE HIGHEST IN THIS TABLE
    # AFTER 5218 until the two Gevachrome rows above overtook them, and RSX II 50 and RSX II 100 agree to 2e-4 because Agfa drew
    # ONE spectral-density panel for both films -- one measurement, not two.
    "AGFA_RSX_II_50": 0.0762,
    "AGFA_RSX_II_100": 0.0764,
    "AGFA_RSX_II_200": 0.0811,
    "EASTMAN_EKTACHROME_7239": 0.0694,
    "EASTMAN_EXR_50D_5245": 0.0781,
    "EASTMAN_EXR_200T_5293": 0.0681,
    "GEVACOLOR_NEG_682": 0.0496,
    "KODAK_EKTACHROME_100D_5285": 0.0731,
    "KODAK_VISION2_50D_5201": 0.0511,
    "KODAK_VISION2_200T_5217": 0.0509,
    "KODAK_VISION2_250D_5205": 0.0481,
    "KODAK_VISION2_500T_5218": 0.1164,
    "KODAK_VISION3_500T_5219": 0.0521,
    "KODAK_VISION_200T_5274": 0.0844,
    "KODAK_VISION_500T_5279": 0.0519,
}

#: How far a pinned stage-12 offset may move before this is called drift.
STAGE12_TOL = 0.004


def reader_response():
    """{band: linear sensitivity on GRID} for the print stock, or None.

    ⚠ ZERO OUTSIDE THE TRACED EXTENT, WHICH IS THE OPPOSITE OF WHAT
    `dye_curves` DOES, and both are right. A dye that stops being plotted is
    still absorbing, so its curve is HELD; a sensitivity that stops being
    plotted has fallen off the bottom of the panel, and a layer that is not
    sensitive there contributes nothing. Holding a sensitivity would invent a
    band edge that runs on forever.
    """
    for s in PRINT_STOCKS:
        if s.name != PRINT_READER or not s.spectral.has_data:
            continue
        sp = s.spectral
        lam = sp.lambda_start_nm + sp.lambda_step_nm * np.arange(len(sp.log_s_r))
        out = {}
        for band, arr in (("r", sp.log_s_r), ("g", sp.log_s_g),
                          ("b", sp.log_s_b)):
            v = np.asarray(arr, dtype=np.float64)
            lin = np.where(v > -3.99, np.power(10.0, v), 0.0)
            out[band] = np.interp(GRID, lam, lin, left=0.0, right=0.0)
        return out
    return None


def stage12_matrix(p, cols, resp):
    """`M_reader . M_status^-1` -- what stage 12 may legitimately hold.

    ⚠ AND HAVING IT IS NOT PERMISSION TO STORE IT. The reader this describes is
    a release PRINT FILM; 164 of the 165 profiles render through SCAN_DI, whose
    reader is a scanner. Putting a print emulsion's response where a scanner's
    belongs is the same class of substitution this module already refused once.
    """
    status, err = raw_matrix(p, cols)
    if status is None:
        return None, err
    m = np.zeros((3, 3))
    for j in range(3):
        own = resp[BAND_OF_ROW[j]]
        a = solve_amount(own, cols[j])
        if a is None:
            return None, ("%s dye reaches no useful density in the print "
                          "stock's %s band" % (DYE_OF_COL[j], BAND_OF_ROW[j]))
        for i in range(3):
            m[i, j] = band_density(resp[BAND_OF_ROW[i]], cols[j], a)
    try:
        return m @ np.linalg.inv(status), None
    except np.linalg.LinAlgError:
        return None, "the status matrix is singular"


def derive(p):
    """(unit-row-sum matrix, ratios, diagnostics) or (None, None, reason)."""
    if not p.dye_density.has_data:
        return None, None, "no three-dye panel"
    lam, cols = dye_curves(p)
    _pm, pk = peak_ratio_matrix(lam, cols)
    for j, dye in enumerate(DYE_OF_COL):
        w = PEAK_WINDOW_NM[dye]
        if not (w[0] <= pk[j] <= w[1]):
            return None, None, ("%s peaks at %.0f nm, outside %.0f-%.0f -- the "
                                "layer assignment is wrong"
                                % (dye, pk[j], w[0], w[1]))
    raw, err = raw_matrix(p, cols)
    if raw is None:
        return None, None, err
    m = normalise_rows(raw)
    if m is None:
        return None, None, "a row sums to zero"
    ratios = {(DYE_OF_COL[j], BAND_OF_ROW[i]): float(raw[i, j])
              for i in range(3) for j in range(3) if i != j}

    # ⚠ THE SIGN PATTERN IS A GATE, NOT A WARNING. Every real dye set agrees
    # that cyan-into-green is a large unwanted term and yellow-into-red is the
    # smallest thing on the panel. A panel that reverses them produces a matrix
    # which is wrong and renders perfectly plausibly, so it is refused here
    # rather than reported and then used anyway.
    if not ratios[("c", "g")] > ratios[("y", "r")]:
        return None, None, ("cyan reads %.4f into green against yellow %.4f "
                            "into red -- inverted, so the panel is suspect"
                            % (ratios[("c", "g")], ratios[("y", "r")]))
    # ⚠ THE SHEET'S OWN NEUTRAL IS THE STRONGEST TEST THERE IS, AND IT ONLY
    # BECAME AVAILABLE ON 2026-08-30 WHEN THE DISCARDED TRACES WERE KEPT.
    # `Neutral - Dmin = k_c*C + k_m*M + k_y*Y` must hold with the three k EQUAL,
    # because equal contribution is what makes the result a visual NEUTRAL.
    # Nothing here is fitted to produce that: the coefficients are free, so a
    # spread near zero is evidence and a large one means one of the dye traces
    # is carrying something that is not its dye.
    nd = p.dye_density
    if nd.d_neutral and nd.d_dmin:
        nn = np.asarray(nd.d_neutral, float) - np.asarray(nd.d_dmin, float)
        A = np.stack([np.asarray(nd.d_cyan, float),
                      np.asarray(nd.d_magenta, float),
                      np.asarray(nd.d_yellow, float)], 1)
        kk, _r, _rk, _sv = np.linalg.lstsq(A, nn, rcond=None)
        spread = float((kk.max() - kk.min()) / max(kk.mean(), 1e-9))
        if kk.min() <= 0.0 or spread > NEUTRAL_SPREAD_MAX:
            return None, None, ("its own midscale neutral does not resolve "
                                "into equal parts of the three dyes: "
                                "k = %s, spread %.3f against %.2f allowed"
                                % (" / ".join("%.3f" % v for v in kk), spread,
                                   NEUTRAL_SPREAD_MAX))

    for k, v in sorted(ratios.items()):
        if not (RATIO_RANGE[0] <= v <= RATIO_RANGE[1]):
            return None, None, ("%s reads %.4f into %s, outside the admissible "
                                "%.2f..%.2f -- no real dye is a third as dense "
                                "outside its own band as inside it"
                                % (k[0], v, k[1], *RATIO_RANGE))
    return m, ratios, dict(peaks=pk, raw=raw, peak_est=_pm,
                           status=status_of(p), row_sums=raw.sum(axis=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="accepted and unused; this "
                    "reads the database, not the corpus")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    args = ap.parse_args(argv)

    if iso.self_check():
        print("[!] the ISO 5-3 tables do not self-check; refusing to derive")
        return 1

    have = [p for p in FILM_PROFILES if p.dye_density.has_data]
    bad = 0
    done, refused = [], []

    print("dye_matrix derived from measured dye spectra through ISO 5-3")
    print("  %d stocks carry a three-dye panel" % len(have))
    print("")
    for p in have:
        m, ratios, diag = derive(p)
        if m is None:
            refused.append((p.name, diag))
            continue
        done.append((p, m, ratios, diag))
        pk = diag["peaks"]
        print("  %-28s %s  peaks %.0f/%.0f/%.0f nm   was %+.4f off-diagonal"
              % (p.name, diag["status"], pk[0], pk[1], pk[2],
                 p.dye_matrix[0][1]))
        for r in m:
            print("        " + "  ".join("%8.4f" % v for v in r))

    for name, why in refused:
        print("  %-28s REFUSED  %s" % (name, why))

    if not done:
        print("[!] nothing derived")
        return 1

    print("")
    print("  %-28s %-8s %-8s %-8s %-8s" % ("", "c->g", "m->b", "y->g", "y->r"))
    outside = 0
    for p, _m, ratios, _d in done:
        print("  %-28s %8.4f %8.4f %8.4f %8.4f"
              % (p.name, ratios[("c", "g")], ratios[("m", "b")],
                 ratios[("y", "g")], ratios[("y", "r")]))
        for k, v in ratios.items():
            if not (RATIO_RANGE[0] <= v <= RATIO_RANGE[1]):
                print("[!] %s: %s into %s is %.3f, outside the admissible "
                      "%.2f..%.2f" % (p.name, k[0], k[1], v, *RATIO_RANGE))
                bad += 1
            lo, hi = SOVIET_BANDS[k]
            if not (lo - 0.02 <= v <= hi + 0.02):
                outside += 1

    # ⚠ THE REFUSAL SET IS PINNED. A newly broken panel must announce itself,
    # and a panel that quietly starts passing must too -- otherwise "12 stocks
    # carry dye spectra" slowly stops meaning "12 stocks are used".
    got = {n for n, _w in refused}
    if got != set(EXPECTED_REFUSALS):
        print("[!] refusals changed: expected %s, got %s"
              % (sorted(EXPECTED_REFUSALS), sorted(got)))
        bad += 1

    # ⚠ AGAINST A SOURCE THE DERIVATION NEVER SAW: four Soviet manufacturing
    # specifications. The claim is about the SET -- early emulsions inside the
    # band, and the set as a whole improving -- because that is what the
    # evidence supports. See OLD_OVER_MODERN_MIN.
    print("")
    print("  magenta into blue, against the Soviet specification band "
          "%.2f-%.2f:" % SOVIET_BANDS[("m", "b")])
    lo, hi = SOVIET_BANDS[("m", "b")]
    old_v, mod_v = [], []
    for p, _m, ratios, _d in done:
        v = ratios[("m", "b")]
        old = p.name in OLD_STOCKS
        (old_v if old else mod_v).append(v)
        print("    %-28s %s %.4f" % (p.name, "early " if old else "later ", v))
        if old and not (lo * OLD_IN_BAND_TOL[0] <= v <= hi * OLD_IN_BAND_TOL[1]):
            print("[!] %s is an early emulsion but its magenta-into-blue %.3f "
                  "falls outside the specification band" % (p.name, v))
            bad += 1
    if old_v and mod_v:
        ratio = float(np.mean(old_v) / max(np.mean(mod_v), 1e-9))
        print("    early mean %.4f, later mean %.4f, ratio %.2f"
              % (np.mean(old_v), np.mean(mod_v), ratio))
        if ratio < OLD_OVER_MODERN_MIN:
            print("[!] the early emulsions are only %.2fx the later ones, "
                  "under %.2f -- forty years of dye chemistry should show"
                  % (ratio, OLD_OVER_MODERN_MIN))
            bad += 1
    print("    %d of %d derived ratios fall outside a Soviet band (one "
          "manufacturer's tolerances, reported not enforced)"
          % (outside, 6 * len(done)))

    for p, m, _r, _d in done:
        for i, row in enumerate(m):
            if abs(row.sum() - 1.0) > 1e-9:
                print("[!] %s row %d sums to %.9f, not 1"
                      % (p.name, i, row.sum()))
                bad += 1

    # ⚠ AND THE DATABASE MUST STILL HOLD WHAT THIS DERIVES. `_MEASURED_DYE_MATRIX`
    # is a literal table, which is this project's convention for traced numbers
    # -- the database must not depend on a derivation running at import time.
    # The cost of that convention is that a literal can drift from its source
    # by a hand edit, a bad merge, or a regeneration nobody re-ran. This is the
    # check that makes the convention safe, and it is the whole reason the
    # derivation is registered as an audit rather than run once and forgotten.
    print("")
    worst, worst_at = 0.0, None
    for p, m, _r, _d in done:
        stored = np.asarray(fp._MEASURED_DYE_MATRIX[p.name], dtype=np.float64)
        e = float(np.abs(stored - m).max())
        if e > worst:
            worst, worst_at = e, p.name
        if e > 1e-5:
            print("[!] %s: the stored table differs from the derivation by "
                  "%.2e -- regenerate _MEASURED_DYE_MATRIX" % (p.name, e))
            bad += 1
    print("  table against derivation: worst %.2e at %s" % (worst, worst_at))

    if set(fp._MEASURED_DYE_MATRIX) != {p.name for p, _m, _r, _d in done}:
        print("[!] _MEASURED_DYE_MATRIX holds %s, the derivation produces %s"
              % (sorted(fp._MEASURED_DYE_MATRIX),
                 sorted(p.name for p, _m, _r, _d in done)))
        bad += 1

    # ⚠⚠ AND THE TABLE MUST STILL NOT BE IN USE. This is the load-bearing
    # assertion of the whole module, and it asserts a NEGATIVE on purpose.
    #
    # These matrices describe the dyes correctly and are the wrong quantity for
    # `dye_matrix`: the stored characteristic curves are already status M or
    # status A densities, so the unwanted absorptions are in them ALREADY, and
    # multiplying by a matrix built from the same absorptions counts them twice.
    # What stage 12 may legitimately hold is `M_reader . M_status^-1`, which is
    # near identity and needs a reader response nothing in this corpus supplies.
    #
    # The danger is not that someone disagrees -- it is that this table looks
    # exactly like something ready to wire in, is better sourced than what it
    # would replace, and is one line away from being adopted by a reader in a
    # hurry. So the refusal is enforced rather than merely written down.
    # ---- queue M1, 2026-08-31: the missing half arrived ---------------------
    # ⚠ THIS SECTION EXISTS BECAUSE THE GAP THIS MODULE NAMED IS NOW CLOSED,
    # AND CLOSING IT DID NOT LICENCE THE ADOPTION. `KODAK_2383_RELEASE` carries
    # a traced spectral sensitivity as of today, so `M_reader . M_status^-1`
    # can be computed for every panel here. It comes out between 0.048 and
    # 0.116 of identity, against raw status off-diagonals reaching +0.24 --
    # which is the module's own argument, measured rather than asserted: the
    # correction actually owed is four to six times smaller than the table that
    # kept looking ready to wire in.
    #
    # ⚠ AND IT STILL MUST NOT BE STORED, FOR A DIFFERENT REASON THAN BEFORE.
    # The reader it describes is a release PRINT FILM. 164 of the 165 profiles
    # set `default_print=SCAN_DI`, so their reader is a scanner; the one
    # exception prints on TECHNICOLOR_IB. NOT ONE STOCK IN THIS DATABASE IS
    # RENDERED THROUGH 2383. Storing this matrix would state that a stock's
    # reader is a film it is never printed on -- the same substitution refused
    # above, wearing better sourcing. What M1 still needs is the SCANNER
    # response, or a profile that actually renders through a print stock.
    resp = reader_response()
    if resp is None:
        print("[!] %s carries no spectral sensitivity; the reader half of the "
              "derivation is unavailable" % PRINT_READER)
        bad += 1
    else:
        print("")
        print("  M_reader . M_status^-1 -- the quantity stage 12 may hold, "
              "with %s as the reader" % PRINT_READER)
        for p, _m, _r, _d in done:
            lam, cols = dye_curves(p)
            s12, err = stage12_matrix(p, cols, resp)
            if s12 is None:
                print("  %-28s REFUSED  %s" % (p.name, err))
                bad += 1
                continue
            off = float(np.abs(s12 - np.eye(3)).max())
            want = EXPECTED_STAGE12.get(p.name)
            drift = want is None or abs(off - want) > STAGE12_TOL
            print("  %-28s max|M - I| = %.4f%s"
                  % (p.name, off, "   DRIFTED" if drift else ""))
            if drift:
                print("        expected %s" % want)
                bad += 1
            if off > 0.30:
                print("[!] %s: the stage-12 matrix is not near identity, which "
                      "contradicts the argument this module rests on" % p.name)
                bad += 1

    if fp._MEASURED_DYE_MATRIX_ADOPTED:
        print("[!] _MEASURED_DYE_MATRIX has been ADOPTED into dye_matrix. It "
              "double-counts crosstalk already present in the status-M/A "
              "curves. Read the block comment beside the table before "
              "re-enabling, and supply a reader response first")
        bad += 1
    for p, _m, _r, _d in done:
        row0 = p.dye_matrix[0]
        if abs(row0[1] - row0[2]) > 1e-9:
            print("[!] %s carries an asymmetric dye_matrix -- the measured "
                  "table, or something like it, has been wired in" % p.name)
            bad += 1

    if args.assert_:
        if bad:
            print("[FAIL] the derived dye matrices do not reproduce")
            return 1
        print("[OK] %d dye matrices derived from traced spectra through ISO "
              "5-3 status responses: sign pattern correct, unit row sums, and "
              "the early emulsions inside a specification band the derivation "
              "never saw (%d refused). The reader half now closes too: with "
              "%s's traced sensitivity, M_reader . M_status^-1 lands 0.048 to "
              "0.116 from identity against raw off-diagonals reaching 0.24, "
              "which is this module's own argument measured. Still NOT "
              "adopted -- no stock in this database renders through that print "
              "stock" % (len(done), len(refused), PRINT_READER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
