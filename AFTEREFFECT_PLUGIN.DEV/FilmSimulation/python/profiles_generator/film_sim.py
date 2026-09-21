"""
Photochemical film simulation.

Rewrite of the original grain-overlay script as an actual photochemical model.
The original added spectrally-unshaped noise to gamma-encoded sRGB pixels; this
version reproduces the physical chain a photon actually travels, in the domain
each step actually happens in.

Pipeline, in order. Order is not cosmetic -- several steps give visibly wrong
results if moved.

     1. Decode sRGB to linear light.
     2. Scale to relative exposure (18% grey == 1.0), apply exposure offset,
        then the taking matrix (identity except for beam-splitter cameras).
     3. Apply stock colour balance (tungsten vs daylight) as exposure gains,
        then veiling flare from the taking lens -- a broad haze that lifts the
        black floor, and the main thing separating uncoated pre-1940 glass from
        modern coated optics.
     4. Apply large-scale coating unevenness, for stocks with loose QC.
     5. Add halation into *linear exposure*: multi-radius, all three channels,
        energy conserving.
     6. Apply emulsion MTF to the exposure -- light scatter inside the gelatin.
        Red is softest because the red layer sits at the bottom of the stack.
     7. Collapse to a single emulsion record where the stock has one: monochrome
        stocks via their own spectral sensitivity (not video luma), additive
        colour stocks via the reseau filter grid.
     8. Convert exposure to density through the per-channel characteristic
        curve. This is where latitude, highlight rolloff and colour crossover
        come from. Reversal stocks run the curve against negated log exposure.
     9. Apply DIR coupler inter-image effects on density.
    10. Apply scanner MTF and per-channel misregistration to the image. The
        scanner is the pre-sampling filter, so it comes before grain.
    11. Add grain in the density domain, variance scaling as sqrt(density),
        spectrally shaped, calibrated to the stock's RMS granularity and
        band-limited by the same scanner transfer.
    12. Apply the dye impurity / scanner crosstalk matrix to the density vector.
    13. Duplication generations, then print. Each generation is an interpositive
        and a dupe negative on gamma-1.0 stock, adding grain and softness without
        compounding contrast. Reversal stocks skip all of this: they are already
        the positive.
    14. Optional print-stock grain, transmittance to display linear, and reseau
        reconstruction for additive colour stocks.
    15. Encode sRGB, dither, quantise to 16 or 8 bit.

Everything spatial is expressed in micrometres or cycles/mm and converted to
pixels from the negative width and the render width, so a profile behaves the
same at 1080p and 8K. That is the single biggest structural difference from the
original script. Note the corollary: rendered granularity does legitimately
depend on scan resolution, because the scanner MTF band-limits the grain before
sampling -- a 2K render shows less grain than a 6K one of the same negative.

Dependencies: numpy, Pillow. No OpenCV, no SciPy. 16-bit PNG writing is done
with stdlib zlib, so there is no extra dependency for it either.

Tested on CPython 3.12, 64-bit, Windows and Linux/WSL2.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import struct
import sys
import zlib
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path

import numpy as np
from PIL import Image

import film_profiles as fp
from film_profiles import (
    FILM_PROFILES,
    FORMATS,
    frame_pitch_mm,
    IDENTITY3,
    Feature,
    FilmProfile,
    PRINT_STOCKS,
    PrintStock,
    ReseauSpec,
    RGBCurves,
    ToneCurve,
    get_print_stock,
    get_profile,
    validate_all,
)
from algo_control_enums import (      # generated from AlgoControlEnums.hpp
    FilmFormatCtrl,
    PrintStockCtrl,
    DupeStockCtrl,
    ProcessVariantCtrl,
    film_format_key,
    print_stock_key,
    process_variant_key,
)

# 18% reflectance is the photographic mid grey reference. Relative exposure is
# normalised so that mid grey sits at exactly 1.0, i.e. logE = 0.
MID_GREY = 0.18

# Industry granularity convention: sigma(D) measured through a 48 um diameter
# circular aperture. Approximated in the frequency domain by a Gaussian with
# sigma = radius / 2, expressed in millimetres.
APERTURE_SIGMA_MM = (48.0 / 2.0) / 2.0 / 1000.0

# Minimum pixels per reseau cell before the additive colour grid can be
# represented at all. See the fallback in simulate() for why three, not two.
RESEAU_MIN_PITCH_PX = 3.0

EPS = 1e-8


# ===========================================================================
# sRGB transfer functions
# ===========================================================================
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Decode sRGB (IEC 61966-2-1) to linear light."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        np.power((x + 0.055) / 1.055, 2.4, dtype=np.float32),
    ).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Encode linear light to sRGB."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(x, 1.0 / 2.4, dtype=np.float32) - 0.055,
    ).astype(np.float32)


# ===========================================================================
# Characteristic curve evaluation
# ===========================================================================
def _softplus(x: np.ndarray, k: float) -> np.ndarray:
    """Numerically safe k * log(1 + exp(x/k)), no overflow for large x."""
    return (k * np.logaddexp(np.float32(0.0), (x / np.float32(k)))).astype(np.float32)


def density(log_e: np.ndarray, c: ToneCurve) -> np.ndarray:
    """Evaluate a characteristic curve: log exposure to optical density.

    Difference of two softplus ramps gives base+fog, toe, straight line,
    shoulder, Dmax -- the real H&D topology with guaranteed monotonicity.
    """
    return (
        np.float32(c.dmin)
        + np.float32(c.gamma)
        * (
            _softplus(log_e - np.float32(c.toe_x), c.toe_k)
            - _softplus(log_e - np.float32(c.shoulder_x), c.shoulder_k)
        )
    ).astype(np.float32)


def _sp_scalar(x: float, k: float) -> float:
    """Scalar softplus, saturating safely for large arguments."""
    z = x / k
    return x if z > 60.0 else k * math.log1p(math.exp(z))


def density_scalar(log_e: float, c: ToneCurve) -> float:
    """Scalar version of :func:`density`, used by the anchor solvers."""
    return c.dmin + c.gamma * (
        _sp_scalar(log_e - c.toe_x, c.toe_k)
        - _sp_scalar(log_e - c.shoulder_x, c.shoulder_k)
    )


def _normalised_transmittance(d: float, c: ToneCurve,
                              black_point_stretch: float = 1.0) -> float:
    """Density to display-normalised transmittance for one curve.

    ⚠ THIS IS THE SAME EXPRESSION AS STAGE 14 AND IT HAS TO STAY THAT WAY. The
    anchor solvers aim at a display value through this function; stage 14 is
    what actually produces it, and if the two diverge a neutral does not land
    where it was solved to. That is why `black_point_stretch` is threaded
    through every solver rather than applied once at the end. See the field's
    own note in `RenderSettings`.
    """
    t_max = 10.0 ** (-c.dmin)                          # clear film: brightest
    t_min = black_point_stretch * 10.0 ** (-c.dmax)    # the black point
    return (10.0 ** (-d) - t_min) / (t_max - t_min)


def _bisect(fn, lo: float, hi: float, target: float, rising: bool) -> float:
    """Solve fn(x) == target on a monotonic fn. 60 iterations is ample."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        above = fn(mid) > target
        if above == rising:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _tint_factor(profile: FilmProfile, c: int) -> float:
    """Residual base-tint multiplier applied to channel ``c`` at the very end."""
    return 1.0 + (profile.base_tint[c] - 1.0) * 0.5


#: Net-density span and resolution of the Callier lookup. See `callier_lut`.
CALLIER_LUT_MIN = -1.0
CALLIER_LUT_MAX = 5.0
CALLIER_LUT_N = 1025


def callier_net(d_net, q: float, s: float):
    """Silberstein & Tuttle's specular density, as a function of NET density.

        10**-D_sp  =  E * 10**-D_diff  +  (1 - E) * 10**-(beta * D_diff)

    Mees, *The Theory of the Photographic Process*, printed page 644, in the
    chapter FIG. 179 belongs to. The book's own definitions: **E** is "a constant
    expressing the fraction of the scattered light which emerges normally or
    quasi-normally; i.e., the amount accepted by the photometric field of a
    densitometer or by the projection lens of such a device as an enlarging
    printer", and **beta** is "unity plus the ratio of scattering to absorption
    coefficients". "If E = 0, beta is numerically equal to Callier's Q. If
    E = 1.0, D_sp = D_diff."

    ⚠ THAT IS C22's FILM x GEOMETRY SPLIT, IN PRINT SINCE 1942. `scanner_specular`
    is `1 - E` and `callier_q` is `beta`. C22 argued the split from first
    principles because no source stated one. A source states one, three pages
    from a figure already in the corpus.

    ⚠ WHAT THIS REPLACED, AND IT WAS NOT MERELY REFINEMENT. The former law was
    `D_read = dmin + (D - dmin) * (1 + s*(Q-1))` -- a linear interpolation of
    the MULTIPLIER between the two readers. Right at both ends, wrong in
    between, because mixing an accepted and a rejected beam averages
    TRANSMITTANCES and not densities. Measured over the database before the
    change: exactly equal at s = 0 and s = 1, and up to **0.21 D** apart at the
    intermediate settings a user actually dials, 1483 of 3740 sampled points
    differing by more than 0.002 D. ⚠ The two agreed precisely where anyone
    would have hand-checked them and diverged everywhere else.

    ⚠ NET DENSITY IS THE ARGUMENT, WHICH IS THIS PROJECT'S CHOICE AND NOT THE
    BOOK'S. Silberstein and Tuttle write plain D. Referencing to dmin is C22's
    reasoning unchanged -- scattering scales with developed silver and clear base
    carries none, so a condenser must not darken the base. Recorded because it is
    a real difference between what the source states and what this computes.

    ⚠ AND IT DOES NOT FIX THE TOE. Expanding for small D gives
    Q -> E + (1-E)*beta, a CONSTANT. Mees FIG. 179, three pages earlier in the
    same chapter, MEASURES Q collapsing to 1.04 at net density 0.055
    (`mees_callier_q.py`). Model and measurement disagree about the toe and the
    measurement wins; a toe correction still has to come from that figure.

    Exact at both ends by construction: E = 1 returns `d_net`, E = 0 returns
    `beta * d_net`. Defined for NEGATIVE net density too -- grain can push a
    pixel below dmin -- and needs no branch there, so the base case stays exact.
    """
    d = np.asarray(d_net, dtype=np.float64)
    e = 1.0 - s
    t = e * np.power(10.0, -d) + (1.0 - e) * np.power(10.0, -q * d)
    return -np.log10(np.maximum(t, 1e-300))


class _QOnly:
    """Carries just `callier_q`, so `callier_lut` can serve `callier_density`.

    `callier_density` takes Q as a number rather than a profile -- it is called
    from the stage list where the profile is not in scope -- while `callier_lut`
    takes a profile so the solve and the stage cannot end up asking different
    questions. One tiny adapter beats two builders that could drift.
    """

    __slots__ = ("callier_q",)

    def __init__(self, q: float) -> None:
        self.callier_q = float(q)


def callier_is_inert(profile, scanner_specular: float) -> bool:
    """True when the law is the identity and no plane may be touched."""
    return float(scanner_specular) <= 0.0 or float(profile.callier_q) == 1.0


def callier_lut(profile, scanner_specular: float):
    """Uniform lookup over NET density, or None when the law is inert.

    ⚠ THE LOOKUP IS PART OF THE LAW RATHER THAN AN OPTIMISATION OF IT, AND THAT
    IS DELIBERATE. `callier_net` costs two `pow` and a `log10` per channel per
    pixel, and neither has an AVX2 intrinsic. A C++ port evaluating it directly
    would either drop the AVX2 engine to scalar for this stage or compute it
    differently in the two twins -- and a law that differs between twins is the
    exact defect `cpp_parity`'s twin check exists to catch. Both engines and this
    reference therefore build the SAME table, with the same bounds, the same
    count and the same interpolation, so parity holds by construction instead of
    by tolerance.

    Returns `(lo, step, values)`. Outside the span the caller extrapolates on the
    end slopes: below the floor the curve is smooth and nearly linear, and above
    the ceiling it is asymptotically `d_net - log10(E)`, slope exactly 1.
    """
    if callier_is_inert(profile, scanner_specular):
        return None
    q = float(profile.callier_q)
    s = float(scanner_specular)
    xs = np.linspace(CALLIER_LUT_MIN, CALLIER_LUT_MAX, CALLIER_LUT_N)
    step = (CALLIER_LUT_MAX - CALLIER_LUT_MIN) / (CALLIER_LUT_N - 1)
    return CALLIER_LUT_MIN, step, callier_net(xs, q, s)


def callier_lut_at(lut, d_net):
    """Linear interpolation in `lut`, with slope extrapolation outside it."""
    lo, step, v = lut
    n = len(v)
    d = np.asarray(d_net, dtype=np.float64)
    t = (d - lo) / step
    i = np.clip(np.floor(t).astype(np.int64), 0, n - 2)
    out = v[i] + (v[i + 1] - v[i]) * (t - i)
    below = t < 0.0
    above = t > (n - 1)
    if np.any(below):
        out = np.where(below, v[0] + (d - lo) * ((v[1] - v[0]) / step), out)
    if np.any(above):
        hi = lo + step * (n - 1)
        out = np.where(above,
                       v[n - 1] + (d - hi) * ((v[n - 1] - v[n - 2]) / step),
                       out)
    return out


def solve_anchors(
    profile: FilmProfile,
    print_stock: PrintStock,
    grey_target: float,
    coupler_scale: float = 1.0,
    scanner_specular: float = 0.853,
    black_point_stretch: float = 1.0,
) -> tuple[float, float, float]:
    """Per-channel exposure anchors that land 18% scene grey on target.

    For a **negative** stock the free parameter is the print exposure offset in
    ``logE_print = offset - D_neg``. That offset is exactly what a lab sets with
    its printer lights and a colourist sets with a lift, and it has to be solved
    rather than guessed: the naive choice ``offset = D_mid`` puts mid grey
    wherever the print curve happens to cross zero, which for a typical print
    stock is around 2% display luminance -- some three stops too dark.

    For a **reversal** stock there is no print stage, so the only free parameter
    is exposure itself. Which is precisely the situation a photographer shooting
    transparency is in, and why they bracket.

    The solve has to include the taking matrix, the negative dye matrix, the
    print dye matrix and the base tint, because all four scale neutral density
    before it reaches the eye. Ignoring them is not a small error: ORWOcolor's
    dye matrix has row sums near 1.22, which on its own throws the mid tone out
    by more than a stop, and Technicolor's taking filters add another 30%.
    Because the matrices couple the channels, the anchors are found by a short
    fixed-point iteration -- they are near-identity, so it converges in a
    handful of sweeps.

    What is deliberately *not* cancelled is the colour-temperature mismatch from
    ``wb_strength``. A real lab would grade that out, but here it is a creative
    control, and per-channel anchoring would neutralise exactly the cast the
    user asked to see. Curve crossover and the off-diagonal colour mixing also
    survive untouched -- only the per-channel scalar throughput is equalised,
    which is precisely what printer lights do.

    Returns:
        Three anchors: print offsets for a negative, log-exposure trims for a
        reversal stock.
    """
    curves = profile.curves.as_tuple()
    neg_m = profile.dye_matrix
    take = profile.taking_matrix

    # ⚠ THE SOLVE HAS TO SEE THE READER'S OPTICS TOO (C22, 2026-08-23). Callier
    # steepens the density the printer or scanner reads, and a lab responds by
    # RE-TIMING the print -- that is what printer lights are for. If the anchor
    # solve is left blind to it, a condenser setting both steepens the tone scale
    # AND shifts mid grey, and the shift is the larger of the two: measured on
    # DOUBLE-X at specular = 1, mid grey moved +48/255 before this was wired in,
    # against a contrast change of a few per cent. One of those two effects is
    # the physics; the other is the lab failing to do its job.
    # ⚠ THE SOLVE EVALUATES THE LAW EXACTLY AND THE PIXEL PASS USES THE TABLE,
    # AND THAT ASYMMETRY IS ON PURPOSE. The solve touches a handful of scalars
    # per iteration, so the two `pow` calls cost nothing there and an exact
    # answer is worth having; the pixel pass touches millions, where the table
    # is what makes the AVX2 twin possible at all. The table is built FROM this
    # same function, so the two cannot drift apart in meaning -- only by the
    # table's own interpolation error, which `cpp_parity` measures.
    _cal_inert = callier_is_inert(profile, scanner_specular)
    _cal_q = float(profile.callier_q)
    _cal_s = float(scanner_specular)

    def _cal_apply(d: list[float]) -> list[float]:
        if _cal_inert:
            return list(d)
        return [curves[k].dmin
                + float(callier_net(d[k] - curves[k].dmin, _cal_q, _cal_s))
                for k in range(3)]

    # Log exposure each record actually receives from a neutral 18% grey, which
    # is 1.0 in relative exposure before the taking filters mix the records.
    log_e_mid = [
        math.log10(max(sum(take[k][j] for j in range(3)), EPS)) for k in range(3)
    ]

    # Flat-field part of the DIR coupler effect. On an even field the edge term
    # vanishes, but the cross-layer term does not: it pushes each layer away
    # from the mean of the three, and because curve crossover means a neutral
    # grey does *not* sit at equal density in all three layers, that shifts the
    # mid tone by a few percent per channel.
    cp_s = profile.couplers.strength * coupler_scale
    couple_flat = cp_s > 0.0 and not profile.is_monochrome

    def _couple(d: list[float]) -> list[float]:
        if not couple_flat:
            return list(d)
        dbar = sum(d) / 3.0
        return [d[k] + cp_s * (d[k] - dbar) for k in range(3)]

    def _neg_density(anchors: list[float]) -> list[float]:
        """Uncoupled per-layer density at neutral grey for given anchors."""
        if profile.is_reversal:
            return [
                density_scalar(-(log_e_mid[k] + anchors[k]), curves[k])
                for k in range(3)
            ]
        return [density_scalar(log_e_mid[k], curves[k]) for k in range(3)]

    if profile.is_reversal:
        trims = [0.0, 0.0, 0.0]
        for _ in range(8):
            frozen = _neg_density(trims)
            for c in range(3):
                # Re-solve one channel with the other two held at their current
                # values; sweeping all three to convergence handles the coupling.
                def fn(t: float, c: int = c, frozen: list[float] = frozen) -> float:
                    d = list(frozen)
                    d[c] = density_scalar(-(log_e_mid[c] + t), curves[c])
                    d = _couple(d)
                    mixed = sum(neg_m[c][k] * d[k] for k in range(3))
                    # A slide is read by the same optics as a negative print, so
                    # the projector's or scanner's directionality applies here too.
                    mixed = _cal_apply([mixed] * 3)[c]
                    return _normalised_transmittance(
                        mixed, curves[c], black_point_stretch)

                target = grey_target / _tint_factor(profile, c)
                trims[c] = _bisect(fn, -8.0, 8.0, target, rising=True)
        return (trims[0], trims[1], trims[2])

    # Neutral negative density, after couplers and the negative's dye matrix.
    d_neg = _couple(_neg_density([0.0, 0.0, 0.0]))
    d_mid = _cal_apply(
        [sum(neg_m[c][k] * d_neg[k] for k in range(3)) for c in range(3)])
    targets = [grey_target / _tint_factor(profile, c) for c in range(3)]
    offsets = solve_stage_offsets(
        d_mid, print_stock.curves.as_tuple(), print_stock.dye_matrix, targets,
        black_point_stretch
    )
    return (offsets[0], offsets[1], offsets[2])


def neutral_mid_density(
    profile: FilmProfile, coupler_scale: float = 1.0
) -> list[float]:
    """Density a neutral 18% grey reaches on the camera negative.

    Includes the taking matrix, the flat-field coupler term and the negative's
    dye matrix -- i.e. everything the scalar chain does to a neutral before the
    image leaves the negative. This is the starting point every subsequent
    printing stage anchors against.
    """
    curves = profile.curves.as_tuple()
    take = profile.taking_matrix
    neg_m = profile.dye_matrix
    log_e_mid = [
        math.log10(max(sum(take[k][j] for j in range(3)), EPS)) for k in range(3)
    ]
    d = [density_scalar(log_e_mid[k], curves[k]) for k in range(3)]
    cp_s = profile.couplers.strength * coupler_scale
    if cp_s > 0.0 and not profile.is_monochrome:
        dbar = sum(d) / 3.0
        d = [d[k] + cp_s * (d[k] - dbar) for k in range(3)]
    return [sum(neg_m[c][k] * d[k] for k in range(3)) for c in range(3)]


def solve_stage_offsets(
    d_mid: list[float],
    dst_curves: tuple[ToneCurve, ToneCurve, ToneCurve],
    dst_matrix,
    targets: list[float],
    black_point_stretch: float = 1.0,
) -> list[float]:
    """Print offsets landing neutral grey on ``targets`` display values.

    One channel is re-solved at a time with the other two frozen, swept to
    convergence, which handles the cross-channel coupling of the dye matrix.
    """
    offsets = list(d_mid)
    for _ in range(8):
        frozen = [
            density_scalar(offsets[k] - d_mid[k], dst_curves[k]) for k in range(3)
        ]
        for c in range(3):

            def fn(off: float, c: int = c, frozen: list[float] = frozen) -> float:
                dp = list(frozen)
                dp[c] = density_scalar(off - d_mid[c], dst_curves[c])
                mixed = sum(dst_matrix[c][k] * dp[k] for k in range(3))
                return _normalised_transmittance(
                    mixed, dst_curves[c], black_point_stretch)

            # More offset means more print exposure, more density, darker print.
            offsets[c] = _bisect(
                fn, d_mid[c] - 8.0, d_mid[c] + 8.0, targets[c], rising=False
            )
    return offsets


def solve_intermediate_offsets(
    d_mid: list[float], dst_curves: tuple[ToneCurve, ToneCurve, ToneCurve]
) -> tuple[list[float], list[float]]:
    """Offsets that centre neutral grey in a duplicating stock's usable range.

    An intermediate generation is not viewed, so there is no display value to
    aim at. Aiming at the midpoint of the stock's density range is what a lab
    does with its printer lights, and it keeps the chain from drifting into the
    toe or the shoulder over three or four generations.

    Returns:
        ``(offsets, new_d_mid)`` -- the second is the neutral density after this
        stage, which the next stage anchors against.
    """
    offsets: list[float] = []
    mids: list[float] = []
    for c in range(3):
        dst = dst_curves[c]
        target_d = 0.5 * (dst.dmin + dst.dmax)
        fn = lambda off, dst=dst, c=c: density_scalar(off - d_mid[c], dst)
        offsets.append(
            _bisect(fn, d_mid[c] - 10.0, d_mid[c] + 10.0, target_d, rising=True)
        )
        mids.append(target_d)
    return offsets, mids


# ===========================================================================
# Colour temperature
# ===========================================================================
def _planck(lam_nm: float, kelvin: float) -> float:
    """Spectral radiance of a blackbody, arbitrary units."""
    lam = lam_nm * 1e-9
    c1 = 3.741771e-16
    c2 = 1.438777e-2
    return c1 / (lam**5 * math.expm1(c2 / (lam * kelvin)))


# ===========================================================================
# MEASURED SPECTRAL SENSITIVITY — the consumer of SpectralSensitivity
# ===========================================================================
#
# WHY THIS BLOCK EXISTS. The profile database carries digitised per-layer
# spectral sensitivity curves (SpectralSensitivity.log_s_r/g/b/pan, sampled at
# lambda_start_nm + k*lambda_step_nm). Until this block was written NOTHING
# read them: the renderer approximated the same physics with three proxies --
#
#   * balance_gains()          three hard-coded "peak" wavelengths 600/550/450
#   * profile.taking_matrix    a hand-fitted 3x3
#   * profile.spectral_weights three hand-fitted monochrome weights
#
# Each proxy answers a question the measured curve answers exactly, so each is
# replaced here BY DERIVATION FROM THE CURVE where the curve exists, and left
# untouched where it does not. Nothing is invented: a stock with no spectral
# data keeps exactly the numbers and the behaviour it had before.
#
# WHAT THIS IS AND IS NOT. This is the "illuminant-conditioned integration"
# path: the layer sensitivities are integrated against a real illuminant SPD
# and against the input's assumed primaries, producing a mixing matrix that is
# DERIVED rather than fitted. It is exact for neutrals and for illuminant
# changes, and it remains an approximation for saturated colours, because a
# three-number RGB input no longer carries the spectral detail that would
# distinguish two metamers. Removing that limit needs spectral input, not more
# film data. Stated here so the improvement is not overclaimed.
#
# NUMERICAL NOTE. Everything in this block is setup-domain: it runs once per
# render, never per pixel, so it computes in float64 and hands float32 to the
# pixel path. That is the same split the engine's precision policy uses.

#: Wavelength grid the INTEGRALS are evaluated on, nanometres.
#:
#: This is NOT the stored sampling of any curve and it does not claim to be. The
#: stored curves are 10 nm for 49 of 53 stocks and 20-25 nm for four; they are
#: interpolated up onto this grid so that the integral is not quantised to
#: whatever sampling the source plot happened to have. Interpolating up for the
#: purpose of integration invents no information -- it changes where the
#: trapezoid rule places its nodes, nothing else.
#:
#: 2 nm rather than 5, measured 2026-08-13. Against a smooth blackbody the choice
#: barely matters: 5 nm differs from a 1 nm reference by 1.1e-3 on the derived
#: balance gains, 10 nm by 3.2e-3. Against a NARROW-LINE illuminant it matters
#: enormously -- with 5 nm mercury lines the red/green layer ratio is wrong by
#: 1.5 % at a 5 nm grid, 52.7 % at 10 nm and 231 % at 25 nm, while 2 nm matches
#: the 1 nm reference exactly. Only blackbody SPDs are integrated today, so 5 nm
#: was adequate by coincidence rather than by design; 2 nm removes the trap that
#: adding a fluorescent or LED illuminant later would silently introduce a
#: double-digit error. Cost measured at 0.129 -> 0.179 ms per full derivation,
#: setup domain, roughly sixty integrals per frame: unmeasurable at frame scale.
_SPECTRAL_LAMBDA_STEP = 2.0
_SPECTRAL_LAMBDA_MIN = 360.0
_SPECTRAL_LAMBDA_MAX = 730.0


def spectral_grid() -> np.ndarray:
    """The common wavelength grid, nanometres, float64."""
    n = int(round((_SPECTRAL_LAMBDA_MAX - _SPECTRAL_LAMBDA_MIN)
                  / _SPECTRAL_LAMBDA_STEP)) + 1
    return (_SPECTRAL_LAMBDA_MIN
            + _SPECTRAL_LAMBDA_STEP * np.arange(n, dtype=np.float64))


def layer_sensitivities(profile) -> np.ndarray | None:
    """Per-layer LINEAR spectral sensitivity on ``spectral_grid()``.

    Returns an array of shape (3, n_lambda) for colour stocks, or (1, n_lambda)
    for monochrome stocks that carry only ``log_s_pan``; ``None`` when the
    profile has no digitised curves at all, which is the signal to every caller
    below to fall back to the pre-existing hand-fitted proxy.

    The stored values are LOG sensitivity, so they are exponentiated here. The
    curve's own sampling is respected: values outside the measured wavelength
    range are treated as zero sensitivity rather than extrapolated, because an
    extrapolated sensitisation tail is an invention and would change the
    integral in the direction that flatters the model.
    """
    sp = profile.spectral
    if not sp.has_data:
        return None

    rows: list[tuple[float, ...]] = []
    if sp.log_s_r and sp.log_s_g and sp.log_s_b:
        rows = [sp.log_s_r, sp.log_s_g, sp.log_s_b]
    elif sp.log_s_pan:
        rows = [sp.log_s_pan]
    else:
        return None

    grid = spectral_grid()
    out = np.zeros((len(rows), grid.size), dtype=np.float64)
    for i, row in enumerate(rows):
        src_lam = (sp.lambda_start_nm
                   + sp.lambda_step_nm * np.arange(len(row), dtype=np.float64))
        src_val = np.asarray(row, dtype=np.float64)
        # Interpolate in LOG space -- a sensitisation curve is smooth in log
        # sensitivity and emphatically not in linear sensitivity, where a 4-decade
        # span would make linear interpolation between samples grossly wrong.
        interp = np.interp(grid, src_lam, src_val,
                           left=-np.inf, right=-np.inf)
        out[i] = np.where(np.isfinite(interp), np.power(10.0, interp), 0.0)

    # ⚠ THE TAKING FILTER IS APPLIED ON THIS PATH TOO, AND LEAVING IT OFF WOULD
    # HAVE BEEN INVISIBLE (queue C39, schema v20). `stored_layer_sensitivities`
    # feeds the GUARDS and this feeds the INTEGRATION. Filtering only the first
    # gives a guard that judges a filtered emulsion and weights derived from a
    # bare one -- the two would disagree about what film they were looking at,
    # and on every stock the guard happens to refuse the disagreement never
    # shows. Inert on 163 of 165 profiles.
    tf = getattr(profile, "taking_filter", None)
    if tf is not None and tf.renders:
        out = out * taking_filter_transmission(tf, grid)[None, :]
    return out


def planck_spd(kelvin: float) -> np.ndarray:
    """Blackbody spectral power distribution on ``spectral_grid()``, float64.

    Normalised to unit value at 560 nm so that integrals against it stay in a
    numerically comfortable range; every use below forms a RATIO, so the
    normalisation cancels and does not affect any result.
    """
    grid = spectral_grid()
    spd = np.array([_planck(float(l), kelvin) for l in grid], dtype=np.float64)
    ref = _planck(560.0, kelvin)
    return spd / ref if ref > 0.0 else spd


def spectral_layer_exposure(profile, spd: np.ndarray) -> np.ndarray | None:
    """Integrate ``spd`` against each layer's measured sensitivity.

    This is the core integral of the whole block:

        E_layer = INTEGRAL S_layer(lambda) * E(lambda) d lambda

    ``spd`` must already be sampled on ``spectral_grid()``. Returns one value
    per layer (3 for colour, 1 for monochrome-pan), or ``None`` when the
    profile carries no curves.
    """
    sens = layer_sensitivities(profile)
    if sens is None:
        return None
    return np.trapezoid(sens * spd[None, :], spectral_grid(), axis=1) \
        if hasattr(np, "trapezoid") else \
        np.trapz(sens * spd[None, :], spectral_grid(), axis=1)


def spectral_balance_gains(profile, scene_kelvin: float) -> tuple[float, ...] | None:
    """Colour-temperature gains computed from the MEASURED curves.

    Replaces ``balance_gains()`` for any stock that carries spectral data. The
    quantity is the same ratio the proxy estimates:

        gain_c = INTEGRAL S_c(l) P(l, T_scene) dl / INTEGRAL S_c(l) P(l, T_stock) dl

    but evaluated over the whole measured sensitisation instead of at one
    assumed peak wavelength. The difference is largest exactly where the proxy
    is weakest: a broad or double-peaked sensitisation, an orthochromatic
    emulsion whose "red peak" does not exist, and any stock whose real peak is
    far from 600/550/450 nm.

    Green is normalised to 1.0, as in the proxy, so overall exposure is
    unchanged and only the colour balance moves.

    Returns ``None`` for a stock with no curves, or for a monochrome stock,
    where a per-channel balance has no meaning.
    """
    sens = layer_sensitivities(profile)
    if sens is None or sens.shape[0] != 3:
        return None

    scene = spectral_layer_exposure(profile, planck_spd(scene_kelvin))
    stock = spectral_layer_exposure(profile, planck_spd(profile.balance_kelvin))
    if scene is None or stock is None:
        return None
    if not np.all(stock > 0.0):
        return None

    ratio = scene / stock
    if ratio[1] <= 0.0:
        return None
    ratio = ratio / ratio[1]
    return tuple(float(v) for v in ratio)


#: Longest wavelength at which the three primary lobes still have usable
#: amplitude. Beyond this a visible-primary basis cannot excite the emulsion at
#: all, so projecting a curve that peaks out here onto that basis answers a
#: different question than the one being asked. 700 nm is where the reddest
#: lobe (600 nm centre, 55 nm width) has fallen to about 16 % of its peak.
_SPECTRAL_BASIS_LAMBDA_MAX = 700.0


#: Largest share of a curve's sensitivity-weighted energy that may lie beyond
#: _SPECTRAL_BASIS_LAMBDA_MAX before a visible-primary projection stops being
#: meaningful. A stock with a fifth of its response in the deep red or the
#: infrared is not describable by three visible lobes, whatever its peak does.
_SPECTRAL_OUT_OF_REACH_MAX = 0.15


def stored_layer_sensitivities(profile):
    """Per-layer LINEAR sensitivity on the curve's OWN sampling, unclipped.

    ⚠ THIS IS NOT :func:`layer_sensitivities`, AND THE DIFFERENCE IS THE WHOLE
    POINT. That function resamples onto :func:`spectral_grid`, which stops at
    ``_SPECTRAL_LAMBDA_MAX`` = 730 nm because that is the domain the renderer
    integrates over. Anything the emulsion does past 730 nm is dropped there,
    correctly -- the renderer cannot act on it.

    The GUARDS, however, must see exactly what the renderer cannot. A guard
    that asks "how much of this emulsion lies outside the basis's reach?" and
    then measures it on a grid that has already discarded the far red is
    answering its own question with the evidence removed.

    ⚠ MEASURED, 2026-08-29. ``KONICA_INFRARED_750`` stores a curve sampled to
    830 nm. On the clipped grid its out-of-reach share read **0.203** and its
    peak read **730 nm**; on its own samples they are **0.437** and **750 nm**.
    The guard refused it either way, so nothing rendered wrong -- but it was
    refusing on a number low by a factor of two and on a peak that was an
    artefact of the grid's last sample. A threshold tested against a number
    that cannot reach it is the same defect this project has now caught three
    times (C20's guard that could not fail, the census that counted the wrong
    field, the 2026-08-26 sweep that no test re-ran).

    Returns ``(lambda_nm, [row, ...])`` with rows in LINEAR sensitivity, or
    ``None`` when the profile carries no digitised curve.
    """
    sp = getattr(profile, "spectral", None)
    if sp is None or not sp.has_data:
        return None

    if sp.log_s_r and sp.log_s_g and sp.log_s_b:
        rows = [sp.log_s_r, sp.log_s_g, sp.log_s_b]
    elif sp.log_s_pan:
        rows = [sp.log_s_pan]
    else:
        return None

    n = len(rows[0])
    lam = (sp.lambda_start_nm
           + sp.lambda_step_nm * np.arange(n, dtype=np.float64))
    lin = [np.power(10.0, np.asarray(r, dtype=np.float64)) for r in rows]

    # ⚠ THE TAKING FILTER IS APPLIED HERE, WHICH MEANS EVERY GUARD AND THE
    # COLLAPSE ALL SEE THE SAME EMULSION (queue C39, schema v20). Putting it
    # anywhere further downstream would let the reach guard judge a bare curve
    # and the weights be derived from a filtered one, which is a split of the
    # exact kind this function was written to close in the first place.
    #
    # ⚠ AND IT IS `profile.taking_filter`, NOT `spectral.measured_through`.
    # The stored curve is what the sheet plotted; this is what the profile
    # assumes in front of the lens. On both infrared stocks the sheet plotted
    # the BARE emulsion, so applying the intended filter is exactly the missing
    # step -- and on the other 163 profiles the filter is empty and this loop
    # does not execute, so their curves are untouched to the last bit.
    tf = getattr(profile, "taking_filter", None)
    if tf is not None and tf.renders:
        t = taking_filter_transmission(tf, lam)
        lin = [row * t for row in lin]
    return lam, lin


def taking_filter_transmission(tf, lam):
    """T(lambda) for a TakingFilter on an arbitrary wavelength grid.

    ⚠ THE IDEAL LONGPASS IS A HARD STEP AND THAT IS DELIBERATE. A real 715 nm
    filter has a finite edge, and modelling one would mean inventing an edge
    width no source states. The question this transmission is actually used to
    answer is "does the usable energy of this emulsion lie inside the
    renderer's spectral basis at all", and for that a step at the wavelength
    the sheet PRINTS is both sufficient and honest. `TakingFilter.model`
    records which kind of thing the caller is holding.
    """
    if tf.model == "measured":
        return np.asarray(tf.transmission, dtype=np.float64)
    if tf.model == "ideal_longpass" and tf.cut_on_nm > 0.0:
        return (np.asarray(lam, dtype=np.float64) >= tf.cut_on_nm).astype(
            np.float64)
    return np.ones_like(np.asarray(lam, dtype=np.float64))


def spectral_out_of_reach(profile) -> float | None:
    """Share of the stock's sensitivity lying beyond the basis's red limit.

    Companion to :func:`spectral_peak_lambda`: the peak catches an emulsion
    whose maximum is in the infrared, this catches one whose maximum is in the
    visible but which carries a substantial infrared shoulder -- the case that
    a peak test alone passes incorrectly.

    Measured on the curve's own samples (see
    :func:`stored_layer_sensitivities`), never on the renderer's clipped grid.
    """
    stored = stored_layer_sensitivities(profile)
    if stored is None:
        return None
    lam, rows = stored
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    beyond = lam > _SPECTRAL_BASIS_LAMBDA_MAX
    total = 0.0
    out = 0.0
    for row in rows:
        total += float(trap(row, lam))
        if int(beyond.sum()) > 1:
            out += float(trap(row[beyond], lam[beyond]))
    return (out / total) if total > 0.0 else None


def spectral_peak_lambda(profile) -> float | None:
    """Wavelength of peak sensitivity, nanometres, per layer maximum.

    Returns the LONGEST per-layer peak, because it is the long-wavelength end
    that a visible-primary basis fails to reach. ``None`` when the stock carries
    no curves.

    Measured on the curve's own samples, for the reason set out in
    :func:`stored_layer_sensitivities`: read off the clipped grid, an infrared
    stock's peak collapses onto the grid's last sample and stops being a fact
    about the emulsion.
    """
    stored = stored_layer_sensitivities(profile)
    if stored is None:
        return None
    lam, rows = stored
    peaks = [float(lam[int(np.argmax(row))]) for row in rows
             if float(row.max()) > 0.0]
    return max(peaks) if peaks else None


def spectral_monochrome_weights(profile) -> tuple[float, ...] | None:
    """Monochrome R/G/B weights derived from the measured pan curve.

    Replaces the hand-fitted ``spectral_weights`` triple for a monochrome stock
    that carries ``log_s_pan``. The input image has already been reduced to
    three channels, so the honest derivation is: integrate the pan sensitivity
    against each of the input's primaries and normalise the three integrals to
    sum to one. That is the exact weight with which each input primary
    contributes to the single silver record.

    This is where an orthochromatic emulsion earns its near-zero red weight
    from its own measured curve rather than from an authored constant.
    """
    sens = layer_sensitivities(profile)
    if sens is None or sens.shape[0] != 1:
        return None

    # ------------------------------------------------------------------
    # GAMUT-REACH GUARD. Refuse the derivation when the emulsion is
    # sensitised substantially outside what three visible primaries can
    # excite. Without this guard the function returns a confident wrong
    # answer, and it was measured doing exactly that:
    #
    #   KONICA_INFRARED_750, sensitised 380-830 nm with a 750 nm peak,
    #   derived to (0.161, 0.193, 0.646) -- BLUE-dominant -- because the
    #   only part of that emulsion the primary lobes can see is its
    #   intrinsic 380-500 nm lobe. The authored triple is (0.55, 0.15,
    #   0.30), red-dominant, which is right for an infrared film. The
    #   derived answer is a true statement about photographing a monitor
    #   and a nonsense one about photographing the world.
    #
    # This reproduces, independently, the finding recorded on 2026-08-03
    # against film_profiles.derived_spectral_response(), which was
    # quarantined for the same reason. The prior decision was correct.
    # ------------------------------------------------------------------
    peak = spectral_peak_lambda(profile)
    if peak is None or peak > _SPECTRAL_BASIS_LAMBDA_MAX:
        return None
    out = spectral_out_of_reach(profile)
    if out is None or out > _SPECTRAL_OUT_OF_REACH_MAX:
        return None

    prim = _srgb_primary_spd()
    grid = spectral_grid()
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    w = np.array([trap(sens[0] * prim[c], grid) for c in range(3)],
                 dtype=np.float64)
    total = float(w.sum())
    if total <= 0.0:
        return None
    return tuple(float(v / total) for v in w)


#: Smooth, strictly positive spectral basis standing in for the sRGB primaries.
#: These are NOT the CIE primaries: sRGB primaries are defined by chromaticity,
#: not by a spectrum, and any RGB triple corresponds to infinitely many spectra.
#: Gaussian lobes centred on the primaries' dominant wavelengths are the
#: standard smooth choice, and the choice is declared here rather than buried,
#: because it is an ASSUMPTION of this path and one of the reasons saturated
#: colour stays approximate (see the block header).
_PRIMARY_CENTRES_NM = (600.0, 540.0, 460.0)
_PRIMARY_WIDTH_NM = 55.0


def _srgb_primary_spd() -> np.ndarray:
    """Three smooth primary SPDs on ``spectral_grid()``, each unit-area."""
    grid = spectral_grid()
    out = np.zeros((3, grid.size), dtype=np.float64)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for c, centre in enumerate(_PRIMARY_CENTRES_NM):
        lobe = np.exp(-0.5 * ((grid - centre) / _PRIMARY_WIDTH_NM) ** 2)
        area = float(trap(lobe, grid))
        out[c] = lobe / area if area > 0.0 else lobe
    return out


def spectral_taking_matrix(profile, scene_kelvin: float = 5500.0) -> np.ndarray | None:
    """The exposure-mixing matrix DERIVED from the measured curves.

    Element [layer][primary] is the response of that layer to that input
    primary under the given illuminant:

        M[l][p] = INTEGRAL S_l(lambda) * P_p(lambda) * I(lambda, T) d lambda

    normalised so each ROW sums to one, which keeps a neutral input neutral and
    confines the matrix's effect to cross-channel mixing -- the part that is
    genuinely the film's spectral character.

    This is the derived replacement for the authored ``taking_matrix``. It is
    returned rather than applied, so the caller can compare the two: a large
    disagreement is a finding about one of them, not something to average away.

    Returns ``None`` for stocks without three-layer curves; a beam-splitter
    stock whose authored matrix encodes real taking FILTERS (not sensitivities)
    must keep its authored matrix, and the caller is responsible for that
    distinction.
    """
    sens = layer_sensitivities(profile)
    if sens is None or sens.shape[0] != 3:
        return None

    grid = spectral_grid()
    illum = planck_spd(scene_kelvin)
    prim = _srgb_primary_spd()
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    m = np.zeros((3, 3), dtype=np.float64)
    for l in range(3):
        for p in range(3):
            m[l][p] = trap(sens[l] * prim[p] * illum, grid)

    rows = m.sum(axis=1, keepdims=True)
    if not np.all(rows > 0.0):
        return None
    return (m / rows).astype(np.float32)


def spectral_exposure_report(profile, scene_kelvin: float = 5500.0) -> dict:
    """Everything this block derives, for one profile, in one call.

    Diagnostic and provenance surface: it reports what was derived from
    measurement, what fell back to an authored proxy, and by how much the two
    disagree. Intended for the audit tooling and for regression tests, so that
    "the spectral path is active" is a checkable statement rather than a claim.
    """
    sens = layer_sensitivities(profile)
    out: dict = {
        "name": profile.name,
        "has_curves": sens is not None,
        "n_layers": 0 if sens is None else int(sens.shape[0]),
        "criterion": profile.spectral.criterion,
        "lambda_step_nm": profile.spectral.lambda_step_nm,
    }
    if sens is None:
        out["source"] = "authored proxy (no measured curves)"
        return out
    out["source"] = "measured curves"

    if sens.shape[0] == 3:
        derived = spectral_taking_matrix(profile, scene_kelvin)
        authored = np.asarray(profile.taking_matrix, dtype=np.float64)
        if derived is not None:
            out["taking_matrix_derived"] = derived.tolist()
            out["taking_matrix_max_abs_diff"] = float(
                np.max(np.abs(derived.astype(np.float64) - authored)))
        g = spectral_balance_gains(profile, scene_kelvin)
        if g is not None:
            out["balance_gains_derived"] = g
            out["balance_gains_proxy"] = balance_gains(
                scene_kelvin, profile.balance_kelvin)
    else:
        w = spectral_monochrome_weights(profile)
        if w is not None:
            out["mono_weights_derived"] = w
            out["mono_weights_authored"] = tuple(
                float(v) for v in profile.spectral_weights)
    return out


def balance_gains(scene_kelvin: float, stock_kelvin: float) -> tuple[float, ...]:
    """Per-channel exposure gains for a colour-temperature mismatch.

    A stock balanced for ``stock_kelvin`` has its layer sensitivities trimmed so
    that illuminant neutral. Shooting under ``scene_kelvin`` therefore multiplies
    each layer's exposure by the ratio of blackbody radiance at that layer's
    peak wavelength. Daylight on tungsten stock gives the familiar heavy blue
    cast, which falls out of the physics instead of being hand-tinted.

    Green is normalised to 1.0 so overall brightness is unchanged.
    """
    lams = (600.0, 550.0, 450.0)  # approximate layer sensitivity peaks
    scene = [_planck(l, scene_kelvin) for l in lams]
    stock = [_planck(l, stock_kelvin) for l in lams]
    ratio = [s / f for s, f in zip(scene, stock)]
    return tuple(r / ratio[1] for r in ratio)


# ===========================================================================
# Frequency-domain helper
# ===========================================================================
class FreqGrid:
    """Radial frequency grids for a given image size and scan resolution.

    Holds the half-spectrum (rfft2) frequency magnitudes in both cycles/mm and
    cycles/pixel, plus the multiplicity weights needed to compute a full-grid
    spectral mean from the half spectrum. Those weights are what let the grain
    field be calibrated analytically to an RMS granularity figure without
    generating and measuring a test render.
    """

    def __init__(self, h: int, w: int, px_per_mm: float, anisotropy: float = 1.0):
        self.h = h
        self.w = w
        self.px_per_mm = px_per_mm

        fy = np.fft.fftfreq(h).astype(np.float32)          # cycles/pixel
        fx = np.fft.rfftfreq(w).astype(np.float32)
        self.fy_cpp = fy[:, None]
        self.fx_cpp = fx[None, :]

        # Anisotropy stretches the vertical correlation length, modelling
        # emulsion coating flow direction. Scaling vertical frequency *up*
        # attenuates vertical detail more, which lengthens vertical correlation
        # -- getting this the wrong way round silently squashes the grain
        # instead of stretching it.
        fy_mm = fy * px_per_mm * max(anisotropy, 1e-6)
        fx_mm = fx * px_per_mm
        self.f_mm = np.sqrt(fy_mm[:, None] ** 2 + fx_mm[None, :] ** 2).astype(
            np.float32
        )

        # Multiplicity: interior columns of the half spectrum stand for two
        # full-grid bins, column 0 and the Nyquist column for one.
        wts = np.full(fx.shape[0], 2.0, dtype=np.float32)
        wts[0] = 1.0
        if w % 2 == 0:
            wts[-1] = 1.0
        self.col_weight = wts[None, :]
        self.n_full = float(h * w)

    def spectral_mean(self, transfer_sq: np.ndarray) -> float:
        """Full-grid mean of |H|^2 computed from the half spectrum.

        For unit-variance white noise input, this equals the variance of the
        filtered output (Parseval), which is exactly what the granularity
        calibration needs.
        """
        return float((transfer_sq * self.col_weight).sum() / self.n_full)

    # -- transfer functions -------------------------------------------------
    def gaussian(self, sigma_um: float) -> np.ndarray:
        """Transfer of a Gaussian blur of the given sigma in micrometres."""
        s_mm = sigma_um / 1000.0
        return np.exp(
            -2.0 * (math.pi**2) * (s_mm**2) * (self.f_mm.astype(np.float32) ** 2)
        ).astype(np.float32)

    def kernel_transfer(self, sigma_px: float, n: int,
                        cutoff: float = 4.0) -> np.ndarray:
        """Exact transfer of the C++ engine's TRUNCATED SEPARABLE Gaussian.

        ⚠ INERT. NO RENDER PATH CALLS THIS. It exists so the parity tooling can
        PREDICT the production engine's blur instead of tolerating the
        difference, and it is placed here rather than in the parity script
        because the formula it implements is the C++ kernel's, and the two must
        never drift apart.

        WHAT IT IS FOR -- queue A2 / C16, 2026-09-02e. `apply_transfer`
        multiplies the DFT by the ANALYTIC Gaussian transfer
        exp(-2 pi^2 sigma^2 f^2); `AlgoGaussianBlurPlaneWrap` convolves with a
        SAMPLED Gaussian truncated at ``cutoff`` sigma (half = ceil(4 sigma),
        minimum 1) and renormalised to unit sum. Those are different operators
        and C16 measured the difference empirically: "6e-5 above ~1.2 px,
        diverging to 1.5e-1 at 0.4 px". This function makes it exact.

        ⚠ AND THE MEASUREMENT THAT EXPLAINS THE 1.2 px. The divergence is not a
        gradual loss of accuracy in the kernel; it is ALIASING, and it lives
        entirely at Nyquist. A spatial kernel's transfer is periodic in
        frequency, so what it applies is the PERIODISED analytic transfer,
        sum over m of T(f + m). At f = Nyquist the m = -1 image lands exactly on
        the m = 0 term and the transfer is DOUBLED. Measured on a 1024-sample
        grid, kernel transfer at Nyquist against analytic:
            sigma 0.60 px   0.3379 vs 0.1692   ratio 2.00
            sigma 0.80 px   0.0850 vs 0.0425   ratio 2.00
            sigma 1.00 px   0.0144 vs 0.0072   ratio 2.00
            sigma 1.20 px   0.0016 vs 0.0008   ratio 2.00
        The ratio is 2.00 at every sigma. ⚠ SO THE TWO FORMS DO NOT CONVERGE
        ABOVE 1.2 px BECAUSE THE KERNEL GETS BETTER -- it does not, it is always
        exactly twice as high at Nyquist. They converge because T(Nyquist)
        itself falls to zero, and twice a vanishing number vanishes. The "1.2
        px" in C16's row is the sigma at which 2*T(Nyquist) drops below 1e-3;
        nothing changes about the kernel there.
        ⚠ BELOW ABOUT 0.8 px THE CLOSED FORM STOPS HOLDING, because the
        truncation at 4 sigma and the renormalisation that follows it become
        significant: at sigma 0.4 the support is 5 taps and the periodised
        prediction is 8.5e-2 away from the actual kernel, at sigma 0.25 it is
        3 taps and 6.0e-1 away. That is why this function builds the taps and
        transforms them rather than summing images of T -- for a five-tap
        renormalised kernel only the taps themselves are the truth.
        """
        half = max(1, int(math.ceil(cutoff * float(sigma_px))))
        x = np.arange(-half, half + 1, dtype=np.float64)
        k = np.exp(-0.5 * (x / float(sigma_px)) ** 2)
        k /= k.sum()
        lag = np.zeros(int(n), dtype=np.float64)
        for i, xx in enumerate(x.astype(int)):
            lag[xx % int(n)] += k[i]
        return np.real(np.fft.rfft(lag))

    def mtf(self, f50_cpmm: float, adjacency: float, adjacency_um: float,
            spec: "fp.MTFSpec | None" = None, channel: int = 1,
            use_kernel: bool = False) -> np.ndarray:
        """Emulsion or scanner MTF, 50% modulation at ``f50_cpmm``.

        50 % at ``f50_cpmm`` exactly, optionally multiplied by a mild
        low-frequency lift representing development adjacency overshoot (real MTF
        curves often exceed 100% at low frequency).

        ⚠ THE ROLLOFF SHAPE NOW COMES FROM `fp.mtf_response` WHEN A SPEC IS PASSED
        (queue item C2, 2026-08-19). Until then this was hardcoded Gaussian for
        everything, while `mtf_tail_a` / `mtf_tail_f_exp` sat in the schema unread
        -- the same state sigma(D) was in before C1. Callers that pass no spec (the
        SCANNER and the DUPE stages, which have an f50 and no MTFSpec) keep the
        Gaussian, which is what they always had.
        Both laws are exactly 0.5 at f50, so a stock gaining a measured rolloff
        changes shape and NOT level. See fp.mtf_response.
        """
        if spec is not None and use_kernel:
            # The two-lobe separable form the C++ twins convolve. Falls back to
            # the law itself when the stock's q is not tabulated, exactly as the
            # C++ side falls back to its legacy Gaussian.
            t = fp.mtf_kernel_response(spec, channel,
                                       self.f_mm).astype(np.float32)
        elif spec is not None:
            t = fp.mtf_response(spec, channel, self.f_mm).astype(np.float32)
        else:
            t = np.exp(
                -math.log(2.0) * (self.f_mm / np.float32(f50_cpmm)) ** 2
            ).astype(np.float32)
        if adjacency > 0.0:
            # Band-pass, as a difference of two Gaussians. A plain unsharp term
            # (1 + a - a*G) would settle at 1 + a for all high frequencies,
            # i.e. a permanent global sharpening -- not an adjacency effect at
            # all. The real thing peaks at the inhibitor diffusion scale and
            # returns to unity at both DC and high frequency.
            lift = 1.0 + adjacency * (
                self.gaussian(adjacency_um * 0.4) - self.gaussian(adjacency_um * 2.0)
            )
            t = (t * lift).astype(np.float32)
        return t

    def multi_gaussian(
        self, radii_um: tuple[float, ...], weights: tuple[float, ...]
    ) -> np.ndarray:
        """Weighted sum of Gaussian transfers -- a long-tailed scatter kernel.

        A single Gaussian gives a tight halo. Real halation has a faint bloom
        reaching far beyond it, and that wide low-amplitude tail is the part the
        eye reads as photochemical.
        """
        wsum = float(sum(weights))
        acc = np.zeros_like(self.f_mm)
        for r, wt in zip(radii_um, weights):
            acc += np.float32(wt / wsum) * self.gaussian(r)
        return acc.astype(np.float32)

    def grain_shape(self, clump_um: float, clump_gain: float) -> np.ndarray:
        """Isotropic grain AMPLITUDE transfer, not the power spectrum.

        ⚠ THE NAME OF THIS QUANTITY WAS WRONG IN THIS DOCSTRING UNTIL 2026-08-24,
        AND THE ERROR WAS LOAD-BEARING. It read "power-spectrum shape (Wiener
        spectrum surrogate)". It is not: the return value is used as an AMPLITUDE
        filter -- `make_grain_field` does `apply_transfer(white, shape_t)`, which
        multiplies the FFT of the noise, and `grain_reference_energy` integrates
        `(h * a) ** 2`, squaring it. So the field's power (Wiener) spectrum is
        this function SQUARED.
            Why it mattered: fitting a measured Wiener spectrum through the wrong
        reading gives a clump_um off by exactly sqrt(2). On ILFORD_HPS the BBC
        Monograph 54 Fig. 8 trace fits 1.90 um under the correct amplitude
        reading and 2.69 um under the mislabelled one. The code was always
        self-consistent; only this comment lied, which is the worst case, because
        nothing fails and the number is quietly wrong.

        Two terms: a high-frequency rolloff set by the mean developed clump
        size, and an extra low-frequency lobe whose amplitude is the clumping
        tendency. Cubic crystals cluster strongly, tabular T-grain crystals lie
        flat and pack evenly.
            ⚠ THE LOW-FREQUENCY LOBE IS NOW KNOWN TO BE ABSENT ON AT LEAST ONE
        REAL EMULSION. A free two-parameter fit to the measured HPS Wiener
        spectrum (268 traced points) drives clump_gain to exactly 0.000, and BBC
        Report T-101 p38 says the same thing in words: grain correlation is
        "substantially confined to about plus or minus one equivalent grain
        diameter", with only small components outside. A low-frequency lobe IS
        long-range correlation, and the document states there is none. The other
        158 stocks keep their estimated clump_gain; see the ILFORD_HPS profile.

        Relationship to the stored parameters:

            f_hi = 1000 / (2 * clump_um)        cycles/mm, amplitude 1/e point
            f_lo = f_hi / 6
            h(f) = exp(-(f/f_hi)^2) * (1 + clump_gain * exp(-(f/f_lo)^2))
            W(f) / W(0) = (h(f) / h(0)) ** 2    <-- compare THIS to a datasheet

        The DC bin is zeroed so grain has exactly zero mean and cannot shift
        overall exposure.
        """
        f_hi = 1000.0 / (2.0 * clump_um)  # cycles/mm
        f_lo = f_hi / 6.0
        t = np.exp(-((self.f_mm / np.float32(f_hi)) ** 2)).astype(np.float32)
        if clump_gain > 0.0:
            t = t * (
                1.0
                + np.float32(clump_gain)
                * np.exp(-((self.f_mm / np.float32(f_lo)) ** 2))
            ).astype(np.float32)
        t[0, 0] = 0.0
        return t.astype(np.float32)

    def shift(self, dy_px: float, dx_px: float) -> np.ndarray:
        """Sub-pixel translation as a phase ramp. Exact, no resampling loss."""
        phase = -2.0 * math.pi * (self.fy_cpp * dy_px + self.fx_cpp * dx_px)
        return np.exp(1j * phase).astype(np.complex64)


#: The channels a CC filter attenuates. A CC filter is named for the colour it
#: IS, and it absorbs the complement: CC15B is blue, so it removes red and
#: green. Additive letters attenuate the other two records, subtractive letters
#: attenuate one.
_CC_ATTENUATES = {
    "R": (1, 2), "G": (0, 2), "B": (0, 1),      # additive
    "C": (0,),   "M": (1,),   "Y": (2,),        # subtractive
}


def _cc_filter_shift(text: str) -> tuple[float, float, float]:
    """Per-channel CREDIT against the printed correction that a CC filter implies.

    ⚠ THIS IS AN INTERPRETATION OF A PRESCRIPTION, AND THE ARITHMETIC IS WHAT
    ACTUALLY REACHES THE FILM. A datasheet does not print "the blue record loses
    0.15 more than the others"; it prints "at 10 s, increase exposure 1 1/2 stops
    and use a CC15B filter". Both instructions act on the same frame: the lens
    opens by the stated stops -- equally on all three records -- and the filter
    then takes part of that back from the records it attenuates. A CC15B is blue,
    so it absorbs red and green by 0.15 density each. What the film receives is
    +1.5 stops of blue and +1.5 stops - 0.15 decades of red and green, and since
    the prescription is what makes the result correct, those ARE the losses:

        the record the filter does NOT attenuate loses the full printed stops;
        every attenuated record loses that MINUS the filter's density.

    So the return is a per-channel credit (>= 0), not a deficit. Getting it
    backwards inflates the worst record by the filter's own value -- 1/3 stop for
    any CC10 -- while leaving the channel ORDERING correct, which is the kind of
    error that still looks plausible in a frame. It WAS written that way first,
    and the 5205 sheet is what caught it: "+2/3 stop and a CC10R" has to come out
    as 2/3 stop on red, not 1 stop.

    CC values are already in DENSITY, i.e. base-ten log exposure -- the same
    unit the curve is indexed in -- so no stops conversion happens here and none
    should: converting to stops and back is where a factor of 0.30103 gets lost.

    Nothing is re-referenced afterwards, and that falls out right for a compound
    prescription: a set attenuating all three records (10Y + 10M + 10C) is a
    neutral-density filter, and crediting all three equally is what an ND in the
    light path does.

    An empty or unparseable string gives (0, 0, 0) -- the ACHROMATIC case, which
    is a statement and not a missing measurement.
    """
    out = [0.0, 0.0, 0.0]
    if not text:
        return (0.0, 0.0, 0.0)
    s = text.upper()
    i = 0
    while i < len(s):
        if not s[i].isdigit():
            i += 1
            continue
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        if j >= len(s) or s[j] not in _CC_ATTENUATES:
            i = j + 1
            continue
        # ⚠ A THREE-DIGIT CC CODE IS THOUSANDTHS, NOT HUNDREDTHS, AND READING
        # IT WRONG IS A FACTOR OF TEN THAT FLIPS THE SIGN OF THE RESULT.
        # Every CC code in the corpus was two digits until 2026-09-01, when
        # AGFA_RSX_II_200 arrived with "075 Y" printed on agfa_films.pdf p6 --
        # a CC7.5Y, i.e. 0.075 density. Read as 75/100 it became a 0.75-density
        # filter, its blue credit swamped the 1-stop printed correction, and
        # `reciprocity_log_shift` returned +0.449 for blue: a LONGER exposure
        # making the film FASTER, which no sensitometry supports. verify.py's
        # "reciprocity never increases effective exposure" guard caught it.
        # The give-away inside the data is monotonicity: that row runs
        # 0 -> 075Y -> 15Y+05C, and 0.075 -> 0.15 ascends while 0.75 -> 0.15
        # does not.
        # ⚠ BIT-FOR-BIT INERT FOR EVERY STOCK THAT EXISTED BEFORE IT. CC075Y
        # is the only three-digit code in the database; the other nine distinct
        # codes are two-digit and take the unchanged branch.
        _digits = s[i:j]
        dens_cc = (int(_digits) / 1000.0
                   if len(_digits) == 3 and _digits[0] == "0"
                   else int(_digits) / 100.0)
        for c in _CC_ATTENUATES[s[j]]:
            out[c] += dens_cc
        i = j + 1
    return tuple(out)                  # type: ignore[return-value]


def resolve_process_variant(profile, variant):
    """The profile as a chosen PROCESS renders it. Returns `profile` unchanged
    when nothing is chosen, so this is inert by default.

    ⚠ WHAT A VARIANT IS AND WHAT IT IS NOT. `ProcessVariant` records a
    DIFFERENT DEVELOPMENT of the same emulsion -- a push, a cross-process, an
    alternate kit -- and where the manufacturer plotted that development
    separately the record carries its own traced ToneCurve set. Selecting one
    is therefore not a tweak to the stored curve: it is a different measured
    curve for the same film, which is exactly why it is applied by REPLACING
    the profile's curves rather than by scaling them.

    ⚠ 24 VARIANTS EXIST ACROSS 6 STOCKS AND ONLY 5 OF THEM CHANGE A PIXEL.
    Four carry their own curves -- PORTRA 800 at EI 1600 and EI 3200, and
    ULTRA COLOR 400UC as E-190 prints it and at EI 800 -- and CINESTILL 800T's
    Cs2 two-bath kit carries `gamma_scale` 0.879. The other nineteen differ
    only in `exposure_index`, which no renderer reads, so selecting one of
    those is a no-op and is left as one rather than being given an invented
    effect. The AGFAPAN developer variants are the whole of that nineteen:
    Agfa print an exposure index per developer and no second curve.

    ⚠⚠ THE SELECTION IS AN ENUMERATOR AND NO LONGER AN INDEX, 2026-09-17.
    `variant` is a `ProcessVariantCtrl` value -- or the integer behind one --
    and it names a DEVELOPMENT, globally. It used to be a position in this
    profile's own `process_variants` tuple, and position is not identity: the
    stored value 2 meant "RODINAL 1+50" on an AGFAPAN, "ECN-2, the base
    stock's native process" on CINESTILL 800T and "EI 3200 (Push 2)" on
    PORTRA 800. Every one of those was in range, so nothing could tell a stale
    selection from a correct one, and inserting a variant re-pointed every
    saved project silently.

    ⚠ A VALUE THIS STOCK DOES NOT OFFER RETURNS THE PROFILE UNTOUCHED, which
    is the same inert path an unselected control takes. Clamping into range
    would render a different development and present it as the one asked for.

    ⚠ AND THE ENUMERATION LIVES IN `AlgoControlEnums.hpp`, not here. The
    database mirrors it in `_PROCESS_VARIANT_IDS` and `verify.py` refuses the
    build if the two disagree, so there is exactly one list of developments in
    the project and this function reads it rather than restating it.
    """
    if variant is None:
        return profile
    key = process_variant_key(variant)
    if not key:
        return profile
    v = None
    for q in getattr(profile, "process_variants", ()) or ():
        if getattr(q, "variant_id", "") == key:
            v = q
            break
    if v is None:
        return profile

    curves = profile.curves
    if getattr(v, "curves", None) is not None:
        curves = v.curves
    elif v.gamma_scale != 1.0 or v.dmin_shift != 0.0:
        # ⚠ THE COEFFICIENT IS SCALED, NOT THE OBSERVABLE SLOPE, and the record
        # means the coefficient: `ProcessVariant.gamma_scale` is documented as
        # multiplying `ToneCurve.gamma`, which is the model parameter. On a
        # curve whose knees are far apart the two are the same number to within
        # a per cent; where they are not, the variant that cares carries its own
        # curves instead and never reaches this branch.
        curves = RGBCurves(
            *[replace(c,
                      gamma=c.gamma * v.gamma_scale,
                      dmin=c.dmin + v.dmin_shift)
              for c in profile.curves.as_tuple()]
        )
    if curves is profile.curves and not v.exposure_index:
        return profile
    ei = v.exposure_index or profile.exposure_index
    return replace(profile, curves=curves, exposure_index=ei)


def development_family(profile):
    """The one coherent (developer, dilution, vessel, edition) group of
    development points this stock's stored curve actually sits on, or None.

    ⚠⚠ A `ProcessingFamily` IS NOT ONE CURVE AND TREATING IT AS ONE IS THE
    DEFECT THIS FUNCTION EXISTS TO PREVENT. The tuple is flat by design -- the
    struct's own docstring says so, to keep the C++ emitter a plain array --
    and after the 1956 harvest it can hold, on a single stock, two developers
    at two dilutions in two vessels across two emulsion GENERATIONS. Reading
    gamma against time straight off that tuple would interpolate between a
    1956 roll film in a small tank and a 2016 sheet in a tray and call the
    result a development curve. `vessel` was added at v28 and `edition` at v35
    precisely so the groups can be told apart; this is the consumer that uses
    them.

    THE GROUP IS CHOSEN, NOT GUESSED:
      1. only points carrying a real gamma are eligible -- a time-only point
         states a temperature, not a contrast, and cannot place a curve;
      2. groups are keyed on all four discriminants;
      3. a group whose developer matches `profile.processing.developer` wins,
         because that is the developer the STORED CURVE was measured in and
         the whole operation is "move along the axis this curve sits on";
      4. failing that, the largest group wins, and only if it is unambiguous.

    Returns a sorted tuple of (minutes, gamma), or None.
    """
    fam = getattr(profile, "processing_family", None)
    if fam is None or not fam.points:
        return None
    groups: dict[tuple, list] = {}
    for q in fam.points:
        if q.gamma <= 0.0:
            continue
        groups.setdefault(
            (q.developer, q.dilution, q.vessel, getattr(q, "edition", "")),
            []).append((float(q.minutes), float(q.gamma)))
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return None

    want = (getattr(profile.processing, "developer", "") or "").strip().lower()
    if want:
        named = {k: v for k, v in groups.items()
                 if k[0].strip().lower() == want}
        if named:
            groups = named
    else:
        # ⚠ FAILING THAT, THE FAMILY'S OWN REFERENCE DEVELOPER (schema v36).
        # `ProcessingSpec.developer` describes the stored curve and is empty on
        # every 1956-sourced stock, because the stored curve is a later sheet.
        # `ProcessingFamily.reference_developer` describes the SOURCE's own
        # characteristic curve, which is the curve these points were drawn
        # beside, and Kodak prints it in the caption. Without it SUPER-XX PAN
        # ties seventeen DK-50 points against seventeen DK-60a and refuses.
        ref = (getattr(fam, "reference_developer", "") or "").strip().lower()
        rdil = (getattr(fam, "reference_dilution", "") or "").strip().lower()
        if ref:
            named = {k: v for k, v in groups.items()
                     if k[0].strip().lower() == ref
                     and (not rdil or k[1].strip().lower() == rdil)}
            if named:
                groups = named
    best = max(groups.values(), key=len)
    # ⚠ A TIE IS A REFUSAL, NOT A COIN TOSS. Two equally large groups for the
    # same developer are two measurements of different processes, and picking
    # one by dictionary order would make the render depend on insertion order.
    if sum(1 for v in groups.values() if len(v) == len(best)) > 1:
        return None
    return tuple(sorted(best))


def development_gamma_scale(profile, minutes: float) -> float:
    """Ratio by which a development time moves this stock's gamma. 1.0 = no-op.

    ⚠ A RATIO AND NOT AN ABSOLUTE GAMMA, because the stored curve is already a
    development and the control moves ALONG that axis rather than replacing it.
    The reference point is the family's gamma at `profile.processing.minutes`,
    the condition the stored curves were measured at, so asking for that time
    returns exactly 1.0 and reproduces every earlier render bit for bit. Using
    the family's absolute gamma instead would silently re-level every stock
    whose traced family and stored curve disagree slightly, which is a
    different and unwanted change.

    ⚠ REFUSES OUTSIDE THE TRACED RANGE RATHER THAN CLAMPING, and the control's
    own contract in AlgoControlEnums.hpp says so: extrapolating a development
    family is not a measurement. A request outside the range, a stock with no
    usable family, and a reference time that is itself outside the range all
    return 1.0 -- the sentinel result.

    ⚠ MONOCHROME ONLY, AND THIS IS A DATA STATEMENT RATHER THAN A CONVENIENCE.
    A single scale applied to three records asserts that the three layers move
    together under development, which this project has measured to be false --
    PORTRA 800 pushed to EI 3200 gains 0.25 of gamma in red against 0.14 in
    blue. Every gamma-bearing family in the database is monochrome except
    ORWOCOLOR NC 3, whose seven points are one channel's worth of data and
    cannot place three curves. A per-channel family would lift this.
    """
    if minutes is None or minutes < 0.0:
        return 1.0
    if not getattr(profile, "is_monochrome", False):
        return 1.0
    pts = development_family(profile)
    if not pts:
        return 1.0
    lo, hi = pts[0][0], pts[-1][0]
    if not (lo <= minutes <= hi):
        return 1.0

    def at(t: float) -> float:
        for (t0, g0), (t1, g1) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                if t1 == t0:
                    return g0
                return g0 + (g1 - g0) * (t - t0) / (t1 - t0)
        return pts[-1][1]

    # ⚠⚠ WHERE THE REFERENCE TIME COMES FROM, AND WHY IT IS NOT SIMPLY
    # `processing.minutes`. That field is the obvious anchor and it is EMPTY on
    # six of the eleven stocks that have a usable family -- including all four
    # the 1956 harvest gave a family to -- so requiring it would leave the
    # control inert on the majority of the data that exists to drive it.
    #
    # The honest fallback is an INVERSION rather than a substitution: the
    # stored curve has a gamma, the family says which development time
    # produces that gamma, and that time IS "the development the stored curves
    # represent" -- which is the sentinel's own definition, quoted from
    # AlgoControlEnums.hpp. Solving for it uses only measured numbers and
    # assumes nothing about which edition the stored curve came from, because
    # the quantity actually used downstream is the RATIO, and a ratio anchored
    # at the stored gamma returns exactly 1.0 there by construction.
    #
    # ⚠ AND IT REFUSES WHEN THE STORED GAMMA IS OUTSIDE THE FAMILY'S RANGE,
    # which is the case that says the two describe different emulsions. There
    # is no defensible reference then, and inventing one would place the whole
    # curve on a family it does not belong to.
    ref = float(getattr(profile.processing, "minutes", 0.0) or 0.0)
    if not (lo <= ref <= hi):
        g_stored = float(profile.curves.g.gamma)
        g_lo, g_hi = pts[0][1], pts[-1][1]
        if not (min(g_lo, g_hi) <= g_stored <= max(g_lo, g_hi)):
            return 1.0
        ref = None
        for (t0, g0), (t1, g1) in zip(pts, pts[1:]):
            if min(g0, g1) <= g_stored <= max(g0, g1):
                ref = (t0 if g1 == g0
                       else t0 + (t1 - t0) * (g_stored - g0) / (g1 - g0))
                break
        if ref is None:
            return 1.0

    g_ref = at(ref)
    if g_ref <= 0.0:
        return 1.0
    return at(minutes) / g_ref


def development_ref_celsius(profile) -> float:
    """The temperature the family's own points were measured at most often.

    `ProcessingSpec.celsius` wins when the source states one; otherwise the
    modal temperature of the family, which is what an equal-contrast table's
    own reference column is.
    """
    fam = getattr(profile, 'processing_family', None)
    spec = getattr(profile, 'processing', None)
    if spec is not None and getattr(spec, 'celsius', 0.0) > 0.0:
        return float(spec.celsius)
    if fam is None or not fam.points:
        return 0.0
    counts: dict[float, int] = {}
    for q in fam.points:
        if q.celsius > 0.0:
            counts[q.celsius] = counts.get(q.celsius, 0) + 1
    if not counts:
        return 0.0
    return float(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def development_equivalent_minutes(profile, minutes: float,
                                   celsius: float) -> float:
    """`minutes` at `celsius`, restated at the family's reference temperature.

    ⚠ THE LAW IS THE STOCK'S OWN. `ProcessingFamily.temperature_coeff_per_c`
    is fitted to that stock's points and is 0.0 wherever the family holds one
    temperature, so this returns `minutes` unchanged on every stock that has
    no measured slope. It never substitutes the population median, because the
    eleven fitted stocks span x0.33 to x0.50 of the time per +10 degC and a
    middle value would be wrong for both ends by a quarter of the effect.
    """
    import math
    fam = getattr(profile, 'processing_family', None)
    if fam is None or minutes <= 0.0 or celsius <= 0.0:
        return minutes
    c = float(getattr(fam, 'temperature_coeff_per_c', 0.0) or 0.0)
    if c >= 0.0:
        return minutes
    ref = development_ref_celsius(profile)
    if ref <= 0.0 or ref == celsius:
        return minutes
    return minutes * math.exp(c * (ref - celsius))


def resolve_development_time(profile, minutes: float, celsius: float = -1.0):
    """The profile as a chosen DEVELOPMENT TIME renders it (queue P61b).

    Returns `profile` itself on the sentinel path, so the identity test that
    guards `resolve_process_variant` guards this too.

    ⚠ THE SCALE MULTIPLIES `ToneCurve.gamma`, THE MODEL COEFFICIENT, exactly as
    `ProcessVariant.gamma_scale` does -- the same choice, made for the same
    reason, and stated here so the two cannot drift apart. `dmin` is NOT
    touched: base fog does move with development time, and
    `DevelopmentPoint.base_fog` exists to hold it, but it is populated on one
    stock in the database and a relation fitted to one stock is not a relation.
    """
    k = development_gamma_scale(
        profile, development_equivalent_minutes(profile, minutes, celsius))
    if k == 1.0:
        return profile
    curves = RGBCurves(*[replace(c, gamma=c.gamma * k)
                         for c in profile.curves.as_tuple()])
    return replace(profile, curves=curves)


def dark_fade_fractions(profile, years: float) -> tuple[float, float, float]:
    """Fraction of each image dye lost to DARK STORAGE after `years`.

    Returns (cyan, magenta, yellow), each 0.0 when the record states no rate
    for that dye -- which today is two of the three on both stocks that carry
    a rate at all.

    ⚠ THE LAW IS FIRST ORDER AND THE SOURCE DEFINES IT THAT WAY. Both
    `DyeStabilitySpec` sources state a time to a 10 % loss from a starting
    density of 1.0, which is a statement about a constant fractional rate: a
    dye losing a tenth of what remains in T years has lost 1 - 0.9**(t/T)
    after t. Nothing is fitted and nothing is extrapolated beyond restating
    the published figure as a function of time.

    ⚠ ONLY THE DYE THE SOURCE NAMES IS FADED, and this is the honest half.
    Wilhelm's Table 19.1 publishes the time for the LEAST STABLE dye and names
    it -- yellow, on both KODAK_EKTAR_125 and KODAK_VERICOLOR_III_160 -- and
    says nothing about the other two. Fading all three at the same rate would
    assert an equality the source explicitly contradicts: its whole subject is
    that "one of the three image dyes -- usually magenta -- is much more stable
    in dark fading than is the least stable dye, and this differential in
    fading rates results in increasingly objectionable color shifts". A
    differential fade is the effect; a uniform one would be a density change
    wearing its costume.

    ⚠ AT THE RECORD'S OWN REFERENCE TEMPERATURE, WITH NO TEMPERATURE CONTROL.
    The source quotes 24 degC and gives factors for two refrigerator
    temperatures -- about 14x longer at 4.4 degC and 20x at 1.7 degC -- but
    three points do not define a continuous law and this project does not fit
    one to invent the values between them. The factors are in the provenance
    string for whoever does.
    """
    spec = getattr(profile, "dye_stability", None)
    if spec is None or not spec.has_data or years is None or years <= 0.0:
        return (0.0, 0.0, 0.0)
    out = []
    for t in (spec.loss_c, spec.loss_m, spec.loss_y):
        if t <= 0.0:
            out.append(0.0)
        else:
            out.append(1.0 - 0.9 ** (float(years) / float(t)))
    return tuple(out)


def resolve_storage_age(profile, years: float):
    """The film as `years` of dark storage leave it (queue P64).

    Returns `profile` itself when nothing fades, so the identity contract that
    guards `resolve_process_variant` and `resolve_development_time` guards this
    one too.

    ⚠ WHICH DYE SITS IN WHICH RECORD. On a chromogenic negative the cyan dye
    forms in the red-sensitive layer, magenta in the green and yellow in the
    blue, so a dye's fade is read in that record and in no other. Losing a
    fraction f of a dye scales the density that dye contributes by (1 - f),
    which on the stored model is a scale on `ToneCurve.gamma`.

    ⚠ `dmin` IS DELIBERATELY NOT TOUCHED, AND THIS IS A REFUSAL RATHER THAN AN
    OMISSION. On a masked negative part of D-min is the orange mask, which is
    dye and does fade, and part is the support, which does not. The schema
    stores their SUM and nothing in the corpus separates them, so any dmin
    change here would be a guess at that split applied to every stock. The
    visible consequence is that a faded negative rendered here loses image dye
    and keeps its mask; the real one loses some mask too.
    """
    f = dark_fade_fractions(profile, years)
    if f == (0.0, 0.0, 0.0):
        return profile
    cs = profile.curves.as_tuple()
    return replace(profile, curves=RGBCurves(
        *[replace(c, gamma=c.gamma * (1.0 - fi)) for c, fi in zip(cs, f)]))


def reciprocity_log_shift(profile, exposure_time_s: float) -> tuple[float, ...]:
    """Per-channel shift of log10 exposure from reciprocity failure.

    ⚠ INERT AT 0.0, AND THAT IS THE CONTRACT. ``exposure_time_s <= 0`` means
    "the caller did not state an exposure time", returns (0, 0, 0), and every
    render made before this stage existed is reproduced bit for bit. The same
    pattern the measured-flag fields use: a stage that has no measurement to
    stand on does nothing rather than guessing.

    Two data sources, tried in that order, because they are not the same claim:

    * ``ReciprocityTable`` (6 stocks) prints the manufacturer's OWN correction
      against time, in stops, optionally with the CC filter that documents
      chromatic failure. Interpolated in log10 t -- the axis the tables are
      printed on -- and HELD FLAT outside the measured range rather than
      extrapolated. Kodak's tables walk the effective exponent from ~0.85 to
      ~0.70 across successive decades, so extrapolating one decade past the last
      entry is not a small error.
    * ``ReciprocitySpec`` (105 stocks) carries one Schwarzschild exponent per
      channel and an onset. E_eff = I * t^p, and metered exposure is H = I * t,
      so log10 H_eff - log10 H = (p - 1) * log10(t / onset) for t > onset.

    ⚠ WHAT THIS MODEL IS NOT. There is no intensity axis anywhere in the
    corpus: every one of the six measured tables is a function of TIME alone.
    Real reciprocity failure is intensity dependent -- the dark parts of a frame
    fail first, which is why a long exposure loses shadow separation as well as
    speed -- and nothing on file can calibrate that. So this is a per-channel
    GLOBAL shift, honest about being one, rather than a per-pixel shadow effect
    with an invented exponent. Stated in the docs and in the queue entry.

    HIRF as well as LIRF: EKTACHROME_64's table starts at 1e-4 s with a 0.5-stop
    correction, so the interpolation is deliberately two-sided. Only that one
    stock measures the short-exposure branch; for everything else a flash
    duration lands on the held-flat first entry, which is why the branch is not
    extrapolated.
    """
    t = float(exposure_time_s)
    if t <= 0.0:
        return (0.0, 0.0, 0.0)

    tab = getattr(profile, "reciprocity_table", None)
    if tab is not None and tab.has_data:
        lt = math.log10(t)
        xs = [math.log10(v) for v in tab.times_s]
        ys = list(tab.stops_correction)
        ccs = list(tab.cc_filters) + [""] * (len(xs) - len(tab.cc_filters))
        chrom = [_cc_filter_shift(c) for c in ccs]
        if lt <= xs[0]:
            stops, ch = ys[0], chrom[0]
        elif lt >= xs[-1]:
            stops, ch = ys[-1], chrom[-1]
        else:
            k = 0
            while k + 1 < len(xs) and xs[k + 1] < lt:
                k += 1
            span = xs[k + 1] - xs[k]
            f = 0.0 if span <= 0.0 else (lt - xs[k]) / span
            stops = ys[k] + f * (ys[k + 1] - ys[k])
            ch = tuple(chrom[k][c] + f * (chrom[k + 1][c] - chrom[k][c])
                       for c in range(3))
        base = -0.30102999566398120 * stops
        return tuple(base + ch[c] for c in range(3))

    rp = profile.reciprocity
    onset = rp.onset_s if rp.onset_s > 0.0 else 1.0
    if t <= onset:
        return (0.0, 0.0, 0.0)
    lr = math.log10(t / onset)
    return (
        (rp.schwarzschild_p_r - 1.0) * lr,
        (rp.schwarzschild_p_g - 1.0) * lr,
        (rp.schwarzschild_p_b - 1.0) * lr,
    )


def callier_density(dens, curves, callier_q, specular, is_monochrome):
    """Stage 12b: the density a SPECULAR reader sees. In place on `dens`.

    Callier's coefficient Q is the ratio of specular to diffuse density for the
    same sample. It is a SILVER-SCATTERING effect: developed silver grains
    scatter the measuring beam out of a condenser system's acceptance angle, so a
    directed source reads a higher density than an integrating sphere does, and
    the whole tone scale steepens by that factor. A chromogenic dye image
    scatters almost nothing, which is why every colour stock in this file carries
    Q = 1.0 and is untouched here at any setting.

    ⚠ THE FIELD WAS WRONG IN SHAPE, WHICH IS WHY THIS TAKES TWO INPUTS (C22).
    `callier_q` sat on the profile as if Q were a property of the FILM. It is a
    property of film x MEASURING GEOMETRY: the same negative on a diffuse LED
    integrating-sphere scanner and on a directed halogen condenser reads two
    different densities. So the film contributes its scattering (Q) and the
    reader contributes how directional it is (`specular`, 0 = fully diffuse,
    1 = fully condenser), and neither alone is the answer.

        10**-(D_read - dmin) = E*10**-(D-dmin) + (1-E)*10**-(Q*(D-dmin))

    with E = 1 - specular. Silberstein & Tuttle, via Mees printed p644 -- see
    `callier_net`, which is the one definition of the law and which builds the
    table this stage reads.

    ⚠ THE LINEAR FORM THAT USED TO BE HERE, `dmin + (D-dmin)*(1+s*(Q-1))`, WAS
    RIGHT AT BOTH ENDS AND WRONG IN BETWEEN, by up to 0.21 D at the settings a
    user actually dials. It interpolated the multiplier; light interpolates
    transmittance. Replaced 2026-08-30, queue M3.

    ⚠ REFERENCED TO dmin, NOT TO ZERO, and that is the physics rather than a
    convenience: the scattering scales with the amount of developed silver, so
    the base carries none of it. Scaling absolute density instead would make a
    condenser darken clear base, which no densitometer measures.

    INERT AT specular = 0, exactly, for every stock -- and inert at ANY setting
    for the 93 colour stocks. That default is the one that reproduces every
    render made before this stage existed.

    ⚠ THE FILM HALF OF THE PRODUCT IS STILL UNSOURCED, and turning `specular` up
    is what makes it visible. The two monochrome values (1.3 negative, 1.25
    reversal) come from `_apply_schema_v2`'s class rule, not from a document;
    what would fix that is one densitometer specification stating a
    diffuse-versus-specular ratio for a named emulsion. Until then the geometry
    axis is exact and the film axis is a class estimate, which is why the control
    ships at 0.
    """
    if specular <= 0.0 or callier_q == 1.0:
        return dens
    lut = callier_lut(_QOnly(callier_q), specular)
    lo, step, vals = lut
    v32 = vals.astype(np.float32)
    n = len(v32)
    hi_x = np.float32(lo + step * (n - 1))
    slope_lo = np.float32((v32[1] - v32[0]) / step)
    slope_hi = np.float32((v32[n - 1] - v32[n - 2]) / step)
    for c in range(3):
        dmin = np.float32(curves[c].dmin)
        # Net density, including the NEGATIVE values grain produces in the base.
        # The law is defined there and needs no branch, which is what keeps the
        # base case exact rather than clamped into a second code path.
        net = dens[:, :, c] - dmin
        t = (net - np.float32(lo)) / np.float32(step)
        i = np.clip(np.floor(t).astype(np.int32), 0, n - 2)
        out = v32[i] + (v32[i + 1] - v32[i]) * (t - i.astype(np.float32))
        out = np.where(t < np.float32(0.0),
                       v32[0] + (net - np.float32(lo)) * slope_lo, out)
        out = np.where(t > np.float32(n - 1),
                       v32[n - 1] + (net - hi_x) * slope_hi, out)
        dens[:, :, c] = out.astype(np.float32) + dmin
    np.maximum(dens, np.float32(0.0), out=dens)
    return dens


def apply_interimage(dens, curves_or_log_e, curves, iie, anchors, reversal):
    """Stage 8b, the VERTICAL half of the DIR chemistry. In place on `dens`.

    ⚠ FACTORED OUT OF `simulate()` ON 2026-08-20 SO THERE IS ONE DEFINITION.
    The C++ port implements the same law in `AlgoStage08b_Interimage`
    (Algo_08_Sim.cpp), and nothing checked the two agreed -- `cpp_parity.py`
    covered the grain and MTF laws only. That is precisely the configuration
    that produced the C1b calling-convention bug: one law, two languages, a
    manual one-off cross-check that guarded nothing. `interimage_parity.py`
    now probes THIS function against THAT one, so the law has to live in a
    function rather than inside the pipeline.
    Byte-for-byte the same arithmetic and the same float32 casts as the inline
    version it replaces -- verified by rendering before and after.

        logE_i' = logE_i + sum_{j != i} a_ij * (D_j - d_ref_j)

    `curves_or_log_e` is the log-exposure array (named for the positional
    signature the C++ side mirrors: source density, source log exposure).
    """
    log_e = curves_or_log_e
    h, w = dens.shape[0], dens.shape[1]
    m = iie.matrix()
    # Density each layer reaches at the mid-grey anchor: the reference the
    # correction is measured from.
    if reversal:
        d_ref = [float(density_scalar(-float(anchors[c]), curves[c]))
                 for c in range(3)]
    else:
        d_ref = [float(density_scalar(0.0, curves[c])) for c in range(3)]
    # density_weighting: 0 = uniform coupling across the curve (negative
    # film, chromogenic development); >0 concentrates it where the
    # neighbouring layer is DENSE (reversal film, whose effects come from
    # iodide released in the first B&W developer and land in high
    # dye-density areas). Weighting is normalised at the mid-grey
    # reference so a neutral stays untouched either way -- that property
    # is the whole point of the stage and must survive the mechanism split.
    dw = float(iie.density_weighting)
    # ---- THE BOUND, queue item C18, closed 2026-09-02 ---------------------
    # ⚠ THE WEIGHT USED TO BE UNBOUNDED, AND THAT WAS THE WHOLE OF C18. The
    # weight is (1 - dw) + dw * D_j / D_ref, so it grows without limit in the
    # density it is handed, and `density_weighting` is the largest undocumented
    # number in the colour path -- 0.65 on 36 reversal stocks, magnitude tier 3,
    # worst-case correction -0.58 logE on VELVIA 50.
    #
    # ⚠ THE CAP IS THE STOCK'S OWN Dmax AND IT IS PROVABLY INERT, which is why
    # it can land without the D1/D2 measurement C18 was waiting for.
    # `ToneCurve.dmax` is not a stored guess: it is exactly the asymptote
    # dmin + gamma*(shoulder_x - toe_x), and `density()` is dmin + gamma times a
    # DIFFERENCE OF TWO SOFTPLUS RAMPS, which is strictly increasing and bounded
    # above by (shoulder_x - toe_x). So D_j < dmax for every finite log E, on
    # both the negative and the reversal branch, and this minimum can never
    # bind. Every render is bit-for-bit what it was; what changes is that the
    # expression is now bounded by construction instead of by the accident that
    # nothing upstream has ever handed it a density above Dmax.
    #
    # What it does NOT do is supply the saturating FORM with a measured
    # asymptote that C18 actually wants. That still needs the wedge
    # measurement; a cap at Dmax is the honest bound available without one,
    # and inventing a saturation constant would have been the other kind of
    # answer. `verify.py` asserts the cap is non-binding across the database.
    delta = None
    for _ in range(int(iie.iterations)):
        delta = [dens[:, :, j] - np.float32(d_ref[j]) for j in range(3)]
        if dw > 0.0:
            for j in range(3):
                ref = max(d_ref[j], 1e-4)
                wj = (1.0 - dw) + dw * (dens[:, :, j] / np.float32(ref))
                # ⚠⚠ THE CAP IS 1.0, AND THE OLD ONE WAS A NO-OP. CORRECTED
                # 2026-09-09 after the owner rendered a real frame.
                #
                # It used to be `(1 - dw) + dw * dmax / ref` -- the value the
                # weight reaches AT dmax. But `density()` approaches dmax only
                # asymptotically, so D < dmax for every finite exposure and
                # that minimum COULD NEVER BIND. The 2026-09-02 note calling
                # it "provably inert ... every render bit-for-bit what it was"
                # was describing a no-op as a fix. Queue item C18 asked for a
                # bound; what landed was a bound that cannot engage.
                #
                # ⚠ WHY THE UNBOUNDED FORM WAS WRONG, in one line: the weight
                # is linear in D_j and it MULTIPLIES (D_j - d_ref), so the
                # product is QUADRATIC in density --
                #     (D_j - Dr)*[(1-dw) + dw*D_j/Dr]
                #       = (1-dw)(D_j - Dr) + (dw/Dr)*D_j*(D_j - Dr)
                # US4729943A says the reversal effect "lands in high dye-
                # density areas", which is ONE factor of density; the
                # `- d_ref` exists only to keep a neutral untouched.
                # Multiplying them DOUBLE-COUNTS density. Measured on
                # FUJI_VELVIA_50: delta*w ran +0.009 at the calibration point
                # to +2.829 at dmax, i.e. -0.402 logE of one-channel shift.
                #
                # ⚠ WHY 1.0 IS PRINCIPLED AND NOT ARBITRARY. Capping at 1
                # turns the weighting from a GAIN into a REDISTRIBUTION: the
                # effect is at FULL strength at and above the reference
                # density and TAPERS BELOW it, so the stage still concentrates
                # where the neighbouring layer is dense -- which is what the
                # patent actually states -- while the stage's total strength
                # stays set by the coefficients, which are calibrated at that
                # reference. No constant is invented.
                #
                # ⚠ IT DELIBERATELY UNDER-MODELS. A convex-but-bounded weight
                # is the physically right shape and needs TWO constants (an
                # asymptote and a half-density) that no surveyed source
                # prints. That is C18's wedge measurement, still open. Until
                # it exists, under-modelling is the honest direction.
                #
                # Measured on the owner's frame, FUJI_VELVIA_50, table apron:
                #   before  R 242.8  G 29.1  B 0.0   R=255 on 15.42 % of frame
                #   after   R  70.1  G  4.6  B 0.0   R=255 on  7.92 %
                #   (interimage off: R 25.4;  original image: R 42.1)
                np.minimum(wj, np.float32(1.0), out=wj)
                delta[j] = delta[j] * wj.astype(np.float32)
        for c in range(3):
            adj = np.zeros((h, w), dtype=np.float32)
            for j in range(3):
                if j == c or m[c][j] == 0.0:
                    continue
                adj += np.float32(m[c][j]) * delta[j]
            # ⚠ REVERSAL SIGN CORRECTED 2026-09-08. `adj` is ADDED outside
            # the negation, not subtracted. It used to read `- adj`, with a
            # comment claiming inhibition "reduces development in both cases".
            # Three independent internal reasons say otherwise; no external
            # source was needed and no database value changed.
            #
            #  1. THE COEFFICIENTS WERE NEVER SOLVED FOR THIS BRANCH.
            #     `_iie_solve` in film_profiles.py picks every coefficient by
            #     driving `_iie_measure`, and that model evaluates ONLY
            #     `density(logE + adj)` -- it has no reversal branch and no
            #     density_weighting. Feeding its answer through
            #     `-(logE + anchor) - adj` delivered the opposite of what was
            #     solved for. FUJI_VELVIA_50 was solved to +42/+45/+25 % IIE
            #     and rendered -17.5/-17.9/-13.5 %. Sign inverted on 78/78
            #     channels across all 26 reversal stocks, no exceptions.
            #  2. IT WAS NOT DOING WHAT ITS OWN COMMENT CLAIMED. Every stored
            #     off-diagonal is NEGATIVE -- 26/26 reversal and 80/80
            #     negative, one convention throughout -- so `- adj` RAISED
            #     this layer's density where the neighbour was dense. That is
            #     enhancement, not inhibition. `+ adj` lowers it, which is
            #     what the comment always said the stage did.
            #  3. IT INVERTED THIS FUNCTION'S DOCSTRING. Measured end to end
            #     through simulate() on moderate-chroma patches, the stage
            #     raised saturation on 80/80 negatives and on 0/26 reversals.
            #     After the fix, 26/26 reversals raise it too.
            #
            # Cost, measured: median |delta| 0.0205 in linear light over the
            # 26 reversal stocks, worst pixel 0.9998 on FUJI_PROVIA_400F.
            # ⚠ 5 STOCKS NOW OVERSHOOT their solved target because
            # `_iie_measure` still ignores density_weighting (0.65 on
            # reversal): FUJICHROME_64T_II, FUJI_PROVIA_100F,
            # FUJI_PROVIA_400F, KODAK_EKTACHROME_100D_5285,
            # SUPER_ANSCOCHROME_1957. Recorded as an open gap in NotFound.md
            # under owner instruction to leave the stored values untouched.
            if reversal:
                dens[:, :, c] = density(
                    -(log_e[:, :, c] + np.float32(anchors[c])) + adj,
                    curves[c],
                )
            else:
                dens[:, :, c] = density(log_e[:, :, c] + adj, curves[c])
    del delta
    return dens


def apply_dir_couplers(dens, cp, grid, coupler_scale, is_monochrome):
    """Stage 9, the LATERAL half of the DIR chemistry. In place on `dens`.

    Factored out for the same reason as `apply_interimage` -- the C++ port is
    `AlgoStage09_DirCoupler` (Algo_09_Sim.cpp) and the two were never compared.

    ⚠ THE TWO IMPLEMENTATIONS DO NOT USE THE SAME BLUR, and the parity check
    has to know it. Here the blur is an FFT multiply by the ANALYTIC Gaussian
    transfer; the C++ side is a separable spatial Gaussian with the kernel
    truncated at 4 sigma (`ALGO_BLUR_SIGMA_CUTOFF`), which drops about 6.3e-5
    of the kernel weight. Both wrap at the edges, so the comparison is valid --
    but it is valid to ~1e-4, not to machine precision. On a FLAT field any
    blur is the identity, so the pointwise algebra is exactly testable there
    and that is the case the parity probe pins hardest.
    """
    if not (cp.active and coupler_scale > 0.0):
        return dens
    s = cp.strength * coupler_scale
    e = cp.edge_strength * coupler_scale
    # ---- THE SUB-PIXEL GATE, ADDED 2026-08-25d (queue item C17) -------------
    # ⚠ THIS GATE EXISTED ON THE C++ SIDE ONLY, AND THAT WAS THE WHOLE DEFECT.
    # `AlgoDirCoupler.hpp` has carried ALGO_COUPLER_MIN_SIGMA_PX = 0.25 since it
    # was written, gating BOTH components (Algo_09_Sim.cpp:1018 and :1023); this
    # reference had no gate at all, so below the threshold the two renderers were
    # not approximating each other -- one ran the stage and the other did not.
    # The crossovers `interimage_parity.py` prints are not exotic scales: the
    # long term switches off below 3.1 px/mm (EASTMAN_5247_1974, radius 80 um)
    # and the edge term below 27.8 px/mm (KODACHROME_64, edge 9 um), and
    # 27.8 px/mm is a 35 mm frame about 670 px wide.
    # THE THRESHOLD IS ADOPTED, NOT CHOSEN. 0.25 px is what the shipped and
    # reviewed C++ constant says, and its stated reason holds identically here:
    # below a quarter pixel the discrete kernel has one significant tap, so the
    # pass is an identity. Taking the existing value makes this a pure PARITY
    # fix with no fidelity judgement folded into it.
    # ⚠ WHAT THIS DOES NOT SETTLE IS QUEUE ITEM C16. The two blurs are still
    # different FORMS -- analytic Gaussian transfer here, truncated separable
    # spatial kernel there -- and they agree to 6e-5 only above about 1.2 px,
    # diverging to 1.5e-1 at 0.4 px. Stored edge_um is 9-13 um, i.e. 0.36-0.60 px
    # at 40 px/mm, which is INSIDE that divergent band and ABOVE this gate. So
    # the gate removes the one-sided-stage defect and leaves the shared-threshold
    # VALUE (0.25 vs ~1.0 px, where the two forms converge) as C16's open
    # decision. Raising it here would change every render and is the owner's.
    _min_px = 0.25
    _radius_px = (cp.radius_um / 1000.0) * grid.px_per_mm
    _edge_px = (cp.edge_um / 1000.0) * grid.px_per_mm
    if _radius_px < _min_px:
        s = 0.0
    if _edge_px < _min_px:
        e = 0.0
    if s > 0.0 and not is_monochrome:
        dbar = dens.mean(axis=2)
        dbar_blur = apply_transfer(dbar, grid.gaussian(cp.radius_um))
        # Pushing each layer away from the locally-blurred mean raises
        # saturation without raising gamma -- the real DIR mechanism.
        for c in range(3):
            dens[:, :, c] += np.float32(s) * (dens[:, :, c] - dbar_blur)
        del dbar, dbar_blur
    if e > 0.0:
        edge_t = grid.gaussian(cp.edge_um)
        for c in range(3):
            blurred = apply_transfer(dens[:, :, c], edge_t)
            dens[:, :, c] += np.float32(e) * (dens[:, :, c] - blurred)
        del edge_t
    # ⚠ THE FLOOR BELONGS INSIDE THIS FUNCTION, and it was outside until
    # 2026-08-20. The C++ twin ends with MAX_VALUE(rO[x], ALGO_ZERO) and its own
    # comment calls it "a physical floor, not a display clamp, so it does not
    # violate the single-final-clamp rule". Python clamped one line LATER, in
    # simulate(), so the PIPELINES agreed and the FUNCTIONS did not -- which
    # nobody could see until interimage_parity.py compared the functions. It
    # showed up as a 0.26 D disagreement on Velvia: a reversal stock whose ramp
    # drives density negative, where the C++ side had already floored it and the
    # Python side had not yet. Rendering is unchanged (max(max(x,0),0) is
    # max(x,0)), and simulate()'s later clamp stays because it also guards the
    # stages between here and there.
    np.maximum(dens, np.float32(0.0), out=dens)
    return dens


#: Hard ceiling on the fraction of net density stage 9c may remove, whatever
#: `strength` and the accumulated restraint multiply out to.
#:
#: ⚠ IT IS A NUMERICAL FLOOR AND NOT A SECOND STRENGTH KNOB. Without it a
#: saturated streak could drive net density to exactly zero and, with the
#: multiply applied in float, slightly through it -- and the stage sits UPSTREAM
#: of the only clamp in the chain, so a negative net density would travel into
#: the scan MTF and be spread around by it before anything caught it.
#: `BromideDragSpec.validate` already refuses `strength` above 0.5, so on any
#: legal record this ceiling never binds; it exists so that the arithmetic
#: cannot, not so that the model can be tuned.
BROMIDE_DRAG_MAX_REMOVED = 0.95


def bromide_drag_alpha(length_mm: float, px_per_mm: float) -> float:
    """The one-pole coefficient for a 1/e decay of `length_mm` on the FILM.

    ⚠ THE POINT IS THAT THE RECORD IS IN MILLIMETRES AND THE FILTER IS IN
    PIXELS. A drag length stored in pixels would render a different physical
    streak at every output resolution and on every gauge, which is the bug this
    project keeps designing out -- every other spatial quantity in the database
    is in micrometres or cycles per millimetre for the same reason. One pixel
    step is 1/px_per_mm millimetres, so the per-step retention is
    exp(-pitch/length) and the stage's output is resolution-independent by
    construction rather than by calibration.
    """
    if length_mm <= 0.0 or px_per_mm <= 0.0:
        return 0.0
    return float(np.exp(-(1.0 / px_per_mm) / length_mm))


def apply_bromide_drag(dens, spec, dmin, dmax, px_per_mm, reversal):
    """Stage 9c -- directional restraint by transported development byproducts.

    Runs IN PLACE on an (h, w, 3) plane of ABSOLUTE density, between stage 9 and
    stage 10. Returns True if it did anything, False if the record is inert.

    THE MODEL, and every choice in it is forced by the physics rather than
    picked:

      1. ONE SOURCE FIELD FOR ALL THREE CHANNELS. The bromide from all three
         layers goes into the same developer, so the restraint a point suffers
         does not depend on which dye it was going to make. The source is the
         mean net density across the three records, normalised by the mean
         (dmax - dmin) so that it is a fraction of full development and not a
         density -- which is what makes `strength` a pure fraction and
         comparable between stocks.
      2. ⚠ INVERTED ON A REVERSAL FILM. The bromide comes from the silver the
         FIRST developer reduces, which on a reversal stock is the negative
         image. So the streaks trail the CLEAR areas of a slide and the DENSE
         areas of a negative. This is the easiest thing in the stage to get
         backwards and it is the one line that decides it.
      3. A ONE-SIDED EXPONENTIAL ALONG THE TRANSPORT AXIS, as a one-pole
         recursion. One-sided because the solution is carried in one direction
         and cannot restrain what the film has not reached yet; exponential
         because the loaded layer is continuously diluted and replenished as it
         travels, which is a first-order loss. The recursion costs two flops per
         pixel regardless of `length_mm`, where an explicit kernel of a
         centimetre-long tail would cost hundreds.
      4. MULTIPLICATIVE ON NET DENSITY, not subtractive. Restraint slows
         development; it cannot remove density that was never going to form. A
         subtractive term drives base+fog negative in the shadows of a streak,
         which is not a subtle error -- it inverts the effect there.

    ⚠ THE FRAME EDGE IS SEEDED, NOT ZEROED, and the alternative is visibly
    wrong. Real film has more film upstream of the frame, which released its own
    bromide; starting the recursion at zero asserts that the frame's leading
    edge entered clean developer and draws a bright band across it that no
    machine produces. `s` is therefore seeded with the edge row's own source
    value, i.e. a semi-infinite uniform upstream at that value, which makes a
    uniform field give s == e exactly and leaves no band at all.

    ⚠ WHAT THIS STAGE CANNOT KNOW is what was in the frames BEFORE this one. A
    real streak is fed by the whole preceding length of film, so a bright object
    in the previous frame trails into this one. That is a temporal coupling and
    this pipeline renders frames independently; the seed above is the best a
    single-frame model can do, and the residual is a real limitation rather than
    an approximation with a bound.
    """
    if not spec.has_data:
        return False
    ref = float(np.mean([dmax[c] - dmin[c] for c in range(3)]))
    if ref <= 0.0:
        return False
    a = bromide_drag_alpha(spec.length_mm, px_per_mm)
    if a <= 0.0:
        return False

    h = dens.shape[0]
    dmin_v = np.asarray(dmin, dtype=np.float32)
    e = (dens - dmin_v).mean(axis=2, dtype=np.float32) * np.float32(1.0 / ref)
    np.clip(e, np.float32(0.0), np.float32(1.0), out=e)
    if reversal:
        e = np.float32(1.0) - e

    af = np.float32(a)
    bf = np.float32(1.0 - a)
    s = np.empty_like(e)
    if spec.direction >= 0:
        s[0] = e[0]
        for y in range(1, h):
            s[y] = af * s[y - 1] + bf * e[y - 1]
    else:
        s[h - 1] = e[h - 1]
        for y in range(h - 2, -1, -1):
            s[y] = af * s[y + 1] + bf * e[y + 1]

    r = s * np.float32(spec.strength)
    np.clip(r, np.float32(0.0), np.float32(BROMIDE_DRAG_MAX_REMOVED), out=r)
    keep = (np.float32(1.0) - r)[:, :, None]
    dens -= dmin_v
    dens *= keep
    dens += dmin_v
    return True


def apply_transfer(plane: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    """Filter one 2D plane by a half-spectrum transfer function."""
    h, w = plane.shape
    spec = np.fft.rfft2(plane)
    spec *= transfer
    out = np.fft.irfft2(spec, s=(h, w))
    return out.astype(np.float32)


# ===========================================================================
# Reseau (additive colour filter grid)
# ===========================================================================
def build_reseau_mask(
    h: int, w: int, px_per_mm: float, spec: ReseauSpec
) -> tuple[np.ndarray, float]:
    """One-hot colour filter grid for an additive-colour stock.

    Dufaycolor's geometry: continuous red lines, with blue and green squares
    chequered between them, each colour taking roughly a third of the area.
    The pitch is physical (lines/mm on the film), so like everything else here
    it converts to pixels from the render width.

    Returns:
        ``(mask, pitch_px)`` where mask is (h, w, 3) and exactly one channel is
        1.0 at each pixel. ``pitch_px`` is returned so the caller can check the
        grid is actually resolvable before using it.
    """
    pitch_px = px_per_mm / spec.lines_per_mm
    if pitch_px <= 0:
        raise ValueError("degenerate reseau pitch")

    yy = (np.arange(h, dtype=np.float32)[:, None] / pitch_px).astype(np.int32)
    xx = (np.arange(w, dtype=np.float32)[None, :] / pitch_px).astype(np.int32)

    mask = np.zeros((h, w, 3), dtype=np.float32)
    band = yy % 3                     # every third cell row is a red line
    chequer = (xx + yy) % 2           # blue/green alternate between the lines
    is_red = np.broadcast_to(band == 0, (h, w))
    is_blue = np.broadcast_to((band != 0) & (chequer == 0), (h, w))
    is_green = np.broadcast_to((band != 0) & (chequer == 1), (h, w))
    mask[:, :, 0] = is_red
    mask[:, :, 1] = is_green
    mask[:, :, 2] = is_blue
    return mask, float(pitch_px)


def reseau_reconstruct(
    record: np.ndarray, mask: np.ndarray, grid: FreqGrid, pitch_px: float, spec: ReseauSpec
) -> np.ndarray:
    """Rebuild colour from a single B&W record viewed back through the grid.

    Projection sends light through the positive and the same reseau in register,
    so each cell contributes only its own colour and the eye integrates. Modelled
    as a mask-weighted local average -- for each channel, blur the masked record
    and divide by the blurred mask, which is the coverage normalisation.

    The blur radius is deliberately comparable to the grid pitch rather than
    much larger. That is what leaves the faint grid texture visible and caps the
    colour resolution well below the luminance resolution, both of which are
    real and characteristic. A large radius would give clean colour and throw
    away the thing that makes the process recognisable.
    """
    sigma_um = spec.reconstruction_pitches * spec.pitch_um()
    blur = grid.gaussian(sigma_um)
    out = np.empty_like(mask)
    for c in range(3):
        num = apply_transfer((record * mask[:, :, c]).astype(np.float32), blur)
        den = apply_transfer(mask[:, :, c], blur)
        out[:, :, c] = num / np.maximum(den, np.float32(1e-4))
    return out


# ===========================================================================
# Grain synthesis
# ===========================================================================

# ---------------------------------------------------------------------------
#  TEMPORAL GRAIN -- queue C7, closed 2026-09-02
# ---------------------------------------------------------------------------
#: Honjo 1989 §4 (Техника кино и телевидения 1989 №4, the Fuji symposium paper
#: already cited on the F-series stocks): at 24 frames per second the eye
#: INTEGRATES over about 0.2 s, i.e. about five frames. Grain is re-rolled every
#: frame and is zero-mean, so five independent samples average down by 1/sqrt(5)
#: and the granularity a viewer perceives in PLAYBACK is 0.447 of what the same
#: emulsion shows in a frozen frame.
HONJO_EYE_INTEGRATION_S: float = 0.2

#: The upper bound on how many frames may be averaged. Beyond a handful the
#: assumption behind the sqrt law -- independent, stationary grain in a static
#: scene -- stops holding: real footage moves, and motion decorrelates the
#: retinal average long before the arithmetic runs out. Capped rather than
#: extrapolated.
TEMPORAL_GRAIN_MAX_FRAMES: float = 8.0


def temporal_grain_scale(fps: float,
                         integration_s: float = HONJO_EYE_INTEGRATION_S
                         ) -> float:
    """Grain amplitude a MOVING image should carry, relative to a still frame.

    ⚠ THIS IS NOT APPLIED ANYWHERE, AND THAT IS THE DECISION RATHER THAN AN
    OVERSIGHT. Queue C7 asked whether the still-frame grain amplitude is 2.24x
    too strong in playback. The physics says yes; the product question -- is this
    plugin judged frame by frame in a viewer, or in motion on a timeline? -- has
    two honest answers and only one can be the default.

    The default stays 1.0, i.e. STILL-FRAME CALIBRATION, for three reasons that
    are about evidence rather than taste:

    1. **Every granularity measurement this database holds is a STILL
       measurement.** rms through a 48 um aperture, Wiener spectra, Selwyn
       constants -- all made on a stationary sample. A renderer whose default
       silently divided them by 2.24 would stop reproducing the numbers it is
       calibrated against, and every parity test and reference render in this
       project would then be checked against a quantity no document states.
    2. **It is not reversible by a user who does not know the rule.** Someone who
       thinks the grain is too weak can raise `grain_scale`; someone who does not
       know 0.447 was already applied cannot tell whether they are correcting the
       emulsion or correcting the model.
    3. **The correction is one multiply on a control that already exists.** Both
       engines carry `grainScale`, so a host that wants motion-correct grain sets
       `grain_scale = temporal_grain_scale(fps)` and has it. Nothing was added to
       `FilmProfile`, nothing to the C++ stage list, and no shipped render moves.

    At 24 fps with the printed 0.2 s this returns 1/sqrt(4.8) = 0.4564; the
    queue's own 1/sqrt(5) = 0.4472 comes from rounding the same 0.2 s to five
    frames.
    """
    n = float(fps) * float(integration_s)
    n = min(max(n, 1.0), TEMPORAL_GRAIN_MAX_FRAMES)
    return 1.0 / math.sqrt(n)


# ---------------------------------------------------------------------------
#  sigma(D) <-> sigma(T), TO FOURTH ORDER -- Takano 1969 eq (2)
# ---------------------------------------------------------------------------
#: Kiyoshi Takano, «写真フィルムの粒状性» / "Granularity of Photographic Film",
#: テレビジョン (J. Inst. Telev. Engrs. Japan) 23(1) 13-23 (1969), §3.2 (1),
#: printed equation (2), with T = 10^-D:
#:
#:     sigma(D) = 0.434 * (sigma(T)/T_bar)
#:                * [ 1 + (1/12)(sigma(T)/T_bar)^2
#:                      + (1/80)(sigma(T)/T_bar)^4 + ... ]
#:
#: ⚠ WHY THIS IS HERE AND NOT IN A DERIVATION SCRIPT. Every granularity figure
#: this project stores is one of these two quantities, and the two are NOT
#: interchangeable: rms granularity as the industry quotes it is sigma(T) read
#: through a 48 um aperture, while `GrainSpec.sigma_shape_*` and everything the
#: renderer does with grain live in DENSITY. The corpus converted between them
#: with the FIRST term alone -- sigma_D = 0.4343 * sigma_t / t_bar -- and that
#: approximation is exactly what failed on BBC Report T-101 Fig. 26, whose
#: measured sigma(T)/T_bar runs 0.39 to 1.64: the law `sigma_D = 0.648*D^0.665`
#: fitted from it was WITHDRAWN for that reason (see the ILFORD_HPS provenance
#: note in film_profiles.py). Takano prints the correction the corpus needed.
#:
#: ⚠ NOTHING ON THE RENDER PATH CALLS THIS. It is a derivation helper: readers
#: and provenance work use it to move a stated sigma(T) into density before it
#: is stored. Adding it moves no pixel on any stock.
SIGMA_T_SERIES_C2: float = 1.0 / 12.0
SIGMA_T_SERIES_C4: float = 1.0 / 80.0


def sigma_density_from_transmittance(sigma_over_t: float) -> float:
    """sigma(D) from the ratio sigma(T)/T_bar, by Takano 1969 eq (2).

    Args:
        sigma_over_t: sigma(T) divided by mean transmittance, dimensionless.
            This is the whole argument -- the series depends on the RATIO only,
            not on T_bar and sigma(T) separately, which is why a stated rms
            granularity can be converted without knowing the density it was
            measured at.

    Returns:
        sigma(D) in density units.

    The series is asymptotic in the ratio and Takano prints three terms. At the
    ratio 0.39 that T-101 Fig. 26's cleanest sample gives, the correction over
    the first-order form is +1.3 %; at its worst sample, ratio 1.64, it is
    +31 %. Below about 0.2 the difference is under 0.4 % and the first-order
    form is adequate -- which is why the corpus got away with it for so long.
    """
    r = float(sigma_over_t)
    return 0.434 * r * (1.0 + SIGMA_T_SERIES_C2 * r ** 2
                        + SIGMA_T_SERIES_C4 * r ** 4)


def sigma_transmittance_from_density(sigma_d: float,
                                     tol: float = 1e-15,
                                     max_iter: int = 60) -> float:
    """The inverse of :func:`sigma_density_from_transmittance`.

    Newton on the printed series. Monotone for every ratio the series is
    meaningful over, so the iteration cannot land on a second root; it is
    written out rather than approximated because the whole point of carrying
    the higher terms is that the round trip closes.
    """
    target = float(sigma_d)
    r = target / 0.434                      # first-order seed
    for _ in range(max_iter):
        f = sigma_density_from_transmittance(r) - target
        if abs(f) < tol:
            break
        d = 0.434 * (1.0 + 3.0 * SIGMA_T_SERIES_C2 * r ** 2
                     + 5.0 * SIGMA_T_SERIES_C4 * r ** 4)
        r -= f / d
    return r


# ===========================================================================
# COUNTER-BASED RANDOMNESS -- schema v48 (FGS-DDS-001 Rev. A C1, §12.5)
# ===========================================================================
#
# ⚠⚠ THIS CLOSES FINDING F1, THE ONE CRITICAL DEFECT IN THE GRAIN STAGE, AND
# IT WAS INVISIBLE TO EVERY PARITY HARNESS THIS PROJECT HAS.
#
# Until now the reference renderer drew grain from `np.random.default_rng(seed)`
# -- one stateful generator, seeded once per render, with NO frame index in it.
# Render frame 0 and frame 1 of the same clip and you get THE SAME GRAIN FIELD,
# pixel for pixel. The C++ engines have always used the counter generator below
# and produce an independent field per frame, as real film does. So the
# reference and the production engines were modelling different physics -- one
# with grain welded to the frame, one with grain that lives -- and both passed
# every test, because every harness in this repository compares ONE frame.
#
# A single-frame comparison is structurally blind to a temporal defect. That is
# the lesson, and `cpp_parity`'s new two-frame probe is the answer to it.
#
# WHY A COUNTER GENERATOR RATHER THAN A SEEDED SEQUENCE. Every value is a PURE
# FUNCTION of (seed, frame, stage, ordinal). The host renders frames out of
# order, speculatively, twice, and from several threads; a sequential generator
# would answer differently each time and the grain would crawl under scrubbing.
# There is no state to carry, share or lock.
#
# ⚠ THIS IS A LINE-FOR-LINE PORT OF `AlgoCounterRng.hpp` AND IT IS VERIFIED AS
# ONE, not merely intended as one: `cpp_parity`'s RNG probe compiles the real
# header and compares counters, uniforms and Box-Muller normals against these
# functions, including negative frame indices, which occur legitimately when a
# defect's birth frame is searched backwards from the start of a clip.

#: SplitMix64 finalising constants. Published values, selected by search for
#: avalanche quality; 0x9E3779B97F4A7C15 is 2^64/phi and doubles as the stream
#: increment. ⚠ Not interchangeable with any other odd constants: substituting
#: them still gives a bijection but degrades the statistics.
RNG_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
RNG_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
RNG_MIX_2 = np.uint64(0x94D049BB133111EB)
RNG_SHIFT_1 = np.uint64(30)
RNG_SHIFT_2 = np.uint64(27)
RNG_SHIFT_3 = np.uint64(31)

#: 2^-53. The uniform is built from the top 53 bits of the mixed value, which
#: is what a double's mantissa holds exactly, so every representable value in
#: [0,1) is reachable and none is favoured. ⚠ Taking fewer bits -- following
#: float32 down, say -- would reduce the generator to about 8 million distinct
#: values, which bands visibly in any field built from it.
RNG_TWO_POW_M53 = 1.0 / 9007199254740992.0


class RngStage(IntEnum):
    """Which generator stream a consumer draws from.

    ⚠ MIRRORS `eALGO_RNG_STAGE` AND THE VALUES ARE FROZEN. They are arbitrary
    but must be distinct and must never be reused or renumbered, because doing
    so changes the appearance of every existing render. Spaced by 0x100 so a
    stage needing sub-streams can take a small offset without colliding.

    Separate streams are not tidiness: without them the coating field and the
    grain field would draw the same numbers and the grain would visibly follow
    the coating streaks.
    """

    COATING_STATIC = 0x0100
    COATING_DRIFT = 0x0200
    FLICKER = 0x0300
    NEG_DEFECTS = 0x0400
    GRAIN_R = 0x0500
    GRAIN_G = 0x0600
    GRAIN_B = 0x0700
    PRINT_GRAIN = 0x0800
    DUPE_GRAIN = 0x0900
    MISREG = 0x0A00
    WEAVE = 0x0B00
    GATE_DEFECTS = 0x0C00


def rng_mix64(z):
    """The SplitMix64 finalising bijection. Equal in, equal out; distinct in,
    distinct out -- so distinct coordinates can never collide onto one value."""
    z = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z + RNG_GOLDEN
        z = (z ^ (z >> RNG_SHIFT_1)) * RNG_MIX_1
        z = (z ^ (z >> RNG_SHIFT_2)) * RNG_MIX_2
        return z ^ (z >> RNG_SHIFT_3)


def rng_counter(seed: int, frame_index: int, stage: "RngStage", ordinal):
    """Pack (seed, frame, stage, ordinal) into one 64-bit counter.

        bits 63..32   seed
        bits 31..24   stage identifier (the HIGH BYTE of the enumerator)
        bits 23..00   ordinal

    ⚠ `frame_index` HAS NO FIELD OF ITS OWN; it is folded into the seed field
    by multiplication with the golden constant. Two reasons, both load-bearing.
    24 bits of ordinal is 16.7 million draws per stage per frame -- ample for a
    4K plane -- and taking bits away from it to house a frame counter would cap
    the render size. And multiplying rather than adding keeps successive frames
    far apart in counter space, which matters because the mixer is being fed
    values that differ in one low bit.

    `frame_index` is SIGNED and may be negative. The cast to unsigned wraps,
    which is well defined here and harmless, because the mixer treats all
    64-bit values alike.
    """
    with np.errstate(over="ignore"):
        frame_salt = np.uint64(np.uint32(int(frame_index) & 0xFFFFFFFF)) * RNG_GOLDEN
        seed_field = (np.uint64(int(seed) & 0xFFFFFFFF) << np.uint64(32)) ^ frame_salt
        stage_field = np.uint64((int(stage) >> 8) & 0xFF) << np.uint64(24)
        ord_field = np.asarray(ordinal, dtype=np.uint64) & np.uint64(0x00FFFFFF)
        return (seed_field ^ stage_field) ^ ord_field


def rng_uniform01(counter):
    """Uniform in [0,1) from the top 53 bits of the mixed counter."""
    return (rng_mix64(counter) >> np.uint64(11)).astype(np.float64) * RNG_TWO_POW_M53


def rng_normal(counter):
    """Standard normal by Box-Muller, cosine branch only.

        z = sqrt(-2 ln u1) * cos(2 pi u2)

    The two uniforms come from two counters rather than from a sequence, because
    there is no sequence -- the second is displaced by the golden constant so
    the mixer's two inputs are far apart despite sharing a request. The sine
    branch would give a second independent value for free but keeping it needs
    state, so it is discarded; that doubles the cost and is accepted.

    u1 is floored at 2^-53. log(0) is -inf and one infinity destroys an entire
    frame; the probability is 2^-53 per draw, which is negligible and not zero.

    ⚠ Box-Muller rather than the ziggurat: this has to be a pure function of its
    counter with no rejection loop, and ziggurat's variable draw count per value
    cannot be indexed deterministically.
    """
    counter = np.asarray(counter, dtype=np.uint64)
    u1 = rng_uniform01(counter)
    u2 = rng_uniform01(counter ^ RNG_GOLDEN)
    u1 = np.where(u1 < RNG_TWO_POW_M53, RNG_TWO_POW_M53, u1)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(
        6.283185307179586476925286766559 * u2)


def counter_normal_plane(h: int, w: int, seed: int, frame_index: int,
                         stage: "RngStage") -> np.ndarray:
    """An h x w plane of unit-variance white noise from the counter generator.

    ⚠ THE ORDINAL IS `y * w + x`, WHICH PINS A PARITY CONTRACT. The C++ twins
    use `y * pitch + x` with the PADDED pitch, so the two agree pixel for pixel
    only when the engine's pitch equals its active width. `cpp_parity` drives
    the engines at pitch == width for exactly this reason, and a region render
    at a different pitch draws different numbers by design -- that is what makes
    a region render's grain independent of the region, rather than of the frame.
    """
    ordinal = np.arange(h * w, dtype=np.uint64).reshape(h, w)
    return rng_normal(rng_counter(seed, frame_index, stage, ordinal)).astype(
        np.float32)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration, tolerating the numpy 1.x / 2.x rename."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def grain_reference_energy(
    clump_um: float, clump_gain: float, f_max: float = 400.0, n: int = 16001
) -> float:
    """Aperture-weighted spectral energy of the grain, over *all* frequencies.

    This is ``2*pi * integral |H(f) A(f)|^2 f df`` with H the grain spectrum and
    A the 48 um measuring aperture, evaluated as a continuous radial integral
    rather than as a sum over the pixel grid.

    Doing it continuously is the whole point, and getting it wrong is subtle.
    The obvious implementation calibrates the discrete field so that its
    aperture-averaged deviation equals the target on the render grid. That
    silently over-amplifies any stock whose grain is finer than a pixel: all of
    its spectral energy folds back into the sampled band, so the calibration
    inflates the amplitude to compensate for detail the grid cannot hold. The
    symptom is that VISION3 50D, at RMS 2.6, renders as grainy as 500T at RMS
    6.6 -- which is exactly backwards, and it was happening here until measured.

    Integrating over the true spectrum instead makes the amplitude a property of
    the emulsion alone. The scanner MTF then band-limits it before sampling,
    just as the real optics do, so a fine-grained stock correctly renders
    smoother than a coarse one at any resolution.

    Note the consequence for resolution: a 2K render genuinely shows less
    granularity than a 6K render of the same negative, converging upward as the
    band widens. That is not a modelling artefact -- it is why 4K scans of old
    negatives look grainier than the 2K masters everyone remembers.
    """
    f = np.linspace(0.0, f_max, n)
    f_hi = 1000.0 / (2.0 * clump_um)
    f_lo = f_hi / 6.0
    h = np.exp(-((f / f_hi) ** 2)) * (1.0 + clump_gain * np.exp(-((f / f_lo) ** 2)))
    a = np.exp(-2.0 * (math.pi**2) * (APERTURE_SIGMA_MM**2) * f**2)
    energy = 2.0 * math.pi * _trapz((h * a) ** 2 * f, f)
    if energy <= 0.0:
        raise RuntimeError("degenerate grain spectrum; check clump size")
    return energy


def _bessel_j1(x: np.ndarray) -> np.ndarray:
    """J1(x) to about 1e-7 absolute, Abramowitz and Stegun 9.4.4 / 9.4.6.

    ⚠ HAND-ROLLED ON PURPOSE. `numpy` ships no Bessel function of the first
    kind of order one and `math` ships none either, so a SciPy dependency
    would be the alternative -- on the REFERENCE engine, for one function, in
    a form neither C++ twin could use. The polynomial is the standard one, it
    is exact enough for a spectrum compared against a traced plot, and it
    ports to C++ verbatim if this ever reaches the render path.
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    out = np.empty_like(ax)

    small = ax < 8.0
    if np.any(small):
        y = x[small] ** 2
        num = x[small] * (
            72362614232.0
            + y * (-7895059235.0
                   + y * (242396853.1
                          + y * (-2972611.439
                                 + y * (15704.48260 + y * (-30.16036606)))))
        )
        den = (
            144725228442.0
            + y * (2300535178.0
                   + y * (18583304.74
                          + y * (99447.43394 + y * (376.9991397 + y))))
        )
        out[small] = num / den

    big = ~small
    if np.any(big):
        z = 8.0 / ax[big]
        y = z * z
        xx = ax[big] - 2.356194491
        p = (
            1.0
            + y * (0.183105e-2
                   + y * (-0.3516396496e-4
                          + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
        )
        q = (
            0.04687499995
            + y * (-0.2002690873e-3
                   + y * (0.8449199096e-5
                          + y * (-0.88228987e-6 + y * (0.105787412e-6))))
        )
        out[big] = (np.sqrt(0.636619772 / ax[big])
                    * (np.cos(xx) * p - z * np.sin(xx) * q)
                    * np.sign(x[big]))
    return out


#: First zero of the normalised jinc in units of 1 / diameter: the first zero
#: of J1 is at 3.8317, and pi * f * d = 3.8317 puts it at f = 1.2197 / d.
BOOLEAN_GRAIN_FIRST_ZERO_OVER_D: float = 3.8317059702075123 / math.pi


def boolean_grain_shape(f_mm, grain_um: float) -> np.ndarray:
    """The Boolean / random-dot grain AMPLITUDE transfer, in closed form.

    ⚠⚠ THIS IS THE PHYSICALLY DERIVED ALTERNATIVE TO `FreqGrid.grain_shape`,
    AND IT IS AN ANALYSIS FUNCTION -- NOTHING ON THE RENDER PATH CALLS IT.
    Added 2026-09-18g from the three granularity papers; see the v46 block in
    `film_profiles`.

    WHERE IT COMES FROM. In a Boolean model of disks of radius r at Poisson
    intensity lambda the two-point covariance of the coverage indicator is

        C(h) = (1 - p)^2 * (exp(lambda * A_bar(h)) - 1)

    with A_bar(h) the mean area of a grain intersected with its own translate
    by h. At low coverage exp(x) - 1 -> x, so C(h) is proportional to A_bar(h)
    -- THE AUTOCORRELATION OF THE DISK -- whose Fourier transform is the
    squared modulus of the disk's own transform. The amplitude transfer is
    therefore the normalised jinc

        h(f) = | 2 J1(pi f d) / (pi f d) |,        d = grain diameter

    with NO FREE PARAMETER beyond d. Contrast `FreqGrid.grain_shape`, which is
    a Gaussian rolloff plus an empirical low-frequency lobe and has two.

    ⚠⚠ THE TWO SHAPES DIFFER IN A WAY THAT MATTERS AND IS TESTABLE. The
    Gaussian has infinite support in space and never reaches zero in
    frequency; the jinc has ZEROS, the first at f = 1.2197 / d, and its
    covariance is identically zero beyond one grain diameter. That compact
    support is the theorem `_BOOLEAN_NO_LONG_RANGE_LOBE` records: a Boolean
    model cannot produce long-range correlation at all, so a non-zero
    `clump_gain` is a departure from the physical model rather than a
    parameter within it.

    ⚠ WHY IT IS NOT THE DEFAULT. Switching the render path to it would move
    every rendered pixel on 191 stocks, and would need the same owner decision
    queue C45 needed for a grain rescale -- plus `_bessel_j1` in both C++
    twins. It is here so the two shapes can be COMPARED against a measured
    Wiener spectrum on equal terms, which this corpus has for exactly one
    stock (ILFORD_HPS, BBC Monograph 54 Fig. 8).

    Args:
        f_mm: spatial frequency, cycles per millimetre.
        grain_um: imaging-centre DIAMETER in micrometres -- the random-dot
            quantity from `film_profiles.random_dot_disk_diameter_um`, NOT
            `GrainSpec.clump_um_*`, which the v46 block measures at about five
            times larger on the 186 stocks whose value is an estimate.

    Returns the amplitude transfer, 1.0 at DC.
    """
    if grain_um <= 0.0:
        raise ValueError("grain_um must be positive")
    # f is cycles/mm and d is micrometres, so d converts to mm here.
    x = np.pi * np.asarray(f_mm, dtype=np.float64) * (grain_um / 1000.0)
    out = np.ones_like(x)
    nz = x > 1e-12
    out[nz] = np.abs(2.0 * _bessel_j1(x[nz]) / x[nz])
    return out.astype(np.float32)


def kernel_axis_transfer(sigma_px: float, n: int, half: bool) -> np.ndarray:
    """Exact DFT of ONE AXIS of the C++ engines' truncated separable Gaussian.

    ⚠⚠ THIS, NOT THE ANALYTIC TRANSFER, IS WHAT THE PRODUCTION ENGINES APPLY,
    AND THE TWO ARE NOT THE SAME OPERATOR. `AlgoGaussianBlurPlaneWrapXY`
    convolves with a Gaussian SAMPLED at integer offsets, truncated at
    ceil(4 sigma) taps either side (minimum 1) and renormalised to unit sum.
    A sampled kernel's transfer is PERIODIC in frequency, so what it applies is
    the periodised analytic transfer, sum over m of T(f + m). At Nyquist the
    m = -1 image lands exactly on the m = 0 term and the transfer is DOUBLED --
    measured at exactly 2.00x for every sigma from 0.6 to 1.2 px (queue C16).

    Until v48 the reference multiplied by the ANALYTIC Gaussian instead, so
    every render disagreed with both C++ engines at the top of the band. It was
    tolerated because the disagreement vanishes as T(Nyquist) itself goes to
    zero, which it does above about 1.2 px. ⚠ THE GRAIN REBASE MAKES THAT
    TOLERANCE UNSAFE: the physical spectrum's sigmas are SMALLER than the
    legacy ones -- by a median factor of 6.7 -- so the rebased stage lands
    squarely in the sigma range where the two operators differ by 2x.

    Since a wrap-around separable convolution is exactly a circular convolution,
    multiplying by this transfer in the frequency domain is not an approximation
    of the C++ blur; it is the same operator, evaluated the cheap way. That is
    what lets the reference run one FFT pair instead of ten spatial blurs and
    still be the same algorithm.

    ⚠ The taps are built and transformed rather than the analytic transfer being
    periodised, because below about 0.8 px the truncation and the renormalisation
    stop being negligible -- at sigma 0.4 the support is 5 taps and a periodised
    prediction is 8.5e-02 away from the real kernel. For a five-tap renormalised
    kernel only the taps themselves are the truth.

    Args:
        sigma_px: standard deviation in pixels; <= 0 returns the identity, which
            is the correct limit and matches the C++ early-out.
        n: axis length in pixels.
        half: True for the rfft axis (returns n//2 + 1 bins), False for the
            full-fft axis (returns n bins).
    """
    m = (int(n) // 2 + 1) if half else int(n)
    if not (sigma_px > 0.0):
        return np.ones(m, dtype=np.float64)
    cutoff = float(fp.GRAIN_BLUR_SIGMA_CUTOFF)
    hw = max(1, int(math.ceil(cutoff * float(sigma_px))))
    hw = min(hw, int(fp.GRAIN_BLUR_MAX_HALF_TAPS))
    x = np.arange(-hw, hw + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma_px)) ** 2)
    k /= k.sum()
    lag = np.zeros(int(n), dtype=np.float64)
    for i, xx in enumerate(x.astype(int)):
        lag[xx % int(n)] += k[i]
    return (np.real(np.fft.rfft(lag)) if half
            else np.real(np.fft.fft(lag)))


def grain_axis_blur_transfer(sigma_px: float, h: int, w: int, aniso: float):
    """The separable truncated blur's transfer on the rfft2 grid, or None.

    Returns None when the blur is the identity, so the caller can skip it
    instead of multiplying by ones -- which in the engines is a full pass over
    the plane to multiply by one, and is where most of the grain stage's time
    was going before v48's factoring.
    """
    if fp.grain_blur_is_identity(sigma_px) and fp.grain_blur_is_identity(
            sigma_px * aniso):
        return None
    tx = kernel_axis_transfer(sigma_px, w, half=True)
    ty = kernel_axis_transfer(sigma_px * aniso, h, half=False)
    return ty[:, None] * tx[None, :]


def grain_transfer(spec, h: int, w: int, px_per_mm: float,
                   scan_sigma_px: float, anisotropy: float):
    """Assemble the FACTORED grain transfer on the rfft2 grid.

        T = [ sum_k w_k B(s_k) ] . [ 1 + g B(s_lobe) ] . B(s_scan)

    ⚠ THE THREE BRACKETS ARE THREE SEPARATE OPERATORS AND THAT IS THE POINT.
    Until v48 the lobe doubled the term count and the scan band limit was folded
    into every term's sigma, so a ten-term stock ran ten full-plane blurs at
    roughly the scan sigma. Both fold out exactly -- a product of Gaussian
    transfers is a Gaussian whose variances add -- leaving the bare grain
    sigmas, which at 4K are 0.289 down to 0.018 px and mostly identities.

    ⚠ ANISOTROPY IS APPLIED TO EVERY BRACKET, band limit included. The model of
    record evaluates the whole product on one pre-stretched frequency grid, so
    stretching the crystal term alone would stretch the emulsion and not the
    scanner. Written out per bracket it is the same law, because
    a*sqrt(sx^2+ss^2) == sqrt((a*sx)^2+(a*ss)^2).
    """
    a = max(float(anisotropy), 1e-6)

    mix = np.zeros((h, w // 2 + 1), dtype=np.float64)
    flat_weight = 0.0
    for s_mm, wgt in spec.terms:
        t = grain_axis_blur_transfer(float(s_mm) * px_per_mm, h, w, a)
        if t is None:
            flat_weight += float(wgt)          # an identity blur is a scalar
        else:
            mix += float(wgt) * t
    if flat_weight != 0.0:
        mix += flat_weight

    if spec.lobe_gain > 0.0 and spec.lobe_sigma_mm > 0.0:
        t = grain_axis_blur_transfer(
            float(spec.lobe_sigma_mm) * px_per_mm, h, w, a)
        mix = mix * (1.0 + spec.lobe_gain * (1.0 if t is None else t))

    t = grain_axis_blur_transfer(max(float(scan_sigma_px), 0.0), h, w, a)
    if t is not None:
        mix = mix * t

    return mix



#: Frame index reserved for the SCANNER FIXED-PATTERN draw. It must be a value
#: no real frame can take, so that the fixed component is the same field on
#: every frame of every clip while staying keyed to the render seed.
#:
#: ⚠ -2**31 IS CHOSEN BECAUSE `rng_counter` CASTS THE FRAME INDEX THROUGH
#: uint32 AND THE CAST WRAPS. Any sentinel inside the range a clip can reach
#: would collide with a real frame and freeze that one frame's emulsion grain
#: into the fixed pattern -- a defect that would appear once in a 68-year clip
#: and be unreproducible when reported.
FIXED_PATTERN_FRAME = -(2 ** 31)

#: Longest half-window the temporal kernel will build, in frames. At rho = 0.95
#: the untruncated kernel needs 58 taps for 3 tau; this caps the cost at 13
#: field draws per frame and the truncation error is folded into the
#: renormalisation below rather than left to bias the variance.
TEMPORAL_MAX_TAPS = 6


def temporal_kernel(rho: float) -> "list[float]":
    """Normalised weights whose autocorrelation at lag 1 is approximately `rho`.

    ⚠⚠ STATELESS BY CONSTRUCTION, AND THAT IS THE WHOLE DESIGN CONSTRAINT. An
    AR(1) recursion -- E_n = rho E_{n-1} + sqrt(1-rho^2) W_n -- is the textbook
    way to correlate successive frames and it CANNOT BE USED HERE: it carries
    state, so rendering frame 5000 would mean rendering frames 0 to 4999 first,
    and a renderer that cannot start at an arbitrary frame cannot be used on a
    shot, in a farm, or by a host that scrubs a timeline. This project's whole
    RNG is counter-based for the same reason.

    So the correlation is built as a SLIDING WEIGHTED SUM over per-frame white
    fields instead:

        E_n = sum_j c_j W_{n+j} / sqrt(sum_j c_j^2),   c_j = exp(-|j| / tau)

    Each `W_k` depends only on (seed, k, stage, pixel), so any frame is
    computable on its own, in any order, on any machine. The normalisation
    fixes Var(E_n) = 1 exactly, which is the "statistically controlled" half of
    the requirement -- the temporal structure changes WHEN grain appears, never
    HOW MUCH of it there is.

    ⚠ THE AUTOCORRELATION IS THE KERNEL'S OWN, NOT AR(1)'s. r(d) = sum_j c_j
    c_{j+d} / sum_j c_j^2, which for an exponential c is close to exp(-d/tau)
    but not equal to it, and is exactly symmetric where AR(1) is causal. Film
    has no arrow of time in its grain, so a symmetric kernel is if anything the
    better description -- but the stored `grain_frame_correlation` is therefore
    the TARGET lag-1 correlation and `tau` is solved to hit it, rather than
    being the AR(1) coefficient of the same name.

    Returns a single-element kernel for rho <= 0, which is the identity: one
    field draw, the pre-v49 path, bit for bit.
    """
    if not (rho > 0.0):
        return [1.0]
    r = min(float(rho), 0.95)
    tau = -1.0 / math.log(r)
    taps = min(TEMPORAL_MAX_TAPS, max(1, int(math.ceil(3.0 * tau))))
    c = [math.exp(-abs(j) / tau) for j in range(-taps, taps + 1)]
    # Solve the kernel's own lag-1 correlation onto the target. One Newton-free
    # bisection on tau, because the closed form is not invertible and a fitted
    # constant here would be the thing this project keeps refusing.
    lo, hi = 1e-3, 60.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cc = [math.exp(-abs(j) / mid) for j in range(-taps, taps + 1)]
        num = sum(cc[i] * cc[i + 1] for i in range(len(cc) - 1))
        den = sum(v * v for v in cc)
        if num / den < r:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    c = [math.exp(-abs(j) / tau) for j in range(-taps, taps + 1)]
    n = math.sqrt(sum(v * v for v in c))
    return [v / n for v in c]


def temporal_white(h: int, w: int, seed: int, frame_index: int,
                   stage: "RngStage", rho: float,
                   fixed_fraction: float) -> np.ndarray:
    """Unit-variance white field carrying the stock's temporal structure.

    Two statistically distinct components, which is the distinction the spec
    draws and the database could not express until v49:

      EMULSION   redrawn per frame, correlated across frames by
                 `temporal_kernel`. Physically the frame-to-frame correlation
                 of a CAMERA NEGATIVE is ZERO -- every frame is a different
                 piece of film and shares no silver grains with its neighbours
                 -- so `grain_frame_correlation` is 0.0 on every stock in the
                 database and this reduces to one draw. The mechanism exists
                 for the cases where it is not zero: a frozen frame, an optical
                 step printer holding one negative frame across several print
                 frames, and any stock whose grain a future measurement shows
                 to persist.

      FIXED      drawn ONCE, keyed on the seed and never on the frame, so it is
                 perfectly correlated across the whole clip. This is a SCANNER
                 property and not a film property -- sensor non-uniformity is
                 identical on every frame it digitises -- which is why its
                 strength arrives as a render CONTROL and not as a column in
                 the film database. Same film x instrument split the project
                 already makes for Callier.

    Variance is preserved exactly: the two components are independent and are
    combined as sqrt(1-f) E + sqrt(f) S, so the field this returns has unit
    variance for every rho and every f and the granularity calibration above
    is untouched by either.
    """
    f = min(max(float(fixed_fraction), 0.0), 1.0)
    c = temporal_kernel(rho)
    if len(c) == 1:
        emul = counter_normal_plane(h, w, seed, frame_index, stage)
    else:
        m = len(c) // 2
        emul = np.zeros((h, w), dtype=np.float32)
        for i, wt in enumerate(c):
            if wt == 0.0:
                continue
            emul += np.float32(wt) * counter_normal_plane(
                h, w, seed, frame_index + (i - m), stage)
    if f <= 0.0:
        return emul
    fixed = counter_normal_plane(h, w, seed, FIXED_PATTERN_FRAME, stage)
    return (np.float32(math.sqrt(1.0 - f)) * emul
            + np.float32(math.sqrt(f)) * fixed)


def make_grain_field(
    h: int,
    w: int,
    px_per_mm: float,
    spec,
    rms_granularity: float,
    scan_sigma_px: float,
    anisotropy: float,
    seed: int,
    frame_index: int,
    stage: "RngStage",
    aperture=None,
    frame_correlation: float = 0.0,
    fixed_fraction: float = 0.0,
) -> np.ndarray:
    """Spectrally-shaped, granularity-calibrated, frame-keyed grain field.

    Returns a zero-mean density-domain field whose amplitude is fixed by the
    stock's RMS granularity and whose spectrum is `spec` -- the factored
    `film_profiles.GrainSpectrum`, which is the ONE definition of the grain
    spectrum and the only thing about it any engine knows.

    ⚠ THE SIGNATURE CHANGED IN v48 AND THE OLD ONE COULD NOT BE KEPT. It took a
    `FreqGrid` and a stateful `rng`, and both had to go: the grid carried the
    analytic transfer this stage no longer applies, and the generator had no
    frame index in it, which is finding F1 -- every frame of a clip got the same
    grain in the reference while the C++ engines re-rolled it, so the two were
    modelling different physics and no single-frame harness could see it.

    The field is built exactly as the C++ engines build it:

        1. white noise from the counter generator at (seed, frame, stage, pixel)
        2. one separable truncated-Gaussian blur per mixture term, skipping the
           terms whose kernel is the identity
        3. one blur for the clustering lobe, applied to their weighted sum
        4. one blur for the scan band limit, applied to the result
        5. mean removed, then the amplitude applied

    Steps 2 to 4 run here as a single frequency-domain multiply, which is not an
    approximation of the blurs but the identical circular convolution, and step
    5's mean removal is the DC bin zeroed, which is the same operation the
    engines perform as a subtraction.

    ⚠ THE AMPLITUDE IS CALIBRATED AGAINST A CONTINUOUS INTEGRAL, NOT AGAINST
    THIS GRID, and that is the part that is invisible until it is measured. A
    grid-referred calibration over-amplifies any stock whose grain is finer than
    a pixel, because all of its spectral energy folds back into the sampled band
    and the calibration inflates the amplitude to compensate for detail the grid
    cannot hold. The symptom is VISION3 50D at rms 2.6 rendering as grainy as
    500T at rms 6.6 -- backwards, and it was happening until it was measured.

    Consequence, and it is physics rather than an artefact: a 2K render genuinely
    shows less granularity than a 6K render of the same negative, converging
    upward as the band widens. That is why 4K scans of old negatives look
    grainier than the 2K masters everyone remembers.
    """
    if rms_granularity <= 0.0 or px_per_mm <= 0.0 or not spec.terms:
        return np.zeros((h, w), dtype=np.float32)

    energy = fp.grain_reference_energy_terms(spec, aperture)
    scale = (rms_granularity / 1000.0) * px_per_mm / math.sqrt(energy)

    transfer = grain_transfer(spec, h, w, px_per_mm, scan_sigma_px, anisotropy)

    # Zero mean, exactly. A field with a non-zero mean would shift the overall
    # density of the frame, so the grain control would double as an exposure
    # control.
    transfer = np.array(transfer, dtype=np.float64, copy=True)
    transfer[0, 0] = 0.0

    white = temporal_white(h, w, seed, frame_index, stage,
                           frame_correlation, fixed_fraction)
    field = np.fft.irfft2(np.fft.rfft2(white) * transfer, s=(h, w))
    return (field * scale).astype(np.float32)


# ===========================================================================
# 16-bit PNG writer (stdlib only)
# ===========================================================================
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, rgb: np.ndarray, bit_depth: int = 16,
              alpha: bool = False) -> None:
    """Write an RGB or RGBA PNG at 8 or 16 bits per channel.

    Pillow cannot write 16-bit RGB PNG, and 8 bits visibly bands in the smooth
    halation bloom and in deep shadow. Rather than pull in a dependency, this
    emits the file directly: signature, IHDR, one zlib-compressed IDAT with
    filter type 0 per scanline, IEND.

    ``alpha`` appends a fully opaque channel and switches the PNG colour type
    from 2 (truecolour) to 6 (truecolour with alpha).

    ⚠ THE ALPHA IS CONSTANT 255 AND CARRIES NO INFORMATION. Nothing in this
    renderer produces transparency -- film is opaque, and every stage works in
    density or in linear light with no coverage term anywhere. The channel
    exists because some comparison and compositing tools expect four channels
    and will not open a three-channel file, so writing it is a convenience for
    the reader, not a property of the image. Anyone computing with it should
    ignore it; anyone diffing two of our renders will find it identical in
    both and contributing nothing to the difference.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an (h, w, 3) array")
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")

    h, w = rgb.shape[:2]
    nchan = 4 if alpha else 3

    if alpha:
        # Opaque, at whatever depth the rest of the image is written in --
        # 255 for 8-bit, 65535 for 16-bit. Writing 255 into a 16-bit alpha
        # would be very nearly transparent, which is the obvious way to get
        # this wrong.
        opaque = (1 << bit_depth) - 1
        rgb = np.concatenate(
            (rgb, np.full((h, w, 1), opaque, dtype=rgb.dtype)), axis=2)

    if bit_depth == 16:
        payload = rgb.astype(">u2").tobytes()
        stride = w * nchan * 2
    else:
        payload = rgb.astype(np.uint8).tobytes()
        stride = w * nchan

    # Prepend the per-scanline filter byte (0 = None) without a Python loop.
    raw = np.zeros((h, stride + 1), dtype=np.uint8)
    raw[:, 1:] = np.frombuffer(payload, dtype=np.uint8).reshape(h, stride)

    # Colour type 6 is truecolour with alpha, 2 is truecolour.
    ihdr = struct.pack(">IIBBBBB", w, h, bit_depth,
                       6 if alpha else 2, 0, 0, 0)
    body = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(body)


# ===========================================================================
# Render settings
# ===========================================================================
@dataclass(slots=True)
class RenderSettings:
    """Everything about the render that is not the film stock itself."""
    #: Render the emulsion MTF through the SEPARABLE KERNEL the C++ engines use
    #: instead of the exact frequency-domain law.
    #:
    #: ⚠ ADDITIVE AND OFF BY DEFAULT. Python keeps the exact law as the
    #: reference -- that is the whole point of having a reference -- and this
    #: switch exists so the three implementations can be compared on the SAME
    #: arithmetic. With it off, Python and C++ differ on the 22 measured stocks
    #: by the kernel's fit error (worst 0.0384 in modulation); with it on, they
    #: should agree to floating-point noise. Turning it on makes Python LESS
    #: accurate, which is why it is not the default.
    mtf_use_kernel: bool = False


    #: Accepts FilmFormatCtrl or the bare FORMAT_GEOM key. The enumerator is
    #: the canonical form and matches the C++ control exactly; the string is
    #: retained so existing callers and saved presets keep working.
    film_format: FilmFormatCtrl | str = "super35"
    #: Accepts PrintStockCtrl or the bare PRINT_STOCKS name.
    print_stock: PrintStockCtrl | str = ""   # "" = the stock's own default
    exposure_stops: float = 0.0
    # -- C8, 2026-08-23: the exposure TIME, seconds. 0.0 = not stated, and the
    # -- reciprocity stage is then skipped entirely, so every render made before
    # -- this field existed is reproduced bit for bit. It is not a duplicate of
    # -- exposure_stops: that one moves the scene along the curve, this one says
    # -- how long the shutter was open, which changes how the EMULSION responds
    # -- to the same amount of light. Seconds rather than shutter angle
    # -- deliberately -- angle x frame rate can only ever produce 1/1000..1/24 s,
    # -- and every sheet on file prints "no correction needed" across exactly
    # -- that span. The corrections live beyond 1 s and below 1e-4 s.
    exposure_time_s: float = 0.0
    #: WHICH DEVELOPMENT, as a `ProcessVariantCtrl` value. `eAS_SHIPPED`
    #: (-1) is the sentinel and means the development the stored curves
    #: already represent.
    #:
    #: ⚠⚠ AN ENUMERATOR AND NOT AN INDEX SINCE 2026-09-17. It used to be
    #: a position in the selected stock's own `process_variants` tuple, so
    #: the same stored number named a different development on every stock
    #: that had one -- always in range, therefore never detectable, and
    #: silently re-pointed by inserting a variant. The value is now global:
    #: `ProcessVariantCtrl.ePORTRA800_EI3200_PUSH2` means that development
    #: and nothing else, and selecting it on a stock that does not offer it
    #: renders the stock as shipped rather than clamping into range.
    #:
    #: ⚠ ONLY 5 OF THE 31 RECORDED DEVELOPMENTS CHANGE A PIXEL -- see
    #: `resolve_process_variant`. The other 26 differ only in exposure index,
    #: which no stage reads.
    process_variant: int = int(ProcessVariantCtrl.eAS_SHIPPED)
    #: Development time in minutes, or < 0 for the sentinel -- "the
    #: development the stored curves represent". See
    #: `resolve_development_time`; inert on every stock with no
    #: gamma-bearing development family, which is most of them.
    development_minutes: float = -1.0
    development_celsius: float = -1.0
    #: Years of DARK STORAGE since processing. 0 = fresh, which is
    #: the default and is inert on every stock. See
    #: `resolve_storage_age`; only stocks carrying a published
    #: dark-fade rate respond, which is two of 185 today.
    storage_years: float = 0.0
    scene_kelvin: float = 5500.0
    wb_strength: float = 0.0
    grey_target: float = 0.18       # display linear value for 18% scene grey
    # -- 2026-09-09c, queue item #303: HOW MUCH OF THE STOCK'S OWN Dmax IS
    # -- STRETCHED DOWN TO OUTPUT ZERO. Stage 14 normalises transmittance as
    # --
    # --     out = (10^-D  -  s * t_min) / (t_max  -  s * t_min)
    # --
    # -- with t_max = 10^-Dmin, t_min = 10^-Dmax and s = this field.
    # --   s = 1.0  Dmax maps to output 0. Every stock uses the whole output
    # --            range. THE SHIPPED BEHAVIOUR AND THE DEFAULT -- at exactly
    # --            1.0 the expression is arithmetically what it always was, so
    # --            every render made before this field existed is reproduced.
    # --   s = 0.0  Dmax maps to its own relative transmittance, 10^-(Dmax-Dmin).
    # --            Nothing clips at the bottom and Dmax becomes RENDER-VISIBLE
    # --            for the first time -- a Polaroid's weak 1.6 D black stops
    # --            looking like Velvia's 3.0 D black.
    # --   between  a linear blend of the two black points.
    # --
    # -- ⚠⚠ WHY IT IS A CONTROL AND NOT A FIX. s = 1.0 destroys real shadow
    # -- information, and that is measured, not argued: on the 2026-09-09 Velvia
    # -- regression frame 13.06 % of the BLUE record arrives at stage 14 with a
    # -- density at or above Dmax and is mapped to exactly 0.0 before the encoder
    # -- ever sees it. At s = 0.0 that is 0.00 %, the whole-frame mean moves only
    # -- 0.7 of one 8-bit code, and the red clipping is untouched. So s = 0 is
    # -- strictly more information.
    # -- ⚠ BUT IT IS ALSO A LOOK, AND IT MOVES EVERY ONE OF THE 184 STOCKS. The
    # -- floor each stock lands on is its own Dmax: Velvia 4.0/3.2/2.2 of 255,
    # -- negatives through SCAN_DI 1.5, and the POLAROID materials 33 to 46,
    # -- because their Dmax really is about 1.6 D. 14 of the 44 reversal stocks
    # -- floor at or under 4 codes; the rest visibly lift.
    # -- ⚠ AND ONE MEASURED COST THE OWNER MUST SEE BEFORE SWITCHING: at s = 0
    # -- the 18 % mid-grey anchor misses its 12 % bound on the four lowest-Dmax
    # -- stocks -- POLAROID_410 0.1669, POLAROID_42 0.1631, POLAROID_47 0.1474,
    # -- POLAROID_51 0.1429, against a worst of 0.0876 at s = 1. Traced to
    # -- stage 12b: Callier multiplies the patch's NET density by 1.62 on a
    # -- monochrome stock (0.4992 -> 0.7591 D measured on POLAROID_42), so the
    # -- scalar solver's residual against the full pixel pass is amplified by
    # -- that factor, and s = 0 no longer compresses the result. That is a
    # -- SOLVER-ACCURACY limit on four stocks, not a contradiction in the
    # -- normalisation -- see doc/DIGITIZATION_QUEUE.md § 0.0e.
    # -- ⚠ WHAT THIS FIELD IS NOT: a fix for the cause. 9.73 of those 13.06
    # -- points come from stage 12's `dye_matrix`, a unit-row-sum SATURATION
    # -- operator (Velvia `_dye(-0.42)`, diagonal 1.28) that is not bounded by
    # -- the channel's own range: at Dmax with the other two records low it
    # -- returns 1.28 * 3.307 - 0.14 * (D_r + D_g) = 4.14, which no curve can
    # -- produce. Bounding it needs a measurement nobody in this corpus has.
    # -- Retuning that tier-3 estimate, or lowering a traced Dmax, to hide the
    # -- overshoot is exactly the move this project forbids.
    black_point_stretch: float = 1.0
    grain_scale: float = 1.0
    halation_scale: float = 1.0
    # -- C22, 2026-08-23: how DIRECTIONAL the reader's optics are. 0 = a diffuse
    # -- integrating sphere; 1 = a condenser or point source, which sees the
    # -- film's full Callier coefficient. Anything between mixes the two.
    # -- ⚠ IT DOES NOTHING ON COLOUR STOCK BY CONSTRUCTION -- Callier is silver
    # -- scattering and a dye image has essentially none, so all 107 colour
    # -- profiles carry Q = 1.0. It moves the 69 monochrome stocks only.
    # --
    # -- ⚠⚠ DEFAULT RAISED 0.0 -> 0.853 ON 2026-09-06, BY OWNER DECISION, AND IT
    # -- MOVES PIXELS. 0.853 is 1 - E with E = 0.1471, the collected-scatter
    # -- fraction fitted to Trumpy & Gschwind 2015 Fig. 5 (after Streiffert
    # -- 1947) by `trumpy_callier_q.py` -- the same fit that gave beta 1.6746.
    # -- At this setting a monochrome stock reads Q 1.485 at net density 1.0,
    # -- i.e. +0.485 D, and Q 1.569 in the deep toe.
    # -- ⚠ WHAT IT IS NOT: a measurement of any scanner. E is a property of
    # -- STREIFFERT'S 1947 DENSITOMETER, and the owner adopted it as a stated
    # -- provisional stand-in for a reader geometry rather than as a claim about
    # -- his own rig. It is an ESTIMATE in the release vocabulary, not a
    # -- measurement, and it is a RENDER CONTROL rather than film data -- no
    # -- profile field changed.
    # -- ⚠ THE FREE ROUTE TO A REAL NUMBER IS RECORDED SO THIS DOES NOT CALCIFY:
    # -- scan one negative twice, once normally and once with a diffuser over
    # -- the light source; the density difference across the tone scale IS Q(D)
    # -- for that scanner, and with beta already stored it solves for s. Until
    # -- then this value stands and says what it is.
    # -- ⚠ 0.0 REMAINS THE VALUE THAT REPRODUCES THE STORED CHARACTERISTIC
    # -- CURVES EXACTLY, because those are DIFFUSE densities. Anyone comparing a
    # -- render against a datasheet must set it back to 0.
    scanner_specular: float = 0.853
    coupler_scale: float = 1.0
    scanner_f50: float = 0.0        # 0 = take from print stock

    # -- measured spectral sensitivity (see the SPECTRAL block above) --------
    # Consumes SpectralSensitivity.log_s_* where the stock carries it. Each
    # flag substitutes a DERIVED quantity for an authored proxy, and each falls
    # back silently to the proxy for the 85 of 161 stocks that have no curves.
    #
    # spectral_balance: ON. Replaces the three assumed peak wavelengths of
    #   balance_gains() with the full measured sensitisation. Safe by
    #   construction -- it is a ratio of the same integral under two
    #   illuminants, normalised to green, so it cannot change overall exposure
    #   and cannot double-count anything downstream.
    #
    # spectral_mono: ON since 2026-08-29, AND THE REASON IS PARITY, NOT A
    #   CHANGE OF MIND ABOUT THE PHYSICS.
    #
    #   ⚠ WHAT THIS FLAG'S PREVIOUS "OFF" ACTUALLY MEANT. The C++ engine has
    #   never had this flag. Algo_07_Sim.cpp calls AlgoSpectralMonoWeights()
    #   unconditionally and falls back to profile.spectral_weights only when
    #   the stock carries no pan curve. So while this side sat OFF, the two
    #   engines rendered DIFFERENT monochrome images for the 24 stocks that
    #   carry a traced pan curve -- both running, both plausible, which is
    #   exactly the failure mode cpp_parity.py was written to prevent and
    #   exactly the one it does not cover (it audits the grain and MTF laws
    #   only). Turning this ON does not introduce a divergence; it ENDS one
    #   that had been shipping. Measured worst case: KODAK_PLUS_X_125, blue
    #   weight 0.110 authored against 0.502 derived.
    #
    #   The earlier argument for OFF still stands as an argument about the
    #   MODEL and is not withdrawn: the derived triple depends on the assumed
    #   primary lobe width (_PRIMARY_WIDTH_NM), which is an assumption, and an
    #   independent analysis on 2026-08-03 reached the same conclusion about
    #   the same construction. The honest fix remains a scene spectral model
    #   (reflectance basis functions under a stated illuminant, Smits or
    #   Jakob-Hanika class), built deliberately -- not a reprojection of data
    #   the database already holds. What changed is that "OFF" was never
    #   buying that caution: it bought a silent Python/C++ split while the
    #   plugin the owner ships derived anyway. One assumption, applied once,
    #   in both engines, is strictly better than one assumption applied in one
    #   of them.
    #
    #   GUARD. spectral_monochrome_weights() refuses any stock whose peak
    #   sensitisation lies beyond the basis's reach, or which carries more
    #   than _SPECTRAL_OUT_OF_REACH_MAX of its energy past it, measured on the
    #   curve's own samples (see stored_layer_sensitivities). KONICA_INFRARED_
    #   750 is refused: peak 750 nm, 0.437 of its energy out of reach, and it
    #   derives to a BLUE-dominant (0.161, 0.193, 0.646) against an authored,
    #   correct, red-dominant (0.55, 0.15, 0.30).
    #
    #   ⚠ THE GUARD DOES NOT CATCH ROLLEI_INFRARED_400, AND THAT IS NOT A
    #   THRESHOLD THAT NEEDS TUNING. That stock's stored curve is the
    #   UNFILTERED sensitisation: it peaks at 410 nm and puts only 0.028 of
    #   its energy past 700 nm, so it is not an infrared-dominant curve and no
    #   out-of-reach test can honestly call it one. Its authored red-dominant
    #   (0.52, 0.20, 0.28) encodes an assumed deep-red/IR taking filter that
    #   NO FIELD IN THE PROFILE RECORDS. The two triples answer different
    #   questions and the database cannot currently tell them apart. Both
    #   engines therefore now derive for this stock, which is correct for the
    #   data on file and wrong for the way the film is used. Raised as queue
    #   row C39 (a taking_filter carrier); do not "fix" it by lowering the
    #   threshold, which would start refusing ordinary panchromatic stocks.
    #
    # spectral_taking: OFF, deliberately. The derived matrix is physically the
    #   right object, but the pipeline already carries cross-channel mixing in
    #   dye_matrix and in InterimageSpec, and substituting a strongly-mixing
    #   taking matrix on top of those would apply the same physics twice --
    #   the double-counting failure the requirements document warns about.
    #   Enabling this is an experiment that must be validated against a
    #   measured reference, not a default. The derived matrix is available
    #   from spectral_taking_matrix() and reported by
    #   spectral_exposure_report() so the disagreement stays visible.
    spectral_balance: bool = True
    spectral_mono: bool = True
    spectral_taking: bool = False
    misreg_scale: float = 1.0       # multiplies the stock's own registration error
    print_grain: bool = True
    flare: float = -1.0             # <0 = use the stock's default_flare
    generations: int = 0            # intermediate interpositive/dupe-negative pairs
    dupe_stock: DupeStockCtrl | str = "DUPE_FINE_GRAIN"
    reseau: bool = True             # allow the additive colour grid
    bit_depth: int = 16
    seed: int = 12345
    max_dim: int = 0                # 0 = no downscale
    # -- schema v4 -----------------------------------------------------------
    #: Lens corner falloff in stops; <0 = use the stock's era default.
    vignette: float = -1.0
    #: Scales all three CoatingSpec defects together (coating field, gate
    #: buckling, edge fog). 0.0 disables them; 1.0 = as profiled.
    coating_scale: float = 1.0
    #: Frame number within the clip.
    #:
    #: ⚠ SINCE v48 THE GRAIN STAGE USES IT TOO, AND THAT IS THE FIX FOR F1.
    #: Until then only the coating field read it -- to slide its
    #: machine-direction structure by one frame pitch -- and grain was drawn
    #: from a generator seeded once per render, so every frame of a clip carried
    #: an IDENTICAL grain field while both C++ engines re-rolled it per frame.
    #: The reference and the production engines were modelling different
    #: physics, and no harness in this repository could see it, because every
    #: harness compares one frame.
    #:
    #: Any frame can still be rendered independently and out of order: every
    #: field is a pure function of (seed, frame, stage, ordinal).
    frame_index: int = 0
    #: Fraction of grain VARIANCE that is frame-locked scanner fixed-pattern
    #: noise rather than emulsion grain. 0 = a pure emulsion field, the film
    #: behaviour; 1 = a pattern identical on every frame, which is what a
    #: sensor's non-uniformity is.
    #:
    #: ⚠⚠ A CONTROL AND NOT A DATABASE COLUMN, ON PURPOSE. Fixed-pattern noise
    #: is a property of the INSTRUMENT, not of the film -- the same negative
    #: scanned on two machines carries two different patterns, and scanned on
    #: none carries neither. Putting it on `FilmProfile` would state that a
    #: 1936 emulsion has a sensor, which is the same category error this
    #: project already corrected for `callier_q` (film x geometry, queue C22).
    #:
    #: ⚠ DEFAULT 0.0, so no shipped render changes. The corpus holds NO
    #: measurement of any scanner's fixed-pattern noise -- queue M1b/P88 proved
    #: the spectral half absent across all eight scanners in the one document
    #: that measures any, and this is the same gap wearing a different hat. The
    #: mechanism is built and inert, which is this project's standard treatment
    #: for a law whose data has not arrived.
    scanner_fixed_pattern: float = 0.0
    # -- schema v48 (FGS-DDS-001 Rev. A §18.4) -------------------------------
    #: Which grain spectrum to render (`fp.GRAIN_SPECTRUM_MODELS`).
    #:
    #: ⚠ THE DEFAULT IS THE PHYSICAL MODEL, WHICH IS A DEPARTURE FROM THE
    #: SPECIFICATION AND IS THE OWNER'S INSTRUCTION. Spec §10.3 keeps
    #: `boolean_jinc` opt-in until gate S5 and §18.1 requires v48 defaults to
    #: reproduce v47 bit-identically (R-N2). The instruction of 2026-09-19 was
    #: to REPLACE the existing grain model, so the physical spectrum is the
    #: default and `legacy_gaussian` is retained, exact, as the comparison path.
    #: R-N2 is therefore re-expressed rather than met: v48 in `legacy_gaussian`
    #: reproduces the v47 SPECTRUM exactly, not the v47 pixels, because the
    #: frame-keyed generator that closes F1 necessarily changes every pixel.
    #:
    #: ⚠⚠ SWITCHING TO `boolean_jinc` IS A VISIBLE CHANGE ON 186 STOCKS AND IS
    #: NOT A REFINEMENT. The physical diameter is a median 6.7x smaller than
    #: `clump_um`, so the spectrum's half-power frequency moves UP by that
    #: factor, and because the calibration is pinned through the 48 um aperture
    #: -- which sees almost none of the band that moved -- the VISIBLE variance
    #: goes UP, not down. Measured over the whole corpus with the shipped code,
    #: new/old rendered variance at three scanner bandwidths:
    #:
    #:     f50 = 40 c/mm    median 1.148    range 1.015 - 1.711
    #:     f50 = 80 c/mm    median 1.533    range 1.016 - 3.515
    #:     f50 = 120 c/mm   median 2.196    range 1.018 - 5.398
    #:
    #: Per stock at 40 / 80 / 120:
    #:
    #:     VISION3 50D      1.044 / 1.102 / 1.200
    #:     PORTRA 400       1.077 / 1.215 / 1.453
    #:     VISION3 500T     1.133 / 1.494 / 2.104
    #:     T-MAX 400        1.212 / 1.894 / 3.021
    #:     SVEMA FOTO 250   1.222 / 1.889 / 2.706
    #:     ILFORD HPS       1.028 / 1.033 / 1.043
    #:
    #: ⚠ ILFORD HPS IS THE ONE THAT VALIDATES THE REST. It is one of the five
    #: stocks whose `clump_um` is a BBC T-101 measurement rather than an
    #: estimate, so its diameter barely moves and neither does its render. The
    #: stocks that move are exactly the ones whose grain size was a guess.
    #:
    #: Up to five times more visible grain at high scan bandwidth, which is the
    #: opposite of the intuitive "finer grain, less visible" and is stated
    #: nowhere in the specification. It is the single largest adoption risk in
    #: this change: on a 4K scan it will read as a regression to anyone who has
    #: not been told.
    spectrum_model: str = "boolean_jinc"
    #: still | motion | frozen (spec §14.4).
    #:
    #: ⚠ `still` IS THE PHYSICALLY FAITHFUL DEFAULT AND THE PRIOR DOCUMENT WAS
    #: WRONG ABOUT THIS. That document argued independent frames at still
    #: amplitude render 2.19x too loud in motion, because the eye integrates
    #: ~0.2 s. It does -- but a real scanned motion frame ALSO carries full
    #: still-frame granularity, and the viewer integrates real footage and
    #: simulated footage identically, so the factor cancels. `motion` remains
    #: available as a PERCEPTUAL MATCHING control and is not emulsion physics;
    #: `frozen` pins one field for the whole clip, for stills work and A/B.
    grain_temporal_mode: str = "still"
    #: Frames per second. ⚠ 0.0 MEANS "THE HOST DID NOT SUPPLY ONE", AND THAT
    #: IS R-T3 IN LITERAL TERMS: the motion amplitude scale applies "only when
    #: the host explicitly supplies a frame rate; it shall never be a silent
    #: default." A default of 24.0 was exactly the silent default the
    #: requirement forbids -- `motion` mode without an fps would quietly have
    #: assumed cinema and scaled the amplitude by 0.456. Now it raises.
    frame_rate: float = 0.0
    #: gaussian | hybrid (spec §18.4). `hybrid` enables the count-gated
    #: compound-Poisson marginal of R-S6. Default ON, because the gate is
    #: computed per pixel and is inert wherever the Gaussian marginal is
    #: defensible -- which is most of most images, and all of a bright one.
    marginal_model: str = "hybrid"
    #: legacy | measured_pchip | saturating (spec §18.4).
    sigma_model: str = "measured_pchip"

    def flare_for(self, profile: FilmProfile) -> float:
        """Veiling flare fraction to use, honouring the per-stock default."""
        return profile.default_flare if self.flare < 0.0 else self.flare

    def vignette_for(self, profile: FilmProfile) -> float:
        """Lens corner falloff in stops, honouring the per-stock era default."""
        return profile.default_vignette if self.vignette < 0.0 else self.vignette


# ===========================================================================
# Schema-v4 defects: lens vignette, web-coherent coating field, gate buckling,
# narrow-gauge edge fog
# ===========================================================================
def vignette_field(h: int, w: int, stops: float) -> np.ndarray:
    """cos^4(theta) illumination falloff, corner pinned to ``stops`` down.

    Real physics rather than a fitted bowl: off-axis illuminance on a flat
    focal plane falls as cos^4(theta) -- one cosine from the tilted exit
    pupil, one from the tilted image plane, two from the inverse-square
    increase in distance. Mechanical vignetting (hoods, filter stacks, an
    undersized rear element) adds to it in real lenses, which is why period
    glass loses more than geometry alone predicts; that surplus is what the
    per-era ``default_vignette`` figure carries.

    Parametrised by the corner loss so the number in the profile is directly
    meaningful: cos(theta_corner) = 2**(-stops/4), and every other pixel
    interpolates by its true angle, tan(theta) = (r / r_corner) *
    tan(theta_corner). Centre is exactly 1.0 by construction.

    Frame-invariant -- compute once per clip, not per frame.
    """
    if stops <= 0.0:
        return np.ones((h, w), dtype=np.float32)
    cos_c = 2.0 ** (-stops / 4.0)
    tan_c = math.sqrt(max(1.0 / (cos_c * cos_c) - 1.0, 0.0))
    yy = (np.arange(h, dtype=np.float64) - (h - 1) * 0.5) / max((h - 1) * 0.5, 1.0)
    xx = (np.arange(w, dtype=np.float64) - (w - 1) * 0.5) / max((w - 1) * 0.5, 1.0)
    # r normalised so the frame corner is exactly 1.0
    r = np.sqrt((yy * yy)[:, None] + (xx * xx)[None, :]) / math.sqrt(2.0)
    c = 1.0 / np.sqrt(1.0 + (r * tan_c) ** 2)      # cos(theta)
    return (c ** 4).astype(np.float32)


def coating_field(
    h: int,
    w: int,
    frame_w_mm: float,
    frame_h_mm: float,
    spec,                    # film.CoatingSpec
    frame_index: int,
    pitch_mm: float,
    seed: int,
) -> np.ndarray:
    """Web-coherent coating sensitivity field, mean 1.0.

    The geometry is the point. Film is coated as a wide web and slit into
    strips afterwards, so the coating pattern lives in WEB coordinates and
    knows nothing about frame boundaries:

      * across the web (the frame's horizontal axis on 35 mm) the structure
        is fixed for the whole roll -- a left-right gradient that does not
        flicker;
      * along the web (vertical) the film advances ``pitch_mm`` per frame,
        so each frame samples a different stretch. That, and only that, is
        the real emulsion-driven frame-to-frame blink.

    Synthesised as a sum of sinusoids in absolute web coordinates rather
    than as filtered noise. Three reasons, all practical: it is an exact
    function of (web position, seed) so any frame can be rendered
    independently and out of order with no state and no seams; the field
    slides continuously instead of being redrawn; and it costs one small
    low-resolution evaluation plus a bilinear upsample instead of the
    full-resolution FFT pair the pre-v4 code ran on every frame.

    Anisotropy comes from drawing the two frequency axes against their own
    correlation lengths, so the field is streaky along the web the way a
    coating hopper's slow drift actually is.
    """
    sigma = float(spec.coating_sigma)
    if sigma <= 0.0:
        return np.ones((h, w), dtype=np.float32)

    # 4 samples per correlation length is the Nyquist-with-headroom rule; the
    # floor of 24 matters because a large-scale field (corr length comparable
    # to the frame) would otherwise be represented by ~8 samples and upsample
    # into a visibly linear ramp instead of a smooth hump.
    lo_x = int(min(max(4.0 * frame_w_mm / max(spec.coating_corr_across_mm, 1e-6),
                       24.0), 192.0))
    lo_y = int(min(max(4.0 * frame_h_mm / max(spec.coating_corr_along_mm, 1e-6),
                       24.0), 192.0))

    # Absolute web offset of this frame, in millimetres. Unperforated formats
    # (sheet, instant) have pitch 0: a single exposure, so no advance.
    y_off_mm = float(frame_index) * float(pitch_mm)

    rng = np.random.default_rng(seed ^ 0x00C0A71C)

    # TWO components, because a coating hopper has two distinct signatures and
    # collapsing them into one 2D field gets the temporal behaviour wrong:
    #
    #   STATIC cross-web profile -- slot and nozzle imperfections are fixed
    #   hardware, so they lay down streaks at fixed x for the entire roll.
    #   A function of x alone: identical on every frame, never flickers.
    #
    #   DRIFTING 2D field -- coating flow wandering over machine time. This is
    #   the part that slides with the web and produces the frame-to-frame
    #   blink.
    #
    # Split evenly in variance (hence /sqrt(2) each). Verified: with a single
    # 2D field the cross-web profile decorrelated frame to frame, contradicting
    # the fixed-streak physics this docstring describes.
    n_comp = 64
    half = sigma / math.sqrt(2.0)

    xs_mm = np.linspace(0.0, frame_w_mm, lo_x, dtype=np.float64)
    ys_mm = np.linspace(0.0, frame_h_mm, lo_y, dtype=np.float64) + y_off_mm

    # -- static cross-web streaks -------------------------------------------
    fxs = rng.normal(0.0, 1.0 / (2.0 * math.pi * spec.coating_corr_across_mm),
                     n_comp)
    phs = rng.uniform(0.0, 2.0 * math.pi, n_comp)
    static = np.zeros(lo_x, dtype=np.float64)
    for k in range(n_comp):
        static += np.cos(2.0 * math.pi * fxs[k] * xs_mm + phs[k])
    static *= half / math.sqrt(n_comp * 0.5)

    # -- drifting 2D field ---------------------------------------------------
    fx = rng.normal(0.0, 1.0 / (2.0 * math.pi * spec.coating_corr_across_mm),
                    n_comp)
    fy = rng.normal(0.0, 1.0 / (2.0 * math.pi * spec.coating_corr_along_mm),
                    n_comp)
    ph = rng.uniform(0.0, 2.0 * math.pi, n_comp)
    drift = np.zeros((lo_y, lo_x), dtype=np.float64)
    for k in range(n_comp):
        drift += np.cos(2.0 * math.pi * (fy[k] * ys_mm[:, None]
                                        + fx[k] * xs_mm[None, :]) + ph[k])
    drift *= half / math.sqrt(n_comp * 0.5)

    lo = (1.0 + static[None, :] + drift).astype(np.float32)
    return _bilinear_upsample(lo, h, w)


def _bilinear_upsample(lo: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear upsample a small field to (h, w). Source stays cache-resident."""
    lh, lw = lo.shape
    if lh == h and lw == w:
        return lo
    yi = np.linspace(0.0, lh - 1.0, h)
    xi = np.linspace(0.0, lw - 1.0, w)
    y0 = np.minimum(yi.astype(np.int32), lh - 2)
    x0 = np.minimum(xi.astype(np.int32), lw - 2)
    fy = (yi - y0).astype(np.float32)[:, None]
    fx = (xi - x0).astype(np.float32)[None, :]
    tl = lo[y0][:, x0]
    tr = lo[y0][:, x0 + 1]
    bl = lo[y0 + 1][:, x0]
    br = lo[y0 + 1][:, x0 + 1]
    top = tl + (tr - tl) * fx
    bot = bl + (br - bl) * fx
    return (top + (bot - top) * fy).astype(np.float32)


def corner_defocus(plane: np.ndarray, loss: float) -> np.ndarray:
    """Radially increasing softness from film buckling in the camera gate.

    The pressure plate holds the middle of the frame against the aperture
    plate; the corners of a curling base lift out of the focal plane, so
    they are focused on a surface that is no longer where the lens put its
    image. Corner SOFTNESS, never corner darkening -- the two get conflated
    constantly and they are different mechanisms.

    Implemented as a 5-tap separable blur blended in by normalised radius
    squared, not as a second frequency-domain pass. A spatially varying
    kernel cannot be expressed as one transfer function, so the honest FFT
    version needs a second full transform per channel -- measured at HD that
    is about as expensive as the entire emulsion-MTF stage. The effect is
    mild by nature (``buckle_mtf_loss`` is 0.03-0.30), so a small fixed
    kernel blended radially is within its own uncertainty and costs a few
    operations per pixel.
    """
    if loss <= 0.0:
        return plane
    h, w = plane.shape
    k = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32)
    k /= k.sum()
    pad = np.pad(plane, ((0, 0), (2, 2)), mode="edge")
    tmp = np.zeros_like(plane)
    for i in range(5):
        tmp += k[i] * pad[:, i:i + w]
    pad2 = np.pad(tmp, ((2, 2), (0, 0)), mode="edge")
    blur = np.zeros_like(plane)
    for i in range(5):
        blur += k[i] * pad2[i:i + h, :]
    yy = (np.arange(h, dtype=np.float32) - (h - 1) * 0.5) / max((h - 1) * 0.5, 1.0)
    xx = (np.arange(w, dtype=np.float32) - (w - 1) * 0.5) / max((w - 1) * 0.5, 1.0)
    r2 = ((yy * yy)[:, None] + (xx * xx)[None, :]) * np.float32(0.5)  # corner=1
    wgt = (np.float32(loss) * r2).astype(np.float32)
    return (plane * (1.0 - wgt) + blur * wgt).astype(np.float32)


def edge_fog_density(h: int, w: int, frame_w_mm: float, spec) -> np.ndarray:
    """Additive density near the film edges. Gauge-driven, not era-driven.

    Standard 8 is 16 mm film slit down the middle AFTER processing, so its
    frame sits at the film edge with no trimmed margin: light leaking past
    the edge of the roll and development edge effects both land inside the
    picture. On 35 mm the margins carry the perforations and get trimmed, so
    this is negligible and the spec leaves it at zero.

    Applied in the density domain, after development, because that is where
    both contributors end up and where the spec's units live.
    """
    if not spec.has_edge_fog or frame_w_mm <= 0.0:
        return np.zeros((h, w), dtype=np.float32)
    x_mm = np.linspace(0.0, frame_w_mm, w, dtype=np.float64)
    d_edge = np.minimum(x_mm, frame_w_mm - x_mm)     # distance to nearer edge
    prof = spec.edge_fog_density * np.exp(-d_edge / spec.edge_fog_mm)
    return np.repeat(prof[None, :].astype(np.float32), h, axis=0)


# ===========================================================================
# The pipeline
# ===========================================================================
def simulate(
    linear_rgb: np.ndarray,
    profile: FilmProfile,
    settings: RenderSettings,
) -> np.ndarray:
    """Run one film stock over a linear-light image. Returns linear-light RGB.

    Args:
        linear_rgb: (h, w, 3) float32 linear light, nominally 0..1 with 0.18
            representing mid grey. Values above 1.0 are welcome and useful --
            real negative has many stops of headroom above diffuse white.
        profile: The film stock.
        settings: Render options.

    Returns:
        (h, w, 3) float32 linear light, display referred.
    """
    # ⚠ FIRST, BEFORE ANYTHING READS A CURVE. The anchor solve, stage 8, the
    # grain amplitude and the dupe chain all read `profile.curves`, so a variant
    # that replaced them later would leave the solve anchored to a development
    # the frame is not being given. Replacing the profile here means every
    # consumer downstream sees one consistent film.
    profile = resolve_process_variant(profile, settings.process_variant)
    # ⚠ AFTER the variant and before anything reads a curve: a variant
    # IS a different development, so the two are mutually exclusive in
    # practice and the host disables this control when one is selected.
    # Ordering them this way means that if a host ignores that rule the
    # time is applied to the variant's own curve rather than to a curve
    # the variant then discards.
    profile = resolve_development_time(
        profile, getattr(settings, 'development_minutes', -1.0),
        getattr(settings, 'development_celsius', -1.0))
    # ⚠ LAST OF THE THREE PROFILE RESOLVERS, AND THE ORDER IS
    # CHRONOLOGICAL: a variant and a development time both describe how
    # the film was PROCESSED, and storage happens after processing. A
    # fade applied before a development change would have the film
    # ageing before it was developed.
    profile = resolve_storage_age(
        profile, getattr(settings, 'storage_years', 0.0))

    h, w = linear_rgb.shape[:2]
    negative_width_mm = FORMATS[film_format_key(settings.film_format)]
    px_per_mm = w / negative_width_mm
    rng = np.random.default_rng(settings.seed)
    print_stock = get_print_stock(
        print_stock_key(settings.print_stock) or profile.default_print)

    # A black and white negative goes onto black and white print stock, so the
    # print must be neutral. Printing it through a colour stock's three slightly
    # different curves leaves a faint but measurable cast on what should be a
    # pure greyscale image.
    if profile.is_monochrome and not profile.is_reversal:
        neutral = print_stock.curves.g
        print_stock = replace(
            print_stock,
            curves=RGBCurves(neutral, neutral, neutral),
            dye_matrix=IDENTITY3,
        )

    # ⚠ TWO GRIDS, AND UNTIL 2026-09-11 THERE WAS ONLY ONE. This line used to
    # read `FreqGrid(h, w, px_per_mm, profile.grain.anisotropy)`, and that one
    # object was then handed to EVERY frequency-domain stage in the renderer.
    #
    # `GrainSpec.anisotropy` is a property of the GRAIN: it models emulsion
    # coating flow, which lengthens the vertical correlation of the developed
    # crystal field. Sharing the stretched grid meant that on the 32 stocks
    # carrying a non-default value (1.02 to 1.10) the same vertical stretch was
    # silently applied to veiling flare, HALATION scatter, EMULSION MTF, the
    # DIR coupler blur, SCAN MTF, the duplication MTF and the reseau
    # reconstruction. None of those is grain, and none has any physical reason
    # to inherit a coating-flow figure -- halation is light scattering in the
    # base, and an MTF is a lens-and-emulsion transfer.
    #
    # It was found on 2026-09-11 while closing the OPPOSITE defect in the C++
    # engines, which read the field nowhere at all. Both were wrong, in
    # opposite directions: C++ applied it to nothing, the reference applied it
    # to everything. The agreed model is that it applies to the camera
    # negative's grain field and to nothing else.
    #
    # `grid` is therefore isotropic, and is what every optical stage uses.
    grid = FreqGrid(h, w, px_per_mm)

    # ⚠ THE SECOND, ANISOTROPIC `grain_grid` IS GONE IN v48, AND ITS ABSENCE IS
    # THE POINT. Stage 11 no longer evaluates a transfer on a frequency grid at
    # all: it blurs with the Gaussian mixture, and anisotropy is a ratio between
    # the two axis sigmas of each blur -- which is exactly how the C++ engines
    # have always applied it. Same law, one implementation instead of two, and
    # one fewer h x w float32 grid allocated per render.
    #
    # It stays confined to the camera negative's grain. Dupe and print emulsions
    # are coated in different factories and carry no anisotropy of their own;
    # attributing the camera negative's coating flow to them would be inventing
    # a measurement, which is why those two call sites pass 1.0 and the C++
    # twins pass ALGO_GRAIN_ANISOTROPY_NONE.

    # -- 2. relative exposure ------------------------------------------------
    exposure = (linear_rgb / np.float32(MID_GREY)).astype(np.float32)
    exposure *= np.float32(2.0**settings.exposure_stops)

    # -- 2b. taking filters --------------------------------------------------
    # Identity for an ordinary integral tripack. For a Technicolor
    # beam-splitter camera the taking filters overlap heavily, and mixing the
    # records in *exposure* is the physical origin of that palette -- it cannot
    # be reproduced by a matrix applied later, because the characteristic curve
    # sits in between and is nonlinear.
    take = np.asarray(profile.taking_matrix, dtype=np.float32)
    # MEASURED SPECTRAL PATH, opt-in only (settings.spectral_taking). See the
    # field's comment in RenderSettings for why this is not a default: the
    # derived matrix mixes channels strongly and the pipeline already carries
    # mixing downstream, so switching it on without a measured reference
    # applies the same physics twice.
    if settings.spectral_taking:
        derived = spectral_taking_matrix(profile, settings.scene_kelvin)
        if derived is not None:
            take = derived
    if not np.allclose(take, np.eye(3, dtype=np.float32)):
        exposure = (exposure.reshape(-1, 3) @ take.T).reshape(h, w, 3)
        exposure = np.ascontiguousarray(exposure, dtype=np.float32)

    # -- 3. stock colour balance --------------------------------------------
    if settings.wb_strength > 0.0 and not profile.is_monochrome:
        # MEASURED SPECTRAL PATH: integrate the stock's own per-layer
        # sensitivity against the two blackbody SPDs instead of sampling three
        # assumed peak wavelengths. Falls back to the proxy when the stock
        # carries no curves. Measured difference at 3200 K on a daylight
        # stock: red gain 1.65-1.69 derived against 1.32 from the proxy, and
        # it varies per stock, which a fixed-peak proxy cannot express.
        gains = None
        if settings.spectral_balance:
            gains = spectral_balance_gains(profile, settings.scene_kelvin)
        if gains is None:
            gains = balance_gains(settings.scene_kelvin, profile.balance_kelvin)
        for c in range(3):
            g = 1.0 + (gains[c] - 1.0) * settings.wb_strength
            exposure[:, :, c] *= np.float32(g)

    # -- 3b. veiling flare from the taking lens -------------------------------
    # A lens effect, not an emulsion one, but era of glass and era of stock go
    # together. Uncoated pre-1940 lenses scattered 6-14% of the light entering
    # them into a broad haze across the frame; anti-reflection coating cut that
    # below 1%.
    #
    # This is the difference between "soft" and "old". Flare lifts the black
    # floor and compresses contrast globally, and no amount of grain, curve or
    # MTF work reproduces it -- a period emulsion rendered without it still has
    # modern blacks, which is the main reason vintage profiles disappoint.
    #
    # Two components: a uniform veil over the whole frame, and a very broad
    # local glare. Direct light is scaled down by the same fraction, so total
    # energy is preserved rather than the image simply being lifted.
    flare = settings.flare_for(profile)
    if flare > 0.0:
        lum = (
            0.30 * exposure[:, :, 0]
            + 0.59 * exposure[:, :, 1]
            + 0.11 * exposure[:, :, 2]
        ).astype(np.float32)
        veil = float(lum.mean())
        broad = apply_transfer(lum, grid.multi_gaussian((1500.0, 6000.0, 20000.0),
                                                        (0.45, 0.35, 0.20)))
        scattered = (0.5 * np.float32(veil) + 0.5 * broad).astype(np.float32)
        exposure *= np.float32(1.0 - flare)
        exposure += (np.float32(flare) * scattered)[:, :, None]
        del lum, broad, scattered

    # -- 4. coating unevenness ----------------------------------------------
    # -- 4b. lens vignette x web-coherent coating field (schema v4) ----------
    # Two mechanisms, one pass. Both are pure per-pixel multipliers on
    # exposure, so they fuse into a single field and a single multiply -- the
    # marginal cost over stage 3 is one extra stream read, not two passes.
    #
    # They are kept conceptually apart because they are different physics and
    # different geometry:
    #   * the vignette is the LENS. cos^4(theta), locked to the frame, fixed
    #     for the whole clip, present in every era (modern glass still loses
    #     0.3-0.5 stop in the corners).
    #   * the coating field is the FILM, and it lives in WEB coordinates. It
    #     cannot be locked to frame corners, because the coating machine never
    #     knew where the frames would fall. Fixed across the web, sliding one
    #     frame pitch per frame along it.
    #
    # This replaces the pre-v4 behaviour, which synthesised isotropic mottle
    # with a full-resolution FFT pair on every frame: wrong geometry (blobs,
    # not streaks), wrong temporal behaviour (frozen for a whole sequence
    # because it was seeded only from settings.seed), and roughly 25x the cost
    # of the low-resolution synthesis used here.
    vig_stops = settings.vignette_for(profile)
    coat = profile.coating
    cs = max(settings.coating_scale, 0.0)
    field = None
    if vig_stops > 0.0:
        field = vignette_field(h, w, vig_stops)
    if cs > 0.0 and coat.has_coating_field:
        eff = dataclasses.replace(
            coat, coating_sigma=coat.coating_sigma * cs
        )
        cf = coating_field(
            h, w, negative_width_mm, negative_width_mm * h / max(w, 1),
            eff, settings.frame_index,
            frame_pitch_mm(film_format_key(settings.film_format)), settings.seed,
        )
        field = cf if field is None else (field * cf)
    if field is not None:
        exposure *= field[:, :, None]
        del field

    # -- 5. halation (in linear exposure) -----------------------------------
    hal = profile.halation
    if hal.active and settings.halation_scale > 0.0:
        # ⚠ ONE KERNEL WHEN THE RADII ARE SHARED, THREE WHEN THEY ARE NOT (C21,
        # schema v11). Building three identical kernels would cost two extra FFT
        # transfers per frame AND -- the part that matters -- would not be
        # bit-identical to the v10 path, because the same value summed in a
        # different order rounds differently in float32. Every stock in the file
        # ships at 1.0, so the shared branch is what actually runs; the
        # per-channel branch exists for the day a measured halo width lands.
        shared = hal.radii_are_shared
        scatter = grid.multi_gaussian(hal.radii_um, hal.weights) if shared else None
        thr = np.float32(2.0**hal.threshold_stops)
        # A loose knee leaks a surprising amount of glow into the mid tones: at
        # CineStill's gain of 1.05 a knee of 0.35*thr lifted an 18% grey card by
        # 16%. Keep it tight enough to stay a highlight effect.
        knee = np.float32(float(thr) * 0.15)
        lum = (
            0.30 * exposure[:, :, 0]
            + 0.59 * exposure[:, :, 1]
            + 0.11 * exposure[:, :, 2]
        ).astype(np.float32)
        gains = hal.gains()
        for c in range(3):
            if gains[c] <= 0.0:
                continue
            # Halation source blends this layer's own exposure with total
            # luminance: light of every wavelength penetrates and returns, but
            # the returning light is weighted towards the deepest-penetrating.
            src = (0.5 * exposure[:, :, c] + 0.5 * lum).astype(np.float32)
            above = _softplus(src - thr, float(knee))
            # Energy conserving: light that scatters away from a point is
            # removed from it and deposited in the surround, rather than being
            # created out of nothing. So a large evenly-lit highlight shows no
            # net change in its interior -- correct, because it is already
            # saturated -- while a small bright source blooms into its
            # neighbourhood and loses a little of its own edge. Adding
            # blur(above) alone instead injects a flat-field brightness lift
            # that scales with gain and contaminates the whole exposure scale.
            # The scatter kernel for THIS record. Radius scaling is a property
            # of the return path -- how deep in the pack the record sits -- so it
            # multiplies all three lobes together rather than reshaping the
            # long-tail mixture, which is a property of the base and is shared.
            k = scatter if shared else grid.multi_gaussian(
                hal.radii_for(c), hal.weights)
            exposure[:, :, c] += np.float32(
                gains[c] * settings.halation_scale
            ) * (apply_transfer(above, k) - above)
        del lum

    np.maximum(exposure, np.float32(0.0), out=exposure)

    # -- 6. emulsion MTF on the exposure ------------------------------------
    # Light scatter happens at exposure time, before development, so it blurs
    # the image but not the grain. Red is softest: the red-sensitive layer sits
    # under two other layers of gelatin.
    f50s = profile.mtf.f50s()
    for c in range(3):
        t = grid.mtf(f50s[c], profile.mtf.adjacency, profile.mtf.adjacency_um,
                     spec=profile.mtf, channel=c,
                     use_kernel=settings.mtf_use_kernel)
        exposure[:, :, c] = apply_transfer(exposure[:, :, c], t)
    np.maximum(exposure, np.float32(0.0), out=exposure)

    # -- 6b. corner defocus from film buckling in the gate (schema v4) --------
    # Needs its own pass: a radially varying blur is not one transfer function,
    # so it cannot ride along inside the MTF stage above. Kept to a 5-tap
    # separable kernel blended by radius rather than a second FFT per channel;
    # see corner_defocus() for why that is within the effect's own uncertainty.
    if cs > 0.0 and coat.has_buckle:
        loss = min(coat.buckle_mtf_loss * cs, 0.9)
        for c in range(3):
            exposure[:, :, c] = corner_defocus(exposure[:, :, c], loss)

    # -- 7. collapse to a single emulsion record ------------------------------
    reseau_mask: np.ndarray | None = None
    reseau_pitch_px = 0.0
    if profile.is_monochrome:
        # Weighted by the stock's own spectral sensitivity, not by video luma.
        # For the orthochromatic stock the red weight is 0.02, which is what
        # makes red render black and a blue sky render white.
        # MEASURED SPECTRAL PATH: the weight with which each input primary
        # reaches the single silver record is the pan curve integrated against
        # that primary. The authored triple it replaces is close to video luma
        # (0.27/0.55/0.18), which is what the comment above says it must NOT
        # be; the derived triple for a panchromatic emulsion is much flatter
        # (~0.34/0.35/0.30), which is why panchromatic film renders a blue sky
        # lighter than the eye does.
        sw = None
        if settings.spectral_mono:
            sw = spectral_monochrome_weights(profile)
        if sw is None:
            sw = profile.spectral_weights
        mono = (
            np.float32(sw[0]) * exposure[:, :, 0]
            + np.float32(sw[1]) * exposure[:, :, 1]
            + np.float32(sw[2]) * exposure[:, :, 2]
        ).astype(np.float32)
        exposure = np.repeat(mono[:, :, None], 3, axis=2)
        del mono
    elif profile.has_reseau and settings.reseau:
        spec = profile.reseau
        mask, pitch_px = build_reseau_mask(h, w, px_per_mm, spec)
        if pitch_px < RESEAU_MIN_PITCH_PX:
            # The Dufay pattern has structure at a third of the cell pitch
            # vertically, so it needs at least three pixels per cell to be
            # represented at all. Below that the mask quantises unevenly, the
            # reconstruction picks up a colour bias of 10-20%, and the output is
            # aliasing noise rather than a mosaic. Real scans of these stocks do
            # moire for the same reason, but emitting garbage is not a useful
            # simulation of that, so fall back to a plain monochrome record.
            print(
                f"[WARN] {profile.name}: reseau pitch is {pitch_px:.2f} px "
                f"({spec.lines_per_mm:g} lines/mm at {px_per_mm:.0f} px/mm); "
                f"mosaic disabled. Render at >= "
                f"{round(RESEAU_MIN_PITCH_PX * spec.lines_per_mm * negative_width_mm):d}"
                f" px wide for this format, or >= "
                f"{round(5.0 * spec.lines_per_mm * negative_width_mm):d} px to see it "
                "properly.",
                file=sys.stderr,
            )
        else:
            # Light passes the grid before reaching the emulsion, so this is an
            # exposure-domain operation. Each cell sees the light that its own
            # filter passes -- and because those filters overlap heavily, a cell
            # under the red filter still records a substantial amount of green.
            # That cross-talk is what makes additive colour pastel; treat the
            # filters as pure and the result is more saturated than Kodachrome.
            fm = np.asarray(spec.filter_matrix, dtype=np.float32)
            record = np.zeros((h, w), dtype=np.float32)
            for c in range(3):
                through = (
                    fm[c, 0] * exposure[:, :, 0]
                    + fm[c, 1] * exposure[:, :, 1]
                    + fm[c, 2] * exposure[:, :, 2]
                ).astype(np.float32)
                record += mask[:, :, c] * through
            # Restore the mean level lost to the filters, so the anchor solve
            # (which cannot see the mask) still lands mid grey correctly. The
            # real speed penalty of about 1.7 stops is carried by the stock's
            # exposure_index instead.
            record /= np.float32(spec.neutral_gain())
            exposure = np.repeat(record[:, :, None], 3, axis=2)
            reseau_mask = mask
            reseau_pitch_px = pitch_px
            del record

    # -- 8. characteristic curve: exposure to density -----------------------
    log_e = np.log10(np.maximum(exposure, np.float32(EPS)), dtype=np.float32)
    del exposure

    # -- 7c. reciprocity failure (C8, 2026-08-23) ---------------------------
    # A per-channel shift of log exposure, applied HERE and nowhere earlier.
    # The placement is the physics: reciprocity failure is a property of the
    # EMULSION's response to the light that reached it, so it must sit after
    # everything optical (flare, halation, the emulsion MTF, the record
    # collapse) and before the characteristic curve. Applying it at stage 2
    # with the camera exposure would let the flare and halation stages see
    # light the lens never delivered.
    # Shifting log_e rather than scaling `exposure` is the same arithmetic and
    # one operation cheaper -- and log_e is what stage 8b reads, so the
    # interimage stage sees the same effective exposure the curve did, which is
    # what a real layer would.
    if settings.exposure_time_s > 0.0:
        _recip = reciprocity_log_shift(profile, settings.exposure_time_s)
        for c in range(3):
            if _recip[c] != 0.0:
                log_e[:, :, c] += np.float32(_recip[c])

    curves = profile.curves.as_tuple()
    reversal = profile.is_reversal
    anchors = solve_anchors(
        profile, print_stock, settings.grey_target, settings.coupler_scale,
        settings.scanner_specular, settings.black_point_stretch,
    )
    dens = np.empty((h, w, 3), dtype=np.float32)
    if reversal:
        # A slide records a positive: more light means *less* density. The curve
        # parameters are expressed against negated log exposure, so toe_x
        # governs the highlight end. There is no print stage afterwards.
        for c in range(3):
            dens[:, :, c] = density(
                -(log_e[:, :, c] + np.float32(anchors[c])), curves[c]
            )
    else:
        for c in range(3):
            dens[:, :, c] = density(log_e[:, :, c], curves[c])

    # -- 8b. interimage effects: cross-layer development inhibition (v5) -----
    # The vertical half of the DIR-coupler chemistry whose lateral half is
    # stage 9. Inhibitor released while one layer develops diffuses into its
    # neighbours and suppresses them, so each layer's EFFECTIVE exposure
    # depends on what the other two are doing:
    #
    #     logE_i' = logE_i + sum_{j != i} a_ij * (D_j - d_ref_j)
    #
    # Referencing to the mid-grey density d_ref is what makes this a colour
    # effect rather than a tone effect: on a neutral every (D_j - d_ref) is
    # ~0, the correction vanishes, and the grey scale is untouched. A
    # saturated colour, where the layers disagree, develops against unequal
    # inhibition and separates further -- saturation rising WITHOUT gamma
    # rising, which no per-channel curve can produce.
    #
    # Implicit equation (D depends on logE' depends on D), solved by
    # fixed-point iteration seeded with the densities just computed. Each pass
    # costs one full curve evaluation per channel, and this is the most
    # expensive stage in the chain, so the count is a profile field the
    # renderer honours rather than a hardcoded loop.
    if profile.interimage.active and not profile.is_monochrome:
        apply_interimage(dens, log_e, curves, profile.interimage,
                         anchors, reversal)
    del log_e

    # -- 9. DIR coupler inter-image effects ---------------------------------
    apply_dir_couplers(dens, profile.couplers, grid,
                       settings.coupler_scale, profile.is_monochrome)

    # -- 9c. bromide drag: the machine's directional restraint ---------------
    # ⚠ HERE, AND NOT NEXT TO STAGE 9, EVEN THOUGH BOTH ARE DEVELOPMENT
    # BYPRODUCTS. Stage 9 is inhibitor diffusing through the gelatin -- inside
    # the coating, isotropic, tens of micrometres. This is loaded developer
    # being dragged across the outside of it -- in the bath, one-sided,
    # millimetres to centimetres, and a property of the machine rather than of
    # the film. They are adjacent in the chain because the density stage 9
    # leaves IS what releases the bromide; they are separate stages because
    # nothing about their scale, symmetry or ownership is shared.
    # Inert on all 176 stocks (queue C23): no source in this corpus measures it.
    apply_bromide_drag(dens, profile.processing.bromide_drag,
                       tuple(c.dmin for c in curves),
                       tuple(c.dmax for c in curves),
                       px_per_mm, reversal)

    np.maximum(dens, np.float32(0.0), out=dens)

    # -- 10. scan the image: MTF plus per-channel misregistration -------------
    # The scan stage comes before grain is added, not after, because the
    # scanner's optical MTF is the *pre-sampling* filter: it band-limits both
    # image and grain before the sensor samples them. Grain is therefore
    # generated already band-limited by the same transfer (see below), which is
    # the only way to avoid fine grain aliasing onto the pixel grid.
    scan_f50 = settings.scanner_f50 or print_stock.mtf_f50
    scan_t = grid.mtf(scan_f50, 0.0, 0.0)
    # Registration error is specified on the negative in micrometres, so it
    # scales with resolution like every other spatial quantity. A few
    # micrometres is invisible as a shift but very visible as an absence -- it
    # softens colour edges the way every real film scan is softened. Three-strip
    # Technicolor used tens of micrometres, which is why its edges fringe.
    mis_px = profile.misregistration_um * px_per_mm / 1000.0 * settings.misreg_scale
    for c in range(3):
        t = scan_t
        if mis_px > 0.0 and not profile.is_monochrome:
            dy = float(rng.normal(0.0, mis_px))
            dx = float(rng.normal(0.0, mis_px))
            t = (scan_t * grid.shift(dy, dx)).astype(np.complex64)
        dens[:, :, c] = apply_transfer(dens[:, :, c], t)

    np.maximum(dens, np.float32(0.0), out=dens)

    # -- 10b. narrow-gauge edge fog (schema v4) -------------------------------
    # Additive density, applied after development because that is where both
    # of its causes land: light leaking past the edge of the roll, and
    # development edge effects. Purely a GAUGE matter -- Standard 8 is 16 mm
    # slit down the middle after processing, so its frame sits at the film
    # edge; 35 mm margins carry the perforations and get trimmed away.
    if cs > 0.0 and coat.has_edge_fog:
        fog = edge_fog_density(h, w, negative_width_mm, coat) * np.float32(cs)
        dens += fog[:, :, None]
        del fog

    # -- 11. grain, in the density domain -------------------------------------
    gs = profile.grain
    if settings.grain_scale > 0.0:
        clumps = gs.clumps()
        grain_um = gs.grain_um_rgb()
        # ⚠ THE STAGE NO LONGER KNOWS WHAT THE SPECTRUM IS. It asks
        # `film_profiles.grain_gauss_terms` for a Gaussian mixture and blurs
        # with it. Both spectral models arrive through the same call and the
        # same mixture, so there is no second code path to keep in step and no
        # way for `legacy_gaussian` to drift from `boolean_jinc` in anything
        # except the coefficients -- which is the whole point of the
        # representation (spec §10.3, and this project's v48 block).
        spec_model = settings.spectrum_model
        aperture = fp.grain_aperture_model(spec_model)
        # ⚠ The C++ stage forms its stream seed as `params.seed ^ seed`, where
        # the second operand is the per-call seed the host supplies. The
        # reference has one seed, so the two agree when the host passes zero --
        # which `cpp_parity` drives it to do, and which is the contract.
        grain_seed = int(settings.seed) & 0xFFFFFFFF
        # The scan MTF band-limits grain BEFORE the sensor samples it, because
        # the scanner lens sits between the film and the sensor. That is why
        # stage 10 runs before this one, and why the band limit is folded into
        # every mixture term's sigma rather than applied afterwards.
        scan_sigma_px = fp.scan_sigma_mm(scan_f50) * px_per_mm
        # ⚠ THE FRAME INDEX IS WHAT CLOSES F1. Until v48 this stage drew from a
        # generator seeded once per render, so every frame of a clip carried
        # the SAME grain -- welded to the image, while the C++ engines re-rolled
        # it per frame. Real film is a fresh emulsion sample every frame.
        frame_index = int(settings.frame_index)
        # ⚠ THE FILM SUPPLIES THE CORRELATION, THE RENDER SUPPLIES THE PATTERN.
        # `grain_frame_correlation` is emulsion physics and lives on the stock;
        # `scanner_fixed_pattern` is instrument physics and lives on the render.
        # Both are 0.0 by default, which is the pre-v49 path exactly.
        grain_rho = float(profile.temporal.grain_frame_correlation)
        grain_fixed = float(settings.scanner_fixed_pattern)
        # Temporal mode (spec §14.4). `still` is the default and is the
        # physically faithful one -- see the RenderSettings note. `frozen` pins
        # one field for the whole clip; `motion` keeps independent fields and
        # scales the amplitude for perceptual matching only.
        temporal_scale = 1.0
        if settings.grain_temporal_mode == "frozen":
            frame_index = 0
        elif settings.grain_temporal_mode == "motion":
            # ⚠ R-T3: refuse rather than assume. A silent 24 fps here would be
            # a 0.456x amplitude change nobody asked for.
            if not (settings.frame_rate > 0.0):
                raise ValueError(
                    "grain_temporal_mode='motion' needs an explicit "
                    "frame_rate; the perceptual scale is 1/sqrt(fps * 0.2) and "
                    "defaulting it would silently rescale grain (R-T3)")
            temporal_scale = temporal_grain_scale(settings.frame_rate)
        # ⚠⚠ NO scanner_fixed_pattern FREEZE HERE, AND ITS ABSENCE IS THE
        # POINT. Spec §18.2 asks the stage to freeze the grain field for a
        # stock whose traced noise is the scanner's; R-T5 says the grain stage
        # "shall introduce no frame-locked noise component". A frozen field is
        # exactly such a component, so the two cannot both be honoured and R-T5
        # wins -- it is a requirement, §18.2 is a schema note, and freezing
        # would in any case render the wrong quantity very steadily rather than
        # fixing the mis-attributed measurement underneath.
        # `GrainSpec.grain_temporal_class` is declarative and `validate`
        # refuses any value but "emulsion".

        if profile.is_monochrome or reseau_mask is not None:
            # One silver image means one grain field, identical in all three
            # channels -- not three independent ones. This covers the additive
            # colour stocks too: a reseau stock has a single panchromatic
            # emulsion behind the filter grid, so it cannot have per-layer grain.
            terms = fp.grain_gauss_terms(
                spec_model, clump_um=clumps[1], clump_gain=gs.clump_gain,
                grain_um=grain_um[1], size_sigma_log=gs.size_sigma_log)
            field = make_grain_field(
                h, w, px_per_mm, terms, gs.rms_granularity, scan_sigma_px,
                gs.anisotropy, grain_seed, frame_index, RngStage.GRAIN_G,
                aperture, grain_rho, grain_fixed)
            fields = (field, field, field)
        else:
            # Per-channel RMS: rms_rgb() falls back to the scalar where the
            # profile sets no override. This is where a tripack's blue layer
            # gets its 1.3x noise (topmost, fastest emulsion) and where
            # Technicolor's three physically different B&W records diverge.
            # (The schema always promised this; the renderer used the scalar
            # for all three channels until 2026-08-01 -- silent bug.)
            rms_c = gs.rms_rgb()
            streams = (RngStage.GRAIN_R, RngStage.GRAIN_G, RngStage.GRAIN_B)
            fields = tuple(
                make_grain_field(
                    h, w, px_per_mm,
                    fp.grain_gauss_terms(
                        spec_model, clump_um=clumps[c],
                        clump_gain=gs.clump_gain, grain_um=grain_um[c],
                        size_sigma_log=gs.size_sigma_log),
                    rms_c[c], scan_sigma_px, gs.anisotropy,
                    grain_seed, frame_index, streams[c], aperture,
                    grain_rho, grain_fixed)
                for c in range(3)
            )
        # sigma(D) SHAPE (queue item C1, 2026-08-18) and its LEVEL (C1b, same
        # day). `fp.grain_sigma` is the one definition of both; this block only
        # multiplies the stock's rms field by it.
        #
        # ⚠ THE `legacy_mid` FACTOR THAT USED TO BE HERE IS GONE. It was
        # sqrt(1 - dmin + fog) -- the legacy law's value at ABSOLUTE density 1.0
        # -- and multiplying by it made the rendered amplitude equal the stored
        # rms at absolute 1.0. Two Kodak sheets in this corpus print the actual
        # convention: "Read at a NET diffuse visual density of 1.0" (5248 p1,
        # 5222 p1). For a masked colour negative those are not the same place --
        # green dmin ~0.58, blue ~0.84, so absolute 1.0 is net 0.42 and net 0.16,
        # deep shadow rather than a midtone. `grain_sigma` now normalises at
        # net 1.0 and the caller multiplies by rms alone, so the renderer
        # reproduces the stored figure where the manufacturer measured it.
        #
        # Measured cost of that correction: a uniform 4-8 % drop (1/sqrt(1+fog))
        # on every stock without a measured shape, identical in all three
        # channels, shape untouched. The two SVEMA stocks whose rms was fitted by
        # rendering carry a compensating factor in their stored values, so they
        # render exactly as before -- the exemption lives in the DATA where it can
        # be read, not in this code path.
        #
        # Why the shape applies to eleven stocks only: `_grain_v2` fills the
        # anchors heuristically for 137 profiles and BOTH branches of that
        # heuristic are known wrong in sign. `sigma_shape_measured` is what keeps
        # them out; see the GrainSpec docstring.
        # ------------------------------------------------------------------
        #  THE COUNT GATE (R-S6, spec ch. 12), applied to the field's MARGINAL.
        #
        #  A Gaussian field has zero skewness at every density. Real film does
        #  not: where the developed grains are countable the marginal is
        #  compound Poisson and positively skewed, and on this corpus that
        #  region starts at net D 0.058 for the median stock and 0.87 for the
        #  coarsest -- the shadows of a negative, and exactly where real film
        #  shows discrete salt-like grain instead of smooth noise.
        #
        #  ⚠ THE GATE IS COMPUTED, NEVER CHOSEN. N_elem comes from the physical
        #  diameter through Nutting's relation and the noise-equivalent element
        #  area; nothing in it reads a stock name or a taste setting.
        #
        #  ⚠⚠ AND THE FIELD IS TRANSFORMED RATHER THAN REPLACED BY A DOT
        #  RENDERER, WHICH IS A DEPARTURE FROM SPEC §12.4.2 AND IS COSTED. The
        #  specification places every grain: at 4K that is about 5 million
        #  grains per channel over 5 % of the frame at the gate's lower edge and
        #  four times that at its upper, i.e. 25-100 ms of scatter against an
        #  8 ms whole-frame budget. The transform reproduces the spec's own
        #  blend in mean, variance and skewness exactly, at O(1) per pixel; see
        #  `fp.grain_marginal_coeff` for what it does not reproduce.
        # ------------------------------------------------------------------
        elem_um2 = fp.grain_element_area_um2(fp.scan_sigma_mm(scan_f50),
                                             1.0 / px_per_mm)
        for c in range(3):
            dmin = curves[c].dmin
            # Poisson statistics of discrete developed crystals: sigma grows
            # as sqrt(density). The fog term keeps grain alive in deep shadow;
            # perfectly clean blacks are one of the loudest digital tells.
            amp = fp.grain_sigma(
                gs, dmin, curves[c].dmax, dens[:, :, c]).astype(np.float32)

            field = fields[c]
            if settings.marginal_model == "hybrid" and elem_um2 > 0.0:
                net = np.maximum(dens[:, :, c] - np.float32(dmin), 0.0)
                n_elem = fp.grain_n_elem(net, grain_um[c], gs.size_sigma_log,
                                         elem_um2)
                coeff = fp.grain_marginal_coeff(n_elem, gs.size_sigma_log)
                if float(np.max(coeff)) > 0.0:
                    # Standardise by the field's own RMS, transform, restore.
                    # The RMS is a property of this plane and this transfer, so
                    # both engines measure it the same way -- one reduction,
                    # the same one `AlgoPlaneMean` already performs.
                    rms = float(np.sqrt(np.mean(
                        np.square(field.astype(np.float64)))))
                    if rms > 0.0:
                        z = field.astype(np.float64) / rms
                        z = ((z + coeff * (z * z - 1.0))
                             / np.sqrt(1.0 + 2.0 * coeff * coeff))
                        field = (z * rms).astype(np.float32)

            dens[:, :, c] += (
                np.float32(settings.grain_scale * temporal_scale)
                * field * amp
            )
        del fields

    np.maximum(dens, np.float32(0.0), out=dens)

    # -- 12. dye impurity / scanner crosstalk -------------------------------
    m = np.asarray(profile.dye_matrix, dtype=np.float32)
    if not np.allclose(m, np.eye(3, dtype=np.float32)):
        dens = dens.reshape(-1, 3) @ m.T
        dens = dens.reshape(h, w, 3).astype(np.float32)

    np.maximum(dens, np.float32(0.0), out=dens)

    # -- 12b. Callier: the density the READER's optics see (C22) --------------
    # Placed here, at the boundary between the developed negative and everything
    # that reads it, because both readers in this chain are affected: an optical
    # printer with a condenser and a scanner with a directed source see the same
    # steepened density. Before stage 13 rather than after, since the print
    # stage's own curve must act on what its optics actually see.
    callier_density(dens, curves, profile.callier_q,
                    settings.scanner_specular, profile.is_monochrome)

    # -- 13. duplication generations, then print -----------------------------
    if reversal:
        # The slide already is the positive. Its own dmin/dmax become the white
        # and black points; no second curve, no inversion.
        out = dens
        final_curves = curves
    else:
        d_mid = neutral_mid_density(profile, settings.coupler_scale)
        # ⚠ AND SO MUST THE PRINT CHAIN'S OWN MID-GREY REFERENCE (C22). This is a
        # SECOND computation of the neutral negative density, used by the dupe
        # generations and the final print, and it has to see the reader's optics
        # for the same reason the anchor solve does. Missing it here is what left
        # mid grey +54/255 out on DOUBLE-X while the anchor solve was already
        # correct -- the two references disagreed, and the print re-timed against
        # the wrong one.
        if not callier_is_inert(profile, settings.scanner_specular):
            _cq = float(profile.callier_q)
            _cs = float(settings.scanner_specular)
            d_mid = [curves[c].dmin
                     + float(callier_net(d_mid[c] - curves[c].dmin, _cq, _cs))
                     for c in range(3)]

        # Nobody ever projected the camera negative. A release print is three or
        # four generations away from it: negative -> interpositive -> dupe
        # negative -> print. Each intermediate adds its own grain and its own
        # MTF loss, and that accumulation is a large part of why archival
        # footage looks the way it does -- far more than the emulsion alone.
        #
        # Stages come in pairs so the polarity always returns to negative before
        # the final print. Duplicating stock runs at gamma 1.0 by design, so
        # contrast does not compound over the chain; grain and softness do.
        stages = 2 * max(0, settings.generations)
        if stages:
            dupe = get_print_stock(print_stock_key(settings.dupe_stock))
            dcurves = dupe.curves.as_tuple()
            dupe_mtf = grid.mtf(dupe.mtf_f50, 0.0, 0.0)
            for _ in range(stages):
                # Printing optics blur what comes IN -- the accumulated image
                # and all grain from earlier generations. This has to happen
                # before the new stock records anything.
                for c in range(3):
                    dens[:, :, c] = apply_transfer(dens[:, :, c], dupe_mtf)

                offs, d_mid = solve_intermediate_offsets(d_mid, dcurves)
                nxt = np.empty_like(dens)
                for c in range(3):
                    nxt[:, :, c] = density(
                        (np.float32(offs[c]) - dens[:, :, c]).astype(np.float32),
                        dcurves[c],
                    )
                dens = nxt

                # This stage's own grain is created in THIS emulsion, so it is
                # not blurred by this stage's optics -- only by later ones.
                # Adding it before the blur (the obvious way round) quietly
                # softens every generation's grain by its own MTF and makes a
                # dupe chain come out cleaner than the original.
                if settings.grain_scale > 0.0 and dupe.grain_rms > 0.0:
                    # ⚠ THE DUPE AND PRINT EMULSIONS STAY ON THE LEGACY
                    # SPECTRUM WHATEVER `spectrum_model` SAYS, AND THAT IS NOT
                    # AN OVERSIGHT. `boolean_jinc` needs a physical grain
                    # diameter, and the only route to one in this corpus is the
                    # random-dot inversion of a PUBLISHED rms granularity. A
                    # duplicating stock's `grain_rms` is not that: it is a
                    # look, fitted, with no datasheet behind it, so inverting
                    # it would manufacture a diameter and present it as
                    # physics. They also carry no anisotropy -- the camera
                    # negative's coating flow is not theirs -- which is why the
                    # C++ twins pass ALGO_GRAIN_ANISOTROPY_NONE here.
                    gfield = make_grain_field(
                        h, w, px_per_mm,
                        fp.grain_gauss_terms(
                            "legacy_gaussian",
                            clump_um=dupe.grain_clump_um, clump_gain=0.30),
                        dupe.grain_rms,
                        fp.scan_sigma_mm(scan_f50) * px_per_mm,
                        1.0, int(settings.seed) & 0xFFFFFFFF,
                        int(settings.frame_index), RngStage.DUPE_GRAIN,
                        fp.grain_aperture_legacy,
                    )
                    for c in range(3):
                        amp = np.sqrt(
                            np.maximum(
                                dens[:, :, c] - np.float32(dcurves[c].dmin), 0.0
                            )
                            + 0.15
                        ).astype(np.float32)
                        dens[:, :, c] += (
                            np.float32(settings.grain_scale) * gfield * amp
                        )
                    del gfield
                np.maximum(dens, np.float32(0.0), out=dens)
            del dupe_mtf

        # logE_print = offset - D, with the offset solved so 18% scene grey lands
        # on the requested display value -- the printer-light setting. Higher
        # scene exposure raises negative density, which lowers print exposure and
        # print density, which brightens the positive. That double inversion is
        # what gives correct rolloff at both ends for free.
        pcurves = print_stock.curves.as_tuple()
        targets = [
            settings.grey_target / _tint_factor(profile, c) for c in range(3)
        ]

        # ---- THE EXPOSURE SIDE OF THE PRINT (schema v25, 2026-09-03) -------
        # ⚠ WHAT COMES OUT OF THE NEGATIVE IS NOT WHAT THE PRINT SEES. `dens`
        # holds STATUS densities -- what a densitometer reads. The print
        # emulsion reads the negative through the printer lamp, the filter
        # pack and its own three sensitisations, and that product puts the
        # negative's magenta dye into the print's RED-sensitive layer and its
        # yellow dye into the GREEN-sensitive one. `offset - D` per channel
        # cannot express any of that; a 3x3 on the way IN can, and Hanson &
        # Kisner 1953 name it: "effective integral printing density".
        #
        # ⚠ THE ANCHOR MUST GO THROUGH THE SAME OPERATOR, and this is the
        # whole reason it is applied here rather than folded in later. The
        # printer-light solve centres a NEUTRAL; if the neutral reference and
        # the image were transformed differently the print would be timed
        # against a grey that no pixel in the frame corresponds to. Rows sum
        # to 1.0 by construction, so on a neutral this changes nothing at all
        # and the offsets come out identical -- which is exactly the property
        # that lets it be switched on without re-timing anything.
        pdm = np.asarray(print_stock.printing_density_matrix, dtype=np.float64)
        if not np.allclose(pdm, np.eye(3)):
            d_mid = list(pdm @ np.asarray(d_mid, dtype=np.float64))
            dens = np.ascontiguousarray(
                (dens.reshape(-1, 3) @ pdm.T.astype(np.float32)
                 ).reshape(h, w, 3), dtype=np.float32)

        offsets = solve_stage_offsets(
            d_mid, pcurves, print_stock.dye_matrix, targets,
            settings.black_point_stretch
        )
        out = np.empty((h, w, 3), dtype=np.float32)
        for c in range(3):
            log_e_print = (np.float32(offsets[c]) - dens[:, :, c]).astype(np.float32)
            out[:, :, c] = density(log_e_print, pcurves[c])
        del dens
        final_curves = pcurves

        pm = np.asarray(print_stock.dye_matrix, dtype=np.float32)
        if not np.allclose(pm, np.eye(3, dtype=np.float32)):
            out = np.ascontiguousarray(
                (out.reshape(-1, 3) @ pm.T).reshape(h, w, 3), dtype=np.float32
            )

    # -- 14. print grain, then transmittance to display linear ---------------
    if not reversal and settings.print_grain and print_stock.grain_rms > 0.0:
        # Print stock grain is finer than negative grain and largely achromatic,
        # so one field serves all three channels. It matters because it is
        # applied *after* the print curve, so unlike negative grain it does not
        # get compressed by the shoulder -- a subtle difference in how grain
        # behaves in highlights that single-stage models cannot produce.
        pfield = make_grain_field(
            h, w, px_per_mm,
            fp.grain_gauss_terms("legacy_gaussian",
                                 clump_um=print_stock.grain_clump_um,
                                 clump_gain=0.25),
            print_stock.grain_rms,
            fp.scan_sigma_mm(scan_f50) * px_per_mm,
            1.0, int(settings.seed) & 0xFFFFFFFF,
            int(settings.frame_index), RngStage.PRINT_GRAIN,
            fp.grain_aperture_legacy,
        )
        for c in range(3):
            amp = np.sqrt(
                np.maximum(out[:, :, c] - np.float32(final_curves[c].dmin), 0.0)
                + 0.15
            ).astype(np.float32)
            out[:, :, c] += pfield * amp
        del pfield

    np.maximum(out, np.float32(0.0), out=out)

    for c in range(3):
        fc = final_curves[c]
        t_max = 10.0 ** (-fc.dmin)   # clear film: the brightest it can be
        # ⚠ THE BLACK POINT IS NOW A CONTROL, not a constant. `t_min` used to be
        # 10^-Dmax unconditionally, which stretched every stock's Dmax down to
        # output zero and destroyed everything at or beyond it. Full rationale,
        # the measured before/after and the one measured cost are in
        # `RenderSettings.black_point_stretch`. At the default 1.0 this is
        # arithmetically the old expression.
        t_min = settings.black_point_stretch * 10.0 ** (-fc.dmax)
        # ⚠ AND DENSITY IS CAPPED AT Dmax FIRST. A layer cannot be denser than
        # its own maximum dye load, yet stages 9 and 12 are unbounded additions
        # in the density domain and do exceed it -- measured, blue on Velvia
        # reaches 4.2439 against a Dmax of 3.3072. At s = 1 the cap is
        # OUTPUT-NEUTRAL (both sides map to zero, verified byte-identical); at
        # s < 1 it is what stops an over-dense pixel encoding to code 0 anyway.
        np.minimum(out[:, :, c], np.float32(fc.dmax), out=out[:, :, c])
        trans = np.power(np.float32(10.0), -out[:, :, c], dtype=np.float32)
        out[:, :, c] = ((trans - t_min) / (t_max - t_min)).astype(np.float32)

    # -- 14b. reseau reconstruction ------------------------------------------
    # Projection sends light back through the same filter grid in register, and
    # only here does the single monochrome record become colour again. Doing it
    # at the very end is not a shortcut: on a real additive print the grid sits
    # in the light path at viewing time, downstream of everything.
    if reseau_mask is not None:
        out = reseau_reconstruct(
            out[:, :, 1], reseau_mask, grid, reseau_pitch_px, profile.reseau
        )

    # Residual printer-light mismatch from the film base colour. A real printer
    # neutralises the orange mask, so only a small residual survives.
    tint = profile.base_tint
    for c in range(3):
        if tint[c] != 1.0:
            out[:, :, c] *= np.float32(1.0 + (tint[c] - 1.0) * 0.5)

    # -- 14c. silver image tone (monochrome only) ----------------------------
    # Developed silver is not spectrally neutral. Fine particles scatter short
    # wavelengths and read warm; coarse filamentary silver reads neutral to
    # blue. The effect is strongest where there is least silver -- the light
    # tones -- and fades as density builds, so it is weighted by the output
    # level rather than applied flat.
    #
    # This runs after the printer-light anchor solve on purpose. base_tint is
    # *compensated* by that solve, which is why it cannot tint a B&W stock at
    # all; this stage is downstream of it and therefore survives.
    if profile.is_monochrome and profile.silver_tone != 0.0:
        tone = np.float32(profile.silver_tone)
        w = out[:, :, 1]                      # bright = least silver = warmest
        out[:, :, 0] *= (1.0 + np.float32(0.28) * tone * w)
        out[:, :, 2] *= (1.0 - np.float32(0.22) * tone * w)

    return np.clip(out, 0.0, 1.0, out=out)


# ===========================================================================
# I/O
# ===========================================================================
def load_linear(path: Path, max_dim: int = 0) -> np.ndarray:
    """Load an image and decode it to linear light.

    Note: an ordinary JPEG or PNG is display referred, so its highlights have
    already been clipped by the camera. Feeding real scene-referred data (EXR,
    or a raw file developed to linear) gives markedly better results, because
    the film's shoulder then has real highlight information to roll off.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max_dim and max(im.size) > max_dim:
            scale = max_dim / max(im.size)
            new = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            im = im.resize(new, resample=Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.uint8)
    return srgb_to_linear(arr.astype(np.float32) / 255.0)


def save_linear(path: Path, linear: np.ndarray, bit_depth: int, rng,
                alpha: bool = False) -> None:
    """Encode linear light to sRGB, dither, quantise and write a PNG."""
    enc = linear_to_srgb(linear)
    peak = float((1 << bit_depth) - 1)
    # Triangular-PDF dither at one LSB removes quantisation banding in the
    # halation bloom and in the shadow rolloff without adding visible noise.
    dither = (rng.random(enc.shape, dtype=np.float32) - rng.random(
        enc.shape, dtype=np.float32
    )) / peak
    q = np.clip((enc + dither) * peak + 0.5, 0.0, peak).astype(
        np.uint16 if bit_depth == 16 else np.uint8
    )
    write_png(path, q, bit_depth, alpha)


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="film_sim",
        description="Physically-modelled film stock simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", type=Path, nargs="?", help="input image")
    p.add_argument(
        "-p",
        "--profile",
        default="all",
        help="stock name, alias or catalogue number (e.g. 5219, velvia), or 'all'",
    )
    p.add_argument("-o", "--outdir", type=Path, default=Path("film_renders"))
    p.add_argument(
        "-f", "--format", dest="film_format", default=None,
        choices=sorted(FORMATS),
        help="override the gauge. Default: each stock's own native gauge "
             "(8 mm stocks render as 8 mm, 35 mm stills as 36 mm, and so on)"
    )
    p.add_argument(
        "--print-stock",
        default="",
        help=(
            "SCAN_DI, KODAK_2383_RELEASE or TECHNICOLOR_IB; "
            "empty uses the stock's own default. Ignored for reversal stocks"
        ),
    )
    p.add_argument("-e", "--exposure", type=float, default=0.0, help="stops")
    p.add_argument("--exposure-time", type=float, default=0.0,
                   dest="exposure_time_s",
                   help="shutter open time in SECONDS; 0 = not stated, which "
                        "leaves the reciprocity stage inert. Corrections exist "
                        "only beyond the stock's onset (typically 1 s) and, on "
                        "the one stock that measures it, below 1e-4 s")
    p.add_argument("--scene-kelvin", type=float, default=5500.0)
    p.add_argument(
        "--wb-strength",
        type=float,
        default=0.0,
        help=(
            "how much colour-temperature mismatch to apply; 0 assumes the "
            "correct on-camera filter was used, 1.0 shows the full cast"
        ),
    )
    p.add_argument(
        "--grey-target",
        type=float,
        default=0.18,
        help="display linear value that 18%% scene grey is printed to",
    )
    p.add_argument(
        "--black-point-stretch", type=float, default=1.0,
        dest="black_point_stretch",
        help="how much of the stock's own Dmax is stretched to output zero. "
             "1 = shipped behaviour (Dmax -> 0, full output range, shadows at "
             "or past Dmax are crushed); 0 = Dmax renders at its own relative "
             "transmittance, nothing clips at the bottom and Dmax becomes "
             "visible. See RenderSettings.black_point_stretch",
    )
    p.add_argument("--grain", type=float, default=1.0, dest="grain_scale")
    p.add_argument("--halation", type=float, default=1.0, dest="halation_scale")
    p.add_argument("--scanner-specular", type=float, default=0.853,
                   dest="scanner_specular",
                   help="reader optics: 0 = diffuse integrating sphere, which "
                        "reproduces the stored (diffuse) characteristic curves "
                        "exactly; 1 = condenser/point source, applying the "
                        "stock's full Callier coefficient. DEFAULT 0.853 = "
                        "1 - E with Streiffert's fitted E = 0.1471 (owner "
                        "decision 2026-09-06; a densitometer geometry adopted "
                        "as a provisional stand-in, NOT a scanner measurement). "
                        "Monochrome stocks only -- colour carries Q = 1.0")
    p.add_argument("--couplers", type=float, default=1.0, dest="coupler_scale")
    p.add_argument("--scanner-f50", type=float, default=0.0, help="cycles/mm, 0=auto")
    p.add_argument(
        "--misreg-scale",
        type=float,
        default=1.0,
        help="multiplies the stock's own channel registration error",
    )
    p.add_argument("--no-print-grain", action="store_true")
    p.add_argument(
        "--flare",
        type=float,
        default=-1.0,
        help=(
            "veiling flare fraction of the taking lens; -1 uses the stock's own "
            "era-appropriate default (0.06-0.14 for pre-1940 uncoated glass, "
            "0 for modern coated lenses)"
        ),
    )
    p.add_argument(
        "-g",
        "--generations",
        type=int,
        default=0,
        help=(
            "intermediate duplication rounds between negative and print; each "
            "adds an interpositive and a dupe negative. 0 = print straight from "
            "the camera negative, 1 = a normal release print, 2-3 = an archival "
            "reissue"
        ),
    )
    p.add_argument("--dupe-stock", default="DUPE_FINE_GRAIN")
    p.add_argument(
        "--no-reseau",
        action="store_true",
        help="disable the additive colour grid on mosaic stocks (Dufaycolor)",
    )
    p.add_argument("--bits", type=int, default=16, choices=(8, 16))
    # ⚠ -8bpp IS NOT A SYNONYM FOR `--bits 8`, AND THE DIFFERENCE IS THE ALPHA.
    # `--bits 8` writes an 8-bit THREE-channel PNG. This writes an 8-bit FOUR-
    # channel one with a constant opaque alpha, because that is what the
    # comparison tools want: a 16-bit render cannot be diffed against an 8-bit
    # source without a requantisation step that invents differences of its own,
    # and several viewers refuse a three-channel file outright.
    #
    # It OVERRIDES --bits rather than conflicting with it, so `-8bpp --bits 16`
    # is 8-bit and not an error -- the flag is the more specific request and
    # silently honouring the more specific one is friendlier than refusing the
    # combination.
    p.add_argument(
        "-8bpp", "--8bpp", dest="eight_bpp", action="store_true",
        help="write 8-bit RGBA PNG with a constant opaque alpha, overriding "
             "--bits; use this when the output must be compared against an "
             "8-bit input",
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-dim", type=int, default=0, help="downscale input, 0=off")
    p.add_argument("--emit-cpp", action="store_true", help="also write C++ tables")
    p.add_argument("--list", action="store_true", help="list stocks and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_all()

    if args.list:
        for prof in FILM_PROFILES:
            kind = "reversal" if prof.is_reversal else "negative"
            mono = " B&W" if prof.is_monochrome else ""
            print(
                f"{prof.name:32s} EI{prof.exposure_index:<5d} "
                f"{prof.balance_kelvin}K  {kind}{mono}"
            )
            if prof.aliases:
                print(f"{'':32s}   aliases: {', '.join(prof.aliases)}")
        return 0

    if args.image is None:
        print("[ERROR] no input image given (use --list to see stocks)", file=sys.stderr)
        return 2
    if not args.image.is_file():
        print(f"[ERROR] not a file: {args.image}", file=sys.stderr)
        return 2

    if args.profile.lower() == "all":
        stocks = list(FILM_PROFILES)
    else:
        try:
            stocks = [get_profile(args.profile)]
        except KeyError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2

    # ⚠ THE OUTPUT ENCODING IS DECIDED ONCE, HERE, AND NOWHERE ELSE.
    # Until 2026-09-11 the two `save_linear` call sites each read `args.bits`
    # directly while `RenderSettings.bit_depth` was set from the same value a
    # few lines below -- three readings of one decision, which is exactly how a
    # new flag gets honoured in one place and silently ignored in the other
    # two. `-8bpp` overrides `--bits`; see the flag's own note in the parser.
    OUT_BITS = 8 if args.eight_bpp else args.bits
    OUT_ALPHA = bool(args.eight_bpp)

    settings = RenderSettings(
        film_format=args.film_format or "super35",
        print_stock=args.print_stock,
        exposure_stops=args.exposure,
        exposure_time_s=args.exposure_time_s,
        scene_kelvin=args.scene_kelvin,
        wb_strength=args.wb_strength,
        grey_target=args.grey_target,
        black_point_stretch=args.black_point_stretch,
        grain_scale=args.grain_scale,
        halation_scale=args.halation_scale,
        scanner_specular=args.scanner_specular,
        coupler_scale=args.coupler_scale,
        scanner_f50=args.scanner_f50,
        misreg_scale=args.misreg_scale,
        print_grain=not args.no_print_grain,
        flare=args.flare,
        generations=args.generations,
        dupe_stock=args.dupe_stock,
        reseau=not args.no_reseau,
        # -8bpp wins over --bits; see the flag's own note.
        bit_depth=OUT_BITS,
        seed=args.seed,
        max_dim=args.max_dim,
    )

    linear = load_linear(args.image, args.max_dim)
    h, w = linear.shape[:2]
    if args.film_format is not None:
        print(f"[INFO] {args.image.name}  {w}x{h}  gauge overridden to "
              f"{args.film_format} ({FORMATS[args.film_format]:.2f} mm) for every stock")
    else:
        print(f"[INFO] {args.image.name}  {w}x{h}  "
              f"each stock rendered at its own native gauge")

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    out_rng = np.random.default_rng(args.seed ^ 0x5EED)

    for stock in stocks:
        # Each stock renders at its own gauge unless the caller overrode it.
        # This is what makes an 8 mm profile actually look like 8 mm: every
        # spatial number in the database is physical (um, cycles/mm), so the
        # gauge is the only thing that turns it into pixels.
        fmt = args.film_format or stock.default_format
        settings = dataclasses.replace(settings, film_format=fmt)
        chain = "reversal (no print)" if stock.is_reversal else (
            settings.print_stock or stock.default_print
        )
        extra = []
        if settings.generations:
            extra.append(f"{settings.generations} dupe gen")
        fl = settings.flare_for(stock)
        if fl > 0.0:
            extra.append(f"flare {fl:.0%}")
        if stock.has_reseau and settings.reseau:
            extra.append("reseau")
        note = ("  " + ", ".join(extra)) if extra else ""
        ppmm = linear.shape[1] / FORMATS[fmt]
        print(f"  -> {stock.name:32s} [{chain}]  {fmt} "
              f"{ppmm:.0f}px/mm{note}", flush=True)
        result = simulate(linear, stock, settings)
        dest = args.outdir / f"{stem}_{stock.name}.png"
        save_linear(dest, result, OUT_BITS, out_rng, OUT_ALPHA)

    # A print stock is not something you can expose in a camera, so it has no
    # profile of its own and `-p all` used to skip it entirely -- which is why
    # TASMA_POSITIVE_28 never appeared. Render each one through a reference
    # negative instead, so every entry in the database produces an image.
    n_prints = 0
    if args.profile.lower() == "all" and not args.print_stock:
        for ps in PRINT_STOCKS:
            mono_print = ps.curves.r == ps.curves.g == ps.curves.b
            if ps.name == "TECHNICOLOR_IB":
                ref = "TECHNICOLOR_THREE_STRIP"
            elif mono_print:
                ref = "EASTMAN_PLUS_X_5231"
            else:
                ref = "KODAK_PORTRA_400"
            neg = get_profile(ref)
            st = dataclasses.replace(
                settings, film_format=neg.default_format, print_stock=ps.name)
            print(f"  -> PRINT {ps.name:26s} [on {ref}]", flush=True)
            res = simulate(linear, neg, st)
            save_linear(args.outdir / f"{stem}_PRINT_{ps.name}.png",
                        res, OUT_BITS, out_rng, OUT_ALPHA)
            n_prints += 1

    if args.emit_cpp:
        import cpp_codegen

        cpp_codegen.generate(args.outdir)

    print(f"[INFO] wrote {len(stocks) + n_prints} render(s) "
          f"({len(stocks)} stocks + {n_prints} print stocks) to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
