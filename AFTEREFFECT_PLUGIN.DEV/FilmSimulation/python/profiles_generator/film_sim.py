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


def _normalised_transmittance(d: float, c: ToneCurve) -> float:
    """Density to display-normalised transmittance for one curve."""
    t_max = 10.0 ** (-c.dmin)   # clear film: the brightest it can be
    t_min = 10.0 ** (-c.dmax)   # Dmax: the darkest
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


def _callier_factor(profile, scanner_specular: float) -> float:
    """1 + specular*(Q-1): the multiplier a directional reader applies (C22).

    One definition, used by both the anchor solve and stage 12b, because the two
    MUST agree -- the solve exists to cancel the level change the stage causes.
    """
    q = float(profile.callier_q)
    s = float(scanner_specular)
    if s <= 0.0 or q == 1.0:
        return 1.0
    return 1.0 + s * (q - 1.0)


def solve_anchors(
    profile: FilmProfile,
    print_stock: PrintStock,
    grey_target: float,
    coupler_scale: float = 1.0,
    scanner_specular: float = 0.0,
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
    _cal = _callier_factor(profile, scanner_specular)

    def _cal_apply(d: list[float]) -> list[float]:
        if _cal == 1.0:
            return list(d)
        return [curves[k].dmin + (d[k] - curves[k].dmin) * _cal for k in range(3)]

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
                    return _normalised_transmittance(mixed, curves[c])

                target = grey_target / _tint_factor(profile, c)
                trims[c] = _bisect(fn, -8.0, 8.0, target, rising=True)
        return (trims[0], trims[1], trims[2])

    # Neutral negative density, after couplers and the negative's dye matrix.
    d_neg = _couple(_neg_density([0.0, 0.0, 0.0]))
    d_mid = _cal_apply(
        [sum(neg_m[c][k] * d_neg[k] for k in range(3)) for c in range(3)])
    targets = [grey_target / _tint_factor(profile, c) for c in range(3)]
    offsets = solve_stage_offsets(
        d_mid, print_stock.curves.as_tuple(), print_stock.dye_matrix, targets
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
                return _normalised_transmittance(mixed, dst_curves[c])

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


def spectral_out_of_reach(profile) -> float | None:
    """Share of the stock's sensitivity lying beyond the basis's red limit.

    Companion to :func:`spectral_peak_lambda`: the peak catches an emulsion
    whose maximum is in the infrared, this catches one whose maximum is in the
    visible but which carries a substantial infrared shoulder -- the case that
    a peak test alone passes incorrectly.
    """
    sens = layer_sensitivities(profile)
    if sens is None:
        return None
    grid = spectral_grid()
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    beyond = grid > _SPECTRAL_BASIS_LAMBDA_MAX
    total = 0.0
    out = 0.0
    for row in sens:
        total += float(trap(row, grid))
        if beyond.any():
            out += float(trap(row[beyond], grid[beyond]))
    return (out / total) if total > 0.0 else None


def spectral_peak_lambda(profile) -> float | None:
    """Wavelength of peak sensitivity, nanometres, per layer maximum.

    Returns the LONGEST per-layer peak, because it is the long-wavelength end
    that a visible-primary basis fails to reach. ``None`` when the stock carries
    no curves.
    """
    sens = layer_sensitivities(profile)
    if sens is None:
        return None
    grid = spectral_grid()
    peaks = [float(grid[int(np.argmax(row))]) for row in sens
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

    def mtf(self, f50_cpmm: float, adjacency: float, adjacency_um: float,
            spec: "fp.MTFSpec | None" = None, channel: int = 1) -> np.ndarray:
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
        if spec is not None:
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
        dens_cc = int(s[i:j]) / 100.0
        for c in _CC_ATTENUATES[s[j]]:
            out[c] += dens_cc
        i = j + 1
    return tuple(out)                  # type: ignore[return-value]


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

        D_read = dmin + (D - dmin) * (1 + specular * (Q - 1))

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
    k = np.float32(1.0 + specular * (callier_q - 1.0))
    for c in range(3):
        dmin = np.float32(curves[c].dmin)
        # (D - dmin) * k + dmin, written so a density below dmin (which grain
        # can produce in the base) scales the same way rather than being clamped
        # into a different branch.
        dens[:, :, c] = (dens[:, :, c] - dmin) * k + dmin
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
    delta = None
    for _ in range(int(iie.iterations)):
        delta = [dens[:, :, j] - np.float32(d_ref[j]) for j in range(3)]
        if dw > 0.0:
            for j in range(3):
                ref = max(d_ref[j], 1e-4)
                wj = (1.0 - dw) + dw * (dens[:, :, j] / np.float32(ref))
                delta[j] = delta[j] * wj.astype(np.float32)
        for c in range(3):
            adj = np.zeros((h, w), dtype=np.float32)
            for j in range(3):
                if j == c or m[c][j] == 0.0:
                    continue
                adj += np.float32(m[c][j]) * delta[j]
            if reversal:
                dens[:, :, c] = density(
                    -(log_e[:, :, c] + np.float32(anchors[c])) - adj,
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


def make_grain_field(
    grid: FreqGrid,
    rng: np.random.Generator,
    clump_um: float,
    clump_gain: float,
    rms_granularity: float,
    band_limit: np.ndarray | None = None,
) -> np.ndarray:
    """Spectrally-shaped, granularity-calibrated grain field.

    Returns a zero-mean density-domain field whose amplitude is fixed by the
    stock's RMS granularity, band-limited by ``band_limit`` (the scanner MTF,
    acting as the pre-sampling anti-alias filter that real scanner optics are).

    The original script instead resized small uniform-noise grids with bilinear
    interpolation, which gives an anisotropic triangular spectrum and visible
    axis-aligned diamond artefacts.

    Args:
        band_limit: Real-valued half-spectrum transfer applied before sampling.
            Must not include a phase term such as misregistration -- shifting an
            independent random field is a no-op statistically, and a complex
            transfer here would make the field complex.
    """
    shape_t = grid.grain_shape(clump_um, clump_gain)
    if band_limit is not None:
        shape_t = (shape_t * band_limit).astype(np.float32)

    # Unit-variance white noise on this grid has a flat spectrum whose discrete
    # mean relates to the continuous integral by 1/px_per_mm**2, hence the
    # factor below.
    scale = (
        (rms_granularity / 1000.0)
        * grid.px_per_mm
        / math.sqrt(grain_reference_energy(clump_um, clump_gain))
    )

    white = rng.standard_normal((grid.h, grid.w), dtype=np.float32)
    field = apply_transfer(white, shape_t)
    return (field * np.float32(scale)).astype(np.float32)


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


def write_png(path: Path, rgb: np.ndarray, bit_depth: int = 16) -> None:
    """Write an RGB PNG at 8 or 16 bits per channel.

    Pillow cannot write 16-bit RGB PNG, and 8 bits visibly bands in the smooth
    halation bloom and in deep shadow. Rather than pull in a dependency, this
    emits the file directly: signature, IHDR, one zlib-compressed IDAT with
    filter type 0 per scanline, IEND.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an (h, w, 3) array")
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")

    h, w = rgb.shape[:2]
    if bit_depth == 16:
        payload = rgb.astype(">u2").tobytes()
        stride = w * 3 * 2
    else:
        payload = rgb.astype(np.uint8).tobytes()
        stride = w * 3

    # Prepend the per-scanline filter byte (0 = None) without a Python loop.
    raw = np.zeros((h, stride + 1), dtype=np.uint8)
    raw[:, 1:] = np.frombuffer(payload, dtype=np.uint8).reshape(h, stride)

    ihdr = struct.pack(">IIBBBBB", w, h, bit_depth, 2, 0, 0, 0)
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

    film_format: str = "super35"
    print_stock: str = ""           # "" = use the stock's own default
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
    scene_kelvin: float = 5500.0
    wb_strength: float = 0.0
    grey_target: float = 0.18       # display linear value for 18% scene grey
    grain_scale: float = 1.0
    halation_scale: float = 1.0
    # -- C22, 2026-08-23: how DIRECTIONAL the reader's optics are. 0 = a diffuse
    # -- integrating sphere (and the default, which reproduces every earlier
    # -- render exactly); 1 = a condenser or point source, which sees the film's
    # -- full Callier coefficient. Anything between mixes the two.
    # -- ⚠ IT DOES NOTHING ON COLOUR STOCK BY CONSTRUCTION -- Callier is silver
    # -- scattering and a dye image has essentially none, so all 93 colour
    # -- profiles carry Q = 1.0. It moves the 66 monochrome stocks only.
    scanner_specular: float = 0.0
    coupler_scale: float = 1.0
    scanner_f50: float = 0.0        # 0 = take from print stock

    # -- measured spectral sensitivity (see the SPECTRAL block above) --------
    # Consumes SpectralSensitivity.log_s_* where the stock carries it. Each
    # flag substitutes a DERIVED quantity for an authored proxy, and each falls
    # back silently to the proxy for the 89 of 142 stocks that have no curves.
    #
    # spectral_balance: ON. Replaces the three assumed peak wavelengths of
    #   balance_gains() with the full measured sensitisation. Safe by
    #   construction -- it is a ratio of the same integral under two
    #   illuminants, normalised to green, so it cannot change overall exposure
    #   and cannot double-count anything downstream.
    #
    # spectral_mono: OFF, and the reason is a measured failure, not caution.
    #   The derivation projects the pan curve onto three primary lobes, and a
    #   stock sensitised outside those lobes therefore derives to nonsense:
    #   KONICA_INFRARED_750 (peak 750 nm) comes out BLUE-dominant at
    #   (0.161, 0.193, 0.646) against an authored, correct, red-dominant
    #   (0.55, 0.15, 0.30). spectral_monochrome_weights() now refuses any
    #   stock whose peak sensitisation lies beyond the basis's reach,
    #   so enabling this flag is safe -- IR and extended-red stocks fall back
    #   automatically. The guard is the peak sensitisation wavelength against
    #   the basis's long-wavelength limit, which is an unambiguous physical
    #   criterion rather than a tuned overlap threshold. It stays OFF by default because for the stocks that DO
    #   pass the guard the derived triple still depends on the assumed primary
    #   lobe width, which is an assumption and not a measurement, and because
    #   an independent analysis on 2026-08-03 reached the same conclusion about
    #   the same construction. The honest fix is a scene spectral model
    #   (reflectance basis functions under a stated illuminant, Smits or
    #   Jakob-Hanika class), built deliberately -- not a reprojection of data
    #   the database already holds.
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
    spectral_mono: bool = False
    spectral_taking: bool = False
    misreg_scale: float = 1.0       # multiplies the stock's own registration error
    print_grain: bool = True
    flare: float = -1.0             # <0 = use the stock's default_flare
    generations: int = 0            # intermediate interpositive/dupe-negative pairs
    dupe_stock: str = "DUPE_FINE_GRAIN"
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
    #: Frame number within the clip. Only the coating field uses it, to slide
    #: its machine-direction structure by one frame pitch per frame. Any frame
    #: can be rendered independently and out of order -- the field is a pure
    #: function of (seed, absolute web position).
    frame_index: int = 0

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
    h, w = linear_rgb.shape[:2]
    negative_width_mm = FORMATS[settings.film_format]
    px_per_mm = w / negative_width_mm
    rng = np.random.default_rng(settings.seed)
    print_stock = get_print_stock(settings.print_stock or profile.default_print)

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

    grid = FreqGrid(h, w, px_per_mm, profile.grain.anisotropy)

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
            frame_pitch_mm(settings.film_format), settings.seed,
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
                     spec=profile.mtf, channel=c)
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
        settings.scanner_specular,
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
        if profile.is_monochrome or reseau_mask is not None:
            # One silver image means one grain field, identical in all three
            # channels -- not three independent ones. This covers the additive
            # colour stocks too: a reseau stock has a single panchromatic
            # emulsion behind the filter grid, so it cannot have per-layer grain.
            field = make_grain_field(
                grid, rng, clumps[1], gs.clump_gain, gs.rms_granularity, scan_t
            )
            fields = (field, field, field)
        else:
            # Per-channel RMS: rms_rgb() falls back to the scalar where the
            # profile sets no override. This is where a tripack's blue layer
            # gets its 1.3x noise (topmost, fastest emulsion) and where
            # Technicolor's three physically different B&W records diverge.
            # (The schema always promised this; the renderer used the scalar
            # for all three channels until 2026-08-01 -- silent bug.)
            rms_c = gs.rms_rgb()
            fields = tuple(
                make_grain_field(
                    grid, rng, clumps[c], gs.clump_gain, rms_c[c], scan_t
                )
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
        for c in range(3):
            dmin = curves[c].dmin
            # Poisson statistics of discrete developed crystals: sigma grows
            # as sqrt(density). The fog term keeps grain alive in deep shadow;
            # perfectly clean blacks are one of the loudest digital tells.
            amp = fp.grain_sigma(
                gs, dmin, curves[c].dmax, dens[:, :, c]).astype(np.float32)
            dens[:, :, c] += (
                np.float32(settings.grain_scale) * fields[c] * amp
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
        _calf = _callier_factor(profile, settings.scanner_specular)
        if _calf != 1.0:
            d_mid = [curves[c].dmin + (d_mid[c] - curves[c].dmin) * _calf
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
            dupe = get_print_stock(settings.dupe_stock)
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
                    gfield = make_grain_field(
                        grid, rng, dupe.grain_clump_um, 0.30, dupe.grain_rms, scan_t
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
        offsets = solve_stage_offsets(
            d_mid, pcurves, print_stock.dye_matrix, targets
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
            grid, rng, print_stock.grain_clump_um, 0.25, print_stock.grain_rms, scan_t
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
        t_min = 10.0 ** (-fc.dmax)   # Dmax: the darkest
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


def save_linear(path: Path, linear: np.ndarray, bit_depth: int, rng) -> None:
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
    write_png(path, q, bit_depth)


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
    p.add_argument("--grain", type=float, default=1.0, dest="grain_scale")
    p.add_argument("--halation", type=float, default=1.0, dest="halation_scale")
    p.add_argument("--scanner-specular", type=float, default=0.0,
                   dest="scanner_specular",
                   help="reader optics: 0 = diffuse integrating sphere "
                        "(default, and what every earlier render used), "
                        "1 = condenser/point source, which applies the stock's "
                        "full Callier coefficient. Monochrome stocks only -- "
                        "colour carries Q = 1.0")
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

    settings = RenderSettings(
        film_format=args.film_format or "super35",
        print_stock=args.print_stock,
        exposure_stops=args.exposure,
        exposure_time_s=args.exposure_time_s,
        scene_kelvin=args.scene_kelvin,
        wb_strength=args.wb_strength,
        grey_target=args.grey_target,
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
        bit_depth=args.bits,
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
        save_linear(dest, result, args.bits, out_rng)

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
                        res, args.bits, out_rng)
            n_prints += 1

    if args.emit_cpp:
        import cpp_codegen

        cpp_codegen.generate(args.outdir)

    print(f"[INFO] wrote {len(stocks) + n_prints} render(s) "
          f"({len(stocks)} stocks + {n_prints} print stocks) to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
