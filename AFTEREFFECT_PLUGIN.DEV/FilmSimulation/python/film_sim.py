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
import math
import struct
import sys
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from film_profiles import (
    FILM_PROFILES,
    FORMATS,
    IDENTITY3,
    Feature,
    FilmProfile,
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


def solve_anchors(
    profile: FilmProfile,
    print_stock: PrintStock,
    grey_target: float,
    coupler_scale: float = 1.0,
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
                    return _normalised_transmittance(mixed, curves[c])

                target = grey_target / _tint_factor(profile, c)
                trims[c] = _bisect(fn, -8.0, 8.0, target, rising=True)
        return (trims[0], trims[1], trims[2])

    # Neutral negative density, after couplers and the negative's dye matrix.
    d_neg = _couple(_neg_density([0.0, 0.0, 0.0]))
    d_mid = [sum(neg_m[c][k] * d_neg[k] for k in range(3)) for c in range(3)]
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

    def mtf(self, f50_cpmm: float, adjacency: float, adjacency_um: float) -> np.ndarray:
        """Emulsion or scanner MTF, 50% modulation at ``f50_cpmm``.

        Gaussian in form so that MTF(f50) = 0.5 exactly, optionally multiplied
        by a mild low-frequency lift representing development adjacency
        overshoot (real MTF curves often exceed 100% at low frequency).
        """
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
        """Isotropic grain power-spectrum shape (Wiener spectrum surrogate).

        Two terms: a high-frequency rolloff set by the mean developed clump
        size, and an extra low-frequency lobe whose amplitude is the clumping
        tendency. Cubic crystals cluster strongly, tabular T-grain barely at
        all, and that difference is what separates HP5's velvety look from
        VISION3's even sand.

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
    scene_kelvin: float = 5500.0
    wb_strength: float = 0.0
    grey_target: float = 0.18       # display linear value for 18% scene grey
    grain_scale: float = 1.0
    halation_scale: float = 1.0
    coupler_scale: float = 1.0
    scanner_f50: float = 0.0        # 0 = take from print stock
    misreg_scale: float = 1.0       # multiplies the stock's own registration error
    print_grain: bool = True
    flare: float = -1.0             # <0 = use the stock's default_flare
    generations: int = 0            # intermediate interpositive/dupe-negative pairs
    dupe_stock: str = "DUPE_FINE_GRAIN"
    reseau: bool = True             # allow the additive colour grid
    bit_depth: int = 16
    seed: int = 12345
    max_dim: int = 0                # 0 = no downscale

    def flare_for(self, profile: FilmProfile) -> float:
        """Veiling flare fraction to use, honouring the per-stock default."""
        return profile.default_flare if self.flare < 0.0 else self.flare


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
    if not np.allclose(take, np.eye(3, dtype=np.float32)):
        exposure = (exposure.reshape(-1, 3) @ take.T).reshape(h, w, 3)
        exposure = np.ascontiguousarray(exposure, dtype=np.float32)

    # -- 3. stock colour balance --------------------------------------------
    if settings.wb_strength > 0.0 and not profile.is_monochrome:
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
    if Feature.UNEVEN_EMULSION in profile.features:
        # Very low frequency multiplicative sensitivity drift, a few percent.
        # Poor QC stocks show this as slow mottle across the frame; it reads as
        # "old film" far more strongly than extra grain does.
        blob = apply_transfer(
            rng.standard_normal((h, w), dtype=np.float32),
            grid.gaussian(negative_width_mm * 1000.0 / 22.0),
        )
        s = float(blob.std()) or 1.0
        exposure *= (1.0 + 0.035 * (blob / s))[:, :, None].astype(np.float32)

    # -- 5. halation (in linear exposure) -----------------------------------
    hal = profile.halation
    if hal.active and settings.halation_scale > 0.0:
        scatter = grid.multi_gaussian(hal.radii_um, hal.weights)
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
            exposure[:, :, c] += np.float32(
                gains[c] * settings.halation_scale
            ) * (apply_transfer(above, scatter) - above)
        del lum

    np.maximum(exposure, np.float32(0.0), out=exposure)

    # -- 6. emulsion MTF on the exposure ------------------------------------
    # Light scatter happens at exposure time, before development, so it blurs
    # the image but not the grain. Red is softest: the red-sensitive layer sits
    # under two other layers of gelatin.
    f50s = profile.mtf.f50s()
    for c in range(3):
        t = grid.mtf(f50s[c], profile.mtf.adjacency, profile.mtf.adjacency_um)
        exposure[:, :, c] = apply_transfer(exposure[:, :, c], t)
    np.maximum(exposure, np.float32(0.0), out=exposure)

    # -- 7. collapse to a single emulsion record ------------------------------
    reseau_mask: np.ndarray | None = None
    reseau_pitch_px = 0.0
    if profile.is_monochrome:
        # Weighted by the stock's own spectral sensitivity, not by video luma.
        # For the orthochromatic stock the red weight is 0.02, which is what
        # makes red render black and a blue sky render white.
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

    curves = profile.curves.as_tuple()
    reversal = profile.is_reversal
    anchors = solve_anchors(
        profile, print_stock, settings.grey_target, settings.coupler_scale
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
    del log_e

    # -- 9. DIR coupler inter-image effects ---------------------------------
    cp = profile.couplers
    if cp.active and settings.coupler_scale > 0.0:
        s = cp.strength * settings.coupler_scale
        e = cp.edge_strength * settings.coupler_scale
        if s > 0.0 and not profile.is_monochrome:
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
            fields = tuple(
                make_grain_field(
                    grid, rng, clumps[c], gs.clump_gain, gs.rms_granularity, scan_t
                )
                for c in range(3)
            )
        for c in range(3):
            dmin = curves[c].dmin
            # Poisson statistics of discrete developed crystals: sigma grows
            # as sqrt(density). The fog term keeps grain alive in deep shadow;
            # perfectly clean blacks are one of the loudest digital tells.
            amp = np.sqrt(
                np.maximum(dens[:, :, c] - np.float32(dmin), np.float32(0.0))
                + np.float32(gs.fog_grain)
            ).astype(np.float32)
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

    # -- 13. duplication generations, then print -----------------------------
    if reversal:
        # The slide already is the positive. Its own dmin/dmax become the white
        # and black points; no second curve, no inversion.
        out = dens
        final_curves = curves
    else:
        d_mid = neutral_mid_density(profile, settings.coupler_scale)

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
        "-f", "--format", dest="film_format", default="super35", choices=sorted(FORMATS)
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
        film_format=args.film_format,
        print_stock=args.print_stock,
        exposure_stops=args.exposure,
        scene_kelvin=args.scene_kelvin,
        wb_strength=args.wb_strength,
        grey_target=args.grey_target,
        grain_scale=args.grain_scale,
        halation_scale=args.halation_scale,
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
    px_per_mm = w / FORMATS[args.film_format]
    print(
        f"[INFO] {args.image.name}  {w}x{h}  "
        f"{args.film_format} ({FORMATS[args.film_format]:.2f} mm wide)  "
        f"{px_per_mm:.1f} px/mm"
    )
    if px_per_mm < 60.0:
        # Grain finer than a pixel cannot be resolved. The scanner MTF stage
        # band-limits it so nothing aliases visibly, but the fine-grained stocks
        # will not show their real structure until the render is big enough.
        print(
            f"[WARN] only {px_per_mm:.0f} px/mm; grain structure of fine stocks "
            f"is below the pixel grid. Render at >= "
            f"{round(60.0 * FORMATS[args.film_format]):d} px wide for this format "
            "to resolve it."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    out_rng = np.random.default_rng(args.seed ^ 0x5EED)

    for stock in stocks:
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
        print(f"  -> {stock.name:32s} [{chain}]{note}", flush=True)
        result = simulate(linear, stock, settings)
        dest = args.outdir / f"{stem}_{stock.name}.png"
        save_linear(dest, result, args.bits, out_rng)

    if args.emit_cpp:
        import cpp_codegen

        cpp_codegen.generate(args.outdir)

    print(f"[INFO] wrote {len(stocks)} render(s) to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
