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
# Format resolution helper
# ===========================================================================
def resolve_format(fmt: str | None) -> str:
    """Resolve format alias or loose naming to a valid key in FORMATS."""
    if not fmt:
        return "ff35" if "ff35" in FORMATS else ("super35" if "super35" in FORMATS else next(iter(FORMATS.keys()), "super35"))
    if fmt in FORMATS:
        return fmt
    
    aliases = {
        "35mm": "ff35" if "ff35" in FORMATS else "super35",
        "135": "ff35" if "ff35" in FORMATS else "super35",
        "35mm_still": "ff35" if "ff35" in FORMATS else "super35",
        "s35": "super35",
        "super_35": "super35",
        "16mm": "super16" if "super16" in FORMATS else ("16mm" if "16mm" in FORMATS else "std16"),
        "s16": "super16",
        "8mm": "super8" if "super8" in FORMATS else "std8",
        "s8": "super8",
    }
    norm = aliases.get(fmt.lower(), fmt)
    if norm in FORMATS:
        return norm
    for k in FORMATS:
        if k.lower() == fmt.lower():
            return k
    return "ff35" if "ff35" in FORMATS else ("super35" if "super35" in FORMATS else next(iter(FORMATS.keys()), "super35"))


def get_format_width_mm(fmt: str) -> float:
    """Safely obtain the frame width in millimetres for any given format name."""
    resolved = resolve_format(fmt)
    return float(FORMATS.get(resolved, 36.0))


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
    curves = profile.curves.as_tuple()
    neg_m = profile.dye_matrix
    take = profile.taking_matrix

    log_e_mid = [
        math.log10(max(sum(take[k][j] for j in range(3)), EPS)) for k in range(3)
    ]

    cp_s = profile.couplers.strength * coupler_scale
    couple_flat = cp_s > 0.0 and not profile.is_monochrome

    def _couple(d: list[float]) -> list[float]:
        if not couple_flat:
            return list(d)
        dbar = sum(d) / 3.0
        return [d[k] + cp_s * (d[k] - dbar) for k in range(3)]

    def _neg_density(anchors: list[float]) -> list[float]:
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
                def fn(t: float, c: int = c, frozen: list[float] = frozen) -> float:
                    d = list(frozen)
                    d[c] = density_scalar(-(log_e_mid[c] + t), curves[c])
                    d = _couple(d)
                    mixed = sum(neg_m[c][k] * d[k] for k in range(3))
                    return _normalised_transmittance(mixed, curves[c])

                target = grey_target / _tint_factor(profile, c)
                trims[c] = _bisect(fn, -8.0, 8.0, target, rising=True)
        return (trims[0], trims[1], trims[2])

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

            offsets[c] = _bisect(
                fn, d_mid[c] - 8.0, d_mid[c] + 8.0, targets[c], rising=False
            )
    return offsets


def solve_intermediate_offsets(
    d_mid: list[float], dst_curves: tuple[ToneCurve, ToneCurve, ToneCurve]
) -> tuple[list[float], list[float]]:
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
    lam = lam_nm * 1e-9
    c1 = 3.741771e-16
    c2 = 1.438777e-2
    return c1 / (lam**5 * math.expm1(c2 / (lam * kelvin)))


_SPECTRAL_LAMBDA_STEP = 2.0
_SPECTRAL_LAMBDA_MIN = 360.0
_SPECTRAL_LAMBDA_MAX = 730.0


def spectral_grid() -> np.ndarray:
    n = int(round((_SPECTRAL_LAMBDA_MAX - _SPECTRAL_LAMBDA_MIN)
                  / _SPECTRAL_LAMBDA_STEP)) + 1
    return (_SPECTRAL_LAMBDA_MIN
            + _SPECTRAL_LAMBDA_STEP * np.arange(n, dtype=np.float64))


def layer_sensitivities(profile) -> np.ndarray | None:
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
        interp = np.interp(grid, src_lam, src_val,
                           left=-np.inf, right=-np.inf)
        out[i] = np.where(np.isfinite(interp), np.power(10.0, interp), 0.0)
    return out


def planck_spd(kelvin: float) -> np.ndarray:
    grid = spectral_grid()
    spd = np.array([_planck(float(l), kelvin) for l in grid], dtype=np.float64)
    ref = _planck(560.0, kelvin)
    return spd / ref if ref > 0.0 else spd


def spectral_layer_exposure(profile, spd: np.ndarray) -> np.ndarray | None:
    sens = layer_sensitivities(profile)
    if sens is None:
        return None
    return np.trapezoid(sens * spd[None, :], spectral_grid(), axis=1) \
        if hasattr(np, "trapezoid") else \
        np.trapz(sens * spd[None, :], spectral_grid(), axis=1)


def spectral_balance_gains(profile, scene_kelvin: float) -> tuple[float, ...] | None:
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


_SPECTRAL_BASIS_LAMBDA_MAX = 700.0
_SPECTRAL_OUT_OF_REACH_MAX = 0.15


def spectral_out_of_reach(profile) -> float | None:
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
    sens = layer_sensitivities(profile)
    if sens is None:
        return None
    grid = spectral_grid()
    peaks = [float(grid[int(np.argmax(row))]) for row in sens
             if float(row.max()) > 0.0]
    return max(peaks) if peaks else None


def spectral_monochrome_weights(profile) -> tuple[float, ...] | None:
    sens = layer_sensitivities(profile)
    if sens is None or sens.shape[0] != 1:
        return None

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


_PRIMARY_CENTRES_NM = (600.0, 540.0, 460.0)
_PRIMARY_WIDTH_NM = 55.0


def _srgb_primary_spd() -> np.ndarray:
    grid = spectral_grid()
    out = np.zeros((3, grid.size), dtype=np.float64)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for c, centre in enumerate(_PRIMARY_CENTRES_NM):
        lobe = np.exp(-0.5 * ((grid - centre) / _PRIMARY_WIDTH_NM) ** 2)
        area = float(trap(lobe, grid))
        out[c] = lobe / area if area > 0.0 else lobe
    return out


def spectral_taking_matrix(profile, scene_kelvin: float = 5500.0) -> np.ndarray | None:
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
    lams = (600.0, 550.0, 450.0)
    scene = [_planck(l, scene_kelvin) for l in lams]
    stock = [_planck(l, stock_kelvin) for l in lams]
    ratio = [s / f for s, f in zip(scene, stock)]
    return tuple(r / ratio[1] for r in ratio)


# ===========================================================================
# Frequency-domain helper
# ===========================================================================
class FreqGrid:
    def __init__(self, h: int, w: int, px_per_mm: float, anisotropy: float = 1.0):
        self.h = h
        self.w = w
        self.px_per_mm = px_per_mm

        fy = np.fft.fftfreq(h).astype(np.float32)
        fx = np.fft.rfftfreq(w).astype(np.float32)
        self.fy_cpp = fy[:, None]
        self.fx_cpp = fx[None, :]

        fy_mm = fy * px_per_mm * max(anisotropy, 1e-6)
        fx_mm = fx * px_per_mm
        self.f_mm = np.sqrt(fy_mm[:, None] ** 2 + fx_mm[None, :] ** 2).astype(
            np.float32
        )

        wts = np.full(fx.shape[0], 2.0, dtype=np.float32)
        wts[0] = 1.0
        if w % 2 == 0:
            wts[-1] = 1.0
        self.col_weight = wts[None, :]
        self.n_full = float(h * w)

    def spectral_mean(self, transfer_sq: np.ndarray) -> float:
        return float((transfer_sq * self.col_weight).sum() / self.n_full)

    def gaussian(self, sigma_um: float) -> np.ndarray:
        s_mm = sigma_um / 1000.0
        return np.exp(
            -2.0 * (math.pi**2) * (s_mm**2) * (self.f_mm.astype(np.float32) ** 2)
        ).astype(np.float32)

    def mtf(self, f50_cpmm: float, adjacency: float, adjacency_um: float,
            spec: "fp.MTFSpec | None" = None, channel: int = 1) -> np.ndarray:
        if spec is not None:
            t = fp.mtf_response(spec, channel, self.f_mm).astype(np.float32)
        else:
            t = np.exp(
                -math.log(2.0) * (self.f_mm / np.float32(f50_cpmm)) ** 2
            ).astype(np.float32)
        if adjacency > 0.0:
            lift = 1.0 + adjacency * (
                self.gaussian(adjacency_um * 0.4) - self.gaussian(adjacency_um * 2.0)
            )
            t = (t * lift).astype(np.float32)
        return t

    def multi_gaussian(
        self, radii_um: tuple[float, ...], weights: tuple[float, ...]
    ) -> np.ndarray:
        wsum = float(sum(weights))
        acc = np.zeros_like(self.f_mm)
        for r, wt in zip(radii_um, weights):
            acc += np.float32(wt / wsum) * self.gaussian(r)
        return acc.astype(np.float32)

    def grain_shape(self, clump_um: float, clump_gain: float) -> np.ndarray:
        f_hi = 1000.0 / (2.0 * clump_um)
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
        phase = -2.0 * math.pi * (self.fy_cpp * dy_px + self.fx_cpp * dx_px)
        return np.exp(1j * phase).astype(np.complex64)


def apply_interimage(dens, curves_or_log_e, curves, iie, anchors, reversal):
    log_e = curves_or_log_e
    h, w = dens.shape[0], dens.shape[1]
    m = iie.matrix()
    if reversal:
        d_ref = [float(density_scalar(-float(anchors[c]), curves[c]))
                 for c in range(3)]
    else:
        d_ref = [float(density_scalar(0.0, curves[c])) for c in range(3)]
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
    if not (cp.active and coupler_scale > 0.0):
        return dens
    s = cp.strength * coupler_scale
    e = cp.edge_strength * coupler_scale
    if s > 0.0 and not is_monochrome:
        dbar = dens.mean(axis=2)
        dbar_blur = apply_transfer(dbar, grid.gaussian(cp.radius_um))
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
    return dens


def apply_transfer(plane: np.ndarray, transfer: np.ndarray) -> np.ndarray:
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
    pitch_px = px_per_mm / spec.lines_per_mm
    if pitch_px <= 0:
        raise ValueError("degenerate reseau pitch")

    yy = (np.arange(h, dtype=np.float32)[:, None] / pitch_px).astype(np.int32)
    xx = (np.arange(w, dtype=np.float32)[None, :] / pitch_px).astype(np.int32)

    mask = np.zeros((h, w, 3), dtype=np.float32)
    band = yy % 3
    chequer = (xx + yy) % 2
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
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def grain_reference_energy(
    clump_um: float, clump_gain: float, f_max: float = 400.0, n: int = 16001
) -> float:
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
    shape_t = grid.grain_shape(clump_um, clump_gain)
    if band_limit is not None:
        shape_t = (shape_t * band_limit).astype(np.float32)

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
    film_format: str = "super35"
    print_stock: str = ""
    exposure_stops: float = 0.0
    scene_kelvin: float = 5500.0
    wb_strength: float = 0.0
    grey_target: float = 0.18
    grain_scale: float = 1.0
    halation_scale: float = 1.0
    coupler_scale: float = 1.0
    scanner_f50: float = 0.0
    spectral_balance: bool = True
    spectral_mono: bool = False
    spectral_taking: bool = False
    misreg_scale: float = 1.0
    print_grain: bool = True
    flare: float = -1.0
    generations: int = 0
    dupe_stock: str = "DUPE_FINE_GRAIN"
    reseau: bool = True
    bit_depth: int = 16
    seed: int = 12345
    max_dim: int = 0
    vignette: float = -1.0
    coating_scale: float = 1.0
    frame_index: int = 0

    def flare_for(self, profile: FilmProfile) -> float:
        return profile.default_flare if self.flare < 0.0 else self.flare

    def vignette_for(self, profile: FilmProfile) -> float:
        return profile.default_vignette if self.vignette < 0.0 else self.vignette


# ===========================================================================
# Schema-v4 defects
# ===========================================================================
def vignette_field(h: int, w: int, stops: float) -> np.ndarray:
    if stops <= 0.0:
        return np.ones((h, w), dtype=np.float32)
    cos_c = 2.0 ** (-stops / 4.0)
    tan_c = math.sqrt(max(1.0 / (cos_c * cos_c) - 1.0, 0.0))
    yy = (np.arange(h, dtype=np.float64) - (h - 1) * 0.5) / max((h - 1) * 0.5, 1.0)
    xx = (np.arange(w, dtype=np.float64) - (w - 1) * 0.5) / max((w - 1) * 0.5, 1.0)
    r = np.sqrt((yy * yy)[:, None] + (xx * xx)[None, :]) / math.sqrt(2.0)
    c = 1.0 / np.sqrt(1.0 + (r * tan_c) ** 2)
    return (c ** 4).astype(np.float32)


def coating_field(
    h: int,
    w: int,
    frame_w_mm: float,
    frame_h_mm: float,
    spec,
    frame_index: int,
    pitch_mm: float,
    seed: int,
) -> np.ndarray:
    sigma = float(spec.coating_sigma)
    if sigma <= 0.0:
        return np.ones((h, w), dtype=np.float32)

    lo_x = int(min(max(4.0 * frame_w_mm / max(spec.coating_corr_across_mm, 1e-6),
                       24.0), 192.0))
    lo_y = int(min(max(4.0 * frame_h_mm / max(spec.coating_corr_along_mm, 1e-6),
                       24.0), 192.0))

    y_off_mm = float(frame_index) * float(pitch_mm)
    rng = np.random.default_rng(seed ^ 0x00C0A71C)

    n_comp = 64
    half = sigma / math.sqrt(2.0)

    xs_mm = np.linspace(0.0, frame_w_mm, lo_x, dtype=np.float64)
    ys_mm = np.linspace(0.0, frame_h_mm, lo_y, dtype=np.float64) + y_off_mm

    fxs = rng.normal(0.0, 1.0 / (2.0 * math.pi * spec.coating_corr_across_mm),
                     n_comp)
    phs = rng.uniform(0.0, 2.0 * math.pi, n_comp)
    static = np.zeros(lo_x, dtype=np.float64)
    for k in range(n_comp):
        static += np.cos(2.0 * math.pi * fxs[k] * xs_mm + phs[k])
    static *= half / math.sqrt(n_comp * 0.5)

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
    r2 = ((yy * yy)[:, None] + (xx * xx)[None, :]) * np.float32(0.5)
    wgt = (np.float32(loss) * r2).astype(np.float32)
    return (plane * (1.0 - wgt) + blur * wgt).astype(np.float32)


def edge_fog_density(h: int, w: int, frame_w_mm: float, spec) -> np.ndarray:
    if not spec.has_edge_fog or frame_w_mm <= 0.0:
        return np.zeros((h, w), dtype=np.float32)
    x_mm = np.linspace(0.0, frame_w_mm, w, dtype=np.float64)
    d_edge = np.minimum(x_mm, frame_w_mm - x_mm)
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
    h, w = linear_rgb.shape[:2]
    fmt = resolve_format(settings.film_format)
    negative_width_mm = get_format_width_mm(fmt)
    px_per_mm = w / negative_width_mm
    rng = np.random.default_rng(settings.seed)
    print_stock = get_print_stock(settings.print_stock or profile.default_print)

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
    take = np.asarray(profile.taking_matrix, dtype=np.float32)
    if settings.spectral_taking:
        derived = spectral_taking_matrix(profile, settings.scene_kelvin)
        if derived is not None:
            take = derived
    if not np.allclose(take, np.eye(3, dtype=np.float32)):
        exposure = (exposure.reshape(-1, 3) @ take.T).reshape(h, w, 3)
        exposure = np.ascontiguousarray(exposure, dtype=np.float32)

    # -- 3. stock colour balance --------------------------------------------
    if settings.wb_strength > 0.0 and not profile.is_monochrome:
        gains = None
        if settings.spectral_balance:
            gains = spectral_balance_gains(profile, settings.scene_kelvin)
        if gains is None:
            gains = balance_gains(settings.scene_kelvin, profile.balance_kelvin)
        for c in range(3):
            g = 1.0 + (gains[c] - 1.0) * settings.wb_strength
            exposure[:, :, c] *= np.float32(g)

    # -- 3b. veiling flare from the taking lens -------------------------------
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

    # -- 4. coating unevenness & lens vignette ------------------------------
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
            frame_pitch_mm(fmt), settings.seed,
        )
        field = cf if field is None else (field * cf)
    if field is not None:
        exposure *= field[:, :, None]
        del field

    # -- 5. halation (in linear exposure) -----------------------------------
    hal = profile.halation
    if hal.active and settings.halation_scale > 0.0:
        scatter = grid.multi_gaussian(hal.radii_um, hal.weights)
        thr = np.float32(2.0**hal.threshold_stops)
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
            src = (0.5 * exposure[:, :, c] + 0.5 * lum).astype(np.float32)
            above = _softplus(src - thr, float(knee))
            exposure[:, :, c] += np.float32(
                gains[c] * settings.halation_scale
            ) * (apply_transfer(above, scatter) - above)
        del lum

    np.maximum(exposure, np.float32(0.0), out=exposure)

    # -- 6. emulsion MTF on the exposure ------------------------------------
    f50s = profile.mtf.f50s()
    for c in range(3):
        t = grid.mtf(f50s[c], profile.mtf.adjacency, profile.mtf.adjacency_um,
                     spec=profile.mtf, channel=c)
        exposure[:, :, c] = apply_transfer(exposure[:, :, c], t)
    np.maximum(exposure, np.float32(0.0), out=exposure)

    # -- 6b. corner defocus from film buckling in the gate -------------------
    if cs > 0.0 and coat.has_buckle:
        loss = min(coat.buckle_mtf_loss * cs, 0.9)
        for c in range(3):
            exposure[:, :, c] = corner_defocus(exposure[:, :, c], loss)

    # -- 7. collapse to a single emulsion record ------------------------------
    reseau_mask: np.ndarray | None = None
    reseau_pitch_px = 0.0
    if profile.is_monochrome:
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
            fm = np.asarray(spec.filter_matrix, dtype=np.float32)
            record = np.zeros((h, w), dtype=np.float32)
            for c in range(3):
                through = (
                    fm[c, 0] * exposure[:, :, 0]
                    + fm[c, 1] * exposure[:, :, 1]
                    + fm[c, 2] * exposure[:, :, 2]
                ).astype(np.float32)
                record += mask[:, :, c] * through
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
        for c in range(3):
            dens[:, :, c] = density(
                -(log_e[:, :, c] + np.float32(anchors[c])), curves[c]
            )
    else:
        for c in range(3):
            dens[:, :, c] = density(log_e[:, :, c], curves[c])

    # -- 8b. interimage effects ---------------------------------------------
    if profile.interimage.active and not profile.is_monochrome:
        apply_interimage(dens, log_e, curves, profile.interimage,
                         anchors, reversal)
    del log_e

    # -- 9. DIR coupler inter-image effects ---------------------------------
    apply_dir_couplers(dens, profile.couplers, grid,
                       settings.coupler_scale, profile.is_monochrome)

    np.maximum(dens, np.float32(0.0), out=dens)

    # -- 10. scan the image -------------------------------------------------
    scan_f50 = settings.scanner_f50 or print_stock.mtf_f50
    scan_t = grid.mtf(scan_f50, 0.0, 0.0)
    mis_px = profile.misregistration_um * px_per_mm / 1000.0 * settings.misreg_scale
    for c in range(3):
        t = scan_t
        if mis_px > 0.0 and not profile.is_monochrome:
            dy = float(rng.normal(0.0, mis_px))
            dx = float(rng.normal(0.0, mis_px))
            t = (scan_t * grid.shift(dy, dx)).astype(np.complex64)
        dens[:, :, c] = apply_transfer(dens[:, :, c], t)

    np.maximum(dens, np.float32(0.0), out=dens)

    # -- 10b. narrow-gauge edge fog -----------------------------------------
    if cs > 0.0 and coat.has_edge_fog:
        fog = edge_fog_density(h, w, negative_width_mm, coat) * np.float32(cs)
        dens += fog[:, :, None]
        del fog

    # -- 11. grain, in the density domain -------------------------------------
    gs = profile.grain
    if settings.grain_scale > 0.0:
        clumps = gs.clumps()
        if profile.is_monochrome or reseau_mask is not None:
            field = make_grain_field(
                grid, rng, clumps[1], gs.clump_gain, gs.rms_granularity, scan_t
            )
            fields = (field, field, field)
        else:
            rms_c = gs.rms_rgb()
            fields = tuple(
                make_grain_field(
                    grid, rng, clumps[c], gs.clump_gain, rms_c[c], scan_t
                )
                for c in range(3)
            )
        for c in range(3):
            dmin = curves[c].dmin
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

    # -- 13. duplication generations, then print -----------------------------
    if reversal:
        out = dens
        final_curves = curves
    else:
        d_mid = neutral_mid_density(profile, settings.coupler_scale)
        stages = 2 * max(0, settings.generations)
        if stages:
            dupe = get_print_stock(settings.dupe_stock)
            dcurves = dupe.curves.as_tuple()
            dupe_mtf = grid.mtf(dupe.mtf_f50, 0.0, 0.0)
            for _ in range(stages):
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
        t_max = 10.0 ** (-fc.dmin)
        t_min = 10.0 ** (-fc.dmax)
        trans = np.power(np.float32(10.0), -out[:, :, c], dtype=np.float32)
        out[:, :, c] = ((trans - t_min) / (t_max - t_min)).astype(np.float32)

    # -- 14b. reseau reconstruction ------------------------------------------
    if reseau_mask is not None:
        out = reseau_reconstruct(
            out[:, :, 1], reseau_mask, grid, reseau_pitch_px, profile.reseau
        )

    tint = profile.base_tint
    for c in range(3):
        if tint[c] != 1.0:
            out[:, :, c] *= np.float32(1.0 + (tint[c] - 1.0) * 0.5)

    # -- 14c. silver image tone (monochrome only) ----------------------------
    if profile.is_monochrome and profile.silver_tone != 0.0:
        tone = np.float32(profile.silver_tone)
        w = out[:, :, 1]
        out[:, :, 0] *= (1.0 + np.float32(0.28) * tone * w)
        out[:, :, 2] *= (1.0 - np.float32(0.22) * tone * w)

    return np.clip(out, 0.0, 1.0, out=out)


# ===========================================================================
# I/O
# ===========================================================================
def load_linear(path: Path, max_dim: int = 0) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max_dim and max(im.size) > max_dim:
            scale = max_dim / max(im.size)
            new = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            im = im.resize(new, resample=Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.uint8)
    return srgb_to_linear(arr.astype(np.float32) / 255.0)


def save_linear(path: Path, linear: np.ndarray, bit_depth: int, rng) -> None:
    enc = linear_to_srgb(linear)
    peak = float((1 << bit_depth) - 1)
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
        help="override the gauge. Default: each stock's own native gauge"
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
        help="how much colour-temperature mismatch to apply",
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
        help="veiling flare fraction of the taking lens; -1 uses stock default",
    )
    p.add_argument(
        "-g",
        "--generations",
        type=int,
        default=0,
        help="intermediate duplication rounds between negative and print",
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
        film_format=resolve_format(args.film_format) if args.film_format else "super35",
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
    if args.film_format is not None:
        fmt_name = resolve_format(args.film_format)
        print(f"[INFO] {args.image.name}  {w}x{h}  gauge overridden to "
              f"{fmt_name} ({get_format_width_mm(fmt_name):.2f} mm) for every stock")
    else:
        print(f"[INFO] {args.image.name}  {w}x{h}  "
              f"each stock rendered at its own native gauge")

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    out_rng = np.random.default_rng(args.seed ^ 0x5EED)

    for stock in stocks:
        raw_fmt = args.film_format or stock.default_format
        fmt = resolve_format(raw_fmt)
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
        ppmm = linear.shape[1] / get_format_width_mm(fmt)
        print(f"  -> {stock.name:32s} [{chain}]  {fmt} "
              f"{ppmm:.0f}px/mm{note}", flush=True)
        result = simulate(linear, stock, settings)
        dest = args.outdir / f"{stem}_{stock.name}.png"
        save_linear(dest, result, args.bits, out_rng)

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
            fmt_neg = resolve_format(neg.default_format)
            st = dataclasses.replace(
                settings, film_format=fmt_neg, print_stock=ps.name)
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