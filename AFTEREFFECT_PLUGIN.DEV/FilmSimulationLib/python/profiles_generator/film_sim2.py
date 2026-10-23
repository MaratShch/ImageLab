#!/usr/bin/env python3
"""film_sim2.py -- the film_sim.py renderer with a SPECTRAL taking side and a
SPECTRAL scanner read-out. A research experiment, Python only, to compare
against film_sim.py.

    python film_sim2.py Card.png -p all -8bpp

Same command line, same flags, same output files as film_sim.py. The only
default that differs is the output folder: film_renders_spectral (so the two
renderers never overwrite each other); -o still overrides it.

NOTHING ELSE CHANGES. film_sim.py, film_profiles.py, the database (format and
content), the C++ sources and every other file are left exactly as they are.
This module imports film_sim and runs its `simulate()` unchanged; for the
duration of each of ITS renders it substitutes, in memory only, the two places
where the image enters and leaves the film. Importing film_sim2 does not alter
what film_sim.simulate() does when it is called directly.

Accuracy is preferred over speed throughout: both ends are computed exactly per
pixel (no interpolated lookup tables in the result); tables are used only as
starting points for the solvers.

1. TAKING SIDE -- RGB to spectrum (Jakob & Hanika 2019), exact per colour
   Each pixel's linear Rec.709 RGB is turned into the smooth spectrum
       s(lambda) = S(c0 t^2 + c1 t + c2),  S(x) = 1/2 + x / (2 sqrt(1 + x^2))
   whose CIE 1931 / D65 colour equals the pixel's (Gauss-Newton on CIELAB,
   exactly the reference optimiser in spectral_upsampling.fit), solved for
   EVERY distinct colour in the frame, warm-started from the frozen 64 x 64
   coefficient table. The spectrum is multiplied by the scene illuminant
   (blackbody at --scene-kelvin) and integrated against each emulsion layer's
   MEASURED spectral sensitivity (film_sim.layer_sensitivities, on
   film_sim.spectral_grid), normalised so a neutral stays neutral -- the same
   definition film_sim's spectral path uses; film_sim interpolates the
   per-layer responses in a table instead, which costs up to ~0.8 % on
   saturated colours. Stocks with no traced three-layer sensitivity, and
   beam-splitter stocks (Technicolor) whose taking filters are authored, keep
   film_sim's authored taking matrix, exactly as film_sim decides.

2. READ-OUT SIDE -- dye spectra instead of the 3x3 dye_matrix, exact per pixel
   For stocks that carry a traced THREE-DYE spectral density panel (cyan,
   magenta, yellow; plus the D-min / orange-mask trace where the sheet prints
   one) and are read by a scanner (SCAN_DI), stage 12's 3x3 `dye_matrix` is
   replaced, per pixel, by:
     a) Status densities (what the characteristic curves are measured in:
        Status A for slides, Status M for negatives; ISO 5-3 responses)
        -> dye AMOUNTS (c, m, y), Newton with the exact Jacobian, to 1e-6 D;
     b) transmittance  T(lambda) = 10^-(Dmin(lambda) + c Dc + m Dm + y Dy);
     c) T(lambda) read by a SCANNER: the measured RGB LED panel x the matching
        Bayer channel of a Sony ILCE-7RM3 -- the same measured rig film_sim
        already uses to derive the 16 negatives' stage-12 matrices.
   1 nm integration grid, 380-780 nm (the grid dye_matrix_from_spectra uses;
   5 nm was measured to cost up to 0.01 D). Panels that stop near 700 nm are
   HELD at their last value (cyan still absorbs there), the convention
   dye_matrix_from_spectra established.
   Step a) avoids double counting: the curves already contain the dyes'
   unwanted absorptions, so they are first converted back into dye amounts and
   only then read with the real reader.
   A dye layer cannot hold more dye than its Dmax: density above Dmax (added
   by stages 9 and 11, which are additive in density) is not dye, so it passes
   unchanged and film_sim caps it at stage 14 as it always does.
   Neutral convention: the scanner is calibrated on the FILM'S OWN GREY
   SCALE, as a real scan is. A per-channel 1-D correction, computed exactly,
   makes every density a neutral ramp reaches on this frame (the stock's three
   curves at equal exposure, with film_sim's own solved anchors, balance gains
   and flat coupler term -- the masked grey of a negative, not an equal-density
   triple) pass the read-out unchanged, so scan timing and grey balance are
   film_sim's and only COLOURS differ. Stocks printed onto a real print stock
   (Technicolor IB) keep film_sim's print path: their reader is a print
   emulsion.

Between the two ends the film works per layer, as real film does (each layer
turns the spectrum into ONE exposure): curves, interimage, fourth layer,
couplers, halation, grain, scan and display are film_sim's own code.

Each render prints which spectral parts ran and their numerical accuracy; the
end of the run prints a summary.

Dependencies: the generator folder it ships in (film_sim.py, film_profiles.py,
spectral_upsampling.py, spectral_upsampling_coeffs.py, iso_5_3_status.py,
scanner_reader_data.py) and numpy -- nothing new. Part of the Python generator
package since 2026-10-22b (owner request) as a RESEARCH renderer: it is not
used by the build, the C++ engines or any gate, and film_sim.py's results do
not depend on it.
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

import film_sim as _fs
from film_sim import *  # noqa: F401,F403  -- same public interface as film_sim
import iso_5_3_status as _iso
import scanner_reader_data as _srd
import spectral_upsampling as _su

__all__ = [n for n in dir(_fs) if not n.startswith("_")] + [
    "simulate", "build_parser", "main", "spectral_report"]

#: Default output folder (film_sim.py: film_renders).
DEFAULT_OUTDIR = Path("film_renders_spectral")

#: The scanner used for the spectral read-out (scanner_reader_data.READER_RIGS).
READER_RIG = "rgb_led_sony_ilce7rm3"
READER_CAMERA = "Sony ILCE-7RM3"

#: Read-out integration grid, nm, and trapezoid weights.
_LAM = np.arange(380.0, 780.0 + 1e-9, 1.0)
_W = np.ones_like(_LAM)
_W[0] = _W[-1] = 0.5

#: Nodes per axis of the table that only seeds the read-out solver.
_SEED_N = 33
#: Read-out solver: tolerance on the Status densities, iteration cap.
_RO_TOL_D = 1e-6
_RO_ITERS = 60
#: Pixels per solver chunk (memory bound: chunk x 3 x 401 doubles).
_RO_CHUNK = 8192

_ORIG_SIMULATE = _fs.simulate
_ORIG_CALLIER = _fs.callier_density
_ORIG_BUILD_PARSER = _fs.build_parser
_ORIG_UPSAMPLING_LUT = _fs.spectral_upsampling_lut
_ORIG_APPLY_LUT = _su.apply_lut
_ORIG_SOLVE_ANCHORS = _fs.solve_anchors

_IN_SPECTRAL = False          # a film_sim2 render is in progress
_ACTIVE_READOUT = None        # its read-out model, or None
_STATS: dict = {}             # accuracy figures of the render in progress
_READOUT_CACHE: dict = {}
_CTX: dict = {}               # settings / anchors of the render in progress
_REPORT: dict = {}            # stock name -> what ran


def _trap(y, x, axis=-1):
    f = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return f(y, x, axis=axis)


# ===========================================================================
# 1. Taking side: exact Jakob-Hanika per distinct colour
# ===========================================================================
class _ExactTaking:
    """Stands in for film_sim's per-stock response table; carries what the
    exact integration needs."""

    def __init__(self, sens, illum, grid):
        self.sens = np.asarray(sens, dtype=np.float64)
        self.illum = np.asarray(illum, dtype=np.float64)
        self.grid = np.asarray(grid, dtype=np.float64)
        self.norm = np.array([_trap(self.sens[l] * self.illum, self.grid)
                              for l in range(3)])
        self.t = ((self.grid - _su.POLY_LAMBDA_MIN)
                  / (_su.POLY_LAMBDA_MAX - _su.POLY_LAMBDA_MIN))


def _lab_rows(xyz):
    f = xyz / _su._WHITE[None, :]
    d = 6.0 / 29.0
    g = np.where(f > d ** 3, np.cbrt(np.maximum(f, 0.0)), f / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116.0 * g[:, 1] - 16.0, 500.0 * (g[:, 0] - g[:, 1]),
                     200.0 * (g[:, 1] - g[:, 2])], axis=1)


def _residual_rows(c, lab_t):
    s = _su.sigmoid((c[:, 0:1] * _su._TN + c[:, 1:2]) * _su._TN + c[:, 2:3])
    return lab_t - _lab_rows(s @ _su._CMF.T)


def _fit_rows(rgb, c0):
    """Vectorised spectral_upsampling.fit: same residual (CIELAB), same
    central-difference Jacobian (h = 1e-4), same coefficient cap, same
    convergence test. Returns (coefficients, dE76)."""
    lab_t = _lab_rows(rgb @ _su._M_RGB2XYZ.T)
    c = c0.copy()
    act = np.ones(len(c), dtype=bool)
    eye = np.eye(3)
    for _ in range(_su.GN_ITERATIONS):
        idx = np.nonzero(act)[0]
        if idx.size == 0:
            break
        ca, la = c[idx], lab_t[idx]
        r = _residual_rows(ca, la)
        jac = np.empty((idx.size, 3, 3))
        for k in range(3):
            e = eye[k] * 1e-4
            jac[:, :, k] = (_residual_rows(ca + e, la)
                            - _residual_rows(ca - e, la)) / 2e-4
        det = np.linalg.det(jac)
        good = np.abs(det) > 1e-300
        step = np.zeros_like(ca)
        if good.any():
            step[good] = np.linalg.solve(jac[good], r[good][:, :, None])[:, :, 0]
        ca = ca - step
        cmax = np.abs(ca).max(axis=1)
        over = cmax > _su.COEFF_CAP
        ca[over] *= (_su.COEFF_CAP / cmax[over])[:, None]
        c[idx] = ca
        done = (np.einsum("ij,ij->i", step, step) < _su.GN_TOLERANCE) | ~good
        act[idx[done]] = False
    de = np.linalg.norm(_residual_rows(c, lab_t), axis=1)
    return c, de


def _table_start(rgbn, table):
    """Warm start: the frozen table's coefficients, bilinearly interpolated
    (face = largest component, ties to the later channel, as apply_lut)."""
    res = table.shape[1]
    r, g, b = rgbn[:, 0], rgbn[:, 1], rgbn[:, 2]
    face = np.zeros(r.shape, dtype=np.int64)
    face = np.where(g >= r, 1, face)
    mx = np.where(face == 1, g, r)
    face = np.where(b >= mx, 2, face)
    m = np.maximum(np.maximum(r, g), b)
    x = np.where(face == 0, g, np.where(face == 1, b, r)) / m
    y = np.where(face == 0, b, np.where(face == 1, r, g)) / m
    fx = np.clip(x * (res - 1), 0.0, res - 1.0)
    fy = np.clip(y * (res - 1), 0.0, res - 1.0)
    ix = np.minimum(fx.astype(np.int64), res - 2)
    iy = np.minimum(fy.astype(np.int64), res - 2)
    ax, ay = (fx - ix)[:, None], (fy - iy)[:, None]
    top = table[face, iy, ix] + (table[face, iy, ix + 1] - table[face, iy, ix]) * ax
    bot = table[face, iy + 1, ix] + (table[face, iy + 1, ix + 1]
                                     - table[face, iy + 1, ix]) * ax
    return top + (bot - top) * ay


_TABLE = None


def _exact_upsample(exposure, model: _ExactTaking):
    """(h, w, 3) linear exposure -> per-layer exposure, exact per colour."""
    global _TABLE
    if _TABLE is None:
        _TABLE = _su.table()
    e = np.maximum(exposure.reshape(-1, 3).astype(np.float64), 0.0)
    m = e.max(axis=1)
    ok = m > 0.0
    out = np.zeros_like(e)
    if ok.any():
        rgbn = e[ok] / (2.0 * m[ok][:, None])          # max component 0.5
        uniq, inv = np.unique(rgbn, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        resp = np.empty_like(uniq)
        de_max = 0.0
        for s in range(0, len(uniq), 4096):
            u = uniq[s:s + 4096]
            c, de = _fit_rows(u, _table_start(u, _TABLE))
            de_max = max(de_max, float(de.max()))
            spec = _su.sigmoid((c[:, 0:1] * model.t + c[:, 1:2]) * model.t
                               + c[:, 2:3])                   # (n, grid)
            for l in range(3):
                resp[s:s + 4096, l] = (2.0 * _trap(
                    spec * (model.sens[l] * model.illum)[None, :], model.grid,
                    axis=1) / model.norm[l])
        out[ok] = resp[inv] * m[ok][:, None]
        _STATS["input_colours"] = int(len(uniq))
        _STATS["input_fit_max_dE76"] = de_max
        # what film_sim's interpolated table would have given, for the report
        try:
            lut = _ORIG_UPSAMPLING_LUT(_STATS["_profile"], _STATS["_kelvin"])
            ref = _ORIG_APPLY_LUT(exposure, lut).reshape(-1, 3).astype(np.float64)
            rel = np.abs(ref[ok] - out[ok]) / np.maximum(out[ok].max(axis=1, keepdims=True), 1e-12)
            _STATS["input_vs_table_max_rel"] = float(rel.max())
        except Exception:                                     # pragma: no cover
            pass
    return out.reshape(exposure.shape).astype(np.float32)


def _upsampling_lut_hook(profile, scene_kelvin: float = 5500.0):
    if not _IN_SPECTRAL:
        return _ORIG_UPSAMPLING_LUT(profile, scene_kelvin)
    sens = _fs.layer_sensitivities(profile)
    if sens is None or sens.shape[0] != 3 or scene_kelvin <= 0.0:
        return None
    model = _ExactTaking(sens, _fs.planck_spd(scene_kelvin), _fs.spectral_grid())
    if not np.all(model.norm > 0.0):
        return None
    _STATS["_profile"], _STATS["_kelvin"] = profile, scene_kelvin
    return model


def _apply_lut_hook(exposure, lut):
    if isinstance(lut, _ExactTaking):
        return _exact_upsample(exposure, lut)
    return _ORIG_APPLY_LUT(exposure, lut)


# ===========================================================================
# 2. Read-out side: dye spectra read by the scanner, exact per pixel
# ===========================================================================
def _hold(lam_src, y, lam_dst):
    y = np.asarray(y, dtype=np.float64)
    return np.interp(lam_dst, lam_src, y, left=y[0], right=y[-1])


def _scanner_responses():
    """(3, L) r g b: measured LED channel power x matching Sony Bayer channel."""
    led = _srd.SCANNER_LED_SPD
    ll = led["lambda_start_nm"] + led["lambda_step_nm"] * np.arange(len(led["r"]))
    cam = _srd.CAMERA_SSF[READER_CAMERA]
    out = []
    for band in ("r", "g", "b"):
        e = np.interp(_LAM, ll, np.asarray(led[band], float), left=0.0, right=0.0)
        cl = cam["lambda_start_nm"] + cam["lambda_step_nm"] * np.arange(len(cam[band]))
        q = np.interp(_LAM, cl, np.clip(np.asarray(cam[band], float), 0.0, None),
                      left=0.0, right=0.0)
        out.append(e * q)
    return np.stack(out)


class _Readout:
    """Status densities -> dye amounts -> spectrum -> scanner densities."""

    def __init__(self, profile):
        d = profile.dye_density
        n = len(d.d_cyan)
        lam = d.lambda_start_nm + d.lambda_step_nm * np.arange(n)
        self.dyes = np.stack([_hold(lam, a, _LAM)
                              for a in (d.d_cyan, d.d_magenta, d.d_yellow)])
        self.has_dmin = bool(d.d_dmin) and len(d.d_dmin) == n
        self.base = _hold(lam, d.d_dmin, _LAM) if self.has_dmin else np.zeros_like(_LAM)
        st = (profile.density_metric or "").strip().lower()
        self.status = np.stack(_iso.responses(st, _LAM)) * _W[None, :]
        self.scan = _scanner_responses() * _W[None, :]
        self.cap = np.array([c.dmax for c in profile.curves.as_tuple()])
        # linear start for the seed table
        d0 = self.density(self.status, np.zeros((1, 3)))[0]
        m0 = np.stack([self.density(self.status, np.eye(3)[j:j + 1])[0] - d0
                       for j in range(3)], axis=1)
        self._lin = (d0, np.linalg.inv(m0))
        # seed table of amounts over 0..cap
        self.seed_axes = [np.linspace(0.0, cp, _SEED_N) for cp in self.cap]
        A, B, C = np.meshgrid(*self.seed_axes, indexing="ij")
        tgt = np.stack([A.ravel(), B.ravel(), C.ravel()], axis=1)
        a0 = self.linear_start(tgt)
        amt, _ = self.solve(tgt, a0)
        self.seed = amt.reshape(_SEED_N, _SEED_N, _SEED_N, 3)
        # exact neutral-diagonal correction, fine 1-D grid
        dd = np.linspace(0.0, float(self.cap.max()), 1201)
        diag = np.stack([dd, dd, dd], axis=1)
        raw = self.raw(diag, self.solve(diag, self.linear_start(diag))[0])
        self.set_correction(diag, raw)
        self.corr_diag = list(self.corr)      # fallback, restored per render

    def set_correction(self, track, raw):
        """Per-channel 1-D maps raw scanner density -> Status density, fitted
        so every point of `track` (K, 3) passes the read-out unchanged."""
        self.corr = []
        for c in range(3):
            o = np.argsort(raw[:, c], kind="stable")
            xs = np.maximum.accumulate(raw[o, c])
            ys = np.maximum.accumulate(track[o, c])
            keep = np.concatenate([[True], np.diff(xs) > 1e-9])
            self.corr.append((xs[keep], ys[keep]))

    def set_grey_track(self, track):
        """Neutral = the FILM'S OWN grey scale on this frame (see simulate)."""
        tc = np.clip(track, 0.0, self.cap[None, :])
        a, _ = self.solve(tc, self.linear_start(tc))
        self.set_correction(tc, self.raw(tc, a))

    # -- band densities of (N, 3) dye amounts through weighted responses
    def density(self, resp, a):
        od = self.base[None, :] + a @ self.dyes
        t = np.power(10.0, -np.clip(od, -2.0, 30.0))
        return -np.log10(np.maximum((t @ resp.T) / resp.sum(axis=1)[None, :],
                                    1e-300))

    def linear_start(self, target):
        d0, minv = self._lin
        return np.clip((target - d0[None, :]) @ minv.T, -1.0, 60.0)

    def solve(self, target, a):
        """Newton, exact Jacobian dD_b/da_j = int(S_b T e_j) / int(S_b T)."""
        a = a.copy()
        act = np.ones(len(a), dtype=bool)
        for _ in range(_RO_ITERS):
            idx = np.nonzero(act)[0]
            if idx.size == 0:
                break
            aa = a[idx]
            od = self.base[None, :] + aa @ self.dyes
            t = np.power(10.0, -np.clip(od, -2.0, 30.0))
            st = t[:, None, :] * self.status[None, :, :]      # (n, 3, L)
            den = st.sum(axis=2)
            d = -np.log10(np.maximum(den / self.status.sum(axis=1)[None, :], 1e-300))
            r = d - target[idx]
            conv = np.abs(r).max(axis=1) < _RO_TOL_D
            act[idx[conv]] = False
            jac = (st @ self.dyes.T) / np.maximum(den[:, :, None], 1e-300)
            det = np.linalg.det(jac)
            good = (np.abs(det) > 1e-300) & ~conv
            if not good.any():
                break
            step = np.zeros_like(aa)
            step[good] = np.linalg.solve(jac[good], r[good][:, :, None])[:, :, 0]
            a[idx] = np.clip(aa - np.clip(step, -2.0, 2.0), -1.0, 60.0)
        res = np.abs(self.density(self.status, a) - target).max(axis=1)
        return a, res

    def raw(self, target, a):
        """Scanner densities of the solved dyes, plus any Status residual the
        dyes could not reach (non-zero only outside what the dyes can make, e.g.
        below the D-min trace), passed with slope 1."""
        return self.density(self.scan, a) + (target - self.density(self.status, a))

    def neutral_correct(self, raw):
        out = np.empty_like(raw)
        for c in range(3):
            xs, ys = self.corr[c]
            x = raw[:, c]
            y = np.interp(x, xs, ys)
            lo, hi = x < xs[0], x > xs[-1]
            y[lo] = ys[0] + (x[lo] - xs[0])
            y[hi] = ys[-1] + (x[hi] - xs[-1])
            out[:, c] = y
        return out

    def seed_lookup(self, dc):
        n = _SEED_N
        idx, frac = [], []
        for c in range(3):
            x = dc[:, c] / (self.cap[c] / (n - 1)) if self.cap[c] > 0 else 0 * dc[:, c]
            i = np.minimum(x.astype(np.int64), n - 2)
            idx.append(i)
            frac.append((x - i)[:, None])
        i0, i1, i2 = idx
        f0, f1, f2 = frac
        T = self.seed
        c00 = T[i0, i1, i2] + (T[i0 + 1, i1, i2] - T[i0, i1, i2]) * f0
        c10 = T[i0, i1 + 1, i2] + (T[i0 + 1, i1 + 1, i2] - T[i0, i1 + 1, i2]) * f0
        c01 = T[i0, i1, i2 + 1] + (T[i0 + 1, i1, i2 + 1] - T[i0, i1, i2 + 1]) * f0
        c11 = T[i0, i1 + 1, i2 + 1] + (T[i0 + 1, i1 + 1, i2 + 1]
                                       - T[i0, i1 + 1, i2 + 1]) * f0
        c0 = c00 + (c10 - c00) * f1
        c1 = c01 + (c11 - c01) * f1
        return c0 + (c1 - c0) * f2

    def apply(self, dens):
        flat = dens.reshape(-1, 3).astype(np.float64)
        out = np.empty_like(flat)
        worst = 0.0
        unreached = 0
        negative = 0
        for s in range(0, len(flat), _RO_CHUNK):
            d = flat[s:s + _RO_CHUNK]
            dc = np.clip(d, 0.0, self.cap[None, :])
            a, res = self.solve(dc, self.seed_lookup(dc))
            worst = max(worst, float(res.max()))
            unreached += int((res >= 1e-4).sum())
            # a Status triple only reachable with a NEGATIVE amount of some dye
            # is a colour the real dyes cannot make: the per-layer curves
            # upstream produced it. Counted, not hidden.
            negative += int((a.min(axis=1) < -0.02).sum())
            out[s:s + _RO_CHUNK] = self.neutral_correct(self.raw(dc, a)) + (d - dc)
        _STATS["readout_px"] = len(flat)
        _STATS["readout_unreached_px"] = unreached
        _STATS["readout_negative_dye_px"] = negative
        _STATS["readout_change_mean_D"] = float(np.abs(out - flat).mean())
        _STATS["readout_change_max_D"] = float(np.abs(out - flat).max())
        return out.reshape(dens.shape).astype(np.float32)


def _readout_for(profile):
    if profile.name not in _READOUT_CACHE:
        st = (profile.density_metric or "").strip().lower()
        _READOUT_CACHE[profile.name] = (_Readout(profile) if st in _iso.STATUS
                                        else None)
    return _READOUT_CACHE[profile.name]


def _grey_track(prof, anchors, cs_flat, settings, spectral_in):
    """(K, 3) densities a NEUTRAL ramp reaches on this frame before stage 12.

    The same arithmetic simulate() applies to a grey pixel: per-channel log
    exposure offsets (authored taking-matrix row sums when the taking is not
    spectral -- the spectral one is neutral-preserving --, colour balance
    gains, reciprocity and age shifts), the curves (reversal: against the
    negated exposure plus the solved anchors), the flat-field coupler term
    (neutral_mid_density's form). Interimage is zero on neutrals by design."""
    x = np.linspace(-8.0, 6.0, 2801)
    off = np.zeros(3)
    if not spectral_in:
        take = np.asarray(prof.taking_matrix, dtype=np.float64)
        off += np.log10(np.maximum(take.sum(axis=1), _fs.EPS))
    if settings.wb_strength > 0.0 and not prof.is_monochrome:
        g = (_fs.spectral_balance_gains(prof, settings.scene_kelvin)
             if settings.spectral_balance else None)
        if g is None:
            g = _fs.balance_gains(settings.scene_kelvin, prof.balance_kelvin)
        off += np.log10([1.0 + (g[c] - 1.0) * settings.wb_strength
                         for c in range(3)])
    if settings.exposure_time_s > 0.0:
        off += np.asarray(_fs.reciprocity_log_shift(prof, settings.exposure_time_s),
                          dtype=np.float64)
    age = float(getattr(settings, "film_age_months", 0.0) or 0.0)
    if age > 0.0:
        off += float(_fs.film_age_log_shift(prof, age))
    curves = prof.curves.as_tuple()
    d = np.empty((x.size, 3))
    for c in range(3):
        le = x + off[c]
        if prof.is_reversal:
            le = -(le + float(anchors[c]))
        d[:, c] = _fs.density(le.astype(np.float32), curves[c]).astype(np.float64)
    cp = float(prof.couplers.strength) * float(cs_flat)
    if cp > 0.0 and not prof.is_monochrome:
        d = d + cp * (d - d.mean(axis=1, keepdims=True))
    return d


def _anchors_hook(*args, **kw):
    res = _ORIG_SOLVE_ANCHORS(*args, **kw)
    if "anchors" not in _CTX:
        prof = args[0] if args else kw["profile"]
        cs = args[3] if len(args) > 3 else kw.get("coupler_scale", 1.0)
        _CTX["anchors"] = (prof, cs, res)
    return res


def _callier_hook(dens, curves, callier_q, specular, is_monochrome):
    """Runs where film_sim's stage 12 ends: the spectral read-out first (only
    while a film_sim2 render is in progress), then film_sim's own Callier."""
    if _ACTIVE_READOUT is not None and dens.ndim == 3 and dens.shape[2] == 3:
        ro = _ACTIVE_READOUT
        ro.corr = list(ro.corr_diag)
        if "anchors" in _CTX:
            prof, cs, anchors = _CTX["anchors"]
            ro.set_grey_track(_grey_track(prof, anchors, cs, _CTX["settings"],
                                          _CTX["spectral_in"]))
            _STATS["neutral"] = "film grey scale"
        else:
            _STATS["neutral"] = "equal-density diagonal (anchors not seen)"
        dens[...] = ro.apply(dens)
        np.maximum(dens, np.float32(0.0), out=dens)
    return _ORIG_CALLIER(dens, curves, callier_q, specular, is_monochrome)


# ===========================================================================
# public interface (same signatures as film_sim)
# ===========================================================================
def simulate(linear_rgb, profile, settings):
    """film_sim.simulate with the spectral taking and spectral read-out."""
    global _IN_SPECTRAL, _ACTIVE_READOUT
    settings = dataclasses.replace(settings, spectral_taking=True,
                                   spectral_upsampling=True)
    spectral_in = (_fs.spectral_taking_matrix(profile, settings.scene_kelvin)
                   is not None and _fs.layer_sensitivities(profile) is not None)
    reader = (_fs.print_stock_key(settings.print_stock) or profile.default_print)
    ro, why = None, ""
    if profile.is_monochrome:
        why = "monochrome"
    elif not profile.dye_density.has_data:
        why = "no three-dye panel"
    elif profile.is_reversal or reader == "SCAN_DI":
        ro = _readout_for(profile)
        if ro is None:
            why = "density metric not Status A/M"
    else:
        why = f"printed on {reader}"
    run_profile = profile
    if ro is not None:
        run_profile = dataclasses.replace(profile, dye_matrix=_fs.IDENTITY3)
    _STATS.clear()
    _CTX.clear()
    _CTX["settings"], _CTX["spectral_in"] = settings, spectral_in
    t0 = time.time()
    saved = (_fs.callier_density, _fs.spectral_upsampling_lut, _su.apply_lut,
             _fs.solve_anchors)
    _fs.callier_density = _callier_hook
    _fs.spectral_upsampling_lut = _upsampling_lut_hook
    _su.apply_lut = _apply_lut_hook
    _fs.solve_anchors = _anchors_hook
    _IN_SPECTRAL, _ACTIVE_READOUT = True, ro
    try:
        out = _ORIG_SIMULATE(linear_rgb, run_profile, settings)
    finally:
        _IN_SPECTRAL, _ACTIVE_READOUT = False, None
        (_fs.callier_density, _fs.spectral_upsampling_lut, _su.apply_lut,
         _fs.solve_anchors) = saved
        _CTX.clear()
    st = {k: v for k, v in _STATS.items() if not k.startswith("_")}
    entry = dict(input="spectral" if spectral_in else "authored taking matrix",
                 readout="spectral" if ro is not None else f"dye_matrix ({why})",
                 seconds=round(time.time() - t0, 1), **st)
    _REPORT[profile.name] = entry
    msg = f"     spectral: input {'ON' if spectral_in else 'off'}"
    if "input_fit_max_dE76" in st:
        msg += (f" ({st['input_colours']} colours, fit dE76 <= "
                f"{st['input_fit_max_dE76']:.2g}, film_sim table differs <= "
                f"{100 * st.get('input_vs_table_max_rel', 0):.2f} %)")
    if ro is not None:
        msg += (f"; read-out ON (mean |dD| {st['readout_change_mean_D']:.4f}, "
                f"max {st['readout_change_max_D']:.3f}; "
                f"{100 * st['readout_negative_dye_px'] / st['readout_px']:.1f} % px "
                f"need a negative dye amount, {st['readout_unreached_px']} px unsolved"
                f"{'; D-min trace used' if ro.has_dmin else ''})")
    else:
        msg += f"; read-out off ({why})"
    print(msg + f"  [{entry['seconds']} s]", flush=True)
    return out


def build_parser():
    p = _ORIG_BUILD_PARSER()
    p.description = ((p.description or "") +
                     "  [film_sim2: exact spectral taking + spectral read-out]")
    p.set_defaults(outdir=DEFAULT_OUTDIR)
    return p


def spectral_report() -> dict:
    """{stock: what ran, accuracy figures, seconds} for this process."""
    return dict(_REPORT)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--emit-cpp" in argv:
        print("[ERROR] film_sim2.py does not write C++ tables (Python-only "
              "experiment); use film_sim.py --emit-cpp", file=sys.stderr)
        return 2
    _fs.build_parser, _fs.simulate = build_parser, simulate
    try:
        rc = _fs.main(argv)
    finally:
        _fs.build_parser, _fs.simulate = _ORIG_BUILD_PARSER, _ORIG_SIMULATE
    if _REPORT:
        n_in = sum(1 for e in _REPORT.values() if e["input"] == "spectral")
        n_out = sum(1 for e in _REPORT.values() if e["readout"] == "spectral")
        print(f"[INFO] film_sim2: {len(_REPORT)} stock(s); spectral input on "
              f"{n_in}, spectral read-out on {n_out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
