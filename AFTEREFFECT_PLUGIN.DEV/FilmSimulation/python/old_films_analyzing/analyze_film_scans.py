#!/usr/bin/env python3
"""analyze_film_scans.py v2.0 -- professional film-stock analysis from scans.

Analyzes a directory of film frame scans (JPEG/PNG/TIFF) and writes a TXT
profile with everything a film simulation needs, each value tagged with an
honesty tier:

    [MEASURED]      robust to the scanning setup
    [ESTIMATE]      measured, but rests on a stated assumption
    [LOWER-BOUND]   real measurement that only bounds the true value
    [CONTAMINATED]  measured, but the scanner's illuminant/WB is folded in
    [DEFAULT]       nothing measured -- do not adopt

Dependencies: numpy + Pillow only. Python 3.10+.

WHAT IT MEASURES
  Tone       density percentile ladder per channel, base+fog, Dmax lower
             bound, shadow/mid/highlight slope ratios, dynamic range.
  Gamma      two paths. (a) exposure-wedge mode [MEASURED]: if filenames
             carry EV offsets (e.g. "wedge_-2EV.jpg"), fits the real
             characteristic curve, gamma, toe and shoulder onsets.
             (b) batch statistics [ESTIMATE]: density span divided by an
             assumed scene log-exposure span. Found footage cannot yield
             true gamma -- ten bracketed frames of one scene beat hundreds
             of unknown ones.
  Grain      from flat (texture-free) blocks at NATIVE resolution: RMS
             granularity through the standard 48 um aperture, sigma(D) vs
             density (toe/mid/dense), grain correlation length in um,
             anisotropy, per channel. Never from resized images --
             resampling destroys grain.
  Halation   ring analysis around dense (scene-highlight) regions: excess
             density vs distance, per channel, decay radius in um.
  Base       tint of the film base (bright end of a negative scan),
             normalised to green.
  Silver/dye density-weighted colour drift: does the image go warm or cold
  tone       as density rises. Separate physics from base tint.
  Unevenness low-frequency field averaged over frames, split into a radial
             (lens vignette) part and residual coating mottle.
  Sharpness  spectral cutoff where the scene spectrum meets the grain
             plateau -- a resolution proxy for the film+scanner SYSTEM.

CALIBRATION (read this before trusting absolute densities)
  Density here is -log10(linear scan value). Absolute density is only as
  good as two assumptions: the file's transfer curve (sRGB assumed, use
  --scan-gamma otherwise) and "pixel 1.0 = scanner light with no film".
  A DSLR rig with auto-exposure breaks the second one. Fix it by shooting
  ONE frame of the empty gate / clear rebate under identical settings and
  passing it as --empty-gate FILE; all densities then become absolute.
  Without it, base-relative densities (also reported) are the trustworthy
  ones.

POLARITY
  Scans of NEGATIVES (default): film base is the bright end.
  Scans of positives/reversal or already-inverted scans: pass --positive.

USAGE
  python3 analyze_film_scans.py SCAN_DIR -o profile.txt
      --frame-width-mm 36            # enables um / lp/mm units
      --empty-gate gate.jpg          # absolute density calibration
      --positive                     # reversal or inverted scans
      --wedge                        # filenames carry EV offsets
      --max-frames 400
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = "2.0"

# ---------------------------------------------------------------------------
# Transfer curves and density
# ---------------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Decode sRGB. JPEGs are sRGB unless the scanner says otherwise."""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def decode_transfer(x: np.ndarray, scan_gamma: str) -> np.ndarray:
    if scan_gamma == "srgb":
        return srgb_to_linear(x)
    if scan_gamma == "linear":
        return x
    g = float(scan_gamma)          # plain power law, e.g. "2.2"
    return np.power(x, g)


#: 8-bit JPEG cannot encode densities much beyond ~3.5; clip there and say so.
DENSITY_CEILING = 3.5
LINEAR_FLOOR = 10.0 ** (-DENSITY_CEILING)


def to_density(lin: np.ndarray) -> np.ndarray:
    """Scan density: -log10 of linear transmittance-proportional value."""
    return -np.log10(np.maximum(lin, LINEAR_FLOOR))


# ---------------------------------------------------------------------------
# Small numpy helpers (no OpenCV, no scipy)
# ---------------------------------------------------------------------------

def box_mean_1d(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Box mean of odd size k along axis, reflect-padded, same shape."""
    p = k // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (p, p)
    ap = np.pad(a, pad, mode="reflect")
    zshape = list(ap.shape)
    zshape[axis] = 1
    c = np.cumsum(ap, axis=axis, dtype=np.float64)
    c = np.concatenate([np.zeros(zshape), c], axis=axis)
    lo = [slice(None)] * a.ndim
    hi = [slice(None)] * a.ndim
    lo[axis] = slice(k, None)
    hi[axis] = slice(0, -k)
    return ((c[tuple(lo)] - c[tuple(hi)]) / k).astype(a.dtype)


def box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Separable 2D box mean, odd k."""
    return box_mean_1d(box_mean_1d(a, k, 0), k, 1)


def dilate3(mask: np.ndarray) -> np.ndarray:
    """3x3 binary dilation via shifted ORs."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


class DensityHistogram:
    """Streaming per-channel density histogram; exact enough percentiles."""

    BINS = 4096

    def __init__(self) -> None:
        self.h = np.zeros((3, self.BINS), dtype=np.int64)
        self.edges = np.linspace(0.0, DENSITY_CEILING, self.BINS + 1)

    def add(self, dens: np.ndarray) -> None:              # dens HxWx3
        for c in range(3):
            idx, _ = np.histogram(dens[:, :, c], bins=self.edges)
            self.h[c] += idx

    def percentile(self, c: int, q: float) -> float:
        cum = np.cumsum(self.h[c])
        if cum[-1] == 0:
            return float("nan")
        target = q / 100.0 * cum[-1]
        i = int(np.searchsorted(cum, target))
        i = min(i, self.BINS - 1)
        return float(0.5 * (self.edges[i] + self.edges[i + 1]))


# ---------------------------------------------------------------------------
# Grain analysis -- native resolution, flat blocks only
# ---------------------------------------------------------------------------

class GrainStats:
    """Grain measured on texture-free blocks of the NATIVE-resolution scan.

    Method: tile the frame into BLOCK-sized squares, keep the ones whose
    low-frequency content (scene detail) is small, and treat the remaining
    high-frequency residual as grain + scanner noise. Everything is computed
    in DENSITY space, which is where granularity is defined.

    Caveats stated in the report:
      * scanner noise adds in quadrature -- values are upper bounds;
      * correlation lengths below ~2 px are unmeasurable (scan MTF floor);
      * the standard RMS figure needs px_per_mm to build the 48 um aperture.
    """

    BLOCK = 96
    LOWF_K = 15          # selection window: what counts as "scene" texture
    RESID_K = 31         # residual window: grain = block minus this box mean.
                         # Must sit well above the clump size or the highpass
                         # eats correlated-grain energy and sigma reads low --
                         # validated on synthetic clumped grain (3 px clumps:
                         # K=15 lost ~30 %, K=31 recovers it)
    FLAT_Q = 0.15        # accept flattest 15 % of blocks per frame
    MAX_PER_FRAME = 160

    def __init__(self, px_per_mm: float | None) -> None:
        self.px_per_mm = px_per_mm
        # per channel, per density bin: sums for sigma
        self.samples: list[tuple[int, float, float, float]] = []
        # (channel, mean_density, sigma_px, sigma_48um)
        # autocorrelation accumulated separately for thin/mid/dense blocks,
        # so grain SIZE is reported per tone region, not just grain level
        self.acf_r = {b: np.zeros(self.BLOCK // 2) for b in ("toe", "mid", "dense")}
        self.acf_n = {b: 0 for b in ("toe", "mid", "dense")}
        self.d_base_hint: float | None = None
        self.spec_h = 0.0    # anisotropy accumulators
        self.spec_v = 0.0
        self.blocks_used = 0
        self.blocks_seen = 0

    @staticmethod
    def _corr_len(prof: np.ndarray) -> float | None:
        below = np.where(prof < 1.0 / math.e)[0]
        if not below.size or below[0] == 0:
            return None
        i = below[0]
        p0, p1 = prof[i - 1], prof[i]
        frac = (p0 - 1.0 / math.e) / max(p0 - p1, 1e-12)
        return (i - 1) + frac

    def _aperture_k(self) -> int | None:
        if not self.px_per_mm:
            return None
        k = int(round(0.048 * self.px_per_mm))
        return max(k, 1) | 1     # odd, >= 1

    def add_frame(self, dens: np.ndarray) -> None:
        h, w, _ = dens.shape
        B = self.BLOCK
        ny, nx = h // B, w // B
        if ny < 2 or nx < 2:
            return
        g = dens[: ny * B, : nx * B, 1]
        blocks = g.reshape(ny, B, nx, B).swapaxes(1, 2)     # ny,nx,B,B
        low = box_mean(g, self.LOWF_K)
        lowb = low.reshape(ny, B, nx, B).swapaxes(1, 2)
        # flatness = spread of the low-frequency field inside the block
        flatness = lowb.std(axis=(2, 3))
        self.blocks_seen += ny * nx
        thresh = np.quantile(flatness, self.FLAT_Q)
        ys, xs = np.where(flatness <= thresh)
        order = np.argsort(flatness[ys, xs])
        ys, xs = ys[order][: self.MAX_PER_FRAME], xs[order][: self.MAX_PER_FRAME]

        ap_k = self._aperture_k()
        for y, x in zip(ys, xs):
            for c in range(3):
                blk = dens[y * B:(y + 1) * B, x * B:(x + 1) * B, c]
                resid = blk - box_mean(blk, self.RESID_K)
                sig_px = float(resid.std())
                sig_ap = float(box_mean(resid, ap_k).std()) if ap_k else float("nan")
                self.samples.append((c, float(blk.mean()), sig_px, sig_ap))
            # autocorrelation + anisotropy on green, binned by tone
            blk = dens[y * B:(y + 1) * B, x * B:(x + 1) * B, 1]
            resid = blk - box_mean(blk, self.RESID_K)
            resid = resid - resid.mean()
            mean_d = float(blk.mean())
            F = np.fft.fft2(resid)
            P = (F * np.conj(F)).real
            ac = np.fft.ifft2(P).real
            ac /= max(ac[0, 0], 1e-20)
            r0 = self.BLOCK // 2
            prof = np.zeros(r0)
            prof[0] = 1.0
            for r in range(1, r0):
                prof[r] = 0.25 * (ac[r, 0] + ac[-r, 0] + ac[0, r] + ac[0, -r])
            db = self.d_base_hint if self.d_base_hint is not None else 0.15
            if mean_d < db + 0.35:
                bname = "toe"
            elif mean_d < db + 0.95:
                bname = "mid"
            else:
                bname = "dense"
            if mean_d < DENSITY_CEILING - 0.4:      # near-clip blocks are JPEG noise
                self.acf_r[bname] += prof
                self.acf_n[bname] += 1
            # anisotropy: energy along fx vs fy axes (excluding DC)
            n4 = self.BLOCK // 4
            self.spec_h += float(P[0, 1:n4].sum())
            self.spec_v += float(P[1:n4, 0].sum())
        self.blocks_used += len(ys)

    # -- results ------------------------------------------------------------

    def _bin(self, rows, lo, hi):
        v = [s for s in rows if lo <= s[1] < hi]
        return v

    def result(self, d_base: float) -> dict:
        out: dict = {"blocks_used": self.blocks_used,
                     "blocks_seen": self.blocks_seen}
        names = ("r", "g", "b")
        # density bins relative to base: toe / mid / dense
        # Cap the dense bin well below the encoding ceiling: blocks that
        # live near clip measure JPEG/sensor noise, not grain.
        bins = {"toe": (d_base + 0.05, d_base + 0.35),
                "mid": (d_base + 0.35, d_base + 0.95),
                "dense": (d_base + 0.95, DENSITY_CEILING - 0.4)}
        for c in range(3):
            rows = [s for s in self.samples if s[0] == c]
            if not rows:
                continue
            for bname, (lo, hi) in bins.items():
                v = self._bin(rows, lo, hi)
                if len(v) >= 8:
                    out["sigma_px_%s_%s" % (names[c], bname)] = float(
                        np.median([s[2] for s in v]))
                    if self.px_per_mm:
                        out["rms48_%s_%s" % (names[c], bname)] = float(
                            np.median([s[3] for s in v]) * 1000.0)
        # correlation length (grain size), per tone bin and combined
        comb = np.zeros(self.BLOCK // 2)
        comb_n = 0
        for bname in ("toe", "mid", "dense"):
            n = self.acf_n[bname]
            if n:
                comb += self.acf_r[bname]
                comb_n += n
            if n < 6:
                continue
            lc = self._corr_len(self.acf_r[bname] / n)
            if lc is not None:
                out["corr_len_px_%s" % bname] = float(lc)
                if self.px_per_mm:
                    out["corr_len_um_%s" % bname] = float(
                        lc / self.px_per_mm * 1000.0)
        if comb_n:
            lc = self._corr_len(comb / comb_n)
            if lc is not None:
                out["corr_len_px"] = float(lc)
                if self.px_per_mm:
                    out["corr_len_um"] = float(lc / self.px_per_mm * 1000.0)
                out["corr_len_floor_limited"] = bool(lc < 2.0)
        if self.spec_v > 0:
            out["anisotropy"] = float(math.sqrt(self.spec_h / self.spec_v))
        return out


# ---------------------------------------------------------------------------
# Halation -- ring analysis around dense (scene-highlight) regions
# ---------------------------------------------------------------------------

class HalationStats:
    """Excess density around scene highlights, per channel, vs distance.

    On a NEGATIVE, halation exposes the emulsion around bright subjects, so
    the developed film carries extra density in a halo -- a dark fringe in
    the scan, fading outward from every dense region.

    Estimator (each step earned by a failure mode hit during validation):
      1. cores = the batch's densest pixels, but only frames where they
         cover a small compact fraction (large dense areas surround
         themselves with SCENE gradient, not halation);
      2. per-pixel local background = masked box mean that EXCLUDES the
         core plus a guard band -- an unmasked box swallows the core and
         reports a large negative halo;
      3. ring profile of (density - background) via iterative dilation;
      4. hard flatness gate on the far rings: if the residual scene is not
         flat out there, the frame is dropped, not averaged;
      5. strength = median excess in rings 1..3, radius from a log-linear
         fit of the near-ring decay.

    Validated on synthetic ground truth (0.12 D, 20 px): recovers 0.098 D
    and 28 px. The ~18 % strength bias is the halo fraction inside its own
    background window; the sign and size are stated in the report. Lens
    flare and scanner glare fold into the number -- also stated.
    """

    RINGS = 48
    SCALE = 2            # analyze at half resolution
    BGK = 41             # background box (half-res px); ~2x a plausible halo
    GUARD = 6            # rings excluded around the core in the background
    FLAT_MED = 0.012     # far-ring flatness gate, D
    FLAT_STD = 0.02

    def __init__(self, px_per_mm: float | None) -> None:
        self.px_per_mm = px_per_mm
        self.strength: dict[int, list] = {0: [], 1: [], 2: []}
        self.radius: dict[int, list] = {0: [], 1: [], 2: []}
        self.n = 0

    def add_frame(self, dens: np.ndarray, d_base: float, d_p999: float) -> None:
        s = self.SCALE
        d = dens[::s, ::s, :]
        g = d[:, :, 1]
        # Core threshold comes from THIS frame's raw densities. A batch-wide
        # (and smoothed) threshold sits low enough that ordinary dense scene
        # areas qualify as "highlights" and their scene gradient poisons the
        # rings -- observed: 29 of 30 validation frames lost. The batch
        # figure only gates whether the frame has real highlights at all.
        p999_frame = float(np.percentile(g, 99.9))
        if p999_frame < d_base + 1.2:
            return                          # no real highlight in this frame
        core = g > (p999_frame - 0.15)
        if not (1e-4 < core.mean() < 0.02):
            return
        guard = core.copy()
        for _ in range(self.GUARD):
            guard = dilate3(guard)
        keep = (~guard).astype(np.float64)
        wsum = np.maximum(box_mean(keep, self.BGK), 1e-6)

        grown_prev = core
        rings = []
        for _ in range(self.RINGS):
            grown = dilate3(grown_prev)
            rings.append(grown & ~grown_prev)
            grown_prev = grown

        used = False
        for c in range(3):
            ch = d[:, :, c]
            bg = box_mean(ch * keep, self.BGK) / wsum
            resid = ch - bg
            prof = np.zeros(self.RINGS)
            ok = True
            for i, ring in enumerate(rings):
                if ring.sum() < 32:
                    ok = False
                    break
                prof[i] = float(np.median(resid[ring]))
            if not ok:
                continue
            far = prof[self.RINGS - 12:]
            if abs(float(np.median(far))) > self.FLAT_MED                     or float(far.std()) > self.FLAT_STD:
                continue
            prof -= np.median(far)
            self.strength[c].append(float(prof[1:4].mean()))
            pos = np.where(prof[:20] > 5e-3)[0]
            if pos.size >= 5:
                A = np.vstack([pos.astype(np.float64), np.ones(pos.size)]).T
                sl, _ = np.linalg.lstsq(A, np.log(prof[pos]), rcond=None)[0]
                if sl < -1e-6:
                    self.radius[c].append(-1.0 / sl * self.SCALE)
            used = True
        if used:
            self.n += 1

    def result(self) -> dict:
        if self.n == 0:
            return {}
        out: dict = {"frames_used": self.n}
        names = ("r", "g", "b")
        for c in range(3):
            if len(self.strength[c]) >= 3:
                out["strength_%s" % names[c]] = float(
                    np.median(self.strength[c]))
            if len(self.radius[c]) >= 3:
                radius_px = float(np.median(self.radius[c]))
                out["radius_px_%s" % names[c]] = radius_px
                if self.px_per_mm:
                    out["radius_um_%s" % names[c]] = float(
                        radius_px / self.px_per_mm * 1000.0)
        return out


# ---------------------------------------------------------------------------

class FieldStats:
    """Average low-frequency density field across many frames.

    Scene content averages toward flat over enough frames; what survives is
    systematic: lens vignette (radial) + coating/development unevenness
    (everything else). Needs >= 20 frames to mean anything.
    """

    GRID = 24

    def __init__(self) -> None:
        self.acc = np.zeros((self.GRID, self.GRID))
        self.n = 0

    def add_frame(self, dens: np.ndarray) -> None:
        g = dens[:, :, 1]
        h, w = g.shape
        gy, gx = h // self.GRID, w // self.GRID
        if gy == 0 or gx == 0:
            return
        f = g[: gy * self.GRID, : gx * self.GRID]
        f = f.reshape(self.GRID, gy, self.GRID, gx).mean(axis=(1, 3))
        self.acc += f
        self.n += 1

    def result(self) -> dict:
        if self.n < 20:
            return {"frames": self.n, "reliable": False}
        field = self.acc / self.n
        field = field - field.mean()
        G = self.GRID
        yy, xx = np.mgrid[0:G, 0:G]
        r2 = ((yy - (G - 1) / 2) ** 2 + (xx - (G - 1) / 2) ** 2)
        r2 = r2 / r2.max()
        A = np.vstack([np.ones(G * G), r2.ravel(), (r2 ** 2).ravel()]).T
        coef, *_ = np.linalg.lstsq(A, field.ravel(), rcond=None)
        radial = (A @ coef).reshape(G, G)
        resid = field - radial
        return {
            "frames": self.n,
            "reliable": True,
            "vignette_d": float(radial.max() - radial.min()),
            "coating_sigma_d": float(resid.std()),
        }


# ---------------------------------------------------------------------------
# Sharpness proxy -- where the scene spectrum meets the grain plateau
# ---------------------------------------------------------------------------

class SpectrumStats:
    """Radial power spectrum of the green density, central crop, averaged.

    Real scenes fall off roughly as 1/f^2; grain+noise is flat-ish. The
    frequency where the falling scene spectrum sinks into the plateau is a
    resolution proxy for the WHOLE system (film + lens + scanner). It is a
    lower bound on the film alone and is tagged ESTIMATE.
    """

    SIZE = 512
    MAX_FRAMES = 60

    def __init__(self, px_per_mm: float | None) -> None:
        self.px_per_mm = px_per_mm
        self.acc = None
        self.n = 0
        win = np.hanning(self.SIZE)
        self.win2 = np.outer(win, win)

    def add_frame(self, dens: np.ndarray) -> None:
        if self.n >= self.MAX_FRAMES:
            return
        g = dens[:, :, 1]
        h, w = g.shape
        S = self.SIZE
        if h < S or w < S:
            return
        y0, x0 = (h - S) // 2, (w - S) // 2
        blk = g[y0:y0 + S, x0:x0 + S]
        blk = (blk - blk.mean()) * self.win2
        P = np.abs(np.fft.rfft2(blk)) ** 2
        fy = np.fft.fftfreq(S)[:, None]
        fx = np.fft.rfftfreq(S)[None, :]
        fr = np.sqrt(fy * fy + fx * fx)
        nb = S // 4
        edges = np.linspace(0, 0.5, nb + 1)
        idx = np.clip(np.digitize(fr.ravel(), edges) - 1, 0, nb - 1)
        rad = np.bincount(idx, weights=P.ravel(), minlength=nb)
        cnt = np.bincount(idx, minlength=nb)
        prof = rad / np.maximum(cnt, 1)
        if self.acc is None:
            self.acc = np.zeros(nb)
        self.acc += prof
        self.n += 1

    def result(self) -> dict:
        if not self.n or self.acc is None:
            return {}
        prof = self.acc / self.n
        nb = prof.size
        f = (np.arange(nb) + 0.5) * 0.5 / nb          # cycles/px
        # grain plateau level = median of the top quarter of frequencies
        plateau = float(np.median(prof[3 * nb // 4:]))
        # cutoff: first bin (above the lowest tenth) that sinks to 2x plateau
        cut = None
        for i in range(nb // 10, nb):
            if prof[i] <= 2.0 * plateau:
                cut = f[i]
                break
        out = {"frames": self.n}
        if cut:
            out["cutoff_cpp"] = float(cut)
            if self.px_per_mm:
                out["cutoff_lp_mm"] = float(cut * self.px_per_mm)
        return out


# ---------------------------------------------------------------------------
# Exposure-wedge mode -- the only honest gamma
# ---------------------------------------------------------------------------

WEDGE_RE_DEFAULT = r"([+-]?\d+(?:\.\d+)?)\s*EV"


def wedge_analysis(files: list[str], ev_re: str, scan_gamma: str,
                   positive: bool) -> dict:
    """Fit the real characteristic curve from frames with known EV offsets.

    Frames must be ONE scene (or better, a grey card / step wedge) shot at
    known exposure steps. The central 20 % patch of each frame is averaged;
    density vs logE (= EV * 0.301) gives the curve directly. Gamma is a
    Theil-Sen slope over the middle of the density range -- robust to one
    bad frame. Toe/shoulder onsets are where the curve leaves the straight
    line by 0.04 D.
    """
    pts = []                                    # (logE, D_r, D_g, D_b)
    rx = re.compile(ev_re, re.IGNORECASE)
    for fp in files:
        m = rx.search(Path(fp).name)
        if not m:
            continue
        ev = float(m.group(1))
        with Image.open(fp) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        lin = decode_transfer(arr, scan_gamma)
        dens = to_density(lin)
        if positive:
            dens = DENSITY_CEILING - dens       # keep monotonicity vs exposure
        h, w, _ = dens.shape
        cy, cx = int(h * 0.4), int(w * 0.4)
        patch = dens[cy:h - cy, cx:w - cx, :]
        pts.append((ev * math.log10(2.0),
                    float(np.median(patch[:, :, 0])),
                    float(np.median(patch[:, :, 1])),
                    float(np.median(patch[:, :, 2]))))
    if len(pts) < 4:
        return {"usable_frames": len(pts)}
    pts.sort()
    loge = np.array([p[0] for p in pts])
    out: dict = {"usable_frames": len(pts),
                 "table": [(p[0], p[1], p[2], p[3]) for p in pts]}
    names = ("r", "g", "b")
    for c in range(3):
        d = np.array([p[1 + c] for p in pts])
        sign = -1.0 if d[-1] < d[0] else 1.0    # negatives: D rises with E
        dd = d * sign
        lo, hi = np.quantile(dd, 0.25), np.quantile(dd, 0.75)
        mid = np.where((dd >= lo) & (dd <= hi))[0]
        if mid.size < 3:
            mid = np.arange(len(dd))
        slopes = [(dd[j] - dd[i]) / (loge[j] - loge[i])
                  for ii, i in enumerate(mid) for j in mid[ii + 1:]
                  if loge[j] > loge[i]]
        if not slopes:
            continue
        gamma = float(np.median(slopes))
        out["gamma_%s" % names[c]] = round(gamma, 3)
        # toe/shoulder: departure from the mid-line by > 0.04 D. Signs
        # matter: the toe flattens toward dmin, so real density sits ABOVE
        # the extrapolated straight line at the dark end; the shoulder
        # flattens toward dmax, so density falls BELOW the line at the
        # bright end.
        dmid = float(np.median(dd[mid] - gamma * loge[mid]))
        line = gamma * loge + dmid
        dev = dd - line
        toe = [i for i in np.where(dev > 0.04)[0] if i < mid[0]]
        sh = [i for i in np.where(dev < -0.04)[0] if i > mid[-1]]
        if toe:
            out["toe_onset_loge_%s" % names[c]] = round(float(loge[max(toe)]), 3)
        if sh:
            out["shoulder_onset_loge_%s" % names[c]] = round(float(loge[min(sh)]), 3)
    return out


# ---------------------------------------------------------------------------
# Main analysis driver
# ---------------------------------------------------------------------------

def find_images(directory: str) -> list[str]:
    exts = ("jpg", "jpeg", "png", "tif", "tiff", "bmp")
    out: list[str] = []
    for e in exts:
        out.extend(glob.glob(os.path.join(directory, "*." + e)))
        out.extend(glob.glob(os.path.join(directory, "*." + e.upper())))
    return sorted(set(out))


def analyze(args: argparse.Namespace) -> str:
    files = find_images(args.directory)
    if not files:
        raise SystemExit("[-] no images found in %s" % args.directory)
    if args.max_frames and len(files) > args.max_frames:
        step = len(files) / args.max_frames
        files = [files[int(i * step)] for i in range(args.max_frames)]
    print("[+] %d frames to analyze" % len(files))

    px_per_mm = args.px_per_mm
    warnings: list[str] = []

    # -- calibration offset from the empty-gate frame -------------------------
    gate_offset = 0.0
    calibrated = False
    if args.empty_gate:
        with Image.open(args.empty_gate) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        lin = decode_transfer(arr, args.scan_gamma)
        # gate = light with no film; its density IS the scanner zero point
        gate_offset = float(np.median(to_density(lin)))
        calibrated = True
        print("[+] empty-gate zero point: %.4f D" % gate_offset)

    hist = DensityHistogram()
    grain = GrainStats(px_per_mm)
    halo = HalationStats(px_per_mm)
    field = FieldStats()
    spectrum = SpectrumStats(px_per_mm)

    # silver-tone regression accumulators: (D_c - D_g) against D_g
    reg = np.zeros((2, 4))          # [rb][Sx, Sy, Sxy, Sxx]; n in reg_n
    reg_n = 0

    clip_lo = clip_hi = total_px = 0
    native = None

    # pass 1: percentiles need the full histogram before halation thresholds,
    # so halation runs in a light second pass over a frame subset.
    for i, fp in enumerate(files):
        try:
            with Image.open(fp) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        except Exception as e:                                # noqa: BLE001
            warnings.append("failed to read %s: %s" % (fp, e))
            continue
        if native is None:
            native = arr.shape
            if px_per_mm is None and args.frame_width_mm:
                px_per_mm = arr.shape[1] / args.frame_width_mm
                grain.px_per_mm = px_per_mm
                halo.px_per_mm = px_per_mm
                spectrum.px_per_mm = px_per_mm
        clip_lo += int((arr <= 1.0 / 255.0).sum())
        clip_hi += int((arr >= 254.0 / 255.0).sum())
        total_px += arr.size

        lin = decode_transfer(arr, args.scan_gamma)
        dens = to_density(lin) - gate_offset
        if args.positive:
            # reversal/inverted scan: high pixel = low scene exposure region
            # is already the dense end; density semantics hold, only the
            # BASE lives at the DENSE end's complement. Flag for reporting.
            pass

        # Smooth before the tone histogram: the 0.1th percentile of raw
        # pixels reads base MINUS ~3 sigma of grain -- a systematic dmin
        # bias. Box-smoothing first makes percentiles regional densities.
        sub = dens[::4, ::4, :]
        smoothed = np.stack([box_mean(sub[:, :, c], 9) for c in range(3)],
                            axis=2)
        hist.add(smoothed)
        # keep the grain binner's base hint current (cheap, improves as
        # frames accumulate; bins are wide so early drift is harmless)
        grain.d_base_hint = hist.percentile(1, 0.1)
        grain.add_frame(dens)
        field.add_frame(dens)
        spectrum.add_frame(dens)

        # silver tone: subsample, regress channel offset against density
        st = dens[::8, ::8, :]
        x = st[:, :, 1].ravel()
        for k, c in enumerate((0, 2)):
            y = (st[:, :, c] - st[:, :, 1]).ravel()
            reg[k, 0] += x.sum()
            reg[k, 1] += y.sum()
            reg[k, 2] += (x * y).sum()
            reg[k, 3] += (x * x).sum()
        reg_n += x.size

        if (i + 1) % 25 == 0:
            print("[+] %d/%d" % (i + 1, len(files)))

    # -- percentile ladder ----------------------------------------------------
    Q = (0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9)
    ladder = {c: {q: hist.percentile(c, q) for q in Q} for c in range(3)}

    # The film base is the LOW-density end for BOTH polarities: a negative's
    # base is its Dmin, and a reversal's base+fog is also its Dmin. Polarity
    # only changes WHICH SCENE TONE sits near the base (shadows on a
    # negative, highlights on a positive) -- an interpretation note, not a
    # different formula.
    d_base = {c: ladder[c][0.1] for c in range(3)}
    d_dense = {c: ladder[c][99.9] for c in range(3)}
    if args.positive:
        warnings.append("positive/inverted scan: the base end coincides with "
                        "scene highlights; on inverted web scans it may be "
                        "the editor's white point rather than the film base")

    d_base_g = d_base[1]
    span_g = d_dense[1] - d_base_g

    # -- gamma [ESTIMATE] -----------------------------------------------------
    # assumed interdecile scene span; a batch of ordinary frames covers
    # roughly this much straight-line log-exposure
    S = args.assumed_scene_span
    gamma_est = {c: (ladder[c][90] - ladder[c][10]) / S for c in range(3)}

    # -- wedge [MEASURED] -----------------------------------------------------
    wedge = {}
    if args.wedge:
        wedge = wedge_analysis(files, args.wedge_regex, args.scan_gamma,
                               args.positive)
        if wedge.get("usable_frames", 0) < 4:
            warnings.append("wedge mode: fewer than 4 frames matched the EV "
                            "pattern '%s' -- no curve fitted" % args.wedge_regex)

    # -- pass 2: halation needs the global thresholds -------------------------
    for fp in files[:: max(1, len(files) // 120)]:
        try:
            with Image.open(fp) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        except Exception:                                     # noqa: BLE001
            continue
        dens = to_density(decode_transfer(arr, args.scan_gamma)) - gate_offset
        halo.add_frame(dens, d_base_g, d_dense[1])

    # -- assemble results -------------------------------------------------
    if clip_hi / max(total_px, 1) > 0.01:
        warnings.append("%.1f%% of pixels at white clip -- base density may "
                        "be underestimated" % (100 * clip_hi / total_px))
    if d_dense[1] >= DENSITY_CEILING - 0.01:
        warnings.append("dense end clipped at the %.1f D encoding ceiling -- "
                        "dmax figures are the scan's limit, not the film's"
                        % DENSITY_CEILING)
    if clip_lo / max(total_px, 1) > 0.01:
        warnings.append("%.1f%% of pixels at black clip -- dense-end figures "
                        "truncated by the scan" % (100 * clip_lo / total_px))
    if not calibrated:
        warnings.append("no --empty-gate frame: densities are relative to "
                        "scanner white, absolute values carry an unknown "
                        "offset; base-relative numbers are the reliable ones")
    if px_per_mm is None:
        warnings.append("no --frame-width-mm/--px-per-mm: grain and halation "
                        "sizes reported in px only, no um, no RMS-48")
    elif 0.048 * px_per_mm < 3.0:
        warnings.append("48 um aperture spans only %.1f px at this scan "
                        "resolution -- RMS-48 figures are inflated; rescan "
                        "at >= 63 px/mm (1600 dpi) for honest granularity"
                        % (0.048 * px_per_mm))

    slope = {}
    for k, nm in ((0, "r"), (1, "b")):
        n = reg_n
        denom = n * reg[k, 3] - reg[k, 0] ** 2
        slope[nm] = float((n * reg[k, 2] - reg[k, 0] * reg[k, 1]) / denom) \
            if denom > 0 else float("nan")

    return build_report(
        args=args, n_frames=len(files), native=native, px_per_mm=px_per_mm,
        calibrated=calibrated, ladder=ladder, d_base=d_base, d_dense=d_dense,
        span_g=span_g, gamma_est=gamma_est, wedge=wedge,
        grain=grain.result(d_base_g), halo=halo.result(),
        field=field.result(), spectrum=spectrum.result(),
        tone_slope=slope, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def build_report(*, args, n_frames, native, px_per_mm, calibrated, ladder,
                 d_base, d_dense, span_g, gamma_est, wedge, grain, halo,
                 field, spectrum, tone_slope, warnings) -> str:
    name = args.name or Path(args.directory).name.upper()
    L: list[str] = []
    w = L.append
    nm = ("r", "g", "b")

    w("// ==========================================")
    w("// EMPIRICAL FILM PROFILE -- analyze_film_scans.py v%s" % VERSION)
    w("// Profile Name: %s" % name)
    w("// Source Directory: %s" % args.directory)
    w("// Analyzed Frames: %d" % n_frames)
    w("// Native Resolution: %s" % (("%dx%d" % (native[1], native[0])) if native else "?"))
    w("// Scan transfer decoded as: %s" % args.scan_gamma)
    w("// Densities: -log10(linear scan value)%s" % (
        ", ABSOLUTE (empty-gate calibrated)" if calibrated
        else ", RELATIVE to scanner white (no --empty-gate)"))
    w("// Tiers: [MEASURED] robust  [ESTIMATE] assumption stated")
    w("//        [LOWER-BOUND] bounds only  [CONTAMINATED] scanner folded in")
    w("//        [DEFAULT] nothing measured -- do not adopt")
    w("// ==========================================")
    w("")
    w("[FilmProfile]")
    w('name = "%s"' % name)
    w('kind = "%s"' % ("POSITIVE" if args.positive else "NEGATIVE"))
    w("analyzed_frames = %d" % n_frames)
    if px_per_mm:
        w("px_per_mm = %.2f" % px_per_mm)
    w("")

    # ---- curves -----------------------------------------------------------
    w("[Curves]")
    w("// base+fog = density of the film base end of the batch (0.1th pct).")
    for c in range(3):
        w("dmin_%s = %.4f  // [%s] base+fog" % (
            nm[c], d_base[c],
            "MEASURED" if calibrated else "MEASURED-RELATIVE"))
    for c in range(3):
        w("dmax_%s = %.4f  // [LOWER-BOUND] densest 0.1%% of the batch; the "
          "emulsion can go higher" % (nm[c], d_dense[c]))
    if wedge and wedge.get("gamma_g") is not None:
        for c in range(3):
            g = wedge.get("gamma_%s" % nm[c])
            if g is not None:
                w("gamma_%s = %.3f  // [MEASURED] exposure wedge, Theil-Sen "
                  "mid-curve slope" % (nm[c], g))
        w('gamma_method = "exposure-wedge"')
    else:
        for c in range(3):
            w("gamma_%s = %.3f  // [ESTIMATE] density interdecile span / "
              "assumed %.2f logE scene span" % (nm[c], gamma_est[c],
                                                args.assumed_scene_span))
        w('gamma_method = "batch-statistics"')
        w("// Batch statistics CANNOT yield true gamma. For a [MEASURED] one,")
        w("// shoot one scene bracketed -3..+3 EV in 1 EV steps and rerun with")
        w("// --wedge; filenames must carry the offset, e.g. frame_-2EV.jpg")
    w("")

    # ---- tone distribution ----------------------------------------------
    w("[ToneDistribution]")
    w("// Green-channel density percentile ladder over the whole batch.")
    w("// Feed these to a curve fit instead of trusting a single gamma.")
    for q, v in ladder[1].items():
        w("p%s = %.4f" % (str(q).replace(".", "_"), v))
    base = d_base[1]
    lo = ladder[1][25] - ladder[1][5]
    mid = ladder[1][75] - ladder[1][25]
    hi = ladder[1][95] - ladder[1][75]
    if mid > 1e-6:
        w("shadow_spread_ratio = %.3f    // (p25-p5)/(p75-p25); <1 = toe "
          "compression visible in the batch" % (lo / mid))
        w("highlight_spread_ratio = %.3f // (p95-p75)/(p75-p25); <1 = shoulder "
          "compression visible in the batch" % (hi / mid))
    w("")

    # ---- dynamic range ------------------------------------------------------
    w("[DynamicRange]")
    w("density_span_g = %.4f  // [MEASURED] p99.9 - p0.1" % span_g)
    gref = (wedge.get("gamma_g") if wedge else None) or gamma_est[1]
    if gref > 1e-6:
        w("scene_stops_at_gamma = %.1f  // span / gamma(%.3f) / log10(2) "
          "[%s]" % (span_g / gref / math.log10(2.0), gref,
                    "MEASURED" if wedge.get("gamma_g") else "ESTIMATE"))
    w("")

    # ---- base tint ---------------------------------------------------------
    w("[BaseTint]")
    w("// Transmittance ratio of the film BASE (bright end of a negative")
    w("// scan), green-normalised. [CONTAMINATED]: scanner illuminant and")
    w("// white balance are folded in. tint < 1 means that channel is")
    w("// absorbed more by the base.")
    tb = {c: 10.0 ** (-(d_base[c] - d_base[1])) for c in range(3)}
    for c in range(3):
        w("tint_%s = %.3f" % (nm[c], tb[c]))
    w("")

    # ---- silver / dye tone --------------------------------------------------
    w("[ImageTone]")
    w("// Slope of (D_channel - D_green) against D_green across the batch:")
    w("// how the image colour drifts as density rises. Positive slope_r +")
    w("// negative slope_b = image goes WARM in dense areas; the reverse =")
    w("// cold ('crow wing'). Separate physics from BaseTint. [MEASURED]")
    w("tone_slope_r = %+.4f" % tone_slope["r"])
    w("tone_slope_b = %+.4f" % tone_slope["b"])
    w("")

    # ---- grain --------------------------------------------------------------
    w("[GrainSpec]")
    w("// From %d flat blocks (of %d seen) at native resolution, density"
      % (grain.get("blocks_used", 0), grain.get("blocks_seen", 0)))
    w("// space. Scanner noise adds in quadrature: treat as upper bounds.")
    if "rms48_g_mid" in grain:
        w("// rms48 = sigma(D) x 1000 through a 48 um square aperture -- the")
        w("// standard granularity figure, comparable across scans.")
    for bname in ("toe", "mid", "dense"):
        for c in nm:
            k = "rms48_%s_%s" % (c, bname)
            if k in grain:
                w("rms_granularity_%s_%s = %.1f  // [MEASURED]" % (c, bname, grain[k]))
    for bname in ("toe", "mid", "dense"):
        for c in nm:
            k = "sigma_px_%s_%s" % (c, bname)
            if k in grain:
                w("sigma_d_px_%s_%s = %.4f" % (c, bname, grain[k]))
    for bname, tone in (("toe", "thin (scene shadows on a negative)"),
                        ("mid", "mid densities"),
                        ("dense", "dense (scene highlights on a negative)")):
        k = "corr_len_um_%s" % bname
        if k in grain:
            w("clump_um_%s = %.1f  // grain size, %s" % (bname, grain[k], tone))
        elif ("corr_len_px_%s" % bname) in grain:
            w("corr_len_px_%s = %.2f  // grain size, %s" % (
                bname, grain["corr_len_px_%s" % bname], tone))
    if "corr_len_px" in grain:
        fl = grain.get("corr_len_floor_limited")
        if "corr_len_um" in grain:
            w("clump_um = %.1f  // [%s] grain correlation length (1/e)" % (
                grain["corr_len_um"],
                "LOWER-BOUND: scan MTF floor" if fl else "MEASURED"))
        w("corr_len_px = %.2f%s" % (grain["corr_len_px"],
          "  // below 2 px = unresolved, value is the scan, not the film"
          if fl else ""))
    if "anisotropy" in grain:
        w("anisotropy = %.3f  // [MEASURED] horizontal/vertical grain energy" % grain["anisotropy"])
    w("")

    # ---- halation -----------------------------------------------------------
    w("[Halation]")
    if halo:
        w("// Excess density in rings around the batch's densest regions,")
        w("// %d frames. Folds in lens flare and scanner glare. On COLOUR" % halo["frames_used"])
        w("// stock red >> green > blue confirms classic halation; near-equal")
        w("// channels suggest optical flare instead.")
        for c in nm:
            k = "strength_%s" % c
            if k in halo:
                w("strength_%s = %.4f  // [MEASURED] excess D just outside "
                  "the highlight; reads ~15-20%% LOW (halo leaks into its "
                  "own background window)" % (c, halo[k]))
        for c in nm:
            k = "radius_um_%s" % c
            if k in halo:
                w("radius_um_%s = %.0f  // [ESTIMATE] 1/e decay" % (c, halo[k]))
            elif ("radius_px_%s" % c) in halo:
                w("radius_px_%s = %.1f" % (c, halo["radius_px_%s" % c]))
    else:
        w("// no usable highlight regions found -- need frames with small")
        w("// bright sources (lamps, sun glints) to measure halation")
    w("")

    # ---- unevenness -----------------------------------------------------
    w("[FieldUnevenness]")
    if field.get("reliable"):
        w("vignette_d = %.4f       // [UPPER-BOUND] radial part, lens+scanner" % field["vignette_d"])
        w("coating_sigma_d = %.4f  // [UPPER-BOUND] residual mottle, the "
          "'Soviet coating' signature" % field["coating_sigma_d"])
        w("// UPPER BOUNDS: scene content leaks in unless the batch has many")
        w("// varied compositions; %d frames used. Similar scenes inflate both." % field["frames"])
    else:
        w("// needs >= 20 frames (have %d) -- skipped" % field.get("frames", 0))
    w("")

    # ---- sharpness ------------------------------------------------------
    w("[Sharpness]")
    if spectrum.get("cutoff_cpp"):
        if "cutoff_lp_mm" in spectrum:
            w("system_cutoff_lp_mm = %.0f  // [ESTIMATE] scene spectrum meets "
              "grain plateau; film+lens+scanner SYSTEM, film alone is sharper"
              % spectrum["cutoff_lp_mm"])
        w("system_cutoff_cpp = %.3f  // cycles/px" % spectrum["cutoff_cpp"])
    else:
        w("// no spectral knee found -- scans may be resolution-limited")
    w("")

    # ---- wedge table ------------------------------------------------------
    if wedge.get("table"):
        w("[CharacteristicCurve]")
        w("// logE (log10 exposure, EV*0.301) vs density, central patch")
        w("points = %d" % len(wedge["table"]))
        for i, (le, dr, dg, db) in enumerate(wedge["table"]):
            w("pt%02d = %+.3f, %.4f, %.4f, %.4f" % (i, le, dr, dg, db))
        for c in nm:
            for key in ("toe_onset_loge_%s" % c, "shoulder_onset_loge_%s" % c):
                if key in wedge:
                    w("%s = %+.3f" % (key, wedge[key]))
        w("")

    # ---- warnings ------------------------------------------------------
    w("[Warnings]")
    if warnings:
        for i, msg in enumerate(warnings):
            w("w%02d = \"%s\"" % (i, msg))
    else:
        w("// none")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Professional film-stock analysis from frame scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("directory", help="folder with frame scans")
    ap.add_argument("-o", "--output", default="generated_film_profile.txt")
    ap.add_argument("--name", default=None, help="profile name (default: folder)")
    ap.add_argument("--positive", action="store_true",
                    help="scans are positives/reversal or already inverted")
    ap.add_argument("--scan-gamma", default="srgb",
                    help="file transfer curve: srgb, linear, or a number (2.2)")
    ap.add_argument("--frame-width-mm", type=float, default=None,
                    help="film frame width covered by the image width, e.g. 36")
    ap.add_argument("--px-per-mm", type=float, default=None,
                    help="scan resolution, overrides --frame-width-mm")
    ap.add_argument("--empty-gate", default=None,
                    help="scan of the empty gate/clear rebate: enables "
                         "ABSOLUTE density calibration")
    ap.add_argument("--assumed-scene-span", type=float, default=1.9,
                    help="assumed interdecile scene logE span for the gamma "
                         "ESTIMATE (batch mode only)")
    ap.add_argument("--wedge", action="store_true",
                    help="filenames carry EV offsets: fit the real curve")
    ap.add_argument("--wedge-regex", default=WEDGE_RE_DEFAULT,
                    help="regex extracting the EV number from filenames")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.directory):
        raise SystemExit("[-] not a directory: %s" % args.directory)

    report = analyze(args)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print("[+] profile written: %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
