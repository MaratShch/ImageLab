"""
Pixel-accurate curve extraction from datasheet plot scans.

Purpose: replace eyeball transcription with programmatic tracing, for the
"use every curve from its graphical representation" pass of 2026-08-02.
numpy + Pillow only, same dependency rule as the rest of the project.

Workflow per plot:
  1. Render the PDF page at >= 600 dpi (pdftoppm) and crop the plot.
  2. auto-detect the axes frame (longest dark row/column runs).
  3. Calibrate: caller supplies the data values of the frame edges, or of
     two reference ticks per axis, read from the printed axis labels.
  4. Seeded tracking: caller supplies one seed pixel per curve (from a
     quick visual look); the tracer follows the darkest path column by
     column with a continuity window, immune to text blocks elsewhere.
  5. Output: sampled (x_data, y_data) arrays at every pixel column,
     downsampled on request.
  6. fit_tonecurve(): least-squares fit of the project's 6-parameter
     softplus ToneCurve model to digitised H&D samples (Nelder-Mead,
     hand-rolled -- no SciPy), reporting RMS and max residual so the fit
     quality is auditable and can be quoted in the profile comment.

Accuracy: at 600 dpi a datasheet curve line is 4-8 px wide; the tracer
takes the ink centroid per column, so transcription error is bounded by
about half the line width -- 0.005-0.01 density units on a typical H&D
plot, an order of magnitude better than visual reading. Axis-label
values remain the accuracy floor (they are printed numbers).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = ["load_gray", "find_frame", "trace_curve", "fit_tonecurve",
           "softplus_curve"]


def load_gray(path: str) -> np.ndarray:
    """Image as float grayscale [0..1], 0 = black ink."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def find_frame(img: np.ndarray, dark: float = 0.5) -> tuple[int, int, int, int]:
    """Locate the rectangular plot frame.

    Returns (left, top, right, bottom) pixel coordinates of the inner
    frame lines: the row/column with the most dark pixels in each half.
    """
    d = img < dark
    h, w = d.shape
    col_score = d.sum(axis=0)
    row_score = d.sum(axis=1)
    left = int(np.argmax(col_score[: w // 2]))
    right = int(w // 2 + np.argmax(col_score[w // 2:]))
    top = int(np.argmax(row_score[: h // 2]))
    bottom = int(h // 2 + np.argmax(row_score[h // 2:]))
    return left, top, right, bottom


def trace_curve(
    img: np.ndarray,
    seed_xy: tuple[int, int],
    x_range: tuple[int, int],
    window: int = 14,
    dark: float = 0.5,
    max_gap: int = 40,
) -> dict[int, float]:
    """Track one curve from a seed pixel, both directions, column by column.

    For each pixel column the tracer looks in a +/- `window` band around
    the previous column's centre for dark-ink pixels and takes their
    centroid. Text, gridlines and other curves outside the band cannot
    capture the track. Columns with no ink (dashed lines, small gaps) are
    tolerated up to `max_gap` consecutive misses.

    Returns {x_px: y_px_centroid}.
    """
    h, w = img.shape
    x0, x1 = x_range
    sx, sy = seed_xy
    out: dict[int, float] = {}

    def centroid(x: int, yc: float) -> float | None:
        lo = max(0, int(yc) - window)
        hi = min(h, int(yc) + window + 1)
        col = img[lo:hi, x]
        mask = col < dark
        if not mask.any():
            return None
        ys = np.nonzero(mask)[0]
        weights = 1.0 - col[ys]
        return float(lo + np.average(ys, weights=weights))

    for step in (+1, -1):
        yc = float(sy)
        misses = 0
        x = sx
        while x0 <= x <= x1:
            c = centroid(x, yc)
            if c is None:
                misses += 1
                if misses > max_gap:
                    break
            else:
                misses = 0
                yc = c
                out[x] = c
            x += step
    return out


def softplus_curve(x: np.ndarray, dmin: float, gamma: float, toe_x: float,
                   toe_k: float, shoulder_x: float, shoulder_k: float
                   ) -> np.ndarray:
    """The project's ToneCurve model, vectorised (must match film_sim).

    ⚠⚠ THE SATURATED BRANCH WAS WRONG UNTIL 2026-09-09, AND "must match
    film_sim" IS THE PART IT FAILED. The body read::

        k * np.log1p(np.exp(np.clip(v / k, -60.0, 60.0)))

    which clips the ARGUMENT, so it returns ~60*k once saturated where the true
    softplus returns v. `film_sim._sp_scalar` returns `x if z > 60.0` and
    `ToneCurve._sample` returns `z`; both are right. Only the FITTER was wrong
    -- and the fitter is what adopts curves.

    Measured on KODAK_PORTRA_400NC's blue record (toe_k 0.09, toe_x -2.89): the
    two models part company at x = 2.51 and are 3.34 D apart by x = 8.0,
    because the two terms saturate at 60*toe_k and 60*shoulder_k, which are
    different numbers, so their difference walks away linearly.

    ⚠ NO ADOPTED CURVE WAS EVER AFFECTED -- a traced panel spans 2-5 decades,
    nowhere near 60 softness constants past a knee. It was not harmless either:
    during queue A3 a gamma-constrained fit of 5285 walked into it and returned
    toe_k 0.0263 with dmax 12.99, a "fit" that was reading the clip and not the
    model. Fixed by returning the asymptote on the saturated side, as film_sim
    does. Both branches of `np.where` are evaluated, so the exp argument stays
    capped for overflow safety while the capped branch's RESULT is discarded.
    """
    def sp(v, k):
        z = v / k
        return np.where(z > 60.0, v,
                        k * np.log1p(np.exp(np.minimum(z, 60.0))))
    return dmin + gamma * (sp(x - toe_x, toe_k) - sp(x - shoulder_x, shoulder_k))


def _nelder_mead(f, x0, steps, iters=4000, tol=1e-10):
    """Minimal Nelder-Mead (no SciPy per project dependency rule)."""
    n = len(x0)
    pts = [np.asarray(x0, dtype=np.float64)]
    for i in range(n):
        p = pts[0].copy()
        p[i] += steps[i]
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2.0 * (cen - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    best = int(np.argmin(vals))
    return pts[best], vals[best]


def fit_tonecurve(x: np.ndarray, d: np.ndarray, init: tuple[float, ...]
                  ) -> tuple[tuple[float, ...], float, float]:
    """Fit the 6-parameter ToneCurve to digitised (logE, density) samples.

    `init` is (dmin, gamma, toe_x, toe_k, shoulder_x, shoulder_k), e.g. the
    current hand-fitted profile values. Enforces the project's monotonicity
    rule shoulder_k <= 1.4 * toe_k via penalty. Returns (params, rms, max_abs).
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    def loss(p):
        dmin, gamma, tx, tk, sx, sk = p
        if gamma <= 0 or tk <= 0.02 or sk <= 0.02 or sx <= tx:
            return 1e9
        pen = 0.0
        if sk > 1.4 * tk:
            pen = 100.0 * (sk - 1.4 * tk) ** 2
        r = softplus_curve(x, *p) - d
        return float(np.mean(r * r)) + pen

    steps = [0.02, 0.03, 0.08, 0.04, 0.08, 0.04]
    p, _ = _nelder_mead(loss, np.asarray(init, dtype=np.float64), steps)
    r = softplus_curve(x, *p) - d
    return tuple(float(v) for v in p), float(np.sqrt(np.mean(r * r))), float(
        np.max(np.abs(r)))

def fit_tonecurve4(x, d, init):
    """`fit_tonecurve` with the SHOULDER HELD FIXED at its init value.

    ⚠ WHY A SECOND ENTRY POINT RATHER THAN A FLAG ON THE FIRST. A datasheet that
    stops inside the straight line contains no information about the shoulder,
    and a six-parameter fit to such a trace still returns a shoulder -- one
    placed whichever side of the last sample the simplex drifted to. On the Fuji
    T3 sheets that put red's shoulder at logH 1.16 and green's at 0.27, which
    extrapolates to a Dmax ladder with RED ABOVE GREEN on a film whose own
    curves put blue highest. Fitting four parameters and declaring the other two
    is honest about what the trace can support; silently fitting six is not.

    Returns (params, rms, max_abs) exactly as `fit_tonecurve` does, with
    params[4] and params[5] equal to init[4] and init[5].
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    sx, sk = float(init[4]), float(init[5])

    def loss(p):
        dmin, gamma, tx, tk = p
        if gamma <= 0 or tk <= 0.02 or sx <= tx:
            return 1e9
        pen = 100.0 * (sk - 1.4 * tk) ** 2 if sk > 1.4 * tk else 0.0
        r = softplus_curve(x, dmin, gamma, tx, tk, sx, sk) - d
        return float(np.mean(r * r)) + pen

    steps = [0.02, 0.03, 0.08, 0.04]
    p, _ = _nelder_mead(loss, np.asarray(init[:4], dtype=np.float64), steps)
    full = (float(p[0]), float(p[1]), float(p[2]), float(p[3]), sx, sk)
    r = softplus_curve(x, *full) - d
    return full, float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))

