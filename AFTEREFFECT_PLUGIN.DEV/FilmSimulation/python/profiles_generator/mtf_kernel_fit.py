#!/usr/bin/env python3
"""mtf_kernel_fit.py -- three-lobe separable fits of the measured MTF law
(2026-10-06, owner decision G5: "better C++ kernel fit").

The measured roll-off law is 1 / (1 + (f/f50)^q). ⚠ SINCE 2026-10-06 ALL THREE
IMPLEMENTATIONS RENDER IT EXACTLY in the frequency domain (the C++ engines on the
owner's FFT library), so the tables this module fits (film_profiles
`_MTF_KERNEL_TABLE` two-lobe, `_MTF_KERNEL_TABLE3` three-lobe) are no longer
emitted to C++ and serve only film_sim's mtf_use_kernel diagnostic. Until that
date the C++ engines, having no FFT, convolved these separable Gaussian lobes. Until 2026-10-06 the two-lobe table served every stock but Technical Pan
and left a worst modulation error of 0.043 (median 0.021) against the law.

This module fits the three-lobe family

    w1 G(x s1) + w2 G(x s2) + (1 - w1 - w2) G(x s3),   G(u) = exp(-ln2 u^2),
    x = f / f50,

minimax on the guard grid x in logspace(-1.3, 0.9, 600) -- the same grid
verify.py checks -- with every lobe weight inside [-WMAX, +WMAX] so the sum
stays well conditioned in float (no large cancelling pair). It is run once,
by hand, when a new exponent is adopted; its output is pasted into
film_profiles._MTF_KERNEL_TABLE3 with the error in the comment, exactly as the
two-lobe rows were. `--check` re-measures the stored rows (no fitting) and is
what the build audit runs.

Routing rule (owner G5): a q is served by THREE lobes when the three-lobe fit
beats the stored two-lobe row by at least ROUTE_MARGIN; otherwise the two-lobe
row stays. A q lives in exactly one table, so the lookup (`mtf_kernel` then
`mtf_kernel3`) needs no order rule.

Usage:
    python3 mtf_kernel_fit.py --fit [q ...]      fit and print rows (all table q by default)
    python3 mtf_kernel_fit.py --check            re-measure the stored tables, assert the rule
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GRID = np.logspace(-1.3, 0.9, 600)
L2 = float(np.log(2.0))
WMAX = 2.0            # |each lobe weight| <= WMAX
ROUTE_MARGIN = 0.002  # three lobes must beat the two-lobe row by at least this
TARGET = 0.010        # owner target for the worst modulation error
#: exponents whose three-lobe best is known to stay above TARGET (a fourth lobe
#: would be needed); reported, allowed up to LOW_Q_CEIL -- or, for the one
#: exponent far below the rest, up to its own recorded ceiling.
LOW_Q_CEIL = 0.016
KNOWN_CEILING = {1.0710: 0.0270}   # KODAK_TECHNICAL_PAN, row of 2026-09 (0.0267)


def law(q: float) -> np.ndarray:
    return 1.0 / (1.0 + GRID ** q)


def two(row) -> np.ndarray:
    w1, s1, s2 = row
    return w1 * np.exp(-L2 * (GRID * s1) ** 2) + (1.0 - w1) * np.exp(-L2 * (GRID * s2) ** 2)


def three(row) -> np.ndarray:
    w1, w2, s1, s2, s3 = row
    return (w1 * np.exp(-L2 * (GRID * s1) ** 2) + w2 * np.exp(-L2 * (GRID * s2) ** 2)
            + (1.0 - w1 - w2) * np.exp(-L2 * (GRID * s3) ** 2))


def err2(q, row) -> float:
    return float(np.max(np.abs(two(row) - law(q))))


def err3(q, row) -> float:
    return float(np.max(np.abs(three(row) - law(q))))


def weights_ok(row) -> bool:
    w1, w2 = row[0], row[1]
    return all(abs(w) <= WMAX + 1e-9 for w in (w1, w2, 1.0 - w1 - w2))


def fit3(q: float):
    """Best (error, row) of the three-lobe family for exponent q, |w| <= WMAX.
    Deterministic: a fixed grid of starts, least squares then a minimax polish."""
    from scipy.optimize import least_squares, minimize
    t = law(q)
    best = None

    def penalty(p):
        return sum(max(0.0, abs(w) - WMAX) ** 2 for w in (p[0], p[1], 1.0 - p[0] - p[1])) * 1e3

    starts = itertools.product((0.3, 0.6, 0.9), (1.0, 1.3, 1.6), (1.8, 2.6, 4.0, 6.0),
                               (-0.5, 0.2, 0.8, 1.4), (-1.0, -0.3, 0.4, 1.0))
    for s1, s2, s3, w1, w2 in starts:
        r = least_squares(lambda p: three(p) - t, [w1, w2, s1, s2, s3],
                          bounds=([-WMAX, -WMAX, 0.01, 0.01, 0.01], [WMAX, WMAX, 40, 40, 40]),
                          max_nfev=2000)
        if not weights_ok(r.x):
            continue
        e = float(np.max(np.abs(three(r.x) - t)))
        if best is None or e < best[0]:
            best = (e, r.x.copy())
    m = minimize(lambda p: float(np.max(np.abs(three(p) - t))) + penalty(p), best[1],
                 method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-10})
    if weights_ok(m.x):
        e = float(np.max(np.abs(three(m.x) - t)))
        if e < best[0]:
            best = (e, m.x.copy())
    # canonical lobe order: ascending sigma (s1 < s2 < s3), w3 implied
    w = [best[1][0], best[1][1], 1.0 - best[1][0] - best[1][1]]
    s = list(best[1][2:5])
    order = sorted(range(3), key=lambda i: s[i])
    w = [w[i] for i in order]; s = [s[i] for i in order]
    row = (round(w[0], 6), round(w[1], 6), round(s[0], 6), round(s[1], 6), round(s[2], 6))
    return err3(q, row), row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="*", type=float, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--legacy2", default=None, help="python file holding the PRE-refit two-lobe table (for --fit)")
    a = ap.parse_args(argv)
    import film_profiles as fp
    t2, t3 = fp._MTF_KERNEL_TABLE, fp._MTF_KERNEL_TABLE3

    if a.fit is not None:
        legacy = dict(t2)
        if a.legacy2:
            ns = {}
            exec(Path(a.legacy2).read_text(), ns)
            legacy.update(ns["TABLE2"])
        qs = a.fit or sorted(set(legacy) | set(t3))
        for q in qs:
            e3, row = fit3(q)
            e2 = err2(q, legacy[q]) if q in legacy else float("nan")
            route = "3" if (not (e2 == e2)) or e3 <= e2 - ROUTE_MARGIN else "2"
            print("    %.4f: (%+.6f, %+.6f, %.6f, %.6f, %.6f),   # max|err| %.4f  vs two-lobe %.4f  route %s"
                  % (q, row[0], row[1], row[2], row[3], row[4], e3, e2, route))
        return 0

    # --check: the stored tables against the rule
    bad, worst2, worst3, low = [], 0.0, 0.0, []
    stock_q = sorted({round(float(p.mtf.mtf_rolloff_q), 4) for p in fp.FILM_PROFILES
                      if p.mtf.mtf_measured and p.mtf.mtf_rolloff_q > 0.0})
    for q in stock_q:
        if q in t2 and q in t3:
            bad.append("q %.4f is in both tables" % q)
        elif q in t3:
            e = err3(q, t3[q]); worst3 = max(worst3, e)
            if not weights_ok(t3[q]):
                bad.append("q %.4f: a lobe weight exceeds +-%.1f" % (q, WMAX))
            if e > TARGET:
                low.append((q, e))
                ceil = KNOWN_CEILING.get(q, LOW_Q_CEIL)
                if e > ceil:
                    bad.append("q %.4f: three-lobe error %.4f over %.4f" % (q, e, ceil))
        elif q in t2:
            e = err2(q, t2[q]); worst2 = max(worst2, e)
            if e > TARGET + 0.0006:
                bad.append("q %.4f: served by two lobes at %.4f, above the %.3f target -- "
                           "fit three (mtf_kernel_fit.py --fit %.4f)" % (q, e, TARGET, q))
        else:
            bad.append("q %.4f is stored on a stock and has no kernel row" % q)
    for msg in bad:
        print("[FAIL] " + msg)
    if bad:
        return 1
    print("[OK] mtf kernels: %d stock exponents; two-lobe worst %.4f, three-lobe worst %.4f; "
          "above the %.3f target (need a fourth lobe): %s"
          % (len(stock_q), worst2, worst3, TARGET,
             ", ".join("q %.2f %.4f" % (q, e) for q, e in low) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
