#!/usr/bin/env python3
"""Colour films whose MTF is printed as ONE combined curve (2026-10-01d,
owner-approved batch item 4).

«Современные фотоматериалы и их обработка» (2004) prints the Konica, Kodak
Vericolor III, Ektachrome 64 and Kodachrome 64 MTF as a single curve, where
Kodak's C-41 sheets print three. Every one of these stocks carried per-layer
f50 values that were ESTIMATES (an era-and-class heuristic) -- except
Ektachrome 64 and Kodachrome 64, whose f50 come from a PUBLISHED MTF point
(Vitale 2009) and are kept, the book's curve standing as corroboration. This
module:

  1. holds the traced combined curves (column tracking on the panels' own
     log-log ruling, `COMBINED` below), which film_profiles stores in
     `MTFSpec.combined_freqs / combined_response`;
  2. solves the per-layer f50 that REPRODUCES the combined curve under two
     stated constraints -- the combined record is the luminance-weighted sum
     of the three layer MTFs (Rec. 709 weights 0.2126 / 0.7152 / 0.0722, the
     one assumption, named), and the layers keep the corpus's MEASURED
     ordering and ratios: r/g 0.615 and b/g 1.16 for negatives (median of the
     15 colour negatives with three genuinely traced layer curves), r/g 0.868
     and b/g 1.447 for reversal (5 such stocks) -- blue sharpest, red softest,
     as the layer stack requires;
  3. with --assert, re-derives every stored f50 and fails on drift.

The f50-only fit (`solve`) uses only the roll-off band 0.25 <= response <= 0.90
after the adjacency hump, because a Gaussian cannot follow the tail.

2026-10-04: THE SHAPE IS ADOPTED TOO (`solve_q`). The deep tail of every one
of these curves is fatter than the Gaussian law, by log10 rms 0.02-0.21 on
the roll-off points; the measured law 1/(1+(f/f50)^q), which the engines
already rendered for 70 other stocks (through the FilmMtfKernel fit then; exactly,
in the frequency domain, since 2026-10-06), fits the whole
roll-off (peak to last point, response <= 0.90) to 0.008-0.015. So each of
the seven now carries `mtf_rolloff_q` and `mtf_measured = True`, with the
per-layer f50 re-solved JOINTLY with q on the five estimated stocks and HELD
at Vitale's published point on Ektachrome 64 and Kodachrome 64 (q alone is
solved there; their fit is poorer, 0.053 / 0.025, because the book's curve
and Vitale's point disagree slightly, and the published point outranks the
trace). The adjacency hump above 100 % is still not fitted: the engine adds
development adjacency as a separate band-pass. ⚠ UNCHANGED ON STOCKS WHOSE
f50 IS MEASURED: a combined curve never overwrites a per-layer measurement.

Usage:  python combined_mtf.py [--assert]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LUMA = (0.2126, 0.7152, 0.0722)
RATIOS = {"negative": (0.615, 1.0, 1.16), "reversal": (0.868, 1.0, 1.447)}
FIT_MAX = 0.90
FIT_MIN = 0.25

#: stock -> (page, figure, ((cycles/mm, response %), ...)) as traced.
COMBINED = {
    "KONICA_CENTURIA_SUPER_400": (94, "3.56", ((3, 117.6), (4, 121.5), (5, 121.5), (7, 121.5), (10, 117.6), (15, 106.7), (20, 95.3), (25, 85.0), (30, 75.9), (40, 62.5), (50, 53.1))),
    "KONICA_CENTURIA_SUPER_1600": (103, "3.68", ((3, 115.7), (4, 119.5), (5, 119.5), (7, 119.5), (10, 115.7), (15, 105.0), (20, 92.2), (25, 82.3), (30, 73.5), (40, 59.5), (50, 49.0), (60, 43.0))),
    "KONICA_VX_100": (106, "3.72", ((3, 123.7), (4, 129.9), (5, 134.2), (7, 138.6), (10, 138.6), (15, 132.0), (25, 110.3), (30, 99.2), (40, 80.9), (50, 65.9), (60, 51.0), (70, 40.4), (80, 32.1))),
    "KONICA_CHROME_R100": (285, "3.198", ((2, 110.8), (3, 110.8), (4, 110.8), (5, 107.6), (7, 103.0), (10, 93.6), (15, 77.9), (20, 62.5), (25, 50.4), (30, 40.9), (40, 26.7), (50, 18.8), (60, 14.0))),
    "KODAK_VERICOLOR_III_160": (184, "3.74", ((3, 111.4), (4, 113.1), (5, 114.0), (7, 114.0), (10, 113.1), (15, 105.5), (20, 94.0), (25, 85.1), (30, 75.2), (40, 61.1), (50, 51.0), (60, 41.9), (70, 36.2), (80, 32.2))),
    "EKTACHROME_64": (238, "3.142", ((3, 109.0), (4, 108.0), (5, 107.1), (7, 105.3), (10, 101.7), (15, 91.8), (20, 80.0), (25, 67.6), (30, 57.7), (40, 39.2), (50, 27.8))),
    # рис. 3.178 (p267) is the same drawing again (identical trace).
    "KODACHROME_64": (265, "3.174", ((3, 112.2), (4, 116.3), (5, 117.9), (7, 121.4), (10, 123.1), (15, 109.8), (20, 94.1), (25, 80.6), (30, 71.4), (40, 47.8), (50, 32.4), (60, 22.7), (70, 15.7), (80, 11.5))),
}


def model(f, s, ratios):
    return sum(w * math.exp(-math.log(2.0) * (f / (s * r)) ** 2)
               for w, r in zip(LUMA, ratios))


def solve(curve, ratios):
    """(f50_r, f50_g, f50_b, rms of log10 residual, n points)."""
    pts = [(f, p / 100.0) for f, p in curve]
    peak_i = max(range(len(pts)), key=lambda i: pts[i][1])
    use = [(f, m) for f, m in pts[peak_i:] if FIT_MIN <= m <= FIT_MAX]
    lo, hi = 1.0, 500.0
    def err(s):
        return sum((math.log10(model(f, s, ratios)) - math.log10(m)) ** 2
                   for f, m in use)
    for _ in range(200):                       # golden-section on a unimodal err
        a = hi - (hi - lo) / 1.618; b = lo + (hi - lo) / 1.618
        if err(a) < err(b):
            hi = b
        else:
            lo = a
    s = 0.5 * (lo + hi)
    rms = math.sqrt(err(s) / len(use))
    return (round(s * ratios[0], 2), round(s, 2), round(s * ratios[2], 2),
            round(rms, 4), len(use))


def model_q(f, s, q, ratios):
    return sum(w / (1.0 + (f / (s * r)) ** q) for w, r in zip(LUMA, ratios))


def _rms_q(use, s, q, ratios):
    return math.sqrt(sum((math.log10(model_q(f, s, q, ratios)) - math.log10(m)) ** 2
                         for f, m in use) / len(use))


def solve_q(curve, ratios, fixed_s=None):
    """(f50_r, f50_g, f50_b, q, rms of log10 residual, n points).

    Joint (f50_g, q) fit of the luminance-weighted three-layer roll-off law to
    every point from the curve's peak down to its last point with response
    <= FIT_MAX. With `fixed_s` the green f50 is held (Vitale stocks) and only
    q is solved. Deterministic: a coarse grid, then a bounded coordinate
    descent, no random starts.
    """
    pts = [(f, p / 100.0) for f, p in curve]
    peak_i = max(range(len(pts)), key=lambda i: pts[i][1])
    use = [(f, m) for f, m in pts[peak_i:] if m <= FIT_MAX]
    if fixed_s is not None:
        qs = [1.2 + 0.001 * i for i in range(4801)]
        q = min(qs, key=lambda qq: _rms_q(use, fixed_s, qq, ratios))
        s = fixed_s
    else:
        best = (9.0, 0.0, 0.0)
        for i in range(31):
            q0 = 1.5 + 0.1 * i
            for j in range(29):
                s0 = 10.0 + 5.0 * j
                e = _rms_q(use, s0, q0, ratios)
                if e < best[0]:
                    best = (e, s0, q0)
        _e, s, q = best
        ds, dq = 2.5, 0.05
        while ds > 1e-4 or dq > 1e-4:
            moved = False
            for cs, cq in ((ds, 0.0), (-ds, 0.0), (0.0, dq), (0.0, -dq)):
                if s + cs <= 0.0 or q + cq <= 0.0:
                    continue
                e = _rms_q(use, s + cs, q + cq, ratios)
                if e < _rms_q(use, s, q, ratios) - 1e-12:
                    s, q, moved = s + cs, q + cq, True
            if not moved:
                ds *= 0.5
                dq *= 0.5
    s = round(s, 2) if fixed_s is None else fixed_s
    q = round(q, 2)
    return (round(s * ratios[0], 2), round(s, 2), round(s * ratios[2], 2), q,
            round(_rms_q(use, s, q, ratios), 4), len(use))


def derive_all():
    import film_profiles as fp
    out = {}
    for name, (_pg, _fig, curve) in COMBINED.items():
        p = fp.get_profile(name)
        kind = "reversal" if p.is_reversal else "negative"
        out[name] = solve(curve, RATIOS[kind])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    import film_profiles as fp
    got = derive_all()
    bad = 0
    for name, (r, g, b, rms, n) in sorted(got.items()):
        p = fp.get_profile(name)
        m = p.mtf
        kind = "reversal" if p.is_reversal else "negative"
        curve = COMBINED[name][2]
        curve_ok = (tuple(m.combined_freqs) == tuple(float(f) for f, _ in curve)
                    and m.combined_source)
        if name in fp.VITALE_2009_ADOPTED:
            # kept: a published MTF point outranks this solve; corroboration.
            # The SHAPE (q) is solved at the held f50 and must match the store.
            dg = g / m.f50_g - 1.0
            _r, _g, _b, q, rq, nq = solve_q(curve, (m.f50_r / m.f50_g, 1.0, m.f50_b / m.f50_g),
                                            fixed_s=m.f50_g)
            ok = (curve_ok and abs(dg) < 0.25 and m.mtf_measured
                  and abs(m.mtf_rolloff_q - q) < 0.011 and rq < 0.06)
            bad += not ok
            print("%s %-28s KEPT (Vitale 2009) %.2f / %.2f / %.2f; this curve "
                  "solves to %.2f / %.2f / %.2f -- green %+.0f %%; q %.2f (rms %.4f)"
                  % ("[OK  ]" if ok else "[FAIL]", name, m.f50_r, m.f50_g,
                     m.f50_b, r, g, b, 100 * dg, q, rq))
            continue
        rq_, gq, bq, q, rq, nq = solve_q(curve, RATIOS[kind])
        drift = max(abs(m.f50_r - rq_), abs(m.f50_g - gq), abs(m.f50_b - bq))
        ok = (drift < 0.011 and curve_ok and m.mtf_measured
              and abs(m.mtf_rolloff_q - q) < 0.011 and rq_ < gq < bq and rq < 0.02)
        bad += not ok
        print("%s %-28s f50 %.2f / %.2f / %.2f  q %.2f  fit rms %.4f log over %d pts "
              "(Gaussian band fit %.2f / %.2f / %.2f, rms %.4f)%s"
              % ("[OK  ]" if ok else "[FAIL]", name, rq_, gq, bq, q, rq, nq, r, g, b, rms,
                 "" if ok else "  stored %.2f / %.2f / %.2f q %.2f"
                 % (m.f50_r, m.f50_g, m.f50_b, m.mtf_rolloff_q)))
    print("[%s] %d combined-curve stocks, per-layer f50 and roll-off exponent reproduce their curve"
          % ("OK" if not bad else "FAIL", len(got)))
    return 1 if (bad and a.do_assert) else 0


if __name__ == "__main__":
    raise SystemExit(main())
