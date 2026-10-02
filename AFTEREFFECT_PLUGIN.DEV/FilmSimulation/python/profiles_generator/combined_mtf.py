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

The fit uses only the roll-off (0.25 <= response <= 0.90 after the adjacency
hump -- the band f50 lives in; the deep tail is fatter than the Gaussian law
these stocks render with, and fitting it would drag f50 down),
because the engine applies development adjacency as a separate band-pass
lift and the stored f50 must not absorb it. ⚠ UNCHANGED ON STOCKS WHOSE f50 IS
MEASURED: a combined curve never overwrites a per-layer measurement.

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
        curve_ok = (tuple(m.combined_freqs) == tuple(float(f) for f, _ in COMBINED[name][2])
                    and m.combined_source)
        if name in fp.VITALE_2009_ADOPTED:
            # kept: a published MTF point outranks this solve; corroboration
            dg = g / m.f50_g - 1.0
            ok = curve_ok and abs(dg) < 0.25
            bad += not ok
            print("%s %-28s KEPT (Vitale 2009) %.2f / %.2f / %.2f; this curve "
                  "solves to %.2f / %.2f / %.2f -- green %+.0f %%"
                  % ("[OK  ]" if ok else "[FAIL]", name, m.f50_r, m.f50_g,
                     m.f50_b, r, g, b, 100 * dg))
            continue
        drift = max(abs(m.f50_r - r), abs(m.f50_g - g), abs(m.f50_b - b))
        ok = drift < 0.011 and curve_ok and not m.mtf_measured and r < g < b and rms < 0.05
        bad += not ok
        print("%s %-28s f50 %.2f / %.2f / %.2f  fit rms %.4f log over %d pts%s"
              % ("[OK  ]" if ok else "[FAIL]", name, r, g, b, rms, n,
                 "" if ok else "  stored %.2f / %.2f / %.2f" % (m.f50_r, m.f50_g, m.f50_b)))
    print("[%s] %d combined-curve stocks, per-layer f50 reproduce their curve"
          % ("OK" if not bad else "FAIL", len(got)))
    return 1 if (bad and a.do_assert) else 0


if __name__ == "__main__":
    raise SystemExit(main())
