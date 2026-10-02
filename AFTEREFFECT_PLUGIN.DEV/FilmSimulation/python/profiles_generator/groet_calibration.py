#!/usr/bin/env python3
"""Replay US 4,082,553's interimage protocol through the renderer's stage 8b.

⚠ THIS IS THE AUDIT BEHIND `film_profiles._GROET_REVERSAL_IIE_SCALE`, not a
renderer stage. Groet (Eastman Kodak, US 4,082,553, 1978) measured a Kodak
E-4-type haloiodide colour reversal coating: red and blue exposed together
through a 21-step 0.15-log-E wedge, green given a uniform flash at several
levels, magenta density read per step (Figs. 4-6, digitised 2026-09-28c into
`film_profiles._US4082553_WEDGES`). The affected (magenta) density rises as
the causers (yellow + cyan) fall -- the reversal interimage effect in the
density domain, with no characteristic curve or flash intensity needed.

This module measures the SAME observable on a database stock by driving
`film_sim.apply_interimage` exactly as `simulate()` does (same anchors, same
reversal branch, same density weighting), and reports the slope
    k = (D_mag at causer 1.05 - D_mag at causer 3.2) / (3.2 - 1.05)
at four affected levels matched to the patent's own (magenta 2.30 / 1.80 /
1.12 / 0.77 at causer maximum).

    python3 groet_calibration.py            # report every colour reversal stock
    python3 groet_calibration.py --fit      # print the scale factors that land
                                            # the calibrated family on target
"""
from __future__ import annotations
import argparse, sys
from dataclasses import replace
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import film_profiles as fp   # noqa: E402
import film_sim as fs        # noqa: E402

LEVELS = (2.30, 1.80, 1.12, 0.77)
D_HI, D_LO = 3.2, 1.05


def patent_target(fig: str = "fig4") -> tuple[float, ...]:
    """Per-level slope between causer 3.2 and 1.05 on the digitised wedge."""
    w = fp._US4082553_WEDGES[fig]
    dc = (np.array(w["yellow"]) + np.array(w["cyan"])) / 2.0
    out = []
    for key in sorted(k for k in w if k.startswith("m")):
        m = np.array(w[key])
        # causer falls monotonically with the step index; interpolate on it
        hi = min(D_HI, float(dc.max()))
        m_hi = float(np.interp(-hi, -dc, m))
        m_lo = float(np.interp(-D_LO, -dc, m))
        out.append((m_lo - m_hi) / (hi - D_LO))
    return tuple(out)


def model_slopes(p, scale: float = 1.0) -> tuple[float, ...]:
    """The model's slope at the patent's four affected levels."""
    iie = p.interimage
    if scale != 1.0:
        iie = replace(iie, a_rg=iie.a_rg * scale, a_rb=iie.a_rb * scale,
                      a_gr=iie.a_gr * scale, a_gb=iie.a_gb * scale,
                      a_br=iie.a_br * scale, a_bg=iie.a_bg * scale)
    S = fs.RenderSettings()
    ps = fs.get_print_stock(p.default_print)
    an = fs.solve_anchors(p, ps, S.grey_target, S.coupler_scale,
                          S.scanner_specular, S.black_point_stretch)
    cv = p.curves.as_tuple()
    n = 161
    lc = np.log10(0.18) + np.linspace(-3.0, 3.0, n)      # causer sweep
    out = []
    for level in LEVELS:
        # green offset giving magenta == level with the causers at maximum
        def mag_at_max(g):
            le = np.zeros((1, 1, 3), np.float32)
            le[0, 0, 0] = le[0, 0, 2] = lc[0]
            le[0, 0, 1] = np.log10(0.18) + g
            d = np.empty_like(le)
            for c in range(3):
                d[0, :, c] = fs.density(-(le[0, :, c] + np.float32(an[c])), cv[c])
            fs.apply_interimage(d, le, cv, iie, an, True)
            return float(d[0, 0, 1])
        lo, hi = -4.0, 4.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if mag_at_max(mid) > level:
                lo = mid
            else:
                hi = mid
        g = 0.5 * (lo + hi)
        le = np.zeros((1, n, 3), np.float32)
        le[0, :, 0] = lc; le[0, :, 2] = lc
        le[0, :, 1] = np.log10(0.18) + g
        d = np.empty_like(le)
        for c in range(3):
            d[0, :, c] = fs.density(-(le[0, :, c] + np.float32(an[c])), cv[c])
        fs.apply_interimage(d, le, cv, iie, an, True)
        dc = (d[0, :, 0] + d[0, :, 2]) / 2.0
        m = d[0, :, 1]
        hi_d = min(D_HI, float(dc.max()) - 0.02)
        m_hi = float(np.interp(-hi_d, -dc, m))
        m_lo = float(np.interp(-D_LO, -dc, m))
        out.append((m_lo - m_hi) / (hi_d - D_LO))
    return tuple(out)


def fit_scale(p, target_mean: float) -> float:
    lo, hi = 0.2, 6.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if np.mean(model_slopes(p, mid)) < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# US 6,521,400 (Kodak, 2003) Table 1, "Red-on-Green IIE (D = 1.0)" -- queue P85
# ---------------------------------------------------------------------------
# The patent's protocol, in its own words: "exposing film to a red step
# exposure as well as to a uniform green flash exposure. The change in green
# density (at a flash level near D=1.0) with increasing red exposure was then
# measured." Replayed here with the three things the patent leaves unstated
# set explicitly: the red step is Kodak's standard 21-step 0.15 log E tablet
# (3.0 log E) centred on the stock's mid-grey anchor; the green flash is the
# one that gives green D = 1.0 at the LEAST red exposure; blue is unexposed.
# ⚠ The answer is insensitive to the tablet: a full 8 log E sweep reads the
# same to 0.001 D on KODAK_EKTACHROME_100D_5285 (0.195 against 0.194).
#: Table 1 check sample 108 (iodide only, no inhibitor) and comparison 107
#: (directly incorporated CMMT, "current commercial technology").
RG_CHECK, RG_CMMT = 0.41, 0.74


def red_on_green(p, scale: float = 1.0, span: float = 3.0, n: int = 21) -> float:
    """Green density change across the red step, the Table 1 observable."""
    iie = p.interimage
    if scale != 1.0:
        iie = replace(iie, a_rg=iie.a_rg * scale, a_rb=iie.a_rb * scale,
                      a_gr=iie.a_gr * scale, a_gb=iie.a_gb * scale,
                      a_br=iie.a_br * scale, a_bg=iie.a_bg * scale)
    S = fs.RenderSettings()
    ps = fs.get_print_stock(p.default_print)
    an = fs.solve_anchors(p, ps, S.grey_target, S.coupler_scale,
                          S.scanner_specular, S.black_point_stretch)
    cv = p.curves.as_tuple()
    base = np.log10(0.18)
    lr = base + np.linspace(-span / 2.0, span / 2.0, n)

    def run(g, lrv):
        le = np.zeros((1, len(lrv), 3), np.float32)
        le[0, :, 0] = lrv
        le[0, :, 1] = base + g
        le[0, :, 2] = base - 6.0            # blue unexposed
        d = np.empty_like(le)
        for c in range(3):
            d[0, :, c] = fs.density(-(le[0, :, c] + np.float32(an[c])), cv[c])
        fs.apply_interimage(d, le, cv, iie, an, True)
        return d[0]

    lo, hi = -4.0, 4.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if run(mid, lr[:1])[0, 1] > 1.0:
            lo = mid
        else:
            hi = mid
    d = run(0.5 * (lo + hi), lr)
    return float(d[-1, 1] - d[0, 1])


def fit_rg_scale(p, target: float = RG_CHECK) -> float:
    lo, hi = 0.2, 8.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if red_on_green(p, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--p85", action="store_true",
                    help="US 6,521,400 red-on-green IIE on every colour "
                         "reversal stock, and the 100D scale with --fit")
    ns = ap.parse_args(argv)
    if ns.p85:
        print("US 6,521,400 Table 1: check %.2f, CMMT %.2f" % (RG_CHECK, RG_CMMT))
        for p in fp.FILM_PROFILES:
            if not (p.is_reversal and not p.is_monochrome and p.interimage.active):
                continue
            line = "%-28s red-on-green %.3f" % (p.name, red_on_green(p))
            if ns.fit and p.name in fp._US6521400_RG_SCALE:
                base = replace(p, interimage=fp._interimage_for(p, groet_scale=False))
                line += "  -> scale %.4f" % fit_rg_scale(base)
            print(line)
        return 0
    tgt = patent_target("fig4")
    print("patent control (Fig. 4) slopes at levels", LEVELS, "->",
          tuple(round(t, 3) for t in tgt), "mean %.3f" % np.mean(tgt))
    for f in ("fig5", "fig6"):
        print("patent %s slopes" % f, tuple(round(t, 3) for t in patent_target(f)))
    for p in fp.FILM_PROFILES:
        if not (p.is_reversal and not p.is_monochrome and p.interimage.active):
            continue
        ms = model_slopes(p)
        line = "%-28s model %s mean %.3f" % (p.name, tuple(round(v, 3) for v in ms), np.mean(ms))
        if ns.fit and p.name in fp._GROET_REVERSAL_FAMILY:
            base = replace(p, interimage=fp._interimage_for(p, groet_scale=False))
            line += "  -> scale %.4f" % fit_scale(base, float(np.mean(tgt)))
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
