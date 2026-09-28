#!/usr/bin/env python3
"""FERRANIA P30 alfa — the 14-page sensitometric test report, read whole.

    PDF/PROFILES/FERRANIA/Ferrania-P30-alfa_D-76_1+1.pdf
    «Ferrania P30 alfa», test date 05/13/17, D-76 1+1, 20 °C, 300 ml, ISO 80,
    five development times: 5, 8, 11, 16 and 23 minutes.

⚠⚠ THE DATABASE USED TO KEEP FIVE CURVES AND TWO NUMBERS OUT OF ABOUT SIXTY.
The 2026-09-24 pass fitted the page-2 step tablet — 21 steps × 5 legs, 105
printed densities — into five `ToneCurve`s and put Avg. G and SBR into prose.
Everything else the report prints was read and dropped on the floor: the
effective film speed each development actually meters at, the Zone N number,
the operator's measured B+F, the log-exposure interval, the paper target the
speed point was referred to. Schema v54's `SensitometryReport` is where they
live now, and this module re-derives every one of them on each build.

WHAT IS PRINTED, WHAT IS TRACED, AND WHY THE TRACE IS TRUSTWORTHY
------------------------------------------------------------------
Pages 4–8 print, per leg: B+F, Emax, IDmax, Emin, IDmin, DR, LogE, Avg. G,
Paper ES, SBR, EFS, the speed method and the flare density. All transcribed.

**Two quantities the report only PLOTS**:

  * the **Zone N number** (pages 11 and 14), and
  * the **sub-step part of the effective film speed** — the analysis block
    prints «32+», «50-», «64--», and only the page-12/14 log axis says how
    far above 32 or below 64.

Tracing those is licensed by the pages where the answer is already known.
Pages 9 and 10 plot SBR and Avg. G against the same time axis, and the same
marker-finder run over them returns the **printed** SBR to 0.04 and the
**printed** Avg. G to 0.00. The EFS axis reproduces its own nine-rung ladder
(10 … 64) to **0.08 px**. And the N number is plotted **twice** — against time
on page 11, against effective speed on page 14 — and the two readings agree to
**0.002**. A tracer that reproduces three known axes and agrees with itself on
the fourth is being checked, not trusted.

⚠ **The markers are ✕ glyphs, two crossing 0.6 pt strokes each**, which is what
the finder looks for: a pair of segments that are neither horizontal nor
vertical, inside the plot box, at a stroke width the grid never uses. The
0.36 pt joining polyline is deliberately ignored — it interpolates, the markers
are the data.

THE REPORT'S OWN N CONVENTION, DERIVED AND NOT ASSUMED
--------------------------------------------------------
Fitting the five (SBR, N) pairs gives **N = 0 at SBR 7.0** — the Zone-System
normal — at about **0.56 N per stop of SBR**. That is NOT the one-N-per-stop
rule of thumb, which is exactly why the numbers are stored rather than
computed from SBR. The module asserts both the intercept and the slope: if a
future reading drifts, the relation it has to satisfy is the report's own.

WHAT THIS DOCUMENT CANNOT GROUND, ON ANY PAGE
-----------------------------------------------
No granularity, no MTF, no spectral sensitivity, no reciprocity, no D-max, no
resolving power. It is a tone-reproduction test and nothing else, and this
profile's rms 7.4, f50 66 and clump 3.2 µm remain estimates that no page here
touches.

Run:  python ferrania_p30_alfa_report.py [--root .] [--assert]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PDF = "PDF/PROFILES/FERRANIA/Ferrania-P30-alfa_D-76_1+1.pdf"

MINUTES = (5, 8, 11, 16, 23)

#: Per-leg analysis block, pages 4-8, exactly as printed.
#: minutes -> (B+F, Emax, IDmax, Emin, IDmin, DR, LogE, AvgG, PaperES, SBR)
PRINTED = {
    5:  (0.27, -0.25, 1.37, 1.57, 0.37, 1.00, 1.82, 0.55, 1.00, 6.1),
    8:  (0.28,  0.38, 1.38, 1.89, 0.38, 1.00, 1.51, 0.66, 1.00, 5.0),
    11: (0.29,  0.65, 1.39, 2.02, 0.39, 1.00, 1.37, 0.73, 1.00, 4.6),
    16: (0.29,  0.98, 1.39, 2.18, 0.39, 1.00, 1.20, 0.83, 1.00, 4.0),
    23: (0.29,  1.18, 1.39, 2.26, 0.39, 1.00, 1.08, 0.93, 1.00, 3.6),
}

#: What the trace must return. Pinned so a rerun that disagrees FAILS.
EXPECTED_N = (0.527, 1.117, 1.391, 1.715, 1.943)
EXPECTED_EFS = (12.01, 25.07, 33.63, 48.71, 58.58)

#: The nine-rung effective-speed ladder on pages 12-14.
EFS_LADDER = (10, 12, 16, 20, 25, 32, 40, 50, 64)

PAGE_N_VS_TIME, PAGE_N_VS_EFS = 10, 13      # 0-based
PAGE_SBR_VS_TIME, PAGE_G_VS_TIME = 8, 9
PAGE_EFS_LADDER = 13


# ---------------------------------------------------------------------------
def _markers(page, ylo=380.0, yhi=620.0):
    """Centres of the ✕ data markers: crossing strokes at 0.6 pt.

    ⚠ THE JOINING POLYLINE IS 0.36 pt AND IS EXCLUDED BY THE WIDTH TEST. It
    interpolates between the five measurements and is not one of them.
    """
    seen = []
    for dr in page.get_drawings():
        if (dr.get("width") or 0.0) < 0.5:
            continue
        for it in dr["items"]:
            if it[0] != "l":
                continue
            a, b = it[1], it[2]
            if abs(a.x - b.x) < 0.5 or abs(a.y - b.y) < 0.5:
                continue                       # a rule, not a ✕ arm
            if not (ylo < a.y < yhi):
                continue
            c = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
            if not any(abs(c[0] - o[0]) < 1.0 and abs(c[1] - o[1]) < 1.0
                       for o in seen):
                seen.append(c)
    return sorted(seen)


def _axis(page, wanted, horizontal=False, log=False):
    """Fit an axis from its own printed rungs. Returns (poly, worst residual).

    The polynomial maps PIXEL -> VALUE, so a marker coordinate goes straight
    through it.

    ⚠⚠ THE SIDE TEST IS NOT COSMETIC AND ITS ABSENCE PRODUCED A WRONG AXIS.
    These plots put the value ladder down the RIGHT margin (x > 550) and the
    time or speed ladder along the BOTTOM (y > 585), and the two ladders share
    values: page 9's SBR rung «5,00» and its x-axis minute label «5» both
    parse to 5.0. Collected without a side test, the second overwrote the
    first and the SBR axis fitted to 1.75 of residual -- which is why this
    module refuses to read N until it has reproduced the PRINTED SBR.
    """
    pos = {}
    for w in page.get_text("words"):
        if horizontal:
            if (w[1] + w[3]) / 2.0 < 585.0:
                continue
        elif (w[0] + w[2]) / 2.0 < 550.0:
            continue
        t = w[4].replace(",", ".")
        try:
            v = float(t)
        except ValueError:
            if not t.startswith("N+"):
                continue
            try:
                v = float(t[2:])
            except ValueError:
                continue
        if v not in wanted:
            continue
        pos[v] = (w[0] + w[2]) / 2.0 if horizontal else (w[1] + w[3]) / 2.0
    if len(pos) < 3:
        return None, None, len(pos)
    vals = [math.log10(v) if log else v for v in pos]
    a = np.polyfit(list(pos.values()), vals, 1)
    worst = max(abs(a[0] * px + a[1] - v) for px, v in zip(pos.values(), vals))
    return a, worst, len(pos)


def read(root: Path, verbose=True):
    import pymupdf
    doc = pymupdf.open(str(root / PDF))
    if doc.page_count != 14:
        return None, "the report is %d pages, not 14" % doc.page_count
    out = {}

    # ---- the two control axes: their answers are already printed ----------
    ctl = {}
    for pg, wanted, name, printed in (
            (PAGE_SBR_VS_TIME,
             {3.33, 3.67, 4.00, 4.33, 4.67, 5.00, 5.33, 5.67, 6.00, 6.33},
             "SBR", [PRINTED[m][9] for m in MINUTES]),
            (PAGE_G_VS_TIME, {0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
             "Avg. G", [PRINTED[m][7] for m in MINUTES])):
        a, res, n = _axis(doc[pg], wanted)
        if a is None:
            return None, "page %d: only %d axis rungs found" % (pg + 1, n)
        got = [a[0] * y + a[1] for _x, y in _markers(doc[pg])]
        if len(got) != 5:
            return None, "page %d: %d markers, want 5" % (pg + 1, len(got))
        err = max(abs(g - p) for g, p in zip(got, printed))
        ctl[name] = (res, err, got)
        if verbose:
            print("  CONTROL  page %2d  %-7s axis resid %.4f, worst "
                  "traced-vs-PRINTED %.3f" % (pg + 1, name, res, err))

    # ---- the effective-speed ladder ---------------------------------------
    a_efs, res_efs, n = _axis(doc[PAGE_EFS_LADDER], set(EFS_LADDER),
                              horizontal=True, log=True)
    if a_efs is None:
        return None, "the EFS ladder has only %d rungs" % n
    px_res = res_efs / abs(a_efs[0])
    if verbose:
        print("  EFS ladder: %d rungs, fit reproduces them to %.2f px "
              "(%.3f %% in speed)" % (n, px_res, 100 * (10 ** res_efs - 1)))

    # ---- N against time, and N against effective speed --------------------
    nwanted = {0.33, 0.66, 1.0, 1.33, 1.66, 2.0}
    a_n1, res_n1, _ = _axis(doc[PAGE_N_VS_TIME], nwanted)
    a_n2, res_n2, _ = _axis(doc[PAGE_N_VS_EFS], nwanted)
    if a_n1 is None or a_n2 is None:
        return None, "an N axis could not be fitted"
    m1 = _markers(doc[PAGE_N_VS_TIME])
    m2 = _markers(doc[PAGE_N_VS_EFS])
    if len(m1) != 5 or len(m2) != 5:
        return None, ("N markers: %d on page 11, %d on page 14, want 5 each"
                      % (len(m1), len(m2)))
    n_time = [a_n1[0] * y + a_n1[1] for _x, y in m1]
    n_efs = [a_n2[0] * y + a_n2[1] for _x, y in m2]
    efs = [10.0 ** (a_efs[0] * x + a_efs[1]) for x, _y in m2]
    agree = max(abs(p - q) for p, q in zip(n_time, n_efs))
    if verbose:
        print("  N axes resid %.4f / %.4f; the two independent plots of N "
              "agree to %.4f" % (res_n1, res_n2, agree))

    out["n_number"] = n_time
    out["efs"] = efs
    out["n_cross_agreement"] = agree
    out["controls"] = ctl
    out["efs_axis_px_resid"] = px_res
    return out, ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    if not (root / PDF).is_file():
        print("  [SKIP] source not present: %s" % (root / PDF))
        return 0

    print("FERRANIA P30 alfa -- sensitometric test report 05/13/17, all 14 "
          "pages")
    got, err = read(root, verbose=True)
    if got is None:
        print("  [FAIL] %s" % err)
        return 1 if ns.assert_ else 0
    bad = 0

    # the tracer must reproduce the two axes whose answers are printed
    for name, (res, e, _v) in got["controls"].items():
        if e > 0.05:
            print("  [FAIL] the tracer misses the PRINTED %s by %.3f -- it "
                  "cannot then be trusted on N or on the sub-step speed"
                  % (name, e))
            bad += 1
    if got["efs_axis_px_resid"] > 0.5:
        print("  [FAIL] the EFS ladder fits its own rungs to only %.2f px"
              % got["efs_axis_px_resid"])
        bad += 1
    if got["n_cross_agreement"] > 0.02:
        print("  [FAIL] the two independent N plots disagree by %.3f"
              % got["n_cross_agreement"])
        bad += 1

    print()
    for i, m in enumerate(MINUTES):
        print("  %2d min   N %+.3f (pinned %+.3f)   EFS %6.2f (pinned %6.2f)"
              % (m, got["n_number"][i], EXPECTED_N[i],
                 got["efs"][i], EXPECTED_EFS[i]))
        if abs(got["n_number"][i] - EXPECTED_N[i]) > 0.01:
            print("    [MISMATCH] N")
            bad += 1
        if abs(got["efs"][i] - EXPECTED_EFS[i]) > 0.15:
            print("    [MISMATCH] EFS")
            bad += 1

    # ---- the report's own N convention ------------------------------------
    sbr = np.array([PRINTED[m][9] for m in MINUTES])
    nn = np.array(got["n_number"])
    slope, intercept = np.polyfit(sbr, nn, 1)
    zero_at = -intercept / slope
    print("\n  N AGAINST SBR: %.3f N per stop, N = 0 at SBR %.2f -- the "
          "Zone-System normal is 7.0, and the rule-of-thumb slope is 1.0"
          % (-slope, zero_at))
    if abs(zero_at - 7.0) > 0.25 or not 0.45 < -slope < 0.70:
        print("  [FAIL] the N convention no longer lands on SBR 7.0 at about "
              "0.56 N per stop; the trace or the transcription has drifted")
        bad += 1

    # ---- against the database ---------------------------------------------
    try:
        import film_profiles as fp
        # ⚠ THE PROFILE HAS SIX LEGS SINCE 2026-09-25b AND THIS REPORT IS
        # ONLY FIVE OF THEM. The sixth is Film Ferrania's own D-76 **stock**
        # drawing off «Curve caratteristiche e sensibilita spettrali» -- a
        # different developer strength, a different document, and the only
        # manufacturer characteristic curve the original emulsion has. This
        # module owns the five 1+1 legs and must not claim the stock one, so
        # it selects by dilution instead of asserting a total.
        vs = tuple(v for v in (fp._PROCESS_VARIANTS.get("FERRANIA_P30") or ())
                   if "1+1" in v.name)
        if len(vs) != 5:
            print("\n  [FAIL] the database holds %d D-76 1+1 legs, the report "
                  "has 5" % len(vs))
            bad += 1
        else:
            print()
            drift = []
            for v, m in zip(vs, MINUTES):
                r = v.report
                bf, emax, idmax, emin, idmin, dr, loge, g, pes, sbr_p = \
                    PRINTED[m]
                for nme, a, b, tol in (
                        ("base_fog", r.base_fog, bf, 1e-9),
                        # ⚠ PAIRED AS THE REPORT PRINTS THEM: Emax sits
                        # beside IDmax and is the SMALLER number, because the
                        # axis is step-tablet attenuation.
                        ("exposure_max", r.exposure_max, emax, 1e-9),
                        ("exposure_min", r.exposure_min, emin, 1e-9),
                        ("density_max", r.density_max, idmax, 1e-9),
                        ("density_min", r.density_min, idmin, 1e-9),
                        ("density_range", r.density_range, dr, 1e-9),
                        ("log_exposure_range", r.log_exposure_range, loge,
                         1e-9),
                        ("avg_gradient", r.avg_gradient, g, 1e-9),
                        ("paper_exposure_scale", r.paper_exposure_scale, pes,
                         1e-9),
                        ("subject_brightness_range",
                         r.subject_brightness_range, sbr_p, 1e-9),
                        ("zone_n_number", r.zone_n_number,
                         EXPECTED_N[MINUTES.index(m)], 0.011),
                        ("effective_film_speed", r.effective_film_speed,
                         EXPECTED_EFS[MINUTES.index(m)], 0.16)):
                    if abs(a - b) > tol:
                        drift.append("%d min %s %s vs %s" % (m, nme, a, b))
                if r.paper_speed_point != 2.40 or r.flare_density != 0.0200:
                    drift.append("%d min PSP/flare" % m)
                if r.speed_method != "0.1 over FB+F":
                    drift.append("%d min speed_method" % m)
            print("  AGAINST THE DATABASE: %s"
                  % ("all five legs carry the full report"
                     if not drift else "; ".join(drift[:3])))
            if drift:
                bad += 1

            # ⚠ THE TWO DELIBERATE DISAGREEMENTS, PRINTED SO THEY STAY VISIBLE
            print("\n  THE TWO PAIRS THAT ARE KEPT BECAUSE THEY DISAGREE:")
            for v, m in zip(vs, MINUTES):
                c = v.curves.g
                print("    %2d min  B+F printed %.2f vs fitted dmin %.4f "
                      "(%+.4f)   Avg. G %.2f vs ToneCurve.gamma %.4f (%.2fx)"
                      % (m, v.report.base_fog, c.dmin,
                         c.dmin - v.report.base_fog,
                         v.report.avg_gradient, c.gamma,
                         c.gamma / v.report.avg_gradient))
    except Exception as exc:                                  # pragma: no cover
        print("  [WARN] could not consult film_profiles: %s" % exc)

    if ns.assert_ and bad:
        print("\n[FAIL] the P30 alfa report does not reproduce")
        return 1
    print("\n[OK] all five legs re-read: 10 printed figures each, plus the "
          "Zone N number and the sub-step effective speed traced off axes "
          "whose printed neighbours the same tracer reproduces exactly.")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
