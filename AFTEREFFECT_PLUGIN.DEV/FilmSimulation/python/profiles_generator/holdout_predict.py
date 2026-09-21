#!/usr/bin/env python3
"""Predict a printed number the model never saw, and measure how far off it is.

⚠⚠ THIS IS THE CLOSEST THING TO GROUND TRUTH THIS PROJECT CAN HAVE, AND THE
REASON IT IS NEEDED IS WORTH STATING. Every other check here compares the
database against the document it came from (does the stored curve redraw the
sheet?) or one engine against another (do Python and AVX2 agree?). Both pass
happily while the MODEL is wrong: the stage-8b sign defect rendered 26 reversal
stocks with the interimage correction backwards for weeks with every
transcription correct and every parity probe green.

A hold-out prediction is different in kind. It asks the model to produce a
number that was NOT used to build it, and compares that against what the
manufacturer printed. A model that is merely well-transcribed fails this; only
a model that is right passes.

⚠⚠ AND THE FIRST CANDIDATE WAS REJECTED FOR BEING CIRCULAR, WHICH IS WHY THIS
DOCSTRING LEADS WITH IT. The obvious test -- recompute ISO speed from the traced
characteristic curve and compare it against the printed exposure index -- CANNOT
BE RUN HERE. `ToneCurve`'s abscissa is RELATIVE: every curve is shifted so that
metered mid grey sits at x = 0 (`FilmProfile.speed_point_x` is 0.0 on all 191),
and that shift was computed FROM the stock's rated speed. Predicting the speed
back would be reading out an input. It would score near-perfect and mean
nothing, which is worse than having no test at all. The absolute lux-second
abscissa the sheets print was normalised away at trace time and would have to be
recovered into a new field before that check becomes honest.

WHAT SURVIVES, AND WHY EACH ONE IS NOT CIRCULAR:

  gamma_vs_time   LEAVE-ONE-OUT over each development family. The law
                  (gamma_infinity, dev_rate_k, induction_t0) is re-fitted with
                  one measured point REMOVED, then asked to predict that point.
                  The held-out gamma is never in the fit, so this tests the LAW
                  rather than the transcription. Slope is invariant to the
                  abscissa normalisation, so nothing here depends on the
                  discarded exposure axis.

  rp_vs_mtf       The printed RESOLVING POWER is predicted from the measured
                  MTF curve through a threshold-modulation model with ONE
                  global constant fitted across the whole set. One free
                  parameter against N observations is a test; one per stock
                  would be a curve fit wearing a test's clothes.

Usage:
    python3 holdout_predict.py [--out PATH] [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import film_profiles as FP          # noqa: E402

DEFAULT_OUT = HERE / "doc" / "HOLDOUT_PREDICTIONS.md"

#: A family group is one (developer, dilution, temperature, vessel). Mixing
#: developers into one fit would measure the spread between chemistries and
#: call it a prediction error.
GROUP_KEYS = ("developer", "dilution", "celsius", "vessel")

#: Leave-one-out needs at least this many points in a group: three to fit the
#: three-parameter law plus one to hold out.
MIN_GROUP = 4

#: Accept bands. ⚠ THESE ARE JUDGEMENTS AND ARE LABELLED AS SUCH -- they decide
#: what the report CALLS a pass, not what it measures. The numbers themselves
#: are printed for every row so a reader can apply their own.
GAMMA_TOL = 0.05        # in gamma units; a traced sheet reads to about 0.02
RP_TOL_PCT = 25.0       # resolving power is a threshold judgement by eye


# ---------------------------------------------------------------------------
#  1. gamma(t), leave one point out
# ---------------------------------------------------------------------------

def _gamma_law(t, g_inf, k, t0):
    """The shipped development law: gamma rises to an asymptote after induction."""
    return g_inf * (1.0 - np.exp(-np.maximum(k, 1e-9) * np.maximum(t - t0, 0.0)))


def _fit_gamma_law(times, gammas):
    """Least squares on (gamma_infinity, dev_rate_k, induction_t0).

    Coarse grid then local refinement -- deliberately not scipy: this module
    runs inside the build gate and the gate has no optional dependencies.
    """
    times = np.asarray(times, float)
    gammas = np.asarray(gammas, float)
    best, bp = np.inf, None
    g_hi = max(float(gammas.max()) * 2.5, 0.2)
    for g_inf in np.linspace(float(gammas.max()) * 0.8, g_hi, 24):
        for k in np.geomspace(0.01, 1.5, 24):
            for t0 in np.linspace(0.0, max(0.0, float(times.min()) * 0.9), 8):
                r = float(((_gamma_law(times, g_inf, k, t0) - gammas) ** 2).sum())
                if r < best:
                    best, bp = r, (g_inf, k, t0)
    g_inf, k, t0 = bp
    for scale in (0.25, 0.06, 0.015):
        for dg in (-1, 0, 1):
            for dk in (-1, 0, 1):
                for dt in (-1, 0, 1):
                    c = (g_inf * (1 + dg * scale), k * (1 + dk * scale),
                         max(0.0, t0 + dt * scale * 5.0))
                    r = float(((_gamma_law(times, *c) - gammas) ** 2).sum())
                    if r < best:
                        best, (g_inf, k, t0) = r, c
    return g_inf, k, t0


def gamma_vs_time() -> dict:
    rows = []
    for p in FP.FILM_PROFILES:
        fam = p.processing_family
        if not fam or not getattr(fam, "points", ()):
            continue
        groups: dict[tuple, list] = {}
        for pt in fam.points:
            if not (pt.gamma > 0.0 and pt.minutes > 0.0):
                continue
            groups.setdefault(tuple(getattr(pt, k) for k in GROUP_KEYS),
                              []).append((float(pt.minutes), float(pt.gamma)))
        for key, pts in groups.items():
            # One measurement per time: a repeated time is a repeat reading,
            # and holding one out while its twin stays in the fit would be a
            # prediction of a number the fit already has.
            byt: dict[float, float] = {}
            for t, g in pts:
                byt.setdefault(t, g)
            pts = sorted(byt.items())
            if len(pts) < MIN_GROUP:
                continue
            times = np.array([t for t, _g in pts])
            gam = np.array([g for _t, g in pts])
            for i in range(len(pts)):
                m = np.ones(len(pts), bool)
                m[i] = False
                try:
                    law = _fit_gamma_law(times[m], gam[m])
                except Exception:
                    continue
                pred = float(_gamma_law(times[i], *law))
                rows.append({
                    "stock": p.name,
                    "developer": str(key[0]), "celsius": key[2],
                    "minutes": float(times[i]),
                    "printed": float(gam[i]), "predicted": pred,
                    "error": pred - float(gam[i]),
                    "n_fit": int(m.sum()),
                })
    errs = np.array([abs(r["error"]) for r in rows]) if rows else np.zeros(0)
    return {
        "name": "gamma_vs_time",
        "what": "development gamma at a held-out time, from the law re-fitted "
                "without that point",
        "truth": "the sheet's own printed gamma at that time",
        "unit": "gamma",
        "tolerance": GAMMA_TOL,
        "n": len(rows),
        "stocks": len({r["stock"] for r in rows}),
        "median_abs": float(np.median(errs)) if len(errs) else 0.0,
        "p90_abs": float(np.percentile(errs, 90)) if len(errs) else 0.0,
        "worst": max(rows, key=lambda r: abs(r["error"])) if rows else None,
        "pass": int((errs <= GAMMA_TOL).sum()),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
#  2. resolving power from the measured MTF
# ---------------------------------------------------------------------------

def _predict_rp(mtf, threshold: float) -> float:
    """Frequency where the measured MTF falls to `threshold`, cycles/mm.

    The classical threshold-modulation model: a target is resolved while the
    film's modulation transfer stays above the eye-plus-grain threshold. One
    constant, fitted once over the whole set below.
    """
    f = np.linspace(1.0, 2000.0, 4000)
    m = np.asarray(FP.mtf_response(mtf, 1, f), float)
    below = np.nonzero(m <= threshold)[0]
    if not len(below):
        return float("nan")
    i = below[0]
    if i == 0:
        return float(f[0])
    f0, f1, m0, m1 = f[i - 1], f[i], m[i - 1], m[i]
    return float(f0 + (m0 - threshold) * (f1 - f0) / max(m0 - m1, 1e-12))


def rp_vs_mtf() -> dict:
    pairs = []
    for p in FP.FILM_PROFILES:
        m = p.mtf
        if not m.mtf_measured:
            continue
        rp = m.resolving_power_lp_mm_highc or m.resolving_power_lp_mm_lowc
        if not rp:
            continue
        pairs.append((p, float(rp),
                      "high" if m.resolving_power_lp_mm_highc else "low"))

    # ⚠ ONE GLOBAL CONSTANT, FITTED ON LOG ERROR. Fitting it per stock would
    # guarantee a pass and prove nothing; fitting on the log makes a factor-of-2
    # miss cost the same whether the film is fast or fine.
    def loss(th):
        e = []
        for p, rp, _k in pairs:
            q = _predict_rp(p.mtf, th)
            if q == q and q > 0:
                e.append((math.log(q) - math.log(rp)) ** 2)
        return sum(e) / max(len(e), 1)

    grid = np.linspace(0.02, 0.45, 44)
    th = float(min(grid, key=loss))
    for _ in range(3):
        span = (grid[1] - grid[0]) / 2
        grid = np.linspace(max(0.005, th - span), th + span, 21)
        th = float(min(grid, key=loss))

    rows = []
    for p, rp, kind in pairs:
        q = _predict_rp(p.mtf, th)
        rows.append({
            "stock": p.name, "printed": rp, "predicted": q,
            "contrast": kind,
            "error_pct": 100.0 * (q - rp) / rp if rp else float("nan"),
            "f50_g": p.mtf.f50_g, "q": p.mtf.mtf_rolloff_q,
        })
    errs = np.array([abs(r["error_pct"]) for r in rows if r["predicted"] == r["predicted"]])

    # ⚠⚠ THE CONDITIONS THE PRINTED NUMBER WAS MEASURED UNDER ARE NOT RECORDED,
    # AND THAT DECIDES HOW THIS RESULT MUST BE READ. `resolving_density`,
    # `resolving_optic` and `resolving_target_contrast` exist in the schema and
    # are EMPTY on every stock carrying a printed resolving power -- counted
    # here rather than asserted, so the day they are filled this note changes
    # itself. Queue P72 measured why that matters: Ooue 1959 Fig. 5 shows
    # resolving power PEAKING at D 0.70-1.05 and falling either side, so a
    # figure printed with no density is the top of a curve and a figure printed
    # at a working density is somewhere on its flank. Two stocks quoted under
    # different unstated conditions cannot be brought onto one threshold, and a
    # residual of this size is what that looks like.
    _cond = sum(1 for p, _rp, _k in pairs
                if p.mtf.resolving_density or p.mtf.resolving_optic
                or p.mtf.resolving_target_contrast)
    return {
        "name": "rp_vs_mtf",
        "conditions_recorded": _cond,
        "inconclusive": _cond == 0,
        "what": "printed resolving power, predicted from the measured MTF "
                f"through one global threshold modulation ({th:.3f})",
        "truth": "the sheet's own printed resolving power",
        "unit": "%",
        "tolerance": RP_TOL_PCT,
        "threshold": th,
        "n": len(rows),
        "stocks": len({r["stock"] for r in rows}),
        "median_abs": float(np.median(errs)) if len(errs) else 0.0,
        "p90_abs": float(np.percentile(errs, 90)) if len(errs) else 0.0,
        "worst": max(rows, key=lambda r: abs(r["error_pct"])) if rows else None,
        "pass": int((errs <= RP_TOL_PCT).sum()),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
#  3. inter-layer speed separation -- REFUSED, and the refusal is the result
# ---------------------------------------------------------------------------

def layer_separation() -> dict:
    """The third prediction, and it is NOT RUN because its truth is prose.

    The r/g/b records were all shifted by ONE number at trace time, so their
    SEPARATION survived the normalisation and is predictable. What is missing is
    the other half: the printed layer-speed statements it would be compared
    against exist in this corpus as SENTENCES inside profile comments and source
    notes ("the pair differs by 0.40 log E of speed at the same contrast"), not
    as a field any program can read.

    ⚠ REPORTING A REFUSAL RATHER THAN A NUMBER IS THE POINT. Scraping those
    sentences with a regular expression would produce a pass rate built on
    whatever the regex happened to match, which is a measurement of the regex.
    The honest fix is a schema carrier -- a `layer_speed_offset` with its own
    ParamSource -- and until that exists this check has no truth to hold out.
    """
    return {
        "name": "layer_separation",
        "what": "inter-layer speed separation against published statements",
        "status": "NOT RUN -- no machine-readable source",
        "n": 0,
    }


# ---------------------------------------------------------------------------

def render(results: list[dict]) -> str:
    L: list[str] = []
    w = L.append
    w("# HOLDOUT_PREDICTIONS.md — the model predicting numbers it was not given")
    w("")
    w("**Generated by `holdout_predict.py`. Do not edit by hand.** "
      f"Schema v{FP.SCHEMA_VERSION}, {len(FP.FILM_PROFILES)} film stocks.")
    w("")
    w("## Why this file exists")
    w("")
    w("Every other check in this project compares the database against the "
      "document it came from, or one engine against another. Both pass while "
      "the **model** is wrong — the stage-8b sign defect rendered 26 reversal "
      "stocks backwards for weeks with every transcription correct and every "
      "parity probe green.")
    w("")
    w("A hold-out prediction asks the model for a number that was **not used "
      "to build it**, and compares it against what the manufacturer printed.")
    w("")
    w("⚠⚠ **THE OBVIOUS TEST — ISO SPEED FROM THE TRACED CURVE — IS CIRCULAR "
      "HERE AND IS NOT RUN.** `ToneCurve`'s abscissa is relative: every curve "
      "is shifted so metered mid grey sits at x = 0, and that shift was "
      "computed from the stock's rated speed. Predicting the speed back would "
      "read out an input, score near-perfect and mean nothing. The absolute "
      "lux-second axis the sheets print was normalised away at trace time and "
      "must be recovered into a field of its own before that check is honest.")
    w("")
    w("## Results")
    w("")
    w("| prediction | n | stocks | median error | 90th pct | within tolerance |")
    w("|---|---|---|---|---|---|")
    for r in results:
        if r.get("status"):
            w(f"| **{r['name']}** | — | — | — | — | *{r['status']}* |")
            continue
        unit = r["unit"]
        verdict = (f"**{r['pass']} / {r['n']}** (±{r['tolerance']:g} {unit})"
                   if not r.get("inconclusive")
                   else f"⚠ **INCONCLUSIVE** — {r['pass']} / {r['n']} inside "
                        f"±{r['tolerance']:g} {unit}, but see below")
        w(f"| **{r['name']}** | {r['n']} | {r['stocks']} | "
          f"{r['median_abs']:.3f} {unit} | {r['p90_abs']:.3f} {unit} | "
          f"{verdict} |")
    w("")
    w("⚠ The tolerances are judgements and decide only what this file CALLS a "
      "pass. Every row's own error is printed below, so a reader can apply "
      "their own.")
    w("")

    for r in results:
        w(f"## {r['name']}")
        w("")
        if r.get("status"):
            w(f"**{r['status']}.** " + (layer_separation.__doc__ or "").strip()
              .split("\n\n", 1)[1].replace("\n", " "))
            w("")
            continue
        w(f"**Predicts:** {r['what']}  ")
        w(f"**Against:** {r['truth']}")
        w("")
        if r.get("inconclusive"):
            w("⚠⚠ **THIS RESULT IS INCONCLUSIVE AND THE REASON IS MISSING "
              "METADATA, NOT A FAILED MODEL.** `resolving_density`, "
              "`resolving_optic` and `resolving_target_contrast` are in the "
              f"schema and are empty on **{r['n'] - r['conditions_recorded']} "
              f"of {r['n']}** rows here — the conditions each printed number "
              "was measured under were never recorded. Queue **P72** measured "
              "why that decides everything: Ooue 1959 Fig. 5 shows resolving "
              "power **peaking at D 0.70–1.05 and falling either side**, so a "
              "figure printed with no density is the top of a curve while "
              "another, printed at a working density, sits on its flank. Two "
              "such numbers cannot be brought onto one threshold.")
            w("")
            w("**What would make it a real test:** fill those three fields "
              "from the same sheets the resolving powers came from — those "
              "sheets are already on disk. Until then this section measures "
              "the spread of unstated conditions, and is reported rather than "
              "scored.")
            w("")
        if r["worst"]:
            worst = r["worst"]
            w("Worst row: `" + worst["stock"] + "` — "
              + ", ".join(f"{k} {v}" for k, v in worst.items() if k != "stock"))
            w("")
        rows = sorted(r["rows"],
                      key=lambda x: -abs(x.get("error", x.get("error_pct", 0))))
        head = [k for k in rows[0] if k != "stock"]
        w("| stock | " + " | ".join(head) + " |")
        w("|---|" + "---|" * len(head))
        for x in rows[:40]:
            cells = []
            for k in head:
                v = x[k]
                cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
            w(f"| {x['stock']} | " + " | ".join(cells) + " |")
        if len(rows) > 40:
            w("")
            w(f"*{len(rows) - 40} further rows omitted; the full set is in "
              f"`holdout_predictions.json`.*")
        w("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--json", default=str(HERE / "holdout_predictions.json"))
    ns = ap.parse_args()

    results = [gamma_vs_time(), rp_vs_mtf(), layer_separation()]
    Path(ns.out).write_text(render(results), encoding="utf-8")
    Path(ns.json).write_text(json.dumps(results, indent=1) + "\n",
                             encoding="utf-8")
    for r in results:
        if r.get("status"):
            print(f"  {r['name']:18s} {r['status']}")
        else:
            print(f"  {r['name']:18s} n={r['n']:4d} over {r['stocks']:3d} "
                  f"stocks, median |err| {r['median_abs']:.3f} {r['unit']}, "
                  f"{r['pass']}/{r['n']} within tolerance")
    print(f"[OK] wrote {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
