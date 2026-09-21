#!/usr/bin/env python3
"""Emit `doc/REALISM_SCORE.md` -- how much of each render is backed by measurement.

⚠⚠ THIS FILE ANSWERS ONE QUESTION AND REFUSES THE OTHER ONE. It answers: **what
fraction of the pixel movement in a render of this stock comes from a MEASURED
number rather than from an estimate?** It does NOT answer "how close is the
render to real film" -- nothing in this project can answer that yet, because no
frame here has ever been compared against a scan of the same scene on the same
stock. Anyone reading a high score as "95 % realistic" is reading it wrong, and
the report says so in its own first paragraph.

    score(stock) = SUM_a  I_a * e(a, stock)  /  SUM_a  I_a

`I_a` is the MEASURED influence of axis `a` -- the mean CIE76 delta-E between a
render using that stock's measurement and a render using the fallback a stock
with no such measurement gets. It is produced by `realism_ablation.py` and
cached with a hash of the render model, so a model change invalidates it rather
than silently re-weighting the score.

`e(a, stock)` is the evidence the stock actually has on that axis, read from its
own `ParamSource` records -- 1.0 traced or measured from a manufacturer document,
down to 0.1 for an assumed value.

⚠ TWO HONEST LIMITS, STATED HERE AND REPEATED IN THE REPORT:

  1. An axis is weighted by what the measurement is worth ON THE STOCKS THAT
     HAVE IT. Applying that weight to a stock that does not have it assumes the
     measurement would have been worth about the same there. That is an
     assumption, not a measurement.
  2. An axis whose influence is ZERO is not scored at all, and those axes are
     listed separately. A carrier nothing on the render path reads cannot make
     a picture more realistic, however well evidenced it is.

Usage:
    python3 gen_realism_score.py [--out PATH] [--assert]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import importlib.util             # noqa: E402

import film_profiles as FP          # noqa: E402
import realism_ablation as AB       # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_gen_film_control_matrix", HERE / "gen_film_control_matrix.py")
MATRIX = importlib.util.module_from_spec(_spec)
sys.modules["_gen_film_control_matrix"] = MATRIX
_spec.loader.exec_module(MATRIX)

DEFAULT_OUT = HERE / "doc" / "REALISM_SCORE.md"

#: ParamSource (tier, status) -> evidence weight.
#:
#: ⚠ THE LADDER IS DELIBERATELY STEEP AT THE BOTTOM. "assumed" is not a weak
#: measurement, it is the absence of one wearing a number's clothes, so it
#: scores 0.1 rather than 0.4. A gentle ladder would let a corpus of assumptions
#: average out to a respectable score, which is the failure this whole file
#: exists to prevent.
EVIDENCE = {
    (1, "traced"): 1.0, (1, "measured"): 1.0, (1, "stated"): 1.0,
    (1, "derived"): 0.8, (1, "estimated"): 0.4, (1, "assumed"): 0.2,
    (2, "traced"): 0.8, (2, "measured"): 0.8, (2, "stated"): 0.7,
    (2, "derived"): 0.5, (2, "estimated"): 0.5, (2, "assumed"): 0.2,
    (3, "traced"): 0.4, (3, "measured"): 0.4, (3, "stated"): 0.3,
    (3, "derived"): 0.25, (3, "estimated"): 0.25, (3, "assumed"): 0.1,
}

#: Axis -> the ParamSource parameter that evidences it, when one exists.
#: Where no per-parameter record exists the structural carrier decides: the
#: stock either has the traced curve set or it does not.
AXIS_PARAM = {
    "tone_curve": "curves.g.gamma",
    "grain_amplitude": "grain.rms_granularity",
    "sharpness_mtf": "mtf.f50_g",
    "halation": "halation.gain_r",
    "callier": "callier_q",
    "dye_matrix": "dye_matrix",
    "edge_effects": "mtf.adjacency",
}

#: Axis -> (carrier test, evidence when the carrier is present and no
#: ParamSource names it, evidence when it is absent).
#: ⚠ ABSENT IS NOT ZERO. A stock with no traced sigma(D) still renders a
#: sigma(D) -- the legacy square-root law -- and that law is itself evidenced
#: (Selwyn, and four traced VISION3 sheets behind its shape). 0.15 says
#: "a defensible default", not "a measurement".
AXIS_CARRIER = {
    "grain_shape": (lambda p: bool(p.grain.sigma_shape_measured), 1.0, 0.15),
    "spectral_sensitivity": (lambda p: bool(p.spectral.has_data), 1.0, 0.15),
    "dye_density": (lambda p: bool(p.dye_density.d_cyan
                                   or p.dye_density.d_neutral), 1.0, 0.15),
    "interimage": (lambda p: any(abs(getattr(p.interimage, f, 0.0)) > 0.0
                                 for f in AB._IIE_FIELDS), 0.5, 0.15),
    "reciprocity": (lambda p: bool(p.reciprocity_table.times_s), 1.0, 0.15),
}


#: Axis -> the FilmControlMatrix predicate that decides whether the axis
#: APPLIES to a stock at all. Imported rather than restated: the matrix already
#: draws the line between "no measurement" and "no such property", and a second
#: copy of that line would drift.
#:
#: ⚠⚠ AN INAPPLICABLE AXIS IS EXCLUDED FROM THE STOCK'S DENOMINATOR, NOT SCORED
#: LOW. A colour negative has no Callier effect because a dye image has no
#: developed silver to scatter light -- scoring it 0.25 there would mark a
#: closed question as a research gap and would rank every colour stock below
#: every monochrome one for a property colour film does not have. This is the
#: same O-versus-? distinction FilmControlMatrix.md exists to keep.
AXIS_APPLIES = {
    "callier": "_callier",
    "interimage": "_colour",
    "dye_matrix": "_colour",
    "halation": "_halation",
}


def _applies(profile, axis: str) -> bool:
    name = AXIS_APPLIES.get(axis)
    if not name:
        return True
    return getattr(MATRIX, name)(profile) != MATRIX.O


def evidence(profile, axis: str) -> tuple[float, str]:
    """(weight, one-word reason) for this stock on this axis."""
    param = AXIS_PARAM.get(axis)
    if param:
        for s in profile.param_sources:
            if s.param == param:
                w = EVIDENCE.get((s.tier, s.status))
                if w is None:
                    w = 0.25
                return w, f"T{s.tier} {s.status}"
        # No record naming it: fall through to the carrier test if there is
        # one, otherwise treat an unevidenced value as an assumption.
    test = AXIS_CARRIER.get(axis)
    if test:
        has, yes, no = test
        return (yes, "carrier present") if has(profile) else (no, "no carrier")
    return 0.1, "no record"


def load_influence(strict: bool) -> dict:
    if not AB.CACHE.is_file():
        raise SystemExit("[FAIL] realism_influence.json is missing -- run "
                         "`python3 realism_ablation.py` first")
    data = json.loads(AB.CACHE.read_text(encoding="utf-8"))
    if data.get("model_hash") != AB.model_hash():
        msg = ("realism_influence.json was measured against a different render "
               "model -- re-run `python3 realism_ablation.py`")
        if strict:
            raise SystemExit(f"[FAIL] {msg}")
        print(f"[WARN] {msg}")
    return data


def score_all(inf: dict):
    axes = inf["axes"]
    live = {k: v["delta_e_median"] for k, v in axes.items()
            if v["delta_e_median"] > 0.0}
    dead = {k: v for k, v in axes.items() if v["delta_e_median"] <= 0.0}
    total = sum(live.values())

    rows = []
    for p in FP.FILM_PROFILES:
        num = 0.0
        den = 0.0
        per = {}
        for a, w in live.items():
            if not _applies(p, a):
                per[a] = (None, "n/a")
                continue
            e, why = evidence(p, a)
            per[a] = (e, why)
            num += w * e
            den += w
        rows.append((p.name, num / den if den else 0.0, per))
    return live, dead, total, rows


def render(inf: dict) -> str:
    live, dead, total, rows = score_all(inf)
    n = len(rows)
    scores = sorted(r[1] for r in rows)
    mean = sum(scores) / n
    med = scores[n // 2]

    L: list[str] = []
    w = L.append

    w("# REALISM_SCORE.md — how much of a render is backed by measurement")
    w("")
    w("**Generated by `gen_realism_score.py`. Do not edit by hand.** "
      f"Schema v{FP.SCHEMA_VERSION}, {n} film stocks.")
    w("")
    w("## ⚠⚠ READ THIS BEFORE THE NUMBER")
    w("")
    w("**This is not a similarity-to-real-film score, and no such score exists "
      "in this project yet.** Nothing here has ever been compared against a "
      "scan of the same scene on the same stock. What this file measures is "
      "narrower and checkable: **the fraction of the pixel movement in a render "
      "that comes from a measured number rather than from an estimate.**")
    w("")
    w("A stock can score high and still look wrong, if the model itself is "
      "wrong. A stock that scores low is *guaranteed* to be partly invention. "
      "The score is a floor on honesty, not a ceiling on quality — and the way "
      "to turn it into a realism number is the ground-truth loop that queue "
      "rows **D1**, **D2a** and **D2b** are waiting on.")
    w("")
    w("## The number")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| corpus mean | **{mean * 100:.1f} %** |")
    w(f"| corpus median | **{med * 100:.1f} %** |")
    w(f"| best | **{max(rows, key=lambda r: r[1])[0]}** "
      f"{max(s for s in scores) * 100:.1f} % |")
    w(f"| worst | **{min(rows, key=lambda r: r[1])[0]}** "
      f"{min(scores) * 100:.1f} % |")
    w(f"| at or above 95 % | **{sum(1 for s in scores if s >= 0.95)}** of {n} |")
    w(f"| at or above 80 % | **{sum(1 for s in scores if s >= 0.80)}** of {n} |")
    w(f"| below 50 % | **{sum(1 for s in scores if s < 0.50)}** of {n} |")
    w("")

    w("## How it is computed")
    w("")
    w("```")
    w("score(stock) = SUM_a  I_a * e(a, stock)  /  SUM_a  I_a")
    w("```")
    w("")
    w("`I_a` — **measured**, not assigned. `realism_ablation.py` renders each "
      "sample stock twice per axis: once with the stock's own measurement, once "
      "with the fallback a stock lacking that measurement gets (the traced curve "
      "becomes the generic class curve, the measured σ(D) becomes the legacy "
      "√ law, the measured MTF becomes the class median, and so on). `I_a` is "
      "the mean CIE76 ΔE between the two.")
    w("")
    w(f"Measured at **{inf['scene_px'][0]}×{inf['scene_px'][1]}** "
      f"(**{inf['px_per_mm']} px/mm** on Super-35) over "
      f"**{len(inf['scenes'])}** scenes — {', '.join(inf['scenes'])} — on a "
      f"stratified sample of **{len(inf['sample'])}** stocks. "
      "⚠ **Grain and MTF influence are sampling-rate dependent**: the same "
      "measurement at 4K would weigh more. The rate is recorded here so the "
      "weights can be read against it.")
    w("")
    w("`e(a, stock)` — the stock's own `ParamSource` record for that axis: "
      "tier 1 traced or measured scores 1.0, tier 2 estimated 0.5, tier 3 "
      "assumed 0.1. Where no record names the parameter, the structural carrier "
      "decides.")
    w("")
    w("⚠ **THE ONE ASSUMPTION, NAMED.** `I_a` is what the measurement is worth "
      "*on the stocks that have it*. Applying that weight to a stock that does "
      "not have it assumes it would be worth about the same there. That is the "
      "only unmeasured step in this file.")
    w("")

    w("## Axes, by what the measurement is worth")
    w("")
    w("| axis | ΔE if unmeasured | weight | applies to | evidence there | "
      "biggest available win |")
    w("|---|---|---|---|---|---|")
    wins = []
    for a, infl in sorted(live.items(), key=lambda kv: -kv[1]):
        appl = [p for p in FP.FILM_PROFILES if _applies(p, a)]
        ev = [evidence(p, a)[0] for p in appl]
        mean_e = sum(ev) / len(ev) if ev else 1.0
        # ⚠ THE GAIN IS AVERAGED OVER THE WHOLE CORPUS, not over the stocks the
        # axis applies to, because the headline score is a corpus mean. An axis
        # that applies to 74 stocks can only lift the corpus by its share.
        share = len(appl) / len(FP.FILM_PROFILES)
        gain = infl * (1.0 - mean_e) / total * share
        wins.append((gain, a, infl, mean_e))
        w(f"| **{a}** — {inf['axes'][a]['what']} | {infl:.2f} | "
          f"{infl / total * 100:4.1f} % | {len(appl)} | {mean_e * 100:.0f} % | "
          f"+{gain * 100:.1f} points if fully measured |")
    w("")
    w("The last column is the whole roadmap: it is the score this corpus would "
      "gain if every stock carried a measurement on that axis. It already "
      "accounts for how much the axis moves a pixel, so it ranks research by "
      "what research is worth.")
    w("")
    top = sorted(wins, reverse=True)[:3]
    w("**In order, the three biggest: "
      + ", ".join(f"`{a}` (+{g * 100:.1f})" for g, a, _i, _e in top) + ".**")
    w("")

    if dead:
        w("## ⚠ Axes measured and NOT scored — the carrier moves no pixel")
        w("")
        w("| axis | what | why it is not scored |")
        w("|---|---|---|")
        for a, v in sorted(dead.items()):
            w(f"| `{a}` | {v['what']} | substituting its fallback changed the "
              f"render by ΔE {v['delta_e_median']:.3f} on "
              f"{v['stocks_measured']} sample stocks — **nothing on the render "
              f"path reads it** |")
        w("")
        w("⚠ **THIS IS A FINDING, NOT A GAP IN THIS REPORT.** A carrier that is "
          "traced, stored, emitted and read by no stage cannot make a picture "
          "more realistic. Either a stage should consume it or the harvesting "
          "effort behind it should be spent elsewhere — and until one of those "
          "happens, it must not be allowed to inflate a realism number.")
        w("")

    w("## What it would take to reach 95 %")
    w("")
    need = [(g, a) for g, a, _i, _e in sorted(wins, reverse=True)]
    running = mean
    steps = []
    for g, a in need:
        if running >= 0.95:
            break
        running += g
        steps.append((a, g, running))
    if running < 0.95:
        w(f"⚠ **Not reachable by measurement alone on the present axis set.** "
          f"Fully measuring every axis in the table above takes the corpus mean "
          f"to **{running * 100:.1f} %**. The remainder is the evidence ladder "
          f"itself: tier-2 sources cap at 0.5–0.8, so a corpus documented "
          f"entirely from secondary literature cannot reach 95 % — it needs "
          f"manufacturer sheets or own measurements on the axes that dominate.")
    else:
        w("Fully measuring these axes, in this order, reaches it:")
        w("")
        w("| # | axis | gain | running |")
        w("|---|---|---|---|")
        for i, (a, g, r) in enumerate(steps, 1):
            w(f"| {i} | `{a}` | +{g * 100:.1f} | {r * 100:.1f} % |")
    w("")

    w("## Per stock")
    w("")
    axes_order = [a for a, _ in sorted(live.items(), key=lambda kv: -kv[1])]
    w("| # | stock | score | " + " | ".join(axes_order) + " |")
    w("|---|---|---|" + "---|" * len(axes_order))
    for i, (name, sc, per) in enumerate(rows):
        cells = " | ".join("n/a" if per[a][0] is None else f"{per[a][0]:.2f}"
                           for a in axes_order)
        w(f"| {i} | {name} | **{sc * 100:.1f} %** | {cells} |")
    w("")
    w("A cell is the evidence weight on that axis: 1.00 is a manufacturer "
      "measurement, 0.10 is an assumption. **n/a** means the axis does not "
      "apply to that film — a dye image has no Callier scatter, a single silver "
      "record has no interimage effect — and those axes are left out of that "
      "stock's denominator rather than scored as gaps.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--assert", dest="strict", action="store_true",
                    help="fail if the influence cache is stale")
    ns = ap.parse_args()

    inf = load_influence(ns.strict)
    text = render(inf)
    out = Path(ns.out)
    out.write_text(text, encoding="utf-8")
    live, _dead, _t, rows = score_all(inf)
    mean = sum(r[1] for r in rows) / len(rows)
    print(f"[OK] wrote {out} -- corpus mean {mean * 100:.1f} %, "
          f"{len(live)} scored axes, {len(rows)} stocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
