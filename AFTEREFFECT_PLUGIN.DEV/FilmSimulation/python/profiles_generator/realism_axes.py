#!/usr/bin/env python3
"""Which realism axes this corpus is ALLOWED TO BE SCORED ON, and why.

⚠⚠ ONE DEFINITION, TWO CONSUMERS. `gen_realism_score.py` uses this to build the
denominator and `verify.py` uses it to guard that the denominator was not
hand-edited. The map below must exist in exactly one place: a second copy would
drift, and a drifted copy of THIS rule silently changes the headline realism
number without changing a single measurement.

THE RULE, STATED ONCE
---------------------

    An axis is DORMANT -- excluded from every stock's realism denominator --
    for exactly as long as NO STOCK IN THE CORPUS carries an INDEPENDENT
    record on it. One such record anywhere promotes the axis back into the
    denominator for all 191 stocks, automatically.

INDEPENDENT means a `ParamSource` at tier 1 or tier 2 whose status is
`traced`, `measured` or `stated` -- a number somebody outside this project put
on paper. It deliberately EXCLUDES `derived`, `estimated` and `assumed` at
every tier, because those are this project's own inference. An axis carried
entirely by our own inference is not a documented axis with poor coverage; it
is an OPEN QUESTION WITH A PLACEHOLDER IN IT, and scoring the placeholder as
evidence is the one failure the whole realism report exists to prevent.

⚠ WHY THIS IS NOT A HAND-MAINTAINED EXCLUSION LIST, WHICH WAS THE ALTERNATIVE.
The owner's constraint (2026-09-22) is that realism must be assessed from
technical datasheets only -- no scanned film exists to compare against. The
naive reading of that is "score only the axes datasheets print", as a literal
list of axis names. That reading is perverse: a Kodak patent or an SMPTE paper
giving interimage coefficients for a named stock is real, checkable, external
evidence, and a name list would throw it away because a patent is not a
datasheet. So the exclusion is keyed on the DATA, not on the source type. Find
one real record on a dormant axis tomorrow and the axis re-enters by itself.

⚠ THE SCORE GOES DOWN WHEN THAT HAPPENS, AND THAT IS CORRECT. Promotion adds a
weighted axis to the denominator that is mostly unevidenced, so the headline
drops. It should: a question that could not be asked has become a question
that can be asked and is mostly unanswered. A metric that only ever rises when
new evidence appears is measuring effort, not knowledge.

WHAT IS DORMANT TODAY (measured, not asserted -- run this file to print it)
--------------------------------------------------------------------------

  interimage     no sheet prints an interlayer coefficient. Patents and SPSE
                 papers might; none traced yet.
  callier        72 tier-2 DERIVED from the Q-factor law, 119 assumed. The law
                 is evidenced; no per-stock Q is.
  dye_matrix     117 records, every one tier-3 estimated.
  halation       84 tier-2 estimated from base/AHU class, none printed.
  grain_shape    zero records of any kind: sigma(D) shape is the legacy law
                 everywhere except the four traced VISION3 sheets behind it.

Usage:
    python3 realism_axes.py            # print the census
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Axis -> the `ParamSource.param` prefixes that evidence it.
#:
#: ⚠ PREFIXES, NOT EXACT NAMES, AND THE DIFFERENCE MATTERS. The evidence for
#: `tone_curve` is spread over `curves.r.gamma`, `curves.g.toe_x`,
#: `curves.b.shoulder_k` and a dozen more; a stock is evidenced on the axis if
#: ANY of them carries a real record. Matching one canonical name -- which is
#: what `gen_realism_score.AXIS_PARAM` does for the per-stock weight -- answers
#: a different and narrower question and must not be reused here.
#:
#: ⚠ `mtf` SPLITS INTO TWO AXES and the split is by prefix, not by file. The
#: transfer function (`mtf.f50_*`, `mtf.mtf_*`, `mtf.cycles_*`) is
#: `sharpness_mtf`; the adjacency terms (`mtf.adjacency*`, `mtf.edge*`) are
#: `edge_effects`, which the ablation measures separately because they move
#: different pixels. A bare `("mtf",)` prefix would merge them and would let 48
#: traced MTF stocks promote an edge-effect axis that has its own 12.
AXIS_EVIDENCE_PREFIX: dict[str, tuple[str, ...]] = {
    "tone_curve":           ("curves",),
    "grain_amplitude":      ("grain.rms_granularity", "grain.rms",
                             "grain.granularity"),
    "grain_shape":          ("grain.sigma_shape",),
    "sharpness_mtf":        ("mtf.f50", "mtf.mtf", "mtf.cycles"),
    "edge_effects":         ("mtf.adjacency", "mtf.edge"),
    "halation":             ("halation",),
    "callier":              ("callier_q",),
    "dye_matrix":           ("dye_matrix",),
    "dye_density":          ("dye_density",),
    "spectral_sensitivity": ("spectral",),
    "reciprocity":          ("reciprocity",),
    "interimage":           ("interimage",),
}

#: The statuses that mean "somebody outside this project wrote this number
#: down". Everything else -- derived, estimated, assumed -- is our own
#: inference and cannot promote an axis.
FOUND_STATUS = frozenset({"traced", "measured", "stated"})

#: The worst tier that still counts as independent. Tier 3 is excluded because
#: a tier-3 "traced" is a trace off a secondary reproduction of unknown
#: provenance; 117 tier-3 dye_matrix rows must not promote dye_matrix.
FOUND_MAX_TIER = 2


def axis_census(profiles) -> dict[str, dict]:
    """Per axis: how many stocks carry an independent record, and the full
    (tier, status) histogram behind that answer."""
    out: dict[str, dict] = {}
    for axis, prefixes in AXIS_EVIDENCE_PREFIX.items():
        hist: dict[tuple[int, str], int] = {}
        stocks = 0
        for p in profiles:
            found = False
            for s in p.param_sources:
                if not s.param.startswith(prefixes):
                    continue
                key = (s.tier, s.status)
                hist[key] = hist.get(key, 0) + 1
                if s.tier <= FOUND_MAX_TIER and s.status in FOUND_STATUS:
                    found = True
            if found:
                stocks += 1
        out[axis] = {
            "independent_stocks": stocks,
            "dormant": stocks == 0,
            "histogram": dict(sorted(hist.items())),
            "records": sum(hist.values()),
        }
    return out


def dormant_axes(profiles) -> frozenset[str]:
    """The axes that must be left out of the realism denominator today."""
    return frozenset(a for a, v in axis_census(profiles).items() if v["dormant"])


def main() -> int:
    sys.path.insert(0, str(HERE))
    import film_profiles as FP          # noqa: PLC0415

    cen = axis_census(FP.FILM_PROFILES)
    n = len(FP.FILM_PROFILES)
    print(f"realism axis census over {n} stocks "
          f"(independent = tier<={FOUND_MAX_TIER} and status in "
          f"{sorted(FOUND_STATUS)})")
    print()
    for axis, v in sorted(cen.items(),
                          key=lambda kv: (-kv[1]["independent_stocks"], kv[0])):
        flag = "DORMANT " if v["dormant"] else "scored  "
        print(f"  {flag} {axis:22s} {v['independent_stocks']:4d} stocks  "
              f"{v['records']:5d} records  {v['histogram']}")
    print()
    print("dormant today: " + ", ".join(sorted(dormant_axes(FP.FILM_PROFILES))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
