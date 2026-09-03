# AlgoControl.hpp — user-facing controls

Replaces the `struct AlgoControls { int dummy; };` placeholder (which still
carried a leftover PCA-white-balance docstring).

## Layout

```
AlgoControls
├── 21 live fields — mirror film_sim.RenderSettings one-for-one
├── bool       filmDamageEnabled   ← hard gate, DEFAULT false
└── FilmDamage damage              ← 17 fields, specified but not yet consumed
```

`FilmDamage` is nested rather than passed separately: one object to hand
around, one thing to serialise with a preset. The gate keeps the inert block
visibly inert — a reader of the live pipeline sees `filmDamageEnabled == false`
and knows the whole group is skipped. Checked **once per frame, not per pixel**.

`getFilmDamageDefault()` is exposed on its own so the panel can have a
"reset this group" button.

## Why the 21 mirror RenderSettings exactly

A C++ render with `getAlgoControlsDefault()` is directly comparable against the
Python reference. That comparability is what let Algo 02 be verified to 1e-15.
**Verified mechanically: 21/21 defaults match `film_sim.RenderSettings`.** If
either side changes, re-run that check or the reference stops being a reference.

## What is deliberately NOT here

**Film properties.** Those are `FilmProfile` data — 89 stocks of measured and
cited values. A control never replaces a profile number, it scales or overrides
it, and every such field says which.

**Stock-coupled damage.** Dye fade (per dye set), base yellowing and shrinkage
(per base material), scratch *colour* (depth decides which dye layers survive,
so it needs the tripack) and blob polarity (white on a print, dark on a
negative, inverted on reversal) all live in `AgingSpec` / `CoatingSpec`. Only
emulsion-*independent* damage is a control — a dust particle on VISION3 looks
like a dust particle on Svema.

## Two conventions worth knowing

**Sentinels.** `flare` and `vignette` default to **−1.0**, meaning "use the
stock's era-appropriate value". Passing **0.0** means "genuinely none". Losing
that distinction would silently discard per-era lens data.

**Damage rates are per SECOND, not per frame**, so defect density stays
physically constant when frame rate changes; the renderer converts using the
clip fps. `weaveAmpXUm`/`weaveAmpYUm` default to **0.0 = defer to the stock's
`TemporalSpec`**, which is already populated on all 89 stocks.

## Statelessness requirement for the damage generators

Every generator must be a pure function of
`(damageSeed, frameIndex, stageId, ordinal)` via a counter-based RNG, with a
bounded birth-frame scan for persistent objects. Any frame must be renderable
alone, out of order, on any thread — the same rule the v4 coating field already
follows, and the reason `frameIndex` is a control rather than internal state.
Set `frameIndex` from layer time × fps, **not** from a running counter.

## Verification

* `test_algocontrol.cpp` — 27 checks, all pass; asserts every default, that the
  nested damage block equals the standalone defaults, and that both sentinels
  are negative.
* Compiles clean under `g++ -std=c++14 -Wall -Wextra -pedantic`.
* Layout: `sizeof(AlgoControls)` printed by `probe.cpp` for ABI reference.
