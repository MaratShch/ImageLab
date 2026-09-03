# RESULT 2026-08-18e — C1: σ(D) chosen, wired, and deliberately limited to five stocks

`sigma_shape_toe/mid/dmax` had been in the schema for weeks — populated, validated, emitted to
C++, and **read by no renderer**. C1 was the decision about how to carry the shape and whether to
wire it. Both parts are now done, and the carrier was chosen by measurement rather than by taste.

## 1. The carrier, decided against the data we actually have

Every candidate was scored against the seven measured σ(D) samples per VISION3 sheet from
`RESULT_2026-08-17f` (dmin, 0.8, 1.0, 1.5, 2.0, 2.5, dmax), as a ratio error at those densities:

| Carrier | floats/profile | mean **max** error | mean **rms** error |
|---|---|---|---|
| legacy `sqrt(D − dmin + fog)` — what shipped | 0 | **245 %** | **127 %** |
| 3 anchors (dmin, 1.0, dmax) | 3 | 41 % | 18 % |
| **3 anchors + interior peak (value + density)** | **5** | **20 %** | **8.6 %** |
| array, 8 samples over the film's own span | 8 | 28 % | 11 % |
| array, 12 samples | 12 | 3.8 % | 2.0 % |

**Chosen: 3 anchors + an interior peak.** The 12-sample array wins on paper and is
**over-parameterised against seven measured points** — it is fitting the samples, not the film. Two
extra floats buy most of the accuracy that is actually available, cost nothing structurally, and
degrade to the three-anchor form by being zero. If a sheet is ever digitised densely enough to
justify an array, the sampler is **one function** (`grain_sigma`) and only that function changes.

The 4th anchor's density was also tested rather than assumed: 0.70 → 28 % max, 0.75 → 20 %,
**0.80 → 19.7 %**, 0.90 → 28 %. All four sheets peak at D ≈ 0.80, so that is where the anchor sits.

Two further fields exist because the anchor DENSITIES are part of the measurement, not of the
render: `sigma_shape_toe_at` and `sigma_shape_dmax_at`.

* The toe anchor is **not** the curve's dmin. On the VISION3 sheets the granularity plot's own
  left-edge plateau reads a mean **+0.051 D** above the sensitometric dmin, and Kodak's footnote
  says why: *"Sensitometric and Diffuse RMS Granularity curves are produced on different
  equipment."* **That is the "whether to fix the calibration" half of C1, and the answer is
  neither fix nor ignore — carry it.** The offset is recorded as the anchor's own density, so the
  shape is evaluated where it was measured. Method rule 14's principle: record the conflict, never
  average it.
* The dmax anchor is where the **trace stopped**, not the curve model's asymptote.

## 2. The wiring, and why it touches five stocks and not 155

`_grain_v2` fills the anchors heuristically for **137 of 155** profiles, and **both branches are
known wrong in sign**:

| branch | fills | measurement says |
|---|---|---|
| colour negative | 0.4 / 1.0 / 1.2 (σ rises) | **falls** — 4 VISION3 sheets, and Kodak's 1985 SMPTE paper in print |
| reversal | 0.7 / 1.0 / 0.5 (σ falls) | **rises ~20×** — EKTACHROME 100D 5285's own sheet |

Wiring the shape in unconditionally would have replaced one wrong grain law with another on 137
stocks while fixing 5. So the schema gained **`sigma_shape_measured`**, and the renderer honours a
shape only when it is set — which is exactly the five stocks traced from a vendor plot:

| stock | toe | mid | dmax | interior peak |
|---|---|---|---|---|
| `KODAK_VISION3_50D_5203` | 0.39 | 1.00 | 0.63 | 1.30 @ D 0.80 |
| `KODAK_VISION3_250D_5207` | 0.59 | 1.00 | 0.57 | 1.28 @ D 0.80 |
| `KODAK_VISION3_200T_5213` | 0.41 | 1.00 | 0.58 | 1.23 @ D 0.80 |
| `KODAK_VISION3_500T_5219` | 0.67 | 1.00 | 0.55 | 1.31 @ D 0.80 |
| `KODAK_EKTACHROME_100D_5285` | 0.15 | 1.00 | 3.10 | none — monotone rise |

`TASMA_FN_64` is **not** flagged, though its toe was measured: its own comment records that the
dense bin "came out BELOW mid, which negatives do not do — leakage again; dmax capped at 1.0
(direction of the data, not its face value)". A capped estimate is not a measurement. The twelve
Soviet class estimates are not flagged either.

**The heuristic's output is now explicitly inert** — filled, visible in the C++ struct, and read by
nothing. That is the correct status for a heuristic whose both branches have a counter-example.

## 3. Level preserved, shape changed — the separation that keeps this safe

`grain_sigma` returns **1.0 at D = 1.0** by construction, and `film_sim.py` multiplies it by the
legacy law's own value at D = 1.0. Consequences:

* **150 unmeasured stocks render bit-for-bit as before.** Asserted over a 36-point density sweep
  in float32 for every one of them: max deviation **< 2 × 10⁻⁶**.
* For the five measured stocks the **density dependence** changes and the amplitude at D = 1.0 is
  **identical** to what it was.

What that does to 5203 (green record), as an amplitude ratio against the old law:

| D | 0.60 | 0.80 | 1.00 | 1.50 | 2.00 | 2.60 |
|---|---|---|---|---|---|---|
| new/old | 0.69× | **1.60×** | 1.00× | 0.65× | 0.47× | **0.33×** |

So the engine was **~3× too grainy at dmax** and **~1.6× too quiet at D ≈ 0.8**, where the real
maximum sits. For the reversal stock 5285 it runs the other way: 0.79× at D 0.6 rising to **1.43×**
at D 2.6.

## 4. ⚠ ONE THING DELIBERATELY NOT DONE — and it needs an owner decision

The legacy law's value at D = 1.0 is `sqrt(1 − dmin + fog)`, typically **0.77–0.95, not 1**. So in
the renderer the stored `rms_granularity` has never been the rms *at D = 1.0* — it has been that
figure times an accidental per-stock factor. Normalising it away would make `rms_granularity` mean
exactly what the datasheets print.

**It was not bundled into this change**, for a reason that is not timidity: it is a **level** change
of up to **+30 %** on grain amplitude for every stock, and several stored rms values are described
in their own comments as *"pipeline-calibrated"* — i.e. tuned against the current, un-normalised
behaviour. Fixing the normalisation without re-checking those would double-count the correction.

The decision, stated plainly:

* **(a) leave it** — grain amplitude unchanged everywhere; `rms_granularity` keeps a per-stock
  fudge factor between the datasheet figure and the render.
* **(b) normalise, and re-derive the pipeline-calibrated rms values** — the datasheet figure then
  means what it says. Costs an audit of every rms marked "pipeline-calibrated" (5247 was one, and
  its printed value has since been found).

## 5. Both renderers, one definition

`film_profiles.grain_sigma()` (Python) and `FilmGrainSigma()` in the generated `film_profiles.hpp`
implement the same law, including the flat hold outside the traced range. Cross-checked by
compiling the emitted header and comparing against Python over nine densities and both code paths:
**worst divergence 5.4 × 10⁻⁷**.

The C++ helper carries the calling convention in its own comment, because the trap is real: a
renderer that today computes `sqrt(D − dmin + fog)` must multiply this result by that same
expression at D = 1.0, or it silently takes decision (b) above by accident.

## 6. State

`231 PASS / 2 FAIL` (both baseline), compile clean on 18 TUs, `film_names.txt` MD5 unchanged.
Schema **v7 → v8** — the struct layout moved, so a consumer of the generated C++ needs to know.

Nine new checks, including the bit-for-bit legacy regression, "only these five are flagged", "no
unflagged profile can produce a shape", the sampler's 1.0-at-D-1.0 contract, the flat hold, and
that every stored interior peak actually exceeds the mid anchor.

**Owner action for the plugin:** `GrainSpec` gained five fields, so the generated struct changed
size. Rebuild; and if the C++ grain path wants the measured shapes, call `FilmGrainSigma()` per the
convention in its comment.
