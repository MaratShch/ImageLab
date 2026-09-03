# Engine convergence — 2026-08-28

**Scope:** algorithm only. No database change, no schema change, no generated
artefact change. Schema stays 18, profile count stays 161.

**Base:** the 2026-08-27 five-archive delivery.

---

## 0. What this session was for

Four defects, all of the same family: **an optimisation or a fix applied to one
instruction-set path and not the other**, or a limit that silently changed the
model instead of refusing. The project's own lesson names the family —

> an optimisation applied to one instruction-set path is a correctness
> divergence unless it is applied to both **or explicitly documented as a mode
> difference**

— and this session closes three instances of it, documents a fourth as a
deliberate mode difference, and finds a fifth that had never been recorded.

---

## 1. `AlgoTypes.hpp` split — the shipped state was not the buildable state

**Defect.** One shared header hard-coded `AlgoType = double`. All 17 vector
translation units carry `static_assert(sizeof(AlgoType) == 4)`, so the AVX2
delivery could not compile against the header it shipped with. Reproduced
exactly on this base:

```
AVX2/AlgoSeparableBlur.cpp:62:32: error: static assertion failed:
    the AVX2 path requires AlgoType to be a 32-bit float
    note: the comparison reduces to '(8 == 4)'
```

Every AVX2 measurement in the project's history was therefore taken against a
hand-edited working copy.

**Fix.** Three files, selected by one project-wide `ALGO_TARGET_AVX2`:

| File | Role |
|---|---|
| `AlgoTypes.hpp` | selector + everything type-independent + the guards |
| `AlgoTypes_Scalar.hpp` | `AlgoType = double`, `ALGO_ALIGN_ELEMS = 4` |
| `AVX2/AlgoTypes.hpp` | `AlgoType = float`, `ALGO_ALIGN_ELEMS = 8`, `#error` without the define |

`AlgoTypes.hpp` now carries `static_assert`s that fire at **every** include site
if either path's invariant is broken, plus one asserting `HighPrecType` is
`double` in both. Auto-detecting `__AVX2__` was reconsidered and re-rejected:
compiling the scalar project with `/arch:AVX2` is normal practice and would
silently demote the double reference path.

**Verified.** Both paths build. The negative test (an AVX2 source compiled
without the define) fires. Scalar output **bit-identical across all 161
stocks** — the split changed nothing.

---

## 2. Wide-σ blur, scalar path — a clamp that was not a clamp

**Defect.** `ALGO_BLUR_MAX_HALF_TAPS = 64` reads as a cost control. Past σ = 16
it changes the filter: the truncated taps are renormalised, so there is no
brightness artefact and nothing looks wrong, while every wide lobe collapses
onto the same ~37 px box (64/√3, the standard deviation of a *uniform* kernel).

**Fix.** Above `ALGO_BLUR_SIGMA_EXACT_MAX = 16` the scalar path resamples,
blurs at reduced resolution with a complete kernel, and reconstructs. At or
below 16 the direct kernel is untouched, because it is exact there and the
exact method wins wherever it is available.

Cell width is the **rational** `R = n / loN`, not an integer block, so the cells
tile one period exactly for any extent. Variance compensation accounts for all
three filters in the cascade:

```
area decimation, cell width R        variance (R^2 - 1)/12
Gaussian sigmaLo on the coarse grid  variance R^2 * sigmaLo^2
linear reconstruction, base 2R       variance R^2 / 6
=>  sigmaLo^2 = ( sigma^2 - (R^2 - 1)/12 - R^2/6 ) / R^2
```

**Validated against an exact periodic Gaussian built independently in double** —
not against the direct kernel, which is the thing under correction.

| σ | max err before | max err after | effective σ before | after |
|---|---|---|---|---|
| 16.0 | 1.36e-05 | 1.36e-05 | 15.99 | 15.99 |
| 20.0 | 1.83e-04 | 8.48e-04 | 19.86 | **20.05** |
| 40.0 | 1.41e-02 | **3.21e-05** | 31.18 | **39.99** |
| 100 | 3.53e-02 | **3.21e-07** | 36.21 | 71.66 |
| 300 | 3.86e-02 | **3.10e-06** | 37.12 | 74.07 |
| 820 | 3.90e-02 | **1.31e-05** | 37.22 | 73.74 |

Constant-field response exact (≤ 2.8e-16) at every σ — the partition-of-unity
property, and not a nicety: this resampling sits in front of halation's
energy-conserving `blur(above) − above`.

At σ = 20 the pointwise error is larger than the direct kernel's while the
*width* is better (20.05 against 19.86). Width is the physically meaningful
quantity, and past σ 24 the direct path degrades without limit, so the threshold
stays at the exactness boundary.

**Blast radius predicted from the database first, then measured.** At 256 px on
super35 the veiling-flare lobes (1500 / 6000 / 20000 µm) exceed 16 px on 34
stocks; halation lobes do not. Predicted 34, **observed 34, zero unexplained**.
At 1024 px the predicted set becomes 50 (34 flare + 28 halation, overlapping).

---

## 3. AVX2 resample geometry — the last blur divergence

**Defect.** The vector path averaged exact `k × k` blocks and wrapped on the row
index. When `k` does not divide the extent this draws more samples than the
period holds: `k = 5` on 512 gives `loW = 103`, and `103 × 5 = 515` samples over
a 512 period, so three columns are double-counted and a seam appears at the
wrap.

**Fix, and it is structural rather than a patch.** The planner and the cell
weights moved into the **shared header** `AlgoSeparableBlur.hpp`, namespace
`AlgoBlurDetail`. Both paths now call the same `planBlur` and the same
`cellWeights`, so they cannot choose different decimation factors, different
reduced extents or different weights. Two copies of one rule *was* the defect;
one copy cannot drift.

The vector path keeps its two-pass structure — the restructuring from a `k × k`
gather to accumulate-rows-then-decimate was a large measured win and is not
undone. Only the weights change.

A latent overrun was fixed in passing: the old vertical accumulator wrote
`loW * k` samples into a `loPitch`-wide row and survived only because the next
reduced plane was rewritten afterwards. It now uses an explicit full-width
region, with the scratch budget checked rather than assumed.

**Cross-path agreement, max |scalar − AVX2| on the texture field, 512 px:**

| σ | before | after |
|---|---|---|
| 20 | 1.235e-02 | **3.09e-07** |
| 300 | — | 6.21e-06 |
| 820 | — | 8.20e-06 |
| **worst, all σ** | 1.235e-02 | **8.20e-06** |

Target was ≤ 1e-5. Met.

---

## 4. Characteristic-curve table — three sites, one header

**Defect.** The engine evaluates a difference-of-softplus curve in three places
— stage 08, stage 08b, stage 13's print curve — and the two paths had drifted
into different mathematics at two of them.

**Measured, per stage, deterministic chain only (damage and grain off), 128 px:**

| stage | scalar vs AVX2, |Δmean| |
|---|---|
| 02 – 12 | 1e-07 – 1e-06 (ordinary float rounding) |
| **13** | **2.4e-03 – 3.2e-03** |
| 17 (output) | 1.0e-03 – 1.8e-03 |

So the largest cross-path disagreement in the engine was never precision.

**Fix.** New shared header `AlgoCurveLut.hpp`: table, construction, scalar
lookup and — under `ALGO_TARGET_AVX2` — the vector gather lookup. Both paths and
all three stages now tabulate the same function over the same domain with the
same interpolation.

**Domain and size were measured, not assumed.** The ten knees of padding
inherited from the AVX2 file were the *dominant* error, larger than the
interpolation the size was chosen to control:

| configuration | interpolation | clamp |
|---|---|---|
| 2048 entries / 10 knees | 8.05e-06 D | 1.26e-04 D |
| **4096 entries / 16 knees** | **4.24e-06 D** | **3.12e-07 D** |

Better on both axes. The residual 4.24e-06 D is smaller than the exact
difference-of-softplus model's **own** worst non-monotonicity (6.61e-06 D at the
extreme toe, where `shoulder_k > toe_k` makes the shoulder ramp decay more
slowly than the toe ramp), and three orders below the ~1e-03 D at which the
source datasheet curves can be read.

`test_curve_lut.cpp` asserts across all 161 stocks / 483 curves: monotonicity
(table's worst backward step equals the exact curve's, to the digit — the table
introduces none), interpolation accuracy, and clamp exactness at ±1e6.

**Scalar cost, 1024 px, best of 3:**

| | before | after | saved |
|---|---|---|---|
| stage 08 | 223.25 ms | 53.24 ms | −170.01 ms |
| stage 08b (`CINESTILL_800T`) | 172.46 ms | 30.98 ms | −141.48 ms |
| stage 13 | 167.20 ms | 26.82 ms | −140.38 ms |
| frame, `AGFA_APX_25` | 1107.50 ms | 780.25 ms | **−327.25 ms** |
| frame, `CINESTILL_800T` | 2562.57 ms | 1983.02 ms | **−579.55 ms** |

**Output change, all 161 stocks at 256 px:** max mean shift 1.20e-06, median
2.50e-08. Bounded and explained: every stock changes because every stock's curve
is now tabulated.

---

## 5. AVX2 stage 13 — a table was tried and REJECTED on measurement

The obvious symmetric move was to tabulate stage 13 in the vector path too. It
was implemented, measured, and **reverted**.

**Stage 13 alone, 1024 px, best of 3:**

| stock | fast softplus | table | verdict |
|---|---|---|---|
| `CINESTILL_800T` | 13.37 ms | 16.72 ms | fast +20.0 % |
| `KODAK_VISION3_500T_5219` | 14.27 ms | 16.91 ms | fast +15.6 % |
| `AGFA_APX_25` | 13.95 ms | 18.09 ms | fast +22.9 % |

The reason is the gather. Stage 08's own notes already carry the per-call
figures — about 0.54 ns per sample for a table lookup against roughly 0.22 ns
for the fast exponential — so a table only wins where it replaces *several*
transcendentals at once. Stage 08 re-evaluates the curve through the interimage
fixed point and wins; stage 13 evaluates once per sample and loses.

**Fidelity cost of keeping the approximation:** the two paths differ at this
stage by 2.4e-03 – 3.2e-03 in plane mean. One part in 255 is 3.9e-03, so the
difference sits **below the quantisation of eight-bit output**. The vector path
stays monotonic across the whole operating range and produces no discontinuity,
no invalid value and no artefact.

This is therefore a **documented mode difference**, recorded at the site in
`AVX2/Algo_13_Sim.cpp`: the scalar path is the higher-precision reference, and
the vector path trades an invisible amount of precision for a measured 15–23 %
on this stage.

**Stage 14 carries the same kind of approximation** and is left alone for the
same reason, now that it is the largest remaining cross-path term
(1.83e-03 mean, likewise under 1/255). It is recorded rather than changed.

---

## 6. `AlgoControls` documentation — the fourteen-item standard

All 40 controls (23 top-level + 17 `FilmDamage`) documented against the code
that consumes them, not against their names. Struct layout verified
bit-identical: `sizeof` 296 / 136, every `offsetof` unchanged.

Findings that contradicted the previous comments, documented rather than
silently corrected:

| Finding |
|---|
| **No enforced upper bound on any numeric control.** Every `RANGE 0..N` was advisory; stages only floor with `MAX_VALUE(x, 0)`. `generations` (capped at 4) is the sole exception. |
| **`filmDamageEnabled` defaults to `true`**, while the header claimed `false` and the requirement says a clean render must be the default. Owner decision; documented as implemented. |
| **8 of 17 `FilmDamage` fields are unconsumed**: `scratchTransport`, `scratchHandling`, `processingQuality`, `dryingMarks`, `storageSeverity`, `colourVeil`, `flickerStops`, `scannerArtifacts`. Nine *are* live, at stages 09b, 15 and 16. |
| **`frameIndex` range contradiction** — header said `>= 0`; the code states, and handles, negatives. |
| **`halationScale` scales gains only, not radii**, despite the header claiming both. Halo *size* is not adjustable from the controls at all. |
| **`grainScale` is gate-only at stage 14** — print grain is added at a literal gain of 1. |
| **`reseau` auto-disable is silent** — the header promised a warning; none is emitted. |
| **`AlgoTemporalFlicker.hpp` cites `flickerBaseHz` and `flickerColourSpread`** — neither field exists anywhere. |
| **`AlgorithmMain.cpp` still labels stages 15 and 16 "STUB / NOT YET MODELLED"** — both are implemented and active. |
| **One real scalar/AVX2 behavioural difference**: stage 04's per-pixel vignette loop is `HighPrecType` in scalar, deliberately narrowed to `AlgoType` in AVX2. |

⚠ **The archive's control set is behind the owner's tree.** `exposureTimeS` and
`scannerSpecular` are used by the owner's `AlgorithmMain.cpp` but appear
**nowhere** in archives 3 or 4 — nor does stage 12b Callier, nor
`AlgoReciprocity`. The delivered header carries both fields, placed per the
handoff's documented order (after `exposureStops`), and their range/default/stage
items are marked UNVERIFIED because the consuming stages are not in this base.
Confirmed compiling in the owner's project.

---

## 7. New test harnesses

| Harness | What it proves |
|---|---|
| `test_dbhash.cpp` | Whole-database fingerprint: per stock an exact bit hash, a 16-bit-quantised hash, mean and max. Turns "output is unchanged" into a diff. |
| `test_blur_pyramid.cpp` | Blur against an exact periodic Gaussian built independently in double. Accuracy, constant-field partition of unity, effective σ, and a cross-path dump. |
| `test_curve_lut.cpp` | Curve table across all 161 stocks: monotonicity against the exact curve's own, interpolation error, clamp exactness. |
| `test_stage_parity.cpp` | Per-stage scalar-vs-AVX2 means under `ALGO_RETAIN_ALL_STAGES=1`. This is what located the stage-13 divergence; a whole-frame figure only says the paths disagree. |

---

## 8. Verified state at close

| Item | Value |
|---|---|
| Schema / profiles | 18 / 161 — unchanged |
| Generated artefacts | 23, regenerated, **content-identical** to the shipped copies |
| `verify.py` | 408 PASS / 1 FAIL (saturation hierarchy, the known baseline) |
| Scalar build | clean, `-Wall -Wextra` |
| AVX2 build | clean, `-Wall -Wextra`, project-wide define |
| `test_full` | 0 failures / 161 stocks, both paths |
| `test_curve_lut` | PASS |
| `test_blur_pyramid` | PASS, both paths |
| Non-finite outputs | 0, both paths |

---

## 9. NOT done, and why

| Task | State |
|---|---|
| **7 — driver restructure** (plane cursor + `PreparedProfile` + `QualityPolicy` + `simMode` + O4 elision) | **Not started.** Its acceptance criterion is bit-identity of the Full preset across all 161 stocks, and that cannot be established here: the base tree is behind the owner's, which carries stages this base does not have. A restructure verified against the wrong driver would prove nothing. It needs the owner's current `AlgorithmMain.cpp`. |
| **8 — Lite preset** | Blocked on 7. |
| **9 — emulsion physics into `PreparedProfile`** | Blocked on 7. Ranking unchanged; the `DevelopmentProgress` push/pull coupling remains the best candidate, and base-thickness→halation is still blocked on the missing antihalation-layer field. |
| **`sceneDuv`** | Approved in principle, deliberately deferred to the same layout change as `simMode` and `QualityPolicy`, so the structure changes once rather than three times. |
| **Film-identifier freeze** | Not re-applied. It is a Python-side change and this session made no database change; re-applying it here would produce a `film_ids.lock` that the owner's tree may already have. |

---

## 10. Files changed

**Engine, scalar:** `AlgoTypes.hpp` (now a selector), **new**
`AlgoTypes_Scalar.hpp`, **new** `AlgoCurveLut.hpp`, `AlgoSeparableBlur.hpp`
(+`AlgoBlurDetail`), `AlgoSeparableBlur.cpp`, `Algo_08_Sim.cpp`,
`Algo_13_Sim.cpp`, `AlgoControl.hpp` (comments + two fields).

**Engine, vector:** **new** `AVX2/AlgoTypes.hpp`, `AVX2/AlgoSeparableBlur.cpp`,
`AVX2/Algo_08_Sim.cpp`, `AVX2/Algo_13_Sim.cpp` (documentation only, after the
measured revert).

**New harnesses:** `test_dbhash.cpp`, `test_blur_pyramid.cpp`,
`test_curve_lut.cpp`, `test_stage_parity.cpp`.

**Python / database / generated artefacts:** unchanged.
