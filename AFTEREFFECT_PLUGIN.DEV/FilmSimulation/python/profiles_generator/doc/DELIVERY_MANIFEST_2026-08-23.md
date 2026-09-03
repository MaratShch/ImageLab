# DELIVERY 2026-08-23 — what is in this archive and what to do with it

Three queue items closed in this pass: **C1e** (per-layer VISION3 grain), **C8** (reciprocity
wired — the last inert data family), **C2b + C24** (nine more colour MTF sheets, and the f50
estimating rule replaced where the measurements reach).

**Build state at the moment this archive was written**

| | |
|---|---|
| `build.py --root <project>` | **OK — 0 failures, 0 warnings** |
| audits | **11 registered, all green** |
| `verify.py` | **303 PASS / 1 FAIL** — the one FAIL is the saturation-hierarchy ordering you asked to leave alone; `build.py` compares the FAIL *set* to a baseline, so a NEW failure fails the build |
| C++ compile | clean on **18 TUs**, `g++ -std=c++14 -Wall -Wextra`, gated on exit 0 **and** zero bytes of output |
| database | **159 stocks, 9 print stocks, 14 gauges, schema v10** |
| `film_names.txt` | MD5 **`e8dc2cb9b594897ce9748fae67f0ffb2`** — **UNCHANGED**, so no enum shift and no ListBox movement |

---

## 1. Unzip over the project root

The archive mirrors your layout, so it can be extracted straight over `C:\WORK\PYTHON.TST`:

```
PYTHON/profile_generator/**     the generator: python, doc/*.md, its copy of the generated C++
*.cpp  *.hpp                    project-root plugin sources (8 changed/new) + the synced generated C++
AVX2/Algo_08_Sim.cpp            the AVX2 variant of stage 8
```

## 2. ⚠ ONE COMPILE-BREAKING CHANGE, ON PURPOSE

`AlgoStage08_CharacteristicCurve` takes a fifth trailing argument:

```cpp
const HighPrecType logEShift[3]     // reciprocity, in decades; all zeros = inert
```

The declaration (`AlgoCharacteristicCurve.hpp`), both definitions (`Algo_08_Sim.cpp` and
`AVX2/Algo_08_Sim.cpp`) and both call sites (`AlgorithmMain.cpp`, `profall.cpp`) are updated in
this archive. **Any other caller in your tree will fail to compile — that is intended**: a
silently ignored shift would render the wrong density with no symptom.

`AlgoControls` also gains `exposureTimeS` (default `0.0`). Every caller in the tree uses
`getAlgoControlsDefault()`, so no positional initialiser breaks.

**Both variants of stage 8 were compiled here**: the scalar TU with `AlgoType = double`, and the
AVX2 TU with `AlgoType = float` (its `static_assert` requires the 4-byte type) — thank you for
sending `FastAriphmetics.hpp` and `FastAriphmeticsAVX.hpp`, which is what made that possible.
`interimage_parity.py` was also re-run against the float build and passes (worst 5.29e-05), so
the switchable typedef is still switchable.

## 3. New in the plugin

| file | what |
|---|---|
| `AlgoReciprocity.hpp` | **NEW.** The reciprocity law: per-channel log-exposure shift for a stated shutter time. Header only, computed once per frame, consumed by stage 8. Mirrors `film_sim.reciprocity_log_shift()`; `cpp_parity.py` compares the two on every build |
| `AlgoControl.hpp` / `.cpp` | `exposureTimeS` + its default, documented against the same physics |
| `AlgoCharacteristicCurve.hpp` | the new stage-8 parameter |
| `Algo_08_Sim.cpp`, `AVX2/Algo_08_Sim.cpp` | the shift added to the logarithm (one hoisted constant per channel) |
| `AlgorithmMain.cpp` | computes the shift before stage 8 and passes it |
| `profall.cpp` | call site updated (passes zeros — the profiler runs the default control set) |

**A UI control is still yours to add.** `exposureTimeS` is reachable only when the panel exposes
it: 0 = off, useful range 1e-5…3600 s on a logarithmic scale. Until then it defaults to 0 and the
stage is inert, which reproduces every earlier render bit for bit.

## 4. Data that changed — rebuild the plugin, data only

**Schema v10 unchanged, `film_names.txt` unchanged.** No enum or ListBox movement in this
delivery; only stored values moved.

| stocks | field | what |
|---|---|---|
| 5219 | per-layer grain rms | **5.92 / 6.60 / 17.84** (was 7.26 / 6.60 / 8.58 from the tier-2 ladder). Blue grain σ on screen **2.33 → 3.60 per 255, ×1.52** |
| 5207, 5203 | per-layer blue rms | **8.92** and **4.71** (b/g 2.123 and 1.813, measured) |
| 5217, 5218, 5245, 5248, 5279 | MTF triple + adjacency + rolloff q | measured: 33.9/58.1/67.4 · 37.6/54.6/69.7 · 37.2/83.8/100.5 · 37.4/75.1/111.2 · 41.1/73.1/76.1. Estimates had been **1.12–1.72× too sharp in red** and **0.70–0.83× too soft in blue** |
| 5205, 5293 | MTF | measured green and blue, **family-anchored red** (their sheets emit red in fragments) — mixed provenance, `mtf_measured` deliberately unset |
| 5203, 5207, 5213, 5219, 5246 | MTF red f50 | **re-anchored to 36.0 cycles/mm** — see §5 |
| 15 stocks | reciprocity | measured `ReciprocityTable` read from each stock's own sheet: 5205, 5217, 5218, 5219, 5201, 5246, 5274, 5279, 5248, 5231, 5247, F-125 8532, F-500 8572, ETERNA Vivid 8547, VISTA 200. Total measured tables **6 → 21** |

⚠ **Render impact is not cosmetic this time.** Bar-sweep target, grain and flare off, worst
channel delta: **5203 45.7/255**, 5248 22.9–26.8, 5217 21.8–26.1, 5219 7.8–8.8, measured at both
48 and 193 px/mm. Unlike the C13 change, this one is visible at preview size, because red now
moves by tens of cycles/mm rather than a few.

## 5. The one class estimate in this delivery, stated plainly

The old f50 rule scaled all three records from one number by a fixed layer ratio
(`f50_r ≈ 0.78 × f50_b`). Seven per-record measurements say that **form** is wrong:

```
red    32.1  33.9  35.4  37.2  37.4  37.6  41.1     mean 36.4, spread +-13 %
green  49.7  54.6  58.1  68.8  73.1  75.1  83.8     spread 52 %
blue   55.5  67.4  69.7  74.0  76.1 100.5 111.2     spread 70 %
```

Red f50 does not scale with the stock's sharpness at all, so no value of `k` fits. Five modern
Kodak cine stocks therefore carry **red = 36.0 exactly**, tagged `[T2-family]`: a class estimate
inside a family measured seven times, not a measurement.

⚠ **Scope is deliberately narrow.** Only stocks whose stored blue lies inside the measured
55–111 cycles/mm range. `EASTMAN_EXR_500T_5296` (blue 42) and every pre-1990 stock are excluded,
as is every other manufacturer — the corpus holds no per-record MTF outside Kodak cine.
`verify.py` asserts 5296 keeps its own 30.0, so a later "finish the family" pass fails instead of
guessing. Green and blue were left at their estimates because the measured blues run 0.96–1.43×
their stored values with no consistent factor: only red is a constant.

## 6. Open items you may want to know about

* **63 colour stocks still carry an estimated f50 triple**, including every non-Kodak stock. The
  corpus's only non-Kodak MTF sheet (Agfa Vista) prints one visual-weighted curve, not three
  records, so it cannot fill a triple — it does say that stock's estimate is ~1.26× too sharp.
* **Two data defects are recorded rather than patched, because the documents are not in this
  working copy**: the Agfa/Konica Schwarzschild exponents were fitted under a different reading
  of a printed correction (24 % light), and Kentmere's stored stops do not reproduce the formula
  their own source string quotes (~40 % light). Both are written out next to the code that would
  change. `PDF/PROFILES` here holds only AGFA, FERRANIA, FUJI, GEVAERT, KODAK, RETRO and SVEMA.
* **`adjacency_um` is contradicted on five stocks now** (the overshoot peaks at ~90–150 µm while
  the field stores 13–22 µm). Untouched — queue **C2c** owns it, because the fix depends on what
  the renderer means by the field.
* **5245 and 5293's stored limiting resolution (100 lines/mm) conflicts with their measured blue
  f50** (100.5, 114.6). Recorded in `_RESOLVING_POWER`; the guard now compares the green record,
  since ISO resolving power is a composite three-layer reading while f50 is per record.

## 7. Where to read next

`PYTHON/profile_generator/doc/PROGRESS.md` — one screen, current state, updated in this pass.
Then `RESULT_2026-08-23_c1e_c8.md` and `RESULT_2026-08-23b_c2b.md` for this delivery's two
reports. Every hand-written document in `doc/` was revised for staleness in this pass, not merely
appended to; four dated `RESULT_*` files carry a supersession banner where a later measurement
overturned something they claimed, with their bodies left verbatim as the audit trail.
