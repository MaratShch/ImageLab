# Functional parity plan: Python ↔ C++, and database utilisation

**2026-09-03 · schema v24 · 175 stocks · additive only — nothing is removed from either side.**

Companion to `DB_ALGORITHM_COVERAGE_2026-09-03.md`, which is the field-by-field census. This
document is the *flow* comparison and the acquisition list.

---

## 0. ⚠ The stage numbering diverges after 14, and the docs do not say so

Both pipelines share stages **1–14 with identical numbering and identical order**. After that they
are different documents:

| | Python `film_sim.py` | C++ `Algo_NN_Sim.cpp` |
|---|---|---|
| 15 | **sRGB encode, dither, quantise** | **gate weave** |
| 16 | — | machine-side defects (gate dirt, one-frame dirt, splices) |
| 17 | — | the single final clamp, narrow to storage type |

⚠ **Reading "stage 15" across the two codebases means two unrelated things.** That is a live trap
for anyone porting a stage by number, and it should be fixed in the docstrings whatever else is
decided.

⚠ **Python's stage 15 must NOT be added to C++.** Python is a standalone renderer that writes PNG
files; the C++ side is a plugin that hands float buffers back to After Effects / Premiere, and the
host owns the transfer function. Adding an sRGB encode there would double-encode every frame. This
is the one asymmetry that is correct by design.

---

## 1. Already implemented and used — the shared core

Stages **1–14 including every sub-stage** exist on both sides and consume the same fields:

| stage | Python | C++ | principal fields |
|---|---|---|---|
| 2 / 2b | relative exposure, taking matrix | `AlgoStage02` / `02b` | `taking_matrix` |
| 3 / 3b | colour balance, veiling flare | `AlgoStage03` / `03b` | `balance_kelvin`, `default_flare` |
| 4 / 4b | coating unevenness, vignette | `AlgoStage04` | `coating_sigma`, `coating_corr_*`, `default_vignette` |
| 5 | halation | `AlgoStage05` | `radii_um`, `weights`, `gain_r/g/b`, `threshold_stops` |
| 6 / 6b | emulsion MTF, corner defocus | `AlgoStage06` / `06b` | `f50_r/g/b`, `adjacency`, `adjacency_um`, `buckle_mtf_loss` |
| 7 | emulsion record collapse, reseau | `AlgoStage07` | `log_s_r/g/b/pan`, `spectral_weights`, `reseau` |
| 7c | reciprocity | `AlgoReciprocity` | `schwarzschild_p_*`, `onset_s`, `times_s`, `stops_correction` |
| 8 / 8b | characteristic curve, interimage | `AlgoStage08` | all 18 `ToneCurve` fields, `a_rg…a_bg`, `iterations`, `density_weighting` |
| 9 | DIR couplers | `AlgoStage09` | `strength`, `radius_um`, `edge_strength`, `edge_um` |
| 10 / 10b | scanner MTF, edge fog | `AlgoStage10` / `10b` | `misregistration_um`, `edge_fog_density/mm` |
| 11 | grain | `AlgoStage11` | grain core + the seven `sigma_shape_*` via `grain_sigma()` / `AlgoGrainAmpBuild` |
| 12 / 12b | dye matrix, Callier | `AlgoStage12`, `AlgoCallier` | `dye_matrix`, `callier_q` |
| 13 | duplication, print | `AlgoStage13` | `PrintStock.curves`, `dye_matrix`, `mtf_f50`, `grain_rms` |
| 14 / 14b / 14c | transmittance, reseau rebuild, silver tone | `AlgoStage14` / `14b` / `14c` | `silver_tone`, `base_tint` |

**This is the majority of the render and it is genuinely at parity.** `cpp_parity.py` holds the
grain, MTF, reciprocity and Callier laws to 2e-05 across the whole database.

---

## 2. Implemented in one language, missing in the other

### 2.1 → Add to **C++** (one item)

**~~`mtf_rolloff_q` — the measured MTF rolloff. 22 stocks.~~ ✅ DONE 2026-09-03** — by option 1
below, the load-time separable fit. What follows is the state before that change, kept because the
reasoning is what chose the option.

Python: `FreqGrid.mtf(..., spec=profile.mtf)` → `fp.mtf_response()`, applying `1/(1+(f/f50)^q)`.
C++: stage 6 builds a separable spatial Gaussian from `f50` alone. `FilmMtfResponse()` is generated
and **has no caller**.

⚠ Registered in `cpp_parity.LAW_BYPASS_BASELINE` with the cause — *the C++ side has no FFT and the
law is a frequency-domain form* — and the magnitude: **correct at f50 by construction, up to 3.8×
too much modulation at 2× f50**. ⚠ **That entry says 9 stocks; the database now has 22.** Correct
the count first — it is the only quantified statement of the exposure.

**Three ways to close it**, in ascending cost:

1. **Fit a separable spatial kernel to the power law, per stock, at load time.** The power law is
   monotone and smooth; a two- or three-Gaussian sum reproduces it to well under a percent. Cost at
   render: **unchanged** — still one separable blur. This is the recommendation.
2. Numerical kernel built by inverse-transforming the law once per stock at load time. Same render
   cost, larger kernel support.
3. An FFT path in C++. Correct, and by far the largest architectural change.

Option 1 needs no new data and no new schema. It needs a stated tolerance and a guard asserting the
fitted kernel matches `FilmMtfResponse()` inside the compiled program — the same pattern
`LAW_EQUIVALENT_IMPL` already uses for `FilmGrainSigma`.

**Also emit to C++** (both are schema gaps, not code gaps):

* the spectral `TakingFilter` struct — `cut_on_nm` reaches C++ as a bare scalar, `transmission`
  never does. **1 stock affected today**, so this is cheap insurance rather than a fix.
* `reciprocity_table.development_correction_pct` — 3 stocks.

### 2.2 → Add to **Python** (four stages)

Python has no temporal chain at all. C++ has four stages Python lacks:

| C++ stage | what it does | data it reads | port difficulty |
|---|---|---|---|
| **3c** temporal flicker | ⚠ **a declared STUB in C++ too** — copies input to output | `flicker_pct`, `flicker_hz` (both unread) | build it once, in both |
| **9b** negative defects | embedded dust, debris, fibres | **user controls only** (`params.damageSeed`) | easy; no data needed |
| **15** gate weave | per-frame translation | `weave_amp_x_um`, `weave_amp_y_um`, `weave_hz_corner` | easy |
| **16** machine defects | gate dirt, one-frame dirt, splices | `dirt_events_per_frame` + controls | easy |

⚠ **The reason to port them is not the render — it is verification.** `cpp_parity.py` can only
check a law that exists on both sides. These four stages are today the **least-verified consumed
code in the system**: they read real database fields and nothing independent confirms what they do
with them. Porting them to Python costs little and converts four unverified stages into checked
ones.

C++ **stage 17** (final clamp) has a Python equivalent folded into its output path; worth aligning
the *name*, not the code.

---

## 3. Present in the database, used by neither

Ordered by data actually present. ⚠ **The "distinct values" column is the honest signal**: a field
with 3 distinct values across 175 stocks is a class rule wearing a data field's clothes.

| # | field(s) | stocks | distinct | verdict |
|---|---|---|---|---|
| 1 | `temporal.flicker_pct`, `flicker_hz` | 175 | 12 | **class estimates**, and the C++ stage is an admitted stub. Implement the stage; do not pretend the data is measured |
| 2 | `grain.size_sigma_log` | 175 | **3** | class rule. Needs real crystal-size distributions before it is worth a code path |
| 3 | `grain.dye_cloud_um` | 106 | **5** | class rule. ⚠ Interacts with **C45**: that rescale was left explicitly conditional on the stored `clump_gain`, and the dye cloud is the other half of the same shape question |
| 4 | `third_party.*` (14 numeric) | 175 | — | ⚠ **deliberately not a render input** — another product's observations. Its value is as a **validator the project does not yet run**, not as data to consume |
| 5 | `dye_density.d_cyan/d_magenta/d_yellow/d_neutral/d_dmin` | **18** | measured | ⚠ **the largest available colour gain, and free at runtime** — still one 3×3 at stage 12, all work offline. `_MEASURED_DYE_MATRIX_ADOPTED = False` is the wired switch |
| 6 | `mtf.resolving_power_lp_mm_lowc/highc` | 46 | measured | blocked on **queue G6** — what Agfa's "lines/mm" axis means. Data-blocked, not code-blocked |
| 7 | `processing.*`, `processing_family.points`, `push.*` | 22 / 11 / 6 | tier 1 on 27 records | contrast and speed vs development. Resolves into a modified `ToneCurve` baked into the existing LUT — **zero per-pixel cost**. Needs a plugin control |
| 8 | `emulsion.grain_um / coated_um / base_um / aspect_ratio / iodide_mol_pct` | 17 / 12 / 12 | tier 1 on 13 records | parameters to stages that already run. ⚠ `base_um` is not emitted to C++ |
| 9 | `aim_density`, `print_grain_index`, PrintStock per-channel MTF and printer light | 13 / 12 | measured | **blocked on the print chain**, not missing |
| 10 | `layer_stack.order / resolving_*` | 5 / 1 | measured | ⚠ **the only expensive item**: splits one shared blur into three |
| 11 | `dye_impurity.ratios` | 4 | measured | too few stocks to generalise |
| 12 | `aging.*` (11 fields) | **0** | — | zero on every stock. Not a gap — a feature with no data |
| 13 | `halation.radius_scale_*`, `mtf_tail_a/f_exp`, `cluster_um`, `speed_point_x`, `trim` | 175 / 175 / 0 / 0 / 0 | 1 value | **inert by data**. The code gap costs nothing today |

**Correctly excluded**: `features` is a *summary* of the numeric fields (`film_profiles.py:64`),
read only by schema helpers that set them. Build-time input, not a render input.

⚠ **One real omission hiding in plain sight**: `exposure_index` is the film's rated speed and
**nothing places the scene on the curve with it** — Python mentions it in a debug string, C++ reads
it only in `AlgoProcessVariant`. One frame-setup scalar, both sides, no new data.

---

## 4. Technical data required to make unsupported parameters usable

This is the acquisition list, ordered by *rendered benefit per document found*.

| priority | what to find | unlocks | current holding |
|---|---|---|---|
| **1** | **Spectral dye-density panels** (cyan/magenta/yellow density vs wavelength) for the main colour stocks — Kodak E-series and Fuji AF3 bulletins print them | `dye_density` → a per-film dye matrix. Biggest colour gain in the project, zero render cost | **18 / 175** |
| **2** | **Time–gamma families**: development time vs gamma vs EI, per developer | `processing_family`, `push` → contrast and speed become controllable rather than fixed at one unnamed development | 11 / 175, tier 1 on 27 records |
| **3** | **An unambiguous statement of Agfa's "lines/mm" ordinate** (queue **G6**) | 46 stocks' `resolving_power_*` become usable, and twelve MTF/resolving pairs resolve | data present, meaning blocked |
| **4** | **B&W negative σ(D) plots** (queue **F2b**) | 55 monochrome negatives leave a placeholder σ(D). ⚠ **Lu & Torquato 1990 now offers a principled one-parameter alternative** — this acquisition is no longer the only route | placeholder on 55 |
| **5** | **Crystal-size distributions and coating thickness** — electron micrographs, patents, Kodak/Agfa emulsion papers | `emulsion.*` (17/12), and `grain.size_sigma_log` / `dye_cloud_um` stop being 3- and 5-valued class rules | tier 1 on 13 records |
| **6** | **Coating order plus per-layer resolving power** | `layer_stack` — per-layer sharpness, the red record softest because it sits deepest. Expensive to render, so low priority despite being physically real | 5 / 175, and `resolving_top` on **1** |
| **7** | **Dye impurity / unwanted-absorption ratios** | `dye_impurity` — the reason colour negatives need masking | 4 / 175 |
| **8** | **A measured Wiener spectrum or multi-aperture rms for a stock THIS DATABASE ACTUALLY CARRIES** | ⚠ would convert **C45** from a class inference into a measurement. All fifteen measurements behind the current clump scale are of films not in the database | 0 / 175 |
| **9** | **Scanner step-wedge and empty-gate scans** (queues **D1/D2**) | splits emulsion σ from scanner σ; also the gate on **C18** | **needs the owner, not a document** |
| **10** | Dye-fade / accelerated-ageing studies | `aging` (11 fields, zero everywhere) | 0 / 175 — lowest priority |

---

## 5. Status of the plan — executed 2026-09-03

| # | item | outcome |
|---|---|---|
| 1 | `LAW_BYPASS_BASELINE` count 9 → 22 | ✅ **the entry was removed entirely** — the law gained a caller, which that file states is the only admissible reason to leave the dict |
| 2 | stage-numbering divergence | ✅ documented here and in `PROGRESS.md` |
| 3 | wire `exposure_index` | ⚠ **REFUSED, and it is not a gap.** `solve_anchors` lands 18 % grey on target, so an EI-derived shift is exactly what it removes — inert before the solve, breaks the grey landing after it. EI is incompatible with automatic neutral anchoring, not missing from the renderer. It is already consumed in `ProcessVariant` resolution |
| 4 | emit `TakingFilter` + `development_correction_pct` | ✅ both emitted; empty / 3 stocks, so no render change |
| 5 | MTF rolloff by separable kernel fit | ✅ **done, both twins.** Worst 0.0384 vs 0.1737; every affected stock improves 1.4–10× |
| 6 | port 3c / 9b / 15 / 16 to Python | ⚠ **3c REFUSED** — `AlgoTemporalFlicker.hpp` records that the spectral shape and the channel split are unspecified, so implementing it means inventing a spectrum. **9b / 15 / 16 NOT DONE** — a bit-exact port needs the counter-based RNG reproduced in Python, and a statistical port would not meet `cpp_parity`'s tolerance. Left undone rather than half done |

⚠ **The consequence of 6, stated rather than buried**: stages 9b, 15 and 16 remain the
least-verified consumed code in the system. They read `weave_amp_x_um`, `weave_amp_y_um`,
`weave_hz_corner` and `dirt_events_per_frame`, and nothing independent checks what they do with
them.

**Then, as documents arrive:** `dye_density` first, the processing block second.

⚠ **Nothing in this plan removes or disables anything.** The single deliberate asymmetry that
should stay is Python's sRGB encode, which belongs to the host in the plugin build.
