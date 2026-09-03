# Database → algorithm coverage: Python vs C++

**2026-09-03 · schema v24 · 175 film stocks · 33 schema dataclasses · 312 Python fields,
309 of them emitted into `film_profiles.hpp`.**

Supersedes `deliver/UnusedDataFields_01_09_2026.md`, which was written at 170 stocks and is
**stale on three of its four top items** — reciprocity, the taking matrix and the processing block
have all gained C++ consumers since.

---

## 0. ⚠ Method, and what it cannot prove

Three passes, each of which found errors the previous one made:

1. **Field enumeration** by `dataclasses.fields()` on every dataclass in `film_profiles.py` — not
   by reading the source, so nothing is missed by eye.
2. **Emission** by parsing the generated `film_profiles.hpp` / `film_profiles_detail.hpp` structs.
   A field absent here *cannot* be used by C++, whatever the engine does.
3. **Consumption** by member-access spelling across the 82 engine translation units.

⚠ **Three traps this hit, recorded because they invalidate a naive answer:**

* **`/root/work/sc` and `/root/work/av` are stale partial copies** — they have no
  `AlgoReciprocity.hpp` at all. Scanning them reports reciprocity as a C++ gap, which it stopped
  being days ago. **`/root/work/proot` is the synced tree** and the only valid target.
* **`.field` alone misses `->field`.** The print chain reads `pPrintStock->grain_rms`; a
  dot-only scan reports the entire print stock as unread. That single omission moved 6 fields.
* **A field can be consumed without its name appearing.** `kind` is reached only through
  `profile.isReversal()`; `mtf_rolloff_q` only through `fp.mtf_response(spec, …)`, which takes the
  whole `MTFSpec`; the `sigma_shape_*` septet only through `fp.grain_sigma()` and its hoisted C++
  twin `AlgoGrainAmpBuild`. **Every conclusion below about a non-trivial field was confirmed by
  reading the call site, not by the grep.**

This is a **reachability floor**, the same standard `cpp_parity.py` sets for itself: it proves a
field is reached, and it proves a field's name is absent, but a name present only in a comment
would count. Nothing below rests on a bare grep.

---

## 1. Answer in one paragraph

**Python is the reference and it is not complete either.** Of the fields that can move a pixel,
**Python consumes essentially all of the per-frame optical and photographic chain and none of the
temporal chain**; **C++ consumes all of that plus the temporal chain**. ⚠ The one law it missed — the measured MTF
rolloff — was closed on 2026-09-03; see A1. The two together still leave a substantial block of harvested but
unconsumed data — most of it recently acquired, most of it thin, and one item of it populated on
every stock in the database.

| | fields that can affect output | consumed | not consumed |
|---|---|---|---|
| **Python** (`film_sim.py`) | ~245 | ~118 | ~127 |
| **C++** (82 TUs in `proot`) | ~245 | ~132 | ~113 |
| **Either implementation** | ~245 | ~136 | **~109** |

Counts are approximate at the margin only because "can affect output" is a judgement for a dozen
label-like fields (`density_metric`, `referred`, `mask_encoding`); the substantive findings below
are exact.

---

## 2. Consumed by BOTH — the working set

No action. Listed so the gaps below are read against something.

| block | fields | where |
|---|---|---|
| `ToneCurve` × 3 | dmin, gamma, toe_x, toe_k, shoulder_x, shoulder_k | Py stage 8 `density()`; C++ `AlgoCharacteristicCurve` / `AlgoCurveLut` |
| `GrainSpec` core | rms_granularity, clump_um_r/g/b, clump_gain, fog_grain, anisotropy, rms_r/g/b | Py `make_grain_field`; C++ `AlgoMakeGrainField` (stage 11/13/14) |
| `GrainSpec` σ(D) | sigma_shape_toe/mid/dmax/peak/peak_at/toe_at/dmax_at/measured | **indirect both sides**: Py `fp.grain_sigma()`; C++ `AlgoGrainAmpBuild`/`AlgoGrainAmpAt`, declared in `LAW_EQUIVALENT_IMPL` |
| `MTFSpec` geometry | f50_r/g/b, adjacency, adjacency_um | Py `FreqGrid.mtf`; C++ `AlgoStage06_EmulsionMtf` |
| `HalationSpec` | radii_um, weights, gain_r/g/b, threshold_stops | Py stage 5; C++ `AlgoHalation` |
| `CouplerSpec` | strength, radius_um, edge_strength, edge_um | Py stage 9; C++ `AlgoDirCoupler` |
| `InterimageSpec` | a_rg…a_bg, iterations, density_weighting | Py `apply_interimage`; C++ `Algo_08_Sim` |
| `ReciprocitySpec` + `ReciprocityTable` | schwarzschild_p_r/g/b, onset_s, times_s, stops_correction | Py `reciprocity_log_shift`; C++ **`AlgoReciprocity.hpp`** ⚠ was a gap on 2026-09-01, now closed |
| `SpectralSensitivity` | lambda_start_nm, lambda_step_nm, log_s_r/g/b/pan | Py stage 7; C++ `AlgoSpectralSensitivity` |
| `CoatingSpec` | coating_sigma, corr_across/along_mm, buckle_mtf_loss, edge_fog_density/mm | Py stage 4; C++ `Algo_04_Sim`, `AlgoCornerDefocus`, `AlgoEdgeFog` |
| profile scalars | taking_matrix, dye_matrix, spectral_weights, base_tint, misregistration_um, silver_tone, default_flare, default_vignette, callier_q, balance_kelvin, is_monochrome, **kind** (via `isReversal()`), reseau, trim | across stages 2b–14 both sides |
| `PrintStock` | curves, dye_matrix, mtf_f50, grain_rms, grain_clump_um | Py print/dupe; C++ `Algo_13_Sim` / `Algo_14_Sim` (via `->`) |

---

## 3. ⚠ Gap A — Python consumes it, C++ does not

### A1. ~~`mtf_rolloff_q` — the one real law divergence~~ ✅ CLOSED THE SAME DAY

**22 stocks carry a measured rolloff exponent.**

* **Python**: `FreqGrid.mtf(..., spec=profile.mtf)` → `fp.mtf_response(spec, channel, f)`, which
  applies `1 / (1 + (f/f50)^q)` when `mtf_measured`.
* **C++**: `AlgoStage06_EmulsionMtf` reads **only** `f50_r/g/b`, `adjacency`, `adjacency_um` and
  builds a separable spatial Gaussian from `ALGO_MTF_SIGMA_MM_PER_INV_F50 / f50[c]`.
  `FilmMtfResponse()` existed in the generated header with **no caller**.

⚠ **This section described the state at the start of 2026-09-03 and was fixed that afternoon.**
`FilmMtfKernel` now carries a two-Gaussian separable fit of the law, keyed on the exact stored q,
and stage 6 in both twins convolves it as two extra lobes of the blur it already ran. Worst
max|error| **0.0384** against **0.1737** for the Gaussian it replaced. The bypass entry is gone from
`cpp_parity.LAW_BYPASS_BASELINE`; the indirection is declared in `LAW_EQUIVALENT_IMPL` with its
bound, and `verify.py` asserts both the bound and that the kernel beats the Gaussian on every row.

### A2. ~~`taking_filter.transmission` / `cut_on_nm`~~ ✅ CLOSED 2026-09-03

Python applies the spectral taking filter before the mono collapse. C++ received only a scalar
`taking_filter_cut_on_nm`; the `TakingFilter` struct and its `transmission` array were never
emitted. **Both are emitted now**, along with
`ReciprocityTable.development_correction_pct`, the other field of a struct that was otherwise
fully visible to C++. Empty on all 175 stocks today, so no render changes — the point is that the
first stock to carry a curve will not be a silent divergence. ⚠ Not to be confused with `taking_matrix`, the 3×3, which **is** consumed by
`AlgoTakingFilters` on both sides — that item in the 2026-09-01 document is closed.

**Data reality**: `transmission` is empty on **all 175** stocks and `cut_on_nm` is set on **1**.
So the live divergence is one infrared stock. Low urgency, but it is a genuine
Python-and-C++-disagree case, not a shared omission.

### A3. `PrintStock` per-channel MTF and the printer light

`mtf_f50_r/g/b`, `mtf_f50_bound`, `mtf_measured`, `printer_light_r/g/b`, `log_e_per_point` are
emitted and read by no C++ stage; C++ uses the single scalar `mtf_f50` instead. Python's print
chain is likewise partial here. Blocked with the rest of the print chain (§5).

---

## 4. ⚠ Gap B — C++ consumes it, Python does not

**The entire temporal chain.** `film_sim.py` is a still-frame reference and has **no temporal
stage at all**: `weave_amp_x_um`, `weave_amp_y_um`, `weave_hz_corner`, `dirt_events_per_frame` are
read by `AlgoGateWeave` / `AlgoGateDefects` and by nothing in Python. Also `process_variants`
resolution and `push`, reached in C++ through `AlgoProcessVariant.hpp`.

**This is a deliberate scope difference, not a defect** — but it has a consequence worth stating:
**those fields have no Python reference to be checked against**, so `cpp_parity` cannot cover
them. They are the least-verified consumed fields in the system.

---

## 5. ⚠ Gap C — consumed by NEITHER, and the data exists

Ranked by how much data is actually sitting there.

| # | field(s) | stocks with data | why unused | where it would go |
|---|---|---|---|---|
| 1 | `temporal.flicker_pct`, `flicker_hz` | **175 / 175**, 12 distinct values | ⚠ `AlgoTemporalFlicker.hpp` is **an explicit STUB — "copies input to output"**. The header even records that an earlier control design referenced two fields that never existed | stage 3c, exposure domain. A per-frame scalar; **zero per-pixel cost** |
| 2 | `grain.size_sigma_log` | **175 / 175**, 3 values | crystal size dispersion; the grain field is generated from one clump scale, not a distribution | stage 11. Would need a second noise octave — moderate cost |
| 3 | `grain.dye_cloud_um` | **106 / 175**, 5 values | the developed dye cloud is larger than the crystal; currently folded into `clump_um` | stage 11, as a second rolloff term. ⚠ Interacts with C45 — the clump rescale was made *conditional* on the stored `clump_gain`, and this is the other half of the same shape question |
| 4 | `third_party.*` (14 numeric) | `color_matrix` **175/175 non-identity**, `dmax` 17 | ⚠ **deliberately not a render input** — a competitor-observation set kept for cross-checking, and using it would import another product's look | nowhere by design; valuable as a *validator* the project does not yet run |
| 5 | `mtf.resolving_power_lp_mm_lowc/highc` | 46 / 175 | a resolution figure, not a transfer function; no adopted conversion to f50 | ⚠ this is **queue G6** — blocked on what Agfa's "lines/mm" axis means |
| 6 | `dye_density.d_cyan/d_magenta/d_yellow/d_neutral/d_dmin` | 18 / 175 | the measured spectral dye set; the renderer still mixes through the generic 3×3 `dye_matrix`. `_MEASURED_DYE_MATRIX_ADOPTED = False` is the wired switch | stage 12. **Zero per-pixel cost** — still one 3×3 — all the work is offline. Biggest colour gain available |
| 7 | `processing.minutes/celsius/contrast_index/progress/…`, `processing_family.points`, `push.*` | 22 / 11 / 6 | contrast and speed vs development; the stored curve sits at one unnamed development | resolves into a modified `ToneCurve` baked into the existing LUT. Needs a plugin control |
| 8 | `emulsion.grain_um / coated_um / base_um / aspect_ratio / iodide_mol_pct` | 17 / 12 / 12 | physical crystal and coating geometry; grain and halation depth are class estimates instead. ⚠ `base_um` is **not emitted to C++** | stages 5 and 11, as parameters to passes that already run |
| 9 | `aim_density`, `print_grain_index` | 13 / 12 | print-chain targets. Meaningless until the print chain is complete | blocked, not missing |
| 10 | `layer_stack.order / resolving_*` | 5 / 1 | coating depth order — the red record sits deepest and is softest | ⚠ **the only expensive item here**: it splits one shared blur into three |
| 11 | `dye_impurity.ratios` | 4 / 175 | unwanted dye absorptions. Four stocks cannot generalise | stage 12, one extra cross-talk term |
| 12 | `reciprocity_table.development_correction_pct` | 3 / 175 | ⚠ **not emitted to C++**; the rest of the table is | stage 2, alongside the reciprocity shift already there |
| 13 | `dye_stability.*` (11) | print stocks | long-term dye loss | an aging chain that does not exist |

---

## 6. Unused and correctly so

* **`aging` — all 11 fields are zero on all 175 stocks.** Nothing in the corpus can populate them.
  Not a gap; a creative-effect feature with no data.
* **`halation.radius_scale_r/g/b` — 1.0 on all 175.** Inert by data, so the code gap costs nothing
  today.
* **`mtf_tail_a` = 1.0 and `mtf_tail_f_exp` = 0.0 on all 175.** The measured rolloff rides entirely
  on `mtf_rolloff_q`; these two are a dormant generalisation.
* **`grain.cluster_um` — 0 on all 175.** Never populated.
* **`speed_point_x` — 0 on all 175**, `trim` 0 on all 175.
* **`FilmProfile.features`** — ⚠ **correctly unused at runtime by both.** `film_profiles.py:64`
  states it is *"a convenience summary of the numeric fields"*, and the flags are read only by
  schema-application helpers that **set** those numeric fields at construction. It is a build-time
  input, not a render input. Emitting it to C++ is harmless but it should not be counted as a gap.
* **`exposure_index`** — read in C++ only by `AlgoProcessVariant`, and in Python only by a debug
  string. ⚠ Still a real omission for *exposure placement*: the film's rated speed does not
  currently decide where the scene lands on the curve. One frame-setup scalar.

---

## 7. Verdict

**No, the two implementations do not collectively consume every simulation-relevant field**, and
the shortfall is not evenly distributed:

1. ~~One true law divergence~~ — **closed 2026-09-03.** What remains is a bounded APPROXIMATION
   rather than a divergence: Python keeps the exact law, C++ convolves the fitted kernel, and the two
   differ by at most 0.0384 in modulation on 22 stocks. `RenderSettings.mtf_use_kernel` renders the
   Python side through the same kernel when the two must be compared on identical arithmetic.
2. **One schema-level divergence** — the spectral `TakingFilter` is never emitted to C++ (1 stock
   affected today), and `reciprocity_table.development_correction_pct` likewise (3 stocks).
3. **One asymmetry by design** — the temporal chain is C++-only and therefore unverified by
   `cpp_parity`.
4. **The largest block is neither implementation's fault**: harvested data with no stage to
   consume it, led by `dye_density` (18 stocks, free at runtime, biggest colour gain) and the
   processing block (22 stocks, free at runtime).
5. **Two fields are populated on every stock in the database and consumed by nothing**:
   `temporal.flicker_pct` / `flicker_hz` against an admitted stub, and `grain.size_sigma_log`.

**Cheapest real gains, in order**: correct the `LAW_BYPASS_BASELINE` count 9 → 22; wire
`exposure_index` into exposure placement; implement the flicker stub; then `dye_density`.
