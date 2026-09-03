# CINESTILL 800T — complete data ledger, accept / reject, with reasons

**Date:** 2026-08-27 · **Profile:** `CINESTILL_800T` · **Schema:** v16 · **Tier:** 2,
`fitted_from = "datasheet_curve"`, `last_reviewed = 2026-08-27`

## 0. Direct answer to the question asked

**No — not all harvested details went into the database, and that is deliberate.** The honest tally:

| | Count | |
|---|---|---|
| Data points harvested for this stock, both sources | **2 076** | 1 440 traced from vendor figure 1, 617 from vendor figure 2, 19 discrete values from FilmLab Pro |
| **Written to the database** | **18 numbers + 1 flag + 2 classification changes** | see §2 |
| **Confirmed existing values without changing them** | **4** | see §3 |
| **Rejected, with a stated reason** | **19 FilmLab Pro items + 617 points + 2 whole figures + 2 text claims** | see §4 |
| Retained in the archive but not the database | **2 057 points** | see §5 |

⚠ **The 1 440-point trace is not "rejected" — it is *represented*.** It became 18 `ToneCurve`
parameters with the fit residual recorded (rms 0.0197 / 0.0248 / 0.0154 D; max 0.039 / 0.060 /
0.053 D). The schema stores an analytic curve, not a point table, which is the project's
established convention for every traced plot — so the compression is the integration, and the
480 raw points per layer stay on disk so the fit can be re-derived or challenged.

---

## 1. The four sources, ranked by evidential class

This ranking decides every accept/reject below. **Source class, not usefulness, is what the rules
turn on.**

| # | Source | Class | Why that class |
|---|---|---|---|
| **S1** | `cs41curves_600x600.png` — CineStill's own sensitometric figure, 3 per-layer curves | **VENDOR / manufacturer-class** | Published by the manufacturer, on their own site, about their own product. Not a formal data sheet (they issue none), but a vendor document. |
| **S2** | `cs41vscs2curves2_PNG_480x480.png` — CS41-vs-CS2 process comparison, 2 curves | **VENDOR, but UNIDENTIFIED** | Same publisher, same class — but which curve is which process is unknown, so no value can be attributed. |
| **S3** | cinestillfilm.com article text | **VENDOR, qualitative** | Manufacturer statements, but prose with two exceptions: the push range and the fog claim. |
| **S4** | FilmLab Pro v2.1 published-data engine | **THIRD-PARTY, hand-authored, tier 3** | Claims digitization from manufacturer publications, names no instrument/operator/date/laboratory, and its rms figures contradict the very datasheets it cites for four other stocks. `NotFound.md` §7.1. |
| — | `Kodak400Sensi*.png` ×2 | **NOT THIS FILM** | Compare a counterfeit "Brand H" 400 against Kodacolor 400 and against CineStill **400D**. Never traced. |

**The governing rule, as you set it:** an official/vendor value is primary; a lower-tier value may
fill a gap but never overwrite a higher-tier one; where both exist and differ, preserve both and
assess the discrepancy.

---

## 2. ACCEPTED — every value now in the database, and its source

### 2.1 Characteristic curves — 18 numbers, from S1 ✅

The only substantive numeric import. Replaced a tier-2 analogy estimate.

| Layer | dmin | gamma | toe_x | toe_k | shoulder_x | shoulder_k |
|---|---|---|---|---|---|---|
| r | **0.1873** | **0.6023** | **−1.4188** | **0.3180** | **1.8145** | **0.4450** |
| g | **0.5258** | **0.6214** | **−1.6912** | **0.3134** | **2.0148** | **0.4385** |
| b | **0.8758** | **0.6088** | **−1.5706** | **0.1467** | **1.9976** | **0.2053** |

**Why accepted:** vendor-published figure; 480 samples per layer at one per pixel column;
calibration over-determined on both axes (9 ordinate labels 0.0–3.5 by 0.5 landing D 0.0 and 3.5
exactly on the frame; 6 log₁₀ labels −4.0…+1.0 spanning the frame; 17 "Camera Stops" labels
−8…+8); fit residuals well under a tenth of a density unit.

**Independent confirmation that the plate is this stock** — the raster never says so in words:
KODAK VISION3 500T 5219, which this emulsion *is* minus the remjet, traces to
**0.1867 / 0.5811 / 0.8374** from its own Kodak H-1 sheet. Worst-channel agreement **0.06 D**.

**What was replaced, and why it mattered:** the old dmin triple was a flat **0.22 / 0.20 / 0.19** —
not merely imprecise but **the wrong kind of description** for a masked colour negative. The traced
values are the orange-mask ladder, spread 0.689 D. Gamma barely moved (0.610 / 0.630 / 0.652 →
0.602 / 0.621 / 0.609), and toe/shoulder landed almost on the old estimate once the mid-grey shift
was applied (red toe −1.52 → −1.419; red shoulder 1.86 → 1.815). **The old estimate was good; the
trace refines it and corrects its dmin convention.**

⚠ **Two caveats recorded with the values, not hidden:**
1. **All three shoulders sit at the `shoulder_k = 1.4 × toe_k` ceiling** the `ToneCurve` docstring
   imposes. The measured shoulders want to be *softer*. 1.4× is this project's monotonicity-safe
   bound for new stocks, so the fit is clamped there and the modelled shoulder is very slightly
   sharper than the plate's.
2. **CineStill's own chart is internally inconsistent by 3.7 %** — 16 stops drawn across exactly
   5.00 decades is 3.20 stops/decade, where a stop is 0.30103 decades. The **log axis was adopted**
   (uniform to ±1 px, lands on the frame edges); the stops axis was used **only for its zero**,
   which places metered mid-grey at log E −1.51681. The same defect appears on figure 2, so it is a
   property of their plotting, not a misread of one figure.

### 2.2 Two classification changes that follow from the curves ✅

| Change | From | To | Why |
|---|---|---|---|
| `mask_encoding` | `neutral_dmin` | **`dmin_ladder`** | The per-channel dmin now *is* the mask. Derived automatically from `_DMIN_LADDER` membership. |
| `_DMIN_LADDER` | absent | **member** | 0.689 D ladder, cross-checked against 5219. Same correction the 2026-08-26 KODAK still-film batch made on eight profiles. |
| `provenance.fitted_from` | `secondary_sources` | **`datasheet_curve`** | The curves are vendor-grounded. New `_VENDOR_TRACED_CURVES` set, kept separate from `_KODAK_STILL_HARVEST_CURVES` because the document class differs — a vendor news page carrying a real figure, not an E-series or H-1 publication. |

⚠ **The profile tier stays 2, deliberately.** CineStill publish no data sheet, so grain, MTF,
halation magnitude, couplers, dye matrix and spectral response are all still estimates or
analogies from 5219. Tier describes the profile; `fitted_from` describes how the curves were got.

### 2.3 Push latitude — 3 fields, from S3, **and it required schema v16** ✅

| Field | Value | Source |
|---|---|---|
| `push.max_push_stops` | **3.0** | *"could even be push processed up to 3 stops further"* |
| `push.base_fog_penalty_per_stop` | **0.0** | *"without any base fog issues"* |
| `push.fog_penalty_stated` | **True** | ⚠ this flag is why the 0.0 above is data and not silence |

**Why a new struct rather than an existing field.** `ProcessingSpec` describes the *one* development
condition the stored curve represents — "+3 stops" beside a single time and temperature would read
as applying to that time, which no source means. `ProcessingFamily` carries measured time-gamma
*points*, and a prose sentence is not a point on that curve. `exposure_index` is the rating;
overwriting it with a pushed EI destroys the rating.

**Why `fog_penalty_stated` exists.** `base_fog_penalty_per_stop = 0.0` alone is ambiguous — it means
either "nobody said" or "someone said there is none". CineStill's claim is precisely the *negative*
one, which is a published fact. Same class of problem as the v15 PGI censoring sentinel, solved the
same way.

### 2.4 Three citations ✅

The vendor figure (with its full calibration, residuals, and the 3.7 % axis finding), the vendor
article text, and the pre-existing 2012 product documentation. The FilmLab Pro record is **not**
cited on this profile — nothing on it came from that source.

---

## 3. CONFIRMED — existing values the sources corroborated without changing

Worth stating separately: a confirmation is a real result, and none of these produced a write.

| Value | Stored | What confirmed it |
|---|---|---|
| `exposure_index` | 800 | S3 *"rated at a box speed of EI 800 in tungsten-balanced light"*; S4 independently `iso: 800` |
| `balance_kelvin` | 3200 | S3 *"optimized for 3200K light"* |
| `Feature.NO_REMJET` | set | S3 — remjet removed by CineStill's "Premoval" process |
| Halation **direction** (red-dominant) | `gain 1.05 / 0.30 / 0.10` | S3 *"red halation glow"*; **and S4 independently**: their tint `(0.85, 0.12, 0.04)` is strongly red, and their `radius_norm 0.012` is **3–4× every other stock in their dataset** — an independent ordering agreement with our extreme values |

---

## 4. REJECTED — every item, with the specific technical reason

### 4.1 FilmLab Pro (S4) — 19 items, none written

Grouped by reason. **In no case was the reason "the source is not a manufacturer."**

**R-A — the field is already populated by a higher-tier value (your rule forbids the overwrite)**

| Item | Theirs | Ours | Note |
|---|---|---|---|
| `gamma` | 0.56 | **0.602 / 0.621 / 0.609** (S1 traced) | one value where the film has three |
| `dmin` | 0.22 | **0.1873 / 0.5258 / 0.8758** (S1 traced) | ⚠ their 0.22 matches our traced **red** dmin 0.217 to 0.003 D — their single number is the red layer's, which is consistent with their own disclosure that display normalisation anchors log E −2.5…+1.8 to 0–1 rather than to physical Dmin–Dmax |
| `mtf50_lp_mm` | 55 | 40 / 48 / 56 | ours is a per-layer analogy estimate, theirs a single hand-authored figure; neither is measured, so there is no gain in swapping one estimate for another and losing our documented derivation |
| `rms` | 11.5 | 8.4 | both estimates. Ours is a VISION3-ladder derivation with a written rationale; theirs is hand-authored with none. ⚠ **This is the closest call in the whole ledger** — see §6.1 |
| `iso` | 800 | 800 | identical; a write would have been a no-op |

**R-B — no field exists, and forcing one would lose or distort information**

| Item | Theirs | Why not stored |
|---|---|---|
| `dmax` | 2.5 | No `dmax` field on any stock. `ToneCurve.dmax` is a *derived property* — ours computes 2.135 / 2.829 / 3.048 from the traced curve. Storing a fourth, independent number would let the stored Dmax disagree with the curve that produces it. |
| `latitude_stops` | 11 | No field. `ToneCurve.latitude_stops` is derived — ours gives 12.31 (green). Same argument. |
| `size_microns` | 3.8 | This is **mean crystal diameter**. Our `clump_um_*` is **developed clump diameter**, which the `GrainSpec` docstring shows depends on development gamma (BBC T-101: D_eq ∝ γ^0.425) and on density (−20 % across the tone scale). They are different physical quantities. Aliasing them would be a category error. |
| `orange_mask` `(0, 0.05, 0.16)` | | Our mask is encoded **in the dmin ladder** (`mask_encoding = dmin_ladder`). Adding a second, differently-scaled mask term would double-count it. Their own disclosure says the mask contributes at 40 % strength in their engine — an engine-specific weight, not a film property. |

**R-C — the quantity is not convertible to our units**

| Item | Theirs | Why not convertible |
|---|---|---|
| `halation.radius_norm` | 0.012 | **A fraction of the image dimension, not a length on film.** It does order monotonically against our `radii_um` (their 0.002 ↔ our 7–9 µm inner lobe; their 0.012 ↔ our 20 µm), but every conversion factor derivable from that is fitted against **our own estimates** — circular, and adds no independent information. Our `radii_um (20, 130, 700)` stands as an estimate. |
| `halation.intensity` 0.5, `threshold` 0.45, `tint` | | The field is populated **and ours is stronger** (gain_r 1.05 vs their 0.5; threshold 0.9 st vs their 0.45 normalised). Retained as corroboration of *ordering* only — §3. |
| `vignette` | 0.08 | Different normalisation entirely: ours is 0.35 in stops, theirs 0.04–0.18 on an unstated scale across their whole set. |

**R-D — the semantics are undefined by the source itself**

Their own field list marks these "semantics unknown", so there is nothing to convert *to*:

- `saturation` 0.95 — no definition, units or basis.
- `speed_offset` 0.15 — no definition.
- `grain_detail` bias terms **0.6 / 0.35 / 0.18** (shadow_bias / midtone / highlight_bias) — the
  harvest's own format note records: *"The three bias terms have NO stated definition, units or
  measurement basis."*

**R-E — the value is in their parameterisation, not a film property**

| Item | Theirs | Why |
|---|---|---|
| `layer_curves` | `r [0.06, 0.93, 0.78, 0.95]`, `g [0, 1, 0.86, 0.96]`, `b [−0.03, 1.05, 0.9, 0.98]` | `{speed_shift, curve_gamma, toe_gamma, shoulder_gamma}` — **relative multipliers** in their engine, applied at their own admitted *"35 % strength"*. Not our `toe_x/toe_k/shoulder_x/shoulder_k`. Numerically meaningless outside their pipeline — and superseded anyway by 480 traced points per layer. |
| `color_matrix` | `[[0.94,0.06,−0.04],[−0.04,1.02,−0.02],[0.04,0.04,0.92]]` | Our `dye_matrix` is populated. Their own disclosure: *"The color matrix blends at 0.18 weight since the per-channel LUTs already encode most color character"* — an engine mixing weight, not a dye property. |
| `crosstalk` | `[[0,0.02,0.05],[0,0,0.01],[0.02,0.01,0]]` | Our `interimage` is **already non-default** on this profile (a_rg −0.247, a_gr −0.264, a_br −0.169) and derived through the US5273870A measurement protocol. Their matrix is "unwanted dye absorptions weighted by local density" — a different quantity on a different scale, with no route between them. |

**R-F — retained as a cross-check rather than a value**

| Item | Theirs | Disposition |
|---|---|---|
| `characteristic_curve_points` | 7 points, `[[−3,0.22],[−2,0.32],[−1,0.62],[0,1.18],[1,1.72],[2,2.18],[3,2.45]]` | **Not used as a curve** — 7 points on an integer log-E grid for a three-layer film, versus 480 per layer from the vendor. **But used as an independent check**, and it passes: average gradient over their −2…+1 window is **0.467**, against **0.455** for our traced red record over the *same absolute* window — agreement to **2.6 %**. Recorded in the archive under `cross_source_comparison`. |

### 4.2 Vendor figure 2 (S2) — 617 traced points, rejected ⚠

Fully calibrated (`D = (433.5 − y)/129.2`, `logE = (x − 366.862)/75.186`, D 0.5–3.0, log E
−4.0…+1.0, stops −8…+8) and both curves traced — 284 and 333 points, both dashed.

**Reason for rejection: the curve identity is not established.** The in-plot annotation block could
not be read reliably off the pixel grid, so **which curve is the CS41 process and which the CS2
process is unknown**, as is which density channel is plotted (one curve per process, not three).

**Measured anyway, because it does not need the assignment:** the two processes differ by
**~12 % in straight-line gamma** — **0.555** (red dashed) against **0.493** (neutral dashed) over
log E −2.07…−0.08, on a base+fog of 0.57–0.62 D for both.

**That is exactly why it matters rather than being a curiosity: adopting the wrong curve would move
stored gamma by 12 %.**

⚠ **Inference was deliberately not used to break the tie.** A two-bath simplified kit "ought" to be
the lower-contrast one — but the red curve's 0.555 also sits *below* figure 1's own green-layer
0.621, so the naive story does not close. Tracked as `DIGITIZATION_QUEUE.md` **T4b**, reduced to
one question.

### 4.3 The two Kodak figures — not this film

`Kodak400Sensi_600x600.png` (counterfeit "Brand H" 400 vs Kodacolor 400) and
`Kodak400Sensi400D_600x600.png` ("Brand H" 400 vs CineStill **400D**). Never traced. **400D is a
different stock and is not in our database at all.**

### 4.4 Two vendor text claims not stored

| Claim | Why not stored |
|---|---|
| *"Xpro C-41"* processing | `ProcessingSpec` fields are all "as printed": developer, dilution, minutes, celsius, agitation, contrast index. The page prints **none** of them. "C-41" is a **process**, not a developer, and putting it in `developer` would be a semantic error — 148 of 161 profiles carry an empty `ProcessingSpec` for exactly this reason. It would fit a process-family tag, which does not exist yet. |
| Formats **135 / 120 / 4×5** | `default_format` is a single value and is already correctly `ff35` for still 35 mm. There is no field for a *list* of available formats, and inventing one to hold marketing availability would not touch a pixel. |

---

## 5. What is retained on disk but not in the database

Not rejected — **archived at full resolution so nothing has to be re-fetched or re-read**:

| File | Contents |
|---|---|
| `doc/thirdparty/cinestill_cs41_raw_px.txt` | figure 1, 480 raw pixel rows × 3 layers, plus the calibration formulae |
| `doc/thirdparty/cinestill_cs41vscs2_raw_px.txt` | figure 2, 382 columns × 2 curves, dashes marked as gaps |
| `doc/thirdparty/cinestill_curves_2026-08-27.json` | 147 KB — both figures calibrated in the figure's native log E *and* this database's convention, every axis label, the fit and its residuals, the measured gamma pair from figure 2, and the full cross-source comparison against FilmLab Pro |
| `doc/thirdparty/filmlabpro_harvest_2026-08-27.json` | the complete FilmLab Pro record, verbatim, including every rejected item above |

---

## 6. Where I could be wrong, and what would change each rejection

### 6.1 The closest call: `rms_granularity` 8.4 (ours) vs 11.5 (theirs)

Both are estimates. Neither is measured. Ours is a derivation down the VISION3 family ladder with a
written rationale; theirs is hand-authored with none, from a source whose rms figures **contradict
the datasheets it cites** for Portra 400 (6.5 vs E-4050's 4), Acros 100 (4.5 vs 7), Velvia 50
(3.8 vs 9) and Kodachrome 64 (6 vs 10). That track record is the reason I did not swap.

⚠ **But it is worth naming the counter-argument**: 800T is a stripped 5219, so it should be
*grainier* than 5219 (rms 6.6) after remjet removal and C-41 cross-processing, and 8.4 is only
27 % above 5219 while their 11.5 is 74 % above. If you want the higher figure, say so — it is one
line, and it would carry a tier-3 third-party citation like the nine halation imports.

### 6.2 What would unlock each remaining rejection

| Rejected item | What would change it |
|---|---|
| figure 2's two curves (617 pts) | **read the annotation block.** Attaching the PNG to the conversation makes it natively viewable — cheapest route by far |
| halation radius in µm | a measured edge-spread or highlight-flare profile from a scan of a known target. **This is an owner measurement, not a document** — no manufacturer sheet in this archive prints a halation radius for any stock |
| rms granularity | any CineStill or independent granularity measurement |
| MTF / f50, resolving power | nothing published by anyone |
| spectral sensitivity | nothing published; FilmLab Pro has no per-stock spectral data at all — theirs is a runtime function, not stored data |
| `dmax`, `latitude_stops` | a schema decision, not a measurement. Both are currently derived from the curve, which is the stricter design |
| push gamma / speed gain per stop | a pushed-development series. The *range* is stored; the *behaviour across it* is not published |
| processing condition | a CineStill statement naming a developer, time and temperature |

---

## 7. One-line summary

**In:** three characteristic curves from 1 440 vendor-traced points (18 parameters, residual
< 0.06 D), the orange-mask ladder and the `dmin_ladder`/`fitted_from` corrections that follow from
it, and the +3-stop push range with its explicit no-fog claim — which needed a new schema struct
rather than an existing field.

**Out:** every FilmLab Pro number, because each one either collides with a higher-tier value, has
no field, is not convertible to our units, has no defined semantics, or is an artifact of their
engine — **not because the source is not a manufacturer.** Plus 617 vendor-traced points whose
curve identity is unknown, and two figures of a different film.

**Everything rejected is archived, itemised above, and each has a named condition that would let it
in.**
