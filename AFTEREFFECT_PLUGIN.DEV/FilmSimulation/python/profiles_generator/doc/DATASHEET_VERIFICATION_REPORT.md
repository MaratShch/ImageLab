# Datasheet verification report — film profile generator

> ⚠ **Status note UPDATED 2026-09-01: the database holds 175 film stocks, 11 print stocks, schema v24.**
> (Read "schema v18" until 2026-08-31, when the constant was found four versions stale — v19–v22
> landed on 2026-08-30/31 with their fields commented and `SCHEMA_VERSION` never bumped. Read
> "schema v15" from 2026-08-26f before that, and "159 stocks, schema v10" from 2026-08-20 before
> that. Every version since v15 is additive and inert, so nothing in this report's findings moves
> with them.) The counts,
> the verify.py assertion and several file paths below are all from 2026-07-31 and are **not current
> state** — use `FilmActiveProfiles.md` (regenerated every build), `NotFound.md` and `PROGRESS.md`
> for that. This report is kept verbatim as the audit record of the 58-stock pass; its *method* and
> its Addendum findings (notably the Callier-coefficient provenance defect, still open) remain
> current, and that is why it is not deleted.
>
> **Test-suite state 2026-09-02**, since every "ALL CHECKS PASSED" line below is a 2026-07-31
> reading: `verify.py` now runs **510 checks, 509 PASS / 1 FAIL** — the one failure is the
> saturation-hierarchy ordering, known and left alone. Alongside it, **30 audit scripts are
> registered and run green**, including cross-language parity audits of the reciprocity, grain,
> MTF, Callier and DIR-coupler stages against the plugin's own C++ (`cpp_parity.py`,
> `interimage_parity.py`, `spectral_mono_parity.py`).
>
> ⚠ **Working-copy caveat 2026-08-23:** this checkout holds only `AGFA`, `FERRANIA`, `FUJI`,
> `GEVAERT`, `KODAK`, `RETRO` and `SVEMA` under `PDF/PROFILES`, so the `KONICA/…` files cited in
> §2, §11 and the Addendum are **not openable here**. This report records where they were read.
>
> **Status note 2026-08-02:** this report describes the 58-stock database as
> verified on 2026-07-31 and is kept verbatim as the audit record. Since then
> the database has grown to **89 stocks**: Konica/Rollei/Kentmere/Eastman
> additions (see `_PROVENANCE_SOURCES` / `_RESOLVING_POWER` /
> `_RECIPROCITY_OVERRIDES` entries dated 2026-07-31/08-01 in
> `film_profiles.py`) and the Soviet reference-book pass of 2026-08-02
> (`SOVIET_EXTRACTION_2026-08-02.md`, `CHANGES_2026-08-02_soviet.md`), which
> also renamed ORWO_UT18 → ORWO_CHROM_UT18 and moved 25 duplicate/no-value
> files into `PDF/PROFILES/DELETE_CANDIDATE/` — some file paths cited below
> now live in that folder. verify.py count assertion is now 89.

**Date:** 2026-07-31
**Scope:** every film stock defined in `profile_generator/film_profiles.py`,
checked against every manufacturer document in `PDF/PROFILES/`.
**Rule applied throughout:** a value was changed only when a manufacturer
document literally prints it. Where nothing is printed, the existing value was
left alone and recorded in `NotFound.md`. No estimation, no interpolation, no
transfer of numbers between emulsions.

---

## 1. Headline result

| | Count |
|---|---|
| Stocks in database | 58 |
| Stocks with usable manufacturer documentation in `PDF/PROFILES/` | **18** |
| Stocks with none — left untouched, logged in `NotFound.md` | **40** |
| Documents examined | 270 PDFs (268 in 14 vendor directories + 2 at root); ~263 unique, 7 byte-identical duplicate pairs |
| Documents with no text layer (unreadable) | 5 |
| **Published values that contradicted the code** | **3** |
| **Previously-empty fields filled from datasheets** | **11** |
| Values confirmed correct (no change needed) | 14 |
| False or missing citations corrected | 7 |
| Confidence tiers corrected (2 down, 2 up) | 4 |

Test suite: `verify.py` now reports **ALL CHECKS PASSED**. It failed *before*
this work began (it asserted 56 stocks against a database of 58); that one-line
assertion was corrected with your approval, and no other test was touched.

---

## 2. Method

1. `pdftotext -layout` over all 270 PDFs into a text cache; 265 yielded text
   (5 are pure scans: `FUJI/Neopan1600.pdf`, `FUJI/Neopan400.pdf`,
   `KODAK/transmision of wratten filters.pdf`, `KONICA/centuria_pro_400.pdf`,
   `KONICA/professional_160.pdf`).
2. A page-aware search helper (`pq.py`) mapped every hit back to a physical PDF
   page, so every claim below cites `file` + `page`.
3. Each stock was searched by product name, brand, catalogue number, and
   plausible alternative spellings including Cyrillic transliterations.
4. Every extracted value was recorded together with the **verbatim source line**,
   and each verbatim line was then machine-checked as a literal substring of the
   cache file. Roughly 1,020 quotations were verified this way; 0 failures.
5. Two archive files that mention nearly every stock —
   `Film_Profile_Model_Domain_Review.pdf` and
   `Модель данных профилей киноплёнки.pdf` — were excluded as evidence: they are
   this project's own design notes, not manufacturer documents. Likewise
   `PDF/PROFILES/SVEMA/*.txt` and `TASMA/*.txt`, which are output of this
   project's own `analyze_film_scans.py`.

Nothing under `PDF/PROFILES/` was modified, renamed, moved or deleted.

---

## 3. Corrections — published value contradicted the code

### 3.1 `FOMAPAN_400_ACTION` · `rms_granularity` 11.5 → **17.5**

> `RMS = 17.5 (Microphen at 20 oC, developed to  = 0.6 (measured at D = 1.0)`
> — `PDF/PROFILES/FOMACOLOR/fomapan-400.pdf` p1

The most consequential single error found. The code understated Foma's published
granularity by 52 %, which for the renderer means this stock was reproducing
roughly two-thirds of its real grain amplitude. Foma states the measurement
conditions explicitly, and they match this database's metric definition
(σ(D)·1000 at D = 1.0), so the figure is adopted verbatim rather than rescaled.

### 3.2 `AGFA_OPTIMA_100` · `rms_granularity` 7.8 → **4.0**

> `Granularity (x 1000):  RMS 4.0`
> — `PDF/PROFILES/AGFA/agfa_films.pdf` p7, under `AGFACOLOR OPTIMA II 100`

The code was almost twice the published figure. This stock had also been marked
`_NO_DATASHEET`; the document exists but specifies the film across the pages of
a multi-film brochure rather than in a single-stock sheet, which is presumably
why the earlier audit missed it. The per-channel `rms_r/g/b` values re-derive
automatically from the corrected figure via `_grain_v2`'s existing stack rule
(Agfa publishes no per-layer granularity).

### 3.3 `POLAROID_667` · curve dmin/gamma/dmax 0.16 / 1.15 / 1.85 → **0.10 / 1.55 / 1.75**

> `At 71o F/21o C:   D-Max = 1.75   D-Min = .10   Slope = 1.55`
> — `PDF/PROFILES/POLAROID/667fds.pdf` p2

The only numeric characteristic-curve data available for any stock in this
database. (Polaroid publishes the same D-Max/D-Min/Slope triple for about a
dozen other pack and sheet types — 53/553/803, 55/85, 572/672/72, 579/679/879 —
none of which has a profile here.) Polaroid defines Slope by the ¼–¾ increment method — the
average gradient of the straight-line section — which is exactly what
`ToneCurve.gamma` represents, so `dmin` and `gamma` are direct transcriptions.
The toe/shoulder separation is then **forced** by the published D-max:
`shoulder_x − toe_x = (1.75 − 0.10) / 1.55 = 1.0645` decades. The pair was
placed symmetrically about the previous mid-scale anchor (+0.035) so that
matching the published densities does not silently shift the stock's exposure
placement; `toe_k` and `shoulder_k` are untouched. Three published constraints,
three fitted parameters — no free choices.

---

## 4. Previously-empty fields filled from datasheets

### 4.1 Resolving power — `MTFSpec.resolving_power_lp_mm_lowc/highc`

The schema already reserved these fields (DM-11) and only two stocks used them.
Nine more are now populated. Every entry is a transcription; nothing is
interpolated between contrasts or between films. `0.0` means the manufacturer
does not publish that contrast.

| Stock | 1.6:1 | 1000:1 | Source |
|---|---|---|---|
| `AGFA_APX_25` | — | 200 | `AGFA/agfapanapx25.pdf` p1 |
| `AGFA_APX_100` | — | 150 | `AGFA/apx100.pdf` p1 |
| `AGFA_APX_400` | — | 110 | `AGFA/apx400.pdf` p1 |
| `AGFA_OPTIMA_100` | 50 | 140 | `AGFA/agfa_films.pdf` p7 |
| `FUJI_PROVIA_400X` | 55 | 135 | `FUJI/Provia_400X_PIB_1007.pdf` p6 |
| `FUJI_SENSIA_100` | 55 | 135 | `FUJI/sensia_100_datasheet.pdf` p4 |
| `FOMAPAN_400_ACTION` | — | 90 ⚠ | `FOMACOLOR/fomapan-400.pdf` p1 |
| `POLAROID_664` | — | 20 ⚠ | `POLAROID/664fds.pdf` p2 |
| `POLAROID_667` | — | 14 ⚠ | `POLAROID/667fds.pdf` p2 |

Two caveats are flagged in the code as well as here, because they are the only
two places in this pass where a judgement was made rather than a transcription:

* **Foma** prints "90 lines per mm" with **no test-object contrast stated**. The
  number is verbatim; assigning it to the high-contrast slot follows the
  industry convention for an unqualified figure, but that label is an
  interpretation of Foma's omission.
* **Polaroid** publishes **ranges** (664: 20–25, 667: 14–20). A single float
  cannot hold a range, so the **lower bound** is recorded. A midpoint would have
  been an invented number.

### 4.2 Reciprocity — `ReciprocitySpec`

Harman/ILFORD print the reciprocity correction as an adjusted **time**,
`Ta = Tm^k`. This model writes effective exposure as `E_eff = I·t^p`. Correct
exposure requires `I·Ta^p == I·Tm`, i.e. `Ta == Tm^(1/p)`, therefore
**`p = 1/k` exactly** — an algebraic identity, not a fit.

| Stock | Published | Derived `p` | `onset_s` | Source |
|---|---|---|---|---|
| `ILFORD_HP5_PLUS_400` | `Ta = Tm^1.31` | 0.7634 | 0.5 | `ILFORD/HP5-Plus_201811.pdf` p2 |
| `ILFORD_DELTA_3200` | `Ta = Tm^1.33` | 0.7519 | 0.5 | `ILFORD/Delta-3200_201811.pdf` p2 |

Both previously carried the generic B&W estimate `p = 0.95, onset 1.0 s`, which
badly understated real long-exposure behaviour. Implemented via a new
`_RECIPROCITY_OVERRIDES` table consulted before the existing heuristics in
`_reciprocity_for` — matching the established `_TEMPORAL_OVERRIDES` pattern, no
architectural change.

Note on Delta 3200's onset: the 2018 sheet is internally inconsistent (it says
no correction is needed from ½ s, then refers to "exposures longer than 1
second"). 0.5 s is used because the 2002 edition states ½ s unambiguously. This
is documented in the code comment.

**Agfa's reciprocity tables were deliberately NOT applied.** Agfa prints a
discrete table (APX 100: 1 s → +1 stop, 10 s → +2, 100 s → +3). A single
Schwarzschild exponent cannot reproduce those three points — `p` would have to
be 0.40 at 10 s and 0.55 at 100 s — so fitting one would have invented data. The
table is transcribed into the code comment instead.

---

## 5. Values confirmed correct

Verified against the printed figure, unchanged, and now carrying a
file-plus-page citation in place of the previous vague "(manufacturer
datasheet)" note:

| Stock | Field | Value | Source |
|---|---|---|---|
| `AGFA_APX_25` | rms | 7.0 | `agfapanapx25.pdf` p1 |
| `AGFA_APX_100` | rms | 9.0 | `apx100.pdf` p1 (+2 corroborating docs) |
| `AGFA_APX_400` | rms | 14.0 | `apx400.pdf` p1 |
| `FUJI_VELVIA_50` | rms | 9 | `velvia_50_datasheet.pdf` p7 |
| `FUJI_PROVIA_400X` | rms | 11 | `Provia_400X_PIB_1007.pdf` p6 |
| `FUJI_SENSIA_100` | rms | 10 | `sensia_100_datasheet.pdf` p4 |
| `FUJI_NEOPAN_ACROS_100` | rms | 7 | `NeopanAcros100.pdf` p4 |
| `FUJI_NEOPAN_ACROS_100` | reciprocity onset 120 s | — | datasheet-backed, not an estimate |
| `EKTACHROME_64` | rms | 11 | `KODAK/e8-Ektachrome_64_EPR.pdf` p5 |
| `EKTACHROME_160T` | rms | 13 | `KODAK/e144-Ektachrome_160T_EPT.pdf` p4 |
| `KODACHROME_64` | rms | 10 | `KODAK/e88-2009_06.pdf` p4 — second independent Kodak publication |
| `ILFORD_HP5_PLUS_400` | ISO 400/27 | — | `HP5-Plus_201811.pdf` p1 |
| `FOMAPAN_400_ACTION` | ISO 400/27 | — | `fomapan-400.pdf` p1 |
| `POLAROID_664` / `667` / `SX70` | ISO 100 / 3000 / 150 | — | `664fds.pdf` p2, `667fds.pdf` p2, `timezfds.pdf` p1 |

One trap avoided: `KODAK/e88-2009_06.pdf` prints `Diffuse rms Granularity: 10`
on p4 under the heading **KODACHROME 64 Film** and `16` on p5 under
**KODACHROME 200 Film**. Only the p4 figure applies here.

---

## 6. Citation accuracy — corrected claims

This category changed no rendered value, but it is where the database's
scientific credibility was weakest: several stocks asserted datasheet
provenance for numbers no manufacturer has ever published.

| Stock | Problem | Action |
|---|---|---|
| `AGFA_OPTIMA_100` | marked `_NO_DATASHEET`, but Agfa documents it | real citation added |
| `POLAROID_664` | marked `_NO_DATASHEET`, but a full Film Data Sheet is on file | real citation added |
| `POLAROID_667` | same | real citation added |
| `POLAROID_SX70` | same, but the document is a product page only | citation added, explicitly annotated "no technical-data section" |
| `ILFORD_HP5_PLUS_400` | tier 1 "datasheet_curve"; its `rms 9.0` is not a published number | tier → 2; `rms` left unchanged, annotated as an estimate |
| `ILFORD_DELTA_3200` | same, `rms 16.0` | tier → 2; same treatment |
| `KODAK_PORTRA_400` | tier 1; its `rms 4.0` is not a Kodak figure | tier kept, `rms` annotated in detail (see below) |
| `_PROVENANCE_SOURCES` | citations imply the documents are on file | archive caveat added listing the 12 entries whose documents are **not** in this repository. ⚠ **REVERSED 2026-08-18: 11 of those 12 documents are in fact on disk** — the caveat was wrong, and it declared absent two documents this project's own code opens. Only CINESTILL 800T is genuinely absent. See `NotFound.md` §0.3 |
| `AGFA_OPTIMA_100` | `[T2]` although speed + granularity + resolving power are all published | promoted to `[T1]` |
| `FUJI_SENSIA_100` | `[T2]` although it has identical evidence to `FUJI_PROVIA_400X` (`[T1]`) | promoted to `[T1]` |
| `POLAROID_664` / `POLAROID_667` | gained real citations | deliberately kept `[T2]`: tier 1 needs a granularity figure and Polaroid publishes none |

### 6.1 Why Ilford was downgraded

**Harman/ILFORD publish no diffuse RMS granularity, no resolving power and no
MTF for any emulsion** — verified across all 18 ILFORD datasheets (20 ILFORD
PDFs, one of which is a Kodak-equivalence table and one a processing chart) and
both Kentmere sheets. Only qualitative prose ("outstanding resolution").

Correction from the independent audit (§11): Harman *does* publish numeric
average gradient for some emulsions — `Gbar 0.62` for Delta 400, a full G-bar
table for Ortho Plus — so "Ilford publishes no numbers" would be too strong.
It publishes no granularity and no sharpness figures, which is what these two
tiers turn on, and no G-bar for either HP5 Plus or Delta 3200 specifically.
What *is* documented for these two is ISO speed, the exact processing conditions
the printed curve represents, the reciprocity formula, and a full development
matrix.

Under this database's own tier definition ("published ISO speed, RMS
granularity or diffuse grain number, **and** an MTF or resolving-power figure
exist"), that is tier 2, not tier 1. `HP5_PLUS`'s `rms 9.0` is additionally
implausible on its face: Agfa's *published* figure for the comparable
cubic-grain APX 400 is 14.0. It was left unchanged — there is no documented
replacement — but it is now flagged in the source so nobody cites it as a
datasheet value.

### 6.2 The Kodak grain-metric problem

Kodak publishes **two mutually incomparable grain metrics** and switched between
them by product class:

* **Diffuse rms granularity** — reversal and B&W films (Ektachrome, Kodachrome,
  T-Max, Tri-X, Plus-X).
* **Print Grain Index (PGI)** — all modern colour negative (Portra, Ektar, Gold,
  Ultra Max). For `PORTRA_400`: **PGI 37 / 59 / 89** for 135 format at 4.4× /
  8.8× / 17.8× magnification (`e4050_portra_400-2016.pdf` p3).

Every Kodak colour-negative sheet states PGI "replaces rms granularity and has a
different scale which cannot be compared to rms granularity", and Kodak
publication E-58 (`Kodak_Print-Grain-Index_E-58.pdf`) publishes **no conversion
factor**. So `PORTRA_400`'s `rms_granularity = 4.0` cannot be derived from
Kodak's published data without inventing a conversion. It was left as-is and
annotated; the PGI triple and the Status M densitometry are recorded in the
comment. If you want PGI machine-readable, that needs a new schema field — a
deliberate design decision I did not take unilaterally.

---

## 7. What no manufacturer publishes

A structural finding worth stating plainly, because it bounds how far this
approach can go:

**Not one of the 270 documents prints a tabulated density-vs-logE array or
tabulated MTF percentages.** Characteristic curves, MTF curves and spectral
sensitivity are published exclusively as plotted images, by every manufacturer,
in every era.

Scalar sensitometry does survive in running text, and the independent audit
(§11) was right to flag that the first draft of this section overstated the
case. What exists: Polaroid's D-Max/D-Min/Slope triple for 667 (§3.3) and for
about a dozen unprofiled pack and sheet types; Harman's numeric average gradient
for Delta 400 (`Gbar 0.62`) and a full G-bar table for Ortho Plus; Kodak's B&W
contrast-index aims (0.56 for Plus-X and Tri-X); and a handful of prose spectral
peaks (Konica Infrared 750 nm, Ilford SFX 720 nm, Rollei PAN 25 400-650 nm).
None of that is a curve, and none of it covers a stock in this database other
than Polaroid 667.

Consequence: `ToneCurve.*`, `MTFSpec.f50_*`, `spectral_weights`, `dye_matrix`
and the halation parameters **remain engineering estimates for all 58 stocks**,
and cannot be grounded by text extraction at all. Grounding them requires
digitising the plots (WebPlotDigitizer or equivalent) — which the module's own
header has recommended since v1. That is now the largest single accuracy
opportunity in the project; see `NotFound.md` §4 (identified, extractable, not
yet extracted).

⚠ **Superseded for MTF, and the recommendation above is what did it.** Plot
digitising happened: `mtf_vector.py` has traced **26 curves off 12 sheets**, and
**23 stocks now carry `mtf_measured`** (⚠ 16 -> 17 on 2026-08-31, queue E3: KONICA_IMPRESA_50, the first entry that is neither vector-traced nor per-layer -- its sheet is a scan, the curve is traced off the bitmap by konica_raster.py, and the panel prints one visual-filter curve, so f50 64.9 is pooled and written to all three fields) (⚠ 15 -> 16 on 2026-08-30: KODAK_PORTRA_400VC only, of the four new PORTRA NC/VC stocks -- the other three have traced f50 triples but their rolloff fits at rms 0.093-0.122 and beats the Gaussian by only 1.2-1.3x, which does not license switching the carrier) (EASTMAN_PLUS_X_5231 and EASTMAN_DOUBLE_X_5222 mono plus the colour
5201, 5274, 5217, 5218, 5245, 5248, 5279 and — traced 2026-08-23, count
corrected here 2026-08-25 — FUJI_SUPER_F125_8532 and FUJICOLOR_SUPER_F500_8572, plus KODAK_EKTACHROME_100D_5285 traced 2026-08-25 -- the first colour REVERSAL stock with a measured MTF, and the largest correction so far: stored f50_g 82.0 against a measured 42.1, i.e. the estimate was 1.95x too sharp),
with the rolloff exponent stored per stock rather than assumed. Two more (5205, 5293) have measured green and blue but
a REFUSED red, so they carry a mixed triple and are deliberately **not** flagged
measured. **63 colour stocks still carry an estimated f50 triple**, so the
sentence above still holds for most of the database — but no longer for all of
it, and no longer as a statement about what text extraction can reach.

What manufacturers *do* publish numerically, by vendor:

| Vendor | RMS granularity | Resolving power | MTF | Numeric gamma |
|---|---|---|---|---|
| Agfa | ✅ | ✅ | ❌ | γ-vs-time tables in the chemicals brochures |
| Fujifilm | ✅ | ✅ both contrasts | ❌ | ❌ |
| Kodak | ✅ reversal/B&W · PGI for colour neg | ✅ B&W only | ❌ | contrast-index aims, B&W only |
| Ilford / Kentmere | ❌ | ❌ | ❌ | ❌ |
| Foma | ✅ with full conditions | ✅ (contrast unstated) | ❌ | target γ in dev tables |
| Polaroid | ❌ | ✅ as a range | ❌ | 667 only |
| ORWO / Filmotec | ✅ with 48 µm aperture | — | ✅ single `m30` factor | — |
| MACO | — | ✅ (up to 330 lp/mm) | ❌ | target γ 0.65 |
| Rollei | ✅ (Infrared only) | ✅ | ❌ | ❌ |

---

## 8. Files

**Modified** (backup taken first — see §9):

* `profile_generator/film_profiles.py` — all data changes above, each with an
  inline source comment naming the PDF and page.
* `profile_generator/verify.py` — one line: stock-count assertion 56 → 58,
  correcting a pre-existing failure.

**Created:**

* `profile_generator/NotFound.md` — the 40 undocumented stocks, the specific
  parameters missing for each, and a ranked list of what is recoverable.
* `profile_generator/DATASHEET_VERIFICATION_REPORT.md` — this file.

**Regenerated:** `film_profiles.hpp`, `film_profiles.cpp` via `cpp_codegen.py`.

**Not modified:** `film_sim.py` — no change proved necessary. Every value the
datasheets supplied fits a field the schema already had. (Note: your brief
referred to `film_simulator.py`; the file in this project is `film_sim.py`.)

**Untouched, as instructed:** everything under `PDF/PROFILES/`.

**Evidence trail** (in the session outputs folder, not the repository):
`facts_AGFA.md`, `facts_FUJI.md`, `facts_ILFORD_MACO.md`,
`facts_KODAK_STILL.md`, `facts_POLAROID_FOMA_ORWO.md`, `facts_CINE_AUDIT.md` —
roughly 1,020 cited extractions with verbatim source lines; plus
`film_profiles.diff`.

---

## 9. Backup

Taken before any edit, as standing instruction:

* `PYTHON/_backup_20260731_011646/` — full copy of `profile_generator/`,
  `old_profile/` and `!FilmValidated!.txt`.
* A second mirror was written to the session outputs folder.

---

## 10. Open questions for you

1. **PGI field?** Adding `print_grain_index` to `GrainSpec` would let Kodak's
   published colour-negative grain data be stored instead of only commented.
   Schema addition, so your call.
2. **Do GOST handbook tables count as provenance?** Iofis 1980 and Gurlev
   contain real published Soviet film data (speed, γ, D₀, D₀max, latitude,
   resolving power). They are state handbooks, not manufacturer datasheets, and
   the Gurlev table carries no Svema attribution. Accepting them would make most
   of the Soviet block groundable; I applied none of it pending your decision.
3. **`EASTMAN_EKTACHROME_7239` designation.** Iofis table 22 p150 prints 7239 as
   **VN**, not **EF**; the EF numbers there are 5241/5242. The profile's aliases
   assume EF.
4. ⚠ **CLOSED — the five "missing" Kodak H-1 sheets** (5203, 5207, 5213, 5219,
   5285) were **never missing**: all five were found on disk on 2026-08-18 and
   read digit for digit (`NotFound.md` §0.3). Nothing to request.
5. ⚠ **CLOSED — `FUJI/Neopan1600.pdf` did not need OCR.** A true digital PDF of
   the same sheet (Ref. AF3-608E) was extracted on 2026-08-15 and supersedes the
   scan entirely; the curve was refitted to 487 traced points. See `Found.md`.

---

## 11. Independent adversarial audit

The whole pass was then re-checked by a second, independent verification run
whose brief was to *falsify* every claim: 30 numbered claims, each traced back to
the cited PDF page, plus a field-level regression diff of all 58 profiles between
the pre-edit backup and the current module.

**Result: 27 verified, 3 refuted, 0 unprovable.** All three refutations were
citation hygiene, not wrong data — every value change, every resolving-power row
and both reciprocity exponents were confirmed correctly transcribed. Specifically
confirmed: the Optima II 100 three-column layout trap was avoided (the 4.0 does
belong to the 100, double-sourced), the Polaroid 667 arithmetic is exact, and
Kodachrome 64/200 are not cross-wired.

Refutations, all now fixed in code and in this report:

1. **Acros page citation was wrong** — `NeopanAcros100.pdf` has 6 pages; the data
   is on **p4**, not p7. Values unaffected. Corrected in three places.
2. **"Ilford publishes no numbers" was too strong** — Harman does print numeric
   average gradient for Delta 400 and Ortho Plus. The granularity / resolving
   power / MTF claim holds, and neither HP5 Plus nor Delta 3200 has a published
   G-bar. Also: 20 ILFORD PDFs, of which 18 are datasheets — not 21.
3. **"667 is the sole numeric curve in the archive" was too strong** — about a
   dozen other Polaroid types print the same triple, and a few sheets give
   numeric spectral peaks in prose. 667 remains the only such data for a stock
   *in this database*. §7 rewritten.

Further defects the audit caught and that are now fixed:

4. `POLAROID_SX70`'s citation invented the word **"Supercolor"**, which appears
   in none of the 270 documents. Corrected to the printed title.
5. The Delta 3200 comment presented a **paraphrase inside quotation marks**. Fact
   correct, quotation marks not earned — reworded as an explicit paraphrase.
6. **The tier rule had been applied in one direction only** (Ilford down, nobody
   up). Optima 100 and Sensia 100 are now promoted; Polaroid 664/667 examined and
   correctly left at T2.
7. **Latent reciprocity discontinuity.** With the bare `E_eff = I·t^p` form and
   `onset_s = 0.5`, t = 0.5 s yields 0.24 stops of spurious failure exactly where
   the datasheet says there is none (it was 0.05 stops with the old p = 0.95, so
   the correct exponents enlarged a pre-existing wart). The `p = 1/k` algebra is
   exact above the onset; the fix is the onset-normalised form
   `E_eff = I·t·(t/onset)^(p−1)`, which is now documented in the code.
   ⚠ **REVERSED 2026-08-23: this is no longer latent — reciprocity is WIRED and
   live** (queue C8). Both renderers consume it: `RenderSettings.exposure_time_s`
   in Python and `AlgoControls::exposureTimeS` in the C++ port, with `0` meaning
   inert by contract, so the normalised form above is now a live requirement and
   not a note for a future renderer. The carrier behind it has grown too:
   **21 measured `ReciprocityTable` entries** (was 6) after 15 were read from
   vendor sheets, and **105 stocks carry a Schwarzschild exponent**. ⚠ Read the
   model for what it is: a per-channel **GLOBAL log-exposure shift**, because no
   source in the corpus has an intensity axis — it cannot express
   exposure-dependent failure within one frame. The wiring is covered by a
   cross-language parity audit against the plugin's own C++ (`cpp_parity.py`).
8. **Unit conflation.** The field is named `..._lp_mm`, but Agfa, Fuji and Foma
   print *lines*/mm while Polaroid prints *line pairs*/mm, and no sheet states an
   equivalence. Values are stored as printed and the discrepancy is now flagged
   in the table's docstring rather than silently normalised.
9. Archive-scope numbers tightened: 270 PDFs across 14 directories (2 at root),
   ~263 unique after 7 duplicate pairs, 5 files with no text layer.
10. Page-number convention clarified in `Found.md`: manufacturer PDFs are cited
    by physical PDF page, the Iofis/Gurlev books by printed book page.
11. `NotFound.md` §11 wrongly said the Konica Infrared 750 files have no text
    layer; they are duplicates that do, and are the source of the 750 nm figure.

**Regression diff:** 58 profiles before, 58 after, none added or removed.
**13 stocks differ and every difference is sanctioned; zero collateral movement
across the other 45.** A comment-stripped token-level diff of the module confirms
no other source change and no accidental deletion, and re-running
`cpp_codegen.py` reproduces `film_profiles.hpp`/`.cpp` byte-for-byte.

`verify.py`: **ALL CHECKS PASSED**, including `grain reproduces datasheet RMS
granularity max err=1.31%`, `granularity never exceeds the datasheet figure max
ratio=0.997`, and `500T grainier than 50D 2.54x = datasheet ratio 2.54x`.

Full audit trail: `VERIFICATION_AUDIT.md` in the session outputs folder.

---

# Addendum — 2026-07-31 (second pass): scan batches + 13 new stocks

## A. Measured scan-batch adoptions (analyzer v2.1 TXT files)

Rule for this pass: measured values win over estimates, BUT a measurement
that contradicts physics or its own provenance is rejected with the reason
recorded in the profile comment. Grain and tints received priority per the
owner's instruction.

| Stock | Adopted | Rejected (reason) |
|---|---|---|
| SVEMA_FN_64 (509 frames) | ⚠ **THIS ROW IS SUPERSEDED — see the 2026-08-18 addendum at the end of this file.** As adopted at the time: gamma 0.83; sigma shape 0.65/1.0/1.65; base_tint (.991,1,.991); silver_tone **+0.40 (sign reversal**, dense areas measure warm; crow-wing shadows survive as the complement) | anisotropy 0.66 (Bayer artefact); absolute dmin (no empty gate) |
| SVEMA_FOTO_250 (26 frames) | gamma 0.85 (measures SAME slope as FN-64, not higher); rms 25->**33 capped** (literal fit demanded ~70 -- beyond any emulsion; mid-bin scene leakage in a 26-frame web batch); shape 0.67/1.69; halation ON 0.14 @ 175 um | corr length (0.86 px, under scan floor) |
| TASMA_FN_64 (132 frames) | gamma **1.03** (Tasma measures HIGHER contrast than Svema); rms 12.4->**20 capped** (fit demanded ~55); shape toe 0.36, dmax capped 1.0 (measured dense<mid = leakage); silver_tone 1.0->**0.30** (measured warm drift is real but 3x gentler than the memory-based guess); halation ON 0.06 @ 120 um | dense-bin shape at face value |
| ORWOCOLOR_NC21 (109 frames) | rms 12->**18 capped** (fit demanded ~92); shape 0.50/1.80; halation g/b raised to 0.15/0.12 | batch gammas 0.92-1.10 (C-41-class chemistry pins ~0.5-0.65; per-channel batch stats on colour scans are scene/mask-polluted); blue-dominant halo (physics: through-base halation is red-dominant) |

Common caveat: every DSLR/web batch lacks --empty-gate, so absolute
densities stay unknowable; base-relative quantities were used throughout.

## B. New stocks (13) — 71 total now

**From the owner's scan batch:** ORWO_UT18 [T2] (78 aged slides; yellow
extremes from the crossover bin medians, halation 0.28 D measured; linear
tone-slope figures NOT used — regression on colour material is
scene-dominated).

**From datasheets [T1 where printed]:**
- KONICA_INFRARED_750 — 640-820 nm band, peak 750, ISO 32 (TDSB-701)
- KONICA_IMPRESA_50 — 63/160 lp/mm, reciprocity p=0.87 fitted from +1/2@10s
- KONICA_VX_100 — RMS 4 printed; 63/125; p=0.77 (+1 stop@10s)
- KONICA_CENTURIA_SUPER_400 — RMS 4; 50/100; p=0.77
- KONICA_CENTURIA_SUPER_1600 — RMS 6; 50/100; p=0.77
- KONICA_CHROME_CENTURIA_100 — RMS 11; 60/140; no correction to 4 s, p=0.80-0.82 from the 64 s row (+CC10C -> p_r highest)
- KONICA_CHROME_R100 — RMS 11; 50/125; the 1 s reciprocity cliff (+1/2 + CC5R)
- ROLLEI_R3 — gamma 0.65 target, base+fog ~0.28 from the curve, p=0.68 fitted from the printed table (severe), clear-PET halation look
- ROLLEI_INFRARED_400 — RMS 11.0 printed, 160 lp/mm, 820 nm reach, AURA halation
- ROLLEI_RETRO_400 — 380-630 nm (dark reds = the retro), 110 lp/mm, triacetate
- KENTMERE_PAN_100 — Ta=Tm^1.26 -> p=0.794; grain/curves T3 (sheet prints none)
- KENTMERE_PAN_400 — Ta=Tm^1.30 -> p=0.769; grain/curves T3

**Not possible from this folder:** 2-3 Konica B&W were requested, but the
KONICA folder contains exactly ONE B&W emulsion (Infrared 750, present
twice). The remaining sheets are colour negative/reversal. Second and third
Konica B&W profiles need additional documents.

## C. Code fix found during the pass

`_grain_v2` silently overwrote author-set `sigma_shape_*` values with the
era heuristic — it had eaten two rounds of measured FN-64 shape adoptions.
The heuristic now applies only when the literal still carries the dataclass
defaults. Verified: measured stocks keep their shapes, untouched stocks
still get the heuristic.

---

# Addendum — 2026-08-18: the SVEMA scan batch is not one emulsion

This addendum supersedes the `SVEMA_FN_64 (509 frames)` row in section A of
the 2026-07-31 addendum above, and corrects a factual error about the
scanning device that was repeated in this file and in three others.

## What was wrong

**The batch mixes two films.** `analyze_film_scans.py v2.1` was pointed at a
folder named `SVEMA-FN64` and analysed all **509 frames as a single
emulsion**. The owner confirms (2026-08-18) that only frames
`PICT0001`–`PICT0067` are certainly **Foto-65**; frames 68 onward are a
**mixture of Foto-32 and Foto-65** that cannot be resolved frame by frame.
Foto-32 was used deliberately at the time for finer grain and higher
resolution when making large prints, so the contamination is
**one-directional** — the mixed batch reads *finer and sharper* than
Foto-65 alone. Nothing was fabricated: the numbers are the genuine output of
a real v2.1 run (`PDF/PROFILES/SVEMA/SVEMA-FN64_generated_film_profile.txt`,
header `Analyzed Frames: 509`). The defect is the label on the folder, not
the analyzer.

**The scanner was misidentified.** This file, `REPORT_FN64_355.md`,
`FilmDatabase_Charecteristics.MD` and the `film_profiles.py` comments all
described a "Bayer-demosaiced DSLR" rig. EXIF on the owner's frames reads
`Make=GCMC`, `Model=Scanner`, `Software=UF15 16/08/20 v0.69`; 4416 px / 36 mm
= 122.7 px/mm = 3116 dpi, 1 px = 8.15 µm. No Bayer mosaic is established.

## The measurement that settles most of it

Over **all 67** confirmed Foto-65 frames: **`max |R−G| = max |B−G| = 0`.**
They are exactly greyscale.

Therefore every per-channel quantity in the 509-frame output — `base_tint`,
`tone_slope_r/_b`, all twelve crossover bins, and the 0.806 / 0.834 / 0.850
gamma spread — originates **entirely in the contaminated 68+ tail** and
cannot be attributed to this emulsion. The analyzer's own
`[SpectralResponse]` note says the same thing from first principles: a scan
of a B&W silver negative carries no memory of which wavelengths exposed it.

## Confirmed-subset re-run vs the mixed batch

Same script, same version, the 67 confirmed frames, `--px-per-mm 122.7`.

| Parameter | mixed 509 | confirmed 67 | outcome |
|---|---|---|---|
| `gamma_g` | 0.834 | **0.677** | both `[ESTIMATE]`; bracket is wide |
| σ(D) toe/mid/dense (green) | 0.0191/0.0292/0.0482 → 0.65/1.00/1.65 | 0.0479/0.0425/0.0435 → **1.13/1.00/1.02** | **sign flips — withdrawn** |
| `corr_len_px` | 3.42 | 3.63 | +6 % coarser, direction as predicted |
| `base_tint` | (0.991, 1.000, 0.991) | (1.000, 1.000, 1.000) | **withdrawn** |
| `tone_slope_r` / `_b` | −0.0205 / +0.0079 | **0.0000 / 0.0000** | **silver_tone withdrawn** |
| all 12 crossover bins | up to ±0.0016 | exactly 0.0000 | artefact |
| `anisotropy` | 0.658 | 0.634 | reproducible; rejection *reason* was wrong |
| halation `strength_g` | 0.1992 | 0.1656 | −17 %, kept |
| `coating_sigma_d` | 0.0643 | 0.1272 | both `[UPPER-BOUND]`; see note |

The σ(D) bins are **not** the explanation. `analyze_film_scans.py` sets them
as absolute offsets from `d_base` (`+0.05 / +0.35 / +0.95`), and the two
`d_base` values differ by only 0.024 D, so the two runs sample essentially
the same absolute density windows. The disagreement is real.

**Stated as unexplained:** confirmed-67 σ_toe (0.0479) is **2.5×** the mixed
batch's σ_toe (0.0191). Adding 442 frames should not move a toe-bin
statistic that far. No account of it survives inspection, which is precisely
why the shape was *withdrawn* rather than replaced with the new triple —
neither run has earned adoption, and the estimator is scanner-noise
dominated by the analyzer's own admission ("treat as upper bounds").

**`coating_sigma_d` and `vignette_d` roughly doubled, and this is NOT
evidence of worse coating.** Both are `[UPPER-BOUND]` figures inflated by
scene similarity, and 67 frames from one shooting period are far less varied
than 509. The bound got looser, not the film worse.

## Adopted changes

| Field | Was | Now | Basis |
|---|---|---|---|
| `base_tint` | (0.991, 1.000, 0.991) `[T2]` | **(1.0, 1.0, 1.0)** | confirmed frames are exactly greyscale |
| `silver_tone` | **+0.40** `[T2]` | **0.0** `[T3]` | the reversal rested on `tone_slope_r −0.0205`, which is 0.0000 on the confirmed frames |
| `sigma_shape_toe/dmax` | 0.65 / 1.65 `[T2]` | **withdrawn → 0.4/1.0/1.2** `[T3]` | the two runs disagree in sign; conflict recorded, never averaged |
| `gamma` basis | 509-frame batch | **Gurlev 1986 p296** (γ_rec 0.8) | method rule 14: printed source outranks a derived estimate. Value 0.830 unchanged |
| `anisotropy` rejection reason | "Bayer mosaic on a DSLR scan" | **open question** | device is a GCMC/UF15 scanner; measurement is reproducible (0.658 / 0.634). Value 1.10 `[T3]` unchanged |
| tier note for `SVEMA_FOTO_65` | measured batch **and** printed source | printed source alone carries tier 2 | the batch is demoted to a bracket; tier 2 survives on Gurlev |

`silver_tone` was set **neutral, not restored to the earlier −0.10.** That
figure came from the same rig and the same class of artefact; restoring it
would swap one unsupported number for another. Image tone of a developed
silver negative is a real physical effect — this records the *absence of an
admissible measurement*, not a claim that Foto-65 is neutral.

The σ(D) fallback 0.4/1.0/1.2 is the defensible default here rather than an
embarrassment: σ ∝ √D is the textbook Poisson-counting result for a B&W
**silver** negative and it rises. The *falling* triples adopted for the Vision3
stocks are measured on **chromogenic** colour negatives, a different mechanism —
that sign must not be transferred to this stock.

⚠ **AND THE ARGUMENT WAS VINDICATED ON 2026-08-25b, from the other direction.**
The first measured σ(D) on a black-and-white stock — KODAK TRI-X Reversal 7266,
traced from its own granularity panel — **RISES 2.8× toward dmax**, against a
stored estimate that had it falling to 0.50. 13 stocks now carry measured
triples, not four, and one of them contradicts the chromogenic sign outright. On
reversal film dmax is the unexposed, fully developed silver, so rising is the
physical direction. The enumeration "the four Vision3 stocks" above is therefore
superseded; the do-not-transfer conclusion it supports is stronger than when it
was written.

## Knock-on caveats recorded but NOT re-fitted

Two other profiles lean on the contaminated batch as a comparator. Neither
value is changed, because in both cases the bias pushes in the direction the
existing adoption already went, and re-fitting on a thin batch would trade a
documented weakness for an undocumented one.

* **`SVEMA_FOTO_250` gamma 0.85** — adopted on the reasoning that its
  26-frame batch (0.844) measures the *same* slope as FN-64's 0.834, "not
  higher". That comparator is the mixed batch; the Foto-65-only subset gives
  0.677. The conclusion still holds directionally but is weaker than it reads.
* **`SVEMA_FOTO_250` rms 25.0** — fitted from a flat-region σ ratio of
  FN250 0.0502 against "SVEMA_FOTO_65's 0.0299" over 3 supplied scans. Which
  3 frames were used is not recorded, so the denominator may be biased **low**
  (Foto-32 is finer), which would make the true ratio *smaller* than 1.68×.
  The shipped value was already capped well below what a literal fit demanded,
  so the bias pushes the same way the cap did.

## Regression guards added

`verify.py` gains 4 checks (169 PASS / 2 FAIL on the day; the 2 failures were the
long-standing saturation-hierarchy and neighbour-pair-coupling ones. ⚠ Current
state 2026-08-23: **304 checks, 303 PASS / 1 FAIL** — only the
saturation-hierarchy ordering still fails; the neighbour-pair-coupling failure is
gone): `base_tint` stays identity, `silver_tone` stays 0.0,
`sigma_shape` is the B&W default and not either scan run, and the
`film_profiles.py` provenance-warning block is still present. The last check
asserts *prose*, deliberately — without the mixed-batch warning the next
reader sees "509-frame batch" and reasonably treats it as one emulsion,
which is exactly the mistake being fixed.

## What would actually settle these

Ranked by what they unlock per unit of effort, and none of them requires
shooting new film:

1. **One `--empty-gate` frame** (scan of the empty film gate, no film) —
   makes every density absolute instead of relative to scanner white. Free.
2. **One step-wedge scan** (Stouffer T2115 or Kodak Q-13) — characterises
   the scanner's own transfer and noise floor, which is what currently makes
   the σ(D) estimator uninterpretable, and would decide the anisotropy
   question outright.
3. **A ±4 EV grey-card bracket with `--wedge`** — replaces the whole gamma
   bracket (0.677–0.834, both resting on an *assumed* 1.90 logE scene span)
   with one MEASURED gamma, toe and shoulder.
