# FUJI NEOPAN 1600 — manufacturer datasheet AF3-608E, full extraction

**Date:** 2026-08-15
**Source:** Fuji Photo Film Co., Ltd., *FUJIFILM DATA SHEET — NEOPAN 1600 Professional*,
Ref. No. **AF3-608E(N)** (EIGI-99.1-HB8-14), printed in Japan.
`PDF/PROFILES/FUJI/datasheet_neopan1600superpresto_en_01.pdf` — 4 pages, **true digital
PDF**, 90 kB.

Confirms the owner's expectation: digitally generated, not a scan. Pages 1–2 contain zero
embedded images; every graph on pages 3–4 is a **300 dpi raster** (the vector paths on
those pages are the FUJIFILM logo in the footer, not curve data — the same trap recorded
in `DIGITIZATION_QUEUE.md` batch 7).

---

## 1. What the sheet contains, section by section

All four pages were read, not only the curve pages.

| § | Content | Status |
|---|---|---|
| 1 | Features: EI 1600, usable range **EI 400 ↔ 1600**, EI 1600 obtained with the same short development as NEOPAN 400 | extracted |
| 2 | 135 format, 24/36-exp and 30.5 m darkroom-loading; **grey-tinted cellulose triacetate, 0.122 mm** | extracted |
| 3 | **Speed EI 1600/33°** | extracted, confirms stored |
| 4 | **Colour sensitivity: panchromatic** | extracted, confirms stored |
| 5 | Exposure guide tables; flash guide-number formula; **filter factors** — Fuji SC-39/SC-48/SC-56/SC-60 = Wratten 1A/8/21/25, daylight 1.0/2.0/4.0/8.0, tungsten 1.0/1.5/3.0/6.0 | extracted; no schema home for filter factors |
| 6 | Safelight Fuji **SLG4** dark green, 20 W at ≥ 1 m | recorded in provenance |
| 7 | **16-developer × 5-temperature × EI development matrix**, EI 250–3200 at 18–26 °C; stop bath 1.5 % acetic 20–30 s; Fujifix 10 min / Super Fujifix 3–5 min at 15–25 °C; wash 20–30 min; Driwel 1:200 | one condition entered, rest catalogued — see §5 |
| 8 | Automatic processors: Kodak Versamat, FP260 (FC), HPD | recorded |
| 9 | **Spectral sensitivity curve**, "Spectrogram to Daylight (5400 K)" | **re-traced at 5 nm — §3** |
| 10 | **Characteristic curves**, SPD at 20 °C, three times, with printed Ḡ | **refitted — §2** |
| 11 | **Time-Ḡ curves** for Super Prodol, Fujidol E, Microfine, D-76 | recorded, no schema home |

---

## 2. Characteristic curve — refitted, and it was a real correction

**Which curve.** The sheet plots three development times in SPD [Super Prodol] at 20 °C
(68 °F), small tank, each labelled with its average gradient: 2¾ min **Ḡ 0.58**, 4¼ min
**Ḡ 0.77**, 6¼ min **Ḡ 0.90**. The processing table gives SPD at 20 °C for EI 1600 as
**4¼ min**, so the middle curve is this film at its rated speed. That is the one fitted.

**How.** Axes calibrated on the printed gridlines — log H −3.0 at x = 264 with
158.5 px/decade, D = 0 at y = 510 with 156.7 px/density. The three strokes were then
followed column by column **with mutual exclusion**, so two tracks cannot collapse onto
one curve. **487 points** recovered for the 4¼ min curve. All six `ToneCurve` parameters
plus the absolute-to-relative log-H offset were fitted to **all 487 points** by
coarse-to-fine least squares — deliberately not reduced to a few representative samples.

**Validation, because the sheet prints the answer.** The same tracer gave the other two
curves Ḡ **0.548** and **0.916** against their printed **0.58** and **0.90**. An earlier
attempt whose tracks merged produced 0.916 twice; it was discarded rather than trusted.

**Result.**

| | before | after |
|---|---|---|
| base+fog (`dmin`) | 0.170 | **0.211** (traced; all three development times converge on it, as they must) |
| curve | `ToneCurve(0.17, 0.610, −1.44, 0.28, 1.62, 0.40)` | `ToneCurve(0.211, 1.030, −0.880, 0.240, 1.000, 0.480)` |
| Dmax | 2.037 | **2.147** (traced 2.256) |
| average gradient Ḡ | — | **0.769** against the printed **0.77** |

**On `gamma` versus Ḡ — this is what produced the old value.** `ToneCurve.gamma` is the
**straight-line slope**; the traced curve's steepest local slope is 0.900. Fuji's Ḡ is the
**average gradient** over 1.5 log H from 0.1 above base+fog, necessarily lower for a curve
with a real toe: 0.77. The old 0.610 was **neither** — it sat below both. The
parameterisation is degenerate, so the new `gamma` of 1.030 must be read together with
`toe_x` −0.880 and `shoulder_x` 1.000, not on its own.

**Three fits were compared and the trade-off is recorded in the source:**

| fit | RMS residual | Ḡ | outcome |
|---|---|---|---|
| unconstrained | 0.0279 | — | **rejected** — violates the schema's `shoulder_k ≤ 2·toe_k` monotonicity guard |
| schema-constrained | 0.0321 | 0.713 | shape-best, but 0.057 off the published Ḡ |
| Ḡ-anchored, first attempt | 0.0402 | 0.728 | **rejected — see below** |
| Ḡ-anchored, corrected | 0.0510 | **0.7688** | **used** |

**The first Ḡ-anchored attempt was wrong, and its own new regression test caught it.** It
measured the average gradient from the log-H where the *traced* curve crosses
base+fog+0.10 — but Ḡ is a property of the curve being described, so the threshold must be
found on the **model**. On a curve with a long toe those two starting points differ enough
to move Ḡ by 0.04: it read 0.769 by the wrong definition and 0.728 by the right one. The
test I had just written failed, which is how it surfaced.

RMS 0.051 density is roughly 1.5 plotted stroke widths — the curve is drawn 4–5 px thick
and the ordinate is 156.7 px per density unit, so the stroke alone is worth about 0.03.
Reproducing Fuji's own published number to 0.001 was preferred over the last 0.02 of shape
residual, because the printed Ḡ is the manufacturer's assertion about the film while the
residual is a property of my tracing.

---

## 3. Spectral sensitivity — re-traced at 5 nm, the finest in the corpus

Previously stored at 10 nm from this same publication (a 2026-08-02 pass). Re-traced
because **the source supports far finer sampling**: the plot's wavelength axis runs
**0.557 nm per pixel**, calibrated on the printed 400/500/600/700 nm gridlines at
x = 113/292/471/651, linear to better than 1 nm. **436 columns** recovered, resampled to
**5 nm over 390–635 nm, 50 samples** (was 28 at 10 nm).

**The 5 nm step is earned, not cosmetic.** This emulsion has a dip near **613 nm** and a
secondary peak near **630 nm**, 17 nm apart. At 10 nm that pair is marginally sampled; at
5 nm it resolves properly, and it is a real feature of the red sensitisation rather than
trace noise.

**Mutual validation.** The 2026-08-02 digitisation was independent of this one. Over the
25 shared samples the two agree to **max 0.016 and mean 0.008 log** — well inside the
plotted stroke width. Both traces are therefore confirmed; this one simply carries more
detail. The old −4.00 sentinels at 380 and 650 nm are gone, because the range now begins
and ends on real measured samples.

Ordinate: the plot gives **relative** log sensitivity with a 1.0-log reference span between
two dashed rules and no absolute zero, so it is stored peak-normalised as `relative_log`,
the database convention. Nothing is extrapolated beyond the plotted stroke.

Consequence for the live spectral path: the derived monochrome weights become
**0.2746 / 0.3285 / 0.3970**.

---

## 4. Also entered

`ProcessingSpec(developer="SPD [Super Prodol]", minutes=4.25, celsius=20.0,
agitation="continuous 1 min then 5 s each minute", contrast_index=0.77)` — the condition
the stored curve actually represents, taken from the curve's own caption cross-referenced
with the processing table. `contrast_index` is Fuji's printed Ḡ and is **not** equal to
`ToneCurve.gamma`; the profile comment says so explicitly.

Provenance rewritten to name the publication, its sections and its page numbers rather
than the product. `base_tint` deliberately **left neutral**: the grey base is already
inside the traced `dmin` of 0.211, and re-tinting would double-count a density the sheet
never quantifies.

---

## 5. Found but not enterable, and what is still absent

**No schema home** (catalogued here so the reading is not lost): the 16-developer ×
5-temperature × EI development matrix (SPD, SPD 1:1, Fujidol E, Fujidol E 1:1, Microfine,
D-76, D-76 1:1, D-76 1:3, Microdol-X, HC-110 Dil.B, T-MAX, T-MAX RS, Xtol, Microphen,
ID-11, ILFOTEC LC29 1:19); the Time-Ḡ curves for four developers; filter factors for
daylight and tungsten; safelight; the full wet-process chemistry; automatic-processor
conditions. `ProcessingSpec` records one condition, not a family — the processing axis of
§A.5 remains the gap it always was.

**Genuinely not printed on this sheet** — searched all four pages: RMS granularity,
resolving power, MTF, Dmin/Dmax as scalars, reciprocity data of any kind, base thickness
beyond the 0.122 mm total, and any spectral figure in tabulated rather than plotted form.
`NotFound.md` records these against the stock.

---

## 6. Files changed

| File | Change |
|---|---|
| `film_profiles.py` | curve refitted; spectral re-traced to 5 nm; `ProcessingSpec` added; provenance rewritten in place |
| `verify.py` | +2 guards — Ḡ 0.77 and base+fog 0.211 must survive; 5 nm sampling must survive |
| `AlgoSpectralSensitivity.hpp` | the stored-sampling comment now reads 5 nm on one, 10 nm on 48, 20–25 nm on four |
| `film_profiles.hpp/.cpp`, `film_enum.hpp`, `film_names.txt` | regenerated, both copies |
| `doc/FilmActiveProfiles.md`, `doc/FilmCurves.md` | regenerated |
| `doc/NotFound.md` | row settled; the "OCR the Neopan1600 scan" priority item retired as no longer needed |
| `doc/FilmDatabase_Charecteristics.MD` (+ Russian mirror, + both copies) | dated sampling-distribution measurement marked with the new figures |
| `doc/Found.md`, `doc/README.md`, `Readme!.txt`, `DIGITIZATION_QUEUE.md` | logged |

## 7. Verification

- `verify.py`: **121 PASS / 2 FAIL** — the same two pre-existing failures (saturation
  hierarchy ordering, neighbour-pair coupling), unchanged.
- Both new guards pass: Ḡ = 0.769 against the printed 0.77; spectral step 5.0 nm, 50
  samples, source naming AF3-608E.
- `film_profiles.cpp` and `AlgoSpectralSensitivity.cpp` compile clean at `-std=c++14`.
- 142 profiles and 9 print stocks load and pass `validate_all()`; the stock renders finite
  and in range.
- Zero duplicate keys in any decoration dict.
