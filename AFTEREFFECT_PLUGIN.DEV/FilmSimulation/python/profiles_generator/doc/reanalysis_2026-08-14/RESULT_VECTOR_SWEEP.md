# RESULT_VECTOR_SWEEP — exhaustive vector-path inventory of `PDF/PROFILES/**`
### and per-layer spectral-sensitivity extraction for DB stocks that hold no spectral curve

**Date:** 2026-08-16 · **Scope:** NotFound.md §4, item *"vector curve sets on ~30 Kodak/Fuji/Ilford/Agfa sheets"*
**Tooling:** PyMuPDF 1.28.0 (MuPDF 1.29.0), exact PDF content-stream geometry. No rasterisation, no pixel tracing.
**Nothing here has been written to the database.** The main thread enters data.

---

## 0. Method, and why it is not a trace

1. **Sweep.** Every PDF under `PDF/PROFILES/**` (428 readable files, 9 413 pages carrying drawings or
   images) was opened and every page's `get_drawings()` output recorded: drawing count, image count,
   maximum path item count, and the count of paths with ≥30 `l`/`c` items.
2. **Trap rejection.** Two known false-positive families were eliminated *structurally*, not by eye:
   * filled paths (`type == "f"`) and closed paths (`closePath == True`) are discarded — this kills the
     Kodak logo (a filled ~30-item cluster) and every glyph outline (600–1350-item filled blocks);
   * only **open stroked** chains survive (`type in {"s","fs"}`, `closePath == False`).
     Note that Kodak's 2000s still-film sheets draw genuine curve segments as `fs` with a *white* fill
     and `closePath == False`; a naive "fill must be None" filter silently drops those whole plots
     (this is what hid the GOLD 100/200 curves on the first pass).
3. **Chain reassembly.** Consecutive `l`/`c` items whose endpoints coincide within 0.35 pt are joined
   into one polyline; separate paths whose endpoints coincide within 0.6 pt are unioned. Kodak splits
   one curve into 3–70 separate path objects, so this step is mandatory.
4. **Plot identification.** A page's `get_text("words")` is searched for `Spectral … Sensitivit…`; the
   plot frame nearest *below* that title is chosen, and only chains inside that frame are used. This is
   the "verify against the plot title near the path's rect" check the brief asked for.
5. **Calibration.** The printed tick labels are read as words with their own rectangles; axis maps are
   fitted by **least squares** on (label-centre pixel, printed value) pairs, with outlier rejection for
   text-layer glitches (Kodak's `−2.0` loses its minus sign in extraction; a non-monotone tick sequence
   is repaired by negating the tail, and `2.0` extracted as `20` is dropped as an outlier — both cases
   are logged per stock below). Residuals are reported for every plot.
6. **Resampling.** Linear interpolation of the exact vector points onto a regular grid.

### Independent validation of the whole chain
`KODAK_PORTRA_400` **already holds** a spectral curve in the DB (E-4050, entered earlier by a different
route). Re-extracting it blind from `e4050_portra_400.pdf` p4 reproduces the stored array to:

| layer | overlapping samples | mean Δ | max abs Δ |
|---|---|---|---|
| `log_s_b` | 14 | −0.042 | 0.15 |
| `log_s_g` | 21 | −0.020 | 0.18 |
| `log_s_r` | 19 | −0.027 | 0.23 |

The mean offsets are ≈ 0.03 log and the max deviations occur only on the steep tails where a 1-sample
lambda shift costs 0.2 log. **The pipeline is validated.**

---

## 1. INVENTORY

### 1.1 Funnel

| stage | pages |
|---|---|
| pages with any drawing or image | 9 413 |
| pages with ≥1 path of ≥30 `l`/`c` items (the raw "vector curve" signal) | **937** |
| …surviving the fill/closePath trap filter and the ≥25-point / ≥60 pt-wide / ≥15 pt-tall chain test | **516** |
| …rejected as logo/glyph/filled art | **421 (45 % of the raw signal)** |

### 1.2 What the 421 rejects were

| max path item count on the page | rejected pages | what it is |
|---|---|---|
| 30–44 | 28 | the **Kodak logo** — a filled ~30-item path cluster, exactly the documented trap |
| 45–119 | 111 | filled logotype/wordmark art, filled legend swatches |
| 120–499 | 266 | filled glyph outlines of headline text (sheets whose titles are converted to curves) |
| ≥500 | 16 | full **glyph-outline text blocks**, 600–1350 items, the second documented trap |

Rejects by folder: KODAK 249, POLAROID 72, FUJI 42, AGFA 18, KONICA 17, FOMACOLOR 10, ILFORD 9,
FERRANIA 3, ROLLEI 1.

### 1.3 Genuine vector-curve pages, by manufacturer

| folder | vector-curve pages | distinct PDFs | of those, pages carrying a **spectral-sensitivity** plot |
|---|---|---|---|
| KODAK | 349 | 148 | 58 |
| POLAROID | 80 | 47 | 49 |
| FUJI | 49 | 27 | 22 |
| AGFA | 23 | 11 | 17 |
| ILFORD | 10 | 7 | 6 |
| KONICA | 2 | 2 | 0 |
| ORWO | 2 | 2 | 0 |
| root (`aimm.it2.18.1996.pdf`) | 1 | 1 | 0 |
| **total** | **516** | **245** | **152 pages in 111 PDFs** |

### 1.4 Curve type mix on those 516 pages (a page often carries several plots)

| curve type (by plot-title text on the page) | pages |
|---|---|
| characteristic (H&D) curves | 238 |
| **spectral sensitivity** | **152** |
| granularity / RMS vs density | 125 |
| MTF / modulation transfer | 119 |
| spectral **dye** density | 54 |
| no recognised title text | 130 |

**Verdict on NotFound.md §4.** The "~30 vector curve sets" figure was a large undercount. The corpus
holds **516 genuine vector curve pages across 245 PDFs**, of which **152 pages in 111 PDFs carry
per-layer or panchromatic spectral-sensitivity plots** whose coordinates are exact. There is no
"~30 that might be logos" problem: the logo/glyph false positives are a *separate* 421 pages and are
fully accounted for above.

---

## 2. CROSS-REFERENCE AGAINST THE DATABASE

`film_profiles.py`: **143 stocks**, of which **35 hold a spectral curve** and **108 do not**
(`spectral is None` or `log_s_r == ()`). All stored curves use `lambda_start_nm=380.0`,
`lambda_step_nm=10.0`, 30–38 samples, criterion `relative_log`, floor `-4.0`.

### 2.1 Ranked candidate list

Rank = (colour, 3 separate layers) > (B&W, single layer); and (stock has no spectral at all) >
(revalidation). Every row was matched to the **same publication the DB already cites** for that stock,
so there is no product-identity risk.

| # | DB stock | spectral now? | source PDF / page | plot | status |
|---|---|---|---|---|---|
| 1 | `KODAK_ULTRAMAX_800` | none | `KODAK/E7024-Ultra_Max_800.pdf` p3 (E-7024) | 3-layer vector | **EXTRACTED** |
| 2 | `KODAK_ULTRAMAX_400` | none | `KODAK/E7023_max_400.pdf` p4 (E-7023) | 3-layer vector | **EXTRACTED** |
| 3 | `KODAK_EKTAR_100` | none | `KODAK/e4046_ektar_100.pdf` p4 (E-4046) | 3-layer vector | **EXTRACTED** |
| 4 | `KODAK_PORTRA_160` | none | `KODAK/e4051_Portra_160.pdf` p4 (E-4051) | 3-layer vector | **EXTRACTED** |
| 5 | `KODAK_PORTRA_800` | none | `KODAK/e4040_portra_800-2016.pdf` p4 (E-4040) | 3-layer vector | **EXTRACTED** |
| 6 | `KODAK_PORTRA_100T` | none | `KODAK/e2468-Portra_100T.pdf` p5 (E-2468) | 3-layer vector | **EXTRACTED** |
| 7 | `KODAK_GOLD_100` + `KODAK_GOLD_200` | none (both) | `KODAK/E7022-Gold_100_200.pdf` p4 (E-7022) | 3-layer vector | **EXTRACTED** (one plot serves both stocks — the sheet plots a single curve set for the family) |
| 8 | `KODAK_TRI_X_400TX` | none | `KODAK/f4017_TriX.pdf` p7 (F-4017) | 1 layer × 2 criteria | **EXTRACTED** |
| 9 | `KODAK_TMAX_100` | none | `KODAK/f4016_tmax_100-2018.pdf` p8 (F-4016) | 1 layer × 2 criteria | **EXTRACTED** |
| 10 | `KODAK_TMAX_P3200` | none | `KODAK/f4001-P3200TMZ-2019.pdf` p7 (F-4001) | 1 layer × 2 criteria | **EXTRACTED** |
| 11 | `KODAK_PLUS_X_125` | none | `KODAK/f4018-125PX-2007.pdf` p9 (F-4018) | 1 vector curve | **EXTRACTED** (criterion ambiguous — see caveat) |
| 12 | `KODAK_BW400CN` | none | `KODAK/f4036-BW400CN.pdf` p5 (F-4036) | 1 layer | **EXTRACTED** |
| 13 | `KODAK_T400CN` | none | `KODAK/f2350-T400CN.pdf` p6 (F-2350) | 1 layer | **EXTRACTED** |
| — | `KODAK_PORTRA_400` | **has one** | `KODAK/e4050_portra_400.pdf` p4 (E-4050) | 3-layer vector | **EXTRACTED as revalidation** (see §0) |
| 14 | `KODAK_ULTRA_COLOR_100UC` / `400UC` | none | E-4035 — **not on disk**; `e4026`/`e4029` are ROYAL SUPRA / SUPRA, different films | — | not extractable from held corpus |
| 15 | `AGFA_APX_25` / `_100` / `_400` | none | `AGFA/agfapanapx25.pdf`, `apx100.pdf`, `apx400.pdf` p2 | 1 vector bezier curve each (13 items), frame (65.5, 93.3, 274.9, 262.9), x ticks 400/500/600/700, y ticks 0/1.0/2.0 | **identified, not extracted** — frames are drawn as `qu` quads and needed a hand frame; geometry confirmed present and clean |
| 16 | `AGFA_VISTA_200` | none | `AGFA/AGFACOLOR Vista 100, 200, 400, 800.pdf` p8 | 8–10 short vector curves, several products on one page | **identified, not extracted** — the page carries the whole 100/200/400/800 family; per-product assignment needs the legend read, not automatable safely |
| 17 | `ILFORD_HP5_PLUS_400`, `ILFORD_DELTA_3200` | none | `ILFORD/HP5+-200407.pdf` p1, `Delta_3200-200209.pdf` p1 | 1 stroked 12-item bezier inside a frame under the heading **"SPECTRAL SENSITIVITY — Wedge spectrogram to tungsten light"** | **identified, not extracted** — the figure is a *wedge spectrogram outline*, and the page carries **no numeric axis tick labels at all**. There is nothing to calibrate against. The plot does not support a sampling step. |
| 18 | `ILFORD_FP4`, `ILFORD_PAN_F` | none | `ILFORD/FP4+-200404.pdf`, `PanF+-200407.pdf` p1 | same | same, **plus** a product-identity caveat: the DB stocks are FP4 / Pan F, the sheets are FP4 **Plus** / Pan F **Plus** |
| 19 | `POLAROID_664`, `POLAROID_667`, `POLAROID_52`, `POLAROID_55_PN_NEG` | none | `POLAROID/664fds.pdf` p3, `667fds.pdf` p3, `52fds.pdf`, `55fds.pdf` | 1 vector curve each; **log-decade** y axis (ticks 0.1 / 1 / 10 / 100 / 1000), x 350–700 nm | **identified, not extracted** — different axis convention (decade log, not log-S), needs a convention decision before entry |
| 20 | `KODAK_TECHNICAL_PAN` | none | `KODAK/KODAK PROFESSIONAL Technical Pan Film.pdf` pp 3, 9, 10, 11 | vector curves present on 4 pages | **identified, not extracted** — multi-plot pages, needs per-plot disambiguation |
| 21 | `KODAK_TMAX_400` | none | `KODAK/f4043_TMax_400-2016.pdf` p7 (F-4043) | 2 vector chains under the D=0.3 / D=1.0 legend | **identified, NOT adopted** — the two criterion curves have inconsistent shapes (peaks at 528 nm and 570 nm, dynamic ranges 0.57 and 1.5 log). Either the chains cross and my union merged them wrongly, or one chain is not a sensitivity curve. **Do not enter without visual confirmation.** |

### 2.2 Vector spectral plots found for stocks that ALREADY hold a curve (revalidation only)
VISION2 5201 / 5205 / 5212 / 5217 / 5218 / 5229 / 5279, EKTACHROME 64T 7280, ELITE Chrome 200 (E-148E),
2383 / 2393 print stocks, ROYAL GOLD 200/400, High Definition 200/400, SUPRA and ROYAL SUPRA families
(E-4026 / E-4029 / E-7006 / E-7013 / E-7017 / E-2509 / E-2519). All are genuine vector 3-layer plots
and can be re-derived at any time by the same script; none is a database gap.

---

## 3. LAYER MAPPING (explicit, as requested)

The Kodak still-film sheets label the three curves **Yellow-Forming Layer / Magenta-Forming Layer /
Cyan-Forming Layer**, and these text labels sit inside the plot frame next to their curve. The mapping
used throughout, verified against those labels *and* against peak-wavelength ordering on every plot:

| DB field | dye-forming layer named on the sheet | spectral sensitisation | typical peak |
|---|---|---|---|
| `log_s_b` | **Yellow**-forming | blue-sensitive (unsensitised silver halide) | 400–470 nm |
| `log_s_g` | **Magenta**-forming | green-sensitive (orthochromatic sensitiser) | 540–552 nm |
| `log_s_r` | **Cyan**-forming | red-sensitive (panchromatic sensitiser) | 617–657 nm |
| `log_s_pan` | — | panchromatic single layer (B&W) | — |

---

## 4. SAMPLING STEP — justification

**Step chosen: 10 nm, grid start 380 nm, 33 samples (380–700 nm).** Reasons:

* it is the project convention — all 35 stored `SpectralSensitivity` records use
  `lambda_start_nm=380.0, lambda_step_nm=10.0`;
* the plots support it comfortably. Kodak prints the 250–750 nm axis across ~200 pt, i.e. 0.401 pt/nm,
  and the vector polylines carry a point every **0.65–1.3 nm** on the dense sheets and every **3.3 nm**
  on the sparsest (E-4050/E-4051/E-4046). A 10 nm grid is 3–15 source points per output sample, so the
  step is *coarser* than the source everywhere — no invention, only decimation;
* it is **not** supported below ~5 nm on the sparse sheets (Portra 160/400, Ektar 100, 3.33 nm point
  spacing): a 1 nm or 2 nm grid there would be re-drawing the illustrator's spline, not the data.
  **Do not resample these below 5 nm.**

Outside each layer's plotted span the array is padded with the project floor **−4.0**. That padding is
**not measured data** — it means "the sheet does not plot the curve there". The measured span is stated
per layer in every block below.

## 5. NORMALISATION

Every Kodak sheet in this batch carries an **absolute** ordinate:
*"Sensitivity = reciprocal of exposure (erg/cm²) required to produce specified density"*, with the
density criterion printed on the plot (colour sheets: 0.2 above D-min, Status M densitometry;
B&W sheets: D = 0.3 and D = 1.0 above D-min / above gross fog, diffuse visual densitometry).

Per project convention the arrays below are **peak-normalised per layer** and should be stored as
`criterion='relative_log'`. The absolute scale is preserved losslessly: for every sample where the
relative value is > −4.0,

```
absolute_logS[i] = relative[i] + peak_abs_logS
```

and `peak_abs_logS` is printed in every block. If the main thread prefers the absolute convention
(as `KODAK_PORTRA_400` uses, `criterion='log_reciprocal_erg_cm2_D0.2_above_dmin'`), add the constant back.

---

## 6. EXTRACTIONS

All arrays: `lambda_start_nm = 380.0`, `lambda_step_nm = 10.0`, 33 samples (380 … 700 nm).
"Interpolated gaps" lists wavelength intervals where the printed curve is broken (dashed rendering, or
the curve passing behind another element); values inside those intervals are linearly interpolated
between the two exact vector endpoints and are flagged rather than hidden.

---

### 6.1 `KODAK_ULTRAMAX_800`

* Source: `PDF/PROFILES/KODAK/E7024-Ultra_Max_800.pdf` page 3 (E-7024, Dec 2007); frame (362.2, 388.9, 562.7, 540.4) pt
* x calibration, 11 printed wavelength ticks 250…750: `nm = 2.490931·px − 654.0003` → **rms 0.271 nm, max residual 0.738 nm**
* y calibration, ticks [0.0, 1.0, 2.0, 3.0, 4.0]: `logS = −0.026795·py + 14.4663` → **rms 0.0049, max residual 0.0088 log**
* Plot conditions: daylight, effective exposure 1/200 s, process C-41, Status M, density 0.2 > D-min

| layer | vector pts | plotted span | point spacing | peak abs logS | peak λ | gaps |
|---|---|---|---|---|---|---|
| `log_s_b` | 176 | 368.2–523.0 nm | 0.65 nm | 2.733 | 408.1 nm | none |
| `log_s_g` | 254 | 388.1–602.9 nm | 0.65 nm | 2.851 | 544.9 nm | none |
| `log_s_r` | 258 | 488.0–702.9 nm | 0.65 nm | 2.970 | 657.0 nm | none |

```python
log_s_b = (-0.99, -0.49, -0.06, -0.00, -0.06, -0.08, -0.14, -0.17, -0.13, -0.04, -0.38, -0.89, -1.48, -2.09, -2.56, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_g = (-4.00, -1.79, -1.46, -1.49, -1.55, -1.59, -1.67, -1.64, -1.57, -1.57, -1.04, -0.68, -0.49, -0.42, -0.32, -0.18, -0.04, -0.01, -0.08, -0.23, -0.51, -1.39, -2.41, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_r = (-4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -2.68, -2.56, -2.36, -2.20, -2.18, -2.16, -2.02, -1.78, -1.59, -0.99, -0.62, -0.47, -0.40, -0.32, -0.23, -0.10, -0.05, -0.13, -0.61, -1.01, -1.51, -2.52)
# peak_abs_logS: b=2.733  g=2.851  r=2.970   (measured spans: b 380-520, g 390-600, r 490-700 nm)
```

### 6.2 `KODAK_ULTRAMAX_400`

* Source: `PDF/PROFILES/KODAK/E7023_max_400.pdf` page 4 (E-7023); frame (76.6, 371.0, 277.6, 522.6) pt
* x: `nm = 2.490925·px + 57.3674` → **rms 0.271 nm, max 0.737 nm**
* y: ticks [0,1,2,3,4] → `logS = −0.026795·py + 13.9906` → **rms 0.0049, max 0.0088**
* Conditions: daylight, 1/100 s, C-41, Status M, 0.2 > D-min

| layer | vector pts | plotted span | point spacing | peak abs logS | peak λ | gaps |
|---|---|---|---|---|---|---|
| `log_s_b` | 151 | 363.4–523.6 nm | 1.15 nm | 2.726 | 469.0 nm | none |
| `log_s_g` | 181 | 388.4–598.6 nm | 1.18 nm | 2.719 | 547.3 nm | none |
| `log_s_r` | 193 | 483.5–698.7 nm | 1.18 nm | 2.742 | 651.2 nm | none |

```python
log_s_b = (-0.92, -0.51, -0.10, -0.02, -0.06, -0.06, -0.05, -0.13, -0.11, -0.00, -0.38, -1.01, -1.67, -2.14, -2.58, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_g = (-4.00, -1.86, -1.56, -1.61, -1.68, -1.76, -1.84, -1.82, -1.77, -1.73, -1.30, -0.79, -0.58, -0.48, -0.35, -0.20, -0.06, -0.00, -0.08, -0.26, -0.56, -1.54, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_r = (-4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -2.44, -2.30, -2.13, -1.98, -1.93, -1.90, -1.78, -1.56, -1.42, -0.98, -0.69, -0.58, -0.48, -0.37, -0.27, -0.10, -0.00, -0.11, -0.57, -1.01, -1.43, -4.00)
# peak_abs_logS: b=2.726  g=2.719  r=2.742   (measured spans: b 380-520, g 390-590, r 490-690 nm)
```

### 6.3 `KODAK_EKTAR_100`

* Source: `PDF/PROFILES/KODAK/e4046_ektar_100.pdf` page 4 (E-4046, 2016); frame (74.2, 356.6, 274.7, 508.0) pt
* x: `nm = 2.490932·px + 63.4167` → **rms 0.271 nm, max 0.738 nm**
* y: ticks [0,1,2,3] → `logS = −0.019960·py + 10.1258` → **rms 0.0038, max 0.0045**
* Conditions: daylight, 1/25 s, Status M, 0.2 > D-min

| layer | vector pts | plotted span | point spacing | peak abs logS | peak λ | interpolated gaps (nm) |
|---|---|---|---|---|---|---|
| `log_s_b` | 58 | 378.1–492.9 | 1.21 nm | 2.190 | 466.9 | 405–413, 443–449, 473–483, 484–490 |
| `log_s_g` | 119 | 393.1–587.8 | 1.21 nm | 1.838 | 543.4 | 440–447, 482–492 |
| `log_s_r` | 97 | 552.8–692.6 | 1.11 nm | 2.059 | 649.7 | 653–661, 680–686 |

```python
log_s_b = (-1.03, -0.54, -0.20, -0.14, -0.17, -0.14, -0.11, -0.19, -0.11, -0.05, -0.46, -1.59, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_g = (-4.00, -4.00, -0.88, -0.91, -1.00, -1.09, -1.24, -1.24, -1.24, -1.23, -0.95, -0.59, -0.45, -0.37, -0.26, -0.12, -0.03, -0.05, -0.07, -0.18, -0.53, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_r = (-4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -1.73, -1.61, -0.82, -0.50, -0.35, -0.30, -0.27, -0.20, -0.09, -0.00, -0.12, -0.75, -1.07, -1.56, -4.00)
# peak_abs_logS: b=2.190  g=1.838  r=2.059   (measured spans: b 380-490, g 400-580, r 560-690 nm)
```

### 6.4 `KODAK_PORTRA_160`

* Source: `PDF/PROFILES/KODAK/e4051_Portra_160.pdf` page 4 (E-4051, 2016); frame (73.8, 348.0, 274.2, 499.6) pt
* x: `nm = 2.490908·px + 64.8100` → **rms 0.270 nm, max 0.735 nm**
* y: printed ticks extract as [−1.0, 0.0, 1.0, **20.0**, 3.0]; the `20.0` is the sheet's `2.0` with the
  decimal point lost by the text layer and was **rejected as an outlier** by the robust fit. The four
  surviving ticks give `logS = −0.026762·py + 12.3540` → **rms 0.0014, max residual 0.0024 log** — the
  cleanest y calibration in the batch, which confirms the rejection was correct.
* Conditions: daylight, 1/50 s, Status M, 0.2 > D-min

| layer | vector pts | plotted span | point spacing | peak abs logS | peak λ | gaps |
|---|---|---|---|---|---|---|
| `log_s_b` | 40 | 378.4–508.2 nm | 3.33 nm | 2.096 | 464.9 nm | none |
| `log_s_g` | 61 | 388.4–588.1 nm | 3.33 nm | 1.876 | 551.5 nm | none |
| `log_s_r` | 55 | 488.2–668.0 nm | 3.33 nm | 2.030 | 624.7 nm | none |

```python
log_s_b = (-0.99, -0.57, -0.16, -0.17, -0.16, -0.19, -0.13, -0.19, -0.10, -0.06, -0.48, -1.35, -1.79, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_g = (-4.00, -1.13, -0.79, -0.86, -0.90, -1.01, -1.02, -1.08, -1.05, -1.00, -0.80, -0.63, -0.48, -0.41, -0.32, -0.22, -0.09, -0.00, -0.11, -0.25, -0.63, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_r = (-4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -1.98, -1.93, -1.82, -1.73, -1.71, -1.72, -1.59, -1.35, -1.11, -0.67, -0.40, -0.23, -0.11, -0.05, -0.04, -0.27, -0.90, -1.12, -4.00, -4.00, -4.00, -4.00)
# peak_abs_logS: b=2.096  g=1.876  r=2.030   (measured spans: b 380-500, g 390-580, r 490-660 nm)
```

### 6.5 `KODAK_PORTRA_800`

* Source: `PDF/PROFILES/KODAK/e4040_portra_800-2016.pdf` page 4 (E-4040, 2016); frame (350.2, 313.7, 550.9, 465.4) pt
* x: `nm = 2.490931·px − 624.1078` → **rms 0.271 nm, max 0.738 nm**
* y: ticks [0,1,2,3,4] → `logS = −0.026796·py + 12.4499` → **rms 0.0049, max 0.0088**
* Conditions: daylight, 1/200 s, Status M, 0.2 > D-min

| layer | vector pts | plotted span | point spacing | peak abs logS | peak λ | gaps |
|---|---|---|---|---|---|---|
| `log_s_b` | 150 | 368.2–523.2 nm | 1.18 nm | 2.730 | 408.2 nm | none |
| `log_s_g` | 197 | 388.2–603.2 nm | 1.18 nm | 2.849 | 545.1 nm | none |
| `log_s_r` | 204 | 488.2–703.3 nm | 1.18 nm | 2.967 | 657.4 nm | none |

```python
log_s_b = (-0.99, -0.50, -0.06, -0.00, -0.06, -0.08, -0.14, -0.17, -0.13, -0.04, -0.36, -0.88, -1.46, -2.09, -2.56, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_g = (-4.00, -1.81, -1.47, -1.49, -1.55, -1.59, -1.67, -1.64, -1.57, -1.58, -1.08, -0.69, -0.50, -0.42, -0.32, -0.18, -0.05, -0.01, -0.07, -0.23, -0.49, -1.33, -2.40, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_r = (-4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -2.69, -2.57, -2.37, -2.21, -2.18, -2.16, -2.03, -1.79, -1.62, -1.02, -0.62, -0.48, -0.40, -0.33, -0.24, -0.11, -0.05, -0.08, -0.59, -0.99, -1.46, -2.48)
# peak_abs_logS: b=2.730  g=2.849  r=2.967   (measured spans: b 380-520, g 390-600, r 490-700 nm)
```

> ⚠ **Flag for the main thread.** `KODAK_PORTRA_800` (E-4040) and `KODAK_ULTRAMAX_800` (E-7024) come out
> *near-identical* — peak wavelengths 408.2/408.1, 545.1/544.9, 657.4/657.0 nm, arrays agreeing to
> ≤0.02 log. The two extractions are from different files with different pixel coordinates, so this is
> not a bug in the sweep: **Kodak has reused the same spectral artwork for the two 800-speed emulsions.**
> Enter both if you accept the sheets at face value, but the provenance note should say so.

### 6.6 `KODAK_PORTRA_100T`

* Source: `PDF/PROFILES/KODAK/e2468-Portra_100T.pdf` page 5 (E-2468); frame (362.7, 77.7, 563.2, 229.2) pt
* x: `nm = 2.490981·px − 655.1104` → **rms 0.271 nm, max 0.739 nm**
* y: ticks [0,1,2,3,4] → `logS = −0.026795·py + 6.1288` → **rms 0.0049, max 0.0088**

| layer | vector pts | plotted span | point spacing | peak abs logS | peak λ | gaps |
|---|---|---|---|---|---|---|
| `log_s_b` | 94 | 368.4–508.6 nm | 1.29 nm | 2.404 | 467.5 nm | none |
| `log_s_g` | 104 | 438.5–588.7 nm | 1.26 nm | 2.197 | 542.4 nm | none |
| `log_s_r` | 104 | 538.6–688.8 nm | 1.26 nm | 2.171 | 617.5 nm | none |

```python
log_s_b = (-1.72, -1.38, -0.77, -0.31, -0.17, -0.09, -0.03, -0.12, -0.08, -0.01, -0.56, -1.20, -1.67, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_g = (-4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -1.73, -1.68, -1.61, -1.48, -1.13, -0.72, -0.47, -0.34, -0.23, -0.08, -0.00, -0.01, -0.05, -0.21, -0.76, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_r = (-4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -1.76, -1.59, -1.40, -1.14, -0.65, -0.29, -0.12, -0.03, -0.00, -0.12, -0.39, -0.77, -1.07, -1.09, -1.00, -4.00, -4.00)
# peak_abs_logS: b=2.404  g=2.197  r=2.171   (measured spans: b 380-500, g 440-580, r 540-680 nm)
```

Note the tungsten balance is visible in the data: the red peak sits at 617.5 nm, ~35 nm shorter than the
daylight stocks, and the red layer is the *least* sensitive of the three — as expected for a 100T film.

### 6.7 `KODAK_GOLD_100` and `KODAK_GOLD_200`

* Source: `PDF/PROFILES/KODAK/E7022-Gold_100_200.pdf` page 4 (E-7022); frame (350.9, 54.4, 551.4, 205.9) pt.
  The identical plot is also in `KODAK/E7022_Gold_200.pdf` p4 (fewer vector points, same coordinates).
* x: `nm = 2.490915·px − 625.7393` → **rms 0.271 nm, max 0.737 nm**
* y: ticks [0,1,2,3,4] → `logS = −0.026796·py + 5.4992` → **rms 0.0049, max 0.0088**
* **This sheet draws its curves as `fs` paths with a white fill** — the family that a naive fill filter
  drops. 65 separate path objects were re-chained into the three layers.

| layer | vector pts | plotted span | point spacing | peak abs logS | peak λ | interpolated gaps |
|---|---|---|---|---|---|---|
| `log_s_b` | 146 | 368.1–508.0 nm | 0.65 nm | 2.375 | 467.4 nm | none |
| `log_s_g` | 170 | 378.2–597.9 nm | 1.28 nm | 2.252 | 546.6 nm | **418.1–428.1 nm** (curve hidden behind the blue curve) |
| `log_s_r` | 213 | 388.1–687.8 nm | 1.28 nm | 2.230 | 651.4 nm | **468.0–483.3 nm** (curve leaves the bottom of the plot — the true value there is *below* the printed 0.0 floor, so the 470/480 nm samples are the axis floor, **not** measurement) |

```python
log_s_b = (-1.12, -0.73, -0.33, -0.24, -0.28, -0.26, -0.20, -0.24, -0.15, -0.03, -0.47, -1.55, -2.09, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_g = (-1.66, -1.30, -1.00, -1.05, -1.14, -1.24, -1.33, -1.34, -1.33, -1.29, -1.04, -0.73, -0.56, -0.47, -0.35, -0.18, -0.03, -0.00, -0.11, -0.38, -0.85, -1.69, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_r = (-4.00, -1.85, -1.61, -1.72, -1.87, -2.02, -2.16, -2.23, -2.25, -2.25, -2.25, -2.14, -2.03, -1.93, -1.83, -1.77, -1.69, -1.59, -1.44, -1.30, -1.01, -0.81, -0.68, -0.57, -0.43, -0.31, -0.13, -0.00, -0.13, -0.68, -1.14, -4.00, -4.00)
# peak_abs_logS: b=2.375  g=2.252  r=2.230
# measured spans: b 380-500, g 380-590 (gap at 420), r 390-690 (index 9,10 = 470,480 nm sit on the axis floor)
```

> The E-7022 sheet covers GOLD 100 **and** GOLD 200 and prints **one** spectral-sensitivity figure for the
> family. Applying it to both DB stocks is what the document supports; it is not evidence that the two
> emulsions are spectrally identical, and the provenance note should say "family figure, E-7022".

### 6.8 `KODAK_PORTRA_400` — revalidation only (stock already holds a curve)

* Source: `PDF/PROFILES/KODAK/e4050_portra_400.pdf` page 4 (E-4050); frame (76.2, 343.1, 276.7, 494.6) pt
* x: `nm = 2.490950·px + 58.4517` → **rms 0.271 nm, max 0.737 nm**; y: `logS = −0.026796·py + 13.2385` → **rms 0.0049, max 0.0088**
* Agreement with the stored array: mean −0.03 log, max 0.23 log (see §0). **No change recommended.**

```python
log_s_b = (-0.82, -0.36, -0.02, -0.02, -0.07, -0.06, -0.07, -0.11, -0.08, -0.06, -0.40, -1.07, -1.53, -1.96, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_g = (-4.00, -1.26, -0.99, -1.03, -1.11, -1.18, -1.23, -1.24, -1.21, -1.14, -0.88, -0.63, -0.49, -0.41, -0.32, -0.21, -0.07, -0.00, -0.09, -0.22, -0.44, -1.15, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
log_s_r = (-4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00, -2.24, -2.09, -1.95, -1.85, -1.82, -1.86, -1.82, -1.57, -1.27, -0.86, -0.56, -0.44, -0.31, -0.23, -0.22, -0.18, -0.00, -0.21, -1.10, -4.00, -4.00, -4.00)
# peak_abs_logS: b=2.642  g=2.562  r=2.642
```

---

### 6.9 `KODAK_TRI_X_400TX` (panchromatic, two density criteria)

* Source: `PDF/PROFILES/KODAK/f4017_TriX.pdf` page 7 (F-4017, Feb 2016); frame (86.9, 291.7, 287.4, 443.0) pt
* x: `nm = 2.492157·px + 31.5949` → **rms 0.271 nm, max 0.738 nm**
* y: ticks [0,1,2,3,4] → `logS = −0.026825·py + 11.8702` → **rms 0.0049, max 0.0089**
* Conditions: 0.5 s effective exposure, diffuse visual densitometry; criteria printed on the plot are
  **D = 0.3 > gross fog** and **D = 1.0 > gross fog**. The higher-lying curve is the 0.3 criterion
  (a lower density criterion needs less exposure, hence more sensitivity); the legend text confirms the
  ordering, `D=0.3` label above `D=1.0`.

| curve | vector pts | plotted span | point spacing | peak abs logS | peak λ | gaps |
|---|---|---|---|---|---|---|
| D = 0.3 > gross fog | 62 | 298.3–666.4 nm | 3.66 nm | 2.744 | 377.1 nm | 23 dash gaps, max 16.7 nm |
| D = 1.0 > gross fog | 129 | 298.3–654.6 nm | 2.55 nm | 1.597 | 379.4 nm | 2 gaps, max 7.5 nm |

```python
# recommended for the DB: the D = 0.3 criterion curve
log_s_pan = (-0.00, -0.01, -0.02, -0.03, -0.04, -0.07, -0.11, -0.15, -0.19, -0.29, -0.37, -0.45, -0.51, -0.50, -0.44, -0.40, -0.35, -0.31, -0.28, -0.27, -0.27, -0.31, -0.37, -0.32, -0.31, -0.42, -0.84, -1.81, -2.46, -4.00, -4.00, -4.00, -4.00)
# peak_abs_logS = 2.744  (measured span 380-660 nm; the curve continues below 380 to 298 nm)
# the D = 1.0 curve, for reference / cross-check:
log_s_pan_D10 = (0.00, -0.01, -0.02, -0.03, -0.04, -0.07, -0.09, -0.11, -0.15, -0.24, -0.33, -0.39, -0.41, -0.38, -0.32, -0.27, -0.23, -0.20, -0.18, -0.17, -0.19, -0.21, -0.21, -0.19, -0.20, -0.31, -0.61, -1.21, -4.00, -4.00, -4.00, -4.00, -4.00)
# peak_abs_logS = 1.597
```

> ⚠ F-4017 covers **TRI-X 320 (320TXP)** *and* **TRI-X 400 (400TX)** and the figure legend names only the
> two density criteria, not the two films. I read the two curves as one film's two criteria (that is what
> the legend says), but the sheet does not state *which* film the figure belongs to. Confirm visually
> before entering, or store it against the family rather than `400TX` alone.

### 6.10 `KODAK_TMAX_100` (panchromatic, two density criteria)

* Source: `PDF/PROFILES/KODAK/f4016_tmax_100-2018.pdf` page 8 (F-4016, 2018); frame (74.6, 498.5, 275.1, 650.0) pt.
  Identical geometry in `f4016_TMax_100.pdf` and `f4016_TMax_100-2016.pdf` p8.
* x: `nm = 2.490920·px + 62.4511` → **rms 0.269 nm, max 0.733 nm**
* y: the printed ticks extract as [2.0, 1.0, 0.0, 1.0, 2.0] — **the sheet's minus signs do not survive
  text extraction**. The sequence is non-monotone in pixel order, so the tail was negated to
  [+2, +1, 0, −1, −2]; `logS = −0.026421·py + 15.1713` → **rms 0.0082, max residual 0.0123 log**. The fact
  that the repaired sequence fits to 0.012 log is itself the proof the repair is right.
* Conditions: D-76, 20 °C, diffuse visual densitometry; criteria D = 0.3 and D = 1.0 above D-min.
* The curves are **dashed**; 19–20 dash gaps of up to 16 nm are bridged by linear interpolation.

| curve | vector pts | plotted span | point spacing | peak abs logS | peak λ |
|---|---|---|---|---|---|
| D = 0.3 > D-min | 66 | 398.5–698.8 nm | 2.42 nm | 1.500 | 398.5 nm |
| D = 1.0 > D-min | 62 | 398.5–668.8 nm | 2.73 nm | 0.500 | 398.5 nm |

```python
# recommended for the DB: the D = 0.3 criterion curve
log_s_pan = (-4.00, -4.00, -0.01, -0.06, -0.13, -0.19, -0.24, -0.30, -0.35, -0.41, -0.48, -0.53, -0.53, -0.49, -0.44, -0.40, -0.40, -0.40, -0.40, -0.32, -0.35, -0.39, -0.45, -0.51, -0.53, -0.51, -0.61, -1.14, -1.82, -2.16, -2.55, -2.96, -4.00)
# peak_abs_logS = 1.500  (measured span 400-690 nm)
log_s_pan_D10 = (-4.00, -4.00, -0.00, -0.02, -0.06, -0.11, -0.15, -0.19, -0.26, -0.33, -0.41, -0.49, -0.49, -0.48, -0.43, -0.41, -0.40, -0.41, -0.43, -0.46, -0.47, -0.46, -0.48, -0.62, -0.78, -0.85, -0.92, -1.10, -1.66, -4.00, -4.00, -4.00, -4.00)
# peak_abs_logS = 0.500
```

The two criteria differ by exactly 1.00 log at the peak, which is the expected separation for the sheet's
stated D-76 contrast — a good internal consistency check on the y calibration.

### 6.11 `KODAK_TMAX_P3200` (panchromatic, two density criteria)

* Source: `PDF/PROFILES/KODAK/f4001-P3200TMZ-2019.pdf` page 7 (F-4001, Apr 2019); frame (363.5, 315.6, 564.1, 467.1) pt.
  Identical plot in `F4001-P3200TMZ-2018.pdf` p7.
* x: `nm = 2.490909·px − 656.1799` → **rms 0.272 nm, max 0.740 nm**
* y: same missing-minus repair as §6.10 → ticks [+3, +2, +1, 0, −1]; `logS = −0.026539·py + 11.3860`
  → **rms 0.0043, max residual 0.0073 log**
* Dashed curves: 14–19 gaps, largest 29.6 nm on the D = 1.0 curve. **The 29.6 nm gap spans three grid
  points (470/480/490 nm) on that curve** — treat those three as interpolation, not measurement.

| curve | vector pts | plotted span | point spacing | peak abs logS | peak λ |
|---|---|---|---|---|---|
| D = 0.3 > D-min | 48 | 419.6–699.9 nm | 5.42 nm | 2.205 | 419.6 nm |
| D = 1.0 > D-min | 53 | 419.6–669.9 nm | 3.26 nm | 1.231 | 469.3 nm |

```python
# recommended for the DB: the D = 0.3 criterion curve
log_s_pan = (-4.00, -4.00, -4.00, -4.00, -0.00, -0.02, -0.05, -0.09, -0.14, -0.22, -0.28, -0.29, -0.30, -0.29, -0.25, -0.21, -0.19, -0.21, -0.22, -0.18, -0.14, -0.18, -0.25, -0.32, -0.35, -0.36, -0.37, -0.71, -1.39, -1.80, -2.21, -2.68, -4.00)
# peak_abs_logS = 2.205  (measured span 420-700 nm)
log_s_pan_D10 = (-4.00, -4.00, -4.00, -4.00, -0.03, -0.03, -0.03, -0.05, -0.06, -0.01, -0.10, -0.19, -0.23, -0.21, -0.18, -0.18, -0.21, -0.23, -0.25, -0.30, -0.32, -0.32, -0.34, -0.51, -0.69, -0.72, -0.80, -1.11, -1.79, -4.00, -4.00, -4.00, -4.00)
# peak_abs_logS = 1.231
```

### 6.12 `KODAK_PLUS_X_125` (panchromatic)

* Source: `PDF/PROFILES/KODAK/f4018-125PX-2007.pdf` page 9 (F-4018, May 2007); frame (86.0, 62.9, 286.4, 214.3) pt
* x: `nm = 2.492223·px + 34.0594` → **rms 0.271 nm, max 0.739 nm**
* y: ticks [−1, 0, 1, 2, 3] → `logS = −0.026809·py + 4.7332` → **rms 0.0049, max 0.0088**
* 203 exact vector points, 379.5–659.8 nm, 1.26 nm point spacing, no gaps, peak abs logS **1.921** @ 407.0 nm

```python
log_s_pan = (-2.25, -0.85, -0.12, -0.01, -0.07, -0.12, -0.20, -0.30, -0.38, -0.46, -0.55, -0.65, -0.73, -0.79, -0.80, -0.76, -0.72, -0.70, -0.65, -0.61, -0.61, -0.65, -0.70, -0.72, -0.70, -0.72, -0.97, -1.48, -4.00, -4.00, -4.00, -4.00, -4.00)
# peak_abs_logS = 1.921   (measured span 380-650 nm)
```

> ⚠ **Criterion ambiguous.** The figure's legend names **both** `D = 1.00 > min` and `D = 0.30 > min`, but
> the page contains only **one** continuous vector curve (203 points) — the second curve is not present as
> vector geometry. I cannot tell from the geometry which criterion the surviving curve is, and I will not
> guess. Either read the figure visually to settle it, or store with
> `criterion='relative_log'` and a provenance note recording the ambiguity.

### 6.13 `KODAK_T400CN` (chromogenic B&W, panchromatic)

* Source: `PDF/PROFILES/KODAK/f2350-T400CN.pdf` page 6 (F-2350); frame (74.7, 469.4, 275.2, 620.8) pt
* x: `nm = 2.490978·px + 62.3220` → **rms 0.271 nm, max 0.739 nm**
* y: ticks [0,1,2,3,4] → `logS = −0.026795·py + 16.6018` → **rms 0.0049, max 0.0088**
* Conditions: daylight, 1/100 s, process C-41, Status M, density 0.2 above D-min
* 183 exact vector points, 398.5–648.7 nm, 1.27 nm spacing, no gaps, peak abs logS **2.366** @ 570.8 nm

```python
log_s_pan = (-4.00, -4.00, -1.13, -0.56, -0.42, -0.42, -0.49, -0.52, -0.55, -0.58, -0.56, -0.50, -0.47, -0.38, -0.24, -0.21, -0.25, -0.19, -0.15, -0.00, -0.05, -0.17, -0.29, -0.33, -0.29, -0.30, -0.89, -4.00, -4.00, -4.00, -4.00, -4.00, -4.00)
# peak_abs_logS = 2.366   (measured span 400-640 nm)
```

### 6.14 `KODAK_BW400CN` (chromogenic B&W, panchromatic)

* Source: `PDF/PROFILES/KODAK/f4036-BW400CN.pdf` page 5 (F-4036); frame (363.0, 344.5, 563.5, 495.6) pt.
  Identical figure in `Kodak PROFESSIONAL BW400CN Film.pdf` p5.
* x: `nm = 2.507032·px − 662.0694` → **rms 1.872 nm, max residual 3.356 nm**
* y: ticks [0,1,2,3,4] → `logS = −0.026515·py + 13.1164` → **rms 0.0204, max residual 0.0293 log**
* 210 exact vector points, 388.7–650.2 nm, 1.19 nm spacing, no gaps, peak abs logS **2.346** @ 571.5 nm

```python
log_s_pan = (-4.00, -1.72, -1.16, -0.57, -0.42, -0.42, -0.49, -0.52, -0.54, -0.57, -0.56, -0.50, -0.46, -0.39, -0.26, -0.20, -0.26, -0.19, -0.16, -0.01, -0.04, -0.14, -0.28, -0.32, -0.29, -0.28, -0.81, -1.75, -4.00, -4.00, -4.00, -4.00, -4.00)
# peak_abs_logS = 2.346   (measured span 390-650 nm)
```

> ⚠ **This sheet's calibration is an order of magnitude worse than the rest of the batch** — 1.9 nm rms,
> 3.4 nm max on the wavelength axis and 0.020 log on the ordinate. The cause is the file itself: its tick
> labels are placed with visibly irregular spacing (label-centre steps of 17.9 to 21.8 pt where 20.05 pt
> is exact), i.e. the sheet was re-typeset and the text positions were rounded. The *curve* geometry is
> exact; only the axis inference is degraded. 3.4 nm is a third of a grid step, so the array is usable but
> should carry the residual in its provenance. Note also that BW400CN and T400CN come out essentially the
> same curve (peaks 571.5 vs 570.8 nm) — consistent with them being the same emulsion family.

---

## 7. WHAT IS REAL AND WHAT WAS A FALSE POSITIVE — plain answer

* **Real.** 516 pages in 245 PDFs carry genuine, exact vector plot curves. 152 of those pages, in 111
  PDFs, are spectral-sensitivity plots. 14 stocks' worth of curves were extracted here (13 of them DB
  stocks with no spectral data at all, plus one revalidation); a further ~20 vector spectral plots exist
  for stocks that already hold curves and are revalidation material only.
* **False positives.** 421 of the 937 raw ≥30-item-path pages had no plot curve at all:
  * **28 pages** — the Kodak logo, exactly the documented ~30-item filled cluster;
  * **16 pages** — glyph-outline text blocks of 500–1350 items, the documented second trap;
  * **377 pages** — other filled artwork (headline text converted to curves, wordmarks, legend swatches).
  All were eliminated structurally by the `type == "f"` / `closePath` test, not by inspection.
* **A third trap, not previously documented.** Kodak's 2000s still-film sheets draw *real* curve segments
  as `fs` paths with a **white fill** and `closePath == False`. Filtering on "fill must be None" — the
  obvious first guess — silently deletes whole plots (GOLD 100/200 was invisible until this was fixed).
  Any future sweep must key on `closePath`, not on `fill`.
* **Genuinely not extractable from the corpus.** The Ilford fact sheets' "SPECTRAL SENSITIVITY — wedge
  spectrogram" figures are vector, but carry **no numeric axis tick labels whatsoever**. There is nothing
  to calibrate against and the plots do not support a sampling step. They stay in NotFound.

## 8. Reproducibility

The sweep and the extractor are deterministic and re-runnable from PyMuPDF alone. Per-plot inputs needed
to reproduce any block above: the PDF path, page number, and the plot frame rectangle quoted in that
block; everything else (tick labels, curve chains, layer assignment) is recovered automatically from the
page. Layer assignment is by peak-wavelength ordering and was cross-checked against the sheets' own
"Yellow-/Magenta-/Cyan-Forming Layer" in-plot labels on every colour plot in §6.
