# DIGITIZATION_QUEUE — archived narrative, batches 1–14

**Archived 2026-08-17.** This is the layered, chronological form of the
digitisation queue as it stood after fourteen batches, preserved verbatim
because the *reasoning* in it is the valuable part: why particular plots were
refused, which traps were hit, and what each failure taught.

The live queue was rewritten as a short working document —
`DIGITIZATION_QUEUE.md` — carrying only the binding method rules, the do-not-trace
list, the genuinely open items with their blockers, and a compact done table.
Nothing here was deleted; the live file points back to this archive for the
detail behind every entry.

---

# Curve digitization — status and production queue

Owner directive 2026-08-02: use every curve from its graphical
representation; where no plot exists, use printed text/table values.
Tool: `digitize_plot.py` (600 dpi render → frame/tick auto-detection →
seeded ink-centroid tracing → least-squares ToneCurve fit; numpy+Pillow
only). Accuracy per traced curve: RMS 0.003–0.007 D, max ≤ 0.035 D —
bounded by the printed line width, ~10× better than visual reading.

## Done (machine-traced and adopted, [T1] shapes)

| Stock | Plot | Samples | Fit RMS / max (D) |
|---|---|---|---|
| KODAK_VISION3_250D_5207 | H-1-5207 p3 sensitometric, 3 layers | 1426/layer | 0.005/0.003/0.005, ≤0.018 |
| FUJI_NEOPAN_ACROS_100 | AF3-095E p5, Microfine 15 min Ḡ=0.65 | 1092 | 0.007, ≤0.035 |

Adoption notes: 5207 moved to dmin_ladder encoding (sheet-absolute
Status M dmins are the mask); ACROS mid-grey anchored at the previous
hand-fit density (speed unchanged), shoulder beyond the printed range
kept at the monotonicity bound and flagged [T3 there].

Spectral sensitivity (visual transcription, v3 pilots, ±0.05–0.1 log):
FUJI_NEOPAN_ACROS_100, KODAK_VISION3_250D_5207, KONICA_INFRARED_750.
Dufaycolor reseau matrix from NSMM measured absorbance curves.

Batch 2 (agent digitization pass, pixel-calibrated against printed
gridlines, ±0.03–0.05 log), adopted 2026-08-02 — **13 stocks now carry
per-emulsion spectral curves**:
FUJI_VELVIA_50 (AF3-0221E2, 2007), FUJI_PROVIA_400X (AF3-0213E, 2006),
FUJI_SENSIA_100 (AF3-091E, 2001), FUJI_NEOPAN_1600 (AF3-608E, 1995,
scanned sheet), KODAK_VISION3_50D_5203 + 200T_5213 (H-1 sheets, rev.
3-26), KODAK_PORTRA_400 (E-4050, Sept 2010 — incl. the real red-layer
shelf at −1.8 across 490–560 nm), ILFORD_HP5_PLUS_400 + ILFORD_DELTA_3200
(Nov 2018 wedge spectrograms, tungsten), KODAK_TRI_X_REVERSAL_200
(H-1-7266). Every curve is per-emulsion — no vendor-shared shapes; B&W
stocks carry their own single pan curves.

## Queue — H&D characteristic curves (plots on file)

Priority 1, Kodak H-1/TI sheets — STATUS CORRECTED 2026-08-16: 5203,
5213, 5219, 5222 were ALREADY machine-traced [T1] on 2026-08-02 (batch
5) — this list was stale. Independent re-traces on 2026-08-16
reproduced all of them (5219 r to 3 decimals) — recorded as mutual
validation. NEWLY DONE 2026-08-16 (batch 9, below): TRI-X 7266,
EKTACHROME 100D 5285 (vector), 2383 print (vector). Remaining: still sheets:
Portra 400/160 (E-4050/E-4051), Ektar (E-4046), T-MAX (F-4016/F-4043),
Tri-X (F-4017), P3200 (F-4001), Gold/UltraMax (E-70xx).
Priority 2, Fuji AF3 sheets (per-developer families for B&W; single
curves for colour): Acros 120, Neopan 1600 (SCANNED, no text layer —
plots still traceable), Velvia 50/100, Provia 100F/400X, Astia,
Sensia 100/200/400, Superia family, Pro 160S/160C/400H/800Z, T64, 64T-II.
Priority 3, Harman/Ilford (curve + G-bar table per sheet): HP5+, FP4+,
Delta 100/400/3200, Pan F+, SFX 200, XP2, Ortho+, Kentmere Pan 100/400.
Priority 4, Soviet books (scanned plots, rougher line work):
Gurlev fig. 176 (Foto-32/65/130/250 family), fig. 177 (МЗ-3Л, СТ-4
1–24 min), fig. 178 (ОЧ-45/ОЧ-180 reversal + kinetics), fig. 197
(ДС-4 / ЦНД-32 / ЦНЛ-32 / ЦНЛ-65 per-layer), fig. 198 (ЦО-22/32Д);
Справочник кинооператора table X-2 companion plots if present.
Priority 5: Polaroid fds sheets (small curves), ORWO Wolfen datasheets
(NP100/UN54/N74/PF2 have curves), Foma, Rollei, Maco, Agfa APX
(agfa_films.pdf p10 curve triptychs), Konica colour sheets.

## Queue — spectral sensitivity plots (extend schema v3)

STATUS AFTER BATCH 3 (2026-08-02, three parallel agents): **35 of 89
stocks digitised** — everything in the archive that prints a spectral
plot for a DB stock is done (Agfa APX trio + Optima, Kodak: Kodachrome
64, Ektachrome 64/160T/100D, Double-X 5222, EXR 500T via TI2082, VISION3
50D/200T/250D, Portra 400, Tri-X 7266; Fuji: Velvia 50, Provia 400X,
Sensia 100, Neopan 1600, ACROS; Harman: HP5+, Delta 3200; Konica: all
six + IR750; Rollei: R3, Infrared, Retro 400; Fomapan 400; Polaroid
664/667).

Verified NO-PLOT sheets (text/table category per owner rule):
AGFA_VISTA_200, KENTMERE_PAN_100, KENTMERE_PAN_400.
Still queued: KODAK_VISION3_500T_5219 — only the brochure is on file;
add the H-1-5219 technical sheet to KODAK/ and trace.
Stocks not in the DB were not digitised (e.g. FP4+, Delta 100/400,
Pan F+, SFX — sheets on file but no profiles; digitise when/if added).
Soviet: no plots for the Foto line — text limits only (Δλ_S per
Gurlev/Chibisov: 645/665/580/630 nm) → class templates anchored at the
printed cut, per owner rule. МЗ-3Л HAS a plot (Gurlev fig. 177 spectral
panel, 300–500 nm) — traceable, feeds TASMA_POSITIVE_28 print stock.
Dufaycolor/Agfacolor: measured OD scans (NSMM/BArch) — Dufaycolor done,
Agfacolor Neu pending (extractedODs_MSI_BArch jpgs).

## Queue — MTF curves (two-term fit into mtf_tail_a/f_exp)

Consumed today only by the C++ port (film_sim.py uses the Gaussian
core; wiring the tail there is a renderer change, listed in Known
limits). Sheets with printed MTF curves: all Kodak H-1/TI motion
sheets (5207 traced region available), T-MAX/Tri-X F-pubs, Fuji AF3
colour sheets, Acros. Fit MTF(f) = a·exp(−ln2(f/f50)²)+(1−a)·exp(−(f/f50)^p).

## Queue — spectral dye density curves

H-1-5207-style "Spectral Dye Density" plots exist on every Kodak colour
sheet + Fuji colour sheets. No schema field yet — needs a v4 struct
(same pattern as SpectralSensitivity) before data entry. Design first,
then batch.

## Queue — Kodak Data Book 1952 + Agfa 2003 (added 2026-08-11)

Seven stocks adopted from these documents with [T2] curve readings; the
printed plots can upgrade all of them:

| Stock | Plot | Where |
|---|---|---|
| KODAK_VERICHROME_1952 | D-logE family vs dev time, sunlight | kodak-films-5.pdf p34 |
| KODAK_PANATOMIC_X_SHEET_1952 | D-logE family (DK-50) | p58 |
| KODAK_TRI_X_SHEET_1952 | D-logE family (DK-50) | p50 |
| KODAK_ORTHO_X_SHEET_1952 | D-logE family (DK-50) | p60 |
| AGFA_OPTIMA_200 | spectral sens. + density + MTF + colour density curves | AGFA stocks.pdf p6 |
| AGFA_OPTIMA_400 | same set | p6 |
| AGFA_PORTRAIT_160 | same set | p5 |

Kodak plots are vector line art in the PDF (21 drawing objects on the
Tri-X page) — render at 600 dpi and machine-trace per method rule 1.
Trace the RECOMMENDED-time member of each family (the profile comments
name it) and keep the family for a future push/pull feature. The Agfa
pages carry per-layer spectral sensitivity — same digitisation route as
the Optima 100 batch-3 curves already adopted.

## Queue — Eastman 1942 MP book (added 2026-08-11, second pass)

`KODAK/Kodak - [1942] - Eastman Motion Picture Films for Professional
Use.pdf` — scan, no text layer, OCR-indexed. Curves are printed line art
on scanned pages (NOT vector): expect scan-grade tracing accuracy, one
tier below the vector-PDF traces.

| Stock / target | Plot | Where (PDF page) |
|---|---|---|
| EASTMAN_SUPER_XX_1938 | Type 1232 H&D (sunlight, IIb, SD-21 65°F) + time-gamma/time-fog family | p49 |
| (same, cross-check) | combined picture-negative H&D figure, all 1942 stocks on one plot | p44 |
| EASTMAN_PLUS_X_5231 predecessor (Type 1231) | H&D + time-gamma family | p48 |
| unprofiled: Background 1213, Background-X 1230, 16 mm 5240/5242 | per-sheet H&D + time-gamma | p46, p47, p50, p51 |
| print-stock data: Release Positive 1301/5301 | sheet incl. D-16 γ 2.10 aim | p56 |

## Queue — SMPTE 1985 Sehlin/Kennel paper (added 2026-08-11, third pass)

`KODAK/Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf` — scan, no
text layer, OCR-indexed. File name says 1983; printed running head says
SMPTE Journal, July 1985. Kodak-authored measured plots for the two
profiled Eastmancolor stocks. HIGH VALUE: Fig 11 is the only measured
granularity-vs-exposure data in the archive for ANY colour negative —
it replaces estimated `sigma_shape_toe/dmax` with measurement.

| Target | Plot | Where (PDF page) |
|---|---|---|
| EASTMAN_5247_1974 grain shape | Fig 7 + Fig 11a, green RMS granularity vs rel. logE | p3, p5 |
| EASTMAN_5294_1983 grain shape | Fig 11b, same | p5 |
| both, MTF check | Fig 12, MTF vs exposure | p5 |
| EASTMAN_5294_1983 curves | Fig 14, green sensitometric, normal/over/under | p5-6 |
| latitude behaviour (both) | Figs 8-10, 15-18, Tables 1-2 | p4-8 |

Caveats to resolve during tracing: granularity axis units and aperture
must be reconciled with the 48 um diffuse-RMS convention before any
absolute value moves into GrainSpec.rms — the SHAPE (ratio toe/mid/dense)
transfers regardless of the absolute calibration; the paper's 5247 is the
EI-125 late version ("5247/II") while the profile targets the 1974-82
EI-100 original, so shapes adopt with a version note.

## Queued 2026-08-13 — new landing (see PDF_LANDING_2026-08-13.md)

* **Kodak Publication F-5 "Professional Black-and-White Films"** — 88 JPG
  page scans, late-1970s edition (ISO speed marking), ~4700x6500 px,
  excellent legibility. DS data-sheet insert, one stock per sheet (DS 3
  Contrast Process Ortho 4154 and DS 18 Tri-X Pan confirmed by sample).
  Each sheet: curve family at 3+ development times with contrast-index
  labels, CI-vs-time for up to 6 developers, dev tables 5 temps x 2
  agitation regimes, resolving power at both target contrasts. The only
  archive source with the processing axis for the professional B&W still
  line. First step: OCR-index all 88 pages, catalogue the DS sheets.
* **Wratten filter transmittances** (KODAK/transmision of wratten
  filters.pdf) — 8 p scan, no text layer; spectral transmittance plates
  for the taking-filter registry.
* **Dufaycolor measured-OD plates** — 3 multispectral JPGs
  (measuredODs_MSI_NSMM_11948/11951/11960): measured spectral density of
  surviving material; first measured data for DUFAYCOLOR_1937.
* **Agfacolor Neu extracted-OD plates** — 2 multispectral JPGs (BArch);
  first measured data for AGFACOLOR_NEU_1936 / NEG_TYPE_B_1943. The three
  German papers (1950/1951/1961) are scans without text layers.
* **«Фотография в прошлом, настоящем и будущем»** — 184 p scan, no text
  layer; unassessed content, low priority until OCR-indexed.

## Queued 2026-08-13 (second batch) — F-5 catalogued

F-5 OCR index complete. DS sheets pp 33-57: Panatomic-X, Plus-X Pan (+Prof),
Tri-X Pan DS18 (+Prof DS20), Verichrome Pan DS22, Royal Pan 4141 DS16,
Royal-X Pan, Ektapan 4162 DS5, Contrast Process Ortho 4154 DS3, HS Infrared
4443, Recording 2475. Each: curve family at 3 dev times w/ CI labels,
CI-vs-time (6 developers), dev tables 5 temps x 2 agitation, RP both
contrasts. ~1979 formulations: digitise when the 1970s versions enter the
database as their own stocks; do NOT graft onto the _1952 profiles.

## Queued 2026-08-13 (third batch) — Vision3 granularity curves

The four Vision3 TI sheets on file (5203/5207/5213/5219) print granularity
as a **Sigma-D-versus-density curve** ("read the number and multiply by
1000"), not a single figure — the exact GRN.1 form of Appendix A. Digitising
the four curves feeds sigma_shape_toe/mid/dmax per stock, replacing the
current single-amplitude rendering on the most-used modern MP line. Also on
file and queued: 7266 Tri-X reversal sheet, 5296 EXR sheet, 5294/7294
Ektachrome 100D sheets ([C3] toward the 5285 profile at most).

## Queued 2026-08-13 (fourth batch) — spectral curve RE-TRACE at 5 nm

Measured justification in `CURVE_RESOLUTION_ANALYSIS_2026-08-13.md`. The present
stored values are a **downsampled** 600 dpi trace, so re-tracing recovers real
information rather than interpolating: `digitize_plot.py` samples every pixel
column and downsamples on request, and 10 nm was the downsample target.

Per-plot human input is required (page and crop, axis calibration values off the
printed labels, one seed pixel per curve), so this is roughly 100 supervised
traces, not a batch job. Ordered by measured value, not by convenience:

1. **FUJI_NEOPAN_ACROS_100** — the only stock with a demonstrated real gradient
   finer than the stored sampling: **1.70 decades per 10 nm at 650 nm**, a factor
   of 50 across one sample interval at the red cut-off. Here the stored curve is
   at or past its Nyquist limit and the interpolator is guessing. The one case
   where 5 nm recovers structure rather than polishing.
2. **The four stocks stored coarser than 10 nm** — AGFACOLOR_NEG_TYPE_B_1943,
   GEVACHROME_902, GEVACOLOR_NEG_682 (25 nm) and FUJICOLOR_A250 (20 nm). Cost up
   to 2 % on the balance derivation now and 231 % under a line source. All four
   come from journal papers or older sheets, so **check first whether the plot
   supports 10 nm at all** — if it does not, record that and leave the sampling
   alone rather than inventing samples.
3. The remaining colour stocks. Measured benefit 0.4–1.1 %; the honest
   justification is future-proofing against non-blackbody illuminants, not
   present error. Low priority, and it should be said as such in each profile
   comment.

DO NOT resample the stored arrays onto a finer grid as a substitute. It
interpolates, adds nothing, and destroys the record of which samples came from
the plot. The engine already integrates on a 2 nm grid, which is finer than any
storage target here and invents nothing.

## Method rules (bind all future entries)

1. Machine-trace whenever the plot is printed; visual transcription only
   where line work is too broken for tracking (state which).
2. Every curve carries: author/publisher, document title + code,
   ORIGINAL document release date; Russian sources cited in the original
   language plus English translation.
3. Fit residuals (RMS/max) quoted in the profile comment; anchoring
   decisions (mid-grey, unconstrained shoulders) stated explicitly.
4. Where plot and printed table disagree, record both, adopt the more
   product-specific, keep the conflict auditable.
5. After every adoption: verify.py green, C++ regenerated + compiled,
   film_names.txt regenerated, docs updated. No exceptions.

---

## Batch 5 — 2026-08-13. Cheltsov & Bongard 1958: four data classes with nowhere to go

Source: `PDF/PROFILES/cheltsov_vs_bongard_sa_tsvetnoe_proiavlenie_trekhsloinykh_sv.pdf`
Чельцов В. С., Бонгард С. А., «Цветное проявление трёхслойных светочувствительных
материалов», М.: Искусство, 1958.

Unlike batches 1–4 this is **not** a tracing queue. Every number below is already
printed as text in the source — no plot digitisation is needed. What blocks it is that
the schema has no field. Recorded verbatim so the reading survives.

### 5.1 Per-layer resolving power — Table 24, p200 (verified from word coordinates)

| Film | yellow | magenta | cyan |
|---|---|---|---|
| Agfacolor negative | 80 (top) | 34 (mid) | 27 (bot) |
| Agfacolor positive | 102 (top) | 42 (mid) | 30 (bot) |
| Eastmancolor negative 5248 | 110 (top) | 46 (mid) | 30 (bot) |
| Eastmancolor positive 5382 | 37 (bot) | 200 (top) | 97 (mid) |
| Duponcolor positive 275 | 35 (bot) | 185 (top) | 52 (mid) |

Units: штрих/мм (lines/mm). Layer tags as printed: в. = верхний (top), с. = средний
(middle), н. = нижний (bottom). **The flat OCR text scrambles the cyan column** — a
first reading assigned 27 to the Eastmancolor negative. Rebuilt from word coordinates:
27 is Agfacolor's, Eastmancolor's is 30. Use the table above, not the raw text.

Blocked on: `MTFSpec` carries three *records* (R/G/B), these are three *physical
layers*, and the mapping requires a layer-order field that does not exist. Entering
them as records would silently assert natural layer order for the three films that do
not have it.

### 5.2 Layer order permutations — no field exists

| Film | order (top → bottom) |
|---|---|
| TsP-6 positive | green/magenta, red/cyan, **blue/yellow at bottom** |
| Eastmancolor positive 5382 | green 510–570, red 660–700, blue 380–480 at bottom |
| Gevacolor positive 952 | green-sensitive top, blue-sensitive bottom |
| Duponcolor positive 275 | blue top, **red middle, green bottom** |
| Telcolor negative | blue top, **red middle, green bottom** |
| Eastmancolor dupe 5245 | magenta top, cyan middle, yellow bottom |

Documented motive: put the magenta record, which dominates perceived sharpness, in the
layer light strikes first. Measured consequence in §5.1.

### 5.3 Processing recipes — the PRC axis (A.5), still empty

Complete published compositions in g/L with time and temperature for: Kodachrome
(five-bath selective re-exposure sequence, p147–149), Agfacolor reversal (amidol first
developer 35 min at 18 °C, colour developer 11 min at 18 °C, p160), Anscocolor reversal
(p162–163), Gevacolor reversal after Thomson (p164–166), Ektachrome (p219),
Eastmancolor (p228 ff.), and Agfacolor negative/positive (p183). Teigaard's comparative
tables 21 and 22 (p195) give times for six manufacturers' negatives side by side.

### 5.4 Silver vs dye gamma divergence — Table 14, p123

| dev. time (min) | γ silver | γ dye | ratio |
|---|---|---|---|
| 3 | 0.25 | 0.33 | 1.32 |
| 4 | 0.34 | 0.68 | 2.00 |
| 5 | 0.47 | 1.03 | 2.19 |
| 6 | 0.69 | 1.74 | 2.42 |
| 8 | 0.73 | 2.09 | 2.86 |

A quantitative statement about a mechanism the engine models only qualitatively: dye
density is proportional to the *surface concentration* of developed silver, not to
silver optical density, because silver covering power changes with development. Also
documented (p123): the dye image's straight-line portion is displaced toward **lower**
exposures relative to the silver image's, sometimes lying entirely in what is the
silver curve's underexposure region.

### 5.5 Dye absorption maxima — usable now if a dye-spectra field is ever added

| Film | yellow | magenta | cyan |
|---|---|---|---|
| Anscocolor reversal | — | 540 nm | 660 nm |
| Anscocolor positive 848 | 440 nm | 540 nm | 660 nm |
| Gevacolor (rev. and neg.) | — | **550 nm** | 660 nm |
| Ferraniacolor reversal | — | = Agfacolor's | ≈ Gevacolor's |
| Agfacolor | plotted, fig. 46/57/59 | plotted | plotted |

Gevacolor's 550 nm magenta is the family outlier — pushed long, visibly bluish, with
heavy unwanted blue absorption; its cyan at 660 nm is pushed short, which the source
notes crowds the magenta band and harms printing selectivity. Both shifts are the
documented reason `GEVACOLOR_NEG_652` carries strong positive dye off-diagonals.

**DO NOT** synthesise spectral sensitivity curves from the peak wavelengths in this
book. Three peaks are not a curve; Gaussians fitted through them would manufacture a
shape nobody measured. The same prohibition as batch 4's "do not resample the stored
arrays".

---

## Batch 6 — 2026-08-14. The Compact Photo-Lab-Index 1979: what NOT to trace

Source: `PDF/PROFILES/pittaro_em_the_compact_photolabindex.pdf` (Pittaro, ed., Morgan &
Morgan, 2nd Compact Edition 1979).

**Figure resolution: the PDF carries a 72 dpi full-page background PLUS 300 dpi
high-resolution overlays for the figures.** Plot areas and axis labels are legible at
300 dpi. Tracing is technically feasible. That is not the constraint.

### 6.1 Wedge spectrograms (6 Ilford films) — DO NOT TRACE

Pages 471, 474, 478, 488, 496, 501. **A wedge spectrogram is not a spectral sensitivity
curve.** It is a photographic strip: one axis wavelength, the other a density wedge, and
the upper envelope of the blackened region encodes relative log sensitivity. Converting
envelope → sensitivity requires the **wedge's own density calibration, which this book
does not print.** Tracing them yields an uncalibrated shape.

They are also redundant. The one thing they would give — sensitisation extent — is
published numerically and more precisely on p559 (see §6.2). **Recommendation: skip.**
This saves roughly six supervised traces that would have produced data we could not
place on an axis.

### 6.2 Ilford sensitivity ranges — already numeric, no tracing needed (p559)

Films and plates, range in Ångströms, with Ilford's own speed/contrast/grain words:

| Material | Range (Å) | Speed | Contrast | Grain |
|---|---|---|---|---|
| Micro-neg Pan Film | 2300–6600 | slow | very high | extremely fine |
| Pan F Film | 2300–6700 | medium | medium–high | very fine |
| FP4 Film | 2300–6700 | medium | medium | very fine |
| Aerial A Film | 2300–6700 | fast | high | fine |
| HP4 Film | 2300–6700 | very fast | medium | medium |
| R.52 Plate | 2300–6600 | slow | very high | very fine |
| R.40 Rapid Process Pan Plate | 2300–6700 | medium | high | fine |
| R.20 Special Rapid Pan Plate | 2300–6500 | medium | medium | fine |
| R.10 Soft Gradation Pan Plate | 2300–6500 | fast | medium | medium |
| FP4 Plate | 2300–6600 | fast | medium | fine |
| R.30 / R.30M Trichrome Plates | 2300–6600 | fast | medium–high | medium |
| Astra III Plate † | 2300–7100 | fast | medium | medium |
| HP3 Plates ‡ | 2300–6700 | very fast | medium | medium |
| Holographic Plate HeNe † | 2300–6700 | very slow | very high | extremely fine |

† laboratory product, made to order only. Astra III "retains most of its speed when
exposure times are very long"; HP3 Plates "…when exposure times are very short" — those
are qualitative reciprocity statements at opposite ends, and the short end is a regime
`ReciprocitySpec` has no term for.

**Every entry starts at 2300 Å = 230 nm. Our spectral grid starts at 360 nm.**

### 6.3 Contrast-index vs development-time curves — traceable, PRC axis (10 pages)

Pages 15, 59, 71, 82, 84, 87, 89, 92, 476, 489. Legible at 300 dpi with numeric axes
(e.g. Pan F: CI 0.2–1.0 vertical, 2–16 minutes horizontal, three developer curves —
Microphen, ID-11, Perceptol — drawn dashed). Dashed lines need gap interpolation.

Blocked on: there is no processing axis in the schema to receive them. For Pan F the
tabulated form is already extracted and needs no tracing at all (see §6.4).

### 6.4 Pan F development matrix — already extracted, tabular (p473)

| Developer | dilution | min → CI 0.55 | min → CI 0.70 |
|---|---|---|---|
| ID-11 (D-76 type) | 1+1 | 9 | 14 |
| ID-11 | 1+3 | 14 | 21 |
| Microphen | 1+1 | 5 | 8 |
| Microphen | 1+3 | 9 | 14 |
| Perceptol | 1+1 | 12 | 17 |
| Perceptol | 1+3 | 15 | 24 |

Speed by developer: ID-11 ASA 50 unchanged; Microphen DIN 20 (≈ ASA 80); Perceptol
**ASA 32 / DIN 16**. Definition given in the source: contrast index is the average
gradient over 1.5 log-exposure units from a point 0.1 above fog.

### 6.5 Characteristic curves (24 pages) — traceable, low priority

Pages 60, 70, 76, 82, 87, 89, 92, 107, 147, 386, 404, 406 and others. Most stocks whose
curves these are either already have datasheet-derived curves or are not in the database.

### 6.6 Kodak reciprocity master table — DONE, do not re-extract from text

Pages 174–175. **Rebuilt from word coordinates on 2026-08-14** and recorded in
`CHANGES_2026-08-14_photo_lab_index.md` §4. It does not survive flat text extraction —
the cells arrive without row/column association and fragments of the rotated running
head interleave with them. Use the reconstruction in that document, not the raw text.

---

## Batch 7 — 2026-08-14. CHECK FOR VECTOR PATHS FIRST

**A method note that should change how every future PDF is approached.**

`PDF/PROFILES/KODAK/Ektachrome_100d.pdf` (H-1-5285, Feb 2010) draws its curves as **PDF
vector polylines, not as a raster figure.** The spectral sensitivity plot is three
56-point paths. Coordinates are therefore *exact*; the only estimated step is axis
calibration, fitted by least squares to the printed tick centres and closing to **0.63 nm**
on wavelength and **0.009 log** on sensitivity — an order of magnitude better than tracing.

**Before queueing any plot for `digitize_plot.py`, run this check:**

```python
import fitz
d = fitz.open(path); p = d[page]
print(len(p.get_drawings()), len(p.get_images()))
print(max((len(x['items']) for x in p.get_drawings()), default=0))
```

A path with tens of `l` or `c` items is a curve. Many images and only short paths means
raster. Measured on the two documents that landed today: **Kodak H-1-5285 is vector**
(144-item paths, 0 images on the curve pages); **the Fujifilm cine manual is raster**
(20–21 images per page, only axis frames as vector).

### 7.1 Kodak H-1-5285 — extractable now, blocked only by the schema

| Curve set | Page | Status |
|---|---|---|
| Spectral sensitivity, 3 layers | 3 | **DONE 2026-08-14**, in the profile |
| **Spectral dye density**, C/M/Y + visual neutral, 400–700 nm, peak-normalised | 3 | Vector, exact, **no field exists** — a 3×3 `dye_matrix` stands in. This is the §A.5 L2 gap the Addendum records as "dye spectral densities absent". |
| Diffuse rms granularity vs density | 3 | Vector. Schema holds 3 scalars, not a curve. |
| Modulation transfer function, 1–600 cycles/mm | 2 | Vector. Schema holds 3 scalars (`f50_r/g/b`). |
| Characteristic curves, Status A, daylight 1/100 s | 2 | Vector. Profile already has datasheet-derived curves. |

Axis calibrations already solved for page 3, reusable:
`nm = 2.491521·x + 62.2789` (tick centres 75.29→250 … 275.77→750);
`log S = −0.026795·y + 13.5771` (431.84→+2.0 … 581.33→−2.0).

### 7.2 Fujifilm cine manual — raster, low priority

Characteristic curves for nine camera stocks plus intermediates and positives, all as
page images. Axes are printed as text (Density 0.0–3.0, Camera stop −6…+6, exposure
3200 K or 5400 K for 1/50 s through a Fuji SC-41, Status M). Traceable if the eight
deferred stocks are ever entered; not worth it before then.

---

## Batch 8 — 2026-08-15. NEOPAN 1600 done; and a tracing lesson worth reusing

`PDF/PROFILES/FUJI/datasheet_neopan1600superpresto_en_01.pdf` (AF3-608E) is a true digital
PDF whose graphs are nevertheless **300 dpi rasters**. The vector paths on its curve pages
are the FUJIFILM footer logo — the same trap as batch 7. Both curve sets have now been
traced and entered; nothing remains queued for this film.

**Three techniques from this trace that are worth reusing on the ~30 curve sets still
queued:**

1. **Mutual exclusion between tracks.** When several curves share a plot they converge at
   one end. A nearest-neighbour follower will happily put two tracks on the same stroke —
   here it produced identical Ḡ for two different development times, which is how the
   failure was spotted. Assign candidates with exclusion and enforce the tracks' vertical
   ordering.
2. **Validate against a printed summary statistic.** This sheet prints Ḡ per curve, so the
   trace was checkable rather than merely plausible: 0.548 / 0.769 / 0.916 against printed
   0.58 / 0.77 / 0.90. Prefer plots that print such a number, and check it before entering
   anything.
3. **Measure a fitted statistic on the MODEL, not on the trace.** Average gradient starts
   from where the curve crosses base+fog+0.10. Taking that threshold from the traced points
   instead of from the fitted curve shifted Ḡ by 0.04 and produced a wrong fit that a
   regression test then caught. Define the statistic on the object you are storing.

**Also confirmed:** fitting the full 6-parameter `ToneCurve` to several hundred traced
points is well-conditioned only if the schema's `shoulder_k <= 2*toe_k` guard is imposed
*inside* the search. Fitting freely and clamping afterwards produces a curve that fails
`validate_all()`.


---

## Batch 9 — 2026-08-16. Queue P1 execution: 7266, 5285, 2383 — and a monotonicity lesson

**Done and adopted ([T1]):**

| Stock | Source | Method | Fit RMS (D) |
|---|---|---|---|
| `KODAK_EKTACHROME_100D_5285` | H-1-5285 p3, 3×13-bezier VECTOR paths | exact bezier sampling, 312 pts/layer | 0.024/0.028/0.028 |
| `KODAK_TRI_X_REVERSAL_200` (7266) | 7266 TI sheet p3, 300 dpi raster | machine trace, 296 pts | 0.0167 |
| `KODAK_2383_RELEASE` print stock | 2383 sheet (2015) p5, 65-71-bezier VECTOR paths | exact sampling ~800 pts/layer, x=0 at the sheet's own LAD aims | 0.018/0.010/0.031 |

**The lesson (binds all future fits): shoulder_k < toe_k makes the sigmoid
difference NON-MONOTONE past the shoulder.** The unconstrained best fits for all
three (RMS as low as 0.0035) had sharper shoulders than toes and produced real
density reversals (up to −0.18 D/logH on 5285 b). Constraint added inside the
search: `toe_k <= shoulder_k <= 2*toe_k`. With `shoulder_k == toe_k` the model is
analytically monotone (constant positive sigmoid-argument gap). Cost: 0.01–0.02 D
of residual. verify.py's monotonicity check now scales its float32 ulp allowance
by each curve's own gamma, because shelf noise is proportional to gamma.

**Deferred from this batch:** Vision3 granularity Sigma-D curves (4 TI sheets,
rasters identified and extracted to page images — tracing queued next); Portra/
Ektar/T-MAX still-film sheets (vector check pending); 5294 (7266-style sheets on
file); KODAK DATA BOOK vol 5 (owner returned the file to local disk 2026-08-16).

---

## Batch 10 — 2026-08-16. NotFound.md §4 worked through; the inventory is now measured

**The "~30 vector curve sets" estimate was wrong by an order of magnitude.** Exhaustive
sweep of `PDF/PROFILES/**`: 9 413 pages carry drawings, 937 have a ≥30-item path, and
**516 pages in 245 PDFs are genuine vector curve plots** — 152 of them spectral
sensitivity, 238 characteristic, 125 granularity, 119 MTF, 54 spectral dye density. The
other 421 are the documented traps, now counted rather than feared: 28 Kodak-logo pages,
111 filled wordmark art, 266 glyph-outline headlines, 16 full glyph text blocks.

**Adopted — 14 stocks gained measured spectral curves** from exact vector coordinates
(calibration ≈0.27 nm rms on wavelength, ≤0.012 log on sensitivity): UltraMax 400/800,
Ektar 100, Portra 160/800/100T, Gold 100 + 200, Tri-X 400TX, T-MAX 100, T-MAX P3200,
Plus-X 125, T400CN, BW400CN. **Stocks with a spectral curve: 53 → 67 of 143.** Portra 400
was re-derived independently as revalidation (agreed to 0.03 log mean; no change).

Storage decisions, all recorded in the profiles: 10 nm sampling (coarser than the source
everywhere — decimation, not invention; do **not** resample the 3.33 nm sheets below 5 nm);
peak-normalised with `peak_abs_logS` preserved in the comment; −4.00 means "not plotted
here", never "measured zero"; the nine layers whose true peak falls between grid points
name their 0.01–0.06 log renormalisation shift; two-criterion B&W sheets store the
speed-defining D = 0.3 curve; Gold 100/200 share one curve because the sheet plots one.

**Still queued, each with its blocker named:** T-MAX 400 F-4043 p7 (the two criterion
curves extract with inconsistent shapes, peaks 528 vs 570 nm — confirm visually first);
APX 25/100/400 (frames drawn as `qu` quads, calibration needs a hand frame); Vista 200
(one page carries the 100/200/400/800 family — needs the legend read); Ilford
HP5+/Delta 3200/FP4/Pan F (**wedge spectrogram outlines with no numeric axis ticks at
all** — nothing to calibrate against, plus FP4 vs FP4 Plus identity); Polaroid
664/667/52/55 (decade-log ordinate — needs a storage-convention decision); Technical Pan
(multi-plot pages); Ultra Color 100UC/400UC (E-4035 not on disk).

**Closed as dead ends, so nobody re-reads them:** KODAK DATA BOOK vol 5 (346 pp swept —
zero RMS granularity, zero numeric gamma, resolving power on 8 pages only, and none of
those three figures maps to a held stock without a generation graft);
`centuria_pro_400.pdf` (2003 brochure, one prose ISO, and the wrong product —
CENTURIA **PRO**, not CENTURIA SUPER); `professional_160.pdf` (no ISO printed anywhere, no
matching stock); Konica IMP50/INF750 (**never image-only** — full text layers, already
mined; IMP50 prints resolving 63/160 lp/mm at both contrasts).

**Reclassified from "OCR" to raster-curve digitisation:** Gevacolor 682 Figs. 7/8/10/11/12
(spectral sensitivity, dye density, sensitometric, MTF, RMS-vs-density), professional_160
p4, IMP50 pp 2–3, INF750 pp 1/3.

### Batch 10a — the Vision3 granularity curves need a new tracer (attempted 2026-08-16)

Attempted and **deliberately not entered**. What was solved: the plot carries two curve
families on one frame (three solid density curves against the left axis, three **dashed**
σ_D curves against a right-hand **log** axis). Density calibration is exact — 0.0–3.0 over
the frame, 147.7 px per density unit on 5207, confirmed by 19 evenly spaced minor ticks at
0.2 D. The right axis reads 150 px per decade (0.10 at D≈2.05, 0.001 at D≈0.02,
cross-checked at 0.01/0.006/0.002 to within a few pixels).

**The useful insight:** `sigma_shape_toe/mid/dmax` are *ratios* normalised at D = 1.0, so a
multiplicative error in the log-axis calibration cancels entirely. Absolute σ accuracy is
therefore not the blocker.

**The actual blocker:** the σ_D curves are dashed, so a nearest-neighbour tracer with a
large `max_gap` bridges across *other* tracks. First attempt produced obvious nonsense —
5219's green ratio came out at 44.3 — the same track-merging failure recorded in batch 8,
made worse by the dashes. What is needed is a **dash-aware mutual-exclusion tracer**:
enforce the three tracks' vertical ordering, bridge only gaps shorter than the measured
dash period, and refuse to accept a bridge that crosses another track's corridor. Until
that exists, these four stocks keep their generic (0.4 / 1.0 / 1.2) estimate rather than
receiving fabricated shape numbers.

---

## Batch 11 — 2026-08-16. `dashtrace.py` written; T-MAX 400 blocker resolved

### Tool: `dashtrace.py` (in the generator directory, permanent)

Dash-aware mutual-exclusion tracer. One ink run may be claimed by at most one track per
column; a track may step at most `max_step` px between columns; gaps are bridged only up
to the **measured** dash period (7 px ink / 6 px gap on the VISION3 sheets → ceiling 12);
`check_ordering` asserts the tracks' vertical order rather than assuming it. On H-1-5207 it
gives **zero ordering violations across 186 density and 33 granularity columns** — the
property that was failing.

**Track classification solved.** A run's identity is decided by following it loosely ±35–40
columns and counting hits: **≥70/80 = solid curve, ~25–50 = dashed curve, <20 = a text
glyph or an axis tick**. This finally separates the in-plot "B/G/R" labels and the tick comb
from real data, which short-window persistence tests could not.

**Correction to an earlier assumption:** the granularity curves are **dashed only on 5207
and 5219**. On 5203 and 5213 they are drawn solid (all six tracks score 80/80), so those two
sheets need no dash handling at all.

**Still not entered.** Coverage remains partial — on 5207 the traced range reaches only
D 1.12 where the real curve starts near 0.9, so the "toe" sample would not be at the toe.
Ratios computed over a partial range would be wrong, so the four stocks keep their generic
(0.4 / 1.0 / 1.2). Remaining work is per-plot seed placement using the classifier above.

### Resolved: `KODAK_TMAX_400` spectral sensitivity — **adopted**

The batch-10 note said the two criterion curves "extract with inconsistent shapes (peaks
528 vs 570 nm, ranges 0.57 and 1.5 log) — do not enter without visual confirmation". The
page was rendered and inspected. **Nothing was mis-traced: the difference is real.** The two
curves are cleanly separated and never cross — solid **D = 0.3** peaks at 571 nm with a
steep red cliff past 630 nm, dashed **D = 1.0** peaks at 529 nm and is flatter. The earlier
reading merged them.

Adopted from F-4043 (2016) p7, PDF vector line art, 1216 sampled points over 359–648 nm,
calibration residual **0.85 nm / 0.009 log**, stored at 10 nm as the **D = 0.3** curve to
match TRI-X 400TX / T-MAX 100 / T-MAX P3200. Stocks with a spectral curve: **67 → 68**.
The `verify.py` guard now covers 15 vector-extracted curves.

---

## Batch 12 — 2026-08-17. Gorokhovskii УФН 1936: period curves, and why period numbers move

`PDF/PROFILES/SOVIET/МЕТОДЫ СПЕКТРАЛЬНОЙ СЕНСИТОМЕТРИИ - 1936.pdf`, 17 pp, full text layer,
all pages reviewed. A **methods review**, not a data source — nothing entered. Two figures
carry measured curves for named products, both **raster**, both several crossing dashed
traces: the case that needs the dash-aware tracer from batch 11.

| Figure | Page | Curves |
|---|---|---|
| Fig. 6 | 513 | Agfa Isochrom-Portraitfilm, Isopan-Portraitfilm, Isopan-ISS-Planfilm (Bilz 1935, criterion **D = 0.1 above fog**) |
| Fig. 7 | 514 | Persenso (Perutz); Monarch, Hypersensitive, Soft-gradation (Ilford); **«Изопанхром ФОКХТ»** (ГОИ Leningrad 1934–35, criterion **D = 1.0 above fog**) |
| Table 3 | 514 | numeric sensitivities at 436 / 546 nm for six named Agfa ortho and pan films, old vs new method |

**None of these products is in the database.** «Изопанхром ФОКХТ» is the tempting one —
`SOVIET_PANCHROM_1939` has no documentation at all — but it is a named product and that
stock is a generic 1939 panchromatic, so attaching it would be the FP4-vs-FP4-Plus graft.
Recorded as a candidate with the caveat rather than silently adopted.

**The methodological finding is the real value, and it binds future period work.** The same
class of material measured at **4–13 %** green(546)/blue(436) sensitivity ratio at ГОИ,
**30–60 %** by Bilz and **20–45 %** by Stobbe — driven by prism versus diffraction
spectrographs, not by the emulsions. Bilz states his own error as **19 %** at D = 0.1.
So: **any single 1930s spectral figure is method-bound**, the criterion (D = 0.1 vs D = 1.0
above fog) must be recorded with it, and two period sources must never be averaged.


---

## Batch 13 — 2026-08-17. APX trio re-extracted; Polaroid conversion attempted and refused

### Adopted: AGFA APX 25 / 100 / 400 spectral curves, from stroked vector paths

The queue had these logged as "frames are drawn as `qu` quads and needed a hand frame".
**They never needed one.** The frame is irrelevant: the axis TEXT gives the calibration
directly, and the curve itself is a single stroked path whose coordinates are exact.
Calibration closes to **0.50 nm and 0.003-0.004 log** on all three -- the cleanest of any
Agfa sheet in the corpus.

**This SUPERSEDES the 2026-08-02 visual transcription and validates it.** Over the 29
comparable samples the two agree to mean -0.014 to -0.021 log and max 0.10-0.16 log, well
inside the +/-0.05-0.1 that visual reading claimed. **Every largest discrepancy sits at the
red cut-off, 650-660 nm** -- where the curve is steepest and an eye is least reliable, which
is precisely where an exact path earns its keep. Vector-extracted spectral curves: 15 -> 18.

Method note worth reusing: when a plot's frame is filled rather than stroked, do not reach
for the raster frame-finder. Fit the calibration to the printed tick LABELS via
`get_text("words")` and ignore the frame entirely.

### Attempted and NOT entered: Polaroid 664 / 667 / 52 / 55 spectral curves

The decade-log ordinate (1 / 10 / 100 / 1000, equally spaced) was confirmed and the
conversion to `relative_log` is simply log10 of the printed value -- the owner approved that
convention. Three of the four sheets nonetheless produced **nothing usable**, and the fourth
is not trustworthy:

| Sheet | Result |
|---|---|
| 664 | traced, but x calibration closes to only **3.24 nm** -- six times worse than the APX sheets -- and the resulting curve is visibly wobbly. Either the label centres are being mis-estimated on this layout or the chosen path mixes the curve with another element |
| 667 | 4 x-labels found, **0 y-labels** |
| 52 | **0 x-labels**, 2 y-labels |
| 55 | **0 labels of either kind** |

The axis labels sit at different page positions on each sheet, so a single label-matching
window cannot serve all four. **Nothing was entered.** The honest next step is per-sheet
label windows plus a visual check of the 664 trace against its page image before any of the
four is adopted -- the same gate that caught the ЛН-8 OCR error. Entering a wobbly curve
because the machinery produced one would be the exact failure this queue exists to prevent.


---

## Batch 14 — 2026-08-17. Vision3 granularity: third attempt, and a recommendation to change instrument

**Nothing entered. The four stocks keep their generic (0.4 / 1.0 / 1.2) sigma(D) triple.**

### What was solved this round

- **Per-plot seeding, finally automatic.** Scanning columns and keeping only those with
  exactly six tracks classified as real (>=25 hits over a +/-35 column follow) and a minimum
  vertical gap of 12 px found clean seeds on 5203 (x=366), 5207 (x=374) and 5219 (x=374).
  5213 has no such column anywhere in its middle 60 %.
- **Ordering violations are now zero** on every sheet, in both families.
- **Coverage reached the full plot width** — 235-356 points on the green granularity track,
  294-422 on the green density track, spanning the real D range rather than stopping at 1.0.
- **The toe anchor problem was solved conceptually**: the granularity curve's LEFT END *is*
  the Dmin region, because the density curve is flat there by construction. So sigma_toe can
  be read at the left end without the density trace having to reach Dmin; only the mid
  anchor needs the density curve, and D = 1.0 sits comfortably inside its traced range.

### Why it is still not adopted

The three sheets produce **mutually contradictory shapes**:

| Stock | toe / mid / dmax |
|---|---|
| VISION3 250D 5207 | 1.00 / 1.00 / **2.56** |
| VISION3 500T 5219 | 0.65 / 1.00 / **0.70** |
| VISION3 50D 5203 | 0.83 / 1.00 / **0.67** |

Three films of one product line, one plot format, one tracer: 5207 says grain grows 2.6x
toward Dmax while the other two say it falls by a third. For a colour negative sigma_D
should RISE with density, since more developed silver means more grain. At least two of
these three are wrong, and possibly all three.

**The important part is that nothing detected it.** Ordering checks passed, coverage was
good, the numbers looked plausible in isolation. Only comparing siblings against physics
exposed it — the same discipline that caught the ~2x RMS bias and the merged NEOPAN tracks.

### Recommendation: change the instrument, do not retry the same way

This is the third attempt across two sessions and the failure mode is subtle every time.
The remaining uncertainty is **track identity**, not calibration or coverage: where the
rising density curves cross the flat granularity curves, a tracer with correct within-family
ordering can still swap a curve for its neighbour without violating anything it checks.

The next attempt should **start** with the check that has been left to the end: render the
traced granularity track as an overlay on the page image and look at it. If the overlay is
right the numbers follow; if it is wrong, no amount of parameter tuning will fix it. Until
someone does that, these four stocks are better served by an honest generic shape than by a
measured-looking wrong one.


---

## Batch 15 — 2026-08-17. Vision3 granularity: fourth attempt, ADOPTED

**The overlay was run first this time, as batch 14 recommended, and it decided everything
within the first look.** Full write-up: `RESULT_2026-08-17f_vision3_granularity.md`.

### What the overlay showed that three passes of numbers could not

Painting a right-seeded, leftward trace onto 5203 showed `Db` leaving the thin B density
curve and continuing down the **bold granularity** curve, `Dg` ending on the R density
curve, `Dr` ending on a granularity curve. Batch 14's §3 diagnosis was exactly right.

The new fact: **the crossing is tangential, and decidable in only one direction.** At
log E ≈ 1.1 on 5203 the rising B density curve passes through the B granularity curve *at
that curve's maximum*. Traced leftward, both branches descend from the junction with similar
slope — neither proximity nor slope can choose, which is why three attempts died there.
Traced rightward from the left plateau, one branch rises and the other sits at slope ≈ 0.
So "seed at the left edge and trace rightward" is not a convenience; it is the only
direction in which the junction has an answer.

### Where the batch-14 plan needed changing

Seeding the density curves by identity at the left edge is **necessary but not sufficient**.
Two sheets do not even present six separated runs there, because the granularity curves lie
on top of the density curves: 5203 gives **5 ink runs for 6 curves**, 5213 gives **4**.

What works is separating the families by **stroke style first** — which is how the sheets
themselves distinguish them, and 5219 prints a legend saying so ("Blue Density … Blue
Grain"). Connected-component width isolates dash from solid on 5207/5219 (three components
of 424–442 px against a largest dash of 11 px — the threshold sits in a wide empty gap);
vertical run length isolates bold from thin on 5203/5213 (4–9 px against 1–3). With the
families in separate ink masks, cross-family capture is **impossible**, not merely checked.

Third change: slope-predictive stepping, extrapolated from the last *real* point. Predicting
from the previous prediction compounds the slope and kills tracks a few columns into any
dash gap — that bug cost a full debugging round and is called out in `dashtrace.py`.

### Result: the four siblings agree

Green record (5213 pooled, its three curves overlap along the whole plot):

| Stock | σ(dmin) | σ(1.0) | σ(dmax) | toe/mid | dmax/mid | peak |
|---|---|---|---|---|---|---|
| 5203 | 1.97 | 5.01 | 3.16 | 0.393 | 0.631 | 1.32× at D 0.80 |
| 5207 | 4.95 | 8.35 | 4.71 | 0.593 | 0.565 | 1.30× at D 0.78 |
| 5213 | 3.04 | 7.48 | 4.37 | 0.406 | 0.584 | 1.24× at D 0.77 |
| 5219 | 7.11 | 10.60 | 5.84 | 0.671 | 0.551 | 1.32× at D 0.79 |

dmax/mid spans 0.551–0.631, ±7 % about 0.583, across four independent sheets — against
batch 14's 2.56 / 0.70 / 0.67. Adopted as 0.39/0.59/0.41/0.67 toe and 0.63/0.57/0.58/0.55
dmax. Stable over ink threshold 0.40–0.50 (ratios move ≤ 0.002); 0.60 derails 5207, so the
operating window is stated rather than assumed.

Identity validated independently: the twelve density plateaus on these plots match the [T1]
H&D ladders to a mean **+0.051 D, positive on every layer of every sheet** — a systematic
offset, which is the signature of the different densitometer the sheets' own footnote warns
about. A random pattern would have indicated a swapped track.

### The premise used to reject batch 14 was wrong, and that is the durable lesson

Batch 14 rejected its numbers because "for a colour negative σ_D should RISE with density".
It does not. Kodak's own **SMPTE Journal, July 1985**, Sehlin/Kennel, "Choosing between ECN
5247 and 5294", p 728 — already in the archive — prints Fig. 8 with the granularity curve
falling monotonically as the density curve rises, Fig. 9 with five falling curves, and the
sentence "overexposing either film significantly decreases granularity". Method rule 10 has
been amended accordingly: check siblings against physics, but check the physics against a
*source*. An unsourced expectation is an assumption, and rule 14 makes it lose to a vendor
document — whether it is being used to accept a result or to reject one.

### Left undone on purpose

* `_grain_v2` still fills 0.4/1.0/**1.2** for 103 non-reversal stocks. The sign is now known
  to be wrong for the colour negatives among them, and it was **not** flipped: the approval
  covered four stocks, and every source is a *chromogenic* negative while that branch also
  fills B&W silver negatives, where σ ∝ √D is the textbook result. Queued in §3 with the
  blocker named — a measured σ(D) for a B&W silver negative.
* The three-anchor schema cannot carry the measured interior peak (D ≈ 0.78, 1.24–1.32× mid).
  Recorded with every value rather than smoothed away.
* Absolute σ is **not** adopted. At D = dmin+1.0 the traced green σ×1000 is 4.55/6.32/5.60/
  7.16 against the stored `rms_granularity` 2.6/4.2/4.6/6.6. The discrepancy narrows
  monotonically as grain coarsens, which is consistent with an additive floor, but no stored
  rms figure was moved and no average was taken. Only the shape is adopted, and ratios are
  immune to any multiplicative error in the σ axis.
* `dashtrace.py`'s claim that a log-calibration error "cancels exactly" in the ratios is
  **corrected**: the axis offset cancels, px-per-decade does not, so px-per-decade is now
  measured per sheet (139.00 / 139.75 / 139.00 / 140.25).


---

## Archived 2026-08-31 — the previous §0 "BLOCKER REVIEW" of DIGITIZATION_QUEUE.md

Moved here whole when §0 was rewritten as a derived snapshot. ⚠ **It is archived rather than
deleted because most of it was ABOUT ITS OWN PAST ERRORS** — three successive miscounts, a
"nothing remaining can be worked" claim made while three workable rows sat in §3, and the
acquisition label that turned out to be wrong on seven of thirteen rows. The corrections are
carried forward into the new §0.5 as rules; the archaeology is here.

## 0. BLOCKER REVIEW — what actually stands between this queue and empty

**Added 2026-08-30, after the M-group batch.** ⚠ **This section is a REVIEW, not a status
line: it exists because "13 items open" says nothing about whether any of them can be
worked on today.** Sorted by what is in the way, not by importance.

**24 rows are open**, by the row parse §3 defines as authoritative (78 struck, 24 live).
2026-08-31 closed C38, K2, K3, **E2**, **D3**, **B3** and **E3**, and advanced **M1** without
closing it.

⚠ **THIS SECTION TRIAGES ONLY THE ROWS IT NAMES, AND SAYING SO IS OVERDUE.** It carried
"Ten rows are open" on 2026-08-30 while the parse said 33, and on 2026-08-31 that sentence was
"corrected" to **"Seven rows are open … nothing remaining here can be worked by tracing harder"**
— which was **wrong twice over**: it re-counted only §0.1–§0.3's own tables, and it asserted the
file was out of workable items when **E2, E4 and F3 were sitting in §3 blocked on nothing**. The
same defect §3's header records for the old category table — *"an item could be opened, filed
nowhere, and vanish from the only summary anyone reads"* — reappeared here in a subsection whose
whole purpose is triage. §0 is a REVIEW of named rows; **§3's parse is the census**, and any count
in this section must come from it.

Where the 24 stand, **recounted 2026-08-31 after the corpus reconciliation**: **5 blocked on
nothing but work** (§0.1 — E4, F3, T1, T2, T3), **6 on an owner decision**, **3 on a small owner
action** (D1 one free scan, D2 one ~$40 wedge, G5 one re-scan), **1 on G1 approval**, **2 on
method**, **1 on configuration** (M1), and **6 on acquisition**.

⚠ **THAT LINE READ "13 ON ACQUISITION" THIS MORNING AND THE NUMBER WAS NEARLY DOUBLE THE TRUTH.**
The reconciliation checked every live row against the owner's real corpus — **475 PDFs, twenty
directories**, against the **56** in this working copy — and found that seven of the thirteen were
not acquisitions at all: B3 and E3 had their documents and are now closed; T1, T2 and T3 named the
wrong publication codes for films that are held; G5 is an owner action by its own description; and
M1 stopped being an acquisition when the 2383 sheet arrived. Only **C14, F1, F2b, G6, K5 and K6**
survive, each re-proved absent by searching the full corpus rather than this checkout. Details in
§0.4.

⚠ **D1 STILL COSTS ONE MINUTE AND IS STILL THE BEST-VALUE OWNER ACTION** — with a correction made the
same day it was written. The 2026-08-31 scan review found the film base present and UNCLIPPED in 50
frames (0.0082 D ± 0.0051 against scanner white), and this paragraph first claimed an empty-gate scan
would make those 50 absolute **retroactively**. ⚠ **It will not.** The strip of one physical film base
ranges 235.8–252.9 across those same 50 frames, which a single piece of base cannot do — the UF15 is
re-exposing per frame, and a batch with no single white point cannot be calibrated by a later gate
frame. The gate frame is worth taking **in the session whose scans are meant to be absolute**, and its
first job is to settle whether this scanner offers a fixed exposure at all.

⚠ **AND "ACQUISITION" IS NOW A SUSPECT LABEL ON THIS LIST.** M1 was filed there and its document was
in the owner's own corpus the whole time — this checkout holds **38 KODAK sheets against 202** on the
owner's machine, and whole directories (`KONICA`, `SOVIET STANDARDS`, `ILFORD`, `SVEMA`, `TASMA`,
`ORWO`) are absent here and present there. Every one of the five audits that reports
`source not present` has its source there too. ⚠ **C8's blocker is stated as "`PDF/PROFILES/KONICA/*`
and `PDF/PROFILES/SOVIET STANDARDS/*` are not present in the corpus this pass could read" — both are
present on the owner's machine.** Before working any acquisition row, check the owner's corpus, not
this checkout.

### 0.1 Blocked on NOTHING — two rows left

⚠ **It was never empty.** B4, C39 and T4b were the three rows §0 had *triaged*; E2, E4 and F3 sit
in §3 with the blocker cell "nothing" and were never carried up here. Listed now.

| item | what it is, and why nothing is in the way |
|---|---|
| ~~E2~~ | ✅ closed 2026-08-31. ⚠ **The row's headline warning pointed the wrong way**: it prescribed negating these curves, and negating them is what would have produced the mirrored set it warned about. Two new pan sets adopted, two confirmed |
| **E4** | Eastman 1942 motion-picture book — Super-XX 1938 and the Plus-X 5231 predecessor. Medium effort, low–medium value. ⚠ **Stage it first**: `Kodak - [1942] - Eastman Motion Picture Films for Professional Use.pdf` is on the owner's machine, not in this checkout |
| **F3** | 5 nm spectral re-trace beyond ACROS. Blocked on nothing and **honest about being near-worthless**: measured benefit 0.4–1.1 %, high effort. Future-proofing, described as such |
| ~~B4~~ | ✅ closed 2026-08-30. ⚠ **The blocker did not exist**: two bugs in our own reading — a gridline fainter than its neighbours, and an embedded raster stored upside down behind the page's flip transform. Mask and dye pair adopted; MTF **refused** as a category error |
| ~~C39~~ | ✅ closed 2026-08-30. The only open row with a live render defect. Fixed by a `TakingFilter` carrier, not by a threshold |
| ~~T4b~~ | ✅ closed 2026-08-30. ⚠ **The blocker was one legend nobody had opened** |

⚠ **Two of those three blockers were never real, and both had been written down as though they
were.** That is the lesson worth carrying into the rows below: a queued cause is a HYPOTHESIS
recorded at the moment work stopped, and it ages worse than the data does. Re-test the blocker
before accepting it — three of the last five closures went that way (ISO 5-3 already in the corpus,
a "defective trace" that was exact, and now these two).

### 0.2 Blocked on an OWNER DECISION — no acquisition, no research

✅ **ALL THREE CLOSED 2026-08-31.**

| item | the decision, and what it turned out to be |
|---|---|
| ~~C38~~ | ✅ closed. ⚠ **Two of the three "disagreements" were ours, not the data's.** 5218 was the audit reading the four-page BROCHURE against a set cited to the technical data sheet; 5231 was the reader pairing a density-criterion caption with the wrong curve. Only 5245's blue tail was a real defect, and it was replaced |
| ~~K2~~ | ✅ closed. `AimDensity` (schema v21): a list of per-EI records holding RANGES, not midpoints. 13 stocks populated from 16 printed tables |
| ~~K3~~ | ✅ closed. `ProcessVariant` **widened** rather than duplicated — a `push_stops` discriminator turns it into "same emulsion, different processing condition". 2 stocks, 3 push sets |

⚠ **The §0.1 pattern repeated exactly.** Two of C38's three blockers did not exist, and both had
been written down as facts. Five of the last eight closures have gone that way. A queued cause is
a HYPOTHESIS recorded at the moment work stopped; re-test it before accepting it.

⚠ **And one of the three was a decision already made wrongly by default.** K3 was filed as "no
carrier exists"; the honest answer was that one existed and had been documented too narrowly.
`ProcessVariant` already held an alternate full response with its own EI and source. Widening a
record beats adding a fourth carrier with the same five fields — and the discriminator field is
what keeps the two meanings apart.

### 0.3 Blocked on ACQUISITION — a specific document this corpus does not hold

| item | the document |
|---|---|
| **M1** | ⚠ one **spectral sensitivity curve set for a colour PRINT stock** (Kodak 2383 or 5383), or a scanner's channel responses |
| **F2b** | one granularity-vs-density plot for a named **B&W NEGATIVE** at a stated aperture |
| **K5** | rms granularity for eight KODAK still stocks — ⚠ *provably not obtainable* from the thirteen sheets held, which all publish Print Grain Index instead |
| **K6** | PORTRA 100T's own sensitometry — ⚠ E-2468 contains none: its CURVES page is PORTRA 160VC's artwork |
| **C14** | a Kodak publication for **EKTAR 125** (1989-1994). ⚠ Swept 2026-08-31: absent from all 475 files on the owner's machine; the only EKTAR 125 document anywhere is the 1989 *PHOTOgraphic* magazine review, which prints no sensitometry |
| **F1** | Bayer JOSA 54 (1964), Wilder JOSA 62 (1972), Trabka JOSA 63 (1973). ⚠ Swept 2026-08-31: no JOSA paper of any kind is in the owner's corpus |
| **G6** | one Agfa-Gevaert MTF sheet whose "lines/mm" axis can be cross-checked against a resolving power. ⚠ Swept 2026-08-31 and it is worse than it looked: `agfa_films.pdf`, `AGFA stocks.pdf` and `FPD1e.pdf` are the SAME publication as `Datasheet_F_PF_E4.pdf` -- two of the four are byte-identical to each other -- so E1's self-inconsistent sheet is the only Agfa MTF source there is, and all four GEVAERT papers are 100 % raster with no text layer at all |

⚠ **T1, T2 and T3 ARE NO LONGER IN THIS TABLE.** The corpus sweep of 2026-08-31 found that all
three name the wrong publications and that the documents they actually want are on the owner's
machine. See their rows in §5.

⚠ **M1 is the highest-value acquisition in the project and the smallest.** One curve set
turns nine already-derived, already-validated dye matrices from stored-and-refused into
rendering — the difference between every colour stock sharing one symmetric crosstalk
shape and each carrying its own measured signature. Everything else for it is built.

⚠ **K5 and K6 are the honest kind of blocked**: it has been *proved* that the documents on
file cannot answer them. They are not waiting on more effort. ⚠ **AND K5's proof got stronger on
2026-08-31, not weaker.** The row said "not obtainable from any of the thirteen"; the sweep tested
**all 201 KODAK files on the owner's machine** for the string "diffuse rms granularity" and found it
in **more than eighty** of them -- every cine stock, most of the EKTACHROME reversals, several B&W
sheets, and the 1996-1998 E-55 / E-88 reversal editions that predate Print Grain Index. **Not one of
them is a PORTRA, GOLD or ULTRA MAX colour negative still film.** The eight stocks K5 names are
exactly the population Kodak switched to Print Grain Index and never published an rms figure for.

### 0.4 What is NOT a blocker, and used to look like one

⚠ **RECONCILIATION, 2026-08-31 — every live row checked against the owner's REAL corpus rather
than this checkout, and the checkout is the smaller thing by an order of magnitude.** This working
copy holds **56 PDFs under `PDF/PROFILES`**; the owner's machine holds **475**, in twenty
directories, of which this checkout has six. That single fact was behind **five** of the thirteen
rows filed under acquisition:

| row | what it recorded | what the sweep found |
|---|---|---|
| **B3** | "P-255 and F-4043, not in this checkout" | both present, plus a second edition of each. **Closed in this batch.** |
| **E3** | "the KONICA files are not openable from this working copy" | all three present, plus sixteen more Konica sheets. **Closed in this batch.** |
| **E4** | filed "ready now, source on disk" | ⚠ the opposite error: the Eastman 1942 book is **not** in this checkout and **is** on the owner's machine. Ready once staged, not ready now. |
| **G2** | "trace the four raster plot sets" | same -- `Rens_vanBets1968Gevachr6.00.pdf` is on the owner's machine, not here. Its blocker (G5, a re-scan) is unchanged. |
| **E5** | listed under *acquisition* in the tier-3 line | ⚠ the paper is on the owner's machine and its FILENAME IS WRONG: `Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf` is **SMPTE Journal, July 1985, p. 724**, presented at the 125th SMPTE conference in 1983 and "received in final form on September 14, 1984". Read off page 1 of a 110 dpi render, because all eleven pages are raster. E0 settled the citation as 1985 in August and the file kept the 1983 name. Its blocker was never acquisition -- it is the axis-unit reconciliation the category table already says it is. |

⚠ **And three rows name the wrong publication entirely** -- see T1, T2 and T3 in §5. The lesson is
the one this section already carries, sharpened: **before filing an acquisition row, check the
OWNER'S corpus, and check the publication code against page 1 of the file rather than the
filename.** Six of this file's rows were wrong about a document, and in every case the document
either existed or was a different film from the one the row named.

- ⚠ **ISO 5-3 was in the corpus the whole time.** `aimm.it2.18.1996.pdf` — the Status A and
  Status M spectral response tables — sat unread while the queue recorded them as missing.
  Before opening an acquisition row, grep the corpus.
- ⚠ **A "defective trace" was not defective.** M2 was queued as "re-trace two panels"; the
  extraction turned out to be exact and the panels' own neutrals were the real evidence.
  Queued causes are hypotheses.
- ⚠ **An AVX2 implementation problem had a data answer.** M3b looked like an
  architecture decision and was closed by a lookup table shared with the reference.
- ⚠ **Two "the two readings disagree" rows were two READERS disagreeing** (C38, 2026-08-31).
  5218's audit was opening the marketing brochure while comparing against a set cited to the
  technical data sheet; 5231's was pairing a density-criterion caption with the wrong curve
  because captions sit above their curves on one Kodak sheet and below them on another. Before
  adjudicating between two readings, check that both are of the thing you think they are.
- ⚠ **"Blocked on METHOD" was blocked on the FRAME** (5248, 2026-08-31). Recorded as needing
  a three-black-curve separator; the separator had worked on five other sheets for a week. The
  reader was taking the first panel frame that CALIBRATED, and on that page it is 40 pt too
  narrow to contain the cyan trace. A recorded diagnosis ages as badly as a recorded blocker.
- ⚠ **A carrier can exist and be documented out of sight** (K3, 2026-08-31). The row said no
  record could hold a push. `ProcessVariant` could, and had been able to since v18; its
  docstring said "a DIFFERENT CHEMISTRY" and nobody re-read the fields. Grep the schema, not
  the prose about it.

### 0.5 The blocker this file cannot list

⚠ **There is no ground-truth harness.** Every audit here checks the database against its
documents; none checks a RENDER against a photograph. So no row below can say how much it
would improve the picture, and the queue cannot be ordered by that. Closing every item
would still leave that unanswered. It needs scanned frames from stocks we model, with the
scanner named — and it is not a digitisation task, which is why it has never appeared in
this queue.

---


---

## Archived 2026-08-31 — the previous §3 dashboard of DIGITIZATION_QUEUE.md

Replaced by a derived table. Kept because it records the three-way count disagreement and the
category table that silently omitted eight rows.

### Where the work stands — recounted 2026-08-26, not estimated

**78 items closed, 24 live — RE-PARSED 2026-08-31 after the corpus reconciliation, and the three numbers
that used to disagree now come from one place.**

⚠ **WHAT WAS WRONG, recorded because it is the same failure this file keeps finding in other
people's work.** Three counts coexisted here and no two agreed: a header sentence (26 live), a
category table that summed to 28, and a row parse that gave 34. None was maintained by the others.
The category table was the worst of the three — it silently omitted **eight real rows** (`B4`,
`C38`, `C39`, `K5`, `K6`, `T1`, `T2`, `T3`, `T4b`), so an item could be opened, filed nowhere, and
vanish from the only summary anyone reads. And **two closed rows were not struck** (`K4`, `T0`):
they carry ✅ in the body but plain `**ID**` in the first cell, so every parser counted them live.
Both are struck now.

**The rule from here: this dashboard is DERIVED. A row is closed when its id is struck (`~~ID~~`);
a ✅ in the body is not enough, because prose is not a field.** 102 rows total, 78 closed, 24 live
at the 2026-08-31 parse — B3 and E3 closed with this batch.

| Category | Live | What it is | Who unblocks it |
|---|---|---|---|
| **Owner decision** | **6** — C4, C7, C16, C18, C19, C2c | zero research effort, already investigated | the owner, in one sentence each |
| **Owner action, free or ~$40** | **3** — D1, D2, G5 | one empty-gate scan, one wedge scan, one 300+ ppi re-scan of three printed pages. ⚠ D3 closed 2026-08-31 once the frames arrived. ⚠ G5 moved here 2026-08-31: it was filed under acquisition, but the row's own text says "the owner holds the source" — it is an owner action, like D1 and D2 | the owner, minutes |
| **Ready now, source verified present** | **5** — E4, **F3**, T1, T2, T3 | trace-and-adopt. ⚠ **RECLASSIFIED 2026-08-31 by the corpus sweep**: E4's source is on the owner's machine and not in this checkout (stage it first); F3 needs no source at all; T1, T2 and T3 named the wrong publications and the ones they want are held — see their rows | nobody — it is just work, once the file is staged |
| **Ready now, KODAK still-film follow-ups** | **0** | ⚠ K1 closed 2026-08-30 (four PORTRA NC/VC profiles), K2 closed 2026-08-31 (`AimDensity`, 13 stocks) | — |
| **Ready, Gevaert group** | **1** — G2 | raster plot sets | G5 (a 300+ ppi re-scan) |
| **Acquisition, proved absent from all 475 files** | **6** — C14, F1, **F2b**, G6, K5, K6 | ⚠ **RE-PROVED 2026-08-31 against the owner's machine, not this checkout.** Each was searched for by publication code, film name and figure caption across the full corpus | the owner or an archive |
| **Blocked on configuration, not acquisition** | **1** — **M1** | the 2383 spectral set arrived 2026-08-31; what is missing now is a profile that renders through a print stock, or a scanner's channel responses | a decision, not a document |
| **Blocked on method / schema** | **2** — C23, **E5** | needs a model or a decided form first | see each row |
| **Adjudication, evidence in hand** | **0** | ⚠ B4 closed 2026-08-30, C38 closed 2026-08-31 | — |

Live total: 6 + 3 + 5 + 0 + 1 + 6 + 1 + 2 + 0 = **24**, and the row parse agrees — **B3 and E3 closed
2026-08-31**, and the reconciliation moved G5 out of acquisition and M1 into its own line without
changing any total. ⚠ If that sum stops
matching the parse, the table is wrong and the parse is right — re-derive it rather than patching
the sentence.

⚠ **AND IT HAD STOPPED MATCHING AGAIN, IN EXACTLY THE WAY THE HEADER ABOVE WARNS ABOUT.** Three
errors, all of the same kind — a row filed nowhere, or filed under a closed id: **`F2` was listed
live and has been closed since 2026-08-30** (its successor `F2b` is a different, acquisition-blocked
row); **`M1` and `F2b` appeared in no category at all**, exactly the "opened, filed nowhere,
vanishes from the only summary anyone reads" failure this table was rebuilt to stop; and **`F3` was
filed under method when its own blocker cell reads "nothing"** while **`E5` was filed under
acquisition when its source is in the archive and its blocker is a unit reconciliation**. Fixed
here, and the §0 header sentence that inherited the same undercount is fixed with it.

⚠ **TWO CATEGORIES ARE EMPTY, AND ONE WAS THE CHEAPEST IN THE FILE.** "Adjudication, evidence in
hand" and "Ready now, KODAK still-film follow-ups" are both at zero. The remaining 28 are dominated
by the one category nobody here can drain: **13 want a document this corpus does not hold**, and
a further 8 want an owner decision or a small owner action. **Two — E4, F3 — are workable today**,
and F3 says of itself that it is worth 0.4–1.1 %.

⚠ **BUT SEE §0: "does not hold" means THIS CHECKOUT.** M1's document was in the owner's corpus all
along, and so are the sources of all five SKIPping audits. The acquisition column is 13 rows only
until somebody looks.

⚠ **AND THE "READY NOW" COLUMN SHRANK ON 2026-08-26 WHEN IT WAS ACTUALLY WORKED.** The B section was
labelled "source on disk, path proven, no dependency" on 2026-08-18. Worked through in full: of four
sub-items, **one was adoptable** (5248, and only after a schema change), one is refused with its
alternatives excluded by measurement (5246), one pointed at **a page with no plot on it** (5247 p4 is
a plate index), and one's sources **are not in this checkout** (B3: P-255 and F-4043). The lesson is
the one this file keeps relearning: a readiness label decays, and the only way to know is to open the
document.

⚠ **THE SHAPE OF THE QUEUE HAS CHANGED, AND IT IS WORTH SAYING PLAINLY.** The 2026-08-18 ranking
was written when most items were *research* — find a document, read a plot. Nine days later the
research half is largely done and **the binding constraint is now the owner's decision queue**:
six items, all investigated, all one-sentence answers, several gating renders on dozens of stocks.
Nothing on the research list gates as much as C18 alone (36 stocks) or C16 (half the largest colour
effect in the chain).

⚠ **AND THE SECOND-BIGGEST CATEGORY IS "JUST WORK WITH NOBODY IN THE WAY"** — seven items, sources
on disk, paths proven. C37 alone is 15 sensitivity panels that became reachable on 2026-08-25 and
have not been read.


---

## Archived 2026-08-31 — the previous "RECOMMENDED ORDER, 2026-08-26"

Superseded: thirteen of the rows it ranks have since closed.

### RECOMMENDED ORDER, 2026-08-26 — derived from dependencies, not from the value column

⚠ **Two things fell out of re-reading every live row, and both change the ranking.**

**FINDING 1 — `F2` HAS BEEN UNBLOCKED FOR EIGHT DAYS AND NOBODY NOTICED.** Its blocker reads
"**depends on C1**, and inert until then". **C1 closed on 2026-08-18.** Same class of staleness as
E0b-orig, and this one is worse, because F2's premise has also grown: the row says "the 103-stock
default" and the live count is **147 stocks** sitting on a σ(D) heuristic against 13 measured.
⚠ **And 34 of those 147 carry a heuristic this project has MEASURED to be wrong in direction** —
the reversal default 0.7 / 1.00 / 0.50 falls toward dmax, while the one measured B&W reversal σ(D)
(C29, TRI-X Reversal) RISES. `verify.py` already asserts that contradiction and cites it. So F2 is
not a tidying item; it is the largest population of knowingly-wrong numbers in the database.

**FINDING 2 — `C2c` AND THE MTF HALF OF `C19` ARE ONE PIECE OF WORK, and its evidence base has
tripled.** Both are the same defect: `adjacency_um` disagrees with the measured overshoot frequency,
in the same direction, on every stock checked. C19 was written when **four** stocks had a traced
MTF. Twelve do now. C19 also states the real problem plainly — `adjacency`, `edge_strength`,
`edge_um` and `radius_um` are **four parameters describing ONE inhibitor diffusion length, currently
set independently** across 87 hand-typed `CouplerSpec` literals with no registry and no tier. The
MTF half needs nothing from anybody.

#### Tier 0 — owner, minutes to days, and it unblocks the most

| Order | Item | Why first |
|---|---|---|
| **1** | **D1** + **D2** | One free frame and one ~$40 wedge scan. Together they are the stated blocker for the STRENGTH half of both **C18** (36 reversal stocks, "the biggest single unpinned colour number") and **C19**. Nothing else on this list unblocks two high-value items for two days' pocket money. |
| **2** | **C7** | One product question — is the plugin judged frame-by-frame or in motion? Zero research: the source (Honjo 1989 §4) is read and the number is 1/√5 = 0.447. High perceptual impact on every frame of every render. |
| **3** | **G5** | One re-scan of three printed pages you already hold. Unblocks **G2** (high) and upgrades two profiles from `[T3]` estimates to traced curves. |

#### Tier 1 — me, no decision needed, in this order

| Order | Item | Why here |
|---|---|---|
| **4** | ~~C37~~ | ✅ closed 2026-08-29 as a registered audit. It yielded **no new sets** — the promise of 13 was wrong — but turned a hand sweep into 11 enforced cross-checks and surfaced **C38**. |
| **5** | **F2** | See Finding 1. 147 stocks, unblocked since 18 Aug, and 34 of them measurably wrong in direction. The 13 measured shapes are now enough to derive a class heuristic per stock KIND rather than one global triple. |
| **6** | **C2c + C19 (MTF half)** | See Finding 2. Twelve traced MTF curves make this measurable now; collapse the four independent parameters onto the one diffusion length they all describe, and re-derive `adjacency_um` from the traced overshoot. |
| **7** | **B4** | ~~B1~~ closed 2026-08-26; its untracked residue is now B4 — the four TI0835 raster plates for `EASTMAN_5247_1983`, all colour-coded, two of them new data. Blocked only on axis calibration. |

#### Tier 2 — me, moderate effort, nothing blocking

**E4** · **T1** · **T2** · **T3** · **G2** (after G5) — ~~B3~~ and ~~E3~~ closed 2026-08-31,
~~E2~~ closed 2026-08-31, ~~G4~~ and ~~E1~~ both closed 2026-08-29. ⚠ E4, T1, T2 and T3 all need
their source STAGED from the owner's machine first; none of the four files is in this checkout.

#### Tier 3 — waits on a decision or a document

**C16** (one number, but it changes every render — yours) · **C18 / C19 strength** (after D2) ·
**G6**, **C14**, **F1** (acquisition — all three re-proved absent 2026-08-31 against the owner's
full 475-file corpus) · **E5** (⚠ NOT acquisition: its paper is held, misfiled under a 1983 name,
and its blocker is the axis-unit reconciliation) · **C4** (yes/no, low value) · ~~D3~~ (closed) ·
**C23** and **F3** last — C23 is the least evidenced item on the list and F3's own measured benefit
is 0.4–1.1 %.

⚠ **Two standing decisions are not rows in this table and should be**, because both are held on the
owner and both are wider than any single item: the **spectral criterion on 16 profiles** (a "D 0.2
above dmin" printed on no sheet in this corpus, while 5222 and 7239 both print theirs), and the
**MTF rolloff architecture in C++** (no FFT exists there, so `FilmMtfResponse` has no stage caller).

---

**Ordering, set 2026-08-18 at the owner's direction and re-checked 2026-08-26, and NOT the order
these items were written in.** Items are ranked on: effort against value, dependencies first,
whether the source material is actually on disk, technical impact, and rising complexity. The
ranking is re-derivable — every claim about source availability below was checked, not assumed.

| # | Item | Blocked on | Effort | Value |
|---|---|---|---|---|
| **A. CLOSED 2026-08-18 — kept here for the audit trail, not live work.** ⚠ These eleven rows sat inside the OPEN section for eight days; struck 2026-08-26 so a reader scanning for live items stops seeing them ||||
| ~~A1~~ | spectral dye density vector/raster audit | — | done | cleared a §3 blocker |
| ~~A2~~ | MTF vector page count (119 / "~156" → **199**) | — | done | two docs corrected |
| ~~A3~~ | B&W silver σ(D), Mees Fig. 302 | — | done | premise disproved |
| ~~A4~~ | `EASTMAN_5247` split into `_1974` `[T3]` reconstruction + `_1983` (EI 125, owns the TI0835 plate) | — | done | mislabelled generation corrected |
| ~~A5~~ | 6 dye-density sets adopted (5205, 5219, 5245, 5274, 5279, 5293), `normalisation="peak_1.0"` | — | done | first colour dye data in the DB |
| ~~A6~~ | `NotFound.md` full rebuild — 13 per-film research blocks, 9 fields each | — | done | acquisition guide usable |
| ~~A7~~ | generated C++ split into 16 data slots + explicit `LoadFilmDataBase()` | — | done | **unblocked the build** — 676 KB single function → compiles in VS2015 SP3 |
| ~~A8~~ | **provenance placeholder closure** — 6 of 8 placeholder-only profiles cited, 5 `verify.py` guards added, fault-injection tested; then **7th and 8th** the same day when the owner supplied `FUJI/52_509.pdf` | — | done | tier claims now match `provenance.sources` **with zero exceptions**; the allowlist is empty and guard 3 caught the change |
| ~~A9~~ | **two new FUJI documents harvested** — `52_509.pdf` (Honjo 1989) and the Super F-125 8532 sheet. `Index.md` updated on disk | — | done | F-125 8530/8630 `f50` corrected **78 → 42 c/mm** (1.86×, the largest sharpness correction in the file); a complete `[T1]` sheet for the 8532 successor is now on file and read |
| ~~A10~~ | **C3** — `SVEMA_FOTO_32` / `_130` tint + silver_tone withdrawn | — | done | two profiles stop claiming a scanner artefact as an emulsion property |
| ~~A11~~ | **B2** — `AGFA_VISTA_200` spectral sensitivity, via the dash-pattern legend | — | done | see B2 below |
| **B. WAS "Ready now — source on disk, path proven, no dependency". ⚠ THAT LABEL WAS WRONG FOR MOST OF THIS SECTION AND IS CORRECTED 2026-08-26.** Written 2026-08-18; of the four sub-items it contained, ONE was adoptable, one is refused with the alternatives excluded by measurement, one pointed at a page with no plot on it, and one's sources are not in this checkout. Worked through in full on 2026-08-26 -- see the rows ||||
| ~~B1~~ | ✅ **WORKED THROUGH IN FULL 2026-08-26 — one adoption, three corrections.** The row read "remainder: 2 sheets (5246, 5248) + `EASTMAN_5247_1983` p4 visual pass" and every part of that needed fixing. **(1) 5248 — ADOPTED, and the SCHEMA was the blocker, not the sheet.** H-1-7248 p3 prints, in words, "Typical densities for a midscale neutral subject and D-min." and draws exactly TWO traces. `SpectralDyeDensity` required cyan AND magenta AND yellow, so a clean two-trace panel could never be stored -- a real published measurement discarded for having the wrong SHAPE. **Schema v14** adds `d_dmin` and a second legal shape; `has_data` deliberately keeps its old meaning so no count moves, and `has_neutral_pair` reports the new one. Traced from 206 and 230 path vertices, axis residuals 1.02 nm / 0.0137 D. ⚠ **Two physical checks, neither fitted:** the neutral exceeds the D-min at all 31 samples by ≥ 0.463 D (it must -- a neutral is mask plus image dyes), and the D-min behaves as an orange mask must, peaking 1.011 in the BLUE at 440 nm and falling monotonically to 0.169 at 700. **(2) 5246 — REFUSED, and the alternatives are now excluded BY MEASUREMENT.** Not a tolerance problem; not a label-matching problem (the 7239/5222 caption matcher was pointed at it and fails -- the labels sit in WHITESPACE, "Cyan" at 558 nm where the cyan dye is 0.37 with four traces within 0.2 D); and not two products on one plate, which was the attractive explanation given the header names 5246 AND 7246 -- no traces pair off, closest pair sd 0.103 D over a 0.330 D range, next closest a CROSSING. ⚠ **SEVEN solid traces coexist at 480-580 nm against FIVE legend entries**, and family C's identity fails everywhere tried (best k spread 136 % against 5201's 5.4 %). The sheet needs a statement of what its two extra traces are; no amount of tracing supplies that. **(3) `5247_1983` p4 — THERE IS NO PLOT ON THAT PAGE.** p4 is a plate INDEX ("15) Graphs — MTF a) (6-83), Characteristic b) (6-83), Spectral Sensitivity…"); the row was matching a line of contents. The actual plates are on **p6-9 and they are RASTERS**, so reading them is new raster work, not "a visual pass". ⚠ **THAT SENTENCE WAS THE ONLY RECORD OF FOUR UNREAD PLATES and nothing tracked it; reconnoitred and promoted to its own row, B4, on 2026-08-29** | — | done | one adoption, one schema shape fixed, three row errors corrected |
| ~~B2~~ | ✅ **DONE 2026-08-18.** `AGFA_VISTA_200` spectral sensitivity adopted; new `agfa_vista.py`, registered in the audit stage. The item's own description was wrong and that is why it stalled — see A9 | — | done | 1 stock gains a measured spectral set; 1 latitude value corrected |
| ~~B3~~ | ✅ **CLOSED 2026-08-31. Both documents were on the owner's machine, in two editions each, and the row's own diagnosis of the WORK was half wrong too.** `KODAK_TECHNICAL_PAN` genuinely had no spectral set and now has one: P-255 p9, "Spectral-Sensitivity Curves", vector paths, the D=0.3-above-D-min curve of a printed pair, 31 samples, absolute peak log sensitivity 1.03. ⚠ **It is one of the two flattest panchromatic curves in the database** — 0.56 decades across 380–680 nm against a field median of 1.12 — which is the trace agreeing with P-255's own prose, *"reasonably uniform spectral sensitivity at all visible wavelengths out to 690 nanometres"*. ⚠ **And 380 nm is the GRID EDGE, not the peak**: the printed panel runs to 250 nm and is still climbing where the stored grid starts, consistent with the panel's own exposure note of 1.4 s visible against 0.2 s ultraviolet. ⚠ **`KODAK_TMAX_400` did NOT "want its second criterion" in the sense the row meant.** Its set was adopted 2026-08-16 from F-4043 (2016) and `NotFound.md` had already recorded that as closed; the sheet's second criterion is a *different measurement of the same emulsion*, and the schema holds one set per stock. What it actually wanted, and now has, is a **cross-edition validation**: the 2007 edition of F-4043 is a file the profile has never read and it reproduces the stored set to **rms 0.005 decades** — closer than the 2016 edition now does. ⚠ **The reader needed two fixes and both were single assumptions.** Its caption test was `"ABOVE" in txt and "=" in txt`, which fits H-1-5222 and matches NEITHER new sheet: F-4043 prints "D=0.3 greater than D-min" and P-255 splits its caption across two text lines. And it committed to the first frame that calibrated, so F-4043 (2007) p11 — whose first calibrating frame yields one curve where the caption pair needs two — failed outright. ⚠ **The second criterion is now MEASURED on every mono sheet, not merely named**, which is a check the single-curve reader could not make: the gap between the two curves is the log-exposure interval between the two densities, so its sign must follow the criteria and it must be wavelength-independent. Measured: 5222 −0.992±0.064, 5231 +1.068±0.088, T-MAX 400 +1.059±0.078, Technical Pan +0.408±0.036. Swapping a pair — the exact C38 failure — flips the sign. Full record: `RESULT_2026-08-31c_reconcile_B3_E3.md` | — | done | **one stock's first spectral set, one stock's first independent validation** |
| **C. Blocked on an OWNER DECISION — zero research effort, gates a lot** ||||
| ~~C1~~ | ✅ **DONE 2026-08-18.** Carrier chosen **by measurement**: 3 anchors + an interior peak (5 floats) — scored against 7 measured σ(D) samples per sheet, it gives 20 % max / 8.6 % rms against the legacy law's 245 % / 127 %; a 12-sample array scores better on paper and is over-parameterised against 7 points. **Wired into `film_sim.py` and mirrored in the generated C++** (`FilmGrainSigma`, cross-checked to 5.4e-07). Honoured for the **5 vendor-traced stocks only**, via a new `sigma_shape_measured` flag — the 137 heuristic shapes stay inert because both branches are wrong in sign. 150 stocks verified **bit-for-bit unchanged**. Schema v7→v8. `RESULT_2026-08-18e_C1_sigma_wiring.md` | — | done | **the engine was ~3× too grainy at dmax and ~1.6× too quiet at the real σ maximum, on the 5 stocks that now know better** |
| ~~C1c~~ | ✅ **DONE 2026-08-18 — the σ(D) harvest is CLOSED.** Every granularity plot in the corpus that is VECTOR art is now read: **8 sheets, all 8 green** under `granularity_vector.py --assert`. Six colour negatives adopted (`5245`, `5246`, `5248`, `5274`, `5279`, `5218`) with the full five-float carrier; the 7th (`5285`) was already in; the 8th (the VISION3 500T brochure) is a **cross-check** of a shape traced earlier from a *raster* plot by a different extractor — shape confirmed (dmax/mid 0.55 vs 0.57, peak 1.32× @ 0.79 vs 1.24× @ 0.76), absolute σ conflicts by a uniform 1.3× and the conflict is **recorded, not averaged**. Six extractor defects found, every one of which had been producing plausible output. `RESULT_2026-08-18f_C1c_sigma_harvest.md` | — | done | **6 stocks: 2.1–2.6× more shadow grain, 2–4× less highlight grain, level unchanged at D = 1.0** |
| ~~C1b~~ | ✅ **DONE 2026-08-18, and the scope was 4x smaller than this entry claimed.** The entry said "up to +30 % grain on every stock"; that assumed `rms_granularity` meant σ at **absolute** D 1.0. Kodak prints the convention — "Read at a **net** diffuse visual density of 1.0" (5248 p1, 5222 p1) — and at net 1.0 the legacy law is `sqrt(1+fog)`, **dmin cancels**. Real cost: a uniform **4–8 % drop**, identical in all three channels. `grain_sigma()` and `FilmGrainSigma()` now normalise at `dmin + 1.0`; schema v8→**v9** (a meaning change with no layout change — v9 data + v8 sampler renders wrong and compiles clean). The rms-audit prerequisite was done: **one** stock family was render-fitted (Svema Foto ×4), preserved exactly by policy (a). `RESULT_2026-08-18g_C1b_C1d_level.md` | — | done | **the stored rms now means the printed figure, at the density the manufacturer measured it** |
| ~~C1d~~ | ✅ **DONE 2026-08-18.** Six vector-traced negatives re-levelled from their own curves at net 1.0: **0.91–1.28×** (median 1.05) — the tier-3 family ladder was roughly right, and my "1.3–1.6× understated" claim was the same absolute-vs-net error. The substantive change is per-layer: measured **blue is 1.9–2.8× green** where the schema's tier-2 ladder assumed 1.3×. ⚠ One conflict on file, not averaged: 5248 prints "Less than 5" at net visual 1.0 and traces to 5.87 green / 4.42 red | — | done | first measured per-layer grain ladder for colour negative |
| **G-group: the GEVAERT folder, reviewed 2026-08-19** ||||
| ~~G1~~ | ✅ **DONE 2026-08-19, owner-approved.** `GEVACHROME_600` and `GEVACHROME_605` added (155 → 157 stocks); printed speeds, per-layer gammas, layer stack and the documented 320 ASA push adopted, granularity/resolving power/Dmax left as tagged `[T3]` estimates because the paper prints none. ⚠ The three layer curves could NOT be separated at 150 ppi (within 1-2 px) — Dmax is a measured lower bound only, toe/shoulder softness is a `[T2]` transfer from GEVACHROME_902. `RESULT_2026-08-19_gevaert.md` | — | done | first Gevaert reversal CAMERA stocks in the database |
| G2 | Trace the four RASTER plot sets in the 1968 Gevachrome paper: p.260 Bild 1a–c **MTF** ("Modulationsübertragung", 6.00 vs 6.05, measured in green/red/blue light), p.262 Bild 2a/2b **spectral sensitivity**, Bild 4 **dye absorption**, p.264 Bilder 5a/5b **characteristic curves** and Bilder 7a/7b **interimage effect** (a neutral wedge through a red filter vs a red-filter-only exposure — a direct measurement of the DIR/interimage behaviour `CouplerSpec` exists for). ⚠ Every page is 100 % raster: measured 1 embedded image and 0 curve paths per page, so this needs the `vision3_granularity`-style raster path, not the vector reader. ⚠ **Two notes from 2026-08-31.** The source is on the owner's machine (`GEVAERT/Rens_vanBets1968Gevachr6.00.pdf`), not in this checkout, so staging comes first; and the raster path E3 built for the KONICA scans (`konica_raster.py` — geometric calibration off the printed grid, gridline re-detection, `dashtrace` tracing) is the closest working precedent there now is | G1 approved first (no profile to hold it otherwise) | medium per plot set | high |
| ~~G3~~ | ⚠ **DUPLICATE ROW, STRUCK 2026-08-26 — AND THE DUPLICATE ITSELF IS THE FINDING.** This row said "PARTLY DONE" and listed Figs. 11 MTF / 7 spectral / 8 dye density as remaining; four rows below, a SECOND G3 row said "DONE" and split those remainders out as G6 and G7. Both were written on 2026-08-19 and both were true when written — the file simply grew two entries for one item, and the stale one kept advertising work that had been re-homed. ⚠ Every remainder it names is now closed: **Fig. 11 MTF** adopted as G3 (r 29 / g 44, replacing estimates 46/54), **Fig. 7 spectral** used as a check on the stored set, **Fig. 8 dye density** closed as **G7** on 2026-08-25d. Nothing here is live. The lesson is the cheap one: a queue with two rows for one item is a queue that cannot be counted, which is why the dashboard at the top of §3 is now PARSED from this file rather than carried forward | — | done (duplicate) | — |
| G5 | ⚠ **ACQUISITION ASK, from G1's limits.** A **300+ ppi grayscale re-scan of printed pages 260, 262 and 264** of Kino-Technik 1968 Nr. 10. The current scan is JPEG colour at 150 ppi with bleed-through from the reverse of the sheet, and that is what blocks three things: separating Bild 5a/5b into three layer curves (they sit within 1-2 px), tracing Bild 2a/2b spectral sensitivity, and tracing Bild 1a-c MTF. The owner holds the source | owner, a re-scan | small | **high — it unblocks G2 and upgrades two new profiles from [T3] estimates to traced curves** |
| G6 | ⚠ **NEW 2026-08-19, a UNIT question worth one document.** Gevacolor 682's MTF abscissa is printed "SPATIAL FREQUENCY (lines/mm)"; this database stores **cycles/mm**. If Agfa-Gevaert meant individual lines (half-cycles), the f50 values just adopted (r 29, g 44) are **2x too high**. The paper prints no test-object description and no resolving-power figure to cross-check, and every Kodak sheet in this corpus labels the same axis "cycles/mm", which is why it was adopted that way. **Resolvable by any other Agfa-Gevaert MTF sheet or an Agfa resolving-power figure for 682** — one document settles it for the whole Gevaert family | one Agfa-Gevaert document not currently held | small once a source exists | medium-high — a factor of 2 in sharpness on 4 stocks  ⚠ **NEW EVIDENCE 2026-08-29, from E1, and it RAISES the stakes rather than settling them.** Agfa's own 2004 professional sheet (F-PF-E4 pp6-7) plots "Transfer factor (%)" against a "Lines per mm" axis worded IDENTICALLY to Gevacolor 682's, for four films this corpus holds. Traced f50 on that axis: Portrait 160 **38.4**, Optima 100 **46.2**, Optima 200 **50.8**, Optima 400 **50.3**. ⚠ Those do NOT rank the way the same sheet's PRINTED resolving powers do — Optima 100 is published at 140 lines/mm at 1000:1 against Optima 200's 130, yet reads LOWER here — so the two "lines/mm" figures on one sheet are not even mutually consistent, and the axis is not safely readable as either convention yet. All four f50 values are therefore held OUT of the database (`MTFSpec.f50` stays an estimate on all four) until this is settled. ⚠ What the same panel DID yield is unit-free and is adopted: the curve peaks at **109-114 %**, above what an MTF can be, and that overshoot is the adjacency effect measured directly — `mtf.adjacency` on all four stocks now comes from it. So G6 is now blocking four MEASURED readings, not one |
| ~~G7~~ | ✅ **DONE 2026-08-25d, and the recorded diagnosis was WRONG.** The entry said the cyan curve's DOTTING merged into components that the solid/dashed threshold classes as solid. It does, and it did not matter: the real blocker was that the dotted CYAN and the dash-dot MAGENTA **cross at about 425 nm**, and for roughly twelve pixel columns they are one ink run. `trace_predictive` accepted that merged run into its slope history; merged ink is nearly flat, so the descending track's fitted slope collapsed from +0.75 to +0.3 px/column, and on separation the flattened prediction landed on the wrong branch. ⚠ **THE TWO CURVES CAME BACK SWAPPED WITH EVERY RESIDUAL STILL SMALL** — two smooth curves, just not the two that were printed, and no ordering check could see it. Fixed with `merge_px` in `dashtrace.trace_predictive`: at a crossing NEITHER track may claim ink and both coast on the slope measured before the merge, because refusing to decide is the correct answer when the ink does not say which curve it belongs to. Default 0.0, so no existing caller changes. **Adopted:** 31 samples per dye on the 400-700 nm grid, traced at one sample per pixel column (512 cyan / 393 magenta / 329 yellow). ⚠ **VALIDATED FROM OUTSIDE BY THE PAPER ITSELF** — traced 1.462 D @ 445.9 nm vs printed 1.46 @ 448 (yellow), 1.474 @ 522.1 vs 1.48 @ 525 (magenta), 1.459 @ 683.1 vs 1.46 @ 687 (cyan) — and the figure's own C / M / Y letters sit at 448 / 528 / 683 nm, one above each traced peak. ⚠ Layer names come from the PEAKS, never the seed order: at the left edge the cyan curve is the TOP one. ⚠ The yellow samples from 580 nm are 0.0 by READING, not by trace — the curve reaches the axis at ~583 nm and is thereafter indistinguishable from it. `verify.py`'s guard was inverted in the same edit, from "stays empty" to "is present and reproduces the printed peaks" | — | done | 682 was the only masked negative in the DB with no dye set |
| ~~G3~~ | ✅ **DONE 2026-08-19.** Fig. 10 traced and adopted; Fig. 6 layer order adopted with the six-emulsion double-layer construction recorded; Fig. 11 read visually for f50 (**r 29 / g 44**, replacing estimates 46/54; blue bounded >50 and left as an estimate; **adjacency 0.11 found to have no support in the figure**); Fig. 7 used as a check on the stored spectral set (peaks agree within one 25 nm grid step). Remainders split out as G6 (units) and G7 (dye curve) rather than left inside a "done" item | — | done | one estimated MTF pair replaced by a reading, one carrier populated, two hazards surfaced |
| ~~G4~~ | ✅ **DONE 2026-08-29, and it produced NO DATA — which is the correct outcome, confirmed by reading rather than assumed.** Webers & Westendorp, "Einführung in die Kopierwerktechnik (XIV)", *Fernseh- und Kino-Technik* 33(7) 1979, pp. 245–247 read in full from 200 dpi page renders (⚠ IMAGE-ONLY SCAN, `pdftotext` yields **3 bytes**). **0 profiles added, 0 profiles modified, 0 schema changes, database stays at 161 stocks.** FOUR TYPES ESTABLISHED, verbatim p245: `Gevachrome S – Typ 700` (für Studioaufnahmen), `Gevachrome – Typ 710` (Reportagefilm), `Gevachrome D – Typ 720` (Material für Tageslicht), `Gevachrome Print-Typ 780` (Kopiermaterial). ⚠ **THE SUBSTANTIVE FINDING IS A GENERATION BOUNDARY, NOT A STOCK.** These are a SECOND generation and the process proves it: the corpus's `GEVACHROME_600` / `_605` / `_902` run the FIRST Gevachrome process — **12 steps, two temperature columns 21 °C / 25 °C, first developer GP 110** (Rens & Van Bets 1968 Tab. IV) — while Tabelle VIII here is **15 steps, one 25 °C column, first developer GP 112**. The three stored profiles therefore MUST NOT be relabelled or re-cited as Gevachrome II; the naming similarity is exactly the trap a future editor falls into. ⚠ **AND THIS ROW'S OWN SUGGESTED CARRIER DOES NOT SURVIVE CONTACT WITH THE SCHEMA.** `ProcessingFamily` holds `DevelopmentPoint` rows and a `DevelopmentPoint` wants **gamma against time**; Tabelle VIII prints developer, minutes and celsius for ONE fixed condition and **no gamma, no contrast index and no fog at any time point**, so populating it needs a fabricated number. `ProcessingSpec` is a member of a profile and there is no profile to hang it on — attaching it to the 1968 stocks would assert the falsehood above. `ProcessVariant` needs the same emulsion with printed curves; neither holds. `layer_stack` refused too: Bild 74 draws a full coating order but its caption is `Prinzipieller Aufbau eines Umkehrfarbfilms` — a generic tutorial diagram, not a measurement of any Gevachrome type. **Net effect on the render path: none.** Transcribed complete to `doc/RESULT_2026-08-29_G4_gevachrome_II.md`: all 15 steps with temperature ±tolerance, time, full tank/regenerator formulae, pH and replenishment — first developer GP 112 (hydroquinone 6 g + Phenidon B 0.5 g, pH 10.20, 180 s), light re-exposure **100 000 lx·s white, BOTH SIDES, under the liquid surface of the wash tank**, colour developer CD-1 / `Gevadiamin-C` (N,N-Diäthyl-p-Phenylendiamin, sulfate 3.6 g OR chlorhydrate 2.7 g), pH 10.70, 255 s, ferricyanide bleach, two fixing baths. DERIVED and labelled as such: wet time steps 1–14 = **975 s (16 min 15 s)**, replenishment **1369 ml/m of 16 mm film** of which 91 % is wash water | — | done | four product identities now citable so a future Gevaert sheet naming them is recognised instead of read as unknown, and three stored profiles protected from a mislabelling the name similarity invites |
| ~~C1e~~ | ✅ **DONE 2026-08-23, and the premise this item was written on was WRONG.** C1e existed because C1d suspected the raster extractor's σ axis of reading ~1.32× high. It does not: the 5219 panel's own right-hand tick comb, fitted on the stored calibration, reproduces 0.001–0.100 at **all 15 ticks within 1 %** (5203 within 1.3 %). The 1.32× was an **absolute-D** comparison; at **net 1.0** the two documents differ by 1.12× in green and 1.25× in blue — a real document conflict, a third the size claimed, and not an axis error. So the raster sheets' per-layer RATIOS are usable, and **three** stocks were adopted, not one: **5219 r/g/b = 5.92 / 6.60 / 17.84** (its own sheet separates all three granularity curves: net-1.0 6.43 / 7.16 / 19.37, ratios 0.897 / 1.000 / 2.703, multiplied onto the frozen pooled 6.6), **5207 rms_b 8.92** (b/g 2.123) and **5203 rms_b 4.71** (b/g 1.813). Green untouched on all three, as agreed. The brochure's b/g of 2.425 is recorded as the conflicting second read, **not averaged**. Reds on 5203/5207 left to fall back to the pooled figure — 5203 draws G and R grain as one ink run and 5207's red track starts at log E 2.5, so 1.05 and 0.85 are not measurements to adopt; the pooled 1.00 sits between the two sheets that DO separate red (0.85, 0.90) while the discarded heuristic's 1.10 sits outside both. **5213 stays on the heuristic** and is pinned there by a guard: its three granularity curves are printed as one bold band, so there is no blue track to read. Render impact on 5219: blue grain σ **2.33 → 3.60 per 255 (×1.52)**, red 1.98 → 1.77, green unchanged | — | done | **the top layer was 2.1× too quiet on the flagship stock** |
| ~~C2~~ | ✅ **DONE 2026-08-19, and the carrier this entry assumed LOST the scoring.** The reserved `mtf_tail_a`/`_f_exp` form scored rms 0.0583 against the one traced curve (PLUS-X 5231, 35 samples above 8 c/mm) while **`1/(1+(f/f50)^q)` scored 0.0375 with ONE parameter** — and its own optimum drives the Gaussian weight to zero. Adopted; `mtf_response()` in Python, `FilmMtfResponse()` in C++, parity extended to 8478 probes sampling out to 6x f50. **Both laws are exactly 0.5 at f50**, so this is a shape-only change and no level decision rides along (the C1b trap, avoided by construction). 156 unmeasured stocks verified float32-exact. Schema v9->v10. `RESULT_2026-08-19b_C2_mtf_curve.md` | — | done | **sharpness beyond f50 was 10x too low on the one stock that has been measured** (0.020 vs 0.245 at 98 c/mm) |
| ~~C2b~~ | ✅ **DONE 2026-08-23. Nine more sheets traced; 3 sheets / 7 curves became 12 / 26, and Agfa joined Kodak.** ⚠ **The extractor had to be repaired first, and that is most of the work: FOUR defects, every one of which returned numbers that looked like MTF measurements** — (1) the 1990s sheets emit all three records as ONE path, so 5218 reported f50 69.7 off a trace running along blue, jumping to green and finishing on red; (2) the log grid passed for a curve and was handed back as 5245's GREEN record at f50 236.8, response to 190 %; (3) greedy label matching double-claimed on 5245, and ranking by height instead swapped red and green on 5248 (whose red STOPS at 115 c/mm while green runs to 191), producing a red record sharper than green; (4) a fragment has an f50 and it is meaningless — 5293's red arc starting at 53 % response reported 32.0. All four were found on the new `--overlay` render, none by reading numbers; the other two plot readers already had that gate and this one did not. **Hypothesis tested and half REFUTED:** `q_R ≤ q_G ≤ q_B` holds 8/8 stocks, so the ORDERING is systematic, but the magnitudes are not per-layer constants (red 1.89–2.77, blue 2.38–3.42, sd 0.32–0.37) and C13's "both reds 1.84–1.89" was a two-sample illusion. **So q cannot be derived for the 156 unmeasured stocks and stays per-stock measured** — the `mtf_measured` flag design is confirmed rather than replaced. The power law beats the Gaussian on all 26 curves (1.1×–5.8×, rms 0.0095–0.132) | — | done | **7 stocks now have a per-record measurement where 2 did; C24 is answerable** |
| C2c | ⚠ **NEW 2026-08-19, split out of C2 rather than left buried in it.** `adjacency_um` disagrees with the MEASURED overshoot frequency on both stocks checked: PLUS-X 5231 peaks at **4.7 c/mm** against a stored 16.0 um, FUJI F-125 at **~9 c/mm** against 13.0 um. Systematic, same direction both times. C2 deliberately did not touch it -- adjacency is a separate development effect from the rolloff, and mixing the two would have made "MTF at f50" depend on it. Needs: decide whether `adjacency_um` means the inhibitor diffusion scale or the overshoot peak's reciprocal, then re-derive from the traced curves. ⚠ **THIRD DATA POINT, 2026-08-31 (E3), and it is the same direction again:** `KONICA_IMPRESA_50`'s visual-filter MTF overshoots to **121.4 % at 6.88 c/mm** against a stored `adjacency_um` of **14.0**. That is now 5231 (4.7 c/mm vs 16.0), FUJI F-125 (~9 vs 13.0) and IMPRESA 50 (6.9 vs 14.0) — three stocks, three sheets, two manufacturers, one direction. The overshoot AMPLITUDE was adopted for IMPRESA 50 (`adjacency` 0.10 → 0.214, measured); the LENGTH was deliberately not touched, because it is this row's decision and not E3's | a decision + the traced curves | small | medium -- it controls visible edge crispness |
| ~~C3~~ | ✅ **DONE 2026-08-18, approved.** `SVEMA_FOTO_32` / `_130` base_tint → identity, silver_tone → 0.0. Both were transfers from `SVEMA_FOTO_65`'s withdrawn measurement; a transfer from a withdrawn measurement is not weak evidence but none. Guarded family-wide | — | done | correctness |
| C4 | ЦО-90Д / ЦО-90Л — argued against, needs a yes/no | a decision | — | low |
| ~~C6~~ | ✅ **DONE 2026-08-20, owner-approved and batched with 5201 so the ListBox shifts once.** `FUJI_SUPER_F125_8532` added at tier **2**: printed scalars are `[T1]` (EI 125 at 3200 K, **rms 3.0 with its convention printed — "a visual diffuse density 1.0 above the minimum density", i.e. NET 1.0, the first Fuji sheet in the corpus to state it**, reciprocity, Status M conditions, FN32 edge mark) while the curves and f50 are a **flagged `[T2]` transfer from 8530**. ⚠ Fuji claims 0.56× 8530's granularity (3.0 against a stored estimate of 5.4) and the claim is adopted as printed, not reconciled. ⚠ **Its sharpness panel is a CONTRAST TRANSFER FUNCTION against a rectangular wave chart, not an MTF** — not read; guarded so nobody flags it later. Remainder split out as **C11** rather than left inside a done item | — | done | a documented cine negative, and the unit hazard caught before it became an f50 |
| ~~C8~~ | ✅ **DONE 2026-08-23. Owner chose the seconds control (option a).** `RenderSettings.exposure_time_s` and `AlgoControls::exposureTimeS`, both defaulting to **0.0 = not stated**, so the stage is inert and every earlier render is bit-identical (asserted, 159 stocks × 3 channels). New `film_sim.reciprocity_log_shift()` and its C++ twin `AlgoReciprocity.hpp`; applied **inside stage 8**, on the log exposure, after everything optical and before the curve — and onto the RETAINED log-E plane, so stage 8b sees the same effective exposure the curve did. Parity added as a **third family in `cpp_parity.py`** (5724 probes, 159 stocks × 12 times from 1e-5 s to 3600 s; worst 1.0e-07 decades). ⚠ **My own queue entry was wrong on the model and is corrected here:** I wrote that the honest form is per-pixel intensity-dependent. It cannot be — **all six measured tables are functions of TIME alone** and no source on file carries an intensity axis, so the honest form is a per-channel GLOBAL shift, documented as such. ⚠ **The data was the bigger half of this task. 15 measured `ReciprocityTable` entries were read from the stocks' own sheets in this corpus** (5205, 5217, 5218, 5219-brochure, 5201, 5246, 5274, 5279, 5248, 5231, 5247, 8532, 8572, 8547, VISTA 200), taking the total from 6 to 21 — because **seven of those stocks held p = 1.0 and rendered NO reciprocity while their own sheets print a correction**, and the rest rendered about half of theirs: a single exponent has nowhere to put the offset a film has already lost by 1 s (5205's sheet prints +2/3 stop at 1 s; its exponent reproduced +1/3). ⚠ Also **one real error of mine, caught by the 5205 sheet**: the CC-filter arithmetic first ADDED the filter's density to the worst record, giving 1 stop where the sheet says 2/3 — right channel ordering, wrong level. The lens opens by the printed stops on all three records and the filter takes part of it back, so the unattenuated record loses exactly the printed stops. Pinned by a guard over all 21 tables | — | done | **105 stocks of measured data rendered nothing; 21 now carry printed corrections and the rest a documented slope** |
| ~~C24~~ | ✅ **DONE 2026-08-23, and the rule was wrong in FORM rather than in its constant.** Seven per-record measurements: measured red f50 reads **32.1 33.9 35.4 37.2 37.4 37.6 41.1** -- mean 36.4, spread ±13 % -- while green spreads 52 % and blue 70 %. **Red does not scale with the stock's sharpness at all**, so no value of k in `f50_r = k·f50_b` can fit; physically the bottom record is limited by scatter through the two layers above it rather than by its own emulsion. The estimates were **1.12–1.72× too sharp in red AND 0.70–0.83× too soft in blue**. **ADOPTED:** five measured triples (5217 33.9/58.1/67.4, 5218 37.6/54.6/69.7, 5245 37.2/83.8/100.5, 5248 37.4/75.1/111.2, 5279 41.1/73.1/76.1) with their measured adjacency and, except on 5279, their rolloff exponent; five modern Kodak cine reds **re-anchored to exactly 36.0** (5203, 5207, 5213, 5219, 5246); 5205 and 5293 given a **mixed** triple (measured green and blue, family-anchored red) with `mtf_measured` deliberately unset. ⚠ **SCOPE HELD:** only stocks whose stored blue lies inside the measured 55–111 c/mm range -- 5296 (blue 42) and every pre-1990 stock excluded, every other maker excluded, and `verify.py` now asserts 5296 keeps its own 30.0 so a later "finish the family" pass fails instead of guessing. Green and blue left at their estimates because the measured blues run 0.96–1.43× them with no consistent factor -- only red is a constant. ⚠ **Render impact 7.8–45.7 per 255** (5203 worst), i.e. visible at preview size, unlike C13's. **63 colour stocks still carry an estimated triple** | — | done | **the sharpness of every modern Kodak cine negative in the file** |
| **New 2026-08-20c, from the colorist-team review (RU)** ||||
| ~~C21~~ | ✅ **DONE 2026-08-23, owner-approved (schema v11).** `HalationSpec` gains `radius_scale_r/g/b`; both C++ flavours now build the halation kernel **once per record** rather than once per frame, with a per-record resolvability skip so a tight record that falls below a quarter pixel is passed through while a wider one still blurs. ⚠ **ALL 159 STOCKS SHIP 1.0 AND `verify.py` PINS THEM THERE** -- and that is the finding, not a shortfall: the path-length geometry (base 100-150 um against an 11-16 um pack) bounds the real per-channel ratio at about **1.1**, so a `LayerStack`-derived set would LOOK measured while moving a render by roughly 1 %. The guard exists precisely to stop a later pass "finishing the feature" from the layer order. Renders are bit-identical to v10; the struct GREW, which is why the version moved | nothing for the schema; still a source for the values (a per-channel edge trace across a specular highlight) | done | delivered as capability, zero pixels until measured |
| ~~C22~~ | ✅ **DONE 2026-08-23, owner-approved.** The field's SHAPE is fixed: the film contributes its silver scattering (`callier_q`, unchanged) and the READER contributes how directional it is (new `scanner_specular` / `AlgoControls::scannerSpecular`, shipped at 0). Law, one definition in each language: `D_read = dmin + (D - dmin) * (1 + specular * (Q - 1))`. ⚠ **REFERENCED TO dmin, NOT TO ZERO** -- the scattering scales with developed silver, so clear base carries none; referenced to zero a condenser would darken the film base, which no densitometer measures. ⚠ **AND THE FACTOR MUST REACH THREE CONSUMERS, WHICH IS THE PART THAT IS EASY TO GET WRONG:** the anchor solve, the print chain's own mid-grey reference and the per-pixel stage. With only two of the three, EASTMAN DOUBLE-X's mid grey moved **+54/255** -- larger than the contrast change the effect is for. With all three it holds to 0.2/255 while contrast rises x1.21-1.23 (mono negative) and x1.11 (mono reversal); all 93 colour stocks are bit-identical at any setting. New stage `AlgoStage12b_Callier` in both flavours, fourth `cpp_parity` family (11448 probes, deliberately including densities BELOW dmin because that is where a zero-referenced law would differ) | ⚠ STILL OPEN, and it is the film half: the two monochrome Q values (1.3 / 1.25) are a class rule with no document behind them. One densitometer specification stating a diffuse-vs-specular ratio for a named emulsion would close it -- which is why the control ships at 0 rather than at some "typical scanner" value | done | medium -- 66 B&W stocks, zero effect on colour |
| ~~C25~~ | ✅ **DONE 2026-08-24, owner-approved. T-101 Fig. 18 digitised — and then NOT used for the stored numbers.** The plate was cracked (bow tracking, dash-period classifier, arc-length walker with a turn limit; three earlier attempts had produced *plausible but wrong* fits) and it validates three ways against printed quantities: traced W(0) 0.617/0.552 µm² against M54's printed 0.62/0.555, Table 4's printed granularity ladder reproduced to 0.04–0.22, and `clump_gain` fitting to **exactly 0.000 on all six** independently. ⚠ **THEN TABLE 2 p28 TURNED OUT TO PRINT THE ANSWER OUTRIGHT** — the measured equivalent grain diameter of all six emulsions, in a table already cited on four profiles since 2026-08-23 for its *other* columns. Stored values come from Table 2 through `D_eq = 1.7473 · clump_um`, not from any trace: HPS **1.431**, Tri-X 5223 **1.259**, Plus-X **0.830**, Pan F **0.859**, 8374 **0.687**, 5302 **0.589** µm. ⚠ **ALL SIX ARE UPPER BOUNDS** — p38 states the printed diameters are "expected to be greater than the true values", instrumental weighting uncorrected, and the report does not give that aperture. **HPS was not special:** all six land in 0.59–1.43 µm against 3.2–40 stored across the file (median 13), i.e. a stored 19 µm puts the grain rolloff at 26 c/mm where Fig. 18 shows Tri-X still at half power at 290. Renders change TEXTURE only — `grain_reference_energy` renormalises, so `rms_granularity` still means what it meant | ⚠ **still open, and deliberately:** the remaining 155 stocks. Six 1963 B&W emulsions do not license rewriting the colour negative and reversal column. Also open: the carrier SHAPE — a free exponent fits n = 1.80 (HPS), 2.01 (Tri-X), 2.43 (Plus-X), 2.4–4.1 (fine grain) against the file's fixed n = 2, which is a **renderer** change, not a data one | done | high — 4 stocks corrected, 2 added, and the field's error direction established |
| ~~C26~~ | ✅ **DONE 2026-08-24, owner-approved, batched with C25 so the ListBox shifts ONCE.** Three profiles added for T-101 emulsions this file had been carrying as **footnotes on other stocks**: `EASTMAN_TRI_X_5223` (tier 2 — the 35 mm CINE negative, 250/320 A.S.A., whose numbers had been sitting in `KODAK_TRI_X_320TXP`'s citation with a note that they belonged to a profile that did not exist), `KODAK_8374` (tier 3 — 16 mm TV recording film, blue+U.V. sensitive) and `KODAK_5302` (tier 2, **a PrintStock**, so it moves no ListBox index — and it is the *unity* of Table 4's granularity ladder, which every grain number taken from that document is anchored on). ⚠ **THE GRAIN BLOCK IS THE MEASURED PART OF ALL THREE AND NOTHING ELSE IS:** T-101 prints no characteristic curve, MTF, spectral sensitivity or resolving power for any of its six emulsions, so tone-curve shape, f50 and spectral weights are flagged estimates inline. ⚠ `KODAK_8374`'s `exposure_index` is an **acknowledged invented placeholder** — Table 1 prints speeds for the other five emulsions and leaves 8374's two speed cells blank, because a recording film was rated against a CRT phosphor, not in A.S.A. | nothing — but `KODAK_TRI_X_400TX` deliberately did NOT move: T-101 measured 5223 (cine, 250/320) and 400TX is the ASA 400 still film, so pushing 5223's number onto it would be a class estimate from one sample (method rule 18). `verify.py` pins that non-move | done | medium — closes a 3-stock gap and makes the T-101 ladder checkable instead of implicit |
| ~~C27~~ | ✅ **DONE 2026-08-24, owner-approved. The F-125 family restructured, and the deciding evidence is a SENTENCE, not a curve.** Two issues of «Техника кино и телевидения» (1989 №4, 1990 №1 — the latter a translation of Fuji's own symposium paper) print **Fuji's four-digit code rule in words**: first digit 8 = colour negative, **SECOND DIGIT = GAUGE** (5 = 35 mm, 6 = 16 mm), last two digits = the film. Applied consistently in three separate tables across all five F-series stocks, and matched by Fuji's own Super F-125 sheet ("35mm Type 8532 / 16mm Type 8632"). **So `FUJI_F125_8630` was never a second emulsion and was REMOVED** — a gauge is `default_format`, which this file already models; `8630` now resolves as an alias of 8530. `8632` was never in the file and is not being added, for the same reason. ⚠ **AND THE SAME RULE KEEPS 8530 AND 8532 APART:** they differ in the LAST TWO digits, the part that names the film, and they measure differently — rms **4.0** vs **3.0** at identical speed (125 tungsten / 80 daylight). **Adopted:** 8530 `rms_granularity` 5.4 (estimate) → **4.0** (printed, 1989 №4 Table 1 p70, verified against the page image; convention confirmed as 48 µm at visual diffuse D 1.0, the same definition the 8532 sheet uses). Database **161 → 160 stocks**, second ListBox shift of the day, signed off separately | ⚠ **still open, and each one is recorded with its reason in `NotFound.md` §1.5 rather than left as a vague gap:** σ(D) (1990 Fig. 4 — F-125 and F-64 converge inside the line width exactly where the validating anchor sits), gammas (Fig. 1 draws F-125 and type A *superimposed at matched speed*, and the abscissa has no numeric labels — Fig. 2 is the better candidate), spectral sensitivity (Fig. 6 states no density criterion). Also open: a THIRD MTF measurement (traced f50 ≈ 33 mm⁻¹) disagrees 8 % with the 1989 table's 0.60 at 30 mm⁻¹ and lands near 8532's 32.07 rather than Honjo's 42.0 — nothing changed, but it reframes the conflict already recorded on 8532 | done | medium — removes a duplicate, measures a grain level, and settles the 85xx/86xx question with a printed rule |
| ~~C28~~ | ✅ **DONE 2026-08-25. T-101 Figs. 20/21/23/24/26 read — and the main result is a RETRACTION plus one adopted law.** Fig. 26 (t̄/σ vs mean optical density, log-log) extracted cleanly: `log10(t̄/σ) = -0.6648·log10(D) - 0.1738`, 1039 columns, rms 0.0063 decades, self-validated by five ✕ markers landing within 2.2 % of densities known from §B.2's printed transmissions, and cross-checked by Fig. 21 on linear axes (exponent 0.668 vs 0.665). ⚠ **AND IT IS NOT CONVERTIBLE TO σ_D.** T-101 §2 defines its σ from a two-level opaque-grain model — σ = √(t̄(1−t̄)), eq. (4) t̄/σ = √(t̄/(1−t̄)) — approached *as the aperture becomes vanishingly small*. Measured σ_t/t̄ runs 0.39 → **1.64**, so the small-signal linearisation σ_D = 0.4343·σ_t/t̄ fails across the whole plate. A mid-session reading of "σ_D = 0.648·D^0.665" is **withdrawn**. ✅ **Consequence: the Mees Fig. 302 conflict never existed** — Mees is Goetz–Gould G at a fixed densitometer aperture (Selwyn regime, which is where this file's 48 µm `rms_granularity` lives) and Fig. 26 is the pinhole limit. Different regimes, not commensurable. **ADOPTED instead, from PRINTED Table 3 (p35, no tracing):** grain size depends on development — `D_eq ∝ γ^n`, n = 0.452 (Pan F), 0.396 (Tri-X), 0.425 pooled; the table's own last column normalises by √γ and reproduces to three decimals. Validated at 2 % against Table 2. That exposed a condition mismatch shipped on 2026-08-24: `ILFORD_PAN_F` stores γ 0.55 but its clump_um came from a γ 1.0 sample, so **0.859 → 0.655 µm**. `EASTMAN_PLUS_X_5231` (0.68 vs 0.64) deliberately NOT moved — +2.5 % is inside the source's own upper-bound caveat | ⚠ **still open:** the σ(D) shape question is exactly where it was, since Fig. 26 turned out not to bear on it — `sigma_shape_*` on 68 mono stocks remains a tier-3 estimate against Mees's four unnamed emulsions, and neither source reaches dmax. Also open: the schema stores ONE clump_um per stock while T-101 Fig. 21 measures it falling 20 % across the tone scale, so every stored value is a mid-scale representative | done | high — one corrected stock, one measured processing law, and a false conflict removed from the record |
| ~~C29~~ | ✅ **DONE 2026-08-25, owner-approved. The FIRST measured σ(D) shape on a black-and-white stock — and it REVERSES the sign this file had been storing for reversal film.** Kodak TRI-X Reversal 7266's own sheet carries a granularity panel and a characteristic curve on the SAME log-E abscissa, so σ and D can be paired without a second document. 52 columns paired; **30 kept** after restricting to |dD/dlogE| > 0.5, because where the curve is flat one density maps to many σ. Result `σ_D ∝ D^1.078` (rms 0.038 decades). **Adopted on `KODAK_TRI_X_REVERSAL_200` only:** `sigma_shape_toe` **0.262** at D 0.352, mid 1.0, `sigma_shape_dmax` **2.829** at D 3.089, `sigma_shape_measured=True`. The stored ESTIMATE was 0.70 / 1.0 / 0.50 — i.e. it had granularity FALLING toward dmax where the sheet shows it RISING 2.8×. On reversal film dmax is the unexposed, fully-developed silver, so rising is the physical direction and the estimate had negative film's shape pasted onto positive film. ⚠ **THE LEVEL IS DELIBERATELY NOT ADOPTED:** the panel implies **22.3** at this file's NET-1.0 convention against a stored **10.0**, but the sheet states the curve uses "modified measuring techniques" without defining them, so only the SHAPE is grounded. rms stays 10.0, the 22.3 is cited. ⚠ **AND THE APPARENT INTERIOR PEAK (2.93× at D 3.16) IS DISCARDED** — it lies inside the ill-conditioned flat zone; re-adding it would be storing an artefact. `verify.py` pins all four facts (shape, no peak, rms kept, scope held) | ⚠ **still open, and scoped on purpose:** the other **34** reversal stocks stay on the contradicted 0.7/1.0/0.5 estimate — one measured sample is not a class (method rule 18), and the conflict is now recorded loudly in `film_profiles.py`'s `GrainSpec` docstring, `NotFound.md` and a counted `verify.py` guard rather than silently "harmonised". The 68 monochrome **NEGATIVE** stocks are also untouched: 7266 is a reversal emulsion and its rising shape is exactly what must NOT be transferred. Best remaining lead for negatives is Kodak publication **H-845** | done | high — first measured B&W σ(D), and it caught a stored shape pointing the wrong way |
| ~~C30~~ | ✅ **DONE 2026-08-25. THE C++ GRAIN STAGE WAS RENDERING 4-18 % LOUD, AND EVERY PARITY CHECK PASSED THROUGHOUT.** `film_profiles.hpp` defines `FilmGrainSigma()` as THE ONE DEFINITION — legacy law **divided by sqrt(1 + fog_grain)** so the shape is exactly 1.0 at NET density 1.0, plus the measured-anchor branch. **It had ZERO callers.** `AlgoAddGrain` inlined its own `sqrt(max(D − dmin, 0) + fog)` with no normalisation. Measured across the database: the C++/Python ratio was **exactly sqrt(1 + fog_grain)** — 1.0392 to 1.1832, mean **1.1027** — reproduced to **3.0e-08**, so that single missing divisor was the entire level error and nothing else in the arithmetic was wrong. ⚠ **AND IT SURVIVED BECAUSE `cpp_parity.py` EVALUATED THE LAW DIRECTLY.** The check called `FilmGrainSigma()` itself, agreed with Python on every stock, and never touched the function that renders. Third instance this month of a guard aimed at the wrong subject (see C20). **FIXED:** the normalisation is applied in the stage, hoisted out of both loops since it depends only on `fogGrain`, which the stage already receives — **so no shared signature changed and the AVX2 twin still compiles untouched**. Verified: the stage now returns **exactly 1.0** at net density 1.0 on all 160 stocks × 3 channels, and the 147 legacy-branch stocks agree with the reference to **4.3e-09** ⚠ **AMENDED 2026-08-30 — THE REMAINDER IS NOW CLOSED TOO, and it was bigger than this row's 'partially' suggested.** What survived 2026-08-25 was not a detail: the stage still could not reach the measured `sigma(D)` anchors at all, because its signature took a loose `fogGrain` value and a signature that cannot express the law never fixes itself. `AlgoAddGrain` now takes `dmin[3]`, `dmax[3]` and the `GrainSpec`; the law lives once in `AlgoGrain.hpp` as `AlgoGrainAmpBuild`/`AlgoGrainAmpAt`, and both twins call it. Measured after: worst relative disagreement against the Python reference **2.52e-07** over 2415 probes, and `|amp - 1|` at NET density 1.0 is **exactly zero** on all 161 stocks x 3 channels. ⚠ A second entry point, `AlgoAddGrainRaw`, was added for stages 13 and 14 and is deliberately NOT pinned: print and dupe stocks carry no published rms, and the reference does not normalise them, so pinning them would have been the same error in the opposite direction. See `RESULT_2026-08-30_C30_C33_C40.md`. | ⚠ **the measured sigma(D) SHAPE is still unreachable from the stage** — 13 stocks. It needs the `GrainSpec` and `dmax`, which means changing a shared signature and moving the AVX2 twin in the same commit; scoped as its own change. Those 13 now get the correct LEVEL and the legacy SHAPE: their error drops from 0.39x–2.2x across the tone scale to a pinned worst of 1.73 at net 2.5, with net 1.0 exact | done | **high — this is the largest single accuracy defect found in the project, and it was in the shipped plugin** |
| ~~C31~~ | ✅ **DONE 2026-08-25. Two validation tiers added, both of which would have caught C30 on their first run.** **(a) A STAGE-LEVEL parity family.** Every other family in `cpp_parity.py` compares a law against a law; this one compiles and calls **`AlgoAddGrain` itself** on a synthetic plane and recovers the amplitude the stage actually applied. The extraction is exact rather than fitted: with the grain field set to 1.0 and gain to 1.0, `amp = out − D` with no arithmetic in between. 2400 probes (160 stocks × 3 channels × 5 densities). It asserts the load-bearing identity — **amplitude exactly 1.0 at NET density 1.0**, the convention `rms_granularity` is quoted at — and judges the two populations separately, exact for the 147 legacy stocks and a pinned, quantified gap for the 13 measured-shape ones, so a scoped defect cannot silently grow **or** silently close. **(b) A CLOSED-LOOP tier in `verify.py`.** Render, measure back through the manufacturer's own convention, compare against the published number. Unlike checking a stored value against the datasheet it came from, this is not circular — it exercises the whole chain including the conventions, and it can fail. Three checks now: the existing rendered-rms-at-net-1.0, plus **a sinusoid at f50 returning exactly 50 % modulation** (5 stocks, both transfer laws) and **the rendered characteristic curve reproducing the stored curve** (3 stocks × 5 exposures, within 0.002 D) | ⚠ **this tier cannot validate against real film** — that needs a scan of a known target under stated processing on a characterised scanner, which is physical and not in the corpus. What it does cover is every failure where the renderer stops honouring data it already holds, which is the class that actually keeps occurring. ⚠ The f50 check needed its sampling fixed: with an arbitrary rate the sine leaks across FFT bins and two stocks read 0.559/0.590 as false failures; px/mm is now chosen so f50 lands on an exact bin | done | high — converts a recurring manual audit into a standing gate |
| ~~C32~~ | ✅ **DONE 2026-08-25 — the AUDIT, not the fix. The bypass rate on the shared-law surface was 2 of 2.** After C30 found `FilmGrainSigma()` with no callers, the obvious question was how many others. The surface turns out to be exactly two functions: `film_sim.py` calls precisely `fp.grain_sigma` and `fp.mtf_response` from the database module — nothing else — and the generator emits exactly those two into `film_profiles.hpp`. **Both were unreachable from the stages.** Not a sample; the whole surface. **GATE ADDED:** `check_law_reachability()` in `cpp_parity.py` fails the build when a published law is called by no stage source, and fails equally when a law recorded as bypassed quietly acquires a caller without the record being removed. ⚠ **Its first run passed on a comment** — `FilmGrainSigma` showed as reached from 1 source, that source being the comment explaining it is NOT called. C++ comments are now stripped before the search. A gate that passes on prose about its own failure is the defect it exists to catch | ⚠ **`FilmMtfResponse` remains bypassed and is now recorded with its cost**: the measured `1/(1+(f/f50)^q)` rolloff is a frequency-domain form and **the C++ side has no FFT at all** — `AlgoSeparableBlur.hpp` opens by arguing why not. Applying it needs an architecture decision (numerical kernel, an FFT path, or a fitted separable equivalent), so the 9 stocks with a measured q render on the legacy Gaussian: exact at f50 by construction, **up to 3.8x too much modulation at 2x f50** | done | high — closes the class C30 was one instance of |
| ~~C33~~ | ✅ **DONE 2026-08-25. Scalar and AVX2 were computing different models within hours of the C30 fix, and nothing would have said so.** The net-1.0 grain normalisation went into the scalar stage; the AVX2 twin was left to its owner — and the two paths were immediately **1.039x to 1.183x apart on grain amplitude**. A divergence in the MODEL, not the vectorisation, which this project's own AVX2 rules forbid. **Mirrored into `AVX2/Algo_11_Sim.cpp` at zero inner-loop cost**: it folds into `gain` before the broadcast, since `gain * (amp * scale) == (gain * scale) * amp`, so the vector chain is unchanged. ⚠ **Verified by compiling the twin under `AlgoType = float` with `-mavx2 -mfma`**; the header was flipped for the test and restored to `double` immediately, so the shipped alias is untouched. **GATE ADDED:** `check_twin_consistency()` asserts that tokens carrying a LAW appear in both a stage and its AVX2 twin. The two files may differ in everything about HOW they compute and in nothing about WHAT ⚠ **AMENDED 2026-08-30 — THE REMAINDER IS NOW CLOSED TOO, and it was bigger than this row's 'partially' suggested.** What survived 2026-08-25 was not a detail: the stage still could not reach the measured `sigma(D)` anchors at all, because its signature took a loose `fogGrain` value and a signature that cannot express the law never fixes itself. `AlgoAddGrain` now takes `dmin[3]`, `dmax[3]` and the `GrainSpec`; the law lives once in `AlgoGrain.hpp` as `AlgoGrainAmpBuild`/`AlgoGrainAmpAt`, and both twins call it. Measured after: worst relative disagreement against the Python reference **2.52e-07** over 2415 probes, and `|amp - 1|` at NET density 1.0 is **exactly zero** on all 161 stocks x 3 channels. ⚠ A second entry point, `AlgoAddGrainRaw`, was added for stages 13 and 14 and is deliberately NOT pinned: print and dupe stocks carry no published rms, and the reference does not normalise them, so pinning them would have been the same error in the opposite direction. See `RESULT_2026-08-30_C30_C33_C40.md`. | ⚠ token-based and therefore crude — it proves a marker is present, not that the arithmetic matches. A floor. The stage-level probe measures the scalar path numerically; **the AVX2 path still has no equivalent numeric probe** | done | high — a whole-model divergence caught the same day it was created |
| ~~C34~~ | ✅ **DONE 2026-08-25. A documentation-consistency gate, after an audit found four hardcoded counts wrong by up to 2.3x.** `build.py` already gated `PROGRESS.md` on a build-facts stamp and that check had never fired — for the good reason that it was the only claim being checked. New `doc_consistency.py` generalises it: a REGISTRY of specific load-bearing sentences, each with the pattern that finds it and the live expression it must equal, run from the audit stage with `--assert`. ⚠ **A PATTERN THAT STOPS MATCHING IS A FAILURE, NOT A PASS** — the sentence was edited or deleted and the claim is now unchecked, which is the state the script exists to end. ⚠ **It deliberately does NOT parse every number**: most numbers in these documents are measured values, residuals, dates and page references, and a checker that guessed would produce noise and be switched off within a week. The registry must grow with each new claim | ⚠ **it cannot catch a wrong CLAIM, only a stale COUNT.** "39 raster pages are on disk and unread" was wrong in its second half, not its first, and no arithmetic detects that. Prose still needs reading | done | high — converts a recurring manual audit into a standing gate |
| ~~C35~~ | ✅ **DONE 2026-08-25. The project-root `doc/` folder reviewed for the first time, and it had never been delivered.** Seven documents, 2661 lines — AVX2 optimisation, type alignment, the defect layer, memory, single-thread work, stage fusion. Every delivery archive to date covered `PYTHON/profile_generator/doc` only, so this folder had gone un-reviewed and un-shipped for the life of the project. **Corrections applied:** stock denominators (93/100 → 160; monochrome 36 → **68 of 160**; reversal 22 → **36 of 160**); three documents still described stages **15, 16 and 09b as stubs** when all three fully render (only 3c is a stub); and `STAGE_FUSION_PROPOSAL`'s central 4K memory argument is **superseded by its own project** — it quotes the pre-ping/pong 2.90 GB float footprint, but M1 shipped and 4K UHD is now ~0.77 GiB, so the case for fusing to reach 2 GB is moot. ⚠ **AND THE FINDING THAT MATTERS MOST: neither AVX2 document states that the vector build requires `AlgoType = float`.** Both read as though it is simply live, while the shipped header sets `double` and 17 AVX2 translation units static_assert against it | ⚠ **not corrected: the performance numbers** — every figure is a 2026-08-11 measurement never repeated, and stage 11 and stage 9 changed on 2026-08-25 beneath tables that time them. Re-measuring is a real task; guessing is worse than a dated figure honestly labelled. ⚠ `D1_TYPE_ALIGNMENT`'s outstanding-work table could not be reconciled at all and is flagged for a re-run rather than patched | done | medium — seven documents brought back into the review perimeter |
| C23 | ⚠ **BROMIDE DRAG / DIRECTIONAL EXHAUSTION -- genuinely absent, and it needs a spec that does not exist yet.** No carrier, no code, no mention anywhere in the database. ⚠ **AND IT IS NOT AN EMULSION PROPERTY.** It is a property of the PROCESSING MACHINE -- transport direction, continuous versus rack-and-tank, replenishment rate, agitation -- so it belongs in a lab/processing spec, not in `FilmProfile`. `ProcessingSpec` (currently descriptive only: which developer the stored curve represents) is the natural home. The effect is **directionally locked to the transport axis**, i.e. anisotropic and oriented, which this pipeline CAN express (the coating field is already anisotropic across-web versus along-web) but nothing currently drives. Wanted: a reference frame showing the streaks, plus any lab literature quantifying the bromide gradient behind a highlight | a lab model + a reference image | medium | low-medium -- strong archival / lab-print signature, but the least evidenced item on the list |
| ~~C36~~ | ✅ **DONE 2026-08-26, and the RESULT IS A REFUSAL.** The MTF panel on H-1-2254 p5 was traced (694x605 px raster, log-log axes calibrated from 10 frequency and 12 response gridlines, worst residual **0.008 decades**) and it turns out **two of the three records never reach 50 % response**: the curves stop at **82.2 cycles/mm** with GREEN still at 53.1 % and RED at 50.6 %. Only BLUE crosses, at **51.9 cycles/mm**. ⚠ **So the stored scalar 72.0 is wrong in BOTH directions at once** -- too sharp for blue, too soft for the two proven >= 82.2 -- and no single number is right about a set spanning a factor of 1.6. Resolved by **schema v13**: `PrintStock` gains `mtf_f50_r/g/b`, `mtf_f50_bound` and `mtf_measured`, with **0.0 meaning CENSORED** and the bound carried alongside -- the same idiom `DyeStabilitySpec` introduced at v12, because the problem is the same one. ⚠ **The legacy scalar is deliberately LEFT UNCHANGED at 72.0**: it is what the reference renderer reads, and changing it would move a render on the strength of a number the sheet does not state. ⚠ **No rolloff exponent is stored, and that refusal is measured too:** blue's traced span reaches only 36-82 cycles/mm, so a carrier normalised at f = 0 would be fitted over 0.36 decades with just **0.16 of them BELOW f50** -- the fit is good where it sits (q 1.78 at rms 0.026, 2.8x better than the Gaussian) and says nothing about the knee, which is what q means. ⚠ A tracing trap found and fixed on the way: a log-log panel is ruled at 1/2/3/5/7/10/20..., those rules are 1 px thick where the curves are 3, and the tracker **followed the 100 % and 70 % RULES** through the flat low-frequency half. Cost, before `_strip_gridlines`: blue's q came back 0.74 at rms 0.063 against 1.78 at 0.026. Both look like fits; only one is of the curve. **Still estimated and still unfixable from this sheet:** `grain_rms`, because H-1-2254 publishes no rms figure at all -- only that granularity is 'similar to VISION Color Intermediate 2242', a stock this database does not hold | — | done | the first CENSORED MTF in the project, and an estimate shown to be wrong in both directions |
| ~~C37~~ | ✅ **DONE 2026-08-29 — AND IT YIELDED NO NEW DATA, WHICH IS THE FINDING.** This row promised "up to 13 new spectral sets plus 2 cross-checks". ⚠ **IT IS THE OTHER WAY AROUND: 0 new sets and 11 cross-checks.** Every stock behind the 15 findable panels ALREADY carries a spectral set, and `spectral_vector.py`'s own docstring had recorded that on 2026-08-26 — the row was written the same day and never updated. Confirmed empirically before starting rather than taken on the label's word. ⚠ **AND THE 2026-08-26 SWEEP WAS PROSE, NOT AN AUDIT**: it ran by hand, its numbers lived in a docstring, and NOT ONE of the eleven sheets was in the registry, so nothing re-ran it and a reader change could have moved any curve silently. **That is what was actually delivered: the registry goes 4 sheets → 11**, every agreement is pinned, and `--assert` now fails on drift. ⚠ **THE COMPARISON ITSELF WAS ALSO WRONG.** It included the samples where the shorter trace dives into its own floor, which measures where each reader stopped drawing rather than the film; `_core_rms` guards one sample in from whichever measured run ends first. 5218 red goes **0.367 → 0.241** and 5217's pinned triple moved **0.109/0.091/0.049 → 0.077/0.086/0.047** with nothing changed on either side. **EIGHT OF ELEVEN AGREE** at core rms ≤ 0.086 decades. ⚠ **THREE DO NOT, and one previously-recorded explanation does not survive inspection:** (1) **5245 blue 0.335** — the docstring blamed truncation plus per-layer normalisation; re-normalising on the shared span changes the number by nothing. Sample by sample the two agree to **±0.06 from 400–480 nm**, the whole peak, and diverge only on the 490–520 tail — where the STORED values run −0.60/−1.15/−1.80/−2.45/−3.10, steps of 0.55/0.65/0.65/0.65, **a straight line**, which a dye tail is not. The stored tail below 490 nm looks EXTRAPOLATED, not read. (2) **5218 0.241/0.210/0.138** — never recorded before, and not truncation: the trace is systematically higher on every rising flank (+0.13…+0.26) and lower on every falling one, on all three layers. A consistent NARROWING, i.e. a wavelength-scale difference or a genuinely different reading. (3) **5231 pan 0.213** — a double-humped panchromatic curve whose two maxima are a quarter-decade apart; the raster reading peaks on the 400 nm hump, the vector trace on 590. ⚠ Also corrected: the queue's **"both 5205 sheets" is ONE document** — `5205t.pdf` and `H-1-5205t.pdf` are byte-identical (md5 edd35d27…) — and **5218's panel is on p3, not the p4 this row gives**. ⚠ **FIVE PANELS ARE FINDABLE AND NOT EXTRACTABLE**, causes measured and recorded in `UNREACHABLE`: 5248 p3 and 5293 p4 have 17 long paths each and **not one coloured** — they draw all three curves in BLACK, so the ink rule says nothing and the mono reader handles only one curve (blocked on METHOD, and the nearest thing to real new work left here); 5219 p3 has no path of 8+ segments at all; 8532 p1 is a Fuji layout with 3 images and 5 long paths; 8547 p1 is 24 images — a RASTER panel, and its stored set came from a raster reading anyway. **Nothing re-adopted**: choosing between a vector trace and an adopted raster reading is the call XX1 made deliberately with its evidence set out, and it is owed the same here — raised as **C38** | nothing | done | a hand sweep became an enforced audit, one recorded diagnosis was overturned, and two disagreements nobody had noticed came to light |
| ~~XX1~~ | ✅ **DONE 2026-08-26, owner-supplied source.** `EASTMAN DOUBLE-X Negative Film 5222.pdf` = KODAK Publication H-1-5222 **Revised 7-15 (JULY 2015)**. ⚠ **IT IS A SECOND EDITION OF A SHEET THE CORPUS ALREADY HELD, AND THE DIFFERENCE IS THE ART, NOT THE CONTENT.** The edition on file (revised 3-26) prints the identical figures -- same plot numbers F010_0029AC and F010_0031AC -- as RASTERS; the 2015 edition draws every plot as VECTOR PATHS and embeds no images at all. Panels that had to be read by hand became measurable. **FOUR ADOPTIONS AND TWO CORRECTIONS.** (1) **MTF measured for the first time**: f50 **42.2** cycles/mm, rolloff q **2.88**, printed adjacency overshoot **+25 %** peaking at 4.1 cycles/mm, replacing the flat estimate 56/56/56 which was **1.33x too sharp**. ⚠ And it gains an external check no estimate could have had: PLUS-X 5231, the corpus's other Kodak B&W cine negative, measures **41.3** off its own sheet -- two sheets, two traces, **2 % apart**, where the estimated pair read 56.0 and 60.0. q is adopted at +25 % where 5279 was refused at +42 %, and the discriminator is the FIT (rms 0.076 here, inside the 0.0095-0.132 band; 5279 returned 0.25-0.34). (2) **Spectral sensitivity re-traced from vector paths, and the DENSITY CRITERION READ FROM THE PANEL'S OWN CAPTION.** ⚠ The panel draws **TWO** curves -- 'D = 0.3 Above Gross Fog' and 'D = 1.0 Above Gross Fog', about 0.55 decades apart -- so a reader that took 'the curve' would have stored whichever the page emitted first. The adopted set is matched to its printed caption by geometry. It agrees with the 2026-08-02 raster reading to **rms 0.037 decades** on the same 430 nm peak: confirmed, not corrected, but now machine-derived with residuals on record. (3) **ProcessingFamily populated** from the five printed per-curve gamma labels (4 min 0.50, 5 min 0.56, 6 1/2 min 0.66, 9 min 0.84, 12 min 1.05 in D-96 at 21 C) -- the 4th stock in the database to carry a processing axis. (4) **Base+fog corrected 0.1977 -> 0.2328.** ⚠ The 2026-08-02 raster trace of this same curve had the SHAPE right and the LEVEL wrong: its gamma is within **0.0004** of the vector refit and it reproduces the vector path to rms 0.0123 D, but its base+fog was **0.035 D low**. Confirmed two independent ways before changing it -- the printed ticks give 0.2369, the frame edges give 0.2281, and 0.1977 sits outside both -- and the mid-grey anchor was re-checked afterwards (D 1.1786 against the recorded 1.178), so this is a level correction only. (5) ⚠ **DEVELOPER CORRECTED, and this is the substantive one:** `_PROCESSING` said **Kodak D-76** from Иофис 1964, while Kodak's own sheet says **D-96 at 21 C** in three separate places -- the PROCESSING table, the sensitometric caption and the MTF caption. D-76 is a still-film developer; 5222 is a motion-picture stock. The Иофис row is kept for what it actually evidences (1963-64 local practice, plus its independent confirmation of the ASA 250/200 pair and the 0,6-0,7 gamma band) rather than deleted -- method rule 4. New audit `kodak_time_gamma.py` re-derives the printed time-gamma series from the DRAWN curves and reproduces four of five within 2 %; ⚠ the **9-minute** point does not (measured 0.798 against printed 0.84) and is stored **as printed**, with the disagreement recorded as a named exemption so a new one elsewhere fails | — | done | an estimate replaced by a measurement on a heavily-used stock, a wrong developer fixed, and a 0.035 D level error found by re-reading a figure the project had already traced |
| ~~XX2~~ | ✅ **DONE 2026-08-26, schema v13.** `DevelopmentPoint` gains **`base_fog`**, additive and inert. ⚠ **WHAT THIS CLOSES IS A SILENCE, NOT A WRONG NUMBER.** `ToneCurve.dmin` is one value and therefore describes ONE development condition -- but nothing in the schema said which, and nothing said that fog moves with development at all. It does, measurably: DOUBLE-X 5222's five traced characteristic curves give base+fog **0.231 / 0.233 / 0.233 / 0.275 / 0.296** at 4 / 5 / 6.5 / 9 / 12 minutes in D-96, a **28 % rise** across the family. ⚠ These are **traced, not printed**, unlike the gammas beside them: the sheet draws a Time-Fog curve but puts no numbers on it, so they come from the left plateau of each curve, measured over 0.12 decades by `kodak_time_gamma.py`. `verify.py` now asserts the link that was previously implicit -- **the stored dmin equals the fog of the stored development condition and of no other** (0.2328 at 6 1/2 min; it would be 0.296 at 12). Same gap `ProcessingFamily` closed for CONTRAST, closed for the other quantity the same plot measures | — | done | every stock's dmin now has a stated condition beside it instead of an implied one |
| **New 2026-08-20b, from the DIR-coupler parity audit** ||||
| C16 | ⚠ **NARROWED 2026-08-25 BY C17's CLOSURE — IT IS NOW A ONE-NUMBER DECISION.** The adjacency/coupler blur is still not the same effect in the two renderers: Python multiplies by the ANALYTIC Gaussian transfer in the frequency domain, C++ convolves a truncated separable spatial kernel. Measured across sigma they agree to **6e-5 above ~1.2 px** and diverge to **1.5e-1 at 0.4 px** — a Gaussian narrower than the sample grid is not represented by either form, so no tolerance fixes it. Stored `edge_um` is 9–13 µm, so at 40 px/mm (a 35 mm frame ~960 px wide) the edge sigma is **0.36–0.60 px**: inside that divergent band and ABOVE the now-shared 0.25 px gate, and `interimage_parity.py`'s S09 ramp rows show it directly at 1.0e-2 to 2.6e-2. ⚠ **WHAT CHANGED:** the two renderers no longer differ in WHETHER the stage runs (C17 gave Python the same gate) — only in HOW it is computed below ~1.2 px. So the remaining options collapse toward one number: (a) render the edge term at a higher internal resolution; (b) **raise the shared threshold to ~1.0 px, where the two forms converge** — which is also the honest model statement, since a 9–13 µm feature at 25 µm/px is below the sampling limit and rendering it anyway is aliasing a sub-pixel feature; (c) replace the frequency-domain reference with the same truncated kernel so they agree by construction. ⚠ (c) makes them agree without making either right, and (b) removes a visible effect from every ordinary-resolution render | **a decision on (a)/(b)/(c)** — recommended **(b)**, but it changes every render and is therefore the owner's | small for (b)/(c), medium for (a) | **medium–high — half of the largest colour effect in the chain, still implementation-dependent below 1.2 px** |
| ~~C17~~ | ✅ **DONE 2026-08-25, and it is a pure parity fix with no fidelity judgement in it.** `AlgoDirCoupler.hpp` has gated BOTH coupler components below `ALGO_COUPLER_MIN_SIGMA_PX` = 0.25 px since it was written (`Algo_09_Sim.cpp:1018` and `:1023`); `film_sim.apply_dir_couplers` had **no gate at all**, so below that scale the two renderers were not approximating each other — one ran the stage and the other did not. Python now carries the same gate at the same threshold. **THE THRESHOLD WAS ADOPTED, NOT CHOSEN:** taking the shipped and reviewed C++ constant is what keeps a fidelity decision out of a parity fix, and its stated reason holds identically on the Python side (below a quarter pixel the discrete kernel has one significant tap, so the pass is an identity). The crossovers are not exotic scales — the long term switches off below **3.1 px/mm** (`EASTMAN_5247_1974`, radius 80 µm) and the edge term below **27.8 px/mm** (`KODACHROME_64`, edge 9 µm), which is a 35 mm frame about 670 px wide. `interimage_parity.py` unchanged at worst **5.335e-05** over 5 stocks × 2 fields × 5760 values, because its probe runs above the gate; its docstring now records the gate as SHARED rather than as an open divergence | ⚠ **this does not settle C16**, which is a different question: the two blurs are still different FORMS, and what remains open is the shared threshold's VALUE | done | medium — removes an implementation-dependent stage boundary |
| C18 | ⚠ **`density_weighting` IS UNBOUNDED AND ITS MAGNITUDE IS TIER 3.** The reversal interimage mechanism is documented (US4729943A: iodide from the first B&W developer, landing in high dye-density areas) but the 0.65 is not. Measured on `FUJI_VELVIA_50`: the per-donor weight rises **0.44 → 1.82** as density goes 0.2 → 3.2, worst-case correction **−0.58 logE ≈ 1.9 stops**, and disabling stage 8b moves saturated patches by up to **143/255**. No cap. This is the largest undocumented number in the colour path and 36 reversal stocks ride on it. Wanted: a saturating form with a measured asymptote — which needs the §D measurement below, not a better guess | a measurement (D1/D2 + three filters), or a decision on the form | small once measured | **high — 36 stocks, and it is the biggest single unpinned colour number** |
| C19 | ⚠ **THE LATERAL HALF HAS NO PROVENANCE CHAIN AT ALL.** `InterimageSpec` is patent-derived, per-stock-solved and asserted against published percentages. Its twin `CouplerSpec` is **87 hand-typed literals** — `CouplerSpec(0.15, 52.0, 0.08, 12.0)` — with no registry, no derivation, no citation and no tier. Worse, its parameters are already contradicted by measurement: `adjacency_um` disagrees with the traced MTF overshoot frequency on **all four** stocks checked (5231 4.7 c/mm vs 16.0 µm; F-125 ~9 vs 13.0; 5201 10.7 vs 16.0; 5274 16.1 vs 18.0). `adjacency`, `edge_strength`, `edge_um` and `radius_um` are four parameters describing **one** inhibitor diffusion length, currently set independently. Merge with **C2c** — same chemistry, same question | nothing to find for the MTF half; the strength needs the wedge measurement | medium | **high — it is the other half of the effect, and the only half with no evidence behind it** |
| ~~C20~~ | ✅ **DONE 2026-08-25. A guard that could not fail, and the old name is the finding.** `verify.py`'s "interimage leaves a neutral untouched" rendered **0.18** — the mid-grey ANCHOR the correction is referenced to, the one point where every `(D_j − d_ref)` is zero and the correction vanishes identically. It was therefore true by construction for **any** value of the interimage matrix, and it promised a property it never tested. **Renamed** to "interimage leaves the ANCHOR neutral untouched (0.18, where it must)", and a **second guard added** that pins the off-anchor movement as intended behaviour: on `KODAK_PORTRA_400` with the stage disabled, grey 0.45 moves **15.9/255** and grey 0.06 moves **6.5/255**, and the guard asserts both the magnitudes and their ordering. That movement is the MECHANISM, not a leak — white-light gamma coming out lower than separation gamma is the patent's own metric for interimage effect. `InterimageSpec`'s docstring is qualified to match: it said the correction "vanishes on a neutral", which is true of ONE neutral, not of the neutral axis, and no real emulsion's is invariant along it | nothing | done | correctness of the record — the render never changed, but a vacuous check now measures what it claimed |
| **New 2026-08-20, from the KODAK folder review** ||||
| ~~C9~~ | ✅ **DONE 2026-08-25, and THIS ENTRY'S OWN DIAGNOSIS WAS WRONG.** C9 recorded the cause as a family-classifier limit — "handles 3 dyes or 3 dyes + neutral, not 3 + neutral + dmin". It never was: family B takes any three of however many curves it is offered, so two extra traces cost it nothing. **The cyan trace never reached the classifier.** Kodak draws it as the yellow-under-magenta overprint — two bit-identical paths of **7 segments each** — and `extract`'s `n < 8` segment filter dropped both, leaving nothing in the 615–700 nm band for any triple to pass the band test on. ⚠ **A diagnosis recorded in a queue is not evidence**; this one survived a fortnight because it was plausible and nobody re-derived it. **FIX: identify traces by INK, not by segment count** (a lower threshold would admit gridline stubs on every other sheet). Kodak's rule is physical — each trace is drawn in the colour of light it concerns, so yellow dye is drawn BLUE, magenta GREEN, cyan RED via the overprint; read off the panel's own legend swatches. New family C = peak_1.0 triple + as-printed neutral + as-printed dmin. **THE VALIDATOR IS THE POINT:** family A's `neutral = C+M+Y` cannot hold when the dyes are normalised and the neutral is not, but `Neutral − Dmin = k·(C+M+Y)` with the three k EQUAL must — that is what makes it a *visual* neutral. Unconstrained least squares returns **0.628 / 0.604 / 0.595**, a 5.4 % spread on three free numbers, at rms **0.019 D**; without the Dmin term rms is 0.085 and the coefficients scatter 0.86–1.61, which is what identifies which dark trace is which. Adopted: `KODAK_VISION2_50D_5201` dye set, peak_1.0, peaks **450 / 540 / 680 nm — identical to 5217 and 5218**, a family check the extractor never saw. The 11 existing sheets are untouched: family C is only tried after A and B fail, and `--assert` proves all 12 reproduce | ⚠ the neutral and dmin traces are NOT stored (as-printed while the dyes are not — one record cannot carry two conventions), and `d_cyan` below 430 nm holds the trace's first value because the plate's cyan curve starts at 430.4 nm. Both stated at the field | done | medium — closes one of 5201's two gaps and adds a reusable ink reader |
| ~~C10~~ | ✅ **DONE 2026-08-25. The FIRST vector-traced spectral sensitivity set in the database**, by a new script `spectral_vector.py`, registered in `build.py`'s audit stage. Every earlier spectral set came from the 2026-08-02 raster batch or `agfa_vista.py`'s dash-legend reader. Layers assigned by the same ink rule C9 established — and the entry's own prediction was right: Kodak draws the red record as yellow under magenta here too, the two paths bit-identical (max difference 0.0). **Assignment then checked three ways, none of them the ink:** the legend swatches (green on "magenta dye forming layer", amber on "cyan"), the absorption bands (peaks **470 / 540 / 650 nm**, ascending), and the independently-adopted 5217/5218 sets — red and green agree to rms **0.05–0.14 decades**. ⚠ **5201's BLUE LAYER PEAKS AT 470 nm WHERE ITS SIBLINGS PEAK AT 410–420**, which is the one disagreement in the cross-check (blue rms 0.24–0.42) and is a **printed** feature: a narrow cusp above log S 2.0 at 470, higher than the 445 bump, then a cliff to zero by 500 — confirmed on a 26× render before adoption. Pinned, because a later "correction" toward the family shape would be undoing a measurement. ⚠ **AND THIS ADOPTION MOVES A RENDER** (the dye set does not): a stock with spectral data takes `spectral_balance_gains()` instead of the 600/550/450 nm proxy, and a red layer peaking at 650 rather than 600 means tungsten drives it harder — **+0.28 stop of red gain at 3200 K**, −0.17 at 10000 K, green the unchanged anchor. Size and direction both asserted | ⚠ **STILL OPEN, and it is a provenance finding rather than a gap: the criterion is printed on NO sheet in this family.** 5201's footnote says only "reciprocal of exposure (erg/cm²) required to produce **specified density**" without naming it; the three existing sets carry `log_reciprocal_erg_cm2_D0.2_above_dmin`, yet 5218 and 5217 print the same unspecified wording and 5219's footnote is not in its text layer at all. Owner decision: 5201 stores what its sheet prints, the other three are LEFT ALONE, and the conflict is recorded with a two-way `verify.py` guard (method rule 4). Best next move: Kodak publication **H-1** *Image Structure*, cited by name on the sheet, absent from the corpus | done | medium — and it generalises: the ink reader works on every H-1 brochure in the corpus |
| ~~C11~~ | ✅ **DONE 2026-08-23, owner-raised at top priority.** Three of the four panels are traced and adopted; the fourth cannot be stored for a SCHEMA reason, not a source reason. **Characteristic curves:** rms 0.005-0.009 D, replacing the `[T2]` transfer from 8530 -- whose dmin turned out ~0.25 D high on every layer and whose red gamma was 16 % too steep. **Spectral sensitivity:** adopted, peaks 469/553/645 nm, Fuji's crossover truncation carried as the -4.00 sentinel exactly as on `FUJI_ETERNA_VIVID_500T_8547`. **Contrast transfer function:** the hazard is RESOLVED, not carried -- the note here said the square-to-sine conversion "needs the chart's duty cycle", and it does not: a rectangular wave chart is 1:1 and **Coltman's inversion** `MTF(f) = (pi/4)[C(f) + C(3f)/3 - C(5f)/5 + C(7f)/7 + C(11f)/11 - C(13f)/13 - ...]` needs only the curve. Printed CTF crossing **37.78** c/mm -> sine **f50 32.07** (robust to +-30-40 % of the extrapolated tail: 30.98-33.49). **Spectral density: NOT stored** -- Fuji plots only mid-scale-neutral and minimum densities and never separates the three dyes, while `SpectralDyeDensity.validate()` requires all three; both traces are quoted in the profile comment so nothing is lost, and `NotFound.md` now files it as a schema-shape mismatch. ⚠ **The x-label hazard this entry flagged was worse than described:** the labels are not merely "out of order without minus signs", they are **genuinely mis-set and non-monotonic** (`-4.5 -3.0 -3.5 -2.0 -2.5 -1.0 -1.5 0.0 0.5 1.0`), the SAME sequence appears on the F-500 8572 sheet, and the origin had to be settled by physics -- fitted toe_x and mid-grey density against the traced Kodak family -- then cross-checked against the two sheets' own speed points (0.577 decades apart against the 0.602 their printed EIs demand). Stock re-tiered 2 -> 1. Same method applied to 8572 in the same pass | done | done | delivered: the only stock whose own sheet contradicted its stored curves now agrees with it |
| ~~C12~~ | ✅ **DONE 2026-08-25, owner-approved — and the class was THREE TIMES the size this entry claimed.** C12 was filed against two profiles. A sweep for `\[T[123]/T[123]\]` found **six** resolving to tier 3 on `fitted_from="analogy"`: the three VISION2 camera negatives (`KODAK_VISION2_500T_5218`, `_200T_5217`, `_250D_5205`) and the three VISION negatives (`KODAK_VISION_500T_5279`, `_200T_5274`, `_250D_5246`). Every one of them owns its own Kodak H-1 / TI sheet, **four have a σ(D) shape traced from that sheet**, and in all six the description says the T3 half is the same single scalar — `rms_granularity`, because from VISION onward Kodak prints granularity CURVES and no rms number. "analogy" was simply false for them. **All six → tier 1**, matching the two precedents already in `_UNTAGGED_TIER` (`EASTMAN_5247_1983` is tier 1 with HAND-FITTED tone curves; `FUJI_SUPER_F125_8532` is tier 1 with a transferred red/blue f50 — both larger residuals than one flagged grain scalar on a profile carrying a manufacturer sheet). ⚠ **THE MECHANISM WAS CLOSED BY A CLASS GUARD, NOT BY LOOSENING THE REGEX.** A mixed tag must now appear in `_UNTAGGED_TIER` **and** may not resolve to 3, or `verify.py` fails — 3 being exactly the value the regex falls back to, an entry resolving there is indistinguishable from the bug. The strict regex is the feature: it forces a human decision on every future mixed tag instead of quietly picking a number | nothing — no value moved, only the tier the profiles report and what `fitted_from` claims | done | correctness — six documented stocks stop understating themselves, and the next mixed tag cannot slip through |
| ~~C13~~ | ✅ **DONE 2026-08-20c, owner-approved.** `KODAK_VISION_200T_5274` now carries its MEASURED MTF: f50 **35.4 / 68.8 / 74.0** cycles/mm (was the estimate 56.0 / 64.0 / 72.0), adjacency **0.162** (was 0.09), rolloff **q = 2.94**, `mtf_measured=True`. Third stock with a traced MTF. ⚠ Green and blue CONFIRMED the estimate to 7 %; **red was 1.58x too sharp**. ⚠ **RENDER IMPACT IS SCALE-DEPENDENT and smaller than the number suggests** -- measured on a bar-sweep target: worst **3.9/255 at 48 px/mm** (a 2K-ish 35 mm frame), **7.1/255 at 96 px/mm**, **11.1/255 at 193 px/mm**. The reason is that f50 lives at 35-74 cycles/mm and a 2K render never reaches those frequencies, so most of the visible change at normal sizes comes from the adjacency term rather than from f50. **The f50 correction earns its keep at scan resolution, not at 2K** -- worth knowing before anyone judges it on a preview. Schema unchanged, `film_names.txt` unchanged, no ListBox shift; data-only rebuild | — | done | correctness, and it raised C24 |
| ~~C38~~ | ✅ **DONE 2026-08-31 — AND TWO OF THE THREE DISAGREEMENTS DID NOT EXIST.** This row asked for one adjudication and a re-read of 5245's tail. The re-read settled 5245; the other two settled themselves the moment the evidence was looked at, and neither was about the film. ⚠ **5218 — WRONG DOCUMENT, and this project's own reader is why.** `5218.pdf` is the four-page BROCHURE H-1-5218; the adopted set is cited to the six-page TECHNICAL DATA sheet H-1-5218t, a different file. `_sign_y_ticks` could not open the technical sheet at all, because that page emits its entire content TWICE at identical coordinates and the duplicate `0.0` label read as an unresolvable macron pair — so C37 fell back to the brochure and recorded *"5218's page is 3, not the 4 this row gives"*. **The page was right; the FILE was wrong.** With coincident duplicates dropped and a second candidate frame tried, the technical sheet reads and agrees at **0.033 / 0.082 / 0.056** — inside the band the other eight occupy. The adopted raster set STANDS; the brochure is registered separately as `5218_brochure` at its own 0.241 / 0.210 / 0.138, now labelled for what it is: Kodak redrew the panel narrower for marketing, red peaking at 640 nm against 650. ⚠ **5231 — WRONG CURVE, and the reader picked it.** The caption-to-curve rule assumed captions sit ABOVE their curves. True on H-1-5222, false on H-1-5231, where both sit UNDER — so the `D=0.3` caption selected the `D=1.0` curve. There is no double-hump problem and never was: paired by SENSITIVITY instead (a lower density criterion needs less exposure, so its curve is higher everywhere — physics, not layout), the trace peaks at **400 nm exactly where the adopted set does** and agrees at **rms 0.063**. The adopted set STANDS. ⚠ **5245 blue — THE ONE REAL DEFECT, and the plot decided it.** The stored tail ran −0.60/−1.15/−1.80/−2.45/−3.10, steps of 0.55/0.65/0.65/0.65. Rendered at 6× the yellow-forming curve plunges from its 460 nm peak and **leaves the bottom of the frame (log S −2.005) at 527.5 nm**, where the stored tail still had it 0.8 decades higher; at 530 nm that tail claims a value ABOVE the last point the curve is drawn at, which no reading of this plot can produce. **Replaced by the trace.** Red and green are untouched — they agree at 0.064 / 0.155 and carry 690/700 nm samples the vector grid cannot supply, so re-adopting them would have lost data to buy nothing. ⚠ **THREE FURTHER FINDINGS FELL OUT OF THE FIXES, none of them queued.** (1) **5248 was never blocked on METHOD.** C37 recorded it as needing "a three-black-curve separator"; `extract_mono` separates three black curves on five other sheets. The real blocker was the frame: the first candidate that CALIBRATES on that page stops at 648 nm and excludes the cyan trace, and no second candidate was tried. It reads now and is registered. (2) **5248's green record had a one-sample notch** — −1.67 / **−2.02** / −1.62 across 450–470 nm, a 0.40-decade spike on a flank the page draws smooth. Corrected to the traced −1.62; every other sample left alone. (3) `verify.py` gained the two **defect-shape** checks these two repairs suggested — no one-sample notch on a flat run, no straight-line tail — and **both fired on real stored data when written**, which is the only reason to trust them | nothing | done | 5245's blue record replaced, two adopted sets confirmed, one "unreachable" panel unblocked, and the reader defect that caused half of it fixed |
| ~~B4~~ | ✅ **CLOSED 2026-08-30, and the blocker did not exist.** The row recorded "THE BLOCKER IS AXIS CALIBRATION, AND IT IS NOT SOLVED… 14 vertical intervals across a 0–3 density axis and 29 horizontal across six decades". ⚠ **Both counts were one short.** The outermost gridline on each axis is fainter than the interior ones (457 and 819 dark pixels against 461 and 819+) and fell under the ink threshold; at 0.45 of the span p7 gives **31 verticals over log E −4.00…+2.00** and **16 horizontals over D 0…3** — 0.2 per interval, exactly round, frame corners at the labelled extremes. All four plates then calibrate the same way. ⚠ **And the embedded raster is stored UPSIDE DOWN** behind the page's own flip transform, which `get_images` bypasses — one fact that produced negative gammas, records stacked red > green > blue, reversed spectral peaks and a D-min that appeared to RISE towards the red, all at once and none of them looking like an orientation problem. ⚠ A third bug: every plate draws its legend in the same inks as its curves, so the red record came out peaking at 2.83 D on a plate whose red curve never passes 1.5. **ADOPTED:** the ORANGE MASK — traced base+fog **0.173 / 0.531 / 0.962** against a stored 0.300 / 0.280 / 0.290, i.e. *no mask at all* on a masked negative, where every other masked stock encodes one (CINESTILL_800T is within 0.09 D); and the **spectral dye density pair** into the v14 carrier that B1 created and left empty (neutral+dmin pairs 10 → 11). **CROSS-CHECKS, unchanged:** spectral peaks 390/578/659 nm against the stored 390/560/660 (worst 18 nm), gammas 0.536/0.598/0.584 against 0.545/0.560/0.580 (inside 7 %). ⚠ **REFUSED: the MTF.** TI0835A gives f50 46.9 c/mm against a stored 24/28/33, and it is captioned **"Diffuse visual"** with ONE curve while `MTFSpec` holds three PER-LAYER numbers. `verify.py` fired four separate checks and every one was right, including "the measured red records of the NEGATIVE family stay clustered near 36". Recorded, not stored; a field for the film's overall visual MTF would be needed. Guarded by `ti0835_plates.py` | — | done | — |
| ~~C39~~ | ✅ **CLOSED 2026-08-30 — the only open row that was rendering wrong, and it is fixed.** New `TakingFilter` carrier (schema v20) on **two** fields, because two different facts needed separating: `SpectralSensitivity.measured_through` is what the plotted curve ALREADY INCLUDES, and `FilmProfile.taking_filter` is what the profile's look ASSUMES in front of the lens. On both IR stocks these differ — the curve is bare, the intended use is filtered — and conflating them was the schema silence. Applied in `film_sim` **before the guard and before the collapse**, on BOTH sensitivity paths; ⚠ filtering only the guard's path would have left the guard judging a filtered emulsion and the weights derived from a bare one, a split invisible on every stock the guard refuses. Result for `ROLLEI_INFRARED_400`: behind the sheet's own **715 nm** filter its peak moves 410 → **720 nm**, out-of-reach 0.028 → **1.000**, the guard refuses, and the authored red-dominant (0.52, 0.20, 0.28) is used instead of the near-flat (0.349, 0.315, 0.336). Mirrored in `AlgoSpectralSensitivity.cpp`; `spectral_mono_parity` **68/68, zero guard gaps**. ⚠ **Only ONE of the two stocks can be fixed and that is the honest outcome:** both sheets name a filter, only Rollei's prints a wavelength. Konica's TDSB-701 says "a Kenko R-1 red filter" — a RED filter, cutting far shorter than 715 nm — so borrowing Rollei's number would be wrong as well as unsourced. Konica carries the designation with an empty model and stays refused on its own bare curve, **by record now rather than by luck**. The `verify.py` check that pinned the defect in place has been REVERSED, not deleted | — | done | — |
| ~~C40~~ | ✅ **DONE 2026-08-30, batched with C30/C33 because both are algorithm-source edits sharing one archive re-issue.** The two tests Python already had are ported into `AlgoSpectralMonoWeights()`, both measured on the profile's own stored samples rather than the 360-730 nm render grid, returning false on refusal so `Algo_07_Sim.cpp` falls back to the authored triple exactly as `film_sim` does. `KONICA_INFRARED_750` no longer renders at a blue-dominant (0.1611, 0.1931, 0.6458); both engines now use its authored (0.55, 0.15, 0.30). `spectral_mono_parity.py` reports **68/68 with no guard gaps**, and `--allow-guard-gap` was REMOVED from the build registration rather than left in as a safety net -- a re-opened gap now fails the build instead of printing a line somebody has learned to skip. ⚠ ROLLEI_INFRARED_400 is still not refused and that is C39, a missing carrier, not a threshold to tune | nothing — the Python implementation is the specification | small — one function, ~20 lines | **medium-high — it is a wrong render that ships today, and the cost of the fix is an afternoon at most** |
| C14 | ⚠ **NEW 2026-08-20. `KODAK_EKTAR_125` is a real gap and the document on file cannot fill it.** `Kodak Ektar 125 - Jack and Sue Drafahl.pdf` is *PHOTOgraphic*, September 1989, pp. 80–82 — a **magazine review** with **no sensitometry whatsoever**: no rms, no gamma, no Dmin/Dmax, no MTF, no resolving power, no spectral data, no reciprocity, no processing table. A profile from it would mean inventing every stored number, so **none was created**. What it DOES document is the full layer construction — **eleven layers**, two blue (one fast, one slow) *"slightly thicker than the single blue, slow layer found in Ektar 25"*, **an extra interlayer between the two green layers** credited with the sharpness gain, a first-of-its-kind **magenta coupler** that raises T-grain speed, and **two red layers with DIAR couplers** — i.e. a complete `LayerStack` and a `CouplerSpec` rationale for a film with no curves. **Wanted: the Kodak publication for Ektar 125 (1989–1994).** ⚠ Not to be confused with `KODAK_EKTAR_100` (E-4046, 2008/2016), which the database already holds and which is a different, later film | a Kodak datasheet not held | small once a sheet exists | medium — a named, dated, structurally documented stock with zero sensitometry |
| ~~C15~~ | ✅ **DONE 2026-08-25d, by a schema bump the owner authorised (v11 → v12).** Both sub-questions answered rather than dodged. **(b) NO TRANSFER**: an intermediate film's coupler set is chosen for archival stability and printing, not camera exposure; one film cannot establish a rate for 160 (method rule 18); and the same refusal was made for the 7266 σ(D) two days earlier. The table is stored on the DI film and NOWHERE else, and `verify.py` asserts no other stock inherits it. **(a) NOT CONVERTED**: "years to 0.10 loss" is a RATE and `AgingSpec` stores a STATE, so instead of forcing years into a field documented as a 0-1 fraction — the category error that stalled this item once already — `PrintStock` gained TWO fields: `aging` (an `AgingSpec`, which `PrintStock` had never had at all) and a NEW `DyeStabilitySpec`. Both INERT, both appended after every v11 field, so a v12 database renders bit-identically to v11. ⚠ **THE PUBLISHED FIGURES ARE CENSORED AND ARE STORED AS CENSORED**: Kodak prints ">100" for every record outliving the test, so `censor_years = 100.0` with a field at 0.0 means "greater than the bound". Storing the number 100 would let later arithmetic average a bound as a measurement. Exactly two entries at 21 °C are finite — **yellow 86 years to a 0.10 density loss, blue 77 years to a 0.1 D-min gain**; the 7 °C column is entirely censored and is not stored. **`KODAK_VISION3_DI_2254` added as the 11th print stock**, appended so no index moves, with curves TRACED from the full sheet H-1-2254 p3 by the new `di_2254.py` (474 samples/record off a 680x704 raster, axis residuals 0.0015 decade / 0.0000 D, fit rms 0.006-0.012 D). ⚠ The fitted gammas **1.05 / 0.96 / 1.04** are the free physical check: an intermediate film exists to change nothing, and nothing in the trace was told that. ⚠ The CATALOGUE-NUMBER COLLISION with `EASTMAN_5254_1968` is now asserted by `verify.py`, not just noted | — | done | the last all-zero data family now has one sourced member, and `PrintStock` has an aging carrier at all |
| C7 | ⚠ **NEW 2026-08-18.** A **temporal grain law** now has a source and no carrier: Honjo 1989 §4 states that at 24 fps the eye integrates ≈0.2 s ≈ 5 frames, so perceived granularity falls by **1/√5 ≈ 0.447** versus a frozen frame. This engine renders **motion** for an AE/PR plugin, so the still-frame grain amplitude is arguably 2.24× too strong in playback — but "correct" depends on whether the plugin is judged frame-by-frame or in motion, which is a product decision, not a physics one | a decision | small once decided | **high perceptual impact, zero research cost** |
| ~~C5~~ | ✅ **DONE 2026-08-20, owner-approved.** `EASTMAN_5247_1983` re-tiered **3 → 1**, tagged `[T1/T2]` because its three tone curves are hand-fitted while everything else on TI0835 is printed. ⚠ **A mixed tag does not resolve itself:** `_provenance_for`'s regex accepts only a bare `[T1]`/`[T2]`/`[T3]`, so `[T1/T2]` falls through to `_UNTAGGED_TIER` and, if unlisted, silently to 3 — which is what has happened to `KODAK_VISION2_500T_5218` and `_200T_5217` (`[T1/T3]`, both at tier 3 to this day, **found while doing this and now a separate small item**). Both new mixed-tag entries are listed explicitly and guarded | — | done | correctness, plus one latent bug class surfaced |
| **D. Blocked on a measurement only the owner can make — cheap** ||||
| D1 | absolute base+fog: **one `--empty-gate` frame** — a scan of the empty gate with NO FILM in it, on the GCMC/UF15 scanner, at the same settings as the film batches. ⚠ **NOT a photograph and not a command**: `--empty-gate` appears only in this project's comments and was never implemented. ⚠ **AND THE 2026-08-31 SCAN REVIEW MADE THIS SHARPER AND MORE VALUABLE.** 50 of the SVEMA-FOTO-64 scanner frames contain an unclipped film-base strip — 226 px = 1.85 mm at 122.7 px/mm — reading **250.24 ± 2.91**, i.e. **0.0082 D ± 0.0051 against scanner white, range 0.0035–0.0340 D**. That reproduces the documented 0.008–0.028 D from the pixels and adds the number the project lacked: **the scanner's per-frame auto-exposure contributes ±0.005 D of scatter**, which is the noise floor under every density reading from this rig and is why the σ(D) estimator is uninterpretable. ⚠ **CORRECTION, same day: the base is measured but probably NOT rescuable.** This row first said an empty-gate frame would convert all 50 strips to absolute density retroactively. The review's own numbers overturn that — the strip of ONE physical film base ranges **235.8 to 252.9** across the 50 frames, and two frames of the same base read 241 and 250. A single piece of base does not vary by 9 levels, so the UF15 is re-exposing PER FRAME; if it is, that batch has no single white point and no later gate frame can calibrate it. **Take the gate frame IN the session whose scans are to be absolute**, and let its first job be to answer the prior question: fixed exposure or per-frame? ⚠ If the scanner is auto-exposure-only with no manual mode, D1 is not achievable at all and the reference must come from D2 instead | 1 minute, free | — | **high** — makes density absolute for every stock on that scanner, and 50 frames are already waiting for the reference |
| D2 | scanner transfer + noise floor: **one step-wedge scan** (Stouffer T2115 / Kodak Q-13) | ~$30–50 | — | **high** — the only way to separate emulsion σ from scanner σ; also settles the anisotropy question |
| ~~D3~~ | ✅ **DONE 2026-08-31 — the test ran and +0.30 did not survive it.** The blocker was "frames not in `SAMPLES/`"; the owner supplied 854 scans on 2026-08-31 and all 132 Tasma frames were read in place. ⚠ **104 OF 132 ARE BIT-EXACTLY NEUTRAL** (R == G == B at every pixel) — greyscale-converted, so they contribute a hard zero by construction. The 28 that carry colour give a midtone cast of **R−G = +7.72 ± 10.72**: ⚠ **the scatter is larger than the mean**, so the 28 do not agree with each other that a cast exists, and an emulsion's silver tone cannot vary that way between frames of one film. ⚠ **This also retires the "+8.6 and +15.6" observation** the profile hoped might rescue the value — those are two draws from exactly this distribution, and the comment records that the LARGER was chosen. That is selecting a tail. ⚠ **And the statistic this row named is the wrong one**: `max |R−G|` over a pictorial frame measures the most saturated OBJECT in the scene, not the emulsion — run over ORWO NC21 it returns 137, correctly reporting that a colour negative is a colour photograph. **`TASMA_FN_64.silver_tone` reverted +0.30 → 0.00**, owner-approved, by the precedent that reverted SVEMA_FOTO_65. ⚠ `TASMA_OCH_45`'s +0.15 is now the last nonzero silver tone in the database, carries no source, and is **flagged not changed** — there are no OCh-45 scans, and refuting by analogy is what the 2026-08-18 pass refused to do | nothing | done | one wrong render parameter removed |
| **E. Ready, moderate effort — source on disk** ||||
| ~~E0~~ | ✅ **DONE 2026-08-18.** All 11 profiles re-verified digit for digit against their own sheets. **5 values changed**, 3 exact agreements confirmed and pinned, 2 plausible "corrections" rejected after checking, 1 split leftover found, Sehlin/Kennel settled as **1985**. Full record: `RESULT_2026-08-18c_E0_reverify.md` | — | done | **the 5247 grain figure was 2.6× above Kodak's own printed bound** |
| ~~E0b~~ | ✅ **DONE 2026-08-18.** All three vector sets extracted; two new audit scripts registered in the build. **7 values changed across 4 profiles**, and one of them (`5285` rms 3.0 → 13.1) is a 4.4× grain correction. Full record: `RESULT_2026-08-18d_E0b_vector.md`. Original scope note below | — | done | 3 dye sets, the first measured reversal σ(D), 1 measured f50 |
| ~~E0b-orig~~ | ✅ **DONE 2026-08-25d.** The 5285 MTF half closed on 2026-08-25g (see the history file); the remaining half — **7239's spectral sensitivity** — closed with the panel-finder widening, which is why the two were one task. ⚠ **THREE INDEPENDENT DEFECTS, NONE IN THE SOURCE.** (1) The caption finder could not see SHORT WORDS: `rot_labels` calls a word rotated when `(y1-y0) > 1.6*(x1-x0)`, true of "SENSITIVITY" and **false of "LOG"**, so the caption "LOG SENSITIVITY" never matched. Replaced by PyMuPDF's per-line writing direction, which is not a heuristic. **Corpus sweep over all 2159 pages: 6 pages reachable by the old rule, 21 by the new one.** (2) The frame picker took the nearest frame and stopped; on 7239 two rects qualify and the tick labels sit BETWEEN them. Candidates are now tried in order until one calibrates. (3) ⚠ **The y axis has a minus sign that is NOT IN THE TEXT LAYER** — it runs 2.0/1.0/0.0/−1.0/−2.0 with the negatives drawn as overbars, so PyMuPDF returns "1.0" and "2.0" twice. The old `setdefault` dropped the duplicates and **happened to keep the right branch**; a sheet emitting the lower branch first would have calibrated MIRRORED, perfectly collinear, inside tolerance, with every stored sensitivity sign-flipped and nothing able to see it. Ticks are now signed by position about the zero tick and the five-tick collinearity test confirms it (0.46 pt worst residual). ⚠ **7239 IS ALSO THE FIRST SET READ WITHOUT THE INK RULE** — its panel is printed entirely in black, so assignment rests on the absorption bands, the ascending peak order and the panel's own in-frame captions (Yellow- 394 / Magenta- 550 / Cyan- 702 nm), one fewer independent check than an inked panel gets, stated in the profile rather than left to be inferred. Peaks 410 / 560 / 660 nm. ⚠ And this sheet **PRINTS ITS DENSITY CRITERION** ("Process: VNF-1", "Density: 1.0", "Densitometry: E.N.D.", "Effective Exposure: 1.4 seconds") where the four older Kodak sets carry a "D 0.2 above dmin" printed on no sheet in this corpus | — | done | 15 more sensitivity panels are now findable; only 7239 has been read |
| ~~E1~~ | ✅ **DONE 2026-08-29. 8 profiles changed, 2 new audits, the processing axis DOUBLED — and THREE FACTUAL ERRORS IN THIS ROW, each of which changed how the work had to be done.** ⚠ **(1) THE KODAK PLOTS ARE NOT VECTOR.** This row said "vector line art ... 21 drawing objects on the Tri-X page — render at 600 dpi and machine-trace". The objects are real — 30 of them — and every one is a single `'l'` item whose endpoints share a y: **horizontal table rules** left by the Acrobat Paper Capture plug-in that OCR'd the scan, none of them inside a plot frame. Each page carries one **JPEG-2000 grayscale raster at 150 dpi** and the curves are in it, so the Kodak half is a SCAN-GRADE raster trace. ⚠ **(2) THE AGFA SHEET IS 2004, NOT 2003** — `F-PF-E4, 4th edition, 08/2004` on p12 (its pp5-9 footer still says `F-PF-E3`, a third-edition leftover, recorded not corrected); byte-identical to `AGFA/FPD1e.pdf`. ⚠ **(3) THE PAGE NUMBERS ARE EACH ONE LOW** — Portrait 160 is printed p6 and Optima 100/200/400 share p7 — and only **4 of the 11 films listed on p1 get plotted panels at all**. ⚠ **AND "7 STOCKS [T2]→[T1]" IS TRUE OF FOUR.** The three Agfa profiles were ALREADY `[T1]`; what this did for them is make a claim they were already making TRUE — their derived provenance said `fitted_from='datasheet_curve'` while the stored curves were `_neg()` calls, the family default toe and shoulder with a dmin and gamma written beside them. **KODAK 1952** (`kodak_1952_curves.py`): all **20** printed (time, gamma) pairs re-derived from the drawn curves, **18 within 2 %**, all within 4.3 %, monotone in development time, association by CONTRAST ORDER so the agreement is a free test. ⚠ **The ESTIMATOR is part of the result**: the fixed 0.3-1.2 net-density window that reproduces H-1-5222 reads **5 % low** here (0.931 against a printed 1.00) because the 1952 toes are far longer; the steepest 0.6-decade chord — which is what a straight-line gamma IS — returns 0.9992. Gammas adopted at the sheets' own recommended tank development, 68 F: VERICHROME **0.780 → 0.744** (D-76 16 min), PANATOMIC-X SHEET **0.700 → 0.852** (+22 %), TRI-X SHEET **0.680 → 0.832** (+22 %), ORTHO-X SHEET **0.720 → 0.800** — and Ortho-X needs no interpolation at all, 9 min is a printed label. Where interpolation IS used it cannot matter: every recommended time falls between two ADJACENT printed curves, so linear and log-time bracket within **0.004 gamma**, and Tri-X's own Time-Gamma inset independently reads 0.827 against the interpolated 0.829-0.832. ⚠ **NEGATIVE RESULT: base+fog is NOT readable from these plots.** Unlike H-1-5222 the curves never plateau — at the leftmost drawn column Tri-X's family is still climbing 0.213/0.183/0.158/0.122/0.102 — so `base_fog` is **0 on all 20 points**, the schema's "not stated". Each plot's separate drawn **"Base Density"** line traces to 0.064/0.050/0.080/0.109, which is **support without fog**: a LOWER BOUND on dmin, not dmin, so all four stored dmin values are unchanged (and all four clear their bound). **AGFA 2004** (`agfa_2004_curves.py`): vector, dash-keyed by layer, keying checked against the printed Blue/Green/Red words AND against peaks ascending b<g<r; axis fits at **0.00 nm** residual. 12 colour-density curves fitted to the corpus's six-parameter ToneCurve at **rms 0.005-0.016 D**, each cross-checked against an independent steepest-chord slope. ⚠ **The shoulder softness had to be CONSTRAINED and that is the most instructive failure here**: unconstrained, Portrait 160's red fitted at `shoulder_k` = **3.0 x** `toe_k` with rms 0.005 D — a good fit — and a `gamma` of 1.018 the curve does not have, because when the two softplus ramps overlap that heavily the parameter STOPS BEING the observable slope. Fitting the ratio inside ToneCurve's own documented 1.4x band fixed it. 3 new spectral sets (census **73 → 76**). ⚠ **AND OPTIMA 100, WHICH THIS ROW DOES NOT LIST, WAS CORRECTED**: the stored 1998 RASTER set puts its red peak at **650 nm**, the 2004 vector page draws the peak at **615-620** with a SHOULDER at 645 — while blue (470) and green (550) agree EXACTLY. Two layers agreeing and one moving 35 nm is a mis-read peak, not a reformulation; the alternative (the 1998 sheet names the film OPTIMA II 100) is recorded in the profile, not dismissed. ⚠ **f50 REFUSED**: the Sharpness panel peaks at **109-114 %**, above what an MTF can be, so its overshoot IS the adjacency and is adopted (unit-free); but its abscissa says "Lines per mm" and whether that is line pairs is **open item G6**, so f50 stays unstored — see the evidence now filed there. **ProcessingFamily 22 points/4 stocks → 42/8**, guard moved not loosened. Render impact up to **0.45 D / 12 eight-bit codes**. Full record: `doc/RESULT_2026-08-29_E1_kodak1952_agfa2004.md` | — | done | 4 genuine [T2]→[T1] upgrades, 2 gamma errors of 22 % corrected, a wrong spectral peak found by cross-reading, and a queue row whose three premises were all wrong |
| ~~E2~~ | ✅ **DONE 2026-08-31 — AND THIS ROW'S HEADLINE WARNING POINTED THE WRONG WAY.** The row's trap was: *"the sheets plot the equivalent energy needed … so log sensitivity is its NEGATION — the peak of the curve is its LOWEST drawn point"*, with the caution that a wrong sign yields a mirrored set passing every band and ordering check. ⚠ **The caution is right and the diagnosis is backwards: negating them is what would have produced the mirror.** The prose describes the MEASUREMENT; Polaroid plots its reciprocal. Four independent proofs, and the last two are decisive: (1) the 667 edition captions its axis **"Spectral Sensitivity (cm²/erg)"** — area per unit energy, so the sheet states the direction itself; (2) every curve falls steeply at its long-wave end, which inverted would mean peak sensitivity exactly where the data stops; (3) **peak plotted value rises with film speed across all four sheets** — EI 50 → 9.8, EI 100 → 15.6, EI 400 → 98.0, EI 3000 → 233.1 — where the inverted reading has the ISO 3000 film needing fifteen times MORE light than the ISO 100 one; (4) the two already-adopted sets are stored UNNEGATED and this reader reproduces them to **rms 0.034 and 0.027 decades**, which a mirror cannot do. ⚠ **Following this row would have mirrored two correct sets while "fixing" them.** **Adopted: POLAROID_52 and POLAROID_55_PN_NEG**, the two new pan sets the row promised; 664 and 667 confirmed and left alone. ⚠ **And the row's second claim was also wrong**: *"667 and 55 … the label sweep returns one or two labels each — they need their own windows"*. They need no windows. All four panels are found by one rule that keys on the AXIS LABELS instead of a page region, and 667 returns four y labels at a dead-uniform 40.3 pt per decade. A fixed window was the blocker. ⚠ Also corrected: the criterion string on all four, from `log_energy_for_neutral_density_0.75` — which names the measurement and contradicts what is stored — to `log_reciprocal_erg_cm2_neutral_D0.75`. New registered audit `polaroid_spectral.py`, 5 checks | nothing | done | 2 new pan sets, 2 confirmed, and a prescribed mirror avoided |
| ~~E3~~ | ✅ **CLOSED 2026-08-31, and the Konica half was never an acquisition — the owner's `PROFILES/KONICA/` holds nineteen sheets.** The Gevaert half had already closed (682 Figs 7/8/10/11/12 across G3, G7 and C1e). What the three Konica files needed was a RASTER reader, and `konica_raster.py` is it: every plot in `IMP50.pdf` and `INF750.pdf` is an embedded bitmap with no paths and no tick text, so calibration is geometric off the printed grid and all seven panels re-detect their own gridlines before a curve is traced. ⚠ **The bitmaps are also stored UPSIDE DOWN** — rotating them 180° leaves the text mirror-reversed, which is how the flip announces itself. **ADOPTED for `KONICA_IMPRESA_50`:** the three Status M characteristic curves (softplus fits at rms 0.009–0.015 D) and the visual-filter MTF. ⚠ **Its Dmin triple was a FAMILY TEMPLATE and wrong in blue by 0.32 D.** It held 0.20 / 0.62 / 1.00; `KONICA_VX_100` holds 0.21 / 0.63 / 1.02 and `KONICA_CENTURIA_SUPER_400` 0.22 / 0.65 / 1.05 — three stocks, one shape, round numbers, all marked `fitted_from='datasheet_curve'`. The sheet reads **0.199 / 0.557 / 0.676**, and p3's minimum-density spectrum sampled at the ISO 5-3 status M band centres 640/540/450 nm reads **0.190 / 0.552 / 0.691** — two figures, two pages, agreeing to 0.005–0.015 D and jointly refuting the stored blue. ⚠ **MTF: f50 is 64.9 c/mm, not the estimated 72**, it overshoots to **121.4 % at 6.88 c/mm**, and the power-law rolloff fits at q 2.20, rms 0.019 against the Gaussian's 0.039. The per-layer 72/80/88 had to go: the sheet prints ONE curve captioned "through visual filter", so a pooled f50 is what was measured, and `verify.py` now carries the exclusion and the property that licenses it. **ADOPTED for `KONICA_INFRARED_750`:** the curve at Konicadol DP 6 min / 20 °C — the sheet's own standard time for the developer its footnote equates to KODAK D-76 — plus the first non-empty `ProcessingSpec` this stock has ever had. ⚠ **Gamma moved 0.72 → 1.70 because all FIFTEEN printed curves are steeper than the value held**, the flattest being Konicadol Fine at 4 min with 0.814. **NOT obtainable and recorded as such:** a dye triple (p3 draws two NEUTRAL spectra, not three dye curves), any per-layer Konica MTF, and INF750's absolute spectral level (its p1 panel's y axis carries no tick labels at all). ⚠ **`professional_160.pdf` is closed as unusable**: all four pages extract zero characters, and `NotFound.md` had already established that its one technical page matches no stock in the database. Full record: `RESULT_2026-08-31c_reconcile_B3_E3.md` | — | done | **two stocks off their own sheets, and a family template caught** |
| E4 | Eastman 1942 MP book — Super-XX 1938, Plus-X 5231 predecessor. ⚠ **Corrected 2026-08-31: the blocker cell said "nothing" and this checkout does not hold the book.** It is on the owner's machine as `KODAK/Kodak - [1942] - Eastman Motion Picture Films for Professional Use.pdf`; stage it and the row is ready | one staging step, not a document | medium | low–medium |
| E5 | SMPTE 1985 Sehlin/Kennel Fig. 11 | **depends on** reconciling its axis units with the 48 µm diffuse-RMS convention | medium | medium — only measured granularity-vs-exposure for a colour negative in the archive |
| **F. Blocked on material NOT held** ||||
| F1 | second source for B&W σ(D): Bayer JOSA 54 (1964), Wilder JOSA 62 (1972), Trabka JOSA 63 (1973) | not on disk — acquisition | — | medium — would settle a measurement-vs-theory conflict |
| ~~C41~~ | ✅ **DONE 2026-08-30, owner chose option A.** Callier is wired into the two places that actually consume it, both twins moving together. **AlgoSolveAnchors** gains `scannerSpecular` and applies the factor at the same two points `film_sim` does -- the reversal branch's `mixed` and the negative branch's `d_mid`. **AlgoStage12b_Callier** is a new pointwise stage between 12 and 13, in place on the stage-12 planes, so it needed no scratch buffer and no re-costing of `AlgoMemHandler`. ⚠ **THE HEADER'S OWN CONSUMER LIST WAS WRONG ON TWO OF THREE ENTRIES AND WRONG IN THE DIRECTION THAT COSTS WORK** -- it named call sites that do not exist, so anyone wiring from it would have CREATED divergences: `AlgoNeutralMidDensity` applies nothing in Python and is deliberately left alone, and Callier is its own stage rather than part of `AlgoStage12_DyeImpurity`. Corrected in place. ⚠ **AND TWO RECORDS OF THE ONE MEASUREMENT BEHIND THE WHOLE ARGUMENT DISAGREED** -- the header said mid grey moved "+54/255 ... against 22 %", `film_sim.py` says "+48/255 ... a few per cent". Nothing settles which transcription is right, so the exact figure is no longer quoted anywhere; the ORDER of the two effects, which both records agree on, is what the argument rests on. Re-measure before quoting a number. **Guarded:** `cpp_parity` gains a STAGE family that drives `AlgoStage12b_Callier` and `AlgoSolveAnchors` themselves rather than the law beside them -- 2475 probes, worst stage 3.83e-07, worst solve 2.77e-07, inert at specular 0, 0 colour stocks moved, 272 monochrome rows moving at full specular. ⚠ The film half of the product is still a class estimate (1.3 / 1.25 from a generator rule, no document), which is why the control still ships at zero | — | done | — |
| ~~F2~~ | ✅ **DONE 2026-08-30, owner decision: "measured rise".** Reversal default 0.7/1.0/0.5 -> **0.21/1.00/2.97**, the mean of the two measurements, which reverses its DIRECTION. Colour-negative default 0.4/1.0/1.2 -> **0.81/1.00/0.68**, agreeing with all eleven measurements. ⚠ **MONOCHROME NEGATIVES DELIBERATELY NOT CHANGED** -- all eleven measured negatives are Kodak COLOUR CINE, and no document in this corpus carries a granularity-versus-density curve for a named B&W NEGATIVE, so applying their triple to the 55 B&W negatives is the class jump method rule 18 forbids. Opened as **F2b**. ⚠ **AND THE RENDER IMPACT IS ZERO, WHICH THIS ROW AND MY OWN PROPOSAL BOTH OVERSTATED**: the wiring honours a shape only when `sigma_shape_measured` is set, the heuristic never sets it, and that has been true since 2026-08-18. These anchors are a documented placeholder read by no renderer. Correcting them makes the description true, not the render different. ⚠ A first attempt also set `sigma_shape_peak=1.38@0.75` from the measurements; verify.py refused it on all 55 stocks, correctly -- the peak is only reachable through the measured-anchor path, so on an unmeasured stock it is a number the data model cannot honour |  ⚠ **INVESTIGATED 2026-08-26 — UNBLOCKED SINCE 2026-08-18, AND THE SCOPE IS 4x WHAT THIS ROW CLAIMED.** The blocker read "**depends on C1**, and inert until then"; **C1 closed 2026-08-18**, so this sat actionable for eight days. The row also says "the 103-stock default" — live count is **147**, against 13 measured. ⚠ **AND BOTH DEFAULTS ARE CONTRADICTED IN DIRECTION BY EVERY MEASUREMENT OF THEIR OWN CLASS — 146 of 147 stocks, not the 34 previously recorded.** Measured: 11 NEGATIVES all **FALL** toward dmax (dmax/mid 0.50–0.90, mean 0.68) while **112 of 113** heuristic negatives **RISE** (1.00–1.80, mean 1.24); 2 REVERSALS both **RISE** (2.83–3.10) while all 34 heuristic reversals **FALL** (0.50 exactly). ⚠ **One real mitigation, for the negatives only:** NO unmeasured stock sets `sigma_shape_peak` (0 of 147) while ALL ELEVEN measured negatives do, at **1.20–1.62 located 0.65–0.80 of the way up the scale**. So the negative default's "1.20 at dmax" is standing in for an INTERIOR PEAK the triple cannot express — the rise is real, it is in the wrong PLACE, and the fall after it is missing. The reversal default has no such excuse: it is simply backwards. ⚠ **NOTHING CHANGED, and this is now an OWNER DECISION on the same footing as C16**, because every option moves 146 renders. Two `verify.py` guards pin the contradiction so it cannot be absorbed silently. **What the measurements would support:** negatives → a class triple from 11 samples (all Kodak colour cine, so applying it to the 51 MONOCHROME negatives is a class jump that needs saying); reversals → direction is unanimous but **n = 2**, and method rule 18 forbids a class estimate from one sample — two is thin, so the honest minimum there may be a FLAT 1.0/1.0/1.0 rather than a confidently wrong sign | **an owner decision on the form** (the measurements are in hand) | small once decided | **high — 146 stocks carry a shape whose direction every measurement of its class contradicts** |
| **F2b** | ⚠ **NEW 2026-08-30, from F2, and it is an ACQUISITION dressed as a measurement.** F2 corrected the sigma(D) placeholder for colour negatives and reversals from eleven and two measurements. The **55 MONOCHROME negatives got nothing**, because every one of those eleven measurements is a Kodak COLOUR CINE stock and no document in this corpus carries a granularity-versus-density curve for a named B&W NEGATIVE. ⚠ The one measured B&W shape in the database, `KODAK_TRI_X_REVERSAL_200`, is REVERSAL, and the 2026-08-25 adoption note already refused to generalise it for exactly this reason. So those 55 stocks keep a triple that now points the opposite way to every negative that has ever been measured here -- not because it is believed, but because nothing better is evidenced. ⚠ What would close it: one granularity plot for a named B&W negative at a stated aperture. BBC T-101 Fig. 18 gives Wiener spectra for six B&W emulsions but at FIXED density, so it cannot yield a shape; Mees Fig. 302 is the right KIND of plot and the 2026-08-25b analysis showed its regime is not commensurable with the 48 um convention this file stores. Most likely source: a Kodak or Ilford technical publication that prints granularity against density, which none of the B&W sheets in this corpus does. ⚠ **A NAMED LEAD, 2026-08-31, from the corpus sweep — and it is NOT a closure.** `ILFORD/AN ANALYSIS OF FILM GRANULARITY.pdf` on the owner's machine is **BBC Engineering Monograph No. 54, August 1964, K. Hacking** — a different document from the BBC T-101 this row already discusses, absent from this checkout, and it prints Wiener spectra for four named negative emulsions (Tri-X, Plus-X, Pan F, Eastman TIB). ⚠ **It still cannot close this row**, and for the reason the row already gives: its Fig. 8 is at a FIXED density of 0.48 above base, so it yields no shape. What it does carry is a **law** — its eq. (4) gives signal-to-noise ∝ D^−0.5 for negative-type emulsions, notes that "recent granularity measurements on a range of Eastman Kodak emulsions, reported by **Higgins and Stultz**" put the exponent nearer −0.4, and states that the REVERSAL case is numerically greater at −0.6 to −0.7. That last clause is the first independent support this file has for the thing F2 refused to assume: negatives and reversals do not share a shape. **The real lead is Higgins and Stultz**, which BBC 54 cites and this corpus does not hold | a document this corpus does not hold — most likely Higgins & Stultz, cited by BBC Monograph 54 | small once found | **medium -- 55 stocks, but the value is INERT until the shape is wired for unmeasured stocks, so this is about truthfulness of the record rather than about renders** |
| **M1** | ⚠ **THE DOCUMENT ARRIVED 2026-08-31 AND THE ROW IS STILL OPEN — BUT THE BLOCKER HAS CHANGED SHAPE, FROM ACQUISITION TO CONFIGURATION.** What this row asked for was *"ONE spectral sensitivity curve set for a colour PRINT stock (Kodak 2383 or 5383) … zero of the 11 print stocks carry one"*. **It is now on file and read.** ⚠ **And it was never an acquisition**: the owner's corpus held three 2383 files all along, two of which carry a VECTOR spectral panel on p6; this checkout simply did not have them. ⚠ **Nor was the panel "laid out differently"**, which is what `spectral_vector`'s header had recorded for a fortnight — its axis runs **−3.0 to +1.0**, because a print emulsion is slower than a camera negative, and the tick reader's value window was `0 <= v <= 6`. Two of five labels survived it. One character. **DONE:** `PrintStock.spectral` (schema v22); `KODAK_2383_RELEASE` populated — the first print stock in the database with a spectral sensitivity — layers peaking 470 / 550 / 680 nm, cross-checked between two editions of the sheet at rms 0.015 / 0.032 / 0.031; both editions registered in the audit. **AND THE DERIVATION CLOSES:** `M_reader · M_status⁻¹` now computes for all 9 adopted panels and lands **0.048 to 0.116 from identity**, against raw status off-diagonals reaching **+0.24** — this project's own argument for refusing the raw table, measured rather than asserted, and pinned in `EXPECTED_STAGE12`. ⚠ **WHAT IS STILL MISSING, AND IT IS NOT A DOCUMENT.** The reader 2383 describes is a release PRINT FILM. **164 of 165 profiles set `default_print=SCAN_DI`, so their reader is a scanner, and NOT ONE stock in this database renders through 2383.** Storing the matrix would state that a stock's reader is a film it is never printed on — the same substitution the module already refused once, wearing better sourcing. What closes M1 now: **a scanner's channel responses**, or a profile that actually renders through a print stock we have | a SCANNER response, or a print-referred profile — no longer a document | the derivation and the carrier are built | **high — 97 stocks still share one symmetric crosstalk shape, and the half that was missing is now measured** |
| ~~M2~~ | ✅ **CLOSED 2026-08-30, and the premise was wrong.** Queued as "re-trace two panels"; taken back to the PDF there is **nothing to re-trace**. Each panel is **five separate vector stroke paths**, so no tracking and no crossing to get wrong; peak positions assign them unambiguously (5218: 380 / 446 / 537 / 680 nm plus the neutral); the stored arrays reproduce the paths (5218 magenta 0.302 stored against 0.276 read off the path at 640 nm); and the panel closes internally — `neutral = 0.478 C + 0.556 M + 0.693 Y + 0.949 Dmin` at relative rms **0.0061**. ⚠ That identity has four free parameters and absorbs a contaminated curve, so it is not evidence. What disqualifies 5218 is physics: its traced magenta sits at ~0.30 of peak across the whole red and **RISES** 0.302 → 0.322 between 640 and 680 nm, and an absorption band decays away from its peak. Nor is the excess the orange mask — fitting 5218-minus-5217 as `k·Dmin + c` gives k = −0.36 with an rms barely better than the difference. **Both refusals stand, now on a physical criterion rather than an outlier argument**, recorded at the refusal site in `dye_matrix_from_spectra.py`. Also staged `5218-Vision2-500T-H-1-5218t.pdf` into the corpus, so `dye_density.py` stops skipping 5218 | — | done | — |
| ~~M2b~~ | ✅ **CLOSED 2026-08-30 in the same batch as M2.** The Midscale Neutral and Minimum Density traces are now extracted and STORED for every panel whose frame yields them cleanly — `KODAK_VISION2_50D_5201`, `5205`, `5218`, `EASTMAN_EXR_50D_5245` and `EASTMAN_EXR_200T_5293`. Required a schema field: `SpectralDyeDensity.normalisation_neutral` (v19), because the dyes are peak-normalised while the pair is as printed and one `normalisation` string could not mean both — the reason family C had been finding these traces, validating them and then throwing them away. Two immediate returns. ⚠ **`Neutral − Dmin = k(C+M+Y)` with the three k EQUAL became a general validator and failed three panels at once** — 5218 (spread 0.31), 5245 (0.32) and **`EASTMAN_EXR_200T_5293` (0.21), which had passed the sign test, the ratio bounds AND the Soviet cross-check and had already been adopted into `_MEASURED_DYE_MATRIX`.** The adopted set is now 9, not 10. ⚠ **And `d_dmin` on a masked colour negative IS THE ORANGE MASK** — 15 stocks now carry it, the first spectral record of the mask anywhere in the database. Enforced in `dye_matrix_from_spectra.py`, reported by `dye_density.py`, guarded in `verify.py` | — | done | — |
| ~~M3~~ | ✅ **CLOSED 2026-08-30 — evaluated AND wired, both twins.** Mees printed p644 gives Silberstein & Tuttle's `10^-Dsp = E·10^-Ddiff + (1-E)·10^-(β·Ddiff)`, the book defining **E** as the fraction of scattered light the reader accepts and **β** as unity plus the scattering-to-absorption ratio — ⚠ **C22's film × geometry split, in print since 1942**, three pages from a figure already in the corpus. Measured against the law it replaced, over 68 monochrome stocks × 5 settings × 11 densities: **identical at both endpoints** s = 0 and s = 1, and **up to 0.2121 D apart in between**, 1483 of 3740 points differing by more than 0.002 D. ⚠ The two agreed precisely where a hand test would look and diverged everywhere a user would actually dial: the old law interpolated the MULTIPLIER, the published one interpolates TRANSMITTANCE, which is what light does. **Wired:** `film_sim.callier_net` (one definition), `AlgoCallierNet` / `AlgoCallierLut` / `AlgoCallierLutAt` in `AlgoCallier.hpp`, stage 12b on the table, and both `Algo_08_Sim.cpp` twins on the exact law in the solve. ⚠ **Inertness is a property of the GUARD, not of the law's arithmetic** — at E = 1 the law is -log10(10^-d), mathematically the identity and not bit-exact (5.6e-17), so all three implementations test `callier_is_inert` and return early; the pixel pass is asserted bit-identical at specular 0. ⚠ It does **not** fix the toe: expanding for small D gives Q → E + (1−E)β, a constant, while FIG. 179 measures Q collapsing to 1.04 at D 0.055 — the measurement wins and the traced shape is still wanted. Parity: Callier law 1.43e-07, STAGE 1.97e-07, SOLVE 2.77e-07, 340 monochrome rows moving, 0 colour. Pinned by `callier_silberstein_tuttle.py`, which now also asserts that the SHIPPED law is still the book's | — | done | — |
| ~~M3b~~ | ✅ **CLOSED 2026-08-30 in the same batch as M3, and the answer was NOT to split the twins.** The problem was real: the law needs two `pow` and a `log10` per channel per pixel and neither has an AVX2 intrinsic. Solved with a **1-D lookup over NET density** — 1025 entries over −1.0 to 5.0, linear interpolation, end-slope extrapolation — built identically by `film_sim.callier_lut`, `AlgoCallierLutBuild` and both flavours, so parity holds by construction; measured interpolation error against the exact law **2.2e-07**, inside the family's existing tolerance. ⚠ **The solve evaluates the law EXACTLY and only the pixel pass uses the table**, because the solve touches a handful of scalars where two `pow` cost nothing. ⚠ `AlgoStage12b_Callier`'s header note said "if this ever grows a branch or a table it MUST be split into twins like Algo_11" — G3 gave it both, and the note was **overridden with the reason recorded in place**: the branch and the table are now the LAW ITSELF, shared by the solve, both flavours and Python, so splitting would create two spellings of one law. Accepted cost: the stage no longer auto-vectorises, and is scalar in both builds at a non-zero setting. It ships at zero, where it returns before touching a pixel. ⚠ **The AVX2 twin of the solve is not compiled by any audit** in this flattened tree, so `TWIN_LAW_TOKENS` gained `Algo_08_Sim.cpp` and the textual twin check is now the guard on it | — | done | — |
| F3 | 5 nm spectral re-trace beyond ACROS | nothing, but measured benefit is 0.4–1.1 % | high | **low** — future-proofing, described as such |

**Why B1 is first among the unblocked work:** the source is on disk, the extraction path is
already proven twice (5285 and 2383, both validated against their own neutral trace), the
schema field exists and is validated, and the roadmap rates it the largest colour-fidelity
gain available from data already owned. It is also small — 5 sheets, not 57 pages, because
most vector dye pages belong to stocks this database does not hold.

**Why F3 is last:** it is the only item on this list whose own entry says the benefit is
0.4–1.1 % and that it is future-proofing against illuminants the engine does not use.

---



---

## Archived 2026-08-31 — the previous "RECOMMENDED ORDER, 2026-08-26"

Superseded: thirteen of the rows it ranks have since closed.

### RECOMMENDED ORDER, 2026-08-26 — derived from dependencies, not from the value column

⚠ **Two things fell out of re-reading every live row, and both change the ranking.**

**FINDING 1 — `F2` HAS BEEN UNBLOCKED FOR EIGHT DAYS AND NOBODY NOTICED.** Its blocker reads
"**depends on C1**, and inert until then". **C1 closed on 2026-08-18.** Same class of staleness as
E0b-orig, and this one is worse, because F2's premise has also grown: the row says "the 103-stock
default" and the live count is **147 stocks** sitting on a σ(D) heuristic against 13 measured.
⚠ **And 34 of those 147 carry a heuristic this project has MEASURED to be wrong in direction** —
the reversal default 0.7 / 1.00 / 0.50 falls toward dmax, while the one measured B&W reversal σ(D)
(C29, TRI-X Reversal) RISES. `verify.py` already asserts that contradiction and cites it. So F2 is
not a tidying item; it is the largest population of knowingly-wrong numbers in the database.

**FINDING 2 — `C2c` AND THE MTF HALF OF `C19` ARE ONE PIECE OF WORK, and its evidence base has
tripled.** Both are the same defect: `adjacency_um` disagrees with the measured overshoot frequency,
in the same direction, on every stock checked. C19 was written when **four** stocks had a traced
MTF. Twelve do now. C19 also states the real problem plainly — `adjacency`, `edge_strength`,
`edge_um` and `radius_um` are **four parameters describing ONE inhibitor diffusion length, currently
set independently** across 87 hand-typed `CouplerSpec` literals with no registry and no tier. The
MTF half needs nothing from anybody.

#### Tier 0 — owner, minutes to days, and it unblocks the most

| Order | Item | Why first |
|---|---|---|
| **1** | **D1** + **D2** | One free frame and one ~$40 wedge scan. Together they are the stated blocker for the STRENGTH half of both **C18** (36 reversal stocks, "the biggest single unpinned colour number") and **C19**. Nothing else on this list unblocks two high-value items for two days' pocket money. |
| **2** | **C7** | One product question — is the plugin judged frame-by-frame or in motion? Zero research: the source (Honjo 1989 §4) is read and the number is 1/√5 = 0.447. High perceptual impact on every frame of every render. |
| **3** | **G5** | One re-scan of three printed pages you already hold. Unblocks **G2** (high) and upgrades two profiles from `[T3]` estimates to traced curves. |

#### Tier 1 — me, no decision needed, in this order

| Order | Item | Why here |
|---|---|---|
| **4** | ~~C37~~ | ✅ closed 2026-08-29 as a registered audit. It yielded **no new sets** — the promise of 13 was wrong — but turned a hand sweep into 11 enforced cross-checks and surfaced **C38**. |
| **5** | **F2** | See Finding 1. 147 stocks, unblocked since 18 Aug, and 34 of them measurably wrong in direction. The 13 measured shapes are now enough to derive a class heuristic per stock KIND rather than one global triple. |
| **6** | **C2c + C19 (MTF half)** | See Finding 2. Twelve traced MTF curves make this measurable now; collapse the four independent parameters onto the one diffusion length they all describe, and re-derive `adjacency_um` from the traced overshoot. |
| **7** | **B4** | ~~B1~~ closed 2026-08-26; its untracked residue is now B4 — the four TI0835 raster plates for `EASTMAN_5247_1983`, all colour-coded, two of them new data. Blocked only on axis calibration. |

#### Tier 2 — me, moderate effort, nothing blocking

**E4** · **T1** · **T2** · **T3** · **G2** (after G5) — ~~B3~~ and ~~E3~~ closed 2026-08-31,
~~E2~~ closed 2026-08-31, ~~G4~~ and ~~E1~~ both closed 2026-08-29. ⚠ E4, T1, T2 and T3 all need
their source STAGED from the owner's machine first; none of the four files is in this checkout.

#### Tier 3 — waits on a decision or a document

**C16** (one number, but it changes every render — yours) · **C18 / C19 strength** (after D2) ·
**G6**, **C14**, **F1** (acquisition — all three re-proved absent 2026-08-31 against the owner's
full 475-file corpus) · **E5** (⚠ NOT acquisition: its paper is held, misfiled under a 1983 name,
and its blocker is the axis-unit reconciliation) · **C4** (yes/no, low value) · ~~D3~~ (closed) ·
**C23** and **F3** last — C23 is the least evidenced item on the list and F3's own measured benefit
is 0.4–1.1 %.

⚠ **Two standing decisions are not rows in this table and should be**, because both are held on the
owner and both are wider than any single item: the **spectral criterion on 16 profiles** (a "D 0.2
above dmin" printed on no sheet in this corpus, while 5222 and 7239 both print theirs), and the
**MTF rolloff architecture in C++** (no FFT exists there, so `FilmMtfResponse` has no stage caller).
