# NotFound.md §4 worked through — 2026-08-16

The owner asked for §4 ("identified, extractable, not yet extracted — gaps in our work") to
be resolved rather than restated. All five items were executed. Three are now closed, one is
partly resolved with a named residue, one is blocked on materials the owner controls.

## 1. Vector curve sets — the "~30 sheets" figure was a large undercount

An exhaustive sweep of `PDF/PROFILES/**`:

| stage | pages |
|---|---|
| pages carrying any drawing or image | 9 413 |
| pages with ≥1 path of ≥30 `l`/`c` items | 937 |
| surviving the fill/closePath + geometry filter — **genuine vector curves** | **516** (in 245 PDFs) |
| rejected as logo / glyph / filled art | 421 |

The two documented traps are now quantified rather than feared: 28 pages are the Kodak
logo (~30-item filled cluster), 111 are filled wordmark art, 266 are glyph outlines of
headline text, 16 are full glyph text blocks of 600–1350 items. Of the 516 genuine pages,
**152 in 111 PDFs carry spectral sensitivity**; 238 carry characteristic curves, 125
granularity, 119 MTF, 54 spectral dye density.

### Adopted: 14 stocks gained a measured spectral curve

Extraction is from exact PDF vector coordinates — only the axis calibration is fitted, and
it closes to **≈0.27 nm rms on wavelength and ≤0.012 log on sensitivity** against the
sheets' own printed tick labels.

`KODAK_ULTRAMAX_800` (E-7024 p3), `KODAK_ULTRAMAX_400` (E-7023 p4), `KODAK_EKTAR_100`
(E-4046 p4), `KODAK_PORTRA_160` (E-4051 p4), `KODAK_PORTRA_800` (E-4040 p4),
`KODAK_PORTRA_100T` (E-2468 p5), `KODAK_GOLD_100` + `KODAK_GOLD_200` (E-7022 p4),
`KODAK_TRI_X_400TX` (F-4017 p7), `KODAK_TMAX_100` (F-4016 p8), `KODAK_TMAX_P3200`
(F-4001 p7), `KODAK_PLUS_X_125` (F-4018 p9), `KODAK_T400CN` (F-2350 p6), `KODAK_BW400CN`
(F-4036 p5).

**Stocks holding a spectral curve: 53 → 67 of 143.**

Decisions recorded in every profile comment:

- **10 nm sampling**, the project convention — and *coarser* than the source everywhere
  (vector points every 0.65–3.3 nm), so this decimates rather than invents. The sparse
  sheets (Portra 160/400, Ektar 100, 3.33 nm spacing) must **not** be resampled below 5 nm.
- **Peak-normalised per layer**, with the sheet's absolute ordinate preserved as
  `peak_abs_logS` in the comment so erg/cm² sensitivity is recoverable losslessly.
- **−4.00 is not a measurement** — it means the sheet does not plot the curve there. The
  measured span is stated per layer.
- **Grid renormalisation is disclosed**: on nine layers the true plotted peak falls between
  10 nm grid points, so the decimated array was shifted by 0.01–0.06 log to satisfy the
  schema's peak = 0.0 rule. Each affected profile names the shift.
- Two-criterion B&W sheets (D = 0.3 and D = 1.0): the **D = 0.3** curve is stored, being
  the speed-defining one. Dashed curves state that their gaps are interpolated.
- `KODAK_GOLD_100`/`_200` share one curve set because **the sheet plots one** for the
  family — the sheet's assertion, not our assumption.
- `KODAK_PORTRA_400` already held a curve; it was independently re-derived and agreed to
  mean 0.03 log. Revalidation only, no change.

### Not entered, and why (the honest residue)

`KODAK_TMAX_400` (F-4043 p7) — the two criterion curves extract with inconsistent shapes
(peaks 528 vs 570 nm); needs visual confirmation before entry. `AGFA_APX_25/100/400` —
frames drawn as `qu` quads, calibration needs a hand frame. `AGFA_VISTA_200` — one page
carries the 100/200/400/800 family, per-product assignment needs the legend. Ilford
HP5+/Delta 3200/FP4/Pan F — the figure is a **wedge spectrogram outline with no numeric
axis ticks**, so there is nothing to calibrate against (plus FP4 vs FP4 Plus identity).
Polaroid 664/667/52/55 — decade-log ordinate, needs a convention decision.
`KODAK_TECHNICAL_PAN` — multi-plot pages. `KODAK_ULTRA_COLOR_100UC/400UC` — E-4035 is not
on disk.

## 2. Image-only OCR bracket — closed entirely

`NewGevacol_Neg_682.pdf` is the Vervoort & Stappaerts SMPTE-Journal paper (89(9), Sept
1980, pp 650–652) — already the cited source; OCR plus visual re-reading confirmed 100 ASA,
3200 K, ECN-2, γ 0.57, Status M, triacetate with removable carbon-black backing, DIR
couplers in the green and red layers, and the 12-element stack. Its RMS table is
**relative only** (σ_D ∝ 1/√n, no aperture, no magnification) and can never yield a 48 µm
figure. `Verpoort_Stapp1980_NewGevacolNeg682.pdf` is a byte-identical duplicate.

`centuria_pro_400.pdf` is **barren** — a March-2003 brochure whose only figure is the prose
"ISO400", and for CENTURIA **PRO** 400, a different product from our CENTURIA SUPER 400.
`professional_160.pdf` yields process and format data but **prints no ISO at all** and
matches no DB stock. Konica IMP50/INF750 **were never image-only** — both have full text
layers and were already mined; only their figures are raster.

## 3. KODAK DATA BOOK vol 5 — closed as a documented dead end

All 346 pages (1150–1495) swept: **zero RMS granularity, zero numeric gamma**, resolving
power on only 8 pages, speeds as ASA/BS/DIN tables, curves as scanned line art. The three
resolving figures it does print (Super-XX Sheet 60 lp/mm, Plus-X Miniature 65 lp/mm,
Super-XX Aero 35 lp/mm at 1.5:1) belong to 1950s–60s UK products that do not map onto any
held stock without a generation graft, so **nothing was entered**. Recording the sweep
stops anyone re-reading 346 pages hoping for numbers that are not there.

## 4. Portra NC/VC (E-190) — closed as a deliberate decision

Fully documented in E-190 (PGI 36/40/44/48/48) but a different emulsion generation from the
2010s films we model. Not merged, by design; candidates stay in `next_week_task.md`.

## 5. Zhurba 1990 — blocked on materials

Eight owner-supplied spreads were read on 2026-08-16 (they produced the first ORWO data in
the corpus). The rest needs a local copy of the book; the online edition's webp pages
return empty to `web_fetch` and no other route is attempted, per the project's web rules.

## Verification

- `verify.py`: **125 PASS / 2 FAIL** — the same two pre-existing failures. A new permanent
  guard asserts all 14 vector-extracted spectral curves are present, at 10 nm, with their
  vector-extraction provenance intact.
- `validate_all()` green (it caught the peak-normalisation issue that produced the
  disclosed grid shifts); C++ regenerated and compiles clean; reports regenerated; master
  document and Russian mirror updated.
