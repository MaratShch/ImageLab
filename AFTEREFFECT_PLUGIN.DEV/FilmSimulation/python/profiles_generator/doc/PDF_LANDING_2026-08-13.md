# PDF landing report — 2026-08-13 rescan of PDF/PROFILES

Owner request: rescan the folder, identify what is new, assess extractability,
report. Inventory: **513 files**; ~**447 not yet referenced** by Found.md /
NotFound.md (the registry diff overcounts slightly where old entries are cited
by title rather than filename, but the bulk is a genuinely new landing).
Extraction itself is NOT done in this pass — this documents what landed, what
each cluster can feed, and in what order to digest it.

## Method

Every cluster sampled with PyMuPDF (first 3 pages, character count): >300
chars = true text PDF, else scan. Scans sampled visually at full resolution.
Per-file classification only for the samples listed; sibling files in a
cluster assumed alike until opened.

## 1. KODAK/kodak-professional-b-w-film/ — 88 JPG page scans (owner flagged)

Kodak Publication **F-5, "Kodak Professional Black-and-White Films"**.
**Dating correction:** the owner's note said 1950–1960; this edition is
**late 1970s / early 1980s** — the Tri-X sheet carries the boxed
"ISO 400/27°" marking (ISO speed notation, ~1979 onward), cover price $5.95,
"WITH REVISED FILM DATA SHEETS" banner. Same publication family as the 1950s
F-5s, later revision. Scan quality is excellent: ~4700×6500 px, clean, fully
legible down to curve-axis ticks.

Structure: narrative handling/processing section plus a **DS data-sheet
insert** (pale-blue pages DS 1…DS ~30), one stock per sheet. Confirmed by
sample: DS 3 Contrast Process Ortho 4154, DS 18 Tri-X Pan. Expected from the
series: Plus-X Pan / Panatomic-X / Verichrome Pan / Royal Pan / Ektapan /
Super-XX sheet / High Speed Infrared and siblings.

**Why this is the highest-value item in the landing:** each DS page carries
what our database lacks END TO END — the processing axis:
* characteristic-curve FAMILIES at 3+ development times, each labelled with
  its contrast index (A.5.7 SEN.2);
* contrast-index-vs-time curves for up to 6 developers (PRC.3);
* development tables at 5 temperatures × 2 agitation regimes (PRC.1, PRC.4);
* resolving power at BOTH target contrasts (RES.3);
* filter factors per illuminant; safelight; base-density notes (the Tri-X
  sheet states grey vs clear base differ by 0.15 D — a BAS.6 data point).

No text layer — extraction is visual digitisation, same workflow as the 1942
Eastman book. OCR-indexing the 88 pages to find every DS sheet is the first
step; queued in DIGITIZATION_QUEUE.md.

## 2. KODAK loose datasheets — ~200 true-text PDFs

Sampled files all carry full text layers. Coverage in three bands:

* **Motion picture, direct profile upgrades:** technical-information sheets
  for the ENTIRE Vision line already in the database — 5201/5203/5205/5207/
  5213/5217/5218/5219/5245/5246/5248/5274/5279/5293/5296 — plus 5294/7294
  (Ektachrome 100D), 5272 internegative, 2383/2393/2302 print stocks
  (PrintStock upgrades), 2374 sound film, Double-X 5222, Plus-X 5231,
  Tri-X reversal 7266, **Ektachrome 7239 (Daylight)** — the 16 mm sibling of
  our tier-3 EASTMAN_EKTACHROME_5239: first vendor sheet touching that
  family. CRT recording films (TNM) are out of scope but archived.
* **Still line:** TMax 100/400/P3200 (multiple editions 1999–2018), Tri-X
  f9-1999 and f4017 editions, Plus-X f8-1997, **Verichrome Pan f7-1996**
  (upgrades KODAK_VERICHROME_1952 lineage), Portra family (e190/e4040/e4051,
  editions 2006–2016), Ektar e4046 (2008–2016), Gold/UltraMax/Elite/
  Ektachrome E100G/E100VS/EPP/E200/64T/320T/P1600, Vericolor, Ektapress,
  BW400CN, T400CN, HIE infrared, Ektapan. Multiple EDITIONS of the same
  sheet (e.g. 4× TMax-100, 3× Portra) — valuable as MF.5 formulation-revision
  evidence, not redundancy.
* **Cross-cutting references:** `estimating_on-film_image_resolution_v8.pdf`
  (20 p, text) — method source for RES.3/RES.1 interpretation;
  `transmision of wratten filters.pdf` (8 p, SCAN) — spectral transmittance
  of the taking filters (SPC.5/filter registry); `copying-6.pdf`,
  Processing Modules 1 and 11 (PRC.1 chemistry definitions).

## 3. POLAROID — 57 data sheets, true text

Type 55 P/N, 52/53/54/57/59, 665, 669, 664-class pack films, 600/SX-70/
Time-Zero/Spectra, ID/UV materials. The database has 3 Polaroid profiles and
2 Polaroid formats; this is the vendor documentation for them plus ~20
candidate stocks. Text layers confirmed (55fds, 665fds sampled).

## 4. DUFAYCOLOR — 8 files, mixed

1938 Dufaycolor Manual (25 p, SCAN), Carson 1934 Kinotechnik paper (SCAN),
the Dufay patent GB 262,386, a 60 p book scan, and — the notable part —
**three measured-OD multispectral JPGs** (`measuredODs_MSI_NSMM_*`) from the
Timeline of Historical Colors project: measured spectral density of surviving
Dufaycolor material. DUFAYCOLOR_1937 is currently a [T3] reconstruction;
these are its first measured data. COL.1-class evidence for a reseau
material.

## 5. AGFA — Agfacolor Neu subfolder + 10 new loose files

* `Agfacolor Neu/`: Schultze/Hörmann 1951, Behrendt/Köllner 1950, Gröger 1961
  (German, SCANS, no text layer) plus **two extracted-OD multispectral JPGs**
  (BArch material) — measured spectral densities for AGFACOLOR_NEU_1936 /
  AGFACOLOR_NEG_TYPE_B_1943, both currently reconstruction-tier.
* Loose: agfa_scala.pdf (Scala 200X reversal — NEW stock candidate, text),
  APX25 datasheet, Aviphot Pan 80, CP30 colour print 2011, Alliance-IR,
  Gevachrome 902 (SCAN), New Gevacolor 682 papers (Verpoort/Stapp 1980) —
  GEVACOLOR_1952's successor family.

## 6. SOVIET + SOVIET STANDARDS — 6 new items

* **«Современные фотоматериалы и их обработка»** — 717 p, TRUE TEXT,
  reference book on 2002–2003 materials from Agfa/Fuji/Kodak/Konica with
  processing detail. Late-era cross-check source for the modern stocks.
* ГОСТ 9160-91 (26 p, text) — 21st GOST in the folder, not yet extracted.
* «МАТЕРИАЛЫ ФОТОГРАФИЧЕСКИЕ» (14 p, text) — standard, unassessed.
* «Фотография в прошлом, настоящем и будущем» — 184 p SCAN, no text layer.
* `D_en / K_en / R100_en` — one-page ENGLISH data sheets: "panchromatic
  surveillance film 500 ISO" (polyester 75 µm, sens. limit 690 nm),
  "vintage aerial film 100 ISO" (polyester 100 µm, 700 nm), "reversal
  panchro PAPER 100 ISO". These read as modern repackager sheets
  (Astrum/Svema-successor class), not period Svema documents — treat as C3
  for any historical profile, C1 only for the modern products themselves.

## 7. Other manufacturers — new stock candidates and upgrades

* **FOMA:** fomapan-100/200/R-100 sheets (FOMAPAN_100 in DB gets vendor
  data; 200 and R100 reversal are new candidates).
* **KONICA:** full VX 100/200/400 + VX-S line, Centuria Pro 400/160(?),
  chrome/csuper series, IMP50, R100 — database has only KONICA_INFRARED_750;
  this is a whole family of candidates with true-text sheets.
* **ROLLEI:** R3 (TARoR3_e, 22 p — upgrades ROLLEI_R3), Infrared (upgrades
  ROLLEI_INFRARED_400), Retro (TARRete), Superpan, PAN25, development guide.
* **ORWO modern:** Wolfen NC400/NC500 colour, UN54, NP100, DP31, PF2, P400 —
  new-era ORWO, all text. Plus rgschwind_digital.pdf (5 MB, unassessed).
* **KENTMERE:** PAN 200 sheet (new candidate; 100/400 already in DB) +
  fresh editions of PAN 100/400.
* **MACO:** IR820c Aura + TA line — new candidates.
* **FERRANIA:** FP3011 sheet + «Curve caratteristiche e sensibilità
  spettrali» (near-no text — likely curve plates; visual).
* **FUJI/ILFORD:** ~25 sheets that the registry diff flags as new — mostly
  additional EDITIONS (Delta/FP4+/PanF+/SFX/XP2 2002-vs-2018, Superia/
  Provia/Velvia/Astia/Sensia/Pro-160/400H/800Z). Edition pairs = MF.5
  revision evidence.

## 8. Root-level references

* **«The Permanence and Care of Color Photographs»** (Wilhelm/Brower, 761 p,
  text) — THE standard reference for dye fade and dark-storage kinetics:
  feeds the entire AgingSpec/UNI.7 axis with measured data.
* `5_1_FilmBaseGuide_2020.pdf` (text) — base identification: BAS.1/BAS.2.
* `film.pdf` (24 p, text), `aimm.it2.18.1996.pdf` — unassessed.
* `spectrumAsset-1@2x.png` — single image, unassessed.

## Priority recommendation (extraction order)

1. **Kodak F-5 DS sheets** — only source in the archive with curve families
   over development for the professional B&W line; feeds the processing axis
   the gap analysis ranks first. Visual digitisation.
2. **Vision-line + 7239 + print-stock technical sheets** — text PDFs, cheap
   to extract, directly upgrade ~20 existing tier-2/3 profiles.
3. **Dufaycolor + Agfacolor Neu measured-OD JPGs** — measured spectral
   density for two reconstruction-tier colour families; small effort.
4. **Wilhelm book** — AgingSpec data (currently 11 fields, all estimates).
5. Polaroid/Konica/Foma/ORWO/Maco candidates — new stocks, owner decision on
   which families are wanted before the database grows again.

## Registry actions taken

* This report created; DIGITIZATION_QUEUE.md updated (F-5 88 pp, Wratten
  filter scan, Dufaycolor/Agfacolor OD plates, Фотография-184p scan).
* Found.md left untouched: entries there are made per-document at extraction
  time, and nothing was extracted in this pass.
