# Changes 2026-08-02 — Soviet reference-book pass

Database: 83 -> 89 stocks (reversal 20 -> 21). verify.py: 67/67 PASSED.
C++ regenerated (film_profiles.hpp / film_profiles.cpp), g++ -std=c++17
object compile clean. film_names.txt written (89 quoted names, order =
C++ std::vector order) by the new gen_film_names.py.

## New stocks (source: Gurlev 1986; details in SOVIET_EXTRACTION_2026-08-02.md)
- SVEMA_FOTO_32   ("SVEMA FOTO-32")   B&W neg, S 32 GOST, γ 0.8, R 135 lin/mm
- SVEMA_FOTO_130  ("SVEMA FOTO-130")  B&W neg, S 130, γ 0.8, R 100, 580 nm cut
- SVEMA_DS_4      ("SVEMA DS-4")      colour neg, unmasked, 5500 K, S 45
- SVEMA_TSNL_32   ("SVEMA TSNL-32")   colour neg, masked, 3200 K, S 32
- SVEMA_TSNL_65   ("SVEMA TSNL-65")   colour neg, masked, 3200 K, S 65
- TASMA_OCH_45    ("TASMA OCH-45")    B&W reversal, S 45, Dmax 1.9 (+ Iofis 1980)

## Renames / aliases
- ORWO_UT18 -> ORWO_CHROM_UT18 (official leaflet W 746: "ORWO CHROM-FILM
  UT 18"); old aliases kept, new chrom/color aliases added.
- SVEMA_FN_64: aliases foto-65 / svema foto-65 etc. added (Gurlev p296
  Foto-65 = same emulsion class); tier raised 3 -> 2 (printed corroboration).

## Data additions
- _PROVENANCE_SOURCES: Russian-book citations (original title + English
  translation + translated author) for SVEMA_FN_64, SVEMA_FOTO_250 and all
  six new stocks; ORWO_CHROM_UT18 leaflet citation.
- _RESOLVING_POWER: Soviet printed R (lines/mm, GOST resolvometry, stored
  as printed) for FOTO-32/FN-64/FOTO-130/FOTO-250/DS-4/TSNL-32/TSNL-65/OCH-45.
- verify.py count assertions 83/20 -> 89/21 (commissioned additions).

## File cleanup
25 files moved to PDF/PROFILES/DELETE_CANDIDATE (byte-identical duplicates,
URL-only www.txt pointers, non-film paper/toner/label docs, and the DjVu
original superseded by its PDF conversion). Nothing deleted.

## Second pass, same day — Chibisov appendix tables + Dufaycolor NSMM
verify.py 67/67 PASSED after every change; C++ and film_names.txt
regenerated; g++ object compile clean.

- Chibisov 1988 Appendix Table I (book p157-158, rotated pages) fully
  transcribed; all appendix tables II-XIV surveyed (details in
  SOVIET_EXTRACTION_2026-08-02.md).
- TASMA_OCH_45: gamma 1.35 -> 1.50 (Chibisov prints gamma_rec 1.6 for
  OCh-45; adopted value sits at the Chibisov-weighted end of the Gurlev
  1.1-1.6 window); R 100 -> 110 mm^-1 (product row supersedes Iofis class
  minimum); f50 31 -> 34 c/mm; Dmin conflict (0.06 vs 0.08) recorded,
  0.08 kept.
- R conflict recorded for the Foto line: Chibisov prints 116/92/75/70
  against Gurlev's 135/110/100/82 (Gurlev kept; both cited).
- Chibisov citations added to SVEMA_FOTO_32/FN_64/FOTO_130/FOTO_250,
  TASMA_OCH_45; new citations for EASTMAN_5247_1974 and EASTMAN_5294_1983
  (Table VIII: RMS granularity and MTF@30 figures recorded, grain NOT
  adopted -- cross-era metric equivalence unverified; 5294 printed green
  MTF@30 0.65 matches the profile's f50_g exactly).
- Table IX confirms every adopted DS-4 / TsNL-32 / TsNL-65 value from
  Gurlev; extra stocks available on request: TsNL-90, TsOD-16/32,
  TsO-65, TsO-T-90L.
- DUFAYCOLOR_1937 (owner instruction): reseau filter_matrix rebuilt from
  the measured NSMM Bradford absorbance curves (items 11948/11951/11960):
  T = 10^-A, band-averaged, uniformly rescaled x4.05, rows normalised to
  sum 0.80 to keep the neutral-grey reconstruction invariant (within-row
  crosstalk ratios preserved exactly). Tier 3 -> 2. NSMM citation added.

## Third pass, same day — schema v3: digitised spectral sensitivity
verify.py 70/70 PASSED (3 new spectral checks); C++ regenerated and
object-compiles; film_names.txt regenerated (content unchanged, 89 names).

- SCHEMA_VERSION 2 -> 3: new SpectralSensitivity struct appended to
  FilmProfile (Python + generated C++). Relative log10 sensitivity,
  peak-normalised per layer, -4.0 floor sentinel, per-stock wavelength
  grid, criterion (y-axis convention of the source plot) and source
  (author, document title/code, ORIGINAL document release date; Russian
  sources cited with original title + English translation). Empty struct
  = legacy spectral_weights / taking_matrix path; other 86 stocks
  unaffected.
- Pilot curves digitised (10 nm pitch, visual transcription, accuracy
  stated per stock): FUJI_NEOPAN_ACROS_100 (AF3-095E sec. 12, 2001),
  KODAK_VISION3_250D_5207 (H-1-5207, rev. March 2026, film 2009; three
  layer curves, sheet-absolute peaks recorded), KONICA_INFRARED_750
  (undated Konica TDS; unlabelled axis read as linear, assumption stated
  in criterion; 380-830 nm grid).
- Generated film_profiles.hpp/.cpp now stamp generation timestamp
  (ISO-8601 UTC) + schema version; film_names.txt deliberately carries
  neither. Spectral tables emitted as std::vector<double> with exact
  float64 shortest-roundtrip literals; pre-v3 fields stay float32 (values
  authored as short decimals — reproduced exactly; the real precision
  bound is transcription accuracy, not literal rounding).
- Each spectral-bearing profile gets a "Spectral curve source" comment in
  the generated .cpp above its literal.

## Fourth pass, same day — machine-traced H&D curves (digitize_plot.py)
verify.py 70/70 PASSED; C++ regenerated + syntax-checked; film_names.txt
regenerated (89 names, unchanged).

- NEW TOOL digitize_plot.py: 600 dpi plot rendering, frame/gridline
  auto-detection, seeded ink-centroid curve tracing, hand-rolled
  Nelder-Mead ToneCurve fitting (numpy+Pillow only). Transcription error
  bounded by printed line width: RMS 0.003-0.007 D.
- KODAK_VISION3_250D_5207: all three characteristic curves machine-traced
  from H-1-5207 p3 (1426 samples/layer, full -8..+8 stop range) and
  refitted [T1]. Sheet-absolute Status M dmins adopted -> stock moved to
  dmin_ladder mask encoding (added to _DMIN_LADDER).
- FUJI_NEOPAN_ACROS_100: Microfine 15-min (G-bar 0.65) curve traced from
  AF3-095E p5 (1092 samples) and refitted [T1]: measured fog 0.122,
  straight-line gamma 0.690, toe shape measured; mid-grey anchor
  preserved; shoulder beyond the printed range flagged [T3 there].
- doc/DIGITIZATION_QUEUE.md: full production queue for the remaining
  H&D / spectral / MTF / dye-density plots across the archive, with
  binding method rules (machine-trace first, citation format, residual
  reporting, conflict recording).

## Fifth pass, same day — spectral batch 2 (10 more stocks)
verify.py 70/70 PASSED; C++ regenerated + object-compiles;
film_names.txt regenerated (89 names, unchanged).

Two parallel digitization agents, pixel-calibrated against printed
gridlines (+/-0.03-0.05 log). Spectral coverage 3 -> 13 stocks:
- Fuji: VELVIA_50, PROVIA_400X, SENSIA_100 (three-layer reversal curves,
  J/cm^2 D1.0-above-Dmin criterion), NEOPAN_1600 (pan, scanned 1995
  sheet).
- Kodak: VISION3_50D_5203, VISION3_200T_5213 (tungsten blue tail
  captured), PORTRA_400 (E-4050 2010; real red-layer shelf -1.8 at
  490-560 nm and green-layer blue tail -1.1 transcribed as printed),
  TRI_X_REVERSAL_200 (pan).
- Ilford/HARMAN: HP5_PLUS_400, DELTA_3200 (tungsten wedge spectrograms,
  Nov 2018; Delta's flat 550-660 nm + 697 nm red reach captured).
Every curve is per-emulsion (no vendor-shared shapes); B&W stocks carry
their own measured pan curves. Citations with original release dates in
Python fields and generated .cpp comments.

## Sixth pass, same day — spectral batch 3 (22 more stocks: 13 -> 35)
verify.py 70/70 PASSED; C++ regenerated + object-compiles;
film_names.txt regenerated (unchanged); FilmCurves.md regenerated.

Three parallel digitization agents, overlay-verified tracing:
- Agfa: APX 25/100/400 (Datenblatt sheets, 08/1995), OPTIMA II 100
  (Range of Films brochure).
- Kodak classics: KODACHROME 64 (E-55, Dec 1996, K64-specific plot),
  EKTACHROME 64 (E-8, 2005), EKTACHROME 160T (E-144, 2007),
  EKTACHROME 100D 5285 (via the 5294 sibling sheet, noted),
  EASTMAN DOUBLE-X 5222 (H-1-5222), EXR 500T 5296 (via the 5298 sibling
  sheet TI2082, 1993, noted).
- Konica: IMPRESA 50, VX 100, CENTURIA SUPER 400/1600, CHROME CENTURIA
  100, CHROME R-100 (CNK-4/CRK-2 criteria as printed).
- Rollei: R3 (superpan to ~715 nm), INFRARED 400 (grid to 830 nm),
  RETRO 400 (its own separate plot, not the shared 100 one).
- Foma: FOMAPAN 400 (linear axis, assumption stated); Polaroid: 664, 667
  (log energy for ND 0.75 criterion).
Honest NO-PLOT findings (moved to text/table category): AGFA_VISTA_200,
KENTMERE_PAN_100, KENTMERE_PAN_400 — their sheets print no spectral
plot. Remaining queued: KODAK_VISION3_500T_5219 (needs the H-1-5219
sheet; only the brochure is on file).

## Seventh pass, same day — owner-supplied datasheets (batch 4: 35 -> 48)
Owner added 24 new files to PDF/PROFILES (~14:06-14:47). Two parallel
digitization agents. verify.py 70/70 PASSED; C++ regenerated +
object-compiles; film_names.txt + FilmCurves.md regenerated.

13 new spectral curve sets adopted:
- KODAK_VISION3_500T_5219 — the previously missing H-1-5219 (March
  2022): queue now EMPTY for VISION3.
- Full VISION2 set: 5217 (H-1-5217, 2004/rev-2005), 5218 (H-1-5218t,
  2006), 5205 (H-1-5205t, 2004).
- Full VISION1 set: 5274 (1997), 5246 (2003), 5279 (1996; -2..+2 axis,
  offset removed by peak normalisation).
- Full EXR set: 5245 (H-1-5245t, 2003), 5248 (H-1-7248, 1999),
  5293 (H-1-5293t, 2003).
- EASTMAN_PLUS_X_5231 (H-1-5231, 1999; D0.3-above-fog curve of the
  printed pair). EASTMAN_5247_1974 (TI0835 rev. 6-93, plate 6-83 —
  documents the post-1979 EI 125T generation; caveat recorded).
- FUJI_ETERNA_VIVID_500T_8547 (KB-0901E, 2009).
Honest findings: KODAK/5239.pdf is a mislabeled VNF-1 processing manual
(byte-identical to Module 11) — no 5239 spectral data exists on file;
KENTMERE-PAN-100_04_07_22.pdf (July 2022 edition) prints no spectral
plot. Every Kodak motion-picture stock in the database now carries its
own digitised per-emulsion spectral curves.

## Eighth pass, same day — machine-traced H&D batch 5 (VISION3 + 5222)
verify.py 70/70 PASSED; C++ regenerated + object-compiles; FilmCurves.md
and film_names.txt regenerated.

H&D characteristic curves machine-traced (digitize_plot.py) and refitted
[T1] from the owner-reviewed sheets:
- KODAK_VISION3_50D_5203 (H-1-5203 p3; 856-1372 samples/layer; RMS
  0.005-0.011 D) — moved to dmin_ladder (0.13/0.57/0.84).
- KODAK_VISION3_200T_5213 (H-1-5213 p3; 1380/layer; RMS 0.003-0.006 D)
  — dmin_ladder (0.17/0.58/0.85).
- KODAK_VISION3_500T_5219 (H-1-5219/TI2647F March 2022 p3; 1467/layer;
  RMS 0.002-0.005 D) — dmin_ladder (0.19/0.58/0.84). All four VISION3
  stocks now carry traced curves (5207 was batch 4a).
- EASTMAN_DOUBLE_X_5222 (H-1-5222 p3, D-96 6.5-min gamma-0.66 control
  curve; 876 samples; RMS 0.007 D; mid-grey anchored to previous
  hand-fit; shoulder beyond plotted range flagged).
Calibration note recorded: the VISION3 sheets' stop-tick pitch vs corner
logH labels disagree by ~4%; the 0.30103/stop convention (same as the
proven 5207 pass) was used consistently.
File note: "eastman 500t 5296 exr - Kodak.pdf" requested by owner was
NOT found in KODAK/ — 5296 keeps the TI2082-sibling spectral data and
estimated curves until that sheet arrives. New files 2254_TI2651.pdf and
the 5294 datasheet copy logged, no DB stock added.

## Ninth pass, same day — Ferrania P30 (batch 6)
verify.py 70/70 PASSED; C++ regenerated + compiles; FilmCurves.md
regenerated (49/89 spectral).

FERRANIA/"Curve caratteristiche e sensibilità spettrali (1).pdf" —
previously overlooked (its own P30 datasheet FP3011 is a best-practices
doc without curves; this second file was never opened). Contains: P 30
New + P33 + Orto H&D curves, wedge-spectrogram photos, comparison plot,
and printed processing (Kodak D-76 stock, 20 °C, 8 min, nominal speed).
- FERRANIA_P30 H&D machine-traced [T1]: 2195 samples, p1 red curve
  cross-checked against the p2 comparison (≤0.01 D agreement); fit RMS
  0.007 D. Measured gamma 1.25 (the documented P30 contrast). Stated
  assumptions: 0.15 logH/step, plotted density is above-Dmin, mid-grey
  anchor preserved. Printed shoulder is D-2.0 axis saturation — flagged.
- FERRANIA_P30 spectral [T2]: envelope of the wedge-spectrogram photo,
  peak 610–630 nm, red cut ~660 nm, ±0.15 log (uncalibrated photo,
  criterion states it).
- Also in file, not digitised (no DB profiles): P33 and Orto (ortho cut
  ~580 nm) — ready if those stocks are ever added.
