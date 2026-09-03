# Found.md — film stock → source document map

> ⚠ **THIS IS A DATED SNAPSHOT, NOT CURRENT STATE** (note added 2026-08-20). It records the
> verification pass of **2026-07-31**, when the database held far fewer stocks. It is kept verbatim
> because it is the audit record of which file each value was read from at that time, and it is still
> the fastest way to answer "which PDF did this stock come from".
> **For current per-stock coverage use `FilmActiveProfiles.md`**, which is regenerated from the
> database on every build and now reports **175** stocks with a live measured/estimated split. (Read 159 until 2026-08-25 and 160 until 2026-08-26f, when `KODAK_PRO_100T_PRT` was added from KODAK publication E-29.) For what
> is still missing, `NotFound.md`. For where the work stands, `PROGRESS.md`.
> Everything added since 2026-07-31 carries its citation in `film_profiles._PROVENANCE_SOURCES`
> rather than here — that registry, not this file, is what `verify.py` enforces.
>
> ⚠ **Working-copy caveat, 2026-08-23: some paths below do not resolve here.** This working copy
> holds only `AGFA`, `FERRANIA`, `FUJI`, `GEVAERT`, `KODAK`, `RETRO` and `SVEMA` under
> `PDF/PROFILES`. The `KONICA/…` files listed under "Unreadable files" and in the 2026-08-02
> spectral pass, and the `SOVIET STANDARDS/` folder cited in the 2026-08-13 entry, are **not
> openable from this checkout** — this file records where they were read, not what is on this disk.

Verification pass of 2026-07-31. One line per film stock that has manufacturer
documentation in `PDF/PROFILES/`, naming every file the data was collected from.

**Page-number convention:** for the manufacturer PDFs in sections below, page
numbers are *physical PDF pages*. For the two Soviet reference books (Iofis,
Gurlev) in the "partial data" section, page numbers are *printed book pages* --
those books are scans whose PDF page numbering does not match the printed
foliation (e.g. Iofis book p73 = physical PDF page 37).

Format: `STOCK` — `file` (pages) — what that file supplied.

---


## 2026-09-02e — queue T3 / T2 / E4 and the owner's Takano

`FUJI/provia_100f_datasheet.pdf` (**AF3-036E**), `FUJI/superia_xtra400_datasheet.pdf`
(**AF3-151E**) and `FUJI/pro_400h_datasheet.pdf` (**AF3-176E**) — staged from the owner's
machine and turned into **three new stocks, ids 172–174**. Characteristic curves and MTF
traced (`fuji_t3_2026.py`), rms granularity / resolving power at both contrasts / base /
speed transcribed as printed. ⚠ **Spectral sensitivity refused on all three**: two of the
sheets scale that ordinate with a bracketed arrow marked "1.0" and not a numbered ladder.

`KODAK/e4046_ektar_100-2016.pdf` (**E-4046**) — panel E4046D is three VECTOR paths and gives
`KODAK_EKTAR_100` **the first measured MTF for a still colour negative in this database**.
⚠ The estimate it replaces was **1.5× too sharp on every record**.

`KODAK/E7022_Gold_200-2016.pdf` (**E-7022**) — ⚠ **carries no MTF panel at all.** All three
copies in the corpus searched page by page; "Modulation Transfer" appears in none of them.
Recorded so the gap has a reason.

`KODAK/VISION3_5219_7219_Technical-data.pdf` (**H-1-5219**, March 2022) and
`KODAK/KODAK-VISION3-250D-5207-7207-technical-information.pdf` (**H-1-5207**, March 2026) —
both VISION3 stocks now cited to their own publications. ⚠ **Panels are rasters** (three
embedded images per page), so this is a citation and not a measurement.

`KODAK/Kodak - [1942] - Eastman Motion Picture Films for Professional Use.pdf` — 98 pages,
image-only, OCR'd page by page. Verified the 2026-08-11 Super-XX harvest (every value
reproduces; **the PDF page number was wrong by one**) and added the SD-21 formula, the 5242
sibling and the Plus-X Type 1231 specification. See queue E4.

`RETRO/JAPAN/31_209.pdf` — **Masao Takano 1968 Part 2**, supplied mid-batch. ⚠ **Not a
duplicate of `23_13.pdf`**, which is *Kiyoshi* Takano's review: different author, different
journal, original experiment. Fig. 11 traced; knowledge base §23n.

## Documented stocks (18)

`AGFA_APX_25` — `AGFA/agfapanapx25.pdf` (p1) — ISO 25, RMS 7.0, resolving power 200 lp/mm @1000:1, Schwarzschild table, layer structure
`AGFA_APX_25` — `AGFA/agfa_films.pdf` (p10) — corroborating RMS 7.0 and 200 lp/mm; spectral, characteristic and sharpness curves (images)
`AGFA_APX_25` — `AGFA/agfa_bw_manual.pdf`, `AGFA/agfa_film_chem.pdf`, `AGFA/agfa_bw_film_chemicals_en.pdf` — γ-vs-time tables per developer (γ 0.55 / 0.65 / 0.75)

`AGFA_APX_100` — `AGFA/apx100.pdf` (p1) — ISO 100/21°, RMS 9.0 (REFINAL 6 min 20 °C), resolving power 150 lp/mm @1000:1, Schwarzschild table, total layer thickness 7 µm
`AGFA_APX_100` — `AGFA/agfa_films.pdf` (p10) — corroborating RMS 9.0 and 150 lp/mm
`AGFA_APX_100` — `AGFA/Datasheet_F_PF_E4.pdf` (p9) — corroborating RMS 9.0
`AGFA_APX_100` — `AGFA/FPD1e.pdf` (p9) — duplicate reprint of the above, not an independent source
`AGFA_APX_100` — `AGFA/agfa_bw_manual.pdf`, `AGFA/agfa_film_chem.pdf`, `AGFA/agfa_bw_film_chemicals_en.pdf` — γ-vs-time tables

`AGFA_APX_400` — `AGFA/apx400.pdf` (p1) — ISO 400, RMS 14.0, resolving power 110 lp/mm @1000:1, Schwarzschild table
`AGFA_APX_400` — `AGFA/agfa_films.pdf` (p10) — corroborating RMS 14.0 and 110 lp/mm
`AGFA_APX_400` — `AGFA/Datasheet_F_PF_E4.pdf` (p9) — corroborating RMS 14.0
`AGFA_APX_400` — `AGFA/agfa_bw_manual.pdf`, `AGFA/agfa_film_chem.pdf`, `AGFA/agfa_bw_film_chemicals_en.pdf` — γ-vs-time tables and processing times

`AGFA_OPTIMA_100` — `AGFA/agfa_films.pdf` (p7) — **RMS 4.0** (corrected the code's 7.8), resolving power 50 lp/mm @1.6:1 and 140 lp/mm @1000:1, total layer thickness 16 µm
`AGFA_OPTIMA_100` — `AGFA/agfa_films.pdf` (p5) — layer design of OPTIMA II 100, film-identification symbol marks
`AGFA_OPTIMA_100` — `AGFA/Datasheet_F_PF_E4.pdf` (p7) — corroborating granularity figures

`FUJI_VELVIA_50` — `FUJI/velvia_50_datasheet.pdf` (p7, p8) — diffuse RMS granularity 9 (48 µm aperture, D 1.0 above Dmin), resolving power 80 lp/mm @1.6:1 and 160 lp/mm @1000:1, base material and thickness per format, layer stack; characteristic / spectral / MTF / dye-density curves as images
`FUJI_VELVIA_50` — `FUJI/AF3-0221E2Velvia50PIB.pdf` — same document (Ref. AF3-0221E2), **not** an independent source

`FUJI_PROVIA_400X` — `FUJI/Provia_400X_PIB_1007.pdf` (p1, p6) — diffuse RMS granularity 11, resolving power 55 lp/mm @1.6:1 and 135 lp/mm @1000:1, push range −½ stop (EI 280) to +2 stops (EI 1600), densitometry Fuji FAD-30S (Status A), base material and thickness, full layer stack, MCCL technology

`FUJI_SENSIA_100` — `FUJI/sensia_100_datasheet.pdf` (p4) — diffuse RMS granularity 10, resolving power 55 lp/mm @1.6:1 and 135 lp/mm @1000:1, base cellulose triacetate 127 µm (135 only), long-exposure threshold "64 seconds or more"

`FUJI_NEOPAN_ACROS_100` — `FUJI/NeopanAcros100.pdf` (p4) — diffuse RMS granularity 7 (Microfine, 48 µm, 12×, D 1.0 above min), resolving power 60 lp/mm @1.6:1 and 200 lp/mm @1000:1, reciprocity: no correction to 120 s, development tables
`FUJI_NEOPAN_ACROS_100` — `FUJI/Acros-120_AF3-083E.pdf` — 120-format edition; corroborates the above but prints different deep-tank Minidol times (both reproduced as printed in the evidence file)

`ILFORD_HP5_PLUS_400` — `ILFORD/HP5-Plus_201811.pdf` (p1, p2, p5) — ISO 400/27°, usable EI 400–3200, reciprocity `Ta = Tm^1.31` (→ Schwarzschild p = 0.7634) with no correction ½ s–1/10 000 s, base 0.125 mm acetate (35 mm) / 0.110 mm (roll) / 0.180 mm polyester (sheet), anti-halation backing, wedge spectrogram to tungsten 2850 K, full development matrix, characteristic-curve processing conditions (ILFOTEC HC 1+31, 6½ min, 20 °C)
`ILFORD_HP5_PLUS_400` — `ILFORD/HP5+-200407.pdf` — 2004 edition; 12 documented differences from the 2018 sheet, prints the reciprocity graph without the formula
`ILFORD_HP5_PLUS_400` — `ILFORD/2006216122447.pdf` — Ilford film-processing chart, cross-film development times

`ILFORD_DELTA_3200` — `ILFORD/Delta-3200_201811.pdf` (p1, p2) — **true ISO speed 1000/31° measured in ID-11** (marketed EI 3200), usable EI 400–6400 and to EI 25000 with published times, reciprocity `Ta = Tm^1.33` (→ p = 0.7519), tabular-grain construction, wedge spectrogram 2856 K, development matrix
`ILFORD_DELTA_3200` — `ILFORD/Delta_3200-200209.pdf` (p1, p2) — 2002 edition; same ISO 1000/31° statement, unambiguous ½ s reciprocity onset (used to resolve the 2018 sheet's internal inconsistency), spectrogram 2850 K

`KODAK_PORTRA_400` — `KODAK/e4050_portra_400-2016.pdf` (p1, p3) — ISO 400, C-41, **Print Grain Index 37 / 59 / 89** (135 at 4.4× / 8.8× / 17.8×), densitometry Status M, base and format table, exposure/complexion density references, characteristic and spectral curves as images
`KODAK_PORTRA_400` — `KODAK/e4050-Portra-400.pdf` — Sept 2010 edition; identical PGI, differs only in 220 format, 120 base thickness (0.10 vs 0.11 mm) and paper list
`KODAK_PORTRA_400` — `KODAK/Kodak_Print-Grain-Index_E-58.pdf` — PGI metric definition; states PGI cannot be compared to rms granularity and publishes no conversion factor

`EKTACHROME_64` — `KODAK/e8-Ektachrome_64_EPR.pdf` (p5) — diffuse rms granularity 11 ("very fine"), EI 64, Process E-6, densitometry Status A, reciprocity / CC-filter table (1 s → CC05R +⅓ stop)

`EKTACHROME_160T` — `KODAK/e144-Ektachrome_160T_EPT.pdf` (p4) — diffuse rms granularity 13 ("very fine"), EI 160 tungsten, Process E-6, densitometry Status A

`KODACHROME_64` — `KODAK/e88-2009_06.pdf` (p4) — diffuse rms granularity **10** under the heading "KODACHROME 64 Film" (p5 gives 16 for KODACHROME 200 — do not cross-wire), EI 64, Process K-14, Status A, reciprocity and CC-filter tables, push explicitly not recommended
`KODACHROME_64` — `KODAK/e55-2009_06.pdf` — professional PKR edition, corroborating; the Dec-1996 E-55 edition is the only one printing the base (5.3-mil acetate)

`POLAROID_664` — `POLAROID/664fds.pdf` (p2) — ISO 100/DIN 21, resolution 20–25 line pairs/mm @1000:1, panchromatic, coaterless, 45 s at 70 °F, 5-row time/temperature/exposure table. Prints the *definitions* of D-Max / D-Min / Slope but no values for this film

`POLAROID_667` — `POLAROID/667fds.pdf` (p2) — ISO 3000/DIN 36, resolution 14–20 lp/mm @1000:1, and **numeric characteristic-curve data: D-Max 1.75, D-Min 0.10, Slope 1.55 at 71 °F/21 °C** — the only such data for any stock in this database (Polaroid publishes the same triple for roughly a dozen other pack and sheet types, e.g. 53/553/803, 55/85, 572/672/72, 579/679/879, none of which has a profile here); 6-row development table, colour-temperature speed shifts, CRT phosphor exposure indices

`POLAROID_SX70` — `POLAROID/timezfds.pdf` (p1) — ISO 150/DIN 23, 3⅛ × 3⅛ in. image, glossy, ~5 min development. Product page only: no technical-data section, hence no curve, resolution, spectral or temperature data

`FOMAPAN_400_ACTION` — `FOMACOLOR/fomapan-400.pdf` (p1) — ISO 400/27°, **RMS 17.5** (Microphen 20 °C, developed to γ 0.6, measured at D 1.0 — corrected the code's 11.5), resolving power 90 lines/mm (contrast not stated), base thicknesses 0.1 / 0.125 / 0.175 mm, 9-developer table, complete Schwarzschild reciprocity table

---

## Reference documents used as method support, not as stock data

`KODAK/Kodak_Print-Grain-Index_E-58.pdf` — definition of Print Grain Index and its incomparability with rms granularity
`AGFA/agfa_films.pdf` (p5) — Agfa's own definitions of granularity and resolving power ("Reference: lines per mm at contrast range 1.6 : 1 or 1000 : 1")
`MISC/Photographic_Emulsions-EJ_Wall-1929.pdf` — historic emulsion chemistry; no product data
`MISC/ColorChecker_Passport_Technical_Report.pdf`, `MISC/Guide_to_Surface_Characteristics_FINAL.pdf` — unrelated to any stock in the database
`PROFILES/FUJI/SS35.pdf` — **FUJIFILM DATA SHEET "NEOPAN SS (135)", Ref. AF3-411E(N), stock data.** ISO 100/21°, orthopanchromatic, full development matrix, spectral sensitivity curve, characteristic-curve family with the average gradient printed on every member, time-Ḡ curves. Source of `FUJI_NEOPAN_SS` (stock 172, added 2026-09-02, queue N1): curve, spectral sensitivity, speed and processing. ⚠ **No image-structure section** — no rms granularity, no resolving power, no MTF, no reciprocity, no base thickness — so that stock's grain and MTF are flagged class estimates
`PROFILES/FUJI/Fuji Sales Guide Curves.pdf` — Fuji sales guide, two raster pages, ⚠ **read but not yet harvested**. Prints **ASA 50 / 100 / 200** for NEOPAN S / SS / SSS in its running text, a per-film γ-versus-development-time panel with a numeric γ axis and a per-film characteristic-curve family with a numeric log-exposure axis and a marked base density, all Minidol 20 °C tank. ⚠ Its 第3図 three-film overlay is **schematic** (axes labelled only 大/小), so it carries ordering and shape, not a curve
`PROFILES/RETRO/JAPAN/23_13.pdf` (Takano 1969) — **method support that changed the engine.** eq (2) gives σ(D) from σ(T) to fourth order and is adopted as `film_sim.sigma_density_from_transmittance` (inert); eq (13) gives the print-chain grain law the engine already satisfies, with one recorded departure. Its Figs. 8, 9 and 13 measure four samples, **none of them a stock in this database** — the aperture series validates `grain_reference_energy` to rms 0.007–0.020 in G, and with Ooue Fig. 24 supplies the five-point clump census queue C45 was missing
`PROFILES/KODAK/Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf` (Sehlin, Kennel et al., *SMPTE Journal* **94**(7) 724-731, **July 1985** — ⚠ the file name's 1983 is wrong) — ⚠ **method support, NOT stock data — a good σ(D) trace that was withdrawn.** Fig. 8 puts DENSITY and RMS GRANULARITY on ONE shared log-exposure abscissa for `EASTMAN_5294_1983`, so the pairing is complete on a single plate: toe 1.571 @ D 0.44 / mid 1.000 / dmax 0.703 @ D 2.08, peak 1.664 @ D 0.53, inside the eleven vendor sheets' envelope. ⚠ **NOT STORED** — the anchors are the figure's plotted density where `sigma_anchors` reads per-layer analytical density, and the traced toe at D 0.44 sits below this stock's green dmin of 0.68; `cpp_parity` rejected it at 5.7e-01 against a 2e-05 tolerance. ⚠ Its LEVEL is refused — the ordinate has no unit, aperture or densitometry — and ⚠ Fig. 12's MTF for `EASTMAN_5247_1983` is refused too: 50 % at 45-58 c/mm against a stored estimate of 24/28/33, but the text calls it a *system* MTF and the curve does not overshoot
`PROFILES/RETRO/JAPAN/26_172.pdf` (Fujimura & Yamamoto 1963) — tone-reproduction study using Neopan SS 35 mm as its test negative. ⚠ **Checked as a Neopan source on 2026-09-02 and rejected**: Fig. 6 is a *system* curve (log B_or of the enlarged print against log B_0 of the scene) carrying the paper's own curve and camera flare, neither of which the paper prints, so it cannot be inverted to the negative's characteristic curve. With `23_7.pdf`, `22_91.pdf` and `23_13.pdf` this is the whole Neopan evidence in the corpus and it is grain only — see `EMULSION_KNOWLEDGE_BASE.md` §23k.8 for why no FUJI NEOPAN profile was created

---

## Partial data from non-manufacturer published sources — NOT applied

These are state-published handbooks, not manufacturer datasheets. Recorded for
completeness; no profile value was changed from them pending a policy decision
(see `DATASHEET_VERIFICATION_REPORT.md` §10.2).

`EASTMAN_DOUBLE_X_5222` — `PDF/Iofis_kinofotoprocessy_materialy_1980.pdf` (table 5, p73) — GOST/DIN/ASA speed only
`EASTMAN_PLUS_X_5231` — `PDF/Iofis_kinofotoprocessy_materialy_1980.pdf` (table 5, p73) — GOST/DIN/ASA speed only
`EASTMAN_5247_1974` — `PDF/Iofis_kinofotoprocessy_materialy_1980.pdf` (table 6, p79) — speed, 35/16 mm, process ECN-2
`EASTMAN_5247_1974` — `PDF/Gurlev_sprav_svetotexnika_materialy.djvu` (p258) — «Истменколор 5247» emulsion layer stack
`EASTMAN_EKTACHROME_7239` — `PDF/Iofis_kinofotoprocessy_materialy_1980.pdf` (table 22, p150) — speed only; printed as designation **VN**, not EF
`SVEMA_FOTO_250` — `PDF/Gurlev_sprav_svetotexnika_materialy.djvu` (p296 §247) — «Фото-250» speed, γ, D₀, D₀max, latitude, resolving power — but a *still* film type with no Svema attribution
`SVEMA_FN_64`, `TASMA_FN_64` — `PDF/Iofis_kinofotoprocessy_materialy_1980.pdf` — GOST cine-negative class specs НК-1…НК-4 (resolving power, RMS granularity, MTF, fog, γ, spectral limit); class-level, not product-level
`EIGHT_MM_BW` — `FOMACOLOR/fomapan_cine_100.pdf`, `FOMACOLOR/foma-cine-ortho.pdf` — genuine cine datasheets supplied in 2×8 mm standard and 2×DS 8 mm super, with RMS granularity, resolving power, base and processing data; a legitimate anchor if this generic is ever grounded

---

## Explicitly excluded as evidence

`PDF/PROFILES/Film_Profile_Model_Domain_Review.pdf` — this project's own design notes
`PDF/PROFILES/Модель данных профилей киноплёнки.pdf` — this project's own design notes
`PDF/PROFILES/SVEMA/SVEMA-FN64_generated_film_profile.txt` — output of this project's `analyze_film_scans.py v2.0`
`PDF/PROFILES/SVEMA/SVEMA-FN250_generated_film_profile.txt` — same
`PDF/PROFILES/TASMA/TASMA-FN64_generated_film_profile.txt` — same
`PDF/Films.xlsx` — renderer tuning sheet, grain in px-at-1080p
`PDF/film.txt` — 315-byte unsourced note
`ROLLEI/Superpan.pdf` — signed third-party test article, not a manufacturer sheet
`PDF/PROFILES/*/www.txt` — download URL lists

## Unreadable files (no text layer)

`FUJI/Neopan1600.pdf` — 8 pages, 0 characters. Blocks `FUJI_NEOPAN_1600`; OCR would likely recover full specs
`FUJI/Neopan400.pdf` — 10 pages, 0 characters; no profile depends on it
`KODAK/transmision of wratten filters.pdf`
`KONICA/centuria_pro_400.pdf`
`KONICA/professional_160.pdf`

---

## Update 2026-08-02 — Soviet reference-book pass

The "no text layer" barrier above was bypassed by rendering the scanned
pages to images and reading them visually. The Gurlev DjVu was converted to
PDF by the owner (`SOVIET/Gurlev_sprav_svetotexnika_materialy.pdf`, 367 pp).
Page-level transcriptions: `SOVIET_EXTRACTION_2026-08-02.md`. Page numbers
below are PRINTED BOOK pages.

`SVEMA_FN_64` — `SOVIET/Gurlev_sprav_svetotexnika_materialy.pdf` (book p296, §247) — Foto-65 column: S 65 GOST, γ_rec 0.8 (СТ-2), D0 0.05, latitude 1.5 logH, R 110 lin/mm, Δλ_S 665 nm. Corroborates the measured profile; tier raised 3 → 2; foto-65 aliases added
`SVEMA_FOTO_250` — same file (book p296, §247) — Foto-250 column now adopted as the product source: S 250, γ 0.8, D0 0.08, D0max 0.3, L 1.5, R 82 lin/mm, Δλ_S 630 nm
`SVEMA_FOTO_32` (NEW) — same file (book p296, §247) — S 32, γ 0.8, D0 0.04, L 1.5, R 135 lin/mm, Δλ_S 645 nm
`SVEMA_FOTO_130` (NEW) — same file (book p296, §247) — S 130, γ 0.8, D0 0.06, L 1.5, R 100 lin/mm, Δλ_S 580 nm (orthopanchromatic cut)
`SVEMA_DS_4` (NEW) — same file (book pp354-355, §306) — ТУ 6-17-622—74, unmasked, S 45, γ_общ 0.8, L 1.2, Dmax 2, D0 0.25/zone, R 63 lin/mm, H&D fig. 197
`SVEMA_TSNL_32` (NEW) — same file (book pp354-355, §306) — ТУ 6-17-441—78, masked, 3200 K, S 32, γ 0.7±0.1 (top +0.1-0.2), L 0.9, Dmax 2.5, mask densities 0.75-1.1/0.25-0.5/0.3, R 58 lin/mm
`SVEMA_TSNL_65` (NEW) — same file (book pp354-355, §306) — masked, 3200 K, S 65, γ 0.7±0.1, L 1.5, Dmax 2.4, Б_к 0.1, mask densities 0.75-1.1/0.4-0.6/0.3, R 63 lin/mm
`TASMA_OCH_45` (NEW) — same file (book p298, §249) — ТУ 6-17-646—74, B&W reversal, S(0.9) 45, γ 1.1-1.6, L 1.05, Dmax 1.9, Dmin 0.08, S^Ж 16, first dev 12 min; plus `SOVIET/Кинофотопроцессы и материалы.pdf` (Iofis 1980, book pp146-147) — R ≥ 100 mm⁻¹, sensitized to 680 nm, bluish triacetate base
`ORWO_CHROM_UT18` (renamed from ORWO_UT18) — `ORWO/ORWO CHROM-FILM UT 18 - web.jpg` (leaflet W 746, VEB Filmfabrik Wolfen) — official name, 18 DIN / 50 ASA / 45 GOST, daylight balance, storage < 18 °C 50-60 % RH

Class-level corroboration (not product-specific, recorded but not adopted
into per-stock fields): `SOVIET/Справочник Кинооператора.pdf` (Gordiychuk/
Pell 1979) table X-2 (ДС-5М/ЛН-7/ЛН-8 colour cine negatives, book p377) and
table X-7 (НК-1…НК-4 B&W cine negatives with RMS granularity and MTF@30
limits, book p382); `SOVIET/TASMA POSITIVE МЗ-3Л.jpg` and Gurlev p296-297
(МЗ-3Л positive: γ_rec 2.5, R 125 lin/mm, λ_max 480 nm, Dmax 3.7) —
available to refit the TASMA_POSITIVE_28 print stock if desired.

### Second pass, same day — Chibisov 1988 appendix + Dufaycolor NSMM

`SVEMA_FOTO_32/FN_64/FOTO_130/FOTO_250` — `SOVIET/Фотография в прошлом настоящем и будущем.pdf` (Chibisov 1988, Appendix Table I, book p157, PDF p164, rotated) — S/γ/γmax/D0-range/L confirmed; R printed 116/92/75/70 vs Gurlev 135/110/100/82 (conflict recorded, Gurlev kept)
`TASMA_OCH_45` — same file (Table I, book p158, PDF p165) — γ_rec 1.6 (max 2.2), D0 0.06, L 1.05, **R 110 mm⁻¹**, 660 nm — γ and R adopted (γ 1.35→1.50, R→110)
`EASTMAN_5247_1974` — same file (Table VIII, book p165, PDF p172) — S 125 GOST, ḡ 0.50, RMS σD×1000 visual 5, MTF@30 0.65 green/0.32 red — recorded, grain not adopted
`EASTMAN_5294_1983` — same file (Table VIII) — S 400, ḡ 0.50, RMS 6, MTF@30 0.65/0.30 — recorded; printed green MTF@30 matches profile f50_g exactly
`SVEMA_DS_4`, `SVEMA_TSNL_32`, `SVEMA_TSNL_65` — same file (Table IX, book p167, PDF p174) — independent confirmation of every adopted Gurlev value
`DUFAYCOLOR_1937` — `DUFAYCOLOR/measuredODs_MSI_NSMM_11948/11951/11960_Dufaycolor_small.jpg` (NSMM Bradford) — measured reseau element absorbance 400-720 nm; filter_matrix rebuilt from these curves (tier 3→2)

### Update 2026-08-02 — digitised curve sources (schema v3 + traced H&D)

Machine-traced H&D characteristic curves (digitize_plot.py, RMS ≤ 0.007 D):
`KODAK_VISION3_250D_5207` — H-1-5207 p3 sensitometric plot (3 layers, 1426 samples each); also the source of the adopted dmin mask ladder
`FUJI_NEOPAN_ACROS_100` — AF3-095E p5, Microfine 15-min curve (1092 samples)

Digitised spectral sensitivity curves (35 stocks; per-stock table with points/resolution: `FilmCurves.md`; full citations in each profile's `spectral.source`):
Fuji AF3 sheets (Velvia 50, Provia 400X, Sensia 100, Neopan 1600, ACROS); Kodak H-1/E/F/TI publications (VISION3 50D/200T/250D, Portra 400 E-4050, Tri-X 7266, Kodachrome 64 E-55, Ektachrome 64 E-8 / 160T E-144 / 100D via H-1-5294, Double-X H-1-5222, EXR 500T via TI2082); Harman Nov-2018 sheets (HP5+, Delta 3200 wedge spectrograms); Agfa 1995 Datenblätter (APX 25/100/400) + Range-of-Films brochure (Optima II 100); all six Konica colour TDS + Infrared 750; Rollei TDS (R3, Infrared, Retro 400); Fomapan 400 sheet; Polaroid 664/667 Film Data Sheets.
Sheets verified to print NO spectral plot: AGFA_VISTA_200, KENTMERE_PAN_100/400.

## 2026-08-13 — first extraction from the new landing

* **Kodak H-1-5239** (`KODAK/Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf`,
  text PDF): diffuse RMS granularity 14 (D=1.0, 48 um), resolving power
  40/100 lp/mm (1.6:1 / 1000:1, ISO 6328-1982), Process VNF-1, Status A,
  reciprocity flat 1 s - 1/10000 s. Adopted into EASTMAN_EKTACHROME_5239 and
  _7239 (rms, resolving power, provenance). See CHANGES_2026-08-13_extraction.md.
* **«Современные фотоматериалы и их обработка»** (`SOVIET STANDARDS/`, 717 pp,
  text): reciprocity correction tables adopted for AGFA_OPTIMA_100/200/400,
  AGFA_PORTRAIT_160, AGFA_VISTA_200, AGFA_APX_100/400 [C1]. 27 correction
  tables total; Kodachrome/T-MAX/Tri-X/Plus-X/Verichrome Pan/Scala tables
  present but era-mismatched to our 1952 profiles (not adopted).
* **Kodak Publication F-5** (~1979 ed., 88 pp scan): indexed, DS catalogue
  recorded in CHANGES_2026-08-13_extraction.md; extraction queued -- 1970s
  formulations are not the 1952 stocks we model.
* **The Compact Photo-Lab-Index** (Pittaro ed., Morgan & Morgan, 2nd Compact
  Edition 1979, 724 pp scan with OCR layer): indexed and mined 2026-08-14.
  Yielded 11 stocks -- 8 Polaroid types with published D-max/D-min/slope/speed/
  resolution and Ilford Pan F, FP4, HP4 [C1] -- plus the Kodak reciprocity
  master table (rebuilt from word coordinates) and Ilford's film-and-plate
  sensitivity ranges in Angstroms. Contains NO per-layer spectral sensitivity
  for any colour film. Full record: CHANGES_2026-08-14_photo_lab_index.md;
  structure and priorities: SURVEY_2026-08-14_photo_lab_index.md.
* **FUJIFILM MOTION PICTURE FILM MANUAL** (ref. KB-1101E, FUJIFILM Corporation,
  2011, 44 pp true PDF): indexed 2026-08-14. Master exposure-index table for the
  whole Fujicolor cine line (9 camera stocks, 35 mm and 16 mm type numbers,
  tungsten and daylight E.I. with the conversion filter named, sideprint codes),
  uniform reciprocity statement per film, exposure and densitometry conditions,
  edge markings, raw-stock storage and X-ray dose data. CONFIRMED our
  FUJI_ETERNA_VIVID_500T_8547 against a second Fujifilm publication (E.I. 500,
  3200 K, Status M) and supplied its reciprocity onset. Curves are raster.
  Eight further cine stocks documented for E.I. and balance only -- deferred to
  next_week_task.md. Full record: CHANGES_2026-08-14b_fuji_kodak_websites.md.
* **KODAK EKTACHROME 100D 5285/7285, publication H-1-5285** (February 2010, 5 pp
  true PDF): extracted 2026-08-14. Settles the NotFound row for this stock.
  **Its curves are PDF VECTOR paths** -- the first in this project extractable
  exactly, with no digitize_plot.py step. Replaced the profile's spectral curves,
  which had been borrowed from the 5294/7294 reintroduction; the borrow was
  validated in the process. Also supplies Process E-6, Status A, the
  illuminant/filter E.I. table, and -- not yet enterable -- spectral dye density,
  MTF and rms granularity curves.
* **FUJIFILM DATA SHEET "NEOPAN 1600 Professional", Ref. No. AF3-608E(N)** (4 pp,
  TRUE DIGITAL PDF): extracted 2026-08-15. Supersedes the unreadable
  `FUJI/Neopan1600.pdf` scan entirely. EI 1600/33 deg, panchromatic, usable EI
  range 400-1600, grey-tinted cellulose triacetate 0.122 mm, filter factors,
  safelight, a 16-developer x 5-temperature x EI development matrix (EI 250-3200
  at 18-26 C), full wet chemistry, automatic-processor conditions, a spectral
  sensitivity curve (re-traced at 5 nm, the finest sampling in the corpus) and
  characteristic curves in SPD at 20 C for three development times with PRINTED
  average gradients Gbar 0.58 / 0.77 / 0.90 plus Time-Gbar curves for four
  developers. Curves are 300 dpi rasters, not vector. The stored curve was
  refitted to 487 traced points; base+fog corrected 0.170 -> 0.211 and the curve
  now reproduces the printed Gbar 0.77 to 0.001. Full record:
  CHANGES_2026-08-15_neopan1600.md.
* **2026-08-15 batch, 15 new files + KODAK DATA BOOK + Zhurba 1984**: full record
  in CHANGES_2026-08-15b_new_references.md, extraction detail in
  reanalysis_2026-08-14/RESULT_{KODAK_NEW,FUJI_AGFA_NEW,ZHURBA}.md. Headlines:
  Technical Pan P-255 (new stock, CI 0.50-2.50 envelope); F-500 8572 and Vista
  sheets settle two "genuinely absent" gaps with RMS corrections of ~2x; Portra
  E-190 documents the NC/VC generation (not merged -- different generation from
  our 2010s stocks); 2383 numeric LAD 1.09/1.06/1.03; 5366/7366 RMS 9 +
  RP 100/200; KODAK DATA BOOK vol 5 (FILMS) located at pp 1150-1495, ~1948-1968,
  post-1960 ASA warning recorded; Zhurba 1984 rotated tables read visually,
  corroborates Gurlev/GOST, zero ORWO content; Zhurba 1990 online pp 44-131
  UNREACHABLE via web_fetch (webp page images) -- local copy requested.

## 2026-08-16b — Zhurba 1990 via owner-supplied screenshots

Table 66 (p124): **first ORWO data in the corpus** — NC21 100/5500 K (confirms stored),
UT-18 50/5500 K daylight (stored 4500 K corrected). Table 2 (p46): post-1987
Фото-32/64/125/250 norms recorded as successor-generation consistency data (our RMS all
inside the printed ceilings). Table 13 (p65) corroborates ОЧ-50 660 нм. Full report:
`reanalysis_2026-08-14/RESULT_ZHURBA1990_SCREENSHOTS.md`.

## 2026-08-16c — official-source web hunt

Kentmere PAN 100/400 official HARMAN sheets (ilfordphoto.com, July 2022) and the Konica
VX 100 sheet (125px mirror) retrieved and entered. Three of four retrieved sheets agreed
EXACTLY with values already in the database (reciprocity p 0.794/0.769, RMS 4, resolving
63/125) — estimates became citations. "No documentation" list 23 → 18 today. Full report:
`CHANGES_2026-08-16c_web_hunt.md`.

## 2026-08-16d — NotFound.md section 4 worked through

**14 stocks gained a measured spectral sensitivity curve** extracted from exact PDF vector
coordinates — UltraMax 400/800, Ektar 100, Portra 160/800/100T, Gold 100 + 200,
Tri-X 400TX, T-MAX 100, T-MAX P3200, Plus-X 125, T400CN, BW400CN. Stocks carrying a
spectral curve: **53 → 67 of 143**. The corpus inventory is now measured rather than
estimated: **516 genuine vector curve pages in 245 PDFs**, 152 of them spectral, against
421 logo/glyph false positives. The OCR bracket is closed entirely; KODAK DATA BOOK vol 5
and two Konica brochures are closed as documented dead ends. Reports:
`CHANGES_2026-08-16d_section4.md`, `reanalysis_2026-08-14/RESULT_VECTOR_SWEEP.md`,
`reanalysis_2026-08-14/RESULT_OCR_IMAGEONLY.md`.

## 2026-08-16e — SVEMA Foto line: apparent conflict with Zhurba Table 2 resolved

The owner asked whether Zhurba 1990 Table 2 (p46) had been used to fine-tune the SVEMA
Foto stocks. It had not — only recorded as provenance — and the check that followed found
our stored resolving power and MTF *below* the norms that table prints. Investigated
against the primary standard already on disk, **ГОСТ 24876-81**: its Table 6 carries three
successive norm sets, and the standard's own note reads «Нормы, указанные в скобках,
вводятся с 01.01.90». Zhurba 1990 prints the 1990-01-01 set; our values are the original
1981 top-category norms, correct for the generation modelled. Both sets are measured by the
same method (ГОСТ 2819-84, named on p14 of 24876-81), so the difference is a **raised norm
for a new emulsion generation, not a test-object-contrast mismatch**. No stored value
changed; provenance on all four stocks rewritten to carry the full three-tier history, and
a permanent `verify.py` guard now asserts the Foto line stays on its own era's norm set.

## 2026-08-17 — ДС-5М specification, and a measured-grain bug it exposed

**`SVEMA_DS_5M` added (143 → 144)** from ТУ 6-17-691-88, a state manufacturing
specification — now the best-documented Soviet stock we hold. Mean gradients 0.60/0.54/0.50,
Dmin ladder 0.70–1.05 / 0.25–0.50 / ≤0.25, latitude ≥1.2, MTF@30 ≥0.30 green and ≥0.15 red,
RMS ≤22 green and ≤30 red, base ОТБ-14 triacetate 0.150 mm, sensitometry at 5500 K. **Two
corpus-wide gap classes closed**: numeric orange-mask density (the Dmin ladder *is* the
mask) and dye-impurity coefficients (seven measured D_вр/D_пол ratios). First real use of
the schema-v7 `ProcessingFamily` carrier.

**Bug found and fixed:** `_grain_v2()` overwrote measured per-channel RMS on every colour
negative. `GEVACOLOR_NEG_682` had been rendering as 17.6/16.0/20.8 instead of its measured
23/16/34 — erasing the documented blue ≫ red > green inversion its own comment explains.
Guarded now.

Gorokhovskii УФН 1936 reviewed in full: a methods paper, nothing entered, but it establishes
that 1930s spectral figures are method-bound (green/blue ratio 4–13 % vs 30–60 % vs 20–45 %
between three investigators). Full report: `CHANGES_2026-08-17_ds5m_and_1936.md`.

## 2026-08-17b — Soviet TU batch: ДС-4 grounded in its own specification

10 unique TU documents reviewed (two files were byte-identical duplicates). **ДС-4 moved off
a handbook paraphrase onto ТУ 6-17-622-84**, its own primary specification: per-layer gammas
corrected to b = g = 0.70 / r = 0.60 (the previously stored spread was inverted and far too
narrow), resolving power 63 → 68 lin/mm per ГОСТ 2819-84, Status M densitometry confirmed,
development 6-8 min with the developer formula, sensitivity balance and fog ceilings new.
Фото-65 / ЦНЛ-65 / ДС-4 gained documented shelf life from the export TU 6-17-1371-86.

**Eight further stocks (ЛН-8, ЛН-9, ЛН-9С, ЦНД-64, ЦО-32Д, ЦО-Т-90ЛМ, ЦО-90Д, ЦО-90Л) are
fully specified in these documents and none is in the database.** Their figures are captured
at OCR level in `RESULT_2026-08-17_SOVIET_TU_BATCH.md` but NOT entered: the OCR detaches
values from row labels on typewritten scans, and every number entered today was verified
visually first. Visual verification queued.

**New lead:** ЦНЛ-65's characteristics are governed by ГОСТ 25130-82, which is not in the
corpus — the same kind of pointer that turned out to be decisive for ДС-4.

## 2026-08-17c — four Soviet stocks added from their own TU specifications

`SVEMA_LN_8`, `SVEMA_LN_9`, `SVEMA_LN_9S`, `SVEMA_CO_32D` — **144 → 148 stocks**, every
figure from ТУ 6-17-1109-88, ТУ 6-17-1443-88 and ТУ 6-17-912-87, each number read visually
from the page image. A **third marking class** was added to `FilmActiveProfiles.md` first,
because TU figures are acceptance LIMITS and would otherwise read as measurements.

Highlights: ЛН-9/ЛН-9С are one emulsion in two antihalation constructions, and the
specification quantifies the consequence (ЛН-9С's Dmin ladder sits 0.05-0.10 D lower) —
a controlled A/B from one document. ЛН-9 is the finest-grained Soviet colour stock we hold
(RMS ≤ 11 both channels) and the only TU giving MTF as a tolerance rather than a minimum.
ЛН-8 carries a red-layer sensitisation limit of 690 nm and a NEGATIVE dye-impurity term.

The visual gate caught real OCR errors, including ЛН-8's minimum layer sensitivity (OCR 60,
page 80). Full detail: `CHANGES_2026-08-17b_four_soviet_stocks.md`.

## 2026-08-17d — backlog harvested into the v7 carriers; a fifth carrier added

Measured data that existed only as prose inside provenance strings is now typed and
validated. `DyeImpurity` added as the fifth v7 carrier, with **26 measured
unwanted/useful density ratios** across ДС-5М, ЛН-8, ЛН-9 and ЛН-9С -- including ЛН-8's
NEGATIVE term (the specification prints "minus 0.05-0.10"), which the validator
deliberately permits. `ReciprocityTable` filled for 6 stocks: the Kodak master-table rows
carry the CC-filter colours that document WHICH channel loses speed (Ektachrome 64 blue,
160T red -- opposite orderings in one table), and Ektachrome 64's +1/2 stop at 1/10 000 s
is failure at the SHORT end that no Schwarzschild exponent can express.
`ProcessingFamily` filled with 17 development points, every one carrying a measured
contrast -- Pan F's full 12-cell table is the richest in the corpus. `LayerStack` attached
to EASTMANCOLOR_5248_1953 only; the Agfacolor row was refused as ambiguous (three held
candidates) and Cheltsov's 5245 dupe refused as a Kodak number reuse.

Inertness re-proven with real data in four carriers: all ten reference stocks still hash
bit-identically, max abs delta 0.000e+00. Full report: `CHANGES_2026-08-17c_carrier_harvest.md`.

⚠ **REVERSED for `ReciprocityTable`: it is no longer inert.** As of 2026-08-23 (C8) reciprocity is
WIRED into both renderers — `RenderSettings.exposure_time_s` / `AlgoControls::exposureTimeS`, `0`
meaning inert — so a stored table now changes pixels whenever an exposure time is supplied. The
carrier holds **21 measured entries** (was the 6 above) after 15 more were read from vendor sheets,
and **105 stocks carry a Schwarzschild exponent**. ⚠ What it models is a per-channel **global
log-exposure shift**, not an intensity-dependent one: no source in the corpus has an intensity axis.
The wiring is covered by a cross-language parity audit against the plugin's own C++
(`cpp_parity.py`), as are the DIR-coupler stages (`interimage_parity.py`).

## 2026-08-17e — DIGITIZATION_QUEUE.md rewritten as a working document

43 kB of fourteen chronological batches -> **9 kB**, with nothing deleted: the layered
narrative is preserved verbatim in `DIGITIZATION_QUEUE_history.md` (same pattern as the
NotFound.md rebuild), and the live file now carries only what a reader needs to act.

The restructure surfaced what had become buried. **13 binding method rules are now at the
top** instead of scattered across batch notes -- including the ones that were each learned
the hard way: check for vector paths first; ignore a filled frame and calibrate to the tick
labels; classify tracks by persistence (>=70/80 solid, 25-50 dashed, <20 glyph or tick);
mutual exclusion always; bridge only to the measured dash period; validate against a printed
statistic; measure fitted statistics on the model not the trace; enforce
toe_k <= shoulder_k <= 2*toe_k inside the search; **and sanity-check siblings against physics,
because internal consistency is not correctness.**

The DO-NOT-TRACE list is now a table with reasons, so refusals stop being re-litigated, and
every open item carries its specific blocker rather than sitting in an undifferentiated list.

## 2026-08-17f — spectral dye density: the first curves, and a self-validating check

`SpectralDyeDensity` filled for **KODAK_EKTACHROME_100D_5285** (H-1-5285 p4) and
**KODAK_2383_RELEASE** (2383 sheet p6), both from PDF vector paths, C/M/Y plus the sheet's
own visual-neutral trace.

**The neutral trace makes this extraction self-validating.** A visual neutral is by
definition the sum of the three dyes, so checking sum(C+M+Y) against it tests curve
identification, axis calibration and sampling in one step. 5285 agrees to **max 0.013 D
across 31 samples** -- the cleanest validation of any curve extraction in this project.
Peaks land where they must: yellow 440, magenta 550, cyan 660 nm. 2383 agrees to 0.128 D
with the discrepancy **systematic and confined to 400-440 nm**, which is expected for a
print stock: the visual neutral includes the base and its UV/blue absorber while the dye
traces do not. Recorded, not tuned away.

**Two corrections to my own earlier claims**, both recorded in the queue: the "54 vector
pages" of dye density was a corpus-wide count -- on the four VISION3 TI sheets these plots
are RASTER, so this batch is 2 stocks, not 6. And `PrintStock` had none of the v7 carriers,
which only surfaced when 2383's validated data had nowhere to go; the field was added there
rather than discarding a good extraction.

verify.py 143 PASS / 2 pre-existing FAIL on the day. All five v7 carriers now hold real data and
the ten reference renders remain bit-identical. (Current state 2026-08-23: **304 checks,
303 PASS / 1 FAIL** — the surviving failure is the saturation-hierarchy ordering, known and left
alone — plus 11 audit scripts, all green, among them the cross-language parity audits of the
reciprocity and DIR-coupler stages against the plugin's own C++, `cpp_parity.py` and
`interimage_parity.py`.)
