# RESULT_OTHERS.md — Manufacturer film data extraction, "OTHERS" scope
Scope: 71 PDFs / 2682 pages listed in `/tmp/list_OTHERS.txt`, root of `PDF/PROFILES/`.
Method: one PyMuPDF pass dumped per-page text (`/tmp/oth/alltext.txt`) and flagged pages
matching Latin+Cyrillic keyword regexes (speed/gamma/RMS/resolving/spectral/density/
reciprocity/Kelvin/MTF/latitude/developer). Only flagged pages were then read in detail.
**Nothing below is estimated — every number is printed in the cited file/page.**

---
## 1. Per-file yield summary
`pp` = pages, `avg` = mean chars/page, `hit` = pages matching high-value keywords,
`IMG` = image-only / near-zero text (needs OCR).

| file | pp | avg c/p | hit pages | flag |
|---|---|---|---|---|
| 5_1_FilmBaseGuide_2020.pdf | 6 | 4123 | 1 |  |
| The Permanence and Care of Color Photographs.pdf | 761 | 4781 | 376 |  |
| aimm.it2.18.1996.pdf | 29 | 1738 | 24 |  |
| AGFA/Agfa_TechnicalDataCP30ColourPrintFilm_17.10.2011.pdf | 5 | 2740 | 4 |  |
| AGFA/FPD1e.pdf | 12 | 2591 | 11 |  |
| AGFA/Meyer_TechnikDesFarbfilmsVervollkommnen_1953.pdf | 2 | 0 | 0 | IMG |
| AGFA/NewGevacol_Neg_682.pdf | 3 | 0 | 0 | IMG |
| AGFA/These-23-11-09fusion2.pdf | 686 | 2825 | 64 |  |
| AGFA/UT-420343_AGFA_Aviphot-Pan80.pdf | 7 | 726 | 4 |  |
| AGFA/agfa-Alliance-IR-Technical-Infosheet.pdf | 2 | 1265 | 2 |  |
| AGFA/agfa-aERRKF-Datenblatt_F_PF_D4.pdf | 12 | 2743 | 10 |  |
| AGFA/agfa_bw_film_chemicals_en.pdf | 16 | 2618 | 12 |  |
| AGFA/agfa_bw_manual.pdf | 69 | 1449 | 46 |  |
| AGFA/agfa_film_chem.pdf | 16 | 2881 | 14 |  |
| AGFA/Agfacolor Neu/Behrendt_Köllner_Fehlerq_1950.pdf | 4 | 0 | 0 | IMG |
| AGFA/Agfacolor Neu/Gröger_Hydrolyse_1961.pdf | 5 | 0 | 0 | IMG |
| AGFA/Agfacolor Neu/SchultzeHörmann_Agfa_1951.pdf | 6 | 0 | 0 | IMG |
| DUFAYCOLOR/Carson_Dufaycolor_Kinotechnik_1934_conv_print.pdf | 3 | 8086 | 2 |  |
| DUFAYCOLOR/Dufay_Dufaycolor_GB000000262386A.pdf | 3 | 0 | 0 | IMG |
| DUFAYCOLOR/Dufaycolor_Manual_1938_print.pdf | 25 | 0 | 0 | IMG |
| DUFAYCOLOR/dufaycolorbook00dufa.pdf | 60 | 1376 | 24 |  |
| FERRANIA/1579.pdf | 4 | 1500 | 3 |  |
| FERRANIA/Curve caratteristiche e sensibilità spettrali (1).pdf | 2 | 283 | 2 |  |
| FERRANIA/FP3011_Datasheet.pdf | 3 | 1747 | 3 |  |
| FOMACOLOR/foma-cine-ortho.pdf | 1 | 5193 | 1 |  |
| FOMACOLOR/fomaortho.pdf | 1 | 5521 | 1 |  |
| FOMACOLOR/fomapan-200.pdf | 2 | 2461 | 1 |  |
| FOMACOLOR/fomapan_cine_100.pdf | 2 | 1792 | 1 |  |
| FOMACOLOR/photographic-emulsion.pdf | 2 | 3662 | 2 |  |
| KONICA/VX200.pdf | 3 | 1419 | 3 |  |
| KONICA/VX400.pdf | 3 | 1838 | 3 |  |
| KONICA/centuria_pro_400.pdf | 2 | 0 | 0 | IMG |
| KONICA/chrocen200.pdf | 3 | 1862 | 3 |  |
| KONICA/csuper100.pdf | 3 | 1901 | 3 |  |
| KONICA/csuper200.pdf | 3 | 1888 | 3 |  |
| KONICA/csuper800.pdf | 3 | 1975 | 3 |  |
| KONICA/professional_160.pdf | 4 | 0 | 0 | IMG |
| MACO/TAcube4e.pdf | 8 | 3574 | 8 |  |
| MACO/TAgenie.pdf | 2 | 2989 | 2 |  |
| MACO/TApo1cD.pdf | 8 | 3236 | 7 |  |
| MISC/ColorChecker_Passport_Technical_Report.pdf | 8 | 1857 | 1 |  |
| MISC/Photographic_Emulsions-EJ_Wall-1929.pdf | 267 | 1365 | 86 |  |
| ORWO/Data_sheet_Wolfen_NC400.pdf | 2 | 745 | 2 |  |
| ORWO/Data_sheet_Wolfen_NC500.pdf | 2 | 869 | 2 |  |
| ORWO/Datasheet_DN21.pdf | 2 | 2028 | 2 |  |
| ORWO/Datasheet_DP31.pdf | 2 | 1924 | 2 |  |
| ORWO/Datasheet_NP100.pdf | 3 | 833 | 2 |  |
| ORWO/Datasheet_PF2_V3.pdf | 2 | 2117 | 2 |  |
| ORWO/Datasheet_UN54.pdf | 2 | 2157 | 2 |  |
| ORWO/rgschwind_digital.pdf | 40 | 1508 | 5 |  |
| ROLLEI/Development_Rollei films.pdf | 1 | 1417 | 1 |  |
| ROLLEI/PAN25eng.pdf | 2 | 1743 | 2 |  |
| ROLLEI/TARIRe.pdf | 2 | 2021 | 1 |  |
| SOVIET/Gurlev_sprav_svetotexnika_materialy.pdf | 367 | 0 | 0 | IMG |
| SOVIET STANDARDS/gost_10691.6-88.pdf | 6 | 1 | 0 | IMG |
| SOVIET STANDARDS/ГОСТ 10691.6-88.pdf | 6 | 1191 | 6 |  |
| SOVIET STANDARDS/ГОСТ 11079-76.pdf | 6 | 910 | 4 |  |
| SOVIET STANDARDS/ГОСТ 2 784 7-88.pdf | 14 | 1143 | 12 |  |
| SOVIET STANDARDS/ГОСТ 20904-82.pdf | 4 | 1230 | 4 |  |
| SOVIET STANDARDS/ГОСТ 20945-80.pdf | 10 | 1354 | 9 |  |
| SOVIET STANDARDS/ГОСТ 21998-76.pdf | 18 | 823 | 17 |  |
| SOVIET STANDARDS/ГОСТ 25636-83 (2).pdf | 7 | 1155 | 7 |  |
| SOVIET STANDARDS/ГОСТ 25636-83.pdf | 10 | 834 | 8 |  |
| SOVIET STANDARDS/ГОСТ 25704-83.pdf | 20 | 823 | 20 |  |
| SOVIET STANDARDS/ГОСТ 25968-83.pdf | 12 | 1164 | 11 |  |
| SOVIET STANDARDS/ГОСТ 26569-85.pdf | 26 | 1212 | 24 |  |
| SOVIET STANDARDS/ГОСТ 2818-91.pdf | 9 | 1694 | 9 |  |
| SOVIET STANDARDS/ГОСТ 2819-84.pdf | 7 | 1983 | 7 |  |
| SOVIET STANDARDS/ГОСТ 4896-80.pdf | 4 | 1099 | 4 |  |
| SOVIET STANDARDS/ГОСТ 8761-75.pdf | 4 | 1495 | 4 |  |
| SOVIET STANDARDS/ГОСТ 9160-91.pdf | 26 | 1528 | 22 |  |

### Yield verdict by tier
**TIER A — primary parameter sources (read in depth):**
- `AGFA/FPD1e.pdf` (12pp, EN) and its German twin `AGFA/agfa-aERRKF-Datenblatt_F_PF_D4.pdf` — the single richest file in scope. Full spec block (speed / RMS / resolving power at both test contrasts / layer thickness / base thickness / reciprocity table / DX code) for Optima 100/200/400, Portrait 160, RSX II 50/100/200, APX 100/400, Scala 200x.
- `AGFA/agfa_film_chem.pdf` (=`agfa_bw_film_chemicals_en.pdf`) — APX 25/100/400 gamma-vs-time developer tables.
- `AGFA/agfa_bw_manual.pdf` (69pp) — APX/Scala RMS with stated measuring developer; rest of the book is paper/chemistry (Dmax figures on p18 are PAPER, not film — do not use as film Dmax).
- `KONICA/csuper100/200/800.pdf`, `KONICA/chrocen200.pdf`, `KONICA/VX200.pdf`, `KONICA/VX400.pdf` — full technical data sheets: RMS + aperture, resolving power at 1.6:1 and 1000:1, reciprocity range, ISO under 4 illuminants with filters, base, process.
- `ORWO/Datasheet_{DN21,DP31,NP100,PF2_V3,UN54}.pdf` — RMS + 48 µm aperture, MTF m30, average gradient g, base type/thickness.
- `FOMACOLOR/{fomapan-200,fomapan_cine_100,fomaortho,foma-cine-ortho}.pdf` — RMS with stated developer and gamma, resolving power, exposure latitude in EV, base per format, 5500 K reference.
- `ROLLEI/TARIRe.pdf`, `ROLLEI/PAN25eng.pdf`, `ROLLEI/Development_Rollei films.pdf` — ISO, RMS, resolving power, spectral range, base, full developer/time matrix incl. R3 and Retro 400.
- `MACO/TAcube4e.pdf`, `MACO/TApo1cD.pdf`, `MACO/TAgenie.pdf` — spectral range in nm, nominal + effective ISO span, base, gamma target 0.65, resolving power.
- `FERRANIA/FP3011_Datasheet.pdf` + `FERRANIA/1579.pdf` — P30 box speed, silver load, 18-row developer/dilution/temp/time chart.
- `SOVIET STANDARDS/ГОСТ 20945-80.pdf` — the ONLY GOST in scope with a real per-type spec table (B&W reversal cine ОЧ-50 / ОЧ-200): speed, contrast coefficient, Dmax, Dmin, latitude, resolving power.
- `DUFAYCOLOR/dufaycolorbook00dufa.pdf` + `DUFAYCOLOR/Carson_Dufaycolor_Kinotechnik_1934_conv_print.pdf` — Dufaycolor speed in period units and réseau line geometry.

**TIER B — methodology / measurement-condition standards (no film data, but they define how the numbers above were measured; useful to the engine):**
- `ГОСТ 25968-83` RMS granularity method; `ГОСТ 2819-84` resolving-power method; `ГОСТ 2818-91` spectrosensitometric test; `ГОСТ 9160-91` general sensitometry of multilayer colour; `ГОСТ 27847-88` exposure conditions; `ГОСТ 10691.6-88` speed numbers for phototechnical films; `ГОСТ 25636-83`, `ГОСТ 26569-85` marking/packing; `ГОСТ 4896-80`, `ГОСТ 8761-75`, `ГОСТ 20904-82`, `ГОСТ 25704-83`, `ГОСТ 11079-76`, `ГОСТ 21998-76` dimensions/perforation/control films.
- `aimm.it2.18.1996.pdf` (ANSI/AIIM), `5_1_FilmBaseGuide_2020.pdf` (NEDCC base identification: nitrate/acetate/polyester dating), `ГОСТ 25636-83 (2).pdf` (reflection density conditions).

**TIER C — large but off-topic for parameters:**
- `AGFA/These-23-11-09fusion2.pdf` (686pp) — French doctoral thesis on private film collecting / archive law. Sampled pp. 6, 190, 200, 244: no emulsion parameters anywhere. **Effectively zero yield despite 64 keyword hits (all incidental: "base de données", "nm", "min").** Do not mine further.
- `The Permanence and Care of Color Photographs.pdf` (761pp, Wilhelm) — dye-fading/permanence, not emulsion sensitometry. Useful only if you later want dye-stability or dark-fading data. 376 keyword hits are almost all incidental.
- `MISC/Photographic_Emulsions-EJ_Wall-1929.pdf` (267pp) — emulsion-making chemistry, 1929; no branded stock parameters.
- `ORWO/rgschwind_digital.pdf` (40pp) — colour-reconstruction research paper; has spectral-sensitivity *discussion* and fading test conditions (70–80 °C, 380–730 nm in 10 nm steps) but no stock data.
- `MISC/ColorChecker_Passport_Technical_Report.pdf` — target colorimetry, not film.
- `FOMACOLOR/photographic-emulsion.pdf` — liquid coating emulsion, no sensitometry (melt 35–40 °C, 3–6 m²/kg, dry ≤50 °C).
- `AGFA/agfa-Alliance-IR-Technical-Infosheet.pdf` — imagesetter recording film; only "spectral sensitivity: infrared (780 nm)" and dot-gain D4.00–D4.20.

---
## 2. Per-film parameter table
Every cell cites file:page. Blank = not printed in this scope.

### AGFA — colour negative (all `AGFA/FPD1e.pdf`, German dup `agfa-aERRKF-Datenblatt_F_PF_D4.pdf`)
| film | speed | RMS (×1000, 48 µm, D=1.0, daylight, Vλ) | resolving 1000:1 | resolving 1.6:1 | layer thickness | base | reciprocity | process |
|---|---|---|---|---|---|---|---|---|
| Agfacolor Optima 100 | ISO 100/21° (p7) | 4.0 (p7) | 140 l/mm (p7) | 50 l/mm (p7) | 16 µm (p7) | 135 = 120 µm, 120 = 95 µm (p7) | no change 1/10000–1 s; +½ … +1½ stop beyond (p6) | AP 70 / C-41 (p4) |
| Agfacolor Optima 200 | ISO 200/24° (p7) | 4.3 (p7) | 130 l/mm (p7) | 50 l/mm (p7) | 18 µm (p7) | 120 µm / 95 µm (p7) | 1/10000–1 s; +1 … +2 (p6) | AP 70 / C-41 |
| Agfacolor Optima 400 | ISO 400/27° (p7) | 4.5 (p7) | 130 l/mm (p7) | 50 l/mm (p7) | 19 µm (p7) | 120 µm / 95 µm (p7) | 1/10000–1 s; +1 … +2 (p6) | AP 70 / C-41 |
| Agfacolor Portrait 160 | ISO 160/23° (p6) | 3.5 (p6) | 150 l/mm (p6) | 60 l/mm (p6) | 18 µm (p6) | 135 = 120 µm, 120/220 = 95 µm (p6) | 1/10000–1 s; +1 … +2 (p6) | AP 70 / C-41 |

Common to all Agfa Professional colour films (FPD1e p2–p3): balanced for **5500 K** mixed sunlight;
white point fixed in manufacture; UV-blocking layer built into the emulsion (no UV filter needed);
production tolerance ±0.5 DIN (=±1/6 stop) speed, ±5 CC colour balance (p1);
layer stack Optima 100 (p5): supercoat / UV filter / blue-sens. yellow / yellow filter / green-sens. magenta /
red filter / red-sens. cyan / anti-halo / base, total 16 µm.
Filter/Kelvin corrections (p3): 5700 K → 81A +1/3 stop; 5300 K → 82A +1/3; 3400 K photo lamp → 80B +1 1/3;
3200 K → 80A +2; fluorescent D → 50R +1, W → 40M +2/3, cold-white → 20C+40M +1, warm-white → 40M+10Y +1.
Colour-density curve reference (p5): exposure daylight 1/100 s, process AP 70/C-41 or AP 44/E-6, Status A and Status M.

### AGFA — colour reversal (FPD1e p8)
| film | speed | RMS | 1000:1 | 1.6:1 | layer | base | reciprocity |
|---|---|---|---|---|---|---|---|
| Agfachrome RSX II 50 | ISO 50/18° | 10.0 | 135 l/mm | 55 l/mm | 25 µm | 120/95 µm | +½ … +1 stop, CC 05B/10B |
| Agfachrome RSX II 100 | ISO 100/21° | 10.0 | 130 l/mm | 50 l/mm | 25 µm | 120/95 µm | +½ … +1 stop, CC 05B/10B |
| Agfachrome RSX II 200 | ISO 200/24° | 12.0 | 120 l/mm | 50 l/mm | 27 µm | 120/95 µm | +1 … +2 stop, CC 075Y/15Y/05C |
Process AP 44 / E-6 (p4). Push/pull ±1 stop keeps colour neutrality fully (p4).

### AGFA — B&W
| film | speed | RMS | resolving | layer | base | gamma / developer | source |
|---|---|---|---|---|---|---|---|
| Agfapan APX 25 Professional | ISO 25/15° (also pushed to ISO 50/18°) | — | — | — | — | γ 0.55 / 0.65 / 0.75 time tables for Rodinal Special, Studional, Refinal, Rodinal 1+25 at 18–24 °C | `agfa_film_chem.pdf` p4, p7, p8 |
| Agfapan APX 100 | ISO 100/21° | 9.0 (Refinal 6 min 20 °C) | 150 l/mm @1000:1 | 7 µm | 135 = 120 µm, 120 = 95 µm | γ 0.65 nominal; Refinal 6 min→ISO 125/22°, Rodinal 1+25 8 min→125/22°, Rodinal 1+50 17 min→**160/23°**, Rodinal Special 4 min→125/22°, Studional 4 min→125/22° (all small tank 20 °C) | `FPD1e.pdf` p9, p10; `agfa_bw_manual.pdf` p12, p39–46 |
| Agfapan APX 400 | ISO 400/27° | 14.0 (Refinal 6 min 20 °C) | 110 l/mm @1000:1 | 10 µm | 120 µm / 95 µm | γ 0.65 nominal; Refinal 5 min→ISO 400/27°, Rodinal 1+25 10 min→320/26°, Rodinal 1+50 30 min→320/26°, Rodinal Special 6 min→400/27°, Studional 6 min→400/27°; Refinal small tank→ISO 500/28° | `FPD1e.pdf` p9, p10; `agfa_bw_manual.pdf` p12, p46 |
| Agfa Scala 200x | ISO 200/24° standard | 11.0 (Scala process) | 120 l/mm @1000:1, 50 l/mm @1.6:1 | 7 µm | 135 = 120 µm, 120 = 95 µm, sheet = **PET 175 µm** | special Scala process, authorised labs only | `FPD1e.pdf` p4, p9; `agfa_bw_manual.pdf` p12 |

Scala 200x push/pull ladder (FPD1e p4): Pull 1 = ISO 100/21°, Standard 200/24°, Push 1 = 400/27°,
Push 2 = 800/30°, Push 3 = 1600/33°. Contrast steepens and Dmax **decreases** with push; pull gives
higher Dmax and −10 % granularity at ISO 100/21°.
APX B&W reciprocity (FPD1e p6): exposure reading 1/10000–½ s or 1 s → +1/+2/+3 stops with
development reduced −10 %/−25 %/−35 % respectively (both APX 100 and 400).
APX developing-time framework (`agfa_film_chem.pdf` p4): all times aim at mean contrast γ 0.65;
γ 0.55 for very contrasty subjects, γ 0.75 for flat subjects. Atomal FF +10 %.
RODINAL SPECIAL at 20 °C (p7): APX 25 γ0.65 = 3 min, APX 100 γ0.65 = 3.5 min, APX 400 γ0.65 = 4 min;
γ0.75 = 5 min for all three; APX 400 push to ISO 1250/32° tabulated.

### AGFA — cine / aerial (not our stocks, documented for completeness)
| film | data | source |
|---|---|---|
| AGFA PRINT CP30 colour print | 120 µm permanent-antistatic GEVAR **polyester** base; green/red/blue-sensitive → magenta/cyan/yellow; +2–3 trimmer points speed and higher contrast vs CP20; process ECP-2E (incl. version without first fix / silver redevelopment); reference exposure tungsten **3200 K**; RMS granularity, MTF, spectral sensitivity and spectral dye density curves present as graphs | `Agfa_TechnicalDataCP30ColourPrintFilm_17.10.2011.pdf` p1–p5 |
| AGFA Aviphot Pan 80 | panchromatic **to 750 nm**; exposable **64–100 ASA**; polyester base PE1 0.10 mm, PE0 0.06 mm; resolving power **287 lp/mm (574 dots/mm) at TOC 1000:1**; RMS from microdensitometric scan with **50 µm** spot; sensitivity = reciprocal of mJ/m² for D=1.0 above fog; process Gevatone 66, G 74 c, 30 °C, 20–70 s (42 s reference); average-gradient/time curves present | `UT-420343_AGFA_Aviphot-Pan80.pdf` p1–p6 |

### KONICA (full data sheets; RMS aperture 48 µm ø at NET diffuse density 1.0)
| film | speed | RMS | 1000:1 | 1.6:1 | reciprocity | base | process | Kelvin / filters |
|---|---|---|---|---|---|---|---|---|
| Konica Color CENTURIA SUPER 100 | ISO 100/21°; tungsten 3200 K = 32/16°, fluor. = 25/15° | 4 | 125 l/mm | 63 l/mm | 1/10000–1 s no loss | triacetate | CNK series / C-41 | daylight; 80B / 80A |
| Konica Color CENTURIA SUPER 200 | ISO 200/24°; tungsten = 64/19°, fluor. = 50/18° | 4 | (graph) | (graph) | 1/10000–1 s | triacetate | CNK / C-41 | daylight; 80B / 80A |
| Konica Color CENTURIA SUPER 800 | ISO 800/30° | 5 | 100 l/mm | 50 l/mm | — | triacetate | CNK-4 / C-41 | daylight |
| Konica Chrome CENTURIA 200 | ISO 200/24°; tungsten 3200 K = 50/18° w/ 80A → 100/21°; fluor. D = 120/22° w/ CC10M+CC30R; W w/ CC30M | 11 | 125 l/mm | 50 l/mm | 1/10000–4 s no loss; >4 s: +2/3 stop CC05C, +1 stop CC10C | triacetate | **CRK-2 or E-6** | daylight/flash |
| Konica Color VX200 | ISO 200/240 (sic, as printed) balanced for daylight | 4 | 100 l/mm | 50 l/mm | 1/10000–1 s | triacetate | CNK-4 / C-41 | 80B, 80A (3200 K) |
| Konica Color VX400 | ISO 400/27° | 4 | 100 l/mm | 50 l/mm | 1/10000–1 s | triacetate | CNK-4 / C-41 | 80B, 80A |
Chrome Centuria 200 layer stack (p1): protective / blue-sens. / yellow filter / magenta filter / green-sens. /
interlayer / red-sens. / interlayer / anti-halation / base. Densitometry Status A (chrome) and Status M (negative);
characteristic curves from daylight 1/125 s. Spectral sensitivity, spectral dye density and MTF curves present as graphs.

### ORWO / FILMOTEC (all RMS read at net visual diffuse density 1.0 above Dmin with **48 µm aperture**)
| film | speed | RMS | MTF m30 | average gradient g | base | process |
|---|---|---|---|---|---|---|
| ORWO DN 21 duplicating negative | — | < 9 | ≥ 0.80 | **g = 0.65** | polyester 125 µm, AHU under-layer | ORWO instruction 1182 (D96), curves at 4.5 / 6.0 / 7.5 min |
| ORWO DP 31 duplicating positive | — | < 9 | ≥ 0.70 | **g = 1.6** | polyester 125 µm, clear, AHU | 1182 (D96) |
| ORWO PF 2 V3 print film | — | ≤ 10 | ≥ 0.70 | **g = 2.8** | polyester 125 µm clear or triacetate 135 µm | instruction 2182 (D97), curves 3 / 5 / 7 min |
| WOLFEN NP100 negative | (ISO 100 class; p1) | 12 (average) | > 0.80 | 0.65 | triacetate, grey, 135 µm | 1182 (D96) |
| ORWO UN 54 universal negative | **ISO 100/21°** | 12 (average) | (graph) | 0.65 | triacetate, grey, 135 µm | 1182 (D96); reversal per instruction 4185 |
| WOLFEN NC400 colour negative | **ISO 400/27°** | — | — | — | triacetate 125 µm, grey | **C-41**; exposure daylight |
| WOLFEN NC500 colour negative | **ISO 400/27°** | — | — | — | triacetate 125 µm, grey | C-41 (no remjet); ECN-2 gives "correspondingly flatter gradation" |
Spectral-sensitivity graphs (equal-energy spectrum, 400–700 nm) present for DN21, DP31, PF2, UN54, NP100.

### FOMA (RMS all measured in Microphen at 20 °C developed to γ = 0.6, read at D = 1.0)
| film | speed | latitude as printed | RMS | resolving | base | reference exposure |
|---|---|---|---|---|---|---|
| FOMAPAN 200 Creative | ISO 200/24° (24° ČSN) | +1 EV → ISO 100/21°, −2 EV → ISO 800/30°, no dev change | **14** | **110 lines/mm** | rollfilm clear polyester 0.1 mm; 35 mm grey/grey-blue/pink triacetate; sheet clear polyester 0.175 mm | daylight 5500 K, 1/20 s |
| FOMAPAN Cine 100 | ISO 100/21° (21° ČSN) | +1 EV → 50/18°, −2 EV → 400/27° | **13.5** | **110 lines/mm** | grey / grey-blue triacetate | daylight 5500 K, 1/20 s |
| FOMA ORTHO 400 | ISO 400/27° (27° ČSN) | +1.5 EV → 160/23°, −2 EV → 1600/33° | **17.5** | (graph) | 120 = bluish polyester 0.1 mm no AH; 35 mm = grey/grey-blue triacetate 0.125 mm; sheet = clear polyester 0.175 mm with dischargeable AH backing | daylight 5500 K, 1/20 s |
| FOMA Cine ORTHO 400 | ISO 400/27° (27° ČSN) | +1.5 EV → 160/23°, −1.5 EV → 1250/32° | **17.5** | (graph) | grey or grey-blue triacetate | daylight 5500 K, 1/20 s |
Ortho films: safelight ≥ 585 nm (orange); high green sensitivity; reversal processing possible via
"Processing set for FOMAPAN R-100". Fixing 18–25 °C 10 min; wash 30 min <15 °C or 15 min warmer.

### ROLLEI / MACO
| film | speed | RMS | resolving | spectral range | base | developer data |
|---|---|---|---|---|---|---|
| ROLLEI INFRARED (400) | **ISO 400/27°** | **11.0** (Refinal, 5 min, 20 °C) | **160 lines/mm at contrast 1000:1** | panchromatic with special IR sensitivity | clear polyester 100 µm, LE 500; protective layer + emulsion + PET 100 µm | RHS 1+7 6 min, RHS 1+12 8:30, RLS 1+4* 18, Rodinal 1+25 7:30 / 1+50 12, D76/ID11 stock 6 (20 °C, 30 s inversion cycles) |
| ROLLEI PAN 25 | **ISO 25/15°** | — | "very good" (no number) | **400–650 nm** | polyester 100 µm crystal clear, LE 500 | RHS 1+7 5, RHS 1+12 7, RLS 1+4* 10, RLC 1+4 7, AM50 1+29 5, Rodinal 1+25 6 / 1+50 11, D76/ID11 5 |
| ROLLEI RETRO 400 | ISO 400 (also rated 200) | — | — | — | — | at ISO 400: RHS 1+7 6, RHS 1+12 8:30, RLC 1+4 8:30, AM50 11, Rodinal 1+25 10 / 1+50 13, D76 9; at ISO 200: RLS 1+4* 12, RLC 1+4 7:30 |
| ROLLEI RETRO 100 | ISO 100 (also 50) | — | — | — | — | ISO 100: RHS 1+7 6, RHS 1+12 8:30, RLS 12(?), RLC 8:30, AM50 10, Rodinal 1+25 7 / 1+50 10, D76 9 |
| ROLLEI R3 | rated **50 / … / 800 / 1600** in the same table (variable-speed emulsion) | — | — | — | — | 2 min presoak recommended; ISO 50 → RHS 1+7 12 min; ISO 800 (spot metering) → 18 / 25; ISO 1600 → 22 |
| ROLLEI ORTHO 25 | ISO 25 | — | — | — | — | RHS 1+7 6, RHS 1+12 8:30, RLS 12, RLC 8:30, Rodinal 1+25 4 / 1+50 6, D76 6 |
| ROLLEI SUPER PAN 200 | ISO 200 | — | — | — | — | 6:30 / 8 / 12 / Rodinal 8 & 17 |
| ROLLEI ATP-1.1 | ISO 25 (also 15) | — | — | — | — | 5 / 6:00 / 6:30 |
| MACO CUBE 400c | nominal **ISO 400/27°**, effective **ISO 100/21° – 6400/39°** by developer/time choice | — | (graph) | **extended panchromatic ≈ 380 nm to 710/730 nm** | polyester 100 µm blue (roll/35 mm), 175 µm blue or clear (sheet) | times determined for a stated gamma; reciprocity (Schwarzschild) diagram present; user test at ISO 50/18°: 1 min prewash 24 °C then LP CUBE XS 1+4, 19.5 min at 24 °C |
| MACO ORT 25 / "GENIUS PRINT" (TAgenie) | nominal **ISO 25/15°** | — | **330 Lp/mm at nominal speed, contrast 1:1000** | — | polyester 175 µm clear | — |
| MACO ORTHO-PAN portrait film (TApo1cD) | daylight **ISO 100/21° – 200/24°** (5400 K); tungsten **ISO 50/18° – 100/21°** | — | claims 250 % higher resolving power than panchromatic MACO films of same speed | **orthopanchromatic ≈ 380–600 nm** | polyester 100 µm | times determined for **gamma 0.65**; fix 3 min 20 °C |
`ROLLEI/Development_Rollei films.pdf` note: RHS = AM74-compatible, RLS = CG512-compatible; all times 20 °C except * = 24 °C.

### FERRANIA
| film | speed | silver | developer chart (as printed) | notes |
|---|---|---|---|---|
| FERRANIA P30 (original, "cinema") | **ISO 80/20°**, box speed; EI 50/18 alternative column | **~5 g/m²** | at **EI 80**: Kodak D-76 stock 20 °C; D-96 stock 21 °C; Ilfosol 3 1:9 20 °C; HC-110 1:63 (H) and 1:31 (B) 20 °C; TMAX 1:6 24 °C; R09/Rodinal 1:100 20 °C semi-stand (3 min presoak, 60 s initial agitation, gentle at 15/30/45 min) and 1:50 → **14 min**; Ilford DD-X 1:5 20 °C → 7.5 min (EI 50), 1:6 20.5 °C → 15 min; ID-11 1:1 20 °C → **13.5 min**; MICROPHEN 1:3 20 °C → **17 min**; XTOL 1:1 → **12 min**, 1:3 → **16 min**; Perceptol stock → **9 min**; Promicrol 1:9 → **8 min**, 1:14 20.5 °C → 8.5 min (EI 50); Paranol S 1:4 24 °C → **11 min** | not DX coded; higher contrast, **low red sensitivity** like 1950s panchromatics; Mk2 = same DNA, smoother tonal transitions | `FP3011_Datasheet.pdf` p1–p3; `1579.pdf` p1–p2 |
| FERRANIA ORTO | **ISO 50/18°** | — | Kodak D-76 stock 20 °C 8 min at nominal speed (`Curve caratteristiche…` p2) | sensitive to UV, blue, green only |
| FERRANIA P33 | **ISO 160/23°** | — | 1–2 min presoak | `1579.pdf` p1, p4 |

### DUFAYCOLOR
| parameter | value as printed | source |
|---|---|---|
| speed, Type D.i roll film / flat film | **400 H&D**; **17 Weston**; **17° Scheiner** (also quoted as 170 Scheiner on several meter scales); **DIN 15/10 in full sunlight, DIN 9/10 other exposures and indoor**; 24° Scheiner and 19°/16–19° Scheiner on other meters; "Group C"/"Class C" | `dufaycolorbook00dufa.pdf` p22 |
| réseau geometry | ~**one million raster elements per square inch**; formerly **15 lines/mm**, improved to **19 lines/mm**; printing roller **1000 lines/inch = 500 colour lines + 500 gaps**; 16 mm version announced with a **400-line** réseau | `Carson_Dufaycolor_Kinotechnik_1934_conv_print.pdf` p1–p3 |
| structure | réseau on the base, thin protective layer over it, then panchromatic emulsion; light passes through the réseau **before** reaching the emulsion; additive process, reversal-processed to a black silver deposit behind each filter element | `dufaycolorbook00dufa.pdf` p9–p14 |
| exposure practice | Type D.a requires a filter under all conditions; Type D.i needs none in daylight; expose for the highlights (reversal), do not apply negative-work subject factors | `dufaycolorbook00dufa.pdf` p15, p17–p21 |
| developer | hydroquinone-ammonia developer of fairly high concentration | `Carson_1934` p3 |
| panchromatic separation reference | flat film through Da/2 filter; panchromatic plates through Ilford tricolour filters | `dufaycolorbook00dufa.pdf` p8 |

### SOVIET — ГОСТ 20945-80, B&W reversal cine film specifications (types ОЧ-50, ОЧ-200)
Columns as printed: ОЧ-50 higher-category / ОЧ-50 first-category / ОЧ-200 higher / ОЧ-200 first.
| parameter | ОЧ-50 | ОЧ-200 | source |
|---|---|---|---|
| Nominal speed S₀.₉ on the reversed image, GOST 10691.4-84 units | **50** | **200** | p4 |
| General speed S₀.₉ range | **50–80** | **200–320** | p4 |
| Contrast coefficient (коэффициент контрастности) | **1.2–1.6** | **1.2–1.6** | p4 |
| Dmax, not less than | **2.2 / 2.0** | **2.4 / 1.8** (column order as printed; verify against original scan) | p4 |
| Dmin incl. base, not more than — colourless base | **0.10 / 0.11** | **0.11 / 0.13** | p4 |
| Dmin incl. base — tinted base | **0.15 / 0.16** | **0.16 / 0.18** | p4 |
| Photographic latitude L, not less than | **0.9 and 1.05** printed for the two types (assignment ambiguous in the text layer) | — | p4 |
| Resolving power R, lines/mm, not less than | **110 / 100** | **95 / 82** | p4 |
| Base | triacetate cellulose, colourless or blue-tinted; base optical density ≤ **0.05** colourless, ≤ **0.10** tinted | | p4 |
| Shrinkage | ≤ **0.3 %** longitudinal and transverse | | p5 |
| Melting point of swollen emulsion | ОЧ-50 ≥ 70 °C (higher category ≥ 80 °C); ОЧ-200 ≥ 50 °C (higher category ≥ 100 °C) | | p4 |
| Sensitometric exposure | on a sensitometer behind an **artificial-daylight filter, 0.05 s**; bleach = potassium dichromate 1.2–1.6 g + H₂SO₄ 5.0 cm³ / 1000 cm³; clearing bath = anhydrous sodium sulphite 50 g / 1000 cm³ | | p6 |
| Ageing allowance | within warranty: speed may fall ≤ 30 %, Dmax ≤ 0.2 below the norms | | p4 |

---
## 3. (a) OUR stocks that gained real documented parameters
| our stock | documented by | what was gained |
|---|---|---|
| AGFA_APX_25 | `AGFA/agfa_film_chem.pdf` p4, p7, p8 | ISO 25/15° (+ push to 50/18°); γ 0.55/0.65/0.75 developing times for Rodinal Special, Studional Liquid, Refinal, Rodinal 1+25 across 18–24 °C |
| AGFA_APX_100 | `AGFA/FPD1e.pdf` p9–p10; `agfa_bw_manual.pdf` p12, p39–46; `agfa_film_chem.pdf` p7 | ISO 100/21°; RMS 9.0; 150 l/mm @1000:1; layer 7 µm; base 120/95 µm; γ 0.65 framework; 5 developer→speed pairs (Rodinal 1+50 → ISO 160/23°); reciprocity +1/+2/+3 stops with −10/−25/−35 % development |
| AGFA_APX_400 | same | ISO 400/27°; RMS 14.0; 110 l/mm @1000:1; layer 10 µm; developer→speed pairs incl. ISO 320/26° and 500/28°; push to ISO 1250/32° |
| AGFA_OPTIMA_100 | `AGFA/FPD1e.pdf` p7 (+ p5 layer stack) | ISO 100/21°; RMS 4.0; 140/50 l/mm; 16 µm total; base 120 µm (135) / 95 µm (120); full layer order |
| AGFA_OPTIMA_200 | `AGFA/FPD1e.pdf` p6–p7 | ISO 200/24°; RMS 4.3; 130/50 l/mm; 18 µm; reciprocity +1…+2 |
| AGFA_OPTIMA_400 | `AGFA/FPD1e.pdf` p6–p7 | ISO 400/27°; RMS 4.5; 130/50 l/mm; 19 µm |
| AGFA_PORTRAIT_160 | `AGFA/FPD1e.pdf` p6 | ISO 160/23°; RMS 3.5; 150/60 l/mm; 18 µm; base 120/95 µm |
| AGFA_SCALA_200X | `AGFA/FPD1e.pdf` p4, p9; `agfa_bw_manual.pdf` p12 | ISO 200/24°; RMS 11.0; 120/50 l/mm; layer 7 µm; PET 175 µm sheet base; full push/pull ladder 100→1600 with contrast/Dmax/granularity direction |
| KONICA_CENTURIA_SUPER_400 | **partially, by family only** — `KONICA/csuper100.pdf`, `csuper200.pdf`, `csuper800.pdf` bracket it (RMS 4→5, 63→50 l/mm @1.6:1, 125→100 l/mm @1000:1, CNK-4/C-41, triacetate, reciprocity 1/10000–1 s) | no 400 sheet in scope; use bracket only if you accept interpolation (**not printed**) |
| KONICA_CHROME_CENTURIA_100 | **adjacent only** — `KONICA/chrocen200.pdf` gives the 200 version: RMS 11, 125/50 l/mm, CRK-2 or E-6, reciprocity 1/10000–4 s then CC05C/CC10C, tungsten/fluorescent ISO+filter table | 100 not in scope |
| KONICA_VX_100 | **adjacent only** — `KONICA/VX200.pdf`, `VX400.pdf` (RMS 4, 100/50 l/mm, CNK-4/C-41, triacetate, 80B/80A) | 100 not in scope |
| FOMAPAN_400_ACTION | **adjacent only** — `FOMACOLOR/fomapan-200.pdf` and `fomapan_cine_100.pdf` give the measurement convention (RMS in Microphen 20 °C to γ=0.6 at D=1.0, resolving power in lines/mm, 5500 K 1/20 s reference, base per format) | no 400 Action sheet in scope |
| ROLLEI_INFRARED_400 | `ROLLEI/TARIRe.pdf` p2; `ROLLEI/Development_Rollei films.pdf` | ISO 400/27°; **RMS 11.0** (Refinal 5 min 20 °C); **160 l/mm @1000:1**; clear polyester 100 µm LE 500; full developer/time row |
| ROLLEI_R3 | `ROLLEI/Development_Rollei films.pdf`; context in `MACO/TAcube4e.pdf` | variable-speed rows ISO 50 / 800 / 1600 with times; 2 min presoak; 20 °C (*24 °C) convention. Same-family MACO CUBE 400c gives spectral range 380–710/730 nm and polyester 100 µm blue base |
| ROLLEI_RETRO_400 | `ROLLEI/Development_Rollei films.pdf` | ISO 400 and ISO 200 developer/time rows across 7 developers |
| FERRANIA_P30 | `FERRANIA/FP3011_Datasheet.pdf` p1–p3; `FERRANIA/1579.pdf` p1–p2 | **ISO 80/20°** box speed (EI 50 alternative); **~5 g silver/m²**; 18-row developer × dilution × temperature × time chart; documented low red sensitivity / high contrast character |
| DUFAYCOLOR_1937 | `dufaycolorbook00dufa.pdf` p8–p22; `Carson_…_1934` p1–p3 | speed 400 H&D / 17 Weston / 17°(170) Scheiner / DIN 15-10 sun, 9-10 other; **réseau 19 lines/mm (was 15), ~1e6 elements per in², roller 1000 lines/inch = 500 colour lines**; layer order; hydroquinone-ammonia developer |
| TASMA_OCH_45 | **adjacent, new** — `SOVIET STANDARDS/ГОСТ 20945-80.pdf` p4–p6 | the successor types ОЧ-50/ОЧ-200 get full GOST norms: speed, contrast coefficient 1.2–1.6, Dmax, Dmin (colourless and tinted base), latitude, resolving power 82–110 l/mm, base density limits, 0.05 s artificial-daylight sensitometry, bleach/clear formulas. This is **not** from Gurlev 1986 or ГОСТ 24876-81 |

### (b) All RMS granularity and resolving-power numbers found in this scope
RMS granularity (all at diffuse density 1.0 above Dmin unless noted):
- Agfa: Portrait 160 = 3.5; Optima 100 = 4.0; Optima 200 = 4.3; Optima 400 = 4.5; APX 100 = 9.0; RSX II 50 = 10.0; RSX II 100 = 10.0; Scala 200x = 11.0; RSX II 200 = 12.0; APX 400 = 14.0 (aperture 48 µm, daylight, Vλ) — `FPD1e.pdf` p6–p9, `agfa_bw_manual.pdf` p12
- Konica: VX200 = 4; VX400 = 4; Centuria Super 100 = 4; Centuria Super 200 = 4; Centuria Super 800 = 5; Chrome Centuria 200 = 11 (aperture 48 µm ø, NET density 1.0)
- ORWO: DN 21 < 9; DP 31 < 9; PF 2 V3 ≤ 10; NP100 = 12 avg; UN 54 = 12 avg (aperture 48 µm)
- Foma: Cine 100 = 13.5; Fomapan 200 = 14; Ortho 400 = 17.5; Cine Ortho 400 = 17.5 (Microphen 20 °C, γ 0.6, D 1.0)
- Rollei Infrared = 11.0 (Refinal 5 min 20 °C)
- Agfa Aviphot Pan 80: RMS stated but as a curve; aperture is **50 µm** spot, not 48 µm

Resolving power (lines/mm):
- @1000:1 — Portrait 160 = 150; APX 100 = 150; Optima 100 = 140; RSX II 50 = 135; Optima 200 = 130; Optima 400 = 130; RSX II 100 = 130; Scala 200x = 120; RSX II 200 = 120; APX 400 = 110; Konica CS100 = 125; Chrome Centuria 200 = 125; Konica CS800 = 100; VX200 = 100; VX400 = 100
- @1.6:1 — Portrait 160 = 60; RSX II 50 = 55; Optima 100/200/400 = 50; RSX II 100/200 = 50; Scala 200x = 50; Konica CS100 = 63; Chrome Centuria 200 = 50; CS800 = 50; VX200 = 50; VX400 = 50
- Other — Rollei Infrared = 160 @1000:1; MACO GENIUS/ORT 25 = **330 Lp/mm @1:1000**; Agfa Aviphot Pan 80 = **287 lp/mm (574 dots/mm) @ TOC 1000:1**; Fomapan 200 = 110; Fomapan Cine 100 = 110
- GOST 20945-80 ОЧ-50 = ≥110 / ≥100; ОЧ-200 = ≥95 / ≥82
MTF/modulation transfer m30: ORWO DN 21 ≥ 0.80; NP100 > 0.80; DP 31 ≥ 0.70; PF 2 V3 ≥ 0.70.

### (c) IMAGE-ONLY files (near-zero extractable text — need OCR or visual inspection)
1. `AGFA/Meyer_TechnikDesFarbfilmsVervollkommnen_1953.pdf` — 2pp, 0 c/p
2. `AGFA/NewGevacol_Neg_682.pdf` — 3pp, 0 c/p — **directly one of our stocks (GEVACOLOR_NEG_682); highest OCR priority in scope**
3. `AGFA/Agfacolor Neu/Behrendt_Köllner_Fehlerq_1950.pdf` — 4pp, 0 c/p
4. `AGFA/Agfacolor Neu/Gröger_Hydrolyse_1961.pdf` — 5pp, 0 c/p
5. `AGFA/Agfacolor Neu/SchultzeHörmann_Agfa_1951.pdf` — 6pp, 0 c/p (relevant to AGFACOLOR_NEU_1936 / NEG_TYPE_3 / NEG_TYPE_B_1943)
6. `DUFAYCOLOR/Dufay_Dufaycolor_GB000000262386A.pdf` — 3pp, 0 c/p (patent; réseau manufacture)
7. `DUFAYCOLOR/Dufaycolor_Manual_1938_print.pdf` — 25pp, 0 c/p — **high OCR priority for DUFAYCOLOR_1937**
8. `KONICA/centuria_pro_400.pdf` — 2pp, 0 c/p — **high OCR priority: closest sheet to KONICA_CENTURIA_SUPER_400**
9. `KONICA/professional_160.pdf` — 4pp, 0 c/p
10. `SOVIET/Gurlev_sprav_svetotexnika_materialy.pdf` — 367pp, 0 c/p (already mined in an earlier session by other means)
11. `SOVIET STANDARDS/gost_10691.6-88.pdf` — 6pp, 1 c/p (duplicate of the text-bearing `ГОСТ 10691.6-88.pdf`; skip)
Near-image-only (text layer present but only axis labels — the substance is in the graphics):
12. `FERRANIA/Curve caratteristiche e sensibilità spettrali (1).pdf` — 2pp, 283 c/p. Only "Orto", "nm 380 400 450 500 550", and "sviluppo in Kodak D-76 stock a 20 °C – 8' a sensibilità nominali". **The P30/Orto characteristic and spectral-sensitivity curves themselves need visual digitisation.**
13. `FERRANIA/1579.pdf` p3–p4 — curve figures and the development-time table are graphics; only headers extract.

Also note: every datasheet in Tier A carries spectral-sensitivity, characteristic-curve, MTF and spectral-dye-density **graphs** whose axes extract but whose curves do not. If the engine needs per-layer spectral sensitivity or D-log H shapes for Agfa Optima/Portrait/APX/Scala, Konica, ORWO, Foma or Rollei, those specific pages need curve digitisation:
`FPD1e.pdf` p6–p9, `KONICA/*` p2–p3, `ORWO/Datasheet_*` p2, `FOMACOLOR/*` p1–p2, `Agfa CP30` p5, `Aviphot Pan 80` p2, p5–p6.

### (d) Zero-yield / no-usable-parameter files (do not mine further)
- `AGFA/These-23-11-09fusion2.pdf` (686pp) — French thesis on private film collections and archive law. **Largest file in scope, zero parameters.**
- `The Permanence and Care of Color Photographs.pdf` (761pp) — dye permanence/fading, no emulsion sensitometry.
- `MISC/Photographic_Emulsions-EJ_Wall-1929.pdf` (267pp) — 1929 emulsion-making chemistry, no branded stock data.
- `ORWO/rgschwind_digital.pdf` (40pp) — digital colour reconstruction research; fading test conditions only.
- `MISC/ColorChecker_Passport_Technical_Report.pdf` — target colorimetry.
- `FOMACOLOR/photographic-emulsion.pdf` — liquid coating emulsion, no sensitometry.
- `AGFA/agfa-Alliance-IR-Technical-Infosheet.pdf` — imagesetter film; only "IR (780 nm)" and dot gain.
- `5_1_FilmBaseGuide_2020.pdf`, `aimm.it2.18.1996.pdf` — base identification / archival standards; useful for base material dating only.
- `SOVIET STANDARDS`: `ГОСТ 10691.6-88`, `11079-76`, `27847-88`, `20904-82`, `21998-76`, `25636-83`, `25636-83 (2)`, `25704-83`, `25968-83`, `26569-85`, `2818-91`, `2819-84`, `4896-80`, `8761-75`, `9160-91` — **methodology, dimensional and marking standards; no per-film parameter tables.** (`ГОСТ 20945-80` is the sole exception, see §2.)
- `AGFA/agfa_bw_film_chemicals_en.pdf` is a **near**-duplicate edition of `agfa_film_chem.pdf` (both 16pp, ~96 % identical text) but it **omits APX 25 entirely** — the APX 25 gamma/time data exists only in `agfa_film_chem.pdf`. Use `agfa_film_chem.pdf`; the `_en` file adds nothing.
- `AGFA/agfa-aERRKF-Datenblatt_F_PF_D4.pdf` is the German edition of `FPD1e.pdf` — identical data.

### (e) Films documented here that we do NOT carry
Agfachrome RSX II 50 / 100 / 200 (full spec: speed, RMS, both resolving contrasts, layer thickness, base, reciprocity + CC filters);
Agfa PRINT CP30 colour print film (GEVAR polyester 120 µm, ECP-2E, 3200 K reference);
Agfa Aviphot Pan 80 (panchromatic to 750 nm, 64–100 ASA, 287 lp/mm, PE0/PE1 base, Gevatone 66 / G 74 c 30 °C);
Agfa Alliance Recording IR (780 nm);
Konica Color CENTURIA SUPER 100 / 200 / 800; Konica Chrome CENTURIA 200; Konica Color VX200 / VX400;
ORWO DN 21, DP 31, PF 2 V3, UN 54, WOLFEN NP100, WOLFEN NC400, WOLFEN NC500;
FOMAPAN 200 Creative, FOMAPAN Cine 100, FOMA ORTHO 400, FOMA Cine ORTHO 400;
ROLLEI PAN 25, ROLLEI ORTHO 25, ROLLEI RETRO 100, ROLLEI SUPER PAN 200, ROLLEI ATP-1.1;
MACO CUBE 400c, MACO GENIUS PRINT / ORT 25, MACO orthopanchromatic portrait film (TApo1cD);
FERRANIA ORTO 50, FERRANIA P33 160, FERRANIA P30 Mk2;
Soviet ОЧ-50 and ОЧ-200 B&W reversal cine films (ГОСТ 20945-80).

### Still absent after this scope (no printed data found anywhere in these 71 files)
AGFA_VISTA_200; AGFACOLOR_NEG_TYPE_3; AGFACOLOR_NEG_TYPE_B_1943; AGFACOLOR_NEU_1936;
CINESTILL_800T; FERRANIACOLOR_NEG_82; FERRANIACOLOR_REVERSAL_1950; GENERIC_BW; GENERIC_COLOR;
GEVACHROME_902; GEVACOLOR_1952; GEVACOLOR_NEG_652; GEVACOLOR_NEG_682 (image-only file exists — OCR);
GEVAERT_PANCHRO_1950; KONICA_CENTURIA_SUPER_1600; KONICA_CHROME_R100; KONICA_IMPRESA_50;
KONICA_INFRARED_750; LUMIERE_LUMICHROME; ORWO_CHROM_UT18; ORWOCOLOR_NC21; ORWOCOLOR_NC24;
SOVIET_PANCHROM_1939; TECHNICOLOR_THREE_STRIP; and all SVEMA_* / TASMA_FN_64 (only the adjacent
ГОСТ 20945-80 ОЧ-50/ОЧ-200 norms were added for TASMA_OCH_45).
No Dmin/Dmax figures were printed for **any** Western stock in this scope — the only Dmax/Dmin numbers
found are the ГОСТ 20945-80 ОЧ-50/ОЧ-200 limits. Agfa/Konica/ORWO/Foma give Dmax only as curve graphs.
