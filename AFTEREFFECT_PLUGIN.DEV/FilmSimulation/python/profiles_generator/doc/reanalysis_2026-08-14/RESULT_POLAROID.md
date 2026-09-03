# POLAROID Film Data Sheets — extraction & cross-check
Source root: PDF/PROFILES/POLAROID/ — 53 PDFs from /tmp/list_POLAROID.txt (~198 pp), all TRUE TEXT.
Full text dump: /tmp/POL_TEXT.txt ; vector-path census: /tmp/POL_VEC.txt
RULE: only printed values recorded. "n/p" = not printed anywhere in that sheet.

## 1. PARAMETER TABLE
Polaroid publishes ONE shared "Technical Data" sheet per emulsion family, so a value found in
e.g. 54fds.pdf p2 is the manufacturer value for T-54 / T-554 / T-664 / T-804 alike.

### B&W peel-apart
| Family (sheet: family header) | ISO | Dmin | Slope | Dmax | Resolution lp/mm | Spectral | Dev time/temp | File+page |
|---|---|---|---|---|---|---|---|---|
| T-51HC (4x5 sheet) | 640/29 pos-daylight; 400/27 pos-tungsten; 80/20 neg-daylight; 50/18 neg-tungsten. Tech table restates "Print 640/29 @5500K, Neg 80/20 @5500K" | n/p | n/p (contrast stated: Print High, Neg Medium) | n/p | Print 20-30; Neg 100-120 | Panchromatic | 30 s @70F; 35 s @65F/18C; 50 s @55F/13C (no exp. adj.) | 51fds.pdf p1 (speed/format), p2 (res/spectral/temp table) |
| T-52 (4x5 sheet) + T-552 (4x5 pack) | 400/27 | n/p | n/p (Contrast: Medium) | n/p | 12-15 | Panchromatic | T-52 20 s / T-552 25 s @70F/21C | 52fds.pdf p1, p2 |
| T-53 (4x5 sheet), T-553 (4x5 pack), T-803 (8x10) | 800/30 | .11 | 1.64 | 1.75 | 12-15 | Panchromatic | 45 s @70F; @71F/21C for D-values | 53fds.pdf p2; 553fds.pdf p2; 803fds.pdf p2 |
| Polapan Pro 100: T-54 (4x5 sheet), T-554 (4x5 pack), T-664 (pack), T-84 (pack), T-804 (8x10) | 100/21 | n/p | n/p (Contrast: Medium) | n/p | 20-25 | Panchromatic | 45 s @70F/21C | 54fds.pdf p2; 554/804/84fds.pdf p2 |
| T-55 P/N (4x5 sheet) | 50/18 | .01 (P) / **.18 (N)** | 1.35 (P) / **1.65 (N)** | 1.65 (P) / **1.55 (N)** | 20-25 (print); **160-180 (negative)** | Panchromatic | 20-25 s @70F/21C; 35 s @65F; 50 s @55F; 60 s @50F | 55fds.pdf p1, p2 (positionally verified word coords) |
| T-56 Sepia (4x5 sheet) | 400/27 | n/p | n/p | n/p | 14-20 | Panchromatic | 45 s @70F | 56fds.pdf p1, p2 |
| T-57 (4x5 sheet) | 3000/36 | n/p | n/p (Medium) | n/p | 14-17 | Panchromatic | 15 s @75F/24C | 57fds.pdf p1, p2 |
| T-72 (4x5 sheet), T-572 (4x5 pack), T-672 (pack) | 400/27 | .10 | 1.65 | 1.70 | 20-22 | Panchromatic | 45 s @70F (572/672); 55 s @70F (72) | 72fds.pdf p1,p2; 572fds.pdf p2; 672fds.pdf p2 |
| T-665 P/N (pack), T-85 (pack) | 80/20 | .10 (P) / .23 (N) | 1.40 (P) / .70 (N) | 1.90 (P) / 1.42 (N) | 13-16 (print); 160-180 (negative) | Panchromatic | 30 s @65F/18C and above | 665fds.pdf p1,p2; 85fds.pdf p1,p2 |
| T-87 (pack), **T-667**, Viva 3000 (pack) | 3000/36 | .10 | 1.55 | 1.75 | 14-20 | Panchromatic | 30 s @75F | 87fds.pdf p1,p2; viva3fds.pdf p2 |
| T-606 Sepia (pack) | 200/24 | n/p | n/p | n/p | 14-20 | Panchromatic | 45 s @70F | 606fds.pdf p1,p2 |
| T-611 (pack, coaterless, low contrast) | 200/24 | n/p | n/p | n/p | n/p | n/p | 45 s @68-80F; temp table 30 s @85F+ → 90 s @50F | 611fds.pdf p1 (1-page sheet) |

### Color peel-apart (Polacolor)
| Family | ISO | Dmin | Slope | Dmax | Res lp/mm | Dev | File+page |
|---|---|---|---|---|---|---|---|
| T-88, T-108, T-669, T-669S (pack); T-559/T-559S (4x5 pack); T-59 (4x5 sheet); T-809 (8x10) | 80/20 | n/p | n/p (Contrast: medium) | n/p | 9 | 60 s @70F/21C | 559fds.pdf p2 (shared header); 59/669/809/88fds.pdf p2 |
| Polacolor Pro 100: T-579, T-679, T-879 | 100/21 | .16 | 1.30 | 1.8 (@71F/21C) | n/p | 90 s @70-95F | 579fds.pdf p2; 679fds.pdf p2; 879fds.pdf p2 |
| Polacolor Pro 100 sheet: T-79, Polacolor Pro 100 4x5 (p100) | 100/21 | n/p | n/p (medium) | n/p | 8 | 90 s @70-95F | t79fds.pdf p2; p100fds.pdf p2 |
| T-64 Tungsten (4x5 sheet + pack) | 64/19 (tungsten 3200K balanced) | n/p | n/p | n/p | 10 | 90 s @70F | 64tfds.pdf p1,p2; 64tpfds.pdf p1,p2 |
| T-689 ProVivid (pack) | 100/21 | n/p | n/p | n/p | n/p | 90 s @70F | 689fds.pdf p1 (1-page) |
| T-89 (pack), 690, Studio 125 (125i) | 89: 100/21 on p1 but tech table says 125/22; 690: 100/21 on p1, table 125/22; 125i/Studio: 125/22 | n/p | n/p | n/p | n/p | 90 s @70-105F (89/690/125i) | 89fds.pdf p1+p2; 690fds.pdf p1+p2; 125ifds.pdf p1+p2; studofds.pdf p1,p2 |
| Polacolor ID UV (eiduv) | 80/20 | n/p | n/p | n/p | n/p | 60 s @75F | eiduvfds.pdf p1 |
| Polacolor 100 ID UV (piduv, Europe) | 100/21 | n/p | n/p | n/p | n/p | 90 s @70F | piduvfds.pdf p1 |
| Viva color pack (non-US) | 80/20 | n/p | n/p | n/p | n/p | 60 s @70F | vivafds.pdf p1 |

### Integral color
| Family | ISO | Res lp/mm | Dev | File+page |
|---|---|---|---|---|
| T-600 / 600 Plus, T-600 NotePad, T-600 Write On (satin finish) | 640/29 | 7-10 | 3 min approx | 600plfds.pdf p1,p2; notepfds.pdf; writefds.pdf |
| Spectra / Spectra Grid / T-990 | 640/29 | 7-10 | 2-3 min approx | splatfds.pdf; specgfds.pdf; 990fds.pdf p1,p2 |
| 500 (Captiva/Vision) | 640/29 | 7-10 | 2 min approx | 500fds.pdf p1,p2 |
| i-Zone Pocket Film / Pocket Sticker | 640/29 | 7-10 | 3 min approx | pocktfds.pdf; pktstfds.pdf p1,p2 |
| T-339 AutoFilm, T-779 | 640/29 | 7-10 | 4 min approx | 339fds.pdf p1,p2; 779fds.pdf p1,p2 |

### Image area / format (printed on p1 of each sheet)
4x5 sheet (T-51/52/54/55/56/57/59/64t/72/79/p100): format 4x5 in (10.2x12.7 cm), image area 3-1/2 x 4-1/2 in (9 x 11.4 cm).
4x5 pack (T-553/554/559/572/579): format 4x5 in, image area 3-1/2 x 4-5/8 in (9 x 11.7 cm).
8x10 sheet (T-803/804/809/879): format 8x10 in (20.3x25.4 cm), image area 7-1/2 x 9-1/2 in (19x24 cm).
3-1/4x4-1/4 pack (T-606/611/64tp/665/669/672/679/689/690/89*/125i/eiduv/piduv): image area 2-7/8 x 3-3/4 in (7.3x9.5 cm).
Square pack (T-84/85/87/88/89/viva): format 3-1/4 x 3-3/8 in (8.3x8.6 cm), image area 2-3/4 x 2-7/8 in (6.9x7.2 cm).
Integral: 600 family 3-1/2x4-1/4 in, image 3-1/8 x 3-1/8 in (7.9x7.9 cm); Spectra/990 4 x 4-1/8 in, image 3-5/8 x 2-7/8 in (9.2x7.3 cm); 500 4-3/8 x 2-1/2 in, image 2-7/8 x 2-1/8 in; i-Zone 6-5/8 x 1-3/8 in, image 1-3/8 x 7/8 in (3.6x2.4 cm); T-339 4-1/2 x 4-1/4 in, image 4x3 in.

### Reciprocity — printed TIME->STOPS tables (only 3 sheets have a real one)
T-72 / T-572 / T-672 (400/27), "Reciprocity Performance" p1:
  <1/15 s -> 400/27, None | 1 s -> 250/25, +2/3 | 4 s -> 200/24, +1 | 16 s -> 125/22, +1 2/3 | 64 s -> 80/20, +2 1/3 | 128 s -> 64/19, +2 2/3
  NOTE internal inconsistency: 72fds.pdf p1 prints 16 s -> +2 1/3 and 64 s -> +1 2/3 (values transposed vs 572fds.pdf/672fds.pdf p1). 572/672 ordering is the monotonic/correct one.
All other sheets give reciprocity ONLY as a log-log "Reciprocity Law Failure" GRAPH (vector), no table.

### Temperature -> processing-time -> equivalent-speed -> exposure-adjust tables (printed)
T-51HC: 70-95F/30 s/None; 65F/35 s/None; 55F/50 s/None (51fds p2)
T-52/552: 95F 15/20 s -1/3; 75-90F 15/20 None; 70F 20/25 None; 65F 25/30 None; 55F 40/40 +1/3 (52fds p2)
T-54 fam: 75-95F 30 s -1/3; 70F 45 None; 65F 60 None; 55F 90 +1/3; 50F 90 +1/3 (54fds p2)
T-55: 75F+ 20 s None; 70F 20 None; 65F 35 None; 55F 50 None; 50F 60 None (55fds p2)
T-57: 95F 15 s -1/3; 75-90F 15 None; 70F 20 None; 65F 30 None; 55F 45 +1/3 (57fds p2)
Polacolor 80 fam (T-88/108/669/559/59/809): 90F 60 s -1/2; 75F 60 None; 70F 60 None; 65F 75 +1/2; 55F 90 +1 (559fds p2)
Polacolor Pro 100 (579/679/879/t79/p100): 70-95F 90 s; 65-69F 120 s; 61-64F 150 s; 55-60F 180 s — speed stays 100/21, adjustment None throughout
125i/Studio/690/89: 70-105F 90 s; 65-69F 120; 61-64F 150; 55-60F 180 — 125/22, None
T-689: 75F+ 90 s; 70F 90; 65F 120; 55F 120
Polacolor ID UV (eiduv): 95F 60 s 200/24 -1; 85F 60 160/23 -1/2; 75F 60 125/22 None; 70F 60 100/21 None; 65F 75 64/19 +1/2; 55F 90 50/18 +1 (*yellow filtration may be required)
Polacolor 100 ID UV (piduv): 95F 200/24 -1..2; 85F 160/23 -2/3..2; 75F 125/22 -1/3..1; 70F 100/21 None; 65F 80/20 +1/3..1; 60F 64/19 +2/3..2; 55F 50/18 +1..2
Viva color: 90F 60 s 160/23 -1 1/3..2; 85F 125/22 -2/3..1; 75F 100/21 None; 70F 80/20 None; 65F 75 s 80/20 +1/2..1; 55F 90 s 50/18 +1..2
T-611: 85F+ 30 s; 75F 45; 70F 45; 65F 60; 60F 75; 50F 90

### Filter factors (printed table, identical across all B&W peel-apart sheets)
Wratten no. 6 / 8 / 15 / 25 / 47 / 58
  @3200K tungsten — aperture adj: 1/3, 1/2, 2/3, 1 1/2, 3 1/2, 3 1/2 ; filter factor: 1.3, 1.4, 1.6, 2.8, 11.2, 11.2
  @5500K daylight — aperture adj: 2/3, 1, 1 1/3, 2 1/2, 2 2/3, 3 1/3 ; filter factor: 1.6, 2, 2.5, 5.6, 6.3, 10
  (52fds.pdf p3, 54fds.pdf p3, 55fds.pdf p3, 56/57/572/606/665/672/72/803/804/84/85/87/viva3 p3)

### Speed variation vs colour temperature (printed, B&W peel-apart)
3200K -1/3 stop; 4800K —; 5500K —; 6500K —; 7500K +1/3; 10,000K +1/3 (52fds.pdf p3)
T-55: 3200K -1/3; 10,000K -1/3, others — (55fds.pdf p3)

### Spectral sensitivity statements
All B&W peel-apart families: "Panchromatic" (explicit). No ortho/blue-only film in this corpus.
Definition printed: "Spectral Sensitivity: Shows the equivalent energy needed at each wavelength in
order to activate the emulsion so that it produces a neutral density of .75."
Wavelength AXIS range given as 350-750 nm on the spectral-sensitivity graphs; NO numeric peak
wavelengths, NO tabulated sensitivity-vs-wavelength values are printed anywhere.
Colour films: "Action Spectra: Shows the film's relative sensitivity..." (graph only, no figures).
MTF: graphs only, spatial frequency axis 0.1-4 cycles/mm (51fds p3) or 1-4 cycles/mm; no numbers.
T-579/679/879 p4: colour-temperature filter recommendations + effective ISO —
 2800K: 1/8 s 80A+5M ISO 32; 1 s 80A+20M 25; 4 s 80A+30M 20; 15 s 80A+40M 16
 3200K: 1/8 s 80A ISO 32; 1 s 80A+10M 25; 4 s 80A+20M 20; 15 s 80A+30M 16
 Intermittency (multi-pop): 2 flashes +1 stop cc5R; 4 +2 cc10M+5Y; 8 +3 cc15M+5Y; 16 +4 cc25M+10Y

## 2. (a) CONFIRMATIONS vs our 1979 trade-book entries
- T-55 Dmin (negative) 0.18 — EXACT match. 55fds.pdf p2.
- POLAROID_664 ISO 100 — CONFIRMED. T-664 is named in the Polapan Pro 100 family header, ISO 100/21. 54fds.pdf p2 / 84fds.pdf p2.
- POLAROID_667 ISO 3000 — CONFIRMED as ISO 3000/DIN 36. T-667 named in the T-87 family header. 87fds.pdf p2 (header), 87fds.pdf p1 (ISO 3000/DIN 36).
- Type 51 Dmax 1.75: no Polaroid Dmax printed, but 51fds.pdf p2 states Print contrast "High" and defines high contrast as slope >1.70, which is consistent with our slope 3.35 being in the high-contrast class. Not a confirmation of the number.

## 2. (b) DISAGREEMENTS — manufacturer wins, our entries need correcting
1. **Type 52 resolution.** Ours 35-40 lp/mm. Polaroid: **12 - 15 line pairs/mm**. 52fds.pdf p2 ("Resolution (1000:1) 12 - 15 line pairs/mm"). Off by ~3x — our value is almost certainly a mis-transcription.
2. **Type 51 resolution.** Ours 28-32 lp/mm. Polaroid: **Print 20 - 30**, Neg 100 - 120 lp/mm. 51fds.pdf p2. Our range sits at the top of / above the printed print range and omits the negative entirely.
3. **Type 55 negative Dmax.** Ours 1.65. Polaroid: **1.55 (N)**; 1.65 is the POSITIVE Dmax. 55fds.pdf p2 (word coords: 1.65(P) x344 / 1.55(N) x345, same column under "D-Max ="). Our entry appears to have taken the positive figure and labelled it negative.
4. **Type 55 negative Slope.** Ours 0.70. Polaroid: **1.65 (N)** (positive is 1.35). 55fds.pdf p2. Note: 0.70 is exactly the negative slope of **T-665/T-85** (665fds.pdf p2, ".70(N)"), so our 55 row looks contaminated by the 665 P/N pack data.
5. **Type 55 negative resolution.** Ours 150-160 lp/mm. Polaroid: **160 - 180 line pairs/mm (negative)**, print 20-25. 55fds.pdf p2.
6. **POLAROID_667 "ASA 2500".** No Polaroid sheet prints 2500 anywhere in the corpus (grep count 0). Printed value is ISO 3000/DIN 36 only. 87fds.pdf p1/p2. Drop the ASA 2500 alternate.
7. **Type 51 Dmin 0.00 / Slope 3.35 / Dmax 1.75 and Type 52 Dmin 0.02 / Slope 1.35 / Dmax 1.75 — UNVERIFIABLE.** Neither 51fds.pdf nor 52fds.pdf prints any D-Max / D-Min / Slope figure on any page (only the generic definitions). These four/three numbers have no manufacturer backing in this corpus. Additionally our Type 52 slope 1.35 equals T-55's positive slope 1.35 (55fds.pdf p2) — another possible cross-contamination.
8. **Types 42, 47, 46L, 146L, 410 — NO Polaroid sheet exists in this corpus** (grep for "Type 42", "Type 47", "46L", "146L", "Type 410" = 0 hits). Cannot be cross-checked here. Suggestive coincidences worth flagging for the next source: our Type 47 row (Dmax 1.70, res 20-22) matches T-72/572/672 exactly on both (72fds.pdf p2: Dmax 1.70, 20-22 lp/mm) while disagreeing on Dmin (.06 vs .10) and slope (1.50 vs 1.65).
9. **POLAROID_SX70 ISO 150 — NOT VERIFIABLE.** No SX-70 sheet in the corpus (grep "SX-70"/"SX 70" = 0). The nearest Polaroid integral sheets (600/Spectra/500/990/i-Zone) are all ISO 640/DIN 29, so ISO 150 cannot be checked or refuted from these files.

## 2. (c) TYPES DOCUMENTED HERE THAT WE DO NOT CARRY
T-51HC (as a 4-way daylight/tungsten pos/neg speed set), T-53, T-553, T-803, T-54, T-554, T-804,
T-84, T-56, T-57, T-606, T-611, T-72, T-572, T-672, T-665, T-85, T-87, Viva 3000, T-59, T-559,
T-559S, T-669, T-669S, T-809, T-88, T-108, T-64 Tungsten (sheet + pack), T-79, Polacolor Pro 100
4x5, T-579, T-679, T-879, T-689 ProVivid, T-89, 690, Studio 125 / 125i, Polacolor ID UV,
Polacolor 100 ID UV, Viva colour, T-600 / 600 Plus, T-600 NotePad, T-600 Write On, Spectra,
Spectra Grid, T-990, 500 (Captiva), i-Zone Pocket Film, i-Zone Pocket Sticker, T-339 AutoFilm, T-779.
Highest-value additions for a physical sim (full Dmin/slope/Dmax + resolution + panchromatic):
T-53/553/803 (.11/1.64/1.75, 12-15), T-72/572/672 (.10/1.65/1.70, 20-22), T-665/85 (P and N sets,
160-180 lp/mm negative), T-87/667/Viva3000 (.10/1.55/1.75, 14-20), Polacolor Pro 100 (.16/1.30/1.8).

## 2. (d) FILES WITH NOTHING USEFUL
- 4x5filmguide.pdf (18 pp) — prose usage guide. Only parenthetical ISO restatements on p2 (Type 52 400/27, 53 800/30, 54 100/21, 55 50/18, 56 sepia 400/27, 57 3000/36, 51 640/400/80/50, 72 400/27, coaterless 800 and 100, 64T 64/19, 79 100/21). No D-values, no resolution.
- packfilms_guide.pdf (14 pp) — prose only; zero ISO/D/resolution values extracted.
- 611fds.pdf, 689fds.pdf, eiduvfds.pdf, piduvfds.pdf, vivafds.pdf — 1-page sheets: speed/format/dev/processing table only, no D-values, no resolution, no spectral statement.
- 125ifds.pdf, studofds.pdf, 690fds.pdf, 89fds.pdf, 579fds.pdf, 679fds.pdf, 879fds.pdf — no resolution figure printed (579/679/879 do give D-values).
- Exact duplicate technical content (same emulsion family reprinted): 553=53, 554=804=84=54, 672=572(=72 modulo the transposed reciprocity row), 85=665, viva3=87, 59=669=809=88=559, 679=879=579, notep=write=600pl, specg=splat=990, pockt=pktst=500, 64tp=64t, t79=p100.

## 2. (e) VECTOR-CURVE PAGES (get_drawings, paths with >=25 items)
Every sheet's p1 has exactly 1 path with 123 items — that is the page border/logo, NOT a curve.
Real curve pages (>=2 large paths) — H&D characteristic curves, Reciprocity Law Failure,
spectral sensitivity, MTF, all vector, all digitisable:
51 p2(18 paths,max76) p3(3,121) | 52 p2(7,71) p3(1,115) | 53/553/803 p2(5,68) p3(1,137)
54/554/804/84 p2(7,73) p3(1,116) | 55 p2(13,72) p3(1,91) | 56/606 p2(7,72) p3(3,121)
57 p2(8,68) | 572/672/72 p2(7,76) p3(3,71) | 665/85 p2(8,70) p3(1,120)
87/viva3 p2(8,72) p3(2,125) | 559/59/669/809/88 p2(14,73) p3(5,67)
579/679/879 p2(12,73) p3(9,87) | 64t/64tp p2(11,74) p3(7,81) | 339/779 p2(11,74)
500/600pl/990/notep/write/specg/splat/pockt/pktst p2(11,71) p3(4,66) p4(3,70)
studo p2(12,80) p3(9,88) | 690/89 p3(10,87) | 125i p3(7,87) | p100/t79 p2(1,27) only
No large paths: 611, 689, eiduv, piduv, viva (p1 only), 4x5filmguide, packfilms_guide.

## 3. PARAMETER CLASSES ABSENT FROM ALL 53 SHEETS
- Numeric reciprocity time->stops table: absent except T-72/572/672. Everywhere else graph-only.
- Numeric spectral sensitivity data (peak wavelength, sensitivity vs nm): absent. Graphs only, 350-750 nm axis.
- Numeric MTF data: absent. Graphs only.
- Granularity / RMS grain: absent (only prose "medium-grain", "fine-grain").
- D-Max / D-Min / Slope for T-51, T-52, T-54 family (incl. T-664), T-56, T-57, T-606, T-611,
  T-64, the Polacolor 80 family, all integral films, T-689, ID-UV, Viva colour: not printed.
- Any characteristic-curve data as numbers (only vector plots).
- Dye-density / spectral dye curves for colour films: absent.
- Base+fog, exposure latitude, gamma: absent.
