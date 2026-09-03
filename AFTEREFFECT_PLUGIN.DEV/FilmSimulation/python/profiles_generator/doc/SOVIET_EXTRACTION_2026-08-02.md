# Soviet reference-book extraction — 2026-08-02

> **UPDATE 2026-08-11:** owner supplied the GOST standards themselves
> (`PDF/PROFILES/SOVIET STANDARDS/`). Norms extracted and reconciled with
> the reference-book values below — two MTF floors raised, one gamma
> conflict recorded. See `CHANGES_2026-08-11_gost_extraction.md`.

Source pass over `PDF/PROFILES/SOVIET/`. Page numbers are PRINTED BOOK pages.
Every number below was transcribed from a rendered scan page read visually;
nothing is interpolated.

## Sources (cite exactly in profiles)

1. Гурлев Д. С. «Справочник по фотографии (светотехника и материалы)»,
   Киев: Техніка, 1986, 368 с.
   [Gurlev D. S., "Handbook of Photography (Light Engineering and
   Materials)", Kyiv: Tekhnika, 1986] — file
   `SOVIET/Gurlev_sprav_svetotexnika_materialy.pdf` (367 PDF pages;
   PDF page = book page − 1).
2. Иофис Е. А. «Кинофотопроцессы и материалы», 2-е изд., М.: Искусство,
   1980, 240 с. [Iofis E. A., "Cine and Photo Processes and Materials",
   2nd ed., Moscow: Iskusstvo, 1980] — file
   `SOVIET/Кинофотопроцессы и материалы.pdf` (2-up scan, PDF page ≈ book/2).
3. Гордийчук И. Б., Пелль В. Г. «Справочник кинооператора», М.: Искусство,
   1979. [Gordiychuk I. B., Pell V. G., "Cinematographer's Handbook",
   Moscow: Iskusstvo, 1979] — file `SOVIET/Справочник Кинооператора.pdf`.
4. Чибисов К. В., Шеберстов В. И., Слуцкин А. А. «Фотография в прошлом,
   настоящем и будущем», М.: Наука, 1988. [Chibisov K. V., Sheberstov V. I.,
   Slutskin A. A., "Photography in the Past, Present and Future", Moscow:
   Nauka, 1988] — GOST speed criteria: S = 0.8/H_cr (general films),
   S = 0.5/H_cr (cine negative), S = 10/H_cr (aerial); GOST/DIN/ASA
   conversion table (book p. 52–53).
5. ORWO CHROM-FILM UT 18 consumer leaflet W 746 (Ausgabe 17c), VEB
   Filmfabrik Wolfen, DDR — file `ORWO/ORWO CHROM-FILM UT 18 - web.jpg`:
   official name "ORWO CHROM-FILM UT 18", REVERSIBLE, 18 DIN / 50 ASA /
   45 GOST, daylight balanced (ORWO filter No. 13 [B12], factor 6, for
   tungsten), storage < 18 °C at 50–60 % RH.

## Gurlev 1986, §246–247, book p. 296 — «Фото» B&W negative photo films
Developer СТ-2 (ST-2). Columns: Фото-32(Т) / Фото-65(Т) / Фото-130(Т) / Фото-250(Т)

| Parameter | Foto-32 | Foto-65 | Foto-130 | Foto-250 |
|---|---|---|---|---|
| t develop, min | 6–10 | 6–10 | 8–14 | 8–14 |
| γ_rec (γ_max 1…1.4) | 0.8 | 0.8 | 0.8 | 0.8 |
| S, GOST nominal | 32 | 65 | 130 | 250 |
| S, GOST range | 28–55 | 55–110 | 110–220 | 220–500 |
| S_eff behind Ж-2 yellow, % | 35 | 35 | 35 | 50 |
| S_eff behind О-2 orange, % | 12 | 12 | 12 | 15 |
| S_eff behind К-5,6 red, % | 3 | 3 | 3 | 3 |
| D0 (fog) | 0.04 | 0.05 | 0.06 | 0.08 |
| D0 max | 0.1 | 0.16 | 0.25 | 0.3 |
| L latitude, logH | 1.5 | 1.5 | 1.5 | 1.5 |
| R, lin/mm | 135 | 110 | 100 | 82 |
| Δλ_S sensitization limit, nm | 645 | 665 | 580 | 630 |

§246: negative photo films = «Фото» type; positive = МЗ-3Л; reversal = ОЧ
type. Emulsion melting point ≥ 32 °C (Т suffix = heat-resistant, ≥ 70 °C).
35 mm rolls 1.65 m / 17 m, base 110–150 µm.

## Gurlev 1986, §248, book p. 296–297 — МЗ-3Л positive
Fine grain, high resolving power, deep black image tone, non-sensitized
(blue-sensitive). Sheet + 35 mm roll; base 135–150 µm. Sensitometric data
(СТ-2): t 4 min; γ_rec 2.5; L 2.5; D0 (without base) 0.04;
S GOST (D=0.2+D0) 1.4; (D=0.85+D0) 8; S^Ж_0.85 0.015; R 125 lin/mm;
L_R 0.9; D_R 1.6; λ_max 480 nm; γ_max 3.1; D_max 3.7.
H&D curve family (СТ-4, 1–24 min) fig. 177.
(Cine positive МЗ-3 corroboration, Iofis 1980 §23 book p. 107: S 2.8–5.5
GOST, γ_rec 2.6, γ@4min 2.8–3.2, R ≥ 108 mm⁻¹, fog ≤ 0.04, D_max ≥ 3.0 on
straight section, latitude 0.9–1.2, melting ≥ 70 °C. Also
`SOVIET/TASMA POSITIVE МЗ-3Л.jpg` (book excerpt, item 273, ТУ 6-17-647—80):
S GOST 2.8–5.5, Δlg H 0.9, R 108 lin/mm, D0 0.04, D_max 3, t_пл 50 °C;
γ = 2.6/2.7–3.1 at 2/3/4 min in СТ-4.)

## Gurlev 1986, §249, book p. 298 — ОЧ-45 / ОЧ-180 B&W reversal
ТУ 6-17-646—74. 35 mm perf 1.65 m; 16 mm unperf 0.45/0.95 m; Rolfilm 60 mm;
base 115–125 µm. First development 12 min:

| Parameter | ОЧ-45 | ОЧ-180 |
|---|---|---|
| γ_rec | 1.1–1.6 | 1.2–1.6 |
| L | 1.05 | 1.05 |
| D_max | 1.9 | 1.8 |
| D_min | 0.08 | 0.05 |
| S_0.9, GOST | 45 | 180 |
| S^Ж_0.9, GOST | 16 | 65 |

Iofis 1980 §29 (book p. 146–147): B&W reversal cine sensitized to 680 nm,
triacetate base tinted bluish-blue; medium-speed class = S 45 GOST,
γ_rec 1.2, R ≥ 100 mm⁻¹, D_min ≤ 0.08, latitude 1.05. Table 21 lists
«Обращаемая ОЧ-45» (GOST 45 at 6000 K / 32 at 3200 K) — the same emulsion
is sold for cine; Tasma (Kazan) was a producer of the ОЧ line.

## Gurlev 1986, §306, book p. 354–355 — colour negative photo films
Unmasked: ДС-4 (ТУ 6-17-622—74). Masked: ЦНД-32, ЦНЛ-32, ЦНЛ-65
(ТУ 6-17-441—78). Development 5–8 min. Daylight 5500 K (Д) and tungsten
3200 K (Л) types. H&D per-layer curve families fig. 197.

| Parameter | ДС-4 | ЦНД-32 | ЦНЛ-32 | ЦНЛ-65 |
|---|---|---|---|---|
| S, GOST | 45 | 32 | 32 | 65 |
| γ_rec (DS-4: overall) | 0.8 | — | — | — |
| γ mid+bottom layers | — | 0.7±0.1 | 0.7±0.1 | 0.7±0.1 |
| (top layer γ higher by 0.1–0.2) | | | | |
| L overall | 1.2 | 0.9 | 0.9 | 1.5 |
| Б_к contrast balance | 0.12 | — | — | 0.1 |
| Б_ч (D_max) | 2 | 2.5 | 2.5 | 2.4 |
| D0 (per spectral zone) | 0.25 | — | — | — |
| D0+D_mask behind blue filter | — | 0.75–1.1 | 0.75–1.1 | 0.75–1.1 |
| … behind green filter | — | 0.25–0.5 | 0.25–0.5 | 0.4–0.6 |
| … behind red filter | — | 0.3 | 0.3 | 0.3 |
| R, lin/mm (white light) | 63 | 58 | 58 | 63 |

Base: sheet 140–150 µm, Rolfilm 90–110 µm, 16/35 mm 110–150 µm.

## Gurlev 1986, §307, book p. 355–356 — colour reversal photo films
ЦО-22 (ТУ 6-17-617—74), ЦО-32Д (ТУ 6-17-912—77) daylight 5500 K;
ЦО-90Л, ЦО-180Л tungsten 3200 K.

| | ЦО-22 | ЦО-32Д | ЦО-90Л | ЦО-180Л |
|---|---|---|---|---|
| S GOST | 22 | 32 | 90 | 180 |
| γ_rec | 1.8–2.2 | 1.8–2.2 | 1.4–1.7 | 1.4–1.7 |
| Б_ч | 1.8 | 1.3…1.8 | 1.6 | 1.6 |
| Б_к | 0.3 | 0.3 | 0.3 | 0.3 |
| L | 1.2 | 1.2 | 1.2 | 1.2 |
| R lin/mm | 70 | 53 | 53 | 50 |
| D0 | 0.25 | 0.25 | 0.25 | 0.25 |

## Gurlev 1986, §308, book p. 356 — colour cine negatives
ДС-5М (ТУ 6-17-691—75) masked 5500 K; ЛН-7, ЛН-8 (ТУ 6-17-1109—80) masked
3200 K (daylight via Kodak Wratten 85 / ORWO K-14 / ОС-6 2 mm):
S GOST 32 / 65 / 100; ḡ 0.55–0.65; γ mid+bottom 0.65±0.05, top ≈0.8±0.05;
(ЛН-8: γ_общ 0.65±0.05, ḡ 1.10 ratio limit).

## Гордийчук/Пелль 1979, разд. X — cine film tables
Table X-2 (book p. 377), colour cine negatives ДС-5М / ЛН-7 / ЛН-8:
balance 5500/3200/3200 K; S GOST ≥ 32/65/100; Б_ч ≤ 2.0/2.5/2.0; S with
conversion filter in daylight 40 (ЛН-7), 60 (ЛН-8); γ all layers 0.65±0.05
(top +0.15±0.05); Б_к ≤ 0.10 / 0.10 / 0.10; L ≥ 1.05/1.5/1.5;
R ≥ 58/63 lin/mm; ЛН-8: ḡ 0.55–0.65, mean gradient ratio ≤ 1.10,
RMS granularity ≤ 2.2 (green filter) / 2.7 (red filter); MTF@30 lin/mm
≥ 0.22 (green-sens layer) / 0.15 (red-sens layer).

Table X-7 (book p. 382), new B&W negative cine set НК-1/НК-2/НК-3/НК-4:
S GOST ≥ 22/65/180/350; γ_rec 0.65; mean gradient 0.60–0.62 / 0.57–0.65;
R ≥ 120/110/90/75 lin/mm; RMS granularity ≤ 2.9/3.3/4.8/8.0;
MTF@30 lin/mm ≥ 0.73/0.70/0.60/0.55; D0 ≤ 0.06/0.08/0.10/0.18;
sensitization limit 660–670 nm.
(Iofis 1980 §16, book p. 70, prints the same class limits.)

Дубль-позитив А-2: S 1.5–3.0 GOST, γ 1.4, L ≥ 1.05, R ≥ 215 lin/mm,
λ 560–580 nm. Дубль-негатив А-2: S 0.6–1.0, γ 0.65, L ≥ 1.8,
R ≥ 180 lin/mm, λ ≤ 600 nm. МЗ-3 positive (book p. 383): mass release
print, good sharpness, fine grain, neutral tone.

## Iofis 1980, Table 5 (book p. 73) — B&W neg cine speed equivalences
Отечественные НК-1/2/3/4 ↔ ORWO NP-55, NP-7 ↔ Kodak Plus-X 5231/7231,
Double-X 5222/7222, Double-4X 5224/7224 ↔ Gevapan 30/36 ↔ Fujifilm Super
Panchromatic ↔ Ilford Super Hypan 2660, with GOST/DIN/ASA columns at
6000 K and 3200 K (e.g. НК-2: GOST 65 → ASA 80 daylight).

## Chibisov 1988, Appendix (second pass, same day) — Таблица I, book p. 157–158
«Таблица I. Фотографические характеристики черно-белых фото- и кинопленок»
(rotated pages; PDF pages 164–165).

Фотопленки общего назначения (p. 157): Фото-32/65/130/250 — S 32/65/130/250;
γ_rec 0.8, γ_max 1.4 (all); D0 0.05–0.1 / 0.1–0.16 / 0.16–0.25 / 0.2–0.3;
L 1.5 (all); **R 116/92/75/70 mm⁻¹** — CONFLICT with Gurlev's 135/110/100/82
(test conditions unstated in both; Gurlev kept, conflict recorded in
`_RESOLVING_POWER` comment and citations). Also ФТ-10…ФТФ-2 phototechnical
films (repro; not simulator targets).

Кинопленки (p. 158): КН-1/2/3 (S 11/32/90, γ 0.65, D0 0.13/0.16/0.20,
R 135/100/78, 650 nm); НК-1/2/3/4 (S 32/90/250/350–500 — differs from
Gordiychuk/Pell's 22/65/180/350; γ 0.65, D0 0.06/0.10/0.12/0.20,
R 120/110/90/75, 670 nm); Звуковая ЗТ-8. Микрат microfilm rows.

Кинопленка обратимая (p. 158): **ОЧ-45: S 45, γ_rec 1.6, γ_max 2.2, D0 0.06,
L 1.05, R 110 mm⁻¹, 660 nm** — adopted: TASMA_OCH_45 gamma raised
1.35 → 1.50 (Chibisov-weighted end of the Gurlev 1.1–1.6 window), R 110
(product row supersedes the Iofis class minimum 100), f50 scaled to 34 c/mm.
Dmin: Chibisov 0.06 vs Gurlev 0.08 — 0.08 kept (newer source). Also ОЧ-180,
ОЧ-Т-45, ОЧ-Т-180, ОЧ-Т-45М rows (not in database).

## Chibisov 1988, Appendix — other tables (survey, PDF pages 166–181)
- Табл. II (p. 159–160): B&W photo PAPERS (Унибром, Фотобром, Новобром,
  Бромпортрет, Контабром, Йодоконт) — S, exposure interval, Dmax by surface.
  Paper, not film; recorded only.
- Табл. III (p. 161): special developer formulas (НИКФИ, ГХ-Б, Д-41/42,
  ORWO-76/80, Д-11) — processing context.
- Табл. IV (p. 162): thermodeveloped microfilm (Dry-Silver, Kodak 784...).
- Табл. V (p. 162): **Polaroid B&W one-step packs** — Type 42 (ASA 200,
  Dmax 1.6, fog 0.02, R 22–28, 650 nm, 15 s), 47 (3000), 55 P/N (50,
  Dmax 1.7, R 14–17, 670 nm, 20 s), 46-L, 413 (IR 920 nm), 410, TL-X.
  Corroboration source for any future Polaroid sheet-film profiles.
- Табл. VI–VII (p. 163–164): holography materials (Микрат ВР, ВРЛ, Kodak
  649F, Agfa 8E/10E) — not simulator targets.
- Табл. VIII (p. 165–166): **foreign colour cine films** — negatives
  Kodak 5247 (S 125 GOST, ḡ 0.50, RMS σD×1000 visual 5, MTF@30 0.65
  green/0.32 red), Kodak 7291 (100), **Kodak 5294 (400, RMS 6, 0.65/0.30)**,
  Agfa XT-125/XT-320, Gevaert 683/693, ORWO NC-3 (64, ḡ 0.59, RMS 11
  green/15 red/10 visual — 2× the Kodak figures; supports the coarse
  ORWOCOLOR grain modelling), Fuji 8511/8512/8514; positives PC-7/PC-12
  (ORWO), Kodak 5384/5380, Gevaert 982, Fuji 8816, Inducolor G-9H.
  Citations added to EASTMAN_5247_1974 and EASTMAN_5294_1983 (grain figures
  recorded, not adopted — cross-era metric equivalence unverified; 5294's
  printed green MTF@30 0.65 exactly matches the profile's f50_g 38 c/mm).
- Табл. IX (p. 167–168): **domestic colour films** — confirms the Gurlev
  values adopted for ДС-4 (γ 0.75–0.85, Бк 0.12, fog 0.25/0.25/0.25, L 1.2,
  R 63), ЦНД-32/ЦНЛ-32 (γ 0.6–0.8, mask 0.75–1.1 / 0.25–0.45 / 0.30, L 0.9,
  R 58), ЦНЛ-65 (mask green 0.40–0.60, L 1.5, R 63). Additional rows not in
  database: ЦНЛ-90 (S 90, γ 0.65±0.05, L 1.3, R 63), ЦОД-16/32; cine ДС-5М
  (S 22 here vs 32 in Gordiychuk/Pell), ЛН-7/ЛН-8; контратипные КП-М/КП-6;
  positives ЦП-8Р (γ 3.0–3.6), ЦП-11 (γ 2.7–3.3). Reversal ЦО-22/32Д/65/90Л/
  180Л/Т-90Л with per-layer Dmax 2.1–2.3, Dmin 0.20–0.25, L 1.1–1.2,
  R 53–82 — ЦО-65 and ЦО-Т-90Л are additional stocks available if wanted.
- Табл. X–XIV (p. 168–172): thermoplastic, electrophotographic, micrography,
  printing plates, non-silver copy materials — outside simulator scope.

## Dufaycolor NSMM measured reseau (2026-08-02, owner instruction)
`DUFAYCOLOR/measuredODs_MSI_NSMM_11948/11951/11960_Dufaycolor_small.jpg`
(National Science and Media Museum, Bradford): absorbance of the dyed reseau
elements, 400–720 nm, three surviving prints, mutually consistent. Mean
curves transcribed, T = 10^(−A), band-averaged (B 420–490, G 500–580,
R 600–700 nm), uniformly rescaled ×4.05 (absolute measured densities carry
base + dye aging; between-band ratios preserved), then row-normalised to a
common row sum 0.80 to keep the renderer's neutral-grey reconstruction
invariant. Result adopted as DUFAYCOLOR_1937 `filter_matrix`; tier raised
3 → 2. Measured signature vs the old estimate: blue element leaks red
strongly, green leaks symmetrically, red is the cleanest element.

## Alias findings
- СВЕМА «Фото-65» is the still-film designation of the FN-64 class emulsion
  (S 65 GOST ≈ EI 64): Gurlev Foto-65 column (γ 0.8, D0 0.05, R 110 lin/mm,
  Δλ_S 665 nm, L 1.5) is consistent with the measured SVEMA_FN_64 profile
  (γ 0.83 measured over 509 frames). Alias added per owner instruction.
- ORWO UT18 official name is «ORWO CHROM-FILM UT 18» (leaflet W 746);
  profile renamed ORWO_CHROM_UT18.
- ОЧ-45 sold both as photo film (Gurlev ТУ 6-17-646—74) and cine reversal
  (Iofis table 21); Tasma branding per owner instruction.
