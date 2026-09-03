# FUJIFILM AF3-xxx Product Information Bulletins — extraction for film-simulation engine
Scope: 21 PDFs listed in /tmp/list_FUJI.txt, all under PDF/PROFILES/FUJI/. All TRUE TEXT except NPZ.pdf.
Only printed values are recorded. Nothing estimated. "n/p" = not printed in this sheet.

## 0. HEADLINE GENERATION VERDICT
**Of our 10 Fuji stocks, exactly ONE is addressed by this file set, and only in a non-135 format sheet:**
FUJI_NEOPAN_ACROS_100 <- `Acros-120_AF3-083E.pdf` (sheet is titled *NEOPAN 100 ACROS (120)*, i.e. 120 roll film only).
All other 20 PDFs document films we do NOT hold. In particular:
- Velvia 100 / Velvia 100F are **not** FUJI_VELVIA_50 (both sheets explicitly contrast themselves with "the current ISO 50 Velvia").
- Provia 100F [RDP III] is **not** FUJI_PROVIA_400X.
- Sensia 200 [RM] / Sensia 400 [RH] are **not** FUJI_SENSIA_100.
- Nothing covers FUJI_NEOPAN_1600, FUJI_ETERNA_VIVID_500T_8547, FUJICOLOR_A250, FUJI_F125_8530, FUJI_F125_8630, FUJICOLOR_SUPER_F500_8572 (no motion-picture stock in this set at all).

---
## 1. PER-FILM TABLES

### 1.1 NEOPAN 100 ACROS (120) — `Acros-120_AF3-083E.pdf` (AF3-083E, 4 pp) — **OUR STOCK (120 format sheet)**
| Parameter | Value | Source |
|---|---|---|
| ISO speed | ISO 100/21° | p1 §3 SPEED |
| Colour sensitivity | Orthopanchromatic (B&W; no balance colour temperature applies) | p1 §4 |
| Balance colour temp (K) | n/p (achromatic film) | — |
| Second exposure index | n/p. Filter table gives **FILTER FACTORS ONLY, NOT film speeds**: SC-39/W1A 1.0 daylight, 1.0 tungsten; SC-48/W8 2.0 / 1.5; SC-56/W21 4.0 / 3.0; SC-60/W25 8.0 / 6.0. FLAG: these are filter factors (filter property), not illuminant-conditioned film indices. | p2 "Filter Recommendations" |
| Developer-conditioned EI | Printed per developer as "EI": 100 (Microfine, Fujidol E, D-76, Minidol, Finedol, T-MAX, X-tol, HC-110 Dil.B, ID-11, Neoprodol 1:1…), 200 (Super Prodol/SPD, SPD 1:1, T-MAX RS, D-76 1:1 variants), 80 (Super Finedol, Microdol-X, Perceptol). FLAG: developer-conditioned, not illuminant-conditioned. | p2 §7(1) |
| RMS granularity | **7** (diffuse RMS) | p3 §10 |
| RMS conditions | Micro-densitometer aperture **48 µm diameter**; magnification 12X; sample density 1.0 above Dmin; processing Microfine | p3 §10 |
| Resolving power | **60 lines/mm @ chart contrast 1.6:1**; **200 lines/mm @ 1000:1** (Microfine, 20 °C/68 °F, small tank) | p3 §11 |
| MTF | **absent** (no MTF section in this B&W sheet) | — |
| Spectral sensitivity curve | Present, §12, single-layer (panchromatic, one curve). **RASTER** — page 3 carries 1 embedded image 890x816 px and no path >=25 items (max 12). Wavelength axis values not in text layer. | p3 §12 |
| Characteristic curves | Present §13, four developers (Fujidol E, Microfine, Super Fujidol-L, D-76). **RASTER** — 5 embedded images on p4. Also §14 **TIME-GAMMA curves** (gamma vs development time) — raster. No numeric gamma, Dmin or Dmax printed. | p4 §13,14 |
| Gamma / Dmin / Dmax numbers | n/p (only time-gamma graph) | — |
| Reciprocity | **No exposure compensation for exposures shorter than 120 s.** Table: exposure time 120–1000 s -> **+1/2 stop**. | p1 §"Reciprocity Characteristics" |
| CC filter for reciprocity | **Not applicable / none named** — B&W film, sheet asks for exposure compensation only. Cannot be read as a colour-achromatic claim. | p1 |
| Process | B&W, no CN-16/E-6/C-41. Small-tank: Microfine 20 °C/68 °F, 7¼–9½ min over 1–10 rolls (capacity 4 rolls, 600 ml); Fujidol E (1 l) 9–12 min, capacity 10; D-76 (1 l) capacity 10. Deep tank 18/20/22/24 °C: Minidol EI100 13½/11/9/7½ min, Finedol EI100 same, Super Finedol EI80 11/8½/7/5¾ min. Hanger-transport 18–26 °C full matrix p2. Stop bath 1.5 % acetic acid, 15–25 °C, 20–30 s. Fix: Fujifix 10 min / Fujifix Super-L 5–10 min / Super Fujifix 3–5 min at 15–25 °C. Wash 20–30 min. | p2 §7 |
| Densitometry status | **n/p** (no Status A/M statement — B&W sheet) | — |
| Base material / thickness | **Cellulose Triacetate, 0.104 mm** (=104 µm) | p1 §2 |
| Safelight | Fuji SLG-4 dark green, 20 W, >=1 m | p2 §6 |

### 1.2 FUJICHROME 64T TYPE II Professional [RTP II] — `RTPIIAF3-024E_1.pdf` (AF3-024E, 6 pp) — not ours
| Parameter | Value | Source |
|---|---|---|
| ISO / balance | ISO 64/19°, **Tungsten-type, 3100 K** | p1 |
| Second index | Daylight (5500 K) **through Wratten No.85B (or Fuji LBA-12)** — FLAG: **conversion-filter index (filter factor), NOT an unfiltered film property**. Daylight+flash need No.85B+81A or LBA-16; low colour temp No.85B+81A / LBA-12+LBA-2. Fluorescent CW +25M+10R (+1½ stops), WW 30R+5M (+1 stop). Footnote *4: "Exposure correction values include filter exposure factors." | p1–p2 |
| RMS granularity | **10**; aperture **48 µm diameter**, sample density 1.0 above Dmin | p5 §16 |
| Resolving power | **55 lines/mm @ 1.6:1; 135 lines/mm @ 1000:1** | p5 §17 |
| Curves | §18 characteristic, §19 spectral sensitivity, §20 MTF, §21 spectral dye density. Wavelength axis labelled "Wavelength (nm)". **RASTER — 9 embedded images on p6** (969x868 etc.), max vector path only 20 items. | p6 |
| Reciprocity | Not required 1/15–64 s. Table: 1/4000–1/30 s **Not Recommended**; 1/15–64 s CC **None**, no correction; **128 s CC None, +1/3 stop; 256 s CC None, +1/2 stop**. **=> explicit "None" colour-compensating filter at all recommended long exposures: ACHROMATIC reciprocity failure.** | p2 |
| Process / densitometry | E-6 / Fujifilm CR-56 (CR-56 stated equivalent to E-6). Times/temps n/p. Viewing per ISO 3664:2000, CIE D50. | p1,p3 |
| Base | Cellulose Triacetate; **135 rolls 127 µm** | p1 |

### 1.3 FUJICHROME T64 Professional [RTP] — `t64_datasheet.pdf` (AF3-178E, 8 pp) — not ours
ISO 64/19°, **tungsten 3200 K** balance (p1); daylight/flash 5500 K index **ISO 32/16° through Wratten No.85B** — FLAG: conversion-filter index, and footnote states "Exposure correction values include filter exposure factors" (p1–p2). RMS **7**, aperture **48 µm** (p6 §17). Resolving **55 lp/mm @1.6:1, 115 lp/mm @1000:1** (p6). Reciprocity: no compensation 1/125 s–2 min; **4 min: CC filter None, +1/3 stop**; shorter than 1/250 s "Not Recommended" -> **achromatic** (p2). Process E-6 / CR-56, **Densitometry Status A**, density 1.0 above D-min (p7). Base: 135 Cellulose Triacetate 127 µm, 120 CTA 98 µm, sheet **Polyester 175 µm** (p1). Curves p6–p7: characteristic, spectral sensitivity, MTF, spectral dye density, axis 400/500/600/700 nm -> **VECTOR** (p6 ndraw=302, longest path 88 items, 0 images; p7 ndraw=28, 0 images).

### 1.4 FUJICHROME Velvia 100 Professional [RVP 100] — `velvia_100_datasheet.pdf` (AF3-202E, 8 pp) — **NOT Velvia 50**
ISO 100/21° daylight, filter None; tungsten 3200 K **ISO 32/16° via Wratten No.80A (LBB-12)** — conversion-filter index. Household tungsten adds No.82C (LBB-2/LBB-4). RMS **8**, "read at a gross diffuse visual density of 1.0, using a 48-micrometre aperture" (p8 §19). Resolving **80 lp/mm @1.6:1, 160 lp/mm @1000:1** (p8). Reciprocity (p2): none 1/4000 s–1 min; **2 min CC 2.5M +1/3; 4 min 2.5M +1/2; 8 min 2.5M +2/3** — CC filter REQUIRED, not achromatic. Process E-6 / CR-56 (p3). Base CTA 127 µm (135) / 98 µm (120) / Polyester 175 µm (sheet) (p1). Curves p7 (spectral sensitivity, ndraw=702, longest path 88, 0 images) and p8 (char./MTF/dye density, ndraw=567, 0 images) -> **VECTOR**. Explicitly says its RMS 8 + ISO 100 "exceed the levels of the current ISO 50 Velvia" (p7) — confirming it is a different product from our Velvia 50.

### 1.5 FUJICHROME Velvia 100F Professional — `velvia_100f_datasheet.pdf` (AF3-148E, 8 pp) — not ours
ISO 100/21° daylight / None; tungsten 3200 K **ISO 32/16° via No.80A (LBB-12)** = conversion-filter index. RMS **8**, 48 µm aperture at gross diffuse visual density 1.0 (p7 §18). Resolving **80 / 160 lp/mm** (1.6:1 / 1000:1) (p7 §19). Reciprocity: 1/4000–1 min none; **2/4/8 min CC 2.5B with +1/3 / +1/2 / +2/3 stop** — CC required. Footnote: "Exposure correction values when using a filter relative to unfiltered exposure results." Process E-6/CR-56, **Densitometry Fuji FAD-30S (Status A)**, 1.0 above D-min (p8). Base CTA 127/98 µm, Polyester 175 µm (p1). Curves p7 **VECTOR** (ndraw=312, longest 89, 0 images); p8 char./spec.sens./MTF/dye **VECTOR** (ndraw=47, 0 images), axis "Wavelength (nm)".

### 1.6 FUJICHROME ASTIA 100F Professional — `astia_100f_datasheet.pdf` (AF3-149E, 8 pp) — not ours
ISO 100/21° daylight / None; tungsten 3200 K **ISO 32/16° via Wratten No.80A (LBB-12)** = conversion-filter index (footnote: "Indicates the effective speed resulting from designated filter use"). RMS **7**, 48 µm aperture at gross diffuse visual density 1.0 (p8 §18). Resolving **60 lp/mm @1.6:1, 140 lp/mm @1000:1** (p8 §19). Reciprocity: none 1/4000 s–1 min; **2/4/8 min CC 5B, +1/3 / +1/2 / +2/3 stop**. Process E-6/CR-56, **Densitometry Fuji FAD-30S (Status A)** (p8). Base CTA 127 µm / 98 µm, Polyester 175 µm (p1). Curves: p7 spectral sensitivity **VECTOR** (ndraw=306, longest 88, 0 images); p8 char./spec.sens./MTF/dye **VECTOR** (ndraw=49, 0 images), axis "Wavelength (nm)".

### 1.7 FUJICHROME PROVIA 100F Professional [RDP III] — `provia_100f_datasheet.pdf` (AF3-036E, 6 pp) — **NOT Provia 400X**
ISO 100/21° daylight / None; tungsten 3200 K **ISO 32/16° via No.80A (LBB-12), +1 2/3 stop** = conversion-filter index. RMS **8**, aperture **48 µm diameter**, sample density 1.0 above Dmin (p5 §16). Resolving **60 / 140 lp/mm** (p6 §17). Reciprocity: adjustments needed for shutter speeds longer than **128 s**; "reciprocity-failure related **color balance and exposure** compensations are required" -> NOT achromatic (p2,p3). Process E-6 / CR-56 / Fuji Hunt C6R; **Densitometry Fuji FAD-30S (Status A)** (p6). Base Cellulose Triacetate; **135 127 µm, 120 104 µm, 220 104 µm, sheets 205 µm** (p1). Curves p6: **RASTER — 9 embedded images** (1004x886 etc.), longest path 20 items.

### 1.8 FUJICHROME Sensia 200 [RM] — `sensia_200_datasheet.pdf` (AF3-080E, 6 pp) — **NOT Sensia 100**
ISO 200/24° daylight / None; tungsten 3200 K **ISO 64/19° via No.80A (LBB-12)** = conversion-filter index. RMS **13**, aperture **48 µm diameter**, 1.0 above Dmin (p4 §14). Resolving **60 / 140 lp/mm** (p4). Reciprocity: none 1/4000–32 s; **1 min CC 5G +2/3; 2–4 min CC 7.5G +1; 8 min Not recommended**. Footnote: "Exposure correction values include filter exposure factors… added to unfiltered exposure meter readings." Process Kodak E-6 / Fujifilm CR-56 (p3). Base Cellulose Triacetate, **127 µm** (p1). Curves p5: char./spec.sens./MTF/dye — **MIXED**: 475 drawings with paths up to 39 items **and** 1 embedded image (941x714); the dye-density panel appears raster, the rest vector.

### 1.9 FUJICHROME Sensia 400 [RH] — `sensia_400_datasheet.pdf` (AF3-081E, 6 pp) — **NOT Sensia 100**
ISO 400/27° daylight / None; tungsten 3200 K via **No.80A (LBB-12)** (second index value not in text layer). RMS **13**, 48 µm aperture (p4 §14). Resolving **55 / 135 lp/mm** (p4). Reciprocity identical structure to Sensia 200: 1/4000–32 s none; 1 min **5G +2/3**; 2–4 min **7.5G +1**; 8 min not recommended (p3). Process E-6/CR-56. Base CTA **127 µm**. Curves p5 **MIXED** (ndraw=484 with 39-item paths, 1 image 941x714).

### 1.10–1.13 FUJICOLOR PRO 160S / PRO 160C / PRO 400H / PRO 800Z — not ours (distinct films)
Files: `pro_160s_datasheet.pdf` (AF3-174E), `AF3-203U_Pro160S_...pdf`, `pro_160c_datasheet.pdf` (AF3-175E), `AF3-204U_Pro160C_...pdf`, `pro_400h_datasheet.pdf` (AF3-176E), `pro_800z_datasheet.pdf` (AF3-177E). All 8 pp.
| Film | ISO (daylight/flash, filter None) | Tungsten 3200 K index | RMS (48 µm) | Resolving 1.6:1 / 1000:1 |
|---|---|---|---|---|
| PRO 160S | 160/23° | **50/18° equivalent via Wratten No.80A (LBB-12)** = conversion filter | **3*** | 63 / 125 lp/mm |
| PRO 160C | 160/23° | **50/18° equivalent via No.80A (LBB-12)** | **3*** | 63 / 125 lp/mm |
| PRO 400H | 400/27° | **100/21° equivalent via No.80A (LBB-12)** | **4*** | 50 / 125 lp/mm |
| PRO 800Z | 800/30° | **200/24° equivalent via No.80A (LBB-12)** | **5*** | 50 / 115 lp/mm |
RMS footnote (* , p7): "Based on Fujifilm measurements. Due to difference in measurement conditions, comparison with color reversal film is not possible." Conditions: aperture **48 µm diameter**, sample density **+1.0 above minimum density**.
Reciprocity — PRO 160S/160C (p2): "**No color balance compensation is required**"; 1/4000–2 s unnecessary, 4 s +1/3. PRO 400H (p2): "**No exposure color balance compensation is required**"; 1/4000–1 s None, 4 s +1/2 stop, 16 s +1 stop, longer than 16 s not recommended. **=> ACHROMATIC reciprocity failure explicitly stated for the PRO negative line.**
Process: **Process C-41 or equivalent, and Fujifilm Process CN-16** (p3). Times/temps n/p. Densitometry status n/p in 160S/160C/400H/800Z sheets.
Base (p1): Roll **Cellulose Triacetate 122 µm (135), 98 µm (120,220)**; Sheet **Polyester 175 µm** (160S/160C only; 400H/800Z roll only).
Curves p8 (char., spectral sensitivity, MTF, spectral dye density): **VECTOR** — ndraw 622–625, 0 embedded images (longest single path 34 items; curves built from many short stroked segments).

### 1.14 FUJICOLOR TRUE DEFINITION 400 [CH] — `True_Definition_DataSheet.pdf` (AF3-196E, 6 pp) — not ours
ISO 400/27° daylight/flash, filter None; tungsten 3200 K **100/21° equivalent via Wratten No.80A (LBB-12)** = conversion-filter index (p1). RMS **5*** (same footnote/conditions as PRO line), aperture 48 µm (p5 §15). Resolving **50 / 125 lp/mm** (p5). Reciprocity (p3): "No exposure **or color balance** compensation is required for 1/4000–2 s"; 4 s +1/3, 16 s +2/3, 64 s +1 — **no CC filter anywhere -> ACHROMATIC**. Process **CN-16, CN-16Q, CN-16FA, CN-16L, CN-16S or C-41** (p3). **Densitometry Status M**, exposure Daylight 1/125 s, density 1.0 above D-min (p6). Base Roll Cellulose Triacetate **122 µm (135)** (p1). Curves p6 **VECTOR** (ndraw=51, 0 images), spectral axis **400–700 nm** ticks.

### 1.15–1.18 FUJICOLOR SUPERIA 100 / 200 / REALA / 1600 — not ours
| Film | File (Ref) | ISO daylight | Tungsten 3200 K index (filter) | RMS | Resolving 1.6:1 / 1000:1 |
|---|---|---|---|---|---|
| SUPERIA 100 [CN] | `superia_100_datasheet.pdf` (AF3-007E, 4 pp) | 100/21° | **25/15°* via LBB-12 (or Kodak No.80A)** | **4** | 63 / 125 lp/mm (p3) |
| SUPERIA 200 [CA] | `superia_200_datasheet.pdf` (AF3-008E, 4 pp) | 200/24° | **50/18°* via LBB-12 (No.80A)** | **4** | 50 / 125 lp/mm (p4) |
| SUPERIA REALA [CS] | `superia_reala_datasheet.pdf` (AF3-967E, 4 pp) | 100/21° | **25/15°* via LBB-12 (No.80A), +2 stops** | **4** | 63 / 125 lp/mm (p3) |
| SUPERIA 1600 [CU] | `superia_1600_datasheet.pdf` (AF3-145E, 6 pp) | 1600/33° | **400/27°* via LBB-12 (No.80A), +2 stops** | **7** | 50 / 125 lp/mm (p5) |
RMS conditions for all four: micro-densitometer aperture **48 µm diameter**, magnification **12X**, sample density **1.0 above minimum density**.
Reciprocity: SUPERIA 100 (p2) 1/4000–2 s unnecessary, 4 s +1/3, 16 s +2/3, 64 s +1; SUPERIA 1600 (p2) 1/4000–2 s unnecessary, 4 s +2/3, 16 s +1½, 64 s +2. Both state "**No exposure or color balance compensation is required** for 1/4000–2 s" and list **no CC filter** at any longer time -> **ACHROMATIC**.
Process: CN-16, CN-16Q, CN-16FA, CN-16L (+CN-16S for 200/REALA/1600) or Kodak C-41. Times/temps n/p.
Densitometry: **Status M**, with printed control aim on an 18 % grey card read through the RED filter — SUPERIA 100 **0.96–1.16**; SUPERIA 200 **0.97–1.16**; SUPERIA REALA **1.02–1.20**; SUPERIA 1600 **0.65–0.85**. SUPERIA 1600 curve captions: Exposure Daylight 1/125 s, Process CN-16 (dye density CN-16X), Densitometry Status M, density 1.0 above D-min (p6).
Base: Cellulose Triacetate (thickness n/p in the Superia sheets).
Curves: SUPERIA 100 p4 **RASTER** (4 images), SUPERIA 200 p4 **RASTER** (4 images), SUPERIA REALA p4 **VECTOR** (ndraw=649, 0 images, paths to 64 items), SUPERIA 1600 p6 **VECTOR** (ndraw=71, 0 images), spectral axis 400–700 nm.

### 1.19 `NPZ.pdf` (6 pp) — **UNUSABLE**
Text layer decodes to mojibake (non-standard font encoding, no ToUnicode). Zero recoverable parameters. Presumed Fujicolor NPZ 800; in any case not one of our stocks.

---
## 2. (a) OUR STOCKS NOW DOCUMENTED, MATCHED BY EXACT GENERATION
| Our stock | Documented here? | Sheet | Caveat |
|---|---|---|---|
| FUJI_NEOPAN_ACROS_100 | **Yes, partially** | `Acros-120_AF3-083E.pdf` | Sheet is explicitly **NEOPAN 100 ACROS (120)** — 120 roll only. Base thickness 0.104 mm is the 120 base; our 135 stock would use a different base gauge, and the resolving/RMS figures are printed for the 120 product. Grain/resolution/reciprocity/EI-per-developer data are emulsion properties and transfer with low risk; base thickness does not. |
| FUJI_VELVIA_50 | No | — | Velvia 100 and Velvia 100F sheets both state they exceed "the current ISO 50 Velvia" — different products. |
| FUJI_PROVIA_400X | No | — | Only Provia 100F [RDP III]. |
| FUJI_SENSIA_100 | No | — | Only Sensia 200 [RM] and Sensia 400 [RH]. |
| FUJI_NEOPAN_1600 | No | — | |
| FUJI_ETERNA_VIVID_500T_8547 | No | — | No motion-picture sheet in scope. |
| FUJICOLOR_A250 | No | — | |
| FUJI_F125_8530 | No | — | |
| FUJI_F125_8630 | No | — | |
| FUJICOLOR_SUPER_F500_8572 | No | — | |

## 3. (b) FILMS DOCUMENTED HERE THAT WE DO NOT HOLD
FUJICOLOR PRO 160S (x2 sheets: AF3-174E, AF3-203U), PRO 160C (x2: AF3-175E, AF3-204U), PRO 400H, PRO 800Z, FUJICOLOR TRUE DEFINITION 400, FUJICOLOR SUPERIA 100 / 200 / REALA / 1600, FUJICHROME PROVIA 100F, ASTIA 100F, Velvia 100, Velvia 100F, Sensia 200, Sensia 400, 64T Type II [RTP II], T64 [RTP], plus the undecodable NPZ. 20 of 21 files.

## 4. (c) VECTOR SPECTRAL-SENSITIVITY CURVES (file + page)
Criterion: page contains a SPECTRAL SENSITIVITY section, 0 embedded images, and stroked path geometry.
| File | Page | Evidence |
|---|---|---|
| `AF3-203U_Pro160S_Product_Information_Bulletin.pdf` | 8 | 622 drawings, 0 images |
| `AF3-204U_Pro160C_Product_Information_Bulletin.pdf` | 8 | 622 drawings, 0 images |
| `pro_160s_datasheet.pdf` | 8 | 622 drawings, 0 images |
| `pro_160c_datasheet.pdf` | 8 | 622 drawings, 0 images |
| `pro_400h_datasheet.pdf` | 8 | 625 drawings, 0 images |
| `pro_800z_datasheet.pdf` | 8 | 625 drawings, 0 images |
| `True_Definition_DataSheet.pdf` | 6 | 51 drawings (longest 34 items), 0 images |
| `superia_1600_datasheet.pdf` | 6 | 71 drawings, 0 images |
| `superia_reala_datasheet.pdf` | 4 | 649 drawings (paths to 64 items), 0 images |
| `astia_100f_datasheet.pdf` | 7 and 8 | p7: 306 drawings, longest path 88 items, 0 images; p8: 49 drawings, 0 images |
| `velvia_100f_datasheet.pdf` | 7 and 8 | p7: 312 drawings, longest 89 items, 0 images |
| `velvia_100_datasheet.pdf` | 7 and 8 | p7: 702 drawings, longest 88 items, 0 images; p8: 567 drawings, 0 images |
| `t64_datasheet.pdf` | 6 and 7 | p6: 302 drawings, longest 88 items, 0 images |
| `sensia_200_datasheet.pdf` | 5 | MIXED — 475 drawings (39-item paths) **plus** 1 image 941x714 |
| `sensia_400_datasheet.pdf` | 5 | MIXED — 484 drawings **plus** 1 image 941x714 |
RASTER-only spectral pages: `Acros-120_AF3-083E.pdf` p3 (1 image 890x816) — **our only stock's curve is raster**; `provia_100f_datasheet.pdf` p6 (9 images); `RTPIIAF3-024E_1.pdf` p6 (9 images); `superia_100_datasheet.pdf` p4 (4 images); `superia_200_datasheet.pdf` p4 (4 images).
Per-layer: colour sheets print R/G/B layer curves (section titled "SPECTRAL SENSITIVITY CURVES", plural); Acros prints a single panchromatic curve. Wavelength axis ticks recoverable as text (400/500/600/700 nm) in `True_Definition` p6, `superia_1600` p6, `t64` p7; elsewhere only the label "Wavelength (nm)".

## 5. (d) ALL RMS GRANULARITY AND RESOLVING-POWER NUMBERS
| Film | RMS | Aperture / conditions | 1.6:1 | 1000:1 |
|---|---|---|---|---|
| NEOPAN 100 ACROS (120) | 7 | 48 µm, 12X, D=1.0 above Dmin, Microfine | 60 | 200 |
| 64T Type II [RTP II] | 10 | 48 µm, D=1.0 above Dmin | 55 | 135 |
| T64 [RTP] | 7 | 48 µm | 55 | 115 |
| Velvia 100 | 8 | 48 µm, gross diffuse visual density 1.0 | 80 | 160 |
| Velvia 100F | 8 | 48 µm, gross diffuse visual density 1.0 | 80 | 160 |
| ASTIA 100F | 7 | 48 µm, gross diffuse visual density 1.0 | 60 | 140 |
| PROVIA 100F | 8 | 48 µm, D=1.0 above Dmin | 60 | 140 |
| Sensia 200 | 13 | 48 µm, D=1.0 above Dmin | 60 | 140 |
| Sensia 400 | 13 | 48 µm, D=1.0 above Dmin | 55 | 135 |
| PRO 160S | 3* | 48 µm, D=+1.0 above Dmin (negative scale, not comparable to reversal) | 63 | 125 |
| PRO 160C | 3* | as above | 63 | 125 |
| PRO 400H | 4* | as above | 50 | 125 |
| PRO 800Z | 5* | as above | 50 | 115 |
| TRUE DEFINITION 400 | 5* | as above | 50 | 125 |
| SUPERIA 100 | 4 | 48 µm, 12X, D=1.0 above Dmin | 63 | 125 |
| SUPERIA 200 | 4 | 48 µm, 12X | 50 | 125 |
| SUPERIA REALA | 4 | 48 µm, 12X | 63 | 125 |
| SUPERIA 1600 | 7 | 48 µm, 12X | 50 | 125 |
All resolving-power figures in lines/mm; contrast column headings are "Test-Object Contrast" (colour sheets) / "Chart Contrast" (Acros).

## 6. (e) USELESS FILES
- `NPZ.pdf` — text layer is mojibake; nothing extractable.
- `AF3-203U_Pro160S_Product_Information_Bulletin.pdf` and `pro_160s_datasheet.pdf` are duplicate content under two Ref. Nos (AF3-203U / AF3-174E); same for `AF3-204U_Pro160C...` / `pro_160c_datasheet.pdf` (AF3-204U / AF3-175E). Two of the four are redundant.
- Everything else is data-bearing but for films we do not hold; only `Acros-120_AF3-083E.pdf` touches our stock list.

## 7. PARAMETER CLASSES ABSENT ACROSS THE ENTIRE SET
- **Process time and temperature**: never printed. Colour sheets only name the process (C-41 / CN-16 / CN-16Q / CN-16FA / CN-16L / CN-16S / E-6 / CR-56 / C6R). Only the B&W Acros sheet gives times and temperatures, and those are for B&W developers.
- **Numeric gamma / contrast index, Dmin, Dmax**: never printed. Only graphs; "1.0 above D-min" appears solely as a densitometry sampling condition.
- **Spectral sensitivity in tabulated numeric form / wavelength range statement**: never printed; graphs only, axis 400–700 nm where tick text survives.
- **Densitometry status** absent from: Acros 120, all four PRO 160S/160C/400H/800Z sheets, Sensia 200/400, Superia REALA curve captions. Present as Status A (T64, 64T II, Provia 100F, Astia 100F, Velvia 100F) or Status M (True Definition, Superia line).
- **Base thickness** absent from all four Superia sheets and from Acros in µm form (given as 0.104 mm).
- **MTF** absent only from Acros (B&W); present in every colour sheet.
