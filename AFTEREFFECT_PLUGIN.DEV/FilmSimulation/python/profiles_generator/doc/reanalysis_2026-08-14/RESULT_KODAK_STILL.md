# KODAK STILL-FILM DATASHEET EXTRACTION — 62 files / 449 pages
Source root: `PDF/PROFILES/KODAK/`. All 62 files opened with PyMuPDF (`fitz`), every page's
`get_text()` read and every page's `get_drawings()` inspected. Nothing below is estimated —
every value is printed in the cited file/page. Where a speed is quoted **through a conversion
filter** it is marked `[FILTER]` and is a filter factor, not a film property.

Kodak's granularity aperture is **48 µm (48-micrometre) at 12X magnification** in every
black-and-white / reversal sheet that states it; motion-picture sheets state 48 µm without
the 12X. Resolving power is always given as an ISO-6328-like pair at **TOC 1.6:1** and
**TOC 1000:1**.

---

## 1. PER-FILM PARAMETER TABLE

### 1.1 KODAK T-MAX Professional Films — `f32-TMAX.pdf` (28 pp, Mar 2002) and `f32-TMAX-200109.pdf` (28 pp, Sep 2001)
The two editions are numerically **identical** (verified by digit-stream diff); the Mar-2002
edition adds only a discontinuance notice pointing to F-4016. Page numbers below are the same
in both files.

| Parameter | T-MAX 100 / TMX | T-MAX 400 / TMY | T-MAX P3200 / TMZ | Source |
|---|---|---|---|---|
| Nominal speed (no filter) | EI/ISO **100/21°** | **400/27°** | nominal **EI 1000** in T-MAX/T-MAX RS Dev., **EI 800** in other Kodak developers; rounded to **EI 800** | p3 (100/400), p17 (P3200) |
| Daylight vs tungsten | "Virtually no difference between the daylight and tungsten film speeds" | same | — | p2 |
| Speed vs developer | T-MAX 100/21; T-MAX RS 100/21; XTOL 100/21; D-76 & D-76(1:1) 100/21; HC-110(B) 100/21; MICRODOL-X **50/18**, MICRODOL-X(1:3) 100/21; DURAFLO RT **80/20** | T-MAX 400/27; T-MAX RS 400/27; XTOL 400/27; D-76 & 1:1 400/27; HC-110(B) **320/26**; MICRODOL-X **200/24**, 1:3 **320/26**; DURAFLO RT 400/27 | T-MAX/RS, XTOL, D-76, HC-110(B), DURAFLO RT all: 800/30, 1600/33, 3200/36, 6400/39; T-MAX/RS additionally 12,500/42 and 25,000/45 | p3 table; p17 table |
| Reversal use | EI 50 (T-MAX 100 Direct Positive Outfit) | — | also usable at EI 400 | p3, p17 |
| Push speeds | 200/24 (normal proc.), 400/27 (2-stop push), 800/30 (3-stop push) | 800/30 (normal), 1600/33 (2-stop), 3200/36 (3-stop) | see above | p3 |
| **Diffuse rms granularity** | **8** | **10** | **18** | p13 (100/400), p23 (P3200) |
| Granularity aperture | net diffuse density **1.00**, **48-micrometre aperture, 12X magnification** | same | same | p13, p23 |
| **Resolving power** | **63 lines/mm @ TOC 1.6:1**; **200 lines/mm @ TOC 1000:1** | **50 lines/mm @ 1.6:1**; **125 lines/mm @ 1000:1** | **40 lines/mm @ 1.6:1**; **125 lines/mm @ 1000:1** | p13, p23 |
| Resolving-power method | "similar to ISO 6328, Photography—Determination of ISO Resolving Power" | same | same | p13, p23 |
| Image-structure process basis | D-76, 68°F (20°C) | same | same | p13, p23 |
| MTF curve | YES (tungsten, D-76 68°F, small tank, diffuse visual) | YES | YES | p14 (100/400), p24 (P3200) |
| Characteristic curves | YES ×4 (T-MAX Dev 75°F 5/7/9/11 min; D-76 68°F 6/8/10/12 min; T-MAX RS 75°F 6/8/10/12 min; DURAFLO RT VERSAMAT 5 @2.2/3/4 ft-min) | YES ×4 (same developer families) | YES ×5 (T-MAX 75°F 6/8/10/12; D-76 75°F 6/8/10/12; T-MAX RS 75°F 8/10/12/14/16; rotary-tube T-MAX 4.5/6/8/10/12; DURAFLO RT 1.5/2/2.5/3/4 ft-min) | p15, p16, p24–25 |
| Spectral-sensitivity curves | YES (D-76 68°F, 1.4 s effective exposure, diffuse visual, at 0.3 and 1.0 above D-min) | YES | YES | p14, p24 |
| Gamma / contrast index | No CI number printed; **development-time adjustment factor matrix** for 20% less / normal / 20% more / 40% more contrast at 65–75°F for T-MAX & T-MAX RS, D-76 & MICRODOL-X, HC-110(B), MICRODOL-X(1:3) | same | — | p13 |
| Dmin / Dmax | not printed numerically (curves only) | " | " | — |
| **Reciprocity** | 1/10,000 s **+1/3 stop**; 1/1000–1/10 none; 1 s **+1/3 stop**; 10 s **+1/2 stop** (or 15 s); 100 s **+1 stop** (or 200 s) | 1/10,000–1/10 none; 1 s **+1/3 stop**; 10 s **+1/2 stop** (or 15 s); 100 s **+1 1/2 stops** (or 300 s) | 1/10,000–1/10 none; 1 s **+1/3 stop**; 10 s **+2/3 stop** (or 15 s); 100 s **+2 stops** (or 400 s) | p4, p18 |
| Reciprocity CC filter | none (B&W) | none | none | — |
| Process / dev. time / temp | D-76 68°F(20°C) & full manual/rotary/machine tables 65–75°F; T-MAX Dev & T-MAX RS at 75°F(24°C); DURAFLO RT VERSAMAT | same | large-tank T-MAX RS 72°F(22°C): EI400 6–8, 800 6–8, 1600 8–10, 3200 10–12, 6400 12–14, 12,500 14–16 min | p5–p12, p19–p23 |
| Densitometry | **Diffuse visual** | same | same | p14–16, p24–25 |
| **Base material / thickness** | 135: **5-mil (0.13 mm) acetate**; 120: **4.7-mil (0.11 mm) acetate**; sheet & READYLOAD: **7-mil (0.18 mm) ESTAR Thick** | 135: 5-mil (0.13 mm) acetate; 120: 4.7-mil (0.11 mm) acetate; sheet: 7-mil (0.18 mm) ESTAR Thick | 135 only: **5-mil (0.13 mm) acetate** | p26; base benefit also p2 |
| Filter factors | full WRATTEN table, daylight + tungsten, identical for TMX/TMY | " | separate table p18 | p4, p18 |
| Orange mask | n/a (B&W) | | | |

### 1.2 KODAK TRI-X Pan / TRI-X Pan Professional — `f9-Tri-X_Pan-199906.pdf` (12 pp, Jun 1999)
| Parameter | TRI-X Pan / **TX** | TRI-X Pan Prof. / **TXP** | TRI-X Pan Prof. / **TXT** | Source |
|---|---|---|---|---|
| Speed | **ISO 400/27°** | **ISO 320/26°** | **ISO 320/26°** | p1 |
| **Diffuse rms granularity** | **17** (Fine) | **16** (Fine) | **16** (Fine) | p7 |
| Granularity aperture | net diffuse density **1.0**, **48-micrometre aperture, 12X magnification** | same | same | p7 |
| Resolving power | **NOT printed** ("High resolving power" prose only, p1) | " | " | p1 |
| **Contrast index** | development recommendations aim at **CI 0.56**; image-structure basis HC-110(B) 68°F, 7 1/2 min small tank | CI 0.56; HC-110(B) 68°F **5 1/2 min** small tank | CI 0.56; HC-110(B) 68°F **7 1/2 min** large tank | p4, p7 |
| Characteristic curves | YES — D-76 68°F large tank 7/9/11 min; T-MAX Dev 75°F small tank 5/7/9/11 min (p8) | YES — HC-110(B) 68°F large tank 3/4/6/9 min (p10) | YES — HC-110(B) 70°F(21°C) large tank 5/7/8/10 min (p11) | p8, p10, p11 |
| Contrast-index curves | YES (D-76, D-76 1:1, HC-110 B, DK-50 1:1, MICRODOL-X, MICRODOL-X 1:3; T-MAX Dev) | YES (D-76, HC-110 B, MICRODOL-X, DK-50 1:1) | YES | p8, p10, p11 |
| MTF curve | YES (D-76 68°F, large tank, 8 min, tungsten, diffuse visual) — TX only | not printed | not printed | p9 |
| Spectral-sensitivity curve | YES (at 1.0 + D-min) — TX only | — | — | p9 |
| **Reciprocity** (exposure AND development) | 1/100,000 **+1 stop, dev +20%**; 1/10,000 **+1/2 stop, +15%**; 1/1000 none, **+10%**; 1/100 & 1/10 none; **1 s +1 stop (or 2 s), –10%**; **10 s +2 stops (or 50 s), –20%**; **100 s +3 stops (or 1200 s), –30%** | same table | same table | p2 |
| Reciprocity nomographs | YES, vector (calc. vs adjusted exposure time, 1–10 s and 10–100 s) | — | — | p2 |
| Dev. times | full 65–75°F small/large tank matrix: T-MAX, T-MAX RS, HC-110(B), D-76, D-76(1:1), DK-50(1:1), MICRODOL-X, MICRODOL-X(1:3) | separate matrix | separate | p4–p6 |
| Densitometry | Diffuse visual | same | same | p8–p11 |
| **Base** | 135-24/36 **5-mil acetate**; 120 **3.6-mil acetate**; 35/70 mm long rolls 5-mil acetate | 120 & 220 **3.6-mil acetate** | sheets **7-mil ESTAR Thick** | p1 |
| Filter factors | WRATTEN daylight/tungsten table | separate table (adds No. 29 for TXT) | " | p3 |

### 1.3 KODAK PLUS-X Pan / PLUS-X Pan Professional — `f8-Plus-X_Pan-199709.pdf` (12 pp, Sep 1997)
| Parameter | PX / PXP | PXE / PXT | Source |
|---|---|---|---|
| Speed | **ISO 125/22°** | **ISO 125/22°** | p1 |
| **Diffuse rms granularity** | **10** (Extremely Fine) | **14** (Very Fine) | p7 |
| Granularity aperture | density **1.0**, **48-micrometre aperture, 12X magnification** | same | p7 |
| Resolving power | **NOT printed** (prose "High resolving power") | " | p1 |
| Contrast index | **CI 0.56**; PX & PXP basis HC-110(B) **70°F (21°C), 5 min large tank**; PXE/PXT basis HC-110(B) **68°F, 6 1/2 min large tank** | " | p3, p7 |
| Characteristic curves | YES p8 (HC-110 B 68°F 5/8/12/16 min; T-MAX Dev 75°F 5/7/9/11 min) | YES p10, p11 | p8, p10, p11 |
| Contrast-index curves | YES p8 (D-76, D-76 1:1, HC-110 B, MICRODOL-X, MICRODOL-X 1:3, T-MAX RS) | YES p10, p11 | p8, p10, p11 |
| MTF curve | YES | YES | p10, p11 |
| **Reciprocity** | 1/100,000 **+1 stop, dev +20%**; 1/10,000 **+1/2 stop, +15%**; 1/1000 none **+10%**; 1/100, 1/10 none; **1 s +1 stop (2 s), –10%**; **10 s +2 stops (50 s), –20%**; **100 s +3 stops (1200 s), –30%** | same | p2 |
| Reciprocity nomographs | YES, vector | | p2 |
| Dev. times | 65–75°F small/large tank: T-MAX, T-MAX RS, XTOL (separate for 135 PX and 120 PXP), HC-110(B), D-76, D-76(1:1), MICRODOL-X, MICRODOL-X(1:3); long rolls HC-110(B) 6 min @68°F or 4 1/4 min @75°F | | p3 |
| Densitometry | Diffuse visual | | p8 |
| **Base** | PX 135 & 35 mm long roll: **5-mil gray acetate**; PXP 120/220: **3.6-mil acetate** | PXE 70 mm: **4-mil ESTAR**; PXT sheets: **7-mil ESTAR Thick** | p1 |
| Filter factors | WRATTEN daylight/tungsten table | same | p2 |

### 1.4 KODAK PROFESSIONAL PLUS-X 125 Film / 125PX — `f4018-125PX-2007.pdf` (10 pp, May 2007)
New-coating-facility successor to F-8. **This is the only Plus-X sheet that prints resolving power.**
| Parameter | Value | Source |
|---|---|---|
| Speed | **125/22°** (EI table for 998/1002/1006 developer codes all 125/22°); push table EI 250/25° "normal" and 500/28° | p2, p7, p10 |
| **Diffuse rms granularity** | **10 (extremely fine)** | p7 |
| Granularity aperture | net diffuse density 1.0, **48-micrometre aperture, 12X magnification** | p7 |
| **Resolving power** | **ISO RPL 50 lines/mm (TOC 1.6:1)**; **125 lines/mm (TOC 1000:1)** | p7 |
| Image-structure basis | HC-110 (Dilution B), **70°F (21°C), 5 min large tank**, agitation 1-min intervals | p7 |
| Characteristic curves | YES — D-76 20°C(68°F) small tank 5/7/11 min; T-MAX Dev 20°C small tank 6/7/11 min; daylight; diffuse visual | p7 |
| Contrast-index curves | YES ×4 (small tank & large tank, 68°F; D-76, D-76 1:1, T-MAX, T-MAX RS, HC-110 B, MICRODOL-X, MICRODOL-X 1:3, XTOL, XTOL 1:1); CI axis **0.3–1.1** | p8 |
| Spectral-sensitivity curve | YES (1.4 s effective exposure, diffuse visual, D=0.30>min and 1.00>min) | p9 |
| MTF curve | **NOT printed** | — |
| **Reciprocity** | 1/100,000 **+1 stop, dev +20%**; 1/10,000 **+1/2 stop, +15%**; 1/1000 none **+10%**; 1/100 & 1/10 none; (1/10/100 s rows continue as F-8) | p2 |
| Roller-transport machine | VERSAMAT 5 & 411: 4.0 ft/min normal (EI 250), 3.0 ft/min (EI 500); VERSAMAT 11: 8.5 ft/min normal, 6.4 ft/min; other roller transport 80°F(27°C): 60 s normal, 80 s | p7 |
| **Base** | 135 & 35 mm long rolls **5-mil gray acetate**; 120 & 220 **3.6-mil acetate** | p1 |

### 1.5 KODAK EKTAPAN Film / EKP — `f10-Ektapan.pdf` (4 pp, Dec 2002)
| Parameter | Value | Source |
|---|---|---|
| Speed | **ISO 100/21°** panchromatic (single value, daylight or tungsten use stated) | p1 |
| **Diffuse rms granularity** | **12 (Extremely Fine)** | p3 |
| Granularity aperture | net diffuse density 1.0, **48-micrometre aperture, 12X magnification** | p3 |
| Resolving power | **NOT printed** ("High resolving power" prose) | p1 |
| Contrast index | development aims at **CI 0.56**; image-structure basis HC-110(B) **68°F (20°C), 6 min large tank** | p2, p3 |
| Characteristic curves | YES (HC-110 Dil B, 68°F, large tank, 5/7/8/12 min) | p3 |
| Contrast-index curves | YES (T-MAX RS, HC-110 Dil A, HC-110 Dil B, D-76, DK-50 1:1, MICRODOL-X; tray 68°F) | p3 |
| MTF / spectral-sensitivity | **NOT printed** | — |
| **Reciprocity** | 1/1000, 1/100, 1/10 none; **1 s +1 stop (2 s), dev –10%**; **10 s +2 stops (50 s), –20%**; **100 s +3 stops (1200 s), –30%** | p2 |
| Reciprocity nomographs | YES, vector | p2 |
| Dev. times | tray & large-tank 65–75°F: T-MAX RS, XTOL, HC-110(B), HC-110(A), D-76, DK-50(1:1), MICRODOL-X; long rolls HC-110(B) | p2, p3 |
| **Base** | sheets & 70 mm/3.5-in rolls: **7-mil ESTAR Thick** (all sizes) | p1 |
| Filter factors | WRATTEN daylight/tungsten table | p2 |

### 1.6 KODAK VERICOLOR III Professional Film — `e26-Vericolor_III.pdf` (6 pp, Apr 1997)
| Parameter | Value | Source |
|---|---|---|
| Speed (no filter) | **ISO 160/23°** daylight or electronic flash (nominal); "many photographers prefer EI 125" | p1 |
| Speed through filters `[FILTER]` | photolamp 3400 K + **No. 80B → 50/18°**; tungsten 3200 K + **No. 80A → 40/17°** | p1 |
| Balance | daylight / electronic flash; designed for 1/10,000–1/10 s | p1 |
| Granularity | **rms NOT given** — **Print Grain Index**: 135 (24×36 mm) 39 / 61 / 91 at 4.4X / 8.8X / 17.8X; 120-220 (6×6 cm) 27 / 39 / 61 at 2.6X / 4.4X / 8.8X; 4×5 sheets <25 / <25 / 38 at 1.2X / 2.1X / 4.2X | p3 |
| Resolving power | NOT printed | — |
| MTF curve | **YES** (C-41, daylight, diffuse visual) | p4 |
| Characteristic curves | YES (C-41, daylight 1/1000 s, Status M, R/G/B) | p4 |
| Spectral-sensitivity curves | YES (C-41, 1.4 s effective exposure, Status M, D=1.0 above D-min) | p4 |
| Spectral-dye-density curves | YES (C-41, Status M, midscale neutral + minimum density) | p4 |
| Note on curve set | curves for **2106 / 4106** are **0.04 lower in overall density** than the printed curves (which apply to 5026 / 6006) | p4 |
| Gamma / CI / Dmin / Dmax | not printed numerically | — |
| **Reciprocity** | **No filter correction or exposure compensation for 1/10,000 s to 1/10 s** (no data beyond 1/10 s) | p2 |
| Process / densitometry | **C-41** (FLEXICOLOR); **Status M** densitometry, red filter (or WRATTEN No. 92) | p1, p3 |
| Aim densities | Gray Card 0.73–0.93; lightest gray-scale step 1.25–1.45; forehead light 1.05–1.35, dark 0.75–1.15 | p3 |
| **Base** | 135 (5026) **5-mil acetate**; 120/220 (6006) **3.6-mil acetate**; 70 mm (2106) 4-mil base variant noted; sheets (4106) **7-mil ESTAR** | p3, p5 |
| **Mask statement** | "**Dye-masking color couplers** — quality color reproduction without supplementary masking"; "Not subject to leuco-cyan-dye formation" | p1 |

### 1.7 KODAK PROFESSIONAL PROFOTO 100 Colour Negative Film — `e2e-Profoto_100.pdf` (4 pp, Jul 1997)
| Parameter | Value | Source |
|---|---|---|
| Speed | **EI 100** (medium speed) | p1 |
| Filter exposure adjustments `[FILTER]` | photolamp 3400 K + No. 80B **+1 2/3 stops**; tungsten 3200 K + No. 80A **+2 stops** (adjustments, not indexes) | p1 |
| Granularity | rms NOT given — **Print Grain Index (135)**: **43 / 65 / 94** at 4.4X / 8.8X / 17.8X | p4 |
| Resolving power / MTF | **NOT printed** | — |
| Characteristic curves | YES (C-41, daylight, Status M, R/G/B) | p3 |
| Spectral-sensitivity curves | YES (C-41, 1/50 s effective exposure, Status M, D=0.2 above D-min) | p3 |
| Spectral-dye-density curves | YES (C-41, midscale neutral + minimum density) | p3 |
| **Reciprocity** | **no filter correction / exposure compensation for 1/10,000 s to 10 s** | p2 |
| Process / densitometry | **C-41** (FLEXICOLOR); **Status M** | p2 |
| Aim densities | Gray Card 1.03–1.23; gray-scale lightest step 1.43–1.63; forehead light 1.33–1.63, dark 1.08–1.48 | p2 |
| Storage | room temperature, 21°C (70°F) or lower | p1 |
| Base / mask | **not printed** | — |

### 1.8 KODAK ULTRA MAX 400 Film — `E7019_en-Ultra_Max_400.pdf` (6 pp, Feb 2007)
| Parameter | Value | Source |
|---|---|---|
| Speed | **ISO/DIN 400/27°** | p1 |
| Granularity | rms NOT given — **Print Grain Index 46** (135, 24×36 mm, 4×6 in print, 4.4X) | p3 |
| Resolving power / MTF | **NOT printed** | — |
| Characteristic curves | YES (daylight, C-41, Status M, R/G/B) | p4 |
| Spectral-sensitivity curves | YES (daylight, 1/100 s, Status M, D=0.2>D-min) | p4 |
| Spectral-dye-density curves | YES (C-41, midscale neutral + minimum density) | p4 |
| **Reciprocity** | **no exposure or filter adjustment 1/10,000 s to 10 s**; "exposures longer than 10 seconds may require compensation and filtration" (no numbers) | p2 |
| Process / densitometry | **C-41** (FLEXICOLOR); **Status M** | p3 |
| Aim densities | Gray Card 0.80–1.00; gray-scale lightest step 1.20–1.40; forehead light 1.10–1.40, dark 0.85–1.25 | p3 |
| Emulsion tech | T-GRAIN, antenna dye sensitization, advanced development accelerators | p1 |
| Base / mask / Dmin / gamma | **not printed** | — |

### 1.9 KODAK PROFESSIONAL PORTRA 400BW Film — `f4012-Portra_400BW.pdf` (8 pp, Jul 2003)
C-41 chromogenic B&W; the notice on p1 names **KODAK PROFESSIONAL BW400CN Film** as the successor.
| Parameter | Value | Source |
|---|---|---|
| Speed | **ISO 400 / 27°** ("True speed of ISO 400"); usable EI 50–1600 | p1, p2 |
| **RMS granularity** | **9 (Extremely fine)**, read at a **net diffuse visual density of 1.00 with a 48-micrometer aperture** (12X not stated) | p5 |
| Print Grain Index (135) | **<25 / 40 / 70** at 4.4X / 8.8X / 17.8X | p5 |
| Resolving power / MTF | **NOT printed** | — |
| Characteristic curve | YES (daylight, Status M, R/G/B, **Log H Ref –1.44**) | p6 |
| Spectral-sensitivity curve | YES (C-41, daylight, 1/100 s, Status M, D=0.2 above D-min) | p6 |
| Spectral-dye-density curves | YES ×2 (C-41, midscale neutral + minimum density) | p6 |
| **Reciprocity** | **no compensation for 1/10,000 to 120 seconds**; exposures >120 s not recommended | p3 |
| Process / densitometry | **C-41** (FLEXICOLOR) only — must NOT be processed in conventional B&W chemicals; **Status M** | p2, p3 |
| Dmin statement | "much lower D-min or base density [than colour negative]; the film base will appear **very light brown**" — i.e. **no orange mask** | p3 |
| Aim densities | Gray Card 0.80–1.00; gray-scale lightest step 1.15–1.35; forehead light 1.05–1.35, dark 0.90–1.20 | p4 |
| **Base** | 135-36 **0.13 mm (0.005 in) acetate**; 120 and 220 **0.10 mm (0.0039 in) acetate** | p2 |
| Filter factors | WRATTEN daylight/tungsten table (No.8 1.4/1.25 … No.47 12.5/16) | p3 |
| Printing filter pack | PORTRA 160NC balance **+5M** | p5 |

### 1.10 KODACHROME Professional Films (PKM / PKR / PKL) — `e55-1996_12.pdf` (8 pp), `e55-2003_08.pdf` (4 pp), `e55-2009_06.pdf` (6 pp)
| Parameter | KODACHROME 25 Prof. / PKM | KODACHROME 64 Prof. / PKR | KODACHROME 200 Prof. / PKL | Source |
|---|---|---|---|---|
| Speed, daylight/flash, **no filter** | **EI 25** | **EI 64** | **EI 200** | e55-1996 p2; e55-2003 p1; e55-2009 p1 |
| `[FILTER]` photolamp 3400 K + 80B | 8 | 20 | 64 | same |
| `[FILTER]` tungsten 3200 K + 80A | 6 | 16 | 50 | same |
| **Diffuse rms granularity** | **9** | **10** | **16** | e55-1996 p5/p6/p7; e55-2009 p3/p4/p5 (printed "Diffue rms Granularity") |
| Granularity aperture | **gross diffuse visual density 1.0, 48-micrometre aperture, 12X magnification** | same | same | same pages |
| Resolving power | NOT printed | NOT printed | NOT printed | — |
| MTF curves | **YES** | **YES** | **YES** | e55-1996 p5–7; e55-2009 p3–5 |
| Characteristic curves | YES (K-14, daylight 1/25 s, Status A) | YES (K-14, daylight 1/50 s, Status A) | YES (K-14, daylight 1/100 s **with a 0.20 ND filter**, Status A) | same |
| Spectral-sensitivity curves | YES (K-14, 1.4 s, **E.N.D.**, D=1.00) | YES | YES | same |
| Spectral-dye-density curves | YES, dyes normalized to visual neutral density 1.0 for a **3200 K viewing illuminant** | same | same | same |
| **Reciprocity (daylight)** | 1/1000–1/100 none; 1/10 **+1/2 stop, no filter**; 1 s **not recommended** | 1/1000–1/100 none; 1/10 **+1/3 stop, CC05R**; 1 s **not recommended** | 1/1000–1/100 none; 1/10 **+1/2 stop, CC10Y**; 1 s **not recommended** | e55-1996 p4 (table columns: 1/1000-1/100, 1/10, 1, 10 s) |
| Push | not recommended | not recommended | **EI 500 (push 1 1/3)**, **EI 800 (push 2)**; balance shifts magenta-red | e55-1996 p4 |
| Process | **K-14** | K-14 | K-14 | p4 |
| Densitometry | **Status A** | Status A | Status A | p5–7 |
| Viewing illuminant | transparencies for **5000 K** viewing/projection | same | same | p1 |
| **Base** | 135-36 **5.3-mil acetate** | 5.3-mil acetate | 5.3-mil acetate | p2 |
| Storage | refrigerate at 55°F (13°C) or lower | same | same | p2 |
| Note | `e55-2003_08.pdf` contains **no IMAGE STRUCTURE section and no curves** — exposure indexes only | | | e55-2003 p1–4 |

### 1.11 KODACHROME 25 / 64 / 200 Films (consumer KM/KR/KL) — `e88-1998_01.pdf`, `e88-2002_03.pdf`, `e88-2005_09.pdf`
| Parameter | KODACHROME 25 / KM | 64 / KR | 200 / KL | Source |
|---|---|---|---|---|
| Speed, daylight/flash, no filter | 25 | **64** | 200 | e88-1998 p1; e88-2002 p1; e88-2005 p1 |
| `[FILTER]` 3400 K + 80B | 8 | 20 | 64 | e88-2002 p1, e88-2005 p1 (e88-1998 lists 80A only) |
| `[FILTER]` 3200 K + 80A | 6 | 16 | 50 | all three |
| **Diffuse rms granularity — Jan 1998 ed.** | **11** | **12** | **19** | e88-1998_01 p5, p6, p7 |
| **Diffuse rms granularity — Mar 2002 ed.** | **9** | **10** | **16** | e88-2002_03 p5, p6, p7 |
| **Diffuse rms granularity — Sep 2005 ed.** | (25 dropped) | **10** | **16** | e88-2005_09 p4, p5 |
| Granularity aperture | gross diffuse visual density 1.0, **48-micrometre aperture, 12X magnification** | same | same | all |
| MTF curves | YES | YES | YES | e88-1998 p5–7; e88-2002 p5–7; e88-2005 p4–5 |
| Characteristic / spectral-sens. / spectral-dye | YES all three, K-14, Status A, dyes normalized for **3200 K** viewing | " | " | same pages |
| Base | 135-24 / 135-36 **5.3-mil acetate** | same | same | e88-1998 p1, e88-2002 p1 |
| Process / densitometry | K-14 / Status A | " | " | — |
| ⚠ Conflict | see §7 — 1998 vs 2002 editions disagree on all three rms values | | | |

### 1.12 EASTMAN EKTACHROME Film (Daylight) 7239 — `Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf` (4 pp, Feb 1999, H-1-5239)
| Parameter | Value | Source |
|---|---|---|
| Speed, **no filter** | **Daylight (5500 K) 160/23°** | p1 (twice: header "Daylight—160/23" and colour-balance table) |
| `[FILTER]` | Tungsten 3000 K + WRATTEN 80A **40/17**; tungsten 3200 K + 80A **40/17**; tungsten photoflood 3400 K + 80B **50/18** | p1 |
| Balance | balanced for **daylight** exposure; **balanced for projection at 5400 K** | p1 |
| Usable range | 1/2× to 2× the normal exposure indexes with little quality loss | p1 |
| **Diffuse RMS granularity** | **14** | p2 |
| Granularity aperture | net diffuse visual density 1.0, **48-micrometer aperture** | p2 |
| **Resolving power** | **TOC 1.6:1 → 40 lines/mm**; **TOC 1000:1 → 100 lines/mm** (ISO 6328-1982 method) | p2 |
| MTF curve | **YES** (ANSI PH2.39-1977(R1990)-like method) | p3 |
| Sensitometric (characteristic) curves | YES (**VNF-1**, Status A) | p3 |
| Spectral-sensitivity + spectral-dye-density | YES; dyes normalized to visual neutral density 1.0 for a **5400 K** viewing illuminant | p3 |
| **Reciprocity** | **no filter or exposure adjustment for 1 second to 1/10,000 second** | p2 |
| Process / densitometry | **VNF-1**; **Status A** | p2, p3 |
| LAD reference | KODAK Publication H-61, LAD—Laboratory Aim Density | p2 |

### 1.13 KODAK VERICHROME Pan Film — `f7-Verichrome-199611.pdf` (4 pp, Nov 1996)
Speed **EI 125** panchromatic, "characteristics similar to KODAK PLUS-X Pan" (p1).
**Diffuse rms granularity 9 (Extremely Fine)**, read at 48-micrometre aperture / 12X (p3).
**Contrast Index = 0.60**; image-structure basis **D-76 (1:1), 68°F (20°C), 9 min small tank** (p2, p3).
Characteristic curves + contrast-index curves YES (p3, vector). MTF NOT printed. Resolving power NOT printed.
Reciprocity: exposure + development adjustment table (p2). Base: **3.6-mil acetate** (120 and 8×5 sizes) (p1).

### 1.14 KODAK PROFESSIONAL Technical Pan Film — `p255.pdf` (12 pp, Feb 2000) and `p255-2003_06.pdf` (12 pp, Jun 2003)
Digit-stream diff: **numerically identical**; the 2003 edition differs only in title ("KODAK PROFESSIONAL"), a discontinuance notice, and size/CAT tables.
| Parameter | Value | Source |
|---|---|---|
| Speed range | **no single speed** — EI 16 (pictorial) to EI 320 (microfilming); pictorial **EI 25/15°** in TECHNIDOL Liquid or XTOL; **EI 64/19°** for high-contrast reversal (T-MAX 100 Direct Positive Outfit) | p2 |
| **Diffuse rms granularity** | **8 (Extremely fine)** | p9 |
| Granularity aperture | net diffuse density 1.0, **48-micrometre aperture, 12X magnification** | p9 |
| Resolving power | **NOT printed numerically** (prose "high resolving power", microfilming to 20X reduction) | p1, p4 |
| **Gamma / contrast index** | TECHNIDOL Liquid (Dil F) 68°F small tank: **5 min = CI 0.50, 7 min = 0.60, 9 min = 0.65, 11 min = 0.70**; associated EI 16–25. HC-110 (Dil D) 68°F: 2 min = 2.80, 4 min = 2.90, 8 min = 3.50 (EI 200 curve set labelled CI 2.50). Application aims: **gamma 0.6–1.0** for continuous-tone copying (HC-110 B 6 min, EI 160; TECHNIDOL 9 min, EI 25); **gamma 2.0** for microfilming (HC-110 Dil D, 8 min, 68°F, EI 125); CI 0.5–0.6 electron micrography; CI ≥1.5 holographic interferometry; gel work HC-110 Dil D 4 min, EI 80 | p4, p5, p10, p11 |
| VERSAMAT 11 EI/CI table | **CI 2.20** — VERSAMAT 885 dev., 85°F (29.4°C), 10 ft/min, 1 rack, **EI 160/23°**; **CI 1.40** — VERSAMAT 641, 85°F, 10 ft/min, 1 rack, **EI 125/22°**; **CI 1.40** — DURAFLO RT, 80°F (26.5°C), 10 ft/min, 2 racks, **EI 160/23°** (indexes based on 1/25-s daylight exposure) | p3 |
| Machine CI targets | ~CI 1.4 at 10 ft/min (3.05 m/min); ~CI 2.2 at 10 ft/min; ~CI 1.4 at 8 ft/min (2.4 m/min) in other configs, 57–60°C dev. | p8, p9 |
| MTF curves | **YES** | p9 |
| Spectral-sensitivity curves | YES (at 0.3 and 1.0 above D-min) | p9 |
| Characteristic curves | YES, many (TECHNIDOL Dil F 5/7/9/11 min; HC-110 sets) | p10, p11 |
| **Reciprocity** | 1/10,000 none + **dev +30%**; 1/1000 none + **+20%**; 1/100 & 1/10 none; **1 s none, –10%**; **10 s +1/2 stop (15 s), –10%**; **100 s +1 1/2 stops, none** | p3 |
| **Base** | 35 mm (2415) **ESTAR-AH Base**; 120 (6415) **3.6-mil acetate**; sheets **ESTAR Thick**; **the ESTAR-AH Base has 0.1 neutral density built in — one-half to one-third that of conventional 35 mm films** | p1, p9 |

### 1.15 KODAK High Speed Infrared Film (HIE / HSI) — `f13-HIE-200006.pdf` (8 pp, Jun 2000)
| Parameter | Value | Source |
|---|---|---|
| Speed, **no filter** | **Daylight/flash 80/20°**; **Tungsten 200/24°** | p3 |
| `[FILTER]` | No. 25/29/89B: daylight 50/18° (TTL-metered: EI 200), tungsten 125/22° (TTL: EI 500); No. 87: 25/15° / 64/19°; No. 87C: 10/11° / 25/15° | p3 |
| Spectral range | sensitive to 900 nm | p1 |
| **Diffuse rms granularity** | **18 (Fine)** | p6 |
| Granularity aperture | net diffuse density 1.0, **48-micrometre aperture, 12X magnification** | p6 |
| Resolving power | NOT printed ("Medium resolving [power]" prose) | p1 |
| **Contrast index ↔ D-max pairs** (T-MAX Dev) | CI **0.65 @ D-max 1.50**; **0.80 @ 1.76**; **0.91 @ 2.00**; **1.03 @ 2.36**; **1.15 @ 2.44** | p4, p5 |
| Image-structure basis | D-76, 68°F (20°C), 10 min | p6 |
| Characteristic curves | YES for HIE and for HSI | p6 |
| Contrast-index curves | YES for HIE and HSI (HC-110 Dil B etc.) | p6 |
| MTF curves | **YES** (HIE and HSI) | p7 |
| Spectral-sensitivity curves | YES (1.4 s, 0.3 and 0.6 + D-min, 68°F) | p7 |
| Filter transmittance data | tabulated % transmittance for No. 87, 87C, 89B from 700 nm upward + vector transmittance/density plot | p2 |
| **Base** | 135-36 and 35 mm long roll **4-mil ESTAR**; 4×5 sheets (HSI) **7-mil ESTAR Thick** | p1 |

### 1.16 KODAK Commercial Film — `f16-Commercial.pdf` (4 pp, Dec 2002)
Blue-sensitive, moderately high contrast. Speeds **all without filter**: **Daylight EI 50/18°**, **Tungsten or Quartz-Iodine EI 8/10°**, **Pulsed-Xenon Arc EI 12/12°** (p1).
**Diffuse rms granularity 12**, net diffuse density 1.0, **48-micrometre aperture, 12X** (p2).
Image-structure basis HC-110, 20°C (68°F), 2 1/4 min (p2). Characteristic curves + contrast-index curves YES (p2, p3). MTF NOT printed. Resolving power NOT printed.
Reciprocity: exposure + development adjustment table plus vector nomographs (p1, p2). Base: **7-mil ESTAR Thick** (p1).

### 1.17 KODAK Professional Copy Film — `f17-Copy.pdf` (4 pp, Mar 2003)
Orthochromatic. Speeds: **Tungsten ISO 12/12°**, **Pulsed-Xenon Arc EI 15/15°** (p1, no filter).
**Diffuse rms granularity 16 (Fine)**, net diffuse density 1.0, **48-micrometre aperture, 12X** (p3).
Image-structure basis HC-110 (Dilution E), 20°C (68°F), 5 min (p3).
Characteristic curves YES; **Reciprocity Curve YES (vector)** — DK-50 (1:1), tungsten, D = 1.0 + D-min, exposure axis to 0.001 s (p3). Spectral-sensitivity curves YES (p3). MTF NOT printed. Resolving power NOT printed. Base **7-mil ESTAR Thick** (p1).

### 1.18 KODAK PROFESSIONAL B/W Duplicating Film SO-132 — `f11-Duplicating_SO-132-200105.pdf` (4 pp, May 2001)
Orthochromatic, medium contrast, one-step duplication. **No ISO/EI printed** — trial exposure 40 s from a tungsten source giving 5 footcandles (54 lux) at the exposure plane (p1). Needs 2/3–1 stop more exposure than SO-339 (p1). **No granularity, no resolving power, no MTF.** Characteristic curves YES (p3, vector); basis VERSAMAT Model 5, 80°F (26.5°C), DURAFLO RT Developer Replenisher, two developer racks, 4.25 ft/min (p3). Base **7-mil ESTAR Thick** (p1). Safelight: KODAK 1A (light red).

### 1.19 KODAK EKTACHROME 100 Plus Professional Film / EPP — `e113-Ektachrome_100_plus_EPP.pdf` (6 pp, Jul 2007)
Speed **EI 100** daylight/flash, no filter; `[FILTER]` 3400 K + 80B **32**, 3200 K + 80A **25** (p2).
**Diffuse rms granularity 11 (very fine)**, gross diffuse visual density 1.0, **48-µm aperture, 12X** (p5).
Process **E-6**; densitometry **Status A**; characteristic curves (daylight 1/100 s), **MTF curves**, spectral-sensitivity, spectral-dye-density all YES (p5); dyes normalized for **5000 K** viewing (p5).
**Reciprocity:** none 1/10,000–1/10 s; **at 1 s use CC025R and increase exposure** (p3).
Base: 135/35 mm **5-mil (0.13 mm)**; 120/220 **3.9-mil (0.10 mm)**; sheets **7-mil (0.18 mm) ESTAR Thick** (p1). Resolving power NOT printed.

### 1.20 KODAK PROFESSIONAL ELITE Chrome / EKTACHROME reversal family
| File (edition) | Film | Speed (no filter) | `[FILTER]` 3200 K + 80A | rms gran. (48 µm, 12X, gross diffuse visual D 1.0) | Reciprocity | Base | Curves |
|---|---|---|---|---|---|---|---|
| `E7014e-Elitechrome_100.pdf` (4 pp, Apr 2005) **≡ `KODAK PROFESSIONAL ELITE Chrome 100 Film.pdf` (byte-identical text)** | ELITE Chrome 100 / EB | **100** | 25 | **8 (extremely fine)** | none 1/10,000–10 s | 135 **5-mil acetate** | char. + spectral-sens. p3; MTF + spectral-dye p4; "**Lower D-min**" feature claim p1 |
| `e126e-Elitechrome_100ec.pdf` (6 pp, Apr 2005) | ELITE Chrome Extra Color 100 / EBX | **100/21** | **25/15** | **11** | none 1/10,000–10 s | 135 **5-mil acetate** | char., spectral-dye, spectral-sens., **MTF** p5 |
| `e148e-Elite_chrome_200.pdf` (6 pp, Apr 2005) **≡ `KODAK PROFESSIONAL ELITE.pdf`** | ELITE Chrome 200 / ED | **200** (daylight-balanced) | 50 | **12** | none 1/10,000–10 s | 135 **5-mil acetate** | char. + **MTF** + spectral-sens. p4; spectral-dye p5 |
| `e149-Elite_chrome_400.pdf` (6 pp, Jan 1998) | ELITE Chrome 400 / EL | **400** | 100 | **19** | none 1/10,000–1/10 s; **at 1 s CC05R +1/3 or 1/2 stop** | 135 **5-mil acetate** | char. + **MTF** p3; spectral-sens. + spectral-dye p4 |
| `e145-Ektachrome_320T_EPJ.pdf` (4 pp, May 2007) | EKTACHROME 320T Prof. / EPJ | **tungsten 3200 K, no filter: ISO 320** | `[FILTER]` 3400 K + 81A **250**; daylight/flash + **85B → 200** | **19 (fine)** | none 1/10,000–1/10 s; **at 1 s +1/3 stop plus a CC filter** | **5.0-mil (0.13 mm)** | char. + **MTF** p3; spectral-sens. + spectral-dye p4. **Tungsten-balanced (3200 K)**; transparencies for 5000 K viewing |
| `e147-Ektachrome_P1600_EPH.pdf` (6 pp, May 2007) | EKTACHROME P1600 Prof. / EPH | **EI 1600** (Process E-6P / push 2; DX-coded 1600) | 3400 K + 80B **500**; 3200 K + 80A **400** | **34 (Coarse) at EI 1600** | none 1/10,000–1/10 s | **5.0-mil (0.13 mm)** | char. + **MTF** p5; spectral-sens. + spectral-dye p5 (Process E-6P/Push 2). First-developer adjustments: EI 1600 +5 min / +7°C(12°F); EI 800 +2 min / +4°C(8°F); EI 400 normal |
| `e4024-2009.pdf` (6 pp, Sep 2009) | EKTACHROME E100G and E100GX | **100** | 3400 K + 80B **32**; 3200 K + 80A **25** | **8 (extremely fine)** (aperture stated but 12X not repeated) | none 1/10,000–10 s; **at 120 s add CC10R**; ≤4 flashes ok, 8 flashes add CC05M | 135/35 mm **5-mil (0.13 mm)**; 120/220 **3.9-mil (0.10 mm)**; sheets **7-mil (0.18 mm) ESTAR Thick** | char. (both films) + spectral-sens. + spectral-dye p5; **MTF p6**. Push 1 = EI 200, 8 min first dev. "Lower D-min for whiter, brighter whites" p1 |
| `e2529-Ektachrome_EDUPE.pdf` (8 pp, May 2004) | EKTACHROME Duplicating Film EDUPE | **EI printed on the film carton** (no fixed number); starting points: tungsten 3200 K 1 s, 5000 K illuminator 1/8 s | flash filtration base **5500 K**; 6000→5500 K add No. 81, 5000→5500 K add No. 82; pulsed xenon **85B**; 5000 K illuminator **10M + 60Y** | **8.7 (extremely fine)** | "excellent reciprocity — no tone-scale compromise from 10 s" (no numeric table) | **5-mil (0.13 mm)** and **8.2-mil (0.21 mm)** | char. p6; **MTF** + spectral-sens. + spectral-dye p7. Process **E-6** |
| `ti2323-Ektachrome_EIR.pdf` (8 pp, Sep 2005) | EKTACHROME Prof. Infrared EIR | with WRATTEN **No. 12** (required): daylight/flash **EI 100 (Process AR-5) / EI 200 (Process E-6)**; tungsten 3200 K with No. 12 + CC20C + Corning CS 1-59 3966 (or No.12 + CC50C): **50 (AR-5) / 100 (E-6)** — all `[FILTER]` values, film cannot be used unfiltered | — | **17 (fine)** | none 1/1000–1/100 s; **at 1/10 s +1 stop and add CC20B** | **4-mil (0.101 mm) ESTAR Base** with fast-drying backing | char. + spectral-sens. p7; **MTF** + spectral-dye p8. Process **AR-5** (EA-5 chemicals) or **E-6** (higher contrast / saturation) |

### 1.21 KODAK ROYAL GOLD family (Process C-41; rms replaced by Print Grain Index)
| File | Film | ISO, daylight/flash, no filter | `[FILTER]` 3400 K + 80B | `[FILTER]` 3200 K + 80A | Print Grain Index (135, 4×6 in, 4.4X) | Reciprocity | Curves |
|---|---|---|---|---|---|---|---|
| `e40-1996_12.pdf` (4 pp, Dec 1996) | ROYAL GOLD 25 | **25/15°** | **8/10°** | **6/9°** | **Less than 25** | none **1/10,000 s to 100 s** | char. + spectral-sens. + spectral-dye p3. "**Built-in dye-masking**"; usable ISO 12–400 |
| `e41-1998_02.pdf` (4 pp, Feb 1998) | ROYAL GOLD 100 | **100/21°** | **32/16°** | **25/15°** | **28** | none 1/10,000–10 s | char. + spectral-sens. + spectral-dye p3; usable ISO 25–800 |
| `e42-1998_02.pdf` (4 pp, Feb 1998) | ROYAL GOLD 200 | **200/24°** | **64/19°** | **50/18°** | **41** | none 1/10,000–10 s | p3; usable ISO 50–1600 |
| `e43-1998_02.pdf` (6 pp, Feb 1998) | ROYAL GOLD 400 | **400/27°** | **125/22°** | **100/21°** | **41** | none 1/10,000–10 s | p4; usable ISO 250–2000 |
| `e44-1998_02.pdf` (6 pp, Feb 1998) | ROYAL GOLD 1000 | **1000/31°** | **320/26°** | **250/25°** | **57** | none 1/10,000–10 s | p4 |
| `e2509-2000_01.pdf` (6 pp, Jan 2000) | ROYAL GOLD 400 (later ed.) | **400/27** | **125/22** | **100/21** | **39** | (per file) | char. + spectral-sens. + spectral-dye p4 |
| `e7006-2002_03.pdf` (4 pp, Mar 2002) | ROYAL GOLD 200 / RB | **200/24°** | **64/19°** | **50/18°** | **32** | (per file) | char. + spectral-sens. p3; spectral-dye p4; usable ISO 25–800 |
All: **Process C-41 (FLEXICOLOR)**, **Status M** densitometry, no rms granularity, no resolving power, no MTF.
⚠ ROYAL GOLD 400 PGI: **41** in `e43-1998_02.pdf` vs **39** in `e2509-2000_01.pdf` (see §7).

### 1.22 KODAK consumer / commercial colour negative (Process C-41, Status M, PGI only)
| File | Film(s) | ISO no filter | `[FILTER]` 3400 K / 3200 K | Print Grain Index | Reciprocity | Curves |
|---|---|---|---|---|---|---|
| `e2328-GA100.pdf` (4 pp, Jul 2003) | Bright Sun Film / GA (ISO 100) | **100/21°** | **32/16°** / (3200 K row listed) | **45** | none 1/10,000–10 s | char. + spectral-sens. p3; spectral-dye p4 |
| `e7013-HD400.pdf` (6 pp, Jan 2003) | High Definition 400 | **400/27°** | **125/22°** / **100/21°** | **39** | (per file) | char. + spectral-dye p5 |
| `e7017-HD200.pdf` (6 pp, Jul 2003) | High Definition 200 / 3992 / HD2 | **200/24°** | **64/19°** / **50/18°** | **32** | (per file) | char. + spectral-sens. + spectral-dye p5 |
| `e4039-Elite.pdf` (10 pp, Jul 2006) | ELITE COLOR 200 and 400 | **200** and **400** | 200: 64 / 50; 400: 125 / 100 | 200 Film **32 / 54 / 84**; 400 Film **39 / 61 / 90** (4.4X / 8.8X / 17.8X) | none 1/10,000–10 s for both | char. (EI 400 and EI 800 push 1) + spectral-sens. + spectral-dye + **MTF** p7, p8. Extensive fluorescent/HID CCT table (T8 830 3000 K, 835 3500 K, 841/741 4100 K, HPS 2100/2200/2700 K, metal halide 3200/4300 K, mercury 3700 K) |
| `e4026-2002_06.pdf` (10 pp, Jun 2002) | ROYAL SUPRA 200 / 400 / 800 | not tabulated as ISO (nominal 200/400/800 in names); exposure **compensation** only: 3400 K + 80B **+1 2/3 stops**, 3200 K + 80A **+2 stops** | — | **200: 32 / 54 / 84; 400: 39 / 61 / 90; 800: 50 / 72 / 101** | 200 & 400 none 1/10,000–10 s; **800 none 1/10,000–1 s** | char. + spectral-sens. + spectral-dye p7–p10 |
| `e4029-2003_05.pdf` (10 pp, May 2003) | SUPRA 200 / 400 / 800 | as above | — | **200: 32 / 54 / 84; 400: 39 / 61 / 90; 800: 50 / 72 / 101** | 200 & 400 none 1/10,000–10 s; **800 none 1/10,000–1 s** | char. + spectral-sens. + spectral-dye p7–p10 |
| `e2519-2003_05.pdf` (10 pp, May 2003) | SUPRA 100 / 400 / 800 (discontinued) | prose "high speed (ISO 400)" for 400 | 3400 K + 80B **+1 2/3 stops** | **100: 27 / 49 / 78; 400: 36 / 58 / 87; 800: 50 / 72 / 101** | **100 & 800 none 1/10,000–1 s; 400 none 1/10,000–10 s** | char. + spectral-sens. + spectral-dye p7–p10 |
| `le1-2003_04.pdf` (10 pp, Apr 2003) | Law Enforcement LE100 / LE400 / LE800 | LE100 EI 100; LE400 EI 400 (push 1 → 800, push 2 → 1600); LE800 EI 800 (push 1 → 1600, push 2 → 3200) | 3400 K + 80B **+1 2/3**; 3200 K + 80A **+2** | **LE100: 28 / 50 / 79; LE400: 41 / 62 / 92; LE800: 53 / 75 / 104** | LE100 & LE400 none 1/10,000–10 s; LE800 shorter | char. + spectral-sens. + spectral-dye p5–p8. Process C-41, Status M |
| `e182-Pro_Films.pdf` (14 pp, Feb 1997) | Pro 100 / PRN, Pro 100T / PRT, Pro 400 / PPF, Pro 400 MC / PMC, Pro 1000 / PMZ | Pro 100 **100/21°**; Pro 400 & 400 MC **400/27°**; Pro 1000 **1000/31°**; **Pro 100T: tungsten 3200 K, no filter, 100/21°** | Pro 100 3400 K + 80B **32/16°**, 3200 K + 80A **25/15°**; Pro 400/400 MC 3400 K **125/22°**, 3200 K **100/21°**; Pro 1000 3400 K **320/26°**, 3200 K **250/25°**; Pro 100T + 81A (3400 K) **80/20°**, + 85B (daylight) **64/19°**, + 85B (flash) **64/19°** | 135: Pro 100 **36 / 56 / 86**; Pro 400 **42 / 63**; Pro 400 MC **37 / 58**; Pro 1000 **57 / 78** (4.4X / 8.8X / 17.8X); separate 120/220 and 4×5 tables | Pro 100/400/400 MC/1000 none 1/10,000–10 s; **Pro 100T none 1/1,000–5 s, then an EI-vs-time table** | char. + spectral-sens. + spectral-dye for **all five** films: p8 (PRN), p9 (PRT), p10 (PPF), p11 (PMC), p12 (PMZ). Status M with Log H Ref: PRN −0.99, PRT −0.86, PPF −1.53, PMC −1.20, PMZ −1.93. "colored-coupler masks" statement p7 |
| `e29-Pro_100T_PRT.pdf` (4 pp, Apr 1999) | Pro 100T / PRT | **tungsten 3200 K, no filter: EI 100/21° for 1/1,000–5 s** | see reciprocity table below | PGI table (p3) | **EI vs time (tungsten 3200 K, no filter): 1/1,000–5 s → 100/21°; 10 s → 80/20°; 30 s → 64/19°; 60 s → 50/18°; 120 s → 40/17°**; 3400 K + 81A 1/1,000–5 s → 80/20° | char. + spectral-sens. + spectral-dye p4. Base 120 **3.9-mil acetate**; sheets **7-mil ESTAR** |

### 1.23 KODAK VERICOLOR Slide Film 5072 and VERICOLOR Print Film 4111 — `e24-Vericolor.pdf` (4 pp, Dec 2002)
Print/duplicating films, **Process C-41** (not C-41B / C-41RA — older design needing a stabilizer) (p2, p4).
Slide Film exposure: **EI 8** with no filters over the light source (WRATTEN filter over the lens), or **EI 2** with a CP60R + CP50 pack over the light source (p2). Exposing source **tungsten-halogen 3200 K** + heat-absorbing glass + No. 2B / CP2B UV absorber (p2, p3). Starting filter pack **20M + 30Y** (p2, p3).
Base: Slide Film 5072 **5-mil (0.13 mm)**; Print Film 4111 **7-mil (0.18 mm) ESTAR Thick** with a retouching surface on both sides (p1, p3).
**No granularity, no resolving power, no MTF, no characteristic curves.**

### 1.24 Motion-picture / intermediate sheets present in the still list
| File | Film | Speed | rms granularity | Resolving power | Reciprocity | Process / densitometry | Base | Curves |
|---|---|---|---|---|---|---|---|---|
| `EASTMAN-DOUBLE-X-technical-information.pdf` (3 pp, Mar 2026, H-1-5222) | EASTMAN DOUBLE-X 5222 / 7222 | **Tungsten (3200 K) 200; Daylight 250** — both **for development to gamma 0.65**, no filter | **14**, net diffuse visual density 1.0, **48-micrometer aperture** | **32 lines/mm @ TOC 1.6:1**; **100 lines/mm @ TOC 1000:1** | **no filter/exposure adjustment 1/10,000 s to 1 second** | **D-96**, 21°C (70°F), control **gamma 0.65 to 0.70** calculated using **Status M densitometry (blue)** | (not stated in text) | **MTF** YES (p2, described); char. + spectral sensitivity YES |
| `EASTMAN-2366-technical-information.pdf` (3 pp) | KODAK Fine Grain Duplicating Positive 2366 | low-speed duplicating (no EI) | **9**, net diffuse visual density 1.0, **48-micrometre aperture** | **100 lines/mm (TOC 1.6:1)**; **200 lines/mm (TOC 1000:1)** | not stated | **D-96**, 70°F (21°C), control **gamma 1.2 to 1.6** | **clear ESTAR safety base**, anti-static layer + carnauba wax lubricant on back | characteristic curve YES; incorporates a **yellow** [filter] layer |
| `EASTMAN-2234-technical-information.pdf` (3 pp) | EASTMAN Fine Grain Duplicating Panchromatic Negative 2234 | (no EI) | **given as a curve only** — "read density on the left scale → characteristic curve → granularity curve → Granularity Sigma D scale on the right, **multiply by 1000** for the rms value"; modified measuring technique | not printed | not stated | **D-96**, 70°F (21°C), to recommended control gamma | **gray ESTAR Base (polyester)**, process-surviving anti-static layer on the back | **MTF**, CHARACTERISTIC, SPECTRAL SENSITIVITY, RMS GRANULARITY curves all present |
| `EASTMAN Color Internegative II Film 5272.pdf` (4 pp, Jun 1998, H-1-5272) | EASTMAN Color Internegative II 5272 / 7272 | balanced for printing with tungsten; exposure in curves at **tungsten 2850 K, 1/100 s** | **Less than 5**, net diffuse visual density 1.0, **48-micrometre aperture** | **80 lines/mm @ TOC 1.6:1**; **160 lines/mm @ TOC 1000:1** | **none 1/1000 to 1/10 s**; adjustments given for 1 s and 5 s | **ECN-2**; **Status M** | **clear acetate safety base with rem-jet** backing | char. + **MTF** + spectral-sens. + spectral-dye p3. "excellent image-structure characteristics and **color-correction masking**"; LAD via 7240 flashed to Status M R/G/B 1.10 |
| `KODAK EKTACHROME 64T Color Reversal Film 7280.pdf` (6 pp, May 2005, H-1-7280t) | EKTACHROME 64T 7280 | **Tungsten 3200 K, no filter: 64**; tungsten photoflood 3400 K, **no filter: 64** | **curve only** — "read with a microdensitometer (red, green, blue) using a **48-micrometre aperture**"; Sigma D × 1000 | not printed | **none 1/10,000 to 1 second** | **E-6** (cine machine only); **Status A** | (not stated) | char. + **MTF** p2; **rms granularity curves (R/G/B) + spectral sensitivity** p3; spectral-dye p4. Dyes normalized for **5000 K** viewing. `[FILTER]` daylight 5500 K + No. 85 → 40; tungsten 3000 K + 82B → 40 |
| `eastman 500t 5296 exr - Kodak.pdf` (6 pp, TI1664, reissued 6-92) | EASTMAN EXR 500T 5296 / 7296 | **Tungsten 3200 K, no filter: 500/28**; tungsten photoflood 3400 K, **no filter: 500/28**; ±150 K without filters | **curve only**, read with a microdensitometer (R, G, B) using a **48-micrometre aperture** | **ISO RPL 50 lines/mm (TOC 1.6:1)**; **ISO RP 100 lines/mm (TOC 1000:1)** | **none 1/1000 to 1 second** | (ECN-2 family) | **acetate safety base with rem-jet backing** | text-only data sheet, **no vector curves** (get_drawings = 0 paths). `[FILTER]` daylight 5500 K + No. 85 → 320/26; tungsten 3000 K + 82B → 320/26; white-flame arcs → 160/23. "emulsion contains a **colored-coupler mask**" |

### 1.25 Non-datasheet files
- `e73-1999_09.pdf` (2 pp, Sep 1999) — "Why a Color May Not Reproduce Correctly". Reference text; **no film-specific data**. Only actionable content: recommends WRATTEN No. 2B UV absorber over lens and light source for UV-fluorescent fabrics.
- `estimating_on-film_image_resolution_v8.pdf` (20 pp) — **not a Kodak publication**. Third-party analysis using the FujiFilm Resolving Power Equation; contains derived/estimated film-resolution figures and an explicit warning that "1000:1" targets inflate resolution by 25–40%, and that 1940-era Kodak data books quoted 45–70 lp/mm rising to 70–100 lp/mm and later up to 225 lp/mm with no stated method. **All numbers here are estimates — excluded from the extraction per the "only what is printed by the manufacturer" rule.**

---

## 2. (a) OUR STOCK NAMES NOW DOCUMENTED
Matched by exact generation / product code where possible.

| Our stock name | Documenting file(s) | Match quality | Single most valuable number gained |
|---|---|---|---|
| `KODAK_TMAX_100` | `f32-TMAX.pdf` p13 (+ `f32-TMAX-200109.pdf` p13) | **Exact** (T-MAX 100 Professional / TMX) | rms **8** @48 µm/12X + RP **63 (1.6:1) / 200 (1000:1) lines/mm** |
| `KODAK_TMAX_400` | `f32-TMAX.pdf` p13 | **Exact** (TMY) | rms **10** + RP **50 / 125 lines/mm** |
| `KODAK_TMAX_P3200` | `f32-TMAX.pdf` p23 | **Exact** (TMZ) | rms **18** + RP **40 / 125 lines/mm** |
| `KODAK_TRI_X_400TX` | `f9-Tri-X_Pan-199906.pdf` p1, p7 | **Exact** (TRI-X Pan / **TX**, ISO 400/27°) | rms **17** @48 µm/12X at CI 0.56 |
| `KODAK_TRI_X_320TXP` | `f9-Tri-X_Pan-199906.pdf` p1, p7 | **Exact** (TRI-X Pan Professional / **TXP**, ISO 320/26°) | rms **16** @48 µm/12X at CI 0.56 |
| `KODAK_PLUS_X_125` | `f4018-125PX-2007.pdf` p7 (primary) and `f8-Plus-X_Pan-199709.pdf` p7 | **Exact** — F-4018 is the 125PX product code; F-8 is the earlier PX/PXP coating | RP **50 (1.6:1) / 125 (1000:1) lines/mm** — the only Plus-X resolving-power pair in the set (with rms **10**) |
| `KODAK_EKTAPAN_100` | `f10-Ektapan.pdf` p1, p3 | **Exact** (EKTAPAN / EKP, ISO 100/21°) | rms **12** @48 µm/12X, CI 0.56 basis |
| `KODAK_VERICOLOR_III_160` | `e26-Vericolor_III.pdf` p1, p3, p4 | **Exact** (VERICOLOR III Professional, ISO 160/23°) | ISO **160/23°** unfiltered + **MTF curve present** (rare for a C-41 portrait film) |
| `KODAK_PROFOTO_100` | `e2e-Profoto_100.pdf` p1, p4 | **Exact** (PROFESSIONAL PROFOTO 100, EI 100) | Print Grain Index **43 / 65 / 94** |
| `KODAK_ULTRAMAX_400` | `E7019_en-Ultra_Max_400.pdf` p1, p3 | **Exact** (ULTRA MAX 400) | ISO **400/27°** + Print Grain Index **46** |
| `EASTMAN_EKTACHROME_7239` | `Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf` p1, p2 | **Exact** (7239, H-1-5239) | RP **40 (1.6:1) / 100 (1000:1) lines/mm** with rms **14** |
| `EASTMAN_EKTACHROME_5239` | same file (header carries **H-1-5239** / H-1-5247) | **Same emulsion family, different width** — data sheet is titled 7239 only; treat 5239 as inherited, flag | as above |
| `KODACHROME_64` | `e55-1996_12.pdf` p6 + `e55-2009_06.pdf` p4 (PKR, Professional) and `e88-1998_01.pdf` p6 / `e88-2002_03.pdf` p6 / `e88-2005_09.pdf` p4 (KR, consumer) | **Exact product name; two distinct products** — pick PKR (Professional) or KR (consumer) deliberately | rms **10** (PKR, and KR in 2002/2005) — but see the 1998-vs-2002 conflict in §7 |
| `KODAK_BW400CN` | `f4012-Portra_400BW.pdf` p1, p5 | **Predecessor only** — F-4012 documents **PORTRA 400BW** and names **BW400CN** as its replacement on p1. Same C-41 T-GRAIN B&W family; **not the same coating** | **RMS granularity 9** at 48-micrometer aperture — the only rms value for a C-41 chromogenic B&W film in the set |
| `KODAK_T400CN` | `f4012-Portra_400BW.pdf` | **Family predecessor/sibling only** — T400CN is not named in the file; flag as unverified | as above |
| `KODAK_VERICHROME_1952` | `f7-Verichrome-199611.pdf` | **Generation mismatch** — F-7 documents the 1996 VERICHROME Pan (EI 125); our stock is the 1952 generation | rms **9** + **CI 0.60** (use only if the 1952 emulsion is accepted as equivalent — it is not) |

### Stocks NOT documented by these 62 files
`EKTACHROME_64`, `EKTACHROME_160T`, `KODACHROME_1938`, `KODACHROME_TYPE_A_1938`,
`KODAK_EKTACHROME_100D_5285`, `KODAK_EKTAPRESS_PJ400`, `KODAK_EKTAR_100`,
`KODAK_GOLD_100`, `KODAK_GOLD_200`, `KODAK_ORTHO_X_SHEET_1952`,
`KODAK_PANATOMIC_X_SHEET_1952`, `KODAK_PORTRA_100T`, `KODAK_PORTRA_160`,
`KODAK_PORTRA_400`, `KODAK_PORTRA_800`, `KODAK_TRI_X_REVERSAL_200`,
`KODAK_TRI_X_SHEET_1952`, `KODAK_ULTRA_COLOR_100UC`, `KODAK_ULTRA_COLOR_400UC`,
`KODAK_ULTRAMAX_800`, and all 12 `KODAK_VISION*` / `EASTMAN` motion-picture stocks
(`VISION2_200T_5217`, `VISION2_250D_5205`, `VISION2_500T_5218`, `VISION3_50D_5203`,
`VISION3_200T_5213`, `VISION3_250D_5207`, `VISION3_500T_5219`, `VISION_200T_5274`,
`VISION_250D_5246`, `VISION_500T_5279`).

**Important negative findings:**
- **No PORTRA colour-negative datasheet exists in this list.** `e40`–`e44` are ROYAL GOLD 25/100/200/400/1000, **not** Portra. PORTRA 160NC / 400UC / 800 appear only as *replacement suggestions* in discontinuance notices (`e26-Vericolor_III.pdf` p1 → PORTRA 160NC; `le1-2003_04.pdf` p1 → PORTRA 400UC and PORTRA 800; `f4012` p5 → PORTRA 160NC filter reference; `E7019` p5 → publication E-4040) with **zero measured data**.
- `e4026` / `e4029` are ROYAL SUPRA / SUPRA, not Portra.
- **KODAK_GOLD_100/200** are only cross-referenced (E-7022 cited in `E7019` p5); no GOLD datasheet is present.
- **KODAK_EKTAR_100** and **KODAK_ULTRA_COLOR_100UC/400UC** have no datasheet here (ULTRA COLOR only as publication E-4035 in `E7019` p5). ROYAL GOLD 25 is described as the replacement for EKTAR 25 (`e40` p1) — not EKTAR 100.

---

## 3. (b) ALL RMS GRANULARITY NUMBERS FOUND
All values read at **density 1.0** with a **48-micrometre aperture**; "12X" = 12X magnification also stated.

| Value | Film | File | Page | Aperture wording |
|---|---|---|---|---|
| **Less than 5** | EASTMAN Color Internegative II 5272 / 7272 | `EASTMAN Color Internegative II Film 5272.pdf` | 2 | net diffuse visual D 1.0, 48-µm |
| **8** | KODAK T-MAX 100 Professional / TMX | `f32-TMAX.pdf`, `f32-TMAX-200109.pdf` | 13 | net diffuse D 1.00, 48-µm, 12X |
| **8** (extremely fine) | ELITE Chrome 100 / EB | `E7014e-Elitechrome_100.pdf`, `KODAK PROFESSIONAL ELITE Chrome 100 Film.pdf` | 3 | 48-µm aperture |
| **8** (extremely fine) | EKTACHROME E100G / E100GX | `e4024-2009.pdf` | 5 | 48-µm aperture |
| **8** (extremely fine) | Technical Pan Film | `p255.pdf`, `p255-2003_06.pdf` | 9 | net diffuse D 1.0, 48-µm, 12X |
| **8.7** (extremely fine) | EKTACHROME Duplicating EDUPE | `e2529-Ektachrome_EDUPE.pdf` | 6 | 48-µm, 12X |
| **9** | KODACHROME 25 Professional / PKM | `e55-1996_12.pdf` | 5 | gross diffuse visual D 1.0, 48-µm, 12X |
| **9** | KODACHROME 25 Professional / PKM | `e55-2009_06.pdf` | 3 | same ("Diffue" typo) |
| **9** | KODACHROME 25 Film / KM | `e88-2002_03.pdf` | 5 | same |
| **9** (extremely fine) | VERICHROME Pan | `f7-Verichrome-199611.pdf` | 3 | 48-µm, 12X |
| **9** (extremely fine) | PORTRA 400BW | `f4012-Portra_400BW.pdf` | 5 | net diffuse visual D 1.00, 48-micrometer |
| **9** | KODAK Fine Grain Duplicating Positive 2366 | `EASTMAN-2366-technical-information.pdf` | 1 | net diffuse visual D 1.0, 48-µm |
| **10** | KODAK T-MAX 400 Professional / TMY | `f32-TMAX.pdf`, `f32-TMAX-200109.pdf` | 13 | net diffuse D 1.00, 48-µm, 12X |
| **10** (extremely fine) | PLUS-X Pan / PX and PLUS-X Pan Prof. / PXP | `f8-Plus-X_Pan-199709.pdf` | 7 | D 1.0, 48-µm, 12X |
| **10** (extremely fine) | PROFESSIONAL PLUS-X 125 / 125PX | `f4018-125PX-2007.pdf` | 7 | net diffuse D 1.0, 48-µm, 12X |
| **10** | KODACHROME 64 Professional / PKR | `e55-1996_12.pdf` p6, `e55-2009_06.pdf` p4 | 6 / 4 | gross diffuse visual D 1.0, 48-µm, 12X |
| **10** | KODACHROME 64 Film / KR | `e88-2002_03.pdf` p6, `e88-2005_09.pdf` p4 | 6 / 4 | same |
| **11** (very fine) | EKTACHROME 100 Plus Professional / EPP | `e113-Ektachrome_100_plus_EPP.pdf` | 5 | 48-µm, 12X |
| **11** | ELITE Chrome Extra Color 100 / EBX | `e126e-Elitechrome_100ec.pdf` | 5 | gross diffuse visual 1.0, 48-µm, 12X |
| **11** | KODACHROME 25 Film / KM | `e88-1998_01.pdf` | 5 | gross diffuse visual D 1.0, 48-µm, 12X |
| **12** | ELITE Chrome 200 / ED | `e148e-Elite_chrome_200.pdf`, `KODAK PROFESSIONAL ELITE.pdf` | 4 | gross diffuse visual D 1.0, 48-µm, 12X |
| **12** (extremely fine) | EKTAPAN / EKP | `f10-Ektapan.pdf` | 3 | net diffuse D 1.0, 48-µm, 12X |
| **12** | KODAK Commercial Film | `f16-Commercial.pdf` | 2 | net diffuse D 1.0, 48-µm, 12X |
| **12** | KODACHROME 64 Film / KR | `e88-1998_01.pdf` | 6 | gross diffuse visual D 1.0, 48-µm, 12X |
| **14** | EASTMAN EKTACHROME (Daylight) 7239 | `Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf` | 2 | net diffuse visual D 1.0, 48-micrometer |
| **14** (very fine) | PLUS-X Pan Prof. / PXE and PXT | `f8-Plus-X_Pan-199709.pdf` | 7 | D 1.0, 48-µm, 12X |
| **14** | EASTMAN DOUBLE-X 5222 / 7222 | `EASTMAN-DOUBLE-X-technical-information.pdf` | 2 | net diffuse visual D 1.0, 48-micrometer |
| **16** (fine) | KODAK Professional Copy Film | `f17-Copy.pdf` | 3 | net diffuse D 1.0, 48-µm, 12X |
| **16** | KODACHROME 200 Professional / PKL | `e55-1996_12.pdf` p7, `e55-2009_06.pdf` p5 | 7 / 5 | gross diffuse visual D 1.0, 48-µm, 12X |
| **16** | KODACHROME 200 Film / KL | `e88-2002_03.pdf` p7, `e88-2005_09.pdf` p5 | 7 / 5 | same |
| **16** (fine) | TRI-X Pan Professional / TXP and TXT | `f9-Tri-X_Pan-199906.pdf` | 7 | net diffuse D 1.0, 48-µm, 12X |
| **17** (fine) | TRI-X Pan / TX | `f9-Tri-X_Pan-199906.pdf` | 7 | net diffuse D 1.0, 48-µm, 12X |
| **17** (fine) | EKTACHROME Prof. Infrared EIR | `ti2323-Ektachrome_EIR.pdf` | 7 | 48-µm, 12X |
| **18** | KODAK T-MAX P3200 Professional / TMZ | `f32-TMAX.pdf`, `f32-TMAX-200109.pdf` | 23 | net diffuse D 1.00, 48-µm, 12X |
| **18** (fine) | High Speed Infrared HIE / HSI | `f13-HIE-200006.pdf` | 6 | net diffuse D 1.0, 48-µm, 12X |
| **19** | ELITE Chrome 400 / EL | `e149-Elite_chrome_400.pdf` | 3 | gross diffuse visual D 1.0, 48-µm, 12X |
| **19** (fine) | EKTACHROME 320T Professional / EPJ | `e145-Ektachrome_320T_EPJ.pdf` | 3 | 48-µm, 12X |
| **19** | KODACHROME 200 Film / KL | `e88-1998_01.pdf` | 7 | gross diffuse visual D 1.0, 48-µm, 12X |
| **34** (Coarse, at EI 1600) | EKTACHROME P1600 Professional / EPH | `e147-Ektachrome_P1600_EPH.pdf` | 5 | 48-µm, 12X |
| **curve only (Sigma D × 1000)** | EASTMAN 2234; EKTACHROME 64T 7280; EASTMAN EXR 500T 5296 | `EASTMAN-2234-technical-information.pdf` p2; `KODAK EKTACHROME 64T…7280.pdf` p2–3; `eastman 500t 5296 exr - Kodak.pdf` p5 | — | microdensitometer, 48-µm aperture (R,G,B) |

**24 distinct rms values; 41 film/edition entries. Every one uses a 48 µm aperture; every B&W and reversal still sheet also states 12X magnification.**

---

## 4. (c) ALL RESOLVING-POWER PAIRS FOUND
Only **10 films** in the whole 62-file set print resolving power. Every one is an ISO-6328-like pair.

| Film | TOC **1.6:1** | TOC **1000:1** | File | Page |
|---|---|---|---|---|
| KODAK T-MAX 100 Professional / TMX | **63 lines/mm** | **200 lines/mm** | `f32-TMAX.pdf` / `f32-TMAX-200109.pdf` | 13 |
| KODAK T-MAX 400 Professional / TMY | **50 lines/mm** | **125 lines/mm** | `f32-TMAX.pdf` / `f32-TMAX-200109.pdf` | 13 |
| KODAK T-MAX P3200 Professional / TMZ | **40 lines/mm** | **125 lines/mm** | `f32-TMAX.pdf` / `f32-TMAX-200109.pdf` | 23 |
| KODAK PROFESSIONAL PLUS-X 125 / 125PX | **50 lines/mm** (ISO RPL) | **125 lines/mm** | `f4018-125PX-2007.pdf` | 7 |
| EASTMAN EKTACHROME (Daylight) 7239 | **40 lines/mm** | **100 lines/mm** | `Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf` | 2 |
| EASTMAN DOUBLE-X 5222 / 7222 | **32 lines/mm** | **100 lines/mm** | `EASTMAN-DOUBLE-X-technical-information.pdf` | 2 |
| KODAK Fine Grain Duplicating Positive 2366 | **100 lines/mm** | **200 lines/mm** | `EASTMAN-2366-technical-information.pdf` | 1 |
| EASTMAN Color Internegative II 5272 / 7272 | **80 lines/mm** | **160 lines/mm** | `EASTMAN Color Internegative II Film 5272.pdf` | 2 |
| EASTMAN EXR 500T 5296 / 7296 | **50 lines/mm** (ISO RPL) | **100 lines/mm** (ISO RP) | `eastman 500t 5296 exr - Kodak.pdf` | 5 |
| *(reference only, not manufacturer data)* | third-party derived figures | — | `estimating_on-film_image_resolution_v8.pdf` | 4, 13–15 |

Method wording in every case: "Determined according to a method similar to the one described in
ISO 6328[-1982], Photography—[Photographic Materials—]Determination of ISO Resolving Power."

---

## 5. (d) VECTOR CURVE SETS (`page.get_drawings()` paths with ≥25 items)
Excluded from this list: page-1 letterhead / logo artwork (typically 12–23 paths of ~112–196 items
on every E-/F-series cover page) and the ~28-item back-cover footer mark. Kept: pages whose text
also names a plotted quantity, so the paths are the curve geometry and are exactly extractable.

| File | Page | Paths ≥25 items | Max items in one path | Quantities plotted |
|---|---|---|---|---|
| `E7014e-Elitechrome_100.pdf` **and** `KODAK PROFESSIONAL ELITE Chrome 100 Film.pdf` | 3 | 4 | 134 | Characteristic curves (E-6, daylight 1/100 s, Status A); Spectral-sensitivity |
| " | 4 | 22 | 194 | Modulation-transfer; Spectral-dye-density (5000 K normalized) |
| `E7019_en-Ultra_Max_400.pdf` | 4 | 1 | 126 | Characteristic (R/G/B); Spectral-sensitivity; Spectral-dye-density |
| `EASTMAN Color Internegative II Film 5272.pdf` | 1 | 12 | 194 | Characteristic (cover-page thumbnail) |
| " | 3 | 5 | 219 | Characteristic; Modulation-transfer; Spectral-sensitivity; Spectral-dye-density |
| `EASTMAN-DOUBLE-X-technical-information.pdf` | 1 | 1 | 166 | (page art; MTF/char/spectral described p2) |
| `KODAK EKTACHROME 64T…7280.pdf` | 2 | 5 | 148 | Sensitometric/characteristic; Modulation-transfer |
| " | 3 | 6 | 194 | **Diffuse rms granularity curves (Granularity SIGMA D × 1000, R/G/B)**; Spectral-sensitivity |
| " | 4 | 5 | 150 | Spectral-dye-density |
| `KODAK PROFESSIONAL ELITE.pdf` **and** `e148e-Elite_chrome_200.pdf` | 4 | 7 | 149 | Characteristic; Modulation-transfer; Spectral-sensitivity |
| " | 5 | 4 | 219 | Spectral-dye-density |
| `Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf` | 3 | 8 | **1066** | Sensitometric (VNF-1, Status A); Modulation-transfer; Spectral-sensitivity; Spectral-dye-density (5400 K) |
| `e113-Ektachrome_100_plus_EPP.pdf` | 5 | 8 | 240 | Characteristic; Modulation-transfer; Spectral-sensitivity; Spectral-dye-density |
| `e126e-Elitechrome_100ec.pdf` | 5 | 8 | 254 | Characteristic; Modulation-transfer; Spectral-sensitivity; Spectral-dye-density |
| `e145-Ektachrome_320T_EPJ.pdf` | 3 | 2 | 248 | Characteristic (tungsten 1/50 s, Status A); Modulation-transfer |
| " | 4 | 18 | 239 | Spectral-sensitivity; Spectral-dye-density |
| `e147-Ektachrome_P1600_EPH.pdf` | 5 | 7 | 275 | Characteristic; Modulation-transfer; Spectral-sensitivity; Spectral-dye-density |
| `e149-Elite_chrome_400.pdf` | 3 | 5 | 135 | Characteristic; Modulation-transfer |
| " | 4 | 4 | 239 | Spectral-sensitivity; Spectral-dye-density |
| `e182-Pro_Films.pdf` | 8 | 6 | 227 | Characteristic + Spectral-sensitivity + Spectral-dye (Pro 100 / PRN) |
| " | 9 | 8 | 251 | same (Pro 100T / PRT) |
| " | 10 | 8 | **512** | same (Pro 400 / PPF) |
| " | 11 | 6 | 214 | same (Pro 400 MC / PMC) |
| " | 12 | 6 | 227 | same (Pro 1000 / PMZ) |
| `e2328-GA100.pdf` | 3 | 4 | 225 | Characteristic; Spectral-sensitivity |
| " | 4 | 15 | 216 | Spectral-dye-density |
| `e2509-2000_01.pdf` | 4 | 6 | 309 | Characteristic; Spectral-sensitivity; Spectral-dye-density |
| `e2519-2003_05.pdf` | 7, 8 | 6, 8 | 245, 249 | Characteristic; Spectral-sensitivity; Spectral-dye (SUPRA 100, 400) |
| " | 9 | 7 | 246 | Characteristic (SUPRA 800) |
| " | 10 | 6 | 268 | Spectral-sensitivity; Spectral-dye |
| `e2529-Ektachrome_EDUPE.pdf` | 6 | 4 | 247 | Characteristic |
| " | 7 | 8 | 280 | Modulation-transfer; Spectral-sensitivity; Spectral-dye-density |
| `e26-Vericolor_III.pdf` | 4 | 8 | 327 | Characteristic (C-41, daylight 1/1000 s, Status M); **Modulation-transfer**; Spectral-sensitivity; Spectral-dye-density |
| `e29-Pro_100T_PRT.pdf` | 4 | 9 | 250 | Characteristic; Spectral-sensitivity; Spectral-dye-density |
| `e2e-Profoto_100.pdf` | 3 | 8 | 224 | Characteristic (C-41, daylight, Status M); Spectral-sensitivity; Spectral-dye-density |
| `e40-1996_12.pdf` | 3 | 12 | 208 | Characteristic; Spectral-sensitivity; Spectral-dye-density |
| `e4024-2009.pdf` | 5 | 13 | 177 | Characteristic (E100G and E100GX); Spectral-sensitivity; Spectral-dye-density |
| " | 6 | 16 | 103 | Modulation-transfer |
| `e4026-2002_06.pdf` | 7, 8 | 7, 9 | 339, 340 | Characteristic; Spectral-sensitivity; Spectral-dye (ROYAL SUPRA 200, 400) |
| " | 9 | 10 | 246 | Characteristic; Spectral-sensitivity (ROYAL SUPRA 800) |
| " | 10 | 3 | 270 | Spectral-dye |
| `e4029-2003_05.pdf` | 7, 8 | 7, 9 | 329, 328 | Characteristic; Spectral-sensitivity; Spectral-dye (SUPRA 200, 400) |
| " | 9 | 8 | 252 | Characteristic; Spectral-sensitivity (SUPRA 800) |
| " | 10 | 3 | 266 | Spectral-dye |
| `e4039-Elite.pdf` | 7 | 2 | 122 | Characteristic; **MTF**; Spectral-sensitivity; Spectral-dye (ELITE COLOR 200) |
| " | 8 | 3 | 126 | Characteristic EI 400 and EI 800 push 1; **MTF**; Spectral-sensitivity; Spectral-dye (400) |
| `e41-1998_02.pdf` | 3 | 7 | 260 | Characteristic; Spectral-sensitivity; Spectral-dye |
| `e42-1998_02.pdf` | 3 | 6 | 227 | Characteristic; Spectral-sensitivity; Spectral-dye |
| `e43-1998_02.pdf` | 4 | 4 | 338 | Characteristic; Spectral-sensitivity; Spectral-dye |
| `e44-1998_02.pdf` | 4 | 6 | 288 | Characteristic; Spectral-sensitivity; Spectral-dye |
| `e55-1996_12.pdf` | 5, 6, 7 | 11, 11, 11 | 112, 160, 144 | Characteristic + **Modulation-transfer** + Spectral-sensitivity + Spectral-dye for PKM, PKR, PKL |
| `e55-2009_06.pdf` | 3, 4, 5 | 8, 5, 6 | 40, 48, 51 | same four sets for PKM, PKR, PKL (curves re-drawn as many short paths) |
| `e7006-2002_03.pdf` | 3 | 5 | 308 | Characteristic; Spectral-sensitivity |
| " | 4 | 15 | 339 | Spectral-dye |
| `e7013-HD400.pdf` | 5 | 9 | 132 | Characteristic; Spectral-dye |
| `e7017-HD200.pdf` | 5 | 7 | 340 | Characteristic; Spectral-sensitivity; Spectral-dye |
| `e88-1998_01.pdf` | 5, 6, 7 | 6, 7, 7 | 207, 200, 248 | Characteristic + **MTF** + Spectral-sensitivity + Spectral-dye for KM, KR, KL |
| `e88-2002_03.pdf` | 5, 6, 7 | 6, 7, 7 | 207, 200, 248 | same for KM, KR, KL |
| `e88-2005_09.pdf` | 4, 5 | 7, 7 | 200, 248 | same for KR, KL |
| `f10-Ektapan.pdf` | 3 | 3 | 199 | Characteristic (HC-110 B 68°F 5/7/8/12 min); **Contrast-index curves** |
| " | 2 | 3 | 144 | **Reciprocity nomographs** (calculated vs adjusted exposure time) |
| `f11-Duplicating_SO-132-200105.pdf` | 3 | 8 | 225 | Characteristic |
| `f13-HIE-200006.pdf` | 2 | 2 | 204 | **Infrared filter transmittance / diffuse density curves (87, 87C, 89B)** |
| " | 6 | 9 | 117 | Characteristic (HIE and HSI); **Contrast-index curves** |
| " | 7 | 5 | 124 | **Modulation-transfer**; Spectral-sensitivity |
| `f16-Commercial.pdf` | 2 | 3 | 154 | **Reciprocity nomographs**; Characteristic |
| " | 3 | 5 | 210 | Characteristic; **Contrast-index curves** |
| `f17-Copy.pdf` | 3 | 5 | 236 | Characteristic; **Reciprocity curve**; Spectral-sensitivity |
| `f32-TMAX.pdf` / `f32-TMAX-200109.pdf` | 5 | 9 | 152 / 150 | (processing-table rules) |
| " | 14 | 6 | 206 / 207 | **Modulation-transfer** (TMX and TMY); Spectral-sensitivity (TMX, TMY) |
| " | 15 | 8 | 235 / 236 | **Characteristic curves TMX** ×4 developer conditions |
| " | 16 | 7 | 223 | **Characteristic curves TMY** ×4 developer conditions |
| " | 19 | 9 | 151 / 155 | (processing tables) |
| " | 24 | 6 | 187 | **Characteristic curves TMZ**; **Modulation-transfer TMZ**; Spectral-sensitivity TMZ |
| " | 25 | 6 | 222 / 223 | **Characteristic curves TMZ** (T-MAX RS large tank; DURAFLO RT; rotary-tube) |
| `f4012-Portra_400BW.pdf` | 2, 3 | 3, 3 | 204, 207 | **Reciprocity / exposure-table graphics** |
| " | 6 | 7 | 233 | Characteristic (Log H Ref −1.44); Spectral-sensitivity; Spectral-dye-density ×2 |
| `f4018-125PX-2007.pdf` | 2 | 6 | 143 | **Reciprocity nomograph** |
| " | 7 | 6 | 134 | **Characteristic curves** (D-76 and T-MAX Dev, 20°C) |
| " | 8 | 4 | 71 | **Contrast-index curves** ×4 (small/large tank, D-76/T-MAX/HC-110/MICRODOL-X/XTOL) |
| " | 9 | 2 | 202 | **Spectral-sensitivity curve** |
| `f7-Verichrome-199611.pdf` | 3 | 6 | 136 | Characteristic; **Contrast-index curves** |
| `f8-Plus-X_Pan-199709.pdf` | 2 | 3 | 144 | **Reciprocity nomographs** |
| " | 8 | 6 | 237 | **Characteristic curves PX**; **Contrast-index curves PX** |
| " | 10 | 3 | 232 | Characteristic; Contrast-index; **Modulation-transfer** (PXP) |
| " | 11 | 4 | 162 | Characteristic; Contrast-index; **Modulation-transfer** (PXE/PXT) |
| `f9-Tri-X_Pan-199906.pdf` | 2 | 3 | 143 | **Reciprocity nomographs** |
| " | 8 | 6 | 226 | **Characteristic curves TX**; **Contrast-index curves TX** |
| " | 9 | 3 | 221 | **Modulation-transfer TX**; Spectral-sensitivity TX |
| " | 10 | 3 | 217 | **Characteristic + Contrast-index curves TXP** |
| " | 11 | 3 | 204 | **Characteristic + Contrast-index curves TXT** |
| `le1-2003_04.pdf` | 5, 6, 7, 8 | 5, 9, 6, 6 | 274, 244, 219, 222 | Characteristic; Spectral-sensitivity; Spectral-dye for LE100 / LE400 / LE800 |
| `p255.pdf` / `p255-2003_06.pdf` | 6 | 13 | 253 / 250 | Characteristic curves (developer series) |
| " | 9 | 3 | 264 / 268 | **Modulation-transfer**; Spectral-sensitivity |
| " | 10 | 19 | 217 | **Characteristic curves** (TECHNIDOL Dil F 5/7/9/11 min with CI labels 0.50/0.60/0.65/0.70) |
| " | 11 | 29 / 30 | 255 | **Characteristic curves + CI/EI annotated series** |
| `ti2323-Ektachrome_EIR.pdf` | 2 | 2 | 88 | (colour-reproduction figure) |
| " | 7 | 6 | 135 | Characteristic; Spectral-sensitivity |
| " | 8 | 21 | 193 | **Modulation-transfer**; Spectral-dye-density |
| `e2e-Profoto_100.pdf` | 1 | 23 | 112 | (cover art only) |
| `EASTMAN-2234-technical-information.pdf` | 1 | 1 | 166 | (page art; RMS-granularity/MTF/char./spectral curves described on p2) |

**File with the single richest vector path: `Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf` p3 — one path with 1066 items.** Next: `e182-Pro_Films.pdf` p10 (512 items).
**`eastman 500t 5296 exr - Kodak.pdf` is the only file in the set with ZERO drawing paths** (curves are absent, data are text-only).

---

## 6. (e) DOCUMENTED FILMS WE DO **NOT** CARRY
Colour reversal: EKTACHROME 100 Plus / EPP; ELITE Chrome 100 / EB; ELITE Chrome Extra Color 100 / EBX; ELITE Chrome 200 / ED; ELITE Chrome 400 / EL; EKTACHROME 320T / EPJ; EKTACHROME P1600 / EPH; EKTACHROME E100G; EKTACHROME E100GX; EKTACHROME Duplicating EDUPE; EKTACHROME Professional Infrared EIR; EKTACHROME 64T 7280; KODACHROME 25 (PKM and KM); KODACHROME 200 (PKL and KL).
Colour negative: ROYAL GOLD 25 / 100 / 200 / 400 / 1000; ROYAL GOLD 200 / RB; Bright Sun / GA (ISO 100); High Definition 200 / 3992 / HD2; High Definition 400; ELITE COLOR 200; ELITE COLOR 400; ROYAL SUPRA 200 / 400 / 800; SUPRA 100 / 200 / 400 / 800; Law Enforcement LE100 / LE400 / LE800; Pro 100 / PRN; Pro 100T / PRT; Pro 400 / PPF; Pro 400 MC / PMC; Pro 1000 / PMZ; PROFESSIONAL PORTRA 400BW; VERICOLOR Slide Film 5072; VERICOLOR Print Film 4111; EASTMAN Color Internegative II 5272 / 7272; EASTMAN EXR 500T 5296 / 7296.
Black-and-white: TRI-X Pan Professional TXT; PLUS-X Pan Professional PXE and PXT; VERICHROME Pan (1996 coating); KODAK Commercial Film; KODAK Professional Copy Film; PROFESSIONAL B/W Duplicating Film SO-132; High Speed Infrared HIE and HSI; PROFESSIONAL Technical Pan Film; T-MAX 100 Professional Plate; EASTMAN DOUBLE-X 5222 / 7222; EASTMAN Fine Grain Duplicating Panchromatic Negative 2234; KODAK Fine Grain Duplicating Positive 2366.

---

## 7. (f) FILES WITH NOTHING USEFUL / LIMITED VALUE
| File | Why |
|---|---|
| `e73-1999_09.pdf` (2 pp) | Reference essay "Why a Color May Not Reproduce Correctly". **No film, no numbers, no curves.** |
| `estimating_on-film_image_resolution_v8.pdf` (20 pp) | **Not a Kodak publication.** Third-party analysis; all resolution numbers are computed/estimated via the FujiFilm RPE, explicitly warns 1000:1 targets inflate results 25–40%. Excluded from extraction. |
| `e55-2003_08.pdf` (4 pp) | KODACHROME 25/64/200 Professional, Aug 2003 — **exposure indexes only; no IMAGE STRUCTURE section, no granularity, no curves.** Superseded by e55-2009_06 for data. |
| `f11-Duplicating_SO-132-200105.pdf` (4 pp) | **No speed, no granularity, no resolving power, no MTF.** Only a characteristic curve and a trial-exposure statement. |
| `e24-Vericolor.pdf` (4 pp) | No granularity, no resolving power, no MTF, no characteristic curves; only EI 8 / EI 2 print-exposure practice and base thickness. |
| `KODAK PROFESSIONAL ELITE Chrome 100 Film.pdf` | **Byte-identical text to `E7014e-Elitechrome_100.pdf`** — duplicate, no new information. |
| `KODAK PROFESSIONAL ELITE.pdf` | **Byte-identical text to `e148e-Elite_chrome_200.pdf`** — duplicate. |
| `f32-TMAX-200109.pdf` | Numerically identical to `f32-TMAX.pdf`; earlier edition, lacks the discontinuance notice. |
| `p255-2003_06.pdf` | Numerically identical to `p255.pdf`; differs in title/notice/CAT numbers only. |
| `EASTMAN-2234-technical-information.pdf` | Useful base/gamma/process data, but **granularity is a curve only and resolving power is absent** — no scalar image-structure values. |

---

## 8. (g) CONFLICTS BETWEEN FILES ON THE SAME FILM
1. **KODACHROME 25 / 64 / 200 Films (consumer KM / KR / KL) — rms granularity.**
   `e88-1998_01.pdf` p5/p6/p7 prints **11 / 12 / 19**.
   `e88-2002_03.pdf` p5/p6/p7 prints **9 / 10 / 16** for the same three products.
   `e88-2005_09.pdf` p4/p5 prints **10 / 16** (25 discontinued).
   All use the same measurement statement (gross diffuse visual density 1.0, 48 µm, 12X) and the
   same curve figure numbers (F002_0486AC etc.). The 2002 revision silently adopted the
   **Professional** (PKM/PKR/PKL) figures already published in `e55-1996_12.pdf` (9 / 10 / 16).
   → **Recommendation: for pre-2002 KODACHROME consumer stock use 11 / 12 / 19; the 9 / 10 / 16
   values are the Professional-film numbers.**
2. **KODAK ROYAL GOLD 400 — Print Grain Index.**
   `e43-1998_02.pdf` prints **41** (135, 4×6 in, 4.4X). `e2509-2000_01.pdf` prints **39** for
   ROYAL GOLD 400 at the same size/magnification. The 2000 sheet also adds the 3200 K index
   (100/21°) absent from the 1998 sheet. Different coating generation is likely but not stated.
3. **KODAK PLUS-X — granularity across editions/coatings.**
   `f8-Plus-X_Pan-199709.pdf` p7: PX/PXP **10**, PXE/PXT **14**.
   `f4018-125PX-2007.pdf` p7: 125PX **10** *and* the resolving-power pair 50/125 that F-8 omits.
   Not strictly a conflict (F-4018 is a new coating facility per its own p1 notice) but the two
   sheets must not be merged: **F-8 development times do not apply to F-4018 film.**
4. **SUPRA family PGI, 100/400/800 vs 200/400/800.**
   `e2519-2003_05.pdf` (SUPRA 100/400/800) gives 400 Film **36 / 58 / 87**, while
   `e4029-2003_05.pdf` (SUPRA 200/400/800) gives 400 Film **39 / 61 / 90**. Same nominal
   "SUPRA 400" name, same publication month (May 2003), different numbers — these are two
   different SUPRA generations sharing a product name.
5. **SUPRA 400 reciprocity limit.** `e2519` p4: SUPRA 400 no adjustment 1/10,000–**10 s**, but
   SUPRA 100 and 800 only to **1 s**. `e4029` p4: SUPRA 200 and 400 to **10 s**, 800 to **1 s**.
   Consistent for 400, but confirms 100 ≠ 200 in that family.
6. **T-MAX P3200 heading inconsistency (not a value conflict).** `f32-TMAX.pdf` p24 heads the
   curve page "KODAK T-MAX **3200** Professional Film / TMZ" while every other reference in both
   editions says **P3200**. Same film, same data.

---

## 9. PARAMETER CLASSES ABSENT ACROSS ALL 62 FILES
- **Dmin and Dmax as printed scalars**: essentially absent. Only exception — `f13-HIE-200006.pdf`
  p4/p5, which pairs contrast index with D-max (0.65↔1.50, 0.80↔1.76, 0.91↔2.00, 1.03↔2.36,
  1.15↔2.44). Colour films give only *aim* red-channel densities for exposure judging (gray card,
  gray-scale step, forehead), which are **not** Dmin/Dmax. `p255` p9 gives the ESTAR-AH base's
  built-in **0.1 neutral density**, the only base-density figure in the set.
- **Balance colour temperature in Kelvin as a film specification**: almost never stated as a
  number for still films. Only the motion-picture-derived sheets state it explicitly
  (`7239`: balanced for daylight, projection **5400 K**; `7280` and `5296`: balanced for
  **tungsten 3200 K**; `5272`: balanced for tungsten printing). Still-film sheets say only
  "daylight-balanced" / "tungsten-balanced" and give 3200 K / 3400 K as *filtered* light-source
  rows. Reversal sheets give the **viewing/projection** illuminant (5000 K) and the
  spectral-dye normalization illuminant (5000 K for E-6, **3200 K** for K-14) — not a taking balance.
- **Gamma as a single number**: absent for all still camera films. Only the motion-picture sheets
  print a control gamma (`DOUBLE-X` 0.65–0.70, `2366` 1.2–1.6, `2234` "recommended control
  gamma"). Still B&W sheets use **contrast index** instead (0.56 for Tri-X / Plus-X / Ektapan,
  0.60 for Verichrome Pan, 0.48–3.50 range for Technical Pan) and T-MAX gives only a relative
  development-time multiplier matrix, no CI number at all.
- **Resolving power**: absent for **52 of 62** files. Notably missing for TRI-X (all codes),
  PLUS-X in the F-8 edition, EKTAPAN, VERICHROME Pan, Technical Pan, HIE, Commercial, Copy Film,
  and every KODACHROME, EKTACHROME, ELITE Chrome, ROYAL GOLD, SUPRA, HD, GA, PROFOTO, ULTRA MAX,
  VERICOLOR and PORTRA 400BW sheet.
- **rms granularity for Process C-41 colour negative**: systematically **replaced by Print Grain
  Index** from ~1996 onward. Every C-41 sheet here states verbatim "It replaces rms granularity
  and has a different scale which cannot be compared to rms granularity". The **only** C-41 film
  in the set that still prints an rms number is PORTRA 400BW (`f4012` p5, rms 9) — because it is
  a chromogenic *black-and-white* film.
- **MTF**: absent for TRI-X TXP/TXT, EKTAPAN, VERICHROME Pan, PLUS-X 125PX (F-4018),
  Commercial, Copy Film, SO-132, PROFOTO 100, ULTRA MAX 400, PORTRA 400BW, and all ROYAL GOLD /
  SUPRA / HD / GA / LE / Pro Films sheets.
- **Base material and thickness**: absent for ULTRA MAX 400, PROFOTO 100, all ROYAL GOLD sheets
  except via CAT tables, DOUBLE-X, and EKTACHROME 64T 7280.
- **Numeric long-exposure/reciprocity data beyond a "no adjustment" window**: absent for most
  colour films. Only `e182`/`e29-Pro_100T_PRT` (EI vs time to 120 s), `e113` (CC025R at 1 s),
  `e149` (CC05R at 1 s), `e145` (CC + 1/3 stop at 1 s), `e4024` (CC10R at 120 s),
  `ti2323` (CC20B at 1/10 s), and the three KODACHROME sheets (CC05R / CC10Y at 1/10 s) give
  actual filtration. All B&W sheets give full exposure+development reciprocity tables.
- **Densitometry status**: consistently present — **Status M** for C-41/ECN-2 negative,
  **Status A** for E-6/K-14 reversal, **diffuse visual** for B&W, **E.N.D.** for KODACHROME
  spectral sensitivity, **VNF-1/Status A** for 7239.
- **Orange-mask statements**: present only as `dye-masking color couplers` (`e26` p1),
  `Built-in dye-masking` (`e40` p1), `colored-coupler masks` (`e182` p7, `e29` p3, `le1` p4),
  `color-correction masking` (`5272` p1), `colored-coupler mask` (`5296` p1). **No sheet prints a
  numeric mask density.** `f4012` p3 explicitly documents the *absence* of a mask ("much lower
  D-min … film base will appear very light brown").
