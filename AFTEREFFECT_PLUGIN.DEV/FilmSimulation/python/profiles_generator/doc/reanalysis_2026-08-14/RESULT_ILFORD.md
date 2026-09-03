# ILFORD / KENTMERE (HARMAN technology) technical information sheets — extraction
Scope: the 19 PDFs listed in `/tmp/list_ILFORD_KENTMERE.txt`, all read in full as true text
(PyMuPDF `fitz`, per-page `get_text()` + `get_drawings()`).
Paths below are relative to `PDF/PROFILES/`.

**Global warning — the "Plus" films are NOT the films we hold.**
`PAN F PLUS`, `FP4 PLUS`, `HP5 PLUS` are modern (1990s-onward) re-engineered emulsions.
They are *not* 1930s–70s Pan F, FP4 or HP4. Nothing in this corpus documents
`ILFORD_PAN_F`, `ILFORD_FP4`, `ILFORD_HP4`, `ILFORD_HP3` or `ILFORD_HPS`.

**Global negative result (verified by regex sweep of the full 236 kB text dump):**
no occurrence of `RMS`, `granularity`, `lines/mm`, `l/mm`, `lp/mm`, `resolving power`,
`MTF`, `modulation`, `1000:1`, `1.6:1`, `Dmin`, `Dmax`, `base+fog`, `contrast index`
anywhere in any of the 19 files. Only two films carry any numeric contrast figure at all
(DELTA 400 and ORTHO PLUS, both as Ḡ / "Gbar" average gradient).

---

## 1. ILFORD DELTA 3200 PROFESSIONAL
Source: `ILFORD/DELTA 3200 technical data sheet F25.pdf` (Jun 2025, 7 pp) — documents our `ILFORD_DELTA_3200`.

| Parameter | Value | Source |
|---|---|---|
| Nominal rating | EI 3200/36° (design point, extended development) | p1 |
| **True ISO speed** | **ISO 1000/31° (1000 ASA, 31 DIN) to daylight**, measured in ID-11, 20 °C/68 °F, intermittent agitation, spiral tank | p1 |
| Usable EI range | EI 400/27 – 6400/39 good; recommended 1600/33 – 6400/39; up to EI 25000/45 with test exposures | p1 |
| Tungsten figure | none printed (wedge spectrogram is *to* tungsten, not a separate speed) | — |
| Spectral sensitivity | "Wedge spectrogram to tungsten light (**2856 K**)" — raster image only (542×260 px), no nm axis figures in text, no spectral-sensitivity curve | p1 |
| Characteristic curves | present: DD-X 1+4 at 7, 9, 12, 16 min / 20 °C; MICROPHEN stock at 7, 9, 12, 16 min / 20 °C. Raster images. | p5 |
| Contrast–time graphs | present: DD-X 1+4 @20 °C; MICROPHEN stock @20 °C. **No numeric gamma / CI axis values in text.** | p5 |
| Dmin / Dmax | not printed | — |
| RMS granularity | **NOT PRINTED** | — |
| Resolving power / test contrast | **NOT PRINTED** | — |
| MTF | **NOT PRINTED** | — |
| Reciprocity | no correction ½ s – 1/10 000 s; >1 s use graph; **Ta = Tm^1.33** (Ta adjusted, Tm metered) | p2 |
| Filter factors | qualitative only; TTL cameras can under-expose by up to 1½ stops with deep red/orange | p2 |
| Developer-dependent SPEED | max film speed (to EI 25000/45) only with ILFOTEC DD-X or MICROPHEN stock; PERCEPTOL tabulated only to EI 3200 | p2, p3, p4 |
| Dev times 20 °C (min) EI 400/800/1600/3200/6400/12500 | DD-X 1+4: 6/7/8/9½/12½/17 · ILFOSOL 3 1+9: 6/7½/10/11/18/– · 1+14: 11/13/15½/17/23/– · ILFOTEC HC 1+15: –/–/5/8/13/– · 1+31: 6/7½/9/14½/–/– · LC29 1+9: –/–/5/8/13/– · 1+19: 6/7½/9/14½/–/– · ID-11 stock: 7/8/9½/10½/13/17 · MICROPHEN stock: 6/7/8/9/12/16½ · PERCEPTOL stock: 11/13/15/18/–/– | p3 |
| Dev times 24 °C (min) same EI order | DD-X 1+4: –/5/6/7/9/12 · ILFOSOL 3 1+9: 5½/7/8/9/15½/– · 1+14: 7/8/10/11/19/– · HC 1+15: –/–/–/5½/8½/– · HC 1+31: 5/6/7/10½/–/– · LC29 1+9: –/–/–/5½/8½/– · 1+19: 5/6/7/10½/–/– · ID-11 stock: 6/7/8/9/11/13½ · MICROPHEN stock: –/5/6/7/9½/13½ · PERCEPTOL stock: 9½/10½/12/15½/–/– · ILFOTEC DD 1+4: 8/8½/9½/10½/13½/19 · RT RAPID 1+1+2 (sec, 26 °C): 54/65/73/84/104/– · 1+1+5: 95/108/120/153/176/– | p4 |
| EI 25000/45 | DD-X 1+4: 25 min @20 °C, 17 min @24 °C · MICROPHEN stock: 22 min @20 °C, 17½ min @24 °C | p4 |
| Base | 35 mm: 0.125 mm / 5-mil acetate. (120 and bulk stated as same product; no separate roll-film base thickness given in this revision) | p1 |
| Safelight | total darkness | p6 |

---

## 2. ILFORD DELTA 100 PROFESSIONAL  *(not currently carried)*
Sources: `ILFORD/Delta_100-200209.pdf` (Sep 2002, 6 pp) and `ILFORD/Delta-100_201811.pdf` (Nov 2018, 6 pp).

| Parameter | Value | Source |
|---|---|---|
| ISO / DIN | ISO 100/21° (2002: "100 ASA, 21 DIN, EI 100/21") to daylight; best at EI 100/21, usable EI 50/18 – 200/24 | 200209 p1; 201811 p1 |
| Spectral sensitivity | wedge spectrogram to tungsten (**2850 K**); axis label "Wavelength nm", ordinate "Sensitivity". 2002 sheet: **vector curve** (7 bezier segments, bbox 325,281–482,334 pt). 2018 sheet: raster. No nm numbers as text. | 200209 p1; 201811 p1 |
| Characteristic curve | present: 120 roll film, ID-11 stock, 8½ min @20 °C, intermittent agitation; "also representative of 35 mm and sheet". 2002 = raster (1192×792 px, bbox 294,295–509,438). No gamma value. | 200209 p4; 201811 p4 |
| RMS granularity / resolving power / MTF / Dmin / Dmax | **NOT PRINTED** | — |
| Reciprocity | 2002: no correction ½ – 1/10 000 s, graph axes 5–30 s metered → 25–175 s adjusted. 2018: no correction 1 – 1/10 000 s, **Ta = Tm^1.26** | 200209 p2; 201811 p2 |
| Filter factors | qualitative only; up to 1½ stops TTL error on deep red/orange | 200209 p1; 201811 p2 |
| Dev times (min @20 °C) EI 50 / 100 / 200 | ID-11 stock 7/8½/10½; 1+1 10/11/13; 1+3 15/20/– · MICROPHEN stock –/6½/8; 1+1 –/10/14; 1+3 –/14/20 · PERCEPTOL stock 12/15/–; 1+1 13/17/–; 1+3 16/22/– · DD-X 1+4 (2002) 9½/12/14, (2018) 8/10½/12½ · ILFOSOL S 1+9 4½/6/–, 1+14 6½/10/– ; ILFOSOL 3 (2018) 1+9 –/5/–, 1+14 –/7½/– · ILFOTEC HC 1+31 5/6/8 · LC29 1+19 5/6/8, 1+29 5½/7½/10 · Kodak D-76 stock 7/9/11, 1+1 9½/12/14, 1+3 14/22/– · HC-110 B 5/6/8 · T-Max 1+4 6/7/8 · Xtol stock 6½/7½/9½ · Microdol-X stock 12/15/– (2002 only) · Rodinal 1+25 7/9/–, 1+50 10/14/– · Acufine stock –/–/5½ | 200209 p3; 201811 p3 |
| Dev times other | ILFOTEC DD 1+4 @24 °C: 8/9½/12½ · RT RAPID 1+1+2 @26 °C: –/40/50 s; 1+1+5: 40/56/75 s | 200209 p3; 201811 p3 |
| Developer-dependent speed | max film speed (EI 200/24) = DD-X or MICROPHEN stock; finest grain EI 100 = PERCEPTOL 1+1, EI 50 = PERCEPTOL stock; max sharpness = ILFOTEC HC 1+31 / ID-11 1+3 | 200209 p2; 201811 p2 |
| Accidental exposure | MICROPHEN stock 10 min for EI 400/27 and above; PERCEPTOL stock 9 min for EI 25/15 and below | 200209 p4; 201811 p4 |
| Base | 35 mm 0.125 mm/5-mil acetate · 120 0.110 mm/4-mil clear acetate + anti-halation backing (clears in development) · sheet 0.180 mm/7-mil polyester + AH backing | 200209 p1; 201811 p1 |
| Safelight | total darkness; brief inspection only with ILFORD 908 very dark green, 15 W (2002 sheet only) | 200209 p5 |

---

## 3. ILFORD DELTA 400 PROFESSIONAL  *(not currently carried)*
Source: `ILFORD/Delta_400-200209.pdf` (Sep 2002, 7 pp). **Richest sheet in the set for curve data.**

| Parameter | Value | Source |
|---|---|---|
| ISO / DIN | ISO 400/27° (400 ASA, 27 DIN, EI 400/27) to daylight, measured ID-11 20 °C intermittent agitation spiral tank; usable EI 200/24 – 3200/36 | p1 |
| Spectral sensitivity | wedge spectrogram to tungsten 2850 K; **vector curve, 21 bezier segments, bbox 316.4,296.5–495.2,353.9 pt** (fully digitisable). Axis text "Wavelength nm" / "Sensitivity"; tick numerals are outlined glyphs, not text. | p1 |
| **Contrast / gamma** | **"The times in bold will produce negatives of normal contrast (Ḡ 0.62)."** The Ḡ glyph sits in a 10.5 pt span gap and is not text-extractable; the value **0.62** is. | p3 |
| Contrast–time graphs | five graphs, y-axis literally "Contrast (Ḡ)": ID-11 stock @(1) 24 °C and (2) 20 °C; MICROPHEN stock @24/20 °C; PERCEPTOL stock @24/20 °C; ILFOTEC DD-X 1+4 @24/20 °C. Vector. | p5, p6 |
| Characteristic curve | 35 mm, ID-11 stock, 8 min @24 °C, intermittent agitation. **Vector** (10-bezier stroked path, bbox 73,343–250,425 pt). | p6 |
| RMS granularity / resolving power / MTF / Dmin / Dmax | **NOT PRINTED** | — |
| Reciprocity | no correction ½ – 1/10 000 s; graph 5–30 s metered → 25–175 s adjusted; no formula given | p2 |
| Filter factors | qualitative only | p1 |
| Dev times 20 °C (min) EI 200/250/320/400/500/800/1600/3200 | ID-11 stock 7/–/–/9½/–/11½/14½/19 · MICROPHEN stock 5/–/–/6½/7½/8½/10½/14 · PERCEPTOL stock 10/12/…/– · DD-X 1+4 6/–/–/8/9½/10½/13½/18 · ILFOTEC HC 1+15 –/–/4/–/–/5½/7½/13; 1+31 5/–/–/7½/–/10/13½/– · LC29 1+19 5/…/7½/…/10/13½/–; 1+29 8½/…/11½/…/17 · ILFOSOL S 1+9 6½/…/9/…/14; 1+14 10/…/13 · D-76 = ID-11 · HC-110 A/B, Microdol-X, T-Max 1+4 5/…/6½/7/8½/10½/13½ · Xtol stock 6/…/7½/8½/10/13/17 · Acufine stock 7/…/9/11/13/16 | p3 |
| Dev times 24 °C | full parallel table (e.g. ID-11 stock 5½/…/8/…/9/11½/15; MICROPHEN stock 4/…/5/6/6½/7½/10) | p4 |
| Machine | ILFOTEC DD 1+4 @24 °C: 6/…/7/…/10/13/14 · RT RAPID 1+1+2 @26 °C: 55/…/65/…/71/84/104 s; 1+1+5: 65/…/78/…/104/127/166 s · T-Max RS @22 °C, Xtol @24 °C tabulated | p3 |
| Developer-dependent speed | max film speed (EI 3200/36) = DD-X or MICROPHEN stock; finest grain EI 200 = PERCEPTOL stock; max sharpness = ILFOSOL S 1+9 / ID-11 1+3 | p2 |
| Base | 35 mm 0.125 mm/5-mil acetate · 120 0.110 mm/4-mil clear acetate + AH backing (no sheet film) | p1 |

---

## 4. ILFORD FP4 PLUS  — *DIFFERENT EMULSION from our `ILFORD_FP4` (1970s). Does not document it.*
Sources: `ILFORD/FP4+-200404.pdf` (Apr 2004, 6 pp), `ILFORD/FP4-Plus_201811.pdf` (Nov 2018, 6 pp).

| Parameter | Value | Source |
|---|---|---|
| ISO / DIN | ISO 125/22° (125 ASA, 22 DIN, EI 125/22) to daylight, measured ID-11 20 °C intermittent agitation spiral tank; usable EI 50/18 – 200/24 | 200404 p1; 201811 p1 |
| Latitude | usable at +6 stops over-exposure / −2 stops under-exposure | 200404 p1; 201811 p1 |
| Spectral sensitivity | wedge spectrogram to tungsten 2850 K. 2004: **vector curve, 12 bezier segments, bbox 336,426–493,488 pt**. 2018: raster (483×244 px). | 200404 p1; 201811 p1 |
| Characteristic curve | 120 roll film, ILFOTEC HC 1+31, 8 min @20 °C intermittent agitation; representative of 35 mm and sheet. 2004: **vector, 9 bezier, bbox 331,609–499,681 pt**. 2018: raster. **No gamma value.** | 200404 p4; 201811 p5 |
| RMS granularity / resolving power / MTF / Dmin / Dmax | **NOT PRINTED** | — |
| Reciprocity | no correction ½ – 1/10 000 s; 2018 gives **Ta = Tm^1.26**; 2004 gives graph only (5–30 s → 25–175 s) | 200404 p2; 201811 p2 |
| Filter factors | qualitative only; up to 1½ stops TTL error on deep red/orange | 200404 p1; 201811 p2 |
| Dev times (min @20 °C) EI 50 / 125 / 200 | DD-X 1+4 8/10/12 · ILFOSOL S 1+9 4½/6½/7½, 1+14 7½/9½/– (2018 ILFOSOL 3 1+9 –/4¼/–, 1+14 –/7½/–) · ILFOTEC HC 1+15 –/4/5, 1+31 6/8/9 · LC29 1+9 –/4/5, 1+19 6/8/9, 1+29 8/12/– · ID-11 stock 6½/8½/10, 1+1 8/11/15, 1+3 17/20/– · MICROPHEN stock –/8/9, 1+1 –/10/14, 1+3 –/14/18 · PERCEPTOL stock 9/12/–, 1+1 13/15/–, 1+3 17/21/– · D-76 stock 6/8/9, 1+1 9/11/15, 1+3 14/16/20 · HC-110 A –/4½/6, B 6/9/12 · T-Max 1+4 –/8/9 · Xtol stock –/8½/10 · Acufine stock –/4/6 · Rodinal 1+25 –/9/13, 1+50 –/15/20 · Microdol-X stock 10/15/– (2004 only) | 200404 p3; 201811 p3 |
| Machine | ILFOTEC DD 1+4 @24 °C 7/8½/11½ · RT RAPID @26 °C 1+1+2 40/45/54 s, 1+1+5 55/65/84 s · ILFOTEC HC 1+11 @24 °C –/70/– s | 200404 p3; 201811 p4 |
| Developer-dependent speed | max film speed = DD-X / MICROPHEN stock; finest grain = DD-X / PERCEPTOL stock; max sharpness = ILFOSOL S(3) / ID-11 1+3 | 200404 p2; 201811 p2 |
| Accidental exposure | MICROPHEN stock 16 min for EI 400/27 and above; PERCEPTOL stock 8½ min for EI 25/15 and below | 200404 p4; 201811 p4 |
| Base | 35 mm 0.125 mm/5-mil acetate · 120/220 0.110 mm/4-mil clear acetate + AH backing · sheet 0.180 mm/7-mil polyester + AH backing | 200404 p1; 201811 p1 |

---

## 5. ILFORD HP5 PLUS  — documents our `ILFORD_HP5_PLUS_400`. **NOT** our `ILFORD_HP4`.
Source: `ILFORD/HP5+-200407.pdf` (Jul 2004, 7 pp).

| Parameter | Value | Source |
|---|---|---|
| ISO / DIN | ISO 400/27° (400 ASA, 27 DIN, EI 400/27) to daylight, measured ID-11 20 °C intermittent agitation spiral tank; usable EI 400/27 – 3200/36 | p1 |
| Spectral sensitivity | wedge spectrogram to tungsten 2850 K; axis "Wavelength nm" / "Sensitivity"; **vector curve, 9 bezier segments, bbox 329,438–487,489 pt**. No nm numerals as text. | p1 |
| Characteristic curve | 35 mm, ILFOTEC HC 1+31, 6½ min @20 °C intermittent agitation; representative of roll and sheet. Axis numerals present as text: density 1.0, 2.0; relative log exposure 1, 2, 3, 4. **No gamma value.** | p5 |
| RMS granularity / resolving power / MTF / Dmin / Dmax | **NOT PRINTED** | — |
| Reciprocity | no correction ½ – 1/10 000 s; >½ s use graph (5–30 s → 25–175 s); no formula | p2 |
| Filter factors | qualitative only; up to 1½ stops TTL error on deep red/orange | p1 |
| Dev times (min @20 °C) EI 250/320/400/800/1600/3200 | DD-X 1+4 –/–/9/10/13/20 · ILFOSOL S 1+9 –/–/7/8½/14/–, 1+14 –/–/9½/14/–/– · ILFOTEC HC 1+15 –/–/3½/5/7½/11, 1+31 –/–/6½/9½/14/– · LC29 1+9 –/–/3½/5/7½/11, 1+19 –/–/6½/9½/14/–, 1+29 –/–/9/–/–/– · ID-11 stock –/–/7½/10½/14/–, 1+1 –/–/13/16½/–/–, 1+3 –/–/20/–/–/– · MICROPHEN stock –/–/6½/8/11/16, 1+1 –/–/12/15/–/–, 1+3 –/–/23/–/–/– · PERCEPTOL stock 13/–/…, 1+1 –/18/…, 1+3 –/25/… · D-76 stock –/–/7½/9½/12½/– · HC-110 A –/–/2½/3¾/5½/9½, B –/–/5/7½/11/– · T-Max 1+4 –/–/6½/8/9½/11½ · Xtol stock –/–/8/11/14/19, 1+1 –/–/12/17/–/– · Acufine stock –/–/4½/6½/9½/– | p3 |
| Machine | ILFOTEC DD 1+4 @24 °C 7/10/14/18 (EI 400/800/1600/3200) · Xtol stock @24 °C 7½/9½/12/16 · T-Max RS stock 4½/5/7/– · RT RAPID 1+1+2 @26 °C 60/75/91/108 s, 1+1+5 70/95/120/166 s · ILFOTEC HC 1+11 @24 °C 55/70/90/130 s · Duraflo RT 60/81/120/166 s | p4 |
| Developer-dependent speed | max film speed = DD-X / MICROPHEN stock; best quality at EI 1600–3200 = DD-X or MICROPHEN stock; finest grain = DD-X / PERCEPTOL; max sharpness = ILFOSOL S / ID-11 1+3 | p2 |
| Accidental exposure | PERCEPTOL stock: 9 min (EI 50/18), 9 min (EI 100/21), 11 min (EI 200/24) | p5 |
| Base | 35 mm 0.125 mm/5-mil acetate · 120 0.110 mm/4-mil clear acetate + AH backing · sheet 0.180 mm/7-mil polyester + AH backing | p1 |

---

## 6. ILFORD ORTHO PLUS  *(not currently carried)* — **the only sheet with a full numeric contrast + filter-factor + tungsten-speed set**
Sources: `ILFORD/Ortho+-200408.pdf` (Aug 2004, 5 pp), `ILFORD/Ortho-Plus_201910.pdf` (Oct 2019, 5 pp).

| Parameter | Value | Source |
|---|---|---|
| Speed — daylight | **ISO 80/20°** (developed to normal contrast in ID-11) | 200408 p1; 201910 p1 |
| Speed — **tungsten** | **ISO 40/17°** (the only separate tungsten speed in the whole corpus). 135 cassettes DX-coded ISO 80; for tungsten set ISO 40 manually or apply 1 stop correction. | 200408 p1; 201910 p1 |
| Sensitisation | **orthochromatic** — blue + green only, no red; handleable under deep red safelight, processing by inspection possible. Reds render much darker than normal. | 200408 p1; 201910 p1 |
| Spectral sensitivity | wedge spectrogram to tungsten 2850 K, axis "Wavelength (nm)" / "Sensitivity". 2004 page is fully outlined vector (no raster); 2019 is raster (620×294 px). No nm numerals as text. | 200408 p1; 201910 p1 |
| **Filter factors (numeric)** | ILFORD 104 Alpha (yellow) **2.5** daylight / **1** tungsten · 109 Delta (deep yellow) **5.5** / **3** · 304 Tricolour Blue **3** / **5** · 404 Tricolour Green **8** / **4.5** | 200408 p1; 201910 p2 |
| **Contrast targets (Ḡ / "Gbar")** | Pictorial **Ḡ 0.62 → Ḡ 0.70** ("normal for in-camera use"); Intermediate **Ḡ 0.80 → Ḡ 1.00**; High **Ḡ 1.2 → Ḡ 1.8** | 200408 p2; 201910 p2 |
| Dev times per Ḡ 0.62 → 0.70 | ID-11 stock 8→10 · ID-11 1+1 10½→13 · ID-11 1+3 16→20 · MICROPHEN stock 9→12 · 1+1 11½→14½ · 1+3 13½→17 · PERCEPTOL stock 13→16 · ILFOSOL S 1+9 4½→6 (2019 ILFOSOL 3 1+9 5:00→6:30, 1+14 7:00→8:30) · ILFOTEC HC 1+15 4→5, 1+31 6→8 · ILFOTEC DD stock/1+4 @24 °C 5½→6½ · RT RAPID 1+1+2 @26 °C 65 s→127 s, 1+1+5 78 s→153 s · **ILFOTEC DD-X 1+4 10:30→13:00 (2019 sheet only)** · Kodak D-76 stock 8→10, 1+1 10½→13 · T-Max RS stock @24 °C 3½→4½ · T-Max 1+4 5→6½ | 200408 p2; 201910 p2 |
| Dev times Ḡ 0.80 → 1.00 | PQ UNIVERSAL 1+9, 20 °C: 4 → 12 min | 200408 p2; 201910 p2 |
| Dev times Ḡ 1.2 → 1.8 | PHENISOL 1+4, 20 °C: 3 → 10 min | 200408 p2; 201910 p2 |
| Characteristic curves | three, one per contrast class: ID-11 stock 8 & 10 min; PQ UNIVERSAL 1+9 4 & 12 min; PHENISOL 1+4 3 & 10 min — all 20 °C intermittent agitation. 2004 p3 axis numerals present as text: 1,2,3,4 (log E) and 1.0, 2.0, 3.0 (density); curve labels "G1.00", "G0.80" printed on the graph. | 200408 p3; 201910 p3 |
| Dmin / Dmax | not printed (density axis reaches 3.0 on the high-contrast graph) | 200408 p3 |
| RMS granularity | **NOT PRINTED** | — |
| Resolving power | **NOT PRINTED** — text claims "high resolution film" / "qualities of a high resolution film" but gives no lines/mm and no test-object contrast | 200408 p1; 201910 p1 |
| MTF | **NOT PRINTED** | — |
| Reciprocity | 2004: no correction ½ – 1/10 000 s, graph only. 2019: no correction 1 – 1/10 000 s, **Ta = Tm^1.25** | 200408 p2; 201910 p2 |
| Base | 2004: sheet only, 0.180 mm / 7-mil polyester + AH backing (high dimensional stability, archival). 2019 adds 35 mm **and** 120 on 0.125 mm / 5-mil acetate (120 base has "excellent anti-halation properties") | 200408 p1; 201910 p1 |
| Safelight | ILFORD 906 dark red, 15 W, minimum 1.2 m / 4 ft distance | 200408 p4; 201910 p4 |

---

## 7. ILFORD PAN F PLUS  — *DIFFERENT EMULSION from our `ILFORD_PAN_F`. Does not document it.*
Sources: `ILFORD/PanF+-200407.pdf` (Jul 2004, 6 pp), `ILFORD/Pan-F-Plus_201812.pdf` (Dec 2018, 6 pp),
`KENTMERE/PANF+ technical data sheet 2026.pdf` (rev **B26**, 6 pp) — **misfiled under KENTMERE; it is an ILFORD PAN F PLUS sheet, not a Kentmere film.**

| Parameter | Value | Source |
|---|---|---|
| ISO / DIN | ISO 50/18° (50 ASA, 18 DIN, EI 50/18°) to daylight, measured ID-11 20 °C intermittent agitation spiral tank; also good at EI 25/15° | 200407 p1; 201812 p1; B26 p1 |
| Spectral sensitivity | wedge spectrogram to tungsten 2850 K. **2004 sheet: vector curve, 18 bezier segments, bbox 342,259–485,308 pt, AND the x-axis numerals are real text — 400, 450, 500, 550, 600, 650 (nm) with ordinate ticks 0.5 and 1.0.** So the printed spectrogram spans ≈400–650+ nm. 2018 and B26: raster only. | 200407 p1; 201812 p1; B26 p1 |
| Characteristic curve | 120 roll film, ILFOTEC HC 1+31, **4 min** @20 °C intermittent agitation; representative of 35 mm. 2004 axis numerals as text: 1,2,3,4 / 1.0, 2.0. **No gamma value.** | 200407 p4; 201812 p4; B26 p4 |
| RMS granularity / resolving power / MTF / Dmin / Dmax | **NOT PRINTED** — text claims "extremely fine grain… outstanding resolution, sharpness and edge contrast" with no numbers | all three |
| Reciprocity | no correction ½ – 1/10 000 s; **Ta = Tm^1.33** (2018 and B26; 2004 graph only, 5–30 s → 25–175 s) | 201812 p2; B26 p2; 200407 p2 |
| Filter factors | qualitative only; up to 1½ stops TTL error on deep red/orange | all three |
| Dev times (min @20 °C) EI 25 / 50 / 64 | DD-X 1+4 7/8/– · ILFOSOL S 1+9 –/4/–, 1+14 –/6/– (2018 & B26: ILFOSOL 3 1+9 –/–/–, 1+14 –/4½/–) · ILFOTEC HC 1+31 –/4/– · LC29 1+19 –/4/–, 1+29 –/5½/– · ID-11 stock 4½/6½/–, 1+1 6/8½/–, 1+3 12½/15/– · MICROPHEN stock –/4½/6, 1+1 –/6/9, 1+3 –/11/14½ · PERCEPTOL stock 9/14/–, 1+1 10½/15/–, 1+3 15/17/– · D-76 stock 4½/6½/–, 1+1 6/8½/–, 1+3 12½/15/– · HC-110 B –/4/– · T-Max 1+4 –/4/– · Xtol stock 5½/6¾/– · Acufine stock –/3½/– · Rodinal 1+25 –/6/–, 1+50 –/11/– · Tetenal Ultrafin 1+10 –/4/–, 1+20 –/8/–, Ultrafin Plus 1+4 –/5/– · Microdol-X stock 12/15/–, 1+3 15/18/– (2004 only) | 200407 p3; 201812 p3; B26 p3 |
| Machine | ILFOTEC DD 1+4 @24 °C 4½/5½/– · T-Max RS stock –/3/– · Xtol stock 4½/6/– · RT RAPID 1+1+2 @26 °C –/40/– s, 1+1+5 45/50/– s · ILFOTEC HC 1+11 @24 °C 50/65/– s | 200407 p3; 201812 p3; B26 p3 |
| Accidental exposure | MICROPHEN stock 8 min (EI 100/21), 12 min (EI 200/24 and above); ID-11 stock 4 min (EI 12/12 and below) | 200407 p4; 201812 p4; B26 p4 |
| Latent-image note | **process within 3 months of exposure** (2018 and B26 only — physically relevant latent-image instability) | 201812 p1, p6; B26 p1 |
| Base | 35 mm 0.125 mm/5-mil acetate · 120 0.110 mm/4-mil clear acetate + AH backing · **B26 adds large-format sheet film 0.180 mm/7-mil polyester + AH backing** (not in 2004/2018) | 200407 p1; 201812 p1; B26 p1 |

---

## 8. ILFORD SFX 200  *(not currently carried)* — **best extractable spectral curve + a full numeric filter table**
Sources: `ILFORD/SFX200-200404.pdf` (Apr 2004, 6 pp), `ILFORD/SFX-200_201811.pdf` (Nov 2018, 6 pp).

| Parameter | Value | Source |
|---|---|---|
| ISO / DIN | ISO 200/24° (200 ASA, 24 DIN, EI 200/24) to daylight, measured ID-11 20 °C intermittent agitation spiral tank | 200404 p1; 201811 p1 |
| **Spectral sensitivity — nm figures** | full panchromatic **plus extended red sensitivity up to 740 nm**, with **peak red sensitivity at 720 nm** | 200404 p1 (both figures); 201811 p1 (740 nm only) |
| Spectral curve | wedge spectrogram to tungsten 2850 K, axis "Wavelength (nm)". **2004: vector, 36-item path with 30 bezier segments, bbox 332,262–497,305 pt — the most completely digitisable spectral curve in the corpus.** 2018: raster (879×436 px). | 200404 p1; 201811 p1 |
| **Filter factors (numeric)** | Wratten 3 very light yellow **2** (1 stop) · 8 yellow **2** (1 / 1⅓) · 12 deep yellow **2.3** (1⅓) · 15 very deep yellow **2.4** (1⅓) · 21 orange **2.4** (1⅓) · 23a reddish orange **2.5** (1⅓) · 25 red **2.8** (1½) · 29 deep red **3** (1⅔) · 89B very deep red **16** (4) · ILFORD SFX very deep red **16** (4) | 200404 p1; 201811 p2 |
| Characteristic curve | ID-11 stock, 10 min @20 °C intermittent agitation. Raster in 2018 (514×364 px); 2004 sheet has no separate characteristic-curve section. **No gamma value.** | 201811 p2 |
| RMS granularity / resolving power / MTF / Dmin / Dmax | **NOT PRINTED** | — |
| **Reciprocity** | **ABSENT from both sheets** — there is no "MAKING LONG EXPOSURES" section, no exponent, no graph. Notable gap for a film intended for deep-red-filter (i.e. long) exposures. | — |
| Practical exposure datum | bright sunlight + deep red filter ≈ 1/30 s at f/5.6 at EI 200/24; bracket ±2 stops | 200404 p3; 201811 p2 |
| TTL error | up to 1½ stops under-exposure with deep red or orange filters | 200404 p2; 201811 p2 |
| Dev times (min @20 °C) EI 200 / 400 / 800 | DD-X 1+4 10/14/– · ILFOSOL S 1+9 9½/11½/19, 1+14 13/19/– (2018 ILFOSOL 3 1+9 6/8½/–, 1+14 9/13½/–) · ILFOTEC HC 1+15 5/7/10½, 1+31 9/13/19 · LC29 1+9 5/7/10½, 1+19 9/13/19, 1+29 11/–/– · ID-11 stock 10/14/18, 1+1 17/–/– · MICROPHEN stock 8½/10½/14½, 1+1 15½/19/– · PERCEPTOL stock 14½/–/–, 1+1 20/–/– · D-76 stock 10/12½/16½, 1+1 14½/–/– · HC-110 A 5/7/10½, B 9/13/19 · T-Max 1+4 8½/10½/12½ · Xtol stock 7/11/– · Tetenal Ultrafin 1+10 10/13/– · Agfa Refinal stock 8/11½/– (2004 only) | 200404 p4; 201811 p4 |
| Machine | ILFOTEC DD 1+4 @24 °C 8½/11½/14 · T-Max RS stock 6/7/9 · Xtol stock 7/9/11½ · RT RAPID @26 °C 1+1+2 54/65/88 s, 1+1+5 65/90/120 s · Duraflo RT stock 100/135/200 s (2004 only) | 200404 p4; 201811 p4 |
| Base | **0.125 mm / 5-mil GREY acetate base, which itself gives the halation protection** (unique in this set — all other ILFORD films use a clear base plus an anti-halation backing) | 200404 p1; 201811 p1 |
| Focus note | red focus shift with some lenses; stop down to smallest workable aperture; APO lenses may need no correction | 200404 p2; 201811 p2 |

---

## 9. ILFORD XP2 SUPER  *(not currently carried)* — chromogenic, C-41
Sources: `ILFORD/XP2_Super-200101.pdf` (Jan 2001, 4 pp), `ILFORD/XP2-Super_201811.pdf` (Nov 2018, 5 pp).

| Parameter | Value | Source |
|---|---|---|
| ISO / DIN | ISO 400/27° (400 ASA, 27 DIN, EI 400/27) to daylight, **ISO speed measured using standard C-41 processing**; usable EI 50/18 – 800/30 | 200101 p1; 201811 p1 |
| Speed-vs-grain inversion | best balance EI 400/27; finer grain EI 200/24; **finest grain EI 50/18** — over-exposure *reduces* grain (opposite of silver films) | 200101 p1; 201811 p2 |
| Spectral sensitivity | wedge spectrogram to tungsten — **2850 K in the 2001 sheet, 2856 K in the 2018 sheet**. 2001 p2 axis is real text: "Wavelength (nm)" with 400, 450, 500, 550, 600, 650, and a vector curve (9-bezier path). 2018: raster (515×264 px). | 200101 p2; 201811 p2 |
| Characteristic curve | present, "processed through standard C41 type chemicals", representative of 35 mm and roll film. Vector graph frame in 2001 (bbox 327,454–511,576 pt); raster in 2018. **No gamma / Dmin / Dmax values.** | 200101 p2; 201811 p3 |
| RMS granularity / resolving power / MTF | **NOT PRINTED** | — |
| Reciprocity | 2001: no correction ½ – 1/10 000 s, graph 5–30 s → 25–175 s. 2018: no correction 1 – 1/10 000 s, **Ta = Tm^1.31** | 200101 p2; 201811 p2 |
| Filter factors | qualitative only; up to 1½ stops TTL error on deep red/orange | 200101 p2; 201811 p2 |
| Development | standard C-41 only, no time table. **Push processing gives NO practical speed increase** (explicit statement that speed is developer-independent here). Replenish as if ISO 200/24 colour negative regardless of exposure index. | 200101 p3; 201811 p3 |
| Dye image | dye (not silver) image; negatives pink / red-brown; contrast virtually unaffected by enlarger illumination type; residual sensitising dye bleaches on long wash / light exposure; Digital ICE usable | 200101 p3; 201811 p3, p4 |
| Base | 35 mm 0.125 mm/5-mil acetate · 120 0.110 mm/4-mil clear acetate + AH backing | 200101 p1; 201811 p1 |

---

## 10. KENTMERE PAN 200  *(not currently carried — we hold PAN 100 and PAN 400, neither of which is documented here)*
Source: `KENTMERE/KENTMERE_PAN_200_technical data sheet.pdf` (Mar 2025, 4 pp).

| Parameter | Value | Source |
|---|---|---|
| ISO / DIN | ISO 200/24° to daylight, measured ILFORD ID-11 20 °C/68 °F intermittent agitation spiral tank | p1 |
| Contrast (qualitative) | "will generally yield **mid-high pictorial contrast** at the given development times" — no Ḡ or gamma number | p1 |
| Spectral sensitivity | **NO SPECTRAL SENSITIVITY SECTION AT ALL** — no spectrogram, no sensitisation class, no nm figures | — |
| Characteristic curve | **ABSENT** — this sheet has no characteristic curve | — |
| RMS granularity / resolving power / MTF / Dmin / Dmax | **NOT PRINTED** ("fine grain and good sharpness", "high quality enlargements or high-resolution scans" only) | p1 |
| Reciprocity | no correction 1 – 1/10 000 s; >1 s use graph; **Ta = Tm^1.26** | p1 |
| Filter factors | qualitative only; up to 1½ stops TTL error on deep red/orange | p1 |
| Dev times (min:sec @20 °C) EI 100 / 200 / 400 | DD-X 1+4 7:00/9:00/12:30 · ILFOSOL 3 1+9 –/5:30/–, 1+14 4:30/7:30/– · ILFOTEC HC 1+15 –/4:00/7:30, 1+31 4:45/6:00/10:00 · LC29 1+9 –/4:00/4:45, 1+19 4:45/6:00/7:30, 1+29 7:00/10:30/– · ID-11 stock 5:00/7:30/–, 1+1 6:00/8:45/–, 1+3 12:30/–/– · MICROPHEN stock –/5:30/8:00, 1+1 –/7:00/10:00, 1+3 –/9:45/– · PERCEPTOL stock 10:00/13:00/–, 1+1 14:00/16:00/– · Kodak D-76 stock 5:00/7:30/–, 1+1 6:00/8:45/–, 1+3 12:30/–/– | p2 |
| Machine | ILFOTEC DD 1+4 @24 °C 4:00/7:00/12:00 · RT RAPID @26 °C 1+1+2 0:40/0:50/0:60, 1+1+5 0:55/1:10/– | p2 |
| Base | **0.125 mm / 5-mil acetate LOW DENSITY base** (wording unique to Kentmere) | p1 |
| Replenishment | ILFOSTOP 60 ml per 135/36; RAPID/HYPAM 40 ml per 135/36 (only sheet giving replenishment volumes) | p3, p4 |

---

## 11. `ILFORD/2006129224892363.pdf` — Kodak→ILFORD conversion table (3 pp, HARMAN technology Ltd)
No physical film parameters. Product-equivalence only. Usable data points:
- BW400CN→XP2 SUPER (ISO 400, C-41); TRI-X 320/400TX→HP5 PLUS (ISO 320/400, push to EI 3200); PLUS-X 125PX→FP4 PLUS (ISO 125); (no Kodak equiv.)→PAN F PLUS (ISO 50); T-MAX 100→DELTA 100; T-MAX 400→DELTA 400; **T-MAX P3200→DELTA 3200, described as "ISO 1000 ultra fast, push to EI 25000"** (independently corroborates the ISO 1000/31° figure); (none)→ORTHO PLUS "ISO 80 Orthochromatic Continuous Tone Copy Film". (p1)
- DELTA films use "ILFORD controlled crystal growth 'core shell' technology"; Kodak T-MAX is T-Grain. (p1)
- Developer equivalences (p3): D-76≡ID-11, T-Max/Xtol≡ILFOTEC DD-X or DD, HC-110≡ILFOTEC LC29 or HC, DK-50≡MICROPHEN, Microdol-X≡PERCEPTOL, Technidol & D-19≡PHENISOL, Duraflo RT≡ILFOTEC RT RAPID. **Useful for mapping non-ILFORD dev times onto ILFORD chemistry.**

## 12. `ILFORD/2006216122447.pdf` — ILFORD FILM PROCESSING CHART, 20 °C/68 °F (1 p, Aug 2004, ref 01074.GB.www)
Single-page matrix: films **PAN F PLUS, FP4 PLUS, HP5 PLUS, DELTA 100 PRO, DELTA 400 PRO, DELTA 3200 PRO, SFX 200** × EI columns (25…12500) × developers ILFOTEC DD-X 1+4, ID-11 stock/1+1/1+3, ILFOTEC HC 1+15/1+31, ILFOTEC LC29 1+9/1+19/1+29, ILFOSOL S 1+9/1+14, MICROPHEN stock/1+1/1+3, PERCEPTOL stock/1+1/1+3, ACUFINE stock, RODINAL 1+25/1+50, D-76 stock/1+1/1+3, HC-110 A 1+15 / B 1+31, MICRODOL-X stock/1+3, T-MAX 1+4, XTOL stock/1+1.
**Caveat:** the cell-to-column mapping is NOT recoverable from the linear text stream (values arrive as a flat number list); reconstructing it requires x-coordinate clustering. All values are duplicated in the per-film sheets above, so treat this file as a redundant cross-check only. Also states: times <5 min may develop unevenly; continuous agitation → reduce by up to 15 %; rotary without pre-rinse → reduce up to 15 %; pre-rinse not recommended.

---

# Summary answers

## (a) Which of OUR stocks does each sheet actually document? (strict on Plus-vs-original)
| Our stock | Documented by | Verdict |
|---|---|---|
| `ILFORD_DELTA_3200` | `ILFORD/DELTA 3200 technical data sheet F25.pdf` (all 7 pp) | **YES, fully.** Plus `2006216122447.pdf` (dev times) and `2006129224892363.pdf` (ISO 1000 corroboration) |
| `ILFORD_HP5_PLUS_400` | `ILFORD/HP5+-200407.pdf` (all 7 pp) | **YES, fully.** Plus `2006216122447.pdf` |
| `ILFORD_PAN_F` | — | **NO.** `PanF+-200407`, `Pan-F-Plus_201812`, `KENTMERE/PANF+ …2026` are all **PAN F PLUS**, a different emulsion. Do not merge. |
| `ILFORD_FP4` | — | **NO.** `FP4+-200404` and `FP4-Plus_201811` are **FP4 PLUS**, a different emulsion. Do not merge. |
| `ILFORD_HP4` | — | **NO.** HP5 PLUS is a later, different emulsion; no HP4 sheet exists here. |
| `ILFORD_HP3` | — | **NO. Nothing.** Still zero documentation. |
| `ILFORD_HPS` | — | **NO. Nothing.** Still zero documentation. |
| `KENTMERE_PAN_100` | — | **NO.** Only a PAN **200** sheet is present. |
| `KENTMERE_PAN_400` | — | **NO.** Only a PAN **200** sheet is present. |

Net: **2 of 9 existing stocks gain documentation.** The 1970s Ilford films (PAN F, FP4, HP4, HP3, HPS) gain nothing.

## (b) Films documented that we do NOT carry (8 candidate new stocks)
1. **ILFORD DELTA 100 PROFESSIONAL** — 2 sheets (2002 + 2018)
2. **ILFORD DELTA 400 PROFESSIONAL** — 1 sheet (2002); *the only camera film with a printed contrast number, Ḡ 0.62*
3. **ILFORD FP4 PLUS** — 2 sheets (2004 + 2018)
4. **ILFORD PAN F PLUS** — 3 sheets (2004, 2018, B26/2026)
5. **ILFORD ORTHO PLUS** — 2 sheets (2004 + 2019); *only dual daylight/tungsten ISO, only full Ḡ ladder, numeric filter factors*
6. **ILFORD SFX 200** — 2 sheets (2004 + 2018); *only nm peak/limit figures, only full Wratten filter-factor table, grey base*
7. **ILFORD XP2 SUPER** — 2 sheets (2001 + 2018); chromogenic C-41
8. **KENTMERE PAN 200** — 1 sheet (2025)

## (c) Exact RMS granularity and resolving-power numbers found
**NONE. Zero. In any of the 19 files.**
- No RMS granularity value, no measuring aperture (no 48 µm / 24 µm), no magnification, no diffuse-density reference.
- No resolving power in lines/mm and therefore no test-object contrast — **no `1000:1`, no `1.6:1`, no ISO 6328 reference anywhere**.
- No MTF curve or MTF table on any page of any file.
- The only substitutes ILFORD prints are (i) marketing adjectives ("exceptionally fine grain", "extremely fine grain", "outstanding resolution", "high resolution film"), and (ii) the per-film **"Finest grain" / "Maximum sharpness" developer-recommendation matrix**, which is an *ordinal* ranking of developers only. It is on: DELTA 3200 p2; Delta 100 200209 p2 / 201811 p2; Delta 400 200209 p2; FP4 Plus 200404 p2 / 201811 p2; HP5 Plus 200407 p2; PAN F Plus 200407 p2 / 201812 p2 / B26 p2; SFX 200 200404 p2 / 201811 p3.
- These fields remain unfilled after this corpus. They will have to come from third-party measurement or non-ILFORD literature.

## (d) Vector-curve files / pages (exactly extractable)
The seven 2001–2004 "FACT SHEET" PDFs contain **zero raster images** — every graph is vector. Note that the initial ">=25 items per path" screen is misleading on these files: the ≥25-item paths are the **logo/title outlines** (bbox y≈40–140 pt) and the **41+30-item temperature-conversion nomograph frames**; the actual data curves are 8–36-item *stroked bezier* paths. Verified data curves:

| File | Page | Vector curve | Path size / bbox (pt) |
|---|---|---|---|
| `ILFORD/SFX200-200404.pdf` | 1 | **spectral sensitivity (extended red)** — best in corpus | 36 items, 30 bezier, 332,262–497,305 |
| `ILFORD/Delta_400-200209.pdf` | 1 | spectral sensitivity | 21 items, 21 bezier, 316,296–495,354 |
| `ILFORD/PanF+-200407.pdf` | 1 | spectral sensitivity (+ nm axis numerals as **text**: 400–650) | 18 items, 18 bezier, 342,259–485,308 |
| `ILFORD/FP4+-200404.pdf` | 1 | spectral sensitivity | 12 items, 12 bezier, 336,426–493,488 |
| `ILFORD/HP5+-200407.pdf` | 1 | spectral sensitivity | 9 items, 9 bezier, 329,438–487,489 |
| `ILFORD/XP2_Super-200101.pdf` | 2 | spectral sensitivity (+ nm axis numerals as **text**: 400–650) | 9 bezier, 90,142–249,197 |
| `ILFORD/Delta_100-200209.pdf` | 1 | spectral sensitivity | 7 bezier, 325,281–482,334 |
| `ILFORD/Delta_400-200209.pdf` | 6 | **characteristic curve** (ID-11 stock 8 min 24 °C) | 10 bezier, 73,343–250,425 |
| `ILFORD/Delta_400-200209.pdf` | 5, 6 | **contrast–time graphs, y-axis "Contrast (Ḡ)"** — 5 graphs (ID-11, MICROPHEN, PERCEPTOL, DD-X, each 20 & 24 °C) | 8-bezier curves inside 18-item frames at 313,149–498,271 / 313,358–497,479 / 312,575–496,697 / 71,95–255,217 / 72,315–259,440 |
| `ILFORD/FP4+-200404.pdf` | 4 | **characteristic curve** (ILFOTEC HC 1+31, 8 min 20 °C) | 9 bezier, 331,609–499,681 |
| `ILFORD/Ortho+-200408.pdf` | 3 | **characteristic curves**, 3 contrast classes (axis numerals + "G0.80"/"G1.00" as text) | 12-bezier curve; frames 313,335–485,471 and 314,122–489,238 |
| `ILFORD/PanF+-200407.pdf` | 4 | characteristic curve frame + polyline (ILFOTEC HC 1+31, 4 min) | frame 326,300–515,425 |
| `ILFORD/HP5+-200407.pdf` | 5 | characteristic curve frame + polyline (ILFOTEC HC 1+31, 6½ min) | frame 327,314–511,436 |
| `ILFORD/Delta_100-200209.pdf` | 4 | temperature nomographs vector (41 + 30 items), **but the characteristic curve on this page is a RASTER image** (1192×792 px, bbox 294,295–509,438) | — |

**Raster-only (NOT exactly extractable) — all 2018–2026 revisions:** `Delta-100_201811` p1/p2/p4 · `FP4-Plus_201811` p1/p2/p4/p5 · `Ortho-Plus_201910` p1/p2/p3 · `Pan-F-Plus_201812` p1/p2/p4 · `SFX-200_201811` p1/p2/p5 · `XP2-Super_201811` p2/p3 · `DELTA 3200 … F25` p1/p2/p5 · `KENTMERE_PAN_200 …` p1/p3 · `KENTMERE/PANF+ … 2026` p1/p2/p4/p6.
**Practical consequence: for any film that has both an old and a new sheet, digitise the graphs from the 2001–2004 vector sheet, not the modern one.** DELTA 3200 and KENTMERE PAN 200 exist only as raster.

## (e) Files with nothing (or almost nothing) useful
- `ILFORD/2006129224892363.pdf` — Kodak→ILFORD conversion table. **No physical parameters.** Only value: corroborates DELTA 3200 ≈ ISO 1000, ORTHO PLUS ISO 80, and gives the developer-equivalence map (D-76≡ID-11, Microdol-X≡PERCEPTOL, HC-110≡LC29/HC, DK-50≡MICROPHEN, Technidol/D-19≡PHENISOL). Pages 2 and 3 (papers, chemistry) are entirely irrelevant.
- `ILFORD/2006216122447.pdf` — processing chart. Redundant with the per-film sheets, and the number-to-column mapping is unrecoverable from text order.
- `ILFORD/XP2-Super_201811.pdf` / `XP2_Super-200101.pdf` — beyond ISO 400/27°, base thickness, the reciprocity exponent 1.31 and the "no push gain" statement, these contribute nothing quantitative (C-41 process, no time tables, no curve numbers).
- `KENTMERE/KENTMERE_PAN_200_technical data sheet.pdf` — thinnest technical sheet in the corpus: **no spectral sensitivity section and no characteristic curve at all.** Only ISO, base, reciprocity exponent and dev times.
- `ILFORD/Ortho-Plus_201910.pdf`, `Delta-100_201811.pdf`, `FP4-Plus_201811.pdf`, `Pan-F-Plus_201812.pdf`, `SFX-200_201811.pdf` — not useless (they add the reciprocity exponents, DD-X for Ortho, updated formats/bases) but their graphs are raster where the older sheet's are vector.

## Parameter classes absent from EVERY sheet in this corpus
1. **RMS granularity** (value, aperture, magnification) — 0/19 files.
2. **Resolving power in lines/mm** and **test-object contrast (1000:1, 1.6:1)** — 0/19 files.
3. **MTF** data or curves — 0/19 files.
4. **Dmin, Dmax, base+fog density** — 0/19 files (only unlabelled density axes, max 3.0 on ORTHO PLUS high-contrast graph).
5. **Numeric gamma / contrast index for camera films** — only two exceptions in 19 files: DELTA 400 (Ḡ 0.62) and ORTHO PLUS (Ḡ 0.62/0.70/0.80/1.00/1.2/1.8). All other films: "average contrast suitable for printing in all enlargers", unquantified.
6. **Numeric spectral sensitivity data** (log-sensitivity vs nm tables) — none; only wedge spectrograms. Only SFX 200 prints nm figures (peak 720 nm, limit 740 nm). Only PAN F Plus 2004 and XP2 Super 2001 print the wavelength axis numerals as text (400–650 nm).
7. **Reciprocity correction tables** — none; only a graph and, in the 2018+ sheets, a power law Ta = Tm^n (n = 1.25 ORTHO, 1.26 DELTA 100 / FP4 PLUS / KENTMERE PAN 200, 1.31 XP2 SUPER, 1.33 PAN F PLUS / DELTA 3200; SFX 200 gives nothing at all).
8. **Numeric filter factors** — only ORTHO PLUS and SFX 200. All other films: qualitative text plus the "TTL can under-expose by up to 1½ stops with deep red/orange" note.
9. **Separate tungsten speed** — only ORTHO PLUS (ISO 40/17° vs daylight ISO 80/20°).
10. **Emulsion-layer thickness, silver coating weight, grain-size distribution, halation/interimage coefficients, spectral dye densities, scanning/printing densitometry** — none anywhere. Only total *base* thickness (0.110 / 0.125 / 0.180 mm) and base material (acetate / grey acetate / polyester) are given.
