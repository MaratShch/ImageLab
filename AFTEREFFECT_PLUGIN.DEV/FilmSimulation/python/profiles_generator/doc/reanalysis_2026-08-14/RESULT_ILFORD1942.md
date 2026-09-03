# ILFORD MANUAL OF PHOTOGRAPHY (ed. J. Mitchell, 1942) — extraction for ILFORD_HP3 / ILFORD_HPS

Source file: `PDF/PROFILES/ILFORD/mitchell_j_ed_the_ilford_manual_of_photography.pdf`
492 PDF pages. Producer: "Adobe Acrobat 11.0.12 Paper Capture Plug-in with ClearScan".

## 0. OCR layer

**An OCR text layer EXISTS** (ClearScan, i.e. a scan with synthesised font — text is searchable but
noisy, and tabular column alignment is destroyed).

* Total extracted text: 1,028,727 chars over 492 pages
* Median 2,333 chars/page; max 3,334; min 0
* Pages with <100 chars (plates/blanks): 0,1,5,7,39,314,315,316,354,355,356,357,358,359,360
* Density buckets: <100 → 15 pages; 100-500 → 25; 500-1500 → 68; 1500-3000 → 339; >3000 → 45

Because tables and all graph annotations are OCR-garbled, **every number below was verified by
rendering the page to PNG at 260-900 dpi and reading the image**, not from the text layer.

**PAGE NUMBER CONVENTION: printed book page = PDF page index − 7.** Both are given below.

## 1. Does HP3 data exist? YES — H.P.3 is a fully documented 1942 product in 3 forms

`H.P.3` occurs on PDF pages 81,83,84,85,86,87,103,161,164,166,167,202,212,413.
The owner's tip was right: printed pp.79-80 are the characteristic-curve pages, and they are
**per-named-product curves, not generic theory**.

### 1.1 ILFORD H.P.3 PLATE — (a) NAMED PRODUCT

Speed table, printed p.74 / PDF 81 (read from image):
> H.P.3 — "A plate of extreme speed specially valuable for the Press, and for all conditions where
> short exposures are essential or lighting is bad" — **H. & D. 6000 | Scheiner 34 | Ilford Speed Group G**

"RECENT INTRODUCTIONS", printed p.78 / PDF 85, verbatim — (a):
> "The first, Ilford H.P.3, is the fastest panchromatic plate available. It has a speed rating of
> **6000 H. & D., 34°, Group G**, and is twice as fast in half-watt light as the older H.P.2 plate.
> The increased speed has been achieved without sacrificing any other quality and there is **no
> increase in grain size or graininess**."

Filter factors, printed p.78 / PDF 85 — (a).
**WARNING: the 4th column "Gamma" is the Ilford *Gamma filter*, NOT contrast gamma.**
| | Red | Green | Blue | Gamma (filter) |
|---|---|---|---|---|
| Daylight | 4 | 6 | 7 | 3½ |
| Half-watt | 2¼ | 7 | 20 | 3 |

Development, printed p.78 / PDF 85 — (a):
> "Formula ID-2 is recommended and the times of development are as follows:"
| | 55°F (13°C) | 65°F (18°C) | 75°F (24°C) |
|---|---|---|---|
| Dish | 7½ min | 5 min | 3½ min |
| Tank | 15 min | 10 min | 7 min |
> "For Fine Grain Development the Ilford M.Q. Borax formula ID-11 is recommended, with which a time
> of 15 mins. at 65°F. is advised."

Characteristic curve, printed p.79 / PDF 86, graph titled "ILFORD H.P.3 PLATE" — (a).
Printed legend verbatim:
> "EXPOSED TO INTERNATIONAL SUNLIGHT / DEVELOPED 4 & 8 MINS. IN ILFORD METOL HYDROQUINONE DEVELOPER
> (ID.2) AT 65°F. / GAMMA TIME DEVELOPMENT CURVES ARE GIVEN FOR ID.11 & ID.2 DEVELOPERS"

Axes as printed: left plot — y = DENSITY, labelled ticks 1.0 and 2.0, gridlines every 0.5,
frame top ≈ D 3.4; x = RELATIVE LOG E, labelled ticks 1, 2, 3, gridlines every 0.5, frame spans
≈ logE 0.5 to 3.3. Right subplot — y = GAMMA, labelled ticks 0.5 and 1.0, zero on the bottom frame,
frame top ≈ γ 1.5; x = DEVELOPMENT TIME IN MINS, labelled 0, 5, 10, 15.

Digitised gamma-vs-development-time (my measurement off the printed curve, calibrated on the printed
tick marks; ±0.05 γ, ±0.3 min):

| gamma | ID-2 (min) | ID-11 (min) |
|---|---|---|
| 0.5 | 1.2 | 3.6 |
| 0.7 | 2.0 | 4.9 |
| 0.8 | 2.7 | 5.8 |
| 1.0 | 4.2 | 8.1 |
| 1.2 | 6.2 | 10.9 |
| 1.4 | 8.6 | 14.5 |

Both curves start from γ≈0 at t≈0.2 (ID-2) and t≈0.8 (ID-11); neither is plotted beyond γ 1.5.
Cross-check (independent): straight-line gradient of the printed D/logE curves gives
**γ(ID-2, 8 min, 65°F) = 1.36** and **γ(ID-2, 4 min) ≈ 0.94** — agrees with the gamma-time curve
(1.35 and 0.96) to within 0.02. Curve shape: long low toe emerging from base+fog at D≈0.05-0.15,
straight from about D 0.3, **no shoulder within the plotted range** (max plotted D ≈ 2.6).
Also consistent with the printed dev recommendation: ID-2 dish 5 min at 65°F → γ≈1.1;
ID-11 15 min at 65°F → γ≈1.45.

### 1.2 ILFORD H.P.3 FLAT FILM ("Hypersensitive Panchromatic H.P.3") — (a)

Speed table, printed p.75 / PDF 82: **H. & D. 5000 | Scheiner 32 | Group F**
> "An extremely fast film for general purposes, especially valuable for night photography indoors and
> out. An excellent film for portraiture"

Characteristic curve, printed p.80 / PDF 87, "ILFORD H.P.3 FILM" — (a).
Printed legend: "EXPOSED TO **HALF-WATT** LIGHT / DEVELOPED 4 & 8 MINS. IN ILFORD METOL HYDROQUINONE
DEVELOPER (ID.2) AT 65°F. / GAMMA DEVELOPMENT TIME CURVES ARE GIVEN FOR ID.11 & ID.2 DEVELOPERS".
Same axis scheme as above.

Digitised gamma vs time (±0.05 γ, ±0.4 min):

| gamma | ID-2 (min) | ID-11 (min) |
|---|---|---|
| 0.4 | 1.2 | 5.2 |
| 0.5 | 1.6 | 6.7 |
| 0.6 | 2.2 | 8.6 |
| 0.7 | 2.8 | 11.1 |
| 0.8 | 3.8 | — |
| 0.9 | 5.4 | — |

D/logE straight-line gradients: **γ(ID-2, 8 min) ≈ 0.94**, **γ(ID-2, 4 min) ≈ 0.79**
(consistent with the gamma-time curve, ~1.0 and ~0.82).

Development table, printed p.195 / PDF 202 (Ilford M.Q. / Ilford Pyro Soda and Certinal):
dish 5¼ / 3½ / 2¼ min and tank 10½ / 7 / 4¼ min at 55 / 65 / 75°F.

### 1.3 SELO H.P.3 ROLL FILM and SELO H.P.3 35 mm — (a)

Speed tables, printed p.76 / PDF 83, both entries identical:
**H. & D. 5000 | Scheiner 32 | Group F**
> "For high speed subjects and for night photography indoors and out. Fully colour-sensitive"

Characteristic curve, printed p.80 / PDF 87, "SELO H.P.3 ROLL FILM" — (a).
Printed legend: "EXPOSED TO INTERNATIONAL SUNLIGHT / DEVELOPED 4 & 8 MINS. IN ILFORD METOL
HYDROQUINONE DEVELOPER (ID2) AT 65°F. / GAMMA DEVELOPMENT TIME CURVES ARE GIVEN FOR ID.11 & ID.2".

Digitised gamma vs time (±0.05 γ, ±0.4 min):

| gamma | ID-2 (min) | ID-11 (min) |
|---|---|---|
| 0.4 | 1.8 | 5.8 |
| 0.5 | 2.0 | 7.1 |
| 0.6 | 2.1 | 8.3 |
| 0.7 | 2.3 | 9.7 |
| 0.8 | 2.6 | 11.2 |
| 0.9 | 3.1 | 12.8 |
| 1.0 | 3.9 | 14.6 |
| 1.1 | 5.2 | — |

The ID-2 curve **plateaus at γ ≈ 1.2** from roughly 11-12 min; ID-11 is still rising at 15 min.
D/logE straight-line gradients: **γ(ID-2, 8 min) ≈ 1.19**, **γ(ID-2, 4 min) ≈ 1.04**
(gamma-time curve gives 1.15 and 1.00 — agreement within 0.04).

Development table, printed p.195 / PDF 202: dish 7½ / 5 / 3½ min, tank 15 / 10 / 7 min at 55/65/75°F.

### 1.4 Other H.P.3 items — (a)

* Printed p.77 / PDF 84 — plates "in order of decreasing contrast": ... Selochrome, Press Ortho
  Series 2, **H.P.3**, Soft Gradation Panchromatic, Golden Iso Zenith, Hypersensitive Panchromatic.
  Flat films decreasing contrast: ... Selochrome, **H.P.3**, Portrait Panchromatic, Hyperchromatic,
  Portrait Ortho Fast. (H.P.3 is near the SOFT end of both lists.)
* Printed p.96 / PDF 103 — Ilford filter exposure-factor tables; indicating number 1 = H.P.3 Plate
  and F.P.3 Plate; indicating number 7 = Hypersensitive Panchromatic Plate, Hypersensitive
  Panchromatic (H.P.3) Roll Film, H.P.3 Film for Leica, H.P.3 16 mm, Series III Panchromatic Cine
  Negative 35 mm, H.P.3 Flat Film. (Confirms H.P.3 also existed as 16 mm and cine stock.)
* Printed p.154 / PDF 161 and p.159 / PDF 166 — exposure-calculator "Plate and Film Numbers" /
  "Emulsion Speed": Selo H.P.3 35 mm −3, Ilford H.P.3 Plate −3, Selo H.P.3 Roll Film −3,
  Ilford F.P.3 Plate −2, Ilford Hypersensitive Panchromatic Plate −1, Ilford Soft Gradation
  Panchromatic Plate +2, Selo Fine Grain Panchromatic Roll Film +3, Selo F.P.2 +3,
  Ilford Special Rapid Panchromatic Plate +4. (Relative log-exposure indices, not speed units.)
* Printed p.205 / PDF 212 — (a) graininess, qualitative only: "When used with a super-speed film such
  as the Selo H.P.3 it gives a graininess comparable with Selo Extra Fine Grain Panchromatic
  developed in ID-11."
* Printed p.406 / PDF 413 — (a) application note: for colour-separation work with repeating-back or
  one-shot cameras "a faster plate, such as the Hypersensitive Panchromatic or H.P.3, is preferred."

## 2. HPS — NOT PRESENT

Searches over the full OCR of all 492 pages for `H.P.S`, `HPS`, `H. P. S`,
`Hypersensitive Panchromatic Special`: **zero hits**. HPS is not in this book.

Trap to avoid: **"Hypersensitive Panchromatic" is a separate, slower 1942 product**
(plate 3500 H&D / 31 Scheiner / Group F) listed alongside H.P.3, and the flat film H.P.3 is
*titled* "Hypersensitive Panchromatic H.P.3". Neither is HPS. Do not use either as HPS evidence.

## 3. All other named Ilford/Selo materials with printed speeds — (a)

PLATES, printed p.74 / PDF 81 (H. & D. | Scheiner | Ilford Speed Group):

| Plate | H. & D. | Scheiner | Group |
|---|---|---|---|
| H.P.3 | 6000 | 34 | G |
| Hypersensitive Panchromatic | 3500 | 31 | F |
| F.P.3 | 4500 | 32 | F |
| Soft Gradation Panchromatic | 1200 | 28 | E |
| Special Rapid Panchromatic | 700 | 25 | D |
| Rapid Process Panchromatic | — | — | — |
| Thin Film Half-tone Panchromatic | — | — | — |
| Press Ortho Series 2 | 3500 | 31 | F |
| Selochrome | 1500 | 29 | E |
| Golden Iso-Zenith | 1400 | 29 | E |
| Iso-Zenith | 700 | 25 | D |
| Chromatic | 135 | 18 | A |
| Special Rapid | 270 | 21 | B |
| Ordinary | — | — | — |

Printed p.75 / PDF 82 also lists Process, Thin Film Hair-tone, Special Lantern, Warm Black Lantern
and Gaslight Lantern plates with no speed figures.

FLAT FILMS, printed p.75-76 / PDF 82-83:

| Flat film | H. & D. | Scheiner | Group |
|---|---|---|---|
| Hypersensitive Panchromatic H.P.3 | 5000 | 32 | F |
| Portrait Panchromatic | 2000 | 30 | E |
| Process Panchromatic | — | — | — |
| Selochrome | 1500 | 29 | E |
| Hyperchromatic | 1500 | 29 | E |
| Portrait Ortho Fast | 700 | 25 | D |
| Commercial Ortho | — | — | — |
| Fine Grain Ordinary | — | — | — |
| Process Film | — | — | — |
| Line Film | — | — | — |
| Diapositive | — | — | — |

SELO ROLL FILMS, printed p.76 / PDF 83:

| Roll film | H. & D. | Scheiner | Group |
|---|---|---|---|
| Selo H.P.3 | 5000 | 32 | F |
| Selo F.P. (Fine Grain Panchromatic) | 1000 | 27 | D |
| Selochrome | 1500 | 29 | E |
| Selo Ortho ("Excellent latitude") | 750 | 26 | D |

35 mm MINIATURE CAMERA FILMS, printed p.76 / PDF 83:

| Film | H. & D. | Scheiner | Group |
|---|---|---|---|
| Selo H.P.3 | 5000 | 32 | F |
| Selo F.P.2 | 1000 | 27 | D |
| Micro-neg ("extremely high resolving power with high contrast") | — | — | — |
| Diapositive | — | — | — |

### 3.1 Characteristic curves printed for 12 NAMED products — (a)

Printed p.79 / PDF 86 "CHARACTERISTIC CURVES OF SOME ILFORD PLATES" — H.P.3 Plate, Soft Gradation
Panchromatic Plate, Special Rapid Panchromatic Plate (these three exposed to half-watt light except
H.P.3 = international sunlight; Soft Gradation and Special Rapid legends say half-watt),
Press Ortho Series 2 Plate, Selochrome Plate, Golden Iso-Zenith Plate (international sunlight).

Printed p.80 / PDF 87 "CHARACTERISTIC CURVES OF SOME ILFORD AND SELO FILMS" — H.P.3 Film
(half-watt), Portrait Panchromatic Film (half-watt, 2850°K), Hyperchromatic Film (half-watt),
Selo H.P.3 Roll Film, Selo F.P. Roll Film, Selochrome Roll Film (international sunlight).

All twelve carry, verbatim in the page header: "Developed in Ilford Metol Hydroquinone Developer ID-2
at dish strength, with Gamma Time Development Curves for both ID-2 and ID-11 Developers", and each
graph plots D vs relative log E for **4 and 8 minutes at 65°F** plus a gamma-vs-development-time
subplot (0-15 min) for ID-2 and ID-11.
Only the three H.P.3 graphs were digitised to a verified standard here; the other nine are legible in
the rendered images and could be digitised the same way if needed.

### 3.2 Resolving power — (a), but NOT for H.P.3

Printed p.73 / PDF 80, verbatim, "for unfiltered half-watt light":
* Gaslight lantern plate — 3,150 lines/inch
* Process films and plates — 3,650 lines/inch
* Alpha lantern plate — 3,800 lines/inch
* Thin film Half-tone plate — 4,200 lines/inch
* "For highest resolving power Ilford manufacture the H.R. plate which can resolve up to
  5,500 lines per inch."
No resolving-power figure is given for H.P.3 or any fast material.

### 3.3 Base / coating — (a) for the Ilford range, not H.P.3-specific

Printed p.71 / PDF 78: base is cellulose nitrate or acetate depending on product; **roll film on
3/1,000 in. base, X-ray 9/1,000 in., flat film generally 8/1,000 in., cine 5/1,000 in.**
Printed p.73 / PDF 80: flat-film notching — one V-cut for non-panchromatic, **two cuts for
panchromatic** materials, at the right of the top short side, emulsion facing operator.
Printed pp.72-73 / PDF 79-80: anti-halation backings; red backing for all non-panchromatic
materials, a special backing for panchromatic; backings dissolve out in the developer.

## 4. GENERIC / TYPICAL material — (b) — do NOT attribute to any product

* Printed pp.64-70 / PDF 71-77 — sensitometry theory: characteristic curve, gamma, effect of contrast
  on speed. Illustrative curves here belong to no product. **(b)**
* Printed pp.68-69 / PDF 75-76 — the three speed systems in use: **H. & D., Scheiner, Din**
  (Warnerke described historically). H&D taken from where the extended straight line cuts the base
  line; Scheiner = minimum visible image, Hefner amyl-acetate lamp at 1 m, sector wheel; Din = 40 W
  filtered lamp, 30-step wedge, 1/20 s constant, criterion 0.1 above fog. **(b)**
  No B.S. and no Weston numbers appear anywhere in the book (0 hits for both).
  No Din number is printed for any product — Din is only described as a system.
* Printed p.87 / PDF 94 — **Figs. 51, 52, 53**: spectral response curves labelled only
  "NON COLOUR SENSITIVE", "HIGHLY ORTHOCHROMATIC", "PANCHROMATIC", each "RESPONSE TO MEAN NOON
  SUNLIGHT", x-axis 4000-7000 Å (colour bands ultra-violet → red), y-axis "RELATIVE SENSITIVITY"
  with no numeric scale. The text states the panchromatic curve "may be taken as typical of
  panchromatic emulsions as a whole". **(b) — these are class curves, NOT H.P.3 data.**
* Fig. 49 (printed p.72) halation/irradiation diagram; Fig. 54 panchromatic-vs-eye visibility. **(b)**
* The word "spectrogram" occurs **nowhere** in the book; there are **no wedge spectrograms** and
  **no product-specific spectral-sensitivity curve** for H.P.3 or any other named material.

## 5. Rendered images kept (for audit)

Under `/sessions/vigilant-wonderful-dijkstra/mnt/outputs/`:
`ilf1942_pdf086_300dpi.png`, `ilf1942_pdf087_300dpi.png` (the two curve pages),
`hp3_plate_curve.png`, `hp3_plate_gamma.png`, `g_hp3flatfilm.png`, `g_selohp3roll.png`,
`ilf1942_p081_400.png`, `ilf1942_p082_400.png`, `ilf1942_p082_flatfilms.png`,
`ilf1942_p083_400.png`, `ilf1942_p085_recent.png`, `ilf1942_p202_300.png`, `ilf1942_p094.png`.

## 6. Verdict

The 1942 Ilford Manual **is genuine primary manufacturer documentation for H.P.3**, in plate,
flat-film and Selo roll/35 mm form, with published speeds, developer, dilution regime (dish/tank),
times at three temperatures, and printed characteristic + gamma-time curves. ILFORD_HP3 should no
longer be described as undocumented.

Caveats that must travel with the citation:
1. This is the **1942** H.P.3, an emulsion that was revised repeatedly over its life; it is not
   interchangeable with later HP3 or with HP4/HP5.
2. Speeds are **H. & D. and Scheiner and Ilford Speed Group** — quote as such; do not convert to
   ISO/ASA. The H&D figure differs between forms (plate 6000 vs film 5000) and is a manufacturer's
   rating from a period when H&D numbers were routinely inflated.
3. The gamma-time and D/logE numbers above are **my digitisation of printed graphs**, not printed
   tabulated values; treat as ±0.05 gamma / ±0.4 min. The two independent readings
   (gamma-time subplot vs D/logE straight-line gradient) agree to within 0.04 gamma for all three
   H.P.3 forms, which is the basis for that error bar.
4. Still missing for H.P.3: spectral sensitivity curve, resolving power, numeric graininess,
   numeric exposure latitude, Dmax/shoulder behaviour, and emulsion/coating detail.
5. **ILFORD_HPS gains nothing from this source** — HPS does not appear in it at all.
