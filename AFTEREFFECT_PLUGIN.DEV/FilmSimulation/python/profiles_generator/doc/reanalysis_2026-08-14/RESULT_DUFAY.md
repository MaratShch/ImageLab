# DUFAYCOLOR — extraction from PDF/PROFILES/DUFAYCOLOR/
Date 2026-08-14. Only printed content transcribed. Graph read-offs are flagged as such.

## Files present (4 PDFs + 1 mhtml + 3 jpgs)
| File | Pages | Text layer | Notes |
|---|---|---|---|
| Dufaycolor_Manual_1938_print.pdf | 25 (each = 1 landscape sheet holding 2 book pages, printed pp.1-48+index) | NONE (0 chars all pages) | pure scan, 1 raster image/page, `get_drawings()` = 0 on every page |
| dufaycolorbook00dufa.pdf ("THE DUFAYCOLOR BOOK", Dufay-Chromex Ltd, London, price 6d) | 60 | OCR text present | 2 images/page; `get_drawings()` = 0 everywhere; NO graphs at all |
| Carson_Dufaycolor_Kinotechnik_1934_conv_print.pdf | 3 | OCR text (German) | abstract in *Die Kinotechnik* Heft 15, 5 Aug 1934, pp.245-247, of W. H. Carson, J.SMPE XXIII(1), July 1934 |
| Dufay_Dufaycolor_GB000000262386A.pdf | 3 | NONE | GB patent 262,386, Louis Dufay + Cie d'Exploitation…, conv. date France 4 Dec 1925, GB appl. 28 Jul 1926, accepted 21 Apr 1927. No curves, no wavelengths, no geometry numbers. |

**VECTOR vs RASTER: every single curve in this folder is RASTER.** `page.get_drawings()` returns an
empty list for every page of every PDF; no path has >=25 items because no paths exist at all.
No curve can be digitised from vector coordinates.

---

## 1. SPECTRAL CURVES — what each one actually plots

### 1a. Fig. 4 "Transmission Of Dufaycolor Reseau"
FILE Dufaycolor_Manual_1938_print.pdf, PDF page 12 (right half) = printed page [21].
**QUANTITY PLOTTED: spectral TRANSMISSION of the three RÉSEAU FILTER ELEMENTS (blue, green, red dyed
mosaic elements).** It is NOT the emulsion's spectral sensitivity and NOT the emulsion-through-réseau
response. Three separate curves, one per réseau colour.
- Abscissa: "WAVE LENGTHS IN MILLIMICRONS", printed ticks 400 450 500 550 600 650 700 (mµ = nm).
  Vertical gridlines every 25 mµ.
- Ordinate: axis titled "TRANSMISSION" — **NO numeric scale, no tick values, no % marks.** 3 unlabelled
  horizontal gridlines only. The diagram is therefore qualitative/schematic in the ordinate.
- Band labels printed above the frame: VIOLET, BLUE, GREEN, ORANGE, RED.
- Raster. Rendered at 600 dpi and inspected.
- GRAPH READ-OFFS (my visual measurement against the printed gridlines, NOT printed numbers):
  blue element from below 400 to ~540 mµ, peak ~425-430; green from ~460 to ~630, peak ~525-530;
  red from ~570 rising to peak ~650 and still high at the 700 right edge (curve is cut off, not zero).
  Blue/green crossover ~500 mµ; green/red crossover ~600 mµ.
- Accompanying printed prose (same page): "The spectral transmissions of the blue, green and red
  reseau elements overlap to some extent as indicated in Fig. 4. The blue and red each overlap the
  green, while the green overlaps both blue and red. There is, however, a blue-violet portion of the
  spectrum which is not overlapped by the green; a green portion which is not overlapped by either
  blue or red elements; and a red portion which is not over[lapped by the green]."

### 1b. Carson 1934 — réseau dye transmission ranges, printed as prose (no figure reproduced)
FILE Carson_Dufaycolor_Kinotechnik_1934_conv_print.pdf, page 2.
"Aus einer schematischen Darstellung geht hervor, daß das Blaufilter zwischen 400 und 550, das
Grünfilter zwischen 475 und 625 und das Rotfilter zwischen 550 und 700 m/j [mµ] durchlässig ist."
=> blue 400-550, green 475-625, red 550-700 mµ. **These are the RÉSEAU FILTER DYE transmission
ranges** (the German text explicitly calls the source a *schematic* representation). Independently
corroborates Fig. 4 read-offs. Same page notes Dufay deliberately uses broadly OVERLAPPING filters,
not narrow-band ones, for higher total transmission and wider exposure/development latitude.

### 1c. Fig. 5 "Transmission Of 'S' Separation Filters"
FILE Dufaycolor_Manual_1938_print.pdf, PDF page 13 (left half) = printed page [22].
**QUANTITY PLOTTED: transmission of the S1/S2/S3 SEPARATION FILTERS (accessory gelatin filters sold
for making colour-separation negatives FROM a finished Dufaycolor transparency).** Nothing to do with
the film's own sensitivity or its réseau. Three narrow humps peaking ~430, ~550, ~655 mµ (read-off).
Abscissa 400-700 "WAVE LENGTHS IN MILLIMICRONS"; ordinate "TRANSMISSION" again with NO numeric scale.
Raster. Filter list on same page: S1 Red / S2 Green / S3 Blue (contact separations);
P1 Red / P2 Green / P3 Blue (enlargement separations); P4 Yellow (gray printer).

### 1d. THERE IS NO SPECTRAL SENSITIVITY CURVE ANYWHERE
No figure in any of the 4 PDFs plots sensitivity (or log sensitivity) against wavelength — not for the
emulsion alone, not for the emulsion seen through the réseau. Checked by: full text grep of the two
OCR'd PDFs; 5x5 contact sheet of all 25 manual sheets; 8x8 contact sheet of all 60 book pages;
`get_drawings()`=0 everywhere. The complete figure inventory of the 1938 manual is:
Fig.1 cross-section of film, Fig.2 photomicrographs of réseau, Fig.3 schematic table of colour
reproduction, Fig.4 réseau transmission, Fig.5 S-filter transmission, Fig.6/7 copying set-up
diagrams, Fig.8 réseau vs half-tone screen comparison, Fig.9 generic characteristic curve,
Fig.10 generic gamma/time curve. The Dufaycolor Book contains NO graphs whatsoever.

---

## 2. SPEED / EXPOSURE FIGURES (verbatim, units as printed, NOT converted)

Dufaycolor_Manual_1938_print.pdf, PDF page 5 right half = printed page [7]:
> "Dufaycolor film has a speed of approximately one-half that of standard panchromatic film or
> one-quarter of super-sensitive panchromatic film. … **Dufaycolor, without a filter and in daylight,
> has a Weston factor of 8 and a Scheiner rating of 18.**"
This is the speed of the WHOLE FILM as exposed through base+réseau (the film is exposed through the
base — Fig. 1 "EXPOSE THIS SIDE"), i.e. emulsion-behind-réseau, not the bare emulsion.

Dufaycolor_Manual_1938_print.pdf, PDF page 6 left half = printed page [8], table
"DUFAYCOLOR FILTERS AND FILM SPEEDS IN ARTIFICIAL LIGHT":
| LIGHT SOURCE | FILTER | WESTON | SCHEINER |
|---|---|---|---|
| Dufaycolor Wonderlite | No Filter | 12 | 20 |
| Photoflood | Photoflood | 3 | 14 |
| High Wattage Mazdas and Projection Lamps | H. W. M. | 2 | 12 |
| Photoflash Bulb | Flash | — | — |

dufaycolorbook00dufa.pdf, PDF page 22 = printed page 14, meter-setting table (OCR; ° read as 0/o
by the OCR, degree signs restored where obvious):
Avo "400 H. & D."; Blendux "170 [=17°] or Class C"; Bewi "17° Scheiner"; Electrodrem "24° Scheiner";
Ilford "Group C"; Leicameter "17 Weston Scheiner"; Ombrux "17° Scheiner"; Photoscop "17° Scheiner";
Prinsen "400 H. & D."; Sixtus "15/10 DIN in full sunlight, 9/10 DIN other exposures and indoor";
Tempiphot "19° Scheiner"; Weston "8 Normal subject"; Leudi "11° Scheiner";
Justophot "16—19° Scheiner"; Justodrem "20—23° Scheiner"; Wynne's Infallible "F78";
Watkin's Bee "65".
NOTE: these are per-meter SETTINGS, and they disagree with each other (17°-24° Scheiner);
they are not a single manufacturer speed rating. The Weston 8 agrees with the 1938 manual.
Same page: for indoor artificial light, meter reading off white blotting paper, "the exposure
indicated is multiplied by 40" for Half-Watt or Photoflood.

dufaycolorbook00dufa.pdf, PDF page 38 = printed page 28, stops for open subjects in bright sunlight:
Winter f/3.5; Spring and Autumn f/4.5; Summer ordinary sunshine f/5.6; Summer bright sunshine f/[6.3?];
Summer open sea f/8 (OCR partially garbled for two entries).
"the lens aperture should be opened to one stop larger than … a normal panchromatic film and to two
stops larger than for Hypersensitive Panchromatic Film."

Carson_Dufaycolor_Kinotechnik_1934, page 3: 16 mm version, "Aufnahmen sollen bei gutem Wetter mit
Objektiven der relativen Oeffnung f : 3,5 ausführbar sein; für die Projektion wird ein Mehrbedarf an
Licht von 20 Prozent gegenüber Schwarzweiß-Filmen angegeben."

## 3. CHARACTERISTIC CURVE / GAMMA / DENSITY

### Fig. 9 "Characteristic Curve of a Typical Panchromatic Emulsion"
Dufaycolor_Manual_1938_print.pdf, PDF page 20 left half = printed page [34]. RASTER.
**QUANTITY PLOTTED: Density (ordinate, 0 to 1.8, ticks .2 .4 .6 .8 1.0 1.2 1.4 1.6 1.8) vs
LOG EXPOSURE (abscissa, 0 to 2.0, ticks 0 .2 .4 .6 .8 1.0 1.2 1.4 1.6 1.8 2.0).**
*** THIS IS A GENERIC TEXTBOOK ILLUSTRATION, NOT A MEASUREMENT OF DUFAYCOLOR. *** The title says
"a Typical Panchromatic Emulsion"; the curve is annotated TOE, STRAIGHT LINE PORTION, SHOULDER,
INERTIA, DENSITY RANGE, GAMMA = TAN θ, with construction points A/B/C. It is the sensitometry-101
diagram accompanying the appendix text (D = Log O, O = 1/T). It must NOT be treated as a Dufaycolor
characteristic curve.

### Fig. 10 "Typical Gamma/Time Curve"
Dufaycolor_Manual_1938_print.pdf, PDF page 22 left half = printed page [38]. RASTER.
Ordinate GAMMA .2 to 2.0 (ticks every .2); abscissa TIME IN MINUTES 1 to 10. Curve runs from
gamma 0.6 at ~1.3 min to ~1.83 at 10 min (read-off).
*** ALSO GENERIC, AND EXPLICITLY NOT DUFAYCOLOR. *** Printed text immediately below, verbatim:
> "TIME OF DEVELOPMENT — **Dufaycolor film is always developed to gamma infinity**, and the gamma
> time curve is only of real use in connection with the making of separation negatives on the
> materials and with the formulae already discussed."
So the manufacturer states there is no finite working gamma for Dufaycolor itself.

### Density figure for the duplicating/print stock (not the camera film)
Carson_Dufaycolor_Kinotechnik_1934, page 2: "**Die maximale Schwärzung beträgt etwa 1,2** und der
Abstufungsbereich soll sehr ausgedehnt sein." — maximum density ≈1.2, referring to the thinner-coated
DUPLICATING emulsion used for release prints from the reversal original, not the camera stock.
Same page: "die Erfahrung gezeigt, daß der Einfluß der Wellenlänge auf das Gamma vernachlässigbar
ist, vorausgesetzt, daß man es bei Gamma-unendlich mißt" — the influence of wavelength on gamma is
negligible provided it is measured at gamma-infinity; an emulsion was chosen "deren Schwärzungskurve
einen ausgedehnten gradlinigen Teil bei kurzem Schwanzstück aufweist" (long straight-line portion,
short toe). No numbers, no curve.

## 4. RÉSEAU GEOMETRY (all printed values, with the internal disagreement noted)

- Dufaycolor_Manual_1938 PDF p2 left half = printed [2]: "a set of parallel ink lines is printed
  thereon, **there being about 500 of these lines to the inch**." Then: second set of ink lines
  "at right angles to the first"; "The reason for making the red lines narrower is to equalise
  approximately the areas of red, blue and green" (that clause is from the Book, see below).
- Dufaycolor_Manual_1938 PDF p16 left half = printed [28], Fig. 8: side-by-side discs,
  "**DUFAYCOLOR SCREEN 1000 lines per inch**" vs "HALF TONE SCREEN 120 lines per inch",
  "**Magnification 72 Diameters**".
  => the 500 vs 1000 lines/inch discrepancy is the ink-line-repeat count vs lines+spaces count.
- Carson 1934 p2 resolves it: "eine stählerne Druckwalze **1000 Linien je Zoll, d. h. 500 Farblinien
  mit der gleichen Zahl Zwischenräume**" (1000 lines per inch = 500 colour lines + 500 spaces).
- dufaycolorbook00dufa PDF p9 = printed p3: "the complete colour pattern (of three colours) is
  reproduced **twenty times per millimetre (500 times per inch)** for some types of material, and
  **23 times per millimetre (600 times per inch)** for other types of material."
  => TWO réseau pitches existed. (23/mm is ~584/in; the book's own inch figures are rounded.)
- dufaycolorbook00dufa PDF p10 = printed p4: "**there being twenty lines to the millimetre and the
  spaces between being equal in width to the lines**"; second set of lines "at right angles to the
  first. The lines this time are **broader** than in the first instance and the spaces between are
  **narrower** so that there are the same number of lines per millimetre as before. … The reason for
  making the **red lines narrower is to equalise approximately the areas of red, blue and green**."
- Carson 1934 p1-2: "etwa **eine Million Rasterelemente auf einen Quadratzoll**" (~1,000,000 réseau
  elements per square inch); pattern = "blauen und roten Vierecken und einer grünen Linie"
  (blue and red squares + a green line); second ruling at "**etwa 45 Grad**" to the first, chosen
  for non-optical reasons though "theoretisch nicht der günstigste"; historical progression
  "Man hielt früher **15 Linien auf den Millimeter** für die Grenze … doch ist es … gelungen,
  **19 Linien auf den Millimeter** … unterzubringen"; 16 mm version to get **400 lines per inch**.
- *** GEOMETRY CONFLICT, unresolved by these sources: *** the 1938 manual and the Book say the second
  ruling is at RIGHT ANGLES (90°) to the first, and give the pattern as blue + green SQUARES with RED
  LINES. Carson 1934 says ~45° and gives blue + red SQUARES with a GREEN LINE. Different vintages
  and/or cine-vs-still product. Do not merge.
- Layer order (1938 Fig. 1, printed [2], "A Greatly Magnified Cross Section Of Dufaycolor Film"):
  EXPOSE THIS SIDE -> Transparent Film Base -> Reseau (Color Filter Screen) -> Light Sensitive
  Emulsion. Book Fig. 1 (printed p4) adds: base / réseau / **thin varnish layer** / emulsion /
  black protective paper cover / adhesive tape.

## 5. FILTER FACTORS, EXPOSURE TABLES, DEVELOPMENT TIMES & TEMPERATURES

### Reversal processing of Dufaycolor itself
Dufaycolor_Manual_1938 PDF p7 right half = printed [11], "DIRECTIONS FOR DEVELOPMENT BY REVERSAL":
green safelight Wratten Series III permissible, film protected from direct rays; white light allowed
after 2 min in the bleach. "The first development is given in the following bath at a temperature of
**68° F**." FIRST DEVELOPER (Metric / Avoirdupois): Metol 1 g / 16 gr; Hydroquinone 8 g / 128 gr;
Sodium Sulphite dry 50 g / 1¾ oz; Sodium Carbonate dry 35 g / 1¼ oz; Potassium Bromide 5 g / 80 gr;
Potassium Thiocyanate (Sulphocyanate) 9 g / 144 gr; Water 1,000 ccs / 35 oz.
Dufaycolor_Manual_1938 PDF p8 left half = printed [12] — first-development times for correct exposure:
**65° F. 5 minutes; 68° F. 4¼ minutes; 72° F. 4 minutes; 75° F. 3 minutes.**
Then: wash 1 min running water, or ½ min STOP BATH (Acetic Acid 28% 50 ccs / 1¾ oz; Water 1,000 ccs /
35 oz) followed by 2 min washing. BICHROMATE BLEACHING BATH: Potassium Bichromate 5 g / 80 gr;
Sulphuric Acid conc. sp.gr. 1.87 10 ccs / 160 minims; Water 1,000 ccs / 35 oz — "Bleach until the
image is clearly visible, which will require **4 minutes** and then wash for **2 minutes**, after
which the film should be cleared for **2 minutes**." Permanganate bleach gives more brilliant colours
but softens the film above **70° F** (use hardener first).

dufaycolorbook00dufa PDF p27 = printed p17 (an EARLIER/different regime — note the disagreement with
the 1938 manual above): Formula A "development should be continued for **three minutes at 65[°F]**
(two-and-a-half minutes at **70°F** or two minutes at **75°F**)"; Formula B "allow **20 per cent.
longer** time in each case", B "gives slightly more contrasty results than A". Two schemes offered:
CONSTANT FIRST DEVELOPMENT and FACTORIAL DEVELOPMENT. "the speed of the emulsion is exceptionally
high"; uniform fog "causes a general flattening of gradation and colour, together with a loss of
density."
dufaycolorbook00dufa PDF p32 = printed p22: redevelopment in used first developer full strength,
"**four to five minutes**"; fix **2 minutes**; wash **about fifteen minutes**; hardening bath
2½% solution above 65°F.

### Filter factors — ALL of them are for the SEPARATION MATERIAL, none for Dufaycolor
Dufaycolor_Manual_1938 PDF p14 right half = printed [25], "EXPOSURE RATIOS FOR SEPARATION NEGATIVES
WITH DUFAYCOLOR FILTERS" (red = 1 throughout; "S" series then "P" series, Blue/Green/Red):
Eastman S.S.Pan.: White Flame Arc P 4 / 2.75 / 1; Mazda S 7 / 16 / 1, P 15 / 6.5 / 1.
Agfa Isopan: Arc P 1.5 / 2 / 1; Mazda S 2.5 / 10 / 1, P 6 / 4 / 1.
Defender XF Pan.: Arc P 2.5 / 1.5 / 1; Mazda S 4 / 7.5 / 1, P 10 / 3.2 / 1.
Wratten Pan. Plates: Arc P 3 / 1.75 / 1; Mazda S 5 / 9 / 1, P 12 / 3.5 / 1.
Gevaert Normal Pan. Plates: Arc P 2.5 / 1 / 1; Mazda S 3.5 / 6 / 1, P 11 / 3 / 1.
Ilford S.G. Pan. Plates: Arc P 3.5 / 2 / 1; Mazda S 5 / 12 / 1, P 12.5 / 4 / 1.
Same page: "As the blue sensation (yellow printer) invariably tends to be soft, from **25% to 100%
extra development time** must be given to it".
dufaycolorbook00dufa PDF p52 = printed p40, filter factors with tricolour red = unity, on Ilford
Special Rapid Panchromatic Plates: Half-watt 1 / 4 / 3½ / 3 (red / green / blue / gamma filter);
Open Arc white-flame carbons 1 / 4 / ? / 1 (OCR garbled two cells). For screen negatives on Process
Panchromatic Plates, Open Arc: 1 / 1½ / 3 / 1. Blue-filter negative needs "about double the
development time"; recommended instead to shoot the yellow printer on a non-colour-sensitive plate
(Ilford Ordinary) with an Ilford filter to kill UV, at "about twice" the red-filter exposure.

## 6. EMULSION SENSITISATION — what is actually stated
- Dufaycolor_Manual_1938 printed [2] (PDF p2 left): "A **panchromatic** emulsion of **extremely high
  sensitivity** is coated over it".
- Dufaycolor_Manual_1938 printed [7] (PDF p5 right): "The **extreme sensitivity of Dufaycolor film to
  light of all colors** makes it essential that the loading and unloading be done in complete
  darkness or with the aid of a very dim **green panchromatic safelight** at least three feet away."
  Also: "Panchromatic emulsions must be standardized to some constant balance of color values, and
  normal daylight is customarily accepted as standard white light." — i.e. the stock is
  DAYLIGHT-BALANCED, with compensating filters for other sources.
- dufaycolorbook00dufa printed p4 (PDF p10): "a coating of a **special very highly sensitive
  panchromatic emulsion**".
- Carson 1934 p2: single panchromatic emulsion behind the réseau; a method was found to isolate the
  réseau dyes from the emulsion "daß sie keine desensibilisierende Wirkung auf diese auszuüben
  vermögen" (so they exert no desensitising action on it); "Dank der beiden Faktoren: gesteigerte
  Empfindlichkeit der modernen panchromatischen Emulsionen und verbesserte Filterfarbstoffe" the film
  reached a speed previously thought unattainable for screen-plate films. Also states an emulsion was
  chosen with "für alle drei Grundfarben tunlichst ausgeglichene Farbenempfindlichkeit und Gradation"
  (colour sensitivity and gradation as balanced as possible across all three primaries) — a
  qualitative claim of a flattish panchromatic response, with NO curve and NO numbers.
  Also: no single emulsion suited daylight + arc + incandescent without a filter, so the CINE
  emulsion was balanced for artificial light.
- Carson 1934 p3: reversal processing standardised; first developer
  "ein Metol-Hydrochinon-Ammoniakentwickler ziemlich hoher Konzentration und auf **18° C**
  temperiert"; after silver dissolution and second exposure, second development in an ordinary
  metol-hydroquinone developer in full light.
- **NO source in this folder names the sensitising dyes.** The only dyes named anywhere are in GB
  262,386 p2 — "rhodamine, fuschine, safranine, auramine, malachite green, methylene blue, carmine
  blue" — and those are candidate SCREEN/RÉSEAU dyes in a 1925 patent, not emulsion sensitisers, and
  in that patent the screen colours are violet/green/orange, not R/G/B.

## 7. VERDICT for stock DUFAYCOLOR_1937
- A `spectral_sensitivity` block **cannot** be populated from these files. There is no spectral
  sensitivity curve of any kind — neither emulsion-alone nor emulsion-through-réseau — in any of the
  four PDFs.
- What CAN be added is a **réseau spectral transmission** entity (3 curves) from
  Dufaycolor_Manual_1938_print.pdf printed p.[21] Fig. 4, wavelength 400-700 mµ, but with
  **ordinate unscaled** — so only band edges / crossovers / peak positions are recoverable, and only
  as graph read-offs from a raster scan. Carson 1934 gives the same information as printed numeric
  ranges (blue 400-550, green 475-625, red 550-700 mµ) and is the better citation because those are
  printed figures rather than read-offs. Both sources call the diagram schematic.
- Speed IS available and citable verbatim: Weston 8 / Scheiner 18 (daylight, no filter) and the
  four-row artificial-light table (Weston 12/3/2, Scheiner 20/14/12) — 1938 manual pp.[7],[8].
  Flag as whole-film (through-base, through-réseau) speed.
- Gamma / characteristic curve: **do NOT populate.** Both curve figures are generic textbook
  illustrations, and the manual states outright that Dufaycolor "is always developed to gamma
  infinity". The only density number (Dmax ≈ 1.2, Carson 1934) belongs to the duplicating stock.
- Réseau geometry IS available: ~500 colour lines/inch = 1000 lines+spaces/inch = 20/mm, with a
  second 23/mm (~600/in) variant; ~10^6 elements per square inch; equalised R/G/B areas achieved by
  narrowing the red lines; crossing angle stated as 90° (1938 manual + Book) but ~45° in Carson 1934
  — record both, do not average.
