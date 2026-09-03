# Provenance repair — 11 stocks with manufacturer datasheets in PDF/PROFILES
Root: `PDF/PROFILES/`. Every value below is transcribed verbatim from the cited FILE + PAGE.
Nothing is estimated, converted or inferred. "NOT PRINTED" = the sheet does not give it.

General verification result: **all 12 proposed pairings are correct** (each PDF's masthead names
the exact product/code). No mis-pairings found.

Vector-drawing note: for every Kodak sheet, page 1 and the last page contain 12–23 paths with
>=25 items whose bounding boxes are ~25x35 pt in the header/footer corner — these are the Kodak
corporate logo/letterhead artwork, NOT data. Only the graph pages listed per stock carry
extractable curve geometry.

---

## EASTMAN_EXR_50D_5245
FILE `KODAK/5245.pdf` — KODAK Publication **H-1-5245t**, **May 2003**, ©Eastman Kodak 2003 (p1).
Masthead: "EASTMAN EXR 50D Film / 5245, 7245" (p1). VERIFIED.

- Exposure index (p1): Daylight — 50; Tungsten (3200 K) — 12. Footnote 1 (p1): "With a KODAK
  WRATTEN Gelatin Filter No. 80A." → the tungsten EI of 12 is a **conversion-filter (No. 80A)
  figure, not a film property**. Film is daylight-balanced.
- Colour-balance table (p1) also prints an anomalous row "Tungsten (3200 K) | None | 50" directly
  beneath "Tungsten (3200 K) | WRATTEN Gelatin No. 80A | 12". Verified in layout-preserving
  extraction; apparent Kodak misprint (row is duplicated from the Daylight row). Do not use.
- Balance colour temperature (p1): daylight; table row "Daylight (5500 K) | None | 50" → **5500 K**.
- Diffuse RMS granularity (p3): "Refer to curve." Aperture: **48-micrometer**, read with a
  microdensitometer (red, green, blue). No single numeric value printed.
- Resolving power: NOT PRINTED (neither 1.6:1 nor 1000:1).
- Contrast/gamma: NOT PRINTED. Dmin: NOT PRINTED numerically. Dmax: NOT PRINTED.
- Reciprocity (p2): "You do not need to make any filter corrections or exposure adjustments for
  exposure times from 1/1000 to 1 second." No CC filter → **achromatic over 1/1000–1 s** (no
  statement outside that range).
- Process (p2): **ECN-2**. Densitometry (p3, p4): **Status M**.
- Base (p1): "clear acetate safety base with **rem-jet backing**". Base thickness NOT PRINTED.
- Mask (p1): "The emulsion contains a colored-coupler mask for good color reproduction in release
  prints." Orange-mask density NOT PRINTED numerically.
- Interimage / DIR coupler: NOT PRINTED.

Vector curve pages (paths with >=25 items):
- p3 — 4 paths (128, 126, 64, 54): characteristic (sensitometric) curves R/G/B + modulation-transfer
  curves R/G/B.
- p4 — 10 paths (213, 152, 139, 128, 127, 121, 113, 94, 89, 64): diffuse RMS granularity curves,
  **spectral sensitivity curves (highest value)**, and spectral dye-density curves.

NOT PRINTED: resolving power (both contrasts), gamma/contrast index, Dmin, Dmax, base thickness,
interimage/DIR statement, numeric mask density, numeric rms value.

---

## EASTMAN_EXR_100T_5248
FILE `KODAK/5248.pdf` — KODAK Publication **H-1-7248**, **March 1999** ("Minor Revision 3-99", p4),
©Eastman Kodak 1999. Masthead: "EASTMAN EXR 100T Color Negative Film 5248 / 7248" (p1). VERIFIED
(publication number is the 7248-series code but the sheet covers 5248 explicitly).

- Exposure index (p1): Tungsten (3200 K) — 100; Daylight — 64. Footnote (p1): "With a KODAK
  WRATTEN Gelatin Filter No. 85." → daylight EI 64 is a **conversion-filter (No. 85) figure**.
- Balance colour temperature (p1): "balanced for exposure with tungsten illumination (**3200 K**)";
  tolerance ±150 K without correction filters.
- Diffuse RMS granularity (p2): **Less than 5**; "Read at a net diffuse visual density of 1.0,
  using a **48-micrometre aperture**."
- Resolving power (p2): **TOC 1.6:1 — 80 lines/mm; TOC 1000:1 — 160 lines/mm** (method per
  ISO 6328-1982).
- Contrast/gamma: NOT PRINTED. Dmin/Dmax: NOT PRINTED numerically.
- Reciprocity (p2): "No filter or exposure adjustments are needed for exposure times from 1/1000 to
  1/10 second. For a 1-second exposure, increase exposure by 1/3 stop." **No CC filter specified**
  → achromatic as printed (no explicit "none" wording).
- Process (p1, p2): **ECN-2**. Densitometry (p3): **Status M**.
- Base (p1): "acetate safety base with **rem-jet backing**". Thickness NOT PRINTED.
- Mask (p1): "colored-coupler mask". Numeric mask density NOT PRINTED.
- Interimage / DIR coupler: NOT PRINTED.

Vector curve pages:
- p3 — 11 paths (291, 229, 205, 186, 142, 134, 132, 94, 94, 56, 26): characteristic curves,
  **spectral-sensitivity curves (highest value)**, spectral-dye-density curves, diffuse RMS
  granularity curves, modulation-transfer curves. (All five graphs live on p3.)

NOT PRINTED: gamma/contrast, Dmin, Dmax, base thickness, interimage/DIR, numeric mask density,
CC filter for reciprocity.

---

## EASTMAN_EXR_200T_5293
FILE `KODAK/5293.pdf` — KODAK Publication **H-1-5293t**, **August 2003**, ©Eastman Kodak 2003 (p1).
Masthead: "EASTMAN EXR 200T Film / 5293, 7293" (p1). VERIFIED.
(Note: p2 of this PDF has a corrupt content stream; the LAD/IDENTIFICATION paragraph is garbled.
The reciprocity and processing text on p2 extracted cleanly and is quoted below.)

- Exposure index (p1): Tungsten (3200 K) — 200; Daylight — 125. Footnote 1 (p1): "With a KODAK
  WRATTEN Gelatin Filter No. 85." → daylight EI 125 is a **conversion-filter (No. 85) figure**.
- Balance colour temperature (p1): tungsten **3200 K**, ±150 K without correction filters.
- Diffuse RMS granularity (p3): "Refer to curve." Aperture **48-micrometer**, microdensitometer
  (red, green, blue). No numeric value.
- Resolving power: NOT PRINTED.
- Contrast/gamma, Dmin, Dmax: NOT PRINTED.
- Reciprocity (p2): "You do not need to make any filter corrections or exposure adjustments for
  exposure times from 1/1000 to 1 second." No CC filter → **achromatic over 1/1000–1 s**.
- Process (p2): **ECN-2**. Densitometry (p3 characteristic curve; p4 spectral sensitivity):
  **Status M**; spectral sensitivity measured at D = 0.4 above D-min, effective exposure 0.013 s.
- Base (p1): "clear acetate safety base with **rem-jet backing**". Thickness NOT PRINTED.
- Mask (p1): "colored-coupler mask". Numeric density NOT PRINTED.
- Interimage / DIR coupler: NOT PRINTED.

Vector curve pages:
- p3 — 5 paths (129, 126, 52, 49, 31): characteristic curves + modulation-transfer curves.
- p4 — 14 paths (138, 127, 126, 125, 125, 124, 117, 63, ...): diffuse RMS granularity curves,
  **spectral sensitivity curves (highest value)**, spectral dye-density curves.

NOT PRINTED: resolving power, gamma, Dmin, Dmax, base thickness, interimage/DIR, numeric mask
density, numeric rms value.

---

## EASTMAN_EXR_500T_5296
FILE `KODAK/eastman 500t 5296 exr - Kodak.pdf` — MPTVI Data Sheet **TI1664**, **Reissued 6-92**,
©Eastman Kodak 1992 (p1). Masthead: "EASTMAN EXR 500T Film 5296, 7296" (p1). VERIFIED.
Plain-text TInet sheet: **zero drawings and zero images on all 6 pages** — no curves at all.

- Exposure index (p1): Tungsten (3200 K) — **500/28 DIN**; Daylight — **320/26 DIN**. Footnote (p2):
  "With a KODAK WRATTEN Gelatin Filter No. 85." → daylight 320 is a **conversion-filter figure**.
- Balance colour temperature (p2): tungsten **3200 K**, ±150 K without correction filters.
- Diffuse RMS granularity (p5): "Refer to curve." (no curve is present in this text sheet).
  Aperture: **48-micrometre**, microdensitometer (red, green, blue). No numeric value.
- Resolving power (p5): **ISO RPL 50 lines/mm (TOC 1.6:1); ISO RP 100 lines/mm (TOC 1000:1)**
  (method per ISO 6328-1982).
- Contrast/gamma, Dmin, Dmax: NOT PRINTED.
- Reciprocity (p3): "You do not need to make any filter corrections or exposure adjustments for
  exposure times from 1/1000 to 1 second." No CC filter → **achromatic over 1/1000–1 s**.
- Process (p3): **ECN-2**. Densitometry status: NOT PRINTED (no curves in this sheet).
- Base (p1): "clear acetate safety base with **rem-jet backing**". Thickness NOT PRINTED.
- Mask (p1): "The emulsion contains a colored-coupler mask". Numeric density NOT PRINTED.
- Interimage / DIR coupler: NOT PRINTED.

Vector curve pages: **none** — 0 drawings on every page (p1–p6).

NOT PRINTED: gamma, Dmin, Dmax, densitometry status, base thickness, interimage/DIR, numeric mask
density, numeric rms value, any curves.

---

## KODAK_VISION2_200T_5217
FILE `KODAK/5217-Vision2-200T.pdf` — KODAK Publication **H-1-5217**, CAT 185 7119, ©Eastman Kodak
2004, **Revised 10-2005** (p4). Masthead: "KODAK VISION2 200T Color Negative Film 5217 / 7217" (p1).
VERIFIED. This is the 4-page brochure-format sheet (less tabulated data than the H-1-…t format).

- Exposure index (p2): Tungsten (3200 K) — 200; Daylight (5500 K) — 125 "(with KODAK WRATTEN
  Gelatin Filter No. 85)" → daylight EI 125 is a **conversion-filter (No. 85) figure**.
- Balance colour temperature (p2): tungsten **3200 K**, ±150 K without correction filters.
- Diffuse RMS granularity: NOT PRINTED numerically — only "the measured granularity is
  exceptionally low" (p3) plus a granularity curve. **Measuring aperture NOT PRINTED.**
- Resolving power: NOT PRINTED.
- Contrast/gamma, Dmin, Dmax: NOT PRINTED.
- Reciprocity (p3): "No filter corrections or exposure adjustments for exposure times from 1/1000 of
  a second to 1/10 second. In the 1-second range, increase exposure 2/3 stop and use a KODAK Color
  Compensating **Filter CC 10R**. In the 10 second range, increase exposure 1 stop and use a KODAK
  Color Compensating **Filter CC 10R**." → **chromatic reciprocity failure** (red-deficient) beyond
  1/10 s.
- Process (p2): **ECN-2**. Densitometry status: **NOT PRINTED** on this sheet.
- Base (p2): "Acetate safety base with **rem-jet backing**". Thickness NOT PRINTED.
- Mask / interimage / DIR: NOT PRINTED.

Vector curve pages:
- p3 — 25 paths (164, 142, 132, 117, 114, 114, 104, 97, 97, 95, 91, 82, 75, 64, 55, ...): all five
  graphs are vector on this page — sensitometric curves, modulation-transfer curves, diffuse RMS
  granularity curves, **spectral-sensitivity curves (highest value)**, spectral dye-density curves.
- p1/p2/p4 each carry 2 paths of 28 items with bbox (-30,-30,630,828) = full-page background/frame
  artwork, not data.

NOT PRINTED: rms granularity value, rms aperture, resolving power (both contrasts), gamma, Dmin,
Dmax, densitometry status, base thickness, interimage/DIR, mask density.

---

## KODAK_VISION2_250D_5205
FILES `KODAK/5205t.pdf` and `KODAK/H-1-5205t.pdf` — **byte-for-byte equivalent text; identical
document** (same 6 pages, same extracted text). KODAK Publication **H-1-5205t**, **August 2004**,
©Eastman Kodak 2004 (p1). Masthead: "KODAK VISION2 250D Color Negative Film 5205 / 7205" (p1).
VERIFIED. Use either file; cite `H-1-5205t.pdf`.

- Exposure index (p1): Daylight (5500 K) — 250; Tungsten (3200 K) — 64 "(with KODAK WRATTEN Gelatin
  Filter No. 80A)" → tungsten EI 64 is a **conversion-filter (No. 80A) figure**.
- Balance colour temperature (p2): daylight **5500 K**.
- Diffuse RMS granularity (p3): "Refer to curve." Aperture **48-micrometer**, microdensitometer
  (red, green, blue). No numeric value.
- Resolving power: NOT PRINTED.
- Contrast/gamma, Dmin, Dmax: NOT PRINTED.
- Reciprocity (p2): "You do not need to make any filter corrections or exposure adjustments for
  exposure times from 1/1000 to 1/10 second. If your exposure is in the 1-second range … increase
  your exposure 2/3 stop and use a KODAK Color Compensating **Filter CC10R**. If your exposure is in
  the 10 second range … increase your exposure by a stop and use a KODAK Color Compensating
  **Filter CC10R**." → **chromatic** failure beyond 1/10 s.
- Process (p2): **ECN-2**. Densitometry (p3, p4): **Status M**; spectral sensitivity at
  D = 0.2 above D-min, effective exposure 1/25 s.
- Base (p1): "acetate safety base with **rem-jet backing**". Thickness NOT PRINTED.
- Mask / interimage / DIR: NOT PRINTED (spectral dye-density curves are labelled "D-mins
  subtracted", p4).

Vector curve pages:
- p3 — 3 paths (125, 53, 52): sensitometric curves + modulation-transfer curves.
- p4 — 14 paths (164, 134, 133, 123, 110, 109, 93, 93, 84, 67, 53, 53, 50, 27): diffuse rms
  granularity curves, **spectral sensitivity curves (highest value)**, spectral dye-density curves.

NOT PRINTED: numeric rms value, resolving power, gamma, Dmin, Dmax, base thickness, interimage/DIR,
mask density.

---

## KODAK_VISION2_500T_5218
PREFERRED FILE `KODAK/5218-Vision2-500T-H-1-5218t.pdf` — KODAK Publication **H-1-5218t**,
**March 2006**, ©Eastman Kodak 2006 (p1). Masthead: "KODAK VISION2 500T Color Negative Film
5218 / 7218 / SO-218" (p1). VERIFIED.
SECONDARY FILE `KODAK/500T - 5218.pdf` — KODAK Publication **H-1-5218**, CAT 147 4188,
©Eastman Kodak 2002, **Revised 10-2005** (p4); 4-page brochure format, masthead "KODAK VISION2 500T
Color Negative Film 5218 / 7218" (p1). VERIFIED but strictly less data (no densitometry status, no
rms aperture). Prefer H-1-5218t.

- Exposure index (H-1-5218t p1): Tungsten (3200 K) — 500; Daylight — 320. Footnote 1: "With a KODAK
  WRATTEN Gelatin Filter No. 85." → daylight EI 320 is a **conversion-filter (No. 85) figure**.
  Same values in `500T - 5218.pdf` p2.
- Balance colour temperature (H-1-5218t p1): tungsten **3200 K**, ±150 K without correction filters.
- Diffuse RMS granularity (H-1-5218t p3): "Refer to curve." Aperture **48-micrometer**,
  microdensitometer (red, green, blue). No numeric value. (`500T - 5218.pdf` p3 gives the curve only,
  no aperture.)
- Resolving power: NOT PRINTED in either file.
- Contrast/gamma, Dmin, Dmax: NOT PRINTED.
- Reciprocity (H-1-5218t p2; identical wording `500T - 5218.pdf` p3): none needed 1/1000–1/10 s;
  1-second range +2/3 stop with **CC 10R**; 10-second range +1 stop with **CC 10R** →
  **chromatic** failure beyond 1/10 s.
- Process (H-1-5218t p2): **ECN-2**. Densitometry (H-1-5218t p3, p4): **Status M**; spectral
  sensitivity at D = 0.2 above D-min.
- Base (H-1-5218t p1): 5218 and 7218 — "acetate safety base with **rem-jet backing**";
  **SO-218 — ESTAR Safety Base with rem-jet backing**. Thickness NOT PRINTED.
- Mask / interimage / DIR: NOT PRINTED (dye-density curves "D-mins subtracted", p4).

Vector curve pages (`5218-Vision2-500T-H-1-5218t.pdf`):
- p3 — 2 paths (144, 118): modulation-transfer curves + sensitometric curves.
- p4 — 13 paths (301, 297, 281, 274, 249, 242, 208, 208, 193, 132, 131, 87, 87): diffuse rms
  granularity curves, **spectral sensitivity curves (highest value)**, spectral dye-density curves.
Vector curve pages (`500T - 5218.pdf`): p2 and p3 each have one 54-item path (a small graphic
element) plus full-page 28-item frame paths — **no extractable data curves**; the graphs on p3 are
drawn as many sub-25-item path fragments.

NOT PRINTED: numeric rms value, resolving power, gamma, Dmin, Dmax, base thickness, interimage/DIR,
mask density.

---

## KODAK_VISION_200T_5274
FILE `KODAK/5274.pdf` — KODAK Publication **H-1-5274**, CAT 855 9585, **April 1997** ("New 4-97-BX",
p6), ©Eastman Kodak 1997. Masthead: "KODAK VISION 200T Color Negative Film 5274 / 7274" (p1).
VERIFIED.

- Exposure index (p1): Tungsten (3200 K) — 200; Daylight (5500 K) — 125 "(with KODAK WRATTEN Gelatin
  Filter No. 85)" → daylight EI 125 is a **conversion-filter (No. 85) figure**.
- Balance colour temperature (p2): tungsten **3200 K**, ±150 K without correction filters.
- Diffuse RMS granularity: NOT PRINTED numerically — "the measured granularity is very low" (p3)
  plus a granularity curve. **Measuring aperture NOT PRINTED.**
- Resolving power: NOT PRINTED.
- Contrast/gamma, Dmin, Dmax: NOT PRINTED.
- Reciprocity (p2): "No filter corrections or exposure adjustments for exposure times from 1/1000 of
  a second to 1 second. If your exposure is in the 10-second range, increase exposure 2/3 stop, and
  use a KODAK Color Compensating **Filter CC10Y**." → achromatic to 1 s, **chromatic (yellow
  correction) at 10 s**.
- Process (p1): **ECN-2**. Densitometry (p3 sensitometric + MTF; p4 spectral sensitivity):
  **Status M**; spectral sensitivity at density 0.4 above D-min, effective exposure 0.013 s.
- Base (p1): "Acetate safety base with **rem-jet backing**". Thickness NOT PRINTED.
- Mask / interimage / DIR: NOT PRINTED. p4 gives "SPECTRAL DYE PEAKS" (cyan/magenta/yellow) rather
  than full spectral dye-density curves.

Vector curve pages:
- p3 — 12 paths (96, 96, 88, 88, 88, 80, 80, 72, 72, 48, 48, 48): diffuse RMS granularity curves,
  sensitometric curves, modulation-transfer curves.
- p4 — 6 paths (240, 232, 168, 104, 80, 80): **spectral-sensitivity curves (highest value)** and
  spectral dye-peak curves.
- p1/p6 — 23 paths each; all bboxes ~(543..552, 75..94) pt = logo artwork, not data.

NOT PRINTED: rms value, rms aperture, resolving power, gamma, Dmin, Dmax, base thickness,
interimage/DIR, mask density.

---

## KODAK_VISION_250D_5246
FILE `KODAK/5246.pdf` — KODAK Publication **H-1-5246t**, **March 2003**, ©Eastman Kodak 2003 (p1).
Masthead: "KODAK VISION 250D Color Negative Film / 5246, 7246" (p1). VERIFIED. 8 pages.

- Exposure index (p1): Daylight (5500 K) — 250; Tungsten (3200 K) — 64 "(with KODAK WRATTEN Gelatin
  Filter No. 80A)" → tungsten EI 64 is a **conversion-filter (No. 80A) figure**.
- Balance colour temperature (p1/p2): daylight **5500 K**.
- Diffuse RMS granularity (p3): "Refer to curve." Aperture **48-micrometer**, microdensitometer
  (red, green, blue). No numeric value.
- Resolving power: NOT PRINTED.
- Contrast/gamma, Dmin, Dmax: NOT PRINTED.
- Reciprocity (p2): "You do not need to make any filter corrections or exposure adjustments for
  exposure times from 1/1000 to 1 second. If your exposure is in the 10 second range … increase your
  exposure 2/3 stop and use a KODAK Color Compensating **Filter CC10Y**." → achromatic to 1 s,
  **chromatic (yellow) at 10 s**.
- Process (p2/p3): **ECN-2**. Densitometry (p4, p5): **Status M**; spectral sensitivity at density
  0.4 above D-min, effective exposure 0.013 s.
- Base (p1): "acetate safety base with **rem-jet backing**". Thickness NOT PRINTED.
- Mask / interimage / DIR: NOT PRINTED (dye-density curves "D-mins subtracted", p5).

Vector curve pages:
- p4 — 5 paths (150, 144, 143, 141, 128): sensitometric curves, modulation-transfer curves, diffuse
  rms granularity curves.
- p5 — 10 paths (214, 157, 129, 128, 128, 128, 127, 127, 126, 48): **spectral sensitivity curves
  (highest value)** and spectral dye-density curves.
- p1/p8 — 13 paths each, bbox ~(520..557, 74..106) pt = logo artwork, not data.

NOT PRINTED: numeric rms value, resolving power, gamma, Dmin, Dmax, base thickness, interimage/DIR,
mask density.

---

## KODAK_VISION_500T_5279
FILE `KODAK/5279.pdf` — KODAK Publication **H-1-5279**, CAT 132 9317, **March 1996** ("New 3/96-BX",
p5), ©Eastman Kodak 1996. Masthead: "KODAK VISION 500T Color Negative Film 5279 / 7279" (p1).
VERIFIED.

- Exposure index (p1): Tungsten (3200 K) — 500; Daylight — 320 "(with KODAK WRATTEN Gelatin
  Filter No. 85)" → daylight EI 320 is a **conversion-filter (No. 85) figure**.
- Balance colour temperature (p1): tungsten **3200 K**, ±150 K without correction filters.
- Diffuse RMS granularity: NOT PRINTED numerically — "the measured granularity is very low" (p2)
  plus a granularity curve. **Measuring aperture NOT PRINTED.**
- Resolving power: NOT PRINTED.
- Contrast/gamma, Dmin, Dmax: NOT PRINTED.
- Reciprocity (p1): "No filter corrections or exposure adjustments for exposure times from 1/1000 of
  a second to 1 second. In the 10-second range, increase exposure 2/3 stop and use a KODAK Color
  Compensating **Filter CC10Y**." → achromatic to 1 s, **chromatic (yellow) at 10 s**.
- Process (p1): **ECN-2**. Densitometry status: **NOT PRINTED** (no "Status M" label on any curve).
- Base (p1): "Acetate safety base with **rem-jet backing**". Thickness NOT PRINTED.
- Masking-coupler statement (p3, "SPECTRAL DYE PEAKS"): "The net negative densities for the cyan dye
  curve are a natural consequence of the level of the **magenta masking coupler**. The level was
  chosen to give flat correction averaged over a range of wavelengths — there will be a slight
  overcorrection at some wavelengths and a slight undercorrection at others." Qualitative only; no
  numeric mask density.
- Interimage / DIR coupler: NOT PRINTED.

Vector curve pages:
- p2 — 7 paths (64, 64, 64, 64, 56, 56, 56): diffuse RMS granularity curves, sensitometric curves,
  modulation-transfer curves.
- p3 — 6 paths (368, 272, 224, 72, 64, 48): **spectral-sensitivity curves (highest value)** and
  spectral dye-peak curves.
- p1/p5 — 23 paths each, bboxes ~(543..552, 75..94) pt = logo artwork, not data.

NOT PRINTED: rms value, rms aperture, resolving power, gamma, Dmin, Dmax, densitometry status,
base thickness, interimage/DIR, numeric mask density.

---

## KONICA_CENTURIA_SUPER_1600
FILE `KONICA/csuper1600.pdf` — "Konica Color CENTURIA SUPER 1600" Technical Data Sheet (p1, p2).
No publication number and no date printed; the sheet's own comparative claim is dated
"( * As of February, 2002)" (p1), so it postdates February 2002. VERIFIED.

- Film speed (p1): Daylight or Electronic Flash — **ISO 1600/33°**, no light-balancing filter;
  Photolamp (3400 K) — **520/28°**, Wratten **No. 80B**; Tungsten (3200 K) — **400/27°**,
  Wratten **No. 80A**. Footnote (p1): "*Includes the exposure factor to obtain best color results
  without special printing." → the 520 and 400 figures are **conversion-filter (80B / 80A) figures,
  not film properties**.
- Balance colour temperature (p1): "color-balanced for daylight"; a numeric Kelvin value for the
  daylight balance is **NOT PRINTED** (only the 3400 K / 3200 K source temperatures are given).
- Diffuse RMS granularity (p2): **6**; "Aperture diameter: **48 µmø**".
- Resolving power (p2): **Test-Object Contrast 1.6:1 — 50 lines/mm; Test-Object Contrast
  1000:1 — 100 lines/mm**.
- Contrast/gamma, Dmin, Dmax: NOT PRINTED.
- Reciprocity (p3): "A wide range of shutter speeds (i.e. 1/10000~1 sec.) can be used without loss of
  film speed and tone reproduction." Reciprocity-failure compensation guide table (p3):
  1/10000–1 s → compensation **None**, colour-compensating filters **None**;
  10 s → **+1 stop**, colour-compensating filters **None**.
  → explicit "None" for CC filters at 10 s ⇒ **achromatic reciprocity failure**.
- Process (p3): "Konica Color Negative Film Process **CNK Series or Process C-41**"; curves labelled
  **CNK-4** (p2, p3). Densitometry: **Status M** (spectral sensitivity and characteristic curves,
  p3); MTF measured "Through visual filter" (p2).
- Base (p1): "**Triacetate base**". Thickness NOT PRINTED. Antihalation type NOT PRINTED (no remjet,
  no AHU statement).
- Emulsion technology (p1): Super MCC (Super Multi-Coated Crystal), UCC (Ultra Consistent Crystal).
  DX code 26-4; emulsion numbers #550–#599; 135 size.
- Interimage / DIR coupler / mask density: NOT PRINTED.

Vector drawings:
- p1 — 504 drawings, **54 paths with >=25 items** (154, 92, 58, 43, 43, 39, then a long run of 38s
  and 28s). These sit in the LAYER STRUCTURE panel (bboxes clustered at
  (215,133)-(404,155) and (310,434)-(471,587)) → they plot the **before/after-processing layer
  structure schematic, not measured data**.
- p2 — 14 drawings, largest 19 items; p3 — 39 drawings, largest 10 items. The MTF, spectral
  dye-density, spectral-sensitivity and characteristic curves on p2/p3 ARE vector but are drawn as
  many small fragments, all **below the 25-item threshold**. No raster images anywhere in the file.

NOT PRINTED: balance Kelvin value, gamma/contrast, Dmin, Dmax, base thickness, antihalation type,
interimage/DIR, mask density, publication number, publication date.

---

## KONICA_IMPRESA_50
FILE `KONICA/IMP50.pdf` — "Konica Color IMPRESA 50 Professional Film", **PUB. NO. TDSN-501** (p3).
No date printed. VERIFIED.

- Film speed (p1): Daylight or Electronic Flash — **ISO 50/18°**, no filter; Photolamp (3400 K) —
  **16/13°**, Wratten **No. 80B**; Tungsten (3200 K) — **12/12°**, Wratten **No. 80A**. Footnote
  (p1): "Includes the exposure factor to obtain best color results without special printing." →
  16 and 12 are **conversion-filter (80B / 80A) figures, not film properties**.
- Balance (p1): "color-balanced for daylight, electronic flash and blue flash bulbs". Numeric Kelvin
  value for the daylight balance: **NOT PRINTED**.
- Diffuse RMS granularity: **NOT PRINTED** — no value and no aperture anywhere in the file
  (confirmed by text extraction and by OCR of all three page rasters).
- Resolving power (p3): **Test-Object Contrast 1.6:1 — 63 lines/mm; Test-Object Contrast
  1000:1 — 160 lines/mm**.
- Contrast/gamma, Dmin, Dmax: NOT PRINTED.
- Reciprocity (p2): "A wide range of shutter speeds (i.e. 1/10000~1 sec.) can be used without loss of
  film speed and tone reproduction." Compensation guide (p2): 1/10000–1 s → **None / None**;
  10 s → **+1/2 stop**, colour-compensating filters **None** → explicit "None" ⇒ **achromatic**.
- Process (p2): "Konica Color Negative Film Process **CNK-4 Series or Process C-41**"; spectral
  dye-density curves labelled "Process: CNK-4" (p3). Densitometry status: **NOT PRINTED**
  (curve annotations are raster; OCR recovers only "Process: CNK-4", "Midscale Density",
  "Minimum Density").
- Base (p1): "**Triacetate base**". Thickness NOT PRINTED. Antihalation type NOT PRINTED.
- Sizes (p1): 135 size 24, 36 exp.; 120 size. Electronic-flash guide numbers table on p2.
- Interimage / DIR coupler / mask density: NOT PRINTED.

Vector drawings: **none qualifying** — p1 has 55 drawings (all 1 item), p2 has 253 drawings (all
1 item), p3 has 0. **All graphs in this file are raster images**: p1 image 1936x880 px (layer
structure) and 729x204 px; p2 image 2008x888 px (spectral sensitivity + characteristic curves);
p3 image 2008x1184 px (spectral dye-density curves + modulation transfer function). Not vector-
extractable; would require curve tracing from the bitmaps.

NOT PRINTED: rms granularity value, rms aperture, balance Kelvin value, gamma/contrast, Dmin, Dmax,
densitometry status, base thickness, antihalation type, interimage/DIR, mask density, sheet date.

---

## KONICA_INFRARED_750
FILE `KONICA/INF750.pdf` — "Konica Infrared 750 Black & White film", **PUB. No. TDSB-701** (p3).
No date printed. VERIFIED. (`KONICA/konica_inf750.pdf` is the **same document** — identical text and
page count; either may be cited.)

- **SPECTRAL SENSITISATION LIMIT (p1, high-value value requested):** "Konica Infrared 750 film has a
  wavelength sensitivity range of **640 nm~820 nm** in addition to the intrinsic sensitivity of the
  silver bromide of **400 nm~500 nm**. The **peak spectral sensitivity occurs at 750 nm**."
  → documented long-wavelength limit = **820 nm**; IR band onset 640 nm; peak 750 nm; blue intrinsic
  band 400–500 nm. The p1 spectral-sensitivity plot's x-axis runs **400–800 nm** (OCR of the graph
  raster), i.e. the plot truncates 20 nm short of the stated 820 nm limit.
- Exposure index (p2): "With no filter the sensitivity of the film is equivalent to **ISO 32**."
  Standard exposure with a **Kenko R-1** filter: **f/5.6 @ 1/60 s** in normal sunny outdoor
  conditions. Tungsten/daylight EI pair and any conversion filter: NOT PRINTED (monochrome stock;
  no conversion filter applies).
- Balance colour temperature: NOT PRINTED / not applicable (black-and-white).
- Filtration (p1): red **Kenko R-1** or orange **Kenko YAS** to cut wavelengths below 520 nm (orange
  cut given as "C4Onm" in the file — an OCR-grade typo in the PDF's own text layer, read as ~640 nm);
  Wratten 25, Wratten 29 (red) and Wratten 15 (orange) give similar results.
- Infrared focusing (p1): use the lens P-mark (infrared-correction mark); some apochromatic lenses
  may not require it.
- Diffuse RMS granularity: **NOT PRINTED** (only "fine grain", p1). Aperture NOT PRINTED.
- Resolving power: **NOT PRINTED** numerically (only "excellent resolving power", p1).
- Contrast/gamma: NOT PRINTED numerically. Contrast is adjustable via development time; standard
  development times (p2): Konicadol DP 6 min @20 °C / 4 min @25 °C; Konicadol Fine 7 min / 5 min;
  Konicadol Super 6 min / 4.5 min. (DP ≡ Kodak D-76; also DK-20 and Ilford ID-68 equivalents listed.)
- Dmin, Dmax: NOT PRINTED.
- Reciprocity: **NOT PRINTED** — no reciprocity table or statement anywhere in the file.
- Process (p2): black-and-white; "The same developers that are used for processing panchromatic films
  can be used." Not ECN-2/C-41. Stop bath 1.5% acetic acid 30 s; Konica fix 10 min or Konica fix
  rapid 3 min; wash 20–30 min @15–25 °C. Densitometry status: NOT PRINTED.
- Base and antihalation (p1, LAYER COMPOSITION — directly answers the antihalation question):
  "A single thin infrared-sensitive emulsion layer is coated on a **colored anti-halation triacetate
  base**." → **anti-halation is a dyed/coloured base (AHU-type), NOT rem-jet**. Base material
  **triacetate**. Thickness NOT PRINTED.
- Characteristic curves (p3) annotations, recovered by OCR: "Exposure: Using Filter R-1 with daylight
  type exposure; Development: **Konicadol DP**, agitating intermittently at **20 °C**" and a second
  family "Development: **Konicadol Super**, agitating intermittently at 20 °C". Axis: Density (D).
- Interimage / DIR coupler / mask density: not applicable (monochrome), NOT PRINTED.

Vector drawings: **none qualifying** — p1 has 0 drawings, p2 has 97 drawings (all 1 item), p3 has 0.
**Both graphs are raster images**: p1 image 1440x276 px = spectral-sensitivity plot (daylight,
without filter; x-axis 400–800 nm) plus a 814x224 px header graphic; p3 image 1976x1432 px =
characteristic curves for Konicadol DP and Konicadol Super. Not vector-extractable.

NOT PRINTED: rms granularity, rms aperture, resolving power number, gamma value, Dmin, Dmax,
reciprocity data of any kind, densitometry status, base thickness, balance colour temperature,
sheet date.

---

## Cross-cutting summary

Never printed on ANY of these 12 sheets: **contrast/gamma value, Dmin, Dmax, base thickness,
interimage-effect or DIR-coupler statement, numeric orange-mask density.** Those six fields must
stay unpopulated (or be sourced elsewhere) — the datasheets do not support them.

Resolving power with both test-object contrasts is printed on only 4 of 11 stocks:
5248 (80 / 160), 5296 (50 / 100), CENTURIA SUPER 1600 (50 / 100), IMPRESA 50 (63 / 160).

A single numeric diffuse RMS granularity is printed on only 2 stocks: 5248 ("Less than 5") and
CENTURIA SUPER 1600 (6). All Kodak VISION/VISION2/EXR sheets say "refer to curve"; the 48-micrometre
aperture is stated for 5245, 5248, 5293, 5296, 5205, 5218 and 5246 but NOT for 5217, 5274 or 5279.

Reciprocity character: **achromatic** for 5245, 5293, 5296 (no filter, 1/1000–1 s), 5248 (+1/3 stop
at 1 s, no filter), CENTURIA SUPER 1600 and IMPRESA 50 (explicit CC filter "None" at 10 s).
**Chromatic** for 5217, 5205, 5218 (CC 10R beyond 1/10 s) and 5274, 5246, 5279 (CC10Y at 10 s).
Not printed at all for INFRARED 750.

Base: rem-jet backed acetate for all ten Kodak motion-picture stocks (5218's SO-218 variant is
ESTAR + rem-jet); plain triacetate for the two Konica colour negatives; **coloured anti-halation
triacetate** for Konica Infrared 750.
