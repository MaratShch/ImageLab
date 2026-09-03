# RESULT_OCR_IMAGEONLY.md — the four "image-only files needing OCR" from NotFound.md §4

**Date:** 2026-08-16. **Scope:** the four items listed in NotFound.md §4 under
"image-only files needing OCR (`NewGevacol_Neg_682.pdf`, `centuria_pro_400.pdf`,
`professional_160.pdf`, Konica IMP50/INF750 raster curves)".

**Method.** Every file was first tested for a text layer with `pymupdf`
`page.get_text()`. Two of the four turned out to have full text layers and needed no
OCR at all. The genuinely image-only files were rendered at 350 dpi and passed through
`tesseract` (v5, `--psm 3`); every load-bearing number was then re-checked by a **visual
read** of a 350-dpi crop of the page image, in the same way the Zhurba scans were
handled. Nothing here is traced off a curve, estimated, or extrapolated.

**Provenance classes used below**
- `TEXT` — printed text layer extracted directly from the PDF.
- `OCR` — tesseract output, not independently re-checked.
- `OCR+VIS` — tesseract output confirmed by visual read of the 350-dpi page crop.
- `VIS` — read visually off the page image only.

---

## 0. Summary table

| # | File | Text layer? | Action | Outcome |
|---|------|-------------|--------|---------|
| 1 | `PDF/PROFILES/AGFA/NewGevacol_Neg_682.pdf` | no (0 chars, 3 pp) | OCR + visual verify | Rich. It is the Vervoort/Stappaerts SMPTE paper already cited in the DB. |
| 2 | `PDF/PROFILES/KONICA/centuria_pro_400.pdf` | no (0 chars, 2 pp) | OCR + visual verify | **No database-relevant numbers.** Marketing brochure. |
| 3 | `PDF/PROFILES/KONICA/professional_160.pdf` | no (0 chars, 4 pp) | OCR + visual verify | Modest: process, densitometry, base materials, curve inventory. **No DB stock matches it.** |
| 4 | `PDF/PROFILES/KONICA/IMP50.pdf`, `INF750.pdf` | **YES** — 1392–2182 chars/page | plain read, no OCR needed | Full datasheets. Already mined into the DB. |

**The NotFound.md §4 premise was partly wrong: IMP50.pdf and INF750.pdf are not
image-only.** Only the *figures* on those pages are raster; all the surrounding
datasheet prose and every printed number is in a real text layer.

---

## 1. `PDF/PROFILES/AGFA/NewGevacol_Neg_682.pdf` — 3 pp, image-only, OCR'd

**Identification (OCR+VIS, p1 + p3 footer).** This is *not* an Agfa-Gevaert datasheet.
It is the reprinted journal paper:

> A. Vervoort and H. Stappaerts, "A New Gevacolor Negative Film Type 682",
> **SMPTE Journal, September 1980, Volume 89, pp. 650–652**. Presented 22 October 1979
> at the Society's 121st Technical Conference, Los Angeles. Originally published in the
> April 1980 BKSTS Journal, reprinted by permission. Authors at Agfa-Gevaert N.V.,
> Septestraat 27, B-2510 Mortsel, Belgium.

That is exactly the source already cited in `film_profiles.py` for
`GEVACOLOR_NEG_682`, so **no new source enters the corpus** — the file simply *is* the
cited paper in scanned form.

### DUPLICATE — important housekeeping finding

`PDF/PROFILES/AGFA/Verpoort_Stapp1980_NewGevacolNeg682.pdf` is **byte-identical** to
`NewGevacol_Neg_682.pdf` (both md5 `3d5057d11db345f49c48d2617ff51854`; per-page 50-dpi
raster hashes also identical). Two names, one document. No need to ever open the second.

### Extracted parameters

| Parameter | Value as printed | Page | Provenance |
|---|---|---|---|
| Exposure index | **100 ASA** ("rated 100 ASA when exposed under tungsten illumination at 3200 K"; restated p2 "an exposure index of 100 ASA") | p1, p2 | OCR+VIS |
| Colour balance | **3200 K** tungsten; usable in daylight "in conjunction with a suitable conversion filter" | p1, p2 | OCR+VIS |
| Process | **ECN-2**; "Total compatibility has been confirmed" with existing ECN-2 machinery/chemistry, no modification | p1, p3 | OCR |
| Gamma | **γ = 0.57**, annotated directly on Fig. 10, Status M measurement | p3 (Fig. 10) | VIS (printed on the plot) |
| Densitometry | **Status M** ("STATUS M MEASUREMENT" on Fig. 10) | p3 | VIS |
| Base | **Triacetate base**, blue/green/red sensitive emulsion layers coated on it | p1 | OCR+VIS |
| Antihalation | Removable **carbon black backing layer**, "specially selected for its antihalation and antistatic properties"; further antihalation from the coloured masking coupler in the red-sensitive layer | p1, p2 | OCR |
| Layer structure | Fig. 6 gives the full stack, printed as labels: protective layer / blue-sens. high-speed (yellow dye coupler) / blue-sens. medium-speed (yellow dye coupler) / **yellow filter layer** / green-sens. high-speed (magenta dye coupler + yellow mask coupler) / green-sens. medium-speed (magenta dye coupler + yellow mask coupler) / interlayer / red-sens. high-speed (cyan dye coupler + red mask coupler) / red-sens. medium-speed (cyan dye coupler + red mask coupler) / subbing layer / base / backing layer | p2 (Fig. 6) | OCR+VIS |
| Double-layer technique | Every emulsion layer is two layers of the same spectral sensitivity: one medium-speed fine-grain, one fast coarser-grain | p1 | OCR |
| DIR couplers | Present in the **green- and red-sensitive layers only** (not blue). Stated purpose: improve granularity/speed relation and enhance edge effects | p1, p2 | OCR |
| RMS granularity model (Table I) | Header: "The RMS granularity, σ_D ≅ [1/√n], of Gevacolor negative film Type 682." Body — number of image-forming centres n (%) → relative negative granularity σ_D: **100 → 1.00; 110 → 0.95; 120 → 0.91; 130 → 0.88**. Text gloss: "an increase of 30% of image forming areas results in a 12% lower RMS granularity" | p2 (Table I) | OCR+VIS |
| Recommended exposure | Incident-light method; at an illumination level of **1076 lx (100 ft/cd)**, favourable results at **f/2.8** and **1/50 s**, "or any equivalent exposure" | p2 | OCR+VIS |
| Storage — raw stock | Long-term storage temperature **below 12 °C (54 °F)** | p3 | OCR+VIS |
| Storage — processed | Closed cans, **21 °C (70 °F)**, **40–60 % RH**, nonarchival | p3 | OCR+VIS |
| Printer trimmer settings (Table III), Type 682 → Gevacolor print film Type 982 | Prefilter R/G/B = ND30/ND30/ND30; Filter R/G/B = —/ND10/ND60; manual light control 12/12/12; automatic light control 25/25/25 | p3 (Table III) | OCR+VIS |
| Relative S/N ratios (Table II) | Gevac.Neg.T682(16)→Gevac.Pos.T982(16), ×200: S/N_L 5.7, S/N_S 0.0. Theor.Neg.(16)→T982(16), ×200: 4.0, −1.7. T682(16)→T982(35), ×200: −2.5, 1.2. Theor.Neg.(16)→T982(35), ×200: −4.9, −0.9. T682(16)→T982(16), ×500: −4.1, −0.4. Theor.Neg.(16)→T982(16), ×500: −6.9, −2.8. All evaluated at a viewing distance equal to five times the screen diagonal | p3 (Table II) | OCR |
| Lineage stated | Gevaert first Gevacolor negative **16 ASA, 1948**; Agfa-Gevaert first **100 ASA masked** colour negative **Type 655, 1968** (Agfa-Gevaert process); **Type 680, early 1974** (ECN-1 compatible); Type 682 = ECN-2. Print stocks named: Gevacolor print film Type 982 (ECP-2), Type 986 (ECP-1) | p1 | OCR |
| Formats | 35 mm and 16 mm professional motion picture | p1 | OCR |

### Curve plots present — NOT traced, existence reported only

| Fig. | Page | What it plots | Axes as printed |
|---|---|---|---|
| Fig. 1 | p1 | Printing sensitometry of a selectively red exposed **unmasked** negative | DENSITY vs LOG REL. EXP., 0–3.60 |
| Fig. 2 | p1 | Printing sensitometry of a selectively red exposed **masked** negative | DENSITY vs LOG REL. EXP., 0–3.60 |
| Fig. 3 | p1 | Schematic of the double-layer technique (3A fine-grain medium speed, 3B fast coarser) | DENSITY vs LOG REL. EXP. |
| Fig. 4 | p2 | Schematic: crystal-level reduction of colour granularity with vs without DIR (not a data plot) | — |
| Fig. 5 | p2 | Influence of DIR coupler on gradation (gamma), with/without DIR curves | DENSITY vs LOG REL. EXP. |
| Fig. 7 | p2 | **Spectral sensitivity of the three emulsion layers** (B, G, R) | ordinate "PHOTOTICITY [log RE(λ)·S(λ)]" vs WAVELENGTH (nm), axis ticks 400 / 500 / 600 / 700 |
| Fig. 8 | p2 | Spectral density curves of the three dyes formed after development | density vs WAVELENGTH (nm) |
| Fig. 9 | p2 | U-V chromaticity diagram, loci of nine colour patches (blue, green, red, yellow, magenta, cyan, foliage, sky blue, flesh/skin tone) for **Type 680 vs Type 682** | u′–v′ |
| Fig. 10 | p3 | **Sensitometric curves**, B/G/R, with γ = 0.57 annotated | DENSITY 0–2.5 vs LOG REL. EXP. 0–4.00; "STATUS M MEASUREMENT" |
| Fig. 11 | p3 | **Modulation transfer function**, B/G/R | RESPONSE (%) 0–120 vs SPATIAL FREQUENCY (lines/mm), log axis ticks 2 / 5 / 10 / 20 / 30 / 40 / 50 |
| Fig. 12 | p3 | **RMS-granularity per layer**, B/G/R | σ_D × 1000 (log, 1–100) vs DENSITY ABOVE D_min (0 – 1.75, ticks 0.25/0.50/0.75/1.00/1.25/1.50/1.75) |

### Caveats that must not be lost

- **No aperture and no magnification is stated anywhere for the RMS granularity in
  Fig. 12 or Table I.** The paper gives σ_D only as a curve against density above D_min.
  Any 48 µm / 24 µm assumption would be an invention. Table I is a *relative* model
  (σ_D ∝ 1/√n), not a set of absolute granularity readings.
- **No resolving power is printed** — neither the 1.6:1 nor the 1000:1 test-object
  contrast figure appears anywhere in the three pages. The paper gives definition only
  as the Fig. 11 MTF.
- **No reciprocity data, no D_max, no base thickness, no spectral limits in nm** are
  printed. Spectral behaviour exists only as the Fig. 7 curve.
- D_min values (B 0.90 / G 0.58 / R ≈ 0.12) are legible as the toe plateaus of Fig. 10
  but are curve reads, not printed figures; the existing DB comment already labels them
  as read from that figure.

### Cross-check against the existing DB entry

`GEVACOLOR_NEG_682` in `film_profiles.py` already carries `exposure_index=100`,
`balance_kelvin=3200`, gamma 0.57, Status M, the DIR/double-layer description and the
per-layer RMS ordering — all sourced from this paper. **The OCR confirms every one of
those printed values and contradicts none.** The only thing to change is the provenance
string, which still reads "NOTE ... is on file but image-only (OCR queued)".

---

## 2. `PDF/PROFILES/KONICA/centuria_pro_400.pdf` — 2 pp, image-only, OCR'd

**Verdict: no database-relevant numbers. This document contains nothing usable and
should not be re-read.**

It is a two-page consumer/trade **marketing brochure** for Konica Minolta CENTURIA
PRO 400 (p1 = cover shot + tagline; p2 = prose blurbs, two side-by-side lip crops
captioned "CENTURIA PRO 400" and "Competitor's ISO400", corporate footer). Both pages
were OCR'd and p2 was additionally read visually at 350 dpi to be certain no small spec
box was being missed. There is none.

| Parameter | Value as printed | Page | Provenance |
|---|---|---|---|
| Speed | **ISO 400** — the only number in the document, and it appears purely as prose ("a high-speed ISO400 film", "an ISO400 film") | p2 | OCR+VIS |
| Publisher / date | Konica Minolta Photo Imaging, Inc., No. 26-2 Nishishinjuku 1-chome, Shinjuku-ku, Tokyo 163-0512, Japan. "All information in this brochure is accurate as of **March 2003**." Code `*AB1 0407T02 M2 Printed in Japan` | p2 | OCR+VIS |
| Family relationship | "Featuring the high-performance technologies of the CENTURIA SUPER film series, Konica Minolta CENTURIA PRO 400 is a **new type** of colour film" | p2 | OCR |

No characteristic curves, no spectral sensitivity, no dye density curves, no MTF, no
granularity number, no resolving power, no process name, no base material, no
reciprocity, no D_min/D_max, no layer structure. Nothing.

**Stock-identity note.** CENTURIA **PRO** 400 is a distinct professional product from
CENTURIA **SUPER** 400. The DB holds `KONICA_CENTURIA_SUPER_400` and
`KONICA_CENTURIA_SUPER_1600`; it holds no CENTURIA PRO 400. This brochure therefore
does **not** describe a stock in the database, and its lone "ISO 400" must not be
attached to `KONICA_CENTURIA_SUPER_400`.

---

## 3. `PDF/PROFILES/KONICA/professional_160.pdf` — 4 pp, image-only, OCR'd

Konica Minolta **PROFESSIONAL 160** ("PRO 160"). pp 1–3 are marketing (cover, portrait
spread, four prose blurbs). **p4 is the only technical page** and carries the curve set
plus a "Size Available" table. p4 was OCR'd and then read visually in two halves.

| Parameter | Value as printed | Page | Provenance |
|---|---|---|---|
| Product name | "Konica Minolta PROFESSIONAL 160", short form "PRO 160" | p1, p4 | OCR+VIS |
| Process | **CNK-4** (stated on all three plots) | p4 | OCR+VIS |
| Densitometry | **Status M** (on characteristic curves and on spectral sensitivity) | p4 | OCR+VIS |
| Characteristic-curve exposure conditions | **Daylight, 1/125 sec.** | p4 | OCR+VIS |
| Spectral sensitivity reference density | **1.0 above D min.** | p4 | OCR+VIS |
| Spectral dye density reference | "Typical densities for a **midscale neutral subject** and **D min.**" — two labelled traces, "Midscale Density" and "Minimum Density" | p4 | OCR+VIS |
| Base — long roll | **Triacetate base**: 35 mm × 100′; perforation non-perforated or double perforated; winding core or spool | p4 | OCR+VIS |
| Base — long roll | **Polyester base**: 35 mm × 100′, 46 mm × 100′, 70 mm × 100′; non-perforated; winding core or spool | p4 | OCR+VIS |
| Formats | 135: 24 exp., 36 exp. 120: 6 exp. and 12 exp. (6×6 cm format). 220: 24 exp. (6×6 cm format) | p4 | OCR+VIS |
| Publisher | Konica Minolta Photo Imaging, Inc., Tokyo 163-0512, Japan. Code `*AB3 0407T02M6 Printed in Japan` | p4 | OCR+VIS |

### Curve plots present — NOT traced

| Plot | Page | Axes as printed |
|---|---|---|
| CHARACTERISTIC CURVES (Y/M/C traces) | p4 | Density (D) 0–3.0+ vs log H, ticks −3.0 / −2.0 / −1.0 / 0.0 |
| SPECTRAL SENSITIVITY (B/G/R traces) | p4 | Relative Speed (log) vs Wavelength (nm), ticks 400 / 500 / 600 / 700 |
| SPECTRAL DYE DENSITY CURVES | p4 | Diffuse Spectral Density (D) 0–2.0 vs Wavelength (nm), ticks 400 / 500 / 600 / 700 |

### Caveats

- **No ISO/EI figure is printed anywhere in the document.** "160" appears only inside
  the product name. The nominal speed must not be inferred from the name and recorded as
  a datasheet value.
- **No RMS granularity, no resolving power (neither contrast), no MTF, no gamma, no
  D_min/D_max number, no reciprocity data, no base thickness, no layer structure, and no
  colour-balance Kelvin figure** are printed. The film is described as daylight-exposed
  on the characteristic-curve plot; no K value is given.
- Undated apart from the print code; the sibling CENTURIA PRO 400 brochure with the
  adjacent code `*AB1 0407T02 M2` is dated March 2003, but that is a neighbouring
  document, not this one — do not date this sheet from it.

### Which DB stock does it match? — **none**

`film_enum.hpp` holds exactly seven Konica entries:
`KONICA_CENTURIA_SUPER_400` (105), `KONICA_CENTURIA_SUPER_1600` (106),
`KONICA_CHROME_CENTURIA_100` (107), `KONICA_CHROME_R100` (108),
`KONICA_IMPRESA_50` (109), `KONICA_INFRARED_750` (110), `KONICA_VX_100` (111).
There is no `KONICA_PROFESSIONAL_160`. This datasheet describes a stock the database
does not model. Its data cannot be attached to any existing entry.

---

## 4. Konica `IMP50.pdf` and `INF750.pdf` — **full text layer, no OCR was needed**

`page.get_text()` returns 1392 / 1850 / 1398 chars on IMP50 pp 1–3 and
2051 / 2182 / 957 chars on INF750 pp 1–3. Only the figures are raster. Everything below
is class **TEXT**, extracted verbatim.

`PDF/PROFILES/KONICA/konica_inf750.pdf` is a **duplicate** of `INF750.pdf` (identical
per-page text, identical page structure).

### 4a. IMP50.pdf — Konica Color IMPRESA 50 Professional, PUB. NO. **TDSN-501**

| Parameter | Value as printed | Page | Provenance |
|---|---|---|---|
| ISO speed, daylight/electronic flash | **50/18°**, no light-balancing filter | p1 | TEXT |
| ISO speed, photolamp 3400 K | **16/13°**, Wratten No. 80B | p1 | TEXT |
| ISO speed, tungsten 3200 K | **12/12°**, Wratten No. 80A | p1 | TEXT |
| Colour balance | Balanced for **daylight, electronic flash and blue flash bulbs**; good with fluorescent, satisfactory with tungsten | p1 | TEXT |
| Base | **Triacetate base** | p1 | TEXT |
| Formats | 135: 24, 36 exp.; 120 size | p1 | TEXT |
| Process | **Konica Color Negative Film Process CNK-4 Series or Process C-41** | p2 | TEXT |
| Reciprocity | **1/10000 – 1 sec: no compensation, no CC filters. 10 sec: +1/2 stop, no CC filters.** Prose: "A wide range of shutter speeds (i.e. 1/10000~1 sec.) can be used without loss of film speed and tone reproduction" | p2 | TEXT |
| **Resolving power** | **Test-Object Contrast 1.6:1 — 63 lines/mm; Test-Object Contrast 1000:1 — 160 lines/mm** (both contrasts printed) | p3 | TEXT |
| Storage | Below **10 °C** recommended for unused film; avoid formaldehyde and other harmful gases | p3 | TEXT |
| Daylight exposure guide | All at 1/125 sec: bright sunlight seascape/snow f/16; bright sunlight f/11; hazy sunlight f/8; cloudy bright f/5.6; cloudy dull / open shade f/4. Valid 2 h after sunrise to 2 h before sunset | p2 | TEXT |
| Electronic-flash guide numbers | BCPS 350/500/700/1000/1400/2000/2800/4000/5600/8000 → feet 28/35/42/49/60/70/84/98/119/140; metres 8/11/13/15/18/21/25/29/35/42 | p2 | TEXT |
| Technology claim | "Simulated Spectral Foundation Technology" | p1 | TEXT |

Curve plots (raster, not traced): **p1** layer-structure diagram (before/after
processing); **p2** SPECTRAL SENSITIVITY and CHARACTERISTIC CURVES; **p3** SPECTRAL DYE
DENSITY CURVES and MODULATION TRANSFER FUNCTION.
No RMS granularity number, no gamma number, no D_min/D_max number and no base thickness
are printed.

### 4b. INF750.pdf — Konica Infrared 750 Black & White, PUB. No. **TDSB-701**

| Parameter | Value as printed | Page | Provenance |
|---|---|---|---|
| Speed | "With no filter the sensitivity of the film is equivalent to **ISO 32**" | p2 | TEXT |
| **Spectral limits** | "wavelength sensitivity range of **640 nm ~ 820 nm** in addition to the intrinsic sensitivity of the silver bromide of **400 nm ~ 500 nm**. The peak spectral sensitivity occurs at **750 nm**" | p1 | TEXT |
| Layer structure | "A **single thin infrared-sensitive emulsion layer** is coated on a **coloured anti-halation triacetate base**. The emulsion is fine grain with excellent resolving power" | p1 | TEXT |
| Base | **Triacetate base** | p1 | TEXT |
| Formats | 135: 24 exposures; 120: 12 exposures (6 cm × 6 cm) | p1 | TEXT |
| Standard exposure | With a **Kenko R-1 filter: f/5.6 @ 1/60 sec** (normal sunny outdoor conditions) | p2 | TEXT |
| Filtration | Red (Kenko R-1) or orange (Kenko YAS) to cut below 520 nm / 640 nm; Wratten 25, Wratten 29 (red) and Wratten 15 (orange) give similar results | p1 | TEXT |
| Darkroom | Handle in **complete darkness** | p2 | TEXT |
| Development times | Konicadol DP: **6 min @ 20 °C, 4 min @ 25 °C** (≡ Kodak D-76). Konicadol Fine: **7 min @ 20 °C, 5 min @ 25 °C** (≡ Kodak DK-20). Konicadol Super: **6 min @ 20 °C, 4.5 min @ 25 °C** (≡ Ilford ID-68). "Final contrast of the film can be adjusted by changing the standard development times" | p2 | TEXT |
| Agitation | Continuous for the first minute, then 5 s at one-minute intervals | p2 | TEXT |
| Stop bath | **1.5 % acetic acid**, 30 s, at developer temperature | p2 | TEXT |
| Fix | Konica (acid layer-hardening) **10 min**; Konica fix rapid **3 min** | p2 | TEXT |
| Wash | Running water **15 ~ 25 °C for 20 ~ 30 min**; rapid route: 2 % anhydrous sodium sulfite 2–3 min after fix, then 5 min running water with agitation | p3 | TEXT |
| Dry | 0.5 % wetting agent for 1 min | p3 | TEXT |
| IR focusing | Use the lens P-mark (infrared correction mark); some apochromats may not need it | p1 | TEXT |

Curve plots (raster, not traced): **p1** SPECTRAL SENSITIVITY, daylight without filter,
abscissa Wavelength (nm); **p3** CHARACTERISTIC CURVE(S).
No RMS granularity, no resolving power number (only the prose claim "excellent resolving
power"), no gamma number, no D_min/D_max, no base thickness are printed.

### Cross-check against the existing DB

Both stocks are already in `film_profiles.py` and already cite these exact files:
`KONICA_INFRARED_750` cites "PDF/PROFILES/KONICA/INF750.pdf (TDSB-701)" and encodes the
640–820 nm sensitisation; `KONICA_IMPRESA_50` cites
"PDF/PROFILES/KONICA/IMP50.pdf ... PUB. No. TDSN-501", carries the resolving-power pair
`(63.0, 160.0)` matching the printed 1.6:1 / 1000:1 values, and carries
`ReciprocitySpec(..., onset_s=1.0)` derived from the printed "+1/2 stop at 10 s, no CC".
**Nothing in either sheet contradicts the DB, and nothing new remains unextracted.**

---

## 5. Disposition of the NotFound.md §4 sub-item

The bracket "image-only files needing OCR (`NewGevacol_Neg_682.pdf`,
`centuria_pro_400.pdf`, `professional_160.pdf`, Konica IMP50/INF750 raster curves)"
**can be struck in full.** Per file:

1. `NewGevacol_Neg_682.pdf` — **closed.** OCR'd and visually verified. It is the
   SMPTE 89(9) 1980 Vervoort/Stappaerts paper already cited; every printed value it
   contains is already in the `GEVACOLOR_NEG_682` entry. Two follow-ups, both editorial:
   drop the "(OCR queued)" clause from the provenance string, and record that
   `Verpoort_Stapp1980_NewGevacolNeg682.pdf` is a byte-identical duplicate.
2. `centuria_pro_400.pdf` — **closed as barren.** OCR'd and visually verified; contains
   one prose "ISO400" and nothing else, and describes CENTURIA **PRO** 400, which is not
   a DB stock. Should be recorded as "no database-relevant content" so nobody re-opens it.
3. `professional_160.pdf` — **closed.** OCR'd and visually verified. Yields process
   CNK-4, Status M, daylight 1/125 s curve conditions, triacetate and polyester long-roll
   bases and formats, plus a three-plot curve inventory on p4. No ISO, no granularity, no
   resolving power, no gamma. No DB stock matches it, so nothing can be written to the
   database; if a `KONICA_PROFESSIONAL_160` stock is ever added, p4 is the vector-curve
   source and belongs in `DIGITIZATION_QUEUE.md`, not in NotFound §4.
4. Konica IMP50 / INF750 — **closed; the item was based on a false premise.** Both files
   have complete text layers and were read directly. Only their figures are raster, and
   those figures are curve plots that belong in `DIGITIZATION_QUEUE.md` (vector/raster
   curve tracing), not in the OCR bracket. Both stocks are already fully mined into the DB.

**Residual, correctly reclassified as curve digitisation rather than OCR:**
`NewGevacol_Neg_682.pdf` Figs. 7, 8, 10, 11, 12 (p2–p3); `professional_160.pdf` p4
(three plots); `IMP50.pdf` p2–p3 (four plots); `INF750.pdf` p1 and p3 (two plots).
All are raster scans, so they need image-based curve reading, not vector extraction.
