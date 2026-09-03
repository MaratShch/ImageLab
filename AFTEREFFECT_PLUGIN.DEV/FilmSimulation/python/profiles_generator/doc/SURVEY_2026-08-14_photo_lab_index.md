# The Compact Photo-Lab-Index (1979) — survey and extraction plan

**Date:** 2026-08-14
**File:** `PDF/PROFILES/pittaro_em_the_compact_photolabindex.pdf` (55 MB, 724 pages)
**Source:** Pittaro, Ernest M. (ed.), *The Compact Photo-Lab-Index — The Cumulative
Formulary of Standard Recommended Photographic Procedures*, Morgan & Morgan Inc.,
145 Palisade Street, Dobbs Ferry NY. Basic set published June 1939; **36th edition
1978; 2nd Compact Edition 1979.**

**Your date is right: 1979.**

**Document state:** scan with an OCR text layer. 1.83 M characters extracted, ~2.5 kB
per page, no page returned empty. Readable throughout — but see §5, the OCR has two
specific failure modes that have already produced one wrong reading.

**This document is a survey only. No database file was modified.**

---

## 1. What this book is

Not a monograph and not a datasheet collection — it is the trade formulary the industry
actually worked from, organised by manufacturer, reprinting each maker's own published
procedures. That shapes what it is good and bad for:

* **Strong on** anything a lab technician needed at the bench: film speeds under each
  illuminant, filter factors, development times, safelight specifications, processing
  chemistry, reciprocity compensation.
* **Weak on** anything only an emulsion engineer needed: there are no per-layer spectral
  sensitivity curves for colour films, no per-film RMS granularity figures, and
  resolving power appears for barely a dozen products.

So it is a **processing and exposure** source far more than an emulsion-physics source.
That is the opposite of the Cheltsov & Bongard book processed yesterday, and the two are
complementary rather than overlapping.

## 2. Section map

| Section | Book pages | PDF pages | Character |
|---|---|---|---|
| Contents / index | 1–15 | 4–15 | Cross-reference index, useful for navigation |
| General technical definitions | — | 174–182 | **Methodology — see §3.6** |
| Agfa-Gevaert | 16–35 | ~17–38 | Films, papers, Agfacolor processes 82/85 |
| Eastman Kodak (still) | ~36–280 | ~43–300 | Largest section by far |
| Eastman Kodak (motion picture) | ~280–300 | 285–300 | Cine stocks, exposure indices |
| Fuji | 375–418 | ~378–420 | Illuminant-conditioned EI ladders |
| GAF | 419–461 | ~421–462 | |
| Ilford | 462–565 | ~465–567 | **Most complete per-film blocks** |
| Polaroid | ~566–605 | ~570–605 | **Most numerically complete — see §3.1** |
| 3M, H&W Control, Beseler, Unicolor, Spiratone | ~605–700 | | Smaller |
| Ansco / Du Pont (historical) | ~700–712 | | Legacy sheet films |
| Minox | 712–715 | | |

## 3. Data classes, by usefulness to this database

### 3.1 TIER 1 — Polaroid film data blocks (PDF 580–600). Highest value.

The Polaroid section prints, per film type, in plain numerals:

**D-max, D-min, slope, ASA speed, DIN speed (daylight *and* tungsten separately),
resolution in lines/mm, a per-film reciprocity compensation table, filter factors, and
a spectral sensitivity plot.**

Those are our fields, by name. Measured examples read directly off the pages:

| Type | ASA | DIN | D-max | D-min | Slope | Resolution |
|---|---|---|---|---|---|---|
| 51 (blue-sensitive, ultra-high contrast) | 320 day / 100 tungsten | 26 / 21 | 1.85 print, 1.75 | .09 / .00 | 3.30 / 3.35 | 28–32 lp/mm |
| 52 | 400 | 27 | 1.75 | .02 | 1.3–1.4 | 35–40 lp/mm |
| 42 | — | 24 | 1.75 | .02 | 1.3–1.4 | 25–28 lp/mm |
| **55 P/N — print** | 50 | 18 | 1.75 | .09 | 1.35 | 22–25 lp/mm |
| **55 P/N — negative** | 50 | 18 | 1.55 | .18 | **0.65** | **150–160 lp/mm** |
| 47 / 107 | 2500 | 36 | 1.6 | .02 | 1.3–1.4 | 20–22 lp/mm |
| 57 | 3000 | — | 1.65 | .11 | 1.45 | — |
| 105 (approx.) | 800 | 30 | 2.8 | .05 | 1.8 | 35–40 lp/mm |
| Polaline 146-L | 200 day / 60 tungsten | 24 / 19 | 2.55 / 2.3 | .03 / .02 | 3.30 / 3.00+ | 40–50 lp/mm |
| Polascope 410 | 10 000 | 41 | 1.6 | .02 | 2.0 | 22–28 lp/mm |

Two things stand out.

**Type 55 P/N's negative at 150–160 lines/mm would be the highest-resolution stock in
our database**, and the book gives its *whole* curve description — D-max 1.55, D-min
0.18, slope 0.65 — separately from the print it is exposed alongside. A slope of 0.65
on a 50-speed film with 160 lp/mm is a genuinely unusual combination and it is fully
specified, not estimated.

**Our Polaroid coverage is the weakest in the database:** three stocks
(`POLAROID_664`, `POLAROID_667`, `POLAROID_SX70`), and the book documents roughly
fifteen types numerically. This is the single largest ratio of new documented data to
authoring effort anywhere in the file.

### 3.2 TIER 1 — Kodak reciprocity + filter compensation master table (PDF 174–175)

Twelve colour films × seven exposure times from 1/10 000 s to 100 s, giving both the
exposure increase in stops **and the colour-correction filter required.** Films covered:
Kodacolor II, Kodacolor 400, Vericolor II Professional Types S and L, Ektachrome 64
Professional and 64, Ektachrome 50 Professional (Tungsten), Ektachrome 200 Professional
and 200, Ektachrome 160 Professional (Tungsten) and 160, Ektachrome Infrared,
Kodachrome 40 Type A (5070), Kodachrome 25, Kodachrome 64.

**Three of those are stocks we already carry** — `KODACHROME_64`, `EKTACHROME_64`,
`EKTACHROME_160T` — so this is upgrade material for existing entries, not just new ones.

**Important limitation, stated before anyone tries to use it.** The table is rounded to
the nearest half stop and most films have only two or three non-zero entries before
"Not Recommended". Fitting a Schwarzschild exponent from that is **ill-conditioned**:
for Kodachrome 64 the data are "no correction from 1/10 000 through 1/10 s, then +1 stop
at 1 s", and depending on whether onset is taken as 0.1 s or 0.5 s the fitted `p` comes
out anywhere from 0.70 to physically impossible. Kodak's half-stop rounding turns a
smooth effect into an apparent step.

What the table *does* give robustly:

* **`onset_s`** — the time at which compensation first becomes non-zero. We currently
  default this to 1.0 almost everywhere. This is a real, documented, per-film value.
* **The sign and colour of the shift** — CC10R for Kodachrome 64, CC10M for Kodachrome
  25 and 40, CC10R→CC15R for Ektachrome 160T, CC20B for Ektachrome Infrared. That
  constrains *which channel's* `schwarzschild_p` should be lowest, which is exactly the
  per-channel spread our `ReciprocitySpec` docstring calls "the origin of long-exposure
  colour casts" and currently sets by estimate.

So: take `onset_s` and the channel ordering as documented; do **not** take a fitted
exponent from this table and call it measured.

### 3.3 TIER 1 — Ilford per-film blocks (PDF 465–567)

The most complete conventional-film blocks in the book. Each film gets: general
properties, **wedge spectrogram to tungsten 2850 K with a labelled wavelength axis**,
film speed as ASA *and* DIN for daylight *and* tungsten, a reciprocity chart,
filter factors for daylight and tungsten separately, safelight specification,
**development times per developer and dilution targeting two named contrast indices**,
a contrast-index-versus-time curve, and a characteristic curve.

Read directly:

| Film | Daylight | Tungsten | Notes |
|---|---|---|---|
| Pan F | ASA 50, DIN 18 | — | ID-11 1+1: **9 min → CI 0.55, 14 min → CI 0.70** |
| FP4 | ASA 125, DIN 22 | ASA 100, DIN 21 | tolerates +6 stops over, −2 under |
| HP4 | ASA 400, DIN 27 | ASA 400, DIN 27 | |

Also documented for Pan F: speed must be reset to DIN 20 in Microphen and to **ASA 32,
DIN 16 in Perceptol** — a developer-dependent speed change, which is PRC-axis data of
exactly the kind Appendix A specifies.

### 3.4 TIER 1 — Fuji illuminant-conditioned exposure indices (PDF ~378–420)

Fuji publishes a speed *ladder by illuminant with the required filter*, e.g. Fujicolor
F-II 400: ASA 400 / 27 DIN daylight no filter; ASA 125 / 22 DIN tungsten with LBB-12 or
Wratten 80B; ASA 200 / 24 DIN with CC-20M + CC-20B; ASA 250 / 25 DIN with CC-20B;
ASA 200 / 24 DIN with CC-30M + CC-10R. Fujichrome R100: ASA 100 / DIN 21 daylight,
ASA 32 / DIN 16 tungsten, indices referenced to 1/250 s.

This is the closest thing in the book to a statement about a colour film's spectral
balance, expressed the way a 1979 photographer needed it.

### 3.5 TIER 2 — plots requiring tracing, with printed axis calibration

`digitize_plot.py` can consume these; each needs the usual supervised setup (page, crop,
axis values, one seed pixel per curve).

| Class | PDF pages | Count | Value |
|---|---|---|---|
| Wedge spectrograms (Ilford, tungsten 2850 K) | 471, 474, 478, 488, 496, 501 | 6 | **Real monochrome sensitisation curves** — feeds the live spectral path |
| Spectral sensitivity plots (Polaroid, Fuji) | 382, 393, 552, 583–595 | 14 | log sensitivity vs nm, axes printed |
| Characteristic curves | 60, 70, 76, 82, 87, 89, 92, 107, 147, 386, 404, 406, … | 24 | D vs relative log E |
| Contrast index curves | 15, 59, 71, 82, 84, 87, 89, 92, 476, 489 | 10 | CI vs development time |
| Reciprocity charts (graphical) | 174–175, 385, 391, 471, 475, 479, 483, 490, 497, 582 | 19 | nominal vs actual exposure time |

The Polaroid Type 51 spectral sensitivity plot even prints its own definition —
"spectral sensitivity equals reciprocal of exposure (ergs/cm²) required to produce a 0.6
visual density" — and the page opposite gives a four-step procedure for integrating it
against an arbitrary illuminant at 10 nm intervals. That is the same integral
`AlgoSpectralSensitivity` performs, described in 1979 terms.

### 3.6 Methodology definitions — confirms our schema semantics (PDF 176–177)

Worth recording because it retro-justifies field definitions we adopted from modern
Kodak sheets:

* **RMS granularity**: 1000 × standard deviation of density, **48 µm aperture**, ANSI
  diffuse visual density (PH2.19-1959), correlating with **12× monocular** viewing.
  Reversal and direct-duplicating films measured at *gross* diffuse density 1.00;
  negative, internegative, slide and print films at *net* diffuse density 1.00.
* **Resolving power**: quoted at **two test-object contrasts, 1.6:1 and 1000:1** — the
  exact pair `_RESOLVING_POWER` stores as a tuple. With the classification ladder:
  ≤50 Low, 63–80 Medium, 100–125 High, 160–200 Very High, 250–500 Extremely High,
  ≥630 Ultra High.
* **Granularity classification ladder**: 45–55 Very Coarse, 33–42 Coarse, 26–30
  Moderately Coarse, 21–24 Medium, 16–20 Fine, 11–15 Very Fine, 6–10 Extremely Fine,
  <5.5 Micro Fine — with the explicit warning that negative and reversal films must not
  be intercompared.
* **MTF**: exposed to sinusoidal patterns at nominal 35 % aerial image modulation, and
  the caution that measured photographic MTF includes development adjacency effects and
  is *not* the emulsion's true optical MTF. Our `MTFSpec` carries a separate `adjacency`
  term, which this justifies.

This does not change any value. It means the reversal-vs-negative measurement-density
distinction and the two-contrast resolving-power convention are 1970s industry standard,
not modern Kodak idiosyncrasies.

## 4. What this book does NOT contain

Stated so nobody goes looking:

* **No per-layer spectral sensitivity curves for colour negative or reversal films.**
  The thing the spectral path most wants is absent. Colour films get illuminant/filter
  tables instead.
* **No per-film RMS granularity numbers** — only the definition and the classification
  ladder. Descriptions say "extremely fine grain" in words.
* **Resolving power for barely a dozen products**, nearly all Polaroid.
* **No dye absorption spectra** and no interimage / DIR coupler data.
* **Nothing after 1979.** My automated name-match reported 33 of our 131 stocks as
  "mentioned", but several are impossible: `ILFORD_DELTA_3200` (1990s), `KODAK_GOLD_100`
  (1986), `ILFORD_HP5_PLUS_400` (1989), the `KODAK_VISION` line (1996+), the T-MAX line
  (1986+). Those matched on substrings — "DELTA", "PAN 400", "5246" — and must be
  discarded. Genuine pre-1979 overlaps are Panatomic-X, Plus-X Pan, Plus-X 5231,
  Double-X 5222, Super-XX, Verichrome Pan, Tri-X (several), Ektapan, Royal-X Pan,
  Ilford HP3 and HPS, Ektachrome and Kodachrome families, Agfacolor, Gevaert.

## 5. OCR failure modes — two confirmed, one already caught

**5.1 Digit splitting.** The OCR breaks numerals with internal spaces: "125" appears as
"1 25", "1250" as "1 250", "DIN 18" as "DIN 1 8". Any regex harvesting `ASA\s*(\d+)`
silently truncates to the first fragment. Several of my first-pass speed reads came back
as "ASA 1".

**5.2 Cross-heading bleed — one wrong reading already produced and corrected.** A
pattern-match for Super-XX Pan's speed returned **ASA 400**. Reading PDF page 71 showed
that 400 belongs to **Kodak Tri-X Pan**, whose block begins halfway down the same page;
Super-XX's own figure is elsewhere. Our `EASTMAN_SUPER_XX_1938` carries EI 100 and was
**not** changed. Caught by reading the page rather than trusting the match — the same
discipline that caught the scrambled Cheltsov table yesterday.

**5.3 Multi-column table scrambling — not yet resolved, and it blocks §3.2.** The Kodak
reciprocity master table on PDF 174–175 is a 12 × 7 grid whose cells arrive in flat text
without reliable row/column association; fragments of the rotated running head
("I >IYOO>I NYVUSY3" — *EASTMAN KODAK* mirrored) are interleaved. **This table must be
rebuilt from word coordinates before any value is taken from it**, exactly as Table 24
was yesterday. Until that is done, no reciprocity figure from this book should enter the
database.

## 6. Recommended extraction order

| Pri | Work | Yield | Effort | Blocked by |
|---|---|---|---|---|
| **P1** | Polaroid batch — ~12 new stocks, upgrade the 3 existing | D-max, D-min, slope, ASA+DIN dual, resolution, reciprocity, all printed as numerals | Medium | nothing |
| **P2** | `onset_s` + reciprocity channel ordering for `KODACHROME_64`, `EKTACHROME_64`, `EKTACHROME_160T` | Upgrades an estimated field on existing stocks | Low | §5.3 — rebuild table from coordinates first |
| **P3** | Ilford dual daylight/tungsten EI, filter factors; tier upgrade for `ILFORD_HP3` / `ILFORD_HPS` | Documented speeds replacing tier-2 estimates | Low | nothing |
| **P4** | Trace 6 Ilford wedge spectrograms | Real monochrome sensitisation curves → live spectral path | High (supervised) | needs your involvement per trace |
| **P5** | Fuji illuminant EI ladders | Colour-balance evidence for Fuji stocks | Low | nothing |
| **P6** | CI-vs-development-time tables, developer-dependent speeds | PRC processing axis | — | axis does not exist in schema |

## 7. Schema gap this book newly exercises

**Dual daylight/tungsten exposure index.** The book gives both for most films — FP4
125/100, Plus-X 5231 80/64, Double-X 5222 250/200, 4-X 5224 500/400, Tri-X Reversal 7278
200/160, Polaroid Type 51 320/100. We store a single `exposure_index`.

For a *monochrome* film the daylight/tungsten pair is not redundant: it is a compact
statement about the emulsion's spectral weighting, since a film whose sensitivity is
biased blue loses more speed under tungsten. Plus-X's 80/64 ratio of 1.25 and Type 51's
320/100 ratio of 3.2 differ by that much because Type 51 is blue-sensitive only. This is
measurable, documented, and physically meaningful, and there is nowhere to put it. It
also partially duplicates what a traced wedge spectrogram would give more completely,
so the two should be decided together rather than separately.

## 8. Verdict

Substantial, and complementary to what we already have rather than overlapping it. The
strongest single opportunity is **Polaroid**: our thinnest-covered manufacturer, and the
one place in this book where D-max, D-min, slope, speed and resolution are all printed
as numerals for a dozen products, including a Type 55 P/N negative at 150–160 lines/mm
that would be exceptional for an instant material. (Assessed after extraction: it
is NOT the sharpest stock — several conventional stocks are documented at 200 lp/mm
at a stated 1000:1 test-object contrast, where this figure states no contrast.)

The reciprocity master table is the most *interesting* find — it would upgrade an
estimated field on three stocks we already ship — but it is the one item gated on
rebuilding a scrambled table from word coordinates first, and its half-stop rounding
means it can supply `onset_s` and a channel ordering, not a fitted exponent.

**Nothing was modified.** Tell me which priorities to run and I will start there.
