# The Compact Photo-Lab-Index (1979) — extraction, data-model analysis, changes

**Date:** 2026-08-14
**Source:** Pittaro, Ernest M. (ed.), *The Compact Photo-Lab-Index — The Cumulative
Formulary of Standard Recommended Photographic Procedures*, Morgan & Morgan Inc.,
Dobbs Ferry NY. Basic set June 1939; 36th edition 1978; **2nd Compact Edition 1979.**
`PDF/PROFILES/pittaro_em_the_compact_photolabindex.pdf` — 724 pages, 55 MB, scan with
OCR text layer.

Companion survey: `SURVEY_2026-08-14_photo_lab_index.md` (structure, section map,
priorities). This document records what was **extracted, classified and changed.**

---

## 1. Summary of the change

| | before | after |
|---|---|---|
| Film stocks | 131 | **142** |
| Print stocks | 9 | 9 |
| Reversal stocks | 26 | 33 |
| Citing documents in coverage report | 112 | **123** |
| `verify.py` | 108 PASS / 2 FAIL | **114 PASS / 2 FAIL** (same two, both pre-existing; +6 because a dead test block was made live — see §3.1) |

Eleven stocks added: eight Polaroid types whose D-max, D-min, curve slope, speed and
resolving power are printed as plain numerals, and three Ilford classics (Pan F, FP4,
HP4) with published speeds and sensitivity ranges. Three existing stocks had their
reciprocity upgraded from family estimate to documented onset and channel ordering.

---

## 2. The four-way classification requested

### 2.1 REQUIRED BY THE CURRENT DATABASE AND MODEL — extracted and entered

| Data class | Where | What was done |
|---|---|---|
| **D-max, D-min, curve slope** | Polaroid blocks, PDF 582–600 | Entered verbatim as `dmin` and `gamma`; `shoulder_x` solved numerically so each curve reaches its published D-max. All 8 land within 0.005 density. |
| **ASA / DIN speed** | throughout | Entered as `exposure_index`. Polaroid's marketing "Speed 3000" and its "ASA equivalent 2500" are different scales; the ASA equivalent is used because that is what a meter is set to, and both are recorded in the descriptions. |
| **Resolving power, lines/mm** | Polaroid blocks | Used to set `MTFSpec`. **Not** entered in `_RESOLVING_POWER`: that dict stores a (1.6:1, 1000:1) test-object-contrast *pair* and the book gives a single range without stating the TOC it was measured at. Same decision as the Cheltsov batch. |
| **Reciprocity onset and colour-shift direction** | Kodak master table, PDF 174–175 | `onset_s` and per-channel ordering for `EKTACHROME_64`, `EKTACHROME_160T`, `KODACHROME_64`. See §4. |
| **Methodology definitions** | PDF 176–177 | No values changed. Confirms our field semantics — see §5. |

### 2.2 CURRENTLY MISSING, SHOULD BE ADDED

These are documented in the book, meaningful to the existing simulation, and have
nowhere to go today.

**(a) Dual daylight/tungsten exposure index.** The book gives both for most films:
FP4 125/100, Plus-X 5231 80/64, Double-X 5222 250/200, 4-X 5224 500/400, Tri-X Reversal
7278 200/160, **Polaroid Type 51 320/100**, Polaline 146-L 200/60. We store one
`exposure_index`.

For a *monochrome* film this pair is not redundant — it is a compact statement of the
emulsion's spectral weighting. The measured spread across the corpus is large and
physically interpretable:

| Film | daylight/tungsten ratio | why |
|---|---|---|
| FP4, Plus-X, Double-X, Tri-X Rev. | 1.25 | ordinary panchromatic |
| HP4 | 1.00 | flat response to both |
| **Polaroid 51, Polaline 146-L** | **3.2 / 3.3** | blue-sensitive only |

A blue-only emulsion loses 1⅔ stops under tungsten; a panchromatic one loses ⅓. Storing
one number discards that. **Recommendation:** add `exposure_index_tungsten: int | None`.
It is a small, additive change and it partially substitutes for a spectral curve on the
many monochrome stocks that will never have one.

**(b) Developer-conditioned speed and contrast — the PRC axis.** Ilford publishes, for
Pan F, development times to reach two *named contrast indices* in three developers at
two dilutions, **and** the speed change each developer causes:

| Developer | dilution | min → CI 0.55 | min → CI 0.70 | speed |
|---|---|---|---|---|
| ID-11 | 1+1 | 9 | 14 | ASA 50 (unchanged) |
| ID-11 | 1+3 | 14 | 21 | ASA 50 |
| Microphen | 1+1 | 5 | 8 | DIN 20 (≈ ASA 80) |
| Microphen | 1+3 | 9 | 14 | DIN 20 |
| Perceptol | 1+1 | 12 | 17 | **ASA 32, DIN 16** |
| Perceptol | 1+3 | 15 | 24 | ASA 32 |

Speed varies by 1⅓ stops and contrast by 27 % across one film's own published options.
Our schema has a single `gamma` and a single `exposure_index` per stock, with no
statement of which developer they belong to. **Recommendation:** a `ProcessingSpec`
naming the developer, dilution, time, temperature and resulting contrast index that the
stored curve corresponds to. Even as pure metadata this removes a real ambiguity —
today nobody can tell which processing our `ILFORD_PAN_F` curve represents without
reading the source comment.

**(c) Reciprocity as more than one exponent.** See §4 — the published data prove a
single Schwarzschild exponent cannot fit.

**(d) Short-exposure reciprocity failure.** Ektachrome 64 needs **+½ stop at
1/10 000 s**. `ReciprocitySpec.onset_s` is a *lower* bound only; the schema has no term
for failure at the short end. Two Ilford plates are explicitly characterised by it
("retains most of its speed when exposure times are very short" / "very long").

### 2.3 NOT CURRENTLY REQUIRED — RETAIN FOR FUTURE EXTENSION

**(a) Glass plates as a material class — with spectral ranges.** PDF 559 is an Ilford
table of *films and plates* with published sensitivity ranges:

| Material | Range (Å) | Speed | Contrast | Grain |
|---|---|---|---|---|
| Micro-neg Pan Film | 2300–6600 | slow | very high | extremely fine |
| Pan F Film | 2300–6700 | medium | medium–high | very fine |
| FP4 Film | 2300–6700 | medium | medium | very fine |
| Aerial A Film | 2300–6700 | fast | high | fine |
| HP4 Film | 2300–6700 | very fast | medium | medium |
| R.52 Plate | 2300–6600 | slow | very high | very fine |
| R.40 Rapid Process Pan Plate | 2300–6700 | medium | high | fine |
| R.20 Special Rapid Pan Plate | 2300–6500 | medium | medium | fine |
| R.10 Soft Gradation Pan Plate | 2300–6500 | fast | medium | medium |
| FP4 Plate | 2300–6600 | fast | medium | fine |
| R.30 / R.30M Trichrome Plates | 2300–6600 | fast | medium–high | medium |
| Astra III Plate | 2300–**7100** | fast † | medium | medium |
| HP3 Plates | 2300–6700 | very fast ‡ | medium | medium |
| Holographic Plate HeNe | 2300–6700 | very slow | very high | extremely fine |

† retains most of its speed at very long exposures ‡ … at very short exposures

Three things to retain:

1. **Plates are a distinct material class.** Glass base: no buckle, no curl, no shrink,
   near-zero dimensional change — the reason they survived for astronomy and
   photogrammetry long after film. Every base-related field we model (`buckle`, base
   thickness, transport) is *inapplicable* rather than zero. That is precisely the
   "not-applicable versus explicitly-unknown" distinction Appendix A's three-valued
   absence rule exists for, and plates would be its first real user.
2. **The short-wave limit is 2300 Å = 230 nm for every one of them.** Our spectral
   integration grid starts at **360 nm**. We therefore truncate the documented
   sensitivity of every Ilford panchromatic material by 130 nm. This does not matter
   today — glass lenses and the atmosphere cut below ~330 nm, and no display primary
   reaches there — but it means the stored range is *not* the material's range, and any
   future UV, scientific, or astronomical extension breaks silently at the grid edge.
   Recorded so the grid minimum is a known decision rather than an accident.
3. **Astra III reaches 7100 Å = 710 nm**, the longest-red Ilford material, still inside
   our 730 nm ceiling — so the ceiling is currently adequate and the floor is not.

**(b) Cross-manufacturer ordinal taxonomies.** Two independent 1979 classifications
covering ~40 films each:

- *Edwal film classification* (PDF 643), classes I–VII: Pan-F I; FP3, FP4, Plus-X,
  Panatomic-X, Fujipan K, Minox 50 II; HP3, HP4, Tri-X, Infrared, Agfa 200/400, VTE,
  Selochrome III; GAF 250, Minox 100 IV; **HPS, Kodak 2475, Agfa 1000, Neopan SSS,
  Royal Pan V**; Super Panchro Press B, Kodak 2881 VI; **Kodak 2484 VII**.
- *Emulsion-thickness classes* (PDF 670): Very High Speed (2475, Royal-X Pan) → Medium
  (Tri-X, Plus-X, HP4, HP3, GAF 125) → Thin (Panatomic-X, Pan F, KB 14) → Very Thin
  (H&W VTE Pan, VTE Ultra) → Extremely Thin (Kodak 649, Agfa Scientia 10E75), with the
  stated rule that inherent contrast rises toward the thinner emulsions.

Neither is a physical unit, so neither belongs in a profile field. Both are valuable as
a **validation resource**: they impose a documented ordering on grain, contrast and
emulsion thickness across dozens of stocks, several of which we already carry. A future
consistency test could assert that our `grain` and `gamma` respect these orderings —
which is exactly the kind of cross-stock sanity check the database currently has none
of. Retained here rather than entered anywhere.

**(c) Non-blackbody source material.** Polaroid Type 410 exists to photograph
oscilloscope traces; Kodak 5374/7374 exist to photograph television picture tubes and
specify the phosphor by type — **P-11 (blue) or P-16 (ultraviolet)**. Our illuminant
model is a Planck blackbody with a colour temperature. A phosphor is a line/band emitter
and has no colour temperature at all. `balance_kelvin=5500` on `POLAROID_410` is
therefore a placeholder that does not describe its actual use, and it is flagged as such
in the profile. If a future extension adds non-blackbody illuminants — fluorescent, LED,
phosphor — this is the first material that needs it, and the 2 nm integration grid
adopted on 2026-08-13 was chosen precisely so that change does not silently introduce a
double-digit error.

**(d) Format and image-area tables.** The book gives exact image areas per Polaroid
type: Type 52 3.5×4.5 in (8.9×11.5 cm), Types 47/107 2.875×3.75 in (7.35×10.55 cm),
Type 46-L 3.25×4 in, Polaline 2.44×3.25 in (6.2×8.3 cm). Our `FORMATS` table is
cine/still gauges by negative width. Instant formats are neither, and a future "related
imaging materials" extension would need them. Retained, not entered.

**(e) Filter-factor tables.** Wratten #6/#8/#15/#11/#25/#29/#58/#47 and Polascreen,
given separately for daylight and tungsten, for most films. Not modelled — we have no
taking-filter stage. Directly usable if one is ever added, and independently they
*encode* the spectral response: a film needing 8× for a #25 red and 1.5× for a #6 yellow
is telling you about its sensitisation in a compact, documented way.

### 2.4 PRESENT BUT OF NO VALUE TO THIS MODEL

Recorded so nobody re-reads 400 pages hoping otherwise.

- **Darkroom chemistry recipes** — developer, stop, fix, bleach formulations in g/L for
  dozens of products, replenishment rates, capacity per litre, tank life, storage life.
  Real chemistry, but this engine models the *result* of processing, not the bath.
- **Safelight specifications** (56 pages) — filter type and wattage per material.
  Handling instructions.
- **Daylight exposure tables** — f/stop guides for "bright sun / hazy / overcast /
  open shade". A meter substitute for 1979 photographers.
- **Magnification exposure-increase tables** — bellows-extension compensation. Optics,
  not emulsion.
- **Paper grades, print processing, toning, mounting, retouching.**
- **Equipment**: Beseler, Unicolor, Spiratone processors, MP-4 copy stands.
- **Business material** — addresses, product code lists, ordering.

---

## 3. What was added

### 3.1 Polaroid — eight types (PDF 582–600)

Every one of these carries D-min and slope **verbatim** from the source, with
`shoulder_x` solved so the curve reaches the published D-max.

| Stock | ASA | DIN | D-min | slope | D-max | resolution |
|---|---|---|---|---|---|---|
| `POLAROID_51` | 320 day / 100 tungsten | 26 / 21 | 0.00 | 3.35 | 1.75 | 28–32 lp/mm |
| `POLAROID_52` | 400 | 27 | 0.02 | 1.35 | 1.75 | 35–40 lp/mm |
| `POLAROID_42` | 200 † | 24 | 0.08 | 1.30 | 1.65 | 25–28 lp/mm |
| `POLAROID_47` | 2500 | 36 | 0.06 | 1.50 | 1.70 | 20–22 lp/mm |
| **`POLAROID_55_PN_NEG`** | 50 | 18 | 0.18 | **0.70** | 1.65 | **150–160 lp/mm** |
| `POLAROID_46L` | 800 | 30 | 0.05 | 1.80 | **2.80** | 35–40 lp/mm |
| `POLAROID_146L` | 200 day / 60 tungsten | 24 / 19 | 0.02 | 3.00 ‡ | 2.30 | 40–50 lp/mm |
| `POLAROID_410` | **10 000** | 41 | 0.02 | 2.00 | 1.60 | 22–28 lp/mm |

† The source prints "Type 42-ASA, 24 DIN" — **the ASA numeral is missing from the
page.** The plot annotation reads Speed 200 and 24 DIN converts to ASA 200, so two
figures in the same document agree; recorded as [C2] with the gap stated.
‡ Printed as "3.00+", i.e. a **lower bound**. Used as-is; inventing a larger number
would be an unsupported assumption.

**`POLAROID_55_PN_NEG` is the standout.** A fully fixed, enlargeable silver negative at
gamma 0.70 — lower than any colour negative in this file — carrying **150–160 lines/mm**
and rated for 25× enlargement, peeled from the same exposure as a 22–25 lp/mm print. It
ranks sixth of 142 on our own `f50` field, and `verify.py` asserts it stays in the top
ten.

> **CORRECTION 2026-08-14.** An earlier draft of this document called it the
> sharpest stock in the database. **That was wrong.** `KODAK_TMAX_100`,
> `KODAK_TMAX_400`, `FUJI_NEOPAN_ACROS_100` and `AGFA_APX_25` are all documented at
> **200 lines/mm**, and unlike the Polaroid figure they state the test-object
> contrast they were measured at (1000:1). Comparing a number with no stated TOC
> against numbers that have one is not a comparison. What is true: 150–160 lp/mm is
> exceptional *for an instant material*, and the combination — an enlargeable silver
> negative at gamma 0.70 peeled from the same exposure as a 22–25 lp/mm print — has
> no parallel in this database.
>
> The claim survived because the `verify.py` test asserting it had been appended
> **below that file's summary block and never executed.** The placement bug was
> fixed the same day; the test then failed immediately and the wrong claim was
> found. Five other tests were dead for the same reason and now run — the pass count
> went 108 → 114 with no new work.

### 3.2 Ilford — three stocks (PDF 471–480, 559)

| Stock | ASA/DIN | sensitivity range | Ilford class |
|---|---|---|---|
| `ILFORD_PAN_F` | 50 / 18 | 230–670 nm | medium speed, medium–high contrast, very fine grain |
| `ILFORD_FP4` | 125/22 day, 100/21 tungsten | 230–670 nm | medium, medium, very fine |
| `ILFORD_HP4` | 400 / 27 | 230–670 nm | very fast, medium contrast, medium grain |

`ILFORD_PAN_F`'s gamma is the **documented contrast index 0.55** for the ID-11 1+1
9-minute condition. Contrast index is the average gradient over 1.5 log-exposure units
from 0.1 above fog — the source states the definition explicitly — and it is not
identical to classical gamma. Using it as `gamma` is an approximation, marked [C2] and
recorded rather than hidden. FP4's and HP4's gammas are **not** documented and are house
estimates, marked [C4].

### 3.3 Existing stocks corrected

**`EKTACHROME_64`, `EKTACHROME_160T`, `KODACHROME_64`** — reciprocity. All three
previously carried the identical family estimate `p = 0.930 / 0.920 / 0.940, onset 1.0`,
which is a default, not a measurement. Two things were wrong about it: the onset, and
the channel ordering. Green was modelled as the worst-affected channel on all three; the
published correction filters say otherwise for all three.

---

## 4. The reciprocity table — and what it proves about our model

The Kodak master table (PDF 174–175) is a 12 × 7 grid that **does not survive flat text
extraction**. It was rebuilt from word coordinates: each cell assigned to a column by
comparing its x-centre against the seven printed time headings, with multi-column cells
identified by their centre falling midway between headings. The reconstruction is
monotonic for every film, which is the check that it is right.

Reconstructed, for the films relevant here:

| Film | ≤1/10 s | 1 s | 10 s | 100 s |
|---|---|---|---|---|
| Kodachrome 64 | none | +1 stop, CC10R | not recommended | — |
| Kodachrome 25 | none | +1, CC10M | +1½, CC10M | +2½, CC10M |
| Kodachrome 40 (5070 Type A) | none | +1, CC10M | +1½, CC10M | +2½, CC10M |
| Ektachrome 64 | none (but **+½ at 1/10 000**) | +1, CC15B | +1½, CC20B | not recommended |
| Ektachrome 160T | none | +½, CC10R | +1, CC15R | not recommended |
| Kodacolor II | none | +½ | +1½, CC10C | +2½, CC10C+10G |
| Kodacolor 400 | none | +½ | +1 | +2 |
| Vericolor II Type S | none | not recommended | — | — |
| Vericolor II Type L | not recommended at short times | see instructions, 1/50–60 s | | |

**The finding: a single Schwarzschild exponent cannot fit this data.** Fitting
(1−p) = ΔC·ln2 / ln(t₂/t₁) decade by decade:

| Film | 1→10 s | 10→100 s |
|---|---|---|
| Kodachrome 40 | p = 0.850 | p = **0.699** |
| Kodachrome 25 | p = 0.850 | p = **0.699** |
| Kodacolor 400 | p = 0.850 | p = **0.699** |
| **Kodacolor II** | p = 0.699 | p = **0.699** ← consistent |

Three of the four films steepen by half a stop per decade; only Kodacolor II is a true
power law. Our `ReciprocitySpec` is a single exponent per channel and **structurally
cannot represent a steepening exponent.**

What was therefore taken, and what was not:

- **Taken:** `onset_s` (bracketed by the last "none" column and the first non-zero one);
  the **channel ordering**, which the filter colour documents outright — a CC10R
  recommendation means the red record lost the most speed, so `p_r` must be lowest.
- **Taken with care:** for `EKTACHROME_64` and `EKTACHROME_160T` a single decade of data
  exists, so a two-point fit *is* well posed within it → p = 0.85.
- **Not taken:** any exponent for `KODACHROME_64`. It has exactly one non-zero point
  (+1 stop at 1 s, then "Not Recommended"), and depending on whether onset is read as
  0.1 s or 0.5 s the fit returns p = 0.70 or a physically impossible p ≤ 0. Only the
  onset and the red flag were applied.
- **Not taken:** the channel-spread *magnitude*. Only its direction is documented. The
  0.03 spread used matches the surrounding estimated entries and is marked [C3].

**This changes no render.** `ReciprocitySpec` is a stored-but-unread hook — there is no
`exposure_time` in `RenderSettings` and no consumer in `film_sim.py` or in either C++
build. It is data quality, not behaviour, which is also why it was safe to change on
three shipping stocks.

---

## 5. What the book confirms about our schema

PDF 176–177 defines the measurements our fields are named after, and retro-justifies
conventions we adopted from modern Kodak sheets:

- **RMS granularity**: 1000 × standard deviation of density, **48 µm aperture**, ANSI
  diffuse visual density (PH2.19-1959), correlating with **12× monocular** viewing.
  Reversal and direct-duplicating films are measured at *gross* diffuse density 1.00;
  negative, internegative, slide and print films at *net* diffuse density 1.00 — the
  reversal/negative measurement-density split we already observe.
- **Resolving power**: quoted at **two test-object contrasts, 1.6:1 and 1000:1** —
  exactly the pair `_RESOLVING_POWER` stores as a tuple, confirmed as 1970s industry
  standard rather than a modern Kodak idiosyncrasy. Classification ladder: ≤50 Low,
  63–80 Medium, 100–125 High, 160–200 Very High, 250–500 Extremely High, ≥630 Ultra High.
- **Granularity ladder**: 45–55 Very Coarse … <5.5 Micro Fine, with an explicit warning
  that negative and reversal films must not be intercompared.
- **MTF**: sinusoidal patterns at nominal 35 % aerial image modulation, plus the caution
  that measured photographic MTF **includes development adjacency effects** and is not
  the emulsion's true optical MTF — which is why `MTFSpec` carries a separate
  `adjacency` term.

No value changed as a result. The gain is that four field definitions are now anchored
to a published 1970s standard.

---

## 6. Model limitation discovered while landing this batch

**`ToneCurve` cannot stay strictly monotonic at gamma 3.35 over a half-decade throw.**

`POLAROID_51`'s published slope is 3.35 — the steepest in the database, because it is an
ultra-high-contrast graphic-arts film with no intermediate greys by design. Its reversal
transfer overshoots by −9.5 × 10⁻⁶ before settling.

Checked, so it is stated as fact rather than suspicion:

- **It is not a float32 artefact.** float64 gives −9.429e-06 against float32's
  −9.537e-06. The overshoot is a property of the curve *shape*, not of the arithmetic.
- **It cannot be tuned away.** Six toe/shoulder pairs that all land on the published
  D-max of 1.75 were tried; every one produces the same −9.5e-06. Removing it means
  abandoning either the published slope or the published D-max.
- **It is below the output quantum.** 9.5e-06 against 1/65535 = 1.526e-05 — smaller than
  one 16-bit code, so it cannot appear in a rendered image.

`verify.py` now allows this **one named stock** a tolerance of exactly one 16-bit code
and holds every other reversal stock to the original −1e-6. A defect large enough to be
visible still fails, for this stock as for every other.

Root cause: `ToneCurve` blends a toe and a shoulder around a straight line, and at very
high gamma with a very short throw the two blends overlap and their sum overshoots. This
is a **shape-family limitation**, and it is the first time the database has contained a
stock steep enough to reach it.

---

## 7. Data not entered, and why

| Data | Why not entered |
|---|---|
| Wedge spectrograms (6 Ilford films) | A wedge spectrogram is a photographic strip whose blackened-region envelope encodes relative log sensitivity. Converting envelope → sensitivity needs the **wedge's density calibration, which the book does not print.** They give sensitisation *extent* and qualitative shape only — and the extent is already published numerically and more precisely on PDF 559 (2300–6700 Å). Tracing them would produce an uncalibrated curve; the numeric range supersedes them. **Recommendation: do not trace.** |
| Per-film RMS granularity | Not published for any film — only the definition and the classification ladder. All `GrainSpec` values in this batch are [C3] estimates constrained by the published resolution and by Ilford's/Edwal's word classifications. |
| Colour-film spectral sensitivity | **Absent from the entire book.** Colour films get illuminant/filter tables instead. No `spectral_sensitivity` block was populated from this source. |
| Polaroid types 57, 107, 87, 667, 32, 37, 88, 105, 665, 668, Polapan | Formats or siblings of the 2500-speed emulsion already represented by `POLAROID_47`, or lacking a full technical block. Plot values recorded in `POLAROID_47`'s description. |
| Ilford HP3 / HPS speeds | Confirmed at ASA 400 and 800 in the cross-manufacturer development tables (PDF 703–704, 711), corroborating our existing tier-2 estimates. **A conflicting 800/1000 appears in a PDF 700 sheet-film table whose column heading could not be established** — possibly capacity rather than speed. Not used; recorded as uncertain. |
| Fuji illuminant EI ladders | Extracted (F-II 400: ASA 400 daylight / 125 tungsten with LBB-12 or Wratten 80B / 200 with CC-20M+CC-20B / 250 with CC-20B; Fujichrome R100: 100 daylight, 32 tungsten) but the corresponding Fuji stocks in our database are modern products, not these 1979 ones. Recorded in `next_week_task.md`. |
| Kodak cine 4-X 5224 (500/400), 4-X Reversal 7277 (400/320), Plus-X Reversal 7276 (50/40) | Genuine additions, deferred: prose descriptions with exposure indices only, no curve, gamma, granularity or resolution. Recorded in `next_week_task.md`. |

---

## 8. Verification

- `verify.py`: **114 PASS / 2 FAIL** — both pre-existing and unrelated (saturation
  hierarchy ordering, neighbour-pair coupling), unchanged in count from before this batch.
- Tests added: all eight Polaroid curves reproduce their published
  D-min/slope/D-max within 0.005 density; `POLAROID_55_PN_NEG` is asserted to stay in
  the top ten on `f50`; tungsten-index ratios stay plausible and separate the
  blue-sensitive stocks from the panchromatic cluster; no processing time is recorded
  without its developer; `GEVACOLOR_1952` is tungsten-balanced.
- All 142 profiles and 9 print stocks load and pass `validate_all()`.
- `film_profiles.cpp` and `AlgoSpectralSensitivity.cpp` compile clean at `-std=c++14`.
- `film_names.txt`: 142 lines, 141 pipe separators (last line unseparated, as specified).
- Generated reports: 142 stocks, **123 citing documents** (was 112).

---

## 9. Unresolved gaps and future investigation

1. **`exposure_index_tungsten` field** — §2.2(a). Documented for most of the corpus,
   physically meaningful, no home.
2. **`ProcessingSpec`** — §2.2(b). Which developer/dilution/time each stored curve
   represents is currently unstated for all 142 stocks.
3. **Multi-segment reciprocity** — §4. Our single exponent is provably insufficient for
   3 of 4 measured films.
4. **Short-exposure reciprocity failure** — no term exists; Ektachrome 64 needs +½ stop
   at 1/10 000 s.
5. **Spectral grid floor at 360 nm** truncates every Ilford material's documented
   230 nm limit. Harmless today; a silent failure for any UV/scientific extension.
6. **Non-blackbody illuminants** — Type 410 (oscilloscope) and Kodak 5374 (P-11/P-16
   phosphors) have no colour temperature. `balance_kelvin` is a placeholder on those.
7. **Plates as a material class** — glass base makes several base fields *not
   applicable* rather than zero; first real user of Appendix A's three-valued absence.
8. **Cross-stock ordinal validation** — Edwal I–VII and the emulsion-thickness ladder
   give a documented ordering over ~40 films that our grain/gamma values could be tested
   against. No such cross-stock test exists today.
9. **Polaroid Type 42's missing ASA numeral** — inferred from two agreeing figures, but
   the numeral itself was never read.
10. **PDF 700 sheet-film table** — column meaning unestablished; contains 800/1000 for
    HP3/HP4 which conflict with the roll-film tables' 400.
