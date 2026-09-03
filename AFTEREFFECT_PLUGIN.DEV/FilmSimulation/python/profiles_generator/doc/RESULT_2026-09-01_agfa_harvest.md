# AGFA harvest — `agfa_films.pdf` (1998) and `agfa_bw_manual.pdf` (2004)

**Date 2026-09-01. Investigation and proposal. NOTHING HAS BEEN WRITTEN TO THE DATABASE.**
`film_profiles.py`, `verify.py`, `build.py` and every MD document are unchanged. One new file
exists, the reader `agfa_1998_curves.py`, which is not yet registered as an audit.

---

## 0. The headline, before the numbers

⚠ **`agfa_films.pdf` IS NOT THE DOCUMENT THE PROJECT RECORDED IT AS, AND THAT MISTAKE HID A WHOLE
EDITION.** `NotFound.md` row 5 and queue **G6** both state that the four Agfa candidates are "the
same publication, two of them byte-identical". The md5s:

| file | internal title | creator | created | md5 |
|---|---|---|---|---|
| `agfa_films.pdf` | **Technical data PF** | PageMaker **6.5** | **1998-10-20** | `edb3dd17…` |
| `AGFA stocks.pdf` | F-PF-E4 | PageMaker 7.0 | 2003-07-18 | `bf9f0c1a…` |
| `FPD1e.pdf` | F-PF-E4 | PageMaker 7.0 | 2003-07-18 | `bf9f0c1a…` |
| `Datasheet_F_PF_E4.pdf` | F-PF-E4 | PageMaker 7.0 | 2003-07-18 | `f693b562…` |

The byte-identical pair is real. The fourth file is **a different edition five years older** —
p12 prints `Technical Data PF / Date: 09/1998 / 1st edition`, against the others' `F-PF-E4 /
08/2004 / 4th edition`. Two editions is a **cross-edition consistency check**, which is precisely
what G6 says it does not have.

⚠ **A SECOND STANDING CLAIM IS ALSO FALSE.** `agfa_2004_curves.py`'s docstring says
`AGFA_OPTIMA_100`'s spectral set was traced "from a RASTER page of the older 1998 brochure".
`agfa_films.pdf` has **zero embedded images on all twelve pages**; pp7–10 carry 116–172 stroked
vector objects. It was transcribed by eye in the 2026-08-02 batch because nobody ran
`get_images()` on it — the same defect that was corrected for the APX spectral sets on 2026-08-17.

⚠ **THE 1998 EDITION IS THE ONLY DOCUMENT IN THE CORPUS THAT PLOTS `AGFACOLOR ULTRA 50` AND
`AGFAPAN APX 25` AT ALL.** The 2004 edition dropped both.

---

## 1. What the two documents are

**`agfa_films.pdf`** — Agfa-Gevaert, *Technical Data PF — Agfa range of films*, 1st edition,
09/1998, 12 pp, 100 % vector, real text layer. Twelve films, four plotted panels each on pp7–10,
tabulated characteristic values beside each column, reciprocity for all twelve on p6, and the full
APX developer/time/temperature tables on p11.

**`agfa_bw_manual.pdf`** — Agfa-Gevaert, *Black & White Handbook*, 2004, 69 pp, mixed vector and
raster. By 2004 the B&W line was down to three films (APX 100, APX 400, Scala 200x); APX 25 is
gone. Pages 10–13 are the film technical data, 15–31 the papers, 33–48 the film developers, 49–55
the paper developers. Its curves are drawn in a red ink at a third scale — a genuinely independent
third artwork for the two surviving APX films.

---

## 2. Tabulated values — direct text extraction, no digitisation

All twelve films, `agfa_films.pdf` pp7–10. **RP** = resolving power, lines/mm.

| film | profile | ISO | RMS ×1000 | RP 1000:1 | RP 1.6:1 | layer µm | base |
|---|---|---|---|---|---|---|---|
| OPTIMA II 100 | `AGFA_OPTIMA_100` | 100/21° | 4.0 | 140 | 50 | 16 | 135 = 120 µm, 120 = 95 µm, sheet = PET 175 µm |
| OPTIMA II 200 | `AGFA_OPTIMA_200` | 200/24° | **4.5** | 130 | 50 | 18 | 120 / 95 |
| OPTIMA II 400 | `AGFA_OPTIMA_400` | 400/27° | 4.5 | 130 | 50 | 19 | 120 / 95 |
| PORTRAIT XPS 160 | `AGFA_PORTRAIT_160` | 160/23° | 3.5 | 150 | 60 | 18 | 120 / 95 |
| **ULTRA 50** | *(none)* | 50/18° | 4.3 | 140 | 50 | 27 | 120 / 95 |
| **RSX II 50** | *(none)* | 50/18° | 10.0 | 125 | 55 | 25 | 120 / 95 |
| **RSX II 100** | *(none)* | 100/21° | 10.0 | 125 | 50 | 25 | 120 / 95, sheet = Acetate 190 µm |
| **RSX II 200** | *(none)* | 200/24° | 12.0 | 110 | 50 | 27 | 120 / 95 |
| SCALA 200x | `AGFA_SCALA_200X` | 200/24° | 11.0 | 120 | 50 | 7 | 120 / 95, sheet = PET 175 µm |
| APX 25 | `AGFA_APX_25` | 25/15° | 7.0 | 200 | — | 3 | 120 / 95 |
| APX 100 | `AGFA_APX_100` | 100/21° | 9.0 | 150 | — | 7 | 120 / 95 |
| APX 400 | `AGFA_APX_400` | 400/27° | 14.0 | 110 | — | 10 | 120 / 95 |

### 2.1 Measurement conditions, printed on p5 and repeated verbatim in the 2004 edition

- **Granularity** — exposure daylight; densitometry **visual filter (Vλ)**; measurement **diffuse
  density 1.0, 48 µm reading aperture**.
  ⚠ **THIS MATCHES THIS PROJECT'S OWN CONVENTION ALMOST EXACTLY** (48 µm aperture, D = 1.0), which
  is unusual and worth stating: most sources in the corpus do not print the aperture at all. The
  one gap is that Agfa say *diffuse density 1.0* and the project convention says *NET density 1.0*;
  the sheet does not say whether base+fog is subtracted.
- **Resolving power** — lines per mm at contrast range **1.6 : 1 or 1000 : 1**. Called "a purely
  visual criterion" by the sheet itself.
- **Sharpness** — an MTF chart; exposure daylight, densitometry visual filter (Vλ).
- **Spectral sensitivity** — equal-energy spectrum, reading density **1.0 above minimum density**.
- **Absorption of the emulsion dyes** — neutral subject of medium brightness; minimum density.
- **Colour density curves** — exposure **daylight 1/100 s**; process **AP 70/C-41** and **AP 44/E-6**;
  densitometry **Status A and Status M**.

### 2.2 Cross-edition comparison, 1998 vs 2004

| film | RMS 1998 → 2004 | RP 1000:1 1998 → 2004 | layer µm |
|---|---|---|---|
| Portrait 160 | 3.5 → 3.5 | 150 → 150 | 18 → 18 |
| Optima 100 | 4.0 → 4.0 | 140 → 140 | 16 → 16 |
| Optima 200 | **4.5 → 4.3** | 130 → 130 | 18 → 18 |
| Optima 400 | 4.5 → 4.5 | 130 → 130 | 19 → 19 |
| RSX II 50 | 10.0 → 10.0 | **125 → 135** | 25 → 25 |
| RSX II 100 | 10.0 → 10.0 | **125 → 130** | 25 → 25 |
| RSX II 200 | 12.0 → 12.0 | **110 → 120** | 27 → 27 |
| Scala 200x | 11.0 → 11.0 | 120 → 120 | 7 → 7 |
| APX 100 | 9.0 → 9.0 | 150 → 150 | 7 → 7 |
| APX 400 | 14.0 → 14.0 | 110 → 110 | 10 → 10 |
| ULTRA 50, APX 25 | *1998 only* | | |

Ten of twelve rows are identical across five years. The four that move all move in the direction of
a product improvement, so they read as genuine revisions rather than typographical drift. **The
database's `AGFA_OPTIMA_200` RMS 4.3 is the 2004 figure and its source string says so — it is
correct, not stale**, but the profile's `era` of "1990s-2000s" silently means the 2004 emulsion.

⚠ **A SEPARATE ISSUE ON `AGFA_OPTIMA_200`:** its stored RMS 4.3 is also exactly `AGFA_VISTA_200`'s
stored 4.3. Two Agfa 200-speed negatives with the same figure is not by itself suspicious — both
are sourced and cited — but it is the shape `NotFound.md` warns about, and the 1998 edition
disambiguates it: Optima II 200 was **4.5** before the revision, Vista 200 has always been 4.3.

### 2.3 Reciprocity, all twelve films (p6) — direct extraction

| film | reading (s) | exposure adjustment (f-stops) | development / filtration |
|---|---|---|---|
| OPTIMA II 100 | 1/10 000–1, 10, 100 | 0, +½, +1½ | — |
| OPTIMA II 200 | 1/10 000–1, 10, 100 | 0, +1, +2 | — |
| OPTIMA II 400 | 1/10 000–1, 10, 100 | 0, +1, +2 | — |
| PORTRAIT XPS 160 | 1/10 000–1, 10, 100 | 0, +1, +2 | — |
| ULTRA 50 | 1/10 000–1, 10, 100 | 0, +1, +2 | — |
| RSX II 50 | 1/10 000–1, 10, 100 | 0, +½, +1 | CC 0, 05B, 10B |
| RSX II 100 | 1/10 000–1, 10, 100 | 0, +½, +1 | CC 0, 05B, 10B |
| RSX II 200 | 1/10 000–1, 10, 100 | 0, +1, +2 | CC 0, 075Y, 15Y+05C |
| APX 25 | 1/10 000–½, 1, 10, 100 | 0, +½, +1, +2 | dev 0, 0, 0, 0 |
| APX 100 | 1/10 000–½, 1, 10, 100 | 0, +1, +2, +3 | dev 0, −10, −25, −35 % |
| APX 400 | 1/10 000–½, 1, 10, 100 | 0, +1, +2, +3 | dev 0, −10, −25, −35 % |
| SCALA 200x | 1/10 000–½, 1, 10, 100 | 0, +½, +1, +2 | — |

⚠ **APX 25 DISAGREES WITH ITS OWN 1995 DATASHEET.** `agfapanapx25.pdf` (08/1995) prints
**none, +1, +1½, +2** for the same four times; this 1998 range brochure prints **0, +½, +1, +2**.
Same manufacturer, same film, three years apart, and the middle two entries differ by half a stop
and a full stop. Recorded, not resolved. The 2004 B&W manual confirms the 1998 APX 100/400 values
(+1 / +2 / +3 with −10 / −25 / −35 %), so only APX 25 is in dispute and only the 1995 sheet
dissents.

### 2.4 Push/pull, RSX II 200 (p9) and SCALA 200x (p9)

**RSX II 200** — Push 1 / 2 / 3 → ISO 400 / 800 / 1600, Pull 1 → ISO 100. Contrast "increasingly
steeper" on push and "flatter" on pull; maximum density "decreasing" on push, "increasing" on pull;
granularity "increasingly coarse-grained" on push, "finer" on pull. **No numbers** — the direction
words are all Agfa print.

**SCALA 200x** — the density-curve panel plots all five steps as measured curves, so for Scala the
same statement exists numerically. See §3.4.

---

## 3. Digitised curves — `agfa_1998_curves.py`

All twelve columns read. Every value below is **curve digitisation**, not text, and is tagged as
such. Axis calibration is fitted from each panel's own printed labels with iterative outlier
rejection; the reported residual is the worst surviving label.

### 3.1 Reader notes — three defects the sheet forced, each of which produced plausible output

1. **The stroked frame is the wrong containment reference.** The p10 characteristic-curve panel's
   only frame rect spans x 64.3–179.7 while the curve is drawn from x 54.4 — the rect is an inner
   grid box. Frame-based containment returned "no curve" on six of twelve columns. The panel is now
   derived from the axis labels, which is what the calibration is fitted to anyway.
2. **The co-linearity cluster chains across axes.** The ordinate reads right-aligned at x 47.2 and
   the abscissa's "−4.0" is centred at 52.4 — 5.2 pt away, outside tolerance — but the ordinate's
   own single-glyph "0" sits at 49.6 and bridges them in two hops. The abscissa label joined the
   ordinate and the fit returned **2.22 D of residual on a 3 D axis**. It did not crash; it traced
   the curve and returned wrong densities. Now rejected iteratively (residual 0.0001 D).
3. **Stroke width does not separate data from furniture on this sheet.** Curves are 0.789 pt on p7
   and p10 but **0.503 pt** on the Portrait spectral panel — *thinner than one of that panel's own
   frames*. Shape does separate them: Agfa draw data as beziers or long non-axis-aligned polylines
   and furniture as `re`/straight `l`.

### 3.2 Sharpness — f50 and overshoot, all twelve

⚠ **THIS IS A CTF, NOT AN MTF.** Every curve exceeds 100 % at low frequency (peaks 102–114 %),
so it is an adjacency-enhanced response. f50 is still a well-defined reading and is what
`MTFSpec.f50` means; the overshoot is direct measured evidence for `CouplerSpec` adjacency.

| film | f50 (sheet units) | peak % | adjacency = peak−1 | RP 1000:1 | f50 / RP |
|---|---|---|---|---|---|
| APX 25 | 78.2 | 105 | 0.05 | 200 | 0.39 |
| APX 100 | 57.6 | 110 | 0.10 | 150 | 0.38 |
| APX 400 | 57.6 | 110 | 0.10 | 110 | 0.52 |
| Optima 100 | 43.7 | 111 | 0.11 | 140 | 0.31 |
| Optima 200 | 46.9 | 109 | 0.09 | 130 | 0.36 |
| Optima 400 | 47.4 | 106 | 0.06 | 130 | 0.36 |
| Portrait 160 | 36.0 | 106 | 0.06 | 150 | 0.24 |
| ULTRA 50 | 42.6 | 114 | 0.14 | 140 | 0.30 |
| RSX II 50 | 29.2 | 104 | 0.04 | 125 | 0.23 |
| RSX II 100 | 31.8 | 108 | 0.08 | 125 | 0.25 |
| RSX II 200 | 21.2 | 110 | 0.10 | 110 | 0.19 |
| SCALA 200x | 30.5 | 102 | 0.02 | 120 | 0.25 |

⚠ **APX 100 AND APX 400 RETURN IDENTICAL SHARPNESS CURVES** — same point count, same endpoints,
same f50 to 0.1. They are two separate path objects in two separate columns with identical
geometry, i.e. **Agfa reused one piece of artwork for both films**. That is a finding about the
source, not a tracing failure, and it means the APX 400 sharpness panel carries no information of
its own. The 2004 edition should be checked before either is adopted.

#### G6 — what this does and does not settle

G6 asks for one Agfa MTF sheet whose "lines/mm" axis can be cross-checked against a resolving
power. **These sheets print both, for the same film, on the same page** — and they are not the four
files G6 re-proved absent. Tani's relation (MTF-50 ≈ ½ × resolving power,
`EMULSION_KNOWLEDGE_BASE.md` 18) predicts f50/RP = 0.5. Measured: **0.19 – 0.52, median 0.30**.

Reading the frequency axis as *half*-cycles would halve every ratio to 0.10–0.26, i.e. **further**
from the relation, not closer. So the evidence **favours "lines/mm" = cycles/mm on both axes, with
no factor-of-2 correction**. ⚠ It does not prove it: the spread is 2.7× and Tani's relation is
itself approximate. **G6 is narrowed, not closed**, and the honest next step is the 2004 edition's
own APX and RSX panels as a second sample.

### 3.3 APX characteristic curves — and a cross-confirmed anomaly

| film | source | D-min | D-max | lg E span | residual |
|---|---|---|---|---|---|
| APX 25 | 1998 p10 | **0.286** | 2.563 | −3.91 … +2.04 | 0.0001 D |
| APX 100 | 1998 p10 | **0.275** | 2.565 | −3.92 … +2.04 | 0.0000 D |
| APX 100 | 2004 manual p10 | **0.261** | 2.671 | −3.96 … +2.06 | 0.0440 D |
| APX 400 | 1998 p10 | **0.129** | 2.825 | −3.90 … +2.05 | 0.0001 D |
| APX 400 | 2004 manual p11 | **0.107** | 2.950 | −3.96 … +2.04 | 0.0440 D |

Stored today: APX 25 **0.10**, APX 100 **0.11**, APX 400 **0.13**.

⚠ **THE MEASURED D-MIN FALLS WITH SPEED AND THAT IS PHYSICALLY BACKWARDS.** A faster, thicker
emulsion (APX 400 is 10 µm against APX 25's 3 µm) should fog *more*, not less. Two independent
artworks five years apart, in different inks at different scales, agree on the inversion, so it is
**not a tracing artefact — it is what Agfa drew**. The likeliest explanations are that the APX 400
panel plots density above base while APX 25/100 plot total density, or that the panels simply do
not share a zero; the sheets state neither. **The stored 0.10 / 0.11 / 0.13 triple is ascending and
physically sensible but matches no reading except APX 400's, and looks like a family template.**

**Recommendation: do not adopt any APX D-min from these curves.** Record the readings and the
inconsistency. The clean way to settle it is queue **D1** — one empty-gate frame makes density
absolute — or a fourth Agfa source.

D-max is a different matter: 2.56 / 2.57 / 2.83 (1998) against 2.67 / 2.95 (2004 manual) is a
consistent, plausible, adoptable-looking set with no ordering problem. There is no `dmax` field on
`ToneCurve`, so it has no carrier today.

### 3.4 SCALA 200x push/pull density curves — five measured steps

Digitised from `agfa_films.pdf` p9. Step names assigned by optimal label-to-curve assignment; the
result is confirmed independently by the sheet's own push/pull table, which states that maximum
density *increases* on pull and *decreases* on push.

| step | D-max, 1998 | D-max, 2004 manual | ISO |
|---|---|---|---|
| Pull 1 | 3.064 | 3.012 | 100/21° |
| Standard | 2.983 | 2.800 | 200/24° |
| Push 1 | 2.740 | 2.543 | 400/27° |
| Push 2 | 2.456 | 2.289 | 800/30° |
| Push 3 | 2.172 | 2.034 | 1600/33° |

All five share lg E −1.94 … +3.04 and D-min 0.024 (1998) / 0.033 (2004). ⚠ The two editions
**disagree on Standard by 0.18 D** and agree on Pull 1 to 0.05 D; the ordering and the spacing
(≈0.27 D per stop) are identical. `PushSpec` has `max_push_stops`, `max_pull_stops` and
`gamma_gain_per_stop` and `AGFA_SCALA_200X` currently has all of them at zero.

### 3.5 Gamma-time — three APX films × five developers, and a specification confirmed

The panel draws **four curves for five printed names**: RODINAL SPECIAL and STUDIONAL LIQUID share
one curve, which is consistent with the p11 processing table giving both the same time at every
temperature.

γ read at each developer's own published reference time from the p11 table:

| film | REFINAL | RODINAL 1+25 | RODINAL 1+50 | RODINAL SPECIAL / STUDIONAL |
|---|---|---|---|---|
| APX 25 | γ(6 min) = **0.652** | γ(6 min) = **0.646** | γ(10 min) = **0.651** | γ(4 min) = **0.654** |
| APX 100 | γ(6 min) = **0.653** | γ(8 min) = **0.652** | γ(17 min) ≈ 0.66 | γ(4 min) = **0.655** |
| APX 400 | γ(6 min) = **0.653** | γ(7 min) ≈ **0.647** | γ(11 min) ≈ **0.644** | γ(4½ min) ≈ **0.652** |

**Eleven independent readings land on γ = 0.65 ± 0.01.** This is not a coincidence and it is not an
inference: `agfa_bw_manual.pdf` states it in words — every speed table in the developer section is
headed *"Film speed (exposure index) (γ = 0.65)"*, and the developing-time tables are indexed by
γ = 0.55 / 0.65 / 0.75. **Agfa specify the whole AGFAPAN line to γ = 0.65**, and the traced curves
reproduce that to one part in sixty-five. It is the strongest self-consistency check in this
harvest and it validates the whole gamma-time digitisation.

Stored today: APX 25 γ 0.64, APX 100 γ 0.62, APX 400 γ 0.66. ⚠ Those are `ToneCurve` softplus model
coefficients and **not** the same quantity as Agfa's γ — `ToneCurve.mid_slope` is the comparable
number. They must not be overwritten with 0.65 without that conversion.

Full γ(t) samples, `agfa_films.pdf` p10, residual 0.001 min / 0.0000 γ:

| film | developer | t range (min) | γ(4) | γ(6) | γ(8) | γ(10) | γ(12) |
|---|---|---|---|---|---|---|---|
| APX 25 | SPECIAL / STUDIONAL | 1.9 – 7.0 | 0.654 | 0.723 | — | — | — |
| APX 25 | REFINAL | 2.9 – 9.9 | 0.589 | 0.652 | 0.705 | — | — |
| APX 25 | RODINAL 1+25 | 2.9 – 10.9 | 0.587 | 0.646 | 0.692 | 0.732 | — |
| APX 25 | RODINAL 1+50 | 5.9 – 16.9 | — | 0.553 | 0.606 | 0.651 | 0.687 |
| APX 100 | SPECIAL / STUDIONAL | 2.9 – 5.9 | 0.655 | — | — | — | — |
| APX 100 | REFINAL | 2.9 – 9.8 | 0.589 | 0.653 | 0.706 | — | — |
| APX 100 | RODINAL 1+25 | 4.2 – 12.3 | — | 0.599 | 0.652 | 0.700 | 0.744 |
| APX 100 | RODINAL 1+50 | 12.3 – 17.9 | — | — | — | — | — |
| APX 400 | SPECIAL / STUDIONAL | 2.8 – 7.0 | 0.631 | 0.717 | — | — | — |
| APX 400 | REFINAL | 3.3 – 9.5 | 0.582 | 0.653 | 0.710 | — | — |
| APX 400 | RODINAL 1+25 | 4.0 – 11.8 | 0.552 | 0.622 | 0.674 | 0.715 | — |
| APX 400 | RODINAL 1+50 | 5.9 – 17.9 | — | 0.552 | 0.591 | 0.627 | 0.661 |

⚠ **THE WEAKEST LINK IN THE ASSIGNMENT** is REFINAL vs RODINAL 1+25 on APX 25: the two labels are
0.0 and 0.2 pt from their assigned curves, essentially a tie. Swapping them changes γ(6) from 0.652
to 0.646 — 0.006 — so the risk is bounded and small, but it is real and is recorded here rather
than smoothed.

### 3.6 Colour density curves — D-min per channel, and a cross-edition validation

Digitised from `agfa_films.pdf` pp7–8. Compared against `agfa_2004_curves.py`'s independent read of
the 2004 edition:

| film | 1998 r / g / b D-min | 2004 r / g / b D-min | worst Δ |
|---|---|---|---|
| Optima 100 | 0.31 / 0.66 / 0.91 | 0.265 / 0.676 / 0.918 | 0.045 |
| Optima 200 | 0.32 / 0.69 / 0.89 | 0.250 / 0.644 / 0.815 | 0.075 |
| Optima 400 | 0.34 / 0.70 / 1.05 | 0.406 / 0.796 / 1.073 | 0.096 |
| Portrait 160 | 0.23 / 0.56 / 0.72 | 0.232 / 0.574 / 0.724 | 0.014 |
| ULTRA 50 | 0.42 / 0.75 / 0.95 | *(not in the 2004 edition)* | — |

Two readers, two documents, five years apart, agreeing to 0.014–0.096 D. Portrait 160 at 0.014 is
essentially exact. **The r < g < b orange-mask ordering holds on all five.**

RSX II reversal D-min, 1998 only: RSX II 50 **0.16 / 0.15 / 0.14**, RSX II 100 **0.14 / 0.15 /
0.14**, RSX II 200 **0.14 / 0.12 / 0.11** — neutral, as a reversal film must be.

### 3.7 Spectral density — the one panel that is genuinely three dyes

⚠ **ON THE FIVE COLOUR NEGATIVES IT IS NOT.** The panel plots *"Medium density"* and *"Minimum
density"* — two aggregate curves, i.e. the schema-v14 **neutral + D-min pair**, not
`d_cyan/d_magenta/d_yellow`. `NotFound.md` already warns that the dye-density count of 12 must not
be "corrected" upward by conflating the two, and this does not correct it.

✔ **ON THE THREE RSX II REVERSAL FILMS IT IS.** Those panels print **Yellow / Magenta / Cyan /
Visual grey** — four labelled curves, three of them separated dyes plus a visual-grey reference.
This is a real `SpectralDyeDensity` source and the **first Agfa one in the corpus**. Currently
captured through the dash key as b/g/r over 380–721 nm; the yellow/magenta/cyan relabelling is a
one-line change and has not been made pending the decision on whether to create the profiles at all.

### 3.8 Spectral sensitivity — cross-source check against what is already stored

The APX spectral sets in the database came from the individual 1995 datasheets. Re-read from the
1998 range brochure, normalised to peak, over the samples above −3.9:

| profile | overlap | rms | max │Δ│ |
|---|---|---|---|
| `AGFA_APX_25` | 28 pts | 0.178 lg | 0.285 |
| `AGFA_APX_100` | 29 pts | 0.046 lg | 0.202 |
| `AGFA_APX_400` | 28 pts | 0.114 lg | 0.229 |

Two different publications plotting the same emulsion. APX 100 agrees closely; APX 25 and 400 differ
more. **No change recommended** — the 1995 individual datasheets are the more specific source and
should stay. This is recorded as a cross-check, and as a bound on how much a "measured" Agfa
spectral curve is worth: about ±0.1–0.2 lg between the maker's own two publications.

---

## 4. `agfa_bw_manual.pdf` — the three-way split the request asked for

### 4.1 Belongs in individual film profiles

- **Confirmation of the tabulated triple** for APX 100 / APX 400 / Scala 200x (ISO, RMS with the
  *"Refinal 6 min, 20 °C"* condition, resolving power) — identical to both film sheets.
- **Reciprocity** for the same three — identical to the 1998 sheet, and therefore a second vote
  against the 1995 APX 25 sheet in the §2.3 dispute.
- **Independent density curves** for APX 100, APX 400 and all five Scala steps — §3.3, §3.4.
- **Developer-dependent effective speed at γ = 0.65**, a real `ProcessVariant` carrier:

| developer | APX 100 | APX 400 |
|---|---|---|
| RODINAL 1+25 | ISO 125/22° @ 18 min | ISO 320/26° @ 15 min |
| RODINAL 1+50 | ISO 160/23° @ 17 min | ISO 400/27° @ 30 min |
| RODINAL SPECIAL 1+15 | ISO 100/21° @ 4 min | ISO 320/26° @ 6 min |
| STUDIONAL LIQUID 1+15 | ISO 100/21° @ 4 min | ISO 320/26° @ 6 min |
| REFINAL | ISO 160/23° @ 6 min | ISO 500/28° @ 5 min |

⚠ **THESE TIMES CONTRADICT THE 1998 SHEET AND NOT BY A LITTLE.** 1998 gives APX 400 in RODINAL 1+25
as **7 min**; the 2004 manual gives **15 min**. RODINAL 1+50 goes 11 min → 30 min. The speeds also
move (APX 100 in REFINAL: ISO 125 → ISO 160). Six years apart with the same product names. Both are
Agfa, both say "small tank or tray at 20 °C", and there is no stated reason. **Recorded as an open
conflict; neither should be adopted as *the* processing time without deciding which edition the
profile represents.**

- **Development time vs temperature and γ**, the `ProcessingFamily` carrier in tabular form:

| developer | film | drum γ 0.55 | drum γ 0.65 | drum γ 0.75 | tank γ 0.65 |
|---|---|---|---|---|---|
| RODINAL 1+25 | APX 100 | 10.4 | 17 | 10 | 18 |
| RODINAL 1+50 | APX 100 | 10.8 | 14 | 19 | 17 |
| RODINAL 1+25 | APX 400 | 10.6 | 11.5 | 24 | 15 |
| RODINAL 1+50 | APX 400 | 10.5 | 15 | — | 30 |
| RODINAL SPECIAL | APX 100 | — | 3.5 | 4 | 4 |
| RODINAL SPECIAL | APX 400 | 3 | 4 | 6 | 6 |
| REFINAL | APX 100 | 3 | 5 | 8 | 6 |
| REFINAL | APX 400 | 3.5 | 4.5 | 6.5 | 5 |

⚠ The RODINAL γ 0.55 column is **non-monotone against γ 0.65** on all four rows (10.4 → 17,
10.6 → 11.5) and the values are suspiciously clustered at 10.4–10.8. This looks like a typesetting
fault in the manual — most likely "0.4" printed as "10.4". **Not adoptable as printed.**

### 4.2 General sensitometric knowledge for the algorithm, no per-film carrier

- **γ = 0.65 is Agfa's line-wide aim for AGFAPAN**, stated in terms and reproduced by the traced
  curves to ±0.01 (§3.5). This is a *specification*, and it is the thing that makes every one of
  Agfa's developer tables comparable.
- **Agfa's own definition of "sharpness"** (manual p10): both resolution and MTF are given, and
  Agfa explain the MTF as the measured contrast loss with decreasing line spacing due to **light
  diffusion within the emulsion layer**. Useful for the C2c/C19 discussion of what `adjacency`
  physically is.
- **Agfa measure competitors' films.** The developer section prints effective speed at γ = 0.65 for
  17 film types across four developers, including **Fuji Neopan 400/1600, Ilford PAN-F Plus, FP4
  Plus, HP5 Plus, Delta 100/400/3200, SFX 200, Kodak Plus-X, Tri-X, T-MAX 100/400/P3200 and
  Recording 2475** — many of which have profiles in this database. ⚠ **This is one manufacturer
  measuring another's product in its own chemistry.** It is a legitimate `ProcessVariant` source and
  an illegitimate source for a film's nominal speed. Listed here, not adopted, and worth a decision
  of its own.
- **Flash guide-number formula**, storage and acclimatisation times (2 h refrigerated, 8 h frozen),
  X-ray guidance — general, no carrier.

### 4.3 Paper / print-chain only — no film carrier, useful for a future print simulation

- **MULTICONTRAST PREMIUM (MCP)** and **MULTICONTRAST CLASSIC (MCC)**: density curves for contrast
  filters **0, 1, 2, 3, 4, 5**, exposed tungsten 3000 K for 10 s, developed in AGFA MULTICONTRAST
  DEVELOPER, read with visual filter (Vλ).
- **Maximum blacks**: MCP 310 RC **2.25**, MCP 312 RC **2.25**, MCP under laser exposure **2.20**,
  MCC 111 **2.30**, MCC 118 **1.60**.
- **Speed**: both papers **ISO P 400** unfiltered (≈ grade 2); **ISO P 160** for filters 0–3½;
  **ISO P 80** for filters 4–5. Exposure doubles at filter 4 and above.
- **Spectral sensitivity** of both papers at reflection densities 0.5 / 1.0 / 1.5, referred to an
  equal-energy spectrum.
- **Reciprocity** of both papers over 0.1–100 s, plotted as speed and as contrast range, with
  Δ lg ER = 0.2 gridding; Agfa state the effect is nearly independent of filtration and that
  contrast stays almost constant.
- **Construction**: RC/PE polyethylene both sides, emulsion direct on the plastic with no intercoat;
  fibre base carries **20–45 g/m² baryta**; silver coating **≈1.5 g/m²**.

⚠ `PrintStock` gained a `spectral` field at schema v22 and the database holds 11 print stocks. The
MCP/MCC data is the most complete paper set in the corpus and **none of it is in the database** —
but building it is a print-chain project, not part of this harvest.

---

## 5. What remains unavailable after both documents

| parameter | status |
|---|---|
| σ(D) granularity-vs-density shape | absent from both. A single RMS figure at D 1.0 is not a shape. Still queue **F2b**. |
| Callier coefficient | absent. Still assumed 1.3 on every Agfa profile. |
| Crystal size, habit, aspect ratio, iodide | absent. `EmulsionSpec` gets only `coated_um`. |
| Absolute base+fog | absent, and §3.3 shows the plotted D-min cannot substitute. Still queue **D1**. |
| Separated dye densities for the colour NEGATIVES | absent by construction — the panel is a two-curve neutral + D-min pair. |
| Layer-by-layer resolving power | absent. `LayerStack` stays empty on every Agfa profile. |
| An unambiguous statement of the MTF frequency unit | absent. §3.2 narrows G6 but does not close it. |
| Which processing edition a profile represents | undecided, and §4.1 shows it changes the times by 2×. |

---

## 6. Proposed database changes — NOT APPLIED, awaiting approval

### 6.1 Safe: direct text extraction, no judgement

| # | change | films |
|---|---|---|
| A | `mtf.resolving_power_lp_mm_lowc` ← 50 / 50 / 50 / 60 | Optima 100, 200, 400, Portrait 160 — currently already set; **verify only** |
| B | `emulsion.coated_um` ← 16 / 18 / 19 / 18 / 7 / 7 / 10 / 3 µm | Optima 100/200/400, Portrait 160, Scala, APX 100, APX 400, APX 25 — all currently **0.0** |
| C | `reciprocity_table` ← the §2.3 rows, per film | all 8 existing Agfa profiles; currently **empty**, only a fitted Schwarzschild p exists |
| D | rewrite the `grain.rms_granularity` ParamSource on APX 25/100/400 | the note *"No published rms for this stock in the corpus"* is **false** — the vendor prints it with developer, time and temperature, on a sheet already named in `provenance.sources`. Values unchanged. |
| E | `processing` ← developer/dilution/time/temperature | APX 25/100/400 — currently entirely empty; ⚠ blocked on the §4.1 edition conflict |

### 6.2 Needs a decision

| # | question |
|---|---|
| F | **Create `AGFA_ULTRA_50` and `AGFA_RSX_II_50/100/200`?** Four new profiles, fully documented (ISO, RMS, RP at both contrasts, layer thickness, base, reciprocity, spectral sensitivity ×3, density curves ×3, sharpness), and the RSX trio brings the **first Agfa separated dye-density set**. New profiles renumber `film_enum.hpp`, so this is a scoped change. |
| G | **Adopt f50 from the CTF?** §3.2. The curve is adjacency-enhanced, the unit question is narrowed but open, and APX 400's panel is a duplicate of APX 100's. |
| H | **Adopt `adjacency` = peak − 1?** §3.2 gives twelve measured overshoots, 0.02–0.14. This is direct evidence for C2c/C19 and is unit-free, so the G6 question does not block it. |
| I | **Adopt the APX D-min?** §3.3. Recommendation: **no** — cross-confirmed but physically inverted. |
| J | **Adopt `PushSpec` for `AGFA_SCALA_200X`?** §3.4 gives ±3 push, 1 pull, and measured D-max per step. |
| K | **Adopt `ProcessingFamily` γ(t) for the three APX?** §3.5, eleven-point validated at γ = 0.65. |
| L | **`AGFA_OPTIMA_200` era.** Its stored 4.3 is the 2004 figure; 1998 said 4.5. Should the profile state which emulsion it is? |
| M | **Should Agfa's measurements of Fuji/Ilford/Kodak films (§4.2) be used at all?** |

### 6.3 Infrastructure

| # | change |
|---|---|
| N | register `agfa_1998_curves.py` in `build.py` (audits 28 → 29) |
| O | add `EXPECTED` pins to it so a change in the source or the reader fails the build, as `agfa_2004_curves.py` does |
| P | correct the false "same publication" claim in `NotFound.md` row 5 and queue **G6**, and the false "RASTER page" claim in `agfa_2004_curves.py`'s docstring |
| Q | copy `agfa_films.pdf` and `agfa_bw_manual.pdf` into the checkout corpus, or accept two more SKIPping audits |
