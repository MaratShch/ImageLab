# H-740 workbook + the RETRO shelf — what is actually usable

**Date 2026-08-18.** Review only: no source file was modified, no profile value was changed.
`PDF/PROFILES/Index.md` was regenerated in place to include the five new documents.

---

## 0. Headline, before the detail

Three of the five documents are worth acting on, and **the single most valuable one is a
document we already owned the text of.** Mees 1942 turns out to be a *re-acquisition* — its
extracted text is byte-identical to `PDF/MEES_1942_fulltext.txt` (md5 `9e215589…`), and
`doc/MEES_1942_EXTRACTION.md` already mined it. What the returned PDF adds is the **page
images**, and those close three items the old extraction had to leave open, because OCR
destroys exactly the formulae that matter.

| Document | Verdict |
|---|---|
| **Basic Photographic Sensitometry Workbook** (Kodak H-740, 2006) | **Act on it.** Official, implementable definitions of gamma / CI / average gradient / ISO speed. Clean vector text, no OCR risk |
| **The Theory of the Photographic Process** (Mees, 1942) | **Act on the images.** Text already mined; images close 3 open questions |
| **Sensitometry of Photographic Emulsions** (Davis & Walters, NBS Sci. Pap. 439) | **One usable number**, and a hard blocker on everything else — brands are anonymised |
| **The Photographic Emulsion** (Carroll, Hubbard & Kretschman) | **Low value.** Emulsion *making* chemistry, not image-structure physics |
| **Ilford Book of Formulae, 3rd ed.** | **Provenance only.** Developer compositions; OCR corrupts every numeral |

---

## 1. Basic Photographic Sensitometry Workbook — KODAK H-740, November 2006

24 pages, fully vector, clean text layer, 7 vector plots, no raster. An official Kodak
publication and a *methods* document — which is what makes it valuable: it does not add data
about a stock, it tells us how the quantities we already store are defined by the people who
published them.

### 1.1 The ISO/ANSI speed method, and the part of it we can use today (pp 10–11)

> Point **A** = the point 0.10 density above D-min. Point **B** = 1.30 log H units to the
> right of A. *"The film is properly developed if the density at this point is 0.80 (± 0.05)
> more than the density at point A."* Then
> **speed = 0.8 / antilog(log H at A)** with H in lux-seconds
> (**800 / antilog(A)** with H in millilux-seconds).
> Round to the standard table; ⅓ step = 0.1 log E (p 11).

The absolute speed formula needs H in physical lux-seconds, which our relative-x curves do
not carry — so we cannot compute an ISO speed. **But the development condition is
scale-invariant**: ΔD over 1.30 log H from the (D-min + 0.10) point depends only on curve
*shape*. That makes it computable on every stored curve right now.

**I ran it over the 53 B&W negatives in the database:**

| | |
|---|---|
| Inside the ISO condition ΔD = 0.80 ± 0.05 | **10 of 53** |
| Typical value elsewhere | ΔD ≈ 0.58–0.64, i.e. average gradient **0.45–0.49** |
| Highest | FERRANIA_P30 ΔD 1.251 (G 0.96), TASMA_FN_64 1.047, FUJI_NEOPAN_1600 0.992 |
| Lowest | KODAK_RECORDING_2475 0.507 (G 0.39), KODAK_ROYAL_PAN_4141 0.531, KODAK_ROYAL_X_PAN_4166 0.561 |
| Closest to ISO-normal | KODAK_TMAX_100 0.794, EASTMAN_DOUBLE_X_5222 0.785, GEVAERT_PANCHRO_1950 0.784, FUJI_NEOPAN_ACROS_100 0.826 |

⚠ **This is a diagnostic, not a defect list, and it must not be turned into a pass/fail
assertion.** A published curve is drawn at the manufacturer's *recommended pictorial*
development, which is often deliberately below the ISO speed-determining contrast; a
high-speed stock at G ≈ 0.40 and Technical Pan at G ≈ 0.45 are both entirely plausible.

What it *does* say, and this is a real modelling issue: **an ISO speed is only defined at
ISO-normal contrast.** Where a stock stores `exposure_index` as its ISO rating but its curve
sits at G ≈ 0.47, the two fields are describing two different development states. That is
worth recording per stock as a derived **[T3]** figure — "curve gradient vs the ISO condition"
— so the mismatch is visible rather than silent. Cheap to add to `FilmActiveProfiles.md`.

### 1.2 Definitions we should align our fitted statistics to

| Quantity | H-740 definition | Our status |
|---|---|---|
| **Gamma** (p 9) | slope of the **straight-line portion only** | matches |
| **Contrast Index** (p 9) | marked-straightedge construction: marks at 0.0 / 0.2 / 2.2; the 0.0 mark on the D-min horizontal, the 0.2 and 2.2 marks both touching the curve; CI = slope of that edge. Explicitly *"unlike gamma"* because **the toe shape influences CI** | we store CI but do not construct it this way |
| **Average Gradient G** (p 10) | slope between any two points, written with the two densities as subscripts (G₀.₁₈₋₁.₇₀) | matches method rule 8's practice |
| ISO contrast band (p 12) | average gradient **0.58–0.65** for "properly developed" | new, usable |

⚠ **One inconsistency in the source, flagged rather than adopted.** The worked CI answer on
p 22 gives A(0.68, 0.18) and B(2.00, 0.98) → CI 0.61, but that chord is 1.54 units long, not
the 2.0 the construction specifies. The prose construction and the answer key disagree. Use
the construction; treat the numeric answer as unreliable, and take the authoritative CI
definition from ANSI/ISO rather than from this workbook if it ever matters to a stored value.

### 1.3 Directly reusable specifications

* **Step tablets (p 6).** 11-step: increment **0.30**; 21-step: increment **0.15**; both span
  **0.05 → 3.05**. This is an exact, official reference wedge — it should be what
  `make_test_chart.py` emits for the comparison loop, instead of an arbitrary ramp.
* **Contrast Index depends on four variables (pp 2, 12): time, temperature, agitation,
  developer activity.** This independently confirms the variable set of the H-24 Module 14
  process-variation figures identified in the roadmap — two Kodak publications, same axes.
* **Time–Contrast Index curve (p 12)** — CI vs development time for a fixed
  developer/temperature/agitation. That is precisely the representation proposed for
  `ProcessSensitivity`, now with a vendor name for it.
* **Slope bands (p 15).** Negatives typically **0.45–0.65**; papers **1.5–3.5**; grade 2 ≈ 2.0.
  A sanity band for our print stocks.
* **Latitude (pp 13–14)** as a construction: place the scene log-range at the speed point and
  count 0.3-log steps to each end of the usable curve. Our `trim`/latitude figures could be
  *computed* this way rather than estimated.
* **Colour negative (p 14):** blue filter reads yellow dye, green reads magenta, red reads
  cyan; the three curves are separated **by the orange mask** and *"if the orange mask wasn't
  there, the blue and green curves would be lying on the red curve"*. Confirms our
  `dmin_ladder` reading of the mask and the layer-order convention.

---

## 2. Mees 1942 — what the page images gave that the OCR could not

### 2.1 The granularity/aperture law — book p 869 (PDF p 866). **Closes an open item.**

`MEES_1942_EXTRACTION.md` recorded: *"the aperture-scaling law was not found in the pages
sampled"*. It is there, and OCR mangled it into unreadable fragments. Read from the page
image, verbatim:

> Let the area of the scanning beam be *a*, the grain area be *a′*, and the mean transparency
> and density be *T* and *D*. Then, if the image is confined to the surface,
> **ΔT = 0.675 · T · √(a′/a) · √((1 − T)/T)**  (14)
> If the image extends into the emulsion,
> **ΔT = 1.022 · √(a′/a) · √D**  (15)

Two things follow, and they are the actionable part:

1. **Granularity ∝ √(a′/a).** It falls as the square root of the measuring aperture area and
   rises as the square root of the *grain* area — i.e. **linearly with grain diameter**. Our
   `rms_granularity` is defined at a 48 µm aperture and `AlgoGrain` re-derives the amplitude
   for the render grid by integrating the continuous spectrum. That integral now has a
   published law to be checked against, and the check is free.
2. **`rms_granularity` and `clump_um` are not independent fields.** σ ∝ clump_um / √a couples
   them. This is the second such relation found in this book — the first was Callier q ↔ grain
   size (p 235, already recorded) — and both point the same way: our weakest-evidenced
   quantity, `clump_um`, is constrained by two quantities we do hold.

⚠ **Do not convert (15) into σ_D without care.** ΔT is a transparency fluctuation; the naive
conversion σ_D = 0.4343·ΔT/T with T = 10^−D diverges at high density, which is unphysical. The
symbol definitions in the surrounding derivation (pp 867–869) have to be read before any
σ_D form is adopted. **The √(a′/a) scaling is unambiguous and is what should be taken.**

### 2.2 A caution that matters more than the law — book p 859 (PDF p 856)

van Kreveld measured granularity through **differently sized diaphragms** on named emulsions —
Ilford Special Rapid (D 0.35 and 1.00), Gevaert Contrast (0.27, 0.88), Gevaert Supercontrast
(0.35, 1.08), plotted in Fig. 296b. Mees's summary, verbatim:

> *"On the whole, the measured granularity is independent of area; but if the galvanometer
> correction is not made, the general trend is upward … An upward trend is also manifested by
> Selwyn's results."*

So the aperture dependence of granularity was **contested in the source itself**, and the
answer depended on the instrument's own response. This is a direct argument for the roadmap's
position that grain noise should be **measured on our own chain** rather than assumed from a
law — and a warning against treating σ√a = constant as settled physics.

### 2.3 Grain size distribution is log-normal — book p 54 (PDF p 51). **Vindicates a schema choice.**

Three candidate size-frequency forms are compared: Gaussian `Y = Ae^{−k(x−α)²}Δx` (1), simple
exponential `Y = Ae^{−kx}Δx` (2), and

> **Y = A e^{−k(log x − α)² } Δx**  (3)

of which Mees states it *"has seemed to have a close correlation with the properties of the
particle size distributions of photographic emulsions, and **in every case it fits the data
much better than either of the other two forms**"*.

`GrainSpec.size_sigma_log` is documented as a **tier-3** log-normal dispersion. The *value*
stays tier-3, but the **choice of a log-normal family is now a cited decision rather than an
assumption** — the same upgrade the existing extraction achieved for the reciprocity exponent
p = 0.95. It also settles the radius distribution to use in the Poisson-cluster grain process
recommended in the roadmap.

---

## 3. Davis & Walters, NBS Scientific Paper No. 439 — one number, and a hard blocker

A US Bureau of Standards survey (1920) of ~90 American plate and film brands: speed,
development, colour sensitiveness, filter factors, scale, resolution, irradiation, halation.
Exactly the kind of authoritative measurement series we want — except for one sentence:

> *"The results are therefore given without the names of the plates or the makers."*

**The brands are anonymised.** Nothing in it can be attached to a named stock without
committing precisely the graft the queue forbids (FP4 vs FP4 Plus, «Изопанхром ФОКХТ»). Its
value is methodology, and class-level statistics for 1920s emulsions.

**The one directly usable figure**, and it is a good one:

> *"The variations in speed that may be expected are from **5 to 25 per cent**, although much
> larger variations are occasionally found."*

That is a **manufacturing-tolerance** number from a standards body. It bounds how precise any
single stored speed can meaningfully be, and it is a candidate new parameter: a per-emulsion
batch variation that a render could sample once per simulated roll. Real footage from two
rolls of the same stock is not identical, and we currently model it as though it were.

---

## 4. The two documents with little to give

**The Photographic Emulsion** (Carroll, Hubbard & Kretschman, Focal Press) is a reprint
collection of *J. Phys. Chem.* and NBS papers from 1927–1930s on **emulsion making** —
comparison of bromides, chemical sensitization, after-ripening, silver-ion/gelatin
equilibrium, sulphite sensitization. It describes how to *make* an emulsion, not how a made
emulsion responds to light. Nothing actionable for the simulator.

**Ilford Book of Formulae, 3rd ed.** is a manufacturer handbook (rule 14: above a third-party
book) giving developer **formulations** by ID number — ID-2, ID-13, ID-19, ID-26, ID-33 and
others — with quantities and development times. There are **no sensitometric curves and no
gamma-versus-time data**, and the OCR corrupts numerals systematically (`z`, `I`, `o` for
digits), so no quantity can be taken without opening the page image. Use it for `processing`
developer provenance only. Absence of Phenidone products dates it to the 1940s–50s.

---

## 5. What to do with this, ranked

| # | Action | Effort | Value |
|---|---|---|---|
| 1 | Add the **ISO contrast diagnostic** (ΔD over 1.30 log H from D-min+0.10) as a reported derived column, **not** an assertion. Makes the curve-vs-EI development mismatch visible on all 53 B&W negatives | low | high — it is a real inconsistency nobody could see before |
| 2 | Use the **11/21-step tablet spec** (0.05→3.05, Δ0.30 / Δ0.15) as the reference wedge in `make_test_chart.py` for the comparison loop | very low | high — the loop needs a defined target |
| 3 | Add the **σ ∝ √(a′/a)** relation as a consistency check between `rms_granularity`, `clump_um` and the 48 µm aperture convention — alongside the existing Callier-q check | low | medium–high |
| 4 | Upgrade the `size_sigma_log` docstring from assumption to **cited** (Mees p 54, eq 3), and adopt log-normal radii in the Poisson-cluster grain work | very low | medium — it de-risks the biggest algorithm change on the roadmap |
| 5 | Record **batch variation 5–25 %** (NBS 439) as a documented manufacturing tolerance; consider a per-roll speed jitter | low | medium |
| 6 | Correct `MEES_1942_EXTRACTION.md` §6: the aperture law **is** on p 869, and the granularity chapter's van Kreveld material on p 859 is now read | very low | housekeeping, but it is the file the project trusts |

None of these changes a stored value. All are additive, and I have made none of them —
say the word and I will.

---

## 6. Still unavailable after this batch

* **No noise power / Wiener spectrum.** Mees 1942 predates the granularity Wiener-spectrum
  literature; the single "Wiener" hit in the whole book is in the back matter. The roadmap's
  item E1 is untouched by this shelf.
* **No interimage or DIR data** — as the existing extraction already established, and
  correctly: DIR couplers postdate this book by roughly thirty years.
* **No per-stock grain geometry.** We gained the *functional form* (log-normal) and the
  *scaling law* (√(a′/a)); we still have no measured clump diameter for any stock in the
  database.
* **No ECN-2 / ECP-2 process modules** — still the outstanding request to Kodak.
* **The aperture dependence is now known to be contested** (§2.2), which strengthens rather
  than weakens the case for measuring our own chain first.
