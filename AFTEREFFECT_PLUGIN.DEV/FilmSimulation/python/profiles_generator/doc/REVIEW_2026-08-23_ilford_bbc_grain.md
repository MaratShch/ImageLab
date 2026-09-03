# REVIEW 2026-08-23 — the two BBC documents: what is harvestable

Two owner-supplied scans, both **BBC Engineering Division**, both with an OCR text layer over page
images. Every number below was read from the **page image**, not from the OCR, and the OCR was used
only to find the pages. Where the OCR and the image disagree the image wins, and one such case is
flagged.

| Ref | Document |
| --- | --- |
| **[M54]** | K. Hacking, *An Analysis of Film Granularity in Television Reproduction*, **BBC Engineering Division Monograph No. 54, August 1964**. 26 PDF pages. `PDF/PROFILES/ILFORD/AN ANALYSIS OF FILM GRANULARITY.pdf` |
| **[T101]** | *Photographic film grain: a study with the aid of an optical correlator*, **BBC Research Department Report No. T-101, 1963/5**. 49 PDF pages. `PDF/PROFILES/ILFORD/1963-05.pdf` |

**Provenance class.** Both are BBC research reports, i.e. **third-party measurements** — method
rule 14 applies, and any Ilford or Kodak sheet would outrank them. But two things raise them above
the Soviet handbook currently cited for HPS: they are **primary measurements** (not a compilation),
and their speed table is explicitly headed **"MANUFACTURERS' DATA"**, so the ASA figures are Ilford's
and Kodak's own, relayed. They are also *contemporaneous* with HPS rather than retrospective.

---

## 1. ⚠ First, two of the figures in your message are NOT in these documents

Before anything else, because these are the two most likely to become wrong stored values:

| Claim | Status in these two documents |
| --- | --- |
| 320 ASA tungsten | ✅ **confirmed twice** — [T101] Table 1 p27, [M54] Table I p12 |
| 400 ASA daylight | ✅ **confirmed** — [T101] Table 1 p27 |
| Wiener spectrum 0.62 µm² | ✅ **confirmed** — [M54] Table I p12 |
| **800 ASA / 30 DIN** | ❌ **absent.** The string "DIN" does not occur in either document; the two "800" hits are a page number and a figure number. |
| **resolving power 40 lp/mm** | ❌ **absent, and there is a specific trap here.** Neither document states a resolving power for any film. "Resolving power" appears only as a property of the *television scanner's beam* ([M54] Fig. 4, p8: beam diameter 2–100 µm against 224–7 c/mm). And **[T101] p38 does contain "0 to 40 cycles/mm" — but that is the assumed system BANDWIDTH for Table 4's relative-granularity comparison, not a film resolving power.** Adopting 40 lp/mm from that sentence would be a unit-and-meaning error of exactly the kind the C11 CTF hazard was. |

Both are plausible figures for HPS from elsewhere — the later 800 ASA rating in particular is
consistent with Ilford's re-rating history. **They need their own source before they go in.**

---

## 2. HPS — what is now quantitative

### 2.1 Directly printed, adoptable as-is

| Quantity | Value | Source |
| --- | --- | --- |
| Speed, tungsten | **320 ASA** | [T101] Table 1 p27; [M54] Table I p12 |
| Speed, daylight | **400 ASA** | [T101] Table 1 p27 |
| BS logarithmic speed | **36°** | [M54] Table I p12 ⚠ see §5 |
| Gauge, as measured | 35 mm | both |
| Description, as published | "Panchromatic film of extreme speed" | [T101] Table 1 |
| Typical use | "Special cine-camera work" | [T101] Table 1 |
| **Process-control gamma** | **0.63** | [T101] Tables 2 p28 and 4 p38 |
| **Wiener spectrum, mean 0–20 c/mm** | **0.62 µm²** | [M54] Table I p12 |
| **Equivalent grain diameter** | **2.5 µm** | [T101] Table 2 p28 |
| Mean-signal-to-rms-noise, t̄/σ | 0.96 at t̄ = 0.31 | [T101] Table 2 p28 |
| Relative granularity (5302 = 1.0, 0–40 c/mm) | **3.9** | [T101] Table 4 p38 |
| Wiener spectrum flatness | falls **≈10 %** over 0–60 c/mm | [T101] p38 |

**Measurement conditions, which matter more than the numbers.** [M54]: uniformly exposed samples,
standard negative developer, all developed to ≈ the same gamma (Eastman TIB control γ = 0.65), mean
optical density **≈ 0.48 above base**, E.E.L. densitometer. [T101]: frames selected at mean
transmission **t̄ ≈ 0.33** relative to base, i.e. **D ≈ 0.48–0.51 above base**; commercial
cine-processing laboratory; equivalent grain diameter from an optical autocorrelator at ×400.

⚠ **Neither is at the project's net-1.0 convention.** Both sit at D ≈ 0.5 above base, so nothing
here can be dropped into `rms_granularity` without a density conversion — see §2.3.

### 2.2 The database's HPS entry, checked against this

| Field | Stored now | These documents | Verdict |
| --- | --- | --- | --- |
| `exposure_index` | 400 | 400 daylight / **320 tungsten** | ✅ consistent; the tungsten figure is now sourced twice, and `_TUNGSTEN_EI` already holds 320 |
| `gamma` | 0.62 (estimate) | **0.63** measured | ✅ **confirmed to 0.01.** An estimate that turns out right is worth pinning |
| `rms_granularity` | 19.0 (estimate) | derivable ≈ **18.5** at D 0.48 (§2.3) | ✅ **confirmed to 3 %**, and now derivable rather than asserted |
| `clump_um` | **26.0** (estimate) | **2.5 µm** measured | ❌ **conflict, ~10×** — see §4, the most consequential finding here |
| `f50` | 26.0 (estimate) | **nothing** | ⚠ still unsourced; and 40 lp/mm is not in these documents |
| `dmin` 0.21, `fog_grain` 0.40 | rendering intent (pushed New-Wave look) | nothing | unchanged, still declared intent |
| `callier_q` | 1.3 (class rule) | **Tri-X measured 2.0–2.34** (§6) | ⚠ needs a decision, not a substitution |

### 2.3 The conversion that makes the Wiener figure usable — and its self-check

A Wiener spectrum in µm² is not an rms granularity, but for a spatially white grain field the two
are one step apart. For an aperture of area *A*, σ²·A = W(0), and Selwyn's G = σ√(2A), so:

```
σ(A) = sqrt( W(0) / A )        A = π·(48/2)² = 1809.6 µm² for the 48 µm aperture
```

⚠ **This is a derived quantity and it is only worth anything because it can be checked against
published values for the other three films in the same table.** Applying it to [M54] Table I:

| Emulsion | W(0) µm² | → σ×1000 at D 0.48 | Published diffuse RMS at **net 1.0** | direction |
| --- | --- | --- | --- | --- |
| Ilford Pan F | 0.10 | **7.4** | ~5–6 (Ilford, modern Pan F Plus) | plausible |
| Kodak Plus-X | 0.14 | **8.8** | 10 (stored on `EASTMAN_PLUS_X_5231`) | ✅ lower at lower density, as it must be |
| Kodak Tri-X | 0.555 | **17.5** | 17 (stored on `KODAK_TRI_X_400TX`) | ✅ agrees |
| **Ilford HPS** | **0.62** | **18.5** | — | **19.0 currently stored as an estimate** |

Three of the four land on or just below their published net-1.0 values, which is the correct
direction (grain rises with density). So the conversion is sound, the HPS 18.5 is credible, and the
stored 19.0 is vindicated rather than replaced. **Recommendation: keep 19.0, and cite this as the
corroboration** — changing 19.0 → 18.5 would be trading a confirmed estimate for a figure measured
at the wrong density.

---

## 3. What is harvestable for the OTHER films — this is the larger prize

Both documents measure a **family of six**, which matters because it clears method rule 18 (no class
estimate from one sample).

### [T101] Table 2, p28 — measured per emulsion, all at t̄ ≈ 0.33

| # | Emulsion | gauge | γ | t̄ | t̄/σ | **equiv. grain dia. µm** | in the database? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Ilford **HPS** | 35 | 0.63 | 0.31 | 0.96 | **2.5** | `ILFORD_HPS` |
| 2 | Kodak **Tri-X type 5223** | 35 | 0.64 | 0.36 | 1.02 | **2.2** | ❌ **no 5223 profile** (there is `KODAK_TRI_X_400TX`, a still stock) |
| 3 | Kodak **Plus-X type 4231** | 35 | 0.64 | 0.33 | 1.19 | **1.45** | `EASTMAN_PLUS_X_5231` — same emulsion, 4231 is the Estar-base number |
| 4 | Ilford **Pan F** | 35 | 1.0 | 0.37 | 1.22 | **1.5** | `ILFORD_PAN_F` |
| 5 | Kodak **8374** TV recording, blue+UV | 16 | 1.0 | 0.33 | 1.28 | **1.2** | ❌ absent |
| 6 | Kodak **5302** fine-grain release positive | 16 | 2.4 | 0.30 | 1.36 | **1.03** | ❌ absent as a print stock |

### [T101] Table 4, p38 — relative granularity, 0–40 c/mm, 5302 = 1.0

HPS **3.9**, Tri-X **3.5**, Plus-X **1.8**, Pan F **1.9**, 8374 **1.3**, 5302 **1.0**.

⚠ Note Pan F (1.9) reads *coarser* than Plus-X (1.8) here despite being four stops slower — because
Pan F was developed to γ 1.0 against Plus-X's 0.64. That is a real and useful datum about
**development gamma driving granularity**, not an inconsistency.

### [M54] Table I, p12 — speed against Wiener level, tungsten

Pan F 16 ASA / 23° / 0.10 · Plus-X 64 / 29° / 0.14 · Tri-X 250 / 35° / 0.555 · HPS 320 / 36° / 0.62.
[M54] Fig. 9 p13 plots these and the text's own conclusion is worth keeping verbatim: *"It is
unlikely that a simple empirical relationship exists between film-speed and granularity: each
unknown emulsion should be measured."* — which is the same lesson C24 reached about f50.

### Other extractable data, per document

| Datum | Value | Source |
| --- | --- | --- |
| **Kodak 5302 print-stock Wiener spectrum** | **0.04 µm², uniform, at D 0.5 above base** | [M54] p16 assumption 2 |
| Grain-vs-density law, negative, telecine signal domain | **S/N ≈ k·D^−0.6**; Higgins & Stultz suggest the exponent is nearer **−0.4** | [M54] eq. (4) p13 |
| Same law, reversal processing | exponent numerically **larger, −0.6 to −0.7** | [M54] p13 |
| Print-through granularity law | S/N ≈ k′(a²γ_p²D_n + c²D_p)^−0.5 | [M54] eq. (5) p13 |
| Grain correlation extent | *"substantially confined to about ± one equivalent grain diameter"* | [T101] p38 |
| Grain size distribution | *"substantially unimodal … no pronounced subsidiary peaks"* | [T101] p38 |
| Autocorrelation shape | Gaussian-like near zero, then a **small negative undershoot (<1 %)** before converging | [T101] p37 |
| Development gamma vs grain size | Table 3 p39 + Figs. 22–24: γ changes equivalent grain diameter measurably | [T101] |
| **Callier quotient, Tri-X 5223** | **log₁₀Q 0.37 → 0.30 over D 0.1 → 1.0**, i.e. Q **2.34 → 2.00**; two gammas (0.94, 0.56) | [T101] Fig. 25 p37 |
| Specular collection angle for that measurement | **0.0016 steradian** | [T101] p37 |

### Digitizable curves

| Figure | Content | Worth tracing? |
| --- | --- | --- |
| **[M54] Fig. 8, p13** | Measured Wiener spectra, 0–150 c/mm, four negatives, µm² — **legible, clean, well-gridded** | **Yes.** This is the grain power spectrum of HPS, Tri-X, Plus-X and Pan F as measured |
| **[T101] Fig. 18, p35** | Wiener spectra of all six emulsions, from 2-D Fourier transform of the measured autocorrelations | **Yes** — six emulsions, and it overlaps [M54] Fig. 8 on four of them, so the two are a **mutual check** |
| **[T101] Fig. 17, p34** | Measured autocorrelation functions, six emulsions | **Yes** — the same information in the conjugate domain, and it is where the ±1-grain-diameter claim and the negative undershoot are visible |
| [T101] Figs. 19, 20, 23, 24 | Pan F at five densities; Pan F and Tri-X at several gammas | Yes, second tier — a measured σ-vs-density *shape*, which is the C1 family of data |
| [T101] Fig. 25, p37 | Callier quotient vs density, Tri-X, two gammas | **Yes** — see §6 |
| [M54] Fig. 10, p13 | Printing-process development characteristic + point gamma | Marginal — a generic printing curve, not a named stock's |
| **[M54] Fig. 12, p16** | ⚠ **the one you pointed at, and it is NOT film data** | **No.** It is *displayed* granularity through a whole television chain: it assumes printing onto Kodak 5302, Lamberts' printing response, a 13 c/mm video bandwidth, a Gaussian system aperture 1.5 dB down at 13 c/mm, and peak white at positive density 0.2. The HPS curve on it is a system result, not an emulsion property. The film data behind it is Fig. 8 |

---

## 4. ⚠ The consequential finding: `clump_um` looks ~10× too large

This is the one item I would not touch without your decision, because it is not about HPS alone.

`GrainSpec.clump_um` is documented as *"mean developed clump diameter, micrometres"*, and
`film_sim.grain_shape()` turns it into a spectrum rolloff at **f_hi = 1000/(2·clump_um)** c/mm.

* HPS stores **26.0 µm** → f_hi = **19 c/mm**. The grain spectrum would be down to 1/e by 19 c/mm.
* [T101] p38 measures HPS's Wiener spectrum falling **≈10 % over 0–60 c/mm**. Solving
  `exp(−(60/f_hi)²) = 0.9` gives f_hi ≈ 185 c/mm, i.e. **clump_um ≈ 2.7 µm**.
* [T101] Table 2 measures HPS's equivalent grain diameter as **2.5 µm** — independently, by
  autocorrelation.
* [M54] Fig. 8 agrees in shape: HPS runs 0.625 µm² at f = 0 and is still ≈0.47 at 100 c/mm.

Two independent measurements and the model's own formula converge on **2.5–2.7 µm** where the
database carries **26 µm**. The stored value is not a slightly-wrong estimate; it puts the grain
rolloff an order of magnitude too low in frequency, which makes rendered HPS grain far coarser and
blobbier than the measurement supports.

**Scope, and why this needs you.** The database's `clump_um` runs 3.2–40 µm with a **median of 13**,
while the six measured emulsions here — spanning a fine-grain release positive to the fastest cine
negative of its day — span **1.03–2.5 µm**. If the measurement is right, the field is
systematically ~5–10× high across the file, and correcting it would change grain *texture* on many
stocks. That is a large visual change and it is your call, not mine.

Three ways to go, in the order I would rank them:

1. **Correct only the six measured emulsions** (four of which have profiles), leave the rest, and
   record the conflict in the `GrainSpec` docstring. Smallest change, fully sourced, and it makes
   the discrepancy visible instead of silent.
2. **Trace [M54] Fig. 8 and [T101] Fig. 18 first**, fit the project's actual grain-spectrum model to
   the measured spectra, and let the fit — not the printed grain diameter — set `clump_um`. Slower,
   but it answers the real question, which is whether `exp(−(f/f_hi)²)·(1 + gain·exp(−(f/f_lo)²))`
   can even represent a measured grain spectrum. That is the same question C2 asked about the MTF
   carrier, and there the measurement changed the model's *form*.
3. Leave it and record the conflict only.

I recommend **2 then 1**: the curves are legible, they are the actual evidence, and a fitted value
carries its own residual.

---

## 5. Two document defects to record, not silently absorb

**[M54] Table I gives HPS 36° BS while the arithmetic wants 38°.** The BS logarithmic scale is
10·log₁₀(ASA) + 10 in effect: Pan F 16 → 23°, Plus-X 64 → 29°, Tri-X 250 → 35° all fit, and
320 ASA → **36.1°**, so **36° is correct and internally consistent** — I read it off the image and
it is 36, not 38. If a 38° figure exists elsewhere it refers to a later, faster rating (which is
where an 800 ASA / 30 DIN claim would sit), not to the 320 ASA film measured here. ⚠ Also note
[M54]'s own footnote to Table I: *"Earlier speed ratings (prior to the revised indices)"* — these
are pre-revision ASA numbers, which for negative stocks means the modern equivalents are about a
stop faster. **That footnote is the most likely origin of a "later 800 ASA" figure**, and it is a
real reason not to treat 320/400 and 800 as a contradiction.

**[T101] p38 misidentifies the HPS emulsion number.** The sentence reads *"the spectrum of the HPS
emulsion (No. 6), for instance, falling by only about 10 % in this frequency range"* — but No. 6 is
Kodak 5302 and HPS is No. 1, per its own Table 1. A printing error in the original. The 10 % figure
belongs to HPS (it is the coarsest, flattest spectrum in Fig. 18); the emulsion number does not.

---

## 6. ⚠ [T101] Fig. 25 is the C22 document I said did not exist

The `RESULT_2026-08-23c` note listed, as C22's one remaining gap: *"one densitometer specification
stating a diffuse-versus-specular ratio for a named emulsion"*. [T101] Fig. 25 p37 is a measurement
of exactly that, on a named emulsion (Eastman Tri-X 5223), at two development gammas, **as a
function of density**:

| diffuse D | log₁₀Q at γ 0.94 | Q | log₁₀Q at γ 0.56 | Q |
| --- | --- | --- | --- | --- |
| ≈0.1 | 0.37 | 2.34 | 0.37 | 2.34 |
| 0.5 | ≈0.335 | 2.16 | ≈0.315 | 2.07 |
| 1.0 | 0.30 | 2.00 | ≈0.29 (extrapolated) | 1.95 |

Three things follow, and they do not all point the same way:

1. **The magnitude is far above the database's class estimate.** `callier_q` = 1.3 on monochrome
   negatives; this measures **2.0–2.34**.
2. **But it does not simply replace it, and this is the C22 thesis restated by the document
   itself.** The collection angle here is **0.0016 steradian** — very nearly collimated, the
   limiting case. A real condenser enlarger or a directed-source scanner accepts a far wider cone
   and reads a lower Q. So this figure is the **upper bound** of what a directional reader can see,
   and 1.3 is a plausible value for an ordinary condenser. The document therefore *supports* the
   split into film-scattering × reader-directionality rather than collapsing it.
3. **Q falls with density, and the current model cannot express that.** `AlgoCallierFactor` treats Q
   as constant. Measured, Q drops ~15 % from D 0.1 to D 1.0, and it also depends on development
   gamma. Both are real second-order effects the model currently ignores.

**Recommendation.** Do not change `callier_q`. Instead: cite this as the anchor for the *geometry*
axis — it fixes what `scanner_specular = 1` should mean physically (Q ≈ 2.3 at near-zero acceptance
angle for a fast silver negative) — and record the density dependence as a known, measured
limitation of the constant-Q form. That converts C22's open item from "no document exists" to "the
document exists and says the shape is slightly wrong", which is a much better place to be.

---

## 7. Summary: what I would adopt, and what needs you

### Adopt directly (all `[T2]` third-party, all with page references)

* `ILFORD_HPS`: gamma **0.63** confirmed (replaces the 0.62 estimate at the same precision);
  tungsten 320 / daylight 400 given a second, contemporaneous, manufacturer-sourced citation;
  Wiener 0.62 µm² and equivalent grain diameter 2.5 µm recorded with their conditions;
  relative granularity 3.9 (5302 = 1) recorded. `rms_granularity` **stays 19.0**, now corroborated
  rather than asserted.
* `ILFORD_PAN_F`, `EASTMAN_PLUS_X_5231`, `KODAK_TRI_X_400TX`: the same measured quartet, with the
  4231/5231 base-number equivalence stated and the 400TX-vs-5223 difference stated as a caveat.
* A `_DATASHEET_SOURCES` citation for both documents, and the `NotFound.md` HPS entry rewritten:
  HPS is no longer "practically undocumented", and the Soviet handbook is no longer the only source.

### Needs your decision

| # | Decision |
| --- | --- |
| 1 | **`clump_um` ~10× too large** (§4). Fix six emulsions only, or trace Figs. 8/18 first and fit the model, or record only? |
| 2 | **Trace the three curve families?** [M54] Fig. 8, [T101] Figs. 17 and 18. These are new *kinds* of data for this project — grain power spectra and autocorrelations — and they would test whether the grain-spectrum model's form is right, exactly as C2 tested the MTF carrier |
| 3 | **Add missing profiles?** Kodak **Tri-X 5223** (35 mm cine negative, distinct from 400TX), Kodak **8374** (16 mm TV recording, blue+UV), Kodak **5302** (16 mm release positive — this one belongs in `PRINT_STOCKS`, and we now have a measured Wiener spectrum **and** a grain diameter for it, which no existing print stock has) |
| 4 | **Callier** (§6): cite as the geometry anchor and record the density dependence, or leave C22 as it stands? |
| 5 | **The two absent figures** (§1): where do 800 ASA / 30 DIN and 40 lp/mm come from? They are not in these documents and I will not store them without a source |

### Insufficiently documented — recorded, not adopted

* HPS `f50` — nothing in either document. Still the estimate 26.0.
* HPS spectral sensitivity, characteristic curve, Dmax, reciprocity — absent from both.
* [M54] eq. (4)'s S/N ∝ D^−0.6 is in the **telecine signal domain**, not σ_D vs D. It is suggestive
  of the project's σ(D) shape but is not directly convertible, and the document itself gives two
  different exponents (−0.6 and Higgins & Stultz's −0.4). Recorded as literature, not as data.
