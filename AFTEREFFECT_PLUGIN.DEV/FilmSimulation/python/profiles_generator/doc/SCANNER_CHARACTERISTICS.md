# SCANNER_CHARACTERISTICS.md — the digitisation half of the chain

**Created 2026-09-01d.** A technical reference for modelling the film-to-digital scanning
pipeline. It is **deliberately separate from the film-profile documentation**, because the
quantities in it are not properties of any film and must never be stored on one.

Primary source: **Flueckiger, Pfluger, Trumpy, Croci, Aydın and Smolic, «Investigation of Film
Material–Scanner Interaction», University of Zurich / DIASTOR, v1.1, 18 February 2018**, 88 pages,
`PDF/PROFILES/RETRO/flueckigeretal_investigationfilmmaterialscannerinteraction_2018_v_1-1b.pdf`.
Re-derived every build by `flueckiger_2018.py`.

---

## 0. The question this document exists to answer

The observed pixel is not the film. It is

```
  physical film characteristics
+ exposure and development
+ film ageing
+ scanning / digitisation
= observed digital film scan
```

and this project currently models the first three and silently assumes the fourth is the identity.
That assumption is wrong, and the report quantifies **how** wrong: two scans of the *same*
Dufaycolor frame on the *same* scanner model at 2K and 4K differ visibly (Fig. 27), and the
measured contrast reproduction at 20 lp/mm ranges from **0.13 to 0.83** across eight machines —
a factor of six on the same test target.

⚠ **But that does not make scanner modelling correct for this project.** §5 gives the verdict.

---

## 1. Parameter classification — the five categories, kept strictly apart

The user's request was explicit that these must not be blurred. They are listed here as the
governing rule for anything added later.

### 1.1 Film-intrinsic — belongs in `film_profiles.py`
Spectral sensitivity, characteristic curve, dye analytical densities, grain (rms, clump, σ(D)),
MTF of the emulsion, halation, base thickness and material, crystal size and habit, Callier β.
**A scanner never changes these.** They are what the film *is*.

### 1.2 Processing-dependent — belongs on the profile, keyed by development
Developer, time, temperature, gamma, push/pull, `ProcessVariant`, reciprocity development
correction. Already modelled. Grain clump size is *also* here (D_eq ∝ γ^0.42) and the schema
stores it as one scalar — a known limitation recorded in `GrainSpec`.

### 1.3 Scanner / digitisation — belongs HERE, never on a film profile
Illumination spectrum and collimation, sensor spectral sensitivity, colour filter array and
de-mosaic, optical MTF, sampling resolution, dynamic range, read noise, A/D bit depth, flare,
clipping, colour management. **None of these is a property of a film**, and a `FilmProfile` field
holding one would be a category error that a future reader would inherit as a fact.

### 1.4 Interaction — the genuinely hard category
Quantities that exist only when a *specific* film meets a *specific* scanner:
- **The Callier effect.** Silver scatters; a collimated source reads a higher density than a
  diffuse one. Q is a film property (β) *and* a geometry property (E). This project already
  splits it correctly — `callier_q` on the profile, `scanner_specular` as a control. §2.8.1 of the
  report is an independent confirmation that the split is the right shape, and adds an open
  question: **Q may be wavelength-dependent**, which would make a tinted film read a different
  *hue* under a condenser than under a diffuser. The report observed exactly that (Fig. 14) and
  could not explain it otherwise.
- **Colour separation.** How much of a dye's side absorption reaches the wrong channel depends on
  both the dye's spectrum and the scanner's spectral bands. This is the product of `dye_density`
  and the scanner's sensitivity, and it cannot be attributed to either alone.
- **Réseau / mosaic aliasing.** Dufaycolor's 500-lines-per-image screen against the sensor's pixel
  grid, at 23°. Moiré is an interaction, not a film property.

### 1.5 General scientific knowledge — for the algorithm, not the database
The report's equations (1)–(9), the Ohta PCA separation method, the physical account of why dye
images do not scatter and silver images do. Recorded in `EMULSION_KNOWLEDGE_BASE.md` and here.

---

## 2. What the report actually measures, and what it only asserts

### 2.1 Scanner MTF — **measured, eight machines** (Fig. 61, digitised)

Fraction of reproduced contrast `F(ξ)`, from the report's equation (8):

```
F(ξ) = ( V_light(ξ) − V_dark(ξ) ) / ( V_light(0) − V_dark(0) )
```

measured on an ARRI AQUA test leader written by an ARRILASER at 4K, green channel, on the central
concentric bar patterns. `F(0) = 1` by definition.

| scanner | 10 | 12 | 14 | 16 | 20 | 28 | 40 lp/mm | px/mm | Nyquist |
|---|---|---|---|---|---|---|---|---|
| Scanity (Digimage) | 0.912 | 0.896 | 0.897 | 0.879 | **0.834** | 0.682 | 0.412 | 186 | 93.0 |
| Scanity (Sound & Vision) | 0.899 | 0.896 | 0.873 | 0.842 | 0.712 | 0.445 | 0.112 | 85 | 42.5 |
| The Director | **0.930** | 0.879 | 0.849 | 0.796 | 0.612 | 0.366 | 0.142 | 152 | 76.0 |
| Northlight 1 | 0.881 | 0.857 | 0.831 | 0.782 | 0.671 | 0.432 | 0.155 | 170 | 85.0 |
| Kinetta | 0.857 | 0.858 | — | 0.808 | 0.650 | 0.399 | 0.130 | 172 | 86.0 |
| ARRISCAN | 0.837 | 0.824 | 0.792 | 0.739 | 0.638 | **0.498** | **0.277** | 170 | 85.0 |
| Altra mk3 | 0.796 | 0.726 | 0.651 | 0.539 | 0.365 | 0.150 | 0.023 | 105 | 52.5 |
| D-Archiver Cine10-A | 0.735 | 0.606 | 0.485 | 0.314 | 0.130 | 0.037 | — | 67 | 33.5 |

⚠ **Table 3 is checked, not transcribed on trust.** Figure 62 replots Figure 61 with the abscissa
divided by these same pixel counts (equation 9), so each series' last point must land at
`max(ξ)/pxl_per_mm`. It does, for all eight: 0.235 / 0.471 / 0.263 / 0.235 / 0.233 / 0.235 /
0.381 / 0.418 lp/pixel.

⚠ **Two caveats the report states itself.** Some curves are *not* monotonic — presumably in-scanner
sharpening, which means F is a **system** response including processing, not a lens MTF. And
"2K / 4K / 6K" is described as "a mere approximation": the machines cover different fractions of
the film width, which is why px/mm ranges 67–186 for nominally comparable scanners.

### 2.2 Illumination and sensor — **tabulated, mostly qualitative** (Table 1)

| scanner | light | components | sensor | shape | chroma | dynamic range | Bayer |
|---|---|---|---|---|---|---|---|
| Sondor Altra mk3 | LED | R, G, B | CCD KAI-04050 | area | colour | 64 dB | yes |
| CIR D-Archiver Cine10-A | LED | white | CCD KAI-4021 | area | colour | 60 dB | yes |
| Kinetta | LED | R, G, B + white | CCD KAI-16070 | area | colour | 70 dB | yes |
| Lasergraphics Director | LED | R, G, B | CCD | area | mono | not published | no |
| Digital Vision Golden Eye | LED | R, G, B | CCD | trilinear | colour | ADC 16 bit | no |
| FilmLight Northlight 1 | HTI | white + MSO filter | CCD | trilinear | colour | not published | no |
| DFT Scanity | LED | 2R, G, B + integrating sphere + beamsplitter | CCD | 3 × TDI linear | mono | ADC 14 bit | no |
| ARRI ARRISCAN 4K | LED | R, G, B | CMOS | area | mono | ADC 14 bit | no |

⚠ **Three of eight publish a dynamic range in dB and three more publish only an ADC bit count.**
The report's §2.2 is explicit that these are different quantities — equation (5) requires
`2ⁿ−1 ≥ FullWell/ReadNoise`, so bit depth *bounds* dynamic range and does not state it.

**Spectral bands, where stated:**
- Ideal narrow-band separation: **460, 520, 680 nm at FWHM 20 nm** (§2.3, Fig. 7) — the positions
  that maximise separation for modern chromogenic dyes.
- DFT Scanity beamsplitter filters: **390–490 / 510–570 / 630–770 nm** (§2.7).
- FilmLight Northlight 1: an **MSO filter** excluding the 500 and 600 nm crossing regions (Fig. 9).
- Kinetta: R, G, B and white LED spectra plotted (Fig. 36) but not tabulated.

⚠ **No scanner's RGB spectral sensitivity is published anywhere in the report.** §2.8 says so in
as many words: "While it has not always been possible to obtain information about the scanners'
spectral sensitivity, we were able to measure the spectral characteristics of the film stocks."
Figure 6 is captioned "**Typical** normalized spectral sensitivities of a color imaging device" —
a textbook illustration, not a measurement of any machine in the study.

### 2.3 What the report does **not** provide
Noise (no read-noise figures, no measured SNR), channel cross-talk as a matrix, flare or
stray-light numbers, clipping thresholds, A/D transfer functions, or any colour-management
profile. Also no per-scanner spectral sensitivity, per §2.2 above.

---

## 3. The physics the report states, which is useful independently of scanners

Equation (1) is the model this database already implements as `dye_density` plus a 3×3
`dye_matrix`, written out:

```
A_film(x,y)(λ) = K_Y(x,y)·Ā_Y(λ) + K_M(x,y)·Ā_M(λ) + K_C(x,y)·Ā_C(λ)
```

with `Ā` the **normalised analytical densities** and `K` the local dye concentrations; the sum
`A_film` is the **integral density**. Valid only when the film does not scatter, with the base
taken as transparent (Hunt 2004). Equation (2): `T% = 10^−A · 100`. Equation (3) puts the scanner
in: `I_out(λ) = I_in(λ)·10^−A_film(λ)`.

⚠ **This is a citation for a modelling choice the project already made and had not justified in
print.** It also states the limit of that choice: masked negatives break it, and §2.1 sets the
mask aside explicitly rather than pretending otherwise.

**Why colour film has Callier Q ≈ 1 and silver film does not** (§2.1, §4 of the companion Trumpy
paper): the refractive indices of the dye clouds and of the gelatin are similar, so a dye image
absorbs without scattering, whereas developed silver particles — "typically 0.2 µm to 2 µm"
[Vitale 2009] — differ optically from gelatin and scatter strongly.

---

## 4. Can 5–7 scanners be taken from this report as simulation presets?

**Partly, and the honest answer is narrower than the question.**

**What is sufficient for a preset today** — for all eight machines:
- a measured **MTF** at 7 spatial frequencies (§2.1 above), which is directly comparable with the
  emulsion MTF the renderer already applies;
- a **sampling resolution** in px/mm and its Nyquist limit, which fixes aliasing and the réseau
  moiré case;
- **light source class** (LED narrow-band composite / LED white / HTI white + filter) and whether
  a **CFA de-mosaic** is in the path;
- for three machines, a **dynamic range in dB**; for three more, an ADC bit count.

**What is missing for a spectrally correct preset** — for every machine:
- the RGB **spectral sensitivity**, which is the single most important scanner parameter for
  colour and is published for none of them;
- noise, flare, cross-talk, clipping behaviour.

So a scanner preset built from this report would be an **MTF + sampling + de-mosaic** model with a
*qualitative* spectral character, not a colorimetric one. That is still worth having — it is
exactly the part that explains the biggest visible differences in the study — but it must not be
dressed up as a spectral scanner model.

Recommended set if this is pursued, chosen to span the measured range rather than to be
comprehensive: **Scanity (Digimage)** as the sharp trilinear/TDI extreme, **ARRISCAN** as the
high-resolution area-CMOS case, **Northlight 1** as the white-light + MSO-filter case,
**Kinetta** as the adjustable composite-light Bayer case, and **D-Archiver Cine10-A** as the soft
low-resolution extreme. Five machines cover F(20 lp/mm) from 0.13 to 0.83.

---

## 5. ⚠ Professional verdict — is scanner modelling justified, or overfitting?

**Justified in one narrow form; overfitting in the general form. The two must not be conflated.**

**The case for.** The project's stated goal is physical realism, and its output is compared, by
users, against *scanned* film — never against film. Every reference image anyone has ever seen of
a given stock came through a scanner or a telecine. A pipeline that models emulsion MTF to two
decimal places and then assumes a perfect sampler is not more honest than one that models the
sampler; it has simply hidden the assumption. And the sampling half is cheap: an MTF multiply in
the frequency domain is a stage this renderer already runs for the emulsion.

**The case against, which is stronger than it looks.** Three arguments:

1. **The user is the scanner.** A film-simulation plugin renders to a chosen output resolution. The
   pixel grid, the resampling and the sharpening are the *user's*, applied after our output. Baking
   one 2015 archival scanner's 170 px/mm sampling into the render imposes a second, invisible
   sampler on top of theirs. Two samplers in series is not more realistic than one.
2. **The data does not support the interesting half.** No spectral sensitivity for any machine
   means the part that would actually change *colour* — the thing users notice — cannot be
   modelled from this source at all. What can be modelled is sharpness, which the user already
   controls directly.
3. **It is unfalsifiable here.** `NotFound.md` row 9 records the project's largest gap: there is no
   ground-truth harness, nothing compares a render against a photograph. Adding a scanner model
   would add free parameters to a system that cannot yet measure whether they help. That is the
   textbook definition of overfitting.

**Therefore:**

- ✅ **Adopt as reference documentation** — this file. The numbers are measured, checkable and
  worth having on file.
- ✅ **Adopt the one genuinely film-side finding**: the Callier effect's possible **wavelength
  dependence** (§2.8.1). That is a film property, it belongs on the profile side, and it is
  already an open queue item (C43/C44).
- ⚠ **Defer a scanner stage** until there is a ground-truth harness to test it against. When one
  exists, the first thing to add is an optional **scanner MTF + sampling** stage driven by a preset
  from §2.1 — inert by default, exactly like `scanner_specular`.
- ❌ **Do not add scanner fields to `FilmProfile`.** Ever. A scanner is not a film. If a scanner
  model is built, it belongs in a separate `ScannerProfile` table with its own provenance, selected
  by a control, defaulting to "none".

---

## 6. What implementing scanner modelling would require

Listed so the cost is visible, not as a recommendation.

| step | work | cost |
|---|---|---|
| `ScannerProfile` struct + table | new dataclass, C++ emitter, codegen, validators, parity family | ~1 schema version, mirrors `ReciprocitySpec` in shape |
| Control | `AlgoControls::scannerProfile` (int32, −1 = none), plus a resolve-once-per-frame helper like `AlgoResolveProcessVariant` | small; frame setup only |
| MTF stage | reuse the existing separable-blur machinery with a second transfer curve; **zero new passes** if folded into the emulsion MTF stage's transfer, since both are multiplies in the same frequency domain | **tiny** per pixel if folded, moderate as a separate pass |
| Sampling / Nyquist | a resample to the scanner grid and back, or a pre-filter at the Nyquist limit | moderate — one extra pass, and it interacts with output resolution |
| De-mosaic | only meaningful at the sensor grid; would require the sampling step first | high |
| Spectral | **blocked** — no source in the corpus publishes a scanner's RGB sensitivity | n/a |
| Bit depth / clipping | a quantiser and a clamp at frame setup | tiny |

**Inertness contract**, non-negotiable and the same one every other stage here follows: at
`scannerProfile < 0` the stage must return before touching a pixel and every prior render must be
reproduced bit-for-bit.

---

## 7. Source register for this document

| ref | what it gives | status |
|---|---|---|
| Flueckiger et al. 2018 §2.1–2.3 | equations (1)–(3), the linear dye model, colour separation | quoted, §3 |
| §2.2 | equations (4)–(5), dynamic range and ADC bit depth | quoted, §2.2 |
| §2.4 Table 1 | light source and sensor per scanner | digitised, §2.2 |
| §2.7 | Scanity filter bands, Northlight MSO filter | quoted, §2.2 |
| §2.8.1 Figs. 13–15 | Callier effect, diffuse vs collimated, possible λ dependence | §1.4, §5 |
| §3.1 Table 2 | model overview for seven manufacturers | not digitised — catalogue, not measurement |
| §4.1 Fig. 61, Fig. 62, Table 3 | **measured scanner MTF and sampling resolution** | digitised, §2.1 |
| §4.2 | eleven-expert subjective rating study | not used — opinion, not measurement |

⚠ **Nothing in this document is stored in `film_profiles.py`.** The film-side harvest from the same
report — the Technicolor dye set and the Dufaycolor réseau transmittances — is recorded in
`EMULSION_KNOWLEDGE_BASE.md` §23h and in the profiles themselves, and is kept separate on purpose.
