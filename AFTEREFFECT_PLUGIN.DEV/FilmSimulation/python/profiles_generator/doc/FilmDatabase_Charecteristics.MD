# Film Simulation — Technical Data Requirements for 90–95 % Fidelity

**Document type:** vendor-neutral engineering requirements and data-acquisition
specification.
**Vendor neutrality:** no manufacturer, product family or individual material is
named. Usable as-is when approaching any manufacturer, processing laboratory,
archive, standards body or research publication.
**Constraint that governs this document:** the simulator must produce images whose
measurable and perceptual characteristics are ~90–95 % consistent with scans of
real film. Nothing else constrains it — not an existing schema, not an existing
pipeline, not what is conveniently available.

**Reviewed 2026-08-25.** Sections **9** (model status) and **Appendix D.2** (gap
analysis) were written against schema v5 and have been revised rather than
appended to: each carries a marked banner naming which of its own claims are now
superseded, and the status log at the end of the file — which had stopped on
2026-08-17 — resumes with a consolidated 2026-08-18 → 2026-08-25 entry. The
requirements themselves (§§1–8) are unchanged and were re-checked: they still
describe what the target demands. Three of them gained *measured* content rather
than corrections — §4.11's joint processing dependence, §4.20's dye-trace
identification and §4.21's granularity pairing — and each is marked in the
consolidated entry. **Vendor neutrality is preserved throughout: no manufacturer,
product family or individual material is named anywhere in this revision.**

---

# 1. What the target actually demands

## 1.1 Fidelity must be decomposed before it can be engineered

A single percentage is not measurable. Four independent tests are, and each
stresses a different part of the data set.

| | Test | Measured quantity | Proposed acceptance for "90–95 %" |
|---|---|---|---|
| **T1** | Sensitometric | per-layer density error over a common exposure series | mean \|ΔD\| ≤ 0.02, max ≤ 0.05 |
| **T2** | Spatial | modulation transfer and noise power | MTF ±3 % absolute to 1.5·f₅₀; σ(D) ±5 % at ≥3 densities; noise-spectrum shape ±10 % |
| **T3** | Colorimetric | ΔE₀₀ against a reference chart | mean ≤ 2.0, max ≤ 4.0 |
| **T4** | Perceptual | forced-choice A/B, trained observer | correct identification ≤ 65 % (chance 50 %) |

T4 is the only test a client will run. It is passed by satisfying T1–T3 **and**
reproducing grain morphology — which is why noise-power and grain-structure data
are not refinements but requirements.

## 1.2 The uncertainty budget — reference data must be better than the target

To claim mean |ΔD| ≤ 0.02, the reference data must be substantially better than
0.02, or the claim is unfalsifiable. A defensible allocation:

| Contributor | Allowable |
|---|---|
| Reference densitometry repeatability | ≤ 0.005 D |
| Curve digitisation / transcription error | ≤ 0.005 D |
| Emulsion batch-to-batch variation | ≤ 0.01 D (irreducible; must be *stated*, not assumed away) |
| Scanner characterisation residual | ≤ 0.005 D |
| **Model error budget remaining** | **≈ 0.015 D** |

Two consequences that shape the whole acquisition strategy:

* **Batch variation is often the largest term.** Speed drift of ~⅓ stop and
  gradient drift of ~5 % between coatings is normal. A model fitted to one
  coating cannot be validated against another to better than that. Every dataset
  must carry its batch identity, and any accuracy claim must state which coating
  it refers to.
* **Digitising a printed plot costs about the same as the densitometer's own
  repeatability.** Machine tracing of a published curve at ~0.005 D is
  achievable; visual reading at 3–5 points is not, by an order of magnitude.

## 1.3 The one ceiling no film data can lift

Film integrates a **spectrum**: the exposure of a layer is ∫S(λ)·E(λ)dλ. A
conventional digital image holds three numbers per pixel that were already
integrated through a *different* set of spectral responses. Subjects metameric to
the capture device are generally **not** metameric to the emulsion. That
distinguishing information was destroyed before the simulator ran.

| Input | Achievable T3 |
|---|---|
| Display-encoded RGB | ~70–75 % |
| Scene-referred linear, declared colorimetry | ~80–85 % |
| Scene-referred linear + capture spectral sensitivities, or multispectral capture | **90–95 % reachable** |

**No metric in this document lifts this.** It is an input-contract requirement and
must appear in any commitment made to a client.

---

# 2. The organising principle: measure along the causal chain

Film is not a look; it is a sequence of physical processes. The data model should
mirror that sequence, because each stage consumes the output of the previous one
and no stage can be characterised in isolation.

```
  [1] SCENE RADIANCE, spectral
        |
  [2] OPTICAL TRANSPORT within the layer stack
        scatter, internal reflection, inter-layer coupling
        |
  [3] PHOTONIC ABSORPTION per layer
        spectral sensitivity
        |
  [4] LATENT IMAGE FORMATION
        statistical, intensity- and time-dependent
        |
  [5] DEVELOPMENT
        chemical amplification: nonlinear, spatially coupled,
        inter-layer coupled, condition-dependent
        |
  [6] IMAGE-FORMING MATERIAL
        dye or silver: spectral, granular, discrete
        |
  [7] MATERIAL HISTORY
        transport, handling, aging, damage
        |
  [8] MEASUREMENT
        scanner or printer: spectral, spatial, noisy
```

Two structural consequences:

* **Stage order in the simulator is not a design choice.** Optical transport
  precedes absorption; grain is created at development, so it is *not* blurred by
  the optics that preceded it; inter-layer chemical coupling depends on developed
  density, so it must follow the transfer function, not precede it. A model that
  orders these differently cannot be corrected by tuning.
* **Stage 8 is part of the system under test.** The reference is a *scan*, not
  the film. Unless the scanner's own transfer function, noise and spectral
  responsivities are measured and removed, the simulator is fitted to the wrong
  system.

---

# 3. The data is a cube, not a list of numbers

The single most consequential modelling decision is recognising that a material's
behaviour is a **multidimensional dataset** indexed by at least:

```
  processing state  x  exposure level  x  wavelength  x  spatial frequency  x  layer
```

Almost everything a manufacturer publishes is a **one-dimensional slice** through
that cube, taken at fixed values of the other axes. Treating those slices as
independent scalars is the fundamental error to avoid.

| Published item | Which slice it is | What the slice conceals |
|---|---|---|
| One characteristic curve | fixed processing, fixed λ-integration | the entire processing axis |
| One granularity value | fixed processing, fixed density, fixed aperture | density and processing dependence, and the whole spatial-frequency axis |
| One resolving-power figure | fixed processing, fixed contrast | the frequency response either side of the limit |
| Spectral sensitivity curve | fixed processing, fixed criterion density | how it shifts with development degree |
| An MTF curve | fixed processing, fixed exposure | exposure dependence of adjacency effects |

**Minimum viable sampling of the cube for the target:**

| Axis | Minimum | Preferred | Why more is not needed |
|---|---|---|---|
| Processing state | 3 | 8–12 (2–4 developers × 3–5 times) | Chemistry is discrete; interpolate in time only |
| Exposure level | 256 pts | 512 pts over 6 decades | 0.012 decade spacing is finer than any printed plot's line width |
| Wavelength | 10 nm / 400–700 | 5 nm / 360–730 | Matches published plot resolution; finer buys nothing |
| Spatial frequency | 20 pts | 32–64 log-spaced, 1–200 cy/mm | Beyond the detectability limit is noise |
| Layer | all sensitised layers | + interlayers and overcoat | Depth drives halation and coupling |

Order of magnitude per material, fully specified: **~10⁴–10⁵ numbers.** Current
practice in most simulation products is 10¹–10². That gap is the subject of this
document.

---

# 4. Required measurements, by causal stage

Each entry gives: the physical quantity → why fidelity depends on it →
representation and sampling → who can supply it → implementation cost → how it is
validated. Priorities: **[E]** essential, **[H]** high, **[A]** advanced,
**[O]** optional.

---

## Stage 1–2 · Optical transport: support, stack, scatter

### 4.1 Support optical and physical properties **[E]**

Support material, **thickness** (60–200 µm), **refractive index** (1.45–1.60),
spectral transmittance of the processed unexposed support, anti-halation
construction and whether it is removed in processing, backing layers.

*Why:* base thickness and index set the **geometry of internal reflection**, and
therefore the scale of halation. Support spectral transmittance is the density
floor beneath every measurement and carries any base colour.

*Representation:* scalars + one **spectral curve** at 10 nm, 380–730 nm.
*Source:* manufacturer documentation; spectrophotometry of a processed blank.
*Cost:* low. *Validation:* per-channel D-min against a processed blank.

### 4.2 Layer stack geometry **[H]**

Ordered list of all layers with **depth below surface** and **thickness** (1–30 µm
each, 10–30 µm total), plus a scattering coefficient per layer.

*Why:* each sensitised layer sits at a different distance from the reflecting rear
surface, which is the physical origin of **halation asymmetry between records**.
Depth also sets the geometry of inter-layer diffusion. Modelling from geometry
replaces several fitted per-channel constants with a few measured ones, and
therefore extrapolates to conditions that were never measured.

*Representation:* ordered records (small multidimensional dataset).
*Source:* manufacturer documentation for total thickness (commonly published);
cross-section microscopy for the full stack.
*Cost:* medium. *Validation:* per-channel halation profiles and per-channel MTF.

### 4.3 Halation point spread function **[H]**

Radial profile of light re-entering the emulsion after internal reflection,
**per layer**.

*Why:* highlight rendering is a signature of the medium — the glow around bright
sources, and its colour asymmetry. A model with the wrong halation is recognised
immediately even when the tone scale is perfect.

*Representation:* **sampled radial profile**, 20–40 radii, log-spaced,
5 µm–2 mm; or a small set of lobes whose radii are *derived* from 4.1 and 4.2.
Amplitude as a fraction of incident, 0–0.2.
*Source:* rarely published beyond the anti-halation construction. Measure by
exposing a small bright aperture against a dark surround and profiling the halo
microdensitometrically.
*Cost:* medium. *Validation:* halo profile per channel; and confirm a flat field
passes through unchanged, since the model must **redistribute** energy, not
create it.

### 4.4 Emulsion turbidity and inter-layer optical crosstalk **[A]**

Scattering coefficient of the emulsion; fraction of light scattered from one
layer's path into another's absorption.

*Why:* turbidity relates diffuse to specular density (the Callier factor) and
sets the optical component of the MTF. Inter-layer scatter desaturates and softens
per-record images in a spatially extended way, distinct from the pointwise dye
crosstalk of 4.13.

*Representation:* scalar coefficient; crosstalk as a **matrix of PSFs**
(layer × layer), fraction 0–0.1.
*Source:* laboratory measurement — narrow-band exposure of one layer, densitometry
of the others. Effectively never published.
*Cost:* high. *Validation:* narrow-band spot exposure, cross-record halo.

---

## Stage 3 · Photonic absorption: spectral response

### 4.5 Spectral sensitivity per layer **[E]**

Relative or absolute sensitivity of each sensitised layer versus wavelength.

*Why:* this is the sole determinant of how a **spectral** scene maps onto per-layer
exposures. It governs the rendering of saturated colours, narrow-band sources and
non-blackbody illuminants; it is the mechanism behind orthochromatic rendering,
extended-red response, and the entire difference between film colour and sensor
colour. **Three broadband scalars cannot substitute** — they can only form a fixed
linear combination of the input's three channels, exact for neutrals and
progressively wrong as saturation rises, and incapable of representing a
narrow-band source at all.

*Representation:* **spectral curve per layer**, log relative sensitivity vs λ.
*Sampling:* 5 nm over 360–730 nm (75 pts) preferred; 10 nm over 400–700 (31 pts)
minimum. Extend to 1100 nm for extended-red or infrared materials.
*Must accompany:* the exposing illuminant (measured SPD or CIE designation) and
the criterion density defining the sensitivity.
*Source:* manufacturer documentation — universally published as a wedge
spectrogram. Otherwise monochromator sensitometry.
*Cost:* medium — the integration is easy; giving the *input* spectral content is
the work (§1.3).
*Validation:* expose through narrow-band and sharp-cut filters, compare per-layer
density; then T3 on a chart containing saturated primaries, where a scalar-weight
model fails visibly.

### 4.6 Input spectral reconstruction **[E — pipeline, not vendor data]**

Three approaches, increasing in cost and accuracy:

1. **Illuminant-conditioned integration** — precompute the mapping by integrating
   layer sensitivities against a chosen illuminant and the input's assumed
   primaries. Low cost. Correct for neutrals; still wrong for saturated colours;
   but it *uses* the spectral data rather than discarding it.
2. **Spectral upsampling** of RGB to a plausible smooth reflectance, then genuine
   integration. Medium cost. Materially better for saturated colour; must be
   documented as an estimate, because it is one.
3. **Capture-SSF-aware path** — the user supplies the camera's spectral
   sensitivities; the scene is re-integrated through the film's. High cost, and a
   product decision because it demands data from the user.

*This is the gate on T3 above ~75 %.*

---

## Stage 4 · Latent image formation: exposure-response behaviour

### 4.7 Reciprocity-law failure **[E where long exposures occur]**

Departure from equal density for equal exposure as exposure time varies over
decades, **per layer**.

*Why:* latent-image formation needs multiple photons per crystal within a
lifetime; at low intensity, sub-latent-image sites regress before completion. A
long exposure is therefore not equivalent to a short one at the same nominal
exposure. In multi-layer materials the failure **differs by layer**, producing a
colour crossover that no shared correction can express.

*Representation:* **sampled correction curve per layer** — compensation vs time.
Preferred over a single exponent, which frequently cannot fit published behaviour
across the whole range.
*Sampling:* 8–12 log-spaced points, 10⁻⁴ s to 10³ s, plus the onset time below
which no correction applies.
*Requires from the simulator:* an **exposure-time input**. This is usually the
blocker, not the data.
*Source:* manufacturer documentation (commonly a table or graph); or a
time-scale sensitometric series at matched exposure.
*Cost:* low. *Validation:* matched-exposure pairs decades apart, compared for
density **and** colour difference.

### 4.8 Intermittency and shutter-profile effects **[O]**

Density difference between one continuous exposure and the same total delivered as
a pulse train, relevant to motion-picture capture and multiple-flash work.
Scalar, or a function of pulse frequency and duty cycle. Correction typically
<0.3 stop. Rarely published; laboratory sensitometry. Cost medium (needs an
exposure-timing model to be meaningful).

### 4.9 Latent-image keeping **[O]**

Density change versus delay between exposure and development, at stated
temperature and humidity. **2-D table**, 4–6 delays × 3 temperatures. Manufacturer
or archival keeping tests. Negligible for normal workflows.

---

## Stage 5 · Development: the axis most often omitted

### 4.10 Processing state definition **[E]**

Developer identity and composition class, dilution, time, temperature, agitation
regime, replenishment/seasoning state, and subsequent steps (stop, fix, bleach,
reversal re-exposure, stabilisation).

*Why:* this is the independent variable of the entire sensitometric and granularity
data set. **Without it, a delivered curve is unreproducible** and cannot be
compared against a laboratory's own control readings.

*Representation:* structured metadata, one record per state.
*Cost:* low as metadata. *Validation:* reproduce a laboratory's control-strip
readings.

### 4.11 Joint dependence of curve, granularity and MTF on processing **[E]**

How the transfer function, maximum density, granularity **and** spatial response
vary *together* across processing states.

*Why:* development is a diffusion-limited chemical amplification. Longer or more
active development raises gradient, raises granularity, and typically **lowers**
MTF through increased lateral diffusion. **These are not independent controls.** A
simulator exposing "contrast" decoupled from grain and sharpness is physically
wrong, and the error is obvious to anyone who has processed film. This is the
largest structural gap in any fixed-state model.

*Representation:* a **bundle indexed by processing state** — {curve set per layer,
granularity curve, MTF, D-max} — plus a **time–gradient function** Ḡ(t) per
developer for interpolation *within* a developer. Interpolate in time; **never
across developers**, since different chemistry is not a continuum.
*Sampling:* 3–5 times per developer, 2–4 developers, 2–4 temperatures.
*Source:* manufacturer documentation — curve families and time–gradient plots are
commonly published, and are usually the single richest thing on a datasheet.
*Cost:* high — schema restructuring, a control surface, and every dependent
quantity becomes state-indexed.
*Validation:* T1 **and** T2 at ≥2 processing states simultaneously, demonstrating
that gradient, grain and MTF all track measurement at once.

### 4.12 Temperature and chemical-activity dependencies **[H]**

Activity versus temperature (thermally activated, Arrhenius-like); sensitivity to
dilution, pH and halide-ion concentration; capacity drift with accumulated
throughput.

*Why:* permits interpolation between published states instead of demanding a
measurement for each, and models real process drift.
*Representation:* coefficients or a **sampled function** of temperature, 3–5
points, 15–40 °C; drift as a **table** of time extension versus throughput.
*Source:* manufacturer time–temperature charts (very commonly published);
laboratory process-control records for drift.
*Cost:* medium. *Validation:* predict a state withheld from fitting and compare
against its measured curve.

### 4.13 Inter-layer (interimage) chemical coupling **[E for colour]**

The change in one layer's effective response caused by development activity in
the others.

*Why:* it raises colour separation **without** raising gradient. On a neutral the
correction cancels, so the grey scale is untouched while saturated colours
separate further. No per-channel curve and no saturation control reproduces that
combination — it is one of the few effects that unambiguously distinguishes a
physical model from a look emulation.

*Representation:* **matrix of functions** — donor × receiver, each entry a function
of donor density rather than a scalar; plus a density-weighting exponent.
*Sampling:* the matrix at 3–5 density levels.
*Source:* derivable from paired sensitometry — density in each layer from a
**single-layer (separation) exposure** versus a **neutral exposure**; the
difference isolates the term. Occasionally published as percentages.
*Cost:* medium; the density dependence and the implicit solve are the work,
since effective exposure depends on density which depends on effective exposure.
*Validation:* separation versus neutral exposures at matched levels.

### 4.14 Lateral inhibitor diffusion **[H]**

Spatial extent and strength of within-layer development inhibition.

*Why:* produces edge enhancement beyond the optical MTF — which is why film can
appear sharper than its MTF alone predicts — and local colour interaction at
boundaries.
*Representation:* **sampled radial kernel**, 10–20 radii over 0–100 µm; or
diffusion length (5–60 µm) plus strength (0–0.3).
*Source:* edge-trace microdensitometry (4.15). Rarely published directly.
*Cost:* low given the data. *Validation:* edge-trace comparison.

### 4.15 Edge response / acutance trace **[H]**

Density profile across a sharp exposure boundary.

*Why:* acutance is perceived sharpness, which is **not** MTF. Chemical development
can produce **overshoot** — density higher than the plateau just inside a light
edge — which cannot arise from any purely optical blur model.
*Representation:* **sampled edge trace**, 20–50 points at 2–5 µm spacing, ±100 µm.
*Source:* microdensitometry of a knife-edge exposure. Rarely published.
*Cost:* low. *Validation:* direct trace comparison. Note that a measured
low-frequency MTF above unity is meaningful here, not an error, and must not be
silently clipped.

### 4.16 Development nonlinearity and local exhaustion **[H]**

Departure from proportionality approaching maximum density; local developer
depletion; the dependence of adjacency effects on development degree and local
density.

*Why:* shapes highlight compression (or shadow compression in reversal materials)
and makes edge effects exposure-dependent rather than fixed.
*Representation:* implicit in a curve family that extends **fully into D-max**;
otherwise an explicit saturation function plus exposure-dependent adjacency
scaling.
*Critical acquisition note:* **published curves are frequently truncated** at the
plotted range. Request untruncated data; the top density decade is where
truncation hides model error.
*Cost:* medium. *Validation:* T1 restricted to the top density decade.

---

## Stage 6a · Image-forming material: transfer function

### 4.17 Characteristic (H&D) curve family **[E]**

Developed density versus log exposure, per layer, for each processing state.

*Why:* the dominant determinant of tonal rendering — black level, shadow
separation, mid-tone contrast, highlight rolloff. The property a colourist judges
first.

*Representation:* **sampled monotonic curve**, one per layer per processing state.
**Not an analytic fit.**
*Sampling:* **256–512 points uniform in log H over [−4.0, +2.0]** → 0.012–0.023
decade spacing, finer than the line width of any printed plot, so the table
becomes limited by the source rather than by the model.
*Why not a few parameters:* an analytic form reproduces the gross shape with a
handful of numbers but leaves a **systematic residual concentrated in the toe and
upper shoulder** — precisely the regions carrying shadow separation and highlight
rolloff, and precisely where the eye goes. At the T1 tolerance of 0.02 D this is
not an acceptable simplification. Monotonicity must be enforced on ingest; a
non-monotonic shoulder solarises every highlight.
*Must accompany:* D-min (base+fog), D-max, average gradient **with its defining
log-H interval**, exposure latitude, and the useful log-H range — these are the
acceptance criteria and regression checks, not render inputs.
*Source:* manufacturer documentation, published per developer and time;
otherwise sensitometry of a calibrated step wedge.
*Cost:* medium — a data-path change, not an algorithmic one, if the renderer
already tabulates the curve.
*Validation:* **T1 directly**, point by point per channel.

### 4.18 Speed and its criterion **[E]**

Exposure required to reach a defined density above base+fog.

*Why:* anchors the whole exposure scale. A mis-anchoring of ⅓ stop fails T1
everywhere even with a perfect curve shape.
*Representation:* scalar + criterion enum + the defining density offset + separate
values per illuminant class where published. "Rated speed", "exposure index" and
"recommended meter setting" are three different numbers and must not be conflated.
*Cost:* low. *Validation:* simulate an 18 % reflectance patch at rated exposure;
compare density.

### 4.19 Masking-coupler density **[E for masked negative materials]**

Density contributed by integral masking couplers, per record and spectrally.

*Why:* sets the overall minimum density and the entire print exposure balance.
**Whether published minimum densities include the mask must be stated** —
ambiguity shifts the whole tone scale of a print simulation.
*Representation:* per-record scalars (0.2–1.2) plus the mask's **spectral** density.
*Cost:* low. *Validation:* D-min per record; print neutrality.

---

## Stage 6b · Image-forming material: spectral character

### 4.20 Spectral density of the image-forming material **[E for colour, H for monochrome]**

Spectral optical density of each image dye, and of developed silver, at stated
concentration.

*Why:* what a scanner, printer or projector records is the **integral of image
transmittance against its own spectral responsivity**. A 3×3 crosstalk matrix is
exact for exactly one illuminant/observer pair; with spectral data the same
material description stays correct under a different scanner or print light. For
monochrome, silver's non-neutral transmittance is why monochrome prints are rarely
truly neutral.

*Representation:* **matrix of spectral curves** — one per dye plus base — ideally
each at 2–3 concentrations, because absorption shape is not perfectly proportional
to concentration.
*Sampling:* 5–10 nm over 380–730 nm, plus peak density per dye.
*Source:* manufacturer documentation (commonly published as spectral dye density
plots); or spectrophotometry of processed single-dye wedges.
*Cost:* high — new structure, spectral integration, and a scanner/observer model
to integrate against.
*Validation:* T3; and as a stronger test, simulate the same material through two
different scanner responsivities and confirm the predicted difference matches
measurement — something a matrix model cannot do.

---

## Stage 6c · Image-forming material: granular structure

### 4.21 Granularity versus density **[E]**

Standard deviation of density in uniformly exposed, uniformly processed areas, at
several density levels.

*Why:* the image is a countable population of discrete developed grains, so
density is a random variable and fluctuation scales with the square root of the
number of grains in the aperture. This sets grain amplitude **and its tonal
placement**; grain in the wrong tonal zone is among the most recognisable
simulation failures.

*Representation:* **sampled curve** σ(D) versus D.
*Sampling:* ≥3 densities (toe / mid / high), 5–10 preferred from 0.2 to D-max. A
single value forces the simulator to assume a dependence — usually a square-root
law, which departs from measurement at both ends.
*Mandatory conditions:* **scanning aperture diameter**, sample density relative to
base+fog, magnification, geometry, processing state. **Granularity is a property
of the film *and the aperture*, not of the film alone** — values scale roughly as
the inverse of aperture diameter, so a factor-of-four discrepancy between two
sources is normal and is not a contradiction. Ingesting a value without its
aperture guarantees a wrong amplitude.
*Source:* manufacturer documentation (usually one value); microdensitometry of
uniform patches for the curve.
*Cost:* low. *Validation:* σ(D) in flat regions of output at matched density and
matched aperture, ±5 %.

### 4.22 Noise power (Wiener) spectrum **[H — required to exceed ~90 %]**

Power spectral density of the density fluctuation versus spatial frequency.

*Why:* **RMS granularity fixes only the total noise power, not its distribution
across scale.** Two materials with identical RMS and identical mean grain size can
look entirely different. This is the metric that distinguishes emulsion families a
filtered-Gaussian noise model renders identically, and it is a principal
contributor to T4 — the test a client will actually run.

*Representation:* **sampled function** W(f).
*Sampling:* 32–64 log-spaced points, 1–200 cycles/mm; state the density and
processing state.
*Source:* **expect to measure this.** Rarely published. Microdensitometer or
high-resolution scan of large uniform patches, Fourier analysis with the
scanner's own noise and MTF removed.
*Cost:* medium if the noise generator band-limits with a small set of shaping
lobes that can be fitted to W(f); high for general spectral shaping.
*Validation:* spectrum of simulated flat patches versus measured, shape ±10 %.

### 4.23 Grain size distribution, morphology and cross-channel correlation **[A / H]**

Distribution of developed grain or dye-cloud sizes; clustering scale; crystal
habit; **dye-cloud diameter relative to the nucleating grain**; correlation of
noise between records.

*Why:* grains are polydisperse and spatially clustered; habit (regular versus
tabular) changes how they pack and therefore the texture at equal mean size. In
colour materials the developed **dye cloud is substantially larger** than the
silver grain that nucleated it, which is why colour grain looks softer than
monochrome grain of equal crystal size. Noise between records is partially
correlated, because a scanner reads all dyes through overlapping bands —
independent per-record noise produces coloured speckle that real film does not
show the same way.

*Representation:* **probability distribution** (log-normal parameter, or a
**histogram** of 10–20 bins) + scalars for clustering scale and dye-cloud
diameter (2–20 µm) + a **3×3 correlation matrix** between records.
*Source:* electron microscopy for crystals; image analysis of high-resolution
scans for developed grain and for correlation.
*Cost:* low for dye-cloud size and correlation — both cheap and visible.
**Very high** for a true discrete-clump point-process generator reproducing
non-Gaussian marginals, which is the physically honest form.
*Validation:* cross-channel correlation and higher moments (skewness, kurtosis) of
simulated versus scanned flat patches. Note that a filtered-Gaussian field has
Gaussian marginals by construction and will fail a kurtosis test against real
grain regardless of how well its variance and spectrum are matched — that
mismatch is the honest measure of how far a Gaussian model can go.

### 4.24 Silver halide composition and coating weight **[A]**

Halide composition, crystal habit and aspect ratio, silver laydown.

*Why:* habit determines packing and clustering, hence noise-spectrum shape;
coating weight bounds achievable maximum density.
*Representation:* enums + scalars (coating weight 1–8 g/m²; aspect ratio 1:1 to
20:1).
*Source:* manufacturer documentation (partial, often qualitative); electron
microscopy for quantitative data.
*Cost:* medium as parameters; very high for an explicit crystal model.

---

## Stage 6d · Image-forming material: spatial response

### 4.25 Modulation transfer function **[E]**

Modulation transfer versus spatial frequency, per layer.

*Why:* governs sharpness and, jointly with 4.22, the texture of fine detail. A
real MTF has a **heavier tail** than a Gaussian of the same half-power frequency:
modelling it as Gaussian underestimates response above roughly 1.5·f₅₀, producing
texture too soft at fine scales even when overall sharpness appears matched — a
T2 failure invisible to any single-number check.

*Representation:* **sampled function** per layer. A two-term parametric form (a
core plus a heavier-tailed term) is an acceptable *fit* to the curve; the curve is
the requirement.
*Sampling:* 20–40 points, 1–200 cycles/mm. State processing state and method.
*Simulation placement:* pre-development spatial filter, so it blurs the image but
**not** the grain, which is created later; it also band-limits the grain field to
prevent aliasing.
*Source:* manufacturer documentation where published; otherwise sinusoidal-target
or slanted-edge measurement with the optical system's own MTF removed.
*Cost:* low–medium. *Validation:* MTF of simulated output on the same target,
±3 % to 1.5·f₅₀.

### 4.26 Resolving power **[H — as a constraint, never a substitute]**

Highest resolvable frequency at stated test-object contrast.

*Why:* it is a **threshold-detection** limit combining MTF *and* granularity, not a
pure optical quantity. Where no MTF curve exists it **constrains** an inferred
MTF: modulation should fall to the detectability threshold at the published limit.
*Representation:* scalar pairs (40–300 lines/mm) with contrast, target type and
processing stated. Values from different conventions are not comparable.
*Cost:* low. *Validation:* simulate the same target; confirm the limit matches.

---

## Stage 7 · Material history

### 4.27 Coating and processing non-uniformity **[H]**

Large-scale density variation: coating streaks along the web, cross-web variation,
mottle, processing unevenness.

*Why:* it reads as "real material" rather than "synthetic texture", and is one of
the strongest cues distinguishing older or lower-tolerance materials. Grain alone
does not supply it.
*Representation:* **spatial statistics** — RMS amplitude (0.005–0.05 D) plus a
**2-D power spectrum** (16–32 bins, 0.01–5 cycles/mm) or correlation lengths along
and across the web; optionally a measured **spatial map** for archival
reproduction.
*Critical:* the field must be **fixed to the material**, oriented to the web axis,
so it moves with the film rather than with the frame.
*Source:* densitometric scanning of processed blank and uniformly exposed samples;
manufacturer tolerance specifications.
*Cost:* low–medium. *Validation:* low-frequency power spectra, simulated versus
scanned.

### 4.28 Format edge effects **[A]**

Density rise near the physical edges of the material from light piping and
edge-region processing differences. **Sampled profile**, 10–20 points over
0–10 mm, 0–0.3 D. Prominent in narrow-gauge materials where the image area sits
close to the edge. Densitometry across the full width. Cost low.

### 4.29 Physical defect population statistics **[O / E if archival reproduction is in scope]**

Areal rate and size distribution of dust, embedded debris, fibres and scratches,
with optical character (opaque, refractive, emulsion-side versus base-side).

*Representation:* **probability distributions** — areal rate (0.001–5 /mm²), size
(histogram or log-normal, 5 µm–2 mm), opacity; orientation statistics for
scratches; and a **spatial point process with clustering**, since contamination is
not uniformly random.
*Modelling requirement:* distinguish defects **embedded in the material** — which
then pass through every subsequent duplication and printing stage, and are
therefore softened and tone-mapped like the image — from defects added **at
projection or scanning**, which are not. Getting this distinction wrong is the
commonest reason simulated dirt reads as an overlay.
*Source:* image analysis of scanned archival material; laboratory cleanliness
specifications.
*Cost:* medium. *Validation:* count, size and opacity distributions compared
**statistically**, never pixel-wise.

### 4.30 Aging and fade kinetics **[O]**

Dye density loss per dye, base yellowing, minimum-density rise, dimensional
change, as functions of time, temperature, humidity and cumulative illumination.
**Rate constants** for an Arrhenius-type model, or **sampled fade curves**, 5–10
time points across ≥3 conditions. Accelerated-aging literature; measurement of
dated archival samples. Relevant only when reproducing the *current condition* of
aged material rather than its as-processed state.

---

## Stage 7b · Temporal behaviour (motion-picture materials)

### 4.31 Grain temporal correlation **[E for motion picture]**

Correlation of the grain pattern between successive frames.

*Why:* **grain is fixed in the material.** One physical frame carries one fixed
pattern; successive frames of a moving strip carry uncorrelated patterns; and
scanning the same frame twice must reproduce the same grain. Grain that
re-randomises every frame reads as electronic noise — this is among the strongest
perceptual cues in motion imagery and one of the most commonly wrong.
*Representation:* correlation coefficient plus an explicit statement of the
mechanism (per-frame identity versus per-material coordinate).
*Implementation requirement:* the noise generator must be a **deterministic
function of material coordinate and frame identity**, so re-rendering any frame
reproduces it exactly and neighbouring frames differ appropriately. This also
makes the render order-independent, which matters for tiled and multi-threaded
rendering.
*Source:* frame-pair analysis of scanned motion-picture material.
*Cost:* low. *Validation:* cross-correlate grain between consecutive simulated
frames; compare to the same statistic on a real scan.

### 4.32 Transport-induced registration variation **[H for motion picture]**

Frame-to-frame positional variation in the gate and its temporal spectrum.
**RMS amplitudes** across and along the web (2–40 µm) plus a **temporal PSD**
(16–32 bins, 0.1 Hz to half frame rate).

*Critical:* the displacement is physical; **any softening introduced by the
resampling used to implement it is an artefact.** A translation must not blur.
This demands an interpolation kernel that preserves high frequencies — a two-tap
kernel loses roughly half the modulation at Nyquist for a half-pixel shift, which
is a large and entirely spurious sharpness loss applied to every frame.
*Source:* motion analysis of scanned material against a fixed reference; or
equipment specifications.
*Cost:* low for the displacement; the requirement is an adequate resampler.
*Validation:* displacement statistics of output; **and confirm no measurable MTF
loss attributable to the resampling.**

### 4.33 Frame-to-frame exposure variation **[A]**

Amplitude (0–10 % exposure) and temporal spectrum of inter-frame exposure
fluctuation from shutter and transport irregularity.
*Placement:* applied **before** the transfer function, so it moves highlights
through the shoulder as a real exposure change does. Applied afterwards it is a
grade, not an exposure — a distinction that is visible in the highlights.
*Cost:* low. *Validation:* frame-mean time series.

### 4.34 Frame geometry and native rate **[E]**

Frame and perforation geometry, image area within the frame, intended frame rate.

*Why:* frame geometry sets the magnification converting micrometre-scale film
properties into pixel-scale image properties — the mechanism by which **one
material description renders correctly at any output resolution**. All temporal
spectra are meaningless without the rate.
*Cost:* low. *Validation:* render at two output resolutions and confirm physical
feature sizes are invariant.

---

## Stage 8 · Measurement chain

### 4.35 Densitometric convention **[E — and a correctness requirement, not a fidelity one]**

For **every** density value delivered: spectral condition (diffuse visual,
printing, one of the standard spectral conditions, or an arbitrary spectral
product), geometry (diffuse / specular / doubly diffuse), and the **Callier
factor** if specular (typically 1.0–1.5, wavelength- and grain-dependent).

*Why this is different in kind from the other items:* a density measured under one
spectral condition is a **different physical quantity** from one measured under
another. They are not interchangeable, and mixing them produces silent systematic
error — the arithmetic remains valid and only the semantics are wrong, so the
fault survives code review and appears only as an unexplained mismatch against a
scan. A model with no consumer for this metadata **cannot detect that it is
comparing incommensurable quantities.**
*Implementation:* store it, and **validate on ingest** — refuse or flag any
dataset with an unstated convention. Convert to a single canonical convention on
load where the conversion is defined.
*Cost:* low. This is the cheapest high-value item in the entire document.

### 4.36 Scanner characterisation **[E]**

The scanner's own MTF, noise floor, spectral responsivities, geometric distortion,
flare and bit depth.

*Why:* **the reference is a scan, not the film.** Unmeasured, the scanner's
signature is attributed to the material and the simulator is fitted to the wrong
system. A scanner whose MTF is unknown makes 4.25 unverifiable; unknown spectral
responsivities make 4.20 unverifiable; an unknown noise floor makes 4.21 and 4.22
unverifiable.
*Representation:* the same forms as 4.25, 4.20, 4.21 — the scanner is
characterised exactly as the film is.
*Cost:* medium. *Validation:* it *is* the validation infrastructure.

### 4.37 Sample provenance and batch identity **[E]**

Coating identity and date, production versus pilot coat, storage history.

*Why:* inter-coating variation of ~⅓ stop in speed and ~5 % in gradient is normal
(§1.2). Any accuracy claim is a claim about a specific coating.
*Cost:* low.

---

# 5. Interactions — why none of the above is independent

A data model storing these as unrelated values will produce internally
inconsistent results even when every individual number is correct.

| Coupling | Physical mechanism | Failure if ignored |
|---|---|---|
| **Processing ↔ curve, granularity, MTF** (4.11) | Development simultaneously raises gradient, raises grain, lowers MTF | "Contrast" that does not move grain and sharpness — obvious to any practitioner |
| **Granularity ↔ density ↔ curve** (4.21, 4.17) | σ depends on developed density, set by the curve | Grain in the wrong tonal zone |
| **Granularity ↔ measuring aperture** (4.21) | σ is film **and** aperture | Amplitude error of several times, silently |
| **Noise spectrum ↔ size distribution ↔ habit** (4.22–4.24) | Spectrum is set by size spread, clustering, crystal form | Right total noise, wrong texture; families render identically |
| **MTF ↔ lateral diffusion ↔ edge trace** (4.25, 4.14, 4.15) | A measured edge is optical **plus** chemical | Fitting one from the other double-counts or cancels; low-frequency MTF above unity is real, not an error |
| **MTF ↔ grain** (4.25, 4.22) | The optical response band-limits grain as well as image | Grain aliasing, or grain sharper than the detail beneath it |
| **Spectral sensitivity ↔ dye density ↔ scanner** (4.5, 4.20, 4.36) | Recorded density is a triple spectral integral | A matrix fitted for one scanner is wrong for another |
| **Halation ↔ base geometry ↔ layer depth** (4.3, 4.1, 4.2) | Halo scale is thickness, index and depth | Per-channel gains become free parameters that do not extrapolate |
| **Inter-layer coupling ↔ density ↔ exposure** (4.13) | Coupling depends on donor density | Saturation error that varies with exposure — reads as inconsistent colour |
| **Reciprocity ↔ layer** (4.7) | Failure differs per layer | Long-exposure colour crossover unmodellable |
| **Adjacency ↔ development degree ↔ local density** (4.16, 4.14) | Edge effects grow with development and local density | Fixed edge effects where reality has variable ones |
| **Grain fixed to material ↔ material moving in gate** (4.31, 4.32) | Two independent facts | Grain that both re-randomises and shifts — two compounding errors |
| **Non-uniformity ↔ material coordinates** (4.27, 4.34) | Coating structure is fixed to the web | Non-uniformity that swims relative to the film |
| **Density convention ↔ every density** (4.35) | Convention defines what a density *is* | Incommensurable quantities compared as equals |

---

# 6. Reference measurement set — what must be shot, processed and scanned

Data enables accuracy; only measurement demonstrates it. This set is required for
every material carrying a 90–95 % claim.

| # | Target | Yields | Test |
|---|---|---|---|
| 1 | Calibrated step wedge, ≥21 steps, ≥4 density units, **exposed to D-max** | curve, D-min, D-max, gradient, latitude | T1 |
| 2 | Reference colour chart, stated illuminant | colorimetric error | T3 |
| 3 | Sinusoidal **and** slanted-edge spatial targets | MTF | T2 |
| 4 | Large uniform patches, ≥3 densities | granularity vs density, noise power spectrum, grain morphology, cross-channel correlation | T2, T4 |
| 5 | Knife-edge exposure | edge trace, lateral diffusion | T2 |
| 6 | Small bright aperture on dark surround | halation PSF per layer | T2 |
| 7 | Matched-exposure time series, decades apart | reciprocity per layer | T1 |
| 8 | Narrow-band / sharp-cut filtered exposures | spectral sensitivity verification | T3 |
| 9 | Processed blank, full material width | base density, mask density, non-uniformity, edge effects | T1 |
| 10 | Repeat scans of one frame; consecutive frames | grain temporal correlation; scanner noise floor | T4 |
| 11 | Items 1, 3, 4 repeated at **≥2 processing states** | the joint processing dependence of 4.11 | T1+T2 |

Item 11 is the one most often omitted and the one that most distinguishes a
physically grounded engine from a tuned one.

---

# 7. Acquisition strategy — what to request, what to expect, what to measure

## 7.1 Request from the manufacturer (usually available)

1. Characteristic curve **families** — every developer and time, **untruncated in
   density** — as data if possible, else high-resolution plots.
2. Complete processing specifications for every state in (1).
3. Time–gradient and time–temperature relationships.
4. Spectral sensitivity curves per layer, with exposing illuminant.
5. Spectral dye density curves; base and mask spectral density.
6. Granularity **with aperture, magnification, density and processing stated**.
7. MTF curves per layer, with processing stated.
8. Resolving power at stated contrasts, with target type.
9. Density convention, geometry, Callier factor, speed criterion.
10. Reciprocity compensation data, per layer.
11. Base material, thickness, refractive index; anti-halation construction and
    removal state; total emulsion thickness.
12. Coating uniformity tolerances.

## 7.2 Expect to be unavailable, and plan to measure

* Noise power spectra (4.22)
* Grain size distribution and morphology (4.23)
* Halation PSF (4.3)
* Edge traces / lateral diffusion (4.14, 4.15)
* Inter-layer coupling as a function of density (4.13)
* Full layer stack geometry (4.2)
* Inter-layer optical crosstalk (4.4)

**These are precisely the items separating 90 % from 95 %.** An in-house
measurement capability per §6 is therefore not an adjunct to this programme — it
is a precondition for its upper half. Plan for a densitometer, a resolvometer or
slanted-edge rig, a characterised scanner, and access to processing at controlled
states.

## 7.3 Handling what cannot be obtained

For any quantity neither published nor measurable, the model must:

* record it as **inferred**, with the inference method and the constraint it was
  derived from (e.g. an MTF inferred from resolving power);
* carry a **confidence class** that propagates to any accuracy claim; and
* **never** present an inferred value in the same form as a measured one.

A simulation that cannot distinguish its measured inputs from its guesses cannot
support an accuracy claim, however good its output looks.

---

# 8. Data volume and priority

Per material, fully specified per §4:

| Category | Order of magnitude |
|---|---|
| Curve families (3 layers × 8 states × 512 pts) | ~1.2 × 10⁴ |
| Spectral (sensitivity + dye density, 4 curves × 75 pts × 2) | ~600 |
| Granularity (curve × states) + noise spectra | ~10³ |
| MTF (3 layers × states × 30 pts) | ~700 |
| Halation, edge, crosstalk kernels | ~500 |
| Scalars, metadata, distributions | ~200 |
| **Total** | **~1.5 × 10⁴ numbers** |

Priority ordering, on the basis of fidelity gained per unit of effort:

| Rank | Item | Priority | Cost | New measurement needed? |
|---|---|---|---|---|
| 1 | Densitometric convention validation (4.35) | E | low | no |
| 2 | Sampled curve families + processing axis (4.17, 4.10–4.11) | E | high | no — usually published |
| 3 | Spectral sensitivity, integrated (4.5, 4.6 option 1) | E | medium | no |
| 4 | MTF as a curve incl. tail (4.25) | E | low | sometimes |
| 5 | Granularity vs density with aperture (4.21) | E | low | sometimes |
| 6 | Grain temporal correlation (4.31) | E (motion) | low | no |
| 7 | Reciprocity + exposure-time input (4.7) | E | low | no |
| 8 | Spectral dye density (4.20) | E (colour) | high | no — usually published |
| 9 | Inter-layer coupling, density-dependent (4.13) | E (colour) | medium | derivable |
| 10 | Noise power spectrum (4.22) | H | medium | **yes** |
| 11 | Dye-cloud size + cross-channel correlation (4.23) | H | low | **yes** |
| 12 | Halation PSF + base/stack geometry (4.3, 4.1, 4.2) | H | medium | partly |
| 13 | Edge trace + lateral diffusion (4.15, 4.14) | H | low | **yes** |
| 14 | Non-uniformity statistics (4.27) | H | low | **yes** |
| 15 | Grain morphology as a point process (4.23) | A | very high | **yes** |
| 16 | Aging, defects, latent-image keeping (4.30, 4.29, 4.9) | O | medium | archive |

**Ranks 1–9 constitute "90 %", and almost none of them requires a measurement
that a manufacturer has not already published.** Ranks 10–14 are the difference
between 90 % and 95 %, and nearly all of them do.

---

# 9. Current model status, in aggregate

> ⚠ **REVISED 2026-08-25. THE 2026-08-13 CENSUS BELOW IS SUPERSEDED AND MOST OF ITS
> "READ ZERO TIMES" LIST IS NO LONGER TRUE.** It was written against schema v5. The
> schema is now **v11** and the leaf-field count has gone **130 → 207** (measured by
> dataclass introspection over `film_profiles`, so it is exact). The original
> 76-read / 54-unread split is *not* restated here, and deliberately: that audit
> searched each identifier in `film_sim.py` alone, which counts a field consumed
> through a shared helper — `grain_sigma`, `mtf_response`, `sigma_anchors`,
> `spectral_balance_gains`, the generated header's inline `mtf_response` — as
> **unread**. Several of the entries below were mislabelled for exactly that
> reason. Any future census must state whether helper-mediated reads count, and
> must be re-derivable; the field count above is given because it is.

**What has since been WIRED, item by item — each of these was on the "read zero
times" list and none of them belongs there now:**

| Was listed as unread | State on 2026-08-25 |
|---|---|
| **spectral sensitivity** (4.5) | **READ.** `layer_sensitivities` exponentiates the stored log curves and `spectral_balance_gains` replaces the three-broadband-scalar proxy for any material carrying them; `spectral_monochrome_weights` does the same for monochrome. 47 materials carry curves. The first **vector-traced** set was added 2026-08-25 and its adoption moved that material's render by **+0.28 stop of red gain at 3200 K** against the proxy — i.e. the substitution this row complained about was real and is now measured, not assumed |
| **reciprocity** (4.7) | **READ, both renderers, since 2026-08-23.** `RenderSettings.exposure_time_s` / `AlgoControls::exposureTimeS`, default 0.0 = "not stated", so every earlier render is bit-identical. **21 materials** carry a measured per-channel table taken from their own sheets; the rest fall back to a fitted Schwarzschild exponent. Parity-audited between the two implementations |
| **granularity-versus-density shape** (4.21) | **READ.** `sigma_anchors` → `grain_sigma`; **13 materials** carry a traced triple (plus an interior peak where measured) behind a `sigma_shape_measured` flag, the rest keep the legacy √(D−dmin) law bit-for-bit. ⚠ And the heuristic that stood in for it was found to be **wrong in sign** for both negative and reversal materials, which is why the flag gates it |
| **the heavier-tail MTF model** (4.25) | **SUPERSEDED, not wired — and that is the finding.** The reserved two-parameter tail form scored rms 0.0583 against a traced curve while the one-parameter `1/(1+(f/f₅₀)^q)` scored **0.0375**. The tail fields are retired in place; `q` is stored and read behind an `mtf_measured` flag on **10 materials**, with 26 curves traced from 12 sheets. Both laws pass through 0.5 at f₅₀ exactly, so a measured rolloff changes shape and never level |
| **densitometric convention** (4.35) | **STILL METADATA ONLY, and still the right first fix.** `density_metric` and `speed_criterion` are consumed by the report generators, not by either renderer. ⚠ **AND THE AUDIT OF THEM TURNED UP SOMETHING WORSE THAN non-consumption:** a criterion string naming a specific density ("0.2 above D-min") is stored on **16 materials** and is printed on **no sheet in the corpus**, while **10 materials** carry a printed 0.4 — and the split follows the sheets exactly, the invented value landing on precisely those materials whose sheets state no number. A convention that is unread is a fidelity gap; a convention that is *wrong* is a correctness one |

**What remains genuinely inert, verified field by field on 2026-08-25 — this is the
current version of the list, and every entry is inert by DESIGN rather than by
neglect:**

* **spectral density of the image-forming material** (4.20) — 11 materials plus one
  print material populated, consumed by nothing. Wiring it would change every
  colour image, which is why it is carried and not connected.
* **inter-material dye-impurity ratios** — 26 measured ratios across 4 materials,
  carried verbatim because the renderer's crosstalk helper takes one saturation
  scalar and cannot express independent terms.
* **aging and fade kinetics** (4.30) — fields defined, values all zero, no consumer.
  Unchanged since 2026-08-13 and still correctly described.
* **published resolving power** (4.26) — stored as documentation, as this document
  itself recommends it should be: a constraint to check a fitted MTF against, never
  a substitute for one.
* **grain size spread, clustering scale, dye-cloud size** (4.23) — still unread.
  This row of the original list survives intact.
* **the retired MTF tail parameters** — declared, populated nowhere, kept so the
  scoring that retired them stays reconstructable.
* **processing temperature and dilution** (4.10, 4.12) — declared on both the
  processing and development-point structures, consumed by neither renderer.

**Characteristic curves remain the clearest under-use, and this paragraph stands as
written.** Source material routinely supplies a *family* across processing states,
each resolvable to several hundred points, while the model retains one curve
reduced to six analytic parameters per layer — discarding the shape residual and
the entire processing dimension. ⚠ **One correction:** the curves are no longer
hand-fitted across the board. Machine-traced families now back a substantial
minority of materials at fit residuals of **0.004–0.009 D**, which is inside the
uncertainty budget of §1.2 — so the gap is now the *processing axis* specifically,
not the per-curve accuracy.

⚠ **AND A FINDING ABOUT THIS DOCUMENT'S OWN CLASS OF CLAIM.** Between 2026-08-18 and
2026-08-25 four separate hardcoded counts in the project's report generator were
found to be wrong — by up to **2.3×** — because nothing compared them to the
database. A census in prose decays silently. Every count in this section that could
be derived now is derived, and the two that cannot (leaf fields, materials
carrying each measured quantity) are stated with the method that produced them.

---

# 10. The engineering answer

**To reach 90 %:** sampled curve families across processing states; spectral
sensitivity actually integrated; MTF as a curve; granularity as a function of
density with its aperture; per-layer reciprocity; spectral dye density;
density-dependent inter-layer coupling; grain fixed to the material. Almost all of
this data already exists in manufacturer documentation — the work is ingestion,
representation and consumption, not discovery.

**To reach 95 %:** noise power spectra, grain morphology and cross-channel
correlation, halation and edge measurements, non-uniformity statistics. This data
is largely unpublished, so the programme must include the capability to measure it
per §6.

**Two things bound the result independently of any of the above.** The input must
be scene-referred and spectrally characterised, or T3 cannot exceed ~75 % (§1.3).
And the reference scanner must itself be characterised, or the simulator is being
fitted to the scanner rather than to the film (4.36).

**And one caution on the claim itself.** "90–95 %" is meaningful only alongside
the test that produced it. Stated bare, it will be read as T4 — indistinguishable
side by side from a real scan — which is the hardest of the four tests and the one
most dependent on the unpublished data of ranks 10–15. Quote the test, the
material, the processing state and the coating batch with the number, or the
number will be understood as a stronger claim than the evidence supports.

---
---

# Appendix A – Recommended Film Database Structure

**Appendix A created: 2026-08-12, 17:11 local (IDT, UTC+3) — 2026-08-12T14:11:17Z.** *(Corrected 2026-08-13: originally stamped 2026-08-11 from a sandbox clock running a day behind.)*

## A.0 Scope, method and how to read this appendix

This appendix answers one question: **given the measurement requirements in §§1–8
above, what must a complete film-profile database contain, and how must each item
be represented?**

It is written clean-sheet. No existing database, schema, file format, profile
structure or simulation stage was consulted or used as a constraint, and nothing
here is justified by "this is close to what we have". §9 of the main document
audited an existing implementation; that audit is deliberately *not* an input to
this appendix.

⚠ **ONE SCOPE NOTE ADDED 2026-08-25, because it is an internal inconsistency a
reader will otherwise trip over.** The main document states vendor neutrality and
holds to it: no manufacturer, product family or individual material is named in
§§1–10. **This appendix does not, and cannot** — D.2 onward audits a specific
implementation and quotes its profile identifiers, several of which embed a
manufacturer and a catalogue number. That is deliberate: an audit of a real
schema cannot be conducted in the abstract. **Treat §§1–10 as the vendor-neutral
document to send outside, and Appendix D as internal.** A.1–A.5 sit in between:
they are clean-sheet and neutral in substance, but they are addressed to this
project. Where a requirement below is expensive, unavailable or awkward, it
is stated with its difficulty rather than omitted or weakened.

Two consequences of that stance are worth flagging before the detail:

* **Roughly two thirds of the content specified here is not a "field".** The
  largest single structural recommendation is that the database is primarily an
  archive of *measurement records* — each carrying its own conditions, units,
  uncertainty and provenance — and only secondarily a set of model parameters.
  A schema built as a flat list of named parameters cannot express the
  requirements of §3 (the data is a cube) or §4.35–4.37 (conditions are part of
  the number) and will lose information permanently.
* **Several items in §4 should not live in the film profile at all.** Scanner
  characterisation, input spectral reconstruction and display encoding are
  properties of the measurement and rendering chain, not of the material. They
  are required by the system, must be stored, and must be *referenced* by
  measurements — but embedding them in a film profile makes every profile
  wrong the moment the scanner is recalibrated. A.5.17 covers this.

Reading order: **A.1** gives the five decisions everything else follows from;
**A.2–A.3** define the type system and the record envelope; **A.4** gives the
object tree; **A.5** is the content catalogue, group by group, with the full
attribute set for each item; **A.6–A.7** cover derived and runtime layers;
**A.8–A.14** cover provenance, precision, completeness, validation,
extensibility and serialisation.

---

## A.1 Five structural decisions that determine everything else

### A.1.1 Three layers, only one of which is authoritative

The database must be built in three layers with a strict one-way dependency:

| Layer | Content | Mutability | Lifetime |
|---|---|---|---|
| **1. Evidence** | Measurement records exactly as measured or digitised, with conditions, units, uncertainty, provenance | Append-only; never edited in place, never overwritten | Permanent |
| **2. Model** | Representations derived from layer 1 for a specific simulation approach — fitted parameters, resampled canonical curves, decomposed kernels | Regenerable at will; deletable | Tied to an algorithm generation |
| **3. Runtime** | Packed, precision-reduced tables consumed by the engine inner loops | Build artefact | Tied to a build |

**Why this is the first decision and not a refinement.** The stated goal —
"preserve sufficient information to allow future improvements to the simulation
algorithms without requiring the original technical measurements to be
recollected" — is achievable *only* if the layer that the engine reads is not the
layer that the data lives in. The instant a fitted parameter replaces the curve it
was fitted to, every future algorithm is permanently capped by the assumptions of
the current one. A three-parameter curve fit is a lossy compression of a
hundred-point measurement, and no amount of care in choosing the three parameters
recovers what the other ninety-seven points said.

**Concretely: layer 2 must be reproducible from layer 1 by running a generator,
and layer 1 must never be reproducible from layer 2.** If a model record cannot
be deleted and regenerated byte-identically from its declared inputs, it is
holding information that belongs in layer 1 and the schema has failed.

Layer 2 records therefore carry the identity of the generator that produced them
(name, version, code revision, parameter set) and their fit diagnostics. This is
what makes a model auditable: when a rendering looks wrong, the question "is the
data wrong or is the fit wrong?" must be answerable without re-measuring anything.

### A.1.2 The measurement conditions are part of the key, not metadata

§3 established that every quantity of interest is a slice through a
multidimensional space (processing state × exposure level × wavelength × spatial
frequency × density × layer × format × batch). The schema must reflect this in its
addressing, not in its comments.

A field named `gamma` is unanswerable — gamma at which development time, measured
over which density interval, in which density metric, on which batch. Therefore:

> **A measurement is identified by the tuple (quantity, subject, condition set),
> and the condition set is a first-class structured object that participates in
> lookup, not a free-text note.**

This has a hard practical consequence: **there is no such thing as "the"
characteristic curve of a material** in this schema. There is a set of curve
records, each bound to a processing state; a request for a curve must name the
state or accept a documented default. The same holds for granularity, MTF and
every quantity in §4.11.

### A.1.3 Absence must be explicitly representable, in three distinct forms

Every condition slot and every quantity must be able to hold, distinguishably:

1. **A specified value** — measured, documented, or deliberately chosen.
2. **Explicitly unknown** — the measurement exists but the condition was not
   reported. This is the common case for historical datasheets.
3. **Not applicable** — the condition has no meaning for this subject
   (development time for a base-transmittance measurement; interimage coupling for
   a monochrome material).

A schema in which "unknown" is represented by a default value, a zero, a null, or
absence-of-key **silently converts missing data into fabricated data**, and it does
so in exactly the cases that matter most: an unstated granularity aperture, an
unstated density metric, an unstated processing state. The requirement of §4.35 —
that a density number without its convention is not usable — can only be enforced
if "convention not stated" is a value the database can hold and a validator can
reject.

### A.1.4 The same property must be able to hold several coexisting measurements

Multiple values for one property are the normal case, not an exception: two
datasheet editions, a datasheet plus an in-house measurement, three coating
batches, two laboratories, a published curve plus a digitisation of the same curve
at higher resolution.

The schema must therefore support **N records per (quantity, subject)** with:

* a `status` field (`active`, `provisional`, `superseded`, `rejected`) —
  superseded and rejected records are retained, with the reason;
* an explicit `supersedes[]` link rather than deletion;
* a **resolution policy** stored as data, not hard-coded in the loader: an
  ordered list of preference rules (prefer higher confidence class, then narrower
  uncertainty, then matching batch, then more recent, then higher sampling
  density), plus optional per-quantity overrides and per-quantity manual pins.

Averaging conflicting records is not a default the database may take on itself.
When two records disagree beyond their combined uncertainty, that is a finding
about the data — it must be surfaced, not smoothed. A "disagreement" report over
the whole archive is a required tool, not a nicety.

### A.1.5 Material identity is a three-level hierarchy, and inheritance is directional

This is the structural answer to the requirement that parameters must not silently
propagate between stocks or between formats.

```
Emulsion            chemistry, layer stack, sensitisation — the thing that responds to light
   │
   └── Coating      Emulsion × Support × coating run/batch — the thing that is actually measured
          │
          └── Product   Coating × format × perforation × winding — the thing that is sold and shot
```

Rules:

* **Every measurement record attaches to the most specific level at which it was
  actually made.** A granularity measured on 35 mm stock attaches to that Product
  if the measurement is format-dependent, to the Coating if it is not.
* **Inheritance is upward-reading only**: a Product may read a Coating's records,
  a Coating may read its Emulsion's. A Product never reads another Product's.
* **Sideways transfer requires an `EquivalenceAssertion` record** — a first-class
  object, not a comment: `{quantity_scope[], subject_a, subject_b, direction,
  evidence source_ref, confidence_class, notes}`. Documented equivalence between
  two designations, or between two gauges of one emulsion, becomes *data with a
  citation* that a validator and a report can see.
* **Format differences are represented by their physics, not by a scale factor.**
  A Product node carries its own image area, frame geometry, transport
  characteristics and any format-specific measurements. The relationship between
  gauges is that a smaller image area magnifies more for the same output, which
  affects granularity aperture-referral (Selwyn), effective MTF referred to the
  output, and defect visibility — each of which is a separate computation from
  separate stored quantities. There is no single "density" or "resolution
  multiplier" that expresses it, and the schema must not offer a field that
  invites one.

**Batch identity is mandatory metadata on every measurement** (§4.37,
§1.2): coating-to-coating variation is comparable to the entire remaining model
error budget, so a record whose batch is unknown has an uncertainty floor that the
database must state rather than the analyst having to remember.

---

## A.2 Representation type system

Every stored value must declare its representation from a closed vocabulary. The
value types below are the complete set required by §4; adding a quantity later
must not require adding a type.

| Token | Structure | Use |
|---|---|---|
| `SCALAR` | one real + optional significant-digit count | single measured or defined magnitudes |
| `INT` | signed integer | counts, indices, layer numbers |
| `BOOL` | true/false/unknown (three-valued) | presence flags |
| `TOKEN` | member of a named controlled vocabulary | conventions, states, classes, units |
| `INTERVAL` | (lo, hi) + closure + meaning (`range`, `tolerance`, `spec_limit`) | manufacturing tolerances, valid domains |
| `VECTOR[n]` | ordered reals + per-index semantic labels | RGB triples, coefficient sets |
| `MATRIX[m×n]` | 2-D array + basis definition for input and output space | channel mixing, colour transforms |
| `CURVE` | x[] (explicit, not assumed uniform), y[], axis descriptors, monotonicity flag, extrapolation policy, optional per-sample σ | any measured 1-D relationship |
| `CURVE_FAMILY` | list of `CURVE` + parameter axis values + interpolation rule along that axis | curve vs development time, MTF vs processing, granularity vs density-and-state |
| `SPECTRUM` | `CURVE` on wavelength + measurement bandwidth + λ units | sensitivity, transmittance, spectral density |
| `SPECTRUM_FAMILY` | `CURVE_FAMILY` with λ as inner axis | spectral density vs density level; sensitivity vs processing |
| `GRID[k]` | k-dimensional sampled array + one axis descriptor per dimension + storage order | genuinely joint dependencies that cannot be factored |
| `PSF` | radial profile (`CURVE`, log-spaced radii) **or** full 2-D kernel + pixel pitch in µm + normalisation convention | halation, turbidity, scatter |
| `PSD` | `CURVE` on spatial frequency + one- or two-sided convention + normalisation + aperture referral | noise power (Wiener) spectra |
| `DIST` | either parametric (family token + parameters) or empirical (bin edges + density, or sample quantiles) | grain size, defect size, exposure jitter |
| `HIST` | bin edges + counts or densities + normalisation flag + total N | raw measured populations |
| `MAP` | 2-D field + physical extent in mm + sampling pitch + edge/repeat semantics | coating non-uniformity, spatial variation across the frame or web |
| `SEQUENCE` | index- or time-ordered series + index meaning + rate | frame-to-frame variation, transport records |
| `PROCESS` | stochastic process family + parameters + correlation structure (spatial and/or temporal) + stationarity flag | defect populations, grain field statistics, weave |
| `FUNCTION` | closed-form identifier + coefficients + validity domain + residual against the data it came from | **layer-2 only**; never a layer-1 representation of a measurement |
| `LUT` | uniform-grid table + domain + entry count + interpolation and clamping rules | **layer-3 only** |
| `BLOB` | external file reference + format + checksum + byte size | raw scans, target images, digitiser exports, instrument dumps |
| `TEXT` | free text, language-tagged | notes, caveats, verbatim source wording |

Three rules on the type system itself:

1. **`FUNCTION` is forbidden in layer 1.** A parametric form is a model of a
   measurement, never the measurement. If the only surviving form of a historical
   quantity is a published formula, the layer-1 record is the *published formula as
   quoted*, typed as `FUNCTION` with `source_is_authoritative: true` and a note
   that no sample data exists — which is a materially different claim from a fit.
2. **`CURVE` stores an explicit x array.** Published sensitometric and MTF data is
   almost never uniformly sampled, and forcing it onto a uniform grid at ingestion
   time performs an irreversible interpolation inside the evidence layer.
   Resampling to a canonical grid is a layer-2 operation.
3. **Aggregate statistics never replace the population they summarise.** Where a
   `DIST` or a scalar summary is stored, the `HIST` or `BLOB` it was computed from
   must be stored too if it exists (A.5 marks this per quantity).

---

## A.3 The universal measurement record

One envelope, used for every quantity in A.5. Uniformity here is what makes
adding a quantity a registry entry rather than a schema migration (A.12).

```
MeasurementRecord
├── record_id             stable, globally unique, never reused
├── quantity              TOKEN from the quantity registry (A.5)
├── subject
│   ├── level             emulsion | coating | product
│   ├── id                identity at that level
│   └── component         layer index | channel | dye | "integral" | not_applicable
├── value                 typed per A.2
├── units                 TOKEN from the unit registry; "dimensionless" is explicit
├── conventions{}         quantity-specific vocabulary: density metric, geometry,
│                         aperture, speed criterion, MTF definition, PSD normalisation
├── conditions            ConditionSet (below)
├── uncertainty
│   ├── kind              none_stated | scalar | per_sample | interval | qualitative
│   ├── value             σ, σ[], or (lo,hi)
│   ├── coverage          k factor or confidence level
│   └── basis             repeatability | reproducibility | digitiser | estimated | spec_limit
├── sampling{}            per axis: name, grid, step or spacing law, count,
│                         extrapolation policy, resolution limit of the instrument
├── significant_digits    as reported — prevents fits implying precision that was never there
├── provenance            ProvenanceRef (A.8)
├── confidence_class      C1 | C2 | C3 | C4  (A.8.3)
├── status                active | provisional | superseded | rejected
├── supersedes[]          record_ids
├── digitisation          method | operator | date | tool | estimated_error | BLOB of the extraction
├── conditions_unknown[]  explicit list of condition slots that the source did not state
└── notes                 TEXT, including verbatim source caveats
```

```
ConditionSet
├── processing_state_id   → ProcessingState object (A.5.13); not_applicable where meaningless
├── batch_id              → Batch object; explicitly unknown is a legal, flagged value
├── product_id / format   → Product node where the measurement is format-bound
├── exposure
│   ├── illuminant        TOKEN + SPECTRUM reference (a name alone is insufficient)
│   ├── filter            reference to filter transmittance SPECTRUM
│   ├── duration_s        SCALAR or INTERVAL
│   ├── modulation        continuous | intermittent (+ profile SEQUENCE)
│   └── level_series      the exposure axis actually used, in log H
├── measurement
│   ├── instrument        make/model/class
│   ├── geometry          diffuse | specular | projection | Callier_corrected (+ Q)
│   ├── spectral_response densitometric convention TOKEN + SPECTRUM reference
│   ├── aperture          diameter_um + shape + effective area
│   ├── scan_step_um      sampling pitch, distinct from aperture
│   ├── bandwidth_nm      spectral slit width
│   └── calibration       reference to the calibration record in force
├── environment           temperature_C, relative_humidity_pct
├── timing                post_exposure_delay_s, post_process_age_days
└── sample_history        storage temperature/duration, prior light exposure, age at exposure
```

**Every slot is individually optional; none is implicitly defaulted.** A slot that
is absent must appear in `conditions_unknown[]` or be marked
`not_applicable`. A loader that fills a missing aperture with a house default is
producing a number that the archive cannot distinguish from a measurement, which
is the failure mode §4.35 exists to prevent.

---

## A.4 Profile organisation — the object tree

```
FilmDatabase
├── schema_version
├── registries
│   ├── quantities            definition, units, allowed conventions, validators
│   ├── units                 with dimensional signature for machine checking
│   ├── conventions           density metrics, geometries, speed criteria, MTF/PSD definitions
│   ├── illuminants           name → SPECTRUM
│   ├── processing_chemistries name → composition, published spec
│   └── vocabularies          material classes, base types, defect kinds
├── sources[]                 bibliographic + access records (A.8.1)
├── measurement_chains[]      densitometers, microdensitometers, scanners (A.5.17)
├── emulsions[]
│   ├── identity, manufacturing, emulsion + layer-stack description
│   └── measurements[]        records whose subject level is `emulsion`
├── coatings[]
│   ├── emulsion_ref, support_ref, batch identity, manufacture date/plant
│   └── measurements[]
├── products[]
│   ├── coating_ref, format, perforation, winding, packaging, commercial identity
│   └── measurements[]
├── supports[]                base materials as reusable objects
├── processing_states[]       reusable, referenced by measurements
├── batches[]
├── equivalence_assertions[]
├── models[]                  layer 2 (A.6)
├── validation_sets[]         reference imagery + expected results (A.5.18)
└── resolution_policies[]     record-selection rules (A.1.4)
```

Four points about this tree that are recommendations, not incidental structure:

* **`sources`, `measurement_chains`, `processing_states`, `supports`,
  `illuminants` and `batches` are top-level and shared.** They are referenced by
  many measurements across many materials. Copying them into each profile
  guarantees they will drift apart, and a scanner recalibration or a corrected
  chemistry spec would then have to be applied in dozens of places.
* **A "film profile" as a user-facing concept is a *view*, not a storage unit** —
  the resolved set of records for one Product under one processing state,
  assembled through the inheritance and resolution rules. What is stored is
  normalised; what is consumed is a resolved snapshot. This is what allows one
  material to be simulated under four development conditions without four
  duplicated profiles.
* **The measurement-chain objects are separate from film profiles by design.**
  Scanner spectral responsivities, flare, MTF, geometric calibration and OECF
  (§4.36) describe the instrument. Any measured density record references the
  chain that produced it. When the scanner is recharacterised, one object changes
  and every dependent model becomes detectably stale (A.7).
* **Nothing in the tree is keyed by the simulation's stage order.** Stage
  ordering is a property of the engine, and grouping stored data by which stage
  consumes it is the single most common way a database becomes unable to support a
  restructured pipeline.

---

## A.5 Content catalogue

### Legend for all tables in A.5

**Class** — necessity, as derived from §1 and §8:

| | meaning |
|---|---|
| **M** | Mandatory. A profile without it is not usable at all; the loader must refuse it. |
| **R** | Required to reach the 90 % target. Absence caps achievable fidelity. |
| **X** | Required to exceed 90 % toward 95 %. |
| **O** | Optional; improves specific scenarios or archival scope. |

**Varies with** — the axes the quantity genuinely depends on: `static`, `λ`
(wavelength), `H` (exposure), `D` (density), `P` (processing state), `T`
(temperature), `RH` (humidity), `f` (spatial frequency), `t` (time/duration),
`pos` (position), `frame`, `batch`, `format`.

**Store** — `direct` (measured value stored as such), `derived` (computed in layer 2
from other stored records), `both` (measurement stored *and* a derived form
cached).

**Raw** — must the original measured data be preserved in addition to any
simplified representation? `yes` means the high-resolution or population-level
data is retained permanently even when a compact form is used for rendering.

**AD** — acquisition difficulty:

| | meaning |
|---|---|
| **AD1** | Published in ordinary product documentation. |
| **AD2** | Available on request, or in standards, technical bulletins, patents or journal literature. |
| **AD3** | Measurable in-house with a densitometer, sensitometer and standard processing. |
| **AD4** | Measurable in-house only with specialist equipment (microdensitometer, scanning aperture rig, controlled-temperature processor, optical bench). |
| **AD5** | Research-grade: requires destructive analysis, electron microscopy, proprietary internal data, or access the manufacturer does not grant. |

---

### A.5.1 Identification and metadata

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| ID.1 | Stable internal key, never reused, not derived from the name | `TOKEN` | — | one | unique | M | static |
| ID.2 | Manufacturer designation, verbatim, plus catalogue/code number | `TEXT`, `TOKEN` | — | — | — | M | static |
| ID.3 | All known alternative designations, national-standard marks, export names, renamings, with the source establishing each | list of `{TEXT, source_ref, confidence_class}` | — | — | — | M | static |
| ID.4 | Material class: negative / reversal / positive-print / intermediate / duplicating; monochrome / colour; masked or unmasked | `TOKEN` set | — | — | closed vocabulary | M | static |
| ID.5 | Intended application: camera original, print, intermediate, still, motion picture, aerial, scientific | `TOKEN` set | — | — | closed vocabulary | M | static |
| ID.6 | Nominal speed as marketed, with the rating system named | `SCALAR` + `TOKEN` | ISO / ASA / DIN / GOST / other | one per system | 1–10000 | M | static |
| ID.7 | Balance: daylight / tungsten / unbalanced, with the reference illuminant identified as a spectrum, not a name | `TOKEN` + illuminant ref | — | — | — | M | static |
| ID.8 | Production period: first and last manufacture, discontinuation date, generational lineage (predecessor/successor links) | `INTERVAL` of dates + links | ISO-8601 | — | — | M | static |
| ID.9 | Data-completeness state of the profile itself: which classes M/R/X are satisfied | computed `TOKEN` + report | — | — | — | M | static |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| ID.1–ID.5 | — | selection, validation gating, model-family choice (a reversal material and a negative material require different curve handling and cannot share a default) | direct | — | AD1 |
| ID.6 | must be cross-checked against SEN.1 + the stated criterion (SEN.4) | display and sanity checking only — **never** as a simulation parameter | direct | — | AD1 |
| ID.7 | illuminant registry; SPC.1 | selects the default exposure illuminant when the caller does not specify one | direct | — | AD1 |
| ID.8 | — | provenance plausibility (a datasheet dated outside the production window is a flag), archival scope | direct | — | AD2 |
| ID.9 | every other record | refuses or downgrades a render whose accuracy claim exceeds its data | derived | — | — |

**Note on ID.3 and ID.6.** A designation is not a measurement, and two names that
appear in one standard are not evidence that two materials are the same material.
ID.3 stores names with the source that links them; asserting the *materials*
equivalent is a separate `EquivalenceAssertion` (A.1.5) carrying its own evidence.
ID.6 is marketing data — the speed actually used by the simulation must be derived
from the curve under the stated criterion, because published speeds differ by
rating system, by criterion, and by era for the same emulsion.

---

### A.5.2 Manufacturing information

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| MF.1 | Manufacturer, plant, and coating line where known | `TOKEN`/`TEXT` | — | — | — | R | static |
| MF.2 | Emulsion number / coating-run identifier, coating date | `TOKEN`, date | ISO-8601 | per batch | — | R | batch |
| MF.3 | Batch-to-batch variation actually observed for key quantities: speed, contrast, Dmin, granularity | per-quantity `DIST` or `INTERVAL` (σ or spec limits) | quantity's own units | ≥3 batches to be meaningful | speed ±⅓ stop, γ ±0.05, Dmin ±0.02 as typical magnitudes | R | batch |
| MF.4 | Manufacturer's release specification and tolerance for each controlled quantity | `INTERVAL` per quantity + `TOKEN` (spec vs observed) | — | — | — | O | static |
| MF.5 | Known formulation revisions within one designation, with dates and what changed | list of `{date, description, source_ref, affected_quantities[]}` | — | — | — | R | static |
| MF.6 | Packaging, winding, expiry policy, storage recommendation as manufactured | `TOKEN`s + `TEXT` | — | — | — | O | static |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| MF.1–MF.2 | every measurement's `batch_id` | keys measurements to batches; enables batch-consistent model resolution | direct | — | AD2 |
| MF.3 | needs ≥3 measured batches of the same quantity | **the uncertainty floor of §1.2** — bounds the accuracy any model can honestly claim, and provides the distribution for deliberate batch-variation rendering | both | yes | AD3 |
| MF.4 | — | validation limits: a measurement outside the manufacturer's own tolerance is a flag on the measurement, not on the material | direct | — | AD2 |
| MF.5 | ID.8; supersession of measurement records | prevents merging pre- and post-revision measurements under one designation | direct | — | AD2 |

**Note on MF.3.** This is not bookkeeping. §1.2 established that coating variation
is of the same order as the entire remaining error budget at the 95 % target. Its
consequence is a *requirement on the database*: without a stored measure of
material variability, a fidelity claim cannot be stated with a meaningful
tolerance, and two honest measurements of the same designation will look like a
contradiction. Where fewer than three batches are available, that fact must be
recorded rather than a variability estimate invented.

---

### A.5.3 Format

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| FMT.1 | Gauge and width, perforation type and pitch, winding | `TOKEN`s + `SCALAR`s | mm | — | 8–70 mm | M | format |
| FMT.2 | Image aperture: width × height, and offset from the perforation datum | `VECTOR[4]` | mm | — | 4×3 to 60×60 mm | M | format |
| FMT.3 | Total support + emulsion thickness as slit, and its tolerance | `SCALAR` + `INTERVAL` | µm | — | 80–250 µm | R | format |
| FMT.4 | Native capture rate(s) and shutter-angle range for motion-picture products | `SCALAR` list + `INTERVAL` | frames/s, degrees | — | 8–120 fps; 0–235° | R | format |
| FMT.5 | Edge markings, frame-line and keycode geometry | `MAP` or geometric description | mm | — | — | O | format |
| FMT.6 | Which measurements in this profile are format-bound versus format-independent | explicit per-quantity `BOOL` | — | — | — | M | static |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| FMT.1–FMT.2 | product node identity | sets the physical mapping between output pixels and millimetres on film — the conversion that makes every spatial quantity (MTF in cyc/mm, granularity aperture in µm, PSF radius in µm, defect size in µm) referable to the rendered image | direct | — | AD1 |
| FMT.3 | OPT.2 (halation depends on base thickness) | scatter geometry | direct | — | AD1 |
| FMT.4 | TMP.* | temporal model rate; exposure duration when shutter angle and rate are known | direct | — | AD1 |
| FMT.6 | A.1.5 inheritance rules | prevents a format-bound measurement being inherited by a different gauge | direct | — | — |

**Note.** FMT.2 is the reason format cannot be reduced to a resolution multiplier.
Every spatial quantity in this database is stored in physical units on the film,
and the format determines the magnification to the output. Granularity referred to
a 48 µm aperture on a 10.26 mm-wide frame and on a 24.9 mm-wide frame produce
different rendered noise for the same stored number, and the conversion involves
Selwyn scaling (GRN.2) and the MTF referred to output frequency (RES.1) — two
separate stored quantities combined by computation. A single scale factor cannot
express it.

---

### A.5.4 Film base / support

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| BAS.1 | Support material and manufacture type: nitrate, diacetate, triacetate, polyester/PET, polycarbonate | `TOKEN` | — | — | closed vocabulary | M | static |
| BAS.2 | Support thickness | `SCALAR` + `INTERVAL` | µm | — | 60–200 µm | R | static |
| BAS.3 | Refractive index of the support | `SCALAR`, or `SPECTRUM` where dispersion matters | dimensionless | 1 value, or 10 nm over the visible | 1.45–1.70 | R | λ |
| BAS.4 | Spectral transmittance of the base as it exists in the finished product, including any tint, grey base or dye | `SPECTRUM` | dimensionless (0–1) or density | 5 nm, 340–730 nm minimum | 0–1 | R | λ |
| BAS.5 | Antihalation provision: type (backing, undercoat, grey base, none), spectral absorbance, and whether it is removed in processing | `TOKEN` + `SPECTRUM` + `BOOL` | density | 5 nm | 0–4 D | R | λ, P |
| BAS.6 | Base + fog density, per channel, in a stated density metric, at a stated processing state | `VECTOR[n_ch]` | density | one per channel per state | 0.02–0.90 | M | P, λ, batch |
| BAS.7 | Surface finish and mechanical properties: gloss, backing texture, curl, dimensional stability under processing | `SCALAR`s / `TOKEN`s | %, ppm | — | — | O | RH, T |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| BAS.1–BAS.2 | FMT.3 | geometry of the scatter path: halation ring radius scales with the emulsion-to-rear-surface distance | direct | — | AD1 |
| BAS.3 | BAS.1 | Fresnel reflectance at the rear surface and the internal angle limit — the amplitude of the halation term (OPT.2) | direct | — | AD2 |
| BAS.4 | CAL.* densitometric convention | multiplicative spectral floor on every transmitted measurement; must be subtracted before dye densities can be interpreted, and added back when rendering to a scan | direct | yes | AD3 |
| BAS.5 | OPT.2 | attenuation of the back-reflected component; the `BOOL` matters because a backing removed in processing does not attenuate exposure-time scatter but does affect the measured base density | direct | yes | AD2 |
| BAS.6 | processing state; batch | the additive density offset and the black floor of the rendered image; also the anchor that makes two datasheets comparable | direct | — | AD1 |

**Note on BAS.4 versus BAS.6.** These are different quantities that datasheets
frequently conflate. BAS.4 is the spectral transmittance of the support itself;
BAS.6 is the measured minimum density of processed unexposed stock, which includes
support, residual antihalation, chemical fog and any residual coupler stain. Both
are required, they are not derivable from each other, and storing only the second —
the usual case — makes it impossible to separate a coloured base from chemical fog
when modelling how the material behaves under a different processing state.

---

### A.5.5 Emulsion and layer stack

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| EMU.1 | Layer inventory in physical order from the exposing surface: every emulsion, filter, interlayer, UV absorber, overcoat and subcoat | ordered list of `{role TOKEN, id}` | — | — | 1–20 layers | R | static |
| EMU.2 | Per-layer thickness, wet and dry | `SCALAR` + `INTERVAL` | µm | one per layer | 0.5–12 µm | X | static, P (swell) |
| EMU.3 | Per-layer sensitisation target and which record set applies to it | link to SPC.1 records | — | — | — | M | static |
| EMU.4 | Layer duplication: whether a colour record is split into fast/slow (or three-part) sublayers, and each sublayer's share of the record's total response | `INT` + `VECTOR` of fractions | dimensionless, sums to 1 | per record | 1–3 sublayers | X | static |
| EMU.5 | Silver halide composition per layer: halide ratios, dopants | `VECTOR` of mole fractions | dimensionless | per layer | sums to 1 | O | static |
| EMU.6 | Coating weight: silver per unit area, gelatin per unit area, per layer and total | `SCALAR` | g/m² | per layer | Ag 0.5–10 g/m² | X | static |
| EMU.7 | Crystal habit and morphology per layer: cubic / octahedral / tabular, aspect ratio, thickness | `TOKEN` + `SCALAR`s / `DIST` | dimensionless, µm | per layer | aspect 1–25 | X | static |
| EMU.8 | Grain size distribution per layer — the population, not a mean | `DIST` + `HIST` of the measured population | µm (equivalent circular diameter) | ≥20 bins, ≥1000 grains counted | 0.02–4 µm | X | static |
| EMU.9 | Interlayer optical density and scatter between records (filter-layer transmittance, gelatin turbidity) | `SPECTRUM` per interlayer + `PSF` | density; µm | 5 nm; log-radial | — | X | λ |
| EMU.10 | Development-inhibitor-releasing chemistry present, and in which layers | `BOOL` + `TOKEN` per layer | — | — | — | R | static |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| EMU.1–EMU.2 | BAS.2, OPT.2, OPT.3 | the geometric depth model: which layer light reaches after passing through which absorbers, and the propagation distances that set inter-layer crosstalk and halation | direct | — | AD4/AD5 |
| EMU.4 | SEN.1, GRN.1 | explains curve shape that a single-layer model cannot produce (extended straight-line, shoulder behaviour) and the density dependence of granularity; each sublayer contributes its own curve and its own grain population | direct | — | AD2 |
| EMU.6–EMU.8 | GRN.1–GRN.4 | grain amplitude scaling, per-grain spatial extent, and the shape of the noise autocorrelation; a size *distribution* produces the frequency dependence that a single mean size cannot | both | **yes** | AD5 |
| EMU.9 | SPC.1, COL.2 | attenuation and blur between records — the optical part of colour crosstalk, which is not the same mechanism as the chemical part (PRC.5) | direct | yes | AD5 |
| EMU.10 | PRC.5, PRC.6 | presence gate for inhibitor diffusion and adjacency behaviour | direct | — | AD2 |

**Note on EMU.8, and the general principle behind it.** A grain population is a
distribution, and its distribution *shape* — not its mean — determines the shape of
the noise power spectrum (GRN.3) and how granularity scales with aperture. Storing
a mean diameter is the clearest example of the reduction this appendix argues
against: it is a single number standing in for a measured histogram, it cannot be
inverted, and any future noise model built on it inherits the current model's
assumption that grain is monodisperse. Where only a mean is available, store the
mean *as a mean*, flagged `C1` with an explicit note that the population is
unavailable — not as a distribution with an invented width.

**Note on the AD ratings in this group.** Most of this group is AD4–AD5, and much
of it will never be obtained for historical materials. That is not a reason to omit
it from the target model. It is a reason to design the model layer so that a
coarser grain representation is a *documented fallback* (A.6.3) rather than the
only representation the schema can express.

---

### A.5.6 Spectral characteristics

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| SPC.1 | Spectral sensitivity of each record/layer: log sensitivity versus wavelength | `SPECTRUM` per layer, with the sensitivity criterion stated | log10 of reciprocal exposure (e.g. log m²/J); criterion `TOKEN` | **2 nm preferred, 5 nm minimum**, 340–730 nm minimum, extended to the material's actual limit (to 900+ nm for extended-red or IR sensitisation) | 4–5 decades of dynamic range; store to the noise floor, not to an arbitrary cut | M | λ, P, criterion |
| SPC.2 | The density criterion at which SPC.1 was measured, and whether sensitivity was determined at fixed density or fixed exposure | `TOKEN` + `SCALAR` | density | per record | — | M | static |
| SPC.3 | Dependence of spectral sensitivity on processing state, where it exists | `SPECTRUM_FAMILY` over P | as SPC.1 | ≥2 states | — | O | λ, P |
| SPC.4 | UV and IR response beyond the visible, and the wavelength limits of the measurement itself | `SPECTRUM` extension + `INTERVAL` | nm | 10 nm | 300–1100 nm | R | λ |
| SPC.5 | Spectral transmittance of any built-in filter layer, as coated | `SPECTRUM` | density or 0–1 | 5 nm | — | R | λ |
| SPC.6 | Exposure illuminant(s) under which all other sensitometry in the profile was performed | reference to illuminant `SPECTRUM` | relative power | 5 nm | — | M | λ |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| SPC.1 | EMU.3; SPC.2 is meaningless without it; SPC.6 | the integrand of ∫ S_layer(λ) · E(λ) dλ that converts a spectral scene radiance into a per-layer exposure — the only physically correct route from scene to film, and the mechanism that reproduces the material's characteristic colour separation and its metameric behaviour | both (curve + optional integrated response for a named illuminant) | **yes** | AD1 |
| SPC.2 | SEN.1 | makes two manufacturers' sensitivity curves comparable; without it the vertical placement of the curve is arbitrary | direct | — | AD1 |
| SPC.4 | BAS.4, SPC.5 | UV response combined with lens and base UV transmittance; a real contributor that a 400–700 nm model silently drops | direct | yes | AD2 |
| SPC.6 | every sensitometric record | defines what "log H" means in SEN.1; two curves measured under different illuminants are not on the same axis | direct | — | AD1 |

> **Measured 2026-08-13.** The sampling requirement below was tested rather than
> assumed, and the test separated two things this note previously ran together:
> the sampling of the STORED curve and the grid the INTEGRAL is evaluated on.
> Against a smooth blackbody illuminant the integration grid barely matters
> (5 nm differs from a 1 nm reference by 1.1e-3 on derived balance gains, 10 nm
> by 3.2e-3). Against a narrow-line illuminant it decides the answer: the
> red/green layer ratio is wrong by **1.5 % at a 5 nm grid, 52.7 % at 10 nm and
> 231 % at 25 nm**, while 2 nm matches 1 nm exactly. The engine's integration
> grid is therefore now 2 nm. Separately, decimating a natively-10 nm stored
> curve costs 0.4–1.1 % at 20–30 nm and up to 8.6 % at 40 nm, so 10 nm storage
> is adequate for the present consumer while 20–25 nm (four stocks) is not.
> Full record: `CURVE_RESOLUTION_ANALYSIS_2026-08-13.md`.

**Note on sampling density.** Sensitisation dyes produce absorption features with
half-widths of 20–40 nm, and sharp-cutting filter layers produce edges steeper
still. 10 nm sampling misplaces a dye peak by up to 5 nm and flattens its shape;
5 nm is adequate for smooth sensitisations and is what published curves usually
support after digitisation; 2 nm is required to resolve narrow features and is the
right target for any curve measured in-house. **The sampling of the stored curve
must never be finer than the measurement's own spectral bandwidth** — a 5 nm grid
generated from a 10 nm-slit measurement asserts resolution that does not exist, so
the bandwidth belongs in the record (A.3) and a validator should compare them.

**Note on why this group is class M.** §1.3 established that spectrally
uncharacterised input caps fidelity at ~70–75 % regardless of everything else. The
matching statement on the film side is that without SPC.1 the simulator can only
apply a fixed channel matrix, which reproduces the material's response to the one
illuminant the matrix was fitted under and to no other. Spectral sensitivity is
the single highest-value curve in the entire database.

---

### A.5.7 Sensitometric characteristics

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| SEN.1 | Characteristic (H&D) curve per channel: density versus log exposure, **at full measured resolution** | `CURVE` per channel per processing state | density vs log10 (lux·s) or log10 (J/m²) | **≤0.05 log H, ≥5.0 log H total span**, i.e. ≥100 points/channel; never fewer than the source provides | D 0.02–5.0; log H spanning toe to shoulder plus ≥0.3 beyond each | M | H, P, λ metric, batch |
| SEN.2 | Curve family over processing state — the same curve at every documented development condition | `CURVE_FAMILY` over P | as SEN.1 | ≥4 states spanning the usable contrast range | — | R | H, P |
| SEN.3 | Density metric and geometry of every curve: Status A/M, ISO visual, printing density, spectral band, diffuse vs specular, Callier Q | `TOKEN` set per curve | — | per curve | — | M | static |
| SEN.4 | Speed point: the criterion (fixed density above base+fog, ISO speed-point construction, inertia, fractional gradient) and the resulting exposure | `TOKEN` + `SCALAR` | log H | per state | — | M | P, criterion |
| SEN.5 | Contrast descriptors *as published*: γ, average gradient, contrast index, with the exact density or log-H interval each was computed over | `SCALAR` + `INTERVAL` each | dimensionless + log H | per state | γ 0.3–4.0 | R | P |
| SEN.6 | Gradient of the curve as a function of exposure — the point-by-point derivative | `CURVE` | dimensionless vs log H | as SEN.1 | 0–5 | R | H, P |
| SEN.7 | Dmax and Dmin per channel, with the exposure that produced Dmax and whether it was saturating | `SCALAR`s + `BOOL` | density | per state | Dmin 0.02–0.9; Dmax 0.8–5.0 | M | P, batch |
| SEN.8 | Curve-shape landmarks located on the measured data: toe onset, straight-line interval, shoulder onset | `SCALAR`s in log H | log H | per state | — | R | H, P |
| SEN.9 | Uncertainty of the curve: repeatability of the sensitometric measurement itself, per-point where available | σ or σ[] | density | per point if possible | 0.005–0.05 D | R | — |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| SEN.1 | SPC.6 (defines log H), SEN.3 (defines D), PRC.1 (defines P), BAS.6 (defines the floor) — **all four are prerequisites; a curve missing any one of them is not interpretable** | the transfer function from integrated per-layer exposure to per-layer image-forming density; the single most consumed object in the database | **both** — full curve retained, monotone canonical resampling and any parametric fit cached in layer 2 | **yes, unconditionally** | AD1/AD3 |
| SEN.2 | PRC.1–PRC.3; must be co-measured with GRN.1 and RES.1 on the same strips | lets the simulation move along the development axis instead of interpolating toward an unmeasured state; the *only* honest way to render push/pull or a non-standard process | both | yes | AD2/AD3 |
| SEN.3 | CAL.1–CAL.3 | selects how stored density relates to what a scanner will measure; mixing metrics silently produces a systematic error of order 0.05–0.2 D | direct | — | AD1 |
| SEN.4–SEN.6, SEN.8 | SEN.1 | reporting, validation, and model initialisation | **derived from SEN.1**, and *also* stored as published where the published value came from data we do not have | — (published values: yes) | AD1 |
| SEN.7 | SEN.1, BAS.6 | clamp limits and the black/white anchors of the render | derived + direct | — | AD1 |
| SEN.9 | — | weighting in curve fitting; the floor on any claim about curve accuracy | direct | — | AD3 |

**This group is where the "do not reduce curves to parameters" requirement bites
hardest, so it is worth being explicit about what is being recommended.**

A published H&D curve carries information in at least five places that a small
parameter set does not reproduce: the *curvature profile* of the toe (which sets
shadow rendering and differs markedly between materials of identical nominal
contrast); the *length and straightness* of the linear region; the *rate of
approach* to the shoulder; any *inflection* from sublayer structure (EMU.4); and
the *divergence between channels* at the extremes, which is what produces
characteristic crossover and colour casts in shadows and highlights. Three or four
parameters can match the mid-tones and the nominal contrast while getting all five
of those wrong.

Therefore the recommendation is: **store the digitised curve at ≤0.05 log H and
never discard it; store any parametric or canonical form as a layer-2 model with
its residual recorded against the curve it came from.** A fit whose maximum
residual exceeds SEN.9 is not an acceptable substitute for the curve and the
database should say so, per-record, rather than leaving it for a colourist to
discover.

**On digitising from a printed chart** — the practical source for most historical
material: the achievable accuracy is roughly 0.02–0.04 D and 0.02–0.05 log H,
limited by line width and axis calibration, and it must be recorded in
`digitisation.estimated_error`. That is *better* than a four-parameter fit to the
same chart, which is the point. Store the axis calibration points used, and keep
the source image as a `BLOB`, so the digitisation can be redone at higher density
later without going back to the document.

---

### A.5.8 Exposure response (beyond the static curve)

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| EXP.1 | Reciprocity-law failure: the shift in effective speed, **and separately in contrast**, as a function of exposure duration at fixed total exposure | two `CURVE`s (Δlog H and Δγ vs log duration) per channel | log H, dimensionless, vs log10 s | ≥3 points per decade over 10⁻⁵–10² s | Δlog H 0 to −1.5; Δγ ±0.5 | R where exposures depart from the reference duration | t, λ, P |
| EXP.2 | The reference exposure duration at which SEN.1 was measured | `SCALAR` | s | one | 10⁻³–10 s | M | static |
| EXP.3 | Intermittency: whether a divided exposure of the same total differs from a continuous one, and by how much | `CURVE` vs duty cycle / segment count | Δlog H | ≥4 conditions | 0 to −0.5 | O | t |
| EXP.4 | Latent-image keeping: density change as a function of delay between exposure and development | `CURVE` per channel | Δdensity or Δlog H vs log10 hours | ≥5 points, 0–10⁴ h | 0 to −0.4 log H | O | t, T, RH |
| EXP.5 | Per-channel reciprocity divergence — the colour shift that long exposure produces | derived from EXP.1 per channel | log H | — | — | R for colour where EXP.1 applies | t, λ |
| EXP.6 | Halation/scatter contribution to apparent exposure at high levels, if measured as an exposure effect rather than optically | `CURVE` vs exposure | density | — | — | O | H |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| EXP.1 | EXP.2 (the anchor), SEN.1 | a per-channel shift applied to the exposure axis *before* the curve, plus a contrast modification — **not** a post-curve density offset, because reciprocity failure changes the curve's shape and not merely its position | direct | yes | AD2/AD3 |
| EXP.2 | — | without it EXP.1 has no origin and cannot be applied at all | direct | — | AD1 |
| EXP.5 | EXP.1 for every channel | the characteristic long-exposure colour shift; requires reciprocity measured per channel, not once for the material | derived | — | AD3 |

**Note.** EXP.1 needs an explicit exposure-duration input to the simulation. If the
rendering interface cannot express "how long was this frame exposed", the data
cannot be used no matter how well it is stored — which is a requirement this
appendix places on the *engine interface*, listed here because it is the reason the
data is worth collecting. Storing Δγ separately from Δlog H matters: treating
reciprocity as a pure speed loss preserves contrast that a real material loses.

---

### A.5.9 Colour response

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| COL.1 | Spectral density of each image-forming dye, individually, normalised, **and** at several density levels | `SPECTRUM` per dye + `SPECTRUM_FAMILY` over density | density | 2 nm preferred / 5 nm minimum, 360–730 nm; ≥4 density levels | 0–4 D | M for colour | λ, D |
| COL.2 | Unwanted absorptions: each dye's density in the other channels' measurement bands, as measured rather than as a fitted matrix | derived from COL.1 + CAL.2 | density | — | — | M for colour | λ, D |
| COL.3 | Masking-coupler density: the spectral density of unreacted coupler and its dependence on exposure | `SPECTRUM` + `CURVE` vs log H | density | 5 nm; ≥10 exposure levels | 0–1.5 D | M for masked negatives | λ, H, P |
| COL.4 | Chemical interimage (interlayer) coupling: the effect of one record's development on another's, as a function of the *inducing* record's density | `MATRIX[n×n]` of `CURVE`s (coefficient vs inducing density), sign convention stated | dimensionless | ≥6 density levels | off-diagonal magnitudes ≪ 1 | R for colour | D, P |
| COL.5 | Dye stability: density change per dye with light exposure, dark storage, temperature and humidity over time | `CURVE_FAMILY` vs log time | relative density | ≥5 time points per condition | 0–1 relative | O | t, T, RH, λ |
| COL.6 | Spectral density of the material as an integral (all dyes present) at several neutral density levels | `SPECTRUM_FAMILY` | density | 5 nm; ≥4 levels | — | R for colour | λ, D |
| COL.7 | For reversal materials: residual dye at Dmin and the spectral character of the highlight | `SPECTRUM` | density | 5 nm | 0–0.5 D | R for reversal | λ |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| COL.1 | SEN.3, CAL.2 | reconstructs the transmittance spectrum of the developed material from per-layer densities, from which any densitometric or scanner response can be computed — the correct alternative to a fixed 3×3 crosstalk matrix | **both**: spectra retained, any matrix derived for a *named* measurement convention and stored as layer 2 | **yes** | AD1/AD2 |
| COL.2 | COL.1 and the *specific* densitometer or scanner spectral response | the channel crosstalk actually observed by a given measurement chain | **derived**, never stored as a primary matrix | — | — |
| COL.3 | SEN.1 (must be co-measured), COL.1 | the exposure-dependent orange density of masked negatives; a constant mask density is wrong in the shadows and wrong in the highlights in opposite directions | direct | yes | AD1 |
| COL.4 | EMU.10, PRC.5, SEN.2 | modifies each record's effective development according to neighbouring records' exposure; **density-dependent, so a single scalar per pair is a linearisation whose error grows toward both extremes** | direct | yes | AD2/AD3 |
| COL.6 | COL.1 | validation: the integral spectrum must be reconstructible from the individual dye spectra, which is a strong consistency check on both | direct | yes | AD1 |

**Note on COL.2, which is the most consequential recommendation in this group.**
Cross-channel crosstalk is commonly stored as a 3×3 matrix. That matrix is not a
property of the film — it is a property of *the film measured by one particular
instrument*. Change the densitometer's status filters, or scan the same negative on
a different scanner, and the correct matrix changes while the material has not.
Storing the matrix therefore bakes an instrument into the material profile.
Storing COL.1 (dye spectra) and computing the matrix per measurement chain costs
one integration at load time and remains correct when the scanner changes,
which per §4.36 it will.

**Note on COL.4.** Interimage coupling is a chemical mechanism (inhibitor released
by one layer suppressing development in another) and is distinct from EMU.9
(optical crosstalk between layers) and from COL.2 (measurement crosstalk). All
three produce colour contamination and all three must be stored separately,
because they respond differently: the chemical term depends on development state
and density, the optical term on geometry and wavelength, the measurement term on
the instrument. Collapsing them into one matrix makes it impossible to be right
about any of them under changed conditions.

---

### A.5.10 Grain and noise characteristics

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| GRN.1 | RMS granularity as a function of density, per channel — **the curve, not a single figure** | `CURVE` per channel per processing state | σ of density (dimensionless); store unscaled and record whether the source used ×1000 | ≥10 density levels from 0.2 D to near Dmax; more where the curve turns | σ 0.002–0.080 | M | D, P, batch |
| GRN.2 | The scanning aperture of every granularity measurement: diameter, shape, effective area, and the scanning step | `SCALAR`s + `TOKEN` | µm, µm², µm | per record | 5–100 µm | M | static |
| GRN.3 | Noise power (Wiener) spectrum of density fluctuation, per channel, at several density levels | `PSD` per channel per density | density²·mm² (equivalently µm² under the stated normalisation); **one- vs two-sided and aperture-referral convention mandatory** | ≥30 log-spaced points, 1–200 cyc/mm; ≥3 density levels | spans 3–4 decades across the band | X | f, D, P |
| GRN.4 | Grain autocorrelation or equivalent spatial extent, per channel | `CURVE` (radial) or derived from GRN.3 | µm | ≥20 radii, 0–50 µm | correlation length 0.3–20 µm | X | D, P |
| GRN.5 | Cross-channel grain correlation matrix — how much of each layer's noise is shared with the others | `MATRIX[n×n]`, or `MATRIX` of `CURVE`s if it varies with density | dimensionless correlation | one matrix minimum; per density band preferred | −1 to +1; typically 0 to +0.5 | X for colour | D, f |
| GRN.6 | Print-through / grain of the material when used as an intermediate, if applicable | as GRN.1 | — | — | — | O | D |
| GRN.7 | Aperture-scaling behaviour: granularity measured at ≥2 apertures, to test rather than assume Selwyn scaling | ≥2 `CURVE`s, one per aperture | as GRN.1 | 2–3 apertures | — | X | aperture |
| GRN.8 | Whether the granularity figure is of the material alone or includes measurement-system noise, and the system noise floor | `BOOL` + `SCALAR` | σ density | per record | — | R | static |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| GRN.1 | GRN.2 (**a granularity number without its aperture is meaningless and must be rejected, not defaulted**); SEN.1 for the density axis; PRC.1 | the amplitude of the noise injected at each point of the image, indexed by the local density that point reached | direct | yes | AD1/AD3 |
| GRN.2 | FMT.2 | referral of stored noise amplitude to the rendered pixel scale; combined with GRN.7 or Selwyn's law to convert between apertures | direct | — | AD1 |
| GRN.3 | GRN.1 (the spectrum's integral over the aperture must reproduce the RMS figure — a strong cross-check), GRN.2 | shapes the noise spatially: gives the correct correlation structure rather than white noise filtered by a Gaussian, which is the difference between noise that reads as grain and noise that reads as digital | direct | **yes** | AD4/AD5 |
| GRN.5 | GRN.1 per channel | determines whether colour noise appears as luminance grain or chroma speckle — perceptually a large effect, and not derivable from per-channel granularity | direct | yes | AD5 |
| GRN.7 | GRN.2 | validates or replaces the aperture-scaling law used for format conversion | direct | yes | AD3 |
| GRN.8 | CAL.* | prevents the measurement chain's own noise being modelled as the film's | direct | — | AD3 |

**Note on GRN.1 as a curve.** A single RMS granularity value is quoted at one
density, usually near 1.0, and granularity varies by a factor of two or more across
the usable density range — rising from the toe, peaking in the mid-to-upper
mid-tones, and falling again as the material saturates. A model driven by one
number necessarily gets the shadows and highlights wrong in opposite directions,
and the error is most visible exactly where viewers look for grain. The curve is
cheap to store and is published for many materials.

**Note on GRN.3, which §8 ranks as one of the two highest-value unpublished
items.** RMS granularity is a single number: the standard deviation seen through
one aperture. It says nothing about how the fluctuation is distributed over spatial
frequency, and two materials with identical RMS granularity and visibly different
grain character are common. The Wiener spectrum is the complete second-order
description, and its absence is a hard ceiling on grain realism. It will not come
from a datasheet; it requires a microdensitometer or a well-characterised
high-resolution scan of uniformly exposed patches (§6). The database must be able
to hold it before it is worth measuring, which is why it is specified here at
class X rather than deferred.

**Note on the aperture, phrased as a rule.** GRN.2 is class M while GRN.1 is the
quantity of interest, because granularity scales roughly as the inverse square root
of aperture area: a figure measured at 24 µm is about 1.4× the same material's
48 µm figure. Importing an unlabelled granularity number therefore risks a 40 %
amplitude error, in an unknown direction. The database must refuse the record
rather than assume the more common convention.

---

### A.5.11 Spatial-resolution characteristics

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| RES.1 | Modulation transfer function per channel/layer — **including the low-frequency region above unity and the high-frequency tail down to the noise floor** | `CURVE` per channel per processing state | response (dimensionless) vs cycles/mm | ≥20 points per decade, log-spaced, from 1 to at least 200 cyc/mm; extend until response < 0.02 | response 0.005–1.5 (values above 1.0 are real and must not be clipped at ingestion) | M | f, P, λ |
| RES.2 | The MTF definition used: sine-wave vs square-wave, whether it is the emulsion alone or includes the exposing optics, and the exposing wavelength | `TOKEN`s + `SCALAR` | nm | per record | — | M | static |
| RES.3 | Resolving power, with the target contrast ratio and the criterion | `SCALAR` + `TOKEN` + `SCALAR` | line pairs/mm; contrast ratio | one per contrast | 20–800 lp/mm; ratios 1.6:1 to 1000:1 | R | f, contrast |
| RES.4 | Edge / knife-edge trace: density versus position across a sharp edge, both directions | `CURVE` per channel | density vs µm | 1–2 µm sampling over ±200 µm | — | X | pos, P, H |
| RES.5 | Adjacency (Eberhard/Kostinsky) effects: the overshoot and undershoot amplitude and their spatial extent, as a function of development | derived from RES.4, or measured directly | density, µm | ≥3 processing states | — | X | pos, P |
| RES.6 | MTF dependence on exposure level, where measured | `CURVE_FAMILY` over H | — | ≥3 exposure levels | — | O | f, H |
| RES.7 | Whether RES.1 was measured on the same batch and processing state as SEN.1 and GRN.1 | `BOOL` + refs | — | — | — | R | — |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| RES.1 | RES.2 (an MTF without its definition cannot be compared to another), FMT.2 for referral to output frequency | the frequency-domain response applied to the latent exposure distribution before development; determines rendered sharpness and, jointly with GRN.3, the perceived grain-to-detail relationship | **both** — full curve retained; any kernel or parametric decomposition is layer 2 | **yes** | AD1 |
| RES.2 | — | selects the conversion between square-wave and sine-wave response (they differ by a known series, and confusing them misstates response by ~10–25 % at mid frequencies) | direct | — | AD1 |
| RES.3 | RES.1 | an independent upper bound: a model whose MTF predicts detail beyond the material's measured resolving power at that contrast is wrong regardless of how well the MTF was fitted. **A constraint, never a substitute for RES.1** | direct | — | AD1 |
| RES.4 | PRC.5, PRC.6, COL.4 | the direct measurement of adjacency behaviour; contains information the MTF cannot carry, because an MTF is a linear description and adjacency effects are nonlinear and asymmetric | direct | yes | AD4 |
| RES.7 | — | prevents a sharpness measurement from one condition being combined with a contrast measurement from another — §4.11's requirement, expressed as a stored flag | direct | — | — |

**Note on the MTF tail, and on values above 1.0.** Both extremes of the MTF are
routinely lost and both matter. At the low end, many materials exceed unity
response between roughly 2 and 20 cyc/mm because of development adjacency effects;
a representation that cannot exceed 1.0, or an ingestion step that clamps to 1.0,
discards the material's acutance and produces a softer rendering than the film. At
the high end, published curves are frequently truncated at 50 % response or at
100 cyc/mm, while the region from there down to a few per cent is what determines
how fine detail dissolves into grain. **Store the curve to the measurement's noise
floor and record where the source stopped**, distinguishing "the response fell
below 0.02 here" from "the chart ended here" — these are different facts and only
one of them is about the film.

**Note on RES.4 versus RES.1.** An edge trace and an MTF are not interchangeable. If
the material were linear and isotropic, one would be the transform of the other;
the reason to measure the edge is precisely that it is not. The asymmetry between
a light-to-dark and a dark-to-light edge is a direct measurement of the nonlinear
adjacency effect, and it is unobtainable from any MTF.

---

### A.5.12 Optical characteristics (scatter, halation, turbidity)

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| OPT.1 | Emulsion turbidity: the within-layer scatter point spread, per layer | `PSF` (radial, log-spaced) | relative amplitude vs µm | ≥40 radii, 0.5–200 µm | normalised to unit integral | X | λ, pos |
| OPT.2 | Halation point spread: the wide, low-amplitude component from reflection at the rear support surface, per channel | `PSF` (radial, log-spaced) | relative amplitude vs µm | ≥60 radii, 1–2000 µm, spanning ≥4 decades of amplitude | integral fraction 0.001–0.10 of the direct term | R | λ, format (via BAS.2) |
| OPT.3 | Inter-layer optical crosstalk: the fraction of light scattered from one record into another, with its spatial distribution | `MATRIX` of `PSF`s | dimensionless + µm | per layer pair | — | X | λ |
| OPT.4 | Total flare/veiling within the material at high exposure ratios, measured as the density in a shadow adjacent to a highlight | `CURVE` vs exposure ratio | density vs log ratio | ≥5 ratios spanning 3+ decades | — | X | H, pos |
| OPT.5 | Surface reflectance and gloss of the emulsion side, and its wavelength dependence | `SPECTRUM` | 0–1 | 10 nm | 0.02–0.10 | O | λ |
| OPT.6 | Whether OPT.1–OPT.3 are separated in the stored data or only their sum was measured | `TOKEN` | — | — | — | R | — |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| OPT.1 | EMU.2, EMU.7 | the narrow component of the exposure spread; overlaps with RES.1 and must not be applied twice (see the note) | direct | yes | AD5 |
| OPT.2 | BAS.2, BAS.3, BAS.5, FMT.3 | the wide halo around bright sources — the effect most responsible for the "photographic" rendering of specular highlights, and one that a Gaussian of any width reproduces badly because the real profile has a much heavier tail | direct | **yes** | AD4 |
| OPT.3 | EMU.1, EMU.9 | wavelength-dependent colour fringing around highlights | direct | yes | AD5 |
| OPT.4 | OPT.2 | validation of the integrated scatter amplitude, obtainable with ordinary equipment where the PSF is not | direct | yes | AD3 |
| OPT.6 | RES.1 | **guards against double-counting**: a measured MTF already contains OPT.1, and often part of OPT.2 | direct | — | — |

**Note on double counting, which is the reason OPT.6 exists.** A measured MTF
already includes turbidity, because turbidity is one of the mechanisms that makes
the MTF fall. Applying a stored MTF *and* a stored turbidity PSF blurs twice. The
database must therefore record, per record, what the measurement already contains,
so the model layer can decompose rather than superimpose: typically MTF minus the
scatter terms gives the residual to be modelled separately, and that subtraction
is a layer-2 operation whose result carries a residual diagnostic. This is a case
where storing raw measurements *plus* an explicit statement of their content is the
only way to avoid an error that is invisible in the data itself.

**Note on OPT.2's sampling.** The halation profile spans three orders of magnitude
in radius and four in amplitude, and it is the tail — not the peak — that produces
the visible halo. Linear radial sampling wastes almost all samples near the origin
and misses the tail; log-spaced radii over 1–2000 µm with amplitude stored
logarithmically is the correct representation. A two-parameter Gaussian fitted to
such data typically underestimates the halo at 500 µm by more than an order of
magnitude.

---

### A.5.13 Processing and development characteristics

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| PRC.1 | Processing state as a reusable named object: chemistry identity, developer composition or commercial process designation, time, temperature, agitation, replenishment rate, sequence of baths | structured object + `TOKEN` | min, °C, — | one object per state | 18–45 °C; 1–30 min | M | — |
| PRC.2 | Which processing state is the manufacturer's reference/normal process | `TOKEN` reference | — | one | — | M | static |
| PRC.3 | Time–gamma (or time–contrast-index) relationship | `CURVE` | γ vs min | ≥5 times | γ 0.3–4.0 | R | P, T |
| PRC.4 | Temperature dependence of contrast, speed and Dmax at fixed time | `CURVE`s vs temperature | per quantity | ≥3 temperatures | — | X | T |
| PRC.5 | Inhibitor (DIR) diffusion range and strength, per layer | `SCALAR` or `CURVE` (strength vs distance) | µm; dimensionless | ≥10 distances, 1–100 µm | range 5–60 µm | X | P, D |
| PRC.6 | Local development exhaustion / large-area effects: the dependence of a patch's density on the mean density of its surround, and the spatial scale of that dependence | `CURVE` (Δdensity vs surround density) + spatial extent | density; mm | ≥5 surround levels; ≥3 scales | — | X | D, pos, P |
| PRC.7 | Agitation regime and its known effect (rotary, dip-and-dunk, continuous machine, stand) | `TOKEN` + notes | — | — | — | R | — |
| PRC.8 | Non-standard processes documented for this material: push/pull, cross-process, reduced-agitation, with the resulting curve family | links to SEN.2 members | — | — | — | O | P |
| PRC.9 | Which quantities were re-measured for each processing state, and which were assumed unchanged | per-quantity `BOOL` matrix | — | — | — | R | — |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| PRC.1 | referenced by every sensitometric, granularity, MTF and colour record | **the index into the data cube along its most-often-omitted axis** (§4.10); without it a curve, a granularity figure and an MTF cannot be known to belong to the same material state | direct | — | AD1 |
| PRC.3 | SEN.2 | interpolation along development where a full curve family is unavailable — a fallback, and one whose error should be recorded by comparison against a measured family where both exist | direct | yes | AD1 |
| PRC.4 | PRC.1 | correction for processing at other than the nominal temperature | direct | yes | AD2 |
| PRC.5 | EMU.10, RES.4, RES.5, COL.4 | the mechanism generating adjacency sharpening and chemical interimage effects; a lateral kernel applied to the developing density field rather than to exposure | direct | yes | AD5 |
| PRC.6 | SEN.1 | large-area development effects: the reason a small patch reads differently against a light and a dark surround | direct | yes | AD4 |
| PRC.9 | SEN.2, GRN.1, RES.1 | makes explicit which parts of a "processing state" are measured and which are inherited — an inherited MTF under a changed development is an assumption, and it must be visible as one | direct | — | — |

**Note on why PRC.1 is a shared top-level object rather than fields inside a
profile.** One processing state applies to many materials, and one material is
measured under many states. Modelling the state as an object referenced by
measurements gives, for free: consistent state identity across materials, the
ability to ask "what do we know about any material in this process", correct
behaviour when a chemistry specification is corrected, and — most importantly —
the impossibility of storing a curve without being able to say what produced it.

---

### A.5.14 Non-uniformity and defect characteristics

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| UNI.1 | Coating non-uniformity: low-spatial-frequency density variation across the width and along the length of the web | `MAP` + `PSD` of the variation | density; mm | ≥1 mm pitch across width; ≥100 mm along length | ±0.005–0.05 D | X | pos, batch |
| UNI.2 | Edge effects of the format: density gradient near the slit edges and perforations | `CURVE` vs distance from edge | density vs mm | 0.1 mm pitch over 0–5 mm | — | O | pos, format |
| UNI.3 | Edge fog and light-piping susceptibility | `CURVE` vs distance | density vs mm | — | — | O | pos |
| UNI.4 | Particulate defect populations, by kind (dust, dirt, fibre, embedded particle, coating blemish): areal count density and size distribution | `PROCESS` per kind: intensity + size `DIST` + clustering/correlation structure | count/m²; µm | size distribution ≥15 bins | count 10–10⁵/m² | O, or M where archival reproduction is in scope | pos, frame, age |
| UNI.5 | Scratch and abrasion statistics: length, width, depth-equivalent density, orientation, occurrence rate | `PROCESS` + `DIST`s | mm, µm, density | — | — | O | frame, age |
| UNI.6 | Processing artefact statistics: drying marks, streaks, uneven development, chemical stains | `PROCESS` + `MAP` exemplars | density | — | — | O | P |
| UNI.7 | Aging and fade kinetics: density loss and colour shift per unit storage time under stated conditions | `CURVE_FAMILY` vs log time | relative density | ≥5 time points; ≥2 condition sets | — | O | t, T, RH, λ |
| UNI.8 | Base physical degradation: shrinkage, vinegar-syndrome progression, curl | `CURVE`s vs time | %, — | — | shrinkage 0–3 % | O | t, T, RH |
| UNI.9 | Whether each defect population is a property of the material, of the handling, or of the specific scanned copy | `TOKEN` per record | — | — | — | R where UNI.4–UNI.6 exist | — |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| UNI.1 | MF.2 batch | low-frequency multiplicative and additive fields; the reason large flat areas of real film are never perfectly even, which is a strong realism cue at large image scales and is invisible in any per-pixel metric | direct | yes | AD3 |
| UNI.4–UNI.6 | FMT.2 for areal-to-frame conversion; UNI.9 | stochastic defect generation, sampled from the stored distributions with a deterministic seed so that a frame renders identically on every pass | direct | **yes** (the measured histograms, not only fitted parameters) | AD3 |
| UNI.7–UNI.8 | COL.5 | archival reproduction of aged material; also the explanation for why a modern scan of old stock does not match the datasheet | direct | yes | AD2 |
| UNI.9 | — | separates the material's intrinsic properties from the condition of one particular print — a distinction that, if lost, permanently attributes one archive copy's dirt to the film stock | direct | — | — |

**Note on UNI.9.** This is a data-hygiene requirement with real consequences. Defect
statistics are almost always derived from scanning aged material, which conflates
three sources: what the manufacturer coated, what handling did to it, and what
happened to that one copy. Storing the attribution keeps a scratch statistic from
becoming a property of the emulsion, and lets a caller ask for a clean render of a
material whose only surviving measurements came from a damaged print.

**Note on representation.** Defect populations are the clearest case in this
appendix for a `PROCESS` rather than a parameter list: what is needed is not "3 %
of pixels are dust" but an intensity measure, a size distribution, and a
clustering structure — because real particulate defects are spatially clustered,
and a spatially uniform Poisson placement at the correct mean count looks
distinctly unlike real dirt.

---

### A.5.15 Temporal characteristics (motion-picture materials)

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| TMP.1 | Grain temporal independence: the correlation between the grain pattern of successive frames | `SCALAR` (with justification) or `CURVE` vs frame lag | correlation | — | 0 for camera original; up to ~1 for a fixed pattern | M for motion picture | frame |
| TMP.2 | Registration variation (gate weave): displacement statistics in both axes, with their temporal power spectrum | `DIST` + `PSD` per axis | µm; µm²·frame | ≥300 frames measured; PSD ≥20 bins | σ 1–20 µm | X for motion picture | frame, format |
| TMP.3 | Frame-to-frame exposure variation (shutter and transport-induced flicker) | `DIST` + `PSD` | log H | ≥300 frames | σ 0.002–0.05 log H | O | frame |
| TMP.4 | Breathing / focus and magnification instability in the gate | `DIST` | % magnification | — | 0–0.3 % | O | frame |
| TMP.5 | Native frame rate and shutter angle actually used for the reference material | `SCALAR`s | frames/s, degrees | — | — | R | — |
| TMP.6 | Whether each temporal quantity describes the camera, the printer, the projector or the scanner transport | `TOKEN` | — | — | — | M where TMP.2–TMP.4 exist | — |
| TMP.7 | Generational history for print and intermediate materials: which materials and processes preceded this one in the chain | ordered list of refs | — | — | — | R for print/intermediate | — |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| TMP.1 | GRN.1, GRN.3 | forces a new grain realisation per frame for camera original, and a *fixed* pattern where the grain belongs to an earlier generation — getting this wrong is immediately visible in motion and invisible in a single frame | direct | — | AD2 |
| TMP.2 | FMT.1 (perforation and pitch), TMP.6 | sub-pixel per-frame displacement; the spectrum matters because weave is temporally correlated, and independent per-frame jitter at the correct amplitude looks like noise rather than like a mechanism | direct | yes | AD4 |
| TMP.6 | every temporal record | prevents a scanner's transport instability being modelled as the camera's | direct | — | — |
| TMP.7 | all of A.5.7–A.5.12 for each generation | the chain determines cumulative MTF, cumulative grain and cumulative contrast; a print stock's own data describes only its own step | direct | — | AD1 |

**Note on TMP.1, which costs nothing to store and is often wrong.** For a camera
original, each frame's grain is an independent realisation — the film moved. For a
duplicated or printed image, the grain of the earlier generation is *fixed in the
image* and moves with the frame content, while the print material adds its own
independent layer. A single Boolean-equivalent per generation, correctly set,
separates two renderings that look entirely different in motion, and TMP.7 is what
makes it possible to set it at all.

---

### A.5.16 Environmental dependencies

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| ENV.1 | Sensitivity and contrast change with pre-exposure storage: temperature, humidity, elapsed time | `GRID[3]` or a set of `CURVE`s | Δlog H, Δγ | ≥3 temperatures × ≥3 times | Δlog H 0 to −1 | O | T, RH, t |
| ENV.2 | Exposure-time ambient temperature effect on response | `CURVE` | Δlog H vs °C | ≥3 points | — | O | T |
| ENV.3 | Humidity effect on emulsion swell and hence on layer thickness, scatter and resolution | `CURVE`s | µm, response | ≥3 points | — | O | RH |
| ENV.4 | Post-processing storage conditions of the specific measured sample | condition record | °C, %, days | — | — | R | t, T, RH |
| ENV.5 | Recommended and extreme storage limits as published | `INTERVAL`s | °C, % | — | — | O | — |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| ENV.1–ENV.3 | SEN.1, RES.1 | corrections when simulating material used outside nominal conditions, and the explanation for a systematic offset between a datasheet and a measurement of aged stock | direct | yes | AD2/AD3 |
| ENV.4 | every measurement | part of the uncertainty budget of the measurement itself; an expired or warm-stored sample is a different material from the datasheet's | direct | — | AD3 |

---

### A.5.17 Measurement chain, calibration and reference information

**These are adjacent entities, referenced by film measurements and stored
separately from them** (A.4). A film profile must never contain a scanner's
properties.

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| CAL.1 | Densitometer definition: spectral products of each channel, geometry (diffuse/specular), aperture, and the standard it conforms to | `SPECTRUM` per channel + `TOKEN`s | relative | 5 nm | — | M | λ |
| CAL.2 | Scanner spectral responsivity per channel, including the illumination spectrum and any IR channel | `SPECTRUM` per channel | relative | 5 nm, 350–750 nm (+ IR band) | — | M | λ |
| CAL.3 | Scanner opto-electronic conversion function: code value versus transmittance or density, per channel | `CURVE` per channel | code value vs density | ≥20 levels spanning the full range | — | M | — |
| CAL.4 | Scanner MTF, per channel and per axis | `CURVE` | response vs cycles/mm at the film plane | ≥15 points | — | R | f |
| CAL.5 | Scanner flare / veiling PSF and its integrated amplitude | `PSF` + `SCALAR` | relative vs µm | log-radial, ≥30 points | 0.0005–0.02 integral | R | pos |
| CAL.6 | Scanner geometric calibration: sampling pitch at the film plane, aspect, distortion, and the mapping from pixel to millimetres | `SCALAR`s + `MAP` | µm/pixel, mm | — | — | M | pos |
| CAL.7 | Scanner noise floor and its dependence on signal level | `CURVE` | σ code value vs level | ≥10 levels | — | R | D |
| CAL.8 | Calibration reference materials used: step wedge identity, its certified densities and uncertainties, calibration date | object + `CURVE` + σ | density | per step | — | M | t |
| CAL.9 | Instrument state history: calibration dates, lamp changes, drift measurements | `SEQUENCE` | — | — | — | R | t |
| CAL.10 | Which measurement chain produced each film measurement record | reference from every record | — | — | — | M | — |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| CAL.1 | SEN.3 | defines what any stored density number means; combined with COL.1 to convert dye spectra into the density any instrument would read | direct | yes | AD2 |
| CAL.2–CAL.3 | COL.1, SEN.1 | converts simulated spectral transmittance into the scanner code values the comparison is made against — **the step that makes a simulation comparable to a real scan at all** | direct | yes | AD3 |
| CAL.4–CAL.5 | RES.1, OPT.2 | separates the scanner's blur and flare from the film's, so that fitting the film's MTF does not silently absorb the scanner's | direct | yes | AD4 |
| CAL.6 | FMT.2, GRN.2 | the pixel↔millimetre mapping every spatial quantity depends on | direct | — | AD3 |
| CAL.8 | — | traceability; the uncertainty floor of every density in the archive | direct | yes | AD2 |
| CAL.10 | — | staleness detection: recharacterising a scanner must invalidate the derived models that depended on it (A.7) | direct | — | — |

**Note on why this group is class M despite describing no film.** §4.36 is blunt
about the consequence of skipping it: fitting a model to an uncharacterised scan
fits the scanner as well as the film, and the resulting profile is correct only on
that scanner. Concretely, an uncharacterised scanner MTF gets absorbed into RES.1
and makes the material appear softer than it is; uncharacterised flare gets
absorbed into OPT.2 and misstates halation; an uncharacterised OECF distorts the
entire characteristic curve. Each of these errors is invisible until the reference
scanner changes, at which point every profile in the database is wrong by an
unknown amount and there is no record of by how much.

---

### A.5.18 Validation and reference data

| ID | Stored content | Representation | Units | Sampling | Valid range | Class | Varies with |
|---|---|---|---|---|---|---|---|
| VAL.1 | Reference scans of exposed and processed material: step wedges, colour targets, resolution targets, uniform patches, edge targets, real scenes | `BLOB` set + full condition records | — | at the scanner's native resolution, undebayered/unsharpened, linear where possible | — | M | P, batch, format |
| VAL.2 | For each reference scan, the scene-referred description of what was photographed: target identity, patch reflectances or transmittances, illuminant spectrum, exposure | structured object + `SPECTRUM`s | — | per patch | — | M | — |
| VAL.3 | Expected metrics computed from each reference scan: per-patch density, granularity per patch, measured MTF, measured noise spectrum | derived `CURVE`s and `SCALAR`s | — | — | — | M | — |
| VAL.4 | Acceptance tolerances per test, as agreed for the fidelity target | `SCALAR`s | density, %, ΔE, dimensionless | — | per §1.1 | M | — |
| VAL.5 | Historical results: every simulation run against each validation set, with the model versions used and the resulting deviations | `SEQUENCE` of result records | — | — | — | R | t |
| VAL.6 | Known-failure catalogue: cases where the model is known not to match, with the suspected cause | list of `{case, deviation, hypothesis, refs}` | — | — | — | R | — |

| ID | Depends on / co-required | Consumed as | Store | Raw | AD |
|---|---|---|---|---|---|
| VAL.1–VAL.2 | CAL.*; PRC.1; MF.2 | the ground truth against which any fidelity percentage is computed; without it no accuracy claim in this document is measurable | direct | **yes — this is raw data by definition** | AD3 |
| VAL.3 | VAL.1 | regression targets | derived | — | — |
| VAL.4 | §1.1 | pass/fail; converts "90–95 %" from an aspiration into a test | direct | — | — |
| VAL.5 | model versions (A.6) | detects regressions and attributes them to a specific model change | direct | — | — |
| VAL.6 | — | prevents rediscovering the same limitation, and marks which gaps are data gaps rather than model gaps | direct | — | — |

**Note.** VAL.1 is the only group in this appendix that cannot be obtained from any
document, at any price, for any material — it must be shot, processed and scanned.
It is also the group without which every number in §1 is unverifiable. A database
that stores film parameters but no reference imagery can be internally consistent
and still be wrong by an unknown margin, permanently.

---

### A.5.19 Provenance and confidence

Not a separate content group: provenance is a mandatory component of **every**
record in A.5.1–A.5.18, specified in A.8. It is listed here only to state that
there is no record type in this design that may omit it — including derived
records, whose provenance is the set of records they were derived from plus the
identity of the generator.

---

## A.6 The derived model layer (layer 2)

### A.6.1 What a model record contains

```
ModelRecord
├── model_id
├── model_type            TOKEN, e.g. "curve.canonical_resample",
│                         "curve.parametric", "mtf.kernel_decomposition",
│                         "noise.spectrum_to_filter", "psf.radial_to_separable",
│                         "spectral.integrated_response", "crosstalk.matrix_for_chain"
├── subject               same addressing as a MeasurementRecord
├── inputs[]              record_ids of every measurement used — complete, ordered
├── generator
│   ├── name, version, code_revision
│   └── parameters        every knob that affected the result
├── output                typed per A.2 (commonly CURVE, MATRIX, LUT, FUNCTION, PSF)
├── fit_diagnostics
│   ├── rms_residual, max_residual, residual_units
│   ├── residual_vs_axis  CURVE — where in the domain the fit is worst
│   ├── n_points, free_parameters
│   └── exceeds_measurement_uncertainty  BOOL, computed against the input's σ
├── validity_domain       outside which the model must not be evaluated
├── assumptions[]         TEXT list of every physical assumption made
├── created_on
└── regenerable           BOOL — must be true; false indicates a schema violation
```

### A.6.2 Rules the model layer must obey

1. **Deleting every model record must lose nothing.** A rebuild from layer 1 must
   reproduce them. This is the property that makes future algorithm work possible
   without re-measurement, and it should be tested periodically, not assumed.
2. **`fit_diagnostics` is mandatory and `exceeds_measurement_uncertainty` must be
   computed.** A fit whose residual exceeds the measurement's own uncertainty is
   discarding real information, and the database should be able to list every such
   model in one query. That list is the agenda for the next round of model work.
3. **Multiple models over one measurement set coexist.** A characteristic curve
   may simultaneously have a monotone spline resample, a parametric fit and a
   direct table; different simulation approaches select different ones, and
   comparing them is a legitimate experiment rather than a migration.
4. **`assumptions[]` is not decoration.** The MTF-minus-scatter decomposition of
   A.5.12, the aperture-scaling law of GRN.7, the Selwyn referral in FMT.6 and the
   Gaussian-product identity commonly used to combine blur terms are all
   assumptions that live in the model layer. Recording them is what lets a later
   worker find every model that depended on an assumption later shown to be wrong.
5. **A model may not reach outside its declared inputs.** If a generator needs the
   scanner's spectral responsivity, that record must appear in `inputs[]` — which
   is what makes the staleness detection of A.7 work at all.

### A.6.3 Fallbacks must be model records, not hidden defaults

Much of what A.5 requires will be unavailable for any given material. The correct
handling is a **model record of a fallback type**, carrying the assumption
explicitly:

* granularity known at one density only → `model_type: "granularity.shape_from_class"`,
  inputs = the one measurement plus a class-typical shape, `confidence_class: C3`,
  assumption recorded;
* MTF truncated at 50 % → `"mtf.tail_extrapolation"`, with the extrapolation law
  named and the extrapolated region marked in the output;
* halation unmeasured → `"psf.halation_from_geometry"`, computed from BAS.2, BAS.3
  and Fresnel reflectance, `C4`;
* no noise spectrum → `"noise.spectrum_from_rms_and_grain_size"`, `C4`.

This is the mechanism that keeps the difficulty of acquisition from corrupting the
archive. The fallback is visible, attributed, individually replaceable when real
data arrives, and — critically — it is **countable**, so a profile can report how
much of its rendering rests on measurements and how much on inference. That report
is what §1's fidelity claim must be qualified by.

---

## A.7 The runtime layer (layer 3)

* Built by a bake step from layer 2 only; never hand-edited; not a source of truth.
* Numeric precision reduced to what the engine needs — float32 lanes, uniform-grid
  `LUT`s, pre-multiplied and pre-normalised kernels, structure-of-arrays layout.
* **Carries a content hash of the full closure of records it derives from**
  (measurements, models, generator versions, measurement chains). Any change
  upstream — including a scanner recalibration or a corrected chemistry
  specification — makes every dependent runtime artefact detectably stale.
* Carries the resolved condition set it was baked for: material, processing state,
  batch, format, target scanner. A runtime artefact is a *snapshot of one point in
  the data cube*, and mixing two of them is a bug the hash should catch.
* Must be discardable and rebuildable in a single automated step.

---

## A.8 Provenance

### A.8.1 The source registry

Sources are top-level objects, cited by many records:

```
Source
├── source_id
├── type            manufacturer_datasheet | technical_bulletin | published_standard |
│                   journal_article | book | patent | internal_measurement |
│                   private_communication | derived_from_other_records
├── issuer / authors, title, edition or revision, publication_date
├── identifier      standard number, DOI, ISBN, patent number, internal report id
├── access          URL or archive location, local path, checksum, retrieval date
├── rights          redistribution status of the content
├── language, and for translated sources the original and the translation used
├── reliability     assessment note: primary measurement, secondary citation,
│                   marketing material, uncorroborated
└── scan_quality    for scanned documents: legibility, whether curves are
                    reproducible at usable accuracy, presence of a text layer
```

### A.8.2 The per-record reference

```
ProvenanceRef
├── source_id
├── locator            page, figure, table, plate, curve label, verbatim caption
├── extraction_method  transcribed | digitised_from_chart | computed_from_other_data |
│                      measured_in_house | estimated
├── extracted_by, extracted_on
├── extraction_artefacts  BLOB: the digitiser session, axis calibration points,
│                         cropped source image
└── verification       independent reviewer, date, method, outcome
```

Three requirements on this structure:

* **The locator must be specific enough to return to.** "Datasheet" is not a
  provenance; "figure 4, lower curve, tungsten exposure" is. A worker three years
  later must be able to reopen the source and check the number without guessing.
* **The extraction artefacts must be kept.** A digitised curve can be re-digitised
  at higher density, or corrected for a misread axis, only if the crop and the axis
  calibration survive. This is the difference between a re-digitisation taking ten
  minutes and taking a day of archive searching.
* **Verification is a separate act from extraction, by a different person where
  possible, and its absence is recorded rather than implied.** Transcription errors
  in unverified numbers are the most common defect in a database of this kind and
  the hardest to find later, because a wrong number that is physically plausible
  produces a plausible wrong rendering.

### A.8.3 Confidence classes

Every record carries exactly one, and the vocabulary is fixed:

| Class | Meaning |
|---|---|
| **C1** | Directly documented manufacturer data, or an in-house measurement of the material itself. |
| **C2** | Supported by documented equivalence between stocks or formats — the equivalence itself being a cited `EquivalenceAssertion`, not an inference. |
| **C3** | Inferred from related materials (same manufacturer, era, class or family). |
| **C4** | Theoretical assumption, estimate, or physically-motivated default. |

Rules:

* **A record's class may never be better than the weakest input it depends on.**
  A parameter derived from a C3 measurement is C3 at best, and the database should
  compute this rather than trusting a hand-entered class.
* **No parameter may be transferred between subjects without an
  `EquivalenceAssertion` and a class demotion.** This is the structural
  implementation of "do not silently propagate technical parameters from one film
  stock to another": the transfer becomes a record with a citation, a scope (which
  quantities the equivalence covers) and a direction, and it is visible in every
  report thereafter.
* **Shared standards, shared designations, shared manufacturers and shared eras are
  not equivalence.** They are grounds for a C3 inference at most, and the
  distinction must be enforced by the schema rather than remembered by the analyst.
* **Profile-level and render-level class summaries are required outputs**: how many
  of the quantities driving this rendering are C1, and which C3/C4 records carry
  the most influence. A fidelity claim built substantially on C4 records is not
  the same claim as one built on C1 records, and only the database can tell the
  difference.

---

## A.9 Precision policy

| Layer | Numeric type | Rationale |
|---|---|---|
| Evidence (1) | **binary64 throughout**, plus a stored `significant_digits` per record | Not because measurements have 15 digits — they have 2 to 4 — but because the archive is re-interpolated, re-fitted and unit-converted repeatedly over its life, and binary64 keeps those operations from accumulating error onto data whose own precision is already the limiting factor. At archive scale the cost is nil (A.13). `significant_digits` is what prevents a derived value from being reported to a precision the source never supported. |
| Model (2) | binary64 for fitting; output in binary64 | Fit residuals must be resolvable well below measurement σ, otherwise the diagnostics of A.6.2 cannot distinguish a fit's error from the fitter's. |
| Runtime (3) | **binary32**, uniform-grid tables, precomputed reciprocals | Per-sample arithmetic precision far exceeds what a density or transmittance value needs; the destination is 16-bit at most. |

Additional rules:

* **Wavelength grids** stored as either an explicit array or `(start, step, count)`
  in binary64 nanometres — never as an implicit convention.
* **Angles** in radians internally, but stored in the unit the source used with the
  unit recorded, and converted in layer 2.
* **Logarithmic quantities** stored in the base the source used, with the base
  recorded. Silent base-10/base-e confusion in exposure and density is a
  catastrophic and easily-made error.
* **Dates** ISO-8601; **counts** signed 32-bit; **identifiers** stable strings,
  never re-used, never derived from a mutable name.
* **Every numeric field carries a unit token with a dimensional signature**, so a
  unit mismatch is a load-time error rather than a rendering artefact. Density,
  log-exposure and dimensionless response are all "just numbers" without this, and
  they are the three quantities most often mixed up.

---

## A.10 Completeness levels

A material is not either "in the database" or "missing". It occupies a level, and
the level bounds what may be claimed about it (§1.1, §8):

| Level | Content | What is achievable |
|---|---|---|
| **L0** | Identity, class, nominal speed, production period (A.5.1) | Catalogue entry only. Must not be rendered. |
| **L1** | + characteristic curve at one processing state, base+fog, density metric, exposure illuminant | Tone reproduction under one condition. T1 testable. |
| **L2** | + spectral sensitivity per layer, dye spectral densities, MTF, granularity-versus-density, format geometry, measurement chain | Colour and spatial behaviour under one condition. T1–T2 testable. |
| **L3** | + curve/granularity/MTF families over ≥4 processing states, interimage coupling, reciprocity, halation, mask density, temporal class | The ~90 % target is reachable. T3 testable. |
| **L4** | + noise power spectra, grain morphology and cross-channel correlation, edge traces, inhibitor diffusion, non-uniformity statistics, layer stack geometry, aperture-scaling verification | The 90–95 % range is reachable. T4 testable. |

**The level must be computed by the database and attached to every render**, and a
requested accuracy claim above the material's level must be refused rather than
silently produced. This turns §1's "quote the test with the number" from a
discipline into a mechanism.

---

## A.11 Validation the database must enforce at load and ingest

Structural — reject the record:

* mandatory conditions absent and not listed in `conditions_unknown[]`;
* density record without a densitometric convention (A.5.7 SEN.3);
* granularity record without an aperture (GRN.2);
* MTF record without its definition (RES.2);
* sensitometric record without an exposure illuminant or a processing state;
* any numeric field without a unit token;
* a `FUNCTION` representation in layer 1 without `source_is_authoritative`;
* a curve whose stored sampling is finer than the instrument's stated resolution.

Physical — flag for review, do not silently correct:

* characteristic curve non-monotonic beyond measurement σ;
* density below base+fog, or above the stated Dmax;
* MTF rising at high frequency, or exceeding ~1.5;
* noise power spectrum with negative values;
* spectral sensitivity with no maximum inside the measured range;
* a probability distribution that does not normalise;
* granularity that does not fall as aperture area grows.

Cross-record consistency — report the disagreement with both records named:

* speed computed from SEN.1 under the criterion in SEN.4 versus the published
  speed in ID.6;
* integral spectral density (COL.6) reconstructed from individual dye spectra
  (COL.1);
* RMS granularity (GRN.1) reconstructed by integrating the noise spectrum (GRN.3)
  over the aperture (GRN.2);
* resolving power (RES.3) against the MTF (RES.1) at the stated target contrast;
* contrast descriptors (SEN.5) recomputed from the curve over the stated interval;
* two records of the same quantity differing by more than their combined
  uncertainty (A.1.4);
* a measurement dated outside the material's production period (ID.8).

These cross-checks are the main defence against transcription error, because each
one is an independent route to a quantity that is also stored directly. They should
run as a batch report over the whole archive, not only at ingest.

---

## A.12 Future-proofing — what must not require redesign

| Future need | How this structure absorbs it |
|---|---|
| More film stocks | New emulsion/coating/product nodes. No schema change. |
| More formats of an existing material | New Product node under the existing Coating; format-bound measurements attach there; A.1.5 prevents contamination between gauges. |
| Still and motion-picture materials in one database | Already unified: still materials simply have no A.5.15 records, and the class of every temporal quantity is conditional on the product type. |
| Higher-resolution spectral data | A new `SPECTRUM` record supersedes; the old one is retained with its own sampling. Nothing assumes a fixed λ grid. |
| Higher-resolution sensitometric curves | Same mechanism. Because `CURVE` stores explicit x arrays, no grid is baked into the schema. |
| Spatially varying characteristics | `MAP` records under the existing envelope; a quantity can move from scalar to map by adding a record, not by changing a type. |
| Temporal behaviour | `SEQUENCE`, `PSD` and `PROCESS` records under the existing envelope. |
| Processing-dependent everything | Already the primary axis: `processing_state_id` is a condition on every record, so any quantity can become a family without a type change. |
| Multiple measurements of one property | Native (A.1.4), including conflicting ones. |
| Alternative models over the same data | Native (A.6.2 rule 3). |
| Improved algorithms | Add a `model_type` and a generator. **Layer 1 does not change.** This is the whole purpose of the three-layer split. |
| A new quantity nobody has thought of | Add a quantity-registry entry with its units, conventions and validators. No schema migration, because records are keyed by quantity token rather than by field name. |
| A new measurement instrument or scanner | New measurement-chain object; existing records keep pointing at the old one; dependent models become detectably stale. |

Two mechanisms make the table above true rather than aspirational:

* **`schema_version` plus forward-compatible unknown-field retention.** A reader of
  an older version must preserve fields it does not understand on write, or a
  round-trip through an old tool silently strips new data.
* **The quantity registry is data, not code.** Units, allowed conventions, valid
  ranges and validators live in the registry. Extending the catalogue is an
  ingestion task; extending it in code is a release.

---

## A.13 Serialisation and physical layout

Recommended split, because the three layers have different access patterns:

| Content | Format | Reason |
|---|---|---|
| Registries, sources, identity, conditions, processing states, equivalences, all scalars and small vectors | Human-readable text (a structured, diffable format), one file per material plus shared registry files | Diffable, reviewable, mergeable, greppable, survivable. The metadata is where errors are made and where review happens. |
| Curves, spectra, PSFs, spectral density families, histograms | Human-readable alongside the metadata **when small**; a numeric array container when large, referenced by path and checksum | A hundred-point curve is worth having in plain text; a 2-D PSF is not. |
| Raw scans, target images, digitiser exports, instrument dumps | Content-addressed `BLOB` store, referenced by checksum | Immutable, deduplicating, and detached from the metadata's revision history. |
| Runtime artefacts | Packed binary, versioned by input closure hash | Load speed; disposable. |

**On volume, since it removes the usual argument for reducing curves.** A fully
specified material at level L4 as catalogued above amounts to roughly
1.5 × 10⁴ numbers — about **120 kB in binary64**. One hundred materials is around
**12 MB**: smaller than a single raw scan of a single frame. The raw imagery
(VAL.1, digitiser sources, instrument dumps) is what needs storage planning, at
10²–10³ MB per material.

There is therefore **no storage justification anywhere in this design for reducing
a measured curve to a handful of parameters.** Reduction is justified only inside
layer 3, only for inner-loop performance, and only with the full data retained
upstream.

---

## A.14 Where the difficulty actually lies

Distribution of the ~133 catalogued quantities by acquisition difficulty and class:

| | Count | Notes |
|---|---|---|
| Class **M** (mandatory) | ≈43 | Mostly AD1–AD3. Nearly all obtainable from documentation plus in-house sensitometry. |
| Class **R** (needed for 90 %) | ≈42 | Mostly AD1–AD3, with the processing-family requirement (SEN.2, PRC.3, PRC.9) being the largest single work item and requiring processing capability rather than documents. |
| Class **X** (needed beyond 90 %) | ≈20 | **Predominantly AD4–AD5.** GRN.3, GRN.5, EMU.8, OPT.1–OPT.3, RES.4, PRC.5, PRC.6, TMP.2. This is the same set §7.2 identified as unpublished — the correspondence is not a coincidence, it is the definition of the 90-to-95 gap. |
| Class **O** | ≈28 | Mixed; individually valuable for specific scenarios. |

Two honest observations about this distribution:

* **The M and R classes are mostly an ingestion and representation problem, not a
  discovery problem.** The data largely exists in documentation; what is required
  is a schema that can hold it with its conditions intact and a pipeline that
  consumes it.
* **The X class is a measurement programme.** It needs a microdensitometer or an
  equivalently characterised high-resolution scanning path, controlled processing,
  and access to unexposed stock — for historical materials, sometimes an
  impossibility that no schema can remedy. The recommendation is not to omit these
  quantities but to make their absence *explicit and countable* through A.6.3 and
  A.10, so that the accuracy claim tracks the evidence instead of the intention.

---

## A.15 What this appendix deliberately does not decide

* **Storage technology.** Relational, document-oriented or file-based are all
  compatible with A.4; the normalisation and the layering are the requirements, the
  engine is not.
* **The specific parametric model families** used in layer 2. That is an algorithm
  decision, and the entire point of the three-layer split is that it can be revised
  without touching the archive.
* **Interpolation schemes** between processing states, batches or exposure levels.
  These are layer-2 concerns and must be recorded as `assumptions[]` where used.
* **Which materials to acquire first.** A programme decision informed by §8.
* **The resolution policy's default ordering.** A.1.4 requires that it be data;
  choosing it is an operational decision, and different projects will weight batch
  match against recency differently.

**One-sentence summary of the blueprint.** Store measurements — not parameters —
each with its conditions, units, uncertainty and citation, keyed by a condition
tuple rather than a field name, at a three-level material hierarchy that forbids
sideways transfer without cited evidence; derive every model from those
measurements through a regenerable layer that records its own residuals and
assumptions; bake a disposable runtime layer from that; and let the database
compute, per material and per render, how much of the result rests on measurement
and how much on inference.

---
---

# Appendix B – How to improve the existing film database

**Appendix B created: 2026-08-12, 17:45 local (IDT, UTC+3) — 2026-08-12T14:45:02Z.** *(Corrected 2026-08-13: originally stamped 2026-08-11 from a sandbox clock running a day behind.)*

## B.0 What was audited, and how

Appendix A defined the target. This appendix compares the **actual current schema**
against it and says what is missing, what is mis-shaped, and in what order to fix
it. Everything below is measured, not recalled.

**Subject of the audit:** `PYTHON/profile_generator/film_profiles.py` (authoring
source), the generated `film_profiles.hpp` / `film_profiles.cpp`, and all
`Algo_*.cpp` / `Algo*.hpp` stage code including the `AVX2/` variants.

**Method.** Every leaf field of every profile structure was enumerated
(113 fields across 15 structures). For each, the stage code was searched for a
member access (`.field` or `->field`) after stripping all comments, so a field
mentioned only in documentation does not count as consumed.

**Method limitations, stated because they bound the conclusions:**

* A field copied by whole-struct aggregate assignment and then read through a
  differently-named reference would be counted as unread. Spot checks found no
  such case, but the search is textual and cannot prove their absence.
* `film_sim.py` reads through accessor methods (`profile.mtf.f50s()`,
  `.clumps()`), so a field-level search there undercounts badly and its result is
  **not** reported. The C++ figure is the reliable one, because the stages read
  struct members directly.
* An earlier audit in this project counted 130 leaf fields with 54 unread. This
  pass counts 113 with 51 unread. **The difference is the enumeration boundary**
  — whether `PrintStock`, `FilmFormat` and pure identity strings are included —
  not a change in the code. Neither count is wrong; they measure slightly
  different sets.

**Scale of the current database:** 100 film profiles, 5 print stocks, one flat
format table, 53 profiles carrying spectral-sensitivity arrays.

---

## B.1 The four findings that matter

> **UPDATE 2026-08-13.** Finding 1 below is **partly closed and partly
> corrected**. The spectral curves now drive colour-temperature balance in both
> implementations, which is a clean win (§D.6). But the other two consumers —
> monochrome weights and the taking matrix — turned out to have been
> investigated and correctly quarantined on 2026-08-03 for a reason this
> appendix did not record: projecting a sensitisation curve onto three visible
> primaries derives nonsense for extended-red and infrared stocks. Re-deriving
> it independently reproduced the failure exactly. The characterisation
> "the largest gap is unread data" was therefore too simple: part of that
> unread data was unread **on purpose**, and the appendix should have said so.
> See §D.6 and `CHANGES_2026-08-13_spectral_path.md`.

**1. The single largest gap is not missing data — it is stored data that is never
read.** 51 of 113 leaf fields are consumed by no stage. Among them is spectral
sensitivity for 53 of the 100 stocks: `log_s_r`, `log_s_g`, `log_s_b`,
`log_s_pan`, `lambda_start_nm`, `lambda_step_nm` are read by **zero** stages.
Appendix A rates spectral sensitivity the highest-value curve in the entire
database (A.5.6) and the current engine ignores it entirely, using
`spectral_weights` (three floats) and `taking_matrix` (3×3) instead.

**2. Every measured curve has already been reduced to parameters, and the
measurement was not kept.** There is no sampled-curve representation anywhere in
the schema. `ToneCurve` is six floats. `MTFSpec` is three `f50` values.
`HalationSpec` is three radii and three weights. Dye spectral density is a 3×3
matrix. Appendix A §A.5.7 and §A.5.11 argue at length that these reductions
discard information that cannot be recovered — and here it has already been
discarded, because the digitised curves that produced these numbers were not
retained in any machine-readable form.

**3. The processing axis does not exist.** There is no developer, time,
temperature, agitation or process-designation field anywhere in the schema, at any
level. Appendix A §A.5.13 calls the processing state "the index into the data cube
along its most-often-omitted axis" and makes it class **M**. The current database
cannot express *which* development produced a curve, cannot hold two curves for
one stock, and therefore cannot represent push/pull or any non-nominal process.

**4. Provenance exists but is not machine-readable at the level where it lives.**
The `Provenance` struct is **per profile**: one `tier` integer, a source tuple, a
`fitted_from` string and a review date, populated from side tables
(`_PROVENANCE_SOURCES`, 52 entries; `_UNTAGGED_TIER`, 26 entries). Per-field
provenance does exist and is often excellent — but as **source-code comments**.
`film_profiles.py` says so explicitly at line 6638: *"per-field provenance lives
in the comments at each field."* That means the confidence class of any individual
number cannot be queried, reported, or propagated into a fidelity claim, and
Appendix A §A.8.3's rule — a derived value may not be better than its weakest
input — cannot be enforced.

---

## B.2 Category 1 — present, correctly shaped, and consumed

Not the focus of this appendix, but it should be recorded that 62 of 113 fields
are read and that several are well designed for their purpose:

| Area | Fields | Assessment |
|---|---|---|
| Tone transfer | `dmin, gamma, toe_x, toe_k, shoulder_x, shoulder_k` | Monotone by construction (difference of two softplus ramps) and consumed through a 2048-entry LUT built per frame. **The parameterisation is the wrong storage form (B.4.1), but the consumption path is exactly right** — see B.7.3. |
| Grain amplitude | `rms_granularity`, `rms_r/g/b`, `clump_um_r/g/b`, `clump_gain`, `fog_grain`, `anisotropy` | Per-channel amplitude and correlation scale, with a documented energy normalisation. Correct shape for what it models. |
| Halation | `radii_um`, `weights`, `gain_r/g/b`, `threshold_stops` | A three-lobe decomposition is a legitimate *model-layer* form (A.6); the defect is that no measured PSF sits behind it. |
| Emulsion sharpness | `f50_r/g/b`, `adjacency`, `adjacency_um` | Per-channel, physically referred to cycles/mm. Ceiling discussed in B.4.2. |
| Interimage | `a_rg … a_bg`, `iterations`, `density_weighting` | Directional 6-coefficient coupling, iterated. Correct topology; missing the density dependence (B.4.5). |
| Coating non-uniformity | `coating_sigma`, `coating_corr_across_mm`, `coating_corr_along_mm`, `buckle_mtf_loss`, `edge_fog_density`, `edge_fog_mm` | Anisotropic correlation lengths distinguishing across-web from along-web. Genuinely closer to A.5.14 than most of the schema. |
| Format geometry | `FilmFormat`: `width_mm`, `height_mm`, `anamorphic_squeeze`, `perf_pitch_mm`, `perfs_per_frame` | Physically correct, in millimetres, with frame pitch derived in one place. |
| Weave | `weave_amp_x_um`, `weave_amp_y_um`, `weave_hz_corner` | Amplitude plus a corner frequency — a temporal spectrum in compact form, which is the right *kind* of representation. |

---

## B.3 Category 2 — fields that exist, hold real values, and are read by nothing

All 51, grouped by what it would take to consume them. **This is the cheapest
fidelity available anywhere in the project: the data is already in the database and
already sourced.**

### B.3.1 Consumable with no new data and no new interface (highest value)

| Field(s) | Structure | What it holds | Why unused / what consuming it requires |
|---|---|---|---|
| `log_s_r`, `log_s_g`, `log_s_b`, `log_s_pan`, `lambda_start_nm`, `lambda_step_nm`, `criterion` — **PARTLY CONSUMED 2026-08-13** | `SpectralSensitivity` | Sampled log-sensitivity curves for **53 stocks** | **Now drives colour-temperature balance** (`AlgoSpectralBalanceGains`, `spectral_balance_gains`), which needs no primary basis and is therefore immune to the gamut-reach failure. The monochrome-weight and taking-matrix consumers remain off: they project onto three visible primaries and derive nonsense for IR stocks. Originally required a spectral exposure path: ∫S(λ)·E(λ)dλ per layer in place of the fixed `taking_matrix`. Needs a spectral input estimate (§1.3 caps this at ~70–75 % for display-encoded RGB, but that is still far above a fixed matrix). **Largest single unrealised gain in the database.** |
| `sigma_shape_toe`, `sigma_shape_mid`, `sigma_shape_dmax` | `GrainSpec` | Granularity shape versus density, in three bands | The grain stage applies one amplitude per channel. Consuming these turns constant-amplitude noise into density-dependent noise — A.5.10's core requirement. Engine change is a per-pixel multiplier indexed by the density already computed. |
| `mtf_tail_a`, `mtf_tail_f_exp` | `MTFSpec` | Gaussian-core weight and tail exponent — a two-component MTF already fitted | The MTF is evaluated as a pure Gaussian, so the tail these describe is discarded. Consuming them requires a two-lobe blur instead of one; the multi-lobe blur machinery already exists (it is used for halation). |
| `size_sigma_log`, `cluster_um`, `dye_cloud_um` | `GrainSpec` | Grain size log-spread, clustering scale, dye-cloud size | Would shape the noise spectrum rather than only scaling it. Needs the grain synthesis to use more than one correlation length. |
| `grain_frame_correlation` | `TemporalSpec` | Frame-to-frame grain correlation | A.5.15 TMP.1. Costs one blend between the current and a retained field. Wrong-by-default for print/duplicate stocks today. |
| `flicker_pct`, `flicker_hz` | `TemporalSpec` | Frame exposure variation amplitude and rate | A.5.15 TMP.3. Applied as a per-frame exposure multiplier. |
| `dye_fade_c/m/y`, `base_yellowing_d`, `dmin_lift`, `shrinkage_pct`, `scratch_rate_base_per_m`, `scratch_rate_emulsion_per_m`, `dust_area_ppm`, `mottle_amplitude`, `mottle_scale_mm`, `scratch_persistence_frames` | `AgingSpec` | 11 fields of aging and defect statistics | **The entire `AgingSpec` is unread.** Defect amplitudes come from `AlgoControls` (`dustLevel`, `debrisLevel`, `fibreLevel`) — user knobs, not film data. So per-stock defect statistics exist and are overridden by a global slider. Consuming them means the controls scale the stock's own statistics rather than replacing them. |
| `fps_native` | `TemporalSpec` | Native frame rate | Needed to convert `weave_hz_corner` and `flicker_hz` from hertz to per-frame — **which means the temporal spectra currently in use are being interpreted without knowing the frame rate.** See B.4.7. |

### B.3.2 Consumable only with a new engine input

| Field(s) | What it holds | Missing input |
|---|---|---|
| `schwarzschild_p_r/g/b`, `onset_s` | Reciprocity-failure law per channel plus its onset | **Exposure duration.** The render interface has no way to say how long the frame was exposed, so the data cannot be applied (A.5.8 note). Adding one scalar to the control structure unlocks all four fields. |

### B.3.3 Should be consumed as validators, not as render parameters

| Field(s) | Correct use |
|---|---|
| `resolving_power_lp_mm_lowc`, `resolving_power_lp_mm_highc` | A.5.11 RES.3: an upper bound on rendered detail. Today: stored, unread, and unable to catch an `f50` that is inconsistent with the stock's own resolving power. |
| `speed_point_x`, `speed_criterion` | Cross-check the curve against `exposure_index` (A.11). Both stored, both unread. |
| `density_metric`, `referred`, `callier_q`, `mask_encoding`, `criterion` | The densitometric conventions of A.5.7 SEN.3 and A.5.6 SPC.2. **Stored as strings and read by nothing** — so the engine cannot know whether two stocks' curves are in comparable metrics, and mixing them is undetectable. |
| `tier`, `sources`, `fitted_from`, `last_reviewed` | Confidence reporting per A.8.3 and level gating per A.10. Stored, unread. |

### B.3.4 Genuinely inert

| Field(s) | Note |
|---|---|
| `exposure_index` | Marketing speed. A.5.1 ID.6 says it should *not* drive simulation — correctly unread, but it should be feeding a validator. |
| `features`, `trim`, `pattern` | Read nowhere in any `.cpp`/`.hpp`. `features` is used inside the Python authoring layer for coating-tier logic; `trim` and `pattern` appear dead on both sides. |

---

## B.4 Category 3 — fields that exist but are incorrectly defined for the requirement

These are not gaps; they are representations that cannot hold what the requirement
needs, so adding data to them will not help.

### B.4.1 `ToneCurve` — six parameters where a curve is required

Current: `dmin, gamma, toe_x, toe_k, shoulder_x, shoulder_k`, evaluated as a
difference of two softplus ramps.

Against A.5.7 SEN.1 (≥100 points at ≤0.05 log H, retained permanently):

* Fixes the curve's **shape family** for all 100 stocks. Toe curvature, shoulder
  approach rate and any sublayer inflection are whatever the softplus pair
  produces, not what the material does. A.5.7 lists five distinct places a
  published curve carries information that a small parameter set cannot.
* The **digitised source curves are not stored**, so the fit residual against the
  data is unknown and unknowable, and re-fitting with a better model requires
  going back to the original documents.
* No processing dimension: one curve per channel per stock, full stop.
* `gamma` as a stored field invites use as *the* contrast when A.5.7 SEN.5
  requires the interval it was computed over. That interval is not stored.

**Verdict: correct as a layer-3 runtime form, wrong as the authoritative record.**

### B.4.2 `MTFSpec` — a single Gaussian that cannot exceed unity

`MTF(f) = exp(−ln2·(f/f50)²)`, so response is **strictly ≤ 1 by construction**.
A.5.11 RES.1 requires values above 1.0 to be storable and rendered, because many
materials exceed unity between roughly 2 and 20 cyc/mm through adjacency effects.
The current schema handles this by a *separate* mechanism (`adjacency`,
`adjacency_um`, plus `CouplerSpec.edge_strength`), which means the material's
measured MTF is split across three unrelated fields with no record of how the
split was derived, and no way to check the reassembled response against the
measurement. The two fitted tail parameters that would widen the form beyond a
Gaussian (`mtf_tail_a`, `mtf_tail_f_exp`) are present and unread (B.3.1).

### B.4.3 `dye_matrix` and `taking_matrix` — instrument baked into material

Both are `Matrix3`. A.5.9 COL.2 argues this specific case: a crosstalk matrix is a
property of *the film measured by one instrument*, not of the film. Consequences
in the current database:

* Changing the reference scanner invalidates every stock's matrix, with no record
  of which measurement chain each matrix was valid for — because **no measurement
  chain is stored at all** (B.5).
* Optical crosstalk (between layers), chemical interimage coupling and measurement
  crosstalk are three different mechanisms that respond differently; here two of
  them are merged into `dye_matrix` while the third has its own struct.
* Dye spectral densities — the underlying physical quantity from which any matrix
  should be computed — are absent. There is nothing to recompute *from*.

### B.4.4 `base_tint` — three floats where a spectrum is required

`tuple[float, float, float]`. A.5.4 BAS.4 requires a sampled spectral
transmittance, because the base interacts with per-layer spectral sensitivity and
with the scanner's spectral responsivity, and neither interaction is expressible in
three channel-referred numbers. There is also no separation between the support's
own transmittance and the measured base+fog (BAS.6 note) — `dmin` inside
`ToneCurve` conflates support, residual antihalation and chemical fog.

### B.4.5 `InterimageSpec` — density-independent coupling

Six directional coefficients, `iterations`, and a single global
`density_weighting`. A.5.9 COL.4 requires each coefficient to be a **curve against
the inducing record's density** (≥6 levels). One global weight is a linearisation
whose error grows toward both ends of the scale — exactly where colour crossover
becomes visible. Topology is right; dimensionality is one short.

### B.4.6 `HalationSpec` — a model with no measurement behind it

Three radii, three weights, three gains. As a *decomposition* this is a good
runtime form and the multi-lobe blur consumes it efficiently. The defect is
architectural: A.5.12 OPT.2 requires a measured radial profile over 1–2000 µm
spanning four decades of amplitude, retained, with the three-lobe fit derived from
it and its residual recorded. Here the fit is the only artefact, so its accuracy
against any real halation measurement is unknown. Additionally `OPT.6` — what the
measured MTF already contains — has no field, so double-counting between `f50` and
halation cannot be checked.

### B.4.7 `TemporalSpec` — hertz without a frame rate

`weave_hz_corner` and `flicker_hz` are in hertz; `fps_native` exists and is
unread (B.3.1). A temporal frequency in hertz cannot be converted to a per-frame
sequence without the rate, so either the stages assume a fixed rate or the values
are being applied in the wrong units. **This is a concrete correctness question
raised by the audit that I have not resolved and am not going to assert either way
without reading the weave stage in detail.** Flagging it as the highest-priority
verification item in B.9.

### B.4.8 `ReciprocitySpec` — a speed-only law

Schwarzschild exponent per channel plus onset. A.5.8 EXP.1 requires **both** a
speed shift and a separate contrast change (Δγ), because reciprocity failure
changes the curve's shape and not only its position. A pure Schwarzschild law
preserves contrast that real material loses. Also all four fields are unread
(B.3.2).

### B.4.9 `Provenance` — right idea, wrong granularity

Per profile, not per measurement (B.1 finding 4). One `tier` integer covers a
stock whose curve may be datasheet-grounded while its granularity is an analogy
from a sibling stock and its halation is a theoretical estimate. Appendix A
requires a class per record and a computed weakest-input rule; neither is possible
at profile granularity. The `_UNTAGGED_TIER` side table (26 entries) covering
stocks whose descriptions lack a tag is a symptom: provenance is being recovered by
parsing prose.

### B.4.10 Format binding by string

`FilmProfile.default_format` is a string into a flat `FilmFormat` table. Against
A.1.5 there is no Product level, so nothing distinguishes a measurement made on
one gauge from one made on another, and nothing prevents inheriting a format-bound
number across gauges. The `FilmFormat` table itself is well formed; the *binding*
is the defect.

---

## B.5 Category 4 — required and entirely absent

No field, no struct, no placeholder. Ordered by Appendix A class.

| Requirement | A.5 ref | Class | Current state |
|---|---|---|---|
| Processing state as an object (chemistry, time, temperature, agitation) | PRC.1 | **M** | Absent at every level |
| Which processing state a curve belongs to | PRC.1/PRC.2 | **M** | Absent — curves are unlabelled |
| Measurement conditions on any number (illuminant spectrum, exposure duration, instrument, aperture, geometry) | A.3 `ConditionSet` | **M** | Absent. Convention *names* exist as unread strings (B.3.3); the conditions themselves do not |
| Measurement uncertainty, anywhere | A.3 | **R** | Absent — no σ field on any quantity |
| Batch / coating-run identity and observed batch variation | MF.2, MF.3 | **R** | Absent. So the uncertainty floor of §1.2 cannot be stated |
| Measurement chain: scanner spectral responsivity, OECF, MTF, flare, geometric calibration | CAL.1–CAL.10 | **M** | Absent as data. Scanner MTF exists only as a *runtime setting* (`scanner_f50`), so the reference instrument is a knob rather than a characterised object |
| Sampled curve representation of any kind | A.2 `CURVE` | **M** | Absent — no array-valued measured curve in the schema except the unread spectral sensitivities |
| Curve / granularity / MTF families over processing state | SEN.2, PRC.9 | **R** | Absent |
| Dye spectral densities | COL.1 | **M** (colour) | Absent — only the 3×3 matrices derived from them |
| Masking-coupler density versus exposure | COL.3 | **M** (masked neg) | Absent; `mask_encoding` is an unread string |
| Noise power (Wiener) spectrum | GRN.3 | **X** | Absent |
| Granularity measurement aperture | GRN.2 | **M** | **Absent — and this is a hard requirement.** `rms_granularity` is stored with no aperture, which A.5.10 says must be rejected rather than defaulted. A 24 µm figure and a 48 µm figure differ by ~1.4× |
| Grain size distribution as a population | EMU.8 | **X** | Absent (`size_sigma_log` exists but unread, and is a width not a population) |
| Cross-channel grain correlation | GRN.5 | **X** | Absent |
| Layer stack: inventory, order, thickness, sublayer split | EMU.1, EMU.2, EMU.4 | **R/X** | Absent entirely. No layer concept in the schema |
| Support thickness, refractive index | BAS.2, BAS.3 | **R** | Absent — halation geometry has no physical basis to derive from |
| Edge / knife-edge trace | RES.4 | **X** | Absent |
| Inhibitor diffusion range, local exhaustion | PRC.5, PRC.6 | **X** | Absent as data (`CouplerSpec.edge_um` is a fitted proxy) |
| Reference imagery and validation sets | VAL.1–VAL.6 | **M** | Absent. No measured ground truth exists in the project at all |
| Equivalence assertions as first-class records | A.1.5 | **M** | Absent — equivalence lives in prose comments |
| Three-level Emulsion/Coating/Product hierarchy | A.1.5 | **M** | Absent — flat profile list |
| Multiple coexisting measurements per property | A.1.4 | **M** | Structurally impossible — one value per field |
| Explicit unknown vs not-applicable | A.1.3 | **M** | Absent. Encoded by sentinel zeros (`resolving_power… = 0 means unknown`, `mtf_tail_f_exp = 0 means unused`), which conflates "unknown", "not applicable" and "measured as zero" |
| Completeness level, computed | A.10 | **M** | Absent |
| Load-time validators | A.11 | **M** | Absent |
| Environmental dependencies | ENV.1–ENV.5 | **O** | Absent |
| Generational chain for print/duplicate stocks | TMP.7 | **R** | Partial — `default_print` names a print stock, but no processing state or cumulative-chain record |

**On sentinel zeros.** This deserves separating out because it is a pattern rather
than a single omission. `0 = unknown` appears in at least four places
(`resolving_power_lp_mm_lowc/highc`, `perf_pitch_mm`, `mtf_tail_f_exp`,
`scanner_f50`). It works, it is documented at each site, and it is still the exact
failure A.1.3 describes: the schema cannot distinguish a measurement that came back
zero from a measurement that was never made. Migrating this is cheap and should
happen with the evidence layer.

---

## B.6 Group-by-group verdict

| A.5 group | Fields present | Fields consumed | Verdict |
|---|---|---|---|
| A.5.1 Identification | 8 | 5 | Good. Missing production period as structured dates, and lineage links |
| A.5.2 Manufacturing | 0 | 0 | **Absent** |
| A.5.3 Format | 5 (separate table) | 5 | Well formed; binding by string is the defect |
| A.5.4 Base | 1 (`base_tint`) + `dmin` | 2 | **Severely under-represented**: no thickness, index, spectral transmittance, antihalation |
| A.5.5 Emulsion / layer stack | 0 | 0 | **Absent** |
| A.5.6 Spectral | 7 | **2 of 7, from 2026-08-13** | `log_s_*` + `lambda_*` now drive colour-temperature balance in both builds. `criterion` still unread. The remaining consumers need a scene spectral model, not more wiring — see §D.6 |
| A.5.7 Sensitometric | 6 + 6 convention strings | 6 | Parametric only, single processing state, conventions unread |
| A.5.8 Exposure response | 4 | **0** | Stored, unusable without an exposure-duration input; contrast term missing |
| A.5.9 Colour | 2 matrices + 8 interimage | 10 | Wrong representation (B.4.3), no dye spectra, no density dependence |
| A.5.10 Grain | 16 | 10 | Amplitude right; density dependence and morphology stored-but-unread; **aperture absent** |
| A.5.11 Spatial resolution | 9 | 5 | Gaussian-only, cannot exceed unity, tail fitted but unread |
| A.5.12 Optical | 7 (halation) | 7 | Model with no measurement behind it; no turbidity, no crosstalk PSF, no double-count guard |
| A.5.13 Processing | 0 | 0 | **Absent — the most consequential single gap** |
| A.5.14 Non-uniformity / defects | 6 coating + 11 aging | 6 | Coating good; all 11 aging fields unread and overridden by controls |
| A.5.15 Temporal | 9 | 3 | Weave consumed; correlation, flicker, rate unread; unit question in B.4.7 |
| A.5.16 Environmental | 0 | 0 | **Absent** |
| A.5.17 Measurement chain | 0 | 0 | **Absent** — scanner is a runtime knob |
| A.5.18 Validation | 0 | 0 | **Absent** |
| A.5.19 Provenance | 4 (per profile) | **0** | Present at wrong granularity, machine-unreadable per field, unread |

---

## B.7 What to actually do

### B.7.1 The reframing that makes this tractable

**`FilmProfile` is already a competent layer-3 runtime artefact.** It is flat,
narrow, parametric, cheap to load, and shaped for the inner loops. Almost every
defect in B.4 is a defect *of using it as the authoritative record*, not a defect
of the struct.

So the recommendation is **not** to restructure `FilmProfile`. It is to **stop
authoring at layer 3**:

```
today:      hand-authored film_profiles.py  ──generates──>  hpp/cpp  ──>  engine
                    (this IS layer 3, authored by hand)

target:     evidence archive  ──generators──>  model records  ──bake──>  film_profiles.hpp/cpp  ──>  engine
            (new, layer 1)                     (new, layer 2)            (unchanged output format)
```

Properties of this path:

* **The engine does not change on day one.** The generated `.hpp`/`.cpp` keeps its
  current shape; only its provenance changes. This is testable: the first
  generator run should produce byte-identical output to the current file.
* Every hand-entered number in `film_profiles.py` becomes a layer-1 measurement
  record with the conditions and citation *already written in the comment beside
  it* — the comments are unusually thorough and are the raw material for the
  migration. Line 6638's "per-field provenance lives in the comments" is a
  migration asset, not only a defect.
* Engine improvements then become opt-in per stage, one at a time, each with a
  measurable accuracy delta.

### B.7.2 Phase 0 — consume what is already stored (no new data, no schema change)

Ordered by expected fidelity gain per unit of work. All of it uses data already in
the database and already sourced.

| # | Change | Data used | Cost | Expected effect |
|---|---|---|---|---|
| 0.1 | Spectral exposure path: ∫S(λ)·E(λ)dλ replacing `taking_matrix` | `log_s_*` for 53 stocks | New stage + spectral input estimate | Largest available. Correct illuminant response and metameric behaviour instead of one fitted matrix |
| 0.2 | Density-dependent grain amplitude | `sigma_shape_toe/mid/dmax` | Per-pixel multiplier on an already-computed density | Removes the shadow/highlight noise error A.5.10 describes |
| 0.3 | Wire `AgingSpec` into the defect stages; make controls scale rather than replace | 11 unread fields | Plumbing | Per-stock defect character instead of one global slider |
| 0.4 | Two-component MTF | `mtf_tail_a`, `mtf_tail_f_exp` | Second blur lobe; machinery exists | Correct high-frequency tail; detail-to-grain transition |
| 0.5 | Grain frame correlation | `grain_frame_correlation` | One blend with a retained field | Fixes print/duplicate stocks, which are wrong by default today |
| 0.6 | Validators: resolving power vs `f50`; `speed_point_x`/`speed_criterion` vs `exposure_index`; density-metric consistency | 8 unread fields | Load-time checks | Catches inconsistent profiles; no render change |
| 0.7 | Confidence reporting per render | `tier`, `sources`, `fitted_from` | Reporting only | Makes any accuracy claim qualifiable per A.10 |
| 0.8 | Exposure-duration control input, then reciprocity | 4 unread fields | One scalar on the control struct | Unlocks EXP.1; still missing Δγ (B.4.8) |

### B.7.3 Phase 1 — the enabler that costs almost nothing

**Stage 8 already builds a 2048-entry LUT from the six curve parameters each
frame.** That means the engine is *already table-driven for the characteristic
curve*. Feeding it a **measured sampled curve** instead of a parametric one is a
change to how the table is filled — the inner loop is untouched, and the
performance is identical.

This is the single cheapest structural improvement available: it converts the
worst representation defect in the database (B.4.1) into a build-time concern.
Requires adding a sampled-curve carrier to the archive and a generator that
resamples onto the LUT grid, with the fit residual recorded per A.6.2.

**The same trick does not work for MTF**, and it is worth being clear about why:
the blur is implemented spatially, exploiting the Gaussian-product identity so that
variances add and no FFT is needed. A measured MTF that is not a Gaussian mixture
cannot be consumed that way. Options are a multi-lobe Gaussian fit with its
residual recorded (cheap, keeps the existing machinery, fits the two-component form
of 0.4) or a frequency-domain path (expensive, changes the performance profile).
**Recommendation: multi-lobe fit in layer 2, measured curve retained in layer 1** —
which is exactly the split Appendix A prescribes and preserves the option to
revisit later.

### B.7.4 Phase 2 — schema additions, engine unchanged

1. **Processing-state objects** and a state reference on every measurement. Even
   with one state per stock initially, this makes the axis expressible and stops
   unlabelled curves accumulating.
2. **Per-field provenance and confidence**, migrated from the comments. Mechanical,
   large, and the enabler for A.8.3's weakest-input rule.
3. **Three-valued absence** replacing sentinel zeros.
4. **Measurement-chain object**, even if the first entry is a rough
   characterisation of whatever scanner produced the reference images. Without it
   nothing else in the archive can be interpreted.
5. **Granularity aperture** on every existing `rms_granularity` value — which will
   require going back to sources for many stocks, and will find some that cannot be
   determined. Those should be marked explicitly unknown, not defaulted.
6. **Emulsion/Coating/Product hierarchy** and equivalence assertions as records.

### B.7.5 Phase 3 — needs measurement, not schema work

Reference imagery first (VAL.1), because without it none of the above can be
scored. Then the X-class items: noise power spectra, grain populations, edge
traces, halation profiles, non-uniformity maps. §A.14 already establishes that this
phase is a measurement programme rather than a data-entry task.

---

## B.8 What not to do

* **Do not widen `FilmProfile` field by field** to close the gaps in B.5. That
  reproduces the present situation at larger scale: more fields, still no
  conditions, still no provenance per number, still one value per property. The
  gaps are structural, and fifteen more floats will not close them.
* **Do not add curve arrays directly into `FilmProfile`.** Sampled curves belong in
  the evidence archive; what reaches the engine is a baked table. Putting hundreds
  of samples into the runtime struct pays a load and cache cost for data the inner
  loop cannot use in that form.
* **Do not delete the parametric fields when measured curves arrive.** They become
  layer-2 model records with residuals — which is how the improvement gets
  measured, and how a regression gets attributed.
* **Do not treat the 51 unread fields as dead weight to be removed.** They are
  sourced data awaiting a consumer, and B.7.2 is built entirely out of them.

---

## B.9 Open verification items from this audit

Recorded rather than asserted, because the audit raised them and did not settle
them:

1. **`fps_native` unread while `weave_hz_corner` and `flicker_hz` are in hertz**
   (B.4.7). Either a fixed rate is assumed somewhere or the temporal spectra are
   being applied in inconsistent units. Needs a read of the weave and flicker
   stages before any claim.
2. **`trim` and `pattern`** appear unread on both the C++ and Python sides. Either
   genuinely dead or accessed by a path the search missed.
3. **Whether any `rms_granularity` value in the database can have its aperture
   recovered** from the sources already cited. This determines whether B.7.4 item 5
   is a data-entry task or a re-sourcing task, and the answer probably differs by
   manufacturer and era.
4. The **method limitations in B.0** apply to every count in this appendix. The
   direction of the findings is not in doubt; individual field classifications may
   shift by one or two on closer reading.

---
---

# Appendix C – Numerical precision policy: float32 by default, higher precision by exception

**Appendix C created: 2026-08-12, 17:54 local (IDT, UTC+3) — 2026-08-12T14:54:29Z.** *(Corrected 2026-08-13: originally stamped 2026-08-11 from a sandbox clock running a day behind.)*

## C.0 The policy

**Stated requirement (project owner, 2026-08-11):**

> Use 32-bit floating-point (`float32`) arithmetic rather than 64-bit
> (`float64`) throughout the simulation. The primary design objective is
> execution performance while maintaining sufficient numerical accuracy for the
> required simulation fidelity. Where performance trades against a small
> numerical error with no meaningful impact on resulting image quality,
> **performance wins**. Where `float32` introduces significant numerical error,
> error accumulation, or loss of precision that could materially degrade the
> accuracy or physical realism of the simulation, **accuracy wins**.

This appendix converts that principle into a test that can be applied to a
specific line of code, registers every site currently claiming the exception,
records the sites checked and cleared, and lists the numeric constants that are
secretly numerical guards.

It supersedes nothing. It is the reasoning behind rule D1 (*float32 lanes;
`HighPrecType` only for setup-time scalars and wide accumulators*), written down
so that the exception list is auditable instead of remembered.

---

## C.1 The operative criterion

The decision is **not** "is this stage important". Importance is uncorrelated with
precision sensitivity, and treating it as the criterion produces double arithmetic
in visually critical loops that do not need it and float32 in obscure setup code
that does.

The quantitative frame:

| quantity | value |
|---|---|
| Destination quantum, 16-bit normalised | 1 / 65535 = **1.53 × 10⁻⁵** |
| float32 unit roundoff (2⁻²⁴) | **5.96 × 10⁻⁸** |
| Headroom, one operation vs one output LSB | **≈ 256 ×** |
| Currently accepted end-to-end deviation vs the scalar reference | **3.2 × 10⁻²** (8.17 DN at 8-bit) |

**Conclusion that drives the whole policy: a single float32 rounding is roughly six
orders of magnitude below anything the image can express.** float32 therefore never
fails because of one rounding. It fails only where a mechanism *amplifies* rounding,
and there are exactly five such mechanisms (C.2). If none of them is present, the
answer is float32 and no further analysis is required.

**Corollary, which is the part most often got wrong:** "this stage is physically
delicate" is not a reason for double. "This expression amplifies rounding by a
factor of N" is. The first is a feeling; the second is a number.

---

## C.2 The five amplification mechanisms

### M1 — Unbounded growth

A quantity increases monotonically without limit, so its absolute representable
spacing grows with its magnitude. Absolute positions along a clip, cumulative
phase, elapsed time, running frame-indexed offsets.

*Worked case — the along-film web offset.* A 35 mm four-perf frame advances
4.75 × 4 = 19.0 mm. One thousand frames is 19 000 mm. 19 000 lies in binade 2¹⁴, so
float32 spacing there is 2¹⁴⁻²³ = 2⁻⁹ = **1.95 µm**. Gate-weave amplitudes are
1–20 µm. The quantisation is therefore **10–200 % of the effect being modelled**,
and the coating field — whose spatial frequency multiplies this offset to form a
cosine argument reaching thousands of radians — decorrelates frame to frame
entirely.

This is not a rounding error. It is a destroyed quantity.

### M2 — Large-N accumulation

Naive summation of N terms accumulates relative error √N·u for random signs and
N·u worst case, u = 5.96 × 10⁻⁸.

*Worked case — a full-frame mean.* N = 1920 × 1080 = 2 073 600.
√N·u = **8.6 × 10⁻⁵**; N·u = **0.12**. For positive-only summands (which a
transmittance or density plane is) the behaviour trends toward the worst case, not
the random walk. A mean feeding a normalisation converts that into a **systematic
gain error** on the whole frame.

Mitigations, in order of preference: wide accumulator (double) — chosen here;
pairwise/blocked summation; Kahan compensation.

### M3 — Cancellation

Subtracting nearly equal quantities. Relative error is magnified by
|a| / |a − b|. Bits lost ≈ log₂ of that ratio.

### M4 — Refinement below mantissa resolution

An iterative narrowing whose interval falls below the representable spacing simply
stops converging. float32's 24-bit significand stalls bisection after ~24 halvings;
every step beyond ~25 is a no-op that returns the same answer.

### M5 — Exponent-range exhaustion

*Worked case — Planck radiance.* λ = 500 nm expressed in metres gives
λ⁵ = 3.13 × 10⁻³², which float32 can represent (min normal 1.18 × 10⁻³⁸). But the
exponential argument reaches 53, so exp(53) = 1.04 × 10²³, and the quotient
3.13 × 10⁻³² / 1.04 × 10²³ = **3 × 10⁻⁵⁵ — flushes to zero in float32** while being
an ordinary number in float64. Sixty decades of span across one expression, six
calls per frame.

**Nothing outside M1–M5 justifies double in a per-pixel loop.** A site claiming the
exception must name its mechanism.

---

## C.3 Exception register — sites where float32 is insufficient

Every entry verified by reading the current tree, not from recollection. All are
already protected.

| Site | Mechanism | Amplification | Status |
|---|---|---|---|
| `AlgoFilmCoord` — along-film origin, `frameIndex × framePitchMm` | **M1** | spacing 1.95 µm at 19 m, against 1–20 µm effects | `HighPrecType`. Correct narrowing pattern, see C.6.1 |
| `Algo_04` coating-field phase synthesis (~10 in-loop sites) | **M1** | cosine argument to 10³ rad; low bits gone in float32 | `HighPrecType`. **Sanctioned exception**: evaluated per pixel but setup-domain in the D1 sense |
| `AlgoPlaneMean` accumulator | **M2** | N = 2.07 × 10⁶; 8.6 × 10⁻⁵ to 0.12 | Wide accumulator, double. Explicitly sanctioned by D1 |
| Grain reference-energy integral, `ALGO_GRAIN_INTEGRAL_N = 16001` | **M2** | 16 001 terms; ≈ 9.5 × 10⁻⁴ worst case, feeding a variance normalisation | `HighPrecType acc`, both scalar and AVX2 paths |
| `AlgoSolveAnchors`, `ALGO_ANCHOR_BISECTIONS = 60` | **M4** | steps 26–60 are no-ops in float32 | double |
| `planckRadiance` | **M5** | 60 decades of span in one expression | double, 6 calls/frame |
| Pyramid-upsample interior bounds (`ceil`/`floor` on a centre of 3.4999999) | **M4**-adjacent: the integer must land correctly | one bit decides an index | double at the hoisted site; runs once per call after the third-pass hoist |
| `buildGaussianKernel` | not a precision requirement — a **reproducibility** one | — | double, so scalar and AVX2 derive identical taps and remain comparable |
| SplitMix64 counter RNG, 64-bit multiply | **not floating point at all** | — | Integer, exact, via `_mm256_mul_epu32` and `a·b = a_l·b_l + ((a_l·b_h + a_h·b_l) ≪ 32)`. **Must never be emulated in float** — the omitted `a_h·b_h` term is exact only in integer arithmetic |
| Particle placement in the 9b / 16 defect fields (position, radius, alpha, phase) | per-particle, computed once | — | double. Setup-time by D1; also what keeps a particle in the same place on every render of a frame |
| `AlgoSpectralSensitivity.cpp` — every integral in the file (added 2026-08-13) | **M5**, and M3 at the long-wavelength end | Planck's λ⁵ in metres ≈ 1e-32 over an exponential reaching 53: the quotient **flushes to zero in float32**. Separately the stored curves are log values spanning 4–5 decades, so exponentiating them produces linear values across five | `HighPrecType` throughout, `expm1` for the M3 end. Setup domain: ~60 integrals per frame, never per pixel, so the cost is unmeasurable |

---

## C.4 Cleared register — checked, and float32 is sufficient

Recorded because a cleared site is as useful as a flagged one: it prevents the same
analysis being redone, and prevents defensive doubles being added later "to be
safe".

| Site | Candidate mechanism | Analysis | Verdict |
|---|---|---|---|
| Pyramid variance compensation `σ_lo = √(σ² − (k²−1)/12) / k` | M3 | At the operating threshold σ = 3.5, k = 2: 12.25 − 0.25 = 12.0. Operand ratio 1.02 : 1, **no bits lost.** Degrades only below σ ≈ 0.6, which the threshold forbids | float32 safe — **but see C.7, the threshold is a numerical guard** |
| Density ↔ transmittance round trip | M3 | At D = 4, T = 10⁻⁴; relative 5.96 × 10⁻⁸ gives ΔD = 5.96 × 10⁻⁸ / ln 10 = **2.6 × 10⁻⁸** | float32 safe by 3 orders |
| Scene-linear exposure across 8 decades | M5 | Eight decades is **exponent** range; float32 provides 38. Mantissa is untouched by dynamic range | float32 safe. A common false alarm |
| Characteristic-curve LUT index, `(logH − lo) × invStep` | M3 | Operands well separated by construction (domain covers the transition plus ten knee widths) | float32 safe |
| Interimage fixed-point iteration, up to 12 curve evaluations per pixel | M2 | A contraction mapping: per-step error is not compounded, it is attenuated toward the fixed point | float32 safe |
| Accumulate-mode blur, `dst += w · result` over lobes | M2 | Three terms. Also, weight 1.0 is exact in IEEE-754, which is why the public path is **bit-identical** to the previous non-accumulating one | float32 safe, and exactly so |
| Halation three-lobe weight normalisation | M2/M3 | Three terms, all positive, similar magnitude | float32 safe |
| Spectral integration ∫S(λ)·E(λ)dλ | M2 | ~80 samples at 5 nm over 340–730 nm; √N·u ≈ 5 × 10⁻⁷ | float32 safe (relevant when A.5.6 is consumed — see B.7.2 item 0.1) |

---

## C.5 Decision procedure for a new site

Applied in order. Stop at the first answer.

1. **Is it integer or bit-exact logic?** Keep it integer. Never migrate exact
   integer arithmetic into floating point for vectorisation convenience.
2. **Does the quantity grow without bound over a clip?** (M1) → double, and narrow
   to a bounded local range as early as possible (C.6.1).
3. **Does the expression sum more than ~10³ terms?** (M2) → wide accumulator. Above
   ~10⁵ terms, a wide accumulator is mandatory regardless of what the result feeds.
4. **Does it subtract quantities whose ratio to their difference exceeds ~10²?**
   (M3) → either double, or algebraically reformulate to avoid the subtraction.
   Reformulation is preferred: it is faster *and* more accurate.
5. **Does it refine an interval by more than ~24 halvings, or iterate to a tolerance
   below 10⁻⁷ relative?** (M4) → double, or reduce the iteration count to what
   float32 can actually deliver.
6. **Does any intermediate span more than ~30 decades?** (M5) → double, or rescale
   the units so the intermediate lands near 1.
7. **Otherwise → float32.** No further justification needed, and none should be
   requested.

Two procedural rules that go with it:

* **A double in a per-pixel loop must name its mechanism at the site** — "M1: web
  offset unbounded along the clip" — not "for precision". An unnamed double is a
  defect, because nobody can later tell whether it is load-bearing.
* **Cleared sites get a comment too.** C.4's value is that the analysis is not
  repeated and that a later reader does not "fix" a correct float32 by widening it.

---

## C.6 Required idioms

### C.6.1 Narrow *after* the subtraction, never before

The correct pattern for an M1 quantity, as implemented in `AlgoFilmCoord`:

```
// along, alongMin are absolute web positions in HighPrecType (M1: unbounded
// growth along the clip). The subtraction brings the value into a bounded
// per-frame range; only THEN is it safe to narrow to float32.
const HighPrecType u = along - w.alongMin;      // still double
...                                              // narrow here, not earlier
```

Narrowing before the subtraction quantises the operands at 1.95 µm and the
difference inherits that error. Narrowing after costs one double subtraction per
call and the error vanishes. **Same instruction count, four orders of accuracy.**

### C.6.2 Wide accumulator, float32 lanes

An M2 accumulation does not require the *lanes* to be double. Multiply and
transform in float32; widen only where terms are added. The vector idiom is
float32 products accumulated into double lanes, or blocked summation with a float32
partial per block and a double running total.

### C.6.3 Prefer reformulation to widening for M3

Cancellation is usually an artefact of how the expression was written. `√(a² − b²)`
becomes `√((a−b)(a+b))`; `1 − cos x` becomes `2 sin²(x/2)`; `exp(x) − 1` becomes
`expm1(x)`. These are faster than the double version *and* more accurate than the
float32 version. Widening should be the second choice, not the first.

### C.6.4 Setup-domain is a role, not a location

`Algo_04`'s coating-field phase is evaluated for every pixel yet is correctly
`HighPrecType`, because the quantity it carries is a clip-absolute web coordinate.
Conversely, particle placement in the defect fields runs once per particle and is
double for determinism, not precision. **"Per-pixel" does not imply float32 and
"once per call" does not imply double** — the mechanism decides, and D1's phrase
"setup-time scalars" means *setup-domain quantities*, wherever they are evaluated.

---

## C.7 Numeric constants that are numerical guards

These read as performance or quality tuning parameters and are in fact protecting
one of M1–M5. Changing any of them requires re-running the corresponding analysis.

| Constant | Value | What it guards |
|---|---|---|
| `ALGO_BLUR_PYRAMID_MIN_SIGMA` | 3.5 | **M3** in the variance compensation. At σ = 3.5 the subtraction loses no bits; below σ ≈ 0.6 with k = 2 it begins to. The current value leaves roughly 6× margin. Lowering this threshold for performance reintroduces cancellation |
| `ALGO_ANCHOR_BISECTIONS` | 60 | **M4**. Correct for double; would be ~25 in float32. If the solver is ever narrowed, the count must be reduced with it or 35 iterations are wasted |
| `ALGO_CURVE_LUT_PAD_KNEES` | 10.0 | Clamping exactness. Both curve ends are asymptotically flat, and a softplus is within k·e⁻¹⁰ of its asymptote at ten knee widths — so clamping is exact rather than defensive. Reducing the padding makes it defensive |
| `ALGO_BLUR_NEGLIGIBLE_SIGMA` | 0.20 | Side tap e⁻¹²·⁵ = 3.7 × 10⁻⁶. Chosen relative to the destination quantum, not arbitrarily |
| `ALGO_GRAIN_INTEGRAL_N` | 16001 | Sets the M2 term count. Raising it increases the accumulation error unless the accumulator stays double |
| `ALGO_BLUR_FUSED_MAX_HALF` | 8 | L2 residency, **not** numerical. Listed to mark it as the one constant in this neighbourhood that is purely a performance choice |

---

## C.8 Relationship to Appendix A §A.9

Appendix A assigns **binary64 to the evidence archive** and **binary32 to the
runtime layer**. That is consistent with this policy, and the reason is worth
stating because it looks like a contradiction:

* The archive is **not in any loop**. One hundred materials at full specification is
  ~12 MB (A.13). There is no performance trade to make.
* The justification is *not* that measurements carry 15 digits — they carry 2 to 4.
  It is that archive values are **re-interpolated, re-fitted, resampled and
  unit-converted repeatedly over the project's life**, and every one of those steps
  is an M2 or M3 opportunity applied to data whose own precision is already the
  binding constraint. Keeping the archive wide means the arithmetic never becomes
  the limiting error.
* `significant_digits` per record (A.9) exists so that a derived value is never
  reported to a precision the source never supported — the complementary discipline
  to storing wide.

So: **wide where data is stored and models are fitted; narrow where pixels are
processed.** Layer 2 fitting in double is what makes A.6.2's residual diagnostics
meaningful — a fit residual cannot be distinguished from the fitter's own error
unless the fitter is well below the measurement σ.

---

## C.9 Empirical support

* **D1 alignment pass, 2026-08-11.** Seven files converted from per-pixel double to
  float32. End-to-end deviation over 15 profiles moved
  **3.2039 × 10⁻² → 3.2038 × 10⁻²** — no failures, all values finite. Runtime
  321.3 → 297.7 ms at HD (min of 3, quiet interval). The conversions were free in
  accuracy terms, which is the policy's central claim demonstrated rather than
  asserted.
* **Gate-weave sampler**, `HighPrecType` throughout a function called 6.2 M times
  per frame that only ever read float samples and returned float: **25.3 → 13.0 ms,
  1.95×**, zero accuracy cost. The clearest single instance of the exception being
  claimed where no mechanism was present.
* **Damage path checked separately** (the end-to-end harness disables it): AVX2 vs
  scalar reference, max 2.50 × 10⁻² (6.38 DN), mean 3.22 × 10⁻³. The largest
  deviation is a defect edge pixel — where a smoothstep in float differs from one
  in double — which is the expected and acceptable location for it.
* **Blur harnesses**, three independent brute-force wrapped-convolution references:
  single-lobe 2 × 10⁻⁵, pyramid band 1.83 × 10⁻², multi-lobe 1.44 × 10⁻². The
  pyramid figures are the pyramid's **shape** error, not a precision error — an
  important distinction, since widening the arithmetic would not reduce them.

---

## C.10 Negative results worth keeping

**The monochrome-neutrality defect is not a precision artefact.** C++ produces
saturation 0.0195 on a monochrome stock where the Python reference produces exactly
0.0000, with systematic R−G = −6 × 10⁻⁴ and B−G = +7 × 10⁻⁴. That magnitude is
**four orders above float32 unit roundoff**, and it is *systematic with a consistent
sign*. Rounding is neither large enough nor sign-coherent. Precision is therefore
eliminated as a cause; the defect is an algorithmic channel asymmetry at or after
stage 7. Recorded here so the hypothesis is not re-tested.

**Violation count is not a cost proxy.** `Algo_09` had 36 in-loop `HighPrecType`
occurrences and its conversion produced no measurable change; `Algo_15` had **two**
and produced 1.95× on the stage. Call frequency is the proxy. Consequence for this
policy: a compliance sweep ordered by occurrence count will spend its effort in the
wrong files.

---

## C.11 Outstanding, for full consistency

Three sites remain unconverted. All are per-particle-bounding-box rather than
per-frame-pixel, so none is expected to be significant, and per C.10 the effort
should not be spent blind.

| Site | Occurrences | Expected gain | Note |
|---|---|---|---|
| `Algo_09` fibre / polygon rasteriser — point lists and distance-to-segment per pixel | ~22 | low | Real violation: no M1–M5 mechanism present |
| `Algo_16` gate-defect blob rasteriser | 8 | very low | Stage totals 1.9 ms, so the ceiling is small |
| `Algo_14` | 1 | unknown | Not yet audited individually |

Also outstanding: the **stage-9b rasteriser conversion is verified correct but its
speed effect is unmeasured** — the sandbox had become noisy (303–335 ms spread
against the earlier 298–300) by the time it was made. It is kept for rule
compliance, not on a performance claim, and is **not** counted in the 297.7 ms
figure of C.9.

---

## C.12 What would invalidate this policy

Stated so the analysis can be re-triggered rather than silently outgrown:

* **A wider output destination.** The 256× headroom of C.1 is computed against a
  16-bit destination. A float32 or 32-bit integer output would shrink it, and every
  C.4 clearance would need re-checking against the new quantum.
* **Much longer clips.** M1's worked case assumed ~10³ frames. At 10⁵ frames the
  web offset reaches 1.9 km, float32 spacing there is 125 µm, and any quantity
  *currently* cleared as bounded should be re-examined for indirect dependence on
  absolute position.
* **Multi-frame temporal accumulation.** Nothing in the engine currently sums
  across frames. A temporal filter, motion-compensated accumulation or exposure
  integration over many frames introduces a new M2 site whose N is the frame count
  and whose result is visible directly.
* **Lowering `ALGO_BLUR_PYRAMID_MIN_SIGMA`** below ≈ 0.6, per C.7.
* **Consuming measured sampled curves** (B.7.3). The LUT build moves from
  evaluating a closed form to resampling measured data; the resampling is layer-2
  work and belongs in double, while the resulting table stays float32. The boundary
  moves, the policy does not.

---
---

# Addendum – Current Database vs. Target Requirements

**Addendum created: 2026-08-13, 11:12 local (IDT, UTC+3) — 2026-08-13T08:12:20Z.**

**Scope:** measured comparison of the **actual** current database against the
target model of Appendix A, plus a measured audit of what the **actual**
documents under `PDF/PROFILES` can and cannot supply. Analysis and
documentation only — no database, generator or engine file was modified in
producing this addendum. Vendor- and stock-neutral throughout: every finding
below is a property of the schema or of the corpus as a whole.

---

## D.0 Method, and what the numbers in this addendum mean

**Database measurement.** The live module was loaded and every profile walked
recursively to leaf level. Results: **schema v5, 121 stocks, 134 leaf field
paths, 122 distinct leaf field names.** Population was counted per field as
"holds a non-zero / non-empty / non-default value" across all 121 stocks.
Engine consumption was measured by searching all stage sources (scalar and
AVX2) for member access after stripping comments.

**Corpus measurement.** All **405 PDFs** under `PDF/PROFILES` were opened and
the first 12 pages of each converted to text: **382 carry a text layer, 23 are
image-only.** The 88 JPG page scans of one book and the loose plate images are
counted separately where relevant. Category presence was then measured by
pattern search across that corpus, and every low-count result was inspected by
hand — which mattered, because three categories that looked present were
false positives (D.3.4).

**Honest limits on these numbers.**

* Field-level searching is textual. A field consumed through an alias or a
  differently-named local would be counted unread; spot checks found none, but
  absence cannot be proven this way.
* Corpus search covers the first 12 pages of each document. Data deeper inside
  the three 700-plus-page reference books is under-counted; those are treated
  as "contains, extent unmeasured" rather than given a count.
* Population counts say a value is *present*, not that it is *documented*. The
  two are different, and the gap between them is itself a finding (D.1.3).
* "Obtainable" in D.3 means the quantity is printed or plotted in a document on
  disk. It does not mean it is already extracted, nor that extraction is cheap.

---

## D.1 How far away is the current database?

### D.1.1 Completeness level of the whole database

Against the Appendix A.10 scale, measured field by field:

| Level | Requires | Current status |
|---|---|---|
| L0 | identity, class, speed, period | **met, 121/121** |
| L1 | + characteristic curve, base+fog, density metric, exposure illuminant | **met in substance, but the curve is parametric not measured** (D.2.1); density metric present on 121/121; exposure illuminant **not stored as a spectrum** (D.2.14) |
| L2 | + per-layer spectral sensitivity, dye spectral densities, MTF, granularity-vs-density, format geometry, measurement chain | **partially met**: spectral sensitivity present on **53/121 (44 %)** and consumed by nothing *(SUPERSEDED — measured 2026-08-13 before the spectral path landed the same day; the curves ARE now consumed by the colour-temperature balance path in both builds, and the corpus is 142 stocks, so the ratio is 53/142. The measurement is left as recorded; see the §D.6 follow-up and `CHANGES_2026-08-13_spectral_path.md`)*; dye spectral densities **absent** (a 3×3 matrix stands in); MTF present as **3 scalars**, not a curve; granularity-vs-density present as **3 shape scalars, unread**; format geometry met; measurement chain **absent** |
| L3 | + curve/granularity/MTF families over ≥4 processing states, interimage, reciprocity, halation, mask density, temporal class | **not met**: the processing axis does not exist at any level; interimage present but density-independent; reciprocity present, **unread**, no Δγ term; halation present as a 3-lobe fit with no measurement behind it; mask density absent; temporal class present but **unread** |
| L4 | + noise power spectra, grain morphology, edge traces, inhibitor diffusion, non-uniformity statistics, layer stack, aperture verification | **not met**: none of these exist as fields |

**Position: the database sits between L1 and L2.** Its tone reproduction is
complete and internally consistent; its spectral, spatial and noise
descriptions are present as reduced surrogates; and the entire processing axis,
measurement chain and material-history layer are absent. Appendix A places the
~90 % fidelity target at L3 and the 90–95 % band at L4.

### D.1.2 Structural distance, stated plainly

Three structural properties of the target are absent, and no amount of
field-filling supplies them:

1. **There is no evidence layer.** The authored source *is* the runtime form.
   Every number is a model parameter; the measurement it came from is not
   stored in machine-readable form anywhere. A.1.1 makes this the first
   decision, because it is the one that determines whether future algorithm
   work needs re-measurement.
2. **Conditions are not part of the key.** A profile holds exactly one value
   per property, so it cannot express one material under two processing states,
   two batches or two measurement chains. A.1.2 requires the condition set to
   participate in lookup.
3. **Absence has one encoding, and it is a number.** `0.0` means "unknown",
   "not applicable" and "measured as zero" simultaneously. Measured: sentinel
   zeros carry the "unknown" meaning in at least four places. A.1.3 requires
   these three states to be distinguishable.

### D.1.3 Two measured facts that quantify the distance

**56 of 122 distinct leaf field names (46 %) are read by no engine stage.**
Not a duplicate of the earlier audit: this run is against the current 121-stock
schema. Among the unread are every convention string, every provenance field,
every spectral-sensitivity array, every reciprocity parameter, the granularity
shape triple, the MTF tail pair, and both resolving-power fields.

**17 leaf fields are populated on ZERO stocks** — they exist, are documented in
the schema, and hold their default on all 121 profiles:

```
grain.cluster_um                   temporal.grain_frame_correlation
mtf.mtf_tail_f_exp                 trim
speed_point_x                      reseau (whole struct on 120/121)
aging.dye_fade_c / _m / _y         aging.base_yellowing_d
aging.dmin_lift                    aging.shrinkage_pct
aging.scratch_rate_base_per_m      aging.scratch_rate_emulsion_per_m
aging.dust_area_ppm                aging.mottle_amplitude
aging.mottle_scale_mm
```

The whole of `AgingSpec` — 11 fields — is **both unpopulated and unread**. It
is schema surface with no data behind it and no consumer in front of it. Any
statement that the model "supports aging" is unsupported by the database.

Population distribution overall: **78 fields at 100 %**, 18 at 50–99 %, 14 at
10–50 %, 7 below 10 %, 17 at zero.

---

## D.2 Item-by-item gap analysis

> ⚠ **READ THE CONSOLIDATED 2026-08-18 → 2026-08-25 ENTRY AT THE END OF THIS FILE
> BEFORE ACTING ON ANYTHING IN D.2.** This analysis was written for schema v5 and the
> schema is now **v11**. Its *structural* reasoning has held up well — the layer
> separation, the measurement-conditions-in-the-key rule and the three forms of
> absence were all adopted — but several of its "Current" cells now describe a state
> the project has left. Specifically: **D.2.3** (granularity as a single amplitude)
> is superseded — a σ(D) shape is stored and read on 13 materials; **D.2.4**
> (spectral sensitivity "entirely unused") is superseded — it is integrated and one
> adoption is known to move a render by 0.28 stop; **D.2.2**'s expectation of a
> heavier-tail MTF form was **tested and lost** to a one-parameter law, and its
> hoped-for per-layer constant does not exist in the measurements; **D.2.9**
> (reciprocity "stored, unread") is superseded — read by both renderers since
> 2026-08-23; **D.2.6**'s neutral-invariance premise needed correcting (the
> correction vanishes at the anchor only). Counts of the form "all 121 stocks" or
> "18 fields per stock" refer to the 2026-08-13 database of 121–144 materials, not to
> the current 160. The **priorities** remain sound; the **inventory** does not.

Each item: **Current → Required → Gap → Action → Representation → Priority →
Rationale.** Priority uses Critical (blocks the target outright) / High
(required for ~90 %) / Medium (required for 90–95 %) / Low (scenario-specific).

### D.2.1 Characteristic curve — the primary reduction

* **Current.** 6 scalars per channel (`dmin, gamma, toe_x, toe_k, shoulder_x,
  shoulder_k`), evaluated as a difference of two softplus ramps. 100 %
  populated, 18 fields per stock. No sampled curve stored anywhere.
* **Required.** A.5.7 SEN.1: sampled curve, ≤0.05 log H, ≥5.0 log H span,
  ≥100 points per channel, retained permanently; parametric form allowed only
  as a derived layer-2 model carrying its residual.
* **Gap.** The curve's *shape family* is fixed for all 121 stocks. Toe
  curvature, straight-line extent, shoulder approach and any sublayer
  inflection are whatever the softplus pair yields. The digitised source data
  is not retained, so the fit residual is unknown and re-fitting requires
  returning to the documents.
* **Action.** Add a sampled-curve carrier to the evidence layer; retain the
  parametric set as a derived model with `rms_residual` and `max_residual`
  against the samples. **Do not delete the scalars** — they become the
  measured-vs-model comparison.
* **Representation.** `CURVE` — explicit x array (log H) + y array (density),
  per channel, per processing state; plus per-sample σ where available.
* **Priority.** **Critical.**
* **Rationale.** This is the most-consumed object in the engine and the largest
  single information loss in the database. It is also the cheapest to fix at
  runtime: stage 8 already builds a 2048-entry LUT per frame, so feeding it
  measured samples instead of a closed form changes only how the table is
  filled — identical inner loop, identical cost.

### D.2.2 MTF — three scalars where a curve is required, and a form that cannot exceed unity

* **Current.** `f50_r/g/b` (100 % populated) evaluated as
  `MTF(f) = exp(−ln2·(f/f50)²)`, plus `adjacency`/`adjacency_um`, plus
  `mtf_tail_a` (100 % populated) and `mtf_tail_f_exp` (**0 % populated**), both
  **unread**.
* **Required.** A.5.11 RES.1: sampled curve, ≥20 points/decade log-spaced,
  1 → ≥200 cyc/mm, extended until response < 0.02, values **above 1.0
  storable and rendered**.
* **Gap.** Three defects at once. (a) A pure Gaussian is response ≤ 1 by
  construction, so the real above-unity region between roughly 2 and 20 cyc/mm
  cannot be represented — it is instead approximated by a separate
  `adjacency` mechanism with no record of how the split was derived and no way
  to check the reassembled response against a measurement. (b) The
  high-frequency tail is not representable: the two fitted tail parameters that
  would widen the form are present and consumed by nothing, one of them empty
  on every stock. (c) Nothing records whether the source MTF already contained
  scatter, so MTF and halation can double-count (A.5.12 OPT.6).
* **Action.** Store the measured MTF curve; derive a multi-lobe Gaussian fit in
  layer 2 with recorded residual; add a field stating what the measurement
  already contains; make the tail parameters either consumed or removed.
* **Representation.** `CURVE` on log-spaced spatial frequency, response
  dimensionless, range 0.005–1.5.
* **Priority.** **Critical.**
* **Rationale.** Rendered sharpness and the grain-to-detail relationship both
  depend on it. A Gaussian clamped at unity renders every stock softer than the
  material and hides its acutance.

### D.2.3 Granularity — a single amplitude where a curve exists

* **Current.** `rms_granularity` (100 %), optional per-channel `rms_r/g/b`
  (56/121), and `sigma_shape_toe/mid/dmax` — **100 % populated and read by no
  stage.** No aperture field.
* **Required.** A.5.10 GRN.1: σ-versus-density curve, ≥10 density levels from
  0.2 D to near Dmax, per channel, per processing state — plus GRN.2, the
  measurement aperture, which A.5.10 makes **mandatory and rejectable**.
* **Gap.** Two distinct problems. (a) The density dependence is *already
  parameterised* in three shape scalars that are populated on every stock and
  consumed by nothing — the data is in the database and the render ignores it,
  so noise amplitude is constant with density. (b) **No aperture is stored at
  all.** Granularity scales roughly as the inverse square root of aperture
  area, so mixing 24 µm and 48 µm figures is a ~40 % amplitude error in an
  unknown direction, and the schema cannot detect it.
* **Action.** (a) Consume the existing shape triple — no new data needed.
  (b) Add a mandatory aperture (diameter, shape, effective area, scan step) to
  every granularity record and a validator that refuses a figure without one.
  (c) Add the σ-vs-density curve as the evidence-layer form.
* **Representation.** `CURVE` (σ_D vs density) per channel; aperture as scalars
  with units; retain the shape triple as the derived compact model.
* **Priority.** **Critical** for the aperture, **High** for consuming the
  shape triple.
* **Rationale.** Granularity varies by a factor of two or more across the
  usable density range, so one amplitude is wrong in the shadows and highlights
  in opposite directions — exactly where grain is looked for. The aperture is a
  correctness requirement, not a fidelity one.

### D.2.4 Spectral sensitivity — present, under-sampled, and entirely unused

* **Current.** `SpectralSensitivity` with variable-length arrays — **the only
  sampled-curve representation in the whole schema.** Populated on **53/121
  stocks (44 %)**: 35 colour (r/g/b) + 18 monochrome (pan). Measured sampling:
  **10 nm step on 49 of 53, 20 nm on 1, 25 nm on 3**; 30–33 samples typical;
  *(measured 2026-08-13; as of 2026-08-15 the distribution is 5 nm on 1, 10 nm on
  48, 20 nm on 1, 25 nm on 3 — `FUJI_NEOPAN_1600` was re-traced at 5 nm from its
  own Fuji datasheet, the first stock in the corpus stored finer than 10 nm)*
  wavelength span mostly **380–700 nm**, extremes 380–830 on two stocks.
  **Read by no engine stage.** The engine uses `spectral_weights` (3 floats)
  and `taking_matrix` (3×3) instead.
* **Required.** A.5.6 SPC.1: 2 nm preferred, **5 nm minimum**, 340–730 nm
  minimum, extended to the material's actual limit, stored to the noise floor,
  4–5 decades of dynamic range, with the measurement bandwidth recorded.
* **Gap.** Three gaps stacked. (a) **Consumption**: the highest-value curve in
  the database drives nothing; a fixed channel matrix reproduces the material's
  response to one illuminant and to no other. (b) **Resolution**: 10 nm is
  twice the stated minimum and misplaces a sensitisation-dye peak by up to
  5 nm; 25 nm on three stocks is far outside it. (c) **Range**: 380–700 nm
  omits the UV shoulder and the extended-red region entirely.
* **Action.** Build the spectral exposure path (∫S(λ)·E(λ)dλ per layer)
  replacing the fixed matrix; re-digitise at ≤5 nm where the source plot
  supports it; extend the stored range to the plot's own limits; add the
  measurement bandwidth field and a validator that refuses a stored sampling
  finer than the stated bandwidth.
* **Representation.** `SPECTRUM` per layer — already structurally correct, only
  under-sampled.
* **Status 2026-08-13.** Consumption **partly closed**: the curves now drive
  colour-temperature balance. NOT closed for the channel matrix, and the reason
  is a measured failure rather than remaining work — see §D.6. Resolution and
  range are untouched and still open.
* **Priority.** **Critical** (consumption), **High** (resolution and range).
* **Rationale.** A.5.6 rates this the highest-value curve in the database; §1.3
  caps fidelity at ~70–75 % without a spectral input path regardless of
  everything else. It is the single largest unrealised gain: the data already
  exists for 44 % of stocks.

### D.2.5 Colour crosstalk — an instrument baked into the material

* **Current.** `dye_matrix` and `taking_matrix`, both fixed 3×3, both 100 %
  populated. No dye spectral densities anywhere.
* **Required.** A.5.9 COL.1: spectral density of **each dye individually**,
  2–5 nm over 360–730 nm, at ≥4 density levels; COL.2 derives any matrix
  **per measurement chain** from those spectra.
* **Gap.** A crosstalk matrix is not a property of the film — it is a property
  of the film *measured by one instrument*. Changing the reference scanner
  changes the correct matrix while the material has not changed, and the
  database stores no measurement chain to attribute the matrix to. Worse, three
  physically distinct mechanisms — chemical interimage, optical inter-layer
  scatter, and measurement crosstalk — are collapsed across `dye_matrix` and
  `InterimageSpec`, and they respond to different variables.
* **Action.** Add per-dye spectral density curves as evidence; derive the
  matrix at load time for a named measurement chain; keep the three mechanisms
  in separate structures.
* **Representation.** `SPECTRUM` per dye + `SPECTRUM_FAMILY` over density;
  matrix becomes a derived layer-2 artefact.
* **Priority.** **Critical** for colour materials.
* **Rationale.** Without the underlying spectra there is nothing to recompute
  *from* when the scanner changes — and per A.5.17 it will.

### D.2.6 Interimage coupling — right topology, one dimension short

* **Current.** 6 directional coefficients (67/121 populated), `iterations`, and
  a single global `density_weighting` (14/121).
* **Required.** A.5.9 COL.4: each coefficient a **curve against the inducing
  record's density**, ≥6 levels.
* **Gap.** One global weight is a linearisation whose error grows toward both
  ends of the scale — precisely where colour crossover becomes visible.
* **Action.** Promote each of the six coefficients from scalar to curve;
  retain the scalar as the derived mid-scale value.
* **Representation.** `MATRIX[3×3]` of `CURVE`s (coefficient vs inducing
  density), with the sign convention stored.
* **Priority.** **High.**
* **Rationale.** Interimage effects are how a colour negative separates
  saturated colours without raising gamma; getting their density dependence
  wrong misrenders exactly the saturated regions they exist to control.

### D.2.7 Halation — a model with no measurement behind it

* **Current.** 3 radii + 3 weights + 3 gains + threshold (radii/weights 100 %,
  gains 62/121).
* **Required.** A.5.12 OPT.2: measured radial PSF, ≥60 log-spaced radii,
  1–2000 µm, spanning ≥4 decades of amplitude, retained; the lobe
  decomposition derived from it with residual recorded.
* **Gap.** The three-lobe form is a legitimate *runtime* representation and the
  multi-lobe blur consumes it efficiently. The defect is that the fit is the
  only artefact — its accuracy against any real halation measurement is
  unknown, and a 2-parameter Gaussian fitted to such data typically
  underestimates the halo at 500 µm by more than an order of magnitude.
* **Action.** Add the measured radial profile as evidence where obtainable;
  where not, mark the lobe set as a documented fallback model (A.6.3) with its
  confidence class, rather than leaving it indistinguishable from measurement.
* **Representation.** `PSF` — radial profile, log-spaced radii, amplitude
  stored logarithmically, normalisation convention recorded.
* **Priority.** **Medium** (Appendix A places measured halation in the 90→95 %
  band).
* **Rationale.** Halation is the effect most responsible for the photographic
  rendering of specular highlights, and its heavy tail is what a Gaussian
  cannot reproduce.

### D.2.8 Base and support — one triple standing in for a spectrum and a stack

* **Current.** `base_tint` (3 floats, 100 %) plus `dmin` inside each tone
  curve. No thickness, no refractive index, no antihalation description, no
  layer inventory.
* **Required.** A.5.4 BAS.2–BAS.6 and A.5.5 EMU.1–EMU.10: support thickness,
  refractive index, spectral transmittance of the finished base, antihalation
  type and spectral absorbance with a flag for removal in processing, base+fog
  per channel, and the ordered layer inventory with thicknesses.
* **Gap.** (a) Three channel-referred numbers cannot express a spectral
  interaction with per-layer sensitivity and scanner responsivity. (b) `dmin`
  conflates support, residual antihalation and chemical fog, so the three
  cannot be separated when modelling a changed processing state. (c) There is
  **no layer concept at all**, so halation geometry has no physical basis to
  derive from and inter-layer effects have nowhere to live.
* **Action.** Add support object (material, thickness, index, spectral
  transmittance, antihalation), separate measured base+fog from support
  transmittance, and add an ordered layer-stack structure.
* **Representation.** `SPECTRUM` for transmittance and antihalation absorbance;
  scalars with units for thickness and index; ordered list for the stack.
* **Priority.** **High** for thickness/index/transmittance (they feed halation
  geometry), **Medium** for the full stack.
* **Rationale.** Scatter geometry is set by the emulsion-to-rear-surface
  distance and the Fresnel reflectance at that surface; without them the
  halation model has no physical anchor and can only be fitted.

### D.2.9 Reciprocity — stored, unread, and missing its contrast term

* **Current.** `ReciprocitySpec`: Schwarzschild exponent per channel +
  `onset_s`. **All four fields unread by the engine.** Overrides exist for a
  minority of stocks; the rest carry heuristic defaults.
* **Required.** A.5.8 EXP.1: **two** curves — Δlog H *and separately* Δγ
  versus log duration, ≥3 points per decade over 10⁻⁵–10² s, per channel; plus
  EXP.2, the reference duration at which the main curve was measured.
* **Gap.** (a) Not consumed, and cannot be: the render interface has no
  exposure-duration input, so the data is inert by construction. (b) A pure
  Schwarzschild law is a speed shift only — it preserves contrast that real
  material loses. (c) The reference duration of the stored characteristic curve
  is not recorded, so even a correct law has no origin to apply from.
* **Action.** Add an exposure-duration control input; add the Δγ term; store
  the reference duration on the curve record; promote the law from exponent to
  measured curve where correction points are printed.
* **Representation.** Two `CURVE`s per channel + reference-duration scalar.
* **Priority.** **High** (interface + reference duration), **Medium** (Δγ).
* **Rationale.** Long-exposure behaviour is a visible, characteristic property;
  and the reference duration is a prerequisite for interpreting the curve at
  all, not an enhancement.

### D.2.10 Processing state — the missing axis

* **Current.** **Nothing.** No developer, time, temperature, agitation,
  replenishment or process-designation field at any level of the schema.
  Several stocks' comments record time–gamma data found in sources; it has
  nowhere to go.
* **Required.** A.5.13 PRC.1 (class **M**): processing state as a reusable
  named object referenced by every sensitometric, granularity, MTF and colour
  record; PRC.3 time–gamma; PRC.9 an explicit record of which quantities were
  re-measured per state.
* **Gap.** The database cannot say what produced a curve, cannot hold two
  curves for one material, and therefore cannot represent push/pull or any
  non-nominal process. A.5.13 calls this "the index into the data cube along
  its most-often-omitted axis".
* **Action.** Introduce a top-level `processing_states[]` collection and a
  state reference on every affected record — even if the first release has one
  state per stock, which alone makes the axis expressible and stops unlabelled
  curves accumulating.
* **Representation.** Shared object + reference; curve families become
  `CURVE_FAMILY` over the state axis.
* **Priority.** **Critical.**
* **Rationale.** Everything in D.2.1, D.2.3 and D.2.2 is a function of
  development. Without this axis the three cannot even be known to belong to
  the same material state, and A.5.11 RES.7's "same batch and state?" question
  is unanswerable.

### D.2.11 Measurement chain — absent as data, present as a runtime knob

* **Current.** No densitometer, scanner or calibration object. Scanner sharpness
  enters as a **runtime setting** derived from the selected print stock's
  `mtf_f50`; the engine has no `scannerF50` field of its own (documented in
  the engine source as a known omission).
* **Required.** A.5.17 CAL.1–CAL.10 (class **M**): densitometer spectral
  products and geometry; scanner spectral responsivity, OECF, MTF, flare PSF,
  geometric calibration, noise floor; calibration reference materials; and a
  chain reference on **every** measured record.
* **Gap.** Fitting a model to an uncharacterised scan fits the scanner as well
  as the film. Concretely: uncharacterised scanner MTF is absorbed into the
  stored `f50` and makes the material look softer than it is; uncharacterised
  flare is absorbed into halation; an uncharacterised OECF distorts the whole
  curve. Each error is invisible until the reference scanner changes, at which
  point every profile is wrong by an unknown amount with no record of by how
  much. Additionally, coupling scan sharpness to the print-stock choice means
  selecting a print stock silently changes scan sharpness — a behaviour no user
  can deduce.
* **Action.** Add measurement-chain objects as a top-level collection, separate
  from film profiles; add a chain reference to measured records; give the engine
  its own scanner parameters and break the print-stock coupling.
* **Representation.** `SPECTRUM` per channel for responsivity and densitometer
  products; `CURVE` for OECF and MTF; `PSF` for flare; scalars for geometry.
* **Priority.** **Critical.**
* **Rationale.** This is the step that makes a simulation comparable to a real
  scan at all. Without it no fidelity percentage is measurable.

### D.2.12 Provenance — right idea, wrong granularity, and unread

* **Current.** `Provenance` per **profile**: one `tier` int, a `sources` tuple,
  `fitted_from`, `last_reviewed`. Measured: tiers **44 / 41 / 36** across
  1 / 2 / 3; `fitted_from` splits identically; `last_reviewed` set on 121/121;
  **45/121 carry the explicit "no manufacturer datasheet" placeholder**. All
  four fields **unread by the engine**. Per-field provenance exists and is
  often excellent — as **source-code comments**, stated as such in the
  authoring module.
* **Required.** A.8.2–A.8.3: provenance per **record**, with a specific
  locator (page, figure, curve label), extraction method, extraction artefacts,
  verification, and a confidence class per record with a computed
  weakest-input rule.
* **Gap.** One tier integer covers a stock whose curve may be datasheet-grounded
  while its granularity is an analogy and its halation a theoretical estimate.
  The class of an individual number cannot be queried, so it cannot be reported
  with a render, and A.8.3's "a derived value may not be better than its
  weakest input" cannot be enforced. The prose comments are a migration asset,
  not a substitute.
* **Action.** Promote provenance to per-field; migrate the existing comments
  mechanically; add the locator and verification fields; compute class
  propagation; expose a per-render confidence summary.
* **Representation.** Structured record per measured value + a shared source
  registry.
* **Priority.** **Critical.**
* **Rationale.** Without it, an accuracy claim cannot be qualified by the
  evidence behind it, which §1 requires and which the owner's own evidence-class
  directive makes mandatory.

### D.2.13 Multiple values, batches and material identity

* **Current.** One value per property per stock. No batch identity, no
  observed batch variation, no emulsion/coating/product hierarchy, no
  equivalence records. Format binds by a **string** into a flat table.
* **Required.** A.1.4 (N coexisting records + resolution policy), A.1.5
  (three-level hierarchy, upward-only inheritance, cited
  `EquivalenceAssertion` for sideways transfer), MF.2–MF.3 (batch identity and
  measured variation).
* **Gap.** Conflicting measurements cannot coexist, so a second measurement
  must overwrite the first and the disagreement is lost. Batch variation is
  unrepresentable, and §1.2 establishes that it is comparable to the entire
  remaining error budget at the 95 % target — so no fidelity claim can be given
  a meaningful tolerance. Nothing structurally prevents a format-bound number
  being applied to a different gauge.
* **Action.** Introduce the hierarchy, a status/supersedes mechanism, a stored
  resolution policy, batch objects, and equivalence assertions as first-class
  records.
* **Representation.** Object graph; per-quantity `DIST`/`INTERVAL` for batch
  variation.
* **Priority.** **High.**
* **Rationale.** This is the structural implementation of "do not silently
  propagate parameters between stocks"; today that rule is enforced by
  discipline and prose only.

### D.2.14 Measurement conditions and units

* **Current.** Convention *names* are stored and 100 % populated —
  `density_metric` (visual_iso 48 / status_m 56 / status_a 17), `referred`
  (scan 120 / print 1), `speed_criterion` (four values), `mask_encoding`
  (three), `callier_q` (1.0 / 1.25 / 1.3). **All are read by no engine stage.**
  The conditions themselves — illuminant spectrum, exposure duration,
  instrument, aperture, geometry, temperature — are absent. No field carries a
  unit token. No field carries an uncertainty.
* **Required.** A.3 `ConditionSet` and A.9: conditions as a structured object
  participating in lookup; a unit token with a dimensional signature on every
  numeric; uncertainty with its basis and coverage; `significant_digits` as
  reported.
* **Gap.** The engine cannot know whether two stocks' curves are in comparable
  metrics, and mixing them is undetectable — a systematic error of order
  0.05–0.2 D. The exposure illuminant is a name at best, never a spectrum, so
  "log H" means something slightly different per stock. Without uncertainty
  there is no floor on any accuracy claim, and no weighting for curve fitting.
* **Action.** Add unit tokens with machine checking; add per-record uncertainty
  and significant digits; promote conventions from stored-and-ignored to
  enforced at load; store illuminants as spectra in a shared registry.
* **Representation.** Structured `ConditionSet`; unit enum; σ or σ[] per record.
* **Priority.** **Critical** for units and convention enforcement, **High** for
  uncertainty.
* **Rationale.** Appendix A's blunt version: a density number without its
  convention is not usable, and a granularity number without its aperture must
  be rejected rather than defaulted.

### D.2.15 Relationships and dependencies that must be represented

* **Current.** Dependencies are enforced by author discipline and prose. Nothing
  in the schema links a curve to the granularity and MTF measured with it;
  nothing marks which quantities are format-bound; nothing prevents
  double-counting between MTF and scatter; nothing records which quantities
  were re-measured for a changed state.
* **Required.** A.5 dependency columns and §5's interaction table — in
  particular: curve↔granularity↔MTF co-measurement (RES.7), the
  MTF-contains-scatter flag (OPT.6), format-bound-ness per quantity (FMT.6),
  per-state re-measurement flags (PRC.9), the dye-spectra→matrix derivation
  chain, and generational history for print/intermediate materials (TMP.7).
* **Gap.** Every one of these is a correctness constraint that the database
  cannot express, so violations are silent. The double-counting case is the
  sharpest: a measured MTF already includes turbidity, so applying both blurs
  twice, and nothing in the data reveals it.
* **Action.** Add explicit link fields and derivation records; implement the
  cross-record validators of A.11 as a batch report (speed from curve vs stated
  speed; integral vs per-dye spectra; RMS vs integrated noise spectrum;
  resolving power vs MTF; contrast descriptors recomputed).
* **Representation.** Reference fields + a derived-model layer with declared
  inputs.
* **Priority.** **High.**
* **Rationale.** These cross-checks are the main defence against transcription
  error, because each reaches a stored quantity by an independent route.

### D.2.16 Parameters that should be split

| Current single parameter | Split into | Why | Priority |
|---|---|---|---|
| `curves.*.dmin` | support transmittance + residual antihalation + chemical fog + coupler stain | The four respond differently to a changed processing state; conflated they cannot be modelled separately (D.2.8) | High |
| `mtf.f50_*` (+`adjacency`) | measured MTF curve + derived lobe set + adjacency/edge term + "what the measurement contains" flag | One scalar currently carries linear response, and a second mechanism carries the nonlinear part with no recorded relationship | Critical |
| `dye_matrix` | per-dye spectral densities + derived per-chain matrix | Instrument and material are conflated (D.2.5) | Critical |
| `grain.rms_granularity` | σ-vs-density curve + aperture + system-noise-floor flag | Amplitude, its density dependence and the measurement condition are one number today | Critical |
| `interimage.a_*` (6) | 6 curves vs inducing density | Density dependence is the missing dimension (D.2.6) | High |
| `reciprocity.schwarzschild_p_*` | Δlog H curve + Δγ curve + reference duration | Speed shift and contrast change are different effects (D.2.9) | High |
| `temporal.*` (weave, flicker, dirt, fps) | move to a transport object referenced by format | These belong to the camera/projector class, not the emulsion; measured earlier this project as not scaling with gauge (ratio 1.71 vs a 5.19 width change) | High |
| `spectral_weights` + `taking_matrix` | spectral sensitivity integral computed per illuminant | Both are pre-integrated surrogates for a curve the database already holds (D.2.4) | Critical |
| `halation.radii_um`/`weights` | measured PSF + derived lobes | (D.2.7) | Medium |
| `coating.*` (6 scalars) | non-uniformity map + PSD of the variation | A field, not three correlation scalars (A.5.14 UNI.1) | Low |

### D.2.17 New structures required (none of which exist today)

| Structure | Purpose | Priority |
|---|---|---|
| Evidence layer (measurement records with conditions, units, uncertainty, provenance) | The one decision that makes future algorithm work possible without re-measurement | **Critical** |
| Derived-model layer with declared inputs, generator identity, fit residuals, assumptions | Makes fits auditable and regenerable | **Critical** |
| `processing_states[]` | The missing axis (D.2.10) | **Critical** |
| Measurement-chain objects | (D.2.11) | **Critical** |
| Source registry + per-record provenance | (D.2.12) | **Critical** |
| Emulsion/Coating/Product hierarchy + equivalence assertions | (D.2.13) | High |
| Batch objects + observed variation | Uncertainty floor per §1.2 | High |
| Layer-stack structure | (D.2.8) | Medium |
| Validation sets: reference scans + expected metrics + tolerances | **No measured ground truth exists in the project at all**; without it no accuracy claim is verifiable | **Critical** |
| Completeness-level computation + load-time validators | Refuse claims above the evidence (A.10, A.11) | High |
| Three-valued absence (specified / unknown / not-applicable) | Replaces sentinel zeros | High |
| Noise power spectrum, grain size distribution, cross-channel grain correlation, edge trace, inhibitor diffusion, local exhaustion, environmental dependencies | The L4 set | Medium |

---

## D.3 Can the required data be obtained from `PDF/PROFILES`?

Measured over 405 PDFs (382 with text layers, 23 image-only) plus one 88-page
JPG book scan and several loose plate images.

### D.3.1 Obtainable, in quantity — numeric, directly transcribable

| Required data | Docs found | Form | Sufficient for the target representation? |
|---|---|---|---|
| RMS granularity, single figure | **141** | printed number | Yes for the scalar; **no** for the σ-vs-density curve |
| Granularity **aperture and reference density** | **125** | prose: "diffuse density 1.0; 48 µm reading aperture" | **Yes** — the mandatory GRN.2 condition is stated in the documents and simply not stored |
| Resolving power | **113**, of which **44** print **both** contrasts (1.6:1 and 1000:1) | printed numbers | Yes — the schema field already matches |
| Densitometric convention | **165** | prose: Status M / Status A / diffuse visual | Yes |
| Processing chemistry, time, temperature | **198** | tables | **Yes** — the processing-state object could be populated immediately |
| Reciprocity data | **193** | correction tables, occasionally formulae | Yes for Δlog H; **Δγ is almost never printed** |
| Print Grain Index (colour negatives) | **47** with numeric values | printed numbers | Only as PGI; vendors state it **cannot** be converted to rms and publish no factor |
| Base thickness, numeric | **11** | prose ("120 micrometre ESTAR", "polyester 125 µm", "total layer thickness 16 µm") | Partially — enough to seed a support object for a minority of materials |
| Batch / emulsion identification method | **55** | prose (edge markings, KEYKODE, emulsion + roll numbers) | Describes *where* batch identity is recorded on film, not the batch values themselves |

### D.3.2 Obtainable as **plots** — high-resolution data behind a reduced field

This is the category the request singles out, and it is the largest single
opportunity in the corpus.

| Required data | Docs | Form | Current DB representation | Note |
|---|---|---|---|---|
| Characteristic curve | **117** | plotted curves, several per sheet | **6 scalars** | Digitisable to ≤0.05 log H |
| Spectral sensitivity | **182** | plotted log-sensitivity curves | arrays on 44 % of stocks at **10 nm** | Plots support finer than 10 nm; current sampling is coarser than the stated minimum |
| Spectral **dye** density | **112** | plotted per-dye curves | **absent** (3×3 matrix stands in) | Directly supplies COL.1 — the fix for D.2.5 |
| MTF | **168** | plotted response-vs-frequency | **3 scalars** | Plots show the above-unity region and the tail that the Gaussian cannot express |
| **Granularity σ-vs-density curve** | **31** | plotted, with an explicit reading procedure ("follow to the Granularity Sigma D scale… multiply by 1000") | **1 scalar + 3 unread shape scalars** | The document form *is* the GRN.1 curve form |
| Contrast-index-vs-development-time | **19** | plotted, up to 6 developers per sheet | **absent** | Directly supplies PRC.3 |
| Curve **family over development** | **19** | multiple labelled curves per plot | **absent** | Directly supplies SEN.2 — the L3 requirement |

**Two observations.** First, every one of these is a case where the source is a
curve and the database stores a point or a parameter — the loss is at
ingestion, not in the sources. Second, the two rarest categories (19 docs each)
are exactly the two that unlock L3, and they cluster in the professional
still-film and B&W motion-picture sheets rather than being spread evenly.

### D.3.3 Obtainable only as qualitative prose

| Required data | Docs | What is actually there |
|---|---|---|
| Grain morphology | 64 | "tabular", "T-GRAIN", "cubic crystals" — a class name, never a size distribution |
| Halation / anti-halation | 65 | layer presence, rem-jet or dye type, "high acutance" — never a PSF or a radial profile |
| Interimage / inter-layer effect | **11**, all qualitative | "free control of the interlayer effect", "Color-Correction Layer technology to control the inter-image effect" — no coefficients, no density dependence |
| Aging / dye stability | 102 | dark-keeping statements and one book-length treatment; per-stock kinetics rare |
| Layer architecture | high count, low value | "three emulsion layers separated by gelatine interlayers" — order and presence, no thicknesses |
| Base material | 149 | material names (ESTAR, acetate, polyester, GEVAR); thickness only in the 11 above; refractive index **never** |

### D.3.4 **Not obtainable from this corpus at all**

Each verified by inspecting every candidate hit rather than trusting the count.

| Required data | Search result | Verdict |
|---|---|---|
| **Noise power (Wiener) spectrum** | 3 apparent hits — **all false positives**: "NPS" is a product name (Fujicolor NPS 160), not "noise power spectrum" | **Zero documents.** A.5.10 GRN.3 rates this one of the two highest-value unpublished items; the corpus confirms it is unpublished |
| **Edge / knife-edge trace** | 3 apparent hits — all the marketing word "acutance" in prose | **Zero documents.** Must be measured in-house (A.5.11 RES.4) |
| **Cross-channel grain correlation** | none | **Zero** |
| **Grain size distribution** (population, not class name) | none | **Zero** |
| **Callier Q** | **0 documents**, yet the field is populated on 121/121 stocks with three distinct values | **Zero.** Every stored Callier value is therefore an assumption, not a transcription — a provenance defect the current schema cannot show |
| Base refractive index | 2 hits, both about infrared focusing, not the support | **Zero** |
| Inhibitor (DIR) diffusion range / local exhaustion | none quantitative | **Zero** |
| Coating non-uniformity statistics | none | **Zero** |
| Transport / weave / registration statistics | 8 hits, all about print shrinkage or pitch specs, none per-frame | **Zero** for the temporal model |
| Scanner characterisation (responsivity, OECF, flare, MTF) | 157 hits mention telecine/scanner/densitometer, none characterise a specific chain | **Zero** — must be measured in-house |
| Measured reference imagery (step wedges, targets, real scans) | not a document category | **Zero.** By definition unobtainable from documents; must be shot, processed and scanned |
| Batch-to-batch observed variation | none | **Zero** — requires ≥3 measured coatings |

### D.3.5 Limitations and uncertainties on extraction

* **Digitising a printed plot** achieves roughly **0.02–0.04 D** and
  **0.02–0.05 log H**, limited by line width and axis calibration. That is
  *better* than a four-parameter fit to the same chart, which is the argument
  for doing it — but it is not measurement-grade, and it must be recorded in
  the extraction metadata along with the axis calibration points and the source
  crop, so it can be redone without returning to the archive.
* **23 PDFs and one 88-page book have no text layer.** Their numeric content
  requires OCR or visual transcription; their plots require digitisation
  either way.
* **Edition multiplicity is a feature, not noise.** Several stocks appear in
  4–5 sheet editions spanning ~20 years. These are formulation-revision
  evidence (A.5.2 MF.5) — but they also mean "the datasheet" is ambiguous
  unless the edition is stored, which today it is not.
* **Successor-name traps.** Materials sharing a name across a decade are
  different formulations. Nothing in the current schema records which edition a
  number came from, so a successor sheet can be grafted onto a predecessor
  profile without leaving a trace.
* **Non-manufacturer sources need their conventions read carefully.** National
  standards and handbooks use their own aperture and contrast conventions; the
  project has already had to reject one granularity limit for that reason.
* **Corpus search covered the first 12 pages per document.** The three
  700-plus-page reference works are certainly richer than their counts here
  suggest; they are recorded as "contains, extent unmeasured".

### D.3.6 What the corpus can and cannot buy

**Can reach L2–L3 for a substantial part of the database using documents
already on disk**, because the following are all present in quantity:
characteristic curves (117), spectral sensitivity (182), dye spectral density
(112), MTF (168), resolving power at both contrasts (44), granularity with its
aperture (125–141), processing chemistry and times (198), reciprocity (193),
curve families and contrast-index-vs-time (19 each), σ-vs-density granularity
curves (31).

**Cannot reach L4 from documents at any effort**, because the corpus contains
**zero** noise power spectra, edge traces, grain size distributions,
cross-channel correlations, non-uniformity statistics, transport statistics and
scanner characterisations. Appendix A §7.2 predicted exactly this set; the audit
confirms it empirically. Those require an in-house measurement programme, and
the reference imagery of A.5.18 requires shooting film — no document can
substitute.

---

## D.4 Action items — implementation plan

Dependency-ordered. Each item is independently completable and independently
verifiable. Effort is relative, not calendar.

### Phase 0 — Consume what is already stored (no new data, no schema change)

| ID | Action | Data used | Effort | Priority |
|---|---|---|---|---|
| P0-1 | Build the spectral exposure path (∫S·E dλ per layer), replacing the fixed channel matrix | spectral arrays on 53 stocks | L | Critical — **PARTLY DONE 2026-08-13**: balance path live in both builds; the channel-matrix half is blocked on a scene spectral model, not on wiring (§D.6) |
| P0-2 | Consume the granularity shape triple: density-dependent noise amplitude | 3 fields × 121 stocks, 100 % populated | S | Critical |
| P0-3 | Enforce the convention strings at load: refuse or flag mixed density metrics, speed criteria, mask encodings | 5 fields × 121 stocks | S | Critical |
| P0-4 | Consume the MTF tail parameters (two-lobe response) or remove them | `mtf_tail_a` populated 121/121; `mtf_tail_f_exp` populated 0/121 | M | High |
| P0-5 | Consume `grain_frame_correlation` and the flicker pair | populated but unread | S | High |
| P0-6 | Add resolving power and speed-criterion **validators** (RP vs f50; speed from curve vs stated EI) | 30–49 stocks carry RP | S | High |
| P0-7 | Expose per-render provenance summary (tier mix, count of C3/C4 inputs) | tier/fitted_from on 121/121 | S | High |
| P0-8 | Decide the fate of the 17 zero-populated fields: populate, or remove with the reason recorded | — | S | Medium |

### Phase 1 — Schema additions that do not change the engine

| ID | Action | Depends on | Effort | Priority |
|---|---|---|---|---|
| P1-1 | Introduce the evidence layer alongside the current authored form; first generator run must reproduce today's output byte-identically | — | L | Critical |
| P1-2 | `processing_states[]` object + state reference on every affected record | P1-1 | M | Critical |
| P1-3 | Per-field provenance and confidence class, migrated from the existing comments | P1-1 | L | Critical |
| P1-4 | Unit tokens with dimensional checking on every numeric | P1-1 | M | Critical |
| P1-5 | Measurement-chain objects + chain reference on measured records | P1-1 | M | Critical |
| P1-6 | Mandatory granularity aperture + validator that rejects a figure without one | P1-1, P1-4 | S | Critical |
| P1-7 | Three-valued absence replacing sentinel zeros | P1-1 | M | High |
| P1-8 | Per-record uncertainty (σ, basis, coverage) + `significant_digits` | P1-1 | M | High |
| P1-9 | Emulsion/Coating/Product hierarchy, batch objects, equivalence assertions | P1-1 | L | High |
| P1-10 | Stable profile IDs decoupled from display order | — | S | High |
| P1-11 | Completeness-level computation per stock; refuse accuracy claims above it | P1-3 | S | High |
| P1-12 | Cross-record validators of A.11 as a batch report | P1-1..P1-5 | M | High |

### Phase 2 — Representation upgrades (schema + engine, one quantity at a time)

| ID | Action | Depends on | Effort | Priority |
|---|---|---|---|---|
| P2-1 | Sampled characteristic curve → the existing per-frame LUT; parametric set demoted to a derived model with residual | P1-1, P1-2 | M | Critical |
| P2-2 | Measured MTF curve + derived multi-lobe fit + "measurement contains scatter" flag | P1-1 | L | Critical |
| P2-3 | Per-dye spectral densities; derive the crosstalk matrix per measurement chain | P1-1, P1-5 | L | Critical |
| P2-4 | Granularity σ-vs-density curve as the evidence form | P1-6 | M | Critical |
| P2-5 | Re-digitise spectral sensitivity at ≤5 nm and to the plots' own wavelength limits; add measurement bandwidth | P1-1 | M | High |
| P2-6 | Interimage coefficients scalar → curve vs inducing density | P1-1 | M | High |
| P2-7 | Curve / granularity / MTF **families** over processing state | P1-2, P2-1 | L | High |
| P2-8 | Exposure-duration control input; Δlog H **and** Δγ; reference duration on the curve record | P1-1 | M | High |
| P2-9 | Support object: thickness, index, spectral transmittance, antihalation; split `dmin` into its four contributors | P1-1 | M | High |
| P2-10 | Move transport (weave, flicker, dirt, fps) off the profile onto a format-referenced transport object | P1-9 | M | High |
| P2-11 | Layer-stack structure | P1-1 | M | Medium |
| P2-12 | Halation measured PSF where obtainable; otherwise mark the lobe set as a documented fallback with its class | P1-3 | M | Medium |

### Phase 3 — Extraction campaigns against the existing corpus

Ordered by value per unit effort, using the D.3 counts.

| ID | Campaign | Source count | Unlocks | Priority |
|---|---|---|---|---|
| P3-1 | Digitise σ-vs-density granularity curves | 31 docs | GRN.1 on the most-used modern line; feeds P0-2 with measured shape | Critical |
| P3-2 | Digitise per-dye spectral density curves | 112 docs | COL.1 → kills the baked-in matrix (P2-3) | Critical |
| P3-3 | Digitise characteristic curves at ≤0.05 log H | 117 docs | SEN.1 (P2-1) | Critical |
| P3-4 | Digitise MTF curves including the above-unity region and the tail | 168 docs | RES.1 (P2-2) | Critical |
| P3-5 | Transcribe processing chemistry, times, temperatures into state objects | 198 docs | PRC.1 (P1-2) | Critical |
| P3-6 | Transcribe granularity apertures and reference densities | 125 docs | GRN.2 (P1-6) — pure transcription, condition already printed | High |
| P3-7 | Digitise curve families over development + contrast-index-vs-time | 19 + 19 docs | SEN.2, PRC.3 → **the L3 gate** | High |
| P3-8 | Re-digitise spectral sensitivity finer and wider | 182 docs | P2-5 | High |
| P3-9 | Transcribe reciprocity tables systematically | 193 docs | EXP.1 (P2-8) | High |
| P3-10 | Transcribe base thickness / material where printed | 11 docs | P2-9 | Medium |
| P3-11 | Record sheet **edition** with every extracted number | all | Removes the successor-name trap (D.3.5) | High |

### Phase 4 — In-house measurement (nothing in the corpus can substitute)

| ID | Campaign | Why it cannot be sourced | Priority |
|---|---|---|---|
| P4-1 | **Reference imagery**: step wedges, colour and resolution targets, uniform patches, edge targets, real scenes — shot, processed, scanned, with full conditions | Not a document category. Without it, no fidelity number in this document is measurable | **Critical** |
| P4-2 | Scanner / measurement-chain characterisation: spectral responsivity, OECF, MTF, flare PSF, geometry, noise floor | Zero documents characterise a specific chain | **Critical** |
| P4-3 | Noise power (Wiener) spectra | Zero documents; verified false-positive-only | Medium (L4 gate) |
| P4-4 | Edge traces, both directions | Zero documents | Medium |
| P4-5 | Grain size distributions and cross-channel correlation | Zero documents | Medium |
| P4-6 | Coating non-uniformity fields; transport statistics per gauge | Zero documents | Low |
| P4-7 | Batch-to-batch variation, ≥3 coatings per material | Zero documents; sets the §1.2 uncertainty floor | Medium |

### Phase 5 — Housekeeping exposed by this analysis

| ID | Action | Priority |
|---|---|---|
| P5-1 | Every stored `callier_q` value is an assumption — **0 documents** in the corpus state a Callier Q, yet the field is populated on 121/121. Mark as C4 or retire | High |
| P5-2 | Break the scanner-sharpness / print-stock coupling; give the engine its own scanner parameters | High |
| P5-3 | Decide the fate of `AgingSpec`: 11 fields, zero population, zero consumption. Either wire it to the defect stages and populate from the available aging literature, or retire it and stop implying support | Medium |
| P5-4 | Retire or populate `reseau` (1/121), `trim`, `speed_point_x`, `cluster_um`, `mtf_tail_f_exp` | Medium |
| P5-5 | Duplicate gauge-variant pairs remain in the database (identical numbers, separate entries differing only by default format). Collapse under the transport-on-format change (P2-10) | Medium |

---

## D.5 Summary judgement

**Distance.** The database is at **L1–L2** against a target of L3 for ~90 % and
L4 for 90–95 %. The three missing structural properties — evidence layer,
conditions in the key, three-valued absence — are architectural, not
incremental.

**The cheapest fidelity available is already paid for.** 56 of 122 field names
are unread; among them the granularity density-shape triple (populated on every
stock) and spectral sensitivity (populated on 44 %). Phase 0 requires **no new
data at all** and closes several of the Critical items.

**The corpus is richer than the schema.** Every high-value quantity except the
L4 set exists on disk as curves and tables while the database stores a point or
a parameter: 117 characteristic-curve documents against 6 scalars per channel;
168 MTF documents against 3 scalars; 112 dye-density documents against a 3×3
matrix; 31 granularity-curve documents against 1 scalar. The loss happens at
ingestion.

**Two ceilings are independent of all of the above.** No reference imagery
exists in the project, so no fidelity claim is currently measurable; and no
measurement chain is characterised, so any model fitted to a scan is fitted to
the scanner as well. Both are Phase 4, both are Critical, and neither can be
closed with documents.

**Nothing in the database was changed in producing this addendum.**

---

## D.6 Status update — 2026-08-13, later the same day

**P0-1 is partly done, and the Appendix B finding 1 gap is closed for two of its
three consumers.** The measured per-layer spectral sensitivity curves now drive
the render in both implementations:

* colour-temperature balance (§D.2.14's illuminant problem, partially): the
  three assumed peak wavelengths are replaced by integration of each layer's own
  curve. Derived red gain at 3200 K on daylight stock is 1.65–1.69 against the
  proxy's 1.32, and it varies per stock, which a fixed-peak proxy cannot express.
* the monochrome RGB collapse: implemented, guarded, and left **OFF by
  default** — see the correction below.

**A correction to this entry, made the same day.** The monochrome-weight and
taking-matrix derivations were built, then found to reproduce a failure that an
earlier analysis (2026-08-03, `film_profiles.derived_spectral_response`) had
already documented and quarantined, and which neither Appendix B nor this
addendum had recorded. Projecting a sensitisation curve onto three visible
primaries cannot excite an emulsion sensitised outside them:
`KONICA_INFRARED_750` (peak 730 nm) derives to a **blue-dominant**
(0.161, 0.193, 0.646) against an authored, correct, red-dominant
(0.55, 0.15, 0.30). A guard now refuses that case on two physical criteria
(peak wavelength beyond the basis limit, or more than 15 % of sensitivity
beyond it), and the guard's own blind spot is recorded rather than hidden:
`ROLLEI_INFRARED_400` sits at ~96 % of peak across 660–680 nm, which the 600 nm
lobe reaches poorly, so its derived triple under-weights red (0.35 against an
authored 0.52) while passing both conditions. That residual is not a threshold
to tune — it is evidence that basis projection is the wrong construction. The
earlier decision to quarantine was right, and the honest fix remains a scene
spectral model (reflectance basis functions under a stated illuminant, Smits or
Jakob-Hanika class), built deliberately.

**What survives as a clean win** is the balance path, and it survives precisely
because it projects onto no basis at all: it is a ratio of one curve integrated
against two blackbody SPDs, so the gamut-reach failure cannot apply to it.

**Not done, deliberately:** the derived taking matrix (§D.2.5). It disagrees with
the authored matrix by up to 0.5 in an off-diagonal element, and wiring it in
would double-count mixing already carried by `dye_matrix` and `InterimageSpec` —
the failure D.2.5 and A.5.12 OPT.6 warn about. It is computed and reported;
resolving it requires the measured reference imagery of P4-1, which still does
not exist.

Code: `AlgoSpectralSensitivity.hpp/.cpp` plus stages 03 and 07 in both scalar
and AVX2; `film_sim.py` spectral function block plus three `RenderSettings`
flags. Python and C++ agree to four decimals; the 68 stocks without curves
render bit-identically. Full record in `CHANGES_2026-08-13_spectral_path.md`.

**Follow-up the same day — integration grid.** The grid the integrals are
evaluated on moved from 5 nm to **2 nm** in both implementations after
measurement showed 5 nm was adequate only because every illuminant in the engine
is a blackbody: against a narrow-line source a 5 nm grid is 1.5 % wrong and a
10 nm grid 52.7 % wrong, while 2 nm matches a 1 nm reference exactly. Cost
0.129 → 0.179 ms per derivation, setup domain. **No stored curve was altered** —
resampling stored 10 nm arrays onto a finer grid would interpolate, add no
information, and destroy the record of which samples came from the plot.
Recovering genuinely finer data means re-tracing the source plots (the present
values are a *downsampled* 600 dpi trace, so the information existed and was
discarded at storage), which is a supervised campaign of roughly 100 curve
traces. Priority order and the measured justification per stock:
`CURVE_RESOLUTION_ANALYSIS_2026-08-13.md`.

**Follow-up the same day — corpus growth, and three gap classes newly exercised.**
`PDF/PROFILES/cheltsov_vs_bongard_sa_tsvetnoe_proiavlenie_trekhsloinykh_sv.pdf`
(Cheltsov & Bongard, *Colour Development of Three-Layer Light-Sensitive Materials*,
Moscow 1958) was indexed and mined. The corpus is now **131 film stocks and 9 print
stocks**; every "121 stocks" figure in §D.1–§D.5 above is a measurement taken on
2026-08-13 before this batch and is left as recorded, because rewriting a dated
measurement to match a later state would destroy its value as a baseline.

What the book contributed is disproportionately *documented* rather than authored:
**six balance colour temperatures that existed nowhere in the corpus** (5900, 5400,
5000, 4000, 3450, 3300 K), one published gamma (0.65, Gevacolor 652), one internally
comparable five-stock resolving-power ladder, and per-layer spectral peaks for one
stock. Before this pass the whole corpus used four distinct balance points; the reflex
"daylight = 5500 K, tungsten = 3200 K" turns out to be a modern convention that no
1950s manufacturer followed.

It also **confirms a previously uncited assumption**: `DUPE_FINE_GRAIN`'s unity gamma,
argued from first principles in a source comment, is stated outright for the Soviet
duplicating stock (p177). Assumption promoted to citation with no behaviour change.

Three A-series requirements are now exercised by real data for the first time, and all
three remain unimplementable in the present schema:

* **A.5.6 / layer-resolved MTF.** Table 24 publishes resolving power *per emulsion
  layer* for five films — a 3.7× spread inside one film (Eastmancolor negative
  110/46/30 lp/mm), with the layer light reaches first resolving best in all five
  cases. `MTFSpec` carries three *records*, not three *layers*, and the mapping needs
  the layer order, which is not stored.
* **A.5.x / layer order.** Five documented films deliberately permute their layers to
  put the magenta record on top, with measured sharpness gains (TsP-6 above 200 lp/mm
  against TsP-1's 70). There is no layer-order field at all.
* **PRC / processing axis.** Complete published bath compositions in g/L with times and
  temperatures for six processes, plus Table 14 quantifying how dye gamma diverges from
  silver gamma with development time (ratio 1.32 → 2.86 over 3→8 min). The processing
  axis A.5 specifies remains entirely empty.

One measurement worth recording against §D.6: `balance_kelvin` is **inert at default
settings**. `RenderSettings.wb_strength` defaults to 0 and gates the whole balance
block, so the spectral balance path landed earlier today — and these six new balance
temperatures — change no render until the user asks for a cast. That is the documented
design (`solve_anchors` treats colour-temperature mismatch as a creative control, not a
lab correction), not a defect, but it bounds how much the live spectral path is
currently doing. Full record: `CHANGES_2026-08-13_cheltsov1958.md`.

**Follow-up 2026-08-14 — The Compact Photo-Lab-Index (1979), and two proofs that the
model is under-specified.** `PDF/PROFILES/pittaro_em_the_compact_photolabindex.pdf`
(Pittaro, ed., Morgan & Morgan, 2nd Compact Edition 1979) was indexed and mined. The
corpus is now **142 film stocks and 9 print stocks**. Eleven stocks added — eight
Polaroid types with published D-max, D-min, curve slope, speed and resolving power, and
Ilford Pan F, FP4 and HP4.

Unlike every prior source in this project, this one is a *trade formulary*: it reprints
each manufacturer's own bench procedures. It is therefore strong on speed, curve shape
and processing and contains **no per-layer spectral sensitivity curves for any colour
film**. Nothing spectral was taken from it.

**It confirms four of our field definitions against a published 1970s standard**
(PDF 176–177): RMS granularity as 1000 × σ(D) through a 48 µm aperture at 12×
magnification, with the reversal/negative gross-vs-net measurement-density split we
already observe; resolving power quoted at **two test-object contrasts, 1.6:1 and
1000:1**, which is exactly the pair `_RESOLVING_POWER` stores; and MTF measured at 35 %
aerial image modulation *including* development adjacency effects, which is why
`MTFSpec` carries a separate `adjacency` term. No value changed; four definitions are
now anchored rather than inherited.

**PROOF 1 — `ReciprocitySpec` is structurally insufficient.** Kodak's published
compensation table (PDF 174–175, rebuilt from word coordinates because it does not
survive flat text extraction) gives stops and correction filters at seven exposure times
for twelve colour films. Fitting a Schwarzschild exponent decade by decade:
Kodachrome 40, Kodachrome 25 and Kodacolor 400 all need **p = 0.850 over 1–10 s and
p = 0.699 over 10–100 s** — the exponent steepens. Only Kodacolor II is a true power law,
flat at 0.699. Our single exponent per channel cannot represent a steepening exponent, so
for three of four measured films it is provably the wrong functional form. What the table
*does* supply cleanly is `onset_s` and the **channel ordering**, which the filter colour
documents outright: a CC10R recommendation means the red record lost the most speed. Also
newly evident: Ektachrome 64 needs **+½ stop at 1/10 000 s**, a SHORT-exposure failure for
which the schema has no term at all — `onset_s` is a lower bound only.

**PROOF 2 — `ToneCurve` has a shape-family limit at high gamma.** `POLAROID_51`'s
published slope is 3.35, the steepest stock the database has ever held. Its reversal
transfer overshoots by −9.5 × 10⁻⁶. This was checked rather than assumed: **float64 gives
−9.429e-06 against float32's −9.537e-06**, so it is a property of the curve shape, not of
the arithmetic; and six different toe/shoulder pairs that all land on the published D-max
produce the identical figure, so it cannot be tuned away without abandoning either the
published slope or the published D-max. It is below the 16-bit output quantum
(1/65535 = 1.526e-05) and therefore invisible, and `verify.py` now allows that one named
stock exactly one code value while holding every other reversal stock to the original
tolerance.

**Three additions the schema needs, each documented by this source and homeless today.**

* **Dual daylight/tungsten exposure index.** Published for most of the corpus and
  physically meaningful for monochrome film, because the ratio *is* a compact statement
  of spectral weighting: 1.25 for ordinary panchromatic stocks (FP4 125/100, Plus-X
  80/64), 1.00 for HP4, and **3.2 for the blue-sensitive-only Polaroid Type 51**
  (320/100). Storing one number discards it.
* **A processing specification.** Ilford publishes, for Pan F alone, development times to
  two named contrast indices across three developers at two dilutions, *and* the speed
  each developer produces — ASA 50 in ID-11, ≈80 in Microphen, **32 in Perceptol**. Speed
  varies by 1⅓ stops and contrast by 27 % within one film's own published options, and
  our schema cannot say which condition a stored curve represents. This is the PRC axis
  §A.5 specifies, and it remains empty.
* **Non-blackbody illuminants.** Polaroid Type 410 photographs oscilloscope traces;
  Kodak 5374/7374 photograph television tubes and specify the phosphor as **P-11 (blue)
  or P-16 (ultraviolet)**. A phosphor has no colour temperature, so `balance_kelvin` is a
  placeholder on those materials and is flagged as such in the profile.

**Retained for a future materials extension, not entered.** PDF 559 is an Ilford table of
films *and glass plates* with published sensitivity ranges in Ångströms. Two things
matter. First, **plates are a distinct material class**: on a glass base, buckle, curl and
shrinkage are *not applicable* rather than zero, which would make plates the first real
user of the three-valued absence rule in §A.3. Second, **every material in that table
runs to 2300 Å = 230 nm**, while our spectral integration grid starts at 360 nm — so we
truncate the documented sensitivity of every Ilford panchromatic material by 130 nm.
Harmless today, because glass optics and the atmosphere cut below ~330 nm and no display
primary reaches there, but it means the stored range is not the material's range and any
future UV or scientific extension fails silently at the grid edge. Recorded so the grid
floor is a known decision rather than an accident. Full record:
`CHANGES_2026-08-14_photo_lab_index.md`.

**Follow-up 2026-08-14 (second entry) — schema v6, two corrections, and a dead-test
discovery.** Acting on the three gaps the Photo-Lab-Index pass identified rather than
deferring them.

**Two new fields — the first additions the PRC axis (§A.5) has ever received.**

* `FilmProfile.exposure_index_tungsten` — the manufacturer's tungsten index where a
  source prints one alongside the daylight figure. Populated on **7 stocks**, and
  populated *only* where an unfiltered pair is printed. The ratio is the point, and it
  separates cleanly along documented physics: ordinary panchromatic stocks cluster at
  **1.25** (Plus-X 5231 80/64, Double-X 5222 250/200, FP4 125/100, Tri-X Reversal
  200/160), `ILFORD_HP4` is genuinely flat at **1.00**, and the two blue-sensitive-only
  Polaroids sit at **3.20 and 3.33** — arrived at independently, which is the
  corroboration that the ratio really is tracking sensitisation and not manufacturer
  habit. `verify.py` now asserts that separation. Deliberately NOT populated:
  `GEVACOLOR_NEG_652`, whose 32 ASA already *is* its tungsten rating, so recording it
  again would manufacture a ratio of 1.00 that looks like HP4's genuine flat response
  and means something entirely different.
* `FilmProfile.processing` (`ProcessingSpec`) — developer, dilution, minutes, °C,
  agitation and contrast-index aim describing **which development condition the stored
  curve represents.** Populated on **2 of 142 stocks**. That count is not an oversight,
  it is the measurement: almost every datasheet in this corpus prints a curve without
  naming the developer that produced it, and the field exists to make that ambiguity
  visible rather than to hide it. All-empty means *explicitly unknown* in the §A.3
  three-valued sense — not zero, not not-applicable. Nothing in the render reads it; it
  is descriptive, and it does not attempt to model what a different developer *would*
  do, which needs a curve family over a processing axis and is a much larger change.

**Two corrections to existing data.**

* `GEVACOLOR_1952` balance **5500 K → 2850 K**. Cheltsov & Bongard 1958 survey every
  Gevacolor negative of the period and all are tungsten — N-5 at 2850 K, 652 at 3200 K;
  no daylight-balanced Gevacolor negative exists in that survey. The identification is
  tight on a figure we already held independently: N-5 is rated 14/10 DIN = 16 ASA on
  the book's own internally-consistent pairs, and this profile's `exposure_index` was
  already 16, from unrelated secondary sources. The **[T3] tag is unchanged on purpose**
  — only the balance temperature became documented; curves, grain, MTF and halation are
  still analogy-derived, and upgrading the whole profile would overclaim what one
  corrected field earns. This *does* change the render at `wb_strength > 0`: blue gain
  under a 5500 K scene moves from 1.00 to about 2.4.
* `FERRANIA_P30` era **"1960s / revived 2017" → "2017 revival"**. Every value in that
  profile comes from the 2017 specification. The 1960s cine stock and the revival share
  a name and a nominal EI, but the revival was coated on rebuilt equipment with modern
  materials and no period measurement backs the wider span. The neorealist history stays
  in the description as history; the era field now states only what the data covers.

**A dead-test discovery, and the wrong claim it was hiding.** `verify.py` ends with a
summary block and `sys.exit`, and **six tests appended after it had never executed** —
including one asserting `POLAROID_55_PN_NEG` was the sharpest stock in the database.
Moving them above the summary made the pass count jump 108 → 114 with no new work, and
the resurrected test failed immediately: **the claim was false.** `KODAK_TMAX_100`,
`KODAK_TMAX_400`, `FUJI_NEOPAN_ACROS_100` and `AGFA_APX_25` are all documented at
**200 lines/mm**, and unlike the Polaroid figure they state the test-object contrast
(1000:1) they were measured at. Comparing a number with no stated TOC against numbers
that have one is not a comparison. The description, this document's predecessor, the
survey and the README all carried the error and are all corrected; the test now asserts
something true and checkable — that the stock stays in the top ten on `f50`, where it
ranks sixth of 142.

The general lesson is worth recording against §A.15: **a test that cannot fail is worse
than no test**, because it reads as verification. Any future append to `verify.py` must
land above the summary block.

**Follow-up 2026-08-14 (third entry) — two vendor documents, and a method finding that
should change how PDFs are approached.** `PDF/PROFILES/FUJI/fujifilm_motion_picture_film_manual.pdf`
(FUJIFILM MOTION PICTURE FILM MANUAL, ref. KB-1101E, 2011) and
`PDF/PROFILES/KODAK/Ektachrome_100d.pdf` (KODAK publication H-1-5285, February 2010) were
extracted. Corpus unchanged at 142 stocks; this pass corrected and verified rather than added.

**METHOD FINDING — check for vector paths before tracing.** The Kodak sheet draws its
curves as **PDF vector polylines, not as a raster figure.** The spectral sensitivity plot
is three 56-point paths, so the coordinates are *exact*; the only estimated step is axis
calibration, fitted by least squares to the printed tick centres and closing to **0.63 nm**
on wavelength and **0.009 log** on sensitivity — an order of magnitude better than any
trace. Every other curve in this project came through `digitize_plot.py` with the error
budget that implies. A two-line check (`page.get_drawings()` versus `page.get_images()`)
distinguishes the cases, and it is now the first step in `DIGITIZATION_QUEUE.md`. Measured
today: Kodak H-1-5285 is vector; the Fujifilm cine manual is raster.

**A cross-product borrow removed, and validated on the way out.**
`KODAK_EKTACHROME_100D_5285`'s spectral curves had come from the **5294/7294** sheet — a
different product, the 2018 Ektachrome reintroduction — borrowed under a *declared*
same-family assumption because no 5285 sheet was on file. H-1-5285 has now landed, so the
curves come from the sheet whose product number the profile bears. Comparing both
peak-normalised on the same 10 nm grid, peaks agree to one sample or better and values
agree within 0.1–0.3 log through every main lobe, so **the declared equivalence was
sound** — this is one of the few times a §A.9 equivalence assertion has been testable
after the fact, and it held. What 5285's own plot adds is measured **skirts**: active
samples 13/13/13 → **16/15/13**, real low-sensitivity tails on the red and green layers
that the earlier digitisation had truncated to the −4.0 floor. Measured render effect:
derived balance gains at 3200 K move to 1.5731/1.0/0.4256 from 1.5886/1.0/0.4380, about
−1 % red and −2.8 % blue, and exactly zero at the film's own 5500 K balance.

**`exposure_index_tungsten` definition tightened — unfiltered pairs only.** Both documents
forced this. A monochrome film needs no conversion filter, so its daylight/tungsten pair is
a pure statement about the emulsion. A **colour** film's second index is quoted *through* a
filter and is then dominated by it: EKTACHROME 100D is 100 daylight and **25 tungsten with
an 80A**, and 100→25 is two stops, which is simply what an 80A costs; every Fujicolor cine
stock is quoted the same way (a No. 85 for tungsten-type films in daylight, an 80A for
daylight-type films under tungsten). Those are **filter factors, not film properties.**
Storing them would make a filter look like a sensitisation difference. Every entry in the
field is therefore a monochrome stock by definition, and `verify.py` asserts it.

**An achromatic reciprocity failure — the first in the corpus.** All nine Fujicolor cine
stocks carry the same statement: no correction from 1/1000 to 1/10 s, +1/3 stop at 1 s,
**and no filter correction.** That last clause is an explicit statement that the three
records lose speed together, which is the opposite of every Kodak colour film in the H-1
master table (each needs a CC filter). So `FUJI_ETERNA_VIVID_500T_8547` carries a
per-channel spread of exactly **zero, and that zero is evidence rather than a default** —
asserted in `verify.py`. `onset_s = 0.1` is documented; the exponent is not, being a
single-point fit, and is recorded as such.

**Two schema gaps newly exercised, both now blocked only by the schema and no longer by
data.** The Kodak sheet's **spectral dye density curves** (400–700 nm, cyan/magenta/yellow
plus visual neutral, peak-normalised) are vector and exactly extractable — and §A.5 L2
records "dye spectral densities **absent**; a 3×3 matrix stands in". The same is true of
its MTF and rms-granularity curves, which we hold as three scalars each. For the first
time these are not missing measurements; they are measurements with nowhere to go.

**Also newly unrepresentable:** REALA 500D and ETERNA Vivid 250D are documented as having a
**fourth colour layer**. Our tripack has three. That joins the layer-order gap from the
Cheltsov batch as a structural limitation rather than a missing value.

**Two websites assessed, nothing entered.** Fujifilm "Film Simulations" are in-camera JPEG
presets in digital cameras, not emulsions, and neither page carries a physical film
measurement — the charts on the sponsored one are unlabelled JPEGs. Worth recording for
one reason: only three simulations map to a specific real film with any confidence
(Acros → Neopan Acros, PRO Neg. Std → NS 160, Classic Neg → Superia 100), and **Classic
Chrome matches no specific emulsion**, merely evoking Kodachrome. Modelling it as
Kodachrome would be a category error. One observation there does converge with our own
data: an independent description of Acros grain "changing as you go from light to dark
areas" is exposure-dependent grain amplitude — exactly what the three `sigma_shape_*`
scalars parameterise on every stock and nothing reads. Full record:
`CHANGES_2026-08-14b_fuji_kodak_websites.md`.

**Follow-up 2026-08-15 (second entry) — fifteen new references, one new stock, and three
2× granularity corrections.** Corpus **143 stocks** (`KODAK_TECHNICAL_PAN` from
publication P-255 — CI 0.50–2.50 from one emulsion, the widest documented processing
envelope in the corpus and the clearest argument yet for the §A.5 processing axis).
Two "genuinely absent" gaps from the verified gap chapter are **settled by their own
sheets**: `FUJICOLOR_SUPER_F500_8572` (RMS corrected 7.4 → **4.0**) and `AGFA_VISTA_200`
(RMS 9.4 → **4.3**, resolving 130/50 entered); `FUJI_ETERNA_VIVID_500T_8547`'s sheet
yielded RMS 6.8 → **3.5**. All three estimates were ~2× too coarse, a consistent bias
worth remembering when reading the remaining [C3] grain values. Portra publication
**E-190** landed but documents the **2006 NC/VC generation**, a different film from our
2010s stocks — recorded as candidates, not merged. The KODAK DATA BOOK (1495 pp, UK
loose-leaf, ~1948–1968) has volume 5 FILMS at pp 1150–1495, queued with a post-1960-ASA
warning. Zhurba 1984's rotated tables were read visually and corroborate Gurlev/GOST;
the book contains **zero ORWO content**. The online Zhurba 1990's target section
(pp 44–131) is **unreachable** — webp page images that web_fetch cannot retrieve — and a
local copy is requested. Full record: `CHANGES_2026-08-15b_new_references.md`.

**Follow-up 2026-08-16 — the ОЧ-45/ОЧ-50 identity, red-marked estimates, and a
current-state gap file.** Three requests, three results. **(1) ОЧ-45 = ОЧ-50, settled
from the standard itself:** ГОСТ 20945-80 carries a reference appendix (p8, added by
Amendment No. 1) — "Соответствие типов кинопленок ранее принятым": before 1987-01-01
ОЧ-45/ОЧ-180, from 1987-01-01 ОЧ-50/ОЧ-200. The rename tracks the sensitivity-scale
transition to ГОСТ 10691.4-84 (old nominal series 22/45/90/180 → ISO-aligned
25/50/100/200): the same emulsion's speed restated on the new scale, **not a
reformulation.** `TASMA_OCH_45` now carries the ОЧ-50 aliases, the appendix citation,
and Table 4's norms under the new designation. **(2) `FilmActiveProfiles.md`** now
renders every estimated value in **red** (`<span style="color:red">…*</span>`) while
documented values stay plain — 1382 red-marked cells against ~600 documented, one
choke-point change in the generator so the marking cannot drift from the evidence test.
**(3) `NotFound.md` rebuilt as a current-state document**: the layered history is
archived verbatim in `NotFound_history.md`, and before the rewrite its own top
follow-up was executed — 18 provenance strings that still denied documents on disk
(12 VISION/EXR/Konica, HP3, three Rollei, 682, Type B) were corrected, taking the
"no documentation of any kind" count from 44 to a verified **23**. The new file
separates: 23 wholly undocumented stocks (each with the documents reviewed before
concluding absence), partial gaps per held stock, corpus-wide missing parameter
classes, extractable-but-unextracted work, and recent settlements.

### 2026-08-16 (later) — Zhurba 1990 read via owner-supplied screenshots

The online copy's pp 44–131 are unreachable webp images; the owner supplied 8 page-spread
screenshots, read visually. Yield: **the first ORWO documentation in the entire corpus**
(Table 66, p124) — `ORWOCOLOR_NC21` 100 ед.ГОСТ / 5500 K (confirms the stored values
exactly) and `ORWO_CHROM_UT18` 50 / **5500 K daylight** (the stored 4500 K was an
estimate and has been corrected). Table 2 (p46) gives norms for the **post-1987
Фото-32/64/125/250 generation** (the same ГОСТ 10691.x-84 renaming as ОЧ-45→ОЧ-50);
recorded in provenance as successor-generation consistency data, not overwritten — our
stored RMS values all sit inside the printed ceilings. Table 13 corroborates ОЧ-50's
660 нм limit. "No documentation" list: 23 → **21** stocks. Candidates queued: NC19,
UT-15/20/21/23, UK-17, Typ L, ЛН-7/8/9, ДС-5М.

### 2026-08-16 (queue P1) — 7266, 5285, 2383 curves adopted; a monotonicity lesson

Digitisation queue Priority 1 executed. Discovery: 5203/5213/5219/5222 were already
[T1]-traced (2026-08-02) — the queue was stale; fresh re-traces reproduced them (5219 r
to 3 decimals). Newly adopted [T1]: `KODAK_EKTACHROME_100D_5285` (H-1-5285 vector
paths — exact), `KODAK_TRI_X_REVERSAL_200` (7266 raster trace, mid-grey anchored), and
the `KODAK_2383_RELEASE` print stock (2015-sheet vector paths, x = 0 at the sheet's own
LAD aims 1.09/1.06/1.03). Lesson recorded: a fit with shoulder_k < toe_k makes the
sigmoid-difference curve non-monotone past the shoulder — the free fits looked better
(RMS 0.0035) and were wrong; all three were refitted under toe_k ≤ shoulder_k ≤ 2·toe_k.
verify.py: 124 PASS / 2 pre-existing FAIL; 3 new permanent guards.

### 2026-08-16 (official-source hunt) — Kentmere and Konica sheets retrieved

The owner asked whether the red (estimated) cells in `FilmActiveProfiles.md` could be
filled from official sources. Measured answer: **yes for whole undocumented stocks, no for
the parameter classes manufacturers never printed.** Retrieved and entered: the official
HARMAN July-2022 sheets for KENTMERE PAN 100/400 (ilfordphoto.com — ISO, 0.125 mm acetate
base, reciprocity Ta = Tm^1.26 / Tm^1.30, full development matrices) and the Konica VX 100
sheet (RMS granularity 4, resolving 63/125 lines/mm, achromatic reciprocity). **Three of
four sheets agreed exactly with values the database already carried** — the estimates were
sound and are now citations. The "no documentation of any kind" list fell 23 → 18 today.
Routes recorded for the rest: archive.org (ORWO Wolfen booklets), patents class G03C for
DIR/layer data, photographic-science journals for Callier and interimage.

### 2026-08-16 (section 4) — 14 stocks gained measured spectral curves

`NotFound.md` §4 ("identified, extractable, not yet extracted") was worked through rather
than restated. The corpus was swept exhaustively: **516 genuine vector curve pages in 245
PDFs**, 152 of them spectral sensitivity — the old "~30 sheets" estimate was an order of
magnitude low, and the 421 logo/glyph false positives are now counted separately rather
than feared. **14 stocks gained a measured spectral sensitivity curve** taken from exact
PDF vector coordinates (UltraMax 400/800, Ektar 100, Portra 160/800/100T, Gold 100 + 200,
Tri-X 400TX, T-MAX 100, T-MAX P3200, Plus-X 125, T400CN, BW400CN), lifting the count from
**53 to 67 of 143 stocks**. Axis calibration closes to 0.27 nm rms on wavelength and
0.012 log on sensitivity against the sheets' own tick labels; sampling is 10 nm, coarser
than the source everywhere, so the step decimates rather than invents. Nine layers whose
true peak falls between grid points carry a disclosed 0.01–0.06 log renormalisation shift.

The OCR bracket is closed entirely, and KODAK DATA BOOK volume 5 is closed as a documented
dead end — 346 pages containing zero RMS granularity figures and zero numeric gammas. What
remains open is listed with the specific blocker for each, including four Ilford wedge
spectrograms that carry no numeric axis ticks and therefore cannot be calibrated at all.

### 2026-08-16 — schema v7: four inert data carriers

Several classes of measured data had been accumulating in the extraction reports with
nowhere to go. v7 adds the fields — and nothing else. `SpectralDyeDensity` (the C/M/Y and
neutral absorption curves that a 3×3 `dye_matrix` cannot express, plotted on **54 vector
pages** in the corpus), `LayerStack` (coating order plus per-**layer** resolving power,
needed because six 1950s films stack unconventionally and a per-layer figure without its
order silently asserts a stacking the film may not have), `ProcessingFamily` (the published
developer × time × contrast matrices, where `ProcessingSpec` holds only one row), and
`ReciprocityTable` (Kodak's exponent walks 0.85 → 0.70 across decades, which one
Schwarzschild exponent cannot express at both ends).

**Every one is inert: nothing on the render path reads them.** That separation is
deliberate — the risk in this data was never in carrying it but in wiring it in, where dye
density curves would change every colour image. Wiring is a later per-feature stage with
its own before/after comparison. Two proofs are now permanent: ten stocks spanning colour
negative, reversal, monochrome, period and Soviet emulsions hash **bit-identically** before
and after the change, and a `verify.py` guard populates every v7 field on a profile,
renders twice and requires zero difference — so a future accidental wiring fails
immediately. Each carrier validates on load, and the rules were tested by trying to break
them; one caught its own author, refusing a probe whose dye densities dipped below zero.
`SCHEMA_VERSION` 5 → 7, `verify.py` 128 PASS / 2 pre-existing FAIL, C++ compiles clean.

### 2026-08-17 — ДС-5М entered from its state specification; a measured-grain bug fixed

**`SVEMA_DS_5M` added, 143 → 144 stocks**, from **ТУ 6-17-691-88** — a state manufacturing
specification rather than a datasheet, which makes it the strongest Soviet source in the
corpus, but which also means every figure is a **norm** and a value at a limit reads as "no
worse than". Entered: mean gradients 0.60 / 0.54 / 0.50 behind blue/green/red, Dmin
0.70–1.05 / 0.25–0.50 / ≤ 0.25, latitude ≥ 1.2, MTF at 30 mm⁻¹ ≥ 0.30 green and ≥ 0.15 red,
RMS granularity ≤ 22 green and ≤ 30 red, base ОТБ-14 triacetate 0.150 mm, sensitometric
exposure 5500 K.

**Two corpus-wide gap classes in §3 of `NotFound.md` are no longer absolute.** The Dmin
ladder *is* the coloured-coupler mask stated numerically — the first such figure anywhere in
the database. And table 2 item 6 prints **seven measured unwanted/useful density ratios**,
the dye-impurity class previously recorded as absent everywhere. The engine's `_dye()`
helper takes a single saturation scalar and cannot express seven independent terms, so they
are carried verbatim in provenance until a dye-impurity carrier exists — the data is kept
even though the renderer cannot yet use it, which is the same discipline as schema v7.

**A live bug surfaced during entry.** `_grain_v2()` derived per-channel RMS for every colour
negative and overwrote whatever the profile literal had set — the very thing the module
already forbids for the σ(D) shape. `GEVACOLOR_NEG_682` carries 23 / 16 / 34 measured off
Fig. 12 of the Vervoort & Stappaerts SMPTE paper, whose point is that the ordering
blue ≫ red > green is *the opposite* of the heuristic because the DIR couplers act on green
and red only; it had been rendering as 17.6 / 16.0 / 20.8 with that inversion erased. The
heuristic is now a fallback only, and that stock's output changes deliberately.

Gorokhovskii's «Методы спектральной сенситометрии» (УФН XVI(4), 1936) was reviewed in full
and **nothing was entered** — it is a methods paper. Its lasting contribution is a warning:
the same class of material gave green/blue sensitivity ratios of 4–13 %, 30–60 % and 20–45 %
in the hands of three different investigators, so any single 1930s spectral figure is
method-bound and two period sources must never be averaged. Its Fig. 7 does carry the only
measured curve for a 1930s Soviet material in the corpus («Изопанхром ФОКХТ»), deliberately
not attached to `SOVIET_PANCHROM_1939` because that would graft a named product onto a
generic one.

---

### 2026-08-18 → 2026-08-25 — consolidated: the log resumes, and what closed while it was silent

⚠ **THIS LOG STOPPED ON 2026-08-17 AND THE PROJECT DID NOT.** Eight days of work are
absent above, including four items this document lists as open gaps and three of its
Appendix D findings. That silence is itself the first entry: a status log that stops
is indistinguishable from a project that stopped, and §9's census went stale for
exactly as long. Consolidated rather than reconstructed day by day — what matters is
which of *this document's* claims changed.

**Database:** 144 → **160 materials**, 10 positive materials, schema **v5 → v11**.

#### Claims in this document that are now closed

| § | Claim as written | State on 2026-08-25 |
|---|---|---|
| 4.21 | granularity-versus-density is stored but unread | **Wired and measured on 13 materials.** ⚠ And the heuristic it replaced is **wrong in sign**: it had granularity falling toward D-max on reversal materials, where measurement shows it *rising* **2.8×**. On a reversal material D-max is the unexposed, fully developed silver, so rising is the physical direction — a negative material's shape had been pasted onto positive material |
| 4.25 | MTF is three scalars where a curve is required | **26 curves traced from 12 sheets, `q` stored and read on 10 materials.** ⚠ Two of this document's expectations failed: `q` is **not** a layer-depth constant (measured 1.89–3.42, far too spread), and the estimated per-layer f₅₀ triple is wrong in **form** — measured red f₅₀ is effectively constant at 36.4 cycles/mm (±13 %) while green spreads 52 % and blue 70 %, so no fixed layer-order ratio can be right |
| 4.7 | reciprocity is present and unread | **Read by both renderers since 2026-08-23**, 21 measured tables, default-inert so prior renders are bit-identical |
| 4.5 | spectral sensitivity substituted at render time | **Integrated.** 47 materials carry curves; the first vector-traced set (2026-08-25) moved its material's render by +0.28 stop of red gain at 3200 K |
| 4.13 | interimage coupling needs a density dimension | Topology unchanged, but the **claim about neutrals was corrected**: the correction vanishes *at the mid-grey anchor only*, not along the neutral axis. Off-anchor neutrals move by 15.9/255 and 6.5/255 on a measured material — the mechanism, not a leak, since white-light gamma below separation gamma is the patent's own metric |
| 4.14 | lateral inhibitor diffusion, per-material | Both renderers now gate each component below **0.25 px** of diffusion radius. ⚠ The gate had existed on **one side only**, so below that scale the two implementations were not approximating each other — one ran the stage and the other did not |
| 4.19 | masking-coupler density is required | The D-min ladder **is** the mask stated numerically, and it is now carried on the materials whose sources print it |
| 4.35 | densitometric convention absent as data | Still metadata-only — **and worse than that**: see the criterion finding in §9 |

#### Findings that change how this document should be read

⚠ **THE BIGGEST SINGLE CORRECTION WAS A RETRACTION, AND IT BEARS ON §1.2's
UNCERTAINTY BUDGET.** A granularity-versus-density law was derived from a research
report, adopted, and **withdrawn the same session** once the report's own model was
read: it defines its σ from a two-level opaque-grain model approached *as the
scanning aperture becomes vanishingly small*, and the measured fractional
fluctuation runs to **164 %** — so the small-signal linearisation that converts σ of
transmittance to σ of density is invalid across the whole plate. **Two granularity
measurements can use the same words, the same symbols and the same material and
still not be the same quantity.** §1.2's budget assumes reference data can be
*better* than the target; this case shows reference data can be **incommensurable**
with the target, which no budget detects. Add a commensurability check ahead of the
budget: *is this the same quantity, measured in the same regime?*

✅ **A CONSEQUENCE WORTH RECORDING: a conflict this project was about to enter never
existed.** Two sources appeared to disagree about whether monochrome granularity
turns over at high density. They were measuring different regimes — one at a fixed
densitometer aperture with grains unresolved, one at the pinhole limit. Asking "are
these commensurable?" dissolved it. **Record-the-conflict rules are for real
disagreements; reaching for one too early parks a permanent falsehood in the record.**

⚠ **PROCESSING IS NOT ONLY AN AXIS OF THE CURVE — IT IS AN AXIS OF GRAIN SIZE.**
§4.11 asks for the joint dependence of curve, granularity and MTF on processing.
Measured, from a printed table: equivalent grain diameter scales as **γ^0.42–0.45**,
against the √γ the source's own normalisation column implies. That immediately
exposed a condition mismatch in adopted data — a grain diameter taken at one
laboratory's development gamma sitting on a material stored at another's, 27 % out.
**A number is only valid at the condition it was measured at**, and §4.10's
processing-state definition is therefore not bookkeeping but a correctness
requirement. The same source measures grain size falling **20 %** across the tone
scale, so any single stored value is a mid-scale representative and should be
labelled as one.

⚠ **§4.20's ASSUMPTION THAT DYE TRACES CAN BE IDENTIFIED BY GEOMETRY IS TOO WEAK.**
Extraction of these panels failed for a fortnight on a plausible, recorded, *wrong*
diagnosis. The real cause: one publisher draws the red-absorbing trace as **two
overprinted paths** because red is not one of the four process inks, and a
segment-count filter dropped both. The reliable identifier is the **ink**, and the
rule behind it is physical rather than decorative — each trace is drawn in the colour
of light it concerns. Where a publisher is consistent, ink is nearly identity;
segment counts, stroke widths and dash patterns are only proxies for it.

**A validator worth adding to §4.20.** Where the three dye traces are peak-normalised
and a visual-neutral trace is printed as-measured, the sum identity in this document
cannot hold. The generalisation that does hold is `Neutral − Dmin = k·(C+M+Y)` with
the three coefficients **equal**, because equal contributions are what make the
result a visual neutral. On the material where it was first applied the unconstrained
solution came out **0.628 / 0.604 / 0.595** — a 5.4 % spread on three free
parameters — at rms **0.019 D**, while dropping the D-min term made the fit 4.5×
worse and scattered the coefficients over 0.86–1.61. **When a normalisation breaks
the obvious identity, look for the generalised one before declaring the data
unusable.**

⚠ **§4.21's PAIRING METHOD NEEDS A CONDITIONING TEST.** Reading granularity against
density from two plots on one abscissa requires inverting D(log E). Wherever that
curve is flat the inverse does not exist, and the raw pairing there produces smooth,
plausible, entirely fictional structure — an apparent 2.93× interior peak in one case,
discarded. **Condition the pairing on |dD/dlog E| before fitting, not after**: 22 of
52 points had to go. On reversal materials the flat zone is **D-max**, the opposite
end from a negative.

⚠ **AND ONE CAUTION ABOUT DOCUMENTS OF THIS KIND.** Several errors closed in this
period were in *prose*, not in data: four hardcoded counts in a report generator
wrong by up to 2.3×; a hand-maintained queue of un-digitised plots left empty long
enough that the report began asserting no such plots existed; a criterion string
naming a density that appears on no source sheet, propagated by each new adoption
matching its siblings. **A specification decays exactly like a census.** The
practical rule this project adopted: any count that can be derived from the data must
be derived in the same expression that presents it, and any claim that cannot be
derived must carry the method that produced it — which is why §9 above now names its
method and refuses to restate the figure whose method it cannot reproduce.
