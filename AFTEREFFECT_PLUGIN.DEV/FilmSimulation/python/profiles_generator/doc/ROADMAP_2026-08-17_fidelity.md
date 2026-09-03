# Fidelity roadmap — what to do next, in what order, and why

**Date 2026-08-17.** Requested by the owner. Opinion and plan; adopts no value, changes no
profile.

**Two scope limits stated first, because they bound every claim below.**

1. I have not seen a render from this engine and have not compared one to a real scan. All
   statements about *visual* fidelity are engineering inference from the model's structure.
2. Part 1's counts come from a survey I ran across all 457 PDFs in `PDF/` using
   `pdfinfo` / `pdftotext` / `pdfimages` on the device. **A "vector page" here means a page
   carrying no raster image**, and the per-class page counts are **document-level upper
   bounds, not plot counts**: a 12-page all-vector datasheet that mentions MTF once in prose
   contributes 12 to the MTF total. The queue's earlier figure of "119 vector MTF pages" was
   a per-plot count and has since been remeasured at **199** by `plot_inventory.py`
   (`NotFound.md` §4). Use my figures to rank seams against each other, not as inventories.
3. ⚠ **Working-copy caveat 2026-08-23.** The 457-PDF survey describes the full archive. **This
   checkout holds only `AGFA`, `FERRANIA`, `FUJI`, `GEVAERT`, `KODAK`, `RETRO` and `SVEMA` under
   `PDF/PROFILES`** — the `KONICA/…` file named in Part 1 ⑤ and the Soviet-standards material are
   not openable here, so none of these counts can be re-derived from this copy alone.

> ## ⚠ PROGRESS UPDATE 2026-08-18 — read before using this roadmap
>
> Written 2026-08-17 and **partly overtaken the next day**. What changed:
>
> * **σ(D) was the roadmap's grain seam, and it is now WIRED** — not just stored. Carrier
>   chosen by measurement (3 anchors + an interior peak; a 12-sample array was rejected as
>   over-parameterised against 7 measured points), read by the renderer for every stock traced
>   from a vendor plot — **12 stocks carry `sigma_shape_measured` as of 2026-08-23**, and schema
>   is now **v10** (this bullet was written at v8, 5 stocks).
>   `RESULT_2026-08-18e_C1_sigma_wiring.md`.
> * **The first measured σ(D) for a colour REVERSAL stock exists** (EKTACHROME 100D 5285),
>   and it rises ~20× from dmin to dmax — the opposite sign to the heuristic. Both halves of
>   that heuristic now have a measured counter-example.
> * **Spectral dye density went from 2 stocks to 11 sets.** Three of the "failed" sheets
>   turned out to be extractor defects, not source problems.
> * **The corpus survey's premise needed correcting**: a 2026-07-31 caveat listed 12
>   documents as not on file and **11 were present**, including one this project's own code
>   already opens. `NotFound.md` §0.3. So treat any "not held" claim in Part 2 as needing a
>   file-by-file check first — that is now a cheap check and it has paid twice.
> * **Part 2 item 5 (Sehlin/Kennel)**: the citation year is settled as **July 1985,
>   pp 724–734**; the filename's 1983 is the conference date. Granularity-vs-exposure data
>   is in Figs 7, 8, 9 plus a referenced Fig. 11.
>
> Still open and still ranked correctly by this document: MTF-as-a-curve (C2), the noise
> power spectrum, and the clustered-grain process in Part 3.

---

# Part 1 — What is still in the documentation we already hold

## 1.1 Survey result

423 of 457 PDFs surveyed (the 34 excluded are >20 MB scans, mostly Soviet standards and
large Kodak books). 403 carry a usable text layer; **19 carry none at all** and need tracing
or OCR.

Documents mentioning a parameter class *and* having at least one vector page:

| Class | Docs | Vector pages (upper bound) | Currently in the database |
|---|---:|---:|---|
| Contrast index | 215 | 1340 | `processing` rows, 26 % coverage |
| Reciprocity | 214 | 1332 | `reciprocity`, `reciprocity_table` — ⚠ **WIRED and live as of 2026-08-23** (C8), not a stored-but-unread field: `RenderSettings.exposure_time_s` / `AlgoControls::exposureTimeS`, with 0 meaning inert. **21 measured `ReciprocityTable` entries** (was 6) after 15 were read from vendor sheets; **105 stocks carry a Schwarzschild exponent**. ⚠ The model is a per-channel **GLOBAL log-exposure shift** — no source in the corpus has an intensity axis, so within-frame exposure-dependent failure is out of reach. Cross-language parity audited against the plugin's C++ (`cpp_parity.py`) |
| Granularity | 213 | 1516 | `rms_granularity` documented for **6 %**; σ(D) shape **MEASURED for 12 stocks and READ by the renderer** (queue items C1 + C1c; schema v8 introduced `sigma_shape_measured`, and **v9 redefined `rms_granularity` as the rms at NET density 1.0** — the convention Kodak prints; schema is v10 today). The heuristic that fills the other 147 is kept out of the measured path, both its branches being wrong in sign. **Per-layer rms is measured on 11 stocks** as of 2026-08-23 (C1e: 5219 5.92/6.60/17.84, 5207 rms_b 8.92, 5203 rms_b 4.71, ratios off the raster sheets, greens frozen; **5213 stays on the heuristic** — its sheet draws one bold band). ⚠ **REVERSED: the old stack ladder (blue 1.30×, red 1.10× of green) is contradicted by nine sheets** — b/g 1.81–2.79, r/g 0.75–1.05 — and is deliberately NOT rescaled, because all nine are Kodak cine negatives and nothing outside that class is measured. ⚠ UPDATED 2026-08-19 and 2026-08-23; this row previously said "5 stocks" and "schema v8" |
| MTF | 156 | 1753 | ⚠ UPDATED 2026-08-19 (C2 DONE): the rolloff SHAPE is now stored and read, as `1/(1+(f/f50)^q)` behind an `mtf_measured` flag — the form was chosen by scoring five candidates against a traced curve, which is how the reserved `mtf_tail_*` pair was found to be the WRONG form (rms 0.0583 vs the power law's 0.0375). ⚠ UPDATED 2026-08-23: measured on **8 stocks** (PLUS-X 5231 mono, plus the colour 5201, 5274, 5217, 5218, 5245, 5248, 5279) from **26 curves traced off 12 sheets** by `mtf_vector.py`; 5205 and 5293 have measured green+blue but a REFUSED red, so they carry a mixed triple and are NOT flagged measured. **63 colour stocks still carry an estimated f50 triple** and keep the legacy Gaussian. The power law beats the Gaussian on all 26 curves and the ordering q_R ≤ q_G ≤ q_B holds 8/8 stocks, ⚠ **but the magnitudes are NOT per-layer constants** (red 1.89–2.77, blue 2.38–3.42) — so q cannot be derived and stays per-stock measured (queue C2b) |
| Latitude | 107 | 1810 | 13 % |
| Resolving power | 105 | 898 | 49 % |
| **Spectral dye density** | **84** | **520** | ~~2 of 154 stocks~~ → **10 film profiles + 1 print stock** as of 2026-08-18 (`dye_density.py`, 11 sets, all re-derivable). Two sheets remain unextracted: 5246 and 5248 |
| Dmax | 63 | 816 | 12 % |

## 1.2 The five documents to revisit first

### ① `KODAK/Processing-KODAK-Motion-Picture-Films-Module-14.pdf` — the biggest unmined find

40 pages, **all vector**, zero references to it anywhere in our docs or code. Title:
*"Effects of Mechanical & Chemical Variations in Process RVNP"*. Its table of contents lists
**Figures 14-2 through 14-14** — sensitometric response to mechanical and chemical process
variation — and **Figures 14-15 through 14-22** — effects of contamination.

**Why this is the top document.** `processing` is currently a label plus a scalar or two. These
figures are *partial derivatives of the characteristic curve with respect to process
conditions*: developer time, temperature, agitation, replenishment rate, pH, and specific
contaminants, each against Dmin / Dmax / gamma / speed. That converts "ECN-2" from a name
into a model, and it is what makes "this reel went through a hot developer at a tired lab in
1978" a physical statement instead of a hand-tuned offset. Real footage varies by lab and by
process drift far more than it varies by emulsion batch; a simulator that cannot express that
variation renders every frame as if it came off a perfectly controlled process, which is
itself a tell.

**Extract:** for each figure, the sensitometric delta per unit of process deviation —
Δgamma, ΔDmin, Δspeed, and per-layer colour balance shift where plotted. Representation: a
new `ProcessSensitivity` record, one per process family, holding the partial derivatives with
their valid range.

⚠ **Honest limitation:** this module and Module 11 are **reversal** processes (RVNP, VNF-1),
not ECN-2. Directly usable for the 34 reversal stocks. For the 120 negatives the *structure*
transfers (which knob moves which parameter, and roughly how steeply) but the coefficients do
not, and must not be presented as if they did. The equivalent ECN-2/ECP-2 module is **not in
the archive** — I searched; no document mentions ECN-2 or ECP-2 outside the film sheets.
That is a specific, well-defined external request (see Part 2, item E5).

### ② `KODAK/Processing-KODAK-Motion-Picture-Films-Module-11.pdf`

Process VNF-1 specifications, with **"Typical Sensitometric Effects of Bleach, Table 11-1"**
and a "Film Structure" section. Same class of data as ①, plus the bleach step isolated. Also
never cited by us.

### ③ `KODAK/kodak-essential-reference-guide-for-filmmakers.pdf` (H-845) — checked, and downgraded

216 pages, 149 vector pages, hits **every** parameter class, and cited by the VISION3 sheets
themselves as *the* image-structure reference — so I expected a great deal. **I read it, and
it is mostly qualitative.** The "Film Structure" chapter describes layers, couplers,
antihalation types and the latent image in prose with almost no numbers; "Basic Sensitometry"
is educational. A digit-density scan of all 216 pages found exactly **one** page whose
numeric content looks like a data table.

Its genuine contributions, and they are narrow:

* **p98–100: perforation dimensions and tolerances**, Bell & Howell / Kodak Standard / 16 mm,
  in inches and mm with ± tolerances. This is real and useful — see Priority 1.4.
* Antihalation density guidance ("to approximately 0.2") — a sanity bound on `coating`.
* Storage/aging temperature regimes (13 °C, −18 to −23 °C, and the effect of heat) — relevant
  to the `aging` field, but as *conditions*, not as measured density-loss rates.

I am recording the downgrade explicitly because the first look at the survey made this
document appear to be the single biggest opportunity, and it is not.

### ④ The 84 spectral-dye-density documents

Only **2 of 154** stocks carried `dye_density` when this was written — **10 film profiles plus 1
print stock** as of 2026-08-18 (§1.1), out of **159** stocks today — and the first two were
validated against their own
neutral trace (5285 to 0.013 D, 2383 to 0.128 D). Meanwhile 84 documents mention spectral dye
density *and* have vector pages. The Kodak cine sheets are the cleanest targets — `5205t`,
`5218`, `5245`, `5246`, `5274`, `5279`, `5293`, `2254` are **entirely vector** (6–8 vector
pages each), and the Fuji still-film PIBs (28 of them: Velvia, Provia, Astia, Pro 160S/160C/
400H/800Z, Sensia, Superia, T64) have 1–6 vector pages each.

**Why it matters more than it looks.** `dye_matrix` is a 3×3 linear approximation to a
spectrally selective process. What a scanner actually measures is the integral of each dye's
spectral absorption against that scanner's illuminant and channel filters. Two films with the
same 3×3 will diverge under a different scanner, and no amount of tone-curve accuracy fixes
it. If the goal is matching a *specific* scan chain, this is the colour-accuracy seam.

### ⑤ The 19 raster-only documents

No text layer at all: must be traced or OCR'd. The ones that matter:

* **`Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf`** — 11 pages. Already established
  today as the corroborating source for falling granularity. Its **Fig. 9** carries five
  granularity-versus-exposure curves for *under- and overexposure*, and **Fig. 11** the
  per-exposure family. This is the only measured granularity-vs-exposure data in the archive
  for a colour negative other than the VISION3 sheets, and it would give a second,
  independent σ(D) family. ⚠ Reconcile its axis units with the 48 µm diffuse-RMS convention
  before any absolute value moves — the queue already flags this.
* `AGFA/NewGevacol_Neg_682.pdf` + `Verpoort_Stapp1980_NewGevacolNeg682.pdf` — Gevacolor 682,
  five figures including RMS-vs-density and spectral dye density.
* `KODAK/5294-7294-datasheet-EN.pdf`, `FUJI/Neopan400.pdf`, `FUJI/A 250.pdf`,
  `KONICA/professional_160.pdf`, `FERRANIA/Curve caratteristiche…`.
* `DUFAYCOLOR/Dufaycolor_Manual_1938_print.pdf` — 25 pages, our only Dufaycolor documentation.

### False positive worth recording

`AGFA/These-23-11-09fusion2.pdf` — 686 pages, 656 vector, and it hit the MTF and latitude
keywords. It is **Frédéric Rolland, "Les collections privées de films de cinéma en support
argentique en France", a 2009 art-history doctoral thesis** (Université de Versailles
Saint-Quentin). No measurement content. I checked it because the survey ranked it first by
vector-page count; keyword-plus-vector ranking has false positives and this was the largest.

## 1.3 Classification the owner asked for

**(a) Available and extractable now, no interpretation needed.** Perforation dimensions and
tolerances (H-845 p98–100). Contrast-index and reciprocity tables in 214–215 documents,
mostly as printed text. Latitude and Dmax figures in 63–107 documents. These are text or
simple tables — parse, do not trace.

**(b) Available, needs digitisation.** Spectral dye density (84 docs, vector — exact
coordinates, so *cheap* per curve). MTF curves (156 docs, vector). Process-variation figures
(H-24 Modules 11 and 14, all vector). Granularity-vs-density families for stocks beyond
VISION3. The 19 raster-only documents (more expensive: trace, with a visual gate).

**(c) Available only qualitatively.** Layer stack architecture and coupler chemistry (H-845,
Module 11 "Film Structure") — order and function, never thicknesses. Antihalation method per
stock — type, not density. Storage and aging — regimes, not rates.

**(d) Available at insufficient resolution.** Granularity as a single quoted rms number for
94 % of stocks, where the physically meaningful object is σ(D) *and* a noise power spectrum.
MTF as one f50 where the real curve overshoots unity — ⚠ true for the 63 colour stocks still on an
estimated triple, but no longer for the 8 stocks that now carry a measured rolloff (§2.3).
Spectral curves at 10 nm where ACROS
demonstrably needs 5 nm at the red cut-off. Soviet ГОСТ/ТУ norms that are *specification
ceilings*, not measurements — already correctly marked in the report's blue/dagger class.

**(e) Apparently absent from the archive entirely.** Interimage effects (all 395 documents
searched, none publishes them — camera negative is characterised with one white-light
series). Developed clump geometry. DIR coupler coefficients. Layer thicknesses. The ECN-2 and
ECP-2 process modules. Any noise power spectrum. Any scanner characterisation.

## 1.4 Would more curve digitisation pay? Yes, but selectively

**Yes, for two classes, because the data is vector.** A vector path gives exact coordinates —
today's VISION3 work needed a supervised trace and four attempts precisely because those
plots are *raster*; the spectral-dye-density and MTF pages listed above are not. Cost per
curve is minutes, accuracy is limited only by axis-label reading, and the machinery already
exists and is validated (0.27–3.93 nm calibration residuals across 18 vector extractions).

**No, for breadth.** Tracing the H&D curve of a 50th stock does not make any frame more
convincing. Depth on the stocks you intend to target beats coverage.

## 1.5 Part 1 priority, impact against effort

| Rank | Work | Effort | Expected fidelity gain |
|---|---|---|---|
| 1 | H-24 Module 14 + 11 → process-sensitivity model | Medium (vector, ~30 figures) | **High** — unlocks lab/process variation, an axis the engine cannot currently express at all |
| 2 | Spectral dye density, ~20 target stocks | Low (vector, proven path) | **High** for colour under a specific scan chain |
| 3 | MTF as a curve with overshoot — ⚠ the **rolloff half is done** (§2.3); what remains is the adjacency overshoot and more traces | Low–medium (vector) | **Medium–high** — local edge contrast, first-order visible |
| 4 | Perforation tolerances → gate weave | Very low (a printed table) | **High for motion**, nil for stills |
| 5 | Sehlin/Kennel Figs 9 and 11 traced | Medium (raster, needs the overlay gate) | Medium — a second independent σ(D) family |
| 6 | Contrast-index / reciprocity / latitude text harvest | Low, mechanical | Low–medium; mostly completes provenance |
| 7 | Remaining 18 raster-only documents | High per document | Low–medium, stock-dependent |

---

# Part 2 — What to seek externally, in priority order

Each item: what, why, representation, resolution, specificity, likely holder.

### E1. Noise power spectrum (Wiener spectrum) of granularity, per density, per layer
* **Why.** This is the highest-value missing dataset in the whole project. `AlgoGrain` already
  builds a spectrum from clump size and a clumping gain, but those two scalars are tier-3
  estimates and the resulting field is Gaussian. A measured NPS both fixes the spectrum and
  gives a target that a better grain process can be fitted against. Grain structure at
  magnification is, in my judgement, the most likely single tell in a still frame.
* **Representation.** Curve: power (D²·mm²) versus spatial frequency (cycles/mm), one per
  density band per layer. 5–8 density bands, 0–100 cycles/mm.
* **Resolution.** ≥ 20 frequency samples per decade; density bands ≤ 0.4 D apart.
* **Specificity.** Film-specific **and** processing-specific — NPS changes with development.
* **Likely holders.** The published optics literature is the realistic route: JOSA and
  Applied Optics carry the classic granularity Wiener-spectrum work (e.g. *Wiener-Spectrum
  Analysis of Photographic Granularity*, JOSA 52(6):669), and the medical-imaging NPS
  methodology literature is directly transferable. Also SMPTE Journal, and the Society for
  Imaging Science and Technology (PS&E / JIST). Manufacturer research divisions measured this
  routinely but rarely published per-product.

### E2. Scanner / telecine characterisation of the reference chain
* **Why.** There is **no scanner noise model in the engine at all** — I grepped for read,
  shot, photon and sensor noise and found nothing. Scanner MTF is modelled, and correctly
  placed before grain so it band-limits both. But a real scan's floor is grain *plus* sensor
  noise, and sensor noise is roughly constant in *transmittance*, so in density it rises
  steeply in the Dmax shadows exactly where grain is dying away. The shadow noise currently
  has the wrong shape, and no film data fixes it.
* **Representation.** Per channel: MTF curve (from a slanted edge), noise variance as a
  function of transmittance (2 coefficients: read + shot), spectral response of each channel
  (curve, 5 nm), illuminant SPD (curve, 5 nm), flare/veiling scalar, and the OETF.
* **Resolution.** MTF to 2× Nyquist; noise from 20+ patches spanning Dmin→Dmax.
* **Specificity.** Condition-dependent: specific to the scanner *model and setting*.
* **Likely holders.** **You**, with a step wedge and an afternoon. This is the single most
  obtainable item on the list and it gates the comparison loop.

### E3. Interimage effect and DIR coupler strength, per stock
* **Why.** Tier 3 for every stock without exception. These set how the extremes of the colour
  gamut behave and are a large part of a stock's colour "signature".
* **Representation.** Interimage: 3×3 matrix of gamma-steepening percentages per receiving
  layer. DIR: coupler strength plus a diffusion length in µm.
* **Resolution.** ±10 % relative would already be transformative against a tier-3 guess.
* **Specificity.** Film-specific and processing-specific.
* **Likely holders.** Patent literature is the only realistic public route — IPC class **G03C**,
  assignees Eastman Kodak and Fuji Photo Film. Worked emulsion examples sometimes give
  coupler loadings and layer structure; *"Color negative films adapted for digital scanning"*
  (US 6,045,983, Kodak) is an example of the genre. Otherwise: manufacturer research
  divisions, and university photographic-science departments with 1980s–90s lab records.

### E4. Grain microstructure: clump size distribution and spatial clustering
* **Why.** `clump_um_*`, `clump_gain`, `size_sigma_log` and `cluster_um` are all tier-3, and
  the report states plainly that clump geometry is never printed and cannot be recovered from
  a scan below the scanner's resolution. These parameters control grain *character* — the
  difference between velvety HP5 and VISION3's even sand.
* **Representation.** A size distribution (log-normal parameters, or a histogram) plus a
  pair-correlation function for clustering. Electron-micrograph derived.
* **Resolution.** Distribution to ±15 %; correlation length to ±20 %.
* **Specificity.** Emulsion-specific, and different before and after development.
* **Likely holders.** Manufacturer research divisions; academic photographic-science
  literature; possibly restoration labs that have done SEM work on film samples.

### E5. ECN-2 and ECP-2 process-variation modules (H-24 family)
* **Why.** We hold Modules 1, 11 and 14 — all reversal. The negative and print equivalents
  would give the same partial derivatives for the processes that the large majority of our
  stocks (159 today, 154 when this was written) actually use.
* **Representation.** Same `ProcessSensitivity` record as Part 1 ①.
* **Likely holders.** Kodak technical support directly; motion-picture laboratories, which
  hold these manuals as working documents; FIAF-affiliated archives; the Internet Archive.

### E6. Spectral dye density for stocks whose sheets omit it
* Covered in Part 1 ④ for the 84 documents we hold. Externally: the manufacturer's technical
  publication for the specific stock, or a measurement of a processed sample on a
  spectrophotometer — 380–730 nm at 5 nm, per dye and for a neutral, which is a
  straightforward lab measurement if a sample and instrument are available.

### E7. Aging, fading and vinegar-syndrome kinetics
* **Why.** The `aging` field exists. H-845 gives storage *regimes*, not rates. If the target
  includes archival material — and "film-restoration specialist" was named as a judge — dye
  fading rates and base shrinkage matter, and they are the difference between "new stock" and
  "a 1975 print".
* **Representation.** Per-dye density loss as a function of time × temperature × RH (an
  Arrhenius pair per dye), plus base shrinkage in % per decade.
* **Likely holders.** This is the one area where the *archival* community publishes well:
  Image Permanence Institute (RIT), FIAF technical commission, national archives'
  preservation departments, and the ISO 18901/18911 standards family.

## 2.1 Draft enquiries

Keep each to one page, state the purpose, ask for data not products, and offer to accept
whatever form the data exists in. A concrete, narrow request is answered far more often than
a general one.

**To Kodak (Motion Picture and Entertainment technical support):**

> We are building a physically-based film-response model for research and restoration-support
> purposes, and we work from your published Technical Data sheets — currently 159 stocks,
> with curves machine-traced from the published plots. Four specific requests:
> 1. Are the **H-24 "Processing KODAK Motion Picture Films" modules for Process ECN-2 and
>    ECP-2** available? We hold Modules 1, 11 and 14 (Process Control, VNF-1, RVNP) and use
>    the "Effects of Mechanical and Chemical Variations" figures. The ECN-2 and ECP-2
>    equivalents are what we need.
> 2. For VISION3 5203/5207/5213/5219: is a **diffuse rms granularity noise power (Wiener)
>    spectrum** available, or granularity measured at apertures other than 48 µm? We have
>    traced the σ-vs-density plots from the TI sheets and need the spectral distribution.
> 3. Do the sheets' **MTF curves** exist in numeric form, and is the low-frequency
>    adjacency response above 100 % published anywhere we have missed?
> 4. Is **spectral dye density** (cyan/magenta/yellow and neutral, 380–730 nm) available for
>    stocks whose TI sheets omit it?
>
> We are happy to receive scans of superseded printed publications; historical data is as
> useful to us as current data.

**To Fujifilm (technical / motion picture division):** the same four questions, naming the
Product Information Bulletins we hold, plus: *is the granularity-versus-density curve
available for the ETERNA and Pro-series stocks, in the same form as the published spectral
dye density curves?*

**To HARMAN technology (Ilford):** *Do you hold, or can you publish, granularity noise
spectra or grain size distributions for HP5 Plus, FP4 Plus, Delta 3200 and Pan F Plus? We
work from your published curves and reciprocity formulae (Ta = Tm^1.31 etc.), which are among
the most complete in our corpus. We would also value confirmation of the emulsion generation
each current datasheet describes.* HARMAN has a live technical function and has historically
answered specific questions.

**To FilmoTec GmbH (ORWO, Wolfen):** ORWO manufacturing is active and technically
approachable. *Do the Wolfen works records include sensitometric or granularity measurements
for the discontinued NC and UT colour stocks (NC21, NC24, UT18)? Our data for these comes
from Soviet-era reference books, which we consider weak for non-Soviet stocks.*

**To Agfa-Gevaert:** temper expectations — the consumer photographic business is long gone and
"AgfaPhoto" is a brand licensee, not the manufacturer. Ask the corporate archive rather than
technical support, and treat Gevacolor 682 (for which we hold two papers) as the specific ask.

**To archives and restoration bodies** — George Eastman Museum (Moving Image Study Center,
Technology Collection, and the Menschel Library, all by appointment; there is also a
Technicolor Online Research Archive), the FIAF technical commission, BFI, EYE Filmmuseum,
Library of Congress NAVCC, Academy Film Archive, and the Image Permanence Institute at RIT:

> We are looking for **laboratory technical records rather than films**: control-strip
> sensitometry logs, densitometer calibration records, process-variation studies, and any
> internal measurement of granularity spectra or dye fading kinetics. We can work from
> photographs of paper records.

⚠ **One specific lead, and its specific unavailability.** The **Kodak Research Laboratories
records — 113 cubic feet, 1888–2006 — are held by Rare Books, Special Collections and
Preservation at the University of Rochester, and are currently CLOSED to researchers, with
reopening anticipated in early 2027.** That is the most likely single home for E1, E3 and E4,
and it is unavailable for roughly the next five months. Put it in the calendar; do not put it
in the plan.

---

# Part 3 — Fallbacks for every high-priority item

| Item | Fallback | Accuracy | Effort | Risk | Fidelity impact |
|---|---|---|---|---|---|
| **E1 NPS** | (a) manufacturer request | exact if granted | low | **likely refused or nonexistent per-product** | high |
| | (b) JOSA / Applied Optics / PS&E literature | good for the *class*, not the stock | medium | may only cover 1960s B&W emulsions | medium–high |
| | (c) **measure it from your own scans** — NPS of a flat patch is a windowed FFT, and the scanner MTF divides out if E2 is done | good, and it is *your* chain | medium | conflates film and scanner noise unless E2 is done first | **high** |
| | (d) derive from clump size + clumping gain (status quo) | tier 3 | none | already in place; Gaussian marginals remain | baseline |
| **E2 scanner** | (a) vendor spec sheet | partial — rarely gives noise | low | incomplete | medium |
| | (b) **own measurement, step wedge + slanted edge** | high | **low** | none | **high** |
| | (c) physically motivated default: read + shot noise with plausible coefficients | order-of-magnitude | very low | wrong shadow noise persists but is at least present | medium |
| **E3 interimage / DIR** | (a) manufacturer | exact | low | almost certainly refused as proprietary | high |
| | (b) patents G03C | [T3] at best — patents describe examples, not products | medium | **must never be labelled as the stock's specification** | medium |
| | (c) **fit from a colour-target scan** of the real stock: interimage is visible as a gamma difference between neutral and separation exposures | good for the specific stock | medium | needs real material and a colour target | high |
| | (d) keep the current tier-3 estimate | tier 3 | none | ceiling on colour extremes | baseline |
| **E4 grain microstructure** | (a) manufacturer / SEM literature | exact | high | mostly unpublished | high |
| | (b) **Poisson-cluster (Neyman–Scott) grain process** fitted to the measured NPS and to the σ(D) already adopted — physically motivated, and it produces the skewed marginals a Gaussian field cannot | good *statistically*, not per-crystal | medium (algorithm work) | none to the database | **high — this is the recommended path** |
| | (c) status quo Gaussian field | correct in RMS and spectrum, wrong in marginals | none | the likeliest visual tell remains | baseline |
| **E5 ECN-2/ECP-2 modules** | (a) Kodak, labs, FIAF, Internet Archive | exact | low–medium | may not survive publicly | high |
| | (b) **transfer the *structure* from the reversal modules we hold**, fit coefficients to any control-strip data available | structure right, coefficients approximate | medium | must be labelled as transferred, never as ECN-2 measured | medium |
| | (c) defer | — | none | process variation stays inexpressible | — |
| **E7 aging** | ISO 18901/18911 + Image Permanence Institute | good, and published | low–medium | none | medium — only matters for archival looks |

**Deferrable without meaningful loss of perceived realism:** film base thickness and tint
beyond what is held (3 % coverage, effect is thousandths of a density unit); the 5 nm spectral
re-trace beyond ACROS (already measured at 0.4–1.1 %); emulsion layer thicknesses (they enter
only through halation and interimage, which are parameterised directly); breadth of stock
coverage. **Not deferrable if motion is in scope:** the four stubbed stages.

---

# Part 4 — The roadmap

## Priority 1 — highest impact, lowest effort

**1.1 Build the comparison loop. Do this before anything else.**
*What:* characterise the reference scanner (E2) and scan a step wedge and colour target on one
or two target stocks. Then implement a comparison harness: per-density-band noise power
spectrum per channel, cross-channel noise correlation, density histogram, MTF from a slanted
edge, dye crosstalk.
*Data:* your own measurements. *Representation:* a `ScannerSpec` record plus a reference-scan
corpus with a metrics report. *Expected gain:* zero directly — and it is still first, because
every item below is currently chosen by judgement rather than measured error, and this
converts the project from opinion to arithmetic. It will also confirm or refute my claims
about grain marginals and shadow noise, which are inferences from reading code.

**1.2 Spectral dye density, ~20 target stocks.** Vector paths, proven extraction path, schema
field exists and is validated. *Gain:* material improvement in colour fidelity under a
specific scan chain; the largest colour gain available from data you already own.

**1.3 H-24 Module 14 + Module 11 → a `ProcessSensitivity` record.** In the archive, all
vector, never touched. *Gain:* high, and it opens an axis of variation the engine cannot
currently express at all. Reversal coefficients only — label them as such.

**1.4 Perforation tolerances (H-845 p98–100) → gate-weave amplitude.** A printed table with
± tolerances; pin-to-perforation play bounds the weave amplitude in µm, which turns a stubbed
stage into a documented one. *Gain:* high for motion, nil for stills. An afternoon.

## Priority 2 — high impact, moderate effort

**2.1 Replace the Gaussian grain field with a Poisson-cluster process.** Algorithm work, not
data: keep the existing clump parameters and σ(D), draw a clustered point process, and let the
marginals come out skewed the way a counted emulsion's do. Fit against 1.1's measured NPS.
*Gain:* in my judgement the largest single improvement available to a still frame — but this
ranking is an inference, and 1.1 is what would confirm it.

**2.2 Scanner noise model** (read + shot, in transmittance) using 1.1's measurement. Small
code change, fixes the shadow noise shape.

**2.3 MTF as a curve with the adjacency overshoot.** ⚠ **THE ROLLOFF HALF IS DONE**
(2026-08-19, queue item C2): the shape is stored, read by both renderers, and the form was chosen by
measurement — of the reserved pair, only `mtf_tail_a` / `mtf_tail_f_exp` lost that scoring and are
inert; the rolloff itself is live. **2026-08-23:** 26 curves traced off 12 sheets, 8 stocks flagged
`mtf_measured`, and q stays **per-stock measured** because the exponents are not per-layer constants
(red 1.89–2.77, blue 2.38–3.42) even though q_R ≤ q_G ≤ q_B holds 8/8. ⚠ **And the f50 estimating
rule was reversed by the same traces:** `f50_r ≈ 0.78 × f50_b` is wrong in FORM — measured red f50 is
effectively constant at 36.4 cycles/mm (32.1–41.1, ±13 %) while green spreads 52 % and blue 70 %.
Five modern Kodak cine stocks (5203, 5207, 5213, 5219, 5246) have their red re-anchored to exactly
36.0, green/blue left at estimates; other makers and all pre-1990 stocks untouched; 63 colour stocks
still on an estimated triple. What REMAINS is the
adjacency half (queue C2c: `adjacency_um` disagrees with the measured overshoot frequency on both
stocks checked) and tracing more curves (C2b). Original text follows: ~156 vector documents;
`mtf_tail_a` and `mtf_tail_f_exp` already exist but are wired only into the C++ port. *Gain:* medium–high,
local edge contrast.

**2.4 The print and intermediate chain.** 2393 Premier, 2242/3242/5242 Intermediate, 3383,
2302 — datasheets already on disk. *Gain:* high, and arguably a precondition for the goal:
"indistinguishable from a film scan" is undefined until you fix whether the target is a camera
negative, an interpositive, or a release print, and most graded footage is not camera
negative.

## Priority 3 — high impact, high effort

**3.1 Grain NPS**, by own measurement (fallback c) once 1.1 exists, supplemented from the
JOSA/Applied Optics literature. **3.2 Interimage and DIR** fitted from colour-target scans
(fallback c), with the patent sweep time-boxed as a secondary. **3.3 The remaining three
motion stubs** — temporal flicker, negative defects, gate defects. **3.4 The 18 remaining
raster-only documents**, starting with Sehlin/Kennel Figs 9 and 11 and Gevacolor 682.

## Priority 4 — optional, diminishing returns

Breadth to more tier-1 stocks. The 5 nm spectral re-trace beyond ACROS. Film base and emulsion
property coverage. Soviet standards beyond what is held. And a calendar note for **early 2027**,
when the Kodak Research Laboratories records at the University of Rochester are expected to
reopen — the best single lead for E1, E3 and E4, and precisely the kind of dependency that
must stay out of the critical path.

## What I would not do

Continue the broad document hunt as the main line of work. The remaining unknowns are, with the
exceptions named above, absent from the published literature rather than from your archive —
and the project's own report already establishes that for interimage across all 395 documents.
More searching cannot produce numbers that were never printed. Measurement can.

---

## Sources consulted for Part 2

- [Kodak Research Laboratories records, University of Rochester RBSCP](https://uorrcl.access.preservica.com/uncategorized/SO_c04c7f5e-450d-4051-9394-82ddce47b61b/) — 113 cu ft, 1888–2006, closed, reopening anticipated early 2027
- [George Eastman Museum — Research Appointments](https://www.eastman.org/research-appointments) and [Technicolor Online Research Archive](https://www.eastman.org/technicolor-online-research-archive)
- [Wiener-Spectrum Analysis of Photographic Granularity, JOSA 52(6):669](https://www.osapublishing.org/abstract.cfm?uri=josa-52-6-669)
- [FilmoTec GmbH (ORWO, Wolfen)](http://www.filmotec.de/?lang=en)
- [US 6,045,983 — Color negative films adapted for digital scanning, Eastman Kodak](https://patents.justia.com/patent/6045983)
