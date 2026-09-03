# Mees 1942 — extracted findings, with page citations

**Source:** C. E. Kenneth Mees, *The Theory of the Photographic Process*,
The Macmillan Company, New York, **1942** (1st edition), 1118 pp.
File archived off the working drive 2026-08-03 — this document preserves
everything taken from it, so the PDF is not needed to act on any of it.

**Edition matters.** Author line is Mees alone, so this is the 1st edition.
**Interimage effects and DIR couplers are absent — 0 pages** — and correctly
so: DIR couplers were invented at Kodak c. 1968–71, three decades after
publication. Interimage came from the patent literature instead (see
`CHANGES_2026-08-03_v5_interimage.md`). The Selwyn references at pp. 221–222
concern quanta per latent image, not granularity.

**Correction, 2026-08-03:** an earlier version of this file claimed the Selwyn
*granularity* law was "absent". That was inferred from a regex that returned
nothing, which is not evidence of absence — and the pages it should have
searched had not been read. Chapter XXI *does* carry the granularity material
(§5 below). The aperture-scaling law was not found in the pages sampled, but
that is now stated as "not found in pages sampled", not as absent.

**The full extracted text layer is preserved as `PDF/MEES_1942_fulltext.txt`
(2 MB).** Mees remains fully searchable without the 356 MB PDF.

## 1. Reciprocity — our model is one-sided (p. 239–241)

Schwarzschild (1899): constant effect when `I·t^p = constant`, with
**p ≈ 0.8**. Then, verbatim (p. 240):

> "it is now well known that this relation, with p constant, is not a valid
> criterion for determining a constant photographic effect over very wide
> ranges of intensity."

Englisch (1901) found **p is variable**. Sheppard & Mees (1903–06) found
*"there is an intensity at which the photographic effect is a maximum for a
given exposure"* — i.e. film fails on **both** sides of an optimum.
Bunsen–Roscoe holds within 5 % near optimal intensity (p. 241, Fig. 74:
catenary, hyperbola, and the Schwarzschild straight lines).

**Implication, not yet implemented.** `ReciprocitySpec` has one constant `p`
and one `onset_s`, so it models only the long-exposure (low-intensity) side.
Real film also has **high-intensity reciprocity failure** at very short
exposures — which is why datasheets state a *range* ("no correction
1/10 000–1 s", Konica). A two-sided spec is the outstanding change.

**Also vindicates a choice we had guessed at.** Our B&W default is p = 0.95,
not Schwarzschild's 0.8. Mees explains why 0.8 is not general: Schwarzschild
was an astronomer working at extremely low intensities with exposures of
minutes to hours. So 0.8 is the astronomical-exposure value and ~0.95 is right
for cine/still. That converts a guess into a cited decision.

## 2. Callier q ties to grain size (p. 235)

Callier showed the ratio q of specular to diffuse density, `D_spec/D_diff`,
*"is closely related to grain size and increases with it"*. Eggert & Küster
measured the relation as

    d ≈ 6.8 · log q

with Nutting's `Π ≈ 1.53·d` for spherical grains giving `P ≈ 10.4·p·log q`.

**Use:** `callier_q` is currently a flat estimate (1.0 dye / 1.25 reversal /
1.30 B&W negative). Inverted, this is an **independent cross-check on
`clump_um`**, our weakest-evidenced quantity — grain size and q are not free
parameters, they are linked. Surfaced into the Processing column of
`FilmActiveProfiles.md`.

## 3. Eberhard effect — magnitude and a grain-size dependence we do not model
(pp. 872–875)

Eberhard exposed circular openings 0.3–30 mm and found the small images denser
than the large, *"as if the smallest had received from one and a half times to
twice as much exposure as the largest"*. Valenkoff's microdensitometer traces
quantified it as `δD = D_max − D_min` and found it **increases with grain
size**: *"very fine-grain plates show no effect, while in coarser-grain plates
the effect may be as much…"*

Two sub-effects we have been conflating (p. 875) — opposite signs, different
causes:

* **border effect** — density *rise* at the margin of a uniformly exposed
  image, caused by fresh developer diffusing in from the unexposed surround;
* **fringe effect** — fog density *fall* along the boundary of a well-exposed
  image, caused by bromide and other reduction products inhibiting development.

**Implication, not yet implemented.** `CouplerSpec` adjacency is
grain-independent; it should scale with `clump_um`, and the two sub-effects
should carry opposite sign rather than one lumped term.

## 4. Turbidity as the physical basis of MTF (p. 868 ff.)

Chapter "Turbidity, sharpness and the Eberhard effect": a plate exposed behind
a knife edge does not produce an image ending abruptly at the edge, because of
diffusion from refraction, reflection and diffraction. Mees prefers
*"diffusion"* over "scattering"/"irradiation" as less ambiguous. Grounds the
`MTFSpec` spread function in measured physics rather than a fitted `f50`.

## 5. Granularity vs graininess — Chapter XXI (pp. 835–839) and pp. 462–465

This is the cluster the first pass indexed but never read. Two distinct
findings, both bearing on our weakest-evidenced area.

### 5a. The two words are not synonyms, and we model only one of them

* **granularity** — the physical inhomogeneity of the developed image.
  Objective, measurable.
* **graininess** — the *visual impression* of that inhomogeneity. Depends on
  magnification and on the observer.

Verbatim (p. 835): grains are *"welded into clumps and to overlap each
other"*, and viewed in transmission one sees *"agglomerations of the grains
separated by spaces within the emulsion … **approximately six clumps being
piled over one another**"*.

**That "~6 clumps piled over one another" is a concrete number we do not
model.** `GrainSpec` has `clump_um` (lateral scale) but nothing for the depth
stacking. The overlap count through the emulsion depth is part of why sigma(D)
behaves as it does, and it is a candidate parameter.

Also, and this is the same conclusion we reached from scan analysis but from
the opposite direction (p. 835): graininess *"may appear at a magnification of
only a few diameters in the case of very fast emulsions — long before
individual grains or even clumps can be distinguished."* Visible graininess
does not require resolvable grain. It is why our renderer must scale grain by
`px_per_mm` rather than by resolved detail.

### 5b. Graininess is coupled to development extent — and to gamma
(pp. 462–463)

From "The Production of Graininess in Development":

* *"development terminated at an early stage results in less graininess than
  development carried nearly to completion"*;
* the mechanism, verbatim: *"a large number of small grains distributed at
  random have less interspacing than larger ones fewer in number"* — the
  physical basis for the clump-size/RMS relationship we currently fit
  empirically;
* Crabtree & Schwingel found *"the degree of development has more influence on
  the graininess of negative than positive materials, and it is now standard
  practice to develop motion-picture negatives to lower γ than the positive,
  both for the original and duplicates"*;
* developer chemistry barely matters *if development is carried to equal
  gammas* — except p-phenylenediamine and silver-halide solvents, which reduce
  grain at the cost of speed;
* Loveland's microscopy (p. 834): development starts at the acute edges of
  triangular/hexadecimal plate crystals and spreads inward; iodide-containing
  emulsions distort more and grain more than pure bromide.

**Implication.** Gamma and grain are not independent axes. If the
development-time curve families are ever adopted (41 sheets in the library
print them), grain must move with gamma, not stay fixed. The Crabtree &
Schwingel asymmetry also independently supports our print stocks carrying
higher gamma than camera negatives.

### 5c. Measurement method, for reference

Jones–Deisch blending-distance instrument (pp. 837–838): graininess taken as
proportional to the distance at which the grain pattern *"blends into a smooth
area"*, normalised by dividing by the blending distance of a fixed engraver's
half-tone screen to cancel observer variability. *"a material with a blending
distance of six feet was twice as grainy as one with a blending distance of
only three feet."*

## 6. Where the material sits, if the PDF is ever reloaded

| Topic | Pages |
|---|---|
| Granularity / graininess | 47 pages; **read: 462–465, 835–839**; unread remainder |
| Reciprocity and intermittency | 239–246 (contiguous chapter) |
| Callier effect | 235 (body), 238 (bibliography) |
| Turbidity, sharpness, Eberhard | 868–875 |
| Resolving power / acutance / spread function | 831–898 |
| Spectral sensitising | 55 pages from 133 |
| Grain size distribution | 51, 57, 544 |
| Interimage / DIR | **none** |

## Outstanding work from this source

1. Two-sided reciprocity (high- and low-intensity failure) — schema change.
2. Grain-size-scaled adjacency, with border and fringe as separate signs.
3. Callier q derived from `clump_um` as a consistency check.
4. Clump depth-overlap count (~6) as a grain parameter.
5. Couple grain to development extent if the curve families are adopted.

None of the three needs the PDF again; the numbers and relations above are
sufficient.
