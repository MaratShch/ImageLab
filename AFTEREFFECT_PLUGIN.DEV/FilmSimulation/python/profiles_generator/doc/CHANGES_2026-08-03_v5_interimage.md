# CHANGES 2026-08-03 (v5) — interimage effects: the derivation, the provenance, and what is still estimated

> ⚠ **THIS FILE WAS MISSING UNTIL 2026-08-20, and four places cited it** as
> holding "the full derivation": `film_profiles.py:14256`, `doc/README.md`,
> `gen_active_profiles.py`, and the generated `FilmActiveProfiles.md`. The
> derivation survived only in a source comment. It is reconstructed here from
> that comment, from `_IIE_TIERS` / `_iie_solve` / `_interimage_for`, and from the
> `verify.py` guards that pin the result — i.e. from the code of record, not from
> memory. Where the reconstruction cannot recover something (see §7) it says so
> rather than inventing it.
>
> **Written 2026-08-20 as part of the DIR-coupler audit.** The dated title is
> kept because it names the schema change it documents.

---

## 1. What v5 added

`InterimageSpec`, and with it the **vertical** half of the DIR-coupler chemistry.

Developing silver releases a development inhibitor. That inhibitor diffuses in
two directions, and the two directions are two different visible effects from one
chemistry:

| direction | effect | where it lives |
|---|---|---|
| **laterally**, within a layer | edge effects, micro-contrast, the MTF overshoot | `CouplerSpec` — stage 9, since v1 |
| **vertically**, into the neighbouring layers | inter-image effects | `InterimageSpec` — stage 8b, added in v5 |

The consequence no per-channel curve can produce: **saturation rising without
gamma rising.** Each layer's effective exposure depends on what its neighbours
are doing, so a saturated red develops its cyan record against less inhibition
than a neutral of the same luminance does, and separates further.

## 2. The model

Applied as a correction to each layer's log exposure *before* the characteristic
curve:

```
logE_i' = logE_i + SUM_{j != i} a_ij * (D_j - d_ref_j)
```

* off-diagonal terms are **negative** — this is inhibition;
* the diagonal is **structurally zero** — a layer's effect on itself is already
  inside its own curve;
* `d_ref_j` is the density layer *j* reaches at the **mid-grey anchor**.

The equation is **implicit** — `D_j` depends on `logE_j'`, which depends on
`D_i` — and is solved by fixed-point iteration seeded with the uncorrected
densities. `iterations` is a profile field rather than a hardcoded loop because
each pass costs a full curve evaluation per channel and a weakly-coupled stock
converges in one.

⚠ **The mid-grey reference is what makes this a colour effect and not a tone
effect — but only AT the anchor.** On a neutral *at* mid-grey every
`(D_j − d_ref_j)` is zero and the correction vanishes. **Off-anchor neutrals do
move**, and that is correct rather than a defect: white-light gamma being lower
than separation gamma *is* the effect, and it is the very quantity the source
patent measures. Measured 2026-08-20 on `KODAK_PORTRA_400`, disabling stage 8b:
grey 0.18 moves **0.00/255**, grey 0.45 moves **16.3/255**, grey 0.06 moves
**6.7/255**. `verify.py`'s "interimage leaves a neutral untouched" guard tests
0.18 only — i.e. exactly the point where it cancels by construction — so that
guard proves less than its name suggests. Recorded, not repaired: the behaviour
is right and the guard's name is optimistic.

✅ **REPAIRED 2026-08-25 (queue item C20).** The guard is renamed to
"interimage leaves the ANCHOR neutral untouched (0.18, where it must)" and a
second guard now pins the off-anchor movement as intended behaviour, so the
property this section had to state in prose is measured. ⚠ **The figures moved
slightly and the current ones are 15.9/255 and 6.5/255**, not the 16.3 and 6.7
recorded here on 2026-08-20 — the render has changed since (measured MTF on ten
stocks, reciprocity wiring, the coupler gate), and this paragraph is left as the
2026-08-20 reading rather than silently restated. The guard asserts a band
(3–30/255 and 1–15/255) plus the ordering, not the exact values, precisely so
that legitimate downstream changes do not read as interimage regressions.

## 3. Provenance — patents, because datasheets do not publish this

**Tier 2**, upgraded from tier 3 on 2026-08-03. Not estimated from stock
generation; **derived from published measurements**.

⚠ **No manufacturer datasheet in the library publishes interimage data.** All
395 documents under `PDF/PROFILES` were searched. The omission is *systematic,
not accidental*: camera negative is characterised with a single white-light
exposure series, and the colour-separation exposure series that would reveal
interimage effects is only ever printed for print stocks.

Patents do publish it, because a patent claiming improved interimage effects has
to demonstrate them.

| source | what it gives |
|---|---|
| **US5273870A** (Agfa-Gevaert) | the metric, defined exactly as this model needs it, and three worked examples |
| **US5451492A** Table II, control row A | `gamma_white` — blue 0.726, green 0.545 (Status M, midscale, C-41 product negative) |
| **US4830954A** | the same per-receiver pattern independently: yellow 5–15 %, magenta 8–35 %, cyan 10–30 % |
| **US4725529A** Table 1 | the decisive one for the asymmetry question — see §5 |
| **US4729943A** | the negative-vs-reversal mechanism split — see §6 |
| Gschwind, Rosselet & Buser, *J. Photographic Science* **41** (1993) p. 86 | the one authoritative quantification located, and it is a **citation, not a document in hand** |

US5273870A's own definition, quoted:

> "the percentage steepening of color gradation during color separation exposure
> with light of the corresponding spectral region in relation to the color
> gradation established on exposure with white light"

citing T. H. James, *The Theory of the Photographic Process*, 4th ed. (1977),
pp. 574 and 614. Measured at density 1.0 over fog.

## 4. The conversion, and why the stored unit is a percentage

Separation exposure holds the other records constant, so the correction term is
constant and `gamma_sep` **is** the intrinsic curve gamma. Neutral exposure moves
all records together, giving

```
gamma_white = gamma0 / (1 - gamma0 * SUM_j a_ij)
hence        SUM_j a_ij = -(IIE/100) / gamma_sep
```

The sign matches the patent's own statement that white-light gradation is the
**lower** one.

⚠ **`_IIE_TIERS` therefore stores the patent's own units — IIE percentage per
RECEIVING layer — and not model coefficients.** The coefficient is derived per
stock from *that stock's* gamma, so a low-contrast and a high-contrast film both
reproduce the published IIE figure instead of sharing an absolute number that
suits only one of them. Verified: a fixed conversion gamma left blue **28 %**
below the patent target; per-stock conversion removes that.

| tier | (blue, green, red) IIE % | source |
|---|---|---|
| strong | 25 / 45 / 42 | US5273870A Ex. 1 invention, DIR in layer B |
| medium | 25 / 33 / 35 | US5273870A Ex. 3 invention |
| mild | 10 / 15 / 15 | US5273870A Ex. 1 **DIR-free control** |
| trace | 5 / 7 / 7 | half the iodide-only baseline |
| none | 0 / 0 / 0 | — |

⚠ **The closed-form algebra is not used as the answer.** It matched the
DIR-free control to 0.9 percentage points but **overshot the strong-DIR tier by
23 points**, because at that coupling strength the feedback between records stops
being linear and the curve's toe and shoulder start contributing. So the linear
result is a *seed*, and `_iie_solve` then corrects it against the model itself,
replicating the patent's measurement protocol (`_iie_measure`) with the same
iteration count and the same mid-grey reference. Same "fit through the full
pipeline" approach already used for grain rms, and for the same reason.
`verify.py` re-derives the published percentages for three stocks and requires
agreement to **< 1 pp**.

## 5. Why the matrix is symmetric per row — the question everyone asks first

Every stored profile has `a_rg == a_rb`, `a_gr == a_gb`, `a_br == a_bg`. The
carrier *can* express per-donor asymmetry; the data never does. That looks wrong
— blue is two hops from red, so surely it should couple more weakly?

**It is not wrong, and the evidence is specific.** The asymmetry is per
**receiver**, not per **distance**:

* blue receives weakly, green and red strongly, in **all three** US5273870A
  examples, and again in US4830954A;
* **US4725529A Table 1 settles it.** It puts the inhibitor in the **DEVELOPER**
  and applies it to **three separate single-layer coatings** — no layer stack at
  all, so no distance to travel — and still measures red receivers at
  0.43–0.72 ΔlogE against blue at 0.24–0.48. That is emulsion susceptibility,
  not geometry.
* a per-hop distance factor has **no numeric support in any of the nine patents
  surveyed**.

Donor identity therefore carries no weighting.

⚠ **A `verify.py` guard asserted the opposite for months.** "neighbour pairs
couple harder than the far red-blue pair" tested `|a_rg| > |a_rb|` — the
per-distance hypothesis — and sat in the FAIL baseline as a "known" failure. It
was unpassable by construction. Replaced 2026-08-20 with the assertion the
evidence supports.

## 6. Reversal stocks use a different mechanism, in a different part of the curve

US4729943A: a negative gets interimage effects "always … during chromogenic
development", while a reversal material gets them "by the release in the first
black-and-white developer of a development inhibitor", and those land in **high
dye-density areas**. The same patent notes the consequence — pushing them harder
**lowers neutral speed**, which is why manufacturers moved to second-developer
DIR instead.

`density_weighting` carries that: 0 = uniform across the curve (negative),
> 0 = scaled by the neighbouring layer's density relative to the mid-grey
reference (reversal). The weighting is normalised at the reference so a neutral
is untouched under either mechanism — that property is the entire point of the
stage and had to survive the mechanism split.

⚠ **`density_weighting = 0.65` is TIER 3 and it is the largest unbounded number
in the colour path.** The mechanism split is documented; the magnitude is not.
Measured 2026-08-20 on `FUJI_VELVIA_50`: the per-donor weight rises **0.44 → 1.82**
as density goes 0.2 → 3.2, giving a worst-case correction of **−0.58 logE ≈ 1.9
stops**, and disabling stage 8b moves saturated patches by up to **143/255**.
There is no cap. A saturating form with a measured asymptote is the honest
upgrade; it needs a measurement, not a better guess.

## 7. What this reconstruction cannot recover

The original 2026-08-03 file, if it ever existed on disk, would have carried the
session's own narrative — which examples were tried and rejected, what the first
conversion attempt got wrong, the per-stock table as adopted that day. **None of
that is recoverable**, and it is not invented here. Everything above is either
quoted from the source comment in `film_profiles.py`, read out of
`_IIE_TIERS` / `_iie_solve` / `_interimage_for`, or measured on 2026-08-20 and
labelled with that date.

## 8. How to replace the estimate with a measurement

One roll, and it is the cheapest high-value measurement on the whole queue:

1. a neutral step wedge;
2. the **same** wedge through red (W25), green (W58) and blue (W47B);
3. an empty-gate reference frame.

The single-colour ("self") gamma comes out steeper than the neutral gamma; the
ratio gives the coupling strength, and the movement of the other two layers'
densities gives the cross terms. This is **queue items D1 + D2 with three
filters added**, and it would take both halves of the DIR chemistry from tier 2/3
to tier 1.

## 9. Cross-renderer status, as of 2026-08-20

Both stages are live in **both** renderers — `film_sim.py` stages 8b and 9,
and `AlgoStage08b_Interimage` / `AlgoStage09_DirCoupler`, called from
`AlgorithmMain.cpp`. Until 2026-08-20 nothing compared them; `interimage_parity.py`
now does, probing the plugin's own translation units. Two findings from that:

* ⚠ **the density floor was inside the C++ stage and outside the Python one** —
  pipelines agreed, functions did not, by 0.26 D on a reversal stock. Fixed: the
  floor now lives inside `apply_dir_couplers`.
* ⚠ **the adjacency term is not the same effect in the two renderers at ordinary
  render sizes.** Stored `edge_um` is 9–13 µm, so at 40 px/mm the edge sigma is
  0.36–0.60 px, and below ~1 px sigma an FFT analytic Gaussian transfer and a
  truncated separable spatial kernel differ by ~1e-1 — measured. Above ~1.2 px
  they agree to 6e-5. Open, and it is a real fidelity question rather than a
  tolerance question.
