# B&W silver-negative σ(D) — and a dead schema field

**Date:** 2026-08-18
**Task:** `DIGITIZATION_QUEUE.md` §3, "σ(D) heuristic sign, 103 stocks" — the largest
open item in the queue by stock count.
**Status:** evidence found and extracted. **No value and no code changed** — because the
first thing the investigation turned up makes the priority question different from the one
the queue asks.
**Reproduce with:** `python mees_granularity.py --root ../.. [--overlay DIR]`

---

## 1. The finding that comes first: nothing reads `sigma_shape_*`

Before extracting anything I checked what consumes the field. The answer is **nothing.**

| Implementation | Reads `sigma_shape_toe/mid/dmax`? | What it actually computes |
|---|---|---|
| `film_sim.py` (Python reference) | **No** — `grep` returns zero hits | `amp = sqrt(max(D − dmin, 0) + fog_grain)` |
| `Algo_11_Sim.cpp` (scalar) | **No** | same, in `AlgoAddGrain()` |
| `AVX2/Algo_11_Sim.cpp` | **No** | same |
| `film_profiles.hpp` | 4 hits — struct field + doc comment only | — |

`sigma_shape` is populated, validated by `verify.py`, emitted into the generated C++,
printed in `FilmActiveProfiles.md` — and **read by no renderer.** It is a dead field.

Two consequences, and the second is uncomfortable:

1. **Flipping the 103-stock heuristic would change zero pixels.** The queue ranks this item
   by stock count, which made it look like the highest-impact task on the list. As things
   stand its rendering impact is exactly nil.
2. **The VISION3 σ(D) adoption of 2026-08-17 is also inert at render time.** Four attempts,
   a full debugging round, a new tool and five regression checks produced *correct stored
   data* — and no change to any rendered frame. That work is not wasted (the data is right,
   the method rules it produced are valuable, and it is a precondition for the fix below),
   but its fidelity benefit is still pending, and the README's framing does not say so.

**This reframes the task.** The useful unit of work is not "fix the default for 103 stocks";
it is **wire σ(D) into the grain stage, then set the defaults from evidence** — one change,
made once, in three implementations.

---

## 2. The queue's premise for the B&W half is false

The queue justified leaving the heuristic alone like this:

> "…this branch also fills B&W SILVER negatives, for which the classical σ ~ √D rise is the
> textbook result and **nothing in the corpus contradicts it**."

Something in the corpus contradicts it.

**C. E. K. Mees, *The Theory of the Photographic Process*, Figure 302** — in
`PDF/PROFILES/RETRO/`, printed page 866 (PDF page 863), chapter "The Physics of the
Developed Image":

> "**Granularity-density curves of four negative emulsions** and of prints made from them
> measured on the Goetz-Gould trace evaluator."

Four B&W silver negative emulsions, granularity against density, data from Goetz and Gould.
This is the objective quantity, not subjective graininess.

### 2.1 The distinction that had to be checked first

The same chapter contains **both** quantities, and confusing them would have produced a
wrong answer that looked right:

* **Graininess** (Figures 287, 288, 290, 291) — subjective, from the Jones–Deisch blending
  distance. Figure 288 peaks near D = 0.3 and falls to zero at *both* ends, and Mees says
  why: *"photographic deposits of zero density or of infinite density obviously cannot have
  any apparent granular structure."*
* **Granularity** (Figure 302) — objective density fluctuation. Kodak publication E-58 puts
  the distinction in one line: *"Granularity describes the physical measurement of density
  variation."*

`sigma_shape` is σ_D. Only Figure 302 is admissible. Figure 288's peak-at-0.3 is a
different quantity and is not used here.

### 2.2 The load-bearing check: density or transparency units?

Figure 302's ordinate is **G**, the Goetz–Gould granularity constant — not σ_D at the 48 µm
diffuse-RMS aperture. Only the *shape* is taken, so a multiplicative constant is
irrelevant. But a density-versus-transparency mix-up would **not** be irrelevant: it would
multiply the curve by 10⁻ᴰ and could invert the conclusion.

Mees settles it on printed page 863, discussing these same curves:

> "Such an evaluation of graininess is based on **relative transparency**; to make it
> correspond with constant illumination, it must be evaluated on a basis of absolute
> transparency, which means that the curves must be **multiplied by the mean transparency
> T_m = 10⁻ᴰ**."

So G is in *relative* transparency units, G ∝ ΔT/T. And since
ΔD = −ΔT/(T ln10), we get **ΔT/T = −ln10 · ΔD**, i.e. **G ∝ σ_D at fixed aperture.**
The 10⁻ᴰ factor Mees describes converts to a *visual* basis — a different quantity, not used
here. Self-consistency check: G·10⁻ᴰ = (ΔT/T)·T = ΔT, absolute transparency deviation,
exactly as Mees states.

---

## 3. Extraction

`mees_granularity.py` re-derives all of it from the PDF and exits non-zero if it stops
reproducing. Method, and the two places it could have gone wrong:

**Calibration is fitted, not assumed.** Each panel is calibrated by least squares against
*its own* printed gridlines (D = 0.5/1.0/1.5/2.0, G = 0/0.05/0.10/0.15). This mattered: the
gridline spacing is not uniform — panel A's 0.05 steps measure 114 and 116 px — and a single
assumed scale missed panel A's 0.05 line by 3.4 px. Fitting per panel brings every residual
under **1.30 px**. Two assertions guard it: the fit residuals, and that printed ink is
actually present at every fitted gridline position (which catches a differently-rendered
page, something a residual check alone cannot see).

**Markers are found by annulus sampling.** For each pixel, the fraction of points on a
circle of radius 7–10 px that are ink. A ring marker or filled dot scores ~1.0; a curve
stroke passing through scores ~0.1, because a straight line meets a circle at two points.
One detector finds open, filled and cross-hatched markers.

**Negative and positive curves are separated by MARKER STYLE** — interior ink fraction
inside r = 4 px: 0.00 open ring, ~1.00 filled, 0.75–0.92 cross-hatched. This is method
rule 15 applied to a different plot.

> ⚠ **Style separation is not a nicety here, it overturned a wrong reading.** My first pass
> assigned families by position — "the lower curve is the negative" — supported by Mees's own
> sentence *"G_p is greater than G_N unless D_p is below approximately 0.4"*. The overlay
> proved that wrong. The two curves in one panel are plotted against **different
> abscissae**: each against *its own sample's* density, negative density for the N curve and
> positive density for the P curve. That is why they cross near D = 0.4, and it means
> position carries no family information at all in the low-density region. The style test is
> objective and it reversed four point assignments.

Panel D's marker styles do **not** separate cleanly (interior 0.29–0.73, no bimodality); its
points are taken from the printed `D_N` label and are flagged as such in the script.

### 3.1 Result — four B&W silver negatives, σ normalised at D = 1.0

| curve | toe D | toe/mid | peak D | peak/mid | top D | top/mid |
|---|---|---|---|---|---|---|
| A_N | 0.369 | 0.683 | 0.589 | 1.078 | 1.505 | 0.872 |
| B_N | 0.232 | 0.608 | 0.838 | 1.003 | 1.018 | 1.000 |
| C_N | 0.073 | 0.412 | 1.166 | 1.005 | 1.166 | 1.005 |
| D_N | 0.141 | 0.546 | 0.758 | 1.134 | 1.138 | 0.923 |

* toe/mid spread **0.412 – 0.683**, and it depends strongly on which density you call the
  toe: at D ≈ 0.07–0.14 it is **0.41–0.55**; at D ≈ 0.23–0.37 it is **0.61–0.68**.
* top/mid spread **0.872 – 1.005** — flat to gently falling above D = 1.0.
* All four peak between D ≈ 0.59 and ≈ 1.17. **None rises above D = 1.0.**

**What is NOT measured.** The negative curves stop between D = 1.02 and D = 1.51. The
schema's third anchor sits at **D = dmax**, which for a B&W negative in this database is
2.0–2.5. That anchor is **unmeasured** and nothing in this figure licenses a value for it.
Only the toe, the mid, and the *direction* above D = 1.0 are supported.

Caveats recorded rather than smoothed: the four emulsions are identified only as A, B, C, D
— no names, no speeds — which is acceptable for a **default** and not acceptable for any
per-stock adoption. All four come from one laboratory and one instrument, and the same
chapter documents that the Goetz–Gould apparatus needed a galvanometer-sluggishness
correction, without which "the general trend is upward". Tier: **[T2]** for the shape.

---

## 4. Three mutually inconsistent positions, stated plainly

| Position | toe (D≈0.1) | D = 1.0 | D = 1.2 | D = 2.2 |
|---|---|---|---|---|
| Current heuristic `_grain_v2` | 0.40 | 1.00 | — | **1.20** |
| Renderer's actual √(D−dmin+fog) law | **0.42** | 1.00 | 1.10 | **1.48** |
| Single-layer random-dot model (Nutting) | 0.17 | 1.00 | 1.28 | **3.32** |
| **Mees Fig 302, measured, 4 emulsions** | **0.41–0.55** | 1.00 | **0.87–1.01** | *not measured* |

The random-dot model is the one worth explaining, because it is the source of the "√D
textbook" intuition and it does **not** support it. With covered fraction p = 1 − 10⁻ᴰ and
σ_p = √(p(1−p)/N), converting to density gives σ_D ∝ √(p/(1−p)) — a *steep monotone rise*,
3.3× at D = 2.2. So the naive theory does not predict √D either; it predicts something much
steeper than the heuristic, and it disagrees with the measurement at both ends.

**Where the measurement and the theory can be reconciled** — stated as an engineering
conclusion, not as a sourced fact, because I have not read the primary papers: the
single-layer model assumes an infinitely thin layer of identical dots. A real emulsion is a
*thick multilayer with a wide grain-size distribution*. Once the upper layers are opaque,
further density comes from grains hidden behind others, which adds to the mean but much less
to the fluctuation — so σ_D saturates and turns over. Newson 2017 (`PDF/`) names the
literature for exactly this regime and it is worth acquiring:

* **Bayer, "Relation Between Granularity and Density for a Random-Dot Model", JOSA 54 (1964)**
* **Wilder, "Crowded Emulsions: Granularity Theory for Multilayers", JOSA 62 (1972)**
* **Trabka, "Alternating renewal model of photographic granularity", JOSA 63 (1973)**

None is on disk. They are the natural second source for this item.

**The agreement worth noting:** the √ law's value at D = dmin is **0.420**, against a
measured 0.41–0.55. The low-density limit is *right*, and for the right reason — sparse
non-overlapping grains are Poisson. The law fails only where the grains crowd.

---

## 5. The engine defect this quantifies

Because the renderer applies √(D − dmin + fog) to **every** stock, and measurement says the
dense end is flat (B&W) or falling (colour negative), today's grain amplitude is too high at
high density — for all 154 stocks, independent of what the data field says:

| | measured dmax/mid | engine dmax/mid | engine is |
|---|---|---|---|
| B&W silver negative (Mees ×4) | ~1.0 (flat, extrapolated from 0.87–1.01) | 1.48 | **1.48× too high** |
| Colour negative (VISION3 ×4) | 0.55–0.63 | 1.48 | **2.54× too high** |

On a negative, high density is a scene **highlight**. So the engine puts roughly 1.5–2.5×
too much grain into the brightest parts of the scene — and real film highlights read clean.
That is a visible, checkable fidelity error, and it is in the *algorithm*, not in the data.

---

## 6. Recommendation

**Do not change the 103-stock default yet.** It is inert, the evidence conflicts with a
theoretical model, and the change should be made once, together with the wiring. Changing it
now would be motion without effect.

**The item to do instead — needs your approval, it is an algorithm change in three
implementations:**

Wire σ(D) into stage 11 so the amplitude comes from the profile's anchors rather than a
hardcoded √ law, keeping √ as the low-density limit where it is correct:

1. **Interpolate the three anchors** (D = dmin, D = 1.0, D = dmax) and multiply the existing
   amplitude by the resulting shape factor, normalised to 1.0 at D = 1.0. The four VISION3
   stocks then behave as measured, and every stock with the default triple keeps the current
   behaviour exactly — so the change is *inert until a triple is set*, which makes it safe to
   land and review separately from any data change.
2. **Then** set the B&W default from §3.1 and the colour-negative default from the VISION3
   quartet, as two separate approvals with the numbers in front of you.
3. `fog_grain` stays as it is — it is the floor under the square root and is a different
   parameter from the toe anchor. They are not redundant, but the docstring does not say so
   and should.

Also worth deciding: the known three-anchor limitation is now confirmed by a *second*
independent dataset. VISION3 peaks at D ≈ 0.78 and Mees's four peak at D ≈ 0.59–1.17 — both
below or at the mid anchor. A three-anchor interpolation through (dmin, 1.0, dmax) cannot
represent an interior peak and understates it by roughly a quarter. **A fourth anchor, or a
short σ(D) array, would carry it** — a schema decision, better made before the wiring than
after.

## 7. What I changed

`mees_granularity.py` — new, the re-runnable extraction above. Nothing else. No profile
value, no default, no algorithm, no document other than this one.

## 8. Corrections owed to existing documents

Three places assert the false premise and should be corrected once you have decided the
above:

* `DIGITIZATION_QUEUE.md` §3 — "nothing in the corpus contradicts it" is wrong; Mees Fig 302
  does, and the blocker it names ("a measured σ(D) for a B&W silver negative") is now met.
* `film_profiles.py`, `_grain_v2` comment — same claim, same correction.
* `GrainSpec` docstring — says "the classical σ ~ √D rise is the textbook result" for B&W.
  The √ law is right at the toe and wrong at the dense end, and the docstring should also
  record that no renderer reads the field.
