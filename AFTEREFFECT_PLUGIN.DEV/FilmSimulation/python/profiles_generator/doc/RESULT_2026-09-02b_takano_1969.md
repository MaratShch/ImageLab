# RESULT 2026-09-02b — Takano 1969, and the first check of the engine's aperture term

Queue **TK1–TK5**. Source: `PDF/PROFILES/RETRO/JAPAN/23_13.pdf`, supplied by the owner after the
G2/C44/C43/C4/C7 + J1/J2 batch. Kiyoshi Takano (高野潔), «写真フィルムの粒状性» / *"Granularity of
Photographic Film"*, テレビジョン (*J. Inst. Telev. Engrs. Japan*) **23**(1) 13–23 (1969). Ten
pages, real text layer, every page one 4299 × 6071 raster (≈600 ppi).

Reader: `takano_1969_granularity.py`, registered in `build.py`'s audit stage.
Knowledge base: `EMULSION_KNOWLEDGE_BASE.md` **§23k**.

---

## 1. What this document is

A review, like Ooue's pair. Almost all of it is other people's work redrawn, and **none of its
samples is a stock in this database**. Five items are still not incidental, and **two of them
touched code**.

---

## 2. ⚠ Fig. 8 — the first measurement the engine's aperture term has ever been checked against

Selwyn granularity `G = √A·σ(D)` against the scanning aperture's `√A`, 3–90 µm, for a colour
negative and for Neopan-SS. Selwyn's constant is *supposed* to be aperture-independent. **Both
curves saturate** — the colour negative at G ≈ 1.04, Neopan-SS at G ≈ 0.63 — which is exactly why
rms granularity replaced it.

`film_sim.grain_reference_energy` already models this: `2π ∫ (h·a)² f df` with a Gaussian aperture
of σ = size/4. ⚠ **That law unchanged, with nothing tuned but one overall constant and `clump_um`,
reproduces both traces to rms 0.007–0.020 in G over a 0.2–1.04 range.** The aperture handling was
inherited from theory and had never been checked against a measurement.

| `clump_gain` | `clump_um` colour neg | rms (G) | `clump_um` Neopan-SS | rms (G) |
|---|---|---|---|---|
| 0.30 (corpus min) | 6.20 | 0.0196 | 4.78 | 0.0162 |
| 0.85 (corpus median) | 3.22 | 0.0167 | 2.46 | 0.0155 |
| 1.50 | 2.38 | 0.0066 | 1.88 | 0.0104 |

⚠ **The size is not separately identifiable.** A 2.6× swing for a residual that moves 0.013 G — the
same non-identifiability the JPS 1965 crystal-size work hit, now quantified.

Traced with a slope-predictive follower: the figure carries error bars whose caps sit within a few
pixels of the curve, and two in-plot captions. Runs longer than 22 px are dropped as error bars;
everything else must land inside a tolerance that grows with the local slope and with how long the
walker has been coasting, and may not coincide with the already-traced curve.

---

## 3. Fig. 13 — two more autocorrelations, and a contradiction with Ooue

| sample | τ½ | `clump_um` = 1.334·τ½ |
|---|---|---|
| Neopan-SSS, D 2.0, Minidol 20 °C 10 min | 1.33 µm | **1.77 µm** |
| cine positive, D 1.7, D-16 20 °C 6 min | 0.65 µm | **0.87 µm** |

Traced **by row, not by column**: both curves are almost vertical over the first micrometre, so a
column carries a 25 px run and the half-width — the one number wanted — lands inside it. φ is
monotone falling on both, so a row scan gives exactly one x per curve.

⚠ **Neither curve goes negative, where Ooue's Fig. 24 does.** §23j.2 records Ooue's anti-correlated
ring past 12 µm as a refutation of the engine's Gaussian shape *and* of Sayanagi's Poisson
placement. Takano's optical autocorrelator shows both approaching zero from above, which the
Gaussian reproduces perfectly. **The disagreement is between two instruments, not between either
and the model**, and it is left standing rather than resolved by preferring the one that suits the
engine.

---

## 4. ⚠ The clump census — the number queue C45 was missing

| source | sample | `clump_um` |
|---|---|---|
| Takano Fig. 13 | cine positive, D-16 | 0.87 |
| Takano Fig. 13 | Neopan-SSS, Minidol | 1.77 |
| Takano Fig. 8 | Neopan-SS (at `clump_gain` 0.85) | 2.46 |
| Takano Fig. 8 | colour negative (at `clump_gain` 0.85) | 3.22 |
| Ooue Part 2 Fig. 24 | Neopan S, D 1.04 | 4.64 |

**median 2.46 µm** against 171 stored `clump_um_g` values running 0.66–40.0, median **13.0**, only
10 below 5 µm. ⚠ **The stored scale is 5.3× every measurement on file.**

⚠ **C45's document blocker is discharged and the row is now an owner decision.** It asked for *"a
granularity Wiener spectrum, or an rms at two or more apertures, for a NAMED stock"* — Fig. 8 is an
rms at thirteen apertures for two named samples. It also revises C45's own headline: the
disagreement is **5.3×, not the twenty** that row claimed from the JPS 1965 band alone.

**Nothing changed.** `clump_um` moves a pixel on 168 stocks (unlike C43, which was inert at the
shipped default); none of the five samples is a stock here; and §2 shows the value is only as well
determined as `clump_gain`, so a rescale must decide both at once or it is fitting one unknown with
two. `verify.py` pins both medians so neither can drift while the question is open.

---

## 5. Fig. 9 — a fourth confirmation that σ(D) turns over, and a disagreement on where

⚠ **The ordinate is broken between 0.03 and 0.06 with a 2.04× scale change across the break.** Only
the magenta curve is traced — the one unbroken stroke, isolated as a single 813 px connected
component, 691 clean columns after dropping the ones a scatter mark touches. The dash-dot yellow
and dashed cyan are **grid readings at ±0.002** and are reported as such: their segments are the
same size as the scatter of triangles and crosses and cannot be separated from them reliably.

Magenta, in this schema's anchors: **toe 0.301 @ D 0.30 / mid 1.000 / dmax 0.301 @ D 2.50, peak
1.002× at D 1.04.**

⚠ **A fourth independent confirmation of the 2026-08-17 correction** to `GrainSpec`'s docstring,
which used to say colour negatives are monotone rising — this time on a Japanese colour negative
measured in 1969 by another laboratory on another instrument.

⚠ **And it disagrees on where the maximum sits.** All eleven measured colour negatives here peak at
D 0.65–0.80 at 1.20–1.62×. This one peaks at D 1.04 at 1.00× — no interior peak above the mid
anchor at all. All eleven are Kodak ECN stocks, so `sigma_shape_peak` is a *family* measurement,
which `sigma_shape_measured` already refuses to generalise.

Layer ratios, both disagreeing with the corpus and ranked the way the explanation predicts:
cyan/magenta **1.15** against nine measured sheets' r/g 0.75–1.05 (10 % above the highest);
yellow/magenta **4.60** against their b/g 1.81–2.79 (65 % above). Takano reads **integral** colour
density through a filter, so each reading carries the orange mask and every layer's absorption in
that band — and the mask absorbs mostly blue. The corpus's ratios are per-layer **analytical**
densities. Two different quantities, not reconciled.

⚠ **Nothing written to a profile**: the sample is named only 「カラーネガフィルム」.

---

## 6. ⚠ eq (2) — the one thing here that changed the engine

Printed at p16 §3.2 (1), with T = 10^−D:

> σ(D) = 0.434·(σ(T)/T̄)·[ 1 + (1/12)(σ(T)/T̄)² + (1/80)(σ(T)/T̄)⁴ + … ]

⚠ **This is the correction the corpus was missing.** Provenance work here converted rms granularity
into density with the **first term alone**, and that is precisely what failed on BBC Report T-101
Fig. 26, whose σ(T)/T̄ runs 0.39–1.64 — the law `σ_D = 0.648·D^0.665` fitted from it was
**withdrawn** for that reason (ILFORD_HPS provenance note).

| σ(T)/T̄ | first order | full series | error of the first-order form |
|---|---|---|---|
| 0.05 | 0.02170 | 0.02171 | +0.02 % |
| 0.39 | 0.16926 | 0.17145 | **+1.3 %** |
| 1.00 | 0.43400 | 0.47559 | +9.6 % |
| 1.64 | 0.71176 | 0.93565 | **+31.5 %** |

**Adopted** as `film_sim.sigma_density_from_transmittance` plus a Newton inverse
`sigma_transmittance_from_density` (round trip closes to 1.1e-16). ⚠ **INERT** — no render path
calls either, so no stored value and no rendered pixel moves.

---

## 7. ⚠ eq (13) — the print chain, satisfied with one departure

Printed at p22 §5:

> F_pr(u,v) = F_pos(u,v) + F_neg(u,v)·R_pr²(u,v)·γ²

R_pr is the response of 「プリント光学系およびポジフィルム」 — **the printing optics *and* the
positive film** — and γ the positive's ΔD/Δlog E.

The engine satisfies all three terms by construction:

| term | where |
|---|---|
| `+ F_pos` | print grain is a separate field added at stage 14, after the print curve, so it adds in power |
| `·γ²` | stage 13's `log_e_print = offset − dens` carries the negative's grain inside `dens`, so the print curve's local slope multiplies it |
| `·R_pr²` | stage 10 applies `scan_t` to the negative density before the print curve; an amplitude transfer on a density field squares in the Wiener spectrum. With no scanner override `scan_f50 = settings.scanner_f50 or print_stock.mtf_f50`, so the default R_pr **is** the positive's own MTF — eq (13) with a contact printer |

⚠ **THE DEPARTURE: stage 14 also band-limits the print stock's own grain by that same transfer.**
eq (13) does not — F_pos is generated *in* the positive emulsion and is not imaged through the
positive's MTF. The duplication chain in the same function gets this right and says so in its own
comment ("This stage's own grain is created in THIS emulsion, so it is not blurred by this stage's
optics"). The print stage does not.

⚠ **Recorded, not changed.** When `scanner_f50` *is* set, `scan_t` is a real scanner and filtering
print grain by it is correct; the error appears only on the fallback, it moves a pixel on every
print render, and it is a **rendering** decision rather than a data one. The exact fix is to pass
the scanner transfer — not the print stock's MTF — as the print grain's band limit, which needs
`scan_t` split into its two factors. A `verify.py` guard pins the present state so a future fix
cannot land silently.

---

## 8. ⚠ FUJI NEOPAN asked for, searched for, and refused

Owner question, mid-batch: *can Neopan S be profiled from these papers?*

**What exists.** Three papers in this corpus measure Neopan **grain**: Ooue `23_7` Fig. 7 (σ against
D at a stated 10 µm aperture for Neopan S / D-76, SS / Microfine, SSS / Pandol), Ooue `22_91`
Figs. 24 and 26, Takano `23_13` Figs. 8 and 13. **That is one of about ten blocks a `FilmProfile`
needs, three times over.**

**What does not exist, searched across all 21 Japanese papers here.** Characteristic curve
(D vs log E), JIS/ASA speed, D-min / D-max / base density, spectral sensitivity, MTF or resolving
power, and an rms granularity at this corpus's 48 µm net-1.0 convention.

⚠ **The nearest miss is `26_172` Fig. 6** — Fujimura & Yamamoto 1963, *"Tone reproduction curves of
Neopan SS (35 mm), 20 °C 6 min"*, three negative sizes. It is a **system** curve, log B_or of the
enlarged print against log B_0 of the scene, carrying the paper's own curve and camera flare. The
paper prints neither, so it cannot be inverted to the negative's characteristic curve.

**Refused on method rule 18.** The precedent is `KODAK_8374` (queue C26): admitted on T-101's grain
data with everything else flagged inline — and even there T-101 printed speeds for five of its six
emulsions, while 8374's `exposure_index` had to be recorded as *"an acknowledged invented
placeholder"*. Neopan has no printed speed at all, so the whole tone scale would be ours: **one
measured block and nine invented ones, under a real film's name.**

**What would unlock it:** one Fuji data sheet, or a 写真年鑑-class table, carrying a Neopan
characteristic curve and a speed. Meanwhile the grain measurements do real work where they already
are — `NotFound.md` rows 1b and 2, and queues F2b and C45.

---

## 9. What changed

| file | change |
|---|---|
| `film_sim.py` | `sigma_density_from_transmittance`, `sigma_transmittance_from_density`, `SIGMA_T_SERIES_C2/C4` — **inert**, no render path |
| `film_profiles.py` | `_TAKANO_SELWYN_APERTURE_1969_NEG/_NEOPAN`, `_TAKANO_APERTURE_FIT_1969`, `_TAKANO_AUTOCORR_1969`, `_TAKANO_CLUMP_CENSUS_1969`, `_TAKANO_SIGMA_SHAPE_1969`, `_TAKANO_LAYER_SIGMA_1969` — reference tables, **no profile field touched** |
| `takano_1969_granularity.py` | new reader, registered in `build.py` |
| `verify.py` | five guards: eq (2) round trip, eq (2) magnitude, eq (13) structure + the recorded departure, the clump census disagreement, and Fig. 9's shape and peak-location disagreement |
| `doc/EMULSION_KNOWLEDGE_BASE.md` | §23k, eight subsections |
| `doc/DIGITIZATION_QUEUE.md` | §0 sixth-pass entry extended; TK1–TK5 added to §4 as closed; **C45 rewritten — its document blocker is discharged and it is now an owner decision**; census 106/83/23 → 111/88/23 |
| `doc/NotFound.md` | rows 1b and 2 updated |

**⚠ No stored profile value changed, and no rendered pixel moves.** Every adoption here is either a
module-level reference table or an inert helper.
