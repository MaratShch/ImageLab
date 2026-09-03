# CHANGES 2026-08-13 — the spectral-sensitivity gap, partly closed

> **READ THIS FIRST.** An earlier draft of this document claimed the gap was
> closed on all three consumers. That was wrong and is corrected throughout.
> One consumer (colour-temperature balance) is a clean win. The other two were
> built, then found to reproduce a failure that a 2026-08-03 analysis had
> already documented and quarantined — projecting a sensitisation curve onto
> three visible primaries derives nonsense for extended-red and infrared
> stocks. They ship OFF by default, guarded, with the guard's own blind spot
> recorded. The earlier quarantine decision was correct.

Closes finding 1 of Appendix B (`FilmDatabase_Charecteristics.MD`): the
digitised per-layer spectral sensitivity curves existed for a large part of the
catalogue and **were read by nothing**. They are now consumed in both the
Python reference and the C++ engine.

## What was consuming what before

Three authored proxies stood in for the measured curves:

| Proxy | What it was | Now |
|---|---|---|
| `balance_gains()` / `AlgoBalanceGains` | blackbody radiance sampled at three **assumed** peaks, 600/550/450 nm | derived by integrating each layer's own measured curve — **default ON** |
| `profile.spectral_weights` | three authored numbers collapsing RGB to one silver record | derived from the pan curve — implemented, guarded, **default OFF**, see the failure section |
| `profile.taking_matrix` | authored 3×3 exposure mixing matrix | derived matrix computed and reported, **deliberately NOT wired in** — see below |

## Python — where the code lives

`PYTHON/profile_generator/film_sim.py`, new block inserted immediately before
`balance_gains()` (search for `MEASURED SPECTRAL SENSITIVITY`). Standalone
functions, as requested:

| Function | Purpose |
|---|---|
| `spectral_grid()` | the common 5 nm / 360–730 nm wavelength grid |
| `layer_sensitivities(profile)` | stored log curves → linear sensitivity on that grid; `None` when the stock has no curves |
| `planck_spd(kelvin)` | blackbody SPD on the grid |
| `spectral_layer_exposure(profile, spd)` | **the core integral** ∫S(λ)·E(λ)dλ per layer |
| `spectral_balance_gains(profile, scene_kelvin)` | colour-temperature gains from the curves |
| `spectral_monochrome_weights(profile)` | RGB→mono weights from the pan curve |
| `spectral_taking_matrix(profile, scene_kelvin)` | the derived mixing matrix |
| `spectral_exposure_report(profile, …)` | diagnostic: what was derived, what fell back, and how far the derived and authored values disagree |

Consumption sites in `film_sim.py`: the balance-gain block inside `simulate()`
(stage 3, stock colour balance), the monochrome collapse (stage 7), and the
taking-filter block (stage 2b, opt-in only). Three new `RenderSettings` flags:
`spectral_balance=True`, `spectral_mono=False`, `spectral_taking=False`.

Interpolation is done in **log** space before exponentiating, because a
sensitisation curve is smooth in log sensitivity and not in linear
sensitivity — linear interpolation across a 4-decade span lands near the larger
endpoint instead of the geometric middle. Outside a curve's own measured range
the sensitivity is **zero, never extrapolated**: an invented tail would bias the
integral in the direction that flatters the model.

## C++ — where the code lives

New files:

* `AlgoSpectralSensitivity.hpp` — interface plus the reasoning, including why
  the taking matrix is computed but not used.
* `AlgoSpectralSensitivity.cpp` — implementation. `AlgoSpectralHasCurves`,
  `AlgoSpectralBalanceGains`, `AlgoSpectralMonoWeights`,
  `AlgoSpectralTakingMatrix`.

Consumption sites, both scalar and AVX2:

* `Algo_03_Sim.cpp` and `AVX2/Algo_03_Sim.cpp` — stock colour balance. Tries
  `AlgoSpectralBalanceGains` first; on `false` the original `AlgoBalanceGains`
  proxy runs exactly as before.
* `Algo_07_Sim.cpp` and `AVX2/Algo_07_Sim.cpp` — monochrome collapse. Tries
  `AlgoSpectralMonoWeights` first; on `false` the authored triple is used.

All of it is setup domain — once per frame, never per pixel — so it computes in
`HighPrecType` and hands `AlgoType` to the pixel path. Two reasons, both
mandatory and both in the precision policy of Appendix C: the curves are log
values spanning 4–5 decades, and the Planck evaluation has λ⁵ in metres (~1e-32)
over an exponential reaching 53, which **flushes to zero in float32** (mechanism
M5). `expm1` rather than `exp(x)−1` for the same reason at the long-wavelength
end (M3).

## Cross-validation: Python vs C++

Independently written, then compared. Agreement to four decimals on every stock
tested:

| Stock | quantity | Python | C++ |
|---|---|---|---|
| KODAK_VISION3_250D_5207 | balance @3200 K | 1.653 / 1.0 / 0.502 | 1.6528 / 1.0000 / 0.5020 |
| AGFA_OPTIMA_100 | balance @3200 K | 1.687 / 1.0 / 0.482 | 1.6869 / 1.0000 / 0.4824 |
| KODAK_PORTRA_400 | balance @3200 K | 1.685 / 1.0 / 0.471 | 1.6847 / 1.0000 / 0.4707 |
| FUJI_NEOPAN_ACROS_100 | mono weights | 0.3423 / 0.3529 / 0.3048 | identical |
| AGFA_APX_100 | mono weights | 0.2620 / 0.3401 / 0.3980 | identical |
| EASTMAN_DOUBLE_X_5222 | mono weights | 0.2938 / 0.3655 / 0.3406 | identical |

## Measured effect on rendering

| Case | mean \|Δ\| | max \|Δ\| |
|---|---|---|
| ACROS 100, mono weights on vs off | 0.0041 | 0.112 (28.6 DN at 8-bit) |
| APX 100, mono weights on vs off | 0.0045 | 0.145 (36.9 DN) |
| PORTRA 400, balance on vs off @3200 K | 0.0184 | 0.078 (19.8 DN) |
| OPTIMA 100, balance on vs off @3200 K | 0.0155 | 0.076 (19.5 DN) |
| SVEMA_FOTO_65 (no curves), all flags on vs off | **bit-identical** | **0** |

The last row is the compatibility contract: **68 of 121 stocks carry no curves
and render exactly as before, bit for bit.**

## The failure this work rediscovered

`film_profiles.derived_spectral_response()` already existed, marked DIAGNOSTIC
ONLY, quarantined on 2026-08-03 for two documented reasons: the construction
cannot reach beyond the display gamut, and it is nearly a no-op for colour.
Neither Appendix B nor the Addendum recorded that, so this work rebuilt the same
construction and reproduced the same failure:

| stock | authored (correct) | derived | why |
|---|---|---|---|
| KONICA_INFRARED_750, peak 730 nm | 0.55 / 0.15 / 0.30 red-dominant | **0.161 / 0.193 / 0.646 blue-dominant** | the primary lobes are ~0 at 730 nm, so the only part visible to them is the intrinsic blue lobe |
| ROLLEI_INFRARED_400, ~96 % of peak at 660–680 nm | 0.52 / 0.20 / 0.28 | 0.348 / 0.314 / 0.338 | the 600 nm lobe reaches 670 nm poorly |

`spectral_monochrome_weights()` now refuses on two physical criteria: peak
sensitisation beyond 700 nm, or more than 15 % of sensitivity beyond it. The
first case is refused. **The second is not**, and that is recorded rather than
tuned away: `ROLLEI_INFRARED_400` passes both conditions while still deriving a
wrong answer, which is evidence that basis projection is the wrong construction
rather than a threshold in need of adjustment. `verify.py` asserts both the
refusal and the blind spot, so neither can regress silently.

Why the balance path is unaffected: it projects onto **no basis at all**. It is
a ratio of one curve integrated against two blackbody SPDs, and blackbody
radiance at 730 nm is perfectly real. The gamut-reach failure cannot apply.

## Two honest findings this work produced

**1. The authored monochrome weights are close to video luma** — and that
observation stands, but it does NOT follow that the derived triple is better.
The derived value depends on the assumed primary lobe width, which is an
assumption, not a measurement. Original note: The comment at the
consumption site says the weights are "the stock's own spectral sensitivity, not
video luma" — and the authored values are 0.27/0.55/0.18, which *is* luma. The
derived values for a panchromatic emulsion are much flatter, near
0.34/0.35/0.30. That flatness is the physically right answer and is why
panchromatic film renders a blue sky lighter than the eye sees it. This is a
**correction, not a refinement**, and it changes monochrome output visibly
(up to ~37 DN).

**2. The derived taking matrix is NOT wired in, deliberately.** It disagrees
with the authored matrix by up to **0.5** in an off-diagonal element, because
real layers overlap spectrally while the authored matrix is identity for an
ordinary tripack. But the pipeline already carries cross-channel mixing in
`dye_matrix` (stage 12) and `InterimageSpec` (stage 8b). Substituting a strongly
mixing taking matrix on top of those would apply the same physics two or three
times — precisely the double-counting failure the requirements document warns
about (A.5.12 OPT.6, D.2.5). Resolving it means deciding which stage owns which
mechanism and validating against a measured reference scan, and **no measured
reference exists in this project yet** (D.3.4, P4-1). Until then the derived
matrix is computed, reported by `spectral_exposure_report()`, and available
behind `spectral_taking=False`.

## The ceiling this does not lift

§1.3 of the requirements document is unchanged by this work: the input carries
three numbers per pixel already integrated through some other set of spectral
responses, so subjects metameric to the camera but not to the emulsion are
indistinguishable before the engine starts. This is the *illuminant-conditioned
integration* path (§4.6 option 1) — exact for neutrals and for illuminant
changes, approximate for saturated colour. Lifting that needs spectral input,
not more film data. The smooth-Gaussian primary basis is declared in both
implementations as an assumption, not hidden.

## Verification

* `verify.py`: **104 PASS / 2 FAIL** — the same two pre-existing failures
  (saturation hierarchy, red-blue interimage pair). Zero new.
* `AlgoSpectralSensitivity.cpp` compiles standalone; `Algo_03_Sim.cpp` and
  `Algo_07_Sim.cpp` pass `-fsyntax-only`; `AVX2/Algo_03_Sim.cpp` compiles clean
  once `AlgoType` is float32. The two AVX2 errors seen in the mounted tree are
  the **pre-existing** stale `AlgoTypes.hpp` (`using AlgoType = double`) and the
  missing `FastAriphmeticsAVX.hpp`, neither caused by this change.
* Appendix B finding 1 and Addendum item **P0-1** are now closed for the
  balance and monochrome paths; P0-1's taking-matrix half remains open by
  design, tracked above.
