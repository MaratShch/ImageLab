#pragma once

// ---------------------------------------------------------------------------
//  AlgoSpectralSensitivity.hpp
//
//  CONSUMER OF THE MEASURED PER-LAYER SPECTRAL SENSITIVITY CURVES.
//
//  WHY THIS FILE EXISTS
//
//  The profile database carries digitised spectral sensitivity for a large part
//  of the catalogue: film::SpectralSensitivity holds log_s_r / log_s_g / log_s_b
//  for colour stocks and log_s_pan for monochrome stocks, sampled on the grid
//  lambda_start_nm + i * lambda_step_nm. Until this file was written NOTHING in
//  the engine read any of it. The same physics was approximated by three
//  proxies instead:
//
//    * AlgoBalanceGains       sampled blackbody radiance at three ASSUMED peak
//                             wavelengths (600 / 550 / 450 nm) rather than
//                             integrating the real sensitisation
//    * profile.spectral_weights  three authored numbers collapsing RGB to one
//                             monochrome record
//    * profile.taking_matrix  an authored 3x3 exposure mixing matrix
//
//  Each proxy answers a question the measured curve answers exactly. This file
//  derives the answer from the curve where the curve exists, and returns a
//  failure indication where it does not, so every caller falls back to the
//  behaviour it had before. A stock with no curves renders bit-identically to
//  the way it rendered before this file existed - that is the compatibility
//  contract, and it is what makes the change safe to land.
//
//  WHAT THIS IS AND WHAT IT IS NOT
//
//  This is illuminant-conditioned integration. The layer sensitivities are
//  integrated against a real blackbody spectral power distribution and against
//  the input's assumed primaries, producing quantities that are DERIVED from
//  measurement rather than fitted by hand. It is exact for neutral subjects and
//  for a change of illuminant colour temperature.
//
//  It is NOT a full spectral render. The input image carries three numbers per
//  pixel that were already integrated through some other set of spectral
//  responses, so the information distinguishing two subjects that are metameric
//  to the camera but not to the emulsion was destroyed before this engine ran.
//  Saturated colour therefore remains approximate no matter how good the film
//  data is. That ceiling is a property of the input, not of this file, and it is
//  recorded here so the improvement is not overclaimed.
//
//  PRECISION
//
//  Everything here is setup domain: it runs once per frame, never per pixel. It
//  therefore computes in HighPrecType and hands AlgoType to the pixel path. Two
//  independent reasons, both mandatory:
//
//    * the stored curves are LOG sensitivity spanning four to five decades, so
//      exponentiating them produces linear values across five decades, and the
//      products against a blackbody SPD span more;
//    * the blackbody evaluation itself has a fifth power of a wavelength in
//      metres (order 1e-32) divided by an exponential whose argument reaches
//      about 53. In float32 that quotient flushes to zero and the result is not
//      merely imprecise, it is wrong. This is the same exponent-range hazard the
//      existing AlgoBalanceGains comment documents.
//
//  Sixty-odd integrals per frame at double cost nothing measurable.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// film::FilmProfile and film::SpectralSensitivity, whose curves this file reads,
// and film::Matrix3 for the derived taking matrix.
#include "film_profiles.hpp"


// ---------------------------------------------------------------------------
//  The common wavelength grid the integrals are evaluated on.
//
//  This is the grid the INTEGRALS are evaluated on. It is NOT the stored
//  sampling of any curve and does not claim to be. Stored curves are 10 nm for
//  48 of the 53 stocks that have one, 5 nm for one (FUJI_NEOPAN_1600, re-traced
//  2026-08-15 from its manufacturer datasheet at 0.557 nm per pixel), and
//  20-25 nm for four; they are interpolated up onto this
//  grid so the integral is not quantised to whatever sampling the source plot
//  happened to have. Interpolating up for the purpose of integration invents no
//  information - it moves the trapezoid rule's nodes and nothing else.
//
//  2 nm rather than 5, measured 2026-08-13. Against a smooth blackbody the
//  choice barely matters: a 5 nm grid differs from a 1 nm reference by 1.1e-3 on
//  the derived balance gains and a 10 nm grid by 3.2e-3. Against a NARROW-LINE
//  illuminant it matters enormously - with 5 nm mercury lines the red/green
//  layer ratio is wrong by 1.5 % at 5 nm, 52.7 % at 10 nm and 231 % at 25 nm,
//  while 2 nm matches the 1 nm reference exactly. Only blackbody SPDs are
//  integrated today, so 5 nm was adequate by coincidence, not by design. 2 nm
//  removes the trap that adding a fluorescent or LED illuminant later would
//  introduce a double-digit error with no warning.
//
//  Cost: ALGO_SPECTRAL_N goes 75 -> 186. All of it is setup domain, about sixty
//  integrals per frame, on stack arrays. Unmeasurable at frame scale.
//
//  360-730 nm covers every sensitisation in the catalogue including the
//  extended-red stocks; samples outside a given curve's own measured range
//  contribute zero, never an extrapolated tail. An extrapolated tail is an
//  invention, and it would bias the integral in the direction that flatters the
//  model.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SPECTRAL_LAMBDA_MIN  = 360.0;
constexpr HighPrecType ALGO_SPECTRAL_LAMBDA_MAX  = 730.0;
constexpr HighPrecType ALGO_SPECTRAL_LAMBDA_STEP = 2.0;

// Number of samples on the grid above, inclusive of both endpoints: 186.
constexpr int32_t ALGO_SPECTRAL_N =
    static_cast<int32_t>((ALGO_SPECTRAL_LAMBDA_MAX - ALGO_SPECTRAL_LAMBDA_MIN)
                         / ALGO_SPECTRAL_LAMBDA_STEP) + 1;


// ---------------------------------------------------------------------------
//  Smooth spectral basis standing in for the input's primaries.
//
//  These are NOT the sRGB primaries. sRGB primaries are defined by
//  chromaticity, not by a spectrum, and any RGB triple corresponds to
//  infinitely many spectra. Gaussian lobes centred on the primaries' dominant
//  wavelengths are the conventional smooth choice when a spectrum has to be
//  reconstructed from three numbers.
//
//  This choice is an ASSUMPTION of this path, declared here rather than buried,
//  and it is one of the reasons saturated colour stays approximate. It is stated
//  in the header so that a later reader looking for the source of a colour
//  discrepancy finds the assumption instead of having to infer it.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_SPECTRAL_PRIMARY_R_NM = 600.0;
constexpr HighPrecType ALGO_SPECTRAL_PRIMARY_G_NM = 540.0;
constexpr HighPrecType ALGO_SPECTRAL_PRIMARY_B_NM = 460.0;
constexpr HighPrecType ALGO_SPECTRAL_PRIMARY_WIDTH_NM = 55.0;


// ---------------------------------------------------------------------------
//  AlgoSpectralHasCurves
//
//  Whether this profile carries usable digitised sensitivity. Every function
//  below returns false and writes nothing when this is false, which is the
//  signal to the caller to keep its existing authored proxy.
//
//  Cheap: inspects vector emptiness only, no arithmetic.
// ---------------------------------------------------------------------------
bool AlgoSpectralHasCurves (const film::FilmProfile& profile) noexcept;


// ---------------------------------------------------------------------------
//  AlgoSpectralBalanceGains
//
//  Colour-temperature gains computed from the MEASURED curves.
//
//  The quantity is the same ratio AlgoBalanceGains estimates:
//
//      gain_c = INTEGRAL S_c(l) P(l, scene) dl / INTEGRAL S_c(l) P(l, stock) dl
//
//  but evaluated over the entire measured sensitisation instead of at one
//  assumed peak wavelength. Green is normalised to exactly 1.0, as in the proxy,
//  so this changes only the balance BETWEEN records and never overall exposure -
//  which is what keeps it from fighting the anchor solve at stage 8.
//
//  The difference from the proxy is not cosmetic. Measured on the current
//  database at 3200 K on daylight-balanced stocks: derived red gain 1.65 to
//  1.69 against the proxy's 1.32, and the derived value VARIES BY STOCK while
//  the proxy cannot, because the proxy asks about one wavelength and the real
//  red layer extends well past it into the region where tungsten light is much
//  stronger.
//
//  Returns false, writing nothing, when the profile has no three-layer curves -
//  including for a monochrome stock, where a per-channel balance has no meaning.
//
//  gains[0] = red, gains[1] = green (exactly 1.0), gains[2] = blue.
// ---------------------------------------------------------------------------
bool AlgoSpectralBalanceGains
(
    const film::FilmProfile& profile,
    const HighPrecType       sceneKelvin,
    AlgoType                 gains[3]
) noexcept;


// ---------------------------------------------------------------------------
//  AlgoSpectralMonoWeights
//
//  Monochrome R/G/B collapse weights derived from the measured pan curve.
//
//  The input has already been reduced to three channels, so the honest
//  derivation is: integrate the pan sensitivity against each input primary and
//  normalise the three integrals to sum to one. That is exactly the weight with
//  which each primary contributes to the single silver record.
//
//  This is where an orthochromatic emulsion earns its near-zero red weight from
//  its own measured curve instead of from an authored constant.
//
//  NOTE ON THE VALUES THIS REPLACES. The authored spectral_weights triple is
//  close to video luma for the panchromatic stocks (0.27 / 0.55 / 0.18), which
//  is precisely what the comment at the consuming site says it must not be. The
//  derived triple for a panchromatic emulsion is much flatter, near
//  0.34 / 0.35 / 0.30, and that flatness is why panchromatic film renders a
//  blue sky lighter than the eye sees it. Switching this on therefore CHANGES
//  monochrome rendering visibly. It is a correction, not a refinement, and it is
//  documented as such.
//
//  Returns false, writing nothing, unless the profile carries log_s_pan.
// ---------------------------------------------------------------------------
bool AlgoSpectralMonoWeights
(
    const film::FilmProfile& profile,
    AlgoType                 weights[3]
) noexcept;


// ---------------------------------------------------------------------------
//  AlgoSpectralTakingMatrix
//
//  The exposure-mixing matrix DERIVED from the measured curves. Element
//  [layer][primary] is that layer's response to that input primary under the
//  given illuminant:
//
//      M[l][p] = INTEGRAL S_l(l) * P_p(l) * I(l, T) dl
//
//  normalised so each ROW sums to one, which keeps a neutral input neutral and
//  confines the matrix's effect to cross-channel mixing - the part that is
//  genuinely the film's spectral character.
//
//  ***  THIS IS DELIBERATELY NOT CALLED BY THE PIPELINE.  ***
//
//  It is provided, and provided complete, because it is the physically correct
//  object and because a later spectral rework will need it. It is not wired in
//  because the pipeline ALREADY carries cross-channel mixing in two other
//  places - profile.dye_matrix at stage 12 and film::InterimageSpec at stage 8b
//  - and substituting a strongly mixing taking matrix on top of those would
//  apply the same physical effect two or three times over. Measured on the
//  current database, the derived matrix disagrees with the authored one by up to
//  0.5 in an off-diagonal element, so this is not a small risk.
//
//  Resolving it properly means deciding which stage owns which mechanism and
//  validating the result against a measured reference scan, and no measured
//  reference exists in this project yet. Until then the honest position is: the
//  derived matrix is computed, reported and available, and the authored matrix
//  keeps driving the render. Enabling this without that validation would be the
//  double-counting failure the requirements document warns about.
//
//  Returns false, writing nothing, unless the profile carries three-layer
//  curves.
// ---------------------------------------------------------------------------
bool AlgoSpectralTakingMatrix
(
    const film::FilmProfile& profile,
    const HighPrecType       sceneKelvin,
    film::Matrix3&           matrixOut
) noexcept;
