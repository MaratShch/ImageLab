#pragma once

// ---------------------------------------------------------------------------
//  AlgoGrain.hpp
//
//  Stage 11 of the film simulation pipeline: grain, in the density domain.
//
//  WHAT GRAIN ACTUALLY IS
//
//  Not noise added to an image. A developed emulsion is a countable population of
//  discrete silver crystals, or of dye clouds grown around them, and the density
//  at any small patch is a random variable because the number of crystals in that
//  patch is. Everything below follows from that one fact.
//
//  CONSEQUENCE 1 -- THE AMPLITUDE GOES AS THE SQUARE ROOT OF DENSITY
//
//  Poisson statistics: for a mean count N the standard deviation is sqrt(N). So
//  grain is strongest in the mid tones and upper mid tones, weaker in the deep
//  shadows where few crystals developed, and weaker again at Dmax where the
//  population saturates. A model that adds constant-amplitude noise gets the
//  shadows loudest, which is backwards and immediately visible.
//
//  The fog term keeps grain alive at zero exposure. Perfectly clean blacks are one
//  of the loudest digital tells there is, and real film has never had them.
//
//  CONSEQUENCE 2 -- IT HAS A SPECTRUM, AND THE SPECTRUM IS THE CHARACTER
//
//  The crystals have a size, so the field has a high-frequency rolloff set by the
//  mean developed clump diameter. They also CLUSTER, and the clustering tendency
//  adds a low-frequency lobe. Cubic crystals clump strongly, tabular T-grain
//  barely at all, and that single difference is what separates a velvety
//  fast black-and-white stock from the even sand of a modern colour negative.
//  Both stocks can share an RMS figure and look nothing alike.
//
//  CONSEQUENCE 3 -- ONE EMULSION MEANS ONE FIELD
//
//  A monochrome stock has a single silver image, so its grain is identical in all
//  three output channels - not three independent fields. So does an additive
//  colour stock: a reseau stock is one panchromatic emulsion behind a filter grid,
//  and it cannot have per-layer grain. Giving either three independent fields
//  produces coloured speckle, which is the signature of a colour-negative grain
//  model applied to black-and-white.
//
//  A tripack does have three separate emulsions, and they differ: the blue-
//  sensitive layer is on top and is the fastest, so it is the grainiest, typically
//  by about a third. That is where a per-channel RMS override earns its place.
//
//  WHY THE AMPLITUDE IS CALIBRATED AGAINST A CONTINUOUS INTEGRAL
//
//  This is the subtle part, and getting it wrong is invisible until it is
//  measured. RMS granularity is defined as the standard deviation of density read
//  through a 48 micrometre aperture. The obvious implementation calibrates the
//  discrete field so that its aperture-averaged deviation matches the target ON
//  THE RENDER GRID. That silently over-amplifies any stock whose grain is finer
//  than a pixel: all of its spectral energy folds back into the sampled band, so
//  the calibration inflates the amplitude to compensate for detail the grid cannot
//  hold. The symptom is a fine-grained stock rendering as grainy as a coarse one,
//  which is exactly backwards.
//
//  Integrating over the TRUE continuous spectrum instead makes the amplitude a
//  property of the emulsion alone. The scan MTF then band-limits it before
//  sampling, just as real optics do, so a fine-grained stock correctly renders
//  smoother than a coarse one at any resolution.
//
//  There is a real consequence: a 2K render genuinely shows less granularity than
//  a 6K render of the same negative, converging upward as the band widens. That is
//  not an artefact. It is why 4K scans of old negatives look grainier than the 2K
//  masters everyone remembers.
//
//  WHY THE BAND LIMIT IS THE SCAN MTF
//
//  The grain field is generated ALREADY band-limited by the scanner's optical
//  transfer, because that lens sits between the film and the sensor and filters
//  grain before it is sampled. This is the only way to stop fine grain aliasing
//  onto the pixel grid. It is also why stage 10 runs before this one.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// The separable Gaussian that shapes the noise spectrum.
#include "AlgoSeparableBlur.hpp"

// Counter-based generator for the white noise.
#include "AlgoCounterRng.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including GrainSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t
#include <cmath>     // AlgoGrainAmpBuild / AlgoGrainAmpAt: std::sqrt


// ---------------------------------------------------------------------------
//  Standard deviation of the measuring aperture, in millimetres.
//
//  The granularity metric is defined through a 48 micrometre circular aperture.
//  Treating that as a Gaussian of matching second moment gives a radius of 24
//  micrometres and a sigma of half that, so 12 micrometres = 0.012 mm.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GRAIN_APERTURE_SIGMA_MM = 0.012;

// ---------------------------------------------------------------------------
//  Ratio between the high-frequency rolloff and the low-frequency clumping lobe.
//
//  Six. The clumping lobe sits at a sixth of the crystal rolloff frequency, which
//  is to say clusters are about six crystals across. That is a property of how
//  emulsions flocculate, not a free parameter.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GRAIN_CLUMP_FREQ_RATIO = 6.0;

// ---------------------------------------------------------------------------
//  Upper limit and sample count for the radial spectral energy integral.
//
//  400 cycles per millimetre is far beyond where the aperture transfer has fallen
//  to nothing, so the integral is effectively complete. 16001 samples makes the
//  trapezoidal step 0.025 cycles/mm, which is small against every feature of the
//  integrand. It runs a handful of times per frame on scalars.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GRAIN_INTEGRAL_FMAX = 400.0;
constexpr int32_t      ALGO_GRAIN_INTEGRAL_N    = 16001;

// ---------------------------------------------------------------------------
//  Density floor added under the square root in the amplitude term for grain
//  created by a DUPE or PRINT emulsion, as distinct from the camera negative.
//
//  The camera negative uses the stock's own fog_grain figure. Intermediate and
//  print stocks do not carry one, and this is the value the reference model uses
//  for them.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_GRAIN_DUPE_FOG = static_cast<AlgoType>(0.15);

// ---------------------------------------------------------------------------
//  Clumping tendency assumed for stocks that do not carry a figure.
//
//  Duplicating stocks and print stocks are fine-grained and comparatively even,
//  and these are the values the reference model uses for them.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_GRAIN_DUPE_CLUMP_GAIN  = static_cast<AlgoType>(0.30);
constexpr AlgoType ALGO_GRAIN_PRINT_CLUMP_GAIN = static_cast<AlgoType>(0.25);

// ---------------------------------------------------------------------------
//  Anisotropy of an emulsion with no measured elongation.
//
//  1.0 is round grain, and it is what every stock without a figure carries. A
//  named constant rather than a bare literal because three call sites outside
//  the camera negative - the dupe generations at 13 and the print emulsion at
//  14 - pass it deliberately, and a naked 1.0 at those sites reads like a
//  placeholder rather than a statement about the stock.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_GRAIN_ANISOTROPY_NONE = static_cast<AlgoType>(1.0);

// ---------------------------------------------------------------------------
//  Positive floor on the anisotropy, matching the reference EXACTLY.
//
//  The reference writes max(anisotropy, 1e-6) before it multiplies the vertical
//  frequency axis by it, so that a zero or negative figure in the database
//  cannot collapse that axis. The same number is applied here, on the same side
//  of the same operation, so a malformed profile renders identically in both
//  engines rather than merely failing to crash in both.
//
//  ⚠ 1e-6 IS NOT A ROUND-NUMBER GUESS AND MUST NOT BE TIDIED. It is a literal
//  in the reference (film_sim.py:1085, FreqGrid.__init__); changing it here
//  would reintroduce, in miniature, the divergence this parameter exists to
//  close.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_GRAIN_ANISOTROPY_MIN = static_cast<AlgoType>(1e-6);



// ---------------------------------------------------------------------------
//  SCHEMA v48 -- THE GRAIN SPECTRUM AS A GAUSSIAN MIXTURE
//
//  WHY THIS EXISTS, AND WHY THERE IS NO FFT HERE.
//
//  The design specification (FGS-DDS-001 Rev. A §10.2.1, §19.1) synthesises the
//  grain field as IDFT[ DFT(white) * h(f) ] and accordingly requires a
//  real-to-complex 2-D transform, per channel, per frame, inside this engine.
//  This engine has never had one. It builds the field with SEPARABLE GAUSSIAN
//  BLURS, which is not an approximation of the frequency-domain form but an
//  exact identity - a product of Gaussian transfers is a Gaussian whose
//  variances add, and the v47 spectrum is a product of Gaussians.
//
//  The physical spectrum is a jinc and is NOT a Gaussian, so following §19.1
//  literally means writing a deterministic 2-D FFT into this file and into the
//  AVX2 twin, against a no-allocation, no-mutable-state policy and an 8 ms
//  frame budget.
//
//  ⚠ THE CONSTRAINT THAT DECIDES IT IS THAT PYTHON, THIS ENGINE AND THE AVX2
//  ENGINE MUST EXECUTE THE SAME ALGORITHM FLOW. Rather than give the engines a
//  transform they cannot afford, the SPECTRUM is put into a form all three
//  already execute: a Gaussian mixture
//
//      h(f) = sum_k w_k * exp(-2 pi^2 s_k^2 f^2)
//
//  fitted ONCE at build time by film_profiles.grain_gauss_terms and emitted
//  into the generated header as GrainSpec::jinc. No engine fits anything, so no
//  two engines can fit differently - which is the exact failure mode this file
//  has already suffered twice (FilmGrainSigma with no caller, anisotropy
//  emitted and never read).
//
//  Cost of the substitution, measured against the exact radius-averaged jinc
//  over the whole corpus: worst relative error 1.4e-03 in the reference energy
//  and in the rendered variance, on the coarsest stock in the database. For
//  scale, correcting the measuring aperture from the Gaussian stand-in to the
//  true disk - which v48 also does - is a 1.0e-02 to 5.3e-02 change. The
//  mixture is nowhere near the limiting error in this model.
//
//  THE LEGACY SPECTRUM IS THE SAME REPRESENTATION WITH ONE OR TWO EXACT TERMS,
//  built by AlgoGrainLegacyTerms below. So both spectral models leave through
//  one code path and there is no parallel implementation to keep in step.
// ---------------------------------------------------------------------------

static_assert(film::FILM_GRAIN_MAX_TERMS == 5,
              "AlgoGrain and the generated header must agree on the mixture "
              "size; changing it changes every emitted profile");

constexpr int32_t ALGO_GRAIN_MAX_TERMS = film::FILM_GRAIN_MAX_TERMS;

// ---------------------------------------------------------------------------
//  WHICH SPECTRAL MODEL THE CAMERA NEGATIVE RENDERS WITH.
//
//  ⚠ true, AND IT IS THE OWNER'S INSTRUCTION RATHER THAN THE SPECIFICATION'S
//  DEFAULT. Spec §10.3 keeps boolean_jinc opt-in until gate S5 and §18.1 wants
//  v48 defaults bit-identical to v47; the instruction of 2026-09-19 was to
//  REPLACE the existing grain model, so the physical spectrum is what ships and
//  the legacy one is retained, exact, as the comparison path.
//
//  ⚠ IT IS A CONSTANT AND NOT YET A CONTROL, DELIBERATELY. AlgoControls is the
//  plugin's public parameter surface and is mirrored into algo_control_enums.py
//  and into the host UI; adding a field to it is an ABI change and belongs in
//  its own commit, not folded into a physics change. Until then the legacy path
//  is reachable through AlgoGrainLegacyTerms(), which is what the parity
//  harness drives and what stages 13 and 14 use.
//
//  ⚠ SWITCHING IS A VISIBLE CHANGE ON 186 STOCKS AND IS NOT A REFINEMENT. The
//  physical diameter is a median 6.7x smaller than clump_um, so the spectrum's
//  half-power frequency moves UP by that factor, and because the calibration is
//  pinned through the 48 micrometre aperture - which sees almost none of the
//  band that moved - the VISIBLE variance goes UP rather than down. Measured
//  over the whole corpus, new/old rendered variance:
//
//      f50 =  40 c/mm   median 1.148   range 1.015 - 1.711
//      f50 =  80 c/mm   median 1.533   range 1.016 - 3.515
//      f50 = 120 c/mm   median 2.196   range 1.018 - 5.398
//
//  Per stock at 40 / 80 / 120: VISION3 50D 1.044 / 1.102 / 1.200, PORTRA 400
//  1.077 / 1.215 / 1.453, VISION3 500T 1.133 / 1.494 / 2.104, T-MAX 400
//  1.212 / 1.894 / 3.021, SVEMA FOTO 250 1.222 / 1.889 / 2.706, and ILFORD HPS
//  1.028 / 1.033 / 1.043.
//
//  ⚠ ILFORD HPS IS THE ONE THAT VALIDATES THE REST: its clump_um is a BBC
//  T-101 MEASUREMENT rather than an estimate, so its diameter barely moves and
//  neither does its render. The stocks that move are exactly the ones whose
//  grain size was a guess. Up to five times more visible grain at high scan
//  bandwidth is the opposite of the intuitive reading and is stated nowhere in
//  the specification.
// ---------------------------------------------------------------------------
constexpr bool ALGO_GRAIN_USE_JINC = true;

// ---------------------------------------------------------------------------
//  1 / (pi * sqrt 2): the LEGACY 1/e frequency f_hi of exp(-(f/f_hi)^2) to the
//  sigma of the equivalent Gaussian transfer exp(-2 pi^2 s^2 f^2).
//
//  ⚠⚠ THIS CORRECTS A TYPO THAT SHIPPED. Until schema v48 this file declared
//
//      // 1 / (pi * sqrt(2)) = 0.22508352815546.   <-- WRONG, as shipped
//
//  and the true value is 0.22507907903927651 - wrong from the sixth digit,
//  1.98e-05 relative. Nothing ever failed, because the REFERENCE never computes
//  a sigma on the legacy path at all: it evaluates exp(-(f/f_hi)^2) straight
//  onto its frequency grid, so this constant existed only here and had nothing
//  to disagree with. Every C++ render since has used a grain correlation length
//  1.98e-05 too long. It is fixed now because v48 gives the constant a second
//  consumer on the Python side, where a wrong value becomes a parity failure
//  instead of a secret.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GRAIN_SIGMA_PER_1E = 0.22507907903927651;

// ---------------------------------------------------------------------------
//  Radius of the 48 micrometre measuring aperture, in MILLIMETRES.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GRAIN_APERTURE_A48_MM = 0.024;


// ---------------------------------------------------------------------------
//  ⚠⚠ THE THRESHOLD BELOW WHICH A BLUR IS THE IDENTITY, AND IT IS WHERE THE
//  GRAIN STAGE'S TIME WENT.
//
//  A truncated Gaussian of half-width 1 has taps [e, 1-2e, e] after
//  renormalisation, with e = exp(-1/(2 sigma^2)) / (1 + 2 exp(-1/(2 sigma^2))).
//  Below this epsilon the outer taps cannot be represented against the centre
//  tap in float32, so the kernel IS the identity and AlgoGaussianBlurPlaneWrapXY
//  becomes two full passes over the plane to multiply by one.
//
//  That is the COMMON case once the spectrum is factored: the bare grain sigmas
//  at 4K run 0.289 down to 0.018 px, and four of the five rungs land here.
//
//  ⚠ FIXED CONSTANT, NOT A TUNING KNOB. The skip decision must be identical in
//  Python and in both engines or they stop computing the same operator, so the
//  rule is one comparison against one stored number and never a heuristic.
//  film_profiles.GRAIN_BLUR_IDENTITY_EPS holds the same value and
//  G-V48-IDENTITY measures the error it admits (2e-09 of the term's weight).
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_GRAIN_BLUR_IDENTITY_EPS = 1e-09;


inline bool AlgoGrainBlurIsIdentity (const HighPrecType sigmaPx) noexcept
{
    if (!(sigmaPx > 0.0))
        return true;

    if (std::ceil(4.0 * sigmaPx) > 1.0)
        return false;

    const HighPrecType e = std::exp(-0.5 / (sigmaPx * sigmaPx));

    return (e / (1.0 + 2.0 * e)) < ALGO_GRAIN_BLUR_IDENTITY_EPS;
}


// ---------------------------------------------------------------------------
//  AlgoBesselJ1 -- J1(x) to about 1e-7 absolute.
//
//  Abramowitz and Stegun 9.4.4 / 9.4.6, a line-for-line port of
//  film_profiles.bessel_j1. Hand-rolled because the C++ standard library ships
//  no Bessel function of the first kind of order one before C++17's
//  std::cyl_bessel_j, which is not available on every target this plugin builds
//  for and is not bit-specified where it is.
//
//  Needed because the 48 micrometre measuring aperture is a DISK and its
//  transfer is a jinc. See AlgoGrainAperture48.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoBesselJ1 (const HighPrecType x) noexcept
{
    const HighPrecType ax = (x < 0.0) ? -x : x;

    if (ax < 8.0)
    {
        const HighPrecType y = x * x;

        const HighPrecType num = x * (72362614232.0
            + y * (-7895059235.0
            + y * (242396853.1
            + y * (-2972611.439
            + y * (15704.48260
            + y * (-30.16036606))))));

        const HighPrecType den = 144725228442.0
            + y * (2300535178.0
            + y * (18583304.74
            + y * (99447.43394
            + y * (376.9991397
            + y))));

        return num / den;
    }

    const HighPrecType z  = 8.0 / ax;
    const HighPrecType y  = z * z;
    const HighPrecType xx = ax - 2.356194491;

    const HighPrecType p = 1.0
        + y * (0.183105e-2
        + y * (-0.3516396496e-4
        + y * (0.2457520174e-5
        + y * (-0.240337019e-6))));

    const HighPrecType q = 0.04687499995
        + y * (-0.2002690873e-3
        + y * (0.8449199096e-5
        + y * (-0.88228987e-6
        + y * (0.105787412e-6))));

    const HighPrecType r = std::sqrt(0.636619772 / ax)
                         * (std::cos(xx) * p - z * std::sin(xx) * q);

    return (x < 0.0) ? -r : r;
}


// ---------------------------------------------------------------------------
//  AlgoGrainJinc -- 2*J1(x)/x, equal to 1 at x = 0. Transform of a disk.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainJinc (const HighPrecType x) noexcept
{
    if (x <= 1e-12)
        return 1.0;

    return 2.0 * AlgoBesselJ1(x) / x;
}


// ---------------------------------------------------------------------------
//  AlgoGrainAperture48 -- transfer of the 48 micrometre CIRCULAR aperture.
//
//  ⚠⚠ THIS REPLACES A GAUSSIAN STAND-IN THAT WAS NEVER LABELLED AS ONE. v47
//  used exp(-2 pi^2 s^2 f^2) with s = 12 um - a Gaussian of matching second
//  moment - inside grainReferenceEnergy. The aperture is a disk:
//
//      A48(f) = 2 J1(2 pi f a) / (2 pi f a),    a = 24 um
//
//  whose first zero is at 25.4 cycles/mm, where the Gaussian is still passing
//  0.160 -- the stand-in keeps a sixth of the signal alive exactly where the
//  real aperture has none. Amplitude half point 14.69 cycles/mm exact against
//  15.62 Gaussian.
//
//  ⚠ THE CORRECTION IS NOT COSMETIC. This integral IS the amplitude
//  calibration, so changing the aperture model rescales every stock's rendered
//  grain by sqrt(E_gauss / E_jinc): 1.0104x at clump 1 um rising to 1.0527x at
//  13 um, worst on the coarsest stocks. The specification asks for the jinc in
//  §10.4 and says nothing about that consequence, which silently breaks its own
//  R-N2 bit-identity clause. Resolved by pairing the aperture with the spectral
//  model: legacy_gaussian keeps the Gaussian stand-in and stays exact, and the
//  jinc arrives with the physical spectrum, where the render is changing anyway.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainAperture48 (const HighPrecType fCyclesPerMm) noexcept
{
    return AlgoGrainJinc(2.0 * 3.1415926535897932385
                         * fCyclesPerMm * ALGO_GRAIN_APERTURE_A48_MM);
}


// ---------------------------------------------------------------------------
//  AlgoGrainApertureLegacy -- the v47 Gaussian stand-in. Retained so that
//  legacy_gaussian renders exactly as v47 did rather than approximately.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainApertureLegacy (const HighPrecType fCyclesPerMm) noexcept
{
    const HighPrecType k = 2.0 * 9.8696044010893586188
                         * ALGO_GRAIN_APERTURE_SIGMA_MM
                         * ALGO_GRAIN_APERTURE_SIGMA_MM;

    return std::exp(-k * fCyclesPerMm * fCyclesPerMm);
}


// ---------------------------------------------------------------------------
//  AlgoGrainLegacyTerms -- the v47 spectrum, as one or two EXACT terms.
//
//  h(f) = exp(-(f/f_hi)^2) * (1 + g * exp(-(f/f_lo)^2)),  f_lo = f_hi/6
//
//  expands to two Gaussians because variances add:
//
//      term 1, weight 1 : sigma_hi
//      term 2, weight g : sqrt(sigma_hi^2 + sigma_lo^2)
//
//  ⚠ THE WEIGHTS ARE 1 AND g AND DO NOT SUM TO ONE. This is spectral shaping of
//  a noise field, not an averaging filter. Handing the two lobes to a multi-lobe
//  blur helper that normalises by the weight sum would silently divide the whole
//  field by (1 + g).
//
//  Nothing is fitted and nothing is approximated here: film_profiles agrees with
//  this to 4.4e-16 over the whole band, which G-V48-LEGACY asserts.
// ---------------------------------------------------------------------------
inline film::GrainSpectrumTerms AlgoGrainLegacyTerms
(
    const AlgoType clumpUm,
    const AlgoType clumpGain
) noexcept
{
    film::GrainSpectrumTerms t{};

    for (int32_t i = 0; i < ALGO_GRAIN_MAX_TERMS; i++)
    {
        t.sigma_mm[i] = 0.0f;
        t.weight[i]   = 0.0f;
    }

    t.lobe_sigma_mm = 0.0f;
    t.lobe_gain     = 0.0f;

    if (clumpUm <= ALGO_ZERO)
    {
        t.count = 0;
        return t;
    }

    const HighPrecType fHi  = 1000.0 / (2.0 * static_cast<HighPrecType>(clumpUm));
    const HighPrecType sHi  = ALGO_GRAIN_SIGMA_PER_1E / fHi;

    t.count        = 1;
    t.sigma_mm[0]  = static_cast<float>(sHi);
    t.weight[0]    = 1.0f;

    const HighPrecType g = static_cast<HighPrecType>(
        MAX_VALUE(clumpGain, ALGO_ZERO));

    if (g > 0.0)
    {
        t.lobe_sigma_mm = static_cast<float>(
            ALGO_GRAIN_SIGMA_PER_1E / (fHi / ALGO_GRAIN_CLUMP_FREQ_RATIO));
        t.lobe_gain     = static_cast<float>(g);
    }

    return t;
}



// ---------------------------------------------------------------------------
//  THE PHYSICAL SPECTRUM, FITTED HERE RATHER THAN TABULATED.
//
//  ⚠ AN EARLIER DRAFT EMITTED THE MIXTURE AS PER-STOCK DATA FROM
//  film_profiles.grain_gauss_terms, which guarantees identical coefficients by
//  construction. It was withdrawn for two reasons, and the second is the real
//  one.
//
//    - It overflowed the generated data slots: 61 extra floats on 191 stocks
//      pushed slot 02 past its 112000-byte ceiling, which would have forced six
//      new .cpp files into the Visual Studio project by hand.
//    - A law that lives in a table is a law this engine cannot check. THIS FILE
//      HAS SHIPPED THAT MISTAKE TWICE: FilmGrainSigma() was audited on every
//      build and called by nothing, and GrainSpec::anisotropy was emitted for
//      months and read by nothing. Both survived because the engine held the
//      DATA and not the LAW. The mixture is computed here so that cpp_parity
//      can put this code and the Python code side by side and watch them agree.
//
//  Cost: about 25 ms per clip, at setup, once. Not per frame, and the 8 ms
//  frame budget (R-N3) is untouched.
//
//  ⚠ THE TWO IMPLEMENTATIONS AGREE TO ABOUT 1e-12, NOT BIT FOR BIT. numpy
//  solves the 4x4 normal equations through LAPACK and this file uses Gaussian
//  elimination with partial pivoting; both are float64 and both are correct,
//  and they land a few ULP apart. That is five orders of magnitude below the
//  mixture's own 1.3e-03 worst-case fit error, and cpp_parity asserts 1e-09.
// ---------------------------------------------------------------------------

//: sqrt(ln 2 / (2 pi^2)). Converts a HALF-POWER frequency to the sigma of the
//: Gaussian transfer exp(-2 pi^2 s^2 f^2) that has it.
//:
//: ⚠ THIS IS THE SAME NUMBER AS AlgoScanSigmaMm's, AND THAT ONE WAS WRONG. Stage
//: 10 declared it as 0.18738564618678 against a true 0.1873906251292776 --
//: 2.66e-05 relative, wrong from the fifth digit -- so every C++ render has
//: band-limited the whole image, not just the grain, with a scanner very
//: slightly too sharp. The reference never noticed because it evaluates the MTF
//: straight onto its frequency grid and derives no sigma at all, which is the
//: same blind spot that hid the kSigma typo above. Fixed in both places, and
//: film_profiles holds the correct value under three names already.
constexpr HighPrecType ALGO_GRAIN_SIGMA_PER_HALF_POWER = 0.1873906251292776;

//: Fit band and grid, and the mixture's shape. ⚠ Must equal
//: GRAIN_INTEGRAL_FMAX_CPMM / GRAIN_FIT_N / GRAIN_MIXTURE_TERMS /
//: GRAIN_MIXTURE_LADDER_RATIO in film_profiles exactly; the two fits are the
//: same fit or they are two different spectra.
constexpr int32_t      ALGO_GRAIN_FIT_N            = 8001;
constexpr int32_t      ALGO_GRAIN_MIXTURE_TERMS    = 5;
constexpr HighPrecType ALGO_GRAIN_LADDER_RATIO     = 2.0;
constexpr int32_t      ALGO_GRAIN_GAUSS_HERMITE_N  = 20;


// ---------------------------------------------------------------------------
//  AlgoGrainPhysShape -- the Boolean / random-dot amplitude transfer.
//
//      h(f)^2 = E_r[ r^4 b(f;r)^2 ] / E_r[ r^4 ],  b = 2 J1(2 pi f r)/(2 pi f r)
//
//  with r log-normal: sigma_ln = size_sigma_log and mu_ln = ln(d/2) -
//  sigma_ln^2/2, so that E[r] = d/2 exactly. The expectation is a 20-point
//  Gauss-Hermite quadrature in ln r.
//
//  ⚠ DISPERSION IS NOT A DETAIL. At d = 1 um the AMPLITUDE half point moves
//  705 -> 579 -> 327 cycles/mm for sigma_ln = 0 / 0.25 / 0.5, and the jinc's
//  zeros fill in completely. A factor of 2.16 in bandwidth, out of a field that
//  sat in the schema unread. (In the POWER convention an MTF sheet would use,
//  the same three are 514 -> 405 -> 203, a factor of 2.53 -- mind which one is
//  being quoted; this review confused them once.)
//
//  At sigma_ln = 0 this is |2 J1(pi f d)/(pi f d)| exactly.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainPhysShape
(
    const HighPrecType fCyclesPerMm,
    const HighPrecType grainUm,
    const HighPrecType sigmaLn
) noexcept
{
    // Probabilists' Gauss-Hermite nodes and NORMALISED weights: weight function
    // exp(-x^2/2), weights summing to one, so the sum is E[g(X)] for X ~ N(0,1)
    // directly. Generated by numpy.polynomial.hermite_e.hermegauss(20) and
    // pinned by G-V48-GH, which regenerates them and compares.
    static const HighPrecType kGhNode[ALGO_GRAIN_GAUSS_HERMITE_N] = {
        -7.619048541679758,   -6.510590157013654,
        -5.5787388058932015,  -4.734581334046055,
        -3.9439673506573163,  -3.1890148165533896,
        -2.458663611172368,   -1.7452473208141268,
        -1.042945348802751,   -0.3469641570813559,
         0.3469641570813559,   1.042945348802751,
         1.7452473208141268,   2.458663611172368,
         3.1890148165533896,   3.9439673506573163,
         4.734581334046055,    5.5787388058932015,
         6.510590157013654,    7.619048541679758
    };

    static const HighPrecType kGhWeight[ALGO_GRAIN_GAUSS_HERMITE_N] = {
        1.2578006724379264e-13, 2.4820623623151797e-10,
        6.127490259982928e-08,  4.4021210902308646e-06,
        1.2882627996192942e-04, 1.8301031310804924e-03,
        1.3997837447100996e-02, 6.150637206397696e-02,
        1.6173933398400003e-01, 2.607930634495548e-01,
        2.607930634495548e-01,  1.6173933398400003e-01,
        6.150637206397696e-02,  1.3997837447100996e-02,
        1.8301031310804924e-03, 1.2882627996192942e-04,
        4.4021210902308646e-06, 6.127490259982928e-08,
        2.4820623623151797e-10, 1.2578006724379264e-13
    };

    if (grainUm <= 0.0)
        return 1.0;

    if (sigmaLn <= 0.0)
    {
        const HighPrecType j = AlgoGrainJinc(3.1415926535897932385
                                             * fCyclesPerMm
                                             * (grainUm / 1000.0));
        return (j < 0.0) ? -j : j;
    }

    const HighPrecType mu = std::log(grainUm / 2.0) - 0.5 * sigmaLn * sigmaLn;

    HighPrecType num = 0.0;
    HighPrecType den = 0.0;

    for (int32_t i = 0; i < ALGO_GRAIN_GAUSS_HERMITE_N; i++)
    {
        const HighPrecType r  = std::exp(mu + sigmaLn * kGhNode[i]);
        const HighPrecType r2 = r * r;
        const HighPrecType r4 = r2 * r2;

        const HighPrecType b = AlgoGrainJinc(2.0 * 3.1415926535897932385
                                             * fCyclesPerMm * (r / 1000.0));

        num += kGhWeight[i] * r4 * b * b;
        den += kGhWeight[i] * r4;
    }

    const HighPrecType q = (den > 0.0) ? (num / den) : 0.0;

    return std::sqrt((q > 0.0) ? q : 0.0);
}


// ---------------------------------------------------------------------------
//  AlgoGrainHalfPowerFreq -- where AlgoGrainPhysShape falls to 0.5.
//
//  Bisection in log f, 200 halvings of a fixed bracket, with NO tolerance test
//  and NO early exit. That is deliberate: a convergence criterion would make
//  the answer depend on the platform's rounding of the criterion, and the
//  reference performs exactly these 200 halvings of exactly this bracket.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainHalfPowerFreq
(
    const HighPrecType grainUm,
    const HighPrecType sigmaLn
) noexcept
{
    HighPrecType lo = 1e-3;
    HighPrecType hi = 1e8;

    for (int32_t i = 0; i < 200; i++)
    {
        const HighPrecType mid = std::sqrt(lo * hi);

        if (AlgoGrainPhysShape(mid, grainUm, sigmaLn) > 0.5)
            lo = mid;
        else
            hi = mid;
    }

    return std::sqrt(lo * hi);
}


// ---------------------------------------------------------------------------
//  AlgoGrainJincTerms -- fit the physical spectrum as a Gaussian mixture.
//
//  Ladder: five sigmas, geometric with ratio 2, CENTRED on the Gaussian whose
//  half-power frequency matches the spectrum's own, capped at the top of the
//  integration band.
//
//    - The cap matters. Fine stocks roll off at 2000 to 6400 cycles/mm, and an
//      uncapped ladder puts four of its five rungs where neither the aperture
//      nor any scanner can see them.
//    - The centring matters. Hanging the rungs BELOW the anchor instead reaches
//      a 9.2e-02 fit error and a weight L1 of 4.42 at the extremes of the
//      corpus, because the spectrum above its half-power point is then left
//      with nothing to fit it.
//
//  Weights: linear least squares on a fixed 8001-point grid, with the last
//  weight eliminated against the exact constraint sum(w) = 1, so h(0) = 1 holds
//  by construction rather than by penalty.
//
//  ⚠ NO ITERATIVE SEARCH AND NO TOLERANCE. Ladder and solve are both closed
//  form given the inputs, so identical inputs give identical coefficients on
//  every machine and every run. An earlier version grid-searched the ladder
//  base and fitted a thousand times better; it was rejected because a search
//  can jump between local minima under a 0.1 % change in grain_um and silently
//  re-texture a stock between two builds.
//
//  ⚠ THE CLUSTERING LOBE IS APPLIED EXACTLY, NOT FITTED, and the first version
//  of this code got that wrong. Fitting the PRODUCT h_phys * (1 + g * lobe)
//  with one ladder fails badly when the two factors live decades apart -
//  SOVIET_PANCHROM_1939 has its lobe at 14 cycles/mm against a jinc half-power
//  of 460 - and least squares answers an unrepresentable target by cancelling
//  enormous opposite-signed terms. Measured over the corpus, L1(w) reached
//  1567, which in the AVX2 engine's float32 field is not a loss of accuracy but
//  a destroyed image. The product needs no fitting at all: the lobe is a
//  Gaussian, variances add, and multiplication distributes over the mixture, so
//  the lobe simply DOUBLES the term count at shifted sigmas with weights scaled
//  by g. No new error and no new conditioning.
// ---------------------------------------------------------------------------
inline film::GrainSpectrumTerms AlgoGrainJincTerms
(
    const HighPrecType grainUm,
    const HighPrecType sigmaLn,
    const HighPrecType clumpUm,
    const HighPrecType clumpGain
) noexcept
{
    film::GrainSpectrumTerms t{};

    for (int32_t i = 0; i < ALGO_GRAIN_MAX_TERMS; i++)
    {
        t.sigma_mm[i] = 0.0f;
        t.weight[i]   = 0.0f;
    }

    t.lobe_sigma_mm = 0.0f;
    t.lobe_gain     = 0.0f;
    t.count         = 0;

    if ((grainUm <= 0.0) || (clumpUm <= 0.0))
        return t;

    const int32_t k = ALGO_GRAIN_MIXTURE_TERMS;

    HighPrecType fTop = AlgoGrainHalfPowerFreq(grainUm, sigmaLn);

    if (fTop > ALGO_GRAIN_INTEGRAL_FMAX)
        fTop = ALGO_GRAIN_INTEGRAL_FMAX;

    HighPrecType sig[ALGO_GRAIN_MIXTURE_TERMS];

    for (int32_t i = 0; i < k; i++)
    {
        const HighPrecType e = static_cast<HighPrecType>(i)
                             - static_cast<HighPrecType>(k - 1) * 0.5;

        sig[i] = ALGO_GRAIN_SIGMA_PER_HALF_POWER
               / (fTop * std::pow(ALGO_GRAIN_LADDER_RATIO, e));
    }

    // Normal equations of the reduced (k-1) system, accumulated in one sweep of
    // the fit grid so the 8001-point design matrix is never materialised.
    const int32_t m = k - 1;

    HighPrecType gram[ALGO_GRAIN_MIXTURE_TERMS - 1]
                     [ALGO_GRAIN_MIXTURE_TERMS - 1] = {};
    HighPrecType rhs[ALGO_GRAIN_MIXTURE_TERMS - 1]  = {};

    const HighPrecType step = ALGO_GRAIN_INTEGRAL_FMAX
                            / static_cast<HighPrecType>(ALGO_GRAIN_FIT_N - 1);

    for (int32_t n = 0; n < ALGO_GRAIN_FIT_N; n++)
    {
        const HighPrecType f = static_cast<HighPrecType>(n) * step;

        HighPrecType a[ALGO_GRAIN_MIXTURE_TERMS];

        for (int32_t i = 0; i < k; i++)
            a[i] = std::exp(-2.0 * 9.8696044010893586188
                            * sig[i] * sig[i] * f * f);

        const HighPrecType tgt = AlgoGrainPhysShape(f, grainUm, sigmaLn);

        // Reduced column i is a[i] - a[k-1]; reduced target is tgt - a[k-1].
        const HighPrecType last = a[k - 1];
        const HighPrecType b    = tgt - last;

        for (int32_t i = 0; i < m; i++)
        {
            const HighPrecType ai = a[i] - last;

            rhs[i] += ai * b;

            for (int32_t j = 0; j < m; j++)
                gram[i][j] += ai * (a[j] - last);
        }
    }

    // Ridge at 1e-14 of the trace. The ladder keeps this well conditioned; the
    // term only guards the degenerate case where two rungs collapse onto each
    // other at the fTop cap.
    HighPrecType trace = 0.0;

    for (int32_t i = 0; i < m; i++)
        trace += gram[i][i];

    for (int32_t i = 0; i < m; i++)
        gram[i][i] += 1e-14 * trace;

    // Gaussian elimination with partial pivoting, in place.
    HighPrecType w[ALGO_GRAIN_MIXTURE_TERMS] = {};

    for (int32_t col = 0; col < m; col++)
    {
        int32_t piv = col;

        for (int32_t r = col + 1; r < m; r++)
        {
            const HighPrecType cand = (gram[r][col] < 0.0)
                                    ? -gram[r][col] : gram[r][col];

            const HighPrecType best = (gram[piv][col] < 0.0)
                                    ? -gram[piv][col] : gram[piv][col];

            if (cand > best)
                piv = r;
        }

        if (piv != col)
        {
            for (int32_t c = 0; c < m; c++)
            {
                const HighPrecType tmp = gram[col][c];
                gram[col][c] = gram[piv][c];
                gram[piv][c] = tmp;
            }

            const HighPrecType tmp = rhs[col];
            rhs[col] = rhs[piv];
            rhs[piv] = tmp;
        }

        const HighPrecType d = gram[col][col];

        if ((d > -1e-300) && (d < 1e-300))
            return t;                       // singular: no usable mixture

        for (int32_t r = col + 1; r < m; r++)
        {
            const HighPrecType fac = gram[r][col] / d;

            if (fac == 0.0)
                continue;

            for (int32_t c = col; c < m; c++)
                gram[r][c] -= fac * gram[col][c];

            rhs[r] -= fac * rhs[col];
        }
    }

    for (int32_t r = m - 1; r >= 0; r--)
    {
        HighPrecType acc = rhs[r];

        for (int32_t c = r + 1; c < m; c++)
            acc -= gram[r][c] * w[c];

        w[r] = acc / gram[r][r];
    }

    HighPrecType sum = 0.0;

    for (int32_t i = 0; i < m; i++)
        sum += w[i];

    w[k - 1] = 1.0 - sum;

    for (int32_t i = 0; i < k; i++)
    {
        t.sigma_mm[i] = static_cast<float>(sig[i]);
        t.weight[i]   = static_cast<float>(w[i]);
    }

    t.count = k;

    const HighPrecType g = (clumpGain > 0.0) ? clumpGain : 0.0;

    if (g <= 0.0)
        return t;

    // ⚠ THE LOBE IS CARRIED, NOT EXPANDED. Multiplying the mixture by
    // (1 + g G(s_lo)) would double the term count at shifted sigmas, which is
    // algebraically identical and ran ten full-plane blurs instead of one.
    t.lobe_sigma_mm = static_cast<float>(
        ALGO_GRAIN_SIGMA_PER_1E
        / ((1000.0 / (2.0 * clumpUm)) / ALGO_GRAIN_CLUMP_FREQ_RATIO));
    t.lobe_gain = static_cast<float>(g);

    return t;
}


// ---------------------------------------------------------------------------
//  AlgoGrainTermsFor -- the spectrum this render is using, for one channel.
//
//  boolean_jinc reads the build-time mixture out of the profile; legacy_gaussian
//  builds its own two exact terms. One entry point, so no call site has to know
//  which model is selected.
// ---------------------------------------------------------------------------
inline film::GrainSpectrumTerms AlgoGrainTermsFor
(
    const film::GrainSpec& grain,
    const int32_t          channel,
    const bool             booleanJinc
) noexcept
{
    const float clumpArr[3] = { grain.clump_um_r, grain.clump_um_g,
                                grain.clump_um_b };

    const float grainArr[3] = { grain.grain_um_r, grain.grain_um_g,
                                grain.grain_um_b };

    const AlgoType clump = static_cast<AlgoType>(clumpArr[channel]);

    if (!booleanJinc)
        return AlgoGrainLegacyTerms(clump,
                                    static_cast<AlgoType>(grain.clump_gain));

    return AlgoGrainJincTerms(static_cast<HighPrecType>(grainArr[channel]),
                              static_cast<HighPrecType>(grain.size_sigma_log),
                              static_cast<HighPrecType>(clumpArr[channel]),
                              static_cast<HighPrecType>(grain.clump_gain));
}


// ---------------------------------------------------------------------------
//  AlgoGrainReferenceEnergy -- the §10.4 calibration integral, over a mixture.
//
//      E = 2 pi * integral |h(f) A(f)|^2 f df
//
//  Continuous and grid-independent, which is the whole point: a grid-referred
//  calibration over-amplifies any stock whose grain is finer than a pixel,
//  because all of that stock's energy folds back into the sampled band and the
//  calibration inflates the amplitude to make up for detail the grid cannot
//  hold. The symptom is a fine stock rendering as grainy as a coarse one.
//
//  ⚠ T_scan IS DELIBERATELY EXCLUDED. The datasheet measures the film, not any
//  scanner; the band limit belongs to this render's observation chain, not to
//  the emulsion's calibration.
//
//  ⚠ THE QUADRATURE MATCHES film_profiles.grain_reference_energy_terms SAMPLE
//  FOR SAMPLE - same 16001 points, same trapezoidal rule, same endpoints - so
//  the two agree to float64 rounding rather than to a tolerance. Making one of
//  them "better" (adaptive, higher order) without the other is how a calibration
//  drifts apart at the fifth digit and nobody notices for a month.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainReferenceEnergy
(
    const film::GrainSpectrumTerms& terms,
    const bool                      jincAperture
) noexcept
{
    const HighPrecType step = ALGO_GRAIN_INTEGRAL_FMAX
                            / static_cast<HighPrecType>(
                                  ALGO_GRAIN_INTEGRAL_N - 1);

    HighPrecType acc  = 0.0;
    HighPrecType prev = 0.0;   // the integrand at f = 0 is zero, by the f factor

    for (int32_t i = 1; i < ALGO_GRAIN_INTEGRAL_N; i++)
    {
        const HighPrecType f = static_cast<HighPrecType>(i) * step;

        // ⚠ THE LOBE IS EXPANDED HERE AND ONLY HERE. This is a 1-D quadrature
        // run once per channel at setup, so the factoring that matters in the
        // renderer buys nothing and would only be a second spelling of the
        // spectrum to keep in step.
        HighPrecType h = 0.0;

        for (int32_t k = 0; k < terms.count; k++)
        {
            const HighPrecType s = static_cast<HighPrecType>(terms.sigma_mm[k]);

            h += static_cast<HighPrecType>(terms.weight[k])
               * std::exp(-2.0 * 9.8696044010893586188 * s * s * f * f);
        }

        if (terms.lobe_gain > 0.0f)
        {
            const HighPrecType sl =
                static_cast<HighPrecType>(terms.lobe_sigma_mm);

            h *= 1.0 + static_cast<HighPrecType>(terms.lobe_gain)
                     * std::exp(-2.0 * 9.8696044010893586188 * sl * sl * f * f);
        }

        const HighPrecType a = jincAperture ? AlgoGrainAperture48(f)
                                            : AlgoGrainApertureLegacy(f);

        const HighPrecType ha = h * a;

        const HighPrecType cur = ha * ha * f;

        acc += 0.5 * (prev + cur) * step;

        prev = cur;
    }

    return 2.0 * 3.1415926535897932385 * acc;
}


// ---------------------------------------------------------------------------
//  Build one grain field from a spectral MIXTURE. The v48 entry point.
//
//  Same contract as AlgoMakeGrainField below, with the spectrum supplied as
//  terms rather than as (clumpUm, clumpGain), and with the aperture model
//  selected to match.
//
//  ⚠ SCRATCH REQUIREMENT CHANGED. The two-lobe form needed three scratch
//  planes; a mixture of up to ten terms needs an ACCUMULATOR as well, because
//  each term is blurred from the same white field and summed. pScrAccum is that
//  accumulator and must be distinct from every other plane.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
//  TEMPORAL GRAIN (spec: temporal film-grain model, v49)
//
//  ⚠⚠ THE MODEL SEPARATES TWO THINGS THAT BOTH LOOK LIKE GRAIN AND BEHAVE
//  NOTHING ALIKE IN TIME.
//
//    EMULSION  Every frame of a motion picture is a DIFFERENT PIECE OF FILM
//              and shares no silver grains with its neighbours, so the
//              physically correct frame-to-frame correlation of camera
//              negative grain is ZERO. That is what this engine already did
//              and it was right. film::TemporalSpec::grain_frame_correlation
//              is 0.0 on every stock in the database for that reason; the
//              mechanism below exists for the cases where it is not -- a
//              frozen frame, an optical step printer holding one negative
//              frame across several print frames, and any stock a future
//              measurement shows to persist.
//
//    SENSOR    A scanner's fixed-pattern noise is IDENTICAL on every frame it
//              digitises. Perfectly correlated, by definition, at every lag.
//              Before v49 nothing in this engine could express it.
//
//  ⚠ STATELESS, WHICH IS WHY IT IS NOT AN AR(1) RECURSION. The textbook way to
//  correlate frames, E_n = rho E_{n-1} + sqrt(1-rho^2) W_n, carries state: it
//  makes frame 5000 depend on frames 0..4999 and a renderer that cannot start
//  at an arbitrary frame is useless to a host that scrubs a timeline or a farm
//  that distributes frames. The whole RNG in this project is counter based for
//  the same reason. So the correlation is a SLIDING WEIGHTED SUM over
//  per-frame white fields instead, each of which is a pure function of its own
//  frame index:
//
//      E_n = sum_j c_j W_{n+j} / sqrt(sum_j c_j^2),   c_j = exp(-|j| / tau)
//
//  ⚠ VARIANCE IS PRESERVED EXACTLY at every setting of both controls, so
//  neither can change HOW MUCH grain there is -- only how it behaves in time.
//  The granularity calibration above is untouched by either, which is what
//  makes rmsGranularity still mean the number the datasheet prints.
// ---------------------------------------------------------------------------

//: Frame index reserved for the fixed-pattern draw. Must be unreachable by a
//: real clip: AlgoRngCounter casts the frame index through uint32 and the cast
//: wraps, so a sentinel inside the reachable range would collide with one real
//: frame and freeze that frame's emulsion grain into the pattern -- a defect
//: visible once in a 68-year clip and unreproducible when reported.
constexpr int32_t ALGO_GRAIN_FIXED_FRAME = -2147483647 - 1;

//: Largest half-window the kernel builds. Caps the cost at 13 field draws per
//: frame; the truncation is absorbed by the renormalisation, never left to
//: bias the variance. film_sim.TEMPORAL_MAX_TAPS must agree.
constexpr int32_t ALGO_GRAIN_TEMPORAL_MAX_TAPS = 6;

//: Weights whose own lag-1 autocorrelation equals `rho`, normalised to unit
//: variance. Returns tapCount = 1 and w[0] = 1 for rho <= 0, which is the
//: identity and the pre-v49 path bit for bit.
//:
//: ⚠ tau IS SOLVED, NOT ASSUMED. The kernel's autocorrelation
//: r(d) = sum_j c_j c_{j+d} / sum_j c_j^2 is close to exp(-d/tau) but not equal
//: to it, so tau is bisected until the kernel's OWN r(1) hits the target. A
//: fitted constant here is exactly the shortcut this project refuses.
inline int32_t AlgoGrainTemporalKernel
(
    const HighPrecType rho,
    HighPrecType* RESTRICT pW            // at least 2*MAX_TAPS+1 entries
) noexcept
{
    if (!(rho > 0.0))
    {
        pW[0] = 1.0;
        return 1;
    }

    const HighPrecType r = (rho < 0.95) ? rho : 0.95;
    const HighPrecType tau0 = -1.0 / std::log(r);

    int32_t taps = static_cast<int32_t>(std::ceil(3.0 * tau0));
    if (taps < 1)                              taps = 1;
    if (taps > ALGO_GRAIN_TEMPORAL_MAX_TAPS)   taps = ALGO_GRAIN_TEMPORAL_MAX_TAPS;

    const int32_t n = 2 * taps + 1;

    HighPrecType lo = 1.0e-3;
    HighPrecType hi = 60.0;
    for (int32_t it = 0; it < 60; it++)
    {
        const HighPrecType mid = 0.5 * (lo + hi);
        HighPrecType num = 0.0;
        HighPrecType den = 0.0;
        HighPrecType prev = 0.0;
        for (int32_t i = 0; i < n; i++)
        {
            const HighPrecType c =
                std::exp(-std::fabs(static_cast<HighPrecType>(i - taps)) / mid);
            den += c * c;
            if (i > 0) { num += prev * c; }
            prev = c;
        }
        if ((num / den) < r) { lo = mid; } else { hi = mid; }
    }

    const HighPrecType tau = 0.5 * (lo + hi);
    HighPrecType sumsq = 0.0;
    for (int32_t i = 0; i < n; i++)
    {
        const HighPrecType c =
            std::exp(-std::fabs(static_cast<HighPrecType>(i - taps)) / tau);
        pW[i]  = c;
        sumsq += c * c;
    }
    const HighPrecType inv = 1.0 / std::sqrt(sumsq);
    for (int32_t i = 0; i < n; i++) { pW[i] *= inv; }
    return n;
}

void AlgoMakeGrainFieldTerms
(
    AlgoType* RESTRICT              pDst,
    AlgoType* RESTRICT              pScrNoise,
    AlgoType* RESTRICT              pScrLobe,
    AlgoType* RESTRICT              pScrWork,
    const int32_t                   sizeX,
    const int32_t                   sizeY,
    const int32_t                   pitch,
    const film::GrainSpectrumTerms& terms,
    const bool                      jincAperture,
    const AlgoType                  rmsGranularity,
    const AlgoType                  scanSigmaPx,
    const AlgoType                  pxPerMm,
    const AlgoType                  anisotropy,
    const eALGO_RNG_STAGE           rngStage,
    const uint32_t                  seed,
    const int32_t                   frameIndex,
    //: ⚠ DEFAULTED TO ZERO SO THE DUPE AND PRINT GRAIN PATHS KEEP THE
    //: PRE-v49 BEHAVIOUR WITHOUT RESTATING IT. Those two stages model
    //: SEPARATE PIECES OF FILM -- an intermediate and a release print -- and
    //: their grain is independent of the negative's and of each other's, so
    //: zero is the physically correct value there and not merely the
    //: convenient one.
    const AlgoType                  frameCorrelation = ALGO_ZERO,
    const AlgoType                  fixedFraction    = ALGO_ZERO
) noexcept;



// ---------------------------------------------------------------------------
//  THE COUNT GATE AND THE MARGINAL CORRECTION (spec ch. 12) -- R-S4(a), R-S6
//
//  A Gaussian field has zero skewness at every density. Real film does not:
//  where the developed grains are COUNTABLE the density marginal is compound
//  Poisson and positively skewed. Solving N_elem(D) = 25 over this corpus puts
//  that region at net D 0.058 for the median stock and up to 0.87 for the
//  coarsest -- the shadows of a negative, and exactly where real film shows
//  discrete salt-like grain rather than smooth noise.
//
//  ⚠ THE GATE IS COMPUTED AND NEVER CHOSEN. N_elem comes from the physical
//  diameter through Nutting's relation and the noise-equivalent element area.
//  Nothing in it reads a stock name, an era or a taste setting, which is the
//  actual content of R-S6.
//
//  ⚠⚠ AND THE FIELD IS TRANSFORMED RATHER THAN REPLACED BY A DOT RENDERER,
//  WHICH IS A COSTED DEPARTURE FROM SPEC §12.4.2. The specification places
//  every grain -- cell grid, Poisson count per cell, log-normal radius, splat
//  through sixteen bucketed kernels. Costed against this corpus at 3840x2160
//  that is about five million grains per channel over five per cent of the
//  frame at the gate's LOWER edge and four times that at its upper: 25 to
//  100 ms of scatter on top of the Gaussian branch, against an 8 ms whole-frame
//  budget (R-N3). It cannot run at 4K, and §23's "cost proportional to the
//  low-count area" is true and still unaffordable, because that area holds
//  hundreds of millions of grains.
//
//  What the sparse branch DELIVERS, per §12.4.3, is a skewed marginal. The
//  spec's own blend sqrt(a) F + sqrt(1-a) S has, for independent unit-variance
//  F and S, exactly mean 0, variance 1 and third moment (1-a)^(3/2) gamma_S.
//  The Cornish-Fisher transform below reproduces all three at O(1) per pixel.
//  The fourth and higher moments it does not, and the spatial arrangement of
//  individual dots it does not reproduce at all -- see AlgoGrainMarginalCoeff.
// ---------------------------------------------------------------------------

//: log10(e). Nutting's constant: D = NUTTING_C * abar * n.
constexpr HighPrecType ALGO_GRAIN_NUTTING_C = 0.4342944819032518;

//: The gate's two edges in grains per resolution element (spec §18.5).
constexpr HighPrecType ALGO_GRAIN_N_MIN = 25.0;
constexpr HighPrecType ALGO_GRAIN_N_HI  = 100.0;

//: Monotonicity clamp on the Cornish-Fisher coefficient. See
//: AlgoGrainMarginalCoeff; film_profiles.GRAIN_MARGINAL_COEFF_MAX must agree.
constexpr HighPrecType ALGO_GRAIN_MARGINAL_COEFF_MAX = 0.1;

//: Quadrature of the noise-equivalent element integral. Fixed grid, shared with
//: film_profiles.grain_element_area_um2 sample for sample.
constexpr int32_t      ALGO_GRAIN_ELEM_N      = 4001;
constexpr HighPrecType ALGO_GRAIN_ELEM_FMAX   = 2000.0;


// ---------------------------------------------------------------------------
//  Mean PROJECTED grain area, E[pi r^2], square micrometres.
//
//  ⚠ E[r^2] AND NOT E[r]^2 -- this is R-S4(a), the half of the requirement the
//  first v48 pass did not build. For a log-normal radius with E[r] = d/2 held
//  fixed, E[r^2] = (d/2)^2 exp(sigma_ln^2), so the mean AREA exceeds the area
//  of the mean radius by 13 % at the corpus-typical sigma_ln 0.35 and 35 % at
//  the 0.55 of a fast pushed stock.
//
//  ⚠ AND IT IS WRONG IN THE DANGEROUS DIRECTION IF OMITTED: ignoring it
//  OVERCOUNTS the grains needed to reach a density, which makes the Gaussian
//  marginal look safer than it is -- in a gate whose only job is to decide when
//  the Gaussian fails.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainMeanAreaUm2
(
    const HighPrecType grainUm,
    const HighPrecType sigmaLn
) noexcept
{
    if (grainUm <= 0.0)
        return 0.0;

    const HighPrecType muR = 0.5 * grainUm;
    const HighPrecType s   = (sigmaLn > 0.0) ? sigmaLn : 0.0;

    return 3.1415926535897932385 * muR * muR * std::exp(s * s);
}


// ---------------------------------------------------------------------------
//  Developed grain centres per square micrometre at a given NET density.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainCenterDensity
(
    const HighPrecType netDensity,
    const HighPrecType grainUm,
    const HighPrecType sigmaLn
) noexcept
{
    const HighPrecType a = AlgoGrainMeanAreaUm2(grainUm, sigmaLn);

    if (a <= 0.0)
        return 0.0;

    const HighPrecType d = (netDensity > 0.0) ? netDensity : 0.0;

    return d / (ALGO_GRAIN_NUTTING_C * a);
}


// ---------------------------------------------------------------------------
//  Noise-equivalent area of the resolution element, square micrometres.
//
//      A_elem = [ integral |T_scan(f) P_pix(f)|^2 df_x df_y ]^-1
//
//  ⚠ NOT THE PIXEL AREA, and using the pixel area is the obvious mistake. The
//  scanner's transfer is usually the wider of the two, so one output sample
//  averages over several pixels' worth of film and sees several times as many
//  grains as its own footprint holds -- at 4K, 88.1 um^2 against a 42.0 um^2
//  pixel. That factor of two lands directly on the gate.
//
//  The integral separates exactly: the isotropic Gaussian band limit factors
//  per axis and the pixel aperture is a box, so the plane integral is the
//  square of one one-dimensional integral.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainElementAreaUm2
(
    const HighPrecType scanSigmaMm,
    const HighPrecType pixelPitchMm
) noexcept
{
    if (pixelPitchMm <= 0.0)
        return 0.0;

    const HighPrecType step = ALGO_GRAIN_ELEM_FMAX
                            / static_cast<HighPrecType>(ALGO_GRAIN_ELEM_N - 1);

    const HighPrecType s = (scanSigmaMm > 0.0) ? scanSigmaMm : 0.0;

    HighPrecType acc  = 0.0;
    HighPrecType prev = 1.0;            // integrand is 1 at f = 0

    for (int32_t i = 1; i < ALGO_GRAIN_ELEM_N; i++)
    {
        const HighPrecType f = static_cast<HighPrecType>(i) * step;

        const HighPrecType t =
            std::exp(-2.0 * 9.8696044010893586188 * s * s * f * f);

        const HighPrecType x = 3.1415926535897932385 * f * pixelPitchMm;

        const HighPrecType p = (x > 1e-12) ? (std::sin(x) / x) : 1.0;

        const HighPrecType cur = (t * p) * (t * p);

        acc += 0.5 * (prev + cur) * step;

        prev = cur;
    }

    const HighPrecType i1 = 2.0 * acc;   // both signs of f

    if (i1 <= 0.0)
        return 0.0;

    return (1.0 / (i1 * i1)) * 1.0e6;    // mm^2 -> um^2
}


// ---------------------------------------------------------------------------
//  The count gate. 1 = pure Gaussian, 0 = fully non-Gaussian.
//
//  ⚠ SMOOTHSTEP IN LOG COUNT, NOT IN DENSITY. Skewness goes as 1/sqrt(N), so
//  equal RATIOS of N are equal steps of non-Gaussianity; a gate linear in N
//  would spend nearly all of its transition where nothing is changing.
//
//  ⚠ N <= 0 RETURNS 1 DELIBERATELY. That is unexposed film, where the only
//  grain is fog; the Gaussian branch already carries the fog floor through
//  s(D) and the correction has nothing to add.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainAlpha (const HighPrecType nElem) noexcept
{
    if (!(nElem > 0.0))
        return 1.0;

    const HighPrecType lo = std::log2(ALGO_GRAIN_N_MIN);
    const HighPrecType hi = std::log2(ALGO_GRAIN_N_HI);

    const HighPrecType u = (std::log2(nElem) - lo) / (hi - lo);

    if (u <= 0.0)
        return 0.0;

    if (u >= 1.0)
        return 1.0;

    return u * u * (3.0 - 2.0 * u);
}


// ---------------------------------------------------------------------------
//  Skewness of the compound-Poisson marginal at the element scale.
//
//  ⚠⚠ THE SPECIFICATION SAYS 1/sqrt(N_elem) AND THAT IS LOW BY
//  exp(6 sigma_ln^2) -- 45 % at sigma_ln 0.25, 152 % at 0.55. §12.4.3 assumes
//  every grain deposits the same density; they do not, the deposit goes as the
//  grain's AREA, and for a compound Poisson sum of variable jumps
//
//      gamma = E[X^3] / ( sqrt(lambda) E[X^2]^(3/2) ),   X proportional to r^2
//
//  which for a log-normal radius collapses to exp(6 sigma_ln^2) / sqrt(N).
//
//  ⚠ MEASURED, NOT ONLY DERIVED: the exact dot renderer gives 1.435 / 1.432 /
//  1.385 times the spec's figure on PORTRA 400 at three densities against a
//  predicted 1.448 for the aggregation used. The law holds to about 1 %.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainSkewness
(
    const HighPrecType nElem,
    const HighPrecType sigmaLn
) noexcept
{
    if (!(nElem > 0.0))
        return 0.0;

    const HighPrecType s = (sigmaLn > 0.0) ? sigmaLn : 0.0;

    return std::exp(6.0 * s * s) / std::sqrt(nElem);
}


// ---------------------------------------------------------------------------
//  The Cornish-Fisher coefficient, gated and clamped.
//
//      z' = ( z + c (z^2 - 1) ) / sqrt(1 + 2 c^2),    c = gamma_eff / 6
//
//  mean 0, variance 1 exactly, skewness gamma_eff to O(c^3).
//
//  ⚠⚠ THE CLAMP IS A MONOTONICITY REQUIREMENT AND NOT A TASTE LIMIT. The
//  derivative is 1 + 2 c z, positive only for z > -1/(2c). Above c = 0.1 that
//  boundary rises into the range a unit-variance field actually visits: at
//  c = 0.43, which the coarsest stock reaches at net D 0.02, it is z = -1.16
//  and roughly one sample in eight would be FOLDED BACK -- deep shadow values
//  turning brighter, which reads as posterisation in the toe rather than as
//  grain. 0.1 stays monotone to z = -5 and caps the imposed skewness at 0.6.
//
//  ⚠ WHAT THE CLAMP COSTS IS STATED, NOT HIDDEN: where theory asks for more
//  than 0.6 -- the extreme toe of the coarsest stocks, N_elem below about six,
//  2.0 % of the stock-density space -- the render is LESS skewed than real
//  film. That is also precisely where a marginal transform of a continuous
//  field stops being the right model, because real film there shows countable
//  discrete specks that no transform of a Gaussian can produce.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoGrainMarginalCoeff
(
    const HighPrecType nElem,
    const HighPrecType sigmaLn
) noexcept
{
    const HighPrecType a = AlgoGrainAlpha(nElem);

    if (a >= 1.0)
        return 0.0;

    const HighPrecType f = 1.0 - a;

    const HighPrecType g = AlgoGrainSkewness(nElem, sigmaLn)
                         * f * std::sqrt(f);

    const HighPrecType c = g / 6.0;

    return (c < ALGO_GRAIN_MARGINAL_COEFF_MAX)
               ? c : ALGO_GRAIN_MARGINAL_COEFF_MAX;
}


// ---------------------------------------------------------------------------
//  Apply the count-gated marginal correction to one finished grain field.
//
//  ⚠ ONE SCALAR IMPLEMENTATION, CALLED BY BOTH ENGINES, AND THAT IS
//  DELIBERATE. The arithmetic would vectorise, but the gate is open over a
//  minority of pixels and a second spelling of a per-pixel law is exactly the
//  shape of divergence this file has shipped twice. The AVX2 twin calls this.
//
//  pField     the unit-shaped field, modified in place
//  pDensity   the density plane the gate reads, same geometry
//  dmin       this channel's base plus fog, subtracted to get NET density
//  elemArea   noise-equivalent element area, square micrometres; <= 0 disables
// ---------------------------------------------------------------------------
inline void AlgoGrainApplyMarginal
(
    AlgoType* RESTRICT       pField,
    const AlgoType* RESTRICT pDensity,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const AlgoType           dmin,
    const HighPrecType       grainUm,
    const HighPrecType       sigmaLn,
    const HighPrecType       elemArea
) noexcept
{
    if ((elemArea <= 0.0) || (grainUm <= 0.0))
        return;

    // N_elem is LINEAR in net density, so the density above which the gate is
    // shut is one divide -- and then the common case is one compare per pixel
    // rather than a log and a square root.
    const HighPrecType perD =
        AlgoGrainCenterDensity(1.0, grainUm, sigmaLn) * elemArea;

    if (perD <= 0.0)
        return;

    const HighPrecType dShut = ALGO_GRAIN_N_HI / perD;

    // The field's own RMS, standardising it before the transform and restoring
    // it after. One reduction, in HighPrecType, in a fixed order.
    HighPrecType acc = 0.0;

    for (int32_t y = 0; y < sizeY; y++)
    {
        const AlgoType* RESTRICT pRow =
            pField + static_cast<std::ptrdiff_t>(y) * pitch;

        for (int32_t x = 0; x < sizeX; x++)
            acc += static_cast<HighPrecType>(pRow[x])
                 * static_cast<HighPrecType>(pRow[x]);
    }

    const HighPrecType n = static_cast<HighPrecType>(sizeX)
                         * static_cast<HighPrecType>(sizeY);

    if (n <= 0.0)
        return;

    const HighPrecType rms = std::sqrt(acc / n);

    if (!(rms > 0.0))
        return;

    const HighPrecType invRms = 1.0 / rms;

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        AlgoType* RESTRICT       pF = pField   + off;
        const AlgoType* RESTRICT pD = pDensity + off;

        for (int32_t x = 0; x < sizeX; x++)
        {
            const HighPrecType net =
                static_cast<HighPrecType>(pD[x]) - static_cast<HighPrecType>(dmin);

            if (!(net > 0.0) || (net >= dShut))
                continue;

            const HighPrecType c = AlgoGrainMarginalCoeff(net * perD, sigmaLn);

            if (c <= 0.0)
                continue;

            const HighPrecType z = static_cast<HighPrecType>(pF[x]) * invRms;

            const HighPrecType zp = (z + c * (z * z - 1.0))
                                  / std::sqrt(1.0 + 2.0 * c * c);

            pF[x] = static_cast<AlgoType>(zp * rms);
        }
    }

    return;
}


// ---------------------------------------------------------------------------
//  Build one zero-mean, spectrally shaped, granularity-calibrated grain field.
//
//  Exposed because three stages need it and must produce statistically identical
//  fields: the camera negative here at 11, each duplicating generation at 13, and
//  the print stock at 14. A second implementation would drift.
//
//  pDst          the finished field
//  pScrNoise     scratch: white noise, then the wide-lobe blur
//  pScrLobe      scratch: the narrow-lobe blur
//  pScrWork      scratch: separable blur workspace
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  clumpUm       mean developed clump diameter, micrometres
//  clumpGain     amplitude of the low-frequency clustering lobe
//  rmsGranularity  target RMS granularity, in the standard metric
//  scanSigmaPx   band limit from the scan optics, as a sigma in pixels
//  pxPerMm       render resolution
//  anisotropy    vertical/horizontal correlation ratio of the emulsion, from
//                GrainSpec::anisotropy. 1.0 is round grain; above 1.0 the grain
//                is stretched DOWN the frame, along the coating flow direction.
//                Floored at ALGO_GRAIN_ANISOTROPY_MIN inside, as the reference
//                floors it. Emulsions that carry no figure - dupe and print
//                stocks - pass ALGO_GRAIN_ANISOTROPY_NONE.
//  rngStage      which generator stream to draw from
//  seed          combined seed
//  frameIndex    clip-relative frame number
//
//  All four planes must be distinct from each other.
// ---------------------------------------------------------------------------
void AlgoMakeGrainField
(
    AlgoType* RESTRICT          pDst,
    AlgoType* RESTRICT          pScrNoise,
    AlgoType* RESTRICT          pScrLobe,
    AlgoType* RESTRICT          pScrWork,
    const int32_t               sizeX,
    const int32_t               sizeY,
    const int32_t               pitch,
    const AlgoType              clumpUm,
    const AlgoType              clumpGain,
    const AlgoType              rmsGranularity,
    const AlgoType              scanSigmaPx,
    const AlgoType              pxPerMm,
    const AlgoType              anisotropy,
    const eALGO_RNG_STAGE       rngStage,
    const uint32_t              seed,
    const int32_t               frameIndex
) noexcept;


// ---------------------------------------------------------------------------
//  AlgoGrainAmp -- the sigma(D) multiplier, hoisted out of the pixel loop.
//
//  ⚠ THIS EXISTS BECAUSE THE STAGE AND THE LAW HAD DRIFTED APART, AND THE
//  DRIFT SHIPPED. `film::FilmGrainSigma()` in the generated header is the one
//  definition and is audited against `film_profiles.grain_sigma()` on every
//  build -- but NOTHING IN THE RENDERER CALLED IT. AlgoAddGrain inlined its own
//  square root, without the net-1.0 normalisation, so every rendered frame was
//  louder than the reference by exactly sqrt(1 + fog_grain): measured
//  1.0392 to 1.1832 across the database, mean 1.1013, and 158 of 161 stocks
//  over 5 %. `rms_granularity` had stopped meaning the figure the datasheets
//  print. A law that is correct and unreachable is not a correct renderer.
//
//  WHY A PRECOMPUTED STRUCT RATHER THAN A PER-PIXEL CALL. FilmGrainSigma builds
//  and insertion-sorts up to four anchors and walks them twice, all of which
//  depends on the STOCK and the CHANNEL and none of which depends on the pixel.
//  Calling it per pixel would be correct and unusable. Everything invariant is
//  computed once per channel here, in HighPrecType, and the inner loop is left
//  with one square root and one multiply on the legacy branch, or at most three
//  compares and one fused multiply-add on the measured branch.
//
//  ⚠ THE BUILDER MIRRORS FilmGrainSigma EXACTLY, INCLUDING ITS USABILITY TEST.
//  It is not a second opinion about the law -- it is the same law with the
//  loop-invariant half lifted out. If FilmGrainSigma changes, this changes with
//  it in the same commit, and `cpp_parity.py`'s stage probe is what proves the
//  two still agree, because it drives AlgoAddGrain itself rather than the law.
// ---------------------------------------------------------------------------
struct AlgoGrainAmp
{
    bool     measured;      ///< false = legacy square-root branch
    int32_t  n;             ///< anchors in xs[], 3 or 4 (measured only)
    AlgoType xs[4];         ///< anchor densities, ascending
    AlgoType slope[4];      ///< segment i covers (xs[i-1], xs[i]]: slope*D+icept
    AlgoType icept[4];      ///< already divided by the net-1.0 reference
    AlgoType loY;           ///< held flat below xs[0]
    AlgoType hiY;           ///< held flat above xs[n-1]
    AlgoType dmin;          ///< legacy: this channel's base plus fog
    AlgoType fog;           ///< legacy: floor under the square root
    AlgoType ampScale;      ///< legacy: 1 / sqrt(1 + fog), the net-1.0 pin
};


// ---------------------------------------------------------------------------
//  Build the per-channel evaluator. Setup domain: runs three times per render,
//  never per pixel, so it computes in HighPrecType throughout.
// ---------------------------------------------------------------------------
inline AlgoGrainAmp AlgoGrainAmpBuild
(
    const film::GrainSpec& grain,
    const AlgoType         dminC,
    const AlgoType         dmaxC
) noexcept
{
    AlgoGrainAmp a;

    a.measured = false;
    a.n        = 0;
    a.loY      = ALGO_ONE;
    a.hiY      = ALGO_ONE;
    a.dmin     = dminC;

    for (int32_t i = 0; i < 4; i++)
    {
        a.xs[i]    = ALGO_ZERO;
        a.slope[i] = ALGO_ZERO;
        a.icept[i] = ALGO_ONE;
    }

    // Legacy half is always filled, so a measured branch that turns out to be
    // unusable falls through to a fully formed evaluator rather than to zeros.
    const HighPrecType fog =
        MAX_VALUE(static_cast<HighPrecType>(grain.fog_grain),
                  static_cast<HighPrecType>(0.0));

    const HighPrecType den = std::sqrt(static_cast<HighPrecType>(1.0) + fog);

    a.fog      = static_cast<AlgoType>(fog);
    a.ampScale = static_cast<AlgoType>(
        (den > static_cast<HighPrecType>(0.0))
            ? (static_cast<HighPrecType>(1.0) / den)
            : static_cast<HighPrecType>(1.0));

    if (!grain.sigma_shape_measured || !(grain.sigma_shape_mid > 0.0f))
        return a;

    const HighPrecType dToe = (grain.sigma_shape_toe_at > 0.0f)
        ? static_cast<HighPrecType>(grain.sigma_shape_toe_at)
        : static_cast<HighPrecType>(dminC);

    const HighPrecType dTop = (grain.sigma_shape_dmax_at > 0.0f)
        ? static_cast<HighPrecType>(grain.sigma_shape_dmax_at)
        : static_cast<HighPrecType>(dmaxC);

    // NET density 1.0. The stored anchors are ratios to the ABSOLUTE 1.0 value
    // because that is how they were traced, so the reference is recomputed here
    // rather than baked into the data -- exactly as FilmGrainSigma does it.
    const HighPrecType dRef = static_cast<HighPrecType>(dminC)
                            + static_cast<HighPrecType>(1.0);

    if (!(dTop > dToe) || !(dRef < dTop))
        return a;

    HighPrecType xs[4];
    HighPrecType ys[4];
    int32_t      n = 0;

    xs[n] = dToe;
    ys[n++] = static_cast<HighPrecType>(grain.sigma_shape_toe);
    xs[n] = static_cast<HighPrecType>(1.0);
    ys[n++] = static_cast<HighPrecType>(grain.sigma_shape_mid);
    xs[n] = dTop;
    ys[n++] = static_cast<HighPrecType>(grain.sigma_shape_dmax);

    if ((grain.sigma_shape_peak > 0.0f) && (grain.sigma_shape_peak_at > 0.0f))
    {
        xs[n] = static_cast<HighPrecType>(grain.sigma_shape_peak_at);
        ys[n++] = static_cast<HighPrecType>(grain.sigma_shape_peak);
    }

    for (int32_t i = 1; i < n; i++)
    {
        const HighPrecType kx = xs[i];
        const HighPrecType ky = ys[i];
        int32_t j = i - 1;
        while ((j >= 0) && (xs[j] > kx))
        {
            xs[j + 1] = xs[j];
            ys[j + 1] = ys[j];
            --j;
        }
        xs[j + 1] = kx;
        ys[j + 1] = ky;
    }

    // The net-1.0 reference value, read off the same piecewise-linear shape.
    HighPrecType mid = ys[n - 1];

    if (dRef <= xs[0])
    {
        mid = ys[0];
    }
    else
    {
        for (int32_t i = 1; i < n; i++)
        {
            if (dRef <= xs[i])
            {
                const HighPrecType t = (xs[i] > xs[i - 1])
                    ? ((dRef - xs[i - 1]) / (xs[i] - xs[i - 1]))
                    : static_cast<HighPrecType>(0.0);
                mid = ys[i - 1] + t * (ys[i] - ys[i - 1]);
                break;
            }
        }
    }

    const HighPrecType invMid = (mid > static_cast<HighPrecType>(0.0))
        ? (static_cast<HighPrecType>(1.0) / mid)
        : static_cast<HighPrecType>(1.0);

    // Fold the normalisation into the segment coefficients, so the pixel loop
    // never divides and never sees the reference at all.
    a.measured = true;
    a.n        = n;
    a.loY      = static_cast<AlgoType>(ys[0] * invMid);
    a.hiY      = static_cast<AlgoType>(ys[n - 1] * invMid);

    for (int32_t i = 0; i < n; i++)
        a.xs[i] = static_cast<AlgoType>(xs[i]);

    for (int32_t i = 1; i < n; i++)
    {
        const HighPrecType dx = xs[i] - xs[i - 1];
        const HighPrecType m  = (dx > static_cast<HighPrecType>(0.0))
            ? ((ys[i] - ys[i - 1]) / dx)
            : static_cast<HighPrecType>(0.0);

        a.slope[i] = static_cast<AlgoType>(m * invMid);
        a.icept[i] = static_cast<AlgoType>((ys[i - 1] - m * xs[i - 1]) * invMid);
    }

    return a;
}


// ---------------------------------------------------------------------------
//  The UNPINNED evaluator, for grain that is not anchored to a published rms.
//
//  ⚠ THE ASYMMETRY IS REAL AND IT IS NOT AN OVERSIGHT. Camera-negative grain is
//  pinned to the stock's own `rms_granularity`, a figure the manufacturer
//  publishes at a stated density -- "Read at a NET diffuse visual density of
//  1.0, using a 48-micrometre aperture" (Kodak 5248 p1, 5222 p1) -- so the
//  amplitude MUST be exactly 1.0 there or the stored number stops meaning what
//  the sheet says. PRINT and DUPLICATION grain have no such figure: the print
//  stock's grain_rms is the field's own amplitude and the weighting here is a
//  look, not a calibration. Normalising it would move every print render away
//  from the reference for no measurement's sake.
//
//  So stages 13 and 14 keep `sqrt(max(D - dmin, 0) + fog)` exactly as
//  film_sim.simulate() computes it, with no ampScale. Verified against the
//  reference: `out[:,:,c] += pfield * np.sqrt(max(out - dmin, 0) + 0.15)`.
// ---------------------------------------------------------------------------
inline AlgoGrainAmp AlgoGrainAmpRaw
(
    const AlgoType dminC,
    const AlgoType fogGrain
) noexcept
{
    AlgoGrainAmp a;

    a.measured = false;
    a.n        = 0;
    a.loY      = ALGO_ONE;
    a.hiY      = ALGO_ONE;
    a.dmin     = dminC;
    a.fog      = MAX_VALUE(fogGrain, ALGO_ZERO);
    a.ampScale = ALGO_ONE;          // deliberately unpinned -- see above

    for (int32_t i = 0; i < 4; i++)
    {
        a.xs[i]    = ALGO_ZERO;
        a.slope[i] = ALGO_ZERO;
        a.icept[i] = ALGO_ONE;
    }

    return a;
}


// ---------------------------------------------------------------------------
//  Evaluate the multiplier at one density. Scalar; the AVX2 twin open-codes the
//  same arithmetic across eight lanes from the same struct.
// ---------------------------------------------------------------------------
inline AlgoType AlgoGrainAmpAt
(
    const AlgoGrainAmp& a,
    const AlgoType      d
) noexcept
{
    if (!a.measured)
    {
        // Poisson statistics of a countable crystal population: the standard
        // deviation grows as the square root of the mean count, and developed
        // density stands in for that count. ampScale is what pins the result to
        // exactly 1.0 at NET density 1.0 -- dmin cancels there, which is why the
        // pin carries no per-channel term.
        const AlgoType developed = MAX_VALUE(d - a.dmin, ALGO_ZERO);
        return static_cast<AlgoType>(std::sqrt(
            static_cast<HighPrecType>(developed + a.fog))) * a.ampScale;
    }

    // Held flat outside the traced range rather than extrapolated. Extrapolating
    // a traced curve past its own endpoints is how a plausible number becomes a
    // fabricated one; holding it flat says "we stop knowing here", which is true.
    if (d <= a.xs[0])
        return a.loY;

    if (d >= a.xs[a.n - 1])
        return a.hiY;

    for (int32_t i = 1; i < a.n; i++)
    {
        if (d <= a.xs[i])
            return a.slope[i] * d + a.icept[i];
    }

    return a.hiY;
}


// ---------------------------------------------------------------------------
//  Add one grain field to three density planes, weighted by local density.
//
//  Exposed for the same reason: stages 13 and 14 add grain with the identical
//  weighting, and the weighting is where the physics lives.
//
//  pDstR/G/B     density planes, modified in place
//  pFieldR/G/B   the three fields; pass the same pointer three times for a stock
//                with a single emulsion
//  dmin          per-channel base plus fog of the curve that produced pDst
//  dmax          per-channel asymptotic maximum density of that same curve --
//                needed by the measured sigma(D) branch, which is anchored on
//                the traced range and falls back to the curve model only when
//                the trace did not record its own endpoints
//  grain         the stock's GrainSpec; the amplitude law reads fog_grain and
//                the five sigma_shape_* fields from it. ⚠ Passing the spec
//                rather than a loose fog value is deliberate: the previous
//                signature took fog_grain alone, which made it structurally
//                impossible for this stage to reach the measured shape and is
//                half of why the bypass lasted
//  gain          user grain scale
// ---------------------------------------------------------------------------
void AlgoAddGrain
(
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const AlgoType* RESTRICT pFieldR,
    const AlgoType* RESTRICT pFieldG,
    const AlgoType* RESTRICT pFieldB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const AlgoType           dmin[3],
    const AlgoType           dmax[3],
    const film::GrainSpec&   grain,
    const AlgoType           gain
) noexcept;


// ---------------------------------------------------------------------------
//  The UNPINNED overload, for print (stage 14) and duplication (stage 13).
//
//  Same loop, same weighting, no net-1.0 pin -- see AlgoGrainAmpRaw for why the
//  two differ and why that difference is the model rather than an omission.
//  ⚠ Do NOT "unify" these by giving this one an ampScale: it would silently
//  move every print and dupe render away from film_sim.simulate().
// ---------------------------------------------------------------------------
void AlgoAddGrainRaw
(
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const AlgoType* RESTRICT pFieldR,
    const AlgoType* RESTRICT pFieldG,
    const AlgoType* RESTRICT pFieldB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const AlgoType           dmin[3],
    const AlgoType           fogGrain,
    const AlgoType           gain
) noexcept;


// ---------------------------------------------------------------------------
//  Stage 11: grain.
//
//  pSrcR/G/B     density in
//  pDstR/G/B     density out, floored at zero
//  pScrNoise     scratch  (see AlgoMakeGrainField)
//  pScrLobe      scratch
//  pScrWork      scratch
//  pScrFieldR/G/B  scratch: the three finished grain fields
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  params        user controls; grainScale scales the amplitude
//  scanSigmaPx   band limit from stage 10, as a sigma in pixels
//  pxPerMm       render resolution
//  hasMosaic     true when stage 7 actually built a reseau record, which means one
//                emulsion and therefore one shared field. NOT the same as
//                profile.has_reseau: the mosaic is skipped when the grid cannot be
//                resolved at this render size, and the grain must follow suit.
//  frameIndex    clip-relative frame number
//  seed          per-call seed
//
//  All six scratch planes must be distinct from each other, from the source and
//  from the destination.
// ---------------------------------------------------------------------------
void AlgoStage11_Grain
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrNoise,
    AlgoType* RESTRICT       pScrLobe,
    AlgoType* RESTRICT       pScrWork,
    AlgoType* RESTRICT       pScrFieldR,
    AlgoType* RESTRICT       pScrFieldG,
    AlgoType* RESTRICT       pScrFieldB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           scanSigmaPx,
    const AlgoType           pxPerMm,
    const bool               hasMosaic,
    const int32_t            frameIndex,
    const uint32_t           seed
) noexcept;
