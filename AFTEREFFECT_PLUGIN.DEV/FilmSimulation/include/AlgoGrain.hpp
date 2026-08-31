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
