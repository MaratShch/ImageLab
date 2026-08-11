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
//  Add one grain field to three density planes, weighted by local density.
//
//  Exposed for the same reason: stages 13 and 14 add grain with the identical
//  square-root weighting, and the weighting is where the physics lives.
//
//  pDstR/G/B     density planes, modified in place
//  pFieldR/G/B   the three fields; pass the same pointer three times for a stock
//                with a single emulsion
//  dmin          per-channel base plus fog of the curve that produced pDst
//  fogGrain      density floor under the square root
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
