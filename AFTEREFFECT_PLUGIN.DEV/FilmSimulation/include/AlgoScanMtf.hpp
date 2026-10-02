#pragma once

// ---------------------------------------------------------------------------
//  AlgoScanMtf.hpp
//
//  Stage 10 of the film simulation pipeline: the scan - optical MTF of the
//  scanner or printer plus per-channel registration error.
//
//  WHY THE SCAN COMES BEFORE GRAIN AND NOT AFTER
//
//  This is the single most important ordering decision in the second half of the
//  chain, and it looks wrong at first glance.
//
//  The scanner's optical MTF is the PRE-SAMPLING filter. Light from the film
//  passes the lens before it reaches the sensor, so the lens band-limits both the
//  image and the grain before either is sampled. Grain is generated at stage 11
//  already band-limited by this same transfer, which is the only way to stop fine
//  grain aliasing onto the pixel grid: a stock whose grain is finer than a pixel
//  must render smooth, not speckled, and it will speckle if the grain is created
//  at full bandwidth and only filtered afterwards.
//
//  So the scan MTF is applied to the image here, and handed to the grain stage as
//  its band limit. Both uses are the same physical filter.
//
//  REGISTRATION ERROR
//
//  The three colour records do not land in perfect register. On a tripack the
//  cause is the scanner's own chromatic aberration and sensor alignment; on a
//  three-strip process the three records are three separate pieces of film that
//  have to be aligned mechanically, and the error is an order of magnitude larger.
//
//  The displacement is specified on the NEGATIVE in micrometres, so it scales with
//  resolution like every other spatial quantity in the engine. A few micrometres
//  is invisible as a shift and very visible as an ABSENCE: it softens colour edges
//  in exactly the way that every real film scan is softened, and rendering without
//  it leaves colour edges cleaner than any scan has ever produced. Three-strip
//  Technicolor used tens of micrometres, which is why its edges visibly fringe.
//
//  A NOTE ON THE SHIFT METHOD
//
//  The frequency-domain reference applies the displacement as a phase ramp, which
//  is an exact band-limited translation. This implementation uses bilinear
//  interpolation with a wrap boundary, which is not identical: it adds a small
//  amount of its own blur, growing with the fractional part of the shift and
//  vanishing at whole pixels. For displacements of a fraction of a pixel - which
//  is what a few micrometres is at any sane resolution - that extra blur is far
//  below the softening the shift itself is there to produce.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// The separable Gaussian that carries out the MTF, and the copy helpers.
#include "AlgoSeparableBlur.hpp"

// Counter-based generator for the registration jitter.
#include "AlgoCounterRng.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, and PrintStock which carries the fallback scan MTF figure.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Smallest MTF sigma, in pixels, worth submitting to a blur.
//
//  Below a quarter of a pixel the discrete kernel has one significant tap.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_SCAN_MIN_SIGMA_PX = static_cast<AlgoType>(0.25);

// ---------------------------------------------------------------------------
//  Smallest registration displacement, in pixels, worth resampling for.
//
//  A hundredth of a pixel. Below that the bilinear weights round to the identity
//  and the resample costs a full pass for nothing.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_SCAN_MIN_SHIFT_PX = static_cast<AlgoType>(0.01);


// ---------------------------------------------------------------------------
//  Gaussian sigma in millimetres for a given 50 per cent modulation frequency.
//
//  The same relation the emulsion MTF uses:  sigma_mm = K / f50  with
//  K = sqrt(ln2 / 2) / pi. Exposed because the grain stage needs the identical
//  band limit, and two copies of the conversion would be one more chance for the
//  image and its grain to be filtered differently.
//
//  Returns zero when f50 is zero or negative, meaning "no figure available",
//  which is treated as perfectly sharp rather than as infinitely soft.
// ---------------------------------------------------------------------------
AlgoType AlgoScanSigmaMm (const AlgoType f50CyclesPerMm) noexcept;


// ---------------------------------------------------------------------------
//  Stage 10: scan MTF plus per-channel registration error.
//
//  pSrcR/G/B     density in
//  pDstR/G/B     density out, floored at zero
//  pScrA         scratch: blur intermediate, and shift source
//  pScrB         scratch: shift destination
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated; misregistration_um lives here
//  params        user controls; misregScale scales the displacement
//  scanF50       50 per cent modulation frequency of the scan, cycles/mm
//  pxPerMm       render resolution
//  frameIndex    clip-relative frame number; the jitter is drawn per frame
//  seed          per-call seed, combined with params.seed by the generator
//
//  The two scratch planes must be distinct from each other, from the source and
//  from the destination.
// ---------------------------------------------------------------------------
void AlgoStage10_ScanMtf
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrA,
    AlgoType* RESTRICT       pScrB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           scanF50,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const uint32_t           seed
) noexcept;
