#pragma once

// ---------------------------------------------------------------------------
//  AlgoDuplication.hpp
//
//  Stage 13 of the film simulation pipeline: duplication generations, then the
//  release print.
//
//  NOBODY EVER PROJECTED A CAMERA NEGATIVE
//
//  That sentence is the reason this stage exists. A release print is three or four
//  generations away from the original:
//
//      camera negative -> interpositive -> dupe negative -> release print
//
//  Every intermediate is a real emulsion. Each one adds its OWN grain and its OWN
//  MTF loss, and that accumulation is a large part of why archival footage looks
//  the way it does - considerably more than the camera emulsion alone. A model that
//  goes straight from negative to print produces something sharper and cleaner
//  than any print anyone has ever seen.
//
//  WHY GENERATIONS COME IN PAIRS
//
//  Each printing step inverts polarity. Stages are therefore counted in PAIRS, so
//  the polarity always returns to negative before the final print. Duplicating
//  stock is manufactured to run at gamma 1.0 precisely so that contrast does NOT
//  compound over the chain - grain and softness do, contrast does not. Getting that
//  wrong turns a four-generation chain into a contrast disaster instead of a soft,
//  grainy, faithful one.
//
//  THE ORDER OF BLUR AND GRAIN WITHIN A GENERATION IS NOT ARBITRARY
//
//  Printing optics blur what comes IN - the accumulated image and all the grain
//  from every earlier generation. Then the new emulsion records it. Then THIS
//  stage's own grain is created, in this emulsion, and so is NOT blurred by this
//  stage's optics; only by later ones.
//
//  Adding grain before the blur - which is the obvious way round, and wrong -
//  quietly softens every generation's own grain by its own MTF, and makes a long
//  dupe chain come out CLEANER than a short one. That is backwards, and it is a
//  mistake that is invisible unless you compare two generation counts.
//
//  THE PRINT
//
//      logE_print = offset - D_negative
//
//  Higher scene exposure raises negative density, which lowers print exposure,
//  which lowers print density, which brightens the positive. That double inversion
//  is what gives correct rolloff at BOTH ends for free: the negative's shoulder
//  becomes the print's shadow rolloff and the negative's toe becomes the print's
//  highlight rolloff. No single curve can produce both.
//
//  The offset is the printer-light setting, solved so that a neutral 18 per cent
//  grey lands on the requested display value. It has to be re-solved here rather
//  than reused from stage 8, because a dupe chain moves the neutral density.
//
//  A REVERSAL STOCK SKIPS ALL OF THIS
//
//  A slide already IS the positive. Its own Dmin and Dmax are the white and black
//  points. No second curve, no inversion, no dupe chain - which is also why a
//  reversal stock's anchors were log-exposure trims consumed back at stage 8.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// The separable Gaussian used for each generation's printing optics.
#include "AlgoSeparableBlur.hpp"

// Curve evaluation and the three offset solvers.
#include "AlgoCharacteristicCurve.hpp"

// Grain field construction, shared with stages 11 and 14.
#include "AlgoGrain.hpp"

// The density matrix helpers, shared with stage 12.
#include "AlgoDyeImpurity.hpp"

// The scan MTF sigma conversion, shared with stage 10.
#include "AlgoScanMtf.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including PrintStock.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Hard cap on duplication generations.
//
//  Four pairs is already beyond any real release chain, and each pair is a full
//  blur plus a full curve plus a full grain field on every pixel. The cap exists so
//  a mistyped control cannot turn one frame into a minute.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_DUPE_MAX_GENERATIONS = 4;


// ---------------------------------------------------------------------------
//  Stage 13: duplication generations, then the release print.
//
//  pSrcR/G/B       density in, from the camera negative
//  pDstR/G/B       density out; the print for a negative, the slide itself for a
//                  reversal stock
//  pScrTmpR/G/B    scratch: one full triple, holding each generation's blurred input
//  pScrNoise       scratch: grain construction
//  pScrLobe        scratch: grain construction
//  pScrWork        scratch: separable blur workspace
//  pScrField       scratch: the finished grain field for one generation
//  sizeX/sizeY     active pixel extent
//  pitch           row stride in ELEMENTS
//  profile         stock being simulated
//  params          user controls; generations, grainScale, greyTarget, couplerScale
//  pPrintStock     the release print stock; may be null, in which case the negative
//                  density is passed through unchanged and the caller gets the
//                  negative's own curves back
//  pDupeStock      the duplicating stock; may be null, which disables the dupe
//                  chain regardless of the generation count
//  scanSigmaPx     band limit for every grain field, from stage 10
//  pxPerMm         render resolution
//  frameIndex      clip-relative frame number
//  seed            per-call seed
//  finalCurvesOut  RECEIVES the curve set that produced the output. Stage 14 needs
//                  it for the transmittance conversion and for print grain, and it
//                  differs by path: the print stock's curves for a negative, the
//                  stock's own for a reversal.
//
//  All seven scratch planes must be distinct from each other, from the source and
//  from the destination.
// ---------------------------------------------------------------------------
void AlgoStage13_Duplication
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrTmpR,
    AlgoType* RESTRICT       pScrTmpG,
    AlgoType* RESTRICT       pScrTmpB,
    AlgoType* RESTRICT       pScrNoise,
    AlgoType* RESTRICT       pScrLobe,
    AlgoType* RESTRICT       pScrWork,
    AlgoType* RESTRICT       pScrField,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const film::PrintStock*  pPrintStock,
    const film::PrintStock*  pDupeStock,
    const AlgoType           scanSigmaPx,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const uint32_t           seed,
    film::RGBCurves&         finalCurvesOut
) noexcept;
