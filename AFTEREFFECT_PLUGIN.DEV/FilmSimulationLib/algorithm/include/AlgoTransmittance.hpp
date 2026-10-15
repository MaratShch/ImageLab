#pragma once

// ---------------------------------------------------------------------------
//  AlgoTransmittance.hpp
//
//  Stage 14 of the film simulation pipeline: print grain, then density to
//  display-linear transmittance.
//
//  PRINT GRAIN, AND WHY IT IS NOT JUST MORE GRAIN
//
//  Print stock has its own grain: finer than negative grain and largely
//  achromatic, so one field serves all three channels.
//
//  What makes it worth modelling separately is WHERE it lands. Print grain is
//  created after the print curve, so unlike negative grain it is NOT compressed by
//  the print stock's shoulder. Negative grain passes through the print curve and
//  gets squeezed in the highlights; print grain does not. The result is a subtle
//  but real difference in how grain behaves at the bright end - and it is a
//  difference a single-stage grain model cannot produce at all, whatever its
//  spectrum.
//
//  DENSITY TO TRANSMITTANCE
//
//  By the definition of optical density, transmittance is ten to the minus density.
//  The stage then normalises against the FINAL CURVE'S OWN endpoints:
//
//      t_max = 10^-Dmin   clear film, the brightest the stock can be
//      t_min = 10^-Dmax   the darkest
//      out   = (10^-D - t_min) / (t_max - t_min)
//
//  Normalising against the stock's own range rather than against absolute
//  transmittance is what makes the output display-referred without a separate
//  grade: a print whose base is denser is not rendered darker overall, because its
//  own clear film is still white. It is also what the anchor solves at stages 8 and
//  13 aim at, so the two must use the same expression or a neutral will not land
//  where it was solved to land.
//
//  WHICH CURVES ARE THE FINAL CURVES
//
//  Not the film's. For a negative the output came off the PRINT stock, so the print
//  stock's endpoints are the right ones; for a reversal stock the slide is the
//  output and its own endpoints apply. Stage 13 reports which set it used, and this
//  stage must be given that set rather than deriving it.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// Grain field construction, shared with stages 11 and 13.
#include "AlgoGrain.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including PrintStock.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Stage 14: print grain, then transmittance.
//
//  pSrcR/G/B     density in
//  pDstR/G/B     display-linear transmittance out, floored at zero and NOT capped
//                at one - the single final clamp belongs to stage 17
//  pScrNoise     scratch: grain construction
//  pScrLobe      scratch: grain construction
//  pScrWork      scratch: separable blur workspace
//  pScrField     scratch: the finished print grain field
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  params        user controls; printGrain and grainScale
//  pPrintStock   print stock, for its grain figures; may be null
//  finalCurves   the curve set stage 13 reported, whose endpoints normalise the
//                transmittance
//  isReversal    true for a slide, which has no print stage and therefore no print
//                grain
//  scanSigmaPx   band limit for the grain field, from stage 10
//  pxPerMm       render resolution
//  frameIndex    clip-relative frame number
//  seed          per-call seed
//
//  All four scratch planes must be distinct from each other, from the source and
//  from the destination.
// ---------------------------------------------------------------------------
void AlgoStage14_Transmittance
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
    AlgoType* RESTRICT       pScrField,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const film::PrintStock*  pPrintStock,
    const film::RGBCurves&   finalCurves,
    const bool               isReversal,
    const AlgoType           scanSigmaPx,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const uint32_t           seed
) noexcept;
