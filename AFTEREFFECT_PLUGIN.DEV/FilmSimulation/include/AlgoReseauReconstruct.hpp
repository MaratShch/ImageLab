#pragma once

// ---------------------------------------------------------------------------
//  AlgoReseauReconstruct.hpp
//
//  Sub-stage 14b of the film simulation pipeline: reseau reconstruction, plus the
//  residual base tint that follows it.
//
//  HOW ADDITIVE COLOUR IS VIEWED
//
//  An additive colour stock - Dufaycolor here - holds a single monochrome record
//  behind a fixed mosaic of colour filters. Projection sends light back through
//  THE SAME grid, in register, so each cell contributes only its own colour and the
//  eye integrates the mosaic into colour.
//
//  WHY THIS RUNS AT THE VERY END, WHICH LOOKS LIKE A SHORTCUT AND IS NOT
//
//  On a real additive print the grid physically sits in the light path AT VIEWING
//  TIME, downstream of the emulsion, downstream of any printing, downstream of
//  everything. Reconstructing earlier would model a print that had somehow been
//  demosaiced before it was projected, which is not a thing that exists.
//
//  THE RECONSTRUCTION ITSELF
//
//  For each channel: blur the record masked to that channel's cells, blur the mask
//  alone, and divide. The second blur is the coverage normalisation - without it,
//  a channel whose cells occupy a third of the area would come out at a third
//  brightness.
//
//  THE BLUR RADIUS IS DELIBERATELY SMALL
//
//  Comparable to the grid pitch, not much larger. That is what leaves the faint
//  grid texture visible and caps the COLOUR resolution well below the LUMINANCE
//  resolution - both of which are real, measurable and characteristic of the
//  process. A large radius would give clean colour and throw away the thing that
//  makes Dufaycolor recognisable as Dufaycolor.
//
//  RESIDUAL BASE TINT
//
//  Applied here, at the end, and only partly. A real printer neutralises the film
//  base colour - that is what printer lights are for - so only a small residual
//  survives into the print. The anchor solves at stages 8 and 13 already aimed at
//  tint-adjusted targets, so this is the other half of the same split, and the two
//  fractions must agree.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// The separable Gaussian used for both the masked record and the mask.
#include "AlgoSeparableBlur.hpp"

// The mask geometry, so this stage reproduces exactly the grid stage 7 used.
#include "AlgoEmulsionRecord.hpp"

// AlgoTintFactor: the residual base tint applied here is the other half of the
// split the anchor solves already applied, so both must read the same constant.
#include "AlgoCharacteristicCurve.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including ReseauSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Floor on the blurred mask before dividing by it.
//
//  The coverage normalisation divides by a blurred one-hot mask. Where the blur
//  radius is small relative to the cell spacing that value can approach zero
//  between cells of a given colour, and the quotient would explode. The floor is
//  small enough not to affect the normalisation where coverage is real.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_RESEAU_COVERAGE_FLOOR = static_cast<AlgoType>(1.0e-4);


// ---------------------------------------------------------------------------
//  Sub-stage 14b: reseau reconstruction, then residual base tint.
//
//  pSrcR/G/B     display-linear transmittance in; for a mosaic stock all three
//                planes carry the same single record
//  pDstR/G/B     transmittance out
//  pScrMasked    scratch: the record masked to one channel's cells
//  pScrMask      scratch: that channel's mask alone
//  pScrNum       scratch: the blurred masked record
//  pScrDen       scratch: the blurred mask
//  pScrWork      scratch: separable blur workspace
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  params        user controls; reseau must match what stage 7 was given
//  pxPerMm       render resolution, so the grid matches stage 7's exactly
//
//  All five scratch planes must be distinct from each other, from the source and
//  from the destination.
// ---------------------------------------------------------------------------
void AlgoStage14b_ReseauReconstruct
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrMasked,
    AlgoType* RESTRICT       pScrMask,
    AlgoType* RESTRICT       pScrNum,
    AlgoType* RESTRICT       pScrDen,
    AlgoType* RESTRICT       pScrWork,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           pxPerMm
) noexcept;
