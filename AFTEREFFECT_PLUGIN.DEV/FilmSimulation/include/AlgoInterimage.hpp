#pragma once

// ---------------------------------------------------------------------------
//  AlgoInterimage.hpp
//
//  Sub-stage 8b of the film simulation pipeline: interimage effects, the
//  cross-layer half of the DIR coupler chemistry.
//
//  PHYSICAL BACKGROUND
//
//  A colour developer does not develop the three layers in isolation. As silver
//  is developed in one layer the coupler releases a development inhibitor, and
//  that inhibitor diffuses VERTICALLY into the neighbouring layers and suppresses
//  development there. So each layer's effective exposure depends on what the
//  other two are doing:
//
//      logE_i' = logE_i + sum over j != i of a_ij * (D_j - D_ref_j)
//
//  Stage 9 is the LATERAL half of the same chemistry - inhibitor diffusing
//  sideways within a layer. Same molecules, two different directions, two
//  different visual consequences, so they are separate stages.
//
//  WHY THE MID-GREY REFERENCE MATTERS
//
//  Subtracting the density a neutral mid grey reaches is what makes this a
//  COLOUR effect rather than a tone effect. On a neutral every (D_j - D_ref_j) is
//  about zero, the correction vanishes, and the grey scale passes through
//  untouched. On a saturated colour, where the three layers disagree strongly,
//  each develops against unequal inhibition and they separate further.
//
//  That is saturation rising WITHOUT gamma rising, which no per-channel tone
//  curve can produce however it is shaped. It is the single mechanism that
//  distinguishes a modern coupler-rich emulsion from a 1950s one, and it is the
//  reason a well-fitted set of curves alone still renders a modern stock flat.
//
//  IT IS AN IMPLICIT EQUATION
//
//  Density depends on the corrected log exposure, which depends on density. The
//  system is solved by fixed-point iteration seeded with the densities stage 8
//  just computed. Each pass costs a full curve evaluation per channel, which
//  makes this among the most expensive stages in the chain, so the iteration
//  count is a profile field the renderer honours rather than a hardcoded loop:
//  a stock whose coefficients are small converges in one pass and should not pay
//  for four.
//
//  DENSITY WEIGHTING, AND WHY IT EXISTS
//
//  Zero means uniform coupling across the whole curve, which is right for a
//  chromogenic negative: the inhibitor is released in proportion to development
//  everywhere.
//
//  Greater than zero concentrates the coupling where the NEIGHBOURING layer is
//  dense, which is right for a reversal stock: its interimage effects come from
//  iodide released in the first black-and-white developer, and that lands in the
//  areas that end up with high dye density.
//
//  The weighting is normalised at the mid-grey reference, so a neutral is
//  untouched under either mechanism. That property is the entire point of the
//  stage and must survive the mechanism split - a weighting scheme that tinted
//  neutrals would be worse than no weighting at all.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// The characteristic curve this stage re-evaluates once per iteration.
#include "AlgoCharacteristicCurve.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including InterimageSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Floor on the mid-grey reference density used to normalise the weighting.
//
//  A stock whose base fog is essentially zero would otherwise divide by zero
//  here. The floor is far below any real base plus fog, so it changes nothing on
//  a real profile and only removes the singularity.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_IIE_REF_FLOOR = static_cast<AlgoType>(1.0e-4);


// ---------------------------------------------------------------------------
//  Sub-stage 8b: interimage effects.
//
//  pSrcR/G/B     density from stage 8
//  pDstR/G/B     corrected density out
//  pLogER/G/B    log exposure retained by stage 8; READ ONLY, and required -
//                the correction re-enters the curve in the exposure domain, and
//                density cannot be inverted back through the shoulder
//  pScrDR/DG/DB  scratch: the three (D_j - D_ref_j) difference planes
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  anchor        anchors from AlgoSolveAnchors; the reversal trim enters the
//                curve argument exactly as it did at stage 8
//
//  The three scratch planes must be distinct from each other, from the source,
//  from the destination and from the log-exposure planes.
// ---------------------------------------------------------------------------
void AlgoStage08b_Interimage
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const AlgoType* RESTRICT pLogER,
    const AlgoType* RESTRICT pLogEG,
    const AlgoType* RESTRICT pLogEB,
    AlgoType* RESTRICT       pScrDR,
    AlgoType* RESTRICT       pScrDG,
    AlgoType* RESTRICT       pScrDB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const HighPrecType       anchor[3]
) noexcept;
