#pragma once

// ---------------------------------------------------------------------------
//  AlgoBromideDrag.hpp
//
//  Stage 9c of the film simulation pipeline: bromide drag, the processing
//  machine's directional restraint of development.
//
//  PHYSICAL BACKGROUND
//
//  Reducing silver halide releases bromide ion into the developer, and bromide
//  is a restrainer. Where a lot of silver develops, the thin layer of solution
//  lying against the emulsion becomes locally loaded with it and locally
//  exhausted of developing agent. In a processing machine the film MOVES through
//  the bath, so that loaded layer is dragged along the transport axis and keeps
//  restraining development where it lands - BEHIND the dense area, and only
//  behind it. What comes out is a one-sided streak, aligned with the web,
//  trailing every heavily developed region. It is the archival lab-print
//  signature, and it is the one large-scale development effect this pipeline
//  had no representation for.
//
//  WHY IT IS NOT STAGE 9, AND WHY FOLDING THEM WOULD BREAK BOTH
//
//  Three effects share one chemistry and share nothing else:
//
//    stage 8b   inhibitor crossing BETWEEN layers          vertical, microns
//    stage 9    inhibitor diffusing WITHIN a layer         isotropic, tens of um
//    stage 9c   loaded developer dragged ACROSS the film   one-sided, mm to cm
//
//  Stage 9's inhibitor is inside the gelatin and goes equally in all directions.
//  This is outside it, in the bath, and goes one way only. Three scales, three
//  symmetries, and - decisively - two different owners: stages 8b and 9 are
//  properties of the COATING and live on FilmProfile, while this one is a
//  property of the MACHINE and lives on ProcessingSpec. The same emulsion in two
//  labs shows two different amounts of it and in a well-agitated tank shows
//  none.
//
//  WHY IT IS AFTER THE CURVE
//
//  Same argument stage 9 makes: the bromide is released BY development, in
//  proportion to the silver reduced, so its amount is a function of DENSITY and
//  not of exposure. The density stage 9 leaves is exactly what releases it,
//  which is why the two stages are adjacent in the chain.
//
//  ⚠ THE SIGN INVERTS ON A REVERSAL FILM
//
//  The bromide comes from the silver the FIRST developer reduces. On a negative
//  that is the image the viewer sees, so streaks trail the DENSE areas. On a
//  reversal stock the first developer makes the NEGATIVE image, complementary to
//  the slide, so streaks trail the CLEAR areas. This is the easiest thing in the
//  stage to get backwards; the source field is inverted on isReversal() and
//  nothing else in the stage depends on the stock kind.
//
//  ⚠ INERT ON EVERY STOCK IN THIS DATABASE, AND THAT IS A STATEMENT ABOUT THE
//  LITERATURE. Queue row C23 asked for a lab model and a reference frame. This
//  is the model. No document in the corpus quantifies a bromide gradient - the
//  nearest, Hariharan PS&E 2(2) p77 (1958), measures the KOSTINSKY effect, which
//  is the short-range two-sided cousin - so every BromideDragSpec ships at zero
//  and the stage returns on its first branch. The day a reference frame arrives,
//  the work is fitting two numbers and not building a pipeline.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// Stock parameters, including ProcessingSpec::bromide_drag.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Hard ceiling on the fraction of net density stage 9c may remove.
//
//  ⚠ A NUMERICAL FLOOR, NOT A SECOND STRENGTH KNOB. Without it a saturated
//  streak could drive net density to exactly zero and, in float, slightly
//  through it - and this stage sits UPSTREAM of the pipeline's single clamp at
//  stage 17, so a negative net density would reach the scan MTF at stage 10 and
//  be spread around by it before anything caught it. The generator already
//  refuses a strength above 0.5, so on any legal record this never binds.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_BROMIDE_MAX_REMOVED = static_cast<AlgoType>(0.95);


// ---------------------------------------------------------------------------
//  Stage 9c: bromide drag.
//
//  pR/pG/pB    ABSOLUTE density, read and written IN PLACE
//  pScrSrc     scratch: the normalised byproduct source field
//  pScrAcc     scratch: the accumulated upstream restraint
//  sizeX/sizeY active pixel extent
//  pitch       row stride in ELEMENTS
//  profile     stock being simulated; supplies the record, the dmin/dmax
//              ladder and the reversal flag
//  pxPerMm     render resolution, used to turn the millimetre drag length into
//              a per-pixel retention coefficient
//
//  ⚠ IN PLACE, LIKE STAGE 12b AND UNLIKE EVERY NUMBERED STAGE. It is a
//  multiply by a scalar field, so a separate destination plane set would cost a
//  full extra image of bandwidth to express a copy. The two scratch planes must
//  be distinct from each other and from the three data planes; Scr_Dbar and
//  Scr_DbarBlur are both dead by this point in AlgorithmMain and are what the
//  engine passes.
//
//  Returns true if the stage did any work, false if the record is inert. The
//  caller uses the return only for its profiling mark.
// ---------------------------------------------------------------------------
bool AlgoStage09c_BromideDrag
(
    AlgoType* RESTRICT       pR,
    AlgoType* RESTRICT       pG,
    AlgoType* RESTRICT       pB,
    AlgoType* RESTRICT       pScrSrc,
    AlgoType* RESTRICT       pScrAcc,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoType           pxPerMm
) noexcept;
