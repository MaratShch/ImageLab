#pragma once

// ---------------------------------------------------------------------------
//  AlgoEdgeFog.hpp
//
//  Sub-stage 10b of the film simulation pipeline: narrow-gauge edge fog.
//
//  PHYSICAL BACKGROUND
//
//  Additive density rising towards the left and right edges of the frame, from
//  two causes that both land in the same place: light leaking past the edge of the
//  film roll, and development edge effects where fresh developer reaches the
//  margin of the strip more freely than its middle.
//
//  IT IS A GAUGE MATTER, NOT AN ERA MATTER
//
//  This is the point that makes the stage worth having. Standard 8 is 16 mm film
//  exposed down one half, turned round, exposed down the other, and then SLIT DOWN
//  THE MIDDLE AFTER PROCESSING. Its picture area therefore sits right at the film
//  edge with no trimmed margin between them, so both effects land inside the
//  image.
//
//  On 35 mm the margins carry the perforations and are trimmed away, so the
//  picture never sees them and the spec leaves the figure at zero. The same
//  emulsion, coated on the same day, shows edge fog on one gauge and none on the
//  other - which is why this belongs to the format and the coating spec rather
//  than to the era.
//
//  WHY IT IS APPLIED IN THE DENSITY DOMAIN
//
//  Both contributors end up as developed silver or dye, and the spec's units are
//  density. Applying it to exposure instead would let the characteristic curve
//  reshape it, so the same physical fog would show up differently depending on
//  where in the frame it fell relative to the toe - which is not how a fogged edge
//  behaves.
//
//  It is applied after development and before printing, so a dupe chain and the
//  print curve act on it exactly as they act on the picture.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including CoatingSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Sub-stage 10b: narrow-gauge edge fog.
//
//  pSrcR/G/B     density in
//  pDstR/G/B     density out
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated; CoatingSpec carries the fog figures
//  params        user controls; coatingScale scales the fog
//  negWidthMm    frame width on the film - the fog profile is measured in
//                millimetres inward from the physical edge, so this is required
//                and a zero disables the stage
//
//  No scratch planes: the fog is a function of the horizontal coordinate alone.
// ---------------------------------------------------------------------------
void AlgoStage10b_EdgeFog
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           negWidthMm
) noexcept;
