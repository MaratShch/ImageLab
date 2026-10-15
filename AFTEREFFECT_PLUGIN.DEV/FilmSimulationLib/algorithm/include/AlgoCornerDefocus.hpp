#pragma once

// ---------------------------------------------------------------------------
//  AlgoCornerDefocus.hpp
//
//  Sub-stage 6b of the film simulation pipeline: corner defocus from film
//  buckling in the camera gate.
//
//  PHYSICAL BACKGROUND
//
//  Film is a curling plastic ribbon held flat only where something holds it. A
//  pressure plate presses the middle of the frame against the aperture plate, so
//  the centre sits in the focal plane. The corners of a curling base lift out of
//  it, and are therefore imaged on a surface that is no longer where the lens
//  put its image. The result is corner SOFTNESS.
//
//  THE CONFUSION THIS EXISTS TO AVOID
//
//  Corner softness and corner darkening are conflated constantly and they are
//  entirely different mechanisms. Darkening is cos^4 vignetting, a property of
//  the lens and of flat-field geometry, and it lives in stage 4. Softness is the
//  film physically not being where it should be, and it lives here. A stock on a
//  thick or badly curling base can be soft in the corners with no measurable
//  falloff, and a fast lens wide open can fall off badly with perfectly sharp
//  corners.
//
//  WHY THIS IS A SEPARATE PASS
//
//  A blur that varies across the frame is not one transfer function, so it
//  cannot be folded into the emulsion MTF stage: the frequency-domain reference
//  needs a whole second transform per channel to express it, which at HD costs
//  about as much as the entire MTF stage.
//
//  WHY A FIXED FIVE-TAP KERNEL IS ENOUGH
//
//  The effect is mild by nature - the buckle loss on real stocks runs from about
//  0.03 to 0.30 - and it is one of the least precisely characterised numbers in
//  the whole profile, since it depends on the camera, the age of the pressure
//  plate and how long the film sat in the gate. A small fixed binomial kernel
//  blended in by radius is comfortably inside the effect's own uncertainty and
//  costs a few operations per pixel instead of a transform.
//
//  BOUNDARY HANDLING
//
//  Edge clamp, NOT wrap. This one differs from every other blur in the engine on
//  purpose: the whole point of the stage is that the CORNERS behave differently
//  from the centre, so wrapping the corner of the frame onto the opposite corner
//  would mix precisely the two regions the effect is trying to distinguish.
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

// Stock parameters, including CoatingSpec::buckle_mtf_loss.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Half-width of the fixed blur kernel, in taps either side of centre.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_DEFOCUS_RADIUS = 2;

// ---------------------------------------------------------------------------
//  Full kernel length, 2 * radius + 1.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_DEFOCUS_TAPS = (2 * ALGO_DEFOCUS_RADIUS) + 1;

// ---------------------------------------------------------------------------
//  Normalised binomial kernel [1 4 6 4 1] / 16.
//
//  Binomial because it is the discrete approximation to a Gaussian with the
//  smallest possible support for its smoothness, and because the divisor is a
//  power of two so the normalisation is exact in binary floating point and the
//  taps sum to precisely one. A kernel whose taps do not sum to one would apply
//  a radially varying brightness change on top of the softness, which is exactly
//  the corner darkening this stage is at pains not to produce.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_DEFOCUS_TAP_0 = static_cast<AlgoType>(1.0 / 16.0);
constexpr AlgoType ALGO_DEFOCUS_TAP_1 = static_cast<AlgoType>(4.0 / 16.0);
constexpr AlgoType ALGO_DEFOCUS_TAP_2 = static_cast<AlgoType>(6.0 / 16.0);

// ---------------------------------------------------------------------------
//  Upper bound on the blend weight.
//
//  At a weight of one the corner would be the fully blurred image with none of
//  the original left, which is beyond anything a gate can do to a frame. The cap
//  keeps a pathological profile or a large user scale from producing a corner
//  that is pure kernel.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_DEFOCUS_MAX_LOSS = static_cast<AlgoType>(0.9);


// ---------------------------------------------------------------------------
//  Sub-stage 6b: corner defocus.
//
//  pSrcR/G/B     linear exposure in
//  pDstR/G/B     linear exposure out
//  pScrH         scratch: horizontal-pass intermediate for one plane
//  pScrV         scratch: fully blurred version of one plane
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  params        user controls; coatingScale scales the buckle loss
//
//  The two scratch planes must be distinct from each other, from the source and
//  from the destination.
// ---------------------------------------------------------------------------
void AlgoStage06b_CornerDefocus
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrH,
    AlgoType* RESTRICT       pScrV,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params
) noexcept;
