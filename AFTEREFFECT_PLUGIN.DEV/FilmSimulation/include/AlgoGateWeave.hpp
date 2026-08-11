#pragma once

// ---------------------------------------------------------------------------
//  AlgoGateWeave.hpp
//
//  Stage 15 of the film simulation pipeline: gate weave and registration
//  instability.
//
//  WHY A TRANSLATION OF A FEW MICROMETRES EARNS A WHOLE STAGE
//
//  Weave is not a mark on the film. It is the CARRIER for every other defect's
//  temporal behaviour, and it decides whether a defect looks attached to the image
//  or attached to the screen.
//
//  Without weave a gate scratch is a mathematically straight, perfectly static
//  line, and it reads as a digital overlay instantly - which is exactly how most
//  film-emulation plugins give themselves away. With weave the image shimmers
//  underneath that line, the line stays put, and the eye reads the line as a
//  physical object in the projector rather than a graphic laid over the picture.
//
//  THE INTERACTION RULE, WHICH THE PIPELINE ORDER ALREADY ENCODES
//
//  Film-borne defects move WITH the image. Machine-fixed defects do NOT.
//
//  That single rule is what sells the effect, and it costs nothing here because
//  the stage order expresses it directly:
//
//      9b   film-borne particulate      before this stage, so it is translated
//      15   THIS STAGE - the weave
//      16   gate dirt, machine-fixed    after this stage, so it is NOT translated
//
//  So gate dirt is stationary in the FRAME while the image jitters beneath it,
//  and embedded dust is stationary relative to the IMAGE. No flag, no special
//  case, no coordinate bookkeeping - the two populations end up on opposite sides
//  of one translation and the physics falls out.
//
//  WHAT MOVES, AND BY HOW MUCH
//
//  A per-frame two-dimensional translation of the image relative to the frame.
//  Amplitudes come from the stock's own TemporalSpec, which carries an era figure
//  for every profile in the database: about 20 to 25 micrometres RMS for 1930s
//  and 1940s material, 10 for the 1950s, 6 by the 1970s and 3 for a modern
//  pin-registered camera. The user control scales that, so era drives the
//  baseline and the control expresses intent.
//
//  Vertical instability exceeds horizontal on vertically-transported formats,
//  which is why the profile carries two amplitudes rather than one.
//
//  THE SPECTRUM IS RED, AND THE PERFORATION FREQUENCY CANNOT BE REPRESENTED
//
//  Real weave is red-noise dominated below about 5 Hz, with additional components
//  at the frame rate and at the perforation-passing frequency - 96 Hz for 4-perf
//  35 mm at 24 frames per second.
//
//  NONE of those high components can be reproduced by a per-frame translation, and
//  it is worth being explicit about why rather than quietly leaving them out. The
//  image is sampled once per frame, so the Nyquist limit of this signal is half
//  the frame rate: 12 Hz at 24 fps. A 96 Hz perforation component sampled at 24 Hz
//  aliases to exactly 0 Hz - a constant offset, not a vibration. Synthesising it
//  and sampling it here would not add the shimmer it produces in reality; it would
//  add a fixed displacement and a false sense that the effect was modelled.
//
//  What CAN be represented is the red-noise part below the frame rate, which is
//  the part the eye reads as weave anyway. The corner frequency comes from the
//  profile.
//
//  STATELESS, LIKE EVERY OTHER TEMPORAL EFFECT HERE
//
//  The displacement is a pure function of the frame index. Rendering frames out of
//  order, scrubbing a timeline backwards, or re-rendering a single frame in
//  isolation all give the same answer, which is a hard requirement for a host that
//  may ask for any frame at any time.
//
//  RESAMPLING
//
//  Bilinear, with the edge clamped. Bilinear rather than anything sharper because
//  the displacement is a fraction of a pixel at every plausible amplitude, and at
//  sub-pixel shifts a windowed-sinc kernel buys nothing measurable while costing
//  several times as much. The clamp matters: a shifted frame exposes up to one
//  pixel of undefined data at one edge, and clamping is what a real gate does -
//  the aperture simply shows slightly more or less of the frame.
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

// Counter-based generator: the displacement is a pure function of frame index.
#include "AlgoCounterRng.hpp"

// AlgoDefectHash, which keys the temporal lattice on the frame index. Shared with
// the defect stages rather than re-derived here: the two must agree, because a
// second mixer would let weave and dirt correlate for no physical reason.
#include "AlgoDefectField.hpp"

// Stock parameters, including TemporalSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t, uint32_t


// ---------------------------------------------------------------------------
//  Octaves in the weave's temporal noise.
//
//  5. The lowest octave has the period set by the profile's corner frequency and
//  each one above halves it, so five octaves span a factor of sixteen in
//  frequency - from the corner down to the frame rate, which is where the
//  representable band ends. A sixth would sit above Nyquist and only alias.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_WEAVE_OCTAVES = 5;

// ---------------------------------------------------------------------------
//  Variance lost to interpolating the temporal lattice.
//
//  26/35, exactly.
//
//  The noise is built by interpolating independent values on a lattice of frame
//  indices, and a smoothly interpolated point is a weighted MEAN of its two
//  neighbours - so its variance is the sum of the squared weights, which is one
//  only exactly at a node. With W = 3u^2 - 2u^3:
//
//      E[W]   = 1/2
//      E[W^2] = 9/5 - 12/6 + 4/7 = 13/35
//      E[(1-W)^2 + W^2] = 1 - 2E[W] + 2E[W^2] = 26/35
//
//  Dividing it out is what makes the requested RMS amplitude the amplitude that
//  actually appears. The same mistake in the two-dimensional dirt field produced a
//  weave that measured three quarters of what was asked for, which is small enough
//  to look plausible and wrong enough to fail a measurement.
//
//  ANYONE CHANGING THE INTERPOLATION KERNEL MUST RECOMPUTE THIS.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_WEAVE_INTERP_VARIANCE = 26.0 / 35.0;

// ---------------------------------------------------------------------------
//  Micrometres per millimetre. Weave amplitudes are quoted in micrometres on the
//  negative; the geometry is in millimetres.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_WEAVE_UM_PER_MM = 1000.0;

// ---------------------------------------------------------------------------
//  Smallest displacement worth resampling the frame for, pixels.
//
//  1/512 of a pixel. Below this the bilinear weights round to a pass-through and
//  the whole stage is an expensive copy, so it takes the copy path instead. This
//  is not a quality compromise: a 1/512 pixel shift is far below the precision of
//  the storage type at any bit depth a host will ever hand over.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_WEAVE_MIN_SHIFT_PX = 1.0 / 512.0;

// ---------------------------------------------------------------------------
//  Generator stream tags for the two axes.
//
//  Separate, so that horizontal and vertical weave are independent. Sharing a
//  stream would make the image travel along a diagonal, which is a distinctive and
//  quite wrong-looking motion.
// ---------------------------------------------------------------------------
constexpr uint32_t ALGO_WEAVE_TAG_X = 0x0057EA01u;
constexpr uint32_t ALGO_WEAVE_TAG_Y = 0x0057EA02u;


// ---------------------------------------------------------------------------
//  AlgoWeaveNoise
//
//  One sample of unit-variance red noise at a given frame index.
//
//  Built from ALGO_WEAVE_OCTAVES lattices whose periods halve, with amplitudes
//  halving too - which is the 1/f amplitude weighting that makes the power
//  spectrum fall as 1/f^2, the red-noise shape weave actually has.
//
//  framePos    frame index, as a real number so the lattice can be interpolated
//  periodLo    period of the lowest octave, in frames
//  seed        roll seed
//  tag         axis tag
// ---------------------------------------------------------------------------
HighPrecType AlgoWeaveNoise
(
    const HighPrecType framePos,
    const HighPrecType periodLo,
    const uint32_t     seed,
    const uint32_t     tag
) noexcept;


// ---------------------------------------------------------------------------
//  Stage 15: gate weave.
//
//  pSrcR/G/B      display-linear transmittance in
//  pDstR/G/B      out
//  pScrA/pScrB    scratch planes, retained in the signature for a future
//                 separable resample; the bilinear path needs neither
//  sizeX/sizeY    active pixel extent
//  pitch          row stride in ELEMENTS
//  profile        stock being simulated; TemporalSpec supplies the era amplitude
//  params         user controls; damage.weaveAmount scales the era figure
//  negWidthMm     frame width on the film
//  negHeightMm    frame height on the film
//  pxPerMm        render resolution
//  frameIndex     clip-relative frame number
//  frameRate      frames per second OF FILM, which sets the Nyquist limit
//  seed           per-call seed, combined with params.damage.damageSeed
// ---------------------------------------------------------------------------
void AlgoStage15_GateWeave
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
    const AlgoType           negWidthMm,
    const AlgoType           negHeightMm,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
) noexcept;
