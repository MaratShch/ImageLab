#pragma once

// ---------------------------------------------------------------------------
//  AlgoGateDefects.hpp
//
//  Stage 16 of the film simulation pipeline: machine-side defects.
//
//      gate dirt        debris lodged in the gate, fixed in the FRAME for a long
//                       run, slowly accreting and occasionally shedding
//      one-frame dirt   the loose particles riding through on this frame only -
//                       what everyone calls sparkle
//      events           splices and edge damage passing through the aperture
//
//  WHY THIS IS A DIFFERENT STAGE FROM 9b, AND NOT A DUPLICATE OF IT
//
//  Stage 9b renders the same MATERIAL - ordinary dust and debris. What differs is
//  where it sits, and that changes everything the viewer perceives:
//
//                       9b, film-borne              16, machine-side
//      position         locked to the film          locked to the frame
//      survives print   yes, and every dupe         no
//      moves with weave yes                         no
//      polarity         bright on the positive      dark on the positive
//      lifetime         forever                     one frame, or a long run
//
//  In still photography this class does not exist as a separate thing - there is
//  no gate the film passes through twice. In motion picture it is a first-class
//  defect and cannot be left out, because it is most of what an audience actually
//  sees.
//
//  POLARITY IS OPPOSITE, AND THIS IS THE PART WORTH GETTING RIGHT
//
//  A speck embedded in the negative blocks printing light, so that patch of print
//  is never exposed and comes out CLEAR - a bright mark. That is stage 9b, and it
//  is why those particles are added as density before the print.
//
//  A speck in the projector gate blocks projection light, so that patch of screen
//  goes DARK. This stage works on the finished positive in transmittance, and
//  applies
//
//      T' = T * (1 - alpha)
//
//  which darkens. Two populations of the same dirt with opposite polarity is not
//  an inconsistency - it is what a real projected print looks like, and rendering
//  both is a large part of why film reads as film.
//
//  THE THREE TEMPORAL CLASSES, AND WHY THE SPLIT MATTERS MORE THAN THE AMOUNT
//
//      loose dust on the film at scan or projection   50 - 80 %   one frame
//      embedded processing dust                       15 - 40 %   stage 9b
//      dirt lodged in the gate                         5 - 20 %   long run
//
//  This stage owns the first and third. A common and very visible mistake is to
//  give one-frame dirt a little frame-to-frame correlation "to make it less
//  noisy". That is precisely backwards: the complete absence of correlation IS the
//  sparkle, and smoothing it destroys the one cue that says the film is moving.
//
//  PERSISTENT GATE DIRT WITHOUT ANY PERSISTENT STATE
//
//  Gate dirt accretes over a screening and occasionally sheds, so the obvious
//  implementation keeps a running population and steps it once per frame. This
//  engine cannot do that: Algorithm_Main is a pure function, frames may be
//  rendered out of order or in isolation, and a host may ask for frame 900 without
//  ever having rendered frame 899.
//
//  So the population is not accumulated - it is DERIVED. Each candidate slot in a
//  fixed pool draws a birth frame and a lifetime from its own stream, keyed on the
//  reel and the slot ordinal. A slot is present at frame f exactly when
//
//      birth <= f < birth + lifetime
//
//  which is a closed-form test costing two draws and no history. Birth times are
//  the arrivals of a Poisson accretion process and lifetimes are exponential with
//  the shed rate, so the population statistics are the ones the model calls for,
//  and the count still grows through a reel and resets at the change. Scrubbing
//  backwards, rendering frame 900 alone, and rendering the reel in order all give
//  the same dirt.
//
//  GATE DIRT DOES NOT MOVE WITH THE PICTURE
//
//  This stage runs AFTER the weave at 15, so it is not displaced by it, while
//  everything film-borne was. The image jitters underneath stationary gate dirt.
//  That inverse relationship is subtle, entirely free here because it falls out of
//  the stage order, and is one of the strongest cues that the dirt is a physical
//  object in a machine rather than a graphic laid over the picture.
//
//  WHERE IT SITS IN THE FRAME
//
//  Concentrated at the aperture edges and corners, where the plate touches the
//  film and scrapes it, decaying inward over a couple of millimetres. Placement is
//  therefore a one-dimensional process along the aperture perimeter with an
//  inward decay, not a uniform scatter over the frame.
//
//  HONEST GRADING OF THE NUMBERS BELOW
//
//  The dataset these constants come from CANNOT calibrate this class. The scanner
//  used for the measurements was demonstrably free of static particulate, which is
//  a clean negative result but leaves gate dirt uncalibrated. The mechanism and
//  the temporal taxonomy are solid; every rate, size and bias below is an
//  engineering estimate within the documented plausible ranges, and is marked as
//  such at each constant. That is a materially weaker footing than the fine-dust
//  constants at stage 9b, which were solved against measurement, and it should not
//  be presented otherwise.
//
//  What IS anchored is the per-frame event rate, which comes from each stock's own
//  TemporalSpec - populated for all 93 profiles, unlike AgingSpec - so era drives
//  the baseline exactly as it does for weave.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"
// 2026-10-20c: this header's compile-time constants live in film_damage_constants.hpp.
#include "film_damage_constants.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Shared defect machinery: hashing, Poisson, power law, blob coverage.
#include "AlgoDefectField.hpp"

// Stock parameters, including TemporalSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t, uint32_t

// ---------------------------------------------------------------------------
//  Stage 16: machine-side defects.
//
//  pSrcR/G/B      display-linear transmittance in
//  pDstR/G/B      out
//  sizeX/sizeY    active pixel extent
//  pitch          row stride in ELEMENTS
//  profile        stock being simulated; TemporalSpec supplies the era event rate
//  params         user controls; damage.gateDirt and damage.damageEvents
//  negWidthMm     frame width on the film, so sizes scale with gauge
//  negHeightMm    frame height on the film
//  pxPerMm        render resolution
//  frameIndex     clip-relative frame number; this stage is entirely temporal
//  frameRate      frames per second OF PROJECTION - correct here, because this
//                 damage happens at the machine rather than on the film
//  seed           per-call seed, combined with params.damage.damageSeed
// ---------------------------------------------------------------------------
void AlgoStage16_GateDefects
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
    const AlgoType           negWidthMm,
    const AlgoType           negHeightMm,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
) noexcept;
