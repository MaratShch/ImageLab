#pragma once

// ---------------------------------------------------------------------------
//  AlgoTemporalFlicker.hpp
//
//  Sub-stage 3c of the film simulation pipeline: temporal exposure flicker.
//
//  ==================== NOT YET MODELLED -- PASS-THROUGH ====================
//
//  This stage is a STUB. It takes every parameter the real implementation will
//  need and copies its input to its output unchanged. It is wired into the chain
//  now so that adding the model later is a change to one function body and to
//  nothing else - no new buffer, no new call site, no change to Algorithm_Main.
//
//  WHY IT IS EMPTY RATHER THAN ABSENT
//
//  We do not yet have trustworthy figures. Flicker is a property of a particular
//  camera's shutter and a particular projector's mechanism, not of the film, and
//  the numbers quoted in the literature are almost all single anecdotal
//  measurements. Inventing a plausible-looking amplitude and calling it
//  simulation would produce output that looks period-correct and is not, which is
//  worse than leaving it flat - a flat result is honestly wrong, and obviously so.
//
//  WHAT THE REAL STAGE WILL DO
//
//  Hand-cranked cameras and early intermittent mechanisms did not deliver equal
//  exposure to successive frames. The variation is a low-frequency random walk
//  rather than white noise, so it reads as a slow breathing of brightness rather
//  than as per-frame noise, and it is partly INDEPENDENT PER CHANNEL because the
//  three layers have different reciprocity behaviour at short exposures.
//
//  It must act on EXPOSURE, before the characteristic curve, which is why it
//  belongs at 3c and not later: a brightness change applied after development is
//  a grade, and a grade does not move highlights through the shoulder the way a
//  genuine exposure change does. That difference is the whole visible signature.
//
//  WHAT IT NEEDS BEFORE IT CAN BE WRITTEN
//
//    - RMS amplitude in stops, by era and by mechanism. TemporalSpec carries
//      flicker_pct and flicker_hz, and AlgoControls carries
//      damage.flickerStops.
//      !! CORRECTED 2026-08-28: this list previously claimed AlgoControls also
//      carried flickerBaseHz and flickerColourSpread. NEITHER FIELD EXISTS,
//      anywhere in either instruction-set tree. The plumbing is therefore ONE
//      control, not three, and the two missing axes below have nowhere to be
//      set from. Whoever implements this stage must either add those fields --
//      APPENDED LAST, per the layout rule -- or derive both from the profile's
//      TemporalSpec and drop them from the control surface. Recorded rather
//      than silently invented.
//    - The spectral shape. A corner frequency alone does not say whether the
//      spectrum rolls off at 1/f or 1/f^2, and the two look quite different.
//    - How much of the variation is common to all three channels and how much is
//      per-channel. There is NO control for that split today; see the
//      correction above.
//
//  Frame index and frame rate are already in the parameter list because the
//  quantity is a function of TIME, and the counter-based generator will make it a
//  pure function of (seed, frameIndex) so that scrubbing and out-of-order
//  rendering stay stable.
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

// Stock parameters, including TemporalSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Sub-stage 3c: temporal exposure flicker.  STUB - copies input to output.
//
//  pSrcR/G/B     linear exposure in
//  pDstR/G/B     linear exposure out
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated; TemporalSpec carries the era figures
//  params        user controls; damage.flickerStops is the only one that
//                exists -- see the correction in the header block above
//  frameIndex    clip-relative frame number; the quantity varies with TIME
//  frameRate     frames per second of FILM, following layer time stretch
//  seed          per-call seed, combined with params.seed by the generator
// ---------------------------------------------------------------------------
void AlgoStage03c_TemporalFlicker
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
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
) noexcept;
