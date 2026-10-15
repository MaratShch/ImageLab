#pragma once

// ---------------------------------------------------------------------------
//  AlgoTemporalFlicker.hpp
//
//  Sub-stage 3c of the film simulation pipeline: temporal exposure flicker.
//
//  ==================== MODELLED 2026-10-10 (damage.flickerStops) =============
//
//  Until 2026-10-10 this stage was a stub that copied its input. It now reads
//  damage.flickerStops, the last FilmDamage control that had no reader, and
//  applies ONE exposure gain per frame, before the characteristic curve.
//
//  WHY IT ACTS ON EXPOSURE
//
//  A brightness change applied after development is a grade, and a grade does
//  not move highlights through the shoulder the way a genuine exposure change
//  does. That difference is the whole visible signature, so the gain is applied
//  here, at 3c, to linear exposure.
//
//  THE MODEL, AND WHAT IN IT IS A DEFINITION
//
//    amplitude   damage.flickerStops x damage.damageStrength, read as the RMS
//                of the gain in photographic STOPS (AlgoControl.hpp item 4).
//    time        t = frameIndex / frameRate; a non-positive frameRate falls
//                back to the stock's TemporalSpec::fps_native, then to 24.
//    spectrum    unit-variance value noise in time: independent standard
//                normal knots every 1 / TemporalSpec::flicker_hz seconds,
//                blended with a smooth-step weight and renormalised so the
//                variance is exactly 1 at every instant. flicker_hz <= 0 falls
//                back to 1 Hz. THE SHAPE IS A DEFINITION: no source in the
//                corpus gives the flicker spectrum, so the roll-off between
//                knots is chosen, not measured.
//    channels    COMMON-MODE ONLY. The header used to note that real flicker is
//                partly independent per layer; no control and no database
//                field gives that split, so none is invented - all three
//                channels take the same gain.
//    time base   a pure function of (damageSeed, frameIndex), so scrubbing and
//                out-of-order rendering stay stable.
//
//  TemporalSpec::flicker_pct (the era RMS in percent) is NOT read: the control
//  is already an absolute RMS in stops, and multiplying the two would need a
//  normalisation no source supplies.
//
//  INERT unless filmDamageEnabled, damageStrength > 0 and flickerStops > 0
//  (AlgoPass03c skips the stage entirely in that case).
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

// Counter-based generator for the knot values.
#include "AlgoCounterRng.hpp"

#include <cmath>     // std::floor, std::sqrt, std::exp2
#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  AlgoFlickerGain
//
//  The linear exposure gain for one frame. Returns exactly 1 on every inert
//  path. Shared by both trees so the two cannot drift.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoFlickerGain
(
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
) noexcept
{
    if (false == params.filmDamageEnabled)
        return 1.0;

    const FilmDamage& dmg = params.damage;

    const HighPrecType strength =
        MAX_VALUE(static_cast<HighPrecType>(dmg.damageStrength), 0.0);
    const HighPrecType stops =
        MAX_VALUE(static_cast<HighPrecType>(dmg.flickerStops), 0.0) * strength;

    if (stops <= 0.0)
        return 1.0;

    HighPrecType fps = static_cast<HighPrecType>(frameRate);
    if (fps <= 0.0)
        fps = static_cast<HighPrecType>(profile.temporal.fps_native);
    if (fps <= 0.0)
        fps = 24.0;

    HighPrecType hz = static_cast<HighPrecType>(profile.temporal.flicker_hz);
    if (hz <= 0.0)
        hz = 1.0;

    const HighPrecType knot = (static_cast<HighPrecType>(frameIndex) / fps) * hz;
    const HighPrecType k0   = std::floor(knot);
    const HighPrecType w    = knot - k0;
    const int64_t      i0   = static_cast<int64_t>(k0);

    const uint32_t rollSeed = static_cast<uint32_t>(dmg.damageSeed) ^ seed;
    const uint64_t base =
        AlgoRngCounter(rollSeed, 0, eALGO_RNG_STAGE::eRNG_FLICKER, 0u);

    const HighPrecType a = AlgoRngNormal(
        AlgoRngMix64(base ^ (static_cast<uint64_t>(i0)     * ALGO_RNG_GOLDEN)));
    const HighPrecType b = AlgoRngNormal(
        AlgoRngMix64(base ^ (static_cast<uint64_t>(i0 + 1) * ALGO_RNG_GOLDEN)));

    const HighPrecType u  = CLAMP_VALUE(w, 0.0, 1.0);
    const HighPrecType sw = u * u * (3.0 - 2.0 * u);
    const HighPrecType v  = (a * (1.0 - sw) + b * sw)
                          / std::sqrt((1.0 - sw) * (1.0 - sw) + sw * sw);

    return std::exp2(stops * v);
}


// ---------------------------------------------------------------------------
//  Sub-stage 3c: temporal exposure flicker - one common gain per frame.
//
//  pSrcR/G/B     linear exposure in
//  pDstR/G/B     linear exposure out
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated; TemporalSpec carries the era figures
//  params        user controls; damage.flickerStops (RMS stops) and the
//                filmDamageEnabled / damageStrength gates
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
