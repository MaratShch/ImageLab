#pragma once

// ---------------------------------------------------------------------------
//  AlgoSilverTone.hpp
//
//  Sub-stage 14c of the film simulation pipeline: silver image tone, monochrome
//  stocks only.
//
//  PHYSICAL BACKGROUND
//
//  Developed silver is not spectrally neutral, and this is why a black-and-white
//  print is almost never actually grey. Fine silver particles scatter short
//  wavelengths and read WARM; coarse filamentary silver reads neutral to slightly
//  blue. Which one a stock produces depends on its crystal habit and on the
//  developer.
//
//  WHY IT IS WEIGHTED BY OUTPUT LEVEL
//
//  The effect is strongest where there is LEAST silver - the light tones - and
//  fades as density builds and the particles overlap. So it is weighted by the
//  output level rather than applied flat. A flat tint would warm the shadows as much
//  as the highlights, which is the opposite of how a warm-toned print looks.
//
//  WHY IT MUST RUN AFTER THE ANCHOR SOLVES, NOT BEFORE
//
//  This is the whole reason the stage exists as a separate step at the end.
//
//  base_tint is COMPENSATED by the printer-light solve - that is precisely what a
//  printer light does, and it is why base_tint cannot tint a black-and-white stock
//  at all: whatever it asks for, the solve removes. This stage is downstream of the
//  solve and therefore survives it.
//
//  Putting the silver tone into base_tint instead, which looks like the obvious
//  place for it, produces exactly nothing.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// The copy helpers used when the stage does not apply.
#include "AlgoSeparableBlur.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including silver_tone.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Warm and cool coefficients of the tone.
//
//  A positive silver_tone lifts red and lowers blue, which is a warm image; a
//  negative one does the reverse. The two magnitudes are deliberately unequal -
//  scattering is stronger at the blue end, so the blue side moves less for the same
//  physical cause, and equal coefficients would read as a plain hue rotation rather
//  than as a warm-toned print.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_SILVER_TONE_RED  = static_cast<AlgoType>(0.28);
constexpr AlgoType ALGO_SILVER_TONE_BLUE = static_cast<AlgoType>(0.22);


// ---------------------------------------------------------------------------
//  Sub-stage 14c: silver image tone.
//
//  pSrcR/G/B     display-linear transmittance in
//  pDstR/G/B     transmittance out
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated; silver_tone and is_monochrome
//
//  No scratch planes: the tone is a pointwise function of the green level.
// ---------------------------------------------------------------------------
void AlgoStage14c_SilverTone
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
    const film::FilmProfile& profile
) noexcept;
