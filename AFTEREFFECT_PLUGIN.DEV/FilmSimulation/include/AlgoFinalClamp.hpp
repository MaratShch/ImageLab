#pragma once

// ---------------------------------------------------------------------------
//  AlgoFinalClamp.hpp
//
//  Stage 17 of the film simulation pipeline: the single final clamp, and the
//  narrowing back to storage type.
//
//  THE ONE AND ONLY CLAMP TO THE DISPLAY RANGE
//
//  Every earlier stage in this engine leaves its output UNCLAMPED at the top. That
//  is deliberate and it is a load-bearing decision, not laziness.
//
//  The characteristic curve's shoulder needs real highlight information above the
//  nominal white point in order to roll it off. Clamp at stage 2 and the shoulder
//  has nothing left to work with, so highlights arrive at it already flat and come
//  out looking exactly like clipped digital highlights - which is the single most
//  common reason a film simulation fails to convince, whatever else it gets right.
//  The same argument applies at every intermediate stage: veiling flare, halation
//  and print grain all legitimately push values above one, and the stages after
//  them are entitled to see that.
//
//  So the range is clamped exactly once, here, at the point where the numbers stop
//  being physical quantities and become display values.
//
//  THE FLOOR IS A DIFFERENT MATTER
//
//  Several earlier stages DO floor at zero, and that is not an inconsistency. A
//  negative exposure and a negative optical density are physically meaningless -
//  the second would be a material that emits light - and the logarithm at stage 8
//  and the exponentiation at stage 14 both require non-negative input. Those are
//  physical floors on quantities that cannot be negative, not display clamps.
//
//  NARROWING BACK TO STORAGE TYPE
//
//  This is also the one place where AlgoType narrows back to ImgType. The widening
//  happened once, in stage 2, at the engine's input boundary; this is the matching
//  output boundary. Between them everything is AlgoType.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Stage 17: final clamp, and narrow to the storage planes.
//
//  pSrcR/G/B     display-linear transmittance in, AlgoType, unclamped
//  pStageR/G/B   the stage's own retained AlgoType output, clamped - kept so the
//                clamped result can be inspected without re-reading the narrowed
//                storage planes, which is the same debugging convenience every
//                other stage in the chain provides
//  pDstR/G/B     STORAGE type destination handed in by the caller
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS, shared by both element types because the
//                arena's padded width satisfies the larger alignment quantum
// ---------------------------------------------------------------------------
void AlgoStage17_FinalClamp
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pStageR,
    AlgoType* RESTRICT       pStageG,
    AlgoType* RESTRICT       pStageB,
    ImgType* RESTRICT        pDstR,
    ImgType* RESTRICT        pDstG,
    ImgType* RESTRICT        pDstB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch
) noexcept;
