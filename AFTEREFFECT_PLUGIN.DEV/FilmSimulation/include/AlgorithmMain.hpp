#pragma once

// ---------------------------------------------------------------------------
//  AlgorithmMain.hpp
//
//  Entry point of the film simulation engine.
//
//  THE PROTOTYPE IS FIXED AT FOUR PARAMETERS AND MUST NOT GROW.
//
//      void Algorithm_Main (const MemHandler&, int32_t sizeX, int32_t sizeY,
//                           const AlgoControls&) noexcept;
//
//  The division of responsibility that makes four parameters sufficient:
//
//    EVERYTHING THAT AFFECTS THE ALGORITHM comes from AlgoControls. That includes
//    the film selection (filmProfile), the gauge (filmFormat), the frame number
//    (frameIndex), the film frame rate (frameRate) and the seed - not only the
//    obvious sliders. If a value changes what the engine computes, it is a control
//    field, not a call argument.
//
//    EVERYTHING THAT IS MEMORY comes from MemHandler. The source and destination
//    planes, every retained stage buffer, every scratch plane, and the two
//    caller-owned lookup tables (profile database and format table) which the
//    engine needs but may not build for itself, since building them allocates.
//
//    ONLY THE IMAGE GEOMETRY is passed directly, because sizeX and sizeY describe
//    the request rather than the film or the memory.
//
//  Algorithm_Main is a pure function of those inputs. It allocates nothing, holds
//  no state, validates nothing, and touches no global.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType.
#include "AlgoTypes.hpp"

// MemHandler: the flat raw-pointer arena structure plus the lookup tables.
#include "AlgoMemHandler.hpp"

// AlgoControls: every parameter that affects the algorithm.
#include "AlgoControl.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Algorithm_Main
//
//  memHandler  arena and lookup tables. Source planes are read; every stage buffer
//              and, once the final stage exists, the destination planes are
//              written. All pointers are pre-validated by the caller.
//  sizeX       active pixels per row. THE authoritative image width.
//  sizeY       active rows.
//  algoCtrl    every parameter that affects the algorithm, including the film
//              stock, the gauge, the frame number, the film frame rate and the
//              seed. Pre-validated; no field is range-checked here.
//
//  Reentrant: safe to run concurrently on different frames, in any order, with one
//  arena per invocation.
// ---------------------------------------------------------------------------
void Algorithm_Main
(
    const MemHandler&    memHandler,
    const int32_t        sizeX,
    const int32_t        sizeY,
    const AlgoControls&  algoCtrl
) noexcept;
