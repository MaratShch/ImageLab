#pragma once

#include <cstdint>
#include "Common.hpp"
#include "AlgoControl.hpp"
#include "AlgoMemHandler.hpp"   // MemHandler, RGBPlanes

// -----------------------------------------------------------------------------
// Pure Automatic White Balance core.
//
// All buffers live in memHandler (planar linear RGB_32f, 709/sRGB primaries):
//   memHandler.input   -> read-only decoded source  (filled by the adapter)
//   memHandler.output  -> balanced result           (drained by the adapter)
//   memHandler.scratch -> internal ping/pong, used only when ctrl.sliderIterCnt > 1
//
// The core knows nothing about host pixel formats, gamma, YUV, or primaries -- all
// of that is the adapter's job. ctrl.colorSpace is ignored here (ingest/egress only).
// Uses ctrl.illuminate, ctrl.chromatic, ctrl.sliderThreshold, ctrl.sliderIterCnt.
// -----------------------------------------------------------------------------
void Algorithm_Main
(
    const MemHandler& memHandler,
    const int32_t sizeX,
    const int32_t sizeY,
    const AlgoControls& algoCtrl
) noexcept;
