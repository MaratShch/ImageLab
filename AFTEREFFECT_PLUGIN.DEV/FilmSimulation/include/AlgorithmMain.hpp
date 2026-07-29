#pragma once

#include <cstdint>
#include "AlgoMemHandler.hpp"     // MemHandler, RGBPlanes
#include "AlgoControl.hpp"     // S_PcaAwbParams

void Algorithm_Main
(
    const MemHandler&      memHandler,
    const int32_t          sizeX,
    const int32_t          sizeY,
    const AlgoControls&    params,
    const int64_t          frameIdx,
    const double           fps
);
