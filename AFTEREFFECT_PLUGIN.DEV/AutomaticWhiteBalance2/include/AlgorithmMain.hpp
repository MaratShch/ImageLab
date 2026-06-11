#pragma once

#include <cstdint>
#include "AlgoMemHandler.hpp"     // MemHandler, RGBPlanes
#include "AlgoControl.hpp"     // S_PcaAwbParams

// -----------------------------------------------------------------------------
// PCA Automatic White Balance core (Cheng bright-and-dark colors PCA).
//
// Planar scene-linear RGB_32f in memHandler.input (read-only) -> memHandler.output.
// Single pass (PCA is closed-form; no iteration, scratch unused). All tuning
// comes from S_PcaAwbParams. sizeX, sizeY are in PIXELS.
//
// Self-contained: no dependency on the gray-point AlgCommonFunctions / AlgoControls.
// -----------------------------------------------------------------------------
void Algorithm_Main
(
    const MemHandler&      memHandler,
    const int32_t          sizeX,
    const int32_t          sizeY,
    const AlgoControls&  params
) noexcept;
