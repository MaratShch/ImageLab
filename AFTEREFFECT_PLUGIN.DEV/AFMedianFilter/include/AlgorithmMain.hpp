#pragma once

#include <cstdint>
#include "Common.hpp"
#include "AlgoControls.hpp"
#include "AlgoMemHandler.hpp"


void Algorithm_Main
(
	const MemHandler& memHandler,
	const int32_t sizeX,
	const int32_t sizeY,
	const AlgoControls& algoCtrl
);

void ApplyMirroredPadding
(
    float* inY, 
    const int32_t width, 
    const int32_t height, 
    const int32_t stride, 
    const int32_t radius // The currently requested UI radius
);

void ProcessImage_Scalar
(
    const float* RESTRICT inY, 
    float* RESTRICT outY, 
    const int32_t sizeX, 
    const int32_t sizeY, 
    const int32_t strideY_Elements, 
    const AlgoControls& ctrl
);