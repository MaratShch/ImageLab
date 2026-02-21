#pragma once

#include <cstdint>
#include "AlgoControls.hpp"
#include "AlgoMemHandler.hpp"
#include "ColorConvert.hpp"

void Algorithm_Main
(
	const MemHandler& memHandler,
	const int32_t sizeX,
	const int32_t sizeY,
	const AlgoControls& algoCtrl
);

void Convert_YUV_to_BGRA_8u
(
	const MemHandler& mem,
	PF_Pixel_BGRA_8u* RESTRICT pOutput,
	int32_t w,
	int32_t h,
	int32_t pitch
);

void Convert_YUV_to_BGRA_8u
(
    const float* RESTRICT pY,
    const float* RESTRICT pU,
    const float* RESTRICT pV,
	PF_Pixel_BGRA_8u* RESTRICT pOutput,
	int32_t w,
	int32_t h,
	int32_t pitch
);