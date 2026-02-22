#pragma once

#include <cstdint>
#include <immintrin.h>
#include "Common.hpp"
#include "CommonPixFormat.hpp"

// Assuming PF_Pixel_BGRA_8u is defined in your Adobe SDK headers
// struct PF_Pixel_BGRA_8u { uint8_t B, G, R, A; };

void AVX2_Convert_YUV_to_BGRA_8u
(
    const float* RESTRICT pY,
    const float* RESTRICT pU,
    const float* RESTRICT pV,
    const PF_Pixel_BGRA_8u* RESTRICT pInput, 
    PF_Pixel_BGRA_8u* RESTRICT pOutput,
    int32_t w,
    int32_t h,
    int32_t src_pitch,
    int32_t dst_pitch
);