#ifndef __IMAGE_LAB2_NOSIE_REDUCTION_ALGO_COLOR_CONVERT__
#define __IMAGE_LAB2_NOSIE_REDUCTION_ALGO_COLOR_CONVERT__

#include <immintrin.h>
#include <cstdint>
#include "Common.hpp"
#include "CommonPixFormat.hpp"

constexpr float MAX_Y = 441.6729f;
constexpr float OFFSET_U = 180.3122f;
constexpr float SPAN_U = 360.6244f;
constexpr float OFFSET_V = 208.2066f;
constexpr float SPAN_V = 416.4132f;


void AVX2_Convert_BGRA_8u_YUV
(
    const PF_Pixel_BGRA_8u* RESTRICT pInput,// Input BGRA_8u (Interleaved)
    float* RESTRICT Y_out,                	// Y plane (Luma)
    float* RESTRICT U_out,                	// U plane (Red-Blue axis)
    float* RESTRICT V_out,                	// V plane (Green-Magenta axis)
    int32_t sizeX,                        	// Width in Pixels
    int32_t sizeY,                        	// Height in Pixels
    int32_t linePitch                     	// Row Pitch in Pixels
) noexcept;


#endif // __IMAGE_LAB2_NOSIE_REDUCTION_ALGO_COLOR_CONVERT__
