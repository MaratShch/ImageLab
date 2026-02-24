#ifndef __IMAGE_LAB2_NOSIE_REDUCTION_ALGO_COLOR_CONVERT__
#define __IMAGE_LAB2_NOSIE_REDUCTION_ALGO_COLOR_CONVERT__

#include <immintrin.h>
#include <cstdint>
#include <algorithm>
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


void AVX2_Convert_VUYA_8u_YUV
(
    const PF_Pixel_VUYA_8u* RESTRICT pInput, // Input VUYA_8u (Interleaved)
    float* RESTRICT Y_out,                   // Y plane (Luma)
    float* RESTRICT U_out,                   // U plane (Red-Blue axis)
    float* RESTRICT V_out,                   // V plane (Green-Magenta axis)
    int32_t sizeX,                           // Width in Pixels
    int32_t sizeY,                           // Height in Pixels
    int32_t linePitch,                       // Row Pitch in Pixels
    bool isBT709
 ) noexcept;

void AVX2_Convert_YUV_to_VUYA_8u
(
    const float* RESTRICT pY,               // Y plan [Orthonormal format]
    const float* RESTRICT pU,               // U plan [Orthonormal format]
    const float* RESTRICT pV,               // V plan [Orthonormal format]  
    const PF_Pixel_VUYA_8u* RESTRICT pInput,// VUYA_8u input pixel (Alpha source)
    PF_Pixel_VUYA_8u* RESTRICT pOutput,     // VUYA_8u denoised output image
    int32_t w,                              // horizontal frame size in pixels
    int32_t h,                              // vertical frame size in lines
    int32_t src_pitch,                      // input buffer line pitch in pixels
    int32_t dst_pitch,                      // output buffer line pitch in pixels
    bool isBT709                            // true for BT709, false for BT601
) noexcept;

#endif // __IMAGE_LAB2_NOSIE_REDUCTION_ALGO_COLOR_CONVERT__
