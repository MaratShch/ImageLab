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

void AVX2_Convert_ARGB_8u_YUV
(
    const PF_Pixel_ARGB_8u* RESTRICT pInput, // Input ARGB_8u (Interleaved)
    float* RESTRICT Y_out,                   // Y plane (Luma)
    float* RESTRICT U_out,                   // U plane (Red-Blue axis)
    float* RESTRICT V_out,                   // V plane (Green-Magenta axis)
    int32_t sizeX,                           // Width in Pixels
    int32_t sizeY,                           // Height in Pixels
    int32_t linePitch                        // Row Pitch in Pixels
) noexcept;

void AVX2_Convert_YUV_to_ARGB_8u
(
    const float* RESTRICT pY,                // Y plane [Orthonormal format]
    const float* RESTRICT pU,                // U plane [Orthonormal format]
    const float* RESTRICT pV,                // V plane [Orthonormal format]
    const PF_Pixel_ARGB_8u* RESTRICT pInput, // ARGB_8u input pixel (Alpha source)
    PF_Pixel_ARGB_8u* RESTRICT pOutput,      // ARGB_8u denoised output
    int32_t w,
    int32_t h,
    int32_t src_pitch,
    int32_t dst_pitch
) noexcept;

void AVX2_Convert_BGRA_16u_YUV
(
    const PF_Pixel_BGRA_16u* RESTRICT pInput, // Input BGRA_16u (Interleaved)
    float* RESTRICT Y_out,                    // Y plane
    float* RESTRICT U_out,                    // U plane
    float* RESTRICT V_out,                    // V plane
    int32_t sizeX,                            // Width in Pixels
    int32_t sizeY,                            // Height in Pixels
    int32_t linePitch                         // Row Pitch in Pixels
) noexcept;

void AVX2_Convert_YUV_to_BGRA_16u
(
    const float* RESTRICT pY,                 // Y plane [Orthonormal format]
    const float* RESTRICT pU,                 // U plane [Orthonormal format]
    const float* RESTRICT pV,                 // V plane [Orthonormal format]
    const PF_Pixel_BGRA_16u* RESTRICT pInput, // BGRA_16u input pixel (Alpha source)
    PF_Pixel_BGRA_16u* RESTRICT pOutput,      // BGRA_16u denoised output
    int32_t w,
    int32_t h,
    int32_t src_pitch,
    int32_t dst_pitch
) noexcept;


void AVX2_Convert_ARGB_16u_YUV
(
    const PF_Pixel_ARGB_16u* RESTRICT pInput, // Input ARGB_16u (Interleaved)
    float* RESTRICT Y_out,                    // Y plane (Luma)
    float* RESTRICT U_out,                    // U plane (Red-Blue axis)
    float* RESTRICT V_out,                    // V plane (Green-Magenta axis)
    int32_t sizeX,                            // Width in Pixels
    int32_t sizeY,                            // Height in Pixels
    int32_t linePitch                         // Row Pitch in Pixels
) noexcept;

void AVX2_Convert_YUV_to_ARGB_16u
(
    const float* RESTRICT pY,                 // Y plane [Orthonormal format]
    const float* RESTRICT pU,                 // U plane [Orthonormal format]
    const float* RESTRICT pV,                 // V plane [Orthonormal format]
    const PF_Pixel_ARGB_16u* RESTRICT pInput, // ARGB_16u input pixel (Alpha source)
    PF_Pixel_ARGB_16u* RESTRICT pOutput,      // ARGB_16u denoised output
    int32_t w,
    int32_t h,
    int32_t src_pitch,
    int32_t dst_pitch
) noexcept;


void AVX2_Convert_BGRA_32f_YUV
(
    const PF_Pixel_BGRA_32f* RESTRICT pInput, // Input BGRA_32f (Interleaved)
    float* RESTRICT Y_out,                    // Y plane (Luma)
    float* RESTRICT U_out,                    // U plane (Red-Blue axis)
    float* RESTRICT V_out,                    // V plane (Green-Magenta axis)
    int32_t sizeX,                            // Width in Pixels
    int32_t sizeY,                            // Height in Pixels
    int32_t linePitch                         // Row Pitch in Pixels
) noexcept;

void AVX2_Convert_YUV_to_BGRA_32f
(
    const float* RESTRICT pY,                 // Y plane [Orthonormal format]
    const float* RESTRICT pU,                 // U plane [Orthonormal format]
    const float* RESTRICT pV,                 // V plane [Orthonormal format]
    const PF_Pixel_BGRA_32f* RESTRICT pInput, // BGRA_32f input pixel (Alpha source)
    PF_Pixel_BGRA_32f* RESTRICT pOutput,      // BGRA_32f denoised output
    int32_t w,
    int32_t h,
    int32_t src_pitch,
    int32_t dst_pitch
) noexcept;

void AVX2_Convert_ARGB_32f_YUV
(
    const PF_Pixel_ARGB_32f* RESTRICT pInput, // Input ARGB_32f (Interleaved)
    float* RESTRICT Y_out,                    // Y plane (Luma)
    float* RESTRICT U_out,                    // U plane (Red-Blue axis)
    float* RESTRICT V_out,                    // V plane (Green-Magenta axis)
    int32_t sizeX,                            // Width in Pixels
    int32_t sizeY,                            // Height in Pixels
    int32_t linePitch                         // Row Pitch in Pixels
) noexcept;

void AVX2_Convert_YUV_to_ARGB_32f
(
    const float* RESTRICT pY,                 // Y plane [Orthonormal format]
    const float* RESTRICT pU,                 // U plane [Orthonormal format]
    const float* RESTRICT pV,                 // V plane [Orthonormal format]
    const PF_Pixel_ARGB_32f* RESTRICT pInput, // ARGB_32f input pixel (Alpha source)
    PF_Pixel_ARGB_32f* RESTRICT pOutput,      // ARGB_32f denoised output
    int32_t w,
    int32_t h,
    int32_t src_pitch,
    int32_t dst_pitch
) noexcept;

void AVX2_Convert_VUYA_32f_YUV
(
    const PF_Pixel_VUYA_32f* RESTRICT pInput, // Input VUYA_32f (Interleaved)
    float* RESTRICT Y_out,                    // Y plane (Luma)
    float* RESTRICT U_out,                    // U plane (Red-Blue axis)
    float* RESTRICT V_out,                    // V plane (Green-Magenta axis)
    int32_t sizeX,                            // Width in Pixels
    int32_t sizeY,                            // Height in Pixels
    int32_t linePitch,                        // Row Pitch in Pixels
    bool isBT709                              // true for BT709, false for BT601
) noexcept;

void AVX2_Convert_YUV_to_VUYA_32f
(
    const float* RESTRICT pY,                 // Y plane [Orthonormal format]
    const float* RESTRICT pU,                 // U plane [Orthonormal format]
    const float* RESTRICT pV,                 // V plane [Orthonormal format]
    const PF_Pixel_VUYA_32f* RESTRICT pInput, // VUYA_32f input pixel (Alpha source)
    PF_Pixel_VUYA_32f* RESTRICT pOutput,      // VUYA_32f denoised output
    int32_t w,
    int32_t h,
    int32_t src_pitch,
    int32_t dst_pitch,
    bool isBT709                              // true for BT709, false for BT601
) noexcept;

void AVX2_Convert_RGB_10u_YUV
(
    const PF_Pixel_RGB_10u* RESTRICT pInput, // Input RGB_10u (Packed 32-bit)
    float* RESTRICT Y_out,                   // Y plane (Luma)
    float* RESTRICT U_out,                   // U plane (Red-Blue axis)
    float* RESTRICT V_out,                   // V plane (Green-Magenta axis)
    int32_t sizeX,                           // Width in Pixels
    int32_t sizeY,                           // Height in Pixels
    int32_t linePitch                        // Row Pitch in Pixels
) noexcept;

void AVX2_Convert_YUV_to_RGB_10u
(
    const float* RESTRICT pY,                // Y plane [Orthonormal format]
    const float* RESTRICT pU,                // U plane [Orthonormal format]
    const float* RESTRICT pV,                // V plane [Orthonormal format]
    PF_Pixel_RGB_10u* RESTRICT pOutput,      // RGB_10u denoised output (No pInput needed!)
    int32_t w,
    int32_t h,
    int32_t dst_pitch                        // Row Pitch in Pixels
) noexcept;

#endif // __IMAGE_LAB2_NOSIE_REDUCTION_ALGO_COLOR_CONVERT__
