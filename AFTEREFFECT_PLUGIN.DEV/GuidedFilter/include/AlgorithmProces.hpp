#ifndef __IMAGE_LAB_GUIDED_FILTER_PROCESS__
#define __IMAGE_LAB_GUIDED_FILTER_PROCESS__

#include <cstdint>
#include <algorithm>
#include "Common.hpp"
#include "AlgoControls.hpp"

// O(1) Separable Box Blur with Clamp-to-Edge boundary logic
void math_BoxBlur_O1
(
    const float* src, 
    float* dst, 
    float* temp, 
    const int32_t width, 
    const int32_t height, 
    const int32_t radius
) noexcept;

// Squares every element in the source plane (dst = src * src)
void math_Square_Elements
(
    const float* src, 
    float* dst, 
    const int32_t width, 
    const int32_t height
) noexcept;

// Computes the 'a' (variance-based) and 'b' (mean-based) coefficients
void math_Compute_Coefficients_AB
(
    const float* mean_I, 
    const float* mean_II, 
    float* coef_a, 
    float* coef_b, 
    const int32_t width, 
    const int32_t height, 
    const float epsilon
) noexcept;

// Applies the linear coefficients to the image and blends it using the 3-way mask
// (Entire_Image: mask ignored; Skin: mask * q + (1 - mask) * I; Background: (1 - mask) * q + mask * I)
void math_Apply_Filter_And_Mask
(
    const float* src_I, 
    const float* mean_a, 
    const float* mean_b, 
    const float* in_Mask, 
    float* dst, 
    const int32_t width, 
    const int32_t height, 
    const FilterMode filterMode
) noexcept;

// Skin confidence c in [0..1] per pixel from orthonormal YUV planes (0..255 engine scale).
// Model: ellipse in the BT.601 Cb/Cr plane, see AlgoControls::skinTolerance / skinSoftness.
void math_Skin_Confidence
(
    const float* plane_Y,
    const float* plane_U,
    const float* plane_V,
    float* conf,
    const int32_t width,
    const int32_t height,
    const float tolerance,
    const float softness
) noexcept;

// In-place region blend. On entry dst holds the filtered plane q.
// Skin:       dst = c * q + (1 - c) * I
// Background: dst = (1 - c) * q + c * I
// Entire_Image: dst unchanged.
void math_Blend_Region
(
    const float* src_I,
    const float* conf,
    float* dst,
    const int32_t width,
    const int32_t height,
    const FilterMode filterMode
) noexcept;

// Selection view ("Show Output" / confidence-map display) into the output planes.
// Uses the same per-pixel filter weight as math_Blend_Region:
//   w = c (Skin) or 1 - c (Background)
//   w > 0  (pixel is filtered, fully or partly) -> out = src (original input pixel, unchanged)
//   w == 0 (pixel is excluded from filtering)    -> out = 0   (black: Y = U = V = 0 -> RGB = 0)
// The source planes, the confidence plane and the filter itself are not modified.
void math_Write_Confidence_Map
(
    const float* conf,
    const float* src_Y,
    const float* src_U,
    const float* src_V,
    float* out_Y,
    float* out_U,
    float* out_V,
    const int32_t width,
    const int32_t height,
    const FilterMode filterMode
) noexcept;

#endif // __IMAGE_LAB_GUIDED_FILTER_PROCESS__