#include <algorithm>
#include <immintrin.h>
#include "ColorConvert.hpp"

void AVX2_Convert_RGB_10u_YUV
(
    const PF_Pixel_RGB_10u* RESTRICT pInput, // Input RGB_10u (Packed 32-bit)
    float* RESTRICT Y_out,                   // Y plane (Luma)
    float* RESTRICT U_out,                   // U plane (Red-Blue axis)
    float* RESTRICT V_out,                   // V plane (Green-Magenta axis)
    int32_t sizeX,                           // Width in Pixels
    int32_t sizeY,                           // Height in Pixels
    int32_t linePitch                        // Row Pitch in Pixels
) noexcept
{
    // 1. Initialize Orthonormal Constants
    const __m256 vSqrt3_inv = _mm256_set1_ps(0.57735027f);
    const __m256 vSqrt2_inv = _mm256_set1_ps(0.70710678f);
    const __m256 vSqrt6_inv = _mm256_set1_ps(0.40824829f);
    const __m256 vTwo       = _mm256_set1_ps(2.0f);
    
    // Scale factor to map Adobe's [0, 1023] range down to the [0.0f, 255.0f] float range
    const __m256 vScale     = _mm256_set1_ps(0.24926686f); 

    // 10-bit mask (0x3FF)
    const __m256i vMask10   = _mm256_set1_epi32(0x3FF);

    const int32_t vecSize = 8;
    const int32_t endX = (sizeX / vecSize) * vecSize;

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const PF_Pixel_RGB_10u* rowIn = pInput + (y * linePitch);
        float* rowY = Y_out + (y * sizeX);
        float* rowU = U_out + (y * sizeX);
        float* rowV = V_out + (y * sizeX);

        int32_t x = 0;

        // =========================================================
        // MAIN AVX2 KERNEL (8 Pixels per loop)
        // =========================================================
        for (; x < endX; x += vecSize)
        {
            // Load 8 packed 32-bit pixels (256 bits total)
            __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowIn + x));

            // Extract 10-bit channels using bit shifts and masks
            // Layout (LSB to MSB): Pad (2), B (10), G (10), R (10)
            __m256i b_i = _mm256_and_si256(_mm256_srli_epi32(packed, 2),  vMask10);
            __m256i g_i = _mm256_and_si256(_mm256_srli_epi32(packed, 12), vMask10);
            __m256i r_i = _mm256_and_si256(_mm256_srli_epi32(packed, 22), vMask10);

            // Convert to float and scale down to [0.0f, 255.0f]
            __m256 fR = _mm256_mul_ps(_mm256_cvtepi32_ps(r_i), vScale);
            __m256 fG = _mm256_mul_ps(_mm256_cvtepi32_ps(g_i), vScale);
            __m256 fB = _mm256_mul_ps(_mm256_cvtepi32_ps(b_i), vScale);

            // Fast Orthonormal FMA Math
            __m256 resY = _mm256_mul_ps(_mm256_add_ps(fR, _mm256_add_ps(fG, fB)), vSqrt3_inv);
            __m256 resU = _mm256_mul_ps(_mm256_sub_ps(fR, fB), vSqrt2_inv);
            __m256 twoG = _mm256_mul_ps(fG, vTwo);
            __m256 resV = _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(fR, fB), twoG), vSqrt6_inv);

            // Store Planar Results
            _mm256_storeu_ps(rowY + x, resY);
            _mm256_storeu_ps(rowU + x, resU);
            _mm256_storeu_ps(rowV + x, resV);
        }

        // =========================================================
        // SCALAR TAIL
        // =========================================================
        for (; x < sizeX; ++x)
        {
            float r = static_cast<float>(rowIn[x].R) * 0.24926686f;
            float g = static_cast<float>(rowIn[x].G) * 0.24926686f;
            float b = static_cast<float>(rowIn[x].B) * 0.24926686f;

            rowY[x] = (r + g + b) * 0.57735027f;
            rowU[x] = (r - b) * 0.70710678f;
            rowV[x] = (r + b - 2.0f * g) * 0.40824829f;
        }
    }
}
