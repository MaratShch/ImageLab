#include <algorithm>
#include <immintrin.h>
#include "ColorConvert.hpp"

void AVX2_Convert_BGRA_32f_YUV
(
    const PF_Pixel_BGRA_32f* RESTRICT pInput, // Input BGRA_32f (Interleaved)
    float* RESTRICT Y_out,                    // Y plane (Luma)
    float* RESTRICT U_out,                    // U plane (Red-Blue axis)
    float* RESTRICT V_out,                    // V plane (Green-Magenta axis)
    int32_t sizeX,                            // Width in Pixels
    int32_t sizeY,                            // Height in Pixels
    int32_t linePitch                         // Row Pitch in Pixels
) noexcept
{
    // 1. Initialize Orthonormal Constants
    const __m256 vSqrt3_inv = _mm256_set1_ps(0.57735027f);
    const __m256 vSqrt2_inv = _mm256_set1_ps(0.70710678f);
    const __m256 vSqrt6_inv = _mm256_set1_ps(0.40824829f);
    const __m256 vTwo       = _mm256_set1_ps(2.0f);
    
    // Scale factor to map Adobe's [0.0f, 1.0f] range up to the algorithm's [0.0f, 255.0f] range
    const __m256 vScale     = _mm256_set1_ps(255.0f); 

    const int32_t vecSize = 8;
    const int32_t endX = (sizeX / vecSize) * vecSize;

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const PF_Pixel_BGRA_32f* rowIn = pInput + (y * linePitch);
        float* rowY = Y_out + (y * sizeX);
        float* rowU = U_out + (y * sizeX);
        float* rowV = V_out + (y * sizeX);

        int32_t x = 0;

        // =========================================================
        // MAIN AVX2 KERNEL (8 Pixels per loop)
        // =========================================================
        for (; x < endX; x += vecSize)
        {
            // Gather 32-bit floats directly into AVX registers.
            // Note: _mm256_set_ps takes arguments in reverse order (7 down to 0)
            __m256 fB = _mm256_set_ps(rowIn[x+7].B, rowIn[x+6].B, rowIn[x+5].B, rowIn[x+4].B, rowIn[x+3].B, rowIn[x+2].B, rowIn[x+1].B, rowIn[x].B);
            __m256 fG = _mm256_set_ps(rowIn[x+7].G, rowIn[x+6].G, rowIn[x+5].G, rowIn[x+4].G, rowIn[x+3].G, rowIn[x+2].G, rowIn[x+1].G, rowIn[x].G);
            __m256 fR = _mm256_set_ps(rowIn[x+7].R, rowIn[x+6].R, rowIn[x+5].R, rowIn[x+4].R, rowIn[x+3].R, rowIn[x+2].R, rowIn[x+1].R, rowIn[x].R);

            // Scale up to [0.0f, 255.0f] range
            fB = _mm256_mul_ps(fB, vScale);
            fG = _mm256_mul_ps(fG, vScale);
            fR = _mm256_mul_ps(fR, vScale);

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
            float b = rowIn[x].B * 255.0f;
            float g = rowIn[x].G * 255.0f;
            float r = rowIn[x].R * 255.0f;

            rowY[x] = (r + g + b) * 0.57735027f;
            rowU[x] = (r - b) * 0.70710678f;
            rowV[x] = (r + b - 2.0f * g) * 0.40824829f;
        }
    }
}