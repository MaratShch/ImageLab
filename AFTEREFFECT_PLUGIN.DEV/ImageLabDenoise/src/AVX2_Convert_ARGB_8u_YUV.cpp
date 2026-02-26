#include <algorithm>
#include <immintrin.h>
#include "ColorConvert.hpp"

void AVX2_Convert_ARGB_8u_YUV
(
    const PF_Pixel_ARGB_8u* RESTRICT pInput, // Input ARGB_8u (Interleaved)
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

    // 2. Prepare Shuffle Masks for ARGB (A=0, R=1, G=2, B=3)
    const __m256i maskR = _mm256_set_epi8(
        -1, -1, -1, 13, -1, -1, -1, 9, -1, -1, -1, 5, -1, -1, -1, 1,
        -1, -1, -1, 13, -1, -1, -1, 9, -1, -1, -1, 5, -1, -1, -1, 1
    );
    const __m256i maskG = _mm256_set_epi8(
        -1, -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2,
        -1, -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2
    );
    const __m256i maskB = _mm256_set_epi8(
        -1, -1, -1, 15, -1, -1, -1, 11, -1, -1, -1, 7, -1, -1, -1, 3,
        -1, -1, -1, 15, -1, -1, -1, 11, -1, -1, -1, 7, -1, -1, -1, 3
    );

    const int32_t vecSize = 8;
    const int32_t endX = (sizeX / vecSize) * vecSize;

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const PF_Pixel_ARGB_8u* rowIn = pInput + (y * linePitch);
        float* rowY = Y_out + (y * sizeX);
        float* rowU = U_out + (y * sizeX);
        float* rowV = V_out + (y * sizeX);

        int32_t x = 0;

        // MAIN AVX2 KERNEL
        for (; x < endX; x += vecSize)
        {
            __m256i argb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowIn + x));

            __m256 fR = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(argb, maskR));
            __m256 fG = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(argb, maskG));
            __m256 fB = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(argb, maskB));

            __m256 resY = _mm256_mul_ps(_mm256_add_ps(fR, _mm256_add_ps(fG, fB)), vSqrt3_inv);
            __m256 resU = _mm256_mul_ps(_mm256_sub_ps(fR, fB), vSqrt2_inv);
            __m256 twoG = _mm256_mul_ps(fG, vTwo);
            __m256 resV = _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(fR, fB), twoG), vSqrt6_inv);

            _mm256_storeu_ps(rowY + x, resY);
            _mm256_storeu_ps(rowU + x, resU);
            _mm256_storeu_ps(rowV + x, resV);
        }

        // SCALAR TAIL
        for (; x < sizeX; ++x)
        {
            float r = static_cast<float>(rowIn[x].R);
            float g = static_cast<float>(rowIn[x].G);
            float b = static_cast<float>(rowIn[x].B);

            rowY[x] = (r + g + b) * 0.57735027f;
            rowU[x] = (r - b) * 0.70710678f;
            rowV[x] = (r + b - 2.0f * g) * 0.40824829f;
        }
    }
}