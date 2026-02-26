#include <algorithm>
#include <immintrin.h>
#include "ColorCOnvert.hpp"


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
) noexcept
{
    // 1. PRE-CALCULATE DIRECT MATRIX COEFFICIENTS
    const float M1_base = 1.73205081f; 
    float M2_base, M3_base, M4_base, M5_base, M6_base, M7_base;

    if (isBT709) 
    {
        M2_base = 0.963177f; M3_base = 0.638942f;
        M4_base = -1.312107f; M5_base = 1.113551f;
        M6_base = 0.910495f; M7_base = 1.025131f;
    } 
    else 
    {
        M2_base = 0.824377f; M3_base = 0.397140f;
        M4_base = -1.252993f; M5_base = 0.991364f;
        M6_base = 1.004402f; M7_base = 1.155454f;
    }

    // Pre-multiply by 255.0f to map Adobe [0.0, 1.0] to Oracle [0.0, 255.0]
    // This saves us from doing a scale multiplication inside the loop!
    const __m256 vM1 = _mm256_set1_ps(M1_base * 255.0f);
    const __m256 vM2 = _mm256_set1_ps(M2_base * 255.0f);
    const __m256 vM3 = _mm256_set1_ps(M3_base * 255.0f);
    const __m256 vM4 = _mm256_set1_ps(M4_base * 255.0f);
    const __m256 vM5 = _mm256_set1_ps(M5_base * 255.0f);
    const __m256 vM6 = _mm256_set1_ps(M6_base * 255.0f);
    const __m256 vM7 = _mm256_set1_ps(M7_base * 255.0f);

    const int32_t vecSize = 8;
    const int32_t endX = (sizeX / vecSize) * vecSize;

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const PF_Pixel_VUYA_32f* rowIn = pInput + (y * linePitch);
        float* rowY = Y_out + (y * sizeX);
        float* rowU = U_out + (y * sizeX);
        float* rowV = V_out + (y * sizeX);

        int32_t x = 0;

        // =========================================================
        // MAIN AVX2 KERNEL (8 Pixels per loop)
        // =========================================================
        for (; x < endX; x += vecSize)
        {
            // Gather 32-bit floats directly. Layout: V, U, Y, A
            __m256 fY = _mm256_set_ps(rowIn[x+7].Y, rowIn[x+6].Y, rowIn[x+5].Y, rowIn[x+4].Y, rowIn[x+3].Y, rowIn[x+2].Y, rowIn[x+1].Y, rowIn[x].Y);
            __m256 fU = _mm256_set_ps(rowIn[x+7].U, rowIn[x+6].U, rowIn[x+5].U, rowIn[x+4].U, rowIn[x+3].U, rowIn[x+2].U, rowIn[x+1].U, rowIn[x].U);
            __m256 fV = _mm256_set_ps(rowIn[x+7].V, rowIn[x+6].V, rowIn[x+5].V, rowIn[x+4].V, rowIn[x+3].V, rowIn[x+2].V, rowIn[x+1].V, rowIn[x].V);

            // Fast Direct FMA Math (Y cancels out for U and V)
            // The 255.0 scaling is already baked into the vM coefficients!
            __m256 resY = _mm256_fmadd_ps(vM1, fY, _mm256_fmadd_ps(vM2, fU, _mm256_mul_ps(vM3, fV)));
            __m256 resU = _mm256_fmadd_ps(vM4, fU, _mm256_mul_ps(vM5, fV));
            __m256 resV = _mm256_fmadd_ps(vM6, fU, _mm256_mul_ps(vM7, fV));

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
            float y_val = rowIn[x].Y;
            float u_val = rowIn[x].U; // No 128 shift required for 32f!
            float v_val = rowIn[x].V;

            rowY[x] = (M1_base * 255.0f) * y_val + (M2_base * 255.0f) * u_val + (M3_base * 255.0f) * v_val;
            rowU[x] = (M4_base * 255.0f) * u_val + (M5_base * 255.0f) * v_val;
            rowV[x] = (M6_base * 255.0f) * u_val + (M7_base * 255.0f) * v_val;
        }
    }
}