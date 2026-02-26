#include <algorithm>
#include <immintrin.h>
#include "ColorConvert.hpp"

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
) noexcept
{
    // 1. PRE-CALCULATE DIRECT INVERSE MATRIX COEFFICIENTS
    const float N1_base = 0.577350f; 
    float N2_base, N3_base, N4_base, N5_base, N6_base, N7_base;

    if (isBT709) 
    {
        N2_base = 0.099275f; N3_base = -0.467683f;
        N4_base = -0.434569f; N5_base = 0.472051f;
        N6_base = 0.385973f; N7_base = 0.556223f;
    } 
    else 
    {
        N2_base = 0.130816f; N3_base = -0.310678f;
        N4_base = -0.472866f; N5_base = 0.405713f;
        N6_base = 0.411049f; N7_base = 0.512784f;
    }

    // Pre-multiply by (1.0f / 255.0f) to map Oracle [0.0, 255.0] back to Adobe [0.0, 1.0]
    const float invScale = 0.00392156862f;
    const __m256 vN1 = _mm256_set1_ps(N1_base * invScale);
    const __m256 vN2 = _mm256_set1_ps(N2_base * invScale);
    const __m256 vN3 = _mm256_set1_ps(N3_base * invScale);
    const __m256 vN4 = _mm256_set1_ps(N4_base * invScale);
    const __m256 vN5 = _mm256_set1_ps(N5_base * invScale);
    const __m256 vN6 = _mm256_set1_ps(N6_base * invScale);
    const __m256 vN7 = _mm256_set1_ps(N7_base * invScale);

    const int32_t vecSize = 8;
    const int32_t endX = (w / vecSize) * vecSize;

    for (int32_t y = 0; y < h; ++y)
    {
        const float* pY_row = pY + (y * w);
        const float* pU_row = pU + (y * w);
        const float* pV_row = pV + (y * w);
        
        const PF_Pixel_VUYA_32f* pIn_row = pInput + (y * src_pitch);
        PF_Pixel_VUYA_32f* pOut_row = pOutput + (y * dst_pitch);

        int32_t x = 0;
        
        // =========================================================
        // MAIN AVX2 KERNEL (8 Pixels per loop)
        // =========================================================
        for (; x < endX; x += vecSize)
        {
            __m256 fY = _mm256_loadu_ps(pY_row + x);
            __m256 fU = _mm256_loadu_ps(pU_row + x);
            __m256 fV = _mm256_loadu_ps(pV_row + x);

            // Fast Inverse FMA Math
            // The (1.0/255.0) scaling is baked in, and NO CLAMPING allows HDR overbrights!
            __m256 resY = _mm256_fmadd_ps(vN1, fY, _mm256_fmadd_ps(vN2, fU, _mm256_mul_ps(vN3, fV)));
            __m256 resU = _mm256_fmadd_ps(vN4, fU, _mm256_mul_ps(vN5, fV));
            __m256 resV = _mm256_fmadd_ps(vN6, fU, _mm256_mul_ps(vN7, fV));

            // Staging arrays for scattering back into the interleaved struct
            alignas(32) float y_arr[8], u_arr[8], v_arr[8];
            _mm256_store_ps(y_arr, resY);
            _mm256_store_ps(u_arr, resU);
            _mm256_store_ps(v_arr, resV);

            for (int32_t i = 0; i < 8; ++i) 
            {
                pOut_row[x+i].V = v_arr[i];
                pOut_row[x+i].U = u_arr[i];
                pOut_row[x+i].Y = y_arr[i];
                pOut_row[x+i].A = pIn_row[x+i].A; // Preserve exact Alpha float
            }
        }

        // =========================================================
        // SCALAR TAIL
        // =========================================================
        for (; x < w; ++x)
        {
            float y_val = pY_row[x];
            float u_val = pU_row[x];
            float v_val = pV_row[x];

            float r_y = (N1_base * invScale) * y_val + (N2_base * invScale) * u_val + (N3_base * invScale) * v_val;
            float r_u = (N4_base * invScale) * u_val + (N5_base * invScale) * v_val;
            float r_v = (N6_base * invScale) * u_val + (N7_base * invScale) * v_val;

            PF_Pixel_VUYA_32f* pOut = pOut_row + x;
            const PF_Pixel_VUYA_32f* pIn = pIn_row + x;

            pOut->V = r_v;
            pOut->U = r_u;
            pOut->Y = r_y;
            pOut->A = pIn->A; 
        }
    }
}