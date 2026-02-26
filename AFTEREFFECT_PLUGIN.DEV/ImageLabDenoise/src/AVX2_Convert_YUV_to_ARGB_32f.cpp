#include <algorithm>
#include <immintrin.h>
#include "ColorConvert.hpp"

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
) noexcept
{
    const __m256 v_c1 = _mm256_set1_ps(0.57735027f);
    const __m256 v_c2 = _mm256_set1_ps(0.70710678f);
    const __m256 v_c3 = _mm256_set1_ps(0.40824829f);
    const __m256 v_c4 = _mm256_set1_ps(0.81649658f);

    // Scale factor to map algorithm [0.0f, 255.0f] back down to Adobe [0.0f, 1.0f]
    const __m256 v_scaleInv = _mm256_set1_ps(0.00392156862f); // 1.0f / 255.0f

    const int32_t vecSize = 8;
    const int32_t endX = (w / vecSize) * vecSize;

    for (int32_t y = 0; y < h; ++y)
    {
        const float* pY_row = pY + (y * w);
        const float* pU_row = pU + (y * w);
        const float* pV_row = pV + (y * w);
        
        const PF_Pixel_ARGB_32f* pIn_row = pInput + (y * src_pitch);
        PF_Pixel_ARGB_32f* pOut_row = pOutput + (y * dst_pitch);

        int32_t x = 0;
        
        // =========================================================
        // MAIN AVX2 KERNEL (8 Pixels per loop)
        // =========================================================
        for (; x < endX; x += vecSize)
        {
            __m256 Y = _mm256_loadu_ps(pY_row + x);
            __m256 U = _mm256_loadu_ps(pU_row + x);
            __m256 V = _mm256_loadu_ps(pV_row + x);

            // Orthonormal Math
            __m256 vY = _mm256_mul_ps(Y, v_c1);
            __m256 vU_c2 = _mm256_mul_ps(U, v_c2);
            __m256 vV_c3 = _mm256_mul_ps(V, v_c3);

            __m256 vY_plus_V_c3 = _mm256_add_ps(vY, vV_c3);

            __m256 R_f = _mm256_add_ps(vY_plus_V_c3, vU_c2);
            __m256 B_f = _mm256_sub_ps(vY_plus_V_c3, vU_c2);
            __m256 G_f = _mm256_fnmadd_ps(V, v_c4, vY); 

            // Scale back down to Adobe 32-bit float range
            // NOTE: We omit min/max clamping to allow HDR overbrights (> 1.0f) to pass through!
            R_f = _mm256_mul_ps(R_f, v_scaleInv);
            G_f = _mm256_mul_ps(G_f, v_scaleInv);
            B_f = _mm256_mul_ps(B_f, v_scaleInv);

            // Staging arrays using the CPU's native write buffer to safely scatter
            // the 32-bit registers back into the 128-bit ARGB_32f struct layout
            alignas(32) float r_arr[8], g_arr[8], b_arr[8];
            _mm256_store_ps(r_arr, R_f);
            _mm256_store_ps(g_arr, G_f);
            _mm256_store_ps(b_arr, B_f);

            for (int32_t i = 0; i < 8; ++i) 
            {
                pOut_row[x+i].A = pIn_row[x+i].A; // Preserve exact original Alpha float
                pOut_row[x+i].R = r_arr[i];
                pOut_row[x+i].G = g_arr[i];
                pOut_row[x+i].B = b_arr[i];
            }
        }

        // =========================================================
        // SCALAR TAIL
        // =========================================================
        for (; x < w; ++x)
        {
            float Y = pY_row[x];
            float U = pU_row[x];
            float V = pV_row[x];

            float r = Y * 0.57735027f + U * 0.70710678f + V * 0.40824829f;
            float g = Y * 0.57735027f - V * 0.81649658f;
            float b = Y * 0.57735027f - U * 0.70710678f + V * 0.40824829f;

            PF_Pixel_ARGB_32f* pOut = pOut_row + x;
            const PF_Pixel_ARGB_32f* pIn = pIn_row + x;

            pOut->A = pIn->A; 
            pOut->R = r * 0.00392156862f;
            pOut->G = g * 0.00392156862f;
            pOut->B = b * 0.00392156862f;
        }
    }
}