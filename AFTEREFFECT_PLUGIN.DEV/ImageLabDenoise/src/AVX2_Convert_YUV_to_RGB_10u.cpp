#include <algorithm>
#include <immintrin.h>
#include "ColorConvert.hpp"


void AVX2_Convert_YUV_to_RGB_10u
(
    const float* RESTRICT pY,                // Y plane [Orthonormal format]
    const float* RESTRICT pU,                // U plane [Orthonormal format]
    const float* RESTRICT pV,                // V plane [Orthonormal format]
    PF_Pixel_RGB_10u* RESTRICT pOutput,      // RGB_10u denoised output (No pInput needed!)
    int32_t w,
    int32_t h,
    int32_t dst_pitch                        // Row Pitch in Pixels
) noexcept
{
    const __m256 v_c1 = _mm256_set1_ps(0.57735027f);
    const __m256 v_c2 = _mm256_set1_ps(0.70710678f);
    const __m256 v_c3 = _mm256_set1_ps(0.40824829f);
    const __m256 v_c4 = _mm256_set1_ps(0.81649658f);

    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_max   = _mm256_set1_ps(1023.0f); // 10-bit max
    const __m256 v_half  = _mm256_set1_ps(0.5f);
    
    // Scale factor to map algorithm [0.0f, 255.0f] back up to Adobe [0, 1023]
    const __m256 v_scale = _mm256_set1_ps(4.0117647f); 

    const int32_t vecSize = 8;
    const int32_t endX = (w / vecSize) * vecSize;

    for (int32_t y = 0; y < h; ++y)
    {
        const float* pY_row = pY + (y * w);
        const float* pU_row = pU + (y * w);
        const float* pV_row = pV + (y * w);
        
        PF_Pixel_RGB_10u* pOut_row = pOutput + (y * dst_pitch);

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

            // Scale back up to 10-bit range
            R_f = _mm256_mul_ps(R_f, v_scale);
            G_f = _mm256_mul_ps(G_f, v_scale);
            B_f = _mm256_mul_ps(B_f, v_scale);

            // Clamp to [0, 1023] and add 0.5f for accurate rounding
            R_f = _mm256_add_ps(_mm256_min_ps(v_max, _mm256_max_ps(v_zero, R_f)), v_half);
            G_f = _mm256_add_ps(_mm256_min_ps(v_max, _mm256_max_ps(v_zero, G_f)), v_half);
            B_f = _mm256_add_ps(_mm256_min_ps(v_max, _mm256_max_ps(v_zero, B_f)), v_half);

            __m256i R_i = _mm256_cvttps_epi32(R_f);
            __m256i G_i = _mm256_cvttps_epi32(G_f);
            __m256i B_i = _mm256_cvttps_epi32(B_f);

            // Shift integers back into the 32-bit packed layout: 
            // B (shift 2), G (shift 12), R (shift 22)
            B_i = _mm256_slli_epi32(B_i, 2);
            G_i = _mm256_slli_epi32(G_i, 12);
            R_i = _mm256_slli_epi32(R_i, 22);

            // Combine the bitfields. 
            // (Padding bits 0-1 remain naturally 0).
            __m256i packed_out = _mm256_or_si256(R_i, _mm256_or_si256(G_i, B_i));

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(pOut_row + x), packed_out);
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

            auto scale_clamp10 = [](float val) noexcept
            {
                return static_cast<A_u_long>(std::max(0.0f, std::min(1023.0f, (val * 4.0117647f) + 0.5f)));
            };

            PF_Pixel_RGB_10u* pOut = pOut_row + x;

            pOut->_pad_ = 0;
            pOut->R = scale_clamp10(r);
            pOut->G = scale_clamp10(g);
            pOut->B = scale_clamp10(b);
        }
    }
}