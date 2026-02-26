#include <algorithm>
#include <immintrin.h>
#include "ColorConvert.hpp"


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
) noexcept
{
    const __m256 v_c1 = _mm256_set1_ps(0.57735027f);
    const __m256 v_c2 = _mm256_set1_ps(0.70710678f);
    const __m256 v_c3 = _mm256_set1_ps(0.40824829f);
    const __m256 v_c4 = _mm256_set1_ps(0.81649658f);

    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255  = _mm256_set1_ps(255.0f);
    const __m256 v_half = _mm256_set1_ps(0.5f);

    // Alpha mask: Isolates the LOWEST 8 bits (Byte 0: A)
    const __m256i alpha_mask = _mm256_set1_epi32(0x000000FF); 

    const int32_t vecSize = 8;
    const int32_t endX = (w / vecSize) * vecSize;

    for (int32_t y = 0; y < h; ++y)
    {
        const float* pY_row = pY + y * w;
        const float* pU_row = pU + y * w;
        const float* pV_row = pV + y * w;
        
        const PF_Pixel_ARGB_8u* pIn_row = pInput + y * src_pitch;
              PF_Pixel_ARGB_8u* pOut_row = pOutput + y * dst_pitch;

        int32_t x = 0;
        
        // MAIN AVX2 KERNEL
        for (; x < endX; x += vecSize)
        {
            __m256 Y = _mm256_loadu_ps(pY_row + x);
            __m256 U = _mm256_loadu_ps(pU_row + x);
            __m256 V = _mm256_loadu_ps(pV_row + x);

            __m256 vY = _mm256_mul_ps(Y, v_c1);
            __m256 vU_c2 = _mm256_mul_ps(U, v_c2);
            __m256 vV_c3 = _mm256_mul_ps(V, v_c3);

            __m256 vY_plus_V_c3 = _mm256_add_ps(vY, vV_c3);

            __m256 R_f = _mm256_add_ps(vY_plus_V_c3, vU_c2);
            __m256 B_f = _mm256_sub_ps(vY_plus_V_c3, vU_c2);
            __m256 G_f = _mm256_fnmadd_ps(V, v_c4, vY); 

            R_f = _mm256_add_ps(_mm256_min_ps(v_255, _mm256_max_ps(v_zero, R_f)), v_half);
            G_f = _mm256_add_ps(_mm256_min_ps(v_255, _mm256_max_ps(v_zero, G_f)), v_half);
            B_f = _mm256_add_ps(_mm256_min_ps(v_255, _mm256_max_ps(v_zero, B_f)), v_half);

            __m256i R_i = _mm256_cvttps_epi32(R_f);
            __m256i G_i = _mm256_cvttps_epi32(G_f);
            __m256i B_i = _mm256_cvttps_epi32(B_f);

            __m256i in_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pIn_row + x));
            
            // Extract the original A byte
            __m256i A_i = _mm256_and_si256(in_pixels, alpha_mask);

            // Shift RGB into ARGB memory layout (A=0, R=1, G=2, B=3)
            R_i = _mm256_slli_epi32(R_i, 8);
            G_i = _mm256_slli_epi32(G_i, 16);
            B_i = _mm256_slli_epi32(B_i, 24);

            __m256i out_pixels = _mm256_or_si256(A_i, _mm256_or_si256(R_i, _mm256_or_si256(G_i, B_i)));

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(pOut_row + x), out_pixels);
        }

        // SCALAR TAIL
        for (; x < w; ++x)
        {
            float Y = pY_row[x];
            float U = pU_row[x];
            float V = pV_row[x];

            float r = Y * 0.57735027f + U * 0.70710678f + V * 0.40824829f;
            float g = Y * 0.57735027f - V * 0.81649658f;
            float b = Y * 0.57735027f - U * 0.70710678f + V * 0.40824829f;

            auto clamp8 = [](float val) noexcept
            {
                return static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val + 0.5f)));
            };

            PF_Pixel_ARGB_8u* pOut = pOut_row + x;
            const PF_Pixel_ARGB_8u* pIn = pIn_row + x;

            pOut->R = clamp8(r);
            pOut->G = clamp8(g);
            pOut->B = clamp8(b);
            pOut->A = pIn->A; 
        }
    }
}