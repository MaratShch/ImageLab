#include <algorithm>
#include <immintrin.h>
#include "ColorConvert.hpp"

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
) noexcept
{
    const __m256 v_c1 = _mm256_set1_ps(0.57735027f);
    const __m256 v_c2 = _mm256_set1_ps(0.70710678f);
    const __m256 v_c3 = _mm256_set1_ps(0.40824829f);
    const __m256 v_c4 = _mm256_set1_ps(0.81649658f);

    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_max   = _mm256_set1_ps(32768.0f); // Adobe 16-bit max
    const __m256 v_half  = _mm256_set1_ps(0.5f);
    
    // Scale factor to map algorithm [0.0f, 255.0f] back up to Adobe [0, 32768]
    const __m256 v_scale = _mm256_set1_ps(128.50196f); // 32768.0f / 255.0f

    const int32_t vecSize = 8;
    const int32_t endX = (w / vecSize) * vecSize;

    for (int32_t y = 0; y < h; ++y)
    {
        const float* pY_row = pY + (y * w);
        const float* pU_row = pU + (y * w);
        const float* pV_row = pV + (y * w);
        
        const PF_Pixel_BGRA_16u* pIn_row = pInput + (y * src_pitch);
        PF_Pixel_BGRA_16u* pOut_row = pOutput + (y * dst_pitch);

        int32_t x = 0;
        
        // =========================================================
        // MAIN AVX2 KERNEL (8 Pixels per loop)
        // =========================================================
        for (; x < endX; x += vecSize)
        {
            __m256 Y = _mm256_loadu_ps(pY_row + x);
            __m256 U = _mm256_loadu_ps(pU_row + x);
            __m256 V = _mm256_loadu_ps(pV_row + x);

            // Orthonormal Math (Generates values in 0.0 to 255.0 range)
            __m256 vY = _mm256_mul_ps(Y, v_c1);
            __m256 vU_c2 = _mm256_mul_ps(U, v_c2);
            __m256 vV_c3 = _mm256_mul_ps(V, v_c3);

            __m256 vY_plus_V_c3 = _mm256_add_ps(vY, vV_c3);

            __m256 R_f = _mm256_add_ps(vY_plus_V_c3, vU_c2);
            __m256 B_f = _mm256_sub_ps(vY_plus_V_c3, vU_c2);
            __m256 G_f = _mm256_fnmadd_ps(V, v_c4, vY); 

            // Scale back up to Adobe 16-bit range
            R_f = _mm256_mul_ps(R_f, v_scale);
            G_f = _mm256_mul_ps(G_f, v_scale);
            B_f = _mm256_mul_ps(B_f, v_scale);

            // Clamp to [0, 32768] and add 0.5f for accurate rounding
            R_f = _mm256_add_ps(_mm256_min_ps(v_max, _mm256_max_ps(v_zero, R_f)), v_half);
            G_f = _mm256_add_ps(_mm256_min_ps(v_max, _mm256_max_ps(v_zero, G_f)), v_half);
            B_f = _mm256_add_ps(_mm256_min_ps(v_max, _mm256_max_ps(v_zero, B_f)), v_half);

            __m256i R_i = _mm256_cvttps_epi32(R_f);
            __m256i G_i = _mm256_cvttps_epi32(G_f);
            __m256i B_i = _mm256_cvttps_epi32(B_f);

            // Staging arrays using the CPU's native write buffer to cleanly 
            // scatter the 32-bit registers back into the 16-bit interleaved struct
            alignas(32) int32_t r_arr[8], g_arr[8], b_arr[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(r_arr), R_i);
            _mm256_store_si256(reinterpret_cast<__m256i*>(g_arr), G_i);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_arr), B_i);

            for (int32_t i = 0; i < 8; ++i) 
            {
                pOut_row[x+i].B = static_cast<A_u_short>(b_arr[i]);
                pOut_row[x+i].G = static_cast<A_u_short>(g_arr[i]);
                pOut_row[x+i].R = static_cast<A_u_short>(r_arr[i]);
                pOut_row[x+i].A = pIn_row[x+i].A; // Preserve exact Alpha
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

            // Scale and Clamp
            auto scale_clamp16 = [](float val) noexcept
            {
                return static_cast<A_u_short>(std::max(0.0f, std::min(32768.0f, (val * 128.50196f) + 0.5f)));
            };

            PF_Pixel_BGRA_16u* pOut = pOut_row + x;
            const PF_Pixel_BGRA_16u* pIn = pIn_row + x;

            pOut->B = scale_clamp16(b);
            pOut->G = scale_clamp16(g);
            pOut->R = scale_clamp16(r);
            pOut->A = pIn->A; 
        }
    }
}