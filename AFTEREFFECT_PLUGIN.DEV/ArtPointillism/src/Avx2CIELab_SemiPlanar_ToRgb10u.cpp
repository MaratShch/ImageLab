#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "Common.hpp"
#include "Avx2ColorConverts.hpp"


/**
 * AVX2 OUTPUT CONVERTER: Semi-Planar Lab -> RGB_10u (Packed 10-bit)
 * 
 * - Converts Lab -> Linear RGB.
 * - Scales to 0..1023 (Linear).
 * - Packs into 32-bit integer: (R<<22) | (G<<12) | (B<<2).
 * - No Alpha source required.
 */
void AVX2_ConvertCIELab_SemiPlanar_To_Rgb
(
    const float*      RESTRICT pL,        // source L (planar)
    const float*      RESTRICT pAB,       // source ab (interleaved)
    PF_Pixel_RGB_10u* RESTRICT pDst,      // destination buffer
    int32_t           sizeX,              // width
    int32_t           sizeY,              // height
    int32_t           dstPitch            // pitch of pDst
) noexcept
{
    const int32_t dstPitchBytes = dstPitch * sizeof(PF_Pixel_RGB_10u);

    // Constants for 10-bit scaling
    const __m256 v_scale = _mm256_set1_ps(1023.0f);
    const __m256 v_half  = _mm256_set1_ps(0.5f);
    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_max   = _mm256_set1_ps(1023.0f);

    uint8_t* pRowDstRaw = reinterpret_cast<uint8_t*>(pDst);

    for (int y = 0; y < sizeY; ++y)
    {
        // Row Pointers
        PF_Pixel_RGB_10u* rowDst = reinterpret_cast<PF_Pixel_RGB_10u*>(pRowDstRaw + (y * dstPitchBytes));
        
        const float* rowL  = pL + (y * sizeX);
        const float* rowAB = pAB + (y * sizeX * 2);

        int x = 0;

        // --- AVX2 LOOP (8 pixels) ---
        for (; x <= sizeX - 8; x += 8)
        {
            // 1. LOAD LAB
            __m256 L = _mm256_loadu_ps(rowL + x);
            __m256 ab0 = _mm256_loadu_ps(rowAB + x * 2);
            __m256 ab1 = _mm256_loadu_ps(rowAB + x * 2 + 8);

            // De-interleave AB
            const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256 s0 = _mm256_permutevar8x32_ps(ab0, perm_idx);
            __m256 s1 = _mm256_permutevar8x32_ps(ab1, perm_idx);
            __m256 a = _mm256_permute2f128_ps(s0, s1, 0x20);
            __m256 b = _mm256_permute2f128_ps(s0, s1, 0x31);

            // 2. CONVERT Lab -> Linear RGB
            __m256 R, G, B;
            // Assumes this inline helper is available from previous code
            AVX2_Lab_to_RGB_Linear_Inline(L, a, b, R, G, B);

            // 3. SCALE & CLAMP (0..1023)
            // Linear scaling without Gamma
            R = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(R, v_scale, v_half)));
            G = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(G, v_scale, v_half)));
            B = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(B, v_scale, v_half)));

            // 4. CONVERT TO INT32
            __m256i iR = _mm256_cvttps_epi32(R);
            __m256i iG = _mm256_cvttps_epi32(G);
            __m256i iB = _mm256_cvttps_epi32(B);

            // 5. PACKING 10-BIT
            // Structure: Pad:2, B:10, G:10, R:10
            // Shifts: B<<2, G<<12, R<<22
            
            __m256i out = _mm256_slli_epi32(iB, 2);              // B at 2-11
            out = _mm256_or_si256(out, _mm256_slli_epi32(iG, 12)); // G at 12-21
            out = _mm256_or_si256(out, _mm256_slli_epi32(iR, 22)); // R at 22-31
            
            // Note: Bits 0-1 (Pad) are left as 0 by the shift logic

            // 6. STORE
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(rowDst + x), out);
        }

        // --- SCALAR FALLBACK ---
        for (; x < sizeX; ++x)
        {
            float l_val = rowL[x];
            float a_val = rowAB[x * 2];
            float b_val = rowAB[x * 2 + 1];

            // Manual Scalar Lab->RGB (Linear)
            float fy = (l_val + 16.0f) / 116.0f;
            float fx = fy + (a_val / 500.0f);
            float fz = fy - (b_val / 200.0f);
			
            auto f_inv = [](float t) noexcept
			{ 
				return (t > 0.206893f) ? (t * t * t) : ((t - 16.0f/116.0f) / 7.787f);
			};
            
            float X = 0.95047f * f_inv(fx);
            float Y = 1.00000f * ((l_val > 8.0f) ? (fy*fy*fy) : (l_val / 903.3f));
            float Z = 1.08883f * f_inv(fz);

            float r_lin =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
            float g_lin = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
            float b_lin =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;

            // Clamp 0..1023
            uint32_t uR = (uint32_t)std::min(std::max(r_lin * 1023.0f + 0.5f, 0.0f), 1023.0f);
            uint32_t uG = (uint32_t)std::min(std::max(g_lin * 1023.0f + 0.5f, 0.0f), 1023.0f);
            uint32_t uB = (uint32_t)std::min(std::max(b_lin * 1023.0f + 0.5f, 0.0f), 1023.0f);

            rowDst[x].R = uR;
            rowDst[x].G = uG;
            rowDst[x].B = uB;
        }
    }
	
	return;
}