#include <algorithm>
#include "AVX2_AlgoColorConvert.hpp"

// =========================================================
// AVX2 ACCELERATED YUV PLANAR TO BGRA INTERLEAVED
// =========================================================
void AVX2_Convert_YUV_to_BGRA_8u
(
    const float* RESTRICT pY,               // Y Orthonormal plan
    const float* RESTRICT pU,               // U Orthonormal plan
    const float* RESTRICT pV,               // V Orthonormal plan
    const PF_Pixel_BGRA_8u* RESTRICT pInput,// BGRA_8u input pixel (used for grab alpha channel values only) 
    PF_Pixel_BGRA_8u* RESTRICT pOutput,     // BGRA_8u denoised output image
    int32_t w,                              // horizontal frame size in pixels
    int32_t h,                              // vertical frame size in lines
    int32_t src_pitch,                      // input buffer (pInput) line pitch in pixels
    int32_t dst_pitch                       // output buffer (pOutput) line pitch in pixels
)
{
    // Transformation Constants
    const __m256 v_c1 = _mm256_set1_ps(0.57735027f); // k1_sqrt3
    const __m256 v_c2 = _mm256_set1_ps(0.70710678f); // k1_sqrt2
    const __m256 v_c3 = _mm256_set1_ps(0.40824829f); // k1_sqrt6
    const __m256 v_c4 = _mm256_set1_ps(0.81649658f); // 2.0f * k1_sqrt6

    // Clamping and Rounding Constants
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255  = _mm256_set1_ps(255.0f);
    const __m256 v_half = _mm256_set1_ps(0.5f);

    // Alpha mask: Keeps the highest 8 bits of a 32-bit integer (0xFF000000)
    const __m256i alpha_mask = _mm256_set1_epi32(0xFF000000); 

    for (int32_t y = 0; y < h; ++y)
    {
        // YUV buffers are tightly packed, so pitch == width
        const float* pY_row = pY + y * w;
        const float* pU_row = pU + y * w;
        const float* pV_row = pV + y * w;
        
        // Host buffers use the provided pitches
        const PF_Pixel_BGRA_8u* pIn_row = pInput + y * src_pitch;
        PF_Pixel_BGRA_8u* pOut_row = pOutput + y * dst_pitch;

        int32_t x = 0;
        const int32_t ww = w - 8;
        
        // Process 8 pixels simultaneously
        for (; x <= ww; x += 8)
        {
            // 1. Load Planar Floats
            __m256 Y = _mm256_loadu_ps(pY_row + x);
            __m256 U = _mm256_loadu_ps(pU_row + x);
            __m256 V = _mm256_loadu_ps(pV_row + x);

            // 2. Optimized FMA Math
            __m256 vY = _mm256_mul_ps(Y, v_c1);
            __m256 vU_c2 = _mm256_mul_ps(U, v_c2);
            __m256 vV_c3 = _mm256_mul_ps(V, v_c3);

            __m256 vY_plus_V_c3 = _mm256_add_ps(vY, vV_c3);

            __m256 R_f = _mm256_add_ps(vY_plus_V_c3, vU_c2);
            __m256 B_f = _mm256_sub_ps(vY_plus_V_c3, vU_c2);
            __m256 G_f = _mm256_fnmadd_ps(V, v_c4, vY); // Equivalent to: vY - (V * c4)

            // 3. Clamp floats to [0, 255] and add 0.5f for rounding
            R_f = _mm256_add_ps(_mm256_min_ps(v_255, _mm256_max_ps(v_zero, R_f)), v_half);
            G_f = _mm256_add_ps(_mm256_min_ps(v_255, _mm256_max_ps(v_zero, G_f)), v_half);
            B_f = _mm256_add_ps(_mm256_min_ps(v_255, _mm256_max_ps(v_zero, B_f)), v_half);

            // 4. Convert to 32-bit Integers (Truncation matches the standard C++ cast)
            __m256i R_i = _mm256_cvttps_epi32(R_f);
            __m256i G_i = _mm256_cvttps_epi32(G_f);
            __m256i B_i = _mm256_cvttps_epi32(B_f);

            // 5. Load Original Interleaved Host Pixels (8 pixels = 32 bytes)
            __m256i in_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pIn_row + x));
            
            // Isolate the Alpha channel (Zeroes out the original RGB)
            __m256i A_i = _mm256_and_si256(in_pixels, alpha_mask);

            // 6. Shift RGB into BGRA memory positions (Little-Endian layout: B G R A)
            // B remains in bits [7:0]
            G_i = _mm256_slli_epi32(G_i, 8);  // Shift G to bits [15:8]
            R_i = _mm256_slli_epi32(R_i, 16); // Shift R to bits [23:16]

            // 7. Combine channels using Bitwise OR
            __m256i out_pixels = _mm256_or_si256(A_i, _mm256_or_si256(R_i, _mm256_or_si256(G_i, B_i)));

            // 8. Store 8 finished pixels back to the interleaved output buffer
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(pOut_row + x), out_pixels);
        }

        // =========================================================
        // SCALAR TAIL (For dimensions not divisible by 8)
        // =========================================================
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

            PF_Pixel_BGRA_8u* pOut = pOut_row + x;
            const PF_Pixel_BGRA_8u* pIn = pIn_row + x;

            pOut->R = clamp8(r);
            pOut->G = clamp8(g);
            pOut->B = clamp8(b);
            pOut->A = pIn->A; // Preserve exact alpha from input
        }
    }
	
	return;
}