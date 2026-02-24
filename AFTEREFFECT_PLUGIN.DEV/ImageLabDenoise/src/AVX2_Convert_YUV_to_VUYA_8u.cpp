#include "ColorConvert.hpp"

// Matches your provided prototype exactly
void AVX2_Convert_YUV_to_VUYA_8u
(
    const float* RESTRICT pY,               // Y plan [Orthonormal format]
    const float* RESTRICT pU,               // U plan [Orthonormal format]
    const float* RESTRICT pV,               // V plan [Orthonormal format]  
    const PF_Pixel_VUYA_8u* RESTRICT pInput,// VUYA_8u input pixel (Alpha source)
    PF_Pixel_VUYA_8u* RESTRICT pOutput,     // VUYA_8u denoised output image
    int32_t w,                              // horizontal frame size in pixels
    int32_t h,                              // vertical frame size in lines
    int32_t src_pitch,                      // input buffer line pitch in pixels
    int32_t dst_pitch,                      // output buffer line pitch in pixels
    bool isBT709                            // true for BT709, false for BT601
) noexcept
{
    // 1. PRE-CALCULATE DIRECT INVERSE MATRIX COEFFICIENTS (Branch outside the loop)
    // N1 = 1.0f / sqrt(3)
    const float N1 = 0.577350f; 
    float N2, N3, N4, N5, N6, N7;

    if (isBT709) 
    {
        N2 = 0.099275f;
        N3 = -0.467683f;
        N4 = -0.434569f;
        N5 = 0.472051f;
        N6 = 0.385973f;
        N7 = 0.556223f;
    } 
    else // BT601
    {
        N2 = 0.130816f;
        N3 = -0.310678f;
        N4 = -0.472866f;
        N5 = 0.405713f;
        N6 = 0.411049f;
        N7 = 0.512784f;
    }

    // 2. LOAD CONSTANTS INTO 256-BIT REGISTERS
    const __m256 vN1 = _mm256_set1_ps(N1);
    const __m256 vN2 = _mm256_set1_ps(N2);
    const __m256 vN3 = _mm256_set1_ps(N3);
    const __m256 vN4 = _mm256_set1_ps(N4);
    const __m256 vN5 = _mm256_set1_ps(N5);
    const __m256 vN6 = _mm256_set1_ps(N6);
    const __m256 vN7 = _mm256_set1_ps(N7);

    // Rounding, clamping, and signed-to-unsigned restoration constants
    const __m256 vZero = _mm256_setzero_ps();
    const __m256 v255  = _mm256_set1_ps(255.0f);
    const __m256 vHalf = _mm256_set1_ps(0.5f);
    const __m256 v128  = _mm256_set1_ps(128.0f);

    // Alpha mask: Isolates the highest 8 bits (Byte 3: A) of the VUYA memory layout
    const __m256i alpha_mask = _mm256_set1_epi32(0xFF000000); 

    const int32_t vecSize = 8;
    const int32_t endX = (w / vecSize) * vecSize;

    for (int32_t y = 0; y < h; ++y)
    {
        // Planar inputs are tightly packed
        const float* pY_row = pY + (y * w);
        const float* pU_row = pU + (y * w);
        const float* pV_row = pV + (y * w);
        
        // Interleaved buffers use their specific pitches
        const PF_Pixel_VUYA_8u* pIn_row = pInput + (y * src_pitch);
        PF_Pixel_VUYA_8u* pOut_row = pOutput + (y * dst_pitch);

        int32_t x = 0;

        // =========================================================
        // MAIN AVX2 KERNEL (8 Pixels per loop)
        // =========================================================
        for (; x < endX; x += vecSize)
        {
            // 1. Load Planar Orthonormal Floats
            __m256 fY = _mm256_loadu_ps(pY_row + x);
            __m256 fU = _mm256_loadu_ps(pU_row + x);
            __m256 fV = _mm256_loadu_ps(pV_row + x);

            // 2. Fast Inverse FMA Math (Y cancels out for U and V!)
            __m256 resY = _mm256_fmadd_ps(vN1, fY, _mm256_fmadd_ps(vN2, fU, _mm256_mul_ps(vN3, fV)));
            __m256 resU = _mm256_fmadd_ps(vN4, fU, _mm256_fmadd_ps(vN5, fV, v128)); // Shift +128 inline
            __m256 resV = _mm256_fmadd_ps(vN6, fU, _mm256_fmadd_ps(vN7, fV, v128)); // Shift +128 inline

            // 3. Clamp to [0, 255] and add 0.5f for accurate truncation rounding
            resY = _mm256_add_ps(_mm256_min_ps(v255, _mm256_max_ps(vZero, resY)), vHalf);
            resU = _mm256_add_ps(_mm256_min_ps(v255, _mm256_max_ps(vZero, resU)), vHalf);
            resV = _mm256_add_ps(_mm256_min_ps(v255, _mm256_max_ps(vZero, resV)), vHalf);

            // 4. Convert to 32-bit Integers
            __m256i Y_i = _mm256_cvttps_epi32(resY);
            __m256i U_i = _mm256_cvttps_epi32(resU);
            __m256i V_i = _mm256_cvttps_epi32(resV);

            // 5. Extract original Alpha channel from pInput
            __m256i in_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pIn_row + x));
            __m256i A_i = _mm256_and_si256(in_pixels, alpha_mask);

            // 6. Shift V, U, Y into VUYA memory layout (Little-Endian layout: V U Y A)
            // V_i remains in bits [7:0]
            U_i = _mm256_slli_epi32(U_i, 8);  // Shift U to bits [15:8]
            Y_i = _mm256_slli_epi32(Y_i, 16); // Shift Y to bits [23:16]

            // 7. Combine channels using Bitwise OR and store
            __m256i out_pixels = _mm256_or_si256(A_i, _mm256_or_si256(Y_i, _mm256_or_si256(U_i, V_i)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(pOut_row + x), out_pixels);
        }

        // =========================================================
        // SCALAR TAIL (For dimensions not divisible by 8)
        // =========================================================
        for (; x < w; ++x)
        {
            float y_val = pY_row[x];
            float u_val = pU_row[x];
            float v_val = pV_row[x];

            float r_y = N1 * y_val + N2 * u_val + N3 * v_val;
            float r_u = N4 * u_val + N5 * v_val + 128.0f;
            float r_v = N6 * u_val + N7 * v_val + 128.0f;

            auto clamp8 = [](float val) noexcept
            {
                return static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val + 0.5f)));
            };

            PF_Pixel_VUYA_8u* pOut = pOut_row + x;
            const PF_Pixel_VUYA_8u* pIn = pIn_row + x;

            pOut->Y = clamp8(r_y);
            pOut->U = clamp8(r_u);
            pOut->V = clamp8(r_v);
            pOut->A = pIn->A; // Preserve exact alpha from input
        }
    }
    
    return;
}