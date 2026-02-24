#include "ColorConvert.hpp"

// Assuming PF_Pixel_VUYA_8u is defined in your headers as:
// typedef struct { A_u_char V; A_u_char U; A_u_char Y; A_u_char A; } PF_Pixel_VUYA_8u;

void AVX2_Convert_BGRA_8u_YUV // Keeping your exact requested name
(
    const PF_Pixel_VUYA_8u* RESTRICT pInput, // Input VUYA_8u (Interleaved)
    float* RESTRICT Y_out,                   // Y plane (Luma)
    float* RESTRICT U_out,                   // U plane (Red-Blue axis)
    float* RESTRICT V_out,                   // V plane (Green-Magenta axis)
    int32_t sizeX,                           // Width in Pixels
    int32_t sizeY,                           // Height in Pixels
    int32_t linePitch,                       // Row Pitch in Pixels
    bool isBT709                             // true for BT709, false for BT601
) noexcept
{
    // 1. PRE-CALCULATE DIRECT MATRIX COEFFICIENTS (Branch outside the loop)
    // M1 = 3 * c1
    const float M1 = 1.73205081f; 
    float M2, M3, M4, M5, M6, M7;

    if (isBT709) 
    {
        M2 = 0.963177f;
        M3 = 0.638942f;
        M4 = -1.312107f;
        M5 = 1.113551f;
        M6 = 0.910495f;
        M7 = 1.025131f;
    } 
    else // BT601
    {
        M2 = 0.824377f;
        M3 = 0.397140f;
        M4 = -1.252993f;
        M5 = 0.991364f;
        M6 = 1.004402f;
        M7 = 1.155454f;
    }

    // 2. LOAD CONSTANTS INTO 256-BIT REGISTERS
    const __m256 vM1 = _mm256_set1_ps(M1);
    const __m256 vM2 = _mm256_set1_ps(M2);
    const __m256 vM3 = _mm256_set1_ps(M3);
    const __m256 vM4 = _mm256_set1_ps(M4);
    const __m256 vM5 = _mm256_set1_ps(M5);
    const __m256 vM6 = _mm256_set1_ps(M6);
    const __m256 vM7 = _mm256_set1_ps(M7);
    const __m256 v128 = _mm256_set1_ps(128.0f); // The UV signed shift

    // 3. PREPARE SHUFFLE MASKS FOR VUYA_8u INTERLEAVED LAYOUT
    // Layout in memory: V (byte 0), U (byte 1), Y (byte 2), A (byte 3)
    const __m256i maskV = _mm256_set_epi8(
        -1, -1, -1, 12, -1, -1, -1, 8, -1, -1, -1, 4, -1, -1, -1, 0,
        -1, -1, -1, 12, -1, -1, -1, 8, -1, -1, -1, 4, -1, -1, -1, 0
    );
    const __m256i maskU = _mm256_set_epi8(
        -1, -1, -1, 13, -1, -1, -1, 9, -1, -1, -1, 5, -1, -1, -1, 1,
        -1, -1, -1, 13, -1, -1, -1, 9, -1, -1, -1, 5, -1, -1, -1, 1
    );
    const __m256i maskY = _mm256_set_epi8(
        -1, -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2,
        -1, -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2
    );

    const int32_t vecSize = 8;
    const int32_t endX = (sizeX / vecSize) * vecSize;

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const PF_Pixel_VUYA_8u* rowIn = pInput + (y * linePitch);
        float* rowY = Y_out + (y * sizeX);
        float* rowU = U_out + (y * sizeX);
        float* rowV = V_out + (y * sizeX);

        int32_t x = 0;

        // =========================================================
        // MAIN AVX2 KERNEL (8 Pixels per loop)
        // =========================================================
        for (; x < endX; x += vecSize)
        {
            // Load 8 VUYA pixels (32 bytes)
            __m256i vuya = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowIn + x));

            // Extract channels to floats
            __m256 fY = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vuya, maskY));
            __m256 fU = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vuya, maskU));
            __m256 fV = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vuya, maskV));

            // Shift U and V from [0, 255] to [-128, 127]
            fU = _mm256_sub_ps(fU, v128);
            fV = _mm256_sub_ps(fV, v128);

            // Fast FMA Math
            // Y_ortho = M1*Y + M2*U + M3*V
            __m256 resY = _mm256_fmadd_ps(vM1, fY, _mm256_fmadd_ps(vM2, fU, _mm256_mul_ps(vM3, fV)));
            
            // U_ortho = M4*U + M5*V  (Notice Y cancels out entirely!)
            __m256 resU = _mm256_fmadd_ps(vM4, fU, _mm256_mul_ps(vM5, fV));
            
            // V_ortho = M6*U + M7*V  (Y cancels out here too!)
            __m256 resV = _mm256_fmadd_ps(vM6, fU, _mm256_mul_ps(vM7, fV));

            // Store Planar Results
            _mm256_storeu_ps(rowY + x, resY);
            _mm256_storeu_ps(rowU + x, resU);
            _mm256_storeu_ps(rowV + x, resV);
        }

        // =========================================================
        // SCALAR TAIL (For dimensions not divisible by 8)
        // =========================================================
        for (; x < sizeX; ++x)
        {
            float y_val = static_cast<float>(rowIn[x].Y);
            float u_val = static_cast<float>(rowIn[x].U) - 128.0f;
            float v_val = static_cast<float>(rowIn[x].V) - 128.0f;

            rowY[x] = M1 * y_val + M2 * u_val + M3 * v_val;
            rowU[x] = M4 * u_val + M5 * v_val;
            rowV[x] = M6 * u_val + M7 * v_val;
        }
    }
	
	return;
}