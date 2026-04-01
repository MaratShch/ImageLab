#include "MosaicColorConvert.hpp"


void rgb2planar
(
    const PF_Pixel_ARGB_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch // MUST BE IN PIXELS, NOT BYTES!
)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256i mask_FF = _mm256_set1_epi32(0xFF);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_8u* pLine = pSrc + j * srcPitch;
        A_long i = 0;

        for (; i < spanX8; i += 8)
        {
            // Memory is A, R, G, B. Little-endian loads this as 0xBBGGRRAA.
            __m256i v_argb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pLine[i]));

            // Extract channels using correct bit-shifts for 0xBBGGRRAA
            __m256i v_r_int = _mm256_and_si256(_mm256_srli_epi32(v_argb, 8), mask_FF);
            __m256i v_g_int = _mm256_and_si256(_mm256_srli_epi32(v_argb, 16), mask_FF);
            __m256i v_b_int = _mm256_and_si256(_mm256_srli_epi32(v_argb, 24), mask_FF);

            _mm256_storeu_ps(&pR[j * sizeX + i], _mm256_cvtepi32_ps(v_r_int));
            _mm256_storeu_ps(&pG[j * sizeX + i], _mm256_cvtepi32_ps(v_g_int));
            _mm256_storeu_ps(&pB[j * sizeX + i], _mm256_cvtepi32_ps(v_b_int));
        }

        for (; i < sizeX; i++)
        {
            pR[j * sizeX + i] = static_cast<float>(pLine[i].R);
            pG[j * sizeX + i] = static_cast<float>(pLine[i].G);
            pB[j * sizeX + i] = static_cast<float>(pLine[i].B);
        }
    }
}


void rgbp2planar
(
    const PF_Pixel_ARGB_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch // Assumed to be in PIXELS based on your snippet
)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;

    const __m256i mask_FF = _mm256_set1_epi32(0xFF);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_8u* pLine = pSrc + j * srcPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Memory is A, R, G, B. Little-endian loads this as 0xBBGGRRAA.
            __m256i v_argb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pLine[i]));

            // Extract channels using correct bit-shifts for 0xBBGGRRAA
            __m256i v_a_int = _mm256_and_si256(v_argb, mask_FF); // Alpha is at Byte 0
            __m256i v_r_int = _mm256_and_si256(_mm256_srli_epi32(v_argb, 8), mask_FF);
            __m256i v_g_int = _mm256_and_si256(_mm256_srli_epi32(v_argb, 16), mask_FF);
            __m256i v_b_int = _mm256_and_si256(_mm256_srli_epi32(v_argb, 24), mask_FF);

            // Convert integer to float
            __m256 v_a_f = _mm256_cvtepi32_ps(v_a_int);
            __m256 v_r_f = _mm256_cvtepi32_ps(v_r_int);
            __m256 v_g_f = _mm256_cvtepi32_ps(v_g_int);
            __m256 v_b_f = _mm256_cvtepi32_ps(v_b_int);

            // Create a mask to protect against Division by Zero (where Alpha > 0)
            __m256 mask_a_gt_0 = _mm256_cmp_ps(v_a_f, v_zero, _CMP_GT_OQ);

            // Calculate un-premultiply factor: 255.0 / Alpha
            __m256 v_factor = _mm256_div_ps(v_255, v_a_f);

            // Apply mask: if Alpha was 0, force the factor to 0.0f
            v_factor = _mm256_and_ps(v_factor, mask_a_gt_0);

            // Un-premultiply and safely clamp to 255.0f
            v_r_f = _mm256_min_ps(_mm256_mul_ps(v_r_f, v_factor), v_255);
            v_g_f = _mm256_min_ps(_mm256_mul_ps(v_g_f, v_factor), v_255);
            v_b_f = _mm256_min_ps(_mm256_mul_ps(v_b_f, v_factor), v_255);

            // Store to planar buffers
            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pR[idx], v_r_f);
            _mm256_storeu_ps(&pG[idx], v_g_f);
            _mm256_storeu_ps(&pB[idx], v_b_f);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            A_u_char a = pLine[i].A;

            if (a > 0)
            {
                float factor = 255.0f / static_cast<float>(a);
                float r = static_cast<float>(pLine[i].R) * factor;
                float g = static_cast<float>(pLine[i].G) * factor;
                float b = static_cast<float>(pLine[i].B) * factor;

                // Clamp to prevent blowout from illegal source pixels
                pR[idx] = r > 255.0f ? 255.0f : r;
                pG[idx] = g > 255.0f ? 255.0f : g;
                pB[idx] = b > 255.0f ? 255.0f : b;
            }
            else
            {
                // If completely transparent, snap RGB to 0
                pR[idx] = 0.0f;
                pG[idx] = 0.0f;
                pB[idx] = 0.0f;
            }
        }
    }
}