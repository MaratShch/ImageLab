#include "MosaicColorConvert.hpp"


void rgb2planar(const PF_Pixel_ARGB_16u* RESTRICT pSrc, const MemHandler& memHndl, A_long sizeX, A_long sizeY, A_long srcPitch)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256 v_scale = _mm256_set1_ps(255.0f / 32767.0f);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_16u* pLine = pSrc + j * srcPitch;
        A_long i = 0;
        for (; i < spanX8; i += 8)
        {
            __m256 v_r = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_epi32(pLine[i + 7].R, pLine[i + 6].R, pLine[i + 5].R, pLine[i + 4].R, pLine[i + 3].R, pLine[i + 2].R, pLine[i + 1].R, pLine[i].R)), v_scale);
            __m256 v_g = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_epi32(pLine[i + 7].G, pLine[i + 6].G, pLine[i + 5].G, pLine[i + 4].G, pLine[i + 3].G, pLine[i + 2].G, pLine[i + 1].G, pLine[i].G)), v_scale);
            __m256 v_b = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_epi32(pLine[i + 7].B, pLine[i + 6].B, pLine[i + 5].B, pLine[i + 4].B, pLine[i + 3].B, pLine[i + 2].B, pLine[i + 1].B, pLine[i].B)), v_scale);

            _mm256_storeu_ps(&pR[j * sizeX + i], v_r);
            _mm256_storeu_ps(&pG[j * sizeX + i], v_g);
            _mm256_storeu_ps(&pB[j * sizeX + i], v_b);
        }
        for (; i < sizeX; i++)
        {
            pR[j * sizeX + i] = static_cast<float>(pLine[i].R) * (255.0f / 32767.0f);
            pG[j * sizeX + i] = static_cast<float>(pLine[i].G) * (255.0f / 32767.0f);
            pB[j * sizeX + i] = static_cast<float>(pLine[i].B) * (255.0f / 32767.0f);
        }
    }
}



void rgbp2planar
(
    const PF_Pixel_ARGB_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch // MUST BE IN PIXELS
)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;

    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_16u* pLine = pSrc + j * srcPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Gather 16-bit channels directly into 32-bit integer AVX vectors.
            // Note: _mm256_set_epi32 takes arguments from highest index to lowest
            __m256i v_r_int = _mm256_set_epi32(pLine[i + 7].R, pLine[i + 6].R, pLine[i + 5].R, pLine[i + 4].R, pLine[i + 3].R, pLine[i + 2].R, pLine[i + 1].R, pLine[i].R);
            __m256i v_g_int = _mm256_set_epi32(pLine[i + 7].G, pLine[i + 6].G, pLine[i + 5].G, pLine[i + 4].G, pLine[i + 3].G, pLine[i + 2].G, pLine[i + 1].G, pLine[i].G);
            __m256i v_b_int = _mm256_set_epi32(pLine[i + 7].B, pLine[i + 6].B, pLine[i + 5].B, pLine[i + 4].B, pLine[i + 3].B, pLine[i + 2].B, pLine[i + 1].B, pLine[i].B);
            __m256i v_a_int = _mm256_set_epi32(pLine[i + 7].A, pLine[i + 6].A, pLine[i + 5].A, pLine[i + 4].A, pLine[i + 3].A, pLine[i + 2].A, pLine[i + 1].A, pLine[i].A);

            // Convert integer to float
            __m256 v_r_f = _mm256_cvtepi32_ps(v_r_int);
            __m256 v_g_f = _mm256_cvtepi32_ps(v_g_int);
            __m256 v_b_f = _mm256_cvtepi32_ps(v_b_int);
            __m256 v_a_f = _mm256_cvtepi32_ps(v_a_int);

            // Create a mask to protect against Division by Zero (Alpha > 0)
            __m256 mask_a_gt_0 = _mm256_cmp_ps(v_a_f, v_zero, _CMP_GT_OQ);

            // The simplified un-premultiply factor: 255.0 / Alpha
            __m256 v_factor = _mm256_div_ps(v_255, v_a_f);

            // Apply mask: if Alpha was 0, force the factor to 0.0f
            v_factor = _mm256_and_ps(v_factor, mask_a_gt_0);

            // Un-premultiply AND clamp to [0.0f, 255.0f] to protect against illegal pixel blowouts
            __m256 v_r = _mm256_min_ps(_mm256_mul_ps(v_r_f, v_factor), v_255);
            __m256 v_g = _mm256_min_ps(_mm256_mul_ps(v_g_f, v_factor), v_255);
            __m256 v_b = _mm256_min_ps(_mm256_mul_ps(v_b_f, v_factor), v_255);

            // Store to planar buffers
            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pR[idx], v_r);
            _mm256_storeu_ps(&pG[idx], v_g);
            _mm256_storeu_ps(&pB[idx], v_b);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            A_u_short a = pLine[i].A;

            if (a > 0)
            {
                // Scalar version of the simplified formula
                float factor = 255.0f / static_cast<float>(a);
                float r = static_cast<float>(pLine[i].R) * factor;
                float g = static_cast<float>(pLine[i].G) * factor;
                float b = static_cast<float>(pLine[i].B) * factor;

                // Clamp to prevent blowout
                pR[idx] = r > 255.0f ? 255.0f : r;
                pG[idx] = g > 255.0f ? 255.0f : g;
                pB[idx] = b > 255.0f ? 255.0f : b;
            }
            else
            {
                // Fully transparent pixels evaluate to pure black/empty in YUV
                pR[idx] = 0.0f;
                pG[idx] = 0.0f;
                pB[idx] = 0.0f;
            }
        }
    }
}