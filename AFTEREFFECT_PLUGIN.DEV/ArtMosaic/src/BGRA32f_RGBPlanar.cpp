#include "MosaicColorConvert.hpp"
#include <limits>

void rgb2planar
(
    const PF_Pixel_BGRA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;

    // Scale up from input [0.0f, 1.0f] to our internal planar [0.0f, 255.0f]
    const __m256 v_scale = _mm256_set1_ps(255.0f);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_BGRA_32f* pLine = pSrc + j * linePitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Gather the 32-bit float channels directly into AVX vectors.
            // Note: _mm256_set_ps takes arguments in reverse order (highest to lowest index)
            __m256 v_b = _mm256_set_ps(pLine[i + 7].B, pLine[i + 6].B, pLine[i + 5].B, pLine[i + 4].B, pLine[i + 3].B, pLine[i + 2].B, pLine[i + 1].B, pLine[i].B);
            __m256 v_g = _mm256_set_ps(pLine[i + 7].G, pLine[i + 6].G, pLine[i + 5].G, pLine[i + 4].G, pLine[i + 3].G, pLine[i + 2].G, pLine[i + 1].G, pLine[i].G);
            __m256 v_r = _mm256_set_ps(pLine[i + 7].R, pLine[i + 6].R, pLine[i + 5].R, pLine[i + 4].R, pLine[i + 3].R, pLine[i + 2].R, pLine[i + 1].R, pLine[i].R);

            // Scale up directly (no integer conversion needed!)
            v_b = _mm256_mul_ps(v_b, v_scale);
            v_g = _mm256_mul_ps(v_g, v_scale);
            v_r = _mm256_mul_ps(v_r, v_scale);

            // Store to planar buffers
            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pB[idx], v_b);
            _mm256_storeu_ps(&pG[idx], v_g);
            _mm256_storeu_ps(&pR[idx], v_r);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            pR[idx] = pLine[i].R * 255.0f;
            pG[idx] = pLine[i].G * 255.0f;
            pB[idx] = pLine[i].B * 255.0f;
        }
    }

    return;
}


void rgbp2planar
(
    const PF_Pixel_BGRA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
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
        // Assuming linePitch is provided in pixel units matching your original code
        const PF_Pixel_BGRA_32f* pLine = pSrc + j * linePitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Gather the 32-bit float channels directly into AVX vectors.
            // Note: _mm256_set_ps takes arguments from highest index to lowest
            __m256 v_b = _mm256_set_ps(pLine[i + 7].B, pLine[i + 6].B, pLine[i + 5].B, pLine[i + 4].B, pLine[i + 3].B, pLine[i + 2].B, pLine[i + 1].B, pLine[i].B);
            __m256 v_g = _mm256_set_ps(pLine[i + 7].G, pLine[i + 6].G, pLine[i + 5].G, pLine[i + 4].G, pLine[i + 3].G, pLine[i + 2].G, pLine[i + 1].G, pLine[i].G);
            __m256 v_r = _mm256_set_ps(pLine[i + 7].R, pLine[i + 6].R, pLine[i + 5].R, pLine[i + 4].R, pLine[i + 3].R, pLine[i + 2].R, pLine[i + 1].R, pLine[i].R);
            __m256 v_a = _mm256_set_ps(pLine[i + 7].A, pLine[i + 6].A, pLine[i + 5].A, pLine[i + 4].A, pLine[i + 3].A, pLine[i + 2].A, pLine[i + 1].A, pLine[i].A);

            // Create a mask to protect against Division by Zero (Alpha > 0.0f)
            __m256 mask_a_gt_0 = _mm256_cmp_ps(v_a, v_zero, _CMP_GT_OQ);

            // Compute combined un-premultiply and scale factor: 255.0f / Alpha
            __m256 v_factor = _mm256_div_ps(v_255, v_a);

            // Apply mask: if Alpha was 0.0f, force the factor to 0.0f
            v_factor = _mm256_and_ps(v_factor, mask_a_gt_0);

            // Multiply by factor and safely clamp to 255.0f to protect internal formats from HDR blowouts
            v_b = _mm256_min_ps(_mm256_mul_ps(v_b, v_factor), v_255);
            v_g = _mm256_min_ps(_mm256_mul_ps(v_g, v_factor), v_255);
            v_r = _mm256_min_ps(_mm256_mul_ps(v_r, v_factor), v_255);

            // Store to planar buffers
            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pB[idx], v_b);
            _mm256_storeu_ps(&pG[idx], v_g);
            _mm256_storeu_ps(&pR[idx], v_r);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            float a = pLine[i].A;

            if (a > 0.0f)
            {
                float factor = 255.0f / a;
                float r = pLine[i].R * factor;
                float g = pLine[i].G * factor;
                float b = pLine[i].B * factor;

                // Clamp to prevent blowout
                pR[idx] = r > 255.0f ? 255.0f : r;
                pG[idx] = g > 255.0f ? 255.0f : g;
                pB[idx] = b > 255.0f ? 255.0f : b;
            }
            else
            {
                pR[idx] = 0.0f;
                pG[idx] = 0.0f;
                pB[idx] = 0.0f;
            }
        }
    }

    return;
}