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

void planar2rgb
(
    const PF_Pixel_BGRA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_BGRA_32f* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;

    // Scale down from planar [0.0f, 255.0f] to output [0.0f, 1.0f]
    constexpr float scaleDown = 1.0f / 255.0f;
    const __m256 v_scale = _mm256_set1_ps(scaleDown);

    constexpr float maxClampVal = 1.0f - std::numeric_limits<float>::epsilon();

    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(maxClampVal);

    // Aligned temporary arrays to quickly extract the raw floats
    CACHE_ALIGN float r_out[8];
    CACHE_ALIGN float g_out[8];
    CACHE_ALIGN float b_out[8];

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_BGRA_32f* pSrcLine = pSrc + j * srcPitch;
        PF_Pixel_BGRA_32f* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;

            // 1. Load planar floats [0.0f - 255.0f]
            __m256 v_r = _mm256_loadu_ps(&pR[idx]);
            __m256 v_g = _mm256_loadu_ps(&pG[idx]);
            __m256 v_b = _mm256_loadu_ps(&pB[idx]);

            // 2. Scale down to [0.0f - 1.0f] using multiplication
            v_r = _mm256_mul_ps(v_r, v_scale);
            v_g = _mm256_mul_ps(v_g, v_scale);
            v_b = _mm256_mul_ps(v_b, v_scale);

            // 3. Clamp safely between 0.0f and (1.0f - FLT_EPSILON)
            v_r = _mm256_min_ps(_mm256_max_ps(v_r, v_zero), v_max);
            v_g = _mm256_min_ps(_mm256_max_ps(v_g, v_zero), v_max);
            v_b = _mm256_min_ps(_mm256_max_ps(v_b, v_zero), v_max);

            // 4. Store straight into aligned temp arrays (no int conversion!)
            _mm256_store_ps(r_out, v_r);
            _mm256_store_ps(g_out, v_g);
            _mm256_store_ps(b_out, v_b);

            // 5. Pack into the destination struct, copying original Alpha
            for (int k = 0; k < 8; ++k)
            {
                pOutLine[i + k].B = b_out[k];
                pOutLine[i + k].G = g_out[k];
                pOutLine[i + k].R = r_out[k];
                pOutLine[i + k].A = pSrcLine[i + k].A;
            }
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;

            float r = pR[idx] * scaleDown;
            float g = pG[idx] * scaleDown;
            float b = pB[idx] * scaleDown;

            // Safe scalar clamping
            r = r < 0.0f ? 0.0f : (r > maxClampVal ? maxClampVal : r);
            g = g < 0.0f ? 0.0f : (g > maxClampVal ? maxClampVal : g);
            b = b < 0.0f ? 0.0f : (b > maxClampVal ? maxClampVal : b);

            pOutLine[i].B = b;
            pOutLine[i].G = g;
            pOutLine[i].R = r;
            pOutLine[i].A = pSrcLine[i].A;
        }
    }
}