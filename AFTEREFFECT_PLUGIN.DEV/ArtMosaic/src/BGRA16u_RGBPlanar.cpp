#include "MosaicColorConvert.hpp"


void rgb2planar
(
    const PF_Pixel_BGRA_16u* RESTRICT pSrc,
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

    // Instead of dividing, we multiply by the reciprocal!
    constexpr float scaleDown = 255.0f / 32767.0f;
    const __m256 v_scale = _mm256_set1_ps(scaleDown);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_BGRA_16u* pLine = pSrc + j * linePitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Gather 16-bit channels directly into 32-bit integer AVX vectors.
            // Note: _mm256_set_epi32 takes arguments in reverse order (highest to lowest index)
            __m256i v_b_int = _mm256_set_epi32(pLine[i + 7].B, pLine[i + 6].B, pLine[i + 5].B, pLine[i + 4].B, pLine[i + 3].B, pLine[i + 2].B, pLine[i + 1].B, pLine[i].B);
            __m256i v_g_int = _mm256_set_epi32(pLine[i + 7].G, pLine[i + 6].G, pLine[i + 5].G, pLine[i + 4].G, pLine[i + 3].G, pLine[i + 2].G, pLine[i + 1].G, pLine[i].G);
            __m256i v_r_int = _mm256_set_epi32(pLine[i + 7].R, pLine[i + 6].R, pLine[i + 5].R, pLine[i + 4].R, pLine[i + 3].R, pLine[i + 2].R, pLine[i + 1].R, pLine[i].R);

            // Convert integer to float
            __m256 v_b_f = _mm256_cvtepi32_ps(v_b_int);
            __m256 v_g_f = _mm256_cvtepi32_ps(v_g_int);
            __m256 v_r_f = _mm256_cvtepi32_ps(v_r_int);

            // Scale down to [0.0f, 255.0f] using multiplication
            v_b_f = _mm256_mul_ps(v_b_f, v_scale);
            v_g_f = _mm256_mul_ps(v_g_f, v_scale);
            v_r_f = _mm256_mul_ps(v_r_f, v_scale);

            // Store to planar buffers
            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pB[idx], v_b_f);
            _mm256_storeu_ps(&pG[idx], v_g_f);
            _mm256_storeu_ps(&pR[idx], v_r_f);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            pR[idx] = static_cast<float>(pLine[i].R) * scaleDown;
            pG[idx] = static_cast<float>(pLine[i].G) * scaleDown;
            pB[idx] = static_cast<float>(pLine[i].B) * scaleDown;
        }
    }

    return;
}


void planar2rgb
(
    const PF_Pixel_BGRA_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_BGRA_16u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;

    // Scaling multiplier to restore [0.0f, 255.0f] back to [0, 32767]
    const float scaleUp = 32767.0f / 255.0f;
    const __m256 v_scale = _mm256_set1_ps(scaleUp);

    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(32767.0f);

    // Aligned temporary arrays to quickly extract AVX results
    alignas(32) int32_t r_out[8];
    alignas(32) int32_t g_out[8];
    alignas(32) int32_t b_out[8];

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_BGRA_16u* pSrcLine = pSrc + j * srcPitch;
        PF_Pixel_BGRA_16u* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;

            // 1. Load planar floats [0.0 - 255.0]
            __m256 v_r_f = _mm256_loadu_ps(&pR[idx]);
            __m256 v_g_f = _mm256_loadu_ps(&pG[idx]);
            __m256 v_b_f = _mm256_loadu_ps(&pB[idx]);

            // 2. Scale back up to [0.0 - 32767.0]
            v_r_f = _mm256_mul_ps(v_r_f, v_scale);
            v_g_f = _mm256_mul_ps(v_g_f, v_scale);
            v_b_f = _mm256_mul_ps(v_b_f, v_scale);

            // 3. Clamp to prevent wrapping
            v_r_f = _mm256_min_ps(_mm256_max_ps(v_r_f, v_zero), v_max);
            v_g_f = _mm256_min_ps(_mm256_max_ps(v_g_f, v_zero), v_max);
            v_b_f = _mm256_min_ps(_mm256_max_ps(v_b_f, v_zero), v_max);

            // 4. Convert to 32-bit integers
            __m256i v_r_i = _mm256_cvtps_epi32(v_r_f);
            __m256i v_g_i = _mm256_cvtps_epi32(v_g_f);
            __m256i v_b_i = _mm256_cvtps_epi32(v_b_f);

            // 5. Store cleanly into aligned temp arrays
            _mm256_store_si256(reinterpret_cast<__m256i*>(r_out), v_r_i);
            _mm256_store_si256(reinterpret_cast<__m256i*>(g_out), v_g_i);
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_out), v_b_i);

            // 6. Pack into the destination struct, restoring the original Alpha
            for (int k = 0; k < 8; ++k)
            {
                pOutLine[i + k].R = static_cast<A_u_short>(r_out[k]);
                pOutLine[i + k].G = static_cast<A_u_short>(g_out[k]);
                pOutLine[i + k].B = static_cast<A_u_short>(b_out[k]);
                pOutLine[i + k].A = pSrcLine[i + k].A;
            }
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;

            float r = pR[idx] * scaleUp;
            float g = pG[idx] * scaleUp;
            float b = pB[idx] * scaleUp;

            // Safe scalar clamping
            r = r < 0.0f ? 0.0f : (r > 32767.0f ? 32767.0f : r);
            g = g < 0.0f ? 0.0f : (g > 32767.0f ? 32767.0f : g);
            b = b < 0.0f ? 0.0f : (b > 32767.0f ? 32767.0f : b);

            pOutLine[i].R = static_cast<A_u_short>(r);
            pOutLine[i].G = static_cast<A_u_short>(g);
            pOutLine[i].B = static_cast<A_u_short>(b);

            // Grab Alpha directly from the source buffer
            pOutLine[i].A = pSrcLine[i].A;
        }
    }

    return;
}