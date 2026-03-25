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

void planar2rgb(const PF_Pixel_ARGB_16u* RESTRICT pSrc, const MemHandler& memHndl, PF_Pixel_ARGB_16u* RESTRICT pDst, A_long sizeX, A_long sizeY, A_long srcPitch, A_long dstPitch)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256 v_scale = _mm256_set1_ps(32767.0f / 255.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(32767.0f);

    CACHE_ALIGN int32_t r_out[8];
    CACHE_ALIGN int32_t g_out[8];
    CACHE_ALIGN int32_t b_out[8];

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_16u* pSrcLine = pSrc + j * srcPitch;
        PF_Pixel_ARGB_16u* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;
            __m256 v_r_f = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale), v_zero), v_max);
            __m256 v_g_f = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale), v_zero), v_max);
            __m256 v_b_f = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale), v_zero), v_max);

            _mm256_store_si256(reinterpret_cast<__m256i*>(r_out), _mm256_cvtps_epi32(v_r_f));
            _mm256_store_si256(reinterpret_cast<__m256i*>(g_out), _mm256_cvtps_epi32(v_g_f));
            _mm256_store_si256(reinterpret_cast<__m256i*>(b_out), _mm256_cvtps_epi32(v_b_f));

            for (int k = 0; k < 8; ++k)
            {
                pOutLine[i + k].A = pSrcLine[i + k].A;
                pOutLine[i + k].R = static_cast<A_u_short>(r_out[k]);
                pOutLine[i + k].G = static_cast<A_u_short>(g_out[k]);
                pOutLine[i + k].B = static_cast<A_u_short>(b_out[k]);
            }
        }
        for (; i < sizeX; i++)
        {
            float r = pR[j * sizeX + i] * (32767.0f / 255.0f);
            float g = pG[j * sizeX + i] * (32767.0f / 255.0f);
            float b = pB[j * sizeX + i] * (32767.0f / 255.0f);
            pOutLine[i].A = pSrcLine[i].A;
            pOutLine[i].R = static_cast<A_u_short>(r < 0.0f ? 0.0f : (r > 32767.0f ? 32767.0f : r));
            pOutLine[i].G = static_cast<A_u_short>(g < 0.0f ? 0.0f : (g > 32767.0f ? 32767.0f : g));
            pOutLine[i].B = static_cast<A_u_short>(b < 0.0f ? 0.0f : (b > 32767.0f ? 32767.0f : b));
        }
    }
}