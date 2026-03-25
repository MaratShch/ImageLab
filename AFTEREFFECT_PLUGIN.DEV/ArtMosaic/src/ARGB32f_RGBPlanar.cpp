#include "MosaicColorConvert.hpp"
#include <limits>


#include <limits> 

void rgb2planar(const PF_Pixel_ARGB_32f* RESTRICT pSrc, const MemHandler& memHndl, A_long sizeX, A_long sizeY, A_long srcPitch)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;
    const A_long spanX8 = sizeX & ~7;
    const __m256 v_scale = _mm256_set1_ps(255.0f);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_32f* pLine = pSrc + j * srcPitch;
        A_long i = 0;

        for (; i < spanX8; i += 8)
        {
            __m256 v_r = _mm256_mul_ps(_mm256_set_ps(pLine[i + 7].R, pLine[i + 6].R, pLine[i + 5].R, pLine[i + 4].R, pLine[i + 3].R, pLine[i + 2].R, pLine[i + 1].R, pLine[i].R), v_scale);
            __m256 v_g = _mm256_mul_ps(_mm256_set_ps(pLine[i + 7].G, pLine[i + 6].G, pLine[i + 5].G, pLine[i + 4].G, pLine[i + 3].G, pLine[i + 2].G, pLine[i + 1].G, pLine[i].G), v_scale);
            __m256 v_b = _mm256_mul_ps(_mm256_set_ps(pLine[i + 7].B, pLine[i + 6].B, pLine[i + 5].B, pLine[i + 4].B, pLine[i + 3].B, pLine[i + 2].B, pLine[i + 1].B, pLine[i].B), v_scale);

            _mm256_storeu_ps(&pR[j * sizeX + i], v_r);
            _mm256_storeu_ps(&pG[j * sizeX + i], v_g);
            _mm256_storeu_ps(&pB[j * sizeX + i], v_b);
        }

        for (; i < sizeX; i++)
        {
            pR[j * sizeX + i] = pLine[i].R * 255.0f;
            pG[j * sizeX + i] = pLine[i].G * 255.0f;
            pB[j * sizeX + i] = pLine[i].B * 255.0f;
        }
    }
}

void planar2rgb(const PF_Pixel_ARGB_32f* RESTRICT pSrc, const MemHandler& memHndl, PF_Pixel_ARGB_32f* RESTRICT pDst, A_long sizeX, A_long sizeY, A_long srcPitch, A_long dstPitch)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;

    const float scaleDown = 1.0f / 255.0f;
    const __m256 v_scale = _mm256_set1_ps(scaleDown);
    const float maxClampVal = 1.0f - std::numeric_limits<float>::epsilon();
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(maxClampVal);

    CACHE_ALIGN float r_out[8];
    CACHE_ALIGN float g_out[8];
    CACHE_ALIGN float b_out[8];

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_32f* pSrcLine = pSrc + j * srcPitch;
        PF_Pixel_ARGB_32f* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;
            __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale), v_zero), v_max);
            __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale), v_zero), v_max);
            __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale), v_zero), v_max);

            _mm256_store_ps(r_out, v_r);
            _mm256_store_ps(g_out, v_g);
            _mm256_store_ps(b_out, v_b);

            for (int k = 0; k < 8; ++k)
            {
                pOutLine[i + k].A = pSrcLine[i + k].A;
                pOutLine[i + k].R = r_out[k];
                pOutLine[i + k].G = g_out[k];
                pOutLine[i + k].B = b_out[k];
            }
        }

        for (; i < sizeX; i++)
        {
            float r = pR[j * sizeX + i] * scaleDown;
            float g = pG[j * sizeX + i] * scaleDown;
            float b = pB[j * sizeX + i] * scaleDown;

            pOutLine[i].A = pSrcLine[i].A;
            pOutLine[i].R = r < 0.0f ? 0.0f : (r > maxClampVal ? maxClampVal : r);
            pOutLine[i].G = g < 0.0f ? 0.0f : (g > maxClampVal ? maxClampVal : g);
            pOutLine[i].B = b < 0.0f ? 0.0f : (b > maxClampVal ? maxClampVal : b);
        }
    }
}