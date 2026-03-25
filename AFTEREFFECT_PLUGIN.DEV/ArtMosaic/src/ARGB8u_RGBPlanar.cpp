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

void planar2rgb
(
    const PF_Pixel_ARGB_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_ARGB_8u* RESTRICT pDst,
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
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);

    // Alpha is at Byte 0 in ARGB_8u memory, which corresponds to the lowest 8 bits in integer.
    const __m256i v_alpha_mask = _mm256_set1_epi32(0x000000FF);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_8u* pSrcLine = pSrc + j * srcPitch;
        PF_Pixel_ARGB_8u* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;
            __m256 v_r_f = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pR[idx]), v_zero), v_255);
            __m256 v_g_f = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pG[idx]), v_zero), v_255);
            __m256 v_b_f = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pB[idx]), v_zero), v_255);

            __m256i v_r_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_r_f), 8);
            __m256i v_g_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_g_f), 16);
            __m256i v_b_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_b_f), 24);

            __m256i v_src_alpha = _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pSrcLine[i])), v_alpha_mask);

            __m256i v_argb = _mm256_or_si256(v_b_i, _mm256_or_si256(v_g_i, _mm256_or_si256(v_r_i, v_src_alpha)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_argb);
        }

        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            float r = pR[idx] < 0.0f ? 0.0f : (pR[idx] > 255.0f ? 255.0f : pR[idx]);
            float g = pG[idx] < 0.0f ? 0.0f : (pG[idx] > 255.0f ? 255.0f : pG[idx]);
            float b = pB[idx] < 0.0f ? 0.0f : (pB[idx] > 255.0f ? 255.0f : pB[idx]);

            pOutLine[i].A = pSrcLine[i].A; // If this is 0, AE will render a black/transparent frame!
            pOutLine[i].R = static_cast<A_u_char>(r);
            pOutLine[i].G = static_cast<A_u_char>(g);
            pOutLine[i].B = static_cast<A_u_char>(b);
        }
    }
}