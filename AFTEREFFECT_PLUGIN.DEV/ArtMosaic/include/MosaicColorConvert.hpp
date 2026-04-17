#pragma once

#include <limits>
#include "ImageMosaicUtils.hpp"
#include "CommonPixFormat.hpp"

void rgb2planar
(
    const PF_Pixel_BGRA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void rgb2planar
(
    const PF_Pixel_ARGB_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void rgb2planar
(
    const PF_Pixel_BGRA_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);


void rgb2planar
(
    const PF_Pixel_ARGB_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void rgb2planar
(
    const PF_Pixel_BGRA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void rgb2planar
(
    const PF_Pixel_ARGB_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);


void vuya2planar
(
    const PF_Pixel_VUYA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    bool is709
);

void planar2vuya
(
    const PF_Pixel_VUYA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_VUYA_8u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    bool is709
);

void vuya2planar
(
    const PF_Pixel_VUYA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    bool is709
);

void planar2vuya
(
    const PF_Pixel_VUYA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_VUYA_32f* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    bool is709
);


void rgb2planar
(
    const PF_Pixel_RGB_10u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch
);

void planar2rgb
(
    const MemHandler& memHndl,
    PF_Pixel_RGB_10u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long dstPitch
);

void rgbp2planar
(
    const PF_Pixel_BGRA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);


void rgbp2planar
(
    const PF_Pixel_BGRA_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void rgbp2planar
(
    const PF_Pixel_BGRA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void rgbp2planar
(
    const PF_Pixel_ARGB_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch // Assumed to be in PIXELS based on your snippet
);

void rgbp2planar
(
    const PF_Pixel_ARGB_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch // MUST BE IN PIXELS
);

void rgbp2planar
(
    const PF_Pixel_ARGB_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch // MUST BE IN PIXELS
);


// ==============================================================================
// 8-BIT (BGRA) - PITCH IN PIXELS
// ==============================================================================
template <bool IS_OPAQUE, bool IS_PREMUL>
void planar2rgb
(
    const PF_Pixel_BGRA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_BGRA_8u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch, // EXPECTS PIXELS
    A_long dstPitch  // EXPECTS PIXELS
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    const __m256 v_inv255 = _mm256_set1_ps(1.0f / 255.0f);
    const __m256i v_alpha_mask = _mm256_set1_epi32(static_cast<int>(0xFF000000));

    for (A_long j = 0; j < sizeY; j++)
    {
        // BULLETPROOF POINTER MATH: 
        // Cast to ptrdiff_t forces 64-bit SIGNED math. Native typed pointers handle the struct size automatically.
        const PF_Pixel_BGRA_8u* pSrcLine = pSrc + (static_cast<ptrdiff_t>(j) * srcPitch);
        PF_Pixel_BGRA_8u* pOutLine = pDst + (static_cast<ptrdiff_t>(j) * dstPitch);

        A_long i = 0;

        if (IS_OPAQUE)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pR[idx]), v_zero), v_255);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pG[idx]), v_zero), v_255);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pB[idx]), v_zero), v_255);

                __m256i v_r_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_r), 16);
                __m256i v_g_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_g), 8);
                __m256i v_b_i = _mm256_cvtps_epi32(v_b);

                __m256i v_bgra = _mm256_or_si256(v_b_i, _mm256_or_si256(v_g_i, _mm256_or_si256(v_r_i, v_alpha_mask)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_bgra);
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx]; float g = pG[idx]; float b = pB[idx];
                pOutLine[i].R = static_cast<A_u_char>(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
                pOutLine[i].G = static_cast<A_u_char>(g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g));
                pOutLine[i].B = static_cast<A_u_char>(b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b));
                pOutLine[i].A = 255;
            }
        }
        else if (IS_PREMUL)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_loadu_ps(&pR[idx]);
                __m256 v_g = _mm256_loadu_ps(&pG[idx]);
                __m256 v_b = _mm256_loadu_ps(&pB[idx]);

                __m256i v_src_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pSrcLine[i]));
                __m256i v_out_alpha = _mm256_and_si256(v_src_pixels, v_alpha_mask);
                __m256 v_a_norm = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(v_src_pixels, 24)), v_inv255);

                v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_r, v_a_norm), v_zero), v_255);
                v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_g, v_a_norm), v_zero), v_255);
                v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_b, v_a_norm), v_zero), v_255);

                __m256i v_r_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_r), 16);
                __m256i v_g_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_g), 8);
                __m256i v_b_i = _mm256_cvtps_epi32(v_b);

                __m256i v_bgra = _mm256_or_si256(v_b_i, _mm256_or_si256(v_g_i, _mm256_or_si256(v_r_i, v_out_alpha)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_bgra);
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                A_u_char a = pSrcLine[i].A;
                float a_norm = a * (1.0f / 255.0f);
                float r = pR[idx] * a_norm; float g = pG[idx] * a_norm; float b = pB[idx] * a_norm;
                pOutLine[i].R = static_cast<A_u_char>(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
                pOutLine[i].G = static_cast<A_u_char>(g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g));
                pOutLine[i].B = static_cast<A_u_char>(b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b));
                pOutLine[i].A = a;
            }
        }
        else
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pR[idx]), v_zero), v_255);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pG[idx]), v_zero), v_255);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pB[idx]), v_zero), v_255);

                __m256i v_r_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_r), 16);
                __m256i v_g_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_g), 8);
                __m256i v_b_i = _mm256_cvtps_epi32(v_b);

                __m256i v_src_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pSrcLine[i]));
                __m256i v_out_alpha = _mm256_and_si256(v_src_pixels, v_alpha_mask);
                __m256i v_bgra = _mm256_or_si256(v_b_i, _mm256_or_si256(v_g_i, _mm256_or_si256(v_r_i, v_out_alpha)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_bgra);
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx]; float g = pG[idx]; float b = pB[idx];
                pOutLine[i].R = static_cast<A_u_char>(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
                pOutLine[i].G = static_cast<A_u_char>(g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g));
                pOutLine[i].B = static_cast<A_u_char>(b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b));
                pOutLine[i].A = pSrcLine[i].A;
            }
        }
    }
}

// ==============================================================================
// 16-BIT (BGRA) - PITCH IN PIXELS
// ==============================================================================
template <bool IS_OPAQUE, bool IS_PREMUL>
void planar2rgb
(
    const PF_Pixel_BGRA_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_BGRA_16u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch, // EXPECTS PIXELS
    A_long dstPitch  // EXPECTS PIXELS
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const float scaleUp = 32767.0f / 255.0f;
    const __m256 v_scale = _mm256_set1_ps(scaleUp);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(32767.0f);
    const __m256 v_inv32767 = _mm256_set1_ps(1.0f / 32767.0f);

    CACHE_ALIGN int32_t r_out[8];
    CACHE_ALIGN int32_t g_out[8];
    CACHE_ALIGN int32_t b_out[8];

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_BGRA_16u* pSrcLine = pSrc + (static_cast<ptrdiff_t>(j) * srcPitch);
        PF_Pixel_BGRA_16u* pOutLine = pDst + (static_cast<ptrdiff_t>(j) * dstPitch);

        A_long i = 0;

        if (IS_OPAQUE)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale), v_zero), v_max);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale), v_zero), v_max);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale), v_zero), v_max);

                _mm256_store_si256(reinterpret_cast<__m256i*>(r_out), _mm256_cvtps_epi32(v_r));
                _mm256_store_si256(reinterpret_cast<__m256i*>(g_out), _mm256_cvtps_epi32(v_g));
                _mm256_store_si256(reinterpret_cast<__m256i*>(b_out), _mm256_cvtps_epi32(v_b));

                for (int k = 0; k < 8; ++k)
                {
                    pOutLine[i + k].R = static_cast<A_u_short>(r_out[k]);
                    pOutLine[i + k].G = static_cast<A_u_short>(g_out[k]);
                    pOutLine[i + k].B = static_cast<A_u_short>(b_out[k]);
                    pOutLine[i + k].A = 32767;
                }
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx] * scaleUp; float g = pG[idx] * scaleUp; float b = pB[idx] * scaleUp;
                pOutLine[i].R = static_cast<A_u_short>(r < 0.0f ? 0.0f : (r > 32767.0f ? 32767.0f : r));
                pOutLine[i].G = static_cast<A_u_short>(g < 0.0f ? 0.0f : (g > 32767.0f ? 32767.0f : g));
                pOutLine[i].B = static_cast<A_u_short>(b < 0.0f ? 0.0f : (b > 32767.0f ? 32767.0f : b));
                pOutLine[i].A = 32767;
            }
        }
        else if (IS_PREMUL)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale);
                __m256 v_g = _mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale);
                __m256 v_b = _mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale);

                __m256 v_a_f = _mm256_set_ps(pSrcLine[i + 7].A, pSrcLine[i + 6].A, pSrcLine[i + 5].A, pSrcLine[i + 4].A,
                    pSrcLine[i + 3].A, pSrcLine[i + 2].A, pSrcLine[i + 1].A, pSrcLine[i + 0].A);

                __m256 v_a_norm = _mm256_mul_ps(v_a_f, v_inv32767);
                v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_r, v_a_norm), v_zero), v_max);
                v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_g, v_a_norm), v_zero), v_max);
                v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_b, v_a_norm), v_zero), v_max);

                _mm256_store_si256(reinterpret_cast<__m256i*>(r_out), _mm256_cvtps_epi32(v_r));
                _mm256_store_si256(reinterpret_cast<__m256i*>(g_out), _mm256_cvtps_epi32(v_g));
                _mm256_store_si256(reinterpret_cast<__m256i*>(b_out), _mm256_cvtps_epi32(v_b));

                for (int k = 0; k < 8; ++k)
                {
                    pOutLine[i + k].R = static_cast<A_u_short>(r_out[k]);
                    pOutLine[i + k].G = static_cast<A_u_short>(g_out[k]);
                    pOutLine[i + k].B = static_cast<A_u_short>(b_out[k]);
                    pOutLine[i + k].A = pSrcLine[i + k].A;
                }
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                A_u_short a = pSrcLine[i].A;
                float a_norm = a * (1.0f / 32767.0f);
                float r = pR[idx] * scaleUp * a_norm; float g = pG[idx] * scaleUp * a_norm; float b = pB[idx] * scaleUp * a_norm;

                pOutLine[i].R = static_cast<A_u_short>(r < 0.0f ? 0.0f : (r > 32767.0f ? 32767.0f : r));
                pOutLine[i].G = static_cast<A_u_short>(g < 0.0f ? 0.0f : (g > 32767.0f ? 32767.0f : g));
                pOutLine[i].B = static_cast<A_u_short>(b < 0.0f ? 0.0f : (b > 32767.0f ? 32767.0f : b));
                pOutLine[i].A = a;
            }
        }
        else
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale), v_zero), v_max);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale), v_zero), v_max);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale), v_zero), v_max);

                _mm256_store_si256(reinterpret_cast<__m256i*>(r_out), _mm256_cvtps_epi32(v_r));
                _mm256_store_si256(reinterpret_cast<__m256i*>(g_out), _mm256_cvtps_epi32(v_g));
                _mm256_store_si256(reinterpret_cast<__m256i*>(b_out), _mm256_cvtps_epi32(v_b));

                for (int k = 0; k < 8; ++k)
                {
                    pOutLine[i + k].R = static_cast<A_u_short>(r_out[k]);
                    pOutLine[i + k].G = static_cast<A_u_short>(g_out[k]);
                    pOutLine[i + k].B = static_cast<A_u_short>(b_out[k]);
                    pOutLine[i + k].A = pSrcLine[i + k].A;
                }
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx] * scaleUp; float g = pG[idx] * scaleUp; float b = pB[idx] * scaleUp;

                pOutLine[i].R = static_cast<A_u_short>(r < 0.0f ? 0.0f : (r > 32767.0f ? 32767.0f : r));
                pOutLine[i].G = static_cast<A_u_short>(g < 0.0f ? 0.0f : (g > 32767.0f ? 32767.0f : g));
                pOutLine[i].B = static_cast<A_u_short>(b < 0.0f ? 0.0f : (b > 32767.0f ? 32767.0f : b));
                pOutLine[i].A = pSrcLine[i].A;
            }
        }
    }
}

// ==============================================================================
// 32-BIT FLOAT (BGRA) - PITCH IN PIXELS
// ==============================================================================
template <bool IS_OPAQUE, bool IS_PREMUL>
void planar2rgb
(
    const PF_Pixel_BGRA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_BGRA_32f* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch, // EXPECTS PIXELS
    A_long dstPitch  // EXPECTS PIXELS
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const float scaleDown = 1.0f / 255.0f;
    const __m256 v_scale = _mm256_set1_ps(scaleDown);

    const float maxClampVal = 1.0f - std::numeric_limits<float>::epsilon();
    const float opaqueAlphaVal = 1.0f;
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(maxClampVal);

    CACHE_ALIGN float r_out[8];
    CACHE_ALIGN float g_out[8];
    CACHE_ALIGN float b_out[8];

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_BGRA_32f* pSrcLine = pSrc + (static_cast<ptrdiff_t>(j) * srcPitch);
        PF_Pixel_BGRA_32f* pOutLine = pDst + (static_cast<ptrdiff_t>(j) * dstPitch);

        A_long i = 0;

        if (IS_OPAQUE)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale), v_zero), v_max);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale), v_zero), v_max);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale), v_zero), v_max);

                _mm256_store_ps(r_out, v_r);
                _mm256_store_ps(g_out, v_g);
                _mm256_store_ps(b_out, v_b);

                for (int k = 0; k < 8; ++k)
                {
                    pOutLine[i + k].B = b_out[k];
                    pOutLine[i + k].G = g_out[k];
                    pOutLine[i + k].R = r_out[k];
                    pOutLine[i + k].A = opaqueAlphaVal;
                }
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx] * scaleDown; float g = pG[idx] * scaleDown; float b = pB[idx] * scaleDown;
                pOutLine[i].B = b < 0.0f ? 0.0f : (b > maxClampVal ? maxClampVal : b);
                pOutLine[i].G = g < 0.0f ? 0.0f : (g > maxClampVal ? maxClampVal : g);
                pOutLine[i].R = r < 0.0f ? 0.0f : (r > maxClampVal ? maxClampVal : r);
                pOutLine[i].A = opaqueAlphaVal;
            }
        }
        else if (IS_PREMUL)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale);
                __m256 v_g = _mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale);
                __m256 v_b = _mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale);

                __m256 v_a = _mm256_set_ps(pSrcLine[i + 7].A, pSrcLine[i + 6].A, pSrcLine[i + 5].A, pSrcLine[i + 4].A,
                    pSrcLine[i + 3].A, pSrcLine[i + 2].A, pSrcLine[i + 1].A, pSrcLine[i + 0].A);

                v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_r, v_a), v_zero), v_max);
                v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_g, v_a), v_zero), v_max);
                v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_b, v_a), v_zero), v_max);

                _mm256_store_ps(r_out, v_r);
                _mm256_store_ps(g_out, v_g);
                _mm256_store_ps(b_out, v_b);

                for (int k = 0; k < 8; ++k)
                {
                    pOutLine[i + k].B = b_out[k];
                    pOutLine[i + k].G = g_out[k];
                    pOutLine[i + k].R = r_out[k];
                    pOutLine[i + k].A = pSrcLine[i + k].A;
                }
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float src_alpha = pSrcLine[i].A;
                float r = pR[idx] * scaleDown * src_alpha; float g = pG[idx] * scaleDown * src_alpha; float b = pB[idx] * scaleDown * src_alpha;
                pOutLine[i].B = b < 0.0f ? 0.0f : (b > maxClampVal ? maxClampVal : b);
                pOutLine[i].G = g < 0.0f ? 0.0f : (g > maxClampVal ? maxClampVal : g);
                pOutLine[i].R = r < 0.0f ? 0.0f : (r > maxClampVal ? maxClampVal : r);
                pOutLine[i].A = src_alpha;
            }
        }
        else
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale), v_zero), v_max);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale), v_zero), v_max);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale), v_zero), v_max);

                _mm256_store_ps(r_out, v_r);
                _mm256_store_ps(g_out, v_g);
                _mm256_store_ps(b_out, v_b);

                for (int k = 0; k < 8; ++k)
                {
                    pOutLine[i + k].B = b_out[k];
                    pOutLine[i + k].G = g_out[k];
                    pOutLine[i + k].R = r_out[k];
                    pOutLine[i + k].A = pSrcLine[i + k].A;
                }
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx] * scaleDown; float g = pG[idx] * scaleDown; float b = pB[idx] * scaleDown;
                pOutLine[i].B = b < 0.0f ? 0.0f : (b > maxClampVal ? maxClampVal : b);
                pOutLine[i].G = g < 0.0f ? 0.0f : (g > maxClampVal ? maxClampVal : g);
                pOutLine[i].R = r < 0.0f ? 0.0f : (r > maxClampVal ? maxClampVal : r);
                pOutLine[i].A = pSrcLine[i].A;
            }
        }
    }
}

// ==============================================================================
// 8-BIT (ARGB) - PITCH IN PIXELS
// ==============================================================================
template <bool IS_OPAQUE, bool IS_PREMUL>
void planar2rgb
(
    const PF_Pixel_ARGB_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_ARGB_8u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch, // EXPECTS PIXELS
    A_long dstPitch  // EXPECTS PIXELS
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    const __m256 v_inv255 = _mm256_set1_ps(1.0f / 255.0f);
    const __m256i v_alpha_mask = _mm256_set1_epi32(0x000000FF);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_8u* pSrcLine = pSrc + (static_cast<ptrdiff_t>(j) * srcPitch);
        PF_Pixel_ARGB_8u* pOutLine = pDst + (static_cast<ptrdiff_t>(j) * dstPitch);

        A_long i = 0;

        if (IS_OPAQUE)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pR[idx]), v_zero), v_255);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pG[idx]), v_zero), v_255);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pB[idx]), v_zero), v_255);

                __m256i v_r_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_r), 8);
                __m256i v_g_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_g), 16);
                __m256i v_b_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_b), 24);

                __m256i v_argb = _mm256_or_si256(v_b_i, _mm256_or_si256(v_g_i, _mm256_or_si256(v_r_i, v_alpha_mask)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_argb);
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx]; float g = pG[idx]; float b = pB[idx];
                pOutLine[i].A = 255;
                pOutLine[i].R = static_cast<A_u_char>(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
                pOutLine[i].G = static_cast<A_u_char>(g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g));
                pOutLine[i].B = static_cast<A_u_char>(b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b));
            }
        }
        else if (IS_PREMUL)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_loadu_ps(&pR[idx]);
                __m256 v_g = _mm256_loadu_ps(&pG[idx]);
                __m256 v_b = _mm256_loadu_ps(&pB[idx]);

                __m256i v_src_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pSrcLine[i]));
                __m256i v_out_alpha = _mm256_and_si256(v_src_pixels, v_alpha_mask);
                __m256 v_a_norm = _mm256_mul_ps(_mm256_cvtepi32_ps(v_out_alpha), v_inv255);

                v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_r, v_a_norm), v_zero), v_255);
                v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_g, v_a_norm), v_zero), v_255);
                v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_b, v_a_norm), v_zero), v_255);

                __m256i v_r_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_r), 8);
                __m256i v_g_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_g), 16);
                __m256i v_b_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_b), 24);

                __m256i v_argb = _mm256_or_si256(v_b_i, _mm256_or_si256(v_g_i, _mm256_or_si256(v_r_i, v_out_alpha)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_argb);
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                A_u_char a = pSrcLine[i].A;
                float a_norm = a * (1.0f / 255.0f);
                float r = pR[idx] * a_norm; float g = pG[idx] * a_norm; float b = pB[idx] * a_norm;
                pOutLine[i].A = a;
                pOutLine[i].R = static_cast<A_u_char>(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
                pOutLine[i].G = static_cast<A_u_char>(g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g));
                pOutLine[i].B = static_cast<A_u_char>(b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b));
            }
        }
        else
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pR[idx]), v_zero), v_255);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pG[idx]), v_zero), v_255);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&pB[idx]), v_zero), v_255);

                __m256i v_r_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_r), 8);
                __m256i v_g_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_g), 16);
                __m256i v_b_i = _mm256_slli_epi32(_mm256_cvtps_epi32(v_b), 24);

                __m256i v_src_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pSrcLine[i]));
                __m256i v_out_alpha = _mm256_and_si256(v_src_pixels, v_alpha_mask);

                __m256i v_argb = _mm256_or_si256(v_b_i, _mm256_or_si256(v_g_i, _mm256_or_si256(v_r_i, v_out_alpha)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_argb);
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx]; float g = pG[idx]; float b = pB[idx];
                pOutLine[i].A = pSrcLine[i].A;
                pOutLine[i].R = static_cast<A_u_char>(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
                pOutLine[i].G = static_cast<A_u_char>(g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g));
                pOutLine[i].B = static_cast<A_u_char>(b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b));
            }
        }
    }
}

// ==============================================================================
// 16-BIT (ARGB) - PITCH IN PIXELS
// ==============================================================================
template <bool IS_OPAQUE, bool IS_PREMUL>
void planar2rgb
(
    const PF_Pixel_ARGB_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_ARGB_16u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch, // EXPECTS PIXELS
    A_long dstPitch  // EXPECTS PIXELS
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const float scaleUp = 32767.0f / 255.0f;
    const __m256 v_scale = _mm256_set1_ps(scaleUp);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(32767.0f);
    const __m256 v_inv32767 = _mm256_set1_ps(1.0f / 32767.0f);

    CACHE_ALIGN int32_t r_out[8];
    CACHE_ALIGN int32_t g_out[8];
    CACHE_ALIGN int32_t b_out[8];

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_16u* pSrcLine = pSrc + (static_cast<ptrdiff_t>(j) * srcPitch);
        PF_Pixel_ARGB_16u* pOutLine = pDst + (static_cast<ptrdiff_t>(j) * dstPitch);

        A_long i = 0;

        if (IS_OPAQUE)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale), v_zero), v_max);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale), v_zero), v_max);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale), v_zero), v_max);

                _mm256_store_si256(reinterpret_cast<__m256i*>(r_out), _mm256_cvtps_epi32(v_r));
                _mm256_store_si256(reinterpret_cast<__m256i*>(g_out), _mm256_cvtps_epi32(v_g));
                _mm256_store_si256(reinterpret_cast<__m256i*>(b_out), _mm256_cvtps_epi32(v_b));

                for (int k = 0; k < 8; ++k)
                {
                    pOutLine[i + k].A = 32767;
                    pOutLine[i + k].R = static_cast<A_u_short>(r_out[k]);
                    pOutLine[i + k].G = static_cast<A_u_short>(g_out[k]);
                    pOutLine[i + k].B = static_cast<A_u_short>(b_out[k]);
                }
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx] * scaleUp; float g = pG[idx] * scaleUp; float b = pB[idx] * scaleUp;
                pOutLine[i].A = 32767;
                pOutLine[i].R = static_cast<A_u_short>(r < 0.0f ? 0.0f : (r > 32767.0f ? 32767.0f : r));
                pOutLine[i].G = static_cast<A_u_short>(g < 0.0f ? 0.0f : (g > 32767.0f ? 32767.0f : g));
                pOutLine[i].B = static_cast<A_u_short>(b < 0.0f ? 0.0f : (b > 32767.0f ? 32767.0f : b));
            }
        }
        else if (IS_PREMUL)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale);
                __m256 v_g = _mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale);
                __m256 v_b = _mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale);

                __m256 v_a_f = _mm256_set_ps(pSrcLine[i + 7].A, pSrcLine[i + 6].A, pSrcLine[i + 5].A, pSrcLine[i + 4].A,
                    pSrcLine[i + 3].A, pSrcLine[i + 2].A, pSrcLine[i + 1].A, pSrcLine[i + 0].A);

                __m256 v_a_norm = _mm256_mul_ps(v_a_f, v_inv32767);
                v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_r, v_a_norm), v_zero), v_max);
                v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_g, v_a_norm), v_zero), v_max);
                v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_b, v_a_norm), v_zero), v_max);

                _mm256_store_si256(reinterpret_cast<__m256i*>(r_out), _mm256_cvtps_epi32(v_r));
                _mm256_store_si256(reinterpret_cast<__m256i*>(g_out), _mm256_cvtps_epi32(v_g));
                _mm256_store_si256(reinterpret_cast<__m256i*>(b_out), _mm256_cvtps_epi32(v_b));

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
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                A_u_short a = pSrcLine[i].A;
                float a_norm = a * (1.0f / 32767.0f);
                float r = pR[idx] * scaleUp * a_norm; float g = pG[idx] * scaleUp * a_norm; float b = pB[idx] * scaleUp * a_norm;
                pOutLine[i].A = a;
                pOutLine[i].R = static_cast<A_u_short>(r < 0.0f ? 0.0f : (r > 32767.0f ? 32767.0f : r));
                pOutLine[i].G = static_cast<A_u_short>(g < 0.0f ? 0.0f : (g > 32767.0f ? 32767.0f : g));
                pOutLine[i].B = static_cast<A_u_short>(b < 0.0f ? 0.0f : (b > 32767.0f ? 32767.0f : b));
            }
        }
        else
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale), v_zero), v_max);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale), v_zero), v_max);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale), v_zero), v_max);

                _mm256_store_si256(reinterpret_cast<__m256i*>(r_out), _mm256_cvtps_epi32(v_r));
                _mm256_store_si256(reinterpret_cast<__m256i*>(g_out), _mm256_cvtps_epi32(v_g));
                _mm256_store_si256(reinterpret_cast<__m256i*>(b_out), _mm256_cvtps_epi32(v_b));

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
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx] * scaleUp; float g = pG[idx] * scaleUp; float b = pB[idx] * scaleUp;
                pOutLine[i].A = pSrcLine[i].A;
                pOutLine[i].R = static_cast<A_u_short>(r < 0.0f ? 0.0f : (r > 32767.0f ? 32767.0f : r));
                pOutLine[i].G = static_cast<A_u_short>(g < 0.0f ? 0.0f : (g > 32767.0f ? 32767.0f : g));
                pOutLine[i].B = static_cast<A_u_short>(b < 0.0f ? 0.0f : (b > 32767.0f ? 32767.0f : b));
            }
        }
    }
}

// ==============================================================================
// 32-BIT FLOAT (ARGB) - PITCH IN PIXELS
// ==============================================================================
template <bool IS_OPAQUE, bool IS_PREMUL>
void planar2rgb
(
    const PF_Pixel_ARGB_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_ARGB_32f* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch, // EXPECTS PIXELS
    A_long dstPitch  // EXPECTS PIXELS
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const float scaleDown = 1.0f / 255.0f;
    const __m256 v_scale = _mm256_set1_ps(scaleDown);

    const float maxClampVal = 1.0f - std::numeric_limits<float>::epsilon();
    const float opaqueAlphaVal = 1.0f;
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(maxClampVal);

    CACHE_ALIGN float r_out[8];
    CACHE_ALIGN float g_out[8];
    CACHE_ALIGN float b_out[8];

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_ARGB_32f* pSrcLine = pSrc + (static_cast<ptrdiff_t>(j) * srcPitch);
        PF_Pixel_ARGB_32f* pOutLine = pDst + (static_cast<ptrdiff_t>(j) * dstPitch);

        A_long i = 0;

        if (IS_OPAQUE)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale), v_zero), v_max);
                __m256 v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale), v_zero), v_max);
                __m256 v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale), v_zero), v_max);

                _mm256_store_ps(r_out, v_r);
                _mm256_store_ps(g_out, v_g);
                _mm256_store_ps(b_out, v_b);

                for (int k = 0; k < 8; ++k)
                {
                    pOutLine[i + k].A = opaqueAlphaVal;
                    pOutLine[i + k].R = r_out[k];
                    pOutLine[i + k].G = g_out[k];
                    pOutLine[i + k].B = b_out[k];
                }
            }
            for (; i < sizeX; i++)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx] * scaleDown; float g = pG[idx] * scaleDown; float b = pB[idx] * scaleDown;
                pOutLine[i].A = opaqueAlphaVal;
                pOutLine[i].R = r < 0.0f ? 0.0f : (r > maxClampVal ? maxClampVal : r);
                pOutLine[i].G = g < 0.0f ? 0.0f : (g > maxClampVal ? maxClampVal : g);
                pOutLine[i].B = b < 0.0f ? 0.0f : (b > maxClampVal ? maxClampVal : b);
            }
        }
        else if (IS_PREMUL)
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                __m256 v_r = _mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scale);
                __m256 v_g = _mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scale);
                __m256 v_b = _mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scale);

                __m256 v_a = _mm256_set_ps(pSrcLine[i + 7].A, pSrcLine[i + 6].A, pSrcLine[i + 5].A, pSrcLine[i + 4].A,
                    pSrcLine[i + 3].A, pSrcLine[i + 2].A, pSrcLine[i + 1].A, pSrcLine[i + 0].A);

                v_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_r, v_a), v_zero), v_max);
                v_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_g, v_a), v_zero), v_max);
                v_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v_b, v_a), v_zero), v_max);

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
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float src_alpha = pSrcLine[i].A;
                float r = pR[idx] * scaleDown * src_alpha; float g = pG[idx] * scaleDown * src_alpha; float b = pB[idx] * scaleDown * src_alpha;
                pOutLine[i].A = src_alpha;
                pOutLine[i].R = r < 0.0f ? 0.0f : (r > maxClampVal ? maxClampVal : r);
                pOutLine[i].G = g < 0.0f ? 0.0f : (g > maxClampVal ? maxClampVal : g);
                pOutLine[i].B = b < 0.0f ? 0.0f : (b > maxClampVal ? maxClampVal : b);
            }
        }
        else
        {
            for (; i < spanX8; i += 8)
            {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
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
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * sizeX + i;
                float r = pR[idx] * scaleDown; float g = pG[idx] * scaleDown; float b = pB[idx] * scaleDown;
                pOutLine[i].A = pSrcLine[i].A;
                pOutLine[i].R = r < 0.0f ? 0.0f : (r > maxClampVal ? maxClampVal : r);
                pOutLine[i].G = g < 0.0f ? 0.0f : (g > maxClampVal ? maxClampVal : g);
                pOutLine[i].B = b < 0.0f ? 0.0f : (b > maxClampVal ? maxClampVal : b);
            }
        }
    }
}