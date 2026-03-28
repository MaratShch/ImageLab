#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "ImageAuxPixFormat.hpp"
#include "ImageMosaicUtils.hpp"
#ifdef _DEBUG
#include <cassert>
#endif


void ArtMosaic::fillProcBuf (Color* pBuf, const A_long pixNumber, const float val) noexcept
{
    constexpr A_long elemInStruct = sizeof(pBuf[0]) / sizeof(pBuf[0].r);
    const A_long rawSize = pixNumber * elemInStruct;
    const A_long rawSize24 = rawSize / 24;
    const A_long rawFract24 = rawSize % 24;

    float* pBufF = reinterpret_cast<float*>(pBuf);

    // Use set1_ps for cleaner broadcasting
    const __m256 fPattern = _mm256_set1_ps(val);

    for (A_long i = 0; i < rawSize24; i++)
    {
        _mm256_storeu_ps(pBufF, fPattern), pBufF += 8;
        _mm256_storeu_ps(pBufF, fPattern), pBufF += 8;
        _mm256_storeu_ps(pBufF, fPattern), pBufF += 8;
    }

    for (A_long i = 0; i < rawFract24; i++)
        pBufF[i] = val;

    return;
}

void ArtMosaic::fillProcBuf(std::unique_ptr<Color[]>& pBuf, const A_long pixNumber, const float val) noexcept
{
    ArtMosaic::fillProcBuf(pBuf.get(), pixNumber, val);
}

void ArtMosaic::fillProcBuf(A_long* pBuf, const A_long pixNumber, const A_long val) noexcept
{
    const A_long rawSize16 = pixNumber / 16;
    const A_long rawFract16 = pixNumber % 16;

    // Use set1_epi32
    const __m256i iPattern = _mm256_set1_epi32(val);
    __m256i* pBufAvxPtr = reinterpret_cast<__m256i*>(pBuf);

    for (A_long i = 0; i < rawSize16; i++)
    {
        // FIX: Changed to storeu (unaligned)
        _mm256_storeu_si256(pBufAvxPtr, iPattern), pBufAvxPtr++;
        _mm256_storeu_si256(pBufAvxPtr, iPattern), pBufAvxPtr++;
    }

    if (0 != rawFract16)
    {
        A_long* pBufPtr = reinterpret_cast<A_long*>(pBufAvxPtr);
        for (A_long i = 0; i < rawFract16; i++)
            pBufPtr[i] = val;
    }

    return;
}

void ArtMosaic::fillProcBuf(std::unique_ptr<A_long[]>& pBuf, const A_long pixNumber, const A_long val) noexcept
{
    ArtMosaic::fillProcBuf(pBuf.get(), pixNumber, val);
}


void ArtMosaic::fillProcBuf(float* pBuf, const A_long pixNumber, const float val) noexcept
{
    const A_long rawSize16 = pixNumber / 16;
    const A_long rawFract16 = pixNumber % 16;

    // Use set1_ps
    const __m256 fPattern = _mm256_set1_ps(val);

    // Better to cast to float* for the pointer arithmetic and storing
    float* pBufAvxPtr = pBuf;

    for (A_long i = 0; i < rawSize16; i++)
    {
        // FIX: Changed to storeu_ps (unaligned)
        _mm256_storeu_ps(pBufAvxPtr, fPattern), pBufAvxPtr += 8;
        _mm256_storeu_ps(pBufAvxPtr, fPattern), pBufAvxPtr += 8;
    }

    if (0 != rawFract16)
    {
        for (A_long i = 0; i < rawFract16; i++)
            pBufAvxPtr[i] = val;
    }

    return;
}


void ArtMosaic::fillProcBuf(std::unique_ptr<float[]>& pBuf, const A_long pixNumber, const float val) noexcept
{
    ArtMosaic::fillProcBuf(pBuf.get(), pixNumber, val);
}


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

void rgb2planar
(
    const PF_Pixel_BGRA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7; // Snap to nearest multiple of 8
    const __m256i mask_FF = _mm256_set1_epi32(0xFF);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_BGRA_8u* pLine = pSrc + j * linePitch;
        A_long i = 0;

        for (; i < spanX8; i += 8)
        {
            // Load 8 pixels (32 bytes) directly into an integer register
            __m256i v_bgra = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pLine[i]));

            // Extract channels assuming little-endian physical memory layout: 
            // Byte 0 = B, Byte 1 = G, Byte 2 = R, Byte 3 = A.
            // (If your struct layout differs physically, just swap these shift values: 0, 8, 16)
            __m256i v_b_int = _mm256_and_si256(v_bgra, mask_FF);
            __m256i v_g_int = _mm256_and_si256(_mm256_srli_epi32(v_bgra, 8), mask_FF);
            __m256i v_r_int = _mm256_and_si256(_mm256_srli_epi32(v_bgra, 16), mask_FF);

            // Convert to 32-bit floats
            __m256 v_b_f = _mm256_cvtepi32_ps(v_b_int);
            __m256 v_g_f = _mm256_cvtepi32_ps(v_g_int);
            __m256 v_r_f = _mm256_cvtepi32_ps(v_r_int);

            // Calculate linear memory index and store to planar buffers
            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pR[idx], v_r_f);
            _mm256_storeu_ps(&pG[idx], v_g_f);
            _mm256_storeu_ps(&pB[idx], v_b_f);
        }

        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            pR[idx] = static_cast<float>(pLine[i].R);
            pG[idx] = static_cast<float>(pLine[i].G);
            pB[idx] = static_cast<float>(pLine[i].B);
        }
    }
    return;
}


void planar2rgb
(
    const PF_Pixel_BGRA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_BGRA_8u* RESTRICT pDst,
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

    // Pre-calculate constants for AVX2 bounding
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);

    // Mask to extract ONLY the top byte (Alpha channel) from a 32-bit integer
    const __m256i v_alpha_mask = _mm256_set1_epi32(0xFF000000);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_BGRA_8u* pSrcLine = pSrc + j * srcPitch;
        PF_Pixel_BGRA_8u* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;

            // 1. Load planar floats
            __m256 v_r_f = _mm256_loadu_ps(&pR[idx]);
            __m256 v_g_f = _mm256_loadu_ps(&pG[idx]);
            __m256 v_b_f = _mm256_loadu_ps(&pB[idx]);

            // 2. Clamp to [0.0f, 255.0f]
            v_r_f = _mm256_min_ps(_mm256_max_ps(v_r_f, v_zero), v_255);
            v_g_f = _mm256_min_ps(_mm256_max_ps(v_g_f, v_zero), v_255);
            v_b_f = _mm256_min_ps(_mm256_max_ps(v_b_f, v_zero), v_255);

            // 3. Convert to 32-bit integers
            __m256i v_r_i = _mm256_cvtps_epi32(v_r_f);
            __m256i v_g_i = _mm256_cvtps_epi32(v_g_f);
            __m256i v_b_i = _mm256_cvtps_epi32(v_b_f);

            // 4. Bit-shift integers into correct BGRA byte positions
            v_r_i = _mm256_slli_epi32(v_r_i, 16);
            v_g_i = _mm256_slli_epi32(v_g_i, 8);
            // v_b_i requires no shift (stays at bottom byte)

            // 5. Load 8 original pixels from pSrc and isolate ONLY the Alpha channel
            __m256i v_src_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pSrcLine[i]));
            __m256i v_src_alpha = _mm256_and_si256(v_src_pixels, v_alpha_mask);

            // 6. Pack R, G, B, and the source Alpha into a single 32-bit block via Bitwise OR
            __m256i v_bgra = _mm256_or_si256(v_b_i,
                _mm256_or_si256(v_g_i,
                    _mm256_or_si256(v_r_i, v_src_alpha)));

            // 7. Store exactly 8 packed structs (32 bytes) back to memory
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_bgra);
        }

        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;

            float r = pR[idx];
            float g = pG[idx];
            float b = pB[idx];

            // Safe scalar clamping
            r = r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r);
            g = g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g);
            b = b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b);

            pOutLine[i].R = static_cast<A_u_char>(r);
            pOutLine[i].G = static_cast<A_u_char>(g);
            pOutLine[i].B = static_cast<A_u_char>(b);

            // Grab Alpha directly from the source buffer
            pOutLine[i].A = pSrcLine[i].A;
        }
    }

    return;
}

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

void vuya2planar
(
    const PF_Pixel_VUYA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    bool is709
)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256i mask_FF = _mm256_set1_epi32(0xFF);
    const __m256 v_128 = _mm256_set1_ps(128.0f);

    // Pre-load Matrix Coefficients to avoid branching in the loop
    const float c_RV = is709 ? 1.5748021f : 1.407500f;
    const float c_GU = is709 ? -0.18732698f : -0.344140f;
    const float c_GV = is709 ? -0.4681240f : -0.716900f;
    const float c_BU = is709 ? 1.85559927f : 1.779000f;

    const __m256 v_cRV = _mm256_set1_ps(c_RV);
    const __m256 v_cGU = _mm256_set1_ps(c_GU);
    const __m256 v_cGV = _mm256_set1_ps(c_GV);
    const __m256 v_cBU = _mm256_set1_ps(c_BU);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_VUYA_8u* pLine = pSrc + j * srcPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Memory layout: V (Byte 0), U (Byte 1), Y (Byte 2), A (Byte 3)
            __m256i v_vuya = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pLine[i]));

            __m256i v_v_int = _mm256_and_si256(v_vuya, mask_FF);
            __m256i v_u_int = _mm256_and_si256(_mm256_srli_epi32(v_vuya, 8), mask_FF);
            __m256i v_y_int = _mm256_and_si256(_mm256_srli_epi32(v_vuya, 16), mask_FF);

            // Convert to floats
            __m256 v_y_f = _mm256_cvtepi32_ps(v_y_int);
            __m256 v_u_f = _mm256_sub_ps(_mm256_cvtepi32_ps(v_u_int), v_128); // U - 128
            __m256 v_v_f = _mm256_sub_ps(_mm256_cvtepi32_ps(v_v_int), v_128); // V - 128

                                                                              // Calculate RGB using Fused Multiply-Add (a * b + c) for maximum speed
            __m256 v_r = _mm256_fmadd_ps(v_cRV, v_v_f, v_y_f); // R = Y + c_RV * V'

            __m256 v_g = _mm256_fmadd_ps(v_cGU, v_u_f, v_y_f); // G = Y + c_GU * U'
            v_g = _mm256_fmadd_ps(v_cGV, v_v_f, v_g);          // G = G + c_GV * V'

            __m256 v_b = _mm256_fmadd_ps(v_cBU, v_u_f, v_y_f); // B = Y + c_BU * U'

                                                               // Store to planar buffers
            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pR[idx], v_r);
            _mm256_storeu_ps(&pG[idx], v_g);
            _mm256_storeu_ps(&pB[idx], v_b);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            float y = static_cast<float>(pLine[i].Y);
            float u = static_cast<float>(pLine[i].U) - 128.0f;
            float v = static_cast<float>(pLine[i].V) - 128.0f;

            pR[j * sizeX + i] = y + c_RV * v;
            pG[j * sizeX + i] = y + c_GU * u + c_GV * v;
            pB[j * sizeX + i] = y + c_BU * u;
        }
    }

    return;
}


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
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    const __m256 v_128 = _mm256_set1_ps(128.0f);
    const __m256i v_alpha_mask = _mm256_set1_epi32(0xFF000000);

    // Pre-load Forward Matrix Coefficients
    float cY_R, cY_G, cY_B, cU_R, cU_G, cU_B, cV_R, cV_G, cV_B;
    if (true == is709)
    {
        cY_R = 0.212600f;  cY_G = 0.715200f;   cY_B = 0.072200f;
        cU_R = -0.114570f;  cU_G = -0.385430f;  cU_B = 0.500000f;
        cV_R = 0.500000f;  cV_G = -0.454150f;  cV_B = -0.045850f;
    }
    else
    {
        cY_R = 0.299000f;  cY_G = 0.587000f;  cY_B = 0.114000f;
        cU_R = -0.168736f;  cU_G = -0.331264f;  cU_B = 0.500000f;
        cV_R = 0.500000f;  cV_G = -0.418688f;  cV_B = -0.081312f;
    }

    const __m256 v_cY_R = _mm256_set1_ps(cY_R), v_cY_G = _mm256_set1_ps(cY_G), v_cY_B = _mm256_set1_ps(cY_B);
    const __m256 v_cU_R = _mm256_set1_ps(cU_R), v_cU_G = _mm256_set1_ps(cU_G), v_cU_B = _mm256_set1_ps(cU_B);
    const __m256 v_cV_R = _mm256_set1_ps(cV_R), v_cV_G = _mm256_set1_ps(cV_G), v_cV_B = _mm256_set1_ps(cV_B);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_VUYA_8u* pSrcLine = pSrc + j * srcPitch;
        PF_Pixel_VUYA_8u* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;

            __m256 v_r_f = _mm256_loadu_ps(&pR[idx]);
            __m256 v_g_f = _mm256_loadu_ps(&pG[idx]);
            __m256 v_b_f = _mm256_loadu_ps(&pB[idx]);

            // Y = cY_R*R + cY_G*G + cY_B*B
            __m256 v_y_f = _mm256_mul_ps(v_cY_R, v_r_f);
            v_y_f = _mm256_fmadd_ps(v_cY_G, v_g_f, v_y_f);
            v_y_f = _mm256_fmadd_ps(v_cY_B, v_b_f, v_y_f);

            // U = cU_R*R + cU_G*G + cU_B*B + 128.0f
            __m256 v_u_f = _mm256_fmadd_ps(v_cU_R, v_r_f, v_128);
            v_u_f = _mm256_fmadd_ps(v_cU_G, v_g_f, v_u_f);
            v_u_f = _mm256_fmadd_ps(v_cU_B, v_b_f, v_u_f);

            // V = cV_R*R + cV_G*G + cV_B*B + 128.0f
            __m256 v_v_f = _mm256_fmadd_ps(v_cV_R, v_r_f, v_128);
            v_v_f = _mm256_fmadd_ps(v_cV_G, v_g_f, v_v_f);
            v_v_f = _mm256_fmadd_ps(v_cV_B, v_b_f, v_v_f);

            // Clamp safely between [0.0f, 255.0f]
            v_y_f = _mm256_min_ps(_mm256_max_ps(v_y_f, v_zero), v_255);
            v_u_f = _mm256_min_ps(_mm256_max_ps(v_u_f, v_zero), v_255);
            v_v_f = _mm256_min_ps(_mm256_max_ps(v_v_f, v_zero), v_255);

            // Convert to integers
            __m256i v_y_i = _mm256_cvtps_epi32(v_y_f);
            __m256i v_u_i = _mm256_cvtps_epi32(v_u_f);
            __m256i v_v_i = _mm256_cvtps_epi32(v_v_f);

            // Bit-shift into correct VUYA byte positions (V=0, U=8, Y=16)
            v_u_i = _mm256_slli_epi32(v_u_i, 8);
            v_y_i = _mm256_slli_epi32(v_y_i, 16);

            // Extract original Alpha from pSrc
            __m256i v_src_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pSrcLine[i]));
            __m256i v_src_alpha = _mm256_and_si256(v_src_pixels, v_alpha_mask);

            // Pack V, U, Y, and A into a single block
            __m256i v_vuya = _mm256_or_si256(v_v_i, _mm256_or_si256(v_u_i, _mm256_or_si256(v_y_i, v_src_alpha)));

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_vuya);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            float r = pR[idx];
            float g = pG[idx];
            float b = pB[idx];

            float y = cY_R * r + cY_G * g + cY_B * b;
            float u = cU_R * r + cU_G * g + cU_B * b + 128.0f;
            float v = cV_R * r + cV_G * g + cV_B * b + 128.0f;

            y = y < 0.0f ? 0.0f : (y > 255.0f ? 255.0f : y);
            u = u < 0.0f ? 0.0f : (u > 255.0f ? 255.0f : u);
            v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);

            pOutLine[i].V = static_cast<A_u_char>(v);
            pOutLine[i].U = static_cast<A_u_char>(u);
            pOutLine[i].Y = static_cast<A_u_char>(y);
            pOutLine[i].A = pSrcLine[i].A;
        }
    }

    return;
}

void vuya2planar
(
    const PF_Pixel_VUYA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    bool is709
)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_scale255 = _mm256_set1_ps(255.0f);

    // Pre-load Matrix Coefficients (BT.709 or BT.601)
    const float c_RV = is709 ? 1.5748021f : 1.407500f;
    const float c_GU = is709 ? -0.18732698f : -0.344140f;
    const float c_GV = is709 ? -0.4681240f : -0.716900f;
    const float c_BU = is709 ? 1.85559927f : 1.779000f;

    const __m256 v_cRV = _mm256_set1_ps(c_RV);
    const __m256 v_cGU = _mm256_set1_ps(c_GU);
    const __m256 v_cGV = _mm256_set1_ps(c_GV);
    const __m256 v_cBU = _mm256_set1_ps(c_BU);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_VUYA_32f* pLine = pSrc + j * srcPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Gather 32-bit floats directly. 
            // Note: _mm256_set_ps takes arguments in reverse order (highest to lowest index)
            __m256 v_v = _mm256_set_ps(pLine[i + 7].V, pLine[i + 6].V, pLine[i + 5].V, pLine[i + 4].V, pLine[i + 3].V, pLine[i + 2].V, pLine[i + 1].V, pLine[i].V);
            __m256 v_u = _mm256_set_ps(pLine[i + 7].U, pLine[i + 6].U, pLine[i + 5].U, pLine[i + 4].U, pLine[i + 3].U, pLine[i + 2].U, pLine[i + 1].U, pLine[i].U);
            __m256 v_y = _mm256_set_ps(pLine[i + 7].Y, pLine[i + 6].Y, pLine[i + 5].Y, pLine[i + 4].Y, pLine[i + 3].Y, pLine[i + 2].Y, pLine[i + 1].Y, pLine[i].Y);

            // Remove chroma bias for 32f color space: U' = U - 0.5, V' = V - 0.5
            v_u = _mm256_sub_ps(v_u, v_half);
            v_v = _mm256_sub_ps(v_v, v_half);

            // Reconstruct RGB [0.0f - 1.0f] via FMA
            __m256 v_r = _mm256_fmadd_ps(v_cRV, v_v, v_y);

            __m256 v_g = _mm256_fmadd_ps(v_cGU, v_u, v_y);
            v_g = _mm256_fmadd_ps(v_cGV, v_v, v_g);

            __m256 v_b = _mm256_fmadd_ps(v_cBU, v_u, v_y);

            // Scale up to your planar buffer range [0.0f - 255.0f]
            v_r = _mm256_mul_ps(v_r, v_scale255);
            v_g = _mm256_mul_ps(v_g, v_scale255);
            v_b = _mm256_mul_ps(v_b, v_scale255);

            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pR[idx], v_r);
            _mm256_storeu_ps(&pG[idx], v_g);
            _mm256_storeu_ps(&pB[idx], v_b);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            float y = pLine[i].Y;
            float u = pLine[i].U - 0.5f;
            float v = pLine[i].V - 0.5f;

            float r = y + c_RV * v;
            float g = y + c_GU * u + c_GV * v;
            float b = y + c_BU * u;

            pR[j * sizeX + i] = r * 255.0f;
            pG[j * sizeX + i] = g * 255.0f;
            pB[j * sizeX + i] = b * 255.0f;
        }
    }

    return;
}


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
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_half = _mm256_set1_ps(0.5f);

    constexpr float scaleDown = 1.0f / 255.0f;
    const __m256 v_scaleDown = _mm256_set1_ps(scaleDown);

    // Clamp limits: enforce 1.0f - FLT_EPSILON per your safety rule
    const float maxClampVal = 1.0f - std::numeric_limits<float>::epsilon();
    const __m256 v_max = _mm256_set1_ps(maxClampVal);

    // Pre-load Forward Matrix Coefficients
    float cY_R, cY_G, cY_B, cU_R, cU_G, cU_B, cV_R, cV_G, cV_B;
    if (true == is709)
    {
        cY_R = 0.212600f;  cY_G = 0.715200f;  cY_B = 0.072200f;
        cU_R = -0.114570f;  cU_G = -0.385430f;  cU_B = 0.500000f;
        cV_R = 0.500000f;  cV_G = -0.454150f;  cV_B = -0.045850f;
    }
    else
    {
        cY_R = 0.299000f;  cY_G = 0.587000f;  cY_B = 0.114000f;
        cU_R = -0.168736f;  cU_G = -0.331264f;  cU_B = 0.500000f;
        cV_R = 0.500000f;  cV_G = -0.418688f;  cV_B = -0.081312f;
    }

    const __m256 v_cY_R = _mm256_set1_ps(cY_R), v_cY_G = _mm256_set1_ps(cY_G), v_cY_B = _mm256_set1_ps(cY_B);
    const __m256 v_cU_R = _mm256_set1_ps(cU_R), v_cU_G = _mm256_set1_ps(cU_G), v_cU_B = _mm256_set1_ps(cU_B);
    const __m256 v_cV_R = _mm256_set1_ps(cV_R), v_cV_G = _mm256_set1_ps(cV_G), v_cV_B = _mm256_set1_ps(cV_B);

    // Aligned temp arrays for scatter
    CACHE_ALIGN float y_out[8];
    CACHE_ALIGN float u_out[8];
    CACHE_ALIGN float v_out[8];

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_VUYA_32f* pSrcLine = pSrc + j * srcPitch;
        PF_Pixel_VUYA_32f* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;

            // Load planar and scale down from [0.0 - 255.0] to [0.0 - 1.0]
            __m256 v_r_f = _mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scaleDown);
            __m256 v_g_f = _mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scaleDown);
            __m256 v_b_f = _mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scaleDown);

            // Y = cY_R*R + cY_G*G + cY_B*B
            __m256 v_y_f = _mm256_mul_ps(v_cY_R, v_r_f);
            v_y_f = _mm256_fmadd_ps(v_cY_G, v_g_f, v_y_f);
            v_y_f = _mm256_fmadd_ps(v_cY_B, v_b_f, v_y_f);

            // U = cU_R*R + cU_G*G + cU_B*B + 0.5f (Bias)
            __m256 v_u_f = _mm256_fmadd_ps(v_cU_R, v_r_f, v_half);
            v_u_f = _mm256_fmadd_ps(v_cU_G, v_g_f, v_u_f);
            v_u_f = _mm256_fmadd_ps(v_cU_B, v_b_f, v_u_f);

            // V = cV_R*R + cV_G*G + cV_B*B + 0.5f (Bias)
            __m256 v_v_f = _mm256_fmadd_ps(v_cV_R, v_r_f, v_half);
            v_v_f = _mm256_fmadd_ps(v_cV_G, v_g_f, v_v_f);
            v_v_f = _mm256_fmadd_ps(v_cV_B, v_b_f, v_v_f);

            // Clamp safely between 0.0f and (1.0f - FLT_EPSILON)
            v_y_f = _mm256_min_ps(_mm256_max_ps(v_y_f, v_zero), v_max);
            v_u_f = _mm256_min_ps(_mm256_max_ps(v_u_f, v_zero), v_max);
            v_v_f = _mm256_min_ps(_mm256_max_ps(v_v_f, v_zero), v_max);

            // Store locally
            _mm256_store_ps(y_out, v_y_f);
            _mm256_store_ps(u_out, v_u_f);
            _mm256_store_ps(v_out, v_v_f);

            // Pack struct and copy original Alpha
            for (int k = 0; k < 8; ++k)
            {
                pOutLine[i + k].V = v_out[k];
                pOutLine[i + k].U = u_out[k];
                pOutLine[i + k].Y = y_out[k];
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

            float y = cY_R * r + cY_G * g + cY_B * b;
            float u = cU_R * r + cU_G * g + cU_B * b + 0.5f;
            float v = cV_R * r + cV_G * g + cV_B * b + 0.5f;

            y = y < 0.0f ? 0.0f : (y > maxClampVal ? maxClampVal : y);
            u = u < 0.0f ? 0.0f : (u > maxClampVal ? maxClampVal : u);
            v = v < 0.0f ? 0.0f : (v > maxClampVal ? maxClampVal : v);

            pOutLine[i].V = v;
            pOutLine[i].U = u;
            pOutLine[i].Y = y;
            pOutLine[i].A = pSrcLine[i].A;
        }
    }

    return;
}
