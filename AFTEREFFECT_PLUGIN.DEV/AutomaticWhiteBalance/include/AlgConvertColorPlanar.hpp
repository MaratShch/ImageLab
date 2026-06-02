#pragma once
// ============================================================================
//  AlgConvertColorPlanar.hpp
//
//  Ingest: decode any supported host format -> planar SCENE-LINEAR RGB_32f,
//  written into memHndl.input.{R,G,B}. Planes are contiguous (stride == width).
//
//  Host row pitch is in PIXELS and SIGNED (may be negative for AE bottom-up
//  renders); row offsets are computed as (int64_t)y*(int64_t)pitch.
// ============================================================================

#include "Common.hpp"            // RESTRICT, CACHE_ALIGN
#include "CommonPixFormat.hpp"   // A_long
#include "AlgoMemHandler.hpp"    // MemHandler, RGBPlanes
#include "AlgConvertPixelTraits.hpp"

template <PixelFormat FMT>
void convert_to_planar_AVX2
(
    const void* RESTRICT srcBuf,
    const MemHandler&    memHndl,
    const A_long         width,
    const A_long         height,
    const A_long         stride_pixels   // host stride in pixels (signed, may be < 0)
) noexcept
{
    using Traits    = PixelTraits<FMT>;
    using PixelType = typename Traits::DataType;

    const PixelType* RESTRICT in_pixels = reinterpret_cast<const PixelType*>(srcBuf);
    float* RESTRICT out_R = memHndl.input.R;
    float* RESTRICT out_G = memHndl.input.G;
    float* RESTRICT out_B = memHndl.input.B;

    for (A_long y = 0; y < height; ++y)
    {
        const PixelType* RESTRICT in_row = in_pixels + (static_cast<int64_t>(y) * static_cast<int64_t>(stride_pixels));
        const int64_t o0 = static_cast<int64_t>(y) * static_cast<int64_t>(width); // contiguous plane row

        A_long x = 0;

        // 1) main AVX2 loop: 8 pixels / iteration
        for (; x <= width - 8; x += 8)
        {
            __m256 vB, vG, vR;
            Traits::LoadAVX2(in_row + x, vB, vG, vR);
            _mm256_storeu_ps(out_R + o0 + x, vR);
            _mm256_storeu_ps(out_G + o0 + x, vG);
            _mm256_storeu_ps(out_B + o0 + x, vB);
        }

        // 2) padded tail (< 8 pixels) via CACHE_ALIGN stack buffer
        const A_long remaining = width - x;
        if (remaining > 0)
        {
            CACHE_ALIGN PixelType tail_src[8] = {};
            for (A_long i = 0; i < remaining; ++i) { tail_src[i] = in_row[x + i]; }

            __m256 vB, vG, vR;
            Traits::LoadAVX2(tail_src, vB, vG, vR);

            CACHE_ALIGN float tR[8], tG[8], tB[8];
            _mm256_store_ps(tR, vR);
            _mm256_store_ps(tG, vG);
            _mm256_store_ps(tB, vB);

            for (A_long i = 0; i < remaining; ++i)
            {
                out_R[o0 + x + i] = tR[i];
                out_G[o0 + x + i] = tG[i];
                out_B[o0 + x + i] = tB[i];
            }
        }
    }
    return;
}
