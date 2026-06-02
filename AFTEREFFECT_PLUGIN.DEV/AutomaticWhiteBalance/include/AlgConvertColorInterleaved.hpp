#pragma once
// ============================================================================
//  AlgConvertColorInterleaved.hpp
//
//  Egress: read planar SCENE-LINEAR RGB_32f from memHndl.output.{R,G,B}
//  (contiguous, stride == width) -> encode to any supported host format.
//
//  origSrcBuf is the ORIGINAL host input frame, used only for alpha passthrough
//  (and ignored by BGRX / RGB_10u). srcLinePitch and dstLinePitch are in PIXELS
//  and SIGNED (either may be negative); row offsets use (int64_t)y*(int64_t)pitch.
// ============================================================================

#include "Common.hpp"            // RESTRICT, CACHE_ALIGN
#include "CommonPixFormat.hpp"   // A_long
#include "AlgoMemHandler.hpp"    // MemHandler, RGBPlanes
#include "AlgConvertPixelTraits.hpp"

template <PixelFormat FMT>
void convert_to_interleaved_AVX2
(
    const MemHandler&    memHndl,
    const void* RESTRICT origSrcBuf,    // original host input frame (alpha source)
    void*       RESTRICT dstBuf,        // host output frame
    const A_long         width,
    const A_long         height,
    const A_long         srcLinePitch,  // origSrcBuf stride in pixels (signed)
    const A_long         dstLinePitch   // dstBuf stride in pixels (signed)
) noexcept
{
    using Traits    = PixelTraits<FMT>;
    using PixelType = typename Traits::DataType;

    const float* RESTRICT in_R = memHndl.output.R;
    const float* RESTRICT in_G = memHndl.output.G;
    const float* RESTRICT in_B = memHndl.output.B;

    const PixelType* RESTRICT orig_pixels = reinterpret_cast<const PixelType*>(origSrcBuf);
    PixelType*       RESTRICT out_pixels  = reinterpret_cast<PixelType*>(dstBuf);

    for (A_long y = 0; y < height; ++y)
    {
        const int64_t o0 = static_cast<int64_t>(y) * static_cast<int64_t>(width); // contiguous plane row
        const PixelType* RESTRICT orig_row = orig_pixels + (static_cast<int64_t>(y) * static_cast<int64_t>(srcLinePitch));
        PixelType*       RESTRICT out_row  = out_pixels  + (static_cast<int64_t>(y) * static_cast<int64_t>(dstLinePitch));

        A_long x = 0;

        // 1) main AVX2 loop: 8 pixels / iteration
        for (; x <= width - 8; x += 8)
        {
            __m256 vR = _mm256_loadu_ps(in_R + o0 + x);
            __m256 vG = _mm256_loadu_ps(in_G + o0 + x);
            __m256 vB = _mm256_loadu_ps(in_B + o0 + x);
            Traits::StoreAVX2(out_row + x, vB, vG, vR, orig_row + x);
        }

        // 2) padded tail (< 8 pixels)
        const A_long remaining = width - x;
        if (remaining > 0)
        {
            CACHE_ALIGN float tR[8] = {0}, tG[8] = {0}, tB[8] = {0};
            CACHE_ALIGN PixelType tail_orig[8] = {}, tail_out[8] = {};

            for (A_long i = 0; i < remaining; ++i)
            {
                tR[i] = in_R[o0 + x + i];
                tG[i] = in_G[o0 + x + i];
                tB[i] = in_B[o0 + x + i];
                tail_orig[i] = orig_row[x + i];
            }

            __m256 vR = _mm256_load_ps(tR);
            __m256 vG = _mm256_load_ps(tG);
            __m256 vB = _mm256_load_ps(tB);
            Traits::StoreAVX2(tail_out, vB, vG, vR, tail_orig);

            for (A_long i = 0; i < remaining; ++i) { out_row[x + i] = tail_out[i]; }
        }
    }
    return;
}
