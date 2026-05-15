#include "AFMFColorInterleaved.hpp"

// ============================================================================
// THE PUBLIC DISPATCH API (Planar RGB to Interleaved Output)
// ============================================================================
void dispatch_convert_to_interleaved
(
    const MemHandler& memHndl,
    const void* RESTRICT origSrcBuf,
    void* RESTRICT dstBuf,
    const A_long width,
    const A_long height,
    const A_long srcLinePitch,
    const A_long dstLinePitch,
    const PixelFormat format
) noexcept
{
    // Localized macro to instantiate the C++14 AVX2 Repack template
#define DISPATCH_INTERLEAVED_FMT(FMT_ENUM) \
        case FMT_ENUM: \
            convert_to_interleaved_AVX2<FMT_ENUM>(memHndl, origSrcBuf, dstBuf, width, height, srcLinePitch, dstLinePitch); \
            break

    switch (format)
    {
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRA_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRX_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRP_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::ARGB_8u);

        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRA_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRX_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRP_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::ARGB_16u);

        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRA_32f);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRX_32f);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRP_32f);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRA_32f_Linear);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRX_32f_Linear);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRP_32f_Linear);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::ARGB_32f);

        DISPATCH_INTERLEAVED_FMT(PixelFormat::RGB_10u);

        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_32f);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_32f);

        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_8u_709);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_8u_709);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_16u_709);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_16u_709);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_32f_709);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_32f_709);

    default:
        // Fallback safety block
        break;
    }

#undef DISPATCH_INTERLEAVED_FMT
}