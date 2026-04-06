#include "PaintColorDispatcherOut.hpp"


// ============================================================================
// THE PUBLIC OUTPUT DISPATCH API
// ============================================================================
void dispatch_convert_to_interleaved
(
    const MemHandler& memHndl,
    const void* RESTRICT origSrcBuf, // The original Adobe Input buffer (for Alpha)
    void* RESTRICT dstBuf,           // The final Adobe Output buffer
    const A_long width,
    const A_long height,
    const A_long srcLinePitch,       // origSrcBuf line pitch in elements
    const A_long dstLinePitch,       // dstBuf line pitch in elements
    const PixelFormat format,
    const RenderQuality quality
) noexcept
{
    const bool isFast = (quality == RenderQuality::Fast_HalfSize);

    #define DISPATCH_INTERLEAVED_FMT(FMT_ENUM) \
        case FMT_ENUM: \
            if (isFast) { \
                convert_to_interleaved_AVX2<FMT_ENUM, true>(memHndl, origSrcBuf, dstBuf, width, height, srcLinePitch, dstLinePitch); \
            } else { \
                convert_to_interleaved_AVX2<FMT_ENUM, false>(memHndl, origSrcBuf, dstBuf, width, height, srcLinePitch, dstLinePitch); \
            } \
            break

    switch (format)
    {
        // Straight RGB Formats
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRA_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::ARGB_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRA_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::ARGB_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRA_32f);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::ARGB_32f);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::RGB_10u);
        
        // Aliases (X, P, and Linear)
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRX_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRP_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRX_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRP_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRX_32f);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRP_32f);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRA_32f_Linear);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRP_32f_Linear);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::BGRX_32f_Linear);

        // YUV Formats (Premiere)
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_8u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_16u);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_32f);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_32f);
        
        // YUV 709 Formats (Premiere Explicit)
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_8u_709);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_8u_709);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_16u_709);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_16u_709);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYA_32f_709);
        DISPATCH_INTERLEAVED_FMT(PixelFormat::VUYP_32f_709);

        default:
            break; 
    }

    #undef DISPATCH_INTERLEAVED_FMT
}