#include "PaintColorDispatcher.hpp"

// ============================================================================
// THE PUBLIC DISPATCH API
// ============================================================================
void dispatch_convert_to_planar
(
    const void* RESTRICT srcBuf,
    const MemHandler& memHndl,
    const A_long width,
    const A_long height,
    const A_long stride_pixels,
    const PixelFormat format,
    const RenderQuality quality
) noexcept
{
    // Evaluate our runtime boolean
    const bool isFast = (quality == RenderQuality::Fast_HalfSize);

    // Localized macro to keep the switch statement perfectly clean and readable.
    // It instantiates both the 'true' (HalfSize) and 'false' (FullSize) C++14 templates.
    #define DISPATCH_PLANAR_FMT(FMT_ENUM) \
        case FMT_ENUM: \
            if (isFast) { \
                convert_to_planar_AVX2<FMT_ENUM, true>(srcBuf, memHndl, width, height, stride_pixels); \
            } else { \
                convert_to_planar_AVX2<FMT_ENUM, false>(srcBuf, memHndl, width, height, stride_pixels); \
            } \
            break

    // Route the runtime Adobe format to the compile-time AVX2 engine
    switch (format)
    {
        DISPATCH_PLANAR_FMT(PixelFormat::BGRA_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRX_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRP_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::ARGB_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRA_16u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRX_16u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRP_16u);
        DISPATCH_PLANAR_FMT(PixelFormat::ARGB_16u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRA_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRX_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRP_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRA_32f_Linear);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRX_32f_Linear);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRP_32f_Linear);
        DISPATCH_PLANAR_FMT(PixelFormat::ARGB_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::RGB_10u);
        
        // YUV Interleaved Formats
        DISPATCH_PLANAR_FMT(PixelFormat::VUYA_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYP_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYA_16u);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYP_16u);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYA_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYP_32f);

        DISPATCH_PLANAR_FMT(PixelFormat::VUYA_8u_709);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYP_8u_709);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYA_16u_709);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYP_16u_709);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYA_32f_709);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYP_32f_709);

        default:
            // Optional: Log an error or fallback to a default format if Adobe 
            // feeds us something totally unexpected.
        break;
    }

    // Clean up the macro so it doesn't leak into the rest of your file
    #undef DISPATCH_PLANAR_FMT
}