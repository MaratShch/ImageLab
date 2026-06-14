#include "AlgConvertDispatcher.hpp"
#include "AlgConvertColorPlanar.hpp"

// ============================================================================
//  PUBLIC INGEST DISPATCH  (host -> planar linear RGB_32f)
// ============================================================================
void dispatch_convert_to_planar
(
    const void* RESTRICT srcBuf,
    const MemHandler&    memHndl,
    const A_long         width,
    const A_long         height,
    const A_long         stride_pixels,
    const PixelFormat    format
) noexcept
{
    #define DISPATCH_PLANAR_FMT(FMT_ENUM) \
        case FMT_ENUM: \
            convert_to_planar_AVX2<FMT_ENUM>(srcBuf, memHndl, width, height, stride_pixels); \
            break

    switch (format)
    {
        DISPATCH_PLANAR_FMT(PixelFormat::BGRA_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRA_16u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRA_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRA_32f_Linear);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRP_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRP_16u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRP_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRP_32f_Linear);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRX_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRX_16u);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRX_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::BGRX_32f_Linear);
        DISPATCH_PLANAR_FMT(PixelFormat::ARGB_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::ARGB_16u);
        DISPATCH_PLANAR_FMT(PixelFormat::ARGB_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::ARGB_32f_Linear);
        DISPATCH_PLANAR_FMT(PixelFormat::PRGB_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::PRGB_32f_Linear);
        DISPATCH_PLANAR_FMT(PixelFormat::XRGB_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::XRGB_32f_Linear);
        DISPATCH_PLANAR_FMT(PixelFormat::RGB_10u);

        DISPATCH_PLANAR_FMT(PixelFormat::VUYA_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYA_8u_709);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYA_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYA_32f_709);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYP_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYP_8u_709);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYP_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYP_32f_709);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYX_8u);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYX_8u_709);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYX_32f);
        DISPATCH_PLANAR_FMT(PixelFormat::VUYX_32f_709);

        default:
            break;
    }

    #undef DISPATCH_PLANAR_FMT
}
