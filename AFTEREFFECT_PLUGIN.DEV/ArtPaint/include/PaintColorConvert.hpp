#pragma once

#include <immintrin.h>
#include <algorithm>
#include "Common.hpp"
#include "AefxDevPatch.hpp"
#include "CommonPixFormat.hpp"
#include "PaintMemHandler.hpp"


void convert_BGRA8_to_PlanarYUV_AVX2
(
    const PF_Pixel_BGRA_8u* RESTRICT srcBuf,
    const MemHandler& memHandler,
    const A_long width,
    const A_long height,
    const A_long stride_pixels
) noexcept;

void convert_PlanarYUV_to_BGRA8_AVX2
(
    const MemHandler& memHandler,
    const PF_Pixel_BGRA_8u* RESTRICT srcBuf,
    PF_Pixel_BGRA_8u* RESTRICT dstBuf,
    const A_long width,
    const A_long height,
    const A_long src_stride_pixels,
    const A_long dst_stride_pixels
) noexcept;
