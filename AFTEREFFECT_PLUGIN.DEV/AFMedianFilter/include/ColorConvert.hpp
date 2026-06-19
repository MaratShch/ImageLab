#ifndef __IMAGE_LAB2_AFMF_ALGO_COLOR_CONVERT__
#define __IMAGE_LAB2_AFMF_ALGO_COLOR_CONVERT__

#include <immintrin.h>
#include <cstdint>
#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"
#include "AFMFPixelTraits.hpp"

void dispatch_convert_to_planar
(
    const void* RESTRICT srcBuf,
    const MemHandler& memHndl,
    const A_long width,
    const A_long height,
    const A_long stride_pixels,
    const PixelFormat format
) noexcept;


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
) noexcept;

#endif // __IMAGE_LAB2_AFMF_ALGO_COLOR_CONVERT__
