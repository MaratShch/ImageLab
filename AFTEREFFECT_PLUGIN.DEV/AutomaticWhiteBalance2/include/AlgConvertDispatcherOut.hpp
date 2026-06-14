#pragma once
// ============================================================================
//  AlgConvertDispatcherOut.hpp -- runtime format -> compile-time egress template
// ============================================================================

#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgConvertPixelTraits.hpp"   // PixelFormat

// Encode memHndl.output.{R,G,B} (linear RGB_32f) -> host frame (any supported format).
// origSrcBuf is the original host input frame (alpha passthrough source).
void dispatch_convert_to_interleaved
(
    const MemHandler&    memHndl,
    const void* RESTRICT origSrcBuf,
    void*       RESTRICT dstBuf,
    const A_long         width,
    const A_long         height,
    const A_long         srcLinePitch,   // origSrcBuf stride in pixels (signed)
    const A_long         dstLinePitch,   // dstBuf stride in pixels (signed)
    const PixelFormat    format
) noexcept;
