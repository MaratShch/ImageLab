#pragma once
// ============================================================================
//  AlgConvertDispatcher.hpp -- runtime format -> compile-time ingest template
// ============================================================================

#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgConvertPixelTraits.hpp"   // PixelFormat

// Decode host frame (any supported format) -> memHndl.input.{R,G,B} (linear RGB_32f).
void dispatch_convert_to_planar
(
    const void* RESTRICT srcBuf,
    const MemHandler&    memHndl,
    const A_long         width,
    const A_long         height,
    const A_long         stride_pixels,   // host stride in pixels (signed)
    const PixelFormat    format
) noexcept;
