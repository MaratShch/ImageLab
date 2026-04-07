#pragma once

#include <type_traits>
#include "Common.hpp"
#include "DenoiseColorInterleaved.hpp"
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"

void dispatch_convert_to_interleaved
(
    const MemHandler& memHndl,
    const void* RESTRICT origSrcBuf, // The original Adobe Input buffer (for Alpha)
    void* RESTRICT dstBuf,           // The final Adobe Output buffer
    const A_long width,
    const A_long height,
    const A_long srcLinePitch,       // origSrcBuf line pitch in elements
    const A_long dstLinePitch,       // dstBuf line pitch in elements
    const PixelFormat format
) noexcept;