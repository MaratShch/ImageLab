#pragma once

#include <type_traits>
#include "Common.hpp"
#include "AlgoColorInterleaved.hpp"
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"

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