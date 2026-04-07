#pragma once

#include <type_traits>
#include "Common.hpp"
#include "DenoiseColorPlanar.hpp"
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"

void dispatch_convert_to_planar
(
    const void* RESTRICT srcBuf,
    const MemHandler& memHndl,
    const A_long width,
    const A_long height,
    const A_long stride_pixels,
    const PixelFormat format
) noexcept;