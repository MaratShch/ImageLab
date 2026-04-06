#pragma once

#include <type_traits>
#include "PaintColorPlanar.hpp"
#include "CommonPixFormat.hpp"
#include "PaintMemHandler.hpp"
#include "PaintAlgoContols.hpp"


void dispatch_convert_to_planar
(
    const void* RESTRICT srcBuf,
    const MemHandler& memHndl,
    const A_long width,
    const A_long height,
    const A_long stride_pixels,
    const PixelFormat format,
    const RenderQuality quality
) noexcept;
