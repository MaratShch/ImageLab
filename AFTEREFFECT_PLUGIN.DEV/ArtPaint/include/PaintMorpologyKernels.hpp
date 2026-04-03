#pragma once

#include <cstring>
#include "Common.hpp"
#include "AefxDevPatch.hpp"
#include "PaintMemHandler.hpp"

bool erode_max_plus_symmetric
(
    const float* RESTRICT imIn,
    float* RESTRICT imOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long nLines,
    const A_long frameSize
) noexcept;

bool dilate_max_plus_symmetric
(
    const float* RESTRICT imIn,
    float* RESTRICT imOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long nLines,
    const A_long frameSize
) noexcept;

int erode_max_plus_symmetric_iterated
(
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const float* RESTRICT imIn,
    float* RESTRICT imOut[],
    const A_long k,
    const A_long n_lines,
    float** pOut,
    const A_long frameSize
) noexcept;

int dilate_max_plus_symmetric_iterated
(
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const float* RESTRICT imIn,
    float* RESTRICT imOut[],
    const A_long k,
    const A_long n_lines,
    float** pOut,
    const A_long frameSize
) noexcept;

A_long morpho_open
(
    float* RESTRICT imIn,
    float* RESTRICT imOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    A_long it,
    A_long nonZeros,
    A_long sizeX,
    A_long sizeY,
    const MemHandler& memHndl
) noexcept;

A_long morpho_close
(
    float* RESTRICT imIn,
    float* RESTRICT imOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    A_long it,
    A_long nonZeros,
    A_long sizeX,
    A_long sizeY,
    const MemHandler& memHndl
) noexcept;

A_long morpho_asf
(
    float* RESTRICT imIn,
    float* RESTRICT imOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    A_long it,
    A_long nonZeros,
    A_long sizeX,
    A_long sizeY,
    const MemHandler& memHndl
) noexcept;
