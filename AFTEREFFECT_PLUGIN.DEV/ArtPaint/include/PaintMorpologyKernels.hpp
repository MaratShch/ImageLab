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

void morpho_open
(
    float* RESTRICT imInOut, 
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept;

void morpho_close
(
    float* RESTRICT imInOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept;

void morpho_asf
(
    float* RESTRICT imInOut,
    const A_long* RESTRICT I,
    const A_long* RESTRICT J,
    const A_long iter,
    const A_long nonZeros,
    const A_long width,
    const A_long height,
    const MemHandler& memHndl
) noexcept;
