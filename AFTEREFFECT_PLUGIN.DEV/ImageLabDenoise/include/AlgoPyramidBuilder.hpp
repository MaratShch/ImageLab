#pragma once

#include <cstdint>
#include "Common.hpp"
#include "AlgoMemHandler.hpp"

// Forward Building (Downscaling & Difference capturing)
void Build_Laplacian_Pyramid
(
    const MemHandler& mem,
    const int32_t srcWidth,
    const int32_t srcHeight
);

// Inverse Building (Upscaling & Details add-back)
void Reconstruct_Laplacian_Level
(
    const float* RESTRICT src_base,
    const float* RESTRICT src_diff,
    float* RESTRICT dst_reconstructed,
    const int32_t dstW,
    const int32_t dstH
);