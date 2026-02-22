#pragma once

#include <cstdint>
#include <immintrin.h>
#include "Common.hpp"
#include "AlgoMemHandler.hpp"

// =========================================================
// AVX2 ACCELERATED LAPLACIAN PYRAMID
// =========================================================

void AVX2_Build_Laplacian_Pyramid
(
    const MemHandler& mem, 
    const int32_t srcWidth, 
    const int32_t srcHeight
);

void AVX2_Reconstruct_Laplacian_Level
(
    const float* RESTRICT src_base,
    const float* RESTRICT src_diff,
    float* RESTRICT dst_reconstructed,
    const int32_t dstW,
    const int32_t dstH
);