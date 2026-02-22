#pragma once

#include <cstdint>
#include <immintrin.h>
#include "Common.hpp"
#include "AlgoMemHandler.hpp"

// =========================================================
// AVX2 ACCELERATED BLOCK MATCHING API
// =========================================================

// Calculates the L2 distance between two 4x4x3 patches using 256-bit SIMD FMA
float AVX2_Calculate_Patch_Distance
(
    const float* RESTRICT Y_plane,
    const float* RESTRICT U_plane,
    const float* RESTRICT V_plane,
    const int32_t ref_x, const int32_t ref_y,
    const int32_t tgt_x, const int32_t tgt_y,
    const int32_t width
) noexcept;

// Extracts and groups similar patches using the AVX2 distance metric
int32_t AVX2_Extract_Similar_Patches
(
    const float* RESTRICT Y_plane,
    const float* RESTRICT U_plane,
    const float* RESTRICT V_plane,
    const int32_t width,
    const int32_t height,
    const int32_t ref_x,
    const int32_t ref_y,
    const float tau_sigma_sq, 
    PatchDistance* RESTRICT search_pool
);