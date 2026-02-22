#pragma once

#include <cstdint>
#include <immintrin.h>
#include "Common.hpp"
#include "AlgoMemHandler.hpp"

// =========================================================
// AVX2 ACCELERATED BAYESIAN FILTER CORE
// =========================================================

// Executes the dual-pass Non-Local Bayes filter using AVX2 intrinsics.
// Must be drop-in interchangeable with Process_Scale_NL_Bayes.
void AVX2_Process_Scale_NL_Bayes
(
    const MemHandler& mem,
    float* RESTRICT Y_scale, 
    float* RESTRICT U_scale, 
    float* RESTRICT V_scale,
    const int32_t width, 
    const int32_t height,
    const float noise_variance_multiplier
);