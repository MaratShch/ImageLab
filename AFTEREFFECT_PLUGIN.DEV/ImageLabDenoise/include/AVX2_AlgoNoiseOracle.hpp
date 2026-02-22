#pragma once

#include <cstdint>
#include <immintrin.h>
#include "AlgoMemHandler.hpp"
#include "AlgoControls.hpp"

// =========================================================
// AVX2 ACCELERATED NOISE ORACLE
// =========================================================

// Executes the blind noise estimation using AVX2 for the DCT projections
void AVX2_Estimate_Noise_Covariances
(
    const MemHandler& mem,
    const int32_t width,
    const int32_t height,
    const AlgoControls& algoCtrl
);