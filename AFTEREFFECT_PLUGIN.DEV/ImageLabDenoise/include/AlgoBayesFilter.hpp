#pragma once

#include <cstdint>
#include "Common.hpp"
#include "AlgoMemHandler.hpp"

void Process_Scale_NL_Bayes
(
    const MemHandler& mem,
    float* RESTRICT Y_scale, 
    float* RESTRICT U_scale, 
    float* RESTRICT V_scale,
    const int32_t width, 
    const int32_t height,
    const float noise_variance_multiplier
);