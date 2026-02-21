#pragma once

#include <cstdint>
#include "Common.hpp"
#include "AlgoMemHandler.hpp"

// Calculates the L2 distance between two 4x4x3 patches
float Calculate_Patch_Distance
(
    const float* RESTRICT Y_plane,
    const float* RESTRICT U_plane,
    const float* RESTRICT V_plane,
    const int32_t ref_x,
    const int32_t ref_y,
    const int32_t tgt_x,
    const int32_t tgt_y,
    const int32_t width
) noexcept;

// Extracts and groups similar patches within a 17x17 window
// Returns the number of validated similar patches (N)
int32_t Extract_Similar_Patches
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

inline float Calculate_Patch_Distance
(
    const float* RESTRICT Y_plane,
    const float* RESTRICT U_plane,
    const float* RESTRICT V_plane,
    const int32_t ref_x, 
	const int32_t ref_y,
    const int32_t tgt_x, 
	const int32_t tgt_y,
    const int32_t width
) noexcept
{
    float diff = 0.0f;
    
    // Hardcoded for kappa = 4
    for (int32_t i = 0; i < 4; ++i) 
    {
        const int32_t row_ref = (ref_y + i) * width;
        const int32_t row_tgt = (tgt_y + i) * width;
        
        for (int32_t j = 0; j < 4; ++j) 
        {
            const int32_t idx_ref = row_ref + ref_x + j;
            const int32_t idx_tgt = row_tgt + tgt_x + j;
            
            const float dY = Y_plane[idx_ref] - Y_plane[idx_tgt];
            const float dU = U_plane[idx_ref] - U_plane[idx_tgt];
            const float dV = V_plane[idx_ref] - V_plane[idx_tgt];
            
            diff += (dY * dY) + (dU * dU) + (dV * dV);
        }
    }
    
    return diff;
}

