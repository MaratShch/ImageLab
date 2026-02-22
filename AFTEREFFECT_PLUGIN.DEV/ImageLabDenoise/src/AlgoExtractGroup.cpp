#include <algorithm>
//#include "AlgoBlockMatch.hpp"
#include "AVX2_AlgoBlockMatch.hpp"

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
)
{
    int32_t pool_count = 0;
    
    // sizeSearchWindow = 17, radius = 8
    constexpr int32_t radius = 8;
    constexpr int32_t patch_size = 4;
    
    const int32_t min_x = std::max(0, ref_x - radius);
    const int32_t max_x = std::min(width - patch_size, ref_x + radius);
    const int32_t min_y = std::max(0, ref_y - radius);
    const int32_t max_y = std::min(height - patch_size, ref_y + radius);

    // 1. Calculate distances for all patches in the 17x17 window
    for (int32_t tgt_y = min_y; tgt_y <= max_y; ++tgt_y) 
    {
        for (int32_t tgt_x = min_x; tgt_x <= max_x; ++tgt_x) 
        {
            // The reference patch is exactly distance 0
            if (tgt_x == ref_x && tgt_y == ref_y) 
            {
                search_pool[pool_count].x = tgt_x;
                search_pool[pool_count].y = tgt_y;
                search_pool[pool_count].distance = 0.0f;
                pool_count++;
                continue;
            }

            float dist = AVX2_Calculate_Patch_Distance
            (
                Y_plane, U_plane, V_plane, 
                ref_x, ref_y, tgt_x, tgt_y, width
            );
            
            search_pool[pool_count].x = tgt_x;
            search_pool[pool_count].y = tgt_y;
            search_pool[pool_count].distance = dist;
            pool_count++;
        }
    }

    // 2. Sort patches by distance (ascending)
    std::sort(search_pool, search_pool + pool_count);

    // 3. Thresholding Logic (Matches IPOL getSimilarPatches)
    // We want at least 32 patches (nSimilarPatches)
    const int32_t base_similar_patches = 32;
    const int32_t safe_count = std::min(pool_count, base_similar_patches);
    
    // The threshold is max(tau_sigma_sq, distance of the 32nd patch)
    float active_threshold = tau_sigma_sq;
    if (safe_count > 0 && search_pool[safe_count - 1].distance > active_threshold) 
    {
        active_threshold = search_pool[safe_count - 1].distance;
    }

    // 4. Count how many patches fall under the active threshold
    int32_t valid_patches = 0;
    for (int32_t i = 0; i < pool_count; ++i) 
    {
        if (search_pool[i].distance <= active_threshold) 
        {
            valid_patches++;
        } 
        else 
        {
            break;
        }
    }

    // Clamp to absolute max if necessary (289 is max in window, 512 is algorithm limit)
    return std::min(valid_patches, 512);
}