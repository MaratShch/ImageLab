#include <algorithm>
#include "AVX2_AlgoBlockMatch.hpp"

// =========================================================
// AVX2 ACCELERATED L2 DISTANCE CALCULATION
// =========================================================
float AVX2_Calculate_Patch_Distance
(
    const float* RESTRICT Y_plane,
	const float* RESTRICT U_plane,
	const float* RESTRICT V_plane,
    const int32_t ref_x,
	const int32_t ref_y,
	const int32_t tgt_x,
	const int32_t tgt_y,
    const int32_t pitch // Pitch is in elements (pixels)
) noexcept
{
    // Master accumulator for the squared differences (holds 8 floats)
    __m256 vSum = _mm256_setzero_ps();

    // Process 2 rows (8 pixels) per iteration. 
    // The loop runs exactly twice to cover the 4x4 patch.
    for (int32_t i = 0; i < 4; i += 2) 
    {
        const int32_t ref_off_0 = (ref_y + i) * pitch + ref_x;
        const int32_t ref_off_1 = (ref_y + i + 1) * pitch + ref_x;
        const int32_t tgt_off_0 = (tgt_y + i) * pitch + tgt_x;
        const int32_t tgt_off_1 = (tgt_y + i + 1) * pitch + tgt_x;

        // --- Y CHANNEL ---
        __m128 y_r0 = _mm_loadu_ps(Y_plane + ref_off_0);
        __m128 y_r1 = _mm_loadu_ps(Y_plane + ref_off_1);
        __m256 y_ref = _mm256_insertf128_ps(_mm256_castps128_ps256(y_r0), y_r1, 1);

        __m128 y_t0 = _mm_loadu_ps(Y_plane + tgt_off_0);
        __m128 y_t1 = _mm_loadu_ps(Y_plane + tgt_off_1);
        __m256 y_tgt = _mm256_insertf128_ps(_mm256_castps128_ps256(y_t0), y_t1, 1);

        __m256 dy = _mm256_sub_ps(y_ref, y_tgt);
        vSum = _mm256_fmadd_ps(dy, dy, vSum); 

        // --- U CHANNEL ---
        __m128 u_r0 = _mm_loadu_ps(U_plane + ref_off_0);
        __m128 u_r1 = _mm_loadu_ps(U_plane + ref_off_1);
        __m256 u_ref = _mm256_insertf128_ps(_mm256_castps128_ps256(u_r0), u_r1, 1);

        __m128 u_t0 = _mm_loadu_ps(U_plane + tgt_off_0);
        __m128 u_t1 = _mm_loadu_ps(U_plane + tgt_off_1);
        __m256 u_tgt = _mm256_insertf128_ps(_mm256_castps128_ps256(u_t0), u_t1, 1);

        __m256 du = _mm256_sub_ps(u_ref, u_tgt);
        vSum = _mm256_fmadd_ps(du, du, vSum);

        // --- V CHANNEL ---
        __m128 v_r0 = _mm_loadu_ps(V_plane + ref_off_0);
        __m128 v_r1 = _mm_loadu_ps(V_plane + ref_off_1);
        __m256 v_ref = _mm256_insertf128_ps(_mm256_castps128_ps256(v_r0), v_r1, 1);

        __m128 v_t0 = _mm_loadu_ps(V_plane + tgt_off_0);
        __m128 v_t1 = _mm_loadu_ps(V_plane + tgt_off_1);
        __m256 v_tgt = _mm256_insertf128_ps(_mm256_castps128_ps256(v_t0), v_t1, 1);

        __m256 dv = _mm256_sub_ps(v_ref, v_tgt);
        vSum = _mm256_fmadd_ps(dv, dv, vSum);
    }

    // Fast horizontal sum of the 8 floats
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(vSum), _mm256_extractf128_ps(vSum, 1));
    __m128 sum64  = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    __m128 sum32  = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(1, 1, 1, 1)));

    return _mm_cvtss_f32(sum32);
}

// =========================================================
// AVX2 ACCELERATED EXTRACTION LOOP (FIXED SORT STABILITY)
// =========================================================
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
)
{
    // 1. LOCK THE REFERENCE PATCH TO INDEX 0
    // This absolutely guarantees the 'Paste Trick' in the aggregation phase 
    // never punches holes in the wrong coordinates due to unstable sorting.
    search_pool[0].x = ref_x;
    search_pool[0].y = ref_y;
    search_pool[0].distance = 0.0f;
    int32_t pool_count = 1; 
    
    constexpr int32_t radius = 8;
    constexpr int32_t patch_size = 4;
    
    const int32_t min_x = std::max(0, ref_x - radius);
    const int32_t max_x = std::min(width - patch_size, ref_x + radius);
    const int32_t min_y = std::max(0, ref_y - radius);
    const int32_t max_y = std::min(height - patch_size, ref_y + radius);

    for (int32_t tgt_y = min_y; tgt_y <= max_y; ++tgt_y) 
    {
        for (int32_t tgt_x = min_x; tgt_x <= max_x; ++tgt_x) 
        {
            // Skip the reference patch, we already locked it at index 0
            if (tgt_x == ref_x && tgt_y == ref_y) continue;

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

    // 2. SORT EVERYTHING EXCEPT INDEX 0
    std::sort(search_pool + 1, search_pool + pool_count);

    const int32_t base_similar_patches = 32;
    const int32_t safe_count = std::min(pool_count, base_similar_patches);
    
    float active_threshold = tau_sigma_sq;
    // Check safe_count > 1 because index 0 is always 0.0f
    if (safe_count > 1 && search_pool[safe_count - 1].distance > active_threshold) 
    {
        active_threshold = search_pool[safe_count - 1].distance;
    }

    int32_t valid_patches = 1; // Reference patch is always valid
    for (int32_t i = 1; i < pool_count; ++i) 
    {
        if (search_pool[i].distance <= active_threshold) valid_patches++;
        else break;
    }

    return std::min(valid_patches, 512);
}

