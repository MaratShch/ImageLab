#include "AlgoLumaManipulation.hpp"

void MixAndNormalizeDensity
(
    const float* RESTRICT luma_src,
    const float* RESTRICT edge_src,
    float* RESTRICT target_dest,
    int pixel_count,
    float sensitivity
) noexcept
{
    // 1. Setup Weights
    if (sensitivity < 0.0f) sensitivity = 0.0f;
    if (sensitivity > 100.0f) sensitivity = 100.0f;

    const float w_edge_scalar = sensitivity / 100.0f;
    const float w_luma_scalar = 1.0f - w_edge_scalar;

    const __m256 v_w_edge = _mm256_set1_ps(w_edge_scalar);
    const __m256 v_w_luma = _mm256_set1_ps(w_luma_scalar);

    // Initialize max vector to zero
    __m256 v_global_max = _mm256_setzero_ps();

    int i = 0;

    // ---------------------------------------------------------
    // 2. MIX PASS (AVX2)
    // Formula: dest = (luma * w_luma) + (edge * w_edge)
    // ---------------------------------------------------------

    // Unrolled Loop (32 pixels per iteration)
    for (; i <= pixel_count - 32; i += 32) {
        // Load Luma
        __m256 l0 = _mm256_loadu_ps(luma_src + i);
        __m256 l1 = _mm256_loadu_ps(luma_src + i + 8);
        __m256 l2 = _mm256_loadu_ps(luma_src + i + 16);
        __m256 l3 = _mm256_loadu_ps(luma_src + i + 24);

        // Load Edge
        __m256 e0 = _mm256_loadu_ps(edge_src + i);
        __m256 e1 = _mm256_loadu_ps(edge_src + i + 8);
        __m256 e2 = _mm256_loadu_ps(edge_src + i + 16);
        __m256 e3 = _mm256_loadu_ps(edge_src + i + 24);

        // Mix: (Luma * w_luma) + (Edge * w_edge)
        // Using FMA: result = (edge * w_edge) + (luma * w_luma)
        // (Note: luma*w_luma computed as intermediate)

        __m256 m0 = _mm256_fmadd_ps(e0, v_w_edge, _mm256_mul_ps(l0, v_w_luma));
        __m256 m1 = _mm256_fmadd_ps(e1, v_w_edge, _mm256_mul_ps(l1, v_w_luma));
        __m256 m2 = _mm256_fmadd_ps(e2, v_w_edge, _mm256_mul_ps(l2, v_w_luma));
        __m256 m3 = _mm256_fmadd_ps(e3, v_w_edge, _mm256_mul_ps(l3, v_w_luma));

        // Store intermediate result
        _mm256_storeu_ps(target_dest + i, m0);
        _mm256_storeu_ps(target_dest + i + 8, m1);
        _mm256_storeu_ps(target_dest + i + 16, m2);
        _mm256_storeu_ps(target_dest + i + 24, m3);

        // Update Max (Reduction step 1)
        v_global_max = _mm256_max_ps(v_global_max, m0);
        v_global_max = _mm256_max_ps(v_global_max, m1);
        v_global_max = _mm256_max_ps(v_global_max, m2);
        v_global_max = _mm256_max_ps(v_global_max, m3);
    }

    // Scalar Fallback for Mix Pass
    float scalar_max = 0.0f;
    for (; i < pixel_count; ++i) {
        float val = (luma_src[i] * w_luma_scalar) + (edge_src[i] * w_edge_scalar);
        target_dest[i] = val;
        if (val > scalar_max) scalar_max = val;
    }

    // Finalize Max Reduction
    float vec_max_scalar = hmax_avx2(v_global_max);
    float max_density = std::max(scalar_max, vec_max_scalar);

    // ---------------------------------------------------------
    // 3. NORMALIZE & COMPRESS PASS (AVX2)
    // ---------------------------------------------------------
    if (max_density > 0.000001f) {
        float scale_factor = 1.0f / max_density;
        float min_floor = 0.20f;
        float range = 1.0f - min_floor;

        // Algebraic Simplification for FMA:
        // Result = min_floor + (Val * scale_factor * range)
        // Result = Val * (scale_factor * range) + min_floor
        // Let S = scale_factor * range
        // Let O = min_floor

        float combined_scale = scale_factor * range;

        const __m256 v_scale = _mm256_set1_ps(combined_scale);
        const __m256 v_offset = _mm256_set1_ps(min_floor);

        i = 0;
        // Unrolled Loop (32 pixels)
        for (; i <= pixel_count - 32; i += 32) {
            __m256 v0 = _mm256_loadu_ps(target_dest + i);
            __m256 v1 = _mm256_loadu_ps(target_dest + i + 8);
            __m256 v2 = _mm256_loadu_ps(target_dest + i + 16);
            __m256 v3 = _mm256_loadu_ps(target_dest + i + 24);

            // Fused Multiply-Add: (Val * S) + O
            v0 = _mm256_fmadd_ps(v0, v_scale, v_offset);
            v1 = _mm256_fmadd_ps(v1, v_scale, v_offset);
            v2 = _mm256_fmadd_ps(v2, v_scale, v_offset);
            v3 = _mm256_fmadd_ps(v3, v_scale, v_offset);

            _mm256_storeu_ps(target_dest + i, v0);
            _mm256_storeu_ps(target_dest + i + 8, v1);
            _mm256_storeu_ps(target_dest + i + 16, v2);
            _mm256_storeu_ps(target_dest + i + 24, v3);
        }

        // Scalar Fallback
        for (; i < pixel_count; ++i) {
            // Recalculating constants here for scalar safety
            float val = target_dest[i];
            float norm = val * scale_factor;
            target_dest[i] = min_floor + (norm * range);
        }
    }
    else {
        // Fallback for empty/black image
        const __m256 v_fallback = _mm256_set1_ps(0.1f);
        i = 0;
        for (; i <= pixel_count - 32; i += 32) {
            _mm256_storeu_ps(target_dest + i, v_fallback);
            _mm256_storeu_ps(target_dest + i + 8, v_fallback);
            _mm256_storeu_ps(target_dest + i + 16, v_fallback);
            _mm256_storeu_ps(target_dest + i + 24, v_fallback);
        }
        for (; i < pixel_count; ++i) target_dest[i] = 0.1f;
    }
}