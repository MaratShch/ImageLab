#include <immintrin.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include "Common.hpp"
#include "ArtPointillismEnums.hpp"
#include "AlgoArtisticsRendering.hpp"

void Integrate_Colors
(
    const JFAPixel* RESTRICT jfa_map,
    const float* RESTRICT src_L,
    const float* RESTRICT src_ab,
    int32_t width,
	int32_t height,
    int32_t num_dots,
    float* RESTRICT acc_L,
    float* RESTRICT acc_a,
    float* RESTRICT acc_b,
    int32_t* RESTRICT acc_count,
    fCIELabPix* RESTRICT out_dot_colors
)
{
    // 1. Zero Accumulators (Use memset for speed)
    // float 0.0f is bitwise 0x00000000
    std::memset(acc_L, 0, num_dots * sizeof(float));
    std::memset(acc_a, 0, num_dots * sizeof(float));
    std::memset(acc_b, 0, num_dots * sizeof(float));
    std::memset(acc_count, 0, num_dots * sizeof(int32_t));

    // 2. Accumulate (Scalar - Memory Bound)
    // Unrolling 4x manually helps modern CPUs pipeline the loads
    const int32_t num_pixels = width * height;
    for (int32_t i = 0; i < num_pixels; ++i)
	{
        int32_t dot_id = jfa_map[i].seed_index;
        if (dot_id >= 0 && dot_id < num_dots)
		{
            acc_L[dot_id] += src_L[i];
            acc_a[dot_id] += src_ab[i*2+0];
            acc_b[dot_id] += src_ab[i*2+1];
            acc_count[dot_id]++;
        }
    }

    // 3. Average (AVX2 Vectorized)
    int32_t i = 0;
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_def_L = _mm256_set1_ps(50.0f);

    for (; i <= num_dots - 8; i += 8)
	{
        // Load counts (int) -> Convert to Float
        __m256i counts_i = _mm256_loadu_si256((const __m256i*)(acc_count + i));
        __m256  counts_f = _mm256_cvtepi32_ps(counts_i);

        // Load Sums
        __m256 sum_L = _mm256_loadu_ps(acc_L + i);
        __m256 sum_a = _mm256_loadu_ps(acc_a + i);
        __m256 sum_b = _mm256_loadu_ps(acc_b + i);

        // Compute Inverse: 1.0 / Count
        // Use approx rcp for speed, or div for precision. Div is safer here.
        // Mask out division by zero
        __m256 mask_valid = _mm256_cmp_ps(counts_f, v_zero, _CMP_GT_OQ);
        
        // Div: Safe way (add epsilon or blend)
        // Let's calculate div, then blend results.
        // Avoid NaN by ensuring divisor is not 0 for calculation
        __m256 safe_counts = _mm256_blendv_ps(v_one, counts_f, mask_valid); 
        __m256 inv_count = _mm256_div_ps(v_one, safe_counts);

        __m256 avg_L = _mm256_mul_ps(sum_L, inv_count);
        __m256 avg_a = _mm256_mul_ps(sum_a, inv_count);
        __m256 avg_b = _mm256_mul_ps(sum_b, inv_count);

        // Apply Dead Dot Fallback (Gray)
        avg_L = _mm256_blendv_ps(v_def_L, avg_L, mask_valid);
        avg_a = _mm256_blendv_ps(v_zero, avg_a, mask_valid);
        avg_b = _mm256_blendv_ps(v_zero, avg_b, mask_valid);

        // Store to AoS (Struct of Arrays -> Array of Structs)
        // We have Planar avg_L, avg_a, avg_b. Need to write [L a b L a b ...]
        // Same Stack Spill trick
        CACHE_ALIGN float tmp[24];
        _mm256_storeu_ps(tmp, avg_L);
        _mm256_storeu_ps(tmp+8, avg_a);
        _mm256_storeu_ps(tmp+16, avg_b);

        fCIELabPix* dst = out_dot_colors + i;
        for (int32_t k=0; k<8; ++k)
		{
            dst[k].L = tmp[k];
            dst[k].a = tmp[k+8];
            dst[k].b = tmp[k+16];
        }
    }

    // Scalar Cleanup
    for (; i < num_dots; ++i)
	{
        if (acc_count[i] > 0)
		{
            float inv = 1.0f / acc_count[i];
            out_dot_colors[i] = { acc_L[i]*inv, acc_a[i]*inv, acc_b[i]*inv };
        } 
		else
		{
            out_dot_colors[i] = { 50.0f, 0.0f, 0.0f };
        }
    }
	
	return;
}