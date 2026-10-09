#include <immintrin.h>
#include "AlgorithmProces.hpp"
#include "AlgoControls.hpp"

// =============================================================================
// AVX2 + FMA3 implementation (same prototype as the Scalar path).
// Target: AVX2/FMA3 CPUs (e.g. i7-7700K, Kaby Lake). No AVX-512 instructions are used.
// FMA rounds once instead of twice, so results may differ from the Scalar path
// in the last float bit; the scalar tail loops keep the Scalar arithmetic.
// =============================================================================

void math_Apply_Filter_And_Mask
(
    const float* src_I, 
    const float* mean_a, 
    const float* mean_b, 
    const float* in_Mask, 
    float* dst, 
    const int32_t width, 
    const int32_t height, 
    const FilterMode filterMode
) noexcept
{
    const int32_t totalPixels = width * height;
    const __m256 vOne = _mm256_set1_ps(1.0f);
    int32_t i = 0;

    // Route 1: Denoise Entire Image (Mask is bypassed)
    if (filterMode == FilterMode::Entire_Image)
    {
        for (; i + 8 <= totalPixels; i += 8)
        {
            const __m256 q = _mm256_fmadd_ps(_mm256_loadu_ps(mean_a + i), _mm256_loadu_ps(src_I + i), _mm256_loadu_ps(mean_b + i));
            _mm256_storeu_ps(dst + i, q);
        }
        for (; i < totalPixels; ++i)
        {
            dst[i] = (mean_a[i] * src_I[i]) + mean_b[i];
        }
    }
    // Route 2: Denoise Skin Only (Filter where Mask == 1.0)
    else if (filterMode == FilterMode::Skin)
    {
        for (; i + 8 <= totalPixels; i += 8)
        {
            const __m256 I = _mm256_loadu_ps(src_I + i);
            const __m256 q = _mm256_fmadd_ps(_mm256_loadu_ps(mean_a + i), I, _mm256_loadu_ps(mean_b + i));
            const __m256 M = _mm256_loadu_ps(in_Mask + i);

            // output = (Mask * filter) + ((1 - Mask) * original)
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(M, q, _mm256_mul_ps(_mm256_sub_ps(vOne, M), I)));
        }
        for (; i < totalPixels; ++i)
        {
            const float I = src_I[i];
            const float q = (mean_a[i] * I) + mean_b[i];
            const float M = in_Mask[i];
            dst[i] = (M * q) + ((1.0f - M) * I);
        }
    }
    // Route 3: Denoise Background Only (Filter where Mask == 0.0)
    else 
    {
        for (; i + 8 <= totalPixels; i += 8)
        {
            const __m256 I = _mm256_loadu_ps(src_I + i);
            const __m256 q = _mm256_fmadd_ps(_mm256_loadu_ps(mean_a + i), I, _mm256_loadu_ps(mean_b + i));
            const __m256 M = _mm256_loadu_ps(in_Mask + i);

            // output = ((1 - Mask) * filter) + (Mask * original)
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_mm256_sub_ps(vOne, M), q, _mm256_mul_ps(M, I)));
        }
        for (; i < totalPixels; ++i)
        {
            const float I = src_I[i];
            const float q = (mean_a[i] * I) + mean_b[i];
            const float M = in_Mask[i];
            dst[i] = ((1.0f - M) * q) + (M * I);
        }
    }
    
    return;
}