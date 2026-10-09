#include <immintrin.h>
#include "AlgorithmProces.hpp"

// =============================================================================
// AVX2 + FMA3 implementation (same prototypes as the Scalar path).
// Target: AVX2/FMA3 CPUs (e.g. i7-7700K, Kaby Lake). No AVX-512 instructions are used.
// =============================================================================

void math_Square_Elements
(
    const float* src, 
    float* dst, 
    const int32_t width, 
    const int32_t height
) noexcept
{
    const int32_t totalPixels = width * height;
    int32_t i = 0;

    for (; i + 8 <= totalPixels; i += 8)
    {
        const __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, v));
    }

    for (; i < totalPixels; ++i)
    {
        const float val = src[i];
        dst[i] = val * val;
    }
}

void math_Compute_Coefficients_AB
(
    const float* mean_I, 
    const float* mean_II, 
    float* coef_a, 
    float* coef_b, 
    const int32_t width, 
    const int32_t height, 
    const float epsilon
) noexcept
{
    const int32_t totalPixels = width * height;

    // 'epsilon' is already expressed in signal units (variance of the 0..255 scale).
    // The caller performs the one and only scaling epsilon * 255^2 (defect D1).
    // var / a / b are evaluated in double, like the Scalar path (defect D3).
    const double  eps_d = static_cast<double>(epsilon);
    const __m256d vEps  = _mm256_set1_pd(eps_d);
    const __m256d vZero = _mm256_setzero_pd();

    int32_t i = 0;
    for (; i + 8 <= totalPixels; i += 8)
    {
        const __m256 vI  = _mm256_loadu_ps(mean_I  + i);
        const __m256 vII = _mm256_loadu_ps(mean_II + i);

        __m256 vA[2], vB[2];
        for (int h = 0; h < 2; ++h)
        {
            const __m256d m_I  = _mm256_cvtps_pd(0 == h ? _mm256_castps256_ps128(vI)  : _mm256_extractf128_ps(vI, 1));
            const __m256d m_II = _mm256_cvtps_pd(0 == h ? _mm256_castps256_ps128(vII) : _mm256_extractf128_ps(vII, 1));

            // max(var, 0): second operand is returned for NaN, matching std::max(0.0, var)
            // var = m_II - m_I * m_I  (fnmadd: -(x*y) + z, single rounding)
            const __m256d var = _mm256_max_pd(_mm256_fnmadd_pd(m_I, m_I, m_II), vZero);
            const __m256d a   = _mm256_div_pd(var, _mm256_add_pd(var, vEps));
            // b = m_I - a * m_I
            const __m256d b   = _mm256_fnmadd_pd(a, m_I, m_I);

            vA[h] = _mm256_castps128_ps256(_mm256_cvtpd_ps(a));
            vB[h] = _mm256_castps128_ps256(_mm256_cvtpd_ps(b));
        }

        _mm256_storeu_ps(coef_a + i, _mm256_insertf128_ps(vA[0], _mm256_castps256_ps128(vA[1]), 1));
        _mm256_storeu_ps(coef_b + i, _mm256_insertf128_ps(vB[0], _mm256_castps256_ps128(vB[1]), 1));
    }

    for (; i < totalPixels; ++i)    // scalar tail: identical to the Scalar path
    {
        const double m_I  = static_cast<double>(mean_I[i]);
        const double m_II = static_cast<double>(mean_II[i]);
        const double variance = std::max(0.0, m_II - (m_I * m_I));
        const double a = variance / (variance + eps_d);
        const double b = m_I - (a * m_I);
        coef_a[i] = static_cast<float>(a);
        coef_b[i] = static_cast<float>(b);
    }
}