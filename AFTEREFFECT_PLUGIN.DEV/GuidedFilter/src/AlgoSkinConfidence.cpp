#include <immintrin.h>
#include <cmath>
#include "AlgorithmProces.hpp"

// =============================================================================
// Skin confidence, region blend and confidence-map output (AVX2 + FMA3 path).
// Same model, constants and processing order as the Scalar path
// (Scalar/src/AlgoSkinConfidence.cpp); see the model description there and in
// AlgoControls.hpp. Unaligned loads/stores only; every SIMD loop is followed by
// a scalar loop for the remaining pixels. No AVX-512 instructions.
// =============================================================================

namespace
{
    constexpr float kInvSqrt2 = 0.70710678118654752f;
    constexpr float kInvSqrt6 = 0.40824829046386302f;

    constexpr float kCbU  = -0.668736f * kInvSqrt2;
    constexpr float kCbV  =  0.993792f * kInvSqrt6;
    constexpr float kCbC  =  26.0f;               // 128 - 102
    constexpr float kCrU  =  0.581312f * kInvSqrt2;
    constexpr float kCrV  =  1.256064f * kInvSqrt6;
    constexpr float kCrC  = -25.0f;               // 128 - 153

    constexpr float kHalfAxisCb = 25.0f;          // (127 - 77) / 2
    constexpr float kHalfAxisCr = 20.0f;          // (173 - 133) / 2
}

void math_Skin_Confidence
(
    const float* plane_Y,
    const float* plane_U,
    const float* plane_V,
    float* conf,
    const int32_t width,
    const int32_t height,
    const float tolerance,
    const float softness
) noexcept
{
    (void)plane_Y;   // the BT.601 chroma is independent of the orthonormal Y

    const int32_t totalPixels = width * height;
    const float invCb = 1.0f / (kHalfAxisCb * tolerance);
    const float invCr = 1.0f / (kHalfAxisCr * tolerance);

    const __m256 vCbU = _mm256_set1_ps(kCbU), vCbV = _mm256_set1_ps(kCbV), vCbC = _mm256_set1_ps(kCbC);
    const __m256 vCrU = _mm256_set1_ps(kCrU), vCrV = _mm256_set1_ps(kCrV), vCrC = _mm256_set1_ps(kCrC);
    const __m256 vInvCb = _mm256_set1_ps(invCb), vInvCr = _mm256_set1_ps(invCr);
    const __m256 vOne = _mm256_set1_ps(1.0f), vZero = _mm256_setzero_ps();

    int32_t i = 0;

    if (softness <= 0.0f)
    {
        for (; i + 8 <= totalPixels; i += 8)
        {
            const __m256 U = _mm256_loadu_ps(plane_U + i);
            const __m256 V = _mm256_loadu_ps(plane_V + i);
            const __m256 x = _mm256_mul_ps(_mm256_fmadd_ps(vCbV, V, _mm256_fmadd_ps(vCbU, U, vCbC)), vInvCb);
            const __m256 y = _mm256_mul_ps(_mm256_fmadd_ps(vCrV, V, _mm256_fmadd_ps(vCrU, U, vCrC)), vInvCr);
            const __m256 d2 = _mm256_fmadd_ps(x, x, _mm256_mul_ps(y, y));
            _mm256_storeu_ps(conf + i, _mm256_and_ps(_mm256_cmp_ps(d2, vOne, _CMP_LE_OQ), vOne));
        }
        for (; i < totalPixels; ++i)
        {
            const float x = (kCbC + kCbU * plane_U[i] + kCbV * plane_V[i]) * invCb;
            const float y = (kCrC + kCrU * plane_U[i] + kCrV * plane_V[i]) * invCr;
            conf[i] = (x * x + y * y <= 1.0f) ? 1.0f : 0.0f;
        }
        return;
    }

    const float d0   = 1.0f - 0.5f * softness;
    const float invS = 1.0f / softness;
    const __m256 vD0 = _mm256_set1_ps(d0), vInvS = _mm256_set1_ps(invS);
    const __m256 vTwo = _mm256_set1_ps(2.0f), vThree = _mm256_set1_ps(3.0f);

    for (; i + 8 <= totalPixels; i += 8)
    {
        const __m256 U = _mm256_loadu_ps(plane_U + i);
        const __m256 V = _mm256_loadu_ps(plane_V + i);
        const __m256 x = _mm256_mul_ps(_mm256_fmadd_ps(vCbV, V, _mm256_fmadd_ps(vCbU, U, vCbC)), vInvCb);
        const __m256 y = _mm256_mul_ps(_mm256_fmadd_ps(vCrV, V, _mm256_fmadd_ps(vCrU, U, vCrC)), vInvCr);
        const __m256 d = _mm256_sqrt_ps(_mm256_fmadd_ps(x, x, _mm256_mul_ps(y, y)));
        const __m256 t = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_sub_ps(d, vD0), vInvS), vZero), vOne);
        // c = 1 - t^2 (3 - 2t)
        const __m256 s = _mm256_mul_ps(_mm256_mul_ps(t, t), _mm256_fnmadd_ps(vTwo, t, vThree));
        _mm256_storeu_ps(conf + i, _mm256_sub_ps(vOne, s));
    }
    for (; i < totalPixels; ++i)
    {
        const float x = (kCbC + kCbU * plane_U[i] + kCbV * plane_V[i]) * invCb;
        const float y = (kCrC + kCrU * plane_U[i] + kCrV * plane_V[i]) * invCr;
        const float d = std::sqrt(x * x + y * y);
        const float t = std::min(std::max((d - d0) * invS, 0.0f), 1.0f);
        conf[i] = 1.0f - t * t * (3.0f - 2.0f * t);
    }
}

void math_Blend_Region
(
    const float* src_I,
    const float* conf,
    float* dst,
    const int32_t width,
    const int32_t height,
    const FilterMode filterMode
) noexcept
{
    const int32_t totalPixels = width * height;
    const __m256 vOne = _mm256_set1_ps(1.0f);
    int32_t i = 0;

    if (FilterMode::Skin == filterMode)
    {
        for (; i + 8 <= totalPixels; i += 8)
        {
            const __m256 c = _mm256_loadu_ps(conf + i);
            const __m256 q = _mm256_loadu_ps(dst + i);
            const __m256 I = _mm256_loadu_ps(src_I + i);
            // c * q + (1 - c) * I
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(c, q, _mm256_mul_ps(_mm256_sub_ps(vOne, c), I)));
        }
        for (; i < totalPixels; ++i)
        {
            const float c = conf[i];
            dst[i] = (c * dst[i]) + ((1.0f - c) * src_I[i]);
        }
    }
    else if (FilterMode::Background == filterMode)
    {
        for (; i + 8 <= totalPixels; i += 8)
        {
            const __m256 c = _mm256_loadu_ps(conf + i);
            const __m256 q = _mm256_loadu_ps(dst + i);
            const __m256 I = _mm256_loadu_ps(src_I + i);
            // (1 - c) * q + c * I
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_mm256_sub_ps(vOne, c), q, _mm256_mul_ps(c, I)));
        }
        for (; i < totalPixels; ++i)
        {
            const float c = conf[i];
            dst[i] = ((1.0f - c) * dst[i]) + (c * src_I[i]);
        }
    }
    // Entire_Image: dst already holds the filtered plane.
}

void math_Write_Confidence_Map
(
    const float* conf,
    const float* src_Y,
    const float* src_U,
    const float* src_V,
    float* out_Y,
    float* out_U,
    float* out_V,
    const int32_t width,
    const int32_t height,
    const FilterMode filterMode
) noexcept
{
    const int32_t totalPixels = width * height;
    const bool background = (FilterMode::Background == filterMode);
    const __m256 vOne = _mm256_set1_ps(1.0f), vZero = _mm256_setzero_ps();
    int32_t i = 0;

    for (; i + 8 <= totalPixels; i += 8)
    {
        // same weight as math_Blend_Region: the pixel is filtered when its weight is > 0
        const __m256 c = _mm256_loadu_ps(conf + i);
        const __m256 m = background ? _mm256_sub_ps(vOne, c) : c;
        const __m256 included = _mm256_cmp_ps(m, vZero, _CMP_GT_OQ);   // all-ones where m > 0
        _mm256_storeu_ps(out_Y + i, _mm256_and_ps(_mm256_loadu_ps(src_Y + i), included));
        _mm256_storeu_ps(out_U + i, _mm256_and_ps(_mm256_loadu_ps(src_U + i), included));
        _mm256_storeu_ps(out_V + i, _mm256_and_ps(_mm256_loadu_ps(src_V + i), included));
    }
    for (; i < totalPixels; ++i)
    {
        const float m = background ? (1.0f - conf[i]) : conf[i];
        const bool included = (m > 0.0f);
        out_Y[i] = included ? src_Y[i] : 0.0f;
        out_U[i] = included ? src_U[i] : 0.0f;
        out_V[i] = included ? src_V[i] : 0.0f;
    }
}