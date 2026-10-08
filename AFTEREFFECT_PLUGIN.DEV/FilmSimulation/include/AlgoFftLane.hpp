#pragma once

// ---------------------------------------------------------------------------
//  AlgoFftLane.hpp  (AVX2 engine overlay)
//
//  Four complex floats per lane vector (LaneAvx2, AVX2 + FMA3, no AVX-512);
//  the real 2D plan's row/column data movement uses the register transposes
//  of fft_real2d_avx2.hpp. Same flow as the scalar overlay; see that file.
//
//  LawPairs needs pow on eight floats. It is evaluated as exp2(halfQ * log2(x))
//  with short polynomials (log2: mantissa reduced to [sqrt(.5), sqrt(2)),
//  2 atanh series in s = (m-1)/(m+1) to s^9; exp2: rounding to the nearest
//  integer and a degree-7 Taylor polynomial on [-0.5, 0.5]). Measured against
//  float64 pow for q = 1.07 .. 4.2 and f2/f50^2 = e^-8 .. e^6: max relative
//  error 1.3e-6 (at f/f50 = 19, where the transfer itself is 4e-6), max
//  absolute error 9.4e-8 -- the order of numpy's float32 x**q in film_sim.
// ---------------------------------------------------------------------------

#include "AlgoTypes.hpp"
#include "fft_real2d_avx2.hpp"

#include <immintrin.h>

using AlgoFftLane = FourierTransform::LaneAvx2;

struct AlgoFftLaneMath
{
    using V = __m256;

    static inline V AddR (const V& x, const float k) noexcept
    {
        return _mm256_add_ps(x, _mm256_set1_ps(k));
    }

    // log2(x), x > 0 normal
    static inline __m256 Log2 (const __m256 x) noexcept
    {
        const __m256i xi = _mm256_castps_si256(x);
        __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
        // mantissa in [1, 2)
        __m256 m = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF)),
                                                       _mm256_set1_epi32(0x3F800000)));
        // move to [sqrt(.5), sqrt(2))
        const __m256 big = _mm256_cmp_ps(m, _mm256_set1_ps(1.41421356237f), _CMP_GT_OQ);
        m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), big);
        e = _mm256_add_epi32(e, _mm256_and_si256(_mm256_castps_si256(big), _mm256_set1_epi32(1)));
        // ln(m) = 2 atanh(s), s = (m-1)/(m+1)
        const __m256 s  = _mm256_div_ps(_mm256_sub_ps(m, _mm256_set1_ps(1.0f)), _mm256_add_ps(m, _mm256_set1_ps(1.0f)));
        const __m256 s2 = _mm256_mul_ps(s, s);
        __m256 p = _mm256_set1_ps(1.0f / 9.0f);
        p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(1.0f / 7.0f));
        p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(1.0f / 5.0f));
        p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(1.0f / 3.0f));
        p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(1.0f));
        const __m256 lnm = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), s), p);
        return _mm256_fmadd_ps(lnm, _mm256_set1_ps(1.44269504088896341f), _mm256_cvtepi32_ps(e));
    }

    // 2^y
    static inline __m256 Exp2 (__m256 y) noexcept
    {
        y = _mm256_min_ps(_mm256_max_ps(y, _mm256_set1_ps(-126.0f)), _mm256_set1_ps(126.0f));
        const __m256 n = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m256 f = _mm256_sub_ps(y, n);                          // [-0.5, 0.5]
        // 2^f = exp(f ln2), Taylor to degree 7 in f*ln2 (|f ln2| <= 0.347)
        const __m256 z = _mm256_mul_ps(f, _mm256_set1_ps(0.69314718055994531f));
        __m256 p = _mm256_set1_ps(1.0f / 5040.0f);
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.0f / 720.0f));
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.0f / 120.0f));
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.0f / 24.0f));
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.0f / 6.0f));
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(0.5f));
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.0f));
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.0f));
        const __m256i ni = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
        return _mm256_mul_ps(p, _mm256_castsi256_ps(ni));
    }

    //: (t, t) pairs, t = 1 / (1 + (f2 / f50^2)^(q/2)); f2 == 0 -> 1.
    static inline V LawPairs (const V& f2, const float halfQ, const float invF50Sq) noexcept
    {
        const __m256 x    = _mm256_mul_ps(f2, _mm256_set1_ps(invF50Sq));
        const __m256 pos  = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OQ);
        const __m256 xs   = _mm256_blendv_ps(_mm256_set1_ps(1.0f), x, pos);        // keep log finite
        const __m256 pw   = Exp2(_mm256_mul_ps(_mm256_set1_ps(halfQ), Log2(xs)));
        const __m256 t    = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), pw));
        return _mm256_blendv_ps(_mm256_set1_ps(1.0f), t, pos);
    }
};
