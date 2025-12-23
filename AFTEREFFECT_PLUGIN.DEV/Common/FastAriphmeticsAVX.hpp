#pragma once
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <immintrin.h>

namespace FastCompute
{
    namespace AVX2
    {
        inline __m256 mm256_fmaf(__m256 a, __m256 b, __m256 c) noexcept
        {
            return _mm256_add_ps(_mm256_mul_ps(a, b), c);
        }

        /* https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c  */
        inline __m256 Log(__m256 a) noexcept
        {
            __m256i aInt = *(__m256i*)(&a);
            __m256i e = _mm256_sub_epi32(aInt, _mm256_set1_epi32(0x3f2aaaab));
            e = _mm256_and_si256(e, _mm256_set1_epi32(0xff800000));

            __m256i subtr = _mm256_sub_epi32(aInt, e);
            __m256 m = *(__m256*)&subtr;

            __m256 i = _mm256_mul_ps(_mm256_cvtepi32_ps(e), _mm256_set1_ps(1.19209290e-7f));// 0x1.0p-23
                                                                                            /* m in [2/3, 4/3] */
            __m256 f = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
            __m256 s = _mm256_mul_ps(f, f);
            /* Compute log1p(f) for f in [-1/3, 1/3] */
            __m256 r = mm256_fmaf(_mm256_set1_ps(0.230836749f), f, _mm256_set1_ps(-0.279208571f));// 0x1.d8c0f0p-3, -0x1.1de8dap-2
            __m256 t = mm256_fmaf(_mm256_set1_ps(0.331826031f), f, _mm256_set1_ps(-0.498910338f));// 0x1.53ca34p-2, -0x1.fee25ap-2

            r = mm256_fmaf(r, s, t);
            r = mm256_fmaf(r, s, f);
            r = mm256_fmaf(i, _mm256_set1_ps(0.693147182f), r);  // 0x1.62e430p-1 // log(2)
            return r;
        }

        inline __m256 Cbrt(__m256 x) noexcept
        {
            __m256i i = _mm256_castps_si256(x);
            // Bit Hack: i/3 approx
            __m256i t = _mm256_add_epi32(_mm256_srli_epi32(i, 2), _mm256_srli_epi32(i, 4));
            t = _mm256_add_epi32(t, _mm256_srli_epi32(t, 4));
            t = _mm256_add_epi32(t, _mm256_srli_epi32(t, 8));
            t = _mm256_add_epi32(t, _mm256_srli_epi32(t, 16));
            t = _mm256_add_epi32(t, _mm256_set1_epi32(0x2a5137a0));

            __m256 y = _mm256_castsi256_ps(t);
            __m256 two = _mm256_set1_ps(2.0f);
            __m256 third = _mm256_set1_ps(0.33333333f);

            // Newton 1 with Reciprocal
            __m256 ryy = _mm256_rcp_ps(_mm256_mul_ps(y, y));
            y = _mm256_mul_ps(third, _mm256_fmadd_ps(two, y, _mm256_mul_ps(x, ryy)));
            // Newton 2
            ryy = _mm256_rcp_ps(_mm256_mul_ps(y, y));
            y = _mm256_mul_ps(third, _mm256_fmadd_ps(two, y, _mm256_mul_ps(x, ryy)));
            return y;
        }

        inline __m256 Exp(__m256 x) noexcept
        {
            // Clamp to avoid overflow
            x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
            x = _mm256_min_ps(x, _mm256_set1_ps(87.0f));

            // Schraudolph approximation: exp(x) ~= (12102203 * x) + 1064986823 (integer reinterpretation)
            // 12102203 ~ 2^23 / ln(2)
            __m256 val = mm256_fmaf(x, _mm256_set1_ps(12102203.0f), _mm256_set1_ps(1064986823.0f));
            return _mm256_castsi256_ps(_mm256_cvtps_epi32(val));
        }

        // Pow(x, y) = Exp(y * Log(x))
        inline __m256 Pow(__m256 x, __m256 y) noexcept
        {
            x = _mm256_max_ps(x, _mm256_set1_ps(1.17549435e-38f));
            return Exp(_mm256_mul_ps(y, Log(x)));
        }


    } /* namespace AVX2 */
}
