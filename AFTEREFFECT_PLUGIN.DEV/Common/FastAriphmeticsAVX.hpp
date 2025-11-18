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

    } /* namespace AVX2 */
}
