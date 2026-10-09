#include <immintrin.h>
#include <cstring>
#include <algorithm> // For std::min and std::max
#include "AlgorithmProces.hpp"

// =============================================================================
// AVX2 implementation of math_BoxBlur_O1 (same prototype as the Scalar path).
//
// Semantics are identical to the Scalar implementation:
//   - window (2r+1) x (2r+1), clamp-to-edge boundary extension,
//   - constant normalisation 1 / (2r+1) per 1D pass,
//   - sliding sums accumulated in double, results stored as float,
//   - horizontal pass first (src -> temp), vertical pass second (temp -> dst),
//   - src == dst (in-place) is allowed.
//
// The vertical pass is naturally SIMD (8/16 adjacent columns per sweep).
// The horizontal pass is executed as a vertical pass on the transposed plane:
//   src --T--> temp --V--> dst --T--> temp --V--> dst
// Only the three buffers of the prototype are used; no extra allocation.
// Target: AVX2 (i7-7700K class); no AVX-512. The sliding update is add/sub only,
// so there is no multiply-add pair to fuse with FMA here.
// =============================================================================

namespace
{
    // ---------------------------------------------------------------------
    // 8x8 float transpose: d[c * dp + r] = s[r * sp + c], r,c in [0..7]
    // ---------------------------------------------------------------------
    inline void Transpose_8x8
    (
        const float* RESTRICT s,
        const int32_t sp,
        float* RESTRICT d,
        const int32_t dp
    ) noexcept
    {
        const __m256 r0 = _mm256_loadu_ps(s + 0 * sp);
        const __m256 r1 = _mm256_loadu_ps(s + 1 * sp);
        const __m256 r2 = _mm256_loadu_ps(s + 2 * sp);
        const __m256 r3 = _mm256_loadu_ps(s + 3 * sp);
        const __m256 r4 = _mm256_loadu_ps(s + 4 * sp);
        const __m256 r5 = _mm256_loadu_ps(s + 5 * sp);
        const __m256 r6 = _mm256_loadu_ps(s + 6 * sp);
        const __m256 r7 = _mm256_loadu_ps(s + 7 * sp);

        const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
        const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
        const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
        const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
        const __m256 t4 = _mm256_unpacklo_ps(r4, r5);
        const __m256 t5 = _mm256_unpackhi_ps(r4, r5);
        const __m256 t6 = _mm256_unpacklo_ps(r6, r7);
        const __m256 t7 = _mm256_unpackhi_ps(r6, r7);

        const __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
        const __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
        const __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
        const __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

        _mm256_storeu_ps(d + 0 * dp, _mm256_permute2f128_ps(u0, u4, 0x20));
        _mm256_storeu_ps(d + 1 * dp, _mm256_permute2f128_ps(u1, u5, 0x20));
        _mm256_storeu_ps(d + 2 * dp, _mm256_permute2f128_ps(u2, u6, 0x20));
        _mm256_storeu_ps(d + 3 * dp, _mm256_permute2f128_ps(u3, u7, 0x20));
        _mm256_storeu_ps(d + 4 * dp, _mm256_permute2f128_ps(u0, u4, 0x31));
        _mm256_storeu_ps(d + 5 * dp, _mm256_permute2f128_ps(u1, u5, 0x31));
        _mm256_storeu_ps(d + 6 * dp, _mm256_permute2f128_ps(u2, u6, 0x31));
        _mm256_storeu_ps(d + 7 * dp, _mm256_permute2f128_ps(u3, u7, 0x31));
    }

    // ---------------------------------------------------------------------
    // Plane transpose: src is (height x width), dst becomes (width x height)
    // dst[x * height + y] = src[y * width + x]. src and dst must not overlap.
    // ---------------------------------------------------------------------
    void Transpose_Plane
    (
        const float* RESTRICT src,
        float* RESTRICT dst,
        const int32_t width,
        const int32_t height
    ) noexcept
    {
        const int32_t yBlk = height & ~7;
        const int32_t xBlk = width & ~7;

        for (int32_t y = 0; y < yBlk; y += 8)
        {
            for (int32_t x = 0; x < xBlk; x += 8)
                Transpose_8x8(src + y * width + x, width, dst + x * height + y, height);

            for (int32_t x = xBlk; x < width; ++x)      // right tail columns
                for (int32_t k = 0; k < 8; ++k)
                    dst[x * height + y + k] = src[(y + k) * width + x];
        }

        for (int32_t y = yBlk; y < height; ++y)          // bottom tail rows
            for (int32_t x = 0; x < width; ++x)
                dst[x * height + y] = src[y * width + x];
    }

    inline __m256d Lo_PD (const __m256 v) noexcept { return _mm256_cvtps_pd(_mm256_castps256_ps128(v)); }
    inline __m256d Hi_PD (const __m256 v) noexcept { return _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)); }

    inline __m256 To_PS (const __m256d lo, const __m256d hi) noexcept
    {
        return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(lo)), _mm256_cvtpd_ps(hi), 1);
    }

    // ---------------------------------------------------------------------
    // Vertical sliding box over G groups of 8 adjacent columns starting at x.
    // Bit-for-bit the same arithmetic sequence as the Scalar vertical pass:
    //   sum  = (r+1) * s[0]; sum += s[clamp(i)], i = 1..r
    //   d[y] = float(sum * inv); sum += (double)s[in] - (double)s[out]
    // ---------------------------------------------------------------------
    template <int G>
    inline void Vertical_Strip
    (
        const float* RESTRICT src,
        float* RESTRICT dst,
        const int32_t width,
        const int32_t height,
        const int32_t radius,
        const int32_t x,
        const __m256d vInv,
        const __m256d vPrime
    ) noexcept
    {
        __m256d acc[2 * G];

        for (int g = 0; g < G; ++g)
        {
            const __m256 v0 = _mm256_loadu_ps(src + x + 8 * g);
            acc[2 * g + 0] = _mm256_mul_pd(vPrime, Lo_PD(v0));
            acc[2 * g + 1] = _mm256_mul_pd(vPrime, Hi_PD(v0));
        }

        for (int32_t i = 1; i <= radius; ++i)
        {
            const int32_t cy = std::min(i, height - 1);
            for (int g = 0; g < G; ++g)
            {
                const __m256 v = _mm256_loadu_ps(src + cy * width + x + 8 * g);
                acc[2 * g + 0] = _mm256_add_pd(acc[2 * g + 0], Lo_PD(v));
                acc[2 * g + 1] = _mm256_add_pd(acc[2 * g + 1], Hi_PD(v));
            }
        }

        for (int32_t y = 0; y < height; ++y)
        {
            const int32_t bottom_idx = std::min(y + radius + 1, height - 1);
            const int32_t top_idx    = std::max(y - radius, 0);

            for (int g = 0; g < G; ++g)
            {
                _mm256_storeu_ps(dst + y * width + x + 8 * g,
                                 To_PS(_mm256_mul_pd(acc[2 * g + 0], vInv), _mm256_mul_pd(acc[2 * g + 1], vInv)));

                const __m256 vIn  = _mm256_loadu_ps(src + bottom_idx * width + x + 8 * g);
                const __m256 vOut = _mm256_loadu_ps(src + top_idx    * width + x + 8 * g);

                acc[2 * g + 0] = _mm256_add_pd(acc[2 * g + 0], _mm256_sub_pd(Lo_PD(vIn), Lo_PD(vOut)));
                acc[2 * g + 1] = _mm256_add_pd(acc[2 * g + 1], _mm256_sub_pd(Hi_PD(vIn), Hi_PD(vOut)));
            }
        }
    }

    // ---------------------------------------------------------------------
    // Vertical clamp-to-edge box blur of a (height x width) plane. src != dst.
    // ---------------------------------------------------------------------
    void Box_Vertical
    (
        const float* RESTRICT src,
        float* RESTRICT dst,
        const int32_t width,
        const int32_t height,
        const int32_t radius
    ) noexcept
    {
        const double inv_window = 1.0 / static_cast<double>(2 * radius + 1);
        const __m256d vInv   = _mm256_set1_pd(inv_window);
        const __m256d vPrime = _mm256_set1_pd(static_cast<double>(radius + 1));

        int32_t x = 0;
        for (; x + 16 <= width; x += 16)    // 16 columns = one 64-byte cache line per row
            Vertical_Strip<2>(src, dst, width, height, radius, x, vInv, vPrime);
        for (; x + 8 <= width; x += 8)
            Vertical_Strip<1>(src, dst, width, height, radius, x, vInv, vPrime);

        for (; x < width; ++x)              // scalar tail: identical to the Scalar path
        {
            double sum = static_cast<double>(radius + 1) * static_cast<double>(src[x]);
            for (int32_t i = 1; i <= radius; ++i)
                sum += static_cast<double>(src[std::min(i, height - 1) * width + x]);

            for (int32_t y = 0; y < height; ++y)
            {
                dst[y * width + x] = static_cast<float>(sum * inv_window);
                const int32_t bottom_idx = std::min(y + radius + 1, height - 1);
                const int32_t top_idx    = std::max(y - radius, 0);
                sum += static_cast<double>(src[bottom_idx * width + x]) - static_cast<double>(src[top_idx * width + x]);
            }
        }
    }
} // anonymous namespace


void math_BoxBlur_O1
(
    const float* src, 
    float* dst, 
    float* temp, 
    const int32_t width, 
    const int32_t height, 
    const int32_t radius
) noexcept
{
    // Safety fallback: if radius is 0 or negative, just copy src to dst
    if (radius <= 0)
    {
        if (src != dst)
            std::memcpy(dst, src, static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float));
        return;
    }

    // PASS 1: HORIZONTAL BLUR, executed as a vertical blur of the transposed plane.
    Transpose_Plane(src, temp, width, height);          // temp : width x height (transposed)
    Box_Vertical(temp, dst, height, width, radius);     // dst  : horizontal result, transposed
    Transpose_Plane(dst, temp, height, width);          // temp : horizontal result, height x width

    // PASS 2: VERTICAL BLUR (temp -> dst)
    Box_Vertical(temp, dst, width, height, radius);
}