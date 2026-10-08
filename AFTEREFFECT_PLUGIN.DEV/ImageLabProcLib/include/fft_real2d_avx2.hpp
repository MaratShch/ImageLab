#pragma once

// ============================================================================
// fft_real2d_avx2.hpp -- register transposes for the real 2D plan, LaneAvx2.
//
// The generic lane IO in fft_real2d.hpp moves data between image rows and the
// packed lanes one scalar at a time. For four lanes that is the dominant cost
// of the row passes, so here the same moves are done as 8x8 float transposes
// (packing 8 image rows into 4 complex lanes, and back) and 4x4 complex
// transposes (separated spectra to and from 8 spectrum rows). The arithmetic
// and its order are unchanged; only the data movement differs.
//
// Include ONLY from translation units compiled with -mavx2 -mfma.
// ============================================================================

#include "fft_real2d.hpp"
#include "fft_lane_avx2.hpp"

namespace FourierTransform
{

namespace avx2_detail
{

inline void Transpose8x8 (__m256* r) noexcept
{
    const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]);
    const __m256 t1 = _mm256_unpackhi_ps(r[0], r[1]);
    const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]);
    const __m256 t3 = _mm256_unpackhi_ps(r[2], r[3]);
    const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]);
    const __m256 t5 = _mm256_unpackhi_ps(r[4], r[5]);
    const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]);
    const __m256 t7 = _mm256_unpackhi_ps(r[6], r[7]);
    const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    r[0] = _mm256_permute2f128_ps(u0, u4, 0x20);
    r[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
    r[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
    r[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
    r[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
    r[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
    r[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
    r[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}

// 4x4 transpose of complex floats viewed as doubles: out[l] = (in0[l], in1[l], in2[l], in3[l])
inline void TransposeC4 (const __m256* in, __m256d* out) noexcept
{
    const __m256d a0 = _mm256_castps_pd(in[0]);
    const __m256d a1 = _mm256_castps_pd(in[1]);
    const __m256d a2 = _mm256_castps_pd(in[2]);
    const __m256d a3 = _mm256_castps_pd(in[3]);
    const __m256d t0 = _mm256_unpacklo_pd(a0, a1);
    const __m256d t1 = _mm256_unpackhi_pd(a0, a1);
    const __m256d t2 = _mm256_unpacklo_pd(a2, a3);
    const __m256d t3 = _mm256_unpackhi_pd(a2, a3);
    out[0] = _mm256_permute2f128_pd(t0, t2, 0x20);
    out[1] = _mm256_permute2f128_pd(t1, t3, 0x20);
    out[2] = _mm256_permute2f128_pd(t0, t2, 0x31);
    out[3] = _mm256_permute2f128_pd(t1, t3, 0x31);
}

} // namespace avx2_detail


template <>
struct Real2DColumnLane<LaneAvx2>
{
    using type = LaneAvx2x2;
};

template <>
struct Real2DLaneIO<LaneAvx2> : public Real2DLaneIOGeneric<LaneAvx2>
{
    using Base = Real2DLaneIOGeneric<LaneAvx2>;
    using V = __m256;
    static constexpr int32_t kGroup = 8;

    static inline void PackRows (const float* const* rows, int32_t width, V* RESTRICT dst) noexcept
    {
        for (int32_t q = 0; q < 8; ++q)
            if (nullptr == rows[q]) { Base::PackRows(rows, width, dst); return; }
        int32_t k = 0;
        for (; k + 8 <= width; k += 8)
        {
            __m256 r[8];
            for (int32_t q = 0; q < 8; ++q) r[q] = _mm256_loadu_ps(rows[q] + k);
            avx2_detail::Transpose8x8(r);
            for (int32_t j = 0; j < 8; ++j) dst[k + j] = r[j];
        }
        alignas(64) float tmp[8];
        for (; k < width; ++k)
        {
            for (int32_t q = 0; q < 8; ++q) tmp[q] = rows[q][k];
            dst[k] = _mm256_load_ps(tmp);
        }
    }

    static inline void StoreSeparatedGroup (float* const* specRows, int32_t k, const V* xa, const V* xb, int32_t cnt, bool full) noexcept
    {
        if (!full) { Base::StoreSeparatedGroup(specRows, k, xa, xb, cnt, false); return; }
        for (int32_t h = 0; h < 2; ++h)
        {
            __m256d ta[4], tb[4];
            avx2_detail::TransposeC4(xa + 4 * h, ta);
            avx2_detail::TransposeC4(xb + 4 * h, tb);
            for (int32_t l = 0; l < 4; ++l)
            {
                _mm256_storeu_pd(reinterpret_cast<double*>(specRows[2 * l]     + 2 * (k + 4 * h)), ta[l]);
                _mm256_storeu_pd(reinterpret_cast<double*>(specRows[2 * l + 1] + 2 * (k + 4 * h)), tb[l]);
            }
        }
    }

    static inline void LoadSeparatedGroup (const float* const* specRows, int32_t k, V* xa, V* xb, int32_t cnt, bool full) noexcept
    {
        if (!full) { Base::LoadSeparatedGroup(specRows, k, xa, xb, cnt, false); return; }
        for (int32_t h = 0; h < 2; ++h)
        {
            __m256 ra[4], rb[4];
            for (int32_t l = 0; l < 4; ++l)
            {
                ra[l] = _mm256_loadu_ps(specRows[2 * l]     + 2 * (k + 4 * h));
                rb[l] = _mm256_loadu_ps(specRows[2 * l + 1] + 2 * (k + 4 * h));
            }
            __m256d ta[4], tb[4];
            avx2_detail::TransposeC4(ra, ta);
            avx2_detail::TransposeC4(rb, tb);
            for (int32_t i = 0; i < 4; ++i)
            {
                xa[4 * h + i] = _mm256_castpd_ps(ta[i]);
                xb[4 * h + i] = _mm256_castpd_ps(tb[i]);
            }
        }
    }

    static inline void UnpackRowsGroup (float* const* rows, int32_t n, const V* z, int32_t cnt, float scale, bool full) noexcept
    {
        if (!full) { Base::UnpackRowsGroup(rows, n, z, cnt, scale, false); return; }
        __m256 r[8];
        for (int32_t i = 0; i < 8; ++i) r[i] = z[i];
        avx2_detail::Transpose8x8(r);
        const __m256 sc = _mm256_set1_ps(scale);
        for (int32_t q = 0; q < 8; ++q) _mm256_storeu_ps(rows[q] + n, _mm256_mul_ps(r[q], sc));
    }
};

} // namespace FourierTransform
