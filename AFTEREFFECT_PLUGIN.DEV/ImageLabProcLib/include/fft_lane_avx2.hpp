#pragma once

// ============================================================================
// fft_lane_avx2.hpp -- four complex floats per __m256 (AVX2 + FMA3).
//
// Include ONLY from translation units compiled with -mavx2 -mfma (/arch:AVX2).
// No AVX-512.
//
// Layout of one V: (re0, im0, re1, im1, re2, im2, re3, im3) -- four INDEPENDENT
// transforms advanced together. Inside the plan engine the four lanes are four
// adjacent spectrum columns (column pass) or four packed pairs of image rows
// (row pass), so every twiddle factor is the SAME in all four lanes and is
// broadcast, never shuffled.
// ============================================================================

#include <immintrin.h>
#include <cstdint>
#include <cstddef>

// MSVC defines __AVX2__ under /arch:AVX2 (which also enables FMA3) but never
// defines __FMA__, so FMA is only checked on GCC / Clang.
#if !defined(__AVX2__)
 #error "fft_lane_avx2.hpp requires AVX2 (-mavx2 -mfma, or /arch:AVX2)"
#endif
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__FMA__)
 #error "fft_lane_avx2.hpp requires FMA3 (add -mfma)"
#endif

namespace FourierTransform
{

struct LaneAvx2
{
    static constexpr int32_t kLanes = 4;
    using Real = float;
    using V = __m256;

    static inline V Load (const float* p) noexcept { return _mm256_loadu_ps(p); }
    static inline void Store (float* p, const V& v) noexcept { _mm256_storeu_ps(p, v); }
    static inline V Zero () noexcept { return _mm256_setzero_ps(); }

    static inline V Add (const V& a, const V& b) noexcept { return _mm256_add_ps(a, b); }
    static inline V Sub (const V& a, const V& b) noexcept { return _mm256_sub_ps(a, b); }
    static inline V MulR (const V& a, float k) noexcept { return _mm256_mul_ps(a, _mm256_set1_ps(k)); }

    // (re, im) -> (im, -re)
    static inline V MulNJ (const V& a) noexcept
    {
        const V sw = _mm256_permute_ps(a, 0xB1);                       // (im, re)
        return _mm256_xor_ps(sw, _mm256_castsi256_ps(_mm256_setr_epi32(0, int32_t(0x80000000), 0, int32_t(0x80000000),
                                                                      0, int32_t(0x80000000), 0, int32_t(0x80000000))));
    }

    // (re, im) -> (-im, re)
    static inline V MulPJ (const V& a) noexcept
    {
        const V sw = _mm256_permute_ps(a, 0xB1);                       // (im, re)
        return _mm256_xor_ps(sw, _mm256_castsi256_ps(_mm256_setr_epi32(int32_t(0x80000000), 0, int32_t(0x80000000), 0,
                                                                      int32_t(0x80000000), 0, int32_t(0x80000000), 0)));
    }

    static inline V Conj (const V& a) noexcept
    {
        return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_setr_epi32(0, int32_t(0x80000000), 0, int32_t(0x80000000),
                                                                     0, int32_t(0x80000000), 0, int32_t(0x80000000))));
    }

    // a * (c + j s), the same factor in every lane:
    //   re' = re*c - im*s,  im' = im*c + re*s   ->  fmaddsub(a, c, swap(a)*s)
    static inline V CMul (const V& a, float c, float s) noexcept
    {
        const V sw = _mm256_permute_ps(a, 0xB1);
        return _mm256_fmaddsub_ps(a, _mm256_set1_ps(c), _mm256_mul_ps(sw, _mm256_set1_ps(s)));
    }

    static inline V MulRV (const V& a, const V& w) noexcept { return _mm256_mul_ps(a, w); }

    // a * w, lane-wise
    static inline V CMulV (const V& a, const V& w) noexcept
    {
        const V wr = _mm256_moveldup_ps(w);                            // (wr, wr)
        const V wi = _mm256_movehdup_ps(w);                            // (wi, wi)
        const V sw = _mm256_permute_ps(a, 0xB1);
        return _mm256_fmaddsub_ps(a, wr, _mm256_mul_ps(sw, wi));
    }
};

} // namespace FourierTransform

namespace FourierTransform
{

// ----------------------------------------------------------------------------
// LaneAvx2x2 -- eight complex floats (two __m256) per vector. Used for the
// column pass of the real 2D plan: eight adjacent spectrum columns are one full
// 64-byte cache line per spectrum row, so a column batch reads and writes whole
// lines and every twiddle broadcast is amortised over twice the work.
// ----------------------------------------------------------------------------
struct LaneAvx2x2
{
    static constexpr int32_t kLanes = 8;
    using Real = float;
    struct V { __m256 a; __m256 b; };
    using H = LaneAvx2;

    static inline V Load (const float* p) noexcept { V v; v.a = _mm256_loadu_ps(p); v.b = _mm256_loadu_ps(p + 8); return v; }
    static inline void Store (float* p, const V& v) noexcept { _mm256_storeu_ps(p, v.a); _mm256_storeu_ps(p + 8, v.b); }
    static inline V Zero () noexcept { V v; v.a = _mm256_setzero_ps(); v.b = v.a; return v; }
    static inline V Add (const V& x, const V& y) noexcept { V v; v.a = H::Add(x.a, y.a); v.b = H::Add(x.b, y.b); return v; }
    static inline V Sub (const V& x, const V& y) noexcept { V v; v.a = H::Sub(x.a, y.a); v.b = H::Sub(x.b, y.b); return v; }
    static inline V MulR (const V& x, float k) noexcept { const __m256 kk = _mm256_set1_ps(k); V v; v.a = _mm256_mul_ps(x.a, kk); v.b = _mm256_mul_ps(x.b, kk); return v; }
    static inline V MulNJ (const V& x) noexcept { V v; v.a = H::MulNJ(x.a); v.b = H::MulNJ(x.b); return v; }
    static inline V MulPJ (const V& x) noexcept { V v; v.a = H::MulPJ(x.a); v.b = H::MulPJ(x.b); return v; }
    static inline V Conj (const V& x) noexcept { V v; v.a = H::Conj(x.a); v.b = H::Conj(x.b); return v; }
    static inline V CMul (const V& x, float c, float s) noexcept
    {
        const __m256 cc = _mm256_set1_ps(c);
        const __m256 ss = _mm256_set1_ps(s);
        V v;
        v.a = _mm256_fmaddsub_ps(x.a, cc, _mm256_mul_ps(_mm256_permute_ps(x.a, 0xB1), ss));
        v.b = _mm256_fmaddsub_ps(x.b, cc, _mm256_mul_ps(_mm256_permute_ps(x.b, 0xB1), ss));
        return v;
    }
    static inline V CMulV (const V& x, const V& w) noexcept { V v; v.a = H::CMulV(x.a, w.a); v.b = H::CMulV(x.b, w.b); return v; }
    static inline V MulRV (const V& x, const V& w) noexcept { V v; v.a = _mm256_mul_ps(x.a, w.a); v.b = _mm256_mul_ps(x.b, w.b); return v; }
};

} // namespace FourierTransform
