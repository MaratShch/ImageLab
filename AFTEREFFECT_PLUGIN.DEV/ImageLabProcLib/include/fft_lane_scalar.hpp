#pragma once

// ============================================================================
// fft_lane_scalar.hpp -- one complex number per "lane vector".
//
// The plan-based engine (fft_plan.hpp, fft_real2d.hpp) is written once against a
// small LANE interface and instantiated twice:
//
//     LaneScalar<T>   one complex value per vector, plain C++ (no intrinsics).
//                     Used by the scalar film engine (T = double) and by the
//                     legacy mixed_radix_fft_* entry points (T = float/double).
//     LaneAvx2        four complex floats per __m256 (fft_lane_avx2.hpp).
//                     Used by the AVX2 film engine only.
//
// Both run the SAME flow -- same factorisation, same butterflies, same twiddle
// tables, same order of operations -- so the scalar and AVX2 engines differ only
// in how many independent transforms one instruction advances.
//
// Interface every lane type provides:
//     kLanes                 complex values per V
//     V                      the vector type
//     Load(p) / Store(p, v)  kLanes interleaved complex values (re, im, re, im ...)
//     Zero()
//     Add, Sub               complex add / subtract, lane-wise
//     MulR(v, k)             multiply by a real scalar k
//     MulNJ(v)               multiply by -j        (re, im) -> ( im, -re)
//     MulPJ(v)               multiply by +j        (re, im) -> (-im,  re)
//     CMul(v, c, s)          multiply by (c + j s), same factor in every lane
//     CMulV(v, w)            multiply by w, lane-wise (w a V)
//     Conj(v)                complex conjugate
//     MulRV(v, w)            component-wise product (re*wr, im*wi) -- a REAL
//                            transfer stored as duplicated pairs (t, t)
// ============================================================================

#include <cstdint>
#include <cstddef>

namespace FourierTransform
{

template <typename T>
struct LaneScalar
{
    static constexpr int32_t kLanes = 1;
    using Real = T;

    struct V
    {
        T r;
        T i;
    };

    static inline V Load (const T* p) noexcept { V v; v.r = p[0]; v.i = p[1]; return v; }
    static inline void Store (T* p, const V& v) noexcept { p[0] = v.r; p[1] = v.i; }
    static inline V Zero () noexcept { V v; v.r = T(0); v.i = T(0); return v; }
    static inline V Make (T r, T i) noexcept { V v; v.r = r; v.i = i; return v; }

    static inline V Add (const V& a, const V& b) noexcept { V v; v.r = a.r + b.r; v.i = a.i + b.i; return v; }
    static inline V Sub (const V& a, const V& b) noexcept { V v; v.r = a.r - b.r; v.i = a.i - b.i; return v; }
    static inline V MulR (const V& a, T k) noexcept { V v; v.r = a.r * k; v.i = a.i * k; return v; }
    static inline V MulNJ (const V& a) noexcept { V v; v.r = a.i; v.i = -a.r; return v; }
    static inline V MulPJ (const V& a) noexcept { V v; v.r = -a.i; v.i = a.r; return v; }
    static inline V Conj (const V& a) noexcept { V v; v.r = a.r; v.i = -a.i; return v; }

    static inline V CMul (const V& a, T c, T s) noexcept
    {
        V v;
        v.r = a.r * c - a.i * s;
        v.i = a.r * s + a.i * c;
        return v;
    }

    static inline V MulRV (const V& a, const V& w) noexcept { V v; v.r = a.r * w.r; v.i = a.i * w.i; return v; }

    static inline V CMulV (const V& a, const V& w) noexcept
    {
        V v;
        v.r = a.r * w.r - a.i * w.i;
        v.i = a.r * w.i + a.i * w.r;
        return v;
    }
};

} // namespace FourierTransform
