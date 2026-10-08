#pragma once

// ============================================================================
// fft_real2d.hpp -- planned 2D FFT of a REAL image, half spectrum.
//
// Spectrum layout is numpy.fft.rfft2's: height rows x (width/2 + 1) complex
// columns, row-major, interleaved (re, im). Each spectrum row is padded to
// specPitch complex values (a multiple of kSpecQuantum) so every row starts on a
// 64-byte boundary and the AVX2 column pass can always read whole vectors; the
// padding columns are written as zero by the forward pass.
//
//   Forward:  real image  ->  rfft2 spectrum        (unscaled, like numpy)
//   Inverse:  spectrum    ->  real image            (scaled by 'scale'; pass
//                                                    1/(W*H) for numpy irfft2)
//
// numpy.fft.irfft2 ignores the imaginary part of the DC column (and of the
// Nyquist column when width is even) of the last axis; Inverse does the same,
// so the two agree for spectra that are not exactly Hermitian (a complex
// transfer such as a sub-pixel shift).
//
// FLOW
//   rows    two real rows packed as one complex sequence (a + j b), transformed,
//           separated by conjugate symmetry. One complex FFT per TWO rows, any
//           width (odd included). LaneAvx2 packs 8 rows into 4 lanes.
//   columns kLanes adjacent spectrum columns transformed together; the gather
//           into the work buffer makes every pass unit-stride and the scatter
//           back applies the digit-reversal permutation for free.
//
// Inverse runs columns first, then rows, matching irfft2 (ifft along axis 0,
// irfft along axis 1). The inverse transforms use conj(FFT(conj(x))), with the
// conjugations folded into the gathers and scatters.
//
// An optional TRANSFER functor is applied while the inverse gathers the
// spectrum, so "multiply by H(f) and invert" costs no extra pass, and the input
// spectrum can be left untouched (specIn != specWork) for reuse with another
// transfer.
// ============================================================================

#include <cstdint>
#include <cstddef>
#include "fft_plan.hpp"

namespace FourierTransform
{

//: spectrum row pitch quantum, complex values (8 floats = 64 B, 8 doubles = 128 B)
constexpr int32_t kSpecQuantum = 8;

//: Lane type used by the column pass (LaneAvx2 -> LaneAvx2x2 in fft_real2d_avx2.hpp).
template <class L>
struct Real2DColumnLane
{
    using type = L;
};



template <typename T>
struct PlanReal2D
{
    int32_t   width;
    int32_t   height;
    int32_t   specWidth;     // width / 2 + 1
    int32_t   specPitch;     // complex values per spectrum row (>= specWidth)
    Plan1D<T> rows;          // length width
    Plan1D<T> cols;          // length height
};


// ----------------------------------------------------------------------------
// Build. mem == nullptr measures. mem must be kPlanAlign-aligned.
// ----------------------------------------------------------------------------
template <typename T>
inline std::size_t PlanReal2DBuild (PlanReal2D<T>* plan, int32_t width, int32_t height, void* mem) noexcept
{
    unsigned char* p = static_cast<unsigned char*>(mem);
    Plan1D<T> tmpR, tmpC;
    Plan1D<T>* pr = (nullptr != mem) ? &plan->rows : &tmpR;
    Plan1D<T>* pc = (nullptr != mem) ? &plan->cols : &tmpC;
    const std::size_t br = Plan1DBuild<T>(pr, width, nullptr);
    const std::size_t brA = (br + (kPlanAlign - 1)) & ~(kPlanAlign - 1);
    const std::size_t bc = Plan1DBuild<T>(pc, height, nullptr);
    if (nullptr != mem)
    {
        Plan1DBuild<T>(pr, width, p);
        Plan1DBuild<T>(pc, height, p + brA);
        plan->width = width;
        plan->height = height;
        plan->specWidth = width / 2 + 1;
        plan->specPitch = ((plan->specWidth + kSpecQuantum - 1) / kSpecQuantum) * kSpecQuantum;
    }
    return brA + bc;
}

template <typename T>
inline int32_t Real2DSpecPitch (int32_t width) noexcept
{
    const int32_t sw = width / 2 + 1;
    return ((sw + kSpecQuantum - 1) / kSpecQuantum) * kSpecQuantum;
}

//: Spectrum buffer size in elements of T (2 per complex).
template <typename T>
inline std::size_t Real2DSpecElements (int32_t width, int32_t height) noexcept
{
    return static_cast<std::size_t>(2) * static_cast<std::size_t>(Real2DSpecPitch<T>(width)) * static_cast<std::size_t>(height);
}

//: Work area for one call, in bytes, for lane type L.
template <class L>
inline std::size_t Real2DWorkBytes (const PlanReal2D<typename L::Real>& p) noexcept
{
    const int32_t nmax = (p.width > p.height) ? p.width : p.height;
    const int32_t wr = Plan1DWorkVectors(p.rows);
    const int32_t wc = Plan1DWorkVectors(p.cols);
    const int32_t wb = (wr > wc) ? wr : wc;
    const std::size_t vb = (sizeof(typename L::V) > sizeof(typename Real2DColumnLane<L>::type::V))
                           ? sizeof(typename L::V) : sizeof(typename Real2DColumnLane<L>::type::V);
    return (static_cast<std::size_t>(nmax) + static_cast<std::size_t>(wb)) * vb + kPlanAlign;
}


// ----------------------------------------------------------------------------
// Transfer functors for Real2DInverse. Mul(row, col0, x) returns x multiplied
// by the transfer at spectrum row 'row', columns col0 .. col0 + kLanes - 1.
// ----------------------------------------------------------------------------
template <class L>
struct TransferNone
{
    inline typename L::V Mul (int32_t, int32_t, const typename L::V& x) const noexcept { return x; }
};


// ----------------------------------------------------------------------------
// Lane helpers that differ between scalar and vector lanes: moving 2*kLanes
// real rows into kLanes packed complex lanes and back, and scattering the
// separated spectra. Generic versions go through a small array; LaneAvx2
// specialisations live in fft_real2d_avx2.hpp.
// ----------------------------------------------------------------------------
template <class L>
struct Real2DLaneIOGeneric
{
    using T = typename L::Real;
    using V = typename L::V;
    static constexpr int32_t K = L::kLanes;

    // dst[k] lane l = (rowA_l[k], rowB_l[k]); rows[2l] = A_l, rows[2l+1] = B_l, nullptr reads 0
    static inline void PackRows (const T* const* rows, int32_t width, V* RESTRICT dst) noexcept
    {
        alignas(64) T tmp[2 * K];
        for (int32_t k = 0; k < width; ++k)
        {
            for (int32_t l = 0; l < K; ++l)
            {
                tmp[2 * l]     = (nullptr != rows[2 * l])     ? rows[2 * l][k]     : T(0);
                tmp[2 * l + 1] = (nullptr != rows[2 * l + 1]) ? rows[2 * l + 1][k] : T(0);
            }
            dst[k] = L::Load(tmp);
        }
    }

    // spec rows: specRows[2l] receives lane l of xa, specRows[2l+1] lane l of xb, at column k
    static inline void StoreSeparated (T* const* specRows, int32_t k, const V& xa, const V& xb) noexcept
    {
        alignas(64) T ta[2 * K];
        alignas(64) T tb[2 * K];
        L::Store(ta, xa);
        L::Store(tb, xb);
        for (int32_t l = 0; l < K; ++l)
        {
            if (nullptr != specRows[2 * l])     { specRows[2 * l][2 * k]     = ta[2 * l]; specRows[2 * l][2 * k + 1]     = ta[2 * l + 1]; }
            if (nullptr != specRows[2 * l + 1]) { specRows[2 * l + 1][2 * k] = tb[2 * l]; specRows[2 * l + 1][2 * k + 1] = tb[2 * l + 1]; }
        }
    }

    // gather lane l of xa / xb from spectrum rows at column k (nullptr reads 0)
    static inline void LoadSeparated (const T* const* specRows, int32_t k, V& xa, V& xb) noexcept
    {
        alignas(64) T ta[2 * K];
        alignas(64) T tb[2 * K];
        for (int32_t l = 0; l < K; ++l)
        {
            const T* ra = specRows[2 * l];
            const T* rb = specRows[2 * l + 1];
            ta[2 * l] = ra ? ra[2 * k] : T(0);  ta[2 * l + 1] = ra ? ra[2 * k + 1] : T(0);
            tb[2 * l] = rb ? rb[2 * k] : T(0);  tb[2 * l + 1] = rb ? rb[2 * k + 1] : T(0);
        }
        xa = L::Load(ta);
        xb = L::Load(tb);
    }

    // z lane l = (a, b) -> rows: A_l[n] = re * scale, B_l[n] = im * scale
    static inline void UnpackRows (T* const* rows, int32_t n, const V& z, T scale) noexcept
    {
        alignas(64) T tz[2 * K];
        L::Store(tz, z);
        for (int32_t l = 0; l < K; ++l)
        {
            if (nullptr != rows[2 * l])     rows[2 * l][n]     = tz[2 * l] * scale;
            if (nullptr != rows[2 * l + 1]) rows[2 * l + 1][n] = tz[2 * l + 1] * scale;
        }
    }

    // ---- group forms (kGroup consecutive columns); 'full' means every row
    //      pointer is valid and the group is complete -- the AVX2
    //      specialisation transposes in registers only then.
    static constexpr int32_t kGroup = 8;

    static inline void StoreSeparatedGroup (T* const* specRows, int32_t k, const V* xa, const V* xb, int32_t cnt, bool) noexcept
    {
        for (int32_t i = 0; i < cnt; ++i) StoreSeparated(specRows, k + i, xa[i], xb[i]);
    }
    static inline void LoadSeparatedGroup (const T* const* specRows, int32_t k, V* xa, V* xb, int32_t cnt, bool) noexcept
    {
        for (int32_t i = 0; i < cnt; ++i) LoadSeparated(specRows, k + i, xa[i], xb[i]);
    }
    static inline void UnpackRowsGroup (T* const* rows, int32_t n, const V* z, int32_t cnt, T scale, bool) noexcept
    {
        for (int32_t i = 0; i < cnt; ++i) UnpackRows(rows, n + i, z[i], scale);
    }
};

//: Specialised for LaneAvx2 in fft_real2d_avx2.hpp.
template <class L>
struct Real2DLaneIO : public Real2DLaneIOGeneric<L>
{
};


namespace real2d_detail
{

template <class L>
inline typename L::V* AlignWork (void* work) noexcept
{
    std::size_t a = reinterpret_cast<std::size_t>(work);
    a = (a + (kPlanAlign - 1)) & ~(kPlanAlign - 1);
    return reinterpret_cast<typename L::V*>(a);
}

} // namespace real2d_detail


// ----------------------------------------------------------------------------
// FORWARD: src (height rows, srcPitch elements apart) -> spec (rfft2 layout,
// specPitch complex per row). Unscaled.
// ----------------------------------------------------------------------------
template <class L>
inline void Real2DForward (const PlanReal2D<typename L::Real>& p,
                           const typename L::Real* RESTRICT src, std::ptrdiff_t srcPitch,
                           typename L::Real* RESTRICT spec, void* work) noexcept
{
    using T  = typename L::Real;
    using V  = typename L::V;
    using IO = Real2DLaneIO<L>;
    constexpr int32_t K = L::kLanes;
    const int32_t W = p.width;
    const int32_t H = p.height;
    const int32_t SW = p.specWidth;
    const std::ptrdiff_t SP = 2 * static_cast<std::ptrdiff_t>(p.specPitch);
    V* RESTRICT buf = real2d_detail::AlignWork<L>(work);
    V* RESTRICT blu = buf + ((W > H) ? W : H);
    const int32_t* RESTRICT pos = p.rows.pos;

    // ---- rows: 2K real rows per batch ------------------------------------
    for (int32_t r0 = 0; r0 < H; r0 += 2 * K)
    {
        const T* rows[2 * K];
        T* srows[2 * K];
        for (int32_t q = 0; q < 2 * K; ++q)
        {
            const int32_t r = r0 + q;
            rows[q]  = (r < H) ? (src + r * srcPitch) : nullptr;
            srows[q] = (r < H) ? (spec + r * SP) : nullptr;
        }
        const bool full = (r0 + 2 * K <= H);
        IO::PackRows(rows, W, buf);
        Plan1DExecute<L>(p.rows, buf, blu);
        const T half = static_cast<T>(0.5);
        constexpr int32_t G = IO::kGroup;
        for (int32_t k0 = 0; k0 < SW; k0 += G)
        {
            const int32_t cnt = (SW - k0 < G) ? (SW - k0) : G;
            V xa[G], xb[G];
            for (int32_t i = 0; i < cnt; ++i)
            {
                const int32_t k  = k0 + i;
                const int32_t km = (0 == k) ? 0 : (W - k);
                const V z  = buf[pos[k]];
                const V zc = L::Conj(buf[pos[km]]);
                xa[i] = L::MulR(L::Add(z, zc), half);               // (Z + conj Z[-k]) / 2
                xb[i] = L::MulR(L::MulNJ(L::Sub(z, zc)), half);     // (Z - conj Z[-k]) / 2j
            }
            IO::StoreSeparatedGroup(srows, k0, xa, xb, cnt, full && (G == cnt));
        }
        // zero the padding columns
        for (int32_t q = 0; q < 2 * K; ++q)
            if (nullptr != srows[q])
                for (int32_t k = SW; k < p.specPitch; ++k) { srows[q][2 * k] = T(0); srows[q][2 * k + 1] = T(0); }
    }

    // ---- columns: KC spectrum columns per batch ----------------------------
    {
        using LC = typename Real2DColumnLane<L>::type;
        using VC = typename LC::V;
        constexpr int32_t KC = LC::kLanes;
        VC* RESTRICT cbuf = reinterpret_cast<VC*>(buf);
        VC* RESTRICT cblu = cbuf + H;
        const int32_t* RESTRICT cpos = p.cols.pos;
        for (int32_t c0 = 0; c0 < SW; c0 += KC)
        {
            T* col = spec + 2 * c0;
            for (int32_t r = 0; r < H; ++r) cbuf[r] = LC::Load(col + r * SP);
            Plan1DExecute<LC>(p.cols, cbuf, cblu);
            for (int32_t k = 0; k < H; ++k) LC::Store(col + k * SP, cbuf[cpos[k]]);
        }
    }
}


// ----------------------------------------------------------------------------
// INVERSE: specIn (optionally times transfer) -> dst (height rows, dstPitch).
// specWork receives the column-inverted spectrum; it may equal specIn (then
// specIn is consumed). Output is multiplied by 'scale' (1/(W*H) = numpy).
// ----------------------------------------------------------------------------
template <class L, class Transfer>
inline void Real2DInverse (const PlanReal2D<typename L::Real>& p,
                           const typename L::Real* specIn, const Transfer& transfer,
                           typename L::Real* specWork,
                           typename L::Real* RESTRICT dst, std::ptrdiff_t dstPitch,
                           typename L::Real scale, void* work) noexcept
{
    using T  = typename L::Real;
    using V  = typename L::V;
    using IO = Real2DLaneIO<L>;
    constexpr int32_t K = L::kLanes;
    const int32_t W = p.width;
    const int32_t H = p.height;
    const int32_t SW = p.specWidth;
    const std::ptrdiff_t SP = 2 * static_cast<std::ptrdiff_t>(p.specPitch);
    V* RESTRICT buf = real2d_detail::AlignWork<L>(work);
    V* RESTRICT blu = buf + ((W > H) ? W : H);

    // ---- columns: inverse = conj(FFT(conj(x))), unscaled -------------------
    // The transfer is applied per K-lane half so one functor serves both the
    // row lane and the (possibly wider) column lane.
    {
        using LC = typename Real2DColumnLane<L>::type;
        using VC = typename LC::V;
        constexpr int32_t KC = LC::kLanes;
        VC* RESTRICT cbuf = reinterpret_cast<VC*>(buf);
        VC* RESTRICT cblu = cbuf + H;
        const int32_t* RESTRICT cpos = p.cols.pos;
        for (int32_t c0 = 0; c0 < SW; c0 += KC)
        {
            const T* cin = specIn + 2 * c0;
            T* cout = specWork + 2 * c0;
            for (int32_t r = 0; r < H; ++r)
            {
                alignas(64) T tmp[2 * KC];
                for (int32_t h = 0; h < KC; h += K)
                    L::Store(tmp + 2 * h, L::Conj(transfer.Mul(r, c0 + h, L::Load(cin + r * SP + 2 * h))));
                cbuf[r] = LC::Load(tmp);
            }
            Plan1DExecute<LC>(p.cols, cbuf, cblu);
            for (int32_t k = 0; k < H; ++k) LC::Store(cout + k * SP, LC::Conj(cbuf[cpos[k]]));
        }
    }

    // ---- rows: 2K rows per batch, Hermitian extension, packed c2r ---------
    const int32_t* RESTRICT pos = p.rows.pos;
    const bool evenW = (0 == (W & 1));
    for (int32_t r0 = 0; r0 < H; r0 += 2 * K)
    {
        const T* srows[2 * K];
        T* drows[2 * K];
        for (int32_t q = 0; q < 2 * K; ++q)
        {
            const int32_t r = r0 + q;
            srows[q] = (r < H) ? (specWork + r * SP) : nullptr;
            drows[q] = (r < H) ? (dst + r * dstPitch) : nullptr;
        }
        // Z[k] = Xa[k] + j Xb[k]; for k > W/2, X[k] = conj(X[W-k]).
        // Fill buf with conj(Z) (inverse via conjugation).
        const bool full = (r0 + 2 * K <= H);
        constexpr int32_t G = IO::kGroup;
        for (int32_t k0 = 0; k0 < SW; k0 += G)
        {
            const int32_t cnt = (SW - k0 < G) ? (SW - k0) : G;
            V xg[G], yg[G];
            IO::LoadSeparatedGroup(srows, k0, xg, yg, cnt, full && (G == cnt));
            for (int32_t i = 0; i < cnt; ++i)
            {
                const int32_t k = k0 + i;
                V xa = xg[i], xb = yg[i];
                if (0 == k || (evenW && (2 * k == W)))
                {
                    // c2r ignores the imaginary part of these bins
                    xa = L::MulR(L::Add(xa, L::Conj(xa)), static_cast<T>(0.5));
                    xb = L::MulR(L::Add(xb, L::Conj(xb)), static_cast<T>(0.5));
                }
                // conj(Xa + j Xb) = conj(Xa) - j conj(Xb)
                buf[k] = L::Add(L::Conj(xa), L::MulNJ(L::Conj(xb)));
                const int32_t km = W - k;
                if (k > 0 && km > k && km < W)
                {
                    // Z[W-k] = conj(Xa) + j conj(Xb); conj(Z[W-k]) = Xa - j Xb
                    buf[km] = L::Add(xa, L::MulNJ(xb));
                }
            }
        }
        Plan1DExecute<L>(p.rows, buf, blu);
        // z[n] = conj(buf[pos[n]]): a = re, b = -im
        for (int32_t n0 = 0; n0 < W; n0 += G)
        {
            const int32_t cnt = (W - n0 < G) ? (W - n0) : G;
            V zg[G];
            for (int32_t i = 0; i < cnt; ++i) zg[i] = L::Conj(buf[pos[n0 + i]]);
            IO::UnpackRowsGroup(drows, n0, zg, cnt, scale, full && (G == cnt));
        }
    }
}

} // namespace FourierTransform
