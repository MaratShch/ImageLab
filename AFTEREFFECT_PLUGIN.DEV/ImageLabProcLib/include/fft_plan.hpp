#pragma once

// ============================================================================
// fft_plan.hpp -- planned mixed-radix complex FFT, caller-owned memory.
//
// WHAT CHANGED AGAINST THE ORIGINAL ITERATIVE ENGINE (fft_mixed_radix.hpp)
//
//   The algorithm is the same decimation-in-frequency mixed-radix flow: one pass
//   per factor, butterfly first, twiddle after, outputs left in digit-reversed
//   positions. What moved is everything that was recomputed per call:
//
//   1. Twiddles. The original evaluated FastCompute::SinCos for every butterfly
//      output of every stage. Here each stage owns a table built ONCE, in double,
//      with the angle reduced exactly (index mod group) before the sin/cos, then
//      rounded to T. Accuracy is better than per-call float sin/cos and the
//      stage loop does a load instead of a polynomial.
//   2. Heap. prime() returned a std::vector, unshuffle allocated a 2N temporary,
//      the CZT allocated three vectors and recursed with more. A plan is built
//      once into memory the CALLER supplies (Plan1DBuild with mem == nullptr
//      returns the byte count), and executing it never allocates. That is what
//      lets the film engine keep its one-arena-per-geometry rule.
//   3. Unshuffle. The digit-reversal permutation is a precomputed index table
//      (pos[k] = slot that holds frequency k). Callers fold it into the copy they
//      already make when they move data out of the work buffer, so there is no
//      separate permutation pass at all.
//   4. Large primes. Bluestein (chirp-z) runs on an inner power-of-two plan
//      built by this same engine -- the original used a recursive mock helper.
//      Odd prime factors up to kPlanMaxDirectPrime use a direct symmetric
//      butterfly, cheaper than Bluestein at those sizes.
//   5. Lanes. Every kernel is a template over a LANE type (fft_lane_scalar.hpp,
//      fft_lane_avx2.hpp). The same flow advances one transform (scalar) or four
//      independent transforms (AVX2) per instruction.
//
// SIGN CONVENTION: forward X[k] = sum_n x[n] exp(-2 pi j n k / N), unscaled --
// identical to numpy.fft.fft and to the original engine.
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <cmath>
#include "Common.hpp"            // RESTRICT
#include "fft_lane_scalar.hpp"

namespace FourierTransform
{

//: Most factors one length may carry (2^31 needs 11 radix-8/4/2 passes).
constexpr int32_t kPlanMaxFactors = 32;
//: Largest odd prime factor served by a direct butterfly; beyond it Bluestein.
constexpr int32_t kPlanMaxDirectPrime = 61;
//: Alignment of every table and work area carved from caller memory.
constexpr std::size_t kPlanAlign = 64;


// ----------------------------------------------------------------------------
// Bump allocator over caller memory. With base == nullptr it only measures.
// ----------------------------------------------------------------------------
struct PlanCarve
{
    unsigned char* base;
    std::size_t    used;

    template <typename X>
    X* Take (std::size_t count) noexcept
    {
        used = (used + (kPlanAlign - 1)) & ~(kPlanAlign - 1);
        X* p = (nullptr != base) ? reinterpret_cast<X*>(base + used) : nullptr;
        used += count * sizeof(X);
        return p;
    }
};


template <typename T>
struct Plan1D
{
    int32_t n;                               // transform length
    int32_t nFactors;                        // DIF passes (0 for n == 1 and for Bluestein)
    int32_t factor[kPlanMaxFactors];         // radix of each pass, first pass first
    int32_t twOffset[kPlanMaxFactors];       // stage twiddle table offset, in complex entries
    int32_t csOffset[kPlanMaxFactors];       // odd-prime constant table offset, in complex entries
    const T*       tw;                       // (c, s) pairs, stage twiddles
    const T*       cs;                       // (cos 2pi k/p, sin 2pi k/p) pairs for odd primes
    const int32_t* pos;                      // pos[k] = slot holding frequency k after Execute
    // Bluestein (chirp-z) for lengths with a prime factor > kPlanMaxDirectPrime
    int32_t        bluestein;                // 0 or 1
    int32_t        m;                        // inner power-of-two length
    const T*       chirp;                    // n pairs, exp(-j pi k^2 / n)
    const T*       kernel;                   // m pairs, FFT of the conjugate chirp kernel, scaled 1/m
    const Plan1D*  inner;                    // the length-m plan
};


// ----------------------------------------------------------------------------
// Factorisation: 8 first (fewest passes), then 4, 2, then odd primes ascending.
// Returns false when a prime factor exceeds kPlanMaxDirectPrime.
// ----------------------------------------------------------------------------
inline bool PlanFactorize (int32_t n, int32_t* f, int32_t& nf) noexcept
{
    nf = 0;
    int32_t r = n;
    while (0 == (r % 8) && r > 1) { f[nf++] = 8; r /= 8; }
    while (0 == (r % 4) && r > 1) { f[nf++] = 4; r /= 4; }
    while (0 == (r % 2) && r > 1) { f[nf++] = 2; r /= 2; }
    for (int32_t p = 3; r > 1 && p <= kPlanMaxDirectPrime; p += 2)
    {
        while (0 == (r % p)) { f[nf++] = p; r /= p; }
    }
    return (1 == r);
}


inline int32_t PlanNextPow2 (int32_t v) noexcept
{
    int32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}


// ----------------------------------------------------------------------------
// Build a plan for length n into mem (or measure, when mem == nullptr).
// mem must be aligned to kPlanAlign. Returns the number of bytes used. 'plan' is written only when mem != nullptr.
// ----------------------------------------------------------------------------
template <typename T>
std::size_t Plan1DBuild (Plan1D<T>* plan, int32_t n, void* mem) noexcept;

namespace plan_detail
{

template <typename T>
std::size_t BuildInto (Plan1D<T>* plan, int32_t n, PlanCarve& carve) noexcept
{
    const double twoPi = 6.283185307179586476925286766559;
    const std::size_t start = carve.used;
    const bool write = (nullptr != carve.base);

    int32_t f[kPlanMaxFactors];
    int32_t nf = 0;
    const bool direct = PlanFactorize(n, f, nf);

    if (direct)
    {
        // stage twiddles: stage i has group g, radix R, stride s = g / R,
        // entries (R - 1) * s when s > 1 (the last pass, s == 1, needs none).
        int32_t twTotal = 0;
        int32_t csTotal = 0;
        int32_t g = n;
        int32_t twOff[kPlanMaxFactors];
        int32_t csOff[kPlanMaxFactors];
        for (int32_t i = 0; i < nf; ++i)
        {
            const int32_t R = f[i];
            const int32_t s = g / R;
            twOff[i] = twTotal;
            twTotal += (s > 1) ? (R - 1) * s : 0;
            csOff[i] = -1;
            if (R > 2 && 0 != (R & 1))
            {
                // share one constant table per distinct prime
                for (int32_t q = 0; q < i; ++q)
                    if (f[q] == R) { csOff[i] = csOff[q]; break; }
                if (csOff[i] < 0) { csOff[i] = csTotal; csTotal += R; }
            }
            g = s;
        }

        T*       tw  = carve.Take<T>(static_cast<std::size_t>(2 * (twTotal > 0 ? twTotal : 1)));
        T*       cs  = carve.Take<T>(static_cast<std::size_t>(2 * (csTotal > 0 ? csTotal : 1)));
        int32_t* pos = carve.Take<int32_t>(static_cast<std::size_t>(n));

        if (write)
        {
            plan->n = n;
            plan->nFactors = nf;
            plan->bluestein = 0;
            plan->m = 0;
            plan->chirp = nullptr;
            plan->kernel = nullptr;
            plan->inner = nullptr;
            plan->tw = tw;
            plan->cs = cs;
            plan->pos = pos;

            g = n;
            for (int32_t i = 0; i < nf; ++i)
            {
                const int32_t R = f[i];
                const int32_t s = g / R;
                plan->factor[i] = R;
                plan->twOffset[i] = twOff[i];
                plan->csOffset[i] = csOff[i];
                if (s > 1)
                {
                    T* t = tw + 2 * twOff[i];
                    for (int32_t k = 0; k < s; ++k)
                    {
                        for (int32_t mm = 1; mm < R; ++mm)
                        {
                            // W_g^(m k) = exp(-2 pi j (m k mod g) / g)
                            const int64_t idx = (static_cast<int64_t>(mm) * k) % g;
                            const double a = -twoPi * static_cast<double>(idx) / static_cast<double>(g);
                            t[0] = static_cast<T>(std::cos(a));
                            t[1] = static_cast<T>(std::sin(a));
                            t += 2;
                        }
                    }
                }
                if (csOff[i] >= 0)
                {
                    T* c = cs + 2 * csOff[i];
                    for (int32_t k = 0; k < R; ++k)
                    {
                        const double a = twoPi * static_cast<double>(k) / static_cast<double>(R);
                        c[2 * k]     = static_cast<T>(std::cos(a));
                        c[2 * k + 1] = static_cast<T>(std::sin(a));
                    }
                }
                g = s;
            }
            for (int32_t i = nf; i < kPlanMaxFactors; ++i)
            {
                plan->factor[i] = 0; plan->twOffset[i] = 0; plan->csOffset[i] = -1;
            }

            // pos(k): first pass radix f0 sends frequency k0 = k mod f0 to
            // sub-block k0 of size n/f0, recursively.
            for (int32_t k = 0; k < n; ++k)
            {
                int32_t rem = k;
                int32_t len = n;
                int32_t p = 0;
                for (int32_t i = 0; i < nf; ++i)
                {
                    const int32_t R = f[i];
                    len /= R;
                    p += (rem % R) * len;
                    rem /= R;
                }
                pos[k] = p;
            }
        }
    }
    else
    {
        // Bluestein: X[k] = w[k] * sum_n (x[n] w[n]) conj(w[k - n]),  w[k] = exp(-j pi k^2 / n)
        const int32_t m = PlanNextPow2(2 * n - 1);
        T*         chirp = carve.Take<T>(static_cast<std::size_t>(2 * n));
        T*         kern  = carve.Take<T>(static_cast<std::size_t>(2 * m));
        int32_t*   pos   = carve.Take<int32_t>(static_cast<std::size_t>(n));
        Plan1D<T>* inner = carve.Take<Plan1D<T>>(1);
        BuildInto<T>(inner, m, carve);

        if (write)
        {
            plan->n = n;
            plan->nFactors = 0;
            for (int32_t i = 0; i < kPlanMaxFactors; ++i)
            {
                plan->factor[i] = 0; plan->twOffset[i] = 0; plan->csOffset[i] = -1;
            }
            plan->tw = nullptr;
            plan->cs = nullptr;
            plan->bluestein = 1;
            plan->m = m;
            plan->chirp = chirp;
            plan->kernel = kern;
            plan->inner = inner;
            plan->pos = pos;

            const double pi = 3.141592653589793238462643383279;
            const int64_t twoN = 2 * static_cast<int64_t>(n);
            for (int32_t k = 0; k < n; ++k)
            {
                // k^2 mod 2n keeps the angle exact for large k
                const int64_t kk = (static_cast<int64_t>(k) * k) % twoN;
                const double a = -pi * static_cast<double>(kk) / static_cast<double>(n);
                chirp[2 * k]     = static_cast<T>(std::cos(a));
                chirp[2 * k + 1] = static_cast<T>(std::sin(a));
                pos[k] = k;
            }

            // kernel b: conj(w[k]) at k and m - k, zero elsewhere; FFT in double
            // precision (naive DFT would be O(m^2); use the inner plan in T, then
            // scale by 1/m). Built through the scalar lane.
            using L = LaneScalar<T>;
            typename L::V* b = reinterpret_cast<typename L::V*>(kern);
            for (int32_t k = 0; k < m; ++k) b[k] = L::Zero();
            for (int32_t k = 0; k < n; ++k)
            {
                const int64_t kk = (static_cast<int64_t>(k) * k) % twoN;
                const double a = pi * static_cast<double>(kk) / static_cast<double>(n);   // conj
                b[k] = L::Make(static_cast<T>(std::cos(a)), static_cast<T>(std::sin(a)));
                if (k > 0) b[m - k] = b[k];
            }
        }
        // kernel transform is completed in Plan1DBuild once the inner plan exists
    }
    return carve.used - start;
}

} // namespace plan_detail


// ============================================================================
// BUTTERFLIES. Forward sign: y[m] = sum_j x[j] exp(-2 pi j j m / R).
// x[] holds R lane vectors, results overwrite it in natural order.
// ============================================================================
namespace plan_detail
{

template <class L>
inline void Bfly2 (typename L::V* x) noexcept
{
    const typename L::V a = x[0], b = x[1];
    x[0] = L::Add(a, b);
    x[1] = L::Sub(a, b);
}

template <class L>
inline void Bfly4 (typename L::V* x) noexcept
{
    const typename L::V t0 = L::Add(x[0], x[2]);
    const typename L::V t1 = L::Sub(x[0], x[2]);
    const typename L::V t2 = L::Add(x[1], x[3]);
    const typename L::V t3 = L::MulNJ(L::Sub(x[1], x[3]));   // -j (x1 - x3)
    x[0] = L::Add(t0, t2);
    x[2] = L::Sub(t0, t2);
    x[1] = L::Add(t1, t3);
    x[3] = L::Sub(t1, t3);
}

template <class L>
inline void Bfly8 (typename L::V* x) noexcept
{
    using V = typename L::V;
    using R = typename L::Real;
    const R h = static_cast<R>(0.70710678118654752440084436210485);
    V e[4] = { x[0], x[2], x[4], x[6] };
    V o[4] = { x[1], x[3], x[5], x[7] };
    Bfly4<L>(e);
    Bfly4<L>(o);
    // w8^1 = (1 - j)/sqrt2, w8^2 = -j, w8^3 = -j w8^1
    const V o1 = L::MulR(L::Add(o[1], L::MulNJ(o[1])), h);
    const V o2 = L::MulNJ(o[2]);
    const V o3 = L::MulNJ(L::MulR(L::Add(o[3], L::MulNJ(o[3])), h));
    x[0] = L::Add(e[0], o[0]);  x[4] = L::Sub(e[0], o[0]);
    x[1] = L::Add(e[1], o1);    x[5] = L::Sub(e[1], o1);
    x[2] = L::Add(e[2], o2);    x[6] = L::Sub(e[2], o2);
    x[3] = L::Add(e[3], o3);    x[7] = L::Sub(e[3], o3);
}

// Odd prime R, symmetric form. cs[2k], cs[2k+1] = cos(2 pi k / R), sin(2 pi k / R).
//   a_j = x_j + x_{R-j},  b_j = x_j - x_{R-j},  j = 1..h, h = (R-1)/2
//   y_0 = x_0 + sum a_j
//   A_m = x_0 + sum_j a_j cos(2 pi j m / R),  B_m = sum_j b_j sin(2 pi j m / R)
//   y_m = A_m - j B_m,   y_{R-m} = A_m + j B_m
template <class L, int32_t R>
inline void BflyOdd (typename L::V* x, const typename L::Real* cs) noexcept
{
    using V = typename L::V;
    constexpr int32_t h = (R - 1) / 2;
    V a[h], b[h];
    V y0 = x[0];
    for (int32_t j = 1; j <= h; ++j)
    {
        a[j - 1] = L::Add(x[j], x[R - j]);
        b[j - 1] = L::Sub(x[j], x[R - j]);
        y0 = L::Add(y0, a[j - 1]);
    }
    const V x0 = x[0];
    for (int32_t mm = 1; mm <= h; ++mm)
    {
        V A = x0;
        V B = L::Zero();
        for (int32_t j = 1; j <= h; ++j)
        {
            const int32_t idx = (j * mm) % R;
            A = L::Add(A, L::MulR(a[j - 1], cs[2 * idx]));
            B = L::Add(B, L::MulR(b[j - 1], cs[2 * idx + 1]));
        }
        const V jB = L::MulNJ(B);           // -j B
        x[mm]     = L::Add(A, jB);
        x[R - mm] = L::Sub(A, jB);
    }
    x[0] = y0;
}

// Runtime-R odd prime (R <= kPlanMaxDirectPrime), same formula.
template <class L>
inline void BflyOddN (typename L::V* x, const typename L::Real* cs, int32_t R) noexcept
{
    using V = typename L::V;
    const int32_t h = (R - 1) / 2;
    V a[(kPlanMaxDirectPrime - 1) / 2], b[(kPlanMaxDirectPrime - 1) / 2];
    V y0 = x[0];
    for (int32_t j = 1; j <= h; ++j)
    {
        a[j - 1] = L::Add(x[j], x[R - j]);
        b[j - 1] = L::Sub(x[j], x[R - j]);
        y0 = L::Add(y0, a[j - 1]);
    }
    const V x0 = x[0];
    for (int32_t mm = 1; mm <= h; ++mm)
    {
        V A = x0;
        V B = L::Zero();
        int32_t idx = 0;
        for (int32_t j = 1; j <= h; ++j)
        {
            idx += mm; if (idx >= R) idx -= R;
            A = L::Add(A, L::MulR(a[j - 1], cs[2 * idx]));
            B = L::Add(B, L::MulR(b[j - 1], cs[2 * idx + 1]));
        }
        const V jB = L::MulNJ(B);
        x[mm]     = L::Add(A, jB);
        x[R - mm] = L::Sub(A, jB);
    }
    x[0] = y0;
}


// One DIF pass of compile-time radix R over the whole buffer.
template <class L, int32_t R>
inline void PassFixed (typename L::V* RESTRICT data, int32_t n, int32_t g,
                       const typename L::Real* RESTRICT tw, const typename L::Real* RESTRICT cs) noexcept
{
    using V = typename L::V;
    const int32_t s = g / R;
    for (int32_t base = 0; base < n; base += g)
    {
        V* RESTRICT blk = data + base;
        for (int32_t k = 0; k < s; ++k)
        {
            V x[R];
            for (int32_t j = 0; j < R; ++j) x[j] = blk[k + j * s];
            if (2 == R)      Bfly2<L>(x);
            else if (4 == R) Bfly4<L>(x);
            else if (8 == R) Bfly8<L>(x);
            else             BflyOdd<L, (R > 2 ? R : 3)>(x, cs);
            blk[k] = x[0];
            if (s > 1)
            {
                const typename L::Real* t = tw + 2 * (k * (R - 1));
                for (int32_t j = 1; j < R; ++j)
                    blk[k + j * s] = L::CMul(x[j], t[2 * (j - 1)], t[2 * (j - 1) + 1]);
            }
            else
            {
                for (int32_t j = 1; j < R; ++j) blk[k + j * s] = x[j];
            }
        }
    }
}

template <class L>
inline void PassGeneric (typename L::V* RESTRICT data, int32_t n, int32_t g, int32_t R,
                         const typename L::Real* RESTRICT tw, const typename L::Real* RESTRICT cs) noexcept
{
    using V = typename L::V;
    const int32_t s = g / R;
    V x[kPlanMaxDirectPrime];
    for (int32_t base = 0; base < n; base += g)
    {
        V* RESTRICT blk = data + base;
        for (int32_t k = 0; k < s; ++k)
        {
            for (int32_t j = 0; j < R; ++j) x[j] = blk[k + j * s];
            BflyOddN<L>(x, cs, R);
            blk[k] = x[0];
            if (s > 1)
            {
                const typename L::Real* t = tw + 2 * (k * (R - 1));
                for (int32_t j = 1; j < R; ++j)
                    blk[k + j * s] = L::CMul(x[j], t[2 * (j - 1)], t[2 * (j - 1) + 1]);
            }
            else
            {
                for (int32_t j = 1; j < R; ++j) blk[k + j * s] = x[j];
            }
        }
    }
}

template <class L>
inline void RunDif (const Plan1D<typename L::Real>& p, typename L::V* data) noexcept
{
    int32_t g = p.n;
    for (int32_t i = 0; i < p.nFactors; ++i)
    {
        const int32_t R = p.factor[i];
        const typename L::Real* tw = p.tw + 2 * p.twOffset[i];
        const typename L::Real* cs = (p.csOffset[i] >= 0) ? (p.cs + 2 * p.csOffset[i]) : nullptr;
        switch (R)
        {
            case 2: PassFixed<L, 2>(data, p.n, g, tw, cs); break;
            case 4: PassFixed<L, 4>(data, p.n, g, tw, cs); break;
            case 8: PassFixed<L, 8>(data, p.n, g, tw, cs); break;
            case 3: PassFixed<L, 3>(data, p.n, g, tw, cs); break;
            case 5: PassFixed<L, 5>(data, p.n, g, tw, cs); break;
            case 7: PassFixed<L, 7>(data, p.n, g, tw, cs); break;
            default: PassGeneric<L>(data, p.n, g, R, tw, cs); break;
        }
        g /= R;
    }
}

} // namespace plan_detail


// ----------------------------------------------------------------------------
// Work vectors Plan1DExecute needs besides 'data' (0 unless Bluestein).
// ----------------------------------------------------------------------------
template <typename T>
inline int32_t Plan1DWorkVectors (const Plan1D<T>& p) noexcept
{
    return p.bluestein ? 2 * p.m : 0;
}


// ----------------------------------------------------------------------------
// In-place forward transform of kLanes independent sequences.
// On return frequency k of every lane is at data[p.pos[k]].
// 'work' must hold Plan1DWorkVectors(p) lane vectors (may be nullptr otherwise).
// ----------------------------------------------------------------------------
template <class L>
inline void Plan1DExecute (const Plan1D<typename L::Real>& p, typename L::V* RESTRICT data,
                           typename L::V* RESTRICT work) noexcept
{
    using V = typename L::V;
    using T = typename L::Real;
    if (!p.bluestein)
    {
        plan_detail::RunDif<L>(p, data);
        return;
    }
    const int32_t n = p.n;
    const int32_t m = p.m;
    const Plan1D<T>& q = *p.inner;
    V* RESTRICT a = work;
    V* RESTRICT c = work + m;
    for (int32_t k = 0; k < n; ++k) a[k] = L::CMul(data[k], p.chirp[2 * k], p.chirp[2 * k + 1]);
    for (int32_t k = n; k < m; ++k) a[k] = L::Zero();
    plan_detail::RunDif<L>(q, a);
    // C = A * B (kernel prescaled 1/m), then inverse via conj -> forward -> conj
    for (int32_t k = 0; k < m; ++k)
        c[k] = L::Conj(L::CMul(a[q.pos[k]], p.kernel[2 * k], p.kernel[2 * k + 1]));
    plan_detail::RunDif<L>(q, c);
    for (int32_t k = 0; k < n; ++k)
        data[k] = L::CMul(L::Conj(c[q.pos[k]]), p.chirp[2 * k], p.chirp[2 * k + 1]);
}


template <typename T>
std::size_t Plan1DBuild (Plan1D<T>* plan, int32_t n, void* mem) noexcept
{
    PlanCarve carve{ static_cast<unsigned char*>(mem), 0 };
    if (n < 1) return 0;
    const std::size_t bytes = plan_detail::BuildInto<T>(plan, n, carve);
    if (nullptr != mem && plan->bluestein)
    {
        // finish the Bluestein kernel: B = FFT(b) / m, stored in natural order
        using L = LaneScalar<T>;
        using V = typename L::V;
        const int32_t m = plan->m;
        V* b = reinterpret_cast<V*>(const_cast<T*>(plan->kernel));
        plan_detail::RunDif<L>(*plan->inner, b);
        // permute to natural order in place by cycle-following on pos
        // (pos is a permutation; use the sign bit of nothing -- do it via a
        //  temporary walk over cycles with a visited test on index order)
        const int32_t* pos = plan->inner->pos;
        for (int32_t start = 0; start < m; ++start)
        {
            // process each cycle once: start must be its smallest member
            int32_t k = pos[start];
            while (k > start) k = pos[k];
            if (k < start) continue;
            // natural[k] = b[pos[k]] along the cycle start -> pos[start] -> ...
            const V first = b[start];
            int32_t cur = start;
            for (;;)
            {
                const int32_t nxt = pos[cur];
                if (nxt == start) { b[cur] = first; break; }
                b[cur] = b[nxt];
                cur = nxt;
            }
        }
        const T inv = static_cast<T>(1.0 / static_cast<double>(m));
        for (int32_t k = 0; k < m; ++k) b[k] = L::MulR(b[k], inv);
    }
    return bytes + kPlanAlign;   // slack for the caller aligning 'mem'
}

} // namespace FourierTransform
