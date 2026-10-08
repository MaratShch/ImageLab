// ============================================================================
// fft.cpp -- the original public entry points, now driven by the planned engine.
//
// Signatures and semantics are unchanged (interleaved complex, forward
// unscaled, inverse scaled by 1/N, 'in' may equal 'out', 'scratch' holds
// 2 * width * height elements for the 2D calls -- kept for compatibility, no
// longer needed).
//
// What changed underneath: every call used to factorise the length into a
// std::vector, evaluate SinCos per butterfly output, allocate a 2N temporary
// for the unshuffle, transpose twice, and fall back to an O(N^2) DFT (N < 128)
// or the mock recursive CZT for lengths with a prime factor above 9. Now the
// call takes a cached Plan1D (fft_plan.hpp): twiddles tabulated once in
// double, odd primes up to 61 by direct butterflies, Bluestein on an inner
// planned power of two beyond that, digit-reversal folded into the copy-out.
//
// Plans are cached per thread for the most recently used lengths, so repeated
// calls at one size allocate nothing after the first. These entry points use
// the plain-C++ lane (no intrinsics); the vectorised path is the real 2D plan
// (fft_real2d.hpp + fft_real2d_avx2.hpp) used directly by the film engine.
// ============================================================================

#include <vector>
#include <cstring>
#include "fft.hpp"
#include "fft_plan.hpp"
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
 #include "fft_real2d_avx2.hpp"     // LaneAvx2, LaneAvx2x2, TransposeC4
 #define FFT_LEGACY_AVX2 1
#else
 #define FFT_LEGACY_AVX2 0
#endif

namespace
{

template <typename T>
struct CachedPlan
{
    int32_t n = 0;
    std::vector<unsigned char> mem;
    FourierTransform::Plan1D<T> plan{};
    std::vector<typename FourierTransform::LaneScalar<T>::V> work;   // n + Bluestein work
    std::vector<unsigned char> wide;                                 // same, 64-byte vectors (AVX2 path)
};

constexpr int32_t kCacheSlots = 4;

template <typename T>
CachedPlan<T>& GetPlan (int32_t n)
{
    thread_local CachedPlan<T> slots[kCacheSlots];
    thread_local int32_t next = 0;
    for (int32_t i = 0; i < kCacheSlots; ++i)
        if (slots[i].n == n) return slots[i];
    CachedPlan<T>& c = slots[next];
    next = (next + 1) % kCacheSlots;
    const std::size_t bytes = FourierTransform::Plan1DBuild<T>(&c.plan, n, nullptr);
    c.mem.assign(bytes + FourierTransform::kPlanAlign, 0);
    std::size_t a = reinterpret_cast<std::size_t>(c.mem.data());
    a = (a + (FourierTransform::kPlanAlign - 1)) & ~(FourierTransform::kPlanAlign - 1);
    FourierTransform::Plan1DBuild<T>(&c.plan, n, reinterpret_cast<void*>(a));
    c.work.resize(static_cast<std::size_t>(n) + static_cast<std::size_t>(FourierTransform::Plan1DWorkVectors(c.plan)) + 1);
    c.n = n;
    return c;
}

// Forward (inverse == false) or inverse (scaled 1/n) transform of one sequence
// read with stride 'si' and written with stride 'so' (both in complex units).
template <typename T>
void Run1D (const T* in, std::ptrdiff_t si, T* out, std::ptrdiff_t so, int32_t n, bool inverse)
{
    using L = FourierTransform::LaneScalar<T>;
    using V = typename L::V;
    if (n < 1) return;
    CachedPlan<T>& c = GetPlan<T>(n);
    V* buf = c.work.data();
    V* blu = buf + n;
    for (int32_t k = 0; k < n; ++k)
    {
        const V v = L::Load(in + 2 * k * si);
        buf[k] = inverse ? L::Conj(v) : v;
    }
    FourierTransform::Plan1DExecute<L>(c.plan, buf, blu);
    const T sc = inverse ? static_cast<T>(1.0 / static_cast<double>(n)) : static_cast<T>(1);
    const int32_t* pos = c.plan.pos;
    for (int32_t k = 0; k < n; ++k)
    {
        V v = buf[pos[k]];
        if (inverse) v = L::MulR(L::Conj(v), sc);
        L::Store(out + 2 * k * so, v);
    }
}

template <typename T>
void Run2D (const T* in, T* out, std::ptrdiff_t width, std::ptrdiff_t height, bool inverse)
{
    const int32_t W = static_cast<int32_t>(width);
    const int32_t H = static_cast<int32_t>(height);
    for (int32_t r = 0; r < H; ++r)
        Run1D<T>(in + 2 * static_cast<std::ptrdiff_t>(r) * W, 1, out + 2 * static_cast<std::ptrdiff_t>(r) * W, 1, W, inverse);
    for (int32_t c = 0; c < W; ++c)
        Run1D<T>(out + 2 * c, W, out + 2 * c, W, H, inverse);
}

#if FFT_LEGACY_AVX2
// ----------------------------------------------------------------------------
// AVX2 build of the library: the complex 2D entry points (float) advance four
// rows / eight columns per instruction through the same planned flow. Rows are
// moved into lanes with 4x4 complex transposes, columns are read as whole
// 64-byte lines. Leftover rows/columns go through the scalar Run1D.
// ----------------------------------------------------------------------------
inline FourierTransform::LaneAvx2x2::V* WideWork (CachedPlan<float>& c)
{
    const std::size_t need = (static_cast<std::size_t>(c.n) + static_cast<std::size_t>(FourierTransform::Plan1DWorkVectors(c.plan)))
                             * sizeof(FourierTransform::LaneAvx2x2::V) + FourierTransform::kPlanAlign;
    if (c.wide.size() < need) c.wide.assign(need, 0);
    std::size_t a = reinterpret_cast<std::size_t>(c.wide.data());
    a = (a + (FourierTransform::kPlanAlign - 1)) & ~(FourierTransform::kPlanAlign - 1);
    return reinterpret_cast<FourierTransform::LaneAvx2x2::V*>(a);
}

template <>
void Run2D<float> (const float* in, float* out, std::ptrdiff_t width, std::ptrdiff_t height, bool inverse)
{
    using namespace FourierTransform;
    using L4 = LaneAvx2;
    using L8 = LaneAvx2x2;
    const int32_t W = static_cast<int32_t>(width);
    const int32_t H = static_cast<int32_t>(height);
    const std::ptrdiff_t RP = 2 * static_cast<std::ptrdiff_t>(W);     // floats per row

    // ---- rows, four at a time -------------------------------------------
    {
        CachedPlan<float>& c = GetPlan<float>(W);
        __m256* buf = reinterpret_cast<__m256*>(WideWork(c));
        __m256* blu = buf + W;
        const int32_t* pos = c.plan.pos;
        const float sc = inverse ? static_cast<float>(1.0 / static_cast<double>(W)) : 1.0f;
        int32_t r0 = 0;
        for (; r0 + 4 <= H; r0 += 4)
        {
            const float* src[4] = { in + r0 * RP, in + (r0 + 1) * RP, in + (r0 + 2) * RP, in + (r0 + 3) * RP };
            float* dst[4] = { out + r0 * RP, out + (r0 + 1) * RP, out + (r0 + 2) * RP, out + (r0 + 3) * RP };
            int32_t k = 0;
            for (; k + 4 <= W; k += 4)
            {
                __m256 v[4];
                for (int32_t l = 0; l < 4; ++l) v[l] = _mm256_loadu_ps(src[l] + 2 * k);
                __m256d t[4];
                avx2_detail::TransposeC4(v, t);
                for (int32_t i = 0; i < 4; ++i)
                    buf[k + i] = inverse ? L4::Conj(_mm256_castpd_ps(t[i])) : _mm256_castpd_ps(t[i]);
            }
            alignas(64) float tmp[8];
            for (; k < W; ++k)
            {
                for (int32_t l = 0; l < 4; ++l) { tmp[2 * l] = src[l][2 * k]; tmp[2 * l + 1] = src[l][2 * k + 1]; }
                buf[k] = inverse ? L4::Conj(_mm256_load_ps(tmp)) : _mm256_load_ps(tmp);
            }
            Plan1DExecute<L4>(c.plan, buf, blu);
            k = 0;
            for (; k + 4 <= W; k += 4)
            {
                __m256 v[4];
                for (int32_t i = 0; i < 4; ++i)
                {
                    v[i] = buf[pos[k + i]];
                    if (inverse) v[i] = L4::MulR(L4::Conj(v[i]), sc);
                }
                __m256d t[4];
                avx2_detail::TransposeC4(v, t);
                for (int32_t l = 0; l < 4; ++l) _mm256_storeu_pd(reinterpret_cast<double*>(dst[l] + 2 * k), t[l]);
            }
            for (; k < W; ++k)
            {
                __m256 v = buf[pos[k]];
                if (inverse) v = L4::MulR(L4::Conj(v), sc);
                _mm256_store_ps(tmp, v);
                for (int32_t l = 0; l < 4; ++l) { dst[l][2 * k] = tmp[2 * l]; dst[l][2 * k + 1] = tmp[2 * l + 1]; }
            }
        }
        for (; r0 < H; ++r0) Run1D<float>(in + r0 * RP, 1, out + r0 * RP, 1, W, inverse);
    }

    // ---- columns, eight at a time ------------------------------------------
    {
        CachedPlan<float>& c = GetPlan<float>(H);
        L8::V* buf = WideWork(c);
        L8::V* blu = buf + H;
        const int32_t* pos = c.plan.pos;
        const float sc = inverse ? static_cast<float>(1.0 / static_cast<double>(H)) : 1.0f;
        int32_t c0 = 0;
        for (; c0 + 8 <= W; c0 += 8)
        {
            float* col = out + 2 * c0;
            for (int32_t r = 0; r < H; ++r)
            {
                const L8::V v = L8::Load(col + r * RP);
                buf[r] = inverse ? L8::Conj(v) : v;
            }
            Plan1DExecute<L8>(c.plan, buf, blu);
            for (int32_t k = 0; k < H; ++k)
            {
                L8::V v = buf[pos[k]];
                if (inverse) v = L8::MulR(L8::Conj(v), sc);
                L8::Store(col + k * RP, v);
            }
        }
        for (; c0 < W; ++c0) Run1D<float>(out + 2 * c0, W, out + 2 * c0, W, H, inverse);
    }
}
#endif // FFT_LEGACY_AVX2

} // namespace


void FourierTransform::mixed_radix_fft_1D (const float* in, float* out, ptrdiff_t size) noexcept
{
    Run1D<float>(in, 1, out, 1, static_cast<int32_t>(size), false);
}

void FourierTransform::mixed_radix_fft_1D (const double* in, double* out, ptrdiff_t size) noexcept
{
    Run1D<double>(in, 1, out, 1, static_cast<int32_t>(size), false);
}

void FourierTransform::mixed_radix_ifft_1D (const float* RESTRICT in, float* RESTRICT out, ptrdiff_t size) noexcept
{
    Run1D<float>(in, 1, out, 1, static_cast<int32_t>(size), true);
}

void FourierTransform::mixed_radix_ifft_1D (const double* RESTRICT in, double* RESTRICT out, ptrdiff_t size) noexcept
{
    Run1D<double>(in, 1, out, 1, static_cast<int32_t>(size), true);
}

void FourierTransform::mixed_radix_fft_2D (const float* RESTRICT in, float* RESTRICT scratch, float* RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept
{
    (void)scratch;
    Run2D<float>(in, out, width, height, false);
}

void FourierTransform::mixed_radix_fft_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept
{
    (void)scratch;
    Run2D<double>(in, out, width, height, false);
}

void FourierTransform::mixed_radix_ifft_2D (const float* RESTRICT in, float* RESTRICT scratch, float* RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept
{
    (void)scratch;
    Run2D<float>(in, out, width, height, true);
}

void FourierTransform::mixed_radix_ifft_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept
{
    (void)scratch;
    Run2D<double>(in, out, width, height, true);
}
