// ---------------------------------------------------------------------------
//  Algo_05_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the luminance plane and the above-threshold extraction. The blur
//  itself is the shared primitive, already vectorised in AlgoSeparableBlur.cpp, which
//  is where nearly all of this stage's time actually goes.
//
//  Pipeline stage 5, in exposure space:
//
//      AlgoSoftplus            numerically safe k * log(1 + exp(x/k))
//      AlgoStage05_Halation    base-reflection halo, energy conserving
//
//  Raw pointers, explicit geometry, no allocation, no mutable state, no
//  validation of inputs.
//
//  ALIGNMENT: EVERY IMAGE ACCESS IS UNALIGNED, DELIBERATELY.
//
//  loadu/storeu rather than load/store on all plane data. The arena's base comes from
//  the host's memory pool, whose alignment argument is a HINT and not a guarantee -
//  the pool was observed returning 0x7fbef37fc010, which is 16 mod 32. Every plane is
//  then that base plus a multiple of the cache line, so every plane is 16 mod 32 too:
//  harmless to the scalar path, and an instant fault for an aligned 256-bit load.
//
//  The alternative was to align the head inside AlgoMemHandler.cpp, but that file is
//  SHARED by both flavours, and making shared infrastructure carry a vector-path
//  concern is the wrong direction - it would also mean the two builds no longer use
//  the same allocator, which is exactly the incompatibility to avoid.
//
//  The cost is nothing measurable. On Haswell and later an unaligned load of data that
//  happens to be aligned runs at the same rate as an aligned one; the only penalty is
//  on cache-line splits, which a 16-byte-offset base produces regardless of which
//  intrinsic is used. What is gained is that this code cannot fault on any base the
//  pool chooses to hand back.
//
//  The one aligned load that REMAINS is the tail-mask table, which is a file-local
//  static carrying AVX2_ALIGN - its alignment is guaranteed by the compiler, not by
//  the allocator.
// ---------------------------------------------------------------------------

// Common.hpp -- AVX2_ALIGN / CACHE_ALIGN are defined here. Included
// DIRECTLY rather than relied on transitively: this file declares an
// aligned buffer, so the macro must not depend on another header's
// include order to be in scope.
#include "Common.hpp"
#include "AlgoHalation.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>


static_assert(sizeof(AlgoType) == 4,
              "the AVX2 path requires AlgoType to be a 32-bit float");

namespace
{
    // ----------------------------------------------------------------------
    //  Lanes in one AVX2 vector of float, and the tail mask for the final
    //  partial vector of a row.
    //
    //  The active width is not generally a multiple of eight - 1023, 1998 and 2816
    //  all appear in the test set. Masked access leaves the row padding untouched,
    //  which keeps the NaN-poison arena test meaningful and stays correct even if
    //  row padding were ever removed.
    // ----------------------------------------------------------------------
    constexpr int32_t ALGO_AVX2_LANES_LOCAL = 8;

    inline __m256i algoTailMaskLocal (const int32_t n) noexcept
    {
        AVX2_ALIGN static const int32_t table[8][8] =
        {
            { 0,  0,  0,  0,  0,  0,  0,  0},
            {-1,  0,  0,  0,  0,  0,  0,  0},
            {-1, -1,  0,  0,  0,  0,  0,  0},
            {-1, -1, -1,  0,  0,  0,  0,  0},
            {-1, -1, -1, -1,  0,  0,  0,  0},
            {-1, -1, -1, -1, -1,  0,  0,  0},
            {-1, -1, -1, -1, -1, -1,  0,  0},
            {-1, -1, -1, -1, -1, -1, -1,  0}
        };
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(&table[n & 7][0]));
    }
}

#include <cmath>   // std::log1p, std::exp, std::pow


// ---------------------------------------------------------------------------
//  ACCURATE VECTOR SOFTPLUS for the above-threshold extraction (2026-10-02).
//
//  The extraction loop used to call the scalar AlgoSoftplus per pixel and per
//  channel - a double std::exp plus std::log1p, about 115 ms of a 1920 x 1080
//  frame on the review machine and the largest scalar loop left in the AVX2
//  path. This is the float vector form of the same function:
//
//      softplus(x, k) = x                       if x / k > ALGO_SOFTPLUS_LINEAR_LIMIT
//                     = k * log1p(exp(x / k))   otherwise
//
//  exp and log are the ACCURATE routines stage 13 already uses (polynomials of
//  about 1e-7 relative error), NOT FastCompute's Schraudolph exp (3 %). log1p
//  uses Kahan's compensated form, so the deep below-threshold tail - where
//  log(1 + u) in float would round u away - keeps full relative precision.
//  Duplicated from Algo_13_Sim.cpp deliberately: each AVX2 unit is
//  self-contained, as the stage 13 / 14 pair already is.
// ---------------------------------------------------------------------------
namespace
{
    FORCE_INLINE __m256 algoExp2AccV05 (__m256 x) noexcept
    {
        x = _mm256_max_ps(x, _mm256_set1_ps(-126.0f));
        x = _mm256_min_ps(x, _mm256_set1_ps( 127.0f));
        const __m256 n = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m256 f = _mm256_sub_ps(x, n);
        __m256 p = _mm256_set1_ps(1.3352819600e-3f);
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6178398092e-3f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5503406540e-2f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022650696e-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718056e-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
        const __m256i bias = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127));
        return _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32(bias, 23)));
    }

    FORCE_INLINE __m256 algoLogAccV05 (__m256 x) noexcept
    {
        x = _mm256_max_ps(x, _mm256_set1_ps(1.17549435e-38f));
        const __m256i xi = _mm256_castps_si256(x);
        __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
        __m256 m = _mm256_castsi256_ps(_mm256_or_si256(
            _mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000)));
        const __m256 hi = _mm256_cmp_ps(m, _mm256_set1_ps(1.41421356237309505f), _CMP_GE_OQ);
        m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), hi);
        e = _mm256_add_epi32(e, _mm256_and_si256(_mm256_castps_si256(hi), _mm256_set1_epi32(1)));
        const __m256 f = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
        __m256 p = _mm256_set1_ps(7.0376836292E-2f);
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(-1.1514610310E-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps( 1.1676998740E-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(-1.2420140846E-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps( 1.4249322787E-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(-1.6668057665E-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps( 2.0000714765E-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(-2.4999993993E-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps( 3.3333331174E-1f));
        const __m256 ff = _mm256_mul_ps(f, f);
        const __m256 r = _mm256_fmadd_ps(p, _mm256_mul_ps(ff, f),
                                         _mm256_fnmadd_ps(_mm256_set1_ps(0.5f), ff, f));
        return _mm256_fmadd_ps(_mm256_cvtepi32_ps(e), _mm256_set1_ps(0.693147180559945309f), r);
    }

    // k * log1p(exp(x / k)), with the linear asymptote above the limit.
    FORCE_INLINE __m256 algoSoftplusAccV05 (const __m256 x, const __m256 k,
                                            const __m256 invK) noexcept
    {
        const __m256 z   = _mm256_mul_ps(x, invK);
        const __m256 lin = _mm256_cmp_ps(z, _mm256_set1_ps(
                               static_cast<float>(ALGO_SOFTPLUS_LINEAR_LIMIT)), _CMP_GT_OQ);
        const __m256 u   = algoExp2AccV05(_mm256_mul_ps(z, _mm256_set1_ps(1.44269504088896341f)));
        // log1p(u) by Kahan's compensation: with w = fl(1 + u) and d = fl(w - 1),
        // log1p(u) = log(w) * u / d exactly up to the log's own error; where
        // w rounds to 1 (u below half an ulp of 1) log1p(u) = u to full precision.
        const __m256 w   = _mm256_add_ps(u, _mm256_set1_ps(1.0f));
        const __m256 d   = _mm256_sub_ps(w, _mm256_set1_ps(1.0f));
        const __m256 one = _mm256_cmp_ps(d, _mm256_setzero_ps(), _CMP_EQ_OQ);
        const __m256 l1p = _mm256_blendv_ps(
            _mm256_mul_ps(algoLogAccV05(w), _mm256_div_ps(u, _mm256_blendv_ps(d, _mm256_set1_ps(1.0f), one))),
            u, one);
        return _mm256_blendv_ps(_mm256_mul_ps(k, l1p), x, lin);
    }
}


// ---------------------------------------------------------------------------
//  Numerically safe softplus
// ---------------------------------------------------------------------------
AlgoType AlgoSoftplus (const AlgoType x, const AlgoType k) noexcept
{
    // Normalised argument. The caller guarantees k > 0, so no divide guard here:
    // a zero knee is a caller error, not a runtime condition to be absorbed.
    const AlgoType z = x / k;

    // Far up the ramp the function is indistinguishable from its asymptote, so
    // return the asymptote directly rather than evaluating an exponential that
    // is about to overflow. The crossover is chosen so the two agree well beyond
    // the last representable bit.
    if (z > ALGO_SOFTPLUS_LINEAR_LIMIT)
        return x;

    // log1p rather than log(1 + e): for large negative z the exponential is tiny
    // and adding one to it in floating point would discard every significant
    // digit it has. log1p keeps them, which matters because this is the region
    // that governs how gently the threshold engages.
    return k * static_cast<AlgoType>(std::log1p(std::exp(
               static_cast<HighPrecType>(z))));
}


// ---------------------------------------------------------------------------
//  Stage 5: halation
// ---------------------------------------------------------------------------
void AlgoStage05_Halation
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrLuma,
    AlgoType* RESTRICT       pScrAbove,
    AlgoType* RESTRICT       pScrBlur,
    AlgoType* RESTRICT       pScrBlurA,
    AlgoType* RESTRICT       pScrBlurB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           pxPerMm,
    const AlgoFreqState&     freq
) noexcept
{
    const film::HalationSpec& hal = profile.halation;

    // User scale, floored at zero. A negative scale would invert the effect into
    // a sharpening halo, which is not a physical state of any film.
    const AlgoType scale = MAX_VALUE(static_cast<AlgoType>(params.halationScale),
                                     ALGO_ZERO);

    // Per-channel strengths. Red is largest on essentially every colour stock,
    // because the red-sensitive layer sits deepest and so nearest the base.
    const AlgoType gainR = static_cast<AlgoType>(hal.gain_r);
    const AlgoType gainG = static_cast<AlgoType>(hal.gain_g);
    const AlgoType gainB = static_cast<AlgoType>(hal.gain_b);

    // Nothing to do when the stock has an effective antihalation backing, when
    // the user has turned the effect off, or when the render is so small that a
    // micrometre-scale radius cannot be represented. In every case the data must
    // still be COPIED: the retained-buffer policy gives this stage its own
    // destination, and leaving it unwritten would put stale contents in the
    // chain for every stage that follows.
    const bool anyGain = (gainR > ALGO_ZERO) || (gainG > ALGO_ZERO)
                                             || (gainB > ALGO_ZERO);

    if ((false == anyGain) || (scale <= ALGO_ZERO) || (pxPerMm <= ALGO_ZERO))
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    // Working planes the separable form needed; the frequency-domain filter
    // keeps its spectrum in the arena (AlgoFrequency.hpp).
    (void)pScrBlurA;
    (void)pScrBlurB;

    // ----------------------------------------------------------------------
    //  Threshold and knee, in linear exposure.
    //
    //  Mid grey is 1.0 in this domain by construction, so the threshold is a
    //  pure power of two above it and threshold_stops reads directly as stops
    //  over an 18 per cent card.
    // ----------------------------------------------------------------------
    const AlgoType thr = static_cast<AlgoType>(
        std::pow(2.0, static_cast<HighPrecType>(hal.threshold_stops)));

    // Knee width. Floored well above zero because the softplus divides by it,
    // and because a knee of exactly zero is a hard threshold that would produce
    // a visible contour around every highlight.
    const AlgoType knee = MAX_VALUE(thr * ALGO_HALATION_KNEE_FRACTION,
                                    static_cast<AlgoType>(1.0e-6));

    // ----------------------------------------------------------------------
    //  Broadcast luminance of the incoming exposure, built once and shared by
    //  all three channels.
    // ----------------------------------------------------------------------
    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        const AlgoType* RESTRICT pR = pSrcR + off;
        const AlgoType* RESTRICT pG = pSrcG + off;
        const AlgoType* RESTRICT pB = pSrcB + off;

        AlgoType* RESTRICT pL = pScrLuma + off;

        // Scene luminance, three FMAs per vector. One plane rather than three
        // because scatter inside the emulsion is broad and nearly achromatic.
        const __m256 wR = _mm256_set1_ps(ALGO_HALATION_LUMA_R);
        const __m256 wG = _mm256_set1_ps(ALGO_HALATION_LUMA_G);
        const __m256 wB = _mm256_set1_ps(ALGO_HALATION_LUMA_B);

        const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
        const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
        const __m256i mt = algoTailMaskLocal(nt);

        int32_t x = 0;

        for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
        {
            __m256 l = _mm256_mul_ps(_mm256_loadu_ps(pR + x), wR);
            l = _mm256_fmadd_ps(_mm256_loadu_ps(pG + x), wG, l);
            l = _mm256_fmadd_ps(_mm256_loadu_ps(pB + x), wB, l);
            _mm256_storeu_ps(pL + x, l);
        }

        if (nt > 0)
        {
            __m256 l = _mm256_mul_ps(_mm256_maskload_ps(pR + x, mt), wR);
            l = _mm256_fmadd_ps(_mm256_maskload_ps(pG + x, mt), wG, l);
            l = _mm256_fmadd_ps(_mm256_maskload_ps(pB + x, mt), wB, l);
            _mm256_maskstore_ps(pL + x, mt, l);
        }
    }

    // ----------------------------------------------------------------------
    //  Per-channel scatter.
    //
    //  Handled through a small table so the three passes are literally the same
    //  code path rather than three copies that can drift apart.
    // ----------------------------------------------------------------------
    const AlgoType* RESTRICT srcPlane[3] = { pSrcR, pSrcG, pSrcB };
    AlgoType* RESTRICT       dstPlane[3] = { pDstR, pDstG, pDstB };
    const AlgoType           chanGain[3] = { gainR, gainG, gainB };

    for (int32_t c = 0; c < 3; c++)
    {
        const AlgoType* RESTRICT pIn  = srcPlane[c];
        AlgoType* RESTRICT       pOut = dstPlane[c];

        // Total strength for this record. Zero means this layer has no path back
        // from the base worth modelling - common on stocks whose backing is
        // effective for the shorter wavelengths only.
        const AlgoType g = chanGain[c] * scale;

        // A record with no path back from the base is left as it is -- and, as
        // in film_sim, still floored at zero with the others (the floor after
        // stage 5 applies to all three records).
        if (g <= ALGO_ZERO)
        {
            const __m256 vZero = _mm256_setzero_ps();
            const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
            const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
            const __m256i mt = algoTailMaskLocal(nt);
            for (int32_t y = 0; y < sizeY; y++)
            {
                const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;
                const AlgoType* RESTRICT pE = pIn + off;
                AlgoType* RESTRICT       pO = pOut + off;
                int32_t x = 0;
                for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
                    _mm256_storeu_ps(pO + x, _mm256_max_ps(_mm256_loadu_ps(pE + x), vZero));
                if (nt > 0)
                    _mm256_maskstore_ps(pO + x, mt, _mm256_max_ps(_mm256_maskload_ps(pE + x, mt), vZero));
            }
            continue;
        }

        // ------------------------------------------------------------------
        //  Above-threshold scatter source.
        //
        //  Built into its own plane because the blur that follows needs the
        //  whole field before it can produce any output pixel.
        // ------------------------------------------------------------------
        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pE = pIn      + off;
            const AlgoType* RESTRICT pL = pScrLuma + off;

            AlgoType* RESTRICT pA = pScrAbove + off;

            // 2026-10-02: vectorised (algoSoftplusAccV05, see the note at the
            // top of this file). Half this layer's own exposure, half the scene
            // luminance; soft knee at the threshold. The masked tail keeps the
            // row padding untouched.
            const __m256 vOwn  = _mm256_set1_ps(ALGO_HALATION_OWN_FRACTION);
            const __m256 vLuma = _mm256_set1_ps(ALGO_HALATION_LUMA_FRACTION);
            const __m256 vThr  = _mm256_set1_ps(thr);
            const __m256 vK    = _mm256_set1_ps(knee);
            const __m256 vInvK = _mm256_set1_ps(ALGO_ONE / knee);
            const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
            const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
            const __m256i mt = algoTailMaskLocal(nt);
            int32_t x = 0;
            for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
            {
                const __m256 src = _mm256_fmadd_ps(vOwn, _mm256_loadu_ps(pE + x),
                                       _mm256_mul_ps(vLuma, _mm256_loadu_ps(pL + x)));
                _mm256_storeu_ps(pA + x,
                    algoSoftplusAccV05(_mm256_sub_ps(src, vThr), vK, vInvK));
            }
            if (nt > 0)
            {
                const __m256 src = _mm256_fmadd_ps(vOwn, _mm256_maskload_ps(pE + x, mt),
                                       _mm256_mul_ps(vLuma, _mm256_maskload_ps(pL + x, mt)));
                _mm256_maskstore_ps(pA + x, mt,
                    algoSoftplusAccV05(_mm256_sub_ps(src, vThr), vK, vInvK));
            }
        }

        // ------------------------------------------------------------------
        //  Spread the scattered light: film_sim's
        //  apply_transfer(above, grid.multi_gaussian(*hal.lobes(c))), the
        //  exact circular convolution, in the frequency domain (shared
        //  AlgoFrequency.cpp, four-lane FFT).
        // ------------------------------------------------------------------
        {
            AlgoFreqTransfer t;
            AlgoHalationTransfer(hal, c, t);
            AlgoFreqFilterPlane(freq, pScrAbove, pScrBlur, pitch, t);
        }

        // ------------------------------------------------------------------
        //  Deposit, conserving energy, and clamp at zero.
        //
        //  The clamp is here rather than deferred because the difference term is
        //  negative wherever a point loses more than it receives, and a negative
        //  exposure has no meaning for the logarithm taken at stage 8. It is a
        //  physical floor - no light at all - not a display clamp, so it does not
        //  violate the single-final-clamp rule.
        // ------------------------------------------------------------------
        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pE = pIn       + off;
            const AlgoType* RESTRICT pA = pScrAbove + off;
            const AlgoType* RESTRICT pS = pScrBlur  + off;

            AlgoType* RESTRICT pO = pOut + off;

            ALGO_VECTOR_HINT
            for (int32_t x = 0; x < sizeX; x++)
            {
                // What arrives from the surround, minus what left this point.
                const AlgoType net = pS[x] - pA[x];

                pO[x] = MAX_VALUE(pE[x] + g * net, ALGO_ZERO);
            }
        }
    }

    return;
}
