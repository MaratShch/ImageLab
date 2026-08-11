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
    const AlgoType           pxPerMm
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

    // ----------------------------------------------------------------------
    //  Scatter kernel, converted from physical units to pixels.
    //
    //  The radii are stored in micrometres ON THE FILM, so the same stock gives
    //  a halo of the same physical size whatever the render resolution, and a
    //  16 mm frame shows a proportionally larger halo than a 35 mm frame of the
    //  same scene. That is the correct behaviour and it is why the conversion
    //  goes through px_per_mm rather than through a pixel count.
    //
    //      sigma_px = (radius_um / 1000) * px_per_mm
    //
    //  A lobe that lands below a fifth of a pixel is dropped rather than
    //  submitted to the blur: its kernel would collapse to a single tap, which
    //  is an identity operation costing a full separable pass.
    // ----------------------------------------------------------------------
    AlgoType sigmaPx[ALGO_BLUR_MAX_LOBES] = { ALGO_ZERO, ALGO_ZERO,
                                              ALGO_ZERO, ALGO_ZERO };
    AlgoType weight [ALGO_BLUR_MAX_LOBES] = { ALGO_ZERO, ALGO_ZERO,
                                              ALGO_ZERO, ALGO_ZERO };

    int32_t lobes = 0;

    for (int32_t k = 0; k < ALGO_HALATION_LOBES; k++)
    {
        // Micrometres to millimetres to pixels, in one expression so there is no
        // intermediate to get the units wrong in.
        const AlgoType s = static_cast<AlgoType>(hal.radii_um[k])
                         * static_cast<AlgoType>(0.001) * pxPerMm;

        const AlgoType w = static_cast<AlgoType>(hal.weights[k]);

        // A quarter pixel: below this the discrete kernel has one significant
        // tap and the pass does nothing but cost time.
        if ((s >= static_cast<AlgoType>(0.25)) && (w > ALGO_ZERO))
        {
            sigmaPx[lobes] = s;
            weight [lobes] = w;
            lobes++;
        }
    }

    // Every lobe fell below the representable radius. The physical effect exists
    // but this render cannot show it, so pass the exposure through unchanged
    // rather than fabricating a one-pixel halo.
    if (0 == lobes)
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

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

        if (g <= ALGO_ZERO)
        {
            // Still a copy, for the reason given above.
            AlgoCopyPlane(pIn, pOut, sizeX, sizeY, pitch);
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

            for (int32_t x = 0; x < sizeX; x++)
            {
                // Half this layer's own exposure, half the scene luminance.
                const AlgoType src = ALGO_HALATION_OWN_FRACTION  * pE[x]
                                   + ALGO_HALATION_LUMA_FRACTION * pL[x];

                // Soft knee at the threshold. Below it this is very close to
                // zero, above it very close to (src - thr), and the transition
                // is smooth in every derivative - which is what keeps the halo
                // from acquiring a contour of its own.
                pA[x] = AlgoSoftplus(src - thr, knee);
            }
        }

        // ------------------------------------------------------------------
        //  Spread the scattered light.
        //
        //  Wrap boundary, matching the circular convolution of the frequency
        //  domain reference. For a kernel whose widest lobe is a fraction of a
        //  millimetre on the film, the wrap contribution is confined to a
        //  handful of edge pixels.
        // ------------------------------------------------------------------
        AlgoMultiGaussianBlurPlaneWrap(pScrAbove, pScrBlur,
                                       pScrBlurA, pScrBlurB,
                                       sizeX, sizeY, pitch,
                                       sigmaPx, weight, lobes);

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
