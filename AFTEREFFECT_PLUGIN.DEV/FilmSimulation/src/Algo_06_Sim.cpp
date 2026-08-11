// ---------------------------------------------------------------------------
//  Algo_06_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the emulsion MTF apply, the non-negative floor, and the five-tap
//  corner defocus - which is a genuine win, being real arithmetic over a fixed narrow
//  kernel with no wrap and unit stride in x.
//
//  Pipeline stage 6 and its sub-stage 6b, in exposure space:
//
//      AlgoStage06_EmulsionMtf     scatter inside the emulsion, per channel
//      AlgoStage06b_CornerDefocus  film buckling in the gate, radially blended
//
//  Both belong to the same numbered pipeline stage and share this translation
//  unit. Raw pointers, explicit geometry, no allocation, no mutable state, no
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

#include "AlgoEmulsionMtf.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>
#include "AlgoCornerDefocus.hpp"


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

#include <cmath>   // std::sqrt


// ---------------------------------------------------------------------------
//  Stage 6: emulsion MTF
// ---------------------------------------------------------------------------
void AlgoStage06_EmulsionMtf
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrBlurA,
    AlgoType* RESTRICT       pScrBlurB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoType           pxPerMm
) noexcept
{
    const film::MTFSpec& mtf = profile.mtf;

    // Per-channel half-modulation frequencies. Red is lowest on a colour stock
    // because its layer lies deepest under the gelatin.
    const AlgoType f50[3] = { static_cast<AlgoType>(mtf.f50_r),
                              static_cast<AlgoType>(mtf.f50_g),
                              static_cast<AlgoType>(mtf.f50_b) };

    // Development adjacency overshoot amplitude, and the diffusion length that
    // sets the frequency at which it peaks. A zero amplitude disables the
    // band-pass entirely and leaves a plain Gaussian.
    const AlgoType adjacency   = static_cast<AlgoType>(mtf.adjacency);
    const AlgoType adjacencyUm = static_cast<AlgoType>(mtf.adjacency_um);

    // Nothing can be expressed at a degenerate resolution. Copy rather than skip,
    // so the destination is never left holding stale contents.
    if (pxPerMm <= ALGO_ZERO)
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    // The two adjacency lobes are the same for all three records, so their
    // standard deviations in pixels are formed once. Micrometres to millimetres
    // to pixels, as everywhere else in the engine.
    const AlgoType adjInnerPx = adjacencyUm * ALGO_MTF_ADJACENCY_INNER
                              * static_cast<AlgoType>(0.001) * pxPerMm;

    const AlgoType adjOuterPx = adjacencyUm * ALGO_MTF_ADJACENCY_OUTER
                              * static_cast<AlgoType>(0.001) * pxPerMm;

    // The adjacency term is only meaningful when both its amplitude and its
    // scale are set, and when that scale is large enough to be represented.
    const bool wantAdjacency = (adjacency > ALGO_ZERO)
                            && (adjOuterPx >= ALGO_MTF_MIN_SIGMA_PX);

    const AlgoType* RESTRICT srcPlane[3] = { pSrcR, pSrcG, pSrcB };
    AlgoType* RESTRICT       dstPlane[3] = { pDstR, pDstG, pDstB };

    for (int32_t c = 0; c < 3; c++)
    {
        const AlgoType* RESTRICT pIn  = srcPlane[c];
        AlgoType* RESTRICT       pOut = dstPlane[c];

        // ------------------------------------------------------------------
        //  Base Gaussian for this record.
        //
        //  sigma_mm = K / f50 with K = sqrt(ln2 / 2) / pi, then millimetres to
        //  pixels. An f50 of zero or below means the stock has no MTF figure for
        //  this layer, which is treated as perfectly sharp rather than as
        //  infinitely soft.
        // ------------------------------------------------------------------
        const AlgoType basePx = (f50[c] > ALGO_ZERO)
                              ? (ALGO_MTF_SIGMA_MM_PER_INV_F50 / f50[c]) * pxPerMm
                              : ALGO_ZERO;

        // Nothing to filter: the emulsion is sharper than this render grid and
        // there is no adjacency lift to apply either.
        if ((basePx < ALGO_MTF_MIN_SIGMA_PX) && (false == wantAdjacency))
        {
            AlgoCopyPlane(pIn, pOut, sizeX, sizeY, pitch);
            continue;
        }

        // ------------------------------------------------------------------
        //  Assemble the lobe set.
        //
        //  Lobe 0 is the base transfer at unit weight. Lobes 1 and 2 are the
        //  adjacency band-pass, and because a product of Gaussian transfers is a
        //  Gaussian whose VARIANCES add, each of them is the base convolved with
        //  its own adjacency lobe: sigma = sqrt(base^2 + lobe^2).
        //
        //  The weights are 1, +a and -a. They sum to one, so the whole filter
        //  passes DC untouched and cannot shift the exposure level - which is
        //  what makes this an edge effect rather than a brightness change.
        // ------------------------------------------------------------------
        AlgoType sigmaPx[ALGO_BLUR_MAX_LOBES] = { ALGO_ZERO, ALGO_ZERO,
                                                  ALGO_ZERO, ALGO_ZERO };
        AlgoType weight [ALGO_BLUR_MAX_LOBES] = { ALGO_ZERO, ALGO_ZERO,
                                                  ALGO_ZERO, ALGO_ZERO };

        const HighPrecType base2 = static_cast<HighPrecType>(basePx)
                                 * static_cast<HighPrecType>(basePx);

        sigmaPx[0] = basePx;
        weight [0] = ALGO_ONE;

        int32_t lobes = 1;

        if (wantAdjacency)
        {
            const HighPrecType in2 = static_cast<HighPrecType>(adjInnerPx)
                                   * static_cast<HighPrecType>(adjInnerPx);

            const HighPrecType out2 = static_cast<HighPrecType>(adjOuterPx)
                                    * static_cast<HighPrecType>(adjOuterPx);

            sigmaPx[1] = static_cast<AlgoType>(std::sqrt(base2 + in2));
            weight [1] = adjacency;

            sigmaPx[2] = static_cast<AlgoType>(std::sqrt(base2 + out2));
            weight [2] = -adjacency;

            lobes = 3;
        }

        // ------------------------------------------------------------------
        //  Filter. Wrap boundary, matching the circular convolution of the
        //  frequency-domain reference.
        // ------------------------------------------------------------------
        AlgoMultiGaussianBlurPlaneWrap(pIn, pOut,
                                       pScrBlurA, pScrBlurB,
                                       sizeX, sizeY, pitch,
                                       sigmaPx, weight, lobes);

        // ------------------------------------------------------------------
        //  Floor at zero.
        //
        //  The negative adjacency lobe can drive a value below zero next to a
        //  hard edge - that is the undershoot side of the overshoot, and it is
        //  real - but a negative exposure has no meaning for the logarithm taken
        //  at stage 8. This is a physical floor, no light at all, not a display
        //  clamp, so it does not violate the single-final-clamp rule.
        // ------------------------------------------------------------------
        if (wantAdjacency)
        {
            for (int32_t y = 0; y < sizeY; y++)
            {
                AlgoType* RESTRICT pRow =
                    pOut + static_cast<std::ptrdiff_t>(y) * pitch;

                ALGO_VECTOR_HINT
                for (int32_t x = 0; x < sizeX; x++)
                    pRow[x] = MAX_VALUE(pRow[x], ALGO_ZERO);   /* see vector floor below */
            }
        }
    }

    return;
}


namespace
{
    // ----------------------------------------------------------------------
    //  Fixed five-tap binomial blur of one plane, with EDGE CLAMP boundaries.
    //
    //  Separable: a horizontal sweep into the first scratch plane, then a
    //  vertical sweep from there into the second.
    //
    //  Edge clamp rather than wrap, because the effect being modelled is
    //  specifically a difference between the middle of the frame and its
    //  corners, and wrapping would fold one into the other.
    // ----------------------------------------------------------------------
    void defocusBlurPlane
    (
        const AlgoType* RESTRICT pSrc,
        AlgoType* RESTRICT       pTmp,
        AlgoType* RESTRICT       pDst,
        const int32_t            sizeX,
        const int32_t            sizeY,
        const int32_t            pitch
    ) noexcept
    {
        // Kernel taps, mirrored about the centre.
        const AlgoType k[ALGO_DEFOCUS_TAPS] =
        {
            ALGO_DEFOCUS_TAP_0, ALGO_DEFOCUS_TAP_1, ALGO_DEFOCUS_TAP_2,
            ALGO_DEFOCUS_TAP_1, ALGO_DEFOCUS_TAP_0
        };

        // ------------------------------------------------------------------
        //  Horizontal sweep.
        // ------------------------------------------------------------------
        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pIn  = pSrc + off;
            AlgoType* RESTRICT       pOut = pTmp + off;

            for (int32_t x = 0; x < sizeX; x++)
            {
                AlgoType acc = ALGO_ZERO;

                for (int32_t t = 0; t < ALGO_DEFOCUS_TAPS; t++)
                {
                    // Sample position for this tap, clamped into the row. The
                    // clamp is what repeats the edge pixel outward.
                    int32_t sx = x + t - ALGO_DEFOCUS_RADIUS;

                    sx = MAX_VALUE(sx, 0);
                    sx = MIN_VALUE(sx, sizeX - 1);

                    acc += k[t] * pIn[sx];
                }

                pOut[x] = acc;
            }
        }

        // ------------------------------------------------------------------
        //  Vertical sweep.
        // ------------------------------------------------------------------
        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pOut =
                pDst + static_cast<std::ptrdiff_t>(y) * pitch;

            // Row pointers for all five taps, resolved once per output row so
            // the inner loop over x is a straight strided read.
            const AlgoType* RESTRICT pRow[ALGO_DEFOCUS_TAPS];

            for (int32_t t = 0; t < ALGO_DEFOCUS_TAPS; t++)
            {
                int32_t sy = y + t - ALGO_DEFOCUS_RADIUS;

                sy = MAX_VALUE(sy, 0);
                sy = MIN_VALUE(sy, sizeY - 1);

                pRow[t] = pTmp + static_cast<std::ptrdiff_t>(sy) * pitch;
            }

            // Five-tap vertical kernel. The row bases are resolved outside this
            // loop and x is unit-stride, so the inner work is five aligned loads and
            // four FMAs per vector - the same shape as the blur's vertical pass, and
            // the reason this stage vectorises well despite being small.
            const __m256 k0 = _mm256_set1_ps(k[0]);
            const __m256 k1 = _mm256_set1_ps(k[1]);
            const __m256 k2 = _mm256_set1_ps(k[2]);
            const __m256 k3 = _mm256_set1_ps(k[3]);
            const __m256 k4 = _mm256_set1_ps(k[4]);

            const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
            const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
            const __m256i mt = algoTailMaskLocal(nt);

            int32_t x = 0;

            for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
            {
                __m256 a = _mm256_mul_ps(_mm256_loadu_ps(pRow[0] + x), k0);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(pRow[1] + x), k1, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(pRow[2] + x), k2, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(pRow[3] + x), k3, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(pRow[4] + x), k4, a);
                _mm256_storeu_ps(pOut + x, a);
            }

            if (nt > 0)
            {
                __m256 a = _mm256_mul_ps(_mm256_maskload_ps(pRow[0] + x, mt), k0);
                a = _mm256_fmadd_ps(_mm256_maskload_ps(pRow[1] + x, mt), k1, a);
                a = _mm256_fmadd_ps(_mm256_maskload_ps(pRow[2] + x, mt), k2, a);
                a = _mm256_fmadd_ps(_mm256_maskload_ps(pRow[3] + x, mt), k3, a);
                a = _mm256_fmadd_ps(_mm256_maskload_ps(pRow[4] + x, mt), k4, a);
                _mm256_maskstore_ps(pOut + x, mt, a);
            }
        }

        return;
    }
}


// ---------------------------------------------------------------------------
//  Sub-stage 6b: corner defocus
// ---------------------------------------------------------------------------
void AlgoStage06b_CornerDefocus
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrH,
    AlgoType* RESTRICT       pScrV,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params
) noexcept
{
    const film::CoatingSpec& coat = profile.coating;

    // The same user scale that drives the coating field also drives the buckle,
    // because both are properties of how the physical film behaves rather than
    // of the image on it. Floored at zero: a negative scale would sharpen the
    // corners, which no gate has ever done.
    const AlgoType scale = MAX_VALUE(static_cast<AlgoType>(params.coatingScale),
                                     ALGO_ZERO);

    // Corner blend weight, capped so the corner always retains some of the
    // original image.
    const AlgoType loss = MIN_VALUE(static_cast<AlgoType>(coat.buckle_mtf_loss)
                                    * scale,
                                    ALGO_DEFOCUS_MAX_LOSS);

    // A stock with no buckle figure, or the effect turned off. Copy, so the
    // destination is never left holding stale contents.
    if (loss <= ALGO_ZERO)
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    // ----------------------------------------------------------------------
    //  Frame geometry for the radial blend.
    //
    //  Normalised so that the centre is zero and each CORNER is exactly one.
    //  The two half extents are floored at one to keep a single-row or
    //  single-column render from dividing by zero.
    // ----------------------------------------------------------------------
    const HighPrecType cy = static_cast<HighPrecType>(sizeY - 1) * 0.5;
    const HighPrecType cx = static_cast<HighPrecType>(sizeX - 1) * 0.5;

    const HighPrecType invHalfY = 1.0 / MAX_VALUE(cy, 1.0);
    const HighPrecType invHalfX = 1.0 / MAX_VALUE(cx, 1.0);

    const AlgoType* RESTRICT srcPlane[3] = { pSrcR, pSrcG, pSrcB };
    AlgoType* RESTRICT       dstPlane[3] = { pDstR, pDstG, pDstB };

    for (int32_t c = 0; c < 3; c++)
    {
        const AlgoType* RESTRICT pIn  = srcPlane[c];
        AlgoType* RESTRICT       pOut = dstPlane[c];

        // Fully blurred version of this record, built once.
        defocusBlurPlane(pIn, pScrH, pScrV, sizeX, sizeY, pitch);

        // RULE D1 ALIGNMENT, 2026-08-11: this loop was HighPrecType per pixel.
        // Same argument as the vignette in stage 4 - it is a normalised radius
        // used as a blend weight, float32 gives ~1e-07 relative on a quantity
        // that ends up at 16-bit, and being double prevented the pixel loop from
        // vectorising at all.
        const AlgoType cxF       = static_cast<AlgoType>(cx);
        const AlgoType cyF       = static_cast<AlgoType>(cy);
        const AlgoType invHalfXF = static_cast<AlgoType>(invHalfX);
        const AlgoType invHalfYF = static_cast<AlgoType>(invHalfY);

        for (int32_t y = 0; y < sizeY; y++)
        {
            // Row-constant normalised vertical offset and its square.
            const AlgoType yn  = (static_cast<AlgoType>(y) - cyF) * invHalfYF;
            const AlgoType yn2 = yn * yn;

            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pSharp = pIn   + off;
            const AlgoType* RESTRICT pSoft  = pScrV + off;

            AlgoType* RESTRICT pO = pOut + off;

            for (int32_t x = 0; x < sizeX; x++)
            {
                const AlgoType xn = (static_cast<AlgoType>(x) - cxF)
                                  * invHalfXF;

                // Radius squared, halved so a corner - where both normalised
                // coordinates are one - lands at exactly one. Squared rather
                // than linear because the defocus grows with the square of the
                // displacement from the held centre, and because it removes a
                // square root from every pixel.
                const AlgoType r2 = (yn2 + xn * xn) * ALGO_HALF;

                // Blend weight: none at the centre, the full capped loss at the
                // corners.
                const AlgoType w = loss * r2;

                // Linear cross-fade. The two weights sum to one at every pixel,
                // so a flat field passes through unchanged and the stage cannot
                // darken or lighten the corners.
                pO[x] = pSharp[x] * (ALGO_ONE - w) + pSoft[x] * w;
            }
        }
    }

    return;
}
