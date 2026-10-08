// ---------------------------------------------------------------------------
//  Algo_06_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the non-negative floors and the five-tap corner defocus; the
//  emulsion MTF itself is the shared frequency-domain filter (AlgoFrequency.cpp,
//  four-lane FFT) -- the corner defocus - which is a genuine win, being real arithmetic over a fixed narrow
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

// Common.hpp -- AVX2_ALIGN / CACHE_ALIGN are defined here. Included
// DIRECTLY rather than relied on transitively: this file declares an
// aligned buffer, so the macro must not depend on another header's
// include order to be in scope.
#include "Common.hpp"
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
    const AlgoType           pxPerMm,
    const AlgoFreqState&     freq
) noexcept
{
    (void)pScrBlurA;
    (void)pScrBlurB;

    const film::MTFSpec& mtf = profile.mtf;

    // Nothing can be expressed at a degenerate resolution. Copy rather than skip,
    // so the destination is never left holding stale contents.
    if (pxPerMm <= ALGO_ZERO)
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    // Per-channel half-modulation frequencies (film_sim: profile.mtf.f50s()).
    const HighPrecType f50[3] = { static_cast<HighPrecType>(mtf.f50_r),
                                  static_cast<HighPrecType>(mtf.f50_g),
                                  static_cast<HighPrecType>(mtf.f50_b) };

    const AlgoType* RESTRICT srcPlane[3] = { pSrcR, pSrcG, pSrcB };
    AlgoType* RESTRICT       dstPlane[3] = { pDstR, pDstG, pDstB };

    const __m256  vZero = _mm256_setzero_ps();
    const int32_t nv    = sizeX / ALGO_AVX2_LANES_LOCAL;
    const int32_t nt    = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
    const __m256i mt    = algoTailMaskLocal(nt);

    for (int32_t c = 0; c < 3; c++)
    {
        const AlgoType* RESTRICT pIn  = srcPlane[c];
        AlgoType* RESTRICT       pOut = dstPlane[c];

        // Floor the incoming exposure (film_sim floors after stage 5 on every
        // frame; idempotent when stage 5 ran). Same flow as the scalar engine.
        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;
            const AlgoType* RESTRICT pI = pIn + off;
            AlgoType* RESTRICT       pO = pOut + off;
            int32_t x = 0;
            for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
                _mm256_storeu_ps(pO + x, _mm256_max_ps(_mm256_loadu_ps(pI + x), vZero));
            if (nt > 0)
                _mm256_maskstore_ps(pO + x, mt, _mm256_max_ps(_mm256_maskload_ps(pI + x, mt), vZero));
        }

        // film_sim grid.mtf(f50s[c], adjacency, adjacency_um, spec, c): the
        // measured law (or the legacy Gaussian) times the adjacency band-pass,
        // EXACT on film_sim's frequency grid -- see the scalar twin.
        AlgoFreqTransfer t;
        AlgoFreqSetMtf(t, f50[c], mtf.mtf_measured,
                       static_cast<HighPrecType>(mtf.mtf_rolloff_q),
                       static_cast<HighPrecType>(mtf.adjacency),
                       static_cast<HighPrecType>(mtf.adjacency_um));

        if (AlgoFreqIsIdentity(t))
            continue;

        AlgoFreqFilterPlane(freq, pOut, pOut, pitch, t);

        // Floor at zero (film_sim: np.maximum after stage 6).
        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow = pOut + static_cast<std::ptrdiff_t>(y) * pitch;
            int32_t x = 0;
            for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
                _mm256_storeu_ps(pRow + x, _mm256_max_ps(_mm256_loadu_ps(pRow + x), vZero));
            if (nt > 0)
                _mm256_maskstore_ps(pRow + x, mt, _mm256_max_ps(_mm256_maskload_ps(pRow + x, mt), vZero));
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
    // One clamped five-tap sample, for the row ends. Evaluated exactly as the
    // interior vector loop does: each product rounded, then summed in tap order.
    FORCE_INLINE AlgoType defocusTapClamped (const AlgoType* RESTRICT pIn,
                                             const int32_t            x,
                                             const int32_t            sizeX,
                                             const AlgoType* RESTRICT k) noexcept
    {
        AlgoType acc = ALGO_ZERO;

        for (int32_t t = 0; t < ALGO_DEFOCUS_TAPS; t++)
        {
            int32_t sx = x + t - ALGO_DEFOCUS_RADIUS;

            sx = MAX_VALUE(sx, 0);
            sx = MIN_VALUE(sx, sizeX - 1);

            const AlgoType prod = k[t] * pIn[sx];   // rounded product, then the add
            acc = acc + prod;
        }

        return acc;
    }

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

            // 2026-10-04: the interior - every x whose five taps lie inside the
            // row - is a straight vector loop of five unaligned loads, five
            // multiplies and four adds; only the two pixels at each end need
            // the clamp. Tap order and rounding are unchanged, so the result
            // is identical to the clamped scalar loop it replaces.
            const int32_t xLo = ALGO_DEFOCUS_RADIUS;
            const int32_t xHi = sizeX - ALGO_DEFOCUS_RADIUS;   // exclusive

            int32_t x = 0;

            for (; x < MIN_VALUE(xLo, sizeX); x++)
                pOut[x] = defocusTapClamped(pIn, x, sizeX, k);

            if (xHi > xLo)
            {
                const __m256 k0 = _mm256_set1_ps(k[0]);
                const __m256 k1 = _mm256_set1_ps(k[1]);
                const __m256 k2 = _mm256_set1_ps(k[2]);
                const __m256 k3 = _mm256_set1_ps(k[3]);
                const __m256 k4 = _mm256_set1_ps(k[4]);

                for (; x + ALGO_AVX2_LANES_LOCAL <= xHi; x += ALGO_AVX2_LANES_LOCAL)
                {
                    // Separate multiply and add, NOT fused: the clamped scalar
                    // loop this mirrors (and the Scalar engine) round each
                    // product before the sum, and the vector interior must
                    // produce the same bits as the two scalar ends of the row.
                    const AlgoType* RESTRICT p = pIn + x - ALGO_DEFOCUS_RADIUS;
                    __m256 a = _mm256_mul_ps(_mm256_loadu_ps(p), k0);
                    a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(p + 1), k1));
                    a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(p + 2), k2));
                    a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(p + 3), k3));
                    a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(p + 4), k4));
                    _mm256_storeu_ps(pOut + x, a);
                }
            }

            for (; x < sizeX; x++)
                pOut[x] = defocusTapClamped(pIn, x, sizeX, k);
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

            // 2026-10-04: explicit vector form of the cross-fade. Per pixel:
            // xn = (x - cx) * invHalfX, r2 = (yn2 + xn * xn) * 0.5, w = loss * r2,
            // out = sharp * (1 - w) + soft * w, every product rounded and no
            // fused step, exactly as the scalar expression reads and as the
            // Scalar engine evaluates it. (The previous comment claimed the
            // scalar loop auto-vectorised; the object code showed it did not.)
            // ⚠ A compiler that contracts intrinsics (GCC at its default
            // -ffp-contract=fast) may still fuse some of these; MSVC does not.
            // Bit-identity with the scalar form holds under no-contraction
            // semantics, which is the owner's toolchain.
            const __m256 vCx   = _mm256_set1_ps(cxF);
            const __m256 vInvX = _mm256_set1_ps(invHalfXF);
            const __m256 vYn2  = _mm256_set1_ps(yn2);
            const __m256 vHalf = _mm256_set1_ps(ALGO_HALF);
            const __m256 vLoss = _mm256_set1_ps(loss);
            const __m256 vOne  = _mm256_set1_ps(ALGO_ONE);
            const __m256 vLane = _mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);

            const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
            const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
            const __m256i mt = algoTailMaskLocal(nt);

            int32_t x = 0;

            for (int32_t v = 0; v <= nv; v++, x += ALGO_AVX2_LANES_LOCAL)
            {
                const bool tail = (v == nv);
                if (tail && (0 == nt))
                    break;

                const __m256 xf = _mm256_add_ps(_mm256_set1_ps(static_cast<AlgoType>(x)), vLane);
                const __m256 xn = _mm256_mul_ps(_mm256_sub_ps(xf, vCx), vInvX);
                const __m256 r2 = _mm256_mul_ps(_mm256_add_ps(vYn2, _mm256_mul_ps(xn, xn)), vHalf);
                const __m256 w  = _mm256_mul_ps(vLoss, r2);
                const __m256 sharp = tail ? _mm256_maskload_ps(pSharp + x, mt) : _mm256_loadu_ps(pSharp + x);
                const __m256 soft  = tail ? _mm256_maskload_ps(pSoft  + x, mt) : _mm256_loadu_ps(pSoft  + x);
                const __m256 out = _mm256_add_ps(_mm256_mul_ps(sharp, _mm256_sub_ps(vOne, w)),
                                                 _mm256_mul_ps(soft, w));
                if (tail) _mm256_maskstore_ps(pO + x, mt, out);
                else      _mm256_storeu_ps(pO + x, out);
            }
        }
    }

    return;
}
