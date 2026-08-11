#if 0
// ---------------------------------------------------------------------------
//  Algo_10_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the pointwise passes. The scan blur is the shared primitive, already
//  vectorised, and the sub-pixel misregistration shift is a bilinear resample whose
//  source index advances non-integrally - a gather, left scalar for the same reason
//  AlgoBilinearUpsample is.
//
//  ALIGNMENT: EVERY IMAGE ACCESS IS UNALIGNED, DELIBERATELY.
//
//  loadu/storeu on all plane data. The arena base comes from the host's memory pool,
//  whose alignment argument is a HINT, not a guarantee - it was observed returning a
//  base 16 mod 32, which faults an aligned 256-bit load. AlgoMemHandler.cpp is SHARED
//  by both flavours and must not be changed to suit the vector path, so the vector
//  path carries no alignment assumption instead. Costs nothing measurable on Haswell
//  and later.
//
//  Pipeline stage 10 and its sub-stage 10b, in the density domain:
//
//      AlgoScanSigmaMm         f50 in cycles/mm to Gaussian sigma in millimetres
//      AlgoStage10_ScanMtf     scanner optics plus per-channel registration error
//      AlgoStage10b_EdgeFog    additive fog near the physical film edges
//
//  Both belong to the same numbered pipeline stage and share this translation
//  unit. Raw pointers, explicit geometry, no allocation, no mutable state, no
//  validation of inputs.
// ---------------------------------------------------------------------------

#include "AlgoScanMtf.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>
#include "AlgoEdgeFog.hpp"


static_assert(sizeof(AlgoType) == 4,
              "the AVX2 path requires AlgoType to be a 32-bit float");

namespace
{
    // ----------------------------------------------------------------------
    //  Lanes per vector, and the tail mask for the final partial vector of a row.
    //
    //  The active width is not generally a multiple of eight. Masked access leaves the
    //  row padding untouched, which keeps the NaN-poison arena test meaningful.
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

#include <cmath>   // std::exp, std::floor


// ---------------------------------------------------------------------------
//  Gaussian sigma in millimetres for a given 50 per cent modulation frequency
// ---------------------------------------------------------------------------
AlgoType AlgoScanSigmaMm (const AlgoType f50CyclesPerMm) noexcept
{
    // A missing or nonsensical figure means the optics are not characterised. That
    // is treated as perfectly sharp - a sigma of zero - rather than as infinitely
    // soft, because a stock with no measurement should render no worse than one
    // with a good one.
    if (f50CyclesPerMm <= ALGO_ZERO)
        return ALGO_ZERO;

    // sqrt(ln(2)/2) / pi = 0.18738564618678. Derived by equating the two exponent
    // forms: the MTF is exp(-ln2 (f/f50)^2) and a Gaussian blur of sigma s
    // millimetres has transfer exp(-2 pi^2 s^2 f^2), so ln2/f50^2 = 2 pi^2 s^2.
    //
    // Written as a literal for the same reason it is in the emulsion MTF header:
    // it is then a compile-time constant on every compiler, and the derivation can
    // be checked against the digits by hand.
    return static_cast<AlgoType>(0.18738564618678) / f50CyclesPerMm;
}


namespace
{
    // ----------------------------------------------------------------------
    //  Translate one plane by a sub-pixel offset, wrapping at the boundaries.
    //
    //  Bilinear, with the integer part of the shift folded into the source index
    //  and only the fractional part carried in the weights.
    //
    //  Wrap rather than clamp, so the stage matches the circular convolution the
    //  rest of the density-domain chain uses, and so a shift cannot manufacture a
    //  band of repeated edge pixels.
    //
    //  dy, dx are the displacement of the IMAGE, so the sample is taken from
    //  (y - dy, x - dx).
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    //  CATMULL-ROM SUB-PIXEL SHIFT, replacing bilinear -- 2026-08-11.
    //
    //  Same argument as the gate-weave sampler in stage 15, and measured in the
    //  same experiment: a bilinear sub-pixel shift is a low-pass filter, and
    //  registration error between the colour records is a TRANSLATION, not a
    //  blur. On Lady.png through EASTMAN_EKTACHROME_5239 this shift alone was
    //  costing 48 per cent of the green record's high-frequency energy
    //  (Laplacian variance 272.7 with it, 520.0 without).
    //
    //  It also double-counted the scan MTF, which is applied in this very stage
    //  a few lines below - the optical softening of the scanner lens is modelled
    //  there explicitly, so charging for it again in the resampler is wrong twice
    //  over.
    //
    //  Catmull-Rom is interpolating, so an integer shift is still the exact
    //  identity, and its half-pixel transfer at Nyquist is ~0.87 against
    //  bilinear's 0.5. Four taps per axis instead of two.
    //
    //  WRAPPED, not clamped, unlike the weave: registration error is a property
    //  of the scanner's optical path across a frame that continues past the
    //  aperture, and the surrounding code documents the circular convention that
    //  the blur passes in this file also use. Keeping the two consistent is what
    //  lets a flat field stay flat through both.
    // ----------------------------------------------------------------------
    //  Catmull-Rom basis expanded into four tap weights at a fixed fraction.
    //
    //  Computed once per call rather than per pixel: the sub-pixel displacement
    //  is the same for the whole frame. The four weights sum to exactly one at
    //  every t, which is what keeps a flat field flat.
    inline void catmullWeights (const AlgoType t, AlgoType c[4]) noexcept
    {
        const AlgoType t2 = t * t;
        const AlgoType t3 = t2 * t;

        c[0] = static_cast<AlgoType>(-0.5) * t3
             +                        t2
             + static_cast<AlgoType>(-0.5) * t;

        c[1] = static_cast<AlgoType>( 1.5) * t3
             + static_cast<AlgoType>(-2.5) * t2
             + ALGO_ONE;

        c[2] = static_cast<AlgoType>(-1.5) * t3
             + static_cast<AlgoType>( 2.0) * t2
             + static_cast<AlgoType>( 0.5) * t;

        c[3] = static_cast<AlgoType>( 0.5) * t3
             + static_cast<AlgoType>(-0.5) * t2;

        return;
    }


    void shiftPlaneWrap
    (
        const AlgoType* RESTRICT pSrc,
        AlgoType* RESTRICT       pDst,
        const int32_t            sizeX,
        const int32_t            sizeY,
        const int32_t            pitch,
        const HighPrecType       dy,
        const HighPrecType       dx
    ) noexcept
    {
        // ------------------------------------------------------------------
        //  THE SHIFT IS A FRAME CONSTANT, SO THE SIXTEEN WEIGHTS ARE TOO.
        //
        //  The first Catmull-Rom version of this function evaluated the cubic
        //  basis per pixel and wrapped every one of its four column indices with
        //  a modulo. Measured, that took stage 10 from 26.3 ms to 88.5 ms and the
        //  whole HD frame from 297.7 ms to 422.4 ms - a 125 ms regression that
        //  undid half of a day's optimisation to buy the sharpness back.
        //
        //  Every record is displaced by the SAME sub-pixel amount, so the four
        //  horizontal and four vertical basis weights are computed ONCE here.
        //  What remains is a fixed 4x4 separable convolution at an integer
        //  offset, and the interior of the plane - everything but three columns
        //  and three rows - needs no wrapping at all, so it is contiguous and
        //  vectorises.
        // ------------------------------------------------------------------
        const HighPrecType fy = std::floor(-dy);
        const HighPrecType fx = std::floor(-dx);

        const int32_t iy = static_cast<int32_t>(fy);
        const int32_t ix = static_cast<int32_t>(fx);

        const AlgoType wy = static_cast<AlgoType>(-dy - fy);
        const AlgoType wx = static_cast<AlgoType>(-dx - fx);

        // Catmull-Rom basis at a fixed fraction, expanded to four tap weights.
        // Sum of the four is exactly one, so a flat field stays flat.
        AVX2_ALIGN AlgoType cx[4];
        AVX2_ALIGN AlgoType cy[4];

        catmullWeights(wx, cx);
        catmullWeights(wy, cy);

        const __m256 vcx0 = _mm256_set1_ps(cx[0]);
        const __m256 vcx1 = _mm256_set1_ps(cx[1]);
        const __m256 vcx2 = _mm256_set1_ps(cx[2]);
        const __m256 vcx3 = _mm256_set1_ps(cx[3]);

        const __m256 vZero = _mm256_setzero_ps();

        // Interior in x: every one of the four taps lands inside [0, sizeX).
        const int32_t xLo = MIN_VALUE(MAX_VALUE(1 - ix, 0), sizeX);
        const int32_t xHi = CLAMP_VALUE(sizeX - 2 - ix, xLo, sizeX);

        const int32_t inner  = xHi - xLo;
        const int32_t vecs   = inner / 8;
        const int32_t tailN  = inner - vecs * 8;

        const __m256i vTail = algoTailMaskLocal(tailN);

        for (int32_t y = 0; y < sizeY; y++)
        {
            // The four source rows, wrapped once per output row rather than per
            // pixel. Wrapped rather than clamped: registration error is a
            // property of the scanner's optical path, and the blur passes in this
            // file use the same circular convention.
            const AlgoType* RESTRICT rows[4];

            for (int32_t k = 0; k < 4; k++)
            {
                const int32_t sy = ((y + iy + k - 1) % sizeY + sizeY) % sizeY;
                rows[k] = pSrc + static_cast<std::ptrdiff_t>(sy) * pitch;
            }

            AlgoType* RESTRICT pOut =
                pDst + static_cast<std::ptrdiff_t>(y) * pitch;

            // --- left edge, wrapped, scalar ---
            for (int32_t x = 0; x < xLo; x++)
            {
                AlgoType acc = ALGO_ZERO;

                for (int32_t ky = 0; ky < 4; ky++)
                {
                    AlgoType h = ALGO_ZERO;

                    for (int32_t kx = 0; kx < 4; kx++)
                    {
                        const int32_t sx =
                            ((x + ix + kx - 1) % sizeX + sizeX) % sizeX;

                        h += cx[kx] * rows[ky][sx];
                    }

                    acc += cy[ky] * h;
                }

                pOut[x] = MAX_VALUE(acc, ALGO_ZERO);
            }

            // --- interior, no wrap, vectorised ---
            //
            //  The four horizontal taps are CONTIGUOUS, so they are four
            //  overlapping unaligned loads rather than a gather. Each row is
            //  filtered horizontally into a register, then the four rows are
            //  combined vertically - the separable form, done without ever
            //  writing an intermediate plane.
            int32_t x = xLo;

            for (int32_t v = 0; v < vecs; v++, x += 8)
            {
                __m256 acc = vZero;

                for (int32_t ky = 0; ky < 4; ky++)
                {
                    const AlgoType* RESTRICT w = rows[ky] + x + ix - 1;

                    __m256 h = _mm256_mul_ps(vcx0, _mm256_loadu_ps(w));
                    h = _mm256_fmadd_ps(vcx1, _mm256_loadu_ps(w + 1), h);
                    h = _mm256_fmadd_ps(vcx2, _mm256_loadu_ps(w + 2), h);
                    h = _mm256_fmadd_ps(vcx3, _mm256_loadu_ps(w + 3), h);

                    acc = _mm256_fmadd_ps(_mm256_set1_ps(cy[ky]), h, acc);
                }

                // Zero floor: the cubic has negative lobes and these are
                // exposures. Positive overshoot is edge acutance and is kept.
                _mm256_storeu_ps(pOut + x, _mm256_max_ps(acc, vZero));
            }

            if (tailN > 0)
            {
                __m256 acc = vZero;

                for (int32_t ky = 0; ky < 4; ky++)
                {
                    const AlgoType* RESTRICT w = rows[ky] + x + ix - 1;

                    __m256 h = _mm256_mul_ps(vcx0,
                                   _mm256_maskload_ps(w, vTail));
                    h = _mm256_fmadd_ps(vcx1,
                            _mm256_maskload_ps(w + 1, vTail), h);
                    h = _mm256_fmadd_ps(vcx2,
                            _mm256_maskload_ps(w + 2, vTail), h);
                    h = _mm256_fmadd_ps(vcx3,
                            _mm256_maskload_ps(w + 3, vTail), h);

                    acc = _mm256_fmadd_ps(_mm256_set1_ps(cy[ky]), h, acc);
                }

                _mm256_maskstore_ps(pOut + x, vTail,
                                    _mm256_max_ps(acc, vZero));

                x += tailN;
            }

            // --- right edge, wrapped, scalar ---
            for (; x < sizeX; x++)
            {
                AlgoType acc = ALGO_ZERO;

                for (int32_t ky = 0; ky < 4; ky++)
                {
                    AlgoType h = ALGO_ZERO;

                    for (int32_t kx = 0; kx < 4; kx++)
                    {
                        const int32_t sx =
                            ((x + ix + kx - 1) % sizeX + sizeX) % sizeX;

                        h += cx[kx] * rows[ky][sx];
                    }

                    acc += cy[ky] * h;
                }

                pOut[x] = MAX_VALUE(acc, ALGO_ZERO);
            }
        }

        return;
    }
}


// ---------------------------------------------------------------------------
//  Stage 10: scan MTF plus per-channel registration error
// ---------------------------------------------------------------------------
void AlgoStage10_ScanMtf
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrA,
    AlgoType* RESTRICT       pScrB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           scanF50,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const uint32_t           seed
) noexcept
{
    // Optical sigma of the scan, converted from millimetres on the film to pixels.
    const AlgoType sigmaPx = AlgoScanSigmaMm(scanF50) * pxPerMm;

    const bool wantBlur = (sigmaPx >= ALGO_SCAN_MIN_SIGMA_PX);

    // ----------------------------------------------------------------------
    //  Registration error, in pixels.
    //
    //  Specified on the negative in micrometres, so it scales with resolution like
    //  every other spatial quantity. Meaningless on a monochrome stock: there is
    //  one record, and a single record cannot be out of register with itself.
    // ----------------------------------------------------------------------
    const AlgoType misPx = static_cast<AlgoType>(profile.misregistration_um)
                         * pxPerMm * static_cast<AlgoType>(0.001)
                         * MAX_VALUE(static_cast<AlgoType>(params.misregScale),
                                     ALGO_ZERO);

    const bool wantShift = (misPx > ALGO_ZERO)
                        && (false == profile.is_monochrome);

    // Seed for this stage's jitter: the caller's global seed combined with the
    // per-call one. Named distinctly from the parameter, because a local that
    // shadows it and is XORed with itself is undefined behaviour - a mistake this
    // engine has already made once, in the coating field.
    const uint32_t scanSeed = static_cast<uint32_t>(params.seed) ^ seed;

    const AlgoType* RESTRICT srcPlane[3] = { pSrcR, pSrcG, pSrcB };
    AlgoType* RESTRICT       dstPlane[3] = { pDstR, pDstG, pDstB };

    for (int32_t c = 0; c < 3; c++)
    {
        const AlgoType* RESTRICT pIn  = srcPlane[c];
        AlgoType* RESTRICT       pOut = dstPlane[c];

        // ------------------------------------------------------------------
        //  Displacement for this record.
        //
        //  Drawn per frame and per channel, and INCLUDING the frame index, because
        //  registration jitter genuinely changes frame to frame - it is the
        //  scanner's transport, not a fixed optical alignment. A pure function of
        //  (seed, frameIndex, stage, ordinal), so scrubbing and out-of-order
        //  rendering stay stable.
        // ------------------------------------------------------------------
        HighPrecType dy = 0.0;
        HighPrecType dx = 0.0;

        if (wantShift)
        {
            // Two independent draws per channel. The ordinal separates the vertical
            // and horizontal components; the channel index separates the records.
            const uint64_t cy = AlgoRngCounter(scanSeed, frameIndex,
                                               eALGO_RNG_STAGE::eRNG_MISREG,
                                               static_cast<uint32_t>(c * 2));

            const uint64_t cx = AlgoRngCounter(scanSeed, frameIndex,
                                               eALGO_RNG_STAGE::eRNG_MISREG,
                                               static_cast<uint32_t>(c * 2 + 1));

            // Normal with the specified RMS. Gaussian rather than uniform because
            // the error is the sum of many small mechanical and optical
            // contributions.
            dy = AlgoRngNormal(cy) * static_cast<HighPrecType>(misPx);
            dx = AlgoRngNormal(cx) * static_cast<HighPrecType>(misPx);
        }

        // Whether the draw actually came out large enough to be worth resampling.
        const bool doShift = wantShift
                          && ((MAX_VALUE(dy, -dy) >= static_cast<HighPrecType>(
                                  ALGO_SCAN_MIN_SHIFT_PX))
                           || (MAX_VALUE(dx, -dx) >= static_cast<HighPrecType>(
                                  ALGO_SCAN_MIN_SHIFT_PX)));

        // ------------------------------------------------------------------
        //  Apply the two operations, arranging the buffers so that the LAST one
        //  performed writes the destination directly and no needless copy is made.
        // ------------------------------------------------------------------
        if (wantBlur && doShift)
        {
            // Blur into scratch, then shift scratch into the destination. The order
            // does not matter mathematically - a convolution and a translation
            // commute - but doing the blur first means the resample reads
            // already-smooth data, where bilinear interpolation is at its most
            // accurate.
            AlgoGaussianBlurPlaneWrap(pIn, pScrB, pScrA,
                                      sizeX, sizeY, pitch, sigmaPx);

            shiftPlaneWrap(pScrB, pOut, sizeX, sizeY, pitch, dy, dx);
        }
        else if (wantBlur)
        {
            AlgoGaussianBlurPlaneWrap(pIn, pOut, pScrA,
                                      sizeX, sizeY, pitch, sigmaPx);
        }
        else if (doShift)
        {
            shiftPlaneWrap(pIn, pOut, sizeX, sizeY, pitch, dy, dx);
        }
        else
        {
            // Neither operation applies. The copy is required by the
            // retained-buffer policy, not optional.
            AlgoCopyPlane(pIn, pOut, sizeX, sizeY, pitch);
        }

        // ------------------------------------------------------------------
        //  Floor at zero.
        //
        //  Neither a Gaussian blur nor a bilinear resample can produce a negative
        //  value from non-negative input, since both are convex combinations. The
        //  floor is here because it costs one streaming pass and guarantees the
        //  invariant every downstream stage relies on, rather than relying on that
        //  reasoning surviving a future change of interpolation kernel.
        // ------------------------------------------------------------------
        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow =
                pOut + static_cast<std::ptrdiff_t>(y) * pitch;

            ALGO_VECTOR_HINT
            for (int32_t x = 0; x < sizeX; x++)
                pRow[x] = MAX_VALUE(pRow[x], ALGO_ZERO);
        }
    }

    return;
}


// ---------------------------------------------------------------------------
//  Sub-stage 10b: narrow-gauge edge fog
// ---------------------------------------------------------------------------
void AlgoStage10b_EdgeFog
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           negWidthMm
) noexcept
{
    const film::CoatingSpec& coat = profile.coating;

    const AlgoType scale = MAX_VALUE(static_cast<AlgoType>(params.coatingScale),
                                     ALGO_ZERO);

    // Peak additive density at the very edge, and the distance inward over which it
    // decays by a factor of e.
    const AlgoType fogD  = static_cast<AlgoType>(coat.edge_fog_density) * scale;
    const AlgoType fogMm = static_cast<AlgoType>(coat.edge_fog_mm);

    // A gauge whose margins are trimmed leaves the density at zero; a frame width
    // of zero means the format could not be resolved and there is no millimetre
    // scale to measure the decay against. Either way the copy is required, not
    // optional, by the retained-buffer policy.
    if ((fogD <= ALGO_ZERO) || (fogMm <= ALGO_ZERO) || (negWidthMm <= ALGO_ZERO))
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    // Millimetres per pixel across the frame. The span is (n - 1) so that the first
    // and last pixel centres land exactly on the two physical edges.
    const HighPrecType mmPerPx = static_cast<HighPrecType>(negWidthMm)
                               / static_cast<HighPrecType>(MAX_VALUE(sizeX - 1, 1));

    const HighPrecType invFogMm = 1.0 / static_cast<HighPrecType>(fogMm);

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        const AlgoType* RESTRICT pR = pSrcR + off;
        const AlgoType* RESTRICT pG = pSrcG + off;
        const AlgoType* RESTRICT pB = pSrcB + off;

        AlgoType* RESTRICT pOR = pDstR + off;
        AlgoType* RESTRICT pOG = pDstG + off;
        AlgoType* RESTRICT pOB = pDstB + off;

        for (int32_t x = 0; x < sizeX; x++)
        {
            // Distance from this pixel to the NEARER of the two edges, in
            // millimetres. Taking the nearer edge is what makes the profile
            // symmetric: both margins fog, and a point in the middle is far from
            // both.
            // RULE D1 ALIGNMENT, 2026-08-11: was HighPrecType per pixel. A
            // position in millimetres across a 35 mm frame, used as the argument
            // of a decaying exponential - float32 resolves it to ~4e-06 mm,
            // which is a thousandth of a pixel.
            const AlgoType xMm = static_cast<AlgoType>(x) * static_cast<AlgoType>(mmPerPx);

            const AlgoType dEdge = MIN_VALUE(xMm,
                                   static_cast<AlgoType>(negWidthMm) - xMm);

            // Exponential decay inward. Exponential rather than linear because both
            // contributors - light leaking round the roll edge and developer
            // diffusing in from the margin - are diffusion processes.
            const AlgoType fog = fogD * static_cast<AlgoType>(
                                     std::exp(-dEdge * invFogMm));

            // ADDITIVE in density, and the same amount on all three records: the
            // fog is developed silver or dye from stray light, and stray light is
            // broadband. It is added rather than multiplied because that is what
            // density from an independent second exposure does.
            pOR[x] = pR[x] + fog;
            pOG[x] = pG[x] + fog;
            pOB[x] = pB[x] + fog;
        }
    }

    return;
}
#endif