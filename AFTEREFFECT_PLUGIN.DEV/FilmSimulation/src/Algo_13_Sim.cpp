// ---------------------------------------------------------------------------
//  Algo_13_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the print curve - the same difference of two softplus ramps as stage 8,
//  so the same vector softplus. The duplication generation loop calls the shared blur,
//  already vectorised.
//
//  ALIGNMENT: EVERY IMAGE ACCESS IS UNALIGNED, DELIBERATELY.
//
//  loadu/storeu on all plane data. The arena base comes from the host's pool, whose
//  alignment argument is a HINT - it was seen returning a base 16 mod 32, which faults
//  an aligned 256-bit load. AlgoMemHandler.cpp is SHARED by both flavours and must not
//  carry a vector-path concern, so the vector path assumes nothing about alignment.
//
//  Pipeline stage 13, in the density domain:
//
//      AlgoStage13_Duplication   dupe generations, then the release print
//
//  Raw pointers, explicit geometry, no allocation, no mutable state, no validation
//  of inputs.
// ---------------------------------------------------------------------------

#include "AlgoDuplication.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>

#include "AlgoHalation.hpp"   // AlgoSoftplus, shared by every curve evaluation


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

    // ----------------------------------------------------------------------
    //  Vector softplus:  k * log(1 + exp(x/k))
    //
    //  Identical in form to the one in the AVX2 stage 8, deliberately: the print curve
    //  here and the negative curve there are the same difference of two softplus ramps,
    //  so if the two implementations diverged the print would stop agreeing with the
    //  negative it was solved against.
    //
    //  The linear asymptote above ALGO_SOFTPLUS_LINEAR_LIMIT is a CORRECTNESS guard,
    //  not an optimisation: far up the ramp softplus equals x to beyond the last
    //  representable bit while the exponential heads for overflow. Selected with a
    //  blend rather than branched, so all eight lanes stay on one path.
    // ----------------------------------------------------------------------
    FORCE_INLINE __m256 algoSoftplusV (const __m256 x,
                                       const __m256 k,
                                       const __m256 invK) noexcept
    {
        const __m256 z = _mm256_mul_ps(x, invK);

        const __m256 vLimit = _mm256_set1_ps(
            static_cast<float>(ALGO_SOFTPLUS_LINEAR_LIMIT));

        // Ordered, non-signalling compare, so a NaN takes the log path rather than
        // raising.
        const __m256 useLinear = _mm256_cmp_ps(z, vLimit, _CMP_GT_OQ);

        const __m256 u = FastCompute::AVX2::Exp(z);

        const __m256 lg =
            FastCompute::AVX2::Log(_mm256_add_ps(u, _mm256_set1_ps(1.0f)));

        return _mm256_blendv_ps(_mm256_mul_ps(k, lg), x, useLinear);
    }
}


namespace
{
    // ----------------------------------------------------------------------
    //  Curve accessor by index, so the three channels share one code path.
    // ----------------------------------------------------------------------
    inline const film::ToneCurve& curveAt
    (
        const film::RGBCurves& set,
        const int32_t          c
    ) noexcept
    {
        return (0 == c) ? set.r
             : (1 == c) ? set.g
                        : set.b;
    }


    // ----------------------------------------------------------------------
    //  Evaluate one characteristic curve over a whole plane, given a per-pixel
    //  log exposure of the form (offset - incomingDensity).
    //
    //  This is the printing operation itself: a positive-going offset minus the
    //  density that is blocking the light. It appears three times in this stage -
    //  once per dupe generation and once for the final print - so it is written once
    //  here rather than three times inline.
    // ----------------------------------------------------------------------
    void printPlane
    (
        const AlgoType* RESTRICT pSrc,
        AlgoType* RESTRICT       pDst,
        const int32_t            sizeX,
        const int32_t            sizeY,
        const int32_t            pitch,
        const film::ToneCurve&   curve,
        const AlgoType           offset
    ) noexcept
    {
        // Curve parameters hoisted into frame constants: they are stored as float
        // while the arithmetic runs in AlgoType, and the compiler cannot prove the
        // profile is unchanged by the stores into the destination.
        const AlgoType dmin  = static_cast<AlgoType>(curve.dmin);
        const AlgoType gamma = static_cast<AlgoType>(curve.gamma);
        const AlgoType toeX  = static_cast<AlgoType>(curve.toe_x);
        const AlgoType toeK  = static_cast<AlgoType>(curve.toe_k);
        const AlgoType shX   = static_cast<AlgoType>(curve.shoulder_x);
        const AlgoType shK   = static_cast<AlgoType>(curve.shoulder_k);

        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pIn  = pSrc + off;
            AlgoType* RESTRICT       pOut = pDst + off;

            // Print exposure. More density in front of the light means less exposure
            // behind it, which is the inversion that makes a positive.
            const __m256 vOffset = _mm256_set1_ps(offset);
            const __m256 vDmin   = _mm256_set1_ps(dmin);
            const __m256 vGamma  = _mm256_set1_ps(gamma);
            const __m256 vToeX   = _mm256_set1_ps(toeX);
            const __m256 vShX    = _mm256_set1_ps(shX);
            const __m256 vToeK   = _mm256_set1_ps(toeK);
            const __m256 vShK    = _mm256_set1_ps(shK);

            // Reciprocals of the knee widths formed once per plane so the pixel loop
            // multiplies. Guarded only against a malformed profile; real curves have
            // positive widths.
            const __m256 vInvToeK =
                _mm256_set1_ps((toeK > ALGO_ZERO) ? (ALGO_ONE / toeK) : ALGO_ZERO);
            const __m256 vInvShK =
                _mm256_set1_ps((shK  > ALGO_ZERO) ? (ALGO_ONE / shK)  : ALGO_ZERO);

            const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
            const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
            const __m256i mt = algoTailMaskLocal(nt);

            int32_t x = 0;

            for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
            {
                const __m256 logE =
                    _mm256_sub_ps(vOffset, _mm256_loadu_ps(pIn + x));

                const __m256 rise =
                    algoSoftplusV(_mm256_sub_ps(logE, vToeX), vToeK, vInvToeK);
                const __m256 fall =
                    algoSoftplusV(_mm256_sub_ps(logE, vShX),  vShK,  vInvShK);

                _mm256_storeu_ps(pOut + x,
                    _mm256_fmadd_ps(vGamma, _mm256_sub_ps(rise, fall), vDmin));
            }

            if (nt > 0)
            {
                const __m256 logE =
                    _mm256_sub_ps(vOffset, _mm256_maskload_ps(pIn + x, mt));

                const __m256 rise =
                    algoSoftplusV(_mm256_sub_ps(logE, vToeX), vToeK, vInvToeK);
                const __m256 fall =
                    algoSoftplusV(_mm256_sub_ps(logE, vShX),  vShK,  vInvShK);

                _mm256_maskstore_ps(pOut + x, mt,
                    _mm256_fmadd_ps(vGamma, _mm256_sub_ps(rise, fall), vDmin));
            }
        }

        return;
    }


    // ----------------------------------------------------------------------
    //  Floor three planes at zero.
    //
    //  Negative optical density has no physical meaning, and grain is zero-mean so
    //  it can drive a light area below base.
    // ----------------------------------------------------------------------
    void floorImage
    (
        AlgoType* RESTRICT pR,
        AlgoType* RESTRICT pG,
        AlgoType* RESTRICT pB,
        const int32_t      sizeX,
        const int32_t      sizeY,
        const int32_t      pitch
    ) noexcept
    {
        AlgoType* RESTRICT plane[3] = { pR, pG, pB };

        for (int32_t c = 0; c < 3; c++)
        {
            for (int32_t y = 0; y < sizeY; y++)
            {
                AlgoType* RESTRICT pRow =
                    plane[c] + static_cast<std::ptrdiff_t>(y) * pitch;

                ALGO_VECTOR_HINT
                for (int32_t x = 0; x < sizeX; x++)
                    pRow[x] = MAX_VALUE(pRow[x], ALGO_ZERO);
            }
        }

        return;
    }
}


// ---------------------------------------------------------------------------
//  Stage 13: duplication generations, then the release print
// ---------------------------------------------------------------------------
void AlgoStage13_Duplication
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrTmpR,
    AlgoType* RESTRICT       pScrTmpG,
    AlgoType* RESTRICT       pScrTmpB,
    AlgoType* RESTRICT       pScrNoise,
    AlgoType* RESTRICT       pScrLobe,
    AlgoType* RESTRICT       pScrWork,
    AlgoType* RESTRICT       pScrField,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const film::PrintStock*  pPrintStock,
    const film::PrintStock*  pDupeStock,
    const AlgoType           scanSigmaPx,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const uint32_t           seed,
    film::RGBCurves&         finalCurvesOut
) noexcept
{
    // ----------------------------------------------------------------------
    //  A reversal stock already IS the positive.
    //
    //  Its own Dmin and Dmax are the white and black points. There is no second
    //  curve, no inversion and no dupe chain - a slide is projected as shot. The
    //  copy is required by the retained-buffer policy, not optional.
    // ----------------------------------------------------------------------
    if (profile.isReversal() || (nullptr == pPrintStock))
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

        // The curve set that produced this output is the stock's own, and stage 14
        // needs it for the transmittance endpoints.
        finalCurvesOut = profile.curves;

        return;
    }

    // Working copy. Every step below is read-modify-write, so the destination starts
    // out holding the incoming negative densities.
    AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    AlgoType* RESTRICT work[3] = { pDstR,    pDstG,    pDstB    };
    AlgoType* RESTRICT tmp [3] = { pScrTmpR, pScrTmpG, pScrTmpB };

    const AlgoType grainGain = MAX_VALUE(static_cast<AlgoType>(params.grainScale),
                                         ALGO_ZERO);

    // Seed for this stage. Named distinctly from the parameter: a local that shadowed
    // it and was XORed with itself would be undefined behaviour.
    const uint32_t dupeSeed = static_cast<uint32_t>(params.seed) ^ seed;

    // ----------------------------------------------------------------------
    //  Neutral density leaving the camera negative.
    //
    //  Every printing generation anchors against this, and each one moves it, so it
    //  is carried through the loop rather than recomputed.
    // ----------------------------------------------------------------------
    HighPrecType dMid[3];

    // ⚠ AND THIS REFERENCE MUST SEE THE READER'S OPTICS TOO (C22) -- see the
    // scalar Algo_13_Sim.cpp for why the two mid-grey references have to agree.
    AlgoNeutralMidDensity(profile,
                          static_cast<HighPrecType>(params.couplerScale),
                          static_cast<HighPrecType>(params.scannerSpecular),
                          dMid);

    // ----------------------------------------------------------------------
    //  The duplication chain.
    //
    //  Generations come in PAIRS so the polarity always returns to negative before
    //  the release print. Capped, because each pair costs a blur, a curve and a
    //  grain field on every pixel.
    // ----------------------------------------------------------------------
    const int32_t generations = CLAMP_VALUE(params.generations, 0,
                                            ALGO_DUPE_MAX_GENERATIONS);

    const int32_t passes = (nullptr != pDupeStock) ? (2 * generations) : 0;

    if (passes > 0)
    {
        const film::RGBCurves& dcurves = pDupeStock->curves;

        // Printing optics of the duplicating step, as a sigma in pixels.
        const AlgoType dupeSigmaPx =
            AlgoScanSigmaMm(static_cast<AlgoType>(pDupeStock->mtf_f50)) * pxPerMm;

        const bool wantDupeBlur = (dupeSigmaPx >= ALGO_SCAN_MIN_SIGMA_PX);

        // Base plus fog of the duplicating stock, needed by the grain weighting.
        const AlgoType dupeDmin[3] =
        {
            static_cast<AlgoType>(dcurves.r.dmin),
            static_cast<AlgoType>(dcurves.g.dmin),
            static_cast<AlgoType>(dcurves.b.dmin)
        };

        const bool wantDupeGrain = (grainGain > ALGO_ZERO)
                                && (pDupeStock->grain_rms > 0.0f);

        for (int32_t pass = 0; pass < passes; pass++)
        {
            // --------------------------------------------------------------
            //  1. The printing optics blur what comes IN.
            //
            //  That is the accumulated image AND all grain from every earlier
            //  generation. It has to happen before the new stock records anything.
            // --------------------------------------------------------------
            for (int32_t c = 0; c < 3; c++)
            {
                if (wantDupeBlur)
                    AlgoGaussianBlurPlaneWrap(work[c], tmp[c], pScrWork,
                                              sizeX, sizeY, pitch, dupeSigmaPx);
                else
                    AlgoCopyPlane(work[c], tmp[c], sizeX, sizeY, pitch);
            }

            // --------------------------------------------------------------
            //  2. Offsets centring the neutral in this stock's usable range, and
            //     the neutral density this generation will hand on.
            //
            //  Nothing views an intermediate, so there is no display value to aim
            //  at; the midpoint of the stock's own range is what a laboratory aims
            //  at and what keeps a four-generation chain out of the toe and the
            //  shoulder.
            // --------------------------------------------------------------
            HighPrecType offs[3];
            HighPrecType nextMid[3];

            AlgoSolveIntermediateOffsets(dMid, dcurves, offs, nextMid);

            for (int32_t k = 0; k < 3; k++)
                dMid[k] = nextMid[k];

            // --------------------------------------------------------------
            //  3. The new emulsion records the blurred image, inverting polarity.
            // --------------------------------------------------------------
            for (int32_t c = 0; c < 3; c++)
                printPlane(tmp[c], work[c], sizeX, sizeY, pitch,
                           curveAt(dcurves, c),
                           static_cast<AlgoType>(offs[c]));

            // --------------------------------------------------------------
            //  4. THIS generation's own grain, added AFTER the blur.
            //
            //  It is created in this emulsion, so it is not blurred by this stage's
            //  optics - only by later ones. Adding it before the blur would soften
            //  every generation's grain by its own MTF and make a long dupe chain
            //  come out cleaner than a short one, which is backwards.
            //
            //  One field for all three channels: duplicating stock is a single
            //  black-and-white emulsion in each of the three separations, and its
            //  grain is achromatic.
            // --------------------------------------------------------------
            if (wantDupeGrain)
            {
                // The pass index enters the frame argument of the generator, so each
                // generation gets an independent field while the whole chain stays a
                // pure function of (seed, frameIndex, pass).
                AlgoMakeGrainField(pScrField, pScrNoise, pScrLobe, pScrWork,
                                   sizeX, sizeY, pitch,
                                   static_cast<AlgoType>(pDupeStock->grain_clump_um),
                                   ALGO_GRAIN_DUPE_CLUMP_GAIN,
                                   static_cast<AlgoType>(pDupeStock->grain_rms),
                                   scanSigmaPx, pxPerMm,
                                   eALGO_RNG_STAGE::eRNG_DUPE_GRAIN,
                                   dupeSeed,
                                   frameIndex * (ALGO_DUPE_MAX_GENERATIONS * 2 + 1)
                                       + pass + 1);

                AlgoAddGrain(work[0], work[1], work[2],
                             pScrField, pScrField, pScrField,
                             sizeX, sizeY, pitch,
                             dupeDmin, ALGO_GRAIN_DUPE_FOG, grainGain);
            }

            floorImage(work[0], work[1], work[2], sizeX, sizeY, pitch);
        }
    }

    // ----------------------------------------------------------------------
    //  The release print.
    //
    //  logE_print = offset - D. Higher scene exposure raised negative density,
    //  which lowers print exposure, which lowers print density, which brightens the
    //  positive. That double inversion is what gives correct rolloff at both ends
    //  for free.
    // ----------------------------------------------------------------------
    const film::RGBCurves& pcurves = pPrintStock->curves;

    // Where a neutral has to land, per channel, carrying its share of the base tint
    // so an orange mask neither prints to a dead neutral nor prints full orange.
    HighPrecType target[3];

    for (int32_t c = 0; c < 3; c++)
        target[c] = static_cast<HighPrecType>(params.greyTarget)
                  / AlgoTintFactor(profile, c);

    // The printer-light setting. Re-solved here rather than reused from stage 8,
    // because the dupe chain above has moved the neutral density.
    HighPrecType offsets[3];

    AlgoSolveStageOffsets(dMid, pcurves, pPrintStock->dye_matrix, target, offsets);

    // Print each record through the print stock's own curve. Written into the
    // scratch triple first, because the print reads the negative density while
    // writing the print density and the two are different quantities.
    for (int32_t c = 0; c < 3; c++)
        printPlane(work[c], tmp[c], sizeX, sizeY, pitch,
                   curveAt(pcurves, c),
                   static_cast<AlgoType>(offsets[c]));

    // ----------------------------------------------------------------------
    //  The print stock's own dye impurity.
    //
    //  Same mechanism as stage 12, on a different set of dyes: print dyes are not
    //  spectrally pure either, and their impurity is what sets the print's gamut.
    // ----------------------------------------------------------------------
    if (AlgoIsIdentityMatrix(pPrintStock->dye_matrix))
    {
        AlgoCopyImage(tmp[0], tmp[1], tmp[2],
                      pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
    }
    else
    {
        AlgoApplyDensityMatrix(tmp[0], tmp[1], tmp[2],
                               pDstR, pDstG, pDstB,
                               sizeX, sizeY, pitch,
                               pPrintStock->dye_matrix);
    }

    floorImage(pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    // The print stock's curves produced this output, so they are the endpoints stage
    // 14 must use for the transmittance conversion and for print grain.
    finalCurvesOut = pcurves;

    return;
}
