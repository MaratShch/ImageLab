// ---------------------------------------------------------------------------
//  Algo_14_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: density to transmittance, which is 10^-d per sample per channel and the
//  second-largest exponential population in the engine after stage 8. Evaluated as
//  Exp2Accurate(-d * log2 10) because there is no vector pow - which is the same base
//  change the scalar twin makes with std::exp2. ⚠ It called the Schraudolph Exp() until
//  2026-09-11; that was a 2.98 % model error, not a rounding one. See the note at the
//  call site.
//
//  ALIGNMENT: EVERY IMAGE ACCESS IS UNALIGNED, DELIBERATELY.
//
//  loadu/storeu on all plane data. The arena base comes from the host's pool, whose
//  alignment argument is a HINT - it was seen returning a base 16 mod 32, which faults
//  an aligned 256-bit load. AlgoMemHandler.cpp is SHARED by both flavours and must not
//  carry a vector-path concern, so the vector path assumes nothing about alignment.
//
//  Pipeline stage 14 and its sub-stages 14b and 14c:
//
//      AlgoStage14_Transmittance       print grain, then density to transmittance
//      AlgoStage14b_ReseauReconstruct  additive colour rebuilt through the grid
//      AlgoStage14c_SilverTone         non-neutral developed silver
//
//  This is where the pipeline leaves the density domain. Everything before is
//  density, everything after is display-linear transmittance.
//
//  Raw pointers, explicit geometry, no allocation, no mutable state, no validation
//  of inputs.
// ---------------------------------------------------------------------------

// Common.hpp -- AVX2_ALIGN / CACHE_ALIGN are defined here. Included
// DIRECTLY rather than relied on transitively: this file declares an
// aligned buffer, so the macro must not depend on another header's
// include order to be in scope.
#include "Common.hpp"
#include "AlgoTransmittance.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>
#include "AlgoReseauReconstruct.hpp"
#include "AlgoSilverTone.hpp"

#include <cmath>   // std::pow


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

    // ----------------------------------------------------------------------
    //  algoExp2AccurateV: 2^x to float precision.
    //
    //  ⚠ IT LIVES HERE, IN THIS TRANSLATION UNIT, AND THAT PLACEMENT IS THE
    //  FIX FOR A REAL INTEGRATION FAILURE. It was first written into
    //  FastAriphmeticsAVX.hpp, which was the obvious home -- that is where the
    //  other vector transcendentals are. But that header is a COMMON
    //  component: in the owner's tree it lives in CPP\Common\include and is
    //  shared by every project, while this file ships in the AVX2 algorithm
    //  archive. The two archives are deliberately separate, so the updated
    //  Common header never reached the build, the stale copy on the include
    //  path won, and the compiler reported
    //
    //      error C2039: 'Exp2Accurate': is not a member of 'FastCompute::AVX2'
    //
    //  A stage may not depend on a change to a component it does not ship
    //  with. One consumer, one translation unit, no cross-archive coupling.
    //
    //  ⚠ AND IT EXISTS AT ALL BECAUSE FastCompute::AVX2::Exp IS NOT ACCURATE
    //  ENOUGH FOR THIS STAGE. That function is the raw Schraudolph bit-hack:
    //  one FMA and a reinterpret, no polynomial refinement. Measured against
    //  exact 10^-d over D = 0..4 it is wrong by up to 2.98 % relative -- 5.57
    //  code values of 8-bit output -- and it returns 0.978161 at D = 0, so the
    //  vector build rendered its WHITE POINT 2.2 % low. That is three orders of
    //  magnitude past float rounding: a MODEL difference, not a precision one,
    //  and the scalar/AVX2 split is only ever allowed to be the latter.
    //
    //  METHOD. Split x into a whole part n and a fraction f in [-0.5, 0.5]:
    //  2^x = 2^n * 2^f. The 2^n factor is EXACT -- it is built by placing
    //  n + 127 directly in the exponent field. 2^f is a degree-5 minimax
    //  polynomial in f, which is what carries the accuracy. Measured 0.000326 %
    //  worst relative error over the same domain, and exactly 1.0 at D = 0.
    // ----------------------------------------------------------------------
    inline __m256 algoExp2AccurateV (__m256 x) noexcept
    {
        // 2^-126 .. 2^127 is the normal-float range; clamping inside it stops
        // the exponent assembly below from overflowing into a NaN.
        x = _mm256_max_ps(x, _mm256_set1_ps(-126.0f));
        x = _mm256_min_ps(x, _mm256_set1_ps( 127.0f));

        // n = round-to-nearest(x), f = x - n in [-0.5, 0.5].
        const __m256 n = _mm256_round_ps(
            x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m256 f = _mm256_sub_ps(x, n);

        // 2^f, degree-5 minimax on [-0.5, 0.5], Horner order.
        __m256 p = _mm256_set1_ps(1.3352819600e-3f);
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6178398092e-3f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5503406540e-2f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022650696e-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718056e-1f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));

        // 2^n by direct exponent construction: (n + 127) << 23.
        const __m256i bias = _mm256_add_epi32(_mm256_cvtps_epi32(n),
                                              _mm256_set1_epi32(127));
        const __m256  pow2n = _mm256_castsi256_ps(
            _mm256_slli_epi32(bias, 23));

        return _mm256_mul_ps(p, pow2n);
    }

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


// ---------------------------------------------------------------------------
//  Stage 14: print grain, then transmittance
// ---------------------------------------------------------------------------
void AlgoStage14_Transmittance
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
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
    const film::RGBCurves&   finalCurves,
    const bool               isReversal,
    const AlgoType           scanSigmaPx,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const uint32_t           seed
) noexcept
{
    // Working copy, so the grain can be added in place before the conversion.
    AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    AlgoType* RESTRICT dstPlane[3] = { pDstR, pDstG, pDstB };

    // Endpoints of the curve set that actually produced this image, per channel.
    const film::ToneCurve* const curve[3] =
    {
        &finalCurves.r, &finalCurves.g, &finalCurves.b
    };

    const AlgoType grainGain = MAX_VALUE(static_cast<AlgoType>(params.grainScale),
                                         ALGO_ZERO);

    // ----------------------------------------------------------------------
    //  Print grain.
    //
    //  Only for a negative: a reversal stock is projected as shot and never touches
    //  a print emulsion.
    //
    //  One field for all three channels. Print grain is finer than negative grain
    //  and largely achromatic, so three independent fields would give it a colour it
    //  does not have.
    // ----------------------------------------------------------------------
    if ((false == isReversal) && params.printGrain && (grainGain > ALGO_ZERO)
        && (nullptr != pPrintStock) && (pPrintStock->grain_rms > 0.0f))
    {
        // Seed named distinctly from the parameter: a local shadowing it and XORed
        // with itself would be undefined behaviour.
        const uint32_t printSeed = static_cast<uint32_t>(params.seed) ^ seed;

        AlgoMakeGrainField(pScrField, pScrNoise, pScrLobe, pScrWork,
                           sizeX, sizeY, pitch,
                           static_cast<AlgoType>(pPrintStock->grain_clump_um),
                           ALGO_GRAIN_PRINT_CLUMP_GAIN,
                           static_cast<AlgoType>(pPrintStock->grain_rms),
                           scanSigmaPx, pxPerMm,
                           // ⚠ ROUND GRAIN, and the STOCK is what says so.
                           // Anisotropy is a measured property of a coated
                           // emulsion; film::PrintStock carries no such field,
                           // because no print stock in the database has been
                           // measured for coating flow. Inheriting the camera
                           // negative's figure would attribute one emulsion's
                           // flow direction to a different emulsion coated in a
                           // different factory.
                           ALGO_GRAIN_ANISOTROPY_NONE,
                           eALGO_RNG_STAGE::eRNG_PRINT_GRAIN,
                           printSeed, frameIndex);

        // Base plus fog of the PRINT curves, not the film's: the grain sits in the
        // print emulsion and its amplitude follows the print's developed density.
        const AlgoType dmin[3] =
        {
            static_cast<AlgoType>(finalCurves.r.dmin),
            static_cast<AlgoType>(finalCurves.g.dmin),
            static_cast<AlgoType>(finalCurves.b.dmin)
        };

        // The gain is one rather than params.grainScale, because the reference
        // applies the user scale to the field's own amplitude here and not a second
        // time in the weighting - print grain is already the smaller of the two
        // contributions and double-scaling it would make the control non-linear.
        // ⚠ AlgoAddGrainRaw, NOT AlgoAddGrain. Print grain carries no published
        // rms figure to be pinned to, so it keeps the UNNORMALISED weighting the
        // reference uses; pinning it would move every print render for no
        // measurement's sake. See AlgoGrainAmpRaw.
        AlgoAddGrainRaw(pDstR, pDstG, pDstB,
                        pScrField, pScrField, pScrField,
                        sizeX, sizeY, pitch, dmin,
                        ALGO_GRAIN_DUPE_FOG, ALGO_ONE);
    }

    // ----------------------------------------------------------------------
    //  Density to display-linear transmittance.
    // ----------------------------------------------------------------------
    for (int32_t c = 0; c < 3; c++)
    {
        // Transmittance is ten to the minus density, by the definition of optical
        // density. Clear film at Dmin is the brightest the stock can be; Dmax is the
        // darkest.
        const HighPrecType tMax = std::pow(10.0,
            -static_cast<HighPrecType>(curve[c]->dmin));

        // \warning THE BLACK POINT IS A CONTROL AS OF 2026-09-09c (queue item
        // #303). tMin used to be 10^-Dmax unconditionally, which stretched every
        // stock's Dmax down to output zero and destroyed everything at or past
        // it -- measured, 13.06 per cent of the blue record on the Velvia
        // regression frame. At the default 1.0 this multiply is the identity and
        // the expression is arithmetically what it always was. The anchor solve
        // at stage 08 sees the SAME value, which is the part that matters: the
        // solve aims at a display number this loop produces.
        const HighPrecType tMin =
            static_cast<HighPrecType>(params.blackPointStretch)
          * std::pow(10.0,
            -static_cast<HighPrecType>(curve[c]->dmax()));

        const HighPrecType span = tMax - tMin;

        // A curve with zero gamma has no span at all and no meaningful
        // normalisation. Passing the reciprocal through as zero maps everything to
        // the black point, which is the only defined answer and cannot produce a
        // division by zero or a not-a-number.
        const AlgoType invSpan = (span > 0.0)
                               ? static_cast<AlgoType>(1.0 / span)
                               : ALGO_ZERO;

        const AlgoType tMinA = static_cast<AlgoType>(tMin);

        const AlgoType dMaxA = static_cast<AlgoType>(curve[c]->dmax());

        AlgoType* RESTRICT pD = dstPlane[c];

        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow = pD + static_cast<std::ptrdiff_t>(y) * pitch;

            // 10^-d as Exp2Accurate(-d * log2(10)), which is EXACTLY the base
            // change the scalar twin makes at Algo_14_Sim.cpp with std::exp2.
            // There is no vector pow, and the change of base is one multiply.
            //
            // ⚠ THIS USED TO CALL FastCompute::AVX2::Exp AND THAT WAS A DEFECT,
            // FIXED 2026-09-11. That function is the raw Schraudolph bit-hack:
            // one FMA and a reinterpret, no polynomial refinement. Measured
            // against exact 10^-d over D = 0..4 it is wrong by up to 2.98 %
            // relative -- 5.57 code values of 8-bit output -- and it returns
            // 0.978161 at D = 0, so the vector build rendered its WHITE POINT
            // 2.2 % low. Two things made that worse than a bare number:
            //
            //   * the tMin/tMax normalisation anchors a few lines above are
            //     computed with exact std::pow, so the vector path was
            //     internally inconsistent -- pixel and anchor disagreed;
            //   * stage 17 clips at 1.0, so the error MOVED WHERE HIGHLIGHTS
            //     CLIP rather than just shifting a level.
            //
            // A 2.98 % model error is three orders of magnitude past float
            // rounding. The scalar/AVX2 split is a PRECISION difference by
            // design and must never become a MODEL difference; this was one.
            // Exp2Accurate is a degree-5 minimax with exact exponent
            // assembly: measured 0.000326 % worst relative error over the same
            // domain, 0.0006 code values, and exactly 1.0 at D = 0.
            const __m256 vNegLog2_10 = _mm256_set1_ps(
                -static_cast<float>(3.32192809488736234787031942948939018));
            const __m256 vZeroL   = _mm256_setzero_ps();
            const __m256 vTMinA   = _mm256_set1_ps(tMinA);
            const __m256 vInvSpan = _mm256_set1_ps(invSpan);
            const __m256 vDMaxA   = _mm256_set1_ps(dMaxA);

            const int32_t nvL = sizeX / ALGO_AVX2_LANES_LOCAL;
            const int32_t ntL = sizeX - nvL * ALGO_AVX2_LANES_LOCAL;
            const __m256i mtL = algoTailMaskLocal(ntL);

            int32_t xv = 0;

            for (int32_t vv = 0; vv < nvL; vv++, xv += ALGO_AVX2_LANES_LOCAL)
            {
                // Density floored at zero before the exponentiation: a negative
                // density would give a transmittance above one - a material that
                // emits light.
                // \warning AND DENSITY IS CAPPED AT Dmax FIRST (2026-09-09c). A
                // layer cannot be denser than its own maximum dye load, yet
                // stages 09 and 12 are unbounded additions in the density
                // domain and do exceed it -- measured, blue on Velvia reaches
                // 4.2439 against a Dmax of 3.3072. At blackPointStretch 1.0 the
                // cap is OUTPUT-NEUTRAL; below 1.0 it is what stops an
                // over-dense pixel encoding to code zero anyway.
                const __m256 dv = _mm256_min_ps(
                    _mm256_max_ps(_mm256_loadu_ps(pRow + xv), vZeroL), vDMaxA);

                const __m256 tv =
                    algoExp2AccurateV(_mm256_mul_ps(dv, vNegLog2_10));

                // Normalised against the stock's OWN range, and deliberately NOT capped
                // at one: the single final clamp belongs to stage 17, and print grain
                // can legitimately push a highlight past the nominal white point.
                _mm256_storeu_ps(pRow + xv, _mm256_max_ps(
                    _mm256_mul_ps(_mm256_sub_ps(tv, vTMinA), vInvSpan), vZeroL));
            }

            if (ntL > 0)
            {
                const __m256 dv = _mm256_min_ps(
                    _mm256_max_ps(_mm256_maskload_ps(pRow + xv, mtL), vZeroL),
                    vDMaxA);

                const __m256 tv =
                    algoExp2AccurateV(_mm256_mul_ps(dv, vNegLog2_10));

                _mm256_maskstore_ps(pRow + xv, mtL, _mm256_max_ps(
                    _mm256_mul_ps(_mm256_sub_ps(tv, vTMinA), vInvSpan), vZeroL));
            }

            // Reference expression, retained beside the vector form it must agree with.
            // Never executes.
            for (int32_t x = 0; x < 0; x++)
            {
                const HighPrecType d = static_cast<HighPrecType>(
                    MIN_VALUE(MAX_VALUE(pRow[x], ALGO_ZERO), dMaxA));

                const AlgoType trans = static_cast<AlgoType>(std::pow(10.0, -d));

                // Normalised against the stock's OWN range, which is what makes the
                // output display referred without a separate grade, and what the
                // anchor solves at stages 8 and 13 aimed at. The two must use the
                // same expression or a neutral will not land where it was solved to.
                //
                // Deliberately NOT capped at one. The single final clamp belongs to
                // stage 17, and print grain can legitimately push a highlight above
                // the nominal white point before then.
                pRow[x] = MAX_VALUE((trans - tMinA) * invSpan, ALGO_ZERO);
            }
        }
    }

    // profile is carried in the signature for symmetry with every other stage and
    // because the print-grain decision may later need the stock's own figures.
    (void)profile;

    return;
}


// ---------------------------------------------------------------------------
//  Sub-stage 14b: reseau reconstruction, then residual base tint
// ---------------------------------------------------------------------------
void AlgoStage14b_ReseauReconstruct
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrMasked,
    AlgoType* RESTRICT       pScrMask,
    AlgoType* RESTRICT       pScrNum,
    AlgoType* RESTRICT       pScrDen,
    AlgoType* RESTRICT       pScrWork,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           pxPerMm
) noexcept
{
    const film::ReseauSpec& spec = profile.reseau;

    // The pitch this render actually produced. Recomputed from the same helper stage
    // 7 used, so the grid here is guaranteed to be the grid the record was made
    // through - no mask travels between the two stages and none can go stale.
    const AlgoType pitchPx = AlgoReseauPitchPx(spec, pxPerMm);

    const bool wantReseau = profile.has_reseau
                         && params.reseau
                         && (pitchPx >= ALGO_RESEAU_MIN_PITCH_PX);

    if (wantReseau)
    {
        // Reciprocal formed once: the per-pixel cell lookup is then a multiply.
        const HighPrecType invPitch = 1.0 / static_cast<HighPrecType>(pitchPx);

        // ------------------------------------------------------------------
        //  Reconstruction blur radius.
        //
        //  Expressed in GRID PITCHES, converted to micrometres by the cell size and
        //  then to pixels, so it tracks the grid at any resolution.
        //
        //  Deliberately comparable to the pitch rather than much larger: that is
        //  what leaves the faint grid texture visible and caps colour resolution
        //  below luminance resolution, both of which are real and characteristic.
        // ------------------------------------------------------------------
        const AlgoType sigmaPx = static_cast<AlgoType>(spec.reconstruction_pitches)
                               * pitchPx;

        // The single monochrome record. The green plane, matching the reference; on a
        // mosaic stock all three planes carry the same values anyway.
        const AlgoType* RESTRICT pRecord = pSrcG;

        AlgoType* RESTRICT dstPlane[3] = { pDstR, pDstG, pDstB };

        for (int32_t c = 0; c < 3; c++)
        {
            // --------------------------------------------------------------
            //  Split the record into "this channel's cells" and "the mask itself".
            //
            //  The mask is one-hot, so masking the record keeps only the cells of
            //  this colour and zeroes the rest.
            // --------------------------------------------------------------
            for (int32_t y = 0; y < sizeY; y++)
            {
                const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

                const AlgoType* RESTRICT pR = pRecord    + off;
                AlgoType* RESTRICT       pM = pScrMasked + off;
                AlgoType* RESTRICT       pK = pScrMask   + off;

                for (int32_t x = 0; x < sizeX; x++)
                {
                    // One when this pixel's cell carries this colour's filter.
                    const AlgoType hit =
                        (AlgoReseauFilterIndex(x, y, invPitch) == c)
                            ? ALGO_ONE : ALGO_ZERO;

                    pM[x] = pR[x] * hit;
                    pK[x] = hit;
                }
            }

            // Blur both. Same kernel, because the quotient is only a correct local
            // average if numerator and denominator were weighted identically.
            AlgoGaussianBlurPlaneWrap(pScrMasked, pScrNum, pScrWork,
                                      sizeX, sizeY, pitch, sigmaPx);

            AlgoGaussianBlurPlaneWrap(pScrMask, pScrDen, pScrWork,
                                      sizeX, sizeY, pitch, sigmaPx);

            // --------------------------------------------------------------
            //  Divide: the coverage normalisation.
            //
            //  Without it a channel occupying a third of the area would come out at
            //  a third brightness. With it, the result is the local average of the
            //  record over this colour's cells only - which is exactly what the eye
            //  integrates when the print is projected through the grid.
            // --------------------------------------------------------------
            AlgoType* RESTRICT pO = dstPlane[c];

            for (int32_t y = 0; y < sizeY; y++)
            {
                const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

                const AlgoType* RESTRICT pN = pScrNum + off;
                const AlgoType* RESTRICT pD = pScrDen + off;

                AlgoType* RESTRICT rO = pO + off;

                for (int32_t x = 0; x < sizeX; x++)
                    rO[x] = pN[x] / MAX_VALUE(pD[x], ALGO_RESEAU_COVERAGE_FLOOR);
            }
        }
    }
    else
    {
        // Not a mosaic stock, or the grid was never built. The copy is required by
        // the retained-buffer policy, not optional.
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
    }

    // ----------------------------------------------------------------------
    //  Residual base tint.
    //
    //  A real printer neutralises the film base colour, so only a small residual
    //  survives. The anchor solves already aimed at tint-adjusted targets using the
    //  same fraction, so the two halves of the split agree by construction: they
    //  read the same constant.
    // ----------------------------------------------------------------------
    AlgoType* RESTRICT outPlane[3] = { pDstR, pDstG, pDstB };

    for (int32_t c = 0; c < 3; c++)
    {
        const AlgoType tint = static_cast<AlgoType>(
            AlgoTintFactor(profile, c));

        // Exactly one means no tint at all on this channel, and skipping the pass
        // saves a streaming read and write of a whole plane.
        if (tint == ALGO_ONE)
            continue;

        AlgoType* RESTRICT pO = outPlane[c];

        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow = pO + static_cast<std::ptrdiff_t>(y) * pitch;

            ALGO_VECTOR_HINT
            for (int32_t x = 0; x < sizeX; x++)
                pRow[x] *= tint;
        }
    }

    return;
}


// ---------------------------------------------------------------------------
//  Sub-stage 14c: silver image tone
// ---------------------------------------------------------------------------
void AlgoStage14c_SilverTone
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
    const film::FilmProfile& profile
) noexcept
{
    const AlgoType tone = static_cast<AlgoType>(profile.silver_tone);

    // Colour stocks form dyes rather than a retained silver image, so there is no
    // silver left to be non-neutral. The copy is required by the retained-buffer
    // policy, not optional.
    if ((false == profile.is_monochrome) || (tone == ALGO_ZERO))
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    // Coefficients folded with the tone once, so the pixel loop is two multiplies
    // and two fused adds.
    const AlgoType kR = ALGO_SILVER_TONE_RED  * tone;
    const AlgoType kB = ALGO_SILVER_TONE_BLUE * tone;

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        const AlgoType* RESTRICT pR = pSrcR + off;
        const AlgoType* RESTRICT pG = pSrcG + off;
        const AlgoType* RESTRICT pB = pSrcB + off;

        AlgoType* RESTRICT pOR = pDstR + off;
        AlgoType* RESTRICT pOG = pDstG + off;
        AlgoType* RESTRICT pOB = pDstB + off;

        ALGO_VECTOR_HINT
        for (int32_t x = 0; x < sizeX; x++)
        {
            // The weight is the OUTPUT LEVEL: bright means least silver, which means
            // warmest. Weighted rather than flat because the effect fades as
            // particles overlap, and a flat tint would warm the shadows as much as
            // the highlights - the opposite of how a warm-toned print looks.
            const AlgoType w = pG[x];

            // Red up and blue down for a positive tone, by unequal amounts:
            // scattering is stronger at the blue end, so the blue side moves less
            // for the same physical cause. Equal coefficients would read as a plain
            // hue rotation instead of a toned print.
            pOR[x] = pR[x] * (ALGO_ONE + kR * w);
            pOG[x] = pG[x];
            pOB[x] = pB[x] * (ALGO_ONE - kB * w);
        }
    }

    return;
}
