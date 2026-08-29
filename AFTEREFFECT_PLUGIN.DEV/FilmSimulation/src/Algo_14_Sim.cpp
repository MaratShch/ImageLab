// ---------------------------------------------------------------------------
//  Algo_14_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: density to transmittance, which is 10^-d per sample per channel and the
//  second-largest exponential population in the engine after stage 8. Evaluated as
//  Exp(-d * ln10) because there is no vector pow - which is what a scalar pow(10, x)
//  does internally anyway.
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
        AlgoAddGrain(pDstR, pDstG, pDstB,
                     pScrField, pScrField, pScrField,
                     sizeX, sizeY, pitch, dmin, ALGO_GRAIN_DUPE_FOG, ALGO_ONE);
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

        const HighPrecType tMin = std::pow(10.0,
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

        AlgoType* RESTRICT pD = dstPlane[c];

        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow = pD + static_cast<std::ptrdiff_t>(y) * pitch;

            // 10^-d as Exp(-d * ln10). There is no vector pow, and the base change is
            // one multiply - which is what a scalar pow(10, x) does internally.
            //
            // ln(10) written to full double precision and narrowed by the compiler, so
            // the constant is the closest float to ln(10) rather than to a decimal.
            const __m256 vNegLn10 = _mm256_set1_ps(
                -static_cast<float>(2.30258509299404568401799145468436421));
            const __m256 vZeroL   = _mm256_setzero_ps();
            const __m256 vTMinA   = _mm256_set1_ps(tMinA);
            const __m256 vInvSpan = _mm256_set1_ps(invSpan);

            const int32_t nvL = sizeX / ALGO_AVX2_LANES_LOCAL;
            const int32_t ntL = sizeX - nvL * ALGO_AVX2_LANES_LOCAL;
            const __m256i mtL = algoTailMaskLocal(ntL);

            int32_t xv = 0;

            for (int32_t vv = 0; vv < nvL; vv++, xv += ALGO_AVX2_LANES_LOCAL)
            {
                // Density floored at zero before the exponentiation: a negative
                // density would give a transmittance above one - a material that
                // emits light.
                const __m256 dv =
                    _mm256_max_ps(_mm256_loadu_ps(pRow + xv), vZeroL);

                const __m256 tv =
                    FastCompute::AVX2::Exp(_mm256_mul_ps(dv, vNegLn10));

                // Normalised against the stock's OWN range, and deliberately NOT capped
                // at one: the single final clamp belongs to stage 17, and print grain
                // can legitimately push a highlight past the nominal white point.
                _mm256_storeu_ps(pRow + xv, _mm256_max_ps(
                    _mm256_mul_ps(_mm256_sub_ps(tv, vTMinA), vInvSpan), vZeroL));
            }

            if (ntL > 0)
            {
                const __m256 dv =
                    _mm256_max_ps(_mm256_maskload_ps(pRow + xv, mtL), vZeroL);

                const __m256 tv =
                    FastCompute::AVX2::Exp(_mm256_mul_ps(dv, vNegLn10));

                _mm256_maskstore_ps(pRow + xv, mtL, _mm256_max_ps(
                    _mm256_mul_ps(_mm256_sub_ps(tv, vTMinA), vInvSpan), vZeroL));
            }

            // Reference expression, retained beside the vector form it must agree with.
            // Never executes.
            for (int32_t x = 0; x < 0; x++)
            {
                const HighPrecType d = static_cast<HighPrecType>(
                    MAX_VALUE(pRow[x], ALGO_ZERO));

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
