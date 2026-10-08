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

// Common.hpp -- AVX2_ALIGN / CACHE_ALIGN are defined here. Included
// DIRECTLY rather than relied on transitively: this file declares an
// aligned buffer, so the macro must not depend on another header's
// include order to be in scope.
#include "Common.hpp"
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

    // sqrt(ln(2)/2) / pi = 0.1873906251292776. Derived by equating the two exponent
    // forms: the MTF is exp(-ln2 (f/f50)^2) and a Gaussian blur of sigma s
    // millimetres has transfer exp(-2 pi^2 s^2 f^2), so ln2/f50^2 = 2 pi^2 s^2.
    //
    // Written as a literal for the same reason it is in the emulsion MTF header:
    // it is then a compile-time constant on every compiler, and the derivation can
    // be checked against the digits by hand.
    // ⚠⚠ THE DIGITS WERE WRONG UNTIL SCHEMA v48 AND THE COMMENT ABOVE INVITED THE
    // CHECK THAT WOULD HAVE CAUGHT THEM. sqrt(0.6931471805599453 / 2) is
    // 0.5887050112577373, and dividing THAT by pi gives 0.1873906251292776 -- not
    // 0.18738564618678, which is what shipped. Wrong from the FIFTH digit,
    // 2.66e-05 relative, and the derivation printed beside it has been correct
    // the whole time.
    //
    // ⚠ IT HID BECAUSE THE REFERENCE ENGINE NEVER COMPUTES THIS SIGMA. Python
    // evaluates exp(-ln2 (f/f50)^2) straight onto its frequency grid, so this
    // constant existed only on the C++ side and had nothing to disagree with. A
    // quantity computed in one engine and not in the other is not covered by
    // parity testing, however thorough that testing is -- which is the general
    // lesson, and it is not about grain.
    //
    // The correct value was already in film_profiles twice, under
    // GRAIN_SIGMA_PER_HALF_POWER and _TAGUCHI_F50_TO_SIGMA_UM / 1000, and v48 adds
    // film_profiles.scan_sigma_mm() as a Python consumer so that the next
    // disagreement is a parity failure instead of a secret. G-V48-KSIGMA pins it.
    return static_cast<AlgoType>(0.1873906251292776) / f50CyclesPerMm;
}


// The sub-pixel misregistration shift was a Catmull-Rom resample here until
// 2026-10-06. It is now film_sim's exact phase ramp, FreqGrid.shift(dy, dx),
// multiplied into the scan transfer and applied by the frequency-domain filter
// in the same pass as the scan MTF (AlgoFrequency.hpp) -- a translation with no
// low-pass side effect at all, which is what the resampler was approximating.


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
    const uint32_t           seed,
    const AlgoFreqState&     freq
) noexcept
{
    // Scratch planes of the separable form, unused since 2026-10-06.
    (void)pScrA;
    (void)pScrB;

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

        // ------------------------------------------------------------------
        //  film_sim:  t = grid.mtf(scan_f50, 0, 0)
        //             if mis_px > 0 and not monochrome: t *= grid.shift(dy, dx)
        //             dens[c] = apply_transfer(dens[c], t)
        //
        //  One exact frequency-domain pass. Until 2026-10-06 this was a
        //  separable Gaussian (skipped below 0.25 px) followed by a Catmull-Rom
        //  resample (skipped below a minimum shift); film_sim applies both
        //  for every frame, and now so does the engine.
        // ------------------------------------------------------------------
        {
            AlgoFreqTransfer t;
            AlgoFreqSetMtf(t, static_cast<HighPrecType>(scanF50), false, 0.0, 0.0, 0.0);
            if (wantShift)
                AlgoFreqAddShift(t, dy, dx);

            if (AlgoFreqIsIdentity(t))
                AlgoCopyPlane(pIn, pOut, sizeX, sizeY, pitch);
            else
                AlgoFreqFilterPlane(freq, pIn, pOut, pitch, t);
        }

        // ------------------------------------------------------------------
        //  Floor at zero (film_sim: np.maximum(dens, 0) after stage 10).
        //
        //  The Gaussian transfer alone cannot take non-negative input below
        //  zero, but the sub-pixel phase ramp is a band-limited interpolation
        //  and rings at a hard edge, so the floor is load-bearing.
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
