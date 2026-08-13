// ---------------------------------------------------------------------------
//  Algo_07_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the monochrome collapse. The reseau path is a per-pixel filter-index
//  lookup - a data-dependent select on a spatial mosaic pattern - and is left scalar:
//  it runs on one stock in the database and vectorising a per-pixel index would need
//  a blend tree deeper than the arithmetic it protects.
//
//  Pipeline stage 7, in exposure space:
//
//      AlgoReseauPitchPx           grid pitch in pixels for a stock and render
//      AlgoReseauFilterIndex       which filter covers a pixel
//      AlgoStage07_EmulsionRecord  collapse to the record the emulsion holds
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

#include "AlgoEmulsionRecord.hpp"

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
#include "AlgoSeparableBlur.hpp"   // AlgoCopyImage
#include "AlgoSpectralSensitivity.hpp"


// ---------------------------------------------------------------------------
//  Grid pitch in pixels
// ---------------------------------------------------------------------------
AlgoType AlgoReseauPitchPx
(
    const film::ReseauSpec& spec,
    const AlgoType          pxPerMm
) noexcept
{
    const AlgoType lpmm = static_cast<AlgoType>(spec.lines_per_mm);

    // A stock with no grid, or a render with no physical scale, has no pitch.
    // Zero is returned rather than infinity so the caller's comparison against
    // the minimum usable pitch does the right thing without a special case.
    if ((lpmm <= ALGO_ZERO) || (pxPerMm <= ALGO_ZERO))
        return ALGO_ZERO;

    // Pixels per millimetre divided by grid lines per millimetre gives pixels
    // per grid line, which is the cell pitch.
    return pxPerMm / lpmm;
}


// ---------------------------------------------------------------------------
//  Which filter covers a pixel
// ---------------------------------------------------------------------------
int32_t AlgoReseauFilterIndex
(
    const int32_t      x,
    const int32_t      y,
    const HighPrecType invPitch
) noexcept
{
    // Pixel coordinates to whole cell indices. Truncation towards zero is
    // correct here because both coordinates are non-negative throughout the
    // frame, so it coincides with the floor the reference takes.
    const int32_t cellY = static_cast<int32_t>(
        static_cast<HighPrecType>(y) * invPitch);

    const int32_t cellX = static_cast<int32_t>(
        static_cast<HighPrecType>(x) * invPitch);

    // Every third cell row is a continuous red line.
    const int32_t band = cellY % ALGO_RESEAU_BAND_PERIOD;

    if (0 == band)
        return 0;   // red

    // Between the red lines, blue and green squares alternate in a chequer.
    // Summing the two cell indices before taking the parity is what makes the
    // pattern a chequer rather than vertical stripes.
    const int32_t chequer = (cellX + cellY) & 1;

    return (0 == chequer) ? 2   // blue
                          : 1;  // green
}


// ---------------------------------------------------------------------------
//  Stage 7: collapse to the emulsion's own record
// ---------------------------------------------------------------------------
void AlgoStage07_EmulsionRecord
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
    const AlgoType           pxPerMm
) noexcept
{
    // ----------------------------------------------------------------------
    //  Case 2: monochrome.
    //
    //  One silver image, formed with the stock's own spectral sensitivity.
    // ----------------------------------------------------------------------
    if (profile.is_monochrome)
    {
        // The stock's own response to each third of the spectrum. For an
        // orthochromatic emulsion the red weight is near zero, which is the
        // whole reason red renders black on early film.
        // MEASURED SPECTRAL PATH. The weight with which each input primary
        // reaches the single silver record is the stock's own pan sensitivity
        // integrated against that primary. Falls back to the authored triple
        // for stocks with no curve, so those render unchanged.
        //
        // NOTE: the authored triple is close to video luma (0.27/0.55/0.18),
        // which is exactly what it must not be; the derived triple is much
        // flatter for a panchromatic emulsion, which is why panchromatic film
        // renders a blue sky lighter than the eye sees it. This is a
        // correction, and it changes monochrome output visibly.
        AVX2_ALIGN AlgoType specW[3];

        if (!AlgoSpectralMonoWeights(profile, specW))
        {
            specW[0] = static_cast<AlgoType>(profile.spectral_weights[0]);
            specW[1] = static_cast<AlgoType>(profile.spectral_weights[1]);
            specW[2] = static_cast<AlgoType>(profile.spectral_weights[2]);
        }

        const AlgoType wR = specW[0];
        const AlgoType wG = specW[1];
        const AlgoType wB = specW[2];

        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pR = pSrcR + off;
            const AlgoType* RESTRICT pG = pSrcG + off;
            const AlgoType* RESTRICT pB = pSrcB + off;

            AlgoType* RESTRICT pOR = pDstR + off;
            AlgoType* RESTRICT pOG = pDstG + off;
            AlgoType* RESTRICT pOB = pDstB + off;

            // ALGO_VECTOR_HINT removed: it is a pragma that must sit immediately
            // before a loop, and the intrinsics below are explicit rather than
            // auto-vectorised, so there is nothing left for it to advise.
            // One record, replicated into all three planes. Replicated rather than
            // carried as a single plane so every stage after this one sees the same
            // three-plane layout and needs no monochrome special case of its own.
            const __m256 vwR = _mm256_set1_ps(wR);
            const __m256 vwG = _mm256_set1_ps(wG);
            const __m256 vwB = _mm256_set1_ps(wB);

            const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
            const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
            const __m256i mt = algoTailMaskLocal(nt);

            int32_t x = 0;

            for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
            {
                __m256 mono = _mm256_mul_ps(_mm256_loadu_ps(pR + x), vwR);
                mono = _mm256_fmadd_ps(_mm256_loadu_ps(pG + x), vwG, mono);
                mono = _mm256_fmadd_ps(_mm256_loadu_ps(pB + x), vwB, mono);

                _mm256_storeu_ps(pOR + x, mono);
                _mm256_storeu_ps(pOG + x, mono);
                _mm256_storeu_ps(pOB + x, mono);
            }

            if (nt > 0)
            {
                __m256 mono = _mm256_mul_ps(_mm256_maskload_ps(pR + x, mt), vwR);
                mono = _mm256_fmadd_ps(_mm256_maskload_ps(pG + x, mt), vwG, mono);
                mono = _mm256_fmadd_ps(_mm256_maskload_ps(pB + x, mt), vwB, mono);

                _mm256_maskstore_ps(pOR + x, mt, mono);
                _mm256_maskstore_ps(pOG + x, mt, mono);
                _mm256_maskstore_ps(pOB + x, mt, mono);
            }
        }

        return;
    }

    // ----------------------------------------------------------------------
    //  Case 3: additive colour screen.
    // ----------------------------------------------------------------------
    if (profile.has_reseau && params.reseau)
    {
        const film::ReseauSpec& spec = profile.reseau;

        const AlgoType pitchPx = AlgoReseauPitchPx(spec, pxPerMm);

        // The grid must be resolvable before it is worth building. Below the
        // floor the mask quantises unevenly and the result is aliasing noise
        // rather than a mosaic, so the stage falls through to the plain
        // three-record copy at the end instead of emitting garbage.
        if (pitchPx >= ALGO_RESEAU_MIN_PITCH_PX)
        {
            // Reciprocal formed once: the per-pixel cell lookup then costs a
            // multiply instead of a divide.
            const HighPrecType invPitch = 1.0 / static_cast<HighPrecType>(pitchPx);

            const film::Matrix3& fm = spec.filter_matrix;

            // ----------------------------------------------------------------
            //  Throughput of each filter for each incident band.
            //
            //  Row f is the filter, column j the incident band. The off-diagonal
            //  terms are large, and that overlap is what makes additive colour
            //  pastel rather than saturated.
            //
            //  Hoisted into named frame constants: the matrix is stored as float
            //  while the arithmetic is AlgoType, and the compiler cannot prove
            //  the matrix is unchanged by the stores into the destination, so it
            //  would otherwise reload and convert all nine values per pixel.
            // ----------------------------------------------------------------
            const AlgoType f00 = static_cast<AlgoType>(fm[0][0]);   // red   cell, red   light
            const AlgoType f01 = static_cast<AlgoType>(fm[0][1]);   // red   cell, green light
            const AlgoType f02 = static_cast<AlgoType>(fm[0][2]);   // red   cell, blue  light
            const AlgoType f10 = static_cast<AlgoType>(fm[1][0]);   // green cell, red   light
            const AlgoType f11 = static_cast<AlgoType>(fm[1][1]);   // green cell, green light
            const AlgoType f12 = static_cast<AlgoType>(fm[1][2]);   // green cell, blue  light
            const AlgoType f20 = static_cast<AlgoType>(fm[2][0]);   // blue  cell, red   light
            const AlgoType f21 = static_cast<AlgoType>(fm[2][1]);   // blue  cell, green light
            const AlgoType f22 = static_cast<AlgoType>(fm[2][2]);   // blue  cell, blue  light

            // ----------------------------------------------------------------
            //  Neutral gain: the mean row sum of the filter matrix.
            //
            //  Dividing by it restores the mean level the filters removed, so a
            //  neutral grey passes the grid unchanged and the scalar anchor
            //  solve at stage 8 - which cannot see the mask - stays valid.
            //
            //  Guarded against a degenerate matrix: a row-sum mean of zero would
            //  mean the grid passes no light at all, in which case leaving the
            //  record unscaled is the only defined behaviour.
            // ----------------------------------------------------------------
            const AlgoType rowSum = (f00 + f01 + f02)
                                  + (f10 + f11 + f12)
                                  + (f20 + f21 + f22);

            const AlgoType neutralGain = rowSum / static_cast<AlgoType>(3.0);

            const AlgoType invNeutral = (neutralGain > ALGO_ZERO)
                                      ? (ALGO_ONE / neutralGain)
                                      : ALGO_ONE;

            // Filter rows indexed by the mask value, so the inner loop selects a
            // row rather than branching three ways.
            const AlgoType fRow[3][3] =
            {
                { f00, f01, f02 },
                { f10, f11, f12 },
                { f20, f21, f22 }
            };

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
                    // Exactly one filter covers this pixel.
                    const int32_t f = AlgoReseauFilterIndex(x, y, invPitch);

                    // What that filter lets through, from all three incident
                    // bands. This is the single silver record at this point.
                    const AlgoType through = fRow[f][0] * pR[x]
                                           + fRow[f][1] * pG[x]
                                           + fRow[f][2] * pB[x];

                    const AlgoType record = through * invNeutral;

                    // Replicated into three planes for the same reason as the
                    // monochrome case: one uniform layout downstream.
                    pOR[x] = record;
                    pOG[x] = record;
                    pOB[x] = record;
                }
            }

            return;
        }
    }

    // ----------------------------------------------------------------------
    //  Case 1: tripack colour, and the fallback when a grid cannot be resolved.
    //
    //  Three independent records pass through. The data must be COPIED rather
    //  than the stage skipped: the retained-buffer policy gives this stage its
    //  own destination, and leaving it unwritten would put stale contents in the
    //  chain for every stage that follows.
    // ----------------------------------------------------------------------
    AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    return;
}
