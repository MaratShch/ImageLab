// ---------------------------------------------------------------------------
//  Algo_17_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the single final clamp and the narrowing to storage type. Pointwise and
//  bandwidth-bound, so the gain is small - but it touches every output sample and is
//  cheap to do.
//
//  ALIGNMENT: EVERY IMAGE ACCESS IS UNALIGNED, DELIBERATELY.
//
//  loadu/storeu on all plane data. The arena base comes from the host's pool, whose
//  alignment argument is a HINT - it was seen returning a base 16 mod 32, which faults
//  an aligned 256-bit load. AlgoMemHandler.cpp is SHARED by both flavours and must not
//  carry a vector-path concern, so the vector path assumes nothing about alignment.
//
//  Pipeline stage 17: the single final clamp, and the narrowing back to storage
//  type.
//
//  The last stage in the chain. Everything upstream deliberately left its output
//  unclamped at the top so the characteristic curve's shoulder had real highlight
//  information to roll off; this is the one place the display range is imposed.
//
//  Raw pointers, explicit geometry, no allocation, no mutable state, no validation
//  of inputs.
// ---------------------------------------------------------------------------

// Common.hpp -- AVX2_ALIGN / CACHE_ALIGN are defined here. Included
// DIRECTLY rather than relied on transitively: this file declares an
// aligned buffer, so the macro must not depend on another header's
// include order to be in scope.
#include "Common.hpp"
#include "AlgoFinalClamp.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>


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
//  Stage 17: final clamp, and narrow to the storage planes
// ---------------------------------------------------------------------------
void AlgoStage17_FinalClamp
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pStageR,
    AlgoType* RESTRICT       pStageG,
    AlgoType* RESTRICT       pStageB,
    ImgType* RESTRICT        pDstR,
    ImgType* RESTRICT        pDstG,
    ImgType* RESTRICT        pDstB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch
) noexcept
{
    const AlgoType* RESTRICT srcPlane  [3] = { pSrcR,   pSrcG,   pSrcB   };
    AlgoType* RESTRICT       stagePlane[3] = { pStageR, pStageG, pStageB };
    ImgType* RESTRICT        dstPlane  [3] = { pDstR,   pDstG,   pDstB   };

    for (int32_t c = 0; c < 3; c++)
    {
        const AlgoType* RESTRICT pIn    = srcPlane  [c];
        AlgoType* RESTRICT       pStage = stagePlane[c];
        ImgType* RESTRICT        pOut   = dstPlane  [c];

        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT rIn    = pIn    + off;
            AlgoType* RESTRICT       rStage = pStage + off;
            ImgType* RESTRICT        rOut   = pOut   + off;

            // THE clamp, eight at a time. Zero to one, once, where the numbers stop
            // being physical quantities and become display values.
            //
            // min then max, in that order: it matches CLAMP_VALUE's ordering and, unlike
            // a branch pair, it maps a NaN to the low bound rather than letting it reach
            // the host.
            {
                const __m256 vLo = _mm256_setzero_ps();
                const __m256 vHi = _mm256_set1_ps(ALGO_ONE);

                const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
                const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
                const __m256i mt = algoTailMaskLocal(nt);

                int32_t xv = 0;

                for (int32_t v = 0; v < nv; v++, xv += ALGO_AVX2_LANES_LOCAL)
                {
                    const __m256 cv = _mm256_max_ps(
                        _mm256_min_ps(_mm256_loadu_ps(rIn + xv), vHi), vLo);

                    // Retained in AlgoType as well, so the clamped result can be
                    // inspected without re-reading the narrowed storage planes.
                    _mm256_storeu_ps(rStage + xv, cv);

                    // The narrowing to storage type. ImgType and AlgoType are the same
                    // 32-bit float in this build, so this store IS the cast.
                    _mm256_storeu_ps(rOut + xv, cv);
                }

                if (nt > 0)
                {
                    const __m256 cv = _mm256_max_ps(
                        _mm256_min_ps(_mm256_maskload_ps(rIn + xv, mt), vHi), vLo);

                    _mm256_maskstore_ps(rStage + xv, mt, cv);
                    _mm256_maskstore_ps(rOut   + xv, mt, cv);
                }
            }

            // Reference expression, never executed.
            for (int32_t x = 0; x < 0; x++)
            {
                // THE clamp. Zero to one, once, at the point where the numbers stop
                // being physical quantities and become display values.
                //
                // CLAMP_VALUE rather than two nested calls, so the ordering of the
                // bounds is stated once and cannot be written backwards.
                const AlgoType v = CLAMP_VALUE(rIn[x], ALGO_ZERO, ALGO_ONE);

                // Retained in AlgoType as well, so the clamped result can be
                // inspected without re-reading the narrowed storage planes - the same
                // debugging convenience every other stage in the chain provides.
                rStage[x] = v;

                // The narrowing back to storage type. The matching widening happened
                // once, in stage 2, at the engine's input boundary; between the two
                // everything is AlgoType. When ImgType and AlgoType are the same type
                // this cast is a no-op and the compiler removes it.
                rOut[x] = static_cast<ImgType>(v);
            }
        }
    }

    return;
}
