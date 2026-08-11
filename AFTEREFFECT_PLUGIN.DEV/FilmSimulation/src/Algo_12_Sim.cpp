#if 0
// ---------------------------------------------------------------------------
//  Algo_12_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the 3x3 dye-impurity mix and the non-negative floor. Both pointwise.
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
//  Pipeline stage 12, in the density domain:
//
//      AlgoIsIdentityMatrix      identity test, shared with stage 13
//      AlgoApplyDensityMatrix    the 3x3 mix, shared with stage 13
//      AlgoStage12_DyeImpurity   the stage itself
//
//  Raw pointers, explicit geometry, no allocation, no mutable state, no validation
//  of inputs.
// ---------------------------------------------------------------------------

#include "AlgoDyeImpurity.hpp"

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

#include <cmath>   // std::fabs


// ---------------------------------------------------------------------------
//  Identity test
// ---------------------------------------------------------------------------
bool AlgoIsIdentityMatrix (const film::Matrix3& m) noexcept
{
    // Written as a loop rather than nine explicit comparisons, so there is no
    // opportunity to transpose an index by hand.
    for (int32_t i = 0; i < 3; i++)
    {
        for (int32_t j = 0; j < 3; j++)
        {
            // Expected value: one on the diagonal, zero off it.
            const AlgoType expected = (i == j) ? ALGO_ONE : ALGO_ZERO;
            const AlgoType actual   = static_cast<AlgoType>(m[i][j]);

            if (std::fabs(actual - expected) > ALGO_DYE_IDENTITY_EPS)
                return false;
        }
    }

    return true;
}


// ---------------------------------------------------------------------------
//  Apply a 3x3 density mixing matrix
// ---------------------------------------------------------------------------
void AlgoApplyDensityMatrix
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
    const film::Matrix3&     m
) noexcept
{
    // The nine coefficients are hoisted into named frame constants for two reasons:
    // the matrix is stored as float while the arithmetic is AlgoType, so converting
    // once removes nine conversions from the inner loop; and the compiler cannot
    // prove the matrix is unchanged by the stores into the destination, so it would
    // otherwise reload all nine on every pixel.
    //
    // Index convention m[out][in]. Row 0 forms the red output; within it column 1
    // is the contribution of the incoming green density.
    const AlgoType m00 = static_cast<AlgoType>(m[0][0]);   // red   from red
    const AlgoType m01 = static_cast<AlgoType>(m[0][1]);   // red   from green
    const AlgoType m02 = static_cast<AlgoType>(m[0][2]);   // red   from blue
    const AlgoType m10 = static_cast<AlgoType>(m[1][0]);   // green from red
    const AlgoType m11 = static_cast<AlgoType>(m[1][1]);   // green from green
    const AlgoType m12 = static_cast<AlgoType>(m[1][2]);   // green from blue
    const AlgoType m20 = static_cast<AlgoType>(m[2][0]);   // blue  from red
    const AlgoType m21 = static_cast<AlgoType>(m[2][1]);   // blue  from green
    const AlgoType m22 = static_cast<AlgoType>(m[2][2]);   // blue  from blue

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        // All six rows at once, because the mix couples the three channels: unlike
        // most stages this cannot be written as three independent passes.
        const AlgoType* RESTRICT pR = pSrcR + off;
        const AlgoType* RESTRICT pG = pSrcG + off;
        const AlgoType* RESTRICT pB = pSrcB + off;

        AlgoType* RESTRICT pOR = pDstR + off;
        AlgoType* RESTRICT pOG = pDstG + off;
        AlgoType* RESTRICT pOB = pDstB + off;

        // All three inputs are read into registers BEFORE any output is formed, so
        // the body stays correct if a caller ever passes the same planes for source
        // and destination - which stage 13 is entitled to want.
        const __m256 v00 = _mm256_set1_ps(m00);
        const __m256 v01 = _mm256_set1_ps(m01);
        const __m256 v02 = _mm256_set1_ps(m02);
        const __m256 v10 = _mm256_set1_ps(m10);
        const __m256 v11 = _mm256_set1_ps(m11);
        const __m256 v12 = _mm256_set1_ps(m12);
        const __m256 v20 = _mm256_set1_ps(m20);
        const __m256 v21 = _mm256_set1_ps(m21);
        const __m256 v22 = _mm256_set1_ps(m22);

        const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
        const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
        const __m256i mt = algoTailMaskLocal(nt);

        int32_t x = 0;

        for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
        {
            const __m256 inR = _mm256_loadu_ps(pR + x);
            const __m256 inG = _mm256_loadu_ps(pG + x);
            const __m256 inB = _mm256_loadu_ps(pB + x);

            __m256 oR = _mm256_mul_ps(inR, v00);
            oR = _mm256_fmadd_ps(inG, v01, oR);
            oR = _mm256_fmadd_ps(inB, v02, oR);

            __m256 oG = _mm256_mul_ps(inR, v10);
            oG = _mm256_fmadd_ps(inG, v11, oG);
            oG = _mm256_fmadd_ps(inB, v12, oG);

            __m256 oB = _mm256_mul_ps(inR, v20);
            oB = _mm256_fmadd_ps(inG, v21, oB);
            oB = _mm256_fmadd_ps(inB, v22, oB);

            _mm256_storeu_ps(pOR + x, oR);
            _mm256_storeu_ps(pOG + x, oG);
            _mm256_storeu_ps(pOB + x, oB);
        }

        if (nt > 0)
        {
            const __m256 inR = _mm256_maskload_ps(pR + x, mt);
            const __m256 inG = _mm256_maskload_ps(pG + x, mt);
            const __m256 inB = _mm256_maskload_ps(pB + x, mt);

            __m256 oR = _mm256_mul_ps(inR, v00);
            oR = _mm256_fmadd_ps(inG, v01, oR);
            oR = _mm256_fmadd_ps(inB, v02, oR);

            __m256 oG = _mm256_mul_ps(inR, v10);
            oG = _mm256_fmadd_ps(inG, v11, oG);
            oG = _mm256_fmadd_ps(inB, v12, oG);

            __m256 oB = _mm256_mul_ps(inR, v20);
            oB = _mm256_fmadd_ps(inG, v21, oB);
            oB = _mm256_fmadd_ps(inB, v22, oB);

            _mm256_maskstore_ps(pOR + x, mt, oR);
            _mm256_maskstore_ps(pOG + x, mt, oG);
            _mm256_maskstore_ps(pOB + x, mt, oB);
        }
    }

    return;
}


// ---------------------------------------------------------------------------
//  Stage 12: dye impurity and scanner crosstalk
// ---------------------------------------------------------------------------
void AlgoStage12_DyeImpurity
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
    const film::Matrix3& m = profile.dye_matrix;

    // Identity fast path. The data must still be COPIED rather than the stage
    // skipped: the retained-buffer policy gives this stage its own destination, and
    // leaving it unwritten would put stale contents in the chain.
    if (AlgoIsIdentityMatrix(m))
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    AlgoApplyDensityMatrix(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB,
                           sizeX, sizeY, pitch, m);

    // ----------------------------------------------------------------------
    //  Floor at zero.
    //
    //  A dye matrix can carry small NEGATIVE off-diagonal terms - a masking
    //  correction, which is a real feature of a masked colour negative - so the mix
    //  can drive a channel below zero where the other two are much denser.
    //  Negative optical density has no physical meaning, and stage 14 raises ten to
    //  its negative. A physical floor, not a display clamp.
    // ----------------------------------------------------------------------
    AlgoType* RESTRICT dstPlane[3] = { pDstR, pDstG, pDstB };

    for (int32_t c = 0; c < 3; c++)
    {
        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow =
                dstPlane[c] + static_cast<std::ptrdiff_t>(y) * pitch;

            // Non-negative floor. A max, not a branch: a negative optical density
            // would be a material that emits light, and the stages downstream take
            // its logarithm or its square root.
            {
                const __m256 vZero = _mm256_setzero_ps();

                const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
                const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
                const __m256i mt = algoTailMaskLocal(nt);

                int32_t x = 0;

                for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
                    _mm256_storeu_ps(pRow + x,
                        _mm256_max_ps(_mm256_loadu_ps(pRow + x), vZero));

                if (nt > 0)
                    _mm256_maskstore_ps(pRow + x, mt,
                        _mm256_max_ps(_mm256_maskload_ps(pRow + x, mt), vZero));
            }
        }
    }

    return;
}
#endif