// ---------------------------------------------------------------------------
//  Algo_02_Sim.cpp   --   AVX2
//
//  Pipeline stage 2 and its sub-stage 2b:
//
//      AlgoStage02_RelativeExposure   mid-grey normalisation and exposure offset
//      AlgoStage02b_TakingFilters     the camera taking matrix
//
//  Same filename, same function names, same prototypes as the scalar build. This
//  translation unit is a drop-in replacement compiled from the AVX2 sub-folder;
//  nothing outside it needs to know which one was linked.
//
//  ALL ARITHMETIC IS FLOAT32. The scalar path remains the reference.
//
//  Raw pointers, explicit geometry, no allocation, no mutable state, no validation
//  of inputs.
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
#include "AlgoRelativeExposure.hpp"
#include "AlgoTakingFilters.hpp"
#include "AlgoSeparableBlur.hpp"   // AlgoCopyImage, for the identity fast path

#include <immintrin.h>
#include <cmath>   // std::pow and std::fabs, both once per frame


// ---------------------------------------------------------------------------
//  The AVX2 path requires a 32-bit sample type.
//
//  Checked rather than assumed. If a build ever sets AlgoType to double while
//  linking these objects, every load below would read half as many pixels as the
//  loop believes it has and walk off the end of the plane - a silent heap overrun
//  rather than a wrong image. A compile error is a much better outcome.
// ---------------------------------------------------------------------------
static_assert(sizeof(AlgoType) == 4,
              "the AVX2 path requires AlgoType to be a 32-bit float");
static_assert(sizeof(ImgType) == 4,
              "the AVX2 path requires ImgType to be a 32-bit float");


namespace
{
    // ----------------------------------------------------------------------
    //  Lanes in one AVX2 vector of float: 256 bits / 32 bits.
    // ----------------------------------------------------------------------
    constexpr int32_t ALGO_AVX2_LANES = 8;


    // ----------------------------------------------------------------------
    //  Is this matrix the identity, to within single-precision noise?
    //
    //  A VERBATIM copy of the scalar predicate, deliberately. 141 of the 142 stocks
    //  take the identity fast path, so if the two builds ever disagreed about which
    //  matrices qualify they would diverge on exactly one stock - and that would
    //  present as a vectorisation bug rather than as the predicate mismatch it
    //  really was.
    //
    //  Written as a loop rather than nine explicit comparisons so there is no
    //  opportunity to transpose an index by hand.
    // ----------------------------------------------------------------------
    inline bool isIdentityMatrix (const film::Matrix3& m) noexcept
    {
        for (int32_t i = 0; i < 3; i++)
        {
            for (int32_t j = 0; j < 3; j++)
            {
                // Expected value: 1.0 on the diagonal, 0.0 off it.
                const AlgoType expected = (i == j) ? ALGO_ONE : ALGO_ZERO;
                const AlgoType actual   = static_cast<AlgoType>(m[i][j]);

                if (std::fabs(actual - expected) > ALGO_TAKING_IDENTITY_EPS)
                    return false;
            }
        }
        return true;
    }


    // ----------------------------------------------------------------------
    //  Tail mask for the final, partial vector of a row.
    //
    //  RULE: the active width is NOT generally a multiple of eight. 1998, 1023 and
    //  2816 all appear in the test set, and only one of those divides evenly.
    //
    //  The mask has all bits set in lanes 0..n-1 and zero above, so a masked load
    //  reads only the active samples and a masked store writes only those. The
    //  padded row IS wide enough to allow an unmasked overspill - padW is a
    //  multiple of eight by construction - but masking is used deliberately anyway,
    //  for two reasons:
    //
    //    - The padding stays untouched, so the NaN-poison arena test remains
    //      meaningful. An unmasked tail would fill padding with values derived from
    //      poison and any later reduction reading past sizeX would report NaN.
    //    - A masked store cannot fault, so this is correct even if a future
    //      allocator stops padding rows at all.
    //
    //  Built from a static table rather than computed, because the alternative -
    //  comparing a broadcast count against an index vector - costs two instructions
    //  per row for a value that only ever takes eight forms.
    // ----------------------------------------------------------------------
    inline __m256i algoTailMask (const int32_t n) noexcept
    {
        // Lane i is active when i < n. Only 1..7 are ever requested; 0 would mean
        // there is no tail and the caller skips this entirely.
        AVX2_ALIGN static const int32_t table[ALGO_AVX2_LANES][ALGO_AVX2_LANES] =
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

        return _mm256_load_si256(
            reinterpret_cast<const __m256i*>(&table[n & (ALGO_AVX2_LANES - 1)][0]));
    }
}


// ---------------------------------------------------------------------------
//  Stage 2: relative exposure
// ---------------------------------------------------------------------------
void AlgoStage02_RelativeExposure
(
    const ImgType* RESTRICT pSrcR,
    const ImgType* RESTRICT pSrcG,
    const ImgType* RESTRICT pSrcB,
    AlgoType* RESTRICT      pDstR,
    AlgoType* RESTRICT      pDstG,
    AlgoType* RESTRICT      pDstB,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           pitch,
    const AlgoControls&     params
) noexcept
{
    // ----------------------------------------------------------------------
    //  One frame constant, exactly as the scalar path forms it.
    //
    //      gain = 2^exposureStops / ALGO_MID_GREY
    //
    //  Computed in double and narrowed once. std::pow runs once per frame, so its
    //  cost is irrelevant and its accuracy is free - and deriving it identically to
    //  the scalar build is what keeps the two comparable. Any difference between
    //  the paths must come from the pixel loop, not from the setup.
    // ----------------------------------------------------------------------
    const AlgoType exposureGain =
        static_cast<AlgoType>(std::pow(2.0, params.exposureStops));

    const AlgoType gain = exposureGain / ALGO_MID_GREY;

    const __m256 vGain = _mm256_set1_ps(gain);

    // Whole vectors per row, and how many samples are left over.
    const int32_t vecCount = sizeX / ALGO_AVX2_LANES;
    const int32_t tailN    = sizeX - (vecCount * ALGO_AVX2_LANES);

    const __m256i vTail = algoTailMask(tailN);

    // ----------------------------------------------------------------------
    //  Three planes in ONE row pass rather than three separate frame passes.
    //
    //  The scalar path also does this, but it matters more here: the three planes
    //  are independent, so interleaving them at row granularity gives the
    //  out-of-order engine three unrelated multiply chains to overlap while any one
    //  of them waits on memory. It also touches three rows that will be needed
    //  together rather than sweeping the whole frame three times.
    // ----------------------------------------------------------------------
    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        const ImgType* RESTRICT pInR = pSrcR + off;
        const ImgType* RESTRICT pInG = pSrcG + off;
        const ImgType* RESTRICT pInB = pSrcB + off;

        AlgoType* RESTRICT pOutR = pDstR + off;
        AlgoType* RESTRICT pOutG = pDstG + off;
        AlgoType* RESTRICT pOutB = pDstB + off;

        int32_t x = 0;

        for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
        {
            // Aligned loads and stores. Every plane base is 32-byte aligned and
            // every row stride in bytes is a multiple of 32, both guaranteed by the
            // arena, so x being a multiple of eight puts these on a boundary.
            const __m256 r = _mm256_loadu_ps(pInR + x);
            const __m256 g = _mm256_loadu_ps(pInG + x);
            const __m256 b = _mm256_loadu_ps(pInB + x);

            _mm256_storeu_ps(pOutR + x, _mm256_mul_ps(r, vGain));
            _mm256_storeu_ps(pOutG + x, _mm256_mul_ps(g, vGain));
            _mm256_storeu_ps(pOutB + x, _mm256_mul_ps(b, vGain));
        }

        if (tailN > 0)
        {
            const __m256 r = _mm256_maskload_ps(pInR + x, vTail);
            const __m256 g = _mm256_maskload_ps(pInG + x, vTail);
            const __m256 b = _mm256_maskload_ps(pInB + x, vTail);

            _mm256_maskstore_ps(pOutR + x, vTail, _mm256_mul_ps(r, vGain));
            _mm256_maskstore_ps(pOutG + x, vTail, _mm256_mul_ps(g, vGain));
            _mm256_maskstore_ps(pOutB + x, vTail, _mm256_mul_ps(b, vGain));
        }
    }

    return;
}


// ---------------------------------------------------------------------------
//  Stage 2b: camera taking filters
// ---------------------------------------------------------------------------
void AlgoStage02b_TakingFilters
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
    const film::Matrix3& m = profile.taking_matrix;

    // ----------------------------------------------------------------------
    //  Identity fast path, and the test must match the scalar build EXACTLY.
    //
    //  141 of the 142 stocks have an identity taking matrix, so this is the common
    //  case rather than an optimisation for a corner. The data is COPIED rather
    //  than the pass skipped: every stage owns its destination and leaving it
    //  unwritten would put stale contents into the chain.
    //
    //  The predicate is deliberately the scalar one, unchanged. If the two builds
    //  ever disagreed about which matrices count as identity, they would diverge on
    //  exactly one stock and the difference would look like a vectorisation bug.
    // ----------------------------------------------------------------------
    if (isIdentityMatrix(m))
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    // Nine coefficients, broadcast once per frame. The matrix is stored as float
    // and the arithmetic is float, so this is a widen-free broadcast.
    //
    // Index convention is m[out][in]: row 0 forms the red record, and column 1
    // within it is the contribution of the incoming green signal.
    const __m256 m00 = _mm256_set1_ps(static_cast<float>(m[0][0]));
    const __m256 m01 = _mm256_set1_ps(static_cast<float>(m[0][1]));
    const __m256 m02 = _mm256_set1_ps(static_cast<float>(m[0][2]));
    const __m256 m10 = _mm256_set1_ps(static_cast<float>(m[1][0]));
    const __m256 m11 = _mm256_set1_ps(static_cast<float>(m[1][1]));
    const __m256 m12 = _mm256_set1_ps(static_cast<float>(m[1][2]));
    const __m256 m20 = _mm256_set1_ps(static_cast<float>(m[2][0]));
    const __m256 m21 = _mm256_set1_ps(static_cast<float>(m[2][1]));
    const __m256 m22 = _mm256_set1_ps(static_cast<float>(m[2][2]));

    const int32_t vecCount = sizeX / ALGO_AVX2_LANES;
    const int32_t tailN    = sizeX - (vecCount * ALGO_AVX2_LANES);

    const __m256i vTail = algoTailMask(tailN);

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        const AlgoType* RESTRICT pInR = pSrcR + off;
        const AlgoType* RESTRICT pInG = pSrcG + off;
        const AlgoType* RESTRICT pInB = pSrcB + off;

        AlgoType* RESTRICT pOutR = pDstR + off;
        AlgoType* RESTRICT pOutG = pDstG + off;
        AlgoType* RESTRICT pOutB = pDstB + off;

        int32_t x = 0;

        for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
        {
            // All three inputs are read into registers BEFORE any output is formed.
            // Source and destination are distinct planes here - ping/pong
            // guarantees it - but the mix couples the channels, so reading first
            // states the dependency plainly and survives a future in-place variant.
            const __m256 r = _mm256_loadu_ps(pInR + x);
            const __m256 g = _mm256_loadu_ps(pInG + x);
            const __m256 b = _mm256_loadu_ps(pInB + x);

            // Three fused multiply-adds per output channel. FMA rounds once for the
            // whole multiply-add instead of twice, so this is very slightly MORE
            // accurate than the scalar expression, not less - one of the few places
            // where the vector path legitimately beats its reference.
            __m256 outR = _mm256_mul_ps(r, m00);
            outR = _mm256_fmadd_ps(g, m01, outR);
            outR = _mm256_fmadd_ps(b, m02, outR);

            __m256 outG = _mm256_mul_ps(r, m10);
            outG = _mm256_fmadd_ps(g, m11, outG);
            outG = _mm256_fmadd_ps(b, m12, outG);

            __m256 outB = _mm256_mul_ps(r, m20);
            outB = _mm256_fmadd_ps(g, m21, outB);
            outB = _mm256_fmadd_ps(b, m22, outB);

            _mm256_storeu_ps(pOutR + x, outR);
            _mm256_storeu_ps(pOutG + x, outG);
            _mm256_storeu_ps(pOutB + x, outB);
        }

        if (tailN > 0)
        {
            const __m256 r = _mm256_maskload_ps(pInR + x, vTail);
            const __m256 g = _mm256_maskload_ps(pInG + x, vTail);
            const __m256 b = _mm256_maskload_ps(pInB + x, vTail);

            __m256 outR = _mm256_mul_ps(r, m00);
            outR = _mm256_fmadd_ps(g, m01, outR);
            outR = _mm256_fmadd_ps(b, m02, outR);

            __m256 outG = _mm256_mul_ps(r, m10);
            outG = _mm256_fmadd_ps(g, m11, outG);
            outG = _mm256_fmadd_ps(b, m12, outG);

            __m256 outB = _mm256_mul_ps(r, m20);
            outB = _mm256_fmadd_ps(g, m21, outB);
            outB = _mm256_fmadd_ps(b, m22, outB);

            _mm256_maskstore_ps(pOutR + x, vTail, outR);
            _mm256_maskstore_ps(pOutG + x, vTail, outG);
            _mm256_maskstore_ps(pOutB + x, vTail, outB);
        }
    }

    return;
}
