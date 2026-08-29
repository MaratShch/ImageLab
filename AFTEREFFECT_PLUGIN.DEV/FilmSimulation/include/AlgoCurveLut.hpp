#pragma once

// ---------------------------------------------------------------------------
//  AlgoCurveLut.hpp -- the tabulated characteristic curve, SHARED by both
//  instruction-set paths and by every stage that evaluates a curve.
//
//  ---------------------------------------------------------------------------
//  WHY THIS FILE EXISTS, AND WHY IT IS A HEADER RATHER THAN FOUR COPIES
//  ---------------------------------------------------------------------------
//
//  The engine evaluates a difference-of-softplus characteristic curve in three
//  places: the negative curve at stage 08, the interimage re-evaluation at
//  stage 08b, and the PRINT curve at stage 13. Each of those existed in two
//  instruction-set variants, and the variants had drifted into using different
//  mathematics:
//
//      stage 08   scalar: exact log1p/exp     AVX2: 2048-entry table
//      stage 13   scalar: exact log1p/exp     AVX2: FastCompute::AVX2::Exp,
//                                                   a fast APPROXIMATE exp
//
//  Measured cost of that second row, per-stage means on a 128 px render with
//  damage and grain disabled so only the deterministic chain is compared:
//
//      stages 02 - 12   scalar vs AVX2 agree to 1e-07 - 1e-06   (float rounding)
//      stage 13         jumps to 2.4e-03 - 3.2e-03              (3000x)
//      stage 17 (out)   1.0e-03 - 1.8e-03
//
//  So the largest cross-path disagreement in the whole engine was not
//  precision. It was one stage using an approximate exponential while its twin
//  used an exact one -- the same defect the blur had, and the same defect
//  stage 08 had, in a third place.
//
//  Putting the table here, once, means every stage and both paths tabulate the
//  SAME function with the SAME domain and the SAME interpolation. Agreement is
//  then structural. It is not something to re-measure after every change,
//  because there is no longer a second copy that could drift.
//
//  ---------------------------------------------------------------------------
//  WHAT IS AND IS NOT PATH-DEPENDENT
//  ---------------------------------------------------------------------------
//
//  Entries are AlgoType, so they are double in the scalar build and float in
//  the vector build. That is the precision rule working as intended: the scalar
//  path stays the double reference. It is NOT a divergence, because the
//  tabulated function, its domain and its interpolation are identical -- only
//  the storage width differs, exactly as it does for every image plane.
//
//  The float storage is also what lets the AVX2 path read two entries with a
//  pair of _mm256_i32gather_ps. The vector lookup itself stays in the AVX2
//  translation units; only the table and its construction live here.
//
//  ---------------------------------------------------------------------------
//  DOMAIN, PADDING AND SIZE -- MEASURED, NOT ASSUMED (2026-08-28)
//  ---------------------------------------------------------------------------
//
//  Both ends of the curve are asymptotically FLAT: below the toe both ramps
//  vanish and the density is dmin; above the shoulder both ramps are linear
//  with unit slope and their difference is the constant (shoulder_x - toe_x).
//  Clamping the argument to the table domain is therefore EXACT rather than
//  approximate -- provided the domain actually covers the transition.
//
//  A softplus is within k*exp(-P) of its asymptote at P knees of padding, and
//  that residual is multiplied by gamma, so the clamp error is about
//  gamma*k*exp(-P). The original ten knees were not enough, and the shortfall
//  was the DOMINANT error in the table -- larger than the interpolation the
//  size was chosen to control:
//
//      P = 10    predicted 1.27e-04 D    measured 1.259e-04 D
//      P = 16    predicted 3.15e-07 D    measured 3.122e-07 D
//
//  Widening the domain widens the step, and interpolation error grows as its
//  square, so the table doubles from 2048 to 4096 to take that back fourfold.
//  Measured across all 161 stocks, 483 curves:
//
//      2048 / 10 knees   interp 8.05e-06 D   clamp 1.26e-04 D
//      4096 / 16 knees   interp 4.24e-06 D   clamp 3.12e-07 D
//
//  Better on BOTH axes than the pair it replaces.
//
//  IS 4.24e-06 D SMALL ENOUGH? The anchor that settles it: the exact
//  difference-of-softplus model is ITSELF non-monotonic by up to 6.61e-06 D at
//  the extreme toe on some stocks, where shoulder_k > toe_k makes the shoulder
//  ramp decay more slowly than the toe ramp. The table's worst error is smaller
//  than the model's own internal inconsistency, and three orders below the
//  ~1e-03 D at which the source datasheet curves can be read at all. Chasing it
//  lower would mean a 16384-entry table -- 393 KB of stack across three
//  channels -- to gain accuracy no input datum carries.
//
//  ---------------------------------------------------------------------------
//  MONOTONICITY
//  ---------------------------------------------------------------------------
//
//  A characteristic curve that folds back solarises a highlight. Linear
//  interpolation between samples of a monotonic function is monotonic, and the
//  entries are computed from the exact curve, so the property holds by
//  construction. test_curve_lut.cpp asserts it across all 161 stocks and 483
//  curves by comparing the table's worst backward step against the exact
//  curve's own -- they are equal to the digit, so the table introduces none.
//
//  ---------------------------------------------------------------------------
//  ALLOCATION AND REENTRANCY
//  ---------------------------------------------------------------------------
//
//  Every table is declared by its caller, on the stack. No static, no heap, no
//  shared mutable state, so the engine's guarantee of arbitrary concurrent
//  frames in any order survives untouched. Three tables is 98 KB in the scalar
//  build and 49 KB in the vector one, live only for the stage that built them.
// ---------------------------------------------------------------------------

#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoTypes.hpp"
#include "AlgoHalation.hpp"      // ALGO_SOFTPLUS_LINEAR_LIMIT
#include "film_profiles.hpp"     // film::ToneCurve

#include <cstdint>
#include <cmath>


constexpr int32_t      ALGO_CURVE_LUT_SIZE      = 4096;
constexpr HighPrecType ALGO_CURVE_LUT_PAD_KNEES = 16.0;


// ---------------------------------------------------------------------------
//  One tabulated curve.
//
//  SIZE+1 entries, not SIZE: the interpolation reads d[i] and d[i+1], so the
//  last valid index must still have a partner. The extra entry holds the
//  asymptote, so an argument exactly at the top of the domain interpolates
//  against the right value rather than off the end of the array.
// ---------------------------------------------------------------------------
struct AlgoCurveLut
{
    AVX2_ALIGN AlgoType d[ALGO_CURVE_LUT_SIZE + 1];

    AlgoType lo;        // argument at entry 0
    AlgoType invStep;   // entries per unit argument
    AlgoType maxIdx;    // SIZE-1, for the clamp
};


// ---------------------------------------------------------------------------
//  Softplus used ONLY to fill the table -- 4097 times per curve per frame,
//  well under a millisecond -- so it is written for exactness rather than
//  speed and stays in HighPrecType in BOTH paths.
//
//  log1p rather than log(1+exp): for large negative arguments the addition
//  would discard every significant digit the exponential has, and that is
//  precisely the region governing the toe.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoCurveSoftplusExact
(
    const HighPrecType x,
    const HighPrecType k
) noexcept
{
    // A malformed profile with a non-positive knee would divide by zero. The
    // limit of the softplus as k goes to zero is max(x, 0), so return that
    // rather than a NaN.
    if (k <= 0.0)
        return MAX_VALUE(x, static_cast<HighPrecType>(0.0));

    const HighPrecType z = x / k;

    // Far up the ramp the function IS its asymptote to the last bit.
    if (z > static_cast<HighPrecType>(ALGO_SOFTPLUS_LINEAR_LIMIT))
        return x;

    return k * std::log1p(std::exp(z));
}


// ---------------------------------------------------------------------------
//  Fill a table for one curve.
//
//  The table is a function of the CURVE ARGUMENT, not of exposure: the
//  reversal negation, the anchor trim and the interimage correction are all
//  applied to the argument by the caller before the lookup, so ONE table serves
//  the negative and reversal paths, every iteration of stage 8b, and the print
//  curve at stage 13.
// ---------------------------------------------------------------------------
inline void AlgoBuildCurveLut
(
    const film::ToneCurve& curve,
    AlgoCurveLut&          lut
) noexcept
{
    const HighPrecType dmin  = static_cast<HighPrecType>(curve.dmin);
    const HighPrecType gamma = static_cast<HighPrecType>(curve.gamma);
    const HighPrecType toeX  = static_cast<HighPrecType>(curve.toe_x);
    const HighPrecType toeK  = static_cast<HighPrecType>(curve.toe_k);
    const HighPrecType shX   = static_cast<HighPrecType>(curve.shoulder_x);
    const HighPrecType shK   = static_cast<HighPrecType>(curve.shoulder_k);

    // Knees floored at a small positive value, so a malformed profile with a
    // zero knee still yields a finite ordered domain rather than a zero-width
    // one that would divide by zero below.
    const HighPrecType pad =
        ALGO_CURVE_LUT_PAD_KNEES * MAX_VALUE(MAX_VALUE(toeK, shK),
                                             static_cast<HighPrecType>(0.05));

    const HighPrecType lo = MIN_VALUE(toeX, shX) - pad;
    const HighPrecType hi = MAX_VALUE(toeX, shX) + pad;

    const HighPrecType span = MAX_VALUE(hi - lo,
                                        static_cast<HighPrecType>(1.0e-6));

    const HighPrecType step = span / static_cast<HighPrecType>(
                                         ALGO_CURVE_LUT_SIZE - 1);

    for (int32_t i = 0; i <= ALGO_CURVE_LUT_SIZE; i++)
    {
        const HighPrecType a = lo + step * static_cast<HighPrecType>(i);

        const HighPrecType rise = AlgoCurveSoftplusExact(a - toeX, toeK);
        const HighPrecType fall = AlgoCurveSoftplusExact(a - shX,  shK);

        lut.d[i] = static_cast<AlgoType>(dmin + gamma * (rise - fall));
    }

    lut.lo      = static_cast<AlgoType>(lo);
    lut.invStep = static_cast<AlgoType>(1.0 / step);
    lut.maxIdx  = static_cast<AlgoType>(ALGO_CURVE_LUT_SIZE - 1);

    return;
}


// ---------------------------------------------------------------------------
//  One density from one curve argument, scalar.
//
//  Clamped BEFORE the index is formed, so no argument can produce a negative or
//  out-of-range subscript -- which on the vector path's gather would read
//  arbitrary memory rather than merely give a wrong answer. The clamp is exact
//  rather than defensive, for the flatness reason set out at the top.
// ---------------------------------------------------------------------------
inline AlgoType AlgoCurveLutEval
(
    const AlgoType      arg,
    const AlgoCurveLut& lut
) noexcept
{
    AlgoType t = (arg - lut.lo) * lut.invStep;

    t = MAX_VALUE(t, ALGO_ZERO);
    t = MIN_VALUE(t, lut.maxIdx);

    const int32_t  i = static_cast<int32_t>(t);
    const AlgoType f = t - static_cast<AlgoType>(i);

    const AlgoType d0 = lut.d[i];
    const AlgoType d1 = lut.d[i + 1];

    return d0 + (d1 - d0) * f;
}


// ---------------------------------------------------------------------------
//  Eight densities from eight curve arguments -- the VECTOR lookup.
//
//  Lives here rather than in one stage's translation unit because stages 08,
//  08b and 13 all need it and all must read the same table. Compiled only into
//  the vector build; the scalar build never sees it.
//
//  Clamped to the domain BEFORE the index is formed, in float, so no lane can
//  produce a negative or out-of-range index -- which on a gather would read
//  arbitrary memory rather than merely give a wrong answer. The clamp is exact
//  rather than defensive, for the flatness reason set out at the top.
//
//  Two gathers rather than one: the pair (i, i+1) cannot be fetched by a single
//  gather, and interleaving the table so both values sit in one 64-bit slot was
//  not taken -- it doubles the footprint, and the second gather hits the cache
//  lines the first just pulled in.
// ---------------------------------------------------------------------------
#if defined(ALGO_TARGET_AVX2)

#include <immintrin.h>

static_assert(sizeof(AlgoType) == 4,
              "the vector curve lookup gathers 32-bit lanes");

inline __m256 AlgoCurveLutEvalV
(
    const __m256        arg,
    const AlgoCurveLut& lut
) noexcept
{
    // Position in table units, clamped into [0, SIZE-1].
    const __m256 pos = _mm256_mul_ps(_mm256_sub_ps(arg,
                                                   _mm256_set1_ps(lut.lo)),
                                     _mm256_set1_ps(lut.invStep));

    const __m256 posC = _mm256_min_ps(_mm256_max_ps(pos,
                                                    _mm256_setzero_ps()),
                                      _mm256_set1_ps(lut.maxIdx));

    // Truncation is floor here because the value is already non-negative.
    const __m256i idx = _mm256_cvttps_epi32(posC);

    // Fractional part, from the integer part converted back - cheaper than a
    // separate floor and exact, since both come from the same value.
    const __m256 frac = _mm256_sub_ps(posC, _mm256_cvtepi32_ps(idx));

    const __m256 d0 = _mm256_i32gather_ps(lut.d, idx, 4);
    const __m256 d1 = _mm256_i32gather_ps(lut.d + 1, idx, 4);

    // Linear interpolation as one FMA on the difference: d0 + frac*(d1-d0).
    return _mm256_fmadd_ps(frac, _mm256_sub_ps(d1, d0), d0);
}



#endif  // ALGO_TARGET_AVX2
