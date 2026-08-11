// ---------------------------------------------------------------------------
//  Algo_08_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//
//  WHAT IS VECTORISED HERE, AND WHAT IS NOT
//
//  Only AlgoStage08_CharacteristicCurve. It measures 121.67 ms of a 605.95 ms frame
//  at 1024 x 1024 - twenty per cent - and it is the only stage in the 4..8 range with
//  substantial arithmetic inside its own translation unit. Per sample per channel it
//  evaluates FIVE transcendentals: one base-ten logarithm for the exposure, then two
//  softplus ramps, each of which is k * log1p(exp(z/k)). At 1 Mpx that is fifteen
//  million transcendental calls per frame, which is why it costs what it does.
//
//  The six frame-setup functions below - AlgoDensityScalar, AlgoTintFactor,
//  AlgoSolveAnchors, AlgoNeutralMidDensity, AlgoSolveStageOffsets,
//  AlgoSolveIntermediateOffsets - are VERBATIM from the scalar unit and stay in
//  HighPrecType. They run once per frame on scalars; profall attributes 0.10 ms to
//  the whole anchor solve. Vectorising them would gain nothing and risk everything:
//  the solve is a sixty-step bisection whose bracket shrinks below float resolution
//  long before it finishes.
//
//  AlgoStage08b_Interimage is left scalar deliberately. It measures 1.38 ms - two
//  tenths of one per cent - because it is inactive on most stocks, and it re-evaluates
//  the curve inside a fixed-point iteration, so vectorising it would multiply the
//  transcendental approximation error by the iteration count for no measurable gain.
//
//  TRANSCENDENTALS COME FROM FastCompute::AVX2, WHICH IS A SPEED CHOICE
//
//  Measured against a double reference over the ranges this stage uses:
//
//      Log   max rel 9.45e-05    max abs 1.52e-05
//      Exp   max rel 2.98e-02    (Schraudolph bit-trick)
//
//  Exp carries three per cent. Propagated through the softplus - whose derivative
//  with respect to its exponential is u/(1+u), at most one - that is up to 0.03
//  absolute in each ramp, and multiplied by gamma, which reaches about 3.5 on the
//  contrasty stocks, roughly 0.1 in density. Around ten per cent in transmittance.
//
//  That is a KNOWN and DELIBERATE trade, taken to measure the fast path end to end
//  before choosing between fast and accurate. It is not a defect to be discovered
//  later: this file will not match the scalar reference to the 1e-6 budget, and it is
//  not meant to yet. The accurate alternative is a range-reduced exp - n =
//  round(x*log2e), r = x - n*ln2, minimax polynomial on r, exponent field written
//  directly - at roughly twelve instructions instead of two.
//
//  Pipeline stage 8: exposure to density.
//
//      AlgoDensityScalar               one value through one characteristic curve
//      AlgoSolveAnchors                per-channel neutral-grey anchoring
//      AlgoTintFactor                  residual base-tint multiplier
//      AlgoNeutralMidDensity           neutral density leaving the negative
//      AlgoSolveStageOffsets           print offsets onto a display target
//      AlgoSolveIntermediateOffsets    offsets centring a dupe stock's range
//      AlgoStage08_CharacteristicCurve the per-pixel pass
//      AlgoStage08b_Interimage         cross-layer development inhibition
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

#include "AlgoCharacteristicCurve.hpp"
#include "AlgoInterimage.hpp"
#include "AlgoHalation.hpp"   // AlgoSoftplus, ALGO_SOFTPLUS_LINEAR_LIMIT

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>

#include <cmath>   // std::log1p, std::exp, std::log10, std::pow


static_assert(sizeof(AlgoType) == 4,
              "the AVX2 path requires AlgoType to be a 32-bit float");


namespace
{
    // ----------------------------------------------------------------------
    //  Lanes in one AVX2 vector of float, and 1/ln(10) for the base change.
    //
    //  There is no vector log10, so the natural logarithm is scaled by 1/ln(10) -
    //  which is precisely what a scalar log10 does internally. The constant is given
    //  to full double precision and narrowed by the compiler, so it is the closest
    //  float to 1/ln(10) rather than the closest float to a typed-out decimal.
    // ----------------------------------------------------------------------
    constexpr int32_t ALGO_AVX2_LANES   = 8;
    constexpr float   ALGO_AVX2_INV_LN10 =
        static_cast<float>(0.434294481903251827651128918916605082);


    // ----------------------------------------------------------------------
    //  Tail mask for the final, partial vector of a row.
    //
    //  The active width is not generally a multiple of eight. Masked access leaves
    //  the row padding untouched, which keeps the NaN-poison arena test meaningful.
    // ----------------------------------------------------------------------
    inline __m256i algoTailMask (const int32_t n) noexcept
    {
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


    // ----------------------------------------------------------------------
    //  CHARACTERISTIC CURVE AS A TABLE.
    //
    //  WHY A TABLE IS THE RIGHT ANSWER HERE, AND WHERE IT IS NOT.
    //
    //  The curve is a difference of two softplus ramps, so evaluating it costs
    //  two Exp and two Log. In stage 8 that is paid once per sample. In stage 8b
    //  it is paid THREE TIMES PER PIXEL PER ITERATION - three channels inside a
    //  fixed-point loop - which at the profile-driven iteration counts in this
    //  database means up to twelve curve evaluations per pixel. Measured on an
    //  HD frame, 8b was 184 ms of a 694 ms frame: the largest single cost in the
    //  engine, larger than halation.
    //
    //  A table collapses all four transcendentals into one gather plus a linear
    //  interpolation, and it is built ONCE PER FRAME PER CHANNEL - 2048 entries,
    //  scalar, in HighPrecType, using log1p exactly as the scalar reference does.
    //
    //  IT IS ALSO MORE ACCURATE, WHICH IS THE UNUSUAL PART. The vector path's
    //  Exp is the Schraudolph bit trick, ~3 per cent relative error, and the
    //  whole-pipeline error against the scalar reference measured 4.37e-02 -
    //  eleven code values at eight bits - with essentially all of it coming from
    //  that approximation inside the curve. A 2048-entry table interpolated
    //  linearly measures 8.67e-06 against the same reference: three orders of
    //  magnitude better, because the table entries themselves are computed the
    //  scalar way and only the interpolation between them is approximate.
    //
    //  WHERE A TABLE WOULD BE WRONG: any stage that evaluates ONE exponential
    //  per sample. Measured, a table lookup with its gather costs 0.54 ns per
    //  sample against Schraudolph's 0.22 ns, so it is 2.4x SLOWER per call and
    //  only wins when it replaces several calls at once. Stage 14's single Exp
    //  is deliberately left alone for exactly this reason.
    //
    //  DOMAIN AND CLAMPING. Both ends of the curve are asymptotically FLAT:
    //  below the toe both ramps vanish and the density is dmin; above the
    //  shoulder both ramps are linear with unit slope and their difference is
    //  the constant (shoulder_x - toe_x). So clamping the argument to the table
    //  domain is not an approximation, it is exact - provided the domain covers
    //  the transition. The padding below is ten knee widths, where a softplus
    //  is within k*exp(-10) = 4.5e-05*k of its asymptote.
    // ----------------------------------------------------------------------
    constexpr int32_t ALGO_CURVE_LUT_SIZE = 2048;

    // Knee widths of padding either side of the transition region. Ten is where
    // the softplus tail falls below the density resolution of the table itself.
    constexpr HighPrecType ALGO_CURVE_LUT_PAD_KNEES = 10.0;


    // ----------------------------------------------------------------------
    //  One tabulated curve.
    //
    //  STACK-LOCAL BY CONSTRUCTION - the caller declares it, so there is no
    //  static, no allocation and no shared mutable state. The engine's
    //  reentrancy guarantee (arbitrary concurrent frames, any order) survives.
    //  Three of these is 24 KB, which sits in L2 and is gone when the stage
    //  returns.
    //
    //  SIZE+1 entries, not SIZE: the interpolation reads d[i] and d[i+1], and
    //  the last valid index must still have a partner. The extra entry holds the
    //  asymptote, so an argument exactly at the top of the domain interpolates
    //  against the right value rather than off the end of the array.
    // ----------------------------------------------------------------------
    struct AlgoCurveLut
    {
        AVX2_ALIGN float d[ALGO_CURVE_LUT_SIZE + 1];

        float lo;        // argument at entry 0
        float invStep;   // entries per unit argument
        float maxIdx;    // SIZE-1 as a float, for the clamp
    };


    // ----------------------------------------------------------------------
    //  Softplus, scalar, VERBATIM in behaviour with the scalar translation unit.
    //
    //  Used only to fill the table - once per entry, 2048 times per channel per
    //  frame, which is under 0.1 ms - so it is written for exactness rather than
    //  speed and stays in HighPrecType. log1p rather than log(1+exp) because for
    //  large negative arguments the addition would discard every significant
    //  digit the exponential has, and that is the region governing the toe.
    // ----------------------------------------------------------------------
    inline HighPrecType curveSoftplusExact
    (
        const HighPrecType x,
        const HighPrecType k
    ) noexcept
    {
        if (k <= 0.0)
            return MAX_VALUE(x, static_cast<HighPrecType>(0.0));

        const HighPrecType z = x / k;

        // Far up the ramp the function IS its asymptote to the last bit.
        if (z > static_cast<HighPrecType>(ALGO_SOFTPLUS_LINEAR_LIMIT))
            return x;

        return k * std::log1p(std::exp(z));
    }


    // ----------------------------------------------------------------------
    //  Fill a table for one channel's curve.
    //
    //  The table is a function of the CURVE ARGUMENT, not of exposure: the
    //  reversal negation, the anchor trim and the interimage correction are all
    //  applied to the argument by the caller before the lookup, so one table
    //  serves the negative and reversal paths and every iteration of 8b.
    // ----------------------------------------------------------------------
    inline void buildCurveLut
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

        // Domain: the transition region plus ten knee widths either side. The
        // knees are floored at a small positive value so a malformed profile with
        // a zero knee still produces a finite, ordered domain rather than a
        // zero-width one that would divide by zero below.
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
            // Entry SIZE is one step past the domain, holding the asymptote so
            // the interpolation at the very top of the range has a partner.
            const HighPrecType a = lo + step * static_cast<HighPrecType>(i);

            const HighPrecType rise = curveSoftplusExact(a - toeX, toeK);
            const HighPrecType fall = curveSoftplusExact(a - shX,  shK);

            lut.d[i] = static_cast<float>(dmin + gamma * (rise - fall));
        }

        lut.lo      = static_cast<float>(lo);
        lut.invStep = static_cast<float>(1.0 / step);
        lut.maxIdx  = static_cast<float>(ALGO_CURVE_LUT_SIZE - 1);

        return;
    }


    // ----------------------------------------------------------------------
    //  Eight densities from eight curve arguments.
    //
    //  Clamped to the domain BEFORE the index is formed, in float, so no lane can
    //  produce a negative or out-of-range index - which on a gather would read
    //  arbitrary memory rather than merely give a wrong answer. The clamp is
    //  exact rather than defensive, for the flatness reason set out above.
    //
    //  Two gathers rather than one: the pair (i, i+1) cannot be fetched by a
    //  single gather, and the alternative - interleaving the table so both
    //  values sit in one 64-bit slot - was not taken because it doubles the
    //  table footprint and the second gather hits the same cache lines the first
    //  just pulled in.
    // ----------------------------------------------------------------------
    inline __m256 algoCurveLutV
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


    // ----------------------------------------------------------------------
    //  Vector softplus:  k * log(1 + exp(x/k))
    //
    //  The shape of the whole characteristic curve. Two of these per sample per
    //  channel, so this function IS stage 8's cost.
    //
    //  THE LINEAR ASYMPTOTE IS NOT AN OPTIMISATION, IT IS A CORRECTNESS GUARD.
    //
    //  Far up the ramp softplus equals its own asymptote x to well beyond the last
    //  representable bit, and the exponential is heading for overflow. The scalar
    //  path returns x directly above ALGO_SOFTPLUS_LINEAR_LIMIT; this does the same
    //  through a blend, so both paths agree in the tail and neither has to survive an
    //  infinity. Selecting rather than branching also keeps all eight lanes on one
    //  code path, which is the point of the exercise.
    //
    //  Both sides of the blend are ALWAYS evaluated - that is how a vector select
    //  works - so the exponential still runs on lanes whose result is discarded.
    //  FastCompute's Exp clamps its argument to +/- 87 internally, which is what
    //  keeps those discarded lanes from producing an infinity that could contaminate
    //  nothing, but would show up under a NaN check.
    //
    //  x       already offset:  logE - toe_x  or  logE - shoulder_x
    //  k       knee width
    //  invK    its reciprocal, formed once per channel outside the pixel loop
    // ----------------------------------------------------------------------
    FORCE_INLINE __m256 algoSoftplusV (const __m256 x,
                                       const __m256 k,
                                       const __m256 invK) noexcept
    {
        const __m256 z = _mm256_mul_ps(x, invK);

        const __m256 vLimit = _mm256_set1_ps(
            static_cast<float>(ALGO_SOFTPLUS_LINEAR_LIMIT));

        // Lanes where the asymptote applies. _CMP_GT_OQ - ordered, non-signalling -
        // so a NaN argument selects the log path rather than raising.
        const __m256 useLinear = _mm256_cmp_ps(z, vLimit, _CMP_GT_OQ);

        // log1p(exp(z)) as log(1 + exp(z)). The addition of one is exact for any
        // exp result at or above the float epsilon, and below that both forms give
        // zero - so the accuracy the scalar log1p buys back is entirely inside a
        // region where the softplus contributes nothing to the density.
        const __m256 u = FastCompute::AVX2::Exp(z);

        const __m256 lg =
            FastCompute::AVX2::Log(_mm256_add_ps(u, _mm256_set1_ps(1.0f)));

        return _mm256_blendv_ps(_mm256_mul_ps(k, lg), x, useLinear);
    }


    // ----------------------------------------------------------------------
    //  Scalar softplus in HighPrecType.
    //
    //  The engine already exports AlgoSoftplus, but that one works in AlgoType,
    //  which is the image type and may be single precision. The anchor solve
    //  runs a sixty-step bisection whose bracket shrinks below the resolution of
    //  a float long before it finishes, so it needs the wider form throughout.
    // ----------------------------------------------------------------------
    inline HighPrecType softplusHP (const HighPrecType x, const HighPrecType k) noexcept
    {
        // Normalised argument. The curve parameters guarantee k > 0.
        const HighPrecType z = x / k;

        // Far up the ramp the function equals its own asymptote to well beyond
        // the last representable bit, so return it and skip an exponential that
        // is heading for overflow.
        if (z > static_cast<HighPrecType>(ALGO_SOFTPLUS_LINEAR_LIMIT))
            return x;

        // log1p rather than log(1 + e): for large negative z the exponential is
        // tiny and adding one to it would discard all of its significant digits.
        return k * std::log1p(std::exp(z));
    }


    // ----------------------------------------------------------------------
    //  Density to display-normalised transmittance, for one curve.
    //
    //  Maps the curve's own working range onto zero to one: clear film - which
    //  is the brightest the stock can be, at Dmin - becomes one, and Dmax
    //  becomes zero. Normalising against the curve's OWN endpoints rather than
    //  against absolute transmittance is what lets the anchor target be
    //  expressed as a plain display value independent of how dense a particular
    //  stock's base happens to be.
    // ----------------------------------------------------------------------
    inline HighPrecType normalisedTransmittance
    (
        const HighPrecType     d,
        const film::ToneCurve& c
    ) noexcept
    {
        // Transmittance is ten to the minus density, by the definition of
        // optical density.
        const HighPrecType tMax = std::pow(10.0, -static_cast<HighPrecType>(c.dmin));
        const HighPrecType tMin = std::pow(10.0, -static_cast<HighPrecType>(c.dmax()));

        const HighPrecType span = tMax - tMin;

        // A curve with zero gamma has Dmax equal to Dmin and no span at all.
        // There is no meaningful normalisation in that case; returning zero
        // keeps the bisection well defined rather than producing a division by
        // zero that would poison the solve.
        if (span <= 0.0)
            return 0.0;

        return (std::pow(10.0, -d) - tMin) / span;
    }


    // ----------------------------------------------------------------------
    //  Curve accessor by index, so the three channels share one code path.
    // ----------------------------------------------------------------------
    inline const film::ToneCurve& curveOf
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
    //  Flat-field part of the DIR coupler effect.
    //
    //  On an even field the lateral edge term of the coupler chemistry vanishes,
    //  but the cross-layer term does not: it pushes each layer away from the mean
    //  of the three. Because curve crossover means a neutral grey does NOT sit at
    //  equal density in all three layers, that shifts the mid tone by a few per
    //  cent per channel, and the anchor solve has to account for it.
    //
    //  d          three densities, modified in place
    //  strength   coupler strength times the user scale; zero disables
    // ----------------------------------------------------------------------
    inline void applyFlatCoupler
    (
        HighPrecType       d[3],
        const HighPrecType strength
    ) noexcept
    {
        if (strength <= 0.0)
            return;

        // Mean of the three layer densities.
        const HighPrecType dbar = (d[0] + d[1] + d[2]) / 3.0;

        // Each layer is pushed further from that mean. Read all three before
        // writing any, since every output depends on the mean of all inputs.
        const HighPrecType d0 = d[0] + strength * (d[0] - dbar);
        const HighPrecType d1 = d[1] + strength * (d[1] - dbar);
        const HighPrecType d2 = d[2] + strength * (d[2] - dbar);

        d[0] = d0;
        d[1] = d1;
        d[2] = d2;

        return;
    }


    // ----------------------------------------------------------------------
    //  Log exposure a neutral 18 per cent grey delivers to each record.
    //
    //  A neutral grey is 1.0 in relative exposure on every channel, so what each
    //  record receives after the taking filters is the sum of that record's row
    //  of the taking matrix. The logarithm of that sum is the record's starting
    //  point on its own curve.
    // ----------------------------------------------------------------------
    inline void neutralLogE
    (
        const film::Matrix3& take,
        HighPrecType         logEOut[3]
    ) noexcept
    {
        for (int32_t k = 0; k < 3; k++)
        {
            const HighPrecType rowSum = static_cast<HighPrecType>(take[k][0])
                                      + static_cast<HighPrecType>(take[k][1])
                                      + static_cast<HighPrecType>(take[k][2]);

            // Floored before the logarithm for the same reason the pixel loop
            // floors exposure: a row that sums to zero would give minus infinity.
            logEOut[k] = std::log10(MAX_VALUE(rowSum,
                                    static_cast<HighPrecType>(
                                        ALGO_CURVE_EXPOSURE_FLOOR)));
        }

        return;
    }
}


// ---------------------------------------------------------------------------
//  Scalar characteristic curve
// ---------------------------------------------------------------------------
HighPrecType AlgoDensityScalar
(
    const HighPrecType     logE,
    const film::ToneCurve& curve
) noexcept
{
    // Rising ramp: flat below the toe, then linear. This alone would climb
    // without limit.
    const HighPrecType rise = softplusHP(
        logE - static_cast<HighPrecType>(curve.toe_x),
        static_cast<HighPrecType>(curve.toe_k));

    // Falling ramp: flat below the shoulder, then linear with the same unit
    // slope, so above the shoulder it exactly cancels the rise and the curve
    // levels off at Dmax.
    const HighPrecType fall = softplusHP(
        logE - static_cast<HighPrecType>(curve.shoulder_x),
        static_cast<HighPrecType>(curve.shoulder_k));

    // Base plus fog, plus the contrast slope applied to the bracket.
    return static_cast<HighPrecType>(curve.dmin)
         + static_cast<HighPrecType>(curve.gamma) * (rise - fall);
}


// ---------------------------------------------------------------------------
//  Residual base-tint multiplier for one channel
// ---------------------------------------------------------------------------
HighPrecType AlgoTintFactor
(
    const film::FilmProfile& profile,
    const int32_t            c
) noexcept
{
    // The target for a channel is divided by this, so a channel whose base is
    // tinted warm is aimed slightly lower and the tint survives into the result
    // rather than being solved away by the very stage meant to preserve it.
    return 1.0 + (static_cast<HighPrecType>(profile.base_tint[c]) - 1.0)
               * ALGO_TINT_RESIDUAL;
}


namespace
{
    // ----------------------------------------------------------------------
    //  Bisection on a monotonic function.
    //
    //  fn        evaluated through a small function object
    //  lo, hi    bracket, assumed to contain the solution
    //  target    value to solve for
    //  rising    true when fn increases with its argument
    //
    //  No convergence test: the iteration count is fixed and generous, which
    //  makes the running time identical for every pixel-independent call and
    //  removes any possibility of a stock with a pathological curve spinning.
    // ----------------------------------------------------------------------
    template <typename Fn>
    HighPrecType bisect
    (
        Fn                 fn,
        HighPrecType       lo,
        HighPrecType       hi,
        const HighPrecType target,
        const bool         rising
    ) noexcept
    {
        for (int32_t i = 0; i < ALGO_ANCHOR_BISECTIONS; i++)
        {
            const HighPrecType mid = 0.5 * (lo + hi);

            // Which half of the bracket the solution lies in depends on both
            // which side of the target we landed and which way the function
            // runs. Comparing the two booleans handles both directions without
            // duplicating the branch.
            const bool above = fn(mid) > target;

            if (above == rising)
                hi = mid;
            else
                lo = mid;
        }

        return 0.5 * (lo + hi);
    }
}


// ---------------------------------------------------------------------------
//  Anchor solve
// ---------------------------------------------------------------------------
void AlgoSolveAnchors
(
    const film::FilmProfile& profile,
    const film::PrintStock*  pPrintStock,
    const HighPrecType       greyTarget,
    const HighPrecType       couplerScale,
    HighPrecType             anchorOut[3]
) noexcept
{
    const film::RGBCurves& curves = profile.curves;
    const film::Matrix3&   negM   = profile.dye_matrix;

    // Where a neutral grey starts on each record's own curve.
    HighPrecType logEMid[3];
    neutralLogE(profile.taking_matrix, logEMid);

    // Flat-field coupler strength. Monochrome stocks have one layer and so no
    // cross-layer term at all.
    const HighPrecType couplerStrength =
        (profile.is_monochrome)
            ? 0.0
            : MAX_VALUE(static_cast<HighPrecType>(profile.couplers.strength)
                        * couplerScale, 0.0);

    // Per-channel display targets, each carrying its share of the base tint.
    HighPrecType target[3];

    for (int32_t c = 0; c < 3; c++)
        target[c] = greyTarget / AlgoTintFactor(profile, c);

    // ----------------------------------------------------------------------
    //  Reversal stock: no print stage, so exposure itself is the free parameter.
    // ----------------------------------------------------------------------
    if (profile.isReversal())
    {
        HighPrecType trim[3] = { 0.0, 0.0, 0.0 };

        for (int32_t sweep = 0; sweep < ALGO_ANCHOR_SWEEPS; sweep++)
        {
            // Densities at the current trims, frozen for this sweep. Two of the
            // three are held while the third is re-solved, and sweeping to
            // convergence is what resolves the coupling between them.
            HighPrecType frozen[3];

            for (int32_t k = 0; k < 3; k++)
                frozen[k] = AlgoDensityScalar(-(logEMid[k] + trim[k]),
                                              curveOf(curves, k));

            for (int32_t c = 0; c < 3; c++)
            {
                // Evaluate the display value this channel reaches for a trial
                // trim, with the other two channels held at their frozen values.
                //
                // A lambda rather than a loose function so it can capture the
                // frozen state without a mutable static, which the reentrancy
                // rule forbids.
                auto fn = [&] (const HighPrecType t) noexcept -> HighPrecType
                {
                    HighPrecType d[3] = { frozen[0], frozen[1], frozen[2] };

                    // The negated log exposure is what a reversal curve is
                    // expressed against: more light means less density.
                    d[c] = AlgoDensityScalar(-(logEMid[c] + t), curveOf(curves, c));

                    applyFlatCoupler(d, couplerStrength);

                    // Through the stock's own dye matrix, which is what actually
                    // scales neutral density before it reaches the eye.
                    const HighPrecType mixed =
                        static_cast<HighPrecType>(negM[c][0]) * d[0]
                      + static_cast<HighPrecType>(negM[c][1]) * d[1]
                      + static_cast<HighPrecType>(negM[c][2]) * d[2];

                    return normalisedTransmittance(mixed, curveOf(curves, c));
                };

                // More exposure on a slide means less density means a brighter
                // result, so the function rises with the trim.
                trim[c] = bisect(fn,
                                 -ALGO_ANCHOR_BRACKET,
                                  ALGO_ANCHOR_BRACKET,
                                  target[c],
                                  true);
            }
        }

        anchorOut[0] = trim[0];
        anchorOut[1] = trim[1];
        anchorOut[2] = trim[2];

        return;
    }

    // ----------------------------------------------------------------------
    //  Negative stock: the free parameter is the print exposure offset.
    // ----------------------------------------------------------------------

    // Neutral density on the camera negative, after the couplers and the
    // negative's own dye matrix. This is the fixed starting point every printing
    // stage anchors against.
    HighPrecType dNeg[3];

    for (int32_t k = 0; k < 3; k++)
        dNeg[k] = AlgoDensityScalar(logEMid[k], curveOf(curves, k));

    applyFlatCoupler(dNeg, couplerStrength);

    HighPrecType dMid[3];

    for (int32_t c = 0; c < 3; c++)
        dMid[c] = static_cast<HighPrecType>(negM[c][0]) * dNeg[0]
                + static_cast<HighPrecType>(negM[c][1]) * dNeg[1]
                + static_cast<HighPrecType>(negM[c][2]) * dNeg[2];

    // Without a print stock there is nothing to print onto and no offset to
    // solve. The negative densities are handed back so the caller still has a
    // defined, meaningful value rather than an uninitialised one.
    if (nullptr == pPrintStock)
    {
        anchorOut[0] = dMid[0];
        anchorOut[1] = dMid[1];
        anchorOut[2] = dMid[2];

        return;
    }

    // Delegate to the public solver, which stage 13 also uses: after a dupe chain
    // the neutral density has moved and the final print offsets have to be
    // re-solved against the new value, so the routine cannot stay private here.
    AlgoSolveStageOffsets(dMid, pPrintStock->curves, pPrintStock->dye_matrix,
                          target, anchorOut);

    return;
}


// ---------------------------------------------------------------------------
//  Density a neutral 18 per cent grey reaches on the camera negative
// ---------------------------------------------------------------------------
void AlgoNeutralMidDensity
(
    const film::FilmProfile& profile,
    const HighPrecType       couplerScale,
    HighPrecType             dMidOut[3]
) noexcept
{
    const film::RGBCurves& curves = profile.curves;

    // Where a neutral grey starts on each record's own curve.
    HighPrecType logEMid[3];
    neutralLogE(profile.taking_matrix, logEMid);

    HighPrecType d[3];

    for (int32_t k = 0; k < 3; k++)
        d[k] = AlgoDensityScalar(logEMid[k], curveOf(curves, k));

    // Flat-field coupler term. A monochrome stock has one layer and therefore no
    // cross-layer inhibition at all.
    const HighPrecType strength =
        (profile.is_monochrome)
            ? 0.0
            : MAX_VALUE(static_cast<HighPrecType>(profile.couplers.strength)
                        * couplerScale, 0.0);

    applyFlatCoupler(d, strength);

    // Through the negative's own dye matrix, which is what actually scales
    // neutral density before the image leaves the negative.
    const film::Matrix3& negM = profile.dye_matrix;

    for (int32_t c = 0; c < 3; c++)
        dMidOut[c] = static_cast<HighPrecType>(negM[c][0]) * d[0]
                   + static_cast<HighPrecType>(negM[c][1]) * d[1]
                   + static_cast<HighPrecType>(negM[c][2]) * d[2];

    return;
}


// ---------------------------------------------------------------------------
//  Print offsets landing a neutral grey on given display targets
// ---------------------------------------------------------------------------
void AlgoSolveStageOffsets
(
    const HighPrecType     dMid[3],
    const film::RGBCurves& dstCurves,
    const film::Matrix3&   dstMatrix,
    const HighPrecType     target[3],
    HighPrecType           offsetOut[3]
) noexcept
{
    // Starting estimate. The offset enters the print exposure as
    // logE_print = offset - D, so seeding it at D puts the print at zero log
    // exposure: the wrong answer, but a well-centred bracket.
    HighPrecType offset[3] = { dMid[0], dMid[1], dMid[2] };

    for (int32_t sweep = 0; sweep < ALGO_ANCHOR_SWEEPS; sweep++)
    {
        // Densities at the current offsets, frozen for this sweep. Two of the
        // three are held while the third is re-solved; sweeping to convergence is
        // what resolves the coupling the destination dye matrix introduces.
        HighPrecType frozen[3];

        for (int32_t k = 0; k < 3; k++)
            frozen[k] = AlgoDensityScalar(offset[k] - dMid[k],
                                          curveOf(dstCurves, k));

        for (int32_t c = 0; c < 3; c++)
        {
            // A lambda rather than a loose function so it can capture the frozen
            // state without a mutable static, which the reentrancy rule forbids.
            auto fn = [&] (const HighPrecType off) noexcept -> HighPrecType
            {
                HighPrecType dp[3] = { frozen[0], frozen[1], frozen[2] };

                dp[c] = AlgoDensityScalar(off - dMid[c], curveOf(dstCurves, c));

                const HighPrecType mixed =
                    static_cast<HighPrecType>(dstMatrix[c][0]) * dp[0]
                  + static_cast<HighPrecType>(dstMatrix[c][1]) * dp[1]
                  + static_cast<HighPrecType>(dstMatrix[c][2]) * dp[2];

                return normalisedTransmittance(mixed, curveOf(dstCurves, c));
            };

            // More offset means more print exposure, which means more density on
            // the print, which means a DARKER result - so the function FALLS as
            // the offset rises.
            offset[c] = bisect(fn,
                               dMid[c] - ALGO_ANCHOR_BRACKET,
                               dMid[c] + ALGO_ANCHOR_BRACKET,
                               target[c],
                               false);
        }
    }

    offsetOut[0] = offset[0];
    offsetOut[1] = offset[1];
    offsetOut[2] = offset[2];

    return;
}


// ---------------------------------------------------------------------------
//  Offsets centring a neutral grey in a duplicating stock's usable range
// ---------------------------------------------------------------------------
void AlgoSolveIntermediateOffsets
(
    const HighPrecType     dMid[3],
    const film::RGBCurves& dstCurves,
    HighPrecType           offsetOut[3],
    HighPrecType           newMidOut[3]
) noexcept
{
    for (int32_t c = 0; c < 3; c++)
    {
        const film::ToneCurve& dst = curveOf(dstCurves, c);

        // Midpoint of this stock's own density range. Nothing views an
        // intermediate, so there is no display value to aim at; the midpoint is
        // what keeps three or four generations from drifting into the toe or the
        // shoulder.
        const HighPrecType targetD = 0.5 * (static_cast<HighPrecType>(dst.dmin)
                                          + static_cast<HighPrecType>(dst.dmax()));

        // Density rises with offset here, because there is no transmittance
        // inversion in the way: this solves in the DENSITY domain, not the display
        // domain, which is why the direction flag is the opposite of the one in
        // AlgoSolveStageOffsets.
        auto fn = [&] (const HighPrecType off) noexcept -> HighPrecType
        {
            return AlgoDensityScalar(off - dMid[c], dst);
        };

        // A wider bracket than the display solve uses: an intermediate stock's
        // range can sit well away from the incoming neutral density.
        offsetOut[c] = bisect(fn,
                              dMid[c] - 10.0,
                              dMid[c] + 10.0,
                              targetD,
                              true);

        // The neutral density AFTER this generation is the target by construction,
        // and it is what the next generation anchors against.
        newMidOut[c] = targetD;
    }

    return;
}


// ---------------------------------------------------------------------------
//  Stage 8: characteristic curve
// ---------------------------------------------------------------------------
void AlgoStage08_CharacteristicCurve
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pLogER,
    AlgoType* RESTRICT       pLogEG,
    AlgoType* RESTRICT       pLogEB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const HighPrecType       anchor[3]
) noexcept
{
    const film::RGBCurves& curves = profile.curves;

    // A slide records a positive image directly, which changes both the sign of
    // the curve argument and whether the anchors are consumed here at all.
    const bool reversal = profile.isReversal();

    const AlgoType* RESTRICT srcPlane [3] = { pSrcR,  pSrcG,  pSrcB  };
    AlgoType* RESTRICT       dstPlane [3] = { pDstR,  pDstG,  pDstB  };
    AlgoType* RESTRICT       logEPlane[3] = { pLogER, pLogEG, pLogEB };

    for (int32_t c = 0; c < 3; c++)
    {
        const film::ToneCurve& curve = curveOf(curves, c);

        // ------------------------------------------------------------------
        //  Curve parameters, hoisted into frame constants.
        //
        //  They are stored as float while the arithmetic runs in AlgoType, and
        //  the compiler cannot prove the profile is unchanged by the stores into
        //  the destination, so it would otherwise reload and convert all six on
        //  every pixel.
        // ------------------------------------------------------------------
        const AlgoType dmin       = static_cast<AlgoType>(curve.dmin);
        const AlgoType gamma      = static_cast<AlgoType>(curve.gamma);
        const AlgoType toeX       = static_cast<AlgoType>(curve.toe_x);
        const AlgoType toeK       = static_cast<AlgoType>(curve.toe_k);
        const AlgoType shoulderX  = static_cast<AlgoType>(curve.shoulder_x);
        const AlgoType shoulderK  = static_cast<AlgoType>(curve.shoulder_k);

        // For a reversal stock the trim shifts the log exposure before the curve
        // sees it. For a negative the anchor is a PRINT offset, consumed at the
        // print stage, so nothing is applied here.
        const AlgoType trim = reversal ? static_cast<AlgoType>(anchor[c])
                                       : ALGO_ZERO;

        const AlgoType* RESTRICT pIn   = srcPlane [c];
        AlgoType* RESTRICT       pOut  = dstPlane [c];
        AlgoType* RESTRICT       pLogE = logEPlane[c];

        // ------------------------------------------------------------------
        //  Vector frame constants, broadcast once per channel.
        //
        //  Reciprocals of the two softplus widths are formed here so the pixel loop
        //  multiplies instead of dividing. A vector divide is an order of magnitude
        //  more expensive than a multiply and these are frame constants, so the
        //  division happens twice per channel rather than twice per sample.
        // ------------------------------------------------------------------
        const __m256 vFloor = _mm256_set1_ps(ALGO_CURVE_EXPOSURE_FLOOR);
        const __m256 vInvLn10 = _mm256_set1_ps(ALGO_AVX2_INV_LN10);
        const __m256 vTrim  = _mm256_set1_ps(trim);

        // ------------------------------------------------------------------
        //  The curve, tabulated once for this channel.
        //
        //  Replaces two softplus evaluations - two Exp and two Log - per sample
        //  with one gather and one FMA, and replaces the Schraudolph
        //  approximation with entries computed the scalar way. Faster AND three
        //  orders of magnitude closer to the reference; see the table's own
        //  commentary above for why that combination is possible here and not
        //  in a stage that evaluates a single exponential.
        //
        //  The remaining transcendental in this loop is the logarithm, which
        //  cannot be tabled: its argument is scene-linear exposure spanning
        //  eight decades, so a table with useful resolution in the highlights
        //  would need millions of entries.
        //
        //  ON THE STACK, one per channel iteration. 8 KB, live for one channel.
        // ------------------------------------------------------------------
        AlgoCurveLut lut;

        buildCurveLut(curve, lut);

        // Unused now that the curve is tabulated, but retained deliberately:
        // dmin/gamma/knees are what the table was BUILT from, and silently
        // dropping the names would hide which profile fields drive this stage.
        (void)dmin; (void)gamma; (void)toeX; (void)toeK;
        (void)shoulderX; (void)shoulderK;

        const int32_t vecCount = sizeX / ALGO_AVX2_LANES;
        const int32_t tailN    = sizeX - (vecCount * ALGO_AVX2_LANES);

        const __m256i vTail = algoTailMask(tailN);

        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pE = pIn   + off;
            AlgoType* RESTRICT       pL = pLogE + off;
            AlgoType* RESTRICT       pD = pOut  + off;

            int32_t x = 0;

            for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
            {
                // Floored before the logarithm: a true zero would give minus
                // infinity, and the floor sits eight decades under mid grey where
                // every stock is flat on its base fog.
                const __m256 e = _mm256_max_ps(_mm256_loadu_ps(pE + x), vFloor);

                // Base ten, because the whole of sensitometry is expressed in
                // decades of exposure and density is itself a base-ten quantity.
                // There is no vector log10, so the natural log is scaled - which is
                // exactly what a scalar log10 does internally anyway.
                const __m256 logE =
                    _mm256_mul_ps(FastCompute::AVX2::Log(e), vInvLn10);

                // RETAINED. The interimage stage reads this rather than recovering
                // it from the density, which is not invertible through the shoulder.
                _mm256_storeu_ps(pL + x, logE);

                // A reversal curve is expressed against NEGATED log exposure - more
                // light gives less density - and the trim shifts the exposure before
                // that negation. Branch on a frame constant, so it is hoisted.
                const __m256 arg = reversal
                    ? _mm256_sub_ps(_mm256_setzero_ps(),
                                    _mm256_add_ps(logE, vTrim))
                    : logE;

                // One table lookup for what was a difference of two softplus
                // ramps: base plus fog, toe, straight line of slope gamma,
                // shoulder, Dmax. Monotonic by construction - the table inherits
                // that from the expression it was built from, so no shoulder can
                // fold back and solarise a highlight.
                _mm256_storeu_ps(pD + x, algoCurveLutV(arg, lut));
            }

            if (tailN > 0)
            {
                const __m256 e =
                    _mm256_max_ps(_mm256_maskload_ps(pE + x, vTail), vFloor);

                const __m256 logE =
                    _mm256_mul_ps(FastCompute::AVX2::Log(e), vInvLn10);

                _mm256_maskstore_ps(pL + x, vTail, logE);

                const __m256 arg = reversal
                    ? _mm256_sub_ps(_mm256_setzero_ps(),
                                    _mm256_add_ps(logE, vTrim))
                    : logE;

                _mm256_maskstore_ps(pD + x, vTail, algoCurveLutV(arg, lut));
            }
        }
    }

    return;
}


// ---------------------------------------------------------------------------
//  Sub-stage 8b: interimage effects
// ---------------------------------------------------------------------------
void AlgoStage08b_Interimage
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const AlgoType* RESTRICT pLogER,
    const AlgoType* RESTRICT pLogEG,
    const AlgoType* RESTRICT pLogEB,
    AlgoType* RESTRICT       pScrDR,
    AlgoType* RESTRICT       pScrDG,
    AlgoType* RESTRICT       pScrDB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const HighPrecType       anchor[3]
) noexcept
{
    const film::InterimageSpec& iie = profile.interimage;

    // A monochrome stock has one emulsion layer, so there is nothing for an
    // inhibitor to diffuse INTO. The copy is required rather than a skip: the
    // retained-buffer policy gives this stage its own destination, and leaving it
    // unwritten would put stale contents in the chain for everything downstream.
    if ((false == iie.active()) || profile.is_monochrome)
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    const film::RGBCurves& curves   = profile.curves;
    const bool             reversal = profile.isReversal();

    // ----------------------------------------------------------------------
    //  Coupling matrix.
    //
    //  Row c is the layer being affected, column j the layer doing the
    //  inhibiting. The diagonal is zero by definition - a layer does not inhibit
    //  itself, that is just its own gamma - so it is never stored and never read.
    //
    //  The two FAR pairs, red-blue and blue-red, are physically weaker than the
    //  adjacent pairs, because the inhibitor has to cross the intervening layer.
    //  The spec carries them separately for that reason.
    // ----------------------------------------------------------------------
    const AlgoType m[3][3] =
    {
        { ALGO_ZERO,
          static_cast<AlgoType>(iie.a_rg),    // green inhibits red
          static_cast<AlgoType>(iie.a_rb) },  // blue  inhibits red (far pair)

        { static_cast<AlgoType>(iie.a_gr),    // red   inhibits green
          ALGO_ZERO,
          static_cast<AlgoType>(iie.a_gb) },  // blue  inhibits green

        { static_cast<AlgoType>(iie.a_br),    // red   inhibits blue (far pair)
          static_cast<AlgoType>(iie.a_bg),    // green inhibits blue
          ALGO_ZERO }
    };

    // ----------------------------------------------------------------------
    //  Mid-grey reference densities.
    //
    //  The density each layer reaches at the neutral anchor. Subtracting it is
    //  what makes the stage a colour effect and not a tone effect: on a neutral
    //  every difference is about zero and the whole correction vanishes.
    //
    //  For a reversal stock the argument is the negated trim, exactly matching how
    //  stage 8 evaluated the curve. For a negative the reference is at zero log
    //  exposure, because mid grey is 1.0 in relative exposure by construction and
    //  log10 of 1 is 0.
    // ----------------------------------------------------------------------
    AlgoType dRef[3];

    for (int32_t c = 0; c < 3; c++)
    {
        const HighPrecType arg = reversal ? -anchor[c] : 0.0;

        dRef[c] = static_cast<AlgoType>(
                      AlgoDensityScalar(arg, curveOf(curves, c)));
    }

    // Weighting exponent on the neighbouring layer's density. Zero gives uniform
    // coupling across the curve, which is chromogenic negative behaviour; greater
    // than zero concentrates the coupling where the neighbour is dense, which is
    // reversal behaviour.
    const AlgoType dw = static_cast<AlgoType>(iie.density_weighting);

    const bool wantWeight = (dw > ALGO_ZERO);

    // Reciprocals of the reference densities, formed once, for the weighting
    // normalisation. Floored so a stock with essentially no base fog does not
    // divide by zero.
    AlgoType invRef[3];

    for (int32_t c = 0; c < 3; c++)
        invRef[c] = ALGO_ONE / MAX_VALUE(dRef[c], ALGO_IIE_REF_FLOOR);

    // Per-channel curve parameters and reversal trims, hoisted out of the pixel
    // loops. They are stored as float while the arithmetic runs in AlgoType, and
    // the compiler cannot prove the profile is unchanged by the stores into the
    // destination, so it would otherwise reload and convert them per pixel.
    AlgoType dmin[3], gamma[3], toeX[3], toeK[3], shX[3], shK[3], trim[3];

    for (int32_t c = 0; c < 3; c++)
    {
        const film::ToneCurve& cv = curveOf(curves, c);

        dmin [c] = static_cast<AlgoType>(cv.dmin);
        gamma[c] = static_cast<AlgoType>(cv.gamma);
        toeX [c] = static_cast<AlgoType>(cv.toe_x);
        toeK [c] = static_cast<AlgoType>(cv.toe_k);
        shX  [c] = static_cast<AlgoType>(cv.shoulder_x);
        shK  [c] = static_cast<AlgoType>(cv.shoulder_k);
        trim [c] = reversal ? static_cast<AlgoType>(anchor[c]) : ALGO_ZERO;
    }

    // The six per-channel curve scalars hoisted above are what the tables were
    // BUILT from and are no longer read per pixel. Retained rather than deleted so
    // the profile fields driving this stage stay visible at the point of use.
    (void)dmin; (void)gamma; (void)toeX; (void)toeK; (void)shX; (void)shK;

    const AlgoType* RESTRICT logEPlane[3] = { pLogER, pLogEG, pLogEB };
    AlgoType* RESTRICT       dstPlane [3] = { pDstR,  pDstG,  pDstB  };
    AlgoType* RESTRICT       diffPlane[3] = { pScrDR, pScrDG, pScrDB };

    // ----------------------------------------------------------------------
    //  The three curves, tabulated once for the WHOLE STAGE.
    //
    //  This is where the table pays for itself. The pixel work below is
    //  iterations x 3 channels curve evaluations per pixel - up to twelve on the
    //  profiles in this database - and every one of them was two softplus
    //  ramps, so four transcendentals. All of that becomes one gather plus one
    //  FMA against a table built three times per frame.
    //
    //  Built BEFORE the iteration loop, not inside it: the curve does not change
    //  between iterations, only its argument does.
    //
    //  24 KB on the stack, released when the stage returns. Not a static, so the
    //  engine stays reentrant across concurrent frames.
    // ----------------------------------------------------------------------
    AlgoCurveLut lut[3];

    for (int32_t c = 0; c < 3; c++)
        buildCurveLut(curveOf(curves, c), lut[c]);

    // Row geometry for the vector loops, computed once. The active width is not
    // generally a multiple of eight, so a masked tail follows every full pass.
    const int32_t vecCount = sizeX / ALGO_AVX2_LANES;
    const int32_t tailN    = sizeX - (vecCount * ALGO_AVX2_LANES);

    const __m256i vTail = algoTailMask(tailN);

    // Seed the iteration with the densities stage 8 computed.
    AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    // ----------------------------------------------------------------------
    //  Fixed-point iteration.
    //
    //  Density depends on the corrected log exposure, which depends on density, so
    //  the system is implicit. The iteration count comes from the profile rather
    //  than being hardcoded because each pass is a full curve evaluation on every
    //  pixel of every channel - the most expensive thing in the chain - and a
    //  stock whose coefficients are small converges in one pass and should not pay
    //  for four.
    // ----------------------------------------------------------------------
    for (int32_t iter = 0; iter < iie.iterations; iter++)
    {
        // ------------------------------------------------------------------
        //  Difference planes, built from the CURRENT densities.
        //
        //  All three must exist before any channel is updated, because each
        //  channel's correction reads the other two. Updating in place channel by
        //  channel would feed a half-updated state into the remaining channels and
        //  turn a symmetric system into an order-dependent one.
        // ------------------------------------------------------------------
        for (int32_t j = 0; j < 3; j++)
        {
            const AlgoType ref    = dRef[j];
            const AlgoType invRj  = invRef[j];

            for (int32_t y = 0; y < sizeY; y++)
            {
                const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

                const AlgoType* RESTRICT pD = dstPlane [j] + off;
                AlgoType* RESTRICT       pE = diffPlane[j] + off;

                int32_t x = 0;

                if (wantWeight)
                {
                    // Weight rising with the neighbour's own density, and
                    // normalised so that it is exactly one AT the reference. That
                    // normalisation is what keeps a neutral untouched under either
                    // mechanism, which is the property the whole stage exists to
                    // have.
                    //
                    // Two FMAs: w = (1-dw) + dw*invRj*D, then e = (D-ref)*w.
                    const __m256 vBase  = _mm256_set1_ps(ALGO_ONE - dw);
                    const __m256 vSlope = _mm256_set1_ps(dw * invRj);
                    const __m256 vRef   = _mm256_set1_ps(ref);

                    for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
                    {
                        const __m256 d = _mm256_loadu_ps(pD + x);
                        const __m256 w = _mm256_fmadd_ps(vSlope, d, vBase);

                        _mm256_storeu_ps(pE + x,
                            _mm256_mul_ps(_mm256_sub_ps(d, vRef), w));
                    }

                    if (tailN > 0)
                    {
                        const __m256 d = _mm256_maskload_ps(pD + x, vTail);
                        const __m256 w = _mm256_fmadd_ps(vSlope, d, vBase);

                        _mm256_maskstore_ps(pE + x, vTail,
                            _mm256_mul_ps(_mm256_sub_ps(d, vRef), w));
                    }
                }
                else
                {
                    const __m256 vRef = _mm256_set1_ps(ref);

                    for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
                        _mm256_storeu_ps(pE + x,
                            _mm256_sub_ps(_mm256_loadu_ps(pD + x), vRef));

                    if (tailN > 0)
                        _mm256_maskstore_ps(pE + x, vTail,
                            _mm256_sub_ps(_mm256_maskload_ps(pD + x, vTail),
                                          vRef));
                }
            }
        }

        // ------------------------------------------------------------------
        //  Re-evaluate each channel's curve against its corrected exposure.
        // ------------------------------------------------------------------
        for (int32_t c = 0; c < 3; c++)
        {
            // Which neighbours actually couple into this channel. Skipping a zero
            // coefficient removes a whole plane read per pixel, and zeros are
            // common: several profiles carry the adjacent pairs only.
            const AlgoType m0 = m[c][0];
            const AlgoType m1 = m[c][1];
            const AlgoType m2 = m[c][2];

            const AlgoType* RESTRICT pL  = logEPlane[c];
            const AlgoType* RESTRICT pE0 = diffPlane[0];
            const AlgoType* RESTRICT pE1 = diffPlane[1];
            const AlgoType* RESTRICT pE2 = diffPlane[2];

            AlgoType* RESTRICT pO = dstPlane[c];

            const AlgoType dm = dmin[c];
            const AlgoType gm = gamma[c];
            const AlgoType tx = toeX[c];
            const AlgoType tk = toeK[c];
            const AlgoType sx = shX[c];
            const AlgoType sk = shK[c];
            const AlgoType tr = trim[c];

            // Vector frame constants for this channel's coupling row.
            const __m256 vM0 = _mm256_set1_ps(m0);
            const __m256 vM1 = _mm256_set1_ps(m1);
            const __m256 vM2 = _mm256_set1_ps(m2);
            const __m256 vTr = _mm256_set1_ps(tr);

            const AlgoCurveLut& lutC = lut[c];

            for (int32_t y = 0; y < sizeY; y++)
            {
                const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

                const AlgoType* RESTRICT rL  = pL  + off;
                const AlgoType* RESTRICT r0  = pE0 + off;
                const AlgoType* RESTRICT r1  = pE1 + off;
                const AlgoType* RESTRICT r2  = pE2 + off;

                AlgoType* RESTRICT rO = pO + off;

                int32_t x = 0;

                for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
                {
                    // Total inhibition this layer receives from the other two,
                    // three FMAs. The diagonal coefficient is zero, so including
                    // all three terms costs one multiply-add and removes a branch
                    // that would otherwise differ per channel.
                    __m256 adj = _mm256_mul_ps(vM0, _mm256_loadu_ps(r0 + x));
                    adj = _mm256_fmadd_ps(vM1, _mm256_loadu_ps(r1 + x), adj);
                    adj = _mm256_fmadd_ps(vM2, _mm256_loadu_ps(r2 + x), adj);

                    const __m256 le = _mm256_loadu_ps(rL + x);

                    // Back into the curve. A reversal stock negates the trimmed
                    // log exposure and the correction is subtracted OUTSIDE that
                    // negation, because inhibition reduces development in both
                    // cases and the sign of the density response is what differs.
                    // Branch on a frame constant, so it is hoisted out of the loop.
                    const __m256 arg = reversal
                        ? _mm256_sub_ps(_mm256_sub_ps(_mm256_setzero_ps(),
                                                      _mm256_add_ps(le, vTr)),
                                        adj)
                        : _mm256_add_ps(le, adj);

                    // The whole reason this stage is now affordable: one gather in
                    // place of two softplus evaluations, on every channel of every
                    // iteration.
                    _mm256_storeu_ps(rO + x, algoCurveLutV(arg, lutC));
                }

                if (tailN > 0)
                {
                    __m256 adj = _mm256_mul_ps(vM0,
                                     _mm256_maskload_ps(r0 + x, vTail));
                    adj = _mm256_fmadd_ps(vM1,
                              _mm256_maskload_ps(r1 + x, vTail), adj);
                    adj = _mm256_fmadd_ps(vM2,
                              _mm256_maskload_ps(r2 + x, vTail), adj);

                    const __m256 le = _mm256_maskload_ps(rL + x, vTail);

                    const __m256 arg = reversal
                        ? _mm256_sub_ps(_mm256_sub_ps(_mm256_setzero_ps(),
                                                      _mm256_add_ps(le, vTr)),
                                        adj)
                        : _mm256_add_ps(le, adj);

                    _mm256_maskstore_ps(rO + x, vTail,
                                        algoCurveLutV(arg, lutC));
                }
            }
        }
    }

    return;
}
