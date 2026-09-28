#pragma once

// ---------------------------------------------------------------------------
//  AlgoCharacteristicCurve.hpp
//
//  Stage 8 of the film simulation pipeline: exposure to density.
//
//  This is the stage that makes film film. Everything before it is optics and
//  everything after it is chemistry and printing, but the characteristic curve -
//  the Hurter and Driffield curve, plotted since 1890 - is where the medium
//  imposes its own tonality on the scene.
//
//  THE CURVE SHAPE
//
//  Density is built as the DIFFERENCE OF TWO SOFTPLUS RAMPS:
//
//      D(logE) = dmin + gamma * ( sp(logE - toe_x, toe_k)
//                               - sp(logE - shoulder_x, shoulder_k) )
//
//  with sp(x, k) = k * log(1 + exp(x/k)).
//
//  That single expression produces the entire real topology. Far below toe_x
//  both ramps are flat and the result is base plus fog. Between toe_x and
//  shoulder_x the first ramp is linear and the second still flat, giving the
//  straight line of slope gamma. Above shoulder_x the second ramp cancels the
//  first and the curve levels at Dmax. The knees at each end are the softplus
//  transitions, and their widths are toe_k and shoulder_k.
//
//  The decisive property is that this form is MONOTONIC BY CONSTRUCTION. A
//  softplus has a derivative strictly between zero and one, so the bracket has a
//  derivative in (-1, 1) and, scaled by a positive gamma, the curve can never
//  turn back on itself. A piecewise fit or a spline through measured points can
//  and does, and a non-monotonic patch in the shoulder produces a visible
//  solarised ring around every highlight.
//
//  NEGATIVE AND REVERSAL ARE NOT THE SAME OPERATION
//
//  A negative records more density where more light fell. A reversal stock - a
//  slide - records a positive image directly: more light gives LESS density. The
//  curve parameters for a reversal stock are expressed against NEGATED log
//  exposure, which means toe_x governs the HIGHLIGHT end rather than the shadow
//  end. Reading a reversal curve as if it were a negative curve inverts the
//  tonality and puts the shoulder in the shadows.
//
//  THE ANCHOR SOLVE
//
//  A neutral 18 per cent grey must land at a predictable place in the output.
//  What is free to move differs by stock type.
//
//  For a NEGATIVE the free parameter is the print exposure offset. That is
//  exactly what a laboratory sets with its printer lights and what a colourist
//  sets with a lift, and it has to be SOLVED rather than guessed: the naive
//  choice of offset equal to the mid-scale density puts grey wherever the print
//  curve happens to cross zero, which on a typical print stock is around two per
//  cent display luminance - roughly three stops too dark.
//
//  For a REVERSAL there is no print stage, so the only free parameter is the
//  exposure itself. Which is precisely the position a photographer shooting
//  transparency is in, and why they bracket.
//
//  The solve must include the taking matrix, the negative dye matrix, the print
//  dye matrix and the base tint, because all four scale neutral density before
//  it reaches the eye. Ignoring them is not a small error: a dye matrix with row
//  sums near 1.22 throws the mid tone out by more than a stop on its own, and a
//  set of Technicolor taking filters adds another thirty per cent. Because those
//  matrices couple the channels, the anchors are found by a short fixed-point
//  sweep - one channel re-solved at a time with the others frozen - which
//  converges in a handful of passes because the matrices are near-identity.
//
//  What is deliberately NOT cancelled is the colour-temperature mismatch from
//  the white balance control. A real laboratory would grade that out, but here
//  it is a creative control, and per-channel anchoring would neutralise exactly
//  the cast the user asked to see. Curve crossover and off-diagonal colour
//  mixing also survive untouched. Only the per-channel scalar throughput is
//  equalised, which is precisely what printer lights do.
//
//  WHERE THE ANCHORS ARE CONSUMED
//
//  For a reversal stock they are log-exposure trims and this stage applies them.
//  For a negative they are print offsets and the PRINT stage consumes them; this
//  stage applies no anchor at all to a negative. That asymmetry is inherent to
//  the two processes, not an inconsistency.
//
//  LOG EXPOSURE IS RETAINED
//
//  The base-ten logarithm of the exposure is written to its own planes and kept.
//  The interimage stage that follows needs it, and recomputing a logarithm for
//  every pixel of every channel is far more expensive than holding three planes.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// ALGO_SOFTPLUS_LINEAR_LIMIT -- the softplus asymptote crossover, shared with
// the halation threshold and with AlgoCurveLut.hpp. Needed here by
// AlgoMeasuredFitAt (schema v55).
#include "AlgoHalation.hpp"

// Stock parameters: ToneCurve, RGBCurves, PrintStock, the dye matrices.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t
#include <cmath>     // std::log1p, std::exp -- the measured-curve end offsets


// ---------------------------------------------------------------------------
//  Floor applied to exposure before the logarithm.
//
//  A true zero exposure has a logarithm of minus infinity, which would propagate
//  through the curve as a not-a-number. This floor is far below anything the
//  curve can distinguish - it corresponds to eight decades under mid grey, where
//  every stock is flat on its base fog - so clamping there changes no visible
//  value while removing the singularity.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_CURVE_EXPOSURE_FLOOR = static_cast<AlgoType>(1.0e-8);

// ---------------------------------------------------------------------------
//  Fixed-point sweeps in the anchor solve.
//
//  Eight. The matrices involved are near-identity, so the coupling between
//  channels is weak and the sweep converges long before this; the count is set
//  for certainty rather than tuned, because it runs once per frame on three
//  scalars and its cost is unmeasurable.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_ANCHOR_SWEEPS = 8;

// ---------------------------------------------------------------------------
//  Bisection iterations per channel per sweep.
//
//  Sixty halvings of an interval reduce it by a factor of 2^60, which takes any
//  starting bracket below the resolution of a double. This is deliberately more
//  than enough rather than minimal, for the same reason as above.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_ANCHOR_BISECTIONS = 60;

// ---------------------------------------------------------------------------
//  Half-width of the initial bisection bracket, in log exposure decades.
//
//  Eight decades either side of the starting estimate. Wide enough to contain
//  the solution for any stock in the database with a large margin, and the cost
//  of the extra width is a few of the sixty halvings above.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_ANCHOR_BRACKET = 8.0;

// ---------------------------------------------------------------------------
//  Fraction of the base tint carried into the anchor target.
//
//  A half. The residual tint of the film base is partly graded out in any real
//  workflow and partly left visible, and splitting the difference is what keeps
//  an orange-masked negative from either printing to a dead neutral - which
//  loses the mask entirely - or printing full orange.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_TINT_RESIDUAL = 0.5;


// ---------------------------------------------------------------------------
//  Scalar characteristic curve: log exposure to optical density.
//
//  The same expression the pixel loop evaluates, in HighPrecType and for one
//  value. Used by the anchor solve, and exposed so a caller can reproduce the
//  scalar chain without duplicating the formula.
// ---------------------------------------------------------------------------
HighPrecType AlgoDensityScalar
(
    const HighPrecType    logE,
    const film::ToneCurve& curve
) noexcept;


// ---------------------------------------------------------------------------
//  Per-channel exposure anchors landing a neutral 18 per cent grey on target.
//
//  profile       stock being simulated
//  pPrintStock   print stock for a negative; may be null for a reversal stock,
//                which has no print stage and never reads it
//  greyTarget    display value a neutral grey should reach, 0 to 1
//  couplerScale  user scale on the coupler strength
//  scannerSpecular  how directional the reader is, 0 diffuse to 1 condenser.
//                ⚠ THE SOLVE HAS TO SEE THE READER'S OPTICS, AND THAT IS THE
//                WHOLE REASON THIS PARAMETER EXISTS (queue C41). Callier
//                steepens the density a printer or scanner reads, and a lab
//                responds by RE-TIMING the print -- that is what printer
//                lights are for. Leave the solve blind to it and a condenser
//                setting both steepens the tone scale AND shifts mid grey,
//                the shift being the larger of the two: measured on
//                EASTMAN_DOUBLE_X_5222 at specular = 1, mid grey moves by
//                roughly a fifth of the output range against a contrast
//                change of a few per cent. One of those effects is the
//                physics; the other is the laboratory failing to do its job.
//                Pass 0 -- the default -- and the factor is exactly 1.0, so
//                every render made before stage 12b existed is reproduced.
//  blackPointStretch
//                the stage-14 black point, threaded in so the SOLVE and the
//                PIXEL PASS use one expression. 1.0 is the shipped behaviour
//                and is arithmetically the old formula; below it the stock's
//                own Dmax stops being stretched to output zero. Getting this
//                wrong does not merely shift the shadows -- it moves mid grey,
//                because the target this solver hits is expressed through the
//                same normalisation. See AlgoControl.hpp.
//  anchorOut     three results: log-exposure trims for a reversal stock, print
//                offsets for a negative
// ---------------------------------------------------------------------------
void AlgoSolveAnchors
(
    const film::FilmProfile& profile,
    const film::PrintStock*  pPrintStock,
    const HighPrecType       greyTarget,
    const HighPrecType       couplerScale,
    const HighPrecType       scannerSpecular,
    const HighPrecType       blackPointStretch,
    HighPrecType             anchorOut[3]
) noexcept;


// ---------------------------------------------------------------------------
//  Stage 8: characteristic curve.
//
//  pSrcR/G/B     linear exposure in
//  pDstR/G/B     optical density out
//  pLogER/G/B    log exposure, RETAINED for the interimage stage that follows
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  anchor        the three values returned by AlgoSolveAnchors; applied here for
//                a reversal stock, carried to the print stage for a negative
//  recipShift    per-channel reciprocity shift in DECADES, added to the log
//                exposure before the curve sees it and before the retained
//                log-exposure plane is written, so stage 8b reads the same
//                effective exposure the curve did. All zeros = inert, and the
//                addition of a floating zero is bit-exact, so a caller that
//                states no exposure time reproduces every earlier render
//                exactly. See AlgoReciprocity.hpp.
//
//  The three log-exposure planes must be distinct from the source and from the
//  destination.
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
    const HighPrecType       anchor[3],
    const HighPrecType       recipShift[3]
) noexcept;


// ---------------------------------------------------------------------------
//  Residual base-tint multiplier for one channel.
//
//  Exposed because the print stage at 13 has to aim at the same tinted targets
//  this stage aimed at, and a second private copy of the expression would be one
//  more place for the two to drift apart.
// ---------------------------------------------------------------------------
HighPrecType AlgoTintFactor
(
    const film::FilmProfile& profile,
    const int32_t            c
) noexcept;


// ---------------------------------------------------------------------------
//  Density a neutral 18 per cent grey reaches on the camera negative.
//
//  Includes the taking matrix, the flat-field coupler term and the negative's dye
//  matrix - everything the scalar chain does to a neutral before the image leaves
//  the negative. This is the starting point every subsequent printing generation
//  anchors against, so stage 13 needs it.
// ---------------------------------------------------------------------------
void AlgoNeutralMidDensity
(
    const film::FilmProfile& profile,
    const HighPrecType       couplerScale,
    HighPrecType             dMidOut[3]
) noexcept;


// ---------------------------------------------------------------------------
//  Print offsets landing a neutral grey on given display targets.
//
//  One channel re-solved at a time with the other two frozen, swept to
//  convergence, which handles the cross-channel coupling of the destination dye
//  matrix. This is the same solver AlgoSolveAnchors uses internally for a
//  negative; it is exposed because after a dupe chain the neutral density has
//  moved and the final print offsets must be re-solved against the new value.
// ---------------------------------------------------------------------------
void AlgoSolveStageOffsets
(
    const HighPrecType     dMid[3],
    const film::RGBCurves& dstCurves,
    const film::Matrix3&   dstMatrix,
    const HighPrecType     target[3],
    const HighPrecType     blackPointStretch,
    HighPrecType           offsetOut[3]
) noexcept;


// ---------------------------------------------------------------------------
//  Offsets centring a neutral grey in a duplicating stock's usable range.
//
//  An intermediate generation is never viewed, so there is no display value to
//  aim at. Aiming at the midpoint of the stock's own density range is what a
//  laboratory does with its printer lights, and it is what keeps a three or four
//  generation chain from drifting into the toe or the shoulder.
//
//  newMidOut receives the neutral density AFTER this generation, which the next
//  generation anchors against.
// ---------------------------------------------------------------------------
void AlgoSolveIntermediateOffsets
(
    const HighPrecType     dMid[3],
    const film::RGBCurves& dstCurves,
    HighPrecType           offsetOut[3],
    HighPrecType           newMidOut[3]
) noexcept;

// ===========================================================================
//  THE MEASURED CHARACTERISTIC CURVE  (schema v55, 2026-09-25c)
// ===========================================================================
//
//  ⚠⚠ ALMOST EVERY CURVE IN THIS DATABASE IS A FIT TO A DRAWING, AND FOR
//  THOSE THE FIT IS THE HONEST REPRESENTATION. A published plot was traced,
//  the trace was fitted to the six softplus parameters, and the individual
//  trace samples are an artefact of the tracing rather than a measurement:
//  storing them would dress pixel positions up as data.
//
//  ⚠ FERRANIA_P30 IS THE FIRST STOCK WHERE THAT IS NOT SO. Page 2 of its
//  sensitometric test report prints a TABLE -- 21 step-tablet steps against
//  five development times, 105 densities, to two decimals, as text. Those are
//  instrument readings. Fitting them and keeping only the fit costs
//  0.0065-0.0116 D rms across the five legs, and the error is concentrated in
//  the toe, which is exactly where a black-and-white negative's shadow
//  rendering is decided.
//
//  ⚠ THE TABLE DOES NOT REPLACE THE CLOSED FORM, IT WINS INSIDE ITS OWN
//  RANGE. The samples span only the exposures the test used -- 2.99 decades
//  here -- and a render routinely asks outside that. Below the darkest sample
//  and above the lightest the six parameters still govern, offset by a
//  constant so the two meet EXACTLY at the boundary. So the fitted toe and
//  shoulder are still load bearing, and the function is continuous.
//
//  ⚠ MONOTONICITY IS NOT ASSUMED, IT IS BUILT IN. The header of this file has
//  warned since it was written that "a piecewise fit or a spline through
//  measured points can and does [turn back on itself], and a non-monotonic
//  patch in the shoulder produces a visible solarised ring around every
//  highlight". That warning is correct, and it is why the shipped slopes are
//  FRITSCH-CARLSON: zero wherever the data turn, clamped to three times the
//  local secant elsewhere, which makes every interval monotone by
//  construction exactly as the softplus is. The slopes are computed ONCE, in
//  film_profiles._hermite_slopes, and emitted into the tables; nothing here
//  estimates a derivative, so Python, this scalar path and the AVX2 table
//  cannot drift apart in the way AlgoCurveLut.hpp's own header note describes.
//
//  ⚠ THERE IS NO USER CONTROL FOR THIS, DELIBERATELY. A measured curve is
//  DATA about a stock, on the same footing as MTFSpec::mtf_measured or
//  GrainSpec::sigma_shape_measured, not a creative choice; and adding a panel
//  control would renumber every bit in film_params_mask.hpp for a switch whose
//  only honest setting is "use the measurement". The Python reference keeps
//  RenderSettings.curve_measured as an A/B for exactly the reason
//  mtf_use_kernel exists -- so the two spellings can be compared -- and its
//  default, true, is what this path always does.
// ---------------------------------------------------------------------------

/// Per-curve constants for the measured branch. Built ONCE per curve per
/// frame, never per pixel: the two end offsets each cost two transcendentals.
struct AlgoMeasuredCtx
{
    const float* x;      ///< ascending abscissae, or nullptr
    const float* d;      ///< density at each abscissa
    const float* m;      ///< Fritsch-Carlson slope at each abscissa
    int32_t      n;      ///< sample count, 0 when inactive
    AlgoType     offLo;  ///< add to the closed form below x[0]
    AlgoType     offHi;  ///< add to the closed form above x[n-1]
};


/// The closed form alone, in HighPrecType, for building the end offsets.
/// Deliberately a SEPARATE spelling from the per-pixel AlgoSoftplus: this runs
/// twice per curve per frame, so it is written for exactness.
inline HighPrecType AlgoMeasuredFitAt
(
    const film::ToneCurve& c,
    const HighPrecType     a
) noexcept
{
    const HighPrecType tk = static_cast<HighPrecType>(c.toe_k);
    const HighPrecType sk = static_cast<HighPrecType>(c.shoulder_k);

    HighPrecType rise;
    if (tk <= static_cast<HighPrecType>(0.0))
    {
        rise = MAX_VALUE(a - static_cast<HighPrecType>(c.toe_x),
                         static_cast<HighPrecType>(0.0));
    }
    else
    {
        const HighPrecType z = (a - static_cast<HighPrecType>(c.toe_x)) / tk;
        rise = (z > static_cast<HighPrecType>(ALGO_SOFTPLUS_LINEAR_LIMIT))
             ? (a - static_cast<HighPrecType>(c.toe_x))
             : tk * std::log1p(std::exp(z));
    }

    HighPrecType fall;
    if (sk <= static_cast<HighPrecType>(0.0))
    {
        fall = MAX_VALUE(a - static_cast<HighPrecType>(c.shoulder_x),
                         static_cast<HighPrecType>(0.0));
    }
    else
    {
        const HighPrecType z = (a - static_cast<HighPrecType>(c.shoulder_x)) / sk;
        fall = (z > static_cast<HighPrecType>(ALGO_SOFTPLUS_LINEAR_LIMIT))
             ? (a - static_cast<HighPrecType>(c.shoulder_x))
             : sk * std::log1p(std::exp(z));
    }

    return static_cast<HighPrecType>(c.dmin)
         + static_cast<HighPrecType>(c.gamma) * (rise - fall);
}


/// Build the per-curve context. Returns an INACTIVE context (n = 0) for every
/// curve that carries no table, which is 199 of the 200 stocks; the per-pixel
/// helper below then costs one predictable branch.
inline AlgoMeasuredCtx AlgoMakeMeasured (const film::ToneCurve& c) noexcept
{
    AlgoMeasuredCtx mc;
    mc.x = nullptr; mc.d = nullptr; mc.m = nullptr; mc.n = 0;
    mc.offLo = ALGO_ZERO; mc.offHi = ALGO_ZERO;

    if (false == c.hasMeasured())
        return mc;

    mc.x = c.meas_x;
    mc.d = c.meas_d;
    mc.m = c.meas_m;
    mc.n = static_cast<int32_t>(c.meas_n);

    const HighPrecType xLo = static_cast<HighPrecType>(c.meas_x[0]);
    const HighPrecType xHi = static_cast<HighPrecType>(c.meas_x[c.meas_n - 1]);

    mc.offLo = static_cast<AlgoType>(
        static_cast<HighPrecType>(c.meas_d[0]) - AlgoMeasuredFitAt(c, xLo));
    mc.offHi = static_cast<AlgoType>(
        static_cast<HighPrecType>(c.meas_d[c.meas_n - 1])
        - AlgoMeasuredFitAt(c, xHi));

    return mc;
}


/// The measured branch in HighPrecType, for the SOLVES.
///
/// ⚠⚠ THIS EXISTS BECAUSE THE ANCHOR SOLVE IS NOT A PIXEL PATH. The per-pixel
/// helper below works in AlgoType, which is FLOAT in the vector build; the
/// anchor solve runs a sixty-step bisection whose bracket shrinks below float
/// resolution long before it finishes, which is why `softplusHP` exists in
/// Algo_08_Sim.cpp for exactly the same reason. Using the AlgoType helper
/// there would make the vector build's mid grey differ from the scalar
/// build's on any stock carrying a table -- a defect that would look like a
/// solver bug and would not be one.
///
/// The arithmetic is otherwise term for term the same as the AlgoType
/// version, deliberately: two spellings of one interpolant is already one
/// more than ideal, and they must not be allowed to drift.
inline HighPrecType AlgoMeasuredDensityHP
(
    const film::ToneCurve& c,
    const HighPrecType     arg,
    const HighPrecType     dFit
) noexcept
{
    if (false == c.hasMeasured())
        return dFit;

    const int32_t n = static_cast<int32_t>(c.meas_n);

    if (arg <= static_cast<HighPrecType>(c.meas_x[0]))
        return dFit + (static_cast<HighPrecType>(c.meas_d[0])
                       - AlgoMeasuredFitAt(
                             c, static_cast<HighPrecType>(c.meas_x[0])));
    if (arg >= static_cast<HighPrecType>(c.meas_x[n - 1]))
        return dFit + (static_cast<HighPrecType>(c.meas_d[n - 1])
                       - AlgoMeasuredFitAt(
                             c, static_cast<HighPrecType>(c.meas_x[n - 1])));

    int32_t lo = 0;
    int32_t hi = n - 1;
    while (hi - lo > 1)
    {
        const int32_t mid = (lo + hi) >> 1;
        if (static_cast<HighPrecType>(c.meas_x[mid]) <= arg) lo = mid; else hi = mid;
    }

    const HighPrecType h  = static_cast<HighPrecType>(c.meas_x[lo + 1])
                          - static_cast<HighPrecType>(c.meas_x[lo]);
    const HighPrecType t  = (arg - static_cast<HighPrecType>(c.meas_x[lo])) / h;
    const HighPrecType t2 = t * t;
    const HighPrecType t3 = t2 * t;

    return ( static_cast<HighPrecType>(2.0) * t3
           - static_cast<HighPrecType>(3.0) * t2
           + static_cast<HighPrecType>(1.0))
             * static_cast<HighPrecType>(c.meas_d[lo])
         + (t3 - static_cast<HighPrecType>(2.0) * t2 + t) * h
             * static_cast<HighPrecType>(c.meas_m[lo])
         + (static_cast<HighPrecType>(-2.0) * t3
           + static_cast<HighPrecType>(3.0) * t2)
             * static_cast<HighPrecType>(c.meas_d[lo + 1])
         + (t3 - t2) * h * static_cast<HighPrecType>(c.meas_m[lo + 1]);
}


/// One density. `dFit` is the closed-form density the caller already computed
/// at `arg`; this returns it unchanged when the curve carries no table.
///
/// The interval search is a bisection rather than a linear scan: 21 samples
/// today, but nothing in the schema caps the count and a linear scan would
/// quietly become the most expensive thing in stage 8 on a 200-point table.
inline AlgoType AlgoMeasuredDensity
(
    const AlgoMeasuredCtx& mc,
    const AlgoType         arg,
    const AlgoType         dFit
) noexcept
{
    if (mc.n < 2)
        return dFit;

    if (arg <= mc.x[0])
        return dFit + mc.offLo;
    if (arg >= mc.x[mc.n - 1])
        return dFit + mc.offHi;

    int32_t lo = 0;
    int32_t hi = mc.n - 1;
    while (hi - lo > 1)
    {
        const int32_t mid = (lo + hi) >> 1;
        if (mc.x[mid] <= arg) lo = mid; else hi = mid;
    }

    const AlgoType h  = mc.x[lo + 1] - mc.x[lo];
    const AlgoType t  = (arg - mc.x[lo]) / h;
    const AlgoType t2 = t * t;
    const AlgoType t3 = t2 * t;

    // Cubic Hermite on [x_lo, x_lo+1]. The four basis functions, written out
    // rather than factored, because this is the one place three engines must
    // agree term for term.
    return ( static_cast<AlgoType>(2.0) * t3
           - static_cast<AlgoType>(3.0) * t2
           + static_cast<AlgoType>(1.0)) * mc.d[lo]
         + (t3 - static_cast<AlgoType>(2.0) * t2 + t) * h * mc.m[lo]
         + (static_cast<AlgoType>(-2.0) * t3
           + static_cast<AlgoType>(3.0) * t2) * mc.d[lo + 1]
         + (t3 - t2) * h * mc.m[lo + 1];
}

