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

// Stock parameters: ToneCurve, RGBCurves, PrintStock, the dye matrices.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


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
//  logEShift     per-channel reciprocity shift from AlgoReciprocityLogShift, in
//                DECADES, added to log10(exposure) before the curve sees it. All
//                zeros when the caller stated no exposure time, which is the
//                default and reproduces every earlier render bit for bit.
//                It lands on the RETAINED log-exposure plane too, so the
//                interimage stage at 8b sees the same effective exposure the
//                curve did - a real layer cannot tell the difference either.
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
    const HighPrecType       logEShift[3]
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
    const HighPrecType       scannerSpecular,
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
