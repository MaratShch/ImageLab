#pragma once

// ---------------------------------------------------------------------------
//  AlgoDyeImpurity.hpp
//
//  Stage 12 of the film simulation pipeline: dye impurity and scanner crosstalk.
//
//  PHYSICAL BACKGROUND
//
//  A cyan dye is supposed to absorb only red light. Real cyan absorbs some green
//  and a little blue as well, and the same is true of the magenta and yellow dyes.
//  So the density a scanner reads in its red channel is not the red record's
//  density: it is that plus unwanted absorption from the other two dyes.
//
//  The same matrix also carries the scanner's own channel crosstalk, because the
//  two are indistinguishable in the measurement and are always characterised
//  together.
//
//  WHY THE CONVENTION IS THE OPPOSITE OF THE TAKING MATRIX
//
//  This trips people up and getting it wrong is subtle rather than obvious.
//
//  The TAKING matrix at stage 2b has POSITIVE off-diagonals and row sums greater
//  than one, because camera filters OVERLAP: a red filter passes some green light,
//  so the red record receives light it was not meant to and gets MORE exposure.
//  Additive.
//
//  The DYE matrix here has row sums near one by construction, because it
//  redistributes absorption rather than creating it. It is subtractive: a unit of
//  unwanted absorption in one channel is a unit that was not clean absorption in
//  another. Row sums departing from one - a stock at 1.22 exists - shift neutral
//  density outright, which is exactly why the anchor solve at stage 8 has to
//  include this matrix rather than treating it as a colour-only correction.
//
//  The two matrices must never be swapped or transposed into each other. They are
//  different physics with different conventions that happen to share a shape.
//
//  WHY IT IS AFTER GRAIN
//
//  A real scanner reads the dye through its own filters, so whatever is in the dye
//  layers - image and grain alike - is mixed by the same matrix. Grain therefore
//  acquires a slight chromatic correlation, and it should: measured grain on colour
//  negative is not channel-independent, and a model that mixes before adding grain
//  produces grain that is more independent than any real scan.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// The copy helpers used by the identity fast path.
#include "AlgoSeparableBlur.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including the dye matrix.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Tolerance for the identity test on a dye matrix.
//
//  Loose enough to catch a matrix written out as an identity in single precision,
//  tight enough that any real characterisation fails it.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_DYE_IDENTITY_EPS = static_cast<AlgoType>(1.0e-6);


// ---------------------------------------------------------------------------
//  Apply a 3x3 density mixing matrix to three planes.
//
//  Exposed because stage 13 has to apply the PRINT stock's dye matrix in exactly
//  the same way after the print curve, and a second copy of the loop would be a
//  second place for the index convention to be got wrong.
//
//  Index convention is m[out][in]: row 0 forms the red output, and within it column
//  1 is the contribution of the incoming green density.
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
) noexcept;


// ---------------------------------------------------------------------------
//  Is a matrix the identity, to within ALGO_DYE_IDENTITY_EPS?
//
//  Exposed so stage 13 can take the same fast path on the print stock's matrix.
// ---------------------------------------------------------------------------
bool AlgoIsIdentityMatrix (const film::Matrix3& m) noexcept;


// ---------------------------------------------------------------------------
//  Stage 12: dye impurity and scanner crosstalk.
//
//  pSrcR/G/B     density in
//  pDstR/G/B     density out, floored at zero
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated; dye_matrix lives here
//
//  No scratch planes: the mix is pointwise across the three channels.
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
) noexcept;
