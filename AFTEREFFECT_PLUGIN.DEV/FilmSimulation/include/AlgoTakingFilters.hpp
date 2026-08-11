#pragma once

// ---------------------------------------------------------------------------
//  AlgoTakingFilters.hpp
//
//  Stage 2b of the film simulation pipeline: camera taking filters.
//
//  Applies the 3x3 mixing matrix that describes the physical colour-separation
//  filters in the camera. For an ordinary integral tripack - one strip of film
//  with three coated emulsion layers - there are no such filters and the matrix
//  is the identity. It matters for beam-splitter and bipack processes, where the
//  camera really does split the light through separate filters onto separate
//  strips of black-and-white film.
//
//  WHY THIS CANNOT BE DONE LATER
//
//  The mixing must happen in EXPOSURE, before the characteristic curve. The
//  curve sits between exposure and density and is strongly nonlinear, so a
//  matrix applied to densities afterwards is not equivalent to the same matrix
//  applied to exposures beforehand. Moving this stage past the curve does not
//  merely shift the result slightly; it removes the mechanism that produces the
//  palette in the first place.
//
//  WHY THE OFF-DIAGONAL TERMS ARE POSITIVE
//
//  A three-strip camera's taking filters have broad, heavily overlapping
//  passbands. Light of a neighbouring colour genuinely reaches a record and ADDS
//  to its exposure, so the off-diagonal coefficients are positive and the row
//  sums exceed unity. That is not a sign error and must not be "corrected".
//
//  A separate 3x3 later in the pipeline models dye impurity - unwanted
//  ABSORPTION by imperfect dyes - and that one is subtractive and carries unit
//  row sums. The two matrices have opposite conventions and must never be
//  interchanged: swapping them would turn additive filter overlap into
//  subtractive contamination and produce a plausible-looking but entirely wrong
//  colour rendering.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

// Arena views and the AlgoType arithmetic type.
#include "AlgoMemHandler.hpp"

// AlgoCopyImage and the filtering primitives, all raw-pointer based.
#include "AlgoSeparableBlur.hpp"

// FilmProfile, and the Matrix3 typedef used for taking_matrix.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  IDENTITY DETECTION TOLERANCE
//
//  1.0e-6. Used to decide whether a stock's taking matrix is the identity and
//  therefore whether the mixing arithmetic can be replaced by a straight copy.
//
//  The matrix is stored as single-precision, so its elements carry about seven
//  significant decimal digits. A tolerance of 1e-6 is loose enough to accept a
//  matrix that is the identity to within single-precision representation, and
//  tight enough to reject any real taking matrix: the smallest off-diagonal
//  coefficient in the whole film database is 0.05, four orders of magnitude
//  above this threshold. There is no matrix anywhere near the boundary, so the
//  exact value is not delicate.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_TAKING_IDENTITY_EPS = static_cast<AlgoType>(1.0e-6);


// ---------------------------------------------------------------------------
//  AlgoStage02b_TakingFilters
//
//  For every pixel:
//
//      dst_r = m[0][0]*src_r + m[0][1]*src_g + m[0][2]*src_b
//      dst_g = m[1][0]*src_r + m[1][1]*src_g + m[1][2]*src_b
//      dst_b = m[2][0]*src_r + m[2][1]*src_g + m[2][2]*src_b
//
//  where m is profile.taking_matrix.
//
//  src  stage-2 output: exposure units, mid grey at 1.0.
//  dst  the stage-2b buffer, same units.
//
//  When the matrix is the identity the data is COPIED rather than skipped. The
//  retained-buffer policy gives every stage its own destination, so leaving it
//  unwritten would leave a stale or uninitialised buffer in the chain and any
//  inspection of it would show garbage. Copying keeps every buffer in the chain
//  a valid image at the cost of one streaming pass.
//
//  Parameters and buffers are NOT validated; the caller has already done so.
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
) noexcept;
