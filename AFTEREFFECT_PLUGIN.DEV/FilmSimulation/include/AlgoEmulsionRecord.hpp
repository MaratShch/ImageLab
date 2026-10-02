#pragma once

// ---------------------------------------------------------------------------
//  AlgoEmulsionRecord.hpp
//
//  Stage 7 of the film simulation pipeline: collapse to the record the emulsion
//  actually holds.
//
//  Up to this point the pipeline has carried three independent exposure records,
//  which is what a modern tripack colour film really has. Not every stock does.
//  This stage decides what the emulsion physically stores.
//
//  THREE CASES
//
//  1. TRIPACK COLOUR. Three separate emulsion layers, each with its own
//     sensitiser and its own dye coupler. Nothing to collapse; the three records
//     pass through untouched.
//
//  2. MONOCHROME. One silver image. The three records are combined using the
//     STOCK'S OWN SPECTRAL SENSITIVITY, not video luma weights. This is not a
//     detail. An orthochromatic stock has a red weight near 0.02, which is
//     exactly what makes a red dress render black and a blue sky render white -
//     the single most recognisable signature of pre-1930 film. Using 0.30 for
//     red instead, as broadcast luma does, destroys it completely.
//
//  3. ADDITIVE COLOUR SCREEN. One panchromatic black-and-white emulsion behind a
//     fixed mosaic of microscopic colour filters. Dufaycolor's reseau is the case
//     modelled here: continuous red lines with blue and green squares chequered
//     between them, each colour taking roughly a third of the area. Colour is
//     recorded as a spatial pattern in a single record and is reconstructed at
//     projection by viewing back through the same grid in register.
//
//  WHY THE RESEAU BELONGS IN THE EXPOSURE DOMAIN
//
//  Light passes the filter grid BEFORE it reaches the emulsion, so the mosaic
//  must be applied to exposure, not to density. Applying it after development
//  would model a grid printed on top of a finished image, which is a different
//  object entirely.
//
//  WHY THE FILTERS ARE NOT TREATED AS PURE
//
//  The reseau filter matrix has large off-diagonal terms. A cell under the red
//  filter still records a substantial amount of green, because dyed-starch
//  filters of the 1930s overlapped heavily. That cross-talk is precisely what
//  makes additive colour look pastel. Treat the filters as pure and the result
//  is more saturated than Kodachrome, which is the opposite of what the process
//  is famous for.
//
//  THE NEUTRAL GAIN DIVISION
//
//  After the mosaic the record is divided by the mean row sum of the filter
//  matrix. This restores the mean level lost to the filters so that a neutral
//  grey comes out of the grid unchanged, which keeps the anchor solve at stage 8
//  valid - that solve works on scalars and cannot see the mask. The real speed
//  penalty of the process, about 1.7 stops, is carried by the stock's exposure
//  index instead, which is where a photographer would meet it.
//
//  WHEN THE GRID CANNOT BE RESOLVED
//
//  The Dufay pattern has structure at a third of the cell pitch vertically, so
//  it needs at least three pixels per cell to be represented at all. Below that
//  the mask quantises unevenly, the reconstruction picks up a colour bias of ten
//  to twenty per cent, and the output is aliasing noise rather than a mosaic.
//  Real scans of these stocks do moire for the same reason, but emitting
//  garbage is not a useful simulation of that, so the stage falls back to a
//  plain monochrome record.
//
//  THE MASK IS COMPUTED, NOT STORED
//
//  The mask is one-hot and depends only on the pixel coordinates and the pitch,
//  so it is derived from integer arithmetic wherever it is needed rather than
//  held in a buffer. Stage 14b, which reconstructs colour by viewing back
//  through the grid, calls the same helper and is guaranteed the same mask
//  without a plane of memory travelling between them.
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

// Stock parameters, including ReseauSpec and the spectral weights.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Smallest grid pitch, in pixels, at which the mosaic is worth building.
//
//  Three, because the Dufay geometry repeats vertically every three cell rows -
//  a red line then two chequered rows - so three pixels per cell is the absolute
//  floor at which each of the three colours can occupy a distinct sample.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_RESEAU_MIN_PITCH_PX = static_cast<AlgoType>(3.0);

// ---------------------------------------------------------------------------
//  Cell rows per vertical repeat of the Dufay pattern.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_RESEAU_BAND_PERIOD = 3;


// ---------------------------------------------------------------------------
//  Grid pitch in pixels for a given stock and render resolution.
//
//  pitch_px = px_per_mm / lines_per_mm
//
//  Returns zero when the stock has no reseau or the geometry is degenerate, so
//  a caller can use the result directly as an "is there a usable grid" test
//  against ALGO_RESEAU_MIN_PITCH_PX.
// ---------------------------------------------------------------------------
AlgoType AlgoReseauPitchPx
(
    const film::ReseauSpec& spec,
    const AlgoType          pxPerMm
) noexcept;


// ---------------------------------------------------------------------------
//  Which of the three filters covers a given pixel.
//
//  Returns 0 for red, 1 for green, 2 for blue. Exactly one filter covers each
//  pixel; there is no blending, because the physical grid is an opaque mosaic of
//  discrete dyed cells.
//
//  Geometry, matching Dufaycolor: every third cell row is a continuous red line,
//  and blue and green squares alternate in a chequer between those lines.
//
//  x, y      pixel coordinates
//  invPitch  reciprocal of the pitch in pixels, passed in so the per-pixel cost
//            is a multiply rather than a divide
// ---------------------------------------------------------------------------
int32_t AlgoReseauFilterIndex
(
    const int32_t  x,
    const int32_t  y,
    const HighPrecType invPitch
) noexcept;


// ---------------------------------------------------------------------------
//  Stage 7: collapse to the emulsion's own record.
//
//  pSrcR/G/B     linear exposure in
//  pDstR/G/B     linear exposure out; for a monochrome or mosaic stock all three
//                planes carry the same single record
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  params        user controls; reseau enables the mosaic
//  pxPerMm       render resolution, used to size the grid
//
//  No scratch planes: every case is either a copy or a pointwise combination,
//  and the mask is derived from the coordinates rather than stored.
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
) noexcept;
