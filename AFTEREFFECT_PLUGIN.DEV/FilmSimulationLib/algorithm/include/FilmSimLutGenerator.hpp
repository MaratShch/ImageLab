#pragma once

// ===========================================================================
//  FilmSimLutGenerator.hpp -- 3D LUT export of the film simulation
//
//  WHAT IT IS
//    FilmSimLutGenerator renders the N x N x N lattice of a 3D LUT through the
//    SAME engine entry point as a frame, Algorithm_Main, with the SAME
//    AlgoControls the host fills for a render, and returns N^3 RGB triplets in
//    .cube order (red varying fastest). FilmSimLutToCubeText turns them into the
//    text of an Adobe / Resolve .cube file, header included. Nothing here writes
//    a file; the host saves the text.
//
//  WHAT A LUT CAN HOLD
//    A 3D LUT maps one colour to one colour. It cannot see neighbours, the frame
//    number or randomness, so every entry is the engine's response to a LARGE
//    UNIFORM PATCH of its colour:
//      included -- exposure, taking filters / spectral taking, colour balance,
//                  reciprocity, ageing, development / push / variants, the
//                  curves, interimage and the Fuji fourth layer, the large-area
//                  part of the DIR couplers, bromide drag at its flat-field
//                  level, dye impurity, Callier, dupe generations, print stock,
//                  silver tone, the final display roll-off;
//      excluded -- grain and print grain, halation, every MTF / blur, scan
//                  misregistration, veiling flare, vignette, coating field /
//                  gate buckle / edge fog, the reseau mosaic, and all film
//                  damage, gate weave and flicker.
//    The excluded controls are forced off on a private copy of the controls;
//    the spatial transfers apply their zero-frequency gain through a private
//    copy of the MemHandler (MemHandler::lutMode, AlgoFreqState::dcOnly). The
//    caller's controls and handler are not changed (its planes are used as
//    scratch, see MEMORY).
//    Twin: lut_export.py (Python reference), held to it by lut_parity.py.
//
//  MEMORY
//    The MemHandler is the caller's frame handler (alloc_memory_buffers), used
//    as scratch: the lattice is rendered in chunks of activeW x activeH colours,
//    so any frame size works (a larger frame means fewer engine calls). The
//    source and destination planes are OVERWRITTEN -- refill them before the
//    next render -- and the call must not run concurrently with a render on the
//    same handler. The returned vector is the only allocation.
//
//  ERRORS
//    An empty vector: no buffers in the handler, a film index outside the
//    database, an unknown enumerator, or the vector could not be allocated.
//    No exception leaves either function.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "AlgoMemHandler.hpp"
#include "AlgoControl.hpp"

//: Lattice edge. small 17 / medium 33 (default) / big 65 / huge 96 points.
enum class LutSizeCtrl : int32_t
{
    eLUT_SIZE_SMALL = 0,    // 17  -- monitors, on-set boxes
    eLUT_SIZE_MEDIUM,       // 33  -- the interchange standard (default)
    eLUT_SIZE_BIG,          // 65  -- grading
    eLUT_SIZE_HUGE,         // 96  -- masters; not every host loads it
    eLUT_SIZE_TOTAL
};

//: Encoding of the LUT's input code values.
enum class LutInputSpaceCtrl : int32_t
{
    eLUT_IN_REC709 = 0,     // Rec.709 video, inverse BT.709 OETF (0.409 = 18 % grey)
    eLUT_IN_SRGB,           // sRGB, IEC 61966-2-1 decode (0.461 = 18 % grey)
    eLUT_IN_ACESCCT,        // ACEScct (S-2016-001), AP1 -> Rec.709 (Bradford D60 -> D65)
    eLUT_IN_TOTAL
};

//: Encoding of the LUT's output code values.
enum class LutOutputSpaceCtrl : int32_t
{
    eLUT_OUT_REC709 = 0,    // Rec.709 / BT.1886 display, gamma 2.4
    eLUT_OUT_SRGB,          // sRGB display, IEC 61966-2-1 encode
    eLUT_OUT_TOTAL
};

//: Interpolation the LUT is INTENDED for. Metadata only: every entry is an exact
//: render of its lattice colour and the host interpolates, so this does not
//: change a single value; it is written into the .cube header.
enum class LutInterpolationCtrl : int32_t
{
    eLUT_INTERP_TRILINEAR = 0,
    eLUT_INTERP_TETRAHEDRAL,
    eLUT_INTERP_TOTAL
};

struct LutGeneratorControls
{
    LutSizeCtrl          size;
    LutInputSpaceCtrl    inputSpace;
    LutOutputSpaceCtrl   outputSpace;
    LutInterpolationCtrl interpolation;
};

//: medium (33), Rec.709 in, Rec.709 out, trilinear -- lut_export.py's defaults.
LutGeneratorControls getLutGeneratorControlsDefault () noexcept;

//: 17 / 33 / 65 / 96, or 0 for an unknown enumerator.
int32_t FilmSimLutEdge (LutSizeCtrl size) noexcept;

//: One .cube line: output code values, 0..1, double precision.
struct LutRGB
{
    double r;
    double g;
    double b;
};

// ---------------------------------------------------------------------------
//  FilmSimLutGenerator
//
//  memHandler  the caller's frame handler, used as scratch (see MEMORY above)
//  algoCtrl    filled exactly as for Algorithm_Main (stock, print stock,
//              exposure, development, ageing, white balance, ...)
//  lutCtrl     size, input / output encoding, intended interpolation
//
//  Returns edge^3 triplets, red fastest, or an empty vector (see ERRORS).
// ---------------------------------------------------------------------------
std::vector<LutRGB> FilmSimLutGenerator (const MemHandler&           memHandler,
                                         const AlgoControls&         algoCtrl,
                                         const LutGeneratorControls& lutCtrl);

// ---------------------------------------------------------------------------
//  FilmSimLutToCubeText
//
//  The .cube text of a LUT produced by FilmSimLutGenerator with the same
//  arguments: TITLE, the generation date, the copyright notice, the stock, the
//  encodings, the size, the intended interpolation, the controls that differ
//  from getAlgoControlsDefault(), the excluded stages, LUT_3D_SIZE,
//  DOMAIN_MIN / DOMAIN_MAX and one "r g b" line per entry with 10 decimals.
//  Locale-independent ('.' decimal point whatever the host's locale).
//
//  dateIso  "YYYY-MM-DD" to print, or nullptr for today's local date.
//
//  Returns an empty string when lut is not edge^3 entries for lutCtrl.size.
// ---------------------------------------------------------------------------
std::string FilmSimLutToCubeText (const std::vector<LutRGB>&  lut,
                                  const MemHandler&           memHandler,
                                  const AlgoControls&         algoCtrl,
                                  const LutGeneratorControls& lutCtrl,
                                  const char*                 dateIso);
