#pragma once

// ---------------------------------------------------------------------------
//  AlgoFilmCoord.hpp
//
//  The film-fixed coordinate system that every defect is generated in.
//
//  THIS IS THE MOST IMPORTANT ARCHITECTURAL DECISION IN THE DEFECT SUBSYSTEM,
//  and it cannot be retrofitted. Everything else here follows from it.
//
//  DEFECTS LIVE ON THE FILM, NOT IN THE FRAME
//
//  A defect's position is (distance ALONG the film, distance ACROSS the film), in
//  millimetres, with the origin at a fixed point on the roll. The frame is then a
//  rectangular WINDOW that slides along the film, one frame pitch per frame.
//
//  Four things fall out of that for free, and every one of them is painful to add
//  later:
//
//    1. THE 90 DEGREE ROTATION. Film runs along the LONG axis of a 35 mm still
//       frame and the SHORT axis of every common cine frame. So a transport
//       scratch is horizontal in a still and vertical in a movie - the classic
//       "rain". Generate in film coordinates and the rotation is automatic;
//       generate in frame coordinates and every scratch in one of the two media
//       is at right angles to reality, which is the most conspicuous possible
//       error.
//    2. DEFECTS CONTINUE ACROSS FRAME LINES. A scratch is a property of a length
//       of film, not of a frame. In film coordinates it simply extends past the
//       window.
//    3. FORMAT CHANGES COST NOTHING. The window changes size; the film does not.
//    4. FILM WEAVE BECOMES EXPRESSIBLE. Weave is the window moving, which is
//       exactly what it physically is.
//
//  THE TRANSPORT AXIS IS DERIVED, NOT AUTHORED
//
//  Which image axis the film runs along is not stored anywhere - it is deduced
//  from the format's own geometry, because the frame pitch IS the transport-axis
//  extent plus the interframe gap:
//
//      transport axis = whichever frame extent is SMALLER than the frame pitch,
//                       taking the smaller positive gap when both qualify
//
//  Verified against all fourteen formats in the database:
//
//      ff35        36.00 x 24.00  pitch 38.00   gap 2.00 / 14.00  -> WIDTH
//      imax15      70.41 x 52.63  pitch 71.25   gap 0.84 / 18.62  -> WIDTH
//      academy35   21.95 x 16.00  pitch 19.00   gap  -   /  3.00  -> HEIGHT
//      super35     24.89 x 18.66  pitch 19.00   gap  -   /  0.34  -> HEIGHT
//      16mm        10.26 x  7.49  pitch  7.62   gap  -   /  0.13  -> HEIGHT
//      super8       5.79 x  4.01  pitch  4.23   gap  -   /  0.22  -> HEIGHT
//
//  35 mm stills and IMAX run horizontally; every 35 mm cine format, 16 mm and
//  8 mm run vertically. Which is correct, and needed no new data.
//
//  SHEET AND PACK FILM HAVE NO TRANSPORT AXIS
//
//  large4x5, medium645 and the two Polaroid formats report a frame pitch of zero,
//  because a sheet is not transported past anything. They therefore have no
//  longitudinal anisotropy and no frame-to-frame film motion: each sheet is its
//  own piece of film. The transport axis defaults to width so the generators have
//  a defined basis, and the anisotropy strength is reported as zero so they draw
//  orientations isotropically.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  AlgoFilmWindow
//
//  Everything a defect generator needs to know about where this frame sits on the
//  film and how to get from millimetres to pixels. A plain aggregate, filled once
//  per frame by AlgoMakeFilmWindow and passed by const reference.
//
//  Not a "view object" in the sense the engine forbids: it carries no image data
//  and no pointers, only the frame's geometry. The rule against wrappers is about
//  hiding buffers and strides from a stage, and this hides neither.
// ---------------------------------------------------------------------------
struct AlgoFilmWindow
{
    // --- the window on the film, millimetres -------------------------------
    // alongMin/alongMax are measured ALONG the transport axis from the roll
    // origin; acrossMin/acrossMax ACROSS it, centred on zero so the film's centre
    // line is at zero and a defect at a fixed cross-film position stays put.
    HighPrecType alongMin;
    HighPrecType alongMax;
    HighPrecType acrossMin;
    HighPrecType acrossMax;

    // --- orientation -------------------------------------------------------
    // true  : the film runs along the image's X axis  (35 mm still, IMAX)
    // false : the film runs along the image's Y axis  (all common cine formats)
    bool transportAlongWidth;

    // Strength of the longitudinal anisotropy the scratch and abrasion
    // generators should apply, 0..1. Zero for sheet film, which has no transport
    // direction and therefore no reason to favour one orientation.
    AlgoType anisotropy;

    // --- scale -------------------------------------------------------------
    AlgoType pxPerMm;      // pixels per millimetre on the film
    AlgoType mmPerPx;      // its reciprocal, formed once

    // --- the frame in pixels ----------------------------------------------
    int32_t sizeX;
    int32_t sizeY;
};


// ---------------------------------------------------------------------------
//  Smallest frame pitch treated as a real transport, millimetres.
//
//  Sheet and pack formats report exactly zero. The threshold is well below any
//  real pitch (Regular 8 is 3.81 mm, the smallest in the database) so it cannot
//  misclassify a genuine gauge.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_FILM_MIN_PITCH_MM = static_cast<AlgoType>(0.5);

// ---------------------------------------------------------------------------
//  Longitudinal anisotropy strength for transported film.
//
//  1.0 means "apply the measured bias in full". The measured bias itself - a
//  3.5:1 preference for the transport axis among light-polarity curvilinear
//  features - lives as a constant in the scratch generator, not here; this field
//  only says whether the film HAS a transport direction at all.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_FILM_ANISOTROPY_TRANSPORTED = static_cast<AlgoType>(1.0);


// ---------------------------------------------------------------------------
//  AlgoMakeFilmWindow
//
//  Compute this frame's window on the film.
//
//  frameWidthMm / frameHeightMm  the picture area on the film
//  framePitchMm                  advance per frame along the film; 0 = sheet film
//  sizeX / sizeY                 render raster
//  frameIndex                    clip-relative; may be negative
//
//  The window is placed with its along-film origin at frameIndex * framePitchMm,
//  so frame 0 starts at the roll origin and negative frame indices run backwards
//  off the head of the roll - which is correct, because a clip may start anywhere.
// ---------------------------------------------------------------------------
inline AlgoFilmWindow AlgoMakeFilmWindow
(
    const AlgoType frameWidthMm,
    const AlgoType frameHeightMm,
    const AlgoType framePitchMm,
    const int32_t  sizeX,
    const int32_t  sizeY,
    const int32_t  frameIndex
) noexcept
{
    AlgoFilmWindow w{};

    w.sizeX = sizeX;
    w.sizeY = sizeY;

    // ----------------------------------------------------------------------
    //  Which image axis does the film run along?
    //
    //  The frame pitch is the transport-axis extent plus the interframe gap, so
    //  the transport extent must be SMALLER than the pitch. When both extents
    //  qualify - which happens on formats whose pitch is much larger than the
    //  frame, such as 35 mm still at 8 perforations - the correct one is the
    //  one with the smaller gap.
    // ----------------------------------------------------------------------
    const bool transported = (framePitchMm > ALGO_FILM_MIN_PITCH_MM);

    if (transported)
    {
        const AlgoType gapAlongWidth  = framePitchMm - frameWidthMm;
        const AlgoType gapAlongHeight = framePitchMm - frameHeightMm;

        const bool widthQualifies  = (gapAlongWidth  > ALGO_ZERO);
        const bool heightQualifies = (gapAlongHeight > ALGO_ZERO);

        w.transportAlongWidth =
            (widthQualifies && heightQualifies) ? (gapAlongWidth < gapAlongHeight)
          : (widthQualifies)                    ? true
                                                : false;

        w.anisotropy = ALGO_FILM_ANISOTROPY_TRANSPORTED;
    }
    else
    {
        // Sheet or pack film. No transport, so no preferred direction. Width is
        // chosen only so the generators have a defined basis vector.
        w.transportAlongWidth = true;
        w.anisotropy          = ALGO_ZERO;
    }

    // ----------------------------------------------------------------------
    //  Scale. This is the single mechanism that makes a 25 micrometre particle
    //  the same physical size on every gauge, and it is derived from the frame's
    //  WIDTH because that is what the raster width spans.
    // ----------------------------------------------------------------------
    w.pxPerMm = (frameWidthMm > ALGO_ZERO)
              ? (static_cast<AlgoType>(sizeX) / frameWidthMm)
              : ALGO_ZERO;

    w.mmPerPx = (w.pxPerMm > ALGO_ZERO) ? (ALGO_ONE / w.pxPerMm) : ALGO_ZERO;

    // ----------------------------------------------------------------------
    //  The window itself.
    //
    //  The extent along the film is whichever frame dimension the transport axis
    //  runs along; the extent across it is the other one. The across coordinate is
    //  centred on zero so that a defect at a fixed cross-film position - a
    //  transport scratch, say - keeps the same image position from frame to frame
    //  without any bookkeeping.
    // ----------------------------------------------------------------------
    const HighPrecType alongExtent  = static_cast<HighPrecType>(
        w.transportAlongWidth ? frameWidthMm  : frameHeightMm);

    const HighPrecType acrossExtent = static_cast<HighPrecType>(
        w.transportAlongWidth ? frameHeightMm : frameWidthMm);

    // For sheet film the pitch is zero, so every "frame" lands on the same patch
    // of film. That is correct: a sheet is one piece of film, and rendering the
    // same sheet twice must give the same defects.
    const HighPrecType origin = static_cast<HighPrecType>(frameIndex)
                              * static_cast<HighPrecType>(framePitchMm);

    w.alongMin  = origin;
    w.alongMax  = origin + alongExtent;
    w.acrossMin = -0.5 * acrossExtent;
    w.acrossMax =  0.5 * acrossExtent;

    return w;
}


// ---------------------------------------------------------------------------
//  AlgoFilmToPixel
//
//  Map a film coordinate to a pixel coordinate in the current window.
//
//  along/across are in millimetres in the film frame of reference; pxX/pxY come
//  back in pixels, in the image frame of reference, with the transport rotation
//  already applied. Fractional and unclamped - a defect may legitimately lie
//  outside the window, and the caller decides whether it still contributes.
// ---------------------------------------------------------------------------
inline void AlgoFilmToPixel
(
    const AlgoFilmWindow& w,
    const HighPrecType    along,
    const HighPrecType    across,
    HighPrecType&         pxX,
    HighPrecType&         pxY
) noexcept
{
    // Position within the window, in millimetres from its origin corner.
    const HighPrecType u = along  - w.alongMin;
    const HighPrecType v = across - w.acrossMin;

    const HighPrecType s = static_cast<HighPrecType>(w.pxPerMm);

    if (w.transportAlongWidth)
    {
        // Stills and IMAX: the film runs left to right across the picture, so the
        // along-film axis IS the image X axis.
        pxX = u * s;
        pxY = v * s;
    }
    else
    {
        // Every common cine format: the film runs bottom to top through the gate,
        // so the along-film axis is the image Y axis. THIS is the 90 degree
        // rotation, and it is the whole reason the coordinate system exists.
        pxX = v * s;
        pxY = u * s;
    }

    return;
}


// ---------------------------------------------------------------------------
//  AlgoFilmAlongExtentMm / AlgoFilmAcrossExtentMm
//
//  The window's extents, for a generator that needs to know how much film it is
//  covering without reasoning about the rotation itself.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoFilmAlongExtentMm (const AlgoFilmWindow& w) noexcept
{
    return w.alongMax - w.alongMin;
}

inline HighPrecType AlgoFilmAcrossExtentMm (const AlgoFilmWindow& w) noexcept
{
    return w.acrossMax - w.acrossMin;
}
