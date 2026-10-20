#pragma once

// ---------------------------------------------------------------------------
//  AlgoDisplayWhite.hpp
//
//  Where clear film lands on the display, and how values above it roll off.
//  2026-10-20b, owner-approved ("both": defect-layer fix AND output white).
//  Shared by stage 8 (the anchor solves), stage 14 (density to display) and
//  stage 17 (the final clamp), in both trees; twin of film_sim.py's
//  DISPLAY_CLEAR_WHITE / white_dmin / highlight_shoulder.
//
//  BEFORE: each channel was normalised against its OWN clear film,
//  out_c = (10^-D - t_min_c) / (10^-Dmin_c - t_min_c), so clear print (or clear
//  slide) was exactly 1.0 in every channel - neutral pure white - and anything
//  denser on the negative than the print could follow (opaque dirt) came out as
//  code 255. The base colour of the print or slide was normalised away.
//
//  NOW:
//    1. ONE white reference for all three channels: the LIGHTEST channel's clear
//       film, Dmin_ref = min(Dmin_r, Dmin_g, Dmin_b). The other channels' clear
//       film lands below it by their own D-min difference, so the base keeps its
//       own colour (a print stock with a denser blue D-min reads warm where it is
//       clear). The overall level is not darkened by a dense base - the lightest
//       channel still defines white - which keeps the design rule of the old
//       normalisation. The anchor solves aim through the same expression, so a
//       neutral mid grey is still neutral.
//    2. Clear film lands at ALGO_DISPLAY_CLEAR_WHITE, not 1.0: 0.947 linear, code
//       249 of 255 after sRGB encoding - the plateau of the largest opaque
//       particle in the owner's reference collage (R5C1, SVEMA, 0.55 mm; core
//       median 249, edge roll-off ~80 um), i.e. where that scanner put "no light
//       through the film". PROVISIONAL: one particle on one scanner.
//    3. Above it a SHOULDER instead of a hard clip:
//           y = min(x, H) + (1 - H) d / ((1 - H) + d),   d = max(x - H, 0)
//       continuous with slope 1 at H and approaching 1.0 asymptotically, so
//       print grain, flare and anything brighter than clear film compress
//       smoothly instead of flattening into a code-255 plateau.
// ---------------------------------------------------------------------------

#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoTypes.hpp"
#include "film_profiles.hpp"

constexpr HighPrecType ALGO_DISPLAY_CLEAR_WHITE = 0.947;


// The white reference density of a final curve set: its lightest D-min.
inline HighPrecType AlgoWhiteDmin (const film::RGBCurves& c) noexcept
{
    const HighPrecType r = static_cast<HighPrecType>(c.r.dmin);
    const HighPrecType g = static_cast<HighPrecType>(c.g.dmin);
    const HighPrecType b = static_cast<HighPrecType>(c.b.dmin);

    return MIN_VALUE(r, MIN_VALUE(g, b));
}


// The shoulder above clear white, for one value already floored at zero.
template <typename T>
inline T AlgoHighlightShoulder (const T v) noexcept
{
    // (1 - H) d / ((1 - H) + d): the same curve as d / (1 + d / (1 - H)) but
    // with no product that can overflow, so it stays monotonic up to FLT_MAX.
    const T h = static_cast<T>(ALGO_DISPLAY_CLEAR_WHITE);
    const T r = static_cast<T>(1.0 - ALGO_DISPLAY_CLEAR_WHITE);
    const T d = MAX_VALUE(v - h, static_cast<T>(0));

    return MIN_VALUE(v, h) + (r * d) / (r + d);
}
