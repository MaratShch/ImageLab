#pragma once

#include "AlgoTypes.hpp"

// ===========================================================================
//  ALGO 02 -- RELATIVE EXPOSURE
//
//    dst[i] = src[i] * gain,   gain = 2^exposureStops
//
//  Operates on ONE plane. Call three times, or loop the channels outside.
//  Deliberately takes raw pointers rather than your RGBPlanes/MemHandler types:
//  the kernel is then testable in isolation against the Python reference with
//  no host or SDK dependency, and the buffer plumbing stays a thin adapter.
//
//  NOT IN PLACE. src and dst must not overlap; that is what RESTRICT
//  asserts to the compiler. Passing src == dst is undefined here.
//
//  NO CLAMPING. Values stay unclamped until the single final clamp at stage 17.
//  The characteristic curve's shoulder needs real highlight information above
//  1.0 to roll off; clamping here is what makes digital highlights look
//  digital. Negative input is passed through untouched for the same reason --
//  stage 5 (halation) and stage 8 (curve) handle the floor.
// ===========================================================================

void Algo_02_Exposure_Plane
(
    const AlgoType* RESTRICT src,
    AlgoType*       RESTRICT dst,
    const std::size_t             count,      // pixels in this plane
    const AlgoType                gain
) noexcept;

// Convenience: compute the gain once per frame, not per pixel.
//   stops > 0 brightens. One stop = a factor of two, by definition.
AlgoType Algo_02_GainFromStops (const double exposureStops) noexcept;
