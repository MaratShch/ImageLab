#pragma once

// ---------------------------------------------------------------------------
//  AlgoRelativeExposure.hpp
//
//  Stage 2 of the film simulation pipeline: relative exposure.
//
//  Converts the caller's scene-linear image into the engine's internal exposure
//  units and applies the camera exposure offset chosen by the cinematographer.
//
//  This is the first stage that touches pixels, and it establishes the unit
//  system every later stage depends on. Everything downstream - the
//  characteristic curve above all - assumes exposure is expressed in multiples
//  of a mid-grey reference, not as raw scene-linear reflectance.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

// MemHandler and the arena geometry constants.
#include "AlgoMemHandler.hpp"

// The control structure carrying exposureStops.
#include "AlgoControl.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  MID-GREY REFERENCE
//
//  0.18, i.e. 18 per cent. This is the reflectance of a standard photographic
//  grey card, the value light meters are calibrated against and the anchor the
//  whole tone scale is built around. Dividing the incoming scene-linear signal
//  by it puts an 18 per cent grey card at exactly 1.0 in exposure units, so a
//  value of 2.0 means one stop above mid grey, 0.5 one stop below, and so on.
//
//  Two reasons this normalisation belongs here and not later:
//
//    - The characteristic curve is indexed by the base-10 logarithm of exposure.
//      Anchoring the curve requires knowing where mid grey sits, and having it
//      land on 1.0 makes log10(exposure) equal 0.0 there, which is the origin
//      the curve parameters are expressed relative to.
//    - The anchor solve that positions each stock's curve is computed against
//      this same reference. If the two disagree, every stock lands at the wrong
//      density and the error looks like a bad curve rather than a bad unit.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_MID_GREY = static_cast<AlgoType>(0.18);


// ---------------------------------------------------------------------------
//  AlgoStage02_RelativeExposure
//
//  For every pixel and every channel:
//
//      dst = (src / ALGO_MID_GREY) * 2^exposureStops
//
//  src  scene-linear planar RGB supplied by the caller, already linearised, in
//       STORAGE type. This is the engine's input boundary and the one place
//       image samples are converted from ImgType up to AlgoType.
//       Values are nominally in [0, 1] for diffuse reflectances but are NOT
//       limited to it: speculars and light sources legitimately exceed 1.0 and
//       must be carried through, because the characteristic curve's shoulder
//       needs real highlight information in order to roll it off.
//
//  dst  the stage-2 buffer. Exposure units, mid grey at 1.0, unclamped.
//
//  The result is deliberately left unclamped. The only clamp in the entire
//  pipeline is the final one; clamping here would flatten highlights before the
//  emulsion ever had the chance to compress them, which is precisely the
//  difference between a filmic roll-off and a digital clip.
//
//  Parameters and buffers are NOT validated. Both have already been checked by
//  the caller, and re-checking them here would be duplicated work in the hottest
//  part of the program.
// ---------------------------------------------------------------------------
//  pSrc*   scene-linear source planes, STORAGE type. The engine input boundary
//          and the one place samples widen from ImgType to AlgoType.
//  pDst*   stage-2 output planes, exposure units, mid grey at 1.0, unclamped.
//  sizeX   active pixels per row. THE authoritative image width.
//  sizeY   active rows.
//  pitch   elements from one row start to the next, for every plane here.
void AlgoStage02_RelativeExposure
(
    const ImgType* RESTRICT pSrcR,
    const ImgType* RESTRICT pSrcG,
    const ImgType* RESTRICT pSrcB,
    AlgoType* RESTRICT      pDstR,
    AlgoType* RESTRICT      pDstG,
    AlgoType* RESTRICT      pDstB,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           pitch,
    const AlgoControls&     params
) noexcept;
