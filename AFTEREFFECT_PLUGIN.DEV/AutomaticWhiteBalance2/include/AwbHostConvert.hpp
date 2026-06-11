#ifndef __IMAGE_LAB_AWB_HOST_CONVERT__
#define __IMAGE_LAB_AWB_HOST_CONVERT__

#include "CommonPixFormat.hpp"   // PF_Pixel_BGRA_8u, A_long, A_u_char
#include "AlgoMemHandler.hpp"    // MemHandler, RGBPlanes

#ifndef RESTRICT
#define RESTRICT __restrict
#endif

// -----------------------------------------------------------------------------
// Host BGRA_8u  <->  algorithm planar RGB_32f (scene-linear, [0,1]) converters.
//
// All sizes and pitches are in PIXELS / ELEMENTS (never bytes). Every pitch is
// signed and may differ from sizeX (host frames can be bottom-up => negative
// pitch; planar buffers may be padded => planarPitch == padW, not sizeX).
// Row base is always   base + (ptrdiff_t)y * pitch ,  so negative pitch works.
//
// Encoding: ingress applies the sRGB EOTF (8u gamma -> scene-linear float);
// egress applies the inverse OETF (linear -> 8u gamma). Build with
// -DAWB_HOST_APPLY_GAMMA=0 to bridge straight 0..255 <-> 0..1 with no curve
// (matches the old non-linear interleaved path).
// -----------------------------------------------------------------------------

// INGRESS: host BGRA_8u  ->  mem.input planar linear RGB_32f.
void Convert_BGRA8u_to_LinearPlanar
(
    MemHandler&                      mem,          // dst planes = mem.input.{R,G,B}
    const PF_Pixel_BGRA_8u* RESTRICT inBGRA,       // host source frame
    const A_long                     sizeX,
    const A_long                     sizeY,
    const A_long                     inPitch,      // BGRA src pitch (pixels, signed)
    const A_long                     planarPitch   // planar dst pitch (pixels, signed; e.g. padW)
) noexcept;

// EGRESS: mem.output planar linear RGB_32f  ->  host BGRA_8u.
// Alpha is copied from inBGRA at the same (x,y).
void Convert_LinearPlanar_to_BGRA8u
(
    const MemHandler&                mem,          // src planes = mem.output.{R,G,B}
    const PF_Pixel_BGRA_8u* RESTRICT inBGRA,       // alpha source (host input frame)
    PF_Pixel_BGRA_8u*       RESTRICT outBGRA,      // host destination frame
    const A_long                     sizeX,
    const A_long                     sizeY,
    const A_long                     inPitch,      // BGRA alpha-source pitch (pixels, signed)
    const A_long                     outPitch,     // BGRA dst pitch (pixels, signed)
    const A_long                     planarPitch   // planar src pitch (pixels, signed; e.g. padW)
) noexcept;

#endif // __IMAGE_LAB_AWB_HOST_CONVERT__
