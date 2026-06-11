#ifndef __IMAGE_LAB_AWB_EGRESS_BGRA8U__
#define __IMAGE_LAB_AWB_EGRESS_BGRA8U__

#include <cstdint>
#include "AlgoMemHandler.hpp"   // RGBPlanes

// -----------------------------------------------------------------------------
// Egress: planar scene-linear RGB_32f  ->  interleaved BGRA_8u (host frame).
//
//   * Encodes linear -> sRGB (OETF) so the 8-bit result is display-encoded,
//     the exact inverse of an sRGB-EOTF ingest. (If your ingest did NOT
//     linearize, build with AWB_EGRESS_APPLY_OETF=0 to skip the curve.)
//   * Clamps to [0,1] before quantizing, then rounds to [0,255].
//   * Byte order per pixel is B, G, R, A (BGRA_8u).
//   * Alpha is copied straight from the input host frame (same x,y).
//
// Layout:
//   * src planes are contiguous: index = y*sizeX + x  (matches MemHandler).
//   * host BGRA frames use a per-row pitch in BYTES (inRowBytes / outRowBytes),
//     which may be negative (e.g. bottom-up AE frames). Row base is computed as
//     base + y*rowBytes for BOTH input and output, matching the ingest side.
//   * sizeX, sizeY are in PIXELS.
// -----------------------------------------------------------------------------
void egress_RGB32f_to_BGRA8u
(
    const RGBPlanes& src,          // planar linear RGB (e.g. mem.output)
    const uint8_t*   inBGRA,       // input host frame (BGRA_8u) -- alpha source
    const int32_t    inRowBytes,   // input row pitch in bytes (may be negative)
    uint8_t*         outBGRA,      // output host frame (BGRA_8u)
    const int32_t    outRowBytes,  // output row pitch in bytes (may be negative)
    const int32_t    sizeX,
    const int32_t    sizeY
) noexcept;

#endif // __IMAGE_LAB_AWB_EGRESS_BGRA8U__
