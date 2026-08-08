// =============================================================================
// LINEARIZE_LUT_SRGB_16BIT_F64.hpp  -  GENERATED, do not edit by hand.
//
// Generated : 2026-07-26 10:33:34 +0300 (UTC 2026-07-26 07:33:34)
// Regenerate with EXACTLY this command line:
//   python gen_linearize_lut_source.py --bits 16 --transfer srgb --dtype double
//
// Combined NORMALIZE + DECODE linearization table.
//   entry[i] = sRGB (IEC 61966-2-1, piecewise) decode of (i / 32767)
//   index    = RAW integer pixel code, 0..32767
//   value    = linear light, element type: double
//
// DECLARATION ONLY: the table data lives in the matching
// .cpp (plain `extern const` C array - single authoritative
// copy, external linkage, no per-TU duplication, no
// constexpr machinery; one file pair serves both C++14 and
// C++20).
//
// ACCURACY: every entry computed in 50-digit decimal arithmetic
// and rounded ONCE to the element type -> each stored value is
// the CORRECTLY-ROUNDED representation of the exact result (for
// long double the single rounding is performed by the compiler
// from a 40-significant-digit literal, correct for any platform
// long double width; note MSVC long double == double).
// =============================================================================

#ifndef __IMAGELAB2_LINEARIZE_LUT_SRGB_16BIT_F64_DECL__
#define __IMAGELAB2_LINEARIZE_LUT_SRGB_16BIT_F64_DECL__

#include <cstddef>

namespace LinLut_srgb_16bit_double
{
    constexpr std::size_t LINEARIZE_LUT_SRGB_16BIT_F64_SIZE = 32768u;

    // Defined in LINEARIZE_LUT_SRGB_16BIT_F64.cpp - plain const
    // array, external linkage, single authoritative copy.
    extern const double LINEARIZE_LUT_SRGB_16BIT_F64[LINEARIZE_LUT_SRGB_16BIT_F64_SIZE];
} // namespace LinLut_srgb_16bit_double

#endif // __IMAGELAB2_LINEARIZE_LUT_SRGB_16BIT_F64_DECL__
