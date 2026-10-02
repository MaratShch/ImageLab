/*
 * CCT_LUT_CIE_1931_2DEG.hpp
 *
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 *
 * Planckian-locus CCT LUT, CIE 1931 2-deg standard observer.
 * Row: { cct [K], u (CIE 1960), v (CIE 1960), Duv }.
 * Every entry lies ON the locus by construction -> Duv == 0.
 *
 * DECLARATION ONLY: the table data lives in the matching .cpp
 * (plain `extern const` C array - single authoritative copy,
 * external linkage, no per-TU duplication, no constexpr
 * machinery; one file pair serves both C++14 and C++20).
 *
 * CMF source     : CIE_xyz_1931_2deg.csv
 * Wavelengths    : 360 .. 830 nm (471 rows; uniform, plain summation (bit-compatible with the runtime builder))
 * CCT grid       : 900 .. 40000 K, step 1 K  (39101 entries)
 * Planck c2      : 0.014388 m*K (ITS-90; matches the Ohno 2013 reference locus)
 * Precision      : STRICT - every entry evaluated in 40-digit decimal
 *                  arithmetic and rounded ONCE to double: each (u,v) is
 *                  the correctly-rounded double of the exact result of
 *                  the specified formula (same quadrature, ITS-90 c2).
 *                  Constants emitted via repr() -> bit-exact reconstruction.
 * Generated on   : 2026-07-26 10:06:51
 * Standard       : C++14 and newer (plain const array, no
 *                  language-level variants needed)
 */

#ifndef __GENERATED_CCT_LUT_CIE_1931_2DEG_DECL_HPP__
#define __GENERATED_CCT_LUT_CIE_1931_2DEG_DECL_HPP__

#include <cstddef>

#ifndef IMAGELAB2_CCT_LUT_ROW_DOUBLE_SHARED
#define IMAGELAB2_CCT_LUT_ROW_DOUBLE_SHARED
namespace CctLutShared
{
    struct CctLutRow_double
    {
        double cct;   // [K]
        double u;     // CIE 1960
        double v;     // CIE 1960
        double Duv;   // 0: on-locus by construction
    };
} // namespace CctLutShared
#endif // IMAGELAB2_CCT_LUT_ROW_DOUBLE_SHARED

namespace CCT_LUT_1931_2DEG
{

    using CctLutRow_double = CctLutShared::CctLutRow_double;

    constexpr double CCT_MIN  = 900.0;
    constexpr double CCT_MAX  = 40000.0;
    constexpr double CCT_STEP = 1.0;

    constexpr std::size_t CCT_LUT_CIE_1931_2DEG_SIZE = 39101u;

    // Defined in CCT_LUT_CIE_1931_2DEG.cpp - plain const
    // array, external linkage, single authoritative copy.
    extern const CctLutRow_double CCT_LUT_CIE_1931_2DEG[CCT_LUT_CIE_1931_2DEG_SIZE];

} // namespace CCT_LUT_1931_2DEG

#endif // __GENERATED_CCT_LUT_CIE_1931_2DEG_DECL_HPP__
