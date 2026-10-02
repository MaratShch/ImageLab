/*
 * Fuji_FLog2.hpp
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 * Fujifilm F-Log2 decode (linearize) + encode, and F-Gamut <-> XYZ (D65).
 * Scene-referred reflection (18% grey -> 0.18). F-Gamut == Rec.2020
 * primaries (matrix included). This is F-Log2 (X-H2S era); the original
 * F-Log uses the same formula SHAPE with different constants - do not mix.
 *   decode: x = y<cut2 ? (y-f)/e : (10^((y-d)/c))/a - b/a
 *   encode: y = x<cut1 ? e*x+f   : c*log10(a*x+b)+d
 * Constants: Fujifilm F-Log2 Data Sheet, via colour-science (verified).
 * Encoded domain: normalized [0,1] full range. Generated: 2026-07-07 07:39:29 | C++14
 */
#ifndef __GENERATED_FUJI_FLOG2_HPP__
#define __GENERATED_FUJI_FLOG2_HPP__
#include <cmath>
namespace Fuji_FLog2
{
    constexpr double cut1 = 0.000889;   // linear domain
    constexpr double cut2 = 0.100686685370811;   // encoded domain
    constexpr double a = 5.555556;
    constexpr double b = 0.064829;
    constexpr double c = 0.245281;
    constexpr double d = 0.384316;
    constexpr double e = 8.799461;
    constexpr double f = 0.092864;

    template<typename T>
    inline T decode(T y) noexcept
    {
        if (y < static_cast<T>(cut2))
            return (y - static_cast<T>(f)) / static_cast<T>(e);
        return std::pow(static_cast<T>(10), (y - static_cast<T>(d)) / static_cast<T>(c))
               / static_cast<T>(a) - static_cast<T>(b) / static_cast<T>(a);
    }

    template<typename T>
    inline T encode(T x) noexcept
    {
        if (x < static_cast<T>(cut1))
            return static_cast<T>(e) * x + static_cast<T>(f);
        return static_cast<T>(c) * std::log10(static_cast<T>(a) * x + static_cast<T>(b))
               + static_cast<T>(d);
    }

    // F-Gamut (== Rec.2020 primaries), D65. Pair with F-Log/F-Log2.
    constexpr double FGamut_to_XYZ[9] =
    {
        0.63695804830129121, 0.14461690358620841, 0.16888097516417208,
        0.26270021201126698, 0.67799807151887115, 0.059301716469861952,
        4.9941065744660742e-17, 0.028072693049087445, 1.0609850577107909,
    };
    constexpr double XYZ_to_FGamut[9] =
    {
        1.7166511879712678, -0.35567078377639244, -0.2533662813736598,
        -0.66668435183248875, 1.6164812366349386, 0.015768545813911117,
        0.01763985744531079, -0.04277061325780853, 0.94210312123547391,
    };

} // namespace Fuji_FLog2
#endif
