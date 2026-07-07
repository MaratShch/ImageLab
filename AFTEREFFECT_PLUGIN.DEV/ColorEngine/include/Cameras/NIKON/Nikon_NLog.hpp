/*
 * Nikon_NLog.hpp
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 * Nikon N-Log decode (linearize) + encode, and N-Gamut <-> XYZ (D65).
 * Scene-referred reflection. Unusual curve: cube-root toe + natural-log body.
 *   decode: y = x<cut2 ? (x/a)^3 - b : exp((x-d)/c)
 *   encode: x = y<cut1 ? a*cbrt(y+b) : c*ln(y)+d
 * Constants: Nikon "N-Log Specification" white paper, via colour-science
 * (verified). Encoded domain: normalized [0,1] full range.
 * Generated: 2026-07-07 07:39:29 | C++14
 */
#ifndef __GENERATED_NIKON_NLOG_HPP__
#define __GENERATED_NIKON_NLOG_HPP__
#include <cmath>
namespace Nikon_NLog
{
    constexpr double cut1 = 0.328;   // linear domain
    constexpr double cut2 = 0.4418377321603128;   // encoded domain
    constexpr double a = 0.635386119257087;
    constexpr double b = 0.0075;
    constexpr double c = 0.1466275659824047;
    constexpr double d = 0.6050830889540567;

    template<typename T>
    inline T decode(T x) noexcept
    {
        if (x < static_cast<T>(cut2))
        {
            const T q = x / static_cast<T>(a);
            return q * q * q - static_cast<T>(b);
        }
        return std::exp((x - static_cast<T>(d)) / static_cast<T>(c));
    }

    template<typename T>
    inline T encode(T y) noexcept
    {
        if (y < static_cast<T>(cut1))
            return static_cast<T>(a) * std::cbrt(y + static_cast<T>(b));
        return static_cast<T>(c) * std::log(y) + static_cast<T>(d);
    }

    // Nikon N-Gamut primaries, D65 white. Pair with N-Log.
    constexpr double NGamut_to_XYZ[9] =
    {
        0.63695804830129121, 0.14461690358620841, 0.16888097516417208,
        0.26270021201126698, 0.67799807151887115, 0.059301716469861952,
        4.9941065744660742e-17, 0.028072693049087445, 1.0609850577107909,
    };
    constexpr double XYZ_to_NGamut[9] =
    {
        1.7166511879712678, -0.35567078377639244, -0.2533662813736598,
        -0.66668435183248875, 1.6164812366349386, 0.015768545813911117,
        0.01763985744531079, -0.04277061325780853, 0.94210312123547391,
    };

} // namespace Nikon_NLog
#endif
