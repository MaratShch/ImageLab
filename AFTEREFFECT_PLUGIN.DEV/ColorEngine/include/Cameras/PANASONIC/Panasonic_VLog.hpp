/*
 * Panasonic_VLog.hpp
 *
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 *
 * Panasonic V-Log decode (linearize) + encode, and V-Gamut <-> XYZ (D65).
 * Scene-referred REFLECTION (18% grey -> 0.18).
 * NOTE: V-Log L (GH/S consumer bodies) is the SAME curve with a truncated
 * dynamic range - this header covers both; only the valid input span differs.
 *   decode: x = (y < cut2) ? (y-0.125)/5.6 : 10^((y-d)/c) - b
 *   encode: y = (x < cut1) ? 5.6*x+0.125  : c*log10(x+b) + d
 * Constants: Panasonic "V-Log/V-Gamut Reference Manual", as tabulated in
 * colour-science (verified numerically against it).
 * Encoded domain: normalized [0,1] full-range code values.
 * Generated: 2026-07-07 07:33:10 | Standard: C++14
 */

#ifndef __GENERATED_PANASONIC_VLOG_HPP__
#define __GENERATED_PANASONIC_VLOG_HPP__

#include <cmath>

namespace Panasonic_VLog
{
    constexpr double cut1 = 0.01;      // linear/log switch, LINEAR domain (encode)
    constexpr double cut2 = 0.181;     // linear/log switch, ENCODED domain (decode)
    constexpr double b    = 0.00873;
    constexpr double c    = 0.241514;
    constexpr double d    = 0.598206;

    // Decode (linearize): encoded y [0,1] -> scene-linear reflection.
    template<typename T>
    inline T decode(T y) noexcept
    {
        if (y < static_cast<T>(cut2))
            return (y - static_cast<T>(0.125)) / static_cast<T>(5.6);
        return std::pow(static_cast<T>(10),
                        (y - static_cast<T>(d)) / static_cast<T>(c))
               - static_cast<T>(b);
    }

    // Encode: scene-linear reflection x -> V-Log code value.
    template<typename T>
    inline T encode(T x) noexcept
    {
        if (x < static_cast<T>(cut1))
            return static_cast<T>(5.6) * x + static_cast<T>(0.125);
        return static_cast<T>(c) *
               std::log10(x + static_cast<T>(b)) + static_cast<T>(d);
    }

    // Panasonic V-Gamut primaries, D65 white. Pair with V-Log / V-Log L.
    constexpr double VGamut_to_XYZ[9] =
    {
        0.67964400000000003, 0.15221100000000001, 0.1186,
        0.26068599999999997, 0.77489399999999997, -0.035580000000000001,
        -0.0093100000000000006, -0.0046119999999999998, 1.1029800000000001,
    };
    constexpr double XYZ_to_VGamut[9] =
    {
        1.589012923567785, -0.31320394337041346, -0.18096495769665685,
        -0.53405454173740552, 1.3960122436883715, 0.10245787256386202,
        0.011179396518452884, 0.0031936025631581892, 0.90553568691373243,
    };

} // namespace Panasonic_VLog

#endif // __GENERATED_PANASONIC_VLOG_HPP__
