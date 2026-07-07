/*
 * DJI_DLog.hpp
 *
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 *
 * DJI D-Log decode (linearize) + encode, and DJI D-Gamut <-> XYZ (D65).
 * Scene-referred (18% grey -> 0.18). This is the documented D-Log of the
 * Inspire / Mavic Cine class (DJI "D-Log White Paper").
 * NOT D-Log M (consumer bodies) - that curve is not rigorously published;
 * do not apply this decode to D-Log M material.
 *   decode: x = (y <= 0.14) ? (y-0.0929)/6.025 : (10^(3.89616*y-2.27752)-0.0108)/0.9892
 *   encode: y = (x <= 0.0078) ? 6.025*x+0.0929 : log10(0.9892*x+0.0108)*0.256663+0.584555
 * Constants: DJI D-Log white paper, as tabulated in colour-science
 * (verified numerically against it).
 * Encoded domain: normalized [0,1] full-range code values.
 * Generated: 2026-07-07 07:33:10 | Standard: C++14
 */

#ifndef __GENERATED_DJI_DLOG_HPP__
#define __GENERATED_DJI_DLOG_HPP__

#include <cmath>

namespace DJI_DLog
{
    // Branch thresholds, both AS PUBLISHED by DJI (encoded domain 0.14,
    // linear domain 0.0078). Note DJI's published encode constants are
    // rounded (0.256663 ~ 1/3.89616, 0.584555 ~ 2.27752/3.89616, 0.0078 ~
    // exact 0.0078175), so encode(decode(y)) carries a ~1e-6 residual that is
    // inherent to DJI's publication - we match the spec (and colour-science)
    // bit-for-bit rather than "improving" it, for interchange compatibility.
    constexpr double kEncCut = 0.14;
    constexpr double kLinCut = 0.0078;

    // Decode (linearize): encoded y [0,1] -> scene-linear.
    template<typename T>
    inline T decode(T y) noexcept
    {
        if (y <= static_cast<T>(kEncCut))
            return (y - static_cast<T>(0.0929)) / static_cast<T>(6.025);
        return (std::pow(static_cast<T>(10),
                         static_cast<T>(3.89616) * y - static_cast<T>(2.27752))
                - static_cast<T>(0.0108)) / static_cast<T>(0.9892);
    }

    // Encode: scene-linear x -> D-Log code value (published DJI constants).
    template<typename T>
    inline T encode(T x) noexcept
    {
        if (x <= static_cast<T>(kLinCut))
            return static_cast<T>(6.025) * x + static_cast<T>(0.0929);
        return std::log10(static_cast<T>(0.9892) * x + static_cast<T>(0.0108))
               * static_cast<T>(0.256663) + static_cast<T>(0.584555);
    }

    // DJI D-Gamut primaries, D65 white. Pair with D-Log.
    constexpr double DGamut_to_XYZ[9] =
    {
        0.6482, 0.19400000000000001, 0.1082,
        0.28299999999999997, 0.81320000000000003, -0.096199999999999994,
        -0.0183, -0.083199999999999996, 1.1902999999999999,
    };
    constexpr double XYZ_to_DGamut[9] =
    {
        1.7256172712954421, -0.43128461463188439, -0.19171752388620866,
        -0.60237083768555555, 1.3905137669530383, 0.16713765354823107,
        -0.015574609452013468, 0.090563922509223974, 0.84885946575493187,
    };

} // namespace DJI_DLog

#endif // __GENERATED_DJI_DLOG_HPP__
