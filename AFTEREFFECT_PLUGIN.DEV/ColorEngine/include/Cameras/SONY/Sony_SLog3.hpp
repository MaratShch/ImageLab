/*
 * Sony_SLog3.hpp
 *
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 *
 * Sony S-Log3 decode (inverse OETF) + S-Gamut3 / S-Gamut3.Cine <-> XYZ (D65).
 * Scene-referred: decode yields reflection; 18% grey -> 0.18.
 * Constants: Sony "Technical Summary for S-Gamut3.Cine / S-Log3" as
 * reproduced by colour-science (verified numerically against it).
 * Encoded input t is a NORMALIZED CODE VALUE in [0,1] (full range assumed;
 * apply legal->full expansion upstream if the container is legal range).
 * Generated: 2026-07-07 06:35:30 | Standard: C++14
 */

#ifndef __GENERATED_SONY_SLOG3_HPP__
#define __GENERATED_SONY_SLOG3_HPP__

#include <cmath>

namespace Sony_SLog3
{
    // Branch threshold and constants (S-Log3 specification).
    constexpr double kThreshold = 171.2102946929 / 1023.0;
    constexpr double kLinSlope  = 0.01125 / (171.2102946929 - 95.0);

    // Decode S-Log3 encoded t (normalized [0,1]) -> scene-linear reflection.
    template<typename T>
    inline T decode(T t) noexcept
    {
        if (t >= static_cast<T>(kThreshold))
            return static_cast<T>(
                (std::pow(static_cast<T>(10),
                          (t * static_cast<T>(1023) - static_cast<T>(420))
                          / static_cast<T>(261.5))
                 * static_cast<T>(0.19)) - static_cast<T>(0.01));
        return (t * static_cast<T>(1023) - static_cast<T>(95))
               * static_cast<T>(kLinSlope);
    }

    // Encode scene-linear reflection x -> S-Log3 code value (normalized),
    // per the Sony forward formula (branch at x = 0.01125 linear):
    //   y = x>=0.01125 ? (420 + log10((x+0.01)/0.19)*261.5)/1023
    //                  : (x*(171.2102946929-95)/0.01125 + 95)/1023
    template<typename T>
    inline T encode(T x) noexcept
    {
        if (x >= static_cast<T>(0.01125))
            return (static_cast<T>(420) +
                    std::log10((x + static_cast<T>(0.01)) / static_cast<T>(0.19))
                    * static_cast<T>(261.5)) / static_cast<T>(1023);
        return (x * static_cast<T>(171.2102946929 - 95.0) / static_cast<T>(0.01125)
                + static_cast<T>(95)) / static_cast<T>(1023);
    }

    // S-Gamut3 primaries, D65 white. Pair with S-Log3.
    constexpr double SGamut3_to_XYZ[9] =
    {
        0.7064827132, 0.12880104980000001, 0.1151721641,
        0.27097967080000002, 0.78660641119999997, -0.057586081999999997,
        -0.0096778453999999993, 0.0046000375000000001, 1.0941355586999999,
    };
    constexpr double XYZ_to_SGamut3[9] =
    {
        1.5073998990431192, -0.24582213740524178, -0.17161168084331921,
        -0.51815172706455204, 1.3553912409400368, 0.12587866812835699,
        0.015511698179580993, -0.0078727714392499912, 0.91191636553300137,
    };

    // S-Gamut3.Cine primaries, D65 white. Pair with S-Log3 (grading-friendly variant).
    constexpr double SGamut3Cine_to_XYZ[9] =
    {
        0.59908392079999995, 0.24892551609999999, 0.1024464902,
        0.21507582010000001, 0.88506850169999995, -0.1001443219,
        -0.0320658495, -0.0276583907, 1.1487819909999999,
    };
    constexpr double XYZ_to_SGamut3Cine[9] =
    {
        1.8467789691470291, -0.52598612292824609, -0.21054521142275334,
        -0.44415326286537299, 1.259442902857083, 0.14939997294040594,
        0.04085542111307653, 0.015640889355356992, 0.86820724869783195,
    };

} // namespace Sony_SLog3

#endif // __GENERATED_SONY_SLOG3_HPP__
