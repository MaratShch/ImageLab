/*
 * RED_Log3G10.hpp
 *
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 *
 * RED Log3G10 (v3, current) decode + REDWideGamutRGB <-> XYZ (D65).
 * Scene-referred; 18% grey -> 0.18. Encoded input t is a normalized float
 * (can be slightly <0 and >1). The v3 formula is symmetric via sign()
 * for t >= 0 branch and linear below 0.
 * Constants: RED "White Paper on REDWideGamutRGB and Log3G10" as
 * reproduced by colour-science (verified numerically against it).
 * Generated: 2026-07-07 06:35:30 | Standard: C++14
 */

#ifndef __GENERATED_RED_LOG3G10_HPP__
#define __GENERATED_RED_LOG3G10_HPP__

#include <cmath>

namespace RED_Log3G10
{
    // Log3G10 v3 constants.
    constexpr double a = 0.224282;
    constexpr double b = 155.975327;
    constexpr double c = 0.01;
    constexpr double g = 15.1927;

    // Decode Log3G10 encoded t -> scene-linear (18% grey -> 0.18).
    template<typename T>
    inline T decode(T t) noexcept
    {
        if (t < static_cast<T>(0))
            return (t / static_cast<T>(g)) - static_cast<T>(c);
        // sign(t) == +1 here; |t| == t
        return (std::pow(static_cast<T>(10), t / static_cast<T>(a))
                - static_cast<T>(1)) / static_cast<T>(b)
               - static_cast<T>(c);
    }

    // Encode scene-linear x -> Log3G10 code value, per the RED v3 forward:
    //   x' = x + c;  y = (x' < 0) ? x'*g : a*log10(x'*b + 1)
    template<typename T>
    inline T encode(T x) noexcept
    {
        const T xo = x + static_cast<T>(c);
        if (xo < static_cast<T>(0))
            return xo * static_cast<T>(g);
        return static_cast<T>(a) *
               std::log10(xo * static_cast<T>(b) + static_cast<T>(1));
    }

    // REDWideGamutRGB primaries, D65 white. Pair with Log3G10.
    constexpr double REDWideGamutRGB_to_XYZ[9] =
    {
        0.73527500000000001, 0.068609000000000003, 0.14657100000000001,
        0.286694, 0.84297900000000003, -0.12967300000000001,
        -0.079681000000000002, -0.34734300000000001, 1.5160819999999999,
    };
    constexpr double XYZ_to_REDWideGamutRGB[9] =
    {
        1.412806612336158, -0.17752236616704917, -0.15177037638116067,
        -0.48620318583506894, 1.290696210836872, 0.15740028369104153,
        -0.037138775804971595, 0.28637575955796257, 0.68767960531004935,
    };

} // namespace RED_Log3G10

#endif // __GENERATED_RED_LOG3G10_HPP__
