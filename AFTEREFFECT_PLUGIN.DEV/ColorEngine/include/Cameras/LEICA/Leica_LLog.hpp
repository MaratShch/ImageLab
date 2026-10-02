/*
 * Leica_LLog.hpp
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 * Leica L-Log decode (linearize) + encode. Scene-referred reflection.
 *   decode: x = y<=cut2 ? (y-b)/a : (10^((y-f)/c) - e)/d
 *   encode: y = x<=cut1 ? a*x+b   : c*log10(d*x+e)+f
 * Gamut: Leica does not publish a dedicated wide gamut for L-Log SL-series
 * usage; footage is commonly interpreted per camera manual (Rec.2020/709
 * container) - select primaries accordingly in the IDT.
 * Constants: Leica L-Log Reference Manual, via colour-science (verified).
 * KNOWN SPEC ARTIFACT: Leica's published constants leave the two branches
 * DISCONTINUOUS at the seam (log-branch value at cut2 is ~0.0061168 vs
 * linear 0.006 - a 1.2e-4 gap in linear light). Consequently
 * encode(decode(y)) can deviate by up to ~9e-4 in a 1-code-value
 * neighborhood of y = 0.138 ONLY (elsewhere the round-trip is exact to
 * double epsilon). colour-science exhibits the identical behavior; we match
 * the published spec rather than privately "fixing" it.
 * Encoded domain: normalized [0,1] full range. Generated: 2026-07-07 08:20:28 | C++14
 */
#ifndef __GENERATED_LEICA_LLOG_HPP__
#define __GENERATED_LEICA_LLOG_HPP__
#include <cmath>
namespace Leica_LLog
{
    constexpr double cut1 = 0.006;   // linear domain
    constexpr double cut2 = 0.138;   // encoded domain
    constexpr double a = 8.0;
    constexpr double b = 0.09;
    constexpr double c = 0.27;
    constexpr double d = 1.3;
    constexpr double e = 0.0115;
    constexpr double f = 0.6;

    template<typename T>
    inline T decode(T y) noexcept
    {
        if (y <= static_cast<T>(cut2))
            return (y - static_cast<T>(b)) / static_cast<T>(a);
        return (std::pow(static_cast<T>(10), (y - static_cast<T>(f)) / static_cast<T>(c))
                - static_cast<T>(e)) / static_cast<T>(d);
    }

    template<typename T>
    inline T encode(T x) noexcept
    {
        if (x <= static_cast<T>(cut1))
            return static_cast<T>(a) * x + static_cast<T>(b);
        return static_cast<T>(c) *
               std::log10(static_cast<T>(d) * x + static_cast<T>(e))
               + static_cast<T>(f);
    }
} // namespace Leica_LLog
#endif
