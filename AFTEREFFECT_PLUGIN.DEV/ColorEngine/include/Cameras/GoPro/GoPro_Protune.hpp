/*
 * GoPro_Protune.hpp
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 * GoPro Protune (Flat) decode (linearize) + encode.
 * ACCURACY NOTE: Protune is a simple published log form (below), exact as a
 * formula - but GoPro's real capture pipeline applies additional processing,
 * so treat Protune linearization as APPROXIMATE for measurement purposes
 * (mark "(approx.)" in UI per the honesty rule). The newer GP-Log
 * (HERO 13 era) is NOT included: not yet in the verification reference
 * (colour-science 0.4.7); add from GoPro's GP-Log white paper when needed.
 * Gamut: GoPro native wide gamut is not rigorously specified; footage is
 * commonly interpreted as Rec.709/2020 per user choice.
 *   decode: x = (113^y - 1) / 112
 *   encode: y = ln(1 + 112*x) / ln(113)
 * Generated: 2026-07-07 07:39:29 | C++14
 */
#ifndef __GENERATED_GOPRO_PROTUNE_HPP__
#define __GENERATED_GOPRO_PROTUNE_HPP__
#include <cmath>
namespace GoPro_Protune
{
    template<typename T>
    inline T decode(T y) noexcept
    {
        return (std::pow(static_cast<T>(113), y) - static_cast<T>(1))
               / static_cast<T>(112);
    }

    template<typename T>
    inline T encode(T x) noexcept
    {
        return std::log1p(static_cast<T>(112) * x)
               / std::log(static_cast<T>(113));
    }
} // namespace GoPro_Protune
#endif
