/*
 * ARRI_LogC3_EI.hpp
 *
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 *
 * Per-Exposure-Index (EI) parameters for the ARRI LogC3 (LogC V3) curve,
 * firmware SUP 3.x, method "Linear Scene Exposure Factor" (scene-referred).
 * Correct decode for illuminant / CCT / white-balance estimation: yields
 * scene-linear values (18%% grey -> ~0.18).
 *
 * Decode (encoded t in [0,1] -> linear x), with break = e*cut + f :
 *     x = (t > break) ? (pow(10,(t-d)/c) - b) / a
 *                     : (t - f) / e
 *
 * NOTE: the branch threshold is computed as e*cut+f at full precision
 * (not a stored rounded value); this matches ARRI's definition and keeps
 * the two curve segments continuous at the break.
 *
 * Valid EI range: 160 .. 1600. The compact parametric form does NOT hold
 * above EI 1600 (soft shoulder). If EI is unknown, ARRI recommends EI 800
 * (< ~10%% exposure-value deviation across EI 200..1600).
 *
 * Constants: ARRI ALEXA LogC Curve - Usage in VFX white paper, as
 * tabulated in colour-science (authoritative reproduction).
 * Generated on: 2026-07-02 11:38:52
 * Standard    : C++14 (no newer features used)
 */

#ifndef __GENERATED_ARRI_LOGC3_EI_HPP__
#define __GENERATED_ARRI_LOGC3_EI_HPP__

#include <array>
#include <cstddef>
#include <cmath>

namespace ARRI_LogC3
{
    // Parameters for one Exposure Index (fields per ARRI VFX white paper).
    struct Params
    {
        int    ei;
        double cut;
        double a;
        double b;
        double c;
        double d;
        double e;
        double f;
    };

    constexpr std::size_t COUNT = 11u;

    // Ordered by ascending EI. Index 7 == EI 800 (the recommended default).
    constexpr std::array<Params, COUNT> TABLE =
    { {
        {  160, 0.005561, 5.555556, 0.080216, 0.269036, 0.381991, 5.842037, 0.092778 },
        {  200, 0.006208, 5.555556, 0.076621, 0.266007, 0.382478, 5.776265, 0.092782 },
        {  250, 0.006871, 5.555556, 0.072941, 0.262978, 0.382966, 5.710494, 0.092786 },
        {  320, 0.007622, 5.555556, 0.068768, 0.259627, 0.383508, 5.637732, 0.092791 },
        {  400, 0.008318, 5.555556, 0.064901, 0.256598, 0.383999, 5.57196, 0.092795 },
        {  500, 0.009031, 5.555556, 0.060939, 0.253569, 0.384493, 5.506188, 0.0928 },
        {  640, 0.00984, 5.555556, 0.056443, 0.250219, 0.38504, 5.433426, 0.092805 },
        {  800, 0.010591, 5.555556, 0.052272, 0.24719, 0.385537, 5.367655, 0.092809 },
        { 1000, 0.011361, 5.555556, 0.047996, 0.244161, 0.386036, 5.301883, 0.092814 },
        { 1280, 0.012235, 5.555556, 0.043137, 0.24081, 0.38659, 5.229121, 0.092819 },
        { 1600, 0.013047, 5.555556, 0.038625, 0.237781, 0.387093, 5.16335, 0.092824 },
    } };

    // Exact EI, else nearest tabulated EI. EI<=0 -> EI 800 (index 7).
    inline const Params& params_for_EI(int ei) noexcept
    {
        std::size_t best = 7u;
        if (ei > 0)
        {
            int bestDiff = -1;
            for (std::size_t i = 0u; i < COUNT; ++i)
            {
                const int diff = (TABLE[i].ei > ei) ? (TABLE[i].ei - ei)
                                                    : (ei - TABLE[i].ei);
                if (bestDiff < 0 || diff < bestDiff) { bestDiff = diff; best = i; }
            }
        }
        return TABLE[best];
    }

    // Decode LogC3 encoded t (normalized [0,1]) -> scene-linear, in type T.
    template<typename T>
    inline T decode(T t, const Params& p) noexcept
    {
        const T brk = static_cast<T>(p.e) * static_cast<T>(p.cut)
                    + static_cast<T>(p.f);
        if (t > brk)
            return (std::pow(static_cast<T>(10),
                             (t - static_cast<T>(p.d)) / static_cast<T>(p.c))
                    - static_cast<T>(p.b)) / static_cast<T>(p.a);
        return (t - static_cast<T>(p.f)) / static_cast<T>(p.e);
    }

    template<typename T>
    inline T decode(T t, int ei) noexcept { return decode<T>(t, params_for_EI(ei)); }

} // namespace ARRI_LogC3

#endif // __GENERATED_ARRI_LOGC3_EI_HPP__
