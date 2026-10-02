/*
 * Canon_CLog3.hpp
 *
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 *
 * Canon Log 3 (v1.2, Canon 2020 spec) decode + Cinema Gamut <-> XYZ (D65).
 * Scene-referred: output is reflection (x * 0.9 applied per Canon spec).
 * IMPORTANT: Canon Log 3 v1.2 constants are defined on NORMALIZED CODE
 * VALUES; input t in [0,1] normalized code value (full-range container
 * assumed - expand legal->full upstream if needed).
 * NOTE: Canon has multiple log variants (CLog / CLog2 / CLog3, and spec
 * versions v1 / v1.2). This header is Canon Log 3 v1.2 ONLY - do not apply
 * it to CLog or CLog2 material.
 * Constants: Canon 2020 specification as reproduced by colour-science
 * (verified numerically against it).
 * Generated: 2026-07-07 06:35:30 | Standard: C++14
 */

#ifndef __GENERATED_CANON_CLOG3_HPP__
#define __GENERATED_CANON_CLOG3_HPP__

#include <cmath>

namespace Canon_CLog3
{
    // Canon Log 3 v1.2 constants.
    constexpr double kLowCut   = 0.097465473;
    constexpr double kHighCut  = 0.15277891;
    constexpr double kOffLow   = 0.12783901;
    constexpr double kOffMid   = 0.12512219;
    constexpr double kOffHigh  = 0.12240537;
    constexpr double kLogDen   = 0.36726845;
    constexpr double kGain     = 14.98325;
    constexpr double kMidSlope = 1.9754798;
    constexpr double kReflect  = 0.9;

    // Decode Canon Log 3 v1.2 encoded t (normalized [0,1]) -> scene-linear
    // reflection (18% grey -> 0.18).
    template<typename T>
    inline T decode(T t) noexcept
    {
        T x;
        if (t < static_cast<T>(kLowCut))
            x = -(std::pow(static_cast<T>(10),
                           (static_cast<T>(kOffLow) - t) / static_cast<T>(kLogDen))
                  - static_cast<T>(1)) / static_cast<T>(kGain);
        else if (t <= static_cast<T>(kHighCut))
            x = (t - static_cast<T>(kOffMid)) / static_cast<T>(kMidSlope);
        else
            x = (std::pow(static_cast<T>(10),
                          (t - static_cast<T>(kOffHigh)) / static_cast<T>(kLogDen))
                 - static_cast<T>(1)) / static_cast<T>(kGain);
        return x * static_cast<T>(kReflect);
    }

    // Encode scene-linear reflection x -> Canon Log 3 v1.2 code value.
    // Branch thresholds are the decode values of kLowCut/kHighCut expressed in
    // the NON-reflection linear domain (precomputed at full double precision;
    // provenance: log_decoding_CanonLog3_v1_2(cut, out_reflection=False)):
    constexpr double kLinLow = -0.013999999898758771;
    constexpr double kLinMid =  0.014000001417377185;

    template<typename T>
    inline T encode(T x) noexcept
    {
        const T xr = x / static_cast<T>(kReflect);   // reflection -> internal
        if (xr < static_cast<T>(kLinLow))
            return static_cast<T>(-kLogDen) *
                   std::log10(-xr * static_cast<T>(kGain) + static_cast<T>(1))
                   + static_cast<T>(kOffLow);
        if (xr <= static_cast<T>(kLinMid))
            return static_cast<T>(kMidSlope) * xr + static_cast<T>(kOffMid);
        return static_cast<T>(kLogDen) *
               std::log10(xr * static_cast<T>(kGain) + static_cast<T>(1))
               + static_cast<T>(kOffHigh);
    }

    // Canon Cinema Gamut primaries, D65 white. Pair with Canon Log 2 / Log 3.
    constexpr double CinemaGamut_to_XYZ[9] =
    {
        0.71604964655152048, 0.12968347787573964, 0.10472280262441158,
        0.26126135752555479, 0.86964214575495979, -0.13090350328051448,
        -0.0096763465750205561, -0.2364816361263487, 1.3352157334612478,
    };
    constexpr double XYZ_to_CinemaGamut[9] =
    {
        1.4898182749321833, -0.26089590218374159, -0.14242652177740084,
        -0.45816657446927289, 1.2616277830502278, 0.15962363162996532,
        -0.070349667722501702, 0.22155766722563811, 0.77618160362710387,
    };

} // namespace Canon_CLog3

#endif // __GENERATED_CANON_CLOG3_HPP__
