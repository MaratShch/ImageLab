/*
 * Apple_Log.hpp
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 * Apple Log Profile decode (linearize) + encode. Scene-referred.
 * Gamut: Apple Log uses Rec.2020 primaries, D65 - use the Rec.2020 matrix
 * (provided in ColorTransformMatrix.hpp / below for convenience).
 * Curve: 3 branches - power-2 log above P_t, parabolic toe, floor below R_0.
 *   P_t = sigma*(R_t - R_0)^2  (exact, compiler-evaluated)
 *   decode: R = P>=P_t ? 2^((P-delta)/gamma)-beta : P>=0 ? sqrt(P/sigma)+R_0 : R_0
 *   encode: P = R>=R_t ? gamma*log2(R+beta)+delta : R>=R_0 ? sigma*(R-R_0)^2 : 0
 * Constants: "Apple Log Profile White Paper", via colour-science (verified).
 * Generated: 2026-07-07 07:39:29 | Standard: C++14
 */
#ifndef __GENERATED_APPLE_LOG_HPP__
#define __GENERATED_APPLE_LOG_HPP__
#include <cmath>
namespace Apple_Log
{
    constexpr double R_0   = -0.05641088;
    constexpr double R_t   = 0.01;
    constexpr double sigma = 47.28711236;
    constexpr double beta  = 0.00964052;
    constexpr double gamma = 0.08550479;
    constexpr double delta = 0.69336945;
    constexpr double P_t   = sigma * (R_t - R_0) * (R_t - R_0);

    template<typename T>
    inline T decode(T P) noexcept
    {
        if (P >= static_cast<T>(P_t))
            return std::exp2((P - static_cast<T>(delta)) / static_cast<T>(gamma))
                   - static_cast<T>(beta);
        if (P >= static_cast<T>(0))
            return std::sqrt(P / static_cast<T>(sigma)) + static_cast<T>(R_0);
        return static_cast<T>(R_0);
    }

    template<typename T>
    inline T encode(T R) noexcept
    {
        if (R >= static_cast<T>(R_t))
            return static_cast<T>(gamma) * std::log2(R + static_cast<T>(beta))
                   + static_cast<T>(delta);
        if (R >= static_cast<T>(R_0))
        {
            const T dr = R - static_cast<T>(R_0);
            return static_cast<T>(sigma) * dr * dr;
        }
        return static_cast<T>(0);
    }
} // namespace Apple_Log
#endif
