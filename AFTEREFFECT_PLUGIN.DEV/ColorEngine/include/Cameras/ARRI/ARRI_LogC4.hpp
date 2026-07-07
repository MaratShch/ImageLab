/*
 * ARRI_LogC4.hpp
 *
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 *
 * ARRI LogC4 curve decode (inverse OETF). Unlike LogC3, LogC4 is
 * EI-INDEPENDENT: a single curve for all exposure indices (only an internal
 * hardware gain changes with EI, not the encoding curve). Hence no EI table.
 *
 * Decode (encoded E_p, normalized float -> relative scene-linear E_scene):
 *     E_scene = (E_p >= 0) ? (2^(14*((E_p - c)/b) + 6) - 64) / a
 *                          :  E_p * s + t
 *
 * Scene-referred: 18%% mid grey encodes to E_p ~= 0.2784 and decodes to ~0.18.
 * The E_p < 0 branch is the linear extension for sub-black (negative) code
 * values; it is rarely exercised but included for exactness.
 *
 * Base-2 exponial: uses std::exp2 (C++11).
 *
 * Constants (a,b,c,s,t): ARRI LogC4 Specification, as tabulated in
 * colour-science (authoritative reproduction). Provenance of a,b,c,t:
 *     a = (2^18 - 16) / 117.45
 *     b = (1023 - 95) / 1023
 *     c = 95 / 1023
 *     t = (2^(14*(-c/b)+6) - 64) / a
 * Generated on: 2026-07-02 11:42:31
 * Standard    : C++14 (no newer features used)
 */

#ifndef __GENERATED_ARRI_LOGC4_HPP__
#define __GENERATED_ARRI_LOGC4_HPP__

#include <cmath>

namespace ARRI_LogC4
{
    // LogC4 curve constants (single set; EI-independent).
    // a, b, c are the EXACT spec definitions, evaluated by the compiler in
    // double precision (no transcribed decimals -> zero transcription error):
    constexpr double a = (262144.0 - 16.0) / 117.45;   // (2^18 - 16) / 117.45
    constexpr double b = (1023.0 - 95.0) / 1023.0;
    constexpr double c = 95.0 / 1023.0;
    // s and t involve 2^x with non-integer x (not constexpr-evaluable in
    // C++14), so they are stored as full-precision double literals computed
    // from the exact spec closed forms:
    //   expo0 = 14*(-c/b) + 6
    //   s = 2^expo0 * ln(2) * 14 / (a*b)     (tangent slope at E_p = 0)
    //   t = (2^expo0 - 64) / a               (decode value at E_p = 0)
    // Both verified to be exact (0 ULP) against the closed forms; s differs
    // from the colour-science tabulation by 1 ULP (this value is the exact
    // tangent, guaranteeing C1 continuity at the branch point).
    constexpr double s = 0.11359720861058911;   // slope of sub-black linear segment
    constexpr double t = -0.01805699611991131;  // offset of sub-black linear segment

    // Decode a LogC4 encoded value E_p (normalized float) to relative
    // scene-linear, computed in type T (use float or double).
    template<typename T>
    inline T decode(T E_p) noexcept
    {
        if (E_p >= static_cast<T>(0))
        {
            const T expo = static_cast<T>(14) *
                           ((E_p - static_cast<T>(c)) / static_cast<T>(b))
                           + static_cast<T>(6);
            return (std::exp2(expo) - static_cast<T>(64)) / static_cast<T>(a);
        }
        return E_p * static_cast<T>(s) + static_cast<T>(t);
    }

    // Encode relative scene-linear E_scene -> LogC4 code value, per the
    // ARRI forward formula:
    //   E_p = (E >= t) ? ((log2(a*E + 64) - 6)/14)*b + c : (E - t)/s
    template<typename T>
    inline T encode(T E_scene) noexcept
    {
        if (E_scene >= static_cast<T>(t))
            return ((std::log2(static_cast<T>(a) * E_scene + static_cast<T>(64))
                     - static_cast<T>(6)) / static_cast<T>(14)) * static_cast<T>(b)
                   + static_cast<T>(c);
        return (E_scene - static_cast<T>(t)) / static_cast<T>(s);
    }

} // namespace ARRI_LogC4

#endif // __GENERATED_ARRI_LOGC4_HPP__
