/*
 * ACES_Transfer.hpp
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 * ACES working-space encodings + AP0/AP1 <-> XYZ matrices.
 *  - ACEScc : pure-log grading encoding (Academy S-2014-003)
 *  - ACEScct: log with linear toe - the common grading interchange
 *             (Academy S-2016-001); constants X_BRK/Y_BRK/A/B exact.
 *  - ACEScg : LINEAR AP1 - no transfer function; matrix only.
 * WHITE POINT: ACES white is ~D60 (xy 0.32168, 0.33767), NOT D65. The
 * matrices below embed it. Chromatically adapt before mixing with the
 * D65-based working space / CCT target.
 * Constants: Academy specifications, via colour-science (verified).
 * Generated: 2026-07-07 08:20:28 | C++14
 */
#ifndef __GENERATED_ACES_TRANSFER_HPP__
#define __GENERATED_ACES_TRANSFER_HPP__
#include <cmath>
namespace ACES_Transfer
{
    // ---------- ACEScct (S-2016-001) ----------
    constexpr double cct_X_BRK = 0.0078125;             // = 2^-7, exact
    constexpr double cct_Y_BRK = 0.155251141552511;
    constexpr double cct_A     = 10.5402377416545;
    constexpr double cct_B     = 0.0729055341958355;

    template<typename T>
    inline T decode_ACEScct(T y) noexcept
    {
        if (y > static_cast<T>(cct_Y_BRK))
            return std::exp2(y * static_cast<T>(17.52) - static_cast<T>(9.72));
        return (y - static_cast<T>(cct_B)) / static_cast<T>(cct_A);
    }

    template<typename T>
    inline T encode_ACEScct(T x) noexcept
    {
        if (x <= static_cast<T>(cct_X_BRK))
            return static_cast<T>(cct_A) * x + static_cast<T>(cct_B);
        return (std::log2(x) + static_cast<T>(9.72)) / static_cast<T>(17.52);
    }

    // ---------- ACEScc (S-2014-003) ----------
    // decode: y < (9.72-15)/17.52 -> (2^(y*17.52-9.72) - 2^-16)*2
    //         else                   2^(y*17.52-9.72);  clamp top at 65504.
    template<typename T>
    inline T decode_ACEScc(T y) noexcept
    {
        constexpr double kLowBrk = (9.72 - 15.0) / 17.52;              // exact
        T lin;
        if (y < static_cast<T>(kLowBrk))
            lin = (std::exp2(y * static_cast<T>(17.52) - static_cast<T>(9.72))
                   - static_cast<T>(1.52587890625e-05)) * static_cast<T>(2); // 2^-16
        else
            lin = std::exp2(y * static_cast<T>(17.52) - static_cast<T>(9.72));
        // top clamp: y >= (log2(65504)+9.72)/17.52  ->  65504 (half-float max)
        if (y >= static_cast<T>(1.4679963120447153))   // (log2(65504)+9.72)/17.52, full repr
            lin = static_cast<T>(65504);
        return lin;
    }

    template<typename T>
    inline T encode_ACEScc(T x) noexcept
    {
        if (x < static_cast<T>(0))
            return (std::log2(static_cast<T>(1.52587890625e-05))       // 2^-16
                    + static_cast<T>(9.72)) / static_cast<T>(17.52);
        if (x < static_cast<T>(3.0517578125e-05))                       // 2^-15, exact
            return (std::log2(static_cast<T>(1.52587890625e-05)
                              + x * static_cast<T>(0.5))
                    + static_cast<T>(9.72)) / static_cast<T>(17.52);
        return (std::log2(x) + static_cast<T>(9.72)) / static_cast<T>(17.52);
    }

    // ACES2065-1 (AP0) primaries, ACES white (~D60). Linear interchange.
    constexpr double AP0_to_XYZ[9] =
    {
        0.9525523959, 0.0, 9.36786e-05,
        0.3439664498, 0.7281660966, -0.0721325464,
        0.0, 0.0, 1.0088251844,
    };
    constexpr double XYZ_to_AP0[9] =
    {
        1.049811017540059, 0.0, -9.748450763173491e-05,
        -0.49590302315673573, 1.3733130458411404, 0.09824003606804335,
        0.0, 0.0, 0.9912520181529283,
    };

    // AP1 primaries (ACEScg/ACEScc/ACEScct), ACES white (~D60).
    constexpr double AP1_to_XYZ[9] =
    {
        0.6624541811085055, 0.13400420645643313, 0.15618768700490773,
        0.2722287167809146, 0.6740817658111483, 0.05368951740793703,
        -0.005574649490394109, 0.004060733528982825, 1.010339100312997,
    };
    constexpr double XYZ_to_AP1[9] =
    {
        1.6410233796943254, -0.3248032941847899, -0.23642469523761217,
        -0.663662858722983, 1.615331591657338, 0.016756347685530137,
        0.011721894328375376, -0.00828444199623741, 0.9883948585390218,
    };

} // namespace ACES_Transfer
#endif
