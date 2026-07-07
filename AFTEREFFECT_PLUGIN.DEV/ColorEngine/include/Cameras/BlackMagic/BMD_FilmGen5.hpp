/*
 * BMD_FilmGen5.hpp
 *
 * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!
 *
 * Blackmagic Film Generation 5 decode (linearize) + encode, and
 * Blackmagic Wide Gamut (Gen 4/5) <-> XYZ (D65). Scene-referred.
 * Curve: natural-log segment with linear toe.
 *   decode: x = (y < LOG_CUT) ? (y-E)/D : exp((y-C)/A) - B,  LOG_CUT = D*LIN_CUT+E
 *   encode: y = (x < LIN_CUT) ? D*x+E   : A*ln(x+B) + C
 * Constants: Blackmagic Generation 5 Color Science documentation, as
 * tabulated in colour-science (verified numerically against it).
 * Encoded domain: normalized [0,1] full-range code values.
 * Generated: 2026-07-07 07:33:10 | Standard: C++14
 */

#ifndef __GENERATED_BMD_FILMGEN5_HPP__
#define __GENERATED_BMD_FILMGEN5_HPP__

#include <cmath>

namespace BMD_FilmGen5
{
    constexpr double A       = 0.08692876065491224;
    constexpr double B       = 0.005494072432257808;
    constexpr double C       = 0.5300133392291939;
    constexpr double D       = 8.283605932402494;
    constexpr double E       = 0.09246575342465753;
    constexpr double LIN_CUT = 0.005;
    constexpr double LOG_CUT = D * LIN_CUT + E;   // exact, compiler-evaluated

    // Decode (linearize): encoded y [0,1] -> scene-linear (18% grey -> 0.18).
    template<typename T>
    inline T decode(T y) noexcept
    {
        if (y < static_cast<T>(LOG_CUT))
            return (y - static_cast<T>(E)) / static_cast<T>(D);
        return std::exp((y - static_cast<T>(C)) / static_cast<T>(A))
               - static_cast<T>(B);
    }

    // Encode: scene-linear x -> Film Gen 5 code value.
    template<typename T>
    inline T encode(T x) noexcept
    {
        if (x < static_cast<T>(LIN_CUT))
            return static_cast<T>(D) * x + static_cast<T>(E);
        return static_cast<T>(A) * std::log(x + static_cast<T>(B))
               + static_cast<T>(C);
    }

    // Blackmagic Wide Gamut (Gen 4/5) primaries, D65 white. Pair with Film Gen 5.
    constexpr double BMDWideGamut_to_XYZ[9] =
    {
        0.60653037221314587, 0.2204080953375982, 0.12347900045836878,
        0.26798940707297414, 0.83273087862234929, -0.10072028569532319,
        -0.029442166015309742, -0.086610606966300657, 1.2048607644426022,
    };
    constexpr double XYZ_to_BMDWideGamut[9] =
    {
        1.8663823403522926, -0.51839734287032091, -0.23460980943156387,
        -0.6003424924985552, 1.3781489624976893, 0.17673183028613085,
        0.0024519937381308975, 0.086399674197522688, 0.83694270731751552,
    };

} // namespace BMD_FilmGen5

#endif // __GENERATED_BMD_FILMGEN5_HPP__
