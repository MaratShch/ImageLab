#pragma once

// ---------------------------------------------------------------------------
//  AlgoStorageAge.hpp
//
//  The film as YEARS OF DARK STORAGE leave it. Header only, resolved ONCE PER
//  FRAME, and the result is a profile - the third member of the family
//  AlgoProcessVariant.hpp and AlgoDevelopmentTime.hpp belong to, built the
//  same way so the three cannot drift apart.
//
//  WHAT THIS CONTROL IS
//  --------------------
//  film::AgingSpec has existed since schema v2 with eleven fields, is ALL
//  ZEROS ON ALL 191 STOCKS, and is read by nothing. It holds a STATE - how
//  much fade a piece of film has already suffered - and there was never any
//  way to arrive at that state, because nothing in the database said how fast
//  a given film fades.
//
//  film::DyeStabilitySpec does say, and as of schema v35 it says it for camera
//  negatives rather than only for one recording film. This resolver is the
//  function that turns the RATE into the STATE: given a storage age it
//  computes the fraction of each image dye lost and applies it.
//
//  THE LAW, AND WHY IT IS NOT FITTED
//  ---------------------------------
//  Both sources state a time to a 10 % loss from a starting density of 1.0.
//  That is a statement about a constant fractional rate, so a dye losing a
//  tenth of what remains in T years has lost
//
//      f(t) = 1 - 0.9^(t/T)
//
//  after t years. Nothing is fitted; this is the published figure restated as
//  a function of time, and at t = T it returns exactly 0.10.
//
//  \warning ONLY THE DYE THE SOURCE NAMES IS FADED. Wilhelm's Table 19.1
//  publishes the time for the LEAST STABLE dye and names it - yellow, on both
//  stocks that carry a rate - and says nothing about the other two. Fading all
//  three together would assert an equality the source explicitly denies: its
//  subject is that "one of the three image dyes -- usually magenta -- is much
//  more stable in dark fading than is the least stable dye, and this
//  differential in fading rates results in increasingly objectionable color
//  shifts". THE DIFFERENTIAL IS THE EFFECT. A uniform fade would be a density
//  change wearing its costume.
//
//  \warning WHICH DYE SITS IN WHICH RECORD. On a chromogenic negative cyan
//  forms in the red-sensitive layer, magenta in the green and yellow in the
//  blue, so a dye's fade is read in that record and no other.
//
//  \warning dmin IS NOT TOUCHED, AND THAT IS A REFUSAL RATHER THAN AN
//  OMISSION. On a masked negative part of D-min is the orange mask, which is
//  dye and does fade, and part is the support, which does not. ToneCurve
//  stores their SUM and nothing in this corpus separates them, so any dmin
//  change here would be a guess at that split applied to every stock. The
//  consequence, stated so it is not filed as a bug: a faded negative rendered
//  here loses image dye and keeps its mask, where the real one loses some of
//  both.
//
//  \warning TEMPERATURE: REFUSED ON 2026-09-17, ADOPTED ON 2026-09-28b
//  (schema v57). The refusal had Table 19.1's two refrigerator factors -
//  three points - and would not fit a law to them. Wilhelm's Table 5.3 (p178,
//  from Bard et al., Eastman Kodak, J. Appl. Photogr. Eng. 6(2), 1980, p44)
//  prints TEN points from -26 to +30 degC, and in Arrhenius coordinates they
//  are one law (adjacent-pair activation energy 71-102 kJ/mol, global 85.5).
//  AlgoStorageTimeFactor reads it: ln F linear in 1/T between printed points,
//  exact at every printed point, HELD at the end values outside the table -
//  never extrapolated. It is Kodak's relation for Kodak's dyes, applied here
//  to every record; for the two Fuji negatives that is a transfer the source
//  does not make, and the provenance says so.
//  The published years are also a 40 % RH figure that HALVES at 60 % RH, and
//  RELATIVE HUMIDITY IS STILL NOT A CONTROL: Table 5.4 is three points for
//  Kodak yellow dyes only. They count dye fading only: the yellowish stain
//  that usually becomes visible first is not modelled at all, because no
//  source in the corpus gives a room-temperature stain rate for a camera film
//  (Table 5.9's stain figures are a 62 degC single-temperature ranking, which
//  the author says does not predict room temperature).
//
//  \warning THE LOSS CRITERION IS READ FROM THE RECORD (schema v57).
//  DyeStabilitySpec::loss_percent is 10 on the Kodak-sheet / Table 19.1 /
//  Table 9.x records and 20 on the Table 5.13 slide films, and the law is
//  f(t) = 1 - (1 - L)^(t/T). At L = 10, (100 - 10) / 100 is exactly the 0.9
//  the pre-v57 law hard-coded, so every older record renders bit for bit.
//
//  INERT BY DEFAULT. storageYears <= 0 is fresh film, the base profile is
//  returned by reference, nothing is copied, and every render made before this
//  file existed is reproduced bit for bit. A stock with no published rate -
//  187 of 201 at schema v57 - takes the same path at any age, and so does
//  storageCelsius at its default 24 degC, the reference temperature of every
//  stored record: the elapsed years are then used as given, uncomputed.
//
//  ONE LAW, TWO LANGUAGES. film_sim.resolve_storage_age() is the reference;
//  cpp_parity.py drives this resolver over every stock and a sweep of ages and
//  compares the curve parameters it yields.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

// film::FilmProfile, film::DyeStabilitySpec, film::ToneCurve.
#include "film_profiles.hpp"

#include <cmath>


// ---------------------------------------------------------------------------
//  Storage temperature: Wilhelm 1993, Table 5.3 (schema v57)
//
//  (degC, RELATIVE STORAGE TIME to the same fade, 1 at 75 degF / 24 degC),
//  warmest first, the Celsius column exactly as printed. The same ten nodes
//  as film_sim.STORAGE_TEMPERATURE_TABLE; cpp_parity.py compares the two
//  resolvers over a temperature sweep that includes every node, a point
//  between each pair and both clamps.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_STORAGE_TEMP_NODES = 10;

constexpr double ALGO_STORAGE_TEMP_C[ALGO_STORAGE_TEMP_NODES] =
{ 30.0, 24.0, 19.0, 13.0, 7.0, 4.0, 0.0, -10.0, -18.0, -26.0 };

constexpr double ALGO_STORAGE_TIME_REL[ALGO_STORAGE_TEMP_NODES] =
{ 0.5, 1.0, 2.0, 4.0, 10.0, 16.0, 28.0, 100.0, 340.0, 1000.0 };


// ---------------------------------------------------------------------------
//  AlgoStorageTimeFactor
//
//  Relative storage time to the same fade at `celsius`, 1 at 24 degC. ln F is
//  linear in 1/T between printed points (the Arrhenius axis), exact at every
//  printed point, and held at the end values outside -26 .. 30 degC.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoStorageTimeFactor
(
    const HighPrecType celsius
) noexcept
{
    const int32_t last = ALGO_STORAGE_TEMP_NODES - 1;

    if (celsius >= static_cast<HighPrecType>(ALGO_STORAGE_TEMP_C[0]))
        return static_cast<HighPrecType>(ALGO_STORAGE_TIME_REL[0]);
    if (celsius <= static_cast<HighPrecType>(ALGO_STORAGE_TEMP_C[last]))
        return static_cast<HighPrecType>(ALGO_STORAGE_TIME_REL[last]);

    for (int32_t i = 0; i < last; i++)
    {
        const HighPrecType t0 = static_cast<HighPrecType>(ALGO_STORAGE_TEMP_C[i]);
        const HighPrecType t1 = static_cast<HighPrecType>(ALGO_STORAGE_TEMP_C[i + 1]);
        const HighPrecType f0 = static_cast<HighPrecType>(ALGO_STORAGE_TIME_REL[i]);
        const HighPrecType f1 = static_cast<HighPrecType>(ALGO_STORAGE_TIME_REL[i + 1]);

        if (celsius == t0)
            return f0;
        if (celsius == t1)
            return f1;
        if ((celsius < t0) && (celsius > t1))
        {
            const HighPrecType k  = static_cast<HighPrecType>(273.15);
            const HighPrecType on = static_cast<HighPrecType>(1);
            const HighPrecType x0 = on / (t0 + k);
            const HighPrecType x1 = on / (t1 + k);
            const HighPrecType x  = on / (celsius + k);
            const HighPrecType w  = (x - x0) / (x1 - x0);
            return std::exp(std::log(f0) + w * (std::log(f1) - std::log(f0)));
        }
    }

    return static_cast<HighPrecType>(1);   // unreachable: contiguous table
}


// ---------------------------------------------------------------------------
//  AlgoStorageEquivalentYears
//
//  `years` spent at `celsius`, restated as years at the record's own
//  reference temperature. Returns `years` itself, uncomputed, when the two
//  are equal - the default - so the pre-v57 arithmetic is reproduced exactly.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoStorageEquivalentYears
(
    const HighPrecType years,
    const HighPrecType celsius,
    const HighPrecType referenceC
) noexcept
{
    if (celsius == referenceC)
        return years;

    return years * AlgoStorageTimeFactor(referenceC)
                 / AlgoStorageTimeFactor(celsius);
}


// ---------------------------------------------------------------------------
//  AlgoDarkFadeFraction
//
//  Fraction of one dye lost after `years` (already restated at the record's
//  reference temperature), from a published time to a `lossPercent` loss.
//  Returns 0 when the source states no time for that dye, which is the common
//  case and is the whole reason the three are computed separately.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDarkFadeFraction
(
    const HighPrecType lossYears,
    const HighPrecType years,
    const HighPrecType lossPercent
) noexcept
{
    if ((lossYears <= static_cast<HighPrecType>(0))
        || (years   <= static_cast<HighPrecType>(0)))
    {
        return static_cast<HighPrecType>(0);
    }

    const HighPrecType hundred = static_cast<HighPrecType>(100);
    const HighPrecType base    = (hundred - lossPercent) / hundred;

    return static_cast<HighPrecType>(1)
         - std::pow(base, years / lossYears);
}


// ---------------------------------------------------------------------------
//  AlgoResolveStorageAge
//
//  base     the stock as the database holds it, or as the two resolvers
//           before this one left it
//  years    years of dark storage since processing; <= 0 is fresh
//  celsius  storage temperature for those years, degC; 24 is the default and
//           the reference temperature of every stored record (schema v57)
//  store    scratch the caller owns for the lifetime of the frame; written to
//           ONLY when at least one dye actually fades
//
//  Returns a reference to `base` on every inert path. The caller must keep
//  `store` alive as long as it uses the result.
// ---------------------------------------------------------------------------
inline const film::FilmProfile& AlgoResolveStorageAge
(
    const film::FilmProfile& base,
    const double             years,
    const double             celsius,
    film::FilmProfile&       store
) noexcept
{
    const HighPrecType t = static_cast<HighPrecType>(years);

    if (t <= static_cast<HighPrecType>(0))
        return base;

    const film::DyeStabilitySpec& d = base.dye_stability;

    // Only reached when some record could fade: a stock with no rate returns
    // zero from all three fractions below whatever the equivalent age is.
    const HighPrecType te = AlgoStorageEquivalentYears(
        t,
        static_cast<HighPrecType>(celsius),
        static_cast<HighPrecType>(d.reference_temp_c));

    const HighPrecType pct = static_cast<HighPrecType>(d.loss_percent);

    const HighPrecType fc =
        AlgoDarkFadeFraction(static_cast<HighPrecType>(d.loss_c), te, pct);
    const HighPrecType fm =
        AlgoDarkFadeFraction(static_cast<HighPrecType>(d.loss_m), te, pct);
    const HighPrecType fy =
        AlgoDarkFadeFraction(static_cast<HighPrecType>(d.loss_y), te, pct);

    const HighPrecType zero = static_cast<HighPrecType>(0);

    if ((fc == zero) && (fm == zero) && (fy == zero))
        return base;

    store = base;

    const HighPrecType one = static_cast<HighPrecType>(1);

    store.curves.r.gamma = static_cast<float>(
        static_cast<HighPrecType>(base.curves.r.gamma) * (one - fc));
    store.curves.g.gamma = static_cast<float>(
        static_cast<HighPrecType>(base.curves.g.gamma) * (one - fm));
    store.curves.b.gamma = static_cast<float>(
        static_cast<HighPrecType>(base.curves.b.gamma) * (one - fy));


    // ⚠⚠ AND THE MEASURED TABLE IS DROPPED (schema v55). A `MeasuredCurve` is
    // 21 densities read at ONE development; scaling the gamma beside it does
    // not scale the table, and because the table WINS inside its own range it
    // would override the very adjustment made above. An adjusted curve is a
    // MODEL, and the measurement it came from no longer describes it, so it
    // falls back to the softplus -- which is exactly what the Python
    // reference's `film_sim._retune` does at the same four sites, guarded by
    // G-V55-MEAS-DROP. Leaving it attached would make this control a no-op on
    // any stock carrying a table, silently.
    film::ToneCurve* const _mc[3] =
    { &store.curves.r, &store.curves.g, &store.curves.b };
    for (int32_t _k = 0; _k < 3; _k++)
    {
        _mc[_k]->meas_x = nullptr;
        _mc[_k]->meas_d = nullptr;
        _mc[_k]->meas_m = nullptr;
        _mc[_k]->meas_n = 0;
    }

    return store;
}
