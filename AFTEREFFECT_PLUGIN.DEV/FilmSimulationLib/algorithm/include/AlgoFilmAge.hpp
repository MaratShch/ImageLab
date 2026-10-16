#pragma once

// ---------------------------------------------------------------------------
//  AlgoFilmAge.hpp -- the 'expired film' control (#209 item 7, 2026-10-16).
//
//  The age of the UNEXPOSED stock at the moment of exposure, in months, read
//  against the only statement about raw-stock ageing in the corpus: the
//  ACCEPTANCE BOUND of a Soviet ТУ / ГОСТ (ToleranceSpec, 17 stocks). A
//  standard prints a flat period during which speed and D-min must stay in
//  the release norms («в течение двух месяцев»), then the worst speed loss
//  and fog rise (negative) or D-max loss (reversal) a roll may show at the
//  END of the guarantee and still be accepted.
//
//  THE MODEL, AND WHAT IN IT IS A DEFINITION
//
//    fraction a   0 up to drift_free_months, linear to 1 at guarantee_months,
//                 HELD AT 1 beyond it. The linear ramp is a definition: the
//                 standard fixes the end points only. Holding at 1 renders an
//                 older roll as the worst roll the standard still accepts --
//                 nothing past the guarantee is extrapolated.
//    speed        a fraction L*a of the speed lost = a shift of log10(1 - L a)
//                 on the exposure the emulsion receives, added to the
//                 reciprocity shift before stage 8 (AlgoFilmAgeLogShift).
//    D-min        + drift * a in every record ("dmin", drift > 0).
//    D-max        - |drift| * a with D-min held ("dmax", drift < 0): gamma x
//                 (1 - |drift| a / (Dmax - Dmin)) per record.
//
//  THE ROW. The stock's is_default tolerance row when it prints the ageing
//  clause, else the FIRST row that does (rows are stored in edition order).
//  The guarantee is the row's own, else BaseSpec::guaranteed_shelf_life_months.
//
//  INERT AT THE DEFAULT. filmAgeMonths is 0 unless the caller moves it; every
//  stock without a bound is inert at any age; `store` is then never written.
//
//  film_sim.film_age_bound / film_age_fraction / film_age_log_shift /
//  resolve_film_age are the reference; cpp_parity.py probes both.
// ---------------------------------------------------------------------------

#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoTypes.hpp"
#include "film_profiles.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>


//  Upper limit of the control, months (fifty years).
constexpr double ALGO_FILM_AGE_MONTHS_MAX = 600.0;


struct AlgoFilmAgeBound
{
    int32_t      guaranteeMonths = 0;
    int32_t      driftFreeMonths = 0;
    HighPrecType speedLossFrac   = 0;
    HighPrecType densityDrift    = 0;
    int32_t      quantity        = 0;   ///< 1 = dmin, 2 = dmax, 0 = none
};


inline bool AlgoFilmAgeRowHas (const film::ToleranceSpec& t) noexcept
{
    const bool q = (t.ageing_drift_quantity == "dmin")
                || (t.ageing_drift_quantity == "dmax");
    return (t.ageing_speed_loss_frac > 0.0f)
        || ((t.ageing_density_drift != 0.0f) && q);
}


inline bool AlgoFilmAgeBoundOf
(
    const film::FilmProfile& p,
    AlgoFilmAgeBound&        b
) noexcept
{
    const film::ToleranceSpec* row = nullptr;
    for (std::size_t i = 0u; i < p.tolerance.size(); ++i)
    {
        if (p.tolerance[i].is_default && AlgoFilmAgeRowHas(p.tolerance[i]))
        {
            row = &p.tolerance[i];
            break;
        }
    }
    if (nullptr == row)
    {
        for (std::size_t i = 0u; i < p.tolerance.size(); ++i)
        {
            if (AlgoFilmAgeRowHas(p.tolerance[i]))
            {
                row = &p.tolerance[i];
                break;
            }
        }
    }
    if (nullptr == row)
        return false;

    const int32_t g = (row->guarantee_months > 0)
                    ? row->guarantee_months
                    : static_cast<int32_t>(p.base.guaranteed_shelf_life_months);
    const int32_t d = row->drift_free_months;
    if ((g <= 0) || (d >= g))
        return false;

    b.guaranteeMonths = g;
    b.driftFreeMonths = d;
    b.speedLossFrac   = static_cast<HighPrecType>(row->ageing_speed_loss_frac);
    b.densityDrift    = static_cast<HighPrecType>(row->ageing_density_drift);
    b.quantity        = (row->ageing_drift_quantity == "dmin") ? 1
                      : (row->ageing_drift_quantity == "dmax") ? 2 : 0;
    return true;
}


inline HighPrecType AlgoFilmAgeFraction
(
    const film::FilmProfile& p,
    const double             months
) noexcept
{
    const HighPrecType zero = static_cast<HighPrecType>(0);
    if (!(months > 0.0))
        return zero;

    AlgoFilmAgeBound b;
    if (!AlgoFilmAgeBoundOf(p, b))
        return zero;

    const HighPrecType g = static_cast<HighPrecType>(b.guaranteeMonths);
    const HighPrecType d = static_cast<HighPrecType>(b.driftFreeMonths);
    const HighPrecType m = static_cast<HighPrecType>(
        (months < ALGO_FILM_AGE_MONTHS_MAX) ? months : ALGO_FILM_AGE_MONTHS_MAX);

    if (m <= d) return zero;
    if (m >= g) return static_cast<HighPrecType>(1);
    return (m - d) / (g - d);
}


//  log10 exposure shift of the speed loss; exactly 0 when fresh or unbounded.
inline HighPrecType AlgoFilmAgeLogShift
(
    const film::FilmProfile& p,
    const double             months
) noexcept
{
    const HighPrecType zero = static_cast<HighPrecType>(0);
    const HighPrecType a = AlgoFilmAgeFraction(p, months);
    if (a <= zero)
        return zero;

    AlgoFilmAgeBound b;
    AlgoFilmAgeBoundOf(p, b);
    const HighPrecType loss = b.speedLossFrac * a;
    if (loss <= zero)
        return zero;
    return static_cast<HighPrecType>(
        std::log10(static_cast<double>(static_cast<HighPrecType>(1) - loss)));
}


inline const film::FilmProfile& AlgoResolveFilmAge
(
    const film::FilmProfile& base,
    const double             months,
    film::FilmProfile&       store
) noexcept
{
    const HighPrecType a = AlgoFilmAgeFraction(base, months);
    if (a <= static_cast<HighPrecType>(0))
        return base;

    AlgoFilmAgeBound b;
    AlgoFilmAgeBoundOf(base, b);

    const film::ToneCurve* const in[3] =
    { &base.curves.r, &base.curves.g, &base.curves.b };

    if ((1 == b.quantity) && (b.densityDrift > static_cast<HighPrecType>(0)))
    {
        store = base;
        film::ToneCurve* const c[3] = { &store.curves.r, &store.curves.g, &store.curves.b };
        const HighPrecType dd = b.densityDrift * a;
        for (int32_t i = 0; i < 3; i++)
        {
            c[i]->dmin = static_cast<float>(static_cast<HighPrecType>(in[i]->dmin) + dd);
            c[i]->meas_x = nullptr;
            c[i]->meas_d = nullptr;
            c[i]->meas_m = nullptr;
            c[i]->meas_n = 0;
        }
        return store;
    }

    if ((2 == b.quantity) && (b.densityDrift < static_cast<HighPrecType>(0)))
    {
        store = base;
        film::ToneCurve* const c[3] = { &store.curves.r, &store.curves.g, &store.curves.b };
        for (int32_t i = 0; i < 3; i++)
        {
            const HighPrecType span = static_cast<HighPrecType>(in[i]->gamma)
                * (static_cast<HighPrecType>(in[i]->shoulder_x)
                   - static_cast<HighPrecType>(in[i]->toe_x));
            if (span <= static_cast<HighPrecType>(0))
                continue;
            HighPrecType f = static_cast<HighPrecType>(1) + b.densityDrift * a / span;
            if (f < static_cast<HighPrecType>(0))
                f = static_cast<HighPrecType>(0);
            c[i]->gamma = static_cast<float>(static_cast<HighPrecType>(in[i]->gamma) * f);
            c[i]->meas_x = nullptr;
            c[i]->meas_d = nullptr;
            c[i]->meas_m = nullptr;
            c[i]->meas_n = 0;
        }
        return store;
    }

    return base;
}
