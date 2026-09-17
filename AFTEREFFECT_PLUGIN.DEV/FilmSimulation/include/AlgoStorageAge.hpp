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
//  \warning NO TEMPERATURE CONTROL. The figures are quoted at 24 degC and the
//  source gives factors for two refrigerator temperatures - about 14x longer
//  at 4.4 degC and 20x at 1.7 degC - but three points do not define a
//  continuous law and this project does not fit one to invent the values
//  between them. The factors are in DyeStabilitySpec::source for whoever does.
//  The published years are also a 40 % RH figure that HALVES at 60 % RH, and
//  they count dye fading only: the yellowish stain that usually becomes
//  visible first is not modelled at all.
//
//  INERT BY DEFAULT. storageYears <= 0 is fresh film, the base profile is
//  returned by reference, nothing is copied, and every render made before this
//  file existed is reproduced bit for bit. A stock with no published rate -
//  188 of 191 today - takes the same path at any age.
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
//  AlgoDarkFadeFraction
//
//  Fraction of one dye lost after `years`, from a published time to a 10 %
//  loss. Returns 0 when the source states no time for that dye, which is the
//  common case and is the whole reason the three are computed separately.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDarkFadeFraction
(
    const HighPrecType lossYears,
    const HighPrecType years
) noexcept
{
    if ((lossYears <= static_cast<HighPrecType>(0))
        || (years   <= static_cast<HighPrecType>(0)))
    {
        return static_cast<HighPrecType>(0);
    }

    return static_cast<HighPrecType>(1)
         - std::pow(static_cast<HighPrecType>(0.9), years / lossYears);
}


// ---------------------------------------------------------------------------
//  AlgoResolveStorageAge
//
//  base     the stock as the database holds it, or as the two resolvers
//           before this one left it
//  years    years of dark storage since processing; <= 0 is fresh
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
    film::FilmProfile&       store
) noexcept
{
    const HighPrecType t = static_cast<HighPrecType>(years);

    if (t <= static_cast<HighPrecType>(0))
        return base;

    const film::DyeStabilitySpec& d = base.dye_stability;

    const HighPrecType fc =
        AlgoDarkFadeFraction(static_cast<HighPrecType>(d.loss_c), t);
    const HighPrecType fm =
        AlgoDarkFadeFraction(static_cast<HighPrecType>(d.loss_m), t);
    const HighPrecType fy =
        AlgoDarkFadeFraction(static_cast<HighPrecType>(d.loss_y), t);

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

    return store;
}
