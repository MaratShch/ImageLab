#pragma once

// ---------------------------------------------------------------------------
//  AlgoDevelopmentTime.hpp
//
//  The film as a chosen DEVELOPMENT TIME renders it. Header only, resolved
//  ONCE PER FRAME, and the result is a profile - not a correction applied to
//  one. The sibling of AlgoProcessVariant.hpp in every structural respect,
//  deliberately: the two answer the same question and must not diverge in how
//  they answer it.
//
//  WHAT THIS CONTROL IS
//  --------------------
//  Every characteristic curve in this database is ONE development. Until this
//  file existed that development was unnamed and unreachable: the database has
//  carried a film::ProcessingFamily since schema v7 - by now 960 development
//  points across 25 stocks - and NO STAGE IN EITHER ENGINE READ A SINGLE ONE
//  OF THEM. The data was inert, and the control constants in
//  AlgoControlEnums.hpp described a slider wired to nothing.
//
//  This resolver moves the stored curve ALONG the axis its own family
//  describes: longer development, more contrast, by the amount that stock's
//  own traced family says and by no other amount.
//
//  WHY A RATIO AND NOT AN ABSOLUTE GAMMA
//  -------------------------------------
//  The stored curve is already a development, so the operation is a move along
//  the axis rather than a replacement of it. The reference is the family's
//  gamma at the development the stored curve represents, so asking for that
//  time returns exactly 1.0 and reproduces every earlier render bit for bit.
//  Taking the family's absolute gamma instead would silently re-level every
//  stock whose traced family and stored curve disagree slightly, which is a
//  different change and an unwanted one.
//
//  WHERE THE REFERENCE TIME COMES FROM
//  -----------------------------------
//  ProcessingSpec::minutes is the obvious anchor and it is EMPTY on six of the
//  eleven stocks that have a usable family - including all four the 1956
//  harvest gave a family to - so requiring it would leave the control inert on
//  most of the data that exists to drive it. The fallback is an INVERSION and
//  not a substitution: the stored curve has a gamma, the family says which
//  time produces that gamma, and that time IS "the development the stored
//  curves represent", which is the sentinel's own definition. It uses only
//  measured numbers. Where the stored gamma falls outside the family's gamma
//  range the two describe different emulsions, there is no defensible
//  reference, and the resolver refuses.
//
//  \warning A ProcessingFamily IS NOT ONE CURVE. The point tuple is flat by
//  design and can hold, on one stock, two developers at two dilutions in two
//  vessels across two emulsion GENERATIONS - which is why DevelopmentPoint
//  carries `vessel` (v28) and `edition` (v35). Reading gamma against time
//  straight off the tuple would interpolate between a 1956 roll film in a
//  small tank and a 2016 sheet in a tray. AlgoDevelopmentGroup picks ONE
//  coherent group and the interpolation never leaves it.
//
//  \warning MONOCHROME ONLY, and that is a statement about the data rather
//  than a convenience. One scale applied to three records asserts the three
//  layers move together under development, which this project has measured to
//  be false - PORTRA 800 pushed to EI 3200 gains 0.25 of gamma in red against
//  0.14 in blue. Every gamma-bearing family in the database is monochrome but
//  one, and that one carries a single channel's worth of points.
//
//  \warning WHAT IT DOES NOT DO, so it is not filed as a bug: it does not move
//  base fog, granularity or speed. All three move with development time in
//  reality; none has a traced relation here. DevelopmentPoint already carries
//  base_fog and exposure_index for the day one exists - base_fog is populated
//  on exactly one stock today, and a relation fitted to one stock is not a
//  relation.
//
//  INERT BY DEFAULT. developmentMinutes < 0 is the sentinel, the base profile
//  is returned by reference, nothing is copied, and every render made before
//  this file existed is reproduced bit for bit. A time outside the stock's own
//  traced range takes the same path - it is TREATED AS THE SENTINEL AND NOT
//  CLAMPED, because extrapolating a development family is not a measurement.
//
//  ONE LAW, TWO LANGUAGES. film_sim.resolve_development_time() is the
//  reference; cpp_parity.py drives this resolver over every stock and a sweep
//  of times and compares the curve parameters it yields.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

// film::FilmProfile, film::ProcessingFamily, film::DevelopmentPoint.
#include "film_profiles.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
//  AlgoDevelopmentGroup
//
//  The one coherent (developer, dilution, vessel, edition) group of points the
//  stored curve sits on, as a sorted (minutes, gamma) list. Empty when the
//  stock has no usable family or when the choice would be ambiguous.
//
//  THE GROUP IS CHOSEN, NOT GUESSED:
//    1. only points carrying a real gamma are eligible - a time-only point
//       states a temperature, not a contrast, and cannot place a curve;
//    2. groups are keyed on all four discriminants;
//    3. a group whose developer matches ProcessingSpec::developer wins,
//       because that is the developer the STORED CURVE was measured in;
//    4. failing THAT, ProcessingFamily::reference_developer - the developer
//       the SOURCE's own characteristic curve was measured in, which the
//       1956 sheets letter in their caption and the profile's ProcessingSpec
//       does not carry because the stored curve is a later coating;
//    5. failing that the largest group wins, and a TIE IS A REFUSAL - two
//       equally large groups are two different processes, and picking one by
//       container order would make the render depend on insertion order.
// ---------------------------------------------------------------------------
inline void AlgoDevelopmentGroup
(
    const film::FilmProfile&                          p,
    std::vector<std::pair<HighPrecType, HighPrecType>>& out
) noexcept
{
    out.clear();

    const std::vector<film::DevelopmentPoint>& pts = p.processing_family.points;
    if (pts.empty())
        return;

    // Keys are built as one string so no ordering container is needed and the
    // comparison cannot disagree with the Python reference's tuple compare.
    std::vector<std::string>                                        keys;
    std::vector<std::vector<std::pair<HighPrecType, HighPrecType>>> groups;
    std::vector<std::string>                                        devs;
    std::vector<std::string>                                        dils;

    for (std::size_t i = 0u; i < pts.size(); ++i)
    {
        const film::DevelopmentPoint& q = pts[i];

        if (q.gamma <= 0.0)
            continue;

        const std::string key =
            q.developer + "\x1f" + q.dilution + "\x1f" +
            q.vessel    + "\x1f" + q.edition;

        std::size_t slot = keys.size();
        for (std::size_t k = 0u; k < keys.size(); ++k)
        {
            if (keys[k] == key)
            {
                slot = k;
                break;
            }
        }
        if (slot == keys.size())
        {
            keys.push_back(key);
            groups.push_back(std::vector<std::pair<HighPrecType, HighPrecType>>());
            devs.push_back(q.developer);
            dils.push_back(q.dilution);
        }
        groups[slot].push_back(
            std::make_pair(static_cast<HighPrecType>(q.minutes),
                           static_cast<HighPrecType>(q.gamma)));
    }

    // A group of one cannot describe an axis.
    std::vector<std::size_t> live;
    for (std::size_t k = 0u; k < groups.size(); ++k)
    {
        if (groups[k].size() >= 2u)
            live.push_back(k);
    }
    if (live.empty())
        return;

    // Prefer the developer the stored curve was measured in. The comparison is
    // case-insensitive and trimmed on both sides, matching the reference.
    auto norm = [](std::string s) -> std::string
    {
        std::size_t a = s.find_first_not_of(" \t");
        std::size_t b = s.find_last_not_of(" \t");
        s = (std::string::npos == a) ? std::string() : s.substr(a, b - a + 1u);
        for (std::size_t i = 0u; i < s.size(); ++i)
        {
            if (('A' <= s[i]) && ('Z' >= s[i]))
                s[i] = static_cast<char>(s[i] - 'A' + 'a');
        }
        return s;
    };

    const std::string want = norm(p.processing.developer);
    if (!want.empty())
    {
        std::vector<std::size_t> named;
        for (std::size_t i = 0u; i < live.size(); ++i)
        {
            if (norm(devs[live[i]]) == want)
                named.push_back(live[i]);
        }
        if (!named.empty())
            live = named;
    }
    else
    {
        // \warning FAILING THAT, THE FAMILY'S OWN REFERENCE DEVELOPER (schema
        // v36), AND WITHOUT IT A STOCK LOSES THE CONTROL WHEN IT GAINS DATA.
        // ProcessingSpec::developer describes the profile's STORED curve and
        // is empty on every 1956-sourced stock, because the stored curve is a
        // later sheet. ProcessingFamily::reference_developer describes the
        // SOURCE's own characteristic curve - the curve these points were
        // drawn beside - and Kodak letters it in the caption. Tracing the
        // 1956 time-gamma insets gave SUPER-XX PAN seventeen DK-50 points
        // against seventeen DK-60a, which is a tie and therefore a refusal,
        // so the stock lost a working development control the moment the
        // measurements arrived.
        const std::string ref  = norm(p.processing_family.reference_developer);
        const std::string rdil = norm(p.processing_family.reference_dilution);
        if (!ref.empty())
        {
            std::vector<std::size_t> named;
            for (std::size_t i = 0u; i < live.size(); ++i)
            {
                if ((norm(devs[live[i]]) == ref)
                    && (rdil.empty() || (norm(dils[live[i]]) == rdil)))
                {
                    named.push_back(live[i]);
                }
            }
            if (!named.empty())
                live = named;
        }
    }

    std::size_t bestN = 0u;
    for (std::size_t i = 0u; i < live.size(); ++i)
        bestN = (groups[live[i]].size() > bestN) ? groups[live[i]].size() : bestN;

    std::size_t nBest = 0u;
    std::size_t pick  = 0u;
    for (std::size_t i = 0u; i < live.size(); ++i)
    {
        if (groups[live[i]].size() == bestN)
        {
            ++nBest;
            pick = live[i];
        }
    }
    if (1u != nBest)
        return;                     // a tie is a refusal, see the note above

    out = groups[pick];

    // Sort by time. Insertion sort: the largest group in the database is 62
    // points and most are five.
    for (std::size_t i = 1u; i < out.size(); ++i)
    {
        std::pair<HighPrecType, HighPrecType> v = out[i];
        std::size_t j = i;
        while ((j > 0u) && (out[j - 1u].first > v.first))
        {
            out[j] = out[j - 1u];
            --j;
        }
        out[j] = v;
    }
}


// ---------------------------------------------------------------------------
//  AlgoDevelopmentGammaScale
//
//  The factor by which `minutes` moves this stock's gamma. Exactly 1.0 on
//  every refusal path, which is what keeps the sentinel bit-exact.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDevelopmentGammaScale
(
    const film::FilmProfile& p,
    const HighPrecType       minutes
) noexcept
{
    const HighPrecType one = static_cast<HighPrecType>(1);

    if (minutes < static_cast<HighPrecType>(0))
        return one;
    if (!p.is_monochrome)
        return one;

    std::vector<std::pair<HighPrecType, HighPrecType>> pts;
    AlgoDevelopmentGroup(p, pts);
    if (pts.size() < 2u)
        return one;

    const HighPrecType lo = pts.front().first;
    const HighPrecType hi = pts.back().first;
    if ((minutes < lo) || (minutes > hi))
        return one;

    auto at = [&pts](const HighPrecType t) -> HighPrecType
    {
        for (std::size_t i = 1u; i < pts.size(); ++i)
        {
            const HighPrecType t0 = pts[i - 1u].first;
            const HighPrecType t1 = pts[i].first;
            if ((t >= t0) && (t <= t1))
            {
                if (t1 == t0)
                    return pts[i - 1u].second;
                return pts[i - 1u].second
                     + (pts[i].second - pts[i - 1u].second)
                     * (t - t0) / (t1 - t0);
            }
        }
        return pts.back().second;
    };

    HighPrecType ref = static_cast<HighPrecType>(p.processing.minutes);

    if ((ref < lo) || (ref > hi))
    {
        // Invert the family at the stored curve's own gamma. See the header
        // note: this is the sentinel's definition expressed in measured
        // numbers, not a substitute anchor.
        const HighPrecType gStored = static_cast<HighPrecType>(p.curves.g.gamma);
        const HighPrecType gA = pts.front().second;
        const HighPrecType gB = pts.back().second;
        const HighPrecType gLo = (gA < gB) ? gA : gB;
        const HighPrecType gHi = (gA < gB) ? gB : gA;

        if ((gStored < gLo) || (gStored > gHi))
            return one;

        bool found = false;
        for (std::size_t i = 1u; i < pts.size(); ++i)
        {
            const HighPrecType g0 = pts[i - 1u].second;
            const HighPrecType g1 = pts[i].second;
            const HighPrecType a  = (g0 < g1) ? g0 : g1;
            const HighPrecType b  = (g0 < g1) ? g1 : g0;

            if ((gStored >= a) && (gStored <= b))
            {
                const HighPrecType t0 = pts[i - 1u].first;
                const HighPrecType t1 = pts[i].first;
                ref = (g1 == g0)
                    ? t0
                    : (t0 + (t1 - t0) * (gStored - g0) / (g1 - g0));
                found = true;
                break;
            }
        }
        if (!found)
            return one;
    }

    const HighPrecType gRef = at(ref);
    if (gRef <= static_cast<HighPrecType>(0))
        return one;

    return at(minutes) / gRef;
}


// ---------------------------------------------------------------------------
//  AlgoResolveDevelopmentTime
//
//  base     the stock as the database holds it, or as a variant overrode it
//  minutes  the requested development, or < 0 for the sentinel
//  store    scratch the caller owns for the lifetime of the frame; written to
//           ONLY when the scale is not exactly 1
//
//  Returns a reference to `base` on every inert path, so the sentinel copies
//  nothing. The caller must keep `store` alive as long as it uses the result.
//
//  \warning THE SCALE MULTIPLIES ToneCurve::gamma, THE MODEL COEFFICIENT, and
//  not the observable mid-scale slope - the same choice AlgoProcessVariant
//  makes for ProcessVariant::gamma_scale, made here for the same reason and
//  stated in both files so they cannot drift apart. dmin is NOT touched.
// ---------------------------------------------------------------------------
inline const film::FilmProfile& AlgoResolveDevelopmentTime
(
    const film::FilmProfile& base,
    const double             minutes,
    film::FilmProfile&       store
) noexcept
{
    const HighPrecType k =
        AlgoDevelopmentGammaScale(base, static_cast<HighPrecType>(minutes));

    if (k == static_cast<HighPrecType>(1))
        return base;

    store = base;

    store.curves.r.gamma =
        static_cast<float>(static_cast<HighPrecType>(base.curves.r.gamma) * k);
    store.curves.g.gamma =
        static_cast<float>(static_cast<HighPrecType>(base.curves.g.gamma) * k);
    store.curves.b.gamma =
        static_cast<float>(static_cast<HighPrecType>(base.curves.b.gamma) * k);

    return store;
}
