#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

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
//  speed. Granularity moves through the measured rms-vs-gamma law (v59).
//  BASE FOG MOVES SINCE #209 (2026-10-16), by the selected group's OWN
//  DevelopmentPoint::base_fog points, where that group carries two or more:
//  six monochrome stocks today (verify G-209-FOG). A stock without fog points moves gamma only;
//  no relation is borrowed across stocks.
//
//  INERT BY DEFAULT. developmentMinutes < 0 is the sentinel, the base profile
//  is returned by reference, nothing is copied, and every render made before
//  this file existed is reproduced bit for bit. A time BELOW the stock's own
//  traced range takes the same path - it is TREATED AS THE SENTINEL AND NOT
//  CLAMPED, because extrapolating a development family is not a measurement.
//  A time ABOVE it does too, unless a fitted RATE LAW governs the group (#209):
//  then the law is evaluated, capped at Glafkides' 0.80 * gamma_inf usable
//  ceiling or at the last traced value, whichever is higher. See
//  AlgoDevelopmentAxis.
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
#include "AlgoControlEnums.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
//  AlgoDevelopmentGroup
//
//  The one coherent group of points the stored curve sits on, as a sorted
//  (minutes, contrast) list. Empty when the stock has no usable family or when
//  the choice would be ambiguous.
//
//  THE GROUP IS CHOSEN, NOT GUESSED:
//    1. only points carrying a real contrast are eligible - a time-only point
//       states a temperature, not a contrast, and cannot place a curve.
//       GAMMA groups first; CONTRAST-INDEX groups (Kodak CI, Fuji G-bar, the
//       «Современные» kinetics panels, 2026-10-01d) only when the stock has
//       no gamma group at all, only groups that RISE with time, and only for
//       a developer the record names (no "largest group" guess);
//    2. groups are keyed on measure kind, developer, dilution, vessel,
//       edition, temperature and film format - a curve is one temperature
//       (рис. 3.256 draws five in one vessel) and one format (Fuji 135/120);
//    3. a group whose developer matches ProcessingSpec::developer wins,
//       because that is the developer the STORED CURVE was measured in;
//    4. failing THAT (ProcessingSpec::developer empty, or a CI family with no
//       match), ProcessingFamily::reference_developer, filtered by
//       reference_dilution when that is set ("stock" == undiluted == "");
//    5. then reference_edition, then the largest group, and a TIE IS A
//       REFUSAL - two equally large groups are two different processes.
//  film_sim.development_family is the reference; cpp_parity probes it.
// ---------------------------------------------------------------------------
inline void AlgoDevelopmentGroup
(
    const film::FilmProfile&                          p,
    std::vector<std::pair<HighPrecType, HighPrecType>>& out,
    std::string*                                      outDeveloper = nullptr,
    std::string*                                      outVessel    = nullptr,
    std::vector<std::pair<HighPrecType, HighPrecType>>* outFog     = nullptr
) noexcept
{
    out.clear();
    if (nullptr != outDeveloper) outDeveloper->clear();
    if (nullptr != outVessel)    outVessel->clear();
    if (nullptr != outFog)       outFog->clear();

    const std::vector<film::DevelopmentPoint>& pts = p.processing_family.points;
    if (pts.empty())
        return;

    // Keys are built as one string plus the two numeric discriminants, so no
    // ordering container is needed and equality matches the Python tuple.
    std::vector<std::string>                                        keys;
    std::vector<int>                                                kinds;
    std::vector<double>                                             cels;
    std::vector<std::vector<std::pair<HighPrecType, HighPrecType>>> groups;
    std::vector<std::string>                                        devs;
    std::vector<std::string>                                        dils;
    std::vector<std::string>                                        eds;
    // #209: the group's vessel (for the rate-law match) and its own base-fog
    // points (minutes, base_fog > 0), carried beside the contrast points.
    std::vector<std::string>                                        vess;
    std::vector<std::vector<std::pair<HighPrecType, HighPrecType>>> fogs;

    for (std::size_t i = 0u; i < pts.size(); ++i)
    {
        const film::DevelopmentPoint& q = pts[i];

        int    kind;
        double v;
        if (q.gamma > 0.0)
        {
            kind = 0;
            v    = q.gamma;
        }
        else if (q.contrast_index > 0.0)
        {
            kind = 1;
            v    = q.contrast_index;
        }
        else
        {
            continue;
        }

        const std::string key =
            q.developer + "\x1f" + q.dilution + "\x1f" +
            q.vessel    + "\x1f" + q.edition  + "\x1f" + q.film_format;

        std::size_t slot = keys.size();
        for (std::size_t k = 0u; k < keys.size(); ++k)
        {
            if ((kinds[k] == kind) && (cels[k] == q.celsius) && (keys[k] == key))
            {
                slot = k;
                break;
            }
        }
        if (slot == keys.size())
        {
            keys.push_back(key);
            kinds.push_back(kind);
            cels.push_back(q.celsius);
            groups.push_back(std::vector<std::pair<HighPrecType, HighPrecType>>());
            devs.push_back(q.developer);
            dils.push_back(q.dilution);
            eds.push_back(q.edition);
            vess.push_back(q.vessel);
            fogs.push_back(std::vector<std::pair<HighPrecType, HighPrecType>>());
        }
        groups[slot].push_back(
            std::make_pair(static_cast<HighPrecType>(q.minutes),
                           static_cast<HighPrecType>(v)));
        if (q.base_fog > 0.0)
        {
            fogs[slot].push_back(
                std::make_pair(static_cast<HighPrecType>(q.minutes),
                               static_cast<HighPrecType>(q.base_fog)));
        }
    }

    auto sortByTime = [](std::vector<std::pair<HighPrecType, HighPrecType>>& g)
    {
        // Insertion sort: the largest group in the database is a few dozen.
        for (std::size_t i = 1u; i < g.size(); ++i)
        {
            std::pair<HighPrecType, HighPrecType> v = g[i];
            std::size_t j = i;
            while ((j > 0u) && (g[j - 1u].first > v.first))
            {
                g[j] = g[j - 1u];
                --j;
            }
            g[j] = v;
        }
    };

    // A group of one cannot describe an axis.
    std::vector<std::size_t> live;
    bool anyGamma = false;
    for (std::size_t k = 0u; k < groups.size(); ++k)
    {
        if (groups[k].size() >= 2u)
        {
            live.push_back(k);
            anyGamma = anyGamma || (0 == kinds[k]);
        }
    }
    if (live.empty())
        return;

    const bool ciOnly = !anyGamma;
    {
        std::vector<std::size_t> keep;
        for (std::size_t i = 0u; i < live.size(); ++i)
        {
            const std::size_t k = live[i];
            if (!ciOnly)
            {
                if (0 == kinds[k])
                    keep.push_back(k);
                continue;
            }
            // A development curve RISES with time (tolerance 0.01).
            std::vector<std::pair<HighPrecType, HighPrecType>> g = groups[k];
            sortByTime(g);
            bool rising = true;
            for (std::size_t j = 1u; j < g.size(); ++j)
            {
                if (g[j].second < g[j - 1u].second - static_cast<HighPrecType>(0.01))
                {
                    rising = false;
                    break;
                }
            }
            if (rising)
                keep.push_back(k);
        }
        live = keep;
    }
    if (live.empty())
        return;

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
    auto ndil = [&norm](const std::string& s) -> std::string
    {
        const std::string n = norm(s);
        return (n == "stock") ? std::string() : n;   // undiluted, printed both ways
    };

    const std::string want = norm(p.processing.developer);
    const std::string ref  = norm(p.processing_family.reference_developer);
    const bool        rdilSet = !norm(p.processing_family.reference_dilution).empty();
    const std::string rdil = ndil(p.processing_family.reference_dilution);

    std::vector<std::size_t> named;
    if (!want.empty())
    {
        for (std::size_t i = 0u; i < live.size(); ++i)
        {
            if (norm(devs[live[i]]) == want)
                named.push_back(live[i]);
        }
    }
    // \warning THE FAMILY'S OWN REFERENCE DEVELOPER (schema v36) - consulted
    // when ProcessingSpec::developer is empty (every 1956-sourced stock: the
    // stored curve is a later sheet, the points sit beside the SOURCE's own
    // curve), and for a contrast-index family whose spelling of the developer
    // differs from the profile's (EKTAPAN: 'KODAK HC-110 (Dilution B)').
    if (named.empty() && !ref.empty() && (ciOnly || want.empty()))
    {
        for (std::size_t i = 0u; i < live.size(); ++i)
        {
            if ((norm(devs[live[i]]) == ref)
                && (!rdilSet || (ndil(dils[live[i]]) == rdil)))
            {
                named.push_back(live[i]);
            }
        }
    }
    if (!named.empty())
        live = named;
    else if (ciOnly)
        return;                     // a CI family is never a guess

    // schema v59: THE EDITION the stored curve follows, when the family names
    // one -- after the developer choice, and only if it matches.
    const std::string red = norm(p.processing_family.reference_edition);
    if (!red.empty())
    {
        std::vector<std::size_t> ed;
        for (std::size_t i = 0u; i < live.size(); ++i)
        {
            if (norm(eds[live[i]]) == red)
                ed.push_back(live[i]);
        }
        if (!ed.empty())
            live = ed;
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
    sortByTime(out);

    if (nullptr != outDeveloper) *outDeveloper = devs[pick];
    if (nullptr != outVessel)    *outVessel    = vess[pick];
    if ((nullptr != outFog) && (fogs[pick].size() >= 2u))
    {
        *outFog = fogs[pick];
        sortByTime(*outFog);
    }
}


// ---------------------------------------------------------------------------
//  AlgoDevelopmentAxis  (#209, 2026-10-16)
//
//  Everything the Development Time control reads, built once: the selected
//  group's points, the traced range, the anchor time, the governing RATE LAW
//  if there is one, and the group's own base-fog points.
//  film_sim.development_axis is the reference.
//
//  THE LAW (schema v29 / v41 fields, first read here). Chosen the way
//  ProcessingFamily.validate checks one, so the law that drives the control
//  is the law the build has already tested against these very points:
//    1. a DevelopmentLaw in `laws` matching the group's developer and vessel
//       (empty strings are wildcards; the first match wins);
//    2. otherwise the family-level law, ONLY when `laws` is empty;
//    3. otherwise none, and the points are read on the linear chord.
//  Inside the traced range the law is evaluated as fitted. Past the last
//  point it continues up to Glafkides' 0.80 * gamma_inf usable ceiling, or
//  is held at the last traced value where the data already exceed that
//  ceiling (Jones & Crabtree's MQ 80 reaches 0.90 on a 0.993 asymptote).
//
//  THE FOG. DevelopmentPoint::base_fog of the SAME group, >= 2 points,
//  linear in time and held flat outside them.
// ---------------------------------------------------------------------------
struct AlgoDevelopmentAxis
{
    std::vector<std::pair<HighPrecType, HighPrecType>> pts;
    std::vector<std::pair<HighPrecType, HighPrecType>> fog;
    HighPrecType lo     = 0;
    HighPrecType hi     = 0;
    HighPrecType ref    = 0;
    bool         hasLaw = false;
    HighPrecType gInf   = 0;
    HighPrecType k      = 0;
    HighPrecType t0     = 0;
    HighPrecType top    = 0;     ///< ceiling past the last traced point

    HighPrecType Raw(const HighPrecType t) const noexcept
    {
        const HighPrecType dt = t - t0;
        if (dt <= static_cast<HighPrecType>(0))
            return static_cast<HighPrecType>(0);
        return gInf * (static_cast<HighPrecType>(1) - std::exp(-k * dt));
    }

    HighPrecType At(const HighPrecType t) const noexcept
    {
        if (hasLaw)
        {
            const HighPrecType g = Raw(t);
            return ((t > hi) && (g > top)) ? top : g;
        }
        for (std::size_t i = 1u; i < pts.size(); ++i)
        {
            const HighPrecType ta = pts[i - 1u].first;
            const HighPrecType tb = pts[i].first;
            if ((t >= ta) && (t <= tb))
            {
                if (tb == ta)
                    return pts[i - 1u].second;
                return pts[i - 1u].second
                     + (pts[i].second - pts[i - 1u].second)
                     * (t - ta) / (tb - ta);
            }
        }
        return pts.back().second;
    }

    HighPrecType FogAt(const HighPrecType t) const noexcept
    {
        if (t <= fog.front().first) return fog.front().second;
        if (t >= fog.back().first)  return fog.back().second;
        for (std::size_t i = 1u; i < fog.size(); ++i)
        {
            const HighPrecType ta = fog[i - 1u].first;
            const HighPrecType tb = fog[i].first;
            if ((t >= ta) && (t <= tb))
            {
                if (tb == ta)
                    return fog[i - 1u].second;
                return fog[i - 1u].second
                     + (fog[i].second - fog[i - 1u].second)
                     * (t - ta) / (tb - ta);
            }
        }
        return fog.back().second;
    }
};

inline bool AlgoDevelopmentAxisBuild
(
    const film::FilmProfile& p,
    AlgoDevelopmentAxis&     ax
) noexcept
{
    if (!p.is_monochrome)
        return false;

    std::string dev;
    std::string vessel;
    AlgoDevelopmentGroup(p, ax.pts, &dev, &vessel, &ax.fog);
    if (ax.pts.size() < 2u)
        return false;

    ax.lo = ax.pts.front().first;
    ax.hi = ax.pts.back().first;

    // The anchor: ProcessingSpec::minutes inside the range, else the time the
    // CHORD gives the stored gamma (unchanged from before #209 -- the anchor is
    // a time, and the same time is found whether or not a law then reads it).
    HighPrecType ref = static_cast<HighPrecType>(p.processing.minutes);
    if ((ref < ax.lo) || (ref > ax.hi))
    {
        const HighPrecType gStored = static_cast<HighPrecType>(p.curves.g.gamma);
        const HighPrecType gA = ax.pts.front().second;
        const HighPrecType gB = ax.pts.back().second;
        const HighPrecType gLo = (gA < gB) ? gA : gB;
        const HighPrecType gHi = (gA < gB) ? gB : gA;

        if ((gStored < gLo) || (gStored > gHi))
            return false;

        bool found = false;
        for (std::size_t i = 1u; i < ax.pts.size(); ++i)
        {
            const HighPrecType g0 = ax.pts[i - 1u].second;
            const HighPrecType g1 = ax.pts[i].second;
            const HighPrecType a  = (g0 < g1) ? g0 : g1;
            const HighPrecType b  = (g0 < g1) ? g1 : g0;

            if ((gStored >= a) && (gStored <= b))
            {
                const HighPrecType ta = ax.pts[i - 1u].first;
                const HighPrecType tb = ax.pts[i].first;
                ref = (g1 == g0)
                    ? ta
                    : (ta + (tb - ta) * (gStored - g0) / (g1 - g0));
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }
    ax.ref = ref;

    // The governing law (see the header note).
    const film::ProcessingFamily& fam = p.processing_family;
    ax.hasLaw = false;
    for (std::size_t i = 0u; i < fam.laws.size(); ++i)
    {
        const film::DevelopmentLaw& L = fam.laws[i];
        if (!L.hasRateLaw())
            continue;
        if (!L.developer.empty() && !dev.empty() && (L.developer != dev))
            continue;
        if (!L.vessel.empty() && !vessel.empty() && (L.vessel != vessel))
            continue;
        ax.hasLaw = true;
        ax.gInf = static_cast<HighPrecType>(L.gamma_infinity);
        ax.k    = static_cast<HighPrecType>(L.dev_rate_k);
        ax.t0   = static_cast<HighPrecType>(L.induction_t0_min);
        break;
    }
    if (!ax.hasLaw && fam.laws.empty() && fam.hasRateLaw())
    {
        ax.hasLaw = true;
        ax.gInf = static_cast<HighPrecType>(fam.gamma_infinity);
        ax.k    = static_cast<HighPrecType>(fam.dev_rate_k);
        ax.t0   = static_cast<HighPrecType>(fam.induction_t0_min);
    }
    if (ax.hasLaw)
    {
        const HighPrecType ceil = static_cast<HighPrecType>(0.80) * ax.gInf;
        const HighPrecType gHi  = ax.Raw(ax.hi);
        ax.top = (ceil > gHi) ? ceil : gHi;
    }
    return true;
}


// ---------------------------------------------------------------------------
//  AlgoDevelopmentTimeEffect
//
//  (gamma scale, base-fog shift) for `minutes`; (1, 0) on every refusal path,
//  which is what keeps the sentinel bit-exact. Below the first traced point a
//  time is refused; above the last one it is refused unless a law governs.
// ---------------------------------------------------------------------------
inline void AlgoDevelopmentTimeEffect
(
    const film::FilmProfile& p,
    const HighPrecType       minutes,
    HighPrecType&            scale,
    HighPrecType&            fogShift
) noexcept
{
    scale    = static_cast<HighPrecType>(1);
    fogShift = static_cast<HighPrecType>(0);

    if (minutes < static_cast<HighPrecType>(0))
        return;

    AlgoDevelopmentAxis ax;
    if (!AlgoDevelopmentAxisBuild(p, ax))
        return;
    if (minutes < ax.lo)
        return;
    if ((minutes > ax.hi) && !ax.hasLaw)
        return;

    const HighPrecType gRef = ax.At(ax.ref);
    if (gRef <= static_cast<HighPrecType>(0))
        return;

    scale = ax.At(minutes) / gRef;
    if (ax.fog.size() >= 2u)
        fogShift = ax.FogAt(minutes) - ax.FogAt(ax.ref);
}


// ---------------------------------------------------------------------------
//  AlgoDevelopmentGammaScale -- the gamma half of the above, kept for callers.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDevelopmentGammaScale
(
    const film::FilmProfile& p,
    const HighPrecType       minutes
) noexcept
{
    HighPrecType k;
    HighPrecType d;
    AlgoDevelopmentTimeEffect(p, minutes, k, d);
    return k;
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
//  stated in both files so they cannot drift apart. dmin moves by the group's
//  own fog shift (#209); exactly unchanged when that shift is zero.
// ---------------------------------------------------------------------------
//  DEVELOPMENT TEMPERATURE, added 2026-09-17d (schema v39)
//  ------------------------------------------------------
//  AlgoControl.hpp's developmentCelsius block stated its own blocker: "time-
//  temperature equivalence charts DO exist for real developers, but none has
//  been adopted here, and substituting a published chart from another
//  developer would be an invented effect". One is adopted now, and it is not
//  borrowed - ProcessingFamily::temperature_coeff_per_c is fitted to each
//  stock's OWN points, 179 series across 11 stocks, and is zero on every
//  stock whose family holds one temperature.
//
//  The law is d ln(t)/dT at CONSTANT CONTRAST, so it converts a requested
//  (time, temperature) into the time at the family's own reference
//  temperature that develops to the same gamma:
//
//      t_equiv = t * exp(coeff * (T_ref - T))
//
//  and the existing time-gamma family then reads that equivalent time. The
//  reference temperature is the one the family's own points were measured at
//  most often, which is what AlgoDevelopmentRefCelsius returns.
//
//  \warning INERT UNLESS BOTH CONTROLS MOVE. celsius < 0 is the sentinel, and
//  a stock with no fitted coefficient takes the identity path, so every render
//  before v39 is reproduced bit for bit.
inline HighPrecType AlgoDevelopmentRefCelsius
(
    const film::FilmProfile& base
) noexcept
{
    const std::vector<film::DevelopmentPoint>& pts =
        base.processing_family.points;
    if (base.processing.celsius > 0.0)
        return static_cast<HighPrecType>(base.processing.celsius);

    HighPrecType best = static_cast<HighPrecType>(0);
    std::size_t bestN = 0;
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        if (pts[i].celsius <= 0.0) continue;
        std::size_t n = 0;
        for (std::size_t j = 0; j < pts.size(); ++j)
            if (pts[j].celsius == pts[i].celsius) ++n;
        if (n > bestN)
        {
            bestN = n;
            best  = static_cast<HighPrecType>(pts[i].celsius);
        }
    }
    return best;
}

//  The requested time, restated at the family's reference temperature.
//  Returns `minutes` unchanged on every inert path.
inline HighPrecType AlgoDevelopmentEquivalentMinutes
(
    const film::FilmProfile& base,
    const HighPrecType       minutes,
    const HighPrecType       celsius
) noexcept
{
    const HighPrecType c = static_cast<HighPrecType>(
        base.processing_family.temperature_coeff_per_c);

    if (minutes <= static_cast<HighPrecType>(0)) return minutes;
    if (celsius <= static_cast<HighPrecType>(0)) return minutes;

    if (c >= static_cast<HighPrecType>(0))
    {
        // #209 (2026-10-16): THE FAMILY'S OWN Q10 where no slope is fitted
        // (Glafkides §86, stored on three stocks). The fitted slope stays
        // first -- it is the stock's own measurement. VALID 13-24 degC at both
        // ends; outside that span the time is kept, as with no rule at all.
        const HighPrecType q10 = static_cast<HighPrecType>(
            base.processing_family.temp_q10);
        if (q10 <= static_cast<HighPrecType>(0))     return minutes;

        const HighPrecType ref = AlgoDevelopmentRefCelsius(base);
        if (ref <= static_cast<HighPrecType>(0))     return minutes;
        if (ref == celsius)                          return minutes;

        const HighPrecType tLo = static_cast<HighPrecType>(13);
        const HighPrecType tHi = static_cast<HighPrecType>(24);
        if ((celsius < tLo) || (celsius > tHi) || (ref < tLo) || (ref > tHi))
            return minutes;

        return minutes * std::pow(q10, (celsius - ref) / static_cast<HighPrecType>(10));
    }

    const HighPrecType ref = AlgoDevelopmentRefCelsius(base);
    if (ref <= static_cast<HighPrecType>(0))     return minutes;
    if (ref == celsius)                          return minutes;

    return minutes * std::exp(c * (ref - celsius));
}

// ---------------------------------------------------------------------------
//  rms multiplier for a development that scales gamma by k (schema v59).
//
//  rms ~ gamma^n (GrainSpec::rms_gamma_exponent), k clamped to [1/span, span]
//  (rms_gamma_span): the law is a fit over that span of measured gammas and is
//  held flat beyond it rather than extrapolated. 1.0 on every stock without a
//  measured law. film_sim.development_rms_factor is the same function.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoDevelopmentRmsFactor
(
    const film::GrainSpec& g,
    const HighPrecType     k
) noexcept
{
    const HighPrecType n    = static_cast<HighPrecType>(g.rms_gamma_exponent);
    const HighPrecType span = static_cast<HighPrecType>(g.rms_gamma_span);
    const HighPrecType one  = static_cast<HighPrecType>(1);

    if (!(n > static_cast<HighPrecType>(0)) || !(span > one)
        || !(k > static_cast<HighPrecType>(0)) || (k == one))
        return one;

    const HighPrecType kc = MIN_VALUE(MAX_VALUE(k, one / span), span);
    return std::pow(kc, n);
}


inline const film::FilmProfile& AlgoResolveDevelopmentTime
(
    const film::FilmProfile& base,
    const double             minutes,
    const double             celsius,
    film::FilmProfile&       store
) noexcept
{
    const HighPrecType t = AlgoDevelopmentEquivalentMinutes(
        base,
        static_cast<HighPrecType>(minutes),
        static_cast<HighPrecType>(celsius));

    HighPrecType k;
    HighPrecType d;
    AlgoDevelopmentTimeEffect(base, t, k, d);

    if ((k == static_cast<HighPrecType>(1)) && (d == static_cast<HighPrecType>(0)))
        return base;

    store = base;

    store.curves.r.gamma =
        static_cast<float>(static_cast<HighPrecType>(base.curves.r.gamma) * k);
    store.curves.g.gamma =
        static_cast<float>(static_cast<HighPrecType>(base.curves.g.gamma) * k);
    store.curves.b.gamma =
        static_cast<float>(static_cast<HighPrecType>(base.curves.b.gamma) * k);

    // #209: the group's own base fog moves D-min (d == 0 leaves it bit-exact:
    // a float widened, plus zero, narrowed back, is the same float).
    store.curves.r.dmin =
        static_cast<float>(static_cast<HighPrecType>(base.curves.r.dmin) + d);
    store.curves.g.dmin =
        static_cast<float>(static_cast<HighPrecType>(base.curves.g.dmin) + d);
    store.curves.b.dmin =
        static_cast<float>(static_cast<HighPrecType>(base.curves.b.dmin) + d);

    // schema v59: the measured rms-vs-gamma law. Every rms figure moves by the
    // same factor; grain_um_* is emitted RESOLVED and is left as it is, so the
    // spectrum keeps its shape and only its level follows the development --
    // film_sim.resolve_development_time pins grain_um for the same reason.
    const HighPrecType f = AlgoDevelopmentRmsFactor(base.grain, k);
    if (f != static_cast<HighPrecType>(1))
    {
        store.grain.rms_granularity = static_cast<float>(
            static_cast<HighPrecType>(base.grain.rms_granularity) * f);
        store.grain.rms_r = static_cast<float>(
            static_cast<HighPrecType>(base.grain.rms_r) * f);
        store.grain.rms_g = static_cast<float>(
            static_cast<HighPrecType>(base.grain.rms_g) * f);
        store.grain.rms_b = static_cast<float>(
            static_cast<HighPrecType>(base.grain.rms_b) * f);
    }


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


// ---------------------------------------------------------------------------
//  RECIPROCITY CONTRAST  (#209 item 4, 2026-10-16)
//
//  ReciprocityTable::development_correction_pct sits beside the exposure
//  correction on six sheets and was read by nothing. A maker who prints
//  "develop 20 % less at 10 s" states that the exposure ALONE raises contrast
//  by exactly what 20 % less development takes away. On the stock's own
//  development axis:
//
//      m = gamma(t_ref) / gamma(t_ref * (1 + pct / 100))
//
//  The corrected time is HELD at the ends of the traced range (at the first
//  point; at the last one where no law governs) -- the largest gain the data
//  support, never an extrapolated one. 1.0 on every inert path.
//  film_sim.reciprocity_development_pct / reciprocity_contrast_scale are the
//  reference.
//
//  \warning READ OFF THE PROFILE BEFORE THE DEVELOPMENT CONTROLS (the anchor
//  of a stock with no ProcessingSpec::minutes is inverted from its stored
//  gamma), APPLIED AFTER THEM -- see AlgorithmMain.cpp.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoReciprocityDevelopmentPct
(
    const film::FilmProfile& p,
    const HighPrecType       exposureTimeS
) noexcept
{
    const HighPrecType zero = static_cast<HighPrecType>(0);
    if (exposureTimeS <= zero)
        return zero;

    const film::ReciprocityTable& tab = p.reciprocity_table;
    const std::size_t n = tab.times_s.size();
    if ((0u == n) || (tab.development_correction_pct.size() != n))
        return zero;

    const HighPrecType lt = static_cast<HighPrecType>(
        std::log10(static_cast<double>(exposureTimeS)));
    const HighPrecType x0 = static_cast<HighPrecType>(
        std::log10(static_cast<double>(tab.times_s[0])));
    const HighPrecType xN = static_cast<HighPrecType>(
        std::log10(static_cast<double>(tab.times_s[n - 1u])));

    if (lt <= x0)
        return static_cast<HighPrecType>(tab.development_correction_pct[0]);
    if (lt >= xN)
        return static_cast<HighPrecType>(tab.development_correction_pct[n - 1u]);

    std::size_t k = 0u;
    while ((k + 1u < n)
           && (static_cast<HighPrecType>(
                   std::log10(static_cast<double>(tab.times_s[k + 1u]))) < lt))
        ++k;

    const HighPrecType xa = static_cast<HighPrecType>(
        std::log10(static_cast<double>(tab.times_s[k])));
    const HighPrecType xb = static_cast<HighPrecType>(
        std::log10(static_cast<double>(tab.times_s[k + 1u])));
    const HighPrecType span = xb - xa;
    const HighPrecType f = (span <= zero) ? zero : (lt - xa) / span;
    const HighPrecType ya = static_cast<HighPrecType>(tab.development_correction_pct[k]);
    const HighPrecType yb = static_cast<HighPrecType>(tab.development_correction_pct[k + 1u]);
    return ya + f * (yb - ya);
}

inline HighPrecType AlgoReciprocityContrastScale
(
    const film::FilmProfile& p,
    const HighPrecType       exposureTimeS
) noexcept
{
    const HighPrecType one = static_cast<HighPrecType>(1);

    const HighPrecType pct = AlgoReciprocityDevelopmentPct(p, exposureTimeS);
    if (pct == static_cast<HighPrecType>(0))
        return one;

    AlgoDevelopmentAxis ax;
    if (!AlgoDevelopmentAxisBuild(p, ax))
        return one;

    HighPrecType t1 = ax.ref * (one + pct / static_cast<HighPrecType>(100));
    if (t1 < ax.lo)
        t1 = ax.lo;
    if ((t1 > ax.hi) && !ax.hasLaw)
        t1 = ax.hi;

    const HighPrecType g0 = ax.At(ax.ref);
    const HighPrecType g1 = ax.At(t1);
    if ((g0 <= static_cast<HighPrecType>(0)) || (g1 <= static_cast<HighPrecType>(0)))
        return one;
    return g0 / g1;
}

//  `base` with its gamma multiplied by `scale`; `base` itself at exactly 1.
//  The measured tables are dropped, as at every other gamma change.
inline const film::FilmProfile& AlgoResolveReciprocityContrast
(
    const film::FilmProfile& base,
    const HighPrecType       scale,
    film::FilmProfile&       store
) noexcept
{
    if (scale == static_cast<HighPrecType>(1))
        return base;

    store = base;
    film::ToneCurve* const c[3] = { &store.curves.r, &store.curves.g, &store.curves.b };
    for (int32_t i = 0; i < 3; i++)
    {
        c[i]->gamma = static_cast<float>(static_cast<HighPrecType>(c[i]->gamma) * scale);
        c[i]->meas_x = nullptr;
        c[i]->meas_d = nullptr;
        c[i]->meas_m = nullptr;
        c[i]->meas_n = 0;
    }
    return store;
}


// ---------------------------------------------------------------------------
//  THE DEVELOPER, schema v60
//
//  film::ProcessingFamily::grain_points holds one film measured in several
//  developers by ONE laboratory (Bernhard W. Schmidt's tests on four stocks).
//  The Developer control selects a row; this returns the two RATIOS that row
//  implies against the family's reference row:
//
//      rms factor        rms_d[index] / rms_d[reference]
//      adjacency factor  development_halo_width_um[index] / ...[reference]
//
//  RATIOS ONLY, because the source's absolute rms scale is not Kodak's 48 um
//  diffuse scale: within one film the instrument cancels. (1, 1) on every
//  inert path -- index < 0, out of range, no rows, no reference row, the
//  reference row itself, or a quantity missing on either row.
//  film_sim.developer_factors is the same function.
// ---------------------------------------------------------------------------
inline bool AlgoDeveloperNameEq
(
    const std::string& a,
    const std::string& b
) noexcept
{
    // Python compares q.developer.strip().lower() to the reference name the
    // same way; ASCII is enough for every developer name in the database.
    std::size_t a0 = 0, a1 = a.size(), b0 = 0, b1 = b.size();
    while ((a0 < a1) && (' ' == a[a0]))     ++a0;
    while ((a1 > a0) && (' ' == a[a1 - 1])) --a1;
    while ((b0 < b1) && (' ' == b[b0]))     ++b0;
    while ((b1 > b0) && (' ' == b[b1 - 1])) --b1;
    if ((a1 - a0) != (b1 - b0)) return false;
    for (std::size_t i = 0; i < (a1 - a0); ++i)
    {
        char ca = a[a0 + i], cb = b[b0 + i];
        if ((ca >= 'A') && (ca <= 'Z')) ca = static_cast<char>(ca - 'A' + 'a');
        if ((cb >= 'A') && (cb <= 'Z')) cb = static_cast<char>(cb - 'A' + 'a');
        if (ca != cb) return false;
    }
    return true;
}

inline void AlgoDeveloperFactors
(
    const film::FilmProfile& base,
    const int32_t            index,
    HighPrecType&            fRms,
    HighPrecType&            fAdj
) noexcept
{
    const HighPrecType one  = static_cast<HighPrecType>(1);
    const HighPrecType zero = static_cast<HighPrecType>(0);
    fRms = one;
    fAdj = one;

    const std::vector<film::DeveloperGrainPoint>& rows =
        base.processing_family.grain_points;
    if ((index < 0) || (static_cast<std::size_t>(index) >= rows.size()))
        return;

    const film::DeveloperGrainPoint* ref = nullptr;
    for (std::size_t i = 0; i < rows.size(); ++i)
        if (AlgoDeveloperNameEq(rows[i].developer,
                base.processing_family.grain_reference_developer))
        {
            ref = &rows[i];
            break;
        }
    if (nullptr == ref)
        return;

    const film::DeveloperGrainPoint& q = rows[static_cast<std::size_t>(index)];
    const HighPrecType qr = static_cast<HighPrecType>(q.rms_d);
    const HighPrecType rr = static_cast<HighPrecType>(ref->rms_d);
    const HighPrecType qh = static_cast<HighPrecType>(q.development_halo_width_um);
    const HighPrecType rh = static_cast<HighPrecType>(ref->development_halo_width_um);
    if ((qr > zero) && (rr > zero)) fRms = qr / rr;
    if ((qh > zero) && (rh > zero)) fAdj = qh / rh;
}

// ---------------------------------------------------------------------------
//  AlgoDeveloperRowFromCtrl -- the stock's own grain_points row for the
//  DeveloperCtrl list-box value, or -1 (render as shipped) when the value is
//  eAS_SHIPPED, out of range, or a developer this stock was not measured in.
//  2026-10-16f. Twin: film_sim.developer_row_for.
// ---------------------------------------------------------------------------
inline int32_t AlgoDeveloperRowFromCtrl
(
    const film::FilmProfile& base,
    const DeveloperCtrl      ctrl
) noexcept
{
    const int32_t v = static_cast<int32_t>(ctrl);
    if ((v <= 0) || (v >= DeveloperCtrlCount))
        return -1;
    const std::string key(DeveloperCtrlKey[v]);
    if (key.empty())
        return -1;
    const std::vector<film::DeveloperGrainPoint>& rows =
        base.processing_family.grain_points;
    for (std::size_t i = 0; i < rows.size(); ++i)
        if (AlgoDeveloperNameEq(rows[i].developer, key))
            return static_cast<int32_t>(i);
    return -1;
}

// ---------------------------------------------------------------------------
//  AlgoResolveDeveloper -- the profile as developer row `index` renders it.
//
//  Same contract as AlgoResolveDevelopmentTime: returns `base` by reference on
//  every inert path and writes `store` only when something moves. rms moves by
//  the rms ratio (grain_um_* is emitted RESOLVED and is left alone, so the
//  spectrum keeps its shape); MTFSpec::adjacency_um moves by the halo ratio.
//  The adjacency STRENGTH is not touched: the source measures a width.
//  Applied in frame setup straight after the development time.
// ---------------------------------------------------------------------------
inline const film::FilmProfile& AlgoResolveDeveloper
(
    const film::FilmProfile& base,
    const int32_t            index,
    film::FilmProfile&       store
) noexcept
{
    HighPrecType fRms, fAdj;
    AlgoDeveloperFactors(base, index, fRms, fAdj);

    const HighPrecType one = static_cast<HighPrecType>(1);
    if ((fRms == one) && (fAdj == one))
        return base;

    store = base;
    if (fRms != one)
    {
        store.grain.rms_granularity = static_cast<float>(
            static_cast<HighPrecType>(base.grain.rms_granularity) * fRms);
        store.grain.rms_r = static_cast<float>(
            static_cast<HighPrecType>(base.grain.rms_r) * fRms);
        store.grain.rms_g = static_cast<float>(
            static_cast<HighPrecType>(base.grain.rms_g) * fRms);
        store.grain.rms_b = static_cast<float>(
            static_cast<HighPrecType>(base.grain.rms_b) * fRms);
    }
    if (fAdj != one)
    {
        store.mtf.adjacency_um = static_cast<decltype(store.mtf.adjacency_um)>(
            static_cast<HighPrecType>(base.mtf.adjacency_um) * fAdj);
    }
    return store;
}
