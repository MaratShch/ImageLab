#pragma once

// ---------------------------------------------------------------------------
//  AlgoReciprocity.hpp
//
//  Reciprocity failure: the per-channel shift of log exposure that a stated
//  shutter time implies. Header only, computed ONCE PER FRAME, consumed by
//  stage 8 as three constants added to log10(exposure).
//
//  WHY IT IS NOT A STAGE OF ITS OWN
//  --------------------------------
//  There are no pixels to walk. The correction the manufacturers publish is a
//  function of TIME alone, so it is three numbers for the whole frame; giving it
//  a stage would mean an extra pass over three planes to add a constant that
//  stage 8 is about to add anyway while it already has the logarithm in a
//  register.
//
//  WHERE IT IS APPLIED, AND WHY THERE
//  ----------------------------------
//  Inside stage 8, on the log exposure, i.e. AFTER everything optical (taking
//  filters, flare, halation, the emulsion MTF, the record collapse) and BEFORE
//  the characteristic curve. That placement is the physics: reciprocity failure
//  is a property of the EMULSION's response to the light that reached it, not of
//  the light itself. Folding it into stage 2 with exposureStops would let the
//  flare and halation stages scatter light the lens never delivered - and would
//  also make a long exposure change the halation threshold, which nothing in any
//  datasheet supports.
//
//  Because the shift lands on the RETAINED log-exposure plane, the interimage
//  stage at 8b sees the same effective exposure the curve did. A real layer has
//  no way to know the difference either.
//
//  WHAT THIS MODEL IS NOT
//  ----------------------
//  It is not intensity dependent, and the reason is that nothing in the film
//  database can make it so: all six measured tables in the corpus are functions
//  of time alone. Real reciprocity failure IS intensity dependent - the darkest
//  parts of a frame fail first, which is why a long exposure loses shadow
//  separation as well as speed - but the exponent that would express that is not
//  published by any manufacturer on file. So this is an honest per-channel
//  global shift rather than a per-pixel effect with an invented exponent.
//
//  INERT BY DEFAULT. exposureTimeS <= 0 means "the caller did not state a
//  time", the shift is exactly zero, and every render made before this file
//  existed is reproduced bit for bit.
//
//  ONE LAW, TWO LANGUAGES. film_sim.reciprocity_log_shift() is the reference;
//  cpp_parity.py compiles this header and compares the two over every stock in
//  the database at a ladder of exposure times.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

// film::FilmProfile, film::ReciprocitySpec, film::ReciprocityTable.
#include "film_profiles.hpp"

#include <cmath>     // std::log10
#include <cstddef>   // std::size_t
#include <string>    // std::string


// ---------------------------------------------------------------------------
//  log10(2), i.e. one stop expressed in decades of exposure. The manufacturers
//  print their corrections in STOPS and the curve is indexed in DECADES, so
//  exactly one conversion happens, here, and it is written out to full double
//  precision rather than as 0.301.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_LOG10_OF_TWO =
    static_cast<HighPrecType>(0.30102999566398120);


// ---------------------------------------------------------------------------
//  AlgoCcFilterShift
//
//  The per-channel CREDIT against the printed correction that a prescribed CC
//  filter implies. Mirrors film_sim._cc_filter_shift() exactly.
//
//  ⚠ THIS IS AN INTERPRETATION OF A PRESCRIPTION, and the arithmetic is what
//  actually reaches the film. A datasheet does not print "the blue record loses
//  0.15 more than the others"; it prints "at 10 s, increase exposure 1 1/2 stops
//  and use a CC15B filter". Both instructions act on the same frame: the lens
//  opens by the stated stops, equally on all three records, and the filter then
//  takes part of that back from the records it attenuates. CC15B is blue, so it
//  absorbs red and green by 0.15 density each. The film therefore receives
//  +1.5 stops of blue and +1.5 stops - 0.15 decades of red and green - and since
//  the prescription is what makes the result correct, those ARE the losses:
//
//      the record the filter does NOT attenuate loses the full printed stops;
//      every attenuated record loses that MINUS the filter's density.
//
//  A CC value is already a DENSITY, i.e. base-ten log exposure - the same unit
//  the curve is indexed in - so there is deliberately no stops conversion here.
//
//  Nothing is re-referenced afterwards, and that falls out correctly for a
//  compound prescription: a set that attenuates all three records (10Y + 10M +
//  10C) is a neutral-density filter, and crediting all three equally is what an
//  ND in the light path does.
//
//  An empty or unparseable string gives all zeros, which is the ACHROMATIC case
//  and is a real, different statement from a missing measurement.
// ---------------------------------------------------------------------------
inline void AlgoCcFilterShift (const std::string& text, HighPrecType out[3]) noexcept
{
    out[0] = out[1] = out[2] = static_cast<HighPrecType>(0);

    const std::size_t n = text.size();

    for (std::size_t i = 0; i < n; /* advanced inside */)
    {
        // Skip anything that is not the start of a number: the "CC" prefix, any
        // separator, and any letter that is not preceded by digits.
        if ((text[i] < '0') || (text[i] > '9'))
        {
            i++;
            continue;
        }

        // Run of digits: the filter's density in hundredths, as printed.
        std::size_t j = i;
        int32_t     v = 0;

        while ((j < n) && (text[j] >= '0') && (text[j] <= '9'))
        {
            v = (v * 10) + static_cast<int32_t>(text[j] - '0');
            j++;
        }

        if (j >= n)
            break;

        // The letter names the colour the filter IS. Additive letters attenuate
        // the other two records, subtractive letters attenuate exactly one.
        // Anything else is not a CC prescription and is skipped rather than
        // guessed at.
        bool hit[3] = { false, false, false };
        bool known  = true;

        switch (text[j])
        {
            case 'R': case 'r':  hit[1] = hit[2] = true;  break;   // red absorbs G,B
            case 'G': case 'g':  hit[0] = hit[2] = true;  break;
            case 'B': case 'b':  hit[0] = hit[1] = true;  break;
            case 'C': case 'c':  hit[0] = true;           break;   // cyan absorbs R
            case 'M': case 'm':  hit[1] = true;           break;
            case 'Y': case 'y':  hit[2] = true;           break;
            default:             known  = false;          break;
        }

        if (known)
        {
            const HighPrecType d =
                static_cast<HighPrecType>(v) / static_cast<HighPrecType>(100);

            for (int32_t c = 0; c < 3; c++)
            {
                if (hit[c])
                    out[c] += d;
            }
        }

        i = j + 1;
    }

    return;
}


// ---------------------------------------------------------------------------
//  AlgoReciprocityLogShift
//
//  profile        stock being simulated
//  exposureTimeS  shutter open time in SECONDS. <= 0 means "not stated" and
//                 yields an exactly zero shift.
//  shift          out, three values ADDED to log10(exposure) by stage 8.
//                 Negative means the emulsion behaved as if it received less
//                 light than it did, which is what a speed loss IS.
//
//  Two data sources, tried in this order, because they are not the same claim:
//
//    * ReciprocityTable (6 stocks in the database) is the manufacturer's own
//      printed correction against time, in stops, optionally with the CC filter
//      that documents chromatic failure. Interpolated in log10 t - the axis the
//      tables are printed on - and HELD FLAT outside the measured range rather
//      than extrapolated. Kodak's tables walk the effective exponent from ~0.85
//      to ~0.70 across successive decades, so extrapolating even one decade past
//      the last entry is not a small error.
//
//    * ReciprocitySpec (105 stocks) carries one Schwarzschild exponent per
//      channel plus an onset. Effective exposure past onset is E = I * t^p, and
//      the metered exposure is H = I * t, so
//
//          log10 H_eff - log10 H = (p - 1) * log10(t / onset)
//
//      which is zero at the onset by construction and negative beyond it for
//      p < 1. 54 stocks carry p = 1.0 in all three channels and are therefore
//      inert at every time - including, correctly, FUJI NEOPAN ACROS, whose
//      sheet states no correction is needed out to 120 s.
//
//  Both branches are two-sided in t. Only EKTACHROME 64 measures the SHORT
//  exposure (high intensity) branch, from 1e-4 s; for every other stock a flash
//  duration lands on the held-flat first entry, which is exactly why the branch
//  is not extrapolated.
// ---------------------------------------------------------------------------
inline void AlgoReciprocityLogShift
(
    const film::FilmProfile& profile,
    const HighPrecType       exposureTimeS,
    HighPrecType             shift[3]
) noexcept
{
    shift[0] = shift[1] = shift[2] = static_cast<HighPrecType>(0);

    if (exposureTimeS <= static_cast<HighPrecType>(0))
        return;

    const film::ReciprocityTable& tab = profile.reciprocity_table;

    if (tab.hasData())
    {
        const std::size_t   n  = tab.times_s.size();
        const HighPrecType  lt = std::log10(exposureTimeS);

        HighPrecType stops = static_cast<HighPrecType>(0);
        HighPrecType chrom[3] = { static_cast<HighPrecType>(0),
                                  static_cast<HighPrecType>(0),
                                  static_cast<HighPrecType>(0) };

        // A cc_filters vector is allowed to be shorter than times_s (or empty),
        // in which case the missing entries are achromatic.
        const std::size_t nc = tab.cc_filters.size();

        const HighPrecType lo = std::log10(static_cast<HighPrecType>(tab.times_s[0]));
        const HighPrecType hi = std::log10(static_cast<HighPrecType>(tab.times_s[n - 1]));

        if (lt <= lo)
        {
            stops = static_cast<HighPrecType>(tab.stops_correction[0]);

            if (nc > 0)
                AlgoCcFilterShift(tab.cc_filters[0], chrom);
        }
        else if (lt >= hi)
        {
            stops = static_cast<HighPrecType>(tab.stops_correction[n - 1]);

            if (nc > (n - 1))
                AlgoCcFilterShift(tab.cc_filters[n - 1], chrom);
        }
        else
        {
            // Bracket: the last node still below lt. The tables are short (2 to
            // 4 entries) and validated ascending, so a linear scan is the whole
            // search.
            std::size_t k = 0;

            while (((k + 1) < n)
                   && (std::log10(static_cast<HighPrecType>(tab.times_s[k + 1])) < lt))
            {
                k++;
            }

            const HighPrecType x0 = std::log10(static_cast<HighPrecType>(tab.times_s[k]));
            const HighPrecType x1 = std::log10(static_cast<HighPrecType>(tab.times_s[k + 1]));
            const HighPrecType sp = x1 - x0;

            const HighPrecType f = (sp <= static_cast<HighPrecType>(0))
                                 ? static_cast<HighPrecType>(0)
                                 : (lt - x0) / sp;

            const HighPrecType y0 = static_cast<HighPrecType>(tab.stops_correction[k]);
            const HighPrecType y1 = static_cast<HighPrecType>(tab.stops_correction[k + 1]);

            stops = y0 + (f * (y1 - y0));

            HighPrecType c0[3] = { static_cast<HighPrecType>(0),
                                   static_cast<HighPrecType>(0),
                                   static_cast<HighPrecType>(0) };
            HighPrecType c1[3] = { static_cast<HighPrecType>(0),
                                   static_cast<HighPrecType>(0),
                                   static_cast<HighPrecType>(0) };

            if (nc > k)        AlgoCcFilterShift(tab.cc_filters[k],     c0);
            if (nc > (k + 1))  AlgoCcFilterShift(tab.cc_filters[k + 1], c1);

            for (int32_t c = 0; c < 3; c++)
                chrom[c] = c0[c] + (f * (c1[c] - c0[c]));
        }

        const HighPrecType base = -ALGO_LOG10_OF_TWO * stops;

        for (int32_t c = 0; c < 3; c++)
            shift[c] = base + chrom[c];

        return;
    }

    const film::ReciprocitySpec& rp = profile.reciprocity;

    const HighPrecType onset = (rp.onset_s > 0.0f)
                             ? static_cast<HighPrecType>(rp.onset_s)
                             : static_cast<HighPrecType>(1);

    if (exposureTimeS <= onset)
        return;

    const HighPrecType lr = std::log10(exposureTimeS / onset);

    shift[0] = (static_cast<HighPrecType>(rp.schwarzschild_p_r)
                - static_cast<HighPrecType>(1)) * lr;
    shift[1] = (static_cast<HighPrecType>(rp.schwarzschild_p_g)
                - static_cast<HighPrecType>(1)) * lr;
    shift[2] = (static_cast<HighPrecType>(rp.schwarzschild_p_b)
                - static_cast<HighPrecType>(1)) * lr;

    return;
}
