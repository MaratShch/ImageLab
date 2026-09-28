#pragma once

// ---------------------------------------------------------------------------
//  AlgoReciprocity.hpp -- reciprocity-law failure as three frame constants.
//
//  ⚠⚠ THIS FILE WAS LOST AND IS RECONSTRUCTED, 2026-09-11. It is referenced by
//  AlgorithmMain.cpp (the include at the top, the call before stage 8) and by
//  cpp_parity.py's reciprocity probe, and it was absent from the tree -- so
//  AlgorithmMain.cpp did not compile, in this tree or in the two deliveries
//  cut from it. Nothing noticed, for one specific reason worth recording:
//  cpp_parity.py tests for the file and SKIPS the whole reciprocity audit when
//  it is missing rather than failing. A skip printed once per build is
//  indistinguishable from a pass to anyone not reading for it. The guard that
//  existed to protect this law was the reason its disappearance was silent.
//
//  The law below is transcribed from the Python reference implementation,
//  which never stopped applying it. It is not a fresh design: the reference
//  is authoritative and this is the C++ spelling of the same arithmetic, so
//  that the parity probe -- 159 stocks x 12 exposure times -- can compare them.
//
//  ---------------------------------------------------------------------------
//  WHAT RECIPROCITY FAILURE IS
//  ---------------------------------------------------------------------------
//
//  The reciprocity law says a photographic effect depends on the PRODUCT of
//  illuminance and time: half the light for twice as long should give the same
//  density. Real silver halide disobeys it at both ends. At long exposures the
//  latent-image specks form too slowly to survive, so the film behaves as if it
//  were slower than its rating, and the loss grows with the logarithm of time.
//  Colour film loses unequally in the three records, which is why a datasheet
//  prescribes a colour-correction filter alongside the extra exposure.
//
//  ⚠ THERE ARE NO PIXELS TO WALK. Every correction on file is a function of
//  TIME ALONE, so the whole stage resolves to three constants that stage 8 adds
//  to the logarithm it is already computing. Zero per-pixel cost.
//
//  ⚠ INERT AT THE DEFAULT. `exposureTimeS` is zero unless the caller states a
//  shutter time; the shift is then exactly zero, and adding a floating zero is
//  the identity. Every render made before this stage existed is reproduced bit
//  for bit.
//
//  ---------------------------------------------------------------------------
//  TWO BRANCHES, AND THE TABLE WINS
//  ---------------------------------------------------------------------------
//
//  A stock that publishes a MEASURED table takes the table branch: the
//  manufacturer's own time / stops / CC-filter rows, interpolated in log time.
//  A stock with only a fitted Schwarzschild exponent takes the spec branch.
//  The table is preferred wherever it exists because it is a measurement and
//  the exponent is a fit to one.
//
//  Both branches return a shift in BASE-TEN LOG EXPOSURE, negative for a loss,
//  in the same unit stage 8 indexes its curve in.
// ---------------------------------------------------------------------------

#include "Common.hpp"
#include "AlgoTypes.hpp"
#include "film_profiles.hpp"

#include <cmath>     // std::log10
#include <cstddef>   // std::size_t
#include <string>


// ---------------------------------------------------------------------------
//  log10(2), to full double precision.
//
//  A datasheet prints its correction in STOPS and the curve is indexed in
//  DECADES, so one conversion stands between them. It is written once, here.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_RECIP_LOG10_2 =
    static_cast<HighPrecType>(0.301029995663981195213738894724);


// ---------------------------------------------------------------------------
//  AlgoCcFilterShift
//
//  Per-channel CREDIT, in density, against the printed stops correction that a
//  CC-filter prescription implies.
//
//  ⚠ THIS IS AN INTERPRETATION OF A PRESCRIPTION, AND THE SIGN IS THE PART TO
//  GET RIGHT. A sheet does not say "the blue record loses 0.15 more than the
//  others"; it says "at 10 s, increase exposure 1 1/2 stops and use a CC15B".
//  Both instructions act on one frame: the lens opens equally on all three
//  records, and the filter then takes part of that back from the records it
//  attenuates. A CC15B is blue, so it absorbs red and green by 0.15 density
//  each. The film receives +1.5 stops of blue and +1.5 stops - 0.15 decades of
//  red and green -- and since the prescription is what makes the result
//  correct, those ARE the losses:
//
//      the record the filter does NOT attenuate loses the full printed stops;
//      every attenuated record loses that MINUS the filter's own density.
//
//  So this returns a per-channel credit (>= 0), not a deficit. Backwards, it
//  inflates the worst record by the filter's value -- a third of a stop for any
//  CC10 -- while leaving the channel ORDERING correct, which is exactly the
//  kind of error that still looks plausible in a rendered frame.
//
//  CC values are already in DENSITY, i.e. base-ten log exposure, the same unit
//  the curve is indexed in. No stops conversion happens here and none should:
//  converting to stops and back is where a factor of 0.30103 goes missing.
//
//  ⚠ A THREE-DIGIT CC CODE IS THOUSANDTHS, NOT HUNDREDTHS, AND READING IT
//  WRONG IS A FACTOR OF TEN THAT FLIPS THE SIGN OF THE RESULT. Every code in
//  the corpus was two digits until AGFA RSX II 200 arrived with "075 Y" -- a
//  CC7.5Y, i.e. 0.075 density. Read as 75/100 it becomes a 0.75-density filter
//  whose blue credit swamps the 1-stop printed correction, and the function
//  returns +0.449 for blue: a LONGER exposure making the film FASTER, which no
//  sensitometry supports. The tell inside the data is monotonicity -- that row
//  runs 0 -> 075Y -> 15Y+05C, and 0.075 -> 0.15 ascends while 0.75 -> 0.15
//  does not.
//
//  An empty or unparseable string gives (0, 0, 0) -- the ACHROMATIC case,
//  which is a statement and not a missing measurement.
// ---------------------------------------------------------------------------
inline void AlgoCcFilterShift (const std::string& text,
                               HighPrecType out[3]) noexcept
{
    out[0] = static_cast<HighPrecType>(0);
    out[1] = static_cast<HighPrecType>(0);
    out[2] = static_cast<HighPrecType>(0);

    const std::size_t n = text.size();
    if (0u == n)
        return;

    std::size_t i = 0u;
    while (i < n)
    {
        const char ci = text[i];
        if (ci < '0' || ci > '9')
        {
            i++;
            continue;
        }

        std::size_t j = i;
        while (j < n && text[j] >= '0' && text[j] <= '9')
            j++;

        if (j >= n)
            break;

        // Uppercase the letter in place of a locale-dependent toupper.
        char letter = text[j];
        if (letter >= 'a' && letter <= 'z')
            letter = static_cast<char>(letter - 'a' + 'A');

        // Which records this letter attenuates. Additive letters attenuate the
        // other two records, subtractive letters attenuate one.
        int chan[2] = { -1, -1 };
        switch (letter)
        {
            case 'R': chan[0] = 1; chan[1] = 2; break;   // additive
            case 'G': chan[0] = 0; chan[1] = 2; break;
            case 'B': chan[0] = 0; chan[1] = 1; break;
            case 'C': chan[0] = 0;              break;   // subtractive
            case 'M': chan[0] = 1;              break;
            case 'Y': chan[0] = 2;              break;
            default:                            break;
        }

        if (chan[0] < 0)
        {
            // Not a filter letter. Skip the run and carry on, exactly as the
            // reference does -- a stray number in a prescription is ignored
            // rather than treated as an error.
            i = j + 1u;
            continue;
        }

        const std::size_t digits = j - i;
        HighPrecType value = static_cast<HighPrecType>(0);
        for (std::size_t k = i; k < j; k++)
            value = value * static_cast<HighPrecType>(10)
                  + static_cast<HighPrecType>(text[k] - '0');

        // The thousandths rule, guarded exactly as the reference guards it:
        // three digits AND a leading zero.
        const HighPrecType densCc =
            (3u == digits && '0' == text[i])
                ? value / static_cast<HighPrecType>(1000)
                : value / static_cast<HighPrecType>(100);

        out[chan[0]] += densCc;
        if (chan[1] >= 0)
            out[chan[1]] += densCc;

        i = j + 1u;
    }
}


// ---------------------------------------------------------------------------
//  AlgoReciprocityLogShift
//
//  profile        stock being simulated
//  exposureTimeS  shutter time in seconds. Zero or negative means "not stated",
//                 and the shift is then exactly zero on all three records.
//  outShift       three base-ten log-exposure shifts, r/g/b. Negative is a loss.
//
//  ⚠ THE TABLE IS HELD FLAT OUTSIDE ITS OWN RANGE AND IS NOT EXTRAPOLATED.
//  Below the first tabulated time the first row is used, above the last the
//  last row. That is deliberate: only one stock in the corpus measures the
//  SHORT-exposure branch, so for everything else a flash duration lands on the
//  held-flat first entry, and extrapolating a long-exposure fit backwards into
//  a regime nobody measured would invent a correction.
// ---------------------------------------------------------------------------
inline void AlgoReciprocityLogShift (const film::FilmProfile& profile,
                                     const HighPrecType       exposureTimeS,
                                     HighPrecType             outShift[3]) noexcept
{
    outShift[0] = static_cast<HighPrecType>(0);
    outShift[1] = static_cast<HighPrecType>(0);
    outShift[2] = static_cast<HighPrecType>(0);

    if (exposureTimeS <= static_cast<HighPrecType>(0))
        return;

    const film::ReciprocityTable& tab = profile.reciprocity_table;
    const std::size_t nt = tab.times_s.size();

    // ----------------------------------------------------------------------
    //  Table branch: the manufacturer's measured rows, interpolated in LOG
    //  time. Preferred wherever it exists, because it is a measurement and the
    //  Schwarzschild exponent below is a fit to one.
    // ----------------------------------------------------------------------
    if (nt > 0u && tab.stops_correction.size() == nt)
    {
        const HighPrecType lt = static_cast<HighPrecType>(
            std::log10(static_cast<double>(exposureTimeS)));

        HighPrecType stops = static_cast<HighPrecType>(0);
        HighPrecType chrom[3] = { static_cast<HighPrecType>(0),
                                  static_cast<HighPrecType>(0),
                                  static_cast<HighPrecType>(0) };

        const HighPrecType x0 = static_cast<HighPrecType>(
            std::log10(static_cast<double>(tab.times_s[0])));
        const HighPrecType xN = static_cast<HighPrecType>(
            std::log10(static_cast<double>(tab.times_s[nt - 1u])));

        // The CC column may be shorter than the time column; a missing entry
        // is the achromatic case, which is what an empty string yields.
        const std::size_t ncc = tab.cc_filters.size();

        if (lt <= x0)
        {
            stops = static_cast<HighPrecType>(tab.stops_correction[0]);
            if (ncc > 0u)
                AlgoCcFilterShift(tab.cc_filters[0], chrom);
        }
        else if (lt >= xN)
        {
            stops = static_cast<HighPrecType>(tab.stops_correction[nt - 1u]);
            if (ncc > nt - 1u)
                AlgoCcFilterShift(tab.cc_filters[nt - 1u], chrom);
        }
        else
        {
            // First interval whose upper edge is not below lt. The loop form
            // mirrors the reference's `while xs[k+1] < lt` exactly, so that a
            // sample landing on a knot resolves to the same interval in both.
            std::size_t k = 0u;
            while (k + 1u < nt
                   && static_cast<HighPrecType>(
                          std::log10(static_cast<double>(tab.times_s[k + 1u])))
                      < lt)
            {
                k++;
            }

            const HighPrecType xk = static_cast<HighPrecType>(
                std::log10(static_cast<double>(tab.times_s[k])));
            const HighPrecType xk1 = static_cast<HighPrecType>(
                std::log10(static_cast<double>(tab.times_s[k + 1u])));

            const HighPrecType span = xk1 - xk;
            const HighPrecType f =
                (span <= static_cast<HighPrecType>(0))
                    ? static_cast<HighPrecType>(0)
                    : (lt - xk) / span;

            const HighPrecType s0 =
                static_cast<HighPrecType>(tab.stops_correction[k]);
            const HighPrecType s1 =
                static_cast<HighPrecType>(tab.stops_correction[k + 1u]);
            stops = s0 + f * (s1 - s0);

            HighPrecType c0[3] = { static_cast<HighPrecType>(0),
                                   static_cast<HighPrecType>(0),
                                   static_cast<HighPrecType>(0) };
            HighPrecType c1[3] = { static_cast<HighPrecType>(0),
                                   static_cast<HighPrecType>(0),
                                   static_cast<HighPrecType>(0) };
            if (ncc > k)
                AlgoCcFilterShift(tab.cc_filters[k], c0);
            if (ncc > k + 1u)
                AlgoCcFilterShift(tab.cc_filters[k + 1u], c1);

            for (int c = 0; c < 3; c++)
                chrom[c] = c0[c] + f * (c1[c] - c0[c]);
        }

        // Stops to decades, negated: a printed "+1 1/2 stops" is a LOSS of
        // 1.5 stops of effective exposure, so the shift is negative.
        const HighPrecType base = -ALGO_RECIP_LOG10_2 * stops;

        for (int c = 0; c < 3; c++)
            outShift[c] = base + chrom[c];

        return;
    }

    // ----------------------------------------------------------------------
    //  Spec branch: the fitted Schwarzschild exponent.
    //
    //  Below the onset the film obeys reciprocity and the shift is zero. Above
    //  it the loss is (p - 1) * log10(t / onset) per record, which is the
    //  Schwarzschild form written in the log-exposure unit the curve uses.
    //
    //  ⚠ AN UNSET ONSET MEANS ONE SECOND, not zero. Zero would put the
    //  logarithm's argument at infinity for every exposure.
    // ----------------------------------------------------------------------
    const film::ReciprocitySpec& rp = profile.reciprocity;

    const HighPrecType onset =
        (static_cast<HighPrecType>(rp.onset_s) > static_cast<HighPrecType>(0))
            ? static_cast<HighPrecType>(rp.onset_s)
            : static_cast<HighPrecType>(1);

    if (exposureTimeS <= onset)
        return;

    const HighPrecType lr = static_cast<HighPrecType>(
        std::log10(static_cast<double>(exposureTimeS / onset)));

    outShift[0] = (static_cast<HighPrecType>(rp.schwarzschild_p_r)
                   - static_cast<HighPrecType>(1)) * lr;
    outShift[1] = (static_cast<HighPrecType>(rp.schwarzschild_p_g)
                   - static_cast<HighPrecType>(1)) * lr;
    outShift[2] = (static_cast<HighPrecType>(rp.schwarzschild_p_b)
                   - static_cast<HighPrecType>(1)) * lr;
}
