#pragma once

// ---------------------------------------------------------------------------
//  AlgoProcessVariant.hpp
//
//  The film as a chosen PROCESS renders it. Header only, resolved ONCE PER
//  FRAME, and the result is a profile - not a correction applied to one.
//
//  WHAT A VARIANT IS, AND WHY IT REPLACES THE CURVES RATHER THAN SCALING THEM
//  --------------------------------------------------------------------------
//  film::ProcessVariant records a DIFFERENT DEVELOPMENT of the same emulsion: a
//  push, a cross-process, an alternate chemistry kit. Where the manufacturer
//  plotted that development separately, the record carries its own TRACED
//  ToneCurve set, read from the same page as the profile's own curves so that
//  the difference between them is the process and nothing else. Selecting one
//  is therefore not a tweak: it is a second measured curve for the same film,
//  and the honest way to apply a measurement is to use it.
//
//  \warning 24 VARIANTS EXIST ACROSS 6 STOCKS AND ONLY 5 OF THEM CHANGE A
//  PIXEL. Four carry their own curves - KODAK PORTRA 800 at EI 1600 and
//  EI 3200, and KODAK ULTRA COLOR 400UC as E-190 prints it and at EI 800 - and
//  CINESTILL 800T's Cs2 two-bath kit carries gamma_scale 0.879. The other
//  nineteen differ only in exposure_index, which no stage reads, so selecting
//  one of those is a no-op and is deliberately left as one rather than given an
//  invented effect. All nineteen are the AGFAPAN developer variants, where Agfa
//  print an exposure index per developer and no second curve.
//
//  WHY A WHOLE PROFILE AND NOT AN EXTRA STAGE ARGUMENT
//  ---------------------------------------------------
//  profile.curves is read in four places in stage 8, twice in stage 11 for the
//  grain amplitude's Dmin and Dmax, and once in stage 13 for the dupe chain.
//  Threading a curve set through all of them would be a wide change with a
//  narrow benefit, and would leave every one of those call sites able to
//  disagree about which development it was rendering. Overriding the profile
//  once, before anything reads it, cannot.
//
//  The copy costs one film::FilmProfile assignment per frame - strings and
//  vectors included, a few microseconds against a frame measured in hundreds of
//  milliseconds - and only happens when a variant is actually selected.
//
//  INERT BY DEFAULT. processVariant < 0 means "the development the stored
//  curves represent", the base profile is returned by reference, nothing is
//  copied, and every render made before this file existed is reproduced bit for
//  bit. An index outside the range, or a variant that changes nothing, takes
//  the same path.
//
//  ONE LAW, TWO LANGUAGES. film_sim.resolve_process_variant() is the reference;
//  cpp_parity.py drives this resolver over every stock and every variant index
//  in the database and compares the curve parameters it yields.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

// film::FilmProfile, film::ProcessVariant, film::RGBCurves.
#include "film_profiles.hpp"

#include <cstdint>


// ---------------------------------------------------------------------------
//  AlgoResolveProcessVariant
//
//  base     the stock as the database holds it
//  index    index into base.process_variants, or < 0 for "no variant chosen"
//  store    scratch the caller owns for the lifetime of the frame; written to
//           ONLY when a variant actually changes something
//
//  Returns a reference to `base` when nothing is selected or the selection
//  changes nothing, and to `store` otherwise. The caller must keep `store`
//  alive as long as it uses the returned reference - it is a frame local in
//  AlgorithmMain for exactly that reason.
// ---------------------------------------------------------------------------
inline const film::FilmProfile& AlgoResolveProcessVariant
(
    const film::FilmProfile& base,
    const int32_t            index,
    film::FilmProfile&       store
) noexcept
{
    if (index < 0)
        return base;

    const std::size_t n = base.process_variants.size();

    if ((0 == n) || (static_cast<std::size_t>(index) >= n))
        return base;

    const film::ProcessVariant& v = base.process_variants[static_cast<std::size_t>(index)];

    // A variant that carries neither curves nor a gamma/dmin change and no
    // exposure index of its own is a label, not a render. Returning `base`
    // keeps the no-copy path for the nineteen AGFAPAN developer records.
    const bool movesCurves =
        v.has_curves
        || (v.gamma_scale != 1.0f)
        || (v.dmin_shift  != 0.0f);

    if (!movesCurves && (0 == v.exposure_index))
        return base;

    store = base;

    if (v.has_curves)
    {
        store.curves = v.curves;
    }
    else if (movesCurves)
    {
        // \warning THE COEFFICIENT IS SCALED, NOT THE OBSERVABLE SLOPE, and the
        // record means the coefficient: ProcessVariant::gamma_scale is
        // documented as multiplying ToneCurve::gamma, which is the model
        // parameter rather than the mid-scale slope. On a curve whose knees are
        // far apart the two agree to within a per cent; where they would not,
        // the variant that cares carries its own curves and never reaches here.
        film::ToneCurve* const c[3] =
        { &store.curves.r, &store.curves.g, &store.curves.b };

        for (int32_t k = 0; k < 3; k++)
        {
            c[k]->gamma = c[k]->gamma * v.gamma_scale;
            c[k]->dmin  = c[k]->dmin  + v.dmin_shift;
        }
    }

    if (0 != v.exposure_index)
        store.exposure_index = v.exposure_index;

    return store;
}
