#pragma once

// ---------------------------------------------------------------------------
//  AlgoProcessVariant.hpp -- the chosen PROCESS, resolved once per frame.
//
//  ⚠⚠ THIS FILE WAS LOST AND IS RECONSTRUCTED, 2026-09-11, together with
//  AlgoReciprocity.hpp. Both are included by AlgorithmMain.cpp and neither
//  existed in the tree, so that translation unit did not compile -- and it
//  shipped that way in at least two deliveries. The build gate did not catch
//  it because build.py's compile step covers the 26 generated database
//  translation units, not the engine driver, and cpp_parity.py SKIPS its
//  reciprocity audit when the header is absent rather than failing.
//
//  The law below is transcribed from the Python reference, which never
//  stopped applying it. The reference is authoritative; this is the C++
//  spelling of the same arithmetic.
//
//  ---------------------------------------------------------------------------
//  WHAT A PROCESS VARIANT IS, AND WHAT IT IS NOT
//  ---------------------------------------------------------------------------
//
//  A variant records a DIFFERENT DEVELOPMENT of the same emulsion -- a push, a
//  pull, a cross-process, an alternate kit -- and where the manufacturer
//  plotted that development separately, the record carries its own traced
//  curve set. It is not a different film and it is not a user grade.
//
//  ⚠ IT IS APPLIED BY OVERRIDING THE PROFILE, NOT BY HANDING A CURVE SET TO
//  EACH CONSUMER, and that is the whole design. The anchor solve, stage 8, the
//  grain amplitude and the duplication chain all read curves independently; if
//  the variant reached only some of them, the render would be a mixture of two
//  developments and no single stage would look wrong. Overriding once, here,
//  before anything reads a curve, is what makes them agree by construction.
//
//  ---------------------------------------------------------------------------
//  THE THREE WAYS A VARIANT CAN CARRY ITS CURVES
//  ---------------------------------------------------------------------------
//
//    1. It has its OWN traced curve set. Used verbatim -- a measurement always
//       beats a transform of another measurement.
//    2. It has only gamma_scale / dmin_shift. The shipped curves are
//       transformed by them.
//    3. It has neither, and only an exposure_index. The curves are untouched
//       and only the rating moves.
//
//  ⚠ THE COEFFICIENT IS SCALED, NOT THE OBSERVABLE SLOPE, and the record means
//  the coefficient: `gamma_scale` multiplies `ToneCurve::gamma`, which is the
//  MODEL PARAMETER, not the measured mid-scale gradient. On a curve whose knees
//  are far apart the two agree to within a per cent; where they do not, the
//  variant that cares carries its own curves and never reaches that branch.
//
//  ⚠ INERT AT THE DEFAULT. `processVariant` is -1 unless the caller selects
//  one, the caller's storage is then never written, and the returned reference
//  binds straight to the database entry -- no copy, no change, bit-identical
//  to a render made before this stage existed.
// ---------------------------------------------------------------------------

#include "Common.hpp"
#include "AlgoTypes.hpp"
#include "film_profiles.hpp"

#include <cstddef>   // std::size_t


// ---------------------------------------------------------------------------
//  AlgoResolveProcessVariant
//
//  asShipped  the stock as the database holds it. Never modified.
//  index      which variant the caller selected. Negative, or out of range,
//             means "none" and the shipped profile is returned unchanged.
//  store      caller-owned storage for the overridden copy. Written ONLY when
//             a variant actually changes something; untouched otherwise.
//
//  Returns a reference that is either `asShipped` itself or `store`.
//
//  ⚠ THE RETURN IS A REFERENCE AND `store` MUST OUTLIVE IT. The caller in
//  AlgorithmMain.cpp declares `variantStore` in the same scope as the render,
//  which is what makes that safe. Returning by value instead would copy a
//  FilmProfile -- vectors and all -- once per frame for a feature that is off
//  by default.
// ---------------------------------------------------------------------------
inline const film::FilmProfile& AlgoResolveProcessVariant
(
    const film::FilmProfile& asShipped,
    const int32_t            index,
    film::FilmProfile&       store
) noexcept
{
    if (index < 0)
        return asShipped;

    const std::size_t n = asShipped.process_variants.size();
    if (0u == n || static_cast<std::size_t>(index) >= n)
        return asShipped;

    const film::ProcessVariant& v =
        asShipped.process_variants[static_cast<std::size_t>(index)];

    // ----------------------------------------------------------------------
    //  Decide the curve set first, without writing anything.
    //
    //  `hasCurves` is the generated flag that says whether the variant's own
    //  curve set was populated -- the C++ spelling of the reference's
    //  `v.curves is not None`.
    // ----------------------------------------------------------------------
    const bool ownCurves = v.has_curves;

    const bool scaled =
        (false == ownCurves)
        && (v.gamma_scale != static_cast<decltype(v.gamma_scale)>(1)
            || v.dmin_shift != static_cast<decltype(v.dmin_shift)>(0));

    // Nothing about the curves moves, and no rating override either: the
    // shipped profile IS the answer, and returning it by reference avoids a
    // per-frame copy.
    if ((false == ownCurves) && (false == scaled)
        && (static_cast<decltype(v.exposure_index)>(0) == v.exposure_index))
    {
        return asShipped;
    }

    store = asShipped;

    if (ownCurves)
    {
        store.curves = v.curves;
    }
    else if (scaled)
    {
        // ⚠ THE COEFFICIENT, NOT THE SLOPE. See the note at the top.
        film::ToneCurve* const c[3] =
            { &store.curves.r, &store.curves.g, &store.curves.b };

        for (int i = 0; i < 3; i++)
        {
            c[i]->gamma = static_cast<decltype(c[i]->gamma)>(
                c[i]->gamma * v.gamma_scale);
            c[i]->dmin = static_cast<decltype(c[i]->dmin)>(
                c[i]->dmin + v.dmin_shift);
        }
    }

    // ⚠ ZERO MEANS "NOT STATED", NOT "ISO 0". A variant that only redevelops
    // without restating a rating keeps the stock's own exposure index.
    if (v.exposure_index != static_cast<decltype(v.exposure_index)>(0))
        store.exposure_index = v.exposure_index;

    return store;
}
