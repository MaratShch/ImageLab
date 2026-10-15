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
//  \warning WHAT IS OFFERED (2026-10-11, six steps). A stock offers a step
//  when it carries a record keyed to it: its own traced development at that
//  push_stops (full curves -- Scala 200x, PORTRA 800, PJ400, PJ800, 400UC,
//  T-MAX P3200, Gevachrome 605, Super Anscochrome), or, for a colour negative
//  without one at Push +1 / +2, the FAMILY MODEL record (gamma_scale and
//  dmin_shift, an estimate). Records that are not a step stay in the database
//  with an empty id and are never selected.
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
//  INERT BY DEFAULT. eAS_SHIPPED means "the development the stored curves
//  represent", the base profile is returned by reference, nothing is copied,
//  and every render made before this file existed is reproduced bit for bit.
//  A value outside the range, a step this stock does not offer, or a variant
//  that changes nothing, takes the same path.
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

// ProcessVariantCtrl, its key table and its validity test. The
// enumeration is defined ONCE, there, and used here rather than
// restated -- see the note below.
#include "AlgoControlEnums.hpp"

// ePUSH_FAMILY_* -- the colour-negative push family model, generated from the
// database (film_profiles.PUSH_FAMILY_MODEL) by cpp_codegen.py.
#include "film_enum.hpp"

#include <cmath>     // std::lround
#include <cstdint>
#include <cstddef>


// ---------------------------------------------------------------------------
//  AlgoResolveProcessVariant
//
//  base     the stock as the database holds it
//  variant  the selected step (ProcessVariantCtrl); eAS_SHIPPED for none
//  store    scratch the caller owns for the lifetime of the frame; written to
//           ONLY when a variant actually changes something
//
//  Returns a reference to `base` when nothing is selected or the selection
//  changes nothing, and to `store` otherwise. The caller must keep `store`
//  alive as long as it uses the returned reference - it is a frame local in
//  AlgorithmMain for exactly that reason.
//
//  \warning THE SELECTION IS AN ENUMERATION AND NO LONGER AN INDEX, 2026-09-17.
//  It used to be a position in `asShipped.process_variants`, and position is
//  not identity: the stored value 2 named a different development on every
//  stock that had one, every such value was in range, and nothing could tell
//  a stale project file from a correct one. `ProcessVariantCtrl` gives each
//  development one global value; this resolver finds it by matching
//  `film::ProcessVariant::variant_id` against `ProcessVariantCtrlKeyOf`, so
//  the stock's own ordering is free to change and a saved selection is not.
//
//  \warning SINCE 2026-10-11 THE VALUES ARE SIX STEPS (As shipped, Pull -2,
//  Pull -1, Push +1, Push +2, Push +3), each stock offering the steps it has
//  a record for -- its own traced development, or for a colour negative the
//  family model at Push +1 / +2. See AlgoControlEnums.hpp.
//
//  \warning A VALUE THIS STOCK DOES NOT OFFER IS "AS SHIPPED", NOT AN ERROR
//  AND NOT A CLAMP. Selecting Push +3 on an AGFAPAN renders the
//  AGFAPAN as its stored curves represent it -- the same inert path an
//  unselected control takes -- because clamping into range would render some
//  other development and present it as the one that was asked for.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  The colour-negative push family model (2026-10-11, owner decision)
//
//  A colour NEGATIVE without its own record for Push +1 / Push +2 is offered
//  that step through one family model: a gamma scale and a D-min shift on all
//  three channels -- the mean of the traced colour-negative push ladders, an
//  ESTIMATE. film_sim.push_family_model_applies() is the reference.
// ---------------------------------------------------------------------------
inline bool AlgoFindProcessRecord
(
    const film::FilmProfile&     p,
    const char* const            key,
    const film::ProcessVariant** out
) noexcept
{
    *out = nullptr;
    if ((nullptr == key) || ('\0' == key[0]))
        return false;
    for (std::size_t i = 0u; i < p.process_variants.size(); ++i)
    {
        if (p.process_variants[i].variant_id == key)
        {
            *out = &p.process_variants[i];
            return true;
        }
    }
    return false;
}

inline bool AlgoPushFamilyApplies
(
    const film::FilmProfile&  p,
    const ProcessVariantCtrl  variant
) noexcept
{
    if ((ProcessVariantCtrl::ePUSH_1 != variant) &&
        (ProcessVariantCtrl::ePUSH_2 != variant))
        return false;
    if (p.is_monochrome || (film::StockKind::Negative != p.kind))
        return false;
    const film::ProcessVariant* rec = nullptr;
    return !AlgoFindProcessRecord(p, ProcessVariantCtrlKeyOf(variant), &rec);
}

//  True when the host should ENABLE `variant` for stock `p`: As shipped always,
//  otherwise a record of the stock's own or the family model. The list box
//  greys every other step.
inline bool AlgoProcessVariantOffered
(
    const film::FilmProfile&  p,
    const ProcessVariantCtrl  variant
) noexcept
{
    if (ProcessVariantCtrl::eAS_SHIPPED == variant)
        return true;
    if (false == ProcessVariantCtrlValid(variant))
        return false;
    const film::ProcessVariant* rec = nullptr;
    return AlgoFindProcessRecord(p, ProcessVariantCtrlKeyOf(variant), &rec)
        || AlgoPushFamilyApplies(p, variant);
}


inline const film::FilmProfile& AlgoResolveProcessVariant
(
    const film::FilmProfile&  asShipped,
    const ProcessVariantCtrl  variant,
    film::FilmProfile&        store
) noexcept
{
    if (false == ProcessVariantCtrlValid(variant))
        return asShipped;

    const char* const key = ProcessVariantCtrlKeyOf(variant);

    // \warning AN EMPTY KEY SELECTS NOTHING, AND THIS TEST IS WHAT SAYS SO
    // (2026-10-11). Since the six-step reduction, records that are stored but
    // not offered carry an EMPTY variant_id; without this test eAS_SHIPPED's
    // empty key would match the first of them. The Python reference refuses
    // an empty key the same way (`film_sim.resolve_process_variant`).
    if ((nullptr == key) || ('\0' == key[0]))
        return asShipped;

    const film::ProcessVariant* sel = nullptr;
    (void)AlgoFindProcessRecord(asShipped, key, &sel);

    if (nullptr == sel)
    {
        // No record of its own. A colour negative at Push +1 / +2 renders the
        // FAMILY MODEL; anything else is not offered and renders as shipped,
        // by reference and with no copy.
        if (false == AlgoPushFamilyApplies(asShipped, variant))
            return asShipped;

        const bool two = (ProcessVariantCtrl::ePUSH_2 == variant);
        const double gs = two ? film::ePUSH_FAMILY_GAMMA_SCALE_2
                              : film::ePUSH_FAMILY_GAMMA_SCALE_1;
        const double ds = two ? film::ePUSH_FAMILY_DMIN_SHIFT_2
                              : film::ePUSH_FAMILY_DMIN_SHIFT_1;

        store = asShipped;
        film::ToneCurve* const c[3] =
        { &store.curves.r, &store.curves.g, &store.curves.b };
        for (int32_t k = 0; k < 3; k++)
        {
            c[k]->gamma = static_cast<float>(static_cast<double>(c[k]->gamma) * gs);
            c[k]->dmin  = static_cast<float>(static_cast<double>(c[k]->dmin)  + ds);
            // An adjusted curve is a model: the measured table is dropped, as
            // on the gamma_scale path below and in film_sim._retune.
            c[k]->meas_x = nullptr;
            c[k]->meas_d = nullptr;
            c[k]->meas_m = nullptr;
            c[k]->meas_n = 0;
        }
        if (0 != asShipped.exposure_index)
            store.exposure_index = static_cast<int32_t>(
                std::lround(static_cast<double>(asShipped.exposure_index)
                            * (two ? 4.0 : 2.0)));
        return store;
    }

    const film::ProcessVariant& v = *sel;

    // A variant that carries neither curves nor a gamma/dmin change and no
    // exposure index of its own is a label, not a render. Returning `base`
    // keeps the no-copy path for a record that only states a speed.
    const bool movesCurves =
        v.has_curves
        || (v.gamma_scale != 1.0f)
        || (v.dmin_shift  != 0.0f);

    if (!movesCurves && (0 == v.exposure_index))
        return asShipped;

    store = asShipped;

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

        // ⚠⚠ AND THE MEASURED TABLE IS DROPPED (schema v55). A
        // `MeasuredCurve` is 21 densities read at ONE development; scaling
        // the gamma beside it does not scale the table, and because the table
        // WINS inside its own range it would override the very adjustment
        // made above. An adjusted curve is a MODEL, and the measurement it
        // came from no longer describes it, so it falls back to the softplus
        // -- which is exactly what the Python reference's `film_sim._retune`
        // does at the same four sites, guarded by G-V55-MEAS-DROP. Leaving it
        // attached would make this control a no-op on any stock carrying a
        // table, silently.
        // \note A variant that supplies its OWN curves never reaches here,
        //       and must not: its table is its own measurement of its own
        //       development, which is precisely what should be rendered.
        for (int32_t k = 0; k < 3; k++)
        {
            c[k]->meas_x = nullptr;
            c[k]->meas_d = nullptr;
            c[k]->meas_m = nullptr;
            c[k]->meas_n = 0;
        }
    }

    if (0 != v.exposure_index)
        store.exposure_index = v.exposure_index;

    return store;
}
