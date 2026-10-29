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
//  dmin_shift, an estimate; the stock's own PushSpec values replace it where
//  stated, v69), or, for a colour REVERSAL stock on Process E-6 / VNF-1, the
//  reversal push model (v69, first-developer time). Records that are not a
//  step stay in the database with an empty id and are never selected.
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

#include <cmath>     // std::lround, std::pow, std::fabs, std::fmax, std::fmin
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

//  Signed stops of a step: Pull -2 .. Push +3; 0 for As shipped / invalid.
//  film_profiles.PROCESS_STEP_STOPS is the same table.
inline int32_t AlgoProcessStepStops (const ProcessVariantCtrl variant) noexcept
{
    switch (variant)
    {
        case ProcessVariantCtrl::ePULL_2: return -2;
        case ProcessVariantCtrl::ePULL_1: return -1;
        case ProcessVariantCtrl::ePUSH_1: return  1;
        case ProcessVariantCtrl::ePUSH_2: return  2;
        case ProcessVariantCtrl::ePUSH_3: return  3;
        default:                          return  0;
    }
}

//  The stock's own stated PushSpec limits (schema v69): a step of n stops is
//  offered only where it lies inside max_push_stops / max_pull_stops (0 = not
//  stated, no stock-level limit). film_profiles.push_step_within_limits.
inline bool AlgoPushStepWithinLimits
(
    const film::FilmProfile&  p,
    const ProcessVariantCtrl  variant
) noexcept
{
    const int32_t n = AlgoProcessStepStops(variant);
    const double  mp = static_cast<double>(p.push.max_push_stops);
    const double  ml = static_cast<double>(p.push.max_pull_stops);
    if ((n > 0) && (mp > 0.0) && (static_cast<double>(n) > mp + 1e-9))
        return false;
    if ((n < 0) && (ml > 0.0) && (static_cast<double>(-n) > ml + 1e-9))
        return false;
    return true;
}

//  A monochrome stock processed in Process C-41 (a dye image).
//  film_profiles.chromogenic_c41.
inline bool AlgoChromogenicC41 (const film::FilmProfile& p) noexcept
{
    return p.processing.developer == "Process C-41";
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
    if (film::StockKind::Negative != p.kind)
        return false;
    // v69: a chromogenic (C-41) monochrome negative joins the family where its
    // maker states a push rating; silver monochrome stays out.
    if (p.is_monochrome &&
        !(AlgoChromogenicC41(p) && (p.push.max_push_stops > 0.0f)))
        return false;
    // v69: the stock's own stated push limit bounds the family model too.
    if (false == AlgoPushStepWithinLimits(p, variant))
        return false;
    const film::ProcessVariant* rec = nullptr;
    return !AlgoFindProcessRecord(p, ProcessVariantCtrlKeyOf(variant), &rec);
}


// ---------------------------------------------------------------------------
//  The colour REVERSAL push model (schema v69, 2026-10-27, owner decision)
//
//  A push / pull of a colour reversal stock is its FIRST developer run for
//  t = t0 (1 + x): x from the process table (J-83 for E-6, Table 11-6 for
//  VNF-1), t0 = ProcessingSpec::minutes. Per record: gamma x (1 + g n) for a
//  push; Dmax - r_c x^q for a push (the shoulder moves so the curve ends
//  there); D-min + fog n where stated; all six interimage terms x (1 + x)^alpha
//  (EP 0 426 181, alpha by receiver grain habit; a push limited by
//  eREV_PUSH_IIE_CAP); exposure index x 2^(n s). film_sim._reversal_push_resolve
//  is the reference.
// ---------------------------------------------------------------------------

//  The process table's first-developer time change for `variant`, minutes;
//  false when the stock is not a colour reversal on a tabled process, or the
//  table has no such step.
inline bool AlgoReversalPushDelta
(
    const film::FilmProfile&  p,
    const ProcessVariantCtrl  variant,
    double&                   deltaMin
) noexcept
{
    deltaMin = 0.0;
    if (p.is_monochrome || (film::StockKind::Reversal != p.kind))
        return false;
    if (!(p.processing.minutes > 0.0f))
        return false;
    const double* row = nullptr;
    if (p.processing.developer == film::eREV_PUSH_E6_PROCESS)
        row = film::eREV_PUSH_E6_DELTA_MIN;
    else if (p.processing.developer == film::eREV_PUSH_VNF1_PROCESS)
        row = film::eREV_PUSH_VNF1_DELTA_MIN;
    else
        return false;
    int32_t idx = -1;
    switch (variant)
    {
        case ProcessVariantCtrl::ePULL_2: idx = 0; break;
        case ProcessVariantCtrl::ePULL_1: idx = 1; break;
        case ProcessVariantCtrl::ePUSH_1: idx = 2; break;
        case ProcessVariantCtrl::ePUSH_2: idx = 3; break;
        case ProcessVariantCtrl::ePUSH_3: idx = 4; break;
        default:                          return false;
    }
    if (0.0 == row[idx])
        return false;
    deltaMin = row[idx];
    return true;
}

//  film_profiles.reversal_push_model_applies.
inline bool AlgoReversalPushApplies
(
    const film::FilmProfile&  p,
    const ProcessVariantCtrl  variant
) noexcept
{
    double d = 0.0;
    if (false == AlgoReversalPushDelta(p, variant, d))
        return false;
    const film::ProcessVariant* rec = nullptr;
    if (AlgoFindProcessRecord(p, ProcessVariantCtrlKeyOf(variant), &rec))
        return false;
    return AlgoPushStepWithinLimits(p, variant);
}

//  x = (t - t0) / t0; 0 when the model does not apply.
//  film_profiles.reversal_push_extension.
inline double AlgoReversalPushExtension
(
    const film::FilmProfile&  p,
    const ProcessVariantCtrl  variant
) noexcept
{
    double d = 0.0;
    if (false == AlgoReversalPushDelta(p, variant, d))
        return 0.0;
    return d / static_cast<double>(p.processing.minutes);
}

//  film_profiles.reversal_push_alpha.
inline double AlgoReversalPushAlpha (const film::FilmProfile& p) noexcept
{
    if (p.emulsion.habit == "tabular")
        return film::eREV_PUSH_ALPHA_TABULAR;
    if ((p.emulsion.habit == "cubic") || (p.emulsion.habit == "octahedral") ||
        (p.emulsion.habit == "nontabular"))
        return film::eREV_PUSH_ALPHA_CONVENTIONAL;
    return film::eREV_PUSH_ALPHA_UNKNOWN;
}

inline double AlgoIieMaxTerm (const film::InterimageSpec& i) noexcept
{
    double m = std::fabs(i.a_rg);
    m = std::fmax(m, std::fabs(i.a_rb));
    m = std::fmax(m, std::fabs(i.a_gr));
    m = std::fmax(m, std::fabs(i.a_gb));
    m = std::fmax(m, std::fabs(i.a_br));
    m = std::fmax(m, std::fabs(i.a_bg));
    return m;
}

inline bool AlgoIieActive (const film::InterimageSpec& i) noexcept
{
    return (i.iterations > 0) &&
           ((0.0 != i.a_rg) || (0.0 != i.a_rb) || (0.0 != i.a_gr) ||
            (0.0 != i.a_gb) || (0.0 != i.a_br) || (0.0 != i.a_bg));
}

//  The factor on all six interimage terms. film_profiles.reversal_push_iie_scale.
inline double AlgoReversalPushIieScale
(
    const film::FilmProfile&  p,
    const ProcessVariantCtrl  variant
) noexcept
{
    const double x = AlgoReversalPushExtension(p, variant);
    if ((0.0 == x) || (false == AlgoIieActive(p.interimage)))
        return 1.0;
    double k = std::pow(1.0 + x, AlgoReversalPushAlpha(p));
    if (k > 1.0)
    {
        const double m = AlgoIieMaxTerm(p.interimage);
        if (m > 0.0)
            k = std::fmin(k, std::fmax(film::eREV_PUSH_IIE_CAP / m, 1.0));
    }
    return k;
}

inline const film::FilmProfile& AlgoReversalPushResolve
(
    const film::FilmProfile&  asShipped,
    const ProcessVariantCtrl  variant,
    film::FilmProfile&        store
) noexcept
{
    const int32_t n = AlgoProcessStepStops(variant);
    const double  x = AlgoReversalPushExtension(asShipped, variant);
    const film::PushSpec& ps = asShipped.push;
    const double loss[3] = { static_cast<double>(ps.dmax_loss_r),
                             static_cast<double>(ps.dmax_loss_g),
                             static_cast<double>(ps.dmax_loss_b) };
    const double q  = static_cast<double>(ps.dmax_loss_exponent);
    const double gg = static_cast<double>(ps.gamma_gain_per_stop);
    const double fg = static_cast<double>(ps.base_fog_penalty_per_stop);

    store = asShipped;
    film::ToneCurve* const c[3] =
    { &store.curves.r, &store.curves.g, &store.curves.b };
    for (int32_t k = 0; k < 3; k++)
    {
        const double g0  = static_cast<double>(c[k]->gamma);
        const double d0  = static_cast<double>(c[k]->dmin);
        const double toe = static_cast<double>(c[k]->toe_x);
        const double sh  = static_cast<double>(c[k]->shoulder_x);
        double g = g0;
        if ((n > 0) && (gg > 0.0))
            g = g0 * (1.0 + gg * static_cast<double>(n));
        double dmin = d0;
        if ((n > 0) && ps.fog_penalty_stated)
            dmin = d0 + fg * static_cast<double>(n);
        const bool lose = (x > 0.0) && (loss[k] > 0.0) && (q > 0.0);
        if ((g == g0) && (dmin == d0) && !lose)
            continue;   // a record nothing moves keeps its curve and table
        double dmaxNew = d0 + g0 * (sh - toe);
        if (lose)
            dmaxNew -= loss[k] * std::pow(x, q);
        const double span = std::fmax((dmaxNew - dmin) / g, 0.05);
        c[k]->gamma      = static_cast<float>(g);
        c[k]->dmin       = static_cast<float>(dmin);
        c[k]->shoulder_x = static_cast<float>(toe + span);
        // An adjusted curve is a model: the measured table is dropped.
        c[k]->meas_x = nullptr;
        c[k]->meas_d = nullptr;
        c[k]->meas_m = nullptr;
        c[k]->meas_n = 0;
    }
    if (0 != asShipped.exposure_index)
    {
        const double s = (ps.speed_gain_per_stop > 0.0f)
                       ? static_cast<double>(ps.speed_gain_per_stop) : 1.0;
        store.exposure_index = static_cast<int32_t>(std::lround(
            static_cast<double>(asShipped.exposure_index)
            * std::pow(2.0, static_cast<double>(n) * s)));
    }
    const double kI = AlgoReversalPushIieScale(asShipped, variant);
    if (1.0 != kI)
    {
        store.interimage.a_rg *= kI;
        store.interimage.a_rb *= kI;
        store.interimage.a_gr *= kI;
        store.interimage.a_gb *= kI;
        store.interimage.a_br *= kI;
        store.interimage.a_bg *= kI;
    }
    return store;
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
        || AlgoPushFamilyApplies(p, variant)
        || AlgoReversalPushApplies(p, variant);
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
        if (AlgoReversalPushApplies(asShipped, variant))
            return AlgoReversalPushResolve(asShipped, variant, store);
        if (false == AlgoPushFamilyApplies(asShipped, variant))
            return asShipped;

        const bool two = (ProcessVariantCtrl::ePUSH_2 == variant);
        const double nStops = two ? 2.0 : 1.0;
        double gs = two ? film::ePUSH_FAMILY_GAMMA_SCALE_2
                        : film::ePUSH_FAMILY_GAMMA_SCALE_1;
        double ds = two ? film::ePUSH_FAMILY_DMIN_SHIFT_2
                        : film::ePUSH_FAMILY_DMIN_SHIFT_1;
        // v69: the stock's OWN stated push behaviour outranks the family
        // estimate (PushSpec gamma gain, stated fog penalty, speed gain).
        const film::PushSpec& ps = asShipped.push;
        if (ps.gamma_gain_per_stop > 0.0f)
            gs = 1.0 + static_cast<double>(ps.gamma_gain_per_stop) * nStops;
        if (ps.fog_penalty_stated)
            ds = static_cast<double>(ps.base_fog_penalty_per_stop) * nStops;
        const double sp = (ps.speed_gain_per_stop > 0.0f)
                        ? static_cast<double>(ps.speed_gain_per_stop) : 1.0;

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
                            * std::pow(2.0, nStops * sp)));
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


// ---------------------------------------------------------------------------
//  AlgoPushRange / AlgoPushWithinRange  (#209 item 6, 2026-10-16)
//
//  The maker's stated push / pull latitude (film::PushSpec::max_push_stops /
//  max_pull_stops; 38 / 31 stocks), for the HOST: how far a panel may offer to
//  push or pull this film, and where to warn. 0 = the sheet is silent, which
//  is not a refusal. Since schema v69 the same limits also bound the steps the
//  resolver offers (AlgoPushStepWithinLimits).
//  film_sim.push_range / push_within_range are the same functions.
// ---------------------------------------------------------------------------
inline void AlgoPushRange
(
    const film::FilmProfile& p,
    double&                  maxPushStops,
    double&                  maxPullStops
) noexcept
{
    maxPushStops = (p.push.max_push_stops > 0.0f) ? static_cast<double>(p.push.max_push_stops) : 0.0;
    maxPullStops = (p.push.max_pull_stops > 0.0f) ? static_cast<double>(p.push.max_pull_stops) : 0.0;
}

inline bool AlgoPushWithinRange
(
    const film::FilmProfile& p,
    const double             stops
) noexcept
{
    double mp;
    double ml;
    AlgoPushRange(p, mp, ml);
    if (stops > 0.0)
        return (mp <= 0.0) || (stops <= mp + 1e-9);
    if (stops < 0.0)
        return (ml <= 0.0) || (-stops <= ml + 1e-9);
    return true;
}
