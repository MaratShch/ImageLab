// ---------------------------------------------------------------------------
//  AlgoBatchPosition.hpp -- render the roll the specification allows, not only
//  the roll the database stores. Schema v52 / control `batchPosition`.
//
//  Copyright (c) Birdaero. All rights reserved.
//
//  WHAT THIS IS FOR
//  ----------------
//  Ten stocks in this database are specified by a MANUFACTURING ACCEPTANCE
//  BAND rather than by typical data -- the nine Soviet ТУ films and the one
//  ГОСТ film. Their `film::ToleranceSpec` records carry, per layer, the
//  lowest and highest contrast a roll could legally have and still ship.
//
//  Until this control existed the band was carried, validated, reported and
//  read by nothing. The profile stored ONE number per layer, and which number
//  it was depended on how the source printed the norm: the mid-point of a
//  two-sided band, or the WORST LEGAL EXAMPLE where the norm was one-sided.
//  A user could not render the other end of the band, and could not see how
//  wide the band was.
//
//  \warning THIS IS NOT A CONTRAST TWEAK AND MUST NOT BE USED AS ONE. It moves
//  the curve only as far as the film's own published specification allows, per
//  layer, by different amounts on each layer, and it does nothing at all on the
//  184 stocks whose manufacturer published typical values instead of limits.
//  A generic contrast control would move every stock by the same amount; this
//  one is a statement about what a real roll of THAT film could have been.
//
//  THE INTERPOLATION IS ANCHORED ON THE STORED VALUE, NOT ON THE BAND CENTRE
//  ------------------------------------------------------------------------
//  position  0  ->  exactly the stored curve, returned BY REFERENCE with no
//                   copy, so a default render is bit-identical to a pre-field
//                   one. This is the property G-BATCH-IDENTITY asserts.
//  position +1  ->  the upper acceptance edge of every banded quantity
//  position -1  ->  the lower acceptance edge
//
//  \warning ANCHORING ON THE STORED VALUE IS THE WHOLE DESIGN, and the obvious
//  alternative is wrong. Interpolating between `gamma_lo` and `gamma_hi` with
//  0 at the band CENTRE would move the curve at position 0 on every stock whose
//  stored gamma is not the centre -- which is all of them where the source
//  printed a one-sided limit, because there the stored value IS an edge. The
//  default render would change, silently, on ten stocks. Anchoring instead
//  makes 0 mean "as the database holds it" on every stock, and the two ends
//  mean "as far as this film's own specification permits from there".
//
//  A CONSEQUENCE WORTH STATING: the two halves are NOT symmetric. On ДС-5М the
//  stored blue gamma is 0.60 in a band of 0.56-0.66, so +1 moves it by 0.06 and
//  -1 by 0.04 -- because the ТУ prints «+0,06 / -0,04», an asymmetric deviation,
//  and this control reproduces the document rather than tidying it.
//
//  WHAT IT MOVES, AND WHAT IT DELIBERATELY DOES NOT
//  -------------------------------------------------
//  MOVES: the per-layer characteristic-curve gamma, and the per-layer D_min
//  where the source prints a two-sided band for it. Both are fields the tone
//  curve already holds, so no pixel stage is added and no stage needs to know
//  this control exists -- exactly the shape of AlgoResolveProcessVariant.
//
//  DOES NOT MOVE, and each omission is a refusal rather than a gap:
//    * SPEED. The bands state an acceptance window for the ISO/ГОСТ speed, but
//      no stage reads `exposure_index` as an exposure, so moving it would
//      change a reported number and not a pixel. A user who wants the speed
//      end of the band should use `exposureStops`, which is honest about being
//      an exposure change.
//    * GRANULARITY and MTF. Their bands are one-sided CEILINGS («не более»),
//      so there is no lower edge to interpolate toward and the stored value is
//      already the ceiling. Moving toward a lower value would invent a
//      cleanliness the document does not grant.
//    * D_max on the reversal stocks. Its band is a one-sided FLOOR and the
//      stored value already sits at it, for the same reason in the opposite
//      direction.
//    * The ageing envelope. `guarantee_months` and the drift bounds describe a
//      roll that has been in a cupboard, which is `AgingSpec`'s subject and a
//      different control's job.
// ---------------------------------------------------------------------------
#pragma once

#include "film_profiles.hpp"

#include <cstddef>


// ---------------------------------------------------------------------------
//  AlgoBatchPositionValid -- the OFF test, so callers do not repeat the
//  epsilon. Anything inside +/- 1e-9 of zero is off.
// ---------------------------------------------------------------------------
inline bool AlgoBatchPositionValid(const double position) noexcept
{
    return (position > 1.0e-9) || (position < -1.0e-9);
}


// ---------------------------------------------------------------------------
//  AlgoResolveBatchPosition
//
//  asShipped  the stock as the database holds it
//  position   -1 .. +1; 0 is "as shipped". Values outside the range are
//             CLAMPED, unlike processVariant's out-of-range handling, and the
//             difference is deliberate: an out-of-range enumerator names a
//             development that might exist, so rendering a neighbour would be
//             a lie, whereas an out-of-range position on a continuous axis has
//             an unambiguous nearest legal meaning -- the edge of the band.
//  store      scratch the caller owns for the frame; written ONLY when the
//             control actually moves something
//
//  Returns `asShipped` by reference when the control is off, when the stock
//  carries no acceptance band, or when its band moves nothing; returns `store`
//  otherwise. The caller must keep `store` alive for as long as it uses the
//  result -- it is a frame local in AlgorithmMain for that reason.
// ---------------------------------------------------------------------------
inline const film::FilmProfile& AlgoResolveBatchPosition
(
    const film::FilmProfile&  asShipped,
    const double              position,
    film::FilmProfile&        store
) noexcept
{
    if (false == AlgoBatchPositionValid(position))
        return asShipped;

    // The record the stock's own stored scalars represent. A stock may carry
    // more than one quality grade; only the default one describes the numbers
    // actually in the profile, so only it may move them.
    const film::ToleranceSpec* band = nullptr;
    for (std::size_t i = 0u; i < asShipped.tolerance.size(); ++i)
    {
        if (asShipped.tolerance[i].is_default)
        {
            band = &asShipped.tolerance[i];
            break;
        }
    }

    // 184 of 194 stocks. Their manufacturers published typical values and
    // disclaimed being a specification, so there is no band to move along and
    // this control is correctly inert on them.
    if (nullptr == band)
        return asShipped;

    const double p    = (position > 1.0) ? 1.0
                      : (position < -1.0) ? -1.0 : position;
    const float  lo[3] = { band->gamma_lo_r, band->gamma_lo_g, band->gamma_lo_b };
    const float  hi[3] = { band->gamma_hi_r, band->gamma_hi_g, band->gamma_hi_b };
    const float  dlo[3] = { band->dmin_min_r, band->dmin_min_g, band->dmin_min_b };
    const float  dhi[3] = { band->dmin_max_r, band->dmin_max_g, band->dmin_max_b };

    bool moves = false;
    for (int c = 0; c < 3; ++c)
    {
        if ((lo[c] > 0.0f) && (hi[c] > 0.0f) && (hi[c] > lo[c]))
            moves = true;
        if ((dlo[c] > 0.0f) && (dhi[c] > 0.0f) && (dhi[c] > dlo[c]))
            moves = true;
    }
    if (false == moves)
        return asShipped;

    store = asShipped;

    film::ToneCurve* const cv[3] =
        { &store.curves.r, &store.curves.g, &store.curves.b };

    for (int c = 0; c < 3; ++c)
    {
        // -- contrast ---------------------------------------------------------
        if ((lo[c] > 0.0f) && (hi[c] > 0.0f) && (hi[c] > lo[c]))
        {
            const double g0   = static_cast<double>(cv[c]->gamma);
            const double edge = (p > 0.0) ? static_cast<double>(hi[c])
                                          : static_cast<double>(lo[c]);
            // \warning A STORED VALUE OUTSIDE ITS OWN BAND WOULD INVERT THIS,
            // and the database no longer contains one: G-TOL-GAMMA-LEGAL
            // asserts every stored gamma lies inside its band, and it found a
            // real violation the day it was written (SVEMA_CNL_65 stored
            // 0.70/0.70/0.85 against a 0.55/0.60/0.65 +/- 0.08 standard). If
            // that guard is ever removed this arithmetic starts moving curves
            // the wrong way, silently.
            const double moved = g0 + ((p < 0.0) ? -p : p) * (edge - g0);
            cv[c]->gamma = static_cast<float>(moved);
        }

        // -- base plus fog ----------------------------------------------------
        // Only where the source prints BOTH edges. A one-sided D_min ceiling
        // leaves the stored value at the ceiling already.
        if ((dlo[c] > 0.0f) && (dhi[c] > 0.0f) && (dhi[c] > dlo[c]))
        {
            const double d0   = static_cast<double>(cv[c]->dmin);
            const double edge = (p > 0.0) ? static_cast<double>(dhi[c])
                                          : static_cast<double>(dlo[c]);
            const double moved = d0 + ((p < 0.0) ? -p : p) * (edge - d0);
            cv[c]->dmin = static_cast<float>(moved);
        }
    }

    return store;
}
