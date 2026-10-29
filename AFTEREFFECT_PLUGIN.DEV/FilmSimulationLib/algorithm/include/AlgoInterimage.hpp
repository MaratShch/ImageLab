#pragma once

// ---------------------------------------------------------------------------
//  AlgoInterimage.hpp
//
//  Sub-stage 8b of the film simulation pipeline: interimage effects, the
//  cross-layer half of the DIR coupler chemistry.
//
//  PHYSICAL BACKGROUND
//
//  A colour developer does not develop the three layers in isolation. As silver
//  is developed in one layer the coupler releases a development inhibitor, and
//  that inhibitor diffuses VERTICALLY into the neighbouring layers and suppresses
//  development there. So each layer's effective exposure depends on what the
//  other two are doing:
//
//      logE_i' = logE_i + sum over j != i of a_ij * (D_j - D_ref_j)
//
//  Stage 9 is the LATERAL half of the same chemistry - inhibitor diffusing
//  sideways within a layer. Same molecules, two different directions, two
//  different visual consequences, so they are separate stages.
//
//  WHY THE MID-GREY REFERENCE MATTERS
//
//  Subtracting the density a neutral mid grey reaches is what makes this a
//  COLOUR effect rather than a tone effect. On a neutral every (D_j - D_ref_j) is
//  about zero, the correction vanishes, and the grey scale passes through
//  untouched. On a saturated colour, where the three layers disagree strongly,
//  each develops against unequal inhibition and they separate further.
//
//  That is saturation rising WITHOUT gamma rising, which no per-channel tone
//  curve can produce however it is shaped. It is the single mechanism that
//  distinguishes a modern coupler-rich emulsion from a 1950s one, and it is the
//  reason a well-fitted set of curves alone still renders a modern stock flat.
//
//  IT IS AN IMPLICIT EQUATION
//
//  Density depends on the corrected log exposure, which depends on density. The
//  system is solved by fixed-point iteration seeded with the densities stage 8
//  just computed. Each pass costs a full curve evaluation per channel, which
//  makes this among the most expensive stages in the chain, so the iteration
//  count is a profile field the renderer honours rather than a hardcoded loop:
//  a stock whose coefficients are small converges in one pass and should not pay
//  for four.
//
//  DENSITY WEIGHTING, AND WHY IT EXISTS
//
//  Zero means uniform coupling across the whole curve, which is right for a
//  chromogenic negative: the inhibitor is released in proportion to development
//  everywhere.
//
//  Greater than zero concentrates the coupling where the NEIGHBOURING layer is
//  dense, which is right for a reversal stock: its interimage effects come from
//  iodide released in the first black-and-white developer, and that lands in the
//  areas that end up with high dye density.
//
//  The weighting is normalised at the mid-grey reference, so a neutral is
//  untouched under either mechanism. That property is the entire point of the
//  stage and must survive the mechanism split - a weighting scheme that tinted
//  neutrals would be worse than no weighting at all.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// The characteristic curve this stage re-evaluates once per iteration.
#include "AlgoCharacteristicCurve.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including InterimageSpec.
#include "film_profiles.hpp"

// AlgoCorrectionLayerSet, the colour-correction layers (2026-10-25).
#include "AlgoSpectralSensitivity.hpp"

#include <cstdint>   // int32_t
#include <cmath>     // std::pow (law C tables)

// AlgoCurveLut, AlgoCurveLutEval[V]: the shared table and lookup (law C, v68).
#include "AlgoCurveLut.hpp"


// ---------------------------------------------------------------------------
//  Floor on the mid-grey reference density used to normalise the weighting.
//
//  A stock whose base fog is essentially zero would otherwise divide by zero
//  here. The floor is far below any real base plus fog, so it changes nothing on
//  a real profile and only removes the singularity.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_IIE_REF_FLOOR = static_cast<AlgoType>(1.0e-4);

// 2026-10-21 (queue P96s): floor on the fourth layer's mixed exposure (relative
// to mid grey) inside log10 -- a guard against log10(0) only. Twin:
// film_sim.FOURTH_LAYER_E_FLOOR.
constexpr AlgoType ALGO_FOURTH_LAYER_E_FLOOR = static_cast<AlgoType>(1.0e-12);
constexpr AlgoType ALGO_FOURTH_LAYER_TEN     = static_cast<AlgoType>(10.0);


// ---------------------------------------------------------------------------
//  2026-10-26c (schema v68): THE REVERSAL LAW C TABLES. Twin of
//  film_sim.iie_rev_tables -- the same domain, size and values, read through
//  the same AlgoCurveLutEval[V] interpolation, so the engines agree by
//  construction.
//
//      u(D) = max(Dmax - D, 0) / (Dmax - Dmin),  s(D) = max(D - Dmin, 0) / (Dmax - Dmin)
//      V(D) = -(u0 (Dmax - Dmin) / pd) (u/u0)^pd   donor release   (vLut[0..2])
//      g(D) = (u/u0)^pr (s/s0)^ph                  receiver gate   (gLut[0..2])
//      s0 = 0.25, u0 = 0.75 -- fixed points of the stock's own scale
//
//  V non-decreasing in D: a donor that developed less never counts as more
//  inhibiting; g >= 0 -- so the Hanson-Horton shift stays >= 0. pd =
//  donor_convexity (0 = unset = 1), pr = receiver_gate, ph = receiver_gate_hl.
// ---------------------------------------------------------------------------
constexpr double ALGO_IIE_REV_LUT_LO = -0.5;
constexpr double ALGO_IIE_REV_LUT_HI =  6.0;

// v68 law C tables (twin: film_sim.iie_rev_tables). Normalised on the stock's
// own scale N = Dmax - Dmin at the fixed point s0 = 0.25 (u0 = 0.75):
//   u = max(Dmax - D, 0)/N, s = max(D - Dmin, 0)/N
//   V(D) = -(u0 N/pd) (u/u0)^pd          donor release, non-decreasing in D
//   g(D) = (u/u0)^pr (s/s0)^ph           receiver gate, 1 at s0
// ⚠ ANCHOR-INDEPENDENT BY DESIGN: dRef is not read (a reference-normalised
// draft made the chemistry depend on the render's grey target).
constexpr double ALGO_IIE_REV_S0 = 0.25;

inline void AlgoBuildIieRevLuts
(
    const film::InterimageSpec& iie,
    const film::RGBCurves&      curves,
    const AlgoType              dRef[3],
    AlgoCurveLut*               vLut,     // [3]
    AlgoCurveLut*               gLut      // [3]
) noexcept
{
    (void)dRef;
    const double pd = (iie.donor_convexity > 0.0) ? iie.donor_convexity : 1.0;
    const double pr = iie.receiver_gate;
    const double ph = iie.receiver_gate_hl;
    const double s0 = ALGO_IIE_REV_S0;
    const double u0 = 1.0 - s0;
    const double step = (ALGO_IIE_REV_LUT_HI - ALGO_IIE_REV_LUT_LO)
                      / static_cast<double>(ALGO_CURVE_LUT_SIZE - 1);
    const film::ToneCurve* cv[3] = { &curves.r, &curves.g, &curves.b };

    for (int32_t c = 0; c < 3; c++)
    {
        const double dmax = static_cast<double>(cv[c]->dmax());
        const double dmin = static_cast<double>(cv[c]->dmin);
        const double rng  = MAX_VALUE(dmax - dmin, 1.0e-3);

        for (int32_t i = 0; i <= ALGO_CURVE_LUT_SIZE; i++)
        {
            const double x  = ALGO_IIE_REV_LUT_LO + step * static_cast<double>(i);
            const double ur = MAX_VALUE(dmax - x, 0.0) / rng / u0;
            vLut[c].d[i] = static_cast<AlgoType>(-(u0 * rng / pd) * std::pow(ur, pd));
            double g = 1.0;
            if (pr != 0.0)
                g *= std::pow(ur, pr);
            if (ph != 0.0)
                g *= std::pow(MAX_VALUE(x - dmin, 0.0) / rng / s0, ph);
            gLut[c].d[i] = static_cast<AlgoType>(g);
        }
        vLut[c].lo      = static_cast<AlgoType>(ALGO_IIE_REV_LUT_LO);
        vLut[c].invStep = static_cast<AlgoType>(1.0 / step);
        vLut[c].maxIdx  = static_cast<AlgoType>(ALGO_CURVE_LUT_SIZE - 1);
        gLut[c].lo      = vLut[c].lo;
        gLut[c].invStep = vLut[c].invStep;
        gLut[c].maxIdx  = vLut[c].maxIdx;
    }
    return;
}


// ---------------------------------------------------------------------------
//  Sub-stage 8b: interimage effects.
//
//  pSrcR/G/B     density from stage 8
//  pDstR/G/B     corrected density out
//  pLogER/G/B    log exposure retained by stage 8; READ ONLY, and required -
//                the correction re-enters the curve in the exposure domain, and
//                density cannot be inverted back through the shoulder
//  pScrDR/DG/DB  scratch: the three (D_j - D_ref_j) difference planes
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  anchor        anchors from AlgoSolveAnchors; the reversal trim enters the
//                curve argument exactly as it did at stage 8
//  clRow         2026-10-21 (queue P96s): the fourth (CL) layer's mixing row
//                (AlgoSpectralFourthLayerRow), read only when clStrength > 0
//  clStrength    the stock's cl_red_inhibition when the stock carries the
//                layer (always on since 2026-10-22e); 0 = no CL term
//  clOffset      2026-10-22: the CL layer's speed relative to green, log10
//                (AlgoSpectralFourthLayerRow offsetOut); 0 = not traced
//  corr          2026-10-25 (schema v66): the colour-correction layers
//                (AlgoSpectralCorrectionLayers), or nullptr / count 0 for none.
//                Negative or reversal; each layer's fixed term joins its
//                target record's correction after the CL term.
//
//  The three scratch planes must be distinct from each other, from the source,
//  from the destination and from the log-exposure planes.
// ---------------------------------------------------------------------------
void AlgoStage08b_Interimage
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const AlgoType* RESTRICT pLogER,
    const AlgoType* RESTRICT pLogEG,
    const AlgoType* RESTRICT pLogEB,
    AlgoType* RESTRICT       pScrDR,
    AlgoType* RESTRICT       pScrDG,
    AlgoType* RESTRICT       pScrDB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const HighPrecType       anchor[3],
    const AlgoType           clRow[3],
    const AlgoType           clStrength,
    const AlgoType           clOffset,
    const AlgoCorrectionLayerSet* corr
) noexcept;
