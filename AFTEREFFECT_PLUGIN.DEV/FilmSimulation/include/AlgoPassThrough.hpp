// ---------------------------------------------------------------------------
//  AlgoPassThrough.hpp
//
//  WHEN A STAGE WOULD ONLY COPY, DO NOT CALL IT (2026-10-04).
//
//  Every stage copies its input to its own destination triple when it has
//  nothing to do, because under the retained-buffer layout a stage's buffer
//  must hold a valid image for whoever reads it. With the production layout
//  (ALGO_RETAIN_ALL_STAGES = 0, two alternating triples) that copy is pure
//  memory traffic: 3 planes read and 3 written, about 6 ms of a 1920 x 1080
//  frame on the review machine, and a typical colour negative at default
//  controls passes through ten such stages.
//
//  Algorithm_Main therefore keeps a "current image" triple and, when one of
//  the predicates below says a stage would only copy, leaves the current
//  triple where it is and moves on. The stage functions are UNCHANGED and
//  still copy when called in that state, so the predicates are not a second
//  implementation of the stage logic: each is the stage's own early-out
//  condition, restated on the same inputs. A predicate that returns false
//  simply means "call the stage"; the stage may still find nothing to do and
//  copy, which is correct and merely slower.
//
//  ⚠ A PREDICATE MUST NEVER RETURN TRUE WHEN THE STAGE WOULD CHANGE A PIXEL.
//  Each one below cites the stage line it mirrors. When a stage's early-out
//  condition changes, change its predicate in the same commit; chain_parity
//  (222 stocks, Python = Scalar = AVX2) and the bench bit-identity run are the
//  gates that catch a divergence.
//
//  Shared by the Scalar and the AVX2 build: the conditions depend on the
//  profile, the controls and the frame geometry, never on pixel data. Under
//  ALGO_RETAIN_ALL_STAGES = 1 every predicate returns false, so the debug
//  layout keeps one valid image per stage buffer exactly as before.
// ---------------------------------------------------------------------------

#ifndef ALGO_PASS_THROUGH_HPP
#define ALGO_PASS_THROUGH_HPP

#include "Common.hpp"
#include "AlgoTypes.hpp"
#include "AlgoControl.hpp"
#include "AlgoMemHandler.hpp"           // ALGO_RETAIN_ALL_STAGES
#include "AlgoTakingFilters.hpp"        // ALGO_TAKING_IDENTITY_EPS
#include "AlgoCornerDefocus.hpp"        // ALGO_DEFOCUS_MAX_LOSS
#include "AlgoDirCoupler.hpp"           // ALGO_COUPLER_MIN_SIGMA_PX
#include "AlgoScanMtf.hpp"              // AlgoScanSigmaMm, ALGO_SCAN_MIN_SIGMA_PX
#include "AlgoDyeImpurity.hpp"          // AlgoIsIdentityMatrix
#include "AlgoCharacteristicCurve.hpp"  // AlgoTintFactor
#include "AlgoGateWeave.hpp"            // ALGO_WEAVE_UM_PER_MM
#include "film_profiles.hpp"

#include <cmath>
#include <cstdint>

#if ALGO_RETAIN_ALL_STAGES
  #define ALGO_FORWARD_PASS_THROUGH 0
#else
  #define ALGO_FORWARD_PASS_THROUGH 1
#endif

// One image: three planes. Plain pointers, no RESTRICT: these alias by
// design (the current image IS one of the stage triples).
struct AlgoPlanes
{
    AlgoType* r;
    AlgoType* g;
    AlgoType* b;
};

// The destination for a stage: its own triple, unless that triple currently
// holds the input image (possible under the two-triple layout after a
// forwarded stage), in which case the other triple of the pair. `alt` is the
// previous stage's triple, which has the opposite parity. Under the retained
// layout `own` never aliases `cur`, so `own` is always returned.
FORCE_INLINE AlgoPlanes AlgoStageDst (const AlgoPlanes own,
                                      const AlgoPlanes alt,
                                      const AlgoPlanes cur) noexcept
{
    return (own.r == cur.r) ? alt : own;
}

// ---- stage 02b: Algo_02_Sim.cpp isIdentityMatrix on the taking matrix ------
FORCE_INLINE bool AlgoPass02b (const film::FilmProfile& profile) noexcept
{
    const film::Matrix3& m = profile.taking_matrix;
    for (int32_t i = 0; i < 3; i++)
        for (int32_t j = 0; j < 3; j++)
        {
            const AlgoType expected = (i == j) ? ALGO_ONE : ALGO_ZERO;
            const AlgoType actual   = static_cast<AlgoType>(m[i][j]);
            if (std::fabs(actual - expected) > ALGO_TAKING_IDENTITY_EPS)
                return false;
        }
    return ALGO_FORWARD_PASS_THROUGH != 0;
}

// ---- stage 03: Algo_03_Sim.cpp "applies" ----------------------------------
FORCE_INLINE bool AlgoPass03 (const film::FilmProfile& profile,
                              const AlgoControls&      params) noexcept
{
    const bool applies = (params.wbStrength > 0.0) && (false == profile.is_monochrome);
    return (ALGO_FORWARD_PASS_THROUGH != 0) && (false == applies);
}

// ---- stage 03b: Algo_03_Sim.cpp flare <= 0 --------------------------------
FORCE_INLINE bool AlgoPass03b (const film::FilmProfile& profile,
                               const AlgoControls&      params) noexcept
{
    const AlgoType flare = (params.flare < 0.0)
                         ? static_cast<AlgoType>(profile.default_flare)
                         : static_cast<AlgoType>(params.flare);
    return (ALGO_FORWARD_PASS_THROUGH != 0) && (flare <= ALGO_ZERO);
}

// ---- stage 03c: a stub that always copies ---------------------------------
FORCE_INLINE bool AlgoPass03c (void) noexcept
{
    return ALGO_FORWARD_PASS_THROUGH != 0;
}

// ---- stage 04: Algo_04_Sim.cpp neither vignette nor coating ---------------
FORCE_INLINE bool AlgoPass04 (const film::FilmProfile& profile,
                              const AlgoControls&      params) noexcept
{
    const AlgoType stops = (params.vignette < 0.0)
                         ? static_cast<AlgoType>(profile.default_vignette)
                         : static_cast<AlgoType>(params.vignette);
    const AlgoType coatScale = MAX_VALUE(static_cast<AlgoType>(params.coatingScale), ALGO_ZERO);
    const bool wantVignette = (stops > ALGO_ZERO);
    const bool wantCoating  = (coatScale > ALGO_ZERO) && (profile.coating.coating_sigma > 0.0);
    return (ALGO_FORWARD_PASS_THROUGH != 0) && (false == wantVignette) && (false == wantCoating);
}

// ---- stage 05: Algo_05_Sim.cpp no gain, no scale, no geometry -------------
FORCE_INLINE bool AlgoPass05 (const film::FilmProfile& profile,
                              const AlgoControls&      params,
                              const AlgoType           pxPerMm) noexcept
{
    const film::HalationSpec& hal = profile.halation;
    const AlgoType scale = MAX_VALUE(static_cast<AlgoType>(params.halationScale), ALGO_ZERO);
    const bool anyGain = (static_cast<AlgoType>(hal.gain_r) > ALGO_ZERO)
                      || (static_cast<AlgoType>(hal.gain_g) > ALGO_ZERO)
                      || (static_cast<AlgoType>(hal.gain_b) > ALGO_ZERO);
    return (ALGO_FORWARD_PASS_THROUGH != 0)
        && ((false == anyGain) || (scale <= ALGO_ZERO) || (pxPerMm <= ALGO_ZERO));
}

// ---- stage 06b: Algo_06_Sim.cpp loss <= 0 ---------------------------------
FORCE_INLINE bool AlgoPass06b (const film::FilmProfile& profile,
                               const AlgoControls&      params) noexcept
{
    const AlgoType scale = MAX_VALUE(static_cast<AlgoType>(params.coatingScale), ALGO_ZERO);
    const AlgoType loss  = MIN_VALUE(static_cast<AlgoType>(profile.coating.buckle_mtf_loss) * scale,
                                     ALGO_DEFOCUS_MAX_LOSS);
    return (ALGO_FORWARD_PASS_THROUGH != 0) && (loss <= ALGO_ZERO);
}

// ---- stage 07: Algo_07_Sim.cpp tripack colour without a mosaic ------------
FORCE_INLINE bool AlgoPass07 (const film::FilmProfile& profile,
                              const bool               hasMosaic) noexcept
{
    return (ALGO_FORWARD_PASS_THROUGH != 0)
        && (false == profile.is_monochrome) && (false == hasMosaic);
}

// ---- stage 08b: Algo_08_Sim.cpp inactive interimage or monochrome ---------
FORCE_INLINE bool AlgoPass08b (const film::FilmProfile& profile) noexcept
{
    return (ALGO_FORWARD_PASS_THROUGH != 0)
        && ((false == profile.interimage.active()) || profile.is_monochrome);
}

// ---- stage 09: Algo_09_Sim.cpp neither long-range nor edge term -----------
FORCE_INLINE bool AlgoPass09 (const film::FilmProfile& profile,
                              const AlgoControls&      params,
                              const AlgoType           pxPerMm) noexcept
{
    const film::CouplerSpec& cp = profile.couplers;
    const AlgoType scale = MAX_VALUE(static_cast<AlgoType>(params.couplerScale), ALGO_ZERO);
    const AlgoType s = static_cast<AlgoType>(cp.strength)      * scale;
    const AlgoType e = static_cast<AlgoType>(cp.edge_strength) * scale;
    const AlgoType radiusPx = static_cast<AlgoType>(cp.radius_um) * static_cast<AlgoType>(0.001) * pxPerMm;
    const AlgoType edgePx   = static_cast<AlgoType>(cp.edge_um)   * static_cast<AlgoType>(0.001) * pxPerMm;
    const bool wantLong = (s > ALGO_ZERO) && (false == profile.is_monochrome)
                       && (radiusPx >= ALGO_COUPLER_MIN_SIGMA_PX);
    const bool wantEdge = (e > ALGO_ZERO) && (edgePx >= ALGO_COUPLER_MIN_SIGMA_PX);
    return (ALGO_FORWARD_PASS_THROUGH != 0) && (false == wantLong) && (false == wantEdge);
}

// ---- stage 09b: Algo_09_Sim.cpp damage off or every level zero ------------
FORCE_INLINE bool AlgoPass09b (const AlgoControls& params) noexcept
{
    if (ALGO_FORWARD_PASS_THROUGH == 0) return false;
    if (false == params.filmDamageEnabled) return true;
    const FilmDamage& dmg = params.damage;
    const HighPrecType strength = MAX_VALUE(static_cast<HighPrecType>(dmg.damageStrength), 0.0);
    if (strength <= 0.0) return true;
    const HighPrecType dust   = MAX_VALUE(static_cast<HighPrecType>(dmg.dustLevel),        0.0) * strength;
    const HighPrecType debris = MAX_VALUE(static_cast<HighPrecType>(dmg.debrisLevel),      0.0) * strength;
    const HighPrecType fibre  = MAX_VALUE(static_cast<HighPrecType>(dmg.fibreLevel),       0.0) * strength;
    const HighPrecType sT     = MAX_VALUE(static_cast<HighPrecType>(dmg.scratchTransport), 0.0) * strength;
    const HighPrecType sH     = MAX_VALUE(static_cast<HighPrecType>(dmg.scratchHandling),  0.0) * strength;
    return (dust <= 0.0 && debris <= 0.0 && fibre <= 0.0 && sT <= 0.0 && sH <= 0.0);
}

// ---- stage 10: Algo_10_Sim.cpp neither blur nor shift ---------------------
FORCE_INLINE bool AlgoPass10 (const film::FilmProfile& profile,
                              const AlgoControls&      params,
                              const AlgoType           scanF50,
                              const AlgoType           pxPerMm) noexcept
{
    const AlgoType sigmaPx = AlgoScanSigmaMm(scanF50) * pxPerMm;
    const bool wantBlur = (sigmaPx >= ALGO_SCAN_MIN_SIGMA_PX);
    const AlgoType misPx = static_cast<AlgoType>(profile.misregistration_um)
                         * pxPerMm * static_cast<AlgoType>(0.001)
                         * MAX_VALUE(static_cast<AlgoType>(params.misregScale), ALGO_ZERO);
    const bool wantShift = (misPx > ALGO_ZERO) && (false == profile.is_monochrome);
    return (ALGO_FORWARD_PASS_THROUGH != 0) && (false == wantBlur) && (false == wantShift);
}

// ---- stage 10b: Algo_10_Sim.cpp no fog ------------------------------------
FORCE_INLINE bool AlgoPass10b (const film::FilmProfile& profile,
                               const AlgoControls&      params,
                               const AlgoType           negWidthMm) noexcept
{
    const AlgoType scale = MAX_VALUE(static_cast<AlgoType>(params.coatingScale), ALGO_ZERO);
    const AlgoType fogD  = static_cast<AlgoType>(profile.coating.edge_fog_density) * scale;
    const AlgoType fogMm = static_cast<AlgoType>(profile.coating.edge_fog_mm);
    return (ALGO_FORWARD_PASS_THROUGH != 0)
        && ((fogD <= ALGO_ZERO) || (fogMm <= ALGO_ZERO) || (negWidthMm <= ALGO_ZERO));
}

// ---- stage 11: Algo_11_Sim.cpp gain <= 0 ----------------------------------
FORCE_INLINE bool AlgoPass11 (const AlgoControls& params) noexcept
{
    const AlgoType gain = MAX_VALUE(static_cast<AlgoType>(params.grainScale), ALGO_ZERO);
    return (ALGO_FORWARD_PASS_THROUGH != 0) && (gain <= ALGO_ZERO);
}

// ---- stage 12: Algo_12_Sim.cpp identity dye matrix -------------------------
FORCE_INLINE bool AlgoPass12 (const film::FilmProfile& profile) noexcept
{
    return (ALGO_FORWARD_PASS_THROUGH != 0) && AlgoIsIdentityMatrix(profile.dye_matrix);
}

// ---- stage 13: Algo_13_Sim.cpp reversal or no print stock -----------------
//      (finalCurves stays profile.curves, which is what the caller initialises
//      it to before the call)
FORCE_INLINE bool AlgoPass13 (const film::FilmProfile& profile,
                              const film::PrintStock*  pPrintStock) noexcept
{
    return (ALGO_FORWARD_PASS_THROUGH != 0)
        && (profile.isReversal() || (nullptr == pPrintStock));
}

// ---- stage 14b: Algo_14_Sim.cpp no mosaic and a unit base tint ------------
FORCE_INLINE bool AlgoPass14b (const film::FilmProfile& profile,
                               const bool               hasMosaic) noexcept
{
    if ((ALGO_FORWARD_PASS_THROUGH == 0) || hasMosaic) return false;
    for (int32_t c = 0; c < 3; c++)
        if (static_cast<AlgoType>(AlgoTintFactor(profile, c)) != ALGO_ONE)
            return false;
    return true;
}

// ---- stage 14c: Algo_14_Sim.cpp colour stock or zero tone -----------------
FORCE_INLINE bool AlgoPass14c (const film::FilmProfile& profile) noexcept
{
    const AlgoType tone = static_cast<AlgoType>(profile.silver_tone);
    return (ALGO_FORWARD_PASS_THROUGH != 0)
        && ((false == profile.is_monochrome) || (tone == ALGO_ZERO));
}

// ---- stage 15: Algo_15_Sim.cpp damage off or zero amplitude ---------------
FORCE_INLINE bool AlgoPass15 (const film::FilmProfile& profile,
                              const AlgoControls&      params,
                              const AlgoType           pxPerMm) noexcept
{
    if (ALGO_FORWARD_PASS_THROUGH == 0) return false;
    if (false == params.filmDamageEnabled) return true;
    const FilmDamage& dmg = params.damage;
    const HighPrecType strength = MAX_VALUE(static_cast<HighPrecType>(dmg.damageStrength), 0.0);
    const HighPrecType level    = MAX_VALUE(static_cast<HighPrecType>(dmg.weaveAmount), 0.0) * strength;
    const HighPrecType ampXpx = (static_cast<HighPrecType>(profile.temporal.weave_amp_x_um)
                                 / ALGO_WEAVE_UM_PER_MM) * level * static_cast<HighPrecType>(pxPerMm);
    const HighPrecType ampYpx = (static_cast<HighPrecType>(profile.temporal.weave_amp_y_um)
                                 / ALGO_WEAVE_UM_PER_MM) * level * static_cast<HighPrecType>(pxPerMm);
    return (ampXpx <= 0.0 && ampYpx <= 0.0);
}

// ---- stage 16: Algo_16_Sim.cpp damage off or no dirt and no events --------
FORCE_INLINE bool AlgoPass16 (const AlgoControls& params) noexcept
{
    if (ALGO_FORWARD_PASS_THROUGH == 0) return false;
    if (false == params.filmDamageEnabled) return true;
    const FilmDamage& dmg = params.damage;
    const HighPrecType strength = MAX_VALUE(static_cast<HighPrecType>(dmg.damageStrength), 0.0);
    if (strength <= 0.0) return true;
    const HighPrecType dirt  = MAX_VALUE(static_cast<HighPrecType>(dmg.gateDirt),     0.0) * strength;
    const HighPrecType event = MAX_VALUE(static_cast<HighPrecType>(dmg.damageEvents), 0.0) * strength;
    return (dirt <= 0.0 && event <= 0.0);
}

#endif // ALGO_PASS_THROUGH_HPP
