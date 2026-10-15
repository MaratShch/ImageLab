#pragma once

// ---------------------------------------------------------------------------
//  AlgoDirCoupler.hpp
//
//  Stage 9 of the film simulation pipeline: DIR coupler lateral effects.
//
//  PHYSICAL BACKGROUND
//
//  DIR stands for Development Inhibitor Releasing. The coupler that forms dye
//  during development also releases a compound that inhibits further development,
//  and that inhibitor DIFFUSES. Stage 8b modelled the vertical half - inhibitor
//  crossing between layers. This is the lateral half: inhibitor spreading
//  sideways WITHIN a layer, from a dense area into the lighter area beside it.
//
//  Same chemistry, same molecules, two directions, two quite different visual
//  results, which is why they are two stages.
//
//  TWO COMPONENTS AT TWO SCALES
//
//  The long-range term pushes each layer away from the LOCALLY BLURRED MEAN of
//  all three. That raises saturation without raising gamma, which is the real DIR
//  mechanism and the thing no tone curve can imitate: the grey scale keeps its
//  contrast while colours separate.
//
//  The short-range term is classic adjacency. Each layer is pushed away from its
//  own blurred self, which sharpens edges. This is unsharp masking in the density
//  domain, arrived at by chemistry rather than by choice, and the reason a
//  coupler-rich negative looks crisper than its MTF alone predicts.
//
//  WHY IT IS AFTER THE CURVE AND NOT BEFORE
//
//  The inhibitor is released BY development in proportion to the dye being formed,
//  so its amount is a function of DENSITY, not of exposure. Modelling it in the
//  exposure domain would make the effect proportional to light rather than to
//  development, and it would then behave wrongly in the shoulder, where a large
//  change in exposure produces almost no change in density and therefore almost no
//  inhibitor.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// Frequency-domain filter (owner FFT library, 2026-10-06).
#include "AlgoFrequency.hpp"

// The separable Gaussian used for both diffusion scales.
#include "AlgoSeparableBlur.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including CouplerSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  ⚠ NO SUB-PIXEL GATE SINCE 2026-10-06 (owner decision). Until then both terms
//  were switched off below ALGO_COUPLER_MIN_SIGMA_PX = 0.25 px, because the old
//  spatial kernel had one tap there. Stage 9 now multiplies by the analytic
//  Gaussian transfer (AlgoFrequency.hpp), which is exact at any radius, and as
//  the radius shrinks the long-range term tends to a plain saturation boost
//  s * (D - mean D) -- which the gate used to cut off abruptly on small frames
//  (about 180-280 px wide for 35 mm). Both terms now run whenever their radius
//  is positive. Twin: film_sim.apply_dir_couplers / coupler_flat_scale.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
//  Coupler scale for the NEUTRAL references (anchor solve, print-chain mid grey).
//
//  ⚠ ADDED 2026-10-02 (owner-approved). Stage 9 then switched its long-range term
//  off when the diffusion radius was under 0.25 px, but
//  AlgoSolveAnchors and AlgoNeutralMidDensity modelled the flat-field coupling
//  unconditionally, so frames narrower than about 180 px (35 mm) anchored for a
//  coupling they never received and rendered mid grey with a cast. Both
//  references saw the same gate as stage 9. Since 2026-10-06 there is no gate:
//  the scale applies whenever the radius is positive. Twin: film_sim.coupler_flat_scale.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoCouplerFlatScale
(
    const film::FilmProfile& profile,
    const HighPrecType       couplerScale,
    const AlgoType           pxPerMm
) noexcept
{
    const AlgoType radiusPx = static_cast<AlgoType>(profile.couplers.radius_um)
                            * static_cast<AlgoType>(0.001) * pxPerMm;

    return (radiusPx > static_cast<AlgoType>(0))
               ? couplerScale : static_cast<HighPrecType>(0.0);
}


// ---------------------------------------------------------------------------
//  Stage 9: DIR coupler lateral effects.
//
//  pSrcR/G/B     density in
//  pDstR/G/B     density out, floored at zero
//  pScrDbar      scratch: mean of the three densities
//  pScrDbarBlur  scratch: blurred mean, and later the blurred single channel
//  pScrBlurA     scratch: separable blur workspace
//  pScrBlurB     scratch: separable blur workspace
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  params        user controls; couplerScale scales both components
//  pxPerMm       render resolution, used to turn micrometre radii into pixels
//
//  The four scratch planes must be distinct from each other, from the source and
//  from the destination.
// ---------------------------------------------------------------------------
void AlgoStage09_DirCoupler
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrDbar,
    AlgoType* RESTRICT       pScrDbarBlur,
    AlgoType* RESTRICT       pScrBlurA,
    AlgoType* RESTRICT       pScrBlurB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           pxPerMm,
    const AlgoFreqState&     freq
) noexcept;
