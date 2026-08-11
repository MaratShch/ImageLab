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

// The separable Gaussian used for both diffusion scales.
#include "AlgoSeparableBlur.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including CouplerSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Smallest diffusion radius, in pixels, worth submitting to a blur.
//
//  Below a quarter of a pixel the discrete kernel has one significant tap and the
//  pass is an identity costing two full sweeps of the image.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_COUPLER_MIN_SIGMA_PX = static_cast<AlgoType>(0.25);


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
    const AlgoType           pxPerMm
) noexcept;
