#pragma once

// ---------------------------------------------------------------------------
//  AlgoEmulsionMtf.hpp
//
//  Stage 6 of the film simulation pipeline: emulsion modulation transfer.
//
//  PHYSICAL BACKGROUND
//
//  An emulsion is a suspension of silver halide crystals in gelatin, and light
//  entering it is scattered by those crystals before it is absorbed. A point of
//  light on the surface therefore exposes a small patch, not a point. The
//  modulation transfer function is how much contrast survives at a given spatial
//  frequency, and f50 - the frequency at which half the contrast survives - is
//  the single number that summarises it.
//
//  WHY THIS IS NOT SIMPLY BLUR
//
//  Two properties distinguish it from an arbitrary softening.
//
//  First, it acts on the EXPOSURE, before development. Grain is created during
//  development, after this point, so the emulsion MTF blurs the image but NOT
//  the grain. Applying a general blur later would smear the grain too, and grain
//  that is smoother than the image it sits on is the immediate visual signature
//  of a film simulation that has its stage order wrong.
//
//  Second, red is softest. The layers are stacked with red at the bottom, so red
//  light traverses two further layers of gelatin before it is recorded and is
//  scattered by both. The per-channel f50 triple carries that, and it is a real
//  and visible asymmetry, not a modelling convenience.
//
//  THE GAUSSIAN FORM AND ITS SIGMA
//
//  The reference expresses the transfer directly in the frequency domain as
//
//      MTF(f) = exp(-ln2 * (f / f50)^2)
//
//  which gives MTF(f50) = 0.5 exactly. A Gaussian blur of standard deviation s
//  millimetres has the transfer exp(-2 pi^2 s^2 f^2). Equating the exponents,
//
//      ln2 / f50^2 = 2 pi^2 s^2
//      s = sqrt(ln2 / 2) / (pi * f50)
//
//  so the frequency-domain filter is exactly a spatial Gaussian blur, and no
//  transform is needed to apply it. The constant sqrt(ln2/2)/pi is folded into
//  ALGO_MTF_SIGMA_MM_PER_INV_F50 below.
//
//  DEVELOPMENT ADJACENCY
//
//  Real MTF curves frequently exceed 100 per cent at low spatial frequency. That
//  is not a measurement error: during development the exhausted developer and
//  the released inhibitor diffuse sideways out of a dense area into an adjacent
//  light one, suppressing development there and exaggerating the edge. The
//  effect peaks at the diffusion scale and returns to unity at both DC and high
//  frequency, so it is a BAND-PASS lift:
//
//      lift(f) = 1 + a * ( G(0.4 * adj) - G(2.0 * adj) )
//
//  A plain unsharp term of the form 1 + a - a*G would instead settle at 1 + a
//  for every high frequency, which is a permanent global sharpening and not an
//  adjacency effect at all.
//
//  Multiplying the base transfer by that lift expands to three Gaussian terms,
//
//      MTF * lift = MTF + a * MTF*G1 - a * MTF*G2
//
//  and a product of Gaussians in the frequency domain is a Gaussian whose
//  variances add. So the whole stage, adjacency included, is a weighted sum of
//  three spatial Gaussian blurs with weights (1, a, -a) and standard deviations
//  (s, sqrt(s^2 + s1^2), sqrt(s^2 + s2^2)). The weights sum to one, so the
//  filter has unit response at DC and cannot shift the overall exposure level.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// The separable multi-lobe Gaussian that carries out the filtering.
#include "AlgoSeparableBlur.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including MTFSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Conversion from f50 in cycles per millimetre to Gaussian sigma in
//  millimetres:
//
//      sigma_mm = sqrt(ln(2) / 2) / (pi * f50)
//
//  The numerator is sqrt(0.6931471805599453 / 2) = 0.5887050112577373 and
//  dividing by pi gives 0.18738564618678, which is the constant stored here.
//  Multiply it by the reciprocal of f50 to obtain sigma.
//
//  Written out as a literal rather than assembled from std::sqrt and M_PI so it
//  is a compile-time constant on every compiler and so the derivation above can
//  be checked against the digits by hand.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_MTF_SIGMA_MM_PER_INV_F50 =
    static_cast<AlgoType>(0.18738564618678);

// ---------------------------------------------------------------------------
//  Adjacency lobe scales, as multiples of the specified diffusion length.
//
//  The inner lobe at 0.4 of the scale and the outer at 2.0 give a band-pass
//  whose peak sits at the diffusion length itself. These are the shape of the
//  effect rather than free parameters: moving them moves the frequency at which
//  the overshoot peaks, which is set by chemistry.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_MTF_ADJACENCY_INNER = static_cast<AlgoType>(0.4);
constexpr AlgoType ALGO_MTF_ADJACENCY_OUTER = static_cast<AlgoType>(2.0);

// ---------------------------------------------------------------------------
//  Smallest sigma in pixels that is worth submitting to a separable blur.
//
//  Below a quarter of a pixel the discrete kernel has a single significant tap,
//  so the pass is an identity that costs two full sweeps of the image.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_MTF_MIN_SIGMA_PX = static_cast<AlgoType>(0.25);


// ---------------------------------------------------------------------------
//  Stage 6: emulsion MTF.
//
//  pSrcR/G/B     linear exposure in
//  pDstR/G/B     linear exposure out, clamped at zero
//  pScrBlurA     scratch: separable blur workspace
//  pScrBlurB     scratch: separable blur workspace
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  pxPerMm       render resolution, used to turn cycles/mm into pixels
//
//  The two scratch planes must be distinct from each other, from the source and
//  from the destination.
// ---------------------------------------------------------------------------
void AlgoStage06_EmulsionMtf
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrBlurA,
    AlgoType* RESTRICT       pScrBlurB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoType           pxPerMm
) noexcept;
