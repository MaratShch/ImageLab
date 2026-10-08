#pragma once

// ---------------------------------------------------------------------------
//  AlgoHalation.hpp
//
//  Stage 5 of the film simulation pipeline: halation.
//
//  PHYSICAL BACKGROUND
//
//  Film is not opaque. Light bright enough to expose the emulsion also passes
//  through it, strikes the back surface of the base, and part of it reflects
//  back up into the emulsion from underneath. It re-enters displaced sideways by
//  roughly twice the base thickness, so a small intense highlight acquires a
//  halo around it. That is halation, and it is the reason a street lamp on film
//  glows in a way a street lamp on a sensor does not.
//
//  Manufacturers fight it with an antihalation backing - a dye or a carbon layer
//  behind the emulsion that absorbs the through-light before it can bounce. Its
//  effectiveness is what separates one stock from another here. The famous case
//  is CineStill, which is Kodak motion picture stock with the remjet backing
//  removed, and glows accordingly.
//
//  WHY RED DOMINATES
//
//  The layers are stacked: blue-sensitive on top, then green, then red at the
//  bottom. Red light therefore penetrates furthest before it is recorded and is
//  closest to the base when it reflects, so the red record picks up the most
//  halation. This is why the halo around a tungsten highlight on colour negative
//  is orange-red rather than white. The per-channel gains carry that.
//
//  ENERGY IS CONSERVED, NOT CREATED
//
//  Light that scatters away from a point is REMOVED from that point and
//  deposited in the surround, rather than being invented. The stage therefore
//  adds
//
//      gain * (blur(above) - above)
//
//  and not gain * blur(above). The difference is not cosmetic. Adding the blur
//  alone injects a flat-field brightness lift proportional to the gain, which
//  contaminates the entire exposure scale and shifts mid grey. With the
//  subtraction, a large evenly lit highlight shows no net change in its interior
//  - correct, because every point there is both losing and receiving the same
//  amount - while a small bright source blooms into its neighbourhood and loses
//  a little of its own edge.
//
//  THE THRESHOLD IS A SOFT KNEE, AND IT IS TIGHT ON PURPOSE
//
//  Halation is a highlight effect. The threshold sits at 2^threshold_stops above
//  mid grey and the knee width is a small fraction of it. A loose knee leaks a
//  surprising amount of glow into the mid tones: at a gain of 1.05 a knee of
//  0.35 of the threshold lifted an 18 per cent grey card by 16 per cent, which
//  is a visible and entirely wrong global brightening.
//
//  THE SOURCE IS A BLEND, NOT THE CHANNEL ALONE
//
//  Light of every wavelength penetrates and returns, so the returning light at
//  any point is not purely that layer's own colour. The scatter source is half
//  the layer's own exposure and half the total luminance, which weights the
//  effect towards the deepest-penetrating light without making each layer blind
//  to the others.
//
//  DOMAIN
//
//  Linear exposure, before the characteristic curve. Halation happens at
//  exposure time inside the film sandwich, so it must precede development.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// The separable multi-lobe Gaussian used to spread the scattered light.
#include "AlgoSeparableBlur.hpp"

#include "AlgoFrequency.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including HalationSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Luminance weights for the scatter source.
//
//  The same broadcast luminance weights used by the veiling flare stage. They
//  are repeated here rather than shared because the two stages are physically
//  unrelated - one is a lens property and one is a base-reflection property -
//  and a future refinement to either must not silently move the other.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_HALATION_LUMA_R = static_cast<AlgoType>(0.30);
constexpr AlgoType ALGO_HALATION_LUMA_G = static_cast<AlgoType>(0.59);
constexpr AlgoType ALGO_HALATION_LUMA_B = static_cast<AlgoType>(0.11);

// ---------------------------------------------------------------------------
//  Split between the layer's own exposure and total luminance in the scatter
//  source. An even split: returning light is neither purely this layer's colour
//  nor colour-blind.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_HALATION_OWN_FRACTION  = static_cast<AlgoType>(0.5);
constexpr AlgoType ALGO_HALATION_LUMA_FRACTION = static_cast<AlgoType>(0.5);

// ---------------------------------------------------------------------------
//  Knee width as a fraction of the threshold. Small deliberately: see the note
//  on the tight knee above. Widening this is the fastest way to make every
//  render subtly and globally too bright.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_HALATION_KNEE_FRACTION = static_cast<AlgoType>(0.15);

// ---------------------------------------------------------------------------
//  Argument beyond which the softplus is replaced by its own asymptote.
//
//  softplus(x, k) = k * log(1 + exp(x/k)) tends to x for large x/k. At x/k = 60
//  the two agree to far beyond the last bit of a double, while exp(60) is still
//  finite - so this is a pure accuracy-preserving guard against the overflow
//  that exp(x/k) would otherwise reach a little further along.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_SOFTPLUS_LINEAR_LIMIT = static_cast<AlgoType>(60.0);

// ---------------------------------------------------------------------------
//  Number of Gaussian lobes in the halation scatter kernel.
//
//  Three, matching the radii and weight triples on HalationSpec. A single
//  Gaussian gives a tight halo with a hard edge; the wide low-amplitude third
//  lobe is the part the eye reads as photochemical rather than as a blur.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_HALATION_LOBES = 3;


// ---------------------------------------------------------------------------
//  The scatter TRANSFER of ONE colour record -- film_sim's
//  grid.multi_gaussian(*hal.lobes(c)). Shared by the Scalar and the AVX2
//  stage 5 so the two cannot drift.
//
//  The radii are stored in micrometres ON THE FILM, so the same stock gives a
//  halo of the same physical size whatever the render resolution; the transfer
//  is evaluated on the frame's f_mm grid (AlgoFrequency.hpp), which carries the
//  px_per_mm conversion.
//
//  PER-CHANNEL RADIUS (schema v11, owner decision G2): radius_scale_r/g/b
//  multiply ALL lobes of that record, including the ring, as lobes(c) does.
//
//  schema v59 THE RING: a SUBTRACTED lobe, appended with a NEGATIVE weight
//  when ring_um > 0 and ring_weight > 0. multi_gaussian normalises by the
//  SIGNED sum, so the kernel is sum(w_i G_i) - ring_weight G_ring over
//  (sum w_i - ring_weight). The ring is the support return's annulus
//  (r_c = 2t/sqrt(n^2-1)): no totally reflected light lands inside it.
//
//  ⚠ 2026-10-06 (FFT, owner decision): NO LOBE IS DROPPED ANY MORE. The
//  separable engine discarded a lobe below 0.25 px (and the ring with it)
//  because its sampled kernel collapsed to one tap there. film_sim never did:
//  a narrow Gaussian's transfer is simply close to 1 out to Nyquist. With the
//  frequency-domain filter the engine applies exactly film_sim's kernel at
//  every resolution, which is the maximum flow of the three implementations.
// ---------------------------------------------------------------------------
inline void AlgoHalationTransfer
(
    const film::HalationSpec& hal,
    const int32_t             channel,
    AlgoFreqTransfer&         t
) noexcept
{
    const float kScale[3] = { hal.radius_scale_r, hal.radius_scale_g, hal.radius_scale_b };
    const HighPrecType k = static_cast<HighPrecType>(kScale[(channel < 0) ? 0 : ((channel > 2) ? 2 : channel)]);

    HighPrecType radiiUm[ALGO_HALATION_LOBES + 1];
    HighPrecType weights[ALGO_HALATION_LOBES + 1];
    int32_t n = 0;
    for (int32_t i = 0; i < ALGO_HALATION_LOBES; i++)
    {
        radiiUm[n] = static_cast<HighPrecType>(hal.radii_um[i]) * k;
        weights[n] = static_cast<HighPrecType>(hal.weights[i]);
        n++;
    }
    if ((hal.ring_um > 0.0f) && (hal.ring_weight > 0.0f))
    {
        radiiUm[n] = static_cast<HighPrecType>(hal.ring_um) * k;
        weights[n] = -static_cast<HighPrecType>(hal.ring_weight);
        n++;
    }
    AlgoFreqSetMultiGaussian(t, radiiUm, weights, n);
}


// ---------------------------------------------------------------------------
//  Numerically safe softplus: k * log(1 + exp(x/k)).
//
//  Exposed because the same ramp shape is used by the characteristic curve, and
//  a second private copy would be one more place for the overflow guard to be
//  forgotten.
//
//  x  argument
//  k  knee width; must be strictly positive
// ---------------------------------------------------------------------------
AlgoType AlgoSoftplus (const AlgoType x, const AlgoType k) noexcept;


// ---------------------------------------------------------------------------
//  Stage 5: halation.
//
//  pSrcR/G/B     linear exposure in
//  pDstR/G/B     linear exposure out, clamped at zero
//  pScrLuma      scratch: broadcast luminance of the source
//  pScrAbove     scratch: the above-threshold scatter source for one channel
//  pScrBlur      scratch: the blurred scatter source for one channel
//  pScrBlurA     scratch: separable blur workspace
//  pScrBlurB     scratch: separable blur workspace
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  params        user controls
//  pxPerMm       render resolution, used to turn micrometre radii into pixels
//
//  All five scratch planes must be DISTINCT from each other and from both the
//  source and the destination. The multi-lobe blur reads its source once per
//  lobe, so a scratch plane aliased onto the source is destroyed by lobe one and
//  lobes two and three then integrate garbage.
// ---------------------------------------------------------------------------
void AlgoStage05_Halation
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrLuma,
    AlgoType* RESTRICT       pScrAbove,
    AlgoType* RESTRICT       pScrBlur,
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
