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
//  THE TRANSFER (film_sim FreqGrid.mtf, 2026-10-06 exact in the engines)
//
//  The reference expresses the transfer in the frequency domain, per channel,
//  with f in cycles/mm on the frame's grid:
//
//      measured stock (mtf_measured, q > 0):  MTF(f) = 1 / (1 + (f/f50)^q)
//      otherwise (the legacy law):           MTF(f) = exp(-ln2 * (f/f50)^2)
//      f50 <= 0:                             MTF(f) = 1
//
//  Both laws give MTF(f50) = 0.5 exactly (film::FilmMtfResponse is the one
//  definition). The engines evaluate the SAME law on the SAME grid and apply it
//  with the owner's FFT (AlgoFrequency.hpp), so stage 6 is now the identical
//  circular convolution in all three implementations. Until 2026-10-06 the
//  engines, having no FFT, convolved separable Gaussians: one for the legacy law
//  and a two- or three-lobe fit (film::FilmMtfKernel / FilmMtfKernel3) for the
//  measured one, and skipped any lobe below 0.25 px. Those tables remain in the
//  database for film_sim's mtf_use_kernel diagnostic only.
//
//  DEVELOPMENT ADJACENCY
//
//  Real MTF curves frequently exceed 100 per cent at low spatial frequency. That
//  is not a measurement error: during development the exhausted developer and
//  the released inhibitor diffuse sideways out of a dense area into an adjacent
//  light one, suppressing development there and exaggerating the edge. The
//  effect peaks at the diffusion scale and returns to unity at both DC and high
//  frequency, so it is a BAND-PASS lift multiplying the base transfer:
//
//      lift(f) = 1 + a * ( G(0.4 * adj) - G(2.0 * adj) )
//
//  (ALGO_FREQ_ADJACENCY_INNER / _OUTER). A plain unsharp term of the form
//  1 + a - a*G would instead settle at 1 + a for every high frequency, which is
//  a permanent global sharpening and not an adjacency effect at all. The lift is
//  1 at DC, so the filter cannot shift the overall exposure level.
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

// Frequency-domain filter (owner FFT library).
#include "AlgoFrequency.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Stock parameters, including MTFSpec.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  Stage 6: emulsion MTF.
//
//  pSrcR/G/B     linear exposure in
//  pDstR/G/B     linear exposure out, clamped at zero
//  pScrBlurA     unused since 2026-10-06 (kept for the stable signature)
//  pScrBlurB     unused since 2026-10-06
//  freq          frequency-domain state (AlgoFrequency.hpp), for this frame
//  sizeX/sizeY   active pixel extent
//  pitch         row stride in ELEMENTS
//  profile       stock being simulated
//  pxPerMm       render resolution, used to turn cycles/mm into pixels
//
//  Source and destination must be distinct planes.
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
    const AlgoType           pxPerMm,
    const AlgoFreqState&     freq
) noexcept;
