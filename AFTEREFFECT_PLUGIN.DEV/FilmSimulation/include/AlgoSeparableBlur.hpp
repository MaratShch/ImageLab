#pragma once

// ---------------------------------------------------------------------------
//  AlgoSeparableBlur.hpp
//
//  Shared low-level filtering primitives. Not a pipeline stage: infrastructure
//  used by the veiling flare, halation, emulsion MTF and coupler stages, which is
//  why it lives in its own translation unit rather than inside any one
//  Algo_NN_Sim.cpp.
//
//  Every function takes RAW POINTERS plus explicit geometry. Nothing is wrapped.
//
//  WHY SEPARABLE GAUSSIAN AND NOT AN FFT
//
//  The reference model filters by multiplying a half-spectrum transfer function,
//  which makes the convolution CIRCULAR - the image wraps at its edges. A Gaussian
//  is separable, so the same operator can be applied as two one-dimensional passes
//  at a fraction of the cost and with no complex arithmetic, provided the boundary
//  wraps in exactly the same way. That is what these functions do, and it is why
//  the boundary mode is WRAP rather than the more usual clamp: the aim is to
//  reproduce the reference, not to pick the prettiest edge behaviour.
//
//  The remaining difference is kernel truncation. A Gaussian has infinite support
//  and the kernel is cut at a finite radius; ALGO_BLUR_SIGMA_CUTOFF sets how far
//  out it is carried and therefore how small that error is.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType.
#include "AlgoTypes.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  KERNEL TRUNCATION RADIUS, in standard deviations
//
//  4.0 sigma. A Gaussian carries 99.9937 per cent of its area within +/- 4 sigma,
//  so the truncated tail is about 6.3e-5 of the total weight. The kernel is
//  renormalised to sum to exactly 1.0 afterwards, so that missing tail becomes a
//  shape error rather than a brightness error - and a shape error of that size is
//  four orders of magnitude below anything the model claims to resolve.
//
//  3.0 sigma would leave 2.7e-3 outstanding, visible as a slightly tight halo on
//  a high-contrast edge. 5.0 sigma costs 25 per cent more work for an improvement
//  nothing downstream can measure.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_BLUR_SIGMA_CUTOFF = static_cast<AlgoType>(4.0);


// ---------------------------------------------------------------------------
//  MAXIMUM KERNEL HALF-WIDTH, in taps
//
//  64. With the 4-sigma cutoff this corresponds to sigma 16. The limit exists so
//  the kernel can live in a fixed-size stack array and the blur can honour the
//  no-heap-allocation rule without a size calculation that could surprise anyone.
//  A sigma above 16 pixels is clamped to this half-width, which widens the
//  effective kernel's error but never overruns the array.
//
//  Array length is 2*64+1 = 129 taps.
// ---------------------------------------------------------------------------
constexpr int32_t ALGO_BLUR_MAX_HALF_TAPS = 64;
constexpr int32_t ALGO_BLUR_MAX_TAPS      = 2 * ALGO_BLUR_MAX_HALF_TAPS + 1;

// Largest number of Gaussian lobes the multi-lobe form accepts.
constexpr int32_t ALGO_BLUR_MAX_LOBES = 4;


// ---------------------------------------------------------------------------
//  AlgoGaussianBlurPlaneWrap
//
//  Blur one plane by an isotropic Gaussian of the given standard deviation in
//  PIXELS, with wrap-around boundary handling to match the reference model's
//  circular convolution.
//
//  pSrc      plane to read. Not modified.
//  pDst      plane to write. Must not be the same plane as pSrc.
//  pScratch  one plane of working storage holding the horizontal pass result.
//            Supplied by the caller from the arena because this code may not
//            allocate. Must differ from both pSrc and pDst.
//  sizeX     active pixels per row.
//  sizeY     active rows.
//  pitch     elements from one row start to the next, for all three planes.
//  sigmaPx   standard deviation in pixels. At or below zero this copies pSrc to
//            pDst unchanged, which is the correct limit of a Gaussian of zero
//            width and removes a special case from every call site.
//
//  Two passes: horizontal pSrc -> pScratch, then vertical pScratch -> pDst. Both
//  wrap. Complexity is O(sizeX * sizeY * taps) with taps = 2*ceil(4*sigma)+1,
//  against O(sizeX * sizeY * taps^2) for a naive two-dimensional kernel.
// ---------------------------------------------------------------------------
void AlgoGaussianBlurPlaneWrap
(
    const AlgoType* RESTRICT pSrc,
    AlgoType* RESTRICT       pDst,
    AlgoType* RESTRICT       pScratch,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const AlgoType           sigmaPx
) noexcept;


// ---------------------------------------------------------------------------
//  AlgoMultiGaussianBlurPlaneWrap
//
//  Weighted sum of up to ALGO_BLUR_MAX_LOBES Gaussian blurs of the same plane:
//
//      dst = sum_k ( weight_k / sum(weights) ) * blur(src, sigma_k)
//
//  A single Gaussian gives a tight, plausible-looking halo. Real light scatter in
//  glass and in emulsion has a faint bloom reaching far beyond it, and that wide
//  low-amplitude tail is the part the eye reads as photochemical rather than
//  digital. Summing a few Gaussians of very different widths is the cheapest way
//  to build such a long-tailed kernel.
//
//  Weights are normalised internally, so they may be given in any convenient
//  scale; only their ratios matter. That keeps the operator energy-preserving
//  whatever the caller passes.
//
//  pScratchA  per-lobe result.
//  pScratchB  separable intermediate inside each blur.
//  Both come from the arena and must differ from pSrc and pDst.
// ---------------------------------------------------------------------------
void AlgoMultiGaussianBlurPlaneWrap
(
    const AlgoType* RESTRICT pSrc,
    AlgoType* RESTRICT       pDst,
    AlgoType* RESTRICT       pScratchA,
    AlgoType* RESTRICT       pScratchB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const AlgoType           sigmaPx[ALGO_BLUR_MAX_LOBES],
    const AlgoType           weight [ALGO_BLUR_MAX_LOBES],
    const int32_t            lobeCount
) noexcept;


// ---------------------------------------------------------------------------
//  AlgoPlaneMean
//
//  Arithmetic mean of the active pixels of one plane, row padding excluded.
//
//  Accumulated in HighPrecType regardless of AlgoType. A single-precision
//  accumulator over two million samples loses the low bits of the running total
//  once it has grown large relative to each addend; measured against a double
//  accumulator the error reaches roughly 5e-4 relative at HD and 1e-2 at 4K. The
//  veiling flare uses this mean directly as its uniform veil, so an error of that
//  size would be a visible shift in the black floor.
// ---------------------------------------------------------------------------
HighPrecType AlgoPlaneMean
(
    const AlgoType* RESTRICT pSrc,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch
) noexcept;


// ---------------------------------------------------------------------------
//  AlgoCopyPlane / AlgoCopyImage
//
//  Copy the ACTIVE pixels of one plane, or of three planes, row by row.
//
//  Row-wise rather than one block copy because the planes carry row padding for
//  alignment. That padding is not part of the image; copying it would be wasteful
//  and, on the final row, a read beyond the plane's useful extent.
//
//  Used by every stage that does not apply to a given stock. Under the retained
//  buffer policy such a stage must still leave a valid image in its own buffer, so
//  it copies rather than skipping the pass.
// ---------------------------------------------------------------------------
void AlgoCopyPlane
(
    const AlgoType* RESTRICT pSrc,
    AlgoType* RESTRICT       pDst,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch
) noexcept;

void AlgoCopyImage
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch
) noexcept;


// ---------------------------------------------------------------------------
//  AlgoBilinearUpsample
//
//  Bilinear interpolation of a small field up to the full frame geometry.
//
//  Source and destination corners are aligned: source index 0 maps to destination
//  index 0, and source index (n-1) to destination index (m-1). This is corner
//  alignment rather than the centre alignment more usual in image resampling, and
//  it is deliberate - the low-resolution field is a sampled continuous function,
//  not an image of pixels, so its first and last samples are the values at the
//  frame edges and must land exactly there.
//
//  loW / loH describe the active extent of the source, which is smaller than the
//  plane it occupies: the field is synthesised into the top-left corner of an
//  arena-sized scratch plane rather than into a plane of its own, so no extra
//  arena entry is needed for it. loPitch is that plane's pitch.
// ---------------------------------------------------------------------------
void AlgoBilinearUpsample
(
    const AlgoType* RESTRICT pLo,
    const int32_t            loW,
    const int32_t            loH,
    const int32_t            loPitch,
    AlgoType* RESTRICT       pDst,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch
) noexcept;
