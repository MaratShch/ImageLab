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
#include <cmath>     // std::floor, std::ceil, std::sqrt -- used by AlgoBlurDetail


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
// ⚠ RAISED 4 -> 6 ON 2026-09-03 AND THE REASON IS STAGE 6, NOT THIS FILE.
// The emulsion MTF used one base lobe plus two adjacency lobes, which is three.
// A measured rolloff now enters as a WEIGHTED PAIR of base Gaussians (see
// film::FilmMtfKernel), and because the adjacency band-pass multiplies the base
// transfer rather than adding to it, each base lobe carries its own inner and
// outer adjacency partner: 2 x 3 = 6. Nothing else in the engine asks for more
// than three, so the extra capacity costs two unused array slots per call and
// no arithmetic.
constexpr int32_t ALGO_BLUR_MAX_LOBES = 6;


// ---------------------------------------------------------------------------
//  ALGO_BLUR_SIGMA_EXACT_MAX -- the largest sigma the direct kernel RENDERS
//  CORRECTLY, as opposed to the largest it accepts without crashing.
//
//  THE DEFECT THIS CONSTANT NAMES
//
//  ALGO_BLUR_MAX_HALF_TAPS reads like a cost control. It is not. Past this
//  sigma it CHANGES THE FILTER BEING APPLIED, and because the truncated taps
//  are renormalised to unit sum there is no brightness artefact and nothing
//  looks obviously wrong. Measured from the emitted kernel's own second moment:
//
//      requested sigma   captured area   effective sigma   ratio
//            16.00          99.994 %          15.99        1.000
//            40.00          89.040 %          31.18        0.779
//           100.00          47.783 %          36.21        0.362
//           820.00           6.221 %          37.22        0.045
//
//  The effective width saturates at 64/sqrt(3) = 36.95 px -- the standard
//  deviation of a UNIFORM kernel of half-width 64. For large sigma the sampled
//  Gaussian is flat across +/-64 px, so renormalisation turns it into a box
//  filter and every wide lobe collapses onto the same ~37 px box regardless of
//  the physics requested.
//
//  What that cost, measured against the database at super35:
//    - halation: 28 stocks affected at 1024 px, 90 at HD, 101 at 4K (worst
//      sigma 108). The halo was under-sized on most colour stocks at delivery
//      resolution, and the error GREW WITH OUTPUT SIZE -- the one direction in
//      which a physical model must not drift.
//    - veiling flare: the three lobes fixed at 1500 / 6000 / 20000 um render as
//      62 / 247 / 823 px at 1024 and ALL THREE came out at ~37 px. The
//      long-tailed sum the stage exists to build was not produced at any
//      resolution, on any of the stocks that use it.
//
//  THE GENERALISABLE LESSON, worth keeping in front of the next reader: when a
//  limit exists for an implementation reason (array size, tap count, iteration
//  cap), verify what the code DOES when the limit binds. A limit that silently
//  changes the model rather than refusing is a correctness defect wearing a
//  performance costume.
//
//  THE RULE THAT FOLLOWS
//    at or below this sigma  ->  direct kernel. It is exact, and using the
//                                exact method wherever it IS exact is the rule.
//    above it                ->  resample - blur - reconstruct (below).
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_BLUR_SIGMA_EXACT_MAX =
    static_cast<AlgoType>(ALGO_BLUR_MAX_HALF_TAPS) / ALGO_BLUR_SIGMA_CUTOFF;   // 16


// ---------------------------------------------------------------------------
//  RESAMPLE-BLUR-RECONSTRUCT parameters, SHARED BY BOTH PATHS.
//
//  These live in the shared header, not in either .cpp, for one reason: the two
//  paths must choose the SAME decimation factor and the SAME reduced extent for
//  the same sigma, or they resample onto different grids and every parity
//  figure measures the grid rather than the arithmetic. Having one definition
//  makes that agreement structural instead of something to re-check.
// ---------------------------------------------------------------------------

// Reduced-resolution sigma aimed for. The decimation factor is derived from it,
// so that whatever sigma is asked for, the low-resolution blur lands in the band
// where the direct kernel is both exact and short.
constexpr AlgoType ALGO_BLUR_PYRAMID_TARGET_SIGMA = static_cast<AlgoType>(4.0);

// Hard ceiling on the decimation factor, so a pathological sigma cannot reduce
// the plane to a handful of samples.
constexpr int32_t ALGO_BLUR_PYRAMID_MAX_K = 8;

// Preview-quality engagement threshold. NOT used by the Full path, which
// engages only above ALGO_BLUR_SIGMA_EXACT_MAX because below that the direct
// kernel is exact and exactness wins. It is defined here so that the Lite/
// Preview quality preset has a named constant to reach for rather than a
// literal, and so the number carries its own justification:
//
//   below sigma 3.5 the decimated plane's own blur is too narrow to be worth
//   the two resampling passes -- measured, the pyramid stopped paying at
//   roughly this width once the decimation itself was vectorised.
constexpr AlgoType ALGO_BLUR_PYRAMID_MIN_SIGMA_PREVIEW = static_cast<AlgoType>(3.5);


// ---------------------------------------------------------------------------
//  AlgoBlurDetail -- the resampling geometry, shared by the scalar and AVX2
//  implementations.
//
//  WHY A RATIONAL CELL WIDTH AND NOT AN INTEGER BLOCK
//
//  The obvious decimation is "average exact k x k blocks". It is wrong on a
//  periodic domain whenever k does not divide the extent. k = 5 on 512 gives
//  loW = ceil(512/5) = 103, and 103 * 5 = 515 samples drawn from a 512 period,
//  so three columns are counted twice and a seam appears at the wrap. Measured,
//  that seam was the whole of the last scalar/AVX2 blur divergence: 1.235e-2 at
//  sigma ~ 20, against 5.6e-7 - 1.0e-6 everywhere else.
//
//  The fix is to stop pretending the cell is an integer. With
//
//      R = n / loN            (rational, slightly less than k)
//
//  coarse cell j covers the half-open interval [j*R, (j+1)*R) of the input
//  axis, and the loN cells tile exactly one period for ANY n and loN. Each
//  input sample contributes its overlap length; the weights sum to exactly R,
//  so dividing by R makes the operator a partition of unity in the forward
//  direction. Reconstruction is linear interpolation between the two nearest
//  cell CENTRES, which is a partition of unity in the reverse direction.
//
//  Together that means a constant field survives the round trip exactly -- and
//  it must, because this resampling sits in front of energy-conserving stages
//  (halation's blur(above) - above) where a constant-field error is not a small
//  error but a change of the physics.
// ---------------------------------------------------------------------------
namespace AlgoBlurDetail
{
    // Wrap an index into [0, n). Two loops rather than a modulo: the offset is
    // always within one period of the range here, and a modulo of a signed
    // value costs a division.
    inline int32_t wrapPeriodic (int32_t i, const int32_t n) noexcept
    {
        while (i < 0)  i += n;
        while (i >= n) i -= n;
        return i;
    }

    // Largest number of input samples one coarse cell can draw from:
    // floor(R) + 2 at worst, and R <= ALGO_BLUR_PYRAMID_MAX_K.
    constexpr int32_t ALGO_BLUR_CELL_MAX_TAPS = ALGO_BLUR_PYRAMID_MAX_K + 2;

    // -----------------------------------------------------------------------
    //  cellWeights
    //
    //  Area weights of coarse cell j over the input axis of extent n, for a
    //  reduced extent of loN. Writes the wrapped input indices and their
    //  NORMALISED weights (already divided by R, so they sum to 1) and returns
    //  how many there are.
    //
    //  Exposed rather than kept private because the AVX2 path precomputes one
    //  axis of these once per call and applies them eight lanes at a time; it
    //  must use these exact weights, not its own.
    // -----------------------------------------------------------------------
    inline int32_t cellWeights
    (
        const int32_t     j,
        const int32_t     n,
        const int32_t     loN,
        int32_t           idxOut[ALGO_BLUR_CELL_MAX_TAPS],
        HighPrecType      wOut  [ALGO_BLUR_CELL_MAX_TAPS]
    ) noexcept
    {
        const HighPrecType R  = static_cast<HighPrecType>(n)
                              / static_cast<HighPrecType>(loN);

        const HighPrecType lo = static_cast<HighPrecType>(j) * R;
        const HighPrecType hi = lo + R;

        const int32_t i0 = static_cast<int32_t>(std::floor(lo));
        const int32_t i1 = static_cast<int32_t>(std::ceil (hi)) - 1;

        const HighPrecType invR = 1.0 / R;

        int32_t count = 0;

        for (int32_t i = i0; i <= i1 && count < ALGO_BLUR_CELL_MAX_TAPS; i++)
        {
            const HighPrecType a = MAX_VALUE(lo, static_cast<HighPrecType>(i));
            const HighPrecType b = MIN_VALUE(hi, static_cast<HighPrecType>(i + 1));
            const HighPrecType w = b - a;

            if (w <= 0.0) continue;

            idxOut[count] = wrapPeriodic(i, n);
            wOut  [count] = w * invR;
            count++;
        }

        return count;
    }

    // -----------------------------------------------------------------------
    //  upWeights
    //
    //  Reconstruction weights for output sample x of an axis of extent n from a
    //  coarse axis of extent loN.
    //
    //  Coarse sample j represents the value at input coordinate (j + 0.5) * R;
    //  output sample x sits at x + 0.5. Both neighbours are taken modulo loN,
    //  so the reconstruction is periodic in the same sense the forward pass is.
    //  Corner alignment would be wrong here -- these are cell averages of a
    //  periodic field, not corner samples of a bounded one.
    // -----------------------------------------------------------------------
    inline void upWeights
    (
        const int32_t  x,
        const int32_t  n,
        const int32_t  loN,
        int32_t&       j0Out,
        int32_t&       j1Out,
        HighPrecType&  fracOut
    ) noexcept
    {
        const HighPrecType R = static_cast<HighPrecType>(n)
                             / static_cast<HighPrecType>(loN);

        const HighPrecType t = (static_cast<HighPrecType>(x) + 0.5) / R - 0.5;

        const HighPrecType fj = std::floor(t);

        j0Out   = wrapPeriodic(static_cast<int32_t>(fj),     loN);
        j1Out   = wrapPeriodic(static_cast<int32_t>(fj) + 1, loN);
        fracOut = t - fj;

        return;
    }

    // -----------------------------------------------------------------------
    //  BlurPlan -- the decision and the derived geometry, in one place.
    //
    //  usePyramid   false means "use the direct kernel", and the direct kernel
    //               is then exact by construction because sigma is at or below
    //               ALGO_BLUR_SIGMA_EXACT_MAX.
    //  k            decimation factor, integer, shared by both paths.
    //  loW / loH    reduced extents, ceil(n / k). NOT n/k: the last cell is
    //               partial and the rational geometry handles it exactly.
    //  sigmaLo      sigma to apply ON the reduced grid.
    //
    //  VARIANCE COMPENSATION -- all THREE filters in the cascade, and the third
    //  is the one that was missing:
    //
    //      area decimation, cell width R      variance (R^2 - 1) / 12
    //      Gaussian sigmaLo on the coarse grid  variance R^2 * sigmaLo^2
    //      linear reconstruction, base 2R     variance R^2 / 6
    //
    //      =>  sigmaLo^2 = ( sigma^2 - (R^2 - 1)/12 - R^2/6 ) / R^2
    //
    //  Omitting the reconstruction term makes the result about 2 per cent
    //  systematically too broad at every sigma. That is a BIAS, not noise: it
    //  does not average away over stocks, and it is invisible to any test that
    //  only compares the two paths against each other, because both were wrong
    //  by the same amount.
    // -----------------------------------------------------------------------
    struct BlurPlan
    {
        bool      usePyramid;
        int32_t   k;
        int32_t   loW;
        int32_t   loH;
        AlgoType  sigmaLo;
    };

    inline BlurPlan planBlur
    (
        const AlgoType sigmaPx,
        const int32_t  sizeX,
        const int32_t  sizeY,
        const AlgoType engageAbove
    ) noexcept
    {
        BlurPlan p;
        p.usePyramid = false;
        p.k          = 1;
        p.loW        = sizeX;
        p.loH        = sizeY;
        p.sigmaLo    = sigmaPx;

        // Strictly greater than: AT the exact-max the direct kernel is still
        // exact, and the exact method wins wherever it is available. Using >=
        // here would push sigma exactly 16 onto the approximate path for no
        // reason, and that off-by-one is precisely how the two paths came to
        // disagree at the threshold.
        if (sigmaPx <= engageAbove)
            return p;

        int32_t k = static_cast<int32_t>(
            sigmaPx / ALGO_BLUR_PYRAMID_TARGET_SIGMA + static_cast<AlgoType>(0.5));

        k = CLAMP_VALUE(k, 2, ALGO_BLUR_PYRAMID_MAX_K);

        const int32_t loW = (sizeX + k - 1) / k;
        const int32_t loH = (sizeY + k - 1) / k;

        if (loW < 4 || loH < 4)
            return p;

        // The two axes have slightly different rational cell widths whenever
        // the extents are not both exact multiples of k. The compensation uses
        // the geometric mean of the two, because the blur that follows is
        // isotropic and applies one sigma to both.
        const HighPrecType Rx = static_cast<HighPrecType>(sizeX)
                              / static_cast<HighPrecType>(loW);
        const HighPrecType Ry = static_cast<HighPrecType>(sizeY)
                              / static_cast<HighPrecType>(loH);

        const HighPrecType R2 = Rx * Ry;              // (geometric mean)^2

        const HighPrecType s  = static_cast<HighPrecType>(sigmaPx);

        const HighPrecType boxVar   = (R2 - 1.0) / 12.0;
        const HighPrecType reconVar = R2 / 6.0;

        const HighPrecType targetVar = s * s - boxVar - reconVar;

        if (targetVar <= 0.0)
            return p;

        p.usePyramid = true;
        p.k          = k;
        p.loW        = loW;
        p.loH        = loH;
        p.sigmaLo    = static_cast<AlgoType>(std::sqrt(targetVar / R2));

        return p;
    }

}   // namespace AlgoBlurDetail


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
