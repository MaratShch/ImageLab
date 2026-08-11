#if 0
// ---------------------------------------------------------------------------
//  AlgoSeparableBlur.cpp   --   AVX2
//
//      AlgoCopyPlane                   one plane, row by row
//      AlgoCopyImage                   three planes
//      AlgoGaussianBlurPlaneWrap       separable Gaussian, circular boundary
//      AlgoMultiGaussianBlurPlaneWrap  weighted sum of Gaussian lobes
//      AlgoPlaneMean                   frame mean, wide accumulator
//      AlgoBilinearUpsample            low-resolution field to full raster
//
//  Same filename, same function names, same prototypes as the scalar build.
//
//  WHY THIS FILE IS THE MOST VALUABLE ONE IN THE ENGINE
//
//  Measured at 1024 x 1024, EIGHT stages call into here - 3b, 5, 6, 9, 10, 11, 13
//  and 14b - and together they are 1212.69 ms of a 1439.28 ms frame: EIGHTY-FOUR
//  PER CENT. Almost none of that time is in the stage files themselves.
//  Algo_05_Sim.cpp measures 815 ms because of what it calls, not what it contains.
//
//  So one translation unit lifts eight stages at once, and it is the only place in
//  the engine where vector width can pay for itself: a halation lobe is a wide
//  kernel, which makes these loops genuinely arithmetic-bound rather than
//  bandwidth-bound. The pointwise stages vectorised earlier returned 1.03x to 1.16x
//  because they had nothing to do but wait for memory. This is different work.
//
//  ALL ARITHMETIC IS FLOAT32, with one deliberate exception noted at AlgoPlaneMean.
//
//  Raw pointers, explicit geometry, no allocation, no mutable state, no validation
//  of inputs.
//
//  ALIGNMENT: EVERY IMAGE ACCESS IS UNALIGNED, DELIBERATELY.
//
//  loadu/storeu rather than load/store on all plane data. The arena's base comes from
//  the host's memory pool, whose alignment argument is a HINT and not a guarantee -
//  the pool was observed returning 0x7fbef37fc010, which is 16 mod 32. Every plane is
//  then that base plus a multiple of the cache line, so every plane is 16 mod 32 too:
//  harmless to the scalar path, and an instant fault for an aligned 256-bit load.
//
//  The alternative was to align the head inside AlgoMemHandler.cpp, but that file is
//  SHARED by both flavours, and making shared infrastructure carry a vector-path
//  concern is the wrong direction - it would also mean the two builds no longer use
//  the same allocator, which is exactly the incompatibility to avoid.
//
//  The cost is nothing measurable. On Haswell and later an unaligned load of data that
//  happens to be aligned runs at the same rate as an aligned one; the only penalty is
//  on cache-line splits, which a 16-byte-offset base produces regardless of which
//  intrinsic is used. What is gained is that this code cannot fault on any base the
//  pool chooses to hand back.
//
//  The one aligned load that REMAINS is the tail-mask table, which is a file-local
//  static carrying AVX2_ALIGN - its alignment is guaranteed by the compiler, not by
//  the allocator.
// ---------------------------------------------------------------------------

#include "AlgoSeparableBlur.hpp"

#include <immintrin.h>
#include <cmath>     // std::exp, std::ceil, both once per kernel
#include <cstring>   // std::memcpy


static_assert(sizeof(AlgoType) == 4,
              "the AVX2 path requires AlgoType to be a 32-bit float");


namespace
{
    constexpr int32_t ALGO_AVX2_LANES = 8;


    // ----------------------------------------------------------------------
    //  Tail mask for a partial vector.
    //
    //  The active width is not generally a multiple of eight. Masked access leaves
    //  the row padding untouched, which keeps the NaN-poison arena test meaningful.
    // ----------------------------------------------------------------------
    inline __m256i algoTailMask (const int32_t n) noexcept
    {
        AVX2_ALIGN static const int32_t table[ALGO_AVX2_LANES][ALGO_AVX2_LANES] =
        {
            { 0,  0,  0,  0,  0,  0,  0,  0},
            {-1,  0,  0,  0,  0,  0,  0,  0},
            {-1, -1,  0,  0,  0,  0,  0,  0},
            {-1, -1, -1,  0,  0,  0,  0,  0},
            {-1, -1, -1, -1,  0,  0,  0,  0},
            {-1, -1, -1, -1, -1,  0,  0,  0},
            {-1, -1, -1, -1, -1, -1,  0,  0},
            {-1, -1, -1, -1, -1, -1, -1,  0}
        };

        return _mm256_load_si256(
            reinterpret_cast<const __m256i*>(&table[n & (ALGO_AVX2_LANES - 1)][0]));
    }


    // ----------------------------------------------------------------------
    //  Gaussian kernel, VERBATIM from the scalar translation unit.
    //
    //  Built in HighPrecType and narrowed once. It runs once per lobe - three times
    //  per halation call - so its width is free, and deriving the taps identically
    //  to the scalar build is what makes the two comparable: any difference between
    //  the paths must then come from the convolution, not from the kernel.
    //
    //  The 1/(sigma sqrt(2 pi)) factor is omitted because the kernel is renormalised
    //  to unit sum below and the constant cancels. Renormalising rather than trusting
    //  the closed form matters because the kernel is TRUNCATED: the tail beyond the
    //  cutoff is discarded, so the analytic factor would leave the sum slightly under
    //  one and darken the image by a fraction of a per cent - a brightness error
    //  rather than the intended shape error.
    // ----------------------------------------------------------------------
    inline int32_t buildGaussianKernel (const AlgoType sigmaPx,
                                        AlgoType taps[ALGO_BLUR_MAX_TAPS]) noexcept
    {
        int32_t half = static_cast<int32_t>(
            std::ceil(static_cast<HighPrecType>(sigmaPx * ALGO_BLUR_SIGMA_CUTOFF)));

        half = MAX_VALUE(half, static_cast<int32_t>(1));
        half = MIN_VALUE(half, ALGO_BLUR_MAX_HALF_TAPS);

        const HighPrecType inv2s2 = 1.0 / (2.0 * static_cast<HighPrecType>(sigmaPx)
                                              * static_cast<HighPrecType>(sigmaPx));

        HighPrecType sum = 0.0;

        for (int32_t t = -half; t <= half; t++)
        {
            const HighPrecType d = static_cast<HighPrecType>(t);
            const HighPrecType w = std::exp(-(d * d) * inv2s2);

            taps[t + half] = static_cast<AlgoType>(w);
            sum           += w;
        }

        if (sum > 0.0)
        {
            const AlgoType inv = static_cast<AlgoType>(1.0 / sum);
            const int32_t  n   = 2 * half + 1;

            for (int32_t t = 0; t < n; t++)
                taps[t] *= inv;
        }

        return half;
    }


    // Wrap an index into [0, n), VERBATIM from the scalar unit. Two comparisons
    // rather than a modulo: the offset is always within one period of the range.
    inline int32_t wrapIndex (int32_t i, const int32_t n) noexcept
    {
        while (i < 0)  i += n;
        while (i >= n) i -= n;
        return i;
    }

    // ----------------------------------------------------------------------
    //  PYRAMID BLUR - the widest lobes are done at reduced resolution.
    //
    //  WHY THIS IS NOT A SPEED-FOR-ACCURACY TRADE.
    //
    //  The direct kernel is clamped at ALGO_BLUR_MAX_HALF_TAPS. Measured across the
    //  database at HD, the widest halation lobe asks for sigma 33.9 px, which needs
    //  +/-136 taps at the 4.0 cutoff and gets +/-64 - so it is already truncated at
    //  1.9 sigma, and 32 PER CENT of all active lobes are in that state. Truncating a
    //  Gaussian at 1.9 sigma is a visible shape error, not a rounding one.
    //
    //  Blurring at 1/k resolution gives sigma/k, which fits a COMPLETE kernel. So for
    //  exactly the lobes that currently suffer most, the pyramid is more faithful to
    //  the Gaussian than the direct path it replaces, and far cheaper: the tap count
    //  falls by k and the sample count by k^2.
    //
    //  THE VARIANCE COMPENSATION, WHICH IS THE PART THAT IS EASY TO GET WRONG.
    //
    //  Box-averaging k samples is itself a low-pass with variance (k^2 - 1)/12 in
    //  full-resolution pixels. Blurring the decimated plane with sigma_lo contributes
    //  (k * sigma_lo)^2. Variances of successive Gaussians add, so to land on the
    //  requested sigma:
    //
    //      sigma^2 = (k^2 - 1)/12 + (k * sigma_lo)^2
    //      sigma_lo = sqrt(sigma^2 - (k^2 - 1)/12) / k
    //
    //  Omitting that term over-blurs by the width of the decimation filter - about
    //  1.1 px at k=4, which is small but systematic and would show as a halo slightly
    //  too soft on every stock at once.
    // ----------------------------------------------------------------------

    // Smallest sigma worth taking to a pyramid. Below this the direct kernel is short
    // anyway and the two resamples would cost more than they save.
    //  LOWERED 6.0 -> 3.5 (2026-08-11), which only became correct once the
    //  decimation above stopped being scalar. Measured at HD before the change,
    //  the sigma 4-to-6 band cost 28.7 Mcycles per call on the direct path -
    //  MORE THAN TWICE the 12.3 Mcycles a sigma-27 pyramid call cost - because a
    //  41-tap separable kernel is 82 multiply-adds per pixel while the pyramid is
    //  a fixed handful. The old threshold existed to avoid a scalar downsample
    //  that cost more than it saved; with that gone, the crossover moves down to
    //  where the arithmetic says it belongs. Below 3.5 the decimated plane's own
    //  sigma falls under 2 samples and a discrete Gaussian stops resembling one,
    //  so the direct path keeps that region.
    constexpr AlgoType ALGO_BLUR_PYRAMID_MIN_SIGMA = static_cast<AlgoType>(3.5);

    // Sigma aimed for at reduced resolution. 4 px keeps the low-resolution kernel
    // complete at the 4.0 cutoff - 17 taps - while staying well above the two-sample
    // limit where a discrete Gaussian stops resembling one.
    constexpr AlgoType ALGO_BLUR_PYRAMID_TARGET_SIGMA = static_cast<AlgoType>(4.0);

    // Hard ceiling on the decimation factor, so a pathological sigma cannot reduce the
    // plane to a handful of samples and lose the shape entirely.
    constexpr int32_t ALGO_BLUR_PYRAMID_MAX_K = 8;


    // ----------------------------------------------------------------------
    //  NEGLIGIBLE SIGMA: below this the Gaussian IS the identity.
    //
    //  At sigma 0.20 the kernel is three taps and the side weight is
    //  exp(-1/(2*0.04)) = exp(-12.5) = 3.7e-06 of the centre. That is below the
    //  quantisation of a 16-bit channel (1.5e-05), so convolving changes nothing
    //  a consumer can represent - while still costing two full passes over the
    //  plane. Replaced by the copy the result already is.
    //
    //  Deliberately conservative: at 0.25 the side weight would be 3.4e-04,
    //  which IS representable at 12 bits, so the threshold sits where the claim
    //  is unarguable rather than where it merely looks small.
    // ----------------------------------------------------------------------
    constexpr AlgoType ALGO_BLUR_NEGLIGIBLE_SIGMA = static_cast<AlgoType>(0.20);


    // ----------------------------------------------------------------------
    //  ACCUMULATE-MODE OUTPUT: dst = w*res, or dst += w*res.
    //
    //  WHY. AlgoMultiGaussianBlurPlaneWrap used to clear the destination, then
    //  for each lobe blur into a scratch plane and run a separate pass reading
    //  the scratch, reading the destination and writing the destination back.
    //  Counted against the code, a three-lobe call was 16 full-plane traversals
    //  when every lobe fused and 22 when none did. Halation makes NINE such
    //  lobe calls per frame (three lobes on each of three channels), so the
    //  accumulate passes alone were ~11 ms of an HD frame at the measured
    //  ~20 GB/s.
    //
    //  Folding the weight and the accumulation into the blur's own final store
    //  removes the clear pass and every accumulate pass: the lobe result is
    //  never written to memory as an intermediate at all.
    //
    //  TEMPLATED ON THE MODE, NOT BRANCHED ON IT. With ACC a compile-time
    //  constant the dead half vanishes, so the non-accumulating path keeps
    //  exactly the instruction sequence it had. The weight multiply IS applied
    //  in both modes - the public entry point passes 1.0, and multiplying by
    //  1.0f is exact in IEEE-754, so that path is bit-identical to before
    //  rather than merely close.
    // ----------------------------------------------------------------------
    template <bool ACC>
    inline void blurEmit
    (
        AlgoType* RESTRICT p,
        const __m256       res,
        const __m256       vW
    ) noexcept
    {
        if (ACC)
            _mm256_storeu_ps(p, _mm256_fmadd_ps(res, vW, _mm256_loadu_ps(p)));
        else
            _mm256_storeu_ps(p, _mm256_mul_ps(res, vW));

        return;
    }

    template <bool ACC>
    inline void blurEmitMasked
    (
        AlgoType* RESTRICT p,
        const __m256       res,
        const __m256       vW,
        const __m256i      mask
    ) noexcept
    {
        if (ACC)
            _mm256_maskstore_ps(p, mask,
                _mm256_fmadd_ps(res, vW, _mm256_maskload_ps(p, mask)));
        else
            _mm256_maskstore_ps(p, mask, _mm256_mul_ps(res, vW));

        return;
    }

    // Scalar form, for the wrapped edges of the upsample.
    template <bool ACC>
    inline void blurEmitScalar
    (
        AlgoType* RESTRICT p,
        const AlgoType     res,
        const AlgoType     w
    ) noexcept
    {
        if (ACC)
            *p += w * res;
        else
            *p  = w * res;

        return;
    }

    // Whole-plane copy or weighted accumulate, for the negligible-sigma exit.
    template <bool ACC>
    inline void blurEmitPlane
    (
        const AlgoType* RESTRICT pSrc,
        AlgoType* RESTRICT       pDst,
        const int32_t            sizeX,
        const int32_t            sizeY,
        const int32_t            pitch,
        const AlgoType           w
    ) noexcept
    {
        const __m256  vW    = _mm256_set1_ps(w);
        const int32_t vecs  = sizeX / ALGO_AVX2_LANES;
        const int32_t tailN = sizeX - (vecs * ALGO_AVX2_LANES);
        const __m256i vT    = algoTailMask(tailN);

        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pI = pSrc + off;
            AlgoType* RESTRICT       pO = pDst + off;

            int32_t x = 0;

            for (int32_t v = 0; v < vecs; v++, x += ALGO_AVX2_LANES)
                blurEmit<ACC>(pO + x, _mm256_loadu_ps(pI + x), vW);

            if (tailN > 0)
                blurEmitMasked<ACC>(pO + x, _mm256_maskload_ps(pI + x, vT),
                                    vW, vT);
        }

        return;
    }


    // ----------------------------------------------------------------------
    //  FUSED SINGLE-SWEEP PATH: the largest kernel half-width it serves.
    //
    //  WHY FUSE AT ALL. The two-pass form reads the source, writes the whole
    //  intermediate plane, reads it back, and writes the destination: FOUR
    //  full-plane traversals, 33 MB of traffic per HD plane. Instrumented at HD,
    //  the blur core was 251 Mcycles of a ~412 Mcycle frame and the calls
    //  dominating it were sub-pixel ones - sigma 0.2 to 1.1, thirteen calls,
    //  50.6 Mcycles - whose kernels are tiny. Those calls are not doing
    //  arithmetic, they are moving memory: ~11 GB/s achieved, against the 24-27
    //  GB/s this machine can stream. Halving the traffic is the only lever that
    //  touches them.
    //
    //  The fused form keeps 2*half+1 horizontally-blurred rows in a rolling
    //  window and emits one output row per source row, so it reads the source
    //  once and writes the destination once: TWO traversals, 16.6 MB.
    //
    //  WHY A LIMIT ON half. The window must live in cache for this to pay - if it
    //  spills to memory the traffic comes straight back. At half = 8 the window is
    //  17 rows, 130 KB at HD width, which fits a typical 256 KB to 1 MB L2. At the
    //  64-tap ceiling it would be 990 KB and the fusion would be pointless. 8 is
    //  chosen because it covers sigma up to 2.0 at the 4.0 cutoff, and every
    //  memory-bound call measured is inside that.
    //
    //  Arithmetic is UNCHANGED - same taps, same order of accumulation per output
    //  sample, same circular boundary. This is a traffic optimisation, not a
    //  numerical one, and the results are expected to match the two-pass path to
    //  the last bit for the same inputs.
    // ----------------------------------------------------------------------
    constexpr int32_t ALGO_BLUR_FUSED_MAX_HALF = 8;


    // ----------------------------------------------------------------------
    //  Horizontal pass for ONE row, wrapped edges, vectorised interior.
    //
    //  Extracted verbatim from the two-pass horizontal loop so both paths run the
    //  same code on the same row rather than two copies that can drift apart. The
    //  interior bounds are passed in because they are frame constants and
    //  recomputing them per row would be pure overhead.
    // ----------------------------------------------------------------------
    inline void blurRowHorizontal
    (
        const AlgoType* RESTRICT pInRow,
        AlgoType* RESTRICT       pOutRow,
        const int32_t            sizeX,
        const int32_t            half,
        const int32_t            n,
        const AlgoType* RESTRICT taps,
        const int32_t            hiStart,
        const int32_t            hiVecs,
        const int32_t            hiTail,
        const __m256i            vHiTail
    ) noexcept
    {
        // --- left edge, wrapped, scalar ---
        for (int32_t x = 0; x < hiStart; x++)
        {
            AlgoType acc = ALGO_ZERO;

            for (int32_t t = -half; t <= half; t++)
                acc += taps[t + half] * pInRow[wrapIndex(x + t, sizeX)];

            pOutRow[x] = acc;
        }

        // --- interior, no wrap, vectorised ---
        int32_t x = hiStart;

        for (int32_t v = 0; v < hiVecs; v++, x += ALGO_AVX2_LANES)
        {
            __m256 acc = _mm256_setzero_ps();

            const AlgoType* RESTRICT pw = pInRow + x - half;

            for (int32_t k = 0; k < n; k++)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(pw + k),
                                      _mm256_broadcast_ss(&taps[k]), acc);

            _mm256_storeu_ps(pOutRow + x, acc);
        }

        if (hiTail > 0)
        {
            __m256 acc = _mm256_setzero_ps();

            const AlgoType* RESTRICT pw = pInRow + x - half;

            for (int32_t k = 0; k < n; k++)
                acc = _mm256_fmadd_ps(_mm256_maskload_ps(pw + k, vHiTail),
                                      _mm256_broadcast_ss(&taps[k]), acc);

            _mm256_maskstore_ps(pOutRow + x, vHiTail, acc);

            x += hiTail;
        }

        // --- right edge, wrapped, scalar ---
        for (; x < sizeX; x++)
        {
            AlgoType acc = ALGO_ZERO;

            for (int32_t t = -half; t <= half; t++)
                acc += taps[t + half] * pInRow[wrapIndex(x + t, sizeX)];

            pOutRow[x] = acc;
        }

        return;
    }


    // ----------------------------------------------------------------------
    //  Box-average a plane down by k in both axes, with a circular boundary.
    //
    //  Box rather than a Gaussian pre-filter because its variance is exactly known -
    //  see the compensation above - and because at these ratios the difference between
    //  the two is far below the truncation error being removed.
    // ----------------------------------------------------------------------
    void pyramidDownsample
    (
        const AlgoType* RESTRICT pSrc,
        const int32_t            sizeX,
        const int32_t            sizeY,
        const int32_t            pitch,
        AlgoType* RESTRICT       pLo,
        const int32_t            loW,
        const int32_t            loH,
        const int32_t            loPitch,
        const int32_t            k
    ) noexcept
    {
        const AlgoType inv = ALGO_ONE / static_cast<AlgoType>(k * k);

        // ------------------------------------------------------------------
        //  TWO PASSES PER OUTPUT ROW, NOT ONE k*k GATHER PER OUTPUT PIXEL.
        //
        //  The original form walked k*k source samples per output pixel and called
        //  wrapIndex on every one of them - two branches per sample, and no
        //  vectorisation possible because consecutive output pixels read
        //  non-overlapping strided blocks.
        //
        //  Measured consequence: the whole blur was 314 Mcycles of a ~400 Mcycle
        //  HD frame, and this function was the reason the pyramid threshold had to
        //  sit at sigma 6 - below that, decimating cost more than it saved, which
        //  left the expensive 4-to-6 sigma band on the direct path with a 41-tap
        //  kernel.
        //
        //  Restructured as: accumulate k source rows into the output row (fully
        //  contiguous, vectorised, wrap only on the row index), then decimate that
        //  accumulation horizontally (one contiguous sweep, 1/k of the samples).
        //  Same arithmetic, same box average, same circular boundary - the result
        //  is unchanged, which is why this is a pure win and not a trade.
        //
        //  The vertical accumulation writes into the DESTINATION row and reads it
        //  back, using it as the accumulator: loW*k <= sizeX + k, and the row is
        //  loPitch wide, so there is room. That avoids needing a scratch row this
        //  function has no way to allocate.
        // ------------------------------------------------------------------
        const int32_t accW = loW * k;   // samples the horizontal pass will consume

        for (int32_t ly = 0; ly < loH; ly++)
        {
            AlgoType* RESTRICT pOut =
                pLo + static_cast<std::ptrdiff_t>(ly) * loPitch;

            // --------------------------------------------------------------
            //  Pass 1: sum k source rows.
            //
            //  Only the FIRST accW samples are needed, and the last block may run
            //  past the plane, so the tail is handled scalar with the wrap. The
            //  interior - which is all but at most k samples - is contiguous and
            //  vectorises.
            // --------------------------------------------------------------
            const int32_t inner = MIN_VALUE(accW, sizeX);

            const int32_t vecEnd = (inner / ALGO_AVX2_LANES) * ALGO_AVX2_LANES;

            for (int32_t dy = 0; dy < k; dy++)
            {
                // Wrapped, to match the circular convolution the direct path uses.
                const int32_t sy = wrapIndex(ly * k + dy, sizeY);

                const AlgoType* RESTRICT pRow =
                    pSrc + static_cast<std::ptrdiff_t>(sy) * pitch;

                if (0 == dy)
                {
                    // First row initialises rather than accumulates, so the row does
                    // not have to be zeroed first.
                    int32_t x = 0;

                    for (; x < vecEnd; x += ALGO_AVX2_LANES)
                        _mm256_storeu_ps(pOut + x, _mm256_loadu_ps(pRow + x));

                    for (; x < inner; x++)
                        pOut[x] = pRow[x];

                    // Samples past the right edge wrap round.
                    for (int32_t x2 = inner; x2 < accW; x2++)
                        pOut[x2] = pRow[wrapIndex(x2, sizeX)];
                }
                else
                {
                    int32_t x = 0;

                    for (; x < vecEnd; x += ALGO_AVX2_LANES)
                        _mm256_storeu_ps(pOut + x,
                            _mm256_add_ps(_mm256_loadu_ps(pOut + x),
                                          _mm256_loadu_ps(pRow + x)));

                    for (; x < inner; x++)
                        pOut[x] += pRow[x];

                    for (int32_t x2 = inner; x2 < accW; x2++)
                        pOut[x2] += pRow[wrapIndex(x2, sizeX)];
                }
            }

            // --------------------------------------------------------------
            //  Pass 2: decimate horizontally, in place, front to back.
            //
            //  Output lx reads samples [lx*k, lx*k+k) and writes index lx, and
            //  lx <= lx*k for every k >= 1, so the write never clobbers a sample a
            //  later output still needs. In place is therefore safe and no second
            //  buffer is required.
            // --------------------------------------------------------------
            for (int32_t lx = 0; lx < loW; lx++)
            {
                const AlgoType* RESTRICT pBlk = pOut + lx * k;

                AlgoType acc = ALGO_ZERO;

                for (int32_t dx = 0; dx < k; dx++)
                    acc += pBlk[dx];

                pOut[lx] = acc * inv;
            }
        }

        return;
    }


    // ----------------------------------------------------------------------
    //  Bilinear upsample from the reduced plane back to full resolution.
    //
    //  CELL-CENTRED, not corner-aligned, which is the counterpart of the box
    //  downsample: low-resolution sample lx represents full-resolution pixels
    //  [lx*k, lx*k + k), whose centre is at lx*k + (k-1)/2. Using a corner-aligned
    //  mapping here instead - as AlgoBilinearUpsample does, correctly, for the coating
    //  field - would shift the halo by half a low-resolution cell, which at k=4 is two
    //  full-resolution pixels of asymmetry.
    //
    //  The source index is wrapped, so the periodic boundary survives the round trip.
    // ----------------------------------------------------------------------
    template <bool ACC>
    void pyramidUpsample
    (
        const AlgoType* RESTRICT pLo,
        const int32_t            loW,
        const int32_t            loH,
        const int32_t            loPitch,
        AlgoType* RESTRICT       pDst,
        const int32_t            sizeX,
        const int32_t            sizeY,
        const int32_t            pitch,
        const int32_t            k,
        const AlgoType           w
    ) noexcept
    {
        // ------------------------------------------------------------------
        //  CELL-CENTRED bilinear upsample, straightforward form.
        //
        //  Low-resolution sample lx represents full-resolution pixels [lx*k, lx*k + k),
        //  whose centre sits at lx*k + (k-1)/2. Corner alignment instead - which is what
        //  AlgoBilinearUpsample does, correctly, for the coating field - would shift the
        //  halo by half a cell, two full-resolution pixels at k=4.
        //
        //  TWO ATTEMPTS TO MAKE THIS CLEVERER WERE BOTH SLOWER, MEASURED:
        //
        //    per-pixel floor and wrap (this version)        978 ms frame
        //    phase table indexed by x % k and x / k        1028 ms   two integer divides
        //    phase table walked cell-major                 1060 ms   worse still
        //
        //  The reasoning behind both was sound - the weight pattern really does repeat
        //  with period k, so the addressing really is redundant - and both lost anyway.
        //  A float floor plus two compare-based wraps is simply cheaper than the integer
        //  arithmetic or the extra loop structure needed to avoid them, and the compiler
        //  keeps this form in registers.
        //
        //  Left in the simple form deliberately, with the numbers recorded so nobody
        //  repeats the experiment expecting a different answer.
        // ------------------------------------------------------------------
        const AlgoType invK   = ALGO_ONE / static_cast<AlgoType>(k);
        const AlgoType centre = static_cast<AlgoType>(k - 1) * static_cast<AlgoType>(0.5);

        // ------------------------------------------------------------------
        //  Interior bounds, HOISTED out of the row loop - 2026-08-11.
        //
        //  These are frame constants: they depend only on centre, loW and k, none
        //  of which vary with y. They were being recomputed on every row, in
        //  HighPrecType, with a ceil and a floor - 1080 times per call at HD for
        //  a value that never changes. Two defects in one: a rule D1 violation
        //  (HighPrecType in a repeated computation) and a plain loop-invariant
        //  that should never have been inside the loop.
        //
        //  HighPrecType is retained HERE and only here: this runs once per call,
        //  it is exactly the "setup-time scalar" D1 sanctions, and the ceil/floor
        //  must land on the correct integer even when centre is a value like
        //  3.4999999 - which is precisely the case float would get wrong.
        //
        //  Solving for the pixels whose two-tap window lies inside the row:
        //    fx >= 0        -> x >= centre
        //    fx <= loW - 2  -> x <= centre + (loW - 2) * k
        // ------------------------------------------------------------------
        const int32_t xLoHoist = MIN_VALUE(
            static_cast<int32_t>(std::ceil(static_cast<HighPrecType>(centre))),
            sizeX);

        const int32_t xHiRaw = static_cast<int32_t>(
            std::floor(static_cast<HighPrecType>(centre)
                       + static_cast<HighPrecType>(loW - 2)
                         * static_cast<HighPrecType>(k)));

        const int32_t xHiHoist = CLAMP_VALUE(xHiRaw + 1, xLoHoist, sizeX);

        for (int32_t y = 0; y < sizeY; y++)
        {
            const AlgoType fy =
                (static_cast<AlgoType>(y) - centre) * invK;

            const int32_t y0i = static_cast<int32_t>(std::floor(fy));

            const AlgoType wy = fy - static_cast<AlgoType>(y0i);

            const AlgoType* RESTRICT pTop =
                pLo + static_cast<std::ptrdiff_t>(wrapIndex(y0i,     loH)) * loPitch;
            const AlgoType* RESTRICT pBot =
                pLo + static_cast<std::ptrdiff_t>(wrapIndex(y0i + 1, loH)) * loPitch;

            AlgoType* RESTRICT pOut = pDst + static_cast<std::ptrdiff_t>(y) * pitch;

            // ==============================================================
            //  VECTORISED INTERIOR, 2026-08-11.
            //
            //  This loop was entirely scalar: two std::floor calls and two
            //  wrapIndex calls per FULL-RESOLUTION pixel, which at HD is 2.07
            //  million iterations per pyramid call, three such calls per frame.
            //  It was the last scalar loop left in the blur core and it was
            //  hiding behind the pyramid's own success - the path was measured
            //  as a whole and looked cheap next to the direct kernel it
            //  replaced, so nobody looked inside it.
            //
            //  The wrap only bites where the source index leaves [0, loW-2].
            //  Everywhere else xb is simply xa+1, so the interior needs no
            //  wrapIndex at all and the two source samples come from one pair
            //  of gathers.
            //
            //  GATHER rather than load: consecutive output pixels do NOT read
            //  consecutive source samples - x0i advances by one every k output
            //  pixels - so the addresses are a repeating staircase, which is
            //  exactly the access pattern gather exists for.
            // ==============================================================

            // First and last output pixel whose 2-tap window is inside the
            // low-resolution row. Solving x0i >= 0 and x0i + 1 <= loW - 1:
            //   fx >= 0            -> x >= centre
            //   fx <= loW - 2      -> x <= centre + (loW - 2) * k
            // Interior bounds: HOISTED to the enclosing scope, see below. Kept as
            // references here only so the code reads in place.
            const int32_t xLo = xLoHoist;
            const int32_t xHi = xHiHoist;

            // --- left edge, wrapped, scalar ---
            for (int32_t x = 0; x < xLo; x++)
            {
                const AlgoType fx = (static_cast<AlgoType>(x) - centre) * invK;

                const int32_t x0i = static_cast<int32_t>(std::floor(fx));

                const AlgoType wx = fx - static_cast<AlgoType>(x0i);

                const int32_t xa = wrapIndex(x0i,     loW);
                const int32_t xb = wrapIndex(x0i + 1, loW);

                const AlgoType top = pTop[xa] + (pTop[xb] - pTop[xa]) * wx;
                const AlgoType bot = pBot[xa] + (pBot[xb] - pBot[xa]) * wx;

                blurEmitScalar<ACC>(pOut + x, top + (bot - top) * wy, w);
            }

            // --- interior, no wrap, vectorised ---
            {
                const __m256 vCentre = _mm256_set1_ps(centre);
                const __m256 vInvK   = _mm256_set1_ps(invK);
                const __m256 vWy     = _mm256_set1_ps(wy);
                const __m256 vW      = _mm256_set1_ps(w);

                const __m256 vLane =
                    _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f,
                                   4.0f, 5.0f, 6.0f, 7.0f);

                const int32_t inner  = xHi - xLo;
                const int32_t vecs   = inner / ALGO_AVX2_LANES;
                const int32_t vTailN = inner - (vecs * ALGO_AVX2_LANES);

                const __m256i vTailM = algoTailMask(vTailN);

                int32_t x = xLo;

                for (int32_t v = 0; v <= vecs; v++)
                {
                    // The last iteration handles the masked remainder; skip it
                    // when the interior divided evenly.
                    const bool isTail = (v == vecs);

                    if (isTail && (0 == vTailN))
                        break;

                    // fx per lane, then its integer part and fraction. floor is
                    // exact here and cvttps is the same value because fx is
                    // non-negative throughout the interior by construction.
                    const __m256 fx = _mm256_mul_ps(
                        _mm256_sub_ps(
                            _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x)),
                                          vLane),
                            vCentre),
                        vInvK);

                    const __m256i i0 = _mm256_cvttps_epi32(fx);

                    const __m256 wx =
                        _mm256_sub_ps(fx, _mm256_cvtepi32_ps(i0));

                    // Four gathers: the two horizontal neighbours on each of the
                    // two source rows. xb is i0+1 with no wrap, which is what
                    // confines this block to the interior.
                    const __m256 ta = _mm256_i32gather_ps(pTop,     i0, 4);
                    const __m256 tb = _mm256_i32gather_ps(pTop + 1, i0, 4);
                    const __m256 ba = _mm256_i32gather_ps(pBot,     i0, 4);
                    const __m256 bb = _mm256_i32gather_ps(pBot + 1, i0, 4);

                    // Two horizontal lerps, then one vertical lerp - identical
                    // arithmetic and identical order to the scalar form above.
                    const __m256 top =
                        _mm256_fmadd_ps(_mm256_sub_ps(tb, ta), wx, ta);
                    const __m256 bot =
                        _mm256_fmadd_ps(_mm256_sub_ps(bb, ba), wx, ba);

                    const __m256 out =
                        _mm256_fmadd_ps(_mm256_sub_ps(bot, top), vWy, top);

                    if (isTail)
                        blurEmitMasked<ACC>(pOut + x, out, vW, vTailM);
                    else
                        blurEmit<ACC>(pOut + x, out, vW);

                    x += ALGO_AVX2_LANES;
                }
            }

            // --- right edge, wrapped, scalar ---
            for (int32_t x = xHi; x < sizeX; x++)
            {
                const AlgoType fx = (static_cast<AlgoType>(x) - centre) * invK;

                const int32_t x0i = static_cast<int32_t>(std::floor(fx));

                const AlgoType wx = fx - static_cast<AlgoType>(x0i);

                const int32_t xa = wrapIndex(x0i,     loW);
                const int32_t xb = wrapIndex(x0i + 1, loW);

                const AlgoType top = pTop[xa] + (pTop[xb] - pTop[xa]) * wx;
                const AlgoType bot = pBot[xa] + (pBot[xb] - pBot[xa]) * wx;

                blurEmitScalar<ACC>(pOut + x, top + (bot - top) * wy, w);
            }
        }

        return;
    }
}


// ---------------------------------------------------------------------------
//  AlgoCopyPlane
//
//  Left as memcpy, deliberately. A pure copy is bound by memory bandwidth, so
//  eight-wide stores buy nothing measurable, and the library memcpy already selects
//  the best available sequence for the size and alignment at hand - including
//  non-temporal stores for large blocks, which hand-written vector code would have
//  to reimplement to match.
// ---------------------------------------------------------------------------
void AlgoCopyPlane
(
    const AlgoType* RESTRICT pSrc,
    AlgoType* RESTRICT       pDst,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch
) noexcept
{
    const std::size_t rowBytes = static_cast<std::size_t>(sizeX) * sizeof(AlgoType);

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;
        std::memcpy(pDst + off, pSrc + off, rowBytes);
    }

    return;
}


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
) noexcept
{
    AlgoCopyPlane(pSrcR, pDstR, sizeX, sizeY, pitch);
    AlgoCopyPlane(pSrcG, pDstG, sizeX, sizeY, pitch);
    AlgoCopyPlane(pSrcB, pDstB, sizeX, sizeY, pitch);
    return;
}


// ---------------------------------------------------------------------------
//  AlgoGaussianBlurPlaneWrap  --  the hot loop of the whole engine
//
//  THE TWO PASSES VECTORISE VERY DIFFERENTLY, AND THAT IS THE WHOLE DESIGN.
//
//  VERTICAL is the easy one, which is the opposite of what the scalar comments
//  suggest. Each output row sums whole rows of the intermediate, so x is unit-stride
//  and eight neighbouring pixels share one broadcast tap. The circular wrap is on
//  Y, and it is resolved once per tap OUTSIDE the x loop - so the inner loop is a
//  clean stream of aligned loads and FMAs with no addressing arithmetic at all. The
//  pass that looked worst scalar, because every tap touched a different cache line,
//  becomes the best behaved.
//
//  HORIZONTAL is the awkward one, because the wrap is on the same axis being
//  vectorised. Wrapping inside the vector loop would need a gather per tap and
//  would be slower than scalar. But the wrap only BITES within one kernel
//  half-width of each end, so the row splits into three regions:
//
//      [0, half)                 scalar, wraps off the left edge
//      [half, sizeX - half)      vector, no wrap possible
//      [sizeX - half, sizeX)     scalar, wraps off the right edge
//
//  The interior is the overwhelming majority for any sane kernel and needs only
//  UNALIGNED loads - a shifted window of the same row - which on this class of core
//  costs nothing extra when the data is in cache.
//
//  ACCUMULATION ORDER IS PRESERVED IN BOTH PASSES. Lane i computes exactly the
//  output the scalar loop computes for that pixel, summing taps in the same
//  ascending order. So the only numerical difference is FMA's single rounding in
//  place of two - which makes the vector result very slightly MORE accurate, not
//  less. There is no reassociation anywhere in this function.
// ---------------------------------------------------------------------------
namespace
{
    // ----------------------------------------------------------------------
    //  The blur, templated on output mode. See blurEmit for why.
    //
    //  ACC == false, w == 1  is the historic behaviour, bit for bit.
    //  ACC == true           accumulates w * result into the destination, which
    //                        is what lets the multi-lobe caller drop its clear
    //                        pass and all of its per-lobe accumulate passes.
    // ----------------------------------------------------------------------
    template <bool ACC>
    void blurPlaneWrapT
    (
        const AlgoType* RESTRICT pSrc,
        AlgoType* RESTRICT       pDst,
        AlgoType* RESTRICT       pScratch,
        const int32_t            sizeX,
        const int32_t            sizeY,
        const int32_t            pitch,
        const AlgoType           sigmaPx,
        const AlgoType           wAcc
    ) noexcept
{
    const __m256 vWAcc = _mm256_set1_ps(wAcc);

    // A Gaussian of zero or negative width is the identity - and so, to any
    // representable precision, is one narrower than ALGO_BLUR_NEGLIGIBLE_SIGMA.
    // See that constant for the arithmetic; the saving is one of the two passes.
    if (sigmaPx < ALGO_BLUR_NEGLIGIBLE_SIGMA)
    {
        blurEmitPlane<ACC>(pSrc, pDst, sizeX, sizeY, pitch, wAcc);
        return;
    }

    // ----------------------------------------------------------------------
    //  Wide lobe: do it at reduced resolution.
    //
    //  Everything below this block is the direct path, unchanged. The decision is made
    //  once per call on a frame constant, so the pyramid costs nothing when it does not
    //  apply - and it applies to exactly the lobes the direct path serves worst.
    // ----------------------------------------------------------------------
    if (sigmaPx >= ALGO_BLUR_PYRAMID_MIN_SIGMA)
    {
        // Decimation factor from the requested sigma, so the reduced-resolution sigma
        // lands near the target whatever was asked for.
        int32_t k = static_cast<int32_t>(
            sigmaPx / ALGO_BLUR_PYRAMID_TARGET_SIGMA + static_cast<AlgoType>(0.5));

        // Clamped to at least 2: a k of 1 would decimate by nothing and simply pay
        // for two extra full-resolution passes. With the threshold at 3.5 the
        // computed k IS 1 for the bottom of the band, so this clamp is now load
        // bearing rather than defensive - k=2 on a sigma-3.5 lobe leaves a
        // reduced-resolution sigma of 1.7, which the compensation below corrects
        // for exactly and which is still above the two-sample floor.
        k = CLAMP_VALUE(k, 2, ALGO_BLUR_PYRAMID_MAX_K);

        const int32_t loW = (sizeX + k - 1) / k;
        const int32_t loH = (sizeY + k - 1) / k;

        // Reduced-resolution rows padded to the vector width, so the low-resolution
        // blur gets the same aligned row starts the full-resolution one has.
        const int32_t loPitch =
            ((loW + ALGO_AVX2_LANES - 1) / ALGO_AVX2_LANES) * ALGO_AVX2_LANES;

        // THREE reduced planes are carved out of the single full-resolution scratch:
        // the decimated source, the separable intermediate, and the blurred result.
        // They fit whenever 3/k^2 <= 1, which holds for every k this path allows - but
        // it is CHECKED rather than trusted, because the fallback is merely slower
        // while an overrun would corrupt the arena.
        const std::ptrdiff_t loPlane =
            static_cast<std::ptrdiff_t>(loPitch) * static_cast<std::ptrdiff_t>(loH);

        const std::ptrdiff_t haveScratch =
            static_cast<std::ptrdiff_t>(pitch) * static_cast<std::ptrdiff_t>(sizeY);

        if ((3 * loPlane) <= haveScratch && loW >= 4 && loH >= 4)
        {
            // Variance compensation. The box decimation is itself a low-pass of
            // variance (k^2 - 1)/12 in full-resolution pixels, so the reduced-
            // resolution Gaussian must be narrower by exactly that much.
            const AlgoType boxVar =
                static_cast<AlgoType>(k * k - 1) / static_cast<AlgoType>(12);

            const AlgoType targetVar = sigmaPx * sigmaPx - boxVar;

            const AlgoType sigmaLo = (targetVar > ALGO_ZERO)
                ? (static_cast<AlgoType>(std::sqrt(
                       static_cast<HighPrecType>(targetVar)))
                   / static_cast<AlgoType>(k))
                : ALGO_ZERO;

            AlgoType* RESTRICT loSrc = pScratch;
            AlgoType* RESTRICT loTmp = pScratch + loPlane;
            AlgoType* RESTRICT loOut = pScratch + 2 * loPlane;

            pyramidDownsample(pSrc, sizeX, sizeY, pitch,
                              loSrc, loW, loH, loPitch, k);

            // The reduced-resolution blur is the DIRECT path, recursively - and at this
            // sigma its kernel is complete, which is the whole point.
            // The reduced-resolution blur NEVER accumulates: it produces the
            // low-resolution lobe in its own plane, and the accumulation happens
            // once, in the upsample that writes the full-resolution destination.
            blurPlaneWrapT<false>(loSrc, loOut, loTmp,
                                  loW, loH, loPitch, sigmaLo, ALGO_ONE);

            pyramidUpsample<ACC>(loOut, loW, loH, loPitch,
                                 pDst, sizeX, sizeY, pitch, k, wAcc);

            return;
        }

        // Fell through: the scratch could not hold three reduced planes, or the plane
        // is too small to decimate meaningfully. The direct path below is correct in
        // every case, only slower.
    }

    AVX2_ALIGN AlgoType taps[ALGO_BLUR_MAX_TAPS];

    const int32_t half = buildGaussianKernel(sigmaPx, taps);
    const int32_t n    = 2 * half + 1;

    // ----------------------------------------------------------------------
    //  Interior bounds for the horizontal pass, shared by both paths below.
    //
    //  When the kernel is wide relative to the plane - a big halation lobe on a
    //  small render - the interior can be empty, and then every pixel takes the
    //  scalar wrapped path. Clamped rather than assumed non-empty, because a
    //  negative count would run the vector loop backwards.
    // ----------------------------------------------------------------------
    const int32_t hiStart = MIN_VALUE(half, sizeX);
    const int32_t hiEnd   = MAX_VALUE(sizeX - half, hiStart);

    const int32_t hiVecs = (hiEnd - hiStart) / ALGO_AVX2_LANES;
    const int32_t hiTail = (hiEnd - hiStart) - (hiVecs * ALGO_AVX2_LANES);

    const __m256i vHiTail = algoTailMask(hiTail);

    // ----------------------------------------------------------------------
    //  FUSED SINGLE-SWEEP PATH.
    //
    //  Chosen when the window fits cache and the plane is tall enough to hold
    //  one. Halves the memory traffic; see ALGO_BLUR_FUSED_MAX_HALF for why that
    //  is the lever that matters here.
    //
    //  The window occupies the first (2*half+1) rows of the caller's scratch
    //  plane, which is exactly what the two-pass path would have used for its
    //  whole intermediate - so no new memory, and less of it touched.
    //
    //  The tallness test is 2*half+1 <= sizeY, not a comfort margin: the window
    //  holds distinct source rows, and if the plane had fewer rows than the
    //  window the same row would occupy two slots and be counted twice. The
    //  two-pass path handles that case correctly, so it takes it.
    // ----------------------------------------------------------------------
    const int32_t win = n;   // 2*half + 1 rows

    if ((half <= ALGO_BLUR_FUSED_MAX_HALF) && (win <= sizeY))
    {
        // Row slots inside the scratch plane.
        AlgoType* RESTRICT rows[2 * ALGO_BLUR_FUSED_MAX_HALF + 1];

        for (int32_t s = 0; s < win; s++)
            rows[s] = pScratch + static_cast<std::ptrdiff_t>(s) * pitch;

        // ------------------------------------------------------------------
        //  Prime the window with source rows -half .. +half, wrapped.
        //
        //  Slot s holds source row (s - half), so after priming slot 0 is the
        //  topmost row the first output needs.
        // ------------------------------------------------------------------
        for (int32_t s = 0; s < win; s++)
        {
            const int32_t sy = wrapIndex(s - half, sizeY);

            blurRowHorizontal(pSrc + static_cast<std::ptrdiff_t>(sy) * pitch,
                              rows[s], sizeX, half, n, taps,
                              hiStart, hiVecs, hiTail, vHiTail);
        }

        // Slot holding the OLDEST row in the window, i.e. row y-half. Rotates by
        // one per output row, which is what makes this O(1) per row rather than
        // a refill.
        int32_t base = 0;

        const int32_t vecCount = sizeX / ALGO_AVX2_LANES;
        const int32_t tailN    = sizeX - (vecCount * ALGO_AVX2_LANES);

        const __m256i vTail = algoTailMask(tailN);

        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pOutRow =
                pDst + static_cast<std::ptrdiff_t>(y) * pitch;

            // --------------------------------------------------------------
            //  Vertical accumulation across the window.
            //
            //  Tap t applies to source row y - half + t, which sits in slot
            //  (base + t) mod win. No wrapIndex here: the wrap was resolved when
            //  the row was loaded into the window.
            // --------------------------------------------------------------
            //  Slot pointers and tap weights resolved into TAP ORDER once per
            //  output row, not once per tap per vector.
            //
            //  The rotation means tap t lives in slot (base+t) mod win, and
            //  computing that inside the x loop put a compare and a select on the
            //  critical path of every FMA - measured, that overhead exceeded the
            //  traffic this path was written to save. Hoisted, the inner loop is a
            //  straight walk of two small arrays that stay in registers or L1.
            const AlgoType* RESTRICT ordered[2 * ALGO_BLUR_FUSED_MAX_HALF + 1];
            AVX2_ALIGN AlgoType      wTap  [2 * ALGO_BLUR_FUSED_MAX_HALF + 1];

            for (int32_t t = 0; t < win; t++)
            {
                const int32_t s = (base + t < win) ? (base + t) : (base + t - win);

                ordered[t] = rows[s];
                wTap   [t] = taps[t];
            }

            int32_t x = 0;

            for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
            {
                __m256 acc = _mm256_setzero_ps();

                for (int32_t t = 0; t < win; t++)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(ordered[t] + x),
                                          _mm256_broadcast_ss(&wTap[t]), acc);

                blurEmit<ACC>(pOutRow + x, acc, vWAcc);
            }

            if (tailN > 0)
            {
                __m256 acc = _mm256_setzero_ps();

                for (int32_t t = 0; t < win; t++)
                    acc = _mm256_fmadd_ps(
                              _mm256_maskload_ps(ordered[t] + x, vTail),
                              _mm256_broadcast_ss(&wTap[t]), acc);

                blurEmitMasked<ACC>(pOutRow + x, acc, vWAcc, vTail);
            }

            // --------------------------------------------------------------
            //  Slide the window: the oldest row is finished with, so overwrite it
            //  with the row the NEXT output will need at the bottom of its window.
            //
            //  Skipped on the final row, where there is no next output - not for
            //  speed but because the row it would load is already in the window
            //  and reloading it is pure waste.
            // --------------------------------------------------------------
            if (y + 1 < sizeY)
            {
                const int32_t sy = wrapIndex(y + 1 + half, sizeY);

                blurRowHorizontal(pSrc + static_cast<std::ptrdiff_t>(sy) * pitch,
                                  rows[base], sizeX, half, n, taps,
                                  hiStart, hiVecs, hiTail, vHiTail);

                base = (base + 1 < win) ? (base + 1) : 0;
            }
        }

        return;
    }

    // ----------------------------------------------------------------------
    //  TWO-PASS PATH: wide kernels, and any plane too short for a window.
    //
    //  Horizontal pass: pSrc -> pScratch.
    // ----------------------------------------------------------------------

    for (int32_t y = 0; y < sizeY; y++)
    {
        const AlgoType* RESTRICT pInRow  =
            pSrc + static_cast<std::ptrdiff_t>(y) * pitch;
        AlgoType* RESTRICT       pOutRow =
            pScratch + static_cast<std::ptrdiff_t>(y) * pitch;

        // --- left edge, wrapped, scalar ---
        for (int32_t x = 0; x < hiStart; x++)
        {
            AlgoType acc = ALGO_ZERO;

            for (int32_t t = -half; t <= half; t++)
                acc += taps[t + half] * pInRow[wrapIndex(x + t, sizeX)];

            pOutRow[x] = acc;
        }

        // --- interior, no wrap, vectorised ---
        int32_t x = hiStart;

        for (int32_t v = 0; v < hiVecs; v++, x += ALGO_AVX2_LANES)
        {
            __m256 acc = _mm256_setzero_ps();

            // Tap t reads the window starting at x - half + (t + half), which is
            // x + t. Unaligned because the window slides by one sample per tap.
            const AlgoType* RESTRICT pw = pInRow + x - half;

            for (int32_t k = 0; k < n; k++)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(pw + k),
                                      _mm256_broadcast_ss(&taps[k]), acc);

            // UNALIGNED store, and this is not laziness. The interior begins at
            // x = half, which is the kernel half-width - an arbitrary integer, not a
            // multiple of eight. An aligned store here faults outright, which is
            // exactly how this was found: test_full dumped core on the first run.
            _mm256_storeu_ps(pOutRow + x, acc);
        }

        if (hiTail > 0)
        {
            __m256 acc = _mm256_setzero_ps();

            const AlgoType* RESTRICT pw = pInRow + x - half;

            // Masked loads, so the tail cannot read past the interior into the
            // region the right-edge scalar loop owns.
            for (int32_t k = 0; k < n; k++)
                acc = _mm256_fmadd_ps(_mm256_maskload_ps(pw + k, vHiTail),
                                      _mm256_broadcast_ss(&taps[k]), acc);

            _mm256_maskstore_ps(pOutRow + x, vHiTail, acc);

            x += hiTail;
        }

        // --- right edge, wrapped, scalar ---
        for (; x < sizeX; x++)
        {
            AlgoType acc = ALGO_ZERO;

            for (int32_t t = -half; t <= half; t++)
                acc += taps[t + half] * pInRow[wrapIndex(x + t, sizeX)];

            pOutRow[x] = acc;
        }
    }

    // ----------------------------------------------------------------------
    //  Vertical pass: pScratch -> pDst.
    //
    //  The wrap is on Y and is hoisted entirely out of the x loop: each tap
    //  contributes one whole row, whose base address is resolved once. What remains
    //  inside is pure streaming FMA.
    // ----------------------------------------------------------------------
    const int32_t vVecs = sizeX / ALGO_AVX2_LANES;
    const int32_t vTailN = sizeX - (vVecs * ALGO_AVX2_LANES);

    const __m256i vTail = algoTailMask(vTailN);

    for (int32_t y = 0; y < sizeY; y++)
    {
        AlgoType* RESTRICT pOutRow = pDst + static_cast<std::ptrdiff_t>(y) * pitch;

        int32_t x = 0;

        for (int32_t v = 0; v < vVecs; v++, x += ALGO_AVX2_LANES)
        {
            __m256 acc = _mm256_setzero_ps();

            for (int32_t k = 0; k < n; k++)
            {
                const int32_t ys = wrapIndex(y + k - half, sizeY);

                const AlgoType* RESTRICT pRow =
                    pScratch + static_cast<std::ptrdiff_t>(ys) * pitch;

                acc = _mm256_fmadd_ps(_mm256_loadu_ps(pRow + x),
                                      _mm256_broadcast_ss(&taps[k]), acc);
            }

            blurEmit<ACC>(pOutRow + x, acc, vWAcc);
        }

        if (vTailN > 0)
        {
            __m256 acc = _mm256_setzero_ps();

            for (int32_t k = 0; k < n; k++)
            {
                const int32_t ys = wrapIndex(y + k - half, sizeY);

                const AlgoType* RESTRICT pRow =
                    pScratch + static_cast<std::ptrdiff_t>(ys) * pitch;

                acc = _mm256_fmadd_ps(_mm256_maskload_ps(pRow + x, vTail),
                                      _mm256_broadcast_ss(&taps[k]), acc);
            }

            blurEmitMasked<ACC>(pOutRow + x, acc, vWAcc, vTail);
        }
    }

    return;
}
}   // anonymous namespace


// ---------------------------------------------------------------------------
//  Public entry point: blur, no accumulation, unit weight.
//
//  One line, and deliberately so - the historic behaviour is now the ACC=false
//  instantiation of the template above, which multiplies by 1.0f. That multiply
//  is exact in IEEE-754, so this path is bit-identical to the code before the
//  accumulate mode was added rather than merely equivalent.
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
) noexcept
{
    blurPlaneWrapT<false>(pSrc, pDst, pScratch,
                          sizeX, sizeY, pitch, sigmaPx, ALGO_ONE);
    return;
}


// ---------------------------------------------------------------------------
//  AlgoMultiGaussianBlurPlaneWrap
//
//  Weighted sum of lobes. The blur itself is delegated per lobe; what is vectorised
//  here is the clear and the accumulate, both pointwise.
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
) noexcept
{
    // Normalising factor, so the operator preserves a flat field whatever scale the
    // caller used. Summed scalar: lobeCount is at most four.
    AlgoType wsum = ALGO_ZERO;
    for (int32_t k = 0; k < lobeCount; k++)
        wsum += weight[k];

    // Degenerate weights: pass the plane through rather than emit zeros, which would
    // silently blacken the frame.
    if (wsum <= ALGO_ZERO)
    {
        AlgoCopyPlane(pSrc, pDst, sizeX, sizeY, pitch);
        return;
    }

    const AlgoType invWsum = ALGO_ONE / wsum;

    // ----------------------------------------------------------------------
    //  ONE PASS PER LOBE, AND NOTHING ELSE.
    //
    //  WHAT THIS REPLACED. The previous form cleared the destination, then for
    //  every lobe blurred into pScratchA and ran a second full pass reading the
    //  scratch, reading the destination and writing it back. Counted against
    //  the code that is 1 + lobeCount*(blur + 3) full-plane traversals: SIXTEEN
    //  for a three-lobe call when every lobe took the fused path, TWENTY-TWO
    //  when none did.
    //
    //  Halation issues NINE lobe calls per frame - three lobes on each of three
    //  colour records - so the clear and accumulate passes alone moved roughly
    //  250 MB per frame at HD, about 11 ms at the measured streaming rate, for
    //  arithmetic that is one FMA per sample.
    //
    //  Now: the FIRST lobe writes w*result straight into the destination, which
    //  is why no clear is needed - it overwrites rather than adds. Every later
    //  lobe accumulates. The lobe result never becomes an intermediate plane, so
    //  pScratchA is now only the blur's own working store.
    //
    //  ORDER OF SUMMATION IS PRESERVED. Lobes are still applied k = 0, 1, 2 ...
    //  and each contributes w[k]*invWsum times its blur, so the floating-point
    //  result is the same sequence of operations as before - the accumulation
    //  simply happens in the store that produced the value rather than in a
    //  separate sweep.
    //
    //  pScratchB IS NOW UNUSED. The parameter stays: the prototype is shared
    //  with the scalar build and must not change, and a caller passing two
    //  distinct scratch planes is not wrong, merely generous.
    // ----------------------------------------------------------------------
    (void)pScratchB;

    for (int32_t k = 0; k < lobeCount; k++)
    {
        const AlgoType w = weight[k] * invWsum;

        if (0 == k)
            blurPlaneWrapT<false>(pSrc, pDst, pScratchA,
                                  sizeX, sizeY, pitch, sigmaPx[k], w);
        else
            blurPlaneWrapT<true>(pSrc, pDst, pScratchA,
                                 sizeX, sizeY, pitch, sigmaPx[k], w);
    }

    return;
}


// ---------------------------------------------------------------------------
//  AlgoPlaneMean
//
//  THE ONE PLACE IN THIS FILE THAT DELIBERATELY DOES NOT COMPUTE IN FLOAT.
//
//  This is a reduction over every sample in the frame - two million at HD, eight at
//  4K - and a flat single-precision accumulator loses its low bits once the running
//  total has grown large relative to each addend. The value sets the black floor of
//  the whole frame at stage 3b, so an error here is a visible lift, not noise.
//
//  The structure keeps both properties:
//
//    - eight float lanes accumulate one row in parallel, so the row is vectorised
//    - the horizontal reduction and the row-to-frame total are in HighPrecType
//
//  Two-level summation also keeps each partial small relative to its addends, which
//  is where a flat accumulator loses precision, and it costs nothing.
//
//  The tail is MASKED rather than allowed to spill into row padding. That matters
//  here more than anywhere else: padding is not part of the image, and letting it
//  into a sum would bias the mean by whatever the allocator happened to leave there
//  - which under the NaN-poison test is a quiet NaN, and the reduction would carry
//  it into every pixel of the frame.
// ---------------------------------------------------------------------------
HighPrecType AlgoPlaneMean
(
    const AlgoType* RESTRICT pSrc,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch
) noexcept
{
    const int32_t vecCount = sizeX / ALGO_AVX2_LANES;
    const int32_t tailN    = sizeX - (vecCount * ALGO_AVX2_LANES);

    const __m256i vTail = algoTailMask(tailN);

    HighPrecType total = 0.0;

    for (int32_t y = 0; y < sizeY; y++)
    {
        const AlgoType* RESTRICT pRow =
            pSrc + static_cast<std::ptrdiff_t>(y) * pitch;

        __m256 acc = _mm256_setzero_ps();

        int32_t x = 0;

        for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(pRow + x));

        if (tailN > 0)
            acc = _mm256_add_ps(acc, _mm256_maskload_ps(pRow + x, vTail));

        // Horizontal reduction, widened. The eight lane totals are each a sum over
        // at most sizeX/8 samples, so they are still small; combining them in double
        // is what stops the FRAME total from losing bits as it grows.
        AVX2_ALIGN AlgoType lane[ALGO_AVX2_LANES];
        _mm256_storeu_ps(lane, acc);

        HighPrecType rowSum = 0.0;

        for (int32_t i = 0; i < ALGO_AVX2_LANES; i++)
            rowSum += static_cast<HighPrecType>(lane[i]);

        total += rowSum;
    }

    const HighPrecType count = static_cast<HighPrecType>(sizeX)
                             * static_cast<HighPrecType>(sizeY);

    return (count > 0.0) ? (total / count) : 0.0;
}


// ---------------------------------------------------------------------------
//  AlgoBilinearUpsample
//
//  Left scalar apart from the degenerate fill, and that is a considered decision
//  rather than an omission. The source index advances by a non-integer step, so
//  eight consecutive destination pixels read eight unrelated source pairs - a
//  gather, not a load. On this class of core a gather of eight 32-bit elements costs
//  more than the eight scalar loads it replaces, so vectorising this would be slower
//  as well as harder to read.
//
//  It is also not hot: this runs on the low-resolution coating field, which is a
//  small fraction of a frame, and profall attributes 1.57 ms to the only stage that
//  leans on it.
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
) noexcept
{
    // A one-sample source has no gradient: fill with that single value.
    if (loW <= 1 || loH <= 1)
    {
        const AlgoType v = pLo[0];

        const __m256 vV = _mm256_set1_ps(v);

        const int32_t vecCount = sizeX / ALGO_AVX2_LANES;
        const int32_t tailN    = sizeX - (vecCount * ALGO_AVX2_LANES);

        const __m256i vTail = algoTailMask(tailN);

        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow = pDst + static_cast<std::ptrdiff_t>(y) * pitch;

            int32_t x = 0;

            for (int32_t vi = 0; vi < vecCount; vi++, x += ALGO_AVX2_LANES)
                _mm256_storeu_ps(pRow + x, vV);

            if (tailN > 0)
                _mm256_maskstore_ps(pRow + x, vTail, vV);
        }
        return;
    }

    // Corner-aligned mapping: destination 0 lands on source 0, and destination
    // (n-1) on source (m-1). Step is therefore (m-1)/(n-1), not m/n.
    const HighPrecType stepX = static_cast<HighPrecType>(loW - 1)
                             / static_cast<HighPrecType>(MAX_VALUE(sizeX - 1, 1));
    const HighPrecType stepY = static_cast<HighPrecType>(loH - 1)
                             / static_cast<HighPrecType>(MAX_VALUE(sizeY - 1, 1));

    for (int32_t y = 0; y < sizeY; y++)
    {
        const HighPrecType sy = static_cast<HighPrecType>(y) * stepY;

        // Left index clamped so y0+1 is always a valid sample; the fractional part
        // then carries the interpolation on the final row.
        int32_t y0 = static_cast<int32_t>(sy);
        y0 = MIN_VALUE(y0, loH - 2);
        y0 = MAX_VALUE(y0, 0);

        const AlgoType fy = static_cast<AlgoType>(sy - static_cast<HighPrecType>(y0));

        const AlgoType* RESTRICT pTop =
            pLo + static_cast<std::ptrdiff_t>(y0) * loPitch;
        const AlgoType* RESTRICT pBot =
            pLo + static_cast<std::ptrdiff_t>(y0 + 1) * loPitch;

        AlgoType* RESTRICT pRow = pDst + static_cast<std::ptrdiff_t>(y) * pitch;

        for (int32_t x = 0; x < sizeX; x++)
        {
            const HighPrecType sx = static_cast<HighPrecType>(x) * stepX;

            int32_t x0 = static_cast<int32_t>(sx);
            x0 = MIN_VALUE(x0, loW - 2);
            x0 = MAX_VALUE(x0, 0);

            const AlgoType fx =
                static_cast<AlgoType>(sx - static_cast<HighPrecType>(x0));

            // Two horizontal interpolations, then one vertical between them.
            const AlgoType top = pTop[x0] + (pTop[x0 + 1] - pTop[x0]) * fx;
            const AlgoType bot = pBot[x0] + (pBot[x0 + 1] - pBot[x0]) * fx;

            pRow[x] = top + (bot - top) * fy;
        }
    }

    return;
}
#endif