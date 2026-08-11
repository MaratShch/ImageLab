#if 0
// ---------------------------------------------------------------------------
//  Algo_15_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: nothing in the pixel loop yet. The weave resample is bilinear with a
//  per-pixel source index, so it is a gather rather than a load; left scalar for the
//  same reason AlgoBilinearUpsample is, and the stage measures under one per cent.
//
//  ALIGNMENT: EVERY IMAGE ACCESS IS UNALIGNED, DELIBERATELY.
//
//  loadu/storeu on all plane data. The arena base comes from the host's pool, whose
//  alignment argument is a HINT - it was seen returning a base 16 mod 32, which faults
//  an aligned 256-bit load. AlgoMemHandler.cpp is SHARED by both flavours and must not
//  carry a vector-path concern, so the vector path assumes nothing about alignment.
//
//  Pipeline stage 15: gate weave and registration instability.
//
//      AlgoWeaveNoise          one sample of unit-variance red noise
//      AlgoStage15_GateWeave   the per-frame translation
//
//  Raw pointers, explicit geometry, no allocation, no mutable state, no validation
//  of inputs.
// ---------------------------------------------------------------------------

#include "AlgoGateWeave.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>
#include "AlgoSeparableBlur.hpp"   // AlgoCopyImage

#include <cmath>   // std::floor, std::sqrt


static_assert(sizeof(AlgoType) == 4,
              "the AVX2 path requires AlgoType to be a 32-bit float");

namespace
{
    // ----------------------------------------------------------------------
    //  Lanes per vector, and the tail mask for the final partial vector of a row.
    //
    //  The active width is not generally a multiple of eight. Masked access leaves the
    //  row padding untouched, which keeps the NaN-poison arena test meaningful.
    // ----------------------------------------------------------------------
    constexpr int32_t ALGO_AVX2_LANES_LOCAL = 8;

    inline __m256i algoTailMaskLocal (const int32_t n) noexcept
    {
        AVX2_ALIGN static const int32_t table[8][8] =
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
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(&table[n & 7][0]));
    }
}


namespace
{
    // ----------------------------------------------------------------------
    //  Smooth step on 0..1, matching the interpolant the variance correction in
    //  AlgoGateWeave.hpp was derived for. Changing one without the other silently
    //  misscales the weave.
    // ----------------------------------------------------------------------
    inline HighPrecType weaveSmoothStep (const HighPrecType t) noexcept
    {
        const HighPrecType u = CLAMP_VALUE(t, 0.0, 1.0);
        return u * u * (3.0 - 2.0 * u);
    }


    // ----------------------------------------------------------------------
    //  A unit-variance value at one integer lattice node.
    //
    //  Keyed on the node index and the octave, so every octave has its own
    //  independent lattice and they cannot correlate.
    // ----------------------------------------------------------------------
    inline HighPrecType weaveNode
    (
        const int32_t  node,
        const int32_t  octave,
        const uint32_t seed,
        const uint32_t tag
    ) noexcept
    {
        // Negative frame indices are legitimate - a clip may start anywhere on the
        // roll - so the index is reinterpreted rather than assumed positive.
        const uint64_t counter = AlgoDefectHash(seed, node, octave, tag);

        return AlgoRngNormal(counter);
    }


    // ----------------------------------------------------------------------
    //  Bilinear fetch with the edge clamped.
    //
    //  The clamp is what a real gate does: a frame shifted in the aperture simply
    //  shows slightly more or less of itself at the edge. Wrapping would bring the
    //  opposite edge into view, which is a distinctive and completely wrong
    //  artefact; leaving the border undefined would show whatever the arena
    //  happened to contain.
    // ----------------------------------------------------------------------
    //  RULE D1 ALIGNMENT, 2026-08-11: this sampler was HighPrecType throughout,
    //  and it runs ONCE PER PIXEL PER CHANNEL - 6.2 million calls at HD.
    //
    //  Every quantity in it is a raster coordinate or a plane sample. Coordinates
    //  reach 4096 and are exact in float to 16.7 million; the samples themselves
    //  ARE float, so widening them to double, interpolating, and narrowing back
    //  bought nothing at all - the two roundings on the way in and out are the
    //  same magnitude as the one it was avoiding. The weave displacement is
    //  sub-pixel, so tx and ty are order one and lose nothing.
    //
    //  This is the clearest D1 violation found in the audit: not a precision
    //  trade, just a widening that never paid for itself.
    // ----------------------------------------------------------------------
    //  CATMULL-ROM RESAMPLE, replacing bilinear -- 2026-08-11.
    //
    //  WHY THIS IS A PHYSICS FIX, NOT A QUALITY PREFERENCE.
    //
    //  Gate weave is a TRANSLATION: the print sits a fraction of a pixel away
    //  from where it sat on the previous frame, and the projector's optics do not
    //  change when it moves. The observer sees the same sharp image at a new
    //  position. Bilinear interpolation cannot express that - it is a two-tap
    //  box-like filter whose transfer at Nyquist falls to cos^2(pi/4) = 0.5 in
    //  amplitude for a half-pixel shift, so it SOFTENS every frame it moves.
    //
    //  Measured on Lady.png through EASTMAN_EKTACHROME_5239, Laplacian variance
    //  of the green record, against the Python reference which has no weave:
    //
    //      source                       216.5
    //      Python reference             447.3   (gamma 1.45 AMPLIFIES detail)
    //      C++ with weave, bilinear     131.9   <- 3.4x softer than reference
    //      C++ weave off                272.7
    //      C++ weave + misreg off       520.0
    //
    //  So bilinear weave alone destroyed 52 per cent of the high-frequency
    //  energy, and the sub-pixel misregistration shift a further 48 per cent -
    //  on a stock whose emulsion MTF (f50 48-60 cycles/mm) sits ABOVE the 38.5
    //  cycles/mm Nyquist limit at HD super35 and should therefore be nearly
    //  transparent at this resolution.
    //
    //  It also DOUBLE-COUNTS the digitisation loss: the scanner's own MTF is
    //  already modelled at stage 10, so adding an interpolation kernel on top
    //  charges for the same physical softening twice.
    //
    //  Catmull-Rom is the correct choice here rather than Lanczos: it is a
    //  four-tap cubic that is C1 continuous and INTERPOLATING (it passes exactly
    //  through the samples, so a zero shift is the identity), its transfer at
    //  Nyquist for a half-pixel shift is ~0.87 against bilinear's 0.5, and four
    //  taps keeps the cost near the two-tap version. Lanczos-3 would be slightly
    //  flatter still at six taps and considerably more ringing.
    //
    //  OVERSHOOT IS REAL AND IS CLAMPED AT ZERO ONLY. The cubic has negative
    //  lobes, so a hard edge can undershoot below the local minimum. The values
    //  here are transmittances and densities, where NEGATIVE is meaningless and
    //  would poison the logarithm downstream - so zero is a physical floor, not a
    //  display clamp. The positive overshoot is deliberately NOT clamped: it is
    //  the edge acutance that the film genuinely has and that bilinear was
    //  removing.
    // ----------------------------------------------------------------------
    inline AlgoType catmullRom1D
    (
        const AlgoType p0,
        const AlgoType p1,
        const AlgoType p2,
        const AlgoType p3,
        const AlgoType t
    ) noexcept
    {
        // Standard Catmull-Rom basis, Horner form: three FMAs after the four
        // coefficient combinations. p1 is the sample at t = 0 and p2 at t = 1.
        const AlgoType a = static_cast<AlgoType>(-0.5) * p0
                         + static_cast<AlgoType>( 1.5) * p1
                         + static_cast<AlgoType>(-1.5) * p2
                         + static_cast<AlgoType>( 0.5) * p3;

        const AlgoType b =                        p0
                         + static_cast<AlgoType>(-2.5) * p1
                         + static_cast<AlgoType>( 2.0) * p2
                         + static_cast<AlgoType>(-0.5) * p3;

        const AlgoType c = static_cast<AlgoType>(-0.5) * p0
                         + static_cast<AlgoType>( 0.5) * p2;

        return ((a * t + b) * t + c) * t + p1;
    }


    //  Catmull-Rom basis expanded into four tap weights at a fixed fraction.
    //
    //  Once per frame, not once per pixel: the gate displaces the whole frame by
    //  the same amount. The four weights sum to exactly one at every t, so a flat
    //  field passes through unchanged and the weave cannot alter overall level.
    inline void weaveWeights (const AlgoType t, AlgoType c[4]) noexcept
    {
        const AlgoType t2 = t * t;
        const AlgoType t3 = t2 * t;

        c[0] = static_cast<AlgoType>(-0.5) * t3 + t2
             + static_cast<AlgoType>(-0.5) * t;
        c[1] = static_cast<AlgoType>( 1.5) * t3
             + static_cast<AlgoType>(-2.5) * t2 + ALGO_ONE;
        c[2] = static_cast<AlgoType>(-1.5) * t3
             + static_cast<AlgoType>( 2.0) * t2
             + static_cast<AlgoType>( 0.5) * t;
        c[3] = static_cast<AlgoType>( 0.5) * t3
             + static_cast<AlgoType>(-0.5) * t2;

        return;
    }
}


// ---------------------------------------------------------------------------
//  One sample of unit-variance red noise
// ---------------------------------------------------------------------------
HighPrecType AlgoWeaveNoise
(
    const HighPrecType framePos,
    const HighPrecType periodLo,
    const uint32_t     seed,
    const uint32_t     tag
) noexcept
{
    // A period below one frame is entirely above Nyquist and can only alias, so
    // the lowest octave is floored there. It is reached only if a profile carries
    // an absurd corner frequency.
    const HighPrecType p0 = MAX_VALUE(periodLo, 1.0);

    HighPrecType sum      = 0.0;
    HighPrecType variance = 0.0;

    for (int32_t oct = 0; oct < ALGO_WEAVE_OCTAVES; oct++)
    {
        // Octave periods halve, so frequencies double.
        const HighPrecType period = p0 / static_cast<HighPrecType>(1 << oct);

        // Amplitudes halve with the period. Amplitude proportional to 1/f makes
        // power proportional to 1/f^2, which is the red-noise shape weave has -
        // and it is why weave drifts rather than vibrates.
        const HighPrecType amp = 1.0 / static_cast<HighPrecType>(1 << oct);

        const HighPrecType pos = framePos / period;

        const HighPrecType fl = std::floor(pos);

        const int32_t node = static_cast<int32_t>(fl);

        const HighPrecType frac = pos - fl;

        const HighPrecType a = weaveNode(node,     oct, seed, tag);
        const HighPrecType b = weaveNode(node + 1, oct, seed, tag);

        const HighPrecType w = weaveSmoothStep(frac);

        sum += amp * (a + (b - a) * w);

        // Each octave contributes its amplitude squared to the total variance.
        variance += amp * amp;
    }

    // Two corrections in one division, and both are needed.
    //
    // The octave sum has variance equal to the sum of the squared amplitudes, not
    // one. And interpolating between independent lattice values loses a further
    // factor, because an interpolated point is a weighted mean of its neighbours -
    // see ALGO_WEAVE_INTERP_VARIANCE for the exact figure and its derivation.
    //
    // Without both, the RMS that appears is not the RMS that was asked for.
    const HighPrecType norm = variance * ALGO_WEAVE_INTERP_VARIANCE;

    return (norm > 0.0) ? (sum / std::sqrt(norm)) : 0.0;
}


// ---------------------------------------------------------------------------
//  Stage 15: gate weave
// ---------------------------------------------------------------------------
void AlgoStage15_GateWeave
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrA,
    AlgoType* RESTRICT       pScrB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           negWidthMm,
    const AlgoType           negHeightMm,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
) noexcept
{
    // The bilinear path resamples straight from source to destination, so neither
    // scratch plane is needed. They stay in the signature because a separable
    // higher-order resample would want them, and changing the signature later
    // would mean revisiting the call site.
    (void)pScrA;
    (void)pScrB;

    // Frame extents are consumed through pxPerMm, which already carries the
    // relationship between film millimetres and pixels.
    (void)negWidthMm;
    (void)negHeightMm;

    // ----------------------------------------------------------------------
    //  Gates, cheapest first.
    // ----------------------------------------------------------------------
    if (false == params.filmDamageEnabled)
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB,
                      sizeX, sizeY, pitch);
        return;
    }

    const FilmDamage& dmg = params.damage;

    const HighPrecType strength =
        MAX_VALUE(static_cast<HighPrecType>(dmg.damageStrength), 0.0);

    const HighPrecType level =
        MAX_VALUE(static_cast<HighPrecType>(dmg.weaveAmount), 0.0) * strength;

    // ----------------------------------------------------------------------
    //  Era amplitude from the stock, user level on top.
    //
    //  TemporalSpec is populated for every profile in the database - about 20 to
    //  25 micrometres RMS for 1930s and 1940s material, 10 for the 1950s, 6 by the
    //  1970s, 3 for a modern pin-registered camera. So the era sets the baseline
    //  and the control expresses intent, which is the arrangement the whole engine
    //  uses for stock-dependent behaviour.
    //
    //  Two amplitudes rather than one because vertical instability exceeds
    //  horizontal on vertically-transported formats, which is nearly all cine.
    // ----------------------------------------------------------------------
    const HighPrecType ampXmm =
        (static_cast<HighPrecType>(profile.temporal.weave_amp_x_um)
         / ALGO_WEAVE_UM_PER_MM) * level;

    const HighPrecType ampYmm =
        (static_cast<HighPrecType>(profile.temporal.weave_amp_y_um)
         / ALGO_WEAVE_UM_PER_MM) * level;

    const HighPrecType scale = static_cast<HighPrecType>(pxPerMm);

    const HighPrecType ampXpx = ampXmm * scale;
    const HighPrecType ampYpx = ampYmm * scale;

    if (ampXpx <= 0.0 && ampYpx <= 0.0)
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB,
                      sizeX, sizeY, pitch);
        return;
    }

    // ----------------------------------------------------------------------
    //  Lowest octave period, in frames.
    //
    //  The profile gives a corner frequency in hertz and the sequence is sampled
    //  once per frame, so the period in frames is the frame rate divided by that
    //  frequency. A 0.8 Hz corner at 24 fps is a 30 frame period - a drift over
    //  more than a second, which is what weave looks like.
    //
    //  Guarded because a profile could carry zero, which would be an infinite
    //  period rather than an error.
    // ----------------------------------------------------------------------
    const HighPrecType corner =
        MAX_VALUE(static_cast<HighPrecType>(profile.temporal.weave_hz_corner),
                  0.01);

    const HighPrecType fps = MAX_VALUE(static_cast<HighPrecType>(frameRate), 1.0);

    const HighPrecType periodLo = fps / corner;

    // ----------------------------------------------------------------------
    //  This frame's displacement.
    //
    //  A pure function of the frame index, so scrubbing and out-of-order rendering
    //  are stable. The roll seed comes from the damage group rather than the
    //  engine seed, so re-rolling the grain does not re-roll the weave.
    // ----------------------------------------------------------------------
    const uint32_t rollSeed = static_cast<uint32_t>(dmg.damageSeed) ^ seed;

    const HighPrecType framePos = static_cast<HighPrecType>(frameIndex);

    const HighPrecType dx = ampXpx * AlgoWeaveNoise(framePos, periodLo,
                                                    rollSeed, ALGO_WEAVE_TAG_X);
    const HighPrecType dy = ampYpx * AlgoWeaveNoise(framePos, periodLo,
                                                    rollSeed, ALGO_WEAVE_TAG_Y);

    // Below this the bilinear weights round to a pass-through and the resample is
    // an expensive copy.
    if (MAX_VALUE(dx < 0.0 ? -dx : dx, dy < 0.0 ? -dy : dy)
        < ALGO_WEAVE_MIN_SHIFT_PX)
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB,
                      sizeX, sizeY, pitch);
        return;
    }

    // ----------------------------------------------------------------------
    //  Resample.
    //
    //  The destination pixel at (x, y) shows the part of the image that the weave
    //  has moved there, so the SOURCE is read at (x - dx, y - dy). The sign is
    //  worth stating because getting it backwards produces motion that is
    //  perfectly plausible in isolation and exactly inverted against the gate dirt
    //  applied at stage 16 - which is precisely the cue this stage exists to
    //  create.
    // ----------------------------------------------------------------------
    const AlgoType* RESTRICT srcPlane[3] = { pSrcR, pSrcG, pSrcB };
    AlgoType* RESTRICT       dstPlane[3] = { pDstR, pDstG, pDstB };

    for (int32_t c = 0; c < 3; c++)
    {
        const AlgoType* RESTRICT pIn  = srcPlane[c];
        AlgoType* RESTRICT       pOut = dstPlane[c];

        // ------------------------------------------------------------------
        //  Frame-constant resample geometry, computed once for the whole plane.
        //
        //  Integer offset and fractional part of the displacement, then the four
        //  Catmull-Rom tap weights per axis. The interior bounds are the pixels
        //  whose whole four-tap support lies inside the raster.
        // ------------------------------------------------------------------
        const AlgoType fxW = std::floor(-static_cast<AlgoType>(dx));
        const AlgoType fyW = std::floor(-static_cast<AlgoType>(dy));

        const int32_t ixW = static_cast<int32_t>(fxW);
        const int32_t iyW = static_cast<int32_t>(fyW);

        AVX2_ALIGN AlgoType cxW[4];
        AVX2_ALIGN AlgoType cyW[4];

        weaveWeights(static_cast<AlgoType>(-static_cast<AlgoType>(dx)) - fxW, cxW);
        weaveWeights(static_cast<AlgoType>(-static_cast<AlgoType>(dy)) - fyW, cyW);

        const __m256 vcxW[4] = { _mm256_set1_ps(cxW[0]), _mm256_set1_ps(cxW[1]),
                                _mm256_set1_ps(cxW[2]), _mm256_set1_ps(cxW[3]) };

        const int32_t xLoW = MIN_VALUE(MAX_VALUE(1 - ixW, 0), sizeX);
        const int32_t xHiW = CLAMP_VALUE(sizeX - 2 - ixW, xLoW, sizeX);

        const int32_t innerW = xHiW - xLoW;
        const int32_t vecsW  = innerW / 8;
        const int32_t tailW  = innerW - vecsW * 8;

        const __m256i vTailW = algoTailMaskLocal(tailW);

        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT rOut = pOut
                + static_cast<std::ptrdiff_t>(y) * pitch;

            // --------------------------------------------------------------
            //  THE WEAVE DISPLACEMENT IS A FRAME CONSTANT.
            //
            //  Every pixel of the frame moves by the same (dx, dy), so the four
            //  horizontal and four vertical Catmull-Rom weights are computed once
            //  per frame (above) and this loop is a fixed 4x4 separable
            //  convolution at an integer offset.
            //
            //  The first Catmull-Rom version called weaveSample per pixel, which
            //  re-derived the cubic basis 2.07 million times and clamped sixteen
            //  indices each time: stage 15 measured 13.0 ms before, 70.3 ms after.
            //  This form keeps the identical arithmetic and vectorises the
            //  interior, where none of the four taps needs clamping.
            // --------------------------------------------------------------
            const int32_t syBase = y + iyW;

            const AlgoType* RESTRICT rows[4];

            for (int32_t k = 0; k < 4; k++)
                rows[k] = pIn + static_cast<std::ptrdiff_t>(
                              CLAMP_VALUE(syBase + k - 1, 0, sizeY - 1)) * pitch;

            int32_t x = 0;

            // --- left edge, clamped, scalar ---
            for (; x < xLoW; x++)
            {
                AlgoType acc = ALGO_ZERO;

                for (int32_t ky = 0; ky < 4; ky++)
                {
                    AlgoType h = ALGO_ZERO;

                    for (int32_t kx = 0; kx < 4; kx++)
                        h += cxW[kx] * rows[ky][CLAMP_VALUE(x + ixW + kx - 1,
                                                            0, sizeX - 1)];

                    acc += cyW[ky] * h;
                }

                rOut[x] = MAX_VALUE(acc, ALGO_ZERO);
            }

            // --- interior, no clamp, vectorised: four contiguous loads a row ---
            for (int32_t v = 0; v < vecsW; v++, x += 8)
            {
                __m256 acc = _mm256_setzero_ps();

                for (int32_t ky = 0; ky < 4; ky++)
                {
                    const AlgoType* RESTRICT w = rows[ky] + x + ixW - 1;

                    __m256 h = _mm256_mul_ps(vcxW[0], _mm256_loadu_ps(w));
                    h = _mm256_fmadd_ps(vcxW[1], _mm256_loadu_ps(w + 1), h);
                    h = _mm256_fmadd_ps(vcxW[2], _mm256_loadu_ps(w + 2), h);
                    h = _mm256_fmadd_ps(vcxW[3], _mm256_loadu_ps(w + 3), h);

                    acc = _mm256_fmadd_ps(_mm256_set1_ps(cyW[ky]), h, acc);
                }

                _mm256_storeu_ps(rOut + x,
                                 _mm256_max_ps(acc, _mm256_setzero_ps()));
            }

            if (tailW > 0)
            {
                __m256 acc = _mm256_setzero_ps();

                for (int32_t ky = 0; ky < 4; ky++)
                {
                    const AlgoType* RESTRICT w = rows[ky] + x + ixW - 1;

                    __m256 h = _mm256_mul_ps(vcxW[0],
                                   _mm256_maskload_ps(w, vTailW));
                    h = _mm256_fmadd_ps(vcxW[1],
                            _mm256_maskload_ps(w + 1, vTailW), h);
                    h = _mm256_fmadd_ps(vcxW[2],
                            _mm256_maskload_ps(w + 2, vTailW), h);
                    h = _mm256_fmadd_ps(vcxW[3],
                            _mm256_maskload_ps(w + 3, vTailW), h);

                    acc = _mm256_fmadd_ps(_mm256_set1_ps(cyW[ky]), h, acc);
                }

                _mm256_maskstore_ps(rOut + x, vTailW,
                                    _mm256_max_ps(acc, _mm256_setzero_ps()));

                x += tailW;
            }

            // --- right edge, clamped, scalar ---
            for (; x < sizeX; x++)
            {
                AlgoType acc = ALGO_ZERO;

                for (int32_t ky = 0; ky < 4; ky++)
                {
                    AlgoType h = ALGO_ZERO;

                    for (int32_t kx = 0; kx < 4; kx++)
                        h += cxW[kx] * rows[ky][CLAMP_VALUE(x + ixW + kx - 1,
                                                            0, sizeX - 1)];

                    acc += cyW[ky] * h;
                }

                rOut[x] = MAX_VALUE(acc, ALGO_ZERO);
            }
        }
    }

    return;
}
#endif