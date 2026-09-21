// ---------------------------------------------------------------------------
//  Algo_11_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the zero-fill, the zero-mean normalisation, and the FINAL ADD - which
//  is the one that matters, being a square root and an FMA per sample per channel, and
//  AVX2 has a real sqrt instruction.
//
//  The white-noise fill is left scalar. It is a counter-based hash per pixel, and a
//  64-bit mix does not vectorise cleanly under AVX2 - there is no 64x64 high multiply -
//  so it would need restructuring rather than translating. Worth measuring before
//  attempting: the two blurs it feeds are already vectorised, so the fill may no
//  longer be where this stage spends its time.
//
//  ALIGNMENT: EVERY IMAGE ACCESS IS UNALIGNED, DELIBERATELY.
//
//  loadu/storeu on all plane data. The arena base comes from the host's memory pool,
//  whose alignment argument is a HINT, not a guarantee - it was observed returning a
//  base 16 mod 32, which faults an aligned 256-bit load. AlgoMemHandler.cpp is SHARED
//  by both flavours and must not be changed to suit the vector path, so the vector
//  path carries no alignment assumption instead. Costs nothing measurable on Haswell
//  and later.
//
//  Pipeline stage 11, in the density domain:
//
//      AlgoMakeGrainField   one calibrated, spectrally shaped, zero-mean field
//      AlgoAddGrain         add fields to density with square-root weighting
//      AlgoStage11_Grain    the camera negative's own grain
//
//  Raw pointers, explicit geometry, no allocation, no mutable state, no validation
//  of inputs.
// ---------------------------------------------------------------------------

// Common.hpp -- AVX2_ALIGN / CACHE_ALIGN are defined here. Included
// DIRECTLY rather than relied on transitively: this file declares an
// aligned buffer, so the macro must not depend on another header's
// include order to be in scope.
#include "Common.hpp"
#include "AlgoGrain.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>


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


    // ======================================================================
    //  VECTOR COUNTER-BASED NORMAL GENERATOR
    //
    //  WHY THIS EXISTS. Measured in isolation on an HD plane, the scalar
    //  counter-RNG below costs 41.6 ms - 20.1 ns per pixel - and a colour stock
    //  draws THREE independent fields, so 125 ms of an HD frame was one scalar
    //  loop. That was 18 per cent of the whole engine, second only to the
    //  interimage stage, and none of it was vectorised because SplitMix64 needs
    //  a 64-bit multiply and AVX2 has none.
    //
    //  WHAT IS PRESERVED EXACTLY:
    //    - the counter construction. Every sample is still a pure function of
    //      (seed, frameIndex, stage, pixel ordinal), so the field is unchanged
    //      by render order, by tiling, by threading or by the host scrubbing
    //      backwards. This is not an optimisation detail, it is the property the
    //      whole design rests on.
    //    - the mixing bijection. Same SplitMix64 constants, same shifts, same
    //      sequence of operations - only the 64-bit multiply is emulated.
    //    - the transform. Still Box-Muller, so the field is still exactly
    //      standard normal with the real Gaussian tails. A bounded generator
    //      would have been cheaper still and would have survived the variance
    //      calibration, but it would have removed the rare bright and dark
    //      specks that a developed emulsion genuinely has.
    //
    //  WHAT DIFFERS FROM THE SCALAR PATH, DELIBERATELY:
    //    - the uniforms are formed from the top 24 bits rather than the top 53,
    //      because the destination is a 32-bit float. 16.7 million distinct
    //      values per uniform, against the float field's own ~2^24 resolution.
    //    - log and cos are the vector approximations rather than libm.
    //
    //  So the field is NOT bit-identical to the scalar one, and cannot be: this
    //  is a statistical equality, not an arithmetic one. It is verified as such -
    //  mean, variance and the rendered RMS granularity - never by differencing
    //  two images.
    // ======================================================================


    // ----------------------------------------------------------------------
    //  64-bit multiply, low half, four lanes at a time.
    //
    //  AVX2 has no 64x64 multiply. It does have _mm256_mul_epu32, which takes the
    //  LOW 32 bits of each 64-bit lane and returns the full 64-bit product, so the
    //  identity
    //
    //      a*b = al*bl + ((al*bh + ah*bl) << 32)     (mod 2^64)
    //
    //  gives the low half in three multiplies. The high-half carries that the
    //  omitted ah*bh term would contribute all land above bit 63 and are
    //  discarded by the modulus anyway, so this is EXACT and not an
    //  approximation - it is the same value the scalar multiply produces.
    // ----------------------------------------------------------------------
    inline __m256i algoMul64Lo (const __m256i a, const __m256i b) noexcept
    {
        const __m256i aHi = _mm256_srli_epi64(a, 32);
        const __m256i bHi = _mm256_srli_epi64(b, 32);

        const __m256i albl = _mm256_mul_epu32(a,   b);      // low x low, full 64
        const __m256i albh = _mm256_mul_epu32(a,   bHi);    // low x high
        const __m256i ahbl = _mm256_mul_epu32(aHi, b);      // high x low

        const __m256i mid = _mm256_add_epi64(albh, ahbl);

        return _mm256_add_epi64(albl, _mm256_slli_epi64(mid, 32));
    }


    // ----------------------------------------------------------------------
    //  SplitMix64 finaliser, four lanes at a time.
    //
    //  Operation for operation the scalar AlgoRngMix64: add the golden constant,
    //  then two rounds of xor-shift-multiply, then a final xor-shift. Same
    //  constants, same shift distances - they were selected by avalanche search
    //  and substituting others would still give a bijection but a worse one.
    // ----------------------------------------------------------------------
    inline __m256i algoMix64V (__m256i z) noexcept
    {
        const __m256i vGolden = _mm256_set1_epi64x(
            static_cast<long long>(ALGO_RNG_GOLDEN));
        const __m256i vMix1 = _mm256_set1_epi64x(
            static_cast<long long>(ALGO_RNG_MIX_1));
        const __m256i vMix2 = _mm256_set1_epi64x(
            static_cast<long long>(ALGO_RNG_MIX_2));

        z = _mm256_add_epi64(z, vGolden);

        z = algoMul64Lo(_mm256_xor_si256(z, _mm256_srli_epi64(z, ALGO_RNG_SHIFT_1)),
                        vMix1);

        z = algoMul64Lo(_mm256_xor_si256(z, _mm256_srli_epi64(z, ALGO_RNG_SHIFT_2)),
                        vMix2);

        return _mm256_xor_si256(z, _mm256_srli_epi64(z, ALGO_RNG_SHIFT_3));
    }


    // ----------------------------------------------------------------------
    //  Eight uniforms in [0, 1) from eight 64-bit counters.
    //
    //  The counters arrive as two vectors of four. Each mixed result contributes
    //  its top 24 bits, which are packed down to 32-bit lanes and scaled by
    //  2^-24. 24 bits because the destination is float: a float mantissa holds
    //  24, so asking for more would produce values the type cannot distinguish.
    //
    //  The pack takes the HIGH halves of the 64-bit results - the bits the
    //  scalar path also uses - by shifting each lane right by 40 and then
    //  gathering the resulting 24-bit values into one vector of eight.
    // ----------------------------------------------------------------------
    inline __m256 algoUniform8 (const __m256i cLo, const __m256i cHi) noexcept
    {
        // Top 24 bits of each 64-bit mix, moved down to the bottom of the lane.
        const __m256i mLo = _mm256_srli_epi64(algoMix64V(cLo), 40);
        const __m256i mHi = _mm256_srli_epi64(algoMix64V(cHi), 40);

        // Each 64-bit lane now holds a value below 2^24, so the low 32-bit half
        // of every lane carries the whole value. Shuffle those four halves of
        // each vector together into one vector of eight 32-bit integers.
        //
        // 0xD8 = _MM_SHUFFLE(3,1,2,0): brings lanes 0,2 (the low halves of the
        // two 64-bit lanes in each 128-bit half) into the low 64 bits.
        const __m256i pLo = _mm256_shuffle_epi32(mLo, 0xD8);
        const __m256i pHi = _mm256_shuffle_epi32(mHi, 0xD8);

        // Interleave the two 128-bit halves so the eight values end up in
        // counter order: lanes 0..3 from cLo, lanes 4..7 from cHi.
        const __m256i a = _mm256_permute4x64_epi64(pLo, 0xD8);   // pack cLo's four
        const __m256i b = _mm256_permute4x64_epi64(pHi, 0xD8);   // pack cHi's four

        const __m256i packed =
            _mm256_permute2x128_si256(a, b, 0x20);   // low 128 of each

        // 2^-24, so the result lies in [0, 1) with 16.7 million distinct values.
        return _mm256_mul_ps(_mm256_cvtepi32_ps(packed),
                             _mm256_set1_ps(5.9604644775390625e-08f));
    }


    // ----------------------------------------------------------------------
    //  cos(2*pi*u) for u in [0, 1), eight lanes.
    //
    //  There is no vector cosine in the shared fast-arithmetic header and none in
    //  AVX2, so this is a Taylor series in the SQUARE of the reduced argument,
    //  which is the right form because cosine is even.
    //
    //  Reduction: t = 2u - 1 maps [0,1) onto [-1,1), and cos(2*pi*u) =
    //  cos(pi*(t+1)) = -cos(pi*t). Six terms in s = t^2 hold the worst-case
    //  error - at the interval ends, s = 1 - to about 2e-05 absolute.
    //
    //  That is far more accuracy than the consumer needs: this cosine only sets
    //  the PHASE of a Box-Muller draw, so an error in it perturbs which normal
    //  value a given pixel receives without altering the distribution the values
    //  are drawn from. The series is carried to six terms anyway because each one
    //  is a single FMA.
    // ----------------------------------------------------------------------
    inline __m256 algoCosTwoPiV (const __m256 u) noexcept
    {
        // t = 2u - 1, in [-1, 1).
        const __m256 t = _mm256_fmsub_ps(u, _mm256_set1_ps(2.0f),
                                         _mm256_set1_ps(1.0f));

        const __m256 s = _mm256_mul_ps(t, t);

        // cos(pi*t) = 1 - (pi^2/2!)s + (pi^4/4!)s^2 - (pi^6/6!)s^3
        //               + (pi^8/8!)s^4 - (pi^10/10!)s^5 + (pi^12/12!)s^6
        // Horner from the highest term down; every step is one FMA.
        __m256 r = _mm256_set1_ps(1.8028508506e-03f);              // +pi^12/12!
        r = _mm256_fmadd_ps(r, s, _mm256_set1_ps(-2.5806891390e-02f)); // -pi^10/10!
        r = _mm256_fmadd_ps(r, s, _mm256_set1_ps( 2.3533063036e-01f)); // +pi^8/8!
        r = _mm256_fmadd_ps(r, s, _mm256_set1_ps(-1.3352627688e+00f)); // -pi^6/6!
        r = _mm256_fmadd_ps(r, s, _mm256_set1_ps( 4.0587121264e+00f)); // +pi^4/4!
        r = _mm256_fmadd_ps(r, s, _mm256_set1_ps(-4.9348022005e+00f)); // -pi^2/2!
        r = _mm256_fmadd_ps(r, s, _mm256_set1_ps( 1.0f));

        // cos(2*pi*u) = -cos(pi*t).
        return _mm256_sub_ps(_mm256_setzero_ps(), r);
    }


    // ----------------------------------------------------------------------
    //  Eight standard normals from a base counter and eight consecutive ordinals.
    //
    //  Box-Muller, exactly as the scalar path: two uniforms per value, the second
    //  counter displaced by the golden constant rather than by one so the mixer's
    //  two inputs are far apart even though they came from the same request. Only
    //  the cosine branch is kept; the sine branch would be a second independent
    //  value but keeping it would require state, and there is none here.
    //
    //  u1 is floored away from zero. log(0) is minus infinity and a single
    //  non-finite value would poison the entire blurred field; the probability is
    //  2^-24 per draw, which is small but not zero at two million draws a frame.
    //
    //  The counter arithmetic mirrors AlgoRngCounter: the seed/stage fields are
    //  passed in already folded, and only the ordinal varies per lane, so the
    //  per-pixel work is one add rather than a repacking.
    // ----------------------------------------------------------------------
    inline __m256 algoNormal8
    (
        const __m256i base,        // seedField ^ stageField, broadcast
        const __m256i ordLo,       // ordinals for lanes 0..3
        const __m256i ordHi        // ordinals for lanes 4..7
    ) noexcept
    {
        const __m256i vGolden = _mm256_set1_epi64x(
            static_cast<long long>(ALGO_RNG_GOLDEN));

        // Counter per lane: the packed seed/stage field xor the ordinal, which is
        // exactly what AlgoRngCounter produces for a 24-bit ordinal.
        const __m256i c1Lo = _mm256_xor_si256(base, ordLo);
        const __m256i c1Hi = _mm256_xor_si256(base, ordHi);

        // Second uniform's counter, displaced by the golden constant.
        const __m256i c2Lo = _mm256_xor_si256(c1Lo, vGolden);
        const __m256i c2Hi = _mm256_xor_si256(c1Hi, vGolden);

        const __m256 u1raw = algoUniform8(c1Lo, c1Hi);
        const __m256 u2    = algoUniform8(c2Lo, c2Hi);

        // Floor at one step of the 24-bit uniform.
        const __m256 u1 = _mm256_max_ps(u1raw,
                                        _mm256_set1_ps(5.9604644775390625e-08f));

        // sqrt(-2 ln u1) * cos(2 pi u2). The square root is the exact hardware
        // instruction; only the logarithm is approximate.
        const __m256 radius = _mm256_sqrt_ps(
            _mm256_mul_ps(_mm256_set1_ps(-2.0f),
                          FastCompute::AVX2::Log(u1)));

        return _mm256_mul_ps(radius, algoCosTwoPiV(u2));
    }

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

#include <cmath>   // std::sqrt, std::exp


// ---------------------------------------------------------------------------
//  ⚠ THE ANONYMOUS-NAMESPACE grainReferenceEnergy() THAT USED TO LIVE HERE IS
//  GONE, AND ITS BEING FILE-STATIC WAS PART OF THE PROBLEM. It integrated the
//  v47 two-lobe spectrum against a GAUSSIAN stand-in for the 48 micrometre
//  aperture -- a SECOND COPY of the scalar engine's identical helper, in a
//  different file, invisible to both. Schema v48 replaces it with
//  AlgoGrainReferenceEnergy() in AlgoGrain.hpp, which this engine, the scalar
//  engine and the parity harness all reach, so there is one integral to be
//  wrong in instead of two to drift apart.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
//  Build one grain field
// ---------------------------------------------------------------------------
void AlgoMakeGrainFieldTerms
(
    AlgoType* RESTRICT              pDst,
    AlgoType* RESTRICT              pScrNoise,
    AlgoType* RESTRICT              pScrLobe,
    AlgoType* RESTRICT              pScrWork,
    const int32_t                   sizeX,
    const int32_t                   sizeY,
    const int32_t                   pitch,
    const film::GrainSpectrumTerms& terms,
    const bool                      jincAperture,
    const AlgoType                  rmsGranularity,
    const AlgoType                  scanSigmaPx,
    const AlgoType                  pxPerMm,
    const AlgoType                  anisotropy,
    const eALGO_RNG_STAGE           rngStage,
    const uint32_t                  seed,
    const int32_t                   frameIndex,
    const AlgoType                  frameCorrelation,
    const AlgoType                  fixedFraction
) noexcept
{
    // A stock with no spectrum or no granularity figure has no modelled grain.
    // Zero the field rather than leaving it, so the caller can add it
    // unconditionally without a second branch.
    if ((terms.count <= 0) || (rmsGranularity <= ALGO_ZERO) ||
        (pxPerMm <= ALGO_ZERO))
    {
        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow =
                pDst + static_cast<std::ptrdiff_t>(y) * pitch;

            ALGO_VECTOR_HINT
            for (int32_t x = 0; x < sizeX; x++)
                pRow[x] = ALGO_ZERO;
        }

        return;
    }

    // ----------------------------------------------------------------------
    //  Spectral shape, as a GAUSSIAN MIXTURE.
    //
    //  ⚠ THE TWO-LOBE FORM THAT USED TO BE HERE IS GONE, AND WITH IT THE
    //  kSigma = 0.22508352815546 LITERAL, WHICH WAS WRONG. 1/(pi*sqrt 2) is
    //  0.22507907903927651 -- the shipped digits were wrong from the sixth
    //  place, 1.98e-05 relative, in BOTH engines, and nothing failed because the
    //  reference never computes a sigma on that path and so had nothing to
    //  disagree with. The correct value now lives once, as
    //  ALGO_GRAIN_SIGMA_PER_1E in AlgoGrain.hpp.
    //
    //  The spectrum arrives as sum_k w_k exp(-2 pi^2 s_k^2 f^2). Each term is
    //  one separable Gaussian blur of the SAME white field, and the results are
    //  summed with the weights -- which is exact, not an approximation, because
    //  a product of Gaussian transfers is a Gaussian whose variances add. The
    //  legacy spectrum is the same machinery with one or two terms.
    //
    //  ⚠ THE WEIGHTS DO NOT SUM TO ONE WHEN A CLUSTERING LOBE IS PRESENT; they
    //  sum to 1 + clump_gain. This is spectral shaping of a noise field, not an
    //  averaging filter, so handing the terms to a multi-lobe blur helper that
    //  normalises by the weight sum would silently divide the field by (1 + g).
    // ----------------------------------------------------------------------
    const HighPrecType scanPx = static_cast<HighPrecType>(
                                    MAX_VALUE(scanSigmaPx, ALGO_ZERO));

    // ----------------------------------------------------------------------
    //  ANISOTROPY -- and this closes a PYTHON/C++ DIVERGENCE FOUND 2026-09-11.
    //
    //  ⚠ WHAT WAS WRONG. `GrainSpec::anisotropy` has been emitted into
    //  film_profiles.hpp all along and NO C++ STAGE READ IT. The reference
    //  model does read it, so for the 33 of 191 stocks that carry a figure -
    //  values 1.02 to 1.10 - Python rendered stretched grain and both C++
    //  engines rendered round grain. Nothing failed and nothing looked broken;
    //  the two engines simply modelled different emulsions, on the stocks whose
    //  coating flow was actually measured.
    //
    //  WHAT IT MEANS PHYSICALLY. Emulsion is poured, and the developed clumps
    //  end up slightly elongated along the direction of that flow. So the grain
    //  of a real negative has a longer correlation length down the frame than
    //  across it, by the ratio stored here.
    //
    //  WHY A SIGMA RATIO IS THE SAME LAW AS THE REFERENCE'S FREQUENCY STRETCH.
    //  The reference multiplies the VERTICAL frequency axis by the anisotropy
    //  before evaluating the radial transfer H(f). For a Gaussian shape that
    //  factorises exactly into a horizontal transfer and a vertical one whose
    //  rolloff frequency is a times LOWER, and a spatial sigma is inversely
    //  proportional to its rolloff frequency - so a times lower in frequency is
    //  a times WIDER in space:
    //
    //      sigma_y = anisotropy * sigma_x
    //
    //  The full derivation is at AlgoGaussianBlurPlaneWrapXY in
    //  AlgoSeparableBlur.hpp. The direction is the part worth checking twice:
    //  a > 1 stretches the grain DOWN the frame. Getting the sense backwards
    //  squashes it instead, and looks plausible either way on a still frame.
    //
    //  WHY IT MULTIPLIES THE COMBINED SIGMA AND NOT JUST THE CRYSTAL TERM. The
    //  reference builds ONE frequency grid and evaluates the grain shape AND
    //  the scan band limit on it, so the stretch applies to their product. Here
    //  the two are already combined by adding variances into each term's
    //  sigma, and scaling a total sigma by a scales every variance in it by
    //  a^2 - which is the same thing. Applying it to sHiPx alone would stretch
    //  the crystals but not the scanner, which the reference does not do.
    // ----------------------------------------------------------------------
    const AlgoType aniso = MAX_VALUE(anisotropy, ALGO_GRAIN_ANISOTROPY_MIN);

    // ----------------------------------------------------------------------
    //  Amplitude calibration.
    //
    //      scale = (rms / 1000) * px_per_mm / sqrt(E)
    //
    //  The thousand converts the granularity metric, which is sigma(D) times a
    //  thousand, back to a density. The px_per_mm factor relates the discrete
    //  variance of unit white noise on this grid to the continuous integral: unit
    //  white noise has flat spectral density 1 per grid cell, and a grid cell is
    //  1/px_per_mm millimetres on a side.
    //
    //  ⚠ THE ENERGY INTEGRAL IS ISOTROPIC AND STAYS THAT WAY. Stretching one
    //  axis by a really does change the field's true variance, by 1/a, so a
    //  reader could reasonably expect a sqrt(a) correction here. There is none,
    //  because the reference has none either: it calibrates against the same
    //  radial integral of the UNSTRETCHED shape. Adding the correction would be
    //  more defensible physics and an immediate divergence from the model of
    //  record, so it is left out and the reason recorded rather than left to be
    //  rediscovered as an apparent omission.
    // ----------------------------------------------------------------------
    const HighPrecType energy = AlgoGrainReferenceEnergy(terms, jincAperture);

    // A degenerate spectrum would give zero energy and an infinite scale. Guarded
    // rather than validated: the profile is pre-validated, but the integral is
    // computed here and its result is this function's own responsibility.
    const AlgoType scale = (energy > 0.0)
        ? static_cast<AlgoType>(
              (static_cast<HighPrecType>(rmsGranularity) / 1000.0)
              * static_cast<HighPrecType>(pxPerMm) / std::sqrt(energy))
        : ALGO_ZERO;

    // ----------------------------------------------------------------------
    //  Unit-variance white noise.
    //
    //  Counter based, so every sample is a pure function of (seed, frameIndex,
    //  stage, pixel ordinal). No state, no sequence, no dependence on the order in
    //  which pixels or frames are visited - which is mandatory, because the host
    //  renders out of order, speculatively, and from several threads.
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    //  Seed and stage fields, folded once for the whole plane.
    //
    //  AlgoRngCounter packs (seed<<32 ^ frameSalt) ^ (stage<<24) ^ ordinal, and
    //  only the ordinal varies per pixel. Everything else is hoisted here, so the
    //  per-lane work is one xor against a broadcast constant rather than a
    //  repacking of four fields.
    //
    //  Computed with the SAME expressions as the scalar helper - including the
    //  signed-to-unsigned cast of a possibly negative frame index, which is well
    //  defined and wraps, and wrapping is harmless because the mixer treats all
    //  64-bit values alike.
    // ----------------------------------------------------------------------
    const uint64_t stageField =
        (static_cast<uint64_t>(UnderlyingType(rngStage) >> 8) & 0xFFull) << 24;

    //  ⚠⚠ THE BASE IS NOW PER TAP, v49. The temporal model draws the emulsion
    //  field as a weighted sum over neighbouring FRAMES, and the frame index
    //  lives inside `seedField`, so the hoisted broadcast has to be rebuilt for
    //  each tap. With frameCorrelation = 0 the kernel is one tap and this is
    //  the pre-v49 expression evaluated once -- same constant, same numbers,
    //  same order.
    auto algoBaseForFrame = [seed, stageField](const int32_t fi) noexcept
    {
        const uint64_t salt =
            static_cast<uint64_t>(static_cast<uint32_t>(fi)) * ALGO_RNG_GOLDEN;
        const uint64_t sf = (static_cast<uint64_t>(seed) << 32) ^ salt;
        return _mm256_set1_epi64x(static_cast<long long>(sf ^ stageField));
    };

    HighPrecType kern[2 * ALGO_GRAIN_TEMPORAL_MAX_TAPS + 1];
    const int32_t taps = AlgoGrainTemporalKernel(
        static_cast<HighPrecType>(frameCorrelation), kern);
    const int32_t half = taps / 2;

    const HighPrecType fClamp =
        (fixedFraction < ALGO_ZERO) ? 0.0
        : ((static_cast<HighPrecType>(fixedFraction) > 1.0)
               ? 1.0 : static_cast<HighPrecType>(fixedFraction));
    const AlgoType wEmul = static_cast<AlgoType>(std::sqrt(1.0 - fClamp));
    const AlgoType wFix  = static_cast<AlgoType>(std::sqrt(fClamp));

    const __m256i vBase = algoBaseForFrame(frameIndex);

    // Lane ordinal offsets. The ordinals of eight consecutive pixels differ by
    // 0..7, so one add per vector produces all eight counters.
    const __m256i vOff0123 = _mm256_setr_epi64x(0, 1, 2, 3);
    const __m256i vOff4567 = _mm256_setr_epi64x(4, 5, 6, 7);

    {
        const int32_t vecCount = sizeX / ALGO_AVX2_LANES_LOCAL;
        const int32_t tailN    = sizeX - (vecCount * ALGO_AVX2_LANES_LOCAL);

        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow =
                pScrNoise + static_cast<std::ptrdiff_t>(y) * pitch;

            // Ordinal of the first pixel in the row. Using the PADDED width keeps
            // the ordinal unique and makes it independent of the active extent, so
            // a region render draws the same numbers as a full-frame one - the
            // property that lets the host tile the frame however it likes.
            const std::ptrdiff_t rowOrd =
                static_cast<std::ptrdiff_t>(y) * pitch;

            int32_t x = 0;

            for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES_LOCAL)
            {
                // The 24-bit ordinal field, per lane. Masked exactly as the scalar
                // helper masks it, so a plane large enough to overflow 24 bits
                // wraps identically on both paths rather than diverging.
                const __m256i vOrdBase = _mm256_set1_epi64x(
                    static_cast<long long>(
                        static_cast<uint64_t>(rowOrd + x) & 0x00FFFFFFull));

                const __m256i ordLo =
                    _mm256_and_si256(_mm256_add_epi64(vOrdBase, vOff0123),
                                     _mm256_set1_epi64x(0x00FFFFFFll));
                const __m256i ordHi =
                    _mm256_and_si256(_mm256_add_epi64(vOrdBase, vOff4567),
                                     _mm256_set1_epi64x(0x00FFFFFFll));

                _mm256_storeu_ps(pRow + x, algoNormal8(vBase, ordLo, ordHi));
            }

            if (tailN > 0)
            {
                const __m256i vOrdBase = _mm256_set1_epi64x(
                    static_cast<long long>(
                        static_cast<uint64_t>(rowOrd + x) & 0x00FFFFFFull));

                const __m256i ordLo =
                    _mm256_and_si256(_mm256_add_epi64(vOrdBase, vOff0123),
                                     _mm256_set1_epi64x(0x00FFFFFFll));
                const __m256i ordHi =
                    _mm256_and_si256(_mm256_add_epi64(vOrdBase, vOff4567),
                                     _mm256_set1_epi64x(0x00FFFFFFll));

                _mm256_maskstore_ps(pRow + x, algoTailMaskLocal(tailN),
                                    algoNormal8(vBase, ordLo, ordHi));
            }
        }

        //  ⚠ THE REMAINING TAPS AND THE FIXED PATTERN, ACCUMULATED IN PLACE.
        //  The loop above laid down the centre tap unweighted, which is the
        //  whole field when taps == 1. Anything further is a correction on top,
        //  so the common path pays nothing: no extra pass, no extra broadcast,
        //  and the scalar twin takes the same branch on the same condition.
        if (taps > 1 || fClamp > 0.0)
        {
            const AlgoType wCentre = static_cast<AlgoType>(kern[half]);

            for (int32_t y = 0; y < sizeY; y++)
            {
                AlgoType* RESTRICT pRow =
                    pScrNoise + static_cast<std::ptrdiff_t>(y) * pitch;
                const std::ptrdiff_t rowOrd =
                    static_cast<std::ptrdiff_t>(y) * pitch;

                for (int32_t x = 0; x < sizeX; x++)
                {
                    const uint32_t ordinal =
                        static_cast<uint32_t>(rowOrd + x);

                    HighPrecType emul =
                        static_cast<HighPrecType>(pRow[x])
                        * ((taps > 1) ? static_cast<HighPrecType>(wCentre)
                                      : 1.0);

                    for (int32_t i = 0; i < taps; i++)
                    {
                        if (i == half) { continue; }
                        emul += kern[i] * AlgoRngNormal(AlgoRngCounter(
                            seed, frameIndex + (i - half), rngStage, ordinal));
                    }

                    HighPrecType v = emul;
                    if (fClamp > 0.0)
                    {
                        const HighPrecType fixed =
                            AlgoRngNormal(AlgoRngCounter(
                                seed, ALGO_GRAIN_FIXED_FRAME, rngStage,
                                ordinal));
                        v = static_cast<HighPrecType>(wEmul) * emul
                            + static_cast<HighPrecType>(wFix) * fixed;
                    }
                    pRow[x] = static_cast<AlgoType>(v);
                }
            }
        }
    }

    // ----------------------------------------------------------------------
    //  THE FACTORED EVALUATION.
    //
    //      M = sum_k w_k B(s_k)[white]
    //      L = M + lobe_gain * B(s_lobe)[M]
    //      F = B(s_scan)[L]
    //
    //  ⚠⚠ THIS IS THE CHANGE THAT BROUGHT THE STAGE BACK INSIDE ITS BUDGET.
    //  The previous form ran one full-plane blur per EXPANDED term with the
    //  scan variance folded into every sigma: measured here, at 3840x2160, one
    //  channel, one thread, 398 ms for a ten-term stock and 120 ms for a
    //  five-term one, against R-N3's 8 ms and 43 ms for the legacy path.
    //
    //  A product of Gaussian transfers is a Gaussian whose variances add and
    //  convolution distributes over a sum, so the lobe and the band limit each
    //  become ONE blur of the accumulated mixture rather than a contribution to
    //  every term -- exactly, not approximately. That in turn exposes the bare
    //  grain sigmas, 0.289 down to 0.018 px at 4K, four of five of which ARE
    //  the identity and cost a scalar multiply instead of two passes.
    //
    //  ⚠ THE WHITE FIELD IS DRAWN ONCE AND REUSED. The mixture sums filtered
    //  copies of ONE field; independent draws per term would add VARIANCES
    //  instead of AMPLITUDES and give the sum of the squares of the weights
    //  rather than the square of their sum.
    //
    //  ⚠ pScrNoise MUST SURVIVE THE LOOP, so the blur writes to pScrLobe and
    //  never back over the noise plane.
    // ----------------------------------------------------------------------
    const int32_t wideX = sizeX & ~7;
    const int32_t tailX = sizeX - wideX;

    AlgoType flatWeight = ALGO_ZERO;
    bool     haveAcc    = false;

    for (int32_t k = 0; k < terms.count; k++)
    {
        const HighPrecType sPx = static_cast<HighPrecType>(terms.sigma_mm[k])
                               * static_cast<HighPrecType>(pxPerMm);

        const AlgoType wgt = static_cast<AlgoType>(terms.weight[k]);

        if (AlgoGrainBlurIsIdentity(sPx)
            && AlgoGrainBlurIsIdentity(sPx * static_cast<HighPrecType>(aniso)))
        {
            flatWeight += wgt;
            continue;
        }

        const AlgoType sigmaX = static_cast<AlgoType>(sPx);

        AlgoGaussianBlurPlaneWrapXY(pScrNoise, pScrLobe, pScrWork,
                                    sizeX, sizeY, pitch,
                                    sigmaX, sigmaX * aniso);

        const __m256 vW = _mm256_set1_ps(wgt);

        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pS = pScrLobe + off;
            AlgoType* RESTRICT       pA = pDst     + off;

            int32_t x = 0;

            if (!haveAcc)
            {
                for (; x < wideX; x += 8)
                    _mm256_storeu_ps(pA + x,
                                     _mm256_mul_ps(vW, _mm256_loadu_ps(pS + x)));

                if (tailX > 0)
                {
                    const __m256i msk = algoTailMaskLocal(tailX);

                    _mm256_maskstore_ps(
                        pA + x, msk,
                        _mm256_mul_ps(vW, _mm256_maskload_ps(pS + x, msk)));
                }
            }
            else
            {
                for (; x < wideX; x += 8)
                    _mm256_storeu_ps(
                        pA + x,
                        _mm256_fmadd_ps(vW, _mm256_loadu_ps(pS + x),
                                        _mm256_loadu_ps(pA + x)));

                if (tailX > 0)
                {
                    const __m256i msk = algoTailMaskLocal(tailX);

                    _mm256_maskstore_ps(
                        pA + x, msk,
                        _mm256_fmadd_ps(vW, _mm256_maskload_ps(pS + x, msk),
                                        _mm256_maskload_ps(pA + x, msk)));
                }
            }
        }

        haveAcc = true;
    }

    if (flatWeight != ALGO_ZERO)
    {
        const __m256 vF = _mm256_set1_ps(flatWeight);

        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pN = pScrNoise + off;
            AlgoType* RESTRICT       pA = pDst      + off;

            int32_t x = 0;

            if (!haveAcc)
            {
                for (; x < wideX; x += 8)
                    _mm256_storeu_ps(pA + x,
                                     _mm256_mul_ps(vF, _mm256_loadu_ps(pN + x)));

                if (tailX > 0)
                {
                    const __m256i msk = algoTailMaskLocal(tailX);

                    _mm256_maskstore_ps(
                        pA + x, msk,
                        _mm256_mul_ps(vF, _mm256_maskload_ps(pN + x, msk)));
                }
            }
            else
            {
                for (; x < wideX; x += 8)
                    _mm256_storeu_ps(
                        pA + x,
                        _mm256_fmadd_ps(vF, _mm256_loadu_ps(pN + x),
                                        _mm256_loadu_ps(pA + x)));

                if (tailX > 0)
                {
                    const __m256i msk = algoTailMaskLocal(tailX);

                    _mm256_maskstore_ps(
                        pA + x, msk,
                        _mm256_fmadd_ps(vF, _mm256_maskload_ps(pN + x, msk),
                                        _mm256_maskload_ps(pA + x, msk)));
                }
            }
        }

        haveAcc = true;
    }

    if (!haveAcc)
    {
        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow =
                pDst + static_cast<std::ptrdiff_t>(y) * pitch;

            ALGO_VECTOR_HINT
            for (int32_t x = 0; x < sizeX; x++)
                pRow[x] = ALGO_ZERO;
        }
    }

    // ---- the clustering lobe, as one blur of the accumulated mixture -------
    if ((terms.lobe_gain > 0.0f) && (terms.lobe_sigma_mm > 0.0f))
    {
        const HighPrecType sPx = static_cast<HighPrecType>(terms.lobe_sigma_mm)
                               * static_cast<HighPrecType>(pxPerMm);

        const AlgoType g = static_cast<AlgoType>(terms.lobe_gain);

        if (AlgoGrainBlurIsIdentity(sPx)
            && AlgoGrainBlurIsIdentity(sPx * static_cast<HighPrecType>(aniso)))
        {
            const __m256 vM = _mm256_set1_ps(ALGO_ONE + g);

            for (int32_t y = 0; y < sizeY; y++)
            {
                AlgoType* RESTRICT pA =
                    pDst + static_cast<std::ptrdiff_t>(y) * pitch;

                int32_t x = 0;

                for (; x < wideX; x += 8)
                    _mm256_storeu_ps(pA + x,
                                     _mm256_mul_ps(vM, _mm256_loadu_ps(pA + x)));

                if (tailX > 0)
                {
                    const __m256i msk = algoTailMaskLocal(tailX);

                    _mm256_maskstore_ps(
                        pA + x, msk,
                        _mm256_mul_ps(vM, _mm256_maskload_ps(pA + x, msk)));
                }
            }
        }
        else
        {
            const AlgoType sigmaX = static_cast<AlgoType>(sPx);

            AlgoGaussianBlurPlaneWrapXY(pDst, pScrLobe, pScrWork,
                                        sizeX, sizeY, pitch,
                                        sigmaX, sigmaX * aniso);

            const __m256 vG = _mm256_set1_ps(g);

            for (int32_t y = 0; y < sizeY; y++)
            {
                const std::ptrdiff_t off =
                    static_cast<std::ptrdiff_t>(y) * pitch;

                const AlgoType* RESTRICT pS = pScrLobe + off;
                AlgoType* RESTRICT       pA = pDst     + off;

                int32_t x = 0;

                for (; x < wideX; x += 8)
                    _mm256_storeu_ps(
                        pA + x,
                        _mm256_fmadd_ps(vG, _mm256_loadu_ps(pS + x),
                                        _mm256_loadu_ps(pA + x)));

                if (tailX > 0)
                {
                    const __m256i msk = algoTailMaskLocal(tailX);

                    _mm256_maskstore_ps(
                        pA + x, msk,
                        _mm256_fmadd_ps(vG, _mm256_maskload_ps(pS + x, msk),
                                        _mm256_maskload_ps(pA + x, msk)));
                }
            }
        }
    }

    // ---- the scan band limit, as one blur of the result --------------------
    {
        if (!(AlgoGrainBlurIsIdentity(scanPx)
              && AlgoGrainBlurIsIdentity(scanPx
                                         * static_cast<HighPrecType>(aniso))))
        {
            const AlgoType sigmaX = static_cast<AlgoType>(scanPx);

            AlgoGaussianBlurPlaneWrapXY(pDst, pScrLobe, pScrWork,
                                        sizeX, sizeY, pitch,
                                        sigmaX, sigmaX * aniso);

            AlgoCopyPlane(pScrLobe, pDst, sizeX, sizeY, pitch);
        }
    }

    // ----------------------------------------------------------------------
    //  Force zero mean, then apply the amplitude.
    //
    //  The reference zeroes the DC bin of the transfer, which is exactly a removal
    //  of the mean. It matters: a field with a non-zero mean would shift the
    //  overall density of the frame, so the grain control would double as an
    //  exposure control.
    // ----------------------------------------------------------------------
    const AlgoType mean = static_cast<AlgoType>(
        AlgoPlaneMean(pDst, sizeX, sizeY, pitch));

    for (int32_t y = 0; y < sizeY; y++)
    {
        AlgoType* RESTRICT pRow = pDst + static_cast<std::ptrdiff_t>(y) * pitch;

        // Zero-mean and scale. The mean came from AlgoPlaneMean, which keeps its
        // accumulator wide on purpose - see that function.
        const __m256 vMean  = _mm256_set1_ps(mean);
        const __m256 vScale = _mm256_set1_ps(scale);

        const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
        const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
        const __m256i mt = algoTailMaskLocal(nt);

        int32_t x = 0;

        for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
            _mm256_storeu_ps(pRow + x, _mm256_mul_ps(
                _mm256_sub_ps(_mm256_loadu_ps(pRow + x), vMean), vScale));

        if (nt > 0)
            _mm256_maskstore_ps(pRow + x, mt, _mm256_mul_ps(
                _mm256_sub_ps(_mm256_maskload_ps(pRow + x, mt), vMean), vScale));
    }

    return;
}


// ---------------------------------------------------------------------------
//  The amplitude evaluator, eight lanes at a time.
//
//  Same struct, same law, same numbers as the scalar path -- only the execution
//  differs, which is the project's rule for a twin. The struct itself is built
//  by the SHARED AlgoGrainAmpBuild() in AlgoGrain.hpp, so the two paths cannot
//  drift in the model even if this loop is rewritten.
// ---------------------------------------------------------------------------
static inline __m256 algoGrainAmpVec
(
    const AlgoGrainAmp& a,
    const __m256        d
) noexcept
{
    if (!a.measured)
    {
        // _mm256_sqrt_ps is a REAL instruction, so this is the rare
        // transcendental in the engine that needs no approximation and no
        // accuracy trade - eight square roots for the price of one, exactly
        // rounded. The floor at zero is a max, not a branch: a negative
        // developed density is physically meaningless and its square root
        // would be a NaN that would propagate through every stage after this.
        const __m256 developed =
            _mm256_max_ps(_mm256_sub_ps(d, _mm256_set1_ps(a.dmin)),
                          _mm256_setzero_ps());

        const __m256 root =
            _mm256_sqrt_ps(_mm256_add_ps(developed, _mm256_set1_ps(a.fog)));

        // ampScale pins the result to exactly 1.0 at NET density 1.0 -- the
        // density the datasheets quote rms_granularity at. Without it this
        // stage ran 1.0392x to 1.1832x loud on every stock (queue C30/C33).
        return _mm256_mul_ps(root, _mm256_set1_ps(a.ampScale));
    }

    // Piecewise linear over at most three segments, held flat outside the
    // traced range. Walked as a cascade of blends rather than a branch: the
    // eight lanes generally fall in different segments, so there is nothing to
    // branch on. Each blend overwrites only the lanes that have passed the
    // segment's lower edge, and because the anchors ascend the LAST segment a
    // lane qualifies for is the one that survives.
    __m256 v = _mm256_set1_ps(a.loY);

    for (int32_t i = 1; i < a.n; i++)
    {
        const __m256 seg = _mm256_fmadd_ps(_mm256_set1_ps(a.slope[i]), d,
                                           _mm256_set1_ps(a.icept[i]));

        const __m256 over = _mm256_cmp_ps(d, _mm256_set1_ps(a.xs[i - 1]),
                                          _CMP_GT_OQ);

        v = _mm256_blendv_ps(v, seg, over);
    }

    const __m256 above = _mm256_cmp_ps(d, _mm256_set1_ps(a.xs[a.n - 1]),
                                       _CMP_GE_OQ);

    return _mm256_blendv_ps(v, _mm256_set1_ps(a.hiY), above);
}


// ---------------------------------------------------------------------------
//  One channel of the add, given a prepared evaluator.
// ---------------------------------------------------------------------------
static inline void algoAddGrainPlane
(
    AlgoType* RESTRICT       pD,
    const AlgoType* RESTRICT pF,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const AlgoGrainAmp&      a,
    const AlgoType           gain
) noexcept
{
    const __m256  vGain = _mm256_set1_ps(gain);
    const int32_t nv    = sizeX / ALGO_AVX2_LANES_LOCAL;
    const int32_t nt    = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
    const __m256i mt    = algoTailMaskLocal(nt);

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        AlgoType* RESTRICT       rD = pD + off;
        const AlgoType* RESTRICT rF = pF + off;

        int32_t x = 0;

        // Unaligned loads and stores throughout: the image buffers carry no
        // alignment guarantee, and the row padding must stay untouched.
        for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
        {
            const __m256 d   = _mm256_loadu_ps(rD + x);
            const __m256 amp = algoGrainAmpVec(a, d);

            const __m256 add = _mm256_mul_ps(
                _mm256_mul_ps(vGain, _mm256_loadu_ps(rF + x)), amp);

            _mm256_storeu_ps(rD + x, _mm256_add_ps(d, add));
        }

        // Mandatory scalar tail, expressed as a masked vector so the partial
        // lane count needs no separate arithmetic path.
        if (nt > 0)
        {
            const __m256 d   = _mm256_maskload_ps(rD + x, mt);
            const __m256 amp = algoGrainAmpVec(a, d);

            const __m256 add = _mm256_mul_ps(
                _mm256_mul_ps(vGain, _mm256_maskload_ps(rF + x, mt)), amp);

            _mm256_maskstore_ps(rD + x, mt, _mm256_add_ps(d, add));
        }
    }
}


// ---------------------------------------------------------------------------
//  Build one grain field -- LEGACY (clumpUm, clumpGain) entry point.
//
//  Kept because stages 13 and 14 have no physical grain diameter to offer: a
//  duplicating or print stock's grain_rms is a fitted look, not a published
//  datasheet figure, so inverting it to a diameter would manufacture physics.
//  They stay on the legacy spectrum deliberately, and this is how they reach it.
//
//  ⚠ NOT A SECOND IMPLEMENTATION. It builds the two exact legacy terms and
//  hands them to the one field builder above.
// ---------------------------------------------------------------------------
void AlgoMakeGrainField
(
    AlgoType* RESTRICT          pDst,
    AlgoType* RESTRICT          pScrNoise,
    AlgoType* RESTRICT          pScrLobe,
    AlgoType* RESTRICT          pScrWork,
    const int32_t               sizeX,
    const int32_t               sizeY,
    const int32_t               pitch,
    const AlgoType              clumpUm,
    const AlgoType              clumpGain,
    const AlgoType              rmsGranularity,
    const AlgoType              scanSigmaPx,
    const AlgoType              pxPerMm,
    const AlgoType              anisotropy,
    const eALGO_RNG_STAGE       rngStage,
    const uint32_t              seed,
    const int32_t               frameIndex
) noexcept
{
    AlgoMakeGrainFieldTerms(pDst, pScrNoise, pScrLobe, pScrWork,
                            sizeX, sizeY, pitch,
                            AlgoGrainLegacyTerms(clumpUm, clumpGain),
                            false,               // legacy Gaussian aperture
                            rmsGranularity, scanSigmaPx, pxPerMm, anisotropy,
                            rngStage, seed, frameIndex);
    return;
}


// ---------------------------------------------------------------------------
//  Add grain fields to density
// ---------------------------------------------------------------------------
void AlgoAddGrain
(
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const AlgoType* RESTRICT pFieldR,
    const AlgoType* RESTRICT pFieldG,
    const AlgoType* RESTRICT pFieldB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const AlgoType           dmin[3],
    const AlgoType           dmax[3],
    const film::GrainSpec&   grain,
    const AlgoType           gain
) noexcept
{
    AlgoType* RESTRICT       dstPlane[3] = { pDstR,   pDstG,   pDstB   };
    const AlgoType* RESTRICT fldPlane[3] = { pFieldR, pFieldG, pFieldB };

    for (int32_t c = 0; c < 3; c++)
    {
        // Setup domain: three times per render, never per pixel. ampScale and
        // the segment coefficients are computed in HighPrecType by the shared
        // builder, so this path and the scalar one start from bit-identical
        // constants.
        const AlgoGrainAmp amp = AlgoGrainAmpBuild(grain, dmin[c], dmax[c]);

        algoAddGrainPlane(dstPlane[c], fldPlane[c],
                          sizeX, sizeY, pitch, amp, gain);
    }

    return;
}


// ---------------------------------------------------------------------------
//  Add grain fields to density, UNPINNED -- print and duplication stocks.
//
//  ⚠ No ampScale here, deliberately: print and dupe grain carry no published
//  rms to be pinned to. See AlgoGrainAmpRaw in AlgoGrain.hpp.
// ---------------------------------------------------------------------------
void AlgoAddGrainRaw
(
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const AlgoType* RESTRICT pFieldR,
    const AlgoType* RESTRICT pFieldG,
    const AlgoType* RESTRICT pFieldB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const AlgoType           dmin[3],
    const AlgoType           fogGrain,
    const AlgoType           gain
) noexcept
{
    AlgoType* RESTRICT       dstPlane[3] = { pDstR,   pDstG,   pDstB   };
    const AlgoType* RESTRICT fldPlane[3] = { pFieldR, pFieldG, pFieldB };

    for (int32_t c = 0; c < 3; c++)
    {
        const AlgoGrainAmp amp = AlgoGrainAmpRaw(dmin[c], fogGrain);

        algoAddGrainPlane(dstPlane[c], fldPlane[c],
                          sizeX, sizeY, pitch, amp, gain);
    }

    return;
}


// ---------------------------------------------------------------------------
//  Stage 11: grain
// ---------------------------------------------------------------------------
void AlgoStage11_Grain
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrNoise,
    AlgoType* RESTRICT       pScrLobe,
    AlgoType* RESTRICT       pScrWork,
    AlgoType* RESTRICT       pScrFieldR,
    AlgoType* RESTRICT       pScrFieldG,
    AlgoType* RESTRICT       pScrFieldB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           scanSigmaPx,
    const AlgoType           pxPerMm,
    const bool               hasMosaic,
    const int32_t            frameIndex,
    const uint32_t           seed
) noexcept
{
    const film::GrainSpec& gs = profile.grain;

    const AlgoType gain = MAX_VALUE(static_cast<AlgoType>(params.grainScale),
                                    ALGO_ZERO);

    // Copy first, then add in place. Grain is additive, so a destination already
    // holding the incoming densities is the natural starting state, and the copy
    // also satisfies the retained-buffer policy when grain is switched off.
    AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    if (gain <= ALGO_ZERO)
        return;

    // Seed for this stage. Named distinctly from the parameter: a local that
    // shadowed it and was XORed with itself would be undefined behaviour, which
    // this engine has already done once.
    const uint32_t grainSeed = static_cast<uint32_t>(params.seed) ^ seed;

    // ----------------------------------------------------------------------
    //  ⚠⚠ NO scanner_fixed_pattern FREEZE, AND ITS ABSENCE IS THE POINT.
    //
    //  Spec §18.2 defines GrainSpec::grain_temporal_class so that the grain
    //  stage "does not re-roll per frame what is physically static" -- i.e. it
    //  asks for the frame index to be pinned for such a stock. R-T5 says the
    //  grain stage "shall introduce no frame-locked noise component" and that
    //  scanner fixed-pattern noise "is not grain and is out of scope".
    //
    //  A frozen grain field IS a frame-locked noise component, so the two
    //  clauses cannot both be honoured. R-T5 wins: it is a requirement, §18.2
    //  is a schema note, and the freeze is the wrong remedy anyway -- if a
    //  stock's traced noise is the scanner's then its rms_granularity is not
    //  emulsion granularity, and freezing renders the wrong quantity very
    //  steadily instead of fixing the measurement underneath.
    //
    //  The field is therefore DECLARATIVE: film_profiles.validate refuses any
    //  value but "emulsion", and this stage reads frameIndex unconditionally.
    // ----------------------------------------------------------------------
    const int32_t grainFrame = frameIndex;

    // Base plus fog per channel, needed by the amplitude weighting.
    const AlgoType dmin[3] =
    {
        static_cast<AlgoType>(profile.curves.r.dmin),
        static_cast<AlgoType>(profile.curves.g.dmin),
        static_cast<AlgoType>(profile.curves.b.dmin)
    };

    // Asymptotic maximum density per channel. ⚠ ONLY the measured sigma(D)
    // branch reads this, and only as a FALLBACK: a traced shape stores its own
    // endpoints (sigma_shape_toe_at / _dmax_at) because the granularity plot and
    // the sensitometric plot are made on different equipment and disagree about
    // where the film starts and stops -- Kodak's own footnote says so, and the
    // VISION3 sheets differ by a mean +0.051 D.
    const AlgoType dmax[3] =
    {
        static_cast<AlgoType>(profile.curves.r.dmax()),
        static_cast<AlgoType>(profile.curves.g.dmax()),
        static_cast<AlgoType>(profile.curves.b.dmax())
    };

    // ----------------------------------------------------------------------
    //  ONE EMULSION MEANS ONE FIELD.
    //
    //  A monochrome stock has a single silver image. So does an additive colour
    //  stock: one panchromatic emulsion behind the filter grid, which cannot have
    //  per-layer grain.
    //
    //  hasMosaic rather than profile.has_reseau, deliberately. Stage 7 skips the
    //  mosaic when the grid cannot be resolved at this render size and falls back
    //  to three ordinary records; the grain has to follow the same decision, or a
    //  low-resolution Dufaycolor render would get three independent fields on what
    //  is really one record.
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    //  The count gate needs the noise-equivalent element area, which depends on
    //  this render's band limit and pixel pitch and nothing else. One setup
    //  quadrature per stage, not per channel and never per pixel.
    // ----------------------------------------------------------------------
    const HighPrecType pitchMm = (pxPerMm > ALGO_ZERO)
        ? (1.0 / static_cast<HighPrecType>(pxPerMm)) : 0.0;

    const HighPrecType scanSigmaMm = (pxPerMm > ALGO_ZERO)
        ? (static_cast<HighPrecType>(scanSigmaPx)
           / static_cast<HighPrecType>(pxPerMm)) : 0.0;

    const HighPrecType elemArea =
        AlgoGrainElementAreaUm2(scanSigmaMm, pitchMm);

    const HighPrecType grainUm[3] =
    {
        static_cast<HighPrecType>(gs.grain_um_r),
        static_cast<HighPrecType>(gs.grain_um_g),
        static_cast<HighPrecType>(gs.grain_um_b)
    };

    const HighPrecType sigmaLn = static_cast<HighPrecType>(gs.size_sigma_log);

    if (profile.is_monochrome || hasMosaic)
    {
        // The green clump figure stands for the single emulsion, matching the
        // reference: a monochrome stock's three clump fields carry the same number,
        // and green is the one the metric is quoted against.
        AlgoMakeGrainFieldTerms(pScrFieldR, pScrNoise, pScrLobe, pScrWork,
                                sizeX, sizeY, pitch,
                                AlgoGrainTermsFor(gs, 1, ALGO_GRAIN_USE_JINC),
                                ALGO_GRAIN_USE_JINC,
                                static_cast<AlgoType>(gs.rms_granularity),
                                scanSigmaPx, pxPerMm,
                                static_cast<AlgoType>(gs.anisotropy),
                                eALGO_RNG_STAGE::eRNG_GRAIN_G,
                                grainSeed, grainFrame,
                                static_cast<AlgoType>(
                                    profile.temporal.grain_frame_correlation),
                                static_cast<AlgoType>(
                                    params.scannerFixedPattern));

        // ⚠ THE MARGINAL CORRECTION IS APPLIED ONCE, to the single shared
        // field, and against the GREEN record's density. A monochrome stock
        // has one emulsion, so it has one grain count; correcting the same
        // field three times against three densities would be three different
        // answers about one silver image.
        AlgoGrainApplyMarginal(pScrFieldR, pDstG, sizeX, sizeY, pitch,
                               dmin[1], grainUm[1], sigmaLn, elemArea);

        // The same field three times. Three independent fields here would produce
        // coloured speckle on a black-and-white image.
        AlgoAddGrain(pDstR, pDstG, pDstB,
                     pScrFieldR, pScrFieldR, pScrFieldR,
                     sizeX, sizeY, pitch, dmin, dmax, gs, gain);
    }
    else
    {
        // ------------------------------------------------------------------
        //  Tripack: three separate emulsions, three independent fields.
        //
        //  Per-channel RMS where the profile overrides it, otherwise the scalar
        //  figure. This is where a tripack's blue layer gets its extra noise - it
        //  is on top and it is the fastest - and where a three-strip process's three
        //  physically different black-and-white records diverge.
        // ------------------------------------------------------------------
        const AlgoType rmsScalar = static_cast<AlgoType>(gs.rms_granularity);

        const AlgoType rms[3] =
        {
            (gs.rms_r > 0.0f) ? static_cast<AlgoType>(gs.rms_r) : rmsScalar,
            (gs.rms_g > 0.0f) ? static_cast<AlgoType>(gs.rms_g) : rmsScalar,
            (gs.rms_b > 0.0f) ? static_cast<AlgoType>(gs.rms_b) : rmsScalar
        };

        const eALGO_RNG_STAGE stream[3] =
        {
            eALGO_RNG_STAGE::eRNG_GRAIN_R,
            eALGO_RNG_STAGE::eRNG_GRAIN_G,
            eALGO_RNG_STAGE::eRNG_GRAIN_B
        };

        AlgoType* RESTRICT field[3] = { pScrFieldR, pScrFieldG, pScrFieldB };

        // Separate generator streams per channel, so the three fields are
        // statistically independent rather than three views of one field.
        for (int32_t c = 0; c < 3; c++)
            AlgoMakeGrainFieldTerms(field[c], pScrNoise, pScrLobe, pScrWork,
                                    sizeX, sizeY, pitch,
                                    AlgoGrainTermsFor(gs, c, ALGO_GRAIN_USE_JINC),
                                    ALGO_GRAIN_USE_JINC,
                                    rms[c], scanSigmaPx, pxPerMm,
                                    static_cast<AlgoType>(gs.anisotropy),
                                    stream[c], grainSeed, grainFrame,
                                    static_cast<AlgoType>(
                                        profile.temporal
                                            .grain_frame_correlation),
                                    static_cast<AlgoType>(
                                        params.scannerFixedPattern));

        // Three emulsions, three counts, three corrections.
        AlgoType* RESTRICT dstPl[3] = { pDstR, pDstG, pDstB };

        for (int32_t c = 0; c < 3; c++)
            AlgoGrainApplyMarginal(field[c], dstPl[c], sizeX, sizeY, pitch,
                                   dmin[c], grainUm[c], sigmaLn, elemArea);

        AlgoAddGrain(pDstR, pDstG, pDstB,
                     pScrFieldR, pScrFieldG, pScrFieldB,
                     sizeX, sizeY, pitch, dmin, dmax, gs, gain);
    }

    // ----------------------------------------------------------------------
    //  Floor at zero.
    //
    //  The field is zero-mean, so half of it is negative and a light area can be
    //  driven below base. Negative optical density has no physical meaning and
    //  stage 14 raises ten to its negative, which would give a transmittance above
    //  one. A physical floor, not a display clamp.
    // ----------------------------------------------------------------------
    AlgoType* RESTRICT dstPlane[3] = { pDstR, pDstG, pDstB };

    for (int32_t c = 0; c < 3; c++)
    {
        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT pRow =
                dstPlane[c] + static_cast<std::ptrdiff_t>(y) * pitch;

            // Non-negative floor. A max, not a branch: a negative optical density
            // would be a material that emits light, and the stages downstream take
            // its logarithm or its square root.
            {
                const __m256 vZero = _mm256_setzero_ps();

                const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
                const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
                const __m256i mt = algoTailMaskLocal(nt);

                int32_t x = 0;

                for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
                    _mm256_storeu_ps(pRow + x,
                        _mm256_max_ps(_mm256_loadu_ps(pRow + x), vZero));

                if (nt > 0)
                    _mm256_maskstore_ps(pRow + x, mt,
                        _mm256_max_ps(_mm256_maskload_ps(pRow + x, mt), vZero));
            }
        }
    }

    return;
}
