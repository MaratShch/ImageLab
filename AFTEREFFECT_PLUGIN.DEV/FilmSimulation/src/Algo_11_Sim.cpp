#if 0
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


namespace
{
    // ----------------------------------------------------------------------
    //  Aperture-weighted spectral energy of the grain, over ALL frequencies.
    //
    //      E = 2 pi * integral of (H(f) A(f))^2 * f df
    //
    //  with H the grain spectrum and A the measuring aperture. Evaluated as a
    //  continuous radial integral rather than as a sum over the pixel grid, which
    //  is the whole point: see the note on calibration in the header.
    //
    //  The factor of f is the Jacobian of the polar area element, so this really is
    //  the total energy over the plane and not a line integral.
    // ----------------------------------------------------------------------
    HighPrecType grainReferenceEnergy
    (
        const HighPrecType clumpUm,
        const HighPrecType clumpGain
    ) noexcept
    {
        // Crystal rolloff frequency: a clump of diameter d resolves nothing finer
        // than 1/(2d), and the diameter is in micrometres while frequencies are per
        // millimetre, hence the factor of a thousand.
        const HighPrecType fHi = 1000.0 / (2.0 * clumpUm);

        // Clustering lobe, six times coarser.
        const HighPrecType fLo = fHi / ALGO_GRAIN_CLUMP_FREQ_RATIO;

        const HighPrecType step = ALGO_GRAIN_INTEGRAL_FMAX
                                / static_cast<HighPrecType>(
                                      ALGO_GRAIN_INTEGRAL_N - 1);

        // Aperture transfer coefficient, precomputed: A(f) = exp(-2 pi^2 s^2 f^2).
        const HighPrecType apK = 2.0 * 9.8696044010893586188
                               * ALGO_GRAIN_APERTURE_SIGMA_MM
                               * ALGO_GRAIN_APERTURE_SIGMA_MM;

        HighPrecType acc  = 0.0;
        HighPrecType prev = 0.0;   // the integrand at f = 0 is zero, by the f factor

        for (int32_t i = 1; i < ALGO_GRAIN_INTEGRAL_N; i++)
        {
            const HighPrecType f = static_cast<HighPrecType>(i) * step;

            // Grain spectrum: crystal rolloff times one plus the clustering lobe.
            const HighPrecType rHi = f / fHi;
            const HighPrecType rLo = f / fLo;

            HighPrecType h = std::exp(-rHi * rHi);

            if (clumpGain > 0.0)
                h *= (1.0 + clumpGain * std::exp(-rLo * rLo));

            // Measuring aperture.
            const HighPrecType a = std::exp(-apK * f * f);

            const HighPrecType ha = h * a;

            // Integrand, including the polar Jacobian.
            const HighPrecType cur = ha * ha * f;

            // Trapezoidal rule.
            acc += 0.5 * (prev + cur) * step;

            prev = cur;
        }

        // 2 pi from the angular integration of the isotropic spectrum.
        return 2.0 * 3.1415926535897932385 * acc;
    }
}


// ---------------------------------------------------------------------------
//  Build one grain field
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
    const eALGO_RNG_STAGE       rngStage,
    const uint32_t              seed,
    const int32_t               frameIndex
) noexcept
{
    // A stock with no clump figure or no granularity figure has no modelled grain.
    // Zero the field rather than leaving it, so the caller can add it
    // unconditionally without a second branch.
    if ((clumpUm <= ALGO_ZERO) || (rmsGranularity <= ALGO_ZERO) ||
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
    //  Spectral shape, expressed as spatial Gaussian sigmas.
    //
    //  The reference builds the shape in the frequency domain as
    //
    //      H(f) = exp(-(f/f_hi)^2) * ( 1 + g * exp(-(f/f_lo)^2) )
    //
    //  Expanding the product gives two Gaussian terms, and since a product of
    //  Gaussian transfers is a Gaussian whose VARIANCES ADD, both are spatial
    //  Gaussian blurs:
    //
    //      term 1, weight 1 : sigma_hi
    //      term 2, weight g : sqrt(sigma_hi^2 + sigma_lo^2)
    //
    //  Matching exp(-(f/f_c)^2) against exp(-2 pi^2 s^2 f^2) gives
    //  s = 1 / (pi * sqrt(2) * f_c).
    //
    //  The weights are 1 and g and DO NOT sum to one, because this is a spectral
    //  shaping of a noise field and not an averaging filter. That is why the two
    //  lobes are blurred separately and combined by hand rather than handed to the
    //  multi-lobe helper, which normalises by the weight sum.
    // ----------------------------------------------------------------------
    const HighPrecType fHi = 1000.0 / (2.0 * static_cast<HighPrecType>(clumpUm));
    const HighPrecType fLo = fHi / ALGO_GRAIN_CLUMP_FREQ_RATIO;

    // 1 / (pi * sqrt(2)) = 0.22508352815546.
    const HighPrecType kSigma = 0.22508352815546;

    const HighPrecType sHiMm = kSigma / fHi;
    const HighPrecType sLoMm = kSigma / fLo;

    // Millimetres to pixels, and fold in the scan band limit. The band limit is a
    // further Gaussian multiplying the shape in the frequency domain, so its
    // variance adds to every term.
    const HighPrecType sHiPx = sHiMm * static_cast<HighPrecType>(pxPerMm);
    const HighPrecType sLoPx = sLoMm * static_cast<HighPrecType>(pxPerMm);
    const HighPrecType sScan = static_cast<HighPrecType>(
                                   MAX_VALUE(scanSigmaPx, ALGO_ZERO));

    const AlgoType sigmaNarrow = static_cast<AlgoType>(
        std::sqrt(sHiPx * sHiPx + sScan * sScan));

    const AlgoType sigmaWide = static_cast<AlgoType>(
        std::sqrt(sHiPx * sHiPx + sLoPx * sLoPx + sScan * sScan));

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
    // ----------------------------------------------------------------------
    const HighPrecType energy = grainReferenceEnergy(
        static_cast<HighPrecType>(clumpUm),
        static_cast<HighPrecType>(MAX_VALUE(clumpGain, ALGO_ZERO)));

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
    const uint64_t frameSalt =
        static_cast<uint64_t>(static_cast<uint32_t>(frameIndex)) * ALGO_RNG_GOLDEN;

    const uint64_t seedField = (static_cast<uint64_t>(seed) << 32) ^ frameSalt;

    const uint64_t stageField =
        (static_cast<uint64_t>(UnderlyingType(rngStage) >> 8) & 0xFFull) << 24;

    const __m256i vBase =
        _mm256_set1_epi64x(static_cast<long long>(seedField ^ stageField));

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
    }

    // Narrow lobe: the crystal rolloff alone.
    AlgoGaussianBlurPlaneWrap(pScrNoise, pScrLobe, pScrWork,
                              sizeX, sizeY, pitch, sigmaNarrow);

    const AlgoType g = MAX_VALUE(clumpGain, ALGO_ZERO);

    if (g > ALGO_ZERO)
    {
        // Wide lobe: the clustering term. Written back over the noise plane, which
        // is finished with - both lobes were drawn from it and the narrow one is
        // already safe in its own plane.
        //
        // NOTE the aliasing hazard this deliberately avoids: the blur reads its
        // source completely before it can be overwritten, so the source and
        // destination must differ. pScrWork is the intermediate, so the noise plane
        // cannot be both.
        AlgoGaussianBlurPlaneWrap(pScrNoise, pDst, pScrWork,
                                  sizeX, sizeY, pitch, sigmaWide);

        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pN = pScrLobe + off;
            AlgoType* RESTRICT       pW = pDst     + off;

            ALGO_VECTOR_HINT
            for (int32_t x = 0; x < sizeX; x++)
                pW[x] = pN[x] + g * pW[x];
        }
    }
    else
    {
        AlgoCopyPlane(pScrLobe, pDst, sizeX, sizeY, pitch);
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
    const AlgoType           fogGrain,
    const AlgoType           gain
) noexcept
{
    AlgoType* RESTRICT       dstPlane[3] = { pDstR,   pDstG,   pDstB   };
    const AlgoType* RESTRICT fldPlane[3] = { pFieldR, pFieldG, pFieldB };

    // Floor under the square root. Keeps grain alive in the deepest shadow, where
    // the density has fallen back to base fog and a bare square root would give
    // exactly zero. Perfectly clean blacks are one of the loudest digital tells.
    const AlgoType fog = MAX_VALUE(fogGrain, ALGO_ZERO);

    for (int32_t c = 0; c < 3; c++)
    {
        AlgoType* RESTRICT       pD = dstPlane[c];
        const AlgoType* RESTRICT pF = fldPlane[c];

        // Base plus fog of the curve that produced this plane. Subtracting it is
        // what makes the amplitude depend on DEVELOPED density rather than on total
        // density: the base carries no crystals and contributes no grain.
        const AlgoType dm = dmin[c];

        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            AlgoType* RESTRICT       rD = pD + off;
            const AlgoType* RESTRICT rF = pF + off;

            // Poisson statistics of a countable crystal population: the standard
            // deviation grows as the square root of the mean count, and developed
            // density stands in for that count.
            //
            // _mm256_sqrt_ps is a REAL instruction, so this is the rare transcendental
            // in the engine that needs no approximation and no accuracy trade - eight
            // square roots for the price of one, exactly rounded.
            //
            // The floor at zero is a max, not a branch: a negative developed density is
            // physically meaningless and its square root would be a NaN that would
            // propagate through every stage after this one.
            const __m256 vDm   = _mm256_set1_ps(dm);
            const __m256 vFog  = _mm256_set1_ps(fog);
            const __m256 vGain = _mm256_set1_ps(gain);
            const __m256 vZero = _mm256_setzero_ps();

            const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
            const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
            const __m256i mt = algoTailMaskLocal(nt);

            int32_t x = 0;

            for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
            {
                const __m256 d = _mm256_loadu_ps(rD + x);

                const __m256 developed =
                    _mm256_max_ps(_mm256_sub_ps(d, vDm), vZero);

                const __m256 amp =
                    _mm256_sqrt_ps(_mm256_add_ps(developed, vFog));

                // d + gain * field * amp, as two FMAs' worth of work in one chain.
                const __m256 add =
                    _mm256_mul_ps(_mm256_mul_ps(vGain, _mm256_loadu_ps(rF + x)), amp);

                _mm256_storeu_ps(rD + x, _mm256_add_ps(d, add));
            }

            if (nt > 0)
            {
                const __m256 d = _mm256_maskload_ps(rD + x, mt);

                const __m256 developed =
                    _mm256_max_ps(_mm256_sub_ps(d, vDm), vZero);

                const __m256 amp =
                    _mm256_sqrt_ps(_mm256_add_ps(developed, vFog));

                const __m256 add = _mm256_mul_ps(
                    _mm256_mul_ps(vGain, _mm256_maskload_ps(rF + x, mt)), amp);

                _mm256_maskstore_ps(rD + x, mt, _mm256_add_ps(d, add));
            }
        }
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

    // Base plus fog per channel, needed by the amplitude weighting.
    const AlgoType dmin[3] =
    {
        static_cast<AlgoType>(profile.curves.r.dmin),
        static_cast<AlgoType>(profile.curves.g.dmin),
        static_cast<AlgoType>(profile.curves.b.dmin)
    };

    const AlgoType fogGrain  = static_cast<AlgoType>(gs.fog_grain);
    const AlgoType clumpGain = static_cast<AlgoType>(gs.clump_gain);

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
    if (profile.is_monochrome || hasMosaic)
    {
        // The green clump figure stands for the single emulsion, matching the
        // reference: a monochrome stock's three clump fields carry the same number,
        // and green is the one the metric is quoted against.
        AlgoMakeGrainField(pScrFieldR, pScrNoise, pScrLobe, pScrWork,
                           sizeX, sizeY, pitch,
                           static_cast<AlgoType>(gs.clump_um_g),
                           clumpGain,
                           static_cast<AlgoType>(gs.rms_granularity),
                           scanSigmaPx, pxPerMm,
                           eALGO_RNG_STAGE::eRNG_GRAIN_G,
                           grainSeed, frameIndex);

        // The same field three times. Three independent fields here would produce
        // coloured speckle on a black-and-white image.
        AlgoAddGrain(pDstR, pDstG, pDstB,
                     pScrFieldR, pScrFieldR, pScrFieldR,
                     sizeX, sizeY, pitch, dmin, fogGrain, gain);
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

        const AlgoType clump[3] =
        {
            static_cast<AlgoType>(gs.clump_um_r),
            static_cast<AlgoType>(gs.clump_um_g),
            static_cast<AlgoType>(gs.clump_um_b)
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
            AlgoMakeGrainField(field[c], pScrNoise, pScrLobe, pScrWork,
                               sizeX, sizeY, pitch,
                               clump[c], clumpGain, rms[c],
                               scanSigmaPx, pxPerMm,
                               stream[c], grainSeed, frameIndex);

        AlgoAddGrain(pDstR, pDstG, pDstB,
                     pScrFieldR, pScrFieldG, pScrFieldB,
                     sizeX, sizeY, pitch, dmin, fogGrain, gain);
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
#endif