// ---------------------------------------------------------------------------
//  Algo_03_Sim.cpp   --   AVX2
//
//  Pipeline stage 3 and its sub-stages:
//
//      AlgoStage03_StockColourBalance   per-channel balance gains
//      AlgoStage03b_VeilingFlare        lens flare: veil plus a broad scatter lobe
//      AlgoStage03c_TemporalFlicker     STUB - copies input to output
//
//  Same filename, same function names, same prototypes as the scalar build.
//
//  ALL ARITHMETIC IS FLOAT32. The scalar path remains the reference.
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

// Common.hpp -- AVX2_ALIGN / CACHE_ALIGN are defined here. Included
// DIRECTLY rather than relied on transitively: this file declares an
// aligned buffer, so the macro must not depend on another header's
// include order to be in scope.
#include "Common.hpp"
#include "AlgoStockColourBalance.hpp"
#include "AlgoVeilingFlare.hpp"
#include "AlgoTemporalFlicker.hpp"
#include "AlgoSeparableBlur.hpp"   // AlgoCopyImage, AlgoPlaneMean, the blur passes
#include "AlgoSpectralSensitivity.hpp"

#include <immintrin.h>
#include <cmath>   // std::expm1, six times per frame


static_assert(sizeof(AlgoType) == 4,
              "the AVX2 path requires AlgoType to be a 32-bit float");


namespace
{
    constexpr int32_t ALGO_AVX2_LANES = 8;


    // ----------------------------------------------------------------------
    //  Tail mask for the final, partial vector of a row.
    //
    //  The active width is not generally a multiple of eight. Masked load and store
    //  leave the row padding untouched, which keeps the NaN-poison arena test
    //  meaningful and makes the tail correct even if row padding were ever removed.
    //
    //  Duplicated per translation unit rather than shared through a new header,
    //  because the AVX2 folder mirrors the scalar file set exactly and introducing a
    //  header with no scalar counterpart would break that correspondence. It is
    //  eight lines and one static table; the cost of the duplication is lower than
    //  the cost of the divergence.
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
    //  Spectral radiance of a blackbody at one wavelength, Planck's law.
    //
    //  A VERBATIM copy of the scalar helper, and it stays in HighPrecType.
    //
    //  This is the one computation in the front end that genuinely cannot be done
    //  in float: lambda^5 for a wavelength in metres is of the order 1e-32 while
    //  the exponential argument reaches about 53, so the intermediates span roughly
    //  sixty decades. It runs six times per frame - three layers, two illuminants -
    //  so the width costs nothing measurable.
    // ----------------------------------------------------------------------
    inline HighPrecType planckRadiance (const HighPrecType lambdaNm,
                                        const HighPrecType kelvin) noexcept
    {
        const HighPrecType lambda = lambdaNm * ALGO_NM_TO_M;

        // lambda^5 as three multiplies rather than std::pow: exact, faster, and
        // free of pow's error for a small integral exponent.
        const HighPrecType l2 = lambda * lambda;
        const HighPrecType l5 = l2 * l2 * lambda;

        const HighPrecType x = ALGO_PLANCK_C2 / (lambda * kelvin);

        return ALGO_PLANCK_C1 / (l5 * std::expm1(x));
    }
}


// ---------------------------------------------------------------------------
//  Balance gains
//
//  Declared in AlgoStockColourBalance.hpp and DEFINED here, exactly as in the
//  scalar translation unit - so the AVX2 object provides it too and the linker sees
//  one definition whichever build is used. Byte-for-byte the scalar implementation:
//  it runs once per frame on three scalars, so there is nothing here to vectorise
//  and every reason to keep it identical.
// ---------------------------------------------------------------------------
void AlgoBalanceGains
(
    const HighPrecType sceneKelvin,
    const HighPrecType stockKelvin,
    AlgoType           gains[3]
) noexcept
{
    // The three layer sensitivity peaks, in channel order R, G, B.
    AVX2_ALIGN const HighPrecType lambdaNm[3] =
    {
        ALGO_LAYER_PEAK_NM_R,
        ALGO_LAYER_PEAK_NM_G,
        ALGO_LAYER_PEAK_NM_B
    };

    // Raw radiance ratio per layer: how much more, or less, light this layer
    // receives under the scene illuminant than under the one it expects.
    AVX2_ALIGN HighPrecType ratio[3];

    for (int32_t c = 0; c < 3; c++)
    {
        const HighPrecType scene = planckRadiance(lambdaNm[c], sceneKelvin);
        const HighPrecType stock = planckRadiance(lambdaNm[c], stockKelvin);

        ratio[c] = scene / stock;
    }

    // Normalise so green is exactly 1.0, leaving the stage to change only the
    // balance BETWEEN records and never the overall brightness. Without this the
    // stage would double as an exposure control and fight the anchor solve.
    const HighPrecType green = ratio[1];

    for (int32_t c = 0; c < 3; c++)
        gains[c] = static_cast<AlgoType>(ratio[c] / green);

    return;
}


// ---------------------------------------------------------------------------
//  Stage 3: stock colour balance
// ---------------------------------------------------------------------------
void AlgoStage03_StockColourBalance
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params
) noexcept
{
    // ----------------------------------------------------------------------
    //  BOTH activity tests, exactly as the scalar path states them.
    //
    //   - wbStrength at or below zero: the user has disabled it.
    //   - a monochrome stock: one silver image has no inter-layer ratio to disturb,
    //     so a colour temperature change is an exposure change and stage 2 already
    //     covered it.
    //
    //  The monochrome half is not optional and not a micro-optimisation. THIRTY-SIX
    //  of the 142 stocks are monochrome. When this same segment was prototyped as a
    //  fused scalar pass and this test was left out, AGFA_APX_100 came out wrong by
    //  1.468 - and the error was exactly zero at the default wbStrength of 0, so it
    //  passed every casual check.
    // ----------------------------------------------------------------------
    const bool applies = (params.wbStrength > 0.0) && (false == profile.is_monochrome);

    if (false == applies)
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    // ----------------------------------------------------------------------
    //  The three gains, derived through the SAME helper the scalar path uses.
    //
    //  AlgoBalanceGains takes HighPrecType and stays double: it evaluates the
    //  Planck radiance, whose intermediates span roughly sixty decades - the fifth
    //  power of a wavelength in metres against an exponential argument near 53.
    //  That is genuinely beyond float, it runs once per frame, and calling the
    //  shared helper means the two builds cannot drift on it.
    // ----------------------------------------------------------------------
    AVX2_ALIGN AlgoType gain[3];

    // MEASURED SPECTRAL PATH. Prefer the stock's own digitised per-layer
    // sensitivity, integrated against the two blackbody SPDs, over the
    // three-assumed-peak proxy below it. AlgoSpectralBalanceGains returns false
    // for the stocks that carry no curves, and the proxy then runs exactly as
    // it did before, so those stocks are unchanged bit for bit.
    //
    // The difference is not cosmetic: measured at 3200 K on daylight stock the
    // derived red gain is 1.65-1.69 against the proxy's 1.32, and it varies by
    // stock, which a fixed-wavelength proxy cannot express.
    if (!AlgoSpectralBalanceGains(profile,
                                  static_cast<HighPrecType>(params.sceneKelvin),
                                  gain))
    {
        AlgoBalanceGains(static_cast<HighPrecType>(params.sceneKelvin),
                         static_cast<HighPrecType>(profile.balance_kelvin),
                         gain);
    }

    // Blend towards unity by the user's strength: g = 1 + (gain - 1) * strength.
    // At 0 this is exactly 1.0, so a disabled effect is bit-exactly neutral.
    const AlgoType strength = static_cast<AlgoType>(params.wbStrength);

    const __m256 vGainR =
        _mm256_set1_ps(ALGO_ONE + (gain[0] - ALGO_ONE) * strength);
    const __m256 vGainG =
        _mm256_set1_ps(ALGO_ONE + (gain[1] - ALGO_ONE) * strength);
    const __m256 vGainB =
        _mm256_set1_ps(ALGO_ONE + (gain[2] - ALGO_ONE) * strength);

    const int32_t vecCount = sizeX / ALGO_AVX2_LANES;
    const int32_t tailN    = sizeX - (vecCount * ALGO_AVX2_LANES);

    const __m256i vTail = algoTailMask(tailN);

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        const AlgoType* RESTRICT pInR = pSrcR + off;
        const AlgoType* RESTRICT pInG = pSrcG + off;
        const AlgoType* RESTRICT pInB = pSrcB + off;

        AlgoType* RESTRICT pOutR = pDstR + off;
        AlgoType* RESTRICT pOutG = pDstG + off;
        AlgoType* RESTRICT pOutB = pDstB + off;

        int32_t x = 0;

        for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
        {
            _mm256_storeu_ps(pOutR + x,
                _mm256_mul_ps(_mm256_loadu_ps(pInR + x), vGainR));
            _mm256_storeu_ps(pOutG + x,
                _mm256_mul_ps(_mm256_loadu_ps(pInG + x), vGainG));
            _mm256_storeu_ps(pOutB + x,
                _mm256_mul_ps(_mm256_loadu_ps(pInB + x), vGainB));
        }

        if (tailN > 0)
        {
            _mm256_maskstore_ps(pOutR + x, vTail,
                _mm256_mul_ps(_mm256_maskload_ps(pInR + x, vTail), vGainR));
            _mm256_maskstore_ps(pOutG + x, vTail,
                _mm256_mul_ps(_mm256_maskload_ps(pInG + x, vTail), vGainG));
            _mm256_maskstore_ps(pOutB + x, vTail,
                _mm256_mul_ps(_mm256_maskload_ps(pInB + x, vTail), vGainB));
        }
    }

    return;
}


// ---------------------------------------------------------------------------
//  Stage 3b: veiling flare
//
//  Two of the three passes here are pointwise and vectorise directly: forming the
//  luminance plane, and compositing the scattered light back over the image. The
//  middle pass is a wide separable blur and is left to the shared blur primitive,
//  which is a barrier for fusion and a separate optimisation target in its own
//  right.
//
//  The frame-mean veil is taken from AlgoPlaneMean unchanged. That helper
//  accumulates in HighPrecType on purpose: a single-precision running total over two
//  million samples loses its low bits once it has grown large relative to each
//  addend, and this value sets the black floor of the whole frame directly. It is
//  one of the reduction sites where float is the wrong choice however fast it is.
// ---------------------------------------------------------------------------
void AlgoStage03b_VeilingFlare
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScratchLuma,
    AlgoType* RESTRICT       pScratchA,
    AlgoType* RESTRICT       pScratchB,
    AlgoType* RESTRICT       pScratchC,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           pxPerMm
) noexcept
{
    // A negative control means "use the stock's own era-appropriate figure", which
    // is how the database supplies a default without the caller knowing the era.
    const AlgoType flare = (params.flare < 0.0)
                         ? static_cast<AlgoType>(profile.default_flare)
                         : static_cast<AlgoType>(params.flare);

    // Inactive on 108 of the 142 stocks, so this is the common path.
    if (flare <= ALGO_ZERO)
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    const int32_t vecCount = sizeX / ALGO_AVX2_LANES;
    const int32_t tailN    = sizeX - (vecCount * ALGO_AVX2_LANES);

    const __m256i vTail = algoTailMask(tailN);

    // ----------------------------------------------------------------------
    //  Luminance plane.
    //
    //  One plane rather than three: glass scatters broadly and almost
    //  achromatically at these scales, so the wavelength dependence of a stray
    //  reflection is far smaller than the difference between the three records it
    //  would be applied to.
    // ----------------------------------------------------------------------
    const __m256 vLumR = _mm256_set1_ps(ALGO_FLARE_LUMA_R);
    const __m256 vLumG = _mm256_set1_ps(ALGO_FLARE_LUMA_G);
    const __m256 vLumB = _mm256_set1_ps(ALGO_FLARE_LUMA_B);

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        const AlgoType* RESTRICT pInR = pSrcR + off;
        const AlgoType* RESTRICT pInG = pSrcG + off;
        const AlgoType* RESTRICT pInB = pSrcB + off;

        AlgoType* RESTRICT pLum = pScratchLuma + off;

        int32_t x = 0;

        for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
        {
            __m256 l = _mm256_mul_ps(_mm256_loadu_ps(pInR + x), vLumR);
            l = _mm256_fmadd_ps(_mm256_loadu_ps(pInG + x), vLumG, l);
            l = _mm256_fmadd_ps(_mm256_loadu_ps(pInB + x), vLumB, l);

            _mm256_storeu_ps(pLum + x, l);
        }

        if (tailN > 0)
        {
            __m256 l = _mm256_mul_ps(_mm256_maskload_ps(pInR + x, vTail), vLumR);
            l = _mm256_fmadd_ps(_mm256_maskload_ps(pInG + x, vTail), vLumG, l);
            l = _mm256_fmadd_ps(_mm256_maskload_ps(pInB + x, vTail), vLumB, l);

            _mm256_maskstore_ps(pLum + x, vTail, l);
        }
    }

    // Uniform veil: the mean luminance of the whole frame, accumulated wide.
    const AlgoType veil =
        static_cast<AlgoType>(AlgoPlaneMean(pScratchLuma, sizeX, sizeY, pitch));

    // ----------------------------------------------------------------------
    //  Broad scatter lobe, through the shared blur.
    //
    //  A wide separable blur is a barrier: it needs the whole plane before the next
    //  pass can start, so it cannot be folded into either neighbour. Left to the
    //  shared primitive so that both builds scatter identically and the difference
    //  between them stays attributable to the pointwise passes here.
    // ----------------------------------------------------------------------
    //  The three radii are specified in micrometres on the FILM and converted to
    //  pixels here, which is what makes the same profile describe the same physical
    //  scatter at any rendering resolution:
    //
    //      sigma_px = sigma_um * pxPerMm / 1000
    const AlgoType umToPx = pxPerMm / static_cast<AlgoType>(1000);

    AVX2_ALIGN AlgoType sigmaPx[ALGO_BLUR_MAX_LOBES];
    AVX2_ALIGN AlgoType weight [ALGO_BLUR_MAX_LOBES];

    sigmaPx[0] = ALGO_FLARE_SIGMA_UM_0 * umToPx;
    sigmaPx[1] = ALGO_FLARE_SIGMA_UM_1 * umToPx;
    sigmaPx[2] = ALGO_FLARE_SIGMA_UM_2 * umToPx;
    sigmaPx[3] = ALGO_ZERO;                       // unused fourth slot

    weight[0]  = ALGO_FLARE_WEIGHT_0;
    weight[1]  = ALGO_FLARE_WEIGHT_1;
    weight[2]  = ALGO_FLARE_WEIGHT_2;
    weight[3]  = ALGO_ZERO;

    // All four planes are distinct, and must be: the blur reads its source once per
    // lobe, so aliasing the source with any scratch plane lets the first lobe
    // destroy it and leaves the remaining lobes blurring the wrong data.
    AlgoMultiGaussianBlurPlaneWrap(pScratchLuma, pScratchA, pScratchB, pScratchC,
                                   sizeX, sizeY, pitch, sigmaPx, weight, 3);

    // ----------------------------------------------------------------------
    //  Composite.
    //
    //  Energy is conserved: the direct image is attenuated by the flare fraction and
    //  the scattered component added in its place. Adding the haze on top instead
    //  would create light and wash the image out twice over.
    // ----------------------------------------------------------------------
    const __m256 vDirect    = _mm256_set1_ps(ALGO_ONE - flare);
    const __m256 vVeilPart  = _mm256_set1_ps(ALGO_FLARE_VEIL_FRACTION * veil);
    const __m256 vBroadFrac = _mm256_set1_ps(ALGO_ONE - ALGO_FLARE_VEIL_FRACTION);
    const __m256 vFlare     = _mm256_set1_ps(flare);

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        const AlgoType* RESTRICT pInR   = pSrcR     + off;
        const AlgoType* RESTRICT pInG   = pSrcG     + off;
        const AlgoType* RESTRICT pInB   = pSrcB     + off;
        const AlgoType* RESTRICT pBroad = pScratchA + off;

        AlgoType* RESTRICT pOutR = pDstR + off;
        AlgoType* RESTRICT pOutG = pDstG + off;
        AlgoType* RESTRICT pOutB = pDstB + off;

        int32_t x = 0;

        for (int32_t v = 0; v < vecCount; v++, x += ALGO_AVX2_LANES)
        {
            // scattered = veilPart + broadFrac * broad;  added = flare * scattered
            const __m256 scattered =
                _mm256_fmadd_ps(_mm256_loadu_ps(pBroad + x), vBroadFrac, vVeilPart);

            const __m256 added = _mm256_mul_ps(vFlare, scattered);

            // The same scattered value goes to all three channels, so the flare is
            // achromatic by construction.
            _mm256_storeu_ps(pOutR + x,
                _mm256_fmadd_ps(_mm256_loadu_ps(pInR + x), vDirect, added));
            _mm256_storeu_ps(pOutG + x,
                _mm256_fmadd_ps(_mm256_loadu_ps(pInG + x), vDirect, added));
            _mm256_storeu_ps(pOutB + x,
                _mm256_fmadd_ps(_mm256_loadu_ps(pInB + x), vDirect, added));
        }

        if (tailN > 0)
        {
            const __m256 scattered = _mm256_fmadd_ps(
                _mm256_maskload_ps(pBroad + x, vTail), vBroadFrac, vVeilPart);

            const __m256 added = _mm256_mul_ps(vFlare, scattered);

            _mm256_maskstore_ps(pOutR + x, vTail, _mm256_fmadd_ps(
                _mm256_maskload_ps(pInR + x, vTail), vDirect, added));
            _mm256_maskstore_ps(pOutG + x, vTail, _mm256_fmadd_ps(
                _mm256_maskload_ps(pInG + x, vTail), vDirect, added));
            _mm256_maskstore_ps(pOutB + x, vTail, _mm256_fmadd_ps(
                _mm256_maskload_ps(pInB + x, vTail), vDirect, added));
        }
    }

    return;
}


// ---------------------------------------------------------------------------
//  Sub-stage 3c: temporal exposure flicker.  STUB - copies input to output.
//
//  Not yet modelled. The copy is not optional: every stage owns its destination, so
//  returning without writing would leave stale contents in the chain.
//
//  AlgoCopyImage rather than a hand-written vector loop. A pure copy is bound by
//  memory bandwidth, not by instruction throughput, so eight-wide stores buy nothing
//  measurable - and the shared primitive is already the one the scalar path uses,
//  which keeps the two builds identical on the 100 per cent of frames where this
//  stage does nothing.
// ---------------------------------------------------------------------------
void AlgoStage03c_TemporalFlicker
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
) noexcept
{
    (void)profile;
    (void)params;
    (void)frameIndex;
    (void)frameRate;
    (void)seed;

    AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    return;
}
