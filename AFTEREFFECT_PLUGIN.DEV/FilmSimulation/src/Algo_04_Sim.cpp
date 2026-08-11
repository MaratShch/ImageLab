#if 0
// ---------------------------------------------------------------------------
//  Algo_04_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the field initialiser, the vignette loop and the final apply.
//  The coating field is built at low resolution and bilinearly upsampled, which is a
//  gather rather than a load, so it is left scalar for the same reason
//  AlgoBilinearUpsample is - eight consecutive destination pixels read eight
//  unrelated source pairs, and a gather of eight 32-bit elements costs more than the
//  eight scalar loads it would replace.
//
//  Pipeline stage 4 and its sub-stage 4b, the last stages in exposure space:
//
//      AlgoVignetteValueAtRadius        pure cos^4 falloff, exposed for testing
//      AlgoStage04_CoatingAndVignette   builds and applies the combined field
//
//  Two mechanisms with different physics and different geometry, but both pure
//  multipliers on exposure, so they are assembled into one field and applied in a
//  single pass.
//
//  Raw pointers, explicit geometry, no allocation, no state, no validation.
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

#include "AlgoCoatingField.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>


static_assert(sizeof(AlgoType) == 4,
              "the AVX2 path requires AlgoType to be a 32-bit float");

namespace
{
    // ----------------------------------------------------------------------
    //  Lanes in one AVX2 vector of float, and the tail mask for the final
    //  partial vector of a row.
    //
    //  The active width is not generally a multiple of eight - 1023, 1998 and 2816
    //  all appear in the test set. Masked access leaves the row padding untouched,
    //  which keeps the NaN-poison arena test meaningful and stays correct even if
    //  row padding were ever removed.
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

#include <cmath>   // std::sqrt, std::cos, std::pow


namespace
{
    // Choose the low-resolution grid extent for one axis.
    //
    //     samples = clamp( 4 * frameExtentMm / correlationLengthMm, 24, 192 )
    //
    // Four samples per correlation length, floored so a very large-scale field is
    // not reduced to a straight ramp, and capped so a very fine-scale field does
    // not cost more than it is worth.
    inline int32_t loResExtent (const AlgoType frameExtentMm,
                                const AlgoType corrMm) noexcept
    {
        const AlgoType corr = MAX_VALUE(corrMm, ALGO_COATING_MIN_CORR_MM);
        const AlgoType want = ALGO_COATING_SAMPLES_PER_CORR * frameExtentMm / corr;

        const int32_t n = static_cast<int32_t>(want);

        return CLAMP_VALUE(n, ALGO_COATING_LORES_MIN, ALGO_COATING_LORES_MAX);
    }
}


// ---------------------------------------------------------------------------
//  cos^4 vignette falloff at a normalised radius
// ---------------------------------------------------------------------------
AlgoType AlgoVignetteValueAtRadius (const AlgoType rNorm, const AlgoType stops) noexcept
{
    // No requested loss means a perfectly even field.
    if (stops <= ALGO_ZERO)
        return ALGO_ONE;

    // Corner cosine from the requested loss in stops. The exponent is divided by
    // the cos^4 power, so raising the result to the fourth gives exactly 2^-stops
    // at the corner:
    //
    //     cos_c^4 = 2^-stops   =>   cos_c = 2^(-stops/4)
    //
    // Held in HighPrecType: pow and the reciprocal that follows lose accuracy
    // rapidly in single precision for small losses, and the result is a
    // multiplicative field where a small error is a visible gradient.
    const HighPrecType cosCorner =
        std::pow(2.0, -static_cast<HighPrecType>(stops)
                       / static_cast<HighPrecType>(ALGO_VIGNETTE_EXPONENT));

    // tan from cos, via tan^2 = 1/cos^2 - 1. Clamped at zero because a cosine of
    // exactly 1 gives a difference of zero that can land marginally negative in
    // floating point, and sqrt of a negative is a quiet NaN.
    const HighPrecType t2 = (1.0 / (cosCorner * cosCorner)) - 1.0;

    const HighPrecType tanCorner = std::sqrt(MAX_VALUE(t2, 0.0));

    // The true angle at this radius scales linearly in TANGENT, not in angle: the
    // image plane is flat, so distance from the axis is proportional to tan.
    const HighPrecType tanTheta = static_cast<HighPrecType>(rNorm) * tanCorner;

    const HighPrecType cosTheta = 1.0 / std::sqrt(1.0 + tanTheta * tanTheta);

    // cos^4 as two squarings.
    const HighPrecType c2 = cosTheta * cosTheta;

    return static_cast<AlgoType>(c2 * c2);
}


// ---------------------------------------------------------------------------
//  Stage 4 + 4b: coating field and lens vignette
// ---------------------------------------------------------------------------
void AlgoStage04_CoatingAndVignette
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pFieldFull,
    AlgoType* RESTRICT       pFieldLo,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           negWidthMm,
    const AlgoType           negHeightMm,
    const AlgoType           framePitchMm,
    const int32_t            frameIndex,
    const uint32_t           seed
) noexcept
{
    // A negative control value defers to the stock's own era figure, exactly as for
    // the flare in the previous stage.
    const AlgoType stops = (params.vignette < 0.0)
                         ? static_cast<AlgoType>(profile.default_vignette)
                         : static_cast<AlgoType>(params.vignette);

    const AlgoType coatScale = MAX_VALUE(static_cast<AlgoType>(params.coatingScale),
                                         ALGO_ZERO);

    const film::CoatingSpec& coat = profile.coating;

    const bool wantVignette = (stops > ALGO_ZERO);
    const bool wantCoating  = (coatScale > ALGO_ZERO) && (coat.coating_sigma > 0.0);

    // Neither mechanism active: copy so the stage buffer still holds a valid image,
    // as the retained-buffer policy requires.
    if ((false == wantVignette) && (false == wantCoating))
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    // ----------------------------------------------------------------------
    //  Start the field at unity, then multiply in each active mechanism.
    // ----------------------------------------------------------------------
    for (int32_t y = 0; y < sizeY; y++)
    {
        AlgoType* RESTRICT pF = pFieldFull + static_cast<std::ptrdiff_t>(y) * pitch;

        // Field starts at unity; each active mechanism multiplies into it.
        const __m256 vOne = _mm256_set1_ps(ALGO_ONE);

        const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
        const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
        const __m256i mt = algoTailMaskLocal(nt);

        int32_t x = 0;
        for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
            _mm256_storeu_ps(pF + x, vOne);
        if (nt > 0)
            _mm256_maskstore_ps(pF + x, mt, vOne);
    }

    // ----------------------------------------------------------------------
    //  Lens vignette, locked to the frame.
    //
    //  The normalised radius runs from 0 at the centre to exactly 1 at the CORNER.
    //  Both axes are normalised to their own half-extent first and the result
    //  divided by sqrt(2), which is what puts the corner rather than the edge
    //  midpoint at unity - the corner is where the requested loss applies.
    //
    //  The span is (n - 1), not n, so the first and last pixel centres land exactly
    //  on -1 and +1. With n the field is very slightly asymmetric, which shows up as
    //  a one-pixel brightness difference between opposite edges.
    // ----------------------------------------------------------------------
    if (wantVignette)
    {
        const HighPrecType cy = static_cast<HighPrecType>(sizeY - 1) * 0.5;
        const HighPrecType cx = static_cast<HighPrecType>(sizeX - 1) * 0.5;

        const HighPrecType halfY = MAX_VALUE(cy, 1.0);
        const HighPrecType halfX = MAX_VALUE(cx, 1.0);

        // Reciprocals of the half extents, formed once. The normalisation is a
        // divide by a frame constant, and a divide costs an order of magnitude
        // more than a multiply, so it is inverted here and multiplied below.
        const HighPrecType invHalfY = 1.0 / halfY;
        const HighPrecType invHalfX = 1.0 / halfX;

        // ------------------------------------------------------------------
        //  Corner geometry, hoisted out of the pixel loop.
        //
        //  AlgoVignetteValueAtRadius derives the corner cosine and its tangent
        //  from the requested loss in stops. Both depend ONLY on stops, which is
        //  constant for the whole frame, so they are formed once here rather than
        //  once per pixel. Calling that function per pixel would evaluate a pow
        //  and a sqrt a million times to produce the same two numbers.
        //
        //  The function is retained unchanged as the single-radius reference and
        //  is what the tests check this loop against.
        //
        //      cos_c^4 = 2^-stops   =>   cos_c = 2^(-stops/4)
        //
        //  and tan^2 follows from tan^2 = 1/cos^2 - 1. Clamped at zero because a
        //  cosine of exactly 1 gives a difference that can land marginally
        //  negative in floating point.
        // ------------------------------------------------------------------
        const HighPrecType cosCorner =
            std::pow(2.0, -static_cast<HighPrecType>(stops)
                           / static_cast<HighPrecType>(ALGO_VIGNETTE_EXPONENT));

        const HighPrecType tanCorner2 =
            MAX_VALUE((1.0 / (cosCorner * cosCorner)) - 1.0, 0.0);

        // The corner radius is normalised to exactly 1 by a factor of 1/sqrt(2),
        // and the loop below works in radius SQUARED, so that factor enters
        // squared as one half. Folding it into the corner tangent here removes
        // both a multiply and, more importantly, the square root that would
        // otherwise be needed to form the radius before squaring it again.
        // ------------------------------------------------------------------
        //  RULE D1 ALIGNMENT, 2026-08-11: this loop was HighPrecType per pixel.
        //
        //  It is pure geometry - a normalised radius and a cos^4 falloff - and
        //  float32 carries it with room to spare: a pixel coordinate up to 16.7
        //  million is EXACT in float, the normalised coordinates are order one,
        //  and the field they produce is a smooth multiplier whose relative
        //  precision at float is ~1e-07. That is four orders of magnitude below
        //  the 16-bit quantisation the result is eventually written to.
        //
        //  What it cost to be double: a scalar double divide per pixel, and the
        //  ALGO_VECTOR_HINT above could never have been honoured, because four
        //  doubles per vector against eight floats means the compiler would have
        //  had to split every load. 2.07 million iterations at HD.
        //
        //  NOT converted, deliberately: the coating field synthesis below. Its
        //  own precision note explains why - the web phase grows without bound
        //  along a clip and float would decorrelate successive frames. That is a
        //  setup-domain quantity in the D1 sense even though it is evaluated per
        //  pixel, and it stays HighPrecType.
        // ------------------------------------------------------------------
        const AlgoType halfTanCorner2 =
            static_cast<AlgoType>(0.5 * tanCorner2);

        const AlgoType cxF       = static_cast<AlgoType>(cx);
        const AlgoType cyF       = static_cast<AlgoType>(cy);
        const AlgoType invHalfXF = static_cast<AlgoType>(invHalfX);
        const AlgoType invHalfYF = static_cast<AlgoType>(invHalfY);

        for (int32_t y = 0; y < sizeY; y++)
        {
            // Row-constant normalised vertical offset and its square.
            const AlgoType yn  = (static_cast<AlgoType>(y) - cyF) * invHalfYF;
            const AlgoType yn2 = yn * yn;

            AlgoType* RESTRICT pF = pFieldFull + static_cast<std::ptrdiff_t>(y) * pitch;

            ALGO_VECTOR_HINT
            for (int32_t x = 0; x < sizeX; x++)
            {
                const AlgoType xn = (static_cast<AlgoType>(x) - cxF) * invHalfXF;

                // Squared tangent of the true ray angle at this pixel. The angle
                // scales linearly in TANGENT, not in angle, because the image
                // plane is flat: distance from the axis is proportional to tan.
                // Working in the square throughout means the radius never has to
                // be rooted.
                const AlgoType t2 = (yn2 + xn * xn) * halfTanCorner2;

                // cos^2 from tan^2, via cos^2 = 1/(1 + tan^2). One divide, and no
                // square root at all.
                const AlgoType c2 = ALGO_ONE / (ALGO_ONE + t2);

                // cos^4 is that squared. Multiplied into the field rather than
                // stored, because the coating pass may already have written here.
                pF[x] *= c2 * c2;
            }
        }
    }

    // ----------------------------------------------------------------------
    //  Web-coherent coating field.
    //
    //  PRECISION NOTE: the whole field synthesis is HighPrecType and deliberately
    //  does not follow the alias down to float. The web offset grows without bound
    //  along a clip - a thousand frames of 35 mm is about 19 metres - and it is
    //  multiplied by a spatial frequency to form a cosine argument, so the phase
    //  reaches many thousands of radians. In single precision the low bits of such
    //  an argument are gone and the field would decorrelate from one frame to the
    //  next, which is exactly the artefact this stage exists to avoid. It is
    //  evaluated on a small low-resolution grid, so the cost is negligible.
    // ----------------------------------------------------------------------
    if (wantCoating)
    {
        const HighPrecType sigma = static_cast<HighPrecType>(coat.coating_sigma)
                                 * static_cast<HighPrecType>(coatScale);

        const HighPrecType corrAcross =
            MAX_VALUE(static_cast<HighPrecType>(coat.coating_corr_across_mm),
                      static_cast<HighPrecType>(ALGO_COATING_MIN_CORR_MM));
        const HighPrecType corrAlong =
            MAX_VALUE(static_cast<HighPrecType>(coat.coating_corr_along_mm),
                      static_cast<HighPrecType>(ALGO_COATING_MIN_CORR_MM));

        const int32_t loW = loResExtent(negWidthMm,  static_cast<AlgoType>(corrAcross));
        const int32_t loH = loResExtent(negHeightMm, static_cast<AlgoType>(corrAlong));

        // Absolute web offset of this frame, in millimetres. Unperforated formats -
        // sheet film, instant - have a pitch of zero: a single exposure, so no
        // advance and no frame-to-frame change.
        const HighPrecType yOffMm = static_cast<HighPrecType>(frameIndex)
                                  * static_cast<HighPrecType>(framePitchMm);

        // Two contributions of equal variance, so each carries sigma/sqrt(2). The
        // further division by sqrt(N/2) is the normalisation that makes a sum of N
        // random-phase cosines have unit variance, since each cosine contributes
        // 1/2 to the total.
        const HighPrecType half =
            sigma * static_cast<HighPrecType>(ALGO_COATING_HALF_VARIANCE_SCALE);

        const HighPrecType compNorm =
            half / std::sqrt(static_cast<HighPrecType>(ALGO_COATING_COMPONENTS) * 0.5);

        const HighPrecType kTwoPi  = 6.283185307179586476925286766559;
        const HighPrecType fSigmaX = 1.0 / (kTwoPi * corrAcross);
        const HighPrecType fSigmaY = 1.0 / (kTwoPi * corrAlong);

        // Seed for this stage's two random streams: the caller's global seed
        // combined with the per-call seed argument.
        //
        // NAMED coatSeed, NOT seed. A local called seed would SHADOW the
        // parameter of the same name, and the initialiser would then read the
        // local during its own initialisation rather than the parameter - which
        // is undefined behaviour, and silently made the coating field depend on
        // whatever happened to be in that stack slot.
        const uint32_t coatSeed = static_cast<uint32_t>(params.seed) ^ seed;

        // Component coefficients. All are pure functions of the seed and the
        // component index, so no state is carried and any frame renders
        // independently.
        //
        // Neither stream includes the frame index. The STATIC component's streaks
        // are fixed hardware and must be identical on every frame; the DRIFTING
        // component's frame dependence enters through the web offset added to the
        // sampled y coordinate, NOT through new random numbers. Re-drawing per frame
        // is precisely the mistake that turns this into large-scale flicker.
        AVX2_ALIGN HighPrecType fxStatic[ALGO_COATING_COMPONENTS];
        AVX2_ALIGN HighPrecType phStatic[ALGO_COATING_COMPONENTS];
        AVX2_ALIGN HighPrecType fxDrift [ALGO_COATING_COMPONENTS];
        AVX2_ALIGN HighPrecType fyDrift [ALGO_COATING_COMPONENTS];
        AVX2_ALIGN HighPrecType phDrift [ALGO_COATING_COMPONENTS];

        for (int32_t k = 0; k < ALGO_COATING_COMPONENTS; k++)
        {
            const uint32_t ord = static_cast<uint32_t>(k);

            const uint64_t cS =
                AlgoRngCounter(coatSeed, 0, eALGO_RNG_STAGE::eRNG_COATING_STATIC, ord);
            const uint64_t cD =
                AlgoRngCounter(coatSeed, 0, eALGO_RNG_STAGE::eRNG_COATING_DRIFT, ord);

            // Three independent values per stream, from displaced counters rather
            // than from a sequence, since there is no sequence.
            fxStatic[k] = AlgoRngNormal(cS) * fSigmaX;
            phStatic[k] = AlgoRngUniformRange(cS ^ 0x5555555555555555ull, 0.0, kTwoPi);

            fxDrift[k]  = AlgoRngNormal(cD) * fSigmaX;
            fyDrift[k]  = AlgoRngNormal(cD ^ 0xAAAAAAAAAAAAAAAAull) * fSigmaY;
            phDrift[k]  = AlgoRngUniformRange(cD ^ 0x3333333333333333ull, 0.0, kTwoPi);
        }

        // Sample positions span the frame extent inclusively, matching the
        // corner-aligned interpolation that scales the result up.
        const HighPrecType stepXmm = static_cast<HighPrecType>(negWidthMm)
                                   / static_cast<HighPrecType>(MAX_VALUE(loW - 1, 1));
        const HighPrecType stepYmm = static_cast<HighPrecType>(negHeightMm)
                                   / static_cast<HighPrecType>(MAX_VALUE(loH - 1, 1));

        // Evaluate into the top-left corner of the supplied scratch plane.
        for (int32_t iy = 0; iy < loH; iy++)
        {
            // WEB coordinate, not frame coordinate: the frame's own offset along the
            // web is added here, and it is the only frame-dependent quantity in the
            // whole field.
            const HighPrecType yMm = static_cast<HighPrecType>(iy) * stepYmm + yOffMm;

            AlgoType* RESTRICT pLo = pFieldLo + static_cast<std::ptrdiff_t>(iy) * pitch;

            for (int32_t ix = 0; ix < loW; ix++)
            {
                const HighPrecType xMm = static_cast<HighPrecType>(ix) * stepXmm;

                HighPrecType acc = 0.0;

                // Static cross-web streaks: a function of x alone, so identical on
                // every frame for the whole roll.
                for (int32_t k = 0; k < ALGO_COATING_COMPONENTS; k++)
                    acc += std::cos(kTwoPi * fxStatic[k] * xMm + phStatic[k]);

                // Drifting two-dimensional field: slides with the web.
                for (int32_t k = 0; k < ALGO_COATING_COMPONENTS; k++)
                    acc += std::cos(kTwoPi * (fyDrift[k] * yMm + fxDrift[k] * xMm)
                                    + phDrift[k]);

                // Mean 1.0 multiplier: unity plus the zero-mean variation.
                pLo[ix] = static_cast<AlgoType>(1.0 + acc * compNorm);
            }
        }

        // Interpolate up into the blur scratch, then multiply into the combined
        // field. Done in two steps rather than one fused loop so the upsample stays
        // a single shared primitive rather than being reimplemented here.
        //
        // The field is reused as its own upsample destination one row at a time,
        // which is safe because the interpolation reads only the low-resolution
        // corner and writes only full-width rows below it.
        // Source-per-destination step in each axis, formed once. These were
        // previously rebuilt as a full divide inside the pixel loop, which put a
        // frame-constant division on every one of the output samples.
        const HighPrecType xScale = static_cast<HighPrecType>(loW - 1)
                                  / static_cast<HighPrecType>(MAX_VALUE(sizeX - 1, 1));
        const HighPrecType yScale = static_cast<HighPrecType>(loH - 1)
                                  / static_cast<HighPrecType>(MAX_VALUE(sizeY - 1, 1));

        // ------------------------------------------------------------------
        //  RULE D1 ALIGNMENT, 2026-08-11: the bilinear upsample below was
        //  HighPrecType per pixel, and the same anti-pattern as the gate-weave
        //  sampler - it WIDENED FLOAT PLANE SAMPLES TO DOUBLE, interpolated, and
        //  narrowed the result straight back. The two roundings on the way in and
        //  out are the same magnitude as the one being avoided, so the widening
        //  bought nothing and cost the loop its chance to vectorise.
        //
        //  The SCALES stay HighPrecType: they are two divisions per call, they
        //  set where every sample lands, and an accumulated coordinate error here
        //  would shift the field rather than merely roughen it.
        // ------------------------------------------------------------------
        const AlgoType xScaleF = static_cast<AlgoType>(xScale);
        const AlgoType yScaleF = static_cast<AlgoType>(yScale);

        for (int32_t y = 0; y < sizeY; y++)
        {
            const AlgoType sy = static_cast<AlgoType>(y) * yScaleF;

            int32_t y0 = static_cast<int32_t>(sy);
            y0 = MIN_VALUE(y0, loH - 2);
            y0 = MAX_VALUE(y0, 0);

            const AlgoType fy = sy - static_cast<AlgoType>(y0);

            const AlgoType* RESTRICT pTop =
                pFieldLo + static_cast<std::ptrdiff_t>(y0) * pitch;
            const AlgoType* RESTRICT pBot =
                pFieldLo + static_cast<std::ptrdiff_t>(MIN_VALUE(y0 + 1, loH - 1)) * pitch;

            AlgoType* RESTRICT pF = pFieldFull + static_cast<std::ptrdiff_t>(y) * pitch;

            for (int32_t x = 0; x < sizeX; x++)
            {
                const AlgoType sx = static_cast<AlgoType>(x) * xScaleF;

                int32_t x0 = static_cast<int32_t>(sx);
                x0 = MIN_VALUE(x0, loW - 2);
                x0 = MAX_VALUE(x0, 0);

                const int32_t  x1 = MIN_VALUE(x0 + 1, loW - 1);
                const AlgoType fx = sx - static_cast<AlgoType>(x0);

                // Two horizontal interpolations, then one vertical between them.
                const AlgoType top = pTop[x0] + (pTop[x1] - pTop[x0]) * fx;
                const AlgoType bot = pBot[x0] + (pBot[x1] - pBot[x0]) * fx;

                pF[x] *= top + (bot - top) * fy;
            }
        }
    }

    // ----------------------------------------------------------------------
    //  Apply the combined field to all three channels.
    //
    //  One field, three multiplies. The field is achromatic: neither mechanism has
    //  any wavelength dependence at this scale - a lens loses light off-axis equally
    //  in all three records, and a coating variation changes the AMOUNT of emulsion
    //  laid down, not its spectral character.
    // ----------------------------------------------------------------------
    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

        const AlgoType* RESTRICT pInR = pSrcR      + off;
        const AlgoType* RESTRICT pInG = pSrcG      + off;
        const AlgoType* RESTRICT pInB = pSrcB      + off;
        const AlgoType* RESTRICT pF   = pFieldFull + off;

        AlgoType* RESTRICT pOutR = pDstR + off;
        AlgoType* RESTRICT pOutG = pDstG + off;
        AlgoType* RESTRICT pOutB = pDstB + off;

        // One field value scales all three records, so the field row is loaded once
        // and reused across the three multiplies - three loads and three stores per
        // vector instead of six loads.
        const int32_t nv = sizeX / ALGO_AVX2_LANES_LOCAL;
        const int32_t nt = sizeX - nv * ALGO_AVX2_LANES_LOCAL;
        const __m256i mt = algoTailMaskLocal(nt);

        int32_t x = 0;

        for (int32_t v = 0; v < nv; v++, x += ALGO_AVX2_LANES_LOCAL)
        {
            const __m256 f = _mm256_loadu_ps(pF + x);

            _mm256_storeu_ps(pOutR + x, _mm256_mul_ps(_mm256_loadu_ps(pInR + x), f));
            _mm256_storeu_ps(pOutG + x, _mm256_mul_ps(_mm256_loadu_ps(pInG + x), f));
            _mm256_storeu_ps(pOutB + x, _mm256_mul_ps(_mm256_loadu_ps(pInB + x), f));
        }

        if (nt > 0)
        {
            const __m256 f = _mm256_maskload_ps(pF + x, mt);

            _mm256_maskstore_ps(pOutR + x, mt,
                _mm256_mul_ps(_mm256_maskload_ps(pInR + x, mt), f));
            _mm256_maskstore_ps(pOutG + x, mt,
                _mm256_mul_ps(_mm256_maskload_ps(pInG + x, mt), f));
            _mm256_maskstore_ps(pOutB + x, mt,
                _mm256_mul_ps(_mm256_maskload_ps(pInB + x, mt), f));
        }
    }

    return;
}
#endif