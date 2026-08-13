// ---------------------------------------------------------------------------
//  Algo_16_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: nothing in the pixel loop yet. Both particulate rasterisers scatter into
//  small bounding boxes - a gate speck is a handful of pixels - so there is nothing to
//  fill a register with, and the splice bar is a rare single-frame event.
//
//  ALIGNMENT: EVERY IMAGE ACCESS IS UNALIGNED, DELIBERATELY.
//
//  loadu/storeu on all plane data. The arena base comes from the host's pool, whose
//  alignment argument is a HINT - it was seen returning a base 16 mod 32, which faults
//  an aligned 256-bit load. AlgoMemHandler.cpp is SHARED by both flavours and must not
//  carry a vector-path concern, so the vector path assumes nothing about alignment.
//
//  Pipeline stage 16: machine-side defects, in the transmittance domain.
//
//      gate dirt        fixed in the frame for a long run, accreting and shedding
//      one-frame dirt   loose particles riding through on this frame only
//      events           splices passing through the aperture
//
//  Raw pointers, explicit geometry, no allocation, no mutable state, no validation
//  of inputs.
// ---------------------------------------------------------------------------

#include "AlgoGateDefects.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>
#include "AlgoFilmCoord.hpp"       // the transport axis, for the splice bar
#include "AlgoSeparableBlur.hpp"   // AlgoCopyImage

#include <cmath>   // std::floor, std::exp, std::log


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
    //  Micrometres per millimetre.
    // ----------------------------------------------------------------------
    constexpr HighPrecType ALGO_GATE_UM_PER_MM = 1000.0;

    // ----------------------------------------------------------------------
    //  Smallest opacity worth rasterising.
    //
    //  One part in a thousand of transmittance is far below one code value at any
    //  bit depth a host will hand over, so a particle fainter than this cannot
    //  change the output and only costs a bounding box.
    // ----------------------------------------------------------------------
    constexpr HighPrecType ALGO_GATE_ALPHA_EPSILON = 1.0e-3;


    // ----------------------------------------------------------------------
    //  One machine-side mark, fully specified before anything is rasterised.
    // ----------------------------------------------------------------------
    struct GateMark
    {
        HighPrecType pxX;
        HighPrecType pxY;
        HighPrecType radiusPx;
        HighPrecType aspect;
        HighPrecType angleRad;
        HighPrecType alpha;
        HighPrecType phase[ALGO_DEFECT_BLOB_HARMONICS];
    };


    // ----------------------------------------------------------------------
    //  Exponential deviate with a given mean.
    //
    //  Inverted in closed form. The uniform is floored away from zero because the
    //  logarithm of zero is not finite, and a single unlucky draw would otherwise
    //  give a particle an infinite lifetime.
    // ----------------------------------------------------------------------
    inline HighPrecType gateExponential
    (
        const uint64_t     counter,
        const HighPrecType mean
    ) noexcept
    {
        const HighPrecType u = MAX_VALUE(AlgoRngUniform01(counter), 1.0e-12);

        return -mean * std::log(u);
    }


    // ----------------------------------------------------------------------
    //  Log-normal deviate parameterised by its MEDIAN.
    //
    //  By the median rather than by the mean of the logarithm, because the median
    //  is the quoted figure and the two differ for a skewed distribution.
    // ----------------------------------------------------------------------
    inline HighPrecType gateLogNormal
    (
        const uint64_t     counter,
        const HighPrecType median,
        const HighPrecType sigmaLn
    ) noexcept
    {
        return median * std::exp(sigmaLn * AlgoRngNormal(counter));
    }


    // ----------------------------------------------------------------------
    //  Rasterise one mark by MULTIPLYING transmittance.
    //
    //  This is the whole polarity difference from stage 9b in one line. A speck in
    //  the gate blocks projection light, so the screen goes darker:
    //
    //      T' = T * (1 - alpha)
    //
    //  where stage 9b, working on the negative before the print, ADDS density and
    //  therefore ends up bright on the positive. Same material, opposite sign,
    //  because of where it sits in the machine.
    //
    //  Neutral across the three channels, deliberately. A particle sitting in the
    //  light path in front of the whole picture attenuates all three equally,
    //  unlike a particle embedded in one emulsion layer - which is why the
    //  per-channel colour weighting at stage 9b has no counterpart here.
    // ----------------------------------------------------------------------
    void gateRasterise
    (
        const GateMark&    m,
        AlgoType* RESTRICT pDstR,
        AlgoType* RESTRICT pDstG,
        AlgoType* RESTRICT pDstB,
        const int32_t      sizeX,
        const int32_t      sizeY,
        const int32_t      pitch,
        const HighPrecType edgePx
    ) noexcept
    {
        if (m.alpha <= ALGO_GATE_ALPHA_EPSILON)
            return;

        const HighPrecType reach = m.radiusPx * m.aspect
                                 * (1.0 + ALGO_GATE_LOBE_DEPTH) + edgePx;

        int32_t x0 = static_cast<int32_t>(std::floor(m.pxX - reach));
        int32_t x1 = static_cast<int32_t>(std::floor(m.pxX + reach)) + 1;
        int32_t y0 = static_cast<int32_t>(std::floor(m.pxY - reach));
        int32_t y1 = static_cast<int32_t>(std::floor(m.pxY + reach)) + 1;

        x0 = MAX_VALUE(x0, 0);
        y0 = MAX_VALUE(y0, 0);
        x1 = MIN_VALUE(x1, sizeX - 1);
        y1 = MIN_VALUE(y1, sizeY - 1);

        if (x1 < x0 || y1 < y0)
            return;

        for (int32_t y = y0; y <= y1; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            AlgoType* RESTRICT rR = pDstR + off;
            AlgoType* RESTRICT rG = pDstG + off;
            AlgoType* RESTRICT rB = pDstB + off;

            const HighPrecType dy = (static_cast<HighPrecType>(y) + 0.5) - m.pxY;

            for (int32_t x = x0; x <= x1; x++)
            {
                const HighPrecType dx =
                    (static_cast<HighPrecType>(x) + 0.5) - m.pxX;

                const HighPrecType cov = AlgoDefectBlobCoverage(
                    dx, dy, m.radiusPx, m.aspect, m.angleRad,
                    ALGO_GATE_LOBE_DEPTH, m.phase, edgePx);

                if (cov <= 0.0)
                    continue;

                // Transmittance factor. Multiplicative, so overlapping marks
                // compose correctly without any special case - two thicknesses of
                // dirt block more light than one, by exactly the product.
                const AlgoType t =
                    static_cast<AlgoType>(1.0 - (m.alpha * cov));

                rR[x] *= t;
                rG[x] *= t;
                rB[x] *= t;
            }
        }

        return;
    }


    // ----------------------------------------------------------------------
    //  Fill a mark's shape parameters from its own stream.
    //
    //  Shared by both dirt populations, because the shape statistics are the same
    //  material - only the size, the opacity and the lifetime differ.
    // ----------------------------------------------------------------------
    inline void gateFillShape
    (
        const uint64_t counter,
        GateMark&      m
    ) noexcept
    {
        m.aspect = 1.0 + (ALGO_GATE_ASPECT_MAX - 1.0)
                 * AlgoRngUniform01(counter);

        // Isotropic. Unlike an abrasion, a lump of debris has no reason to align
        // with the transport - it was not dragged anywhere, it landed.
        m.angleRad = AlgoRngUniform01(counter + 1u)
                   * 6.283185307179586476925286766559;

        for (int32_t k = 0; k < ALGO_DEFECT_BLOB_HARMONICS; k++)
            m.phase[k] = AlgoRngUniform01(counter + 4u + static_cast<uint64_t>(k))
                       * 6.283185307179586476925286766559;

        return;
    }


    // ----------------------------------------------------------------------
    //  Place a gate particle, biased towards the aperture edge.
    //
    //  Gate dirt collects where the aperture plate contacts and scrapes the film,
    //  so the distance inward from the edge is exponential rather than uniform. A
    //  minority ignores the bias entirely, because a population confined strictly
    //  to the border reads as a frame decoration rather than as dirt.
    //
    //  Positions are in PIXELS, in the frame, and this stage runs after the weave -
    //  so they do not move with the picture. That is the point of the class.
    // ----------------------------------------------------------------------
    inline void gatePlace
    (
        const uint64_t     counter,
        const int32_t      sizeX,
        const int32_t      sizeY,
        const HighPrecType decayPx,
        GateMark&          m
    ) noexcept
    {
        const HighPrecType w = static_cast<HighPrecType>(sizeX);
        const HighPrecType h = static_cast<HighPrecType>(sizeY);

        if (AlgoRngUniform01(counter) < ALGO_GATE_INTERIOR_SHARE)
        {
            m.pxX = AlgoRngUniform01(counter + 1u) * w;
            m.pxY = AlgoRngUniform01(counter + 2u) * h;

            return;
        }

        // Which of the four edges, chosen in proportion to their lengths so that a
        // wide frame does not concentrate dirt on its short sides.
        const HighPrecType perim = 2.0 * (w + h);

        const HighPrecType s = AlgoRngUniform01(counter + 1u) * perim;

        // Distance inward from whichever edge was chosen.
        const HighPrecType inward = gateExponential(counter + 2u, decayPx);

        if (s < w)
        {
            m.pxX = s;
            m.pxY = inward;
        }
        else if (s < (w + h))
        {
            m.pxX = w - inward;
            m.pxY = s - w;
        }
        else if (s < (2.0 * w + h))
        {
            m.pxX = s - (w + h);
            m.pxY = h - inward;
        }
        else
        {
            m.pxX = inward;
            m.pxY = s - (2.0 * w + h);
        }

        return;
    }
}


// ---------------------------------------------------------------------------
//  Stage 16: machine-side defects
// ---------------------------------------------------------------------------
void AlgoStage16_GateDefects
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
    const AlgoType           negWidthMm,
    const AlgoType           negHeightMm,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
) noexcept
{
    // The unconditional copy. Every path below multiplies on top of it.
    AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    if (false == params.filmDamageEnabled)
        return;

    const FilmDamage& dmg = params.damage;

    const HighPrecType strength =
        MAX_VALUE(static_cast<HighPrecType>(dmg.damageStrength), 0.0);

    if (strength <= 0.0)
        return;

    const HighPrecType dirtLevel =
        MAX_VALUE(static_cast<HighPrecType>(dmg.gateDirt),     0.0) * strength;
    const HighPrecType eventLevel =
        MAX_VALUE(static_cast<HighPrecType>(dmg.damageEvents), 0.0) * strength;

    if (dirtLevel <= 0.0 && eventLevel <= 0.0)
        return;

    // ----------------------------------------------------------------------
    //  Scale and seed.
    //
    //  The roll seed is the damage seed rather than the engine seed, so re-rolling
    //  the grain does not re-roll the dirt in the projector.
    // ----------------------------------------------------------------------
    const HighPrecType scale = static_cast<HighPrecType>(pxPerMm);

    const uint32_t rollSeed = static_cast<uint32_t>(dmg.damageSeed) ^ seed;

    const HighPrecType fps = MAX_VALUE(static_cast<HighPrecType>(frameRate), 1.0);

    // Edge transition, wider than film-borne dirt because the gate plate is out of
    // the film plane and the optics image it softer.
    const HighPrecType edgePx = MAX_VALUE(
        (ALGO_GATE_EDGE_UM / ALGO_GATE_UM_PER_MM) * scale,
        ALGO_GATE_EDGE_MIN_PX);

    // ----------------------------------------------------------------------
    //  Which reel is this frame on, and how far into it?
    //
    //  The population resets at a reel change, because that is the moment the gate
    //  was opened and what was in it was disturbed. std::floor rather than integer
    //  division: frame indices may be negative, and integer division truncates
    //  towards zero, which would make frames -1 and 0 share a reel.
    // ----------------------------------------------------------------------
    const HighPrecType reelFrames = ALGO_GATE_REEL_SECONDS * fps;

    const HighPrecType framePos = static_cast<HighPrecType>(frameIndex);

    const HighPrecType reelReal = std::floor(framePos / reelFrames);

    const int32_t reelIndex = static_cast<int32_t>(reelReal);

    // Frames elapsed since this reel started.
    const HighPrecType intoReel = framePos - (reelReal * reelFrames);

    // ----------------------------------------------------------------------
    //  GATE DIRT - persistent, frame-fixed, DERIVED rather than accumulated.
    //
    //  Each slot in the pool draws a birth frame and a lifetime from its own
    //  stream, and is present exactly when birth <= intoReel < birth + lifetime.
    //  No history, no accumulation, no dependence on which frames were rendered
    //  before this one - which is what lets the host ask for any frame at any time
    //  and get the same answer.
    //
    //  Slots below the initial count are born at the reel start; the rest are the
    //  arrivals of a Poisson accretion process, whose k-th arrival time is the sum
    //  of k exponential gaps.
    // ----------------------------------------------------------------------
    if (dirtLevel > 0.0)
    {
        const HighPrecType initial = ALGO_GATE_INITIAL_COUNT * dirtLevel;

        const HighPrecType accretion = ALGO_GATE_ACCRETION_PER_FRAME * dirtLevel;

        const HighPrecType gapFrames =
            (accretion > 0.0) ? (1.0 / accretion) : reelFrames;

        const HighPrecType lifeMean =
            (ALGO_GATE_SHED_PER_FRAME > 0.0)
                ? (1.0 / ALGO_GATE_SHED_PER_FRAME)
                : reelFrames;

        const HighPrecType decayPx = ALGO_GATE_EDGE_DECAY_MM * scale;

        // ------------------------------------------------------------------
        //  A small lambda so the two populations below - what survived the reel
        //  change, and what has accreted since - share one body. They differ only
        //  in their birth time and their stream key.
        // ------------------------------------------------------------------
        const auto emit = [&](const uint64_t sc, const HighPrecType birth) noexcept
        {
            if (birth > intoReel)
                return;

            const HighPrecType life = gateExponential(sc + 32u, lifeMean);

            // Shed already.
            if ((birth + life) <= intoReel)
                return;

            GateMark m{};

            gatePlace(sc + 64u, sizeX, sizeY, decayPx, m);
            gateFillShape(sc + 80u, m);

            const HighPrecType dMm = CLAMP_VALUE(
                gateLogNormal(sc + 96u, ALGO_GATE_SIZE_MEDIAN_MM,
                              ALGO_GATE_SIZE_SIGMA_LN),
                ALGO_GATE_SIZE_MIN_MM, ALGO_GATE_SIZE_MAX_MM);

            m.radiusPx = 0.5 * dMm * scale;

            m.alpha = ALGO_GATE_ALPHA_MIN
                    + (ALGO_GATE_ALPHA_MAX - ALGO_GATE_ALPHA_MIN)
                    * AlgoRngUniform01(sc + 112u);

            gateRasterise(m, pDstR, pDstG, pDstB, sizeX, sizeY, pitch, edgePx);
        };

        // ------------------------------------------------------------------
        //  What the reel change did not remove.
        //
        //  Present from frame zero of the reel, and subject to the same shedding as
        //  everything else, so this population decays away while the accreted one
        //  builds up. Given negative ordinals so it cannot collide with the
        //  accretion stream.
        // ------------------------------------------------------------------
        const int32_t initialSlots = static_cast<int32_t>(initial);

        for (int32_t j = 0; j < initialSlots && j < ALGO_GATE_WINDOW; j++)
        {
            emit(AlgoDefectHash(rollSeed, reelIndex, -1 - j,
                                ALGO_GATE_TAG_SLOT), 0.0);
        }

        // ------------------------------------------------------------------
        //  Arrival ordinals worth examining for THIS frame.
        //
        //  The k-th accreted particle arrives around k gaps into the reel, so the
        //  most recent arrival has ordinal floor(intoReel / gap) and anything with
        //  a much smaller ordinal has almost certainly shed. The window therefore
        //  ends at the current frame and reaches back far enough to cover the
        //  exponential lifetime tail.
        //
        //  Sliding rather than fixed-from-the-reel-head, because a pool numbered
        //  from the head runs out: once its last ordinal has arrived the gate can
        //  accrete nothing more, and the population collapses instead of growing.
        //
        //  Arrival times are a jittered regular sequence - ordinal k arrives at
        //  (k + u) gaps - rather than a cumulative sum of exponential gaps. A
        //  cumulative sum is the exact Poisson construction but requires walking
        //  every ordinal from the reel head, which is precisely the state this
        //  design exists to avoid. The jittered sequence has the same mean rate and
        //  is slightly more regular than Poisson; since no measurement of gate
        //  accretion exists at all, that regularity is far below the uncertainty in
        //  the rate itself, whereas statelessness is a hard requirement.
        // ------------------------------------------------------------------
        const int32_t kNewest = static_cast<int32_t>(
            std::floor(intoReel / gapFrames));

        const int32_t kOldest = kNewest - (ALGO_GATE_WINDOW - 1);

        for (int32_t k = kOldest; k <= kNewest; k++)
        {
            // Negative ordinals never arrived - this is the head of the reel.
            if (k < 0)
                continue;

            // Keyed on the reel and the ordinal, so the whole population changes at
            // a reel change and nothing else does.
            const uint64_t sc = AlgoDefectHash(rollSeed, reelIndex, k,
                                               ALGO_GATE_TAG_SLOT);

            emit(sc, (static_cast<HighPrecType>(k) + AlgoRngUniform01(sc))
                     * gapFrames);
        }
    }

    // ----------------------------------------------------------------------
    //  ONE-FRAME DIRT - the sparkle.
    //
    //  Drawn fresh every frame with NO correlation to the frames either side. The
    //  temptation is to give it a little frame-to-frame persistence so it looks
    //  less noisy; that is precisely backwards, because the complete absence of
    //  correlation IS the sparkle, and smoothing it removes the one cue that says
    //  the film is running.
    //
    //  The rate comes from the stock's own TemporalSpec - 0.1 events per frame for
    //  modern material, 3.0 for the 1930s and 1940s - so era drives the baseline
    //  and the control expresses intent.
    // ----------------------------------------------------------------------
    if (dirtLevel > 0.0)
    {
        const HighPrecType eraRate =
            MAX_VALUE(static_cast<HighPrecType>(
                          profile.temporal.dirt_events_per_frame), 0.0);

        const HighPrecType mean = eraRate * ALGO_SPARKLE_EVENT_SHARE * dirtLevel;

        if (mean > 0.0)
        {
            // Keyed on the frame index, which is exactly right here: this
            // population must be independent every frame.
            const uint64_t fc = AlgoDefectHash(rollSeed, frameIndex, 0,
                                               ALGO_GATE_TAG_SPARKLE);

            int32_t count = AlgoDefectPoisson(fc, mean);

            count = MIN_VALUE(count, ALGO_SPARKLE_MAX_PER_FRAME);

            for (int32_t n = 0; n < count; n++)
            {
                const uint64_t pc = fc
                    + (static_cast<uint64_t>(n) + 1u) * 0x200u;

                GateMark m{};

                // Uniform over the frame. Loose dirt has no reason to prefer the
                // aperture edge - it is riding on the film, not lodged in the gate.
                m.pxX = AlgoRngUniform01(pc)      * static_cast<HighPrecType>(sizeX);
                m.pxY = AlgoRngUniform01(pc + 1u) * static_cast<HighPrecType>(sizeY);

                gateFillShape(pc + 8u, m);

                const HighPrecType dMm = CLAMP_VALUE(
                    gateLogNormal(pc + 24u, ALGO_SPARKLE_SIZE_MEDIAN_MM,
                                  ALGO_SPARKLE_SIZE_SIGMA_LN),
                    ALGO_SPARKLE_SIZE_MIN_MM, ALGO_SPARKLE_SIZE_MAX_MM);

                m.radiusPx = 0.5 * dMm * scale;

                m.alpha = ALGO_SPARKLE_ALPHA_MIN
                        + (ALGO_SPARKLE_ALPHA_MAX - ALGO_SPARKLE_ALPHA_MIN)
                        * AlgoRngUniform01(pc + 40u);

                gateRasterise(m, pDstR, pDstG, pDstB, sizeX, sizeY, pitch, edgePx);
            }
        }
    }

    // ----------------------------------------------------------------------
    //  EVENTS - splices passing through the aperture.
    //
    //  A splice is where two pieces of film were joined. It passes as a bar ACROSS
    //  the film, at right angles to the transport, because that is how film is cut
    //  and joined - and since the transport axis is derived from the format's own
    //  geometry, the bar comes out horizontal on a 35 mm still and vertical on
    //  every cine gauge with no per-format code here at all.
    //
    //  Which frames carry one is decided by dividing the timeline into intervals
    //  and asking whether THIS frame falls in the visible window of its interval's
    //  splice. That is a closed-form test, so it needs no history, and it gives the
    //  right mean rate without ever enumerating the frames in between.
    // ----------------------------------------------------------------------
    if (eventLevel > 0.0)
    {
        const HighPrecType intervalFrames =
            ALGO_SPLICE_INTERVAL_SECONDS * fps / eventLevel;

        if (intervalFrames >= 1.0)
        {
            const HighPrecType slotReal = std::floor(framePos / intervalFrames);

            const int32_t slot = static_cast<int32_t>(slotReal);

            const uint64_t sc = AlgoDefectHash(rollSeed, slot, 0,
                                               ALGO_GATE_TAG_SPLICE);

            // Where in this interval the splice sits, and how far this frame is
            // past it.
            const HighPrecType at = AlgoRngUniform01(sc) * intervalFrames;

            const HighPrecType since = (framePos - (slotReal * intervalFrames)) - at;

            if (since >= 0.0 &&
                since < static_cast<HighPrecType>(ALGO_SPLICE_FRAMES))
            {
                // The window tells us which image axis the film runs along, so the
                // bar can be laid across it. Frame pitch is not needed for a
                // stationary bar, so zero is passed and the window reports no
                // transport for sheet formats - which is correct: a sheet has no
                // splices.
                const AlgoFilmWindow w = AlgoMakeFilmWindow(
                    negWidthMm, negHeightMm,
                    static_cast<AlgoType>(0), sizeX, sizeY, frameIndex);

                // Thickness and position along the transport axis.
                const HighPrecType extentPx = w.transportAlongWidth
                    ? static_cast<HighPrecType>(sizeX)
                    : static_cast<HighPrecType>(sizeY);

                const HighPrecType thick =
                    MAX_VALUE(extentPx * ALGO_SPLICE_THICKNESS_FRAC, 1.0);

                // The join travels through the aperture, so it sits further along
                // on the second frame than on the first.
                const HighPrecType centre = extentPx
                    * (0.15 + 0.7 * AlgoRngUniform01(sc + 1u))
                    + since * thick;

                const HighPrecType lo = centre - 0.5 * thick;
                const HighPrecType hi = centre + 0.5 * thick;

                const AlgoType t =
                    static_cast<AlgoType>(1.0 - ALGO_SPLICE_ALPHA);

                for (int32_t y = 0; y < sizeY; y++)
                {
                    const std::ptrdiff_t off =
                        static_cast<std::ptrdiff_t>(y) * pitch;

                    AlgoType* RESTRICT rR = pDstR + off;
                    AlgoType* RESTRICT rG = pDstG + off;
                    AlgoType* RESTRICT rB = pDstB + off;

                    // On a vertically-transported format the bar spans whole rows,
                    // so the row test is done once rather than per pixel.
                    if (false == w.transportAlongWidth)
                    {
                        const HighPrecType py =
                            static_cast<HighPrecType>(y) + 0.5;

                        if (py < lo || py > hi)
                            continue;

                        for (int32_t x = 0; x < sizeX; x++)
                        {
                            rR[x] *= t;
                            rG[x] *= t;
                            rB[x] *= t;
                        }
                    }
                    else
                    {
                        for (int32_t x = 0; x < sizeX; x++)
                        {
                            const HighPrecType px =
                                static_cast<HighPrecType>(x) + 0.5;

                            if (px < lo || px > hi)
                                continue;

                            rR[x] *= t;
                            rG[x] *= t;
                            rB[x] *= t;
                        }
                    }
                }
            }
        }
    }

    return;
}
