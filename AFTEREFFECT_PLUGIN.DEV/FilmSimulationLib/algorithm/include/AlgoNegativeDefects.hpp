#pragma once

// ---------------------------------------------------------------------------
//  AlgoNegativeDefects.hpp
//
//  Sub-stage 9b of the film simulation pipeline: negative-side defects.
//
//  THREE PARTICULATE CLASSES AND ONE ABRASION CLASS ARE NOW MODELLED
//
//      fine dust           the dominant class, and the one that carries the look
//      coarse debris       rare, large, opaque, hard-edged
//      hair and fibres     long, constant width, curved, with free ends
//      scratches           damage IN the film rather than on it: a transport
//                          tramline with a run length along the web, and single
//                          handling marks locked to the film
//
//  Since 2026-10-10 the stage also applies development mottle and air bells
//  (processingQuality), water spots (dryingMarks) and a one-record colour veil
//  (colourVeil), from AlgoProcessingDefects.hpp. Storage fade is NOT here: it is
//  a profile resolver (AlgoStorageAge.hpp), which storageSeverity feeds.
//
//  ⚠ THE SCRATCH CLASS IS THE ONE PLACE THIS STAGE HAS A TIME AXIS, and the rest
//  of this header's insistence that it has none is still correct. Dust, debris
//  and fibres are baked into the film and only TRANSLATE with the web; nothing is
//  born and nothing dies. A transport scratch is different in kind: it is abraded
//  by continuous contact while the film runs past a fixed object, so it exists
//  over a RUN of web - a length, expressed in frames through the frame pitch -
//  and outside that run the film is clean. That run length is what
//  TemporalSpec::scratch_persistence_frames is, and it is the only field in this
//  stage that reads the frame index for anything but the window position.
//  See the frameIndex note on the prototype below.
//
//  WHAT "NEGATIVE SIDE" MEANS, AND WHY IT IS A SEPARATE STAGE FROM 16
//
//  Damage divides by WHERE IT HAPPENED, and the division decides how it behaves
//  in time.
//
//  A negative-side defect is baked into the film emulsion: a particle embedded in
//  the gelatin while it was swollen and tacky during drying, a camera-gate
//  scratch, a processing mark. It was recorded once and is part of the image for
//  ever, so it survives printing, every subsequent generation duplicates it, and
//  it advances with the film.
//
//  A gate-side defect - stage 16 - happens at projection or telecine: loose dirt
//  on the gate, a hair in the light path. It is applied to the finished positive,
//  does not survive re-printing, and is tied to the machine rather than the film.
//
//  So this stage is applied BEFORE the print at 13, and that ordering is
//  load-bearing: the dupe chain and the print curve act on the damage exactly as
//  they act on the picture. Camera-original damage composited after the print
//  would be sharper and cleaner than the image around it, which reads as a
//  digital overlay instantly.
//
//  THE EMBEDDED / LOOSE SPLIT, AND WHY THIS STAGE ONLY GETS PART OF THE DUST
//
//  Measured dust divides into three temporal populations:
//
//      loose dust on the film during the scan   50 - 80 %   one frame only
//      embedded processing dust                 15 - 40 %   locked to the film
//      dirt lodged in the gate                   5 - 20 %   fixed for a long run
//
//  Only the embedded population is a property of the negative, so only that
//  population belongs here. The one-frame "sparkle" and the gate population are
//  machine-side and belong to stage 16. This stage therefore renders the embedded
//  fraction of the requested level, and the rest arrives when stage 16 is
//  written. Putting all of it here would make every particle survive the print
//  and duplicate through the generations, which is wrong for the majority of
//  real dirt.
//
//  Being explicit about that share also prevents a subtler error. If this stage
//  rendered the whole level, enabling dust at 1.0 would put roughly three times
//  the measured embedded density onto the negative, and it would all be
//  reproducible on re-scan - the exact opposite of the characteristic behaviour,
//  which is that most dirt moves between passes.
//
//  WHY THE DISTRIBUTION IS NOT UNIFORM, AND WHY THAT IS THE WHOLE POINT
//
//  Uniform random scatter is the single most recognisable failure of existing
//  film-emulation products. Real dirt arrives in patches: measured on blank film,
//  the local particle rate has a coefficient of variation near 0.9, so some
//  regions of a frame carry several times the average and others almost none.
//
//  Placement therefore runs through the clumped intensity field in
//  AlgoDefectField.hpp, and the field is keyed on INTEGER FILM COORDINATES rather
//  than on the frame index. A dirty patch of film stays dirty as it travels
//  through the gate, a particle straddling a frame line matches on both sides,
//  and nothing boils.
//
//  RANDOM, BUT NOT CONTROLLABLE, AND NOT REPEATING
//
//  Every particle draws its own size, opacity, shape harmonics, elongation,
//  orientation and colour from its own counter-based stream. Nothing about an
//  individual particle is exposed as a control: a user setting the amount of dust
//  is not choosing how any one speck looks, in the same way that loading a roll
//  of film is not choosing where the dirt lands.
//
//  Determinism and regularity are different things. Every draw is a pure function
//  of (seed, film cell, ordinal), so scrubbing the timeline and rendering frames
//  out of order are both stable, while no two particles and no two patches of
//  film are alike.
//
//  HOW A PARTICLE MEETS THE IMAGE
//
//  This stage works in DENSITY, which is what makes the whole thing cheap. An
//  opaque speck on the negative transmits a fraction (1 - alpha) of the light
//  that reaches it, and density is the negative base-ten logarithm of
//  transmittance, so the particle is exactly an additive density:
//
//      D' = D - log10(1 - alpha)
//
//  No new colour domain, no extra buffer, no separate compositing pass. Stacked
//  particles compose correctly by simple addition, because multiplying
//  transmittances is adding logarithms - so overlapping specks darken the way two
//  real specks would, without any special case.
//
//  Alpha is capped just below one rather than at one. A perfectly opaque particle
//  is an infinite density, which would poison every stage downstream; the cap
//  turns it into a large but finite value that the print stage clips normally.
//
//  DEFECTS ARE NOT ACHROMATIC, AND THIS SURPRISES PEOPLE
//
//  Measured on colour material, the ratio of the smallest to the largest
//  per-channel deviation of a detected speck has a median near 0.55. Particles
//  are coloured - lint is not grey, and a particle sitting in one emulsion layer
//  blocks that layer's light more than the others. Rendering dirt as a neutral
//  density is a visible tell, so each particle carries three channel weights.
//
//  EDGES ARE NEVER SHARPER THAN THE OPTICS
//
//  A particle boundary is rendered with a soft transition whose width is the
//  system point-spread function, floored at half a pixel. Nothing in a real
//  imaging chain has an edge sharper than its own PSF, and a hard-edged speck is
//  the second most reliable way to make dust look composited - the first being
//  uniform placement.
//
//  Shapes are irregular by construction: a radius modulated by low-order angular
//  harmonics, with elongation up to 3:1 for dust and a lobed, concave outline for
//  debris. Circles read as digital immediately.
//
//  ORIENTATION FOLLOWS THE TRANSPORT AXIS
//
//  The mildly elongated members of the measured population are not isotropic:
//  they split 54 per cent along the transport axis against 17 per cent across it,
//  inheriting the same longitudinal bias as the scratches. Elongated particles
//  therefore take a biased orientation, and because the film coordinate system
//  knows which image axis the film runs along, that bias comes out horizontal on
//  a 35 mm still and vertical on every cine format with no per-format code.
//
//  FORMAT SCALING IS AUTOMATIC
//
//  Everything is placed per square millimetre of film over the window this frame
//  occupies, so a 16 mm frame receives about eleven times fewer particles than a
//  35 mm still frame while each one covers about 3.3 times more of the frame
//  width. Both halves of that fall out of the geometry; neither is coded.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// The single source of the engine's numeric types and alignment policy.
#include "AlgoTypes.hpp"
// 2026-10-20c: this header's compile-time constants live in film_damage_constants.hpp.
#include "film_damage_constants.hpp"

// Buffer layout and the geometry fields that travel with it.
#include "AlgoMemHandler.hpp"

// User-facing controls, pre-validated by the caller.
#include "AlgoControl.hpp"

// Film-fixed coordinates: the window on the film and the transport rotation.
#include "AlgoFilmCoord.hpp"

// The clumped spatial process and the sampling primitives.
#include "AlgoDefectField.hpp"

// Stock parameters, including TemporalSpec and AgingSpec.
#include "film_profiles.hpp"

// 2026-10-20b: truncated log-normal and the stochastic primitives.
#include "AlgoDefectOptics.hpp"
#include <cmath>

#include <cstdint>   // int32_t, uint32_t

// Radius table of one outline, relative to the particle's mean radius.
// n == 0: no table (factor 1), which is also the value-initialised state, so
// any particle that does not ask for an outline renders as before.
struct AlgoDefectOutline
{
    int32_t      n;
    HighPrecType rMax;
    HighPrecType r[ALGO_OUTLINE_SAMPLES];
};

inline void AlgoDefectOutlineMake
(
    const uint32_t     seed,
    const HighPrecType radiusPx,
    AlgoDefectOutline& o
) noexcept
{
    o.n    = 0;
    o.rMax = 1.0;

    if (!(radiusPx >= ALGO_OUTLINE_MIN_RADIUS_PX))
        return;

    const HighPrecType twoPi = 6.283185307179586476925286766559;

    // Polygon base.
    const int32_t span = ALGO_OUTLINE_VERTEX_MAX - ALGO_OUTLINE_VERTEX_MIN + 1;
    const int32_t K = ALGO_OUTLINE_VERTEX_MIN + MIN_VALUE(static_cast<int32_t>(
        AlgoRngUniform01(AlgoDefectHash(seed, 0, 0, ALGO_DEFECT_TAG_OUTLINE))
        * static_cast<HighPrecType>(span)), span - 1);

    HighPrecType va[ALGO_OUTLINE_VERTEX_MAX + 1];
    HighPrecType vx[ALGO_OUTLINE_VERTEX_MAX + 1];
    HighPrecType vy[ALGO_OUTLINE_VERTEX_MAX + 1];

    for (int32_t k = 0; k < K; k++)
    {
        const HighPrecType uj = AlgoRngUniform01(AlgoDefectHash(seed, 1, k, ALGO_DEFECT_TAG_OUTLINE));
        const HighPrecType ur = AlgoRngUniform01(AlgoDefectHash(seed, 2, k, ALGO_DEFECT_TAG_OUTLINE));
        const HighPrecType a  = twoPi * (static_cast<HighPrecType>(k)
                              + ALGO_OUTLINE_VERTEX_JITTER * (uj - 0.5))
                              / static_cast<HighPrecType>(K);
        const HighPrecType rv = std::exp(ALGO_OUTLINE_VERTEX_SIGMA_LN
                                         * AlgoDefectNormalQuantile(ur));
        va[k] = a;
        vx[k] = rv * std::cos(a);
        vy[k] = rv * std::sin(a);
    }

    // Close the ring one turn later.
    va[K] = va[0] + twoPi;
    vx[K] = vx[0];
    vy[K] = vy[0];

    // Roughness harmonics.
    const int32_t mTop = MIN_VALUE(ALGO_OUTLINE_MMAX,
        static_cast<int32_t>(std::floor(3.14159265358979323846 * radiusPx)));

    HighPrecType amp[ALGO_OUTLINE_MMAX + 1];
    HighPrecType phs[ALGO_OUTLINE_MMAX + 1];

    // Rayleigh with median 1: sigma = 1 / sqrt(2 ln 2).
    const HighPrecType raySigma = 0.84932180028801904272;

    for (int32_t m = ALGO_OUTLINE_MMIN; m <= mTop; m++)
    {
        const HighPrecType u1 = MAX_VALUE(AlgoRngUniform01(
            AlgoDefectHash(seed, 3, m, ALGO_DEFECT_TAG_OUTLINE)), 1.0e-12);
        const HighPrecType u2 = AlgoRngUniform01(
            AlgoDefectHash(seed, 4, m, ALGO_DEFECT_TAG_OUTLINE));

        amp[m] = ALGO_OUTLINE_ROUGH_C
               * std::pow(static_cast<HighPrecType>(m), -ALGO_OUTLINE_ROUGH_BETA)
               * raySigma * std::sqrt(-2.0 * std::log(u1));
        phs[m] = twoPi * u2;
    }

    // Each sample finds its own polygon edge (K <= 12, so a linear search).
    HighPrecType sum = 0.0;
    {
        for (int32_t i = 0; i < ALGO_OUTLINE_SAMPLES; i++)
        {
            // Angle of sample i, expressed in [va[0], va[0] + 2 pi) so the ring
            // of edges va[0] .. va[K] covers it exactly once.
            HighPrecType th = twoPi * static_cast<HighPrecType>(i)
                            / static_cast<HighPrecType>(ALGO_OUTLINE_SAMPLES);
            th = va[0] + (th - va[0]) - twoPi * std::floor((th - va[0]) / twoPi);

            int32_t s = 0;
            while ((s < K - 1) && (th >= va[s + 1]))
                s++;

            const HighPrecType ux = std::cos(th);
            const HighPrecType uy = std::sin(th);
            const HighPrecType ex = vx[s + 1] - vx[s];
            const HighPrecType ey = vy[s + 1] - vy[s];
            // lambda u = P_s + t E  =>  lambda = (P_s x E) / (u x E).
            const HighPrecType den = ux * ey - uy * ex;
            const HighPrecType num = vx[s] * vy[s + 1] - vy[s] * vx[s + 1];
            HighPrecType rp = (std::fabs(den) > 1.0e-12) ? (num / den) : 1.0;

            if (!(rp > 0.0))
                rp = 1.0;

            HighPrecType rough = 1.0;

            for (int32_t m = ALGO_OUTLINE_MMIN; m <= mTop; m++)
                rough += amp[m] * std::cos(static_cast<HighPrecType>(m) * th + phs[m]);

            o.r[i] = rp * MAX_VALUE(rough, ALGO_OUTLINE_FLOOR);
            sum   += o.r[i];
        }
    }

    const HighPrecType inv = static_cast<HighPrecType>(ALGO_OUTLINE_SAMPLES) / sum;
    HighPrecType rMax = 0.0;

    for (int32_t i = 0; i < ALGO_OUTLINE_SAMPLES; i++)
    {
        o.r[i] *= inv;
        rMax = MAX_VALUE(rMax, o.r[i]);
    }

    o.rMax = rMax;
    o.n    = ALGO_OUTLINE_SAMPLES;
}

// Outline factor at angle theta (radians, any range): linear interpolation.
inline HighPrecType AlgoDefectOutlineAt
(
    const AlgoDefectOutline& o,
    const HighPrecType       theta
) noexcept
{
    if (o.n <= 0)
        return 1.0;

    const HighPrecType invTwoPi = 0.15915494309189533576888376337251;
    HighPrecType t = theta * invTwoPi;
    t -= std::floor(t);
    t *= static_cast<HighPrecType>(o.n);

    const int32_t i0 = MIN_VALUE(static_cast<int32_t>(t), o.n - 1);
    const int32_t i1 = (i0 + 1 == o.n) ? 0 : (i0 + 1);
    const HighPrecType f = t - static_cast<HighPrecType>(i0);

    return o.r[i0] + (o.r[i1] - o.r[i0]) * f;
}

// Texture fade with local opacity (exactly 1 at or below the knee).
inline HighPrecType AlgoDefectTextureFade (const HighPrecType a) noexcept
{
    if (a <= ALGO_DEFECT_TEXTURE_FADE_ALPHA)
        return 1.0;

    return CLAMP_VALUE((1.0 - a) / (1.0 - ALGO_DEFECT_TEXTURE_FADE_ALPHA), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
//  TRANSPORT SCRATCH EVENT SCHEDULE - 2026-10-20b. Pure functions of
//  (seed, mean gap, ordinal / time), shared by stage 9b and the tests, so what
//  the tests measure is exactly what renders. See the constants block above.
//
//  E_k - E_(k-1) = AlgoScratchGap(k) for every integer k; E_0 lies at a random
//  phase inside its own gap, so t = 0 is not special.
// ---------------------------------------------------------------------------
inline HighPrecType AlgoScratchGap
(
    const uint32_t     seed,
    const HighPrecType meanGap,
    const int32_t      k
) noexcept
{
    return meanGap * (ALGO_SCRATCH_GAP_LO
        + (ALGO_SCRATCH_GAP_HI - ALGO_SCRATCH_GAP_LO)
          * AlgoRngUniform01(AlgoDefectHash(seed, 1, k, ALGO_DEFECT_TAG_SCRATCH_GAP)));
}

inline HighPrecType AlgoScratchEvent0
(
    const uint32_t     seed,
    const HighPrecType meanGap
) noexcept
{
    return AlgoScratchGap(seed, meanGap, 0) * AlgoRngUniform01(
        AlgoDefectHash(seed, 2, 0, ALGO_DEFECT_TAG_SCRATCH_GAP));
}

// The newest event at or before t: returns its ordinal and writes its time.
// O(|t| / meanGap) gap draws from event 0, bounded at 1e7 steps.
inline int32_t AlgoScratchLastEvent
(
    const uint32_t     seed,
    const HighPrecType meanGap,
    const HighPrecType t,
    HighPrecType&      eLastOut
) noexcept
{
    constexpr int64_t kWalkMax = 10000000;

    int32_t      kLast = 0;
    HighPrecType eLast = AlgoScratchEvent0(seed, meanGap);
    int64_t      steps = 0;

    if (t >= eLast)
    {
        for (;;)
        {
            const HighPrecType eNext = eLast + AlgoScratchGap(seed, meanGap, kLast + 1);

            if (eNext > t || ++steps > kWalkMax)
                break;

            eLast = eNext;
            kLast++;
        }
    }
    else
    {
        while (eLast > t && ++steps <= kWalkMax)
        {
            eLast -= AlgoScratchGap(seed, meanGap, kLast);
            kLast--;
        }
    }

    eLastOut = eLast;
    return kLast;
}

// Lifetime of event k, seconds: log-normal with MEAN lifeMean, truncated to
// [LIFE_LO, LIFE_HI] x mean (keyed on the event's own counter sc).
inline HighPrecType AlgoScratchLifetime
(
    const uint64_t     sc,
    const HighPrecType lifeMean
) noexcept
{
    const HighPrecType median = lifeMean * ALGO_SCRATCH_LIFE_MEAN_CORR * std::exp(
        -0.5 * ALGO_SCRATCH_LIFE_SIGMA_LN * ALGO_SCRATCH_LIFE_SIGMA_LN);

    return AlgoDefectLogNormalTrunc(sc + 16u, median, ALGO_SCRATCH_LIFE_SIGMA_LN,
                                    ALGO_SCRATCH_LIFE_LO * lifeMean,
                                    ALGO_SCRATCH_LIFE_HI * lifeMean);
}

// ---------------------------------------------------------------------------
//  Sub-stage 9b: negative-side defects.
//
//  pSrcR/G/B      density in
//  pDstR/G/B      density out
//  sizeX/sizeY    active pixel extent
//  pitch          row stride in ELEMENTS
//  profile        stock being simulated
//  params         user controls; the film-damage group and damageSeed
//  negWidthMm     frame width on the film, so defect sizes scale with gauge
//  negHeightMm    frame height on the film
//  framePitchMm   web advance per frame, for defects that drift along the film
//  pxPerMm        render resolution
//  frameIndex     clip-relative frame number, used to DRIFT defects along the
//                 web (frameIndex * framePitchMm)
//
//                 ⚠ THIS LINE ONCE READ "defects have lifetimes", WHICH WAS
//                 FALSE OF THE PARTICULATE CLASSES AND STILL IS. Corrected
//                 2026-09-11. Dust, debris and fibres have no birth and no
//                 death: each is baked into the film and translates with the
//                 web, which is exactly why this stage takes the FILM frame
//                 rate rather than the projection one.
//
//                 ⚠ QUEUE P44 IS NOW CLOSED, AND NOT BY WIRING THE FIELD INTO
//                 THE OLD SENTENCE. The claim cost something concrete: a
//                 2026-09-11 field audit read that comment and listed
//                 `TemporalSpec::scratch_persistence_frames` - populated on all
//                 191 stocks - as a field the stage "can use directly". It
//                 could not, because the particulate classes have no lifetime
//                 for it to set, and P44 said correctly that giving it one is a
//                 modelling decision rather than a wiring job.
//
//                 The decision taken is that the field is NOT a birth/death
//                 lifetime for a particle. Its docstring calls it the "mean
//                 lifetime of a RUNNING SCRATCH", and a running scratch is
//                 abraded by continuous contact as the web passes a fixed
//                 object - so it is a RUN LENGTH ALONG THE FILM, which the
//                 frame pitch turns into a length in millimetres. A new
//                 SCRATCH class consumes it that way: each transport scratch
//                 has a start frame and a run, is present while
//                 start <= frameIndex < start + run, and marks exactly the
//                 stretch of web its abrader touched. The particulate classes
//                 are untouched and remain lifetime-free.
//  frameRate      frames per second OF FILM - correct here, because this damage
//                 is baked into the film rather than happening at projection
//  seed           per-call seed, combined with params.damageSeed by the generator
// ---------------------------------------------------------------------------
void AlgoStage09b_NegativeDefects
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
    const AlgoType           framePitchMm,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
    // 2026-10-20b: optional OUTPUT, the additive defect density (material on
    // or in the film, burnished base) this stage put on the image, for the
    // grain stage (see AlgoStage11_Grain). nullptr = not wanted. Removals are
    // excluded (floored at zero).
    ,
    AlgoType*                pDefR = nullptr,
    AlgoType*                pDefG = nullptr,
    AlgoType*                pDefB = nullptr
) noexcept;
