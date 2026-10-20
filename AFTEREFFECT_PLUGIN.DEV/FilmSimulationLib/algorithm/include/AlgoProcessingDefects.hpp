#pragma once

// ---------------------------------------------------------------------------
//  AlgoProcessingDefects.hpp
//
//  Three negative-side damage classes for sub-stage 9b, added 2026-10-10 so
//  that no FilmDamage control is left without a reader:
//
//      processingQuality   development mottle and air bells
//      dryingMarks         water spots with a tide line
//      colourVeil          a flat density offset in ONE dye record
//
//  Header only, SHARED by the scalar and the AVX2 trees. Every computation is
//  HighPrecType in both builds and only the store into the AlgoType plane
//  rounds - the same arrangement the 9b particulate and scratch rasterisers
//  already use, for the same reason: these classes scatter into small boxes
//  or walk a smooth field, and there is nothing to pack a register with.
//
//  DOMAIN. Stage 9b works on ABSOLUTE negative density per record, D_min
//  included. A development effect acts on the DEVELOPED part only, so mottle
//  and air bells scale the net density D - D_min and never touch the base and
//  mask; a drying residue and a veil are additive.
//
//  ⚠ EVERY AMPLITUDE, RATE AND SIZE IN THIS FILE IS A DEFINITION, NOT A
//  MEASUREMENT. The corpus holds no measured figure for any of them: no
//  database field exists for drying marks, processing.progress is UNKNOWN on
//  most stocks, and no source gives a mottle spectrum or an air-bell rate.
//  Each constant below states what it defines, so that a level of 1.0 means a
//  stated, reproducible amount of damage - not a claim about any real roll.
//  Replace a constant with a measured figure when one is digitised; nothing
//  else needs to change.
//
//  TIME. Every class here is LOCKED TO THE FILM: it is keyed on film
//  coordinates (AlgoFilmCoord.hpp), so it advances with the web from frame to
//  frame and a re-render of the same frame reproduces it exactly. The veil is
//  a roll property and is the same on every frame.
//
//  INERT unless filmDamageEnabled, damageStrength > 0 and the class level > 0
//  - the caller (Algo_09_Sim.cpp) gates all three before calling.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType.
#include "AlgoTypes.hpp"
// 2026-10-20c: this header's compile-time constants live in film_damage_constants.hpp.
#include "film_damage_constants.hpp"

// Counter-based generator: every draw is a pure function of its key.
#include "AlgoCounterRng.hpp"

// The frame's window on the film, and film-to-pixel mapping.
#include "AlgoFilmCoord.hpp"

#include <cmath>     // std::floor, std::sqrt, std::exp, std::log
#include <cstddef>   // std::ptrdiff_t
#include <cstdint>   // int32_t, int64_t, uint32_t, uint64_t

// ===========================================================================
//  Small helpers
// ===========================================================================

// Cubic smooth step on 0..1.
inline HighPrecType AlgoProcSmooth (const HighPrecType t) noexcept
{
    const HighPrecType u = CLAMP_VALUE(t, 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
}

// One 64-bit key for a 2-D integer cell of one stream.
inline uint64_t AlgoProcCellKey
(
    const uint32_t        seed,
    const eALGO_RNG_STAGE stage,
    const uint32_t        sub,
    const int64_t         ia,
    const int64_t         ic
) noexcept
{
    const uint64_t base = AlgoRngCounter(seed, 0, stage, sub);
    const uint64_t ka   = static_cast<uint64_t>(ia) * ALGO_RNG_GOLDEN;
    const uint64_t kc   = static_cast<uint64_t>(ic) * ALGO_RNG_MIX_2;
    return AlgoRngMix64(base ^ ka) ^ kc;
}

// Uniform in [-1, 1) at one lattice node.
inline HighPrecType AlgoProcLattice
(
    const uint32_t        seed,
    const eALGO_RNG_STAGE stage,
    const uint32_t        sub,
    const int64_t         ia,
    const int64_t         ic
) noexcept
{
    return 2.0 * AlgoRngUniform01(AlgoProcCellKey(seed, stage, sub, ia, ic)) - 1.0;
}

// Poisson draw by inversion. A pure function of its key; the mean is small.
inline int32_t AlgoProcPoisson (const uint64_t key, const HighPrecType mean) noexcept
{
    if (mean <= 0.0)
        return 0;

    const HighPrecType u = AlgoRngUniform01(key);
    HighPrecType p   = std::exp(-mean);
    HighPrecType cdf = p;
    int32_t      k   = 0;

    while ((u > cdf) && (k < ALGO_PROCDEF_MAX_PER_CELL))
    {
        k++;
        p   *= mean / static_cast<HighPrecType>(k);
        cdf += p;
    }

    return k;
}

// Pixel centre to film coordinates in the current window (the inverse of
// AlgoFilmToPixel).
inline void AlgoProcPixelToFilm
(
    const AlgoFilmWindow& w,
    const int32_t         x,
    const int32_t         y,
    HighPrecType&         along,
    HighPrecType&         across
) noexcept
{
    const HighPrecType mm = static_cast<HighPrecType>(w.mmPerPx);
    const HighPrecType px = (static_cast<HighPrecType>(x) + 0.5) * mm;
    const HighPrecType py = (static_cast<HighPrecType>(y) + 0.5) * mm;

    if (w.transportAlongWidth)
    {
        along  = w.alongMin  + px;
        across = w.acrossMin + py;
    }
    else
    {
        along  = w.alongMin  + py;
        across = w.acrossMin + px;
    }

    return;
}

// ---------------------------------------------------------------------------
//  One octave of value noise, with a four-node cache.
//
//  Within one image row one film coordinate is constant, so consecutive
//  pixels share their lattice cell for cellMm * pxPerMm pixels and the four
//  node values are drawn once per cell rather than once per pixel.
// ---------------------------------------------------------------------------
struct AlgoProcNoiseOctave
{
    HighPrecType    cellMm;
    uint32_t        seed;
    eALGO_RNG_STAGE stage;
    uint32_t        sub;
    int64_t         ia;
    int64_t         ic;
    bool            valid;
    HighPrecType    v00, v10, v01, v11;

    HighPrecType eval (const HighPrecType along, const HighPrecType across) noexcept
    {
        const HighPrecType fa = along  / cellMm;
        const HighPrecType fc = across / cellMm;
        const HighPrecType la = std::floor(fa);
        const HighPrecType lc = std::floor(fc);
        const int64_t      na = static_cast<int64_t>(la);
        const int64_t      nc = static_cast<int64_t>(lc);

        if ((false == valid) || (na != ia) || (nc != ic))
        {
            ia = na;
            ic = nc;
            v00 = AlgoProcLattice(seed, stage, sub, na,     nc);
            v10 = AlgoProcLattice(seed, stage, sub, na + 1, nc);
            v01 = AlgoProcLattice(seed, stage, sub, na,     nc + 1);
            v11 = AlgoProcLattice(seed, stage, sub, na + 1, nc + 1);
            valid = true;
        }

        const HighPrecType ta = AlgoProcSmooth(fa - la);
        const HighPrecType tc = AlgoProcSmooth(fc - lc);
        const HighPrecType a0 = v00 + (v10 - v00) * ta;
        const HighPrecType a1 = v01 + (v11 - v01) * ta;
        return a0 + (a1 - a0) * tc;
    }
};

// ---------------------------------------------------------------------------
//  Visit every placement cell overlapping the window plus a margin, and every
//  mark in it. `emit(key, along, across)` receives the mark's own stream key
//  and its film position.
// ---------------------------------------------------------------------------
template <typename EmitFn>
inline void AlgoProcForEachMark
(
    const AlgoFilmWindow& w,
    const uint32_t        seed,
    const eALGO_RNG_STAGE stage,
    const HighPrecType    perCm2,
    const HighPrecType    marginMm,
    EmitFn&&              emit
) noexcept
{
    const HighPrecType cell = ALGO_PROCDEF_CELL_MM;
    const int64_t a0 = static_cast<int64_t>(std::floor((w.alongMin  - marginMm) / cell));
    const int64_t a1 = static_cast<int64_t>(std::floor((w.alongMax  + marginMm) / cell));
    const int64_t c0 = static_cast<int64_t>(std::floor((w.acrossMin - marginMm) / cell));
    const int64_t c1 = static_cast<int64_t>(std::floor((w.acrossMax + marginMm) / cell));

    // Area of one cell in cm^2 is exactly 1 (10 mm x 10 mm).
    const HighPrecType mean = perCm2 * (cell * cell) / 100.0;

    for (int64_t ia = a0; ia <= a1; ia++)
    {
        for (int64_t ic = c0; ic <= c1; ic++)
        {
            const uint64_t ck = AlgoProcCellKey(seed, stage, 0u, ia, ic);
            const int32_t  n  = AlgoProcPoisson(ck, mean);

            for (int32_t j = 0; j < n; j++)
            {
                // Each mark draws from its own sub-stream; ordinal 0 is the count.
                const uint64_t mk = AlgoRngMix64(ck ^ (static_cast<uint64_t>(j + 1)
                                                       * ALGO_RNG_MIX_1));
                const HighPrecType along  =
                    (static_cast<HighPrecType>(ia) + AlgoRngUniform01(mk))      * cell;
                const HighPrecType across =
                    (static_cast<HighPrecType>(ic) + AlgoRngUniform01(mk + 1u)) * cell;
                emit(mk, along, across);
            }
        }
    }

    return;
}

// ---------------------------------------------------------------------------
//  Rasterise a radial profile around a film position. `fn(rPx, wR, wG, wB)`
//  is called for every pixel within maxRadiusPx of the centre with that
//  pixel's radius and its three density values by reference.
// ---------------------------------------------------------------------------
template <typename PixFn>
inline void AlgoProcRadial
(
    const AlgoFilmWindow& w,
    const HighPrecType    along,
    const HighPrecType    across,
    const HighPrecType    maxRadiusPx,
    AlgoType* RESTRICT    pR,
    AlgoType* RESTRICT    pG,
    AlgoType* RESTRICT    pB,
    const int32_t         sizeX,
    const int32_t         sizeY,
    const int32_t         pitch,
    PixFn&&               fn
) noexcept
{
    HighPrecType cx = 0.0;
    HighPrecType cy = 0.0;
    AlgoFilmToPixel(w, along, across, cx, cy);

    const int32_t x0 = MAX_VALUE(static_cast<int32_t>(std::floor(cx - maxRadiusPx)), 0);
    const int32_t x1 = MIN_VALUE(static_cast<int32_t>(std::floor(cx + maxRadiusPx)), sizeX - 1);
    const int32_t y0 = MAX_VALUE(static_cast<int32_t>(std::floor(cy - maxRadiusPx)), 0);
    const int32_t y1 = MIN_VALUE(static_cast<int32_t>(std::floor(cy + maxRadiusPx)), sizeY - 1);

    if ((x0 > x1) || (y0 > y1))
        return;

    for (int32_t y = y0; y <= y1; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;
        const HighPrecType   dy  = static_cast<HighPrecType>(y) + 0.5 - cy;

        for (int32_t x = x0; x <= x1; x++)
        {
            const HighPrecType dx = static_cast<HighPrecType>(x) + 0.5 - cx;
            const HighPrecType r  = std::sqrt(dx * dx + dy * dy);

            if (r > maxRadiusPx)
                continue;

            fn(r, pR[off + x], pG[off + x], pB[off + x]);
        }
    }

    return;
}

// ===========================================================================
//  processingQuality: development mottle + air bells
// ===========================================================================
inline void AlgoProcessingDefects
(
    const AlgoFilmWindow& w,
    const HighPrecType    level,       // processingQuality x damageStrength
    const uint32_t        seed,
    const HighPrecType    dminR,
    const HighPrecType    dminG,
    const HighPrecType    dminB,
    AlgoType* RESTRICT    pR,
    AlgoType* RESTRICT    pG,
    AlgoType* RESTRICT    pB,
    const int32_t         sizeX,
    const int32_t         sizeY,
    const int32_t         pitch
) noexcept
{
    if (level <= 0.0)
        return;

    // ---- mottle: one development factor per position, the same on all three
    // records (agitation moves one developer through every layer) -----------
    const HighPrecType amp = ALGO_MOTTLE_AMPLITUDE * level;

    for (int32_t y = 0; y < sizeY; y++)
    {
        AlgoProcNoiseOctave lo = { ALGO_MOTTLE_CELL_MM_LO, seed,
                                   eALGO_RNG_STAGE::eRNG_PROCESS_DEFECTS, 1u,
                                   0, 0, false, 0.0, 0.0, 0.0, 0.0 };
        AlgoProcNoiseOctave hi = { ALGO_MOTTLE_CELL_MM_HI, seed,
                                   eALGO_RNG_STAGE::eRNG_PROCESS_DEFECTS, 2u,
                                   0, 0, false, 0.0, 0.0, 0.0, 0.0 };

        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;
        AlgoType* RESTRICT rR = pR + off;
        AlgoType* RESTRICT rG = pG + off;
        AlgoType* RESTRICT rB = pB + off;

        for (int32_t x = 0; x < sizeX; x++)
        {
            HighPrecType along = 0.0;
            HighPrecType across = 0.0;
            AlgoProcPixelToFilm(w, x, y, along, across);

            const HighPrecType n = ALGO_MOTTLE_WEIGHT_LO * lo.eval(along, across)
                                 + ALGO_MOTTLE_WEIGHT_HI * hi.eval(along, across);
            const HighPrecType f = 1.0 + amp * n;

            rR[x] = static_cast<AlgoType>(dminR + (static_cast<HighPrecType>(rR[x]) - dminR) * f);
            rG[x] = static_cast<AlgoType>(dminG + (static_cast<HighPrecType>(rG[x]) - dminG) * f);
            rB[x] = static_cast<AlgoType>(dminB + (static_cast<HighPrecType>(rB[x]) - dminB) * f);
        }
    }

    // ---- air bells: a bubble on the emulsion keeps developer off a disc, so
    // the net density there is reduced toward D_min ---------------------------
    const HighPrecType pxPerMm = static_cast<HighPrecType>(w.pxPerMm);

    AlgoProcForEachMark(w, seed, eALGO_RNG_STAGE::eRNG_PROCESS_DEFECTS,
                        ALGO_AIRBELL_PER_CM2 * level, ALGO_AIRBELL_R_MAX_MM,
        [&](const uint64_t mk, const HighPrecType along, const HighPrecType across) noexcept
        {
            const HighPrecType rMm = ALGO_AIRBELL_R_MIN_MM
                + (ALGO_AIRBELL_R_MAX_MM - ALGO_AIRBELL_R_MIN_MM) * AlgoRngUniform01(mk + 2u);
            const HighPrecType depth = ALGO_AIRBELL_DEPTH_MIN
                + (ALGO_AIRBELL_DEPTH_MAX - ALGO_AIRBELL_DEPTH_MIN) * AlgoRngUniform01(mk + 3u);

            const HighPrecType rPx    = rMm * pxPerMm;
            const HighPrecType edgePx = MAX_VALUE(ALGO_AIRBELL_EDGE_FRAC * rPx,
                                                  ALGO_PROCDEF_EDGE_MIN_PX);

            AlgoProcRadial(w, along, across, rPx + edgePx, pR, pG, pB, sizeX, sizeY, pitch,
                [&](const HighPrecType r, AlgoType& dR, AlgoType& dG, AlgoType& dB) noexcept
                {
                    const HighPrecType c    = 1.0 - AlgoProcSmooth((r - (rPx - edgePx)) / (2.0 * edgePx));
                    const HighPrecType keep = 1.0 - depth * c;
                    dR = static_cast<AlgoType>(dminR + (static_cast<HighPrecType>(dR) - dminR) * keep);
                    dG = static_cast<AlgoType>(dminG + (static_cast<HighPrecType>(dG) - dminG) * keep);
                    dB = static_cast<AlgoType>(dminB + (static_cast<HighPrecType>(dB) - dminB) * keep);
                });
        });

    return;
}

// ===========================================================================
//  dryingMarks: water spots with a tide line
//
//  A drop left on the film dries from the edge inward and deposits its
//  dissolved solids as a ring. The residue scatters and absorbs on all three
//  records alike, so the density added is neutral.
// ===========================================================================
inline void AlgoDryingMarks
(
    const AlgoFilmWindow& w,
    const HighPrecType    level,       // dryingMarks x damageStrength
    const uint32_t        seed,
    AlgoType* RESTRICT    pR,
    AlgoType* RESTRICT    pG,
    AlgoType* RESTRICT    pB,
    const int32_t         sizeX,
    const int32_t         sizeY,
    const int32_t         pitch
) noexcept
{
    if (level <= 0.0)
        return;

    const HighPrecType pxPerMm = static_cast<HighPrecType>(w.pxPerMm);

    AlgoProcForEachMark(w, seed, eALGO_RNG_STAGE::eRNG_DRYING_MARKS,
                        ALGO_DRYING_PER_CM2 * level,
                        ALGO_DRYING_R_MAX_MM + 3.0 * ALGO_DRYING_RIM_MM,
        [&](const uint64_t mk, const HighPrecType along, const HighPrecType across) noexcept
        {
            const HighPrecType rMm = ALGO_DRYING_R_MIN_MM
                + (ALGO_DRYING_R_MAX_MM - ALGO_DRYING_R_MIN_MM) * AlgoRngUniform01(mk + 2u);
            const HighPrecType crest = ALGO_DRYING_RIM_D_MIN
                + (ALGO_DRYING_RIM_D_MAX - ALGO_DRYING_RIM_D_MIN) * AlgoRngUniform01(mk + 3u);

            const HighPrecType rPx = rMm * pxPerMm;
            const HighPrecType wPx = MAX_VALUE(ALGO_DRYING_RIM_MM * pxPerMm,
                                               ALGO_PROCDEF_EDGE_MIN_PX);

            AlgoProcRadial(w, along, across, rPx + 3.0 * wPx, pR, pG, pB, sizeX, sizeY, pitch,
                [&](const HighPrecType r, AlgoType& dR, AlgoType& dG, AlgoType& dB) noexcept
                {
                    const HighPrecType u      = (r - rPx) / wPx;
                    const HighPrecType ring   = std::exp(-u * u);
                    const HighPrecType inside = 1.0 - AlgoProcSmooth(u + 1.0);
                    const HighPrecType dd     = crest * (ring + ALGO_DRYING_INSIDE_FRAC * inside);
                    dR = static_cast<AlgoType>(static_cast<HighPrecType>(dR) + dd);
                    dG = static_cast<AlgoType>(static_cast<HighPrecType>(dG) + dd);
                    dB = static_cast<AlgoType>(static_cast<HighPrecType>(dB) + dd);
                });
        });

    return;
}

// ===========================================================================
//  colourVeil: a flat additive density offset in ONE dye record
//
//  AlgoControl.hpp item 9: a level shift with no neutral point anywhere in the
//  tonal range, and level only - contrast in the record is unchanged. Which
//  record is a roll property, drawn once from the damage seed. A monochrome
//  stock has one record, so there the offset is neutral (all three planes).
// ===========================================================================
inline void AlgoColourVeil
(
    const HighPrecType    level,       // colourVeil x damageStrength, density
    const uint32_t        seed,
    const bool            monochrome,
    AlgoType* RESTRICT    pR,
    AlgoType* RESTRICT    pG,
    AlgoType* RESTRICT    pB,
    const int32_t         sizeX,
    const int32_t         sizeY,
    const int32_t         pitch
) noexcept
{
    if (level <= 0.0)
        return;

    const uint64_t key = AlgoRngCounter(seed, 0, eALGO_RNG_STAGE::eRNG_COLOUR_VEIL, 0u);
    const int32_t  rec = MIN_VALUE(static_cast<int32_t>(3.0 * AlgoRngUniform01(key)), 2);

    const HighPrecType addR = (monochrome || (rec == 0)) ? level : 0.0;
    const HighPrecType addG = (monochrome || (rec == 1)) ? level : 0.0;
    const HighPrecType addB = (monochrome || (rec == 2)) ? level : 0.0;

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;
        AlgoType* RESTRICT rR = pR + off;
        AlgoType* RESTRICT rG = pG + off;
        AlgoType* RESTRICT rB = pB + off;

        for (int32_t x = 0; x < sizeX; x++)
        {
            rR[x] = static_cast<AlgoType>(static_cast<HighPrecType>(rR[x]) + addR);
            rG[x] = static_cast<AlgoType>(static_cast<HighPrecType>(rG[x]) + addG);
            rB[x] = static_cast<AlgoType>(static_cast<HighPrecType>(rB[x]) + addB);
        }
    }

    return;
}
