#pragma once

// ---------------------------------------------------------------------------
//  AlgoScannerArtifacts.hpp
//
//  damage.scannerArtifacts, consumed at the end of stage 16 (2026-10-10).
//  Three digitisation artefacts, applied to the finished positive in
//  TRANSMITTANCE, multiplicatively:
//
//      illumination shading   a radial fall-off of the scanner's light field
//      fixed-pattern banding  one gain per sensor column and per colour line,
//                             as a trilinear line-scan sensor produces
//      Newton's rings         interference between film base and carrier
//                             glass; ring radius goes as sqrt(k * lambda), so
//                             the three channels ring at different radii
//
//  Header only, SHARED by both trees, HighPrecType arithmetic in both.
//
//  ⚠ A SCANNER LAYER, NOT A FILM DEFECT. AlgoControl.hpp item 9: "clean scan
//  of damaged film" and "good film, cheap scanner" are both common requests.
//  So nothing here is keyed on the film window: shading and banding live in
//  image coordinates, the rings in millimetres of the carrier plane. And
//  nothing here depends on the frame index - the same scanner, the same
//  pattern on every frame of the roll.
//
//  ⚠ EVERY AMPLITUDE AND SIZE BELOW IS A DEFINITION, NOT A MEASUREMENT. No
//  database field and no source in the corpus gives a figure for any of them.
//  The three nominal channel wavelengths only set the RATIO of ring radii
//  through the sqrt(lambda) law.
// ---------------------------------------------------------------------------

#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoTypes.hpp"
#include "AlgoCounterRng.hpp"

#include <cmath>     // std::sqrt, std::cos, std::exp
#include <cstddef>   // std::ptrdiff_t
#include <cstdint>   // int32_t, uint32_t, uint64_t


// ---- definitions -----------------------------------------------------------
// Transmittance loss in the corners at scannerArtifacts x strength = 1.
constexpr HighPrecType ALGO_SCAN_SHADING_CORNER = 0.06;
// Peak per-column gain error at level 1 (gains uniform in [1 - a, 1 + a]).
constexpr HighPrecType ALGO_SCAN_BANDING_AMP    = 0.004;
// Newton's rings: modulation depth at level 1, first-ring radius at the green
// wavelength, and the Gaussian radius of the contact patch, film millimetres.
constexpr HighPrecType ALGO_SCAN_RINGS_AMP      = 0.03;
constexpr HighPrecType ALGO_SCAN_RINGS_R1_MM    = 1.0;
constexpr HighPrecType ALGO_SCAN_RINGS_PATCH_MM = 6.0;
// Nominal channel wavelengths, nm. Only their ratios are used.
constexpr HighPrecType ALGO_SCAN_LAMBDA_R       = 610.0;
constexpr HighPrecType ALGO_SCAN_LAMBDA_G       = 540.0;
constexpr HighPrecType ALGO_SCAN_LAMBDA_B       = 465.0;


inline void AlgoScannerArtifacts
(
    const HighPrecType  level,      // scannerArtifacts x damageStrength
    const uint32_t      seed,       // damageSeed ^ stage salt; NO frame index
    const HighPrecType  pxPerMm,
    AlgoType* RESTRICT  pR,
    AlgoType* RESTRICT  pG,
    AlgoType* RESTRICT  pB,
    const int32_t       sizeX,
    const int32_t       sizeY,
    const int32_t       pitch
) noexcept
{
    if ((level <= 0.0) || (sizeX <= 0) || (sizeY <= 0))
        return;

    const eALGO_RNG_STAGE st = eALGO_RNG_STAGE::eRNG_SCANNER;

    const HighPrecType cx = 0.5 * static_cast<HighPrecType>(sizeX);
    const HighPrecType cy = 0.5 * static_cast<HighPrecType>(sizeY);
    const HighPrecType invCorner2 = 1.0 / (cx * cx + cy * cy);

    const HighPrecType shade = ALGO_SCAN_SHADING_CORNER * level;
    const HighPrecType band  = ALGO_SCAN_BANDING_AMP    * level;
    const HighPrecType rings = ALGO_SCAN_RINGS_AMP      * level;

    // Ring centre: one draw per roll, anywhere in the frame.
    const HighPrecType ringX =
        AlgoRngUniform01(AlgoRngCounter(seed, 0, st, 1u)) * static_cast<HighPrecType>(sizeX);
    const HighPrecType ringY =
        AlgoRngUniform01(AlgoRngCounter(seed, 0, st, 2u)) * static_cast<HighPrecType>(sizeY);

    const HighPrecType mmPerPx = (pxPerMm > 0.0) ? (1.0 / pxPerMm) : 0.0;
    const HighPrecType patch2  = ALGO_SCAN_RINGS_PATCH_MM * ALGO_SCAN_RINGS_PATCH_MM;

    // r_k = sqrt(k) * r1 * sqrt(lambda / lambda_G): phase pi * (r / r1_c)^2.
    const HighPrecType pi = 3.14159265358979323846;
    const HighPrecType r1R = ALGO_SCAN_RINGS_R1_MM * std::sqrt(ALGO_SCAN_LAMBDA_R / ALGO_SCAN_LAMBDA_G);
    const HighPrecType r1G = ALGO_SCAN_RINGS_R1_MM;
    const HighPrecType r1B = ALGO_SCAN_RINGS_R1_MM * std::sqrt(ALGO_SCAN_LAMBDA_B / ALGO_SCAN_LAMBDA_G);
    const HighPrecType kR = pi / (r1R * r1R);
    const HighPrecType kG = pi / (r1G * r1G);
    const HighPrecType kB = pi / (r1B * r1B);

    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;
        AlgoType* RESTRICT rR = pR + off;
        AlgoType* RESTRICT rG = pG + off;
        AlgoType* RESTRICT rB = pB + off;

        const HighPrecType dy  = static_cast<HighPrecType>(y) + 0.5 - cy;
        const HighPrecType ry  = (static_cast<HighPrecType>(y) + 0.5 - ringY) * mmPerPx;

        for (int32_t x = 0; x < sizeX; x++)
        {
            const HighPrecType dx = static_cast<HighPrecType>(x) + 0.5 - cx;
            const HighPrecType s  = 1.0 - shade * (dx * dx + dy * dy) * invCorner2;

            // One gain per sensor column and colour line, ordinals 0x10000+.
            const uint32_t col = static_cast<uint32_t>(x) * 3u;
            const HighPrecType bR = 1.0 + band * (2.0 * AlgoRngUniform01(AlgoRngCounter(seed, 0, st, 0x10000u + col))      - 1.0);
            const HighPrecType bG = 1.0 + band * (2.0 * AlgoRngUniform01(AlgoRngCounter(seed, 0, st, 0x10000u + col + 1u)) - 1.0);
            const HighPrecType bB = 1.0 + band * (2.0 * AlgoRngUniform01(AlgoRngCounter(seed, 0, st, 0x10000u + col + 2u)) - 1.0);

            const HighPrecType rx = (static_cast<HighPrecType>(x) + 0.5 - ringX) * mmPerPx;
            const HighPrecType r2 = rx * rx + ry * ry;
            const HighPrecType env = rings * std::exp(-r2 / patch2);
            const HighPrecType nR = 1.0 + env * std::cos(kR * r2);
            const HighPrecType nG = 1.0 + env * std::cos(kG * r2);
            const HighPrecType nB = 1.0 + env * std::cos(kB * r2);

            rR[x] = static_cast<AlgoType>(static_cast<HighPrecType>(rR[x]) * s * bR * nR);
            rG[x] = static_cast<AlgoType>(static_cast<HighPrecType>(rG[x]) * s * bG * nG);
            rB[x] = static_cast<AlgoType>(static_cast<HighPrecType>(rB[x]) * s * bB * nB);
        }
    }

    return;
}
