#pragma once

// ---------------------------------------------------------------------------
//  AlgoStockColourBalance.hpp
//
//  Stage 3 of the film simulation pipeline: stock colour balance.
//
//  Applies the per-layer exposure imbalance produced when a film is shot under
//  light of a different colour temperature from the one it was manufactured for.
//
//  PHYSICAL BACKGROUND
//
//  A colour film's three emulsion layers have their sensitivities trimmed during
//  manufacture so that ONE particular illuminant renders neutral. A tungsten
//  stock is trimmed for roughly 3200 K, a daylight stock for roughly 5500 K.
//  Shoot a tungsten stock in daylight and every layer receives a different
//  multiple of the exposure it was trimmed to expect, because the two
//  illuminants have different spectral radiance at each layer's sensitivity
//  peak. Blue rises steeply, red falls, and the image goes strongly blue.
//
//  That cast is therefore not a colour-grading choice applied on top of the
//  simulation: it falls out of blackbody physics, and it is why this stage
//  computes Planck's law rather than interpolating a hand-made table of tints.
//
//  MONOCHROME
//
//  Skipped entirely for a black-and-white stock. A single silver image has one
//  spectral sensitivity, so there is no inter-layer ratio to disturb: changing
//  the illuminant changes overall exposure, which the exposure control already
//  covers, not the balance between records.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

// Arena views and the AlgoType arithmetic type.
#include "AlgoMemHandler.hpp"

// AlgoCopyImage and the filtering primitives, all raw-pointer based.
#include "AlgoSeparableBlur.hpp"

// The control structure carrying sceneKelvin and wbStrength.
#include "AlgoControl.hpp"

// FilmProfile, for balance_kelvin and is_monochrome.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  LAYER SENSITIVITY PEAKS, nanometres
//
//  600 nm red, 550 nm green, 450 nm blue.
//
//  These are the approximate wavelengths at which each emulsion layer is most
//  sensitive, and they are where the two blackbody spectra are sampled. They are
//  intentionally a single representative wavelength per layer rather than an
//  integral over the layer's full spectral sensitivity curve: the balance shift
//  is a smooth, slowly varying function of colour temperature, and sampling at
//  the peak captures it to well within the accuracy of the colour temperature
//  the user supplies in the first place.
//
//  The values are close to the sensitisation maxima of real tripack emulsions,
//  where the cyan-forming layer peaks in the 640-660 nm region, the
//  magenta-forming layer near 550 nm and the yellow-forming layer near 440 nm.
//  600 nm is used for red rather than 650 nm because it sits nearer the centre
//  of the layer's response once the yellow filter layer and the overlying
//  emulsions have absorbed the short-wavelength end.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_LAYER_PEAK_NM_R = 600.0;
constexpr HighPrecType ALGO_LAYER_PEAK_NM_G = 550.0;
constexpr HighPrecType ALGO_LAYER_PEAK_NM_B = 450.0;


// ---------------------------------------------------------------------------
//  PLANCK RADIATION CONSTANTS
//
//  c1 = 3.741771e-16 W m^2
//       The first radiation constant, 2*pi*h*c^2, where h is Planck's constant
//       and c the speed of light in vacuum.
//
//  c2 = 1.438777e-2 m K
//       The second radiation constant, h*c/k, where k is Boltzmann's constant.
//
//  Both are in SI base units, so the wavelength passed to the evaluation must be
//  in METRES. The layer peaks above are in nanometres and are converted by the
//  1e-9 factor below.
//
//  Only the RATIO of two radiances at the same wavelength is ever used, and that
//  ratio is then normalised so green is exactly 1.0. Both constants therefore
//  cancel almost completely and the absolute scale is irrelevant - which is why
//  the reference model describes the result as being in arbitrary units. They
//  are written out in full anyway so the expression is recognisably Planck's law
//  and can be checked against a textbook rather than taken on trust.
// ---------------------------------------------------------------------------
constexpr HighPrecType ALGO_PLANCK_C1 = 3.741771e-16;
constexpr HighPrecType ALGO_PLANCK_C2 = 1.438777e-2;

// Nanometres to metres.
constexpr HighPrecType ALGO_NM_TO_M = 1.0e-9;


// ---------------------------------------------------------------------------
//  AlgoBalanceGains
//
//  Per-channel exposure gains for a colour temperature mismatch.
//
//  For each layer, the gain is the ratio of blackbody spectral radiance at that
//  layer's peak wavelength under the scene illuminant to the radiance under the
//  illuminant the stock was balanced for:
//
//      gain_c = L(lambda_c, sceneKelvin) / L(lambda_c, stockKelvin)
//
//  The three gains are then divided by the green gain, so green is exactly 1.0
//  and the operation changes only the balance BETWEEN records, never the overall
//  brightness. Without that normalisation the stage would double as an exposure
//  control and would fight the anchor solve.
//
//  Returned by output parameter rather than by a small struct so the function
//  stays usable from a context where a return-value copy would be unhelpful, and
//  so there is no temptation to add a constructor to a type used in hot code.
//
//  gains[0] = red, gains[1] = green (always 1.0), gains[2] = blue.
// ---------------------------------------------------------------------------
//  NOTE ON PRECISION: the two temperatures are HighPrecType, not AlgoType, and
//  the evaluation inside is likewise fixed at high precision. This is deliberate.
//  The fifth power of a wavelength in metres is of the order 1e-32 and the
//  exponential's argument reaches about 53, so the intermediates span roughly
//  sixty decades. Following AlgoType down to float here would not merely lose
//  accuracy, it would underflow and return wrong gains. It runs six times per
//  frame, so the cost of holding it at double is unmeasurable.
void AlgoBalanceGains
(
    const HighPrecType sceneKelvin,
    const HighPrecType stockKelvin,
    AlgoType           gains[3]
) noexcept;


// ---------------------------------------------------------------------------
//  AlgoStage03_StockColourBalance
//
//  For every pixel and every channel c:
//
//      g_c = 1 + (gain_c - 1) * wbStrength
//      dst = src * g_c
//
//  The interpolation on wbStrength is a user control, not physics: 0 disables the
//  effect completely and 1 applies the full physical mismatch. It is written as a
//  blend from unity rather than as a multiply on the gain so that the neutral
//  position is exactly 1.0 for every channel at any colour temperature, with no
//  residual tint from rounding.
//
//  src  stage-2b output: exposure units, mid grey at 1.0.
//  dst  the stage-3 buffer, same units.
//
//  Copied rather than skipped when the stage does not apply - a monochrome stock,
//  or wbStrength of zero. The retained-buffer policy gives this stage its own
//  destination, so it must leave a valid image there for any later inspection.
//
//  Parameters and buffers are NOT validated; the caller has already done so.
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
) noexcept;
