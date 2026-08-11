#pragma once

// ---------------------------------------------------------------------------
//  AlgoVeilingFlare.hpp
//
//  Sub-stage 3b of the film simulation pipeline: veiling flare from the taking
//  lens.
//
//  PHYSICAL BACKGROUND
//
//  Light entering a lens does not all reach the film by the intended path. Some
//  reflects off air-glass interfaces, off the inside of the barrel and off the
//  aperture blades, and arrives spread broadly across the frame as a haze.
//  Uncoated pre-war glass scattered something like 6 to 14 per cent of the
//  incident light this way; anti-reflection coating brought it below 1 per cent.
//
//  This is a LENS property, not an emulsion one. It is carried on the film
//  profile all the same, because era of glass and era of stock go together: a
//  1930s emulsion was almost never exposed through modern coated optics.
//
//  WHY IT MATTERS MORE THAN IT LOOKS
//
//  Flare lifts the black floor and compresses contrast across the whole frame.
//  Nothing else in the pipeline does that. Grain, curve shape and MTF all leave
//  the deepest black exactly where it was, so a period emulsion rendered without
//  flare still has modern blacks - which is the single most common reason a
//  vintage profile looks wrong despite every other parameter being right.
//
//  ENERGY IS PRESERVED, NOT ADDED
//
//  The direct image is scaled DOWN by the flare fraction and the scattered
//  component added back in its place. Light that scatters is light that did not
//  reach its intended position, so simply adding a haze on top would create
//  energy and wash the image out twice over.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType. The single place numeric types are chosen.
#include "AlgoTypes.hpp"

// Arena views and the AlgoType arithmetic type.
#include "AlgoMemHandler.hpp"

// Filtering primitives used for the broad scatter lobe.
#include "AlgoSeparableBlur.hpp"

// The control structure carrying the flare override.
#include "AlgoControl.hpp"

// FilmProfile, for default_flare.
#include "film_profiles.hpp"

#include <cstdint>   // int32_t


// ---------------------------------------------------------------------------
//  LUMINANCE WEIGHTS
//
//  0.30 red, 0.59 green, 0.11 blue.
//
//  The scatter is computed from a single luminance plane rather than three times
//  over, because glass scatters broadly and almost achromatically at these
//  scales - the wavelength dependence of a stray reflection is far smaller than
//  the difference between the three records it would be applied to.
//
//  These are the historical NTSC luminance coefficients rather than the Rec.709
//  set (0.2126 / 0.7152 / 0.0722). They match the reference model, and the
//  choice is defensible on its own terms: the quantity wanted here is roughly
//  how much light is present, not a colorimetrically correct luminance, and the
//  result is then blurred over hundreds of pixels. The three weights sum to
//  exactly 1.00, so a neutral frame produces a veil equal to its own level.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_FLARE_LUMA_R = static_cast<AlgoType>(0.30);
constexpr AlgoType ALGO_FLARE_LUMA_G = static_cast<AlgoType>(0.59);
constexpr AlgoType ALGO_FLARE_LUMA_B = static_cast<AlgoType>(0.11);


// ---------------------------------------------------------------------------
//  SCATTER LOBE RADII, micrometres of standard deviation on the film
//
//  1500, 6000 and 20000 micrometres - that is 1.5 mm, 6 mm and 20 mm. On a 35 mm
//  frame 24.89 mm wide, the widest lobe is comparable to the whole frame, which
//  is the point: lens flare is not a halo around highlights but a wash across
//  the entire image.
//
//  Specified in micrometres on the FILM rather than in pixels, so that the same
//  numbers describe the same physical scatter at any rendering resolution. They
//  are converted to pixels through px_per_mm at the point of use.
//
//  Three lobes rather than one because a single Gaussian falls off far too
//  quickly to represent scatter: the near field would be right and the far field
//  absent. The weights are heaviest on the tightest lobe and lightest on the
//  broadest, giving a long low tail rather than a uniform grey.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_FLARE_SIGMA_UM_0 = static_cast<AlgoType>(1500.0);
constexpr AlgoType ALGO_FLARE_SIGMA_UM_1 = static_cast<AlgoType>(6000.0);
constexpr AlgoType ALGO_FLARE_SIGMA_UM_2 = static_cast<AlgoType>(20000.0);

// Relative weights of the three lobes. They sum to 1.00 as written, but the
// filtering primitive normalises them anyway so the values remain meaningful if
// one is ever changed in isolation.
constexpr AlgoType ALGO_FLARE_WEIGHT_0 = static_cast<AlgoType>(0.45);
constexpr AlgoType ALGO_FLARE_WEIGHT_1 = static_cast<AlgoType>(0.35);
constexpr AlgoType ALGO_FLARE_WEIGHT_2 = static_cast<AlgoType>(0.20);


// ---------------------------------------------------------------------------
//  UNIFORM VEIL FRACTION
//
//  0.5. The scattered light is taken as half a completely uniform veil at the
//  frame's mean level, and half the broad local lobe computed above:
//
//      scattered = 0.5 * frameMean + 0.5 * broadLobe
//
//  The split represents two physically different contributions. Multiple
//  reflections between elements bounce so many times that all positional
//  information is lost and the result is genuinely flat across the frame; single
//  scattering events retain a weak memory of where the light came from and give
//  the broad lobe. Half and half is the reference model's apportionment.
//
//  It also has a useful practical consequence: because half the term is exactly
//  the frame mean, the accuracy demanded of the broad lobe is halved, which is
//  what makes a low-resolution evaluation of it acceptable.
// ---------------------------------------------------------------------------
constexpr AlgoType ALGO_FLARE_VEIL_FRACTION = static_cast<AlgoType>(0.5);


// ---------------------------------------------------------------------------
//  AlgoStage03b_VeilingFlare
//
//  For every pixel and every channel:
//
//      luma       = 0.30*R + 0.59*G + 0.11*B
//      veil       = mean(luma) over the frame
//      broad      = multiGaussian(luma, three lobes)
//      scattered  = 0.5*veil + 0.5*broad
//      dst        = src*(1 - flare) + flare*scattered
//
//  The same scattered plane is added to all three channels, so the flare is
//  achromatic by construction.
//
//  flare fraction: taken from params.flare when that is non-negative, otherwise
//  from profile.default_flare. A negative control value means "use the stock's
//  own era-appropriate figure", which is how the profile database supplies a
//  sensible default without the caller having to know the era.
//
//  src       stage-3 output: exposure units, mid grey at 1.0.
//  dst       the stage-3b buffer, same units.
//  scratchL  one plane: the luminance image.
//  scratchA  one plane: the accumulated weighted lobe sum.
//  scratchB  one plane: per-lobe blur result.
//  scratchC  one plane: separable intermediate inside each blur.
//
//  FOUR DISTINCT PLANES ARE REQUIRED, counting the luminance. The multi-lobe blur
//  reads its source once per lobe, so if any scratch plane aliases the source the
//  first lobe overwrites it and every later lobe blurs the wrong data. The symptom
//  is subtle - a plausible but wrong flare amplitude - which is exactly why the
//  requirement is spelled out here rather than left to the caller to infer.
//  pxPerMm   pixels per millimetre of film, used to convert the lobe radii.
//
//  Copied rather than skipped when the flare fraction is zero: the
//  retained-buffer policy gives this stage its own destination and it must leave
//  a valid image there.
//
//  Parameters and buffers are NOT validated; the caller has already done so.
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
) noexcept;
