#ifndef __IMAGELAB2_COMPUTE_SUPERPIXEL_HPP__
#define __IMAGELAB2_COMPUTE_SUPERPIXEL_HPP__

#include <cstdint>
#include <cmath>
#include <type_traits>
#include "super_pixel.hpp"


// Reduce a linear, interleaved RGB image to one super-pixel for CCT/Duv.
//
// Templated on:
//   TIn  - input buffer element type  (float for the canonical buffer; double allowed)
//   TOut - SuperPixel field type       (recommend double)
// Both restricted to floating-point. The per-pixel comparisons run in TIn;
// the ACCUMULATOR is fixed at double (not templated) - summing millions of
// values in float32 loses low-order bits, so double is mandatory here.
//
// Input : RGB_interleaved - linear, working-space RGB, reference white = 1.0,
//                           unclamped (HDR >1 and small negatives allowed).
//                           Layout [R0 G0 B0 R1 G1 B1 ...], 3 elems/pixel.
// Output: super            - weighted-mean linear RGB of included pixels;
//                            all-zero if none qualified (caller: low-confidence).
//
// Single-threaded, scalar. Exclusion rules per Rules_for_exclude_pixels:
// reject dark, near-clip, over-saturated (colored objects), zero-energy,
// negative, and non-finite pixels; soft-weight the highlight-taper and
// mid-saturation bands.
template
<
    typename TIn,
    typename TOut,
    typename std::enable_if<std::is_floating_point<TIn >::value>::type* = nullptr,
    typename std::enable_if<std::is_floating_point<TOut>::value>::type* = nullptr
>
void compute_superpixel
(
    const TIn*     RGB_interleaved, // linear RGB, reference white = 1.0
    int32_t        sizeX,           // horizontal image size in pixels
    int32_t        sizeY,           // vertical image size in pixels
    SuperPixel<TOut>& super         // out: computed super-pixel
)
{
    // ---- exclusion-rule constants (typed as TIn) ----
    constexpr TIn kYDark     = static_cast<TIn>(0.010);   // exclude below this luminance
    constexpr TIn kChClip    = static_cast<TIn>(0.95);    // exclude at/above (any channel)
    constexpr TIn kTaperLo   = static_cast<TIn>(0.90);    // highlight roll-off start (max ch)
    constexpr TIn kSatFull   = static_cast<TIn>(0.20);    // <= this: full weight
    constexpr TIn kSatMax    = static_cast<TIn>(0.60);    // >  this: exclude (colored object)
    constexpr TIn kEnergyMin = static_cast<TIn>(1.0e-6);  // exclude if (R+G+B) <= this

    // Luminance weights for the exposure gate - match the WORKING space.
    // Defaults: linear Rec.2020 (our working space).
    // Rec.709/sRGB: {0.2126, 0.7152, 0.0722}.
    constexpr TIn kLumaR = static_cast<TIn>(0.2627);
    constexpr TIn kLumaG = static_cast<TIn>(0.6780);
    constexpr TIn kLumaB = static_cast<TIn>(0.0593);

    // Spatial subsample stride (1 = every pixel); raise to 4..8 for speed.
    constexpr int32_t kStride = 1;

    double rSum = 0.0, gSum = 0.0, bSum = 0.0, wSum = 0.0;

    const int64_t nPix = static_cast<int64_t>(sizeX) * static_cast<int64_t>(sizeY);

    for (int64_t i = 0; i < nPix; i += kStride)
    {
        const TIn* px = RGB_interleaved + i * 3;
        const TIn R = px[0];
        const TIn G = px[1];
        const TIn B = px[2];

        // --- degenerate / invalid gate ---
        if (!(std::isfinite(R) && std::isfinite(G) && std::isfinite(B)))
            continue;                                       // NaN / Inf
        if (R < static_cast<TIn>(0) || G < static_cast<TIn>(0) || B < static_cast<TIn>(0))
            continue;                                       // out-of-gamut negative
        if ((R + G + B) <= kEnergyMin)
            continue;                                       // black / no energy

        // --- luminance / exposure gate ---
        const TIn maxc = (R > G) ? ((R > B) ? R : B) : ((G > B) ? G : B);
        if (maxc >= kChClip)
            continue;                                       // near clip
        const TIn Y = kLumaR * R + kLumaG * G + kLumaB * B;
        if (Y < kYDark)
            continue;                                       // shadow noise
        const TIn wLum = (maxc > kTaperLo)
                       ? (kChClip - maxc) / (kChClip - kTaperLo)
                       : static_cast<TIn>(1);

        // --- saturation / neutrality gate ---
        const TIn minc = (R < G) ? ((R < B) ? R : B) : ((G < B) ? G : B);
        const TIn sat  = (maxc - minc) / maxc;              // maxc > kEnergyMin here
        if (sat > kSatMax)
            continue;                                       // strongly colored object
        const TIn wSat = (sat > kSatFull)
                       ? (kSatMax - sat) / (kSatMax - kSatFull)
                       : static_cast<TIn>(1);

        // --- accumulate (double) ---
        const double w = static_cast<double>(wLum) * static_cast<double>(wSat);
        if (w <= 0.0)
            continue;
        rSum += w * static_cast<double>(R);
        gSum += w * static_cast<double>(G);
        bSum += w * static_cast<double>(B);
        wSum += w;
    }

    if (wSum > 0.0)
    {
        const double inv = 1.0 / wSum;
        super.r = static_cast<TOut>(rSum * inv);
        super.g = static_cast<TOut>(gSum * inv);
        super.b = static_cast<TOut>(bSum * inv);
    }
    else
    {
        super.r = super.g = super.b = static_cast<TOut>(0);  // no valid pixels
    }
}

#endif // __IMAGELAB2_COMPUTE_SUPERPIXEL_HPP__