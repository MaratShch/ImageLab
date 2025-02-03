#ifndef __FUZZY_MEDIAN_LOGIC_UTILS__
#define __FUZZY_MEDIAN_LOGIC_UTILS__

#include "CompileTimeUtils.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ColorTransformMatrix.hpp"
#include "ColorTransform.hpp"

template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline T ClampPixelValue
(
    const T& input,
    const T& black,
    const T& white
) noexcept
{
    T output;
    output.R = CLAMP_VALUE(input.R, black.R, white.R);
    output.G = CLAMP_VALUE(input.G, black.G, white.G);
    output.B = CLAMP_VALUE(input.B, black.B, white.B);
    output.A = input.A; // not touch Alpha Channel 
    return output;
}


inline fXYZPix Rgb2Xyz
(
    const fRGB& in
) noexcept
{
    auto varValue = [&](const float in) { return ((in > 0.040450f) ? FastCompute::Pow((in + 0.0550f) / 1.0550f, 2.40f) : (in / 12.92f)); };

    const float var_R = varValue(in.R) * 100.f;
    const float var_G = varValue(in.G) * 100.f;
    const float var_B = varValue(in.B) * 100.f;

    fXYZPix out;
    out.X = var_R * 0.4124f + var_G * 0.3576f + var_B * 0.1805f;
    out.Y = var_R * 0.2126f + var_G * 0.7152f + var_B * 0.0722f;
    out.Z = var_R * 0.0193f + var_G * 0.1192f + var_B * 0.9505f;

    return out;
}


inline fCIELabPix Xyz2CieLab
(
    const fXYZPix& in
) noexcept
{
    constexpr float fRef[3] = {
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][0],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][1],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][2],
    };

    auto varValue = [&](const float in) { return ((in > 0.008856f) ? FastCompute::Cbrt(in) : (in * 7.787f + 16.f / 116.f)); };

    const float var_X = varValue(in.X / fRef[0]);
    const float var_Y = varValue(in.Y / fRef[1]);
    const float var_Z = varValue(in.Z / fRef[2]);

    fCIELabPix out;
    out.L = CLAMP_VALUE(116.f * var_Y - 16.f, -100.f, 100.f);       // L
    out.a = CLAMP_VALUE(500.f * (var_X - var_Y), -128.f, 128.f);    // a
    out.b = CLAMP_VALUE(200.f * (var_Y - var_Z), -128.f, 128.f);    // b

    return out;
}


inline fXYZPix CieLab2Xyz
(
    const fCIELabPix& in
) noexcept
{
    constexpr float fRef[3] = {
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][0],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][1],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][2],
    };

    const float var_Y = (in.L + 16.f) / 116.f;
    const float var_X = in.a / 500.f + var_Y;
    const float var_Z = var_Y - in.b / 200.f;

    const float y1 = ((var_Y > 0.2068930f) ? (var_Y * var_Y * var_Y) : ((var_Y - 16.f / 116.f) / 7.787f));
    const float x1 = ((var_X > 0.2068930f) ? (var_X * var_X * var_X) : ((var_X - 16.f / 116.f) / 7.787f));
    const float z1 = ((var_Z > 0.2068930f) ? (var_Z * var_Z * var_Z) : ((var_Z - 16.f / 116.f) / 7.787f));

    fXYZPix out;
    out.X = x1 * fRef[0];
    out.Y = y1 * fRef[1];
    out.Z = z1 * fRef[2];

    return out;
}

inline fRGB Xyz2Rgb
(
    const fXYZPix& in
) noexcept
{
    const float var_X = in.X / 100.f;
    const float var_Y = in.Y / 100.f;
    const float var_Z = in.Z / 100.f;

    const float r1 = var_X *  3.2406f + var_Y * -1.5372f + var_Z * -0.4986f;
    const float g1 = var_X * -0.9689f + var_Y *  1.8758f + var_Z *  0.0415f;
    const float b1 = var_X *  0.0557f + var_Y * -0.2040f + var_Z *  1.0570f;

    auto varValue = [&](const float in) { return ((in > 0.0031308f) ? (1.055f * FastCompute::Pow(in, 1.0f / 2.40f) - 0.055f) : (in * 12.92f)); };

    fRGB out;
    constexpr float white = 1.f - FLT_EPSILON;
    out.R = CLAMP_VALUE(varValue(r1), 0.f, white);
    out.G = CLAMP_VALUE(varValue(g1), 0.f, white);
    out.B = CLAMP_VALUE(varValue(b1), 0.f, white);

    return out;
}

#endif // __FUZZY_MEDIAN_LOGIC_UTILS__