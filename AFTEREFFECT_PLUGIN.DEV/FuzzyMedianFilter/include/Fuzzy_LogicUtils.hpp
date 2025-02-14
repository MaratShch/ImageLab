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
    out.X = var_R * 0.4124564f + var_G * 0.3575761f + var_B * 0.1804375f;
    out.Y = var_R * 0.2126729f + var_G * 0.7151522f + var_B * 0.0721750f;
    out.Z = var_R * 0.0193339f + var_G * 0.1191920f + var_B * 0.9503041f;

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

    const float r1 = var_X *  3.2404562f + var_Y * -1.5371385f + var_Z * -0.4985314f;
    const float g1 = var_X * -0.9692660f + var_Y *  1.8760108f + var_Z *  0.0415560f;
    const float b1 = var_X *  0.0556434f + var_Y * -0.2040259f + var_Z *  1.0572252f;

    auto varValue = [&](const float in) { return ((in > 0.0031308f) ? (1.055f * FastCompute::Pow(in, 1.0f / 2.40f) - 0.055f) : (in * 12.92f)); };

    fRGB out;
    constexpr float white = 1.f - FLT_EPSILON;
    out.R = CLAMP_VALUE(varValue(r1), 0.f, white);
    out.G = CLAMP_VALUE(varValue(g1), 0.f, white);
    out.B = CLAMP_VALUE(varValue(b1), 0.f, white);

    return out;
}

inline fYUV fRgb2Yuv
(
    const fRGB& in,
    const eCOLOR_SPACE& cSpace
) noexcept
{
    fYUV out;
    const float* __restrict cstm = RGB2YUV[cSpace];
    out.Y = in.R * cstm[0] + in.G * cstm[1] + in.B * cstm[2];
    out.U = in.R * cstm[3] + in.G * cstm[4] + in.B * cstm[5];
    out.V = in.R * cstm[6] + in.G * cstm[7] + in.B * cstm[8];

    return out;
}


inline fRGB fYuv2Rgb
(
    const fYUV& in,
    const eCOLOR_SPACE& cSpace
) noexcept
{
    fRGB out;
    const float* __restrict cstm = YUV2RGB[cSpace];
    out.R = in.Y * cstm[0] + in.U * cstm[1] + in.V * cstm[2];
    out.G = in.Y * cstm[3] + in.U * cstm[4] + in.V * cstm[5];
    out.B = in.Y * cstm[6] + in.U * cstm[7] + in.V * cstm[8];

    return out;
}


#endif // __FUZZY_MEDIAN_LOGIC_UTILS__