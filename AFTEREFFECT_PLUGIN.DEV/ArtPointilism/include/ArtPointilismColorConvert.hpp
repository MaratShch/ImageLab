#define FAST_COMPUTE_EXTRA_PRECISION

#include "Common.hpp"
#include "Param_Utils.h"
#include "CompileTimeUtils.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ColorTransformMatrix.hpp"
#include "ColorTransform.hpp"
#include "FastAriphmetics.hpp"


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
    auto varValue = [&](const float& in) { return ((in > 0.040450f) ? FastCompute::Pow((in + 0.0550f) / 1.0550f, 2.40f) : (in / 12.92f)); };

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

    auto varValue = [&](const float& in) { return ((in > 0.008856f) ? FastCompute::Cbrt(in) : (in * 7.787f + 16.f / 116.f)); };

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

    auto varValue = [&](const float& in) { return ((in > 0.0031308f) ? (1.055f * FastCompute::Pow(in, 1.0f / 2.40f) - 0.055f) : (in * 12.92f)); };

    fRGB out;
    out.R = CLAMP_VALUE(varValue(r1), f32_value_black, f32_value_white);
    out.G = CLAMP_VALUE(varValue(g1), f32_value_black, f32_value_white);
    out.B = CLAMP_VALUE(varValue(b1), f32_value_black, f32_value_white);

    return out;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void ConvertToCIELab
(
    const T*    __restrict pRGB,
    fCIELabPix* __restrict pLab,
    const A_long          sizeX,
    const A_long          sizeY,
    const A_long          rgbPitch,
    const A_long          labPitch
) noexcept
{
    float sRgbCoeff = 1.0f / static_cast<float>(u8_value_white);
    if (std::is_same<T, PF_Pixel_BGRA_16u>::value || std::is_same<T, PF_Pixel_ARGB_16u>::value)
        sRgbCoeff = 1.0f / static_cast<float>(u16_value_white);
    else if (std::is_same<T, PF_Pixel_BGRA_32f>::value || std::is_same<T, PF_Pixel_ARGB_32f>::value)
        sRgbCoeff = 1.0f;

    for (A_long j = 0; j < sizeY; j++)
    {
        const T*    __restrict pRgbLine = pRGB + j * rgbPitch;
        fCIELabPix* __restrict pLabLine = pLab + j * labPitch;

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            fRGB inPix;
            inPix.R = pRgbLine[i].R * sRgbCoeff;
            inPix.G = pRgbLine[i].G * sRgbCoeff;
            inPix.B = pRgbLine[i].B * sRgbCoeff;

            pLabLine[i] = Xyz2CieLab(Rgb2Xyz(inPix));
        }
    }

    return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void ConvertToCIELab
(
    const T*    __restrict pYUV,
    fCIELabPix* __restrict pLab,
    const A_long          sizeX,
    const A_long          sizeY,
    const A_long          rgbPitch,
    const A_long          labPitch
) noexcept
{
    float sYuvCoeff = 1.0f / static_cast<float>(u8_value_white);
    float fUVSub = 128.f;
    if (std::is_same<T, PF_Pixel_VUYA_16u>::value)
        sYuvCoeff = 1.0f / static_cast<float>(u16_value_white), fUVSub = 0.f;
    else if (std::is_same<T, PF_Pixel_VUYA_32f>::value)
        sYuvCoeff = 1.0f, fUVSub = 0.f;

    // color space transfer matrix
    constexpr float ctm[9] =
    {
        YUV2RGB[BT709][0], YUV2RGB[BT709][1], YUV2RGB[BT709][2],
        YUV2RGB[BT709][3], YUV2RGB[BT709][4], YUV2RGB[BT709][5],
        YUV2RGB[BT709][6], YUV2RGB[BT709][7], YUV2RGB[BT709][8]
    };

    for (A_long j = 0; j < sizeY; j++)
    {
        const T*    __restrict pYuvLine = pYUV + j * rgbPitch;
        fCIELabPix* __restrict pLabLine = pLab + j * labPitch;

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            fYUV inYuvPix;
            inYuvPix.Y =  static_cast<float>(pYuvLine[i].Y) * sYuvCoeff;
            inYuvPix.U = (static_cast<float>(pYuvLine[i].U) - fUVSub) * sYuvCoeff;
            inYuvPix.V = (static_cast<float>(pYuvLine[i].V) - fUVSub) * sYuvCoeff;

            fRGB inPix;
            inPix.R = inYuvPix.Y * ctm[0] + inYuvPix.U * ctm[1] + inYuvPix.V * ctm[2];
            inPix.G = inYuvPix.Y * ctm[3] + inYuvPix.U * ctm[4] + inYuvPix.V * ctm[5];
            inPix.B = inYuvPix.Y * ctm[6] + inYuvPix.U * ctm[7] + inYuvPix.V * ctm[8];

            pLabLine[i] = Xyz2CieLab(Rgb2Xyz(inPix));
        }
    }

    return;
}


inline void ConvertToCIELab
(
    const PF_Pixel_RGB_10u* __restrict pRGB,
    fCIELabPix* __restrict pLab,
    const A_long          sizeX,
    const A_long          sizeY,
    const A_long          rgbPitch,
    const A_long          labPitch
) noexcept
{
    constexpr float sRgbCoeff = 1.0f / static_cast<float>(u10_value_white);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_RGB_10u* __restrict pRgbLine = pRGB + j * rgbPitch;
        fCIELabPix* __restrict pLabLine = pLab + j * labPitch;

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            fRGB inPix;
            inPix.R = pRgbLine[i].R * sRgbCoeff;
            inPix.G = pRgbLine[i].G * sRgbCoeff;
            inPix.B = pRgbLine[i].B * sRgbCoeff;

            pLabLine[i] = Xyz2CieLab(Rgb2Xyz(inPix));
        }
    }

    return;
}