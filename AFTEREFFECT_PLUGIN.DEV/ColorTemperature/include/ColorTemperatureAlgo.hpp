#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__ 
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__

#include "CommonAdobeAE.hpp"
#include "FastAriphmetics.hpp"
#include "CommonAuxPixFormat.hpp"
#include "CommonPixFormatSFINAE.hpp"
#include "AlgoProcStructures.hpp"
#include "CctLut.hpp"


template<typename T, typename U, typename std::enable_if<is_RGB_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline _tRGB<U> sRGB2LinearRGB (const T& in, const U coeff) noexcept
{
    /*
    if C_srgb <= 0.04045:
    C_linear = C_srgb / 12.92
    if C_srgb > 0.04045:
    C_linear = pow((C_srgb + 0.055) / 1.055, 2.4)
    */
    constexpr U threshold{ static_cast<U>(0.04045) };
    constexpr U reciproc1{ static_cast<U>(1.0) / static_cast<U>(12.920) };
    constexpr U reciproc2{ static_cast<U>(1.0) / static_cast<U>(1.0550) };
    _tRGB<U> out;

    const U R{ static_cast<U>(in.R) * coeff };
    const U G{ static_cast<U>(in.G) * coeff };
    const U B{ static_cast<U>(in.B) * coeff };

    out.R = (threshold <= R ? (R * reciproc1) : (FastCompute::Pow((R + static_cast<U>(0.0550)) * reciproc2, static_cast<U>(2.40))));
    out.G = (threshold <= G ? (G * reciproc1) : (FastCompute::Pow((G + static_cast<U>(0.0550)) * reciproc2, static_cast<U>(2.40))));
    out.B = (threshold <= B ? (B * reciproc1) : (FastCompute::Pow((B + static_cast<U>(0.0550)) * reciproc2, static_cast<U>(2.40))));

    return out;
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline _tRGB<T> sRGB2LinearRGB (const PF_Pixel_RGB_10u& in, const T coeff) noexcept
{
    /*
    if C_srgb <= 0.04045:
    C_linear = C_srgb / 12.92
    if C_srgb > 0.04045:
    C_linear = pow((C_srgb + 0.055) / 1.055, 2.4)
    */
    constexpr T threshold{ static_cast<T>(0.04045) };
    constexpr T reciproc1{ static_cast<T>(1.0) / static_cast<T>(12.920) };
    constexpr T reciproc2{ static_cast<T>(1.0) / static_cast<T>(1.0550) };
    _tRGB<T> out;

    const T R{ static_cast<T>(in.R) * coeff };
    const T G{ static_cast<T>(in.G) * coeff };
    const T B{ static_cast<T>(in.B) * coeff };

    out.R = (threshold <= R ? (R * reciproc1) : (FastCompute::Pow((R + static_cast<T>(0.0550)) * reciproc2, static_cast<T>(2.40))));
    out.G = (threshold <= G ? (G * reciproc1) : (FastCompute::Pow((G + static_cast<T>(0.0550)) * reciproc2, static_cast<T>(2.40))));
    out.B = (threshold <= B ? (B * reciproc1) : (FastCompute::Pow((B + static_cast<T>(0.0550)) * reciproc2, static_cast<T>(2.40))));

    return out;
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline _tXYZPix<T> sRgb2XYZ
(
    const _tRGB<T>& in
) noexcept
{
    auto varValue = [&](const T inVal) { return ((inVal > static_cast<T>(0.04045)) ? std::pow((inVal + static_cast<T>(0.055)) / static_cast<T>(1.055), static_cast<T>(2.40)) : (inVal / static_cast<T>(12.92))); };

    const T var_R = varValue(in.R) * static_cast<T>(100);
    const T var_G = varValue(in.G) * static_cast<T>(100);
    const T var_B = varValue(in.B) * static_cast<T>(100);

    _tXYZPix<T> out;
    out.X = var_R * static_cast<T>(0.4124564) + var_G * static_cast<T>(0.3575761) + var_B * static_cast<T>(0.1804375);
    out.Y = var_R * static_cast<T>(0.2126729) + var_G * static_cast<T>(0.7151522) + var_B * static_cast<T>(0.0721750);
    out.Z = var_R * static_cast<T>(0.0193339) + var_G * static_cast<T>(0.1191920) + var_B * static_cast<T>(0.9503041);

    return out;
}


template<typename T, typename U, typename std::enable_if<is_RGB_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline void Convert2PixComponents
(
    const T*  __restrict pSrc,
    PixComponentsStr<U>* __restrict pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const U& coeff
) noexcept
{
    for (A_long j = 0; j < sizeY; j++)
    {
        const T*  __restrict pSrcLine = pSrc + j * srcPitch;
        PixComponentsStr<U>* __restrict pDstLine = pDst + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
        {
            _tXYZPix<U> xyz = sRgb2XYZ(sRGB2LinearRGB(pSrcLine[i], coeff));
            
            const U XYZ_sum = xyz.X + xyz.Y + xyz.Z;
            const U x = xyz.X / XYZ_sum;
            const U y = xyz.Y / XYZ_sum;

            U u, v;
            xy_to_uv (x, y, u, v);
            
            pDst[i].Y = xyz.Y;
            pDst[i].u = u;
            pDst[i].v = v;
        } // for (A_long i = 0; i < sizeX; i++)

    } // for (A_long j = 0; j < sizeY; j++)

    return;
}


template<typename T, typename U, typename std::enable_if<is_YUV_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline void Convert2PixComponents
(
    const T*  __restrict pSrc,
    PixComponentsStr<U>* __restrict pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const U& coeff,
    bool isBT709 = true
) noexcept
{
#if 0
    for (A_long j = 0; j < sizeY; j++)
    {
        const T*  __restrict pSrcLine = pSrc + j * srcPitch;
        PixComponentsStr<U>* __restrict pDstLine = pDst + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
        {
            _tXYZPix<U> xyz = sRgb2XYZ(sRGB2LinearRGB(pSrcLine[i], coeff));

            const U XYZ_sum = xyz.X + xyz.Y + xyz.Z;
            const U x = xyz.X / XYZ_sum;
            const U y = xyz.Y / XYZ_sum;

            U u, v;
            xy_to_uv (x, y, u, v);

            pDst[i].Y = xyz.Y;
            pDst[i].u = u;
            pDst[i].v = v;
        } // for (A_long i = 0; i < sizeX; i++)
    
    } // for (A_long j = 0; j < sizeY; j++)
#endif
    return;
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void Convert2PixComponents
(
    const PF_Pixel_RGB_10u*  __restrict pSrc,
    PixComponentsStr<T>* __restrict pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const T& coeff
) noexcept
{
    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_RGB_10u* __restrict pSrcLine = pSrc + j * srcPitch;
        PixComponentsStr<T>* __restrict pDstLine = pDst + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
        {
            _tXYZPix<T> xyz = sRgb2XYZ(sRGB2LinearRGB(pSrcLine[i], coeff));

            const T XYZ_sum = xyz.X + xyz.Y + xyz.Z;
            const T x = xyz.X / XYZ_sum;
            const T y = xyz.Y / XYZ_sum;

            T u, v;
            xy_to_uv (x, y, u, v);

            pDst[i].Y = xyz.Y;
            pDst[i].u = u;
            pDst[i].v = v;
        } // for (A_long i = 0; i < sizeX; i++)

    } // for (A_long j = 0; j < sizeY; j++)

    return;
}





#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__