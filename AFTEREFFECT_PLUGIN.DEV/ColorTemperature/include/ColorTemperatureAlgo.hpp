#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__ 
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__

#include "CommonAdobeAE.hpp"
#include "FastAriphmetics.hpp"
#include "CommonAuxPixFormat.hpp"
#include "CommonPixFormatSFINAE.hpp"

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline _tRGB<T> sRGB2LinearRGB (const _tRGB<T>& in) noexcept
{
    /*
        if C_srgb <= 0.04045:
            C_linear = C_srgb / 12.92
        if C_srgb > 0.04045:
            C_linear = pow((C_srgb + 0.055) / 1.055, 2.4)
    */
    constexpr T threshold{ static_cast<T>(0.04045) };
    constexpr T reciproc1{ static_cast<T>(1) / static_cast<T>(12.920) };
    constexpr T reciproc2{ static_cast<T>(1) / static_cast<T>(1.0550) };
    _tRGB<T> out;

    out.R = (threshold <= in.R ? (in.R * reciproc1) : (FastCompute::Pow ((in.R + static_cast<T>(0.0550)) * reciproc2, static_cast<T>(2.40))));
    out.G = (threshold <= in.G ? (in.G * reciproc1) : (FastCompute::Pow ((in.G + static_cast<T>(0.0550)) * reciproc2, static_cast<T>(2.40))));
    out.B = (threshold <= in.B ? (in.B * reciproc1) : (FastCompute::Pow ((in.B + static_cast<T>(0.0550)) * reciproc2, static_cast<T>(2.40))));

    return out;
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline _tRGB<T> LinearRGB2sRGB (const _tRGB<T>& in) noexcept
{
    /*
        if C_linear <= 0.0031308:
            C_srgb = C_linear * 12.92
        if C_linear > 0.0031308:
            C_srgb = 1.055 * pow(C_linear, 1.0/2.4) - 0.055
    */

    constexpr threshold{ static_cast<T>(0.0031308) };
    constexpr reciproc{ static_cast<T>(1.0) / static_cast<T>(2.40) };
    _tRGB<T> out;

    out.R = (threshold <= in.R ? (in.R * static_cast<T>(12.92)) : (FastCompute::Pow(in.R, reciproc) * static_cast<T>(1.055) - static_cast<T>(0.055)));
    out.G = (threshold <= in.G ? (in.G * static_cast<T>(12.92)) : (FastCompute::Pow(in.G, reciproc) * static_cast<T>(1.055) - static_cast<T>(0.055)));
    out.B = (threshold <= in.B ? (in.B * static_cast<T>(12.92)) : (FastCompute::Pow(in.B, reciproc) * static_cast<T>(1.055) - static_cast<T>(0.055)));

    return out;
}


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

template<typename T, typename U, typename std::enable_if<is_RGB_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline void Convert2Linear_sRGB
(
    const T*  __restrict pSrc,
    _tRGB<U>* __restrict pDst,
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
        _tRGB<U>* __restrict pDstLine = pDst + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
            pDstLine[i] = sRGB2LinearRGB(pSrcLine[i], coeff);
    }
    return;
}


template<typename T, typename U, typename std::enable_if<is_YUV_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline void Convert2Linear_sRGB
(
    const T*  __restrict pSrc,
    _tRGB<U>* __restrict pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const U& coeff,
    bool isBT709 = true
) noexcept
{
    for (A_long j = 0; j < sizeY; j++)
    {
        const T*  __restrict pSrcLine = pSrc + j * srcPitch;
        _tRGB<U>* __restrict pDstLine = pDst + j * dstPitch;

//        for (A_long i = 0; i < sizeX; i++)
//            pDstLine[i] = sRGB2LinearRGB(pSrcLine[i], coeff);
    }
    return;
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void Convert2Linear_sRGB
(
    const PF_Pixel_RGB_10u*  __restrict pSrc,
    _tRGB<T>* __restrict pDst,
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
                      _tRGB<T>* __restrict pDstLine = pDst + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
            pDstLine[i] = sRGB2LinearRGB (pSrcLine[i], coeff);
    }
    return;
}

#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__