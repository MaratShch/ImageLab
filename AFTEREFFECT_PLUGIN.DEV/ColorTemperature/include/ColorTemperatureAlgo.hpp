#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__ 
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__

#include "FastAriphmetics.hpp"
#include "CommonAuxPixFormat.hpp"


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



#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__