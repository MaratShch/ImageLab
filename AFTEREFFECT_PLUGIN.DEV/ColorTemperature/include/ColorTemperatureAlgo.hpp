#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__ 
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__

#include "CommonAdobeAE.hpp"
#include "FastAriphmetics.hpp"
#include "CommonAuxPixFormat.hpp"
#include "CommonPixFormatSFINAE.hpp"
#include "ColorTransformMatrix.hpp"
#include "AlgoRules.hpp"
#include "AlgoProcStructures.hpp"
#include "cct_interface.hpp"
#include "CctLut.hpp"
#include <utility>
#include <array>


using AdaptationMatrixT = std::array<AlgoProcT, 9>;
AdaptationMatrixT computeAdaptationMatrix
(
    AlgoCCT::CctHandleF32* cctHandle,
    eCOLOR_OBSERVER observer,
    eCctType cctValueType,
    const std::pair<AlgoProcT, AlgoProcT>& cct_duv_src,
    const std::pair<AlgoProcT, AlgoProcT>& cct_duv_dst
);

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
    auto varValue = [&](const T& inVal) { return ((inVal > static_cast<T>(0.04045)) ? std::pow((inVal + static_cast<T>(0.055)) / static_cast<T>(1.055), static_cast<T>(2.40)) : (inVal / static_cast<T>(12.92))); };

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
inline void AdjustCct
(
    const T*  __restrict pSrc,
          T*  __restrict pDst,
    const AdaptationMatrixT& matrix,
    A_long sizeX, 
    A_long sizeY,
    A_long srcPitch, 
    A_long dstPitch, 
    U white
)
{
    A_long i, j;

    for (j = 0; j < sizeY; j++)
    {
        const T* __restrict pSrcLine = pSrc + j * srcPitch;
              T* __restrict pDstLine = pDst + j * dstPitch;

        for (i = 0; i < sizeX; i++)
        {
            pDstLine[i].A = pSrcLine[i].A;
            pDstLine[i].R = CLAMP_VALUE(pSrcLine[i].R * matrix[0] + pSrcLine[i].G * matrix[1] + pSrcLine[i].B * matrix[2], static_cast<U>(0), white);
            pDstLine[i].G = CLAMP_VALUE(pSrcLine[i].R * matrix[3] + pSrcLine[i].G * matrix[4] + pSrcLine[i].B * matrix[5], static_cast<U>(0), white);
            pDstLine[i].B = CLAMP_VALUE(pSrcLine[i].R * matrix[6] + pSrcLine[i].G * matrix[7] + pSrcLine[i].B * matrix[8], static_cast<U>(0), white);
        }
    }
    return;
}


template<typename T, typename U, typename std::enable_if<is_YUV_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline void AdjustCct
(
    const T*  __restrict pSrc,
          T*  __restrict pDst,
    const AdaptationMatrixT& matrix,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    U white,
    bool isBT709 = true
)
{
    return;
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void AdjustCct
(
    const PF_Pixel_RGB_10u*  __restrict pSrc,
          PF_Pixel_RGB_10u*  __restrict pDst,
    const AdaptationMatrixT& matrix,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    T white
)
{
    return;
}


template<typename T, typename U, typename std::enable_if<is_RGB_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline std::pair<U, U> Convert2PixComponents
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
    U accumYu, accumYv, accumY;
    accumYu = accumYv = accumY = { static_cast<U>(0) };

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

            accumY  += xyz.Y;
            accumYu += (xyz.Y * u);
            accumYv += (xyz.Y * v);
        } // for (A_long i = 0; i < sizeX; i++)

    } // for (A_long j = 0; j < sizeY; j++)

    // compute weighted chromaticity values
    const U weighted_u = accumYu / accumY;
    const U weighted_v = accumYv / accumY;

    return std::make_pair(weighted_u, weighted_v);
}


template<typename T, typename U, typename std::enable_if<is_YUV_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline std::pair<U, U> Convert2PixComponents
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
    U accumYu, accumYv, accumY;
    accumYu = accumYv = accumY = { static_cast<U>(0) };

    const float* __restrict ctm = (true == isBT709 ? YUV2RGB[1] : YUV2RGB[0]);

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
    const U weighted_u = 0;// accumYu / accumY;
    const U weighted_v = 0;// accumYv / accumY;

    return std::make_pair(weighted_u, weighted_v);
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline std::pair<T, T> Convert2PixComponents
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
    T accumYu, accumYv, accumY;
    accumYu = accumYv = accumY = { static_cast<T>(0) };

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
 
            accumY += xyz.Y;
            accumYu += (xyz.Y * u);
            accumYv += (xyz.Y * v);
       
        } // for (A_long i = 0; i < sizeX; i++)

    } // for (A_long j = 0; j < sizeY; j++)

      // compute weighted chromaticity values
    const T weighted_u = accumYu / accumY;
    const T weighted_v = accumYv / accumY;

    return std::make_pair(weighted_u, weighted_v);
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void uvToXy (const T u, const T v, T& x, T& y) noexcept
{
    constexpr T epsilon{ 1e-5 }; // Small threshold to avoid division instability
    const T denom = (static_cast<T>(2) * u - static_cast<T>(8) * v + static_cast<T>(4));

    if (std::abs(denom) < epsilon)
    {
        // Degenerate case: set to D65 or another safe default white point
        x = static_cast<T>(0.3127);
        y = static_cast<T>(0.3290);
    }
    else
    {
        x = (static_cast<T>(3) * u) / denom;
        y = (static_cast<T>(2) * v) / denom;
    }
    return;
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void uvToXYZ (T u, T v, T& X, T& Y, T& Z) noexcept
{
    constexpr T epsilon{ 1e-5 }; // Small threshold to avoid division instability
    const T denom = (static_cast<T>(2) * u - static_cast<T>(8) * v + static_cast<T>(4));

    if (std::abs(denom) < epsilon)
    {
        // Degenerate case: set to D65 safe default white point
        u = static_cast<T>(0.1978);
        v = static_cast<T>(0.4684);
    }

    const T x = (static_cast<T>(3) * u) / denom;
    const T y = (static_cast<T>(2) * v) / denom;
    const T z = static_cast<T>(1) - x - y;

    if (std::abs(y) < epsilon)
    {
        // Degenerate chromaticity: fallback to D65 XYZ
        X = static_cast<T>(0.95047);
        Y = static_cast<T>(1.00000);
        Z = static_cast<T>(1.08883);
    }
    else
    {
        X = x / y;
        Y = static_cast<T>(1);
        Z = z / y;
    }

    return;
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline void uvToXYZ (std::pair<T, T> uv, T& X, T& Y, T& Z) noexcept
{
    return uvToXYZ (uv.first, uv.second, X, Y, Z);
}


#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORTIHM_IMPLEMENTATIONS__