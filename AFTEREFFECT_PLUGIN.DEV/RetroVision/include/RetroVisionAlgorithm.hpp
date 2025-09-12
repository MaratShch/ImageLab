#ifndef __IMAGE_LAB_RETRO_VISION_FILTER_ALGORITHM__
#define __IMAGE_LAB_RETRO_VISION_FILTER_ALGORITHM__

#include <type_traits>
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "RetroVisionPalette.hpp"
#include "FastAriphmetics.hpp"
#include "ColorTransformMatrix.hpp"

template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline constexpr bool FloatEqual(const T& val1, const T& val2) noexcept
{
    return (val1 >= (val2 - std::numeric_limits<T>::epsilon()) && val1 <= (val2 + std::numeric_limits<T>::epsilon()));
}


template<typename T, typename U, typename std::enable_if<is_RGB_Variants<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline _tRGB<U> ToLinearRGB (const T& in, const U coeff) noexcept
{
//    constexpr U threshold{ static_cast<U>(0.04045) };
//    constexpr U reciproc1{ static_cast<U>(1.0) / static_cast<U>(12.920) };
//    constexpr U reciproc2{ static_cast<U>(1.0) / static_cast<U>(1.0550) };
    _tRGB<U> out;

    const U R{ static_cast<U>(in.R) * coeff };
    const U G{ static_cast<U>(in.G) * coeff };
    const U B{ static_cast<U>(in.B) * coeff };

    out.R = R;// (threshold <= R ? (R * reciproc1) : (FastCompute::Pow((R + static_cast<U>(0.0550)) * reciproc2, static_cast<U>(2.40))));
    out.G = G;// (threshold <= G ? (G * reciproc1) : (FastCompute::Pow((G + static_cast<U>(0.0550)) * reciproc2, static_cast<U>(2.40))));
    out.B = B;// (threshold <= B ? (B * reciproc1) : (FastCompute::Pow((B + static_cast<U>(0.0550)) * reciproc2, static_cast<U>(2.40))));

    return out;
}


template<typename T, typename U, typename std::enable_if<is_YUV_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline _tRGB<U> ToLinearRGB(const T& in, const U coeff) noexcept
{
    _tRGB<U> out{};

    const float* __restrict yuv2rgb = RGB2YUV[BT709];

    if (std::is_same<T, PF_Pixel_VUYA_8u>::value)
    {
        PF_Pixel_BGRA_8u bgraPix;
        constexpr U diff = static_cast<U>(128);

        const U compU = static_cast<U>(in.U) - diff;
        const U compV = static_cast<U>(in.V) - diff;
        const U compY = static_cast<U>(in.Y);

        bgraPix.R = compY * yuv2rgb[0] + compU * yuv2rgb[1] + compV * yuv2rgb[2];
        bgraPix.G = compY * yuv2rgb[3] + compU * yuv2rgb[4] + compV * yuv2rgb[5];
        bgraPix.B = compY * yuv2rgb[6] + compU * yuv2rgb[7] + compV * yuv2rgb[8];
        bgraPix.A = 0.f;
        out = ToLinearRGB (bgraPix, coeff);
    }
    else
    {
        PF_Pixel_BGRA_32f bgraPix;

        bgraPix.R = in.Y * yuv2rgb[0] + in.U * yuv2rgb[1] + in.V * yuv2rgb[2];
        bgraPix.G = in.Y * yuv2rgb[3] + in.U * yuv2rgb[4] + in.V * yuv2rgb[5];
        bgraPix.B = in.Y * yuv2rgb[6] + in.U * yuv2rgb[7] + in.V * yuv2rgb[8];
        bgraPix.A = 0.f;
        out = ToLinearRGB (bgraPix, coeff);
    }

    return out;
}


template<typename T, typename U, typename std::enable_if<is_SupportedImageBuffer<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline void Convert2sRGB
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

        __VECTOR_ALIGNED__
        for (A_long i = 0; i < sizeX; i++)
            pDstLine[i] = ToLinearRGB(pSrcLine[i], coeff);
    }

    return;
}


template <typename T>
inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type euclidean_distance(const _tRGB<T>& color1, const _tRGB<T>& color2) noexcept
{
    const T rDiff = color1.R - color2.R;
    const T gDiff = color1.G - color2.G;
    const T bDiff = color1.B - color2.B;

    return FastCompute::Sqrt(rDiff * rDiff + gDiff * gDiff + bDiff * bDiff);
}



void CGA_Simulation (const fRGB* __restrict input, fRGB* __restrict output, A_long sizeX, A_long sizeY, const CGA_Palette& palette);
void EGA_Simulation (const fRGB* __restrict input, fRGB* __restrict output, A_long sizeX, A_long sizeY, const EGA_Palette& palette);
void VGA16_Simulation (const fRGB* __restrict input, fRGB* __restrict output, A_long sizeX, A_long sizeY, const VGA_Palette16& palette);
void VGA256_Simulation(const fRGB* __restrict input, fRGB* __restrict output, A_long sizeX, A_long sizeY, const VGA_Palette256& palette);
void Hercules_Simulation(const fRGB* __restrict input, fRGB* __restrict output, A_long sizeX, A_long sizeY, float threshold);

void RetroResolution_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    PF_ParamDef* __restrict params[]
);

#endif // __IMAGE_LAB_RETRO_VISION_FILTER_ALGORITHM__