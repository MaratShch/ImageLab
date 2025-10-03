#ifndef __IMAGE_LAB_RETRO_VISION_FILTER_ALGORITHM__
#define __IMAGE_LAB_RETRO_VISION_FILTER_ALGORITHM__

#include <type_traits>
#include <vector>
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "RetroVisionPalette.hpp"
#include "FastAriphmetics.hpp"
#include "ColorTransformMatrix.hpp"
#include "RetroVisionControls.hpp"



using CoordinatesVector = std::vector<A_long>;
using SuperPixels = std::vector<fRGB>;


fRGB* RetroResolution_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
);

std::vector<uint32_t> ScanLines_Simulation
(
    fRGB* input,
    fRGB* output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
);

void PhosphorGlow_Simulation
(
    fRGB* input,
    fRGB* output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
);

void PhosphorGlow_SimulationHelper
(
    const fRGB* __restrict in,
          fRGB* __restrict out,
    const A_long sizeX,
    const A_long sizeY,
    const float strength,
    const float opacity
) noexcept;

void AppertureGrill_Simulation
(
    const fRGB** input,
          fRGB** output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
);


void Hercules_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    float threshold,
    PEntry<float> whiteColor
);

void CGA_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const CGA_PaletteF32& p
);

void EGA_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const EGA_PaletteF32& palette
);

void Vga_Simulation16
(
    const fRGB* __restrict input,
    fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const VGA_Palette16F32& palette /* 16 palette entrties */
);

void Vga_Simulation256
(
    const fRGB* __restrict input,
    fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const VGA_Palette256F32& palette /* 256 palette entrties */
);


template <typename T, std::enable_if_t<is_RETRO_PALETTE<T>::value>* = nullptr>
inline SuperPixels ConvertToPalette(const SuperPixels& superPixels, const T& palette)
{
    const A_long spSize = static_cast<A_long>(superPixels.size());  // size of elements in Super Pixels vector
    const A_long paletteSize = static_cast<A_long>(palette.size()); // size of element in palette

    SuperPixels colorMap(spSize); // output colormap

    // lambda expression to find closest value of target palette
    auto findClosestColorIndex = [&](const T& palette, const fRGB& rgb) -> A_long
    {
        A_long bestIndex = 0;
        float bestDist = std::numeric_limits<float>::max();
        const A_long paletteSize = static_cast<A_long>(palette.size());

        for (A_long i = 0; i < paletteSize; ++i)
        {
            const float dr = rgb.R - palette[i].r;
            const float dg = rgb.G - palette[i].g;
            const float db = rgb.B - palette[i].b;

            // Try to use perceptual color weighting
            const float dist = 0.3f * dr * dr + 0.59f * dg * dg + 0.11f * db * db;
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIndex = i;
            }
        }
        return bestIndex;
    };

    for (A_long idx = 0; idx < spSize; idx++)
    {
        const A_long paletteIdx = findClosestColorIndex(palette, superPixels[idx]);
        const fRGB outColor = {
            palette[paletteIdx].r,
            palette[paletteIdx].g,
            palette[paletteIdx].b
        };
        colorMap[idx] = outColor;
    }

    return colorMap;
}


inline void RestoreTargetView
(
    fRGB* __restrict output,
    const CoordinatesVector& X,
    const CoordinatesVector& Y,
    const SuperPixels& colorMap,
    const A_long linePitch
)
{
    const A_long yBlocks = static_cast<A_long>(Y.size()) - 1;
    const A_long xBlocks = static_cast<A_long>(X.size()) - 1;
    A_long colorMapIdx = 0;

    for (A_long j = 0; j < yBlocks; j++)
    {
        const A_long startLine = Y[j];
        const A_long stopLine = Y[j + 1];

        for (A_long i = 0; i < xBlocks; i++)
        {
            // Process one CGA block
            for (A_long yb = startLine; yb < stopLine; yb++)
                for (A_long xb = X[i]; xb < X[i + 1]; xb++)
                    output[yb * linePitch + xb] = colorMap[colorMapIdx];

            colorMapIdx++;
        }
    }
    return;
}



inline CoordinatesVector ComputeBloksCoordinates(const A_long& origSize, const A_long& targetSize)
{
    const A_long vectorSize = FastCompute::Min(origSize, targetSize);
    CoordinatesVector out(vectorSize + 1);

    if (origSize < targetSize)
    {
        for (A_long i = 0; i <= vectorSize; i++)
            out[i] = FastCompute::Min(i, origSize);
    }
    else
    {
        const A_long scaleFactor = origSize / targetSize;
        const A_long fraction = origSize % targetSize;
        A_long compensationPool = fraction, idx;

        for (A_long i = idx = 0; i <= vectorSize; i++)
        {
            out[i] = FastCompute::Min(idx, origSize);
            idx += scaleFactor;
            if (0 < compensationPool)
            {
                idx++, compensationPool--;
            }
        }
    }
    return out;
}


inline SuperPixels ComputeSuperpixels
(
    const fRGB* __restrict input,
    const CoordinatesVector& X,
    const CoordinatesVector& Y,
    const A_long linePitch
)
{
    // size of coordinates vectors
    const A_long sizeX = static_cast<A_long>(X.size());
    const A_long sizeY = static_cast<A_long>(Y.size());

    SuperPixels superPixel(sizeX * sizeY);
    A_long vecIdx = 0;

    for (auto& itY = Y.begin() + 1; itY != Y.end(); ++itY)
    {
        const A_long yPrev = *(itY - 1);
        const A_long yCurr = *itY;

        for (auto& itX = X.begin() + 1; itX != X.end(); ++itX)
        {
            const A_long xPrev = *(itX - 1);
            const A_long xCurr = *itX;

            A_long j, i, num = 0;
            fRGB superPix{};

            for (j = yPrev; j < yCurr; j++)
                for (i = xPrev; i < xCurr; i++)
                {
                    superPix.R += input[j * linePitch + i].R;
                    superPix.G += input[j * linePitch + i].G;
                    superPix.B += input[j * linePitch + i].B;
                    num++;
                }

            const float fNum = static_cast<float>(num);
            // normalize Superpixel value
            superPix.R /= fNum;
            superPix.G /= fNum;
            superPix.B /= fNum;

            superPixel[vecIdx] = superPix;
            vecIdx++;
        }
    }
    return superPixel;
}

template <typename T>
inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type euclidean_distance(const _tRGB<T>& color1, const _tRGB<T>& color2) noexcept
{
    const T rDiff = color1.R - color2.R;
    const T gDiff = color1.G - color2.G;
    const T bDiff = color1.B - color2.B;

    return FastCompute::Sqrt(rDiff * rDiff + gDiff * gDiff + bDiff * bDiff);
}


template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline constexpr bool FloatEqual(const T& val1, const T& val2) noexcept
{
    return (val1 >= (val2 - std::numeric_limits<T>::epsilon()) && val1 <= (val2 + std::numeric_limits<T>::epsilon()));
}


template<typename T, typename U, typename std::enable_if<is_RGB_Variants<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline _tRGB<U> GammaAdjust (const T& in, const U gamma, const U normalize) noexcept
{
    _tRGB<U> out;

    const U R = static_cast<U>(in.R) / normalize;
    const U G = static_cast<U>(in.G) / normalize;
    const U B = static_cast<U>(in.B) / normalize;

    out.R = CLAMP_VALUE(FastCompute::Pow(R, gamma), static_cast<U>(0), static_cast<U>(1));
    out.G = CLAMP_VALUE(FastCompute::Pow(G, gamma), static_cast<U>(0), static_cast<U>(1));
    out.B = CLAMP_VALUE(FastCompute::Pow(B, gamma), static_cast<U>(0), static_cast<U>(1));
    return out;
}


template<typename T, typename U, typename std::enable_if<is_YUV_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline _tRGB<U> GammaAdjust (const T& in, const U gamma, const U normalize) noexcept
{
    _tRGB<U> out;

    constexpr U yuv2rgb[9] =
    {
        YUV2RGB[BT709][0], YUV2RGB[BT709][1], YUV2RGB[BT709][2],
        YUV2RGB[BT709][3], YUV2RGB[BT709][4], YUV2RGB[BT709][5],
        YUV2RGB[BT709][6], YUV2RGB[BT709][7], YUV2RGB[BT709][8]
    };

    if (std::is_same<T, PF_Pixel_VUYA_8u>::value)
    {
        PF_Pixel_BGRA_8u bgraPix;
        constexpr U diff = static_cast<U>(128);

        const U compU = static_cast<U>(in.U) - diff;
        const U compV = static_cast<U>(in.V) - diff;
        const U compY = static_cast<U>(in.Y);

        bgraPix.R = static_cast<decltype(bgraPix.R)>(compY * yuv2rgb[0] + compU * yuv2rgb[1] + compV * yuv2rgb[2]);
        bgraPix.G = static_cast<decltype(bgraPix.G)>(compY * yuv2rgb[3] + compU * yuv2rgb[4] + compV * yuv2rgb[5]);
        bgraPix.B = static_cast<decltype(bgraPix.B)>(compY * yuv2rgb[6] + compU * yuv2rgb[7] + compV * yuv2rgb[8]);
        bgraPix.A = static_cast<decltype(bgraPix.A)>(0);
        out = GammaAdjust (bgraPix, gamma, normalize);
    }
    else
    {
        PF_Pixel_BGRA_32f bgraPix;

        bgraPix.R = in.Y * yuv2rgb[0] + in.U * yuv2rgb[1] + in.V * yuv2rgb[2];
        bgraPix.G = in.Y * yuv2rgb[3] + in.U * yuv2rgb[4] + in.V * yuv2rgb[5];
        bgraPix.B = in.Y * yuv2rgb[6] + in.U * yuv2rgb[7] + in.V * yuv2rgb[8];
        bgraPix.A = 0.f;
        out = GammaAdjust (bgraPix, gamma, normalize);
    }

    return out;
}


template<typename T, typename U, typename std::enable_if<is_SupportedImageBuffer<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline void AdjustGammaValueToProc
(
    const T*  __restrict pSrc,
    _tRGB<U>* __restrict pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const U& gamma,
    const U& normalize
)
{
    for (A_long j = 0; j < sizeY; j++)
    {
        const T*  __restrict pSrcLine = pSrc + j * srcPitch;
        _tRGB<U>* __restrict pDstLine = pDst + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
            pDstLine[i] = GammaAdjust (pSrcLine[i], gamma, normalize);
    }
    return;
}


template<typename T, typename U, typename std::enable_if<is_no_alpha_channel<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline T RestoreImage
(
    const        T  src,
    const _tRGB<U>  proc,
    const U& normalize
) noexcept
{
    T out;
    
    (void)src;

    const U R = proc.R * normalize;
    const U G = proc.G * normalize;
    const U B = proc.B * normalize;

    out.R = static_cast<decltype(out.R)>(CLAMP_VALUE(R, static_cast<U>(0), normalize));
    out.G = static_cast<decltype(out.R)>(CLAMP_VALUE(G, static_cast<U>(0), normalize));
    out.B = static_cast<decltype(out.R)>(CLAMP_VALUE(B, static_cast<U>(0), normalize));

    return out;
}


template<typename T, typename TT, typename std::enable_if<is_YUV_proc<T>::value && std::is_floating_point<TT>::value>::type* = nullptr>
inline T RestoreImage
(
    const        T  src,
    const _tRGB<TT> proc,
    const TT& normalize
) noexcept
{
    constexpr TT rgb2yuv[9] =
    {
        RGB2YUV[BT709][0], RGB2YUV[BT709][1], RGB2YUV[BT709][2],
        RGB2YUV[BT709][3], RGB2YUV[BT709][4], RGB2YUV[BT709][5],
        RGB2YUV[BT709][6], RGB2YUV[BT709][7], RGB2YUV[BT709][8]
    };

    const TT Y = normalize * (proc.R * rgb2yuv[0] + proc.G * rgb2yuv[1] + proc.B * rgb2yuv[2]);
    const TT U = normalize * (proc.R * rgb2yuv[3] + proc.G * rgb2yuv[4] + proc.B * rgb2yuv[5]);
    const TT V = normalize * (proc.R * rgb2yuv[6] + proc.G * rgb2yuv[7] + proc.B * rgb2yuv[8]);

    T out;

    if (std::is_same<T, PF_Pixel_VUYA_8u>::value)
    {
        out.Y = static_cast<decltype(out.Y)>(CLAMP_VALUE(Y, static_cast<TT>(0), normalize));
        out.U = static_cast<decltype(out.U)>(CLAMP_VALUE(U + static_cast<TT>(128), static_cast<TT>(0), normalize));
        out.V = static_cast<decltype(out.V)>(CLAMP_VALUE(V + static_cast<TT>(128), static_cast<TT>(0), normalize));
    }
    else
    {
        out.Y = static_cast<decltype(out.Y)>(CLAMP_VALUE(Y, static_cast<TT>(0), normalize));
        out.U = static_cast<decltype(out.U)>(CLAMP_VALUE(U, static_cast<TT>(0), normalize));
        out.V = static_cast<decltype(out.V)>(CLAMP_VALUE(V, static_cast<TT>(0), normalize));
    }

    out.A = src.A;

    return out;
}


template<typename T, typename U, typename std::enable_if<is_RGB_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline T RestoreImage
(
    const        T  src,
    const _tRGB<U>  proc,
    const U& normalize
) noexcept
{
    T out;

    const U R = proc.R * normalize;
    const U G = proc.G * normalize;
    const U B = proc.B * normalize;

    out.A = src.A;
    out.R = static_cast<decltype(out.R)>(CLAMP_VALUE(R, static_cast<U>(0), normalize));
    out.G = static_cast<decltype(out.R)>(CLAMP_VALUE(G, static_cast<U>(0), normalize));
    out.B = static_cast<decltype(out.R)>(CLAMP_VALUE(B, static_cast<U>(0), normalize));

    return out;
}

    
template<typename T, typename U, typename std::enable_if<is_SupportedImageBuffer<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
inline void RestoreImage
(
    const        T* __restrict pSrc,
    const _tRGB<U>* __restrict pProc,
                 T* __restrict pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const U& normalize
) noexcept
{
    for (A_long j = 0; j < sizeY; j++)
    {
        const       T*  __restrict pSrcLine  = pSrc  + j * srcPitch;
        const _tRGB<U>* __restrict pProcLine = pProc + j * sizeX;
                    T*  __restrict pDstLine  = pDst  + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
            pDstLine[i] = RestoreImage (pSrcLine[i], pProcLine[i], normalize);
    }
    return;
}


#endif // __IMAGE_LAB_RETRO_VISION_FILTER_ALGORITHM__