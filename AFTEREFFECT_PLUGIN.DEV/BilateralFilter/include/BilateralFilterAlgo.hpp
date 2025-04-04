#ifndef __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_ALGO__
#define __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_ALGO__

#define FAST_COMPUTE_EXTRA_PRECISION

#include "Common.hpp"
#include "Param_Utils.h"
#include "CompileTimeUtils.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ColorTransformMatrix.hpp"
#include "ColorTransform.hpp"
#include "BilateralFilterEnum.hpp"
#include "GaussMesh.hpp"
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


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
void Rgb2CIELab
(
    const T*    __restrict pRGB,
    fCIELabPix* __restrict pLab,
    const A_long&          sizeX,
    const A_long&          sizeY,
    const A_long&          rgbPitch,
    const A_long&          labPitch
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


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
void BilateralFilterAlgorithm
(
    const fCIELabPix* __restrict pCieLab,
    const T* __restrict pSrc, // used only for get Alpha channel values
          T* __restrict pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    A_long fRadius,
    float fSigma,
    const T& blackPix, // black (minimal) color pixel value - used for clamping
    const T& whitePix  // white (maximal) color pixel value - used for clamping
) noexcept
{
    const float divider = 2.0f * fSigma * fSigma;
    const A_long labLinePitch{ sizeX };
    A_long iMin, iMax, jMin, jMax, m, n, s;

    A_long meshPitch = -1;
    const MeshT* __restrict pMesh = getMeshHandler()->getCenterMesh(meshPitch);

    if (nullptr != pMesh && meshPitch > 0)
    {
        A_long meshUp, meshDown;
        A_long meshLeft, meshRight;

        meshUp = 0;
        for (A_long j = 0; j < sizeY; j++)
        {
            const T* __restrict pSrcLine = pSrc + j * srcPitch;
                  T* __restrict pDstLine = pDst + j * dstPitch;

            // number of lines with coeficcients on top of filtered pixel
            meshUp   = FastCompute::Min(meshUp, fRadius);
            // number of lines with coefficients to down of filtered pixel
            meshDown = FastCompute::Min((sizeY - 1) - j, fRadius);

            jMin = FastCompute::Max(0, j - fRadius);
            jMax = FastCompute::Min(j + fRadius, sizeY - 1);

            for (A_long i = 0, meshLeft = 0; i < sizeX; meshLeft++, i++)
            {
                iMin = FastCompute::Max(0, i - fRadius);
                iMax = FastCompute::Min(i + fRadius, sizeX - 1);

                // number of coefficients from left side of filtered pixel
                meshLeft = FastCompute::Min(meshLeft, fRadius);
                // number of coefficients from right side of filtered pixel
                meshRight = FastCompute::Min((sizeX - 1), fRadius);

                const MeshT* pMeshStartline = pMesh - (meshUp * meshPitch) - meshLeft;
                const fCIELabPix& pixLab = pCieLab[j * labLinePitch + i];
                float fNorm = 0.f;
                m = s = 0;

                // Compute Gaussian range weights and calculate bilateral filter responce
                float bSum1 = 0.f, bSum2 = 0.f, bSum3 = 0.f;
                __VECTORIZATION__
                for (A_long k = jMin; k <= jMax; k++, s++)
                {
                    const MeshT* __restrict pMeshLine = pMeshStartline + s * meshPitch;

                    for (A_long l = iMin, n = 0; l <= iMax; l++, n++)
                    {
                        const fCIELabPix& pixWindow = pCieLab[k * labLinePitch + l];
                        const float dL = pixWindow.L - pixLab.L;
                        const float da = pixWindow.a - pixLab.a;
                        const float db = pixWindow.b - pixLab.b;

                        const float dotComp = dL * dL + da * da + db * db;
                        const float pF = FastCompute::Exp(-dotComp / divider) * pMeshLine[n];
                        fNorm += pF;

                        bSum1 += (pF * pixWindow.L);
                        bSum2 += (pF * pixWindow.a);
                        bSum3 += (pF * pixWindow.b);

                        m++;
                    }// for (A_long l = iMin; l <= iMax; l++)
                }// for (A_long k = jMin; k <= jMax; k++)

                fCIELabPix filteredPix;
                filteredPix.L = bSum1 / fNorm;
                filteredPix.a = bSum2 / fNorm;
                filteredPix.b = bSum3 / fNorm;

                const fRGB outPix = Xyz2Rgb(CieLab2Xyz(filteredPix));

                pDstLine[i].A = pSrcLine[i].A; // copy Alpha-channel from sources buffer 'as-is'
                pDstLine[i].R = static_cast<decltype(pDstLine[i].R)>(CLAMP_VALUE(outPix.R * whitePix.R, static_cast<float>(blackPix.R), static_cast<float>(whitePix.R)));
                pDstLine[i].G = static_cast<decltype(pDstLine[i].G)>(CLAMP_VALUE(outPix.G * whitePix.G, static_cast<float>(blackPix.G), static_cast<float>(whitePix.G)));
                pDstLine[i].B = static_cast<decltype(pDstLine[i].B)>(CLAMP_VALUE(outPix.B * whitePix.B, static_cast<float>(blackPix.B), static_cast<float>(whitePix.B)));

            }// for (A_long i = 0; i < sizeX; i++)

            meshUp++;
        }// for (A_long j = 0; j < sizeY; j++)

    }// if (nullptr != pMesh && meshPitch > 0)

    return;
}


#endif // __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_ALGO__