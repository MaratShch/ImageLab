#include <vector>
#include <cmath>
#include "CommonDebugUtils.hpp"
#include "RetroVisionPalette.hpp"
#include "RetroVisionAlgorithm.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionControls.hpp"

//#define _SAVE_TMP_RESULT_FOR_DEBUG

using CoordinatesVector = std::vector<A_long>;
using SuperPixels = std::vector<fRGB>;

inline CoordinatesVector ComputeBloksCoordinates (const A_long& origSize, const A_long& targetSize)
{
    const A_long vectorSize = FastCompute::Min(origSize, targetSize);
    CoordinatesVector out (vectorSize + 1);

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

        for (A_long i = 0; i < static_cast<int>(palette.size()); ++i)
        {
            float dr = rgb.R - palette[i].r;
            float dg = rgb.G - palette[i].g;
            float db = rgb.B - palette[i].b;

           float dist = FastCompute::Sqrt(dr * dr + dg * dg + db * db);
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
        const A_long paletteIdx = findClosestColorIndex (palette, superPixels[idx]);
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


void CGA_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const CGA_PaletteF32& p
)
{
    // Split original resolution on blocks and compute X an Y coordinates for every block
    const CoordinatesVector xCor = ComputeBloksCoordinates (sizeX, CGA_width);
    const CoordinatesVector yCor = ComputeBloksCoordinates (sizeY, CGA_height);

    // compute Super Pixel for every image block
    const SuperPixels superPixels = ComputeSuperpixels (input, xCor, yCor, sizeX);

    // Convert super Pixels to selected CGA palette pixels
    SuperPixels colorMap = ConvertToPalette (superPixels, p);

    // Restore Target Image (convert original image to CGA palette and simulate CGA resolution)
    RestoreTargetView (output, xCor, yCor, colorMap, sizeX);

#if defined(_DEBUG) && defined(_SAVE_TMP_RESULT_FOR_DEBUG)
    const bool bSaveResult = dbgFileSave("D://output_cga.raw", output, CGA_width, CGA_height);
#endif

    return;
}


void EGA_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const EGA_PaletteF32& palette
)
{
    return;
}



void Hercules_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    float threshold,
    PEntry<float> whiteColor
)
{
    return;
}


template <typename T, std::enable_if_t<is_VGA_RETRO_PALETTE<T>::value>* = nullptr>
void Vga_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const T& palette /* 16 or 256 palette entrties */
)
{
    const size_t paletteSize = palette.size();
    const int32_t targetSizeX = (16 == paletteSize ? VGA16_width  : VGA256_width );
    const int32_t targetSizeY = (16 == paletteSize ? VGA16_height : VGA256_height);

    return;
}



void RetroResolution_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const RVControls& controlParams
)
{
    // Simulate Retro-Monitor view
    switch (controlParams.monitor)
    {
        case RetroMonitor::eRETRO_BITMAP_CGA:
        {
            const CGA_PaletteF32& palette = (PaletteCGA::eRETRO_PALETTE_CGA1 == controlParams.cga_palette ?
                (0 == controlParams.cga_intencity_bit ? CGA0_f32 : CGA0i_f32) :
                    (0 == controlParams.cga_intencity_bit ? CGA1_f32 : CGA1i_f32));

            CGA_Simulation (input, output, sizeX, sizeY, palette);
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_EGA:
        {
            const EGA_PaletteF32& palette = getEgaPalette(controlParams.ega_palette);
            EGA_Simulation (input, output, sizeX, sizeY, palette);
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_VGA:
        {
            if (PaletteVGA::eRETRO_PALETTE_VGA_16_BITS == controlParams.vga_palette)
                Vga_Simulation (input, output, sizeX, sizeY, VGA_Standard16_f32);
            else
                Vga_Simulation (input, output, sizeX, sizeY, VGA_Standard256_f32);
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_HERCULES:
        default:
        {
            const PEntry<float> whiteLevel = HERCULES_White_ColorF32[UnderlyingType(controlParams.white_color_hercules)];
            const float threshold = static_cast<float>(controlParams.hercules_threshold) / 255.f;
            Hercules_Simulation (input, output, sizeX, sizeY, threshold, whiteLevel);
        }
        break;
    }

    // Simulate CRT Artifacts

    return;
}