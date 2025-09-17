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
    const EGA_Palette& palette
)
{
    CACHE_ALIGN const EGA_PaletteF32 p = {{
        { static_cast<float>(palette[0].r ) / 255.f, static_cast<float>(palette[0].g ) / 255.f, static_cast<float>(palette[0].b ) / 255.f },
        { static_cast<float>(palette[1].r ) / 255.f, static_cast<float>(palette[1].g ) / 255.f, static_cast<float>(palette[1].b ) / 255.f },
        { static_cast<float>(palette[2].r ) / 255.f, static_cast<float>(palette[2].g ) / 255.f, static_cast<float>(palette[2].b ) / 255.f },
        { static_cast<float>(palette[3].r ) / 255.f, static_cast<float>(palette[3].g ) / 255.f, static_cast<float>(palette[3].b ) / 255.f },
        { static_cast<float>(palette[4].r ) / 255.f, static_cast<float>(palette[4].g ) / 255.f, static_cast<float>(palette[4].b ) / 255.f },
        { static_cast<float>(palette[5].r ) / 255.f, static_cast<float>(palette[5].g ) / 255.f, static_cast<float>(palette[5].b ) / 255.f },
        { static_cast<float>(palette[6].r ) / 255.f, static_cast<float>(palette[6].g ) / 255.f, static_cast<float>(palette[6].b ) / 255.f },
        { static_cast<float>(palette[7].r ) / 255.f, static_cast<float>(palette[7].g ) / 255.f, static_cast<float>(palette[7].b ) / 255.f },
        { static_cast<float>(palette[8].r ) / 255.f, static_cast<float>(palette[8].g ) / 255.f, static_cast<float>(palette[8].b ) / 255.f },
        { static_cast<float>(palette[9].r ) / 255.f, static_cast<float>(palette[9].g ) / 255.f, static_cast<float>(palette[9].b ) / 255.f },
        { static_cast<float>(palette[10].r) / 255.f, static_cast<float>(palette[10].g) / 255.f, static_cast<float>(palette[10].b) / 255.f },
        { static_cast<float>(palette[11].r) / 255.f, static_cast<float>(palette[11].g) / 255.f, static_cast<float>(palette[11].b) / 255.f },
        { static_cast<float>(palette[12].r) / 255.f, static_cast<float>(palette[12].g) / 255.f, static_cast<float>(palette[12].b) / 255.f },
        { static_cast<float>(palette[13].r) / 255.f, static_cast<float>(palette[13].g) / 255.f, static_cast<float>(palette[13].b) / 255.f },
        { static_cast<float>(palette[14].r) / 255.f, static_cast<float>(palette[14].g) / 255.f, static_cast<float>(palette[14].b) / 255.f },
        { static_cast<float>(palette[15].r) / 255.f, static_cast<float>(palette[15].g) / 255.f, static_cast<float>(palette[15].b) / 255.f }
    }};

    // Split original resolution on blocks and compute X an Y coordinates for every block
    const CoordinatesVector xCor = ComputeBloksCoordinates (sizeX, EGA_width);
    const CoordinatesVector yCor = ComputeBloksCoordinates (sizeY, EGA_height);

    // compute Super Pixel for every image block
    const SuperPixels superPixels = ComputeSuperpixels (input, xCor, yCor, sizeX);

    // Convert super Pixels to selected EGA palette pixels
    SuperPixels colorMap = ConvertToPalette (superPixels, p);

    // Restore Target Image (convert original image to EGA palette and simulate EGA resolution)
    RestoreTargetView (output, xCor, yCor, colorMap, sizeX);

#if defined(_DEBUG) && defined(_SAVE_TMP_RESULT_FOR_DEBUG)
    const bool bSaveResult = dbgFileSave("D://output_ega.raw", output, EGA_width, EGA_height);
#endif

    return;
}


void VGA16_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const VGA_Palette16& palette
)
{
    CACHE_ALIGN const VGA_Palette16F32 p = {{
        { static_cast<float>(palette[0].r ) / 255.f, static_cast<float>(palette[0].g ) / 255.f, static_cast<float>(palette[0].b ) / 255.f },
        { static_cast<float>(palette[1].r ) / 255.f, static_cast<float>(palette[1].g ) / 255.f, static_cast<float>(palette[1].b ) / 255.f },
        { static_cast<float>(palette[2].r ) / 255.f, static_cast<float>(palette[2].g ) / 255.f, static_cast<float>(palette[2].b ) / 255.f },
        { static_cast<float>(palette[3].r ) / 255.f, static_cast<float>(palette[3].g ) / 255.f, static_cast<float>(palette[3].b ) / 255.f },
        { static_cast<float>(palette[4].r ) / 255.f, static_cast<float>(palette[4].g ) / 255.f, static_cast<float>(palette[4].b ) / 255.f },
        { static_cast<float>(palette[5].r ) / 255.f, static_cast<float>(palette[5].g ) / 255.f, static_cast<float>(palette[5].b ) / 255.f },
        { static_cast<float>(palette[6].r ) / 255.f, static_cast<float>(palette[6].g ) / 255.f, static_cast<float>(palette[6].b ) / 255.f },
        { static_cast<float>(palette[7].r ) / 255.f, static_cast<float>(palette[7].g ) / 255.f, static_cast<float>(palette[7].b ) / 255.f },
        { static_cast<float>(palette[8].r ) / 255.f, static_cast<float>(palette[8].g ) / 255.f, static_cast<float>(palette[8].b ) / 255.f },
        { static_cast<float>(palette[9].r ) / 255.f, static_cast<float>(palette[9].g ) / 255.f, static_cast<float>(palette[9].b ) / 255.f },
        { static_cast<float>(palette[10].r) / 255.f, static_cast<float>(palette[10].g) / 255.f, static_cast<float>(palette[10].b) / 255.f },
        { static_cast<float>(palette[11].r) / 255.f, static_cast<float>(palette[11].g) / 255.f, static_cast<float>(palette[11].b) / 255.f },
        { static_cast<float>(palette[12].r) / 255.f, static_cast<float>(palette[12].g) / 255.f, static_cast<float>(palette[12].b) / 255.f },
        { static_cast<float>(palette[13].r) / 255.f, static_cast<float>(palette[13].g) / 255.f, static_cast<float>(palette[13].b) / 255.f },
        { static_cast<float>(palette[14].r) / 255.f, static_cast<float>(palette[14].g) / 255.f, static_cast<float>(palette[14].b) / 255.f },
        { static_cast<float>(palette[15].r) / 255.f, static_cast<float>(palette[15].g) / 255.f, static_cast<float>(palette[15].b) / 255.f }
    }};

    // Split original resolution on blocks and compute X an Y coordinates for every block
    const CoordinatesVector xCor = ComputeBloksCoordinates (sizeX, VGA16_width);
    const CoordinatesVector yCor = ComputeBloksCoordinates (sizeY, VGA16_height);

    // compute Super Pixel for every image block
    const SuperPixels superPixels = ComputeSuperpixels (input, xCor, yCor, sizeX);

    // Convert super Pixels to selected VGA-16 palette pixels
    SuperPixels colorMap = ConvertToPalette (superPixels, p);

    // Restore Target Image (convert original image to VGA-16 palette and simulate VGA resolution)
    RestoreTargetView (output, xCor, yCor, colorMap, sizeX);

#if defined(_DEBUG) && defined(_SAVE_TMP_RESULT_FOR_DEBUG)
    const bool bSaveResult = dbgFileSave("D://output_vga16.raw", output, VGA16_width, VGA16_height);
#endif

    return;
}


void VGA256_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const VGA_Palette256& palette
)
{
    CACHE_ALIGN VGA_Palette256F32 p;
    __VECTOR_ALIGNED__
    for (A_long i = 0; i < palette.size(); i++)
    {
        p[i].r = static_cast<float>(palette[i].r) / 255.f;
        p[i].g = static_cast<float>(palette[i].g) / 255.f;
        p[i].b = static_cast<float>(palette[i].b) / 255.f;
    }

    // Split original resolution on blocks and compute X an Y coordinates for every block
    const CoordinatesVector xCor = ComputeBloksCoordinates (sizeX, VGA256_width);
    const CoordinatesVector yCor = ComputeBloksCoordinates (sizeY, VGA256_height);

    // compute Super Pixel for every image block
    const SuperPixels superPixels = ComputeSuperpixels (input, xCor, yCor, sizeX);

    // Convert super Pixels to selected VGA-256 palette pixels
    SuperPixels colorMap = ConvertToPalette (superPixels, p);

    // Restore Target Image (convert original image to VGA-256 palette and simulate VGA resolution)
    RestoreTargetView (output, xCor, yCor, colorMap, sizeX);

#if defined(_DEBUG) && defined(_SAVE_TMP_RESULT_FOR_DEBUG)
    const bool bSaveResult = dbgFileSave("D://output_vga256.raw", output, VGA256_width, VGA256_height);
#endif

    return;
}


void Hercules_Simulation
(
    const fRGB* __restrict input,
    fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    float threshold
)
{
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
    const RetroMonitor& monitor = controlParams.monitor;

    switch (monitor)
    {
        case RetroMonitor::eRETRO_BITMAP_CGA:
        break;

        case RetroMonitor::eRETRO_BITMAP_EGA:
        break;

        case RetroMonitor::eRETRO_BITMAP_VGA:
        break;

        case RetroMonitor::eRETRO_BITMAP_HERCULES:
        default:
        break;
    }

    // Simulate CRT Artifacts


#if 0
    // check moitor type
    const RetroMonitor monitor = static_cast<RetroMonitor>(params[UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY)]->u.bd.value - 1);
    switch (monitor)
    {
        case RetroMonitor::eRETRO_BITMAP_CGA:
        {
            const PaletteCGA palette = static_cast<PaletteCGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE)]->u.bd.value - 1);
            const A_long intencity = params[UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT)]->u.bd.value;
            switch (palette)
            {
                case PaletteCGA::eRETRO_PALETTE_CGA1:
                {
                    const CGA_PaletteF32& paletteCGA1 = (0 == intencity ? CGA0_f32 : CGA0i_f32);
                    CGA_Simulation (input, output, sizeX, sizeY, paletteCGA1);
                }
                break;
                case PaletteCGA::eRETRO_PALETTE_CGA2:
                {
                    const CGA_PaletteF32& paletteCGA2 = (0 == intencity ? CGA1_f32 : CGA1i_f32);
                    CGA_Simulation (input, output, sizeX, sizeY, paletteCGA2);
                }
                break;
            }
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_EGA:
        {
            const PaletteEGA palette = static_cast<PaletteEGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE)]->u.bd.value - 1);
            switch (palette)
            {
                case PaletteEGA::eRETRO_PALETTE_EGA_STANDARD:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Standard_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_KING_QUESTS:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_KQ3_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_KYRANDIA:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Kyrandia_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_THEXDER:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Thexder_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_DUNE:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Dune_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_DOOM:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Doom_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_METAL_MUTANT:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_MetalMutant_u8);
                break;

                case PaletteEGA::eRETRO_PALETTE_EGA_WOLFENSTEIN:
                default:
                    EGA_Simulation (input, output, sizeX, sizeY, EGA_Wolfenstein_u8);
                break;
            }
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_VGA:
        {
            const PaletteVGA palette = static_cast<PaletteVGA>(params[UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE)]->u.bd.value - 1);
            if (PaletteVGA::eRETRO_PALETTE_VGA_16_BITS == palette)
                VGA16_Simulation  (input, output, sizeX, sizeY, VGA_Standard16_u8);
            else
                VGA256_Simulation (input, output, sizeX, sizeY, VGA_Standard256_u8);
        }
        break;

        case RetroMonitor::eRETRO_BITMAP_HERCULES:
        default:
        {
            const int32_t bwThreshold = params[UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD)]->u.sd.value;
            const float binThreshold = bwThreshold / 255.f;
            Hercules_Simulation (input, output, sizeX, sizeY, binThreshold);
        }
        break;
    }
#endif

    return;
}