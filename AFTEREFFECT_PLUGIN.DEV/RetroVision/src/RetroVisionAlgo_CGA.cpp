#include "RetroVisionAlgorithm.hpp"
#include "RetroVisionEnum.hpp"
#include "PaletteCGA.hpp"


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
    const CoordinatesVector xCor = ComputeBloksCoordinates(sizeX, CGA_width);
    const CoordinatesVector yCor = ComputeBloksCoordinates(sizeY, CGA_height);

    // compute Super Pixel for every image block
    const SuperPixels superPixels = ComputeSuperpixels(input, xCor, yCor, sizeX);

    // Convert super Pixels to selected CGA palette pixels
    SuperPixels colorMap = ConvertToPalette(superPixels, p);

    // Restore Target Image (convert original image to CGA palette and simulate CGA resolution)
    RestoreTargetView (output, xCor, yCor, colorMap, sizeX);

#if defined(_DEBUG) && defined(_SAVE_TMP_RESULT_FOR_DEBUG)
    const bool bSaveResult = dbgFileSave("D://output_cga.raw", output, CGA_width, CGA_height);
#endif

    return;
}

