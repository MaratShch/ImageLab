#include "RetroVisionAlgorithm.hpp"
#include "RetroVisionEnum.hpp"
#include "PaletteEGA.hpp"


void EGA_Simulation
(
    const fRGB* __restrict input,
          fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const EGA_PaletteF32& p
)
{
    // Split original resolution on blocks and compute X an Y coordinates for every block
    const CoordinatesVector xCor = ComputeBloksCoordinates(sizeX, EGA_width);
    const CoordinatesVector yCor = ComputeBloksCoordinates(sizeY, EGA_height);

    // compute Super Pixel for every image block
    const SuperPixels superPixels = ComputeSuperpixels(input, xCor, yCor, sizeX);

    // Convert super Pixels to selected EGA palette pixels
    SuperPixels colorMap = ConvertToPalette (superPixels, p);

    // Restore Target Image (convert original image to CGA palette and simulate EGA resolution)
    RestoreTargetView (output, xCor, yCor, colorMap, sizeX);

#if defined(_DEBUG) && defined(_SAVE_TMP_RESULT_FOR_DEBUG)
    const bool bSaveResult = dbgFileSave("D://output_ega.raw", output, CGA_width, CGA_height);
#endif

    return;
}


