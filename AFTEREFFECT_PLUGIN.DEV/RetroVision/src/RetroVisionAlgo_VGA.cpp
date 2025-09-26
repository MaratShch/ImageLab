#include "RetroVisionAlgorithm.hpp"
#include "RetroVisionEnum.hpp"
#include "PaletteVGA.hpp"


void Vga_Simulation16
(
    const fRGB* __restrict input,
    fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const VGA_Palette16F32& p /* 16 palette entrties */
)
{
    // Split original resolution on blocks and compute X an Y coordinates for every block
    const CoordinatesVector xCor = ComputeBloksCoordinates(sizeX, VGA16_width);
    const CoordinatesVector yCor = ComputeBloksCoordinates(sizeY, VGA16_height);

    // compute Super Pixel for every image block
    const SuperPixels superPixels = ComputeSuperpixels(input, xCor, yCor, sizeX);

    // Convert super Pixels to selected EGA palette pixels
    SuperPixels colorMap = ConvertToPalette(superPixels, p);

    // Restore Target Image (convert original image to VGA palette and simulate VGA resolution)
    RestoreTargetView(output, xCor, yCor, colorMap, sizeX);

#if defined(_DEBUG) && defined(_SAVE_TMP_RESULT_FOR_DEBUG)
    const bool bSaveResult = dbgFileSave("D://output_vga16.raw", output, CGA_width, CGA_height);
#endif

    return;
}


void Vga_Simulation256
(
    const fRGB* __restrict input,
    fRGB* __restrict output,
    A_long sizeX,
    A_long sizeY,
    const VGA_Palette256F32& p /* 256 palette entrties */
)
{
    // Split original resolution on blocks and compute X an Y coordinates for every block
    const CoordinatesVector xCor = ComputeBloksCoordinates(sizeX, VGA256_width);
    const CoordinatesVector yCor = ComputeBloksCoordinates(sizeY, VGA256_height);

    // compute Super Pixel for every image block
    const SuperPixels superPixels = ComputeSuperpixels(input, xCor, yCor, sizeX);

    // Convert super Pixels to selected EGA palette pixels
    SuperPixels colorMap = ConvertToPalette (superPixels, p);

    // Restore Target Image (convert original image to VGA palette and simulate VGA resolution)
    RestoreTargetView(output, xCor, yCor, colorMap, sizeX);

#if defined(_DEBUG) && defined(_SAVE_TMP_RESULT_FOR_DEBUG)
    const bool bSaveResult = dbgFileSave("D://output_vga256.raw", output, CGA_width, CGA_height);
#endif

    return;
}

