#include "RetroVisionAlgorithm.hpp"
#include "RetroVisionEnum.hpp"
#include "PaletteHercules.hpp"


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
    A_long j = 0, i = 0, idx;
    constexpr fRGB colorBlack{ 0.f, 0.f, 0.f };
    const fRGB colorWhite = 
    {
        whiteColor.r,
        whiteColor.g,
        whiteColor.b
    };

    // Split original resolution on blocks and compute X an Y coordinates for every block
    const CoordinatesVector xCor = ComputeBloksCoordinates(sizeX, Hercules_width);
    const CoordinatesVector yCor = ComputeBloksCoordinates(sizeY, Hercules_height);

    // Compute SuperPixel from every block
    const SuperPixels superPixels = ComputeSuperpixels (input, xCor, yCor, sizeX);

    const A_long yBlocks = static_cast<A_long>(yCor.size());
    const A_long xBlocks = static_cast<A_long>(xCor.size());

    // Simulate Hercules monitor
    for (idx = 0, j = 1; j < yBlocks; j++)
    {
        for (i = 1; i < xBlocks; i++)
        {
            // get SuperPixel value
            const fRGB superPix = superPixels[idx];
            // convert SuperPixel from RB to B/W value
            const float pixValue = 0.21260f * superPix.R + 0.71520f * superPix.G + 0.07220f * superPix.B;
            // check if the target pixel should be converted to Black or White color
            const fRGB bwPixel = ((pixValue > threshold) ? colorWhite : colorBlack);

            // get ROI coordinates (line and pixels idnesexes)
            const A_long lineStart  = yCor[j - 1];
            const A_long lineStop   = yCor[j];
            const A_long pixelStart = xCor[i - 1];
            const A_long pixelStop  = xCor[i];

            for (A_long k = lineStart; k < lineStop; k++)
                for (A_long l = pixelStart; l < pixelStop; l++)
                    output[k * sizeX + l] = bwPixel;
                
            // forward to next SuperPixel
            idx++;
        }
    }
    return;
}

