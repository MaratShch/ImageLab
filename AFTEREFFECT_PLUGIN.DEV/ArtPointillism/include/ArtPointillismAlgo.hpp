#ifndef __IMAGE_LAB_ART_POINTILISM_ALGORITHM_DEFINITIONS__
#define __IMAGE_LAB_ART_POINTILISM_ALGORITHM_DEFINITIONS__

#include "CommonAuxPixFormat.hpp"
#include "ArtPointillismColorConvert.hpp"
#include "ArtPointillismPainters.hpp"
#include "FastAriphmetics.hpp"


inline void CIELab_LumaInvert (const fCIELabPix* RESTRICT pSrc, float* RESTRICT pLumaDst, A_long sizeX, A_long sizeY) noexcept
{
    // Invert Luma: 1.0 (White) becomes 0.0 (No Dots)
    //         0.0 (Black) becomes 1.0 (Max Dots)

    const A_long lumaSize = sizeX * sizeY;
    for (A_long i = 0; i < lumaSize; i++)
        pLumaDst[i] = 1.f - pSrc[i].L / 100.f;
    return;
}


#endif // __IMAGE_LAB_ART_POINTILISM_ALGORITHM_DEFINITIONS__