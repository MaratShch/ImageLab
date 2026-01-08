#include <algorithm>
#include <cstdint>
#include "AefxDevPatch.hpp"
#include "Common.hpp"
#include "CommonAuxPixFormat.hpp"

void CIELab_LumaInvert (const fCIELabPix* RESTRICT pSrc, float* RESTRICT pLumaDst, A_long sizeX, A_long sizeY) noexcept;
void CIELab_LumaInvert (const float* RESTRICT pSrc, float* RESTRICT pLumaDst, A_long sizeX, A_long sizeY) noexcept;

void LumaEdgeDetection (const float* RESTRICT pSrc, float* RESTRICT pDst, A_long sizeX, A_long sizeY) noexcept;

void MixAndNormalizeDensity (const float* RESTRICT luma_src, const float* RESTRICT edge_src, float* RESTRICT target_dest, A_long sizeX, A_long sizeY, float sensitivity) noexcept;
void MixAndNormalizeDensity
(
    const float* RESTRICT luma_src, 
    const float* RESTRICT edge_src, 
    float* RESTRICT target_dest, 
    int pixel_count, 
    float sensitivity
) noexcept;