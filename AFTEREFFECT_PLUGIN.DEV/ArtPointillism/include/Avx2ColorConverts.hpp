#ifndef __AVX2_COLOR_CONVERSION_INTERFACES__
#define __AVX2_COLOR_CONVERSION_INTERFACES__

#include <immintrin.h>
#include <cstdint>
#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "FastAriphmetics.hpp"

// API's for convert between BGRA_8u and CIELab
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_BGRA_8u* RESTRICT pRGB,
    float*                  RESTRICT pL,
    float*                  RESTRICT pAB,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           srcPitch,
    const int32_t           labPitch
) noexcept;

void AVX2_ConvertCIELab_SemiPlanar_ToRgb
(
    const PF_Pixel_BGRA_8u* RESTRICT pSrc, // for take alpha channel only
    const float*            RESTRICT pL,   // source L (planar)
    const float*            RESTRICT pAB,  // source ab (interleaved)
    PF_Pixel_BGRA_8u*       RESTRICT pDst, // destination buffer
    int32_t           sizeX,               // size of image (width)
    int32_t           sizeY,               // number of lines (height)
    int32_t           srcPitch,            // pitch of pSrc in PIXELS
    int32_t           dstPitch             // pitch of pDst in PIXELS 
) noexcept;


// API's for convert between ARGB_8u and CIELab
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_ARGB_8u* RESTRICT pRGB,
    float*                  RESTRICT pL,
    float*                  RESTRICT pAB,
    int32_t           sizeX,
    int32_t           sizeY,
    int32_t           srcPitch,
    int32_t           labPitch
) noexcept;

void AVX2_ConvertCIELab_SemiPlanar_ToRgb
(
    const PF_Pixel_ARGB_8u* RESTRICT pSrc, // Original ARGB source (for Alpha)
    const float*            RESTRICT pL,   // source L (planar)
    const float*            RESTRICT pAB,  // source ab (interleaved)
    PF_Pixel_ARGB_8u*       RESTRICT pDst, // destination buffer
    int32_t           sizeX,
    int32_t           sizeY,
    int32_t           srcPitch, // Pitch in PIXELS
    int32_t           dstPitch  // Pitch in PIXELS 
) noexcept;


#endif // __AVX2_COLOR_CONVERSION_INTERFACES__