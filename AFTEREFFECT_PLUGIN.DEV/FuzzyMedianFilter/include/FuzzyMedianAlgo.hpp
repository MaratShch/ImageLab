#ifndef __IMAGE_LAB_FUZZY_MEDIAN_FILTER_ALGORITHM__
#define __IMAGE_LAB_FUZZY_MEDIAN_FILTER_ALGORITHM__

#include "A.h"
#include "Common.hpp"
#include "Param_Utils.h"
#include "Fuzzy_LogicUtils.hpp"
#include "Fuzzy_Logic_3x3.hpp"
#include "Fuzzy_Logic_5x5.hpp"
#include "Fuzzy_Logic_7x7.hpp"


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void Rgb2CIELab
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



template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void Yuv2CIELab
(
    const T*    __restrict pYUV,
    fCIELabPix* __restrict pLab,
    const A_long&          sizeX,
    const A_long&          sizeY,
    const A_long&          rgbPitch,
    const A_long&          labPitch,
    const eCOLOR_SPACE&    colorSpace
) noexcept
{
    float sYuvCoeff = 1.0f / static_cast<float>(u8_value_white);
    float fUVSub = 128.f;
    if (std::is_same<T, PF_Pixel_VUYA_16u>::value)
        sYuvCoeff = 1.0f / static_cast<float>(u16_value_white), fUVSub = 0.f;
    else if (std::is_same<T, PF_Pixel_VUYA_32f>::value)
        sYuvCoeff = 1.0f, fUVSub = 0.f;

    // color space transfer matrix
    const float* __restrict cstm = YUV2RGB[colorSpace];

    for (A_long j = 0; j < sizeY; j++)
    {
        const T*    __restrict pYUVLine = pYUV + j * rgbPitch;
        fCIELabPix* __restrict pLabLine = pLab + j * labPitch;

        __VECTORIZATION__
        for (A_long i = 0; i < sizeX; i++)
        {
            fYUV inYuvPix;
            inYuvPix.Y =  static_cast<float>(pYUVLine[i].Y) * sYuvCoeff;
            inYuvPix.U = (static_cast<float>(pYUVLine[i].U) - fUVSub) * sYuvCoeff;
            inYuvPix.V = (static_cast<float>(pYUVLine[i].V) - fUVSub) * sYuvCoeff;

            fRGB inPix;
            inPix.R = inYuvPix.Y * cstm[0] + inYuvPix.U * cstm[1] + inYuvPix.V * cstm[2];
            inPix.G = inYuvPix.Y * cstm[3] + inYuvPix.U * cstm[4] + inYuvPix.V * cstm[5];
            inPix.B = inYuvPix.Y * cstm[6] + inYuvPix.U * cstm[7] + inYuvPix.V * cstm[8];

            pLabLine[i] = Xyz2CieLab(Rgb2Xyz(inPix));
        } // for (A_long i = 0; i < sizeX; i++)
    }

    return;
}


inline void Rgb2CIELab
(
    const PF_Pixel_RGB_10u* __restrict pRGB,
    fCIELabPix* __restrict pLab,
    const A_long&          sizeX,
    const A_long&          sizeY,
    const A_long&          rgbPitch,
    const A_long&          labPitch
) noexcept
{
    constexpr float sRgbCoeff = 1.0f / static_cast<float>(u10_value_white);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_RGB_10u*  __restrict pRgbLine = pRGB + j * rgbPitch;
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


#endif // __IMAGE_LAB_FUZZY_MEDIAN_FILTER_ALGORITHM__