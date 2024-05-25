#ifndef __PERCEPTUAL_COLOR_CORRECTION_ALGO__
#define __PERCEPTUAL_COLOR_CORRECTION_ALGO__

#include "CommonAdobeAE.hpp"
#include "CommonAuxPixFormat.hpp"
#include "CommonPixFormatSFINAE.hpp"


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void QuickWhiteBalance
(
	const T* __restrict pSrcImage,
	fRGB*    __restrict pDstImage,
	A_long	   sizeX,
	A_long     sizeY,
	A_long     linePitch
) noexcept
{
	float sumR, sumG, sumB;
	A_long i, j;

	float maxPixValue = static_cast<float>(u8_value_white);
	if (std::is_same<T, PF_Pixel_BGRA_16u>::value || std::is_same<T, PF_Pixel_ARGB_16u>::value)
		maxPixValue = static_cast<float>(u16_value_white);
	else if (std::is_same<T, PF_Pixel_BGRA_32f>::value || std::is_same<T, PF_Pixel_ARGB_32f>::value)
		maxPixValue = f32_value_white;

	sumR = sumG = sumB = 0.f;
	const float reciprocMaxPixVal = 1.f / maxPixValue;
	const float reciprocImgSize   = 1.f / static_cast<float>(sizeX * sizeY);

	for (j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrcImage + j * linePitch;
		for (i = 0; i < sizeX; i++)
		{
			sumR += (pSrcLine[i].R * reciprocMaxPixVal);
			sumG += (pSrcLine[i].G * reciprocMaxPixVal);
			sumB += (pSrcLine[i].B * reciprocMaxPixVal);
		}
	}

	const float scaleFactorR = reciprocMaxPixVal * 0.5f / (sumR * reciprocImgSize);
	const float scaleFactorG = reciprocMaxPixVal * 0.5f / (sumG * reciprocImgSize);
	const float scaleFactorB = reciprocMaxPixVal * 0.5f / (sumB * reciprocImgSize);

	for (j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrcImage + j * linePitch;
		fRGB*    __restrict pDstLine = pDstImage + j * sizeX;
		for (i = 0; i < sizeX; i++)
		{
			pDstLine[i].R = (pSrcLine[i].R * scaleFactorR);
			pDstLine[i].G = (pSrcLine[i].G * scaleFactorG);
			pDstLine[i].B = (pSrcLine[i].B * scaleFactorB);
		}
	}

	return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
void QuickWhiteBalance
(
	const T* __restrict pSrcImage,
	fRGB*    __restrict pWbImage,
	A_long	sizeX,
	A_long  sizeY,
	A_long  linePitch
) noexcept
{

	return;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void dbgBufferShow
(
	const T*    __restrict pSrcOrigImage,
	const fRGB* __restrict pSrcImage,
	      T*    __restrict pDstImage,
	A_long	   sizeX,
	A_long     sizeY,
	A_long     srcLinePitch,
	A_long     dstLinePitch
) noexcept
{
	constexpr float zer0 = 0.f;

	float maxPixValue = static_cast<float>(u8_value_white);
	if (std::is_same<T, PF_Pixel_BGRA_16u>::value || std::is_same<T, PF_Pixel_ARGB_16u>::value)
		maxPixValue = static_cast<float>(u16_value_white);
	else if (std::is_same<T, PF_Pixel_BGRA_32f>::value || std::is_same<T, PF_Pixel_ARGB_32f>::value)
		maxPixValue = f32_value_white;

	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict     pOrigSrcLine  = pSrcOrigImage + j * srcLinePitch;
		const fRGB* __restrict  pBalancedLine = pSrcImage + j * sizeX;
		      T* __restrict     pDstLine = pDstImage + j * dstLinePitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i].A = pOrigSrcLine[i].A;
			pDstLine[i].R = CLAMP_VALUE(pBalancedLine[i].R * maxPixValue, zer0, maxPixValue);
			pDstLine[i].G = CLAMP_VALUE(pBalancedLine[i].G * maxPixValue, zer0, maxPixValue);
			pDstLine[i].B = CLAMP_VALUE(pBalancedLine[i].B * maxPixValue, zer0, maxPixValue);
		}
	}

	return;
}


#endif /* __PERCEPTUAL_COLOR_CORRECTION_ALGO__ */
