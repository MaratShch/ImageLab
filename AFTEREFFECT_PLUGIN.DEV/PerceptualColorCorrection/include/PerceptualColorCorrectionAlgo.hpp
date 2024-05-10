#ifndef __PERCEPTUAL_COLOR_CORRECTION_ALGO__
#define __PERCEPTUAL_COLOR_CORRECTION_ALGO__

#include "CommonAdobeAE.hpp"
#include "CommonAuxPixFormat.hpp"
#include "CommonPixFormatSFINAE.hpp"


template <typename U, typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void QuickWhiteBalance
(
	const T* __restrict pSrcImage,
	fRGB*    __restrict pDstImage,
	A_long	   sizeX,
	A_long     sizeY,
	A_long     linePitch,
	const U&   maxPixValue
) noexcept
{
	U sumR, sumG, sumB;
	A_long i, j;

	sumR = sumG = sumB = static_cast<U>(0.0);
	const U reciprocMaxPixVal = static_cast<U>(1.0) / maxPixValue;
	const U reciprocImgSize   = static_cast<U>(1.0) / static_cast<U>(sizeX * sizeY);

	for (j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrcImage + j * linePitch;
		for (i = 0; i < sizeX; i++)
		{
			sumR += (static_cast<U>(pSrcLine[i].R) * reciprocMaxPixVal);
			sumG += (static_cast<U>(pSrcLine[i].G) * reciprocMaxPixVal);
			sumB += (static_cast<U>(pSrcLine[i].B) * reciprocMaxPixVal);
		}
	}

	constexpr U half{ 0.5 };
	const U scaleFactorR = reciprocMaxPixVal * half / (sumR * reciprocImgSize);
	const U scaleFactorG = reciprocMaxPixVal * half / (sumG * reciprocImgSize);
	const U scaleFactorB = reciprocMaxPixVal * half / (sumB * reciprocImgSize);

	for (j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = pSrcImage + j * linePitch;
		fRGB*    __restrict pDstLine = pDstImage + j * sizeX;
		for (i = 0; i < sizeX; i++)
		{
			pDstLine[i].R = (static_cast<U>(pSrcLine[i].R) * scaleFactorR);
			pDstLine[i].G = (static_cast<U>(pSrcLine[i].G) * scaleFactorG);
			pDstLine[i].B = (static_cast<U>(pSrcLine[i].B) * scaleFactorB);
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
	A_long  linePitch,
	float   maxPixValue
) noexcept
{

	return;
}


template <typename U, typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void dbgBufferShow
(
	const T*    __restrict pSrcOrigImage,
	const fRGB* __restrict pSrcImage,
	      T*    __restrict pDstImage,
	A_long	   sizeX,
	A_long     sizeY,
	A_long     srcLinePitch,
	A_long     dstLinePitch,
	const U&   maxPixValue
) noexcept
{
	constexpr U zer0 = static_cast<U>(0);

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
