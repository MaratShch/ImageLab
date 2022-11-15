#pragma once

#include "SegmentationStructs.hpp"
#include "CommonPixFormat.hpp"
#include "ImageAuxPixFormat.hpp"


template <class T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void utils_prepare_data
(
	const T*  __restrict pSrc,
	fDataRGB* __restrict pTmp,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& pitchSrc,
	const float&  reciproc
) noexcept
{
	A_long i, j;

	if (reciproc > 0.f)
	{
		for (j = 0; j < sizeY; j++)
		{
			const A_long srcLineIdx = j * pitchSrc;
			const A_long tmpLineIdx = j * sizeX;
			
			__VECTOR_ALIGNED__
			for (i = 0; i < sizeX; i++)
			{
				pTmp[tmpLineIdx + i].R = static_cast<float>(pSrc[srcLineIdx + i].R) * reciproc;
				pTmp[tmpLineIdx + i].G = static_cast<float>(pSrc[srcLineIdx + i].G) * reciproc;
				pTmp[tmpLineIdx + i].B = static_cast<float>(pSrc[srcLineIdx + i].B) * reciproc;
			}
		}
	}
	else
	{
		for (j = 0; j < sizeY; j++)
		{
			const A_long srcLineIdx = j * pitchSrc;
			const A_long tmpLineIdx = j * sizeX;

			__VECTOR_ALIGNED__
			for (i = 0; i < sizeX; i++)
			{
				pTmp[tmpLineIdx + i].R = static_cast<float>(pSrc[srcLineIdx + i].R);
				pTmp[tmpLineIdx + i].G = static_cast<float>(pSrc[srcLineIdx + i].G);
				pTmp[tmpLineIdx + i].B = static_cast<float>(pSrc[srcLineIdx + i].B);
			}
		}
	}
	return;
}


template <class T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void utils_prepare_data
(
	const T*  __restrict pSrc,
	fDataRGB* __restrict pTmp,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& pitchSrc,
	const float&  reciproc
) noexcept
{
	return;
}


std::vector<int32_t> ftc_utils_segmentation
(
	int32_t* inHist, 
	int32_t inHistSize, 
	float epsilon, 
	bool circularHist
) noexcept;


std::vector<Hsegment> compute_color_palette
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const fDataRGB*         __restrict bgra,
	const A_long width,
	const A_long height,
	float Smin,
	int32_t nbins,
	int32_t nbinsS,
	int32_t nbinsI,
	float qH,
	float qS,
	float qI,
	std::vector<int32_t> ftcseg,
	float eps
) noexcept;


CostData cost_merging
(
	const int32_t* __restrict hist,
	std::vector<CostData>& listCosts,
	std::vector<int>& separators,
	std::vector<int>& maxima,
	int i1,
	int i2,
	float logeps
) noexcept;

std::vector<Isegment> compute_gray_palette
(
	const PF_Pixel_HSI_32f* __restrict hsi,
	const fDataRGB*         __restrict imgSrc,
	const A_long sizeX,
	const A_long sizeY,
	float Smin,
	int32_t nbinsI,
	float qI,
	std::vector<int32_t> ftcsegI
) noexcept;

void get_segmented_image
(
	std::vector<Isegment>& Isegments,
	std::vector<Hsegment>& Hsegments,
	const PF_Pixel_BGRA_8u* __restrict srcBgra,
	fDataRGB* __restrict fRGB,
	PF_Pixel_BGRA_8u* __restrict dstBgra,
	A_long sizeX,
	A_long sizeY,
	A_long srcPitch,
	A_long dstPitch
) noexcept;

A_long convert2HSI
(
	const fDataRGB* __restrict   pRGB,
	PF_Pixel_HSI_32f* __restrict pHSI,
	int32_t* __restrict histH,
	int32_t* __restrict histI,
	const A_long& sizeX,
	const A_long& sizeY,
	const float& qH,
	const float& qI,
	const float& sMin
) noexcept;