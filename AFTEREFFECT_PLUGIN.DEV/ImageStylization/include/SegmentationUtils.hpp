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
	A_long i, j;
	const float* __restrict yuv2rgb = YUV2RGB[BT709];
	const float uSub = (reciproc < 0.f ? 128.f : 0.f);
	const float vSub = (reciproc < 0.f ? 128.f : 0.f);
	const float mult = (reciproc < 0.f ? 1.f : 255.f);

	for (j = 0; j < sizeY; j++)
	{
		const A_long srcLineIdx = j * pitchSrc;
		const A_long tmpLineIdx = j * sizeX;

		__VECTOR_ALIGNED__
		for (i = 0; i < sizeX; i++)
		{
			pTmp[tmpLineIdx + i].R = (static_cast<float>(pSrc[srcLineIdx + i].Y)         * yuv2rgb[0] + 
				                     (static_cast<float>(pSrc[srcLineIdx + i].U) - uSub) * yuv2rgb[1] + 
				                     (static_cast<float>(pSrc[srcLineIdx + i].V) - vSub) * yuv2rgb[2]) * mult;

			pTmp[tmpLineIdx + i].G = (static_cast<float>(pSrc[srcLineIdx + i].Y)         * yuv2rgb[3] +
				                     (static_cast<float>(pSrc[srcLineIdx + i].U) - uSub) * yuv2rgb[4] +
				                     (static_cast<float>(pSrc[srcLineIdx + i].V) - vSub) * yuv2rgb[5]) * mult;

			pTmp[tmpLineIdx + i].B = (static_cast<float>(pSrc[srcLineIdx + i].Y)         * yuv2rgb[6] +
				                     (static_cast<float>(pSrc[srcLineIdx + i].U) - uSub) * yuv2rgb[7] +
				                     (static_cast<float>(pSrc[srcLineIdx + i].V) - vSub) * yuv2rgb[8]) * mult;
		}
	}

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

void assemble_segmented_image
(
	std::vector<Isegment>& Isegments,
	std::vector<Hsegment>& Hsegments,
	float* __restrict fR,
	float* __restrict fG,
	float* __restrict fB,
	A_long w,
	A_long h
) noexcept;

void store_segmented_image
(
	const PF_Pixel_BGRA_8u* __restrict srcBgra,
	PF_Pixel_BGRA_8u* __restrict dstBgra,
	const float* __restrict fR,
	const float* __restrict fG,
	const float* __restrict fB,
	A_long w,
	A_long h,
	A_long srcPitch,
	A_long dstPitch
) noexcept;

void store_segmented_image
(
	const PF_Pixel_ARGB_8u* __restrict srcBgra,
	PF_Pixel_ARGB_8u* __restrict dstBgra,
	const float* __restrict fR,
	const float* __restrict fG,
	const float* __restrict fB,
	A_long w,
	A_long h,
	A_long srcPitch,
	A_long dstPitch
) noexcept;

void store_segmented_image
(
	const PF_Pixel_ARGB_16u* __restrict srcBgra,
	PF_Pixel_ARGB_16u* __restrict dstBgra,
	const float* __restrict fR,
	const float* __restrict fG,
	const float* __restrict fB,
	A_long w,
	A_long h,
	A_long srcPitch,
	A_long dstPitch
) noexcept;

void store_segmented_image
(
	const PF_Pixel_BGRA_16u* __restrict srcBgra,
	PF_Pixel_BGRA_16u* __restrict dstBgra,
	const float* __restrict fR,
	const float* __restrict fG,
	const float* __restrict fB,
	A_long w,
	A_long h,
	A_long srcPitch,
	A_long dstPitch
) noexcept;

void store_segmented_image
(
	const PF_Pixel_BGRA_32f* __restrict srcBgra,
	PF_Pixel_BGRA_32f* __restrict dstBgra,
	const float* __restrict fR,
	const float* __restrict fG,
	const float* __restrict fB,
	A_long w,
	A_long h,
	A_long srcPitch,
	A_long dstPitch
) noexcept;

void store_segmented_image
(
	const PF_Pixel_VUYA_8u* __restrict srcBgra,
	PF_Pixel_VUYA_8u* __restrict dstBgra,
	const float* __restrict fR,
	const float* __restrict fG,
	const float* __restrict fB,
	A_long w,
	A_long h,
	A_long srcPitch,
	A_long dstPitch
) noexcept;

void store_segmented_image
(
	const PF_Pixel_VUYA_32f* __restrict srcBgra,
	PF_Pixel_VUYA_32f* __restrict dstBgra,
	const float* __restrict fR,
	const float* __restrict fG,
	const float* __restrict fB,
	A_long w,
	A_long h,
	A_long srcPitch,
	A_long dstPitch
) noexcept;