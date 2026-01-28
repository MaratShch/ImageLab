#pragma once

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Image Flipping";
constexpr char strCopyright[] = "\n2019-2023. ImageLab2 Copyright(c).\rImage Flipping plugin.";
constexpr int EqualizationFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int EqualizationFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int EqualizationFilter_VersionSub = 0;
#ifdef _DEBUG
constexpr int EqualizationFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int EqualizationFilter_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int EqualizationFilter_VersionBuild = 1;

typedef enum {
	IMAGE_FLIP_FILTER_INPUT = 0,
	IMAGE_FLIP_HORIZONTAL_CHECKBOX,
	IMAGE_FLIP_VERTICAL_CHECKBOX,
	IMAGE_EQUALIZATION_FILTER_TOTAL_PARAMS
}Item;

constexpr char fH[] = "Flip horizontal";
constexpr char fV[] = "Flip vertical";


template <typename T>
inline void FlipHorizontal
(
	const T* __restrict srcImg,
	      T* __restrict dstImg,
	const A_long sizeY,
	const A_long sizeX,
	const A_long srcLinePitch,
	const A_long dstLinePitch
) noexcept
{
	A_long j, i;
	for (j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = srcImg + j * srcLinePitch + sizeX - 1;
		      T* __restrict pDstLine = dstImg + j * dstLinePitch;
		__VECTORIZATION__
		for (i = 0; i < sizeX; i++)
		    *pDstLine++ = *pSrcLine--;
	}
	return;
}


template <typename T>
inline void FlipVertical
(
	const T* __restrict srcImg,
	      T* __restrict dstImg,
	const A_long sizeY,
	const A_long sizeX,
	const A_long srcLinePitch,
	const A_long dstLinePitch
) noexcept
{
	A_long j, i;
	for (j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = srcImg + (sizeY - 1 - j) * srcLinePitch;
		      T* __restrict pDstLine = dstImg + j * dstLinePitch;
		__VECTORIZATION__
		for (i = 0; i < sizeX; i++)
			*pDstLine++ = *pSrcLine++;
	}
	return;
}


template <typename T>
inline void FlipHorizontalAndVertical
(
	const T* __restrict srcImg,
	      T* __restrict dstImg,
	const A_long sizeY,
	const A_long sizeX,
	const A_long srcLinePitch,
	const A_long dstLinePitch
) noexcept
{
	A_long j, i;
	for (j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = srcImg + (sizeY - 1 - j) * srcLinePitch + sizeX - 1;
		      T* __restrict pDstLine = dstImg + j * dstLinePitch;
		__VECTORIZATION__
		for (i = 0; i < sizeX; i++)
			*pDstLine++ = *pSrcLine--;
	}
	return;
}


template <typename T>
inline void ImageProcess
(
	const T* __restrict srcImg,
	      T* __restrict dstImg,
	const A_long sizeY,
	const A_long sizeX,
	const A_long srcLinePitch,
	const A_long dstLinePitch,
	const A_long flipType
) noexcept
{
	switch (flipType)
	{
		case 1:
			FlipHorizontal (srcImg, dstImg, sizeY, sizeX, srcLinePitch, dstLinePitch);
		break;

		case 2:
			FlipVertical (srcImg, dstImg, sizeY, sizeX, srcLinePitch, dstLinePitch);
		break;

		case 3:
			FlipHorizontalAndVertical (srcImg, dstImg, sizeY, sizeX, srcLinePitch, dstLinePitch);
		break;

		default:
			Image_SimpleCopy (srcImg, dstImg, sizeY, sizeX, srcLinePitch, dstLinePitch);
		break;
	}

	return;
}
