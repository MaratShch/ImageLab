#pragma once

#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "ColorTransformMatrix.hpp"


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr,
          typename U, std::enable_if_t<is_YUV_proc<U>::value>* = nullptr>
void imgRGB2YUV
(
	const T* __restrict srcImage,
	      U* __restrict dstImage,
	eCOLOR_SPACE transformSpace,
	int32_t sizeX,
	int32_t sizeY,
	int32_t src_line_pitch,
	int32_t dst_line_pitch,
	int32_t subtractor = 0
) noexcept
{
	const float* __restrict colorMatrix = RGB2YUV[transformSpace];

	for (int32_t j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = srcImage + j * src_line_pitch;
		      U* __restrict pDstLine = dstImage + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (int32_t i = 0; i < sizeX; i++)
		{
			pDstLine[i].A = pSrcLine[i].A;
			pDstLine[i].Y = pSrcLine[i].R * colorMatrix[0] + pSrcLine[i].G * colorMatrix[1] + pSrcLine[i].B * colorMatrix[2];
			pDstLine[i].U = pSrcLine[i].R * colorMatrix[3] + pSrcLine[i].G * colorMatrix[4] + pSrcLine[i].B * colorMatrix[5] - subtractor;
			pDstLine[i].V = pSrcLine[i].R * colorMatrix[6] + pSrcLine[i].G * colorMatrix[7] + pSrcLine[i].B * colorMatrix[8] - subtractor;
		}
	}
	return;
}

template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr,
	      typename U, std::enable_if_t<is_RGB_proc<U>::value>* = nullptr>
inline void imgYUV2RGB
(
	const T* __restrict srcImage,
	      U* __restrict dstImage,
	eCOLOR_SPACE transformSpace,
	int32_t sizeX,
	int32_t sizeY,
	int32_t src_line_pitch,
	int32_t dst_line_pitch,
	int32_t addendum 
) noexcept
{
	const float* __restrict colorMatrix = YUV2RGB[transformSpace];

	for (int32_t j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = srcImage + j * src_line_pitch;
		      U* __restrict pDstLine = dstImage + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (int32_t i = 0; i < sizeX; i++)
		{
			pDstLine[i].A = pSrcLine[i].A;
			pDstLine[i].R = pSrcLine[i].Y * colorMatrix[0] + pSrcLine[i].U * colorMatrix[1] + pSrcLine[i].V * colorMatrix[2];
			pDstLine[i].G = pSrcLine[i].Y * colorMatrix[3] + pSrcLine[i].U * colorMatrix[4] + pSrcLine[i].V * colorMatrix[5] + addendum;
			pDstLine[i].B = pSrcLine[i].Y * colorMatrix[6] + pSrcLine[i].U * colorMatrix[7] + pSrcLine[i].V * colorMatrix[8] + addendum;
		}
	}
	return;
}

