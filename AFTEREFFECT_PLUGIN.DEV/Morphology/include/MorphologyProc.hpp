#pragma once
#include <type_traits>
#include "CompileTimeUtils.hpp"
#include "CommonPixFormat.hpp"
#include "SE_Interface.hpp"


template <typename T, typename U>
typename std::enable_if<std::is_same<T, PF_Pixel_VUYA_8u>::value || std::is_same<T, PF_Pixel_VUYA_32f>::value, T>::type>
inline T ImgErodeOnEdge
(
	const T* __restrict pImg,
	const U& compareVal,
	const int32_t& imgStride,
	const SE_Type* __restrict pSe,
	const int32_t& seSize,
	const int32_t& numbLine,
	const int32_t& numbPix,
	const int32_t& sizeX,
	const int32_t& sizeY
) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	outPix.A = pImg[0].A;
	return outPix;
}


template <typename T, typename U>
typename std::enable_if<std::is_same<T, PF_Pixel_VUYA_8u>::value || std::is_same<T, PF_Pixel_VUYA_32f>::value, T>::type>
inline T ImgErode
(
	const T* __restrict pImg,
	const U& compareVal,
	const int32_t& imgStride,
	const SE_Type* __restrict pSe,
	const int32_t& seSize
) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	outPix.A = pImg[0].A;
	return outPix;
}




template <typename T, typename U>
inline T ImgErodeOnEdge
(
	const T* __restrict pImg,
	const U& compareVal, 
	const int32_t& imgStride, 
	const SE_Type* __restrict pSe,
	const int32_t& seSize, 
	const int32_t& numbLine, 
	const int32_t& numbPix,
	const int32_t& sizeX,
	const int32_t& sizeY
) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	const int32_t halfSe{ seSize >> 1 };
	const int32_t lineTop    = numbLine - halfSe;
	const int32_t lineBottom = numbLine + halfSe;
	const int32_t pixLeft    = numbPix  - halfSe;
	const int32_t pixRight   = numbPix  + halfSe;

	for (int32_t j = lineTop; j < lineBottom; j++)
	{
		const int32_t linIdx = MIN_VALUE((sizeY - 1), MAX_VALUE(0, j));
		const T* pLine = pImg + linIdx * imgStride;

		for (int32_t i = pixLeft; i < pixRight; i++)
		{
			const int32_t idxPix = MIN_VALUE((sizeX - 1), MAX_VALUE(0, i));
			if (0 != *pSe++)
			{
				outPix.R = MIN_VALUE(pLine[idxPix].R, outPix.R);
				outPix.G = MIN_VALUE(pLine[idxPix].G, outPix.G);
				outPix.B = MIN_VALUE(pLine[idxPix].B, outPix.B);
			}
		}
	}

	/* copy alpha channel value from source buffer */
	outPix.A = pImg[0].A;
	return outPix;
}


template <typename T, typename U>
inline T ImgErode 
(
	const T* __restrict pImg,
	const U& compareVal,
	const int32_t& imgStride,
	const SE_Type* __restrict pSe,
	const int32_t& seSize
) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	const int32_t halfSe{ seSize >> 1 };

#ifndef __NVCC__
__VECTORIZATION__
__LOOP_UNROLL(3)
#endif
	for (int32_t j = -halfSe; j <= halfSe; j++)
	{
		const T* pLine = pImg + j * imgStride;
#ifndef __NVCC__
__LOOP_UNROLL(3)
#endif
		for (int32_t i = -halfSe; i <= halfSe; i++)
		{
			if (0 != *pSe++)
			{
				outPix.R = MIN_VALUE(pLine[i].R, outPix.R);
				outPix.G = MIN_VALUE(pLine[i].G, outPix.G);
				outPix.B = MIN_VALUE(pLine[i].B, outPix.B);
			}
		}
	}

	/* copy alpha channel value from source buffer */
	outPix.A = pImg[0].A;

	return outPix;
}


template <typename T, typename U>
typename std::enable_if<std::is_same<T, PF_Pixel_VUYA_8u>::value || std::is_same<T, PF_Pixel_VUYA_32f>::value, T>::type>
inline T ImgDilateOnEdge
(
	const T* __restrict pImg,
	const U& compareVal,
	const int32_t& imgStride,
	const SE_Type* __restrict pSe,
	const int32_t& seSize,
	const int32_t& numbLine,
	const int32_t& numbPix,
	const int32_t& sizeX,
	const int32_t& sizeY
) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	outPix.A = pImg[0].A;
	return outPix;
}

template <typename T, typename U>
typename std::enable_if<std::is_same<T, PF_Pixel_VUYA_8u>::value || std::is_same<T, PF_Pixel_VUYA_32f>::value, T>::type>
inline T ImgDilate(const T* pImg, const U& compareVal, const int32_t& imgStride, const SE_Type* pSe, const int32_t& seSize) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	outPix.A = pImg[0].A;
	return outPix;
}



template <typename T, typename U>
inline T ImgDilateOnEdge
(
	const T* __restrict pImg,
	const U& compareVal,
	const int32_t& imgStride,
	const SE_Type* __restrict pSe,
	const int32_t& seSize,
	const int32_t& numbLine,
	const int32_t& numbPix,
	const int32_t& sizeX,
	const int32_t& sizeY
) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	const int32_t halfSe{ seSize >> 1 };
	const int32_t lineTop = numbLine - halfSe;
	const int32_t lineBottom = numbLine + halfSe;
	const int32_t pixLeft = numbPix - halfSe;
	const int32_t pixRight = numbPix + halfSe;

	for (int32_t j = lineTop; j < lineBottom; j++)
	{
		const int32_t linIdx = MIN_VALUE((sizeY - 1), MAX_VALUE(0, j));
		const T* pLine = pImg + linIdx * imgStride;

		for (int32_t i = pixLeft; i < pixRight; i++)
		{
			const int32_t idxPix = MIN_VALUE((sizeX - 1), MAX_VALUE(0, i));
			if (0 != *pSe++)
			{
				outPix.R = MAX_VALUE(pLine[idxPix].R, outPix.R);
				outPix.G = MAX_VALUE(pLine[idxPix].G, outPix.G);
				outPix.B = MAX_VALUE(pLine[idxPix].B, outPix.B);
			}
		}
	}

	/* copy alpha channel value from source buffer */
	outPix.A = pImg[0].A;
	return outPix;
}


template <typename T, typename U>
inline T ImgDilate
(
	const T* pImg,
	const U& compareVal,
	const int32_t& imgStride,
	const SE_Type* pSe,
	const int32_t& seSize
) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	const int32_t halfSe{ seSize >> 1 };

#ifndef __NVCC__
	__VECTORIZATION__
	__LOOP_UNROLL(3)
#endif
	for (int32_t j = -halfSe; j <= halfSe; j++)
	{
		const T* pLine = pImg + j * imgStride;
#ifndef __NVCC__
	__LOOP_UNROLL(3)
#endif
		for (int32_t i = -halfSe; i <= halfSe; i++)
		{
			if (0 != *pSe++)
			{
				outPix.R = MAX_VALUE(pLine[i].R, outPix.R);
				outPix.G = MAX_VALUE(pLine[i].G, outPix.G);
				outPix.B = MAX_VALUE(pLine[i].B, outPix.B);
			}
		}
	}

	/* copy alpha channel value from source buffer */
	outPix.A = pImg[0].A;
	return outPix;
}
