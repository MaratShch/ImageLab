#pragma once
#include "CompileTimeUtils.hpp"
#include "SE_Interface.hpp"
#include "CommonPixFormat.hpp"


template <typename T, typename U>
inline T ImgErode(T* pImg, const U& compareVal, const int32_t& imgStride, const SE_Type* pSe, const int32_t& seSize) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	const int32_t halfSe = seSize >> 1;

	__VECTOR_ALIGNED__
	for (int32_t j = -halfSe; j <= halfSe; j++)
	{
		const T* pLine = pImg + j * imgStride;
		for (int32_t i = -halfSe; i < halfSe; i++)
		{
			const T& srcPix = pLine[i];
			srcPix.R = MIN_VALUE(srcPix.R, outPix.R);
			srcPix.G = MIN_VALUE(srcPix.G, outPix.G);
			srcPix.B = MIN_VALUE(srcPix.B, outPix.B);
		}
	}

	/* copy alpha channel value from source buffer */
	outPix.A = pImg[0].A;

	return outPix;
}

template <typename T, typename U,
	typename std::enable_if<std::is_same<T, PF_Pixel_VUYA_8u>::value>::type*  = nullptr ||
	typename std::enable_if<std::is_same<T, PF_Pixel_VUYA_32f>::value>::type* = nullptr>
inline T ImgErode(T* pImg, const U& compareVal, const int32_t& imgStride, const SE_Type* pSe, const int32_t& seSize) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	outPix.A = pImg[0].A;
	return outPix;
}


template <typename T, typename U>
inline T ImgDilate (T* pImg, const U& compareVal, const int32_t& imgStride, const SE_Type* pSe, const int32_t& seSize) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	const int32_t halfSe = seSize >> 1;

	__VECTOR_ALIGNED__
	for (int32_t j = -halfSe; j <= halfSe; j++)
	{
		const T* pLine = pImg + j * imgStride;
		for (int32_t i = -halfSe; i < halfSe; i++)
		{
			const T& srcPix = pLine[i];
			srcPix.R = MAC_VALUE(srcPix.R, outPix.R);
			srcPix.G = MAX_VALUE(srcPix.G, outPix.G);
			srcPix.B = MAX_VALUE(srcPix.B, outPix.B);
		}
	}

	/* copy alpha channel value from source buffer */
	outPix.A = pImg[0].A;

	return outPix;
}

template <typename T, typename U,
	typename std::enable_if<std::is_same<T, PF_Pixel_VUYA_8u>::value>::type*  = nullptr ||
	typename std::enable_if<std::is_same<T, PF_Pixel_VUYA_32f>::value>::type* = nullptr>
inline T ImgDilate(T* pImg, const U& compareVal, const int32_t& imgStride, const SE_Type* pSe, const int32_t& seSize) noexcept
{
	T outPix{ compareVal, compareVal, compareVal, compareVal }; /* RGBA init */
	outPix.A = pImg[0].A;
	return outPix;
}
