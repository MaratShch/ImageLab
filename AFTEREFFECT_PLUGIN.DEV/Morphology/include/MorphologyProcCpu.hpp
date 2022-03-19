#include "MorphologyProc.hpp"


template <typename T, typename U>
inline void Morphology_Erode
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long&             seSize,
	const A_long&             height,
	const A_long&             width,
	const A_long&             pitch,
	const U&                  compareVal
) noexcept
{
	int32_t i, j;
	const int32_t seHalfSize{ seSize >> 1 };
	const int32_t w1{ width - seHalfSize };
	const int32_t w2{ height - seHalfSize };

	__VECTOR_ALIGNED__
	for (j = 0; j < seHalfSize; j++)
	{
		const int32_t idx{ j * pitch };
		for (i = 0; i < width; i++)
			pDst[idx + i] = ImgErodeOnEdge(pSrc, compareVal, pitch, pSe, seSize, j, i, width, height);
	}

	__VECTOR_ALIGNED__
	for (; j < w2; j++)
	{
		const int32_t idx{ j * pitch };

		for (i = 0; i < seHalfSize; i++)
			pDst[idx + i] = ImgErodeOnEdge(pSrc, compareVal, pitch, pSe, seSize, j, i, width, height);

		for (; i < w1; i++)
			pDst[idx + i] = ImgErode(pSrc + idx + i, compareVal, pitch, pSe, seSize);

		for (; i < width; i++)
			pDst[idx + i] = ImgErodeOnEdge(pSrc, compareVal, pitch, pSe, seSize, j, i, width, height);
	}

	__VECTOR_ALIGNED__
	for (; j < height; j++)
	{
		const int32_t idx{ j * pitch };
		for (i = 0; i < width; i++)
			pDst[idx + i] = ImgErodeOnEdge(pSrc, compareVal, pitch, pSe, seSize, j, i, width, height);
	}

	return;
}


template <typename T, typename U>
inline void Morphology_Dilate
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long&             seSize,
	const A_long&             height,
	const A_long&             width,
	const A_long&             pitch,
	const U&                  compareVal
) noexcept
{
	int32_t i, j;
	const int32_t seHalfSize{ seSize >> 1 };
	const int32_t w1{ width - seHalfSize };
	const int32_t w2{ height - seHalfSize };

	__VECTOR_ALIGNED__
	for (j = 0; j < seHalfSize; j++)
	{
		const int32_t idx{ j * pitch };
		for (i = 0; i < width; i++)
			pDst[idx + i] = ImgDilateOnEdge(pSrc, compareVal, pitch, pSe, seSize, j, i, width, height);
	}

	__VECTOR_ALIGNED__
	for (; j < w2; j++)
	{
		const int32_t idx{ j * pitch };

		for (i = 0; i < seHalfSize; i++)
			pDst[idx + i] = ImgDilateOnEdge(pSrc, compareVal, pitch, pSe, seSize, j, i, width, height);

		for (; i < w1; i++)
			pDst[idx + i] = ImgDilate(pSrc + idx + i, compareVal, pitch, pSe, seSize);
		
		for (; i < width; i++)
			pDst[idx + i] = ImgDilateOnEdge(pSrc, compareVal, pitch, pSe, seSize, j, i, width, height);
	}

	__VECTOR_ALIGNED__
	for (; j < height; j++)
	{
		const int32_t idx{ j * pitch };
		for (i = 0; i < width; i++)
			pDst[idx + i] = ImgDilateOnEdge(pSrc, compareVal, pitch, pSe, seSize, j, i, width, height);
	}

	return;
}


template <typename T, typename U>
inline void Morphology_Open
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long              seSize,
	const A_long              height,
	const A_long              widt,
	const A_long              pitch,
	const U&                  minVal,
	const U&                  maxVal
) noexcept
{
	return;
}


template <typename T, typename U>
inline void Morphology_Close
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long              seSize,
	const A_long              height,
	const A_long              widt,
	const A_long              pitch,
	const U&                  minVal,
	const U&                  maxVal
) noexcept
{
	return;
}


template <typename T, typename U>
inline void Morphology_Thin
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long              seSize,
	const A_long              height,
	const A_long              widt,
	const A_long              pitch,
	const U&                  val
) noexcept
{
	return;
}


template <typename T, typename U>
inline void Morphology_Thick
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long              seSize,
	const A_long              height,
	const A_long              widt,
	const A_long              pitch,
	const U&                  val
) noexcept
{
	return;
}


template <typename T, typename U>
inline void Morphology_Gradient
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long              seSize,
	const A_long              height,
	const A_long              widt,
	const A_long              pitch,
	const U&                  val
) noexcept
{
	return;
}