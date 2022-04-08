#include "MorphologyProc.hpp"
#include <memory>


template <typename T, typename U>
inline void Morphology_Erode
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long&             seSize,
	const A_long&             height,
	const A_long&             width,
	const A_long&             srcPitch,
	const A_long&             dstPitch,
	const U&                  compareVal
) noexcept
{
	A_long i, j;
	const A_long seHalfSize{ seSize >> 1 };
	const A_long w1{ width - seHalfSize };
	const A_long w2{ height - seHalfSize };

	__VECTOR_ALIGNED__
	for (j = 0; j < seHalfSize; j++)
	{
		const A_long idx{ j * dstPitch };
		for (i = 0; i < width; i++)
			pDst[idx + i] = ImgErodeOnEdge(pSrc, compareVal, srcPitch, pSe, seSize, j, i, width, height);
	}

	__VECTOR_ALIGNED__
	for (; j < w2; j++)
	{
		const A_long idx{ j * dstPitch };

		for (i = 0; i < seHalfSize; i++)
			pDst[idx + i] = ImgErodeOnEdge(pSrc, compareVal, srcPitch, pSe, seSize, j, i, width, height);

		for (; i < w1; i++)
			pDst[idx + i] = ImgErode(pSrc + idx + i, compareVal, srcPitch, pSe, seSize);

		for (; i < width; i++)
			pDst[idx + i] = ImgErodeOnEdge(pSrc, compareVal, srcPitch, pSe, seSize, j, i, width, height);
	}

	__VECTOR_ALIGNED__
	for (; j < height; j++)
	{
		const A_long idx{ j * dstPitch };
		for (i = 0; i < width; i++)
			pDst[idx + i] = ImgErodeOnEdge(pSrc, compareVal, srcPitch, pSe, seSize, j, i, width, height);
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
	const A_long&             srcPitch,
	const A_long&             dstPitch,
	const U&                  compareVal
) noexcept
{
	A_long i, j;
	const A_long seHalfSize{ seSize >> 1 };
	const A_long w1{ width - seHalfSize };
	const A_long w2{ height - seHalfSize };

	__VECTOR_ALIGNED__
	for (j = 0; j < seHalfSize; j++)
	{
		const A_long idx{ j * dstPitch };
		for (i = 0; i < width; i++)
			pDst[idx + i] = ImgDilateOnEdge(pSrc, compareVal, srcPitch, pSe, seSize, j, i, width, height);
	}

	__VECTOR_ALIGNED__
	for (; j < w2; j++)
	{
		const A_long idx{ j * dstPitch };

		for (i = 0; i < seHalfSize; i++)
			pDst[idx + i] = ImgDilateOnEdge(pSrc, compareVal, srcPitch, pSe, seSize, j, i, width, height);

		for (; i < w1; i++)
			pDst[idx + i] = ImgDilate(pSrc + idx + i, compareVal, srcPitch, pSe, seSize);
		
		for (; i < width; i++)
			pDst[idx + i] = ImgDilateOnEdge(pSrc, compareVal, srcPitch, pSe, seSize, j, i, width, height);
	}

	__VECTOR_ALIGNED__
	for (; j < height; j++)
	{
		const A_long idx{ j * dstPitch };
		for (i = 0; i < width; i++)
			pDst[idx + i] = ImgDilateOnEdge(pSrc, compareVal, srcPitch, pSe, seSize, j, i, width, height);
	}

	return;
}




template <typename T, typename U>
inline void Morphology_Open /* Erode -> Dilate */
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long&             seSize,
	const A_long&             height,
	const A_long&             width,
	const A_long&             srcPitch,
	const A_long&             dstPitch,
	const U&                  valErode,
	const U&                  valDilate
) noexcept
{
	/* allocate temporary memory storage */
	const size_t tmpMemSize = CreateAlignment(height * width, CPU_PAGE_SIZE);
	T* pTmpStorage = new T[tmpMemSize];
	if (nullptr != pTmpStorage)
	{
		Morphology_Erode  (pSrc, pTmpStorage, pSe, seSize, height, width, srcPitch, width, valErode);
		Morphology_Dilate (pTmpStorage, pDst, pSe, seSize, height, width, width, dstPitch, valDilate);
		delete [] pTmpStorage;
		pTmpStorage = nullptr;
	}

	return;
}


template <typename T, typename U>
inline void Morphology_Close /* Dilate -> Erode */
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long&             seSize,
	const A_long&             height,
	const A_long&             width,
	const A_long&             srcPitch,
	const A_long&             dstPitch,
	const U&                  valErode,
	const U&                  valDilate
) noexcept
{
	/* allocate temporary memory storage */
	const size_t tmpMemSize = CreateAlignment(height * width, CPU_PAGE_SIZE);
	T* pTmpStorage = new T[tmpMemSize];
	if (nullptr != pTmpStorage)
	{
		Morphology_Dilate (pSrc, pTmpStorage, pSe, seSize, height, width, srcPitch, width, valDilate);
		Morphology_Erode  (pTmpStorage, pDst, pSe, seSize, height, width, width, dstPitch, valErode);
		delete [] pTmpStorage;
		pTmpStorage = nullptr;
	}
	return;
}


template <typename T, typename U>
inline void Morphology_Thin
(
	const T*       __restrict pSrc,
	T*             __restrict pDst,
	const SE_Type* __restrict pSe,
	const A_long&             seSize,
	const A_long&             height,
	const A_long&             width,
	const A_long&             srcPitch,
	const A_long&             dstPitch,
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
	const A_long&             seSize,
	const A_long&             height,
	const A_long&             widt,
	const A_long&             srcPitch,
	const A_long&             dstPitch,
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
	const A_long&             seSize,
	const A_long&             height,
	const A_long&             widt,
	const A_long&             srcPitch,
	const A_long&             dstPitch,
	const U&                  val
) noexcept
{
	return;
}