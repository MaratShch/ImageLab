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
	constexpr size_t tmpBufSize = CreateAlignment(maxSeElemNumber, 16);
	constexpr size_t centralElementIdx = tmpBufSize >> 1;
	CACHE_ALIGN T    tmpBuf[tmpBufSize]{};

	const A_long seHalfSize{ seSize >> 1};
	const A_long xSlices{ width  / seSize };
	const A_long ySlices{ height / seSize };
	const A_long xFraction{ width  % seSize };
	const A_long yFraction{ height % seSize };

	A_long i, j, k, l;
	
	for (j = 0; j < height; j += seSize)
	{

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
	const U&                  minVal,
	const U&                  maxVal
) noexcept
{
	constexpr size_t tmpBufSize = CreateAlignment(maxSeElemNumber, 16);
	constexpr size_t centralElementIdx = tmpBufSize >> 1;
	CACHE_ALIGN T    tmpBuf[tmpBufSize]{};

	const A_long xSlices{ width  / seSize };
	const A_long ySlices{ height / seSize };
	const A_long xFraction{ width  % seSize };
	const A_long yFraction{ height % seSize };

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