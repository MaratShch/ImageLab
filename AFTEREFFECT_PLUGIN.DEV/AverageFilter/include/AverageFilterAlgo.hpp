#ifndef __IMAGE_LAB_AVERAGE_FILTER_ALGORITHM_IMPLEMENTATION__
#define __IMAGE_LAB_AVERAGE_FILTER_ALGORITHM_IMPLEMENTATION__

#include "CommonPixFormat.hpp"
#include "CommonPixFormatSFINAE.hpp"

using AverSumType = float;


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
void AverageFilterAlgo
(
	const T* __restrict pSrcImg,
	      T* __restrict pDstImg,
	A_long sizeX,
	A_long sizeY,
	A_long srcPitch,
	A_long dstPitch,
	A_long windowWidth
) noexcept
{
	A_long i, j;
	const A_long filterRadius = windowWidth >> 1;
	const A_long filterWindow = windowWidth * windowWidth;
	const AverSumType reciproc = static_cast<AverSumType>(1) / static_cast<AverSumType>(windowWidth);
	AverSumType aSum = 0;
	
	for (j = 0; j < sizeY; j++)
	{
		const A_long jMin = j - filterRadius;
		const A_long jMax = j + filterRadius;

		for (i = 0; i < sizeX; i++)
		{
			const A_long iMin = i - filterRadius;
			const A_long iMax = i + filterRadius;

		} /* for (i = 0; i < sizeX; i++) */

	} /* for (j = 0; j < sizeY; j++) */

	return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
void AverageFilterAlgo
(
	const T* __restrict pSrcImg,
	      T* __restrict pDstImg,
	A_long sizeX,
	A_long sizeY,
	A_long srcPitch,
	A_long dstPitch,
	A_long windowWidth
) noexcept
{
	A_long i, j;
	const A_long filterRadius = windowWidth >> 1;
	const A_long filterWindow = windowWidth * windowWidth;
	const AverSumType reciproc = static_cast<AverSumType>(1) / static_cast<AverSumType>(windowWidth);
	AverSumType aSum = 0;

	for (j = 0; j < sizeY; j++)
	{
		const A_long jMin = j - filterRadius;
		const A_long jMax = j + filterRadius;

		for (i = 0; i < sizeX; i++)
		{
			const A_long iMin = i - filterRadius;
			const A_long iMax = i + filterRadius;

		} /* for (i = 0; i < sizeX; i++) */

	} /* for (j = 0; j < sizeY; j++) */

	return;
}


template <typename T, std::enable_if_t<is_no_alpha_channel<T>::value>* = nullptr>
void AverageFilterAlgo
(
	const T* __restrict pSrcImg,
	      T* __restrict pDstImg,
	A_long sizeX,
	A_long sizeY,
	A_long srcPitch,
	A_long dstPitch,
	A_long windowWidth
) noexcept
{
	A_long i, j;
	const A_long filterRadius = windowWidth >> 1;
	const A_long filterWindow = windowWidth * windowWidth;
	const AverSumType reciproc = static_cast<AverSumType>(1) / static_cast<AverSumType>(windowWidth);
	AverSumType aSum = 0;

	for (j = 0; j < sizeY; j++)
	{
		const A_long jMin = j - filterRadius;
		const A_long jMax = j + filterRadius;

		for (i = 0; i < sizeX; i++)
		{
			const A_long iMin = i - filterRadius;
			const A_long iMax = i + filterRadius;

		} /* for (i = 0; i < sizeX; i++) */

	} /* for (j = 0; j < sizeY; j++) */

	return;
}


#endif /* __IMAGE_LAB_AVERAGE_FILTER_ALGORITHM_IMPLEMENTATION__ */
