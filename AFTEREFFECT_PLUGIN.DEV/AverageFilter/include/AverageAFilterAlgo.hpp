#ifndef __IMAGE_LAB_ARIPHMETIC_AVERAGE_FILTER_ALGORITHM_IMPLEMENTATION__
#define __IMAGE_LAB_ARIPHMETIC_AVERAGE_FILTER_ALGORITHM_IMPLEMENTATION__

#include "CommonPixFormat.hpp"
#include "CommonPixFormatSFINAE.hpp"

// ariphmetic average type
using TAAverSum = float;


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
A_long AverageFilterAlgo
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
	const A_long filterRadius = windowWidth >> 1;
	const A_long filterWindow = windowWidth * windowWidth;
	const TAAverSum reciproc = static_cast<TAAverSum>(1) / static_cast<TAAverSum>(windowWidth);
	
	for (A_long j = 0; j < sizeY; j++)
	{
		const A_long jMin = j - filterRadius;
		const A_long jMax = j + filterRadius;

		for (A_long i = 0; i < sizeX; i++)
		{
			const A_long iMin = i - filterRadius;
			const A_long iMax = i + filterRadius;

			TAAverSum aSumR = 0, asumG = 0, asumB = 0;
		} /* for (i = 0; i < sizeX; i++) */

	} /* for (j = 0; j < sizeY; j++) */

	return PF_Err_NONE;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
A_long AverageFilterAlgo
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
	const A_long filterRadius = windowWidth >> 1;
	const A_long filterWindow = windowWidth * windowWidth;
	const TAAverSum reciproc = static_cast<TAAverSum>(1) / static_cast<TAAverSum>(windowWidth);

	for (A_long j = 0; j < sizeY; j++)
	{
		const A_long jMin = j - filterRadius;
		const A_long jMax = j + filterRadius;

		for (A_long i = 0; i < sizeX; i++)
		{
			const A_long iMin = i - filterRadius;
			const A_long iMax = i + filterRadius;

			TAAverSum aSumR = 0, asumG = 0, asumB = 0;
		} /* for (i = 0; i < sizeX; i++) */

	} /* for (j = 0; j < sizeY; j++) */

	return PF_Err_NONE;
}


template <typename T, std::enable_if_t<is_no_alpha_channel<T>::value>* = nullptr>
A_long AverageFilterAlgo
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
	const A_long filterRadius = windowWidth >> 1;
	const A_long filterWindow = windowWidth * windowWidth;
	const TAAverSum reciproc = static_cast<TAAverSum>(1) / static_cast<TAAverSum>(windowWidth);

	for (A_long j = 0; j < sizeY; j++)
	{
		const A_long jMin = j - filterRadius;
		const A_long jMax = j + filterRadius;

		for (A_long i = 0; i < sizeX; i++)
		{
			const A_long iMin = i - filterRadius;
			const A_long iMax = i + filterRadius;

			TAAverSum aSumR = 0, asumG = 0, asumB = 0;
		} /* for (i = 0; i < sizeX; i++) */

	} /* for (j = 0; j < sizeY; j++) */

	return PF_Err_NONE;
}


#endif /* __IMAGE_LAB_ARIPHMETIC_AVERAGE_FILTER_ALGORITHM_IMPLEMENTATION__ */
