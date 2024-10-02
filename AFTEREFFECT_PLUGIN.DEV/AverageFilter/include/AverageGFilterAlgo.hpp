#ifndef __IMAGE_LAB_GEOMETRIC_AVERAGE_FILTER_ALGORITHM_IMPLEMENTATION__
#define __IMAGE_LAB_GEOMETRIC_AVERAGE_FILTER_ALGORITHM_IMPLEMENTATION__

#include "CommonPixFormat.hpp"
#include "CommonPixFormatSFINAE.hpp"

// geometric average type
using TGAverSum = float;


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
A_long GeomethricAverageFilterAlgo
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
	const TGAverSum reciproc = static_cast<TAAverSum>(1) / static_cast<TAAverSum>(filterWindow);
	constexpr TGAverSum One{ 1 };
	constexpr TGAverSum Ten{ 10 };

	for (A_long j = 0; j < sizeY; j++)
	{
		const A_long jMin = j - filterRadius;
		const A_long jMax = j + filterRadius;

		__VECTORIZATION__
		for (A_long i = 0; i < sizeX; i++)
		{
			const A_long iMin = i - filterRadius;
			const A_long iMax = i + filterRadius;
			TAAverSum aSumR = 0, aSumG = 0, aSumB = 0;

			for (A_long k = jMin; k <= jMax; k++)
			{
				const A_long lineIdx = FastCompute::Min(sizeY - 1, FastCompute::Max(0, k));
				const A_long jIdx = lineIdx * srcPitch;
				
				for (A_long l = iMin; l <= iMax; l++)
				{
					const A_long iIdx = FastCompute::Min(sizeX - 1, FastCompute::Max(0, l));
					const A_long inPix = jIdx + iIdx;

					aSumR += (FastCompute::Log10f(static_cast<TGAverSum>(pSrcImg[inPix].R) + One));
					aSumG += (FastCompute::Log10f(static_cast<TGAverSum>(pSrcImg[inPix].G) + One));
					aSumB += (FastCompute::Log10f(static_cast<TGAverSum>(pSrcImg[inPix].B) + One));
				} // for (A_long l = iMin; l <= iMax; l++)

			} // for (A_long k = jMin; k <= jMax; j++)

			const A_long srcPixIdx = j * srcPitch + i;
			const A_long dstPixIdx = j * dstPitch + i;

			pDstImg[dstPixIdx].A = pSrcImg[srcPixIdx].A;
			pDstImg[dstPixIdx].R = static_cast<decltype(pDstImg[dstPixIdx].R)>(FastCompute::Pow(Ten, aSumR * reciproc) - One);
			pDstImg[dstPixIdx].G = static_cast<decltype(pDstImg[dstPixIdx].G)>(FastCompute::Pow(Ten, aSumG * reciproc) - One);
			pDstImg[dstPixIdx].B = static_cast<decltype(pDstImg[dstPixIdx].B)>(FastCompute::Pow(Ten, aSumB * reciproc) - One);

		} // for (i = 0; i < sizeX; i++)

	} // for (j = 0; j < sizeY; j++)

	return PF_Err_NONE;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
A_long GeomethricAverageFilterAlgo
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
	const TGAverSum reciproc = static_cast<TAAverSum>(1) / static_cast<TAAverSum>(filterWindow);
	constexpr TGAverSum One{ 1 };
	constexpr TGAverSum Ten{ 10 };

	for (A_long j = 0; j < sizeY; j++)
	{
		const A_long jMin = j - filterRadius;
		const A_long jMax = j + filterRadius;

		__VECTORIZATION__
		for (A_long i = 0; i < sizeX; i++)
		{
			const A_long iMin = i - filterRadius;
			const A_long iMax = i + filterRadius;
			TAAverSum aSumY = 0;

			for (A_long k = jMin; k <= jMax; k++)
			{
				const A_long lineIdx = FastCompute::Min(sizeY - 1, FastCompute::Max(0, k));
				const A_long jIdx = lineIdx * srcPitch;

				for (A_long l = iMin; l <= iMax; l++)
				{
					const A_long iIdx = FastCompute::Min(sizeX - 1, FastCompute::Max(0, l));
					const A_long inPix = jIdx + iIdx;

					aSumY += (FastCompute::Log10f(static_cast<TGAverSum>(pSrcImg[inPix].Y) + One));
				} // for (A_long l = iMin; l <= iMax; l++)

			} // for (A_long k = jMin; k <= jMax; j++)

			const A_long srcPixIdx = j * srcPitch + i;
			const A_long dstPixIdx = j * dstPitch + i;

			pDstImg[dstPixIdx].A = pSrcImg[srcPixIdx].A;
			pDstImg[dstPixIdx].U = pSrcImg[srcPixIdx].U;
			pDstImg[dstPixIdx].V = pSrcImg[srcPixIdx].V;
			pDstImg[dstPixIdx].Y = static_cast<decltype(pDstImg[dstPixIdx].Y)>(FastCompute::Pow(Ten, aSumY * reciproc) - One);

		} // for (i = 0; i < sizeX; i++)

	} // for (j = 0; j < sizeY; j++)

	return PF_Err_NONE;
}


template <typename T, std::enable_if_t<is_no_alpha_channel<T>::value>* = nullptr>
A_long GeomethricAverageFilterAlgo
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
	const TGAverSum reciproc = static_cast<TAAverSum>(1) / static_cast<TAAverSum>(filterWindow);
	constexpr TGAverSum One{ 1 };
	constexpr TGAverSum Ten{ 10 };

	for (A_long j = 0; j < sizeY; j++)
	{
		const A_long jMin = j - filterRadius;
		const A_long jMax = j + filterRadius;

		__VECTORIZATION__
		for (A_long i = 0; i < sizeX; i++)
		{
			const A_long iMin = i - filterRadius;
			const A_long iMax = i + filterRadius;
			TAAverSum aSumR = 0, aSumG = 0, aSumB = 0;

			for (A_long k = jMin; k <= jMax; k++)
			{
				const A_long lineIdx = FastCompute::Min(sizeY - 1, FastCompute::Max(0, k));
				const A_long jIdx = lineIdx * srcPitch;

				for (A_long l = iMin; l <= iMax; l++)
				{
					const A_long iIdx = FastCompute::Min(sizeX - 1, FastCompute::Max(0, l));
					const A_long inPix = jIdx + iIdx;
					
					aSumR += (FastCompute::Log10f(static_cast<TGAverSum>(pSrcImg[inPix].R) + One));
					aSumG += (FastCompute::Log10f(static_cast<TGAverSum>(pSrcImg[inPix].G) + One));
					aSumB += (FastCompute::Log10f(static_cast<TGAverSum>(pSrcImg[inPix].B) + One));
				} // for (A_long l = iMin; l <= iMax; l++)

			} // for (A_long k = jMin; k <= jMax; j++)

			const A_long dstPixIdx = j * dstPitch + i;
			pDstImg[dstPixIdx].R = static_cast<decltype(pDstImg[dstPixIdx].R)>(FastCompute::Pow(Ten, aSumR * reciproc) - One);
			pDstImg[dstPixIdx].G = static_cast<decltype(pDstImg[dstPixIdx].G)>(FastCompute::Pow(Ten, aSumG * reciproc) - One);
			pDstImg[dstPixIdx].B = static_cast<decltype(pDstImg[dstPixIdx].B)>(FastCompute::Pow(Ten, aSumB * reciproc) - One);

		} // for (i = 0; i < sizeX; i++)

	} // for (j = 0; j < sizeY; j++)

	return PF_Err_NONE;
}


#endif // __IMAGE_LAB_GEOMETRIC_AVERAGE_FILTER_ALGORITHM_IMPLEMENTATION__
