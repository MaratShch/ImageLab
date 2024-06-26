#include "Avx2Median.hpp"
#include "Avx2MedianInternal.hpp"


inline void PartialVectorSort_4elem (__m256i a[4]) noexcept
{
	/*
	median element in [1] index

	0  x
	0  0

	*/
	VectorSort16uPacked (a[0], a[2]);
	VectorSort16uPacked (a[3], a[1]);
	VectorSort16uPacked (a[0], a[1]);
	VectorSort16uPacked (a[2], a[3]);
	VectorSort16uPacked (a[1], a[2]);
}


inline void PartialVectorSort_6elem (__m256i a[6]) noexcept
{
	/*
	median element in [2] index

	0  0  x
	0  0  0
	*/
	VectorSort16uPacked (a[1], a[2]);
	VectorSort16uPacked (a[4], a[5]);
	VectorSort16uPacked (a[0], a[2]);
	VectorSort16uPacked (a[3], a[5]);
	VectorSort16uPacked (a[0], a[1]);
	VectorSort16uPacked (a[3], a[4]);
	VectorSort16uPacked (a[0], a[4]);
	VectorSort16uPacked (a[1], a[5]);
	VectorSort16uPacked (a[0], a[2]);
	VectorSort16uPacked (a[1], a[3]);
	VectorSort16uPacked (a[2], a[3]);
}

inline void PartialVectorSort_9elem (__m256i a[9]) noexcept
{
	/*

	median element in [4] index

	0  0  0
	0  X  0
	0  0  0

	*/
	VectorSort16uPacked (a[1], a[2]);
	VectorSort16uPacked (a[4], a[5]);
	VectorSort16uPacked (a[7], a[8]);
	VectorSort16uPacked (a[0], a[1]);
	VectorSort16uPacked (a[3], a[4]);
	VectorSort16uPacked (a[6], a[7]);
	VectorSort16uPacked (a[1], a[2]);
	VectorSort16uPacked (a[4], a[5]);
	VectorSort16uPacked (a[7], a[8]);
	VectorSort16uPacked (a[0], a[3]);
	VectorSort16uPacked (a[5], a[8]);
	VectorSort16uPacked (a[4], a[7]);
	VectorSort16uPacked (a[3], a[6]);
	VectorSort16uPacked (a[1], a[4]);
	VectorSort16uPacked (a[2], a[5]);
	VectorSort16uPacked (a[4], a[7]);
	VectorSort16uPacked (a[4], a[2]);
	VectorSort16uPacked (a[6], a[4]);
	VectorSort16uPacked (a[4], a[2]);
}


inline void LoadLinePixel0 (uint64_t* __restrict pSrc, __m256i elemLine[2]) noexcept
{
	elemLine[0] = _mm256_loadu_si256 (reinterpret_cast<__m256i*>(pSrc));
	elemLine[1] = _mm256_loadu_si256 (reinterpret_cast<__m256i*>(pSrc + 1));
}

inline void LoadLinePixelLast(uint64_t* __restrict pSrc, __m256i elemline[2]) noexcept
{
	elemline[0] = _mm256_loadu_si256 (reinterpret_cast<__m256i*>(pSrc - 1));
	elemline[1] = _mm256_loadu_si256 (reinterpret_cast<__m256i*>(pSrc));
}


inline void LoadLinePixel (uint64_t* __restrict pSrc, __m256i elemLine[3]) noexcept
{
	elemLine[0] = _mm256_loadu_si256 (reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[1] = _mm256_loadu_si256 (reinterpret_cast<__m256i*>(pSrc));
	elemLine[2] = _mm256_loadu_si256 (reinterpret_cast<__m256i*>(pSrc + 1));
}

inline __m256i LoadFirstLineWindowPixel0 (uint64_t* __restrict pSrc, uint64_t* __restrict pNext, __m256i  elem[4]) noexcept
{
	                                  //  +----- 
	LoadLinePixel0 (pSrc, elem);      //  | X  0 
	LoadLinePixel0 (pNext, elem + 2); //  | 0  0
	return elem[0];
}


inline __m256i LoadLastLineWindowPixel0 (uint64_t* __restrict pPrev, uint64_t* __restrict pSrc, __m256i  elem[4]) noexcept
{
	LoadLinePixel0 (pPrev, elem);     //  | 0  0 
	LoadLinePixel0 (pSrc,  elem + 2); //  | X  0
	return elem[2];                   //  + ---- 
}

inline __m256i LoadFirstLineWindowPixelLast(uint64_t* __restrict pSrc, uint64_t* pNext, __m256i elem[4]) noexcept
{
							  			  //  ------+ 
	LoadLinePixelLast (pSrc, elem);       //   0  X |
	LoadLinePixelLast (pNext, elem + 2);  //   0  0 |
	return elem[1];
}

inline __m256i LoadLastLineWindowPixelLast(uint64_t* __restrict pPrev, uint64_t* pSrc, __m256i elem[4]) noexcept
{
	LoadLinePixelLast (pPrev, elem);      //   0  0 |
	LoadLinePixelLast (pSrc,  elem + 2);  //   0  X |
	return elem[1];                      //  ------+
}


inline __m256i LoadFirstLineWindowPixel (uint64_t* __restrict pSrc, uint64_t* __restrict pNext, __m256i  elem[6]) noexcept
{
	                                  // -------
	LoadLinePixel (pSrc, elem);       // 0  X  0
	LoadLinePixel (pNext, elem + 3);  // 0  0  0
	return elem[1];
}

inline __m256i LoadLastLineWindowPixel (uint64_t* __restrict pPrev, uint64_t* __restrict pSrc, __m256i  elem[6]) noexcept
{
	// -------
	LoadLinePixel (pPrev, elem);      // 0  0  0
	LoadLinePixel (pSrc,  elem + 3);  // 0  X  0
	return elem[4];                  // -------
}

inline __m256i LoadWindowPixel0 (uint64_t* __restrict pPrev, uint64_t* __restrict pSrc, uint64_t* __restrict pNext, __m256i elem[6]) noexcept
{
	LoadLinePixel0 (pPrev, elem);          //  | 0  0
	LoadLinePixel0 (pSrc,  elem + 2);      //  | X  0
	LoadLinePixel0 (pNext, elem + 4);      //  | 0  0
	return elem[2];
}

inline __m256i LoadWindowPixelLast (uint64_t* __restrict pPrev, uint64_t* __restrict pSrc, uint64_t* __restrict pNext, __m256i elem[6]) noexcept
{
	LoadLinePixelLast (pPrev, elem);          //  0  0 | 
	LoadLinePixelLast (pSrc,  elem + 2);      //  0  X |
	LoadLinePixelLast (pNext, elem + 4);      //  0  0 |
	return elem[3];
}

inline __m256i LoadWindow (uint64_t* __restrict pPrev, uint64_t* __restrict pSrc, uint64_t* __restrict pNext, __m256i elem[9]) noexcept
{
	LoadLinePixel (pPrev, elem);          //  0  0  0
	LoadLinePixel (pSrc,  elem + 3);      //  0  X  0
	LoadLinePixel (pNext, elem + 6);      //  0  0  0
	return elem[4];
}



/*
	make median filter with kernel 3x3 from packed format - BGRA444_8u by AVX2 instructions set:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

	LSB                            MSB
	+-------------------------------+
	| B | G | R | A | B | G | R | A | ...
	+-------------------------------+

*/
bool AVX2::Median::median_filter_3x3_RGB_4444_16u
(
	uint64_t* __restrict pInImage,
	uint64_t* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	const A_long& chanelMaskL,
	const A_long& chanelMaskH
	) noexcept
{
//	if (sizeY < 3 || sizeX < 40)
//		return Scalar::scalar_median_filter_3x3_BGRA_4444_8u(pInImage, pOutImage, sizeY, sizeX, linePitch);

//	CACHE_ALIGN PF_Pi xel_BGRA_8u  ScalarElem[9];
	constexpr A_long pixelsInVector{ static_cast<A_long>(sizeof(__m256i) / sizeof(uint64_t)) };

	A_long i, j;
	const A_long vectorLoadsInLine = sizeX / pixelsInVector;
	const A_long vectorizedLineSize = vectorLoadsInLine * pixelsInVector;
	const A_long lastPixelsInLine = sizeX - vectorizedLineSize;
	const A_long lastIdx = lastPixelsInLine - 1;

	const A_long shortSizeY{ sizeY - 1 };
	const A_long shortSizeX{ sizeX - pixelsInVector };

	const __m256i rgbMaskVector = _mm256_setr_epi32
	(
		chanelMaskL, /* mask A component for low part  of pixel 1 */
		chanelMaskH, /* mask A component for high part of pixel 1 */
		chanelMaskL, /* mask A component for low part  of pixel 2 */
		chanelMaskH, /* mask A component for high part of pixel 2 */
		chanelMaskL, /* mask A component for low part  of pixel 3 */
		chanelMaskH, /* mask A component for high part of pixel 3 */
		chanelMaskL, /* mask A component for low part  of pixel 4 */
		chanelMaskH  /* mask A component for high part of pixel 4 */
	);

#ifdef _DEBUG
	__m256i vecData[9]{};
#else
	CACHE_ALIGN __m256i vecData[9];
#endif

	/* PROCESS FIRST LINE IN FRAME */
	{
		uint64_t* __restrict pSrcVecCurrLine = pInImage;
		uint64_t* __restrict pSrcVecNextLine = pInImage + srcLinePitch;
		__m256i*  __restrict pSrcVecDstLine  = reinterpret_cast<__m256i*  __restrict>(pOutImage);

		/* process left frame edge in first line */
		const __m256i srcFirstPixel = LoadFirstLineWindowPixel0 (pSrcVecCurrLine, pSrcVecNextLine, vecData);
		PartialVectorSort_4elem (vecData);
		StoreByMask16u (pSrcVecDstLine, srcFirstPixel, vecData[1], rgbMaskVector);
		pSrcVecDstLine++;

		/* process first line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = LoadFirstLineWindowPixel (pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			PartialVectorSort_6elem (vecData);
			StoreByMask16u (pSrcVecDstLine, srcOrig, vecData[2], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* last pixel in first line */
		const __m256i srcOrigRight = LoadFirstLineWindowPixelLast (pSrcVecCurrLine + shortSizeX, pSrcVecNextLine + shortSizeX, vecData);
		PartialVectorSort_4elem (vecData);
		StoreByMask16u (pSrcVecDstLine, srcFirstPixel, vecData[1], rgbMaskVector);
#if 0
		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		if (0 != lastPixelsInLine)
		{
			PF_Pixel_BGRA_8u* pLastPixels = reinterpret_cast<PF_Pixel_BGRA_8u*>(pSrcVecDstLine);
			PF_Pixel_BGRA_8u* pSrcVecCurrLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + vectorizedLineSize);
			PF_Pixel_BGRA_8u* pSrcVecNextLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + vectorizedLineSize + linePitch);
			i = 0;

			for (; i < lastIdx; i++)
			{
				const PF_Pixel_BGRA_8u currentPixel = MedianLoad3x3::LoadWindowScalar_RGB(pSrcVecCurrLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, ScalarElem);
				MedianSort::PartialScalarSort_9_elem_RGB(ScalarElem);
				MedianStore::ScalarStore_RGB(pLastPixels + i, currentPixel, ScalarElem[4]);
			}
			const PF_Pixel_BGRA_8u currentPixel = MedianLoad3x3::LoadWindowScalarRight_RGB(pSrcVecCurrLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, ScalarElem);
			MedianSort::PartialScalarSort_9_elem_RGB(ScalarElem);
			MedianStore::ScalarStore_RGB(pLastPixels + i, currentPixel, ScalarElem[4]);
		} /* if (0 != lastPixelsInLine) */
#endif
	}

	/* PROCESS LINES IN FRAME FROM 1 to SIZEY-1 */
	for (j = 1; j < shortSizeY; j++)
	{
		uint64_t* __restrict pSrcVecPrevLine = (pInImage  + (j - 1) * srcLinePitch);
		uint64_t* __restrict pSrcVecCurrLine = (pInImage  + j       * srcLinePitch);
		uint64_t* __restrict pSrcVecNextLine = (pInImage  + (j + 1) * srcLinePitch);
		__m256i*  __restrict pSrcVecDstLine  = reinterpret_cast<__m256i*  __restrict>(pOutImage + j       * dstLinePitch);

		/* load first vectors from previous, current and next line */
		/* process left frame edge in first line */
		const __m256i srcOrigLeft = LoadWindowPixel0 (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		PartialVectorSort_6elem (vecData);
		StoreByMask16u (pSrcVecDstLine, srcOrigLeft, vecData[2], rgbMaskVector);
		pSrcVecDstLine++;

		/* process line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = LoadWindow (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			PartialVectorSort_9elem (vecData);
			StoreByMask16u (pSrcVecDstLine, srcOrig, vecData[4], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in last line */
		const __m256i srcOrigRight = LoadWindowPixelLast (pSrcVecPrevLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecNextLine + shortSizeX, vecData);
		PartialVectorSort_6elem (vecData);
		StoreByMask16u (pSrcVecDstLine, srcOrigRight, vecData[2], rgbMaskVector);
#if 0
		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		if (0 != lastPixelsInLine)
		{
			PF_Pixel_BGRA_8u* pLastPixels = reinterpret_cast<PF_Pixel_BGRA_8u*>(pSrcVecDstLine);
			PF_Pixel_BGRA_8u* pSrcVecPrevLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + (j - 1) * linePitch + vectorizedLineSize);
			PF_Pixel_BGRA_8u* pSrcVecCurrLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + j      * linePitch + vectorizedLineSize);
			PF_Pixel_BGRA_8u* pSrcVecNextLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + (j + 1) * linePitch + vectorizedLineSize);
			i = 0;

			for (; i < lastIdx; i++)
			{
				const PF_Pixel_BGRA_8u currentPixel = MedianLoad3x3::LoadWindowScalar_RGB(pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, ScalarElem);
				MedianSort::PartialScalarSort_9_elem_RGB(ScalarElem);
				MedianStore::ScalarStore_RGB(pLastPixels + i, currentPixel, ScalarElem[4]);
			}
			const PF_Pixel_BGRA_8u currentPixel = MedianLoad3x3::LoadWindowScalarRight_RGB(pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, ScalarElem);
			MedianSort::PartialScalarSort_9_elem_RGB(ScalarElem);
			MedianStore::ScalarStore_RGB(pLastPixels + i, currentPixel, ScalarElem[4]);
		}
#endif
	} /* END: process frame lines from 1 to sizeY-1 */

	/* PROCESS LAST FRAME LINE */
	{
		uint64_t* __restrict pSrcVecPrevLine = (pInImage  + (j - 1) * srcLinePitch);
		uint64_t* __restrict pSrcVecCurrLine = (pInImage  + j       * srcLinePitch);
		 __m256i* __restrict pSrcVecDstLine  = reinterpret_cast <__m256i* __restrict>(pOutImage + j       * dstLinePitch);

		/* process left frame edge in last line */
		const __m256i srcOrigLeft = LoadLastLineWindowPixel0 (pSrcVecPrevLine, pSrcVecCurrLine, vecData);
		PartialVectorSort_4elem (vecData);
		StoreByMask16u (pSrcVecDstLine, srcOrigLeft, vecData[1], rgbMaskVector);
		pSrcVecDstLine++;

		/* process first line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = LoadLastLineWindowPixel (pSrcVecPrevLine + i, pSrcVecCurrLine + i, vecData);
			PartialVectorSort_6elem (vecData);
			StoreByMask16u (pSrcVecDstLine, srcOrig, vecData[2], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in last line */
		const __m256i srcOrigRight = LoadLastLineWindowPixelLast (pSrcVecPrevLine + shortSizeX, pSrcVecCurrLine + shortSizeX, vecData);
		PartialVectorSort_4elem (vecData);
		StoreByMask16u (pSrcVecDstLine, srcOrigRight, vecData[1], rgbMaskVector);
#if 0
		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		if (0 != lastPixelsInLine)
		{
			PF_Pixel_BGRA_8u* pLastPixels = reinterpret_cast<PF_Pixel_BGRA_8u*>(pSrcVecDstLine);
			PF_Pixel_BGRA_8u* pSrcVecPrevLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + (j - 1) * linePitch + vectorizedLineSize);
			PF_Pixel_BGRA_8u* pSrcVecCurrLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + j       * linePitch + vectorizedLineSize);
			i = 0;

			for (; i < lastIdx; i++)
			{
				const PF_Pixel_BGRA_8u currentPixel = MedianLoad3x3::LoadWindowScalar_RGB(pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecCurrLine + i, ScalarElem);
				MedianSort::PartialScalarSort_9_elem_RGB(ScalarElem);
				MedianStore::ScalarStore_RGB(pLastPixels + i, currentPixel, ScalarElem[4]);
			}
			const PF_Pixel_BGRA_8u currentPixel = MedianLoad3x3::LoadWindowScalarRight_RGB(pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecCurrLine + i, ScalarElem);
			MedianSort::PartialScalarSort_9_elem_RGB(ScalarElem);
			MedianStore::ScalarStore_RGB(pLastPixels + i, currentPixel, ScalarElem[4]);
		}
#endif
	}

	return true;
}