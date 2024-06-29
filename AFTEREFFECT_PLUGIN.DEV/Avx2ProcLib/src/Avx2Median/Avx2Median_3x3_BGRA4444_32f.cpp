#include "Avx2Median.hpp"
#include "Avx2MedianInternal.hpp"


inline void PartialVectorSort_4elem (__m256 a[4]) noexcept
{
	/*
	median element in [1] index

	0  x
	0  0

	*/
	VectorSort32fPacked (a[0], a[2]);
	VectorSort32fPacked (a[3], a[1]);
	VectorSort32fPacked (a[0], a[1]);
	VectorSort32fPacked (a[2], a[3]);
	VectorSort32fPacked (a[1], a[2]);
}


inline void PartialVectorSort_6elem (__m256 a[6]) noexcept
{
	/*
	median element in [2] index

	0  0  x
	0  0  0
	*/
	VectorSort32fPacked (a[1], a[2]);
	VectorSort32fPacked (a[4], a[5]);
	VectorSort32fPacked (a[0], a[2]);
	VectorSort32fPacked (a[3], a[5]);
	VectorSort32fPacked (a[0], a[1]);
	VectorSort32fPacked (a[3], a[4]);
	VectorSort32fPacked (a[0], a[4]);
	VectorSort32fPacked (a[1], a[5]);
	VectorSort32fPacked (a[0], a[2]);
	VectorSort32fPacked (a[1], a[3]);
	VectorSort32fPacked (a[2], a[3]);
}

inline void PartialVectorSort_9elem (__m256 a[9]) noexcept
{
	/*

	median element in [4] index

	0  0  0
	0  X  0
	0  0  0

	*/
	VectorSort32fPacked (a[1], a[2]);
	VectorSort32fPacked (a[4], a[5]);
	VectorSort32fPacked (a[7], a[8]);
	VectorSort32fPacked (a[0], a[1]);
	VectorSort32fPacked (a[3], a[4]);
	VectorSort32fPacked (a[6], a[7]);
	VectorSort32fPacked (a[1], a[2]);
	VectorSort32fPacked (a[4], a[5]);
	VectorSort32fPacked (a[7], a[8]);
	VectorSort32fPacked (a[0], a[3]);
	VectorSort32fPacked (a[5], a[8]);
	VectorSort32fPacked (a[4], a[7]);
	VectorSort32fPacked (a[3], a[6]);
	VectorSort32fPacked (a[1], a[4]);
	VectorSort32fPacked (a[2], a[5]);
	VectorSort32fPacked (a[4], a[7]);
	VectorSort32fPacked (a[4], a[2]);
	VectorSort32fPacked (a[6], a[4]);
	VectorSort32fPacked (a[4], a[2]);
}


inline void LoadLinePixel0 (__m128* __restrict pSrc, __m256 elemLine[2]) noexcept
{
	elemLine[0] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc));
	elemLine[1] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc + 1));
}

inline void LoadLinePixelLast (__m128* __restrict pSrc, __m256 elemline[2]) noexcept
{
	elemline[0] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc - 1));
	elemline[1] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc));
}


inline void LoadLinePixel (__m128* __restrict pSrc, __m256 elemLine[3]) noexcept
{
	elemLine[0] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc - 1));
	elemLine[1] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc));
	elemLine[2] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc + 1));
}

inline __m256 LoadFirstLineWindowPixel0 (__m128* __restrict pSrc, __m128* __restrict pNext, __m256  elem[4]) noexcept
{
	                                  //  +----- 
	LoadLinePixel0 (pSrc, elem);      //  | X  0 
	LoadLinePixel0 (pNext, elem + 2); //  | 0  0
	return elem[0];
}


inline __m256 LoadLastLineWindowPixel0 (__m128* __restrict pPrev, __m128* __restrict pSrc, __m256 elem[4]) noexcept
{
	LoadLinePixel0 (pPrev, elem);     //  | 0  0 
	LoadLinePixel0 (pSrc,  elem + 2); //  | X  0
	return elem[2];                   //  + ---- 
}

inline __m256 LoadFirstLineWindowPixelLast (__m128* __restrict pSrc, __m128* pNext, __m256 elem[4]) noexcept
{
							  			  //  ------+ 
	LoadLinePixelLast (pSrc, elem);       //   0  X |
	LoadLinePixelLast (pNext, elem + 2);  //   0  0 |
	return elem[1];
}

inline __m256 LoadLastLineWindowPixelLast (__m128* __restrict pPrev, __m128* pSrc, __m256 elem[4]) noexcept
{
	LoadLinePixelLast(pPrev, elem);      //   0  0 |
	LoadLinePixelLast(pSrc,  elem + 2);  //   0  X |
	return elem[1];                      //  ------+
}


inline __m256 LoadFirstLineWindowPixel (__m128* __restrict pSrc, __m128* __restrict pNext, __m256 elem[6]) noexcept
{
	                                  // -------
	LoadLinePixel (pSrc, elem);       // 0  X  0
	LoadLinePixel (pNext, elem + 3);  // 0  0  0
	return elem[1];
}

inline __m256 LoadLastLineWindowPixel (__m128* __restrict pPrev, __m128* __restrict pSrc, __m256 elem[6]) noexcept
{
	// -------
	LoadLinePixel (pPrev, elem);      // 0  0  0
	LoadLinePixel (pSrc,  elem + 3);  // 0  X  0
	return elem[4];                  // -------
}

inline __m256 LoadWindowPixel0(__m128* __restrict pPrev, __m128* __restrict pSrc, __m128* __restrict pNext, __m256 elem[6]) noexcept
{
	LoadLinePixel0 (pPrev, elem);          //  | 0  0
	LoadLinePixel0 (pSrc,  elem + 2);      //  | X  0
	LoadLinePixel0 (pNext, elem + 4);      //  | 0  0
	return elem[2];
}

inline __m256 LoadWindowPixelLast (__m128* __restrict pPrev, __m128* __restrict pSrc, __m128* __restrict pNext, __m256 elem[6]) noexcept
{
	LoadLinePixelLast (pPrev, elem);          //  0  0 | 
	LoadLinePixelLast (pSrc,  elem + 2);      //  0  X |
	LoadLinePixelLast (pNext, elem + 4);      //  0  0 |
	return elem[3];
}

inline __m256 LoadWindow (__m128* __restrict pPrev, __m128* __restrict pSrc, __m128* __restrict pNext, __m256 elem[9]) noexcept
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

bool AVX2::Median::median_filter_3x3_RGB_4444_32f
(
	__m128* __restrict pInImage,
	__m128* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch
) noexcept
{
//	if (sizeY < 3 || sizeX < 40)
//		return Scalar::scalar_median_filter_3x3_BGRA_4444_8u(pInImage, pOutImage, sizeY, sizeX, linePitch);

	constexpr A_long pixelsInVector{ static_cast<A_long>(sizeof(__m256) / sizeof(__m128)) };

	A_long i, j;
	const A_long vectorLoadsInLine = sizeX / pixelsInVector;
	const A_long vectorizedLineSize = vectorLoadsInLine * pixelsInVector;
	const A_long lastPixelsInLine = sizeX - vectorizedLineSize;
	const A_long lastIdx = lastPixelsInLine - 1;

	const A_long shortSizeY{ sizeY - 1 };
	const A_long shortSizeX{ sizeX - pixelsInVector };

	constexpr float fMaskEnable = static_cast<float>(0xFFFFFFFFu);
	constexpr int blendMask = 0x77;

#ifdef _DEBUG
	__m256 vecData[9]{};
#else
	CACHE_ALIGN __m256 vecData[9];
#endif

	/* PROCESS FIRST LINE IN FRAME */
	{
		__m128* __restrict pSrcVecCurrLine = reinterpret_cast<__m128* __restrict>(pInImage);
		__m128* __restrict pSrcVecNextLine = reinterpret_cast<__m128* __restrict>(pInImage + srcLinePitch);
		__m256*  __restrict pSrcVecDstLine  = reinterpret_cast<__m256*  __restrict>(pOutImage);

		/* process left frame edge in first line */
		const __m256 srcFirstPixel = LoadFirstLineWindowPixel0 (pSrcVecCurrLine, pSrcVecNextLine, vecData);
		PartialVectorSort_4elem (vecData);
		StoreByMask32f (pSrcVecDstLine, srcFirstPixel, vecData[1], blendMask);
		pSrcVecDstLine++;

		/* process first line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256 srcOrig = LoadFirstLineWindowPixel (pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			PartialVectorSort_6elem (vecData);
			StoreByMask32f (pSrcVecDstLine, srcOrig, vecData[2], blendMask);
			pSrcVecDstLine++;
		}

		/* last pixel in first line */
		const __m256 srcOrigRight = LoadFirstLineWindowPixelLast (pSrcVecCurrLine + shortSizeX, pSrcVecNextLine + shortSizeX, vecData);
		PartialVectorSort_4elem (vecData);
		StoreByMask32f (pSrcVecDstLine, srcFirstPixel, vecData[1], blendMask);
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
		__m128* __restrict pSrcVecPrevLine = (pInImage  + (j - 1) * srcLinePitch);
		__m128* __restrict pSrcVecCurrLine = (pInImage  + j       * srcLinePitch);
		__m128* __restrict pSrcVecNextLine = (pInImage  + (j + 1) * srcLinePitch);
		__m256* __restrict pSrcVecDstLine  = reinterpret_cast<__m256*  __restrict>(pOutImage + j * dstLinePitch);

		/* load first vectors from previous, current and next line */
		/* process left frame edge in first line */
		const __m256 srcOrigLeft = LoadWindowPixel0 (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		PartialVectorSort_6elem (vecData);
		StoreByMask32f (pSrcVecDstLine, srcOrigLeft, vecData[2], blendMask);
		pSrcVecDstLine++;

		/* process line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256 srcOrig = LoadWindow (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			PartialVectorSort_9elem (vecData);
			StoreByMask32f (pSrcVecDstLine, srcOrig, vecData[4], blendMask);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in last line */
		const __m256 srcOrigRight = LoadWindowPixelLast (pSrcVecPrevLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecNextLine + shortSizeX, vecData);
		PartialVectorSort_6elem (vecData);
		StoreByMask32f (pSrcVecDstLine, srcOrigRight, vecData[2], blendMask);
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
		__m128* __restrict pSrcVecPrevLine = (pInImage  + (j - 1) * srcLinePitch);
		__m128* __restrict pSrcVecCurrLine = (pInImage  + j       * srcLinePitch);
		__m256* __restrict pSrcVecDstLine  = reinterpret_cast <__m256* __restrict>(pOutImage + j * dstLinePitch);

		/* process left frame edge in last line */
		const __m256 srcOrigLeft = LoadLastLineWindowPixel0 (pSrcVecPrevLine, pSrcVecCurrLine, vecData);
		PartialVectorSort_4elem (vecData);
		StoreByMask32f (pSrcVecDstLine, srcOrigLeft, vecData[1], blendMask);
		pSrcVecDstLine++;

		/* process first line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256 srcOrig = LoadLastLineWindowPixel (pSrcVecPrevLine + i, pSrcVecCurrLine + i, vecData);
			PartialVectorSort_6elem (vecData);
			StoreByMask32f (pSrcVecDstLine, srcOrig, vecData[2], blendMask);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in last line */
		const __m256 srcOrigRight = LoadLastLineWindowPixelLast (pSrcVecPrevLine + shortSizeX, pSrcVecCurrLine + shortSizeX, vecData);
		PartialVectorSort_4elem (vecData);
		StoreByMask32f (pSrcVecDstLine, srcOrigRight, vecData[1], blendMask);
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