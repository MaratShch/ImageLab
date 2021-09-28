#include "Avx2Median.hpp"


namespace MedianSort
{
	inline void VectorSort8uPacked(__m256i& a, __m256i& b) noexcept
	{
		const __m256i t = a;
		a = _mm256_min_epu8(t, b);
		b = _mm256_max_epu8(t, b);
	}

	inline void VectorSort16uPacked(__m256i& a, __m256i& b) noexcept
	{
		const __m256i t = a;
		a = _mm256_min_epu16(t, b);
		b = _mm256_max_epu16(t, b);
	}

	inline void VectorSort32fPacked(__m256& a, __m256& b) noexcept
	{
		const __m256 t = a;
		a = _mm256_min_ps(t, b);
		b = _mm256_max_ps(t, b);
	}

	inline void PartialSort_9_elem_8u (__m256i a[9]) noexcept
	{
		VectorSort8uPacked(a[1], a[2]); 
		VectorSort8uPacked(a[4], a[5]); 
		VectorSort8uPacked(a[7], a[8]);
		VectorSort8uPacked(a[0], a[1]); 
		VectorSort8uPacked(a[3], a[4]); 
		VectorSort8uPacked(a[6], a[7]);
		VectorSort8uPacked(a[1], a[2]); 
		VectorSort8uPacked(a[4], a[5]); 
		VectorSort8uPacked(a[7], a[8]);
		VectorSort8uPacked(a[0], a[3]); 
		VectorSort8uPacked(a[5], a[8]); 
		VectorSort8uPacked(a[4], a[7]);
		VectorSort8uPacked(a[3], a[6]); 
		VectorSort8uPacked(a[1], a[4]); 
		VectorSort8uPacked(a[2], a[5]);
		VectorSort8uPacked(a[4], a[7]); 
		VectorSort8uPacked(a[4], a[2]); 
		VectorSort8uPacked(a[6], a[4]);
		VectorSort8uPacked(a[4], a[2]);
	}

	inline void SortFloat (float& a, float& b) noexcept
	{
		if (a > b)
		{
			const float t{ a };
			a = b;
			b = t;
		}
	}

	inline void SortInt (int32_t& a, int32_t& b) noexcept
	{
		const int32_t d{ a - b };
		const int32_t m{ ~(d >> 8) };
		b += d & m;
		a -= d & m;
	}

	template <typename T>
	inline void ScalarSortRGB (T& x, T& y) noexcept
	{
		int32_t b1 = static_cast<int32_t>(x.B);
		int32_t b2 = static_cast<int32_t>(y.B);
		int32_t g1 = static_cast<int32_t>(x.G);
		int32_t g2 = static_cast<int32_t>(y.G);
		int32_t r1 = static_cast<int32_t>(x.R);
		int32_t r2 = static_cast<int32_t>(y.R);

		SortInt (b1, b2);
		SortInt (g1, g2);
		SortInt (r1, r2);
		
		x.B = b1;
		x.G = g1;
		x.R = r1;

		y.B = b2;
		y.G = g2;
		y.R = r2;
	}

	template <typename T>
	inline void PartialScalarSort_9_elem_RGB (T a[9]) noexcept
	{
		ScalarSortRGB(a[1], a[2]);
		ScalarSortRGB(a[4], a[5]);
		ScalarSortRGB(a[7], a[8]);
		ScalarSortRGB(a[0], a[1]);
		ScalarSortRGB(a[3], a[4]);
		ScalarSortRGB(a[6], a[7]);
		ScalarSortRGB(a[1], a[2]);
		ScalarSortRGB(a[4], a[5]);
		ScalarSortRGB(a[7], a[8]);
		ScalarSortRGB(a[0], a[3]);
		ScalarSortRGB(a[5], a[8]);
		ScalarSortRGB(a[4], a[7]);
		ScalarSortRGB(a[3], a[6]);
		ScalarSortRGB(a[1], a[4]);
		ScalarSortRGB(a[2], a[5]);
		ScalarSortRGB(a[4], a[7]);
		ScalarSortRGB(a[4], a[2]);
		ScalarSortRGB(a[6], a[4]);
		ScalarSortRGB(a[4], a[2]);
	}


}; /* namespace MedianSort */


namespace MedianLoad
{
	inline void LoadLineFromLeft_4444_8u_packed (uint32_t* pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	}

	inline void LoadLine_4444_8u_packed (uint32_t* pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
		elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc    ));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	}

	inline void LoadLineFromRigth_444_8u_packed (uint32_t* pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc-1));
		elemLine[1] = elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	}

	inline const __m256i LoadWindowLeft (uint32_t* pPrev, uint32_t* pCurr, uint32_t* pNext, __m256i elem[9]) noexcept
	{
		LoadLineFromLeft_4444_8u_packed (pPrev, elem);
		LoadLineFromLeft_4444_8u_packed (pCurr, elem + 3);
		LoadLineFromLeft_4444_8u_packed (pNext, elem + 6);
		return elem[4]; /* return current element from source */
	}

	inline const __m256i LoadWindowRight (uint32_t* pPrev, uint32_t* pCurr, uint32_t* pNext, __m256i elem[9]) noexcept
	{
		LoadLineFromRigth_444_8u_packed (pPrev, elem);
		LoadLineFromRigth_444_8u_packed (pCurr, elem + 3);
		LoadLineFromRigth_444_8u_packed (pNext, elem + 6);
		return elem[4]; /* return current element from source */
	}

	inline const __m256i LoadWindow (uint32_t* pPrev, uint32_t* pCurr, uint32_t* pNext, __m256i elem[9]) noexcept
	{
		LoadLine_4444_8u_packed (pPrev, elem);
		LoadLine_4444_8u_packed (pCurr, elem + 3);
		LoadLine_4444_8u_packed (pNext, elem + 6);
		return elem[4]; /* return current element from source */
	}

	template <typename T>
	inline void LoadLineScalarLeft_RGB_packed (T* pSrc, T elemLine[3]) noexcept
	{
		elemLine[0] = elemLine[1] = *pSrc;
		elemLine[2] = *(pSrc + 1);
	}

	template <typename T>
	inline void LoadLineScalar_RGB_packed (T* pSrc, T elemLine[3]) noexcept
	{
		elemLine[0] = *(pSrc - 1);
		elemLine[1] = *pSrc;
		elemLine[2] = *(pSrc + 1);
	}

	template <typename T>
	inline void LoadLineScalarRight_RGB_packed (T* pSrc, T elemLine[3]) noexcept
	{
		elemLine[0] = *(pSrc - 1);
		elemLine[1] = elemLine[2] = *pSrc;
	}

	template <typename T>
	inline const T LoadWindowScalarLeft_RGB(T* pPrev, T* pCurr, T* pNext, T elem[9]) noexcept
	{
		LoadLineScalarLeft_RGB_packed (pPrev, elem);
		LoadLineScalarLeft_RGB_packed (pCurr, elem + 3);
		LoadLineScalarLeft_RGB_packed (pNext, elem + 6);
		return elem[4];
	}

	template <typename T>
	inline const T LoadWindowScalar_RGB (T* pPrev, T* pCurr, T* pNext, T elem[9]) noexcept
	{
		LoadLineScalar_RGB_packed (pPrev, elem);
		LoadLineScalar_RGB_packed (pCurr, elem + 3);
		LoadLineScalar_RGB_packed (pNext, elem + 6);
		return elem[4];
	}

	template <typename T>
	inline const T LoadWindowScalarRight_RGB (T* pPrev, T* pCurr, T* pNext, T elem[9]) noexcept
	{
		LoadLineScalarRight_RGB_packed (pPrev, elem);
		LoadLineScalarRight_RGB_packed (pCurr, elem + 3);
		LoadLineScalarRight_RGB_packed (pNext, elem + 6);
		return elem[4];
	}

}; /* namespace MedianLoad */


namespace MedianStore
{
	inline void StoreByMask8u (__m256i* __restrict pDst, const __m256i& valueOrig, const __m256i& valueMedian, const __m256i& storeMask) noexcept
	{
		_mm256_storeu_si256(pDst, _mm256_blendv_epi8(valueOrig, valueMedian, storeMask));
	}

	template <typename T>
	inline void ScalarStore_RGB (T* __restrict pDst, const T& valueOrig, const T& valueMedian) noexcept
	{
		pDst->B = valueMedian.B;
		pDst->G = valueMedian.G;
		pDst->R = valueMedian.R;
		pDst->A = valueOrig.A;
	}

}; /* namespace MedianStore  */


namespace Scalar
{
	template <typename T>
	static bool scalar_median_filter_3x3_BGRA_4444_8u
	(
		T* __restrict pInImage,
		T* __restrict pOutImage,
		A_long sizeY,
		A_long sizeX,
		A_long linePitch
	) noexcept
	{
		/* input buffer to small for perform median 3x3 */
		if (sizeX < 3 || sizeY < 3)
			return false;

		return true;
	}

	template <typename T>
	static bool scalar_median_filter_3x3_BGRA_4444_8u_luma_only
	(
		T* __restrict pInImage,
		T* __restrict pOutImage,
		A_long sizeY,
		A_long sizeX,
		A_long linePitch
	) noexcept
	{
		/* input buffer to small for perform median 3x3 */
		if (sizeX < 3 || sizeY < 3)
			return false;

		return true;
	}

}; /* namespace Scalar */


namespace Internal
{
	inline __m256i Convert_bgra2yuv_8u (__m256i& a) noexcept
	{
		__m256i aLow  = _mm256_cvtepu8_epi16 (_mm256_extracti128_si256(a, 0)); /* convert 4 low BGRA pixels from uint8_t to int16_t		*/
		__m256i aHigh = _mm256_cvtepu8_epi16 (_mm256_extracti128_si256(a, 1)); /* convert 4 high BGRA pixels from uint8_t to int16_t	*/
		return{ 0 };
	}

	inline __m256i Convert_yuv2bgra_8u (__m256i& a) noexcept
	{

	}

	inline __m256i Convert_argb2yuv_8u (__m256i& a) noexcept
	{
		__m256i aLow  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a, 0));  /* convert 4 low ARGB pixels from uint8_t to int16_t		*/
		__m256i aHigh = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a, 1));  /* convert 4 high ARGB pixels from uint8_t to int16_t	*/
		return{ 0 };
	}

	inline __m256i Convert_yuv2argb_8u (__m256i& a) noexcept
	{
		return{ 0 };
	}

}; /* namespace Internal */


/*
	make median filter with kernel 3x3 from packed format - BGRA444_8u by AVX2 instructions set:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

	LSB                            MSB 
	+-------------------------------+
	| B | G | R | A | B | G | R | A | ...
	+-------------------------------+

*/
bool AVX2::Median::median_filter_3x3_BGRA_4444_8u
(
	PF_Pixel_BGRA_8u* __restrict pInImage,
	PF_Pixel_BGRA_8u* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long linePitch
) noexcept
{
	if (sizeY < 3 || sizeX < 40)
		return Scalar::scalar_median_filter_3x3_BGRA_4444_8u (pInImage, pOutImage, sizeY, sizeX, linePitch);

	CACHE_ALIGN PF_Pixel_BGRA_8u  ScalarElem[9];
	constexpr A_long pixelsInVector = static_cast<A_long>(sizeof(__m256i) / PF_Pixel_BGRA_8u_size);
	constexpr int bgrMask{ 0x00FFFFFF }; /* BGRa */

	A_long i, j;
	const A_long vectorLoadsInLine = sizeX / pixelsInVector;
	const A_long vectorizedLineSize = vectorLoadsInLine * pixelsInVector;
	const A_long lastPixelsInLine = sizeX - vectorizedLineSize;
	const A_long lastIdx = lastPixelsInLine - 1;

	const A_long shortSizeY{ sizeY - 1 };
	const A_long shortSizeX{ sizeX - pixelsInVector };

	const __m256i rgbMaskVector = _mm256_setr_epi32
	(
		bgrMask, /* mask Y component for 1 pixel */
		bgrMask, /* mask Y component for 2 pixel */
		bgrMask, /* mask Y component for 3 pixel */
		bgrMask, /* mask Y component for 4 pixel */
		bgrMask, /* mask Y component for 5 pixel */
		bgrMask, /* mask Y component for 6 pixel */
		bgrMask, /* mask Y component for 7 pixel */
		bgrMask  /* mask Y component for 8 pixel */
	);

#ifdef _DEBUG
	__m256i vecData[9]{};
#else
	__m256i vecData[9];
#endif

	/* PROCESS FIRST LINE IN FRAME (for pixels line -1 we takes pixels from current line) as VECTOR */
	{
		uint32_t* pSrcVecCurrLine = reinterpret_cast<uint32_t*>(pInImage);
		uint32_t* pSrcVecNextLine = reinterpret_cast<uint32_t*>(pInImage + linePitch);
		 __m256i* pSrcVecDstLine  = reinterpret_cast<__m256i*> (pOutImage);

		/* process left frame edge in first line */
		const __m256i srcOrigLeft = MedianLoad::LoadWindowLeft (pSrcVecCurrLine, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], rgbMaskVector);
		pSrcVecDstLine++;

		/* process first line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = MedianLoad::LoadWindow (pSrcVecCurrLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrig, vecData[4], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in first line */
		const __m256i srcOrigRight = MedianLoad::LoadWindowRight (pSrcVecCurrLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecNextLine + shortSizeX, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigRight, vecData[4], rgbMaskVector);

		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		if (0 != lastPixelsInLine)
		{
			PF_Pixel_BGRA_8u* pLastPixels     = reinterpret_cast<PF_Pixel_BGRA_8u*>(pSrcVecDstLine);
			PF_Pixel_BGRA_8u* pSrcVecCurrLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + vectorizedLineSize);
			PF_Pixel_BGRA_8u* pSrcVecNextLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + vectorizedLineSize + linePitch);
			i = 0;

			for (; i < lastIdx; i++)
			{
				const PF_Pixel_BGRA_8u currentPixel = MedianLoad::LoadWindowScalar_RGB (pSrcVecCurrLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, ScalarElem);
				MedianSort::PartialScalarSort_9_elem_RGB (ScalarElem);
				MedianStore::ScalarStore_RGB (pLastPixels + i, currentPixel, ScalarElem[4]);
			}
			const PF_Pixel_BGRA_8u currentPixel = MedianLoad::LoadWindowScalarRight_RGB (pSrcVecCurrLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, ScalarElem);
			MedianSort::PartialScalarSort_9_elem_RGB (ScalarElem);
			MedianStore::ScalarStore_RGB (pLastPixels + i, currentPixel, ScalarElem[4]);
		} /* if (0 != lastPixelsInLine) */
	}

	/* PROCESS LINES IN FRAME FROM 1 to SIZEY-1 */
	for (j = 1; j < shortSizeY; j++)
	{
		uint32_t* pSrcVecPrevLine = reinterpret_cast<uint32_t*>(pInImage + (j - 1) * linePitch);
		uint32_t* pSrcVecCurrLine = reinterpret_cast<uint32_t*>(pInImage + j       * linePitch);
		uint32_t* pSrcVecNextLine = reinterpret_cast<uint32_t*>(pInImage + (j + 1) * linePitch);
		 __m256i* pSrcVecDstLine  = reinterpret_cast<__m256i*>(pOutImage + j       * linePitch);

		/* load first vectors from previous, current and next line */
		/* process left frame edge in first line */
		const __m256i srcOrigLeft = MedianLoad::LoadWindowLeft (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], rgbMaskVector);
		pSrcVecDstLine++;

		/* process line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = MedianLoad::LoadWindow (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrig, vecData[4], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in last line */
		const __m256i srcOrigRight = MedianLoad::LoadWindowRight (pSrcVecPrevLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecNextLine + shortSizeX, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigRight, vecData[4], rgbMaskVector);

		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		if (0 != lastPixelsInLine)
		{
			PF_Pixel_BGRA_8u* pLastPixels = reinterpret_cast<PF_Pixel_BGRA_8u*>(pSrcVecDstLine);
			PF_Pixel_BGRA_8u* pSrcVecPrevLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + (j - 1) * linePitch + vectorizedLineSize);
			PF_Pixel_BGRA_8u* pSrcVecCurrLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage +  j      * linePitch + vectorizedLineSize);
			PF_Pixel_BGRA_8u* pSrcVecNextLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + (j + 1) * linePitch + vectorizedLineSize);
			i = 0;

			for (; i < lastIdx; i++)
			{
				const PF_Pixel_BGRA_8u currentPixel = MedianLoad::LoadWindowScalar_RGB (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, ScalarElem);
				MedianSort::PartialScalarSort_9_elem_RGB (ScalarElem);
				MedianStore::ScalarStore_RGB (pLastPixels + i, currentPixel, ScalarElem[4]);
			}
			const PF_Pixel_BGRA_8u currentPixel = MedianLoad::LoadWindowScalarRight_RGB (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, ScalarElem);
			MedianSort::PartialScalarSort_9_elem_RGB (ScalarElem);
			MedianStore::ScalarStore_RGB (pLastPixels + i, currentPixel, ScalarElem[4]);
		}

	} /* END: process frame lines from 1 to sizeY-1 */

	  /* PROCESS LAST FRAME LINE */
	{
		uint32_t* pSrcVecPrevLine = reinterpret_cast<uint32_t*>(pInImage + (j - 1) * linePitch);
		uint32_t* pSrcVecCurrLine = reinterpret_cast<uint32_t*>(pInImage + j      * linePitch);
	     __m256i* pSrcVecDstLine  = reinterpret_cast <__m256i*>(pOutImage + j * linePitch);

		/* process left frame edge in last line */
		const __m256i srcOrigLeft = MedianLoad::LoadWindowLeft (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecCurrLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], rgbMaskVector);
		pSrcVecDstLine++;

		/* process first line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = MedianLoad::LoadWindow (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecCurrLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrig, vecData[4], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in last line */
		const __m256i srcOrigRight = MedianLoad::LoadWindowRight (pSrcVecPrevLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecCurrLine + shortSizeX, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigRight, vecData[4], rgbMaskVector);

		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		if (0 != lastPixelsInLine)
		{
			PF_Pixel_BGRA_8u* pLastPixels = reinterpret_cast<PF_Pixel_BGRA_8u*>(pSrcVecDstLine);
			PF_Pixel_BGRA_8u* pSrcVecPrevLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + (j - 1) * linePitch + vectorizedLineSize);
			PF_Pixel_BGRA_8u* pSrcVecCurrLine = reinterpret_cast<PF_Pixel_BGRA_8u*>(pInImage + j       * linePitch + vectorizedLineSize);
			i = 0;

			for (; i < lastIdx; i++)
			{
				const PF_Pixel_BGRA_8u currentPixel = MedianLoad::LoadWindowScalar_RGB (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecCurrLine + i, ScalarElem);
				MedianSort::PartialScalarSort_9_elem_RGB (ScalarElem);
				MedianStore::ScalarStore_RGB (pLastPixels + i, currentPixel, ScalarElem[4]);
			}
			const PF_Pixel_BGRA_8u currentPixel = MedianLoad::LoadWindowScalarRight_RGB (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecCurrLine + i, ScalarElem);
			MedianSort::PartialScalarSort_9_elem_RGB (ScalarElem);
			MedianStore::ScalarStore_RGB (pLastPixels + i, currentPixel, ScalarElem[4]);
		}

	}

	return true;
}


/*
	make median filter with kernel 3x3 from packed format - BGRA444_8u by AVX2 instructions set:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

	LSB                            MSB
	+-------------------------------+
	| B | G | R | A | B | G | R | A | ...
	+-------------------------------+

*/
bool AVX2::Median::median_filter_3x3_BGRA_4444_8u_luma_only
(
	PF_Pixel_BGRA_8u* __restrict pInImage,
	PF_Pixel_BGRA_8u* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long linePitch
) noexcept
{
	if (sizeY < 3 || sizeX < 40)
		return Scalar::scalar_median_filter_3x3_BGRA_4444_8u_luma_only (pInImage, pOutImage, sizeY, sizeX, linePitch);

	CACHE_ALIGN PF_Pixel_BGRA_8u  ScalarElem[9];
	constexpr A_long pixelsInVector = static_cast<A_long>(sizeof(__m256i) / PF_Pixel_BGRA_8u_size);
	constexpr int bgrMask{ 0x00FFFFFF }; /* BGRa */

	return true;
}



/*
	make median filter with kernel 3x3 from packed format - VUYA_4444_8u by AVX2 instructions set on each color channel
	with temporary convert YUVC image to RGB on the fly:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

	+-------------------------------+
	| V | U | Y | A | V | U | Y | A | ...
	+-------------------------------+

*/
bool median_filter_3x3_VUYA_4444_8u
(
	const PF_Pixel_VUYA_8u* __restrict pInImage,
	PF_Pixel_VUYA_8u* __restrict pOutImage,
	A_long sizeX,
	A_long sizeY,
	A_long linePitch
) noexcept
{
	return false;
}

/*
	make median filter with kernel 3x3 from packed format VUYA_4444_8u by AVX2 instructions set on luminance channel only:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

	LSB                            MSB
	+-------------------------------+
	| * | * | Y | * | * | * | Y | * | ...
	+-------------------------------+

*/
bool AVX2::Median::median_filter_3x3_VUYA_4444_8u_luma_only
(
	PF_Pixel_VUYA_8u* __restrict pInImage,
	PF_Pixel_VUYA_8u* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
 	A_long linePitch
) noexcept
{
	if (sizeY < 3 || sizeX < 48)
		return false;

	constexpr A_long pixelsInVector = static_cast<A_long>(sizeof(__m256i) / PF_Pixel_VUYA_8u_size);
	constexpr int lumaMask{ 0x00FF0000 }; /* vuYa */

	A_long i, j;
	const A_long vectorLoadsInLine = sizeX / pixelsInVector;
	const A_long vectorizedLineSize  = vectorLoadsInLine * pixelsInVector;
	const A_long lastPixelsInLine  = sizeX - vectorizedLineSize;

	const A_long shortSizeY { sizeY - 1 };
	const A_long shortSizeX { sizeX - pixelsInVector };

	const __m256i lumaMaskVector = _mm256_setr_epi32
	(
		lumaMask, /* mask Y component for 1 pixel */
		lumaMask, /* mask Y component for 2 pixel */
		lumaMask, /* mask Y component for 3 pixel */
		lumaMask, /* mask Y component for 4 pixel */
		lumaMask, /* mask Y component for 5 pixel */
		lumaMask, /* mask Y component for 6 pixel */
		lumaMask, /* mask Y component for 7 pixel */
		lumaMask  /* mask Y component for 8 pixel */
	);

#ifdef _DEBUG
	__m256i vecData[9]{};
#else
	__m256i vecData[9];
#endif

	/* PROCESS FIRST LINE IN FRAME (for pixels line -1 we takes pixels from current line) as VECTOR */
	{
		uint32_t* pSrcVecCurrLine = reinterpret_cast<uint32_t*>(pInImage);
		uint32_t* pSrcVecNextLine = reinterpret_cast<uint32_t*>(pInImage + linePitch);
		 __m256i* pSrcVecDstLine  = reinterpret_cast<__m256i*> (pOutImage);

		/* process left frame edge in first line */
		const __m256i srcOrigLeft = MedianLoad::LoadWindowLeft (pSrcVecCurrLine, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], lumaMaskVector);
		pSrcVecDstLine++;

		/* process first line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = MedianLoad::LoadWindow (pSrcVecCurrLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrig, vecData[4], lumaMaskVector);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in first line */
		const __m256i srcOrigRight = MedianLoad::LoadWindowRight (pSrcVecCurrLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecNextLine + shortSizeX, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigRight, vecData[4], lumaMaskVector);

		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		{

		}
	}

	/* PROCESS LINES IN FRAME FROM 1 to SIZEY-1 */
	for (j = 1; j < shortSizeY; j++)
	{
		uint32_t* pSrcVecPrevLine = reinterpret_cast<uint32_t*>(pInImage + (j - 1) * linePitch);
		uint32_t* pSrcVecCurrLine = reinterpret_cast<uint32_t*>(pInImage +  j      * linePitch);
		uint32_t* pSrcVecNextLine = reinterpret_cast<uint32_t*>(pInImage + (j + 1) * linePitch);
		 __m256i* pSrcVecDstLine  = reinterpret_cast<__m256i*>(pOutImage +  j * linePitch);

		/* load first vectors from previous, current and next line */
		/* process left frame edge in first line */
		const __m256i srcOrigLeft = MedianLoad::LoadWindowLeft (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], lumaMaskVector);
		pSrcVecDstLine++;

		/* process line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = MedianLoad::LoadWindow (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrig, vecData[4], lumaMaskVector);
			pSrcVecDstLine++;
		} 

		/* process rigth frame edge in last line */
		const __m256i srcOrigRight = MedianLoad::LoadWindowRight (pSrcVecPrevLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecNextLine + shortSizeX, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigRight, vecData[4], lumaMaskVector);

		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		{
		}

	} /* END: process frame lines from 1 to sizeY-1 */

	  /* PROCESS LAST FRAME LINE */
	{
		uint32_t* pSrcVecPrevLine = reinterpret_cast<uint32_t*>(pInImage  + (j - 1) * linePitch);
		uint32_t* pSrcVecCurrLine = reinterpret_cast<uint32_t*>(pInImage  +  j      * linePitch);
		 __m256i* pSrcVecDstLine  = reinterpret_cast <__m256i*>(pOutImage +  j * linePitch);

		/* process left frame edge in last line */
		const __m256i srcOrigLeft = MedianLoad::LoadWindowLeft (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecCurrLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], lumaMaskVector);
		pSrcVecDstLine++;

		/* process first line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = MedianLoad::LoadWindow (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecCurrLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrig, vecData[4], lumaMaskVector);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in last line */
		const __m256i srcOrigRight = MedianLoad::LoadWindowRight (pSrcVecPrevLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecCurrLine + shortSizeX, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigRight, vecData[4], lumaMaskVector);

		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		{
		}
	}


	return true;
}
