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
}; /* namespace MedianSort */


namespace MedianLoad
{
	inline void LoadLineFromLeft (const __m256i* pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = elemLine[1] = _mm256_loadu_si256(pSrc);
		elemLine[2] = _mm256_loadu_si256(pSrc + 1);
	}

	inline void LoadLine (const __m256i* pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(pSrc - 1);
		elemLine[1] = _mm256_loadu_si256(pSrc    );
		elemLine[2] = _mm256_loadu_si256(pSrc + 1);
	}

	inline void LoadLineFromRigth (const __m256i* pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(pSrc - 1);
		elemLine[1] = elemLine[2] = _mm256_loadu_si256(pSrc);
	}

	inline const __m256i LoadWindowFromLeft (const __m256i* pPrev, const __m256i* pCurr, const __m256i* pNext, __m256i elem[9]) noexcept
	{
		LoadLineFromLeft(pPrev, elem);
		LoadLineFromLeft(pCurr, elem + 3);
		LoadLineFromLeft(pNext, elem + 6);
		return elem[4]; /* return current element from source */
	}

	inline const __m256i LoadWindow (const __m256i* pPrev, const __m256i* pCurr, const __m256i* pNext, __m256i elem[9]) noexcept
	{
		LoadLine (pPrev, elem);
		LoadLine (pCurr, elem + 3);
		LoadLine (pNext, elem + 6);
		return elem[4]; /* return current element from source */
	}

	inline const __m256i LoadWindowFromRight (const __m256i* pPrev, const __m256i* pCurr, const __m256i* pNext, __m256i elem[9]) noexcept
	{
		LoadLineFromRigth (pPrev, elem);
		LoadLineFromRigth (pCurr, elem + 3);
		LoadLineFromRigth (pNext, elem + 6);
		return elem[4]; /* return current element from source */
	}

}; /* namespace MedianLoad */

namespace MedianStore
{
	inline void StoreByMask8u (__m256i* __restrict pDst, const __m256i& valueOrig, const __m256i& valueMedian, const __m256i& storeMask) noexcept
	{
		const __m256i storeVal = _mm256_blendv_epi8(valueOrig, valueMedian, storeMask);
		_mm256_storeu_si256(pDst, storeVal);
	}

}; /* namespace MedianStore  */


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
	const PF_Pixel_BGRA_8u* __restrict pInImage,
	      PF_Pixel_BGRA_8u* __restrict pOutImage,
	      A_long sizeX,
	      A_long sizeY,
	      A_long linePitch
) noexcept
{
	A_long x, y;

	for (y = 0; y < sizeY; y++)
	{
	//	const PF_Pixel_BGRA_8u* p 
	}

	return true;
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
	const PF_Pixel_VUYA_8u* __restrict pInImage,
	      PF_Pixel_VUYA_8u* __restrict pOutImage,
	      A_long sizeY,
	      A_long sizeX,
 	      A_long linePitch
) noexcept
{
	if (sizeY < 3 || sizeX < 48)
		return false;

	constexpr size_t pixelsInVector{ sizeof(__m256i) / sizeof(PF_Pixel_VUYA_8u) };
	constexpr int lumaMask{ 0x00FF0000 }; /* vuYa */

	A_long i = 0, j = 0;
	const A_long shortSizeY = sizeY - 1; 
	const A_long vectorLoadsInLine = static_cast<A_long>(sizeX) / pixelsInVector;
	const A_long shortVectorLoadsInLine = vectorLoadsInLine - 1;
	const A_long vectorizedPixels  = vectorLoadsInLine * pixelsInVector;
	const A_long lastPixelsInLine  = sizeX - vectorizedPixels;

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
		const __m256i* pSrcVecCurrLine = reinterpret_cast<const __m256i*>(pInImage);
		const __m256i* pSrcVecNextLine = reinterpret_cast<const __m256i*>(pInImage + linePitch);
		      __m256i* pSrcVecDstLine  = reinterpret_cast<__m256i*>(pOutImage);

		/* process left frame edge in first line */
		const __m256i srcOrigLeft = MedianLoad::LoadWindowFromLeft (pSrcVecCurrLine, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], lumaMaskVector);

		/* process first line */
		for (i = 1; i < shortVectorLoadsInLine; i++)
		{
			const __m256i srcOrig = MedianLoad::LoadWindow (pSrcVecCurrLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine + i, srcOrig, vecData[4], lumaMaskVector);
		}

		/* process rigth frame edge in first line */
		const __m256i srcOrigRight = MedianLoad::LoadWindowFromRight (pSrcVecCurrLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine + shortVectorLoadsInLine, srcOrigRight, vecData[4], lumaMaskVector);

		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		{

		}
	}

	/* PROCESS LINES IN FRAME FROM 1 to SIZEY-1 */
	for (j = 1; j < shortSizeY; j++)
	{
		const __m256i* pSrcVecPrevLine = reinterpret_cast<const __m256i*>(pInImage + (j - 1) * linePitch);
		const __m256i* pSrcVecCurrLine = reinterpret_cast<const __m256i*>(pInImage +  j      * linePitch);
		const __m256i* pSrcVecNextLine = reinterpret_cast<const __m256i*>(pInImage + (j + 1) * linePitch);
		      __m256i* pSrcVecDstLine  = reinterpret_cast<__m256i*>(pOutImage + j * linePitch);

		/* load first vectors from previous, current and next line */
		/* process left frame edge in first line */
		const __m256i srcOrigLeft = MedianLoad::LoadWindowFromLeft (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], lumaMaskVector);

		/* process line */
		for (i = 1; i < shortVectorLoadsInLine; i++)
		{
			const __m256i srcOrig = MedianLoad::LoadWindow (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine + i, srcOrig, vecData[4], lumaMaskVector);
		} 

		/* process rigth frame edge in last line */
		const __m256i srcOrigRight = MedianLoad::LoadWindowFromRight (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine + shortVectorLoadsInLine, srcOrigRight, vecData[4], lumaMaskVector);

		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		{
		}

	} /* END: process frame lines from 1 to sizeY-1 */

	  /* PROCESS LAST FRAME LINE */
	{
		const __m256i* pSrcVecPrevLine = reinterpret_cast<const __m256i*>(pInImage + (j - 1) * linePitch);
		const __m256i* pSrcVecCurrLine = reinterpret_cast<const __m256i*>(pInImage + j       * linePitch);
		      __m256i* pSrcVecDstLine  = reinterpret_cast<__m256i*>(pOutImage + j * linePitch);

		/* process left frame edge in last line */
		const __m256i srcOrigLeft = MedianLoad::LoadWindowFromLeft (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecCurrLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], lumaMaskVector);

		/* process first line */
		for (i = 1; i < shortVectorLoadsInLine; i++)
		{
			const __m256i srcOrig = MedianLoad::LoadWindow (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecCurrLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine + i, srcOrig, vecData[4], lumaMaskVector);
		}

		/* process rigth frame edge in last line */
		const __m256i srcOrigRight = MedianLoad::LoadWindowFromRight (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecCurrLine + i, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine + shortVectorLoadsInLine, srcOrigRight, vecData[4], lumaMaskVector);

		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		{
		}
	}


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
