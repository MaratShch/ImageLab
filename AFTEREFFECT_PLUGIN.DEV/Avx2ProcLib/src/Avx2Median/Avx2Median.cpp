#include "Avx2MedianInternal.hpp"
#include "Avx2MedianScalar.hpp"






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

	const __m256i& lumaMaskVector = _mm256_setr_epi32
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
		const __m256i srcOrigLeft = MedianLoad3x3::LoadWindowLeft (pSrcVecCurrLine, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], lumaMaskVector);
		pSrcVecDstLine++;

		/* process first line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = MedianLoad3x3::LoadWindow (pSrcVecCurrLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrig, vecData[4], lumaMaskVector);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in first line */
		const __m256i srcOrigRight = MedianLoad3x3::LoadWindowRight (pSrcVecCurrLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecNextLine + shortSizeX, vecData);
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
		const __m256i srcOrigLeft = MedianLoad3x3::LoadWindowLeft (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], lumaMaskVector);
		pSrcVecDstLine++;

		/* process line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = MedianLoad3x3::LoadWindow (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrig, vecData[4], lumaMaskVector);
			pSrcVecDstLine++;
		} 

		/* process rigth frame edge in last line */
		const __m256i srcOrigRight = MedianLoad3x3::LoadWindowRight (pSrcVecPrevLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecNextLine + shortSizeX, vecData);
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
		const __m256i srcOrigLeft = MedianLoad3x3::LoadWindowLeft (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecCurrLine, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigLeft, vecData[4], lumaMaskVector);
		pSrcVecDstLine++;

		/* process first line */
		for (i = pixelsInVector; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = MedianLoad3x3::LoadWindow (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecCurrLine + i, vecData);
			MedianSort::PartialSort_9_elem_8u (vecData);
			MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrig, vecData[4], lumaMaskVector);
			pSrcVecDstLine++;
		}

		/* process rigth frame edge in last line */
		const __m256i srcOrigRight = MedianLoad3x3::LoadWindowRight (pSrcVecPrevLine + shortSizeX, pSrcVecCurrLine + shortSizeX, pSrcVecCurrLine + shortSizeX, vecData);
		MedianSort::PartialSort_9_elem_8u (vecData);
		MedianStore::StoreByMask8u (pSrcVecDstLine, srcOrigRight, vecData[4], lumaMaskVector);

		/* process rest of pixels (non vectorized) if the sizeX isn't aligned to AVX2 vector size */
		{
		}
	}


	return true;
}
