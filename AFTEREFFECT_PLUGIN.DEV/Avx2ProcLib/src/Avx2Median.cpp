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

	inline void Sort32fPacked(__m256& a, __m256& b) noexcept
	{
		const __m256 t = a;
		a = _mm256_min_ps(t, b);
		b = _mm256_max_ps(t, b);
	}

};

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
	      A_long sizeX,
	      A_long sizeY,
 	      A_long linePitch
) noexcept
{
	if (sizeY < 3 || sizeX < 48)
		return false;

	constexpr size_t pixelsInVector = sizeof(__m256i) / sizeof(PF_Pixel_VUYA_8u);
	A_long i = 0, j = 0;
	const A_long shortSizeY = sizeY - 2; /* frame lines, except first and last */
	const A_long vectorLoadsInLine = static_cast<A_long>(sizeX) / pixelsInVector;
	const A_long vectorizedPixels  = vectorLoadsInLine * pixelsInVector;
	const A_long lastPixelsInLine  = sizeX - vectorizedPixels;

	constexpr int lumaMask = 0x00FF0000; /* vuYa */
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

	__m256i vecData[9]{};

	/* process first frame line */

	/* START: process frame lines from 1 to sizeY-1 */
	for (j = 1; j < shortSizeY; j++)
	{
		/* get pointer on current frame line */
		const __m256i* pSrcVecPrevLine = reinterpret_cast<const __m256i*>(pInImage + (j - 1) * linePitch);
		const __m256i* pSrcVecCurrLine = reinterpret_cast<const __m256i*>(pInImage +  j      * linePitch);
		const __m256i* pSrcVecNextLine = reinterpret_cast<const __m256i*>(pInImage + (j + 1) * linePitch);

		/* load first vectors from previous, current and next line */
		vecData[0] = _mm256_and_si256(_mm256_loadu_si256(pSrcVecPrevLine++), lumaMaskVector);
		vecData[3] = _mm256_and_si256(_mm256_loadu_si256(pSrcVecCurrLine++), lumaMaskVector);
		vecData[6] = _mm256_and_si256(_mm256_loadu_si256(pSrcVecNextLine++), lumaMaskVector);

		/* process line */
		for (i = 0; i < vectorizedPixels; i++)
		{

		} 

		for (i = 0; i < lastPixelsInLine; i++)
		{

		} /* for (i = 0; i < lastPixelsInLine; i++) */

	} /* END: process frame lines from 1 to sizeY-1 */


	/* process last frame line */

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
