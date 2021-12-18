#include "Avx2Median.hpp"
#include "Avx2MedianInternal.hpp"


inline void LoadLinePixel0 (uint32_t* __restrict pSrc, __m256i elemLine[4]) noexcept
{
	//  | X  0  0  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 3));
}

inline void LoadLinePixel1 (uint32_t* __restrict pSrc, __m256i elemLine[5]) noexcept
{
	//  | 0  X  0  0  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 3));
}

inline void LoadLinePixel2 (uint32_t* __restrict pSrc, __m256i elemLine[6]) noexcept
{
	// | 0  0  X  0  0  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	elemLine[5] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 3));
}

inline void LoadLinePixel (uint32_t* __restrict pSrc, __m256i elemLine[7]) noexcept
{
	// 0  0  0  X  0  0  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 3));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[5] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	elemLine[6] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 3));
}

inline void LoadLinePixelNminus2 (uint32_t* __restrict pSrc, __m256i elemLine[6]) noexcept
{
	// 0  0  0  X  0  0 |
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 3));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[5] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
}

inline void LoadLinePixelNminus1 (uint32_t* __restrict pSrc, __m256i elemLine[5]) noexcept
{
	// 0  0  0  X  0 |
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 3));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
}

inline void LoadLinePixelN (uint32_t* __restrict pSrc, __m256i elemLine[4]) noexcept
{
	// 0  0  0  X |
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 3));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
}


inline __m256i load_line0_pixel0
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[16]
) noexcept
{ 
	LoadLinePixel0 (pSrc, elem);
	LoadLinePixel0 (pNext1, elem + 4);
	LoadLinePixel0 (pNext2, elem + 8);
	LoadLinePixel0 (pNext3, elem + 12);
	return elem[0];
}

inline __m256i load_line0_pixel1
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[20]
) noexcept
{
	LoadLinePixel1 (pSrc,   elem);
	LoadLinePixel1 (pNext1, elem + 5);
	LoadLinePixel1 (pNext2, elem + 10);
	LoadLinePixel1 (pNext3, elem + 15);
	return elem[1];
}

inline __m256i load_line0_pixel2
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[24]
) noexcept
{
	LoadLinePixel2 (pSrc, elem);
	LoadLinePixel2 (pNext1, elem + 6);
	LoadLinePixel2 (pNext2, elem + 12);
	LoadLinePixel2 (pNext3, elem + 18);
	return elem[2];
}

inline __m256i load_line0_pixel
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[28]
) noexcept
{
	LoadLinePixel (pSrc, elem);
	LoadLinePixel (pNext1, elem + 7);
	LoadLinePixel (pNext2, elem + 14);
	LoadLinePixel (pNext3, elem + 21);
	return elem[3];
}

inline __m256i load_line0_pixel_n2
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[24]
) noexcept
{
	LoadLinePixelNminus2 (pSrc, elem);
	LoadLinePixelNminus2 (pNext1, elem + 6);
	LoadLinePixelNminus2 (pNext2, elem + 12);
	LoadLinePixelNminus2 (pNext3, elem + 18);
	return elem[3];
}

inline __m256i load_line0_pixel_n1
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[20]
) noexcept
{
	LoadLinePixelNminus1 (pSrc, elem);
	LoadLinePixelNminus1 (pNext1, elem + 5);
	LoadLinePixelNminus1 (pNext2, elem + 10);
	LoadLinePixelNminus1 (pNext3, elem + 15);
	return elem[3];
}

inline __m256i load_line0_pixel_n
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[16]
) noexcept
{
	LoadLinePixelN (pSrc, elem);
	LoadLinePixelN (pNext1, elem + 4);
	LoadLinePixelN (pNext2, elem + 8);
	LoadLinePixelN (pNext3, elem + 12);
	return elem[3];
}

inline __m256i load_line1_pixel0
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[20]
) noexcept
{
	LoadLinePixel0 (pPrev,  elem);
	LoadLinePixel0 (pSrc,   elem + 4);
	LoadLinePixel0 (pNext1, elem + 8);
	LoadLinePixel0 (pNext2, elem + 12);
	LoadLinePixel0 (pNext3, elem + 16);
	return elem[7];
}

inline __m256i load_line1_pixel1
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[25]
) noexcept
{
	LoadLinePixel1 (pPrev,  elem);
	LoadLinePixel1 (pSrc,   elem + 5);
	LoadLinePixel1 (pNext1, elem + 10);
	LoadLinePixel1 (pNext2, elem + 15);
	LoadLinePixel1 (pNext3, elem + 20);
	return elem[8];
}

inline __m256i load_line1_pixel2
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[30]
) noexcept
{
	LoadLinePixel2 (pPrev,  elem);
	LoadLinePixel2 (pSrc,   elem + 6);
	LoadLinePixel2 (pNext1, elem + 12);
	LoadLinePixel2 (pNext2, elem + 18);
	LoadLinePixel2 (pNext3, elem + 24);
	return elem[9];
}

inline __m256i load_line1_pixel
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[35]
) noexcept
{
	LoadLinePixel (pPrev,  elem);
	LoadLinePixel (pSrc,   elem + 7); 
	LoadLinePixel (pNext1, elem + 14);
	LoadLinePixel (pNext2, elem + 21);
	LoadLinePixel (pNext3, elem + 28);
	return elem[10];
}

inline __m256i load_line1_pixel_n2
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[30]
)
{
	LoadLinePixelNminus2 (pPrev,  elem);
	LoadLinePixelNminus2 (pSrc,   elem + 6);
	LoadLinePixelNminus2 (pNext1, elem + 12);
	LoadLinePixelNminus2 (pNext2, elem + 18);
	LoadLinePixelNminus2 (pNext3, elem + 24);
	return elem[9];
}

inline void PartialSort_16_elem_8u (__m256i a[16]) noexcept
{
	/* median elemnet in index 7 */
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[3],  a[2]);
	VectorSort8uPacked (a[4],  a[5]);
	VectorSort8uPacked (a[7],  a[6]);
	VectorSort8uPacked (a[8],  a[9]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[12], a[13]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[0],  a[2]);
	VectorSort8uPacked (a[6],  a[4]);
	VectorSort8uPacked (a[8],  a[10]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[1],  a[3]);
	VectorSort8uPacked (a[7],  a[5]);
	VectorSort8uPacked (a[9],  a[11]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[2],  a[3]);
	VectorSort8uPacked (a[5],  a[4]);
	VectorSort8uPacked (a[7],  a[6]);
	VectorSort8uPacked (a[8],  a[9]);
	VectorSort8uPacked (a[10], a[11]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[0],  a[4]);
	VectorSort8uPacked (a[12], a[8]);
	VectorSort8uPacked (a[1],  a[5]);
	VectorSort8uPacked (a[13], a[9]);
	VectorSort8uPacked (a[2],  a[6]);
	VectorSort8uPacked (a[14], a[10]);
	VectorSort8uPacked (a[3],  a[7]);
	VectorSort8uPacked (a[15], a[11]);
	VectorSort8uPacked (a[0],  a[2]);
	VectorSort8uPacked (a[4],  a[6]);
	VectorSort8uPacked (a[10], a[8]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[1],  a[3]);
	VectorSort8uPacked (a[5],  a[7]);
	VectorSort8uPacked (a[11], a[9]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[2],  a[3]);
	VectorSort8uPacked (a[4],  a[5]);
	VectorSort8uPacked (a[6],  a[7]);
	VectorSort8uPacked (a[9],  a[8]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[0],  a[8]);
	VectorSort8uPacked (a[1],  a[9]);
	VectorSort8uPacked (a[2],  a[10]);
	VectorSort8uPacked (a[3],  a[11]);
	VectorSort8uPacked (a[4],  a[12]);
	VectorSort8uPacked (a[5],  a[13]);
	VectorSort8uPacked (a[6],  a[14]);
	VectorSort8uPacked (a[7],  a[15]);
	VectorSort8uPacked (a[0],  a[4]);
	VectorSort8uPacked (a[1],  a[5]);
	VectorSort8uPacked (a[2],  a[6]);
	VectorSort8uPacked (a[3],  a[7]);
	VectorSort8uPacked (a[4],  a[6]);
	VectorSort8uPacked (a[5],  a[7]);
	VectorSort8uPacked (a[6],  a[7]);
}

inline void PartialSort_20_elem_8u (__m256i a[20]) noexcept
{
	/* median element in index 9 */
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[3],  a[2]);
	VectorSort8uPacked (a[4],  a[5]);
	VectorSort8uPacked (a[7],  a[6]);
	VectorSort8uPacked (a[8],  a[9]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[12], a[13]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[16], a[17]);
	VectorSort8uPacked (a[19], a[18]);
	VectorSort8uPacked (a[0],  a[2]);
	VectorSort8uPacked (a[1],  a[3]);
	VectorSort8uPacked (a[6],  a[4]);
	VectorSort8uPacked (a[7],  a[5]);
	VectorSort8uPacked (a[8],  a[10]);
	VectorSort8uPacked (a[9],  a[11]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[16], a[18]);
	VectorSort8uPacked (a[17], a[19]);
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[2],  a[3]);
	VectorSort8uPacked (a[5],  a[4]);
	VectorSort8uPacked (a[7],  a[6]);
	VectorSort8uPacked (a[8],  a[9]);
	VectorSort8uPacked (a[10], a[11]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[16], a[17]);
	VectorSort8uPacked (a[18], a[19]);
	VectorSort8uPacked (a[0],  a[4]);
	VectorSort8uPacked (a[14], a[10]);
	VectorSort8uPacked (a[1],  a[5]);
	VectorSort8uPacked (a[15], a[11]);
	VectorSort8uPacked (a[2],  a[6]);
	VectorSort8uPacked (a[16], a[12]);
	VectorSort8uPacked (a[3],  a[7]);
	VectorSort8uPacked (a[17], a[13]);
	VectorSort8uPacked (a[4],  a[8]);
	VectorSort8uPacked (a[18], a[14]);
	VectorSort8uPacked (a[5],  a[9]);
	VectorSort8uPacked (a[19], a[15]);
	VectorSort8uPacked (a[0],  a[2]);
	VectorSort8uPacked (a[4],  a[6]);
	VectorSort8uPacked (a[13], a[11]);
	VectorSort8uPacked (a[17], a[15]);
	VectorSort8uPacked (a[1],  a[3]);
	VectorSort8uPacked (a[5],  a[7]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[18], a[16]);
	VectorSort8uPacked (a[2],  a[4]);
	VectorSort8uPacked (a[6],  a[8]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[19], a[17]);
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[2],  a[3]);
	VectorSort8uPacked (a[4],  a[5]);
	VectorSort8uPacked (a[6],  a[7]);
	VectorSort8uPacked (a[8],  a[9]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[17], a[16]);
	VectorSort8uPacked (a[19], a[18]);
	VectorSort8uPacked (a[0],  a[9]);
	VectorSort8uPacked (a[1],  a[10]);
	VectorSort8uPacked (a[2],  a[11]);
	VectorSort8uPacked (a[3],  a[12]);
	VectorSort8uPacked (a[4],  a[13]);
	VectorSort8uPacked (a[5],  a[14]);
	VectorSort8uPacked (a[6],  a[15]);
	VectorSort8uPacked (a[7],  a[16]);
	VectorSort8uPacked (a[8],  a[17]);
	VectorSort8uPacked (a[9],  a[18]);
	VectorSort8uPacked (a[10], a[19]);
	VectorSort8uPacked (a[0],  a[4]);
	VectorSort8uPacked (a[1],  a[5]);
	VectorSort8uPacked (a[2],  a[6]);
	VectorSort8uPacked (a[3],  a[7]);
	VectorSort8uPacked (a[4],  a[8]);
	VectorSort8uPacked (a[5],  a[9]);
	VectorSort8uPacked (a[4],  a[6]);
	VectorSort8uPacked (a[5],  a[7]);
	VectorSort8uPacked (a[6],  a[8]);
	VectorSort8uPacked (a[8],  a[9]);
}

inline void PartialSort_24_elem_8u (__m256i a[20]) noexcept
{

}

inline void PartialSort_28_elem_8u (__m256i a[28]) noexcept
{

}

inline void PartialSort_30_elem_8u (__m256i a[30] ) noexcept
{
}

inline void PartialSort_35_elem_8u(__m256i a[35]) noexcept
{

}

/*
	make median filter with kernel 5x5 from packed format - BGRA444_8u by AVX2 instructions set:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

	LSB                            MSB
	+-------------------------------+
	| B | G | R | A | B | G | R | A | ...
	+-------------------------------+

*/
bool AVX2::Median::median_filter_7x7_BGRA_4444_8u
(
	uint32_t* __restrict pInImage,
	uint32_t* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	const A_long& chanelMask /* 0x00FFFFFF <- BGRa */
) noexcept
{
//	if (sizeY < 7 || sizeX < 40)
//		return Scalar::scalar_median_filter_7x7_BGRA_4444_8u(pInImage, pOutImage, sizeY, sizeX, linePitch);

	constexpr A_long startOffset = 3;
	constexpr A_long pixelsInVector{ static_cast<A_long>(sizeof(__m256i) / sizeof(uint32_t)) };
	constexpr A_long startPosition{ pixelsInVector * startOffset };

	A_long i, j;
	const A_long vectorLoadsInLine = sizeX / pixelsInVector;
	const A_long vectorizedLineSize = vectorLoadsInLine * pixelsInVector;
	const A_long lastPixelsInLine = sizeX - vectorizedLineSize;
	const A_long lastIdx = lastPixelsInLine - startOffset;

	const A_long shortSizeY { sizeY - startOffset };
	const A_long shortSizeX { sizeX - pixelsInVector * 3 };
	const A_long shortSizeX2{ sizeX - pixelsInVector * 2};
	const A_long shortSizeX3{ sizeX - pixelsInVector };

	const __m256i rgbMaskVector = _mm256_setr_epi32
	(
		chanelMask, /* mask A component for 1 pixel */
		chanelMask, /* mas234Zk A component for 2 pixel */
		chanelMask, /* mask A component for 3 pixel */
		chanelMask, /* mask A component for 4 pixel */
		chanelMask, /* mask A component for 5 pixel */
		chanelMask, /* mask A component for 6 pixel */
		chanelMask, /* mask A component for 7 pixel */
		chanelMask  /* mask A component for 8 pixel */
	);

#ifdef _DEBUG
	__m256i vecData[49]{};
#else
	__m256i vecData[49];
#endif

	/* PROCESS FIRST LINE IN FRAME */
	{
		uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage);
		uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch);
		uint32_t* __restrict pSrcVecNextLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 2);
		uint32_t* __restrict pSrcVecNextLine3 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 3);
		__m256i*  __restrict pSrcVecDstLine   = reinterpret_cast<__m256i*  __restrict>(pOutImage);

		/* process pixel 0 */
		const __m256i pix0 = load_line0_pixel0 (pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, pSrcVecNextLine3, vecData);
		PartialSort_16_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix0, vecData[8], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 1 */
		const __m256i pix1 = load_line0_pixel1 (pSrcVecCurrLine  + pixelsInVector, 
			                                    pSrcVecNextLine1 + pixelsInVector,
			                                    pSrcVecNextLine2 + pixelsInVector,
			                                    pSrcVecNextLine3 + pixelsInVector,
			                                    vecData);
		PartialSort_20_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix1, vecData[10], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 2 */
		const __m256i pix2 = load_line0_pixel2 (pSrcVecCurrLine  + pixelsInVector * 2,
			                                    pSrcVecNextLine1 + pixelsInVector * 2,
			                                    pSrcVecNextLine2 + pixelsInVector * 2,
			                                    pSrcVecNextLine3 + pixelsInVector * 2,
			                                    vecData);
		PartialSort_24_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix2, vecData[12], rgbMaskVector);
		pSrcVecDstLine++;

		/* process rest of pixesl */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i pix = load_line0_pixel (pSrcVecCurrLine  + i,
				                                  pSrcVecNextLine1 + i,
				                                  pSrcVecNextLine2 + i,
				                                  pSrcVecNextLine3 + i,
				                                  vecData);
			PartialSort_28_elem_8u (vecData);
			StoreByMask8u(pSrcVecDstLine, pix2, vecData[14], rgbMaskVector);
			pSrcVecDstLine++;
		}
		
		/* process pixel N - 2 */
		const __m256i pixn2 = load_line0_pixel_n2 (pSrcVecCurrLine  + i, 
			                                       pSrcVecNextLine1 + i,
			                                       pSrcVecNextLine2 + i,
			                                       pSrcVecNextLine3 + i,
			                                       vecData);
		PartialSort_24_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn2, vecData[12], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel N - 2 */
		const __m256i pixn1 = load_line0_pixel_n1 (pSrcVecCurrLine  + i + pixelsInVector,
			                                       pSrcVecNextLine1 + i + pixelsInVector,
			                                       pSrcVecNextLine2 + i + pixelsInVector,
			                                       pSrcVecNextLine3 + i + pixelsInVector,
			                                       vecData);
		PartialSort_20_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn1, vecData[10], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel N */
		const __m256i pixn = load_line0_pixel_n1 (pSrcVecCurrLine  + i + pixelsInVector * 2,
			                                      pSrcVecNextLine1 + i + pixelsInVector * 2,
			                                      pSrcVecNextLine2 + i + pixelsInVector * 2,
			                                      pSrcVecNextLine3 + i + pixelsInVector * 2,
			                                      vecData);
		PartialSort_16_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn, vecData[8], rgbMaskVector);
	}
	
	/* PROCESS SECOND LINE IN FRAME */
	{
		uint32_t* __restrict pSrcVecPrevLine  = reinterpret_cast<uint32_t* __restrict>(pInImage);
		uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch);
		uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 2);
		uint32_t* __restrict pSrcVecNextLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 3);
		uint32_t* __restrict pSrcVecNextLine3 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 4);
		__m256i*  __restrict pSrcVecDstLine = reinterpret_cast<__m256i*  __restrict>(pOutImage);

		/* process pixel 0 */
		const __m256i pix0 = load_line1_pixel0 (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, pSrcVecNextLine3, vecData);
		PartialSort_16_elem_8u(vecData);
		StoreByMask8u(pSrcVecDstLine, pix0, vecData[8], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 1 */
		const __m256i pix1 = load_line1_pixel1 (pSrcVecPrevLine  + pixelsInVector,
			                                    pSrcVecCurrLine  + pixelsInVector,
			                                    pSrcVecNextLine1 + pixelsInVector,
			                                    pSrcVecNextLine2 + pixelsInVector,
			                                    pSrcVecNextLine3 + pixelsInVector,
			                                    vecData);
		PartialSort_20_elem_8u(vecData);
		StoreByMask8u(pSrcVecDstLine, pix1, vecData[10], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 2 */
		const __m256i pix2 = load_line1_pixel2 (pSrcVecPrevLine  + pixelsInVector * 2,
			                                    pSrcVecCurrLine  + pixelsInVector * 2,
			                                    pSrcVecNextLine1 + pixelsInVector * 2,
			                                    pSrcVecNextLine2 + pixelsInVector * 2,
			                                    pSrcVecNextLine3 + pixelsInVector * 2,
			                                    vecData);
		PartialSort_30_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix2, vecData[15], rgbMaskVector);
		pSrcVecDstLine++;

		/* process rest of pixesl */
		/* process rest of pixesl */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i pix = load_line1_pixel (pSrcVecPrevLine  + i,
				                                  pSrcVecCurrLine  + i,
				                                  pSrcVecNextLine1 + i,
				                                  pSrcVecNextLine2 + i,
				                                  pSrcVecNextLine3 + i,
				                                  vecData);
			PartialSort_35_elem_8u (vecData);
			StoreByMask8u(pSrcVecDstLine, pix, vecData[17], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process pixel N - 2 */
		const __m256i pixn2 = load_line1_pixel_n2 (pSrcVecPrevLine  + i,
			                                       pSrcVecCurrLine  + i,
			                                       pSrcVecNextLine1 + i,
			                                       pSrcVecNextLine2 + i,
			                                       pSrcVecNextLine3 + i,
			                                       vecData);
		PartialSort_30_elem_8u(vecData);
		StoreByMask8u(pSrcVecDstLine, pixn2, vecData[15], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel N - 1 */

		/* process pixel N */

	}

	/* PROCESS THIRD LINE IN FRAME */
	{
		/* process first pixel */

		/* process pixel 1 */

		/* process pixel 2 */

		/* process rest of pixesl */

		/* process pixel N - 2 */

		/* process pixel N - 1 */

		/* process pixel N */

	}

	/* PROCESS REST OF LINES IN FRAME */
	{
		/* process first pixel */

		/* process pixel 1 */

		/* process pixel 2 */

		/* process rest of pixesl */

		/* process pixel N - 2 */

		/* process pixel N - 1 */

		/* process pixel N */

	}

	/* PROCESS LINE 'N MINUS 2' IN FRAME */
	{
		/* process first pixel */

		/* process pixel 1 */

		/* process pixel 2 */

		/* process rest of pixesl */

		/* process pixel N - 2 */

		/* process pixel N - 1 */

		/* process pixel N */

	}
	
	/* PROCESS LINE 'N MINUS 1' IN FRAME */
	{
		/* process first pixel */

		/* process pixel 1 */

		/* process pixel 2 */

		/* process rest of pixesl */

		/* process pixel N - 2 */

		/* process pixel N - 1 */

		/* process pixel N */

	}
	
	/* PROCESS LAST LINE IN FRAME */
	{
		/* process first pixel */

		/* process pixel 1 */

		/* process pixel 2 */

		/* process rest of pixesl */

		/* process pixel N - 2 */

		/* process pixel N - 1 */

		/* process pixel N */

	}

	return true;
}