#include "Avx2Histogram.hpp"

/* 
	make histogram from packed format - BGRA444_8u by AVX2 instructions set:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:
	
	lsb                             msb
	+-------------------------------+
	| B | G | R | A | B | G | R | A | ...
	+-------------------------------+
	
*/
/* AVX2 optimizations */
void AVX2::Histogram::make_histogram_BGRA4444_8u
(
	const PF_Pixel_BGRA_8u* __restrict pImage,
	HistBin*  __restrict  pFinalHistogramR,
	HistBin*  __restrict  pFinalHistogramG,
	HistBin*  __restrict  pFinalHistogramB,
	A_long histBufSizeBytes,
	A_long sizeX,
	A_long sizeY,
	A_long linePitch
) noexcept
{
	CACHE_ALIGN HistBin histR[8][256]{};
	CACHE_ALIGN HistBin histG[8][256]{};
	CACHE_ALIGN HistBin histB[8][256]{};

	constexpr A_long pixSize = static_cast<A_long>(PF_Pixel_BGRA_8u_size);
	constexpr A_long loadElems = Avx2BytesSize / pixSize;
	const A_long loopCnt   = sizeX / loadElems;
	const A_long loopFract = sizeX - loadElems * loopCnt;

	A_long x, y, z;

	for (y = 0; y < sizeY; y++)
	{
		const __m256i* __restrict pBufVector = reinterpret_cast<const __m256i* __restrict>(pImage + y * linePitch);

		/* AVX2 vector part */
		for (x = 0; x < loopCnt; x++)
		{
			/* non-aligned load 8 packet pixels at once */
			const __m256i packetSrcPix = _mm256_loadu_si256(pBufVector);

			/* 1 pixel extract */
			histB[0][_mm256_extract_epi8(packetSrcPix, 0) ]++; /* B */
			histG[0][_mm256_extract_epi8(packetSrcPix, 1) ]++; /* G */
			histR[0][_mm256_extract_epi8(packetSrcPix, 2) ]++; /* R */
			/* skip alpha */

			/* 2 pixel extract */
			histB[1][_mm256_extract_epi8(packetSrcPix, 4) ]++; /* B */
			histG[1][_mm256_extract_epi8(packetSrcPix, 5) ]++; /* G */
			histR[1][_mm256_extract_epi8(packetSrcPix, 6) ]++; /* R */
			/* skip alpha */

			/* 3 pixel extract */
			histB[2][_mm256_extract_epi8(packetSrcPix, 8) ]++; /* B */
			histG[2][_mm256_extract_epi8(packetSrcPix, 9) ]++; /* G */
			histR[2][_mm256_extract_epi8(packetSrcPix, 10)]++; /* R */
			/* skip alpha */
			
			/* 4 pixel extract */
			histB[3][_mm256_extract_epi8(packetSrcPix, 12)]++; /* B */
			histG[3][_mm256_extract_epi8(packetSrcPix, 13)]++; /* G */
			histR[3][_mm256_extract_epi8(packetSrcPix, 14)]++; /* R */
			/* skip alpha */

			/* 5 pixel extract */
			histB[4][_mm256_extract_epi8(packetSrcPix, 16)]++; /* B */
			histG[4][_mm256_extract_epi8(packetSrcPix, 17)]++; /* G */
			histR[4][_mm256_extract_epi8(packetSrcPix, 18)]++; /* R */
			/* skip alpha */

		    /* 6 pixel extract */
			histB[5][_mm256_extract_epi8(packetSrcPix, 20)]++; /* B */
			histG[5][_mm256_extract_epi8(packetSrcPix, 21)]++; /* G */
			histR[5][_mm256_extract_epi8(packetSrcPix, 22)]++; /* R */
			/* skip alpha */

			/* 7 pixel extract */
			histB[6][_mm256_extract_epi8(packetSrcPix, 24)]++; /* B */
			histG[6][_mm256_extract_epi8(packetSrcPix, 25)]++; /* G */
			histR[6][_mm256_extract_epi8(packetSrcPix, 26)]++; /* R */
			/* skip alpha */

			/* 8 pixel extract */
			histB[7][_mm256_extract_epi8(packetSrcPix, 28)]++; /* B */
			histG[7][_mm256_extract_epi8(packetSrcPix, 29)]++; /* G */
			histR[7][_mm256_extract_epi8(packetSrcPix, 30)]++; /* R */

			pBufVector++;
		} /* x = 0; x < loopCnt; x++ */

		/* scalar part - no vectorizing for complete processing rest of pixels in end of line */
		const PF_Pixel_BGRA_8u* pBuScalar = reinterpret_cast<const PF_Pixel_BGRA_8u*>(pBufVector);
		for (z = 0; z < loopFract; z++)
		{
			histB[z][pBuScalar[z].B]++;
			histG[z][pBuScalar[z].G]++;
			histR[z][pBuScalar[z].R]++;
		} /* for (z = x; z < loopFract; z++) */

	} /* for (y = 0; y < sizeY; y++) */

    /* final path - merge all temporary histogram buffers to final histogram */
	__VECTOR_ALIGNED__
	for (x = 0; x < 256; x++)
	{
		pFinalHistogramB[x] = histB[0][x] + histB[1][x] + histB[2][x] + histB[3][x] + histB[4][x] + histB[5][x] + histB[6][x] + histB[7][x];
		pFinalHistogramG[x] = histG[0][x] + histG[1][x] + histG[2][x] + histG[3][x] + histG[4][x] + histG[5][x] + histG[6][x] + histG[7][x];
		pFinalHistogramR[x] = histR[0][x] + histR[1][x] + histR[2][x] + histR[3][x] + histR[4][x] + histR[5][x] + histR[6][x] + histR[7][x];
	}

	return;
}



/*
	make histogram from packed format - BGRA444_8u by AVX2 instructions set:

	Image buffer layout [each cell - 16 bits unsigned in range between 0...32767] :

	lsb                             msb
	+-------------------------------+
	| B | G | R | A | B | G | R | A | ...
	+-------------------------------+

*/
void AVX2::Histogram::make_histogram_BGRA4444_16u
(
	const PF_Pixel_BGRA_16u* __restrict pImage,
	HistBin*  __restrict  pFinalHistogramR,
	HistBin*  __restrict  pFinalHistogramG,
	HistBin*  __restrict  pFinalHistogramB,
	A_long histBufSizeBytes,
	A_long sizeX,
	A_long sizeY,
	A_long linePitch
) noexcept
{
	CACHE_ALIGN HistBin histR[4][32768]{};
	CACHE_ALIGN HistBin histG[4][32768]{};
	CACHE_ALIGN HistBin histB[4][32768]{};

	constexpr A_long pixSize = static_cast<A_long>(PF_Pixel_BGRA_16u_size);
	constexpr A_long loadElems = Avx2BytesSize / pixSize;
	const A_long loopCnt   = sizeX / loadElems;
	const A_long loopFract = sizeX - loadElems * loopCnt;

	A_long x, y, z;

	for (y = 0; y < sizeY; y++)
	{
		const __m256i* __restrict pBufVector = reinterpret_cast<const __m256i* __restrict>(pImage + y * linePitch);

		/* AVX2 vector part */
		for (x = 0; x < loopCnt; x++)
		{
			/* non-aligned load 8 packet pixels at once */
			const __m256i packetSrcPix = _mm256_loadu_si256(pBufVector);

			/* 1 pixel extract */
			histB[0][_mm256_extract_epi16(packetSrcPix, 0)]++; /* B */
			histG[0][_mm256_extract_epi16(packetSrcPix, 1)]++; /* G */
			histR[0][_mm256_extract_epi16(packetSrcPix, 2)]++; /* R */
			/* skip alpha */

			/* 2 pixel extract */
			histB[1][_mm256_extract_epi16(packetSrcPix, 4)]++; /* B */
			histG[1][_mm256_extract_epi16(packetSrcPix, 5)]++; /* G */
			histR[1][_mm256_extract_epi16(packetSrcPix, 6)]++; /* R */
			/* skip alpha */

			/* 3 pixel extract */
			histB[2][_mm256_extract_epi16(packetSrcPix, 8)]++; /* B */
			histG[2][_mm256_extract_epi16(packetSrcPix, 9)]++; /* G */
			histR[2][_mm256_extract_epi16(packetSrcPix, 10)]++; /* R */
			/* skip alpha */

			/* 4 pixel extract */
			histB[3][_mm256_extract_epi16(packetSrcPix, 12)]++; /* B */
			histG[3][_mm256_extract_epi16(packetSrcPix, 13)]++; /* G */
			histR[3][_mm256_extract_epi16(packetSrcPix, 14)]++; /* R */
			/* skip alpha */

			pBufVector++;
		} /* for (x = 0; x < loopCnt; x++) */

		  /* scalar part - no vectorizing for complete processing rest of pixels in end of line */
		const PF_Pixel_BGRA_16u* pBuScalar = reinterpret_cast<const PF_Pixel_BGRA_16u*>(pBufVector);
		for (z = 0; z < loopFract; z++)
		{
			histB[z][pBuScalar[z].B]++;
			histG[z][pBuScalar[z].G]++;
			histR[z][pBuScalar[z].R]++;
		} /* for (z = x; z < loopFract; z++) */
	} /* for (y = 0; y < sizeY; y++) */

    /* final path - merge all temporary histogram buffers to final histogram */
	__VECTOR_ALIGNED__
	for (x = 0; x < 32768; x++)
	{
		pFinalHistogramB[x] = histB[0][x] + histB[1][x] + histB[2][x] + histB[3][x];
		pFinalHistogramG[x] = histG[0][x] + histG[1][x] + histG[2][x] + histG[3][x];
		pFinalHistogramR[x] = histR[0][x] + histR[1][x] + histR[2][x] + histR[3][x];
	}

	return;
}


/*
	make luminance histogram from packed format - VUYA)444_8u by AVX2 instructions set:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

	lsb                             msb
	+-------------------------------+
	| V | U | Y | A | V | U | Y | A | ...
	+-------------------------------+

*/
/* AVX2 optimizations */
void AVX2::Histogram::make_luma_histogram_VUYA4444_8u
(
	const PF_Pixel_VUYA_8u* __restrict pImage,
	HistBin*  __restrict  pFinalHistogramY,
	A_long histBufSizeBytes,
	A_long sizeX,
	A_long sizeY,
	A_long linePitch
) noexcept
{
	CACHE_ALIGN HistBin histY[8][256]{};

	constexpr A_long pixSize = static_cast<A_long>(PF_Pixel_VUYA_8u_size);
	constexpr A_long loadElems = Avx2BytesSize / pixSize;
	const A_long loopCnt = sizeX / loadElems;
	const A_long loopFract = sizeX - loadElems * loopCnt;

	A_long x, y, z;

	for (y = 0; y < sizeY; y++)
	{
		const __m256i* __restrict pBufVector = reinterpret_cast<const __m256i* __restrict>(pImage + y * linePitch);

		/* AVX2 vector part */
		for (x = 0; x < loopCnt; x++)
		{
			/* non-aligned load 8 packet pixels at once */
			const __m256i packetSrcPix = _mm256_loadu_si256(pBufVector);

			/* 1 pixel extract */
			histY[0][_mm256_extract_epi8(packetSrcPix, 2) ]++; /* Y */
			
			/* 2 pixel extract */
			histY[1][_mm256_extract_epi8(packetSrcPix, 6) ]++; /* Y */
		    
			/* 3 pixel extract */
			histY[2][_mm256_extract_epi8(packetSrcPix, 10)]++; /* Y */

		    /* 4 pixel extract */
			histY[3][_mm256_extract_epi8(packetSrcPix, 14)]++; /* Y */
	
		    /* 5 pixel extract */
			histY[4][_mm256_extract_epi8(packetSrcPix, 18)]++; /* Y */

		    /* 6 pixel extract */
			histY[5][_mm256_extract_epi8(packetSrcPix, 22)]++; /* y*/

		    /* 7 pixel extract */
			histY[6][_mm256_extract_epi8(packetSrcPix, 26)]++; /* Y */

		    /* 8 pixel extract */
			histY[7][_mm256_extract_epi8(packetSrcPix, 30)]++; /* Y */

			pBufVector++;
		} /* x = 0; x < loopCnt; x++ */

		  /* scalar part - no vectorizing for complete processing rest of pixels in end of line */
		const PF_Pixel_VUYA_8u* pBuScalar = reinterpret_cast<const PF_Pixel_VUYA_8u*>(pBufVector);
		for (z = 0; z < loopFract; z++)
		{
			histY[z][pBuScalar[z].Y]++;
		} /* for (z = x; z < loopFract; z++) */

	} /* for (y = 0; y < sizeY; y++) */

	/* final path - merge all temporary histogram buffers to final histogram */
	__VECTOR_ALIGNED__
	for (x = 0; x < 256; x++)
	{
		pFinalHistogramY[x] = histY[0][x] + histY[1][x] + histY[2][x] + histY[3][x] + histY[4][x] + histY[5][x] + histY[6][x] + histY[7][x];
	}

	return;
}


void AVX2::Histogram::make_histogram_binarization
(
	const HistBin* __restrict pHistogram,
	uint8_t*       __restrict pBinHistogram,
	A_long                    histElemSize
) noexcept
{
#if 0
	constexpr __m256i  zeroVal{ 0 };
	constexpr A_long elemSize = static_cast<A_long>(sizeof(pHistogram[0]));
	constexpr A_long loadElems = Avx2BytesSize / elemSize;
	const A_long loopCnt = histElemSize / (loadElems * 2);

	const __m256i* __restrict pBufVector = reinterpret_cast<const __m256i* __restrict>(pHistogram);

	for (A_long x = 0; x < loopCnt; x++)
	{
		/* load 2 vectors 8 x 32 bits*/
		__m256i packetVal1 = _mm256_loadu_si256(pBufVector++);
		__m256i packetVal2 = _mm256_loadu_si256(pBufVector++);

		/* compare each vector with zero */
		__m256i cmpVal1 = _mm256_cmpgt_epi32(packetVal1, zeroVal);
		__m256i cmpVal2 = _mm256_cmpgt_epi32(packetVal2, zeroVal);

		/* shift each result rigth on 31 bits for make binarization */
		__m256i binVal1 = _mm256_srli_epi32(cmpVal1, 31);
		__m256i binVal2 = _mm256_srli_epi32(cmpVal2, 31);

		/* not completed yet ... */
	}
#endif
	return;
}
