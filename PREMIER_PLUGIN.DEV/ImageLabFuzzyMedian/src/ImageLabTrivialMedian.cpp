#include "ImageLabFuzzyMedian.h"

static inline void SortU8 (int& a, int& b)
{
	int d = a - b;
	int m = ~(d >> 8);
	b += d&m;
	a -= d&m;
}

static inline void SortU8 (int* a)
{
	/* bitonic sort */
	SortU8(a[1], a[2]); SortU8(a[4], a[5]); SortU8(a[7], a[8]);
	SortU8(a[0], a[1]); SortU8(a[3], a[4]); SortU8(a[6], a[7]);
	SortU8(a[1], a[2]); SortU8(a[4], a[5]); SortU8(a[7], a[8]);
	SortU8(a[0], a[3]); SortU8(a[5], a[8]); SortU8(a[4], a[7]);
	SortU8(a[3], a[6]); SortU8(a[1], a[4]); SortU8(a[2], a[5]);
	SortU8(a[4], a[7]); SortU8(a[4], a[2]); SortU8(a[6], a[4]);
	SortU8(a[4], a[2]);
}


bool median_filter_3x3_BGRA_4444_8u_frame
(
	const	csSDK_uint32* __restrict srcBuf,
	        csSDK_uint32* __restrict dstBuf,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch
)
{
	CACHE_ALIGN csSDK_int32 kWindow[3][9] = { 0 }; /* 0 = B, 1 = G, 2 = R */
	csSDK_int32 i, j, k, l, m, idx;
	constexpr csSDK_int32 kernelRadius = 1;
	constexpr csSDK_int32 medianElement = 4;
	const csSDK_int32 jIdxMax = height - kernelRadius;
	const csSDK_int32 iIdxMax = width - kernelRadius;
	csSDK_int32 medianR, medianG, medianB;
	csSDK_int32 lineIdx, pixIdx;

	idx = m = 0;
	medianR = medianG = medianB = 0;
	lineIdx = pixIdx = 0;

	for (j = 0; j < height; j++)
	{
	__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			m = 0;
			/* collect pixels from Kernel Window */
			for (l = j - kernelRadius; l <= (j + kernelRadius); l++)
			{
				lineIdx = MIN(jIdxMax, MAX(0, l));
				for (k = i - kernelRadius; k <= (i + kernelRadius); k++)
				{
					pixIdx = MIN(iIdxMax, MAX(0, k));
					idx = lineIdx * linePitch + pixIdx;

					kWindow[0][m] = static_cast<csSDK_int32> (srcBuf[idx] & 0x000000FFu);
					kWindow[1][m] = static_cast<csSDK_int32>((srcBuf[idx] & 0x0000FF00u) >> 8);
					kWindow[2][m] = static_cast<csSDK_int32>((srcBuf[idx] & 0x00FF0000u) >> 16);

					m++;
				} /* for (k = i - kernelRadius; k <= kernelRadius; k++) */
			} /* for (l = j - kernelRadius; l <= kernelRadius; l++) */

//			gnomesort(&kWindow[0][0], &kWindow[0][9]);
//			gnomesort(&kWindow[1][0], &kWindow[1][9]);
//			gnomesort(&kWindow[2][0], &kWindow[2][9]);

			SortU8 (kWindow[0]);
			SortU8 (kWindow[1]);
			SortU8 (kWindow[2]);

			medianB = kWindow[0][medianElement];
			medianG = kWindow[1][medianElement];
			medianR = kWindow[2][medianElement];

			idx = j * linePitch + i;
			dstBuf[idx] = medianB | (medianG << 8) | (medianR << 16) | (srcBuf[idx] & 0xFF000000u); /* copy ALPHA values from source buffer */

		}/* width */

	} /* height */

	return true;
}