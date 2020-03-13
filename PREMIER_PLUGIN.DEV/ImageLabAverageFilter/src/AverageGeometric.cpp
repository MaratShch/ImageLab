#include "ImageLabAverageFilter.h"

bool average_filter_BGRA4444_8u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_uint32 R, G, B;
	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float accB, accG, accR;
	csSDK_uint32 newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accB = accG = accR = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;

					B = ((srcPix[inPix] & 0x000000FFu));
					G = ((srcPix[inPix] & 0x0000FF00u) >> 8);
					R = ((srcPix[inPix] & 0x00FF0000u) >> 16);

					accB += fLog10Tbl[B];
					accG += fLog10Tbl[G];
					accR += fLog10Tbl[R];
				}
			}

			if (smallWindowSize == windowSize)
			{
				accB *= div_on_9; /* accX = accX / 9 */
				accG *= div_on_9;
				accR *= div_on_9;
			}
			else
			{
				accB *= div_on_25; /* accX = accX / 25 */
				accG *= div_on_25;
				accR *= div_on_25;
			}

			const double powB = fast_pow(10.0, accB);
			const double powG = fast_pow(10.0, accG);
			const double powR = fast_pow(10.0, accR);

			newB = static_cast<csSDK_uint32>(powB) - 1;
			newG = static_cast<csSDK_uint32>(powG) - 1;
			newR = static_cast<csSDK_uint32>(powR) - 1;

			outPix = j * linePitch + i;

			dstPix[outPix] = (srcPix[outPix] & 0xFF000000u) | /* keep Alpha channel */
											   (newR << 16) |
				                               (newG << 8)  |
				                                newB;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_BGRA4444_16u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_uint32 R, G, B;
	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float accB, accG, accR;
	csSDK_uint32 newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accB = accG = accR = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 1;
					inPix = jIdx + iIdx;

					B = ((srcPix[inPix] & 0x00000FFFFu));
					G = ((srcPix[inPix] & 0xFFFF0000u) >> 16);
					R = ((srcPix[inPix + 1] & 0x0000FFFFu));

					accB += fLog10Tbl[B];
					accG += fLog10Tbl[G];
					accR += fLog10Tbl[R];
				}
			}

			if (smallWindowSize == windowSize)
			{
				accB *= div_on_9; /* accX = accX / 9 */
				accG *= div_on_9;
				accR *= div_on_9;
			}
			else
			{
				accB *= div_on_25; /* accX = accX / 25 */
				accG *= div_on_25;
				accR *= div_on_25;
			}

			const double powB = fast_pow(10.0, accB);
			const double powG = fast_pow(10.0, accG);
			const double powR = fast_pow(10.0, accR);

			newB = static_cast<csSDK_uint32>(powB) - 1;
			newG = static_cast<csSDK_uint32>(powG) - 1;
			newR = static_cast<csSDK_uint32>(powR) - 1;

			outPix = j * linePitch + (i << 1);

			dstPix[outPix] = (newG << 16) | newB;
			dstPix[outPix + 1] = (srcPix[outPix + 1] & 0xFFFF0000u) | newR;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_VUYA4444_8u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 Y, U, V;
	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	csSDK_int32 newY;
	csSDK_uint32 inPix, outPix;
	
	float accY;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accY = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;

					Y = ((srcPix[inPix] & 0x00FF0000u) >> 16);

					accY += fLog10Tbl[Y];
				}
			}

			accY *= ((smallWindowSize == windowSize) ? div_on_9 : div_on_25);
			const double powR = fast_pow(10.0, accY);
			newY = static_cast<csSDK_uint32>(powR) - 1;

			outPix = j * linePitch + i;

			dstPix[outPix] = (srcPix[outPix] & 0xFF00FFFFu) | /* keep Alpha and V, U channels */
												(newY << 16);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_BGRA4444_32f_averageGeometric
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float R, G, B;
	float accB, accG, accR;
	float newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accB = accG = accR = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 2;
					inPix = jIdx + iIdx;

					B = (srcPix[inPix]);
					G = (srcPix[inPix + 1]);
					R = (srcPix[inPix + 2]);

					accB += fast_log10f(B + 1.0f);
					accG += fast_log10f(G + 1.0f);
					accR += fast_log10f(R + 1.0f);
				}
			}

			if (smallWindowSize == windowSize)
			{
				accB *= div_on_9; /* accX = accX / 9 */
				accG *= div_on_9;
				accR *= div_on_9;
			}
			else
			{
				accB *= div_on_25; /* accX = accX / 25 */
				accG *= div_on_25;
				accR *= div_on_25;
			}

			const double powB = fast_pow(10.0, accB);
			const double powG = fast_pow(10.0, accG);
			const double powR = fast_pow(10.0, accR);

			newB = static_cast<float>(powB) - 1.0f;
			newG = static_cast<float>(powG) - 1.0f;
			newR = static_cast<float>(powR) - 1.0f;

			outPix = j * linePitch + (i << 2);

			dstPix[outPix]     = newB;
			dstPix[outPix + 1] = newG;
			dstPix[outPix + 2] = newR;
			dstPix[outPix + 3] = srcPix[outPix + 3];

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_VUYA4444_32f_averageGeometric
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float Y, accY, newY;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accY = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 2;
					inPix = jIdx + iIdx;
					Y = (srcPix[inPix + 2]);
					accY += fast_log10f(Y + 1.0f);
				}
			}

			accY *= ((smallWindowSize == windowSize) ? div_on_9 : div_on_25);

			const double powY = fast_pow(10.0, accY);
			newY = static_cast<float>(powY) - 1.0f;

			outPix = j * linePitch + (i << 2);

			/* copy ALPHA, U and V channels from source buffer */
			dstPix[outPix]     = srcPix[outPix];
			dstPix[outPix + 1] = srcPix[outPix + 1];
			dstPix[outPix + 2] = newY;
			dstPix[outPix + 3] = srcPix[outPix + 3];

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_ARGB4444_8u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_uint32 R, G, B;
	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float accB, accG, accR;
	csSDK_uint32 newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accB = accG = accR = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;

					R = ((srcPix[inPix] & 0x0000FF00u) >> 8);
					G = ((srcPix[inPix] & 0x00FF0000u) >> 16);
					B = ((srcPix[inPix] & 0xFF000000u) >> 24);

					accR += fLog10Tbl[R];
					accG += fLog10Tbl[G];
					accB += fLog10Tbl[B];
				}
			}

			if (smallWindowSize == windowSize)
			{
				accR *= div_on_9; /* accX = accX / 9 */
				accG *= div_on_9;
				accB *= div_on_9;
			}
			else
			{
				accR *= div_on_25; /* accX = accX / 25 */
				accG *= div_on_25;
				accB *= div_on_25;
			}

			const double powR = fast_pow(10.0, accR);
			const double powG = fast_pow(10.0, accG);
			const double powB = fast_pow(10.0, accB);

			newR = static_cast<csSDK_uint32>(powR) - 1;
			newG = static_cast<csSDK_uint32>(powG) - 1;
			newB = static_cast<csSDK_uint32>(powB) - 1;

			outPix = j * linePitch + i;

			dstPix[outPix] = (srcPix[outPix] & 0x000000FFu) | /* keep Alpha channel */
					(newR << 8)  |
					(newG << 16) |
					(newB << 24);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_RGB444_10u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_uint32 R, G, B;
	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float accB, accG, accR;
	csSDK_uint32 newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accB = accG = accR = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;

					B = ((srcPix[inPix] & 0x00000FFCu) >> 2);
					G = ((srcPix[inPix] & 0x003FF000u) >> 12);
					R = ((srcPix[inPix] & 0xFFC00000u) >> 22);

					accB += fLog10Tbl[B];
					accG += fLog10Tbl[G];
					accR += fLog10Tbl[R];
				}
			}

			if (smallWindowSize == windowSize)
			{
				accB *= div_on_9;
				accG *= div_on_9;
				accR *= div_on_9; /* accX = accX / 9 */
			}
			else
			{
				accB *= div_on_25;
				accG *= div_on_25;
				accR *= div_on_25; /* accX = accX / 25 */
			}

			const double powB = fast_pow(10.0, accB);
			const double powG = fast_pow(10.0, accG);
			const double powR = fast_pow(10.0, accR);

			newB = static_cast<csSDK_uint32>(powB) - 1;
			newG = static_cast<csSDK_uint32>(powG) - 1;
			newR = static_cast<csSDK_uint32>(powR) - 1;

			outPix = j * linePitch + i;
			dstPix[outPix] = (newB << 2) | (newG << 12) | (newR << 22);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_ARGB4444_16u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_uint32 R, G, B;
	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float accB, accG, accR;
	csSDK_uint32 newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accB = accG = accR = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m)) << 1);
					inPix = jIdx + iIdx;

					R = ((srcPix[inPix]     & 0xFFFF0000u) >> 16);
					G = ((srcPix[inPix + 1] & 0x0000FFFFu));
					B = ((srcPix[inPix + 1] & 0xFFFF0000u) >> 16);

					accR += fLog10Tbl[R];
					accG += fLog10Tbl[G];
					accB += fLog10Tbl[B];
				}
			}

			if (smallWindowSize == windowSize)
			{
				accR *= div_on_9; /* accX = accX / 9 */
				accG *= div_on_9;
				accB *= div_on_9;
			}
			else
			{
				accR *= div_on_25; /* accX = accX / 25 */
				accG *= div_on_25;
				accB *= div_on_25;
			}

			const double powR = fast_pow(10.0, accR);
			const double powG = fast_pow(10.0, accG);
			const double powB = fast_pow(10.0, accB);

			newR = static_cast<csSDK_uint32>(powR) - 1;
			newG = static_cast<csSDK_uint32>(powG) - 1;
			newB = static_cast<csSDK_uint32>(powB) - 1;

			outPix = j * linePitch + (i << 1);

			dstPix[outPix] = (srcPix[outPix] & 0x0000FFFFu) | /* keep Alpha channel */
												(newR << 16);
			dstPix[outPix + 1] = newG | (newB << 16);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}

bool average_filter_ARGB4444_32f_averageGeometric
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float R, G, B;
	float accB, accG, accR;
	float newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accB = accG = accR = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 2;
					inPix = jIdx + iIdx;

					R = (srcPix[inPix + 1]);
					G = (srcPix[inPix + 2]);
					B = (srcPix[inPix + 3]);

					accB += fast_log10f(B + 1.0f);
					accG += fast_log10f(G + 1.0f);
					accR += fast_log10f(R + 1.0f);
				}
			}

			if (smallWindowSize == windowSize)
			{
				accR *= div_on_9; /* accX = accX / 9 */
				accG *= div_on_9;
				accB *= div_on_9;
			}
			else
			{
				accR *= div_on_25; /* accX = accX / 25 */
				accG *= div_on_25;
				accB *= div_on_25;
			}

			const double powR = fast_pow(10.0, accR);
			const double powG = fast_pow(10.0, accG);
			const double powB = fast_pow(10.0, accB);

			newR = static_cast<float>(powR) - 1.0f;
			newG = static_cast<float>(powG) - 1.0f;
			newB = static_cast<float>(powB) - 1.0f;

			outPix = j * linePitch + (i << 2);

			dstPix[outPix]     = srcPix[outPix];
			dstPix[outPix + 1] = newR;
			dstPix[outPix + 2] = newG;
			dstPix[outPix + 3] = newB;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}
