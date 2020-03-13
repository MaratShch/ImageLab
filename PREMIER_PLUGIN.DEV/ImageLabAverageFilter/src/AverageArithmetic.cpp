#include "ImageLabAverageFilter.h"

bool average_filter_BGRA4444_8u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	      csSDK_uint32* __restrict dstPix,
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
	csSDK_uint32 accB, accG, accR;
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

			accB = accG = accR = 0;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;

					accB += ((srcPix[inPix] & 0x000000FFu));
					accG += ((srcPix[inPix] & 0x0000FF00u) >> 8);
					accR += ((srcPix[inPix] & 0x00FF0000u) >> 16);
				}
			}

			if (smallWindowSize == windowSize)
			{
				newB = div_by9(accB);
				newG = div_by9(accG);
				newR = div_by9(accR);
			}
			else
			{
				newB = div_by25(accB);
				newG = div_by25(accG);
				newR = div_by25(accR);
			}

			outPix = j * linePitch + i;

			dstPix[outPix] = (srcPix[outPix] & 0xFF000000u) | /* keep Alpha channel */
				                               (newR << 16) |
				                               (newG << 8)  |
				                                newB;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_BGRA4444_16u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	      csSDK_uint32* __restrict dstPix,
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
	csSDK_uint32 accB, accG, accR;
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

			accB = accG = accR = 0;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 1;
					inPix = jIdx + iIdx;

					accB += ((srcPix[inPix]  & 0x00000FFFFu));
					accG += ((srcPix[inPix]  & 0xFFFF0000u) >> 16);
					accR += ((srcPix[inPix+1]& 0x0000FFFFu));
				}
			}

			if (smallWindowSize == windowSize)
			{
				newB = div_by9(accB);
				newG = div_by9(accG);
				newR = div_by9(accR);
			}
			else
			{
				newB = div_by25(accB);
				newG = div_by25(accG);
				newR = div_by25(accR);
			}

			outPix = j * linePitch + (i << 1);

			dstPix[outPix]     = (newG << 16) | newB;
			dstPix[outPix + 1] = (srcPix[outPix + 1] & 0xFFFF0000u) | newR;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_VUYA4444_8u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
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
	csSDK_int32 accY;
	csSDK_int32 newY;

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

			accY = 0;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;
					accY += ((srcPix[inPix] & 0x00FF0000u) >> 16);
				}
			}

			newY = (smallWindowSize == windowSize) ? div_by9(accY) : div_by25(accY);

			outPix = j * linePitch + i;

			dstPix[outPix] = (srcPix[outPix] & 0xFF00FFFFu) | /* keep Alpha channel */
		                               			(newY << 16);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_BGRA4444_32f_averageArithmetic
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

			accB = accG = accR = 0.f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 2;
					inPix = jIdx + iIdx;

					accB += (srcPix[inPix]  );
					accG += (srcPix[inPix+1]);
					accR += (srcPix[inPix+2]);
				}
			}

			if (smallWindowSize == windowSize)
			{
				newB = accB * div_on_9;
				newG = accG * div_on_9;
				newR = accR * div_on_9;
			}
			else
			{
				newB = accB * div_on_25;
				newG = accG * div_on_25;
				newR = accR * div_on_25;
			}

			outPix = j * linePitch + (i << 2);

			dstPix[outPix]     = newB;
			dstPix[outPix + 1] = newG;
			dstPix[outPix + 2] = newR;
			dstPix[outPix + 3] = srcPix[outPix + 3]; /* copy ALPHA channel from source */

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_VUYA4444_32f_averageArithmetic
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
	float accY, newY;

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

			accY = 0.f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 2;
					inPix = jIdx + iIdx;
					accY += (srcPix[inPix + 2]);
				}
			}

			newY = accY * ((smallWindowSize == windowSize) ? div_on_9 : div_on_25);

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


bool average_filter_ARGB4444_8u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
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
	csSDK_uint32 accB, accG, accR;
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

			accB = accG = accR = 0;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;

					accR += ((srcPix[inPix] & 0x0000FF00u) >> 8);
					accG += ((srcPix[inPix] & 0x00FF0000u) >> 16);
					accB += ((srcPix[inPix] & 0xFF000000u) >> 24);
				}
			}

			if (smallWindowSize == windowSize)
			{
				newR = div_by9(accR);
				newG = div_by9(accG);
				newB = div_by9(accB);
			}
			else
			{
				newR = div_by25(accR);
				newG = div_by25(accG);
				newB = div_by25(accB);
			}

			outPix = j * linePitch + i;

			dstPix[outPix] = (srcPix[outPix] & 0x000000FFu) | /* keep Alpha channel */
					(newR << 8)  |
					(newG << 16) |
					(newB << 24);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool average_filter_RGB444_10u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
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
	csSDK_uint32 accB, accG, accR;
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

			accB = accG = accR = 0;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;

					accB += ((srcPix[inPix] & 0x00000FFCu) >> 2);
					accG += ((srcPix[inPix] & 0x003FF000u) >> 12);
					accR += ((srcPix[inPix] & 0xFFC00000u) >> 22);
				}
			}

			if (smallWindowSize == windowSize)
			{
				newB = div_by9(accB);
				newG = div_by9(accG);
				newR = div_by9(accR);
			}
			else
			{
				newB = div_by25(accB);
				newG = div_by25(accG);
				newR = div_by25(accR);
			}

			outPix = j * linePitch + i;
			dstPix[outPix] = (newB << 2) | (newG << 12) | (newR << 22);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}

bool average_filter_ARGB4444_16u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
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
	csSDK_uint32 accB, accG, accR;
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

			accB = accG = accR = 0;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m)) << 1);
					inPix = jIdx + iIdx;

					accR += ((srcPix[inPix]     & 0xFFFF0000u) >> 16);
					accG += ((srcPix[inPix + 1] & 0x0000FFFFu));
					accB += ((srcPix[inPix + 1] & 0xFFFF0000u) >> 16);
				}
			}

			if (smallWindowSize == windowSize)
			{
				newR = div_by9(accR);
				newG = div_by9(accG);
				newB = div_by9(accB);
			}
			else
			{
				newR = div_by25(accR);
				newG = div_by25(accG);
				newB = div_by25(accB);
			}

			outPix = j * linePitch + (i << 1);

			dstPix[outPix] = (srcPix[outPix] & 0x0000FFFFu) | /* keep Alpha channel */
												(newR << 16);
			dstPix[outPix + 1] = newG | (newB << 16);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}

bool average_filter_ARGB4444_32f_averageArithmetic
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

			accB = accG = accR = 0.f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 2;
					inPix = jIdx + iIdx;

					accR += (srcPix[inPix + 1]);
					accG += (srcPix[inPix + 2]);
					accB += (srcPix[inPix + 3]);
				}
			}

			if (smallWindowSize == windowSize)
			{
				newB = accB * div_on_9;
				newG = accG * div_on_9;
				newR = accR * div_on_9;
			}
			else
			{
				newB = accB * div_on_25;
				newG = accG * div_on_25;
				newR = accR * div_on_25;
			}

			outPix = j * linePitch + (i << 2);

			dstPix[outPix]     = srcPix[outPix];  /* copy ALPHA channel from source */
			dstPix[outPix + 1] = newR;
			dstPix[outPix + 2] = newG;
			dstPix[outPix + 3] = newB;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}
