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
				accB *= 0.1111111f; /* accX = accX / 9 */
				accG *= 0.1111111f;
				accR *= 0.1111111f;
			}
			else
			{
				accB *= 0.040f; /* accX = accX / 25 */
				accG *= 0.040f;
				accR *= 0.040f;
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
				accB *= 0.1111111f; /* accX = accX / 9 */
				accG *= 0.1111111f;
				accR *= 0.1111111f;
			}
			else
			{
				accB *= 0.040f; /* accX = accX / 25 */
				accG *= 0.040f;
				accR *= 0.040f;
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
