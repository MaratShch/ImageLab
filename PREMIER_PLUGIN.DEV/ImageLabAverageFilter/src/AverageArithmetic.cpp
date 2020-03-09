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