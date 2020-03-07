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
	const csSDK_int32 lastLine = height - winHalfSize;
	const csSDK_int32 lastPix = width - winHalfSize;

	csSDK_int32 iIdx, jIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	csSDK_uint32 accB, accG, accR;
	csSDK_uint32 newB, newG, newR;

	csSDK_uint32 inPix, outPix;

	for (j = 0; j < lastLine; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		for (i = 0; i < lastPix; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accB = accG = accR = 0;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				jIdx = ((l <= 0) ? 0 : l) * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = ((m <= 0) ? 0 : m);
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

			dstPix[outPix] = (srcPix[outPix] & 0xFF000000u) |
				                               (newR << 16) |
				                               (newG << 8)  |
				                                newB;

		} /* for (i = 0; i < lastPix; i++) */
	} /* for (j = 0; j < lastLine; j++) */

	for (j = lastLine; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		for (i = lastPix; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			accB = accG = accR = 0;

		}
	}


	return true;
}
