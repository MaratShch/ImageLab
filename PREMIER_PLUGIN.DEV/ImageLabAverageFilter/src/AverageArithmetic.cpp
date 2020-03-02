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
	csSDK_int32 iIdx, jIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	csSDK_int32 accB, accG, accR;
	csSDK_int32 newB, newG, newR;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{

			accB = accG = accR = 0;

			jMin = j - winHalfSize;
			jMax = j + winHalfSize;



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

		} /* for (i = 0; i < width; i++) */
	} /* for (j = 0; j < height; j++) */

	return true;
}


#if 0
bool median_filter_ARGB_4444_8u_frame(const VideoHandle theData, const csSDK_int32& kernelWidth)
{
// protection on kernel size
if (MaxKernelWidth < kernelWidth || 3 > kernelWidth)
return false;

CACHE_ALIGN uint32_t windowR[MaxKernelElemSize];
CACHE_ALIGN uint32_t windowG[MaxKernelElemSize];
CACHE_ALIGN uint32_t windowB[MaxKernelElemSize];

prRect box = { 0 };

// Get the frame dimensions
((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

// Calculate dimensions
const csSDK_int32 height = box.bottom - box.top;
const csSDK_int32 width = box.right - box.left;
const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
const csSDK_int32 linePitch = rowbytes >> 2;

// Create copies of pointer to the source, destination frames
const csSDK_uint32* __restrict srcPix = reinterpret_cast<const csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));


const csSDK_int32 border = kernelWidth >> 1;
const csSDK_int32 medianPosition = ((kernelWidth * kernelWidth) >> 1) + 1;
const csSDK_int32 lastLine = height - border;
const csSDK_int32 lastPixel = width - border;

csSDK_int32 idxPix = 0;
csSDK_int32 i, j, k, l, m;

const size_t lineBytesSize = width * sizeof(dstPix[0]);

// copy border lines in top of frame
__VECTOR_ALIGNED__
for (j = 0; j < border; j++)
{
	idxPix = j * linePitch;
	memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
}

// perform Median Filter on frame
__VECTOR_ALIGNED__
for (j = border; j < lastLine; j++)
{
// copy border pixels from start of line
idxPix = j * linePitch;
	
for (i = 0; i < border; i++)
	dstPix[idxPix + i] = srcPix[idxPix + i];

for (i = border; i < lastPixel; i++)
{
// Pick up window elements
const csSDK_int32 lMin = j - border;
const csSDK_int32 lMax = j + border;
const csSDK_int32 mMin = i - border;
const csSDK_int32 mMax = i + border;

k = 0;

for (l = lMin; l <= lMax; l++)
{
	idxPix = l * linePitch + mMin;

	for (m = mMin; m <= mMax; m++)
	{
		windowR[k] = (srcPix[idxPix] & 0x0000FF00u) >> 8;
		windowG[k] = (srcPix[idxPix] & 0x00FF0000u) >> 16;
		windowB[k] = (srcPix[idxPix] & 0xFF000000u) >> 24;
		k++, idxPix++;
	} /* for (m = i - border; m = i + border; i++) */

} /* for (l = j - border; l < j + border; l++) */


				  // Order elements (only half of them)
				gnomesort(windowB, windowB + k);
				gnomesort(windowG, windowG + k);
				gnomesort(windowR, windowR + k);

				idxPix = (j * linePitch + i);
				dstPix[idxPix] = (srcPix[idxPix] & 0xFFu) | /* copy ALPHA channel value as is */
					(windowR[medianPosition] << 8) |
					(windowG[medianPosition] << 16) |
					(windowB[medianPosition] << 24);


			} /* for (i = border; i < lastPixel; i++) */

			  // copy border pixels from end of line
			idxPix = j * linePitch;

			for (/* i counter already set in exit from main loop */; i < width; i++)
				dstPix[idxPix + i] = srcPix[idxPix + i];

		} /* for (j = border; j < lastLine; j++) */

		  // copy border lines in bottom of frame
		__VECTOR_ALIGNED__

			for (j = lastLine; j < height; j++)
			{
				idxPix = j * linePitch;
				memcpy(&dstPix[idxPix], &srcPix[idxPix], lineBytesSize);
			}


		return true;
}
#endif