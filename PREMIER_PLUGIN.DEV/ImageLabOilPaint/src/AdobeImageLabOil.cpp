#include "ImageLabOilPaint.h"

void processDataSlice (
	const csSDK_uint32* __restrict srcImage,
	      csSDK_uint32* __restrict dstImage,
	short int*	        __restrict rHist,
	short int*	        __restrict gHist,
	short int*	        __restrict bHist,
	const int                      width,
	const int                      height,
	const int                      linePitch,
	const int                      windowSize)
{
	int i, j, k, l;

	const int shortHeight = height - windowSize;
	const int shortWidth  = width  - windowSize;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// cleanup histogramm's buffers
			__VECTOR_ALIGNED__  memset(rHist, 0, histSizeBytes);
			__VECTOR_ALIGNED__  memset(gHist, 0, histSizeBytes);
			__VECTOR_ALIGNED__  memset(bHist, 0, histSizeBytes);

			// get coordinates of first pixel in the window 
			const csSDK_uint32* startPoint = srcImage + j * linePitch + i;
			
			// save Alpha value of destination pixel
			const int alphaValue = ((*startPoint) >> 24) & 0xFFu;
			
			const int horizontalWinSize = MIN(width - i,  windowSize);
			const int verticalWinSize   = MIN(height - j, windowSize);

			// get local histogram per each color band from image window
			for (k = 0; k < verticalWinSize; k++)
			{
				for (l = 0; l < horizontalWinSize; l++)
				{
					const csSDK_uint32 pixel = *(startPoint + k * linePitch + l);
					
					const unsigned char r     =        pixel  & 0xFFu;
					const unsigned char g     = (pixel >> 8)  & 0xFFu;
					const unsigned char b     = (pixel >> 16) & 0xFFu;

					rHist[r]++;
					gHist[g]++;
					bHist[b]++;

				} // for (l = 0; l < windowSize; l++)
			} // for (k = 0; k < windowSize; k++)

			// search maximal number of pixels with same value
			short int rMaxPos = -1;
			short int gMaxPos = -1;
			short int bMaxPos = -1;

			for (k = 0; k < histSize; k++)
			{
				__VECTOR_ALIGNED__
					rMaxPos = MAX(rMaxPos, rHist[k]);
				__VECTOR_ALIGNED__
					gMaxPos = MAX(gMaxPos, gHist[k]);
				__VECTOR_ALIGNED__
					bMaxPos = MAX(bMaxPos, bHist[k]);
			} // // search maximal number of pixels with same value

			// build destination pixel;
			const csSDK_uint32 dstPixel =	alphaValue << 24 |
											bMaxPos << 16    |
											gMaxPos << 8     |
											rMaxPos;

			*dstImage++ = dstPixel;
		} // for (i = 0; i < shortWidth; i++)

		dstImage += linePitch - width;

	} // for (j = 0; j < shortHeight; j++)

	return;
}


csSDK_int32 processFrame(VideoHandle theData)
{
	CACHE_ALIGN short int rHist[histSize];
	CACHE_ALIGN short int gHist[histSize];
	CACHE_ALIGN short int bHist[histSize];

	prRect box = { 0 };
	csSDK_int32 errCode = fsNoErr;

	// Get the data from specsHandle
	const FilterParamHandle filterParamH = reinterpret_cast<FilterParamHandle>((*theData)->specsHandle);


	if (nullptr != filterParamH)
	{
		// Get the frame dimensions
		((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

		// Calculate dimensions
		const csSDK_int32 height = box.bottom - box.top;
		const csSDK_int32 width = box.right - box.left;
		const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

		const int lineSize = rowbytes >> 2;

		// Create copies of pointer to the source, destination frames
		csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
		csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

		processDataSlice(srcPix, dstPix, rHist, gHist, bHist, width, height, lineSize, 5);

	}

	return errCode;
}


// filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	csSDK_int32 errCode = fsNoErr;
	FilterParamHandle filterParamH = nullptr;

	switch (selector)
	{
		case fsInitSpec:
		{
			if ((*theData)->specsHandle)
			{

			}
			else
			{
				filterParamH = reinterpret_cast<FilterParamHandle>(((*theData)->piSuites->memFuncs->newHandle)(sizeof(FilterParamStr)));
				if (nullptr == filterParamH)
					break; // memory allocation failed

				// save the filter parameters inside of Premier handler
				(*theData)->specsHandle = reinterpret_cast<char**>(filterParamH);
			}
		}
		break;

		case fsHasSetupDialog:
				errCode = fsHasNoSetupDialog;
		break;

		case fsSetup:
		break;

		case fsExecute:
			errCode = processFrame(theData);
		break;

		case fsDisposeData:
		break;

		case fsCanHandlePAR:
			errCode = prEffectCanHandlePAR;
		break;

		case fsGetPixelFormatsSupported:
			errCode = imageLabPixelFormatSupported(theData);
		break;

		case fsCacheOnLoad:
			errCode = fsDoNotCacheOnLoad;
		break;

		default:
			// unhandled case
		break;

	}

	return errCode;
}