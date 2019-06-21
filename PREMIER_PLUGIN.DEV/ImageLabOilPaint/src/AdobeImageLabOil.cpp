#include "ImageLabOilPaint.h"


static csSDK_int32 processFrame(VideoHandle theData)
{
	prRect box = { 0 };
	csSDK_int32 errCode = fsNoErr;
	int i, j, idx;
	int xIdx, yIdx;
	float randomValue1, randomValue2;

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