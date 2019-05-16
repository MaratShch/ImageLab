#include "ImageLabBilateral.h"
#include <windows.h>

double* pBuffer1 = nullptr;
double* pBuffer2 = nullptr;

csSDK_int32 processFrame(VideoHandle theData)
{
	const double sigma_r = 0.100;
	const int radius = 5;

	csSDK_int32 errCode = fsNoErr;
	// execute filter
	prRect box = { 0 };
	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	DebugBreak();

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

// single thread synchronous processing
	BGRA_convert_to_CIELab(srcPix, pBuffer1, width, height, rowbytes);

	bilateral_filter_color(pBuffer1, pBuffer2, width, height, radius, sigma_r);

	CIELab_convert_to_BGRA(pBuffer2, srcPix, dstPix, width, height, rowbytes);

	return errCode;
}


// Bilateral-RGB filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	csSDK_int32 errCode = fsNoErr;

	switch (selector)
	{
		case fsInitSpec:
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