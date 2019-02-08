#include "AdobeImageLabHDR.h"


// ImageLabHDR filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	csSDK_int32 errCode = imNoErr;

	switch (selector)
	{
		case fsInitSpec:
		break;

		case fsHasSetupDialog:
		break;

		case fsSetup:
		break;

		case fsExecute:
		break;

		case fsDisposeData:
		break;

		case fsCanHandlePAR:
		break;

		case fsGetPixelFormatsSupported:
			errCode = imageLabPixelFormatSupported(theData);
		break;

		case fsCacheOnLoad:
		break;

		default:
			// unhandled case
		break;
		
	}

	return errCode;
}