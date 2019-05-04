#include "ImageLabBilateral.h"
#include <windows.h>

static bool CreateParallelJob(const unsigned int& numCpu)
{
	return true;
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
			break;

		case fsCacheOnLoad:
			break;

		default:
			// unhandled case
			break;

	}

	return errCode;
}