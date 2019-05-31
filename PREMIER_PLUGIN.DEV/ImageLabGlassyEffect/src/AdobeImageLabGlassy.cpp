#include "ImageLabGlassyEffect.h"


// Bilateral-RGB filter entry point
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

			}
		}
		break;

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
		break;

		case fsSetup:
		break;

		case fsExecute:
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