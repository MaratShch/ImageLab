#include "ImageLabMonoTonal.h"


bool process_VUYA_4444_8u_frame (const VideoHandle theData)
{
	return true;
}



csSDK_int32 selectProcessFunction(const VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	csSDK_int32 errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*			PPixSuite = nullptr;
		const SPErr err = SPBasic->AcquireSuite(strPpixSuite, 1, (const void**)&PPixSuite);

		if (nullptr != PPixSuite && kSPNoError == err)
		{
			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
//			case PrPixelFormat_BGRA_4444_8u:
//				processSucceed = process_BGRA_4444_8u_frame(theData);
//				break;

			case PrPixelFormat_VUYA_4444_8u:
//			case PrPixelFormat_VUYA_4444_8u_709:
				processSucceed = process_VUYA_4444_8u_frame (theData);
			break;

//			case PrPixelFormat_BGRA_4444_16u:
//				processSucceed = process_BGRA_4444_16u_frame(theData);
//				break;

//			case PrPixelFormat_BGRA_4444_32f:
//				processSucceed = process_BGRA_4444_32f_frame(theData);
//				break;

//			case PrPixelFormat_VUYA_4444_32f:
//			case PrPixelFormat_VUYA_4444_32f_709:
//				processSucceed = process_VUYA_4444_32f_frame(theData);
//				break;

				// ============ native AE formats ============================= //
//			case PrPixelFormat_ARGB_4444_8u:
//				processSucceed = process_ARGB_4444_8u_frame(theData);
//				break;

//			case PrPixelFormat_ARGB_4444_16u:
//				processSucceed = process_ARGB_4444_16u_frame(theData);
//				break;

//			case PrPixelFormat_ARGB_4444_32f:
//				processSucceed = process_ARGB_4444_32f_frame(theData);
//				break;

				// =========== miscellanous formats =========================== //
//			case PrPixelFormat_RGB_444_10u:
//				processSucceed = process_RGB_444_10u_frame(theData);
//				break;

				default:
					processSucceed = false;
				break;
			}

			errCode = (true == processSucceed) ? fsNoErr : errCode;
		}
	}

	return errCode;
}



// Bilateral-RGB filter entry point
PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData)
{
	csSDK_int32 errCode = fsNoErr;

	switch (selector)
	{
		case fsExecute:
			errCode = selectProcessFunction (theData);
		break;

		case fsInitSpec:
		break;

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
		break;

		case fsSetup:
		break;

		case fsDisposeData:
		break;

		case fsCanHandlePAR:
			errCode = prEffectCanHandlePAR;
		break;

		case fsGetPixelFormatsSupported:
			errCode = imageLabPixelFormatSupported (theData);
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