#include "AdobeImageLabDilation.h"


csSDK_int32 selectProcessFunction(VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	csSDK_int32 errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*			PPixSuite = nullptr;
		SPBasic->AcquireSuite(strPpixSuite, 1, (const void**)&PPixSuite);

		if (nullptr != PPixSuite)
		{
			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
//					processSucceed = processSepiaBGRA4444_8u_slice (theData);
				break;

				case PrPixelFormat_VUYA_4444_8u:
//					processSucceed = processSepiaVUYA4444_8u_BT601_slice (theData);
				break;

				case PrPixelFormat_VUYA_4444_8u_709:
//					processSucceed = processSepiaVUYA4444_8u_BT709_slice(theData);
				break;

				case PrPixelFormat_BGRA_4444_16u:
//					processSucceed = processSepiaBGRA4444_16u_slice (theData);
				break;

				case PrPixelFormat_BGRA_4444_32f:
//					processSucceed = processSepiaBGRA4444_32f_slice (theData);
				break;

				case PrPixelFormat_VUYA_4444_32f:
//					processSucceed = processSepiaVUYA4444_32f_BT601_slice(theData);
				break;

				case PrPixelFormat_VUYA_4444_32f_709:
//					processSucceed = processSepiaVUYA4444_32f_BT709_slice(theData);
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
//					processSucceed = processSepiaARGB4444_8u_slice(theData);
				break;

				case PrPixelFormat_ARGB_4444_16u:
//					processSucceed = processSepiaARGB4444_16u_slice(theData);
				break;

				case PrPixelFormat_ARGB_4444_32f:
//					processSucceed = processSepiaARGB4444_32f_slice(theData);
				break;

				// =========== miscellanous formats =========================== //
				case PrPixelFormat_RGB_444_10u:
//					processSucceed = processSepiaRGB444_10u_slice(theData);
				break;

				default:
					processSucceed = false;
				break;
			}

			errCode = (true == processSucceed) ? fsNoErr : errCode;
		}
	}

	return errCode;
}

// ImageLabHDR filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	csSDK_int32		errCode = fsNoErr;

	switch (selector)
	{
		case fsInitSpec:
		break;

		case fsSetup:
		break;

		case fsExecute:
			errCode = selectProcessFunction(theData);
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
