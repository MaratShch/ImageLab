#include "AdobeImageLabAWB.h"


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
			PrPixelFormat			pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			switch (pixelFormat)
			{
//				case PrPixelFormat_VUYA_4444_8u:
//					break;
				case PrPixelFormat_BGRA_4444_8u:
					processSucceed = procesBGRA4444_8u_slice (theData);
					break;
#if 0
				case PrPixelFormat_VUYA_4444_8u_709:
					break;
				case PrPixelFormat_ARGB_4444_8u:
					break;
				case PrPixelFormat_BGRX_4444_8u:
					break;
				case PrPixelFormat_VUYX_4444_8u:
					break;
				case PrPixelFormat_VUYX_4444_8u_709:
					break;
				case PrPixelFormat_XRGB_4444_8u:
					break;
				case PrPixelFormat_BGRP_4444_8u:
					break;
				case PrPixelFormat_VUYP_4444_8u:
					break;
				case PrPixelFormat_VUYP_4444_8u_709:
					break;
				case PrPixelFormat_PRGB_4444_8u:
					break;
				case PrPixelFormat_BGRA_4444_16u:
					break;
				case PrPixelFormat_VUYA_4444_16u:
					break;
				case PrPixelFormat_ARGB_4444_16u:
					break;
				case PrPixelFormat_BGRX_4444_16u:
					break;
				case PrPixelFormat_XRGB_4444_16u:
					break;
				case PrPixelFormat_BGRP_4444_16u:
					break;
				case PrPixelFormat_PRGB_4444_16u:
					break;
				case PrPixelFormat_RGB_444_10u:
					break;
				case PrPixelFormat_YUYV_422_8u_601:
					break;
				case PrPixelFormat_YUYV_422_8u_709:
					break;
				case PrPixelFormat_UYVY_422_8u_601:
					break;
				case PrPixelFormat_UYVY_422_8u_709:
					break;
				case PrPixelFormat_V210_422_10u_601:
					break;
				case PrPixelFormat_V210_422_10u_709:
					break;
				case PrPixelFormat_RGB_444_12u_PQ_709:
					break;
				case PrPixelFormat_RGB_444_12u_PQ_P3:
					break;
				case PrPixelFormat_RGB_444_12u_PQ_2020:
					break;
#endif
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
	csSDK_int32 errCode = fsNoErr;

	switch (selector)
	{
		case fsInitSpec:
			errCode = fsNoErr;
		break;

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
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
