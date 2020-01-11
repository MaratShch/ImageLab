#include "ImageLabFuzzyMedian.h"
#include <windows.h>


csSDK_int32 selectProcessFunction (const VideoHandle theData, const bool& advFlag, const int32_t& kernelSize)
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
				case PrPixelFormat_BGRA_4444_8u:
					median_filter_BGRA_4444_8u_frame (theData, kernelSize);
				break;

				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
					median_filter_VUYA_4444_8u_frame (theData, kernelSize);
				break;

				case PrPixelFormat_BGRA_4444_16u:
					median_filter_BGRA_4444_16u_frame(theData, kernelSize);
				break;

				case PrPixelFormat_BGRA_4444_32f:
				break;

				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
				break;

				case PrPixelFormat_ARGB_4444_16u:
				break;

				case PrPixelFormat_ARGB_4444_32f:
				break;

				// =========== miscellanous formats =========================== //
				case PrPixelFormat_RGB_444_10u:
				break;

				// =========== Packed uncompressed formats ==================== //
				case PrPixelFormat_YUYV_422_8u_601:
				case PrPixelFormat_YUYV_422_8u_709:
				break;

				case PrPixelFormat_UYVY_422_8u_601:
				case PrPixelFormat_UYVY_422_8u_709:
				break;

				case PrPixelFormat_UYVY_422_32f_601:
				case PrPixelFormat_UYVY_422_32f_709:
				break;

				case PrPixelFormat_V210_422_10u_601:
				case PrPixelFormat_V210_422_10u_709:
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


// Bilateral-RGB filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	filterParamsH	paramsH = nullptr;
	csSDK_int32 errCode = fsNoErr;

	switch (selector)
	{
	case fsInitSpec:
		if ((*theData)->specsHandle)
		{
			// In a filter that has a need for a more complex setup dialog
			// you would present your platform specific user interface here,
			// storing results in the specsHandle (which you've allocated).
		}
		else
		{
			paramsH = reinterpret_cast<filterParamsH>(((*theData)->piSuites->memFuncs->newHandle)(sizeof(filterParams)));

			// Memory allocation failed, no need to continue
			if (nullptr == paramsH)
				break;

			IMAGE_LAB_MEDIAN_FILTER_PARAM_HANDLE_INIT(paramsH);
			(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
		}

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
		break;

		case fsSetup:
		break;

		case fsExecute:
		{
			// Get the data from specsHandle
			paramsH = (filterParamsH)(*theData)->specsHandle;
			const bool advFlag = (nullptr != paramsH ? ((*paramsH)->checkbox ? true : false) : false);
			const int32_t kernelSize = (true == advFlag) ? 3 :			// reset kernel size to 3 if used Fuzzy Algorithm 
				(static_cast<int32_t>((*paramsH)->kernelSize | 0x1u));	// kernel size must be odd number
			errCode = selectProcessFunction (theData, advFlag, kernelSize);
		}
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