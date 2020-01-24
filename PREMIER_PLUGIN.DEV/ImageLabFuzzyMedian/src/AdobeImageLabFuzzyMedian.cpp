#include "ImageLabFuzzyMedian.h"
#include <windows.h>


csSDK_int32 selectProcessFunction (const VideoHandle theData, const bool& advFlag, const int32_t& kernelSize)
{
	constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*	SPBasic = nullptr;
	csSDK_int32     errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*	  PPixSuite = nullptr;
		const SPErr err = SPBasic->AcquireSuite(strPpixSuite, 1, (const void**)&PPixSuite);

		if (nullptr != PPixSuite && kSPNoError == err)
		{
			// Get the pixels format
			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			// Get the frame dimensions
			prRect box = {};
			((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

			// Calculate dimensions
			const csSDK_int32 height = box.bottom - box.top;
			const csSDK_int32 width  = box.right - box.left;
			const csSDK_int32 linePitch = (((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination)) >> 2;

			// Check is frame dimensions are correct
			if (0 >= height || 0 >= width || 0 >= linePitch || linePitch < width)
				return fsBadFormatIndex;

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				{
					const csSDK_uint32* __restrict srcPix = reinterpret_cast<const csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
					      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
					processSucceed = median_filter_BGRA_4444_8u_frame (srcPix, dstPix, height, width, linePitch);
				}
				break;

				case PrPixelFormat_ARGB_4444_8u:
				{
					const csSDK_uint32* __restrict srcPix = reinterpret_cast<const csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
					      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
					processSucceed = median_filter_ARGB_4444_8u_frame (srcPix, dstPix, height, width, linePitch);
				}

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
			const bool advFlag = (*paramsH)->checkbox ? true : false;

			const int32_t kernelSize = (true == advFlag) ? 3 :		// reset kernel size to 3 if used Fuzzy Algorithm 
						kernel_width((*paramsH)->kernelRadius);		// kernel size must be odd number

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