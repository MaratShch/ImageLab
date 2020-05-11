#include "ImageLabColorCorrectionRGB.h"


csSDK_int32 selectProcessFunction (const VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	constexpr long         PpixSuiteVersion = 1l;
	SPBasicSuite*		   SPBasic = nullptr;
	filterParamsH		   paramsH = nullptr;
	csSDK_int32 errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*	  PPixSuite = nullptr;
		const SPErr err = SPBasic->AcquireSuite(strPpixSuite, PpixSuiteVersion, (const void**)&PPixSuite);

		if (nullptr != PPixSuite && kSPNoError == err)
		{
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

			paramsH = reinterpret_cast<filterParamsH>((*theData)->specsHandle);
			if (nullptr == paramsH)
				return fsBadFormatIndex;

			void* __restrict srcImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
			void* __restrict dstImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
			
			if (nullptr == srcImg || nullptr == dstImg || nullptr == (*paramsH))
				return fsBadFormatIndex;

			const csSDK_int16 addR = (*paramsH)->R;
			const csSDK_int16 addG = (*paramsH)->G;
			const csSDK_int16 addB = (*paramsH)->B;

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				{
					const csSDK_uint32* __restrict src = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict dst = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);
				    RGB_Correction_BGRA4444_8u(src, dst, width, height, linePitch, addR, addG, addB);
				}
				break;

				case PrPixelFormat_BGRA_4444_16u:
				{
					const csSDK_uint32* __restrict src = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict dst = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);
				    RGB_Correction_BGRA4444_16u(src, dst, width, height, linePitch, addR, addG, addB);
				}
				break;

				case PrPixelFormat_BGRA_4444_32f:
				{
					const float* __restrict src = reinterpret_cast<const float* __restrict>(srcImg);
					      float* __restrict dst = reinterpret_cast<float* __restrict>(dstImg);
	    			RGB_Correction_BGRA4444_32f(src, dst, width, height, linePitch, addR, addG, addB);
				}
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
				{
					const csSDK_uint32* __restrict src = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict dst = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);
					RGB_Correction_ARGB4444_8u(src, dst, width, height, linePitch, addR, addG, addB);
				}
				break;

				case PrPixelFormat_ARGB_4444_16u:
				{
					const csSDK_uint32* __restrict src = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict dst = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);
    				RGB_Correction_ARGB4444_16u(src, dst, width, height, linePitch, addR, addG, addB);
				}
				break;

				case PrPixelFormat_ARGB_4444_32f:
				{
					const float* __restrict src = reinterpret_cast<const float* __restrict>(srcImg);
					      float* __restrict dst = reinterpret_cast<float* __restrict>(dstImg);
				    RGB_Correction_ARGB4444_32f(src, dst, width, height, linePitch, addR, addG, addB);
				}
				break;

				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
				{
					const csSDK_uint32* __restrict src = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict dst = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);
				    RGB_Correction_VUYA4444_8u (src, dst, width, height, linePitch, addR, addG, addB, PrPixelFormat_VUYA_4444_8u_709 == pixelFormat);
				}
				break;

				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				{
					const float* __restrict src = reinterpret_cast<const float* __restrict>(srcImg);
					      float* __restrict dst = reinterpret_cast<float* __restrict>(dstImg);
 				    RGB_Correction_VUYA4444_32f(src, dst, width, height, linePitch, addR, addG, addB, PrPixelFormat_VUYA_4444_32f_709 == pixelFormat);
				}
				break;

				case PrPixelFormat_RGB_444_10u:
				{
					const csSDK_uint32* __restrict src = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict dst = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);
					RGB_Correction_RGB444_10u(src, dst, width, height, linePitch, addR, addG, addB);
				}
				break;

				default:
					processSucceed = false;
				break;
			}

			SPBasic->ReleaseSuite (strPpixSuite, PpixSuiteVersion);
			errCode = (true == processSucceed) ? fsNoErr : errCode;
		}
	}

	return errCode;
}



// ImageLabHDR filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	filterParamsH	paramsH = nullptr;
	csSDK_int32		errCode = fsNoErr;

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

				(*paramsH)->R = 0;
				(*paramsH)->G = 0;
				(*paramsH)->B = 0;
				(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
			}
		break;

		case fsSetup:
		break;

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
		break;

		case fsExecute:
			errCode = selectProcessFunction(theData);
		break;

		case fsDisposeData:
			/* dispose handle */
			(*theData)->piSuites->memFuncs->disposeHandle((*theData)->specsHandle);
			(*theData)->specsHandle = nullptr;
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
