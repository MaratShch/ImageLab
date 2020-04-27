#include "ImageLabAnisotropicDiffusion.h"


csSDK_int32 selectProcessFunction (const VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	filterParamsH	       paramsH = nullptr;
	constexpr long         siteVersion = 1l;
	csSDK_int32            errCode = fsBadFormatIndex;
	bool                   processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*	  PPixSuite = nullptr;
		const SPErr err = SPBasic->AcquireSuite(strPpixSuite, siteVersion, (const void**)&PPixSuite);

		if (nullptr != PPixSuite && kSPNoError == err)
		{
			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			// Get the frame dimensions
			prRect box = {};
			((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

			// Calculate dimensions
			const csSDK_int32 height = box.bottom - box.top;
			const csSDK_int32 width = box.right - box.left;
			const csSDK_int32 linePitch = (((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination)) >> 2;

			// Check is frame dimensions are correct
			if (0 >= height || 0 >= width || 0 >= linePitch || linePitch < width)
				return fsBadFormatIndex;

			void* __restrict srcImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
			void* __restrict dstImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
			if (nullptr == srcImg || nullptr == dstImg)
				return fsBadFormatIndex;

			paramsH = reinterpret_cast<filterParamsH>((*theData)->specsHandle);
			if (nullptr == paramsH)
				return fsBadFormatIndex;

			AlgMemStorage algTmpStorage = getAlgStorageStruct();
			const size_t requiredMem = width * height * sizeof(float);
			if (requiredMem > algTmpStorage.memBytesSize)
			{
				if (false == algMemStorageRealloc(width, height, algTmpStorage))
					return fsBadFormatIndex;

				setAlgStorageStruct(algTmpStorage);
			}

			const float dispersion = static_cast<float>((*paramsH)->sliderLevelDispersion);
			const float timeStep   = static_cast<float>((*paramsH)->sliderTimeStep) / 10.0f;
			const float noiseLevel = static_cast<float>((*paramsH)->sliderNoiseLevel);


			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

				    process_VUYA_4444_8u_buffer (pSrcPix, &algTmpStorage, pDstPix, width, height,
												 linePitch, dispersion, timeStep, noiseLevel);
				}
				break;

				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				{
					const float* __restrict pSrcPix = reinterpret_cast<const float* __restrict>(srcImg);
					      float* __restrict pDstPix = reinterpret_cast<float* __restrict>(dstImg);

					process_VUYA_4444_32f_buffer (pSrcPix, &algTmpStorage, pDstPix, width, height,
						                          linePitch, dispersion, timeStep, noiseLevel);
				}
				break;

				case PrPixelFormat_BGRA_4444_8u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					process_BGRA_4444_8u_buffer (pSrcPix, &algTmpStorage, pDstPix, width, height,
						                         linePitch, dispersion, timeStep, noiseLevel);
				}
				break;

				case PrPixelFormat_BGRA_4444_32f:
				{
					const float* __restrict pSrcPix = reinterpret_cast<const float* __restrict>(srcImg);
					      float* __restrict pDstPix = reinterpret_cast<float* __restrict>(dstImg);

					process_BGRA_4444_32f_buffer (pSrcPix, &algTmpStorage, pDstPix, width, height,
						                          linePitch, dispersion, timeStep, noiseLevel);
				}
				break;

				case PrPixelFormat_ARGB_4444_8u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					process_ARGB_4444_8u_buffer (pSrcPix, &algTmpStorage, pDstPix, width, height,
						                         linePitch, dispersion, timeStep, noiseLevel);
				}

				case PrPixelFormat_ARGB_4444_32f:
				{
					const float* __restrict pSrcPix = reinterpret_cast<const float* __restrict>(srcImg);
					      float* __restrict pDstPix = reinterpret_cast<float* __restrict>(dstImg);

					process_ARGB_4444_32f_buffer (pSrcPix, &algTmpStorage, pDstPix, width, height,
						                          linePitch, dispersion, timeStep, noiseLevel);
				}

				default:
					processSucceed = false;
				break;
			}

			SPBasic->ReleaseSuite (strPpixSuite, siteVersion);
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

				IMAGE_LAB_FILTER_PARAM_HANDLE_INIT(paramsH);
				(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
			}
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
