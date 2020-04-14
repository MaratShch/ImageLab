#include <mutex>
#include <windows.h>
#include "ImageLabFuzzyMedian.h"


inline bool checkAlgMemoryStorage (const csSDK_int32& width, const csSDK_int32& height, const AlgMemStorage& algMemStorage)
{
	const csSDK_int32 imageBuffer = width * height * size_fuzzy_pixel;
	const csSDK_int32 totalMemory = CreateAlignment(MAX(imageBuffer, size_total_hist_buffers), CPU_PAGE_SIZE);
	return (nullptr == algMemStorage.pFuzzyBuffer || totalMemory < algMemStorage.memSize) ? false : true;
}

void algMemStorageFree (AlgMemStorage& algMemStorage)
{
	if (nullptr != algMemStorage.pFuzzyBuffer)
	{
		_aligned_free(algMemStorage.pFuzzyBuffer);
		algMemStorage.pFuzzyBuffer = nullptr;
		algMemStorage.memSize = 0;
		algMemStorage.pFine = algMemStorage.pCoarse = nullptr;
		algMemStorage.pH = nullptr;
		algMemStorage.stripeNum = algMemStorage.stripeSize = 0;
	}
	return;
}

/* realloc memory storage for perform Fuzzy Median Filter or Histogramm Based Median Filter */
bool algMemStorageRealloc(const csSDK_int32& width, const csSDK_int32& height, AlgMemStorage& algMemStorage)
{
	const csSDK_int32 imageBuffer = width * height * size_fuzzy_pixel + CACHE_LINE * 2; /* add 128 bytes spare */
	const csSDK_int32 totalMemory = CreateAlignment(MAX(imageBuffer, size_total_hist_buffers + CACHE_LINE * 2), CPU_PAGE_SIZE);
	bool bRet = false;

	/* free previoulsy allocated memory */
	algMemStorageFree (algMemStorage);

	if (nullptr == algMemStorage.pFuzzyBuffer)
	{
		void* pMem = _aligned_malloc(totalMemory, CPU_PAGE_SIZE);
		if (nullptr != pMem)
		{
			/* build Memory layout for Fuzzy Median Algorithm */
#ifdef _DEBUG
			__VECTOR_ALIGNED__
			memset(pMem, 0, totalMemory);
#endif
			algMemStorage.memSize = totalMemory;
			algMemStorage.pFuzzyBuffer = pMem;

			/* build Memory Layout for Histogram Based Median Filter (add 64 bytes between Coarse and Fine and between Histogram and Coarse for DBG putpose) */
			const unsigned long long pFineAddress = reinterpret_cast<const unsigned long long>(pMem);
			const unsigned long long pCoarseAddress = CreateAlignment(pFineAddress + size_fine + CACHE_LINE, static_cast<unsigned long long>(CACHE_LINE));
			const unsigned long long pHistogramm = CreateAlignment(pCoarseAddress  + size_hist_obj + CACHE_LINE, static_cast<unsigned long long>(CACHE_LINE));

			algMemStorage.pFine   = reinterpret_cast<HistElem* __restrict>(pFineAddress);
			algMemStorage.pCoarse = reinterpret_cast<HistElem* __restrict>(pCoarseAddress);
			algMemStorage.pH = reinterpret_cast<HistogramObj* __restrict>(pHistogramm);

			bRet = true;
		}
	}

	return bRet;
}


csSDK_int32 selectProcessFunction (const VideoHandle theData, const csSDK_int16 advFlag, const csSDK_int16 kernelRadius, AlgMemStorage& algMemStorage)
{
	constexpr char* strPpixSuite = "Premiere PPix Suite";
	constexpr long  lSiteVersion = 1l;
	SPBasicSuite*	SPBasic = nullptr;
	filterParamsH	paramsH = nullptr;
	csSDK_int32     errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*	PPixSuite = nullptr;
		const SPErr err = SPBasic->AcquireSuite(strPpixSuite, lSiteVersion, (const void**)&PPixSuite);

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
			const csSDK_int32 width  = box.right  - box.left;
			const csSDK_int32 linePitch = (((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination)) >> 2;

			// Check is frame dimensions are correct
			if (0 >= height || 0 >= width || 0 >= linePitch || linePitch < width)
				return fsBadFormatIndex;

			if (kernelRadius > 1 || 0 != advFlag) 
			{
				if (false == checkAlgMemoryStorage (width, height, algMemStorage))
				{
					/* required memory re-allocation */
					if (false == algMemStorageRealloc (width, height, algMemStorage))
						return fsBadFormatIndex; /* memory re-allocation failed */
					else
						setAlgStorageStruct (algMemStorage);
				}
			}

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				{
					csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
					csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
					if (0 != advFlag)
						processSucceed = fuzzy_median_filter_BGRA_4444_8u_frame(srcPix, dstPix, height, width, linePitch, algMemStorage);
					else if (1 == kernelRadius)
						processSucceed = median_filter_3x3_BGRA_4444_8u_frame(srcPix, dstPix, height, width, linePitch);
					else
						processSucceed = median_filter_BGRA_4444_8u_frame (srcPix, dstPix, height, width, linePitch, algMemStorage, kernelRadius);
				}
				break;

				case PrPixelFormat_ARGB_4444_8u:
				{
					csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
					csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

					processSucceed = (advFlag ?
						fuzzy_median_filter_ARGB_4444_8u_frame (srcPix, dstPix, height, width, linePitch, algMemStorage) :
						median_filter_ARGB_4444_8u_frame (srcPix, dstPix, height, width, linePitch, algMemStorage, kernelRadius) );
				}

				default:
					processSucceed = false;
				break;
			}

			SPBasic->ReleaseSuite(strPpixSuite, lSiteVersion);
			errCode = (true == processSucceed) ? fsNoErr : errCode;
		}
	}

	return errCode;
}


// Bilateral-RGB filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	filterParamsH	paramsH = nullptr;
	csSDK_int32		errCode = fsNoErr;

	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

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

			IMAGE_LAB_MEDIAN_FILTER_PARAM_HANDLE_INIT (paramsH);

			(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
		}
		break;

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
		break;

		case fsSetup:
		break;

		case fsExecute:
		{
			// Get the data from specsHandle
			paramsH = reinterpret_cast<filterParamsH>((*theData)->specsHandle);
			if (nullptr != paramsH)
			{
				const csSDK_int16 advFlag = ((*paramsH)->checkbox ? 1 : 0);
				const csSDK_int16 kernelRadius = make_odd((*paramsH)->kernelRadius); // kernel radius should be odd
				errCode = selectProcessFunction (theData, advFlag, kernelRadius, (*paramsH)->AlgMemStorage);
			}
			else
				errCode = fsUnsupported;
		}
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
			errCode = fsUnsupported;
		break;

	}

	return errCode;
}