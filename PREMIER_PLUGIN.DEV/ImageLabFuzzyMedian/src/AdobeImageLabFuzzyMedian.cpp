#include "ImageLabFuzzyMedian.h"
#include <windows.h>

void free_coarse (const VideoHandle& theData, AlgMemStorage& algMemStorage)
{
	char* ptr = reinterpret_cast<char*>(algMemStorage.pCoarse_addr);
	if (nullptr != ptr)
	{
		algMemStorage.pCoarse_addr = algMemStorage.pCoarse = nullptr;
		memset(ptr, 0, size_coarse + size_mem_align);
		((*theData)->piSuites->memFuncs->disposePtr)(ptr);
		ptr = nullptr;
	}
	return;
}


void free_fine (const VideoHandle& theData, AlgMemStorage& algMemStorage)
{
	char* ptr = reinterpret_cast<char*>(algMemStorage.pFine_addr);
	if (nullptr != ptr)
	{
		algMemStorage.pFine_addr = algMemStorage.pFine = nullptr;
		memset(ptr, 0, size_fine + size_mem_align);
		((*theData)->piSuites->memFuncs->disposePtr)(ptr);
		ptr = nullptr;
	}
	return;
}

csSDK_int32 allocate_coarse (const VideoHandle& theData, AlgMemStorage& algMemStorage)
{
	csSDK_int32 ret = 0;
	HistElem* ptr = nullptr;
	constexpr csSDK_uint32 totalSize = static_cast<csSDK_uint32>(size_coarse + size_mem_align);
	constexpr unsigned long long alignmend = static_cast<unsigned long long>(size_mem_align);

	// apply to SweePie memory site
	ptr = reinterpret_cast<HistElem*>(((*theData)->piSuites->memFuncs->newPtr)(totalSize));
	if (nullptr != ptr)
	{
		algMemStorage.pCoarse_addr = ptr;
		unsigned long long addr = CreateAlignment(reinterpret_cast<unsigned long long>(ptr), alignmend);
		algMemStorage.pCoarse = reinterpret_cast<HistElem* __restrict>(addr);
	}
	else
	{
		algMemStorage.pCoarse_addr = algMemStorage.pCoarse = nullptr;
		ret |= 1;
	}

	return ret;
}


csSDK_int32 allocate_fine (const VideoHandle& theData, AlgMemStorage& algMemStorage)
{
	csSDK_int32 ret = 0;
	HistElem* ptr = nullptr;
	constexpr csSDK_uint32 totalSize = static_cast<csSDK_uint32>(size_fine + size_mem_align);
	constexpr unsigned long long alignmend = static_cast<unsigned long long>(size_mem_align);

	// apply to SweePie memory site
	ptr = reinterpret_cast<HistElem* __restrict>(((*theData)->piSuites->memFuncs->newPtr)(totalSize));
	if (nullptr != ptr)
	{
		algMemStorage.pFine_addr = ptr;
		unsigned long long addr = CreateAlignment(reinterpret_cast<unsigned long long>(ptr), alignmend);
		algMemStorage.pFine = reinterpret_cast<HistElem* __restrict>(addr);
	}
	else
	{
		algMemStorage.pFine_addr = algMemStorage.pFine = nullptr;
		ret |= 1;
	}

	return ret;
}



csSDK_int32 selectProcessFunction (const VideoHandle theData, const csSDK_int8& advFlag, const csSDK_int16& kernelRadius, AlgMemStorage& algMemStorage)
{
	constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*	SPBasic = nullptr;
	csSDK_int32     errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*	PPixSuite = nullptr;
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
					processSucceed = median_filter_BGRA_4444_8u_frame (srcPix, dstPix, height, width, linePitch, algMemStorage, kernelRadius);
				}
				break;

				case PrPixelFormat_ARGB_4444_8u:
				{
					const csSDK_uint32* __restrict srcPix = reinterpret_cast<const csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
					      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
					processSucceed = median_filter_ARGB_4444_8u_frame (srcPix, dstPix, height, width, linePitch, algMemStorage, kernelRadius);
				}

				default:
					processSucceed = false;
				break;
			}

			SPBasic->ReleaseSuite(strPpixSuite, 1);
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
			if (0 == allocate_coarse(theData, (*paramsH)->AlgMemStorage) && 0 == allocate_fine(theData, (*paramsH)->AlgMemStorage))
			{
				(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
			}
			else
			{
				free_coarse(theData, (*paramsH)->AlgMemStorage);
				free_fine  (theData, (*paramsH)->AlgMemStorage);
				memset(&((*paramsH)->AlgMemStorage), 0, sizeof((*paramsH)->AlgMemStorage));
				// free handler itself
				((*theData)->piSuites->memFuncs->disposeHandle)(reinterpret_cast<char**>(paramsH));
				paramsH = nullptr;
				(*theData)->specsHandle = nullptr;
			}
		}

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
				const csSDK_int8 advFlag = ((*paramsH)->checkbox ? 1 : 0);
				if (1 == advFlag) /* use kernel radius 1 in case of Fuzzy Algorihm */ 
					(*paramsH)->kernelRadius = static_cast<csSDK_int16>(MinKernelRadius);

				const csSDK_int16 kernelRadius = ((*paramsH)->kernelRadius) | 1; // as minimal kernel radius should be 1
				errCode = selectProcessFunction (theData, advFlag, kernelRadius, (*paramsH)->AlgMemStorage);
			}
			else
				errCode = fsUnsupported;
		}
		break;

		case fsDisposeData:
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
			errCode = fsUnsupported;
		break;

	}

	return errCode;
}