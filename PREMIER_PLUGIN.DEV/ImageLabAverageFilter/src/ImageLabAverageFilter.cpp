#include "ImageLabAverageFilter.h"
#include <math.h>


void init_1og10_table(float* pTable, int table_size)
{
	for (int i = 0; i < table_size; i++)
	{
		pTable[i] = log10f(static_cast<float>(i + 1));
	}
	return;
}


float* allocate_aligned_log_table(const VideoHandle& theData, filterParamsH filtersParam)
{
	float* fAlignedAddress = nullptr;
	const int pLog10TableBytesSize = (*filtersParam)->pLog10TableSize * sizeof(float);

	if (0 != pLog10TableBytesSize)
	{
		const int totalMemSize = pLog10TableBytesSize + CACHE_LINE;
		constexpr unsigned long long alignmend = static_cast<unsigned long long>(size_mem_align);

		float* fPtr = reinterpret_cast<float*>(((*theData)->piSuites->memFuncs->newPtr)(totalMemSize));
		if (nullptr != fPtr)
		{
			unsigned long long aligned_address = CreateAlignment(reinterpret_cast<unsigned long long>(fPtr), alignmend);

			(*filtersParam)->pLog10Table = fPtr;
			(*filtersParam)->pLog10TableAligned = fAlignedAddress = reinterpret_cast<float* __restrict>(aligned_address);
		}
	}

	return fAlignedAddress;
}


csSDK_int32 selectProcessFunction (const VideoHandle theData)
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
				break;

#if 0
			case PrPixelFormat_VUYA_4444_8u:
			case PrPixelFormat_VUYA_4444_8u_709:
				processSucceed = process_VUYA_4444_8u_frame(theData);
				break;

			case PrPixelFormat_BGRA_4444_16u:
				processSucceed = process_BGRA_4444_16u_frame(theData);
				break;

			case PrPixelFormat_BGRA_4444_32f:
				processSucceed = process_BGRA_4444_32f_frame(theData);
				break;

			case PrPixelFormat_VUYA_4444_32f:
			case PrPixelFormat_VUYA_4444_32f_709:
				processSucceed = process_VUYA_4444_32f_frame(theData);
				break;

				// ============ native AE formats ============================= //
			case PrPixelFormat_ARGB_4444_8u:
				processSucceed = process_ARGB_4444_8u_frame(theData);
				break;

			case PrPixelFormat_ARGB_4444_16u:
				processSucceed = process_ARGB_4444_16u_frame(theData);
				break;

			case PrPixelFormat_ARGB_4444_32f:
				processSucceed = process_ARGB_4444_32f_frame(theData);
				break;

				// =========== miscellanous formats =========================== //
			case PrPixelFormat_RGB_444_10u:
				processSucceed = process_RGB_444_10u_frame(theData);
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

				(*paramsH)->checkbox_window_size = 0;
				(*paramsH)->chackbox_average_type = 0;
				(*paramsH)->pLog10TableSize = alg10TableSize;

				float* alignedAddress = nullptr;
				if (nullptr != (alignedAddress = allocate_aligned_log_table(theData, paramsH)))
				{
					init_1og10_table(alignedAddress, alg10TableSize);
					(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
				}
				else
				{
					((*theData)->piSuites->memFuncs->disposeHandle)(reinterpret_cast<char**>(paramsH));
					(*theData)->specsHandle = nullptr;
				}
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
