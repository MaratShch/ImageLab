#include "ImageLabColorCorrectionHSL.h"


bool allocate_aligned_buffer (const VideoHandle& theData, filterParamsH filtersParam, const size_t& newFrameSize)
{
	char* memAlignedAddress = reinterpret_cast<char*>((*filtersParam)->tmpMem.tmpBufferAlignedPtr);
	char* memAddress = reinterpret_cast<char*>((*filtersParam)->tmpMem.tmpBufferPtr);
	size_t frameSize = (*filtersParam)->tmpMem.tmpBufferSize;
	constexpr size_t colorChannels = 3;
	bool allocSucceed = false;

	/* free old memory buffer in first for avoid memory leaks */
	if (nullptr != memAddress)
	{
#ifdef _DEBUG
		/* cleanup temporary memory contains for DBG purpose only */
		memset(memAddress, 0, frameSize);
#endif
		((*theData)->piSuites->memFuncs->disposePtr)(memAddress);
		memAddress = memAlignedAddress = nullptr;
		(*filtersParam)->tmpMem.tmpBufferAlignedPtr = (*filtersParam)->tmpMem.tmpBufferAlignedPtr = nullptr;
		(*filtersParam)->tmpMem.tmpBufferSize = (*filtersParam)->tmpMem.tmpBufferSizeBytes = frameSize = 0;
	}

	/* allocate new memory buffer */
	const csSDK_uint32 totalMemorySize = static_cast<csSDK_uint32>(newFrameSize * colorChannels * sizeof(float) + CACHE_LINE);
	memAddress = ((*theData)->piSuites->memFuncs->newPtr)(totalMemorySize);

	/* crate alignment on PAGE SIZE and if memory buffer allocation succeed - save the pointer and size in handle structure */
	if (nullptr != memAddress)
	{
#ifdef _DEBUG
		/* cleanup new allocated memory storage for DBG purpose only */
		memset(memAddress, 0, totalMemorySize);
#endif
		constexpr unsigned long long alignmend = static_cast<unsigned long long>(CACHE_LINE);
		unsigned long long aligned_address = CreateAlignment(reinterpret_cast<unsigned long long>(memAddress), alignmend);

		(*filtersParam)->tmpMem.tmpBufferSize = newFrameSize;
		(*filtersParam)->tmpMem.tmpBufferSizeBytes = (totalMemorySize - CACHE_LINE);
		(*filtersParam)->tmpMem.tmpBufferPtr = reinterpret_cast<void*>(memAddress);
		(*filtersParam)->tmpMem.tmpBufferAlignedPtr = reinterpret_cast<void* __restrict>(aligned_address);
		allocSucceed = true;
	}

	return allocSucceed;
}


csSDK_int32 selectProcessFunction (const VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	filterParamsH		   paramsH = nullptr;
	filterMemoryHandle*    pTmpMemory = nullptr;
	csSDK_int32 errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*	  PPixSuite = nullptr;
		const SPErr err = SPBasic->AcquireSuite(strPpixSuite, 1l, (const void**)&PPixSuite);

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

			paramsH = reinterpret_cast<filterParamsH>((*theData)->specsHandle);
			if (nullptr == paramsH)
				return fsBadFormatIndex;

			// Allocate temporary buffer & check this buffer dimensions
			pTmpMemory = &(*paramsH)->tmpMem;
			const size_t newFramePixelsSize = height * width;

			if (nullptr == pTmpMemory->tmpBufferAlignedPtr || pTmpMemory->tmpBufferSize < newFramePixelsSize)
			{
				/* re-allocate temporary buffer */
				if (false == allocate_aligned_buffer(theData, paramsH, newFramePixelsSize))
					return fsBadFormatIndex;
			}

			void* __restrict srcImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
			void* __restrict dstImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
			if (nullptr == srcImg || nullptr == dstImg)
				return fsBadFormatIndex;


			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
				case PrPixelFormat_ARGB_4444_8u:
				case PrPixelFormat_RGB_444_10u:
				case PrPixelFormat_BGRA_4444_16u:
				case PrPixelFormat_ARGB_4444_16u:
				case PrPixelFormat_BGRA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				case PrPixelFormat_ARGB_4444_32f:
				break;

				default:
					processSucceed = false;
				break;
			}

			SPBasic->ReleaseSuite (strPpixSuite, 1l);
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

				(*paramsH)->hue_corse_level = 0.0f;
				(*paramsH)->hue_fine_level = 0.0f;
				(*paramsH)->saturation_level = 0.0f;
				(*paramsH)->luminance_level = 0.0f;
				(*paramsH)->compute_precise = '\0';
				(*paramsH)->tmpMem.tmpBufferSize = (*paramsH)->tmpMem.tmpBufferSizeBytes = 0;
				(*paramsH)->tmpMem.tmpBufferPtr = (*paramsH)->tmpMem.tmpBufferAlignedPtr = nullptr;

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
			/* free temporary memory storage */
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
