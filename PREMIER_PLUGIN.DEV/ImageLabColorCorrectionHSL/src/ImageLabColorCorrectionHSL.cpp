#include "ImageLabColorCorrectionHSL.h"


void free_aligned_buffer (filterMemoryHandle* fMemHndl)
{
	if (nullptr != fMemHndl)
	{
		if (nullptr != fMemHndl->tmpBufferAlignedPtr)
		{
			_aligned_free(fMemHndl->tmpBufferAlignedPtr);
			fMemHndl->tmpBufferAlignedPtr = nullptr;
		}
		fMemHndl->tmpBufferSizeBytes = 0;
	}

	return;
}


void* allocate_aligned_buffer (filterMemoryHandle* fTmpMemory, const size_t& newSize)
{
	void* memAlignedAddress = fTmpMemory->tmpBufferAlignedPtr;
	size_t frameSize = fTmpMemory->tmpBufferSizeBytes;

	/* free old memory buffer in first for avoid memory leaks */
	if (nullptr != memAlignedAddress)
	{
#ifdef _DEBUG
		/* cleanup temporary memory contains for DBG purpose only */
		memset(memAlignedAddress, 0, frameSize);
#endif
		_aligned_free(memAlignedAddress);
		memAlignedAddress = nullptr;
		fTmpMemory->tmpBufferAlignedPtr = nullptr;
		fTmpMemory->tmpBufferSizeBytes = 0;
	}

	/* allocate new memory buffer */
	memAlignedAddress = _aligned_malloc (newSize, CACHE_LINE);

	/* crate alignment on PAGE SIZE and if memory buffer allocation succeed - save the pointer and size in handle structure */
	if (nullptr != memAlignedAddress)
	{
#ifdef _DEBUG
		/* cleanup new allocated memory storage for DBG purpose only */
		memset(memAlignedAddress, 0, newSize);
#endif
		fTmpMemory->tmpBufferAlignedPtr = memAlignedAddress;
		fTmpMemory->tmpBufferSizeBytes = newSize;
	}

	return memAlignedAddress;
}

inline const float normalize_hue_wheel(const float& wheel_value)
{
	const float tmp = wheel_value / 360.0f;
	const int intPart = static_cast<int>(tmp);
	return (tmp - static_cast<float>(intPart)) * 360.0f;
}

csSDK_int32 selectProcessFunction (const VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	constexpr long         PpixSuiteVersion = 1l;
	SPBasicSuite*		   SPBasic = nullptr;
	filterMemoryHandle*    pMemHndl = nullptr;
	float*                 pTmpBuffer = nullptr;
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

			const csSDK_int32 tmpBufPixWidth = 3 * sizeof(float);

			// Check temporary buffer dimensions and re-allocate if require more memory 
			const size_t newFramePixelsSize = height * width;
			const size_t newFrameBytesSize = newFramePixelsSize * tmpBufPixWidth;
			pMemHndl = get_tmp_memory_handler();
			if (nullptr == pMemHndl->tmpBufferAlignedPtr || pMemHndl->tmpBufferSizeBytes < newFrameBytesSize)
			{
				/* re-allocate temporary buffer */
				if (nullptr == allocate_aligned_buffer (pMemHndl, newFrameBytesSize))
					return fsBadFormatIndex;
			}

			void* __restrict srcImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
			void* __restrict dstImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
			pTmpBuffer = reinterpret_cast<float* __restrict>(pMemHndl->tmpBufferAlignedPtr);
			
			if (nullptr == srcImg || nullptr == dstImg || nullptr == pTmpBuffer)
				return fsBadFormatIndex;

			/* acquire setting */
			const float addHue = normalize_hue_wheel((*paramsH)->hue_corse_level) + (*paramsH)->hue_fine_level;
			const float addLuminance = (*paramsH)->luminance_level;
			const float addSaturation = (*paramsH)->saturation_level;

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				{
					const csSDK_uint32* __restrict src = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict dst = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					bgr_to_hsl_precise_BGRA4444_8u(src, pTmpBuffer, width, height, linePitch, addHue, addLuminance, addSaturation);
					hsl_to_bgr_precise_BGRA4444_8u(src, pTmpBuffer, dst, width, height, linePitch);
				}
				break;

				case PrPixelFormat_BGRA_4444_32f:
				{
					const float* __restrict src = reinterpret_cast<const float* __restrict>(srcImg);
					      float* __restrict dst = reinterpret_cast<float* __restrict>(dstImg);

					bgr_to_hsl_precise_BGRA4444_32f(src, pTmpBuffer, width, height, linePitch, addHue, addLuminance, addSaturation);
					hsl_to_bgr_precise_BGRA4444_32f(src, pTmpBuffer, dst, width, height, linePitch);
				}
				break;

	
				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
				case PrPixelFormat_ARGB_4444_8u:
				case PrPixelFormat_RGB_444_10u:
				case PrPixelFormat_BGRA_4444_16u:
				case PrPixelFormat_ARGB_4444_16u:
				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				case PrPixelFormat_ARGB_4444_32f:
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

				(*paramsH)->hue_corse_level  = 0.0f;
				(*paramsH)->hue_fine_level   = 0.0f;
				(*paramsH)->saturation_level = 0.0f;
				(*paramsH)->luminance_level  = 0.0f;
				(*paramsH)->pTmpMem = nullptr;
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
