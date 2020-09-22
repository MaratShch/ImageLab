#include "AdobeImageLabColorSubstitution.h"

template <typename T>
inline void simple_image_copy
(
	const T* __restrict srcPix,
	      T* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& words
)
{
	const size_t lineBytesLength = (words * width) * sizeof(T);

	__VECTOR_ALIGNED__
	for (csSDK_int32 i = 0; i < height; i++)
	{
		memcpy (&dstPix[i*linePitch], &srcPix[i*linePitch], lineBytesLength);
	}
	return;
}



csSDK_int32 selectProcessFunction(VideoHandle theData)
{
	static constexpr char* strPpixSuite{ "Premiere PPix Suite" };
	SPBasicSuite*		   SPBasic = nullptr;
	filterParamsH	       paramsH = nullptr;
	constexpr long         siteVersion{ 1 };
	csSDK_int32            errCode = fsBadFormatIndex;
	prColor                colorFrom = 0x0, colorTo = 0x0;
	csSDK_int32            colorTolerance = 0;
	bool                   replaceColor = false;
	bool                   showMask = false;
	bool                   processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*	PPixSuite = nullptr;
		const SPErr     err = SPBasic->AcquireSuite(strPpixSuite, siteVersion, (const void**)&PPixSuite);

		if (kSPNoError == err && nullptr != PPixSuite)
		{
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

			if (nullptr != (paramsH = reinterpret_cast<filterParamsH>((*theData)->specsHandle)))
			{
				colorFrom      = 0x00FFFFFFu & (*paramsH)->fromColor;
				colorTo        = 0x00FFFFFFu & (*paramsH)->toColor;
				colorTolerance = (*paramsH)->colorTolerance;
				replaceColor   = ((colorFrom != colorTo) ? true : false);
				showMask       = ((0u != ((*paramsH)->showMask) && true == replaceColor) ? true : false);
			}

			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			prSuiteError  siteErr = suiteError_Fail;
			if (suiteError_NoError != (siteErr = PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat)))
				return fsBadFormatIndex;

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
						  csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					true == replaceColor ?
						  colorSubstitute_BGRA_4444_8u (pSrcPix, pDstPix, height, width, linePitch, colorFrom, colorTo, colorTolerance, showMask) :
						  simple_image_copy (pSrcPix, pDstPix, width, height, linePitch, 1);
				}
				break;

				case PrPixelFormat_BGRA_4444_16u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
						  csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					true == replaceColor ?
						  colorSubstitute_BGRA_4444_16u(pSrcPix, pDstPix, height, width, linePitch, colorFrom, colorTo, colorTolerance, showMask) :
						  simple_image_copy(pSrcPix, pDstPix, width, height, linePitch, 2);
				}
				break;

				case PrPixelFormat_BGRA_4444_32f:
				{
					const float* __restrict pSrcPix = reinterpret_cast<const float* __restrict>(srcImg);
						  float* __restrict pDstPix = reinterpret_cast<float* __restrict>(dstImg);

					true == replaceColor ?
						colorSubstitute_BGRA_4444_32f (pSrcPix, pDstPix, height, width, linePitch, colorFrom, colorTo, colorTolerance, showMask) :
						simple_image_copy(pSrcPix, pDstPix, width, height, linePitch, 4);
				}
				break;

				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

				}
				break;

				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				{
					const float* __restrict pSrcPix = reinterpret_cast<const float* __restrict>(srcImg);
					float* __restrict pDstPix = reinterpret_cast<float* __restrict>(dstImg);

				}
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
 						  csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					true == replaceColor ?
						colorSubstitute_ARGB_4444_8u(pSrcPix, pDstPix, height, width, linePitch, colorFrom, colorTo, colorTolerance, showMask) :
						simple_image_copy(pSrcPix, pDstPix, width, height, linePitch, 1);
				}
				break;

				case PrPixelFormat_ARGB_4444_16u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
						  csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

				    true == replaceColor ?
						  colorSubstitute_ARGB_4444_16u(pSrcPix, pDstPix, height, width, linePitch, colorFrom, colorTo, colorTolerance, showMask) :
						  simple_image_copy(pSrcPix, pDstPix, width, height, linePitch, 1);
				}
				break;

				case PrPixelFormat_ARGB_4444_32f:
				{
					const float* __restrict pSrcPix = reinterpret_cast<const float* __restrict>(srcImg);
					      float* __restrict pDstPix = reinterpret_cast<float* __restrict>(dstImg);

					true == replaceColor ?
						colorSubstitute_ARGB_4444_32f(pSrcPix, pDstPix, height, width, linePitch, colorFrom, colorTo, colorTolerance, showMask) :
						simple_image_copy(pSrcPix, pDstPix, width, height, linePitch, 4);
				}
				break;

				case PrPixelFormat_RGB_444_10u:
				{
					const csSDK_uint32* __restrict src = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
						  csSDK_uint32* __restrict dst = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);
				}
				break;

				default:
					processSucceed = false;
				break;
			}

			SPBasic->ReleaseSuite(strPpixSuite, siteVersion);
			errCode = (true == processSucceed) ? fsNoErr : errCode;
		}
	}

	return errCode;
}


PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	filterParamsH		paramsH = nullptr;
	csSDK_int32			errCode = fsNoErr;
	constexpr csSDK_uint32	filterParamSize = sizeof(filterParams);

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
			paramsH = reinterpret_cast<filterParamsH>(((*theData)->piSuites->memFuncs->newHandle)(filterParamSize));

			if (nullptr != paramsH)
			{
				(*paramsH)->fromColor = SET_RGB_888(128, 128, 128);
				(*paramsH)->toColor   = SET_RGB_888(128, 128, 128);
				(*paramsH)->colorTolerance = 10;
				(*paramsH)->showMask = 0;
			}

			(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
		}
		break;


	case fsSetup:
		break;

	case fsExecute:
		errCode = selectProcessFunction(theData);
		break;

	case fsDisposeData:
		if (nullptr != (*theData)->specsHandle)
		{
			(*theData)->piSuites->memFuncs->disposeHandle((*theData)->specsHandle);
			(*theData)->specsHandle = nullptr;
		}
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