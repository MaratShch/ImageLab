﻿#include "ImageLabReflect.h"

template <typename T>
inline void simple_image_copy
(
	T* __restrict srcPix,
	T* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch
)
{
	__VECTOR_ALIGNED__
	for (csSDK_int32 i = 0; i < height; i++)
	{
		memcpy(&dstPix[i*linePitch], &srcPix[i*linePitch], width * sizeof(T));
	}
	return;
}

template <typename T>
inline void horizontal_flip
(
	T* __restrict srcPix,
	T* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch
)
{
	T* pSrc;
	T* pDst;
	csSDK_int32 i, j, k;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		pSrc = &srcPix[j * linePitch];
		pDst = &dstPix[j * linePitch];

		for (i = 0, k = width-1; i < width; i++, k--)
		{
			pDst[i] = pSrc[k];
		}
	}
	return;
}

template <typename T>
inline void vertical_flip
(
	T* __restrict srcPix,
	T* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch
)
{
	csSDK_int32 i, j, k;
	__VECTOR_ALIGNED__
	for (j = 0, k = height - 1; j < height; j++, k--)
	{
		memcpy(&dstPix[k*linePitch], &srcPix[j*linePitch], width * sizeof(T));
	}
	return;
}

template <typename T>
inline void both_directions_flip
(
	T* __restrict srcPix,
	T* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch
)
{
	T* pSrc;
	T* pDst;
	csSDK_int32 i, j, k, l;

	__VECTOR_ALIGNED__
	for (j = 0, k = height - 1; j < height; j++, k--)
	{
		pSrc = &srcPix[k * linePitch];
		pDst = &dstPix[j * linePitch];

		for (i = 0, l = width - 1; i < width; i++, l--)
		{
			pDst[i] = pSrc[l];
		}
	}

	return;
}

template <typename T>
bool reflect_image
(
	T* __restrict srcPix,
	T* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& reflectDirection
)
{
	switch (reflectDirection)
	{
		case reflectNo:
			simple_image_copy (srcPix, dstPix, width, height, linePitch);
		break;

		case reflectHorizontal:
			horizontal_flip (srcPix, dstPix, width, height, linePitch);
		break;

		case reflectVertical:
			vertical_flip (srcPix, dstPix, width, height, linePitch);
		break;

		case reflectBothDirections:
			both_directions_flip (srcPix, dstPix, width, height, linePitch);
		break;
	}
	return true;
}


csSDK_int32 selectProcessFunction (const VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	filterParamsH		   paramsH = nullptr;
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

			void* __restrict srcImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
			void* __restrict dstImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
			if (nullptr == srcImg || nullptr == dstImg)
				return fsBadFormatIndex;

			const csSDK_int32& reflectDirection = ((*paramsH)->checkbox_reflect_horizontal) | (((*paramsH)->checkbox_reflect_vertical) << 1);

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
				case PrPixelFormat_ARGB_4444_8u:
				case PrPixelFormat_RGB_444_10u:
				{
					csSDK_uint32* __restrict pSrcPix = reinterpret_cast<csSDK_uint32* __restrict>(srcImg);
					csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);
					processSucceed = reflect_image (pSrcPix, pDstPix, width, height, linePitch, reflectDirection);
				}
				break;

				case PrPixelFormat_BGRA_4444_16u:
				case PrPixelFormat_ARGB_4444_16u:
				{
					csSDK_uint64* __restrict pSrcPix = reinterpret_cast<csSDK_uint64* __restrict>(srcImg);
					csSDK_uint64* __restrict pDstPix = reinterpret_cast<csSDK_uint64* __restrict>(dstImg);
					processSucceed = reflect_image (pSrcPix, pDstPix, width, height, linePitch, reflectDirection);
				}
				break;

				case PrPixelFormat_BGRA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				case PrPixelFormat_ARGB_4444_32f:
				{
					__m128i* __restrict pSrcPix = reinterpret_cast<__m128i* __restrict>(srcImg);
					__m128i* __restrict pDstPix = reinterpret_cast<__m128i* __restrict>(dstImg);
					processSucceed = reflect_image (pSrcPix, pDstPix, width, height, linePitch, reflectDirection);
				}
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

				(*paramsH)->checkbox_reflect_horizontal = '\0';
				(*paramsH)->checkbox_reflect_vertical = '\0';
				(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
			}
			break;

		case fsSetup:
		break;

		case fsExecute:
			errCode = selectProcessFunction(theData);
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
		break;
		
	}

	return errCode;
}
