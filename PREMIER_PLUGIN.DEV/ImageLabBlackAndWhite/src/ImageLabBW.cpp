#include "AdobeImageLabBW.h"
#include <math.h>

constexpr unsigned int BT601 = 0u;
constexpr unsigned int BT709 = 1u;

CACHE_ALIGN constexpr float coeff [][3] = 
{
	// BT.601
	{
		0.2990f,  0.5870f,  0.1140f
	},

	// BT.709
	{
		0.2126f,   0.7152f,  0.0722f
	}

};

CACHE_ALIGN float R_coeff[256] = {};
CACHE_ALIGN float G_coeff[256] = {};
CACHE_ALIGN float B_coeff[256] = {};


void initCompCoeffcients(void)
{
	int i;

	__VECTOR_ALIGNED__
	for (i = 0; i < 256; i++)
		R_coeff[i] = 0.2126f * pow(static_cast<float>(i), 2.20f);

	__VECTOR_ALIGNED__
	for (i = 0; i < 256; i++)
		G_coeff[i] = 0.7152f * pow(static_cast<float>(i), 2.20f);

	__VECTOR_ALIGNED__
	for (i = 0; i < 256; i++)
		B_coeff[i] = 0.0722f * pow(static_cast<float>(i), 2.20f);
}


bool processVUYA_4444_8u_slice(VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
	
		// first pass - accquire color statistics
	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				*dstImg++ = (*srcImg++ & 0xFFFF0000u) | 0x00008080u;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}



bool processBGRA4444_8u_slice (VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	const float* __restrict lpCoeff = (width > 800) ? coeff[BT709] : coeff[BT601];

	unsigned int alpha;
	unsigned int Luma;
	float R, G, B;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
		for (int i = 0; i < width; i++)
		{
			const csSDK_uint32 BGRAPixel = *srcImg++;

			alpha = BGRAPixel & 0xFF000000u;
			R = static_cast<float>((BGRAPixel & 0x00FF0000) >> 16);
			G = static_cast<float>((BGRAPixel & 0x0000FF00) >> 8);
			B = static_cast<float>( BGRAPixel & 0x000000FF);

			Luma = static_cast<unsigned int>(R * lpCoeff[0] + G * lpCoeff[1] + B * lpCoeff[2]);

			const csSDK_uint32 BWPixel = alpha | 
									Luma << 16 |
									Luma << 8  |
									Luma;

			*dstImg++ = BWPixel;
		}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}


bool processAdvancedBGRA4444_8u_slice(VideoHandle theData)
{
#if !defined __INTEL_COMPILER 
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);
	const int linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	csSDK_uint32* __restrict srcImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	csSDK_uint32* __restrict dstImg = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	constexpr float fLumaExp = 1.0f / 2.20f;

	float fLuma;
	unsigned int Luma;
	unsigned int alpha;
	int R, G, B;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 BGRAPixel = *srcImg++;

				alpha = BGRAPixel & 0xFF000000u;
				R = static_cast<int>((BGRAPixel & 0x00FF0000) >> 16);
				G = static_cast<int>((BGRAPixel & 0x0000FF00) >> 8);
				B = static_cast<int>(BGRAPixel & 0x000000FF);

				fLuma = pow((R_coeff[R] + G_coeff[G] + B_coeff[B]), fLumaExp);
				Luma = static_cast<unsigned int>(fLuma);

				const csSDK_uint32 BWPixel = alpha |
					Luma << 16 |
					Luma << 8 |
					Luma;

				*dstImg++ = BWPixel;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}


csSDK_int32 selectProcessFunction(VideoHandle theData, bool advancedAlg)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	csSDK_int32 errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*			PPixSuite = nullptr;
		SPBasic->AcquireSuite(strPpixSuite, 1, (const void**)&PPixSuite);

		if (nullptr != PPixSuite)
		{
			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			switch (pixelFormat)
			{
				case PrPixelFormat_BGRA_4444_8u:
					processSucceed = (true == advancedAlg ? processAdvancedBGRA4444_8u_slice(theData) : processBGRA4444_8u_slice(theData));
				break;

				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
					processSucceed = processVUYA_4444_8u_slice(theData);
				break;
#if 0
				case PrPixelFormat_ARGB_4444_8u:
					break;
				case PrPixelFormat_BGRX_4444_8u:
					break;
				case PrPixelFormat_VUYX_4444_8u:
					break;
				case PrPixelFormat_VUYX_4444_8u_709:
					break;
				case PrPixelFormat_XRGB_4444_8u:
					break;
				case PrPixelFormat_BGRP_4444_8u:
					break;
				case PrPixelFormat_VUYP_4444_8u:
					break;
				case PrPixelFormat_VUYP_4444_8u_709:
					break;
				case PrPixelFormat_PRGB_4444_8u:
					break;
				case PrPixelFormat_BGRA_4444_16u:
					break;
				case PrPixelFormat_VUYA_4444_16u:
					break;
				case PrPixelFormat_ARGB_4444_16u:
					break;
				case PrPixelFormat_BGRX_4444_16u:
					break;
				case PrPixelFormat_XRGB_4444_16u:
					break;
				case PrPixelFormat_BGRP_4444_16u:
					break;
				case PrPixelFormat_PRGB_4444_16u:
					break;
				case PrPixelFormat_RGB_444_10u:
					break;
				case PrPixelFormat_YUYV_422_8u_601:
					break;
				case PrPixelFormat_YUYV_422_8u_709:
					break;
				case PrPixelFormat_UYVY_422_8u_601:
					break;
				case PrPixelFormat_UYVY_422_8u_709:
					break;
				case PrPixelFormat_V210_422_10u_601:
					break;
				case PrPixelFormat_V210_422_10u_709:
					break;
				case PrPixelFormat_RGB_444_12u_PQ_709:
					break;
				case PrPixelFormat_RGB_444_12u_PQ_P3:
					break;
				case PrPixelFormat_RGB_444_12u_PQ_2020:
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
	bool			advFlag = false;

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

				(*paramsH)->checkbox = 0;
				(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
			}
		break;

		case fsSetup:
		break;

		case fsExecute:
			// Get the data from specsHandle
			paramsH = (filterParamsH)(*theData)->specsHandle;
			advFlag = (nullptr != paramsH ? ((*paramsH)->checkbox ? true : false) : false);
			errCode = selectProcessFunction(theData, advFlag);
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


