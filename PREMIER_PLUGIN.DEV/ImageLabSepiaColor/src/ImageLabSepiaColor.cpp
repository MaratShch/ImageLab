#include "AdobeImageLabSepia.h"
#include <math.h>

CACHE_ALIGN constexpr float SepiaMatrix[9] = 
{
	0.3930f, 0.7690f, 0.1890f,
	0.3490f, 0.6860f, 0.1680f,
	0.2720f, 0.5340f, 0.1310f
};


bool processSepiaBGRA4444_8u_slice(VideoHandle theData)
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

	unsigned int alpha;
	unsigned int Luma;
	float R, G, B;
	int newR, newG, newB;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			for (int i = 0; i < width; i++)
			{
				const csSDK_uint32 BGRAPixel = *srcImg++;

				alpha = BGRAPixel & 0xFF000000u;
				R = static_cast<float>((BGRAPixel & 0x00FF0000) >> 16);
				G = static_cast<float>((BGRAPixel & 0x0000FF00) >> 8);
				B = static_cast<float>(BGRAPixel & 0x000000FF);

				newR = static_cast<int>(R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2]);
				newG = static_cast<int>(R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5]);
				newB = static_cast<int>(R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8]);

				const csSDK_uint32 BWPixel = alpha |
					(CLAMP_RGB8(newR)) << 16 |
					(CLAMP_RGB8(newG)) << 8  |
					(CLAMP_RGB8(newB));

				*dstImg++ = BWPixel;
			}

		srcImg += linePitch - width;
		dstImg += linePitch - width;
	}

	return true;
}


csSDK_int32 selectProcessFunction(VideoHandle theData)
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
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
					processSucceed = processSepiaBGRA4444_8u_slice(theData);
				break;

				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
				break;

				case PrPixelFormat_BGRA_4444_16u:
				break;

				case PrPixelFormat_BGRA_4444_32f:
				break;

				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
				break;

				case PrPixelFormat_ARGB_4444_16u:
				break;

				case PrPixelFormat_ARGB_4444_32f:
				break;

				// =========== miscellanous formats =========================== //
				case PrPixelFormat_RGB_444_10u:
				break;

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
	csSDK_int32		errCode = fsNoErr;

	switch (selector)
	{
		case fsInitSpec:
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


