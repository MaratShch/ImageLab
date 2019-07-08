#include "AdobeImageLabAWB.h"

// define color space conversion matrix's
CACHE_ALIGN double constexpr RGB2YUV[LAST][9] =
{
	// BT.601
#if 0
	{
		0.299000,  0.587000,  0.114000,
	   -0.168740, -0.331260,  0.500000,
		0.500000, -0.418690, -0.081310
	},
#endif
	{
		0.299,  0.587,  0.114,
	   -0.299, -0.587,  0.886,
	    0.701, -0.587, -0.114
	},

	// BT.709
	{
		0.212600,   0.715200,  0.072200,
	   -0.114570,  -0.385430,  0.500000,
		0.500000,  -0.454150, -0.045850
	},

	// BT.2020
	{
		0.262700,   0.678000,  0.059300,
	   -0.139630,  -0.360370,  0.500000,
		0.500000,  -0.459790, -0.040210
	},

	// SMPTE 240M
	{
		0.212200,   0.701300,  0.086500,
	   -0.116200,  -0.383800,  0.500000,
		0.500000,  -0.445100, -0.054900
	}
};



CACHE_ALIGN double constexpr YUV2RGB[LAST][9] =
{


};

#ifdef _DEBUG

#endif

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
			PrPixelFormat			pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			switch (pixelFormat)
			{
//				case PrPixelFormat_VUYA_4444_8u:
//					break;
				case PrPixelFormat_BGRA_4444_8u:
					processSucceed = procesBGRA4444_8u_slice (theData, RGB2YUV[STD_BT601], YUV2RGB[STD_BT601]);
					break;
#if 0
				case PrPixelFormat_VUYA_4444_8u_709:
					break;
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
	csSDK_int32 errCode = fsNoErr;

	switch (selector)
	{
		case fsInitSpec:
			errCode = fsNoErr;
		break;

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
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
