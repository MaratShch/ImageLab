#include "AdobeImageLabHDR.h"

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData)
{
	csSDK_int32 pixFormatResult = imNoErr;

	if (nullptr != theData)
	{
		switch ((*theData)->pixelFormatIndex)
		{
			case 0:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_8u; ;
			break;

			case 1:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_8u_709;
			break;

#if 0
			case 1:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRA_4444_8u;
			break;

			case 2:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_8u_709;
			break;

			case 3:
				(*theData)->pixelFormatSupported = PrPixelFormat_ARGB_4444_8u;
			break;

			case 4:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRX_4444_8u;
			break;

			case 5:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYX_4444_8u;
			break;

			case 6: 
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYX_4444_8u_709;
			break;

			case 7:
				(*theData)->pixelFormatSupported = PrPixelFormat_XRGB_4444_8u;
			break;

			case 8:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRP_4444_8u;
			break;

			case 9:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYP_4444_8u;
			break;

			case 10:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYP_4444_8u_709;
			break;

			case 11:
				(*theData)->pixelFormatSupported = PrPixelFormat_PRGB_4444_8u;
			break;

			case 12:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRA_4444_16u;
			break;

			case 13:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_16u;
			break;

			case 14:
				(*theData)->pixelFormatSupported = PrPixelFormat_ARGB_4444_16u;
			break;

			case 15:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRX_4444_16u;
			break;

			case 16:
				(*theData)->pixelFormatSupported = PrPixelFormat_XRGB_4444_16u;
			break;

			case 17:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRP_4444_16u;
			break;

			case 18:
				(*theData)->pixelFormatSupported = PrPixelFormat_PRGB_4444_16u;
			break;

			case 19:
				(*theData)->pixelFormatSupported = PrPixelFormat_RGB_444_10u;
			break;

			case 20:
				(*theData)->pixelFormatSupported = PrPixelFormat_YUYV_422_8u_601;
			break;

			case 21:
				(*theData)->pixelFormatSupported = PrPixelFormat_YUYV_422_8u_709;
			break;

			case 22:
				(*theData)->pixelFormatSupported = PrPixelFormat_UYVY_422_8u_601;
			break;

			case 23:
				(*theData)->pixelFormatSupported = PrPixelFormat_UYVY_422_8u_709;
			break;

			case 24:
				(*theData)->pixelFormatSupported = PrPixelFormat_V210_422_10u_601;
			break;

			case 25:
				(*theData)->pixelFormatSupported = PrPixelFormat_V210_422_10u_709;
			break;

			case 26:
				(*theData)->pixelFormatSupported = PrPixelFormat_RGB_444_12u_PQ_709;
			break;

			case 27:
				(*theData)->pixelFormatSupported = PrPixelFormat_RGB_444_12u_PQ_P3;
			break;

			case 28:
				(*theData)->pixelFormatSupported = PrPixelFormat_RGB_444_12u_PQ_2020;
			break;
#endif
			default:
				pixFormatResult = fsBadFormatIndex;
			break;
		}
	}
	else
	{
		pixFormatResult = imMemErr;
	}

	return pixFormatResult;
}