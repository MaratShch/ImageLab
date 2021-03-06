#include "ImageLabMonoTonal.h"

csSDK_int32 imageLabPixelFormatSupported (const VideoHandle theData)
{
	csSDK_int32 pixFormatResult = imNoErr;

	if (nullptr != theData)
	{
		switch ((*theData)->pixelFormatIndex)
		{
			// BGRA format's group (native AP format)
			case 0:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRA_4444_8u;
			break;

			case 1:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRA_4444_16u;
			break;

			case 2:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRA_4444_32f;
			break;

			case 3:
				(*theData)->pixelFormatSupported = PrPixelFormat_RGB_444_10u;
			break;

			// ARGB format's group (native AE format)
			case 4:
				(*theData)->pixelFormatSupported = PrPixelFormat_ARGB_4444_8u;
			break;

			case 5:
				(*theData)->pixelFormatSupported = PrPixelFormat_ARGB_4444_16u;
			break;

			case 6:
				(*theData)->pixelFormatSupported = PrPixelFormat_ARGB_4444_32f;
			break;

			// YUVA format's group (native AP format)
			case 7:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_8u;
			break;

			case 8:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_8u_709;
			break;

			case 9:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_32f;
			break;

			// miscellanous formats
			case 10:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_32f_709;
			break;

			// Packed format
			case 11:
				(*theData)->pixelFormatSupported = PrPixelFormat_YUYV_422_8u_601;
			break;

			case 12:
				(*theData)->pixelFormatSupported = PrPixelFormat_YUYV_422_8u_709;
			break;

			case 13:
				(*theData)->pixelFormatSupported = PrPixelFormat_UYVY_422_8u_601;
			break;

			case 14:
				(*theData)->pixelFormatSupported = PrPixelFormat_UYVY_422_8u_709;
			break;

			case 15:
				(*theData)->pixelFormatSupported = PrPixelFormat_UYVY_422_32f_601;
			break;

			case 16:
				(*theData)->pixelFormatSupported = PrPixelFormat_UYVY_422_32f_709;
			break;

			case 17:
				(*theData)->pixelFormatSupported = PrPixelFormat_V210_422_10u_601;
			break;

			case 18:
				(*theData)->pixelFormatSupported = PrPixelFormat_V210_422_10u_709;
			break;

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
