#include "ImageLabMonoTonal.h"

csSDK_int32 imageLabPixelFormatSupported (const VideoHandle theData)
{
	csSDK_int32 pixFormatResult = imNoErr;

	if (nullptr != theData)
	{
		switch ((*theData)->pixelFormatIndex)
		{
#if 0
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

			case 10:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_32f_709;
			break;
#endif
			case 0:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_8u;
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
