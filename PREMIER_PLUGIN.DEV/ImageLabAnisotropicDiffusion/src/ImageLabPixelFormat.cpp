#include "ImageLabAnisotropicDiffusion.h"

csSDK_int32 imageLabPixelFormatSupported (const VideoHandle theData)
{
	csSDK_int32 pixFormatResult = imNoErr;

	if (nullptr != theData)
	{
		switch ((*theData)->pixelFormatIndex)
		{
			case 0:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_32f;// PrPixelFormat_BGRA_4444_16u;
			break;

#if 0
			case 0:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_8u;
			break;

			case 1:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_8u_709;
			break;

			case 2:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRA_4444_8u;
			break;

			case 3:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_32f;
			break;

			case 4:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_32f_709;
			break;
/////////////////////
			case 1:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRA_4444_16u;
			break;

			case 7:
				(*theData)->pixelFormatSupported = PrPixelFormat_ARGB_4444_8u;
			break;

			case 8:
				(*theData)->pixelFormatSupported = PrPixelFormat_ARGB_4444_16u;
			break;

			case 10:
				(*theData)->pixelFormatSupported = PrPixelFormat_RGB_444_10u;
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