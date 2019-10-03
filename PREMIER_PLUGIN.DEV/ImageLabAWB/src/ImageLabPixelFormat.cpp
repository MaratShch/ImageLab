#include "AdobeImageLabAWB.h"


csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData)
{
	csSDK_int32 pixFormatResult = imNoErr;

	if (nullptr != theData)
	{
		switch ((*theData)->pixelFormatIndex)
		{
			case 0:
				(*theData)->pixelFormatSupported = PrPixelFormat_BGRA_4444_8u;
			break;

			case 1:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_8u;
			break;

			case 2:
				(*theData)->pixelFormatSupported = PrPixelFormat_VUYA_4444_8u_709;
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