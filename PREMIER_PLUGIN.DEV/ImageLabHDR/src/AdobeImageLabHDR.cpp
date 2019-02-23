#include "AdobeImageLabHDR.h"


// ImageLabHDR filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	csSDK_int32 errCode = fsNoErr;
	FilterParamHandle filterParamH = nullptr;


	switch (selector)
	{
		case fsInitSpec:
			errCode = fsNoErr;

			if ((*theData)->specsHandle)
			{

			}
			else
			{
				filterParamH = reinterpret_cast<FilterParamHandle>(((*theData)->piSuites->memFuncs->newHandle)(sizeof(FilterParamStr)));
				if (nullptr == filterParamH)
					break; // memory allocation failed

				IMAGE_LAB_FILTER_PARAM_HANDLE_INIT(filterParamH);
				
				// get memory internally allocated on DLL connected to process
				(*filterParamH)->pMemHandler = GetStreamMemory();

				// save the filter parameters inside of Premier handler
				(*theData)->specsHandle = reinterpret_cast<char**>(filterParamH);
			}
		break;

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
		break;

		case fsSetup:
		break;

		case fsExecute:
			errCode = fsNoErr;
			// Get the data from specsHandle
			filterParamH = reinterpret_cast<FilterParamHandle>((*theData)->specsHandle);
			if (nullptr != filterParamH)
			{
				// execute filter
				prRect box = { 0 };
				// Get the frame dimensions
				((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

				// Calculate dimensions
				const csSDK_int32 height = box.bottom - box.top;
				const csSDK_int32 width = box.right - box.left;
				const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

				// Create copies of pointer to the source, destination frames
				csSDK_uint32* srcPix = reinterpret_cast<csSDK_uint32*>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
				csSDK_uint32* dstPix = reinterpret_cast<csSDK_uint32*>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

				const double totalPixels = static_cast<double>(height * width);
				// get left threshold from slider
				const csSDK_int32 sliderLeftPosition  = (*filterParamH)->sliderLeft;
				const csSDK_int32 sliderRightPosition = (*filterParamH)->sliderRight;

				const double leftCount  = (totalPixels * static_cast<double>(sliderLeftPosition)) / 100.0;
				const double rigthCount = totalPixels - (totalPixels * static_cast<double>(sliderRightPosition)) / 100.0;

				// cleanup buffer for histogram
				void* pHist = GetHistogramBuffer();
				void* pBin  = GetBinarizationBuffer();
				void* pLut  = GetLUTBuffer();

				if (nullptr != pHist && nullptr != pBin && nullptr != pLut)
				{
					memset(pHist, 0, IMAGE_LAB_HIST_BUFFER_SIZE);
					memset(pLut,  0, IMAGE_LAB_LUT_BUFFER_SIZE);
					memset(pBin,  0, IMAGE_LAB_BIN_BUFFER_SIZE);

					// check pixel format

					// handle fields mode !!!

					// convert to YUV (if required)

					// compute histogram from Y

					// make histogtam binarization

					// generate LUT

					// apply LUT 

					// back convert from YUV to input format (if required)
#if 0
					// DBG loop
					for (int vert = 0; vert < height; ++vert)
					{
						for (int horiz = 0; horiz < width; ++horiz)
						{
							csSDK_uint32 Y_value = (*srcPix & 0x00FF0000) >> 16;

							*dstPix = (*srcPix & 0xFF00FFFFu) + ((Y_value / 2) << 16);

								++srcPix; ++dstPix;

						}
					}
#endif
				}
			}
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
		break;

		default:
			// unhandled case
		break;
		
	}

	return errCode;
}
