#include "ImageLabGlassyEffect.h"


static csSDK_int32 processFrame(VideoHandle theData)
{
	prRect box = { 0 };
	FilterParamHandle filterParamH = nullptr;
	csSDK_int32 errCode = fsNoErr;

	int i, j, k, m, idx;
	int xIdx, yIdx;
	float randomValue1, randomValue2;

	// Get the data from specsHandle
	filterParamH = reinterpret_cast<FilterParamHandle>((*theData)->specsHandle);

	if (nullptr != filterParamH)
	{
		// Get the frame dimensions
		((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

		// Calculate dimensions
		const csSDK_int32 height = box.bottom - box.top;
		const csSDK_int32 width = box.right - box.left;
		const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

		// Create copies of pointer to the source, destination frames
		csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
		csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

		const int sliderPosition = GET_WINDOW_SIZE_FROM_SLIDER((*filterParamH)->sliderPosition);
		const float* __restrict pRandomValues = (*filterParamH)->pBufRandom;

		const csSDK_int32 shortWidth = width - sliderPosition * 2;

		idx = 0;

		for (j = 0; j < height; j++)
		{
			for (i = 0; i < shortWidth; i++)
			{
				randomValue1 = pRandomValues[idx++];
				idx &= idxMask;

				randomValue2 = pRandomValues[idx++];
				idx &= idxMask;

				xIdx = static_cast<int>(randomValue1 * sliderPosition);
				yIdx = static_cast<int>(randomValue2 * sliderPosition);

				const int pixOffset = yIdx * width + xIdx;

				*dstPix = *(srcPix + pixOffset);
				srcPix++;
				dstPix++;
			}

			for (i = shortWidth; i < width; i++)
			{
				randomValue1 = pRandomValues[idx++];
				idx &= idxMask;

				randomValue2 = pRandomValues[idx++];
				idx &= idxMask;

				xIdx = static_cast<int>(1.50f * randomValue1 * sliderPosition);
				yIdx = static_cast<int>(randomValue2 * sliderPosition);

				const int pixOffset = yIdx * width + xIdx;

				*dstPix = *(srcPix + pixOffset);
				dstPix++;

				if (i & 0x01)
					srcPix++;
			}

			dstPix += (rowbytes / 4) - width;
			srcPix += (rowbytes / 4) - width + sliderPosition;
		}

	}

	return errCode;
}


// filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	csSDK_int32 errCode = fsNoErr;
	FilterParamHandle filterParamH = nullptr;

	switch (selector)
	{
		case fsInitSpec:
		{
			if ((*theData)->specsHandle)
			{

			}
			else
			{
				filterParamH = reinterpret_cast<FilterParamHandle>(((*theData)->piSuites->memFuncs->newHandle)(sizeof(FilterParamStr)));
				if (nullptr == filterParamH)
					break; // memory allocation failed

			    // get memory internally allocated on DLL connected to process
				(*filterParamH)->pBufRandom = get_random_values_buffer();
				// init default slider position
				(*filterParamH)->sliderPosition = 0;

				// save the filter parameters inside of Premier handler
				(*theData)->specsHandle = reinterpret_cast<char**>(filterParamH);
			}
		}
		break;

		case fsHasSetupDialog:
				errCode = fsHasNoSetupDialog;
		break;

		case fsSetup:
		break;

		case fsExecute:
			errCode = processFrame(theData);
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