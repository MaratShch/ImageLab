#include "AdobeImageLabHDR.h"
#include "ImageLabHDR.h"

#if 0
static int allocateParamStructure(
	VideoHandle	theData,
	PImageLabHDR_ParamStr imageLabParamStr)
{
	imageLabParamStr = reinterpret_cast<PImageLabHDR_ParamStr>((*theData)->piSuites->memFuncs->newHandle(sizeof(*imageLabParamStr)));

	if (nullptr != imageLabParamStr)
	{
		// initialize by default values
		IMAGE_LAB_HDR_PSTR_PARAM_INIT(imageLabParamStr);

		// store the filter data in Video Handle
		(*theData)->specsHandle = reinterpret_cast<char**>(imageLabParamStr);
	}
	else
	{
		(*theData)->specsHandle = nullptr;
	}

	return imNoErr;
}


static int initializeFilter(
	VideoHandle	theData,
	PImageLabHDR_ParamStr imageLabParamStr)
{
	int errCode = imNoErr;

	if (nullptr != theData)
	{
		if (nullptr != (*theData)->specsHandle)
		{
			imageLabParamStr = reinterpret_cast<PImageLabHDR_ParamStr>((*theData)->specsHandle);
		}
		else
		{
			errCode = allocateParamStructure(theData, imageLabParamStr);
		}
	}

	return errCode;
}


// ImageLabHDR filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	PImageLabHDR_ParamStr pParamStr = nullptr;
	csSDK_int32 errCode = imNoErr;

	switch (selector)
	{
		case fsInitSpec:
			errCode = allocateParamStructure(theData, pParamStr);
		break;

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
		break;

		case fsSetup:
			errCode = fsNoErr;
		break;

		case fsExecute:
			errCode = fsNoErr;

			// Get the data from specsHandle
			pParamStr = (PImageLabHDR_ParamStr)(*theData)->specsHandle;

			if (pParamStr)
			{
				break;
			}
			else
			{
				errCode = fsNoErr;
			}
			break;

//		case fsDisposeData:
//		break;

		case fsCanHandlePAR:
			errCode = prEffectCanHandlePAR;
		break;
			
		case fsGetPixelFormatsSupported:
			errCode = imageLabPixelFormatSupported(theData);
		break;

//		case fsCacheOnLoad:
//		break;

//		default:
//			// unhandled case
//		break;
		
	}

	return errCode;
}
#else

typedef struct filterParams
{
	prColor		color;
	csSDK_int32	slider;
	char		checkbox;
} filterParams, *filterParamsP, **filterParamsH;


PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	csSDK_int32		result = fsUnsupported,
		height = 0,
		width = 0,
		rowbytes = 0,
		callbackTime = 0;

	csSDK_uint32	*srcpix = 0,
		*dstpix = 0,
		pixelRotate = 0,
		alpha = 0,
		redSource = 0,
		greenSource = 0,
		blueSource = 0,
		redAdd = 0,
		greenAdd = 0,
		blueAdd = 0;

	prRect			box;

	filterParamsH	paramsH = 0;

	switch (selector)
	{
	case	fsInitSpec:
		// Since we don't pop a dialog, our response to this
		// "silent setup" selector is the same as our normal
		// setup. Respond to fsInitSpec (if possible) creating
		// a default specsHandle withOUT user intervention.
		result = fsNoErr;

		if ((*theData)->specsHandle)
		{
			// In a filter that has a need for a more complex setup dialog
			// you would present your platform specific user interface here,
			// storing results in the specsHandle (which you've allocated).
		}
		else
		{
			paramsH = (filterParamsH)((*theData)->piSuites->memFuncs->newHandle)(sizeof(filterParams));

			// Memory allocation failed, no need to continue
			if (!paramsH)
			{
				break;
			}
			// Color param is stored as 0x00BBGGRR,
			// unlike pixels which are 0xAARRGGBB
			(*paramsH)->color = 0x00FF80FF;
			(*paramsH)->slider = 0;
			(*paramsH)->checkbox = 0;

			(*theData)->specsHandle = (char**)paramsH;
		}
		break;

		// This sample has no setup dialog, and relies on Premiere
		// to draw the UI for the parameters in the Effect Control panel
	case	fsHasSetupDialog:
		result = fsHasNoSetupDialog;
		break;

	case	fsSetup:
		result = fsNoErr;
		break;

		//	NOTE: During fsCanHandlePAR, OR together a combination
		//	of flags (from PrSDKEffect.h). DO NOT return a filter
		//	error code.  This selector is only sent if
		//	notUnityPixelAspectRatio and notAnyPixelAspectRatio
		//	are set in the PiPL (.r file).  This selector allows
		//	the filter to ask for a unity PAR during setup, and
		//	the native source PAR during execute, or vice versa.
	case fsCanHandlePAR:
		result = prEffectCanHandlePAR;
		break;

		// New selector for Premiere Pro: successively called to get pixel formats
		// supported.  Set pixelFormatSupported to a PrPixelFormat (i.e. fourCC's)
		// defined in PrSDKPixelFormat.h
	case fsGetPixelFormatsSupported:
		result = fsNoErr;

		switch ((*theData)->pixelFormatIndex)
		{
		case 0:
			(*theData)->pixelFormatSupported = PrPixelFormat_BGRA_4444_8u;
			break;

			/*
			If you had more formats, you would add more cases like this:
			case 1:
			(*theData)->pixelFormatSupported =  PrPixelFormat_BGRA_4444_8u;
			break;
			*/

		default:
			result = fsBadFormatIndex;
			break;
		}
		break;

	case fsExecute:
		result = fsNoErr;

		// Get the data from specsHandle
		paramsH = (filterParamsH)(*theData)->specsHandle;

		if (paramsH)
		{
			// Get the frame dimensions
			((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

			// If the checkbox is checked or slider is non-zero, use callback to get the frame/field n frames/fields ahead,
			// where n is the offset specified by the slider
			if (((*paramsH)->checkbox || (*paramsH)->slider) && (*theData)->callBack)
			{
				callbackTime = (*theData)->part + (*theData)->tdb->sampleSize * (*paramsH)->slider;
				(*theData)->callBack(callbackTime, (*theData)->destination, &box, (char **)(*theData)->privateData);
			}

			// Do pixel shifting algorithm
			else
			{
				// Get the color parameter and separate into components
				pixelRotate = (csSDK_uint32)(*paramsH)->color;
				redAdd = pixelRotate & 0x000000ff;
				greenAdd = (pixelRotate & 0x0000ff00) >> 8;
				blueAdd = (pixelRotate & 0x00ff0000) >> 16;

				// Calculate dimensions
				height = box.bottom - box.top;
				width = box.right - box.left;
				rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

				// Create copies of pointer to the source, destination frames
				srcpix = (csSDK_uint32*)((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source);
				dstpix = (csSDK_uint32*)((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination);

				// Loop through each pixel and perform the following:
				// Add B value of color param to B value of pixel
				// Add G value of color param to G value of pixel
				// Add R value of color param to R value of pixel
				// Discard overflow of each addition
				// Do not disturb A
				for (int vert = 0; vert < height; ++vert)
				{
					for (int horiz = 0; horiz < width; ++horiz)
					{
						// Save off alpha
						alpha = *srcpix & 0xff000000;
						// Separate colors
						redSource = (*srcpix & 0x00ff0000) >> 16;
						greenSource = (*srcpix & 0x0000ff00) >> 8;
						blueSource = *srcpix & 0x000000ff;
						// Add param color values and truncate to 8 bits
						redSource = (redSource + redAdd) & 0x000000ff;
						greenSource = (greenSource + greenAdd) & 0x000000ff;
						blueSource = (blueSource + blueAdd) & 0x000000ff;
						// Combine components
						*dstpix = (alpha | (redSource << 16) | (greenSource << 8) | blueSource);

						++dstpix;
						++srcpix;
					}

					srcpix += (rowbytes / 4) - width;
					dstpix += (rowbytes / 4) - width;
				} // else
			} // if (((*paramsH)->sliderS) && (*theData)->callBack)

		} // if (specsH)

		  // No specsHandle?!  How'd that happen?
		  // Oh well, there's nothing we can do at this point.
		else
		{
			result = fsNoErr;
		}
		break;
	}

	// This filter doesn't check for errors.
	// But be sure to return fsUnsupported for unsupported selectors.
	return(result);
}
#endif