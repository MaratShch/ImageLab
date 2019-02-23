#include "AdobeImageLabHDR.h"


static void computeLumaHistogramFrom_VUYA_4444_8u(const csSDK_uint32* __restrict srcBuffer, int* __restrict pHistogram, const int size)
{
	unsigned int Y1, Y2, Y3, Y4;
	const int size_aligned = size & 0x7FFFFFFC;

	for (int cnt = 0; cnt < size_aligned; cnt += 4)
	{
		Y1 = (srcBuffer[cnt]     & 0x00FF0000) >> 16;
		Y2 = (srcBuffer[cnt + 1] & 0x00FF0000) >> 16;
		Y3 = (srcBuffer[cnt + 2] & 0x00FF0000) >> 16;
		Y4 = (srcBuffer[cnt + 3] & 0x00FF0000) >> 16;

		pHistogram[Y1]++;
		pHistogram[Y2]++;
		pHistogram[Y3]++;
		pHistogram[Y4]++;
	}

	return;
}

static void computeHistogramBinarization(const int* __restrict pHist, byte* __restrict pBin, const int histSize, const int left, const int right)
{
	int accumCnt = 0;
	int cnt;

	// make binarization
	for (cnt = 0; cnt < histSize; cnt++)
	{
		accumCnt += pHist[cnt];
		if (accumCnt < left)
			continue;
		if (accumCnt > right)
			break;

		pBin[cnt] = ((pHist[cnt] == 0) ? 0 : 1);
	}

	// make cumulative SUM
	for (cnt = 1; cnt < histSize; cnt++)
	{
		pBin[cnt] = pBin[cnt] + pBin[cnt - 1];
	}
}


static void generateLUT_8u(const byte* __restrict pCumSum, byte* __restrict pLUT, const int size)
{
	const double maxIndex = pCumSum[size - 1]; // max cumulative value
	const double lutCoeff = 255.0 / maxIndex;
	double dVal;

	int i;

	for (i = 0; i < size; i++)
	{
		dVal = lutCoeff * static_cast<double>(pCumSum[i]);
		pLUT[i] = static_cast<byte>(dVal);
	}

	return;
}


static void applyLUTtoVUYA_4444_8u(const csSDK_uint32* __restrict srcBuffer,
								         csSDK_uint32* __restrict dstBuffer,
	                                       const byte* __restrict pLUT,
	                                                    const int size)
{
	int i;
	byte YInValue;
	byte YOutValue;

	for (i = 0; i < size; i++)
	{
		YInValue = (srcBuffer[i] & 0x00FF0000) >> 16;
		YOutValue = pLUT[YInValue];
		dstBuffer[i] = (srcBuffer[i] & 0xFF00FFFF) + (YOutValue << 16);
	}

	return;
}

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

				const int leftCount  = static_cast<int>((totalPixels * static_cast<double>(sliderLeftPosition)) / 100.0);
				const int rightCount = static_cast<int>(totalPixels - (totalPixels * static_cast<double>(sliderRightPosition)) / 100.0);

				// cleanup buffer for histogram
				void* pHist = GetHistogramBuffer();
				void* pBin  = GetBinarizationBuffer();
				void* pLut  = GetLUTBuffer();

				if (nullptr != pHist && nullptr != pBin && nullptr != pLut)
				{
					memset(pHist, 0, IMAGE_LAB_HIST_BUFFER_SIZE);
					memset(pLut,  0, IMAGE_LAB_LUT_BUFFER_SIZE);
					memset(pBin,  0, IMAGE_LAB_BIN_BUFFER_SIZE);

					// handle fields mode !!!

					// compute histogram from Y
					computeLumaHistogramFrom_VUYA_4444_8u(srcPix, reinterpret_cast<int*>(pHist), static_cast<int>(totalPixels));

					// make histogtam binarization
					computeHistogramBinarization(reinterpret_cast<int*>(pHist), reinterpret_cast<byte*>(pBin), 256, leftCount, rightCount);
					
					// generate LUT
					generateLUT_8u(reinterpret_cast<byte*>(pBin), reinterpret_cast<byte*>(pLut), 256);

					// apply LUT 
					applyLUTtoVUYA_4444_8u(srcPix, dstPix, reinterpret_cast<byte*>(pLut), static_cast<int>(totalPixels));

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
