#include "AdobeImageLabHDR.h"


static void computeLumaHistogramFrom_VUYA_4444_8u(const csSDK_uint32* __restrict srcBuffer,
												  int* __restrict pHistogram,
												  const int width,
												  const int height,
												  const int rowBytes)
{
	int i, j;
	int Y1, Y2, Y3, Y4;
	const int widthAligned = width & 0x7FFFFFFC;

	for (i = 0; i < height; i++)
	{
		__VECTOR_ALIGNED__
		for (j = 0; j < widthAligned; j += 4)
		{
			Y1 = (*srcBuffer++ & 0x00FF0000) >> 16;
			Y2 = (*srcBuffer++ & 0x00FF0000) >> 16;
			Y3 = (*srcBuffer++ & 0x00FF0000) >> 16;
			Y4 = (*srcBuffer++ & 0x00FF0000) >> 16;

			pHistogram[Y1]++;
			pHistogram[Y2]++;
			pHistogram[Y3]++;
			pHistogram[Y4]++;
		}
		// fix interlaced mode (fields)
		srcBuffer += (rowBytes / 4) - width;
	}

	return;
}

static void computeHistogramBinarization(const int* __restrict pHist, byte* __restrict pBin, const int histSize, const int left, const int right)
{
	int accumCnt = 0;
	int cnt;

	__VECTOR_ALIGNED__
	// make binarization
	for (cnt = 0; cnt < histSize; cnt++)
	{
		accumCnt += pHist[cnt];
		if (accumCnt < left)
			continue;
		if (accumCnt > right)
			break;

		pBin[cnt] = ((pHist[cnt] == 0) ? 0u : 1u);
	}
	
	return;
}


static void computeCumulativeSum(const byte* __restrict pBin, byte* __restrict pCumSum, const int histSize)
{
	pCumSum[0] = pBin[0];

	__VECTOR_ALIGNED__
	for (int cnt = 1; cnt < histSize; cnt++)
	{
		pCumSum[cnt] = pBin[cnt] + pCumSum[cnt - 1];
	}
	return;
}


static void generateLUT_8u(const byte* __restrict pCumSum, byte* __restrict pLUT, const int size)
{
	const double maxIndex = static_cast<double>(pCumSum[size - 1]); // max cumulative value
	const double lutCoeff = 255.0 / maxIndex;
	double dVal;

	int i;

	__VECTOR_ALIGNED__
	for (i = 0; i < size; i++)
	{
		dVal = lutCoeff * static_cast<double>(pCumSum[i]);
		pLUT[i] = static_cast<byte>(dVal);
	}

	return;
}


static void applyLUTtoVUYA_4444_8u(csSDK_uint32* __restrict srcBuffer,
								   csSDK_uint32* __restrict dstBuffer,
	                                 const byte* __restrict pLUT,
	                                 const int width,
	                                 const int height,
	                                 const int rowBytes)
{
	int i, j;
	byte YInValue;
	byte YOutValue;


	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			YInValue = (*srcBuffer & 0x00FF0000) >> 16;
			YOutValue = pLUT[YInValue];
			*dstBuffer = (*srcBuffer & 0xFF00FFFF) + (YOutValue << 16);
			++srcBuffer;
			++dstBuffer;
		}

		// fix interlaced mode (fields)
		srcBuffer += (rowBytes / 4) - width;
		dstBuffer += (rowBytes / 4) - width;
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
				csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
				csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

				const double totalPixels = static_cast<double>(height * width);
				// get left threshold from slider
				const csSDK_int32 sliderLeftPosition  = (*filterParamH)->sliderLeft;
				const csSDK_int32 sliderRightPosition = (*filterParamH)->sliderRight;

				const int leftCount  = static_cast<int>((totalPixels * static_cast<double>(sliderLeftPosition)) / 1000.0);
				const int rightCount = static_cast<int>(totalPixels - (totalPixels * static_cast<double>(sliderRightPosition)) / 1000.0);

				// cleanup buffer for histogram
				void* __restrict pHist = GetHistogramBuffer();
				void* __restrict pBin  = GetBinarizationBuffer();
				void* __restrict pSum  = GetCumSumBuffer();
				void* __restrict pLut  = GetLUTBuffer();

				if (nullptr != pHist && nullptr != pBin && nullptr != pSum && nullptr != pLut)
				{
					memset(pHist, 0, IMAGE_LAB_HIST_BUFFER_SIZE);
					memset(pBin,  0, IMAGE_LAB_BIN_BUFFER_SIZE);
					memset(pSum,  0, IMAGE_LAB_CUMSUM_BUFFER_SIZE);
					memset(pLut,  0, IMAGE_LAB_LUT_BUFFER_SIZE);

					// compute histogram from Y
					computeLumaHistogramFrom_VUYA_4444_8u(srcPix, reinterpret_cast<int*>(pHist), width, height, rowbytes);

					// make histogtam binarization
					computeHistogramBinarization(reinterpret_cast<int* __restrict>(pHist), reinterpret_cast<byte* __restrict>(pBin), 256, leftCount, rightCount);
					
					// compute cumulative sum
					computeCumulativeSum(reinterpret_cast<byte* __restrict>(pBin), reinterpret_cast<byte* __restrict>(pSum), 256);

					// generate LUT
					generateLUT_8u(reinterpret_cast<byte* __restrict>(pSum), reinterpret_cast<byte* __restrict>(pLut), 256);

					// apply LUT 
					applyLUTtoVUYA_4444_8u(srcPix, dstPix, reinterpret_cast<byte* __restrict>(pLut), width, height, rowbytes);

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
			errCode = fsDoNotCacheOnLoad;
		break;

		default:
			// unhandled case
		break;
		
	}

	return errCode;
}
