#include "ImageLabBilateral.h"
#include <windows.h>

CACHE_ALIGN static float gMesh[11][11] = { 0 };

void gaussian_weights(const float sigma, const int radius /* radius size in range of 3 to 10 */)
{
	int i, j;
	int x, y;

	const int size = 11;
	const float divider = 2.0f * (sigma * sigma); // 2 * sigma ^ 2

	__VECTOR_ALIGNED__
	for (y = -radius, j = 0; j < size; j++, y++)
	{
		for (x = -radius, i = 0; i < size; i++, x++)
		{
			const float dSum = static_cast<float>((x * x) + (y * y));
			gMesh[j][i] = aExp(-dSum / divider);
		}
	}

	return;
}


bool process_VUYA_4444_8u_frame (const VideoHandle theData, const int radius)
{
	CACHE_ALIGN float pF[maxWinSize][maxWinSize] = {};
	CACHE_ALIGN float pH[maxWinSize][maxWinSize] = {};

	prRect box = { 0 };

	// Get the frame dimensions
	((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

	// Calculate dimensions
	const csSDK_int32 height = box.bottom - box.top;
	const csSDK_int32 width = box.right - box.left;
	const csSDK_int32 rowbytes = ((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination);

	const int lastLineIdx  = height - 1;
	const int lastPixelIdx = width - 1;
	const int linePitch = rowbytes >> 2;

	// Create copies of pointer to the source, destination frames
	const csSDK_uint32* __restrict srcPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
	      csSDK_uint32* __restrict dstPix = reinterpret_cast<csSDK_uint32* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));

	constexpr float sigma = 0.10f;
	constexpr float divider = 2.00f * sigma * sigma;

	int i, j, k, l;
	float Y, dY, normF, bSum;
	int finalY;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			int iMin, iMax, jMin, jMax;

			// define processing window coordinates
			iMin = MAX(i - radius, 0);
			iMax = MIN(i + radius, lastPixelIdx);
			jMin = MAX(j - radius, 0);
			jMax = MIN(j + radius, lastLineIdx);

			// define process window sizes
			const int jDiff = (jMax - jMin) + 1;
			const int iDiff = (iMax - iMin) + 1;

			const int pixOffset = j * linePitch + i;
			const csSDK_uint32 inPixel = srcPix[pixOffset];

			const float Yref = static_cast<float>((inPixel & 0x00FF0000u) >> 16);

			// compute Gaussian intensity weights
			for (k = 0; k < jDiff; k++)
			{
				const int jIdx = (jMin + k) * linePitch + iMin;

				for (l = 0; l < iDiff; l++)
				{
					Y = static_cast<float>((srcPix[jIdx + l] & 0x00FF0000u) >> 16);
					dY = Y - Yref;
					pH[k][l] = aExp(-dY / divider); // pH[0][3] incorrect!!!!
				} // for (m = 0; m < iDiff; m++)

			} // for (k = 0; k < jDiff; k++)


			// calculate Bilateral Filter responce
			normF = 0.0f;
			int jIdx, iIdx;

			jIdx = jMin - j + radius;

			for (k = 0; k < jDiff; k++)
			{
				iIdx = iMin - i + radius;
				__VECTOR_ALIGNED__
				for (l = 0; l < iDiff; l++)
				{
					pF[k][l] = pH[k][l] * gMesh[jIdx][iIdx];
					normF += pF[k][l];
					iIdx++;
				}
				jIdx++;
			}

			
			bSum = 0.0f;

			for (k = 0; k < jDiff; k++)
			{
				const int kIdx = (jMin + k) * linePitch + iMin;
				for (l = 0; l < iDiff; l++)
				{
					Y = static_cast<float>((srcPix[kIdx + l] & 0x00FF0000u) >> 16);
					bSum += (pF[k][l] * Y);
				}
			}

			// compute destination pixel
			finalY = CLAMP_U8(static_cast<int>(bSum / normF));
			dstPix[pixOffset] = (srcPix[pixOffset] & 0xFF00FFFFu) | (finalY << 16);

		} // for (i = 0; i < width; i++)

	} // for (j = 0; j < height; j++)

	return true;
}



csSDK_int32 selectProcessFunction (const VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	csSDK_int32 errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*			PPixSuite = nullptr;
		SPBasic->AcquireSuite (strPpixSuite, 1, (const void**)&PPixSuite);

		if (nullptr != PPixSuite)
		{
			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				break;

				case PrPixelFormat_VUYA_4444_8u:
					processSucceed = process_VUYA_4444_8u_frame (theData);
				break;

				case PrPixelFormat_VUYA_4444_8u_709:
				break;

				case PrPixelFormat_BGRA_4444_16u:
				break;

				case PrPixelFormat_BGRA_4444_32f:
				break;

				case PrPixelFormat_VUYA_4444_32f:
				break;

				case PrPixelFormat_VUYA_4444_32f_709:
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
				break;

				case PrPixelFormat_ARGB_4444_16u:
				break;

				case PrPixelFormat_ARGB_4444_32f:
				break;

				// =========== miscellanous formats =========================== //
				case PrPixelFormat_RGB_444_10u:
				break;

				default:
					processSucceed = false;
				break;
			}

			errCode = (true == processSucceed) ? fsNoErr : errCode;
		}
	}

	return errCode;
}


// Bilateral-RGB filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	csSDK_int32 errCode = fsNoErr;

	switch (selector)
	{
		case fsInitSpec:
		break;

		case fsHasSetupDialog:
			errCode = fsHasNoSetupDialog;
		break;

		case fsSetup:
		break;

		case fsExecute:
			errCode = selectProcessFunction (theData);
		break;

		case fsDisposeData:
		break;
		
		case fsCanHandlePAR:
			errCode = prEffectCanHandlePAR;
		break;

		case fsGetPixelFormatsSupported:
			errCode = imageLabPixelFormatSupported (theData);
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