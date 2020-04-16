#include "ImageLabVideoPuzzle.h"
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>


bool make_mosaic_map (mosaicMap* __restrict pMap, const csSDK_int16 blocksNumber)
{
	if (nullptr == pMap || blocksNumber < minBlocksNumber || blocksNumber > maxBlocksNumber)
		return false;

	std::vector<csSDK_int16> blockVertical;
	std::vector<csSDK_int16> blockHorizontal;
	csSDK_int16 i;

	/* build vector */
	for (i = 0; i < blocksNumber; i++)
	{
		blockVertical.push_back(i);
		blockHorizontal.push_back(i);
	} /* for (i = 0; i < blocksNumber; i++) */

	/* initialize random device */
	std::random_device rd;
	std::mt19937 g(rd()); /* use "mersenne twister" for generate random values */

	/* shuffle numbers */
	std::shuffle(blockVertical.begin(),   blockVertical.end(),   g);
	std::shuffle(blockHorizontal.begin(), blockHorizontal.end(), g);

	for (i = 0; i < blocksNumber; i++)
	{
		pMap[i].mapIdx[lineIdx] = blockHorizontal[i];
		pMap[i].mapIdx[rowIdx]  = blockVertical[i];
	} /* for (i = 0; i < blocksNumber; i++) */

	return true;
}


template <typename T>
bool make_mosaic_image
(
	const T* __restrict srcPix,
	T* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const mosaicMap* __restrict pMosaic,
	const csSDK_int16& blocksNumber
)
{
	T* __restrict srcBlock = nullptr;
	T* __restrict dstBlock = nullptr;
	const csSDK_int32 totalBlocks = blocksNumber * blocksNumber;
	csSDK_int32 i, j, k;

	for (k = 0; k < totalBlocks; k++)
	{

	}

	return true;
}


csSDK_int32 selectProcessFunction (const VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	filterParamsH		   paramsH = nullptr;
	constexpr long         siteVersion = 1l;
	csSDK_int32            errCode = fsBadFormatIndex;
	bool                   processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*	  PPixSuite = nullptr;
		const SPErr err = SPBasic->AcquireSuite(strPpixSuite, siteVersion, (const void**)&PPixSuite);

		if (nullptr != PPixSuite && kSPNoError == err)
		{
			PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
			PPixSuite->GetPixelFormat((*theData)->source, &pixelFormat);

			// Get the frame dimensions
			prRect box = {};
			((*theData)->piSuites->ppixFuncs->ppixGetBounds)((*theData)->destination, &box);

			// Calculate dimensions
			const csSDK_int32 height = box.bottom - box.top;
			const csSDK_int32 width = box.right - box.left;
			const csSDK_int32 linePitch = (((*theData)->piSuites->ppixFuncs->ppixGetRowbytes)((*theData)->destination)) >> 2;

			// Check is frame dimensions are correct
			if (0 >= height || 0 >= width || 0 >= linePitch || linePitch < width)
				return fsBadFormatIndex;

			paramsH = reinterpret_cast<filterParamsH>((*theData)->specsHandle);
			if (nullptr == paramsH)
				return fsBadFormatIndex;

			void* __restrict srcImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
			void* __restrict dstImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
			if (nullptr == srcImg || nullptr == dstImg)
				return fsBadFormatIndex;

			/* get mosaic INFO */
			const csSDK_int16 blocksNumber = (*paramsH)->sliderBlocksNumber;
			mosaicMap* __restrict map = (*paramsH)->map;

			((*paramsH)->frameCnt) += 1;
			if ((*paramsH)->frameCnt > (*paramsH)->sliderFrameDuration)
			{
				(*paramsH)->frameCnt = 0;
				make_mosaic_map (map, blocksNumber);
			}

			switch (pixelFormat)
			{
				// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
				case PrPixelFormat_ARGB_4444_8u:
				case PrPixelFormat_RGB_444_10u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					      csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);
					processSucceed = make_mosaic_image (pSrcPix, pDstPix, width, height, linePitch, map, blocksNumber);
				}
				break;

				case PrPixelFormat_BGRA_4444_16u:
				case PrPixelFormat_ARGB_4444_16u:
				{
					const csSDK_uint64* __restrict pSrcPix = reinterpret_cast<const csSDK_uint64* __restrict>(srcImg);
					      csSDK_uint64* __restrict pDstPix = reinterpret_cast<csSDK_uint64* __restrict>(dstImg);
					processSucceed = make_mosaic_image (pSrcPix, pDstPix, width, height, linePitch, map, blocksNumber);
				}
				break;

				case PrPixelFormat_BGRA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				case PrPixelFormat_ARGB_4444_32f:
				{
					const __m128i* __restrict pSrcPix = reinterpret_cast<const __m128i* __restrict>(srcImg);
					      __m128i* __restrict pDstPix = reinterpret_cast<__m128i* __restrict>(dstImg);
					processSucceed = make_mosaic_image (pSrcPix, pDstPix, width, height, linePitch, map, blocksNumber);
				}
				break;

				default:
					processSucceed = false;
				break;
			}

			SPBasic->ReleaseSuite (strPpixSuite, siteVersion);
			errCode = (true == processSucceed) ? fsNoErr : errCode;
		}
	}

	return errCode;
}


// ImageLabHDR filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	filterParamsH	paramsH = nullptr;
	csSDK_int32		errCode = fsNoErr;

	switch (selector)
	{
		case fsInitSpec:
			if ((*theData)->specsHandle)
			{
				// In a filter that has a need for a more complex setup dialog
				// you would present your platform specific user interface here,
				// storing results in the specsHandle (which you've allocated).
			}
			else
			{
				paramsH = reinterpret_cast<filterParamsH>(((*theData)->piSuites->memFuncs->newHandle)(sizeof(filterParams)));

				// Memory allocation failed, no need to continue
				if (nullptr == paramsH)
					break;

				make_mosaic_map((*paramsH)->map, defBlocksNumber);
				(*paramsH)->sliderBlocksNumber = defBlocksNumber;
				(*paramsH)->sliderFrameDuration = defMosaicMapDuration;
				(*paramsH)->frameCnt = 0;
				(*theData)->specsHandle = reinterpret_cast<char**>(paramsH);
			}
			break;

		case fsSetup:
		break;

		case fsExecute:
			errCode = selectProcessFunction(theData);
		break;

		case fsDisposeData:
			(*theData)->piSuites->memFuncs->disposeHandle((*theData)->specsHandle);
			(*theData)->specsHandle = nullptr;
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
