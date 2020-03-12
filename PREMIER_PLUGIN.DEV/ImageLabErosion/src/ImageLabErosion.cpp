#include "AdobeImageLabErosion.h"


bool erosion_BGRA_4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	csSDK_uint16 B, G, R;
	csSDK_uint16 newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			newB = newG = newR = 0u;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;

					B = ((srcPix[inPix] & 0x000000FFu));
					G = ((srcPix[inPix] & 0x0000FF00u) >> 8);
					R = ((srcPix[inPix] & 0x00FF0000u) >> 16);

					newB = MAX(newB, B);
					newG = MAX(newG, G);
					newR = MAX(newR, R);
				}
			}

			outPix = j * linePitch + i;
			
			dstPix[outPix] = (srcPix[outPix] & 0xFF000000u) | /* keep Alpha channel */
					(static_cast<csSDK_uint32>(newR) << 16) |
											  (newG  << 8)  |
					                           newB;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool erosion_VUYA_4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	csSDK_uint16 Y, newY;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			newY = 0u;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;
					Y = ((srcPix[inPix] & 0x00FF0000u) >> 16);
					newY = MAX(newY, Y);
				}
			}

			outPix = j * linePitch + i;
			dstPix[outPix] = (srcPix[outPix] & 0xFF00FFFFu) | /* keep Alpha channel */
					(static_cast<csSDK_uint32>(newY) << 16);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool erosion_BGRA_4444_16u
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	csSDK_uint16 B, G, R;
	csSDK_uint16 newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			newB = newG = newR = 0u;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 1;
					inPix = jIdx + iIdx;

					B = ((srcPix[inPix]     & 0x0000FFFFu));
					G = ((srcPix[inPix]     & 0xFFFF0000u) >> 16);
					R = ((srcPix[inPix + 1] & 0x0000FFFFu));

					newB = MAX(newB, B);
					newG = MAX(newG, G);
					newR = MAX(newR, R);
				}
			}

			outPix = j * linePitch + (i << 1);

			dstPix[outPix]     = newB | (newG << 16);
			dstPix[outPix + 1] = newR | (srcPix[outPix + 1] & 0xFFFF0000u);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool erosion_BGRA_4444_32f
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float B, G, R;
	float newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			newB = newG = newR = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 2;
					inPix = jIdx + iIdx;

					B = srcPix[inPix];
					G = srcPix[inPix + 1];
					R = srcPix[inPix + 2];

					newB = MAX(newB, B);
					newG = MAX(newG, G);
					newR = MAX(newR, R);
				}
			}

			outPix = j * linePitch + (i << 2);

			dstPix[outPix]     = newB;
			dstPix[outPix + 1] = newG;
			dstPix[outPix + 2] = newR;
			dstPix[outPix + 3] = srcPix[outPix + 3];

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool erosion_VUYA_4444_32f
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float Y, newY;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			newY = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 2;
					inPix = jIdx + iIdx;
					Y = srcPix[inPix + 2];
					newY = MAX(newY, Y);
				}
			}

			outPix = j * linePitch + (i << 2);

			dstPix[outPix]     = srcPix[outPix];
			dstPix[outPix + 1] = srcPix[outPix + 1];
			dstPix[outPix + 2] = newY;
			dstPix[outPix + 3] = srcPix[outPix + 3];

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool erosion_ARGB_4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	csSDK_uint16 B, G, R;
	csSDK_uint16 newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			newB = newG = newR = 0u;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = MIN(lastPix, MAX(0, m));
					inPix = jIdx + iIdx;

					R = ((srcPix[inPix] & 0x0000FF00u) >> 8);
					G = ((srcPix[inPix] & 0x00FF0000u) >> 16);
					B = ((srcPix[inPix] & 0xFF000000u) >> 24);

					newR = MAX(newR, R);
					newG = MAX(newG, G);
					newB = MAX(newB, B);
				}
			}

			outPix = j * linePitch + i;

			dstPix[outPix] = (srcPix[outPix] & 0x000000FFu) | /* keep Alpha channel */
				                                (newR << 8) |
				    (static_cast<csSDK_uint32>(newG) << 16) |
				    (static_cast<csSDK_uint32>(newB) << 24);

			} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


bool erosion_ARGB_4444_16u
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	csSDK_uint16 B, G, R;
	csSDK_uint16 newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			newB = newG = newR = 0u;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 1;
					inPix = jIdx + iIdx;

					R = ((srcPix[inPix]     & 0xFFFF0000u) >> 16);
					G = ((srcPix[inPix + 1] & 0x0000FFFFu));
					B = ((srcPix[inPix + 1] & 0xFFFF0000u) >> 16);

					newR = MAX(newR, R);
					newG = MAX(newG, G);
					newB = MAX(newB, B);
				}
			}

			outPix = j * linePitch + (i << 1);

			dstPix[outPix]     = (srcPix[outPix] & 0x0000FFFFu) | (static_cast<csSDK_uint32>(newR) << 16);
			dstPix[outPix + 1] = newG | (static_cast<csSDK_uint32>(newB) << 16);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}

bool erosion_ARGB_4444_32f
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
)
{
	const csSDK_int32 winHalfSize = windowSize >> 1;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPix = width - 1;

	csSDK_int32 iIdx, jIdx, lineIdx;
	csSDK_int32 i, j, l, m;
	csSDK_int32 iMin, iMax, jMin, jMax;
	float B, G, R;
	float newB, newG, newR;
	csSDK_uint32 inPix, outPix;

	for (j = 0; j < height; j++)
	{
		jMin = j - winHalfSize;
		jMax = j + winHalfSize;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			iMin = i - winHalfSize;
			iMax = i + winHalfSize;

			newB = newG = newR = 0.0f;

			for (l = jMin; l <= jMax; l++) /* kernel lines */
			{
				lineIdx = MIN(lastLine, MAX(0, l));
				jIdx = lineIdx * linePitch;

				for (m = iMin; m <= iMax; m++) /* kernel rows */
				{
					iIdx = (MIN(lastPix, MAX(0, m))) << 2;
					inPix = jIdx + iIdx;

					R = srcPix[inPix + 1];
					G = srcPix[inPix + 2];
					B = srcPix[inPix + 3];

					newR = MAX(newR, R);
					newG = MAX(newG, G);
					newB = MAX(newB, B);
				}
			}

			outPix = j * linePitch + (i << 2);

			dstPix[outPix]     = srcPix[outPix];
			dstPix[outPix + 1] = newR;
			dstPix[outPix + 2] = newG;
			dstPix[outPix + 3] = newB;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return true;
}


csSDK_int32 selectProcessFunction(VideoHandle theData)
{
	static constexpr char* strPpixSuite = "Premiere PPix Suite";
	SPBasicSuite*		   SPBasic = nullptr;
	csSDK_int32 errCode = fsBadFormatIndex;
	bool processSucceed = true;

	// acquire Premier Suites
	if (nullptr != (SPBasic = (*theData)->piSuites->utilFuncs->getSPBasicSuite()))
	{
		PrSDKPPixSuite*	  PPixSuite = nullptr;
		const SPErr err = SPBasic->AcquireSuite(strPpixSuite, 1l, (const void**)&PPixSuite);

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

			void* __restrict srcImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->source));
			void* __restrict dstImg = reinterpret_cast<void* __restrict>(((*theData)->piSuites->ppixFuncs->ppixGetPixels)((*theData)->destination));
			if (nullptr == srcImg || nullptr == dstImg)
				return fsBadFormatIndex;


			switch (pixelFormat)
			{
					// ============ native AP formats ============================= //
				case PrPixelFormat_BGRA_4444_8u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					processSucceed = erosion_BGRA_4444_8u (pSrcPix, pDstPix, width, height, linePitch, defaultWindowSize);
				}
				break;

				case PrPixelFormat_VUYA_4444_8u:
				case PrPixelFormat_VUYA_4444_8u_709:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					processSucceed = erosion_VUYA_4444_8u (pSrcPix, pDstPix, width, height, linePitch, defaultWindowSize);
				}
				break;

				case PrPixelFormat_BGRA_4444_16u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					processSucceed = erosion_BGRA_4444_16u (pSrcPix, pDstPix, width, height, linePitch, defaultWindowSize);
				}
				break;

				case PrPixelFormat_BGRA_4444_32f:
				{
					const float* __restrict pSrcPix = reinterpret_cast<const float* __restrict>(srcImg);
					float* __restrict pDstPix = reinterpret_cast<float* __restrict>(dstImg);

					processSucceed = erosion_BGRA_4444_32f (pSrcPix, pDstPix, width, height, linePitch, defaultWindowSize);
				}
				break;

				case PrPixelFormat_VUYA_4444_32f:
				case PrPixelFormat_VUYA_4444_32f_709:
				{
					const float* __restrict pSrcPix = reinterpret_cast<const float* __restrict>(srcImg);
					float* __restrict pDstPix = reinterpret_cast<float* __restrict>(dstImg);

					processSucceed = erosion_VUYA_4444_32f (pSrcPix, pDstPix, width, height, linePitch, defaultWindowSize);
				}
				break;

				// ============ native AE formats ============================= //
				case PrPixelFormat_ARGB_4444_8u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					processSucceed = erosion_ARGB_4444_8u (pSrcPix, pDstPix, width, height, linePitch, defaultWindowSize);
				}
				break;

				case PrPixelFormat_ARGB_4444_16u:
				{
					const csSDK_uint32* __restrict pSrcPix = reinterpret_cast<const csSDK_uint32* __restrict>(srcImg);
					csSDK_uint32* __restrict pDstPix = reinterpret_cast<csSDK_uint32* __restrict>(dstImg);

					processSucceed = erosion_ARGB_4444_16u (pSrcPix, pDstPix, width, height, linePitch, defaultWindowSize);
				}
				break;

				case PrPixelFormat_ARGB_4444_32f:
				{
					const float* __restrict pSrcPix = reinterpret_cast<const float* __restrict>(srcImg);
					float* __restrict pDstPix = reinterpret_cast<float* __restrict>(dstImg);

					processSucceed = erosion_ARGB_4444_32f (pSrcPix, pDstPix, width, height, linePitch, defaultWindowSize);
				}
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

// ImageLabHDR filter entry point
PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData)
{
 _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
 _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	csSDK_int32		errCode = fsNoErr;

	switch (selector)
	{
		case fsInitSpec:
		break;

		case fsSetup:
		break;

		case fsExecute:
			errCode = selectProcessFunction(theData);
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
