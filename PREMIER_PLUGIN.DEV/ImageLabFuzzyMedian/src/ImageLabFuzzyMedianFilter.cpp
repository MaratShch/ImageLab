#include "ImageLabFuzzyMedian.h"
#include "RgbHsvConverts.h"
#include <assert.h> 


void fuzzy_filter_median_3x3
(
	float* __restrict	pBuffer,
	const  csSDK_int32&	width,
	const  csSDK_int32& height
)
{
	if (nullptr == pBuffer)
		return;

	/* currently, lets check in-place processing for avoid additional memory allocation */
	return;
}


bool fuzzy_median_filter_BGRA_4444_8u_frame
(
	const csSDK_uint32* __restrict pSrc,
	csSDK_uint32*       __restrict pDst,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch,
	const AlgMemStorage& algMem
)
{
	const csSDK_int32 memSize = height * width * size_fuzzy_pixel;
	bool bResult = false;

	if (nullptr != algMem.pFuzzyBuffer || memSize > algMem.memSize)
	{
		/* first convert BGR color space to HSV color spase */
		convert_rgb_to_hsv_4444_BGRA8u (pSrc, reinterpret_cast<float*>(algMem.pFuzzyBuffer), width, height, linePitch);

		/* perform fuzzy median filter on V channel */
		fuzzy_filter_median_3x3 (reinterpret_cast<float*>(algMem.pFuzzyBuffer), width, height);

		/* back convert processed buffer from HSV color space to RGB color space */
		convert_hsv_to_rgb_4444_BGRA8u (pSrc, reinterpret_cast<float*>(algMem.pFuzzyBuffer), pDst, width, height, linePitch);
	
		bResult = true;
	}

	return bResult;
}


bool fuzzy_median_filter_ARGB_4444_8u_frame
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32*       __restrict dstPix,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch,
	const AlgMemStorage& algMem
)
{
	return true;
}

