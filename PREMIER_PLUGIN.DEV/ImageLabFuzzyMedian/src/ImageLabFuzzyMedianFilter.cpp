#include "ImageLabFuzzyMedian.h"
#include "RgbHsvConverts.h"
#include <assert.h> 
#include <float.h>


/* fast SQUARE ROOT COMPUTATION */
static inline float asqrt (const float& x)
{
	const float xHalf = 0.50f * x;
	int   tmp = 0x5F3759DF - (*(int*)&x >> 1); //initial guess
	float xRes = *(float*)&tmp;
	xRes *= (1.50f - (xHalf * xRes * xRes));
	return xRes * x;
}

inline float get_matrix_std
(
	const float* __restrict pBuffer,	/* buffer pointer								*/	
	const csSDK_int32& winSize,			/* size of MATRIX'  window						*/
	const csSDK_int32& winPitch			/* pitch for get next element from next line	*/
)
{
	float fVal, fSum;
	float mean, variance;
	csSDK_int32 k, l;
	const csSDK_int32 winSqSize = winSize * winSize; /* number of elements in MATRIX */

	/* compute MEAN */
	mean = 0.0f;
	for (k = 0; k < winSize; k++)
	{
		const float* __restrict pLine = &pBuffer[k * winPitch];
		for (l = 0; l < winSize; l++)
		{
			fVal = pLine[OFFSET_V(l * 3)]; /* multiple to 3 because we need only V channel in H,S,V buffer layout */
			mean += fVal;
		}
	}
	mean /= winSqSize;

	/* compute VARIANCE */
	fSum = 0.0f;
	for (k = 0; k < winSize; k++)
	{
		const float* __restrict pLine = &pBuffer[k * winPitch];
		for (l = 0; l < winSize; l++)
		{
			/* subtract mean from each element */
			fVal = pLine[OFFSET_V(l * 3)] - mean; /* multiple to 3 because we need only V channel in H,S,V buffer layout */
			/* squaring each element and add to sum */
			fSum += (fVal * fVal);
		}
	}

	variance = (fSum / static_cast<float>(winSqSize - 1));

	return asqrt(variance);
}


float get_min_std
(
	float* __restrict	pBuffer,
	const  csSDK_int32&	width,
	const  csSDK_int32& height
)
{
	constexpr csSDK_int32 winSize = 8;
	const csSDK_int32 widthMax  = width  - winSize;
	const csSDK_int32 heightMax = height - winSize;
	csSDK_int32 i, j, idx = 0;
	float fStd;
	float fStdMin = FLT_MAX;

	// test matrix STD: expected value 2.73 (from Matlab)
	// const float fff[9] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
	// fStd = get_matrix_std(fff, 3, 3);

	const csSDK_int32 linePitch = width * 3;

	for (j = 0; j < heightMax; j += winSize)
	{
		for (i = 0; i < widthMax; i += winSize)
		{
			idx = j * linePitch + i * 3;
			const float* __restrict pMatrix = &pBuffer[idx];

			fStd = get_matrix_std(pMatrix, winSize, linePitch);
			fStdMin = ((0 == fStd) ? fStdMin : MIN(fStdMin, fStd));
		} /* for (i = 0; i < widthMax; i += winSize) */

	} /* for (j = 0; j < heightMax; j += winSize) */

	return fStdMin;
}


void fuzzy_filter_median_3x3
(
	float* __restrict	pBuffer,
	const  csSDK_int32&	width,
	const  csSDK_int32& height
)
{
	/* get STD */
	const float fImgStdMin = get_min_std(pBuffer, width, height);
	const float p = 6 * fImgStdMin;
	const csSDK_int32 maxPix = width - 1;
	const csSDK_int32 maxLine = height - 1;
	csSDK_int32 i, j;

	/* currently, lets check in-place processing for avoid additional memory allocation */
	for (j = 1; j < maxLine; j++)
	{
		for (i = 1; i < maxLine; i++)
		{

		}

	} /* for (j = 1; j < maxLine; j++) */
	
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

