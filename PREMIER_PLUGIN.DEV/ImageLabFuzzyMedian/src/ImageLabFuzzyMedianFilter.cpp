#include "ImageLabFuzzyMedian.h"
#include "RgbHsvConverts.h"
#include <assert.h> 
#include <float.h>

#ifdef _DEBUG
static float fDbgStd[2048] = {};
static unsigned int dbgCnt = 0u;
static float* pInput = nullptr;
#endif


/* fast SQUARE ROOT COMPUTATION */
static inline float asqrt (const float& x)
{
	const float xHalf = 0.50f * x;
	int   tmp = 0x5F3759DF - (*(int*)&x >> 1); //initial guess
	float xRes = *(float*)&tmp;
	xRes *= (1.50f - (xHalf * xRes * xRes));
	return xRes * x;
}

static inline float get_matrix_std
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
	/* use fast square root computation for STD */
	return asqrt(variance);
}


float get_min_std
(
	float* __restrict	pBuffer,
	const  csSDK_int32&	width,
	const  csSDK_int32& height,
	const  csSDK_int32& linePitch
)
{
	constexpr csSDK_int32 winSize = 8;
	const csSDK_int32 widthMax  = width  - winSize;
	const csSDK_int32 heightMax = height - winSize;
	csSDK_int32 i, j, idx = 0;
	float fStd;
	float fStdMin = FLT_MAX;

	// test matrix STD: expected value 2.73 (from Matlab)
	// constexpr float H = 0.f, S = 0.f, V = 1.0f ... 9.0f;
	// constexpr float fff[27] = { H, S, 1.0f, H, S, 2.0f, H, S, 3.0f, H, S, 4.0f, H, S, 5.0f, H, S, 6.0f, H, S, 7.0f, H, S, 8.0f, H, S, 9.0f };
	// fStd = get_matrix_std(fff, 3, 9);
	//
#ifdef _DEBUG
	dbgCnt = 0u;
#endif

	for (j = 0; j < heightMax; j += winSize)
	{
		const float* pLine = &pBuffer[j * linePitch];

		for (i = 0; i < widthMax; i += winSize)
		{
			fStd = get_matrix_std(&pLine[i * 3], winSize, linePitch);
#ifdef _DEBUG
			fDbgStd[dbgCnt] = fStd;
			dbgCnt++;
#endif
			fStdMin = ((0.0f == fStd) ? fStdMin : MIN(fStdMin, fStd));

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
	float* fPixel = nullptr;
	csSDK_int32 i, j;
	const csSDK_int32 maxPix  = width - 2;
	const csSDK_int32 maxLine = height - 2;

	csSDK_int32 idxNE, idxE, idxS, idxW;
	csSDK_int32 idxK11, idxK12, idxK13, idxK21, idxK22, idxK23, idxK31, idxK32, idxK33;
	float uu, dd, ll, rr, mm, corr, diff;
	float a, b, c, d;
	float dA, dB, dC;

#ifdef _DEBUG
	pInput = pBuffer;
#endif

	const csSDK_int32 linePitch = width * 3;

	/* get STD */
	const float fImgStdMin = get_min_std(pBuffer, width, height, linePitch);
	const float p = 6 * fImgStdMin;

	/* currently, lets check in-place processing for avoid additional memory allocation */
	for (j = 2; j < maxLine; j++)
	{
		for (i = 2; i < maxPix; i++)
		{
			corr = 0.f;

			/* index of Pixel [0, 0] inside of kernel */
			idxK11 = OFFSET_V(j * linePitch + i * 3);
			/* index of Pixel [0, 1] inside of kernel */
			idxK12 = idxK11 + 3;
			/* index of Pixel [0, 2] inside of kernel */
			idxK13 = idxK11 + 6;
			/* index of Pixel [1, 0] inside of kernel */
			idxK21 = OFFSET_V((j + 1) * linePitch + i * 3);
			/* index of Pixel [1, 1] inside of kernel */
			idxK22 = idxK21 + 3;
			/* index of Pixel [1, 2] inside of kernel */
			idxK23 = idxK21 + 6;
			/* index of Pixel [2, 0] insode of kernel */
			idxK31 = OFFSET_V((j + 2) * linePitch + i * 3);
			/* index of Pixel [2, 1] insode of kernel */
			idxK32 = idxK31 + 3;
			/* index of Pixel [2, 2] insode of kernel */
			idxK33 = idxK31 + 6;


			/* index of Norht-East pixel outside of kernel*/
			idxNE = OFFSET_V((j - 2) * linePitch + (i - 2) * 3);
			/* index of East pixel outside of kernel */
			idxE  = OFFSET_V(j * linePitch + (i - 2) * 3);
			/* index of South pixel outside of kernel */
			idxS  = OFFSET_V((j + 2) * linePitch + i * 3);
			/* index of West pixel outside of kernel */
			idxW = OFFSET_V(j + linePitch + (i + 2) * 3);

			uu  = pBuffer[idxNE];
			ll  = pBuffer[idxW];
			dd  = pBuffer[idxS];
			rr  = pBuffer[idxE];

			mm = (pBuffer[idxK11] + pBuffer[idxK12] + pBuffer[idxK13] + pBuffer[idxK21]) / 4.0f;

		/*	
				|
				|->
				|
		*/
			a = pBuffer[idxK12];
			b = pBuffer[idxK22];
			c = pBuffer[idxK23];

			dA = a - pBuffer[idxK13];
			dB = b - pBuffer[idxK23];
			dC = c - pBuffer[idxK33];

			d = (MIN(abs(dA), MIN(abs(dB), abs(dC)))) / p;
			d = MIN(d, 1.f);
			d = 1.0f - d;

			diff = pBuffer[idxK23] - b;
			corr = corr + d * diff;

		/*
				|
			  <-|
				|
		*/
			dA = a - pBuffer[idxK11];
			dB = b - pBuffer[idxK21];
			dC = c - pBuffer[idxK31];

			d = (MIN(abs(dA), MIN(abs(dB), abs(dC)))) / p;
			d = MIN(d, 1.f);
			d = 1.0f - d;
			
			diff = pBuffer[idxK21] - b;
			corr = corr + d * diff;

		/*
			     ^ 
			     |
			    ___

		*/
			a = pBuffer[idxK21];
			b = pBuffer[idxK22];
			c = pBuffer[idxK23];

			dA = a - pBuffer[idxK11];
			dB = b - pBuffer[idxK12];
			dC = c - pBuffer[idxK13];

			d = (MIN(abs(dA), MIN(abs(dB), abs(dC)))) / p;
			d = MIN(d, 1.f);
			d = 1.0f - d;

			diff = pBuffer[idxK12] - b;
			corr = corr + d * diff;

		/*
			   ___	
				|
			    V 

		*/
			dA = a - pBuffer[idxK31];
			dB = b - pBuffer[idxK32];
			dC = c - pBuffer[idxK33];

			d = (MIN(abs(dA), MIN(abs(dB), abs(dC)))) / p;
			d = MIN(d, 1.f);
			d = 1.0f - d;

			diff = pBuffer[idxK32] - b;
			corr = corr + d * diff;

		/*
			  \  
             <-\
			    \
		*/
			a = pBuffer[idxK11];
			b = pBuffer[idxK22];
			c = pBuffer[idxK33];

			dA = a - ll;
			dB = b - pBuffer[idxK31];
			dC = c - dd;

			d = (MIN(abs(dA), MIN(abs(dB), abs(dC)))) / p;
			d = MIN(d, 1.f);
			d = 1.0f - d;
	
			diff = pBuffer[idxK31] - b;
			corr = corr + d * diff;

		/*
	         \
			  \->
			   \
		*/
			dA = a - uu;
			dB = b - pBuffer[idxK13];
			dC = c - rr;

			d = (MIN(abs(dA), MIN(abs(dB), abs(dC)))) / p;
			d = MIN(d, 1.f);
			d = 1.0f - d;

			diff = pBuffer[idxK13] - b;
			corr = corr + d * diff;

		/*
		        /		
			 <-/ 
			  /
		*/

			a = pBuffer[idxK13];
			b = pBuffer[idxK22];
			c = pBuffer[idxK31];

			dA = a - uu;
			dB = b - pBuffer[idxK11];
			dC = c - ll;

			d = (MIN(abs(dA), MIN(abs(dB), abs(dC)))) / p;
			d = MIN(d, 1.f);
			d = 1.0f - d;

			diff = pBuffer[idxK11] - b;
			corr = corr + d * diff;

		/*
			   /
			  /->
			 /
		*/
			dA = a - rr;
			dB = b - pBuffer[idxK33];
			dC = c - dd;

			d = (MIN(abs(dA), MIN(abs(dB), abs(dC)))) / p;
			d = MIN(d, 1.f);
			d = 1.0f - d;

			diff = pBuffer[idxK33] - b;
			corr = corr + d * diff;

			pBuffer[OFFSET_V(j * linePitch + i * 3)] += corr / 8.0f;
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

