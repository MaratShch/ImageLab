#include "ImageLabProcessTmpBuffer.h"
#include "ImageLabColorConvert.h"

#ifndef OFFSET_B32f
 #define OFFSET_B32f(idx) ((idx) * 4)
#endif

#ifndef OFFSET_G32f
 #define OFFSET_G32f(idx) ((idx) * 4 + 1)
#endif

#ifndef OFFSET_R32f
 #define OFFSET_R32f(idx) ((idx) * 4 + 2)
#endif

#ifndef OFFSET_A32f
 #define OFFSET_A32f(idx) ((idx) * 4 + 3)
#endif


static inline void process_BGRA_4444_32f_buffer
(
	const float* __restrict pSrc,
	      float* __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          noiseLevel,
	const float&          timeStep
)
{
	csSDK_int32 i, j;
	csSDK_int32 k1, k2, pixIdx, dstIdx;
	const csSDK_int32 lastLine  = height - 1;
	const csSDK_int32 lastPixel = width - 1;

	float north, west, east, south, current;
	float diffNorth, diffWest, diffEast, diffSouth;
	float sum;
	float B, G, R;

	const float* __restrict pRgb2Yuv = (width > 800 ? coeff_RGB2YUV[BT709] : coeff_RGB2YUV[BT601]);
	dstIdx = pixIdx = 0;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		k1 = MAX(0, j - 1);
		k2 = MIN(lastLine, j + 1);

		const float* prevLine = &pSrc[k1 * linePitch];
		const float* currLine = &pSrc[j  * linePitch];
		const float* nextLine = &pSrc[k2 * linePitch];

		for (i = 0; i < width; i++)
		{
			// north 
			B = prevLine[OFFSET_B32f(i)];
			G = prevLine[OFFSET_G32f(i)];
			R = prevLine[OFFSET_R32f(i)];
			/* Y value of north pixel*/
			north = 255.0f * (R * pRgb2Yuv[0] + G * pRgb2Yuv[1] + B * pRgb2Yuv[2]);

			// west
			pixIdx = MAX(0, i - 1);
			B = currLine[OFFSET_B32f(pixIdx)];
			G = currLine[OFFSET_G32f(pixIdx)];
			R = currLine[OFFSET_R32f(pixIdx)];
			/* Y value of west pixel */
			west = 255.0f * (R * pRgb2Yuv[0] + G * pRgb2Yuv[1] + B * pRgb2Yuv[2]);

			// current
			B = currLine[OFFSET_B32f(i)];
			G = currLine[OFFSET_G32f(i)];
			R = currLine[OFFSET_R32f(i)];
			/* Y Value of current pixel */
			current = 255.0f * (R * pRgb2Yuv[0] + G * pRgb2Yuv[1] + B * pRgb2Yuv[2]);

			// east
			pixIdx = MIN(lastPixel, i + 1);
			B = currLine[OFFSET_B32f(pixIdx)];
			G = currLine[OFFSET_G32f(pixIdx)];
			R = currLine[OFFSET_R32f(pixIdx)];
			/* Y value of east pixel */
			east = 255.0f * (R * pRgb2Yuv[0] + G * pRgb2Yuv[1] + B * pRgb2Yuv[2]);

			// south
			B = nextLine[OFFSET_B32f(i)];
			G = nextLine[OFFSET_G32f(i)];
			R = nextLine[OFFSET_R32f(i)];
			/* Y value of south pixel */
			south = 255.0f * (R * pRgb2Yuv[0] + G * pRgb2Yuv[1] + B * pRgb2Yuv[2]);

			diffNorth = north - current;
			diffWest  = west  - current;
			diffEast  = east  - current;
			diffSouth = south - current;

			sum = g_function (diffNorth, noiseLevel) * diffNorth +
			      g_function (diffWest,  noiseLevel) * diffWest  +
			      g_function (diffEast,  noiseLevel) * diffEast  +
			      g_function (diffSouth, noiseLevel) * diffSouth;

			// save to DST new Y value only
			pDst[dstIdx] = current + sum * timeStep;
			dstIdx++;
		}
	}
	return;
}


static inline void process_BGRA_4444_32f_buffer
(
	const float* __restrict pSrc1, /* get SRC buffer for put ALPHA values to destination */
	      float* __restrict pSrc2, /* contains only Y value */
	      float* __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          noiseLevel,
	const float&          timeStep
)
{
	csSDK_int32 i, j;
	csSDK_int32 k1, k2, origIdx;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPixel = width - 1;

	float north, west, east, south, current;
	float diffNorth, diffWest, diffEast, diffSouth;
	float sum;
	float Y, U, V;
	float B, G, R;

	const float* __restrict pRgb2Yuv = (width > 800 ? coeff_RGB2YUV[BT709] : coeff_RGB2YUV[BT601]);
	const float* __restrict pYuv2Rgb = (width > 800 ? coeff_YUV2RGB[BT709] : coeff_YUV2RGB[BT601]);

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		k1 = MAX(0, j - 1);
		k2 = MIN(lastLine, j + 1);

		const float* prevLine =    &pSrc2[k1 * width];
		const float* currLine =    &pSrc2[j  * width];
		const float* currSrcLine = &pSrc1[j  * linePitch];
		const float* nextLine =    &pSrc2[k2 * width];

		for (i = 0; i < width; i++)
		{
			north   = prevLine[i];
			west    = currLine[MAX(0, i - 1)];
			current = currLine[i];
			east    = currLine[MIN(lastPixel, i + 1)];
			south   = nextLine[i];

			diffNorth = north - current;
			diffWest  = west  - current;
			diffEast  = east  - current;
			diffSouth = south - current;

			sum = g_function (diffNorth, noiseLevel) * diffNorth +
			      g_function (diffWest,  noiseLevel) * diffWest  +
			      g_function (diffEast,  noiseLevel) * diffEast  +
			      g_function (diffSouth, noiseLevel) * diffSouth;

			/* re-compute Y value */
			Y = (current + sum * timeStep) / 255.f;

			/* read original R,G,B values from source buffer */
			B = currSrcLine[OFFSET_B32f(i)];
			G = currSrcLine[OFFSET_G32f(i)];
			R = currSrcLine[OFFSET_R32f(i)];

			/* restore U and V bands from source buffer */
			U = R * pRgb2Yuv[3] + G * pRgb2Yuv[4] + B * pRgb2Yuv[5];
			V = R * pRgb2Yuv[6] + G * pRgb2Yuv[7] + B * pRgb2Yuv[8];

			/* re-compute R,G,B values from new Y and restored U and V bands */
			R = Y * pYuv2Rgb[0] + U * pYuv2Rgb[1] + V * pYuv2Rgb[2];
			G = Y * pYuv2Rgb[3] + U * pYuv2Rgb[4] + V * pYuv2Rgb[5];
			B = Y * pYuv2Rgb[6] + U * pYuv2Rgb[7] + V * pYuv2Rgb[8];

			origIdx = j * linePitch + i * 4;

			/* save to destination buffer */
			pDst[origIdx    ] = CLAMP_32F(B);		
			pDst[origIdx + 1] = CLAMP_32F(G);
			pDst[origIdx + 2] = CLAMP_32F(R);
			pDst[origIdx + 3] = pSrc1[origIdx + 3];	/* copy A value from SRC */
		}
	}

	return;
}


void process_BGRA_4444_32f_buffer
(
	const float*  __restrict pSrc,
	const AlgMemStorage* __restrict pTmpBuffers,
	      float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          dispersion,
	const float&          timeStep,
	const float&          noiseLevel
)
{
	float* __restrict pBuffers[2]
	{
		reinterpret_cast<float* __restrict>(pTmpBuffers->pTmp1),
		reinterpret_cast<float* __restrict>(pTmpBuffers->pTmp2)
	};

	const float fDispersion { dispersion };
	const float fTimeStep   { timeStep   };
	const float fNoiseLevel { noiseLevel };

	float currentDispersion = 0.0f;
	float currentTimeStep = MIN(fTimeStep, fDispersion - currentDispersion);
	constexpr float minimalStep = 0.001f;

	csSDK_uint32 ping, pong;
	csSDK_int32 iterCnt;

	iterCnt = 0;
	ping = 0u;
	pong = ping ^ 0x1u;

	do
	{
		if (0 == iterCnt)
		{
			process_BGRA_4444_32f_buffer (pSrc, pBuffers[ping], width, height, linePitch, fNoiseLevel, currentTimeStep);
		}
		else if (currentDispersion + fTimeStep < fDispersion)
		{
			process_float_raw_buffer (pBuffers[ping], pBuffers[pong], width, height, fNoiseLevel, currentTimeStep);
			ping ^= 0x1;
			pong ^= 0x1;
		}
		else
		{
			process_BGRA_4444_32f_buffer (pSrc, pBuffers[ping], pDst, width, height, linePitch, fNoiseLevel, currentTimeStep);
		}

		iterCnt++;
		currentDispersion += currentTimeStep; 
		currentTimeStep = MIN(fTimeStep, fDispersion - currentDispersion);

	} while (currentDispersion <= fDispersion && currentTimeStep > minimalStep);

	return;
}


