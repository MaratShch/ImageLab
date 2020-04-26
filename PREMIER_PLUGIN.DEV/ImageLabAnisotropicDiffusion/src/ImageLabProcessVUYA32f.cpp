#include "ImageLabProcessTmpBuffer.h"

#ifndef OFFSET_Y32f
 #define OFFSET_Y32f(idx) ((idx) * 4 + 2)
#endif

static inline void process_VUYA_4444_32f_buffer
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
			north = 255.0f * prevLine[OFFSET_Y32f(i)];

			// west
			pixIdx = OFFSET_Y32f(MAX(0, i - 1));
			west   = 255.0f * currLine[pixIdx];

			// current
			pixIdx  = OFFSET_Y32f(i);
			current = 255.0f * currLine[pixIdx];

			// east
			pixIdx = OFFSET_Y32f(MIN(lastPixel, i + 1));
			east = 255.0f * currLine[pixIdx];

			// south
			south = 255.0f * nextLine[OFFSET_Y32f(i)];

			diffNorth = north - current;
			diffWest  = west  - current;
			diffEast  = east  - current;
			diffSouth = south - current;

			sum = g_function (diffNorth, noiseLevel) * diffNorth +
			      g_function (diffWest,  noiseLevel) * diffWest  +
			      g_function (diffEast,  noiseLevel) * diffEast  +
			      g_function (diffSouth, noiseLevel) * diffSouth;

			// save to DST new Y value only
			pDst[dstIdx] = CLAMP_U8(current + sum * timeStep);
			dstIdx++;
		}
	}
	return;
}


static inline void process_VUYA_4444_32f_buffer
(
	const float* __restrict pSrc1, /* get SRC buffer for put U,V and ALPHA values to destination */
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
	float Y;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		k1 = MAX(0, j - 1);
		k2 = MIN(lastLine, j + 1);

		const float* prevLine = &pSrc2[k1 * width];
		const float* currLine = &pSrc2[j  * width];
		const float* nextLine = &pSrc2[k2 * width];

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

			Y = CLAMP_32F((current + sum * timeStep) / 255.0f);
			origIdx = j * linePitch + i * 4;

			pDst[origIdx    ] = pSrc1[origIdx];     /* copy V value from SRC */
			pDst[origIdx + 1] = pSrc1[origIdx + 1]; /* copy U value from SRC */
			pDst[origIdx + 2] = Y;					/* copy computed Y value */
			pDst[origIdx + 3] = pSrc1[origIdx + 3];	/* copy A value from SRC */
		}
	}

	return;
}


void process_VUYA_4444_32f_buffer
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
			process_VUYA_4444_32f_buffer (pSrc, pBuffers[ping], width, height, linePitch, fNoiseLevel, currentTimeStep);
		}
		else if (currentDispersion + fTimeStep < fDispersion)
		{
			process_float_raw_buffer (pBuffers[ping], pBuffers[pong], width, height, fNoiseLevel, currentTimeStep);
			ping ^= 0x1;
			pong ^= 0x1;
		}
		else
		{
			process_VUYA_4444_32f_buffer (pSrc, pBuffers[ping], pDst, width, height, linePitch, fNoiseLevel, currentTimeStep);
		}

		iterCnt++;
		currentDispersion += currentTimeStep; 
		currentTimeStep = MIN(fTimeStep, fDispersion - currentDispersion);

	} while (currentDispersion <= fDispersion && currentTimeStep > minimalStep);

	return;
}


