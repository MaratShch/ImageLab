#include "ImageLabProcessTmpBuffer.h"

static inline void process_VUYA_4444_8u_buffer
(
	const csSDK_uint32*  __restrict pSrc,
	      float*         __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          noiseLevel,
	const float&          timeStep,
	const csSDK_int16&    gAdvanced
)
{
	csSDK_int32 i, j;
	csSDK_int32 k1, k2, lineIdx, dstIdx;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPixel = width - 1;
	csSDK_int32 north, west, east, south, current;

	float diffNorth, diffWest, diffEast, diffSouth;
	float sum;

	dstIdx = 0;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		k1 = MAX(0, j - 1);
		k2 = MIN(lastLine, j + 1);

		const csSDK_uint32* prevLine = &pSrc[k1 * linePitch];
		const csSDK_uint32* nextLine = &pSrc[k2 * linePitch];

		lineIdx = j * linePitch;

		for (i = 0; i < width; i++)
		{
			north   = (prevLine[i] & 0x00FF0000) >> 16;
			west    = (pSrc[lineIdx + MAX(0, i - 1)] & 0x00FF0000) >> 16;
			current = (pSrc[lineIdx + i] & 0x00FF0000) >> 16;
			east    = (pSrc[lineIdx + MIN(lastPixel, i + 1)] & 0x00FF0000) >> 16;
			south   = (nextLine[i] & 0x00FF0000) >> 16;

			diffNorth = static_cast<float>(north - current);
			diffWest  = static_cast<float>(west  - current);
			diffEast  = static_cast<float>(east  - current);
			diffSouth = static_cast<float>(south - current);

			if (gAdvanced)
				sum = g_function_advanced(diffNorth, noiseLevel) * diffNorth +
				      g_function_advanced(diffWest,  noiseLevel) * diffWest  +
				      g_function_advanced(diffEast,  noiseLevel) * diffEast  +
				      g_function_advanced(diffSouth, noiseLevel) * diffSouth;
			else
				sum = g_function_simple(diffNorth, noiseLevel) * diffNorth +
				      g_function_simple(diffWest,  noiseLevel) * diffWest  +
				      g_function_simple(diffEast,  noiseLevel) * diffEast  +
				      g_function_simple(diffSouth, noiseLevel) * diffSouth;

			pDst[dstIdx] = static_cast<float>(current) + sum * timeStep;
			dstIdx++;
		}
	}
	return;
}


static inline void process_VUYA_4444_8u_buffer
(
	const csSDK_uint32* __restrict pSrc1, /* get SRC buffer for put U,V and ALPHA values to destination */
	float*		  __restrict pSrc2,
	csSDK_uint32* __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          noiseLevel,
	const float&          timeStep,
	const csSDK_int16&    gAdvanced
)
{
	csSDK_int32 i, j;
	csSDK_int32 k1, k2, lineIdx, origIdx;
	csSDK_int32 Y;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPixel = width - 1;

	float north, west, east, south, current;
	float diffNorth, diffWest, diffEast, diffSouth;
	float sum;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		k1 = MAX(0, j - 1);
		k2 = MIN(lastLine, j + 1);

		const float* prevLine = &pSrc2[k1 * width];
		const float* nextLine = &pSrc2[k2 * width];

		lineIdx = j * width;

		for (i = 0; i < width; i++)
		{
			north   = prevLine[i];
			west    = pSrc2[lineIdx + MAX(0, i - 1)];
			current = pSrc2[lineIdx + i];
			east    = pSrc2[lineIdx + MIN(lastPixel, i + 1)];
			south   = nextLine[i];

			diffNorth = north - current;
			diffWest  = west  - current;
			diffEast  = east  - current;
			diffSouth = south - current;

			if (gAdvanced)
				sum = g_function_advanced(diffNorth, noiseLevel) * diffNorth +
				      g_function_advanced(diffWest,  noiseLevel) * diffWest  +
				      g_function_advanced(diffEast,  noiseLevel) * diffEast  +
				      g_function_advanced(diffSouth, noiseLevel) * diffSouth;
			else
				sum = g_function_simple(diffNorth, noiseLevel) * diffNorth +
				      g_function_simple(diffWest,  noiseLevel) * diffWest  +
				      g_function_simple(diffEast,  noiseLevel) * diffEast  +
				      g_function_simple(diffSouth, noiseLevel) * diffSouth;

			Y = CLAMP_U8(static_cast<csSDK_int32>(current + sum * timeStep));

			origIdx = j * linePitch;
			pDst[origIdx + i] = (pSrc1[origIdx + i] & 0xFF00FFFF) | (Y << 16); /* U,V and ALPHA taken from source */
		}
	}

	return;
}



void process_VUYA_4444_8u_buffer
(
	const csSDK_uint32*  __restrict pSrc,
	const AlgMemStorage* __restrict pTmpBuffers,
	csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          dispersion,
	const float&          timeStep,
	const float&          noiseLevel,
	const csSDK_int16&    gAdvanced
)
{
	if (nullptr == pSrc || nullptr == pTmpBuffers || nullptr == pDst)
		return;

	float* __restrict pBuffers[2]
	{
		reinterpret_cast<float* __restrict>(pTmpBuffers->pTmp1),
		reinterpret_cast<float* __restrict>(pTmpBuffers->pTmp2)
	};

	float currentDispersion = 0.0f;
	float currentTimeStep = MIN(timeStep, dispersion - currentDispersion);
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
			process_VUYA_4444_8u_buffer (pSrc, pBuffers[ping], width, height, linePitch, noiseLevel, currentTimeStep, gAdvanced);
		}
		else if (currentDispersion + timeStep < dispersion)
		{
			process_float_raw_buffer (pBuffers[ping], pBuffers[pong], width, height, noiseLevel, currentTimeStep, gAdvanced);
			ping ^= 0x1;
			pong ^= 0x1;
		}
		else
		{
			process_VUYA_4444_8u_buffer (pSrc, pBuffers[ping], pDst, width, height, linePitch, noiseLevel, currentTimeStep, gAdvanced);
		}

		iterCnt++;
		currentDispersion += currentTimeStep;
		currentTimeStep = MIN(timeStep, dispersion - currentDispersion);

	} while (currentDispersion <= dispersion && currentTimeStep > minimalStep);

	return;
}


