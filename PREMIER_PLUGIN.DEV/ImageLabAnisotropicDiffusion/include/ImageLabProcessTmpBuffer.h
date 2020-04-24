#include "ImageLabAnisotropicDiffusion.h"
#include "ImageLabGFunction.h"

/* Because this part of algorithm expected to be common for all formats - let's write it in separate  include file
   and include this file to relevant c++ code 
*/
static inline void process_float_raw_buffer
(
	float*  __restrict pSrc,
	float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const float&          noiseLevel,
	const float&          timeStep,
	const csSDK_int16&    gAdvanced
)
{
	csSDK_int32 i, j;
	csSDK_int32 k1, k2, lineIdx, dstIdx;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPixel = width - 1;

	float north, west, east, south, current;
	float diffNorth, diffWest, diffEast, diffSouth;
	float sum;

	dstIdx = 0;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		k1 = MAX(0, j - 1);
		k2 = MIN(lastLine, j + 1);

		const float* prevLine = &pSrc[k1 * width];
		const float* nextLine = &pSrc[k2 * width];

		lineIdx = j * width;

		for (i = 0; i < width; i++)
		{
			north   = prevLine[i];
			west    = pSrc[lineIdx + MAX(0, i - 1)];
			current = pSrc[lineIdx + i];
			east    = pSrc[lineIdx + MIN(lastPixel, i + 1)];
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

			pDst[dstIdx] = current + sum * timeStep;
			dstIdx++;
		}
	}
	return;
}

