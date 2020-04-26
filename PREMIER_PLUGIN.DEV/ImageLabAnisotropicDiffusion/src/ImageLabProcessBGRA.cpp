#include "ImageLabProcessTmpBuffer.h"
#include "ImageLabColorConvert.h"

static inline void process_BGRA_4444_8u_buffer
(
	const csSDK_uint32*  __restrict pSrc,
	      float*         __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          noiseLevel,
	const float&          timeStep
)
{
	csSDK_int32 i, j;
	csSDK_int32 k1, k2, lineIdx, pixIdx, dstIdx;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPixel = width - 1;
	
	float north, west, east, south, current;
	float B, G, R;
	float diffNorth, diffWest, diffEast, diffSouth;
	float sum;

	const float* __restrict pRgb2Yuv = (width > 800 ? coeff_RGB2YUV[BT709] : coeff_RGB2YUV[BT601]);
	dstIdx = pixIdx = 0;

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
			// notrh
			B = static_cast<float> (prevLine[i] & 0x000000FF);
			G = static_cast<float>((prevLine[i] & 0x0000FF00) >> 8);
			R = static_cast<float>((prevLine[i] & 0x00FF0000) >> 16);
			north = R * pRgb2Yuv[0] + G * pRgb2Yuv[1] + B * pRgb2Yuv[2];

			// west
			pixIdx = lineIdx + MAX(0, i - 1);
			B = static_cast<float> (pSrc[pixIdx] & 0x000000FF);
			G = static_cast<float>((pSrc[pixIdx] & 0x0000FF00) >> 8);
			R = static_cast<float>((pSrc[pixIdx] & 0x00FF0000) >> 16);
			west = R * pRgb2Yuv[0] + G * pRgb2Yuv[1] + B * pRgb2Yuv[2];

			// current
			pixIdx = lineIdx + i;
			B = static_cast<float> (pSrc[pixIdx] & 0x000000FF);
			G = static_cast<float>((pSrc[pixIdx] & 0x0000FF00) >> 8);
			R = static_cast<float>((pSrc[pixIdx] & 0x00FF0000) >> 16);
			current = R * pRgb2Yuv[0] + G * pRgb2Yuv[1] + B * pRgb2Yuv[2];

			// east
			pixIdx = lineIdx + MIN(lastPixel, i + 1);
			B = static_cast<float> (pSrc[pixIdx] & 0x000000FF);
			G = static_cast<float>((pSrc[pixIdx] & 0x0000FF00) >> 8);
			R = static_cast<float>((pSrc[pixIdx] & 0x00FF0000) >> 16);
			east = R * pRgb2Yuv[0] + G * pRgb2Yuv[1] + B * pRgb2Yuv[2];

			// south
			B = static_cast<float> (nextLine[i] & 0x000000FF);
			G = static_cast<float>((nextLine[i] & 0x0000FF00) >> 8);
			R = static_cast<float>((nextLine[i] & 0x00FF0000) >> 16);
			south = R * pRgb2Yuv[0] + G * pRgb2Yuv[1] + B * pRgb2Yuv[2];

			diffNorth = static_cast<float>(north - current);
			diffWest  = static_cast<float>(west  - current);
			diffEast  = static_cast<float>(east  - current);
			diffSouth = static_cast<float>(south - current);

			sum = g_function(diffNorth, noiseLevel) * diffNorth +
	  			  g_function(diffWest, noiseLevel)  * diffWest  +
			 	  g_function(diffEast, noiseLevel)  * diffEast  +
				  g_function(diffSouth, noiseLevel) * diffSouth;

			pDst[dstIdx] = static_cast<float>(current) + sum * timeStep;
			dstIdx++;
		}
	}
	return;
}


static inline void process_BGRA_4444_8u_buffer
(
	const csSDK_uint32* __restrict pSrc1, /* get SRC buffer for put U,V and ALPHA values to destination */
	      float*	    __restrict pSrc2,
	csSDK_uint32*       __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          noiseLevel,
	const float&          timeStep
)
{
	csSDK_int32 i, j;
	csSDK_int32 k1, k2, lineIdx, origIdx;
	const csSDK_int32 lastLine = height - 1;
	const csSDK_int32 lastPixel = width - 1;
	csSDK_int32 B, G, R;

	float Y, U, V;
	float north, west, east, south, current;
	float diffNorth, diffWest, diffEast, diffSouth;
	float sum;

	const float* __restrict pRgb2Yuv = (width > 800 ? coeff_RGB2YUV[BT709] : coeff_RGB2YUV[BT601]);
	const float* __restrict pYuv2Rgb = (width > 800 ? coeff_YUV2RGB[BT709] : coeff_YUV2RGB[BT601]);

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
			north = prevLine[i];
			west = pSrc2[lineIdx + MAX(0, i - 1)];
			current = pSrc2[lineIdx + i];
			east = pSrc2[lineIdx + MIN(lastPixel, i + 1)];
			south = nextLine[i];

			diffNorth = north - current;
			diffWest  = west  - current;
			diffEast  = east  - current;
			diffSouth = south - current;

			sum = g_function(diffNorth, noiseLevel) * diffNorth +
				  g_function(diffWest,  noiseLevel) * diffWest  +
				  g_function(diffEast,  noiseLevel) * diffEast  +
				  g_function(diffSouth, noiseLevel) * diffSouth;

			Y = current + sum * timeStep;
			
			/* get U value - convert R,G,B from source image */
			origIdx = j * linePitch + i;

			U = static_cast<float> (pSrc1[origIdx] & 0x000000FF)        * pRgb2Yuv[5] +
				static_cast<float>((pSrc1[origIdx] & 0x0000FF00) >> 8)  * pRgb2Yuv[4] +
				static_cast<float>((pSrc1[origIdx] & 0x00FF0000) >> 16) * pRgb2Yuv[3];

			V = static_cast<float> (pSrc1[origIdx] & 0x000000FF)        * pRgb2Yuv[8] +
				static_cast<float>((pSrc1[origIdx] & 0x0000FF00) >> 8)  * pRgb2Yuv[7] +
				static_cast<float>((pSrc1[origIdx] & 0x00FF0000) >> 16) * pRgb2Yuv[6];

			R = static_cast<csSDK_int32>(Y * pYuv2Rgb[0] + U * pYuv2Rgb[1] + V * pYuv2Rgb[2]);
			G = static_cast<csSDK_int32>(Y * pYuv2Rgb[3] + U * pYuv2Rgb[4] + V * pYuv2Rgb[5]);
			B = static_cast<csSDK_int32>(Y * pYuv2Rgb[6] + U * pYuv2Rgb[7] + V * pYuv2Rgb[8]);

			pDst[origIdx] = (pSrc1[origIdx] & 0xFF000000u) |
				                      (CLAMP_U8(R) << 16)  |
				                      (CLAMP_U8(G) << 8)   |
				                       CLAMP_U8(B);
		}
	}

	return;
}


void process_BGRA_4444_8u_buffer
(
	const csSDK_uint32*  __restrict pSrc,
	const AlgMemStorage* __restrict pTmpBuffers,
	csSDK_uint32*  __restrict pDst,
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
			process_BGRA_4444_8u_buffer(pSrc, pBuffers[ping], width, height, linePitch, noiseLevel, currentTimeStep);
		}
		else if (currentDispersion + timeStep < dispersion)
		{
			process_float_raw_buffer(pBuffers[ping], pBuffers[pong], width, height, noiseLevel, currentTimeStep);
			ping ^= 0x1;
			pong ^= 0x1;
		}
		else
		{
			process_BGRA_4444_8u_buffer(pSrc, pBuffers[ping], pDst, width, height, linePitch, noiseLevel, currentTimeStep);
		}

		iterCnt++;
		currentDispersion += currentTimeStep;
		currentTimeStep = MIN(timeStep, dispersion - currentDispersion);

	} while (currentDispersion <= dispersion && currentTimeStep > minimalStep);

	return;
}

