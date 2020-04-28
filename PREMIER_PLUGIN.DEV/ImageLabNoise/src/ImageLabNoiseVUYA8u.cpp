#include "ImageLabNoise.h"
#include "ImageLabRandom.h"
#include "ImageLabColorConvert.h"

void add_color_noise_VUYA4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha,
	bool                  isBT709
)
{
	const float* __restrict pYUV2RGB = (true == isBT709) ? coeff_YUV2RGB[BT709] : coeff_YUV2RGB[BT601];
	const float* __restrict pRGB2YUV = (true == isBT709) ? coeff_RGB2YUV[BT709] : coeff_RGB2YUV[BT601];

	const csSDK_int32 level_noise = static_cast<csSDK_int32>(noiseVolume);
	const csSDK_int32 level_image = static_cast<csSDK_int32>(100 - noiseVolume);
	
	csSDK_int32 i, j, idx;
	csSDK_uint32 noise;
	csSDK_uint32 A, nA;
	csSDK_int32 nV, nU, nY;
	float R, G, B;
	float V, U, Y;
	float nR, nG, nB;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			noise = romuTrio32_random();

			idx = j * linePitch + i;
			V = static_cast<float> (pSrc[idx] & 0x000000FFu)        - 128.f;
			U = static_cast<float>((pSrc[idx] & 0x0000FF00u) >> 8 ) - 128.f;
			Y = static_cast<float>((pSrc[idx] & 0x00FF0000u) >> 16);
			A = static_cast<csSDK_uint32>(pSrc[idx] >> 24);

			R = Y * pYUV2RGB[0] + U * pYUV2RGB[1] + V * pYUV2RGB[2];
			G = Y * pYUV2RGB[3] + U * pYUV2RGB[4] + V * pYUV2RGB[5];
			B = Y * pYUV2RGB[6] + U * pYUV2RGB[7] + V * pYUV2RGB[8];

			nR = (R * level_image + static_cast<float>(static_cast<csSDK_int32> (noise & 0x000000FFu)        * level_noise)) / 100.f;
			nG = (G * level_image + static_cast<float>(static_cast<csSDK_int32>((noise & 0x0000FF00u) >> 8)  * level_noise)) / 100.f;
			nB = (B * level_image + static_cast<float>(static_cast<csSDK_int32>((noise & 0x00FF0000u) >> 16) * level_noise)) / 100.f;

			nY = static_cast<csSDK_int32>(nR * pRGB2YUV[0] + nG * pRGB2YUV[1] + nB * pRGB2YUV[2]);
			nU = static_cast<csSDK_int32>(nR * pRGB2YUV[3] + nG * pRGB2YUV[4] + nB * pRGB2YUV[5]) + 128;
			nV = static_cast<csSDK_int32>(nR * pRGB2YUV[6] + nG * pRGB2YUV[7] + nB * pRGB2YUV[8]) + 128;

			nA = (0 == noiseOnAlpha) ? A :
				(A * static_cast<csSDK_uint32>(level_image) + (noise >> 24) * static_cast<csSDK_uint32>(level_noise)) / 100;

			pDst[idx] = nV | (nU << 8) | (nY << 16) | (nA << 24);
		}
	}

	return;
}

void add_bw_noise_VUYA4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha,
	bool                  isBT709
)
{
	const float* __restrict pYUV2RGB = (true == isBT709) ? coeff_YUV2RGB[BT709] : coeff_YUV2RGB[BT601];
	const float* __restrict pRGB2YUV = (true == isBT709) ? coeff_RGB2YUV[BT709] : coeff_RGB2YUV[BT601];

	const csSDK_int32 level_noise = static_cast<csSDK_int32>(noiseVolume);
	const csSDK_int32 level_image = static_cast<csSDK_int32>(100 - noiseVolume);

	csSDK_int32 i, j, idx;
	csSDK_int32 noise;
	csSDK_int32 A, nA;
	csSDK_int32 nV, nU, nY;
	float R, G, B;
	float V, U, Y;
	float nR, nG, nB;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			noise = static_cast<csSDK_int32>(romuTrio32_random() & 0xFFu);

			idx = j * linePitch + i;
			V = static_cast<float> (pSrc[idx] & 0x000000FFu)       - 128.f;
			U = static_cast<float>((pSrc[idx] & 0x0000FF00u) >> 8) - 128.f;
			Y = static_cast<float>((pSrc[idx] & 0x00FF0000u) >> 16);
			A = static_cast<csSDK_int32>((pSrc[idx] & 0xFF000000u) >> 24);

			R = Y * pYUV2RGB[0] + U * pYUV2RGB[1] + V * pYUV2RGB[2];
			G = Y * pYUV2RGB[3] + U * pYUV2RGB[4] + V * pYUV2RGB[5];
			B = Y * pYUV2RGB[6] + U * pYUV2RGB[7] + V * pYUV2RGB[8];

			const float noised = static_cast<float>(noise * level_noise);
			nR = (R * level_image + noised) / 100.f;
			nG = (G * level_image + noised) / 100.f;
			nB = (B * level_image + noised) / 100.f;

			nY = static_cast<csSDK_int32>(nR * pRGB2YUV[0] + nG * pRGB2YUV[1] + nB * pRGB2YUV[2]);
			nU = static_cast<csSDK_int32>(nR * pRGB2YUV[3] + nG * pRGB2YUV[4] + nB * pRGB2YUV[5]) + 128;
			nV = static_cast<csSDK_int32>(nR * pRGB2YUV[6] + nG * pRGB2YUV[7] + nB * pRGB2YUV[8]) + 128;

			nA = (0 == noiseOnAlpha) ? A :
				(A * static_cast<csSDK_uint32>(level_image) + noise * static_cast<csSDK_uint32>(level_noise)) / 100;

			pDst[idx] = nV | (nU << 8) | (nY << 16) | (nA << 24);
		}
	}

	return;
}