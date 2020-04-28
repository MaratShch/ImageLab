#include "ImageLabNoise.h"
#include "ImageLabRandom.h"
#include "ImageLabColorConvert.h"

void add_color_noise_VUYA4444_32f
(
	const float*  __restrict pSrc,
	      float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha,
	bool                  isBT709
)
{
	constexpr float fDiv = static_cast<float>(0xFFFFFFFFu);
	const float* __restrict pYUV2RGB = (true == isBT709) ? coeff_YUV2RGB[BT709] : coeff_YUV2RGB[BT601];
	const float* __restrict pRGB2YUV = (true == isBT709) ? coeff_RGB2YUV[BT709] : coeff_RGB2YUV[BT601];

	const csSDK_int32 level_noise = static_cast<csSDK_int32>(noiseVolume);
	const csSDK_int32 level_image = static_cast<csSDK_int32>(100 - noiseVolume);

	csSDK_int32 i, j, idx;
	float noiseR, noiseG, noiseB, noiseA;
	float A, nA;
	float nV, nU, nY;
	float R, G, B;
	float V, U, Y;
	float nR, nG, nB;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			noiseR = static_cast<float>(romuTrio32_random()) / fDiv;
			noiseG = static_cast<float>(romuTrio32_random()) / fDiv;
			noiseB = static_cast<float>(romuTrio32_random()) / fDiv;

			idx = j * linePitch + i * 4;
			V = pSrc[idx    ];
			U = pSrc[idx + 1];
			Y = pSrc[idx + 2];
			A = pSrc[idx + 3];

			R = Y * pYUV2RGB[0] + U * pYUV2RGB[1] + V * pYUV2RGB[2];
			G = Y * pYUV2RGB[3] + U * pYUV2RGB[4] + V * pYUV2RGB[5];
			B = Y * pYUV2RGB[6] + U * pYUV2RGB[7] + V * pYUV2RGB[8];

			nR = (R * level_image + noiseR * level_noise) / 100.f;
			nG = (G * level_image + noiseG * level_noise) / 100.f;
			nB = (B * level_image + noiseB * level_noise) / 100.f;

			nY = nR * pRGB2YUV[0] + nG * pRGB2YUV[1] + nB * pRGB2YUV[2];
			nU = nR * pRGB2YUV[3] + nG * pRGB2YUV[4] + nB * pRGB2YUV[5];
			nV = nR * pRGB2YUV[6] + nG * pRGB2YUV[7] + nB * pRGB2YUV[8];

			nA = (0 == noiseOnAlpha) ? A :
				(A * level_image + (static_cast<float>(romuTrio32_random()) / fDiv) * level_noise) / 100.f;

			pDst[idx    ] = nV;
			pDst[idx + 1] = nU;
			pDst[idx + 2] = nY;
			pDst[idx + 3] = nA;
		}
	}

	return;
}

void add_bw_noise_VUYA4444_32f
(
	const float*  __restrict pSrc,
	      float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha,
	bool                  isBT709
)
{
	constexpr float fDiv = static_cast<float>(0xFFFFFFFFu);
	const float* __restrict pYUV2RGB = (true == isBT709) ? coeff_YUV2RGB[BT709] : coeff_YUV2RGB[BT601];
	const float* __restrict pRGB2YUV = (true == isBT709) ? coeff_RGB2YUV[BT709] : coeff_RGB2YUV[BT601];

	const csSDK_int32 level_noise = static_cast<csSDK_int32>(noiseVolume);
	const csSDK_int32 level_image = static_cast<csSDK_int32>(100 - noiseVolume);

	csSDK_int32 i, j, idx;
	float noise;
	float A, nA;
	float nV, nU, nY;
	float R, G, B;
	float V, U, Y;
	float nR, nG, nB;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			noise = static_cast<float>(romuTrio32_random()) / fDiv;

			idx = j * linePitch + i * 4;
			V = pSrc[idx    ];
			U = pSrc[idx + 1];
			Y = pSrc[idx + 2];
			A = pSrc[idx + 3];

			R = Y * pYUV2RGB[0] + U * pYUV2RGB[1] + V * pYUV2RGB[2];
			G = Y * pYUV2RGB[3] + U * pYUV2RGB[4] + V * pYUV2RGB[5];
			B = Y * pYUV2RGB[6] + U * pYUV2RGB[7] + V * pYUV2RGB[8];

			const float noised = static_cast<float>(noise * level_noise);

			nR = (R * level_image + noised) / 100.f;
			nG = (G * level_image + noised) / 100.f;
			nB = (B * level_image + noised) / 100.f;

			nY = nR * pRGB2YUV[0] + nG * pRGB2YUV[1] + nB * pRGB2YUV[2];
			nU = nR * pRGB2YUV[3] + nG * pRGB2YUV[4] + nB * pRGB2YUV[5];
			nV = nR * pRGB2YUV[6] + nG * pRGB2YUV[7] + nB * pRGB2YUV[8];

			nA = (0 == noiseOnAlpha) ? A : ((A * level_image + noised) / 100.f);

			pDst[idx    ] = nV;
			pDst[idx + 1] = nU;
			pDst[idx + 2] = nY;
			pDst[idx + 3] = nA;
		}
	}
	return;
}