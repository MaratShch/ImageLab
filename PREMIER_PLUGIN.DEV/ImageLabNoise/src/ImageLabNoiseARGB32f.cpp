#include "ImageLabNoise.h"
#include "ImageLabRandom.h"

void add_color_noise_ARGB4444_32f
(
	const float*  __restrict pSrc,
	      float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
)
{
	constexpr float fDiv = static_cast<float>(0xFFFFFFFFu);
	const csSDK_int32 level_noise = static_cast<csSDK_int32>(noiseVolume);
	const csSDK_int32 level_image = static_cast<csSDK_int32>(100 - noiseVolume);
	csSDK_int32 i, j, idx;
	float noiseB, noiseG, noiseR;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			noiseR = static_cast<float>(romuTrio32_random()) / fDiv;
			noiseG = static_cast<float>(romuTrio32_random()) / fDiv;
			noiseB = static_cast<float>(romuTrio32_random()) / fDiv;

			idx = j * linePitch + i * 4;

			pDst[idx ] = (0 == noiseOnAlpha ?
				pSrc[idx] : ((pSrc[idx] * level_image + (static_cast<float>(romuTrio32_random()) / fDiv) * level_noise) / 100.f));

			pDst[idx + 1] = (pSrc[idx + 1] * level_image + noiseR * level_noise) / 100.f;
			pDst[idx + 2] = (pSrc[idx + 2] * level_image + noiseG * level_noise) / 100.f;
			pDst[idx + 3] = (pSrc[idx + 3] * level_image + noiseB * level_noise) / 100.f;
		}
	}

	return;
}

void add_bw_noise_ARGB4444_32f
(
	const float*  __restrict pSrc,
	      float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
)
{
	constexpr float fDiv = static_cast<float>(0xFFFFFFFFu);
	const csSDK_int32 level_noise = static_cast<csSDK_int32>(noiseVolume);
	const csSDK_int32 level_image = static_cast<csSDK_int32>(100 - noiseVolume);
	csSDK_int32 i, j, idx;
	float noise;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			noise = static_cast<float>(romuTrio32_random()) / fDiv;
			const float noised = noise * level_noise;

			idx = j * linePitch + i * 4;

			pDst[idx] = (0 == noiseOnAlpha ?
				pSrc[idx] : ((pSrc[idx] * level_image + noised) / 100.f));

			pDst[idx + 1] = (pSrc[idx + 1] * level_image + noised) / 100.f;
			pDst[idx + 2] = (pSrc[idx + 2] * level_image + noised) / 100.f;
			pDst[idx + 3] = (pSrc[idx + 3] * level_image + noised) / 100.f;
		}
	}

	return;
}