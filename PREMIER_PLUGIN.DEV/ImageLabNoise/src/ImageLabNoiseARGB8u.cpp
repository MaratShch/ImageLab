#include "ImageLabNoise.h"
#include "ImageLabRandom.h"

void add_color_noise_ARGB4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
)
{
	const csSDK_int32 level_noise = static_cast<csSDK_int32>(noiseVolume);
	const csSDK_int32 level_image = static_cast<csSDK_int32>(100 - noiseVolume);
	
	csSDK_int32 i, j, idx;
	csSDK_uint32 nB, nG, nR, nA;
	csSDK_uint32 noise;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			idx = j * linePitch + i;

			noise = romuTrio32_random();

			nA = ( (noise & 0x000000FFu)        * level_noise +  (pSrc[idx] & 0x000000FFu)        * level_image) / 100;
			nR = (((noise & 0x0000FF00u) >> 8)  * level_noise + ((pSrc[idx] & 0x0000FF00u) >> 8)  * level_image) / 100;
			nG = (((noise & 0x00FF0000u) >> 16) * level_noise + ((pSrc[idx] & 0x00FF0000u) >> 16) * level_image) / 100;
			nB = (((noise & 0xFF000000u) >> 24) * level_noise + ((pSrc[idx] & 0xFF000000u) >> 24) * level_image) / 100;
			pDst[idx] = (nB << 24) | (nG << 16) | (nR << 8) |
				 (0 == noiseOnAlpha ? (pSrc[idx] & 0x000000FFu) : nA);
		}
	}

	return;
}

void add_bw_noise_ARGB4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
)
{
	const csSDK_int32 level_noise = static_cast<csSDK_int32>(noiseVolume);
	const csSDK_int32 level_image = static_cast<csSDK_int32>(100 - noiseVolume);
	csSDK_int32 i, j, idx;
	csSDK_uint32 nB, nG, nR, nA;
	csSDK_uint32 noise;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			idx = j * linePitch + i;

			noise = romuTrio32_random() & 0x000000FFu;

			const csSDK_uint32 noised = noise * level_noise;

			nA = (( pSrc[idx] & 0x000000FFu)        * level_image + noised) / 100;
			nR = (((pSrc[idx] & 0x0000FF00u) >> 8)  * level_image + noised) / 100;
			nG = (((pSrc[idx] & 0x00FF0000u) >> 16) * level_image + noised) / 100;
			nB = (((pSrc[idx] & 0xFF000000u) >> 24) * level_image + noised) / 100;
			pDst[idx] = (nB << 24)| (nG << 16) | (nR << 8) | 
				(0 == noiseOnAlpha ? (pSrc[idx] & 0x000000FFu) : nA);
		}
	}

	return;
}