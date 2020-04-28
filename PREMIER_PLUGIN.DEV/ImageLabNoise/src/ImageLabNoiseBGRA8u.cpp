#include "ImageLabNoise.h"
#include "ImageLabRandom.h"

void add_color_noise_BGRA4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseDencity,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
)
{
	const csSDK_int32 level_noise = static_cast<csSDK_int32>(noiseVolume);
	const csSDK_int32 level_image = static_cast<csSDK_int32>(100 - noiseVolume);
	
	csSDK_int32 i, j;
	csSDK_uint32 nB, nG, nR, nA;
	csSDK_uint32 noise;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			noise = romuTrio32_random();
			nB = ( (noise & 0x000000FFu)        * level_noise +  (pSrc[j * linePitch + i] & 0x000000FFu)        * level_image) / 100;
			nG = (((noise & 0x0000FF00u) >> 8)  * level_noise + ((pSrc[j * linePitch + i] & 0x0000FF00u) >> 8)  * level_image) / 100;
			nR = (((noise & 0x00FF0000u) >> 16) * level_noise + ((pSrc[j * linePitch + i] & 0x00FF0000u) >> 16) * level_image) / 100;
			nA = (((noise & 0xFF000000u) >> 24) * level_noise + ((pSrc[j * linePitch + i] & 0xFF000000u) >> 24) * level_image) / 100;
			pDst[j * linePitch + i] = nB | (nG << 8) | (nR << 16) | (0 != noiseOnAlpha ? (nA << 24) : (pSrc[j * linePitch + i] & 0xFF000000u));
		}
	}

	return;
}


void add_bw_noise_BGRA4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseDencity,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
)
{
	const csSDK_int32 level_noise = static_cast<csSDK_int32>(noiseVolume);
	const csSDK_int32 level_image = static_cast<csSDK_int32>(100 - noiseVolume);
	csSDK_int32 i, j;
	csSDK_uint32 nB, nG, nR, nA;
	csSDK_uint32 noise;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			noise = romuTrio32_random() & 0x000000FFu;
			nB = (( pSrc[j * linePitch + i] & 0x000000FFu)        * level_image + noise * level_noise) / 100;
			nG = (((pSrc[j * linePitch + i] & 0x0000FF00u) >> 8)  * level_image + noise * level_noise) / 100;
			nR = (((pSrc[j * linePitch + i] & 0x00FF0000u) >> 16) * level_image + noise * level_noise) / 100;
			nA = (((pSrc[j * linePitch + i] & 0xFF000000u) >> 24) * level_image + noise * level_noise) / 100;
			pDst[j * linePitch + i] = nB | (nG << 8) | (nR << 16) | (0 != noiseOnAlpha ? (nA << 24) : (pSrc[j * linePitch + i] & 0xFF000000u));
		}
	}

	return;
}