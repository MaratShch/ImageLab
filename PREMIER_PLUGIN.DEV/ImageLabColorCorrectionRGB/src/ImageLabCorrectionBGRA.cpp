#include "ImageLabColorCorrectionRGB.h"

void RGB_Correction_BGRA4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	      csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB
)
{
	csSDK_int32 i, j, idx;
	csSDK_int16 newR, newG, newB;

	for (j = 0; j < height; j++)
	{
		idx = j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			newB = CLAMP_RGB8(static_cast<csSDK_int16> (srcPix[idx+i] & 0x000000FFu)        + addB);
			newG = CLAMP_RGB8(static_cast<csSDK_int16>((srcPix[idx+i] & 0x0000FF00u) >> 8)  + addG);
			newR = CLAMP_RGB8(static_cast<csSDK_int16>((srcPix[idx+i] & 0x00FF0000u) >> 16) + addR);

			/* copy ALPHA channel from source buffer */
			dstPix[idx + i] = newB | (newG << 8) | (static_cast<csSDK_int32>(newR) << 16) | (srcPix[idx + i] & 0xFF000000u);
		}
	}
	return;
}


void RGB_Correction_BGRA4444_16u
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB
)
{
	csSDK_int32 i, j, idx;
	csSDK_int32 newR, newG, newB;

	constexpr csSDK_int16 factor = 256;
	const csSDK_int32 _addR = static_cast<csSDK_int32>(addR * factor);
	const csSDK_int32 _addG = static_cast<csSDK_int32>(addG * factor);
	const csSDK_int32 _addB = static_cast<csSDK_int32>(addB * factor);

	for (j = 0; j < height; j++)
	{
		idx = j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			newB = CLAMP_RGB16(static_cast<csSDK_int32> (srcPix[idx + i * 2]     & 0x0000FFFFu)        + _addB);
			newG = CLAMP_RGB16(static_cast<csSDK_int32>((srcPix[idx + i * 2]     & 0xFFFF0000u) >> 16) + _addG);
			newR = CLAMP_RGB16(static_cast<csSDK_int32> (srcPix[idx + i * 2 + 1] & 0x0000FFFFu)        + _addR);

			/* copy ALPHA channel from source buffer */
			dstPix[idx + i * 2]     = newB | (newG << 16);
			dstPix[idx + i * 2 + 1] = newR | (srcPix[idx + i * 2 + 1] & 0xFFFF0000u);
		}
	}
	return;
}


void RGB_Correction_BGRA4444_32f
(
	const float* __restrict srcPix,
	      float* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB
)
{
	csSDK_int32 i, j, idx;
	csSDK_int32 newR, newG, newB;

	constexpr float factor = 1.0f/256.0f;
	const float _addR = static_cast<float>(addR) * factor;
	const float _addG = static_cast<float>(addG) * factor;
	const float _addB = static_cast<float>(addB) * factor;

	for (j = 0; j < height; j++)
	{
		idx = j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			dstPix[idx + i * 4    ] = CLAMP_RGB32F(srcPix[idx + i * 4    ] + _addB);
			dstPix[idx + i * 4 + 1] = CLAMP_RGB32F(srcPix[idx + i * 4 + 1] + _addG);
			dstPix[idx + i * 4 + 2] = CLAMP_RGB32F(srcPix[idx + i * 4 + 2] + _addR);
			dstPix[idx + i * 4 + 3] = srcPix[idx + i * 4 + 3];
		}
	}
	return;
}