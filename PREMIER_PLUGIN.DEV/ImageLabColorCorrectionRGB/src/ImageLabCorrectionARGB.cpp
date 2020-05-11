#include "ImageLabColorCorrectionRGB.h"

void RGB_Correction_ARGB4444_8u
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
			newR = CLAMP_RGB8(static_cast<csSDK_int16>((srcPix[idx + i] & 0x0000FF00u) >> 8)  + addR);
			newG = CLAMP_RGB8(static_cast<csSDK_int16>((srcPix[idx + i] & 0x00FF0000u) >> 16) + addG);
			newB = CLAMP_RGB8(static_cast<csSDK_int16>((srcPix[idx + i] & 0xFF000000u) >> 24) + addB);

			/* copy ALPHA channel from source buffer */
			dstPix[idx + i] = (srcPix[idx + i] & 0x000000FFu) | (newR << 8) | (newG << 16) | (newB << 24);
		}
	}
	return;
}


void RGB_Correction_ARGB4444_16u
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
			newR = CLAMP_RGB16(static_cast<csSDK_int32>((srcPix[idx + i * 2]     & 0xFFFF0000u) >> 16) + _addR);
			newG = CLAMP_RGB16(static_cast<csSDK_int32> (srcPix[idx + i * 2 + 1] & 0x0000FFFFu)        + _addG);
			newB = CLAMP_RGB16(static_cast<csSDK_int32>((srcPix[idx + i * 2 + 1] & 0xFFFF0000u) >> 16) + _addB);

			/* copy ALPHA channel from source buffer */
			dstPix[idx + i * 2]     = newB | (newG << 16);
			dstPix[idx + i * 2 + 1] = newR | (srcPix[idx + i * 2 + 1] & 0xFFFF0000u);
		}
	}
	return;
}


void RGB_Correction_ARGB4444_32f
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
			dstPix[idx + i * 4    ] = srcPix[idx + i * 4];
			dstPix[idx + i * 4 + 1] = CLAMP_RGB32F(srcPix[idx + i * 4 + 1] + _addR);
			dstPix[idx + i * 4 + 2] = CLAMP_RGB32F(srcPix[idx + i * 4 + 2] + _addG);
			dstPix[idx + i * 4 + 3] = CLAMP_RGB32F(srcPix[idx + i * 4 + 3] + _addB);
		}
	}
	return;
}

void RGB_Correction_RGB444_10u
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

	const csSDK_int32 _addR = addR * 4;
	const csSDK_int32 _addG = addG * 4;
	const csSDK_int32 _addB = addB * 4;

	for (j = 0; j < height; j++)
	{
		idx = j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			newB = CLAMP_RGB10(static_cast<csSDK_int32>((srcPix[idx + i] & 0x00000FFCu) >> 2)  + _addB);
			newG = CLAMP_RGB10(static_cast<csSDK_int32>((srcPix[idx + i] & 0x003FF000u) >> 12) + _addG);
			newR = CLAMP_RGB10(static_cast<csSDK_int32>((srcPix[idx + i] & 0xFFC00000u) >> 22) + _addR);

			/* copy ALPHA channel from source buffer */
			dstPix[idx + i] = (newB << 2) | (newG << 12) | (newR << 22);
		}
	}
	return;
}