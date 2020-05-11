#include "ImageLabColorCorrectionRGB.h"

// define color space conversion matrix's
CACHE_ALIGN float constexpr RGB2YUV[][9] =
{
	// BT.601
	{
		0.299000f,  0.587000f,  0.114000f,
	   -0.168736f, -0.331264f,  0.500000f,
		0.500000f, -0.418688f, -0.081312f
	},

	// BT.709
	{
		0.212600f,   0.715200f,  0.072200f,
	   -0.114570f,  -0.385430f,  0.500000f,
		0.500000f,  -0.454150f, -0.045850f
	},

	// BT.2020
	{
		0.262700f,   0.678000f,  0.059300f,
	   -0.139630f,  -0.360370f,  0.500000f,
		0.500000f,  -0.459790f, -0.040210f
	},

	// SMPTE 240M
	{
		0.212200f,   0.701300f,  0.086500f,
	   -0.116200f,  -0.383800f,  0.500000f,
		0.500000f,  -0.445100f, -0.054900f
	}
};

CACHE_ALIGN float constexpr YUV2RGB[][9] =
{
	// BT.601
	{
		1.000000f,  0.000000f,  1.407500f,
		1.000000f, -0.344140f, -0.716900f,
		1.000000f,  1.779000f,  0.000000f
	},

	// BT.709
	{
		1.000000f,  0.00000000f,  1.5748021f,
		1.000000f, -0.18732698f, -0.4681240f,
		1.000000f,  1.85559927f,  0.0000000f
	},

	// BT.2020
	{
		1.000000f,  0.00000000f,  1.4745964f,
		1.000000f, -0.16454810f, -0.5713517f,
		1.000000f,  1.88139998f,  0.0000000f
	},

	// SMPTE 240M
	{
		1.000000f,  0.0000000f,  1.5756000f,
		1.000000f, -0.2253495f, -0.4767712f,
		1.000000f,  1.8270219f,  0.0000000f
	}
};


void RGB_Correction_VUYA4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	      csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB,
	const csSDK_int32 isBT709
)
{
	csSDK_int32 i, j, idx;
	csSDK_int32 newV, newU, newY;
	float Y, U, V;
	float R, G, B;

	const float* __restrict pYUV2RGB = YUV2RGB[isBT709 & 0x1];
	const float* __restrict pRGB2YUV = RGB2YUV[isBT709 & 0x1];

	for (j = 0; j < height; j++)
	{
		idx = j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			V = static_cast<float>(static_cast<csSDK_int32>((srcPix[idx + i] & 0x000000FFu)       - 128));
			U = static_cast<float>(static_cast<csSDK_int32>((srcPix[idx + i] & 0x0000FF00u) >> 8) - 128);
			Y = static_cast<float>(static_cast<csSDK_int32>((srcPix[idx + i] & 0x00FF0000u) >> 16));

			R = (Y * pYUV2RGB[0] + U * pYUV2RGB[1] + V * pYUV2RGB[2]) + static_cast<float>(addR);
			G = (Y * pYUV2RGB[3] + U * pYUV2RGB[4] + V * pYUV2RGB[5]) + static_cast<float>(addG);
			B = (Y * pYUV2RGB[6] + U * pYUV2RGB[7] + V * pYUV2RGB[8]) + static_cast<float>(addB);

			newY = CLAMP_RGB8(static_cast<csSDK_int32>(R * pRGB2YUV[0] + G * pRGB2YUV[1] + B * pRGB2YUV[2]));
			newU = CLAMP_RGB8(static_cast<csSDK_int32>(R * pRGB2YUV[3] + G * pRGB2YUV[4] + B * pRGB2YUV[5]) + 128);
			newV = CLAMP_RGB8(static_cast<csSDK_int32>(R * pRGB2YUV[6] + G * pRGB2YUV[7] + B * pRGB2YUV[8]) + 128);

			dstPix[idx + i] = newV | (newU << 8) | (newY << 16) | (srcPix[idx + i] & 0xFF000000u);
		}
	}
	return;
}


void RGB_Correction_VUYA4444_32f
(
	const float* __restrict srcPix,
	      float* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB,
	const csSDK_int32 isBT709
)
{
	csSDK_int32 i, j, idx;
	float Y, U, V, A;
	float newY, newU, newV;
	float R, G, B;

	constexpr float factor = 1.0f / 256.0f;
	const float _addR = static_cast<float>(addR) * factor;
	const float _addG = static_cast<float>(addG) * factor;
	const float _addB = static_cast<float>(addB) * factor;

	const float* __restrict pYUV2RGB = YUV2RGB[isBT709 & 0x1];
	const float* __restrict pRGB2YUV = RGB2YUV[isBT709 & 0x1];

	for (j = 0; j < height; j++)
	{
		idx = j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			V = srcPix[idx + i * 4    ];
			U = srcPix[idx + i * 4 + 1];
			Y = srcPix[idx + i * 4 + 2];
			A = srcPix[idx + i * 4 + 3];

			R = CLAMP_RGB32F((Y * pYUV2RGB[0] + U * pYUV2RGB[1] + V * pYUV2RGB[2]) + _addR);
			G = CLAMP_RGB32F((Y * pYUV2RGB[3] + U * pYUV2RGB[4] + V * pYUV2RGB[5]) + _addG);
			B = CLAMP_RGB32F((Y * pYUV2RGB[6] + U * pYUV2RGB[7] + V * pYUV2RGB[8]) + _addB);

			newY = R * pRGB2YUV[0] + G * pRGB2YUV[1] + B * pRGB2YUV[2];
			newU = R * pRGB2YUV[3] + G * pRGB2YUV[4] + B * pRGB2YUV[5];
			newV = R * pRGB2YUV[6] + G * pRGB2YUV[7] + B * pRGB2YUV[8];

			dstPix[idx + i * 4    ] = newV;
			dstPix[idx + i * 4 + 1] = newU;
			dstPix[idx + i * 4 + 2] = newY;
			dstPix[idx + i * 4 + 3] = A;
		}
	}
	return;
}