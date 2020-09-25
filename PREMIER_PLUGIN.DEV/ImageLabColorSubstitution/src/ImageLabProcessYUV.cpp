#include "AdobeImageLabColorSubstitution.h"

// define color space conversion matrix's
CACHE_ALIGN constexpr float RGB2YUV[][9] =
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
	}
};


CACHE_ALIGN constexpr float YUV2RGB[][9] =
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
	}
};



void colorSubstitute_VUYA_4444_8u
(
	const csSDK_uint32* __restrict pSrc,
	      csSDK_uint32* __restrict pDst,
	const csSDK_int32&             height,
	const csSDK_int32&             width,
	const csSDK_int32&             linePitch,
	const prColor&                 from,
	const prColor&                 to,
	const csSDK_int32&             tolerance,
	const bool&                    showMask,
	const bool&                    isBT709
)
{
	csSDK_int32 i = 0, j = 0;
	float R = 0.f, G = 0.f, B = 0.f;
	float newR = 0.f, newG = 0.f, newB = 0.f;
	csSDK_int32 Y = 0, U = 0, V = 0;
	csSDK_int32 newY = 0, newU = 0, newV = 0;

	const csSDK_int32 fromB = (from & 0x00FF0000) >> 16;
	const csSDK_int32 fromG = (from & 0x0000FF00) >> 8;
	const csSDK_int32 fromR = (from & 0x000000FF);

	const csSDK_int32 toB = (to & 0x00FF0000) >> 16;
	const csSDK_int32 toG = (to & 0x0000FF00) >> 8;
	const csSDK_int32 toR = (to & 0x000000FF);

	const float rMin = fromR - tolerance;
	const float rMax = fromR + tolerance;

	const float gMin = fromG - tolerance;
	const float gMax = fromG + tolerance;

	const float bMin = fromB - tolerance;
	const float bMax = fromB + tolerance;

	const float addR = fromR + toR;
	const float addG = fromG + toG;
	const float addB = fromB + toB;

	const float* __restrict yuv2rgb = ((true == isBT709) ? YUV2RGB[1] : YUV2RGB[0]);
	const float* __restrict rgb2yuv = ((true == isBT709) ? RGB2YUV[1] : RGB2YUV[0]);

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* __restrict pSrcLine = pSrc + j * linePitch;
		      csSDK_uint32* __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			Y = ( pSrcLine[i] & 0x00FF0000u) >> 16;
			U = ((pSrcLine[i] & 0x0000FF00u) >> 8) - 128;
			V = ( pSrcLine[i] & 0x000000FFu) - 128;

			R = static_cast<float>(Y) * yuv2rgb[0] + static_cast<float>(U) * yuv2rgb[1] + static_cast<float>(V) * yuv2rgb[2];
			G = static_cast<float>(Y) * yuv2rgb[3] + static_cast<float>(U) * yuv2rgb[4] + static_cast<float>(V) * yuv2rgb[5];
			B = static_cast<float>(Y) * yuv2rgb[6] + static_cast<float>(U) * yuv2rgb[7] + static_cast<float>(V) * yuv2rgb[8];

			if ((R < rMax) && (R > rMin) && (G < gMax) && (G > gMin) && (B < bMax) && (B > bMin))
			{
				if (false == showMask)
				{
					newR = CLAMP_RGB8(addR - R);
					newG = CLAMP_RGB8(addG - G);
					newB = CLAMP_RGB8(addB - B);
				}
				else
					newR = newG = newB = 0xFF;
			}
			else
			{
				if (false == showMask)
				{
					newR = R;
					newG = G;
					newB = B;
				}
				else
				{
					newR = newG = newB = 0x0;
				}
			}

			newY = static_cast<csSDK_int32>(newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2]);
			newU = static_cast<csSDK_int32>(newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5] + 128.f);
			newV = static_cast<csSDK_int32>(newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8] + 128.f);

			pDstLine[i] = (pSrcLine[i] & 0xFF000000u) | (newY << 16) | (newU << 8) | newV;
			
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}

void colorSubstitute_VUYA_4444_32f
(
	const float* __restrict pSrc,
	      float* __restrict pDst,
	const csSDK_int32&      height,
	const csSDK_int32&      width,
	const csSDK_int32&      linePitch,
	const prColor&          from,
	const prColor&          to,
	const csSDK_int32&      tolerance,
	const bool&             showMask,
	const bool&             isBT709
)
{
	csSDK_int32 i = 0, j = 0, idx = 0;
	float R = 0.f, G = 0.f, B = 0.f;
	float newR = 0.f, newG = 0.f, newB = 0.f;
	float Y = 0.f, U = 0.f, V = 0.f, A = 0.f;
	float newY = 0.f, newU = 0.f, newV = 0.f;

	const float fromB = ((from & 0x00FF0000) >> 16) / 256.f;
	const float fromG = ((from & 0x0000FF00) >> 8)  / 256.f;
	const float fromR = (from & 0x000000FF) / 256.f;

	const float toB = ((to & 0x00FF0000) >> 16) / 256.f;
	const float toG = ((to & 0x0000FF00) >> 8)  / 256.f;
	const float toR = (to & 0x000000FF) / 256.f;

	const float fTolerance = tolerance / 256.f;

	const float rMin = fromR - fTolerance;
	const float rMax = fromR + fTolerance;

	const float gMin = fromG - fTolerance;
	const float gMax = fromG + fTolerance;

	const float bMin = fromB - fTolerance;
	const float bMax = fromB + fTolerance;

	const float addR = fromR + toR;
	const float addG = fromG + toG;
	const float addB = fromB + toB;

	const float* __restrict yuv2rgb = ((true == isBT709) ? YUV2RGB[1] : YUV2RGB[0]);
	const float* __restrict rgb2yuv = ((true == isBT709) ? RGB2YUV[1] : RGB2YUV[0]);

	for (j = 0; j < height; j++)
	{
		const float* __restrict pSrcLine = pSrc + j * linePitch;
		      float* __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i * 4;
			V = pSrcLine[idx    ];
			U = pSrcLine[idx + 1];
			Y = pSrcLine[idx + 2];
			A = pSrcLine[idx + 3];

			R = Y * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2];
			G = Y * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5];
			B = Y * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8];

			if ((R < rMax) && (R > rMin) && (G < gMax) && (G > gMin) && (B < bMax) && (B > bMin))
			{
				if (false == showMask)
				{
					newR = CLAMP_RGB32F(addR - R);
					newG = CLAMP_RGB32F(addG - G);
					newB = CLAMP_RGB32F(addB - B);
				}
				else
					newR = newG = newB = f32_white;
			}
			else
			{
				if (false == showMask)
				{
					newR = R;
					newG = G;
					newB = B;
				}
				else
				{
					newR = newG = newB = f32_black;
				}
			}
			
			newY = newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2];
			newU = newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5];
			newV = newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8];

			pDstLine[idx    ] = newV;
			pDstLine[idx + 1] = newU;
			pDstLine[idx + 2] = newY;
			pDstLine[idx + 3] = A;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}
