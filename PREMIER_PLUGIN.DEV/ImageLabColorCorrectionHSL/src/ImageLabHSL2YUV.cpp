#include "ImageLabColorCorrectionHSL.h"

CACHE_ALIGN constexpr float RGB2YUV[][9] =
{
	// BT.601
	{
		0.2990000f,  0.587000f,  0.114000f,
	   -0.1687360f, -0.331264f,  0.500000f,
		0.5000000f, -0.418688f, -0.081312f
	},

	// BT.709
	{
		0.212600f,   0.715200f,  0.072200f,
	   -0.114570f,  -0.385430f,  0.500000f,
		0.500000f,  -0.454150f, -0.045850f
	}
};

/*
	INPUT RANGES  [floating point] -> H: 0...360  S,L : 0...100
	OUTPUT RANGES [fixed point]    -> Y,U,V : 0...255
*/
bool hsl_to_yuv_precise_VUYA4444_8u
(
	const csSDK_uint32* __restrict srcPix, /* src buffer used only for copy alpha channel values for destination */
	const float*  __restrict pTmpBuffer,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch
)
{
	float H, L, S;
	float tmpR, tmpG, tmpB;
	float tmpVal1, tmpVal2, h;
	float fR, fG, fB;
	float R, G, B;
	csSDK_int32 i, j, k;
	csSDK_int32 Y, U, V;
	constexpr float reciproc3 = 1.0f / 3.0f;

	const float* const __restrict pRgb2YuvMatrix = RGB2YUV[0];

	k = 0;
	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* srcLine = &srcPix[j * linePitch];
		      csSDK_uint32* dstLine = &dstPix[j * linePitch];

			  for (i = 0; i < width; i++)
			  {
				  /* When H between range 0 and 360, and S and L in range bewteen 0 and 100 */
				  H = pTmpBuffer[OFFSET_H(k)] / 360.0f;
				  S = pTmpBuffer[OFFSET_S(k)] / 100.0f;
				  L = pTmpBuffer[OFFSET_L(k)] / 100.0f;

				  if (0.0f == S)
				  {
					  R = G = B = L * 255.0f;
				  } /* if (0.0f == S) */
				  else
				  {
					  tmpVal2 = (L < 0.50f) ? (L * (1.0f + S)) : (L + S - (L * S));
					  tmpVal1 = 2.0f * L - tmpVal2;

					  tmpG = H;
					  tmpR = H + reciproc3;
					  tmpB = H - reciproc3;

					  if (tmpR > 1.0f)
						  tmpR -= 1.0f;
					  if (tmpB < 0.0f)
						  tmpB += 1.0f;

					  /* restore RGB channels */
					  fR = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpR);
					  fG = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpG);
					  fB = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpB);

					  R = fR * 255.0f;
					  G = fG * 255.0f;
					  B = fB * 255.0f;
				  }

				  Y = static_cast<csSDK_int32>(R * pRgb2YuvMatrix[0] + G * pRgb2YuvMatrix[1] + B * pRgb2YuvMatrix[2]);
				  U = static_cast<csSDK_int32>(R * pRgb2YuvMatrix[3] + G * pRgb2YuvMatrix[4] + B * pRgb2YuvMatrix[5]) + 128;
				  V = static_cast<csSDK_int32>(R * pRgb2YuvMatrix[6] + G * pRgb2YuvMatrix[7] + B * pRgb2YuvMatrix[8]) + 128;

				  dstLine[i] = CLAMP_RGB8(V) | (CLAMP_RGB8(U) << 8) | (CLAMP_RGB8(Y) << 16) | (srcLine[i] & 0xFF000000u);

			k += 3;
		}
	}

	return true;
}


bool hsl_to_yuv_precise_VUYA4444_8u_709
(
	const csSDK_uint32* __restrict srcPix, /* src buffer used only for copy alpha channel values for destination */
	const float*  __restrict pTmpBuffer,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch
)
{
	float H, L, S;
	float tmpR, tmpG, tmpB;
	float tmpVal1, tmpVal2, h;
	float fR, fG, fB;
	float R, G, B;
	csSDK_int32 i, j, k;
	csSDK_int32 Y, U, V;
	constexpr float reciproc3 = 1.0f / 3.0f;

	const float* const __restrict pRgb2YuvMatrix = RGB2YUV[1];

	k = 0;
	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* srcLine = &srcPix[j * linePitch];
		csSDK_uint32* dstLine = &dstPix[j * linePitch];

		for (i = 0; i < width; i++)
		{
			/* When H between range 0 and 360, and S and L in range bewteen 0 and 100 */
			H = pTmpBuffer[OFFSET_H(k)] / 360.0f;
			S = pTmpBuffer[OFFSET_S(k)] / 100.0f;
			L = pTmpBuffer[OFFSET_L(k)] / 100.0f;

			if (0.0f == S)
			{
				R = G = B = L * 255.0f;
			} /* if (0.0f == S) */
			else
			{
				tmpVal2 = (L < 0.50f) ? (L * (1.0f + S)) : (L + S - (L * S));
				tmpVal1 = 2.0f * L - tmpVal2;

				tmpG = H;
				tmpR = H + reciproc3;
				tmpB = H - reciproc3;

				if (tmpR > 1.0f)
					tmpR -= 1.0f;
				if (tmpB < 0.0f)
					tmpB += 1.0f;

				/* restore RGB channels */
				fR = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpR);
				fG = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpG);
				fB = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpB);

				R = fR * 255.0f;
				G = fG * 255.0f;
				B = fB * 255.0f;
			}

			Y = static_cast<csSDK_int32>(R * pRgb2YuvMatrix[0] + G * pRgb2YuvMatrix[1] + B * pRgb2YuvMatrix[2]);
			U = static_cast<csSDK_int32>(R * pRgb2YuvMatrix[3] + G * pRgb2YuvMatrix[4] + B * pRgb2YuvMatrix[5]) + 128;
			V = static_cast<csSDK_int32>(R * pRgb2YuvMatrix[6] + G * pRgb2YuvMatrix[7] + B * pRgb2YuvMatrix[8]) + 128;

			dstLine[i] = CLAMP_RGB8(V) | (CLAMP_RGB8(U) << 8) | (CLAMP_RGB8(Y) << 16) | (srcLine[i] & 0xFF000000u);

			k += 3;
		}
	}

	return true;
}


/*
INPUT RANGES  [floating point] -> H: 0...360  S,L : 0...100
OUTPUT RANGES [fixed point]    -> Y,U,V : 0...255
*/
bool hsl_to_yuv_precise_VUYA4444_32f
(
	const float* __restrict srcPix, /* src buffer used only for copy alpha channel values for destination */
	const float*  __restrict pTmpBuffer,
	float* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch
)
{
	float H, L, S;
	float R, G, B;
	float Y, U, V;
	float tmpR, tmpG, tmpB;
	float tmpVal1, tmpVal2, h;
	csSDK_int32 i, j, k, idx;
	constexpr float reciproc3 = 1.0f / 3.0f;

	const float* const __restrict pRgb2YuvMatrix = RGB2YUV[0];

	k = idx = 0;
	for (j = 0; j < height; j++)
	{
		const float* srcLine = &srcPix[j * linePitch];
		      float* dstLine = &dstPix[j * linePitch];

		for (i = 0; i < width; i++)
		{
			/* When H between range 0 and 360, and S and L in range bewteen 0 and 100 */
			H = pTmpBuffer[OFFSET_H(k)] / 360.0f;
			S = pTmpBuffer[OFFSET_S(k)] / 100.0f;
			L = pTmpBuffer[OFFSET_L(k)] / 100.0f;

			if (0.0f == S)
			{
				R = G = B = L;
			} /* if (0.0f == S) */
			else
			{
				tmpVal2 = (L < 0.50f) ? (L * (1.0f + S)) : (L + S - (L * S));
				tmpVal1 = 2.0f * L - tmpVal2;

				tmpG = H;
				tmpR = H + reciproc3;
				tmpB = H - reciproc3;

				if (tmpR > 1.0f)
					tmpR -= 1.0f;
				if (tmpB < 0.0f)
					tmpB += 1.0f;

				/* restore RGB channels */
				R = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpR);
				G = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpG);
				B = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpB);

			}

			Y = R * pRgb2YuvMatrix[0] + G * pRgb2YuvMatrix[1] + B * pRgb2YuvMatrix[2];
			U = R * pRgb2YuvMatrix[3] + G * pRgb2YuvMatrix[4] + B * pRgb2YuvMatrix[5];
			V = R * pRgb2YuvMatrix[6] + G * pRgb2YuvMatrix[7] + B * pRgb2YuvMatrix[8];

			idx = i << 2;
			dstLine[idx]     = V;
			dstLine[idx + 1] = U;
			dstLine[idx + 2] = Y;
			dstLine[idx + 3] = srcLine[idx + 3];

			k += 3;
		}
	}

	return true;
}

bool hsl_to_yuv_precise_VUYA4444_32f_709
(
	const float* __restrict srcPix, /* src buffer used only for copy alpha channel values for destination */
	const float*  __restrict pTmpBuffer,
	float* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch
)
{
	float H, L, S;
	float R, G, B;
	float Y, U, V;
	float tmpR, tmpG, tmpB;
	float tmpVal1, tmpVal2, h;
	csSDK_int32 i, j, k, idx;
	constexpr float reciproc3 = 1.0f / 3.0f;

	const float* const __restrict pRgb2YuvMatrix = RGB2YUV[1];

	k = idx = 0;
	for (j = 0; j < height; j++)
	{
		const float* srcLine = &srcPix[j * linePitch];
		float* dstLine = &dstPix[j * linePitch];

		for (i = 0; i < width; i++)
		{
			/* When H between range 0 and 360, and S and L in range bewteen 0 and 100 */
			H = pTmpBuffer[OFFSET_H(k)] / 360.0f;
			S = pTmpBuffer[OFFSET_S(k)] / 100.0f;
			L = pTmpBuffer[OFFSET_L(k)] / 100.0f;

			if (0.0f == S)
			{
				R = G = B = L;
			} /* if (0.0f == S) */
			else
			{
				tmpVal2 = (L < 0.50f) ? (L * (1.0f + S)) : (L + S - (L * S));
				tmpVal1 = 2.0f * L - tmpVal2;

				tmpG = H;
				tmpR = H + reciproc3;
				tmpB = H - reciproc3;

				if (tmpR > 1.0f)
					tmpR -= 1.0f;
				if (tmpB < 0.0f)
					tmpB += 1.0f;

				/* restore RGB channels */
				R = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpR);
				G = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpG);
				B = restore_rgb_channel_value(tmpVal1, tmpVal2, tmpB);

			}

			Y = R * pRgb2YuvMatrix[0] + G * pRgb2YuvMatrix[1] + B * pRgb2YuvMatrix[2];
			U = R * pRgb2YuvMatrix[3] + G * pRgb2YuvMatrix[4] + B * pRgb2YuvMatrix[5];
			V = R * pRgb2YuvMatrix[6] + G * pRgb2YuvMatrix[7] + B * pRgb2YuvMatrix[8];

			idx = i << 2;
			dstLine[idx] = V;
			dstLine[idx + 1] = U;
			dstLine[idx + 2] = Y;
			dstLine[idx + 3] = srcLine[idx + 3];

			k += 3;
		}
	}

	return true;
}