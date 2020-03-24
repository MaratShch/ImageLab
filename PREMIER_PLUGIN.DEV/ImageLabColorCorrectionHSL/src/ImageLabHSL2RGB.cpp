#include "ImageLabColorCorrectionHSL.h"


/*
	INPUT RANGES  [floating point] -> H: 0...360  S,L : 0...100
	OUTPUT RANGES [fixed point]    -> R,G,B : 0...255
*/
bool hsl_to_bgr_precise_BGRA4444_8u
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
	csSDK_uint32 R, G, B;
	csSDK_int32 i, j, k;
	constexpr float reciproc3 = 1.0f / 3.0f;

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
				R = G = B = CLAMP_RGB8(static_cast<csSDK_int32>(L * 255.0f));
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
				fR = restore_rgb_channel_value (tmpVal1, tmpVal2, tmpR);
				fG = restore_rgb_channel_value (tmpVal1, tmpVal2, tmpG);
				fB = restore_rgb_channel_value (tmpVal1, tmpVal2, tmpB);

				R = CLAMP_RGB8(static_cast<csSDK_int32>(fR * 255.0f));
				G = CLAMP_RGB8(static_cast<csSDK_int32>(fG * 255.0f));
				B = CLAMP_RGB8(static_cast<csSDK_int32>(fB * 255.0f));

			}

			dstLine[i] = (srcLine[i] & 0xFF000000u) | (R << 16) | (G << 8) | B;	

			k += 3;
		}
	}

	return true;
}


/*
	INPUT RANGES  [floating point] -> H: 0...360  S,L : 0...100
	OUTPUT RANGES [fixed point]    -> R,G,B : 0...1.0
*/
bool hsl_to_bgr_precise_BGRA4444_32f
(
	const float* __restrict srcPix, /* src buffer used only for copy alpha channel values for destination */
	const float* __restrict pTmpBuffer,
	float* __restrict dstPix,
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
	csSDK_int32 i, j, k, idx;
	constexpr float reciproc3 = 1.0f / 3.0f;

	k = idx = 0;
	for (j = 0; j < height; j++)
	{
		const float* __restrict srcLine = &srcPix[j * linePitch];
		      float* __restrict dstLine = &dstPix[j * linePitch];

		for (i = 0; i < width; i++)
		{
			/* When H between range 0 and 360, and S and L in range bewteen 0 and 100 */
			H = pTmpBuffer[OFFSET_H(k)] / 360.0f;
			S = pTmpBuffer[OFFSET_S(k)] / 100.0f;
			L = pTmpBuffer[OFFSET_L(k)] / 100.0f;

			if (0.0f == S)
			{
				R = G = B = CLAMP_RGB32F(L);
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

				R = CLAMP_RGB32F(fR);
				G = CLAMP_RGB32F(fG);
				B = CLAMP_RGB32F(fB);
			}

			idx = i * 4;
			dstLine[idx    ] = B;
			dstLine[idx + 1] = G;
			dstLine[idx + 2] = R;
			dstLine[idx + 3] = srcLine[idx + 3];

			k += 3;
		}
	}

	return true;
}

/*
	INPUT RANGES  [floating point] -> H: 0...360  S,L : 0...100
	OUTPUT RANGES [fixed point]    -> R,G,B : 0...1.0
*/
bool hsl_to_bgr_precise_BGRA4444_16u
(
	const csSDK_uint32* __restrict srcPix, /* src buffer used only for copy alpha channel values for destination */
	const float* __restrict pTmpBuffer,
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
	csSDK_int32 R, G, B;
	csSDK_int32 i, j, k, idx;
	constexpr float reciproc3 = 1.0f / 3.0f;

	k = idx = 0;
	H = L = S = 0.0f;

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* __restrict srcLine = &srcPix[j * linePitch];
		      csSDK_uint32* __restrict dstLine = &dstPix[j * linePitch];

		for (i = 0; i < width; i++)
		{
			/* When H between range 0 and 360, and S and L in range bewteen 0 and 100 */
			H = pTmpBuffer[OFFSET_H(k)] / 360.0f;
			S = pTmpBuffer[OFFSET_S(k)] / 100.0f;
			L = pTmpBuffer[OFFSET_L(k)] / 100.0f;

			if (0.0f == S)
			{
				R = G = B = CLAMP_RGB16(static_cast<csSDK_int32>(L * 32768.0f));
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

				R = CLAMP_RGB16(static_cast<csSDK_int32>(fR * 32768.0f));
				G = CLAMP_RGB16(static_cast<csSDK_int32>(fG * 32768.0f));
				B = CLAMP_RGB16(static_cast<csSDK_int32>(fB * 32768.0f));
			}

			idx = i * 2;
			dstLine[idx    ] = (B | (G << 16));
			dstLine[idx + 1] = (R | (srcLine[idx + 1] & 0xFFFF0000u));

			k += 3;
		}
	}

	return true;
}

/*
	INPUT RANGES  [floating point] -> H: 0...360  S,L : 0...100
	OUTPUT RANGES [fixed point]    -> R,G,B : 0...255
*/
bool hsl_to_bgr_precise_ARGB4444_8u
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
	csSDK_uint32 R, G, B;
	csSDK_int32 i, j, k;
	constexpr float reciproc3 = 1.0f / 3.0f;

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
				R = G = B = CLAMP_RGB8(static_cast<csSDK_int32>(L * 255.0f));
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

				R = CLAMP_RGB8(static_cast<csSDK_int32>(fR * 255.0f));
				G = CLAMP_RGB8(static_cast<csSDK_int32>(fG * 255.0f));
				B = CLAMP_RGB8(static_cast<csSDK_int32>(fB * 255.0f));

			}

			dstLine[i] = (srcLine[i] & 0x000000FFu) | (R << 8) | (G << 16) | (B << 24);

			k += 3;
		}
	}

	return true;
}

/*
	INPUT RANGES  [floating point] -> H: 0...360  S,L : 0...100
	OUTPUT RANGES [fixed point]    -> R,G,B : 0...1.0
*/
bool hsl_to_bgr_precise_ARGB4444_16u
(
	const csSDK_uint32* __restrict srcPix, /* src buffer used only for copy alpha channel values for destination */
	const float* __restrict pTmpBuffer,
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
	csSDK_int32 R, G, B;
	csSDK_int32 i, j, k, idx;
	constexpr float reciproc3 = 1.0f / 3.0f;

	k = idx = 0;
	H = L = S = 0.0f;

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* __restrict srcLine = &srcPix[j * linePitch];
		csSDK_uint32* __restrict dstLine = &dstPix[j * linePitch];

		for (i = 0; i < width; i++)
		{
			/* When H between range 0 and 360, and S and L in range bewteen 0 and 100 */
			H = pTmpBuffer[OFFSET_H(k)] / 360.0f;
			S = pTmpBuffer[OFFSET_S(k)] / 100.0f;
			L = pTmpBuffer[OFFSET_L(k)] / 100.0f;

			if (0.0f == S)
			{
				R = G = B = CLAMP_RGB16(static_cast<csSDK_int32>(L * 32768.0f));
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

				R = CLAMP_RGB16(static_cast<csSDK_int32>(fR * 32768.0f));
				G = CLAMP_RGB16(static_cast<csSDK_int32>(fG * 32768.0f));
				B = CLAMP_RGB16(static_cast<csSDK_int32>(fB * 32768.0f));

			}

			idx = i * 2;
			dstLine[idx]     = ((srcLine[idx] & 0x0000FFFFu) | (R << 16));
			dstLine[idx + 1] = (G | (B << 16));

			k += 3;
		}
	}

	return true;
}

bool hsl_to_bgr_precise_ARGB4444_32f
(
	const float* __restrict srcPix, /* src buffer used only for copy alpha channel values for destination */
	const float* __restrict pTmpBuffer,
	float* __restrict dstPix,
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
	csSDK_int32 i, j, k, idx;
	constexpr float reciproc3 = 1.0f / 3.0f;

	k = idx = 0;
	for (j = 0; j < height; j++)
	{
		const float* __restrict srcLine = &srcPix[j * linePitch];
		float* __restrict dstLine = &dstPix[j * linePitch];

		for (i = 0; i < width; i++)
		{
			/* When H between range 0 and 360, and S and L in range bewteen 0 and 100 */
			H = pTmpBuffer[OFFSET_H(k)] / 360.0f;
			S = pTmpBuffer[OFFSET_S(k)] / 100.0f;
			L = pTmpBuffer[OFFSET_L(k)] / 100.0f;

			if (0.0f == S)
			{
				R = G = B = CLAMP_RGB32F(L);
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

				R = CLAMP_RGB32F(fR);
				G = CLAMP_RGB32F(fG);
				B = CLAMP_RGB32F(fB);
			}

			idx = i * 4;
			dstLine[idx    ] = srcLine[idx];
			dstLine[idx + 1] = R;
			dstLine[idx + 2] = G;
			dstLine[idx + 3] = B;

			k += 3;
		}
	}

	return true;
}