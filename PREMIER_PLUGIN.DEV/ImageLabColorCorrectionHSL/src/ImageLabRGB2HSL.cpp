#include "ImageLabColorCorrectionHSL.h"


/* 
   INPUT RANGES  -> R,G,B: 0...255
   OUTPUT RANGES -> H: 0.f...360.f  S,L: 0.f...100.f
*/
bool bgr_to_hsl_precise_BGRA4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	float* __restrict pTmpBuffer,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const float addHue,
	const float addLuminance,
	const float addSaturation
)
{
	float H, S, L;
	float R, G, B;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k;

	k = 0;
	R = G = B = 0;

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			B = static_cast<float> (srcLine[i] & 0x000000FFu) / 255.0f;
			G = static_cast<float>((srcLine[i] & 0x0000FF00u) >> 8) / 255.0f;
			R = static_cast<float>((srcLine[i] & 0x00FF0000u) >> 16) / 255.0f;

			minVal = MIN(B, MIN(G, R));
			maxVal = MAX(B, MAX(G, R));

			sumMaxMin = maxVal + minVal;
			L = sumMaxMin * 50.0f; /* luminance value in percents = 100 * (max + min) / 2 */

			if (maxVal == minVal)
			{
				S = H = 0.0f;
			}
			else
			{
				subMaxMin = maxVal - minVal;

				S = (100.0f * subMaxMin) / ((L < 50.0f) ? sumMaxMin : (2.0f - sumMaxMin));

				if (R == maxVal)
					H = (60.0f * (G - B)) / subMaxMin;
				else if (G == maxVal)
					H = (60.0f * (B - R)) / subMaxMin + 120.0f;
				else
					H = (60.0f * (R - G)) / subMaxMin + 240.0f;
			}

			pTmpBuffer[OFFSET_H(k)] = CLAMP_H (H + addHue);			/* new HUE value in degrees			*/
			pTmpBuffer[OFFSET_S(k)] = CLAMP_LS(S + addSaturation);	/* new SATURATION value	in percents	*/
			pTmpBuffer[OFFSET_L(k)] = CLAMP_LS(L + addLuminance);	/* new LUMINANCE value in percents	*/

			k += 3;
		}
	}

	return true;
}


/*
	INPUT RANGES  -> R,G,B: 0...1.0
	OUTPUT RANGES -> H: 0.f...360.f  S,L: 0.f...100.f
*/
bool bgr_to_hsl_precise_BGRA4444_32f
(
	const float* __restrict srcPix,
	float* __restrict pTmpBuffer,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const float addHue,
	const float addLuminance,
	const float addSaturation
)
{
	float H, S, L;
	float R, G, B;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k, idx;

	k = idx = 0;
	R = G = B = 0;

	for (j = 0; j < height; j++)
	{
		const float* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i * 4;
			B = srcLine[idx    ];
			G = srcLine[idx + 1];
			R = srcLine[idx + 2];

			minVal = MIN(B, MIN(G, R));
			maxVal = MAX(B, MAX(G, R));

			sumMaxMin = maxVal + minVal;
			L = sumMaxMin * 50.0f; /* luminance value in percents = 100 * (max + min) / 2 */

			if (maxVal == minVal)
			{
				S = H = 0.0f;
			}
			else
			{
				subMaxMin = maxVal - minVal;

				S = (100.0f * subMaxMin) / ((L < 50.0f) ? sumMaxMin : (2.0f - sumMaxMin));

				if (R == maxVal)
					H = (60.0f * (G - B)) / subMaxMin;
				else if (G == maxVal)
					H = (60.0f * (B - R)) / subMaxMin + 120.0f;
				else
					H = (60.0f * (R - G)) / subMaxMin + 240.0f;
			}

			pTmpBuffer[OFFSET_H(k)] = CLAMP_H(H + addHue);			/* new HUE value in degrees			*/
			pTmpBuffer[OFFSET_S(k)] = CLAMP_LS(S + addSaturation);	/* new SATURATION value	in percents	*/
			pTmpBuffer[OFFSET_L(k)] = CLAMP_LS(L + addLuminance);	/* new LUMINANCE value in percents	*/

			k += 3;
		}
	}

	return true;
}


/*
	INPUT RANGES  -> R,G,B: 0...32768
	OUTPUT RANGES -> H: 0.f...360.f  S,L: 0.f...100.f
*/
bool bgr_to_hsl_precise_BGRA4444_16u
(
	const csSDK_uint32* __restrict srcPix,
	float* __restrict pTmpBuffer,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const float addHue,
	const float addLuminance,
	const float addSaturation
)
{
	float H, S, L;
	float R, G, B;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k, idx;

	k = idx = 0;
	R = G = B = 0;

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i * 2;
			B = static_cast<float>((srcLine[idx    ] & 0x0000FFFFu)) / 32768.0f;
			G = static_cast<float>((srcLine[idx    ] & 0xFFFF0000u) >> 16) / 32768.0f;
			R = static_cast<float>((srcLine[idx + 1] & 0x0000FFFFu)) / 32768.0f;

			minVal = MIN(B, MIN(G, R));
			maxVal = MAX(B, MAX(G, R));

			sumMaxMin = maxVal + minVal;
			L = sumMaxMin * 50.0f; /* luminance value in percents = 100 * (max + min) / 2 */

			if (maxVal == minVal)
			{
				S = H = 0.0f;
			}
			else
			{
				subMaxMin = maxVal - minVal;

				S = (100.0f * subMaxMin) / ((L < 50.0f) ? sumMaxMin : (2.0f - sumMaxMin));

				if (R == maxVal)
					H = (60.0f * (G - B)) / subMaxMin;
				else if (G == maxVal)
					H = (60.0f * (B - R)) / subMaxMin + 120.0f;
				else
					H = (60.0f * (R - G)) / subMaxMin + 240.0f;
			}

			pTmpBuffer[OFFSET_H(k)] = CLAMP_H (H + addHue);			/* new HUE value in degrees			*/
			pTmpBuffer[OFFSET_S(k)] = CLAMP_LS(S + addSaturation);	/* new SATURATION value	in percents	*/
			pTmpBuffer[OFFSET_L(k)] = CLAMP_LS(L + addLuminance);	/* new LUMINANCE value in percents	*/

			k += 3;
		}
	}

	return true;
}

/*
	INPUT RANGES  -> R,G,B: 0...255
	OUTPUT RANGES -> H: 0.f...360.f  S,L: 0.f...100.f
*/
bool bgr_to_hsl_precise_ARGB4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	float* __restrict pTmpBuffer,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const float addHue,
	const float addLuminance,
	const float addSaturation
)
{
	float H, S, L;
	float R, G, B;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k;

	k = 0;
	R = G = B = 0;

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				R = static_cast<float>((srcLine[i] & 0x0000FF00u) >> 8)  / 255.0f;
				G = static_cast<float>((srcLine[i] & 0x00FF0000u) >> 16) / 255.0f;
				B = static_cast<float>((srcLine[i] & 0xFF000000u) >> 24) / 255.0f;

				minVal = MIN(B, MIN(G, R));
				maxVal = MAX(B, MAX(G, R));

				sumMaxMin = maxVal + minVal;
				L = sumMaxMin * 50.0f; /* luminance value in percents = 100 * (max + min) / 2 */

				if (maxVal == minVal)
				{
					S = H = 0.0f;
				}
				else
				{
					subMaxMin = maxVal - minVal;

					S = (100.0f * subMaxMin) / ((L < 50.0f) ? sumMaxMin : (2.0f - sumMaxMin));

					if (R == maxVal)
						H = (60.0f * (G - B)) / subMaxMin;
					else if (G == maxVal)
						H = (60.0f * (B - R)) / subMaxMin + 120.0f;
					else
						H = (60.0f * (R - G)) / subMaxMin + 240.0f;
				}

				pTmpBuffer[OFFSET_H(k)] = CLAMP_H (H + addHue);			/* new HUE value in degrees			*/
				pTmpBuffer[OFFSET_S(k)] = CLAMP_LS(S + addSaturation);	/* new SATURATION value	in percents	*/
				pTmpBuffer[OFFSET_L(k)] = CLAMP_LS(L + addLuminance);	/* new LUMINANCE value in percents	*/

				k += 3;
			}
	}

	return true;
}

/*
	INPUT RANGES  -> R,G,B: 0...32768
	OUTPUT RANGES -> H: 0.f...360.f  S,L: 0.f...100.f
*/
bool bgr_to_hsl_precise_ARGB4444_16u
(
	const csSDK_uint32* __restrict srcPix,
	float* __restrict pTmpBuffer,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const float addHue,
	const float addLuminance,
	const float addSaturation
)
{
	float H, S, L;
	float R, G, B;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k, idx;

	k = idx = 0;
	R = G = B = 0.0f;

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				idx = i * 2;
				R = static_cast<float>((srcLine[idx] & 0xFFFF0000u) >> 16) / 32768.0f;
				G = static_cast<float>((srcLine[idx + 1] & 0x0000FFFFu))   / 32768.0f;
				B = static_cast<float>((srcLine[idx + 1] & 0xFFFF0000u) >> 16) / 32768.0f;

				minVal = MIN(B, MIN(G, R));
				maxVal = MAX(B, MAX(G, R));

				sumMaxMin = maxVal + minVal;
				L = sumMaxMin * 50.0f; /* luminance value in percents = 100 * (max + min) / 2 */

				if (maxVal == minVal)
				{
					S = H = 0.0f;
				}
				else
				{
					subMaxMin = maxVal - minVal;

					S = (100.0f * subMaxMin) / ((L < 50.0f) ? sumMaxMin : (2.0f - sumMaxMin));

					if (R == maxVal)
						H = (60.0f * (G - B)) / subMaxMin;
					else if (G == maxVal)
						H = (60.0f * (B - R)) / subMaxMin + 120.0f;
					else
						H = (60.0f * (R - G)) / subMaxMin + 240.0f;
				}

				pTmpBuffer[OFFSET_H(k)] = CLAMP_H (H + addHue);			/* new HUE value in degrees			*/
				pTmpBuffer[OFFSET_S(k)] = CLAMP_LS(S + addSaturation);	/* new SATURATION value	in percents	*/
				pTmpBuffer[OFFSET_L(k)] = CLAMP_LS(L + addLuminance);	/* new LUMINANCE value in percents	*/

				k += 3;
			}
	}

	return true;
}

/*
	INPUT RANGES  -> R,G,B: 0...1.0
	OUTPUT RANGES -> H: 0.f...360.f  S,L: 0.f...100.f
*/
bool bgr_to_hsl_precise_ARGB4444_32f
(
	const float* __restrict srcPix,
	float* __restrict pTmpBuffer,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const float addHue,
	const float addLuminance,
	const float addSaturation
)
{
	float H, S, L;
	float R, G, B;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k, idx;

	k = idx = 0;
	R = G = B = 0;

	for (j = 0; j < height; j++)
	{
		const float* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i * 4;
			R = srcLine[idx + 1];
			G = srcLine[idx + 2];
			B = srcLine[idx + 3];

			minVal = MIN(B, MIN(G, R));
			maxVal = MAX(B, MAX(G, R));

			sumMaxMin = maxVal + minVal;
			L = sumMaxMin * 50.0f; /* luminance value in percents = 100 * (max + min) / 2 */

			if (maxVal == minVal)
			{
				S = H = 0.0f;
			}
			else
			{
				subMaxMin = maxVal - minVal;

				S = (100.0f * subMaxMin) / ((L < 50.0f) ? sumMaxMin : (2.0f - sumMaxMin));

				if (R == maxVal)
					H = (60.0f * (G - B)) / subMaxMin;
				else if (G == maxVal)
					H = (60.0f * (B - R)) / subMaxMin + 120.0f;
				else
					H = (60.0f * (R - G)) / subMaxMin + 240.0f;
			}

			pTmpBuffer[OFFSET_H(k)] = CLAMP_H(H + addHue);			/* new HUE value in degrees			*/
			pTmpBuffer[OFFSET_S(k)] = CLAMP_LS(S + addSaturation);	/* new SATURATION value	in percents	*/
			pTmpBuffer[OFFSET_L(k)] = CLAMP_LS(L + addLuminance);	/* new LUMINANCE value in percents	*/

			k += 3;
		}
	}

	return true;
}


/*
	INPUT RANGES  -> R,G,B: 0...1023
	OUTPUT RANGES -> H: 0.f...360.f  S,L: 0.f...100.f
*/
bool bgr_to_hsl_precise_RGB444_10u
(
	const csSDK_uint32* __restrict srcPix,
	float* __restrict pTmpBuffer,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const float addHue,
	const float addLuminance,
	const float addSaturation
)
{
	float H, S, L;
	float R, G, B;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k;

	k = 0;
	R = G = B = 0;

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			B = (static_cast<float>((srcLine[i] & 0x00000FFCu) >> 2)  / 1024.0f);
			G = (static_cast<float>((srcLine[i] & 0x003FF000u) >> 12) / 1024.0f);
			R = (static_cast<float>((srcLine[i] & 0xFFC00000u) >> 22) / 1024.0f);

			minVal = MIN(B, MIN(G, R));
			maxVal = MAX(B, MAX(G, R));

			sumMaxMin = maxVal + minVal;
			L = sumMaxMin * 50.0f; /* luminance value in percents = 100 * (max + min) / 2 */

			if (maxVal == minVal)
			{
				S = H = 0.0f;
			}
			else
			{
				subMaxMin = maxVal - minVal;

				S = (100.0f * subMaxMin) / ((L < 50.0f) ? sumMaxMin : (2.0f - sumMaxMin));

				if (R == maxVal)
					H = (60.0f * (G - B)) / subMaxMin;
				else if (G == maxVal)
					H = (60.0f * (B - R)) / subMaxMin + 120.0f;
				else
					H = (60.0f * (R - G)) / subMaxMin + 240.0f;
			}

			pTmpBuffer[OFFSET_H(k)] = CLAMP_H (H + addHue);			/* new HUE value in degrees			*/
			pTmpBuffer[OFFSET_S(k)] = CLAMP_LS(S + addSaturation);	/* new SATURATION value	in percents	*/
			pTmpBuffer[OFFSET_L(k)] = CLAMP_LS(L + addLuminance);	/* new LUMINANCE value in percents	*/

			k += 3;
		}
	}

	return true;
}