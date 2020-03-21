#include "ImageLabColorCorrectionHSL.h"


/* 
   INPUT RANGES  -> R,G,B: 0...255
   OUTPUT RANGES -> H: 0.f...360.f  S,L: 0.f...100.f
*/
bool bgr_to_hsl_precise_BGRA4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	void* __restrict tmpBuf,
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

	float* __restrict pTmpBuffer = reinterpret_cast<float* __restrict>(tmpBuf);

	k = 0;
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
INPUT RANGES  -> R,G,B: 0...255
OUTPUT RANGES -> H: 0...360  S,L: 0...100
*/
bool bgr_to_hsl_BGRA4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	void* __restrict tmpBuf,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int32 addHue,
	const csSDK_int32 addLuminance,
	const csSDK_int32 addSaturation
)
{
	csSDK_int32 i, j, k;
	csSDK_int16 minVal, maxVal, sumMaxMin, subMaxMin;
	csSDK_int16 H, S, L;
	csSDK_int16 R, G, B;

	csSDK_int16* pTmpBuffer = reinterpret_cast<csSDK_int16*>(tmpBuf);

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* srcLine = &srcPix[j * linePitch];

		for (i = 0; i < width; i++)
		{
			B = static_cast<csSDK_int16> (srcLine[i] & 0x000000FFu);
			G = static_cast<csSDK_int16>((srcLine[i] & 0x0000FF00u) >> 8);
			R = static_cast<csSDK_int16>((srcLine[i] & 0x00FF0000u) >> 16);

			minVal = MIN(B, MIN(G, R));
			maxVal = MAX(B, MAX(G, R));

			sumMaxMin = maxVal + minVal;
			L = (sumMaxMin * 50 + 128) >> 8;

			if (maxVal == minVal)
			{
				S = H = 0;
			}
			else
			{
				subMaxMin = maxVal - minVal;

				S = (100 * subMaxMin) / ((L < 50) ? sumMaxMin : (510 - sumMaxMin));

				if (R == maxVal)
					H = (60 * (G - B)) / subMaxMin;
				else if (G == maxVal)
					H = (60 * (B - R)) / subMaxMin + 120;
				else
					H = (60 * (R - G)) / subMaxMin + 240;
			}

			const csSDK_int16 newHue = CLAMP_H (H + addHue);
			const csSDK_int16 newSat = CLAMP_LS(S + addSaturation);
			const csSDK_int16 newLum = CLAMP_LS(L + addLuminance);

			pTmpBuffer[OFFSET_H(k)] = newHue;
			pTmpBuffer[OFFSET_S(k)] = newSat;
			pTmpBuffer[OFFSET_L(k)] = newLum;

			k += 3;
		}
	}

	return true;
}
