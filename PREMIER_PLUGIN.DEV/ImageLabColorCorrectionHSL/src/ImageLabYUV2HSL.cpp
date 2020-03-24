#include "ImageLabColorCorrectionHSL.h"

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


bool yuv_to_hsl_precise_VUYA4444_8u
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
	float V, U, Y;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k;
	csSDK_int32 y, u, v;

	k = 0;
	R = G = B = 0;

	const float* const __restrict pYuv2RgbMatrix = YUV2RGB[0];
	
	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			v = static_cast<csSDK_int32> (srcLine[i] & 0x000000FF);
			u = static_cast<csSDK_int32>((srcLine[i] & 0x0000FF00) >> 8);
			y = static_cast<csSDK_int32>((srcLine[i] & 0x00FF0000) >> 16);

			V = static_cast<float>(v - 128);
			U = static_cast<float>(u - 128);
			Y = static_cast<float>(y);

			/* convert VUY pixel to RGB in first */
			R = (Y * pYuv2RgbMatrix[0] + U * pYuv2RgbMatrix[1] + V * pYuv2RgbMatrix[2]) / 255.0f;
			G = (Y * pYuv2RgbMatrix[3] + U * pYuv2RgbMatrix[4] + V * pYuv2RgbMatrix[5]) / 255.0f;
			B = (Y * pYuv2RgbMatrix[6] + U * pYuv2RgbMatrix[7] + V * pYuv2RgbMatrix[8]) / 255.0f;

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


bool yuv_to_hsl_precise_VUYA4444_8u_709
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
	float V, U, Y;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k;
	csSDK_int32 y, u, v;

	k = 0;
	R = G = B = 0;

	const float* const __restrict pYuv2RgbMatrix = YUV2RGB[1];

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			v = static_cast<csSDK_int32> (srcLine[i] & 0x000000FF);
			u = static_cast<csSDK_int32>((srcLine[i] & 0x0000FF00) >> 8);
			y = static_cast<csSDK_int32>((srcLine[i] & 0x00FF0000) >> 16);

			V = static_cast<float>(v - 128);
			U = static_cast<float>(u - 128);
			Y = static_cast<float>(y);

			/* convert VUY pixel to RGB in first */
			R = (Y * pYuv2RgbMatrix[0] + U * pYuv2RgbMatrix[1] + V * pYuv2RgbMatrix[2]) / 255.0f;
			G = (Y * pYuv2RgbMatrix[3] + U * pYuv2RgbMatrix[4] + V * pYuv2RgbMatrix[5]) / 255.0f;
			B = (Y * pYuv2RgbMatrix[6] + U * pYuv2RgbMatrix[7] + V * pYuv2RgbMatrix[8]) / 255.0f;

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


bool yuv_to_hsl_precise_VUYA4444_32f
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
	float V, U, Y;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k, idx;
	csSDK_int32 y, u, v;

	k = idx = 0;
	R = G = B = 0;

	const float* const __restrict pYuv2RgbMatrix = YUV2RGB[0];

	for (j = 0; j < height; j++)
	{
		const float* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				idx = i << 2;

				V = srcLine[idx    ];
				U = srcLine[idx + 1];
				Y = srcLine[idx + 2];

				/* convert VUY pixel to RGB in first */
				R = (Y * pYuv2RgbMatrix[0] + U * pYuv2RgbMatrix[1] + V * pYuv2RgbMatrix[2]);
				G = (Y * pYuv2RgbMatrix[3] + U * pYuv2RgbMatrix[4] + V * pYuv2RgbMatrix[5]);
				B = (Y * pYuv2RgbMatrix[6] + U * pYuv2RgbMatrix[7] + V * pYuv2RgbMatrix[8]);

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


bool yuv_to_hsl_precise_VUYA4444_32f_709
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
	float V, U, Y;
	float minVal, maxVal;
	float sumMaxMin, subMaxMin;
	csSDK_int32 i, j, k, idx;
	csSDK_int32 y, u, v;

	k = idx = 0;
	R = G = B = 0;

	const float* const __restrict pYuv2RgbMatrix = YUV2RGB[1];

	for (j = 0; j < height; j++)
	{
		const float* srcLine = &srcPix[j * linePitch];

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i << 2;

			V = srcLine[idx];
			U = srcLine[idx + 1];
			Y = srcLine[idx + 2];

			/* convert VUY pixel to RGB in first */
			R = (Y * pYuv2RgbMatrix[0] + U * pYuv2RgbMatrix[1] + V * pYuv2RgbMatrix[2]);
			G = (Y * pYuv2RgbMatrix[3] + U * pYuv2RgbMatrix[4] + V * pYuv2RgbMatrix[5]);
			B = (Y * pYuv2RgbMatrix[6] + U * pYuv2RgbMatrix[7] + V * pYuv2RgbMatrix[8]);

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