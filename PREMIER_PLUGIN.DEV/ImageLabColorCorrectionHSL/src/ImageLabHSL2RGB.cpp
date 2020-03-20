#include "ImageLabColorCorrectionHSL.h"


template<typename T>
inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type
restore_rgb_channel_value(const T& t1, const T& t2, const T& t3)
{
	T val;

	if (6.0f * t3 < 1.0f)
	{
		val = t1 + (t2 - t1) * 6.0f * t3;
	}
	else if (2.0f * t3 < 1.0f)
	{
		val = t2;
	}
	else if (3.0f * t3 < 2.0f)
	{
		val = t1 + (t2 - t1) * (0.666f - t3) * 6.0f;
	}
	else
	{
		val = t1;
	}

	return val;
}

bool hsl_to_bgr_precise_BGRA4444_8u
(
	const csSDK_uint32* __restrict srcPix, /* src buffer used only for copy alpha channel values for destination */
	const float*  __restrict tmpBuf,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch
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
			H = tmpBuf[OFFSET_H(k)] / 360.0f;
			S = tmpBuf[OFFSET_S(k)] / 100.0f;
			L = tmpBuf[OFFSET_L(k)] / 100.0f;

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
					tmpR = tmpR - 1.0f;
				if (tmpB < 0.0f)
					tmpB = tmpB + 1.0f;

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
