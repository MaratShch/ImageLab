#include "ImageLabFuzzyMedian.h"

// r,g,b values are from 0 to 255 (8 bits unsigned)
// h = [0,360], s = [0,1], v = [0,1]
void convert_hsv_to_rgb_4444_BGRA8u
(
	const csSDK_uint32* __restrict pSrc,
	const float*  __restrict pHSV, /* buffer layout: H, S, V*/
	csSDK_uint32* __restrict pDst,
	csSDK_int32 width,
	csSDK_int32 height,
	csSDK_int32 linePitch
)
{
	csSDK_int32 j, i, k, idx;
	csSDK_int32 R, G, B;
	float H, S, V;
	float newR, newG, newB;
	float hh, p, q, t, ff;

	if (nullptr == pSrc || nullptr == pHSV || nullptr == pDst)
		return;

	constexpr float reciproc_60 = 1.0f / 60.0f;

	R = G = B = 0;
	H = S = V = 0.f;
	k = 0;

	const csSDK_int32 tripleWidth = width * 3;

	for (j = 0; j < height; j++)
	{
		const        float* __restrict HSVImg  = &pHSV[j * tripleWidth];
		const csSDK_uint32* __restrict srcImg  = &pSrc[j * linePitch];
		      csSDK_uint32* __restrict dstLine = &pDst[j * linePitch];

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i * 3;
			H = HSVImg[OFFSET_H(idx)];
			S = HSVImg[OFFSET_S(idx)];
			V = HSVImg[OFFSET_V(idx)];

			if (0.f == S)
			{
				/* not color pixel - B/W */
				newR = newG = newB = V;
			}
			else
			{
				hh = H * reciproc_60; /* sector 0 to 5 */
				k = static_cast<csSDK_int32>(hh);
				ff = hh - static_cast<float>(k);

				p = V * (1.0f - S);
				q = V * (1.0f - S * ff);
				t = V * (1.0f - S * (1.0f - ff));

				switch (k)
				{
					case 0:
						newR = V;
						newG = t;
						newB = p;
					break;
					case 1:
						newR = q;
						newG = V;
						newB = p;
					break;
					case 2:
						newR = p;
						newG = V;
						newB = t;
					break;
					case 3:
						newR = p;
						newG = q;
						newB = V;
					break;
					case 4:
						newR = t;
						newG = p;
						newB = V;
					break;
					case 5:
					default:
						newR = V;
						newG = p;
						newB = q;
					break;
				}

			}

			R = static_cast<csSDK_int32>(newR * 255.0f);
			G = static_cast<csSDK_int32>(newG * 255.0f);
			B = static_cast<csSDK_int32>(newB * 255.0f);

			dstLine[i] = (srcImg[i] & 0xFF000000u) |
						   ((CLAMP_RGB8(R)) << 16) |
						   ((CLAMP_RGB8(G)) << 8)  |
				            (CLAMP_RGB8(B));

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}