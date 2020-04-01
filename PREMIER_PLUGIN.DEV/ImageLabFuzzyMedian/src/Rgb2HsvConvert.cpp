#include "ImageLabFuzzyMedian.h"

// r,g,b values are from 0 to 255 (8 bits unsigned)
// h = [0,360], s = [0,1], v = [0,1]
void convert_rgb_to_hsv_4444_BGRA8u
(
	const csSDK_uint32* __restrict pSrc,
	float* __restrict pDstV, /* buffer layout: H, S, V*/
	csSDK_int32 width,
	csSDK_int32 height,
	csSDK_int32 linePitch
)
{
	float R, G, B;
	float H, S, V;
	float minVal, maxVal, delta;
	csSDK_int32 i, j, idx;

	const csSDK_int32 tripleWidth = width * 3; /* for store HSV values */

	R = G = B = 0.0f;
	H = S = V = 0.0f;

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* __restrict srcLine = &pSrc [j * linePitch];
		             float* __restrict dstLine = &pDstV[j * tripleWidth];

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			B = static_cast<float> (srcLine[i] & 0x000000FFu)        / 255.0f;
			G = static_cast<float>((srcLine[i] & 0x0000FF00u) >> 8)  / 255.0f;
			R = static_cast<float>((srcLine[i] & 0x00FF0000u) >> 16) / 255.0f;

			minVal = MIN(R, MIN(G, B));
			V = maxVal = MAX(R, MAX(G, B));

			if (0 == maxVal)
				S = H = 0.0f;
			else
			{
				delta = maxVal - minVal;
				S = delta / maxVal;

				if (maxVal == R)
					H = (G - B) / delta;		/* between Yellow and Magenta*/
				else if (maxVal == G)
					H = 2 + (B - R) / delta;	/* between Cyan and Yellow */
				else
					H = 4 + (R - G) / delta;    /* between Magenta and Cyan */

				H *= 60.0f; /* convert to degrees */
				if (H < 0)
					H += 360.0f;
			}

			idx = i * 3;

			dstLine[idx    ] = H;
			dstLine[idx + 1] = S;
			dstLine[idx + 2] = V;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}