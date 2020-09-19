#include "AdobeImageLabColorSubstitution.h"

void colorSubstitute_BGRA_4444_8u
(
	const csSDK_uint32* __restrict pSrc,
	      csSDK_uint32* __restrict pDst,
	const csSDK_int32&             height,
	const csSDK_int32&             width,
	const csSDK_int32&             linePitch,
	const prColor&                 from,
	const prColor&                 to,
	const csSDK_int32&             tolerance
)
{
	csSDK_int32 i = 0, j = 0;
	csSDK_int32 R = 0, G = 0, B = 0;
	csSDK_int32 newR = 0, newG = 0, newB = 0;

	const csSDK_int32 fromB = (from & 0x00FF0000) >> 16;
	const csSDK_int32 fromG = (from & 0x0000FF00) >> 8;
	const csSDK_int32 fromR = (from & 0x000000FF);

	const csSDK_int32 toB = (to & 0x00FF0000) >> 16;
	const csSDK_int32 toG = (to & 0x0000FF00) >> 8;
	const csSDK_int32 toR = (to & 0x000000FF);

	const csSDK_int32 rMin = fromR - tolerance;
	const csSDK_int32 rMax = fromR + tolerance;

	const csSDK_int32 gMin = fromG - tolerance;
	const csSDK_int32 gMax = fromG + tolerance;

	const csSDK_int32 bMin = fromB - tolerance;
	const csSDK_int32 bMax = fromB + tolerance;

	const csSDK_int32 addR = fromR + toR;
	const csSDK_int32 addG = fromG + toG;
	const csSDK_int32 addB = fromB + toB;
	

	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* __restrict pSrcLine = pSrc + j * linePitch;
		      csSDK_uint32* __restrict pDstLine = pDst + j * linePitch;
		 
		for (i = 0; i < width; i++)
		{
			R = (pSrcLine[i] & 0x00FF0000u) >> 16;
			G = (pSrcLine[i] & 0x0000FF00u) >> 8;
			B = (pSrcLine[i] & 0x000000FFu);

			if ( (R < rMax) && (R > rMin) && (G < gMax) && (G > gMin) && (B < bMax) && (B > bMin) )
			{
				newR = CLAMP_RGB8(addR - R);
				newG = CLAMP_RGB8(addG - G);
				newB = CLAMP_RGB8(addB - B);
			}
			else
			{
				newR = R;
				newG = G;
				newB = B;
			}

			pDstLine[i] = (pSrcLine[i] & 0xFF000000u) | (newR << 16) | (newG << 8) | newB;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}


void colorMask_BGRA_4444_8u
(
	const csSDK_uint32* __restrict pSrc,
	csSDK_uint32* __restrict pDst,
	const csSDK_int32&             height,
	const csSDK_int32&             width,
	const csSDK_int32&             linePitch,
	const prColor&                 from,
	const prColor&                 to,
	const csSDK_int32&             tolerance
)
{
	return;
}