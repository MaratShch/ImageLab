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
	const csSDK_int32&             tolerance,
	const bool&                    showMask
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
		 
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			R = (pSrcLine[i] & 0x00FF0000u) >> 16;
			G = (pSrcLine[i] & 0x0000FF00u) >> 8;
			B = (pSrcLine[i] & 0x000000FFu);

			if ( (R < rMax) && (R > rMin) && (G < gMax) && (G > gMin) && (B < bMax) && (B > bMin) )
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

			pDstLine[i] = (pSrcLine[i] & 0xFF000000u) | (newR << 16) | (newG << 8) | newB;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}


void colorSubstitute_BGRA_4444_16u
(
	const csSDK_uint32* __restrict pSrc,
	      csSDK_uint32* __restrict pDst,
	const csSDK_int32&             height,
	const csSDK_int32&             width,
	const csSDK_int32&             linePitch,
	const prColor&                 from,
	const prColor&                 to,
	const csSDK_int32&             tolerance,
	const bool&                    showMask
)
{
	csSDK_int32 i = 0, j = 0, idx = 0;
	csSDK_int32 R = 0, G = 0, B = 0;
	csSDK_int32 newR = 0, newG = 0, newB = 0;

	const csSDK_int32 adaptedTolerance = tolerance << 7;
	const csSDK_int32 fromB = ((from & 0x00FF0000) >> 16) << 7;
	const csSDK_int32 fromG = ((from & 0x0000FF00) >> 8)  << 7;
	const csSDK_int32 fromR =  (from & 0x000000FF)        << 7;

	const csSDK_int32 toB = ((to & 0x00FF0000) >> 16) << 7;
	const csSDK_int32 toG = ((to & 0x0000FF00) >> 8)  << 7;
	const csSDK_int32 toR =  (to & 0x000000FF)        << 7;

	const csSDK_int32 rMin = fromR - adaptedTolerance;
	const csSDK_int32 rMax = fromR + adaptedTolerance;

	const csSDK_int32 gMin = fromG - adaptedTolerance;
	const csSDK_int32 gMax = fromG + adaptedTolerance;

	const csSDK_int32 bMin = fromB - adaptedTolerance;
	const csSDK_int32 bMax = fromB + adaptedTolerance;

	const csSDK_int32 addR = fromR + toR;
	const csSDK_int32 addG = fromG + toG;
	const csSDK_int32 addB = fromB + toB;


	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* __restrict pSrcLine = pSrc + j * linePitch;
		      csSDK_uint32* __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i << 1;

			B = (pSrcLine[idx    ] & 0x0000FFFFu);
			G = (pSrcLine[idx    ] & 0xFFFF0000u) >> 16;
			R = (pSrcLine[idx + 1] & 0x0000FFFFu);

			if ((R < rMax) && (R > rMin) && (G < gMax) && (G > gMin) && (B < bMax) && (B > bMin))
			{
				if (false == showMask)
				{
					newR = CLAMP_RGB16(addR - R);
					newG = CLAMP_RGB16(addG - G);
					newB = CLAMP_RGB16(addB - B);
				}
				else
					newR = newG = newB = 0x7FFF;
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

			pDstLine[idx    ] = (newG << 16) | newB;
			pDstLine[idx + 1] = (pSrcLine[idx + 1] & 0xFFFF0000u) | newR;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}


void colorSubstitute_BGRA_4444_32f
(
	const float* __restrict pSrc,
	      float* __restrict pDst,
	const csSDK_int32&      height,
	const csSDK_int32&      width,
	const csSDK_int32&      linePitch,
	const prColor&          from,
	const prColor&          to,
	const csSDK_int32&      tolerance,
	const bool&             showMask
)
{
	csSDK_int32 i = 0, j = 0, idx = 0;
	float R = 0.f, G = 0.f, B = 0.f, A = 0.f;
	float newR = 0.f, newG = 0.f, newB = 0.f;

	const float adaptedTolerance = static_cast<float>(tolerance) / 256.0f;
	const float fromB = static_cast<float>((from & 0x00FF0000) >> 16) / 256.0f ;
	const float fromG = static_cast<float>((from & 0x0000FF00) >> 8)  / 256.0f;
	const float fromR = static_cast<float>( from & 0x000000FF) / 256.0f;

	const float toB = static_cast<float>((to & 0x00FF0000) >> 16) / 256.0f;
	const float toG = static_cast<float>((to & 0x0000FF00) >> 8)  / 256.0f;
	const float toR = static_cast<float>( to & 0x000000FF) / 256.0f;

	const float rMin = fromR - adaptedTolerance;
	const float rMax = fromR + adaptedTolerance;

	const float gMin = fromG - adaptedTolerance;
	const float gMax = fromG + adaptedTolerance;

	const float bMin = fromB - adaptedTolerance;
	const float bMax = fromB + adaptedTolerance;

	const float addR = fromR + toR;
	const float addG = fromG + toG;
	const float addB = fromB + toB;


	for (j = 0; j < height; j++)
	{
		const float* __restrict pSrcLine = pSrc + j * linePitch;
		      float* __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i << 2;

			B = pSrcLine[idx    ];
			G = pSrcLine[idx + 1];
			R = pSrcLine[idx + 2];
			A = pSrcLine[idx + 3];

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

			pDstLine[idx    ] = newB;
			pDstLine[idx + 1] = newG;
			pDstLine[idx + 2] = newR;
			pDstLine[idx + 3] = A;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}


void colorSubstitute_ARGB_4444_8u
(
	const csSDK_uint32* __restrict pSrc,
	csSDK_uint32* __restrict pDst,
	const csSDK_int32&             height,
	const csSDK_int32&             width,
	const csSDK_int32&             linePitch,
	const prColor&                 from,
	const prColor&                 to,
	const csSDK_int32&             tolerance,
	const bool&                    showMask
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

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			R = (pSrcLine[i] & 0x0000FF00u) >> 8;
			G = (pSrcLine[i] & 0x00FF0000u) >> 16;
			B = (pSrcLine[i] & 0xFF000000u) >> 24;

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

			pDstLine[i] = (pSrcLine[i] & 0x000000FFu) | (newR << 8) | (newG << 16) | (newB << 24);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}


void colorSubstitute_ARGB_4444_16u
(
	const csSDK_uint32* __restrict pSrc,
	csSDK_uint32* __restrict pDst,
	const csSDK_int32&             height,
	const csSDK_int32&             width,
	const csSDK_int32&             linePitch,
	const prColor&                 from,
	const prColor&                 to,
	const csSDK_int32&             tolerance,
	const bool&                    showMask
)
{
	csSDK_int32 i = 0, j = 0, idx = 0;
	csSDK_int32 R = 0, G = 0, B = 0;
	csSDK_int32 newR = 0, newG = 0, newB = 0;

	const csSDK_int32 adaptedTolerance = tolerance << 7;
	const csSDK_int32 fromB = ((from & 0x00FF0000) >> 16) << 7;
	const csSDK_int32 fromG = ((from & 0x0000FF00) >> 8) << 7;
	const csSDK_int32 fromR = (from & 0x000000FF) << 7;

	const csSDK_int32 toB = ((to & 0x00FF0000) >> 16) << 7;
	const csSDK_int32 toG = ((to & 0x0000FF00) >> 8) << 7;
	const csSDK_int32 toR = (to & 0x000000FF) << 7;

	const csSDK_int32 rMin = fromR - adaptedTolerance;
	const csSDK_int32 rMax = fromR + adaptedTolerance;

	const csSDK_int32 gMin = fromG - adaptedTolerance;
	const csSDK_int32 gMax = fromG + adaptedTolerance;

	const csSDK_int32 bMin = fromB - adaptedTolerance;
	const csSDK_int32 bMax = fromB + adaptedTolerance;

	const csSDK_int32 addR = fromR + toR;
	const csSDK_int32 addG = fromG + toG;
	const csSDK_int32 addB = fromB + toB;


	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* __restrict pSrcLine = pSrc + j * linePitch;
		csSDK_uint32* __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i << 1;

			R = (pSrcLine[idx    ] & 0xFFFF0000u);
			G = (pSrcLine[idx + 1] & 0x0000FFFFu);
			B = (pSrcLine[idx + 1] & 0xFFFF0000u) >> 16;

			if ((R < rMax) && (R > rMin) && (G < gMax) && (G > gMin) && (B < bMax) && (B > bMin))
			{
				if (false == showMask)
				{
					newR = CLAMP_RGB16(addR - R);
					newG = CLAMP_RGB16(addG - G);
					newB = CLAMP_RGB16(addB - B);
				}
				else
					newR = newG = newB = 0x7FFF;
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

			pDstLine[idx    ] = (pSrcLine[idx] & 0x0000FFFFu) | (newR << 16);
			pDstLine[idx + 1] = newG  | (newB << 16);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}


void colorSubstitute_ARGB_4444_32f
(
	const float* __restrict pSrc,
	float* __restrict pDst,
	const csSDK_int32&      height,
	const csSDK_int32&      width,
	const csSDK_int32&      linePitch,
	const prColor&          from,
	const prColor&          to,
	const csSDK_int32&      tolerance,
	const bool&             showMask
)
{
	csSDK_int32 i = 0, j = 0, idx = 0;
	float R = 0.f, G = 0.f, B = 0.f, A = 0.f;
	float newR = 0.f, newG = 0.f, newB = 0.f;

	const float adaptedTolerance = static_cast<float>(tolerance) / 256.0f;
	const float fromB = static_cast<float>((from & 0x00FF0000) >> 16) / 256.0f;
	const float fromG = static_cast<float>((from & 0x0000FF00) >> 8) / 256.0f;
	const float fromR = static_cast<float>(from & 0x000000FF) / 256.0f;

	const float toB = static_cast<float>((to & 0x00FF0000) >> 16) / 256.0f;
	const float toG = static_cast<float>((to & 0x0000FF00) >> 8) / 256.0f;
	const float toR = static_cast<float>(to & 0x000000FF) / 256.0f;

	const float rMin = fromR - adaptedTolerance;
	const float rMax = fromR + adaptedTolerance;

	const float gMin = fromG - adaptedTolerance;
	const float gMax = fromG + adaptedTolerance;

	const float bMin = fromB - adaptedTolerance;
	const float bMax = fromB + adaptedTolerance;

	const float addR = fromR + toR;
	const float addG = fromG + toG;
	const float addB = fromB + toB;


	for (j = 0; j < height; j++)
	{
		const float* __restrict pSrcLine = pSrc + j * linePitch;
		float* __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i << 2;

			A = pSrcLine[idx    ];
			R = pSrcLine[idx + 1];
			G = pSrcLine[idx + 2];
			B = pSrcLine[idx + 3];

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

			pDstLine[idx    ] = A;
			pDstLine[idx + 1] = newR;
			pDstLine[idx + 2] = newG;
			pDstLine[idx + 3] = newB;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}
