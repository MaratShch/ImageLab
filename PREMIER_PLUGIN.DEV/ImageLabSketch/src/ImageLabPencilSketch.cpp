#include "ImageLabSketch.h"

void ImageMakePencilSketch_BGRA_4444_8u
(
	const csSDK_uint32* __restrict pSrc,
	const float* __restrict  vGradient,
	const float* __restrict  hGradient,
	csSDK_uint32* __restrict pDst,
	const csSDK_int32&       width,
	const csSDK_int32&       height,
	const csSDK_int32&       linePitch,
	const csSDK_int32&       enhancement
)
{
	csSDK_int32 i = 0, j = 0;
	csSDK_int32 negVal = 0;
	float sqrtVal = 0.f;
	const float multiplyer = static_cast<float>(enhancement) / static_cast<float>(enhancementDivider);

	for (j = 0; j < height; j++)
	{
		const float*        __restrict pSrc1Line = vGradient + j * width;
		const float*        __restrict pSrc2Line = hGradient + j * width;
		const csSDK_uint32* __restrict pSrcLine = pSrc + j * linePitch;
		csSDK_uint32*       __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float& v1{ pSrc1Line[i] };
			const float& v2{ pSrc2Line[i] };

			sqrtVal = MIN (255.0f, multiplyer * sqrt(v1 * v1 + v2 * v2));
			negVal = static_cast<csSDK_int32>(255.0f - sqrtVal);
			pDstLine[i] = negVal | (negVal << 8) | (negVal << 16) | (pSrcLine[i] & 0xFF000000u);
		}
	}
	return;
}


void ImageMakePencilSketch_BGRA_4444_16u
(
	const csSDK_uint32* __restrict pSrc,
	const float*        __restrict  vGradient,
	const float*        __restrict  hGradient,
	csSDK_uint32*       __restrict pDst,
	const csSDK_int32&       width,
	const csSDK_int32&       height,
	const csSDK_int32&       linePitch,
	const csSDK_int32&       enhancement
)
{
	csSDK_int32 i = 0, j = 0, idx = 0;
	csSDK_int32 negVal = 0;
	float sqrtVal = 0.f;
	const float multiplyer = static_cast<float>(enhancement) / static_cast<float>(enhancementDivider);

	for (j = 0; j < height; j++)
	{
		const float*        __restrict pSrc1Line = vGradient + j * width;
		const float*        __restrict pSrc2Line = hGradient + j * width;
		const csSDK_uint32* __restrict pSrcLine = pSrc + j * linePitch;
		csSDK_uint32*       __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float& v1{ pSrc1Line[i] };
			const float& v2{ pSrc2Line[i] };

			idx = i * 2;
			sqrtVal = MIN(32767.0f, multiplyer * sqrt(v1 * v1 + v2 * v2));
			negVal = static_cast<csSDK_int32>(32767.0f - sqrtVal);
			pDstLine[idx    ] = negVal | (negVal << 16);
			pDstLine[idx + 1] = negVal | (pSrcLine[idx + 1] & 0xFFFF0000u);
		}
	}
	return;
}


void ImageMakePencilSketch_BGRA_4444_32f
(
	const float* __restrict pSrc,
	const float* __restrict vGradient,
	const float* __restrict hGradient,
	float*       __restrict pDst,
	const csSDK_int32&      width,
	const csSDK_int32&      height,
	const csSDK_int32&      linePitch,
	const csSDK_int32&      enhancement
)
{
	csSDK_int32 i = 0, j = 0, idx = 0;
	float negVal = 0.f;
	float sqrtVal = 0.f;
	const float multiplyer = (static_cast<float>(enhancement) / static_cast<float>(enhancementDivider));
	constexpr float maxFloat = 1.0f - FLT_EPSILON;

	for (j = 0; j < height; j++)
	{
		const float* __restrict pSrc1Line = vGradient + j * width;
		const float* __restrict pSrc2Line = hGradient + j * width;
		const float* __restrict pSrcLine  = pSrc + j * linePitch;
		      float* __restrict pDstLine  = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float& v1{ pSrc1Line[i] };
			const float& v2{ pSrc2Line[i] };

			idx = i * 4;
			sqrtVal = MIN(maxFloat, multiplyer * sqrt(v1 * v1 + v2 * v2));
			negVal = maxFloat - sqrtVal;
			pDstLine[idx    ] = pDstLine[idx + 1] = pDstLine[idx + 2] = negVal;
			pDstLine[idx + 3] = pSrcLine[idx + 3];
		}
	}
}


void ImageMakePencilSketch_VUYA_4444_8u
(
	const csSDK_uint32* __restrict pSrc,
	const float* __restrict  vGradient,
	const float* __restrict  hGradient,
	csSDK_uint32* __restrict pDst,
	const csSDK_int32&       width,
	const csSDK_int32&       height,
	const csSDK_int32&       linePitch,
	const csSDK_int32&       enhancement
)
{
	csSDK_int32 i = 0, j = 0;
	csSDK_uint32 negVal = 0;
	float sqrtVal = 0.f;
	const float multiplyer = static_cast<float>(enhancement) / static_cast<float>(enhancementDivider);

	for (j = 0; j < height; j++)
	{
		const float*        __restrict pSrc1Line = vGradient + j * width;
		const float*        __restrict pSrc2Line = hGradient + j * width;
		const csSDK_uint32* __restrict pSrcLine = pSrc + j * linePitch;
		csSDK_uint32*       __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float& v1{ pSrc1Line[i] };
			const float& v2{ pSrc2Line[i] };

			sqrtVal = MIN(255.0f, multiplyer * sqrt(v1 * v1 + v2 * v2));
			negVal = static_cast<csSDK_uint32>(255.0f - sqrtVal);
			pDstLine[i] = 0x00008080u | (negVal << 16) | (pSrcLine[i] & 0xFF000000u);
		}
	}
	return;
}

void ImageMakePencilSketch_VUYA_4444_32f
(
	const float* __restrict pSrc,
	const float* __restrict  vGradient,
	const float* __restrict  hGradient,
	float*       __restrict pDst,
	const csSDK_int32&       width,
	const csSDK_int32&       height,
	const csSDK_int32&       linePitch,
	const csSDK_int32&       enhancement
)
{
	csSDK_int32 i = 0, j = 0, idx = 0;
	float negVal = 0.f;
	float sqrtVal = 0.f;
	const float multiplyer = static_cast<float>(enhancement) / static_cast<float>(enhancementDivider);
	constexpr float maxFloat = 1.0f - FLT_EPSILON;

	for (j = 0; j < height; j++)
	{
		const float* __restrict pSrc1Line = vGradient + j * width;
		const float* __restrict pSrc2Line = hGradient + j * width;
		const float* __restrict pSrcLine = pSrc + j * linePitch;
		float*       __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				idx = i * 4;
				const float& v1{ pSrc1Line[i] };
				const float& v2{ pSrc2Line[i] };

				sqrtVal = MIN(maxFloat, multiplyer * sqrt(v1 * v1 + v2 * v2));
				negVal = maxFloat - sqrtVal;
				pDstLine[idx    ] = pDstLine[idx + 1] = 0.f;
				pDstLine[idx + 2] = negVal;
				pDstLine[idx + 3] = pSrcLine[idx + 3];
			}
	}
	return;
}


void ImageMakePencilSketch_ARGB_4444_8u
(
	const csSDK_uint32* __restrict pSrc,
	const float* __restrict  vGradient,
	const float* __restrict  hGradient,
	csSDK_uint32* __restrict pDst,
	const csSDK_int32&       width,
	const csSDK_int32&       height,
	const csSDK_int32&       linePitch,
	const csSDK_int32&       enhancement
)
{
	csSDK_int32 i = 0, j = 0;
	csSDK_uint32 negVal = 0u;
	float sqrtVal = 0.f;
	const float multiplyer = static_cast<float>(enhancement) / static_cast<float>(enhancementDivider);

	for (j = 0; j < height; j++)
	{
		const float*        __restrict pSrc1Line = vGradient + j * width;
		const float*        __restrict pSrc2Line = hGradient + j * width;
		const csSDK_uint32* __restrict pSrcLine = pSrc + j * linePitch;
		csSDK_uint32*       __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float& v1{ pSrc1Line[i] };
			const float& v2{ pSrc2Line[i] };

			sqrtVal = MIN(255.0f, multiplyer * sqrt(v1 * v1 + v2 * v2));
			negVal = static_cast<csSDK_uint32>(255.0f - sqrtVal);
			pDstLine[i] = (pSrcLine[i] & 0x000000FFu) | (negVal << 8) | (negVal << 16) | (negVal << 24);
		}
	}
	return;
}


void ImageMakePencilSketch_ARGB_4444_16u
(
	const csSDK_uint32* __restrict pSrc,
	const float*        __restrict  vGradient,
	const float*        __restrict  hGradient,
	csSDK_uint32*       __restrict pDst,
	const csSDK_int32&       width,
	const csSDK_int32&       height,
	const csSDK_int32&       linePitch,
	const csSDK_int32&       enhancement
)
{
	csSDK_int32 i = 0, j = 0, idx = 0;
	csSDK_uint32 negVal = 0;
	float sqrtVal = 0.f;
	const float multiplyer = static_cast<float>(enhancement) / static_cast<float>(enhancementDivider);

	for (j = 0; j < height; j++)
	{
		const float*        __restrict pSrc1Line = vGradient + j * width;
		const float*        __restrict pSrc2Line = hGradient + j * width;
		const csSDK_uint32* __restrict pSrcLine = pSrc + j * linePitch;
		csSDK_uint32*       __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				const float& v1{ pSrc1Line[i] };
				const float& v2{ pSrc2Line[i] };

				idx = i * 2;
				sqrtVal = MIN(32767.0f, multiplyer * sqrt(v1 * v1 + v2 * v2));
				negVal = static_cast<csSDK_uint32>(32767.0f - sqrtVal);
				pDstLine[idx    ] = (pSrcLine[idx] & 0x0000FFFFu) | (negVal << 16);
				pDstLine[idx + 1] = negVal | (negVal << 16);
			}
	}
	return;
}


void ImageMakePencilSketch_ARGB_4444_32f
(
	const float* __restrict pSrc,
	const float* __restrict vGradient,
	const float* __restrict hGradient,
	float*       __restrict pDst,
	const csSDK_int32&      width,
	const csSDK_int32&      height,
	const csSDK_int32&      linePitch,
	const csSDK_int32&      enhancement
)
{
	csSDK_int32 i = 0, j = 0, idx = 0;
	float negVal = 0.f;
	float sqrtVal = 0.f;
	const float multiplyer = (static_cast<float>(enhancement) / static_cast<float>(enhancementDivider));
	constexpr float maxFloat = 1.0f - FLT_EPSILON;

	for (j = 0; j < height; j++)
	{
		const float* __restrict pSrc1Line = vGradient + j * width;
		const float* __restrict pSrc2Line = hGradient + j * width;
		const float* __restrict pSrcLine = pSrc + j * linePitch;
		float* __restrict pDstLine = pDst + j * linePitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float& v1{ pSrc1Line[i] };
			const float& v2{ pSrc2Line[i] };

			idx = i * 4;
			sqrtVal = MIN(maxFloat, multiplyer * sqrt(v1 * v1 + v2 * v2));
			negVal = maxFloat - sqrtVal;
			pDstLine[idx] = pSrcLine[idx];
			pDstLine[idx + 1] = pDstLine[idx + 2] = pDstLine[idx + 3] = negVal;
		}
	}
}