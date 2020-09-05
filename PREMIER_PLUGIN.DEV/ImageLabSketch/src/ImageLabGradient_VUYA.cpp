#include "ImageLabSketch.h"

void ImageGradientVertical_VUYA_4444_8u
(
	const csSDK_uint32* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch
)
{
	float*   __restrict pDst = nullptr;
	const csSDK_uint32* __restrict firstLineSrc = pSrcBuf;
	const csSDK_uint32* __restrict secondLineSrc = pSrcBuf + linePitch;

	csSDK_int32 i = 0, j = 0;


	/* compute image gradient on first line */
	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		pDstBuf[i] = static_cast<float>(((secondLineSrc[i] & 0x00FF0000u) >> 16) - ((firstLineSrc[i] & 0x00FF0000u) >> 16));
	}

	/* compute image gradient for next lines */
	const csSDK_int32 lastLine = height - 1;
	
	for (j = 1; j < lastLine; j++)
	{
		const csSDK_uint32* __restrict prevLineSrc = pSrcBuf + linePitch * (j - 1);
		const csSDK_uint32* __restrict nextLineSrc = pSrcBuf + linePitch * (j + 1);
		pDst = pDstBuf + width * j;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			pDstBuf[i] = static_cast<float>((nextLineSrc[i] & 0x00FF0000u) >> 16) - static_cast<float>((prevLineSrc[i] & 0x00FF0000u) >> 16);
		}
	}

	/* compute image gradient for last line */
	const csSDK_uint32* __restrict prevLineSrc = pSrcBuf + linePitch * (lastLine - 1);
	const csSDK_uint32* __restrict lastLineSrc = pSrcBuf + linePitch * lastLine;
	pDst = pDstBuf + width * (height - 1);

	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		pDstBuf[i] = static_cast<float>((lastLineSrc[i] & 0x00FF0000u) >> 16) - static_cast<float>((prevLineSrc[i] & 0x00FF0000u) >> 16);
	}

	return;
}

void ImageGradientHorizontal_VUYA_4444_8u
(
	const csSDK_uint32* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch
)
{
	const csSDK_int32 lastIdx = width - 1;
	csSDK_int32 i = 0, j = 0;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* __restrict pSrc = pSrcBuf + linePitch * j;
		float* __restrict pDst = pDstBuf + width * j;

		/* first row */
		pDst[0] = static_cast<float>((pSrc[1] & 0x00FF0000u) >> 16) - static_cast<float>((pSrc[0] & 0x00FF0000u) >> 16);

		/* next row's */
		for (i = 1; i < lastIdx; i++)
		{
			pDst[i] = static_cast<float>((pSrc[i + 1] & 0x00FF0000u) >> 16) - static_cast<float>((pSrc[i - 1] & 0x00FF0000u) >> 16);
		}
		
		/* last row */
		pDst[i] = static_cast<float>((pSrc[i] & 0x00FF0000u) >> 16) - static_cast<float>((pSrc[i - 1] & 0x00FF0000u) >> 16);
	}

	return;
}


void ImageGradientVertical_VUYA_4444_16u
(
	const csSDK_uint32* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch
)
{
	float*   __restrict pDst = nullptr;
	const csSDK_uint32* __restrict firstLineSrc  = pSrcBuf;
	const csSDK_uint32* __restrict secondLineSrc = pSrcBuf + linePitch;

	csSDK_int32 i = 0, j = 0;
	csSDK_int32 idx = 0;

	/* compute image gradient on first line */
	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		idx = i * 2;
		pDstBuf[i] = ( static_cast<float>(secondLineSrc[idx + 1] & 0x0000FFFFu) - 
			           static_cast<float>(firstLineSrc [idx + 1] & 0x0000FFFFu) );
	}

	/* compute image gradient for next lines */
	const csSDK_int32 lastLine = height - 1;

	for (j = 1; j < lastLine; j++)
	{
		const csSDK_uint32* __restrict prevLineSrc = pSrcBuf + linePitch * (j - 1);
		const csSDK_uint32* __restrict nextLineSrc = pSrcBuf + linePitch * (j + 1);
		pDst = pDstBuf + width * j;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i * 2;
			pDstBuf[i] = (static_cast<float>(nextLineSrc[idx + 1] & 0x0000FFFFu) -
				          static_cast<float>(prevLineSrc[idx + 1] & 0x0000FFFFu));
		}
	}

	/* compute image gradient for last line */
	const csSDK_uint32* __restrict prevLineSrc = pSrcBuf + linePitch * (lastLine - 1);
	const csSDK_uint32* __restrict lastLineSrc = pSrcBuf + linePitch * lastLine;
	pDst = pDstBuf + width * (height - 1);

	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		idx = i * 2;
		pDstBuf[i] = (static_cast<float>(lastLineSrc[idx + 1] & 0x0000FFFFu) -
			          static_cast<float>(prevLineSrc[idx + 1] & 0x0000FFFFu));
	}
	return;

}

void ImageGradientHorizontal_VUYA_4444_16u
(
	const csSDK_uint32* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch
)
{
	const csSDK_int32 lastIdx = width - 1;

	csSDK_int32 i = 0, j = 0;
	csSDK_int32 idx = 0;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* __restrict pSrc = pSrcBuf + linePitch * j;
		float* __restrict pDst = pDstBuf + width * j;

		/* first row */
		pDst[0] = (static_cast<float>(pSrc[3] & 0x0000FFFFu) - static_cast<float>(pSrc[1] & 0x0000FFFFu));

		/* next row's */
		for (i = 1; i < lastIdx; i++)
		{
			idx = 2 * i;
			pDst[i] = ( static_cast<float>(pSrc[idx + 3] & 0x0000FFFFu) - 
				        static_cast<float>(pSrc[idx - 1] & 0x0000FFFFu) );
		}

		/* last row */
		pDst[i] = ( static_cast<float>(pSrc[idx + 1] & 0x0000FFFFu) -
			        static_cast<float>(pSrc[idx - 1] & 0x0000FFFFu) );
	}

	return;
}


void ImageGradientVertical_VUYA_4444_32f
(
	const float* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch
)
{
	float*   __restrict pDst = nullptr;
	const float* __restrict firstLineSrc  = pSrcBuf;
	const float* __restrict secondLineSrc = pSrcBuf + linePitch;

	csSDK_int32 i = 0, j = 0;
	csSDK_int32 idx = 0;

	/* compute image gradient on first line */
	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		idx = i * 4 + 2;
		pDstBuf[i] = secondLineSrc[idx] - firstLineSrc[idx];
	}

	/* compute image gradient for next lines */
	const csSDK_int32 lastLine = height - 1;

	for (j = 1; j < lastLine; j++)
	{
		const float* __restrict prevLineSrc = pSrcBuf + linePitch * (j - 1);
		const float* __restrict nextLineSrc = pSrcBuf + linePitch * (j + 1);
		pDst = pDstBuf + width * j;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i * 4 + 2;
			pDstBuf[i] = nextLineSrc[idx] - prevLineSrc[idx];
		}
	}

	/* compute image gradient for last line */
	const float* __restrict prevLineSrc = pSrcBuf + linePitch * (lastLine - 1);
	const float* __restrict lastLineSrc = pSrcBuf + linePitch * lastLine;
	pDst = pDstBuf + width * (height - 1);

	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		idx = i * 4 + 2;
		pDstBuf[i] = lastLineSrc[idx] - prevLineSrc[idx];
	}

	return;
}

void ImageGradientHorizontal_VUYA_4444_32f
(
	const float* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch
)
{
	const csSDK_int32 lastIdx = width - 1;

	csSDK_int32 i = 0, j = 0;
	csSDK_int32 idx = 0;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		const float* __restrict pSrc = pSrcBuf + linePitch * j;
		float* __restrict pDst = pDstBuf + width * j;

		/* first row */
		pDst[0] = pSrc[6] - pSrc[2];

		/* next row's */
		for (i = 1; i < lastIdx; i++)
		{
			idx = i * 4;
			pDst[i] = pSrc[idx + 6] - pSrc[idx - 2];
		}

		/* last row */
		pDst[i] = pSrc[idx + 2] - pSrc[idx - 2];
	}

	return;
}