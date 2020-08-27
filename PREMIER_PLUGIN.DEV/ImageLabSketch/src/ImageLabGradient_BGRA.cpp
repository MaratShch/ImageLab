#include "ImageLabSketch.h"

void ImageGradientVertical_BGRA_4444_8u
(
	const csSDK_uint32* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch,
	const bool&         isBT709
)
{
	float*   __restrict pDst = nullptr;
	const csSDK_uint32* __restrict firstLineSrc = pSrcBuf;
	const csSDK_uint32* __restrict secondLineSrc = pSrcBuf + linePitch;
	const float* __restrict Rgb2Yuv = (true == isBT709) ? RGB2YUV[1] : RGB2YUV[0];

	csSDK_int32 i = 0, j = 0;

	/* compute image gradient on first line */
	//#pragma unroll(8)
	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		const csSDK_uint32& fLinePixel{ firstLineSrc[i] };
		const csSDK_uint32& sLinePixel{ secondLineSrc[i] };

		pDstBuf[i] = ( static_cast<float>((sLinePixel & 0x00FF0000u) > 16) * Rgb2Yuv[0] +
			           static_cast<float>((sLinePixel & 0x0000FF00u) > 8 ) * Rgb2Yuv[1] +
			           static_cast<float>( sLinePixel & 0x000000FFu)       * Rgb2Yuv[2] ) -
			         ( static_cast<float>((fLinePixel & 0x00FF0000U) > 16) * Rgb2Yuv[0] +
				       static_cast<float>((fLinePixel & 0x0000FF00U) > 8)  * Rgb2Yuv[1] +
				       static_cast<float>( fLinePixel)                     * Rgb2Yuv[2]);
	}

	/* compute image gradient for next lines */
	const csSDK_int32 lastLine = height - 1;
	
	for (j = 1; j < lastLine; j++)
	{
		const csSDK_uint32* __restrict prevLineSrc = pSrcBuf + linePitch * (j - 1);
		const csSDK_uint32* __restrict nextLineSrc = pSrcBuf + linePitch * (j + 1);
		pDst = pDstBuf + width * j;

		//#pragma unroll(8)
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const csSDK_uint32& fLinePixel{ prevLineSrc[i] };
			const csSDK_uint32& sLinePixel{ nextLineSrc[i] };

			pDstBuf[i] = ( static_cast<float>((sLinePixel & 0x00FF0000u) > 16) * Rgb2Yuv[0] +
				           static_cast<float>((sLinePixel & 0x0000FF00u) > 8)  * Rgb2Yuv[1] +
				           static_cast<float>( sLinePixel & 0x000000FFu)       * Rgb2Yuv[2] ) -
				         ( static_cast<float>((fLinePixel & 0x00FF0000U) > 16) * Rgb2Yuv[0] +
					       static_cast<float>((fLinePixel & 0x0000FF00U) > 8)  * Rgb2Yuv[1] +
					       static_cast<float>( fLinePixel)                     * Rgb2Yuv[2] );
		}
	}

	/* compute image gradient for last line */
	const csSDK_uint32* __restrict prevLineSrc = pSrcBuf + linePitch * (lastLine - 1);
	const csSDK_uint32* __restrict lastLineSrc = pSrcBuf + linePitch * lastLine;
	pDst = pDstBuf + width * (height - 1);

	//#pragma unroll(8)
	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		const csSDK_uint32& pLinePixel{ prevLineSrc[i] };
		const csSDK_uint32& lLinePixel{ lastLineSrc[i] };

		pDstBuf[i] = ( static_cast<float>((lLinePixel & 0x00FF0000u) > 16) * Rgb2Yuv[0] +
			           static_cast<float>((lLinePixel & 0x0000FF00u) > 8)  * Rgb2Yuv[1] +
			           static_cast<float>( lLinePixel & 0x000000FFu)       * Rgb2Yuv[2] ) -
			         ( static_cast<float>((pLinePixel & 0x00FF0000U) > 16) * Rgb2Yuv[0] +
				       static_cast<float>((pLinePixel & 0x0000FF00U) > 8)  * Rgb2Yuv[1] +
				       static_cast<float>(pLinePixel)                      * Rgb2Yuv[2] );
	}
	return;
}

void ImageGradientHorizontal_BGRA_4444_8u
(
	const csSDK_uint32* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch,
	const bool&         isBT709
)
{
	const float* __restrict Rgb2Yuv = (true == isBT709) ? RGB2YUV[1] : RGB2YUV[0];
	const csSDK_int32 lastIdx = width - 1;

	csSDK_int32 i = 0, j = 0;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		const csSDK_uint32* __restrict pSrc = pSrcBuf + linePitch * j;
		float* __restrict pDst = pDstBuf + linePitch * j;

		const csSDK_uint32& pixFirst{ pSrc[0] };
		const csSDK_uint32& pixSecond{ pSrc[1] };

		/* first row */
		pDst[0] = ( static_cast<float>((pixSecond & 0x00FF0000U) >> 16) * Rgb2Yuv[0] +
			        static_cast<float>((pixSecond & 0x0000FF00U) >> 8)  * Rgb2Yuv[1] +
			        static_cast<float>( pixSecond & 0x000000FFu)        * Rgb2Yuv[2] ) -
			      ( static_cast<float>((pixFirst  & 0x00FF0000U) >> 16) * Rgb2Yuv[0] +
				    static_cast<float>((pixFirst  & 0x0000FF00U) >> 8)  * Rgb2Yuv[1] +
				    static_cast<float>( pixFirst  & 0x000000FFU)        * Rgb2Yuv[2] );

		/* next row's */
		for (i = 1; i < lastIdx; i++)
		{
			const csSDK_uint32& pixPrev{ pSrc[i - 1] };
			const csSDK_uint32& pixNext{ pSrc[i + 1] };

			pDst[i] = ( static_cast<float>((pixNext & 0x00FF0000U) >> 16) * Rgb2Yuv[0] +
				        static_cast<float>((pixNext & 0x0000FF00U) >> 8)  * Rgb2Yuv[1] +
				        static_cast<float>( pixNext & 0x000000FFu)        * Rgb2Yuv[2] ) -
				      ( static_cast<float>((pixPrev & 0x00FF0000U) >> 16) * Rgb2Yuv[0] +
					    static_cast<float>((pixPrev & 0x0000FF00U) >> 8)  * Rgb2Yuv[1] +
					    static_cast<float>( pixPrev & 0x000000FFU)        * Rgb2Yuv[2] );
		}

		/* last row */
		const csSDK_uint32& pixPrev{ pSrc[i - 1] };
		const csSDK_uint32& pixNext{ pSrc[i] };

		pDst[i] = ( static_cast<float>((pixNext & 0x00FF0000U) >> 16) * Rgb2Yuv[0] +
			        static_cast<float>((pixNext & 0x0000FF00U) >> 8)  * Rgb2Yuv[1] +
			        static_cast<float>( pixNext & 0x000000FFu)        * Rgb2Yuv[2] ) -
			      ( static_cast<float>((pixPrev & 0x00FF0000U) >> 16) * Rgb2Yuv[0] +
				    static_cast<float>((pixPrev & 0x0000FF00U) >> 8)  * Rgb2Yuv[1] +
				    static_cast<float>( pixPrev & 0x000000FFU)        * Rgb2Yuv[2] );
	}

	return;
}