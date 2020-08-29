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
		float* __restrict pDst = pDstBuf + width * j;

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


void ImageGradientVertical_BGRA_4444_16u
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
	const csSDK_uint32* __restrict firstLineSrc  = pSrcBuf;
	const csSDK_uint32* __restrict secondLineSrc = pSrcBuf + linePitch;
	const float* __restrict Rgb2Yuv = (true == isBT709) ? RGB2YUV[1] : RGB2YUV[0];

	csSDK_int32 i = 0, j = 0;
	csSDK_int32 idx = 0;

	/* compute image gradient on first line */
	//#pragma unroll(8)
	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		idx = i * 2;
		const csSDK_uint32& fLinePixelL{ firstLineSrc [idx    ] };
		const csSDK_uint32& fLinePixelH{ firstLineSrc [idx + 1] };
		const csSDK_uint32& sLinePixelL{ secondLineSrc[idx    ] };
		const csSDK_uint32& sLinePixelH{ secondLineSrc[idx + 1] };

		pDstBuf[i] = ( static_cast<float>( sLinePixelH & 0x0000FFFFu)         * Rgb2Yuv[0] +
			           static_cast<float>((sLinePixelL & 0xFFFF0000u) >> 16)  * Rgb2Yuv[1] +
			           static_cast<float>( sLinePixelL & 0x0000FFFFu)         * Rgb2Yuv[2] ) -
			         ( static_cast<float>( fLinePixelH & 0x0000FFFFu)         * Rgb2Yuv[0] +
				       static_cast<float>((fLinePixelL & 0xFFFF0000u) >> 16)  * Rgb2Yuv[1] +
	        		   static_cast<float>( fLinePixelL & 0x0000FFFFu)         * Rgb2Yuv[2] );
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
			idx = i * 2;
			const csSDK_uint32& fLinePixelL{ firstLineSrc [idx    ] };
			const csSDK_uint32& fLinePixelH{ firstLineSrc [idx + 1] };
			const csSDK_uint32& sLinePixelL{ secondLineSrc[idx    ] };
			const csSDK_uint32& sLinePixelH{ secondLineSrc[idx + 1] };

			pDstBuf[i] = ( static_cast<float>( sLinePixelH & 0x0000FFFFu)         * Rgb2Yuv[0] +
				           static_cast<float>((sLinePixelL & 0xFFFF0000u) >> 16)  * Rgb2Yuv[1] +
				           static_cast<float>( sLinePixelL & 0x0000FFFFu)         * Rgb2Yuv[2] ) -
				         ( static_cast<float>( fLinePixelH & 0x0000FFFFu)         * Rgb2Yuv[0] +
					       static_cast<float>((fLinePixelL & 0xFFFF0000u) >> 16)  * Rgb2Yuv[1] +
					       static_cast<float>( fLinePixelL & 0x0000FFFFu)         * Rgb2Yuv[2]);
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
		idx = i * 2;
		const csSDK_uint32& pLinePixelL{ prevLineSrc[idx    ] };
		const csSDK_uint32& pLinePixelH{ prevLineSrc[idx + 1] };
		const csSDK_uint32& lLinePixelL{ lastLineSrc[idx    ] };
		const csSDK_uint32& lLinePixelH{ lastLineSrc[idx + 1] };

		pDstBuf[i] = ( static_cast<float>( lLinePixelH & 0x0000FFFFu)         * Rgb2Yuv[0] +
			           static_cast<float>((lLinePixelL & 0xFFFF0000u) >> 16)  * Rgb2Yuv[1] +
			           static_cast<float>( lLinePixelL & 0x0000FFFFu)         * Rgb2Yuv[2] ) -
			         ( static_cast<float>( pLinePixelH & 0x0000FFFFu)         * Rgb2Yuv[0] +
			       	   static_cast<float>((pLinePixelL & 0xFFFF0000u) >> 16)  * Rgb2Yuv[1] +
				       static_cast<float>( pLinePixelL & 0x0000FFFFu)         * Rgb2Yuv[2] );
	}
	return;
}

void ImageGradientHorizontal_BGRA_4444_16u
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
	csSDK_int32 idx = 0;

	__VECTOR_ALIGNED__
		for (j = 0; j < height; j++)
		{
			const csSDK_uint32* __restrict pSrc = pSrcBuf + linePitch * j;
			float* __restrict pDst = pDstBuf + width * j;

			idx = 0;
			const csSDK_uint32& pixFirstL { pSrc[idx    ] };
			const csSDK_uint32& pixFirstH { pSrc[idx + 1] };
			idx = 2;
			const csSDK_uint32& pixSecondL{ pSrc[idx    ] };
			const csSDK_uint32& pixSecondH{ pSrc[idx + 1] };

			/* first row */
			pDst[0] = ( static_cast<float>( pixSecondH & 0x0000FFFFu)           * Rgb2Yuv[0] +
				        static_cast<float>((pixSecondL & 0xFFFF0000u) >> 16)    * Rgb2Yuv[1] +
				        static_cast<float>( pixSecondL & 0x0000FFFFu)           * Rgb2Yuv[2] ) -
				      ( static_cast<float>( pixFirstH  & 0x0000FFFFu)           * Rgb2Yuv[0] +
					    static_cast<float>((pixFirstL  & 0xFFFF0000u) >> 16)    * Rgb2Yuv[1] +
					    static_cast<float>( pixFirstL  & 0x0000FFFFu)           * Rgb2Yuv[2] );

			/* next row's */
			for (i = 1; i < lastIdx; i++)
			{
				idx = 2 * i;
				const csSDK_uint32& pixPrevL{ pSrc[idx - 2] };
				const csSDK_uint32& pixPrevH{ pSrc[idx - 1] };
				const csSDK_uint32& pixNextL{ pSrc[idx + 2] };
				const csSDK_uint32& pixNextH{ pSrc[idx + 3] };

				pDst[i] = ( static_cast<float>( pixNextH & 0x0000FFFFu)        * Rgb2Yuv[0] +
					        static_cast<float>((pixNextL & 0xFFFF0000u) >> 16) * Rgb2Yuv[1] +
					        static_cast<float>( pixNextL & 0x0000FFFFu)        * Rgb2Yuv[2] ) -
					      ( static_cast<float>( pixPrevH & 0x0000FFFFu)        * Rgb2Yuv[0] +
						    static_cast<float>((pixPrevL & 0xFFFF0000u) >> 16) * Rgb2Yuv[1] +
						    static_cast<float>( pixPrevL & 0x0000FFFFu)        * Rgb2Yuv[2] );
			}

			/* last row */
			const csSDK_uint32& pixPrevL{ pSrc[idx - 2] };
			const csSDK_uint32& pixPrevH{ pSrc[idx - 1] };
			const csSDK_uint32& pixNextL{ pSrc[idx    ] };
			const csSDK_uint32& pixNextH{ pSrc[idx + 1] };

			pDst[i] = ( static_cast<float>( pixNextH & 0x0000FFFFu)        * Rgb2Yuv[0] +
				        static_cast<float>((pixNextL & 0xFFFF0000u) >> 16) * Rgb2Yuv[1] +
				        static_cast<float>( pixNextL & 0x0000FFFFu)        * Rgb2Yuv[2] ) -
				      ( static_cast<float>( pixPrevH & 0x0000FFFFu)        * Rgb2Yuv[0] +
					    static_cast<float>((pixPrevL & 0xFFFF0000u) >> 16) * Rgb2Yuv[1] +
					    static_cast<float>( pixPrevL & 0x0000FFFFu)        * Rgb2Yuv[2] );
		}

	return;
}


void ImageGradientVertical_BGRA_4444_32f
(
	const float* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch,
	const bool&         isBT709
)
{
	float*   __restrict pDst = nullptr;
	const float* __restrict firstLineSrc  = pSrcBuf;
	const float* __restrict secondLineSrc = pSrcBuf + linePitch;
	const float* __restrict Rgb2Yuv = (true == isBT709) ? RGB2YUV[1] : RGB2YUV[0];

	csSDK_int32 i = 0, j = 0;
	csSDK_int32 idx = 0;

	/* compute image gradient on first line */
	//#pragma unroll(4)
	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		idx = i * 4;
		const float& fLinePixelB{ firstLineSrc [idx    ] };
		const float& fLinePixelG{ firstLineSrc [idx + 1] };
		const float& fLinePixelR{ firstLineSrc [idx + 2] };
		const float& sLinePixelB{ secondLineSrc[idx    ] };
		const float& sLinePixelG{ secondLineSrc[idx + 1] };
		const float& sLinePixelR{ secondLineSrc[idx + 2] };

		pDstBuf[i] = ( sLinePixelR * Rgb2Yuv[0] + sLinePixelG * Rgb2Yuv[1] + sLinePixelB * Rgb2Yuv[2] ) -
			         ( fLinePixelR * Rgb2Yuv[0] + fLinePixelG * Rgb2Yuv[1] + fLinePixelB * Rgb2Yuv[2] );
	}

	/* compute image gradient for next lines */
	const csSDK_int32 lastLine = height - 1;

	for (j = 1; j < lastLine; j++)
	{
		const float* __restrict prevLineSrc = pSrcBuf + linePitch * (j - 1);
		const float* __restrict nextLineSrc = pSrcBuf + linePitch * (j + 1);
		pDst = pDstBuf + width * j;

		//#pragma unroll(4)
		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			idx = i * 4;
			const float& fLinePixelB{ prevLineSrc[idx    ] };
			const float& fLinePixelG{ prevLineSrc[idx + 1] };
			const float& fLinePixelR{ prevLineSrc[idx + 2] };
			const float& sLinePixelB{ nextLineSrc[idx    ] };
			const float& sLinePixelG{ nextLineSrc[idx + 1] };
			const float& sLinePixelR{ nextLineSrc[idx + 2] };

			pDstBuf[i] = ( sLinePixelR * Rgb2Yuv[0] + sLinePixelG * Rgb2Yuv[1] + sLinePixelB * Rgb2Yuv[2] ) -
				         ( fLinePixelR * Rgb2Yuv[0] + fLinePixelG * Rgb2Yuv[1] + fLinePixelB * Rgb2Yuv[2] );
		}
	}

	/* compute image gradient for last line */
	const float* __restrict prevLineSrc = pSrcBuf + linePitch * (lastLine - 1);
	const float* __restrict lastLineSrc = pSrcBuf + linePitch * lastLine;
	pDst = pDstBuf + width * (height - 1);

	//#pragma unroll(4)
	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		idx = i * 4;
		const float& fLinePixelB{ prevLineSrc[idx    ] };
		const float& fLinePixelG{ prevLineSrc[idx + 1] };
		const float& fLinePixelR{ prevLineSrc[idx + 2] };
		const float& sLinePixelB{ lastLineSrc[idx    ] };
		const float& sLinePixelG{ lastLineSrc[idx + 1] };
		const float& sLinePixelR{ lastLineSrc[idx + 2] };

		pDstBuf[i] = ( sLinePixelR * Rgb2Yuv[0] + sLinePixelG * Rgb2Yuv[1] + sLinePixelB * Rgb2Yuv[2] ) -
	         		 ( fLinePixelR * Rgb2Yuv[0] + fLinePixelG * Rgb2Yuv[1] + fLinePixelB * Rgb2Yuv[2] );
	}
	return;
}

void ImageGradientHorizontal_BGRA_4444_32f
(
	const float* __restrict pSrcBuf,
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
	csSDK_int32 idx = 0;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		const float* __restrict pSrc = pSrcBuf + linePitch * j;
		float* __restrict pDst = pDstBuf + width * j;

		idx = 0;
		const float& pixFirstB{ pSrc[idx    ] };
		const float& pixFirstG{ pSrc[idx + 1] };
		const float& pixFirstR{ pSrc[idx + 2] };

		idx = 4;
		const float& pixSecondR{ pSrc[idx    ] };
		const float& pixSecondG{ pSrc[idx + 1] };
		const float& pixSecondB{ pSrc[idx + 2] };

		/* first row */
		pDst[0] = ( pixSecondR * Rgb2Yuv[0] + pixSecondG * Rgb2Yuv[1] + pixSecondB * Rgb2Yuv[2] ) -
				  ( pixFirstR  * Rgb2Yuv[0] + pixFirstG  *Rgb2Yuv[1]  + pixFirstB  * Rgb2Yuv[2] );

		/* next row's */
		for (i = 1; i < lastIdx; i++)
		{
			idx = i * 4;
			const float& pixPrevB{ pSrc[idx - 4] };
			const float& pixPrevG{ pSrc[idx - 3] };
			const float& pixPrevR{ pSrc[idx - 2] };

			const float& pixNextB{ pSrc[idx + 4] };
			const float& pixNextG{ pSrc[idx + 5] };
			const float& pixNextR{ pSrc[idx + 6] };

			pDst[i] = (pixNextR * Rgb2Yuv[0] + pixNextG * Rgb2Yuv[1] + pixNextB * Rgb2Yuv[2] ) -
					  (pixPrevR * Rgb2Yuv[0] + pixPrevG * Rgb2Yuv[1] + pixPrevB * Rgb2Yuv[2] );
		}

		/* last row */
		const float& pixPrevB{ pSrc[idx - 4] };
		const float& pixPrevG{ pSrc[idx - 3] };
		const float& pixPrevR{ pSrc[idx - 2] };
		const float& pixNextB{ pSrc[idx    ] };
		const float& pixNextG{ pSrc[idx + 1] };
		const float& pixNextR{ pSrc[idx + 2] };

		pDst[i] = ( pixNextR * Rgb2Yuv[0] + pixNextG * Rgb2Yuv[1] + pixNextB * Rgb2Yuv[2] ) -
				  ( pixPrevR * Rgb2Yuv[0] + pixPrevG * Rgb2Yuv[1] + pixPrevB * Rgb2Yuv[2] );
	}

	return;
}