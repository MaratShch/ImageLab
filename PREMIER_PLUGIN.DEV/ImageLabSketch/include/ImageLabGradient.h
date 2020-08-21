#pragma once

#include "ImageLabSketch.h"

template <typename T>
static inline void ImageGradientVertical_RGB (
	const T* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch,
	const bool&         isBT709
)
{
	float*   __restrict pDst = nullptr;
	const T* __restrict firstLineSrc = pSrcBuf;
	const T* __restrict secondLineSrc = pSrcBuf + linePitch;
	const float* __restrict Rgb2Yuv = (true == isBT709) ? RGB2YUV[1] : RGB2YUV[0];

	csSDK_int32 i, j;

	/* compute image gradient on first line */
//#pragma unroll(8)
	for (i = 0; i < width; i++)
	{
		const T& fLinePixel{ firstLineSrc[i] };
		const T& sLinePixel{ secondLineSrc[i] };

		pDstBuf[i] = ( static_cast<float>(sLinePixel.R) * Rgb2Yuv[0] +
			           static_cast<float>(sLinePixel.G) * Rgb2Yuv[1] +
			           static_cast<float>(sLinePixel.B) * Rgb2Yuv[2] ) -
			         ( static_cast<float>(fLinePixel.R) * Rgb2Yuv[0] +
				       static_cast<float>(fLinePixel.G) * Rgb2Yuv[1] +
				       static_cast<float>(fLinePixel.B) * Rgb2Yuv[2] );
	}

	/* compute image gradient for next lines */
	const csSDK_int32 lastLine = height - 1;

	for (j = 1; j < lastLine; j++)
	{
		const T* __restrict prevLineSrc = pSrcBuf + linePitch * (j - 1);
		const T* __restrict nextLineSrc = pSrcBuf + linePitch * (j + 1);
		pDst = pDstBuf + width * j;

//#pragma unroll(8)
		for (i = 0; i < width; i++)
		{
			const T& fLinePixel{ prevLineSrc[i] };
			const T& sLinePixel{ nextLineSrc[i] };

			pDst[i] = ( static_cast<float>(sLinePixel.R) * Rgb2Yuv[0] +
				        static_cast<float>(sLinePixel.G) * Rgb2Yuv[1] +
				        static_cast<float>(sLinePixel.B) * Rgb2Yuv[2] ) -
				      ( static_cast<float>(fLinePixel.R) * Rgb2Yuv[0] +
					    static_cast<float>(fLinePixel.G) * Rgb2Yuv[1] +
					    static_cast<float>(fLinePixel.B) * Rgb2Yuv[2] );
		}
	}

	/* compute image gradient for last line */
	const T* __restrict prevLineSrc = pSrcBuf + linePitch * (lastLine - 1);
	const T* __restrict lastLineSrc = pSrcBuf + linePitch * lastLine;
	pDst = pDstBuf + width * (height - 1);

//#pragma unroll(8)
	for (i = 0; i < width; i++)
	{
		const T& pLinePixel{ prevLineSrc[i] };
		const T& lLinePixel{ lastLineSrc[i] };

		pDst[i] = ( static_cast<float>(lLinePixel.R) * Rgb2Yuv[0] +
			        static_cast<float>(lLinePixel.G) * Rgb2Yuv[1] +
			        static_cast<float>(lLinePixel.B) * Rgb2Yuv[2] ) -
			      ( static_cast<float>(pLinePixel.R) * Rgb2Yuv[0] +
				    static_cast<float>(pLinePixel.G) * Rgb2Yuv[1] +
				    static_cast<float>(pLinePixel.B) * Rgb2Yuv[2] );
	}
	return;
}

template <typename T>
static inline void ImageGradientHorizontal_RGB (
	const T* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch,
	const bool&         isBT709
)
{
	const float* __restrict Rgb2Yuv = (true == isBT709) ? RGB2YUV[1] : RGB2YUV[0];
	const csSDK_int32 lastIdx = width - 1;

	csSDK_int32 i, j;

	for (j = 0; j < height; j++)
	{
		const T* __restrict pSrc = pSrcBuf + linePitch * j;
		  float* __restrict pDst = pDstBuf + linePitch * j;

		const T& pixFirst { pSrc[i] };
		const T& pixSecond{ pSrc[i + 1] };

		/* first row */
		pDst[0] = ( static_cast<float>(pixSecond.R) * Rgb2Yuv[0] +
			        static_cast<float>(pixSecond.G) * Rgb2Yuv[1] +
			        static_cast<float>(pixSecond.B) * Rgb2Yuv[2] ) -
			      ( static_cast<float>(pixFirst.R)  * Rgb2Yuv[0] +
				    static_cast<float>(pixFirst.G)  * Rgb2Yuv[1] +
				    static_cast<float>(pixFirst.B)  * Rgb2Yuv[2] );

		/* next row's */
		for (i = 1; i < lastIdx; i++)
		{
			const T& pixPrev { pSrc[i - 1] };
			const T& pixNext { pSrc[i + 1] };

			pDst[i] = ( static_cast<float>(pixNext.R) * Rgb2Yuv[0] +
				        static_cast<float>(pixNext.G) * Rgb2Yuv[1] +
				        static_cast<float>(pixNext.B) * Rgb2Yuv[2] ) -
				      ( static_cast<float>(pixPrev.R) * Rgb2Yuv[0] +
					    static_cast<float>(pixPrev.G) * Rgb2Yuv[1] +
					    static_cast<float>(pixPrev.B) * Rgb2Yuv[2] );
		}

		/* last row */
		const T& pixPrev{ pSrc[i - 1] };
		const T& pixNext{ pSrc[i] };

		pDst[i] = ( static_cast<float>(pixNext.R) * Rgb2Yuv[0] +
			        static_cast<float>(pixNext.G) * Rgb2Yuv[1] +
			        static_cast<float>(pixNext.B) * Rgb2Yuv[2] ) -
			      ( static_cast<float>(pixPrev.R) * Rgb2Yuv[0] +
				    static_cast<float>(pixPrev.G) * Rgb2Yuv[1] +
				    static_cast<float>(pixPrev.B) * Rgb2Yuv[2]);
	}

	return;
}


template <typename T>
static inline void ImageGradientVertical_YUV (
	const T* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch
)
{
	float*   __restrict pDst = nullptr;
	const T* __restrict firstLineSrc = pSrcBuf;
	const T* __restrict secondLineSrc = pSrcBuf + linePitch;

	csSDK_int32 i, j;

	/* compute image gradient on first line */
	//#pragma unroll(8)
	for (i = 0; i < width; i++)
	{
		const T& fLinePixel{ firstLineSrc[i] };
		const T& sLinePixel{ secondLineSrc[i] };

		pDstBuf[i] = static_cast<float>(sLinePixel.Y) -
			         static_cast<float>(fLinePixel.Y);
	}

	/* compute image gradient for next lines */
	const csSDK_int32 lastLine = height - 1;

	for (j = 1; j < lastLine; j++)
	{
		const T* __restrict prevLineSrc = pSrcBuf + linePitch * (j - 1);
		const T* __restrict nextLineSrc = pSrcBuf + linePitch * (j + 1);
		pDst = pDstBuf + width * j;

		//#pragma unroll(8)
		for (i = 0; i < width; i++)
		{
			const T& fLinePixel{ prevLineSrc[i] };
			const T& sLinePixel{ nextLineSrc[i] };

			pDst[i] = static_cast<float>(sLinePixel.Y) -
				      static_cast<float>(fLinePixel.Y);
		}
	}

	/* compute image gradient for last line */
	const T* __restrict prevLineSrc = pSrcBuf + linePitch * (lastLine - 1);
	const T* __restrict lastLineSrc = pSrcBuf + linePitch * lastLine;
	pDst = pDstBuf + width * (height - 1);

	//#pragma unroll(8)
	for (i = 0; i < width; i++)
	{
		const T& pLinePixel{ prevLineSrc[i] };
		const T& lLinePixel{ lastLineSrc[i] };

		pDst[i] = static_cast<float>(lLinePixel.Y) -
			      static_cast<float>(pLinePixel.Y);
	}
	return;
}


template <typename T>
static inline void ImageGradientHorizontal_YUV(
	const T* __restrict pSrcBuf,
	float*   __restrict pDstBuf,
	const csSDK_int32&  width,
	const csSDK_int32&  height,
	const csSDK_int32&  linePitch
)
{
	const csSDK_int32 lastIdx = width - 1;
	csSDK_int32 i, j;

	for (j = 0; j < height; j++)
	{
		const T* __restrict pSrc = pSrcBuf + linePitch * j;
		  float* __restrict pDst = pDstBuf + linePitch * j;

		const T& pixFirst { pSrc[i] };
		const T& pixSecond{ pSrc[i + 1] };

		/* first row */
		pDst[0] = ( static_cast<float>(pixSecond.Y) -
			        static_cast<float>(pixFirst.Y) );

		/* next row's */
		for (i = 1; i < lastIdx; i++)
		{
			const T& pixPrev{ pSrc[i - 1] };
			const T& pixNext{ pSrc[i + 1] };

			pDst[i] = ( static_cast<float>(pixNext.Y) -
				        static_cast<float>(pixPrev.Y) );
		}

		/* last row */
		const T& pixPrev{ pSrc[i - 1] };
		const T& pixNext{ pSrc[i] };

		pDst[i] = ( static_cast<float>(pixNext.Y) -
			        static_cast<float>(pixPrev.Y) );
	}

	return;
}
