#pragma once

template <typename T>
inline void ImageRGB_GradientVertical
(
	const T*     __restrict pSrcBuffer,
	const float* __restrict pColorTransform,
	float*       __restrict pTmpBuffer,
	const A_long&           height,
	const A_long&           width,
	const A_long&           src_line_pitch
) noexcept
{
	/* process first line in buffer */
	const T* __restrict pFirstLineSrc  = pSrcBuffer;
	const T* __restrict pSecondLineSrc = pSrcBuffer + src_line_pitch;

	A_long i, j;

	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		const T& fLinePixel = pFirstLineSrc[i];
		const T& sLinePixel = pSecondLineSrc[i];

		pTmpBuffer[i] =
			( static_cast<float>(fLinePixel.R) * pColorTransform[0] +
			  static_cast<float>(fLinePixel.G) * pColorTransform[1] +
			  static_cast<float>(fLinePixel.B) * pColorTransform[2] ) -
			( static_cast<float>(sLinePixel.R) * pColorTransform[0] +
              static_cast<float>(sLinePixel.G) * pColorTransform[1] +
              static_cast<float>(sLinePixel.B) * pColorTransform[2] );
	} /* for (i = 0; i < width; i++) */

	/* process rest of lines in buffer */
	const A_long lastLine = height - 1;
	for (j = 1; j < lastLine; j++)
	{
		const T* __restrict pPrevLineSrc = pSrcBuffer + src_line_pitch * (j - 1);
		const T* __restrict pNextLineSrc = pSrcBuffer + src_line_pitch * (j + 1);
		float*   __restrict	pDst = pTmpBuffer + width * j;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const T& pPrevPixel = pPrevLineSrc[i];
			const T& pNextPixel = pNextLineSrc[i];

			pDst[i] =
				( static_cast<float>(pPrevPixel.R) * pColorTransform[0] +
				  static_cast<float>(pPrevPixel.G) * pColorTransform[1] +
				  static_cast<float>(pPrevPixel.B) * pColorTransform[2] ) -
				( static_cast<float>(pNextPixel.R) * pColorTransform[0] +
				  static_cast<float>(pNextPixel.G) * pColorTransform[1] +
				  static_cast<float>(pNextPixel.B) * pColorTransform[2] );
		} /* for (i = 0; i < width; i++) */
	} /* for (j = 1; j < lastLine; j++) */
	
	/* process last line in buffer */
	const T* __restrict prevLineSrc = pSrcBuffer + src_line_pitch * (lastLine - 1);
	const T* __restrict lastLineSrc = pSrcBuffer + src_line_pitch * lastLine;
	float*   __restrict pDst = pTmpBuffer + width * lastLine;

	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		const T& pPrevPixel = prevLineSrc[i];
		const T& pNextPixel = lastLineSrc[i];

		pDst[i] =
			( static_cast<float>(pPrevPixel.R) * pColorTransform[0] +
			  static_cast<float>(pPrevPixel.G) * pColorTransform[1] +
			  static_cast<float>(pPrevPixel.B) * pColorTransform[2] ) -
			( static_cast<float>(pNextPixel.R) * pColorTransform[0] +
			  static_cast<float>(pNextPixel.G) * pColorTransform[1] +
			  static_cast<float>(pNextPixel.B) * pColorTransform[2] );
	} /* for (i = 0; i < width; i++) */

	return;
}


template <typename T>
inline void ImageRGB_GradientHorizontal
(
	const T*     __restrict pSrcBuffer,
	const float* __restrict pColorTransform,
	float*       __restrict pTmpBuffer,
	const A_long&           height,
	const A_long&           width,
	const A_long&           src_line_pitch
) noexcept
{
	A_long i, j;
	const A_long lastIdx = width - 1;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		const T* __restrict pSrc = pSrcBuffer + src_line_pitch * j;
		float*   __restrict pDst = pTmpBuffer + width * j;

		const T& pixFirst = pSrc[0];
		const T& pixSecond = pSrc[1];

		/* first row */
		pDst[0] =
			( static_cast<float>(pixSecond.R) * pColorTransform[0] +
			  static_cast<float>(pixSecond.G) * pColorTransform[1] +
			  static_cast<float>(pixSecond.B) * pColorTransform[2] ) -
			( static_cast<float>(pixFirst.R)  * pColorTransform[0] +
			  static_cast<float>(pixFirst.G)  * pColorTransform[1] +
			  static_cast<float>(pixFirst.B)  * pColorTransform[2] );

		/* next row's */
		for (i = 1; i < lastIdx; i++)
		{
			const T& pixPrev = pSrc[i - 1];
			const T& pixNext = pSrc[i + 1];

			pDst[i] =
				( static_cast<float>(pixNext.R) * pColorTransform[0] +
				  static_cast<float>(pixNext.G) * pColorTransform[1] +
				  static_cast<float>(pixNext.B) * pColorTransform[2] ) -
				( static_cast<float>(pixPrev.R) * pColorTransform[0] +
				  static_cast<float>(pixPrev.G) * pColorTransform[1] +
				  static_cast<float>(pixPrev.B) * pColorTransform[2] );
		} /* for (i = 1; i < lastIdx; i++) */

		/* last row */
		const T& pixPrev = pSrc[i - 1];
		const T& pixNext = pSrc[i    ];

		pDst[i] =
			( static_cast<float>(pixNext.R) * pColorTransform[0] +
			  static_cast<float>(pixNext.G) * pColorTransform[1] +
			  static_cast<float>(pixNext.B) * pColorTransform[2] ) -
			( static_cast<float>(pixPrev.R) * pColorTransform[0] +
			  static_cast<float>(pixPrev.G) * pColorTransform[1] +
			  static_cast<float>(pixPrev.B) * pColorTransform[2] );
	} /* for (j = 0; j < height; j++) */

	return;
}


template <typename T>
inline void ImageYUV_GradientVertical
(
	const T* __restrict pSrcBuffer,
	float*   __restrict pTmpBuffer,
	const A_long&       height,
	const A_long&       width,
	const A_long&       src_line_pitch
) noexcept
{
	/* process first line in buffer */
	const T* __restrict pFirstLineSrc = pSrcBuffer;
	const T* __restrict pSecondLineSrc = pSrcBuffer + src_line_pitch;

	A_long i, j;

	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		const T& fLinePixel = pFirstLineSrc[i];
		const T& sLinePixel = pSecondLineSrc[i];
		pTmpBuffer[i] = static_cast<float>(fLinePixel.Y) - static_cast<float>(sLinePixel.Y);
	} /* for (i = 0; i < width; i++) */

	/* process rest of lines in buffer */
	const A_long lastLine = height - 1;
	for (j = 1; j < lastLine; j++)
	{
		const T* __restrict pPrevLineSrc = pSrcBuffer + src_line_pitch * (j - 1);
		const T* __restrict pNextLineSrc = pSrcBuffer + src_line_pitch * (j + 1);
		float*   __restrict	pDst = pTmpBuffer + width * j;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const T& pPrevPixel = pPrevLineSrc[i];
			const T& pNextPixel = pNextLineSrc[i];
			pDst[i] =  static_cast<float>(pPrevPixel.Y) - static_cast<float>(pNextPixel.Y);
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 1; j < lastLine; j++) */

	  /* process last line in buffer */
	const T* __restrict prevLineSrc = pSrcBuffer + src_line_pitch * (lastLine - 1);
	const T* __restrict lastLineSrc = pSrcBuffer + src_line_pitch * lastLine;
	float*   __restrict pDst = pTmpBuffer + width * lastLine;

	__VECTOR_ALIGNED__
	for (i = 0; i < width; i++)
	{
		const T& pPrevPixel = prevLineSrc[i];
		const T& pNextPixel = lastLineSrc[i];
		pDst[i] = static_cast<float>(pPrevPixel.Y) - static_cast<float>(pNextPixel.Y);
	} /* for (i = 0; i < width; i++) */

	return;
}


template <typename T>
inline void ImageYUV_GradientHorizontal
(
	const T* __restrict pSrcBuffer,
	float*   __restrict pTmpBuffer,
	const A_long&       height,
	const A_long&       width,
	const A_long&       src_line_pitch
) noexcept
{
	A_long i, j;
	const A_long lastIdx = width - 1;

	__VECTOR_ALIGNED__
	for (j = 0; j < height; j++)
	{
		const T* __restrict pSrc = pSrcBuffer + src_line_pitch * j;
		float*   __restrict pDst = pTmpBuffer + width * j;

		const T& pixFirst  = pSrc[0];
		const T& pixSecond = pSrc[1];

		/* first row */
			pDst[0] = static_cast<float>(pixSecond.Y) - static_cast<float>(pixFirst.Y);

			/* next row's */
			for (i = 1; i < lastIdx; i++)
			{
				const T& pixPrev = pSrc[i - 1];
				const T& pixNext = pSrc[i + 1];
				pDst[i] = static_cast<float>(pixNext.Y) - static_cast<float>(pixPrev.Y);
			} /* for (i = 1; i < lastIdx; i++) */

			  /* last row */
			const T& pixPrev = pSrc[i - 1];
			const T& pixNext = pSrc[i];
			pDst[i] = static_cast<float>(pixNext.Y) - static_cast<float>(pixPrev.Y);
	} /* for (j = 0; j < height; j++) */
	
	return;
}


template <typename T>
inline void ImageRGB_ComputeGradient
(
	const T*     __restrict pSrcBuffer,
	const float* __restrict pColorTransform,
	float*       __restrict pTmpBuffer1,
	float*       __restrict pTmpBuffer2,
	const A_long&           height,
	const A_long&           width,
	const A_long&           src_line_pitch
) noexcept
{
	ImageRGB_GradientVertical  (pSrcBuffer, pColorTransform, pTmpBuffer1, height, width, src_line_pitch);
	ImageRGB_GradientHorizontal(pSrcBuffer, pColorTransform, pTmpBuffer2, height, width, src_line_pitch);
	return;
}


template <typename T>
inline void ImageYUV_ComputeGradient
(
	const T*     __restrict pSrcBuffer,
	float*       __restrict pTmpBuffer1,
	float*       __restrict pTmpBuffer2,
	const A_long&           height,
	const A_long&           width,
	const A_long&           src_line_pitch
) noexcept
{
	ImageYUV_GradientVertical  (pSrcBuffer, pTmpBuffer1, height, width, src_line_pitch);
	ImageYUV_GradientHorizontal(pSrcBuffer, pTmpBuffer2, height, width, src_line_pitch);
	return;
}