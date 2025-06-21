#pragma once

#include "FastAriphmetics.hpp"

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


inline void ImageBW_GradientHorizontal
(
    const float*  __restrict pSrcBuffer,
          float*  __restrict pDstBuffer,
    const A_long       width,
    const A_long       height
) noexcept
{
    const A_long lastPixel = width - 1;

    __VECTOR_ALIGNED__
    for (A_long j = 0; j < height; j++)
    {
        const float* __restrict pSrcLine = pSrcBuffer + j * width;
              float* __restrict pDstLine = pDstBuffer + j * width;

        pDstLine[0] = pSrcLine[1] - pSrcLine[0];
        
        for (A_long i = 1; i < lastPixel; i++)
            pDstLine[i] = pSrcLine[i + 1] - pSrcLine[i - 1];

        pDstLine[lastPixel] = pSrcLine[lastPixel] - pSrcLine[lastPixel - 1];

    } // for (A_long j = 0; j < height; j++)

    return;
}


inline void ImageBW_GradientVertical
(
    const float*  __restrict pSrcBuffer,
          float*  __restrict pDstBuffer,
    const A_long       width,
    const A_long       height
) noexcept
{
    A_long j, i;
    const A_long lastLine = height - 1;

    // process first line in frame
    for (i = 0; i < width; i++)
        pDstBuffer[i] = pSrcBuffer[i] - pSrcBuffer[i + width];

    // process rest lines of frame except last
    for (j = 1; j < lastLine; j++)
    {
        const float* __restrict pCurrLine = pSrcBuffer + (j - 1) * width;
        const float* __restrict pNextLine = pSrcBuffer + (j + 1) * width;
              float* __restrict pDestLine = pDstBuffer + j * width;

        for (i = 0; i < width; i++)
            pDestLine[i] = pNextLine[i] - pCurrLine[i];

    } // for (j = 0; j < lastLine; j++)

    // process last line in frame
    const float* __restrict pCurrLine = pSrcBuffer + lastLine * width;
    const float* __restrict pNextLine = pSrcBuffer + (lastLine - 1) * width;
          float* __restrict pDestLine = pDstBuffer + lastLine * width;
    for (i = 0; i < width; i++)
        pDestLine[i] = pNextLine[i] - pCurrLine[i];

    return;
}

constexpr float GradientThreshold = 7.0f;

inline void ImageBW_GradientMerge
(
    float*  __restrict pBuffer1, // gray-level image
    float*  __restrict pBuffer2, // buffer with vertical gradient
    float*  __restrict pBuffer3, // buffer with horizontal gradient
    const A_long       width,
    const A_long       height,
    const float        threshold = GradientThreshold
) noexcept
{
    const A_long totalPix = height * width;
    __VECTOR_ALIGNED__
    for (A_long i = 0; i < totalPix; i++)
    {
        const float fVal = pBuffer1[i];
        const bool bMask = (threshold >= FastCompute::Sqrt(pBuffer2[i] * pBuffer2[i] + pBuffer3[i] * pBuffer3[i]) ? true : false);
        pBuffer1[i] = (true == bMask ? fVal : 0.f);
    }
    return;
}


inline void ImageBW_ComputeGradientBin
(
    float*  __restrict pBuffer1, // in/out buffer
    float*  __restrict pBuffer2, // temporary result with vertical gradient
    float*  __restrict pBuffer3, // temporary result with horizontal gradient
    const A_long       width,
    const A_long       height
) noexcept
{
    ImageBW_GradientVertical  (pBuffer1, pBuffer2, width, height);
    ImageBW_GradientHorizontal(pBuffer1, pBuffer3, width, height);
    ImageBW_GradientMerge (pBuffer1, pBuffer2, pBuffer3, width, height);
    return;
}


inline void ImageBW_AutomaticThreshold
(
    const float* __restrict pInBuffer,
    float* __restrict pOutBuffer,
    const A_long width,
    const A_long height
) noexcept
{
    // Calculate Otsu threshold ...
    CACHE_ALIGN uint32_t histBuf[256]{};
    const A_long imgSize = width * height;
    A_long i = 0;

    for (i = 0; i < imgSize; i++)
        histBuf[static_cast<uint8_t>(pInBuffer[i])]++;

    uint64_t totalIntencitySum = 0u;
    __VECTOR_ALIGNED__
    for (i = 0; i < 256; i++)
        totalIntencitySum += (static_cast<uint64_t>(histBuf[i]) * i);

    uint32_t sum_background = 0u;      // Cumulative sum of intensities for background
    uint32_t weight_background = 0u;   // Cumulative sum of pixel counts for background
    uint32_t weight_foreground = 0u;   // Cumulative sum of pixel counts for foreground
    const uint32_t numPixels = static_cast<uint32_t>(imgSize);

    float max_variance = 0.f, optimal_threshold = 0.f;

    for (i = 0; i < 256; i++)
    {
        weight_background += histBuf[i];
        if (0u == weight_background)
            continue;

        weight_foreground = numPixels - weight_background;
        if (0u == weight_foreground)
            break; // Remaining pixels are all background, no more separation possible

        sum_background += static_cast<uint32_t>(i) * histBuf[i];
        const float mean_background = static_cast<float>(sum_background) / static_cast<float>(weight_background);

        const uint64_t sum_foreground_direct = totalIntencitySum - static_cast<uint64_t>(sum_background);
        const float mean_foreground = static_cast<float>(sum_foreground_direct) / static_cast<float>(weight_foreground);

        const float between_class_variance = static_cast<float>(weight_background) * static_cast<float>(weight_foreground) * 
            (mean_background - mean_foreground) * (mean_background - mean_foreground);

        if (between_class_variance > max_variance)
        {
            max_variance = between_class_variance;
            optimal_threshold = static_cast<float>(i);
        }
    } // for (i = 0; i < 256; i++)

    // Apply calculated threshold for separate background from foreground
    const float otsu_threshold_val{ optimal_threshold };
    for (i = 0; i < imgSize; i++)
        pOutBuffer[i] = (pInBuffer[i] > otsu_threshold_val ? 255.f : 0.f);

    return;
}
