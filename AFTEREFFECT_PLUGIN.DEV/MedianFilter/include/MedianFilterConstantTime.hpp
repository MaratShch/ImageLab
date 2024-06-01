#include "CommonAdobeAE.hpp"
#include "CommonPixFormatSFINAE.hpp"
#include "Avx2Histogram.hpp"
#include "FastAriphmetics.hpp"
#include <array>

using HistElem = uint16_t;

template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void initHistogram
(
	const T*  __restrict pSrc,
	HistElem* __restrict hR,
	HistElem* __restrict hG,
	HistElem* __restrict hB,
	const A_long&        kerRadius,
	const A_long&        sizeX,
	const A_long&        sizeY,
	const A_long&        numbLine,
	const A_long&        linePitch
) noexcept
{
	const A_long lineTop    = numbLine - kerRadius;
	const A_long lineBottom = numbLine + kerRadius;
	const A_long pixLeft    = -kerRadius;
	const A_long pixRight   = kerRadius;

	for (A_long j = lineTop; j < lineBottom; j++)
	{
		const A_long lineIdx = std::min((sizeY - 1), std::max(0, j));
		const T* __restrict pLine = pSrc + lineIdx * linePitch;

		for (A_long i = pixLeft; i < pixRight; i++)
		{
			const A_long idxPix = std::min((sizeX - 1), std::max(0, i));
			const T pixel = pLine[idxPix];

			hR[pixel.R]++;
			hG[pixel.G]++;
			hB[pixel.B]++;

		} /* for (A_long i = pixLeft; i < pixRight; i++) */
	} /* for (A_long j = lineTop; j < LineBottom; j++) */
	return;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void updateHistogram
(
	const T*  __restrict pSrc,
	HistElem* __restrict hR,
	HistElem* __restrict hG,
	HistElem* __restrict hB,
	const A_long&        kerRadius,
	const A_long&        sizeX,
	const A_long&        sizeY,
	const A_long&        numbLine,
	const A_long&        numbPix,
	const A_long&        linePitch
) noexcept
{
	const A_long lineTop    = numbLine - kerRadius;
	const A_long lineBottom = numbLine + kerRadius;
	const A_long pixRight   = numbPix  + kerRadius;
	const A_long pixPrev    = numbPix  - kerRadius - 1;

	const A_long idxPixPrev = std::min((sizeX - 1), std::max(0, pixPrev));
	const A_long idxPixNext = std::min((sizeX - 1), pixRight);

	for (A_long j = lineTop; j < lineBottom; j++)
	{
		const A_long lineIdx      = std::min((sizeY - 1), std::max(0, j));
		const T* __restrict pLine = pSrc + lineIdx * linePitch;

		const T pixelPrev = pLine[idxPixPrev];
		/* remove previous row */
		hR[pixelPrev.R]--;
		hG[pixelPrev.G]--;
		hB[pixelPrev.B]--;

		const T pixelLast = pLine[idxPixNext];
		/* add new row */
		hR[pixelLast.R]++;
		hG[pixelLast.G]++;
		hB[pixelLast.B]++;
	} /* for (A_long i = pixLeft; i < pixRight; i++) */

	return;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline T medianPixel
(
	const T&             pSrc,
	HistElem* __restrict hR,
	HistElem* __restrict hG,
	HistElem* __restrict hB,
	size_t               histSize,
	A_long               medianSample
) noexcept
{
	A_long idxR, idxG, idxB, samples;

	for (samples = idxR = 0; idxR < histSize && samples < medianSample; idxR++)
		samples += hR[idxR];

	for (samples = idxG = 0; idxG < histSize && samples < medianSample; idxG++)
		samples += hG[idxG];

	for (samples = idxB = 0; idxB < histSize && samples < medianSample; idxB++)
		samples += hB[idxB];

	T outPix;
	outPix.A = pSrc.A;
	outPix.R = idxR;
	outPix.G = idxG;
	outPix.B = idxB;

	return outPix;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void median_filter_constant_time_RGB
(
	const T* __restrict pInImage,
	      T* __restrict pOutImage,
	const std::array<HistElem*, 3>& histArray,
	size_t histSize,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept
{
	HistElem* __restrict pHistR = histArray[0];
	HistElem* __restrict pHistG = histArray[1];
	HistElem* __restrict pHistB = histArray[2];
	const size_t histBytesSize  = sizeof(HistElem) * histSize;
	const A_long medianSample   = (kernelSize * kernelSize) >> 1;
	const A_long kernelRadius   = kernelSize >> 1;

	A_long i, j;

	for (j = 0; j < sizeY; j++)
	{
		AVX2::Histogram::clean_hist_buffer (pHistR, histBytesSize);
		AVX2::Histogram::clean_hist_buffer (pHistG, histBytesSize);
		AVX2::Histogram::clean_hist_buffer (pHistB, histBytesSize);

		initHistogram (pInImage, pHistR, pHistG, pHistB, kernelRadius, sizeX, sizeY, j, srcLinePitch);
		const A_long srcPixIdx = j * srcLinePitch;
		const A_long dstPixIdx = j * dstLinePitch;

		pOutImage[dstPixIdx] = medianPixel(pInImage[srcPixIdx], pHistR, pHistG, pHistB, histSize, medianSample);

		for (i = 1; i < sizeX; i++)
		{
			updateHistogram (pInImage, pHistR, pHistG, pHistB, kernelRadius, sizeX, sizeY, j, i, srcLinePitch);
			pOutImage[dstPixIdx + i] = medianPixel (pInImage[srcPixIdx + i], pHistR, pHistG, pHistB, histSize, medianSample);
		}
	}

	return;
}