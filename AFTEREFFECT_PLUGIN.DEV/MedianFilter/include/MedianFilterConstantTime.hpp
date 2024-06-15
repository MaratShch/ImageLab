#include "CommonAdobeAE.hpp"
#include "CommonPixFormatSFINAE.hpp"
#include "Avx2Histogram.hpp"
#include "FastAriphmetics.hpp"
#include <array>

using HistElem   = uint16_t;
using HistHolder = std::array<HistElem*, 3>;

constexpr float fFloatScaler = 255.f;

/////////////////////////////////////////////////////////////////////////////////////////////
//  Templated version for RGB (BGRA or ARGB) pixel format                                  //
/////////////////////////////////////////////////////////////////////////////////////////////
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

	for (A_long j = lineTop; j <= lineBottom; j++)
	{
		const A_long lineIdx = FastCompute::Min((sizeY - 1), FastCompute::Max(0, j));
		const T* __restrict pLine = pSrc + lineIdx * linePitch;

		for (A_long i = pixLeft; i <= pixRight; i++)
		{
			const A_long idxPix = FastCompute::Min((sizeX - 1), FastCompute::Max(0, i));
			const T& pixel = pLine[idxPix];

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
	const A_long pixPrev    = numbPix - kerRadius - 1;

	const A_long idxPixPrev = FastCompute::Min((sizeX - 1), FastCompute::Max(0, pixPrev));
	const A_long idxPixNext = FastCompute::Min((sizeX - 1), pixRight);

	for (A_long j = lineTop; j <= lineBottom; j++)
	{
		const A_long lineIdx = FastCompute::Min((sizeY - 1), FastCompute::Max(0, j));
		const T* __restrict pLine = pSrc + lineIdx * linePitch;

		const T& pixelPrev = pLine[idxPixPrev];
		/* remove previous row */
		hR[pixelPrev.R]--;
		hG[pixelPrev.G]--;
		hB[pixelPrev.B]--;

		const T& pixelLast = pLine[idxPixNext];
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
	const HistElem* __restrict hR,
	const HistElem* __restrict hG,
	const HistElem* __restrict hB,
	A_long               histSize,
	A_long               medianSample
) noexcept
{
	A_long idxR, idxG, idxB, samplesR, samplesG, samplesB;

	for (samplesR = idxR = 0; idxR < histSize; ++idxR)
	{
		samplesR += static_cast<A_long>(hR[idxR]);
		if (samplesR >= medianSample)
			break;
	}
	for (samplesG = idxG = 0; idxG < histSize; ++idxG)
	{
		samplesG += static_cast<A_long>(hG[idxG]);
		if (samplesG >= medianSample)
			break;
	}
	for (samplesB = idxB = 0; idxB < histSize; ++idxB)
	{
		samplesB += static_cast<A_long>(hB[idxB]);
		if (samplesB >= medianSample)
			break;
	}

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
	const HistHolder& histArray,
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
	const A_long medianSample = (kernelSize * kernelSize) >> 1;
	const A_long kernelRadius = kernelSize >> 1;

	for (A_long j = 0; j < sizeY; j++)
	{
		AVX2::Histogram::clean_hist_buffer (pHistR, histBytesSize);
		AVX2::Histogram::clean_hist_buffer (pHistG, histBytesSize);
		AVX2::Histogram::clean_hist_buffer (pHistB, histBytesSize);

		initHistogram (pInImage, pHistR, pHistG, pHistB, kernelRadius, sizeX, sizeY, j, srcLinePitch);
		const A_long srcPixIdx = j * srcLinePitch;
		const A_long dstPixIdx = j * dstLinePitch;

		pOutImage[dstPixIdx] = medianPixel (pInImage[srcPixIdx], pHistR, pHistG, pHistB, histSize, medianSample);

		for (A_long i = 1; i < sizeX; i++)
		{
			updateHistogram (pInImage, pHistR, pHistG, pHistB, kernelRadius, sizeX, sizeY, j, i, srcLinePitch);
			pOutImage[dstPixIdx + i] = medianPixel (pInImage[srcPixIdx + i], pHistR, pHistG, pHistB, histSize, medianSample);
		}
	}

	return;
}


/////////////////////////////////////////////////////////////////////////////////////////////
//  Templated version for VUYA pixel format                                                //
/////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void initHistogram
(
	const T*  __restrict pSrc,
	HistElem* __restrict hLuma,
	const A_long&        kerRadius,
	const A_long&        sizeX,
	const A_long&        sizeY,
	const A_long&        numbLine,
	const A_long&        linePitch
) noexcept
{
	const A_long lineTop = numbLine - kerRadius;
	const A_long lineBottom = numbLine + kerRadius;
	const A_long pixLeft = -kerRadius;
	const A_long pixRight = kerRadius;

	for (A_long j = lineTop; j <= lineBottom; j++)
	{
		const A_long lineIdx = FastCompute::Min((sizeY - 1), FastCompute::Max(0, j));
		const T* __restrict pLine = pSrc + lineIdx * linePitch;

		for (A_long i = pixLeft; i <= pixRight; i++)
		{
			const A_long idxPix = FastCompute::Min((sizeX - 1), FastCompute::Max(0, i));
			const T& pixel = pLine[idxPix];

			hLuma[pixel.Y]++;
		} /* for (A_long i = pixLeft; i < pixRight; i++) */
	} /* for (A_long j = lineTop; j < LineBottom; j++) */
	return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void updateHistogram
(
	const T*  __restrict pSrc,
	HistElem* __restrict hLuma,
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

	const A_long idxPixPrev = FastCompute::Min((sizeX - 1), FastCompute::Max(0, pixPrev));
	const A_long idxPixNext = FastCompute::Min((sizeX - 1), pixRight);

	for (A_long j = lineTop; j <= lineBottom; j++)
	{
		const A_long lineIdx = FastCompute::Min((sizeY - 1), FastCompute::Max(0, j));
		const T* __restrict pLine = pSrc + lineIdx * linePitch;

		const T& pixelPrev = pLine[idxPixPrev];
		const T& pixelLast = pLine[idxPixNext];

		/* remove previous row */
		hLuma[pixelPrev.Y]--;
		/* add new row         */
		hLuma[pixelLast.Y]++;

	} /* for (A_long i = pixLeft; i < pixRight; i++) */

	return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline T medianPixel
(
	const T&             pSrc,
	const HistElem* __restrict hLuma,
	A_long               histSize,
	A_long               medianSample
) noexcept
{
	A_long idxLuma;
	for (A_long samplesLuma = idxLuma = 0; idxLuma < histSize; ++idxLuma)
	{
		samplesLuma += static_cast<A_long>(hLuma[idxLuma]);
		if (samplesLuma >= medianSample)
			break;
	}

	T outPix;
	outPix.A = pSrc.A;
	outPix.V = pSrc.V;
	outPix.U = pSrc.U;
	outPix.Y = idxLuma;

	return outPix;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void median_filter_constant_time_YUV
(
	const T*  __restrict pInImage,
	      T*  __restrict pOutImage,
	HistElem* __restrict histLuma,
	size_t histSize,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept
{
	const size_t histBytesSize = sizeof(histLuma[0]) * histSize;
	const A_long medianSample  = (kernelSize * kernelSize) >> 1;
	const A_long kernelRadius  = kernelSize >> 1;

	for (A_long j = 0; j < sizeY; j++)
	{
		AVX2::Histogram::clean_hist_buffer (histLuma, histBytesSize);

		initHistogram (pInImage, histLuma, kernelRadius, sizeX, sizeY, j, srcLinePitch);
		const A_long srcPixIdx = j * srcLinePitch;
		const A_long dstPixIdx = j * dstLinePitch;

		pOutImage[dstPixIdx] = medianPixel (pInImage[srcPixIdx], histLuma, histSize, medianSample);

		for (A_long i = 1; i < sizeX; i++)
		{
			updateHistogram (pInImage, histLuma, kernelRadius, sizeX, sizeY, j, i, srcLinePitch);
			pOutImage[dstPixIdx + i] = medianPixel (pInImage[srcPixIdx + i], histLuma, histSize, medianSample);
		}
	}

	return;
}


/////////////////////////////////////////////////////////////////////////////////////////////
//  Version for 32-bits per color in VUYA pixel format                                     //
/////////////////////////////////////////////////////////////////////////////////////////////
inline void initHistogram
(
	const PF_Pixel_VUYA_32f*  __restrict pSrc,
	      HistElem* __restrict hLuma,
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

	for (A_long j = lineTop; j <= lineBottom; j++)
	{
		const A_long lineIdx = FastCompute::Min((sizeY - 1), FastCompute::Max(0, j));
		const PF_Pixel_VUYA_32f* __restrict pLine = pSrc + lineIdx * linePitch;

		for (A_long i = pixLeft; i <= pixRight; i++)
		{
			const A_long idxPix = FastCompute::Min((sizeX - 1), FastCompute::Max(0, i));
			const PF_Pixel_VUYA_32f& pixel = pLine[idxPix];

			hLuma[static_cast<int32_t>(pixel.Y * fFloatScaler)]++;
		} /* for (A_long i = pixLeft; i < pixRight; i++) */
	} /* for (A_long j = lineTop; j < LineBottom; j++) */
	return;
}


inline PF_Pixel_VUYA_32f medianPixel
(
	const PF_Pixel_VUYA_32f&   pSrc,
	const HistElem* __restrict hLuma,
	A_long                     histSize,
	A_long                     medianSample
) noexcept
{
	A_long idxLuma;

	for (A_long samplesLuma = idxLuma = 0; idxLuma < histSize; ++idxLuma)
	{
		samplesLuma += static_cast<A_long>(hLuma[idxLuma]);
		if (samplesLuma >= medianSample)
			break;
	}

	constexpr float fRecip255 = 1.f / fFloatScaler;
	PF_Pixel_VUYA_32f outPix;
	outPix.A = pSrc.A;
	outPix.V = pSrc.V;
	outPix.U = pSrc.U;
	outPix.Y = static_cast<float>(idxLuma) * fRecip255;

	return outPix;
}


inline void updateHistogram
(
	const PF_Pixel_VUYA_32f* __restrict pSrc,
	HistElem* __restrict hLuma,
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
	const A_long pixRight   = numbPix + kerRadius;
	const A_long pixPrev    = numbPix - kerRadius - 1;

	const A_long idxPixPrev = FastCompute::Min((sizeX - 1), FastCompute::Max(0, pixPrev));
	const A_long idxPixNext = FastCompute::Min((sizeX - 1), pixRight);

	for (A_long j = lineTop; j <= lineBottom; j++)
	{
		const A_long lineIdx = FastCompute::Min((sizeY - 1), FastCompute::Max(0, j));
		const PF_Pixel_VUYA_32f* __restrict pLine = pSrc + lineIdx * linePitch;

		const PF_Pixel_VUYA_32f& pixelPrev = pLine[idxPixPrev];
		const PF_Pixel_VUYA_32f& pixelLast = pLine[idxPixNext];

		const A_long prevLumaIdx = static_cast<A_long>(pixelPrev.Y * fFloatScaler);
		const A_long nextLumaIdx = static_cast<A_long>(pixelLast.Y * fFloatScaler);
		/* remove previous row */
		hLuma[prevLumaIdx]--;
		/* add new row         */
		hLuma[nextLumaIdx]++;

	} /* for (A_long i = pixLeft; i < pixRight; i++) */

	return;
}

inline void median_filter_constant_time_YUV_32f
(
	const PF_Pixel_VUYA_32f*  __restrict pInImage,
	      PF_Pixel_VUYA_32f*  __restrict pOutImage,
	HistElem* __restrict histLuma,
	size_t histSize,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept
{
	const size_t histBytesSize = sizeof(histLuma[0]) * histSize;
	const A_long medianSample = (kernelSize * kernelSize) >> 1;
	const A_long kernelRadius = kernelSize >> 1;

	for (A_long j = 0; j < sizeY; j++)
	{
		AVX2::Histogram::clean_hist_buffer (histLuma, histBytesSize);

		initHistogram (pInImage, histLuma, kernelRadius, sizeX, sizeY, j, srcLinePitch);
		const A_long srcPixIdx = j * srcLinePitch;
		const A_long dstPixIdx = j * dstLinePitch;

		pOutImage[dstPixIdx] = medianPixel (pInImage[srcPixIdx], histLuma, histSize, medianSample);

		for (A_long i = 1; i < sizeX; i++)
		{
			updateHistogram (pInImage, histLuma, kernelRadius, sizeX, sizeY, j, i, srcLinePitch);
			pOutImage[dstPixIdx + i] = medianPixel (pInImage[srcPixIdx + i], histLuma, histSize, medianSample);
		}
	}

	return;
}

/////////////////////////////////////////////////////////////////////////////////////////////
//  Version for 32-bits per color in BGRA pixel format                                     //
/////////////////////////////////////////////////////////////////////////////////////////////
inline void initHistogram
(
	const PF_Pixel_BGRA_32f* __restrict pSrc,
	      HistElem*          __restrict hR,
	      HistElem*          __restrict hG,
	      HistElem*          __restrict hB,
	const A_long&        kerRadius,
	const A_long&        sizeX,
	const A_long&        sizeY,
	const A_long&        numbLine,
	const A_long&        linePitch
) noexcept
{
	const A_long lineTop = numbLine - kerRadius;
	const A_long lineBottom = numbLine + kerRadius;
	const A_long pixLeft = -kerRadius;
	const A_long pixRight = kerRadius;

	for (A_long j = lineTop; j <= lineBottom; j++)
	{
		const A_long lineIdx = FastCompute::Min((sizeY - 1), FastCompute::Max(0, j));
		const PF_Pixel_BGRA_32f* __restrict pLine = pSrc + lineIdx * linePitch;

		for (A_long i = pixLeft; i <= pixRight; i++)
		{
			const A_long idxPix = FastCompute::Min((sizeX - 1), FastCompute::Max(0, i));
			const PF_Pixel_BGRA_32f& pixel = pLine[idxPix];

			const A_long rIdx = static_cast<A_long>(pixel.R * fFloatScaler);
			const A_long gIdx = static_cast<A_long>(pixel.G * fFloatScaler);
			const A_long bIdx = static_cast<A_long>(pixel.B * fFloatScaler);

			hR[rIdx]++;
			hG[gIdx]++;
			hB[bIdx]++;
		} /* for (A_long i = pixLeft; i < pixRight; i++) */
	} /* for (A_long j = lineTop; j < LineBottom; j++) */
	return;
}


inline PF_Pixel_BGRA_32f medianPixel
(
	const PF_Pixel_BGRA_32f&   pSrc,
	const HistElem* __restrict hR,
	const HistElem* __restrict hG,
	const HistElem* __restrict hB,
	A_long               histSize,
	A_long               medianSample
) noexcept
{
	A_long idxR, idxG, idxB;

	for (A_long samplesR = idxR = 0; idxR < histSize; ++idxR)
	{
		samplesR += static_cast<A_long>(hR[idxR]);
		if (samplesR >= medianSample)
			break;
	}
	for (A_long samplesG = idxG = 0; idxG < histSize; ++idxG)
	{
		samplesG += static_cast<A_long>(hG[idxG]);
		if (samplesG >= medianSample)
			break;
	}
	for (A_long samplesB = idxB = 0; idxB < histSize; ++idxB)
	{
		samplesB += static_cast<A_long>(hB[idxB]);
		if (samplesB >= medianSample)
			break;
	}

	constexpr float fRecip255 = 1.f / fFloatScaler;

	PF_Pixel_BGRA_32f outPix;
	outPix.A = pSrc.A;
	outPix.R = static_cast<float>(idxR) * fRecip255;
	outPix.G = static_cast<float>(idxG) * fRecip255;
	outPix.B = static_cast<float>(idxB) * fRecip255;

	return outPix;
}


inline void updateHistogram
(
	const PF_Pixel_BGRA_32f*  __restrict pSrc,
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
	const A_long lineTop = numbLine - kerRadius;
	const A_long lineBottom = numbLine + kerRadius;
	const A_long pixRight = numbPix + kerRadius;
	const A_long pixPrev = numbPix - kerRadius - 1;

	const A_long idxPixPrev = FastCompute::Min((sizeX - 1), FastCompute::Max(0, pixPrev));
	const A_long idxPixNext = FastCompute::Min((sizeX - 1), pixRight);

	for (A_long j = lineTop; j <= lineBottom; j++)
	{
		const A_long lineIdx = FastCompute::Min((sizeY - 1), FastCompute::Max(0, j));
		const PF_Pixel_BGRA_32f* __restrict pLine = pSrc + lineIdx * linePitch;

		const PF_Pixel_BGRA_32f& pixelPrev = pLine[idxPixPrev];
		const A_long rPrevIdx = static_cast<A_long>(pixelPrev.R * fFloatScaler);
		const A_long gPrevIdx = static_cast<A_long>(pixelPrev.G * fFloatScaler);
		const A_long bPrevIdx = static_cast<A_long>(pixelPrev.B * fFloatScaler);
		/* remove previous row */
		hR[rPrevIdx]--;
		hG[gPrevIdx]--;
		hB[bPrevIdx]--;

		const PF_Pixel_BGRA_32f& pixelLast = pLine[idxPixNext];
		const A_long rNextIdx = static_cast<A_long>(pixelLast.R * fFloatScaler);
		const A_long gNextIdx = static_cast<A_long>(pixelLast.G * fFloatScaler);
		const A_long bNextIdx = static_cast<A_long>(pixelLast.B * fFloatScaler);
		/* add new row */
		hR[rNextIdx]++;
		hG[gNextIdx]++;
		hB[bNextIdx]++;
	} /* for (A_long i = pixLeft; i < pixRight; i++) */

	return;
}

inline void median_filter_constant_time_RGB_32f
(
	const PF_Pixel_BGRA_32f* __restrict pInImage,
	      PF_Pixel_BGRA_32f* __restrict pOutImage,
	const HistHolder& histArray,
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
	const size_t histBytesSize = sizeof(HistElem) * histSize;
	const A_long medianSample = (kernelSize * kernelSize) >> 1;
	const A_long kernelRadius = kernelSize >> 1;

	for (A_long j = 0; j < sizeY; j++)
	{
		AVX2::Histogram::clean_hist_buffer (pHistR, histBytesSize);
		AVX2::Histogram::clean_hist_buffer (pHistG, histBytesSize);
		AVX2::Histogram::clean_hist_buffer (pHistB, histBytesSize);

		initHistogram (pInImage, pHistR, pHistG, pHistB, kernelRadius, sizeX, sizeY, j, srcLinePitch);
		const A_long srcPixIdx = j * srcLinePitch;
		const A_long dstPixIdx = j * dstLinePitch;

		pOutImage[dstPixIdx] = medianPixel (pInImage[srcPixIdx], pHistR, pHistG, pHistB, histSize, medianSample);

		for (A_long i = 1; i < sizeX; i++)
		{
			updateHistogram (pInImage, pHistR, pHistG, pHistB, kernelRadius, sizeX, sizeY, j, i, srcLinePitch);
			pOutImage[dstPixIdx + i] = medianPixel (pInImage[srcPixIdx + i], pHistR, pHistG, pHistB, histSize, medianSample);
		}
	}

	return;
}