#include "MedianFilter.hpp"
#include "MedianFilterConstantTime.hpp"



bool median_filter_constant_time_BGRA_4444_8u
(
	const uint32_t* __restrict pSrcBuffer,
	      uint32_t* __restrict pDstBuffer,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept
{
	constexpr size_t histSize{ UCHAR_MAX + 1 };
	CACHE_ALIGN HistElem pChannelHistR[histSize];
	CACHE_ALIGN HistElem pChannelHistG[histSize];
	CACHE_ALIGN HistElem pChannelHistB[histSize];

	const HistHolder pHistArray = { pChannelHistR , pChannelHistG , pChannelHistB };
	const PF_Pixel_BGRA_8u* __restrict pSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pSrcBuffer);
	      PF_Pixel_BGRA_8u* __restrict pDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(pDstBuffer);

	median_filter_constant_time_RGB (pSrc, pDst, pHistArray, histSize, sizeY, sizeX, srcLinePitch, dstLinePitch, kernelSize);

	return true;
}


bool median_filter_constant_time_BGRA_4444_16u
(
	const uint32_t* __restrict pSrcBuffer,
	      uint32_t* __restrict pDstBuffer,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept
{
	constexpr size_t histSize{ CreateAlignment(SHRT_MAX + 2, 32)};
	CACHE_ALIGN HistElem pChannelHistR[histSize];
	CACHE_ALIGN HistElem pChannelHistG[histSize];
	CACHE_ALIGN HistElem pChannelHistB[histSize];

	const HistHolder pHistArray = { pChannelHistR , pChannelHistG , pChannelHistB };
	const PF_Pixel_BGRA_16u* __restrict pSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pSrcBuffer);
	      PF_Pixel_BGRA_16u* __restrict pDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(pDstBuffer);

	median_filter_constant_time_RGB (pSrc, pDst, pHistArray, histSize, sizeY, sizeX, srcLinePitch, dstLinePitch, kernelSize);

	return true;
}


bool median_filter_constant_time_BGRA_4444_32f
(
	const PF_Pixel_BGRA_32f* __restrict pSrcBuffer,
	      PF_Pixel_BGRA_32f* __restrict pDstBuffer,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept
{
	constexpr size_t histSize{ UCHAR_MAX + 1 };
	CACHE_ALIGN HistElem pChannelHistR[histSize];
	CACHE_ALIGN HistElem pChannelHistG[histSize];
	CACHE_ALIGN HistElem pChannelHistB[histSize];

	const HistHolder pHistArray = { pChannelHistR , pChannelHistG , pChannelHistB };
	median_filter_constant_time_RGB_32f (pSrcBuffer, pDstBuffer, pHistArray, histSize, sizeY, sizeX, srcLinePitch, dstLinePitch, kernelSize);

	return true;
}


bool median_filter_constant_time_VUYA_4444_8u
(
	const uint32_t* __restrict pSrcBuffer,
	      uint32_t* __restrict pDstBuffer,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept
{
	constexpr size_t histSize{ UCHAR_MAX + 1 };
	CACHE_ALIGN HistElem pChannelHistY[histSize];

	const PF_Pixel_VUYA_8u* __restrict pSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pSrcBuffer);
	      PF_Pixel_VUYA_8u* __restrict pDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(pDstBuffer);

	median_filter_constant_time_YUV (pSrc, pDst, pChannelHistY, histSize, sizeY, sizeX, srcLinePitch, dstLinePitch, kernelSize);

	return true;
}


bool median_filter_constant_time_VUYA_4444_32f
(
	const PF_Pixel_VUYA_32f* __restrict pSrcBuffer,
	      PF_Pixel_VUYA_32f* __restrict pDstBuffer,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept
{
	constexpr size_t histSize{ UCHAR_MAX + 1 }; /* internally convert pixel color from f32 to u8 */
	CACHE_ALIGN HistElem pChannelHistY[histSize];

	median_filter_constant_time_YUV_32f (pSrcBuffer, pDstBuffer, pChannelHistY, histSize, sizeY, sizeX, srcLinePitch, dstLinePitch, kernelSize);

	return true;
}