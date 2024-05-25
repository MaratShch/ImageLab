#include "MedianFilter.hpp"
#include "MedianFilterConstantTime.hpp"

bool median_filter_constant_time_BGRA_4444_8u
(
	const uint32_t* __restrict pSrcBuffer,
	      uint32_t* __restrict pDstBuffer,
	A_long sizeX,
	A_long sizeY,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept
{
	CACHE_ALIGN uint16_t pChannelHistR[256]{};
	CACHE_ALIGN uint16_t pChannelHistG[256]{};
	CACHE_ALIGN uint16_t pChannelHistB[256]{};
	uint16_t* pHist[3] = { pChannelHistR , pChannelHistG , pChannelHistB };

	const PF_Pixel_BGRA_8u* __restrict pSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pSrcBuffer);
	      PF_Pixel_BGRA_8u* __restrict pDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(pDstBuffer);

	median_filter_constant_time_RGB (pSrc, pDst, pHist, sizeY, sizeX, srcLinePitch, dstLinePitch, kernelSize);
	return true;
}