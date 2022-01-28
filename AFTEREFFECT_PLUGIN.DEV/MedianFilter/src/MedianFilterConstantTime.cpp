#include "MedianFilter.hpp"

bool median_filter_constant_time_BGRA_4444_8u
(
	uint32_t* __restrict pInImage,
	uint32_t* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	const A_long& kernelSize
) noexcept
{
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pInImage);
	PF_Pixel_BGRA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pOutImage);
	A_long i, j;

	for (j = 0; j < sizeY; j++)
	{
		CACHE_ALIGN uint32_t mHist[3][256]{};

		for (i = 0; i < sizeX; i++)
		{

		}
	}

	return true;
}