#pragma once

#include <climits>
#include <immintrin.h>
#include "Avx2Log.hpp"
#include "Common.hpp"
#include "CommonPixFormat.hpp"


namespace AVX2
{
	namespace Median
	{

		bool median_filter_3x3_BGRA_4444_8u
		(
			uint32_t* __restrict pInImage,
			uint32_t* __restrict pOutImage,
			A_long sizeX,
			A_long sizeY,
			A_long srcLinePitch,
			A_long dstLinePitch,
			const A_long& chanelMask = 0x00FFFFFF
		) noexcept;

		bool median_filter_5x5_BGRA_4444_8u
		(
			uint32_t* __restrict pInImage,
			uint32_t* __restrict pOutImage,
			A_long sizeY,
			A_long sizeX,
			A_long srcLinePitch,
			A_long dstLinePitch,
			const A_long& chanelMask = 0x00FFFFFF
		) noexcept;

		bool median_filter_7x7_BGRA_4444_8u
		(
			uint32_t* __restrict pInImage,
			uint32_t* __restrict pOutImage,
			A_long sizeY,
			A_long sizeX,
			A_long srclinePitch,
			A_long dstLinePitch,
			const A_long& chanelMask = 0x00FFFFFF
		) noexcept;


	} /* namespace Median */

} /* namespace AVX2 */