#pragma once

#include <climits>
#include "Avx2BitonicSort.hpp"
#include "Avx2Log.hpp"
#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "LibExport.hpp"

namespace AVX2
{
	namespace Median
	{

		bool median_filter_3x3_RGB_4444_8u
		(
			uint32_t* __restrict pInImage,
			uint32_t* __restrict pOutImage,
			A_long sizeX,
			A_long sizeY,
			A_long srcLinePitch,
			A_long dstLinePitch,
			const A_long& chanelMask = 0x00FFFFFF
		) noexcept;

		bool median_filter_3x3_RGB_4444_16u
		(
			uint64_t* __restrict pInImage,
			uint64_t* __restrict pOutImage,
			A_long sizeY,
			A_long sizeX,
			A_long srcLinePitch,
			A_long dstLinePitch,
			const A_long& chanelMaskL = 0xFFFFFFFF,
			const A_long& chanelMaskH = 0x0000FFFF
		) noexcept;

		bool median_filter_3x3_RGB_4444_32f
		(
			__m128* __restrict pInImage,
			__m128* __restrict pOutImage,
			A_long sizeX,
			A_long sizeY,
			A_long srcLinePitch,
			A_long dstLinePitch
		) noexcept;

		bool median_filter_5x5_RGB_4444_8u
		(
			uint32_t* __restrict pInImage,
			uint32_t* __restrict pOutImage,
			A_long sizeY,
			A_long sizeX,
			A_long srcLinePitch,
			A_long dstLinePitch,
			const A_long& chanelMask = 0x00FFFFFF
		) noexcept;

		bool median_filter_5x5_RGB_4444_16u
		(
			uint64_t* __restrict pInImage,
			uint64_t* __restrict pOutImage,
			A_long sizeY,
			A_long sizeX,
			A_long srcLinePitch,
			A_long dstLinePitch,
			const A_long& chanelMaskL = 0xFFFFFFFF,
			const A_long& chanelMaskH = 0x0000FFFF
		) noexcept;

		bool median_filter_7x7_RGB_4444_8u
		(
			uint32_t* __restrict pInImage,
			uint32_t* __restrict pOutImage,
			A_long sizeY,
			A_long sizeX,
			A_long srclinePitch,
			A_long dstLinePitch,
			const A_long& chanelMask = 0x00FFFFFF
		) noexcept;

		bool median_filter_7x7_RGB_4444_16u
		(
			uint64_t* __restrict pInImage,
			uint64_t* __restrict pOutImage,
			A_long sizeY,
			A_long sizeX,
			A_long srcLinePitch,
			A_long dstLinePitch,
			const A_long& chanelMaskL = 0xFFFFFFFF,
			const A_long& chanelMaskH = 0x0000FFFF
		) noexcept;

	} /* namespace Median */

} /* namespace AVX2 */