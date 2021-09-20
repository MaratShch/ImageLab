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
			const PF_Pixel_BGRA_8u* __restrict pInImage,
			      PF_Pixel_BGRA_8u* __restrict pOutImage,
			      A_long sizeX,
			      A_long sizeY,
			      A_long linePitch
		) noexcept;
	
		bool median_filter_3x3_VUYA_4444_8u_luma_only
		(
			const PF_Pixel_VUYA_8u* __restrict pInImage,
			      PF_Pixel_VUYA_8u* __restrict pOutImage,
			      A_long sizeX,
			      A_long sizeY,
			      A_long linePitch
		) noexcept;

		bool median_filter_3x3_VUYA_4444_8u
		(
			const PF_Pixel_VUYA_8u* __restrict pInImage,
			PF_Pixel_VUYA_8u* __restrict pOutImage,
			A_long sizeX,
			A_long sizeY,
			A_long linePitch
		) noexcept;

	} /* namespace Median */

} /* namespace AVX2 */