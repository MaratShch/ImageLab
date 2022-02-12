#pragma once

#include <climits>
#include <immintrin.h>
#include "Avx2Log.hpp"
#include "Avx2Array.hpp"
#include "Common.hpp"
#include "CommonPixFormat.hpp"

namespace AVX2
{
	namespace Morphology
	{
		namespace Erode
		{

			inline void Erode_3x3_8U
			(
				const __m256i  src[9],	/* source pixels		*/
				const __m256i& selem,	/* structured element	*/
				const __m256i& smask,	/* store mask			*/
				      __m256i& dst		/* destination pixel	*/
			) noexcept;

		};
	};
};


namespace AVX2
{
	namespace Morphology
	{
		namespace Dilate
		{

		};
	};
};


namespace AVX2
{
	namespace Morphology
	{
		namespace Open
		{

		};
	};
};


namespace AVX2
{
	namespace Morphology
	{
		namespace Close
		{

		};
	};
};