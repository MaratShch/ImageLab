#include "Avx2Median.hpp"


inline void StoreByMask8u (__m256i* __restrict pDst, const __m256i& valueOrig, const __m256i& valueMedian, const __m256i& storeMask) noexcept
{
	_mm256_storeu_si256(pDst, _mm256_blendv_epi8(valueOrig, valueMedian, storeMask));
}

inline void StoreByMask16u (__m256i* __restrict pDst, const __m256i& valueOrig, const __m256i& valueMedian, const __m256i& storeMask) noexcept
{
	StoreByMask8u (pDst, valueOrig, valueMedian, storeMask); /* let's reuse 8u store function just with change the mask for correct store 16u data */
}

inline void SortFloat (float& a, float& b) noexcept
{
	if (a > b)
	{
		const float t{ a };
		a = b;
		b = t;
	}
}

inline void SortInt (int32_t& a, int32_t& b) noexcept
{
	const int32_t d{ a - b };
	const int32_t m{ ~(d >> 8) };
	b += d & m;
	a -= d & m;
}
