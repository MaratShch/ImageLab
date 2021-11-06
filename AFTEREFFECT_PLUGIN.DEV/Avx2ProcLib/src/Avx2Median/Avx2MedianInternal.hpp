#include "Avx2Median.hpp"


inline void StoreByMask8u(__m256i* __restrict pDst, const __m256i& valueOrig, const __m256i& valueMedian, const __m256i& storeMask) noexcept
{
	_mm256_storeu_si256(pDst, _mm256_blendv_epi8(valueOrig, valueMedian, storeMask));
}

inline void VectorSort8uPacked (__m256i& a, __m256i& b) noexcept
{
	const __m256i t = a;
	a = _mm256_min_epu8(t, b);
	b = _mm256_max_epu8(t, b);
}

inline void VectorSort16uPacked(__m256i& a, __m256i& b) noexcept
{
	const __m256i t = a;
	a = _mm256_min_epu16(t, b);
	b = _mm256_max_epu16(t, b);
}

inline void VectorSort32fPacked(__m256& a, __m256& b) noexcept
{
	const __m256 t = a;
	a = _mm256_min_ps(t, b);
	b = _mm256_max_ps(t, b);
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

#if 0
	template <typename T>
	inline void ScalarSortRGB (T& x, T& y) noexcept
	{
		int32_t b1 = static_cast<int32_t>(x.B);
		int32_t b2 = static_cast<int32_t>(y.B);
		int32_t g1 = static_cast<int32_t>(x.G);
		int32_t g2 = static_cast<int32_t>(y.G);
		int32_t r1 = static_cast<int32_t>(x.R);
		int32_t r2 = static_cast<int32_t>(y.R);

		SortInt (b1, b2);
		SortInt (g1, g2);
		SortInt (r1, r2);
		
		x.B = b1;
		x.G = g1;
		x.R = r1;

		y.B = b2;
		y.G = g2;
		y.R = r2;
	}

	template <typename T>
	inline void PartialScalarSort_9_elem_RGB (T a[9]) noexcept
	{
		ScalarSortRGB(a[1], a[2]);
		ScalarSortRGB(a[4], a[5]);
		ScalarSortRGB(a[7], a[8]);
		ScalarSortRGB(a[0], a[1]);
		ScalarSortRGB(a[3], a[4]);
		ScalarSortRGB(a[6], a[7]);
		ScalarSortRGB(a[1], a[2]);
		ScalarSortRGB(a[4], a[5]);
		ScalarSortRGB(a[7], a[8]);
		ScalarSortRGB(a[0], a[3]);
		ScalarSortRGB(a[5], a[8]);
		ScalarSortRGB(a[4], a[7]);
		ScalarSortRGB(a[3], a[6]);
		ScalarSortRGB(a[1], a[4]);
		ScalarSortRGB(a[2], a[5]);
		ScalarSortRGB(a[4], a[7]);
		ScalarSortRGB(a[4], a[2]);
		ScalarSortRGB(a[6], a[4]);
		ScalarSortRGB(a[4], a[2]);
	}


}; /* namespace MedianSort */


namespace MedianLoad5x5
{
	inline void LoadLineFromLeft0_4444_8u_packed (uint32_t* __restrict pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
		elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	}

	inline void LoadLineFromLeft1_4444_8u_packed (uint32_t* __restrict pSrc, __m256i elemLine[4]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
		elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
		elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	}

	inline void LoadLine_4444_8u_packed (uint32_t* __restrict pSrc, __m256i elemLine[5]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
		elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
		elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
		elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	}

	inline void LoadLineFromRight0_4444_8u_packed(uint32_t* __restrict pSrc, __m256i elemLine[4]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
		elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
		elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	}

	inline void LoadLineFromRight1_4444_8u_packed(uint32_t* __restrict pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
		elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	}

	/* load from Pixel [row = 0, line = 0] */
	inline void LoadWindowsPixel00 (uint32_t* __restrict pSrc, uint32_t* __restrict pNext0, uint32_t* __restrict pNext1, __m256i elem[9])
	{
		LoadLineFromLeft0_4444_8u_packed(pSrc, elem);
		LoadLineFromLeft0_4444_8u_packed(pNext0, elem + 3);
		LoadLineFromLeft0_4444_8u_packed(pNext1, elem + 6);
	}

	inline void LoadWindowsLeft1FirstLine (uint32_t* __restrict pSrc, uint32_t* __restrict pNext0, uint32_t* __restrict pNext1, __m256i elem[16])
	{
		LoadLineFromLeft1_4444_8u_packed(pSrc, elem);
		LoadLineFromLeft1_4444_8u_packed(pNext0, elem + 4);
		LoadLineFromLeft1_4444_8u_packed(pNext1, elem + 8);
	}

#if 0
	inline void LoadLineFromLeft0_4444_8u_packed (uint32_t* pSrc, __m256i elemLine[5]) noexcept
	{
		elemLine[0] = elemLine[1] = elemLine[2] =_mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
		elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
		elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	}

	inline void LoadLineFromLeft1_4444_8u_packed (uint32_t* pSrc, __m256i elemLine[5]) noexcept
	{
		elemLine[0] = elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
		elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
		elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	}

	inline void LoadLine_4444_8u_packed (uint32_t* pSrc, __m256i elemLine[5]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
		elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
		elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
		elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	}

	inline void LoadLineFromRigth0_444_8u_packed(uint32_t* pSrc, __m256i elemLine[5]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
		elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
		elemLine[2] = elemLine[3] = elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	}
#endif
}; /* namespace MedianLoad5x5 */

namespace MedianLoad3x3
{
	inline void LoadLineFromLeft_4444_8u_packed (uint32_t* pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	}

	inline void LoadLine_4444_8u_packed (uint32_t* pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
		elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc    ));
		elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	}

	inline void LoadLineFromRigth_444_8u_packed (uint32_t* pSrc, __m256i elemLine[3]) noexcept
	{
		elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc-1));
		elemLine[1] = elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	}


	inline const __m256i LoadWindowLeftFirstLine (uint32_t* pCurr, uint32_t* pNext, __m256i elem[4]) noexcept
	{
		LoadLineFromLeft_4444_8u_packed(pCurr, elem);
		LoadLineFromLeft_4444_8u_packed(pNext, elem + 2);
		return elem[1]; /* return current element from source */
	}

	inline const __m256i LoadWindowLeft (uint32_t* pPrev, uint32_t* pCurr, uint32_t* pNext, __m256i elem[9]) noexcept
	{
		LoadLineFromLeft_4444_8u_packed (pPrev, elem);
		LoadLineFromLeft_4444_8u_packed (pCurr, elem + 3);
		LoadLineFromLeft_4444_8u_packed (pNext, elem + 6);
		return elem[4]; /* return current element from source */
	}


	inline const __m256i LoadWindowRight (uint32_t* pPrev, uint32_t* pCurr, uint32_t* pNext, __m256i elem[9]) noexcept
	{
		LoadLineFromRigth_444_8u_packed (pPrev, elem);
		LoadLineFromRigth_444_8u_packed (pCurr, elem + 3);
		LoadLineFromRigth_444_8u_packed (pNext, elem + 6);
		return elem[4]; /* return current element from source */
	}

	inline const __m256i LoadWindow (uint32_t* pPrev, uint32_t* pCurr, uint32_t* pNext, __m256i elem[9]) noexcept
	{
		LoadLine_4444_8u_packed (pPrev, elem);
		LoadLine_4444_8u_packed (pCurr, elem + 3);
		LoadLine_4444_8u_packed (pNext, elem + 6);
		return elem[4]; /* return current element from source */
	}

	template <typename T>
	inline void LoadLineScalarLeft_RGB_packed (T* pSrc, T elemLine[3]) noexcept
	{
		elemLine[0] = elemLine[1] = *pSrc;
		elemLine[2] = *(pSrc + 1);
	}

	template <typename T>
	inline void LoadLineScalar_RGB_packed (T* pSrc, T elemLine[3]) noexcept
	{
		elemLine[0] = *(pSrc - 1);
		elemLine[1] = *pSrc;
		elemLine[2] = *(pSrc + 1);
	}

	template <typename T>
	inline void LoadLineScalarRight_RGB_packed (T* pSrc, T elemLine[3]) noexcept
	{
		elemLine[0] = *(pSrc - 1);
		elemLine[1] = elemLine[2] = *pSrc;
	}

	template <typename T>
	inline const T LoadWindowScalarLeft_RGB(T* pPrev, T* pCurr, T* pNext, T elem[9]) noexcept
	{
		LoadLineScalarLeft_RGB_packed (pPrev, elem);
		LoadLineScalarLeft_RGB_packed (pCurr, elem + 3);
		LoadLineScalarLeft_RGB_packed (pNext, elem + 6);
		return elem[4];
	}

	template <typename T>
	inline const T LoadWindowScalar_RGB (T* pPrev, T* pCurr, T* pNext, T elem[9]) noexcept
	{
		LoadLineScalar_RGB_packed (pPrev, elem);
		LoadLineScalar_RGB_packed (pCurr, elem + 3);
		LoadLineScalar_RGB_packed (pNext, elem + 6);
		return elem[4];
	}

	template <typename T>
	inline const T LoadWindowScalarRight_RGB (T* pPrev, T* pCurr, T* pNext, T elem[9]) noexcept
	{
		LoadLineScalarRight_RGB_packed (pPrev, elem);
		LoadLineScalarRight_RGB_packed (pCurr, elem + 3);
		LoadLineScalarRight_RGB_packed (pNext, elem + 6);
		return elem[4];
	}

}; /* namespace MedianLoad3x3 */


namespace MedianStore
{
	inline void StoreByMask8u (__m256i* __restrict pDst, const __m256i& valueOrig, const __m256i& valueMedian, const __m256i& storeMask) noexcept
	{
		_mm256_storeu_si256(pDst, _mm256_blendv_epi8(valueOrig, valueMedian, storeMask));
	}

	template <typename T>
	inline void ScalarStore_RGB (T* __restrict pDst, const T& valueOrig, const T& valueMedian) noexcept
	{
		pDst->B = valueMedian.B;
		pDst->G = valueMedian.G;
		pDst->R = valueMedian.R;
		pDst->A = valueOrig.A;
	}

}; /* namespace MedianStore  */

#endif