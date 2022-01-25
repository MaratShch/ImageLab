#include "Avx2Median.hpp"
#include "Avx2MedianInternal.hpp"


inline void LoadLinePixel0 (uint32_t* __restrict pSrc, __m256i elemLine[4]) noexcept
{
	//  | X  0  0  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 3));
}

inline void LoadLinePixel1 (uint32_t* __restrict pSrc, __m256i elemLine[5]) noexcept
{
	//  | 0  X  0  0  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 3));
}

inline void LoadLinePixel2 (uint32_t* __restrict pSrc, __m256i elemLine[6]) noexcept
{
	// | 0  0  X  0  0  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	elemLine[5] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 3));
}

inline void LoadLinePixel (uint32_t* __restrict pSrc, __m256i elemLine[7]) noexcept
{
	// 0  0  0  X  0  0  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 3));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[5] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
	elemLine[6] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 3));
}

inline void LoadLinePixelNminus2 (uint32_t* __restrict pSrc, __m256i elemLine[6]) noexcept
{
	// 0  0  0  X  0  0 |
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 3));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[5] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
}

inline void LoadLinePixelNminus1 (uint32_t* __restrict pSrc, __m256i elemLine[5]) noexcept
{
	// 0  0  0  X  0 |
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 3));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
}

inline void LoadLinePixelN (uint32_t* __restrict pSrc, __m256i elemLine[4]) noexcept
{
	// 0  0  0  X |
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 3));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
}


inline __m256i load_line0_pixel0
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[16]
) noexcept
{ 
	LoadLinePixel0 (pSrc, elem);
	LoadLinePixel0 (pNext1, elem + 4);
	LoadLinePixel0 (pNext2, elem + 8);
	LoadLinePixel0 (pNext3, elem + 12);
	return elem[0];
}

inline __m256i load_line0_pixel1
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[20]
) noexcept
{
	LoadLinePixel1 (pSrc,   elem);
	LoadLinePixel1 (pNext1, elem + 5);
	LoadLinePixel1 (pNext2, elem + 10);
	LoadLinePixel1 (pNext3, elem + 15);
	return elem[1];
}

inline __m256i load_line0_pixel2
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[24]
) noexcept
{
	LoadLinePixel2 (pSrc, elem);
	LoadLinePixel2 (pNext1, elem + 6);
	LoadLinePixel2 (pNext2, elem + 12);
	LoadLinePixel2 (pNext3, elem + 18);
	return elem[2];
}

inline __m256i load_line0_pixel
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[28]
) noexcept
{
	LoadLinePixel (pSrc, elem);
	LoadLinePixel (pNext1, elem + 7);
	LoadLinePixel (pNext2, elem + 14);
	LoadLinePixel (pNext3, elem + 21);
	return elem[3];
}

inline __m256i load_line0_pixel_n2
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[24]
) noexcept
{
	LoadLinePixelNminus2 (pSrc, elem);
	LoadLinePixelNminus2 (pNext1, elem + 6);
	LoadLinePixelNminus2 (pNext2, elem + 12);
	LoadLinePixelNminus2 (pNext3, elem + 18);
	return elem[3];
}

inline __m256i load_line0_pixel_n1
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[20]
) noexcept
{
	LoadLinePixelNminus1 (pSrc, elem);
	LoadLinePixelNminus1 (pNext1, elem + 5);
	LoadLinePixelNminus1 (pNext2, elem + 10);
	LoadLinePixelNminus1 (pNext3, elem + 15);
	return elem[3];
}

inline __m256i load_line0_pixel_n
(
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[16]
) noexcept
{
	LoadLinePixelN (pSrc, elem);
	LoadLinePixelN (pNext1, elem + 4);
	LoadLinePixelN (pNext2, elem + 8);
	LoadLinePixelN (pNext3, elem + 12);
	return elem[3];
}

inline __m256i load_line1_pixel0
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[20]
) noexcept
{
	LoadLinePixel0 (pPrev,  elem);
	LoadLinePixel0 (pSrc,   elem + 4);
	LoadLinePixel0 (pNext1, elem + 8);
	LoadLinePixel0 (pNext2, elem + 12);
	LoadLinePixel0 (pNext3, elem + 16);
	return elem[7];
}

inline __m256i load_line1_pixel1
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[25]
) noexcept
{
	LoadLinePixel1 (pPrev,  elem);
	LoadLinePixel1 (pSrc,   elem + 5);
	LoadLinePixel1 (pNext1, elem + 10);
	LoadLinePixel1 (pNext2, elem + 15);
	LoadLinePixel1 (pNext3, elem + 20);
	return elem[8];
}

inline __m256i load_line1_pixel2
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[30]
) noexcept
{
	LoadLinePixel2 (pPrev,  elem);
	LoadLinePixel2 (pSrc,   elem + 6);
	LoadLinePixel2 (pNext1, elem + 12);
	LoadLinePixel2 (pNext2, elem + 18);
	LoadLinePixel2 (pNext3, elem + 24);
	return elem[9];
}

inline __m256i load_line1_pixel
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[35]
) noexcept
{
	LoadLinePixel (pPrev,  elem);
	LoadLinePixel (pSrc,   elem + 7); 
	LoadLinePixel (pNext1, elem + 14);
	LoadLinePixel (pNext2, elem + 21);
	LoadLinePixel (pNext3, elem + 28);
	return elem[10];
}

inline __m256i load_line1_pixel_n2
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[30]
) noexcept
{
	LoadLinePixelNminus2 (pPrev,  elem);
	LoadLinePixelNminus2 (pSrc,   elem + 6);
	LoadLinePixelNminus2 (pNext1, elem + 12);
	LoadLinePixelNminus2 (pNext2, elem + 18);
	LoadLinePixelNminus2 (pNext3, elem + 24);
	return elem[9];
}

inline __m256i load_line1_pixel_n1
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[25]
) noexcept
{
	LoadLinePixelNminus1(pPrev,  elem);
	LoadLinePixelNminus1(pSrc,   elem + 5);
	LoadLinePixelNminus1(pNext1, elem + 10);
	LoadLinePixelNminus1(pNext2, elem + 15);
	LoadLinePixelNminus1(pNext3, elem + 20);
	return elem[8];
}

inline __m256i load_line1_pixel_n
(
	uint32_t* __restrict pPrev,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[20]
) noexcept
{
	LoadLinePixelN (pPrev,  elem);
	LoadLinePixelN (pSrc,   elem + 4);
	LoadLinePixelN (pNext1, elem + 8);
	LoadLinePixelN (pNext2, elem + 12);
	LoadLinePixelN (pNext3, elem + 16);
	return elem[7];
}

inline __m256i load_line2_pixel0
(
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[24]
) noexcept
{
	LoadLinePixel0 (pPrev2, elem);
	LoadLinePixel0 (pPrev1, elem + 4);
	LoadLinePixel0 (pSrc,   elem + 8);
	LoadLinePixel0 (pNext1, elem + 12);
	LoadLinePixel0 (pNext2, elem + 16);
	LoadLinePixel0 (pNext3, elem + 20);
	return elem[8];
}

inline __m256i load_line2_pixel1
(
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[30]
) noexcept
{
	LoadLinePixel1 (pPrev2, elem);
	LoadLinePixel1 (pPrev1, elem + 5);
	LoadLinePixel1 (pSrc,   elem + 10);
	LoadLinePixel1 (pNext1, elem + 15);
	LoadLinePixel1 (pNext2, elem + 20);
	LoadLinePixel1 (pNext3, elem + 25);
	return elem[11];
}

inline __m256i load_line2_pixel2
(
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[36]
) noexcept
{
	LoadLinePixel2 (pPrev2, elem);
	LoadLinePixel2 (pPrev1, elem + 6);
	LoadLinePixel2 (pSrc,   elem + 12);
	LoadLinePixel2 (pNext1, elem + 18);
	LoadLinePixel2 (pNext2, elem + 24);
	LoadLinePixel2 (pNext3, elem + 30);
	return elem[14];
}

inline __m256i load_line2_pixel
(
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[42]
) noexcept
{
	LoadLinePixel (pPrev2, elem);
	LoadLinePixel (pPrev1, elem + 7);
	LoadLinePixel (pSrc,   elem + 14);
	LoadLinePixel (pNext1, elem + 21);
	LoadLinePixel (pNext2, elem + 28);
	LoadLinePixel (pNext3, elem + 35);
	return elem[17];
}

inline __m256i load_line2_pixel_n2
(
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[36]
) noexcept
{
	LoadLinePixelNminus2 (pPrev2, elem);
	LoadLinePixelNminus2 (pPrev1, elem + 6);
	LoadLinePixelNminus2 (pSrc,   elem + 12);
	LoadLinePixelNminus2 (pNext1, elem + 18);
	LoadLinePixelNminus2 (pNext2, elem + 24);
	LoadLinePixelNminus2 (pNext3, elem + 30);
	return elem[15];
}

inline __m256i load_line2_pixel_n1
(
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[30]
) noexcept
{
	LoadLinePixelNminus1 (pPrev2, elem);
	LoadLinePixelNminus1 (pPrev1, elem + 5);
	LoadLinePixelNminus1 (pSrc,   elem + 10);
	LoadLinePixelNminus1 (pNext1, elem + 15);
	LoadLinePixelNminus1 (pNext2, elem + 20);
	LoadLinePixelNminus1 (pNext3, elem + 25);
	return elem[13];
}

inline __m256i load_line2_pixel_n
(
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[24]
) noexcept
{
	LoadLinePixelN (pPrev2, elem);
	LoadLinePixelN (pPrev1, elem + 4);
	LoadLinePixelN (pSrc,   elem + 8);
	LoadLinePixelN (pNext1, elem + 12);
	LoadLinePixelN (pNext2, elem + 16);
	LoadLinePixelN (pNext3, elem + 20);
	return elem[11];
}

inline __m256i load_line_pixel0
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[24]
) noexcept
{
	LoadLinePixel0 (pPrev3, elem);
	LoadLinePixel0 (pPrev2, elem + 4);
	LoadLinePixel0 (pPrev1, elem + 8);
	LoadLinePixel0 (pSrc,   elem + 12);
	LoadLinePixel0 (pNext1, elem + 16);
	LoadLinePixel0 (pNext2, elem + 20);
	LoadLinePixel0 (pNext3, elem + 24);
	return elem[8];
}

inline __m256i load_line_pixel1
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[35]
) noexcept
{
	LoadLinePixel1 (pPrev3, elem);
	LoadLinePixel1 (pPrev2, elem + 5);
	LoadLinePixel1 (pPrev1, elem + 10);
	LoadLinePixel1 (pSrc,   elem + 15);
	LoadLinePixel1 (pNext1, elem + 20);
	LoadLinePixel1 (pNext2, elem + 25);
	LoadLinePixel1 (pNext3, elem + 30);
	return elem[16];
}

inline __m256i load_line_pixel2
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[42]
) noexcept
{
	LoadLinePixel2 (pPrev3, elem);
	LoadLinePixel2 (pPrev2, elem + 6);
	LoadLinePixel2 (pPrev1, elem + 12);
	LoadLinePixel2 (pSrc,   elem + 18);
	LoadLinePixel2 (pNext1, elem + 24);
	LoadLinePixel2 (pNext2, elem + 30);
	LoadLinePixel2 (pNext3, elem + 36);
	return elem[19];
}

inline __m256i load_line_pixel
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[49]
) noexcept
{
	LoadLinePixel (pPrev3, elem);
	LoadLinePixel (pPrev2, elem + 7);
	LoadLinePixel (pPrev1, elem + 14);
	LoadLinePixel (pSrc,   elem + 21);
	LoadLinePixel (pNext1, elem + 28);
	LoadLinePixel (pNext2, elem + 35);
	LoadLinePixel (pNext3, elem + 42);
	return elem[24];
}

inline __m256i load_line_pixel_n2
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[42]
) noexcept
{
	LoadLinePixelNminus2 (pPrev3, elem);
	LoadLinePixelNminus2 (pPrev2, elem + 6);
	LoadLinePixelNminus2 (pPrev1, elem + 12);
	LoadLinePixelNminus2 (pSrc,   elem + 18);
	LoadLinePixelNminus2 (pNext1, elem + 24);
	LoadLinePixelNminus2 (pNext2, elem + 30);
	LoadLinePixelNminus2 (pNext3, elem + 36);
	return elem[21];
}

inline __m256i load_line_pixel_n1
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[35]
) noexcept
{
	LoadLinePixelNminus1 (pPrev3, elem);
	LoadLinePixelNminus1 (pPrev2, elem + 5);
	LoadLinePixelNminus1 (pPrev1, elem + 10);
	LoadLinePixelNminus1 (pSrc,   elem + 15);
	LoadLinePixelNminus1 (pNext1, elem + 20);
	LoadLinePixelNminus1 (pNext2, elem + 25);
	LoadLinePixelNminus1 (pNext3, elem + 30);
	return elem[18];
}

inline __m256i load_line_pixel_n
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	uint32_t* __restrict pNext3,
	__m256i   elem[28]
) noexcept
{
	LoadLinePixelN (pPrev3, elem);
	LoadLinePixelN (pPrev2, elem + 4);
	LoadLinePixelN (pPrev1, elem + 8);
	LoadLinePixelN (pSrc,   elem + 12);
	LoadLinePixelN (pNext1, elem + 16);
	LoadLinePixelN (pNext2, elem + 20);
	LoadLinePixelN (pNext3, elem + 24);
	return elem[15];
}

inline __m256i load_line_n2_pixel0
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	__m256i   elem[24]
) noexcept
{
	LoadLinePixel0 (pPrev3, elem);
	LoadLinePixel0 (pPrev2, elem + 4);
	LoadLinePixel0 (pPrev1, elem + 8);
	LoadLinePixel0 (pSrc,   elem + 12);
	LoadLinePixel0 (pNext1, elem + 16);
	LoadLinePixel0 (pNext2, elem + 20);
	return elem[12];
}

inline __m256i load_line_n2_pixel1
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	__m256i   elem[30]
) noexcept
{
	LoadLinePixel1 (pPrev3, elem);
	LoadLinePixel1 (pPrev2, elem + 5);
	LoadLinePixel1 (pPrev1, elem + 10);
	LoadLinePixel1 (pSrc,   elem + 15);
	LoadLinePixel1 (pNext1, elem + 20);
	LoadLinePixel1 (pNext2, elem + 25);
	return elem[16];
}

inline __m256i load_line_n2_pixel2
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	__m256i   elem[36]
) noexcept
{
	LoadLinePixel2 (pPrev3, elem);
	LoadLinePixel2 (pPrev2, elem + 6);
	LoadLinePixel2 (pPrev1, elem + 12);
	LoadLinePixel2 (pSrc,   elem + 18);
	LoadLinePixel2 (pNext1, elem + 24);
	LoadLinePixel2 (pNext2, elem + 30);
	return elem[20];
}

inline __m256i load_line_n2_pixel
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	__m256i   elem[42]
) noexcept
{
	LoadLinePixel (pPrev3, elem);
	LoadLinePixel (pPrev2, elem + 7);
	LoadLinePixel (pPrev1, elem + 14);
	LoadLinePixel (pSrc,   elem + 21);
	LoadLinePixel (pNext1, elem + 28);
	LoadLinePixel (pNext2, elem + 35);
	return elem[24];
}

inline __m256i load_line_n2_pixel_n2
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	__m256i   elem[36]
) noexcept

{
	LoadLinePixelNminus2 (pPrev3, elem);
	LoadLinePixelNminus2 (pPrev2, elem + 6);
	LoadLinePixelNminus2 (pPrev1, elem + 12);
	LoadLinePixelNminus2 (pSrc,   elem + 18);
	LoadLinePixelNminus2 (pNext1, elem + 24);
	LoadLinePixelNminus2 (pNext2, elem + 30);
	return elem[21];
}

inline __m256i load_line_n2_pixel_n1
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	__m256i   elem[30]
) noexcept
{
	LoadLinePixelNminus1 (pPrev3, elem);
	LoadLinePixelNminus1 (pPrev2, elem + 5);
	LoadLinePixelNminus1 (pPrev1, elem + 10);
	LoadLinePixelNminus1 (pSrc,   elem + 15);
	LoadLinePixelNminus1 (pNext1, elem + 20);
	LoadLinePixelNminus1 (pNext2, elem + 25);
	return elem[18];
}

inline __m256i load_line_n2_pixel_n
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	uint32_t* __restrict pNext2,
	__m256i   elem[24]
) noexcept
{
	LoadLinePixelN (pPrev3, elem);
	LoadLinePixelN (pPrev2, elem + 4);
	LoadLinePixelN (pPrev1, elem + 8);
	LoadLinePixelN (pSrc,   elem + 12);
	LoadLinePixelN (pNext1, elem + 16);
	LoadLinePixelN (pNext2, elem + 20);
	return elem[15];
}

inline __m256i load_line_n1_pixel0
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	__m256i   elem[20]
) noexcept
{
	LoadLinePixel0 (pPrev3, elem);
	LoadLinePixel0 (pPrev2, elem + 4);
	LoadLinePixel0 (pPrev1, elem + 8);
	LoadLinePixel0 (pSrc,   elem + 12);
	LoadLinePixel0 (pNext1, elem + 16);
	return elem[12];
}

inline __m256i load_line_n1_pixel1
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	__m256i   elem[25]
) noexcept
{
	LoadLinePixel1 (pPrev3, elem);
	LoadLinePixel1 (pPrev2, elem + 5);
	LoadLinePixel1 (pPrev1, elem + 10);
	LoadLinePixel1 (pSrc,   elem + 15);
	LoadLinePixel1 (pNext1, elem + 20);
	return elem[16];
}

inline __m256i load_line_n1_pixel2
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	__m256i   elem[30]
) noexcept
{
	LoadLinePixel2 (pPrev3, elem);
	LoadLinePixel2 (pPrev2, elem + 6);
	LoadLinePixel2 (pPrev1, elem + 12);
	LoadLinePixel2 (pSrc,   elem + 18);
	LoadLinePixel2 (pNext1, elem + 24);
	return elem[20];
}

inline __m256i load_line_n1_pixel
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	__m256i   elem[35]
) noexcept
{
	LoadLinePixel (pPrev3, elem);
	LoadLinePixel (pPrev2, elem + 7);
	LoadLinePixel (pPrev1, elem + 14);
	LoadLinePixel (pSrc,   elem + 21);
	LoadLinePixel (pNext1, elem + 28);
	return elem[24];
}

inline __m256i load_line_n1_pixel_n2
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	__m256i   elem[30]
) noexcept
{
	LoadLinePixelNminus2 (pPrev3, elem);
	LoadLinePixelNminus2 (pPrev2, elem + 6);
	LoadLinePixelNminus2 (pPrev1, elem + 12);
	LoadLinePixelNminus2 (pSrc,   elem + 18);
	LoadLinePixelNminus2 (pNext1, elem + 24);
	return elem[20];
}

inline __m256i load_line_n1_pixel_n1
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	__m256i   elem[25]
) noexcept
{
	LoadLinePixelNminus1 (pPrev3, elem);
	LoadLinePixelNminus1 (pPrev2, elem + 5);
	LoadLinePixelNminus1 (pPrev1, elem + 10);
	LoadLinePixelNminus1 (pSrc,   elem + 15);
	LoadLinePixelNminus1 (pNext1, elem + 20);
	return elem[18];
}

inline __m256i load_line_n1_pixel_n
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	uint32_t* __restrict pNext1,
	__m256i   elem[20]
) noexcept
{
	LoadLinePixelN (pPrev3, elem);
	LoadLinePixelN (pPrev2, elem + 4);
	LoadLinePixelN (pPrev1, elem + 8);
	LoadLinePixelN (pSrc,   elem + 12);
	LoadLinePixelN (pNext1, elem + 16);
	return elem[16];
}

inline __m256i load_line_n_pixel0
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	__m256i elem[16]
) noexcept
{
	LoadLinePixel0 (pPrev3, elem);
	LoadLinePixel0 (pPrev2, elem + 4);
	LoadLinePixel0 (pPrev1, elem + 8);
	LoadLinePixel0 (pSrc,   elem + 12);
	return elem[12];
}

inline __m256i load_line_n_pixel1
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	__m256i elem[20]
) noexcept
{
	LoadLinePixel1 (pPrev3, elem);
	LoadLinePixel1 (pPrev2, elem + 5);
	LoadLinePixel1 (pPrev1, elem + 10);
	LoadLinePixel1 (pSrc,   elem + 15);
	return elem[16];
}

inline __m256i load_line_n_pixel2
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	__m256i elem[24]
) noexcept
{
	LoadLinePixel2 (pPrev3, elem);
	LoadLinePixel2 (pPrev2, elem + 6);
	LoadLinePixel2 (pPrev1, elem + 12);
	LoadLinePixel2 (pSrc,   elem + 18);
	return elem[20];
}

inline __m256i load_line_n_pixel
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	__m256i elem[28]
) noexcept
{
	LoadLinePixel (pPrev3, elem);
	LoadLinePixel (pPrev2, elem + 7);
	LoadLinePixel (pPrev1, elem + 14);
	LoadLinePixel (pSrc,   elem + 21);
	return elem[24];
}

inline __m256i load_line_n_pixel_n2
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	__m256i elem[24]
) noexcept
{
	LoadLinePixelNminus2 (pPrev3, elem);
	LoadLinePixelNminus2 (pPrev2, elem + 6);
	LoadLinePixelNminus2 (pPrev1, elem + 12);
	LoadLinePixelNminus2 (pSrc,   elem + 18);
	return elem[21];
}

inline __m256i load_line_n_pixel_n1
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	__m256i elem[20]
) noexcept
{
	LoadLinePixelNminus1 (pPrev3, elem);
	LoadLinePixelNminus1 (pPrev2, elem + 5);
	LoadLinePixelNminus1 (pPrev1, elem + 10);
	LoadLinePixelNminus1 (pSrc,   elem + 15);
	return elem[18];
}

inline __m256i load_line_n_pixel_n
(
	uint32_t* __restrict pPrev3,
	uint32_t* __restrict pPrev2,
	uint32_t* __restrict pPrev1,
	uint32_t* __restrict pSrc,
	__m256i elem[16]
) noexcept
{
	LoadLinePixelN (pPrev3, elem);
	LoadLinePixelN (pPrev2, elem + 4);
	LoadLinePixelN (pPrev1, elem + 8);
	LoadLinePixelN (pSrc,   elem + 12);
	return elem[15];
}


inline void PartialSort_16_elem_8u (__m256i a[16]) noexcept
{
	/* median elemnet in index 7 */
	VectorSort8uPacked (a[0 ], a[1 ]);
	VectorSort8uPacked (a[3 ], a[2 ]);
	VectorSort8uPacked (a[4 ], a[5 ]);
	VectorSort8uPacked (a[7 ], a[6 ]);
	VectorSort8uPacked (a[8 ], a[9 ]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[12], a[13]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[0 ], a[2 ]);
	VectorSort8uPacked (a[6 ], a[4 ]);
	VectorSort8uPacked (a[8 ], a[10]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[1 ], a[3 ]);
	VectorSort8uPacked (a[7 ], a[5 ]);
	VectorSort8uPacked (a[9 ], a[11]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[0 ], a[1 ]);
	VectorSort8uPacked (a[2 ], a[3 ]);
	VectorSort8uPacked (a[5 ], a[4 ]);
	VectorSort8uPacked (a[7 ], a[6 ]);
	VectorSort8uPacked (a[8 ], a[9 ]);
	VectorSort8uPacked (a[10], a[11]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[0 ], a[4 ]);
	VectorSort8uPacked (a[12], a[8 ]);
	VectorSort8uPacked (a[1 ], a[5 ]);
	VectorSort8uPacked (a[13], a[9 ]);
	VectorSort8uPacked (a[2 ], a[6 ]);
	VectorSort8uPacked (a[14], a[10]);
	VectorSort8uPacked (a[3 ], a[7 ]);
	VectorSort8uPacked (a[15], a[11]);
	VectorSort8uPacked (a[0 ], a[2 ]);
	VectorSort8uPacked (a[4 ], a[6 ]);
	VectorSort8uPacked (a[10], a[8 ]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[1 ], a[3 ]);
	VectorSort8uPacked (a[5 ], a[7 ]);
	VectorSort8uPacked (a[11], a[9 ]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[0 ], a[1 ]);
	VectorSort8uPacked (a[2 ], a[3 ]);
	VectorSort8uPacked (a[4 ], a[5 ]);
	VectorSort8uPacked (a[6 ], a[7 ]);
	VectorSort8uPacked (a[9 ], a[8 ]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[0 ], a[8 ]);
	VectorSort8uPacked (a[1 ], a[9 ]);
	VectorSort8uPacked (a[2 ], a[10]);
	VectorSort8uPacked (a[3 ], a[11]);
	VectorSort8uPacked (a[4 ], a[12]);
	VectorSort8uPacked (a[5 ], a[13]);
	VectorSort8uPacked (a[6 ], a[14]);
	VectorSort8uPacked (a[7 ], a[15]);
	VectorSort8uPacked (a[0 ], a[4 ]);
	VectorSort8uPacked (a[1 ], a[5 ]);
	VectorSort8uPacked (a[2 ], a[6 ]);
	VectorSort8uPacked (a[3 ], a[7 ]);
	VectorSort8uPacked (a[4 ], a[6 ]);
	VectorSort8uPacked (a[5 ], a[7 ]);
	VectorSort8uPacked (a[6 ], a[7 ]);
}

inline void PartialSort_20_elem_8u (__m256i a[20]) noexcept
{
	/* median element in index 9 */
	VectorSort8uPacked (a[0 ], a[1 ]);
	VectorSort8uPacked (a[3 ], a[2 ]);
	VectorSort8uPacked (a[4 ], a[5 ]);
	VectorSort8uPacked (a[7 ], a[6 ]);
	VectorSort8uPacked (a[8 ], a[9 ]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[12], a[13]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[16], a[17]);
	VectorSort8uPacked (a[19], a[18]);
	VectorSort8uPacked (a[0 ], a[2 ]);
	VectorSort8uPacked (a[1 ], a[3 ]);
	VectorSort8uPacked (a[6 ], a[4 ]);
	VectorSort8uPacked (a[7 ], a[5 ]);
	VectorSort8uPacked (a[8 ], a[10]);
	VectorSort8uPacked (a[9 ], a[11]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[16], a[18]);
	VectorSort8uPacked (a[17], a[19]);
	VectorSort8uPacked (a[0 ], a[1 ]);
	VectorSort8uPacked (a[2 ], a[3 ]);
	VectorSort8uPacked (a[5 ], a[4 ]);
	VectorSort8uPacked (a[7 ], a[6 ]);
	VectorSort8uPacked (a[8 ], a[9 ]);
	VectorSort8uPacked (a[10], a[11]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[16], a[17]);
	VectorSort8uPacked (a[18], a[19]);
	VectorSort8uPacked (a[0 ], a[4 ]);
	VectorSort8uPacked (a[14], a[10]);
	VectorSort8uPacked (a[1 ], a[5 ]);
	VectorSort8uPacked (a[15], a[11]);
	VectorSort8uPacked (a[2 ], a[6 ]);
	VectorSort8uPacked (a[16], a[12]);
	VectorSort8uPacked (a[3 ], a[7 ]);
	VectorSort8uPacked (a[17], a[13]);
	VectorSort8uPacked (a[4 ], a[8 ]);
	VectorSort8uPacked (a[18], a[14]);
	VectorSort8uPacked (a[5 ], a[9 ]);
	VectorSort8uPacked (a[19], a[15]);
	VectorSort8uPacked (a[0 ], a[2 ]);
	VectorSort8uPacked (a[4 ], a[6 ]);
	VectorSort8uPacked (a[13], a[11]);
	VectorSort8uPacked (a[17], a[15]);
	VectorSort8uPacked (a[1 ], a[3 ]);
	VectorSort8uPacked (a[5 ], a[7 ]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[18], a[16]);
	VectorSort8uPacked (a[2 ], a[4 ]);
	VectorSort8uPacked (a[6 ], a[8 ]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[19], a[17]);
	VectorSort8uPacked (a[0 ], a[1 ]);
	VectorSort8uPacked (a[2 ], a[3 ]);
	VectorSort8uPacked (a[4 ], a[5 ]);
	VectorSort8uPacked (a[6 ], a[7 ]);
	VectorSort8uPacked (a[8 ], a[9 ]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[17], a[16]);
	VectorSort8uPacked (a[19], a[18]);
	VectorSort8uPacked (a[0 ], a[9 ]);
	VectorSort8uPacked (a[1 ], a[10]);
	VectorSort8uPacked (a[2 ], a[11]);
	VectorSort8uPacked (a[3 ], a[12]);
	VectorSort8uPacked (a[4 ], a[13]);
	VectorSort8uPacked (a[5 ], a[14]);
	VectorSort8uPacked (a[6 ], a[15]);
	VectorSort8uPacked (a[7 ], a[16]);
	VectorSort8uPacked (a[8 ], a[17]);
	VectorSort8uPacked (a[9 ], a[18]);
	VectorSort8uPacked (a[10], a[19]);
	VectorSort8uPacked (a[0 ], a[4 ]);
	VectorSort8uPacked (a[1 ], a[5 ]);
	VectorSort8uPacked (a[2 ], a[6 ]);
	VectorSort8uPacked (a[3 ], a[7 ]);
	VectorSort8uPacked (a[4 ], a[8 ]);
	VectorSort8uPacked (a[5 ], a[9 ]);
	VectorSort8uPacked (a[4 ], a[6 ]);
	VectorSort8uPacked (a[5 ], a[7 ]);
	VectorSort8uPacked (a[6 ], a[8 ]);
	VectorSort8uPacked (a[8 ], a[9 ]);
}

inline void PartialSort_25_elem_8u(__m256i a[25]) noexcept
{
	VectorSort8uPacked (a[0 ], a[1 ]);
	VectorSort8uPacked (a[3 ], a[4 ]);
	VectorSort8uPacked (a[2 ], a[4 ]);
	VectorSort8uPacked (a[2 ], a[3 ]);
	VectorSort8uPacked (a[6 ], a[7 ]);
	VectorSort8uPacked (a[5 ], a[7 ]);
	VectorSort8uPacked (a[5 ], a[6 ]);
	VectorSort8uPacked (a[9 ], a[10]);
	VectorSort8uPacked (a[8 ], a[10]);
	VectorSort8uPacked (a[8 ], a[9 ]);
	VectorSort8uPacked (a[12], a[13]);
	VectorSort8uPacked (a[11], a[13]);
	VectorSort8uPacked (a[11], a[12]);
	VectorSort8uPacked (a[15], a[16]);
	VectorSort8uPacked (a[14], a[16]);
	VectorSort8uPacked (a[14], a[15]);
	VectorSort8uPacked (a[18], a[19]);
	VectorSort8uPacked (a[17], a[19]);
	VectorSort8uPacked (a[17], a[18]);
	VectorSort8uPacked (a[21], a[22]);
	VectorSort8uPacked (a[20], a[22]);
	VectorSort8uPacked (a[20], a[21]);
	VectorSort8uPacked (a[23], a[24]);
	VectorSort8uPacked (a[2 ], a[5 ]);
	VectorSort8uPacked (a[3 ], a[6 ]);
	VectorSort8uPacked (a[0 ], a[6 ]);
	VectorSort8uPacked (a[0 ], a[3 ]);
	VectorSort8uPacked (a[4 ], a[7 ]);
	VectorSort8uPacked (a[1 ], a[7 ]);
	VectorSort8uPacked (a[1 ], a[4 ]);
	VectorSort8uPacked (a[11], a[14]);
	VectorSort8uPacked (a[8 ], a[14]);
	VectorSort8uPacked (a[8 ], a[11]);
	VectorSort8uPacked (a[12], a[15]);
	VectorSort8uPacked (a[9 ], a[15]);
	VectorSort8uPacked (a[9 ], a[12]);
	VectorSort8uPacked (a[13], a[16]);
	VectorSort8uPacked (a[10], a[16]);
	VectorSort8uPacked (a[10], a[13]);
	VectorSort8uPacked (a[20], a[23]);
	VectorSort8uPacked (a[17], a[23]);
	VectorSort8uPacked (a[17], a[20]);
	VectorSort8uPacked (a[21], a[24]);
	VectorSort8uPacked (a[18], a[24]);
	VectorSort8uPacked (a[18], a[21]);
	VectorSort8uPacked (a[19], a[22]);
	VectorSort8uPacked (a[9 ], a[18]);
	VectorSort8uPacked (a[0 ], a[18]);
	VectorSort8uPacked (a[8 ], a[17]);
	VectorSort8uPacked (a[0 ], a[9 ]);
	VectorSort8uPacked (a[10], a[19]);
	VectorSort8uPacked (a[1 ], a[19]);
	VectorSort8uPacked (a[1 ], a[10]);
	VectorSort8uPacked (a[11], a[20]);
	VectorSort8uPacked (a[2 ], a[20]);
	VectorSort8uPacked (a[12], a[21]);
	VectorSort8uPacked (a[2 ], a[11]);
	VectorSort8uPacked (a[3 ], a[21]);
	VectorSort8uPacked (a[3 ], a[12]);
	VectorSort8uPacked (a[13], a[22]);
	VectorSort8uPacked (a[4 ], a[22]);
	VectorSort8uPacked (a[4 ], a[13]);
	VectorSort8uPacked (a[14], a[23]);
	VectorSort8uPacked (a[5 ], a[23]);
	VectorSort8uPacked (a[5 ], a[14]);
	VectorSort8uPacked (a[15], a[24]);
	VectorSort8uPacked (a[6 ], a[24]);
	VectorSort8uPacked (a[6 ], a[15]);
	VectorSort8uPacked (a[7 ], a[16]);
	VectorSort8uPacked (a[7 ], a[19]);
	VectorSort8uPacked (a[13], a[21]);
	VectorSort8uPacked (a[15], a[23]);
	VectorSort8uPacked (a[7 ], a[13]);
	VectorSort8uPacked (a[7 ], a[15]);
	VectorSort8uPacked (a[1 ], a[9 ]);
	VectorSort8uPacked (a[3 ], a[11]);
	VectorSort8uPacked (a[5 ], a[17]);
	VectorSort8uPacked (a[11], a[17]);
	VectorSort8uPacked (a[9 ], a[17]);
	VectorSort8uPacked (a[4 ], a[10]);
	VectorSort8uPacked (a[6 ], a[12]);
	VectorSort8uPacked (a[7 ], a[14]);
	VectorSort8uPacked (a[4 ], a[6 ]);
	VectorSort8uPacked (a[4 ], a[7 ]);
	VectorSort8uPacked (a[12], a[14]);
	VectorSort8uPacked (a[10], a[14]);
	VectorSort8uPacked (a[6 ], a[7 ]);
	VectorSort8uPacked (a[10], a[12]);
	VectorSort8uPacked (a[6 ], a[10]);
	VectorSort8uPacked (a[6 ], a[17]);
	VectorSort8uPacked (a[12], a[17]);
	VectorSort8uPacked (a[7 ], a[17]);
	VectorSort8uPacked (a[7 ], a[10]);
	VectorSort8uPacked (a[12], a[18]);
	VectorSort8uPacked (a[7 ], a[12]);
	VectorSort8uPacked (a[10], a[18]);
	VectorSort8uPacked (a[12], a[20]);
	VectorSort8uPacked (a[10], a[20]);
	VectorSort8uPacked (a[10], a[12]);
}

inline void PartialSort_24_elem_8u(__m256i a[24]) noexcept
{
	VectorSort8uPacked (a[0 ], a[1 ]);
	VectorSort8uPacked (a[3 ], a[4 ]);
	VectorSort8uPacked (a[2 ], a[4 ]);
	VectorSort8uPacked (a[2 ], a[3 ]);
	VectorSort8uPacked (a[6 ], a[7 ]);
	VectorSort8uPacked (a[5 ], a[7 ]);
	VectorSort8uPacked (a[5 ], a[6 ]);
	VectorSort8uPacked (a[9 ], a[10]);
	VectorSort8uPacked (a[8 ], a[10]);
	VectorSort8uPacked (a[8 ], a[9 ]);
	VectorSort8uPacked (a[12], a[13]);
	VectorSort8uPacked (a[11], a[13]);
	VectorSort8uPacked (a[11], a[12]);
	VectorSort8uPacked (a[15], a[16]);
	VectorSort8uPacked (a[14], a[16]);
	VectorSort8uPacked (a[14], a[15]);
	VectorSort8uPacked (a[18], a[19]);
	VectorSort8uPacked (a[17], a[19]);
	VectorSort8uPacked (a[17], a[18]);
	VectorSort8uPacked (a[21], a[22]);
	VectorSort8uPacked (a[20], a[22]);
	VectorSort8uPacked (a[20], a[21]);
	VectorSort8uPacked (a[2 ], a[5 ]);
	VectorSort8uPacked (a[3 ], a[6 ]);
	VectorSort8uPacked (a[0 ], a[6 ]);
	VectorSort8uPacked (a[0 ], a[3 ]);
	VectorSort8uPacked (a[4 ], a[7 ]);
	VectorSort8uPacked (a[1 ], a[7 ]);
	VectorSort8uPacked (a[1 ], a[4 ]);
	VectorSort8uPacked (a[11], a[14]);
	VectorSort8uPacked (a[8 ], a[14]);
	VectorSort8uPacked (a[8 ], a[11]);
	VectorSort8uPacked (a[12], a[15]);
	VectorSort8uPacked (a[9 ], a[15]);
	VectorSort8uPacked (a[9 ], a[12]);
	VectorSort8uPacked (a[13], a[16]);
	VectorSort8uPacked (a[10], a[16]);
	VectorSort8uPacked (a[10], a[13]);
	VectorSort8uPacked (a[20], a[23]);
	VectorSort8uPacked (a[17], a[23]);
	VectorSort8uPacked (a[17], a[20]);
	VectorSort8uPacked (a[18], a[21]);
	VectorSort8uPacked (a[19], a[22]);
	VectorSort8uPacked (a[9 ], a[18]);
	VectorSort8uPacked (a[0 ], a[18]);
	VectorSort8uPacked (a[8 ], a[17]);
	VectorSort8uPacked (a[0 ], a[9 ]);
	VectorSort8uPacked (a[10], a[19]);
	VectorSort8uPacked (a[1 ], a[19]);
	VectorSort8uPacked (a[1 ], a[10]);
	VectorSort8uPacked (a[11], a[20]);
	VectorSort8uPacked (a[2 ], a[20]);
	VectorSort8uPacked (a[12], a[21]);
	VectorSort8uPacked (a[2 ], a[11]);
	VectorSort8uPacked (a[3 ], a[21]);
	VectorSort8uPacked (a[3 ], a[12]);
	VectorSort8uPacked (a[13], a[22]);
	VectorSort8uPacked (a[4 ], a[22]);
	VectorSort8uPacked (a[4 ], a[13]);
	VectorSort8uPacked (a[14], a[23]);
	VectorSort8uPacked (a[5 ], a[23]);
	VectorSort8uPacked (a[5 ], a[14]);
	VectorSort8uPacked (a[6 ], a[15]);
	VectorSort8uPacked (a[7 ], a[16]);
	VectorSort8uPacked (a[7 ], a[19]);
	VectorSort8uPacked (a[13], a[21]);
	VectorSort8uPacked (a[15], a[23]);
	VectorSort8uPacked (a[7 ], a[13]);
	VectorSort8uPacked (a[7 ], a[15]);
	VectorSort8uPacked (a[1 ], a[9 ]);
	VectorSort8uPacked (a[3 ], a[11]);
	VectorSort8uPacked (a[5 ], a[17]);
	VectorSort8uPacked (a[11], a[17]);
	VectorSort8uPacked (a[9 ], a[17]);
	VectorSort8uPacked (a[4 ], a[10]);
	VectorSort8uPacked (a[6 ], a[12]);
	VectorSort8uPacked (a[7 ], a[14]);
	VectorSort8uPacked (a[4 ], a[6 ]);
	VectorSort8uPacked (a[4 ], a[7 ]);
	VectorSort8uPacked (a[12], a[14]);
	VectorSort8uPacked (a[10], a[14]);
	VectorSort8uPacked (a[6 ], a[7 ]);
	VectorSort8uPacked (a[10], a[12]);
	VectorSort8uPacked (a[6 ], a[10]);
	VectorSort8uPacked (a[6 ], a[17]);
	VectorSort8uPacked (a[12], a[17]);
	VectorSort8uPacked (a[7 ], a[17]);
	VectorSort8uPacked (a[7 ], a[10]);
	VectorSort8uPacked (a[12], a[18]);
	VectorSort8uPacked (a[7 ], a[12]);
	VectorSort8uPacked (a[10], a[18]);
	VectorSort8uPacked (a[12], a[20]);
	VectorSort8uPacked (a[10], a[20]);
	VectorSort8uPacked (a[10], a[12]);
}

inline void PartialSort_28_elem_8u (__m256i a[28]) noexcept
{
	/* 14 x 2 */
	FullSort_14_elem_8u (a);
	FullSort_14_elem_8u (a + 14);

	/* merge between 2 x 14 */
	VectorSort8uPacked (a[0 ], a[27]);
	VectorSort8uPacked (a[1 ], a[26]);
	VectorSort8uPacked (a[2 ], a[25]);
	VectorSort8uPacked (a[3 ], a[24]);
	VectorSort8uPacked (a[4 ], a[23]);
	VectorSort8uPacked (a[5 ], a[22]);
	VectorSort8uPacked (a[6 ], a[21]);
	VectorSort8uPacked (a[7 ], a[20]);
	VectorSort8uPacked (a[8 ], a[19]);
	VectorSort8uPacked (a[9 ], a[18]);
	VectorSort8uPacked (a[10], a[17]);
	VectorSort8uPacked (a[11], a[16]);
	VectorSort8uPacked (a[12], a[15]);
	VectorSort8uPacked (a[13], a[14]);
	VectorSort8uPacked (a[0 ], a[14]);
	VectorSort8uPacked (a[1 ], a[15]);
	VectorSort8uPacked (a[2 ], a[16]);
	VectorSort8uPacked (a[3 ], a[17]);
	VectorSort8uPacked (a[4 ], a[18]);
	VectorSort8uPacked (a[5 ], a[19]);
	VectorSort8uPacked (a[6 ], a[20]);
	VectorSort8uPacked (a[7 ], a[21]);
	VectorSort8uPacked (a[8 ], a[22]);
	VectorSort8uPacked (a[9 ], a[23]);
	VectorSort8uPacked (a[10], a[24]);
	VectorSort8uPacked (a[11], a[25]);
	VectorSort8uPacked (a[12], a[26]);
	VectorSort8uPacked (a[13], a[27]);
	VectorSort8uPacked (a[6 ], a[9 ]);
	VectorSort8uPacked (a[7 ], a[10]);
	VectorSort8uPacked (a[8 ], a[11]);
	VectorSort8uPacked (a[9 ], a[12]);
	VectorSort8uPacked (a[10], a[13]);
	VectorSort8uPacked (a[12], a[14]);
	VectorSort8uPacked (a[10], a[12]);
	VectorSort8uPacked (a[11], a[13]);
	VectorSort8uPacked (a[12], a[13]);
}

inline void PartialSort_30_elem_8u (__m256i a[30] ) noexcept
{
	/* 15 x 2 */
	FullSort_15_elem_8u (a);
	FullSort_15_elem_8u (a + 15);

	/* merge sort 2 x 15 */
	VectorSort8uPacked (a[0 ], a[29]);
	VectorSort8uPacked (a[1 ], a[28]);
	VectorSort8uPacked (a[2 ], a[27]);
	VectorSort8uPacked (a[3 ], a[26]);
	VectorSort8uPacked (a[4 ], a[25]);
	VectorSort8uPacked (a[5 ], a[24]);
	VectorSort8uPacked (a[6 ], a[23]);
	VectorSort8uPacked (a[7 ], a[22]);
	VectorSort8uPacked (a[8 ], a[21]);
	VectorSort8uPacked (a[9 ], a[20]);
	VectorSort8uPacked (a[10], a[19]);
	VectorSort8uPacked (a[11], a[18]);
	VectorSort8uPacked (a[12], a[17]);
	VectorSort8uPacked (a[13], a[16]);
	VectorSort8uPacked (a[14], a[15]);
	VectorSort8uPacked (a[0 ], a[15]);
	VectorSort8uPacked (a[1 ], a[16]);
	VectorSort8uPacked (a[2 ], a[17]);
	VectorSort8uPacked (a[3 ], a[18]);
	VectorSort8uPacked (a[4 ], a[19]);
	VectorSort8uPacked (a[5 ], a[20]);
	VectorSort8uPacked (a[6 ], a[21]);
	VectorSort8uPacked (a[7 ], a[22]);
	VectorSort8uPacked (a[8 ], a[23]);
	VectorSort8uPacked (a[9 ], a[24]);
	VectorSort8uPacked (a[10], a[25]);
	VectorSort8uPacked (a[11], a[26]);
	VectorSort8uPacked (a[12], a[27]);
	VectorSort8uPacked (a[13], a[28]);
	VectorSort8uPacked (a[14], a[29]);
	VectorSort8uPacked (a[7 ], a[11]);
	VectorSort8uPacked (a[8 ], a[12]);
	VectorSort8uPacked (a[9 ], a[13]);
	VectorSort8uPacked (a[10], a[14]);
	VectorSort8uPacked (a[11], a[13]);
	VectorSort8uPacked (a[12], a[14]);
	VectorSort8uPacked (a[12], a[13]);
	VectorSort8uPacked (a[13], a[14]);
}

inline void PartialSort_35_elem_8u (__m256i a[35]) noexcept
{
	/* 17 + 18 */
	FullSort_17_elem_8u (a);
	FullSort_18_elem_8u (a + 17);

	/* merge sort 17 + 18 */
	VectorSort8uPacked (a[0 ], a[34]);
	VectorSort8uPacked (a[1 ], a[33]);
	VectorSort8uPacked (a[2 ], a[32]);
	VectorSort8uPacked (a[3 ], a[31]);
	VectorSort8uPacked (a[4 ], a[30]);
	VectorSort8uPacked (a[5 ], a[29]);
	VectorSort8uPacked (a[6 ], a[28]);
	VectorSort8uPacked (a[7 ], a[27]);
	VectorSort8uPacked (a[8 ], a[26]);
	VectorSort8uPacked (a[9 ], a[25]);
	VectorSort8uPacked (a[10], a[24]);
	VectorSort8uPacked (a[11], a[23]);
	VectorSort8uPacked (a[12], a[22]);
	VectorSort8uPacked (a[13], a[21]);
	VectorSort8uPacked (a[14], a[20]);
	VectorSort8uPacked (a[15], a[19]);
	VectorSort8uPacked (a[16], a[18]);
	VectorSort8uPacked (a[17], a[18]);
	VectorSort8uPacked (a[0 ], a[8 ]);
	VectorSort8uPacked (a[1 ], a[9 ]);
	VectorSort8uPacked (a[2 ], a[10]);
	VectorSort8uPacked (a[3 ], a[11]);
	VectorSort8uPacked (a[4 ], a[12]);
	VectorSort8uPacked (a[5 ], a[13]);
	VectorSort8uPacked (a[6 ], a[14]);
	VectorSort8uPacked (a[7 ], a[15]);
	VectorSort8uPacked (a[8 ], a[16]);
	VectorSort8uPacked (a[9 ], a[17]);
	VectorSort8uPacked (a[10], a[18]);
	VectorSort8uPacked (a[10], a[14]);
	VectorSort8uPacked (a[11], a[15]);
	VectorSort8uPacked (a[12], a[16]);
	VectorSort8uPacked (a[13], a[17]);
	VectorSort8uPacked (a[14], a[18]);
	VectorSort8uPacked (a[14], a[15]);
	VectorSort8uPacked (a[16], a[17]);
	VectorSort8uPacked (a[14], a[16]);
	VectorSort8uPacked (a[16], a[17]);
}

inline void PartialSort_36_elem_8u (__m256i a[36]) noexcept
{
	/* 18 x 2 */
	FullSort_18_elem_8u (a);
	FullSort_18_elem_8u (a + 18);

	/* merge sort 2 x 18 */
	VectorSort8uPacked (a[0 ], a[35]);
	VectorSort8uPacked (a[1 ], a[34]);
	VectorSort8uPacked (a[2 ], a[33]);
	VectorSort8uPacked (a[3 ], a[32]);
	VectorSort8uPacked (a[4 ], a[31]);
	VectorSort8uPacked (a[5 ], a[30]);
	VectorSort8uPacked (a[6 ], a[29]);
	VectorSort8uPacked (a[7 ], a[28]);
	VectorSort8uPacked (a[8 ], a[27]);
	VectorSort8uPacked (a[9 ], a[26]);
	VectorSort8uPacked (a[10], a[25]);
	VectorSort8uPacked (a[11], a[24]);
	VectorSort8uPacked (a[12], a[23]);
	VectorSort8uPacked (a[13], a[22]);
	VectorSort8uPacked (a[14], a[21]);
	VectorSort8uPacked (a[15], a[20]);
	VectorSort8uPacked (a[16], a[19]);
	VectorSort8uPacked (a[17], a[18]);
	VectorSort8uPacked (a[0 ], a[9 ]);
	VectorSort8uPacked (a[1 ], a[10]);
	VectorSort8uPacked (a[2 ], a[11]);
	VectorSort8uPacked (a[3 ], a[12]);
	VectorSort8uPacked (a[4 ], a[13]);
	VectorSort8uPacked (a[5 ], a[14]);
	VectorSort8uPacked (a[6 ], a[15]);
	VectorSort8uPacked (a[7 ], a[16]);
	VectorSort8uPacked (a[8 ], a[17]);
	VectorSort8uPacked (a[10], a[13]);
	VectorSort8uPacked (a[11], a[14]);
	VectorSort8uPacked (a[12], a[15]);
	VectorSort8uPacked (a[13], a[16]);
	VectorSort8uPacked (a[14], a[17]);
	VectorSort8uPacked (a[13], a[15]);
	VectorSort8uPacked (a[14], a[16]);
	VectorSort8uPacked (a[15], a[17]);
	VectorSort8uPacked (a[16], a[17]);
}

inline void PartialSort_42_elem_8u (__m256i a[42]) noexcept
{
	/* 21 x 2 */
	FullSort_21_elem_8u (a);
	FullSort_21_elem_8u (a + 21);

	/* merge sort */
	VectorSort8uPacked (a[0 ], a[41]);
	VectorSort8uPacked (a[1 ], a[40]);
	VectorSort8uPacked (a[2 ], a[39]);
	VectorSort8uPacked (a[3 ], a[38]);
	VectorSort8uPacked (a[4 ], a[37]);
	VectorSort8uPacked (a[5 ], a[36]);
	VectorSort8uPacked (a[6 ], a[35]);
	VectorSort8uPacked (a[7 ], a[34]);
	VectorSort8uPacked (a[8 ], a[33]);
	VectorSort8uPacked (a[9 ], a[32]);
	VectorSort8uPacked (a[10], a[31]);
	VectorSort8uPacked (a[11], a[30]);
	VectorSort8uPacked (a[12], a[29]);
	VectorSort8uPacked (a[13], a[28]);
	VectorSort8uPacked (a[14], a[27]);
	VectorSort8uPacked (a[15], a[26]);
	VectorSort8uPacked (a[16], a[25]);
	VectorSort8uPacked (a[17], a[24]);
	VectorSort8uPacked (a[18], a[23]);
	VectorSort8uPacked (a[19], a[22]);
	VectorSort8uPacked (a[20], a[21]);
	VectorSort8uPacked (a[0 ], a[10]);
	VectorSort8uPacked (a[1 ], a[11]);
	VectorSort8uPacked (a[2 ], a[12]);
	VectorSort8uPacked (a[3 ], a[13]);
	VectorSort8uPacked (a[4 ], a[14]);
	VectorSort8uPacked (a[5 ], a[15]);
	VectorSort8uPacked (a[6 ], a[16]);
	VectorSort8uPacked (a[7 ], a[17]);
	VectorSort8uPacked (a[8 ], a[18]);
	VectorSort8uPacked (a[9 ], a[19]);
	VectorSort8uPacked (a[10], a[20]);
	VectorSort8uPacked (a[5 ], a[15]);
	VectorSort8uPacked (a[6 ], a[16]);
	VectorSort8uPacked (a[7 ], a[17]);
	VectorSort8uPacked (a[8 ], a[18]);
	VectorSort8uPacked (a[9 ], a[19]);
	VectorSort8uPacked (a[10], a[20]);
	VectorSort8uPacked (a[15], a[18]);
	VectorSort8uPacked (a[17], a[19]);
	VectorSort8uPacked (a[18], a[20]);
	VectorSort8uPacked (a[18], a[19]);
	VectorSort8uPacked (a[19], a[20]);
}

inline void PartialSort_49_elem_8u (__m256i a[49]) noexcept
{
	/* sort lower 24 element using best NET */
	FullSort_24_elem_8u (a);
	FullSort_24_elem_8u (a + 24);

	/* merge sort */
	VectorSort8uPacked (a[0 ], a[48]);
	VectorSort8uPacked (a[1 ], a[47]);
	VectorSort8uPacked (a[2 ], a[46]);
	VectorSort8uPacked (a[3 ], a[45]);
	VectorSort8uPacked (a[4 ], a[44]);
	VectorSort8uPacked (a[5 ], a[43]);
	VectorSort8uPacked (a[6 ], a[42]);
	VectorSort8uPacked (a[7 ], a[41]);
	VectorSort8uPacked (a[8 ], a[40]);
	VectorSort8uPacked (a[9 ], a[39]);
	VectorSort8uPacked (a[10], a[38]);
	VectorSort8uPacked (a[11], a[37]);
	VectorSort8uPacked (a[12], a[36]);
	VectorSort8uPacked (a[13], a[35]);
	VectorSort8uPacked (a[14], a[34]);
	VectorSort8uPacked (a[15], a[33]);
	VectorSort8uPacked (a[16], a[32]);
	VectorSort8uPacked (a[17], a[31]);
	VectorSort8uPacked (a[18], a[30]);
	VectorSort8uPacked (a[19], a[29]);
	VectorSort8uPacked (a[20], a[28]);
	VectorSort8uPacked (a[21], a[27]);
	VectorSort8uPacked (a[22], a[26]);
	VectorSort8uPacked (a[23], a[25]);
	VectorSort8uPacked (a[24], a[25]);
	VectorSort8uPacked (a[23], a[24]);
	VectorSort8uPacked (a[0],  a[12]);
	VectorSort8uPacked (a[1],  a[13]);
	VectorSort8uPacked (a[2],  a[14]);
	VectorSort8uPacked (a[3],  a[15]);
	VectorSort8uPacked (a[4],  a[16]);
	VectorSort8uPacked (a[5],  a[17]);
	VectorSort8uPacked (a[6],  a[18]);
	VectorSort8uPacked (a[7],  a[19]);
	VectorSort8uPacked (a[8],  a[20]);
	VectorSort8uPacked (a[9],  a[21]);
	VectorSort8uPacked (a[10], a[22]);
	VectorSort8uPacked (a[11], a[23]);
	VectorSort8uPacked (a[12], a[24]);
	VectorSort8uPacked (a[13], a[19]);
	VectorSort8uPacked (a[14], a[20]);
	VectorSort8uPacked (a[15], a[21]);
	VectorSort8uPacked (a[16], a[22]);
	VectorSort8uPacked (a[17], a[23]);
	VectorSort8uPacked (a[18], a[24]);
	VectorSort8uPacked (a[16], a[21]);
	VectorSort8uPacked (a[17], a[22]);
	VectorSort8uPacked (a[18], a[23]);
	VectorSort8uPacked (a[19], a[24]);
	VectorSort8uPacked (a[17], a[21]);
	VectorSort8uPacked (a[18], a[22]);
	VectorSort8uPacked (a[19], a[23]);
	VectorSort8uPacked (a[20], a[24]);
	VectorSort8uPacked (a[21], a[23]);
	VectorSort8uPacked (a[22], a[24]);
	VectorSort8uPacked (a[23], a[24]);
}

/*
	make median filter with kernel 7x7 from packed format - BGRA444_8u by AVX2 instructions set:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

	LSB                            MSB
	+-------------------------------+
	| B | G | R | A | B | G | R | A | ...
	+-------------------------------+

*/
#pragma comment(linker, "/STACK:4194304")
bool AVX2::Median::median_filter_7x7_BGRA_4444_8u
(
	uint32_t* __restrict pInImage,
	uint32_t* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	const A_long& channelMask /* 0x00FFFFFF <- BGRa */
) noexcept
{
//	if (sizeY < 7 || sizeX < 40)
//		return Scalar::scalar_median_filter_7x7_BGRA_4444_8u(pInImage, pOutImage, sizeY, sizeX, linePitch);

	constexpr A_long startOffset = 3;
	constexpr A_long pixelsInVector{ static_cast<A_long>(sizeof(__m256i) / sizeof(uint32_t)) };
	constexpr A_long startPosition{ pixelsInVector * startOffset };

	A_long i, j;
	const A_long vectorLoadsInLine = sizeX / pixelsInVector;
	const A_long vectorizedLineSize = vectorLoadsInLine * pixelsInVector;
	const A_long lastPixelsInLine = sizeX - vectorizedLineSize;
	const A_long lastIdx = lastPixelsInLine - startOffset;

	const A_long shortSizeY { sizeY - startOffset};
	const A_long shortSizeX { sizeX - pixelsInVector * 3 };

	const __m256i rgbMaskVector = _mm256_setr_epi32
	(
		channelMask, /* mask A component for 1 pixel */
		channelMask, /* mask A component for 2 pixel */
		channelMask, /* mask A component for 3 pixel */
		channelMask, /* mask A component for 4 pixel */
		channelMask, /* mask A component for 5 pixel */
		channelMask, /* mask A component for 6 pixel */
		channelMask, /* mask A component for 7 pixel */
		channelMask  /* mask A component for 8 pixel */
	);

#ifdef _DEBUG
	__m256i vecData[49]{};
#else
	CACHE_ALIGN __m256i vecData[49];
#endif

	/* PROCESS FIRST LINE IN FRAME */
	{
		uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage);
		uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch);
		uint32_t* __restrict pSrcVecNextLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 2);
		uint32_t* __restrict pSrcVecNextLine3 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 3);
		__m256i*  __restrict pSrcVecDstLine   = reinterpret_cast<__m256i*  __restrict>(pOutImage);

		/* process pixel 0 */
		const __m256i pix0 = load_line0_pixel0 (pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, pSrcVecNextLine3, vecData);
		PartialSort_16_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix0, vecData[7], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 1 */
		const __m256i pix1 = load_line0_pixel1 (pSrcVecCurrLine  + pixelsInVector, 
			                                    pSrcVecNextLine1 + pixelsInVector,
			                                    pSrcVecNextLine2 + pixelsInVector,
			                                    pSrcVecNextLine3 + pixelsInVector,
			                                    vecData);
		PartialSort_20_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix1, vecData[9], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 2 */
		const __m256i pix2 = load_line0_pixel2 (pSrcVecCurrLine  + pixelsInVector * 2,
			                                    pSrcVecNextLine1 + pixelsInVector * 2,
			                                    pSrcVecNextLine2 + pixelsInVector * 2,
			                                    pSrcVecNextLine3 + pixelsInVector * 2,
			                                    vecData);
		PartialSort_24_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix2, vecData[11], rgbMaskVector);
		pSrcVecDstLine++;

		/* process rest of pixesl */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i pix = load_line0_pixel (pSrcVecCurrLine  + i,
				                                  pSrcVecNextLine1 + i,
				                                  pSrcVecNextLine2 + i,
				                                  pSrcVecNextLine3 + i,
				                                  vecData);
			PartialSort_28_elem_8u (vecData);
			StoreByMask8u(pSrcVecDstLine, pix2, vecData[13], rgbMaskVector);
			pSrcVecDstLine++;
		}
		
		/* process pixel N - 2 */
		const __m256i pixn2 = load_line0_pixel_n2 (pSrcVecCurrLine  + i, 
			                                       pSrcVecNextLine1 + i,
			                                       pSrcVecNextLine2 + i,
			                                       pSrcVecNextLine3 + i,
			                                       vecData);
		PartialSort_24_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn2, vecData[11], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N - 1 */
		const __m256i pixn1 = load_line0_pixel_n1 (pSrcVecCurrLine  + i,
			                                       pSrcVecNextLine1 + i,
			                                       pSrcVecNextLine2 + i,
			                                       pSrcVecNextLine3 + i,
			                                       vecData);
		PartialSort_20_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn1, vecData[9], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N */
		const __m256i pixn = load_line0_pixel_n (pSrcVecCurrLine  + i,
			                                     pSrcVecNextLine1 + i,
			                                     pSrcVecNextLine2 + i,
			                                     pSrcVecNextLine3 + i,
			                                     vecData);
		PartialSort_16_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn, vecData[7], rgbMaskVector);
	}
	
	/* PROCESS SECOND LINE IN FRAME */
	{
		uint32_t* __restrict pSrcVecPrevLine  = reinterpret_cast<uint32_t* __restrict>(pInImage);
		uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch);
		uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 2);
		uint32_t* __restrict pSrcVecNextLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 3);
		uint32_t* __restrict pSrcVecNextLine3 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 4);
		__m256i*  __restrict pSrcVecDstLine = reinterpret_cast<__m256i*    __restrict>(pOutImage+ dstLinePitch);

		/* process pixel 0 */
		const __m256i pix0 = load_line1_pixel0 (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, pSrcVecNextLine3, vecData);
		PartialSort_20_elem_8u(vecData);
		StoreByMask8u(pSrcVecDstLine, pix0, vecData[9], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 1 */
		const __m256i pix1 = load_line1_pixel1 (pSrcVecPrevLine  + pixelsInVector,
			                                    pSrcVecCurrLine  + pixelsInVector,
			                                    pSrcVecNextLine1 + pixelsInVector,
			                                    pSrcVecNextLine2 + pixelsInVector,
			                                    pSrcVecNextLine3 + pixelsInVector,
			                                    vecData);
		PartialSort_25_elem_8u(vecData);
		StoreByMask8u(pSrcVecDstLine, pix1, vecData[12], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 2 */
		const __m256i pix2 = load_line1_pixel2 (pSrcVecPrevLine  + pixelsInVector * 2,
			                                    pSrcVecCurrLine  + pixelsInVector * 2,
			                                    pSrcVecNextLine1 + pixelsInVector * 2,
			                                    pSrcVecNextLine2 + pixelsInVector * 2,
			                                    pSrcVecNextLine3 + pixelsInVector * 2,
			                                    vecData);
		PartialSort_30_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix2, vecData[14], rgbMaskVector);
		pSrcVecDstLine++;

		/* process rest of pixesl */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i pix = load_line1_pixel (pSrcVecPrevLine  + i,
				                                  pSrcVecCurrLine  + i,
				                                  pSrcVecNextLine1 + i,
				                                  pSrcVecNextLine2 + i,
				                                  pSrcVecNextLine3 + i,
				                                  vecData);
			PartialSort_35_elem_8u (vecData);
			StoreByMask8u(pSrcVecDstLine, pix, vecData[17], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process pixel N - 2 */
		const __m256i pixn2 = load_line1_pixel_n2 (pSrcVecPrevLine  + i,
			                                       pSrcVecCurrLine  + i,
			                                       pSrcVecNextLine1 + i,
			                                       pSrcVecNextLine2 + i,
			                                       pSrcVecNextLine3 + i,
			                                       vecData);
		PartialSort_30_elem_8u (vecData);
		StoreByMask8u(pSrcVecDstLine, pixn2, vecData[14], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N - 1 */
		const __m256i pixn1 = load_line1_pixel_n1 (pSrcVecPrevLine  + i,
												   pSrcVecCurrLine  + i,
												   pSrcVecNextLine1 + i,
												   pSrcVecNextLine2 + i,
												   pSrcVecNextLine3 + i,
												   vecData);

		PartialSort_25_elem_8u (vecData);
		StoreByMask8u(pSrcVecDstLine, pixn2, vecData[12], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N */
		const __m256i pixn = load_line1_pixel_n (pSrcVecPrevLine  + i,
			                                     pSrcVecCurrLine  + i,
			                                     pSrcVecNextLine1 + i,
			                                     pSrcVecNextLine2 + i,
			                                     pSrcVecNextLine3 + i,
			                                     vecData);
		PartialSort_20_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn, vecData[9], rgbMaskVector);
	}

	/* PROCESS THIRD LINE IN FRAME */
	{
		uint32_t* __restrict pSrcVecPrevLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage);
		uint32_t* __restrict pSrcVecPrevLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch);
		uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 2);
		uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 3);
		uint32_t* __restrict pSrcVecNextLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 4);
		uint32_t* __restrict pSrcVecNextLine3 = reinterpret_cast<uint32_t* __restrict>(pInImage + srcLinePitch * 5);
		__m256i*  __restrict pSrcVecDstLine = reinterpret_cast<__m256i*    __restrict>(pOutImage+ dstLinePitch * 2);

		/* process first pixel */
		const __m256i pix0 = load_line2_pixel0 (pSrcVecPrevLine2, 
			                                    pSrcVecPrevLine1,
			                                    pSrcVecCurrLine,
			                                    pSrcVecNextLine1,
			                                    pSrcVecNextLine2,
			                                    pSrcVecNextLine3,
			                                    vecData);
		PartialSort_24_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix0, vecData[11], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 1 */
		const __m256i pix1 = load_line2_pixel1 (pSrcVecPrevLine2 + pixelsInVector,
            			                        pSrcVecPrevLine1 + pixelsInVector,
			                                    pSrcVecCurrLine  + pixelsInVector,
			                                    pSrcVecNextLine1 + pixelsInVector,
			                                    pSrcVecNextLine2 + pixelsInVector,
			                                    pSrcVecNextLine3 + pixelsInVector,
			                                    vecData);
		PartialSort_30_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix1, vecData[14], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 2 */
		const __m256i pix2 = load_line2_pixel2 (pSrcVecPrevLine2 + pixelsInVector * 2,
			                                    pSrcVecPrevLine1 + pixelsInVector * 2,
			                                    pSrcVecCurrLine  + pixelsInVector * 2,
			                                    pSrcVecNextLine1 + pixelsInVector * 2,
			                                    pSrcVecNextLine2 + pixelsInVector * 2,
			                                    pSrcVecNextLine3 + pixelsInVector * 2,
			                                    vecData);
		PartialSort_36_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix2, vecData[17], rgbMaskVector);
		pSrcVecDstLine++;

		/* process rest of pixesl */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i pix = load_line2_pixel (pSrcVecPrevLine2 + i,
				                                  pSrcVecPrevLine1 + i,
				                                  pSrcVecCurrLine  + i,
				                                  pSrcVecNextLine1 + i,
				                                  pSrcVecNextLine2 + i,
				                                  pSrcVecNextLine3 + i,
				                                  vecData);
			PartialSort_42_elem_8u (vecData);
			StoreByMask8u (pSrcVecDstLine, pix, vecData[20], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process pixel N - 2 */
		const __m256i pixn2 = load_line2_pixel_n2 (pSrcVecPrevLine2 + i,
 			                                       pSrcVecPrevLine1 + i,
			                                       pSrcVecCurrLine  + i,
			                                       pSrcVecNextLine1 + i,
			                                       pSrcVecNextLine2 + i,
			                                       pSrcVecNextLine3 + i,
			                                       vecData);
		PartialSort_36_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn2, vecData[17], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N - 1 */
		const __m256i pixn1 = load_line2_pixel_n1 (pSrcVecPrevLine2 + i,
			                                       pSrcVecPrevLine1 + i,
			                                       pSrcVecCurrLine  + i,
			                                       pSrcVecNextLine1 + i,
			                                       pSrcVecNextLine2 + i,
			                                       pSrcVecNextLine3 + i,
			                                       vecData);
		PartialSort_30_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn1, vecData[14], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N */
		const __m256i pixn = load_line2_pixel_n (pSrcVecPrevLine2 + i,
			                                     pSrcVecPrevLine1 + i,
			                                     pSrcVecCurrLine  + i,
			                                     pSrcVecNextLine1 + i,
			                                     pSrcVecNextLine2 + i,
			                                     pSrcVecNextLine3 + i,
			                                     vecData);
		PartialSort_24_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn, vecData[11], rgbMaskVector);
	}

	/* PROCESS REST OF LINES IN FRAME */
	for (j = startOffset; j < shortSizeY; j++)
	{
			uint32_t* __restrict pSrcVecPrevLine3 = reinterpret_cast<uint32_t* __restrict>(pInImage  + (j - 3) * srcLinePitch);
			uint32_t* __restrict pSrcVecPrevLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage  + (j - 2) * srcLinePitch);
			uint32_t* __restrict pSrcVecPrevLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage  + (j - 1) * srcLinePitch);
			uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage  +  j      * srcLinePitch);
			uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage  + (j + 1) * srcLinePitch);
			uint32_t* __restrict pSrcVecNextLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage  + (j + 2) * srcLinePitch);
			uint32_t* __restrict pSrcVecNextLine3 = reinterpret_cast<uint32_t* __restrict>(pInImage  + (j + 3) * srcLinePitch);
			__m256i*  __restrict pSrcVecDstLine   = reinterpret_cast<__m256i*  __restrict>(pOutImage +  j      * dstLinePitch);

			/* process first pixel */
			const __m256i pix0 = load_line_pixel0 (pSrcVecPrevLine3,
				                                   pSrcVecPrevLine2,
				                                   pSrcVecPrevLine1,
				                                   pSrcVecCurrLine,
				                                   pSrcVecNextLine1,
				                                   pSrcVecNextLine2,
				                                   pSrcVecNextLine3,
				                                   vecData);
			PartialSort_28_elem_8u (vecData);
			StoreByMask8u (pSrcVecDstLine, pix0, vecData[13], rgbMaskVector);
			pSrcVecDstLine++;

			/* process pixel 1 */
			const __m256i pix1 = load_line_pixel1 (pSrcVecPrevLine3 + pixelsInVector,
				                                   pSrcVecPrevLine2 + pixelsInVector,
				                                   pSrcVecPrevLine1 + pixelsInVector,
				                                   pSrcVecCurrLine  + pixelsInVector,
				                                   pSrcVecNextLine1 + pixelsInVector,
				                                   pSrcVecNextLine2 + pixelsInVector,
				                                   pSrcVecNextLine3 + pixelsInVector,
				                                   vecData);
			PartialSort_35_elem_8u (vecData);
			StoreByMask8u(pSrcVecDstLine, pix1, vecData[17], rgbMaskVector);
			pSrcVecDstLine++;

			/* process pixel 2 */
			const __m256i pix2 = load_line_pixel2 (pSrcVecPrevLine3 + pixelsInVector * 2,
                    				               pSrcVecPrevLine2 + pixelsInVector * 2,
				                                   pSrcVecPrevLine1 + pixelsInVector * 2,
				                                   pSrcVecCurrLine  + pixelsInVector * 2,
				                                   pSrcVecNextLine1 + pixelsInVector * 2,
				                                   pSrcVecNextLine2 + pixelsInVector * 2,
				                                   pSrcVecNextLine3 + pixelsInVector * 2,
				                                   vecData);
			PartialSort_42_elem_8u (vecData);
			StoreByMask8u(pSrcVecDstLine, pix2, vecData[20], rgbMaskVector);
			pSrcVecDstLine++;

			/* process rest of pixesl */
			for (i = startPosition; i < shortSizeX; i += pixelsInVector)
			{
				const __m256i pix = load_line_pixel (pSrcVecPrevLine3 + i,
					                                 pSrcVecPrevLine2 + i,
					                                 pSrcVecPrevLine1 + i,
					                                 pSrcVecCurrLine  + i,
					                                 pSrcVecNextLine1 + i,
					                                 pSrcVecNextLine2 + i,
					                                 pSrcVecNextLine3 + i,
					                                 vecData);
				PartialSort_49_elem_8u (vecData);
				StoreByMask8u (pSrcVecDstLine, pix, vecData[24], rgbMaskVector);
				pSrcVecDstLine++;
			}

			/* process pixel N - 2 */
			const __m256i pixn2 = load_line_pixel_n2 (pSrcVecPrevLine3 + i,
				                                      pSrcVecPrevLine2 + i,
				                                      pSrcVecPrevLine1 + i,
				                                      pSrcVecCurrLine  + i,
				                                      pSrcVecNextLine1 + i,
				                                      pSrcVecNextLine2 + i,
				                                      pSrcVecNextLine3 + i,
				                                      vecData);
			PartialSort_42_elem_8u (vecData);
			StoreByMask8u (pSrcVecDstLine, pixn2, vecData[20], rgbMaskVector);
			pSrcVecDstLine++;
			i += pixelsInVector;

			/* process pixel N - 1 */
			const __m256i pixn1 = load_line_pixel_n1 (pSrcVecPrevLine3 + i,
				                                      pSrcVecPrevLine2 + i,
				                                      pSrcVecPrevLine1 + i,
				                                      pSrcVecCurrLine  + i,
				                                      pSrcVecNextLine1 + i,
				                                      pSrcVecNextLine2 + i,
				                                      pSrcVecNextLine3 + i,
				                                      vecData);
			PartialSort_35_elem_8u (vecData);
			StoreByMask8u (pSrcVecDstLine, pixn2, vecData[17], rgbMaskVector);
			pSrcVecDstLine++;
			i += pixelsInVector;

			/* process pixel N */
			const __m256i pixn = load_line_pixel_n (pSrcVecPrevLine3 + i,
				                                    pSrcVecPrevLine2 + i,
				                                    pSrcVecPrevLine1 + i,
				                                    pSrcVecCurrLine  + i,
				                                    pSrcVecNextLine1 + i,
				                                    pSrcVecNextLine2 + i,
				                                    pSrcVecNextLine3 + i,
				                                    vecData);
			PartialSort_28_elem_8u (vecData);
			StoreByMask8u (pSrcVecDstLine, pixn2, vecData[13], rgbMaskVector);
	}

	/* PROCESS LINE 'N MINUS 2' IN FRAME */
	{
		uint32_t* __restrict pSrcVecPrevLine3 = reinterpret_cast<uint32_t*  __restrict>(pInImage  + (j - 3) * srcLinePitch);
		uint32_t* __restrict pSrcVecPrevLine2 = reinterpret_cast<uint32_t*  __restrict>(pInImage  + (j - 2) * srcLinePitch);
		uint32_t* __restrict pSrcVecPrevLine1 = reinterpret_cast<uint32_t*  __restrict>(pInImage  + (j - 1) * srcLinePitch);
		uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t*  __restrict>(pInImage  +  j      * srcLinePitch);
		uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t*  __restrict>(pInImage  + (j + 1) * srcLinePitch);
		uint32_t* __restrict pSrcVecNextLine2 = reinterpret_cast<uint32_t*  __restrict>(pInImage  + (j + 2) * srcLinePitch);
		__m256i*  __restrict pSrcVecDstLine   = reinterpret_cast<__m256i *  __restrict>(pOutImage +  j      * dstLinePitch);

		/* process first pixel */
		const __m256i pix0 = load_line_n2_pixel0 (pSrcVecPrevLine3,
			                                      pSrcVecPrevLine2,
			                                      pSrcVecPrevLine1,
			                                      pSrcVecCurrLine,
			                                      pSrcVecNextLine1,
			                                      pSrcVecNextLine2,
			                                      vecData);
		PartialSort_24_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix0, vecData[11], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 1 */
		const __m256i pix1 = load_line_n2_pixel1 (pSrcVecPrevLine3 + pixelsInVector,
			                                      pSrcVecPrevLine2 + pixelsInVector,
			                                      pSrcVecPrevLine1 + pixelsInVector,
			                                      pSrcVecCurrLine  + pixelsInVector,
			                                      pSrcVecNextLine1 + pixelsInVector,
			                                      pSrcVecNextLine2 + pixelsInVector,
			                                      vecData);
		PartialSort_30_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix1, vecData[14], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 2 */
		const __m256i pix2 = load_line_n2_pixel2 (pSrcVecPrevLine3 + pixelsInVector * 2,
			                                      pSrcVecPrevLine2 + pixelsInVector * 2,
			                                      pSrcVecPrevLine1 + pixelsInVector * 2,
			                                      pSrcVecCurrLine  + pixelsInVector * 2,
			                                      pSrcVecNextLine1 + pixelsInVector * 2,
			                                      pSrcVecNextLine2 + pixelsInVector * 2,
			                                      vecData);

		PartialSort_36_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix2, vecData[17], rgbMaskVector);
		pSrcVecDstLine++;

		/* process rest of pixesl */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i pix = load_line_n2_pixel (pSrcVecPrevLine3 + i,
				                                    pSrcVecPrevLine2 + i,
				                                    pSrcVecPrevLine1 + i,
				                                    pSrcVecCurrLine  + i,
				                                    pSrcVecNextLine1 + i,
				                                    pSrcVecNextLine2 + i,
				                                    vecData);
			PartialSort_42_elem_8u (vecData);
			StoreByMask8u (pSrcVecDstLine, pix, vecData[20], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process pixel N - 2 */
		const __m256i pixn2 = load_line_n2_pixel_n2 (pSrcVecPrevLine3 + i,
			                                         pSrcVecPrevLine2 + i,
			                                         pSrcVecPrevLine1 + i,
			                                         pSrcVecCurrLine  + i,
			                                         pSrcVecNextLine1 + i,
			                                         pSrcVecNextLine2 + i,
			                                         vecData);

		PartialSort_36_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn2, vecData[17], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N - 1 */
		const __m256i pixn1 = load_line_n2_pixel_n1 (pSrcVecPrevLine3 + i,
			                                         pSrcVecPrevLine2 + i,
			                                         pSrcVecPrevLine1 + i,
			                                         pSrcVecCurrLine  + i,
			                                         pSrcVecNextLine1 + i,
			                                         pSrcVecNextLine2 + i,
			                                         vecData);

		PartialSort_30_elem_8u (vecData);
		StoreByMask8u(pSrcVecDstLine, pixn1, vecData[14], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N */
		const __m256i pixn = load_line_n2_pixel_n (pSrcVecPrevLine3 + i,
			                                       pSrcVecPrevLine2 + i,
			                                       pSrcVecPrevLine1 + i,
			                                       pSrcVecCurrLine  + i,
			                                       pSrcVecNextLine1 + i,
			                                       pSrcVecNextLine2 + i,
			                                       vecData);

		PartialSort_24_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn, vecData[11], rgbMaskVector);
	}
	
	/* PROCESS LINE 'N MINUS 1' IN FRAME */
	{
		j += 1;
		uint32_t* __restrict pSrcVecPrevLine3 = reinterpret_cast<uint32_t* __restrict>(pInImage  + (j - 3) * srcLinePitch);
		uint32_t* __restrict pSrcVecPrevLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage  + (j - 2) * srcLinePitch);
		uint32_t* __restrict pSrcVecPrevLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage  + (j - 1) * srcLinePitch);
		uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage  +  j      * srcLinePitch);
		uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage  + (j + 1) * srcLinePitch);
		__m256i*  __restrict pSrcVecDstLine   = reinterpret_cast<__m256i *  __restrict>(pOutImage + j      * dstLinePitch);

		/* process first pixel */
		const __m256i pix0 = load_line_n1_pixel0 (pSrcVecPrevLine3,
			                                      pSrcVecPrevLine2,
			                                      pSrcVecPrevLine1,
			                                      pSrcVecCurrLine,
			                                      pSrcVecNextLine1,
			                                      vecData);
		PartialSort_20_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix0, vecData[9], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 1 */
		const __m256i pix1 = load_line_n1_pixel1 (pSrcVecPrevLine3 + pixelsInVector,
			                                      pSrcVecPrevLine2 + pixelsInVector,
			                                      pSrcVecPrevLine1 + pixelsInVector,
			                                      pSrcVecCurrLine  + pixelsInVector,
			                                      pSrcVecNextLine1 + pixelsInVector,
			                                      vecData);
		PartialSort_25_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix1, vecData[12], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 2 */
		const __m256i pix2 = load_line_n1_pixel2 (pSrcVecPrevLine3 + pixelsInVector * 2,
			                                      pSrcVecPrevLine2 + pixelsInVector * 2,
			                                      pSrcVecPrevLine1 + pixelsInVector * 2,
			                                      pSrcVecCurrLine  + pixelsInVector * 2,
			                                      pSrcVecNextLine1 + pixelsInVector * 2,
			                                      vecData);
		PartialSort_30_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix2, vecData[14], rgbMaskVector);
		pSrcVecDstLine++;

		/* process rest of pixesl */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i pix = load_line_n1_pixel (pSrcVecPrevLine3 + i,
				                                    pSrcVecPrevLine2 + i,
				                                    pSrcVecPrevLine1 + i,
				                                    pSrcVecCurrLine  + i,
				                                    pSrcVecNextLine1 + i,
				                                    vecData);
			PartialSort_35_elem_8u (vecData);
			StoreByMask8u (pSrcVecDstLine, pix, vecData[17], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process pixel N - 2 */
		const __m256i pixn2 = load_line_n1_pixel_n2 (pSrcVecPrevLine3 + i,
			                                         pSrcVecPrevLine2 + i,
			                                         pSrcVecPrevLine1 + i,
			                                         pSrcVecCurrLine  + i,
			                                         pSrcVecNextLine1 + i,
			                                         vecData);
		PartialSort_30_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn2, vecData[14], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N - 1 */
		const __m256i pixn1 = load_line_n1_pixel_n1 (pSrcVecPrevLine3 + i,
			                                         pSrcVecPrevLine2 + i,
			                                         pSrcVecPrevLine1 + i,
			                                         pSrcVecCurrLine  + i,
			                                         pSrcVecNextLine1 + i,
			                                         vecData);
		PartialSort_25_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn1, vecData[12], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N */
		const __m256i pixn = load_line_n1_pixel_n (pSrcVecPrevLine3 + i,
			                                       pSrcVecPrevLine2 + i,
			                                       pSrcVecPrevLine1 + i,
			                                       pSrcVecCurrLine  + i,
			                                       pSrcVecNextLine1 + i,
			                                       vecData);
		PartialSort_20_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn1, vecData[9], rgbMaskVector);
		pSrcVecDstLine++;
	}
	
	/* PROCESS LAST LINE IN FRAME */
	{
		j += 1;
		uint32_t* __restrict pSrcVecPrevLine3 = reinterpret_cast<uint32_t* __restrict>(pInImage   + (j - 3) * srcLinePitch);
		uint32_t* __restrict pSrcVecPrevLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage   + (j - 2) * srcLinePitch);
		uint32_t* __restrict pSrcVecPrevLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage   + (j - 1) * srcLinePitch);
		uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage   +  j      * srcLinePitch);
		__m256i*  __restrict pSrcVecDstLine   = reinterpret_cast<__m256i *  __restrict>(pOutImage +  j      * dstLinePitch);

		/* process first pixel */
		const __m256i pix0 = load_line_n_pixel0 (pSrcVecPrevLine3,
			                                     pSrcVecPrevLine2,
			                                     pSrcVecPrevLine1,
			                                     pSrcVecCurrLine,
			                                     vecData);
		PartialSort_16_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix0, vecData[7], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 1 */
		const __m256i pix1 = load_line_n_pixel1 (pSrcVecPrevLine3 + pixelsInVector,
			                                     pSrcVecPrevLine2 + pixelsInVector,
			                                     pSrcVecPrevLine1 + pixelsInVector,
			                                     pSrcVecCurrLine  + pixelsInVector,
			                                     vecData);
		PartialSort_20_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pix1, vecData[9], rgbMaskVector);
		pSrcVecDstLine++;

		/* process pixel 2 */
		const __m256i pix2 = load_line_n_pixel2 (pSrcVecPrevLine3 + pixelsInVector * 2,
			                                     pSrcVecPrevLine2 + pixelsInVector * 2,
			                                     pSrcVecPrevLine1 + pixelsInVector * 2,
			                                     pSrcVecCurrLine  + pixelsInVector * 2,
			                                     vecData);
		PartialSort_24_elem_8u(vecData);
		StoreByMask8u(pSrcVecDstLine, pix1, vecData[11], rgbMaskVector);
		pSrcVecDstLine++;

		/* process rest of pixesl */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i pix = load_line_n_pixel (pSrcVecPrevLine3 + i,
				                                   pSrcVecPrevLine2 + i,
				                                   pSrcVecPrevLine1 + i,
				                                   pSrcVecCurrLine  + i,
				                                   vecData);
			PartialSort_28_elem_8u (vecData);
			StoreByMask8u(pSrcVecDstLine, pix, vecData[13], rgbMaskVector);
			pSrcVecDstLine++;
		}

		/* process pixel N - 2 */
		const __m256i pixn2 = load_line_n_pixel_n2 (pSrcVecPrevLine3 + i,
			                                        pSrcVecPrevLine2 + i,
			                                        pSrcVecPrevLine1 + i,
			                                        pSrcVecCurrLine  + i,
			                                        vecData);
		PartialSort_24_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn2, vecData[11], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N - 1 */
		const __m256i pixn1 = load_line_n_pixel_n1 (pSrcVecPrevLine3 + i,
			                                        pSrcVecPrevLine2 + i,
			                                        pSrcVecPrevLine1 + i,
			                                        pSrcVecCurrLine  + i,
			                                        vecData);
		PartialSort_20_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn2, vecData[9], rgbMaskVector);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process pixel N */
		const __m256i pixn = load_line_n_pixel_n (pSrcVecPrevLine3 + i,
			                                      pSrcVecPrevLine2 + i,
			                                      pSrcVecPrevLine1 + i,
			                                      pSrcVecCurrLine  + i,
			                                      vecData);
		PartialSort_16_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, pixn2, vecData[7], rgbMaskVector);
	}

	return true;
}