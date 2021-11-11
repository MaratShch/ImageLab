#include "Avx2Median.hpp"
#include "Avx2MedianInternal.hpp"


inline void LoadLinePixel0 (uint32_t* __restrict pSrc, __m256i elemLine[3]) noexcept
{
	//  | X  0  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
}

inline void LoadLinePixel1 (uint32_t* __restrict pSrc, __m256i elemLine[4]) noexcept
{
	//  | 0  X  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
}

inline void LoadLinePixel (uint32_t* __restrict pSrc, __m256i elemLine[5]) noexcept
{
	// 0  0  0
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
	elemLine[4] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 2));
}

inline void LoadLinePixelBeforeLast (uint32_t* __restrict pSrc, __m256i elemLine[4]) noexcept
{
	// 0  X  0 |
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
	elemLine[3] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc + 1));
}

inline void LoadLinePixelLast (uint32_t* __restrict pSrc, __m256i elemLine[3]) noexcept
{
	//  0  0  X |
	elemLine[0] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 2));
	elemLine[1] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc - 1));
	elemLine[2] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(pSrc));
}

inline __m256i LoadFirstLineWindowPixel0 (uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[9]) noexcept
{
							  		   //  +-------- 
	LoadLinePixel0 (pSrc,   elem);     //  | X  0  0
	LoadLinePixel0 (pNext1, elem + 3); //  | 0  0  0
	LoadLinePixel0 (pNext2, elem + 6); //  | 0  0  0
	return elem[0];                     
}

inline __m256i LoadFirstLineWindowPixel1 (uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[12]) noexcept
{
									   //  +----------- 
	LoadLinePixel1 (pSrc, elem);       //  | 0  X  0  0
	LoadLinePixel1 (pNext1, elem + 4); //  | 0  0  0  0
	LoadLinePixel1 (pNext2, elem + 8); //  | 0  0  0  0
	return elem[1];
}

inline __m256i LoadFirstLineWindowPixel (uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[15]) noexcept
{
									   //  +----------- 
	LoadLinePixel (pSrc, elem);        //  | 0  0  X  0  0
	LoadLinePixel (pNext1, elem + 5);  //  | 0  0  0  0  0
	LoadLinePixel (pNext2, elem + 10); //  | 0  0  0  0  0
	return elem[2];
}

inline __m256i LoadFirstLineWindowPixelBeforeLast (uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[12]) noexcept
{
												//  +----------- 
	LoadLinePixelBeforeLast (pSrc, elem);       //  | 0  0  X  0
	LoadLinePixelBeforeLast (pNext1, elem + 4);	//  | 0  0  0  0
	LoadLinePixelBeforeLast (pNext2, elem + 8);	//  | 0  0  0  0
	return elem[2];
}

inline __m256i LoadFirstLineWindowPixelLast (uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[9]) noexcept
{
											//  +----------- 
	LoadLinePixelLast (pSrc, elem);         //  | 0  0  X
	LoadLinePixelLast (pNext1, elem + 3);	//  | 0  0  0
	LoadLinePixelLast (pNext2, elem + 6);	//  | 0  0  0
	return elem[2];
}


inline __m256i LoadSecondLineWindowPixel0 (uint32_t* __restrict pPrev, uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[12]) noexcept
{
	LoadLinePixel0(pPrev,  elem);     //  | 0  0  0
	LoadLinePixel0(pSrc,   elem + 3); //  | X  0  0
	LoadLinePixel0(pNext1, elem + 6); //  | 0  0  0
	LoadLinePixel0(pNext2, elem + 9); //  | 0  0  0
	return elem[3];
}

inline __m256i LoadSecondLineWindowPixel1 (uint32_t* __restrict pPrev, uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[16]) noexcept
{
	LoadLinePixel1 (pPrev,  elem);      //  | 0  0  0  0
	LoadLinePixel1 (pSrc,   elem + 4);  //  | 0  X  0  0
	LoadLinePixel1 (pNext1, elem + 8);  //  | 0  0  0  0
	LoadLinePixel1 (pNext2, elem + 12); //  | 0  0  0  0
	return elem[5];
}

inline __m256i LoadSecondLineWindowPixel (uint32_t* __restrict pPrev, uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[20]) noexcept
{
	LoadLinePixel (pPrev,  elem);       //  | 0  0  0  0  0
	LoadLinePixel (pSrc,   elem + 5);   //  | 0  0  X  0  0
	LoadLinePixel (pNext1, elem + 10);  //  | 0  0  0  0  0
	LoadLinePixel (pNext2, elem + 15);  //  | 0  0  0  0  0
	return elem[7];
}

inline __m256i LoadSecondLineWindowPixelBeforeLast (uint32_t* __restrict pPrev, uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[16]) noexcept
{
	LoadLinePixelBeforeLast (pPrev, elem);       //  0  0  0  0  |  
	LoadLinePixelBeforeLast (pSrc, elem);        //  0  0  X  0  | 
	LoadLinePixelBeforeLast (pNext1, elem + 4);	 //  0  0  0  0  |
	LoadLinePixelBeforeLast (pNext2, elem + 8);	 //  0  0  0  0  |
	return elem[6];
}

inline __m256i LoadSecondLineWindowPixelLast (uint32_t* __restrict pPrev, uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[12]) noexcept
{
	LoadLinePixelLast(pPrev, elem);          //  0  0  0  |  
	LoadLinePixelLast(pSrc, elem);           //  0  0  X  | 
	LoadLinePixelLast(pNext1, elem + 4);	 //  0  0  0  |
	LoadLinePixelLast(pNext2, elem + 8);	 //  0  0  0  |
	return elem[5];
}


inline __m256i LoadWindowPixel0 (uint32_t* __restrict pPrev2, uint32_t* __restrict pPrev1, uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[15]) noexcept
{
	LoadLinePixel0 (pPrev2, elem);         //  | 0  0  0
	LoadLinePixel0 (pPrev1, elem + 3);     //  | 0  0  0
	LoadLinePixel0 (pSrc,   elem + 6);     //  | X  0  0
	LoadLinePixel0 (pNext1, elem + 9);     //  | 0  0  0
	LoadLinePixel0 (pNext2, elem + 12);    //  | 0  0  0
	return elem[6];
}

inline __m256i LoadWindowPixel1 (uint32_t* __restrict pPrev2, uint32_t* __restrict pPrev1, uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[20]) noexcept
{
	LoadLinePixel1 (pPrev2, elem);         //  | 0  0  0  0
	LoadLinePixel1 (pPrev1, elem + 3);     //  | 0  0  0  0
	LoadLinePixel1 (pSrc,   elem + 6);     //  | 0  X  0  0
	LoadLinePixel1 (pNext1, elem + 9);     //  | 0  0  0  0
	LoadLinePixel1 (pNext2, elem + 12);    //  | 0  0  0  0
	return elem[9];
}

inline __m256i LoadWindowPixel (uint32_t* __restrict pPrev2, uint32_t* __restrict pPrev1, uint32_t* __restrict pSrc, uint32_t* __restrict pNext1, uint32_t* __restrict pNext2, __m256i elem[25]) noexcept
{
	LoadLinePixel1 (pPrev2, elem);         //  | 0  0  0  0  0
	LoadLinePixel1 (pPrev1, elem + 5);     //  | 0  0  0  0  0
	LoadLinePixel1 (pSrc,   elem + 10);    //  | 0  0  X  0  0
	LoadLinePixel1 (pNext1, elem + 15);    //  | 0  0  0  0  0
	LoadLinePixel1 (pNext2, elem + 20);    //  | 0  0  0  0  0
	return elem[12];
}


inline void PartialSort_9_elem_8u (__m256i a[9]) noexcept
{
	/*
	median element in [4] index

	0  0  0
	0  X  0
	0  0  0
	*/
	VectorSort8uPacked(a[1], a[2]);
	VectorSort8uPacked(a[4], a[5]);
	VectorSort8uPacked(a[7], a[8]);
	VectorSort8uPacked(a[0], a[1]);
	VectorSort8uPacked(a[3], a[4]);
	VectorSort8uPacked(a[6], a[7]);
	VectorSort8uPacked(a[1], a[2]);
	VectorSort8uPacked(a[4], a[5]);
	VectorSort8uPacked(a[7], a[8]);
	VectorSort8uPacked(a[0], a[3]);
	VectorSort8uPacked(a[5], a[8]);
	VectorSort8uPacked(a[4], a[7]);
	VectorSort8uPacked(a[3], a[6]);
	VectorSort8uPacked(a[1], a[4]);
	VectorSort8uPacked(a[2], a[5]);
	VectorSort8uPacked(a[4], a[7]);
	VectorSort8uPacked(a[4], a[2]);
	VectorSort8uPacked(a[6], a[4]);
	VectorSort8uPacked(a[4], a[2]);
}


inline void PartialSort_12_elem_8u (__m256i a[12]) noexcept
{
}

inline void PartialSort_15_elem_8u (__m256i a[15]) noexcept
{
	/* median elemnet in index 7 */
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[3],  a[2]);
	VectorSort8uPacked (a[4],  a[5]);
	VectorSort8uPacked (a[7],  a[6]);
	VectorSort8uPacked (a[8],  a[9]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[12], a[13]);
	VectorSort8uPacked (a[0],  a[2]);
	VectorSort8uPacked (a[6],  a[4]);
	VectorSort8uPacked (a[8],  a[10]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[1],  a[3]);
	VectorSort8uPacked (a[7],  a[5]);
	VectorSort8uPacked (a[9],  a[11]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[2],  a[3]);
	VectorSort8uPacked (a[5],  a[4]);
	VectorSort8uPacked (a[7],  a[6]);
	VectorSort8uPacked (a[8],  a[9]);
	VectorSort8uPacked (a[10], a[11]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[0],  a[4]);
	VectorSort8uPacked (a[12], a[8]);
	VectorSort8uPacked (a[1],  a[5]);
	VectorSort8uPacked (a[13], a[9]);
	VectorSort8uPacked (a[2],  a[6]);
	VectorSort8uPacked (a[14], a[10]);
	VectorSort8uPacked (a[3],  a[7]);
	VectorSort8uPacked (a[15], a[11]);
	VectorSort8uPacked (a[0],  a[2]);
	VectorSort8uPacked (a[4],  a[6]);
	VectorSort8uPacked (a[10], a[8]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[1],  a[3]);
	VectorSort8uPacked (a[5],  a[7]);
	VectorSort8uPacked (a[11], a[9]);
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[2],  a[3]);
	VectorSort8uPacked (a[4],  a[5]);
	VectorSort8uPacked (a[6],  a[7]);
	VectorSort8uPacked (a[9],  a[8]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[0],  a[8]);
	VectorSort8uPacked (a[1],  a[9]);
	VectorSort8uPacked (a[2],  a[10]);
	VectorSort8uPacked (a[3],  a[11]);
	VectorSort8uPacked (a[4],  a[12]);
	VectorSort8uPacked (a[5],  a[13]);
	VectorSort8uPacked (a[6],  a[14]);
	VectorSort8uPacked (a[7],  a[15]);
	VectorSort8uPacked (a[0],  a[4]);
	VectorSort8uPacked (a[1],  a[5]);
	VectorSort8uPacked (a[2],  a[6]);
	VectorSort8uPacked (a[3],  a[7]);
	VectorSort8uPacked (a[4],  a[6]);
	VectorSort8uPacked (a[5],  a[7]);
	VectorSort8uPacked (a[6],  a[7]);
}

inline void PartialSort_16_elem_8u (__m256i a[16]) noexcept
{
	/* median elemnet in index 7 */
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[3],  a[2]);
	VectorSort8uPacked (a[4],  a[5]);
	VectorSort8uPacked (a[7],  a[6]);
	VectorSort8uPacked (a[8],  a[9]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[12], a[13]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[0],  a[2]);
	VectorSort8uPacked (a[6],  a[4]);
	VectorSort8uPacked (a[8],  a[10]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[1],  a[3]);
	VectorSort8uPacked (a[7],  a[5]);
	VectorSort8uPacked (a[9],  a[11]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[2],  a[3]);
	VectorSort8uPacked (a[5],  a[4]);
	VectorSort8uPacked (a[7],  a[6]);
	VectorSort8uPacked (a[8],  a[9]);
	VectorSort8uPacked (a[10], a[11]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[0],  a[4]);
	VectorSort8uPacked (a[12], a[8]);
	VectorSort8uPacked (a[1],  a[5]);
	VectorSort8uPacked (a[13], a[9]);
	VectorSort8uPacked (a[2],  a[6]);
	VectorSort8uPacked (a[14], a[10]);
	VectorSort8uPacked (a[3],  a[7]);
	VectorSort8uPacked (a[15], a[11]);
	VectorSort8uPacked (a[0],  a[2]);
	VectorSort8uPacked (a[4],  a[6]);
	VectorSort8uPacked (a[10], a[8]);
	VectorSort8uPacked (a[14], a[12]);
	VectorSort8uPacked (a[1],  a[3]);
	VectorSort8uPacked (a[5],  a[7]);
	VectorSort8uPacked (a[11], a[9]);
	VectorSort8uPacked (a[15], a[13]);
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[2],  a[3]);
	VectorSort8uPacked (a[4],  a[5]);
	VectorSort8uPacked (a[6],  a[7]);
	VectorSort8uPacked (a[9],  a[8]);
	VectorSort8uPacked (a[11], a[10]);
	VectorSort8uPacked (a[13], a[12]);
	VectorSort8uPacked (a[15], a[14]);
	VectorSort8uPacked (a[0],  a[8]);
	VectorSort8uPacked (a[1],  a[9]);
	VectorSort8uPacked (a[2],  a[10]);
	VectorSort8uPacked (a[3],  a[11]);
	VectorSort8uPacked (a[4],  a[12]);
	VectorSort8uPacked (a[5],  a[13]);
	VectorSort8uPacked (a[6],  a[14]);
	VectorSort8uPacked (a[7],  a[15]);
	VectorSort8uPacked (a[0],  a[4]);
	VectorSort8uPacked (a[1],  a[5]);
	VectorSort8uPacked (a[2],  a[6]);
	VectorSort8uPacked (a[3],  a[7]);
	VectorSort8uPacked (a[4],  a[6]);
	VectorSort8uPacked (a[5],  a[7]);
	VectorSort8uPacked (a[6],  a[7]);
}

inline void PartialSort_20_elem_8u (__m256i a[20]) noexcept
{
}

inline void PartialSort_25_elem_8u (__m256i a[25]) noexcept
{
	/*

	median element in [12] index

	0  0  0  0  0
	0  0  0  0  0
	0  0  X  0  0
	0  0  0  0  0
	0  0  0  0  0

	*/
	VectorSort8uPacked (a[0],  a[1]);
	VectorSort8uPacked (a[3],  a[4]);
	VectorSort8uPacked (a[2],  a[4]);
	VectorSort8uPacked (a[2],  a[3]);
	VectorSort8uPacked (a[6],  a[7]);
	VectorSort8uPacked (a[5],  a[7]);
	VectorSort8uPacked (a[5],  a[6]);
	VectorSort8uPacked (a[9],  a[10]);
	VectorSort8uPacked (a[8],  a[10]);
	VectorSort8uPacked (a[8],  a[9]);
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
	VectorSort8uPacked (a[2],  a[5]);
	VectorSort8uPacked (a[3],  a[6]);
	VectorSort8uPacked (a[0],  a[6]);
	VectorSort8uPacked (a[0],  a[3]);
	VectorSort8uPacked (a[4],  a[7]);
	VectorSort8uPacked (a[1],  a[7]);
	VectorSort8uPacked (a[1],  a[4]);
	VectorSort8uPacked (a[11], a[14]);
	VectorSort8uPacked (a[8],  a[14]);
	VectorSort8uPacked (a[8],  a[11]);
	VectorSort8uPacked (a[12], a[15]);
	VectorSort8uPacked (a[9],  a[15]);
	VectorSort8uPacked (a[9],  a[12]);
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
	VectorSort8uPacked (a[9],  a[18]);
	VectorSort8uPacked (a[0],  a[18]);
	VectorSort8uPacked (a[8],  a[17]);
	VectorSort8uPacked (a[0],  a[9]);
	VectorSort8uPacked (a[10], a[19]);
	VectorSort8uPacked (a[1],  a[19]);
	VectorSort8uPacked (a[1],  a[10]);
	VectorSort8uPacked (a[11], a[20]);
	VectorSort8uPacked (a[2],  a[20]);
	VectorSort8uPacked (a[12], a[21]);
	VectorSort8uPacked (a[2],  a[11]);
	VectorSort8uPacked (a[3],  a[21]);
	VectorSort8uPacked (a[3],  a[12]);
	VectorSort8uPacked (a[13], a[22]);
	VectorSort8uPacked (a[4],  a[22]);
	VectorSort8uPacked (a[4],  a[13]);
	VectorSort8uPacked (a[14], a[23]);
	VectorSort8uPacked (a[5],  a[23]);
	VectorSort8uPacked (a[5],  a[14]);
	VectorSort8uPacked (a[15], a[24]);
	VectorSort8uPacked (a[6],  a[24]);
	VectorSort8uPacked (a[6],  a[15]);
	VectorSort8uPacked (a[7],  a[16]);
	VectorSort8uPacked (a[7],  a[19]);
	VectorSort8uPacked (a[13], a[21]);
	VectorSort8uPacked (a[15], a[23]);
	VectorSort8uPacked (a[7],  a[13]);
	VectorSort8uPacked (a[7],  a[15]);
	VectorSort8uPacked (a[1],  a[9]);
	VectorSort8uPacked (a[3],  a[11]);
	VectorSort8uPacked (a[5],  a[17]);
	VectorSort8uPacked (a[11], a[17]);
	VectorSort8uPacked (a[9],  a[17]);
	VectorSort8uPacked (a[4],  a[10]);
	VectorSort8uPacked (a[6],  a[12]);
	VectorSort8uPacked (a[7],  a[14]);
	VectorSort8uPacked (a[4],  a[6]);
	VectorSort8uPacked (a[4],  a[7]);
	VectorSort8uPacked (a[12], a[14]);
	VectorSort8uPacked (a[10], a[14]);
	VectorSort8uPacked (a[6],  a[7]);
	VectorSort8uPacked (a[10], a[12]);
	VectorSort8uPacked (a[6],  a[10]);
	VectorSort8uPacked (a[6],  a[17]);
	VectorSort8uPacked (a[12], a[17]);
	VectorSort8uPacked (a[7],  a[17]);
	VectorSort8uPacked (a[7],  a[10]);
	VectorSort8uPacked (a[12], a[18]);
	VectorSort8uPacked (a[7],  a[12]);
	VectorSort8uPacked (a[10], a[18]);
	VectorSort8uPacked (a[12], a[20]);
	VectorSort8uPacked (a[10], a[20]);
	VectorSort8uPacked (a[10], a[12]);
}

/*
	make median filter with kernel 5x5 from packed format - BGRA444_8u by AVX2 instructions set:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

	LSB                            MSB
	+-------------------------------+
	| B | G | R | A | B | G | R | A | ...
	+-------------------------------+

*/
bool AVX2::Median::median_filter_5x5_BGRA_4444_8u
(
	PF_Pixel_BGRA_8u* __restrict pInImage,
	PF_Pixel_BGRA_8u* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long linePitch
) noexcept
{
	//if (sizeY < 3 || sizeX < 40)
		//		return Scalar::scalar_median_filter_3x3_BGRA_4444_8u(pInImage, pOutImage, sizeY, sizeX, linePitch);

		//	CACHE_ALIGN PF_Pixel_BGRA_8u  ScalarElem[9];
	constexpr A_long pixelsInVector{ static_cast<A_long>(sizeof(__m256i) / PF_Pixel_BGRA_8u_size) };
	constexpr int bgrMask{ 0x00FFFFFF }; /* BGRa */

	A_long i, j;
	const A_long vectorLoadsInLine = sizeX / pixelsInVector;
	const A_long vectorizedLineSize = vectorLoadsInLine * pixelsInVector;
	const A_long lastPixelsInLine = sizeX - vectorizedLineSize;
	const A_long lastIdx = lastPixelsInLine - 2;

	const A_long shortSizeY{ sizeY - 2 };
	const A_long shortSizeX{ sizeX - pixelsInVector * 2};
	const A_long shortSizeX2{ sizeX - pixelsInVector };

	const __m256i rgbMaskVector = _mm256_setr_epi32
	(
		bgrMask, /* mask A component for 1 pixel */
		bgrMask, /* mask A component for 2 pixel */
		bgrMask, /* mask A component for 3 pixel */
		bgrMask, /* mask A component for 4 pixel */
		bgrMask, /* mask A component for 5 pixel */
		bgrMask, /* mask A component for 6 pixel */
		bgrMask, /* mask A component for 7 pixel */
		bgrMask  /* mask A component for 8 pixel */
	);

#ifdef _DEBUG
	__m256i vecData[25]{};
#else
	__m256i vecData[25];
#endif

	/* PROCESS FIRST LINE IN FRAME */
	{
		uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage);
		uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage + linePitch);
		uint32_t* __restrict pSrcVecNextLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage + linePitch * 2);
		__m256i*  __restrict pSrcVecDstLine   = reinterpret_cast<__m256i*  __restrict>(pOutImage);

		/* process first pixel */
		const __m256i srcFirstPixel = LoadFirstLineWindowPixel0 (pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, vecData);
		PartialSort_9_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, srcFirstPixel, vecData[4], rgbMaskVector);

		/* process second pixel */
		pSrcVecCurrLine++, pSrcVecNextLine1++, pSrcVecNextLine2++, pSrcVecDstLine++;
		const __m256i srcSecondPixel = LoadFirstLineWindowPixel1 (pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, vecData);
		PartialSort_12_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, srcSecondPixel, vecData[5], rgbMaskVector);

		/* process rest of pixels */
		for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = LoadFirstLineWindowPixel (pSrcVecCurrLine + i, pSrcVecNextLine1 + i, pSrcVecNextLine2 + i, vecData);
			PartialSort_15_elem_8u (vecData);
			StoreByMask8u (pSrcVecDstLine, srcSecondPixel, vecData[7], rgbMaskVector);
			pSrcVecDstLine++;
		} /* for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector) */

		/* process one before last pixel */
		const __m256i srcPixelBeforeLast = LoadFirstLineWindowPixelBeforeLast (pSrcVecCurrLine + shortSizeX, pSrcVecNextLine1 + shortSizeX, pSrcVecNextLine2 + shortSizeX, vecData);
		PartialSort_12_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, srcPixelBeforeLast, vecData[5], rgbMaskVector);
		pSrcVecDstLine++;

		/* process last pixel */
		const __m256i srcPixelLast = LoadFirstLineWindowPixelBeforeLast (pSrcVecCurrLine  + shortSizeX2,
			                                                             pSrcVecNextLine1 + shortSizeX2,
			                                                             pSrcVecNextLine2 + shortSizeX2,
			                                                             vecData);
		PartialSort_9_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, srcPixelLast, vecData[4], rgbMaskVector);
	}

	/* PROCESS SECOND LINE IN FRAME */
	{
		uint32_t* __restrict pSrcVecPrevLine  = reinterpret_cast<uint32_t* __restrict>(pInImage);
		uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage + linePitch);
		uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage + linePitch * 2);
		uint32_t* __restrict pSrcVecNextLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage + linePitch * 3);
		__m256i*  __restrict pSrcVecDstLine   = reinterpret_cast<__m256i*  __restrict>(pOutImage+ linePitch);

		/* process first pixel */
		const __m256i srcFirstPixel = LoadSecondLineWindowPixel0 (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, vecData);
		PartialSort_12_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, srcFirstPixel, vecData[4], rgbMaskVector);

		/* process second pixel */
		pSrcVecPrevLine++, pSrcVecCurrLine++, pSrcVecNextLine1++, pSrcVecNextLine2++, pSrcVecDstLine++;
		const __m256i srcSecondPixel = LoadSecondLineWindowPixel1 (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, vecData);
		PartialSort_16_elem_8u(vecData);
		StoreByMask8u(pSrcVecDstLine, srcSecondPixel, vecData[5], rgbMaskVector);

		/* process rest of pixels */
		for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector)
		{
			const __m256i srcOrig = LoadSecondLineWindowPixel (pSrcVecPrevLine+ i, pSrcVecCurrLine + i, pSrcVecNextLine1 + i, pSrcVecNextLine2 + i, vecData);
			PartialSort_20_elem_8u(vecData);
			StoreByMask8u (pSrcVecDstLine, srcSecondPixel, vecData[9], rgbMaskVector);
		} /* for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector) */

		/* process one before last pixel */
		const __m256i srcPixelBeforeLast = LoadSecondLineWindowPixelBeforeLast (pSrcVecPrevLine  + shortSizeX,
			                                                                    pSrcVecCurrLine  + shortSizeX,
			                                                                    pSrcVecNextLine1 + shortSizeX,
			                                                                    pSrcVecNextLine2 + shortSizeX,
			                                                                    vecData);
		PartialSort_16_elem_8u (vecData);
		StoreByMask8u (pSrcVecDstLine, srcPixelBeforeLast, vecData[7], rgbMaskVector);
		pSrcVecDstLine++;

		/* process last pixel */
		const __m256i srcPixelLast = LoadSecondLineWindowPixelLast (pSrcVecPrevLine  + shortSizeX2,
			                                                        pSrcVecCurrLine  + shortSizeX2,
			                                                        pSrcVecNextLine1 + shortSizeX2,
			                                                        pSrcVecNextLine2 + shortSizeX2,
			                                                        vecData);
		PartialSort_12_elem_8u(vecData);
		StoreByMask8u (pSrcVecDstLine, srcPixelLast, vecData[5], rgbMaskVector);
	}

	/* PROCESS REST OF LINES IN FRAME */
	{
		/* PROCESS LINES IN FRAME FROM 2 to SIZEY-2 */
		for (j = 2; j < shortSizeY; j++)
		{
			uint32_t* __restrict pSrcVecPrevLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage + (j - 2) * linePitch);
			uint32_t* __restrict pSrcVecPrevLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage + (j - 1) * linePitch);
			uint32_t* __restrict pSrcVecCurrLine  = reinterpret_cast<uint32_t* __restrict>(pInImage +  j      * linePitch);
			uint32_t* __restrict pSrcVecNextLine1 = reinterpret_cast<uint32_t* __restrict>(pInImage + (j + 1) * linePitch);
			uint32_t* __restrict pSrcVecNextLine2 = reinterpret_cast<uint32_t* __restrict>(pInImage + (j + 2) * linePitch);
			__m256i*  __restrict pSrcVecDstLine   = reinterpret_cast<__m256i*  __restrict>(pOutImage + j * linePitch);

			const __m256i srcFirstPixel = LoadWindowPixel0(pSrcVecPrevLine2, pSrcVecPrevLine1, pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, vecData);
			PartialSort_15_elem_8u(vecData);
			StoreByMask8u(pSrcVecDstLine, srcFirstPixel, vecData[7], rgbMaskVector);
			pSrcVecPrevLine2++, pSrcVecPrevLine1++, pSrcVecCurrLine++, pSrcVecNextLine1++, pSrcVecNextLine2++, pSrcVecDstLine++;

			const __m256i srcSecondPixel = LoadWindowPixel1(pSrcVecPrevLine2, pSrcVecPrevLine1, pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, vecData);
			PartialSort_20_elem_8u(vecData);
			StoreByMask8u (pSrcVecDstLine, srcFirstPixel, vecData[9], rgbMaskVector);
			pSrcVecPrevLine2++, pSrcVecPrevLine1++, pSrcVecCurrLine++, pSrcVecNextLine1++, pSrcVecNextLine2++, pSrcVecDstLine++;

			/* process rest of pixels */
			for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector)
			{
				const __m256i srcPixel = LoadWindowPixel(pSrcVecPrevLine2 + i, pSrcVecPrevLine1 + i, pSrcVecCurrLine + i, pSrcVecNextLine1 + i, pSrcVecNextLine2 + i, vecData);
				PartialSort_25_elem_8u (vecData);
				StoreByMask8u (pSrcVecDstLine, srcFirstPixel, vecData[12], rgbMaskVector);
				pSrcVecDstLine++;
			} /* for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector) */

			/* process pixelsin line bedofre last */

			/* process last pixels in line */

		} /* for (j = 2; j < shortSizeY; j++) */  
	}

	/* PROCESS LINE BEFORE LAST */
	{

	}

	/* PROCESS LAST LINE */
	{

	}

	return true;
}