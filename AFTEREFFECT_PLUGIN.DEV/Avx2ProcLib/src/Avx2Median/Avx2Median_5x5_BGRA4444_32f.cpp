#include "Avx2Median.hpp"
#include "Avx2MedianInternal.hpp"


inline void LoadLinePixel0 (__m128* __restrict pSrc, __m256 elemLine[3]) noexcept
{
	//  | X  0  0
	elemLine[0] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc));
	elemLine[1] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc + 1));
	elemLine[2] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc + 2));
}

inline void LoadLinePixel1 (__m128* __restrict pSrc, __m256 elemLine[4]) noexcept
{
	//  | 0  X  0  0
	elemLine[0] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc - 1));
	elemLine[1] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc));
	elemLine[2] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc + 1));
	elemLine[3] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc + 2));
}

inline void LoadLinePixel (__m128* __restrict pSrc, __m256 elemLine[5]) noexcept
{
	// 0  0  X  0  0
	elemLine[0] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc - 2));
	elemLine[1] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc - 1));
	elemLine[2] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc));
	elemLine[3] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc + 1));
	elemLine[4] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc + 2));
}

inline void LoadLinePixelBeforeLast (__m128* __restrict pSrc, __m256 elemLine[4]) noexcept
{
	// 0  0  X  0 |
	elemLine[0] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc - 2));
	elemLine[1] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc - 1));
	elemLine[2] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc));
	elemLine[3] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc + 1));
}

inline void LoadLinePixelLast (__m128* __restrict pSrc, __m256 elemLine[3]) noexcept
{
	//  0  0  X |
	elemLine[0] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc - 2));
	elemLine[1] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc - 1));
	elemLine[2] = _mm256_loadu_ps (reinterpret_cast<float*>(pSrc));
}

inline __m256 LoadFirstLineWindowPixel0 (__m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[9]) noexcept
{
                                       //  +-------- 
	LoadLinePixel0 (pSrc,   elem);     //  | X  0  0
	LoadLinePixel0 (pNext1, elem + 3); //  | 0  0  0
	LoadLinePixel0 (pNext2, elem + 6); //  | 0  0  0
	return elem[0];                     
}

inline __m256 LoadFirstLineWindowPixel1 (__m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[12]) noexcept
{
                                       //  +----------- 
	LoadLinePixel1 (pSrc, elem);       //  | 0  X  0  0
	LoadLinePixel1 (pNext1, elem + 4); //  | 0  0  0  0
	LoadLinePixel1 (pNext2, elem + 8); //  | 0  0  0  0
	return elem[1];
}

inline __m256 LoadFirstLineWindowPixel (__m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[15]) noexcept
{
                                       //  +----------- 
	LoadLinePixel (pSrc, elem);        //  | 0  0  X  0  0
	LoadLinePixel (pNext1, elem + 5);  //  | 0  0  0  0  0
	LoadLinePixel (pNext2, elem + 10); //  | 0  0  0  0  0
	return elem[2];
}

inline __m256 LoadFirstLineWindowPixelBeforeLast (__m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[12]) noexcept
{
                                                //  +----------- 
	LoadLinePixelBeforeLast (pSrc, elem);       //  | 0  0  X  0
	LoadLinePixelBeforeLast (pNext1, elem + 4);	//  | 0  0  0  0
	LoadLinePixelBeforeLast (pNext2, elem + 8);	//  | 0  0  0  0
	return elem[2];
}

inline __m256 LoadFirstLineWindowPixelLast (__m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[9]) noexcept
{
                                            //  +----------- 
	LoadLinePixelLast (pSrc, elem);         //  | 0  0  X
	LoadLinePixelLast (pNext1, elem + 3);	//  | 0  0  0
	LoadLinePixelLast (pNext2, elem + 6);	//  | 0  0  0
	return elem[2];
}


inline __m256 LoadSecondLineWindowPixel0 (__m128* __restrict pPrev, __m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[12]) noexcept
{
	LoadLinePixel0(pPrev,  elem);     //  | 0  0  0
	LoadLinePixel0(pSrc,   elem + 3); //  | X  0  0
	LoadLinePixel0(pNext1, elem + 6); //  | 0  0  0
	LoadLinePixel0(pNext2, elem + 9); //  | 0  0  0
	return elem[3];
}

inline __m256 LoadSecondLineWindowPixel1 (__m128* __restrict pPrev, __m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[16]) noexcept
{
	LoadLinePixel1 (pPrev,  elem);      //  | 0  0  0  0
	LoadLinePixel1 (pSrc,   elem + 4);  //  | 0  X  0  0
	LoadLinePixel1 (pNext1, elem + 8);  //  | 0  0  0  0
	LoadLinePixel1 (pNext2, elem + 12); //  | 0  0  0  0
	return elem[5];
}

inline __m256 LoadSecondLineWindowPixel (__m128* __restrict pPrev, __m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[20]) noexcept
{
	LoadLinePixel (pPrev,  elem);       //  | 0  0  0  0  0
	LoadLinePixel (pSrc,   elem + 5);   //  | 0  0  X  0  0
	LoadLinePixel (pNext1, elem + 10);  //  | 0  0  0  0  0
	LoadLinePixel (pNext2, elem + 15);  //  | 0  0  0  0  0
	return elem[7];
}

inline __m256 LoadSecondLineWindowPixelBeforeLast (__m128* __restrict pPrev, __m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[16]) noexcept
{
	LoadLinePixelBeforeLast (pPrev, elem);       //  0  0  0  0  |  
	LoadLinePixelBeforeLast (pSrc, elem);        //  0  0  X  0  | 
	LoadLinePixelBeforeLast (pNext1, elem + 4);	 //  0  0  0  0  |
	LoadLinePixelBeforeLast (pNext2, elem + 8);	 //  0  0  0  0  |
	return elem[6];
}

inline __m256 LoadSecondLineWindowPixelLast (__m128* __restrict pPrev, __m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[12]) noexcept
{
	LoadLinePixelLast (pPrev, elem);         //  0  0  0  |  
	LoadLinePixelLast (pSrc, elem);          //  0  0  X  | 
	LoadLinePixelLast (pNext1, elem + 4);	 //  0  0  0  |
	LoadLinePixelLast (pNext2, elem + 8);	 //  0  0  0  |
	return elem[5];
}


inline __m256 LoadWindowPixel0 (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[15]) noexcept
{
	LoadLinePixel0 (pPrev2, elem);         //  | 0  0  0
	LoadLinePixel0 (pPrev1, elem + 3);     //  | 0  0  0
	LoadLinePixel0 (pSrc,   elem + 6);     //  | X  0  0
	LoadLinePixel0 (pNext1, elem + 9);     //  | 0  0  0
	LoadLinePixel0 (pNext2, elem + 12);    //  | 0  0  0
	return elem[6];
}

inline __m256 LoadWindowPixel1 (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[20]) noexcept
{
	LoadLinePixel1 (pPrev2, elem);         //  | 0  0  0  0
	LoadLinePixel1 (pPrev1, elem + 3);     //  | 0  0  0  0
	LoadLinePixel1 (pSrc,   elem + 6);     //  | 0  X  0  0
	LoadLinePixel1 (pNext1, elem + 9);     //  | 0  0  0  0
	LoadLinePixel1 (pNext2, elem + 12);    //  | 0  0  0  0
	return elem[9];
}

inline __m256 LoadWindowPixel (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[25]) noexcept
{
	LoadLinePixel1 (pPrev2, elem);         //  | 0  0  0  0  0
	LoadLinePixel1 (pPrev1, elem + 5);     //  | 0  0  0  0  0
	LoadLinePixel1 (pSrc,   elem + 10);    //  | 0  0  X  0  0
	LoadLinePixel1 (pNext1, elem + 15);    //  | 0  0  0  0  0
	LoadLinePixel1 (pNext2, elem + 20);    //  | 0  0  0  0  0
	return elem[12];
}

inline __m256 LoadWindowPixelBeforeLast (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[20]) noexcept
{
	LoadLinePixelBeforeLast(pPrev2, elem);         //  | 0  0  0  0
	LoadLinePixelBeforeLast(pPrev1, elem + 4);     //  | 0  0  0  0
	LoadLinePixelBeforeLast(pSrc,   elem + 8);     //  | 0  0  X  0
	LoadLinePixelBeforeLast(pNext1, elem + 12);    //  | 0  0  0  0
	LoadLinePixelBeforeLast(pNext2, elem + 16);    //  | 0  0  0  0
	return elem[10];
}

inline __m256 LoadWindowPixelLast (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m128* __restrict pNext1, __m128* __restrict pNext2, __m256 elem[15]) noexcept
{
	LoadLinePixelLast (pPrev2, elem);         //  | 0  0  0
	LoadLinePixelLast (pPrev1, elem + 3);     //  | 0  0  0
	LoadLinePixelLast (pSrc,   elem + 6);     //  | 0  0  X
	LoadLinePixelLast (pNext1, elem + 9);     //  | 0  0  0
	LoadLinePixelLast (pNext2, elem + 12);    //  | 0  0  0
	return elem[8];
}

inline __m256 LoadWindowBeforeLastLineFirstPixel (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m128* __restrict pNext1, __m256 elem[12]) noexcept
{
	LoadLinePixel0 (pPrev2, elem);        //  |  0  0  0
	LoadLinePixel0 (pPrev1, elem + 3);    //  |  0  0  0
	LoadLinePixel0 (pSrc,   elem + 6);    //  |  X  0  0
	LoadLinePixel0(pNext1,  elem + 9);    //  |  0  0  0
	return elem[6];
}

inline __m256 LoadWindowBeforeLastLineSecondPixel (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m128* __restrict pNext1, __m256 elem[16]) noexcept
{
	LoadLinePixel1 (pPrev2, elem);        //  |  0  0  0  0
	LoadLinePixel1 (pPrev1, elem + 4);    //  |  0  0  0  0
	LoadLinePixel1 (pSrc,   elem + 8);    //  |  0  X  0  0
	LoadLinePixel1 (pNext1, elem + 12);   //  |  0  0  0  0
	return elem[9];
}

inline __m256 LoadWindowBeforeLastLine (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m128* __restrict pNext1, __m256 elem[20]) noexcept
{
	LoadLinePixel (pPrev2, elem);        //  |  0  0  0  0  0
	LoadLinePixel (pPrev1, elem + 5);    //  |  0  0  0  0  0
	LoadLinePixel (pSrc,   elem + 10);   //  |  0  0  X  0  0
	LoadLinePixel (pNext1, elem + 15);   //  |  0  0  0  0  0
	return elem[12];
}

inline __m256 LoadWindowBeforeLastLineBeforeLastPixel (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m128* __restrict pNext1, __m256 elem[16]) noexcept
{
	LoadLinePixelBeforeLast (pPrev2, elem);        //  |  0  0  0  0
	LoadLinePixelBeforeLast (pPrev1, elem + 4);    //  |  0  0  0  0
	LoadLinePixelBeforeLast (pSrc,   elem + 8);    //  |  0  0  X  0
	LoadLinePixelBeforeLast (pNext1, elem + 12);   //  |  0  0  0  0
	return elem[10];
}

inline __m256 LoadWindowBeforeLastLineLastPixel (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m128* __restrict pNext1, __m256 elem[12]) noexcept
{
	LoadLinePixelLast (pPrev2, elem);        //  |  0  0  0
	LoadLinePixelLast (pPrev1, elem + 4);    //  |  0  0  0
	LoadLinePixelLast (pSrc,   elem + 8);    //  |  0  0  X
	LoadLinePixelLast (pNext1, elem + 12);   //  |  0  0  0
	return elem[8];
}


inline __m256 LoadWindowLastLineFirstPixel (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m256 elem[9]) noexcept
{
	LoadLinePixel0 (pPrev2, elem);        //  |  0  0  0
	LoadLinePixel0 (pPrev1, elem + 3);    //  |  0  0  0
	LoadLinePixel0 (pSrc,   elem + 6);    //  |  X  0  0
	return elem[6];
}

inline __m256 LoadWindowLastLineSecondPixel (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m256 elem[12]) noexcept
{
	LoadLinePixel1 (pPrev2, elem);        //  |  0  0  0  0
	LoadLinePixel1 (pPrev1, elem + 4);    //  |  0  0  0  0
	LoadLinePixel1 (pSrc,   elem + 8);    //  |  0  X  0  0
	return elem[9];
}

inline __m256 LoadWindowLastLine (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m256 elem[15]) noexcept
{
	LoadLinePixel(pPrev2, elem);        //  |  0  0  0  0  0
	LoadLinePixel(pPrev1, elem + 5);    //  |  0  0  0  0  0
	LoadLinePixel(pSrc,   elem + 10);   //  |  0  0  X  0  0
	return elem[12];
}

inline __m256 LoadWindowLastLineBeforeLastPixel (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m256 elem[12]) noexcept
{
	LoadLinePixelBeforeLast (pPrev2, elem);        //  |  0  0  0  0
	LoadLinePixelBeforeLast (pPrev1, elem + 4);    //  |  0  0  0  0
	LoadLinePixelBeforeLast (pSrc,   elem + 8);    //  |  0  0  X  0
	return elem[10];
}

inline __m256 LoadWindowLastLineLastPixel (__m128* __restrict pPrev2, __m128* __restrict pPrev1, __m128* __restrict pSrc, __m256 elem[9]) noexcept
{
	LoadLinePixelLast (pPrev2, elem);        //  |  0  0  0
	LoadLinePixelLast (pPrev1, elem + 3);    //  |  0  0  0
	LoadLinePixelLast (pSrc,   elem + 6);    //  |  0  0  X
	return elem[8];
}


inline void PartialSort_9_elem_32f (__m256 a[9]) noexcept
{
	/*
		median element in [4] index

		0  0  0
		0  X  0
		0  0  0

	*/
	VectorSort32fPacked (a[1], a[2]);
	VectorSort32fPacked (a[4], a[5]);
	VectorSort32fPacked (a[7], a[8]);
	VectorSort32fPacked (a[0], a[1]);
	VectorSort32fPacked (a[3], a[4]);
	VectorSort32fPacked (a[6], a[7]);
	VectorSort32fPacked (a[1], a[2]);
	VectorSort32fPacked (a[4], a[5]);
	VectorSort32fPacked (a[7], a[8]);
	VectorSort32fPacked (a[0], a[3]);
	VectorSort32fPacked (a[5], a[8]);
	VectorSort32fPacked (a[4], a[7]);
	VectorSort32fPacked (a[3], a[6]);
	VectorSort32fPacked (a[1], a[4]);
	VectorSort32fPacked (a[2], a[5]);
	VectorSort32fPacked (a[4], a[7]);
	VectorSort32fPacked (a[4], a[2]);
	VectorSort32fPacked (a[6], a[4]);
	VectorSort32fPacked (a[4], a[2]);
}

inline void PartialSort_12_elem_32f (__m256 a[12]) noexcept
{
	/* median elemnet in index 5 */
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[3],  a[2]);
	VectorSort32fPacked (a[4],  a[5]);
	VectorSort32fPacked (a[7],  a[6]);
	VectorSort32fPacked (a[8],  a[9]);
	VectorSort32fPacked (a[11], a[10]);
	VectorSort32fPacked (a[0],  a[2]);
	VectorSort32fPacked (a[1],  a[3]);
	VectorSort32fPacked (a[6],  a[4]);
	VectorSort32fPacked (a[7],  a[5]);
	VectorSort32fPacked (a[8],  a[10]);
	VectorSort32fPacked (a[9],  a[11]);
	VectorSort32fPacked (a[0],  a[4]);
	VectorSort32fPacked (a[1],  a[5]);
	VectorSort32fPacked (a[2],  a[6]);
	VectorSort32fPacked (a[3],  a[7]);
	VectorSort32fPacked (a[0],  a[2]);
	VectorSort32fPacked (a[4],  a[6]);
	VectorSort32fPacked (a[10], a[8]);
	VectorSort32fPacked (a[11], a[9]);
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[2],  a[3]);
	VectorSort32fPacked (a[4],  a[5]);
	VectorSort32fPacked (a[6],  a[7]);
	VectorSort32fPacked (a[9],  a[8]);
	VectorSort32fPacked (a[11], a[10]);
	VectorSort32fPacked (a[0],  a[8]);
	VectorSort32fPacked (a[1],  a[9]);
	VectorSort32fPacked (a[2],  a[10]);
	VectorSort32fPacked (a[3],  a[11]);
	VectorSort32fPacked (a[0],  a[4]);
	VectorSort32fPacked (a[1],  a[5]);
	VectorSort32fPacked (a[2],  a[6]);
	VectorSort32fPacked (a[3],  a[7]);
	VectorSort32fPacked (a[5],  a[7]);
}


inline void PartialSort_15_elem_32f (__m256 a[15]) noexcept
{
	/* median elemnet in index 7 */
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[3],  a[2]);
	VectorSort32fPacked (a[4],  a[5]);
	VectorSort32fPacked (a[7],  a[6]);
	VectorSort32fPacked (a[8],  a[9]);
	VectorSort32fPacked (a[11], a[10]);
	VectorSort32fPacked (a[12], a[13]);
	VectorSort32fPacked (a[0],  a[2]);
	VectorSort32fPacked (a[6],  a[4]);
	VectorSort32fPacked (a[8],  a[10]);
	VectorSort32fPacked (a[14], a[12]);
	VectorSort32fPacked (a[1],  a[3]);
	VectorSort32fPacked (a[7],  a[5]);
	VectorSort32fPacked (a[9],  a[11]);
	VectorSort32fPacked (a[15], a[13]);
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[2],  a[3]);
	VectorSort32fPacked (a[5],  a[4]);
	VectorSort32fPacked (a[7],  a[6]);
	VectorSort32fPacked (a[8],  a[9]);
	VectorSort32fPacked (a[10], a[11]);
	VectorSort32fPacked (a[13], a[12]);
	VectorSort32fPacked (a[0],  a[4]);
	VectorSort32fPacked (a[12], a[8]);
	VectorSort32fPacked (a[1],  a[5]);
	VectorSort32fPacked (a[13], a[9]);
	VectorSort32fPacked (a[2],  a[6]);
	VectorSort32fPacked (a[14], a[10]);
	VectorSort32fPacked (a[3],  a[7]);
	VectorSort32fPacked (a[15], a[11]);
	VectorSort32fPacked (a[0],  a[2]);
	VectorSort32fPacked (a[4],  a[6]);
	VectorSort32fPacked (a[10], a[8]);
	VectorSort32fPacked (a[14], a[12]);
	VectorSort32fPacked (a[1],  a[3]);
	VectorSort32fPacked (a[5],  a[7]);
	VectorSort32fPacked (a[11], a[9]);
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[2],  a[3]);
	VectorSort32fPacked (a[4],  a[5]);
	VectorSort32fPacked (a[6],  a[7]);
	VectorSort32fPacked (a[9],  a[8]);
	VectorSort32fPacked (a[11], a[10]);
	VectorSort32fPacked (a[13], a[12]);
	VectorSort32fPacked (a[0],  a[8]);
	VectorSort32fPacked (a[1],  a[9]);
	VectorSort32fPacked (a[2],  a[10]);
	VectorSort32fPacked (a[3],  a[11]);
	VectorSort32fPacked (a[4],  a[12]);
	VectorSort32fPacked (a[5],  a[13]);
	VectorSort32fPacked (a[6],  a[14]);
	VectorSort32fPacked (a[7],  a[15]);
	VectorSort32fPacked (a[0],  a[4]);
	VectorSort32fPacked (a[1],  a[5]);
	VectorSort32fPacked (a[2],  a[6]);
	VectorSort32fPacked (a[3],  a[7]);
	VectorSort32fPacked (a[4],  a[6]);
	VectorSort32fPacked (a[5],  a[7]);
	VectorSort32fPacked (a[6],  a[7]);
}

inline void PartialSort_16_elem_32f (__m256 a[16]) noexcept
{
	/* median elemnet in index 7 */
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[3],  a[2]);
	VectorSort32fPacked (a[4],  a[5]);
	VectorSort32fPacked (a[7],  a[6]);
	VectorSort32fPacked (a[8],  a[9]);
	VectorSort32fPacked (a[11], a[10]);
	VectorSort32fPacked (a[12], a[13]);
	VectorSort32fPacked (a[15], a[14]);
	VectorSort32fPacked (a[0],  a[2]);
	VectorSort32fPacked (a[6],  a[4]);
	VectorSort32fPacked (a[8],  a[10]);
	VectorSort32fPacked (a[14], a[12]);
	VectorSort32fPacked (a[1],  a[3]);
	VectorSort32fPacked (a[7],  a[5]);
	VectorSort32fPacked (a[9],  a[11]);
	VectorSort32fPacked (a[15], a[13]);
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[2],  a[3]);
	VectorSort32fPacked (a[5],  a[4]);
	VectorSort32fPacked (a[7],  a[6]);
	VectorSort32fPacked (a[8],  a[9]);
	VectorSort32fPacked (a[10], a[11]);
	VectorSort32fPacked (a[13], a[12]);
	VectorSort32fPacked (a[15], a[14]);
	VectorSort32fPacked (a[0],  a[4]);
	VectorSort32fPacked (a[12], a[8]);
	VectorSort32fPacked (a[1],  a[5]);
	VectorSort32fPacked (a[13], a[9]);
	VectorSort32fPacked (a[2],  a[6]);
	VectorSort32fPacked (a[14], a[10]);
	VectorSort32fPacked (a[3],  a[7]);
	VectorSort32fPacked (a[15], a[11]);
	VectorSort32fPacked (a[0],  a[2]);
	VectorSort32fPacked (a[4],  a[6]);
	VectorSort32fPacked (a[10], a[8]);
	VectorSort32fPacked (a[14], a[12]);
	VectorSort32fPacked (a[1],  a[3]);
	VectorSort32fPacked (a[5],  a[7]);
	VectorSort32fPacked (a[11], a[9]);
	VectorSort32fPacked (a[15], a[13]);
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[2],  a[3]);
	VectorSort32fPacked (a[4],  a[5]);
	VectorSort32fPacked (a[6],  a[7]);
	VectorSort32fPacked (a[9],  a[8]);
	VectorSort32fPacked (a[11], a[10]);
	VectorSort32fPacked (a[13], a[12]);
	VectorSort32fPacked (a[15], a[14]);
	VectorSort32fPacked (a[0],  a[8]);
	VectorSort32fPacked (a[1],  a[9]);
	VectorSort32fPacked (a[2],  a[10]);
	VectorSort32fPacked (a[3],  a[11]);
	VectorSort32fPacked (a[4],  a[12]);
	VectorSort32fPacked (a[5],  a[13]);
	VectorSort32fPacked (a[6],  a[14]);
	VectorSort32fPacked (a[7],  a[15]);
	VectorSort32fPacked (a[0],  a[4]);
	VectorSort32fPacked (a[1],  a[5]);
	VectorSort32fPacked (a[2],  a[6]);
	VectorSort32fPacked (a[3],  a[7]);
	VectorSort32fPacked (a[4],  a[6]);
	VectorSort32fPacked (a[5],  a[7]);
	VectorSort32fPacked (a[6],  a[7]);
}

inline void PartialSort_20_elem_32f (__m256 a[20]) noexcept
{
	/* median element in index 9 */
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[3],  a[2]);
	VectorSort32fPacked (a[4],  a[5]);
	VectorSort32fPacked (a[7],  a[6]);
	VectorSort32fPacked (a[8],  a[9]);
	VectorSort32fPacked (a[11], a[10]);
	VectorSort32fPacked (a[12], a[13]);
	VectorSort32fPacked (a[15], a[14]);
	VectorSort32fPacked (a[16], a[17]);
	VectorSort32fPacked (a[19], a[18]);
	VectorSort32fPacked (a[0],  a[2]);
	VectorSort32fPacked (a[1],  a[3]);
	VectorSort32fPacked (a[6],  a[4]);
	VectorSort32fPacked (a[7],  a[5]);
	VectorSort32fPacked (a[8],  a[10]);
	VectorSort32fPacked (a[9],  a[11]);
	VectorSort32fPacked (a[14], a[12]);
	VectorSort32fPacked (a[15], a[13]);
	VectorSort32fPacked (a[16], a[18]);
	VectorSort32fPacked (a[17], a[19]);
    VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[2],  a[3]);
	VectorSort32fPacked (a[5],  a[4]);
	VectorSort32fPacked (a[7],  a[6]);
	VectorSort32fPacked (a[8],  a[9]);
	VectorSort32fPacked (a[10], a[11]);
	VectorSort32fPacked (a[13], a[12]);
	VectorSort32fPacked (a[15], a[14]);
	VectorSort32fPacked (a[16], a[17]);
	VectorSort32fPacked (a[18], a[19]);
	VectorSort32fPacked (a[0],  a[4]);
	VectorSort32fPacked (a[14], a[10]);
	VectorSort32fPacked (a[1],  a[5]);
	VectorSort32fPacked (a[15], a[11]);
	VectorSort32fPacked (a[2],  a[6]);
	VectorSort32fPacked (a[16], a[12]);
	VectorSort32fPacked (a[3],  a[7]);
	VectorSort32fPacked (a[17], a[13]);
	VectorSort32fPacked (a[4],  a[8]);
	VectorSort32fPacked (a[18], a[14]);
	VectorSort32fPacked (a[5],  a[9]);
	VectorSort32fPacked (a[19], a[15]);
	VectorSort32fPacked (a[0],  a[2]);
	VectorSort32fPacked (a[4],  a[6]);
	VectorSort32fPacked (a[13], a[11]);
	VectorSort32fPacked (a[17], a[15]);
	VectorSort32fPacked (a[1],  a[3]);
	VectorSort32fPacked (a[5],  a[7]);
	VectorSort32fPacked (a[14], a[12]);
	VectorSort32fPacked (a[18], a[16]);
	VectorSort32fPacked (a[2],  a[4]);
	VectorSort32fPacked (a[6],  a[8]);
	VectorSort32fPacked (a[15], a[13]);
	VectorSort32fPacked (a[19], a[17]);
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[2],  a[3]);
	VectorSort32fPacked (a[4],  a[5]);
	VectorSort32fPacked (a[6],  a[7]);
	VectorSort32fPacked (a[8],  a[9]);
	VectorSort32fPacked (a[11], a[10]);
	VectorSort32fPacked (a[13], a[12]);
	VectorSort32fPacked (a[15], a[14]);
	VectorSort32fPacked (a[17], a[16]);
	VectorSort32fPacked (a[19], a[18]);
	VectorSort32fPacked (a[0],  a[9]);
	VectorSort32fPacked (a[1],  a[10]);
	VectorSort32fPacked (a[2],  a[11]);
	VectorSort32fPacked (a[3],  a[12]);
	VectorSort32fPacked (a[4],  a[13]);
	VectorSort32fPacked (a[5],  a[14]);
	VectorSort32fPacked (a[6],  a[15]);
	VectorSort32fPacked (a[7],  a[16]);
	VectorSort32fPacked (a[8],  a[17]);
	VectorSort32fPacked (a[9],  a[18]);
	VectorSort32fPacked (a[10], a[19]);
	VectorSort32fPacked (a[0],  a[4]);
	VectorSort32fPacked (a[1],  a[5]);
	VectorSort32fPacked (a[2],  a[6]);
	VectorSort32fPacked (a[3],  a[7]);
	VectorSort32fPacked (a[4],  a[8]);
	VectorSort32fPacked (a[5],  a[9]);
	VectorSort32fPacked (a[4],  a[6]);
	VectorSort32fPacked (a[5],  a[7]);
	VectorSort32fPacked (a[6],  a[8]);
	VectorSort32fPacked (a[8],  a[9]);
}

inline void PartialSort_25_elem_32f (__m256 a[25]) noexcept
{
	/*

	median element in [12] index

	0  0  0  0  0
	0  0  0  0  0
	0  0  X  0  0
	0  0  0  0  0
	0  0  0  0  0

	*/
	VectorSort32fPacked (a[0],  a[1]);
	VectorSort32fPacked (a[3],  a[4]);
	VectorSort32fPacked (a[2],  a[4]);
	VectorSort32fPacked (a[2],  a[3]);
	VectorSort32fPacked (a[6],  a[7]);
	VectorSort32fPacked (a[5],  a[7]);
	VectorSort32fPacked (a[5],  a[6]);
	VectorSort32fPacked (a[9],  a[10]);
	VectorSort32fPacked (a[8],  a[10]);
	VectorSort32fPacked (a[8],  a[9]);
	VectorSort32fPacked (a[12], a[13]);
	VectorSort32fPacked (a[11], a[13]);
	VectorSort32fPacked (a[11], a[12]);
	VectorSort32fPacked (a[15], a[16]);
	VectorSort32fPacked (a[14], a[16]);
	VectorSort32fPacked (a[14], a[15]);
	VectorSort32fPacked (a[18], a[19]);
	VectorSort32fPacked (a[17], a[19]);
	VectorSort32fPacked (a[17], a[18]);
	VectorSort32fPacked (a[21], a[22]);
	VectorSort32fPacked (a[20], a[22]);
	VectorSort32fPacked (a[20], a[21]);
	VectorSort32fPacked (a[23], a[24]);
	VectorSort32fPacked (a[2],  a[5]);
	VectorSort32fPacked (a[3],  a[6]);
	VectorSort32fPacked (a[0],  a[6]);
	VectorSort32fPacked (a[0],  a[3]);
	VectorSort32fPacked (a[4],  a[7]);
	VectorSort32fPacked (a[1],  a[7]);
	VectorSort32fPacked (a[1],  a[4]);
	VectorSort32fPacked (a[11], a[14]);
	VectorSort32fPacked (a[8],  a[14]);
	VectorSort32fPacked (a[8],  a[11]);
	VectorSort32fPacked (a[12], a[15]);
	VectorSort32fPacked (a[9],  a[15]);
	VectorSort32fPacked (a[9],  a[12]);
	VectorSort32fPacked (a[13], a[16]);
	VectorSort32fPacked (a[10], a[16]);
	VectorSort32fPacked (a[10], a[13]);
	VectorSort32fPacked (a[20], a[23]);
	VectorSort32fPacked (a[17], a[23]);
	VectorSort32fPacked (a[17], a[20]);
	VectorSort32fPacked (a[21], a[24]);
	VectorSort32fPacked (a[18], a[24]);
	VectorSort32fPacked (a[18], a[21]);
	VectorSort32fPacked (a[19], a[22]);
	VectorSort32fPacked (a[9],  a[18]);
	VectorSort32fPacked (a[0],  a[18]);
	VectorSort32fPacked (a[8],  a[17]);
	VectorSort32fPacked (a[0],  a[9]);
	VectorSort32fPacked (a[10], a[19]);
	VectorSort32fPacked (a[1],  a[19]);
	VectorSort32fPacked (a[1],  a[10]);
	VectorSort32fPacked (a[11], a[20]);
	VectorSort32fPacked (a[2],  a[20]);
	VectorSort32fPacked (a[12], a[21]);
	VectorSort32fPacked (a[2],  a[11]);
	VectorSort32fPacked (a[3],  a[21]);
	VectorSort32fPacked (a[3],  a[12]);
	VectorSort32fPacked (a[13], a[22]);
	VectorSort32fPacked (a[4],  a[22]);
	VectorSort32fPacked (a[4],  a[13]);
	VectorSort32fPacked (a[14], a[23]);
	VectorSort32fPacked (a[5],  a[23]);
	VectorSort32fPacked (a[5],  a[14]);
	VectorSort32fPacked (a[15], a[24]);
	VectorSort32fPacked (a[6],  a[24]);
	VectorSort32fPacked (a[6],  a[15]);
	VectorSort32fPacked (a[7],  a[16]);
	VectorSort32fPacked (a[7],  a[19]);
	VectorSort32fPacked (a[13], a[21]);
	VectorSort32fPacked (a[15], a[23]);
	VectorSort32fPacked (a[7],  a[13]);
	VectorSort32fPacked (a[7],  a[15]);
	VectorSort32fPacked (a[1],  a[9]);
	VectorSort32fPacked (a[3],  a[11]);
	VectorSort32fPacked (a[5],  a[17]);
	VectorSort32fPacked (a[11], a[17]);
	VectorSort32fPacked (a[9],  a[17]);
	VectorSort32fPacked (a[4],  a[10]);
	VectorSort32fPacked (a[6],  a[12]);
	VectorSort32fPacked (a[7],  a[14]);
	VectorSort32fPacked (a[4],  a[6]);
	VectorSort32fPacked (a[4],  a[7]);
	VectorSort32fPacked (a[12], a[14]);
	VectorSort32fPacked (a[10], a[14]);
	VectorSort32fPacked (a[6],  a[7]);
	VectorSort32fPacked (a[10], a[12]);
	VectorSort32fPacked (a[6],  a[10]);
	VectorSort32fPacked (a[6],  a[17]);
	VectorSort32fPacked (a[12], a[17]);
	VectorSort32fPacked (a[7],  a[17]);
	VectorSort32fPacked (a[7],  a[10]);
	VectorSort32fPacked (a[12], a[18]);
	VectorSort32fPacked (a[7],  a[12]);
	VectorSort32fPacked (a[10], a[18]);
	VectorSort32fPacked (a[12], a[20]);
	VectorSort32fPacked (a[10], a[20]);
	VectorSort32fPacked (a[10], a[12]);
}

/*
	make median filter with kernel 5x5 from packed format - BGRA444_8u by AVX2 instructions set:

	Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

	LSB                            MSB
	+-------------------------------+
	| B | G | R | A | B | G | R | A | ...
	+-------------------------------+

*/
bool AVX2::Median::median_filter_5x5_RGB_4444_32f
(
	__m128* __restrict pInImage,
	__m128* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch
) noexcept
{
	//if (sizeY < 5 || sizeX < 40)
		//		return Scalar::scalar_median_filter_5x5_BGRA_4444_8u(pInImage, pOutImage, sizeY, sizeX, linePitch);

		//	CACHE_ALIGN PF_Pixel_BGRA_8u  ScalarElem[9];
	constexpr A_long pixelsInVector{ static_cast<A_long>(sizeof(__m256) / sizeof(__m128)) };
	constexpr A_long startPosition = pixelsInVector * 2;

	A_long i, j;
	const A_long vectorLoadsInLine = sizeX / pixelsInVector;
	const A_long vectorizedLineSize = vectorLoadsInLine * pixelsInVector;
	const A_long lastPixelsInLine = sizeX - vectorizedLineSize;
	const A_long lastIdx = lastPixelsInLine - 2;

	const A_long shortSizeY { sizeY - 2 };
	const A_long shortSizeX { sizeX - pixelsInVector * 2};

	constexpr int blendMask = 0x77;

#ifdef _DEBUG
	__m256 vecData[25]{};
#else
	CACHE_ALIGN __m256 vecData[25];
#endif

	/* PROCESS FIRST LINE IN FRAME */
	{
		__m128* __restrict pSrcVecCurrLine  = pInImage;
		__m128* __restrict pSrcVecNextLine1 = (pInImage + srcLinePitch);
		__m128* __restrict pSrcVecNextLine2 = (pInImage + srcLinePitch * 2);
 		__m256* __restrict pSrcVecDstLine   = reinterpret_cast<__m256* __restrict>(pOutImage);

		/* process first pixel */
		const __m256 srcFirstPixel = LoadFirstLineWindowPixel0 (pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, vecData);
		PartialSort_9_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcFirstPixel, vecData[4], blendMask);
		pSrcVecDstLine++;

		/* process second pixel */
		const __m256 srcSecondPixel = LoadFirstLineWindowPixel1 (pSrcVecCurrLine + pixelsInVector, pSrcVecNextLine1 + pixelsInVector, pSrcVecNextLine2 + pixelsInVector, vecData);
		PartialSort_12_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcSecondPixel, vecData[5], blendMask);
		pSrcVecDstLine++;

		/* process rest of pixels */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256 srcOrig = LoadFirstLineWindowPixel (pSrcVecCurrLine + i, pSrcVecNextLine1 + i, pSrcVecNextLine2 + i, vecData);
			PartialSort_15_elem_32f (vecData);
			StoreByMask32f (pSrcVecDstLine, srcOrig, vecData[7], blendMask);
			pSrcVecDstLine++;
		} /* for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector) */

		/* process one before last pixel */
		const __m256 srcPixelBeforeLast = LoadFirstLineWindowPixelBeforeLast (pSrcVecCurrLine  + i,
			                                                                  pSrcVecNextLine1 + i,
			                                                                  pSrcVecNextLine2 + i,
			                                                                  vecData);
		PartialSort_12_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcPixelBeforeLast, vecData[5], blendMask);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process last pixel */
		const __m256 srcPixelLast = LoadFirstLineWindowPixelLast (pSrcVecCurrLine  + i,
			                                                      pSrcVecNextLine1 + i,
			                                                      pSrcVecNextLine2 + i,
			                                                      vecData);
		PartialSort_9_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcPixelLast, vecData[4], blendMask);
	}

	/* PROCESS SECOND LINE IN FRAME */
	{
		__m128* __restrict pSrcVecPrevLine  = pInImage;
		__m128* __restrict pSrcVecCurrLine  = (pInImage + srcLinePitch);
		__m128* __restrict pSrcVecNextLine1 = (pInImage + srcLinePitch * 2);
		__m128* __restrict pSrcVecNextLine2 = (pInImage + srcLinePitch * 3);
		__m256*  __restrict pSrcVecDstLine  = reinterpret_cast<__m256* __restrict>(pOutImage + dstLinePitch);

		/* process first pixel */
		const __m256 srcFirstPixel = LoadSecondLineWindowPixel0 (pSrcVecPrevLine, pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, vecData);
		PartialSort_12_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcFirstPixel, vecData[5], blendMask);
		pSrcVecDstLine++;

		/* process second pixel */
		const __m256 srcSecondPixel = LoadSecondLineWindowPixel1 (pSrcVecPrevLine  + pixelsInVector,
			                                                      pSrcVecCurrLine  + pixelsInVector,
			                                                      pSrcVecNextLine1 + pixelsInVector,
			                                                      pSrcVecNextLine2 + pixelsInVector,
			                                                      vecData);
		PartialSort_16_elem_32f(vecData);
		StoreByMask32f (pSrcVecDstLine, srcSecondPixel, vecData[7], blendMask);
		pSrcVecDstLine++;

		/* process rest of pixels */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256 srcOrig = LoadSecondLineWindowPixel (pSrcVecPrevLine + i, pSrcVecCurrLine + i, pSrcVecNextLine1 + i, pSrcVecNextLine2 + i, vecData);
			PartialSort_20_elem_32f(vecData);
			StoreByMask32f (pSrcVecDstLine, srcOrig, vecData[9], blendMask);
			pSrcVecDstLine++;
		} /* for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector) */

		/* process one before last pixel */
		const __m256 srcPixelBeforeLast = LoadSecondLineWindowPixelBeforeLast (pSrcVecPrevLine  + i,
			                                                                   pSrcVecCurrLine  + i,
			                                                                   pSrcVecNextLine1 + i,
			                                                                   pSrcVecNextLine2 + i,
			                                                                   vecData);
		PartialSort_16_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcPixelBeforeLast, vecData[7], blendMask);
		pSrcVecDstLine++;
		i += pixelsInVector;

		/* process last pixel */
		const __m256 srcPixelLast = LoadSecondLineWindowPixelLast (pSrcVecPrevLine  + i,
			                                                       pSrcVecCurrLine  + i,
			                                                       pSrcVecNextLine1 + i,
			                                                       pSrcVecNextLine2 + i,
			                                                       vecData);
		PartialSort_12_elem_32f(vecData);
		StoreByMask32f (pSrcVecDstLine, srcPixelLast, vecData[5], blendMask);
	}

	/* PROCESS REST OF LINES IN FRAME */
	{
		/* PROCESS LINES IN FRAME FROM 2 to SIZEY-2 */
		for (j = 2; j < shortSizeY; j++)
		{
			__m128* __restrict pSrcVecPrevLine2 = (pInImage + (j - 2) * srcLinePitch);
			__m128* __restrict pSrcVecPrevLine1 = (pInImage + (j - 1) * srcLinePitch);
			__m128* __restrict pSrcVecCurrLine  = (pInImage +  j      * srcLinePitch);
			__m128* __restrict pSrcVecNextLine1 = (pInImage + (j + 1) * srcLinePitch);
			__m128* __restrict pSrcVecNextLine2 = (pInImage + (j + 2) * srcLinePitch);
			__m256* __restrict pSrcVecDstLine   = reinterpret_cast<__m256* __restrict>(pOutImage + j * dstLinePitch);

			const __m256 srcFirstPixel = LoadWindowPixel0 (pSrcVecPrevLine2, pSrcVecPrevLine1, pSrcVecCurrLine, pSrcVecNextLine1, pSrcVecNextLine2, vecData);
			PartialSort_15_elem_32f (vecData);
			StoreByMask32f (pSrcVecDstLine, srcFirstPixel, vecData[7], blendMask);
			pSrcVecDstLine++;

			const __m256 srcSecondPixel = LoadWindowPixel1 (pSrcVecPrevLine2 + pixelsInVector,
				                                            pSrcVecPrevLine1 + pixelsInVector,
				                                            pSrcVecCurrLine  + pixelsInVector,
				                                            pSrcVecNextLine1 + pixelsInVector,
				                                            pSrcVecNextLine2 + pixelsInVector,
				                                            vecData);
			PartialSort_20_elem_32f (vecData);
			StoreByMask32f (pSrcVecDstLine, srcSecondPixel, vecData[9], blendMask);
			pSrcVecDstLine++;

			/* process rest of pixels */
			for (i = startPosition; i < shortSizeX; i += pixelsInVector)
			{
				const __m256 srcPixel = LoadWindowPixel(pSrcVecPrevLine2 + i, pSrcVecPrevLine1 + i, pSrcVecCurrLine + i, pSrcVecNextLine1 + i, pSrcVecNextLine2 + i, vecData);
				PartialSort_25_elem_32f (vecData);
				StoreByMask32f (pSrcVecDstLine, srcPixel, vecData[12], blendMask);
				pSrcVecDstLine++;
			} /* for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector) */

			/* process pixels in line bedofre last */
			const __m256 srcOrigRight2 = LoadWindowPixelBeforeLast (pSrcVecPrevLine2 + i,
				                                                    pSrcVecPrevLine1 + i,
				                                                    pSrcVecCurrLine  + i,
				                                                    pSrcVecNextLine1 + i,
																    pSrcVecNextLine2 + i,
				                                                    vecData);
			PartialSort_20_elem_32f (vecData);
			StoreByMask32f (pSrcVecDstLine, srcOrigRight2, vecData[9], blendMask);
			pSrcVecDstLine++;
			i += pixelsInVector;
			/* process last pixels in line */
			const __m256 srcOrigRight1 = LoadWindowPixelLast (pSrcVecPrevLine2 + i,
                                                              pSrcVecPrevLine1 + i,
				                                              pSrcVecCurrLine  + i,
                                                              pSrcVecNextLine1 + i,
                                                              pSrcVecNextLine2 + i,
				                                              vecData);
			PartialSort_15_elem_32f (vecData);
			StoreByMask32f (pSrcVecDstLine, srcOrigRight1, vecData[7], blendMask);
		} /* for (j = 2; j < shortSizeY; j++) */  
	}

	/* PROCESS LINE BEFORE LAST */
	{
		__m128* __restrict pSrcVecPrev2Line = (pInImage  + (j - 2) * srcLinePitch);
		__m128* __restrict pSrcVecPrev1Line = (pInImage  + (j - 1) * srcLinePitch);
		__m128* __restrict pSrcVecCurrLine  = (pInImage  +  j      * srcLinePitch);
		__m128* __restrict pSrcVecNextLine  = (pInImage  + (j + 1) * srcLinePitch);
		__m256* __restrict pSrcVecDstLine   = reinterpret_cast <__m256* __restrict>(pOutImage + j * dstLinePitch);

		const __m256 srcFirstPixel = LoadWindowBeforeLastLineFirstPixel (pSrcVecPrev2Line, pSrcVecPrev1Line, pSrcVecCurrLine, pSrcVecNextLine, vecData);
		PartialSort_12_elem_32f(vecData);
		StoreByMask32f (pSrcVecDstLine, srcFirstPixel, vecData[5], blendMask);
		pSrcVecDstLine++;

		const __m256 srcSecondPixel = LoadWindowBeforeLastLineSecondPixel (pSrcVecPrev2Line + pixelsInVector,
			                                                               pSrcVecPrev1Line + pixelsInVector,
			                                                               pSrcVecCurrLine  + pixelsInVector,
			                                                               pSrcVecNextLine  + pixelsInVector,
			                                                               vecData);
		PartialSort_16_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcSecondPixel, vecData[7], blendMask);
		pSrcVecDstLine++;

		/* process rest of pixels */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256 srcPixel = LoadWindowBeforeLastLine (pSrcVecPrev2Line + i, pSrcVecPrev1Line + i, pSrcVecCurrLine + i, pSrcVecNextLine + i, vecData);
			PartialSort_20_elem_32f (vecData);
			StoreByMask32f (pSrcVecDstLine, srcPixel, vecData[9], blendMask);
			pSrcVecDstLine++;
		} /* for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector) */

		const __m256 srcBeforeLastPixel = LoadWindowBeforeLastLineBeforeLastPixel (pSrcVecPrev2Line + i,
			                                                                       pSrcVecPrev1Line + i,
			                                                                       pSrcVecCurrLine  + i,
			                                                                       pSrcVecNextLine  + i,
			                                                                       vecData);
		PartialSort_16_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcBeforeLastPixel, vecData[7], blendMask);
		pSrcVecDstLine++;
		i += pixelsInVector;
		const __m256 srcLastPixel = LoadWindowBeforeLastLineLastPixel (pSrcVecPrev2Line + i,
			                                                           pSrcVecPrev1Line + i,
			                                                           pSrcVecCurrLine  + i,
			                                                           pSrcVecNextLine  + i,
			                                                           vecData);
		PartialSort_12_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcLastPixel, vecData[5], blendMask);
	}

	/* PROCESS LAST LINE */
	{
		j = j + 1;
		__m128* __restrict pSrcVecPrev2Line = (pInImage  + (j - 2) * srcLinePitch);
		__m128* __restrict pSrcVecPrev1Line = (pInImage  + (j - 1) * srcLinePitch);
		__m128* __restrict pSrcVecCurrLine  = (pInImage  +  j      * srcLinePitch);
		__m256* __restrict pSrcVecDstLine   = reinterpret_cast <__m256* __restrict>(pOutImage +  j * dstLinePitch);

		const __m256 srcFirstPixel = LoadWindowLastLineFirstPixel (pSrcVecPrev2Line, pSrcVecPrev1Line, pSrcVecCurrLine, vecData);
		PartialSort_9_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcFirstPixel, vecData[4], blendMask);
		pSrcVecDstLine++;

		const __m256 srcSecondPixel = LoadWindowLastLineSecondPixel (pSrcVecPrev2Line + pixelsInVector,
			                                                         pSrcVecPrev1Line + pixelsInVector,
			                                                         pSrcVecCurrLine  + pixelsInVector,
			                                                         vecData);
		PartialSort_12_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcSecondPixel, vecData[5], blendMask);
		pSrcVecDstLine++;

		/* process rest of pixels */
		for (i = startPosition; i < shortSizeX; i += pixelsInVector)
		{
			const __m256 srcPixel = LoadWindowLastLine (pSrcVecPrev2Line + i, pSrcVecPrev1Line + i, pSrcVecCurrLine + i, vecData);
			PartialSort_15_elem_32f (vecData);
			StoreByMask32f (pSrcVecDstLine, srcPixel, vecData[7], blendMask);
			pSrcVecDstLine++;
		} /* for (i = pixelsInVector * 2; i < shortSizeX; i += pixelsInVector) */

		const __m256 srcBeforeLastPixel = LoadWindowLastLineBeforeLastPixel (pSrcVecPrev2Line + i,
			                                                                 pSrcVecPrev1Line + i,
			                                                                 pSrcVecCurrLine  + i,
			                                                                 vecData);
		PartialSort_12_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcBeforeLastPixel, vecData[5], blendMask);
		pSrcVecDstLine++;
		i += pixelsInVector;

		const __m256 srcLastPixel = LoadWindowLastLineLastPixel (pSrcVecPrev2Line + i,
			                                                     pSrcVecPrev1Line + i,
			                                                     pSrcVecCurrLine  + i,
			                                                     vecData);
		PartialSort_9_elem_32f (vecData);
		StoreByMask32f (pSrcVecDstLine, srcLastPixel, vecData[4], blendMask);
	}

	return true;
}