#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#include <mutex>
#include <atomic>
#include <memory> 

#define CACHE_LINE  32
#define CACHE_ALIGN __declspec(align(CACHE_LINE))

#define AVX2_ALIGN __declspec(align(32))
#define AVX512_ALIGN __declspec(align(64))

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#else
#define __VECTOR_ALIGNED__
#endif

const int CIELabBufferbands = 3;
const size_t CIELabBufferSize = (CIELabBufferbands * 2048 * 1080) * sizeof(double); /* components order: L, a, b */
const size_t CIELabBufferAlign = CACHE_LINE;

const int jobsQueueSize = 64;
const int maxRadiusSize = 10;
const double Exp = 2.718281828459;

typedef struct
{
	void*  pData;
	unsigned int sizeX;
	unsigned int sizeY;
}Async_Jobs;


typedef struct
{
	size_t strSizeOf;
	std::mutex go;
	std::mutex notify;
	std::shared_ptr<double*> convertTablePtr;
	std::atomic<unsigned int> idxHead;
	std::atomic<unsigned int> idxTail;
	Async_Jobs jobsQueue[jobsQueueSize];
}AsyncQueue;


template<typename T>
T MIN(T a, T b) { return ((a < b) ? a : b); }

template<typename T>
T MAX(T a, T b) { return ((a > b) ? a : b); }

template<typename T>
T EXP(T val) {
	return pow(val, Exp); // powf for floating
}


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);
csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
void BGRA_convert_to_CIELab(const unsigned int* __restrict pBGRA, double* __restrict pCEILab, const int sampNumber);
void CIELab_convert_to_BGRA(const double* __restrict pCIELab, const unsigned int* __restrict pSrcBGRA, unsigned int* __restrict pDstBGRA, const int sampNumber);

void CreateColorConvertTable(void);
void DeleteColorConevrtTable(void);

void gaussian_weights(const double sigma, const int radius = 5 /* radius size in range of 3 to 10 */);
void bilateral_filter_color(const double* __restrict pCIELab, double* __restrict pFiltered, const int sizeX, const int sizeY, const int radius, const double sigmaR);

#ifdef __cplusplus
}
#endif
