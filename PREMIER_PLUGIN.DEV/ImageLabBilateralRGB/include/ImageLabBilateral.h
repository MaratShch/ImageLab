#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#include <windows.h>
#include <mutex>
#include <atomic>
#include <memory> 

#define CACHE_LINE  32
#define CACHE_ALIGN __declspec(align(CACHE_LINE))

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#else
#define __VECTOR_ALIGNED__
#endif

const int CIELabBufferbands = 3;
const size_t CIELabBufferSize = CIELabBufferbands * sizeof(double) * 1024 * 128;
const size_t CIELabBufferAlign = CACHE_LINE;

const size_t jobsQueueSize = 64;

typedef struct
{
	size_t dataSize;
	void*  pData;
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


void CreateColorConvertTable(void);
void DeleteColorConevrtTable(void);


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);
csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
void BGRA_convert_to_CIELab(const unsigned int* __restrict pBGRA, const double* __restrict pTable, double* __restrict pCEILab, const int& sampNumber);
void CIELab_convert_to_BGRA(const double* __restrict pCIELab, const unsigned int* __restrict pSrcBGRA, unsigned int* __restrict pDstBGRA, const int& sampNumber);

#ifdef __cplusplus
}
#endif
