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

#define CACHE_LINE  64
#define CACHE_ALIGN __declspec(align(CACHE_LINE))

#define AVX2_ALIGN __declspec(align(32))
#define AVX512_ALIGN __declspec(align(64))

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#else
#define __VECTOR_ALIGNED__
#endif

static constexpr int maxCPUCores = 64;
static constexpr int CIELabBufferbands = 3;
static constexpr size_t CIELabBufferSize = (CIELabBufferbands * 1024 * 256) * sizeof(double); /* components order: L, a, b */
static constexpr size_t CIELabBufferAlign = CACHE_LINE;

static constexpr int jobsQueueSize = 64;
static constexpr int maxRadiusSize = 10;
static constexpr double Exp = 2.718281828459;


typedef struct
{
	void*  pSlice;
	unsigned int sizeX;
	unsigned int sizeY;
}Async_Jobs;


typedef struct
{
	size_t		strSizeOf;
	DWORD       idx;
	std::atomic<int> head;
	std::atomic<int> tail;
	std::atomic<bool> mustExit;
	std::condition_variable cv;
	bool		bNewJob;
	bool        bJobComplete;
	Async_Jobs	jobsQueue[jobsQueueSize];
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
void DeleteColorConvertTable(void);

void gaussian_weights(const double sigma, const int radius = 5 /* radius size in range of 3 to 10 */);
void bilateral_filter_color(const double* __restrict pCIELab, double* __restrict pFiltered, const int sizeX, const int sizeY, const int radius, const double sigmaR);

csSDK_int32 processFrame(VideoHandle theData);

DWORD WINAPI ProcessThread(LPVOID pParam);
void createTaskServers(const unsigned int dbgLimit = UINT_MAX);
void deleteTaskServers(const unsigned int dbgLimit = UINT_MAX);
void startParallelJobs(const unsigned int dbgLimit = UINT_MAX);

void startAsyncJobs(void);
void waitForAsynJobs(void);

BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */);


#ifdef __cplusplus
}
#endif
