#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"


#define CACHE_LINE  64
#define CACHE_ALIGN __declspec(align(CACHE_LINE))

#define AVX2_ALIGN __declspec(align(32))
#define AVX512_ALIGN __declspec(align(64))

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#else
#define __VECTOR_ALIGNED__
#endif

template<typename T>
T MIN(T a, T b) { return ((a < b) ? a : b); }

template<typename T>
T MAX(T a, T b) { return ((a > b) ? a : b); }

template<typename T>
T GET_WINDOW_SIZE_FROM_SLIDER(T slider_pos) { return slider_pos + static_cast<T>(3); }

// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

constexpr int histSize = 256;
constexpr int histSizeBytes = sizeof(short int) * histSize;


typedef struct
{
// filter setting
	csSDK_int16	sliderPosition;
}FilterParamStr, *PFilterParamStr, **FilterParamHandle;


PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);
csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);

BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */);
csSDK_int32 processFrame(VideoHandle theData);

void processDataSlice(
	const csSDK_uint32* __restrict srcImage,
	csSDK_uint32*       __restrict dstImage,
	short int*	        __restrict rHist,
	short int*	        __restrict gHist,
	short int*	        __restrict bHist,
	const int                      width,
	const int                      height,
	const int                      linePitch,
	const int                      windowSize);

#ifdef __cplusplus
}
#endif
