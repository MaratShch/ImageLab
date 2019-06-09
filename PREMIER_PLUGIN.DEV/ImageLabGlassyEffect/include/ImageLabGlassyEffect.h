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

int constexpr randomBufSize = 4096;
int constexpr idxMask = (randomBufSize-1); // [0 ... 1023]

// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif


typedef struct
{
// filter setting
	csSDK_int16	sliderPosition;
// memory handler
	const float* pBufRandom;
}FilterParamStr, *PFilterParamStr, **FilterParamHandle;


PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);
csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);

BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */);

static unsigned int utils_get_random_value(void);
static void generateRandowValues(float* pBuffer, const size_t& bufSize);

const float* const get_random_values_buffer(void);
static csSDK_int32 processFrame(VideoHandle theData);


#ifdef __cplusplus
}
#endif
