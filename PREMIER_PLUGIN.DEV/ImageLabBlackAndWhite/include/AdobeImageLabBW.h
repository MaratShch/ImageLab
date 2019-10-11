#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#if !defined __INTEL_COMPILER 
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#ifndef FILTER_NAME_MAX_LENGTH
#define FILTER_NAME_MAX_LENGTH	32
#endif

#ifndef CACHE_LINE
#define CACHE_LINE	64
#endif

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
T CLAMP_RGB8(T val) { return ((val > 0xFF) ? 0xFF : (val < 0) ? 0 : val); }


typedef struct filterParams
{
	char		checkbox;
} filterParams, *filterParamsP, **filterParamsH;


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

csSDK_int32 selectProcessFunction(VideoHandle theData, bool advancedAlg = false);

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);

bool processBGRA4444_8u_slice(VideoHandle theData);
bool processAdvancedBGRA4444_8u_slice(VideoHandle theData);

bool processBGRA4444_16u_slice(VideoHandle theData);
bool processAdvancedBGRA4444_16u_slice(VideoHandle theData);

bool processBGRA4444_32f_slice(VideoHandle theData);

bool processVUYA_4444_8u_slice(VideoHandle theData);

void initCompCoeffcients(void);