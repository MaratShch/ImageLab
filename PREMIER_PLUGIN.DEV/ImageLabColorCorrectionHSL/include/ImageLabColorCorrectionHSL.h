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


typedef struct filterParams
{
	char checkbox_reflect_horizontal;
	char checkbox_reflect_vertical;
} filterParams, *filterParamsP, **filterParamsH;

constexpr csSDK_int32 reflectNo = 0;
constexpr csSDK_int32 reflectHorizontal = 1;
constexpr csSDK_int32 reflectVertical = 2;
constexpr csSDK_int32 reflectBothDirections = 3;


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction(const VideoHandle theData);

template <typename T>
bool reflect_image
(
	T* __restrict srcPix,
	T* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& reflectDirection
);
