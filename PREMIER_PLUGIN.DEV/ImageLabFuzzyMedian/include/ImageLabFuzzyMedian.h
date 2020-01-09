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

template <typename T>
inline void swap(T& a, T& b)
{
	const T tmpVar = a;
	a = b;
	b = tmpVar;
}

template <typename T>
inline void swapEx(T& a, T& b)
{
	a = a ^ b;
	b = a ^ b;
	a = a ^ b;
}

template <typename T>
inline void gnomesort (T* l, T* r) {
	T* i = l;
	while (i < r) {
		if (i == l || *(i - 1) <= *i) i++;
		else swapEx(*(i - 1), *i), i--;
	}
}


constexpr int MaxKernelWidth = 15;
constexpr int MaxKernelElemSize = MaxKernelWidth * MaxKernelWidth;

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

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);

csSDK_int32 selectProcessFunction(const VideoHandle theData, const bool& advFlag = false);
bool median_filter_BGRA_4444_8u_frame (const VideoHandle theData, const csSDK_int32& kernelWidth = 3);