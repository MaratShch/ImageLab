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
	T tmpVar = std::move(a);
	a = std::move(b);
	b = std::move(tmpVar);
}

template <typename T>
inline void swapEx(T& a, T& b) // a != b && a , b = integral types
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
		else swap(*(i - 1), *i), i--;
	}
}


template <typename T>
inline void selectionsort (T* l, T* r) {
	for (T* i = l; i < r; i++) {
		T minz = *i, *ind = i;
		for (T* j = i + 1; j < r; j++) {
			if (*j < minz) minz = *j, ind = j;
		}
		swap(*i, *ind);
	}
}

constexpr int MaxKernelWidth = 13;
constexpr int MaxKernelElemSize = MaxKernelWidth * MaxKernelWidth;

typedef struct filterParams
{
	short int	kernelSize;
	char		checkbox;
} filterParams, *filterParamsP, **filterParamsH;

constexpr unsigned short int kernelSizeDefault = 3u;
constexpr char fuzzyAlgorithmDisabled = '\0';

#ifndef IMAGE_LAB_MEDIAN_FILTER_PARAM_HANDLE_INIT
#define IMAGE_LAB_MEDIAN_FILTER_PARAM_HANDLE_INIT(_param_handle) \
 (*_param_handle)->kernelSize = kernelSizeDefault;               \
 (*_param_handle)->checkbox =  fuzzyAlgorithmDisabled;
#endif


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);

csSDK_int32 selectProcessFunction(const VideoHandle theData, const bool& advFlag = false, const int32_t& kernelSize = kernelSizeDefault);
bool median_filter_BGRA_4444_8u_frame (const VideoHandle theData, const csSDK_int32& kernelWidth = kernelSizeDefault);
bool median_filter_VUYA_4444_8u_frame (const VideoHandle theData, const csSDK_int32& kernelWidth = kernelSizeDefault);
