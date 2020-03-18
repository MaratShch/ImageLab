﻿#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#define CACHE_LINE		64
#define CPU_PAGE_SIZE	4096
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
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(T x, T a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}

typedef struct filterMemoryHandle
{
	size_t tmpBufferSize;
	size_t tmpBufferSizeBytes;
	void*  tmpBufferPtr;
	void*  __restrict tmpBufferAlignedPtr;
}filterMemoryHandle;

typedef struct filterParams
{
	float hue_corse_level; /* from 0 till 359 degrees		*/
	float hue_fine_level;  /* from -5.0 till + 5.0 degrees	*/
	float saturation_level;/* from -100.0 till 100.0 */
	float luminance_level; /* from -100.0 till 100.0 */
	csSDK_uint8 compute_precise; /* 0 - fast model, !0 - precise model  */
	filterMemoryHandle tmpMem;
} filterParams, *filterParamsP, **filterParamsH;


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

bool allocate_aligned_buffer(const VideoHandle& theData, filterParamsH filtersParam, const size_t& newSize);

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction(const VideoHandle theData);

