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

constexpr int smallWindowSize = 3;
constexpr int largeWindowSize = 5;
constexpr int alg10TableSize = 65536;
constexpr csSDK_int32 size_mem_align = CACHE_LINE;

template<typename T>
T MIN(T a, T b) { return ((a < b) ? a : b); }

template<typename T>
T MAX(T a, T b) { return ((a > b) ? a : b); }

template<typename T>
T CLAMP_U8(T val) { return ((val > 0xFF) ? 0xFF : val); }

template<typename T>
T CLAMP_U10(T val) { return ((val > 0x3FF) ? 0x3FF : val); }

template<typename T>
T CLAMP_U16(T val) { return ((val > 32768) ? 32768 : val); }


template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(T x, T a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}


typedef struct filterParams
{
	char checkbox_window_size; /* 3x3 if not selected or 5x5 if selected */
	char chackbox_average_type;/* arithmetic average if not selected or geometric average if selected */
	float* __restrict pLog10Table;
	float* __restrict pLog10TableAligned;
	size_t pLog10TableSize;
} filterParams, *filterParamsP, **filterParamsH;


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

void init_1og10_table(float* pTable, int table_size);
float* allocate_aligned_log_table(const VideoHandle& theData, filterParamsH filtersParam);
csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction(const VideoHandle theData);
