#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#include "ImageLabSorting.h"

#define CACHE_LINE  64
#define CACHE_ALIGN __declspec(align(CACHE_LINE))

#define AVX2_ALIGN __declspec(align(32))
#define AVX512_ALIGN __declspec(align(64))

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#else
#define __VECTOR_ALIGNED__
#endif

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(T x, T a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type make_odd(T x)
{
	return (x | 1);
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type kernel_width(T x)
{
	return make_odd(x * 2);
}

constexpr int MinKernelRadius = 1;
constexpr int MaxKernelRadius = 40;

constexpr int MinKernelWidth = kernel_width(MinKernelRadius);
static_assert((MinKernelWidth & 0x1), "Kernel width value must be ODD");

constexpr int MaxKernelWidth  = kernel_width(MaxKernelRadius);
static_assert((MaxKernelWidth & 0x1), "Kernel width value must be ODD");



typedef struct filterParams
{
	short int	kernelRadius;
	char		checkbox;
} filterParams, *filterParamsP, **filterParamsH;

constexpr char fuzzyAlgorithmDisabled = '\0';

#ifndef IMAGE_LAB_MEDIAN_FILTER_PARAM_HANDLE_INIT
#define IMAGE_LAB_MEDIAN_FILTER_PARAM_HANDLE_INIT(_param_handle) \
 (*_param_handle)->kernelRadius = MinKernelWidth;             \
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

csSDK_int32 selectProcessFunction (const VideoHandle theData, const bool& advFlag = false, const int32_t& kernelSize = MinKernelWidth);

bool median_filter_BGRA_4444_8u_frame (	const csSDK_uint32* __restrict srcPix,
										  	  csSDK_uint32* __restrict dstPix,
										const csSDK_int32& height,
										const csSDK_int32& width,
										const csSDK_int32& linePitch);


bool median_filter_ARGB_4444_8u_frame(const csSDK_uint32* __restrict srcPix,
											csSDK_uint32* __restrict dstPix,
									  const csSDK_int32& height,
									  const csSDK_int32& width,
									  const csSDK_int32& linePitch);
