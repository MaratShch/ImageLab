#pragma once

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
#define __ASSUME_ALIGNED(a, align_val) __assume_aligned(a, align_val)
#else
#define __VECTOR_ALIGNED__
#define __ASSUME_ALIGNED__         
#endif


typedef struct filterParams
{
	char checkbox_mirror_horizontal;
	char checkbox_mirror_vertical;
} filterParams, *filterParamsP, **filterParamsH;

constexpr csSDK_int32 mirrorNo = 0;
constexpr csSDK_int32 mirrorHorizontal = 1;
constexpr csSDK_int32 mirrorVertical = 2;
constexpr csSDK_int32 mirrorDiagonal = (mirrorVertical | mirrorHorizontal);


static inline constexpr float getFrameProprotions (const csSDK_int32& width, const csSDK_int32& height)
{
	return static_cast<float>(width) / static_cast<float>(height);
}


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);