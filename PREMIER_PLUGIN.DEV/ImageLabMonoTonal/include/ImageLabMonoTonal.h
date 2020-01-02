#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#undef IMAGE_LAB_FILTER_PARAM_HANDLE_INIT

#define CACHE_LINE  64
#define CACHE_ALIGN __declspec(align(CACHE_LINE))

#define AVX2_ALIGN __declspec(align(32))
#define AVX512_ALIGN __declspec(align(64))

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#else
#define __VECTOR_ALIGNED__
#endif

typedef enum
{
	convertBT601,
	convertBT709
}CONVERT_MATRIX;

template<typename T>
T CLAMP_RGB8(T val) { return ((val > 0xFF) ? 0xFF : (val < 0) ? 0 : val); }

template<typename T>
T CLAMP_RGB10(T val) { return ((val > 0x3FF) ? 0x3FF : (val < 0) ? 0 : val); }

template<typename T>
T CLAMP_RGB16(T val) { return ((val > 0xFFFF) ? 0xFFFF : (val < 0) ? 0 : val); }


typedef struct FilterParams
{
	prColor		Color;
	uint32_t    isInitialized;
}SFilterParams, *PFilterParams, **FilterParamsHandle;

constexpr prColor nonInitializedColor = 0x0u;

#ifndef IMAGE_LAB_FILTER_PARAM_HANDLE_INIT
#define IMAGE_LAB_FILTER_PARAM_HANDLE_INIT(_param_handle)		\
 (*_param_handle)->Color = nonInitializedColor;					\
 (*_param_handle)->isInitialized = 0u;
#endif

// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported (const VideoHandle theData);
csSDK_int32 selectProcessFunction (const VideoHandle theData);

bool copy_4444_8u_frame (const VideoHandle theData);
bool copy_4444_16u_frame(const VideoHandle theData);
bool copy_4444_32f_frame(const VideoHandle theData);

bool process_VUYA_4444_8u_frame (const VideoHandle theData, const prColor color = nonInitializedColor, const CONVERT_MATRIX convertMatrix = convertBT601);
bool process_VUYA_4444_32f_frame(const VideoHandle theData, const prColor color = nonInitializedColor, const CONVERT_MATRIX convertMatrix = convertBT601);

bool process_BGRA_4444_8u_frame (const VideoHandle theData, const prColor color = nonInitializedColor);
bool process_BGRA_4444_16u_frame(const VideoHandle theData, const prColor color = nonInitializedColor);
bool process_BGRA_4444_32f_frame(const VideoHandle theData, const prColor color = nonInitializedColor);

bool process_ARGB_4444_8u_frame (const VideoHandle theData, const prColor color = nonInitializedColor);
bool process_ARGB_4444_16u_frame(const VideoHandle theData, const prColor color = nonInitializedColor);
bool process_ARGB_4444_32f_frame(const VideoHandle theData, const prColor color = nonInitializedColor);

bool process_RGB_444_10u_frame  (const VideoHandle theData, const prColor color = nonInitializedColor);
