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

template<typename T>
inline constexpr T MIN(const T a, const T b) { return ((a < b) ? a : b); }

template<typename T>
inline constexpr T MAX(const T a, const T b) { return ((a > b) ? a : b); }



template<typename T>
inline const T CLAMP_U8 (const T val)
{
	constexpr T minVal{ 0x0 };
	constexpr T maxVal{ 0xFF };
	return (MAX(MIN(val, maxVal), minVal));
}

template<typename T>
inline const T CLAMP_U16(const T val)
{
	constexpr T minVal{ 0x0 };
	constexpr T maxVal{ 0xFFFF };
	return (MAX(MIN(val, maxVal), minVal));
}

template<typename T>
inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type CLAMP_32F (const T val)
{
	constexpr T one_minus_epsilon { 1.f - (FLT_EPSILON) };
	constexpr T zero_plus_epsilon { 0.f + (FLT_EPSILON) };
	return (MAX(zero_plus_epsilon, MIN(val, one_minus_epsilon)));
}


#pragma pack(push)
#pragma pack(1)
typedef struct filterParams
{
	// filter setting
	csSDK_int16	sliderVolume;	/* Noise volume			*/
	csSDK_int8	checkColorNoise;/* use color noise		*/
	csSDK_int8	checkAlpha;		/* use Alpha channel	*/
} filterParams, *filterParamsP, **filterParamsH;
#pragma pack(pop)

#ifndef IMAGE_LAB_FILTER_PARAM_HANDLE_INIT
#define IMAGE_LAB_FILTER_PARAM_HANDLE_INIT(_param_handle) \
 (*_param_handle)->sliderVolume    = 20;                  \
 (*_param_handle)->checkColorNoise = 1;                   \
 (*_param_handle)->checkAlpha      = 0;
#endif

constexpr size_t handleSize = sizeof(filterParams);

// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported (const VideoHandle theData);
csSDK_int32 selectProcessFunction (const VideoHandle theData);
inline csSDK_uint32 romuTrio32_random (void);

void add_color_noise_BGRA4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
);
void add_bw_noise_BGRA4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
);

void add_color_noise_VUYA4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha,
	bool                  isBT709 = true
);
void add_bw_noise_VUYA4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha,
	bool                  isBT709 = true
);

void add_color_noise_VUYA4444_32f
(
	const float*  __restrict pSrc,
	      float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha,
	bool                  isBT709 = true
);
void add_bw_noise_VUYA4444_32f
(
	const float*  __restrict pSrc,
	      float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha,
	bool                  isBT709 = true
);

void add_color_noise_BGRA4444_32f
(
	const float*  __restrict pSrc,
	float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
);
void add_bw_noise_BGRA4444_32f
(
	const float*  __restrict pSrc,
	float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
);

void add_color_noise_ARGB4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
);
void add_bw_noise_ARGB4444_8u
(
	const csSDK_uint32*  __restrict pSrc,
	csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
);

void add_color_noise_ARGB4444_32f
(
	const float*  __restrict pSrc,
	float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
);
void add_bw_noise_ARGB4444_32f
(
	const float*  __restrict pSrc,
	float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const csSDK_int32&    noiseVolume,
	const csSDK_int32&    noiseOnAlpha
);
