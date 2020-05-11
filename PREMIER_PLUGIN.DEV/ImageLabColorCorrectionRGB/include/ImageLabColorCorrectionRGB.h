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
#else
#define __VECTOR_ALIGNED__
#endif

template<typename T>
constexpr T MIN(const T a, const T b) { return ((a < b) ? a : b); }

template<typename T>
constexpr T MAX(const T a, const T b) { return ((a > b) ? a : b); }


template<typename T>
T CLAMP_RGB8(T val)
{
	return (MAX(static_cast<T>(0), MIN(val, static_cast<T>(255))));
}

template<typename T>
T CLAMP_RGB10(T val)
{
	return (MAX(static_cast<T>(0), MIN(val, static_cast<T>(1023))));
}

template<typename T>
T CLAMP_RGB16(T val)
{
	return (MAX(static_cast<T>(0), MIN(val, static_cast<T>(0xFFFF))));
}

constexpr float one_minus_epsilon = 1.0f - (FLT_EPSILON);
constexpr float zero_plus_epsilon = (FLT_EPSILON);

template<typename T>
const typename std::enable_if<std::is_floating_point<T>::value, T>::type CLAMP_RGB32F(T val)
{
	return (MAX(0.0f, MIN(val, one_minus_epsilon)));
}



typedef struct filterParams
{
	csSDK_int16 R;	/* from -100.0 till 100.0 */
	csSDK_int16 G;	/* from -100.0 till 100.0 */
	csSDK_int16 B;	/* from -100.0 till 100.0 */
} filterParams, *filterParamsP, **filterParamsH;


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

void RGB_Correction_BGRA4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	      csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB
);

void RGB_Correction_BGRA4444_16u
(
	const csSDK_uint32* __restrict srcPix,
	      csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB
);

void RGB_Correction_BGRA4444_32f
(
	const float* __restrict srcPix,
	      float* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB
);

void RGB_Correction_ARGB4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	      csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB
);

void RGB_Correction_ARGB4444_16u
(
	const csSDK_uint32* __restrict srcPix,
	      csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB
);

void RGB_Correction_ARGB4444_32f
(
	const float* __restrict srcPix,
	      float* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB
);

void RGB_Correction_VUYA4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	      csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB,
	const csSDK_int32 isBT709
);

void RGB_Correction_VUYA4444_32f
(
	const float* __restrict srcPix,
     	  float* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB,
	const csSDK_int32 isBT709
);

void RGB_Correction_RGB444_10u
(
	const csSDK_uint32* __restrict srcPix,
	      csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const csSDK_int16 addR,
	const csSDK_int16 addG,
	const csSDK_int16 addB
);