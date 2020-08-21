#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#if !defined __INTEL_COMPILER 
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#include "ImageLabSketchPixFormat.h"
#include "ImageLabGradient.h"

#ifndef FILTER_NAME_MAX_LENGTH
#define FILTER_NAME_MAX_LENGTH	32
#endif

#ifndef CACHE_LINE
#define CACHE_LINE	64
#endif

#define CACHE_ALIGN __declspec(align(CACHE_LINE))
#define AVX2_ALIGN __declspec(align(32))
#define AVX512_ALIGN __declspec(align(64))

#ifndef CPU_PAGE_SIZE
#define CPU_PAGE_SIZE	4096
#endif

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#else
#define __VECTOR_ALIGNED__
#endif

template<typename T>
T MIN(T a, T b) { return ((a < b) ? a : b); }

template<typename T>
T MAX(T a, T b) { return ((a > b) ? a : b); }

template<typename T>
T CLAMP_RGB8(T val) { return ((val < 0) ? 0 : (val > 0xFF) ? 0xFF : val); }

template<typename T>
T CLAMP_RGB10(T val) { return ((val < 0) ? 0 : (val > 0x3FF) ? 0x3FF : val); }

template<typename T>
T CLAMP_RGB16(T val) { return ((val < 0) ? 0 : (val > 0xFFFF) ? 0xFFFF : val); }

template <typename T>
inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(T x, T a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}

typedef struct _AlgMemStorage
{
	size_t           bytesSize;
	void* __restrict pBuf1;
	void* __restrict pBuf2;
}AlgMemStorage;

typedef struct filterParams
{
	csSDK_int8 checkbox1;
	csSDK_int8 checkbox2;
	unsigned char __padding[6];
	AlgMemStorage* pAlgMem;
} filterParams, *filterParamsP, **filterParamsH;

constexpr csSDK_int32 filterParamSize = sizeof(filterParams);


// define color space conversion matrix's
CACHE_ALIGN constexpr float RGB2YUV[2][9] =
{
	// BT.601
	{
		0.299000f,  0.587000f,  0.114000f,
	   -0.168736f, -0.331264f,  0.500000f,
		0.500000f, -0.418688f, -0.081312f
	},

	// BT.709
	{
		0.212600f,   0.715200f,  0.072200f,
	   -0.114570f,  -0.385430f,  0.500000f,
		0.500000f,  -0.454150f, -0.045850f
	}
};

CACHE_ALIGN constexpr float YUV2RGB[2][9] =
{
	// BT.601
	{
		1.000000f,  0.000000f,  1.407500f,
		1.000000f, -0.344140f, -0.716900f,
		1.000000f,  1.779000f,  0.000000f
	},

	// BT.709
	{
		1.000000f,  0.00000000f,  1.5748021f,
		1.000000f, -0.18732698f, -0.4681240f,
		1.000000f,  1.85559927f,  0.0000000f
	}
};


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

AlgMemStorage* getAlgStorageStruct(void);
void algMemStorageFree   (AlgMemStorage* pAlgMemStorage);
bool algMemStorageRealloc(const csSDK_int32& width, const csSDK_int32& height, AlgMemStorage* pAlgMemStorage);

csSDK_int32 imageLabPixelFormatSupported (const VideoHandle theData);
csSDK_int32 selectProcessFunction (VideoHandle theData);

template <typename T>
void process_RGB_buffer
(
	const T*  __restrict  pSrc,
	      T*  __restrict  pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	AlgMemStorage*        pMemDesc
);
