#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#ifndef FILTER_NAME_MAX_LENGTH
#define FILTER_NAME_MAX_LENGTH	32
#endif

#ifndef CACHE_LINE
#define CACHE_LINE	64
#endif

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


inline double asqrt(const double &x)
{
	double         xHalf = 0.50 * x;
	long long int  tmp = 0x5FE6EB50C7B537AAl - (*(long long int*)&x >> 1); //initial guess
	double         xRes = *(double*)&tmp;
	xRes *= (1.5 - (xHalf * xRes * xRes));
	return xRes * x;
}

inline float asqrtF (const float& x)
{
	float xHalf = 0.5f * x;
	int   tmp = 0x5F3759DF - (*(int*)&x >> 1); //initial guess
	float xRes = *(float*)&tmp;

	xRes *= (1.5f - (xHalf * xRes * xRes));
	return xRes * x;
}


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
	STD_BT601,
	STD_BT709,
	STD_BT2020,
	STD_SMPTE,
	LAST
}eSIGNAL_TYPE;

constexpr char strSignalType[][8] = 
{
	"BT.601",
	"BT.709",
	"BT.2020",
	"SMPTE"
};

constexpr int TemporarySize = 1024;
constexpr int maxIterCount = 32;

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);
csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction(VideoHandle theData);

bool procesBGRA4444_8u_slice(
	VideoHandle theData,
	const double* __restrict pMatrixIn,
	const double* __restrict pMatrixOut, 
	const int iterCnt = 10);



#ifdef __cplusplus
}
#endif
