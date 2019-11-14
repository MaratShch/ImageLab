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

constexpr int maxWinSize = 11;


template<typename T>
T MIN(T a, T b) { return ((a < b) ? a : b); }

template<typename T>
T MAX(T a, T b) { return ((a > b) ? a : b); }

template<typename T>
T CLAMP_U8(T val) { return ((val > 0xFF) ? 0xFF : (val < 0) ? 0 : val); }

template<typename T>
T EXP(T val) {
	return powf(Exp, val); // powf for floating
}

inline float aExp(const float & fVal)
{
	float y = 1.0f + fVal / 1024.0f;
	y *= y; y *= y; y *= y; y *= y;
	y *= y; y *= y; y *= y; y *= y;
	y *= y; y *= y;
	return y;
}


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);
BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */);

#ifdef __cplusplus
}
#endif

void gaussian_weights(const float sigma = 3.0f, const int radius = 5);

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction (const VideoHandle theData);

bool process_VUYA_4444_8u_frame(const VideoHandle theData, const int radius = 5);