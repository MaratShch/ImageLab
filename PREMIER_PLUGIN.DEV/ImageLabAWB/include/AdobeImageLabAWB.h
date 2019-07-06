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

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);
csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction(VideoHandle theData);

bool procesBGRA4444_8u_slice(VideoHandle theData);


#ifdef __cplusplus
}
#endif
