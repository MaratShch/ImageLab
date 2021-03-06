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
constexpr T MIN(const T a, const T b) { return ((a < b) ? a : b); }

template<typename T>
constexpr T MAX(const T a, const T b) { return ((a > b) ? a : b); }

constexpr csSDK_int16 minBlocksNumber = 2;				/* minimal blocks number per one dimension				*/
constexpr csSDK_int16 maxBlocksNumber = 16;				/* maximal blocks number per one dimension				*/
constexpr csSDK_int16 defBlocksNumber = 4;				/* default blocks number per one dimension				*/
constexpr csSDK_int16 minMosaicMapDuration = 2;		    /* minimal number of frames for use current mosaic map	*/
constexpr csSDK_int16 defMosaicMapDuration = 10;		/* default number of frames for use current mosaic map	*/
constexpr csSDK_int32 durationCoeff = 300;				/* maximal number of frame for use current mosaic map	*/
constexpr csSDK_int16 infiniteMosaicMapDuration = -1;

constexpr csSDK_int16 maxMosaicMapSize = maxBlocksNumber * maxBlocksNumber;
constexpr csSDK_int16 lineIdx = 0;
constexpr csSDK_int16 rowIdx  = 1;


typedef struct filterParams
{
	// filter setting
	csSDK_int16	sliderBlocksNumber;
	csSDK_int16	sliderFrameDuration;
	csSDK_int32 frameCnt;
	csSDK_int32 currentBlocksNumber;
	csSDK_int16 map[maxMosaicMapSize];
} filterParams, *filterParamsP, **filterParamsH;



// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported (const VideoHandle theData);
bool make_puzzle_map(csSDK_int16* __restrict pMap, const csSDK_int16 blocksNumber);


template <typename T>
bool make_puzzle_image
(
	const T* __restrict srcPix,
	T* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int16* __restrict pMosaic = nullptr,
	const csSDK_int16& blocksNumber = defBlocksNumber
);