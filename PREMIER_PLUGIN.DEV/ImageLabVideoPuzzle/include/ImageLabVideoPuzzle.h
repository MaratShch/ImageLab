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

constexpr csSDK_int16 minBlocksNumber = 2;				/* minimal blocks number per one dimension				*/
constexpr csSDK_int16 maxBlocksNumber = 16;				/* maximal blocks number per one dimension				*/
constexpr csSDK_int16 defBlocksNumber = 4;				/* maximal blocks number per one dimension				*/
constexpr csSDK_int16 minMosaicMapDuration = 20;		/* minimal number of frames for use current mosaic map	*/
constexpr csSDK_int16 defMosaicMapDuration = 200;		/* default number of frames for use current mosaic map	*/
constexpr csSDK_int16 maxMosaicMapDuration = SHRT_MAX;	/* maximal number of frame for use current mosaic map	*/
constexpr csSDK_int16 infiniteMosaicMapDuration = -1;

constexpr csSDK_int16 maxMosaicMapSize = maxBlocksNumber * maxBlocksNumber;
constexpr csSDK_int16 lineIdx = 0;
constexpr csSDK_int16 rowIdx  = 1;

typedef union mosaicMap
{
	csSDK_int32 mapElem;
	csSDK_int16 mapIdx[2];
}mosaicMap;

typedef struct filterParams
{
	// filter setting
	csSDK_int16	sliderBlocksNumber;
	csSDK_int16	sliderFrameDuration;
	csSDK_int32 frameCnt;
	mosaicMap   map[maxMosaicMapSize];
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
bool make_puzzle_map (mosaicMap* __restrict pMap, const csSDK_int16 blocksNumber);


template <typename T>
bool make_puzzle_image
(
	const T* __restrict srcPix,
	T* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const mosaicMap* __restrict pMosaic = nullptr,
	const csSDK_int16& blocksNumber = defBlocksNumber
);