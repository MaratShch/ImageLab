#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#include "ImageLabSorting.h"


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

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(T x, T a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type make_odd(T x)
{
	return (x | 1);
}

template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type kernel_width(T x)
{
	return make_odd(x * 2);
}

constexpr char fuzzyAlgorithmDisabled = '\0';

constexpr int MaxCpuJobs = 8;
constexpr int MinKernelRadius = 1;
constexpr int MaxKernelRadius = 40;
constexpr int FuzzyMedianRadius = MinKernelRadius;

constexpr int MinKernelWidth = kernel_width(MinKernelRadius);
static_assert((MinKernelWidth & 0x1), "Kernel width value must be ODD");

constexpr int MaxKernelWidth  = kernel_width(MaxKernelRadius);
static_assert((MaxKernelWidth & 0x1), "Kernel width value must be ODD");

typedef	uint32_t	HistElem;
constexpr csSDK_int32 sizeOfHistElem = static_cast<csSDK_int32>(sizeof(HistElem));

constexpr csSDK_int32 used_mem_size = CreateAlignment(512 * 1024, CPU_PAGE_SIZE);
constexpr csSDK_int32 size_coarse   = CreateAlignment(3 * 16 * 1024 * sizeOfHistElem, CPU_PAGE_SIZE);
constexpr csSDK_int32 size_fine     = CreateAlignment(16 * size_coarse, CPU_PAGE_SIZE);
constexpr csSDK_int32 size_mem_align = CACHE_LINE;


typedef struct mHistogram
{
	HistElem coarse[16];
	HistElem fine[16][16];
} mHistogram;

typedef struct AlgMemStorage
{
	csSDK_size_t			strSizeOf;
	csSDK_int32				stripeSize;
	csSDK_int32				stripeNum;
	// not aligned adresses
	HistElem*				pFine_addr;
	HistElem* 				pCoarse_addr;
	// manually aligned adresses
	HistElem* __restrict	pFine;
	HistElem* __restrict	pCoarse;
	CACHE_ALIGN mHistogram	h[4];
}AlgMemStorage, *AlgMemStorageP;


typedef struct filterParams
{
	csSDK_int16	kernelRadius;
	csSDK_int8	checkbox;
	AlgMemStorage AlgMemStorage;
} filterParams, *filterParamsP, **filterParamsH;


 inline void IMAGE_LAB_MEDIAN_FILTER_PARAM_HANDLE_INIT (const filterParamsH& _param_handle)
{
	memset(*_param_handle, 0, sizeof(filterParams));
    (*_param_handle)->checkbox = fuzzyAlgorithmDisabled;
	(*_param_handle)->kernelRadius = MinKernelRadius;
	(*_param_handle)->AlgMemStorage.strSizeOf = sizeof((*_param_handle)->AlgMemStorage);
	return;
}


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction(const VideoHandle theData, const csSDK_int8& advFlag, const csSDK_int16& kernelRadius, AlgMemStorage& algMemStorage);
csSDK_int32 allocate_coarse(const VideoHandle& theData, AlgMemStorage& algMemStorage);
csSDK_int32 allocate_fine  (const VideoHandle& theData, AlgMemStorage& algMemStorage);
void free_coarse(const VideoHandle& theData, AlgMemStorage& algMemStorage);
void free_fine  (const VideoHandle& theData, AlgMemStorage& algMemStorage);


bool median_filter_BGRA_4444_8u_frame (	const csSDK_uint32* __restrict srcPix,
										csSDK_uint32*       __restrict dstPix,
										const csSDK_int32& height,
										const csSDK_int32& width,
										const csSDK_int32& linePitch,
										AlgMemStorage&     algMem,
										const csSDK_int16& kernelRadius);


bool median_filter_ARGB_4444_8u_frame(const csSDK_uint32* __restrict srcPix,
									  csSDK_uint32* __restrict dstPix,
									  const csSDK_int32& height,
									  const csSDK_int32& width,
									  const csSDK_int32& linePitch,
									  AlgMemStorage&     algMem,
	                                  const csSDK_int16& kernelRadius);


bool fuzzy_median_filter_BGRA_4444_8u_frame
(
	const csSDK_uint32* __restrict srcBuf,
	csSDK_uint32*       __restrict dstBuf,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch,
	const   csSDK_int16& kernelRadius
);

bool fuzzy_median_filter_ARGB_4444_8u_frame
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32*       __restrict dstPix,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch,
	const   csSDK_int16& kernelRadius
);