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

#define OFFSET_H(idx)	(idx)
#define OFFSET_S(idx)	(idx+1)
#define OFFSET_V(idx)	(idx+2)

template<typename T>
const T CLAMP_H(const T hue)
{
	constexpr T hueMin{ 0 };
	constexpr T hueMax{ 360 };

	if (hue < hueMin)
		return (hue + hueMax);
	else if (hue >= hueMax)
		return (hue - hueMax);
	return hue;
}

template<typename T>
const T CLAMP_SV(const T ls)
{
	constexpr T lsMin{ 0 };
	constexpr T lsMax{ 1 };
	return MAX(lsMin, MIN(lsMax, ls));
}

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

constexpr int fuzzyAlgorithmDisabled = 0;
constexpr int fuzzyAlgorithmEnabled  = 1;

constexpr int MaxCpuJobs = 8;
constexpr int MinKernelRadius = 1;
constexpr int MaxKernelRadius = 40;
constexpr int FuzzyMedianRadius = MinKernelRadius;

constexpr int MinKernelWidth = kernel_width(MinKernelRadius);
static_assert((MinKernelWidth & 0x1), "Kernel width value must be ODD");

constexpr int MaxKernelWidth  = kernel_width(MaxKernelRadius);
static_assert((MaxKernelWidth & 0x1), "Kernel width value must be ODD");

typedef	uint16_t	HistElem;
constexpr csSDK_int32 sizeOfHistElem = static_cast<csSDK_int32>(sizeof(HistElem));

template<typename T>
T CLAMP_RGB8(T val)
{
	return (MAX(static_cast<T>(0), MIN(val, static_cast<T>(255))));
}

typedef struct mHistogram
{
	HistElem coarse[16];
	HistElem fine[16][16];
} mHistogram;

typedef struct HistogramObj
{
	mHistogram h[4];
} HistogramObj;

typedef struct AlgMemStorage
{
	csSDK_size_t				strSizeOf;
	csSDK_int32					stripeSize;
	csSDK_int32					stripeNum;
	csSDK_size_t				memSize;
	void*     __restrict		pFuzzyBuffer;
	// aligned adresses for histogram buffers
	HistElem* __restrict		pFine;
	HistElem* __restrict		pCoarse;
	HistogramObj* __restrict	pH;
}AlgMemStorage, *AlgMemStorageP;

constexpr csSDK_int32 used_mem_size = CreateAlignment(512 * 1024, CPU_PAGE_SIZE);
constexpr csSDK_int32 size_coarse = CreateAlignment(3 * 16 * 1024 * sizeOfHistElem, CPU_PAGE_SIZE);
constexpr csSDK_int32 size_fine = CreateAlignment(16 * size_coarse, CPU_PAGE_SIZE);
constexpr csSDK_int32 size_mem_align = CACHE_LINE;
constexpr csSDK_int32 size_hist_obj = CreateAlignment(static_cast<csSDK_int32>(sizeof(HistogramObj)), CACHE_LINE);
constexpr csSDK_int32 size_total_hist_buffers = CreateAlignment(size_coarse + size_fine + size_hist_obj, CPU_PAGE_SIZE);
constexpr csSDK_int32 size_fuzzy_pixel = sizeof(float) * 3; /* number of bands - HSV */

AlgMemStorage& getAlgStorageStruct(void);
void setAlgStorageStruct(const AlgMemStorage& storage);
void algMemStorageFree (AlgMemStorage& algMemStorage);
bool algMemStorageRealloc (const csSDK_int32& width, const csSDK_int32& height, AlgMemStorage& algMemStorage);

#pragma pack(push)
#pragma pack(1)
typedef struct filterParams
{
	char		checkbox;
	csSDK_int16	kernelRadius;
	AlgMemStorage AlgMemStorage;
} filterParams, *filterParamsP, **filterParamsH;
#pragma pack(pop)

constexpr size_t hndlSize = sizeof(filterParams);


#define IMAGE_LAB_MEDIAN_FILTER_PARAM_HANDLE_INIT(_param_handle)	\
    (*_param_handle)->checkbox = '\0';								\
	(*_param_handle)->kernelRadius = 1;								
//	(*_param_handle)->AlgMemStorage = getAlgStorageStruct();

  
// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

BOOL APIENTRY DllMain (HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */);
PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction(const VideoHandle theData, const csSDK_int16 advFlag, const csSDK_int16 kernelRadius, AlgMemStorage& algMemStorage);

/* ================== HISTOGRAM BASED MEDIAN FILTER ========================== */
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

/* ================== TRIVIAL MEDIAN FILTER (for SMALL KERNEL) ========================== */
bool median_filter_3x3_BGRA_4444_8u_frame
(
	const	csSDK_uint32* __restrict srcBuf,
	        csSDK_uint32* __restrict dstBuf,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch
);
bool median_filter_3x3_VUYA_4444_8u_frame
(
	const	csSDK_uint32* __restrict srcBuf,
	        csSDK_uint32* __restrict dstBuf,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch
);

/* ================== FUZZY BASED MEDIAN FILTER (with Kernel 3x3 ========================== */
bool fuzzy_median_filter_BGRA_4444_8u_frame
(
	const csSDK_uint32* __restrict srcBuf,
	csSDK_uint32*       __restrict dstBuf,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch,
	const AlgMemStorage& algMem
);

bool fuzzy_median_filter_ARGB_4444_8u_frame
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32*       __restrict dstPix,
	const	csSDK_int32& height,
	const	csSDK_int32& width,
	const	csSDK_int32& linePitch,
	const AlgMemStorage& algMem
);

void fuzzy_filter_median_3x3
(
	float* __restrict	pBuffer,
	const  csSDK_int32&	width,
	const  csSDK_int32& height
);
inline float get_matrix_std
(
	const float* __restrict pBuffer,	/* buffer pointer								*/
	const csSDK_int32& winSize,			/* size of MATRIX'  window						*/
	const csSDK_int32& winPitch			/* pitch for get next element from next line	*/
);