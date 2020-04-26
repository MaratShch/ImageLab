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

template <typename T>
inline constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(T x, T a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}

template<typename T>
inline const T CLAMP_U8 (const T val)
{
	constexpr T minVal{ 0 };
	constexpr T maxVal{ 255 };
	return (MAX(MIN(val, maxVal), minVal));
}


template<typename T>
inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type CLAMP_32F (const T val)
{
	constexpr T one_minus_epsilon { 1.f - (FLT_EPSILON) };
	constexpr T zero_plus_epsilon { 0.f + (FLT_EPSILON) };
	return (MAX(zero_plus_epsilon, MIN(val, one_minus_epsilon)));
}


typedef struct AlgMemStorage
{
	size_t memBytesSize;
	void* __restrict pTmp1;
	void* __restrict pTmp2;
}AlgMemStorage;


#pragma pack(push)
#pragma pack(1)
typedef struct filterParams
{
	// filter setting
	csSDK_int16	sliderLevelDispersion;	/* t			*/
	csSDK_int16	sliderTimeStep;			/* deltaT		*/
	csSDK_int16	sliderNoiseLevel;		/* k			*/	
	AlgMemStorage memStorage;
} filterParams, *filterParamsP, **filterParamsH;
#pragma pack(pop)

#ifndef IMAGE_LAB_FILTER_PARAM_HANDLE_INIT
#define IMAGE_LAB_FILTER_PARAM_HANDLE_INIT(_param_handle)	\
 (*_param_handle)->sliderLevelDispersion = 2;			    \
 (*_param_handle)->sliderTimeStep = 5;                      \
 (*_param_handle)->sliderNoiseLevel = 1;                    \
 (*_param_handle)->memStorage = getAlgStorageStruct();
#endif


constexpr size_t memStorSize = sizeof(AlgMemStorage);
constexpr size_t handleSize = sizeof(filterParams);

// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

BOOL APIENTRY DllMain (HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */);
PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported (const VideoHandle theData);

void algMemStorageFree (AlgMemStorage& algMemStorage);
bool algMemStorageRealloc (const csSDK_int32& width, const csSDK_int32& height, AlgMemStorage& algMemStorage);
AlgMemStorage& getAlgStorageStruct(void);
void setAlgStorageStruct(const AlgMemStorage& storage);

void process_VUYA_4444_8u_buffer
(
	const csSDK_uint32*  __restrict pSrc,
	const AlgMemStorage* __restrict pTmpBuffers,
	csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    heigth,
	const csSDK_int32&    linePitch,
	const float&          dispersion,
	const float&          timeStep,
	const float&          noiseLevel
);

void process_BGRA_4444_8u_buffer
(
	const csSDK_uint32*  __restrict pSrc,
	const AlgMemStorage* __restrict pTmpBuffers,
	csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          dispersion,
	const float&          timeStep,
	const float&          noiseLevel
);

void process_VUYA_4444_32f_buffer
(
	const float*  __restrict pSrc,
	const AlgMemStorage* __restrict pTmpBuffers,
	float*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          dispersion,
	const float&          timeStep,
	const float&          noiseLevel
);

void process_BGRA_4444_16u_buffer
(
	const csSDK_uint32*  __restrict pSrc,
	const AlgMemStorage* __restrict pTmpBuffers,
	      csSDK_uint32*  __restrict pDst,
	const csSDK_int32&    width,
	const csSDK_int32&    height,
	const csSDK_int32&    linePitch,
	const float&          dispersion,
	const float&          timeStep,
	const float&          noiseLevel
);
