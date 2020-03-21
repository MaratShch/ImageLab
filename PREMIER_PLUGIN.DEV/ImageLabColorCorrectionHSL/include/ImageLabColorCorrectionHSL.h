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

#define OFFSET_H(idx)	idx
#define OFFSET_S(idx)	(idx+1)
#define OFFSET_L(idx)	(idx+2)

template<typename T>
constexpr T MIN(const T a, const T b) { return ((a < b) ? a : b); }

template<typename T>
constexpr T MAX(const T a, const T b) { return ((a > b) ? a : b); }

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
const T CLAMP_LS(const T ls)
{
	constexpr T lsMin{ 0 };
	constexpr T lsMax{ 100 };
	return MAX(lsMin, MIN(lsMax, ls));
}

template<typename T>
const typename std::enable_if<std::is_integral<T>::value, T>::type CLAMP_RGB8(T val)
{
	return (MAX(static_cast<T>(0), MIN(val, static_cast<T>(255))));
}



typedef struct filterMemoryHandle
{
	size_t tmpBufferSizeBytes;
	void*  __restrict tmpBufferAlignedPtr;
}filterMemoryHandle;

typedef struct filterParams
{
	float hue_corse_level; /* from 0 till 359 degrees		*/
	float hue_fine_level;  /* from -10.0 till + 10.0 degrees	*/
	float saturation_level;/* from -100.0 till 100.0 */
	float luminance_level; /* from -100.0 till 100.0 */
	csSDK_uint8 compute_precise; /* 0 - fast model, !0 - precise model  */
	filterMemoryHandle* pTmpMem;
} filterParams, *filterParamsP, **filterParamsH;


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */);
PREMPLUGENTRY DllExport xFilter (short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif

filterMemoryHandle* get_tmp_memory_handler (void);
void* get_tmp_buffer (size_t* pBufBytesSize);
void  set_tmp_buffer (void* __restrict pBuffer, const size_t& bufBytesSize);

void* allocate_aligned_buffer (filterMemoryHandle* fTmpMemory, const size_t& newFrameSize);
void  free_aligned_buffer (filterMemoryHandle* fTmpMemory);

inline const float normalize_hue_wheel(const float& wheel_value);

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction(const VideoHandle theData);

bool bgr_to_hsl_precise_BGRA4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	void* __restrict tmpBuf,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch,
	const float newHue = 0.f,
	const float newLuminance = 0.f,
	const float newSaturation = 0.f
);
bool hsl_to_bgr_precise_BGRA4444_8u
(
	const csSDK_uint32* __restrict srcPix,
	const void*  __restrict tmpBuf,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32 width,
	const csSDK_int32 height,
	const csSDK_int32 linePitch
);
