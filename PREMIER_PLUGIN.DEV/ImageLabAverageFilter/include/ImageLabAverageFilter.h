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

constexpr int smallWindowSize = 3;
constexpr int largeWindowSize = 5;
constexpr int alg10TableSize = 65536;
constexpr int size_mem_align = CACHE_LINE;

constexpr int averageArithmetic = 1;
constexpr int averageGeometric  = 2;

template<typename T>
T MIN(T a, T b) { return ((a < b) ? a : b); }

template<typename T>
T MAX(T a, T b) { return ((a > b) ? a : b); }

template<typename T>
T CLAMP_U8(T val) { return ((val > 0xFF) ? 0xFF : val); }

template<typename T>
T CLAMP_U10(T val) { return ((val > 0x3FF) ? 0x3FF : val); }

template<typename T>
T CLAMP_U16(T val) { return ((val > 32768) ? 32768 : val); }


template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type CreateAlignment(T x, T a)
{
	return (x > 0) ? ((x + a - 1) / a * a) : a;
}

constexpr float div_on_9 = 1.0f / 9.0f;
constexpr float div_on_25 = 1.0f / 25.0f;

constexpr uint64_t magic_constant1 = 0xAAAAAAABull;
constexpr uint64_t magic_constant2 = 0x38E38E39ull;
constexpr uint64_t magic_constant3 = 0x147AE148ull;

constexpr static inline uint32_t div_by3 (const uint32_t& divideMe)
{
	return (uint32_t)((magic_constant1 * divideMe) >> 33);
}

constexpr static inline uint32_t div_by9 (const uint32_t& divideMe)
{
	return (uint32_t)((magic_constant2 * divideMe) >> 33);
}

constexpr static inline uint32_t div_by25(const uint32_t& divideMe)
{
	return (uint32_t)((magic_constant3 * divideMe) >> 33);
}


// This is a fast approximation to log2()
static inline float fast_log2f (const float& X)
{
	int E;
	const float F = frexpf(fabsf(X), &E);
	float Y = 1.23149591368684f;
	Y *= F;
	Y += -4.11852516267426f;
	Y *= F;
	Y += 6.02197014179219f;
	Y *= F;
	Y += -3.13396450166353f;
	Y += E;
	return(Y);
}

static inline float fast_log10f (const float& x)
{
	return fast_log2f(x) * 0.3010299956639812f;
}


static inline double fast_pow (const double&& a, const double&& b)
{
	union {
		double d;
		struct {
			int a;
			int b;
		} s;
	} u = { a };
	u.s.b = (int)(b * (u.s.b - 1072632447) + 1072632447);
	u.s.a = 0;
	return u.d;
}

typedef struct filterParams
{
	char checkbox_window_size; /* 3x3 if not selected or 5x5 if selected */
	char checkbox_average_type;/* arithmetic average if not selected or geometric average if selected */
	const float* __restrict pLog10TableAligned;
	size_t pLog10TableSize;
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

inline void init_log10_table (float* pTable, const int& table_size);
inline float* alocate_log10_table (const int& table_size);
inline void  free_log10_table (float* fPtr);
const float* get_log10_table_ptr (void);


csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction(const VideoHandle theData);

bool average_filter_BGRA4444_8u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_BGRA4444_8u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_BGRA4444_16u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_BGRA4444_16u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_VUYA4444_8u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_VUYA4444_8u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_BGRA4444_32f_averageArithmetic
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_BGRA4444_32f_averageGeometric
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_VUYA4444_32f_averageArithmetic
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_VUYA4444_32f_averageGeometric
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_ARGB4444_8u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_ARGB4444_8u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_ARGB4444_16u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_ARGB4444_16u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_ARGB4444_32f_averageArithmetic
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_ARGB4444_32f_averageGeometric
(
	const float* __restrict srcPix,
	float* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_RGB444_10u_averageArithmetic
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

bool average_filter_RGB444_10u_averageGeometric
(
	const csSDK_uint32* __restrict srcPix,
	csSDK_uint32* __restrict dstPix,
	const float*  __restrict fLog10Tbl,
	const csSDK_int32& width,
	const csSDK_int32& height,
	const csSDK_int32& linePitch,
	const csSDK_int32& windowSize
);

