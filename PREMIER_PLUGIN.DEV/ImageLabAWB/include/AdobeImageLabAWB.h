#pragma once

#include "PrSDKEffect.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPixelFormatSuite.h"
#include "PrSDKSequenceInfoSuite.h"
#include "SDK_File.h"

#if !defined __INTEL_COMPILER 
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

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

template<typename T>
T CLAMP_RGB8(T val) { return ((val > 0xFF) ? 0xFF : (val < 0) ? 0 : val); }

inline double asqrt(const double& x)
{
	const double   xHalf = 0.50 * x;
	long long int  tmp = 0x5FE6EB50C7B537AAl - (*(long long int*)&x >> 1); //initial guess
	double         xRes = *(double*)&tmp;
	xRes *= (1.50 - (xHalf * xRes * xRes));
	return xRes * x;
}

inline float asqrt (const float& x)
{
	const float xHalf = 0.50f * x;
	int   tmp = 0x5F3759DF - (*(int*)&x >> 1); //initial guess
	float xRes = *(float*)&tmp;
	xRes *= (1.50f - (xHalf * xRes * xRes));
	return xRes * x;
}


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

typedef enum
{
	DAYLIGHT,
	OLD_DAYLIGHT,
	OLD_DIRECT_SUNLIGHT_AT_NOON,
	MID_MORNING_DAYLIGHT,
	NORTH_SKY_DAYLIGHT,
	DAYLIGHT_FLUORESCENT_F1,
	COOL_FLUERESCENT,
	WHITE_FLUORESCENT,
	WARM_WHITE_FLUORESCENT,
	DAYLIGHT_FLUORESCENT_F5,
	COOL_WHITE_FLUORESCENT
}eILLIUMINATE;


constexpr int minIterCount = 2;
constexpr int maxIterCount = 12;
constexpr int defIterCount = minIterCount;

constexpr int grayThresholdMin = 10;
constexpr int grayThresholdMax = 80;
constexpr int grayThresholdDef = 30;


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif

PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);

#ifdef __cplusplus
}
#endif


typedef struct
{
	// filter setting
	csSDK_int16	 sliderIterCnt;
	csSDK_int16	 sliderGrayThr;
	eILLIUMINATE illuminate;
	eSIGNAL_TYPE signalType;
}FilterParamStr, *PFilterParamStr, **FilterParamHandle;


#ifndef IMAGE_LAB_AWB_FILTER_PARAM_HANDLE_INIT
#define IMAGE_LAB_AWB_FILTER_PARAM_HANDLE_INIT(_param_handle)	\
 (*_param_handle)->sliderIterCnt = defIterCount;				\
 (*_param_handle)->sliderGrayThr = grayThresholdDef;			\
 (*_param_handle)->illuminate = DAYLIGHT;						\
 (*_param_handle)->signalType = STD_BT601;
#endif


template<typename T>
inline void copy_src2dst(
	const T* __restrict pSrc,
	T* __restrict pDst,
	const csSDK_int32& height,
	const csSDK_int32& width,
	const csSDK_int32& rowbytes
)
{
	const auto elemSize = sizeof(pSrc[0]);
	const auto linePitch = rowbytes / elemSize;
	const auto bytesSize = elemSize * width;

	for (int j = 0; j < height; j++)
	{
		__VECTOR_ALIGNED__
			memcpy(pDst, pSrc, bytesSize);

		pSrc += linePitch;
		pDst += linePitch;
	}

	return;
}

template<typename T>
inline const double get_iteration_count(const T& slider_pos)
{
	return (static_cast<double>(slider_pos) / 100.0);
}

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);
csSDK_int32 selectProcessFunction(VideoHandle theData);

bool procesBGRA4444_8u_slice(
	VideoHandle theData,
	const double* __restrict pMatrixIn,
	const double* __restrict pMatrixOut);

bool procesVUYA4444_8u_slice(VideoHandle theData,
	const double* __restrict pMatrixIn,
	const double* __restrict pMatrixOut);


const double* const GetIlluminate(const eILLIUMINATE illuminateIdx = DAYLIGHT);

void matrix_3x3_Inversion(const double* __restrict in, double* __restrict out);
