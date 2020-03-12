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

#define AVX2_ALIGN __declspec(align(32))
#define AVX512_ALIGN __declspec(align(64))

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#else
#define __VECTOR_ALIGNED__
#endif

#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) ((a) < (b) ? (a) : (b))
#endif

#define IMAGE_LAB_HDR_THRESHOLD_MIN	    0
#define IMAGE_LAB_HDR_THRESHOLW_MAX		20
#define IMAGE_LAB_HDR_THRESHOLD_DEFAULT	1

#define IMAGE_LAB_MAX_PIXEL_SIZE_BITS	8

enum
{
	IMAGE_LAB_HDR_EQUALIZATION_INVALID = 0,
	IMAGE_LAB_HDR_EQUALIZATION_LINEAR,
	IMAGE_LAB_HDR_EQUALIZATION_LINEAR_NEG,
	IMAGE_LAB_HDR_EQUALIZATION_SINUS,
	IMAGE_LAB_HDR_EQUALIZATION_EXPONENT,
	IMAGE_LAB_HDR_EQUALIZTION_TOTAL_TYPES
};

#define IMAGE_LAB_EQUALIZATION_DEFAULT		IMAGE_LAB_HDR_EQUALIZATION_LINEAR
#define IMAGE_LAB_HISTAVERAGE_DEPTH_DEFAULT	1
#define IMAGE_LAB_HIST_AVERAGE_DEPTH_MAX	30


#if(IMAGE_LAB_MAX_PIXEL_SIZE_BITS <= 8)
#define IMAGE_LAB_HIST_BUFFER_SIZE			(256 * sizeof(int))
#define IMAGE_LAB_BIN_BUFFER_SIZE			(256 * sizeof(byte))
#define IMAGE_LAB_CUMSUM_BUFFER_SIZE		(256 * sizeof(byte))
#define IMAGE_LAB_LUT_BUFFER_SIZE			(256 * sizeof(byte))
#else
#define IMAGE_LAB_HIST_BUFFER_SIZE			(65536 * sizeof(int))
#define IMAGE_LAB_BIN_BUFFER_SIZE			(65536 * sizeof(byte))
#define IMAGE_LAB_CUMSUM_BUFFER_SIZE		(65536 * sizeof(unsigned short))
#define IMAGE_LAB_LUT_BUFFER_SIZE			(65536 * sizeof(unsigned short))
#endif


typedef struct
{
	// filter setting
	csSDK_int16	sliderLeft;
	csSDK_int16	sliderRight;
	// memory handler
	void* pMemHandler;
}FilterParamStr, *PFilterParamStr, **FilterParamHandle;

typedef struct
{
	size_t strSizeoF;
	size_t parallel_streams;
	void* __restrict pBufPoolHistogram;
	void* __restrict pBufPoolBinary;
	void* __restrict pBufCumSum;
	void* __restrict pBufPoolLUT;
}ImageLAB_MemStr, *PImageLAB_MemStr, **FilterMemHandle;


#ifndef IMAGE_LAB_HDR_STR_PARAM_INIT
#define IMAGE_LAB_HDR_STR_PARAM_INIT(_param_str)						\
 _param_str.sliderLeft = 2;										     	\
 _param_str.sliderRight = 2;											\
 _param_str.pMemHandler = nullptr;
#endif

#ifndef IMAGE_LAB_HDR_PSTR_PARAM_INIT
#define IMAGE_LAB_HDR_PSTR_PARAM_INIT(_param_str_ptr)						\
 _param_str_ptr->sliderLeft = 2;											\
 _param_str_ptr->sliderRight = 2;                                           \
 _param_str_ptr->pMemHandler = nullptr;										
#endif


#ifndef IMAGE_LAB_FILTER_PARAM_HANDLE_INIT
#define IMAGE_LAB_FILTER_PARAM_HANDLE_INIT(_param_handle)					\
 (*_param_handle)->sliderLeft = 2;									        \
 (*_param_handle)->sliderRight = 2;                                         \
 (*_param_handle)->pMemHandler = nullptr;									
#endif


// Declare plug-in entry point with C linkage
#ifdef __cplusplus
extern "C" {
#endif
	PREMPLUGENTRY DllExport xFilter(short selector, VideoHandle theData);
#ifdef __cplusplus
}
#endif

csSDK_int32 imageLabPixelFormatSupported(const VideoHandle theData);

void* GetStreamMemory(void);
void* GetHistogramBuffer(void);
void* GetBinarizationBuffer(void);
void* GetCumSumBuffer(void);
void* GetLUTBuffer(void);


