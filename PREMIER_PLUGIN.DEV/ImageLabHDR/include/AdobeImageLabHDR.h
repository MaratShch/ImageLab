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


#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) ((a) < (b) ? (a) : (b))
#endif

#define IMAGE_LAB_CPU_ONLY	0
#define IMAGE_LAB_GPU_ONLY	1
#define IMAGE_LAB_CPU_GPU	2

#define IMAGE_LAB_MAX_IMAGE_WIDTH	4096
#define IMAGE_LAB_MAX_IMAGE_HEIGHT	3072

#define IMAGE_LAB_HDR_THRESHOLD_MIN	    0
#define IMAGE_LAB_HDR_THRESHOLW_MAX		20
#define IMAGE_LAB_HDR_THRESHOLD_DEFAULT	1

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

typedef struct
{
	size_t	strSizeOf;
	// configuration setting
	int cudaEnabled;
	int maxImageWidth;
	int maxImageHeight;
	// filter setting
	int thresholdLow;
	int thresholdHigh;
	int equalizationFunction;
	int histogramAverageDepth;
}FilterParamStr, *PFilterParamStr, **FilterParamHandle;

#ifndef IMAGE_LAB_HDR_STR_PARAM_INIT
#define IMAGE_LAB_HDR_STR_PARAM_INIT(_param_str)						\
 _param_str.strSizeOf = sizeof(_param_str);								\
 _param_str.cudaEnabled = IMAGE_LAB_CPU_ONLY;							\
 _param_str.maxImageWidth = IMAGE_LAB_MAX_IMAGE_WIDTH;					\
 _param_str.maxImageHeight = IMAGE_LAB_MAX_IMAGE_HEIGHT;				\
 _param_str.thresholdLow = IMAGE_LAB_HDR_THRESHOLD_DEFAULT;				\
 _param_str.thresholdHigh = IMAGE_LAB_HDR_THRESHOLD_MIN;				\
 _param_str.equalizationFunction = IMAGE_LAB_EQUALIZATION_DEFAULT;		\
 _param_str.histogramAverageDepth = IMAGE_LAB_HISTAVERAGE_DEPTH_DEFAULT;
#endif

#ifndef IMAGE_LAB_HDR_PSTR_PARAM_INIT
#define IMAGE_LAB_HDR_PSTR_PARAM_INIT(_param_str_ptr)						\
 _param_str_ptr->strSizeOf = sizeof(* _param_str_ptr);						\
 _param_str_ptr->cudaEnabled = IMAGE_LAB_CPU_ONLY;							\
 _param_str_ptr->maxImageWidth = IMAGE_LAB_MAX_IMAGE_WIDTH;					\
 _param_str_ptr->maxImageHeight = IMAGE_LAB_MAX_IMAGE_HEIGHT;				\
 _param_str_ptr->thresholdLow = IMAGE_LAB_HDR_THRESHOLD_DEFAULT;			\
 _param_str_ptr->thresholdHigh = IMAGE_LAB_HDR_THRESHOLD_MIN;				\
 _param_str_ptr->equalizationFunction = IMAGE_LAB_EQUALIZATION_DEFAULT;		\
 _param_str_ptr->histogramAverageDepth = IMAGE_LAB_HISTAVERAGE_DEPTH_DEFAULT;
#endif


#ifndef IMAGE_LAB_FILTER_PARAM_HANDLE_INIT
#define IMAGE_LAB_FILTER_PARAM_HANDLE_INIT(_param_handle)					\
 (*_param_handle)->strSizeOf = sizeof(FilterParamStr);						\
 (*_param_handle)->cudaEnabled = IMAGE_LAB_CPU_ONLY;						\
 (*_param_handle)->maxImageWidth = IMAGE_LAB_MAX_IMAGE_WIDTH;				\
 (*_param_handle)->maxImageHeight = IMAGE_LAB_MAX_IMAGE_HEIGHT;				\
 (*_param_handle)->thresholdLow = IMAGE_LAB_HDR_THRESHOLD_DEFAULT;			\
 (*_param_handle)->thresholdHigh = IMAGE_LAB_HDR_THRESHOLD_MIN;				\
 (*_param_handle)->equalizationFunction = IMAGE_LAB_EQUALIZATION_DEFAULT;	\
 (*_param_handle)->histogramAverageDepth = IMAGE_LAB_HISTAVERAGE_DEPTH_DEFAULT;
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
