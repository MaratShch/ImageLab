#include <windows.h>

#ifndef min
 #define min(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
 #define max(a,b) ((a) < (b) ? (a) : (b))
#endif

#define IMAGE_LAB_HDR_CPU_ONLY	0
#define IMAGE_LAB_HDR_GPU_ONLY	1
#define IMAGE_LAB_HDR_CPU_GPU	2

#define IMAGE_LAB_HDR_MAX_IMAGE_WIDTH	4096
#define IMAGE_LAB_HDR_MAX_IMAGE_HEIGHT	3072

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

#define IMAGE_LAB_HDR_EQUALIZATION_DEFAULT		IMAGE_LAB_HDR_EQUALIZATION_LINEAR
#define IMAGE_LAB_HDR_HISTAVERAGE_DEPTH_DEFAULT	1
#define IMAGE_LAB_HDR_HIST_AVERAGE_DEPTH_MAX	30

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
}ImageLabHDR_ParamStr, *PImageLabHDR_ParamStr;

#ifndef IMAGE_LAB_HDR_STR_PARAM_INIT
#define IMAGE_LAB_HDR_STR_PARAM_INIT(_param_str)						\
 _param_str.strSizeOf = sizeof(_param_str);								\
 _param_str.cudaEnabled = IMAGE_LAB_HDR_CPU_ONLY;						\
 _param_str.maxImageWidth = IMAGE_LAB_HDR_MAX_IMAGE_WIDTH;				\
 _param_str.maxImageHeight = IMAGE_LAB_HDR_MAX_IMAGE_HEIGHT;			\
 _param_str.thresholdLow = IMAGE_LAB_HDR_THRESHOLD_DEFAULT;				\
 _param_str.thresholdHigh = IMAGE_LAB_HDR_THRESHOLD_MIN;				\
 _param_str.equalizationFunction = IMAGE_LAB_HDR_EQUALIZATION_DEFAULT;	\
 _param_str.histogramAverageDepth = IMAGE_LAB_HDR_HISTAVERAGE_DEPTH_DEFAULT;
#endif

#ifndef IMAGE_LAB_HDR_PSTR_PARAM_INIT
#define IMAGE_LAB_HDR_PSTR_PARAM_INIT(_param_str_ptr)						\
 _param_str_ptr->strSizeOf = sizeof(* _param_str_ptr);						\
 _param_str_ptr->cudaEnabled = IMAGE_LAB_HDR_CPU_ONLY;						\
 _param_str_ptr->maxImageWidth = IMAGE_LAB_HDR_MAX_IMAGE_WIDTH;				\
 _param_str_ptr->maxImageHeight = IMAGE_LAB_HDR_MAX_IMAGE_HEIGHT;			\
 _param_str_ptr->thresholdLow = IMAGE_LAB_HDR_THRESHOLD_DEFAULT;			\
 _param_str_ptr->thresholdHigh = IMAGE_LAB_HDR_THRESHOLD_MIN;				\
 _param_str_ptr->equalizationFunction = IMAGE_LAB_HDR_EQUALIZATION_DEFAULT;	\
 _param_str_ptr->histogramAverageDepth = IMAGE_LAB_HDR_HISTAVERAGE_DEPTH_DEFAULT;
#endif

typedef struct
{
	size_t	strSizeOf;
	void*   pMainBuffer;
	int*    pHistogramSequence[IMAGE_LAB_HDR_HIST_AVERAGE_DEPTH_MAX];
	size_t  mainBufferBytesSize;
	int     histogramBlocks;
}ImageLabHDR_SystemMemoryBlock, *PImageLabHDR_SystemMemoryBlock;

bool ImageLabHDR_AllocSystemMemory(PImageLabHDR_SystemMemoryBlock* ppBlock);
void ImageLabHDR_FreeSystemMemory(PImageLabHDR_SystemMemoryBlock pBlock);