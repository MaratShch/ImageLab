#pragma once

#include <stdint.h>

#ifndef CPU_PAGE_SIZE
#define CPU_PAGE_SIZE	4096
#endif

#define CPU_PAGE_ALIGN __declspec(align(CPU_PAGE_SIZE))

#ifndef CACHE_LINE
#define CACHE_LINE  64
#endif

#define CACHE_ALIGN __declspec(align(CACHE_LINE))

#define AVX2_ALIGN   __declspec(align(32))
#define AVX512_ALIGN __declspec(align(64))

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector always) \
                           __pragma(vector aligned)
#define __VECTORIZATION__ __pragma(vector always) \
                          __pragma(vector unaligned)  
#define __LOOP_UNROLL(min) __pragma(loop_count(min))
#else
#define __VECTOR_ALIGNED__
#define __VECTORIZATION__
#define __LOOP_UNROLL(min)
#endif


constexpr int IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR = 3;
constexpr int IMAGE_LAB_AE_PLUGIN_VERSION_MINOR = 0;
constexpr int PremierId = 'PrMr';

constexpr PF_ParamFlags   defaultFlags   = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
constexpr PF_ParamUIFlags defaultUiFlags = PF_PUI_NONE;


template <typename T>
static inline void AEFX_CLR_STRUCT_EX(T& str) noexcept
{
	memset (static_cast<void*>(&str), 0, sizeof(T));
}


static inline void AEFX_INIT_PARAM_STRUCTURE (PF_ParamDef& strDef, const PF_ParamFlags& paramFlag = defaultFlags, const PF_ParamUIFlags& uiFlag = defaultUiFlags) noexcept
{
	AEFX_CLR_STRUCT_EX(strDef);
	strDef.flags = paramFlag;
	strDef.ui_flags = uiFlag;
}


template <typename T>
inline void Image_SimpleCopy
(
	const T* __restrict srcBuffer,
	      T* __restrict dstBuffer,
	const int32_t&      height,
	const int32_t&      width,
	const int32_t&      src_line_pitch,
	const int32_t&      dst_line_pitch
) noexcept
{
	for (int32_t j = 0; j < height; j++)
	{
		const T* __restrict pSrcLine = srcBuffer + j * src_line_pitch;
		      T* __restrict pDstLine = dstBuffer + j * dst_line_pitch;
		__VECTORIZATION__
		for (int32_t i = 0; i < width; i++) { pDstLine[i] = pSrcLine[i]; }
	}
	return;
}


