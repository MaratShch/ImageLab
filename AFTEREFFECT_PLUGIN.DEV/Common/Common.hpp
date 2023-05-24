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

template <typename T>
inline void AEFX_CLR_STRUCT_EX(T& str) noexcept
{
	memset (static_cast<void*>(&str), 0, sizeof(T));
}
