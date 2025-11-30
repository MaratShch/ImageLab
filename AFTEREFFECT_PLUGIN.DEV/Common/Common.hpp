#pragma once

#include <cstdint>

#if defined(_MSC_VER)
 #define FORCE_INLINE __forceinline
#else
 #define FORCE_INLINE __attribute__((always_inline)) inline
#endif

#ifndef CPU_PAGE_SIZE
 #define CPU_PAGE_SIZE	4096
#endif

#if defined(_MSC_VER)
 #define CPU_PAGE_ALIGN 	__declspec(align(CPU_PAGE_SIZE))
#else // GCC
 #define CPU_PAGE_ALIGN     __attribute__((aligned(CPU_PAGE_SIZE)))
#endif
	
#ifndef CACHE_LINE
#define CACHE_LINE  64
#endif

#if defined(_MSC_VER)
 #define CACHE_ALIGN	__declspec(align(CACHE_LINE))
 #define AVX2_ALIGN   	__declspec(align(32))
 #define AVX512_ALIGN 	__declspec(align(64))
#else // GCC
 #define CACHE_ALIGN    __attribute__((aligned(CACHE_LINE)))
 #define AVX2_ALIGN     __attribute__((aligned(32)))
 #define AVX512_ALIGN   __attribute__((aligned(64)))
#endif
	
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

constexpr uint32_t app_tag (const char a, const char b, const char c, const char d) noexcept
{
    return (static_cast<uint32_t>(a) << 24) |
           (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(c) << 8)  |
            static_cast<uint32_t>(d);
}

constexpr uint32_t PremierId = app_tag ('P', 'r', 'M', 'r');

template <typename T>
inline void AEFX_CLR_STRUCT_EX(T& str) noexcept
{
	memset (static_cast<void*>(&str), 0, sizeof(T));
}

inline void* ComputeAddress (const void* pAddr, const size_t bytes_offset) noexcept
{
	const size_t ptr = reinterpret_cast<const size_t>(pAddr);
	return reinterpret_cast<void*>(ptr + bytes_offset);
}


