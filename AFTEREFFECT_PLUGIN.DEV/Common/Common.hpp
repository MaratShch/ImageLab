#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__INTEL_COMPILER) || defined(_MSC_VER)
 #include <intrin.h>
 #ifndef RESTRICT
  #define RESTRICT __restrict
 #endif
#else
 #include "AefxDevPatch.hpp"
 #ifndef RESTRICT
  #define RESTRICT __restrict__
 #endif
#endif

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
 #define __VECTOR_ALIGNED__ __pragma(loop(ivdep))
 #define __VECTORIZATION__ __pragma(loop(ivdep))
 #define __LOOP_UNROLL(min) __pragma(loop(unroll(min)))
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
    // This prevents you from accidentally clearing a complex class 
    // (which would corrupt memory/vtable).
    static_assert(std::is_trivially_copyable<T>::value,
        "CRITICAL ERROR: AEFX_CLR_STRUCT_EX called on a non-trivial type! "
        "You cannot use memset on classes with constructors, destructors, or virtual functions.");

    std::memset(static_cast<void*>(&str), 0, sizeof(T));
}


// Overload for read-only memory
inline const void* ComputeAddress (const void* pAddr, const std::ptrdiff_t bytes_offset) noexcept
{
    // 1. Cast to a byte-sized pointer (unsigned char* is standard for raw memory)
    const auto* byte_ptr = static_cast<const unsigned char*>(pAddr);

    // 2. The compiler safely handles the signed pointer arithmetic
    return static_cast<const void*>(byte_ptr + bytes_offset);
}

// Overload for writable memory
inline void* ComputeAddress (void* pAddr, const std::size_t bytes_offset) noexcept
{
    // 1. Cast to a byte-sized pointer (unsigned char* is standard for raw memory)
    auto* byte_ptr = static_cast<unsigned char*>(pAddr);

    // 2. The compiler safely handles the signed pointer arithmetic
    return static_cast<void*>(byte_ptr + bytes_offset);
}


inline uint64_t RDTSC() noexcept
{
#if defined(__INTEL_COMPILER) || defined(_MSC_VER)
	return __rdtsc();
#elif defined(__GNUC__) || defined(__clang__)
	uint32_t hi, lo;
	__asm__ volatile("rdtsc" : "=a" (lo), "=d" (hi));
	return ((uint64_t)hi << 32) | lo;
#else
    #error "RDTSC not supported on this compiler/platform"
#endif	
}
