#pragma once

#include <limits.h>
#include <iostream>

#ifdef _WINDLL
#define _WINDOWS
#endif

#ifndef CACHE_LINE
#define CACHE_LINE	64
#endif

#ifdef _WINDOWS
#include <windows.h>
#define CACHE_ALIGN __declspec(align(CACHE_LINE))
#define CLASS_EXPORT __declspec(dllexport)
#define RDTSC __rdtsc()
#else
#include <stdlib.h>
#undef _WINDOWS_LL_API
#define CACHE_ALIGN 
#define RDTSC 0
#endif

#ifndef RESTRICT
 #ifdef _WINDOWS
  #define RESTRICT  __restrict
 #else
  #define RESTRICT
 #endif
#endif // RESTRICT

//#define _WINDOWS_LL_API

constexpr size_t defaultMemAlign = 4096;
constexpr size_t one_megabyte = 1024u * 1024u;
constexpr size_t defaulStorageSize = 8u * one_megabyte;
constexpr size_t workerQueueSize = 4u; /* 4 jobs per one worker */
constexpr size_t workerCacheSize = 64536u; /* 64K worker cache size */

constexpr uint32_t maxAlgParams = 8u;
constexpr uint32_t priorityNormal = 45u;
constexpr uint32_t priorityLow = 20u;
constexpr uint32_t priorityHigh = 60u;
constexpr uint32_t timeoutDefault = 1000; /* 1000 mS */




#define CLASS_NON_COPYABLE(TypeName)            \
TypeName(TypeName const&) = delete;             \
TypeName& operator = (TypeName const&) = delete

#define CLASS_NON_MOVABLE(TypeName)             \
TypeName(TypeName &&) = delete;                 \
TypeName& operator = (TypeName&&) = delete


#ifdef _WINDOWS
#ifdef _WINDOWS_LL_API
inline void* allocate_aligned_mem(const size_t& size, const size_t& align = defaultMemAlign)
{
    return VirtualAlloc(
        NULL,
        size,
        MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN,
        PAGE_READWRITE
        ); 
}
inline void free_aligned_mem(void* p) 
{
    VirtualFree(p, 0, MEM_RELEASE);
}

#else
inline void* allocate_aligned_mem(const size_t size, const size_t align = defaultMemAlign) { return _aligned_malloc(size, align); }
inline void free_aligned_mem(void* p) { _aligned_free(p); }
#endif

#else // not _WINDOWS

inline void* allocate_aligned_mem(const size_t& size, const size_t& align = defaultMemAlign) { return aligned_alloc(size, align); }
inline void free_aligned_mem(void* p) { free(p); }
  
#endif

typedef union algParams
{
    char     ch;
    int8_t   i8;
    uint8_t  ui8;
    int16_t  i16;
    uint16_t ui16;
    int32_t  i32;
    uint32_t ui32;
    int64_t  i64;
    uint64_t ui64;
    size_t   size;
    float    f32;
    double   f64;
    void* RESTRICT pR;
    void*    p;
} algParams;

#ifdef __cplusplus 
extern "C" {
#endif

#ifdef _WINDOWS
	BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */);
#endif /* _WINDOWS */

#ifdef __cplusplus
}
#endif
