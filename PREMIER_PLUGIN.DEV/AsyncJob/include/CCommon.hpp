#pragma once

#include <limits.h>
#include <iostream>

#ifdef _WINDLL
#define _WINDOWS
#endif

#ifdef _WINDOWS
#include <windows.h>
#else
#include <stdlib.h>
#undef _WINDOWS_LL_API
#endif

//#define _WINDOWS_LL_API

template <typename T>
T MAX(T a, T b) { return (a > b) ? a : b; }

template <typename T>
T MIN(T a, T b) { return (a < b) ? a : b; }

constexpr size_t defaultMemAlign = 4096;
constexpr size_t one_megabyte = 1024u * 1024u;
constexpr size_t defaulStorageSize = 8u * one_megabyte;

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
