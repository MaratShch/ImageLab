#pragma once

#include <cstdint>

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

inline uint64_t RDTSC() noexcept
{
#ifdef _MSC_VER
	return __rdtsc();
#elif defined(__GNUC__) || defined(__clang__)
	uint32_t hi, lo;
	__asm__ volatile("rdtsc" : "=a" (lo), "=d" (hi));
	return ((uint64_t)hi << 32) | lo;
#else
    #error "RDTSC not supported on this compiler/platform"
#endif	
}
