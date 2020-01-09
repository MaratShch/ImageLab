#pragma once

#define CACHE_LINE  64
#define CACHE_ALIGN __declspec(align(CACHE_LINE))

#define AVX2_ALIGN __declspec(align(32))
#define AVX512_ALIGN __declspec(align(64))

#if defined __INTEL_COMPILER 
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#else
#define __VECTOR_ALIGNED__
#endif


constexpr int IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR = 0;
constexpr int IMAGE_LAB_AE_PLUGIN_VERSION_MINOR = 1;


#ifndef AEFX_CLR_STRUCT_EX
#define	AEFX_CLR_STRUCT_EX(STRUCT)			             \
	constexpr size_t size = sizeof(STRUCT);	             \
	constexpr void* ptr = static_cast<void*>(&(STRUCT)); \
	memset(ptr, 0, size);
#endif
