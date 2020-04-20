#ifndef __IMAGE_LAB_HARDWARE_PLATFORM___
#define __IMAGE_LAB_HARDWARE_PLATFORM___


#define CACHE_LINE			64
#define CPU_PAGE_SIZE		4096
#define VECTOR_SIZE_SSE		16
#define VECTOR_SIZE_AVX		16
#define VECTOR_SIZE_AVX2	32
#define VECTOR_SIZE_AVX512	64



#define CACHE_ALIGN		__declspec(align(CACHE_LINE))
#define CPU_PAGE_ALIGN	__declspec(align(CPU_PAGE_SIZE))
#define AVX2_ALIGN		__declspec(align(VECTOR_SIZE_AVX2))
#define AVX512_ALIGN	__declspec(align(VECTOR_SIZE_AVX512))

#if defined __INTEL_COMPILER
#pragma warning(disable:161)
#define __INTEL__
#define __VECTOR_ALIGNED__ __pragma(vector aligned)
#define __ASSUME_ALIGNED(a, align_val) __assume_aligned(a, align_val)
#else
#pragma warning(disable:4068)
#define __VECTOR_ALIGNED__
#define __ASSUME_ALIGNED__         
#endif



#endif // __IMAGE_LAB_HARDWARE_PLATFORM___