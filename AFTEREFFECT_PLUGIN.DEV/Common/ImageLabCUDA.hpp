#ifndef __IMAGE_LAB2_CUDA_SPECIFIC_DEFINES__
#define __IMAGE_LAB2_CUDA_SPECIFIC_DEFINES__

#ifdef __NVCC__
 #define RESTRICT	__restrict__
#else
 #define RESTRICT	__restrict
#endif

#endif /* __IMAGE_LAB2_CUDA_SPECIFIC_DEFINES__ */