#ifndef __IMAGE_LAB2_CUDA_SPECIFIC_DEFINES__
#define __IMAGE_LAB2_CUDA_SPECIFIC_DEFINES__

#ifdef __NVCC__
 #define RESTRICT	__restrict__
 #define INLINE_CALL inline
 #define DEVICE      __device__ 
#else
 #define RESTRICT	__restrict
 #define INLINE_CALL inline
 #define DEVICE 
#endif

#ifndef DEVICE_INLINE_CALL
 #define DEVICE_INLINE_CALL  DEVICE INLINE_CALL
#endif

#endif /* __IMAGE_LAB2_CUDA_SPECIFIC_DEFINES__ */