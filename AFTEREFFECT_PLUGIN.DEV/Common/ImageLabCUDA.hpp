#ifndef __IMAGE_LAB2_CUDA_SPECIFIC_DEFINES__
#define __IMAGE_LAB2_CUDA_SPECIFIC_DEFINES__

#ifdef __NVCC__
 #define RESTRICT	__restrict__
 #define INLINE_CALL inline
 #define FORCE_INLINE_CALL __forceinline__
 #define DEVICE      __device__ 
#else
 #define RESTRICT	__restrict
 #define INLINE_CALL inline
 #define FORCE_INLINE_CALL INLINE_CALL
 #define DEVICE 
#endif

#ifndef DEVICE_INLINE_CALL
 #define DEVICE_INLINE_CALL  DEVICE INLINE_CALL
#endif

#ifndef DEVICE_FORCE_INLINE_CALL
 #define DEVICE_FORCE_INLINE_CALL  DEVICE FORCE_INLINE_CALL
#endif

#endif /* __IMAGE_LAB2_CUDA_SPECIFIC_DEFINES__ */