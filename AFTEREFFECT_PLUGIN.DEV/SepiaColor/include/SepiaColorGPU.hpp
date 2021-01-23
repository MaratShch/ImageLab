#pragma once

#include <cuda_runtime.h>

#ifdef __NVCC__
 /* Put here device specific includes files */
 #include <cuda_fp16.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
 #define CUDA_KERNEL_CALL extern "C"
#ifdef __cplusplus
}
#endif


typedef struct Pixel16
{
	unsigned short x;
	unsigned short y;
	unsigned short z;
	unsigned short w;
}Pixel16, *PPixel16;


CUDA_KERNEL_CALL
bool SepiaColorLoadMatrix_CUDA(void);

CUDA_KERNEL_CALL
void SepiaColor_CUDA
(
	float* destBuf,
	int destPitch,
	int	is16f,
	int width,
	int height
);
