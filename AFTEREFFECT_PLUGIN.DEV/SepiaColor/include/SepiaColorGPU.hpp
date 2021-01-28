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
	unsigned short x; /* BLUE	*/
	unsigned short y; /* GREEN	*/
	unsigned short z; /* RED	*/	
	unsigned short w; /* ALLPHA	*/
}Pixel16, *PPixel16;

#ifndef CLAMP_VALUE
#define CLAMP_VALUE(val, val_min, val_max) \
	(((val) < (val_min)) ? (val_min) : (((val) > (val_max)) ? (val_max) : (val)))
#endif

CUDA_KERNEL_CALL
bool SepiaColorLoadMatrix_CUDA(void);

CUDA_KERNEL_CALL
void SepiaColor_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height
);