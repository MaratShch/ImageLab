#ifndef __IMAGE_LAB_BILATERAL_FILTER_GPU_HANDLERS__
#define __IMAGE_LAB_BILATERAL_FILTER_GPU_HANDLERS__

#include <cuda_runtime.h>
#include "FuzzyMedianFilterEnum.hpp"

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
	unsigned short w; /* ALPHA	*/
}Pixel16, *PPixel16;

constexpr size_t Pixel16Size = sizeof(Pixel16);

CUDA_KERNEL_CALL
void FuzzyMedianFilter_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	int fRadius,
    float fSigma
);


#endif // __IMAGE_LAB_BILATERAL_FILTER_GPU_HANDLERS__