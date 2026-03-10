#ifndef __IMAGE_LAB_DENOISE_GPU_ACCELERATOR_DEFINITIONS_ALGO__
#define __IMAGE_LAB_DENOISE_GPU_ACCELERATOR_DEFINITIONS_ALGO__

#include <cuda_runtime.h>
#include "ImageLabCUDA.hpp"
#include "AlgoControls.hpp"

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

CUDA_KERNEL_CALL
void ImageLabDenoise_CUDA
(
    const float* RESTRICT inBuffer,
    float* RESTRICT outBuffer,
    int srcPitch,
    int dstPitch,
    int width,
    int height,
    const AlgoControls* RESTRICT algoGpuParams,
    int frameCount,
    cudaStream_t stream
);


#endif // __IMAGE_LAB_DENOISE_GPU_ACCELERATOR_DEFINITIONS_ALGO__