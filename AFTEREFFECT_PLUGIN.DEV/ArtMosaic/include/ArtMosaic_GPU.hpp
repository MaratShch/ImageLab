#ifndef __IMAGE_LAB_ART_MOSAIC_GPU_ACCELERATOR_DEFINITIONS_ALGO__
#define __IMAGE_LAB_ART_MOSAIC_GPU_ACCELERATOR_DEFINITIONS_ALGO__

#include <cuda_runtime.h>
#include "ImageLabCUDA.hpp"

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
void ImageLabMosaic_CUDA
(
    const float* RESTRICT inBuffer,
    float* RESTRICT outBuffer,
    int srcPitch,
    int dstPitch,
    int width,
    int height,
    int cellsNumber,
    int frameCount,
    cudaStream_t stream
);

#endif // __IMAGE_LAB_ART_MOSAIC_GPU_ACCELERATOR_DEFINITIONS_ALGO__