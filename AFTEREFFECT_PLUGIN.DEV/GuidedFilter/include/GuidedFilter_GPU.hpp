#ifndef __IMAGE_LAB_GUIDED_FILTER_GPU_ALGO__
#define __IMAGE_LAB_GUIDED_FILTER_GPU_ALGO__

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
void ImageLabGuidedFilter32_CUDA
(
    const float*  RESTRICT inBuffer,
    float*  RESTRICT outBuffer,
    int                    srcPitch,
    int                    dstPitch,
    int                    width,
    int                    height,
    AlgoControls* RESTRICT algoGpuParams,
    int                    frameCount,
    cudaStream_t           stream
);

CUDA_KERNEL_CALL
void ImageLabGuidedFilter_CleanupGPUF32();

#endif // __IMAGE_LAB_GUIDED_FILTER_GPU_ALGO__