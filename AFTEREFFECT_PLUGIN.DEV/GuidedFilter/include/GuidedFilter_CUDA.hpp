#ifndef __IMAGE_LAB_GUIDED_FILTER_CUDA__
#define __IMAGE_LAB_GUIDED_FILTER_CUDA__

// ============================================================================
// Guided Filter, CUDA path (CUDA 10.2+, compute capability 6.1 = GTX 1060).
//
// Same AlgoControls structure as the CPU paths (radius, epsilon, filterMode,
// showConfidenceMap, skinTolerance, skinSoftness), same mathematics:
//   BGRA_32f [0..1] interleaved -> orthonormal YUV planes (0..255 scale)
//   -> guided filter per plane (clamp-to-edge box sums in double)
//   -> optional skin confidence / region blend / confidence map
//   -> BGRA_32f interleaved, alpha copied from the input.
//
// inBuffer / outBuffer are DEVICE pointers (cudaMalloc'ed by the caller, the
// input pre-filled with cudaMemcpyHostToDevice). srcPitch / dstPitch are in
// PIXELS (4 floats each). One frame per call; frameCount is not used
// (same convention as ImageLabAFMF32_CUDA).
// algoGpuParams is a HOST pointer (read on the host before the launches).
// ============================================================================

#include <cuda_runtime.h>
#include "ImageLabCUDA.hpp"
#include "AlgoControls.hpp"

#ifndef CUDA_KERNEL_CALL
 #define CUDA_KERNEL_CALL
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
    const AlgoControls* RESTRICT algoGpuParams,
    int                    frameCount,
    cudaStream_t           stream
);

CUDA_KERNEL_CALL
void ImageLabGuidedFilter_CleanupGPUF32();

#endif // __IMAGE_LAB_GUIDED_FILTER_CUDA__