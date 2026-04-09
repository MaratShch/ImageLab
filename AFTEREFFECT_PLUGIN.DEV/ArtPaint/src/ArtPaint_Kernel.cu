#include "ArtPaint_GPU.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include <algorithm>
#include <cmath>


CUDA_KERNEL_CALL
void ArtPaint_CUDA
(
    const float* RESTRICT inBuffer, // source (input) buffer
    float* RESTRICT outBuffer,      // destination (output) buffer
    int srcPitch,                   // source buffer pitch in pixels 
    int dstPitch,                   // destination buffer pitch in pixels
    int width,                      // horizontal image size in pixels
    int height,                     // vertical image size in lines
    const AlgoControls* algoGpuParams, // algorithm controls
    int frameCounter,
    cudaStream_t stream
)
{
    cudaDeviceSynchronize();

    return;
}