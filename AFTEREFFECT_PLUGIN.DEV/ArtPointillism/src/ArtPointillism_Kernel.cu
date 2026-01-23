#include "ArtPointillism_GPU.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include <cuda_runtime.h>
#include <math.h>



CUDA_KERNEL_CALL
void ArtPointillism_CUDA
(
    const float* RESTRICT inBuffer, // source (input) buffer
    float* RESTRICT outBuffer,      // destination (output) buffer
    int srcPitch,                   // source buffer pitch in pixels 
    int dstPitch,                   // destination buffer pitch in pixels
    int is16f,                      // is 16 or 32 float bit width
    int width,                      // horizontal image size in pixels
    int height,                     // vertical image size in lines
    const PontillismControls& algoGpuParams // algorithm controls
)
{
    dim3 blockDim(16, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    return;
}
