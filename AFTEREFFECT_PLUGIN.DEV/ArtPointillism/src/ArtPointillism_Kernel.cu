#include "ArtPointillism_GPU.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include <cuda_runtime.h>
#include <math.h>



CUDA_KERNEL_CALL
void ArtPointillism_CUDA
(
    float* inBuffer,
    float* outBuffer,
    int destPitch, 
    int srcPitch, 
    int is16f, 
    int width, 
    int height, 
    const PontillismControls& algoGpuParams
)
{
    dim3 blockDim(16, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    return;
}
