#include "ImageLabCUDA.hpp"
#include "BilateralFilterGPU.hpp"
#include "CommonAuxPixFormat.hpp"
#include <cuda_runtime.h>

// Intermediate CIE-Lab image (float32, L,a,b - channels)
float* RESTRICT gpuLabImage{ nullptr };

// Constant memory with gauss mesh values
__constant__ float cGpuMesh[gpuMaxMeshSize];

//////////////////////// DEVICE CODE ////////////////////////////////////////////////
inline __device__ float4 HalfToFloat4(Pixel16 in)
{
    return make_float4(__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

inline __device__ Pixel16 FloatToHalf4(float4 in)
{
    Pixel16 v;
    v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
    return v;
}
/////////////////////////////////////////////////////////////////////////////////////



CUDA_KERNEL_CALL
bool LoadGpuMesh_CUDA (const float* hostMesh)
{
    /* SepiaMatrix array is defined in "SepiaMatrix.hpp" include file */
    constexpr size_t loadSize = sizeof(cGpuMesh);
    const cudaError_t err = cudaMemcpyToSymbol (cGpuMesh, hostMesh, loadSize);
    return (cudaSuccess == err) ? true : false;
}


CUDA_KERNEL_CALL
void BilateralFilter_CUDA
(
    float* RESTRICT inBuf,
    float* RESTRICT outBuf,
    int destPitch,
    int srcPitch,
    int	is16f,
    int width,
    int height,
    int fRadius
)
{
    // allocate memory for CIE-Lab intermediate buffer
    if (cudaSuccess == cudaMalloc((void**)&gpuLabImage, width * height * sizeof(fCIELabPix)))
    {
        dim3 blockDim(32, 32, 1);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

        // convert image from  RGB color space to CIE-Lab color space

        // perform Bilateral Filter with specific radius

        // convert back image from CIE-Lab color space to RGB space

        // free all temporary allocated resources
        cudaFree(gpuLabImage);
        gpuLabImage = nullptr;
    }

    return;
}