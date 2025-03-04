#include "ImageLabCUDA.hpp"
#include "AuthomaticWhiteBalanceGPU.hpp"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "AlgCorrectionMatrix.hpp"
#include <cuda_runtime.h>
#include <math.h>


float4* RESTRICT gpuImage[2]{ nullptr };

//////////////////////// PURE DEVICE CODE ///////////////////////////////////////////
inline __device__ float4 HalfToFloat4(Pixel16 in) noexcept
{
    return make_float4(__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

inline __device__ Pixel16 FloatToHalf4(float4 in) noexcept
{
    Pixel16 v;
    v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
    return v;
}


CUDA_KERNEL_CALL
void AuthomaticWhiteBalance_CUDA
(
    float* inBuf,
    float* outBuf,
    int destPitch,
    int srcPitch,
    int	is16f,
    int width,
    int height,
    const eILLUMINATE illuminant,
    const eChromaticAdaptation chroma,
    const eCOLOR_SPACE color_space,
    const float gray_threshold,
    unsigned int iter_cnt
)
{
    float4* gpuTmpImage = nullptr;
    float4* src = nullptr;
    float4* dst = nullptr;
    int srcIdx = 0, dstIdx = 1;
    int inPitch = 0, outPitch = 0;

    dim3 blockDim(32, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    // allocate memory for intermediate processing results
    const unsigned int blocksNumber = FastCompute::Min(2u, (iter_cnt - 1u));
    if (blocksNumber > 0)
    {
        const unsigned int frameSize = static_cast<unsigned int>(width) * static_cast<unsigned int>(height);
        if (cudaSuccess == cudaMalloc(reinterpret_cast<void**>(&gpuTmpImage), blocksNumber * frameSize * sizeof(float4)))
        {
            gpuImage[0] = reinterpret_cast<float4* RESTRICT>(gpuTmpImage);
            gpuImage[1] = (2u == blocksNumber ? gpuImage[0] + frameSize : nullptr);
        }
    } // if (blocksNumber > 0)
    
    // MAIN PROC LOOP
    for (unsigned int i = 0u; i < iter_cnt; i++)
    {

    } // for (unsigned int i = 0u; i < iter_cnt; i++)

    // Free/Cleanup resources before exit
    if (nullptr != gpuTmpImage)
    {
        cudaFree(gpuTmpImage);
        gpuTmpImage = gpuImage[0] = gpuImage[1] = nullptr;
    }

   return;
}