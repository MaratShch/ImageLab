#include "BlackAndWhiteGPU.hpp"
#include "ColorTransformMatrix.hpp"
#include "ImageLabCUDA.hpp"
#include <cuda_runtime.h>


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


__global__ void kBlackAndWhiteCUDA
(
	const float4* RESTRICT inImg,
	float4*  RESTRICT outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight
)
{
    float4 inPix;
    float4 outPix;

    float constexpr colorMatrix[3] = { RGB2YUV[BT709][0], RGB2YUV[BT709][1], RGB2YUV[BT709][2] };

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= inWidth || y >= inHeight) return;

    if (in16f)
    {
        Pixel16*  in16 = (Pixel16*)inImg;
        inPix = HalfToFloat4(in16[y * inPitch + x]);
    }
    else
    {
        inPix = inImg[y * inPitch + x];
    }
    
    //     R          G          B           
    outPix.z = outPix.y = outPix.x = (inPix.z * colorMatrix[0] + inPix.y * colorMatrix[1] + inPix.y * colorMatrix[2]);
    outPix.w = inPix.w; /* ALPHA channel	*/

    if (in16f)
    {
        Pixel16*  out16 = (Pixel16*)outImg;
        out16[y * outPitch + x] = FloatToHalf4(outPix);
    }
    else
    {
        outImg[y * outPitch + x] = outPix;
    }

    return;
}



__global__ void kBlackAndWhiteAdvancedCUDA
(
    const float4* RESTRICT inImg,
    float4*  RESTRICT outImg,
    const int outPitch,
    const int inPitch,
    const int in16f,
    const int inWidth,
    const int inHeight
)
{
    float4 inPix;
    float4 outPix;

    float constexpr fCoeff[3] = { 0.2126f,   0.7152f,  0.0722f };
    float constexpr fLumaExp = 1.0f / 2.20f;
    float constexpr logCompensation = 0.10f;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= inWidth || y >= inHeight) return;

    if (in16f)
    {
        Pixel16*  in16 = (Pixel16*)inImg;
        inPix = HalfToFloat4(in16[y * inPitch + x]);
    }
    else
    {
        inPix = inImg[y * inPitch + x];
    }

    // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#math-libraries
    // replace POW(X, N) by EXP(LOG(X) * N) 
    const float Z = exp(log(inPix.z + logCompensation) * 2.20f) - logCompensation;
    const float Y = exp(log(inPix.y + logCompensation) * 2.20f) - logCompensation;
    const float X = exp(log(inPix.x + logCompensation) * 2.20f) - logCompensation;

    const float tmpVal  = fCoeff[0] * Z + fCoeff[1] * Y + fCoeff[2] * X;
    const float bwPixel = exp(log(tmpVal + logCompensation) * fLumaExp) - logCompensation;

    //     R          G          B           
    outPix.z = outPix.y = outPix.x = bwPixel;
    outPix.w = inPix.w; /* ALPHA channel	*/

    if (in16f)
    {
        Pixel16*  out16 = (Pixel16*)outImg;
        out16[y * outPitch + x] = FloatToHalf4(outPix);
    }
    else
    {
        outImg[y * outPitch + x] = outPix;
    }

    return;
}


CUDA_KERNEL_CALL
void BlackAndWhiteFilter_CUDA
(
    float* inBuf,
    float* outBuf,
    int destPitch,
    int srcPitch,
    int	is16f,
    int width,
    int height
)
{
    dim3 blockDim(16, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    kBlackAndWhiteCUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height);
    cudaDeviceSynchronize();

    return;
}


CUDA_KERNEL_CALL
void BlackAndWhiteFilterAdvanced_CUDA
(
    float* inBuf,
    float* outBuf,
    int destPitch,
    int srcPitch,
    int	is16f,
    int width,
    int height
)
{
    dim3 blockDim(16, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    kBlackAndWhiteAdvancedCUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height);
    cudaDeviceSynchronize();

    return;
}
