#include "ImageLabCUDA.hpp"
#include "AverageFilterGPU.hpp"


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


__global__ void kAverageFilterCUDA
(
	float4*  RESTRICT inImg,
	float4*  RESTRICT outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight,
	const int kernelSize,
	const int isGeometric
)
{
	float4 inPix;
	float4 tempPix;
	float4 outPix;

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= inWidth || y >= inHeight) return;

	if (in16f) {
		Pixel16*  in16 = (Pixel16*)inImg;
		inPix = HalfToFloat4(in16[y * inPitch + x]);
	}
	else {
		inPix = inImg[y * inPitch + x];
	}


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
void AverageFilter_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	int kernelSize,
	int isGeometric
)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kAverageFilterCUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height, kernelSize, isGeometric);

	cudaDeviceSynchronize();

	return;
}