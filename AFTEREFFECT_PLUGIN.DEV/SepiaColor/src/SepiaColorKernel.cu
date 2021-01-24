#include "SepiaColorGPU.hpp"
#include "SepiaMatrix.hpp"
#include "FastAriphmetics.hpp"


__constant__ float gpuSepiaMatrix[9];


inline __device__ float4 HalfToFloat4 (Pixel16 in)
{
	return make_float4 (__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

inline __device__ Pixel16 FloatToHalf4(float4 in)
{
	Pixel16 v;
	v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
	return v;
}


__global__ void kSepiaColorCUDA
(
	float4* inImg,
	float4* outImg,
	const int destPitch,
	const int in16f,
	const int inWidth,
	const int inHeight
)
{
	float4 in;
	float4 out;

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= inWidth || y >= inHeight) return;

	if (in16f) {
		Pixel16*  in16 = (Pixel16*)outImg;
		out = HalfToFloat4(in16[y *  destPitch + x]);
	}
	else {
		out = outImg[y *  destPitch + x];
	}


	return;
}


CUDA_KERNEL_CALL
bool SepiaColorLoadMatrix_CUDA(void)
{
	/* SepiaMatrix array is defined in "SepiaMatrix.hpp" include file */
	constexpr size_t loadSize = sizeof(SepiaMatrix);
	const cudaError_t err = cudaMemcpyToSymbol(gpuSepiaMatrix, SepiaMatrix, loadSize);

#ifdef _DEBUG
	float dbg_gpuSepiaMatrix[9] = {};
	cudaMemcpyFromSymbol(dbg_gpuSepiaMatrix, gpuSepiaMatrix, loadSize);
	for (int i = 0; i < 9; i++)
	{
		if (dbg_gpuSepiaMatrix[i] != SepiaMatrix[i])
			return false;
	}
	return true;
#else
	return (cudaSuccess == err) ? true : false;
#endif
}


CUDA_KERNEL_CALL
void SepiaColor_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int	is16f,
	int width,
	int height
)
{
	dim3 blockDim(16, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kSepiaColorCUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, is16f, width, height);

	cudaDeviceSynchronize();

	return;
}
