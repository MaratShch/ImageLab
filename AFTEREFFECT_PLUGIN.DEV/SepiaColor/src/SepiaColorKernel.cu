#include "SepiaColorGPU.hpp"
#include "SepiaMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "ImageLabCUDA.hpp"

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
	float4*  RESTRICT inImg,
	float4*  RESTRICT outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight
)
{
	float4 inPix;
	float4 tempPix;
	float4 outPix;

	constexpr float value_black = 0.f;
	constexpr float FLT_EPSILON = 1.19209290e-07F;
	constexpr float value_white = 1.0f - FLT_EPSILON;

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= inWidth || y >= inHeight) return;

	if (in16f) {
		Pixel16*  in16 = (Pixel16*)inImg;
		inPix  = HalfToFloat4(in16 [y * inPitch  + x]);
	}
	else {
		inPix  = inImg[y * inPitch  + x];
	}

				/* bgRa */                    /* bGra */                    /* Bgra */
	tempPix.z = inPix.z * gpuSepiaMatrix[0] + inPix.y * gpuSepiaMatrix[1] + inPix.x * gpuSepiaMatrix[2]; /* RED channel		*/
	tempPix.y = inPix.z * gpuSepiaMatrix[3] + inPix.y * gpuSepiaMatrix[4] + inPix.x * gpuSepiaMatrix[5]; /* GREEN channel	*/
	tempPix.x = inPix.z * gpuSepiaMatrix[6] + inPix.y * gpuSepiaMatrix[7] + inPix.x * gpuSepiaMatrix[8]; /* BLUE channel	*/

	outPix.z = CLAMP_VALUE(tempPix.z, value_black, value_white);
	outPix.y = CLAMP_VALUE(tempPix.y, value_black, value_white);
	outPix.x = CLAMP_VALUE(tempPix.x, value_black, value_white);
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
bool SepiaColorLoadMatrix_CUDA(void)
{
	/* SepiaMatrix array is defined in "SepiaMatrix.hpp" include file */
	constexpr size_t loadSize = sizeof(SepiaMatrix);
	const cudaError_t err = cudaMemcpyToSymbol(gpuSepiaMatrix, SepiaMatrix, loadSize);

#ifdef _DEBUG
	float dbg_gpuSepiaMatrix[9] = {};
	const cudaError_t errDbg = cudaMemcpyFromSymbol(dbg_gpuSepiaMatrix, gpuSepiaMatrix, loadSize);
	if (cudaSuccess != errDbg) return false;
	for (int i = 0; i < 9;  i++)
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
	float* RESTRICT inBuf,
	float* RESTRICT outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height
)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kSepiaColorCUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height);

	cudaDeviceSynchronize();

	return;
}
