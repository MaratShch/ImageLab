#ifndef IMAGELAB2_SEPIA_COLOR_GPU_CU
#define IMAGELAB2_SEPIA_COLOR_GPU_CU

#include "SepiaColorGPU.hpp"


typedef struct Pixel16
{
	unsigned short x;
	unsigned short y;
	unsigned short z;
	unsigned short w;
}Pixel16, *PPixel16;

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


void SepiaColor_CUDA
(
	float* destBuf,
	int destPitch,
	int	is16f,
	int width,
	int height
)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	cudaDeviceSynchronize();

	return;
}

#endif /* IMAGELAB2_SEPIA_COLOR_GPU_CU */