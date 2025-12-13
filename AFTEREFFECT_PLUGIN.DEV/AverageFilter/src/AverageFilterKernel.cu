#include "ImageLabCUDA.hpp"
#include "AverageFilterGPU.hpp"
#include <cuda_runtime.h>
#include <math.h>

using GPUAverageT = float;

inline __device__ float4 HalfToFloat4 (const Pixel16& in) noexcept
{
	return make_float4(__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

inline __device__ Pixel16 FloatToHalf4 (const float4& in) noexcept
{
	Pixel16 v;
	v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
	return v;
}


__device__ float4 kAverageFilterGeometricCUDA
(
	const float4* RESTRICT inImg,
	int x,
	int y,
	int sizeX,
	int sizeY,
	int pitch,
	int kernelSize,
	int in16f
)
{
	float4 outPix;
	float4 inPix;

	GPUAverageT rSum{0}, gSum{0}, bSum{0};
	int count = 0;
 	const int filterRadius = kernelSize >> 1;

	// This compensation is needed to avoid computing the logarithm of a zero value.
	constexpr GPUAverageT logCompensation{0.1};

	// Loop through the window
	for (int wy = -filterRadius; wy <= filterRadius; ++wy)
	{
		for (int wx = -filterRadius; wx <= filterRadius; ++wx)
		{
			// Calculate the neighboring pixel coordinates
			const int nx = x + wx;
			const int ny = y + wy;

			// Ensure the neighbor is within the image boundaries
			if (nx >= 0 && nx < sizeX && ny >= 0 && ny < sizeY)
			{
				if (in16f)
				{
					Pixel16* in16 = (Pixel16*)inImg;
					inPix = HalfToFloat4(in16[ny * pitch + nx]);
				}
				else {
					inPix = inImg[ny * pitch + nx];
				}

				// Add the pixel values to the sum (BGRA channels)
				bSum += log10(inPix.x + logCompensation); // B
				gSum += log10(inPix.y + logCompensation); // G
				rSum += log10(inPix.z + logCompensation); // R
				count++;
			} // if (nx >= 0 && nx < sizeX && ny >= 0 && ny < sizeY)
		} // for (int wx = -filterRadius; wx <= filterRadius; ++wx)
	} // for (int wy = -filterRadius; wy <= filterRadius; ++wy)

	const GPUAverageT reciproc = static_cast<GPUAverageT>(1) / static_cast<GPUAverageT>(count);
	constexpr GPUAverageT Ten{ 10 };

	outPix.w = inImg[y * pitch + x].w;
    outPix.x = static_cast<decltype(outPix.x)>(pow(Ten, bSum * reciproc) - logCompensation);
    outPix.y = static_cast<decltype(outPix.y)>(pow(Ten, gSum * reciproc) - logCompensation);
    outPix.z = static_cast<decltype(outPix.z)>(pow(Ten, rSum * reciproc) - logCompensation);

	return outPix;
}


__device__ float4 kAverageFilterAriphmeticCUDA
(
	const float4* RESTRICT inImg,
	int x,
	int y,
	int sizeX,
	int sizeY,
	int pitch,
	int kernelSize,
	int in16f
)
{
	float4 outPix;
	float4 inPix;

	GPUAverageT rSum{ 0 }, gSum{ 0 }, bSum{ 0 };
	int count = 0;
	const int filterRadius = kernelSize >> 1;

	// Loop through the window
	for (int wy = -filterRadius; wy <= filterRadius; ++wy)
	{
		for (int wx = -filterRadius; wx <= filterRadius; ++wx)
		{
			// Calculate the neighboring pixel coordinates
			const int nx = x + wx;
			const int ny = y + wy;

			// Ensure the neighbor is within the image boundaries
			if (nx >= 0 && nx < sizeX && ny >= 0 && ny < sizeY)
			{
				// input
				if (in16f)
				{
					Pixel16* in16 = (Pixel16*)inImg;
					inPix = HalfToFloat4(in16[ny * pitch + nx]);
				}
				else {
					inPix = inImg[ny * pitch + nx];
				}

				// Add the pixel values to the sum (BGRA channels)
				bSum += static_cast<GPUAverageT>(inPix.x); // B
				gSum += static_cast<GPUAverageT>(inPix.y); // G
				rSum += static_cast<GPUAverageT>(inPix.z); // R
				count++;
			} // if (nx >= 0 && nx < sizeX && ny >= 0 && ny < sizeY)
		} // for (int wx = -filterRadius; wx <= filterRadius; ++wx)
	} // for (int wy = -filterRadius; wy <= filterRadius; ++wy)

	const GPUAverageT reciproc = static_cast<GPUAverageT>(1) / static_cast<GPUAverageT>(count);
	outPix.w = inImg[y * pitch + x].w;
	outPix.x = static_cast<decltype(outPix.x)>(bSum * reciproc);
	outPix.y = static_cast<decltype(outPix.y)>(gSum * reciproc);
	outPix.z = static_cast<decltype(outPix.z)>(rSum * reciproc);

	return outPix;
}


__global__ void kAverageFilterCUDA
(
	const float4* RESTRICT inImg,
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
	float4 outPix;

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= inWidth || y >= inHeight) return;

	outPix = ((0 == isGeometric) ?
		 kAverageFilterAriphmeticCUDA (inImg, x, y, inWidth, inHeight, inPitch, kernelSize, in16f) :
		 kAverageFilterGeometricCUDA  (inImg, x, y, inWidth, inHeight, inPitch, kernelSize, in16f) );

	// output
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

	kAverageFilterCUDA <<< gridDim, blockDim, 0 >>> ((const float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height, kernelSize, isGeometric);

	cudaDeviceSynchronize();

	return;
}