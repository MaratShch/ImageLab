#include "FastAriphmetics.hpp"
#include "ImageLabCUDA.hpp"
#include "ColorCorrectionGPU.hpp"
#include "CompileTimeUtils.hpp"
#include "RGB2CMYK.hpp"

inline __device__ float4 HalfToFloat4 (const Pixel16& in) noexcept
{
	return make_float4 (__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

inline __device__ Pixel16 FloatToHalf4 (const float4& in) noexcept
{
	Pixel16 v;
	v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
	return v;
}

__global__ void kColorCorrection_CMYK_CUDA
(
	float4*  RESTRICT inImg,
	float4*  RESTRICT outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight,
	const float add_C,
	const float add_M,
	const float add_Y,
	const float add_K
)
{
	float4 inPix;
	float4 outPix;

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= inWidth || y >= inHeight) return;

	if (in16f) {
		Pixel16* RESTRICT in16 = (Pixel16* RESTRICT)inImg;
		inPix  = HalfToFloat4(in16 [y * inPitch  + x]);
	}
	else {
		inPix  = inImg[y * inPitch  + x];
	}

	if (0.f != add_C || 0.f != add_M || 0.f != add_Y || 0.f != add_K)
	{
		float C, M, Y, K;
		float newC, newM, newY, newK;
		float newR, newG, newB;

		constexpr float reciproc100 = 1.0f / 100.0f;

		rgb_to_cmyk(
			inPix.z,
			inPix.y,
			inPix.x,
			C,
			M,
			Y,
			K);

		newC = CLAMP_VALUE(C + add_C * reciproc100, 0.f, 1.f);
		newM = CLAMP_VALUE(M + add_M * reciproc100, 0.f, 1.f);
		newY = CLAMP_VALUE(Y + add_Y * reciproc100, 0.f, 1.f);
		newK = CLAMP_VALUE(K + add_K * reciproc100, 0.f, 1.f);

		cmyk_to_rgb (newC, newM, newY, newK, newR, newG, newB);

		outPix.z = newR;
		outPix.y = newG;
		outPix.x = newB;
		outPix.w = inPix.w;
	}
	else 
	{
		outPix = inPix;
	}

	if (in16f)
	{
		Pixel16* RESTRICT out16 = (Pixel16* RESTRICT)outImg;
		out16[y * outPitch + x] = FloatToHalf4(outPix);
	}
	else
	{
		outImg[y * outPitch + x] = outPix;
	}

	return;
}


__global__ void kColorCorrection_RGB_CUDA
(
	float4*  RESTRICT inImg,
	float4*  RESTRICT outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight,
	const float R,
	const float G,
	const float B
)
{
	float4 inPix;
	float4 outPix;

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= inWidth || y >= inHeight) return;

	if (in16f) {
		Pixel16* RESTRICT in16 = (Pixel16* RESTRICT)inImg;
		inPix = HalfToFloat4(in16[y * inPitch + x]);
	}
	else {
		inPix = inImg[y * inPitch + x];
	}

	if (0.f != R || 0.f != G || 0.f != B)
	{
		constexpr float reciproc100 = 1.0f / 100.0f;
		constexpr float flt_EPSILON = 1.19209290e-07F;
		constexpr float f32_value_black = 0.f;
		constexpr float f32_value_white = 1.0f - flt_EPSILON;

		outPix.z = CLAMP_VALUE(inPix.z + R * reciproc100, f32_value_black, f32_value_white);
		outPix.y = CLAMP_VALUE(inPix.y + G * reciproc100, f32_value_black, f32_value_white);
		outPix.x = CLAMP_VALUE(inPix.x + B * reciproc100, f32_value_black, f32_value_white);
		outPix.w = inPix.w;
	}
	else
	{
		outPix = inPix;
	}

	if (in16f)
	{
		Pixel16* RESTRICT out16 = (Pixel16* RESTRICT)outImg;
		out16[y * outPitch + x] = FloatToHalf4(outPix);
	}
	else
	{
		outImg[y * outPitch + x] = outPix;
	}

	return;
}



CUDA_KERNEL_CALL
void ColorCorrection_CMYK_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	float C,
	float M,
	float Y,
	float K
) noexcept
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kColorCorrection_CMYK_CUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height, C, M, Y, K);

	cudaDeviceSynchronize();

	return;
}


CUDA_KERNEL_CALL
void ColorCorrection_RGB_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	float R,
	float G,
	float B
) noexcept
{
	dim3 blockDim(16, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kColorCorrection_RGB_CUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height, R, G, B);

	cudaDeviceSynchronize();

	return;
}
