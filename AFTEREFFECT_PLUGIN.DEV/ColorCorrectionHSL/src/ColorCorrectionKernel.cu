#include "ColorCorrectionGPU.hpp"
#include "FastAriphmetics.hpp"
#include "ImageLabCUDA.hpp"
#include "ColorConverts_GPU.hpp"


inline __device__ float4 HalfToFloat4 (Pixel16 in) noexcept
{
	return make_float4 (__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

inline __device__ Pixel16 FloatToHalf4 (float4 in) noexcept
{
	Pixel16 v;
	v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
	return v;
}

template<typename T>
inline __device__ T clamp_hue (T hue) noexcept
{
	constexpr T hueMin{ 0.f };
	constexpr T hueMax{ 360.f };

	if (hue < hueMin)
		return (hue + hueMax);
	else if (hue >= hueMax)
		return (hue - hueMax);
	return hue;
}

template<typename T>
inline __device__ T clamp_ls (T ls) noexcept
{
	constexpr T vMin{ 0.f };
	constexpr T vMax{ 100.f };

	return GPU::CLAMP_VALUE(ls, vMin, vMax);
}


__global__ void kColorCorrection_HSL_CUDA
(
	float4*  RESTRICT inImg,
	float4*  RESTRICT outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight,
	const float hue,
	const float sat,
	const float lum
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

	if (0.f != hue || 0.f != sat || 0.f != lum)
	{
		float H, S, L;
		float R, G, B;
		float newHue, newSat, newLum;
		float Hue, Sat, Lum;

		GPU::sRgb2hsl(
					inPix.z,	/* R */
					inPix.y,	/* G */
					inPix.x,	/* B */
					H,			/* H */
					S,			/* S */
					L);			/* L */

		newHue = H + hue;
		newSat = S + sat;
		newLum = L + lum;

		constexpr float reciproc100 = 1.0f / 100.0f;
		constexpr float reciproc360 = 1.0f / 360.0f;

		Hue = clamp_hue(newHue) * reciproc360;
		Sat = clamp_ls(newSat)  * reciproc100;
		Lum = clamp_ls(newLum)  * reciproc100;

		constexpr float f32_value_black = 0.f;
		constexpr float f32_value_white = 1.0f - GPU::flt_EPSILON;

		GPU::hsl2sRgb(
					Hue,		/* H */
					Sat,		/* S */
					Lum,		/* L */
					R,			/* R */
					G,			/* G */
					B);			/* B */

		outPix.z = GPU::CLAMP_VALUE(R, f32_value_black, f32_value_white);
		outPix.y = GPU::CLAMP_VALUE(G, f32_value_black, f32_value_white);
		outPix.x = GPU::CLAMP_VALUE(B, f32_value_black, f32_value_white);
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


__global__ void kColorCorrection_HSV_CUDA
(
	float4*  RESTRICT inImg,
	float4*  RESTRICT outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight,
	const float hue,
	const float sat,
	const float val
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

	if (0.f != hue || 0.f != sat || 0.f != val)
	{
		float H, S, V;
		float R, G, B;
		float newHue, newSat, newVal;
		float Hue, Sat, Val;

		GPU::sRgb2hsv(
			inPix.z,	/* R */
			inPix.y,	/* G */
			inPix.x,	/* B */
			H,			/* H */
			S,			/* S */
			V);			/* V */

		constexpr float reciproc100 = 1.0f / 100.0f;

		newHue = H + hue;
		newSat = S + sat * reciproc100;
		newVal = V + val * reciproc100;

		Hue = clamp_hue(newHue);
		Sat = clamp_ls(newSat);
		Val = clamp_ls(newVal);

		constexpr float f32_value_black = 0.f;
		constexpr float f32_value_white = 1.0f - GPU::flt_EPSILON;

		GPU::hsv2sRgb(
			Hue,		/* H */
			Sat,		/* S */
			Val,		/* V */
			R,			/* R */
			G,			/* G */
			B);			/* B */

		outPix.z = GPU::CLAMP_VALUE(R, f32_value_black, f32_value_white);
		outPix.y = GPU::CLAMP_VALUE(G, f32_value_black, f32_value_white);
		outPix.x = GPU::CLAMP_VALUE(B, f32_value_black, f32_value_white);
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


__global__ void kColorCorrection_HSI_CUDA
(
	float4*   inImg,
	float4*   outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight,
	const float hue,
	const float sat,
	const float intens
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

	if (0.f != hue || 0.f != sat || 0.f != intens)
	{
		float H, S, I;
		float R, G, B;
		float newHue, newSat, newInt;
		float Hue, Sat, Intens;

		GPU::sRgb2hsi(
			inPix.z,	/* R */
			inPix.y,	/* G */
			inPix.x,	/* B */
			H,			/* H */
			S,			/* S */
			I);			/* I */

		constexpr float reciproc100 = 1.0f / 100.0f;

		newHue = H + hue;
		newSat = S + sat * reciproc100;
		newInt = I + intens * reciproc100;

		Hue = clamp_hue(newHue);
		Sat = clamp_ls(newSat);
		Intens = clamp_ls(newInt);

		constexpr float f32_value_black = 0.f;
		constexpr float f32_value_white = 1.0f - GPU::flt_EPSILON;

		GPU::hsi2sRgb(
			Hue,		/* H */
			Sat,		/* S */
			Intens,		/* I */
			R,			/* R */
			G,			/* G */
			B);			/* B */

		outPix.z = GPU::CLAMP_VALUE(R, f32_value_black, f32_value_white);
		outPix.y = GPU::CLAMP_VALUE(G, f32_value_black, f32_value_white);
		outPix.x = GPU::CLAMP_VALUE(B, f32_value_black, f32_value_white);
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


__global__ void kColorCorrection_HSP_CUDA
(
	float4*   inImg,
	float4*   outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight,
	const float hue,
	const float sat,
	const float per
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

	if (0.f != hue || 0.f != sat || 0.f != per)
	{
		float H, S, P;
		float R, G, B;
		float newHue, newSat, newPer;
		float Hue, Sat, Per;

		GPU::sRgb2hsp(
			inPix.z,	/* R */
			inPix.y,	/* G */
			inPix.x,	/* B */
			H,			/* H */
			S,			/* S */
			P);			/* P */

		constexpr float reciproc100 = 1.0f / 100.0f;
		constexpr float reciproc360 = 1.0f / 360.0f;

		newHue = H + hue * reciproc360;
		newSat = S + sat * reciproc100;
		newPer = P + per * reciproc100;

		Hue = GPU::CLAMP_VALUE(newHue, 0.f, 1.0f);
		Sat = GPU::CLAMP_VALUE(newSat, 0.f, 1.0f);
		Per = GPU::CLAMP_VALUE(newPer, 0.f, 1.0f);

		constexpr float f32_value_black = 0.f;
		constexpr float f32_value_white = 1.0f - GPU::flt_EPSILON;

		GPU::hsp2sRgb(
			Hue,		/* H */
			Sat,		/* S */
			Per,		/* P */
			R,			/* R */
			G,			/* G */
			B);			/* B */

		outPix.z = GPU::CLAMP_VALUE(R, f32_value_black, f32_value_white);
		outPix.y = GPU::CLAMP_VALUE(G, f32_value_black, f32_value_white);
		outPix.x = GPU::CLAMP_VALUE(B, f32_value_black, f32_value_white);
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


__global__ void kColorCorrection_HSLuv_CUDA
(
	float4*   inImg,
	float4*   outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight,
	const float hue,
	const float sat,
	const float vol
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

	if (0.f != hue || 0.f != sat || 0.f != vol)
	{
		float H, S, L;
		float R, G, B;
		float newHue, newSat, newLuv;
		float Hue, Sat, Luv;

		GPU::sRgb2hsLuv(
			inPix.z,	/* R */
			inPix.y,	/* G */
			inPix.x,	/* B */
			H,			/* H */
			S,			/* S */
			L);			/* P */

		newHue = H + hue;
		newSat = S + sat;
		newLuv = L + vol;

		Hue = GPU::CLAMP_VALUE(newHue, 0.f, 360.0f);
		Sat = GPU::CLAMP_VALUE(newSat, 0.f, 100.0f);
		Luv = GPU::CLAMP_VALUE(newLuv, 0.f, 100.0f);

		constexpr float f32_value_black = 0.f;
		constexpr float f32_value_white = 1.0f - GPU::flt_EPSILON;

		GPU::hsLuv2sRgb(
			Hue,		/* H */
			Sat,		/* S */
			Luv,		/* P */
			R,			/* R */
			G,			/* G */
			B);			/* B */

		outPix.z = GPU::CLAMP_VALUE(R, f32_value_black, f32_value_white);
		outPix.y = GPU::CLAMP_VALUE(G, f32_value_black, f32_value_white);
		outPix.x = GPU::CLAMP_VALUE(B, f32_value_black, f32_value_white);
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


__global__ void kColorCorrection_HPLuv_CUDA
(
	float4*   inImg,
	float4*   outImg,
	const int outPitch,
	const int inPitch,
	const int in16f,
	const int inWidth,
	const int inHeight,
	const float hue,
	const float sat,
	const float per
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

	if (0.f != hue || 0.f != sat || 0.f != per)
	{
		float H, S, P;
		float R, G, B;
		float newHue, newSat, newPer;
		float Hue, Sat, Per;

		GPU::sRgb2hsp(
			inPix.z,	/* R */
			inPix.y,	/* G */
			inPix.x,	/* B */
			H,			/* H */
			S,			/* S */
			P);			/* P */

		constexpr float reciproc100 = 1.0f / 100.0f;
		constexpr float reciproc360 = 1.0f / 360.0f;

		newHue = H + hue * reciproc360;
		newSat = S + sat * reciproc100;
		newPer = P + per * reciproc100;

		Hue = GPU::CLAMP_VALUE(newHue, 0.f, 1.0f);
		Sat = GPU::CLAMP_VALUE(newSat, 0.f, 1.0f);
		Per = GPU::CLAMP_VALUE(newPer, 0.f, 1.0f);

		constexpr float f32_value_black = 0.f;
		constexpr float f32_value_white = 1.0f - GPU::flt_EPSILON;

		GPU::hsp2sRgb(
			Hue,		/* H */
			Sat,		/* S */
			Per,		/* P */
			R,			/* R */
			G,			/* G */
			B);			/* B */

		outPix.z = GPU::CLAMP_VALUE(R, f32_value_black, f32_value_white);
		outPix.y = GPU::CLAMP_VALUE(G, f32_value_black, f32_value_white);
		outPix.x = GPU::CLAMP_VALUE(B, f32_value_black, f32_value_white);
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
void ColorCorrection_HSL_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	float hue,
	float sat,
	float lum
) 
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kColorCorrection_HSL_CUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height, hue, sat, lum);

	cudaDeviceSynchronize();

	return;
}

CUDA_KERNEL_CALL
void ColorCorrection_HSV_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	float hue,
	float sat,
	float val
)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kColorCorrection_HSV_CUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height, hue, sat, val);

	cudaDeviceSynchronize();

	return;
}

CUDA_KERNEL_CALL
void ColorCorrection_HSI_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	float hue,
	float sat,
	float intens
)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kColorCorrection_HSI_CUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height, hue, sat, intens);

	cudaDeviceSynchronize();

	return;
}


CUDA_KERNEL_CALL
void ColorCorrection_HSP_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	float hue,
	float sat,
	float per
)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kColorCorrection_HSP_CUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height, hue, sat, per);

	cudaDeviceSynchronize();

	return;
}


CUDA_KERNEL_CALL
void ColorCorrection_HSLuv_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	float hue,
	float sat,
	float vol
)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kColorCorrection_HSLuv_CUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height, hue, sat, vol);

	cudaDeviceSynchronize();

	return;
}


CUDA_KERNEL_CALL
void ColorCorrection_HPLuv_CUDA
(
	float* inBuf,
	float* outBuf,
	int destPitch,
	int srcPitch,
	int	is16f,
	int width,
	int height,
	float hue,
	float sat,
	float vol
)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	kColorCorrection_HPLuv_CUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, destPitch, srcPitch, is16f, width, height, hue, sat, vol);

	cudaDeviceSynchronize();

	return;
}
