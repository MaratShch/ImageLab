#include "ColorCorrectionCieLAB_GPU.hpp"
#include "CommonAuxPixFormat.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include <cuda_runtime.h>


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


__device__ fCIELabPix kRgb2CIELab
(
    const float4 inPixel,
    const float* fRef
)
{
    /* in first convert: sRGB -> XYZ */
    constexpr float reciproc12 = 1.f / 12.92f;
    constexpr float reciproc16 = 16.f / 116.f;
    constexpr float reciproc1 = 1.f / 1.055f;

    const float varR = ((inPixel.z > 0.040450f) ? powf((inPixel.z + 0.055f) * reciproc1, 2.40f) : inPixel.z * reciproc12); // R
    const float varG = ((inPixel.y > 0.040450f) ? powf((inPixel.y + 0.055f) * reciproc1, 2.40f) : inPixel.y * reciproc12); // G
    const float varB = ((inPixel.x > 0.040450f) ? powf((inPixel.x + 0.055f) * reciproc1, 2.40f) : inPixel.x * reciproc12); // B

    const float X = varR * 41.240f + varG * 35.760f + varB * 18.050f;
    const float Y = varR * 21.260f + varG * 71.520f + varB * 7.2200f;
    const float Z = varR * 1.9300f + varG * 11.920f + varB * 95.050f;

    /* convert: XYZ - > Cie-L*ab */
    const float varX = X / fRef[0];
    const float varY = Y / fRef[1];
    const float varZ = Z / fRef[2];

    const float vX = (varX > 0.0088560f) ? cbrtf(varX) : 7.7870f * varX + reciproc16;
    const float vY = (varY > 0.0088560f) ? cbrtf(varY) : 7.7870f * varY + reciproc16;
    const float vZ = (varZ > 0.0088560f) ? cbrtf(varZ) : 7.7870f * varZ + reciproc16;

    fCIELabPix pixelLAB;
    pixelLAB.L = CLAMP_VALUE(116.0f * vX - 16.f, -100.0f, 100.0f);
    pixelLAB.a = CLAMP_VALUE(500.0f * (vX - vY), -128.0f, 128.0f);
    pixelLAB.b = CLAMP_VALUE(200.0f * (vY - vZ), -128.0f, 128.0f);

    return pixelLAB;
}

__device__ fCIELabPix kAdjustValues
(
    const fCIELabPix inPixel,
    const float L,
    const float A,
    const float B
)
{
    fCIELabPix outPixel;

    outPixel.L = CLAMP_VALUE(inPixel.L + L, -100.0f, 100.0f);
    outPixel.a = CLAMP_VALUE(inPixel.a + A, -128.0f, 128.0f);
    outPixel.b = CLAMP_VALUE(inPixel.b + B, -128.0f, 128.0f);

    return outPixel;
}

__device__ float4 kCIELab2RGB
(
    const fCIELabPix pixelCIELab,
    const float4 inPixel,
    const float* fRef
) noexcept
{
    constexpr float reciproc_16_116 = 16.f / 116.f;

    /* CIEL*a*b -> XYZ */
    const float var_Y = (pixelCIELab.L + 16.f) / 116.0f;
    const float var_X = pixelCIELab.a / 500.0f + var_Y;
    const float	var_Z = var_Y - pixelCIELab.b / 200.0f;

    const float x1 = (var_X > 0.2068930f) ? var_X * var_X * var_X : (var_X - reciproc_16_116) / 7.7870f;
    const float y1 = (var_Y > 0.2068930f) ? var_Y * var_Y * var_Y : (var_Y - reciproc_16_116) / 7.7870f;
    const float z1 = (var_Z > 0.2068930f) ? var_Z * var_Z * var_Z : (var_Z - reciproc_16_116) / 7.7870f;

    const float X = x1 * fRef[0] / 100.0f;
    const float Y = y1 * fRef[1] / 100.0f;
    const float Z = z1 * fRef[2] / 100.0f;

    const float var_R = X *  3.24060f + Y * -1.53720f + Z * -0.49860f;
    const float var_G = X * -0.96890f + Y *  1.87580f + Z *  0.04150f;
    const float var_B = X *  0.05570f + Y * -0.20400f + Z *  1.05700f;

    constexpr float reciproc24 = 1.0f / 2.40f;
    const float R = (var_R > 0.0031308f ? 1.0550f * (powf(var_R, reciproc24)) - 0.0550f : 12.920f * var_R);
    const float G = (var_G > 0.0031308f ? 1.0550f * (powf(var_G, reciproc24)) - 0.0550f : 12.920f * var_G);
    const float B = (var_B > 0.0031308f ? 1.0550f * (powf(var_B, reciproc24)) - 0.0550f : 12.920f * var_B);

    float4 outPix;
    constexpr float value_black = 0.0f;
    constexpr float flt_EPSILON = 1.19209290e-07F;
    constexpr float value_white = 1.0f - flt_EPSILON;

    outPix.z = CLAMP_VALUE(R, value_black, value_white);
    outPix.y = CLAMP_VALUE(G, value_black, value_white);
    outPix.x = CLAMP_VALUE(B, value_black, value_white);
    outPix.w = inPixel.w;

    return outPix;
}

__global__ void kColorCorrectionCieLAB_CUDA
(
    const float4* RESTRICT inImg,
          float4* RESTRICT outImg,
    const float*  RESTRICT colorMatrix,
    int srcPitch,
    int dstPitch,
    int in16f,
    int sizeX,
    int sizeY,
    float L,
    float A,
    float B
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= sizeX || y >= sizeY) return;

    if (0.0f == L && 0.0f == A && 0.0f == B)
    {
        outImg[y * dstPitch + x] = inImg[y * srcPitch + x];
    }
    else
    {
        float4 inPix;

        // input
        if (in16f)
        {
		    Pixel16* in16 = (Pixel16*)inImg;
		    inPix  = HalfToFloat4(in16 [y * srcPitch + x]);
	    }
	    else
        {
		    inPix  = inImg[y * srcPitch + x];
	    }
        const float fReferences[3] = { colorMatrix[0], colorMatrix[1], colorMatrix[2] };

        // convert input pixel from RGB to CIELab color domain
        const fCIELabPix pixCILELab = kRgb2CIELab(inPix, fReferences);
        // adjusted new pixel value in correspondence to control parameters
        const fCIELabPix newPixCIELab = kAdjustValues(pixCILELab, L, A, B);
        // back convert from CIELab to RGB color domain
        const float4 outPix = kCIELab2RGB(newPixCIELab, inPix, fReferences);

        // output
        if (in16f)
        {
            Pixel16*  out16 = (Pixel16*)outImg;
            out16[y * dstPitch + x] = FloatToHalf4(outPix);
        }
        else
        {
            outImg[y * dstPitch + x] = outPix;
        }
    }

	return;
}



CUDA_KERNEL_CALL
void ColorCorrectionCieLAB_CUDA
(
    float* inBuf,
    float* outBuf,
    int destPitch,
    int srcPitch,
    int	is16f,
    int width,
    int height,
    float L,
    float A,
    float B,
    const float* colorMatrix
)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    kColorCorrectionCieLAB_CUDA <<< gridDim, blockDim, 0 >>> ((float4*)inBuf, (float4*)outBuf, colorMatrix, srcPitch, destPitch, is16f, width, height, L, A, B);

	cudaDeviceSynchronize();

	return;
}
