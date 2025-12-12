#include "ColorCorrectionCieLAB_GPU.hpp"
#include "CommonAuxPixFormat.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include <cuda_runtime.h>
#include <math.h>

// Computations reference [https://www.easyrgb.com/en/convert.php#inputFORM]: 
// R = 255, G = 128, B = 64
// XYZ: X = 49.889, Y = 37.075, Z = 9.378  
// CIELab: L = 67.333,  a = 44.136,  b = 55.352

inline __device__ float4 HalfToFloat4 (const Pixel16& in) noexcept
{
	return make_float4 (__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

inline __device__ Pixel16 FloatToHalf4(const float4& in) noexcept
{
	Pixel16 v;
	v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
	return v;
}


inline __device__ float compute_varRGB (const float in) noexcept
{
    /*
        if ( var_R > 0.04045 ) 
            var_R = ( ( var_R + 0.055 ) / 1.055 ) ^ 2.4
        else
            var_R = var_R / 12.92
    */
    return ((in > 0.040450f) ? pow((in + 0.0550f) / 1.0550f, 2.40f) : (in / 12.92f));
}

inline __device__ float compute_varXYZ (const float in) noexcept
{
    /*
        if (var_X > 0.008856) 
            var_X = var_X ^ (1 / 3)
        else                   
            var_X = (7.787 * var_X) + (16 / 116)
    */
    return ((in > 0.008856f) ? powf(in, 1.0f / 3.0f) : (in * 7.787f + 16.f / 116.f));
}

inline __device__ float compute_varRGB2 (const float in) noexcept
{
    /*
        if ( var_R > 0.0031308 ) 
            var_R = 1.055 * ( var_R ^ ( 1 / 2.4 ) ) - 0.055
        else
            var_R = 12.92 * var_R
    */
    return ((in > 0.0031308f) ? (1.055f * powf(in, 1.0f / 2.40f) - 0.055f) : (in * 12.92f));
}

inline __device__ float clamp_value (const float in, const float min, const float max) noexcept
{
    return (in < min ? min : (in > max ? max : in));
}


inline __device__ float4 kRGB2XYZ
(
    const float4& inPixel
) noexcept
{
    float4 out;

    const float var_R = compute_varRGB(inPixel.z) * 100.f;
    const float var_G = compute_varRGB(inPixel.y) * 100.f;
    const float var_B = compute_varRGB(inPixel.x) * 100.f;
    
    out.x = var_R * 0.4124f + var_G * 0.3576f + var_B * 0.1805f;
    out.y = var_R * 0.2126f + var_G * 0.7152f + var_B * 0.0722f;
    out.z = var_R * 0.0193f + var_G * 0.1192f + var_B * 0.9505f;
    out.w = inPixel.w; // save original value from alpha channel

    return out;
}

inline __device__ float4 kXYZ2RGB
(
    const float4& inPixel
) noexcept
{
    float4 out;

    const float var_X = inPixel.x / 100.f;
    const float var_Y = inPixel.y / 100.f;
    const float var_Z = inPixel.z / 100.f;

    const float r1 = var_X *  3.2406f + var_Y * -1.5372f + var_Z * -0.4986f;
    const float g1 = var_X * -0.9689f + var_Y *  1.8758f + var_Z *  0.0415f;
    const float b1 = var_X *  0.0557f + var_Y * -0.2040f + var_Z *  1.0570f;

    constexpr float whiteMax = (1.0f - FLT_EPSILON);
    out.z = clamp_value (compute_varRGB2(r1), 0.f, whiteMax);
    out.y = clamp_value (compute_varRGB2(g1), 0.f, whiteMax);
    out.x = clamp_value (compute_varRGB2(b1), 0.f, whiteMax);
    out.w = inPixel.w; // save original value from alpha channel
    return out;
}


inline __device__ float4 kXYZ2CIELab
(
    const float4& inPixel, // x = L, y = a, z = b, w = Alpha
    const float  fRef[3]
) noexcept
{
    float4 out;

    const float var_X = compute_varXYZ(inPixel.x / fRef[0]);
    const float var_Y = compute_varXYZ(inPixel.y / fRef[1]);
    const float var_Z = compute_varXYZ(inPixel.z / fRef[2]);

    out.x = clamp_value(116.f * var_Y - 16.f,    -100.f, 100.f); // L
    out.y = clamp_value(500.f * (var_X - var_Y), -128.f, 128.f); // a
    out.z = clamp_value(200.f * (var_Y - var_Z), -128.f, 128.f); // b
    out.w = inPixel.w; // save original value from alpha channel

    return out;
}

inline __device__ float4 kCIELab2XYZ
(
    const float4& inPixel, // x = L, y = a, z = b, w = Alpha
    const float fRef[3]
) noexcept
{
    float4 out;
    
    const float var_Y = (inPixel.x + 16.f) / 116.f;
    const float var_X = inPixel.y / 500.f + var_Y;
    const float var_Z = var_Y - inPixel.z / 200.f;

    const float y1 = ((var_Y > 0.2068930f) ? (var_Y * var_Y * var_Y) : ((var_Y - 16.f / 116.f) / 7.787f));
    const float x1 = ((var_X > 0.2068930f) ? (var_X * var_X * var_X) : ((var_X - 16.f / 116.f) / 7.787f));
    const float z1 = ((var_Z > 0.2068930f) ? (var_Z * var_Z * var_Z) : ((var_Z - 16.f / 116.f) / 7.787f));

    out.x = x1 * fRef[0];
    out.y = y1 * fRef[1];
    out.z = z1 * fRef[2];
    out.w = inPixel.w; // save original value from alpha channel

    return out;
}

__device__ float4 kAdjustValues
(
    const float4& inPixel,
    float L,
    float A,
    float B
) noexcept
{
    float4 outPixel;
    outPixel.x = clamp_value(inPixel.x + L, -100.f, 100.f);
    outPixel.y = clamp_value(inPixel.y + A, -128.f, 128.f);
    outPixel.z = clamp_value(inPixel.z + B, -128.f, 128.f);
    outPixel.w = inPixel.w; // save original value from alpha channel
    return outPixel;
}


__device__ float4 kRGB2CIELab
(
    const float4& rgbPixel,
    const float fRef[3]
) noexcept
{
    return kXYZ2CIELab(kRGB2XYZ(rgbPixel), fRef);
}


__device__ float4 kCIELab2RGB
(
    const float4& pixelCIELab,
    const float  fRef[3]
) noexcept
{
    return kXYZ2RGB (kCIELab2XYZ(pixelCIELab, fRef));
}


__global__ void kColorCorrectionCieLAB_CUDA
(
    const float4* RESTRICT inImg,
    float4* RESTRICT outImg,
    int srcPitch,
    int dstPitch,
    int in16f,
    int sizeX,
    int sizeY,
    float L,
    float A,
    float B,
    float fRefX,
    float fRefY,
    float fRefZ
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

        const float fReferences[3] = { fRefX, fRefY, fRefZ };

        // convert input pixel from RGB to CIELab color domain
        const float4 pixCILELab = kRGB2CIELab (inPix, fReferences);

        // adjusted new pixel value in correspondence to control parameters
        const float4 newPixCIELab = kAdjustValues (pixCILELab, L, A, B);

        // back convert from CIELab to RGB color domain
        const float4 outPix = kCIELab2RGB (newPixCIELab, fReferences);

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
    const float* cMatrix
)
{
	dim3 blockDim(16, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    const float f1 = cMatrix[0];
    const float f2 = cMatrix[1];
    const float f3 = cMatrix[2];

    kColorCorrectionCieLAB_CUDA <<< gridDim, blockDim, 0 >>> ((const float4*)inBuf, (float4*)outBuf, srcPitch, destPitch, is16f, width, height, L, A, B, f1, f2, f3);

	cudaDeviceSynchronize();

	return;
}
