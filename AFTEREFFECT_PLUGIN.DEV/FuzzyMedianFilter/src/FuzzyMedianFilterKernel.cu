#include "ImageLabCUDA.hpp"
#include "FuzzyMedianFilterGPU.hpp"
#include "ColorTransformMatrix.hpp"
#include <cuda_runtime.h>
#include <math.h>


// Intermediate CIE-Lab image (float32, L,a,b,A - channels)
float4* RESTRICT gpuLabImage{ nullptr };


//////////////////////// PURE DEVICE CODE ///////////////////////////////////////////
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

inline __device__ float CLAMP
(
    const float& in,
    const float& minVal,
    const float& maxVal
) noexcept
{
    return (in < minVal ? minVal : (in > maxVal ? maxVal : in));
}


inline __device__ float4 Rgb2Xyz
(
    const float4& in
) noexcept
{
    auto varValue = [&](const float inVal) { return ((inVal > 0.040450f) ? powf((inVal + 0.0550f) / 1.0550f, 2.40f) : (inVal / 12.92f)); };

    const float var_B = varValue(in.x) * 100.f;
    const float var_G = varValue(in.y) * 100.f;
    const float var_R = varValue(in.z) * 100.f;

    float4 out;
    out.x = var_R * 0.4124f + var_G * 0.3576f + var_B * 0.1805f;
    out.y = var_R * 0.2126f + var_G * 0.7152f + var_B * 0.0722f;
    out.z = var_R * 0.0193f + var_G * 0.1192f + var_B * 0.9505f;
    out.w = in.w; //copy Alpha channel from source pixel

    return out;
}

inline __device__ float4  Xyz2CieLab
(
    const float4& in
) noexcept
{
    constexpr float fRef[3] = {
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][0],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][1],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][2],
    };

    auto varValue = [&](const float inVal) { return ((inVal > 0.008856f) ? cbrtf(inVal) : (inVal * 7.787f + 16.f / 116.f)); };

    const float var_X = varValue(in.x / fRef[0]);
    const float var_Y = varValue(in.y / fRef[1]);
    const float var_Z = varValue(in.z / fRef[2]);

    float4 out;
    out.w = in.w;
    out.x = CLAMP(116.f *  var_Y - 16.f,   -100.f, 100.f);    // L
    out.y = CLAMP(500.f * (var_X - var_Y), -128.f, 128.f);    // a
    out.z = CLAMP(200.f * (var_Y - var_Z), -128.f, 128.f);    // b

    return out;
}

inline __device__ float4 CieLab2Xyz
(
    const float4& in
) noexcept
{
    constexpr float fRef[3] = {
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][0],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][1],
        cCOLOR_ILLUMINANT[CieLabDefaultObserver][CieLabDefaultIlluminant][2],
    };

    const float var_Y = (in.x + 16.f) / 116.f;
    const float var_X =  in.y / 500.f + var_Y;
    const float var_Z =  var_Y - in.z / 200.f;

    const float y1 = ((var_Y > 0.2068930f) ? (var_Y * var_Y * var_Y) : ((var_Y - 16.f / 116.f) / 7.787f));
    const float x1 = ((var_X > 0.2068930f) ? (var_X * var_X * var_X) : ((var_X - 16.f / 116.f) / 7.787f));
    const float z1 = ((var_Z > 0.2068930f) ? (var_Z * var_Z * var_Z) : ((var_Z - 16.f / 116.f) / 7.787f));

    float4 out;
    out.w = in.w; // copy Alpha channel from source buffer
    out.x = x1 * fRef[0];
    out.y = y1 * fRef[1];
    out.z = z1 * fRef[2];

    return out;
}

inline __device__ float4 Xyz2Rgb
(
    const float4& in
) noexcept
{
    const float var_X = in.x / 100.f;
    const float var_Y = in.y / 100.f;
    const float var_Z = in.z / 100.f;

    const float r1 = var_X *  3.2406f + var_Y * -1.5372f + var_Z * -0.4986f;
    const float g1 = var_X * -0.9689f + var_Y *  1.8758f + var_Z *  0.0415f;
    const float b1 = var_X *  0.0557f + var_Y * -0.2040f + var_Z *  1.0570f;

    auto varValue = [&](const float in) { return ((in > 0.0031308f) ? (1.055f * powf(in, 1.0f / 2.40f) - 0.055f) : (in * 12.92f)); };

    float4 out;
    constexpr float FLT_EPSILON{ 1.192092896e-07F };
    constexpr float white{ 1.f - FLT_EPSILON };

    out.z = CLAMP(varValue(r1), 0.f, white);
    out.y = CLAMP(varValue(g1), 0.f, white);
    out.x = CLAMP(varValue(b1), 0.f, white);
    out.w = in.w;

    return out;
}



__global__
void RGBToCIELabKernel
(
    const float4* RESTRICT inBuf,  // RGB source buffer (coming from Host)
          float4* RESTRICT outBuf, // CIE-Lab destination buffer (allocated on device)
    int width,                     // width of the RGB image buffer in pixels
    int height,                    // height of the RGB image buffer in pixels
    int srcPitch,                  // line pitch of the input RGB buffer
    int dstPitch,                  // line pitch of the output CIE-Lab buffer
    int is16f
)
{
    float4 inPix;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (is16f)
    {
        Pixel16*  in16 = (Pixel16*)inBuf;
        inPix = HalfToFloat4(in16[y * srcPitch + x]);
    }
    else
        inPix = inBuf[y * srcPitch + x];

    outBuf[y * dstPitch + x] = Xyz2CieLab (Rgb2Xyz(inPix));
    return;
}


__global__
void FuzzyMedianFilterKernel
(
    const float4* RESTRICT LabBuf, // CIE-Lab source buffer (allocated on device)
          float4* RESTRICT outBuf, // Processed IMage in RGBA format (output buffer)
    int width,                     // width of the RGB image buffer in pixels
    int height,                    // height of the RGB image buffer in pixels
    int srcPitch,                  // line pitch of the input RGB buffer
    int dstPitch,                  // line pitch of the output CIE-Lab buffer
    int fRadius,                   // filter radius (define processing window)
    int is16f,
    float fSigma
)
{
    float4 inPix;
    float4 outPix;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const float fSigmaSq = fSigma * fSigma;


    return;
}


__global__
void FuzzyMedianBypassKernel
(
    const float4* RESTRICT srcBuf, // RGB source buffer
          float4* RESTRICT dstBuf, // RGB destination buffer
    int width,                     // width of the RGB image buffer in pixels
    int height,                    // height of the RGB image buffer in pixels
    int srcPitch,                  // line pitch of the input RGB buffer
    int dstPitch                   // line pitch of the output RGB buffer
)
{   // filter raidus equal to zero, so let's simply copy input buffer content into output buffer
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    dstBuf[y * dstPitch + x] = srcBuf[y * srcPitch + x];
    return;
}



CUDA_KERNEL_CALL
void FuzzyMedianFilter_CUDA
(
    float* RESTRICT inBuf,
    float* RESTRICT outBuf,
    int dstPitch,
    int srcPitch,
    int	is16f,
    int width,
    int height,
    int fWindowSize,
    float fSigma
)
{
    dim3 blockDim(32, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    if (0 == fWindowSize)
        FuzzyMedianBypassKernel <<< gridDim, blockDim >>> (reinterpret_cast<const float4* RESTRICT>(inBuf), reinterpret_cast<float4* RESTRICT>(outBuf), width, height, srcPitch, dstPitch);
    else
    {
        // allocate memory for CIE-Lab intermediate buffer
        if (cudaSuccess == cudaMalloc (reinterpret_cast<void**>(&gpuLabImage), width * height * sizeof(float4)))
        {
            // Launch first kernel for convert image from RGB color space to CIE-Lab color space
            const int labPitch = width;
            RGBToCIELabKernel <<< gridDim, blockDim >>> (reinterpret_cast<const float4* RESTRICT>(inBuf), reinterpret_cast<float4* RESTRICT>(gpuLabImage), width, height, srcPitch, labPitch, is16f);

            // Synchronize to ensure the first kernel has completed
            cudaDeviceSynchronize();

            // perform Bilateral Filter with specific radius and convert back image from CIE-Lab color space to RGB space
            FuzzyMedianFilterKernel <<< gridDim, blockDim >>> (reinterpret_cast<const float4* RESTRICT>(gpuLabImage), reinterpret_cast<float4* RESTRICT>(outBuf), width, height, srcPitch, dstPitch, fWindowSize, is16f, fSigma);

            // free all temporary allocated resources
            cudaFree (gpuLabImage);
            gpuLabImage = nullptr;
        } // if (cudaSuccess == cudaMalloc ...
    }

    return;
}