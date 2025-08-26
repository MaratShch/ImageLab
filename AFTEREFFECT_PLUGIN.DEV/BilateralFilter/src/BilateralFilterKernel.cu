#include "ImageLabCUDA.hpp"
#include "BilateralFilterGPU.hpp"
#include "ColorTransformMatrix.hpp"
#include <cuda_runtime.h>
#include <math.h>

// Constant memory with gauss mesh values
__constant__ float cGpuMesh[gpuMaxMeshSize];

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

inline __device__ const float* RESTRICT getCenterMesh (int& meshPitch) noexcept { meshPitch = gpuMaxWindowSize;  return &cGpuMesh[meshCenter]; }

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
    auto varValue = [&](const float& inVal) { return ((inVal > 0.040450f) ? powf((inVal + 0.0550f) / 1.0550f, 2.40f) : (inVal / 12.92f)); };

    const float var_B = varValue(in.x) * 100.f;
    const float var_G = varValue(in.y) * 100.f;
    const float var_R = varValue(in.z) * 100.f;

    float4 out;
    out.x = var_R * 0.4124564f + var_G * 0.3575761f + var_B * 0.1804375f;
    out.y = var_R * 0.2126729f + var_G * 0.7151522f + var_B * 0.0721750f;
    out.z = var_R * 0.0193339f + var_G * 0.1191920f + var_B * 0.9503041f;
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

    auto varValue = [&](const float& inVal) { return ((inVal > 0.008856f) ? cbrtf(inVal) : (inVal * 7.787f + 16.f / 116.f)); };

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

    const float r1 = var_X *  3.2404562f + var_Y * -1.5371385f + var_Z * -0.4985314f;
    const float g1 = var_X * -0.9692660f + var_Y *  1.8760108f + var_Z *  0.0415560f;
    const float b1 = var_X *  0.0556434f + var_Y * -0.2040259f + var_Z *  1.0572252f;

    auto varValue = [&](const float& in) { return ((in > 0.0031308f) ? (1.055f * powf(in, 1.0f / 2.40f) - 0.055f) : (in * 12.92f)); };

    float4 out;
    constexpr float FLT_EPSILON{ 1.192092896e-07F };
    constexpr float white{ 1.f - FLT_EPSILON };

    out.z = CLAMP(varValue(r1), 0.f, white);
    out.y = CLAMP(varValue(g1), 0.f, white);
    out.x = CLAMP(varValue(b1), 0.f, white);
    out.w = in.w;

    return out;
}
/////////////////////////////////////////////////////////////////////////////////////



CUDA_KERNEL_CALL
bool LoadGpuMesh_CUDA (const float* hostMesh)
{
    /* Mesh array with algorithm for compute coefficients defined into "BilateralFilter_GPU.cpp" as private class' method */
    constexpr size_t loadSize = sizeof(cGpuMesh);
    const cudaError_t err = cudaMemcpyToSymbol (cGpuMesh, hostMesh, loadSize);
    return (cudaSuccess == err) ? true : false;
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
void BilateralFilterKernel
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

    const float divider = 2.0f * fSigma * fSigma;

    int meshPitch = -1;
    const float* gpuMeshCenter = getCenterMesh (meshPitch);

    float fNorm = 0.f;
    float bSum1 = 0.f, bSum2 = 0.f, bSum3 = 0.f;
 
    // get processed pixel 
    inPix = LabBuf[y * srcPitch + x];

    // Loop through the window
    for (int wy = -fRadius; wy <= fRadius; ++wy)
    {
        for (int wx = -fRadius; wx <= fRadius; ++wx)
        {
            // Calculate the neighboring pixel coordinates
            const int nx = x + wx;
            const int ny = y + wy;

            // Ensure the neighbor is within the image boundaries
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                const float meshValue = *(gpuMeshCenter + wy * meshPitch + wx);
                const float4 pixWindow = LabBuf[ny * srcPitch + nx];

                const float dL = pixWindow.x - inPix.x; // L - differences
                const float da = pixWindow.y - inPix.y; // a - differences
                const float db = pixWindow.z - inPix.z; // b - differences

                const float dotComp = dL * dL + da * da + db * db;
                const float pF = expf(-dotComp / divider) * meshValue;
                fNorm += pF;

                bSum1 += (pF * pixWindow.x);
                bSum2 += (pF * pixWindow.y);
                bSum3 += (pF * pixWindow.z);
            } // if (nx >= 0 && nx < width && ny >= 0 && ny < height)

        } // for (int wx = -fRadius; wx <= fRadius; ++wx)
    } // for (int wy = -fRadius; wy <= fRadius; ++wy)

    float4 filteredLabPix;
    filteredLabPix.w = inPix.w;        // copy alpha channel from input pixel
    filteredLabPix.x = bSum1 / fNorm;  // filtered L channel
    filteredLabPix.y = bSum2 / fNorm;  // filtered a channel
    filteredLabPix.z = bSum3 / fNorm;  // filtered b channel

    // convert back to RGB color space
    outPix = Xyz2Rgb(CieLab2Xyz(filteredLabPix));

    if (is16f)
    {
        Pixel16*  out16 = (Pixel16*)outBuf;
        out16[y * dstPitch + x] = FloatToHalf4(outPix);
    }
    else
        outBuf[y * dstPitch + x] = outPix;

    return;
}


__global__
void BilateralBypassKernel
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
void BilateralFilter_CUDA
(
    float* RESTRICT inBuf,
    float* RESTRICT outBuf,
    int dstPitch,
    int srcPitch,
    int	is16f,
    int width,
    int height,
    int fRadius,
    float fSigma
)
{
    dim3 blockDim(16, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    if (0 == fRadius)
        BilateralBypassKernel <<< gridDim, blockDim >>> (reinterpret_cast<const float4* RESTRICT>(inBuf), reinterpret_cast<float4* RESTRICT>(outBuf), width, height, srcPitch, dstPitch);
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
            BilateralFilterKernel <<< gridDim, blockDim >>> (reinterpret_cast<const float4* RESTRICT>(gpuLabImage), reinterpret_cast<float4* RESTRICT>(outBuf), width, height, srcPitch, dstPitch, fRadius, is16f, fSigma);

            // free all temporary allocated resources
            cudaFree (gpuLabImage);
            gpuLabImage = nullptr;
        } // if (cudaSuccess == cudaMalloc ...
    }

    cudaDeviceSynchronize();

    return;
}