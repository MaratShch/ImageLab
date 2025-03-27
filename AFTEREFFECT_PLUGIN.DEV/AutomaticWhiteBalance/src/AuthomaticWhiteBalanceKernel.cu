#include "ImageLabCUDA.hpp"
#include "AuthomaticWhiteBalanceGPU.hpp"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "AlgCorrectionMatrix.hpp"
#include <cuda_runtime.h>
#include <math.h>

float4* RESTRICT gpuImage[2]{ nullptr };

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


inline __device__ const float4 rgb2yuv (const float4& rgb, const eCOLOR_SPACE& color_space) noexcept
{
    float4 out;

    float constexpr dev_RGB2YUV[][9] =
    {
        // BT.601
        {
             0.299000f,  0.587000f,  0.114000f,
            -0.168736f, -0.331264f,  0.500000f,
             0.500000f, -0.418688f, -0.081312f
        },

        // BT.709
        {
             0.212600f,   0.715200f,  0.072200f,
            -0.114570f,  -0.385430f,  0.500000f,
             0.500000f,  -0.454150f, -0.045850f
        },

        // BT.2020
        {
             0.262700f,   0.678000f,  0.059300f,
            -0.139630f,  -0.360370f,  0.500000f,
             0.500000f,  -0.459790f, -0.040210f
        },

        // SMPTE 240M
        {
             0.212200f,   0.701300f,  0.086500f,
            -0.116200f,  -0.383800f,  0.500000f,
             0.500000f,  -0.445100f, -0.054900f
        }
    };

    return out;
}


__global__
void CollectRgbStatistics_CUDA
(
    const float4* RESTRICT srcBuf,
          float4* RESTRICT dstBuf,
    int   sizeX,
    int   sizeY,
    int   srcPitch,
    int   dstPitch,
    int   is16f,
    const eCOLOR_SPACE color_space
)
{
    float4 inPix;
    float4 yuvPix;

    // Shared memory for partial results
    __shared__ float uPartialSum;
    __shared__ float vPartialSum;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= sizeX || y >= sizeY) return;

    if (is16f)
    {
        const Pixel16* in16 = reinterpret_cast<const Pixel16*>(srcBuf);
        inPix = HalfToFloat4(in16[y * srcPitch + x]);
    }
    else
    {
        inPix = srcBuf[y * srcPitch + x];
    }

    yuvPix = rgb2yuv (inPix, color_space);

    return;
}


CUDA_KERNEL_CALL
void AuthomaticWhiteBalance_CUDA
(
    float* inBuf,
    float* outBuf,
    int destPitch,
    int srcPitch,
    int	is16f,
    int width,
    int height,
    const eILLUMINATE illuminant,
    const eChromaticAdaptation chroma,
    const eCOLOR_SPACE color_space,
    const float gray_threshold,
    unsigned int iter_cnt
)
{
    float4* gpuTmpImage = nullptr;
    float4* src = nullptr;
    float4* dst = nullptr;
    int srcIdx = 0, dstIdx = 1;
    int inPitch, outPitch;
    float uAvg, vAvg;

    dim3 blockDim(16, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    // allocate memory for intermediate processing results
    const unsigned int blocksNumber = FastCompute::Min(2u, (iter_cnt - 1u));
    if (blocksNumber > 0)
    {
        const unsigned int frameSize = static_cast<unsigned int>(width) * static_cast<unsigned int>(height);
        if (cudaSuccess == cudaMalloc(reinterpret_cast<void**>(&gpuTmpImage), blocksNumber * frameSize * sizeof(float4)))
        {
            gpuImage[0] = reinterpret_cast<float4* RESTRICT>(gpuTmpImage);
            gpuImage[1] = (2u == blocksNumber ? gpuImage[0] + frameSize : nullptr);
        }
    } // if (blocksNumber > 0)
    
    // MAIN PROC LOOP
    for (unsigned int i = 0u; i < iter_cnt; i++)
    {
        uAvg = vAvg = 0.f;

        if (0u == i)
        {
            dstIdx++;
            dstIdx &= 0x01;
            src = reinterpret_cast<float4*>(inBuf);
            dst = (1u == iter_cnt) ? reinterpret_cast<float4*>(outBuf) : gpuImage[dstIdx];
            inPitch = srcPitch;
            outPitch = (1u == iter_cnt) ? destPitch : width;
        }
        else if ((iter_cnt - 1u) == i)
        {
            src = gpuImage[dstIdx];
            dst = reinterpret_cast<float4*>(outBuf);
            inPitch = width;
            outPitch = destPitch;
        } /* if (k > 0) */
        else
        {
            srcIdx = dstIdx;
            dstIdx++;
            dstIdx &= 0x1;
            src = gpuImage[srcIdx];
            dst = gpuImage[dstIdx];
            inPitch = outPitch = width;
        }

        CollectRgbStatistics_CUDA <<< gridDim, blockDim >>> (src, dst, width, height, inPitch, outPitch, is16f, color_space);


    } // for (unsigned int i = 0u; i < iter_cnt; i++)

    // Free/Cleanup resources before exit
    if (nullptr != gpuTmpImage)
    {
        cudaFree(gpuTmpImage);
        gpuTmpImage = gpuImage[0] = gpuImage[1] = nullptr;
    }

   return;
}