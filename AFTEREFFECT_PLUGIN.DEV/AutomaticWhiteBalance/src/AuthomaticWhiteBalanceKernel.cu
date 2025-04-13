#include "ImageLabCUDA.hpp"
#include "AuthomaticWhiteBalanceGPU.hpp"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "AlgCorrectionMatrix.hpp"
#include <cuda_runtime.h>
#include <math.h>

float4* RESTRICT gpuImage[2]{ nullptr };
float3* RESTRICT gpuTmpResults{ nullptr }; // x -> U_avg, y -> V_avg, z -> grayCount

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


inline __device__ const float4 rgb2yuv
(
    const float4& rgb,  /* z = R, y = G, x = B, w = A */
    const eCOLOR_SPACE& color_space
) noexcept
{
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

    float3 in;
    float4 out;

    in.x = rgb.x * 255.0f;
    in.y = rgb.y * 255.0f;
    in.z = rgb.z * 255.0f;

    out.x = in.z * dev_RGB2YUV[color_space][0] + in.y * dev_RGB2YUV[color_space][1] + in.x * dev_RGB2YUV[color_space][2]; // Y
    out.y = in.z * dev_RGB2YUV[color_space][3] + in.y * dev_RGB2YUV[color_space][4] + in.x * dev_RGB2YUV[color_space][5]; // U
    out.z = in.z * dev_RGB2YUV[color_space][6] + in.y * dev_RGB2YUV[color_space][7] + in.x * dev_RGB2YUV[color_space][8]; // V
    out.w = rgb.w; //copy Alpha channel directly from source pixel

    return out;
}


__global__
void CollectRgbStatistics_CUDA
(
    const float4* __restrict__ srcBuf,
    float3* __restrict__ blockResults,
    int sizeX,
    int sizeY,
    int srcPitch,
    int is16f,
    const eCOLOR_SPACE color_space,
    float gray_threshold
)
{
    // Shared memory to accumulate per-block statistics
    __shared__ float3 sharedSum; // x: U, y: V, z: totalGray

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int blockIdx_x = blockIdx.x;
    const int blockIdx_y = blockIdx.y;

    // Linear block ID for writing to blockResults
    const int blockId = blockIdx_y * gridDim.x + blockIdx_x;

    const int x = blockIdx_x * blockDim.x + tx;
    const int y = blockIdx_y * blockDim.y + ty;

    // Only thread (0,0) initializes shared memory
    if (tx == 0 && ty == 0)
    {
        sharedSum.x = sharedSum.y = sharedSum.z = 0.0f;
    }

    __syncthreads();

    float4 inPix;

    // Skip if outside bounds
    if (x < sizeX && y < sizeY)
    {
        if (is16f)
        {
            const Pixel16* in16 = reinterpret_cast<const Pixel16*>(srcBuf);
            inPix = HalfToFloat4(in16[y * srcPitch + x]);
        }
        else
        {
            inPix = __ldg(&srcBuf[y * srcPitch + x]);
        }

        const float4 yuvPix = rgb2yuv(inPix, color_space);

        const float fVal = (fabsf(yuvPix.y) + fabsf(yuvPix.z)) / fmaxf(yuvPix.x, FLT_EPSILON);

        if (fVal < gray_threshold)
        {
            atomicAdd(&sharedSum.x, yuvPix.y); // U
            atomicAdd(&sharedSum.y, yuvPix.z); // V
            atomicAdd(&sharedSum.z, 1.0f);     // Count
        }
    }

    __syncthreads();

    // After reduction, store result to global memory from thread (0,0)
    if (tx == 0 && ty == 0)
        blockResults[blockId] = sharedSum;

    return;
}


__global__
void ReduceBlockResults_CUDA
(
    const float3* __restrict__ blockResults,
    float3* __restrict__ totalSum,
    int numBlocks
)
{
    extern __shared__ float3 shared[]; // This syntax is specifically used to declare a shared memory array whose size is not known at compile time.

    const int tid = threadIdx.x;
    const int globalId = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    float3 val = { 0.0f, 0.0f, 0.0f };
    if (globalId < numBlocks)
        val = blockResults[globalId];

    shared[tid] = val;
    __syncthreads();

    // In-place reduction in shared memory (parallel sum reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid].x += shared[tid + stride].x;
            shared[tid].y += shared[tid + stride].y;
            shared[tid].z += shared[tid + stride].z;
        }
        __syncthreads();
    }

    // Only thread 0 writes result
    if (tid == 0)
        totalSum[0] = shared[0];

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
    float4* RESTRICT gpuTmpImage = nullptr;
    float4* RESTRICT src = nullptr;
    float4* RESTRICT dst = nullptr;
    int srcIdx = 0, dstIdx = 1;
    int inPitch, outPitch;

    const dim3 blockDim(16, 32, 1);
    const dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);
    const size_t numMemProcBlocks = std::max(32ull, static_cast<size_t>(gridDim.x * gridDim.y)) * sizeof(float3);

    // allocate memory for temporary blocks coefficicens
    const cudaError_t mallocResult0 = cudaMalloc(reinterpret_cast<void**>(&gpuTmpResults), numMemProcBlocks);
    if (cudaSuccess == mallocResult0 && nullptr != gpuTmpResults)
    {
        // allocate memory for intermediate processing results
        const unsigned int blocksNumber = std::min(2u, (iter_cnt - 1u));
        if (blocksNumber > 0)
        {
            const unsigned int frameSize = static_cast<unsigned int>(width) * static_cast<unsigned int>(height);
            const cudaError_t mallocResult1 = cudaMalloc(reinterpret_cast<void**>(&gpuTmpImage), blocksNumber * frameSize * sizeof(float4));
            if (cudaSuccess == mallocResult1 && nullptr != gpuTmpImage)
            {
                gpuImage[0] = reinterpret_cast<float4* RESTRICT>(gpuTmpImage);
                gpuImage[1] = (2u == blocksNumber ? gpuImage[0] + frameSize : nullptr);
            }
            else
            {
                cudaFree(gpuTmpResults);
                gpuTmpResults = nullptr;
                return;
            }
        } // if (blocksNumber > 0)

        // MAIN PROC LOOP
        for (unsigned int i = 0u; i < iter_cnt; i++)
        {
            if (0u == i)
            {   // First iteration case
                dstIdx++;
                dstIdx &= 0x01;
                src = reinterpret_cast<float4* RESTRICT>(inBuf);
                dst = (1u == iter_cnt) ? reinterpret_cast<float4* RESTRICT>(outBuf) : gpuImage[dstIdx];
                inPitch = srcPitch;
                outPitch = (1u == iter_cnt) ? destPitch : width;
            }
            else if ((iter_cnt - 1u) == i)
            {   // Last iteration case
                src = gpuImage[dstIdx];
                dst = reinterpret_cast<float4* RESTRICT>(outBuf);
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

            // cleanup memory for temporary results before start processing kernels
            cudaMemset (gpuTmpResults, 0, numMemProcBlocks);

            // collect RGB statistics
            CollectRgbStatistics_CUDA <<< gridDim, blockDim >>> (src, gpuTmpResults, width, height, inPitch, is16f, color_space, gray_threshold);

            // reduce RGB statistics results (accumulate to single values)
            const int threadsPerBlock = 256;
            const int sharedMemBytes = threadsPerBlock * sizeof(float3);  // Dynamic shared memory size

            float3 h_result{};
            float3* RESTRICT d_finalResult = nullptr;
            cudaMalloc (reinterpret_cast<void**>(&d_finalResult), sizeof(float3));  // Allocates GPU memory
            
            ReduceBlockResults_CUDA <<<1, threadsPerBlock, sharedMemBytes >>> (gpuTmpResults, d_finalResult, numMemProcBlocks);
            cudaMemcpy (&h_result, d_finalResult, sizeof(float3), cudaMemcpyDeviceToHost);

            const float U_avg = h_result.x / h_result.z;
            const float V_avg = h_result.y / h_result.z;

            cudaFree(d_finalResult);
            d_finalResult = nullptr;

        } // for (unsigned int i = 0u; i < iter_cnt; i++)

        // Free/Cleanup resources before exit
        if (nullptr != gpuTmpImage)
        {
            cudaFree(gpuTmpImage);
            gpuTmpImage = gpuImage[0] = gpuImage[1] = nullptr;
        }

        if (nullptr != gpuTmpResults)
        {
            cudaFree(gpuTmpResults);
            gpuTmpResults = nullptr;
        }

    } //  if (cudaSuccess == mallocResult0 && nullptr != gpuTmpResults)

   return;
}