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
DEVICE_INLINE_CALL float4 HalfToFloat4(Pixel16 in) noexcept
{
    return make_float4(__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

DEVICE_INLINE_CALL Pixel16 FloatToHalf4(float4 in) noexcept
{
    Pixel16 v;
    v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
    return v;
}


DEVICE_INLINE_CALL float CLAMP
(
    const float& in,
    const float& minVal,
    const float& maxVal
) noexcept
{
    return (in < minVal ? minVal : (in > maxVal ? maxVal : in));
}


DEVICE_INLINE_CALL const float4 rgb2yuv
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
    const float4* RESTRICT srcBuf,
    float3* RESTRICT blockResults,
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
            inPix = srcBuf[y * srcPitch + x];
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
    size_t numBlocks
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


__global__
void ImageRGBCorrection_CUDA
(
    const float4* RESTRICT srcBuf,
    float4* RESTRICT dstBuf,
    int width,
    int height,
    int srcPitch,
    int dstPitch,
    int is16f,
    float correctionR,
    float correctionG,
    float correctionB
)
{
    float4 inPix;
    float4 outPix;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (is16f)
    {
        // Pixel16*  in16 = (Pixel16*)srcBuf; // Original was non-const, but srcBuf is const float4*
        const Pixel16*  in16 = reinterpret_cast<const Pixel16*>(srcBuf); // Corrected to const Pixel16*
        inPix = HalfToFloat4(in16[y * srcPitch + x]);
    }
    else
    {
        inPix = srcBuf[y * srcPitch + x];
    }

    constexpr float black{ 0.f };
    constexpr float white{ 1.f - FLT_EPSILON };

    outPix.w = inPix.w;                                            // ALPHA channel
    outPix.x = CLAMP(inPix.x * correctionB, black, white);  // B
    outPix.y = CLAMP(inPix.y * correctionG, black, white);  // G
    outPix.z = CLAMP(inPix.z * correctionR, black, white);  // R

    if (is16f)
    {
        Pixel16*  out16 = (Pixel16*)dstBuf;
        out16[y * dstPitch + x] = FloatToHalf4(outPix);
    }
    else
    {
        dstBuf[y * dstPitch + x] = outPix;
    }

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
    const eChromaticAdaptation chromaticAdapt,
    const eCOLOR_SPACE color_space,
    const float gray_threshold,
    unsigned int iter_cnt
)
{
    float4* RESTRICT gpuTmpImage = nullptr; // This was a local variable, shadowing the global gpuImage
    float4* RESTRICT src = nullptr;
    float4* RESTRICT dst = nullptr;
    int srcIdx = 0, dstIdx = 1; // These were local, not using the global gpuImage directly for indexing
    int inPitch, outPitch;
    float U_avg = 0.f, U_avg_prev = 0.f;
    float V_avg = 0.f, V_avg_prev = 0.f;
    constexpr float algAWBepsilon = 1e-05f;

    const dim3 blockDim(16, 32, 1);
    const dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);
    // In your original code, gpuTmpResults was a global. numMemProcBlocks was calculated for this global.
    // I'll use the global gpuTmpResults here as per your original structure.
    const size_t numMemProcBlocks = std::max(32ull, static_cast<size_t>(gridDim.x * gridDim.y)) * sizeof(float3);


    // allocate memory for temporary blocks coefficicens
    // Uses global gpuTmpResults
    const cudaError_t mallocResult0 = cudaMalloc(reinterpret_cast<void**>(&gpuTmpResults), numMemProcBlocks);
    if (cudaSuccess == mallocResult0 && nullptr != gpuTmpResults)
    {
        // allocate memory for intermediate processing results
        // Uses global gpuImage for ping-pong targets
        const unsigned int blocksNumber = std::min(2u, (iter_cnt > 1u ? 2u : 0u)); // Adjusted logic to avoid alloc if iter_cnt <=1
        if (blocksNumber > 0) // Only allocate if more than 1 iteration potentially uses ping-pong
        {
            // const unsigned int frameSize = static_cast<unsigned int>(width) * static_cast<unsigned int>(height);
            // Your original gpuImage global was float4*, let's assume frameSize is elements.
            const size_t frameSizeElements = static_cast<size_t>(width) * static_cast<size_t>(height);
            // cudaError_t mallocResult1 = cudaMalloc(reinterpret_cast<void**>(&gpuTmpImage), blocksNumber * frameSize * sizeof(float4));
            // This `gpuTmpImage` was local. The results were assigned to the global `gpuImage[]`
            float4* local_gpuTmpImage_alloc = nullptr; // Temporary for allocation
            const cudaError_t mallocResult1 = cudaMalloc(reinterpret_cast<void**>(&local_gpuTmpImage_alloc), blocksNumber * frameSizeElements * sizeof(float4));

            if (cudaSuccess == mallocResult1 && nullptr != local_gpuTmpImage_alloc)
            {
                // Assign to the global gpuImage array
                ::gpuImage[0] = reinterpret_cast<float4* RESTRICT>(local_gpuTmpImage_alloc);
                ::gpuImage[1] = (2u == blocksNumber ? ::gpuImage[0] + frameSizeElements : nullptr);
            }
            else
            {
                cudaFree(::gpuTmpResults); // Use global
                ::gpuTmpResults = nullptr; // Nullify global
                                           // cudaFree(local_gpuTmpImage_alloc); // Free if it was partially allocated then failed to set global
                return;
            }
        } // if (blocksNumber > 0)

          // MAIN PROC LOOP
        for (unsigned int i = 0u; i < iter_cnt; /* value will be incremented in end of loop */)
        {
            // Ping-pong logic using local src/dst and global ::gpuImage for storage
            if (0u == i)
            {   // First iteration case
                // dstIdx was initialized to 1. Let's use a clear ping-pong logic.
                // Let's say current_dst_buffer_idx = 0 for first output if iter_cnt > 1
                int current_dst_buffer_idx = 0; // Or some consistent indexing
                src = reinterpret_cast<float4* RESTRICT>(inBuf);
                dst = (1u == iter_cnt) ? reinterpret_cast<float4* RESTRICT>(outBuf) : ::gpuImage[current_dst_buffer_idx];
                inPitch = srcPitch;
                outPitch = (1u == iter_cnt) ? destPitch : width;
                // Update dstIdx for next iteration's source selection if iter_cnt > 1
                if (iter_cnt > 1) dstIdx = current_dst_buffer_idx;
            }
            else // Subsequent iterations
            {
                srcIdx = dstIdx; // dstIdx from previous iteration becomes srcIdx for current
                dstIdx = 1 - srcIdx; // Ping-pong: toggle between 0 and 1 for ::gpuImage indices

                src = ::gpuImage[srcIdx];
                dst = ((iter_cnt - 1u) == i) ? reinterpret_cast<float4* RESTRICT>(outBuf) : ::gpuImage[dstIdx];
                inPitch = width;
                outPitch = ((iter_cnt - 1u) == i) ? destPitch : width;
            }


            // cleanup memory for temporary results before start processing kernels
            // Uses global gpuTmpResults
            cudaMemset(::gpuTmpResults, 0, numMemProcBlocks);

            // collect RGB statistics
            // Uses global gpuTmpResults as output
            CollectRgbStatistics_CUDA << < gridDim, blockDim >> > (src, ::gpuTmpResults, width, height, inPitch, is16f, color_space, gray_threshold);

            // Synchronize to ensure the first kernel has completed
            cudaDeviceSynchronize();

            // reduce RGB statistics results (accumulate to single values)
            const int threadsPerBlock = 512; // This was threadsPerBlockReduce in later versions
            const size_t sharedMemBytes = threadsPerBlock * sizeof(float3);  // Dynamic shared memory size

            float3 h_result{}; // Host-side result
            float3* RESTRICT d_finalResult = nullptr; // Device-side buffer for final sum
                                                      // numMemProcBlocks was the size in bytes. ReduceBlockResults_CUDA expects number of float3 elements.
                                                      // So, numInputElementsToReduce should be (gridDim.x * gridDim.y) or numMemProcBlocks / sizeof(float3)
                                                      // Let's use gridDim.x * gridDim.y which is the number of blocks launched by CollectRgbStatistics_CUDA
            size_t num_statistic_blocks = static_cast<size_t>(gridDim.x) * gridDim.y;

            cudaMalloc(reinterpret_cast<void**>(&d_finalResult), sizeof(float3));  // Allocates GPU memory

                                                                                   // Uses global gpuTmpResults as input
            ReduceBlockResults_CUDA << <1, threadsPerBlock, sharedMemBytes >> > (::gpuTmpResults, d_finalResult, num_statistic_blocks);

            // Synchronize to ensure the second kernel has completed
            cudaDeviceSynchronize();

            cudaMemcpy(&h_result, d_finalResult, sizeof(float3), cudaMemcpyDeviceToHost);

            // Check for division by zero for gray count
            if (h_result.z > 0.0f) {
                U_avg = h_result.x / h_result.z;
                V_avg = h_result.y / h_result.z;
            }
            else {
                U_avg = 0.0f; // Or some default to prevent NaN/Inf propagation
                V_avg = 0.0f;
            }


            // host API inline call
            float correctionMatrix[3]{}; // R, G, B
                                         // This function needs to be defined elsewhere, part of your AlgCorrectionMatrix.hpp?
                                         // extern void compute_correction_matrix(float U_avg, float V_avg, const eCOLOR_SPACE color_space, const eILLUMINATE illuminant, const eChromaticAdaptation chromaticAdapt, float* outCorrectionMatrix);
            compute_correction_matrix(U_avg, V_avg, color_space, illuminant, chromaticAdapt, correctionMatrix);

            // apply Color Correctin Matrix on the image 
            ImageRGBCorrection_CUDA << < gridDim, blockDim >> > (src, dst, width, height, inPitch, outPitch, is16f, correctionMatrix[0], correctionMatrix[1], correctionMatrix[2]);

            // Synchronize to ensure the last kernel has completed
            cudaDeviceSynchronize();

            const float U_avg_diff = U_avg - U_avg_prev;
            const float V_avg_diff = V_avg - V_avg_prev;

            const float normVal = std::sqrt(U_avg_diff * U_avg_diff + V_avg_diff * V_avg_diff);

            cudaFree(d_finalResult);
            d_finalResult = nullptr;

            U_avg_prev = U_avg;
            V_avg_prev = V_avg;

            // Loop update logic
            if (i < iter_cnt - 1) { // If not already the designated last iteration
                if (normVal < algAWBepsilon || h_result.z < 1.0f) { // Converged or no gray pixels
                    i = iter_cnt - 1; // Force the next iteration to be the last one
                }
                else {
                    i++; // Proceed to next iteration
                }
            }
            else { // This was the last iteration (either planned or forced)
                i++; // Increment to terminate loop
            }

        } // for (unsigned int i = 0u; i < iter_cnt; )

          // Free/Cleanup resources before exit
          // Free the buffers assigned to global gpuImage pointers
        if (nullptr != ::gpuImage[0])
        {
            cudaFree(::gpuImage[0]); // This frees the entire block if gpuImage[1] was an offset
            ::gpuImage[0] = nullptr;
            ::gpuImage[1] = nullptr;
        }
        // No, gpuTmpImage was local, ::gpuImage[0] holds the actual pointer to free.

        if (nullptr != ::gpuTmpResults) // Global
        {
            cudaFree(::gpuTmpResults);
            ::gpuTmpResults = nullptr;
        }

    } //  if (cudaSuccess == mallocResult0 && nullptr != gpuTmpResults)

      // Synchronize to ensure the last kernel has completed
    cudaDeviceSynchronize();

    return;
}