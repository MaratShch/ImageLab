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
DEVICE_FORCE_INLINE_CALL float4 HalfToFloat4(Pixel16 in) noexcept
{
    return make_float4(__half2float(in.x), __half2float(in.y), __half2float(in.z), __half2float(in.w));
}

DEVICE_FORCE_INLINE_CALL Pixel16 FloatToHalf4(float4 in) noexcept
{
    Pixel16 v;
    v.x = __float2half_rn(in.x); v.y = __float2half_rn(in.y); v.z = __float2half_rn(in.z); v.w = __float2half_rn(in.w);
    return v;
}


DEVICE_FORCE_INLINE_CALL float CLAMP
(
    const float& in,
    const float& minVal,
    const float& maxVal
) noexcept
{
    return (in < minVal ? minVal : (in > maxVal ? maxVal : in));
}


DEVICE_FORCE_INLINE_CALL const float4 rgb2yuv
(
    const float4& rgb,  /* z = R, y = G, x = B, w = A */
    const eCOLOR_SPACE& color_space
) noexcept
{
    static constexpr float dev_RGB2YUV[][9] =
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


DEVICE_FORCE_INLINE_CALL const float3 yuv2rgb
(
    const float3& yuv,  /* z = Y, y = V, x = U, w = A */
    const eCOLOR_SPACE& color_space
) noexcept
{
    static constexpr float devYUV2RGB[][9] =
    {
        // BT.601
        {
            1.000000f,  0.000000f,  1.407500f,
            1.000000f, -0.344140f, -0.716900f,
            1.000000f,  1.779000f,  0.000000f
        },

        // BT.709
        {
            1.000000f,  0.00000000f,  1.5748021f,
            1.000000f, -0.18732698f, -0.4681240f,
            1.000000f,  1.85559927f,  0.0000000f
        },

        // BT.2020
        {
            1.000000f,  0.00000000f,  1.4745964f,
            1.000000f, -0.16454810f, -0.5713517f,
            1.000000f,  1.88139998f,  0.0000000f
        },

        // SMPTE 240M
        {
            1.000000f,  0.0000000f,  1.5756000f,
            1.000000f, -0.2253495f, -0.4767712f,
            1.000000f,  1.8270219f,  0.0000000f
        } 
    };

    float3 in;
    float3 out;

    in.x = yuv.x * 255.0f;
    in.y = yuv.y * 255.0f;
    in.z = yuv.z * 255.0f;

    out.x = in.z * devYUV2RGB[color_space][0] + in.y * devYUV2RGB[color_space][1] + in.x * devYUV2RGB[color_space][2]; // R
    out.y = in.z * devYUV2RGB[color_space][3] + in.y * devYUV2RGB[color_space][4] + in.x * devYUV2RGB[color_space][5]; // G
    out.z = in.z * devYUV2RGB[color_space][6] + in.y * devYUV2RGB[color_space][7] + in.x * devYUV2RGB[color_space][8]; // B

    return out;
}


DEVICE_FORCE_INLINE_CALL const float3 computeOutMatrix
(
    const eILLUMINATE& illuminant,
    const eChromaticAdaptation& chromaticAdapt,
    const float3 xyzEst
) noexcept
{
    static constexpr float dev_illuminate[12][3] = {
        { 95.0470f,  100.0000f, 108.8830f }, // DAYLIGHT - D65 (DEFAULT)
        { 98.0740f,  100.0000f, 118.2320f }, // OLD_DAYLIGHT
        { 99.0927f,  100.0000f,  85.3130f }, // OLD_DIRECT_SUNLIGHT_AT_NOON
        { 95.6820f,  100.0000f,  92.1490f }, // MID_MORNING_DAYLIGHT
        { 94.9720f,  100.0000f, 122.6380f }, // NORTH_SKY_DAYLIGHT
        { 92.8340f,  100.0000f, 103.6650f }, // DAYLIGHT_FLUORESCENT_F1
        { 99.1870f,  100.0000f,  67.3950f }, // COOL_FLUERESCENT
        { 103.7540f, 100.0000f,  49.8610f }, // WHITE_FLUORESCENT
        { 109.1470f, 100.0000f,  38.8130f }, // WARM_WHITE_FLUORESCENT
        { 90.8720f,  100.0000f,  98.7230f }, // DAYLIGHT_FLUORESCENT_F5
        { 100.3650f, 100.0000f,  67.8680f }  // COOL_WHITE_FLUORESCENT
    };
    static constexpr float dev_tblColorAdaptation[5][9] = {
        { 0.73280f,  0.4296f, -0.16240f, -0.7036f, 1.69750f, 0.0061f, 0.0030f,  0.0136f, 0.98340f }, // CAT-02
        { 0.40024f,  0.7076f, -0.08081f, -0.2263f, 1.16532f, 0.0457f, 0.0f,     0.0f,    0.91822f }, // VON-KRIES
        { 0.89510f,  0.2664f, -0.16140f, -0.7502f, 1.71350f, 0.0367f, 0.0389f, -0.0685f, 1.02960f }, // BRADFORD
        { 1.26940f, -0.0988f, -0.17060f, -0.8364f, 1.80060f, 0.0357f, 0.0297f, -0.0315f, 1.00180f }, // SHARP
        { 0.79820f,  0.3389f, -0.13710f, -0.5918f, 1.55120f, 0.0406f, 0.0008f,  0.2390f, 0.97530f }, // CMCCAT2000
    };
    static constexpr float dev_tblColorAdaptationInv[5][9] = {
        { 1.096124f, -0.278869f, 0.182745f,	0.454369f, 0.473533f,  0.072098f, -0.009628f, -0.005698f, 1.015326f }, // INV CAT-02
        { 1.859936f, -1.129382f, 0.219897f, 0.361191f, 0.638812f,  0.0f,       0.0f,       0.0f,      1.089064f }, // INV VON-KRIES
        { 0.986993f, -0.147054f, 0.159963f, 0.432305f, 0.518360f,  0.049291f, -0.008529f,  0.040043f, 0.968487f }, // INV BRADFORD
        { 0.815633f,  0.047155f, 0.137217f, 0.379114f, 0.576942f,  0.044001f, -0.012260f,  0.016743f, 0.995519f }, // INV SHARP
        { 1.062305f, -0.256743f, 0.160018f, 0.407920f, 0.55023f,   0.034437f, -0.100833f, -0.134626f, 1.016755f }, // INV CMCCAT2000
    };

    const float* RESTRICT illuminate = dev_illuminate[illuminant];
    const float* RESTRICT colorAdaptation = dev_tblColorAdaptation[chromaticAdapt];
    const float* RESTRICT colorAdaptationInv = dev_tblColorAdaptationInv[chromaticAdapt];

    const float3 gainT =
    {
        illuminate[0] * colorAdaptation[0] + illuminate[1] * colorAdaptation[1] + illuminate[2] * colorAdaptation[2],
        illuminate[0] * colorAdaptation[3] + illuminate[1] * colorAdaptation[4] + illuminate[2] * colorAdaptation[5],
        illuminate[0] * colorAdaptation[6] + illuminate[1] * colorAdaptation[7] + illuminate[2] * colorAdaptation[8]
    };

    const float3 gainE =
    {
        xyzEst.x * colorAdaptation[0] + xyzEst.y * colorAdaptation[1] + xyzEst.z * colorAdaptation[2],
        xyzEst.x * colorAdaptation[3] + xyzEst.y * colorAdaptation[4] + xyzEst.z * colorAdaptation[5],
        xyzEst.x * colorAdaptation[6] + xyzEst.y * colorAdaptation[7] + xyzEst.z * colorAdaptation[8]
    };

    const float3 finalGain =
    {
        gainT.x / gainE.x,
        gainT.y / gainE.y,
        gainT.z / gainE.z
    };

    const float3 mulGain[3] =
    {
        {
            finalGain.x * colorAdaptation[0],
            finalGain.x * colorAdaptation[1],
            finalGain.x * colorAdaptation[2]
        },
        {
            finalGain.y * colorAdaptation[3],
            finalGain.y * colorAdaptation[4],
            finalGain.y * colorAdaptation[5]
        },
        {
            finalGain.z * colorAdaptation[6],
            finalGain.z * colorAdaptation[7],
            finalGain.z * colorAdaptation[8]
        }
    };

    const float3 outMatrix[3] =
    {
        {
            colorAdaptationInv[0] * mulGain[0].x + colorAdaptationInv[1] * mulGain[1].x + colorAdaptationInv[2] * mulGain[2].x,
            colorAdaptationInv[0] * mulGain[0].y + colorAdaptationInv[1] * mulGain[1].y + colorAdaptationInv[2] * mulGain[2].y,
            colorAdaptationInv[0] * mulGain[0].z + colorAdaptationInv[1] * mulGain[1].z + colorAdaptationInv[2] * mulGain[2].z
        },
        {
            colorAdaptationInv[3] * mulGain[0].x + colorAdaptationInv[4] * mulGain[1].x + colorAdaptationInv[5] * mulGain[2].x,
            colorAdaptationInv[3] * mulGain[0].y + colorAdaptationInv[4] * mulGain[1].y + colorAdaptationInv[5] * mulGain[2].y,
            colorAdaptationInv[3] * mulGain[0].z + colorAdaptationInv[4] * mulGain[1].z + colorAdaptationInv[5] * mulGain[2].z,
        },
        {
            colorAdaptationInv[6] * mulGain[0].x + colorAdaptationInv[7] * mulGain[1].x + colorAdaptationInv[8] * mulGain[2].x,
            colorAdaptationInv[6] * mulGain[0].y + colorAdaptationInv[7] * mulGain[1].y + colorAdaptationInv[8] * mulGain[2].y,
            colorAdaptationInv[6] * mulGain[0].x + colorAdaptationInv[7] * mulGain[1].z + colorAdaptationInv[8] * mulGain[2].z
        }
    };

    const float3 mult[3] =
    {
        {
            XYZtosRGB[0] * outMatrix[0].x + XYZtosRGB[1] * outMatrix[1].x + XYZtosRGB[2] * outMatrix[2].x,
            XYZtosRGB[0] * outMatrix[0].y + XYZtosRGB[1] * outMatrix[1].y + XYZtosRGB[2] * outMatrix[2].y,
            XYZtosRGB[0] * outMatrix[0].z + XYZtosRGB[1] * outMatrix[1].z + XYZtosRGB[2] * outMatrix[2].z
        },
        {
            XYZtosRGB[3] * outMatrix[0].x + XYZtosRGB[4] * outMatrix[1].x + XYZtosRGB[5] * outMatrix[2].x,
            XYZtosRGB[3] * outMatrix[0].y + XYZtosRGB[4] * outMatrix[1].y + XYZtosRGB[5] * outMatrix[2].y,
            XYZtosRGB[3] * outMatrix[0].z + XYZtosRGB[4] * outMatrix[1].z + XYZtosRGB[5] * outMatrix[2].z
        },
        {
            XYZtosRGB[6] * outMatrix[0].x + XYZtosRGB[7] * outMatrix[1].x + XYZtosRGB[8] * outMatrix[2].x,
            XYZtosRGB[6] * outMatrix[0].y + XYZtosRGB[7] * outMatrix[1].y + XYZtosRGB[8] * outMatrix[2].y,
            XYZtosRGB[6] * outMatrix[0].z + XYZtosRGB[7] * outMatrix[1].z + XYZtosRGB[8] * outMatrix[2].z
        }
    };

    const float3 d_outCorrectionMatrix
    {
        mult[0].x * sRGBtoXYZ[0] + mult[0].y * sRGBtoXYZ[3] + mult[0].z * sRGBtoXYZ[6],
        mult[1].x * sRGBtoXYZ[1] + mult[1].y * sRGBtoXYZ[4] + mult[1].z * sRGBtoXYZ[7],
        mult[2].x * sRGBtoXYZ[2] + mult[2].y * sRGBtoXYZ[5] + mult[2].z * sRGBtoXYZ[8]
    };

    return d_outCorrectionMatrix;
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
    float3* __restrict__ d_avgResult, // Renamed to reflect it holds averages + count
    size_t numBlocks
)
{
    // Shared memory for reduction within this block
    extern __shared__ float3 shared[];

    const int tid = threadIdx.x;
    const int globalId = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    float3 val = { 0.0f, 0.0f, 0.0f };
    if (globalId < numBlocks) 
    {
        // Read safely: ensure we don't read out of bounds for blockResults
        // Note: numMemProcBlocks passed to the caller was calculated based on gridDim,
        // which might be larger than the actual number of blocks containing data if the
        // image dimensions weren't exact multiples of blockDim. 
        // A safer approach might be to pass the actual gridDim.x * gridDim.y as numBlocks.
        // Assuming numBlocks passed in is correct for now.
        val = blockResults[globalId];
    }

    shared[tid] = val;
    __syncthreads(); // Ensure all loads are complete

                     // In-place reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid].x += shared[tid + stride].x; // Sum U
            shared[tid].y += shared[tid + stride].y; // Sum V
            shared[tid].z += shared[tid + stride].z; // Sum Count
        }
        __syncthreads(); // Synchronize after each reduction step
    }

    // Only thread 0 writes the final average results
    if (tid == 0)
    {
        float totalU = shared[0].x;
        float totalV = shared[0].y;
        float totalCount = shared[0].z;

        // Calculate averages directly here on the GPU
        // Avoid division by zero if no gray pixels were found
        float avgU = (totalCount > 0.0f) ? (totalU / totalCount) : 0.0f;
        float avgV = (totalCount > 0.0f) ? (totalV / totalCount) : 0.0f;

        // Store AvgU, AvgV, and the raw Count in the output buffer
        d_avgResult[0].x = avgU;       // Store U_avg
        d_avgResult[0].y = avgV;       // Store V_avg
        d_avgResult[0].z = totalCount; // Store total gray count
    }

    return; 
}


__global__ void CalculateCorrectionMatrix_Kernel
(
    const float3* __restrict__ d_avgResult,        // Input: {U_avg, V_avg, totalCount}
    float3* __restrict__ d_outCorrectionMatrix,     // Output: {corrR, corrG, corrB}
    const eCOLOR_SPACE color_space,
    const eILLUMINATE illuminant,
    const eChromaticAdaptation chromaticAdapt
)
{
    // This kernel is launched with <<<1, 1>>> so threadIdx.x and blockIdx.x are 0
    float3 yuv;
    yuv.z = 100.f;
    yuv.y = d_avgResult[0].y;
    yuv.x = d_avgResult[0].x;

    const float3 restored = yuv2rgb(yuv, color_space);

    // calculate xy chromaticity
    const float xX = restored.x * sRGBtoXYZ[0] + restored.y * sRGBtoXYZ[1] + restored.z * sRGBtoXYZ[2];
    const float xY = restored.x * sRGBtoXYZ[3] + restored.y * sRGBtoXYZ[4] + restored.z * sRGBtoXYZ[5];
    const float xZ = restored.x * sRGBtoXYZ[6] + restored.y * sRGBtoXYZ[7] + restored.z * sRGBtoXYZ[8];

    const float xXYZSum = xX + xY + xZ;
    const float xyEst[2] = { xX / xXYZSum, xY / xXYZSum };
    const float xyEstDiv = 100.0f / xyEst[1];

    // Converts xyY chromaticity to CIE XYZ.
    const float3 xyzEst = 
    { 
        xyEstDiv * xyEst[0], 
        100.0f,
        xyEstDiv * (1.0f - xyEst[0] - xyEst[1])
    };

    const float3 outMatrix = computeOutMatrix (illuminant, chromaticAdapt, xyzEst);

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
    const float3* RESTRICT d_correctionMatrix // CHANGED: Now a pointer
)
{
    float4 inPix;
    float4 outPix;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Read correction factors ONCE per thread (or once per block if broadcast is efficient)
    // Since d_correctionMatrix is small and read by all threads, it will be cached.
    const float correctionR = d_correctionMatrix->x; // Assuming R is at index 0
    const float correctionG = d_correctionMatrix->y; // Assuming G is at index 1
    const float correctionB = d_correctionMatrix->z; // Assuming B is at index 2

    if (is16f)
    {
        // IMPORTANT: The reinterpret_cast for srcBuf should ideally happen once outside the loop if possible,
        // or ensure the type passed in is already Pixel16* if is16f is true.
        // For now, keeping your original structure.
        const Pixel16* in16 = reinterpret_cast<const Pixel16*>(srcBuf); // const added
        inPix = HalfToFloat4(in16[y * srcPitch + x]);
    }
    else
    {
        inPix = srcBuf[y * srcPitch + x];
    }

    constexpr float black{ 0.f };
    constexpr float white{ 1.f - FLT_EPSILON }; // (or use 1.0f and rely on __float2half_rn behavior for clamping to [0,1] for half)

    outPix.w = inPix.w;
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
}


CUDA_KERNEL_CALL
void AuthomaticWhiteBalance_CUDA
(
    float* inBuf,       // Assuming this is PF_EffectWorld* or similar, cast to float*
    float* outBuf,      // Assuming this is PF_EffectWorld* or similar, cast to float*
    int destPitch,      // Pitch in pixels for destination
    int srcPitch,       // Pitch in pixels for source
    int	is16f,          // Boolean flag for 16-bit float
    int width,
    int height,
    const eILLUMINATE illuminant,
    const eChromaticAdaptation chromaticAdapt,
    const eCOLOR_SPACE color_space,
    const float gray_threshold,
    unsigned int iter_cnt
)
{
    // Device pointers for ping-pong buffers for intermediate images
    float4* RESTRICT gpuImage_ping = nullptr; // Base pointer for ping-pong allocation
    float4* RESTRICT gpuImage_pong[2] = { nullptr, nullptr }; // Pointers to individual buffers

    float4* RESTRICT src_ptr = nullptr;
    float4* RESTRICT dst_ptr = nullptr;
    int current_src_idx = 0;
    int current_dst_idx = 1; // For ping-pong indexing if iter_cnt > 1
    int current_in_pitch, current_out_pitch;

    // Host variables for convergence check (these will be updated from GPU data)
    float U_avg_host = 0.f, U_avg_prev_host = 0.f;
    float V_avg_host = 0.f, V_avg_prev_host = 0.f;
    constexpr float algAWBepsilon = 1e-05f;

    // Kernel launch configurations
    // For CollectRgbStatistics_CUDA and ImageRGBCorrection_CUDA
    const dim3 blockDimKernel(16, 32, 1); // 512 threads per block
    const dim3 gridDimKernel((width + blockDimKernel.x - 1) / blockDimKernel.x, (height + blockDimKernel.y - 1) / blockDimKernel.y, 1);

    // For ReduceBlockResults_CUDA
    const int threadsPerBlockReduce = 512; // Max reasonable for a single block reduction
    const size_t sharedMemBytesReduce = threadsPerBlockReduce * sizeof(float3);
    const size_t numBlocksInGridForStats = static_cast<size_t>(gridDimKernel.x) * gridDimKernel.y; // Number of blocks from CollectRgbStatistics

                                                                                                   // Device memory allocations
    float3* RESTRICT d_blockResultsSum = nullptr;     // Stores per-block sums from CollectRgbStatistics
    float3* RESTRICT d_avgResult = nullptr;           // Stores {U_avg, V_avg, totalCount} from ReduceBlockResults
    float3* RESTRICT d_correctionMatrixGpu = nullptr; // Stores {corrR, corrG, corrB} from CalculateCorrectionMatrix_Kernel

    cudaError_t err;

    // Allocate memory for per-block sums from CollectRgbStatistics_CUDA
    const size_t blockResultsSumBytes = numBlocksInGridForStats * sizeof(float3);
    err = cudaMalloc(reinterpret_cast<void**>(&d_blockResultsSum), blockResultsSumBytes);
    if (err != cudaSuccess) {
        //fprintf(stderr, "CUDA Error: Failed to allocate d_blockResultsSum (%s)\n", cudaGetErrorString(err));
        goto cleanup_and_exit;
    }

    // Allocate memory for the final reduced average result {U_avg, V_avg, totalCount}
    err = cudaMalloc(reinterpret_cast<void**>(&d_avgResult), sizeof(float3));
    if (err != cudaSuccess) {
        //fprintf(stderr, "CUDA Error: Failed to allocate d_avgResult (%s)\n", cudaGetErrorString(err));
        goto cleanup_and_exit;
    }

    // Allocate memory for the correction matrix {R_corr, G_corr, B_corr} on GPU
    err = cudaMalloc(reinterpret_cast<void**>(&d_correctionMatrixGpu), sizeof(float3));
    if (err != cudaSuccess) {
        //fprintf(stderr, "CUDA Error: Failed to allocate d_correctionMatrixGpu (%s)\n", cudaGetErrorString(err));
        goto cleanup_and_exit;
    }

    // Allocate ping-pong buffers for intermediate images if iter_cnt > 1
    if (iter_cnt > 1) {
        const size_t frameSizeElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t frameSizeBytes = frameSizeElements * sizeof(float4);
        // Allocate a single contiguous block for two frames if needed
        err = cudaMalloc(reinterpret_cast<void**>(&gpuImage_ping), 2 * frameSizeBytes);
        if (err != cudaSuccess) {
            //fprintf(stderr, "CUDA Error: Failed to allocate gpuImage_ping (%s)\n", cudaGetErrorString(err));
            goto cleanup_and_exit;
        }
        gpuImage_pong[0] = gpuImage_ping;
        gpuImage_pong[1] = gpuImage_ping + frameSizeElements;
    }

    // MAIN PROCESSING LOOP
    for (unsigned int i = 0u; i < iter_cnt; /* i incremented based on convergence */)
    {
        if (i == 0) { // First iteration
            src_ptr = reinterpret_cast<float4* RESTRICT>(inBuf);
            current_in_pitch = srcPitch;
            if (iter_cnt == 1) { // Single iteration: output directly to outBuf
                dst_ptr = reinterpret_cast<float4* RESTRICT>(outBuf);
                current_out_pitch = destPitch;
            }
            else { // Multiple iterations: output to first ping-pong buffer
                dst_ptr = gpuImage_pong[current_dst_idx]; // Use gpuImage_pong[1] (or [0] if you prefer)
                current_out_pitch = width; // Pitches for intermediate buffers are just width
            }
        }
        else { // Subsequent iterations
               // src is the dst from the previous iteration
            src_ptr = gpuImage_pong[current_dst_idx]; // This was the destination in the previous step
            current_in_pitch = width;

            // Swap ping-pong buffers
            current_src_idx = current_dst_idx;
            current_dst_idx = 1 - current_src_idx; // Toggle 0 and 1

            if (i == iter_cnt - 1) { // Last iteration: output to final outBuf
                dst_ptr = reinterpret_cast<float4* RESTRICT>(outBuf);
                current_out_pitch = destPitch;
            }
            else { // Intermediate iteration: output to the other ping-pong buffer
                dst_ptr = gpuImage_pong[current_dst_idx];
                current_out_pitch = width;
            }
        }

        // Clear the buffer for per-block sums (can be asynchronous if using streams)
        cudaMemset(d_blockResultsSum, 0, blockResultsSumBytes);
        // cudaMemsetAsync(d_blockResultsSum, 0, blockResultsSumBytes, stream_id); // With streams

        // 1. Collect RGB statistics per block
        CollectRgbStatistics_CUDA << < gridDimKernel, blockDimKernel >> > (
            src_ptr, d_blockResultsSum,
            width, height, current_in_pitch, is16f,
            color_space, gray_threshold
            );
        // cudaGetLastError(); // Optional: Check for kernel launch errors in debug

        // 2. Reduce block results to get overall U_avg, V_avg, totalCount
        //    Input is d_blockResultsSum, output is d_avgResult
        //    numBlocksInGridForStats is the number of float3 elements in d_blockResultsSum
        ReduceBlockResults_CUDA << < 1, threadsPerBlockReduce, sharedMemBytesReduce >> > (
            d_blockResultsSum, d_avgResult, numBlocksInGridForStats
            );
        // cudaGetLastError();

        // 3. Calculate correction matrix on GPU
        //    Input is d_avgResult, output is d_correctionMatrixGpu
        CalculateCorrectionMatrix_Kernel << < 1, 1 >> > (
            d_avgResult, d_correctionMatrixGpu,
            color_space, illuminant, chromaticAdapt
            );
        // cudaGetLastError();

        // 4. Apply color correction to the image
        //    Input is src_ptr and d_correctionMatrixGpu, output is dst_ptr
        ImageRGBCorrection_CUDA << < gridDimKernel, blockDimKernel >> > (
            src_ptr, dst_ptr,
            width, height, current_in_pitch, current_out_pitch, is16f,
            d_correctionMatrixGpu
            );
        // cudaGetLastError();

        // --- Host-side convergence check (still a bottleneck) ---
        // We need U_avg and V_avg from d_avgResult on the host for this.
        // This cudaMemcpy is synchronous by default and will stall the CPU.
        float3 h_avgResult_for_convergence;
        cudaMemcpy(&h_avgResult_for_convergence, d_avgResult, sizeof(float3), cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize(); // Not strictly needed *immediately after* a sync DtoH memcpy
        // unless subsequent operations depend on ImageRGBCorrection_CUDA's side effects
        // before this convergence check.
        // However, the logic is: AWB is applied, then we check.
        // So all GPU work for this iteration should be done.

        U_avg_host = h_avgResult_for_convergence.x;
        V_avg_host = h_avgResult_for_convergence.y;
        // float totalGrayCount_host = h_avgResult_for_convergence.z; // If needed

        // Synchronize to ensure ImageRGBCorrection_CUDA (and all prior kernels in this iteration)
        // has completed before the host makes a decision based on its output (via U_avg_host).
        // This is crucial because the convergence test relies on the state *after* correction.
        cudaDeviceSynchronize();

        const float U_avg_diff = U_avg_host - U_avg_prev_host;
        const float V_avg_diff = V_avg_host - V_avg_prev_host;
        const float normVal = std::sqrt(U_avg_diff * U_avg_diff + V_avg_diff * V_avg_diff);

        U_avg_prev_host = U_avg_host;
        V_avg_prev_host = V_avg_host;

        // Loop control based on convergence
        if (i < iter_cnt - 1) { // If not already set to be the last iteration
            if (normVal < algAWBepsilon || h_avgResult_for_convergence.z < 1.0f) { // Converged or no gray pixels
                i = iter_cnt - 1; // Force next iteration to be the "last iteration"
            }
            else {
                i++; // Continue to next iteration
            }
        }
        else { // This was already planned as the last iteration or forced to be
            i++; // Increment to terminate the loop
        }
    } // END of main processing loop

cleanup_and_exit:
    // Free all allocated CUDA memory
    cudaFree(gpuImage_ping); // Safe to call cudaFree on nullptr
    cudaFree(d_blockResultsSum);
    cudaFree(d_avgResult);
    cudaFree(d_correctionMatrixGpu);

    // It's good practice to have a final synchronize if the caller expects results to be ready on GPU,
    // or if the outBuf is immediately used by the host without further CUDA calls from the plugin.
    // The Adobe SDK might handle synchronization after the plugin returns.
    // For safety during development, a final sync can be useful.
    cudaDeviceSynchronize();

    return;
}