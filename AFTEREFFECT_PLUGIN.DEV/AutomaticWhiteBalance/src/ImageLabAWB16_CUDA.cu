// ============================================================================
//  ImageLabAWB16_Kernel.cu  --  "16f-accuracy" Automatic White Balance (CUDA)
//
//  Self-contained translation unit (no templates; separate from the 32f unit).
//
//  I/O is fp32 BGRA (identical buffers to the 32f kernel). "16f accuracy" is
//  obtained NOT by storing a half image, but by quantizing every pixel value to
//  the IEEE binary16 grid on read and on store:  q16(v) = half->float(float->half(v)).
//  The stored fp32 values are therefore exactly half-representable, so the result
//  is bit-identical to processing in a real half buffer -- but with NO pack/unpack
//  kernels and NO scratch working image.
//
//  MEMORY: the apply is per-pixel independent, so it runs IN-PLACE on outBuffer
//  (exactly like the 32f kernel). The only allocation is the 32-byte reduction
//  accumulator. The input buffer is read-only (const) and never modified.
//
//  PRECISION: per-pixel math is FP32 (q16 only pins values to the half grid at
//  the storage boundaries, matching half storage without Pascal's slow native-FP16
//  ALU path); the gray-estimate reduction accumulates in DOUBLE.
//
//  Color: encoded sRGB in/out (linearize on read, re-encode + clamp[0,1] on write).
//  Optional per-kernel timing: build with -D__GPU_KERNEL_INSTRUMENTATION.
// ============================================================================

#include "AuthomaticWhiteBalanceGPU.hpp"     // CUDA_KERNEL_CALL, RESTRICT, AlgoControls, cuda_fp16.h (__NVCC__)
#include "CUDA\AwbGpuColorMath.cuh"       // awb_host::Estimate / build_matrix / chromaticity (self-contained)
#include "CUDA\AwbGpuCommon.cuh"          // device helpers, AwbMat3, KT_* timing macros

namespace
{
    constexpr int   gMaxIter  = 16;
    constexpr float gConvEps2 = 1.0e-8f;
    constexpr int   kBlk      = 16;          // 16x16 = 256 threads (power of two)

    inline int clampi(int v, int lo, int hi) noexcept { return v < lo ? lo : (v > hi ? hi : v); }

    // Round an fp32 value to the nearest IEEE binary16-representable value.
    // This is the ONLY thing that makes the 16f path differ from the 32f path.
    __device__ __forceinline__ float q16(float v) { return __half2float(__float2half(v)); }
}

// ---------------------------------------------------------------------------
//  KERNEL 1 : masked gray-estimate reduction (values pinned to the half grid)
// ---------------------------------------------------------------------------
__global__ void awbReduce16
(
    const float* __restrict__ in,
    int   pitch,
    int   width,
    int   height,
    double* __restrict__ gSum,
    unsigned long long* __restrict__ gCount,
    float thr
)
{
    extern __shared__ float sh[];
    const int bs  = blockDim.x * blockDim.y;
    float* sR = sh; float* sG = sh + bs; float* sB = sh + 2 * bs; float* sC = sh + 3 * bs;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int y   = blockIdx.y * blockDim.y + threadIdx.y;

    float lr = 0.0f, lg = 0.0f, lb = 0.0f, lc = 0.0f;
    if (x < width && y < height)
    {
        const long long o = (static_cast<long long>(y) * pitch + x) * 4;
        const float b = awb_gpu::srgb_to_linear(q16(in[o + 0]));
        const float g = awb_gpu::srgb_to_linear(q16(in[o + 1]));
        const float r = awb_gpu::srgb_to_linear(q16(in[o + 2]));
        lc = awb_gpu::gray_select(r, g, b, thr, lr, lg, lb);
    }
    sR[tid] = lr; sG[tid] = lg; sB[tid] = lb; sC[tid] = lc;
    __syncthreads();

    for (int s = bs >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sR[tid] += sR[tid + s]; sG[tid] += sG[tid + s];
            sB[tid] += sB[tid + s]; sC[tid] += sC[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(gSum + 0, static_cast<double>(sR[0]));
        atomicAdd(gSum + 1, static_cast<double>(sG[0]));
        atomicAdd(gSum + 2, static_cast<double>(sB[0]));
        atomicAdd(gCount,   static_cast<unsigned long long>(sC[0] + 0.5f));
    }
}

// ---------------------------------------------------------------------------
//  KERNEL 2 : apply 3x3 CAT, in-place safe, values pinned to the half grid.
//  Output clamped to [0,1] (same fix as 32f: stops boosted-blue >1 from wrapping
//  to ~0 in a downstream 8-bit conversion -> yellow highlights).
// ---------------------------------------------------------------------------
__global__ void awbApply16
(
    const float* __restrict__ src, int sPitch,
    float*       __restrict__ dst, int dPitch,
    int width, int height,
    AwbMat3 M
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const long long so = (static_cast<long long>(y) * sPitch + x) * 4;
    const long long do_= (static_cast<long long>(y) * dPitch + x) * 4;

    const float b = awb_gpu::srgb_to_linear(q16(src[so + 0]));
    const float g = awb_gpu::srgb_to_linear(q16(src[so + 1]));
    const float r = awb_gpu::srgb_to_linear(q16(src[so + 2]));
    const float a = q16(src[so + 3]);

    float nr, ng, nb;
    awb_gpu::apply_matrix(M, r, g, b, nr, ng, nb);

    dst[do_ + 0] = q16(fminf(awb_gpu::linear_to_srgb(nb), 1.0f));
    dst[do_ + 1] = q16(fminf(awb_gpu::linear_to_srgb(ng), 1.0f));
    dst[do_ + 2] = q16(fminf(awb_gpu::linear_to_srgb(nr), 1.0f));
    dst[do_ + 3] = a;
}

// ===========================================================================
//  PUBLIC LAUNCHER  (fp32 BGRA in/out; in-place on outBuffer; no scratch image)
// ===========================================================================
CUDA_KERNEL_CALL
void ImageLabAWB16_CUDA
(
    const float*  RESTRICT inBuffer,
    float*  RESTRICT       outBuffer,
    int                    srcPitch,
    int                    dstPitch,
    int                    width,
    int                    height,
    const AlgoControls* RESTRICT algoGpuParams,
    cudaStream_t           stream
)
{
    const int   iterCnt   = clampi(algoGpuParams->sliderIterCnt, 1, gMaxIter);
    const float threshold = static_cast<float>(algoGpuParams->sliderThreshold) * 0.01f;

    // ---- ONE tiny allocation: just the reduction accumulators --------------
    // No half working image: values are pinned to the half grid via q16(), and
    // the apply runs in-place on outBuffer (per-pixel independent).
    const size_t accBytes = 3 * sizeof(double) + sizeof(unsigned long long); // 32

    void* arena = nullptr;
    if (cudaSuccess != cudaMalloc(&arena, accBytes))
        return;

    double*             gSum   = reinterpret_cast<double*>(arena);
    unsigned long long* gCount = reinterpret_cast<unsigned long long*>(
                                     static_cast<char*>(arena) + 3 * sizeof(double));

    const dim3   block(kBlk, kBlk);
    const dim3   grid((width + kBlk - 1) / kBlk, (height + kBlk - 1) / kBlk);
    const size_t shBytes = static_cast<size_t>(4) * block.x * block.y * sizeof(float);

    KT_DECL(stream);

    auto estimate_build = [&](const float* buf, int pitch, awb_host::Estimate& e, float Mout[9])
    {
        cudaMemsetAsync(arena, 0, accBytes, stream);
        KT_BEG("reduce");
        awbReduce16<<<grid, block, shBytes, stream>>>(buf, pitch, width, height, gSum, gCount, threshold);
        KT_END();
        double hS[3] = { 0.0, 0.0, 0.0 };
        unsigned long long hC = 0;
        cudaMemcpyAsync(hS,  gSum,   3 * sizeof(double),         cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&hC, gCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        e.sumR = hS[0]; e.sumG = hS[1]; e.sumB = hS[2];
        e.count = static_cast<long long>(hC);
        awb_host::build_matrix(e, *algoGpuParams, Mout);
    };

    auto to_mat3 = [](const float M[9]) { AwbMat3 r; for (int i = 0; i < 9; ++i) r.m[i] = M[i]; return r; };

    // ---- pass 0 : estimate from source, write balanced result to output ----
    awb_host::Estimate est; float M[9];
    estimate_build(inBuffer, srcPitch, est, M);
    KT_BEG("apply");
    awbApply16<<<grid, block, 0, stream>>>(inBuffer, srcPitch, outBuffer, dstPitch, width, height, to_mat3(M));
    KT_END();

    if (iterCnt <= 1)
    {
        cudaStreamSynchronize(stream);
        KT_REPORT("AWB16");
        cudaFree(arena);
        return;
    }

    // ---- iterative refinement (in-place on outBuffer) ----------------------
    float px, py; awb_host::chromaticity(est, px, py);
    for (int k = 1; k < iterCnt; ++k)
    {
        awb_host::Estimate ek; float Mk[9];
        estimate_build(outBuffer, dstPitch, ek, Mk);

        float cx, cy; awb_host::chromaticity(ek, cx, cy);
        const float du = cx - px, dv = cy - py;
        if ((du * du + dv * dv) < gConvEps2) break;
        px = cx; py = cy;

        KT_BEG("apply");
        awbApply16<<<grid, block, 0, stream>>>(outBuffer, dstPitch, outBuffer, dstPitch, width, height, to_mat3(Mk));
        KT_END();
    }

    cudaDeviceSynchronize();
    KT_REPORT("AWB16");
    cudaFree(arena);
}

// Single definition per module (see note in the 32f unit).
CUDA_KERNEL_CALL
void ImageLabAWB16_CleanupGPU()
{
}
