// ============================================================================
//  ImageLabAWB32_Kernel.cu   --  BGRA_32f Automatic White Balance (CUDA)
//
//  Self-contained translation unit (no templates). Pipeline per frame:
//    pass 0 : reduce(inBuffer) -> gray estimate -> host builds 3x3 CAT
//             -> apply(inBuffer -> outBuffer)
//    iter k : reduce(current) -> estimate -> (chromaticity convergence test)
//             -> apply(current -> next); ping-pong, final result lands in outBuffer
//
//  Mirrors the validated CPU Algorithm_Main: same gray metric, same matrix build
//  (reused on the host), same iteration / convergence (gConvEps2 = 1e-8, max 16).
//
//  Color: inBuffer/outBuffer are gamma sRGB BGRA float (matching the CPU BGRA_32f
//  path). Each pixel is linearized on read and re-encoded on write; the gray-
//  estimate reduction accumulates in DOUBLE (precise over megapixel frames).
//  >> If Premiere hands this path SCENE-LINEAR float, drop the srgb_to_linear /
//     linear_to_srgb calls (one spot each in the two kernels). <<
//
//  Memory: ONE cudaMalloc (arena); sub-buffers sliced from it; freed at the end.
// ============================================================================

#include "AuthomaticWhiteBalanceGPU.hpp"     // CUDA_KERNEL_CALL, RESTRICT, AlgoControls, cudaStream_t, cuda_fp16.h
#include "CUDA\AwbGpuColorMath.cuh"       // awb_host::Estimate / build_matrix / chromaticity (self-contained, no CPU lib)
#include "CUDA\AwbGpuCommon.cuh"          // device helpers, AwbMat3

namespace
{
    constexpr int   gMaxIter = 16;
    constexpr float gConvEps2 = 1.0e-8f;
    constexpr int   kBlk = 16;          // 16x16 = 256 threads (power of two)

    inline int clampi(int v, int lo, int hi) noexcept { return v < lo ? lo : (v > hi ? hi : v); }
}

// ---------------------------------------------------------------------------
//  KERNEL 1 : masked gray-estimate reduction (BGRA_32f, gamma sRGB in)
//  Block-local sums in shared memory, one atomicAdd per block to double globals.
// ---------------------------------------------------------------------------
__global__ void awbReduce32
(
    const float* __restrict__ in,
    int   pitch,                 // BGRA pixels/row
    int   width,
    int   height,
    double* __restrict__ gSum,   // [3] : sum R,G,B (linear) over selected pixels
    unsigned long long* __restrict__ gCount,
    float thr
)
{
    extern __shared__ float sh[];
    const int bs = blockDim.x * blockDim.y;
    float* sR = sh; float* sG = sh + bs; float* sB = sh + 2 * bs; float* sC = sh + 3 * bs;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    float lr = 0.0f, lg = 0.0f, lb = 0.0f, lc = 0.0f;
    if (x < width && y < height)
    {
        const long long o = (static_cast<long long>(y) * pitch + x) * 4;
        const float b = awb_gpu::srgb_to_linear(in[o + 0]);
        const float g = awb_gpu::srgb_to_linear(in[o + 1]);
        const float r = awb_gpu::srgb_to_linear(in[o + 2]);
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
        atomicAdd(gCount, static_cast<unsigned long long>(sC[0] + 0.5f));
    }
}

// ---------------------------------------------------------------------------
//  KERNEL 2 : apply 3x3 CAT (decode -> matrix -> floor 0 -> encode), alpha copied
// ---------------------------------------------------------------------------
__global__ void awbApply32
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
    const long long do_ = (static_cast<long long>(y) * dPitch + x) * 4;

    const float b = awb_gpu::srgb_to_linear(src[so + 0]);
    const float g = awb_gpu::srgb_to_linear(src[so + 1]);
    const float r = awb_gpu::srgb_to_linear(src[so + 2]);
    const float a = src[so + 3];

    float nr, ng, nb;
    awb_gpu::apply_matrix(M, r, g, b, nr, ng, nb);

    // Clamp to [0,1] to match the CPU integer/SDR reference: a white-balance that
    // boosts blue can push bright pixels >1, which a downstream 8-bit conversion
    // WRAPS toward 0 -> yellow highlights. (For a true HDR float project, leave
    // these unclamped and clamp in the float->8-bit export stage instead.)
    dst[do_ + 0] = fminf(awb_gpu::linear_to_srgb(nb), 1.0f);
    dst[do_ + 1] = fminf(awb_gpu::linear_to_srgb(ng), 1.0f);
    dst[do_ + 2] = fminf(awb_gpu::linear_to_srgb(nr), 1.0f);
    dst[do_ + 3] = a;
}

// ===========================================================================
//  PUBLIC LAUNCHER
// ===========================================================================
CUDA_KERNEL_CALL
void ImageLabAWB32_CUDA
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
    const int   iterCnt = clampi(algoGpuParams->sliderIterCnt, 1, gMaxIter);
    const float threshold = static_cast<float>(algoGpuParams->sliderThreshold) * 0.01f;

    // ---- ONE tiny allocation: just the reduction accumulators --------------
    // apply is per-pixel independent -> done in-place on outBuffer, so there is
    // no scratch image and no final copy (this removes the big per-call
    // cudaMalloc/cudaFree that dominated the wall-clock).
    const size_t accBytes = 3 * sizeof(double) + sizeof(unsigned long long); // 32

    void* arena = nullptr;
    if (cudaSuccess != cudaMalloc(&arena, accBytes))
        return;

    double*             gSum = reinterpret_cast<double*>(arena);
    unsigned long long* gCount = reinterpret_cast<unsigned long long*>(
        static_cast<char*>(arena) + 3 * sizeof(double));

    const dim3   block(kBlk, kBlk);
    const dim3   grid((width + kBlk - 1) / kBlk, (height + kBlk - 1) / kBlk);
    const size_t shBytes = static_cast<size_t>(4) * block.x * block.y * sizeof(float);

    KT_DECL(stream);

    // host-side estimate (reduction) + matrix build (reuses validated CPU code)
    auto estimate_build = [&](const float* buf, int pitch, awb_host::Estimate& e, float Mout[9])
    {
        cudaMemsetAsync(arena, 0, accBytes, stream);             // zero accumulators only
        KT_BEG("reduce");
        awbReduce32 << <grid, block, shBytes, stream >> >(buf, pitch, width, height, gSum, gCount, threshold);
        KT_END();
        double hS[3] = { 0.0, 0.0, 0.0 };
        unsigned long long hC = 0;
        cudaMemcpyAsync(hS, gSum, 3 * sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&hC, gCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);                            // sums needed on host
        e.sumR = hS[0]; e.sumG = hS[1]; e.sumB = hS[2];
        e.count = static_cast<long long>(hC);
        awb_host::build_matrix(e, *algoGpuParams, Mout);
    };

    auto to_mat3 = [](const float M[9]) { AwbMat3 r; for (int i = 0; i < 9; ++i) r.m[i] = M[i]; return r; };

    // ---- pass 0 : estimate from source, write balanced result to output ----
    awb_host::Estimate est; float M[9];
    estimate_build(inBuffer, srcPitch, est, M);
    KT_BEG("apply");
    awbApply32 << <grid, block, 0, stream >> >(inBuffer, srcPitch, outBuffer, dstPitch, width, height, to_mat3(M));
    KT_END();

    if (iterCnt <= 1)
    {
        cudaStreamSynchronize(stream);
        KT_REPORT("AWB32");
        cudaFree(arena);
        return;
    }

    // ---- iterative refinement (in-place on outBuffer) ----------------------
    // apply reads and writes the same pixel, so src==dst is safe; re-estimate
    // runs on the corrected image. No ping-pong scratch, no final copy.
    float px, py; awb_host::chromaticity(est, px, py);
    for (int k = 1; k < iterCnt; ++k)
    {
        awb_host::Estimate ek; float Mk[9];
        estimate_build(outBuffer, dstPitch, ek, Mk);

        float cx, cy; awb_host::chromaticity(ek, cx, cy);
        const float du = cx - px, dv = cy - py;
        if ((du * du + dv * dv) < gConvEps2) break;              // converged
        px = cx; py = cy;

        KT_BEG("apply");
        awbApply32 << <grid, block, 0, stream >> >(outBuffer, dstPitch, outBuffer, dstPitch, width, height, to_mat3(Mk));
        KT_END();
    }

    cudaDeviceSynchronize();
    KT_REPORT("AWB32");
    cudaFree(arena);
}

// All GPU memory is allocated and freed per call, so there is no persistent
// state to release. (Single definition per module: if you ever link the 32f and
// 16f units into one binary, keep only one ImageLabDenoise_CleanupGPU.)
CUDA_KERNEL_CALL
void ImageLabAWB32_CleanupGPU()
{
}
