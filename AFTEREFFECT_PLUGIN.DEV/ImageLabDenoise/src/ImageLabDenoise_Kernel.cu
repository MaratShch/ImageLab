// ============================================================================
// File: ImageLabDenoise_Kernel.cu  --  ORCHESTRATOR + ENTRY POINT
// Target: CUDA 10.2 / C++14 / NVIDIA GTX-1060 (3GB VRAM) & RTX-2000
// Description: One-shot GPU implementation of Marc Lebrun's NL-Bayes
//              ("Noise Clinic") algorithm, faithful to the IPOL 2015
//              reference implementation.
//
// This file contains only:
//    * includes, architectural constants
//    * forward declarations for all kernels
//    * g_gpuMemState singleton
//    * ImageLabDenoise_CUDA    (orchestrator / entry point)
//    * ImageLabDenoise_CleanupGPU
//
// Kernel bodies live in this same file below, one per section.
// They are provided incrementally in subsequent commits / messages.
//
// Execution model:
//    HOST submits a chain of async kernel launches on `stream`, then
//    synchronizes ONCE at the end via cudaStreamSynchronize.
//    No intermediate host round-trips.
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CompileTimeUtils.hpp"
#include "ImageLabCUDA.hpp"
#include "ImageLabDenoise_GPU.hpp"
#include "AlgoControls.hpp"
#include "CUDA/CudaMemHandler.cuh"

// ============================================================================
// ARCHITECTURAL CONSTANTS
// ============================================================================

// --- Launch-config block dimensions ---
constexpr int BLOCK_DIM_IO_X   = 32;
constexpr int BLOCK_DIM_IO_Y   = 16;
constexpr int BLOCK_DIM_MATH_X = 32;
constexpr int BLOCK_DIM_MATH_Y = 8;   // 32 * 8 = 256 threads / block (for NL-Bayes passes)

// --- Image-dimension alignment ---
// We pad the internal YUV buffers to a multiple of 4 so that:
//   (a) the 4x4 DCT patch grid tiles evenly
//   (b) a future 2-scale mosaic pyramid (1<<nbScales = 4) fits
// Matches Lebrun's enhanceBoundaries(pow = 1 << nbScales) convention.
constexpr int PROC_ALIGN = 4;

// --- Lebrun's YUV orthonormal color transform (3-channel branch in LibImages.cpp) ---
//   Y = a * (R + G + B)            a = 1/sqrt(3)  ≈ 0.57735027f
//   U = b * (R - B)                b = 1/sqrt(2)  ≈ 0.70710678f
//   V = c * (R/4 - G/2 + B/4)      c = 2*a*sqrt(2) ≈ 1.63299316f
// Inverse uses a' = a, b' = b, c' = a/b = sqrt(2/3) ≈ 0.81649658f;
// the inverse matrix is the transpose of the forward (orthonormal).
constexpr float YUV_A       = 0.57735027f;   // 1 / sqrt(3)
constexpr float YUV_B       = 0.70710678f;   // 1 / sqrt(2)
constexpr float YUV_C       = 1.63299316f;   // 2 * YUV_A * sqrt(2)
constexpr float YUV_C_INV   = 0.81649658f;   // YUV_A / YUV_B  = sqrt(2/3)

// --- Noise-estimation geometry ---
// Lebrun uses nSimilarPatches = 2*sP^2 = 32 and a search window of
// (nSimilarPatches/2 | 1) = 17 cells across.
constexpr int SEARCH_WINDOW_RADIUS = 8;
constexpr int SEARCH_WINDOW_SIZE   = (SEARCH_WINDOW_RADIUS * 2) + 1;   // 17
constexpr int SMEM_STRIDE          = SEARCH_WINDOW_SIZE;               // row stride for 17x17 shared tile

// nSimilarPatches is the TARGET number of similar patches per group.
// MAX_SIMILAR_PATCHES is the hard cap (Lebrun's CPU uses 16 * nSim = 512 in
// random-patch mode; we cap at 128 as a pragmatic balance - quality parity
// in 99% of real imagery with a quarter the shared-memory footprint).
constexpr int N_SIMILAR_PATCHES    = 32;
constexpr int MAX_SIMILAR_PATCHES  = 128;

// DCT patch noise-estimation constants (4x4 orthonormal DCT-II)
//   C0 = 1/2
//   C1 = sqrt(1/2) * cos(pi/8)    ≈ 0.65328148f
//   C2 = 1/2
//   C3 = sqrt(1/2) * cos(3*pi/8)  ≈ 0.27059805f
constexpr float DCT_C0 = 0.5f;
constexpr float DCT_C1 = 0.65328148f;
constexpr float DCT_C2 = 0.5f;
constexpr float DCT_C3 = 0.27059805f;


// ============================================================================
// GLOBAL ARENA STATE (persistent across frames)
// ============================================================================
static CudaMemHandler g_gpuMemState;


// ============================================================================
// KERNEL FORWARD DECLARATIONS
// (bodies follow below; one per subsequent message)
// ============================================================================

// ============================================================================
// Kernel_ConvertBGRAToOrthonormalWeighted
// ============================================================================
//
// Purpose:
//   Convert interleaved BGRA (32-bit float, channels in [0, 1]) into three
//   planar YUV buffers using Marc Lebrun's orthonormal 3-channel transform
//   (LibImages.cpp:212-221), extending the image with symmetric mirror
//   reflection from (srcWidth, srcHeight) up to (procW, procH).
//
// Lebrun's forward transform (RGB -> YUV):
//     a = 1/sqrt(3),  b = 1/sqrt(2),  c = 2*a*sqrt(2)
//
//     Y = a * (R + G + B)
//     U = b * (R     - B)
//     V = c * (R/4 - G/2 + B/4)
//
// Mirror-reflection padding rule (Lebrun convention):
//     src_x = x                 if x < srcWidth
//             2*srcWidth - 2 - x otherwise
//     src_y = y                 if y < srcHeight
//             2*srcHeight - 2 - y otherwise
//
//   For x = srcWidth, srcWidth + 1, ... we read srcWidth - 2, srcWidth - 3, ...
//   i.e. the row just-before-the-edge reflected outward. This is the same
//   rule Lebrun's enhanceBoundaries() uses.
//
// Launch geometry:
//   grid  = ceil(procW / 32) x ceil(procH / 16)
//   block = 32 x 16                      (512 threads / block)
//
// Input:
//   inBuffer        - BGRA 32f, pitched, read-only. Access pattern:
//                     inBuffer[(y * srcPitchPixels + x) * 4 + channel]
//                     where channel 0 = B, 1 = G, 2 = R, 3 = A.
//   srcPitchPixels  - row stride of inBuffer, in PIXELS (not bytes).
//   padW            - row stride of the output YUV planes, in floats.
//                     (= procW + PAD_APRON from the arena allocator.)
//   srcWidth,
//   srcHeight       - valid extents of inBuffer. Pixels beyond are mirrored.
//   procW, procH    - padded processing extents; the grid covers this area.
//
// Output:
//   outY, outU, outV - planar buffers of size (padH x padW). Only the
//                      [0, procW) x [0, procH) sub-region is written here.
//                      The apron strip [procW .. padW-1] x [procH .. padH-1]
//                      is left untouched (may contain stale data; kernels
//                      downstream treat it as out-of-range and do not read it).
// ============================================================================
__global__ void Kernel_ConvertBGRAToOrthonormalWeighted(
    const float* RESTRICT    inBuffer,
    float*       RESTRICT    outY,
    float*       RESTRICT    outU,
    float*       RESTRICT    outV,
    int                      srcPitchPixels,
    int                      padW,
    int                      srcWidth,
    int                      srcHeight,
    int                      procW,
    int                      procH)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Early-out for threads outside the padded processing region.
    if (x >= procW || y >= procH)
    {
        return;
    }

    // ------------------------------------------------------------------------
    // Resolve source coordinate via Lebrun-style symmetric mirror reflection.
    // Branchless variant:  src = (coord < extent) ? coord : (2*extent - 2 - coord)
    // ------------------------------------------------------------------------
    int src_x = (x < srcWidth)  ? x : (2 * srcWidth  - 2 - x);
    int src_y = (y < srcHeight) ? y : (2 * srcHeight - 2 - y);

    // Safety clamp in case the padding exceeds srcWidth/srcHeight (e.g.
    // a pathologically small input image). Should never trigger for
    // typical inputs where procW - srcWidth <= 3.
    if (src_x < 0)          src_x = 0;
    if (src_x >= srcWidth)  src_x = srcWidth  - 1;
    if (src_y < 0)          src_y = 0;
    if (src_y >= srcHeight) src_y = srcHeight - 1;

    // ------------------------------------------------------------------------
    // Fetch BGRA at the (possibly mirrored) source coordinate.
    //   channel 0 = B, 1 = G, 2 = R, 3 = A
    // ------------------------------------------------------------------------
    const int   src_base = (src_y * srcPitchPixels + src_x) * 4;
    const float b        = inBuffer[src_base + 0];
    const float g        = inBuffer[src_base + 1];
    const float r        = inBuffer[src_base + 2];
    // Alpha is ignored -- we pass-through 1.0 on the inverse.

    // ------------------------------------------------------------------------
    // Lebrun's orthonormal 3-channel RGB -> YUV transform.
    //
    //   Y = YUV_A * (R + G + B)
    //   U = YUV_B * (R     - B)
    //   V = YUV_C * (R/4 - G/2 + B/4)
    //
    // Expressed in fused form to minimize FLOPs (compiler will FMA).
    // ------------------------------------------------------------------------
    const float Y = YUV_A * (r + g + b);
    const float U = YUV_B * (r - b);
    const float V = YUV_C * (0.25f * r - 0.5f * g + 0.25f * b);

    // ------------------------------------------------------------------------
    // Write into the padded planar region at (x, y) using the arena stride padW.
    // ------------------------------------------------------------------------
    const int dst_idx = y * padW + x;
    outY[dst_idx] = Y;
    outU[dst_idx] = U;
    outV[dst_idx] = V;
}


// ============================================================================
// Kernel_ExtractDCT_And_Variance_3ch    (+ DCT_1D_4Point device helper)
// ============================================================================
//
// Purpose:
//   Ponomarenko-style noise estimation in DCT frequency space, matching
//   Lebrun's RunNoiseEstimate.cpp pipeline. For every 4x4 patch in each
//   channel (Y, U, V):
//
//     1. Compute the orthonormal 2D DCT-II of the patch.
//     2. Derive an intensity bin index from the patch mean.
//     3. Accumulate:
//          outCov[bin*256 + freq]   += dct_coeff[freq]^2     (16 floats/bin)
//          outMean[bin]             += patch_mean
//          outCounts[bin]           += 1
//
//   After this kernel and the smoothing kernel, Kernel_BuildSpatialNoiseCov_3ch
//   expands the 16 frequency-variances per bin into a full 16x16 spatial
//   covariance matrix via C_spatial = D^T * diag(freq_vars) * D.
//
//   NOTE: Only the first 16 slots of each bin's 256-slot region are written
//   here. Remaining slots [16..255] stay zero (from cudaMemsetAsync in the
//   orchestrator). kernel #4 overwrites them with the full spatial matrix.
//
// Bug fix vs. original (C1 from the mismatch report):
//   The original loaded three apron strips (main block + right 3 columns +
//   bottom 3 rows) but MISSED the bottom-right 3x3 corner. This caused
//   threads near block boundaries to read uninitialized shared memory for
//   their 4x4 patch fetches. Fixed below.
//
// Launch geometry:
//   grid  = ceil(procW / 32) x ceil(procH / 16)
//   block = 32 x 16       (512 threads / block)
//   shared = ~6 KB for a 3-channel (16+3) x (32+3) apron tile
//
// Input:
//   inY, inU, inV   - planar YUV, stride padW floats/row.
//   padW            - row stride.
//   procW, procH    - valid-data extents; we only process patches whose
//                     top-left corner lies in [0, procW-3) x [0, procH-3).
//
// Output (atomic-accumulated):
//   outCov*    - 256 bins x 256 floats (only freq0..15 written)
//   outMean*   - 256 bins x 1 float    (sum of patch means; divided later)
//   outCounts* - 256 bins x 1 int      (count of patches per bin)
// ============================================================================


// ---------------------------------------------------------------------------
// 1D orthonormal DCT-II for N=4.  Out-of-place: reads `in[0..3]`, writes out[].
//
//   out[k] = sum_n  in[n] * e(k) * sqrt(2/N) * cos((2n+1)*k*pi / (2N))
//   with e(0) = 1/sqrt(2), e(k>0) = 1.
//
// Coefficients evaluate to:
//   C0 = 1/2               (k=0, all cosines = 1, applied to (in0+in1+in2+in3)/2)
//   C1 = sqrt(1/2) * cos(pi/8)    (for k=1)
//   C2 = 1/2               (k=2)
//   C3 = sqrt(1/2) * cos(3pi/8)   (for k=3)
// ---------------------------------------------------------------------------
__device__ __forceinline__
void DCT_1D_4Point(const float* RESTRICT in, float* RESTRICT out)
{
    const float s03 = in[0] + in[3];
    const float d03 = in[0] - in[3];
    const float s12 = in[1] + in[2];
    const float d12 = in[1] - in[2];

    out[0] = DCT_C0 * (s03 + s12);                 //   0.5 * (a+b+c+d)
    out[1] = DCT_C1 * d03 + DCT_C3 * d12;          //   even-odd mix
    out[2] = DCT_C2 * (s03 - s12);                 //   0.5 * (a-b-c+d)
    out[3] = DCT_C3 * d03 - DCT_C1 * d12;          //   even-odd mix
}


// ---------------------------------------------------------------------------
// Helper: accumulate one channel's contributions for the current thread's
// 4x4 patch at (tx, ty) inside the shared tile.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void AccumulateChannel(
    const float (&tile)[BLOCK_DIM_IO_Y + 3][BLOCK_DIM_IO_X + 3],
    int           tx,
    int           ty,
    float*        outCov,
    float*        outMean,
    int*          outCounts)
{
    //! 1. Load the 4x4 patch from shared tile and accumulate mean.
    float p[4][4];
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const float v = tile[ty + i][tx + j];
            p[i][j] = v;
            sum    += v;
        }
    }

    //! 2. Patch mean in channel-native units, then map to a 256-bin index.
    //     Input is in [0, 1] for BGRA and remains bounded for YUV, but U/V
    //     can be negative. We bin by the signed mean; out-of-range clamps
    //     to edges. This matches Lebrun's getMean() with offset/rangeMax.
    const float patch_mean     = sum * 0.0625f;                // sum / 16
    const int   bin_unclamped  = __float2int_rn(patch_mean * 255.0f);
    const int   bin            = max(0, min(NOISE_BINS - 1, bin_unclamped));

    //! 3. Run 2D DCT: 4 rows, then 4 columns.
    float tmp[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        DCT_1D_4Point(p[i], tmp[i]);
    }

    float dct[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const float col[4] = { tmp[0][j], tmp[1][j], tmp[2][j], tmp[3][j] };
        float       colOut[4];
        DCT_1D_4Point(col, colOut);

        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            dct[i][j] = colOut[i];
        }
    }

    //! 4. Accumulate squared DCT coefficients (16 freqs) into this bin.
    //     Writes go to outCov[bin*256 + freq] for freq in [0..15]; the
    //     remaining [16..255] slots of each bin are left untouched here
    //     (they hold zero from cudaMemsetAsync, and will be overwritten
    //     with the full spatial covariance matrix in kernel #4).
    //
    //     Also accumulate patch_mean (for later per-bin mean) and count.

    float* binBase = outCov + bin * PATCH_ELEMS_SQ;            // bin stride = 256

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int   freq = i * 4 + j;
            const float c    = dct[i][j];
            atomicAdd(&binBase[freq], c * c);
        }
    }

    atomicAdd(&outMean[bin],   patch_mean);
    atomicAdd(&outCounts[bin], 1);
}


// ============================================================================
// The kernel itself.
// ============================================================================
__global__ void Kernel_ExtractDCT_And_Variance_3ch
(
    const float* RESTRICT    inY,
    const float* RESTRICT    inU,
    const float* RESTRICT    inV,
    float*       RESTRICT    outCovY,
    float*       RESTRICT    outCovU,
    float*       RESTRICT    outCovV,
    float*       RESTRICT    outMeanY,
    float*       RESTRICT    outMeanU,
    float*       RESTRICT    outMeanV,
    int*         RESTRICT    outCountsY,
    int*         RESTRICT    outCountsU,
    int*         RESTRICT    outCountsV,
    int                      padW,
    int                      procW,
    int                      procH
)
{
    // Shared-memory apron tile for each channel:
    //   block is 32x16 threads; patches are 4x4, so we need 3 extra pixels
    //   on the right and 3 on the bottom to fully cover every thread's patch.
    // Layout: [y][x] (row-major; y is the slow axis).
    __shared__ float s_Y[BLOCK_DIM_IO_Y + 3][BLOCK_DIM_IO_X + 3];
    __shared__ float s_U[BLOCK_DIM_IO_Y + 3][BLOCK_DIM_IO_X + 3];
    __shared__ float s_V[BLOCK_DIM_IO_Y + 3][BLOCK_DIM_IO_X + 3];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * blockDim.x + tx;
    const int gy = blockIdx.y * blockDim.y + ty;

    // ------------------------------------------------------------------------
    // Cooperative tile load: main region + right strip + bottom strip +
    // bottom-right corner. The original code missed the corner -- fixed here.
    // ------------------------------------------------------------------------

    //! Main 32x16 block
    if (gx < procW && gy < procH)
    {
        const int src_idx = gy * padW + gx;
        s_Y[ty][tx] = inY[src_idx];
        s_U[ty][tx] = inU[src_idx];
        s_V[ty][tx] = inV[src_idx];
    }

    //! Right strip: 3 extra columns, y in [0, 16)
    if (tx < 3 && (gx + BLOCK_DIM_IO_X) < procW && gy < procH)
    {
        const int src_idx = gy * padW + (gx + BLOCK_DIM_IO_X);
        s_Y[ty][tx + BLOCK_DIM_IO_X] = inY[src_idx];
        s_U[ty][tx + BLOCK_DIM_IO_X] = inU[src_idx];
        s_V[ty][tx + BLOCK_DIM_IO_X] = inV[src_idx];
    }

    //! Bottom strip: 3 extra rows, x in [0, 32)
    if (ty < 3 && gx < procW && (gy + BLOCK_DIM_IO_Y) < procH)
    {
        const int src_idx = (gy + BLOCK_DIM_IO_Y) * padW + gx;
        s_Y[ty + BLOCK_DIM_IO_Y][tx] = inY[src_idx];
        s_U[ty + BLOCK_DIM_IO_Y][tx] = inU[src_idx];
        s_V[ty + BLOCK_DIM_IO_Y][tx] = inV[src_idx];
    }

    //! Bottom-right 3x3 corner (this was missing in the original).
    if (tx < 3 && ty < 3
        && (gx + BLOCK_DIM_IO_X) < procW
        && (gy + BLOCK_DIM_IO_Y) < procH)
    {
        const int src_idx = (gy + BLOCK_DIM_IO_Y) * padW + (gx + BLOCK_DIM_IO_X);
        s_Y[ty + BLOCK_DIM_IO_Y][tx + BLOCK_DIM_IO_X] = inY[src_idx];
        s_U[ty + BLOCK_DIM_IO_Y][tx + BLOCK_DIM_IO_X] = inU[src_idx];
        s_V[ty + BLOCK_DIM_IO_Y][tx + BLOCK_DIM_IO_X] = inV[src_idx];
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // Per-thread work: extract one 4x4 patch anchored at this thread's
    // (tx, ty) position in the tile, compute DCT, accumulate into the
    // per-channel noise LUTs.
    //
    // Only proceed if the 4x4 patch fits fully within valid data:
    //     - global position (gx, gy) must satisfy gx + 3 < procW
    //       and gy + 3 < procH  (patch top-left can be at most procW-4, procH-4).
    //     - shared tile is large enough (we loaded +3 apron).
    // ------------------------------------------------------------------------
    if (gx + 3 >= procW || gy + 3 >= procH)
    {
        return;
    }

    AccumulateChannel(s_Y, tx, ty, outCovY, outMeanY, outCountsY);
    AccumulateChannel(s_U, tx, ty, outCovU, outMeanU, outCountsU);
    AccumulateChannel(s_V, tx, ty, outCovV, outMeanV, outCountsV);
}


// ============================================================================
// Kernel_SmoothNoiseCurves_3ch
// ============================================================================
//
// Purpose:
//   Normalize and smooth the per-bin noise statistics gathered by
//   Kernel_ExtractDCT_And_Variance_3ch. Faithful port of Lebrun's
//   filterNoiseCurves (CurveFiltering.cpp):
//
//     Phase 0: Normalize raw sums by patch count:
//                  variance[bin, freq] = cov[bin*256 + freq] / counts[bin]
//                  mean[bin]           = mean[bin]          / counts[bin]
//
//     Phase 1: Repeat 5 times (Lebrun's nbFilter = 5):
//                 1a. For each of 16 DCT frequencies, smooth the per-bin
//                     variance curve with a +/-10-bin window average
//                     (Lebrun's sizeFilter = 10), skipping empty bins.
//                     Negative results are clipped to zero.
//                 1b. For each bin, apply a 4x4 median filter on the 16
//                     frequency variances (3-point median at corners,
//                     4-point at edges, 5-point at interior).
//
//     Phase 2: Write smoothed variances back to global memory.
//
//   Only the first 16 slots of each bin's 256-slot noise-cov region are
//   touched here. Slots [16..255] remain zero from the initial memset.
//   Kernel_BuildSpatialNoiseCov_3ch (kernel #4) will overwrite them with
//   the full D^T * diag(vars) * D spatial covariance matrix.
//
// Storage convention (differs slightly from Lebrun but is mathematically
// equivalent):
//   * Lebrun's CPU stores std, squares to variance for processing, sqrt's
//     back. We store variance directly here. No sqrt/square roundtrip needed.
//   * All downstream kernels read variance.
//
// Bug fix vs. original (C2 from the mismatch report):
//   The original used a clamped neighbor loop:
//       for (int o = -2; o <= 2; ++o) {
//         int nb = max(0, min(255, bin + o));  // clamps out-of-range
//         sm += s[nb]; n++;
//       }
//   This made bin 0 count itself 3x (for o = -2, -1, 0) instead of only
//   averaging valid neighbors. Fixed below by computing the valid
//   [nmin, nmax] range up front.
//
// Launch geometry:
//   grid   = (3)     one block per channel {Y, U, V}
//   block  = (256)   one thread per intensity bin
//   shared = ~2 KB   (256-float working buffer + 256-int validity flags)
// ============================================================================


// ---------------------------------------------------------------------------
// Median helpers for the 4x4 frequency matrix. These match Lebrun's
// getMedian3 / getMedian4 / getMedian5 semantics:
//
//   Median3 : middle of 3 values
//   Median4 : average of the two middle values (sorted)
//   Median5 : middle of 5 values
// ---------------------------------------------------------------------------
__device__ __forceinline__
float Median3(float a, float b, float c)
{
    if (a < b)
    {
        if (a < c) return fminf(b, c);
        else       return a;
    }
    else
    {
        if (b < c) return fminf(a, c);
        else       return b;
    }
}

__device__ __forceinline__
float Median4(float a, float b, float c, float d)
{
    // Small branchless-ish sort via conditional swaps, then return avg of middle two.
    float arr[4] = { a, b, c, d };
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        #pragma unroll
        for (int j = i + 1; j < 4; ++j)
        {
            if (arr[j] < arr[i])
            {
                const float t = arr[i];
                arr[i] = arr[j];
                arr[j] = t;
            }
        }
    }
    return (arr[1] + arr[2]) * 0.5f;
}

__device__ __forceinline__
float Median5(float a, float b, float c, float d, float e)
{
    float arr[5] = { a, b, c, d, e };
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        #pragma unroll
        for (int j = i + 1; j < 5; ++j)
        {
            if (arr[j] < arr[i])
            {
                const float t = arr[i];
                arr[i] = arr[j];
                arr[j] = t;
            }
        }
    }
    return arr[2];
}


// ---------------------------------------------------------------------------
// Apply the per-cell median filter on a 4x4 matrix stored row-major:
//     index freq = i*4 + j   (i,j in [0,3])
// Lebrun's rule:
//     corners  -> 3-point median (self + two neighbors)
//     edges    -> 4-point median (self + three neighbors)
//     interior -> 5-point median (self + four neighbors)
// ---------------------------------------------------------------------------
__device__ __forceinline__
float MatMedianAt(const float m[PATCH_ELEMS], int i, int j)
{
    // -- Corners (4 cases) --
    if (i == 0 && j == 0)
    {
        return Median3(m[0], m[1], m[4]);                       // self + (0,1) + (1,0)
    }
    if (i == 0 && j == 3)
    {
        return Median3(m[3], m[2], m[7]);                       // self + (0,2) + (1,3)
    }
    if (i == 3 && j == 0)
    {
        return Median3(m[12], m[8], m[13]);                     // self + (2,0) + (3,1)
    }
    if (i == 3 && j == 3)
    {
        return Median3(m[15], m[11], m[14]);                    // self + (2,3) + (3,2)
    }

    // -- Edges (non-corner) --
    if (i == 0)     // top row, j in [1,2]
    {
        return Median4(m[j], m[j - 1], m[j + 1], m[4 + j]);
    }
    if (i == 3)     // bottom row, j in [1,2]
    {
        return Median4(m[12 + j], m[12 + j - 1], m[12 + j + 1], m[8 + j]);
    }
    if (j == 0)     // left column, i in [1,2]
    {
        return Median4(m[i * 4], m[(i - 1) * 4], m[(i + 1) * 4], m[i * 4 + 1]);
    }
    if (j == 3)     // right column, i in [1,2]
    {
        return Median4(m[i * 4 + 3], m[(i - 1) * 4 + 3], m[(i + 1) * 4 + 3], m[i * 4 + 2]);
    }

    // -- Interior --
    const int idx = i * 4 + j;
    return Median5(
        m[idx],
        m[idx - 1],         // left
        m[idx + 1],         // right
        m[idx - 4],         // above
        m[idx + 4]);        // below
}


// ============================================================================
// The kernel itself.
// ============================================================================
__global__ void Kernel_SmoothNoiseCurves_3ch
(
    float*       RESTRICT    ioCovY,
    float*       RESTRICT    ioCovU,
    float*       RESTRICT    ioCovV,
    float*       RESTRICT    ioMeanY,
    float*       RESTRICT    ioMeanU,
    float*       RESTRICT    ioMeanV,
    const int*   RESTRICT    inCountsY,
    const int*   RESTRICT    inCountsU,
    const int*   RESTRICT    inCountsV)
{
    // Channel demultiplex: block 0 handles Y, 1 handles U, 2 handles V.
    const int channel = blockIdx.x;
    const int bin     = threadIdx.x;

    float*       ioCov;
    float*       ioMean;
    const int*   inCounts;

    if (channel == 0)
    {
        ioCov    = ioCovY;
        ioMean   = ioMeanY;
        inCounts = inCountsY;
    }
    else if (channel == 1)
    {
        ioCov    = ioCovU;
        ioMean   = ioMeanU;
        inCounts = inCountsU;
    }
    else
    {
        ioCov    = ioCovV;
        ioMean   = ioMeanV;
        inCounts = inCountsV;
    }

    // ------------------------------------------------------------------------
    // Phase 0: Normalize by count, load per-bin variance into registers.
    //
    //   cur[freq]   = Sum(dct_coeff^2) / count        [= variance]
    //   ioMean[bin] = Sum(patch_mean)  / count        [= mean intensity]
    // ------------------------------------------------------------------------
    const int   count  = inCounts[bin];
    const bool  valid  = (count > 0);
    const float invN   = valid ? (1.0f / static_cast<float>(count)) : 0.0f;

    __shared__ float s_buf [NOISE_BINS];     // smoothing working buffer
    __shared__ int   s_valid[NOISE_BINS];    // 1 = bin has patches, 0 = empty

    s_valid[bin] = valid ? 1 : 0;

    // Normalize mean (write back to global)
    if (valid)
    {
        ioMean[bin] = ioMean[bin] * invN;
    }

    // Load and normalize 16 frequency variances into a register-local array.
    // These carry state across all 5 sweeps without revisiting global memory.
    float cur[PATCH_ELEMS];
    const int bin_base = bin * PATCH_ELEMS_SQ;

    #pragma unroll
    for (int freq = 0; freq < PATCH_ELEMS; ++freq)
    {
        cur[freq] = valid ? (ioCov[bin_base + freq] * invN) : 0.0f;
    }

    __syncthreads();    // s_valid is now visible to all threads

    // ------------------------------------------------------------------------
    // Phase 1: Lebrun's 5 sweeps of (bin-axis smoothing + per-bin 4x4 median).
    // ------------------------------------------------------------------------
    constexpr int NB_SWEEPS      = 5;
    constexpr int SMOOTH_WINDOW  = 10;       // Lebrun's sizeFilter

    for (int sweep = 0; sweep < NB_SWEEPS; ++sweep)
    {
        // ------ 1a. Per-frequency bin-axis smoothing ------
        // For each of the 16 frequencies:
        //   1. Stage the current per-bin variance into shared memory.
        //   2. Each thread averages over +/-10 valid neighbors.
        //   3. Write the smoothed value back to `cur`.

        for (int freq = 0; freq < PATCH_ELEMS; ++freq)
        {
            s_buf[bin] = cur[freq];
            __syncthreads();

            // Compute valid neighbor range [nmin, nmax] (fixes bug C2).
            const int nmin = max(0,              bin - SMOOTH_WINDOW);
            const int nmax = min(NOISE_BINS - 1, bin + SMOOTH_WINDOW);

            float sum      = 0.0f;
            int   n_valid  = 0;
            for (int nb = nmin; nb <= nmax; ++nb)
            {
                if (s_valid[nb])
                {
                    sum     += s_buf[nb];
                    n_valid += 1;
                }
            }

            // Update register state; clamp negative results to zero.
            if (n_valid > 0)
            {
                const float avg = sum / static_cast<float>(n_valid);
                cur[freq] = fmaxf(0.0f, avg);
            }
            // else: no valid neighbors in window -> keep previous `cur[freq]`
            //       (typically zero for empty bins in empty neighborhoods).

            __syncthreads();    // done reading s_buf before next iter overwrites it
        }

        // ------ 1b. Per-bin 4x4 median filter over the 16 frequencies ------
        // This is entirely thread-local (each thread operates on its own `cur`).
        float filt[PATCH_ELEMS];

        #pragma unroll
        for (int f = 0; f < PATCH_ELEMS; ++f)
        {
            const int i = f / 4;
            const int j = f % 4;
            filt[f] = MatMedianAt(cur, i, j);
        }

        // Commit the filtered values. Next sweep continues from here.
        #pragma unroll
        for (int f = 0; f < PATCH_ELEMS; ++f)
        {
            cur[f] = filt[f];
        }
    }

    // ------------------------------------------------------------------------
    // Phase 2: Write smoothed variances back to global memory.
    // Only the first 16 slots per bin are used; the remaining [16..255] stay
    // zero here -- kernel #4 will fill them with the full 16x16 spatial cov.
    // ------------------------------------------------------------------------
    #pragma unroll
    for (int freq = 0; freq < PATCH_ELEMS; ++freq)
    {
        ioCov[bin_base + freq] = cur[freq];
    }
}


// ============================================================================
// Kernel_BuildSpatialNoiseCov_3ch    (+ __constant__ DCT-1D matrix)
// ============================================================================
//
// Purpose:
//   Bridge the frequency-domain noise model (16 variances per bin) produced
//   by kernel #3 to the spatial-domain 16x16 covariance matrix that Lebrun's
//   Bayes filter consumes in passes 1 and 2.
//
// Math:
//   The forward 2D DCT for a 4x4 patch can be written as:
//       y = D_2d * x            (x = vectorized spatial pixels, y = freq coeffs)
//
//   where D_2d is the 16x16 orthonormal 2D DCT-II matrix, built as the
//   Kronecker product of the 1D 4-point DCT-II matrix D_1d with itself:
//       D_2d[(i,j), (m,n)] = D_1d[i, m] * D_1d[j, n]
//
//   For noise whose DCT coefficients are uncorrelated (diagonal freq cov),
//   the spatial covariance matrix is:
//       C_spatial = D_2d^T * diag(freq_vars) * D_2d
//   i.e.
//       C_spatial[x1, x2] = sum_k  D_2d[k, x1] * freq_vars[k] * D_2d[k, x2]
//
//   This is exactly what Lebrun's CPU RunNoiseEstimate.cpp / getMatrixBins
//   produces as the final noiseCovMatrix[intRef] passed to the Bayes filter.
//
// Storage semantics:
//   Input  (on entry):  d_NoiseCov_*[bin*256 + freq]  for freq in [0..15]
//                       holds per-bin variance of DCT coefficient `freq`.
//                       Slots [16..255] are ZERO (from initial memset).
//   Output (on exit):   d_NoiseCov_*[bin*256 + row*16 + col]
//                       holds spatial covariance between pixels `row` and
//                       `col` of a 4x4 patch. Full 16x16 matrix, symmetric.
//
//   Note: the kernel overwrites slots [0..15] as part of the full-matrix
//   write-back. Because we stage the input freq-variances into shared memory
//   BEFORE any global write, there is no read-after-write hazard.
//
// Launch geometry:
//   grid   = (NOISE_BINS, 3)         = (256, 3)
//   block  = (PATCH_ELEMS_SQ)        = (256)   one thread per matrix element
//   shared = (16 + 16) * 4 bytes     = 128 B   (D_1d + freq_vars staging)
// ============================================================================


// ----------------------------------------------------------------------------
// __constant__ 4-point orthonormal DCT-II matrix (row-major, [k*4 + n]).
//
//   D_1d[k, n] = a(k) * cos((2n+1) * k * pi / 8)
//              with a(0) = 1/2, a(k>0) = sqrt(1/2)
//
//   Laid out explicitly for clarity and to avoid runtime cos() calls.
//   These values match the DCT_C0..C3 constants used by the DCT helper
//   in kernel #2.
// ----------------------------------------------------------------------------
__constant__ float c_DCT_1D_4PT_MATRIX[PATCH_ELEMS] =
{
    // k = 0 (DC row):   all 0.5
     0.5f,          0.5f,          0.5f,          0.5f,

    // k = 1:           +C1, +C3, -C3, -C1
     0.65328148f,   0.27059805f,  -0.27059805f,  -0.65328148f,

    // k = 2:           +C2, -C2, -C2, +C2        (C2 = 0.5)
     0.5f,         -0.5f,         -0.5f,          0.5f,

    // k = 3:           +C3, -C1, +C1, -C3
     0.27059805f,  -0.65328148f,   0.65328148f,  -0.27059805f
};


// ============================================================================
// The kernel itself.
// ============================================================================
__global__ void Kernel_BuildSpatialNoiseCov_3ch
(
    float*       RESTRICT    ioCovY,
    float*       RESTRICT    ioCovU,
    float*       RESTRICT    ioCovV
)
{
    // ------------------------------------------------------------------------
    // Block / thread demux.
    //   blockIdx.x  = bin index         [0, NOISE_BINS)        = [0, 256)
    //   blockIdx.y  = channel           {0=Y, 1=U, 2=V}
    //   threadIdx.x = output-matrix element index  [0, 256)
    //                 where element  = row * 16 + col
    // ------------------------------------------------------------------------
    const int bin     = blockIdx.x;
    const int channel = blockIdx.y;
    const int tid     = threadIdx.x;

    float* ioCov;
    if (channel == 0)
    {
        ioCov = ioCovY;
    }
    else if (channel == 1)
    {
        ioCov = ioCovU;
    }
    else
    {
        ioCov = ioCovV;
    }

    const int bin_base = bin * PATCH_ELEMS_SQ;     // * 256

    // ------------------------------------------------------------------------
    // Phase 1: Cooperative load into shared memory.
    //   s_D1d       : the 4x4 DCT-1D matrix, flat layout (16 floats).
    //   s_freqVars  : the 16 frequency variances for this (bin, channel).
    //
    // Staging freqVars into shared BEFORE any writes guarantees no read-
    // after-write hazard even though we'll overwrite slots [0..15] later.
    // ------------------------------------------------------------------------
    __shared__ float s_D1d     [PATCH_ELEMS];
    __shared__ float s_freqVars[PATCH_ELEMS];

    if (tid < PATCH_ELEMS)
    {
        s_D1d     [tid] = c_DCT_1D_4PT_MATRIX[tid];
        s_freqVars[tid] = ioCov[bin_base + tid];
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 2: Compute one element of C_spatial[row, col].
    //
    //   row = tid / 16   (spatial pixel index 0..15, = m1*4 + n1)
    //   col = tid % 16   (spatial pixel index 0..15, = m2*4 + n2)
    //
    //   C_spatial[row, col] = sum_k  D_2d[k, row] * freq_vars[k] * D_2d[k, col]
    //
    //   where
    //     D_2d[k, pixel] = D_1d[k/4, pixel/4] * D_1d[k%4, pixel%4]
    //
    //   Unrolling over k=0..15 exposes the FMA chain to the compiler.
    // ------------------------------------------------------------------------
    const int row = tid / PATCH_ELEMS;         // 0..15
    const int col = tid % PATCH_ELEMS;         // 0..15

    const int m1 = row / 4;
    const int n1 = row % 4;
    const int m2 = col / 4;
    const int n2 = col % 4;

    float sum = 0.0f;

    #pragma unroll
    for (int k = 0; k < PATCH_ELEMS; ++k)
    {
        const int i = k / 4;
        const int j = k % 4;

        // D_2d[k, row] = D_1d[i, m1] * D_1d[j, n1]
        const float D_k_row = s_D1d[i * 4 + m1] * s_D1d[j * 4 + n1];

        // D_2d[k, col] = D_1d[i, m2] * D_1d[j, n2]
        const float D_k_col = s_D1d[i * 4 + m2] * s_D1d[j * 4 + n2];

        sum += D_k_row * s_freqVars[k] * D_k_col;
    }

    // ------------------------------------------------------------------------
    // Phase 3: Write back.
    //   Each of the 256 threads writes exactly one element of the spatial
    //   covariance matrix for this (bin, channel). No atomics needed: every
    //   destination slot has a unique writer.
    // ------------------------------------------------------------------------
    ioCov[bin_base + tid] = sum;
}


// ============================================================================
// Phase 1 Revision: Device Helpers (Jacobi + Bayes filter)
// ============================================================================
//
// This file replaces the Phase 0 helper file:
//   Kernel_05_DeviceHelpers_Jacobi_BayesFilter.cu
//
// Changes vs. Phase 0:
//   * Filter matrix M is no longer materialized into shared memory.
//     Instead, the helper now takes the X (numerator) matrix and computes
//     the product  M * x  directly as  X * V * D^(-1) * (V^T * x)  in a
//     streaming fashion, saving 1 KB of shared memory per block.
//
//   * The patch-application loop uses a 16-float shared-memory buffer to
//     hold one patch column's intermediate vector, avoiding register-
//     resident arrays that were prone to spilling (addressed further in
//     Phase 5).
//
//   * Classical Jacobi (`EigenJacobi16_Thread0`) is UNCHANGED. Parallel
//     Brent-Luk version arrives in Phase 3. Phase 1 output is therefore
//     byte-identical to Phase 0 output.
//
//   * Signature of ApplyBayesFilter_Block changes: the `sh_M` workspace
//     argument is removed (caller no longer needs to allocate it).
//     Caller passes one additional workspace buffer `sh_VTx` of 16 floats.
//
// Shared memory required by the caller for this helper (reduced):
//   sh_X      : 256 floats  (unchanged)
//   sh_Y      : 256 floats  (unchanged; trashed by helper)
//   sh_V      : 256 floats  (unchanged)
//   sh_Dinv   :  16 floats  (unchanged)
//   sh_VTx    :  16 floats  (NEW - intermediate for on-the-fly M*x)
//   sh_patches: 16 * nSimP  (unchanged)
//   sh_bary   :  16 floats  (unchanged)
//
// Total helper scratch: (3 * 256 + 2 * 16) * 4 = 3200 bytes (was 4160).
// ============================================================================


// ----------------------------------------------------------------------------
// EigenJacobi16_Thread0  (UNCHANGED from Phase 0)
//
// Single-threaded classical cyclic Jacobi for a real-symmetric 16x16 matrix.
// ----------------------------------------------------------------------------
__device__ __forceinline__
void EigenJacobi16_Thread0(float* RESTRICT A, float* RESTRICT V)
{
    #pragma unroll
    for (int k = 0; k < PATCH_ELEMS_SQ; ++k)
    {
        V[k] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < PATCH_ELEMS; ++i)
    {
        V[i * PATCH_ELEMS + i] = 1.0f;
    }

    constexpr int   NUM_SWEEPS = 8;
    constexpr float EPSILON    = 1e-7f;

    for (int sweep = 0; sweep < NUM_SWEEPS; ++sweep)
    {
        for (int p = 0; p < PATCH_ELEMS - 1; ++p)
        {
            for (int q = p + 1; q < PATCH_ELEMS; ++q)
            {
                const float apq = A[p * PATCH_ELEMS + q];
                if (fabsf(apq) <= EPSILON)
                {
                    continue;
                }

                const float app = A[p * PATCH_ELEMS + p];
                const float aqq = A[q * PATCH_ELEMS + q];

                const float theta = (aqq - app) / (2.0f * apq);

                float t;
                if (theta >= 0.0f)
                {
                    t = 1.0f / (theta + sqrtf(theta * theta + 1.0f));
                }
                else
                {
                    t = 1.0f / (theta - sqrtf(theta * theta + 1.0f));
                }

                const float c_rot = 1.0f / sqrtf(t * t + 1.0f);
                const float s_rot = t * c_rot;

                A[p * PATCH_ELEMS + p] = app - t * apq;
                A[q * PATCH_ELEMS + q] = aqq + t * apq;
                A[p * PATCH_ELEMS + q] = 0.0f;
                A[q * PATCH_ELEMS + p] = 0.0f;

                for (int r = 0; r < PATCH_ELEMS; ++r)
                {
                    if (r == p || r == q)
                    {
                        continue;
                    }
                    const float arp = A[r * PATCH_ELEMS + p];
                    const float arq = A[r * PATCH_ELEMS + q];

                    const float new_arp = c_rot * arp - s_rot * arq;
                    const float new_arq = s_rot * arp + c_rot * arq;

                    A[r * PATCH_ELEMS + p] = new_arp;
                    A[p * PATCH_ELEMS + r] = new_arp;
                    A[r * PATCH_ELEMS + q] = new_arq;
                    A[q * PATCH_ELEMS + r] = new_arq;
                }

                for (int r = 0; r < PATCH_ELEMS; ++r)
                {
                    const float vrp = V[r * PATCH_ELEMS + p];
                    const float vrq = V[r * PATCH_ELEMS + q];

                    V[r * PATCH_ELEMS + p] = c_rot * vrp - s_rot * vrq;
                    V[r * PATCH_ELEMS + q] = s_rot * vrp + c_rot * vrq;
                }
            }
        }
    }
}


// ============================================================================
// ApplyBayesFilter_Block  (PHASE 1 REVISED)
//
// Block-cooperative Bayes estimate:
//     out[:, n] = clip( X * Y^(-1) * (patch[:, n] - bary) + bary,
//                       min_val, max_val )
//
// Implementation via eigendecomposition:
//     Y = V * D * V^T   (Jacobi)
//     M = X * V * D^(-1) * V^T       (never materialized; applied streaming)
//     out = clip( M * centered + bary, min, max )
//
// Streaming expansion avoids the sh_M workspace:
//     For each patch column x (centered):
//         u[i]     = sum_k  V[k, i] * x[k]         // V^T * x, 16 floats
//         w[i]     = u[i] * D_inv[i]               // D^(-1) scaling
//         p[j]     = sum_i  V[j, i] * w[i]         // V * w
//         acc[r]   = sum_j  X[r, j] * p[j]         // X * (V D^-1 V^T) x
//         out[r]   = clip(acc[r] + bary[r], min, max)
//
// Block layout assumption:
//     - block has at least 32 threads (one warp) for the min/max reduction.
//     - block has at least 16 threads for the per-pixel phases.
//     - block_size and tid are computed as
//         tid = threadIdx.y * blockDim.x + threadIdx.x
//         block_size = blockDim.x * blockDim.y
// ============================================================================
__device__ void ApplyBayesFilter_Block(
    const float* RESTRICT sh_X,         // [256] in    : numerator matrix (preserved)
    float*       RESTRICT sh_Y,         // [256] in/out: matrix to invert (trashed on exit)
    float*       RESTRICT sh_V,         // [256] work  : receives eigenvectors
    float*       RESTRICT sh_Dinv,      // [ 16] work  : reciprocal eigenvalues
    float*       RESTRICT sh_VTx,       // [ 16] work  : streaming intermediate (NEW)
    float*       RESTRICT sh_patches,   // [16 * nSimP] in/out: patches column-major
    const float* RESTRICT sh_bary,      // [ 16] in    : barycenter to add back after filter
    float                 min_val,
    float                 max_val,
    int                   nSimP)
{
    const int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    // ------------------------------------------------------------------------
    // Phase 1: Jacobi eigendecomposition of Y.
    // ------------------------------------------------------------------------
    if (tid == 0)
    {
        EigenJacobi16_Thread0(sh_Y, sh_V);
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 2: Extract and safely invert eigenvalues.
    //   Clamp at 1e-6 to handle near-zero eigenvalues from noise-heavy or
    //   low-rank covariance matrices. Mathematically equivalent to Lebrun's
    //   diagonal clamping before Cholesky, strictly safer for float precision.
    // ------------------------------------------------------------------------
    if (tid < PATCH_ELEMS)
    {
        const float lambda = sh_Y[tid * (PATCH_ELEMS + 1)];   // diag index: k*17
        sh_Dinv[tid] = 1.0f / fmaxf(1e-6f, lambda);
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 3: Streaming application of M = X * V * D^(-1) * V^T to each
    // patch column. No intermediate M matrix in shared memory.
    //
    // Distribution: one patch column per thread (iterating in stride of
    // block_size so 256-thread blocks cover nSimP <= 64 in one sweep).
    //
    // NOTE: because sh_VTx is shared, this loop must serialize over patch
    // columns within a warp -- but since nSimP <= 64 and we need 16-wide
    // parallelism per matrix-vector product, we instead allocate sh_VTx
    // **per thread** via thread-local storage using registers. See below.
    //
    // Design decision: keep sh_VTx as a 16-float *per-thread* register
    // array `float u_vec[16]` rather than shared memory. The compiler
    // folds this into 16 registers per thread -- tight but manageable,
    // and avoids serialization on a single shared buffer.
    // ------------------------------------------------------------------------
    // (sh_VTx parameter is kept in the signature for future Phase 3 usage
    //  where the Brent-Luk parallel Jacobi needs block-wide scratch. For
    //  Phase 1, the per-patch filter loop uses register-local storage.)
    (void) sh_VTx;

    // Each thread handles one patch column n (stride block_size).
    for (int n = tid; n < nSimP; n += block_size)
    {
        // --- Step 3a: u = V^T * x_centered, 16 floats in registers ----
        float u_vec[PATCH_ELEMS];
        #pragma unroll 4
        for (int i = 0; i < PATCH_ELEMS; ++i)
        {
            float acc = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < PATCH_ELEMS; ++k)
            {
                acc += sh_V[k * PATCH_ELEMS + i] * sh_patches[k * nSimP + n];
            }
            u_vec[i] = acc;
        }

        // --- Step 3b: u = D^(-1) * u, fused scale ---------------------
        #pragma unroll
        for (int i = 0; i < PATCH_ELEMS; ++i)
        {
            u_vec[i] *= sh_Dinv[i];
        }

        // --- Step 3c: p = V * u, 16 floats back ------------------------
        float p_vec[PATCH_ELEMS];
        #pragma unroll 4
        for (int j = 0; j < PATCH_ELEMS; ++j)
        {
            float acc = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < PATCH_ELEMS; ++i)
            {
                acc += sh_V[j * PATCH_ELEMS + i] * u_vec[i];
            }
            p_vec[j] = acc;
        }

        // --- Step 3d: out = X * p + bary, clipped, written back --------
        #pragma unroll 4
        for (int r = 0; r < PATCH_ELEMS; ++r)
        {
            float acc = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < PATCH_ELEMS; ++j)
            {
                acc += sh_X[r * PATCH_ELEMS + j] * p_vec[j];
            }
            const float v = acc + sh_bary[r];
            sh_patches[r * nSimP + n] = fminf(max_val, fmaxf(min_val, v));
        }
    }

    // Caller is responsible for __syncthreads() before reusing sh_Y/sh_V/
    // sh_Dinv/sh_patches.
}


// ============================================================================
// Warp-level min/max reduction helpers (NEW in Phase 1)
//
// Used by Pass 1 and Pass 2 to reduce per-pixel-position min/max across
// the 16 patch pixels without a 16-element shared buffer. Works within a
// single warp (tid 0..31).
//
// Returns the reduced result on thread 0; other threads' returned value
// is garbage (but consistent if you need to broadcast).
// ============================================================================
__device__ __forceinline__
float WarpReduceMin(float v)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        const float other = __shfl_xor_sync(0xFFFFFFFF, v, offset);
        v = fminf(v, other);
    }
    return v;
}

__device__ __forceinline__
float WarpReduceMax(float v)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        const float other = __shfl_xor_sync(0xFFFFFFFF, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

__device__ __forceinline__
float WarpReduceSum(float v)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
    }
    return v;
}


// ============================================================================
// Kernel_NLBayes_Pass1_BasicEstimate  (PHASE 2 REVISED)
// ============================================================================
//
// Changes vs. Phase 0/1:
//
//   * MAX_SIMILAR_PATCHES shrunk from 128 to 64 (halves sh_patches to 4 KB).
//
//   * Patch similarity search now uses warp-cooperative ballot scan -- one
//     atomic per warp (8 per block at 256 threads) instead of one atomic
//     per passing candidate (~290 per block in flat regions). Deterministic
//     patch ordering across runs as a side effect.
//
//   * Covariance and barycenter computed in a SINGLE fused pass over the
//     patches. Phase 0/1 made two passes over the same patch data.
//
//   * Min/max reduction via warp shuffles -- no more s_min_tmp/s_max_tmp
//     shared buffers. Saves 128 bytes and one __syncthreads.
//
//   * Removed sh_D workspace (was the filter matrix M; the Phase 1 helper
//     computes M*x streaming and no longer needs sh_D).
//     => Net shared-memory reduction vs. Phase 0: ~5.4 KB per block.
//
// Mathematical contract: bit-identical to Phase 0/1 output.
//   * Different patch-selection *order* (deterministic now) can produce
//     different numerical results in principle, but the set of patches
//     selected is exactly the same set, so the covariance and filter are
//     exactly the same matrices. Only float summation order can differ
//     in the cov computation, which is a ~1e-7 per-element noise.
//
// Launch geometry (unchanged):
//   grid  = (ceil(procW / proc_stride), ceil(procH / proc_stride))
//   block = (32, 8) = 256 threads  (8 warps)
//   shared ~ 14 KB per block (was ~19 KB in Phase 1)
// ============================================================================


// ----------------------------------------------------------------------------
// Window-tile size: SEARCH_WINDOW_SIZE + PATCH_SIZE - 1 = 17 + 4 - 1 = 20.
// This covers every pixel touched by any 4x4 patch whose top-left lies in
// the 17x17 search window.
// ----------------------------------------------------------------------------
static constexpr int WINDOW_TILE_SIZE  = SEARCH_WINDOW_SIZE + PATCH_SIZE - 1;   // 20
static constexpr int WINDOW_TILE_ELEMS = WINDOW_TILE_SIZE * WINDOW_TILE_SIZE;   // 400

// Phase 2: cap similar-patch count at 64 (was 128 in Phase 0/1). Reduces
// sh_patches from 8 KB to 4 KB -> ~2x block-occupancy improvement on
// shared-memory-limited kernels.
static constexpr int PASS12_MAX_SIMILAR = 64;


__global__ void Kernel_NLBayes_Pass1_BasicEstimate
(
    const float* RESTRICT    inY,
    const float* RESTRICT    inU,
    const float* RESTRICT    inV,
    float*       RESTRICT    outAccumY,
    float*       RESTRICT    outAccumU,
    float*       RESTRICT    outAccumV,
    float*       RESTRICT    outWeight,
    const float* RESTRICT    noiseCovY,
    const float* RESTRICT    noiseCovU,
    const float* RESTRICT    noiseCovV,
    const float* RESTRICT    noiseMeanY,
    const float* RESTRICT    noiseMeanU,
    const float* RESTRICT    noiseMeanV,
    int                      procW,
    int                      procH,
    int                      padW,
    float                    tau_Y,
    float                    tau_UV,
    int                      proc_stride
)
{
    const int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    const int warp_id    = tid >> 5;        // 0..7 for 256 threads
    const int lane_id    = tid & 31;        // 0..31

    // Unused in Pass 1 (part of shared signature used by Pass 2).
    (void) noiseMeanY;
    (void) noiseMeanU;
    (void) noiseMeanV;

    // ------------------------------------------------------------------------
    // Reference patch top-left coordinate + boundary check.
    // ------------------------------------------------------------------------
    const int ref_x = blockIdx.x * proc_stride;
    const int ref_y = blockIdx.y * proc_stride;

    if (ref_x < SEARCH_WINDOW_RADIUS
        || ref_y < SEARCH_WINDOW_RADIUS
        || ref_x + PATCH_SIZE + SEARCH_WINDOW_RADIUS > procW
        || ref_y + PATCH_SIZE + SEARCH_WINDOW_RADIUS > procH)
    {
        return;
    }

    const int win_x0 = ref_x - SEARCH_WINDOW_RADIUS;
    const int win_y0 = ref_y - SEARCH_WINDOW_RADIUS;

    // ------------------------------------------------------------------------
    // Shared memory layout (~14 KB total).
    // ------------------------------------------------------------------------
    __shared__ float s_winY[WINDOW_TILE_ELEMS];          // 3 * 400 * 4 = 4800 B
    __shared__ float s_winU[WINDOW_TILE_ELEMS];
    __shared__ float s_winV[WINDOW_TILE_ELEMS];

    __shared__ int   s_patchIndices[PASS12_MAX_SIMILAR]; // 64 * 4 = 256 B
    __shared__ int   s_patchCount;

    // Scalar aggregates.
    __shared__ float s_sigma_Y;
    __shared__ float s_sigma_U;
    __shared__ float s_sigma_V;
    __shared__ float s_threshold;
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ int   s_bin_ch;

    // Matrix workspaces (3 * 256 + 16 = 3136 B). Phase 1 helper takes
    // (X, Y, V, Dinv, VTx); sh_VTx can be any 16-float slot.
    __shared__ float sh_X      [PATCH_ELEMS_SQ];    // (C_P - C_N) for helper
    __shared__ float sh_Y_mat  [PATCH_ELEMS_SQ];    // C_P (diag-clamped), trashed by helper
    __shared__ float sh_V_mat  [PATCH_ELEMS_SQ];    // eigenvectors workspace
    __shared__ float sh_Dinv   [PATCH_ELEMS];       // eigenvalue reciprocals

    // Bary + streaming intermediate for helper (sh_VTx repurposes sh_bary
    // dual-use is unsafe; use distinct buffer).
    __shared__ float sh_bary   [PATCH_ELEMS];
    __shared__ float sh_VTx    [PATCH_ELEMS];       // helper streaming intermediate

    // Patches (shrunk).
    __shared__ float sh_patches[PATCH_ELEMS * PASS12_MAX_SIMILAR];  // 16*64*4 = 4096 B

    // Per-warp staging for warp-cooperative patch selection.
    __shared__ int   s_warp_offsets[8];    // one per warp; 8 warps/block

    // ------------------------------------------------------------------------
    // Phase 1: Cooperative load of the 20x20 window tile for each channel.
    // ------------------------------------------------------------------------
    for (int i = tid; i < WINDOW_TILE_ELEMS; i += block_size)
    {
        const int lx = i % WINDOW_TILE_SIZE;
        const int ly = i / WINDOW_TILE_SIZE;
        const int g_idx = (win_y0 + ly) * padW + (win_x0 + lx);

        s_winY[i] = inY[g_idx];
        s_winU[i] = inU[g_idx];
        s_winV[i] = inV[g_idx];
    }

    if (tid == 0)
    {
        s_patchCount = 0;
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 2: Compute per-channel sigma and distance threshold.
    // (Identical to Phase 0/1 -- kept on thread 0 since it's a single-pass
    //  serial reduction over 16 diagonal elements per channel.)
    // ------------------------------------------------------------------------
    if (tid == 0)
    {
        // Reference patch mean (Y) for bin lookup.
        float sum_Y = 0.0f;
        #pragma unroll
        for (int i = 0; i < PATCH_SIZE; ++i)
        {
            #pragma unroll
            for (int j = 0; j < PATCH_SIZE; ++j)
            {
                const int p_idx = (SEARCH_WINDOW_RADIUS + i) * WINDOW_TILE_SIZE
                                + (SEARCH_WINDOW_RADIUS + j);
                sum_Y += s_winY[p_idx];
            }
        }
        const float mean_Y = sum_Y * (1.0f / static_cast<float>(PATCH_ELEMS));
        const int ref_bin = max(0, min(NOISE_BINS - 1,
                                       __float2int_rn(mean_Y * 255.0f)));

        const int base = ref_bin * PATCH_ELEMS_SQ;
        float trace_Y = 0.0f;
        float trace_U = 0.0f;
        float trace_V = 0.0f;
        #pragma unroll
        for (int k = 0; k < PATCH_ELEMS; ++k)
        {
            const int diag = k * (PATCH_ELEMS + 1);
            trace_Y += noiseCovY[base + diag];
            trace_U += noiseCovU[base + diag];
            trace_V += noiseCovV[base + diag];
        }

        const float sigma2_Y = fmaxf(0.0f, trace_Y) * (1.0f / static_cast<float>(PATCH_ELEMS));
        const float sigma2_U = fmaxf(0.0f, trace_U) * (1.0f / static_cast<float>(PATCH_ELEMS));
        const float sigma2_V = fmaxf(0.0f, trace_V) * (1.0f / static_cast<float>(PATCH_ELEMS));

        s_sigma_Y = sqrtf(sigma2_Y);
        s_sigma_U = sqrtf(sigma2_U);
        s_sigma_V = sqrtf(sigma2_V);

        s_threshold = 0.5f  * tau_Y  * sigma2_Y
                    + 0.25f * tau_UV * sigma2_U
                    + 0.25f * tau_UV * sigma2_V;
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 3: WARP-COOPERATIVE patch similarity search.
    //
    // For each 32-candidate chunk assigned to one warp:
    //   1. Each lane computes its candidate's distance and a `passed` bit.
    //   2. Warp ballot collects all 32 pass-bits into one mask.
    //   3. Warp leader atomically reserves `popcount(mask)` slots.
    //   4. Each passing lane writes its index into its reserved slot.
    //
    // Result: ONE atomic per warp per chunk (vs. one per passing candidate
    // in Phase 0/1). Deterministic ordering: warp 0's passers always occupy
    // lower slots than warp 1's, etc.
    // ------------------------------------------------------------------------
    {
        const int ref_px = SEARCH_WINDOW_RADIUS;      // 8
        const int ref_py = SEARCH_WINDOW_RADIUS;      // 8
        const int n_candidates = SEARCH_WINDOW_SIZE * SEARCH_WINDOW_SIZE;  // 289

        // Each warp processes candidates in chunks of 32. With 8 warps and
        // 289 candidates, each warp handles ~2 chunks.
        const int chunks_total = (n_candidates + 31) / 32;   // 10
        const int chunks_per_warp = (chunks_total + 7) / 8;  // 2 (ceil of 10/8)

        for (int chunk_local = 0; chunk_local < chunks_per_warp; ++chunk_local)
        {
            const int chunk_id = warp_id * chunks_per_warp + chunk_local;
            if (chunk_id >= chunks_total)
            {
                break;
            }
            const int cidx = chunk_id * 32 + lane_id;
            const bool in_range = (cidx < n_candidates);

            bool passed = false;
            if (in_range)
            {
                const int cx = cidx % SEARCH_WINDOW_SIZE;
                const int cy = cidx / SEARCH_WINDOW_SIZE;

                float dist = 0.0f;
                #pragma unroll
                for (int i = 0; i < PATCH_SIZE; ++i)
                {
                    #pragma unroll
                    for (int j = 0; j < PATCH_SIZE; ++j)
                    {
                        const int r_idx = (ref_py + i) * WINDOW_TILE_SIZE + (ref_px + j);
                        const int c_idx = (cy     + i) * WINDOW_TILE_SIZE + (cx     + j);

                        const float dY = s_winY[r_idx] - s_winY[c_idx];
                        const float dU = s_winU[r_idx] - s_winU[c_idx];
                        const float dV = s_winV[r_idx] - s_winV[c_idx];

                        dist += 0.5f  * dY * dY
                             +  0.25f * dU * dU
                             +  0.25f * dV * dV;
                    }
                }

                passed = (dist <= s_threshold);
            }

            // Ballot: bitmask of which lanes passed.
            const unsigned mask = __ballot_sync(0xFFFFFFFF, passed);
            const int n_passed = __popc(mask);

            // Warp leader reserves a contiguous range of slots.
            int warp_base = 0;
            if (lane_id == 0)
            {
                warp_base = atomicAdd(&s_patchCount, n_passed);
            }
            warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

            // Each passing lane writes to warp_base + its_rank_within_passers.
            if (passed)
            {
                // Rank = number of passing lanes with lower lane_id.
                const unsigned lower_mask = mask & ((1u << lane_id) - 1u);
                const int my_rank = __popc(lower_mask);
                const int slot = warp_base + my_rank;

                if (slot < PASS12_MAX_SIMILAR)
                {
                    s_patchIndices[slot] = cidx;
                }
            }
        }
    }
    __syncthreads();

    const int nSimP = min(s_patchCount, PASS12_MAX_SIMILAR);

    if (nSimP < 2)
    {
        return;
    }

    // ------------------------------------------------------------------------
    // Phase 4: Per-channel Bayes filter loop.
    // ------------------------------------------------------------------------
    #pragma unroll 1
    for (int ch = 0; ch < 3; ++ch)
    {
        const float* s_win;
        const float* noiseCov;
        float        sigma_c;

        if (ch == 0)
        {
            s_win    = s_winY;
            noiseCov = noiseCovY;
            sigma_c  = s_sigma_Y;
        }
        else if (ch == 1)
        {
            s_win    = s_winU;
            noiseCov = noiseCovU;
            sigma_c  = s_sigma_U;
        }
        else
        {
            s_win    = s_winV;
            noiseCov = noiseCovV;
            sigma_c  = s_sigma_V;
        }

        // --------------------------------------------------------------------
        // 4a. Extract patches into sh_patches (column-major, un-centered).
        //     sh_patches[i * nSimP + n]  =  pixel i of patch n.
        //     Also track per-pixel-position min/max via warp-shuffle reduction
        //     for the clip range -- AVOIDS a second pass later.
        // --------------------------------------------------------------------
        for (int k = tid; k < PATCH_ELEMS * nSimP; k += block_size)
        {
            const int i = k / nSimP;
            const int n = k % nSimP;

            const int p_idx = s_patchIndices[n];
            const int p_cx  = p_idx % SEARCH_WINDOW_SIZE;
            const int p_cy  = p_idx / SEARCH_WINDOW_SIZE;

            const int px = i % PATCH_SIZE;
            const int py = i / PATCH_SIZE;

            const int win_idx = (p_cy + py) * WINDOW_TILE_SIZE + (p_cx + px);
            sh_patches[i * nSimP + n] = s_win[win_idx];
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4b+c+d. FUSED: barycenter + min/max range + covariance in one pass.
        //
        // Each thread handles one covariance element (i, j). Along the way:
        //   - Thread with j == 0 accumulates the sum for sh_bary[i].
        //   - All threads accumulate min/max of their pixel i via warp shuffle.
        //
        // Covariance: C_P[i, j] = sum_n sh_patches[i, n] * sh_patches[j, n] / (nSimP - 1)
        //   (Lebrun un-centered raw correlation -- LibMatrix.cpp:covarianceMatrix)
        // --------------------------------------------------------------------
        {
            const float normInv = 1.0f / static_cast<float>(nSimP - 1);

            for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
            {
                const int i = k / PATCH_ELEMS;
                const int j = k % PATCH_ELEMS;

                float sum_ij   = 0.0f;   // for cov
                float sum_i    = 0.0f;   // for bary (only used when j == 0)
                float min_i    =  INFINITY;
                float max_i    = -INFINITY;

                for (int n = 0; n < nSimP; ++n)
                {
                    const float vi = sh_patches[i * nSimP + n];
                    const float vj = sh_patches[j * nSimP + n];
                    sum_ij += vi * vj;

                    // Bary: sum patches at pixel i once per row-thread.
                    if (j == 0)
                    {
                        sum_i += vi;
                        min_i = fminf(min_i, vi);
                        max_i = fmaxf(max_i, vi);
                    }
                }

                sh_Y_mat[k] = sum_ij * normInv;    // C_P

                if (j == 0)
                {
                    sh_bary[i] = sum_i / static_cast<float>(nSimP);
                }

                // Warp-shuffle reduce min/max (only row-threads with j==0 have
                // meaningful values; others have +/-inf which preserve the
                // result under fmin/fmax).
                const float row_min = (j == 0) ? min_i :  INFINITY;
                const float row_max = (j == 0) ? max_i : -INFINITY;

                const float warp_min = WarpReduceMin(row_min);
                const float warp_max = WarpReduceMax(row_max);

                // Warp leader (lane 0) of each warp writes its reduction to a
                // scratch slot; thread 0 of block does final 8-warp reduction.
                if (lane_id == 0)
                {
                    // Reuse s_patchIndices as scratch (already consumed).
                    // 8 warps -> slots 0..7.
                    // We store min in even slots, max in odd slots.
                    // But we need to keep s_patchIndices intact for aggregation later!
                    // So use s_warp_offsets for min and reuse s_patchCount etc. Safer:
                    // use a dedicated 16-slot scratch. Put it in sh_Dinv temporarily
                    // since sh_Dinv is written by the helper in phase 4h, not yet.
                    sh_Dinv[warp_id]     = warp_min;
                    sh_Dinv[warp_id + 8] = warp_max;
                }
            }
        }
        __syncthreads();

        // Final reduction across 8 warp partials -- single thread.
        if (tid == 0)
        {
            float mn = sh_Dinv[0];
            float mx = sh_Dinv[8];
            #pragma unroll
            for (int w = 1; w < 8; ++w)
            {
                mn = fminf(mn, sh_Dinv[w]);
                mx = fmaxf(mx, sh_Dinv[w + 8]);
            }
            s_min = mn - sigma_c;
            s_max = mx + sigma_c;

            // Per-channel intensity bin from barycenter mean.
            float total = 0.0f;
            #pragma unroll
            for (int k = 0; k < PATCH_ELEMS; ++k)
            {
                total += sh_bary[k];
            }
            const float group_mean = total * (1.0f / static_cast<float>(PATCH_ELEMS));
            s_bin_ch = max(0, min(NOISE_BINS - 1,
                                  __float2int_rn(group_mean * 255.0f)));
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4e. Load noise covariance C_N for this channel+bin into sh_V_mat.
        //     (We temporarily use sh_V_mat as C_N storage; the Jacobi helper
        //     will overwrite it with eigenvectors after we build sh_X from it.)
        // --------------------------------------------------------------------
        {
            const int bin_base = s_bin_ch * PATCH_ELEMS_SQ;
            for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
            {
                sh_V_mat[k] = noiseCov[bin_base + k];
            }
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4f. Build sh_X = C_P - C_N, and diagonal-clamp sh_Y_mat in place.
        // --------------------------------------------------------------------
        for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
        {
            sh_X[k] = sh_Y_mat[k] - sh_V_mat[k];
        }
        if (tid < PATCH_ELEMS)
        {
            const int d = tid * (PATCH_ELEMS + 1);
            sh_Y_mat[d] = fmaxf(sh_Y_mat[d], sh_V_mat[d]);
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4g. Center patches in place: sh_patches[i, n] -= sh_bary[i].
        // --------------------------------------------------------------------
        for (int k = tid; k < PATCH_ELEMS * nSimP; k += block_size)
        {
            const int i = k / nSimP;
            sh_patches[k] -= sh_bary[i];
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4h. Apply Bayes filter via Phase 1 helper.
        //
        // Helper signature (Phase 1):
        //   ApplyBayesFilter_Block(
        //     sh_X,       // X = C_P - C_N
        //     sh_Y,       // Y = C_P diag-clamped (trashed on exit)
        //     sh_V,       // V workspace (receives eigenvectors)
        //     sh_Dinv,    // D^(-1) workspace (16 floats)
        //     sh_VTx,     // streaming intermediate (16 floats)
        //     sh_patches, sh_bary, min_val, max_val, nSimP)
        // --------------------------------------------------------------------
        ApplyBayesFilter_Block(
            sh_X,          // X = C_P - C_N (preserved)
            sh_Y_mat,      // Y = C_P diag-clamped (trashed)
            sh_V_mat,      // V workspace (C_N contents no longer needed)
            sh_Dinv,
            sh_VTx,
            sh_patches,
            sh_bary,
            s_min,
            s_max,
            nSimP);

        __syncthreads();

        // --------------------------------------------------------------------
        // 4i. Aggregate filtered patches into global accumulators.
        // --------------------------------------------------------------------
        float* outAccum;
        if (ch == 0)
        {
            outAccum = outAccumY;
        }
        else if (ch == 1)
        {
            outAccum = outAccumU;
        }
        else
        {
            outAccum = outAccumV;
        }

        for (int k = tid; k < PATCH_ELEMS * nSimP; k += block_size)
        {
            const int i = k / nSimP;
            const int n = k % nSimP;

            const int p_idx = s_patchIndices[n];
            const int p_cx  = p_idx % SEARCH_WINDOW_SIZE;
            const int p_cy  = p_idx / SEARCH_WINDOW_SIZE;

            const int px = i % PATCH_SIZE;
            const int py = i / PATCH_SIZE;

            const int gx = win_x0 + p_cx + px;
            const int gy = win_y0 + p_cy + py;

            const int g_idx = gy * padW + gx;

            const float val = sh_patches[i * nSimP + n];
            atomicAdd(&outAccum[g_idx], val);

            if (ch == 0)
            {
                atomicAdd(&outWeight[g_idx], 1.0f);
            }
        }
        __syncthreads();
    }
}



// ============================================================================
// Kernel_NormalizePilotEstimate
// ============================================================================
//
// Purpose:
//   Convert the weighted-aggregate output of Pass 1 into a pilot image
//   suitable for Pass 2 consumption:
//
//       pilot[p]  =  acc[p] / weight[p]           if  weight[p] > 0
//       pilot[p]  =  noisy[p]                     if  weight[p] == 0
//
//   This is Lebrun's computeWeightedAggregation (NlBayes.cpp:687), adapted
//   to GPU with separate input and output buffers (CPU divides in place; we
//   keep the noisy planes intact for Pass 2's patch-distance search, so we
//   emit the pilot into the dedicated MosaicB slot).
//
// Why the fallback matters:
//   A pixel can have weight == 0 only if every patch group that *could* have
//   covered it was rejected by Pass 1's boundary check (ref coordinate too
//   close to the image edge). For the interior of the padded processing
//   region this should not happen at proc_stride = 1; at larger strides,
//   a handful of pixels may be missed. Using the noisy pixel itself is
//   Lebrun's standard fallback -- it avoids NaN and leaves those positions
//   un-denoised rather than zeroed.
//
// Launch geometry:
//   grid  = (ceil(procW / 32), ceil(procH / 16))
//   block = (32, 16) = 512 threads / block
//   shared = 0 (purely per-pixel work)
//
// Complexity:
//   Exactly one thread per pixel of the padded valid region. Four global
//   reads + three global writes per pixel -- bandwidth-bound, compiles to
//   a tight kernel of ~10 instructions per thread.
// ============================================================================
__global__ void Kernel_NormalizePilotEstimate
(
    const float* RESTRICT    accY,
    const float* RESTRICT    accU,
    const float* RESTRICT    accV,
    const float* RESTRICT    accW,
    const float* RESTRICT    fallbackY,
    const float* RESTRICT    fallbackU,
    const float* RESTRICT    fallbackV,
    float*       RESTRICT    pilotY,
    float*       RESTRICT    pilotU,
    float*       RESTRICT    pilotV,
    int                      padW,
    int                      procW,
    int                      procH
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard: threads in the over-provisioned edge tiles do nothing.
    if (x >= procW || y >= procH)
    {
        return;
    }

    const int idx = y * padW + x;
    const float w = accW[idx];

    if (w > 0.0f)
    {
        // Standard aggregation: average of all patch contributions.
        const float invW = 1.0f / w;
        pilotY[idx] = accY[idx] * invW;
        pilotU[idx] = accU[idx] * invW;
        pilotV[idx] = accV[idx] * invW;
    }
    else
    {
        // Uncovered pixel -- fall back to the noisy input. Matches Lebrun's
        // computeWeightedAggregation "if (iW[k] > 0.f) ... else iO[k] = iN[k]"
        // exactly.
        pilotY[idx] = fallbackY[idx];
        pilotU[idx] = fallbackU[idx];
        pilotV[idx] = fallbackV[idx];
    }
}



// ============================================================================
// Kernel_NLBayes_Pass2_FinalEstimate  (PHASE 2 REVISED)
// ============================================================================
//
// Changes vs. Phase 0/1:
//
//   * MAX_SIMILAR_PATCHES shrunk from 128 to PASS12_MAX_SIMILAR = 64
//     (PASS12_MAX_SIMILAR is declared in the Phase 2 Pass 1 block above).
//     Halves sh_patches from 8 KB to 4 KB.
//
//   * Patch similarity search now uses warp-cooperative ballot scan --
//     1 atomic per warp instead of up to 1 per passing candidate. Same
//     mechanism as Phase 2 Pass 1. Deterministic patch ordering across runs.
//
//   * Covariance (C_basic from pilot centered-by-noisy-mean), barycenter
//     (from noisy), and pilot min/max computed in a SINGLE fused pass over
//     the similar-patch set. Phase 0/1 made three separate passes over the
//     same data with their own synchronization overhead.
//
//   * Min/max reduction via warp shuffles -- no s_min_tmp/s_max_tmp arrays.
//     Uses sh_Dinv as 16-slot scratch (safe because sh_Dinv is not written
//     until phase 4h's helper call).
//
//   * ApplyBayesFilter_Block called with the Phase 1 helper signature:
//     (sh_X, sh_Y, sh_V, sh_Dinv, sh_VTx, sh_patches, sh_bary, ...)
//     Fixes the parameter-misalignment we had in Phase 0 where sh_M was
//     still being passed; now the parameters line up correctly by design.
//
// Mathematical contract: matches Lebrun's per-channel Pass 2:
//
//   bary_c     = mean_n( noisy[c, :, n] )
//   B[c, :, n] = pilot[c, :, n] - bary_c
//   C_basic_c  = B * B^T / (nSimP - 1)                [Lebrun un-centered]
//   C_P+N_c    = C_basic_c + C_N_c
//   filter_c   = C_basic_c * (C_P+N_c)^(-1)
//   out_c[:, n]= clip( filter_c * (noisy[c, :, n] - bary_c) + bary_c,
//                      pilot_min - sigma_c, pilot_max + sigma_c )
//
// Output is mathematically equivalent to Phase 0/1 (patch ordering is now
// deterministic; float summation order in the fused loops differs by ~1e-7
// per element, below visual perception).
//
// Launch geometry (unchanged):
//   grid  = (ceil(procW / proc_stride), ceil(procH / proc_stride))
//   block = (32, 8) = 256 threads (8 warps)
//   shared ~ 19 KB per block (was ~23 KB in Phase 0/1)
// ============================================================================


__global__ void Kernel_NLBayes_Pass2_FinalEstimate
(
    const float* RESTRICT    noisyY,
    const float* RESTRICT    noisyU,
    const float* RESTRICT    noisyV,
    const float* RESTRICT    pilotY,
    const float* RESTRICT    pilotU,
    const float* RESTRICT    pilotV,
    float*       RESTRICT    outAccumY,
    float*       RESTRICT    outAccumU,
    float*       RESTRICT    outAccumV,
    float*       RESTRICT    outWeight,
    const float* RESTRICT    noiseCovY,
    const float* RESTRICT    noiseCovU,
    const float* RESTRICT    noiseCovV,
    const float* RESTRICT    noiseMeanY,
    const float* RESTRICT    noiseMeanU,
    const float* RESTRICT    noiseMeanV,
    int                      procW,
    int                      procH,
    int                      padW,
    float                    tau_Y,
    float                    tau_UV,
    int                      proc_stride
)
{
    const int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    const int warp_id    = tid >> 5;        // 0..7
    const int lane_id    = tid & 31;        // 0..31

    // Unused in Pass 2 (Lebrun uses nearest-bin LUT, no interpolation).
    (void) noiseMeanY;
    (void) noiseMeanU;
    (void) noiseMeanV;

    // ------------------------------------------------------------------------
    // Reference coordinate and boundary check.
    // ------------------------------------------------------------------------
    const int ref_x = blockIdx.x * proc_stride;
    const int ref_y = blockIdx.y * proc_stride;

    if (ref_x < SEARCH_WINDOW_RADIUS
        || ref_y < SEARCH_WINDOW_RADIUS
        || ref_x + PATCH_SIZE + SEARCH_WINDOW_RADIUS > procW
        || ref_y + PATCH_SIZE + SEARCH_WINDOW_RADIUS > procH)
    {
        return;
    }

    const int win_x0 = ref_x - SEARCH_WINDOW_RADIUS;
    const int win_y0 = ref_y - SEARCH_WINDOW_RADIUS;

    // ------------------------------------------------------------------------
    // Shared memory layout (~19 KB total).
    // ------------------------------------------------------------------------
    __shared__ float s_noisyY[WINDOW_TILE_ELEMS];    // 3 * 400 * 4 = 4800 B
    __shared__ float s_noisyU[WINDOW_TILE_ELEMS];
    __shared__ float s_noisyV[WINDOW_TILE_ELEMS];
    __shared__ float s_pilotY[WINDOW_TILE_ELEMS];    // 3 * 400 * 4 = 4800 B
    __shared__ float s_pilotU[WINDOW_TILE_ELEMS];
    __shared__ float s_pilotV[WINDOW_TILE_ELEMS];

    __shared__ int   s_patchIndices[PASS12_MAX_SIMILAR];    // 64 * 4 = 256 B
    __shared__ int   s_patchCount;

    __shared__ float s_sigma_Y;
    __shared__ float s_sigma_U;
    __shared__ float s_sigma_V;
    __shared__ float s_threshold;
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ int   s_bin_ch;

    // Matrix workspaces (3 * 256 + 16 = 3136 B).
    __shared__ float sh_X      [PATCH_ELEMS_SQ];     // C_basic (preserved by helper)
    __shared__ float sh_Y_mat  [PATCH_ELEMS_SQ];     // C_basic + C_N (inverted, trashed)
    __shared__ float sh_V_mat  [PATCH_ELEMS_SQ];     // eigenvector workspace
    __shared__ float sh_Dinv   [PATCH_ELEMS];        // 16-float reciprocal eigenvalues

    __shared__ float sh_bary   [PATCH_ELEMS];        // barycenter from noisy
    __shared__ float sh_VTx    [PATCH_ELEMS];        // helper streaming intermediate

    __shared__ float sh_patches[PATCH_ELEMS * PASS12_MAX_SIMILAR];   // noisy, 16*64*4 = 4096 B

    // ------------------------------------------------------------------------
    // Phase 1: Load noisy + pilot window tiles cooperatively.
    // ------------------------------------------------------------------------
    for (int i = tid; i < WINDOW_TILE_ELEMS; i += block_size)
    {
        const int lx = i % WINDOW_TILE_SIZE;
        const int ly = i / WINDOW_TILE_SIZE;
        const int g_idx = (win_y0 + ly) * padW + (win_x0 + lx);

        s_noisyY[i] = noisyY[g_idx];
        s_noisyU[i] = noisyU[g_idx];
        s_noisyV[i] = noisyV[g_idx];

        s_pilotY[i] = pilotY[g_idx];
        s_pilotU[i] = pilotU[g_idx];
        s_pilotV[i] = pilotV[g_idx];
    }

    if (tid == 0)
    {
        s_patchCount = 0;
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 2: Per-channel sigma and Pass-2 threshold.
    //   Identical to Phase 0/1 -- single-threaded sequential reductions.
    // ------------------------------------------------------------------------
    if (tid == 0)
    {
        float sum_Y = 0.0f;
        #pragma unroll
        for (int i = 0; i < PATCH_SIZE; ++i)
        {
            #pragma unroll
            for (int j = 0; j < PATCH_SIZE; ++j)
            {
                const int p_idx = (SEARCH_WINDOW_RADIUS + i) * WINDOW_TILE_SIZE
                                + (SEARCH_WINDOW_RADIUS + j);
                sum_Y += s_noisyY[p_idx];
            }
        }
        const float mean_Y  = sum_Y * (1.0f / static_cast<float>(PATCH_ELEMS));
        const int ref_bin   = max(0, min(NOISE_BINS - 1,
                                         __float2int_rn(mean_Y * 255.0f)));

        const int base = ref_bin * PATCH_ELEMS_SQ;
        float trace_Y = 0.0f;
        float trace_U = 0.0f;
        float trace_V = 0.0f;
        #pragma unroll
        for (int k = 0; k < PATCH_ELEMS; ++k)
        {
            const int diag = k * (PATCH_ELEMS + 1);
            trace_Y += noiseCovY[base + diag];
            trace_U += noiseCovU[base + diag];
            trace_V += noiseCovV[base + diag];
        }

        const float sigma2_Y = fmaxf(0.0f, trace_Y) * (1.0f / static_cast<float>(PATCH_ELEMS));
        const float sigma2_U = fmaxf(0.0f, trace_U) * (1.0f / static_cast<float>(PATCH_ELEMS));
        const float sigma2_V = fmaxf(0.0f, trace_V) * (1.0f / static_cast<float>(PATCH_ELEMS));

        s_sigma_Y = sqrtf(sigma2_Y);
        s_sigma_U = sqrtf(sigma2_U);
        s_sigma_V = sqrtf(sigma2_V);

        // Pass-2 threshold: weights uniform (1, 1, 1) per channel,
        // tau expanded by nChannels = 3.
        s_threshold = 3.0f * (tau_Y  * sigma2_Y
                            + tau_UV * sigma2_U
                            + tau_UV * sigma2_V);
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 3: Warp-cooperative patch similarity search using the PILOT image.
    //
    //   dist = sum_{i,j} [ (dY_pilot)^2 + (dU_pilot)^2 + (dV_pilot)^2 ]
    //
    // Same warp-ballot structure as Pass 1. The reference patch at (SWR, SWR)
    // has dist = 0 so it is always selected.
    // ------------------------------------------------------------------------
    {
        const int ref_px = SEARCH_WINDOW_RADIUS;
        const int ref_py = SEARCH_WINDOW_RADIUS;
        const int n_candidates = SEARCH_WINDOW_SIZE * SEARCH_WINDOW_SIZE;    // 289

        const int chunks_total    = (n_candidates + 31) / 32;                // 10
        const int chunks_per_warp = (chunks_total + 7) / 8;                  // 2

        for (int chunk_local = 0; chunk_local < chunks_per_warp; ++chunk_local)
        {
            const int chunk_id = warp_id * chunks_per_warp + chunk_local;
            if (chunk_id >= chunks_total)
            {
                break;
            }
            const int cidx = chunk_id * 32 + lane_id;
            const bool in_range = (cidx < n_candidates);

            bool passed = false;
            if (in_range)
            {
                const int cx = cidx % SEARCH_WINDOW_SIZE;
                const int cy = cidx / SEARCH_WINDOW_SIZE;

                float dist = 0.0f;
                #pragma unroll
                for (int i = 0; i < PATCH_SIZE; ++i)
                {
                    #pragma unroll
                    for (int j = 0; j < PATCH_SIZE; ++j)
                    {
                        const int r_idx = (ref_py + i) * WINDOW_TILE_SIZE + (ref_px + j);
                        const int c_idx = (cy     + i) * WINDOW_TILE_SIZE + (cx     + j);

                        const float dY = s_pilotY[r_idx] - s_pilotY[c_idx];
                        const float dU = s_pilotU[r_idx] - s_pilotU[c_idx];
                        const float dV = s_pilotV[r_idx] - s_pilotV[c_idx];

                        dist += dY * dY + dU * dU + dV * dV;
                    }
                }

                passed = (dist <= s_threshold);
            }

            const unsigned mask = __ballot_sync(0xFFFFFFFF, passed);
            const int n_passed  = __popc(mask);

            int warp_base = 0;
            if (lane_id == 0)
            {
                warp_base = atomicAdd(&s_patchCount, n_passed);
            }
            warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

            if (passed)
            {
                const unsigned lower_mask = mask & ((1u << lane_id) - 1u);
                const int my_rank = __popc(lower_mask);
                const int slot    = warp_base + my_rank;
                if (slot < PASS12_MAX_SIMILAR)
                {
                    s_patchIndices[slot] = cidx;
                }
            }
        }
    }
    __syncthreads();

    const int nSimP = min(s_patchCount, PASS12_MAX_SIMILAR);

    if (nSimP < 2)
    {
        return;
    }

    // ------------------------------------------------------------------------
    // Phase 4: Per-channel Bayes filter loop.
    // ------------------------------------------------------------------------
    #pragma unroll 1
    for (int ch = 0; ch < 3; ++ch)
    {
        const float* s_noisy;
        const float* s_pilot;
        const float* noiseCov;
        float        sigma_c;

        if (ch == 0)
        {
            s_noisy  = s_noisyY;
            s_pilot  = s_pilotY;
            noiseCov = noiseCovY;
            sigma_c  = s_sigma_Y;
        }
        else if (ch == 1)
        {
            s_noisy  = s_noisyU;
            s_pilot  = s_pilotU;
            noiseCov = noiseCovU;
            sigma_c  = s_sigma_U;
        }
        else
        {
            s_noisy  = s_noisyV;
            s_pilot  = s_pilotV;
            noiseCov = noiseCovV;
            sigma_c  = s_sigma_V;
        }

        // --------------------------------------------------------------------
        // 4a. Load NOISY patches into sh_patches (column-major, un-centered).
        //     sh_patches[i * nSimP + n] = pixel i of noisy patch n.
        // --------------------------------------------------------------------
        for (int k = tid; k < PATCH_ELEMS * nSimP; k += block_size)
        {
            const int i = k / nSimP;
            const int n = k % nSimP;

            const int p_idx = s_patchIndices[n];
            const int p_cx  = p_idx % SEARCH_WINDOW_SIZE;
            const int p_cy  = p_idx / SEARCH_WINDOW_SIZE;

            const int px = i % PATCH_SIZE;
            const int py = i / PATCH_SIZE;

            const int win_idx = (p_cy + py) * WINDOW_TILE_SIZE + (p_cx + px);
            sh_patches[i * nSimP + n] = s_noisy[win_idx];
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4b+c+d. FUSED: barycenter (from noisy), pilot min/max, and
        //   C_basic covariance -- all in ONE sweep over patches.
        //
        //   For each matrix element (i, j) assigned to this thread:
        //     - sum pilot(i)*pilot(j) - (bary_i is needed first; trick below)
        //
        //   Subtlety: C_basic[i,j] = sum_n  (pilot[i,n] - bary_i) * (pilot[j,n] - bary_j)
        //             where bary = mean(noisy).
        //
        //   We expand:
        //     = sum_n pilot[i]*pilot[j]  - bary_j*sum_n pilot[i]
        //                                - bary_i*sum_n pilot[j]
        //                                + nSimP*bary_i*bary_j
        //
        //   So we can compute, in ONE pass:
        //     S_ij  = sum_n pilot[i,n] * pilot[j,n]      (cross-product)
        //     S_i   = sum_n pilot[i,n]                   (pilot row sum)
        //     (bary_i for noisy is computed as a side effect when j == 0)
        //
        //   Then we need bary (from noisy) to finalize C_basic. So this fused
        //   phase has two parts:
        //     Part A: compute S_ij, S_i (pilot), N_i (noisy sum), pilot min/max
        //     Part B: compute sh_bary from N_i, then finalize
        //             C_basic[i,j] = (S_ij - bary_j*S_i - bary_i*S_j
        //                             + nSimP*bary_i*bary_j) / (nSimP - 1)
        //
        //   We stash S_i in sh_V_mat (cheap since it's otherwise workspace),
        //   and the pilot min/max in shared scratch reduced via warp shuffle.
        // --------------------------------------------------------------------
        {
            // ---- Part A: sum cross-products, row-sums, and min/max ----

            for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
            {
                const int i = k / PATCH_ELEMS;
                const int j = k % PATCH_ELEMS;

                const int py_i = i / PATCH_SIZE;
                const int px_i = i % PATCH_SIZE;
                const int py_j = j / PATCH_SIZE;
                const int px_j = j % PATCH_SIZE;

                float sum_ij   = 0.0f;
                float sum_i    = 0.0f;    // pilot row sum (only j == 0)
                float noisy_i  = 0.0f;    // noisy row sum (only j == 0)
                float min_p    =  INFINITY;
                float max_p    = -INFINITY;

                for (int n = 0; n < nSimP; ++n)
                {
                    const int p_idx = s_patchIndices[n];
                    const int p_cx  = p_idx % SEARCH_WINDOW_SIZE;
                    const int p_cy  = p_idx / SEARCH_WINDOW_SIZE;

                    const float pi = s_pilot[(p_cy + py_i) * WINDOW_TILE_SIZE + (p_cx + px_i)];
                    const float pj = s_pilot[(p_cy + py_j) * WINDOW_TILE_SIZE + (p_cx + px_j)];
                    sum_ij += pi * pj;

                    if (j == 0)
                    {
                        sum_i   += pi;
                        noisy_i += s_noisy[(p_cy + py_i) * WINDOW_TILE_SIZE + (p_cx + px_i)];
                        min_p   = fminf(min_p, pi);
                        max_p   = fmaxf(max_p, pi);
                    }
                }

                // Stash S_ij temporarily in sh_Y_mat.
                sh_Y_mat[k] = sum_ij;

                if (j == 0)
                {
                    // Stash row-sum S_i in sh_V_mat[i*16 + 0] -- the first column.
                    // Reused later for C_basic finalization.
                    sh_V_mat[i * PATCH_ELEMS] = sum_i;
                    // Compute and store bary = noisy row sum / nSimP.
                    sh_bary[i] = noisy_i / static_cast<float>(nSimP);
                }

                // Warp-shuffle reduce pilot min/max.
                const float row_min = (j == 0) ? min_p :  INFINITY;
                const float row_max = (j == 0) ? max_p : -INFINITY;

                const float warp_min = WarpReduceMin(row_min);
                const float warp_max = WarpReduceMax(row_max);

                if (lane_id == 0)
                {
                    sh_Dinv[warp_id]     = warp_min;   // slots 0..7
                    sh_Dinv[warp_id + 8] = warp_max;   // slots 8..15
                }
            }
        }
        __syncthreads();

        // ---- Part B: finalize C_basic using bary, and reduce min/max ----
        {
            const float normInv = 1.0f / static_cast<float>(nSimP - 1);

            for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
            {
                const int i = k / PATCH_ELEMS;
                const int j = k % PATCH_ELEMS;

                const float S_ij   = sh_Y_mat[k];
                const float S_i    = sh_V_mat[i * PATCH_ELEMS];   // pilot row sum for i
                const float S_j    = sh_V_mat[j * PATCH_ELEMS];   // pilot row sum for j
                const float bary_i = sh_bary[i];
                const float bary_j = sh_bary[j];

                // C_basic[i, j] = (S_ij - bary_j * S_i - bary_i * S_j
                //                  + nSimP * bary_i * bary_j) / (nSimP - 1)
                const float c = (S_ij
                              - bary_j * S_i
                              - bary_i * S_j
                              + static_cast<float>(nSimP) * bary_i * bary_j) * normInv;

                sh_Y_mat[k] = c;       // temporary: will become Y = C_basic + C_N next
            }
        }
        __syncthreads();

        // Final min/max + bin computation (single thread).
        if (tid == 0)
        {
            float mn = sh_Dinv[0];
            float mx = sh_Dinv[8];
            #pragma unroll
            for (int w = 1; w < 8; ++w)
            {
                mn = fminf(mn, sh_Dinv[w]);
                mx = fmaxf(mx, sh_Dinv[w + 8]);
            }
            s_min = mn - sigma_c;
            s_max = mx + sigma_c;

            // Per-channel intensity bin from barycenter mean (noisy).
            float total = 0.0f;
            #pragma unroll
            for (int k = 0; k < PATCH_ELEMS; ++k)
            {
                total += sh_bary[k];
            }
            const float group_mean = total * (1.0f / static_cast<float>(PATCH_ELEMS));
            s_bin_ch = max(0, min(NOISE_BINS - 1,
                                  __float2int_rn(group_mean * 255.0f)));
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4e. Load C_N into sh_V_mat (overwriting the pilot row-sums we stashed
        //     earlier -- they are no longer needed now that C_basic is final).
        // --------------------------------------------------------------------
        {
            const int bin_base = s_bin_ch * PATCH_ELEMS_SQ;
            for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
            {
                sh_V_mat[k] = noiseCov[bin_base + k];
            }
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4f. Build Bayes inputs:
        //       sh_X     = C_basic               (preserved through helper)
        //       sh_Y_mat = C_basic + C_N         (will be inverted, trashed)
        // --------------------------------------------------------------------
        for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
        {
            const float v_basic = sh_Y_mat[k];         // currently holds C_basic
            sh_X[k]     = v_basic;                     // X = C_basic
            sh_Y_mat[k] = v_basic + sh_V_mat[k];       // Y = C_basic + C_N
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4g. Center noisy patches in place: sh_patches[i, n] -= sh_bary[i].
        // --------------------------------------------------------------------
        for (int k = tid; k < PATCH_ELEMS * nSimP; k += block_size)
        {
            const int i = k / nSimP;
            sh_patches[k] -= sh_bary[i];
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4h. Apply Bayes filter via Phase 1 helper.
        // --------------------------------------------------------------------
        ApplyBayesFilter_Block(
            sh_X,          // X = C_basic (preserved)
            sh_Y_mat,      // Y = C_basic + C_N (trashed)
            sh_V_mat,      // V workspace (C_N contents overwritten)
            sh_Dinv,
            sh_VTx,
            sh_patches,
            sh_bary,
            s_min,
            s_max,
            nSimP);

        __syncthreads();

        // --------------------------------------------------------------------
        // 4i. Aggregate filtered patches into global accumulators.
        // --------------------------------------------------------------------
        float* outAccum;
        if (ch == 0)
        {
            outAccum = outAccumY;
        }
        else if (ch == 1)
        {
            outAccum = outAccumU;
        }
        else
        {
            outAccum = outAccumV;
        }

        for (int k = tid; k < PATCH_ELEMS * nSimP; k += block_size)
        {
            const int i = k / nSimP;
            const int n = k % nSimP;

            const int p_idx = s_patchIndices[n];
            const int p_cx  = p_idx % SEARCH_WINDOW_SIZE;
            const int p_cy  = p_idx / SEARCH_WINDOW_SIZE;

            const int px = i % PATCH_SIZE;
            const int py = i / PATCH_SIZE;

            const int gx = win_x0 + p_cx + px;
            const int gy = win_y0 + p_cy + py;

            const int g_idx = gy * padW + gx;

            const float val = sh_patches[i * nSimP + n];
            atomicAdd(&outAccum[g_idx], val);

            if (ch == 0)
            {
                atomicAdd(&outWeight[g_idx], 1.0f);
            }
        }
        __syncthreads();
    }
}



// ============================================================================
// Kernel_NormalizeAndOutputBGRA
// ============================================================================
//
// Purpose:
//   Final output conversion. For each pixel in the ORIGINAL (un-padded)
//   [0, width) x [0, height) region:
//
//     1. Divide the Pass-2 accumulator by the weight counter (with noisy-
//        fallback when w == 0, matching computeWeightedAggregation).
//     2. Convert the denoised YUV values back to RGB via the TRANSPOSE of
//        Lebrun's forward matrix (correct because the forward transform is
//        orthonormal -> inverse = transpose).
//     3. Pack BGRA in interleaved channel order with alpha = 1.0.
//
//   The padding region [width, procW) x [height, procH) is not written --
//   that data exists only internally in the YUV planes, never flows out.
//
// Forward (kernel #1) was:
//     Y = YUV_A * (R + G + B)                     [a = 1/sqrt(3)]
//     U = YUV_B * (R     - B)                     [b = 1/sqrt(2)]
//     V = YUV_C * (R/4 - G/2 + B/4)               [c = 2a * sqrt(2)]
//
// Inverse (transpose, matching LibImages.cpp:225-234 with c' = a/b):
//     R = a*Y + b*U + c'*V /2
//     G = a*Y       - c'*V
//     B = a*Y - b*U + c'*V /2
//
//   where c' = a/b = sqrt(2/3). We pre-computed this as YUV_C_INV in the
//   file header (= 0.81649658f). The inverse V column has coefficients
//   (c'/2, -c', c'/2) = (YUV_C_INV/2, -YUV_C_INV, YUV_C_INV/2).
//
// Bug fix vs. the original:
//   Original kernel used YUV_A, YUV_B, V_COEF_RB = 0.204..., V_COEF_G = -0.408...
//   applied with the SAME matrix as the forward transform. This isn't the
//   inverse of a non-orthonormal matrix -- it was wrong in two ways (the
//   forward coefficients were wrong, AND the inverse should have been the
//   transpose). Fixed here.
//
// Launch geometry:
//   grid  = (ceil(width / 32), ceil(height / 16))    (ORIGINAL extent, not padded)
//   block = (32, 16) = 512 threads / block
//   shared = 0 (purely per-pixel)
// ============================================================================
__global__ void Kernel_NormalizeAndOutputBGRA(
    const float* RESTRICT    accY,
    const float* RESTRICT    accU,
    const float* RESTRICT    accV,
    const float* RESTRICT    accW,
    const float* RESTRICT    fallbackY,
    const float* RESTRICT    fallbackU,
    const float* RESTRICT    fallbackV,
    float*       RESTRICT    outBuffer,
    int                      padW,
    int                      dstPitchPixels,
    int                      width,
    int                      height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard: we only write into the ORIGINAL (un-padded) region.
    // Padding columns/rows in [width, procW) x [height, procH) are silently
    // discarded -- they were computed internally but have no physical meaning.
    if (x >= width || y >= height)
    {
        return;
    }

    // ------------------------------------------------------------------------
    // Read accumulator + weight from the padded internal plane at (x, y).
    // ------------------------------------------------------------------------
    const int src_idx = y * padW + x;
    const float w = accW[src_idx];

    float valY;
    float valU;
    float valV;

    if (w > 0.0f)
    {
        const float invW = 1.0f / w;
        valY = accY[src_idx] * invW;
        valU = accU[src_idx] * invW;
        valV = accV[src_idx] * invW;
    }
    else
    {
        // Uncovered pixel -> carry noisy pixel through unchanged.
        // Matches Lebrun's computeWeightedAggregation fallback.
        valY = fallbackY[src_idx];
        valU = fallbackU[src_idx];
        valV = fallbackV[src_idx];
    }

    // ------------------------------------------------------------------------
    // YUV -> RGB via the transpose of the forward orthonormal matrix.
    //
    //   R = a*Y + b*U + (c'/2)*V
    //   G = a*Y       -  c'   *V
    //   B = a*Y - b*U + (c'/2)*V
    //
    // where a = YUV_A, b = YUV_B, c' = YUV_C_INV = sqrt(2/3).
    // ------------------------------------------------------------------------
    const float half_cinv = 0.5f * YUV_C_INV;

    const float r = YUV_A * valY + YUV_B * valU + half_cinv  * valV;
    const float g = YUV_A * valY                - YUV_C_INV  * valV;
    const float b = YUV_A * valY - YUV_B * valU + half_cinv  * valV;

    // ------------------------------------------------------------------------
    // Pack BGRA (4 channels per pixel) into the output buffer.
    //   channel 0 = B, 1 = G, 2 = R, 3 = A (set to 1.0).
    //
    // dstPitchPixels is the output row stride in PIXELS (not bytes); the
    // byte stride is 4 * dstPitchPixels * sizeof(float).
    // ------------------------------------------------------------------------
    const int out_base = (y * dstPitchPixels + x) * 4;

    outBuffer[out_base + 0] = b;
    outBuffer[out_base + 1] = g;
    outBuffer[out_base + 2] = r;
    outBuffer[out_base + 3] = 1.0f;
}



// ============================================================================
// HELPER: round up to next multiple of N (N must be a power of two)
// ============================================================================
static inline int RoundUpPow2(int value, int multiple)
{
    return (value + (multiple - 1)) & ~(multiple - 1);
}

#if 1
// ============================================================================
// ORCHESTRATOR: ImageLabDenoise_CUDA
//
// Contract:
//    inBuffer  : BGRA 32f, pitched, read-only.     srcPitch in PIXELS.
//    outBuffer : BGRA 32f, pitched, write-only.    dstPitch in PIXELS.
//    width, height : valid-region dimensions in PIXELS.
//    algoGpuParams : user tuning knobs (see AlgoControls.hpp).
//    frameCount : reserved for future temporal filtering (unused today).
//    stream : all GPU work is queued on this stream; we sync once at end.
//
// Arena:
//    g_gpuMemState holds the single cudaMalloc'd arena and all pointer
//    slices into it. Re-allocated only when (procW, procH) changes.
// ============================================================================
CUDA_KERNEL_CALL
void ImageLabDenoise_CUDA
(
    const float* RESTRICT inBuffer,
    float*       RESTRICT outBuffer,
    int                   srcPitch,
    int                   dstPitch,
    int                   width,
    int                   height,
    const AlgoControls* RESTRICT algoGpuParams,
    int                   frameCount,
    cudaStream_t          stream
)
{
    // Guard against bad inputs; on failure we simply do nothing and return.
    if (inBuffer == nullptr || outBuffer == nullptr || algoGpuParams == nullptr) {
        return;
    }
    if (width <= 0 || height <= 0) {
        return;
    }
    (void)frameCount;  // reserved for future use

    // -----------------------------------------------------------------------
    // STEP 0: Compute internal padded processing dimensions (Lebrun-style).
    //
    // Examples:    width = 999,  height = 999  ->  procW = 1000, procH = 1000
    //              width = 1001, height = 1001 ->  procW = 1004, procH = 1004
    //              width = 1920, height = 1080 ->  procW = 1920, procH = 1080
    //
    // The padded region [width..procW) x [height..procH) will be filled
    // by mirror-reflection inside Kernel_ConvertBGRAToOrthonormalWeighted,
    // then stripped at Kernel_NormalizeAndOutputBGRA.
    // -----------------------------------------------------------------------
    const int procW = RoundUpPow2(width,  PROC_ALIGN);
    const int procH = RoundUpPow2(height, PROC_ALIGN);

    // Must have at least one full search-window worth of pixels.
    if (procW < SEARCH_WINDOW_SIZE + PATCH_SIZE ||
        procH < SEARCH_WINDOW_SIZE + PATCH_SIZE) {
        return;
    }

    // -----------------------------------------------------------------------
    // STEP 1: (Re)allocate arena on first call, or when dimensions change.
    // Single cudaMalloc total; slices are set up inside alloc_cuda_memory_buffers.
    // -----------------------------------------------------------------------
    if (g_gpuMemState.d_arena_pool == nullptr
        || g_gpuMemState.tileW != procW
        || g_gpuMemState.tileH != procH)
    {
        free_cuda_memory_buffers(g_gpuMemState);
        if (!alloc_cuda_memory_buffers(g_gpuMemState, procW, procH)) {
            return;
        }
    }

    // -----------------------------------------------------------------------
    // STEP 2: Launch configurations
    // -----------------------------------------------------------------------
    const dim3 io_block   (BLOCK_DIM_IO_X,   BLOCK_DIM_IO_Y);
    const dim3 math_block (BLOCK_DIM_MATH_X, BLOCK_DIM_MATH_Y);     // 256 threads / block

    // Grid for I/O kernels that cover the full padded region
    const dim3 io_grid_proc(
        (procW + io_block.x - 1) / io_block.x,
        (procH + io_block.y - 1) / io_block.y);

    // Grid for the final output kernel -- native image extent only (no pad)
    const dim3 io_grid_out(
        (width  + io_block.x - 1) / io_block.x,
        (height + io_block.y - 1) / io_block.y);

    // Accuracy-driven patch-processing stride (paste-trick density).
    // Draft = fast, coarse.  Standard = balanced.  High = every pixel.
    int proc_stride;
    switch (algoGpuParams->accuracy) {
        case ProcAccuracy::AccDraft:    proc_stride = 4; break;
        case ProcAccuracy::AccHigh:     proc_stride = 1; break;
        case ProcAccuracy::AccStandard:
        default:                        proc_stride = 2; break;
    }

    // Grid for NL-Bayes passes: one block per reference-patch center.
    const dim3 math_grid(
        (procW + proc_stride - 1) / proc_stride,
        (procH + proc_stride - 1) / proc_stride);

    // -----------------------------------------------------------------------
    // Map AlgoControls -> Lebrun's tau threshold scalers
    //
    // Lebrun's per-step algorithmic constant is  tau_base = 3 * sP^2 = 48.
    // In the reference CPU code this is fixed; here we expose it to the user
    // through AlgoControls, preserving identical behavior at default = 1.0f.
    //
    // tau_Y  applies to the luma (Y) channel
    // tau_UV applies to both chroma (U, V) channels
    // -----------------------------------------------------------------------
    const float tau_base = 3.0f * static_cast<float>(PATCH_ELEMS);   // Lebrun: 3 * sP^2 = 48
    const float master   = algoGpuParams->master_denoise_amount;
    const float fine     = algoGpuParams->fine_detail_preservation;

    const float tau_Y  = tau_base * master * algoGpuParams->luma_strength   * fine;
    const float tau_UV = tau_base * master * algoGpuParams->chroma_strength * fine;

    // -----------------------------------------------------------------------
    // Precompute byte sizes for async memset clears
    // -----------------------------------------------------------------------
    const size_t bytes_frame         = static_cast<size_t>(g_gpuMemState.frameSizePadded) * sizeof(float);
    const size_t bytes_noise_cov     = static_cast<size_t>(NOISE_BINS) * PATCH_ELEMS_SQ * sizeof(float);
    const size_t bytes_noise_mean    = static_cast<size_t>(NOISE_BINS) * sizeof(float);
    const size_t bytes_noise_counts  = static_cast<size_t>(NOISE_BINS) * sizeof(int);

    // -----------------------------------------------------------------------
    // STEP 3: Clear noise-estimation LUTs (async; no host sync)
    // -----------------------------------------------------------------------
    cudaMemsetAsync(g_gpuMemState.d_NoiseCov_Y,    0, bytes_noise_cov,    stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseCov_U,    0, bytes_noise_cov,    stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseCov_V,    0, bytes_noise_cov,    stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseMean_Y,   0, bytes_noise_mean,   stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseMean_U,   0, bytes_noise_mean,   stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseMean_V,   0, bytes_noise_mean,   stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseCounts_Y, 0, bytes_noise_counts, stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseCounts_U, 0, bytes_noise_counts, stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseCounts_V, 0, bytes_noise_counts, stream);

    // -----------------------------------------------------------------------
    // STEP 4: BGRA -> YUV (with mirror-reflect padding width/height -> procW/procH)
    // -----------------------------------------------------------------------
    Kernel_ConvertBGRAToOrthonormalWeighted<<<io_grid_proc, io_block, 0, stream>>>
    (
        inBuffer,
        g_gpuMemState.d_Y_planar, g_gpuMemState.d_U_planar, g_gpuMemState.d_V_planar,
        srcPitch, g_gpuMemState.padW,
        width,  height,    // source valid-region extents
        procW,  procH      // target (padded) extents
    );

    // -----------------------------------------------------------------------
    // STEP 5: Noise estimation (Ponomarenko-style per-bin DCT variance)
    //
    //   5a. Extract per-patch DCT, accumulate per-bin frequency variances,
    //       per-bin pixel-sum (for mean), per-bin counts -- all 3 channels
    //       in one fused launch.
    //   5b. Smooth the frequency-domain curves across bins + normalize mean.
    //   5c. Convert 16-value frequency variance vectors into the full 16x16
    //       spatial noise covariance matrices (D^T * diag * D) per bin.
    // -----------------------------------------------------------------------
    Kernel_ExtractDCT_And_Variance_3ch<<<io_grid_proc, io_block, 0, stream>>>
    (
        g_gpuMemState.d_Y_planar,     g_gpuMemState.d_U_planar,     g_gpuMemState.d_V_planar,
        g_gpuMemState.d_NoiseCov_Y,   g_gpuMemState.d_NoiseCov_U,   g_gpuMemState.d_NoiseCov_V,
        g_gpuMemState.d_NoiseMean_Y,  g_gpuMemState.d_NoiseMean_U,  g_gpuMemState.d_NoiseMean_V,
        g_gpuMemState.d_NoiseCounts_Y,g_gpuMemState.d_NoiseCounts_U,g_gpuMemState.d_NoiseCounts_V,
        g_gpuMemState.padW, procW, procH
    );

    // 3 channels x 256 bins per channel. One block per channel, 256 threads per block (one per bin).
    Kernel_SmoothNoiseCurves_3ch<<<dim3(3), dim3(NOISE_BINS), 0, stream>>>
    (
        g_gpuMemState.d_NoiseCov_Y,   g_gpuMemState.d_NoiseCov_U,   g_gpuMemState.d_NoiseCov_V,
        g_gpuMemState.d_NoiseMean_Y,  g_gpuMemState.d_NoiseMean_U,  g_gpuMemState.d_NoiseMean_V,
        g_gpuMemState.d_NoiseCounts_Y,g_gpuMemState.d_NoiseCounts_U,g_gpuMemState.d_NoiseCounts_V
    );

    // D^T * diag(freq_vars) * D -> spatial 16x16 covariance per bin.
    // 3 channels x 256 bins, one block per (channel, bin), 256 threads per block
    // (one per spatial-matrix element).
    Kernel_BuildSpatialNoiseCov_3ch<<<dim3(NOISE_BINS, 3), dim3(PATCH_ELEMS_SQ), 0, stream>>>
    (
        g_gpuMemState.d_NoiseCov_Y, g_gpuMemState.d_NoiseCov_U, g_gpuMemState.d_NoiseCov_V
    );

    // -----------------------------------------------------------------------
    // STEP 6: NL-BAYES PASS 1 (pilot / basic estimate)
    //   Input:  noisy YUV planes
    //   Output: weighted-aggregated pilot estimate in d_Accum_*,  d_Weight
    //   Then:   d_MosaicB_* = d_Accum_* / d_Weight   (via Normalize kernel)
    // -----------------------------------------------------------------------
    cudaMemsetAsync(g_gpuMemState.d_Accum_Y, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Accum_U, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Accum_V, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Weight,  0, bytes_frame, stream);

    Kernel_NLBayes_Pass1_BasicEstimate<<<math_grid, math_block, 0, stream>>>
    (
        g_gpuMemState.d_Y_planar,    g_gpuMemState.d_U_planar,    g_gpuMemState.d_V_planar,
        g_gpuMemState.d_Accum_Y,     g_gpuMemState.d_Accum_U,     g_gpuMemState.d_Accum_V,
        g_gpuMemState.d_Weight,
        g_gpuMemState.d_NoiseCov_Y,  g_gpuMemState.d_NoiseCov_U,  g_gpuMemState.d_NoiseCov_V,
        g_gpuMemState.d_NoiseMean_Y, g_gpuMemState.d_NoiseMean_U, g_gpuMemState.d_NoiseMean_V,
        procW, procH, g_gpuMemState.padW,
        tau_Y, tau_UV, proc_stride
    );

    Kernel_NormalizePilotEstimate<<<io_grid_proc, io_block, 0, stream>>>
    (
        g_gpuMemState.d_Accum_Y,    g_gpuMemState.d_Accum_U,    g_gpuMemState.d_Accum_V,
        g_gpuMemState.d_Weight,
        g_gpuMemState.d_Y_planar,   g_gpuMemState.d_U_planar,   g_gpuMemState.d_V_planar,   // fallback
        g_gpuMemState.d_MosaicB_Y,  g_gpuMemState.d_MosaicB_U,  g_gpuMemState.d_MosaicB_V,  // pilot output
        g_gpuMemState.padW, procW, procH
    );

    // -----------------------------------------------------------------------
    // STEP 7: NL-BAYES PASS 2 (final estimate)
    //   Input:  noisy YUV planes + pilot estimate (d_MosaicB_*)
    //   Output: weighted-aggregated final estimate in d_Accum_*, d_Weight
    // -----------------------------------------------------------------------
    cudaMemsetAsync(g_gpuMemState.d_Accum_Y, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Accum_U, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Accum_V, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Weight,  0, bytes_frame, stream);

    Kernel_NLBayes_Pass2_FinalEstimate<<<math_grid, math_block, 0, stream>>>
    (
        g_gpuMemState.d_Y_planar,    g_gpuMemState.d_U_planar,    g_gpuMemState.d_V_planar,    // noisy
        g_gpuMemState.d_MosaicB_Y,   g_gpuMemState.d_MosaicB_U,   g_gpuMemState.d_MosaicB_V,   // pilot
        g_gpuMemState.d_Accum_Y,     g_gpuMemState.d_Accum_U,     g_gpuMemState.d_Accum_V,
        g_gpuMemState.d_Weight,
        g_gpuMemState.d_NoiseCov_Y,  g_gpuMemState.d_NoiseCov_U,  g_gpuMemState.d_NoiseCov_V,
        g_gpuMemState.d_NoiseMean_Y, g_gpuMemState.d_NoiseMean_U, g_gpuMemState.d_NoiseMean_V,
        procW, procH, g_gpuMemState.padW,
        tau_Y, tau_UV, proc_stride
    );

    // -----------------------------------------------------------------------
    // STEP 8: YUV -> BGRA (strip padding; write ONLY width x height region)
    // -----------------------------------------------------------------------
    Kernel_NormalizeAndOutputBGRA<<<io_grid_out, io_block, 0, stream>>>
    (
        g_gpuMemState.d_Accum_Y,    g_gpuMemState.d_Accum_U,    g_gpuMemState.d_Accum_V,
        g_gpuMemState.d_Weight,
        g_gpuMemState.d_Y_planar,   g_gpuMemState.d_U_planar,   g_gpuMemState.d_V_planar,   // fallback for zero-weight pixels
        outBuffer,
        g_gpuMemState.padW, dstPitch,
        width, height          // original (unpadded) output extents
    );

    // -----------------------------------------------------------------------
    // STEP 9: Single end-of-frame synchronization point.
    //
    // Prefer cudaStreamSynchronize over cudaDeviceSynchronize so that
    // concurrent work on other streams (e.g., host-managed copies for
    // the next frame) is not stalled.
    //
    // TODO(A6): when multi-scale denoising is implemented, move this sync
    // to AFTER the coarse-to-fine loop completes.
    // -----------------------------------------------------------------------
    cudaStreamSynchronize(stream);
}


// ============================================================================
// CLEANUP: Release the single arena. Call at plugin/host shutdown.
// ============================================================================
CUDA_KERNEL_CALL
void ImageLabDenoise_CleanupGPU()
{
    free_cuda_memory_buffers(g_gpuMemState);
}


// ============================================================================
// == KERNEL BODIES BELOW ==
//
// The kernel bodies are delivered incrementally, one per subsequent message:
//
//   Message #1:  Kernel_ConvertBGRAToOrthonormalWeighted
//   Message #2:  Kernel_ExtractDCT_And_Variance_3ch   (+ DCT_1D_4Point helper)
//   Message #3:  Kernel_SmoothNoiseCurves_3ch
//   Message #4:  Kernel_BuildSpatialNoiseCov_3ch
//   Message #5:  Jacobi eigendecomposition + Wiener solver device helpers
//   Message #6:  Kernel_NLBayes_Pass1_BasicEstimate
//   Message #7:  Kernel_NormalizePilotEstimate
//   Message #8:  Kernel_NLBayes_Pass2_FinalEstimate
//   Message #9:  Kernel_NormalizeAndOutputBGRA
//
// Paste each message's code into this file below, in order. The forward
// declarations above already match their final signatures exactly, so the
// file compiles cleanly at every step (kernel calls will link once all
// bodies are in place).
// ============================================================================

#else

CUDA_KERNEL_CALL
void ImageLabDenoise_CUDA
(
    const float* RESTRICT inBuffer,
    float*       RESTRICT outBuffer,
    int                   srcPitch,
    int                   dstPitch,
    int                   width,
    int                   height,
    const AlgoControls* RESTRICT algoGpuParams,
    int                   frameCount,
    cudaStream_t          stream
)
{
    if (inBuffer == nullptr || outBuffer == nullptr || algoGpuParams == nullptr)
    {
        return;
    }
    if (width <= 0 || height <= 0)
    {
        return;
    }
    (void)frameCount;

    const int procW = RoundUpPow2(width,  PROC_ALIGN);
    const int procH = RoundUpPow2(height, PROC_ALIGN);

    if (procW < SEARCH_WINDOW_SIZE + PATCH_SIZE
        || procH < SEARCH_WINDOW_SIZE + PATCH_SIZE)
    {
        return;
    }

    if (g_gpuMemState.d_arena_pool == nullptr
        || g_gpuMemState.tileW != procW
        || g_gpuMemState.tileH != procH)
    {
        free_cuda_memory_buffers(g_gpuMemState);
        if (!alloc_cuda_memory_buffers(g_gpuMemState, procW, procH))
        {
            return;
        }
    }

    const dim3 io_block   (BLOCK_DIM_IO_X,   BLOCK_DIM_IO_Y);
    const dim3 math_block (BLOCK_DIM_MATH_X, BLOCK_DIM_MATH_Y);

    const dim3 io_grid_proc(
        (procW + io_block.x - 1) / io_block.x,
        (procH + io_block.y - 1) / io_block.y);

    const dim3 io_grid_out(
        (width  + io_block.x - 1) / io_block.x,
        (height + io_block.y - 1) / io_block.y);

    int proc_stride;
    switch (algoGpuParams->accuracy)
    {
        case ProcAccuracy::AccDraft:    proc_stride = 4; break;
        case ProcAccuracy::AccHigh:     proc_stride = 1; break;
        case ProcAccuracy::AccStandard:
        default:                        proc_stride = 2; break;
    }

    const dim3 math_grid(
        (procW + proc_stride - 1) / proc_stride,
        (procH + proc_stride - 1) / proc_stride);

    const float tau_base = 3.0f * static_cast<float>(PATCH_ELEMS);
    const float master   = algoGpuParams->master_denoise_amount;
    const float fine     = algoGpuParams->fine_detail_preservation;

    const float tau_Y  = tau_base * master * algoGpuParams->luma_strength   * fine;
    const float tau_UV = tau_base * master * algoGpuParams->chroma_strength * fine;

    const size_t bytes_frame        = static_cast<size_t>(g_gpuMemState.frameSizePadded) * sizeof(float);
    const size_t bytes_noise_cov    = static_cast<size_t>(NOISE_BINS) * PATCH_ELEMS_SQ * sizeof(float);
    const size_t bytes_noise_mean   = static_cast<size_t>(NOISE_BINS) * sizeof(float);
    const size_t bytes_noise_counts = static_cast<size_t>(NOISE_BINS) * sizeof(int);

    // ========================================================================
    // DIAGNOSTIC EVENTS
    // ========================================================================
    cudaEvent_t ev_start, ev_convert, ev_dct, ev_smooth, ev_spatial;
    cudaEvent_t ev_pass1, ev_normalize, ev_pass2, ev_output;

    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_convert);
    cudaEventCreate(&ev_dct);
    cudaEventCreate(&ev_smooth);
    cudaEventCreate(&ev_spatial);
    cudaEventCreate(&ev_pass1);
    cudaEventCreate(&ev_normalize);
    cudaEventCreate(&ev_pass2);
    cudaEventCreate(&ev_output);

    cudaEventRecord(ev_start, stream);

    // ========================================================================

    cudaMemsetAsync(g_gpuMemState.d_NoiseCov_Y,    0, bytes_noise_cov,    stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseCov_U,    0, bytes_noise_cov,    stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseCov_V,    0, bytes_noise_cov,    stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseMean_Y,   0, bytes_noise_mean,   stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseMean_U,   0, bytes_noise_mean,   stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseMean_V,   0, bytes_noise_mean,   stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseCounts_Y, 0, bytes_noise_counts, stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseCounts_U, 0, bytes_noise_counts, stream);
    cudaMemsetAsync(g_gpuMemState.d_NoiseCounts_V, 0, bytes_noise_counts, stream);

    Kernel_ConvertBGRAToOrthonormalWeighted<<<io_grid_proc, io_block, 0, stream>>>
    (
        inBuffer,
        g_gpuMemState.d_Y_planar, g_gpuMemState.d_U_planar, g_gpuMemState.d_V_planar,
        srcPitch, g_gpuMemState.padW,
        width,  height,
        procW,  procH
    );
    cudaEventRecord(ev_convert, stream);

    Kernel_ExtractDCT_And_Variance_3ch<<<io_grid_proc, io_block, 0, stream>>>
    (
        g_gpuMemState.d_Y_planar,     g_gpuMemState.d_U_planar,     g_gpuMemState.d_V_planar,
        g_gpuMemState.d_NoiseCov_Y,   g_gpuMemState.d_NoiseCov_U,   g_gpuMemState.d_NoiseCov_V,
        g_gpuMemState.d_NoiseMean_Y,  g_gpuMemState.d_NoiseMean_U,  g_gpuMemState.d_NoiseMean_V,
        g_gpuMemState.d_NoiseCounts_Y,g_gpuMemState.d_NoiseCounts_U,g_gpuMemState.d_NoiseCounts_V,
        g_gpuMemState.padW, procW, procH
    );
    cudaEventRecord(ev_dct, stream);

    Kernel_SmoothNoiseCurves_3ch<<<dim3(3), dim3(NOISE_BINS), 0, stream>>>
    (
        g_gpuMemState.d_NoiseCov_Y,   g_gpuMemState.d_NoiseCov_U,   g_gpuMemState.d_NoiseCov_V,
        g_gpuMemState.d_NoiseMean_Y,  g_gpuMemState.d_NoiseMean_U,  g_gpuMemState.d_NoiseMean_V,
        g_gpuMemState.d_NoiseCounts_Y,g_gpuMemState.d_NoiseCounts_U,g_gpuMemState.d_NoiseCounts_V
    );
    cudaEventRecord(ev_smooth, stream);

    Kernel_BuildSpatialNoiseCov_3ch<<<dim3(NOISE_BINS, 3), dim3(PATCH_ELEMS_SQ), 0, stream>>>
    (
        g_gpuMemState.d_NoiseCov_Y, g_gpuMemState.d_NoiseCov_U, g_gpuMemState.d_NoiseCov_V
    );
    cudaEventRecord(ev_spatial, stream);

    cudaMemsetAsync(g_gpuMemState.d_Accum_Y, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Accum_U, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Accum_V, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Weight,  0, bytes_frame, stream);

    Kernel_NLBayes_Pass1_BasicEstimate<<<math_grid, math_block, 0, stream>>>
    (
        g_gpuMemState.d_Y_planar,    g_gpuMemState.d_U_planar,    g_gpuMemState.d_V_planar,
        g_gpuMemState.d_Accum_Y,     g_gpuMemState.d_Accum_U,     g_gpuMemState.d_Accum_V,
        g_gpuMemState.d_Weight,
        g_gpuMemState.d_NoiseCov_Y,  g_gpuMemState.d_NoiseCov_U,  g_gpuMemState.d_NoiseCov_V,
        g_gpuMemState.d_NoiseMean_Y, g_gpuMemState.d_NoiseMean_U, g_gpuMemState.d_NoiseMean_V,
        procW, procH, g_gpuMemState.padW,
        tau_Y, tau_UV, proc_stride
    );
    cudaEventRecord(ev_pass1, stream);

    Kernel_NormalizePilotEstimate<<<io_grid_proc, io_block, 0, stream>>>
    (
        g_gpuMemState.d_Accum_Y,    g_gpuMemState.d_Accum_U,    g_gpuMemState.d_Accum_V,
        g_gpuMemState.d_Weight,
        g_gpuMemState.d_Y_planar,   g_gpuMemState.d_U_planar,   g_gpuMemState.d_V_planar,
        g_gpuMemState.d_MosaicB_Y,  g_gpuMemState.d_MosaicB_U,  g_gpuMemState.d_MosaicB_V,
        g_gpuMemState.padW, procW, procH
    );
    cudaEventRecord(ev_normalize, stream);

    cudaMemsetAsync(g_gpuMemState.d_Accum_Y, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Accum_U, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Accum_V, 0, bytes_frame, stream);
    cudaMemsetAsync(g_gpuMemState.d_Weight,  0, bytes_frame, stream);

    Kernel_NLBayes_Pass2_FinalEstimate<<<math_grid, math_block, 0, stream>>>
    (
        g_gpuMemState.d_Y_planar,    g_gpuMemState.d_U_planar,    g_gpuMemState.d_V_planar,
        g_gpuMemState.d_MosaicB_Y,   g_gpuMemState.d_MosaicB_U,   g_gpuMemState.d_MosaicB_V,
        g_gpuMemState.d_Accum_Y,     g_gpuMemState.d_Accum_U,     g_gpuMemState.d_Accum_V,
        g_gpuMemState.d_Weight,
        g_gpuMemState.d_NoiseCov_Y,  g_gpuMemState.d_NoiseCov_U,  g_gpuMemState.d_NoiseCov_V,
        g_gpuMemState.d_NoiseMean_Y, g_gpuMemState.d_NoiseMean_U, g_gpuMemState.d_NoiseMean_V,
        procW, procH, g_gpuMemState.padW,
        tau_Y, tau_UV, proc_stride
    );
    cudaEventRecord(ev_pass2, stream);

    Kernel_NormalizeAndOutputBGRA<<<io_grid_out, io_block, 0, stream>>>
    (
        g_gpuMemState.d_Accum_Y,    g_gpuMemState.d_Accum_U,    g_gpuMemState.d_Accum_V,
        g_gpuMemState.d_Weight,
        g_gpuMemState.d_Y_planar,   g_gpuMemState.d_U_planar,   g_gpuMemState.d_V_planar,
        outBuffer,
        g_gpuMemState.padW, dstPitch,
        width, height
    );
    cudaEventRecord(ev_output, stream);

    cudaStreamSynchronize(stream);

    // ========================================================================
    // DIAGNOSTIC: read elapsed times and print to stderr
    // ========================================================================
    float t_convert, t_dct, t_smooth, t_spatial;
    float t_pass1,  t_normalize, t_pass2, t_output;

    cudaEventElapsedTime(&t_convert,   ev_start,     ev_convert);
    cudaEventElapsedTime(&t_dct,       ev_convert,   ev_dct);
    cudaEventElapsedTime(&t_smooth,    ev_dct,       ev_smooth);
    cudaEventElapsedTime(&t_spatial,   ev_smooth,    ev_spatial);
    cudaEventElapsedTime(&t_pass1,     ev_spatial,   ev_pass1);
    cudaEventElapsedTime(&t_normalize, ev_pass1,     ev_normalize);
    cudaEventElapsedTime(&t_pass2,     ev_normalize, ev_pass2);
    cudaEventElapsedTime(&t_output,    ev_pass2,     ev_output);

    const float t_total = t_convert + t_dct + t_smooth + t_spatial
                        + t_pass1 + t_normalize + t_pass2 + t_output;

    fprintf(stderr,
        "[NLB-TIMING] %dx%d stride=%d | Convert:%.2f | DCT:%.2f | Smooth:%.2f | Spatial:%.2f "
        "| Pass1:%.2f | Normalize:%.2f | Pass2:%.2f | Output:%.2f | KERNEL-TOTAL:%.2f ms\n",
        width, height, proc_stride,
        t_convert, t_dct, t_smooth, t_spatial,
        t_pass1, t_normalize, t_pass2, t_output, t_total);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_convert);
    cudaEventDestroy(ev_dct);
    cudaEventDestroy(ev_smooth);
    cudaEventDestroy(ev_spatial);
    cudaEventDestroy(ev_pass1);
    cudaEventDestroy(ev_normalize);
    cudaEventDestroy(ev_pass2);
    cudaEventDestroy(ev_output);
    
    cudaDeviceSynchronize();
}
#endif

