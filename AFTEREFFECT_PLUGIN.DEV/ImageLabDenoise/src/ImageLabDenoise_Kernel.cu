// ============================================================================
// File: ImageLabDenoise_Kernel.cu  --  KERNELS + ORCHESTRATOR + ENTRY POINT
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
// Matches  enhanceBoundaries(pow = 1 << nbScales) convention.
constexpr int PROC_ALIGN = 4;

// ---  YUV orthonormal color transform (3-channel branch in LibImages.cpp) ---
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
// MAX_SIMILAR_PATCHES is the hard cap ( CPU uses 16 * nSim = 512 in
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
//   planar YUV buffers using orthonormal 3-channel transform, 
//   extending the image with symmetric mirror
//   reflection from (srcWidth, srcHeight) up to (procW, procH).
//
//  forward transform (RGB -> YUV):
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
//   rule  enhanceBoundaries() uses.
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
__global__ void Kernel_ConvertBGRAToOrthonormalWeighted
(
    const float* RESTRICT    inBuffer,
    float*       RESTRICT    outY,
    float*       RESTRICT    outU,
    float*       RESTRICT    outV,
    int                      srcPitchPixels,
    int                      padW,
    int                      srcWidth,
    int                      srcHeight,
    int                      procW,
    int                      procH
)
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
    //  orthonormal 3-channel RGB -> YUV transform.
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
//    RunNoiseEstimate.cpp pipeline. For every 4x4 patch in each
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
void AccumulateChannel
(
    const float (&tile)[BLOCK_DIM_IO_Y + 3][BLOCK_DIM_IO_X + 3],
    int           tx,
    int           ty,
    float*        outCov,
    float*        outMean,
    int*          outCounts
)
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
    //     to edges. This matches  getMean() with offset/rangeMax.
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
//   Kernel_ExtractDCT_And_Variance_3ch. Faithful port of 
//   filterNoiseCurves (CurveFiltering.cpp):
//
//     Phase 0: Normalize raw sums by patch count:
//                  variance[bin, freq] = cov[bin*256 + freq] / counts[bin]
//                  mean[bin]           = mean[bin]          / counts[bin]
//
//     Phase 1: Repeat 5 times ( nbFilter = 5):
//                 1a. For each of 16 DCT frequencies, smooth the per-bin
//                     variance curve with a +/-10-bin window average
//                     ( sizeFilter = 10), skipping empty bins.
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
//   *  CPU stores std, squares to variance for processing, sqrt's
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
// Median helpers for the 4x4 frequency matrix. These match 
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
//  rule:
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
    // Phase 1:  5 sweeps of (bin-axis smoothing + per-bin 4x4 median).
    // ------------------------------------------------------------------------
    constexpr int NB_SWEEPS      = 5;
    constexpr int SMOOTH_WINDOW  = 10;       //  sizeFilter

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
//   by kernel #3 to the spatial-domain 16x16 covariance matrix that 
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
//   This is exactly what  CPU RunNoiseEstimate.cpp / getMatrixBins
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
// Device Helpers: Jacobi eigendecomposition + Lebrun Wiener/Bayes filter
// ============================================================================
//
// Purpose:
//   Shared mathematical core used by both Kernel_NLBayes_Pass1_BasicEstimate
//   and Kernel_NLBayes_Pass2_FinalEstimate. Implements  Bayes step:
//
//     M    = X * Y^(-1)           (filter matrix, 16x16)
//     out  = M * (patch - bary) + bary, clipped to [min, max]
//
//   where the 16x16 matrix Y is inverted via Jacobi eigendecomposition
//   (chosen for numerical stability with near-rank-deficient patch covariance
//   matrices, which are common in flat image regions).
//
//    two passes plug different (X, Y) pairs into the same machine:
//
//     Pass 1 (pilot):   X = C_P - C_N         with diag(C_P) <- max(diag(C_P), diag(C_N))
//                       Y = C_P (diag-clamped)
//                       ==> filter = (C_P - C_N) * C_P^(-1)
//
//     Pass 2 (final):   X = C_basic            (empirical cov from pilot patches)
//                       Y = C_basic + C_N
//                       ==> filter = C_basic * (C_basic + C_N)^(-1)
//
// Implementation structure:
//   EigenJacobi16_Thread0       -- single-threaded cyclic Jacobi for 16x16.
//                                  Runs on thread 0 only while rest of block
//                                  idles; the Jacobi itself is inherently
//                                  sequential (120 rotations/sweep must be
//                                  applied in order since each modifies A).
//
//   ApplyBayesFilter_Block      -- full block-cooperative pipeline.
//                                  Calls Jacobi via thread 0, then uses all
//                                  threads for: eigenvalue inversion,
//                                  M = X * V * D^(-1) * V^T double-matmul,
//                                  per-patch filter application, clip, and
//                                  barycenter add-back.
//
// Shared-memory contract for ApplyBayesFilter_Block:
//   Caller allocates and passes in five 16x16 matrices and one 16-vector of
//   shared memory (plus the patch group and barycenter). Total scratch:
//   (256 + 256 + 256 + 16) * 4 = 3136 bytes of block-scoped workspace.
//
// Performance:
//   The Jacobi inner loop (~960 rotations for 8 sweeps x 120 pairs) runs on
//   thread 0 only -- ~20 microseconds per block on GTX 1060. Rest of block
//   idles during this time. For future optimization, a parallel-Jacobi scheme
//   (Brent-Luk) could reduce this to ~2-3 us but with significantly more
//   complex code. Since there are typically <10000 blocks per frame, total
//   Jacobi cost is around 20 ms per frame -- acceptable for denoising.
// ============================================================================


// ----------------------------------------------------------------------------
// EigenJacobi16_Thread0
//
// In-place classical cyclic Jacobi eigendecomposition of a real symmetric
// 16x16 matrix. Single-threaded: intended to be invoked inside
// `if (threadIdx == 0) { ... }`.
//
// On entry:
//   A[0..255]   symmetric 16x16 matrix, row-major (A[i, j] = A[i*16 + j]).
//   V[0..255]   uninitialized; will be filled by this routine.
//
// On exit:
//   A           off-diagonal ~ 0 (to float precision);
//               diagonal contains the 16 eigenvalues (unsorted).
//   V           columns are orthonormal eigenvectors:
//               V^T * A_original * V == diag(A_final).
//
// Parameters fixed:
//   NUM_SWEEPS = 8       enough to converge 16x16 real-symmetric to ~1e-6.
//   EPSILON    = 1e-7    rotations skipped when |apq| falls below this.
// ----------------------------------------------------------------------------
__device__ __forceinline__
void EigenJacobi16_Thread0(float* RESTRICT A, float* RESTRICT V)
{
    // --- Initialize V to identity ---------------------------------------
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

                // --- Compute rotation parameters ------------------------
                //  theta = (aqq - app) / (2 * apq)
                //  t     = sign(theta) / (|theta| + sqrt(theta^2 + 1))
                //  c     = 1 / sqrt(t^2 + 1)
                //  s     = t * c
                // Using the branch that avoids catastrophic cancellation
                // (matches Golub & Van Loan, Algorithm 8.4.2).
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

                // --- Update the (p, q) 2x2 sub-block --------------------
                A[p * PATCH_ELEMS + p] = app - t * apq;
                A[q * PATCH_ELEMS + q] = aqq + t * apq;
                A[p * PATCH_ELEMS + q] = 0.0f;
                A[q * PATCH_ELEMS + p] = 0.0f;

                // --- Rotate all other rows/columns of A -----------------
                // For every r != p, q:  a_rp' = c*a_rp - s*a_rq
                //                       a_rq' = s*a_rp + c*a_rq
                // We update both halves of the symmetric matrix to keep it
                // consistent.
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
                    A[p * PATCH_ELEMS + r] = new_arp;       // symmetry
                    A[r * PATCH_ELEMS + q] = new_arq;
                    A[q * PATCH_ELEMS + r] = new_arq;       // symmetry
                }

                // --- Accumulate the rotation into V (V_new = V * R) -----
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
// ApplyBayesFilter_Block
//
// Block-cooperative implementation of  Bayes estimate:
//
//   M = X * Y^(-1)
//   for each patch n in [0, nSimP):
//     patch[:, n] = clip( M * (patch_centered[:, n]) + bary, min_val, max_val )
//
// The block is assumed to have at least 16 threads (we rely on thread 0..15
// for per-eigenvalue work and on the full block for matrix ops).
//
// Arguments:
//   sh_X       [256]  in : left factor (preserved on exit).
//   sh_Y       [256]  in/out : matrix to invert (TRASHED on exit).
//   sh_V       [256]  workspace : receives eigenvectors.
//   sh_M       [256]  workspace : receives filter matrix M = X * Y^(-1).
//                                  (caller may read it after the call if useful.)
//   sh_Dinv    [16]   workspace : reciprocal eigenvalues of Y.
//   sh_patches [16*nSimP] in/out : patches stored column-major
//                                  (sh_patches[i * nSimP + n] = pixel i of patch n).
//                                  On entry: centered (patch - bary).
//                                  On exit : filtered + bary-added + clipped.
//   sh_bary    [16]   in : barycenter to re-add after filtering.
//   min_val    float  in : clip floor.
//   max_val    float  in : clip ceiling.
//   nSimP      int    in : number of patches in the group.
//
// Does NOT __syncthreads() at the very end. Caller is responsible for syncing
// before reusing any of the shared buffers.
// ============================================================================
__device__ void ApplyBayesFilter_Block
(
    const float* RESTRICT sh_X,
    float*       RESTRICT sh_Y,
    float*       RESTRICT sh_V,
    float*       RESTRICT sh_M,
    float*       RESTRICT sh_Dinv,
    float*       RESTRICT sh_patches,
    const float* RESTRICT sh_bary,
    float                 min_val,
    float                 max_val,
    int                   nSimP
)
{
    const int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    // ------------------------------------------------------------------------
    // Phase 1: Jacobi eigendecomposition of Y.
    //   On exit: diag(sh_Y) = eigenvalues, sh_V = eigenvector matrix.
    //   Thread 0 runs the sequential Jacobi; all other threads wait at sync.
    // ------------------------------------------------------------------------
    if (tid == 0)
    {
        EigenJacobi16_Thread0(sh_Y, sh_V);
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 2: Extract and safely invert eigenvalues.
    //   Clamp at a small positive floor (1e-6) to absorb near-zero values
    //   that would otherwise produce Inf/NaN in the inverse. This is the
    //   numerical stabilizer  diagonal-clamping trick was meant to
    //   emulate; keeping it in the eigenvalue domain is mathematically
    //   equivalent for Y = C_P (Pass 1) and strictly safer for Y = C_basic
    //   + C_N (Pass 2).
    // ------------------------------------------------------------------------
    if (tid < PATCH_ELEMS)
    {
        const float lambda = sh_Y[tid * (PATCH_ELEMS + 1)];  // diagonal offset: k*16 + k = k*17
        sh_Dinv[tid] = 1.0f / fmaxf(1e-6f, lambda);
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 3: Compute intermediate  P = X * V * D^(-1), store in sh_Y.
    //   P[i, d] = D_inv[d] * sum_k( X[i, k] * V[k, d] )
    //   Each thread handles one element of the 16x16 result, striding over
    //   the 256-element range in chunks of block_size.
    //   The diagonal scale is fused into the store to save one pass + sync.
    // ------------------------------------------------------------------------
    for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
    {
        const int i = k / PATCH_ELEMS;      // 0..15
        const int d = k % PATCH_ELEMS;      // 0..15

        float sum = 0.0f;
        #pragma unroll
        for (int col = 0; col < PATCH_ELEMS; ++col)
        {
            sum += sh_X[i * PATCH_ELEMS + col] * sh_V[col * PATCH_ELEMS + d];
        }
        sh_Y[i * PATCH_ELEMS + d] = sum * sh_Dinv[d];
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 4: Compute M = P * V^T, store in sh_M.
    //   M[i, j] = sum_d( P[i, d] * V[j, d] )        // V^T[d, j] = V[j, d]
    // ------------------------------------------------------------------------
    for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
    {
        const int i = k / PATCH_ELEMS;
        const int j = k % PATCH_ELEMS;

        float sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < PATCH_ELEMS; ++d)
        {
            sum += sh_Y[i * PATCH_ELEMS + d] * sh_V[j * PATCH_ELEMS + d];
        }
        sh_M[i * PATCH_ELEMS + j] = sum;
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 5: Per-patch application: out = clip(M * centered + bary, min, max).
    //   Distribute columns across the block. Each thread loads its column
    //   into a 16-float register array, applies M via two-level unrolled
    //   loop, adds bary, clips, and writes back.
    //   Data layout: sh_patches[i * nSimP + n] where i = pixel, n = patch.
    // ------------------------------------------------------------------------
    for (int n = tid; n < nSimP; n += block_size)
    {
        // Load centered patch column into registers (kept as register array
        // because both loops are unrolled at compile time).
        float col_in[PATCH_ELEMS];
        #pragma unroll
        for (int i = 0; i < PATCH_ELEMS; ++i)
        {
            col_in[i] = sh_patches[i * nSimP + n];
        }

        // Apply M, add barycenter, clip, write back.
        #pragma unroll
        for (int i = 0; i < PATCH_ELEMS; ++i)
        {
            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < PATCH_ELEMS; ++j)
            {
                acc += sh_M[i * PATCH_ELEMS + j] * col_in[j];
            }
            const float v = acc + sh_bary[i];
            sh_patches[i * nSimP + n] = fminf(max_val, fmaxf(min_val, v));
        }
    }

    // NOTE: no trailing __syncthreads().  Caller is responsible for
    // synchronizing before reusing sh_Y, sh_V, sh_M, sh_Dinv, or sh_patches.
}


// ============================================================================
// Kernel_NLBayes_Pass1_BasicEstimate
// ============================================================================
//
// Purpose:
//   First pass of  NL-Bayes (the "pilot" or "basic" estimate). For
//   each reference pixel on the processing grid:
//
//     1. Load a 20x20 window tile per channel (17x17 search area + 3-pixel
//        apron so every candidate's 4x4 patch fits within shared memory).
//     2. Determine per-channel noise sigma from the LUT trace; derive the
//        channel-weighted patch-distance threshold.
//     3. Search all (SW*SW) = 289 candidate patches in the window.
//        Each candidate below threshold is added to the similar-patch list
//        (capped at MAX_SIMILAR_PATCHES = 128).
//     4. For each channel c in {Y, U, V}:
//          4a. Extract the nSimP similar patches into a (16 x nSimP)
//              column-major shared matrix.
//          4b. Compute the per-pixel-position barycenter across the group.
//          4c. Compute the clip range [min - sigma, max + sigma] over the
//              un-centered group values.
//          4d. Build  un-centered raw correlation matrix
//                C_P = P * P^T / (nSimP - 1)
//          4e. Determine the channel's intensity bin from the barycenter
//              mean, load C_N from the 16x16 noise-cov LUT.
//          4f. Compute X = C_P - C_N; diagonal-clamp C_P for PSD guarantee.
//          4g. Center patches in place (patch - bary).
//          4h. Call ApplyBayesFilter_Block(X, C_P, ..., sh_patches, bary)
//              which leaves the filtered, bary-added, clipped result in
//              sh_patches.
//          4i. Atomically aggregate each filtered patch pixel back into
//              d_Accum_c at its original global coordinate.
//              Also increment d_Weight once per (patch, pixel) -- but only
//              when ch == 0, so the weight counter is shared across channels
//              (matches Lebrun: all three channel weights are always equal).
//
// Mathematical contract:
//   Filter applied is exactly  Pass-1 Bayes estimate:
//       denoised = bary + (C_P - C_N) * C_P^(-1) * (patch - bary)
//     = bary + X * Y^(-1) * (patch - bary)
//   with clip to [min - sigma, max + sigma] per channel.
//
// Launch geometry:
//   grid  = (ceil(procW / proc_stride), ceil(procH / proc_stride))
//   block = (BLOCK_DIM_MATH_X, BLOCK_DIM_MATH_Y) = (32, 8) = 256 threads
//   shared ~ 18 KB per block (fits comfortably on all CC >= 6.1)
// ============================================================================


// ----------------------------------------------------------------------------
// Window-tile size: SEARCH_WINDOW_SIZE + PATCH_SIZE - 1 = 17 + 4 - 1 = 20.
// This covers every pixel touched by any 4x4 patch whose top-left lies in
// the 17x17 search window.
// ----------------------------------------------------------------------------
static constexpr int WINDOW_TILE_SIZE = SEARCH_WINDOW_SIZE + PATCH_SIZE - 1;   // 20
static constexpr int WINDOW_TILE_ELEMS = WINDOW_TILE_SIZE * WINDOW_TILE_SIZE;  // 400


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

    // (void) casts suppress unused-parameter warnings for buffers we don't use
    // in Pass 1. They are part of the shared signature used by Pass 2.
    (void) noiseMeanY;
    (void) noiseMeanU;
    (void) noiseMeanV;

    // ------------------------------------------------------------------------
    // Reference patch top-left coordinate.
    //
    // Valid range (so that the 17x17 search window and each candidate's 4x4
    // patch all stay within [0, procW) x [0, procH)):
    //   ref_x in [SWR, procW - PATCH_SIZE - SWR]
    //   ref_y in [SWR, procH - PATCH_SIZE - SWR]
    // Blocks outside this range return immediately (the grid is slightly
    // overprovisioned near the edges).
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

    // Top-left of the window in global image coordinates.
    const int win_x0 = ref_x - SEARCH_WINDOW_RADIUS;
    const int win_y0 = ref_y - SEARCH_WINDOW_RADIUS;

    // ------------------------------------------------------------------------
    // Shared memory layout.
    // ------------------------------------------------------------------------
    __shared__ float s_winY[WINDOW_TILE_ELEMS];          // 400 * 4 =   1600 B
    __shared__ float s_winU[WINDOW_TILE_ELEMS];
    __shared__ float s_winV[WINDOW_TILE_ELEMS];          // -> 3 * 1600 = 4800 B

    __shared__ int   s_patchIndices[MAX_SIMILAR_PATCHES]; // 128 * 4 =   512 B
    __shared__ int   s_patchCount;

    // Scalar aggregates.
    __shared__ float s_sigma_Y;
    __shared__ float s_sigma_U;
    __shared__ float s_sigma_V;
    __shared__ float s_threshold;

    // Matrix workspaces (4 x 256 + 16 = 4160 B), reused across channels.
    __shared__ float sh_A    [PATCH_ELEMS_SQ];    // C_P (clamped), then trashed by helper
    __shared__ float sh_B    [PATCH_ELEMS_SQ];    // C_N, then V workspace
    __shared__ float sh_C    [PATCH_ELEMS_SQ];    // X = C_P - C_N (preserved during helper)
    __shared__ float sh_D    [PATCH_ELEMS_SQ];    // filter matrix M workspace
    __shared__ float sh_Dinv [PATCH_ELEMS];

    __shared__ float sh_bary    [PATCH_ELEMS];
    __shared__ float sh_patches [PATCH_ELEMS * MAX_SIMILAR_PATCHES];  // 16*128*4 = 8192 B

    // Per-channel reductions.
    __shared__ float s_min_tmp[PATCH_ELEMS];
    __shared__ float s_max_tmp[PATCH_ELEMS];
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ int   s_bin_ch;

    // ------------------------------------------------------------------------
    // Phase 1: Cooperative load of the 20x20 window tile for each channel.
    // All pixels are guaranteed in-bounds by the early-return above.
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
    //
    //   sigma_c    = sqrt( max(0, trace(C_N_c)) / 16 )        [Lebrun, NlBayes.cpp:847]
    //   sigmaMean  = 0.5*sigma_Y^2 + 0.25*sigma_U^2 + 0.25*sigma_V^2
    //   threshold  = 0.5*tau_Y*sigma_Y^2 + 0.25*tau_UV*(sigma_U^2 + sigma_V^2)
    //
    // At default controls (tau_Y = tau_UV = 48), this reduces exactly to
    //   tau * sigmaMean  with tau = 3*sP^2.
    // ------------------------------------------------------------------------
    if (tid == 0)
    {
        // Reference patch mean (Y channel) for bin lookup.
        float sum_Y = 0.0f;
        #pragma unroll
        for (int i = 0; i < PATCH_SIZE; ++i)
        {
            #pragma unroll
            for (int j = 0; j < PATCH_SIZE; ++j)
            {
                const int p_idx = (SEARCH_WINDOW_RADIUS + i) * WINDOW_TILE_SIZE + (SEARCH_WINDOW_RADIUS + j);
                sum_Y += s_winY[p_idx];
            }
        }
        const float mean_Y = sum_Y * (1.0f / static_cast<float>(PATCH_ELEMS));
        const int   ref_bin = max(0, min(NOISE_BINS - 1,
                                         __float2int_rn(mean_Y * 255.0f)));

        // Sum of diagonal of each channel's 16x16 noise cov at this bin.
        const int base = ref_bin * PATCH_ELEMS_SQ;
        float trace_Y = 0.0f;
        float trace_U = 0.0f;
        float trace_V = 0.0f;
        #pragma unroll
        for (int k = 0; k < PATCH_ELEMS; ++k)
        {
            const int diag = k * (PATCH_ELEMS + 1);    // k*16 + k = k*17
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

        // Distance threshold, with channel weights folded in (Y=0.5, U=V=0.25
        // to match  getWeight convention). tau_Y and tau_UV already
        // include the user's AlgoControls modulation.
        s_threshold = 0.5f  * tau_Y  * sigma2_Y
                    + 0.25f * tau_UV * sigma2_U
                    + 0.25f * tau_UV * sigma2_V;
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 3: Patch similarity search.
    //
    // For each candidate patch top-left (cx, cy) in the 17x17 search grid:
    //   dist = sum_{i,j} [ 0.5*(dY)^2 + 0.25*(dU)^2 + 0.25*(dV)^2 ]
    // If dist <= threshold, atomically claim a slot in s_patchIndices.
    //
    // The reference patch itself (at (SWR, SWR)) has dist = 0 and is always
    // selected -- guarantees at least one patch in the group.
    // ------------------------------------------------------------------------
    {
        const int ref_px = SEARCH_WINDOW_RADIUS;      // 8
        const int ref_py = SEARCH_WINDOW_RADIUS;      // 8
        const int n_candidates = SEARCH_WINDOW_SIZE * SEARCH_WINDOW_SIZE;  // 289

        for (int cidx = tid; cidx < n_candidates; cidx += block_size)
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

            if (dist <= s_threshold)
            {
                const int slot = atomicAdd(&s_patchCount, 1);
                if (slot < MAX_SIMILAR_PATCHES)
                {
                    s_patchIndices[slot] = cidx;
                }
            }
        }
    }
    __syncthreads();

    const int nSimP = min(s_patchCount, MAX_SIMILAR_PATCHES);

    // Need at least 2 patches to compute a covariance; otherwise skip this block.
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
        //     sh_patches[i * nSimP + n]  =  pixel i of patch n
        // --------------------------------------------------------------------
        for (int k = tid; k < PATCH_ELEMS * nSimP; k += block_size)
        {
            const int i = k / nSimP;       // 0..15, pixel within patch
            const int n = k % nSimP;       // 0..nSimP-1, patch index

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
        // 4b. Compute barycenter: sh_bary[i] = mean over patches of pixel i.
        // 16 threads in parallel handle the 16 pixel positions.
        // --------------------------------------------------------------------
        if (tid < PATCH_ELEMS)
        {
            float sum = 0.0f;
            for (int n = 0; n < nSimP; ++n)
            {
                sum += sh_patches[tid * nSimP + n];
            }
            sh_bary[tid] = sum / static_cast<float>(nSimP);
        }

        // --------------------------------------------------------------------
        // 4c. Clip range: min/max of the un-centered group, widened by +/-sigma.
        //     First reduce to 16 per-row min/max; then thread 0 reduces to scalar.
        // --------------------------------------------------------------------
        if (tid < PATCH_ELEMS)
        {
            float mn =  INFINITY;
            float mx = -INFINITY;
            for (int n = 0; n < nSimP; ++n)
            {
                const float v = sh_patches[tid * nSimP + n];
                mn = fminf(mn, v);
                mx = fmaxf(mx, v);
            }
            s_min_tmp[tid] = mn;
            s_max_tmp[tid] = mx;
        }
        __syncthreads();

        if (tid == 0)
        {
            float mn = s_min_tmp[0];
            float mx = s_max_tmp[0];
            #pragma unroll
            for (int k = 1; k < PATCH_ELEMS; ++k)
            {
                mn = fminf(mn, s_min_tmp[k]);
                mx = fmaxf(mx, s_max_tmp[k]);
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
        // 4d. Raw (un-centered) correlation matrix C_P = P * P^T / (nSimP - 1)
        //     Written into sh_A. Matches  LibMatrix.cpp covarianceMatrix.
        //
        //     Note: the normalization uses (nSimP - 1),  convention.
        //     We access the column-major sh_patches for both operands.
        // --------------------------------------------------------------------
        {
            const float normInv = 1.0f / static_cast<float>(nSimP - 1);

            for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
            {
                const int i = k / PATCH_ELEMS;
                const int j = k % PATCH_ELEMS;

                float sum = 0.0f;
                for (int n = 0; n < nSimP; ++n)
                {
                    sum += sh_patches[i * nSimP + n] * sh_patches[j * nSimP + n];
                }
                sh_A[k] = sum * normInv;
            }
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4e. Load noise covariance C_N for this channel+bin into sh_B.
        // --------------------------------------------------------------------
        {
            const int bin_base = s_bin_ch * PATCH_ELEMS_SQ;
            for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
            {
                sh_B[k] = noiseCov[bin_base + k];
            }
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4f. Build the Bayes-filter inputs:
        //       sh_C = C_P - C_N             (= X for the helper)
        //       sh_A[diag] = max(sh_A[diag], sh_B[diag])   (diagonal PD clamp)
        //                                    (= Y for the helper; overwritten)
        //     These two operations touch disjoint elements and can run in
        //     one fused loop.
        // --------------------------------------------------------------------
        for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
        {
            sh_C[k] = sh_A[k] - sh_B[k];
        }
        if (tid < PATCH_ELEMS)
        {
            const int d = tid * (PATCH_ELEMS + 1);      // diagonal index k*17
            sh_A[d] = fmaxf(sh_A[d], sh_B[d]);
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
        // 4h. Apply Bayes filter.
        //     Helper consumes:  sh_C (= X, read-only),  sh_A (= Y, trashed).
        //     Helper workspace: sh_B (= V), sh_D (= M), sh_Dinv.
        //     Helper mutates:   sh_patches (filtered + bary + clipped).
        // --------------------------------------------------------------------
        ApplyBayesFilter_Block(
            sh_C,          // X = C_P - C_N
            sh_A,          // Y = C_P (diag-clamped); trashed
            sh_B,          // V workspace (previously held C_N; no longer needed)
            sh_D,          // M workspace
            sh_Dinv,
            sh_patches,    // input: centered.  output: filtered + bary + clipped.
            sh_bary,
            s_min,
            s_max,
            nSimP);

        // The helper does NOT trailing-sync; do it here before global writes.
        __syncthreads();

        // --------------------------------------------------------------------
        // 4i. Aggregate filtered patches into global accumulators.
        //
        //     Each (patch n, pixel i) contributes its filtered value to the
        //     target Accum buffer at the corresponding global pixel. Weight
        //     is incremented ONCE per (patch, pixel) -- only during ch==0
        //     -- since the GPU uses a single shared weight buffer (mathematically
        //     equivalent to  three always-identical per-channel weights).
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
            const int i = k / nSimP;       // 0..15
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
        __syncthreads();   // ensure channel loop's shared buffers are safe to reuse
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
//   This is  computeWeightedAggregation (NlBayes.cpp:687), adapted
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
//    standard fallback -- it avoids NaN and leaves those positions
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
        // Uncovered pixel -- fall back to the noisy input. Matches 
        // computeWeightedAggregation "if (iW[k] > 0.f) ... else iO[k] = iN[k]"
        // exactly.
        pilotY[idx] = fallbackY[idx];
        pilotU[idx] = fallbackU[idx];
        pilotV[idx] = fallbackV[idx];
    }
}



// ============================================================================
// Kernel_NLBayes_Pass2_FinalEstimate
// ============================================================================
//
// Purpose:
//   Second (and final) pass of  NL-Bayes. For each reference pixel
//   on the processing grid:
//
//     1. Load two window tiles per channel (noisy + pilot = 6 tiles total).
//     2. Compute per-channel sigma (trace-based) and distance threshold.
//        Threshold follows  Step-2 convention: weights are uniform
//        (w_Y = w_U = w_V = 1), and tau carries an extra factor of nChannels
//        relative to Pass 1 (tau_base = 3*sP^2*nChannels = 144 at defaults).
//     3. Search similar patches using the PILOT image (not noisy). This is
//        the key idea of NL-Bayes: the pilot's lower noise makes the patch-
//        match much more reliable than direct noisy-to-noisy matching.
//     4. For each channel c:
//          4a. Load noisy patches into sh_patches (uncentered).
//          4b. Compute barycenter from NOISY ( convention).
//          4c. Compute clip range [min - sigma, max + sigma] from PILOT
//              values (un-centered), widened by sigma_c.
//          4d. Build C_basic = B * B^T / (nSimP - 1)
//              where B[i, n] = pilot[i, n] - bary_noisy[i]
//              i.e. covariance of pilot CENTERED BY NOISY MEAN -- this is
//               subtle oracle-variance convention.
//          4e. Determine channel's intensity bin from barycenter mean,
//              load C_N from the 16x16 LUT.
//          4f. Build Bayes inputs: X = C_basic (preserved);
//                                  Y = C_basic + C_N (will be inverted).
//          4g. Center noisy patches in place: sh_patches -= bary_noisy.
//          4h. Call ApplyBayesFilter_Block -- computes
//                  M = X * Y^(-1) = C_basic * (C_basic + C_N)^(-1)
//              and applies it to sh_patches with bary-add + clip.
//          4i. Atomically aggregate filtered values into d_Accum_c and
//              increment d_Weight once per (patch, pixel) at ch == 0.
//
// Mathematical contract (Lebrun NlBayes.cpp:500-595, per-channel
// simplification of the 48x48 cross-channel joint filter):
//
//   bary_c     = mean_n( noisy[c, :, n] )
//   B[c, :, n] = pilot[c, :, n] - bary_c
//   C_basic_c  = B * B^T / (nSimP - 1)
//   C_P+N_c    = C_basic_c + C_N_c
//   filter_c   = C_basic_c * (C_P+N_c)^(-1)
//   out_c[:, n]= clip( filter_c * (noisy[c, :, n] - bary_c) + bary_c,
//                      min_c - sigma_c, max_c + sigma_c )
//
// Known deviation from Lebrun:
//   The CPU reference processes all three channels as a joint 48-dimensional
//   vector with a 48x48 covariance and a full 48x48 inversion. Our per-
//   channel simplification (16x16 per channel, independently) preserves the
//   mathematical structure of  filter but loses the cross-channel
//   correlations, which typically contribute a small but nonzero quality
//   gain on color images. This is the A4 item acknowledged upfront.
//
// Launch geometry:
//   grid  = (ceil(procW / proc_stride), ceil(procH / proc_stride))
//   block = (BLOCK_DIM_MATH_X, BLOCK_DIM_MATH_Y) = (32, 8) = 256 threads
//   shared ~ 23 KB per block
// ============================================================================


__global__ void Kernel_NLBayes_Pass2_FinalEstimate(
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
    int                      proc_stride)
{
    const int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    // These buffers are part of the shared kernel signature but unused in
    // Pass 2 -- Lebrun uses the nearest integer bin for the LUT, not the
    // interpolated per-bin mean. Kept in the signature for future multi-
    // scale / interpolation work.
    (void) noiseMeanY;
    (void) noiseMeanU;
    (void) noiseMeanV;

    // ------------------------------------------------------------------------
    // Reference coordinate and boundary check (same logic as Pass 1).
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
    // Shared memory layout (~23 KB total).
    // ------------------------------------------------------------------------
    __shared__ float s_noisyY[WINDOW_TILE_ELEMS];        // 3 * 400 * 4 = 4800 B
    __shared__ float s_noisyU[WINDOW_TILE_ELEMS];
    __shared__ float s_noisyV[WINDOW_TILE_ELEMS];
    __shared__ float s_pilotY[WINDOW_TILE_ELEMS];        // 3 * 400 * 4 = 4800 B
    __shared__ float s_pilotU[WINDOW_TILE_ELEMS];
    __shared__ float s_pilotV[WINDOW_TILE_ELEMS];

    __shared__ int   s_patchIndices[MAX_SIMILAR_PATCHES];  // 128 * 4 = 512 B
    __shared__ int   s_patchCount;

    __shared__ float s_sigma_Y;
    __shared__ float s_sigma_U;
    __shared__ float s_sigma_V;
    __shared__ float s_threshold;

    // Matrix workspaces (4 x 256 + 16 = 4160 B)
    __shared__ float sh_A    [PATCH_ELEMS_SQ];    // C_basic, then Y = C_basic + C_N (trashed)
    __shared__ float sh_B    [PATCH_ELEMS_SQ];    // C_N, then V workspace
    __shared__ float sh_C    [PATCH_ELEMS_SQ];    // X = C_basic (preserved for helper)
    __shared__ float sh_D    [PATCH_ELEMS_SQ];    // filter matrix M workspace
    __shared__ float sh_Dinv [PATCH_ELEMS];

    __shared__ float sh_bary    [PATCH_ELEMS];
    __shared__ float sh_patches [PATCH_ELEMS * MAX_SIMILAR_PATCHES];  // noisy, 16*128*4 = 8192 B

    __shared__ float s_min_tmp[PATCH_ELEMS];
    __shared__ float s_max_tmp[PATCH_ELEMS];
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ int   s_bin_ch;

    // ------------------------------------------------------------------------
    // Phase 1: Load both noisy and pilot window tiles.
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
    // Phase 2: sigma per channel + Pass-2 threshold.
    //
    //   threshold = 3 * ( tau_Y * sigma_Y^2 + tau_UV * (sigma_U^2 + sigma_V^2) )
    //
    // The extra factor of 3 (= nChannels) vs. Pass 1 is  Step-2 tau
    // expansion (3 * sP^2 * nChannels instead of 3 * sP^2). At default user
    // controls (tau_Y = tau_UV = 48), this gives 144 * (sY^2 + sU^2 + sV^2)
    // which is equivalent to   144 * sigmaMean  (weight_c = 1).
    //
    // Deliberately sigma-scaled (not literal-constant 144) so the threshold
    // self-adapts to the input image's amplitude -- floating-point images
    // in [0, 1] will otherwise see every patch pass the literal threshold.
    // ------------------------------------------------------------------------
    if (tid == 0)
    {
        // Reference-patch mean from noisy Y channel for the bin lookup.
        // Using noisy Y is  Step-1 convention; for Step-2 the per-
        // channel bin is refined per channel in phase 4e below.
        float sum_Y = 0.0f;
        #pragma unroll
        for (int i = 0; i < PATCH_SIZE; ++i)
        {
            #pragma unroll
            for (int j = 0; j < PATCH_SIZE; ++j)
            {
                const int p_idx = (SEARCH_WINDOW_RADIUS + i) * WINDOW_TILE_SIZE + (SEARCH_WINDOW_RADIUS + j);
                sum_Y += s_noisyY[p_idx];
            }
        }
        const float mean_Y  = sum_Y * (1.0f / static_cast<float>(PATCH_ELEMS));
        const int   ref_bin = max(0, min(NOISE_BINS - 1,
                                         __float2int_rn(mean_Y * 255.0f)));

        // Sigma per channel from trace of 16x16 noise cov at this bin.
        const int base = ref_bin * PATCH_ELEMS_SQ;
        float trace_Y = 0.0f;
        float trace_U = 0.0f;
        float trace_V = 0.0f;
        #pragma unroll
        for (int k = 0; k < PATCH_ELEMS; ++k)
        {
            const int diag = k * (PATCH_ELEMS + 1);   // k*17
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

        // Step-2 threshold: weights = 1 per channel; tau expanded by nChannels.
        s_threshold = 3.0f * (tau_Y  * sigma2_Y
                            + tau_UV * sigma2_U
                            + tau_UV * sigma2_V);
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 3: Patch similarity search using the PILOT image.
    //
    //   dist = sum_{i,j} [ (dY_pilot)^2 + (dU_pilot)^2 + (dV_pilot)^2 ]
    //
    // Weights are uniform (all 1) per  estimateSimilarPatchesStep2.
    // The reference patch (at (SWR, SWR) relative to the window) has dist=0
    // and is always included.
    // ------------------------------------------------------------------------
    {
        const int ref_px = SEARCH_WINDOW_RADIUS;
        const int ref_py = SEARCH_WINDOW_RADIUS;
        const int n_candidates = SEARCH_WINDOW_SIZE * SEARCH_WINDOW_SIZE;

        for (int cidx = tid; cidx < n_candidates; cidx += block_size)
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

            if (dist <= s_threshold)
            {
                const int slot = atomicAdd(&s_patchCount, 1);
                if (slot < MAX_SIMILAR_PATCHES)
                {
                    s_patchIndices[slot] = cidx;
                }
            }
        }
    }
    __syncthreads();

    const int nSimP = min(s_patchCount, MAX_SIMILAR_PATCHES);

    // Need at least 2 patches for a meaningful covariance.
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
        // 4a. Extract NOISY patches into sh_patches (column-major, uncentered).
        //     sh_patches[i * nSimP + n]  =  pixel i of noisy patch n.
        // --------------------------------------------------------------------
        for (int k = tid; k < PATCH_ELEMS * nSimP; k += block_size)
        {
            const int i = k / nSimP;            // 0..15
            const int n = k % nSimP;            // 0..nSimP-1

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
        // 4b. Barycenter from NOISY patches ( Step-2 convention).
        //     sh_bary[i] = mean_n( noisy[i, n] )
        // --------------------------------------------------------------------
        if (tid < PATCH_ELEMS)
        {
            float sum = 0.0f;
            for (int n = 0; n < nSimP; ++n)
            {
                sum += sh_patches[tid * nSimP + n];
            }
            sh_bary[tid] = sum / static_cast<float>(nSimP);
        }

        // --------------------------------------------------------------------
        // 4c. Clip range from PILOT values (un-centered), widened by sigma_c.
        //     Thread tid < 16 reduces per pixel-position; thread 0 reduces
        //     further to scalar min/max.
        // --------------------------------------------------------------------
        if (tid < PATCH_ELEMS)
        {
            const int i  = tid;
            const int py = i / PATCH_SIZE;
            const int px = i % PATCH_SIZE;

            float mn =  INFINITY;
            float mx = -INFINITY;
            for (int n = 0; n < nSimP; ++n)
            {
                const int p_idx = s_patchIndices[n];
                const int p_cx  = p_idx % SEARCH_WINDOW_SIZE;
                const int p_cy  = p_idx / SEARCH_WINDOW_SIZE;

                const int win_idx = (p_cy + py) * WINDOW_TILE_SIZE + (p_cx + px);
                const float v = s_pilot[win_idx];
                mn = fminf(mn, v);
                mx = fmaxf(mx, v);
            }
            s_min_tmp[tid] = mn;
            s_max_tmp[tid] = mx;
        }
        __syncthreads();

        if (tid == 0)
        {
            float mn = s_min_tmp[0];
            float mx = s_max_tmp[0];
            #pragma unroll
            for (int k = 1; k < PATCH_ELEMS; ++k)
            {
                mn = fminf(mn, s_min_tmp[k]);
                mx = fmaxf(mx, s_max_tmp[k]);
            }
            s_min = mn - sigma_c;
            s_max = mx + sigma_c;

            // Per-channel intensity bin from barycenter mean (Lebrun uses the
            // mean of NOISY patches to pick the bin -- our bary IS the noisy
            // per-pixel mean, so averaging across pixel positions reproduces
            // the same bin).
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
        // 4d. C_basic = B * B^T / (nSimP - 1), where
        //       B[i, n] = pilot[i, n] - bary_noisy[i]
        //
        // Computed directly from the pilot window + sh_bary, without
        // materializing a sh_patches_basic buffer -- saves 8 KB of shared
        // memory at the cost of a few extra loads per matrix element.
        // --------------------------------------------------------------------
        {
            const float normInv = 1.0f / static_cast<float>(nSimP - 1);

            for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
            {
                const int i = k / PATCH_ELEMS;
                const int j = k % PATCH_ELEMS;

                const int py_i = i / PATCH_SIZE;
                const int px_i = i % PATCH_SIZE;
                const int py_j = j / PATCH_SIZE;
                const int px_j = j % PATCH_SIZE;

                const float bary_i = sh_bary[i];
                const float bary_j = sh_bary[j];

                float sum = 0.0f;
                for (int n = 0; n < nSimP; ++n)
                {
                    const int p_idx = s_patchIndices[n];
                    const int p_cx  = p_idx % SEARCH_WINDOW_SIZE;
                    const int p_cy  = p_idx / SEARCH_WINDOW_SIZE;

                    const float bi = s_pilot[(p_cy + py_i) * WINDOW_TILE_SIZE + (p_cx + px_i)] - bary_i;
                    const float bj = s_pilot[(p_cy + py_j) * WINDOW_TILE_SIZE + (p_cx + px_j)] - bary_j;

                    sum += bi * bj;
                }
                sh_A[k] = sum * normInv;        // sh_A = C_basic
            }
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4e. Load C_N for this (channel, bin) into sh_B.
        // --------------------------------------------------------------------
        {
            const int bin_base = s_bin_ch * PATCH_ELEMS_SQ;
            for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
            {
                sh_B[k] = noiseCov[bin_base + k];
            }
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // 4f. Build the Bayes-filter inputs:
        //       sh_C = C_basic              (= X, preserved through helper)
        //       sh_A = C_basic + C_N        (= Y, inverted and trashed)
        //
        //     Fused in one pass: every k-th thread writes its element of
        //     sh_C and updates its element of sh_A.
        // --------------------------------------------------------------------
        for (int k = tid; k < PATCH_ELEMS_SQ; k += block_size)
        {
            const float v_basic = sh_A[k];
            sh_C[k] = v_basic;                      // X = C_basic
            sh_A[k] = v_basic + sh_B[k];            // Y = C_basic + C_N
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
        // 4h. Apply Bayes filter.
        //     Helper consumes:  sh_C (= X, read-only),  sh_A (= Y, trashed).
        //     Helper workspace: sh_B (= V), sh_D (= M), sh_Dinv.
        //     Helper mutates:   sh_patches (filtered + bary + clipped).
        // --------------------------------------------------------------------
        ApplyBayesFilter_Block(
            sh_C,          // X = C_basic
            sh_A,          // Y = C_basic + C_N; trashed
            sh_B,          // V workspace (was C_N; no longer needed)
            sh_D,          // M workspace
            sh_Dinv,
            sh_patches,    // in: centered noisy.  out: filtered + bary + clipped.
            sh_bary,
            s_min,
            s_max,
            nSimP);

        __syncthreads();    // helper does not trailing-sync

        // --------------------------------------------------------------------
        // 4i. Aggregate filtered patches back to global accumulators.
        //     Weight is incremented once per (patch, pixel) at ch == 0 only,
        //     matching the shared-weight-buffer convention.
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
        __syncthreads();    // ensure channel's shared buffers are safe to reuse
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
//         forward matrix (correct because the forward transform is
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
        // Matches  computeWeightedAggregation fallback.
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
        if (!alloc_cuda_memory_buffers(g_gpuMemState, procW, procH))
        {
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
    // Map AlgoControls ->  tau threshold scalers
    //
    //  per-step algorithmic constant is  tau_base = 3 * sP^2 = 48.
    // In the reference CPU code this is fixed; here we expose it to the user
    // through AlgoControls, preserving identical behavior at default = 1.0f.
    //
    // tau_Y  applies to the luma (Y) channel
    // tau_UV applies to both chroma (U, V) channels
    // -----------------------------------------------------------------------
    constexpr float tau_base = 3.0f * static_cast<float>(PATCH_ELEMS);
    const float master   = algoGpuParams->master_denoise_amount;
    const float fine     = algoGpuParams->fine_detail_preservation;

    const float tau_Y  = tau_base * master * algoGpuParams->luma_strength   * fine;
    const float tau_UV = tau_base * master * algoGpuParams->chroma_strength * fine;

    // -----------------------------------------------------------------------
    // Precompute byte sizes for async memset clears
    // -----------------------------------------------------------------------
    const size_t bytes_frame             = static_cast<size_t>(g_gpuMemState.frameSizePadded) * sizeof(float);
    constexpr size_t bytes_noise_cov     = static_cast<size_t>(NOISE_BINS) * PATCH_ELEMS_SQ * sizeof(float);
    constexpr size_t bytes_noise_mean    = static_cast<size_t>(NOISE_BINS) * sizeof(float);
    constexpr size_t bytes_noise_counts  = static_cast<size_t>(NOISE_BINS) * sizeof(int);

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
