// ============================================================================
//  AFMF_FP32_CUDA.cu
//
//  Adaptive Frequency Median Filter (Erkan, Enginoglu, Thanh, Hieu, 2020)
//  CUDA / FP32 implementation for Adobe After Effects / Premiere Pro.
//
//  Target compute capability: sm_61 (Pascal, GTX 1060) and up.
//  Tested with CUDA 10.2 toolkit.
//
//  I/O CONTRACT:
//    Input  : BGRA_32f interleaved, each channel in [0.0, 1.0]
//    Output : BGRA_32f interleaved, each channel in [0.0, 1.0]
//    Pitch  : in pixels (BGRA pixels per row); float-stride = pitch * 4
//
//  INTERNAL PROCESSING:
//    The kernel splits BGRA into 3 single-channel planar buffers, scales
//    them to [0.0, 255.0] internally (matching the CPU/AVX2 pipeline's
//    convention bit-for-bit), runs AFMF per channel, then merges back
//    and rescales to [0.0, 1.0] on store. Alpha passes through verbatim.
//
//    Working in [0,255] internally means tolerance (default 0.1), the
//    noise-map +128 centering, and the [0,255] clamp limits are reused
//    from the CPU code unchanged.
//
//  PIPELINE PER FRAME:
//    1. split_bgra_to_planar:  BGRA[0,1] -> 3 planes [0,255]
//    2. afmf_kernel_planar:    per-channel, iterations + ping-pong
//    3. merge_planar_to_bgra:  3 planes [0,255] -> BGRA[0,1] (+ alpha)
//       OR merge_noise_map_to_bgra for noise-map output
//
//  MEMORY FOOTPRINT (internal):
//    Per call: FOUR cudaMalloc allocations (3 channel planes + 1 shared
//      ping-pong scratch).  inBuffer / outBuffer are caller-owned device
//      buffers; the kernels read/write them directly with no extra
//      staging.
//    HD : ~32 MB,   4K UHD : ~133 MB.   All freed on return.
// ============================================================================

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#include "AlgoControls.hpp"
#include "AFMedianFilterEnum.hpp"
#include "AFMedian_GPU.hpp"


// ----------------------------------------------------------------------------
namespace {

constexpr int   kBlockX     = 32;
constexpr int   kBlockY     = 16;
constexpr int   kMaxRadius  = 8;
constexpr int   kTileX      = kBlockX + 2 * kMaxRadius;   // 48
constexpr int   kTileY      = kBlockY + 2 * kMaxRadius;   // 32

constexpr float kFromUnit   = 255.0f;            // [0,1] -> [0,255]
constexpr float kToUnit     = 1.0f / 255.0f;     // [0,255] -> [0,1]


// ----------------------------------------------------------------------------
// Edge-clamped read from a single-channel planar buffer.
// All planar buffers are in [0,255] internal scale.
// ----------------------------------------------------------------------------
__device__ __forceinline__
float load_clamped_planar(const float* RESTRICT in,
                          int x, int y, int W, int H, int pitch)
{
    const int cx = max(0, min(x, W - 1));
    const int cy = max(0, min(y, H - 1));
    return in[static_cast<size_t>(cy) * pitch + cx];
}

// ----------------------------------------------------------------------------
// Insertion sort -- per-thread local buffer.  Caller supplies the size n,
// which is a compile-time constant inside the templated AFMF kernel.
// ----------------------------------------------------------------------------
__device__ __forceinline__
void sort_window(float* RESTRICT a, int n)
{
    for (int i = 1; i < n; ++i)
    {
        const float key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key)
        {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

// ----------------------------------------------------------------------------
// Definition-7 frequency median over a SORTED window.
// SIEM indices:  leftMid = (m-1)/2,  rightMid = m/2
// Tie-break:     freq(left) <= freq(right) -> return right; else left.
// ----------------------------------------------------------------------------
__device__ __forceinline__
float frequency_median_def7(const float* RESTRICT sorted, int n)
{
    if (n <= 0) return 0.0f;
    if (n == 1) return sorted[0];

    int m = 1;
    for (int i = 1; i < n; ++i)
    {
        if (sorted[i] != sorted[i - 1]) ++m;
    }

    const int leftMid  = (m - 1) / 2;
    const int rightMid =  m      / 2;

    int   siem_idx = 0;
    float u_left   = 0.0f, u_right = 0.0f;
    int   f_left   = 0,    f_right = 0;

    int i = 0;
    while (i < n)
    {
        const float v = sorted[i];
        int f = 1;
        while (i + f < n && sorted[i + f] == v) ++f;

        if (siem_idx == leftMid)  { u_left  = v; f_left  = f; }
        if (siem_idx == rightMid) { u_right = v; f_right = f; break; }

        ++siem_idx;
        i += f;
    }

    if (leftMid == rightMid) return u_left;
    return (f_left <= f_right) ? u_right : u_left;
}


// ---------------------------------------------------------------------------
// 9-element optimal sorting network (25 compare-swaps, depth 7) on scalars
// passed by reference.  After __forceinline__ expansion the references alias
// the caller's locals, every index is a literal constant, and the 9 values
// stay in registers throughout the sort -- the local-memory traffic that
// dominates the pointer-based insertion sort is gone on this path.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void sort_network_9_f32(float& a0, float& a1, float& a2, float& a3, float& a4,
                        float& a5, float& a6, float& a7, float& a8)
{
    #define CSWAP_F(x, y) do {                          \
        const float lo__ = fminf((x), (y));             \
        const float hi__ = fmaxf((x), (y));             \
        (x) = lo__;                                     \
        (y) = hi__;                                     \
    } while (0)

    CSWAP_F(a0, a3); CSWAP_F(a1, a7); CSWAP_F(a2, a5); CSWAP_F(a4, a8);
    CSWAP_F(a0, a7); CSWAP_F(a2, a4); CSWAP_F(a3, a8); CSWAP_F(a5, a6);
    CSWAP_F(a0, a2); CSWAP_F(a1, a3); CSWAP_F(a4, a5); CSWAP_F(a7, a8);
    CSWAP_F(a1, a4); CSWAP_F(a3, a6); CSWAP_F(a5, a7);
    CSWAP_F(a0, a1); CSWAP_F(a2, a4); CSWAP_F(a3, a5); CSWAP_F(a6, a8);
    CSWAP_F(a2, a3); CSWAP_F(a4, a5); CSWAP_F(a6, a7);
    CSWAP_F(a1, a2); CSWAP_F(a3, a4); CSWAP_F(a5, a6);

    #undef CSWAP_F
}


// ---------------------------------------------------------------------------
// Frequency-median (def 7) for 9 pre-sorted scalars.  Mirrors the pointer-
// based frequency_median_def7 algorithm but works entirely on register-
// resident values.  Used on the rare "corrupted" sub-path of R = 1.
// ---------------------------------------------------------------------------
__device__ __forceinline__
float freq_median_9_f32(float s0, float s1, float s2, float s3, float s4,
                        float s5, float s6, float s7, float s8)
{
    // Distinct flags: 1 if this value differs from the previous one.
    const int d1 = (s1 != s0) ? 1 : 0;
    const int d2 = (s2 != s1) ? 1 : 0;
    const int d3 = (s3 != s2) ? 1 : 0;
    const int d4 = (s4 != s3) ? 1 : 0;
    const int d5 = (s5 != s4) ? 1 : 0;
    const int d6 = (s6 != s5) ? 1 : 0;
    const int d7 = (s7 != s6) ? 1 : 0;
    const int d8 = (s8 != s7) ? 1 : 0;

    // Cumulative unique-index per position (0..m-1).
    const int u0 = 0;
    const int u1 = u0 + d1;
    const int u2 = u1 + d2;
    const int u3 = u2 + d3;
    const int u4 = u3 + d4;
    const int u5 = u4 + d5;
    const int u6 = u5 + d6;
    const int u7 = u6 + d7;
    const int u8 = u7 + d8;

    const int m        = u8 + 1;          // 1..9 distinct values
    const int leftMid  = (m - 1) >> 1;
    const int rightMid =  m      >> 1;

    // Value at the leftMid-th unique run -- any position with u == leftMid
    // holds it, since they share the same value; the ternary chain compiles
    // to a chain of select instructions on Turing.
    const float u_left =
        (u0 == leftMid) ? s0 :
        (u1 == leftMid) ? s1 :
        (u2 == leftMid) ? s2 :
        (u3 == leftMid) ? s3 :
        (u4 == leftMid) ? s4 :
        (u5 == leftMid) ? s5 :
        (u6 == leftMid) ? s6 :
        (u7 == leftMid) ? s7 : s8;

    const float u_right =
        (u0 == rightMid) ? s0 :
        (u1 == rightMid) ? s1 :
        (u2 == rightMid) ? s2 :
        (u3 == rightMid) ? s3 :
        (u4 == rightMid) ? s4 :
        (u5 == rightMid) ? s5 :
        (u6 == rightMid) ? s6 :
        (u7 == rightMid) ? s7 : s8;

    // Frequency = number of positions sharing that unique index.
    const int f_left =
          (u0 == leftMid) + (u1 == leftMid) + (u2 == leftMid)
        + (u3 == leftMid) + (u4 == leftMid) + (u5 == leftMid)
        + (u6 == leftMid) + (u7 == leftMid) + (u8 == leftMid);

    const int f_right =
          (u0 == rightMid) + (u1 == rightMid) + (u2 == rightMid)
        + (u3 == rightMid) + (u4 == rightMid) + (u5 == rightMid)
        + (u6 == rightMid) + (u7 == rightMid) + (u8 == rightMid);

    if (leftMid == rightMid) return u_left;
    return (f_left <= f_right) ? u_right : u_left;
}


// ============================================================================
//  AFMF kernel - operates on a SINGLE-CHANNEL planar buffer in [0,255].
//  Identical algorithm to the CPU/AVX2 pipeline.
//
//  Templated on radius R (1..8) so that:
//   - the per-thread `window` array is sized to (2R+1)^2 at compile time,
//     so for the common small-radius cases (R = 1..3) the stack frame is
//     small enough to keep nvcc from reserving 1156 B/thread up front;
//   - the slow-path r-loop has compile-time bounds and the compiler can
//     fully unroll it for R = 1, 2, 3 and partially for the rest.
//
//  Note: nvcc still places window[] in local memory because the sort/median
//  helpers access it with a runtime, data-dependent index.  Getting window[]
//  into registers requires a fixed-shape sort (sorting network) with all
//  compile-time indices -- a follow-up change for the R=1 path.
// ============================================================================
template<int R>
__launch_bounds__(kBlockX * kBlockY, 2)
__global__ void afmf_kernel_planar_fp32_r(
    const float* RESTRICT in_plane,
          float* RESTRICT out_plane,
    int W, int H,
    int srcPitch, int dstPitch,
    float tol)
{
    constexpr int kWinSize = (2 * R + 1) * (2 * R + 1);

    __shared__ float tile[kTileY][kTileX];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x  * kBlockX;
    const int by = blockIdx.y  * kBlockY;

    // Cooperative shared-memory tile load with edge clamping.
    for (int dy = ty; dy < kTileY; dy += kBlockY)
    {
        for (int dx = tx; dx < kTileX; dx += kBlockX)
        {
            const int gx = bx + dx - kMaxRadius;
            const int gy = by + dy - kMaxRadius;
            tile[dy][dx] = load_clamped_planar(in_plane, gx, gy, W, H, srcPitch);
        }
    }
    __syncthreads();

    const int ox = bx + tx;
    const int oy = by + ty;
    if (ox >= W || oy >= H) return;

    const int   cy     = ty + kMaxRadius;
    const int   cx     = tx + kMaxRadius;
    const float center = tile[cy][cx];

    // -----------------------------------------------------------------------
    //  R = 1 specialization: 9 scalar registers across fast + slow paths.
    //  No window[] array, no pointer-passing, no local-memory traffic on the
    //  sort.  The compiler DCEs this block for R >= 2 (R is a template arg).
    // -----------------------------------------------------------------------
    if (R == 1)
    {
        float w0 = tile[cy - 1][cx - 1];
        float w1 = tile[cy - 1][cx    ];
        float w2 = tile[cy - 1][cx + 1];
        float w3 = tile[cy    ][cx - 1];
        float w4 = tile[cy    ][cx    ];   // == center
        float w5 = tile[cy    ][cx + 1];
        float w6 = tile[cy + 1][cx - 1];
        float w7 = tile[cy + 1][cx    ];
        float w8 = tile[cy + 1][cx + 1];

        // 3x3 fast path: min/max over the 9 scalars.
        const float min1 = fminf(fminf(fminf(w0, w1), fminf(w2, w3)),
                                 fminf(fminf(w4, w5), fminf(fminf(w6, w7), w8)));
        const float max1 = fmaxf(fmaxf(fmaxf(w0, w1), fmaxf(w2, w3)),
                                 fmaxf(fmaxf(w4, w5), fmaxf(fmaxf(w6, w7), w8)));

        if (center > min1 + tol && center < max1 - tol)
        {
            out_plane[static_cast<size_t>(oy) * dstPitch + ox] = center;
            return;
        }

        // Slow path: sort the same 9 scalars in place (sorting network).
        sort_network_9_f32(w0, w1, w2, w3, w4, w5, w6, w7, w8);

        const float wMin = w0;
        const float wMed = w4;
        const float wMax = w8;

        float result;
        if (wMed > wMin && wMed < wMax)
        {
            const bool corrupted = (center <= wMin + tol)
                                || (center >= wMax - tol);
            result = corrupted
                   ? freq_median_9_f32(w0, w1, w2, w3, w4, w5, w6, w7, w8)
                   : center;
        }
        else
        {
            result = center;
        }

        out_plane[static_cast<size_t>(oy) * dstPitch + ox] = result;
        return;
    }

    // ---- 3x3 fast path (R >= 2 path) --------------------------------------
    float min1 = center;
    float max1 = center;

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy)
    {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx)
        {
            const float v = tile[cy + dy][cx + dx];
            min1 = fminf(min1, v);
            max1 = fmaxf(max1, v);
        }
    }

    if (center > min1 + tol && center < max1 - tol)
    {
        out_plane[static_cast<size_t>(oy) * dstPitch + ox] = center;
        return;
    }

    // ---- Adaptive-radius slow path ----------------------------------------
    // window[] is compile-time sized; for R <= 3 it stays in registers.
    float window[kWinSize];
    bool  found   = false;
    int   bestN   = 0;
    float bestMin = 0.0f;
    float bestMax = 0.0f;

    for (int r = 1; r <= R; ++r)
    {
        const int n = (2 * r + 1) * (2 * r + 1);

        int k = 0;
        for (int dy = -r; dy <= r; ++dy)
        {
            for (int dx = -r; dx <= r; ++dx)
            {
                window[k++] = tile[cy + dy][cx + dx];
            }
        }

        sort_window(window, n);

        const float wMin = window[0];
        const float wMax = window[n - 1];
        const float wMed = window[n / 2];

        if (wMed > wMin && wMed < wMax)
        {
            found   = true;
            bestN   = n;
            bestMin = wMin;
            bestMax = wMax;
            break;
        }
    }

    // ---- Final corruption check + replacement -----------------------------
    float result;
    if (found)
    {
        const bool corrupted = (center <= bestMin + tol)
                            || (center >= bestMax - tol);
        result = corrupted ? frequency_median_def7(window, bestN) : center;
    }
    else
    {
        result = center;
    }

    out_plane[static_cast<size_t>(oy) * dstPitch + ox] = result;
}


// ----------------------------------------------------------------------------
// Host-side dispatcher: pick the right template instantiation based on the
// runtime radius (1..8, clamped by AlgoControls::Sanitize).
// ----------------------------------------------------------------------------
static inline void launch_afmf_fp32(
    int radius,
    const float*  in,
          float*  out,
    int W, int H,
    int srcPitch, int dstPitch,
    float tol,
    dim3 grid, dim3 block, cudaStream_t stream)
{
    switch (radius)
    {
        case 1:  afmf_kernel_planar_fp32_r<1><<<grid, block, 0, stream>>>(in, out, W, H, srcPitch, dstPitch, tol); break;
        case 2:  afmf_kernel_planar_fp32_r<2><<<grid, block, 0, stream>>>(in, out, W, H, srcPitch, dstPitch, tol); break;
        case 3:  afmf_kernel_planar_fp32_r<3><<<grid, block, 0, stream>>>(in, out, W, H, srcPitch, dstPitch, tol); break;
        case 4:  afmf_kernel_planar_fp32_r<4><<<grid, block, 0, stream>>>(in, out, W, H, srcPitch, dstPitch, tol); break;
        case 5:  afmf_kernel_planar_fp32_r<5><<<grid, block, 0, stream>>>(in, out, W, H, srcPitch, dstPitch, tol); break;
        case 6:  afmf_kernel_planar_fp32_r<6><<<grid, block, 0, stream>>>(in, out, W, H, srcPitch, dstPitch, tol); break;
        case 7:  afmf_kernel_planar_fp32_r<7><<<grid, block, 0, stream>>>(in, out, W, H, srcPitch, dstPitch, tol); break;
        default: afmf_kernel_planar_fp32_r<8><<<grid, block, 0, stream>>>(in, out, W, H, srcPitch, dstPitch, tol); break;
    }
}


// ============================================================================
//  Split BGRA[0,1] interleaved -> 3 single-channel planes [0,255]
// ============================================================================
__global__ void split_bgra_to_planar_fp32(
    const float* RESTRICT bgra_in,         // BGRA interleaved [0,1]
          float* RESTRICT plane_B,         // [0,255]
          float* RESTRICT plane_G,
          float* RESTRICT plane_R,
    int W, int H,
    int bgraPitchPixels,                   // pixels per row of BGRA
    int planarPitch)                       // floats per row of planar
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const size_t bgraIdx   = (static_cast<size_t>(y) * bgraPitchPixels + x) * 4u;
    const size_t planarIdx =  static_cast<size_t>(y) * planarPitch + x;

    plane_B[planarIdx] = bgra_in[bgraIdx + 0] * kFromUnit;
    plane_G[planarIdx] = bgra_in[bgraIdx + 1] * kFromUnit;
    plane_R[planarIdx] = bgra_in[bgraIdx + 2] * kFromUnit;
    // Alpha is handled at merge time, read from original.
}


// ============================================================================
//  Merge 3 planes [0,255] -> BGRA[0,1] interleaved, alpha pass-through.
// ============================================================================
__global__ void merge_planar_to_bgra_fp32(
    const float* RESTRICT plane_B,
    const float* RESTRICT plane_G,
    const float* RESTRICT plane_R,
    const float* RESTRICT bgra_orig,       // original BGRA in [0,1] for alpha
          float* RESTRICT bgra_out,        // output BGRA in [0,1]
    int W, int H,
    int planarPitch,
    int srcBgraPitchPixels,
    int dstBgraPitchPixels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const size_t srcIdx    = (static_cast<size_t>(y) * srcBgraPitchPixels + x) * 4u;
    const size_t dstIdx    = (static_cast<size_t>(y) * dstBgraPitchPixels + x) * 4u;
    const size_t planarIdx =  static_cast<size_t>(y) * planarPitch + x;

    float b = plane_B[planarIdx] * kToUnit;
    float g = plane_G[planarIdx] * kToUnit;
    float r = plane_R[planarIdx] * kToUnit;

    // Defensive clamp -- the algorithm preserves values in [0,255] but
    // numerical drift could nudge a hair outside.
    b = fmaxf(0.0f, fminf(1.0f, b));
    g = fmaxf(0.0f, fminf(1.0f, g));
    r = fmaxf(0.0f, fminf(1.0f, r));

    bgra_out[dstIdx + 0] = b;
    bgra_out[dstIdx + 1] = g;
    bgra_out[dstIdx + 2] = r;
    bgra_out[dstIdx + 3] = bgra_orig[srcIdx + 3];      // alpha pass-through
}


// ============================================================================
//  Merge with centered-residual noise-map post-processing.
//  Per-channel:   out = clamp((orig_255 - filt_255) + 128, 0, 255) / 255
// ============================================================================
__global__ void merge_noise_map_to_bgra_fp32(
    const float* RESTRICT plane_B,         // filtered B in [0,255]
    const float* RESTRICT plane_G,
    const float* RESTRICT plane_R,
    const float* RESTRICT bgra_orig,       // original BGRA in [0,1]
          float* RESTRICT bgra_out,        // output BGRA in [0,1]
    int W, int H,
    int planarPitch,
    int srcBgraPitchPixels,
    int dstBgraPitchPixels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const size_t srcIdx    = (static_cast<size_t>(y) * srcBgraPitchPixels + x) * 4u;
    const size_t dstIdx    = (static_cast<size_t>(y) * dstBgraPitchPixels + x) * 4u;
    const size_t planarIdx =  static_cast<size_t>(y) * planarPitch + x;

    // Originals back to [0,255] for parity with the CPU code
    const float origB_255 = bgra_orig[srcIdx + 0] * kFromUnit;
    const float origG_255 = bgra_orig[srcIdx + 1] * kFromUnit;
    const float origR_255 = bgra_orig[srcIdx + 2] * kFromUnit;

    float dB = (origB_255 - plane_B[planarIdx]) + 128.0f;
    float dG = (origG_255 - plane_G[planarIdx]) + 128.0f;
    float dR = (origR_255 - plane_R[planarIdx]) + 128.0f;

    dB = fmaxf(0.0f, fminf(255.0f, dB)) * kToUnit;
    dG = fmaxf(0.0f, fminf(255.0f, dG)) * kToUnit;
    dR = fmaxf(0.0f, fminf(255.0f, dR)) * kToUnit;

    bgra_out[dstIdx + 0] = dB;
    bgra_out[dstIdx + 1] = dG;
    bgra_out[dstIdx + 2] = dR;
    bgra_out[dstIdx + 3] = bgra_orig[srcIdx + 3];      // alpha pass-through
}

} // anonymous namespace

  // ============================================================================
  //  ImageLabAFMF32_CUDA  --  host-side aggregator
  // ============================================================================
CUDA_KERNEL_CALL
void ImageLabAFMF32_CUDA(const float*  RESTRICT inBuffer,
    float*  RESTRICT outBuffer,
    int                    srcPitch,    // BGRA pixels/row
    int                    dstPitch,    // BGRA pixels/row
    int                    width,
    int                    height,
    const AlgoControls* RESTRICT algoGpuParams,
    int                    frameCount,
    cudaStream_t           stream)
{
    // ------------------------------------------------------------------------
    // 1. Pull controls into host memory.
    //
    //    cudaMemcpyDefault uses UVA to detect the actual memory type of
    //    algoGpuParams (device / host / managed) and routes accordingly.
    //    This is required because the host caller is observed to pass a
    //    pointer to a stack-allocated AlgoControls (host memory), e.g.:
    //        const AlgoControls algoParams = getAlgoControlsDefault();
    //        ImageLabAFMF32_CUDA(..., &algoParams, ...);
    //    A fixed cudaMemcpyDeviceToHost direction would error out here
    //    and put the CUDA context into a sticky error state that causes
    //    every subsequent kernel launch on this thread to fail silently.
    // ------------------------------------------------------------------------
    AlgoControls ctrl{};
    cudaMemcpyAsync(&ctrl, algoGpuParams, sizeof(AlgoControls),
        cudaMemcpyDefault, stream);
    cudaStreamSynchronize(stream);
    ctrl.Sanitize();

    const int   radius = ctrl.radius;
    const float tolerance = ctrl.tolerance;       // already in [0,255] scale
    const int   iterations = ctrl.iterations;
    const bool  wantNoise =
        (ctrl.outputType == AFMF_Output::AFMF_OUTPUT_NOISE_MAP);

    // ------------------------------------------------------------------------
    // 2. Defensive frame count.
    //
    //    The host caller is observed to pass frameCount = 0 (e.g.:
    //        const int frameCount = 0;
    //        ImageLabAFMF32_CUDA(..., frameCount, stream);
    //    ), which with a strict `for (f = 0; f < frameCount; ++f)` loop
    //    means *no kernels run at all* and the output buffer is left
    //    untouched.  Treating any non-positive value as a request to
    //    process a single frame keeps the function useful for that
    //    common single-frame call site.
    // ------------------------------------------------------------------------
    const int effFrameCount = (frameCount > 0) ? frameCount : 1;

    // ------------------------------------------------------------------------
    // 3. Grid setup
    // ------------------------------------------------------------------------
    const dim3 block(kBlockX, kBlockY, 1);
    const dim3 grid((width + kBlockX - 1) / kBlockX,
        (height + kBlockY - 1) / kBlockY,
        1);

    // ------------------------------------------------------------------------
    // 4. Allocate internal planar work buffers from a single arena.
    //
    //    LAYOUT inside the arena (each slot 256-byte aligned):
    //      offset 0 * slot :  plane_B   (channel B working plane)
    //      offset 1 * slot :  plane_G   (channel G working plane)
    //      offset 2 * slot :  plane_R   (channel R working plane)
    //      offset 3 * slot :  scratch   (shared ping-pong scratch)
    //
    //    Each slot is rounded up to kSubBufAlign so every sub-buffer base
    //    address sits on a 256-byte boundary -- matches the alignment
    //    cudaMalloc itself returns and keeps coalesced 128-byte global
    //    loads happy on Turing/Pascal.
    //
    //    Single cudaMalloc / single cudaFree.
    // ------------------------------------------------------------------------
    constexpr size_t kSubBufAlign = 256u;        // bytes per slot boundary
    constexpr int    kSubBufCount = 4;           // 3 planes + 1 scratch

    const int    planarPitch = width;
    const size_t planeBytes = static_cast<size_t>(planarPitch)
        * static_cast<size_t>(height)
        * sizeof(float);

    const size_t slotBytes =
        (planeBytes + kSubBufAlign - 1u) & ~(kSubBufAlign - 1u);
    const size_t arenaBytes = slotBytes * static_cast<size_t>(kSubBufCount);

    uint8_t* arena = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&arena), arenaBytes) != cudaSuccess
        || arena == nullptr)
    {
        // Allocation failure: leave outBuffer untouched and return.
        return;
    }

    float* const plane_B = reinterpret_cast<float*>(arena + 0u * slotBytes);
    float* const plane_G = reinterpret_cast<float*>(arena + 1u * slotBytes);
    float* const plane_R = reinterpret_cast<float*>(arena + 2u * slotBytes);
    float* const scratch = reinterpret_cast<float*>(arena + 3u * slotBytes);

    // ------------------------------------------------------------------------
    // 5. Process each frame.  inBuffer / outBuffer are device pointers
    //    (the caller allocates them via cudaMalloc and pre-fills inBuffer
    //    via cudaMemcpyHostToDevice).  The kernels read/write them directly.
    // ------------------------------------------------------------------------
    const size_t bgraInFloatsPerFrame =
        static_cast<size_t>(height) * srcPitch * 4u;
    const size_t bgraOutFloatsPerFrame =
        static_cast<size_t>(height) * dstPitch * 4u;

    for (int f = 0; f < effFrameCount; ++f)
    {
        const float* frameIn = inBuffer + f * bgraInFloatsPerFrame;
        float* frameOut = outBuffer + f * bgraOutFloatsPerFrame;

        // 5a. Split BGRA[0,1] -> 3 planes [0,255]
        split_bgra_to_planar_fp32 << <grid, block, 0, stream >> >(
            frameIn, plane_B, plane_G, plane_R,
            width, height, srcPitch, planarPitch);

        // 5b. Process each of the three colour channels
        float* channels[3] = { plane_B, plane_G, plane_R };

        for (int c = 0; c < 3; ++c)
        {
            float* channelPlane = channels[c];

            // Ping-pong between channelPlane and the shared scratch.
            const float* src = channelPlane;
            float* dst = scratch;

            for (int it = 0; it < iterations; ++it)
            {
                launch_afmf_fp32(
                    radius,
                    src, dst,
                    width, height,
                    planarPitch, planarPitch,
                    tolerance,
                    grid, block, stream);

                const float* prevSrc = src;
                src = dst;
                dst = const_cast<float*>(prevSrc);
            }

            // If the final write landed in scratch, copy back into the
            // channel's main plane (next channel reuses scratch, and merge
            // reads from plane_B/G/R).
            if (src == scratch)
            {
                cudaMemcpyAsync(channelPlane, scratch, planeBytes,
                    cudaMemcpyDeviceToDevice, stream);
            }
        }

        // 5c. Merge 3 planes -> BGRA[0,1] (or centered noise map)
        if (wantNoise)
        {
            merge_noise_map_to_bgra_fp32 << <grid, block, 0, stream >> >(
                plane_B, plane_G, plane_R,
                frameIn, frameOut,
                width, height,
                planarPitch,
                srcPitch, dstPitch);
        }
        else
        {
            merge_planar_to_bgra_fp32 << <grid, block, 0, stream >> >(
                plane_B, plane_G, plane_R,
                frameIn, frameOut,
                width, height,
                planarPitch,
                srcPitch, dstPitch);
        }
    }

    cudaDeviceSynchronize();

    if (arena != nullptr)
        cudaFree(arena);

    return;
}

// ============================================================================
// CLEANUP 
// ============================================================================
CUDA_KERNEL_CALL
void ImageLabDenoise_CleanupGPUF32()
{
}

