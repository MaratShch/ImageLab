// ============================================================================
//  AFMF_FP16_CUDA.cu
//
//  Adaptive Frequency Median Filter with native FP16 internal computation.
//  Target: sm_61 (Pascal) and up.  CUDA 10.2 compatible.
//
//  I/O CONTRACT  (identical to AFMF_FP32_CUDA.cu):
//    Input  : BGRA_32f interleaved, channels in [0.0, 1.0]
//    Output : BGRA_32f interleaved, channels in [0.0, 1.0]
//    Pitch  : in pixels (BGRA pixels per row)
//
//  PRECISION:
//    Internal AFMF compute is __half throughout:
//      - shared-memory tile, per-thread window buffer, sort, frequency median,
//        fast-path min/max, and all comparisons run in FP16.
//    Range scaling [0,1] <-> [0,255] runs in FP32 at the boundary kernels
//      to keep a single rounding step per direction.
//    Noise-map merge runs in FP32 (originals are FP32, no benefit to forcing
//      them through __half just to subtract).
//
//    Pascal note: FP16 arithmetic runs at 1/64 FP32 throughput on GP106.
//    This kernel is the precision-comparison tool; on Pascal hardware the
//    FP32 path is faster for production use.
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>

#include "AlgoControls.hpp"
#include "AFMedianFilterEnum.hpp"
#include "AFMedian_GPU.hpp"


namespace FP16 {

constexpr int   kBlockX     = 32;
constexpr int   kBlockY     = 16;
constexpr int   kMaxRadius  = kernelRadiusMax;
constexpr int   kTileX      = kBlockX + 2 * kMaxRadius;   // 48
constexpr int   kTileY      = kBlockY + 2 * kMaxRadius;   // 32
constexpr int   kMaxWindow  = (2 * kMaxRadius + 1) * (2 * kMaxRadius + 1);   // 289

constexpr float kFromUnit   = 255.0f;
constexpr float kToUnit     = 1.0f / 255.0f;


// ----------------------------------------------------------------------------
// __half min / max via predicated select.
// CUDA 10.2 doesn't guarantee __hmin / __hmax for plain (non-vector) __half;
// the ternary compiles to a single predicated move. Image data is non-NaN
// by construction so NaN-aware variants aren't needed.
// ----------------------------------------------------------------------------
__device__ __forceinline__ __half hmin_h(__half a, __half b)
{ return __hlt(a, b) ? a : b; }

__device__ __forceinline__ __half hmax_h(__half a, __half b)
{ return __hgt(a, b) ? a : b; }


// ----------------------------------------------------------------------------
// 9-input min/max reduction.
//
// On CUDA 11.0+ this uses __hmax2 / __hmin2 (packed __half2 intrinsics): the
// 9 values are arranged as four __half2 pairs plus w8, reduced through 3
// packed ops + 2 scalar ops instead of 8 scalar ops -- reduced instruction
// issue pressure on Turing/Volta (where max.f16x2 is emulated) and a direct
// hardware speedup on sm_80+ (where max.f16x2 is native).
//
// On CUDA 10.x the packed intrinsics don't exist, and a manual emulation
// (unpack + scalar compare + repack) is no faster than the scalar tree, so
// we just emit the scalar tree directly and keep the old codepath bit-exact.
// ----------------------------------------------------------------------------
__device__ __forceinline__
__half min9_h(__half w0, __half w1, __half w2, __half w3, __half w4,
              __half w5, __half w6, __half w7, __half w8)
{
#if __CUDACC_VER_MAJOR__ >= 11
    const __half2 p01 = __halves2half2(w0, w1);
    const __half2 p23 = __halves2half2(w2, w3);
    const __half2 p45 = __halves2half2(w4, w5);
    const __half2 p67 = __halves2half2(w6, w7);

    const __half2 m0123 = __hmin2(p01, p23);
    const __half2 m4567 = __hmin2(p45, p67);
    const __half2 m_all = __hmin2(m0123, m4567);

    return hmin_h(hmin_h(__low2half(m_all), __high2half(m_all)), w8);
#else
    return hmin_h(hmin_h(hmin_h(w0, w1), hmin_h(w2, w3)),
                  hmin_h(hmin_h(w4, w5), hmin_h(hmin_h(w6, w7), w8)));
#endif
}

__device__ __forceinline__
__half max9_h(__half w0, __half w1, __half w2, __half w3, __half w4,
              __half w5, __half w6, __half w7, __half w8)
{
#if __CUDACC_VER_MAJOR__ >= 11
    const __half2 p01 = __halves2half2(w0, w1);
    const __half2 p23 = __halves2half2(w2, w3);
    const __half2 p45 = __halves2half2(w4, w5);
    const __half2 p67 = __halves2half2(w6, w7);

    const __half2 m0123 = __hmax2(p01, p23);
    const __half2 m4567 = __hmax2(p45, p67);
    const __half2 m_all = __hmax2(m0123, m4567);

    return hmax_h(hmax_h(__low2half(m_all), __high2half(m_all)), w8);
#else
    return hmax_h(hmax_h(hmax_h(w0, w1), hmax_h(w2, w3)),
                  hmax_h(hmax_h(w4, w5), hmax_h(hmax_h(w6, w7), w8)));
#endif
}


// ----------------------------------------------------------------------------
__device__ __forceinline__
__half load_clamped_planar_h(const __half* RESTRICT in,
                             int x, int y, int W, int H, int pitch)
{
    const int cx = max(0, min(x, W - 1));
    const int cy = max(0, min(y, H - 1));
    return in[static_cast<size_t>(cy) * pitch + cx];
}


// ----------------------------------------------------------------------------
// Insertion sort over __half values.
// ----------------------------------------------------------------------------
__device__ __forceinline__
void sort_window_h(__half* RESTRICT a, int n)
{
    for (int i = 1; i < n; ++i)
    {
        const __half key = a[i];
        int j = i - 1;
        while (j >= 0 && __hgt(a[j], key))
        {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}


// ----------------------------------------------------------------------------
// Definition-7 frequency median over a SORTED __half window.
// FP16 quantisation (~0.25 in upper half of [0,255]) may collapse values that
// FP32 would consider distinct -- this is algorithmically benign for
// noise-frequency analysis at this precision.
// ----------------------------------------------------------------------------
__device__ __forceinline__
__half frequency_median_def7_h(const __half* RESTRICT sorted, int n)
{
    const __half kZero = __float2half(0.0f);
    if (n <= 0) return kZero;
    if (n == 1) return sorted[0];

    int m = 1;
    for (int i = 1; i < n; ++i)
    {
        if (__hne(sorted[i], sorted[i - 1])) ++m;
    }

    const int leftMid  = (m - 1) / 2;
    const int rightMid =  m      / 2;

    int    siem_idx = 0;
    __half u_left   = kZero, u_right = kZero;
    int    f_left   = 0,     f_right = 0;

    int i = 0;
    while (i < n)
    {
        const __half v = sorted[i];
        int f = 1;
        while (i + f < n && __heq(sorted[i + f], v)) ++f;

        if (siem_idx == leftMid)  { u_left  = v; f_left  = f; }
        if (siem_idx == rightMid) { u_right = v; f_right = f; break; }

        ++siem_idx;
        i += f;
    }

    if (leftMid == rightMid) return u_left;
    return (f_left <= f_right) ? u_right : u_left;
}


// ---------------------------------------------------------------------------
// 9-element optimal sorting network on __half scalars (25 compare-swaps).
// References are aliased to the caller's locals after inlining, so every
// index is compile-time and the values stay in registers across the sort.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void sort_network_9_h(__half& a0, __half& a1, __half& a2, __half& a3, __half& a4,
                      __half& a5, __half& a6, __half& a7, __half& a8)
{
    #define CSWAP_H(x, y) do {                          \
        const __half lo__ = hmin_h((x), (y));           \
        const __half hi__ = hmax_h((x), (y));           \
        (x) = lo__;                                     \
        (y) = hi__;                                     \
    } while (0)

    CSWAP_H(a0, a3); CSWAP_H(a1, a7); CSWAP_H(a2, a5); CSWAP_H(a4, a8);
    CSWAP_H(a0, a7); CSWAP_H(a2, a4); CSWAP_H(a3, a8); CSWAP_H(a5, a6);
    CSWAP_H(a0, a2); CSWAP_H(a1, a3); CSWAP_H(a4, a5); CSWAP_H(a7, a8);
    CSWAP_H(a1, a4); CSWAP_H(a3, a6); CSWAP_H(a5, a7);
    CSWAP_H(a0, a1); CSWAP_H(a2, a4); CSWAP_H(a3, a5); CSWAP_H(a6, a8);
    CSWAP_H(a2, a3); CSWAP_H(a4, a5); CSWAP_H(a6, a7);
    CSWAP_H(a1, a2); CSWAP_H(a3, a4); CSWAP_H(a5, a6);

    #undef CSWAP_H
}


// ---------------------------------------------------------------------------
// Frequency-median (def 7) for 9 pre-sorted __half scalars.  All ops on
// register-resident values; no pointer, no array.
// ---------------------------------------------------------------------------
__device__ __forceinline__
__half freq_median_9_h(__half s0, __half s1, __half s2, __half s3, __half s4,
                       __half s5, __half s6, __half s7, __half s8)
{
    // Distinct flags using __half not-equal.
    const int d1 = __hne(s1, s0) ? 1 : 0;
    const int d2 = __hne(s2, s1) ? 1 : 0;
    const int d3 = __hne(s3, s2) ? 1 : 0;
    const int d4 = __hne(s4, s3) ? 1 : 0;
    const int d5 = __hne(s5, s4) ? 1 : 0;
    const int d6 = __hne(s6, s5) ? 1 : 0;
    const int d7 = __hne(s7, s6) ? 1 : 0;
    const int d8 = __hne(s8, s7) ? 1 : 0;

    const int u0 = 0;
    const int u1 = u0 + d1;
    const int u2 = u1 + d2;
    const int u3 = u2 + d3;
    const int u4 = u3 + d4;
    const int u5 = u4 + d5;
    const int u6 = u5 + d6;
    const int u7 = u6 + d7;
    const int u8 = u7 + d8;

    const int m        = u8 + 1;
    const int leftMid  = (m - 1) >> 1;
    const int rightMid =  m      >> 1;

    const __half u_left =
        (u0 == leftMid) ? s0 :
        (u1 == leftMid) ? s1 :
        (u2 == leftMid) ? s2 :
        (u3 == leftMid) ? s3 :
        (u4 == leftMid) ? s4 :
        (u5 == leftMid) ? s5 :
        (u6 == leftMid) ? s6 :
        (u7 == leftMid) ? s7 : s8;

    const __half u_right =
        (u0 == rightMid) ? s0 :
        (u1 == rightMid) ? s1 :
        (u2 == rightMid) ? s2 :
        (u3 == rightMid) ? s3 :
        (u4 == rightMid) ? s4 :
        (u5 == rightMid) ? s5 :
        (u6 == rightMid) ? s6 :
        (u7 == rightMid) ? s7 : s8;

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
//  AFMF kernel specialized for radius = 1.  The 9 window values stay in
//  scalar registers across both fast and slow paths; the sort is a fixed
//  25-comparator network with compile-time indices, and the corruption
//  fallback uses freq_median_9_h on the same 9 register values.  No window[]
//  array, no pointer-passing into the sort, no local-memory traffic on the
//  hot path -- which is the whole point of having this specialization split
//  out from the generic kernel below.
// ============================================================================
__launch_bounds__(kBlockX * kBlockY, 2)
__global__ void afmf_kernel_planar_fp16_r1(
    const __half* RESTRICT in_plane,
          __half* RESTRICT out_plane,
    int W, int H,
    int srcPitch, int dstPitch,
    __half tol)
{
    __shared__ __half tile[kTileY][kTileX];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x  * kBlockX;
    const int by = blockIdx.y  * kBlockY;

    for (int dy = ty; dy < kTileY; dy += kBlockY)
    {
        for (int dx = tx; dx < kTileX; dx += kBlockX)
        {
            const int gx = bx + dx - kMaxRadius;
            const int gy = by + dy - kMaxRadius;
            tile[dy][dx] = load_clamped_planar_h(in_plane, gx, gy, W, H, srcPitch);
        }
    }
    __syncthreads();

    const int ox = bx + tx;
    const int oy = by + ty;
    if (ox >= W || oy >= H) return;

    const int    cy     = ty + kMaxRadius;
    const int    cx     = tx + kMaxRadius;
    const __half center = tile[cy][cx];

    // Load the 3x3 neighborhood into 9 scalars that live in registers across
    // both fast and slow paths.
    __half w0 = tile[cy - 1][cx - 1];
    __half w1 = tile[cy - 1][cx    ];
    __half w2 = tile[cy - 1][cx + 1];
    __half w3 = tile[cy    ][cx - 1];
    __half w4 = tile[cy    ][cx    ];   // == center
    __half w5 = tile[cy    ][cx + 1];
    __half w6 = tile[cy + 1][cx - 1];
    __half w7 = tile[cy + 1][cx    ];
    __half w8 = tile[cy + 1][cx + 1];

    // 3x3 fast path: min/max reduction over the 9 scalars.
    // Uses __half2 packed intrinsics on CUDA 11+, scalar tree on CUDA 10.x.
    const __half min1 = min9_h(w0, w1, w2, w3, w4, w5, w6, w7, w8);
    const __half max1 = max9_h(w0, w1, w2, w3, w4, w5, w6, w7, w8);

    const __half min_p_tol = __hadd(min1, tol);
    const __half max_m_tol = __hsub(max1, tol);

    if (__hgt(center, min_p_tol) && __hlt(center, max_m_tol))
    {
        out_plane[static_cast<size_t>(oy) * dstPitch + ox] = center;
        return;
    }

    // Slow path: sort the 9 scalars in place.
    sort_network_9_h(w0, w1, w2, w3, w4, w5, w6, w7, w8);

    const __half wMin = w0;
    const __half wMed = w4;
    const __half wMax = w8;

    __half result;
    if (__hgt(wMed, wMin) && __hlt(wMed, wMax))
    {
        const __half wm_p_tol = __hadd(wMin, tol);
        const __half wM_m_tol = __hsub(wMax, tol);
        const bool   corrupted = __hle(center, wm_p_tol)
                              || __hge(center, wM_m_tol);
        result = corrupted
               ? freq_median_9_h(w0, w1, w2, w3, w4, w5, w6, w7, w8)
               : center;
    }
    else
    {
        result = center;
    }

    out_plane[static_cast<size_t>(oy) * dstPitch + ox] = result;
}


// ============================================================================
//  AFMF kernel - native __half compute on a single-channel plane [0,255].
//
//  Non-templated: a runtime `radius` parameter selects how many slow-path
//  rings get computed.  The window[] array stays in local memory regardless,
//  and on this codebase's __half path nvcc generates better code from the
//  unconditional runtime version than from a per-radius template
//  instantiation (an empirical result -- templating cost ~20% on RTX 2000
//  for FP16 even though it modestly helped FP32).
// ============================================================================
__launch_bounds__(kBlockX * kBlockY, 2)
__global__ void afmf_kernel_planar_fp16(
    const __half* RESTRICT in_plane,
          __half* RESTRICT out_plane,
    int W, int H,
    int srcPitch, int dstPitch,
    int radius, __half tol)
{
    __shared__ __half tile[kTileY][kTileX];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x  * kBlockX;
    const int by = blockIdx.y  * kBlockY;

    for (int dy = ty; dy < kTileY; dy += kBlockY)
    {
        for (int dx = tx; dx < kTileX; dx += kBlockX)
        {
            const int gx = bx + dx - kMaxRadius;
            const int gy = by + dy - kMaxRadius;
            tile[dy][dx] = load_clamped_planar_h(in_plane, gx, gy, W, H, srcPitch);
        }
    }
    __syncthreads();

    const int    ox     = bx + tx;
    const int    oy     = by + ty;
    if (ox >= W || oy >= H) return;

    const int    cy     = ty + kMaxRadius;
    const int    cx     = tx + kMaxRadius;
    const __half center = tile[cy][cx];

    // ---- 3x3 fast path -----------------------------------------------------
    // Load the 3x3 neighborhood into scalars so the reduction can go through
    // min9_h / max9_h (packed __half2 path on CUDA 11+).
    const __half w0_3 = tile[cy - 1][cx - 1];
    const __half w1_3 = tile[cy - 1][cx    ];
    const __half w2_3 = tile[cy - 1][cx + 1];
    const __half w3_3 = tile[cy    ][cx - 1];
    const __half w5_3 = tile[cy    ][cx + 1];
    const __half w6_3 = tile[cy + 1][cx - 1];
    const __half w7_3 = tile[cy + 1][cx    ];
    const __half w8_3 = tile[cy + 1][cx + 1];

    const __half min1 = min9_h(w0_3, w1_3, w2_3, w3_3, center,
                               w5_3, w6_3, w7_3, w8_3);
    const __half max1 = max9_h(w0_3, w1_3, w2_3, w3_3, center,
                               w5_3, w6_3, w7_3, w8_3);

    // (center > min1 + tol) && (center < max1 - tol)
    const __half min_p_tol = __hadd(min1, tol);
    const __half max_m_tol = __hsub(max1, tol);

    if (__hgt(center, min_p_tol) && __hlt(center, max_m_tol))
    {
        out_plane[static_cast<size_t>(oy) * dstPitch + ox] = center;
        return;
    }

    // ---- Adaptive-radius slow path ----------------------------------------
    __half window[kMaxWindow];
    bool   found   = false;
    int    bestN   = 0;
    __half bestMin = __float2half(0.0f);
    __half bestMax = __float2half(0.0f);

    for (int r = 1; r <= radius; ++r)
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

        sort_window_h(window, n);

        const __half wMin = window[0];
        const __half wMax = window[n - 1];
        const __half wMed = window[n / 2];

        if (__hgt(wMed, wMin) && __hlt(wMed, wMax))
        {
            found   = true;
            bestN   = n;
            bestMin = wMin;
            bestMax = wMax;
            break;
        }
    }

    // ---- Final corruption check + replacement -----------------------------
    __half result;
    if (found)
    {
        const __half bm_p_tol = __hadd(bestMin, tol);
        const __half bm_m_tol = __hsub(bestMax, tol);
        const bool   corrupted = __hle(center, bm_p_tol)
                              || __hge(center, bm_m_tol);
        result = corrupted ? frequency_median_def7_h(window, bestN) : center;
    }
    else
    {
        result = center;
    }

    out_plane[static_cast<size_t>(oy) * dstPitch + ox] = result;
}


// ============================================================================
//  Split BGRA-FP32 [0,1] -> 3 single-channel __half planes [0,255]
//  Scaling x255 done in FP32 before __float2half: one rounding step total.
// ============================================================================
__global__ void split_bgra_fp32_to_planar_fp16(
    const float*  RESTRICT bgra_in,        // FP32 BGRA [0,1]
          __half* RESTRICT plane_B,        // __half [0,255]
          __half* RESTRICT plane_G,
          __half* RESTRICT plane_R,
    int W, int H,
    int bgraPitchPixels,
    int planarPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const size_t bgraIdx   = (static_cast<size_t>(y) * bgraPitchPixels + x) * 4u;
    const size_t planarIdx =  static_cast<size_t>(y) * planarPitch + x;

    plane_B[planarIdx] = __float2half(bgra_in[bgraIdx + 0] * kFromUnit);
    plane_G[planarIdx] = __float2half(bgra_in[bgraIdx + 1] * kFromUnit);
    plane_R[planarIdx] = __float2half(bgra_in[bgraIdx + 2] * kFromUnit);
    // Alpha pulled at merge time from the original FP32 BGRA buffer.
}


// ============================================================================
//  Merge 3 __half planes [0,255] -> BGRA-FP32 [0,1], alpha pass-through
//  __half2float is lossless; the x(1/255) scale and clamp run in FP32.
// ============================================================================
__global__ void merge_planar_fp16_to_bgra_fp32(
    const __half* RESTRICT plane_B,
    const __half* RESTRICT plane_G,
    const __half* RESTRICT plane_R,
    const float*  RESTRICT bgra_orig,       // FP32 BGRA [0,1] for alpha
          float*  RESTRICT bgra_out,        // FP32 BGRA [0,1]
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

    float b = __half2float(plane_B[planarIdx]) * kToUnit;
    float g = __half2float(plane_G[planarIdx]) * kToUnit;
    float r = __half2float(plane_R[planarIdx]) * kToUnit;

    // Defensive clamp -- algorithm preserves values in [0,255] but FP16
    // round-trip and quantisation could nudge a hair outside.
    b = fmaxf(0.0f, fminf(1.0f, b));
    g = fmaxf(0.0f, fminf(1.0f, g));
    r = fmaxf(0.0f, fminf(1.0f, r));

    bgra_out[dstIdx + 0] = b;
    bgra_out[dstIdx + 1] = g;
    bgra_out[dstIdx + 2] = r;
    bgra_out[dstIdx + 3] = bgra_orig[srcIdx + 3];      // alpha pass-through
}


// ============================================================================
//  Centered-residual noise-map merge (runs in FP32 -- originals are FP32).
// ============================================================================
__global__ void merge_noise_map_fp16_to_bgra_fp32(
    const __half* RESTRICT plane_B,        // __half filtered [0,255]
    const __half* RESTRICT plane_G,
    const __half* RESTRICT plane_R,
    const float*  RESTRICT bgra_orig,      // FP32 BGRA [0,1]
          float*  RESTRICT bgra_out,       // FP32 BGRA [0,1]
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

    const float origB_255 = bgra_orig[srcIdx + 0] * kFromUnit;
    const float origG_255 = bgra_orig[srcIdx + 1] * kFromUnit;
    const float origR_255 = bgra_orig[srcIdx + 2] * kFromUnit;

    const float filtB_255 = __half2float(plane_B[planarIdx]);
    const float filtG_255 = __half2float(plane_G[planarIdx]);
    const float filtR_255 = __half2float(plane_R[planarIdx]);

    float dB = (origB_255 - filtB_255) + 128.0f;
    float dG = (origG_255 - filtG_255) + 128.0f;
    float dR = (origR_255 - filtR_255) + 128.0f;

    dB = fmaxf(0.0f, fminf(255.0f, dB)) * kToUnit;
    dG = fmaxf(0.0f, fminf(255.0f, dG)) * kToUnit;
    dR = fmaxf(0.0f, fminf(255.0f, dR)) * kToUnit;

    bgra_out[dstIdx + 0] = dB;
    bgra_out[dstIdx + 1] = dG;
    bgra_out[dstIdx + 2] = dR;
    bgra_out[dstIdx + 3] = bgra_orig[srcIdx + 3];      // alpha pass-through
}

// ============================================================================
//  ImageLabAFMF16_CUDA -- host-side aggregator
//  Same prototype as the FP32 path. Internals are __half.
// ============================================================================
CUDA_KERNEL_CALL
void ImageLabAFMF16_CUDA
(
    const float* RESTRICT inBuffer,
          float* RESTRICT outBuffer,
    int                   srcPitch,
    int                   dstPitch,
    int                   width,
    int                   height,
    const AlgoControls* RESTRICT algoGpuParams,
    int                   frameCount,
    cudaStream_t          stream
)
{
    const int   radius = algoGpuParams->radius;
    // Tolerance converted to __half once, reused across all kernel launches
    const __half tolerance_h = __float2half(algoGpuParams->tolerance);
    const int   iterations = algoGpuParams->iterations;
    const bool  wantNoise =
        (algoGpuParams->outputType == AFMF_Output::AFMF_OUTPUT_NOISE_MAP);

    const dim3 block(kBlockX, kBlockY, 1);
    const dim3 grid((width + kBlockX - 1) / kBlockX,
        (height + kBlockY - 1) / kBlockY,
        1);

    // ------------------------------------------------------------------------
    // Allocate internal __half planar work buffers from a single arena.
    //
    //    LAYOUT inside the arena (each slot 256-byte aligned):
    //      offset 0 * slot :  plane_B   (channel B __half plane)
    //      offset 1 * slot :  plane_G   (channel G __half plane)
    //      offset 2 * slot :  plane_R   (channel R __half plane)
    //      offset 3 * slot :  scratch   (shared __half ping-pong scratch)
    //
    //    Single cudaMalloc / single cudaFree.
    // ------------------------------------------------------------------------
    constexpr size_t kSubBufAlign = 256u;        // bytes per slot boundary
    constexpr int    kSubBufCount = 4;           // 3 planes + 1 scratch

    const int    planarPitch = width;
    const size_t planeBytes = static_cast<size_t>(planarPitch)
        * static_cast<size_t>(height)
        * sizeof(__half);

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

    __half* const plane_B = reinterpret_cast<__half*>(arena + 0u * slotBytes);
    __half* const plane_G = reinterpret_cast<__half*>(arena + 1u * slotBytes);
    __half* const plane_R = reinterpret_cast<__half*>(arena + 2u * slotBytes);
    __half* const scratch = reinterpret_cast<__half*>(arena + 3u * slotBytes);

    // ------------------------------------------------------------------------
    // inBuffer / outBuffer are device pointers from the host caller; the
    // kernels read/write them directly with no staging.
    // ------------------------------------------------------------------------
    const size_t bgraInFloatsPerFrame =
        static_cast<size_t>(height) * srcPitch * 4u;
    const size_t bgraOutFloatsPerFrame =
        static_cast<size_t>(height) * dstPitch * 4u;

       const float* frameIn = inBuffer;
        float* frameOut = outBuffer;

        // Split BGRA-FP32 [0,1] -> 3 __half planes [0,255]
        split_bgra_fp32_to_planar_fp16 << <grid, block, 0, stream >> >(
            frameIn, plane_B, plane_G, plane_R,
            width, height, srcPitch, planarPitch);

        // Per-channel AFMF (native __half internal) with shared scratch.
        __half* channels[3] = { plane_B, plane_G, plane_R };

        for (int c = 0; c < 3; ++c)
        {
            __half* channelPlane = channels[c];

            const __half* src = channelPlane;
            __half* dst = scratch;

            for (int it = 0; it < iterations; ++it)
            {
                if (radius == 1)
                {
                    afmf_kernel_planar_fp16_r1 << <grid, block, 0, stream >> >(
                        src, dst,
                        width, height,
                        planarPitch, planarPitch,
                        tolerance_h);
                }
                else
                {
                    afmf_kernel_planar_fp16 << <grid, block, 0, stream >> >(
                        src, dst,
                        width, height,
                        planarPitch, planarPitch,
                        radius, tolerance_h);
                }

                const __half* prevSrc = src;
                src = dst;
                dst = const_cast<__half*>(prevSrc);
            }

            // Copy back into the channel's main plane if the final write
            // landed in scratch.
            if (src == scratch)
            {
                cudaMemcpyAsync(channelPlane, scratch, planeBytes,
                    cudaMemcpyDeviceToDevice, stream);
            }
        }

        // Merge __half planes [0,255] -> BGRA-FP32 [0,1] (or noise map)
        if (wantNoise)
        {
            merge_noise_map_fp16_to_bgra_fp32 << <grid, block, 0, stream >> >(
                plane_B, plane_G, plane_R,
                frameIn, frameOut,
                width, height,
                planarPitch,
                srcPitch, dstPitch);
        }
        else
        {
            merge_planar_fp16_to_bgra_fp32 << <grid, block, 0, stream >> >(
                plane_B, plane_G, plane_R,
                frameIn, frameOut,
                width, height,
                planarPitch,
                srcPitch, dstPitch);
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
void ImageLabDenoise_CleanupGPUF16()
{
}

} // FP16 namespace


