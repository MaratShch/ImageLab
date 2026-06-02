#ifndef __IMAGE_LAB_AWB_GPU_COMMON_DEVICE_HELPERS__
#define __IMAGE_LAB_AWB_GPU_COMMON_DEVICE_HELPERS__
// ============================================================================
//  AwbGpuCommon.cuh
//
//  Format-agnostic __device__ helpers shared by the 32f and 16f kernels.
//  These are plain inline device functions -- NOT templates and NOT kernels.
//  The two kernel sets (32f / 16f) remain fully separate; they only share this
//  scalar math so the color science stays identical between them.
//
//  Mirrors the validated CPU core exactly:
//    - sRGB EOTF/OETF : same piecewise curve as the CPU converter
//    - gray metric     : orthonormal opponent basis, gWhiteGuard=0.98,
//                        gBlackGuard=1e-4, chroma2 < (threshold*Y)^2
//    - apply           : full 3x3 in linear RGB, negatives floored, no upper clamp
// ============================================================================

#include <cuda_runtime.h>
#include <cfloat>

// 3x3 linear-RGB correction matrix, passed by value to the apply kernels.
struct AwbMat3 { float m[9]; };

// ----------------------------------------------------------------------------
//  OPTIONAL PER-KERNEL TIMING  (enable with -D__GPU_KERNEL_INSTRUMENTATION)
//  When the macro is undefined every KT_* expands to a no-op -> zero overhead.
//  When defined, each launch is bracketed by CUDA events and per-kernel totals
//  are printed at the end of the launcher. Timing forces a per-launch
//  synchronize (it serializes the stream), so use it only for profiling builds.
// ----------------------------------------------------------------------------
#ifdef __GPU_KERNEL_INSTRUMENTATION
#include <cstdio>
struct AwbKernelTimer
{
    static const int MAXK = 48;                 // pack + (reduce+apply)*16 + unpack + copy
    cudaStream_t stream;
    cudaEvent_t  b[MAXK], e[MAXK];
    const char*  lbl[MAXK];
    int          n;
    explicit AwbKernelTimer(cudaStream_t s) : stream(s), n(0)
    { for (int i = 0; i < MAXK; ++i) { cudaEventCreate(&b[i]); cudaEventCreate(&e[i]); lbl[i] = 0; } }
    ~AwbKernelTimer() { for (int i = 0; i < MAXK; ++i) { cudaEventDestroy(b[i]); cudaEventDestroy(e[i]); } }

    // Async stream markers ONLY -- no synchronize here, so the stream is not
    // serialized and the measured kernel times stay representative.
    inline void begin(const char* name) { if (n < MAXK) { lbl[n] = name; cudaEventRecord(b[n], stream); } }
    inline void end()                    { if (n < MAXK) { cudaEventRecord(e[n], stream); ++n; } }

    // Called ONCE on exit, AFTER the caller has synchronized the stream.
    inline void report(const char* tag) const
    {
        printf("============================================================\n");
        printf("%s : per-kernel GPU timing\n", tag);
        float total = 0.f;
        for (int i = 0; i < n; ++i)
        {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, b[i], e[i]);
            total += ms;
            printf("   %-10s : %9.4f mSec\n", lbl[i], ms);
        }
        printf("   %-10s : %9.4f mSec\n", "SUM", total);
        printf("============================================================\n");
    }
};
#define KT_DECL(s)     AwbKernelTimer kt(s)
#define KT_BEG(name)   kt.begin(name)
#define KT_END()       kt.end()
#define KT_REPORT(tag) kt.report(tag)
#else
#define KT_DECL(s)     ((void)0)
#define KT_BEG(name)   ((void)0)
#define KT_END()       ((void)0)
#define KT_REPORT(tag) ((void)0)
#endif

namespace awb_gpu
{
    // ---- exact sRGB transfer (same formula as the CPU path) ----------------
    // powf maps to the fast intrinsic under --use_fast_math; its ~2^-21 relative
    // error is far below 8/16-bit quantization (the exact curve round-trips at
    // 0 codes), so results match the CPU converter.
    __device__ __forceinline__ float srgb_to_linear(float c) noexcept
    {
        c = (c > 0.0f) ? c : 0.0f;                               // floor negatives
        return (c <= 0.04045f) ? c * (1.0f / 12.92f)
                               : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
    }
    __device__ __forceinline__ float linear_to_srgb(float c) noexcept
    {
        c = (c > 0.0f) ? c : 0.0f;                               // floor negatives (HDR>1 kept)
        return (c <= 0.0031308f) ? 12.92f * c
                                 : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
    }

    // ---- gray classification (linear RGB) ----------------------------------
    // Returns 1.0f and fills lr/lg/lb with the linear RGB if the pixel is a
    // selected (near-gray, not clipped, not black) sample; else returns 0.0f.
    __device__ __forceinline__ float gray_select(float r, float g, float b, float thr,
                                                 float& lr, float& lg, float& lb) noexcept
    {
        const float INV_SQRT2 = 0.70710678118f;
        const float INV_SQRT3 = 0.57735026919f;
        const float INV_SQRT6 = 0.40824829046f;
        const float gBlackGuard = 1.0e-4f;
        const float gWhiteGuard = 0.98f;

        const float mx = fmaxf(r, fmaxf(g, b));
        const float mn = fminf(r, fminf(g, b));
        const float Y  = (r + g + b)        * INV_SQRT3;
        const float C1 = (r - g)            * INV_SQRT2;
        const float C2 = (r + g - 2.0f * b) * INV_SQRT6;
        const float chroma2 = C1 * C1 + C2 * C2;
        const float Ysafe   = fmaxf(Y, FLT_EPSILON);
        const float rhs     = thr * Ysafe;

        if (mx < gWhiteGuard && mn >= gBlackGuard && chroma2 < rhs * rhs)
        {
            lr = r; lg = g; lb = b;
            return 1.0f;
        }
        lr = lg = lb = 0.0f;
        return 0.0f;
    }

    // ---- 3x3 apply in linear RGB (negatives floored, no upper clamp) -------
    __device__ __forceinline__ void apply_matrix(const AwbMat3& M,
                                                 float r, float g, float b,
                                                 float& nr, float& ng, float& nb) noexcept
    {
        nr = fmaxf(M.m[0] * r + M.m[1] * g + M.m[2] * b, 0.0f);
        ng = fmaxf(M.m[3] * r + M.m[4] * g + M.m[5] * b, 0.0f);
        nb = fmaxf(M.m[6] * r + M.m[7] * g + M.m[8] * b, 0.0f);
    }
} // namespace awb_gpu

#endif // __IMAGE_LAB_AWB_GPU_COMMON_DEVICE_HELPERS__
