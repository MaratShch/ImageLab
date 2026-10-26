#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "GuidedFilter_GPU.hpp"

// ============================================================================
// Guided Filter on CUDA (CUDA 10.2 API only; target sm_61, GTX 1060).
//
// Mirrors the CPU implementation (Scalar / AVX2 paths):
//   1. split BGRA_32f [0..1] -> orthonormal YUV planes on the 0..255 scale
//   2. per plane, 4 passes with clamp-to-edge box sums:
//        hs_I, hs_II = horizontal sums of I, I^2        (row prefix sums, double)
//        a, b        = from the vertical sums            (double; a = var/(var+eps), b = mI - a*mI)
//        hs_a, hs_b  = horizontal sums of a, b
//        q           = vsum(hs_a) * I + vsum(hs_b)
//   3. Skin / Background modes: skin confidence c from the filtered chroma,
//      classification skin <=> c >= 0.5 (Background = exact complement), output = region blend
//      or the selection view (input pixels of the selected class, black elsewhere), as on the CPU
//   4. merge back to BGRA_32f, alpha copied from the input
//
// All scratch memory comes from ONE arena (single cudaMalloc / cudaFree),
// each slot 256-byte aligned. Buffer sizes and pitches are in pixels.
// ============================================================================

namespace
{
    // ------------------------------------------------------------------------
    // Launch geometry
    // ------------------------------------------------------------------------
    constexpr int kBlockX    = 16;     // 2D kernels (split / merge / elementwise)
    constexpr int kBlockY    = 16;
    constexpr int kThreads1D = 256;    // row-scan and column kernels
    constexpr int kRowBlock  = 2048;   // prefix-sum segment (doubles) in shared memory
    constexpr int kRowBlockPadded = kRowBlock + (kRowBlock >> 4) + 1;   // padded P size (~17 KB with the warp totals)
    constexpr int kMinBand   = 64;     // minimum rows per vertical band

    // ------------------------------------------------------------------------
    // Colour constants (identical to the CPU paths)
    // ------------------------------------------------------------------------
    __constant__ float c_ortho_y   =  0.57735027f;   // 1/sqrt(3)
    __constant__ float c_ortho_u   =  0.70710678f;   // 1/sqrt(2)
    __constant__ float c_ortho_v   =  0.40824829f;   // 1/sqrt(6)
    __constant__ float c_ortho_v_g = -0.81649658f;   // -2/sqrt(6)

    // skin model (AlgoControls.hpp / AlgoSkinConfidence.cpp)
    constexpr float kInvSqrt2 = 0.70710678118654752f;
    constexpr float kInvSqrt6 = 0.40824829046386302f;
    __constant__ float c_CbU  = -0.668736f * kInvSqrt2;
    __constant__ float c_CbV  =  0.993792f * kInvSqrt6;
    __constant__ float c_CbC  =  26.0f;               // 128 - 102
    __constant__ float c_CrU  =  0.581312f * kInvSqrt2;
    __constant__ float c_CrV  =  1.256064f * kInvSqrt6;
    __constant__ float c_CrC  = -25.0f;               // 128 - 153
    constexpr float kHalfAxisCb = 25.0f;
    constexpr float kHalfAxisCr = 20.0f;
    __constant__ float c_skinBoundary = 0.5f;         // skin <=> c >= 0.5 (d <= 1), as on the CPU

    DEVICE_FORCE_INLINE_CALL int clampi (const int v, const int lo, const int hi) { return v < lo ? lo : (v > hi ? hi : v); }

    // padded shared-memory index of the prefix array: one spare double per 16
    DEVICE_FORCE_INLINE_CALL int pidx (const int k) { return k + (k >> 4); }

    // ------------------------------------------------------------------------
    // 1. BGRA_32f [0..1] interleaved -> orthonormal YUV planes (0..255 scale)
    //    pixel layout in memory: B, G, R, A (4 floats); pitch in pixels
    // ------------------------------------------------------------------------
    __global__ void split_bgra32f_to_ortho_yuv
    (
        const float* RESTRICT in,
        float* RESTRICT pY,
        float* RESTRICT pU,
        float* RESTRICT pV,
        const int width,
        const int height,
        const int srcPitch
    )
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        const float* RESTRICT px = in + (static_cast<size_t>(y) * srcPitch + x) * 4;
        const float B = px[0] * 255.0f, G = px[1] * 255.0f, R = px[2] * 255.0f;

        const size_t o = static_cast<size_t>(y) * width + x;
        pY[o] = (R + G + B) * c_ortho_y;
        pU[o] = (R - B) * c_ortho_u;
        pV[o] = fmaf(R + B, c_ortho_v, G * c_ortho_v_g);
    }

    // ------------------------------------------------------------------------
    // 2a. Horizontal clamp-to-edge box sums of one plane: one block per row.
    //     Prefix sum P in double in shared memory (segmented for long rows):
    //       P[0] = 0, P[k] = sum of v[base .. base+k-1],  v = s or s^2
    //       box(x) = P[min(x+r, W-1)-base+1] - P[max(x-r,0)-base]
    //                + max(0, r-x)*v[0] + max(0, x+r-(W-1))*v[W-1]
    //     Scan: each thread owns a contiguous chunk of P (chunk total, then the
    //     chunk prefix); the chunk totals are scanned with warp shuffles.
    //     P is stored padded (pidx: one spare double per 16) so that threads
    //     walking their chunks hit different shared-memory banks.
    //     Requires blockDim.x to be a multiple of 32 (launched with kThreads1D).
    //     Dynamic shared memory: (kRowBlockPadded + 32) doubles.
    // ------------------------------------------------------------------------
    template <bool SQ>
    __global__ void gf_hsum_rows
    (
        const float* RESTRICT src,
        float* RESTRICT dst,
        const int width,
        const int radius
    )
    {
        extern __shared__ double sh[];
        double* RESTRICT P    = sh;                     // kRowBlockPadded (padded prefix)
        double* RESTRICT wsum = sh + kRowBlockPadded;   // one inclusive total per warp (<= 32)

        const int row = blockIdx.y * gridDim.x + blockIdx.x;   // one block per row (grid may be 2D for > 65535 rows)
        const float* RESTRICT s = src + static_cast<size_t>(row) * width;
        float* RESTRICT d = dst + static_cast<size_t>(row) * width;

        const int T = blockDim.x, t = threadIdx.x;
        const int lane = t & 31, warp = t >> 5, nWarps = (T + 31) >> 5;
        const double inv_window = 1.0 / static_cast<double>(2 * radius + 1);
        const int W = width;
        const int seg = kRowBlock - 2 * radius - 2;     // output pixels per segment (radius <= 255 -> >= 1536)

        auto val = [&](const int x) -> double { const double v = s[x]; return SQ ? v * v : v; };
        const double v0 = val(0), vW = val(W - 1);

        for (int x0 = 0; x0 < W; x0 += seg)
        {
            const int x1   = min(x0 + seg, W);
            const int base = max(x0 - radius, 0);
            const int end  = min(x1 + radius, W - 1);
            const int n    = end - base + 1;            // <= kRowBlock - 1

            __syncthreads();                            // previous segment's reads of P / wsum are done
            // load (coalesced): P[k+1] = v[base + k]
            for (int k = t; k < n; k += T) P[pidx(k + 1)] = val(base + k);
            if (t == 0) P[0] = 0.0;
            __syncthreads();

            // 1) total of this thread's contiguous chunk
            const int chunk = (n + T - 1) / T;
            const int start = t * chunk;
            const int cnt   = max(0, min(chunk, n - start));
            double run = 0.0;
            for (int i = 0; i < cnt; ++i) run += P[pidx(start + i + 1)];

            // 2) exclusive scan of the chunk totals: inside each warp by shuffles,
            //    then across the warps (warp 0 scans the per-warp totals)
            double incl = run;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1)
            {
                const double nb = __shfl_up_sync(0xFFFFFFFFu, incl, off);
                if (lane >= off) incl += nb;
            }
            if (31 == lane) wsum[warp] = incl;
            __syncthreads();
            if (0 == warp)
            {
                double w = (lane < nWarps) ? wsum[lane] : 0.0;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1)
                {
                    const double nb = __shfl_up_sync(0xFFFFFFFFu, w, off);
                    if (lane >= off) w += nb;
                }
                if (lane < nWarps) wsum[lane] = w;
            }
            __syncthreads();
            double acc = (incl - run) + ((warp > 0) ? wsum[warp - 1] : 0.0);

            // 3) inclusive prefix of the chunk
            for (int i = 0; i < cnt; ++i)
            {
                const int p = pidx(start + i + 1);
                acc += P[p];
                P[p] = acc;
            }
            __syncthreads();

            // outputs (coalesced)
            for (int x = x0 + t; x < x1; x += T)
            {
                const int lo = x - radius, hi = x + radius;
                double sum = P[pidx(min(hi, W - 1) - base + 1)] - P[pidx(max(lo, 0) - base)];
                if (lo < 0)     sum += static_cast<double>(-lo) * v0;
                if (hi > W - 1) sum += static_cast<double>(hi - (W - 1)) * vW;
                d[x] = static_cast<float>(sum * inv_window);
            }
        }
    }

    // ------------------------------------------------------------------------
    // 2b. Vertical clamp-to-edge box sums: one thread per column, rows in bands.
    //     MODE 1 (coef) : s0 = hs_I, s1 = hs_II -> d0 = a, d1 = b
    //                     (window sums, means and variance in double; a, b in float)
    //     MODE 2 (apply): s0 = hs_a, s1 = hs_b, I  -> d0 = q = mean_a * I + mean_b
    // ------------------------------------------------------------------------
    template <int MODE>
    __global__ void gf_vsum_cols
    (
        const float* RESTRICT s0,
        const float* RESTRICT s1,
        const float* RESTRICT I,
        float* RESTRICT d0,
        float* RESTRICT d1,
        const int width,
        const int height,
        const int radius,
        const float epsilon,
        const int bandRows
    )
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= width) return;
        const int y0 = blockIdx.y * bandRows;
        const int y1 = min(y0 + bandRows, height);
        if (y0 >= height) return;

        const double inv_window = 1.0 / static_cast<double>(2 * radius + 1);

        // prime the window centred on y0 (clamp-to-edge)
        double a0 = 0.0, a1 = 0.0;
        for (int j = -radius; j <= radius; ++j)
        {
            const size_t o = static_cast<size_t>(clampi(y0 + j, 0, height - 1)) * width + x;
            a0 += s0[o];
            a1 += s1[o];
        }

        for (int y = y0; y < y1; ++y)
        {
            const size_t o = static_cast<size_t>(y) * width + x;
            const double m0 = a0 * inv_window, m1 = a1 * inv_window;
            if (MODE == 1)
            {
                // variance in double (m1 - m0^2 cancels); the division and b in float
                const float variance = static_cast<float>(fmax(0.0, m1 - m0 * m0));
                const float mean     = static_cast<float>(m0);
                const float a        = variance / (variance + epsilon);
                d0[o] = a;
                d1[o] = fmaf(-a, mean, mean);           // mean - a * mean
            }
            else
            {
                d0[o] = fmaf(static_cast<float>(m0), I[o], static_cast<float>(m1));
            }
            const size_t oB = static_cast<size_t>(clampi(y + radius + 1, 0, height - 1)) * width + x;
            const size_t oT = static_cast<size_t>(clampi(y - radius, 0, height - 1)) * width + x;
            a0 += static_cast<double>(s0[oB]) - static_cast<double>(s0[oT]);
            a1 += static_cast<double>(s1[oB]) - static_cast<double>(s1[oT]);
        }
    }

    // ------------------------------------------------------------------------
    // 3. Skin confidence c in [0..1] from the filtered chroma planes (same model as the CPU)
    // ------------------------------------------------------------------------
    __global__ void skin_confidence
    (
        const float* RESTRICT qU,
        const float* RESTRICT qV,
        float* RESTRICT conf,
        const int totalPixels,
        const float invCb,
        const float invCr,
        const float softness
    )
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= totalPixels) return;
        const float U = qU[i], V = qV[i];
        const float x = fmaf(c_CbV, V, fmaf(c_CbU, U, c_CbC)) * invCb;
        const float y = fmaf(c_CrV, V, fmaf(c_CrU, U, c_CrC)) * invCr;
        const float d2 = fmaf(x, x, y * y);
        if (softness <= 0.0f)
        {
            conf[i] = (d2 <= 1.0f) ? 1.0f : 0.0f;
            return;
        }
        const float d  = sqrtf(d2);
        const float d0 = 1.0f - 0.5f * softness;
        const float tt = fminf(fmaxf((d - d0) / softness, 0.0f), 1.0f);
        conf[i] = 1.0f - tt * tt * (3.0f - 2.0f * tt);
    }

    // region blend, in place on the filtered plane q: q = w*q + (1-w)*I, weight confined to the selected class
    //   Skin: w = max(0, 2c - 1) (0 for every non-skin pixel) ; Background: w = max(0, 1 - 2c) (0 for every skin pixel)
    __global__ void blend_region
    (
        const float* RESTRICT I,
        const float* RESTRICT conf,
        float* RESTRICT q,
        const int totalPixels,
        const int skin
    )
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= totalPixels) return;
        const float c = conf[i];
        const float w = fmaxf(skin ? fmaf(2.0f, c, -1.0f) : fmaf(-2.0f, c, 1.0f), 0.0f);
        q[i] = fmaf(w, q[i], (1.0f - w) * I[i]);
    }

    // selection view (confidence-map display), same classification as blend_region (skin <=> c >= 0.5):
    //   pixel of the selected class (Skin: skin, Background: non-skin) -> original input pixel (iY, iU, iV)
    //   pixel of the other class (left unfiltered by blend_region)      -> black (Y = U = V = 0)
    __global__ void write_confidence_map
    (
        const float* RESTRICT conf,
        const float* RESTRICT iY,
        const float* RESTRICT iU,
        const float* RESTRICT iV,
        float* RESTRICT oY,
        float* RESTRICT oU,
        float* RESTRICT oV,
        const int totalPixels,
        const int skin
    )
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= totalPixels) return;
        const bool isSkin = (conf[i] >= c_skinBoundary);
        const bool included = skin ? isSkin : !isSkin;
        oY[i] = included ? iY[i] : 0.0f;
        oU[i] = included ? iU[i] : 0.0f;
        oV[i] = included ? iV[i] : 0.0f;
    }

    // ------------------------------------------------------------------------
    // 4. orthonormal YUV planes (0..255) -> BGRA_32f [0..1]; alpha copied from the input
    // ------------------------------------------------------------------------
    __global__ void merge_ortho_yuv_to_bgra32f
    (
        const float* RESTRICT pY,
        const float* RESTRICT pU,
        const float* RESTRICT pV,
        const float* RESTRICT in,
        float* RESTRICT out,
        const int width,
        const int height,
        const int srcPitch,
        const int dstPitch
    )
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        const size_t o = static_cast<size_t>(y) * width + x;
        const float Yb = pY[o] * c_ortho_y;
        const float Urb = pU[o] * c_ortho_u;
        const float Vrb = pV[o] * c_ortho_v;

        float R = (Yb + Urb + Vrb) * (1.0f / 255.0f);
        float G = fmaf(pV[o], c_ortho_v_g, Yb) * (1.0f / 255.0f);
        float B = (Yb - Urb + Vrb) * (1.0f / 255.0f);
#if (_KEEP_CLAMP_TO_1 != 0)
        const float white = 1.0f - 1.1920929e-7f;        // f32_value_white (1.0f - FLT_EPSILON)
        R = fminf(fmaxf(R, 0.0f), white); G = fminf(fmaxf(G, 0.0f), white); B = fminf(fmaxf(B, 0.0f), white);
#endif
        const float* RESTRICT pin  = in  + (static_cast<size_t>(y) * srcPitch + x) * 4;
        float* RESTRICT       pout = out + (static_cast<size_t>(y) * dstPitch + x) * 4;
        pout[0] = B; pout[1] = G; pout[2] = R; pout[3] = pin[3];
    }

    // ------------------------------------------------------------------------
    // Host helpers
    // ------------------------------------------------------------------------
    float sanitize_float (const float v, const float vMin, const float vMax, const float vDef)
    {
        if (!(v == v) || std::fabs(v) == HUGE_VALF) return vDef;    // NaN / Inf
        return std::min(std::max(v, vMin), vMax);
    }

    AlgoControls sanitize_controls (const AlgoControls& in)
    {
        AlgoControls c = in;
        c.radius        = std::min(std::max(c.radius, AlgoRadiusMin), AlgoRadiusMax);
        c.epsilon       = sanitize_float(c.epsilon,       AlgoEpsilonMin,       AlgoEpsilonMax,       AlgoEpsilonDef);
        c.skinTolerance = sanitize_float(c.skinTolerance, AlgoSkinToleranceMin, AlgoSkinToleranceMax, AlgoSkinToleranceDef);
        c.skinSoftness  = sanitize_float(c.skinSoftness,  AlgoSkinSoftnessMin,  AlgoSkinSoftnessMax,  AlgoSkinSoftnessDef);
        return c;
    }

    bool valid_mode (const FilterMode m)
    {
        return (FilterMode::Entire_Image == m || FilterMode::Skin == m || FilterMode::Background == m);
    }

    // one plane: I -> q (4 passes), scratch s0..s3 are full planes
    void guided_filter_plane
    (
        const float* RESTRICT I,
        float* RESTRICT q,
        float* RESTRICT s0, float* RESTRICT s1, float* RESTRICT s2, float* RESTRICT s3,
        const int width, const int height, const int radius, const float scaled_epsilon,
        const dim3 gridRows, const dim3 gridCols, const int bandRows, const size_t shmem, cudaStream_t stream
    )
    {
        const dim3 block1D(kThreads1D, 1, 1);
        gf_hsum_rows<false><<<gridRows, block1D, shmem, stream>>>(I,  s0, width, radius);   // hs_I
        gf_hsum_rows<true><<<gridRows, block1D, shmem, stream>>>(I,  s1, width, radius);   // hs_II
        gf_vsum_cols<1><<<gridCols, block1D, 0, stream>>>(s0, s1, nullptr, s2, s3, width, height, radius,
                                                          scaled_epsilon, bandRows);   // a, b
        gf_hsum_rows<false><<<gridRows, block1D, shmem, stream>>>(s2, s0, width, radius);   // hs_a
        gf_hsum_rows<false><<<gridRows, block1D, shmem, stream>>>(s3, s1, width, radius);   // hs_b
        gf_vsum_cols<2><<<gridCols, block1D, 0, stream>>>(s0, s1, I, q, nullptr, width, height, radius, 0.0f, bandRows);   // q
    }
} // anonymous namespace


CUDA_KERNEL_CALL
void ImageLabGuidedFilter32_CUDA
(
    const float*  RESTRICT inBuffer,
          float*  RESTRICT outBuffer,
    int                    srcPitch,
    int                    dstPitch,
    int                    width,
    int                    height,
    AlgoControls* RESTRICT algoGpuParams,
    int                    frameCount,
    cudaStream_t           stream
)
{
    // One frame per call (same convention as ImageLabAFMF32_CUDA): frameCount is not used.
    // Previously frameCount was treated as a frame count, so frameCount == 0 launched no kernel
    // and outBuffer was never written (black output).
    (void)frameCount;
    if (nullptr == inBuffer || nullptr == outBuffer || nullptr == algoGpuParams ||
        width <= 0 || height <= 0 || srcPitch < width || dstPitch < width)
        return;

    // ------------------------------------------------------------------------
    // 1. Controls (same validation as the CPU paths)
    // ------------------------------------------------------------------------
    const AlgoControls ctrl   = sanitize_controls(*algoGpuParams);
    const bool  passThrough   = !valid_mode(algoGpuParams->filterMode);
    const int   radius        = ctrl.radius;
    const float scaledEpsilon = ctrl.epsilon * 65025.0f;              // epsilon * 255^2 (the only scaling)
    const bool  entire        = (FilterMode::Entire_Image == ctrl.filterMode);
    const int   skin          = (FilterMode::Skin == ctrl.filterMode) ? 1 : 0;
    const bool  wantMap       = ctrl.showConfidenceMap && !entire;
    const float invCb         = 1.0f / (kHalfAxisCb * ctrl.skinTolerance);
    const float invCr         = 1.0f / (kHalfAxisCr * ctrl.skinTolerance);

    // ------------------------------------------------------------------------
    // 2. Grid setup
    // ------------------------------------------------------------------------
    const dim3 block2D(kBlockX, kBlockY, 1);
    const dim3 grid2D((width + kBlockX - 1) / kBlockX, (height + kBlockY - 1) / kBlockY, 1);

    const int  totalPixels = width * height;
    const dim3 block1D(kThreads1D, 1, 1);
    const dim3 grid1D((totalPixels + kThreads1D - 1) / kThreads1D, 1, 1);

    // one block per row; grid.x limited to 65535 -> fold into y when taller
    const int rowsX = std::min(height, 65535);
    const dim3 gridRows(rowsX, (height + rowsX - 1) / rowsX, 1);
    // one thread per column, rows in bands (band >= 2r to amortise the window priming)
    const int  bandRows = std::max(kMinBand, 2 * radius);
    const dim3 gridCols((width + kThreads1D - 1) / kThreads1D, (height + bandRows - 1) / bandRows, 1);
    const size_t shmem = static_cast<size_t>(kRowBlockPadded + 32) * sizeof(double);   // ~17 KB

    // ------------------------------------------------------------------------
    // 3. Arena: 10 planar float slots, each 256-byte aligned, one cudaMalloc.
    //      0..2 : in_Y, in_U, in_V          3..5 : out_Y, out_U, out_V
    //      6..9 : scratch s0..s3 (hs_I/hs_II/a/b; s0 reused for the skin confidence)
    // ------------------------------------------------------------------------
    constexpr size_t kSubBufAlign = 256u;
    constexpr int    kSubBufCount = 10;

    const size_t planeBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float);
    const size_t slotBytes  = (planeBytes + kSubBufAlign - 1u) & ~(kSubBufAlign - 1u);
    const size_t arenaBytes = slotBytes * static_cast<size_t>(kSubBufCount);

    uint8_t* arena = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&arena), arenaBytes) != cudaSuccess || arena == nullptr)
    {
        // Allocation failure: leave outBuffer untouched and return.
        return;
    }

    float* const slot[kSubBufCount] = {
        reinterpret_cast<float*>(arena + 0u * slotBytes), reinterpret_cast<float*>(arena + 1u * slotBytes),
        reinterpret_cast<float*>(arena + 2u * slotBytes), reinterpret_cast<float*>(arena + 3u * slotBytes),
        reinterpret_cast<float*>(arena + 4u * slotBytes), reinterpret_cast<float*>(arena + 5u * slotBytes),
        reinterpret_cast<float*>(arena + 6u * slotBytes), reinterpret_cast<float*>(arena + 7u * slotBytes),
        reinterpret_cast<float*>(arena + 8u * slotBytes), reinterpret_cast<float*>(arena + 9u * slotBytes) };
    float* const inY = slot[0];  float* const inU = slot[1];  float* const inV = slot[2];
    float* const outY = slot[3]; float* const outU = slot[4]; float* const outV = slot[5];
    float* const s0 = slot[6];   float* const s1 = slot[7];   float* const s2 = slot[8];   float* const s3 = slot[9];
    float* const conf = s0;

    // ------------------------------------------------------------------------
    // 4. Frame
    // ------------------------------------------------------------------------
    {
        const float* RESTRICT frameIn  = inBuffer;
        float* RESTRICT       frameOut = outBuffer;

        // 4a. split BGRA[0..1] -> orthonormal YUV planes [0..255]
        split_bgra32f_to_ortho_yuv<<<grid2D, block2D, 0, stream>>>(frameIn, inY, inU, inV, width, height, srcPitch);

        if (passThrough)
        {
            // unknown filterMode: contract violation -> image passes through unchanged
            cudaMemcpyAsync(outY, inY, planeBytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(outU, inU, planeBytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(outV, inV, planeBytes, cudaMemcpyDeviceToDevice, stream);
        }
        else
        {
            // 4b. guided filter of the three planes over the whole image
            guided_filter_plane(inY, outY, s0, s1, s2, s3, width, height, radius, scaledEpsilon, gridRows, gridCols, bandRows, shmem, stream);
            guided_filter_plane(inU, outU, s0, s1, s2, s3, width, height, radius, scaledEpsilon, gridRows, gridCols, bandRows, shmem, stream);
            guided_filter_plane(inV, outV, s0, s1, s2, s3, width, height, radius, scaledEpsilon, gridRows, gridCols, bandRows, shmem, stream);

            // 4c. region selection (Entire_Image: filtered image, no classification)
            if (!entire)
            {
                skin_confidence<<<grid1D, block1D, 0, stream>>>(outU, outV, conf, totalPixels, invCb, invCr, ctrl.skinSoftness);
                if (wantMap)
                {
                    write_confidence_map<<<grid1D, block1D, 0, stream>>>(conf, inY, inU, inV, outY, outU, outV, totalPixels, skin);
                }
                else
                {
                    blend_region<<<grid1D, block1D, 0, stream>>>(inY, conf, outY, totalPixels, skin);
                    blend_region<<<grid1D, block1D, 0, stream>>>(inU, conf, outU, totalPixels, skin);
                    blend_region<<<grid1D, block1D, 0, stream>>>(inV, conf, outV, totalPixels, skin);
                }
            }
        }

        // 4d. merge planes -> BGRA_32f [0..1]
        merge_ortho_yuv_to_bgra32f<<<grid2D, block2D, 0, stream>>>(outY, outU, outV, frameIn, frameOut, width, height, srcPitch, dstPitch);
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
void ImageLabGuidedFilter_CleanupGPUF32()
{
}