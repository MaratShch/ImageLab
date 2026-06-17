// ============================================================================
//  ImageLabPCA32_CUDA.cu
//
//  CUDA (compute >= sm_60; targets sm_61 / sm_75) port of the Cheng
//  "bright-and-dark colors" PCA Automatic White Balance, float32 pipeline.
//
//  Faithful to the validated CPU/AVX2 path (AlgorithmMain_PCA.cpp +
//  AwbCorrection.cpp):
//  Just 4 kernels (the three former <<<1,1>>> finalize steps are folded into
//  the tail of their reduction via a threadfence / last-block barrier):
//    k_mean    mean direction over valid pixels (valid = max(rgb) < satThr &&
//              luma >= blackY, luma from colorSpace); last block normalizes it.
//    k_hist    4096-bin projection histogram (gProjMax = 1.74); last block then
//              derives the bright/dark thresholds (k = fraction*count).
//    k_moment  uncentered 3x3 second moment over the selected extremes; last
//              block runs the Jacobi 3x3 + builds the CAT matrix (illuminate,
//              chromatic).
//    k_apply   out = max(M * rgb, 0); alpha copied through.
//
//  HOST IS ISOLATED FROM THE RUN-TIME: the host only reads algoGpuParams (a
//  host-side struct), performs ONE cudaMalloc for a single small "arena",
//  enqueues the 4 kernels on the stream, frees the arena, and synchronizes
//  last. Mean / histogram / cut / moment / eigensolve / matrix build all run in
//  device kernels writing into the arena -- there is NO device->host readback
//  of intermediate results. The last-block barrier uses a per-stage counter in
//  the arena (blocksDone1/2/3); __threadfence() before the atomicInc publishes
//  each block's partials so the finalizing block sees the complete reduction.
//
//  Buffer format: BGRA, 32f interleaved (float4: .x=B .y=G .z=R .w=A).
//
//  COLOR PIPELINE (defaults match the proven-clean gray-point kernel exactly):
//  by default every pass decodes sRGB->linear on load, the PCA runs in linear
//  RGB (the CPU/AVX2 reference domain), apply re-encodes linear->sRGB on store
//  and clamps the result to [0,1]. Two opt-OUT macros relax this for a true
//  scene-linear / HDR float project:
//
//  _GPU_INPUT_LINEAR (undefined by default):
//     undefined -> incoming buffer is sRGB-encoded (the Premiere GPU render
//                  buffer): decode sRGB->linear on load (every pass) and
//                  re-encode linear->sRGB on store. This is what the gray-point
//                  kernel does and what keeps the image clean.
//     defined   -> incoming buffer is ALREADY scene-linear: skip both the
//                  decode and the encode (PCA still runs in linear).
//
//  _GPU_HDR_OUTPUT (undefined by default):
//     undefined -> clamp corrected RGB to [0,1] as the final step before store
//                  (same as the gray-point kernel's fminf(...,1.0f)). Prevents
//                  the float->8-bit WRAP that turns >1.0 highlights into false
//                  hues (red wrap -> cyan, blue wrap -> yellow).
//     defined   -> no upper clamp; >1.0 super-whites pass through for a true
//                  32f HDR destination that clamps at its own export stage.
//  The alpha channel is ALWAYS copied verbatim from input to output.
//
//  Macros (CUDA_KERNEL_CALL, RESTRICT, ...) come from the project headers and
//  are never redefined here.
// ============================================================================

#include <cuda_runtime.h>
#include <math.h>

#include "AuthomaticWhiteBalance2GPU.hpp"
#include "ColorTransformMatrix.hpp"

// ----------------------------------------------------------------------------
//  Fixed algorithm constants (identical to the CPU/AVX2 reference)
// ----------------------------------------------------------------------------
#define AWB_PCA_BINS      4096
#define AWB_PCA_PROJMAX   1.74f      // max projection (c.u.) for channels < 1 (~sqrt 3)
#define AWB_MIN_VALID     16         // CPU returns "fail" when valid count < 16

// ----------------------------------------------------------------------------
//  Device-resident constant tables (baked into the module; no host copy).
//  Values copied verbatim from ColorTransformMatrix.hpp / AlgCorrectionMatrix.hpp.
// ----------------------------------------------------------------------------
// luminance weights = first row of RGB2YUV[colorSpace]  (BT601,BT709,BT2020,SMPTE240M)
__device__ const float cLuma[4][3] =
{
    { 0.299000f, 0.587000f, 0.114000f },
    { 0.212600f, 0.715200f, 0.072200f },
    { 0.262700f, 0.678000f, 0.059300f },
    { 0.212200f, 0.701300f, 0.086500f }
};

__device__ const float cSRGBtoXYZ[9] =
{
    0.4124564f, 0.3575761f, 0.1804375f,
    0.2126729f, 0.7151522f, 0.0721750f,
    0.0193339f, 0.1191920f, 0.9503041f
};
__device__ const float cXYZtosRGB[9] =
{
    3.240455f, -1.537139f, -0.498532f,
   -0.969266f,  1.876011f,  0.041556f,
    0.055643f, -0.204026f,  1.057225f
};

// GetIlluminate() target white points (Y=100), indexed by eILLUMINATE (0..10)
__device__ const float cIlluminate[11][3] =
{
    {  95.0470f, 100.0000f, 108.8830f }, // DAYLIGHT (D65)
    {  98.0740f, 100.0000f, 118.2320f }, // OLD_DAYLIGHT
    {  99.0927f, 100.0000f,  85.3130f }, // OLD_DIRECT_SUNLIGHT_AT_NOON
    {  95.6820f, 100.0000f,  92.1490f }, // MID_MORNING_DAYLIGHT
    {  94.9720f, 100.0000f, 122.6380f }, // NORTH_SKY_DAYLIGHT
    {  92.8340f, 100.0000f, 103.6650f }, // DAYLIGHT_FLUORESCENT_F1
    {  99.1870f, 100.0000f,  67.3950f }, // COOL_FLUERESCENT
    { 103.7540f, 100.0000f,  49.8610f }, // WHITE_FLUORESCENT
    { 109.1470f, 100.0000f,  38.8130f }, // WARM_WHITE_FLUORESCENT
    {  90.8720f, 100.0000f,  98.7230f }, // DAYLIGHT_FLUORESCENT_F5
    { 100.3650f, 100.0000f,  67.8680f }  // COOL_WHITE_FLUORESCENT
};

// GetColorAdaptation() cone matrices, indexed by eChromaticAdaptation (0..4)
__device__ const float cCat[5][9] =
{
    { 0.73280f,  0.4296f, -0.16240f, -0.7036f, 1.69750f, 0.0061f, 0.0030f,  0.0136f, 0.98340f }, // CAT02
    { 0.40024f,  0.7076f, -0.08081f, -0.2263f, 1.16532f, 0.0457f, 0.0f,     0.0f,    0.91822f }, // VON-KRIES
    { 0.89510f,  0.2664f, -0.16140f, -0.7502f, 1.71350f, 0.0367f, 0.0389f, -0.0685f, 1.02960f }, // BRADFORD
    { 1.26940f, -0.0988f, -0.17060f, -0.8364f, 1.80060f, 0.0357f, 0.0297f, -0.0315f, 1.00180f }, // SHARP
    { 0.79820f,  0.3389f, -0.13710f, -0.5918f, 1.55120f, 0.0406f, 0.0008f,  0.2390f, 0.97530f }  // CMCCAT2000
};
__device__ const float cCatInv[5][9] =
{
    { 1.096124f, -0.278869f, 0.182745f, 0.454369f, 0.473533f, 0.072098f, -0.009628f, -0.005698f, 1.015326f }, // INV CAT02
    { 1.859936f, -1.129382f, 0.219897f, 0.361191f, 0.638812f, 0.0f,       0.0f,       0.0f,      1.089064f }, // INV VON-KRIES
    { 0.986993f, -0.147054f, 0.159963f, 0.432305f, 0.518360f, 0.049291f, -0.008529f,  0.040043f, 0.968487f }, // INV BRADFORD
    { 0.815633f,  0.047155f, 0.137217f, 0.379114f, 0.576942f, 0.044001f, -0.012260f,  0.016743f, 0.995519f }, // INV SHARP
    { 1.062305f, -0.256743f, 0.160018f, 0.407920f, 0.55023f,  0.034437f, -0.100833f, -0.134626f, 1.016755f }  // INV CMCCAT2000
};

// ----------------------------------------------------------------------------
//  Single arena: ONE cudaMalloc holds every intermediate. No other allocation.
// ----------------------------------------------------------------------------
struct PcaArena
{
    double       accum[3];            // pass1 : sum R,G,B over valid pixels
    double       count;               // pass1 : valid pixel count
    double       mean[3];             // finalize_mean : unit mean direction
    double       thr[2];              // cut   : darkThr, brightThr
    double       moment[6];           // pass3 : S00,S01,S02,S11,S12,S22
    float        M[9];                // solve_build : 3x3 correction matrix
    int          ok;                  // validity flag (0 -> identity fallback)
    unsigned int blocksDone1;         // last-block barrier counter for kernel 1 (mean)
    unsigned int blocksDone2;         // last-block barrier counter for kernel 2 (hist)
    unsigned int blocksDone3;         // last-block barrier counter for kernel 3 (moment)
    unsigned int hist[AWB_PCA_BINS];  // pass2 : projection histogram
};

// ----------------------------------------------------------------------------
//  Small device helpers
// ----------------------------------------------------------------------------
__device__ __forceinline__ float dev_eotf(float c)   // sRGB -> linear
{
    c = (c > 0.f) ? c : 0.f;
    return (c <= 0.04045f) ? (c * (1.f / 12.92f))
                           : powf((c + 0.055f) * (1.f / 1.055f), 2.4f);
}
__device__ __forceinline__ float dev_oetf(float c)   // linear -> sRGB
{
    c = (c > 0.f) ? c : 0.f;
    return (c <= 0.0031308f) ? (c * 12.92f)
                             : (1.055f * powf(c, 1.f / 2.4f) - 0.055f);
}

// Load B,G,R from a BGRA float4 and linearize (sRGB -> linear) BY DEFAULT.
// The Premiere GPU render buffer is display-encoded (sRGB), which is why the
// proven-clean gray-point kernel calls srgb_to_linear in BOTH its estimate and
// its apply. The PCA algorithm (like the CPU/AVX2 reference) operates in linear
// RGB, so we must decode here too. Define _GPU_INPUT_LINEAR only for a project
// whose float buffer is already scene-linear (then we skip decode AND encode).
__device__ __forceinline__ void dev_load_rgb(const float4 p, float& r, float& g, float& b)
{
    r = p.z; g = p.y; b = p.x;
#ifndef _GPU_INPUT_LINEAR
    r = dev_eotf(r); g = dev_eotf(g); b = dev_eotf(b);
#endif
}

__inline__ __device__ double warpReduceSum(double v)
{
    for (int off = warpSize / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}
__inline__ __device__ double blockReduceSum(double v)
{
    __shared__ double sh[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid  = threadIdx.x / warpSize;
    v = warpReduceSum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nWarp = (blockDim.x + warpSize - 1) / warpSize;
    v = ((int)threadIdx.x < nWarp) ? sh[lane] : 0.0;
    if (wid == 0) v = warpReduceSum(v);
    return v;                          // valid in thread 0
}

// cyclic Jacobi eigensolver for a real symmetric 3x3 (single thread).
__device__ void dev_jacobi3(double A[3][3], double w[3], double V[3][3])
{
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) V[i][j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 12; ++sweep)
    {
        const double off = fabs(A[0][1]) + fabs(A[0][2]) + fabs(A[1][2]);
        if (off < 1.0e-20) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q)
            {
                if (fabs(A[p][q]) < 1.0e-300) continue;
                const double phi = 0.5 * atan2(2.0 * A[p][q], A[q][q] - A[p][p]);
                const double c = cos(phi), s = sin(phi);
                for (int k = 0; k < 3; ++k) { const double a = A[k][p], b = A[k][q]; A[k][p] = c*a - s*b; A[k][q] = s*a + c*b; }
                for (int k = 0; k < 3; ++k) { const double a = A[p][k], b = A[q][k]; A[p][k] = c*a - s*b; A[q][k] = s*a + c*b; }
                for (int k = 0; k < 3; ++k) { const double a = V[k][p], b = V[k][q]; V[k][p] = c*a - s*b; V[k][q] = s*a + c*b; }
            }
    }
    w[0] = A[0][0]; w[1] = A[1][1]; w[2] = A[2][2];
}

// build the 3x3 linear-RGB CAT matrix (port of build_correction_matrix_linear).
__device__ void dev_build_matrix(const double e[3], int illumIdx, int chromIdx, float M[9])
{
    M[0]=1.f; M[1]=0.f; M[2]=0.f; M[3]=0.f; M[4]=1.f; M[5]=0.f; M[6]=0.f; M[7]=0.f; M[8]=1.f;

    const float er = (float)e[0], eg = (float)e[1], eb = (float)e[2];

    const float Xe = er*cSRGBtoXYZ[0] + eg*cSRGBtoXYZ[1] + eb*cSRGBtoXYZ[2];
    const float Ye = er*cSRGBtoXYZ[3] + eg*cSRGBtoXYZ[4] + eb*cSRGBtoXYZ[5];
    const float Ze = er*cSRGBtoXYZ[6] + eg*cSRGBtoXYZ[7] + eb*cSRGBtoXYZ[8];

    const float sum = Xe + Ye + Ze;
    if (sum <= 1.0e-7f) return;
    const float xe = Xe / sum, ye = Ye / sum;
    if (ye <= 1.0e-7f) return;
    const float kY = 100.0f / ye;
    const float estXYZ[3] = { kY*xe, 100.0f, kY*(1.0f - xe - ye) };

    const float* tgt  = cIlluminate[illumIdx];
    const float* A    = cCat[chromIdx];
    const float* Ainv = cCatInv[chromIdx];

    const float coneT[3] =
    {
        tgt[0]*A[0] + tgt[1]*A[1] + tgt[2]*A[2],
        tgt[0]*A[3] + tgt[1]*A[4] + tgt[2]*A[5],
        tgt[0]*A[6] + tgt[1]*A[7] + tgt[2]*A[8]
    };
    const float coneE[3] =
    {
        estXYZ[0]*A[0] + estXYZ[1]*A[1] + estXYZ[2]*A[2],
        estXYZ[0]*A[3] + estXYZ[1]*A[4] + estXYZ[2]*A[5],
        estXYZ[0]*A[6] + estXYZ[1]*A[7] + estXYZ[2]*A[8]
    };
    const float g0 = (fabsf(coneE[0]) > 1.0e-7f) ? (coneT[0]/coneE[0]) : 1.f;
    const float g1 = (fabsf(coneE[1]) > 1.0e-7f) ? (coneT[1]/coneE[1]) : 1.f;
    const float g2 = (fabsf(coneE[2]) > 1.0e-7f) ? (coneT[2]/coneE[2]) : 1.f;

    const float D[9] =
    {
        g0*A[0], g0*A[1], g0*A[2],
        g1*A[3], g1*A[4], g1*A[5],
        g2*A[6], g2*A[7], g2*A[8]
    };

    float Mx[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Mx[r*3+c] = Ainv[r*3+0]*D[0*3+c] + Ainv[r*3+1]*D[1*3+c] + Ainv[r*3+2]*D[2*3+c];

    float T1[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            T1[r*3+c] = cXYZtosRGB[r*3+0]*Mx[0*3+c] + cXYZtosRGB[r*3+1]*Mx[1*3+c] + cXYZtosRGB[r*3+2]*Mx[2*3+c];

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            M[r*3+c] = T1[r*3+0]*cSRGBtoXYZ[0*3+c] + T1[r*3+1]*cSRGBtoXYZ[1*3+c] + T1[r*3+2]*cSRGBtoXYZ[2*3+c];
}

// ============================================================================
//  KERNELS
// ============================================================================
__global__ void k_mean(const float4* RESTRICT in, int pitch, int w, int h,
                       float satThr, float blackY, int csIdx, PcaArena* RESTRICT a)
{
    const float lr = cLuma[csIdx][0], lg = cLuma[csIdx][1], lb = cLuma[csIdx][2];
    const long long total = (long long)w * (long long)h;
    double sR = 0.0, sG = 0.0, sB = 0.0, sC = 0.0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x)
    {
        const int x = (int)(i % w), y = (int)(i / w);
        float r, g, b; dev_load_rgb(in[(long long)y * pitch + x], r, g, b);
        const float mx  = fmaxf(r, fmaxf(g, b));
        const float lum = lr*r + lg*g + lb*b;
        if (mx < satThr && lum >= blackY) { sR += r; sG += g; sB += b; sC += 1.0; }
    }
    sR = blockReduceSum(sR); __syncthreads();
    sG = blockReduceSum(sG); __syncthreads();
    sB = blockReduceSum(sB); __syncthreads();
    sC = blockReduceSum(sC); __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAdd(&a->accum[0], sR); atomicAdd(&a->accum[1], sG);
        atomicAdd(&a->accum[2], sB); atomicAdd(&a->count,    sC);
        __threadfence();                                    // publish our partials device-wide
        // FOLDED-IN finalize: the last block to arrive normalizes the mean direction.
        if (atomicInc(&a->blocksDone1, gridDim.x) == gridDim.x - 1u)
        {
            const double sr = a->accum[0], sg = a->accum[1], sb = a->accum[2];
            if (a->count < (double)AWB_MIN_VALID) { a->ok = 0; a->mean[0]=a->mean[1]=a->mean[2]=0.0; }
            else {
                const double mn = sqrt(sr*sr + sg*sg + sb*sb);
                if (mn <= 1.0e-12) { a->ok = 0; a->mean[0]=a->mean[1]=a->mean[2]=0.0; }
                else { a->mean[0]=sr/mn; a->mean[1]=sg/mn; a->mean[2]=sb/mn; a->ok = 1; }
            }
        }
    }
}

__global__ void k_hist(const float4* RESTRICT in, int pitch, int w, int h,
                       float satThr, float blackY, int csIdx, float invBinW,
                       float fraction, PcaArena* RESTRICT a)
{
    const int ok0 = a->ok;
    if (ok0)
    {
        const float lr = cLuma[csIdx][0], lg = cLuma[csIdx][1], lb = cLuma[csIdx][2];
        const float ux = (float)a->mean[0], uy = (float)a->mean[1], uz = (float)a->mean[2];
        const long long total = (long long)w * (long long)h;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
             i += (long long)gridDim.x * blockDim.x)
        {
            const int x = (int)(i % w), y = (int)(i / w);
            float r, g, b; dev_load_rgb(in[(long long)y * pitch + x], r, g, b);
            const float mx  = fmaxf(r, fmaxf(g, b));
            const float lum = lr*r + lg*g + lb*b;
            if (mx < satThr && lum >= blackY)
            {
                int bin = (int)((r*ux + g*uy + b*uz) * invBinW);
                if (bin < 0) bin = 0; else if (bin >= AWB_PCA_BINS) bin = AWB_PCA_BINS - 1;
                atomicAdd(&a->hist[bin], 1u);
            }
        }
    }
    if (threadIdx.x == 0)
    {
        __threadfence();                                    // publish histogram device-wide
        // FOLDED-IN cut: the last block derives the bright/dark thresholds.
        // (atomicInc runs for EVERY block so the count is exact; the work is guarded by ok.)
        if (atomicInc(&a->blocksDone2, gridDim.x) == gridDim.x - 1u && a->ok)
        {
            const long long cnt = (long long)(a->count);
            long long k = (long long)((double)fraction * (double)cnt + 0.5);
            if (k < 1) k = 1;
            if (2*k > cnt) k = cnt / 2;
            if (k < 1) { a->ok = 0; return; }
            long long acc = 0; int darkBin = 0;
            for (int b = 0; b < AWB_PCA_BINS; ++b) { acc += a->hist[b]; if (acc >= k) { darkBin = b; break; } }
            acc = 0; int brightBin = AWB_PCA_BINS - 1;
            for (int b = AWB_PCA_BINS - 1; b >= 0; --b) { acc += a->hist[b]; if (acc >= k) { brightBin = b; break; } }
            const double binW = (double)AWB_PCA_PROJMAX / (double)AWB_PCA_BINS;
            const double darkThr   = (darkBin + 1) * binW;
            const double brightThr = brightBin * binW;
            if (brightThr <= darkThr) { a->ok = 0; return; }
            a->thr[0] = darkThr; a->thr[1] = brightThr;
        }
    }
}

__global__ void k_moment(const float4* RESTRICT in, int pitch, int w, int h,
                         float satThr, float blackY, int csIdx,
                         int illumIdx, int chromIdx, PcaArena* RESTRICT a)
{
    const int ok0 = a->ok;
    double m00=0,m01=0,m02=0,m11=0,m12=0,m22=0;
    if (ok0)
    {
        const float lr = cLuma[csIdx][0], lg = cLuma[csIdx][1], lb = cLuma[csIdx][2];
        const float ux = (float)a->mean[0], uy = (float)a->mean[1], uz = (float)a->mean[2];
        const double darkThr = a->thr[0], brightThr = a->thr[1];
        const long long total = (long long)w * (long long)h;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
             i += (long long)gridDim.x * blockDim.x)
        {
            const int x = (int)(i % w), y = (int)(i / w);
            float r, g, b; dev_load_rgb(in[(long long)y * pitch + x], r, g, b);
            const float mx  = fmaxf(r, fmaxf(g, b));
            const float lum = lr*r + lg*g + lb*b;
            if (mx < satThr && lum >= blackY)
            {
                const double proj = (double)r*ux + (double)g*uy + (double)b*uz;
                if (proj <= darkThr || proj >= brightThr)
                {
                    m00 += (double)r*r; m01 += (double)r*g; m02 += (double)r*b;
                    m11 += (double)g*g; m12 += (double)g*b; m22 += (double)b*b;
                }
            }
        }
    }
    m00 = blockReduceSum(m00); __syncthreads();
    m01 = blockReduceSum(m01); __syncthreads();
    m02 = blockReduceSum(m02); __syncthreads();
    m11 = blockReduceSum(m11); __syncthreads();
    m12 = blockReduceSum(m12); __syncthreads();
    m22 = blockReduceSum(m22); __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAdd(&a->moment[0], m00); atomicAdd(&a->moment[1], m01); atomicAdd(&a->moment[2], m02);
        atomicAdd(&a->moment[3], m11); atomicAdd(&a->moment[4], m12); atomicAdd(&a->moment[5], m22);
        __threadfence();                                    // publish moment partials device-wide
        // FOLDED-IN solve_build: the last block runs Jacobi + builds the CAT matrix.
        if (atomicInc(&a->blocksDone3, gridDim.x) == gridDim.x - 1u)
        {
            a->M[0]=1.f; a->M[1]=0.f; a->M[2]=0.f; a->M[3]=0.f; a->M[4]=1.f; a->M[5]=0.f; a->M[6]=0.f; a->M[7]=0.f; a->M[8]=1.f;
            if (a->ok)
            {
                double A[3][3] = { { a->moment[0], a->moment[1], a->moment[2] },
                                   { a->moment[1], a->moment[3], a->moment[4] },
                                   { a->moment[2], a->moment[4], a->moment[5] } };
                double wv[3], V[3][3];
                dev_jacobi3(A, wv, V);
                int top = 0; if (wv[1] > wv[top]) top = 1; if (wv[2] > wv[top]) top = 2;
                double e[3] = { V[0][top], V[1][top], V[2][top] };
                if (e[0] + e[1] + e[2] < 0.0) { e[0] = -e[0]; e[1] = -e[1]; e[2] = -e[2]; }
                if (e[0] < 0.0 || e[1] < 0.0 || e[2] < 0.0) { a->ok = 0; }
                else {
                    const double en = sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
                    if (en <= 1.0e-12) { a->ok = 0; }
                    else { e[0]/=en; e[1]/=en; e[2]/=en; dev_build_matrix(e, illumIdx, chromIdx, a->M); }
                }
            }
        }
    }
}

__global__ void k_apply(const float4* RESTRICT in, float4* RESTRICT out,
                        int srcPitch, int dstPitch, int w, int h, const PcaArena* RESTRICT a)
{
    const float m0=a->M[0], m1=a->M[1], m2=a->M[2];
    const float m3=a->M[3], m4=a->M[4], m5=a->M[5];
    const float m6=a->M[6], m7=a->M[7], m8=a->M[8];
    const long long total = (long long)w * (long long)h;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x)
    {
        const int x = (int)(i % w), y = (int)(i / w);
        const float4 p = in[(long long)y * srcPitch + x];

        float r, g, b; dev_load_rgb(p, r, g, b);          // linear (decoded unless _GPU_INPUT_LINEAR)
        float nr = m0*r + m1*g + m2*b;
        float ng = m3*r + m4*g + m5*b;
        float nb = m6*r + m7*g + m8*b;
        nr = fmaxf(nr, 0.f); ng = fmaxf(ng, 0.f); nb = fmaxf(nb, 0.f);
#ifndef _GPU_INPUT_LINEAR
        nr = dev_oetf(nr); ng = dev_oetf(ng); nb = dev_oetf(nb);   // linear -> sRGB (matches input encoding)
#endif
#ifndef _GPU_HDR_OUTPUT
        // Clamp to [0,1] as the final op before store -- ON BY DEFAULT, exactly
        // like the gray-point kernel's fminf(...,1.0f). A white balance that
        // boosts a channel pushes bright highlights > 1.0; an unclamped float
        // WRAPS on the downstream 8-bit cast (red wrap -> cyan, blue wrap ->
        // yellow). Define _GPU_HDR_OUTPUT only for a true 32f HDR destination
        // that clamps at its own export stage.
        nr = fminf(nr, 1.f); ng = fminf(ng, 1.f); nb = fminf(nb, 1.f);
#endif
        float4 o;
        o.x = nb; o.y = ng; o.z = nr;   // BGR
        o.w = p.w;                      // alpha copied through, ALWAYS
        out[(long long)y * dstPitch + x] = o;
    }
}

// ============================================================================
//  HOST ENTRY POINT (f32)
// ============================================================================
CUDA_KERNEL_CALL
void ImageLabPCA32_CUDA
(
    const float*  RESTRICT inBuffer,
    float*        RESTRICT outBuffer,
    int                    srcPitch,
    int                    dstPitch,
    int                    width,
    int                    height,
    const AlgoControls* RESTRICT algoGpuParams,
    int                    frameCount,
    cudaStream_t           stream
)
{
    (void)frameCount;
    if (inBuffer == nullptr || outBuffer == nullptr || algoGpuParams == nullptr ||
        width <= 0 || height <= 0)
        return;

    // ---- derive launch scalars from the (host-side) control struct ----
    float pct = algoGpuParams->percentExtremePixels;
    pct = (pct < 1.f) ? 1.f : (pct > 10.f ? 10.f : pct);
    const float fraction = pct * 0.01f;

    float satThr = algoGpuParams->saturationThreshold;
    satThr = (satThr < 0.80f) ? 0.80f : (satThr > 1.00f ? 1.00f : satThr);

    float blackY = algoGpuParams->blackLevelThreshold;
    blackY = (blackY < 0.00f) ? 0.00f : (blackY > 0.10f ? 0.10f : blackY);

    int csIdx = (int)algoGpuParams->colorSpace;
    csIdx = (csIdx < BT601) ? BT601 : (csIdx > SMPTE240M ? SMPTE240M : csIdx);
    int illumIdx = (int)algoGpuParams->illuminate;
    illumIdx = (illumIdx < 0) ? 0 : (illumIdx > 10 ? 10 : illumIdx);
    int chromIdx = (int)algoGpuParams->chromatic;
    chromIdx = (chromIdx < 0) ? 0 : (chromIdx > 4 ? 4 : chromIdx);
    // algoGpuParams->observer is intentionally unused (matches the CPU path).

    const float invBinW = (float)AWB_PCA_BINS / AWB_PCA_PROJMAX;

    // ---- ONE allocation: the arena ----
    PcaArena* d_arena = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_arena), sizeof(PcaArena)) != cudaSuccess)
        return;
    cudaMemsetAsync(d_arena, 0, sizeof(PcaArena), stream);

    const long long total = (long long)width * (long long)height;
    const int  block = 256;
    long long  g = (total + block - 1) / block;
    if (g < 1) g = 1;
    if (g > 65535) g = 65535;
    const int  grid = (int)g;

    const float4* in4  = reinterpret_cast<const float4*>(inBuffer);
    float4*       out4 = reinterpret_cast<float4*>(outBuffer);

    // ---- whole pipeline enqueued on the stream; no host readback ----
    // 4 kernels total. The three former <<<1,1>>> steps (finalize/cut/solve) are
    // folded into the tail of their reduction via a threadfence/last-block barrier,
    // so the host still only enqueues work -- no readback, no extra synchronization.
    k_mean   <<<grid, block, 0, stream>>>(in4, srcPitch, width, height, satThr, blackY, csIdx, d_arena);
    k_hist   <<<grid, block, 0, stream>>>(in4, srcPitch, width, height, satThr, blackY, csIdx, invBinW, fraction, d_arena);
    k_moment <<<grid, block, 0, stream>>>(in4, srcPitch, width, height, satThr, blackY, csIdx, illumIdx, chromIdx, d_arena);
    k_apply  <<<grid, block, 0, stream>>>(in4, out4, srcPitch, dstPitch, width, height, d_arena);

    // ---- release the arena (blocks until queued work completes) ----
    cudaFree(d_arena);

    // required by the test harness so the host may grab the output safely
    cudaDeviceSynchronize();
}

// ============================================================================
//  CLEANUP  (nothing persistent: the arena is allocated and freed per call)
// ============================================================================
CUDA_KERNEL_CALL
void ImageLabPCA_CleanupGPU (void)
{
    // CUDA cleanup -- no persistent device state to release.
}
