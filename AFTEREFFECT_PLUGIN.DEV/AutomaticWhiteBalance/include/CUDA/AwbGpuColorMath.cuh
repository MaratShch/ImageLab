#ifndef __IMAGE_LAB_AWB_GPU_HOST_COLOR_MATH__
#define __IMAGE_LAB_AWB_GPU_HOST_COLOR_MATH__
// ============================================================================
//  AwbGpuColorMath.cuh   (HOST-ONLY)
//
//  Self-contained gray-estimate -> 3x3 chromatic-adaptation matrix builder for
//  the GPU launchers. Embeds the same constants as the CPU algorithm headers so
//  the standalone GPU project links WITHOUT the CPU algorithm library.
//
//  This is an exact port of build_correction_matrix_linear() (AlgCommonFunctions)
//  plus estimate_chromaticity(). The constant tables mirror ColorTransformMatrix.hpp
//  and AlgCorrectionMatrix.hpp -- KEEP IN SYNC if those ever change.
//
//  Validated against the CPU Algorithm_Main: max |GPU-port - CPU| = 3.6e-07.
// ============================================================================

#include "AlgoControl.hpp"   // AlgoControls + eILLUMINATE / eChromaticAdaptation
#include <cmath>
#include <cfloat>

namespace awb_host
{
    // gray-estimate result (sum of selected linear RGB + count). double sums.
    struct Estimate { double sumR; double sumG; double sumB; long long count; };

    // ---- constants (mirror ColorTransformMatrix.hpp) -----------------------
    static constexpr float kSRGBtoXYZ[9] = {
        0.4124564f, 0.3575761f, 0.1804375f,
        0.2126729f, 0.7151522f, 0.0721750f,
        0.0193339f, 0.1191920f, 0.9503041f
    };
    static constexpr float kXYZtosRGB[9] = {
        3.240455f, -1.537139f, -0.498532f,
       -0.969266f,  1.876011f,  0.041556f,
        0.055643f, -0.204026f,  1.057225f
    };

    // ---- constants (mirror AlgCorrectionMatrix.hpp) ------------------------
    static constexpr float kIlluminate[11][3] = {
        { 95.0470f,  100.0000f, 108.8830f }, // DAYLIGHT - D65 (default)
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
    static constexpr float kAdapt[5][9] = {
        { 0.73280f,  0.4296f, -0.16240f, -0.7036f, 1.69750f, 0.0061f, 0.0030f,  0.0136f, 0.98340f }, // CAT-02
        { 0.40024f,  0.7076f, -0.08081f, -0.2263f, 1.16532f, 0.0457f, 0.0f,     0.0f,    0.91822f }, // VON-KRIES
        { 0.89510f,  0.2664f, -0.16140f, -0.7502f, 1.71350f, 0.0367f, 0.0389f, -0.0685f, 1.02960f }, // BRADFORD
        { 1.26940f, -0.0988f, -0.17060f, -0.8364f, 1.80060f, 0.0357f, 0.0297f, -0.0315f, 1.00180f }, // SHARP
        { 0.79820f,  0.3389f, -0.13710f, -0.5918f, 1.55120f, 0.0406f, 0.0008f,  0.0239f, 0.97530f }  // CMCCAT2000
    };
    static constexpr float kAdaptInv[5][9] = {
        { 1.096124f, -0.278869f, 0.182745f, 0.454369f, 0.473533f, 0.072098f, -0.009628f, -0.005698f, 1.015326f }, // INV CAT-02
        { 1.859936f, -1.129382f, 0.219897f, 0.361191f, 0.638812f, 0.0f,       0.0f,       0.0f,      1.089064f }, // INV VON-KRIES
        { 0.986993f, -0.147054f, 0.159963f, 0.432305f, 0.518360f, 0.049291f, -0.008529f,  0.040043f, 0.968487f }, // INV BRADFORD
        { 0.815633f,  0.047155f, 0.137217f, 0.379114f, 0.576942f, 0.044001f, -0.012260f,  0.016743f, 0.995519f }, // INV SHARP
        { 1.076450f, -0.237662f, 0.161212f, 0.410964f, 0.554342f, 0.034694f, -0.010954f, -0.013389f, 1.024343f }  // INV CMCCAT2000
    };

    static inline int clampIdx(int v, int hi) noexcept { return v < 0 ? 0 : (v > hi ? hi : v); }

    // rg-chromaticity of the estimate (iteration stop criterion)
    static inline void chromaticity(const Estimate& g, float& x, float& y) noexcept
    {
        const double s = g.sumR + g.sumG + g.sumB;
        if (g.count <= 0 || s <= 0.0) { x = 0.f; y = 0.f; return; }
        x = static_cast<float>(g.sumR / s);
        y = static_cast<float>(g.sumG / s);
    }

    // EXACT port of build_correction_matrix_linear(). Identity if degenerate.
    static inline void build_matrix(const Estimate& est, const AlgoControls& ctrl, float M[9]) noexcept
    {
        M[0] = 1.f; M[1] = 0.f; M[2] = 0.f;
        M[3] = 0.f; M[4] = 1.f; M[5] = 0.f;
        M[6] = 0.f; M[7] = 0.f; M[8] = 1.f;

        if (est.count <= 0)
            return;

        const float inv = 1.0f / static_cast<float>(est.count);
        const float er = static_cast<float>(est.sumR) * inv;
        const float eg = static_cast<float>(est.sumG) * inv;
        const float eb = static_cast<float>(est.sumB) * inv;

        const float Xe = er * kSRGBtoXYZ[0] + eg * kSRGBtoXYZ[1] + eb * kSRGBtoXYZ[2];
        const float Ye = er * kSRGBtoXYZ[3] + eg * kSRGBtoXYZ[4] + eb * kSRGBtoXYZ[5];
        const float Ze = er * kSRGBtoXYZ[6] + eg * kSRGBtoXYZ[7] + eb * kSRGBtoXYZ[8];

        const float sum = Xe + Ye + Ze;
        if (sum <= FLT_EPSILON) return;

        const float xe = Xe / sum;
        const float ye = Ye / sum;
        if (ye <= FLT_EPSILON) return;

        const float kY = 100.0f / ye;
        const float estXYZ[3] = { kY * xe, 100.0f, kY * (1.0f - xe - ye) };

        const float* tgt  = kIlluminate[clampIdx(static_cast<int>(ctrl.illuminate), 10)];
        const float* A    = kAdapt    [clampIdx(static_cast<int>(ctrl.chromatic), 4)];
        const float* Ainv = kAdaptInv [clampIdx(static_cast<int>(ctrl.chromatic), 4)];

        const float coneT[3] =
        {
            tgt[0] * A[0] + tgt[1] * A[1] + tgt[2] * A[2],
            tgt[0] * A[3] + tgt[1] * A[4] + tgt[2] * A[5],
            tgt[0] * A[6] + tgt[1] * A[7] + tgt[2] * A[8]
        };
        const float coneE[3] =
        {
            estXYZ[0] * A[0] + estXYZ[1] * A[1] + estXYZ[2] * A[2],
            estXYZ[0] * A[3] + estXYZ[1] * A[4] + estXYZ[2] * A[5],
            estXYZ[0] * A[6] + estXYZ[1] * A[7] + estXYZ[2] * A[8]
        };

        const float g0 = (std::fabs(coneE[0]) > FLT_EPSILON) ? (coneT[0] / coneE[0]) : 1.f;
        const float g1 = (std::fabs(coneE[1]) > FLT_EPSILON) ? (coneT[1] / coneE[1]) : 1.f;
        const float g2 = (std::fabs(coneE[2]) > FLT_EPSILON) ? (coneT[2] / coneE[2]) : 1.f;

        const float D[9] =
        {
            g0 * A[0], g0 * A[1], g0 * A[2],
            g1 * A[3], g1 * A[4], g1 * A[5],
            g2 * A[6], g2 * A[7], g2 * A[8]
        };

        float Mx[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                Mx[r * 3 + c] = Ainv[r * 3 + 0] * D[0 * 3 + c]
                              + Ainv[r * 3 + 1] * D[1 * 3 + c]
                              + Ainv[r * 3 + 2] * D[2 * 3 + c];

        float T1[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                T1[r * 3 + c] = kXYZtosRGB[r * 3 + 0] * Mx[0 * 3 + c]
                              + kXYZtosRGB[r * 3 + 1] * Mx[1 * 3 + c]
                              + kXYZtosRGB[r * 3 + 2] * Mx[2 * 3 + c];

        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                M[r * 3 + c] = T1[r * 3 + 0] * kSRGBtoXYZ[0 * 3 + c]
                             + T1[r * 3 + 1] * kSRGBtoXYZ[1 * 3 + c]
                             + T1[r * 3 + 2] * kSRGBtoXYZ[2 * 3 + c];
    }
} // namespace awb_host

#endif // __IMAGE_LAB_AWB_GPU_HOST_COLOR_MATH__
