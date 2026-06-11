// ============================================================================
//  AwbCorrection.cpp  --  AVX2: scalar CAT matrix build + AVX2 matrix apply
//
//  Same external names / prototypes / includes as the scalar reference.
//  build_correction_matrix_linear runs once per frame -> kept scalar.
//  apply_correction is AVX2 (8 px/iter) + scalar tail for the trailing total&7.
//  No heap allocation; only CACHE_ALIGN stack scratch where needed.
// ============================================================================
#include <cmath>
#include <cfloat>
#include <immintrin.h>               // AVX2 + FMA

#include "AwbCorrection.hpp"          // AwbEstimate + declarations
#include "ColorTransformMatrix.hpp"   // sRGBtoXYZ, XYZtosRGB
#include "AlgCorrectionMatrix.hpp"    // GetIlluminate / GetColorAdaptation / GetColorAdaptationInv

#ifndef RESTRICT
#define RESTRICT __restrict
#endif

// ----------------------------------------------------------------------------
//  build_correction_matrix_linear : scalar, unchanged (once-per-frame 3x3 math)
// ----------------------------------------------------------------------------
void build_correction_matrix_linear
(
    const AwbEstimate&         est,
    const eILLUMINATE          illuminate,
    const eChromaticAdaptation chromatic,
    float                      M[9]
) noexcept
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

    const float Xe = er * sRGBtoXYZ[0] + eg * sRGBtoXYZ[1] + eb * sRGBtoXYZ[2];
    const float Ye = er * sRGBtoXYZ[3] + eg * sRGBtoXYZ[4] + eb * sRGBtoXYZ[5];
    const float Ze = er * sRGBtoXYZ[6] + eg * sRGBtoXYZ[7] + eb * sRGBtoXYZ[8];

    const float sum = Xe + Ye + Ze;
    if (sum <= FLT_EPSILON)
        return;

    const float xe = Xe / sum;
    const float ye = Ye / sum;
    if (ye <= FLT_EPSILON)
        return;

    const float kY = 100.0f / ye;
    const float estXYZ[3] = { kY * xe, 100.0f, kY * (1.0f - xe - ye) };

    const float* RESTRICT tgt  = GetIlluminate(illuminate);
    const float* RESTRICT A     = GetColorAdaptation(chromatic);
    const float* RESTRICT Ainv  = GetColorAdaptationInv(chromatic);

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
            T1[r * 3 + c] = XYZtosRGB[r * 3 + 0] * Mx[0 * 3 + c]
                          + XYZtosRGB[r * 3 + 1] * Mx[1 * 3 + c]
                          + XYZtosRGB[r * 3 + 2] * Mx[2 * 3 + c];

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            M[r * 3 + c] = T1[r * 3 + 0] * sRGBtoXYZ[0 * 3 + c]
                         + T1[r * 3 + 1] * sRGBtoXYZ[1 * 3 + c]
                         + T1[r * 3 + 2] * sRGBtoXYZ[2 * 3 + c];
}

// ----------------------------------------------------------------------------
//  apply_correction : AVX2 (8 px/iter) + scalar tail. Floors negatives at 0,
//  no upper clamp. Element-wise (in-place safe).
// ----------------------------------------------------------------------------
void apply_correction
(
    const float* RESTRICT sR, const float* RESTRICT sG, const float* RESTRICT sB,
    float*       RESTRICT dR, float*       RESTRICT dG, float*       RESTRICT dB,
    const int64_t total,
    const float   M[9]
) noexcept
{
    const __m256 vm0 = _mm256_set1_ps(M[0]), vm1 = _mm256_set1_ps(M[1]), vm2 = _mm256_set1_ps(M[2]);
    const __m256 vm3 = _mm256_set1_ps(M[3]), vm4 = _mm256_set1_ps(M[4]), vm5 = _mm256_set1_ps(M[5]);
    const __m256 vm6 = _mm256_set1_ps(M[6]), vm7 = _mm256_set1_ps(M[7]), vm8 = _mm256_set1_ps(M[8]);
    const __m256 vzero = _mm256_setzero_ps();

    int64_t n = 0;
    for (; n + 8 <= total; n += 8)
    {
        const __m256 vr = _mm256_loadu_ps(sR + n);
        const __m256 vg = _mm256_loadu_ps(sG + n);
        const __m256 vb = _mm256_loadu_ps(sB + n);

        __m256 nr = _mm256_fmadd_ps(vm0, vr, _mm256_fmadd_ps(vm1, vg, _mm256_mul_ps(vm2, vb)));
        __m256 ng = _mm256_fmadd_ps(vm3, vr, _mm256_fmadd_ps(vm4, vg, _mm256_mul_ps(vm5, vb)));
        __m256 nb = _mm256_fmadd_ps(vm6, vr, _mm256_fmadd_ps(vm7, vg, _mm256_mul_ps(vm8, vb)));

        nr = _mm256_max_ps(nr, vzero);
        ng = _mm256_max_ps(ng, vzero);
        nb = _mm256_max_ps(nb, vzero);

        _mm256_storeu_ps(dR + n, nr);
        _mm256_storeu_ps(dG + n, ng);
        _mm256_storeu_ps(dB + n, nb);
    }

    // scalar tail: remaining (total & 7) pixels
    const float m0 = M[0], m1 = M[1], m2 = M[2];
    const float m3 = M[3], m4 = M[4], m5 = M[5];
    const float m6 = M[6], m7 = M[7], m8 = M[8];
    for (; n < total; ++n)
    {
        const float r = sR[n], g = sG[n], b = sB[n];
        const float nr = m0 * r + m1 * g + m2 * b;
        const float ng = m3 * r + m4 * g + m5 * b;
        const float nb = m6 * r + m7 * g + m8 * b;
        dR[n] = (nr > 0.f) ? nr : 0.f;
        dG[n] = (ng > 0.f) ? ng : 0.f;
        dB[n] = (nb > 0.f) ? nb : 0.f;
    }
}
