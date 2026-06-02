#include <cmath>
#include <cfloat>
#include <algorithm>
#include <immintrin.h>      // AVX2 + FMA + POPCNT

#include "AlgCommonFunctions.hpp"
#include "ColorTransformMatrix.hpp"   // sRGBtoXYZ, XYZtosRGB
#include "AlgCorrectionMatrix.hpp"    // GetIlluminate / GetColorAdaptation / GetColorAdaptationInv

#ifndef RESTRICT
#define RESTRICT __restrict
#endif

namespace
{
    // selection guards (scene-linear units)
    constexpr float gBlackGuard = 1.0e-4f;
    constexpr float gWhiteGuard = 0.98f;

    // orthonormal opponent basis (achromatic axis = (1,1,1)/sqrt3)
    constexpr float gInvSqrt2 = 0.70710678118f;
    constexpr float gInvSqrt3 = 0.57735026919f;
    constexpr float gInvSqrt6 = 0.40824829046f;

    // horizontal sum of a __m256d (4 doubles) -> scalar
    inline double hsum_pd(const __m256d v) noexcept
    {
        const __m128d lo = _mm256_castpd256_pd128(v);
        const __m128d hi = _mm256_extractf128_pd(v, 1);
        __m128d s = _mm_add_pd(lo, hi);             // 2 doubles
        s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));   // 1 double
        return _mm_cvtsd_f64(s);
    }

    // widen 8 floats -> doubles and add the two halves: (lo4 + hi4) as __m256d
    inline __m256d ps_lohi_to_pd_sum(const __m256 v) noexcept
    {
        const __m128 lo = _mm256_castps256_ps128(v);
        const __m128 hi = _mm256_extractf128_ps(v, 1);
        return _mm256_add_pd(_mm256_cvtps_pd(lo), _mm256_cvtps_pd(hi));
    }
} // anonymous namespace


// =============================================================================
// AVX2 masked reduction. Accumulators in double (precise over large frames);
// count via movemask + popcount (exact). step > 1 falls back to a scalar strided
// scan (subsampling path).
// =============================================================================
GrayEstimate collect_gray_estimate
(
    const float* RESTRICT R,
    const float* RESTRICT G,
    const float* RESTRICT B,
    const int64_t total,
    const float   threshold,
    const int32_t step
) noexcept
{
    double sumR = 0.0, sumG = 0.0, sumB = 0.0;
    int64_t cnt = 0;

    // shared scalar pixel test (used by tail and by the strided/subsample path)
    auto addScalar = [&](const int64_t n) noexcept
    {
        const float r = R[n], g = G[n], b = B[n];
        const float mx = std::max(r, std::max(g, b));
        const float mn = std::min(r, std::min(g, b));
        const float Y  = (r + g + b)        * gInvSqrt3;
        const float C1 = (r - g)            * gInvSqrt2;
        const float C2 = (r + g - 2.0f * b) * gInvSqrt6;
        const float chroma2 = C1 * C1 + C2 * C2;                  // squared chroma (no sqrt)
        const float Ysafe   = (Y > FLT_EPSILON) ? Y : FLT_EPSILON;
        const float rhs     = threshold * Ysafe;
        if (mx < gWhiteGuard && mn >= gBlackGuard && chroma2 < rhs * rhs)
        {
            sumR += r; sumG += g; sumB += b; ++cnt;
        }
    };

    if (step > 1)
    {
        const int64_t st = static_cast<int64_t>(step);
        for (int64_t n = 0; n < total; n += st)
            addScalar(n);
    }
    else
    {
        const __m256 vWhite  = _mm256_set1_ps(gWhiteGuard);
        const __m256 vBlack  = _mm256_set1_ps(gBlackGuard);
        const __m256 vThresh = _mm256_set1_ps(threshold);
        const __m256 vEps    = _mm256_set1_ps(FLT_EPSILON);
        const __m256 vTwo    = _mm256_set1_ps(2.0f);
        const __m256 vIS2    = _mm256_set1_ps(gInvSqrt2);
        const __m256 vIS3    = _mm256_set1_ps(gInvSqrt3);
        const __m256 vIS6    = _mm256_set1_ps(gInvSqrt6);

        __m256d aR = _mm256_setzero_pd();
        __m256d aG = _mm256_setzero_pd();
        __m256d aB = _mm256_setzero_pd();

        int64_t n = 0;
        const int64_t vend = total & ~static_cast<int64_t>(7);

        for (; n < vend; n += 8)
        {
            const __m256 r = _mm256_loadu_ps(R + n);
            const __m256 g = _mm256_loadu_ps(G + n);
            const __m256 b = _mm256_loadu_ps(B + n);

            const __m256 mx = _mm256_max_ps(r, _mm256_max_ps(g, b));
            const __m256 mn = _mm256_min_ps(r, _mm256_min_ps(g, b));

            const __m256 rg  = _mm256_add_ps(r, g);
            const __m256 Y   = _mm256_mul_ps(_mm256_add_ps(rg, b), vIS3);
            const __m256 C1  = _mm256_mul_ps(_mm256_sub_ps(r, g), vIS2);
            const __m256 C2  = _mm256_mul_ps(_mm256_sub_ps(rg, _mm256_mul_ps(vTwo, b)), vIS6);

            const __m256 chroma2 = _mm256_fmadd_ps(C1, C1, _mm256_mul_ps(C2, C2)); // squared (no sqrt)
            const __m256 Ysafe   = _mm256_max_ps(Y, vEps);
            const __m256 rhs     = _mm256_mul_ps(vThresh, Ysafe);
            const __m256 rhs2    = _mm256_mul_ps(rhs, rhs);

            const __m256 mHi   = _mm256_cmp_ps(mx, vWhite, _CMP_LT_OQ);      // not clipped
            const __m256 mLo   = _mm256_cmp_ps(mn, vBlack, _CMP_GE_OQ);      // not black
            const __m256 mGr   = _mm256_cmp_ps(chroma2, rhs2, _CMP_LT_OQ);   // near gray (squared)
            const __m256 keep  = _mm256_and_ps(_mm256_and_ps(mHi, mLo), mGr);

            aR = _mm256_add_pd(aR, ps_lohi_to_pd_sum(_mm256_and_ps(keep, r)));
            aG = _mm256_add_pd(aG, ps_lohi_to_pd_sum(_mm256_and_ps(keep, g)));
            aB = _mm256_add_pd(aB, ps_lohi_to_pd_sum(_mm256_and_ps(keep, b)));

            cnt += static_cast<int64_t>(_mm_popcnt_u32(
                       static_cast<unsigned>(_mm256_movemask_ps(keep))));
        }

        sumR = hsum_pd(aR);
        sumG = hsum_pd(aG);
        sumB = hsum_pd(aB);

        for (; n < total; ++n)   // scalar tail
            addScalar(n);
    }

    GrayEstimate e;
    e.sum.R = sumR;
    e.sum.G = sumG;
    e.sum.B = sumB;
    e.count = cnt;
    return e;
}

// =============================================================================
// Scalar (runs once per frame; not worth vectorizing). Unchanged.
// =============================================================================
void build_correction_matrix_linear
(
    const GrayEstimate& est,
    const AlgoControls& ctrl,
    float M[9]
) noexcept
{
    M[0] = 1.f; M[1] = 0.f; M[2] = 0.f;
    M[3] = 0.f; M[4] = 1.f; M[5] = 0.f;
    M[6] = 0.f; M[7] = 0.f; M[8] = 1.f;

    if (est.count <= 0)
        return;

    const float inv = 1.0f / static_cast<float>(est.count);
    const float er = static_cast<float>(est.sum.R) * inv;
    const float eg = static_cast<float>(est.sum.G) * inv;
    const float eb = static_cast<float>(est.sum.B) * inv;

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

    const float* RESTRICT tgt  = GetIlluminate(ctrl.illuminate);
    const float* RESTRICT A    = GetColorAdaptation(ctrl.chromatic);
    const float* RESTRICT Ainv = GetColorAdaptationInv(ctrl.chromatic);

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

// =============================================================================
// AVX2 map: 3x3 matrix per pixel, 8 pixels/iter, 3 FMA chains. Scalar tail.
// Negatives floored at 0; no upper clamp (scene-linear / HDR safe).
// =============================================================================
void apply_correction
(
    const float* RESTRICT sR, const float* RESTRICT sG, const float* RESTRICT sB,
    float*       RESTRICT dR, float*       RESTRICT dG, float*       RESTRICT dB,
    const int64_t total,
    const float   M[9]
) noexcept
{
    const __m256 m0 = _mm256_set1_ps(M[0]);
    const __m256 m1 = _mm256_set1_ps(M[1]);
    const __m256 m2 = _mm256_set1_ps(M[2]);
    const __m256 m3 = _mm256_set1_ps(M[3]);
    const __m256 m4 = _mm256_set1_ps(M[4]);
    const __m256 m5 = _mm256_set1_ps(M[5]);
    const __m256 m6 = _mm256_set1_ps(M[6]);
    const __m256 m7 = _mm256_set1_ps(M[7]);
    const __m256 m8 = _mm256_set1_ps(M[8]);
    const __m256 vz = _mm256_setzero_ps();

    int64_t n = 0;
    const int64_t vend = total & ~static_cast<int64_t>(7);

    for (; n < vend; n += 8)
    {
        const __m256 r = _mm256_loadu_ps(sR + n);
        const __m256 g = _mm256_loadu_ps(sG + n);
        const __m256 b = _mm256_loadu_ps(sB + n);

        const __m256 nr = _mm256_fmadd_ps(r, m0, _mm256_fmadd_ps(g, m1, _mm256_mul_ps(b, m2)));
        const __m256 ng = _mm256_fmadd_ps(r, m3, _mm256_fmadd_ps(g, m4, _mm256_mul_ps(b, m5)));
        const __m256 nb = _mm256_fmadd_ps(r, m6, _mm256_fmadd_ps(g, m7, _mm256_mul_ps(b, m8)));

        _mm256_storeu_ps(dR + n, _mm256_max_ps(nr, vz));
        _mm256_storeu_ps(dG + n, _mm256_max_ps(ng, vz));
        _mm256_storeu_ps(dB + n, _mm256_max_ps(nb, vz));
    }

    for (; n < total; ++n)   // scalar tail
    {
        const float r = sR[n], g = sG[n], b = sB[n];
        const float nr = M[0] * r + M[1] * g + M[2] * b;
        const float ng = M[3] * r + M[4] * g + M[5] * b;
        const float nb = M[6] * r + M[7] * g + M[8] * b;
        dR[n] = (nr > 0.f) ? nr : 0.f;
        dG[n] = (ng > 0.f) ? ng : 0.f;
        dB[n] = (nb > 0.f) ? nb : 0.f;
    }
}