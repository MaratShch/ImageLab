// ============================================================================
//  AlgorithmMain_PCA.cpp   --  AVX2 PCA illuminant estimation
//
//  AVX2 acceleration of the Cheng bright-and-dark PCA estimator. Same external
//  name / prototype / includes as the scalar reference. pca_estimate is AVX2
//  (8 px/iter) with scalar tails for the trailing total&7 pixels; the histogram
//  increment is a scalar per-8 scatter (AVX2 has no scatter). No heap allocation;
//  CACHE_ALIGN stack scratch only. build_correction_matrix_linear stays scalar.
//
//    colorSpace -> selects luminance weights (RGB2YUV luma row).
// ============================================================================
#ifdef _MSC_VER
    // Microsoft Visual Studio specific
    #include <intrin.h>
#endif

#include <cstdint>
#include <cstring>
#include <cmath>
#include <immintrin.h>      // AVX2 + FMA

#include "Common.hpp"
#include "AlgoControl.hpp"
#include "AlgorithmMain.hpp"
#include "AwbCorrection.hpp"   // AwbEstimate, build_correction_matrix_linear, apply_correction

namespace
{
    constexpr int    gPcaBins  = 4096;   // projection-histogram resolution
    constexpr double gProjMax  = 1.74;   // max c.u for channels < 1 (~sqrt(3))

    inline float clampf(float v, float lo, float hi) noexcept
    { return v < lo ? lo : (v > hi ? hi : v); }

    inline float chmax(float r, float g, float b) noexcept
    { return r > g ? (r > b ? r : b) : (g > b ? g : b); }

    // horizontal sum of an __m256 accumulated in double (precision-safe)
    inline double hsum_pd(const __m256 v) noexcept
    {
        CACHE_ALIGN float t[8];
        _mm256_store_ps(t, v);
        return (double)t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
    }

#ifdef _MSC_VER
    // Microsoft Visual Studio specific
    inline int popcount32(unsigned int x) noexcept {return __popcnt(x);}
#else
    // GCC / Clang specific
    inline int popcount32(unsigned int x) noexcept {return __builtin_popcount(x);}
#endif

    // cyclic Jacobi eigensolver for a real symmetric 3x3 (scalar).
    void jacobi3(double A[3][3], double w[3], double V[3][3]) noexcept
    {
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) V[i][j] = (i == j) ? 1.0 : 0.0;
        for (int sweep = 0; sweep < 12; ++sweep)
        {
            const double off = std::fabs(A[0][1]) + std::fabs(A[0][2]) + std::fabs(A[1][2]);
            if (off < 1.0e-20) break;
            for (int p = 0; p < 2; ++p)
                for (int q = p + 1; q < 3; ++q)
                {
                    if (std::fabs(A[p][q]) < 1.0e-300) continue;
                    const double phi = 0.5 * std::atan2(2.0 * A[p][q], A[q][q] - A[p][p]);
                    const double c = std::cos(phi), s = std::sin(phi);
                    for (int k = 0; k < 3; ++k) { const double a = A[k][p], b = A[k][q]; A[k][p] = c*a - s*b; A[k][q] = s*a + c*b; }
                    for (int k = 0; k < 3; ++k) { const double a = A[p][k], b = A[q][k]; A[p][k] = c*a - s*b; A[q][k] = s*a + c*b; }
                    for (int k = 0; k < 3; ++k) { const double a = V[k][p], b = V[k][q]; V[k][p] = c*a - s*b; V[k][q] = s*a + c*b; }
                }
        }
        w[0] = A[0][0]; w[1] = A[1][1]; w[2] = A[2][2];
    }

    // AVX2 Cheng bright-and-dark PCA on planar LINEAR RGB (contiguous, length total).
    bool pca_estimate(const float* RESTRICT R, const float* RESTRICT G, const float* RESTRICT B,
                      const int64_t total, const float fraction, const float satThr,
                      const float blackY, const float lumaR, const float lumaG, const float lumaB,
                      double e_out[3]) noexcept
    {
        const __m256 vsat = _mm256_set1_ps(satThr);
        const __m256 vblk = _mm256_set1_ps(blackY);
        const __m256 vlr  = _mm256_set1_ps(lumaR);
        const __m256 vlg  = _mm256_set1_ps(lumaG);
        const __m256 vlb  = _mm256_set1_ps(lumaB);

        // ---- pass 1: unit mean direction over valid pixels -----------------
        __m256 aR = _mm256_setzero_ps(), aG = _mm256_setzero_ps(), aB = _mm256_setzero_ps();
        int64_t cnt = 0;
        int64_t i = 0;
        for (; i + 8 <= total; i += 8)
        {
            const __m256 vr = _mm256_loadu_ps(R + i), vg = _mm256_loadu_ps(G + i), vb = _mm256_loadu_ps(B + i);
            const __m256 vmax = _mm256_max_ps(_mm256_max_ps(vr, vg), vb);
            const __m256 vlum = _mm256_fmadd_ps(vlr, vr, _mm256_fmadd_ps(vlg, vg, _mm256_mul_ps(vlb, vb)));
            const __m256 mk = _mm256_and_ps(_mm256_cmp_ps(vmax, vsat, _CMP_LT_OQ),
                                            _mm256_cmp_ps(vlum, vblk, _CMP_GE_OQ));
            aR = _mm256_add_ps(aR, _mm256_and_ps(vr, mk));
            aG = _mm256_add_ps(aG, _mm256_and_ps(vg, mk));
            aB = _mm256_add_ps(aB, _mm256_and_ps(vb, mk));
            cnt += popcount32(static_cast<unsigned>(_mm256_movemask_ps(mk)));
        }
        double sr = hsum_pd(aR), sg = hsum_pd(aG), sb = hsum_pd(aB);
        for (; i < total; ++i)
        {
            const float r = R[i], g = G[i], b = B[i];
            if (chmax(r, g, b) >= satThr) continue;
            if (lumaR*r + lumaG*g + lumaB*b < blackY) continue;
            sr += r; sg += g; sb += b; ++cnt;
        }
        if (cnt < 16) return false;
        const double mnrm = std::sqrt(sr*sr + sg*sg + sb*sb);
        if (mnrm <= 1.0e-12) return false;
        const double ux = sr / mnrm, uy = sg / mnrm, uz = sb / mnrm;

        // ---- pass 2: projection histogram (vector proj, scalar scatter) ----
        const double invBinW = static_cast<double>(gPcaBins) / gProjMax;
        CACHE_ALIGN uint32_t hist[gPcaBins];
        std::memset(hist, 0, sizeof(hist));
        const __m256 vux = _mm256_set1_ps(static_cast<float>(ux));
        const __m256 vuy = _mm256_set1_ps(static_cast<float>(uy));
        const __m256 vuz = _mm256_set1_ps(static_cast<float>(uz));
        const __m256 vibw = _mm256_set1_ps(static_cast<float>(invBinW));
        i = 0;
        for (; i + 8 <= total; i += 8)
        {
            const __m256 vr = _mm256_loadu_ps(R + i), vg = _mm256_loadu_ps(G + i), vb = _mm256_loadu_ps(B + i);
            const __m256 vmax = _mm256_max_ps(_mm256_max_ps(vr, vg), vb);
            const __m256 vlum = _mm256_fmadd_ps(vlr, vr, _mm256_fmadd_ps(vlg, vg, _mm256_mul_ps(vlb, vb)));
            const __m256 mk = _mm256_and_ps(_mm256_cmp_ps(vmax, vsat, _CMP_LT_OQ),
                                            _mm256_cmp_ps(vlum, vblk, _CMP_GE_OQ));
            const __m256 vproj = _mm256_fmadd_ps(vux, vr, _mm256_fmadd_ps(vuy, vg, _mm256_mul_ps(vuz, vb)));
            CACHE_ALIGN float bb[8];
            _mm256_store_ps(bb, _mm256_mul_ps(vproj, vibw));
            const int mbits = _mm256_movemask_ps(mk);
            for (int l = 0; l < 8; ++l)
            {
                if (mbits & (1 << l))
                {
                    int bin = static_cast<int>(bb[l]);
                    if (bin < 0) bin = 0; else if (bin >= gPcaBins) bin = gPcaBins - 1;
                    ++hist[bin];
                }
            }
        }
        for (; i < total; ++i)
        {
            const float r = R[i], g = G[i], b = B[i];
            if (chmax(r, g, b) >= satThr) continue;
            if (lumaR*r + lumaG*g + lumaB*b < blackY) continue;
            int bin = static_cast<int>((r*ux + g*uy + b*uz) * invBinW);
            if (bin < 0) bin = 0; else if (bin >= gPcaBins) bin = gPcaBins - 1;
            ++hist[bin];
        }

        // ---- bright/dark cut ----------------------------------------------
        int64_t k = static_cast<int64_t>(static_cast<double>(fraction) * cnt + 0.5);
        if (k < 1) k = 1;
        if (2 * k > cnt) k = cnt / 2;
        if (k < 1) return false;
        int64_t acc = 0; int darkBin = 0;
        for (int b = 0; b < gPcaBins; ++b) { acc += hist[b]; if (acc >= k) { darkBin = b; break; } }
        acc = 0; int brightBin = gPcaBins - 1;
        for (int b = gPcaBins - 1; b >= 0; --b) { acc += hist[b]; if (acc >= k) { brightBin = b; break; } }
        const double binW = gProjMax / static_cast<double>(gPcaBins);
        const double darkThr   = (darkBin + 1) * binW;
        const double brightThr = brightBin * binW;
        if (brightThr <= darkThr) return false;

        // ---- pass 3: UNcentered second moment over selected extremes -------
        const __m256 vdark = _mm256_set1_ps(static_cast<float>(darkThr));
        const __m256 vbrt  = _mm256_set1_ps(static_cast<float>(brightThr));
        __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps(), s02 = _mm256_setzero_ps();
        __m256 s11 = _mm256_setzero_ps(), s12 = _mm256_setzero_ps(), s22 = _mm256_setzero_ps();
        i = 0;
        for (; i + 8 <= total; i += 8)
        {
            const __m256 vr = _mm256_loadu_ps(R + i), vg = _mm256_loadu_ps(G + i), vb = _mm256_loadu_ps(B + i);
            const __m256 vmax = _mm256_max_ps(_mm256_max_ps(vr, vg), vb);
            const __m256 vlum = _mm256_fmadd_ps(vlr, vr, _mm256_fmadd_ps(vlg, vg, _mm256_mul_ps(vlb, vb)));
            const __m256 mkValid = _mm256_and_ps(_mm256_cmp_ps(vmax, vsat, _CMP_LT_OQ),
                                                 _mm256_cmp_ps(vlum, vblk, _CMP_GE_OQ));
            const __m256 vproj = _mm256_fmadd_ps(vux, vr, _mm256_fmadd_ps(vuy, vg, _mm256_mul_ps(vuz, vb)));
            const __m256 sel = _mm256_or_ps(_mm256_cmp_ps(vproj, vdark, _CMP_LE_OQ),
                                            _mm256_cmp_ps(vproj, vbrt,  _CMP_GE_OQ));
            const __m256 mk = _mm256_and_ps(mkValid, sel);
            const __m256 rM = _mm256_and_ps(vr, mk), gM = _mm256_and_ps(vg, mk), bM = _mm256_and_ps(vb, mk);
            s00 = _mm256_fmadd_ps(rM, rM, s00); s01 = _mm256_fmadd_ps(rM, gM, s01); s02 = _mm256_fmadd_ps(rM, bM, s02);
            s11 = _mm256_fmadd_ps(gM, gM, s11); s12 = _mm256_fmadd_ps(gM, bM, s12); s22 = _mm256_fmadd_ps(bM, bM, s22);
        }
        double S00 = hsum_pd(s00), S01 = hsum_pd(s01), S02 = hsum_pd(s02);
        double S11 = hsum_pd(s11), S12 = hsum_pd(s12), S22 = hsum_pd(s22);
        for (; i < total; ++i)
        {
            const float r = R[i], g = G[i], b = B[i];
            if (chmax(r, g, b) >= satThr) continue;
            if (lumaR*r + lumaG*g + lumaB*b < blackY) continue;
            const double proj = r*ux + g*uy + b*uz;
            if (proj <= darkThr || proj >= brightThr)
            {
                S00 += (double)r*r; S01 += (double)r*g; S02 += (double)r*b;
                S11 += (double)g*g; S12 += (double)g*b; S22 += (double)b*b;
            }
        }

        double A[3][3] = { {S00,S01,S02}, {S01,S11,S12}, {S02,S12,S22} };
        double w[3], V[3][3];
        jacobi3(A, w, V);
        int top = 0; if (w[1] > w[top]) top = 1; if (w[2] > w[top]) top = 2;
        double e[3] = { V[0][top], V[1][top], V[2][top] };

        if (e[0] + e[1] + e[2] < 0.0) { e[0] = -e[0]; e[1] = -e[1]; e[2] = -e[2]; }
        if (e[0] < 0.0 || e[1] < 0.0 || e[2] < 0.0) return false;
        const double en = std::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
        if (en <= 1.0e-12) return false;
        e_out[0] = e[0]/en; e_out[1] = e[1]/en; e_out[2] = e[2]/en;
        return true;
    }
}

// ============================================================================
void Algorithm_Main
(
    const MemHandler&     memHandler,
    const int32_t         sizeX,
    const int32_t         sizeY,
    const AlgoControls&   params
) noexcept
{
    if (!mem_handler_valid(memHandler) || sizeX <= 0 || sizeY <= 0)
        return;

    const int64_t total = static_cast<int64_t>(sizeX) * static_cast<int64_t>(sizeY);

    const float* RESTRICT iR = memHandler.input.R;
    const float* RESTRICT iG = memHandler.input.G;
    const float* RESTRICT iB = memHandler.input.B;
    float* RESTRICT oR = memHandler.output.R;
    float* RESTRICT oG = memHandler.output.G;
    float* RESTRICT oB = memHandler.output.B;

    const float fraction = clampf(params.percentExtremePixels, 1.0f, 10.0f) * 0.01f;
    const float satThr   = clampf(params.saturationThreshold,  0.80f, 1.00f);
    const float blackY   = clampf(params.blackLevelThreshold,  0.00f, 0.10f);

    // luminance weights follow the selected colorSpace (RGB2YUV luma row)
    int csIdx = static_cast<int>(params.colorSpace);
    if (csIdx < BT601) csIdx = BT601; else if (csIdx > SMPTE240M) csIdx = SMPTE240M;
    const float* RESTRICT yw = RGB2YUV[csIdx];
    const float lumaR = yw[0], lumaG = yw[1], lumaB = yw[2];

    AwbEstimate est;
    double e[3];
    if (pca_estimate(iR, iG, iB, total, fraction, satThr, blackY, lumaR, lumaG, lumaB, e))
    {
        est.sumR = e[0]; est.sumG = e[1]; est.sumB = e[2]; est.count = 1;
    }
    else
    {
        est.sumR = est.sumG = est.sumB = 0.0; est.count = 0;
    }

    float M[9];
    build_correction_matrix_linear(est, params.illuminate, params.chromatic, M);
    apply_correction(iR, iG, iB, oR, oG, oB, total, M);
}
