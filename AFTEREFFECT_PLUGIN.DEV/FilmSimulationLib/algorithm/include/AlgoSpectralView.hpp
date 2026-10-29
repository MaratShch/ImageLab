// ---------------------------------------------------------------------------
//  AlgoSpectralView.hpp  --  shared by the scalar and AVX2 engines
//
//  2026-10-27c (owner: "the dye spectra -- add immediately to database and use
//  it in algorithm"). Twin of pg/spectral_view.py, which is the reference and
//  documents the physics in full. Summary:
//
//  A colour TRANSPARENCY carrying its own traced three-dye panel
//  (SpectralDyeDensity: cyan, magenta, yellow) is VIEWED through those dyes
//  instead of having its three status-A densities shown one channel at a time
//  with stage 12's 3x3 dye_matrix standing in for the crosstalk:
//
//    1. record densities (STATUS densities) -> dye amounts a, so that the
//       status densities of T = 10^-(a . dyes) equal them (Newton, exactly
//       ALGO_SPECVIEW_NEWTON_ITERS steps from the analytical dye-peak solve,
//       exact Jacobian; no tolerance test);
//    2. T(lambda) under CIE D55, CIE 1931 2 deg, Bradford-adapted to D65,
//       linear sRGB -- one 3 x 31 matrix (ALGO_SPECVIEW_VIEW_*), normalised so
//       a clear film is (1, 1, 1);
//    3. relative to the clear film (D-min), as an EQUIVALENT density per
//       channel, scaled so the anchored mid grey keeps its density:
//           D'_c = D-min_c + scale_c * -log10(max(rel_c, floor_c)).
//
//  The viewed RGB is tabulated per frame on a 33^3 grid over [D-min, D-max]
//  in HighPrecType and read by trilinear interpolation. Stage 12 does NOT
//  apply dye_matrix to these stocks (AlgoPass12), and the reversal anchor
//  solve uses the identity in its place (Algo_08_Sim.cpp, both twins).
//
//  The tables below are the Python reference's own arrays printed with repr()
//  -- iso_5_3_status responses on 400..700 nm / 10 nm and spectral_view.VIEW --
//  and verify.py G-V70-SPECVIEW-TWIN asserts they are equal.
// ---------------------------------------------------------------------------
#ifndef ALGO_SPECTRAL_VIEW_HPP
#define ALGO_SPECTRAL_VIEW_HPP

#include <cmath>
#include <cstdint>
#include <vector>

#include "AlgoTypes.hpp"
#include "film_profiles.hpp"
#include "AlgoCharacteristicCurve.hpp"
#include "AlgoCallier.hpp"

constexpr int32_t ALGO_SPECVIEW_NLAM          = 31;     // 400..700 nm / 10 nm
constexpr int32_t ALGO_SPECVIEW_LUT_N         = 33;
constexpr int32_t ALGO_SPECVIEW_NEWTON_ITERS  = 8;
constexpr HighPrecType ALGO_SPECVIEW_LAM0     = 400.0;
constexpr HighPrecType ALGO_SPECVIEW_DLAM     = 10.0;

constexpr HighPrecType ALGO_SPECVIEW_STATUS_A_R[ALGO_SPECVIEW_NLAM] =
{
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.4723125024327164e-08, 7.379042301291015e-06, 0.003698281797802663, 0.4345102241715714, 1.0, 0.7430191378967022,
    0.40179081084894, 0.19319683170169222, 0.07943282347242814, 0.03556313185689854, 0.014621771744567183, 0.005970352865838366,
    0.00241546083444494
};

constexpr HighPrecType ALGO_SPECVIEW_STATUS_A_G[ALGO_SPECVIEW_NLAM] =
{
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.7782794100389228e-08, 2.818382931264449e-06, 0.00044668359215096305, 0.06637430704019089,
    0.6053408747539136, 1.0, 0.805378441199066, 0.44055486350655343, 0.166341265037017, 0.040644332916521274,
    0.005834451042737447, 0.00037931498497368167, 7.568328950209736e-06, 1.5100801541641463e-07, 3.0130060241861156e-09, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0
};

constexpr HighPrecType ALGO_SPECVIEW_STATUS_A_B[ALGO_SPECVIEW_NLAM] =
{
    1.0046157902783966e-09, 6.338697112569263e-06, 0.03999447497610974, 0.6591738952443212, 1.0, 0.8165823713585922,
    0.4168693834703355, 0.1096478196143185, 0.009749896377173866, 0.0003681289736425313, 1.4621771744567184e-05, 5.82103217770871e-07,
    2.31739464996848e-08, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0
};

constexpr HighPrecType ALGO_SPECVIEW_STATUS_M_R[ALGO_SPECVIEW_NLAM] =
{
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 8.109610578538388e-09, 3.228494121712638e-06, 0.0012852866599436154, 0.3013006024186122,
    1.0, 0.7925013304804718, 0.3784425847170936, 0.17864875748520498, 0.07498942093324558, 0.03097419299216581,
    0.012560299636948755
};

constexpr HighPrecType ALGO_SPECVIEW_STATUS_M_G[ALGO_SPECVIEW_NLAM] =
{
    0.0, 0.0, 0.0, 8.165823713585906e-09, 9.375620069258812e-08, 1.076465213629835e-06,
    1.2359474334445094e-05, 0.00014190575216890925, 0.00161064563517827, 0.014321878992735439, 0.06367955209079156, 0.18706821403658017,
    0.4226686142656033, 0.7447319739059889, 1.0, 0.9885530946569391, 0.6576578373554199, 0.2870780582024692,
    0.08222426499470711, 0.014859356422870075, 0.0017338039977541375, 0.00011748975549395303, 7.413102413009177e-06, 4.677351412871981e-07,
    2.95120922666639e-08, 1.8620871366628656e-09, 0.0, 0.0, 0.0, 0.0,
    0.0
};

constexpr HighPrecType ALGO_SPECVIEW_STATUS_M_B[ALGO_SPECVIEW_NLAM] =
{
    4.008667176273026e-06, 0.0012676518658578462, 0.12912192736135333, 0.4285485203974392, 0.7430191378967022, 1.0,
    0.9015711376059571, 0.5533501092157371, 0.2202926463053456, 0.055335010921573655, 0.009772372209558112, 0.0007112135136533295,
    4.487453899331323e-06, 2.831391995799379e-08, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0
};

constexpr HighPrecType ALGO_SPECVIEW_VIEW_R[ALGO_SPECVIEW_NLAM] =
{
    0.0006565577806778406, 0.0022109734443221387, 0.006939392614243594, 0.01288916850634296, 0.01690621173074714, 0.012408742348817657,
    0.0004694373108512539, -0.015090966718598288, -0.03101783854748623, -0.04262636516196397, -0.060284802044311914, -0.08083937379061132,
    -0.09133977161371327, -0.08507844703735623, -0.05790557696291452, -0.01967005054791758, 0.02977334723784739, 0.08454551692841392,
    0.142632100058301, 0.17936652195651676, 0.21337711107600393, 0.21621440103838646, 0.1892569735691927, 0.13950225298788524,
    0.10064823630838045, 0.061810796962433474, 0.03671281509298225, 0.020287969895951774, 0.010420018050426333, 0.004482111653621188,
    0.0023425358725277664
};

constexpr HighPrecType ALGO_SPECVIEW_VIEW_G[ALGO_SPECVIEW_NLAM] =
{
    -0.0006185189648818428, -0.002115577356389692, -0.006774808825123884, -0.01314020523005431, -0.01882400222302049, -0.017711059211392,
    -0.010306046834468688, 0.0028332672169190617, 0.019601915617403088, 0.035508860875755985, 0.05919209845617472, 0.09109982169030341,
    0.12253089453403615, 0.1462954399710942, 0.14839733837016605, 0.1435844252101433, 0.12451472465927171, 0.09840486949188365,
    0.07051464010444837, 0.03799338824412607, 0.014623175916150678, -0.0019921510785850622, -0.009792452284138766, -0.010565359838970428,
    -0.009147252163977873, -0.006175869807216623, -0.0038530822723078445, -0.0021804925043704833, -0.0011398686096498185, -0.0004962837648430023,
    -0.0002618293884855628
};

constexpr HighPrecType ALGO_SPECVIEW_VIEW_B[ALGO_SPECVIEW_NLAM] =
{
    0.004989765542879776, 0.017167529014142893, 0.05578763286233301, 0.11354355607771621, 0.180260554048197, 0.2088970505787826,
    0.2010612552115123, 0.15314098376031407, 0.09750154193113619, 0.05051306861787959, 0.02602345164829947, 0.008590794825050605,
    -0.004984582529517758, -0.012323069199043557, -0.015909048764074788, -0.017425404443432175, -0.016513829026277484, -0.014388407507780181,
    -0.01193014512388857, -0.008447396338969492, -0.00607809378836303, -0.004003907229184286, -0.0024528067977241143, -0.0013732416270726788,
    -0.0007979091037313219, -0.000416545676225148, -0.00022307701876783263, -0.00011653331264113849, -5.7226081531228563e-05, -2.3829064310283514e-05,
    -1.2131485708957218e-05
};


// ---------------------------------------------------------------------------
//  Which stocks: a colour transparency with its own three-dye panel and no
//  additive reseau (spectral_view.applies).
// ---------------------------------------------------------------------------
inline bool AlgoSpectralViewApplies(const film::FilmProfile& p) noexcept
{
    return p.isReversal() && !p.is_monochrome && !p.has_reseau
        && p.dye_density.hasData();
}


// np.interp with the ends held: one dye curve onto the common grid.
inline void AlgoSpecViewResample
(
    const std::vector<double>& v,
    const double               start,
    const double               step,
    HighPrecType               out[ALGO_SPECVIEW_NLAM]
) noexcept
{
    const int32_t n = static_cast<int32_t>(v.size());
    for (int32_t l = 0; l < ALGO_SPECVIEW_NLAM; l++)
    {
        const HighPrecType x = ALGO_SPECVIEW_LAM0 + ALGO_SPECVIEW_DLAM * l;
        const HighPrecType x0 = start;
        const HighPrecType xn = start + step * static_cast<HighPrecType>(n - 1);
        if (x <= x0)       { out[l] = v[0]; continue; }
        if (x >= xn)       { out[l] = v[n - 1]; continue; }
        // the bracketing interval, located the way np.interp locates it
        int32_t i = static_cast<int32_t>(std::floor((x - x0) / step));
        if (i < 0) i = 0;
        if (i > n - 2) i = n - 2;
        while ((i < n - 2) && (start + step * (i + 1) <= x)) i++;
        while ((i > 0) && (start + step * i > x)) i--;
        const HighPrecType xa = start + step * i;
        const HighPrecType xb = start + step * (i + 1);
        const HighPrecType t = (x - xa) / (xb - xa);
        out[l] = v[i] + t * (v[i + 1] - v[i]);
    }
}


// 3x3 solve, Gaussian elimination with partial pivoting.
inline bool AlgoSpecViewSolve3(HighPrecType A[3][3], HighPrecType b[3]) noexcept
{
    for (int32_t c = 0; c < 3; c++)
    {
        int32_t piv = c;
        for (int32_t r = c + 1; r < 3; r++)
            if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        if (piv != c)
        {
            for (int32_t k = 0; k < 3; k++) { const HighPrecType t = A[c][k]; A[c][k] = A[piv][k]; A[piv][k] = t; }
            const HighPrecType t = b[c]; b[c] = b[piv]; b[piv] = t;
        }
        if (!(std::fabs(A[c][c]) > 1e-300)) return false;
        for (int32_t r = c + 1; r < 3; r++)
        {
            const HighPrecType f = A[r][c] / A[c][c];
            for (int32_t k = c; k < 3; k++) A[r][k] -= f * A[c][k];
            b[r] -= f * b[c];
        }
    }
    for (int32_t r = 2; r >= 0; r--)
    {
        HighPrecType acc = b[r];
        for (int32_t k = r + 1; k < 3; k++) acc -= A[r][k] * b[k];
        b[r] = acc / A[r][r];
    }
    return true;
}


struct AlgoSpecViewDyes
{
    HighPrecType dye[3][ALGO_SPECVIEW_NLAM];   // cyan, magenta, yellow
    HighPrecType S[3][ALGO_SPECVIEW_NLAM];     // status r, g, b
    HighPrecType ssum[3];
    HighPrecType peakA[3][3];                  // dyes at their own peaks
};


inline void AlgoSpecViewDyesInit(const film::FilmProfile& p, AlgoSpecViewDyes& d) noexcept
{
    const film::SpectralDyeDensity& dd = p.dye_density;
    AlgoSpecViewResample(dd.d_cyan,    dd.lambda_start_nm, dd.lambda_step_nm, d.dye[0]);
    AlgoSpecViewResample(dd.d_magenta, dd.lambda_start_nm, dd.lambda_step_nm, d.dye[1]);
    AlgoSpecViewResample(dd.d_yellow,  dd.lambda_start_nm, dd.lambda_step_nm, d.dye[2]);
    const bool m = (p.density_metric == "status_m");
    for (int32_t l = 0; l < ALGO_SPECVIEW_NLAM; l++)
    {
        d.S[0][l] = m ? ALGO_SPECVIEW_STATUS_M_R[l] : ALGO_SPECVIEW_STATUS_A_R[l];
        d.S[1][l] = m ? ALGO_SPECVIEW_STATUS_M_G[l] : ALGO_SPECVIEW_STATUS_A_G[l];
        d.S[2][l] = m ? ALGO_SPECVIEW_STATUS_M_B[l] : ALGO_SPECVIEW_STATUS_A_B[l];
    }
    for (int32_t i = 0; i < 3; i++)
    {
        HighPrecType s = 0.0;
        for (int32_t l = 0; l < ALGO_SPECVIEW_NLAM; l++) s += d.S[i][l];
        d.ssum[i] = s;
    }
    int32_t peak[3];
    for (int32_t j = 0; j < 3; j++)
    {
        int32_t best = 0;
        for (int32_t l = 1; l < ALGO_SPECVIEW_NLAM; l++)
            if (d.dye[j][l] > d.dye[j][best]) best = l;
        peak[j] = best;
    }
    for (int32_t i = 0; i < 3; i++)
        for (int32_t j = 0; j < 3; j++)
            d.peakA[i][j] = d.dye[j][peak[i]];
}


// Viewed linear RGB (D55, CIE 1931, D65-adapted sRGB; clear film = 1) of the
// status densities D.
inline void AlgoSpecViewRgb(const AlgoSpecViewDyes& d, const HighPrecType D[3],
                            HighPrecType rgb[3]) noexcept
{
    HighPrecType A[3][3];
    HighPrecType a[3] = { D[0], D[1], D[2] };
    for (int32_t i = 0; i < 3; i++) for (int32_t j = 0; j < 3; j++) A[i][j] = d.peakA[i][j];
    AlgoSpecViewSolve3(A, a);
    HighPrecType T[ALGO_SPECVIEW_NLAM];
    for (int32_t it = 0; it < ALGO_SPECVIEW_NEWTON_ITERS; it++)
    {
        for (int32_t l = 0; l < ALGO_SPECVIEW_NLAM; l++)
            T[l] = std::pow(10.0, -(a[0] * d.dye[0][l] + a[1] * d.dye[1][l] + a[2] * d.dye[2][l]));
        HighPrecType st[3], J[3][3], r[3];
        for (int32_t i = 0; i < 3; i++)
        {
            HighPrecType s = 0.0;
            for (int32_t l = 0; l < ALGO_SPECVIEW_NLAM; l++) s += d.S[i][l] * T[l];
            st[i] = s;
            r[i] = -std::log10(s / d.ssum[i]) - D[i];
            for (int32_t j = 0; j < 3; j++)
            {
                HighPrecType q = 0.0;
                for (int32_t l = 0; l < ALGO_SPECVIEW_NLAM; l++) q += d.S[i][l] * T[l] * d.dye[j][l];
                J[i][j] = q / s;
            }
        }
        if (!AlgoSpecViewSolve3(J, r)) break;
        a[0] -= r[0]; a[1] -= r[1]; a[2] -= r[2];
    }
    for (int32_t l = 0; l < ALGO_SPECVIEW_NLAM; l++)
        T[l] = std::pow(10.0, -(a[0] * d.dye[0][l] + a[1] * d.dye[1][l] + a[2] * d.dye[2][l]));
    HighPrecType R = 0.0, G = 0.0, B = 0.0;
    for (int32_t l = 0; l < ALGO_SPECVIEW_NLAM; l++)
    {
        R += ALGO_SPECVIEW_VIEW_R[l] * T[l];
        G += ALGO_SPECVIEW_VIEW_G[l] * T[l];
        B += ALGO_SPECVIEW_VIEW_B[l] * T[l];
    }
    rgb[0] = R; rgb[1] = G; rgb[2] = B;
}


struct AlgoSpecViewTable
{
    HighPrecType lo[3], hi[3], step[3], scale[3], floor_[3];
    std::vector<HighPrecType> rel;   // LUT_N^3 x 3, index ((i*N + j)*N + k)*3 + c
    bool valid = false;
};


inline void AlgoSpecViewTrilinear(const AlgoSpecViewTable& t, const HighPrecType D[3],
                                  HighPrecType out[3]) noexcept
{
    const int32_t N = ALGO_SPECVIEW_LUT_N;
    HighPrecType f[3];
    int32_t i0[3];
    for (int32_t c = 0; c < 3; c++)
    {
        HighPrecType u = (D[c] - t.lo[c]) / t.step[c];
        if (!(u > 0.0)) u = 0.0;                  // NaN-safe clamp
        if (u > static_cast<HighPrecType>(N - 1)) u = static_cast<HighPrecType>(N - 1);
        int32_t i = static_cast<int32_t>(std::floor(u));
        if (i > N - 2) i = N - 2;
        i0[c] = i;
        f[c] = u - static_cast<HighPrecType>(i);
    }
    out[0] = out[1] = out[2] = 0.0;
    for (int32_t di = 0; di < 2; di++)
    {
        const HighPrecType wi = di ? f[0] : 1.0 - f[0];
        for (int32_t dj = 0; dj < 2; dj++)
        {
            const HighPrecType wj = dj ? f[1] : 1.0 - f[1];
            for (int32_t dk = 0; dk < 2; dk++)
            {
                const HighPrecType wk = dk ? f[2] : 1.0 - f[2];
                const HighPrecType w = wi * wj * wk;
                const std::size_t idx = ((static_cast<std::size_t>(i0[0] + di) * N
                                          + static_cast<std::size_t>(i0[1] + dj)) * N
                                         + static_cast<std::size_t>(i0[2] + dk)) * 3u;
                out[0] += w * t.rel[idx + 0];
                out[1] += w * t.rel[idx + 1];
                out[2] += w * t.rel[idx + 2];
            }
        }
    }
}


// The anchored mid grey a reversal render reaches (spectral_view.
// reversal_mid_density + the reader's Callier, as film_sim applies it).
inline void AlgoSpecViewMidDensity
(
    const film::FilmProfile& p,
    const HighPrecType       anchor[3],
    const HighPrecType       couplerFlatScale,
    const HighPrecType       scannerSpecular,
    HighPrecType             dMid[3]
) noexcept
{
    const film::ToneCurve* cv[3] = { &p.curves.r, &p.curves.g, &p.curves.b };
    for (int32_t k = 0; k < 3; k++)
    {
        const HighPrecType rowSum = static_cast<HighPrecType>(p.taking_matrix[k][0])
                                  + static_cast<HighPrecType>(p.taking_matrix[k][1])
                                  + static_cast<HighPrecType>(p.taking_matrix[k][2]);
        const HighPrecType le = std::log10((rowSum > static_cast<HighPrecType>(ALGO_CURVE_EXPOSURE_FLOOR))
                                           ? rowSum : static_cast<HighPrecType>(ALGO_CURVE_EXPOSURE_FLOOR));
        dMid[k] = AlgoDensityScalar(-(le + anchor[k]), *cv[k]);
    }
    const HighPrecType cs = static_cast<HighPrecType>(p.couplers.strength) * couplerFlatScale;
    if ((cs > 0.0) && !p.is_monochrome)
    {
        const HighPrecType dbar = (dMid[0] + dMid[1] + dMid[2]) / 3.0;
        const HighPrecType d0 = dMid[0] + cs * (dMid[0] - dbar);
        const HighPrecType d1 = dMid[1] + cs * (dMid[1] - dbar);
        const HighPrecType d2 = dMid[2] + cs * (dMid[2] - dbar);
        dMid[0] = d0; dMid[1] = d1; dMid[2] = d2;
    }
    for (int32_t k = 0; k < 3; k++)
        dMid[k] = AlgoCallierApplyScalar(dMid[k],
                                         static_cast<HighPrecType>(cv[k]->dmin),
                                         AlgoCallierQChannel(p, k),
                                         scannerSpecular);
}


inline void AlgoSpecViewBuild
(
    const film::FilmProfile& p,
    const HighPrecType       dMid[3],
    AlgoSpecViewTable&       t
) noexcept
{
    t.valid = false;
    if (!AlgoSpectralViewApplies(p))
        return;
    AlgoSpecViewDyes d;
    AlgoSpecViewDyesInit(p, d);
    const film::ToneCurve* cv[3] = { &p.curves.r, &p.curves.g, &p.curves.b };
    const int32_t N = ALGO_SPECVIEW_LUT_N;
    for (int32_t c = 0; c < 3; c++)
    {
        t.lo[c] = static_cast<HighPrecType>(cv[c]->dmin);
        t.hi[c] = static_cast<HighPrecType>(cv[c]->dmin)
                + static_cast<HighPrecType>(cv[c]->gamma)
                * (static_cast<HighPrecType>(cv[c]->shoulder_x)
                   - static_cast<HighPrecType>(cv[c]->toe_x));
        t.step[c] = (t.hi[c] - t.lo[c]) / static_cast<HighPrecType>(N - 1);
    }
    HighPrecType white[3];
    AlgoSpecViewRgb(d, t.lo, white);
    t.rel.assign(static_cast<std::size_t>(N) * N * N * 3u, 0.0);
    for (int32_t i = 0; i < N; i++)
        for (int32_t j = 0; j < N; j++)
            for (int32_t k = 0; k < N; k++)
            {
                // the node densities exactly as np.linspace forms them
                const HighPrecType D[3] = {
                    (i == N - 1) ? t.hi[0] : t.lo[0] + i * t.step[0],
                    (j == N - 1) ? t.hi[1] : t.lo[1] + j * t.step[1],
                    (k == N - 1) ? t.hi[2] : t.lo[2] + k * t.step[2] };
                HighPrecType rgb[3];
                AlgoSpecViewRgb(d, D, rgb);
                const std::size_t idx = ((static_cast<std::size_t>(i) * N + j) * N + k) * 3u;
                t.rel[idx + 0] = rgb[0] / white[0];
                t.rel[idx + 1] = rgb[1] / white[1];
                t.rel[idx + 2] = rgb[2] / white[2];
            }
    HighPrecType relMid[3];
    AlgoSpecViewTrilinear(t, dMid, relMid);
    for (int32_t c = 0; c < 3; c++)
    {
        const HighPrecType rm = (relMid[c] > 1e-300) ? relMid[c] : 1e-300;
        t.scale[c] = (dMid[c] - t.lo[c]) / (-std::log10(rm));
        t.floor_[c] = std::pow(10.0, -(t.hi[c] - t.lo[c]) / t.scale[c]);
    }
    t.valid = true;
}


// In place on the three density planes (AlgoType): D -> equivalent D'.
inline void AlgoSpecViewApply
(
    AlgoType* RESTRICT        pR,
    AlgoType* RESTRICT        pG,
    AlgoType* RESTRICT        pB,
    const int32_t             sizeX,
    const int32_t             sizeY,
    const int32_t             pitch,
    const AlgoSpecViewTable&  t
) noexcept
{
    if (!t.valid)
        return;
    for (int32_t y = 0; y < sizeY; y++)
    {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;
        AlgoType* RESTRICT r = pR + off;
        AlgoType* RESTRICT g = pG + off;
        AlgoType* RESTRICT b = pB + off;
        for (int32_t x = 0; x < sizeX; x++)
        {
            const HighPrecType D[3] = { static_cast<HighPrecType>(r[x]),
                                        static_cast<HighPrecType>(g[x]),
                                        static_cast<HighPrecType>(b[x]) };
            HighPrecType rel[3];
            AlgoSpecViewTrilinear(t, D, rel);
            r[x] = static_cast<AlgoType>(t.lo[0] + t.scale[0] * -std::log10((rel[0] > t.floor_[0]) ? rel[0] : t.floor_[0]));
            g[x] = static_cast<AlgoType>(t.lo[1] + t.scale[1] * -std::log10((rel[1] > t.floor_[1]) ? rel[1] : t.floor_[1]));
            b[x] = static_cast<AlgoType>(t.lo[2] + t.scale[2] * -std::log10((rel[2] > t.floor_[2]) ? rel[2] : t.floor_[2]));
        }
    }
}

#endif // ALGO_SPECTRAL_VIEW_HPP
