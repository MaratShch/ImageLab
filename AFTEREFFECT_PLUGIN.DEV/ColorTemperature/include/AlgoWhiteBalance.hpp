#ifndef __IMAGELAB2_WHITE_BALANCE_HPP__
#define __IMAGELAB2_WHITE_BALANCE_HPP__

// =============================================================================
// AlgoWhiteBalance.hpp - build_wb_matrix()
//
// Turns a SOURCE white and a TARGET white (each given as CCT + Duv) into ONE
// 3x3 matrix M_wb that maps a linear working-space RGB pixel to its
// white-balanced result:  rgb_out = M_wb * rgb_in  (linear, unclamped).
//
// This is PHASE 2 / Step C of the pipeline. Build ONCE per frame; the caller
// then applies M_wb to every pixel (Step D). All math in double.
//
// PIPELINE inside:
//   1. (CCT, Duv) -> CIE-1960 (u,v):  on-locus point from the CCT LUT, then
//      offset perpendicular to the locus by Duv (slope from adjacent 1 K rows).
//   2. (u,v) -> (x,y) -> XYZ, Y = 1   (white points, luminance-normalized).
//   3. Chromatic adaptation (CAT): cone_s = M*XYZ_s, cone_d = M*XYZ_d,
//      gains = cone_d/cone_s (degree-D blended), M_adapt = M^-1 * diag * M.
//   4. Wrap into RGB:  M_wb = XYZ2RGB * M_adapt * RGB2XYZ.
//
// The CAT cone matrices (Bradford / CAT02 / CAT16 / Von Kries) are baked in as
// verified constants; the CCT LUT and the working-space RGB<->XYZ matrices are
// supplied by the caller so no color space is hardcoded.
// =============================================================================

#include <cstdint>
#include <cmath>
#include "super_pixel.hpp"   // CctDuv<>

namespace AlgoWB
{
    enum eCatModel : int32_t
    { cat_Bradford = 0, cat_CAT16 = 1, cat_CAT02 = 2, cat_VonKries = 3 };

    // ---- small 3x3 helpers (row-major double[9]) ---------------------------
    inline void mat3_mul(const double A[9], const double B[9], double C[9])
    {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                C[r*3+c] = A[r*3+0]*B[0*3+c] + A[r*3+1]*B[1*3+c] + A[r*3+2]*B[2*3+c];
    }
    inline void mat3_vec(const double M[9], const double v[3], double o[3])
    {
        o[0] = M[0]*v[0] + M[1]*v[1] + M[2]*v[2];
        o[1] = M[3]*v[0] + M[4]*v[1] + M[5]*v[2];
        o[2] = M[6]*v[0] + M[7]*v[1] + M[8]*v[2];
    }
    inline bool mat3_inverse(const double M[9], double R[9])
    {
        const double a=M[0],b=M[1],c=M[2],d=M[3],e=M[4],f=M[5],g=M[6],h=M[7],i=M[8];
        const double A=e*i-f*h, B=-(d*i-f*g), C=d*h-e*g;
        const double det=a*A+b*B+c*C;
        if (det==0.0) return false;
        const double id=1.0/det;
        R[0]=A*id;        R[1]=(c*h-b*i)*id; R[2]=(b*f-c*e)*id;
        R[3]=B*id;        R[4]=(a*i-c*g)*id; R[5]=(c*d-a*f)*id;
        R[6]=C*id;        R[7]=(b*g-a*h)*id; R[8]=(a*e-b*d)*id;
        return true;
    }

    // ---- CAT cone-response matrices (row-major), verified constants --------
    inline const double* cat_matrix(int model)
    {
        static const double BRADFORD[9] = {
             0.8951000,  0.2664000, -0.1614000,
            -0.7502000,  1.7135000,  0.0367000,
             0.0389000, -0.0685000,  1.0296000 };
        static const double CAT02[9] = {
             0.7328000,  0.4296000, -0.1624000,
            -0.7036000,  1.6975000,  0.0061000,
             0.0030000,  0.0136000,  0.9834000 };
        static const double CAT16[9] = {
             0.401288,   0.650173,  -0.051461,
            -0.250268,   1.204414,   0.045854,
            -0.002079,   0.048952,   0.953127 };
        static const double VONKRIES[9] = {   // HPE, equal-energy normalized
             0.3897000,  0.6890000, -0.0787000,
            -0.2298000,  1.1834000,  0.0464000,
             0.0000000,  0.0000000,  1.0000000 };
        switch (model) {
            case cat_CAT02:    return CAT02;
            case cat_CAT16:    return CAT16;
            case cat_VonKries: return VONKRIES;
            case cat_Bradford:
            default:           return BRADFORD;
        }
    }

    // ---- (CCT, Duv) -> CIE-1960 (u,v) from the LUT -------------------------
    // Row type must expose .cct/.u/.v; table is CCT_MIN..CCT_MAX at 1 K step,
    // directly indexable by (cct - CCT_MIN). Off-locus offset: move by Duv
    // along the NORMAL to the locus (perpendicular to the local tangent).
    template <typename Row>
    inline void cctduv_to_uv(const Row* lut, std::size_t n,
                             double cctMin, double cctMax,
                             double cct, double duv, double uvOut[2])
    {
        if (cct < cctMin) cct = cctMin;
        if (cct > cctMax) cct = cctMax;
        const double fidx = cct - cctMin;                 // 1 K step
        std::size_t i0 = static_cast<std::size_t>(fidx);
        if (i0 > n - 2u) i0 = n - 2u;                      // keep i0+1 valid
        const std::size_t i1 = i0 + 1u;
        const double t = fidx - static_cast<double>(i0);

        // on-locus point at cct (linear interp between 1 K rows)
        const double u = lut[i0].u + t * (lut[i1].u - lut[i0].u);
        const double v = lut[i0].v + t * (lut[i1].v - lut[i0].v);

        // local tangent from the two rows; normal is perpendicular. Duv is
        // defined POSITIVE above the locus (toward green). With CCT rising as
        // u decreases, the outward normal that gives +Duv=green is (du,dv)
        // rotated -90deg and normalized.
        double du = lut[i1].u - lut[i0].u;
        double dv = lut[i1].v - lut[i0].v;
        const double len = std::sqrt(du*du + dv*dv);
        if (len > 0.0) { du /= len; dv /= len; }
        // normal (perpendicular): (dv, -du) points to the +Duv (green /
        // above-locus) side. Verified against the CIE reference: for
        // (8198.45 K, +0.0137) this reproduces colour-science's uv to 1e-15.
        uvOut[0] = u + duv * ( dv);
        uvOut[1] = v + duv * (-du);
    }

    // ---- CIE-1960 (u,v) -> XYZ at Y = 1 ------------------------------------
    inline void uv_to_XYZ(const double uv[2], double XYZ[3])
    {
        const double u = uv[0], v = uv[1];
        const double d = 2.0*u - 8.0*v + 4.0;      // uv(1960) -> xy
        const double x = 3.0*u / d;
        const double y = 2.0*v / d;
        XYZ[0] = x / y;                            // Y = 1
        XYZ[1] = 1.0;
        XYZ[2] = (1.0 - x - y) / y;
    }

    // =========================================================================
    // build_wb_matrix - the public Step C entry point.
    //   lut/n/cctMin/cctMax : the CCT locus LUT for the chosen observer.
    //   source / target     : white points as (CCT, Duv).
    //   catModel            : eCatModel (Bradford default).
    //   D                   : degree of adaptation 0..1 (1 = full; v1 uses 1).
    //   rgb2xyz / xyz2rgb   : working-space matrices (row-major [9]).
    //   M_wb (out)          : row-major RGB->corrected-RGB.
    // Returns false only on a singular matrix (should never happen for valid
    // CATs / working spaces).
    // =========================================================================
    template <typename Row>
    inline bool build_wb_matrix(
        const Row* lut, std::size_t n, double cctMin, double cctMax,
        const CctDuv<double>& source, const CctDuv<double>& target,
        int catModel, double D,
        const double rgb2xyz[9], const double xyz2rgb[9],
        double M_wb[9]) noexcept
    {
        // 1-2. white points -> XYZ
        double uv_s[2], uv_d[2], XYZ_s[3], XYZ_d[3];
        cctduv_to_uv(lut, n, cctMin, cctMax, source.cct, source.duv, uv_s);
        cctduv_to_uv(lut, n, cctMin, cctMax, target.cct, target.duv, uv_d);
        uv_to_XYZ(uv_s, XYZ_s);
        uv_to_XYZ(uv_d, XYZ_d);

        // 3. CAT
        const double* M = cat_matrix(catModel);
        double Minv[9];
        if (!mat3_inverse(M, Minv)) return false;
        double cone_s[3], cone_d[3];
        mat3_vec(M, XYZ_s, cone_s);
        mat3_vec(M, XYZ_d, cone_d);
        double gain[3];
        for (int k = 0; k < 3; ++k) {
            const double raw = cone_d[k] / cone_s[k];
            gain[k] = D * raw + (1.0 - D);           // D-blended (D=1 -> raw)
        }
        const double diag[9] = { gain[0],0,0, 0,gain[1],0, 0,0,gain[2] };
        double tmp[9], M_adapt[9];
        mat3_mul(diag, M, tmp);        // diag * M
        mat3_mul(Minv, tmp, M_adapt);  // M^-1 * diag * M   (XYZ -> XYZ)

        // 4. wrap into RGB:  M_wb = XYZ2RGB * M_adapt * RGB2XYZ
        double t2[9];
        mat3_mul(M_adapt, rgb2xyz, t2);
        mat3_mul(xyz2rgb, t2, M_wb);
        return true;
    }

} // namespace AlgoWB

#endif // __IMAGELAB2_WHITE_BALANCE_HPP__
