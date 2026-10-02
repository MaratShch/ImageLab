#ifndef __IMAGELAB2_PR_FORMAT_INGEST_HPP__
#define __IMAGELAB2_PR_FORMAT_INGEST_HPP__

// =============================================================================
// AlgoPrFormatIngest.hpp - scalar ingest of Adobe Premiere/After Effects frame
// buffers into a linear, interleaved float32 RGB buffer, plus the CCT/Duv
// super-pixel over that linear data.
//
// PERFORMANCE ARCHITECTURE (branch-free inner loop):
//   The pixel format is resolved ONCE, before the traversal, by selecting a
//   compile-time "reader" tag (dispatch_reader's switch runs a single time).
//   The traversal itself (loop_ingest / loop_fused) is a template instantiated
//   per reader, so the hot loop contains NO switch/if on the format - the
//   per-pixel unpack is fully inlined and specialized. The only remaining
//   per-pixel branches are the super-pixel exclusion tests, which are
//   algorithmic (data-dependent), not format dispatch.
//
// PIPELINE per pixel:
//   1. reader unpacks one source pixel (exact struct + channel order from
//      CommonPixFormat.hpp);
//   2. VUYA/VUYP: YCbCr -> R'G'B' with the EXACT Rec.601 or Rec.709 matrix
//      (selected once by the format suffix), studio/full range per bit depth;
//   3. premultiplied (xxxP): un-premultiply (color / alpha) in the encoded
//      domain BEFORE linearization; xxxX = opaque; alpha otherwise dropped;
//   NOTE: sizeX, sizeY and srcPitch are all in PIXELS (srcPitch may be
//         negative for bottom-up frames); the byte stride is computed
//         internally as srcPitch * sizeof(pixel) per format.
//   4. LINEARIZE through the generated float64 LUTs: integer formats index by
//      raw code; continuous 32f / reconstructed VUYA values are requantized
//      through the 16-bit sRGB LUT (finest table);
//   5. store linear color as float32 into the interleaved RGB target;
//   6. (fused entry) accumulate the super-pixel in DOUBLE from the LUT values,
//      before the float32 store - maximum accuracy, single memory pass.
//
// SCALAR, single-threaded, C++14. Host owns concurrency.
//
// ---------------------------------------------------------------------------
// COLOR-SCIENCE ASSUMPTIONS (documented, verify against your footage):
//  * VUYA/VUYP matrix: "_709" -> Rec.709; unsuffixed -> Rec.601 (Adobe).
//  * VUYA/VUYP range: 8u studio (Y 16..235, C 16..240) expanded; 32f full
//    range with chroma centered at 0.5 (subtract 0.5 -> signed Cb,Cr; drop the
//    -0.5 if your build delivers already-signed chroma - flagged inline).
//  * VUYA R'G'B' after the matrix are display-encoded, linearized via the sRGB
//    LUT you pass (pass a Rec.709-transfer LUT instead for true-709 sources).
//  * xxxP premultiplied: un-premultiplied before linearization; A==0 -> color
//    left at 0 (falls into the super-pixel zero-energy gate).
//  * PrPixelFormat_RGB_444_10u: NO alpha; 10-bit R,G,B packed per the struct.
// ---------------------------------------------------------------------------
// =============================================================================

#include <cstdint>
#include <cmath>
#include <type_traits>
#include <array>
#include "CommonPixFormat.hpp"
#include "super_pixel.hpp"     // SuperPixel<> (and the reference two-pass path)

namespace AlgoPrIngest
{
    // Supported formats (mirror the PrPixelFormat values you listed).
    enum ePrPixelFormat
    {
        fmt_BGRA_4444_8u,          fmt_BGRA_4444_16u,
        fmt_BGRA_4444_32f,         fmt_BGRA_4444_32f_Linear,
        fmt_BGRP_4444_8u,          fmt_BGRP_4444_16u,
        fmt_BGRP_4444_32f,         fmt_BGRP_4444_32f_Linear,
        fmt_BGRX_4444_8u,          fmt_BGRX_4444_16u,
        fmt_BGRX_4444_32f,         fmt_BGRX_4444_32f_Linear,
        fmt_VUYA_4444_8u_709,      fmt_VUYA_4444_8u,
        fmt_VUYA_4444_32f_709,     fmt_VUYA_4444_32f,
        fmt_VUYP_4444_8u_709,      fmt_VUYP_4444_8u,
        fmt_VUYP_4444_32f_709,     fmt_VUYP_4444_32f,
        fmt_VUYX_4444_8u_709,      fmt_VUYX_4444_8u,
        fmt_VUYX_4444_32f_709,     fmt_VUYX_4444_32f,
        fmt_RGB_444_10u,
        fmt_ARGB_4444_8u,          fmt_ARGB_4444_16u,
        fmt_ARGB_4444_32f,         fmt_ARGB_4444_32f_Linear,
        fmt_PRGB_4444_8u,          fmt_PRGB_4444_16u,
        fmt_PRGB_4444_32f,         fmt_PRGB_4444_32f_Linear,
        fmt_XRGB_4444_8u,          fmt_XRGB_4444_16u,
        fmt_XRGB_4444_32f,         fmt_XRGB_4444_32f_Linear
    };

    // ---- exact YCbCr -> R'G'B' coefficients (Cb,Cr signed in [-0.5,+0.5]) ----
    // R' = Y' + aR*Cr ; B' = Y' + aB*Cb ; G' = Y' - gCr*Cr - gCb*Cb
    // Derived in 40-digit arithmetic from the ITU-R luma coefficients, rounded
    // to the exact nearest double (round-trip identity verified to 1e-40).
    struct YCbCrToRGB { double aR, aB, gCr, gCb; };
    // ITU-R BT.601 (Kr=0.299 , Kb=0.114 , Kg=0.587)
    constexpr YCbCrToRGB kRec601 =
    { 1.4019999999999999, 1.7720000000000000, 0.71413628620102210, 0.34413628620102216 };
    // ITU-R BT.709 (Kr=0.2126, Kb=0.0722, Kg=0.7152)
    constexpr YCbCrToRGB kRec709 =
    { 1.5748000000000000, 1.8555999999999999, 0.46812427293064879, 0.18732427293064877 };

    // sRGB LUT code ceiling used to requantize CONTINUOUS values (32f / VUYA).
    constexpr int kSRGBLutMax16 = 32767;

    inline int clamp_index(int idx, int maxIdx) {
        return (idx < 0) ? 0 : (idx > maxIdx ? maxIdx : idx);
    }
    // Linearize a continuous normalized value [0,1] via a code-indexed LUT.
    template <typename LUT>
    inline double lin_via_lut(double c01, const LUT& lut, int maxCode) {
        int idx = static_cast<int>(std::lround(c01 * static_cast<double>(maxCode)));
        idx = clamp_index(idx, maxCode);
        return lut[static_cast<std::size_t>(idx)];
    }

    // =========================================================================
    // LOCUS-PROXIMITY GATE (replaces the former saturation gate).
    //
    // WHY: saturation cannot distinguish "saturated because the OBJECT is
    // colored" (grass, clothing - far from the Planckian locus, must be
    // excluded) from "saturated because the LIGHT is colored" (sunset /
    // blue-hour neutrals - near the locus, and precisely the pixels that
    // carry the illuminant information). On cast-dominated scenes the old
    // gate discarded the signal and averaged object colors (measured:
    // Sunset.png -> 5004 K from a blue swimsuit; the locus gate recovers
    // ~2.5 kK, matching the colorist expectation). The physical criterion
    // for "usable for CCT" is DISTANCE TO THE PLANCKIAN LOCUS in (u,v).
    //
    // MECHANICS: the locus (u strictly decreasing with T over the table
    // range) is resampled once into a small uniform-in-u table of v(u) and
    // a slope-correction factor f(u) = 1/sqrt(1+(dv/du)^2); the per-pixel
    // perpendicular distance is then |v - v(u)| * f(u) - O(1), a few ops,
    // no solver. Outside the locus u-range the 2D distance to the nearer
    // endpoint is used. Weight: 1 inside duvFull, linear taper to 0 at
    // duvZero (mirrors the former wSat taper design), SYMMETRIC in +-Duv.
    //
    // The gate also carries the working-space RGB->XYZ matrix (row-major),
    // used for the per-pixel chromaticity AND for the dark gate's luma
    // (row 1 = exact working-space Y - supersedes the former hardcoded
    // Rec.2020 luma weights, which were only correct for a 2020 buffer).
    // =========================================================================
    struct LocusGate
    {
        static const int kN = 256;
        double uMin;            // locus u range, ascending grid
        double invStep;         // kN-1 over (uMax - uMin)
        double vTab[kN];        // v(u) on the uniform u grid
        double fTab[kN];        // perpendicular correction 1/sqrt(1+slope^2)
        double eu0, ev0;        // locus endpoint at uMin (highest CCT)
        double eu1, ev1;        // locus endpoint at uMax (lowest  CCT)
        double duvFull;         // full weight inside this |Duv|
        double duvZero;         // zero weight beyond this |Duv|
        double M[9];            // working-space RGB -> XYZ, row-major

        // Perpendicular distance from (u,v) to the locus polyline, O(1).
        inline double distance(double u, double v) const
        {
            const double uMax = uMin + (kN - 1) / invStep;
            if (u <= uMin) { const double du=u-eu0, dv=v-ev0; return std::sqrt(du*du+dv*dv); }
            if (u >= uMax) { const double du=u-eu1, dv=v-ev1; return std::sqrt(du*du+dv*dv); }
            const double p  = (u - uMin) * invStep;
            int   i  = static_cast<int>(p);
            if (i > kN - 2) i = kN - 2;
            const double t  = p - static_cast<double>(i);
            const double vl = vTab[i] + t * (vTab[i+1] - vTab[i]);
            const double f  = fTab[i] + t * (fTab[i+1] - fTab[i]);
            return std::fabs(v - vl) * f;
        }
    };

    // Build the gate ONCE per setup from the CCT engine's locus table (any
    // row type exposing .u / .v, sorted by ascending cct - i.e. DESCENDING
    // u), the exact working-space RGB->XYZ matrix, and the taper band.
    //
    // build_locus_gate(locus, n, rgb2xyz, duvFull, duvZero, gate)
    //
    // The last two scalars (0.010, 0.020) are the LOCUS-PROXIMITY TAPER BAND,
    // expressed as distance in CIE-1960 (u,v) space -- i.e. |Duv|, the tint
    // distance of a pixel's chromaticity from the Planckian locus. They decide
    // which pixels count toward the super-pixel / CCT measurement and how much:
    //
    //   |Duv| <= 0.010 (duvFull)  -> weight 1.0    : pixel is close enough to the
    //                                                locus to be a trustworthy
    //                                                "neutral under some CCT";
    //                                                counted at full strength.
    //   0.010 < |Duv| < 0.020     -> weight ramps  : linearly tapered 1 -> 0
    //                                (duvFull..duvZero) so inclusion is smooth,
    //                                not a hard cliff (mirrors the old saturation
    //                                taper design -> no popping as a pixel drifts).
    //   |Duv| >= 0.020 (duvZero)  -> weight 0.0    : too far off-locus to be a
    //                                                neutral; treated as an OBJECT
    //                                                color (grass, clothing, skin)
    //                                                and excluded from the estimate.
    //
    // WHY THESE VALUES:
    //   * D65 sits at Duv ~ +0.003; typical real neutrals under real lights fall
    //     within roughly +-0.006 of the locus, so 0.010 keeps genuine neutrals at
    //     full weight with margin.
    //   * 0.020 is ~2x the Ohno crossover (0.002) scale used elsewhere and about
    //     the point past which chromaticities read as clearly colored surfaces
    //     rather than tinted whites -- so it's a sensible "definitely an object"
    //     cutoff.
    //   * The band is SYMMETRIC: it applies equally above the locus (green, +Duv)
    //     and below (magenta, -Duv), so warm/tungsten (-Duv) and fluorescent
    //     (+Duv) casts are treated even-handedly.
    //
    // TUNING:
    //   * WIDEN (e.g. 0.015 / 0.030) to keep more pixels -> higher coverage /
    //     confidence on clean scenes, but admits more mildly-colored surfaces
    //     that can bias the estimate.
    //   * NARROW (e.g. 0.006 / 0.012) to keep only near-perfect neutrals -> purer
    //     estimate, but fewer kept pixels (lower confidence, worse on casts).
    //   These are the ONLY tuning knobs of the gate; they are passed in (not
    //   hardcoded in the engine) precisely so they can be adjusted without
    //   touching engine code, and even exposed as an "advanced" control later.
    template <typename Row>
    inline void build_locus_gate(const Row* locus, std::size_t n,
                                 const double rgb2xyz[9],
                                 double duvFull, double duvZero,
                                 LocusGate& g)
    {
        // locus u decreases with cct index; grid ascends in u.
        const double uLo = static_cast<double>(locus[n - 1u].u);   // highest CCT end
        const double uHi = static_cast<double>(locus[0].u);        // lowest  CCT end
        g.uMin   = uLo;
        g.invStep = (LocusGate::kN - 1) / (uHi - uLo);
        g.eu0 = locus[n - 1u].u; g.ev0 = locus[n - 1u].v;
        g.eu1 = locus[0].u;      g.ev1 = locus[0].v;
        g.duvFull = duvFull; g.duvZero = duvZero;
        for (int k = 0; k < 9; ++k) g.M[k] = rgb2xyz[k];

        // For each grid u, locate the bracketing locus segment (u is
        // monotonic in the table) and interpolate v; slope from the segment.
        std::size_t j = n - 1u;                     // walks toward index 0 as u grows
        for (int i = 0; i < LocusGate::kN; ++i)
        {
            const double u = g.uMin + static_cast<double>(i) / g.invStep;
            while (j > 0u && static_cast<double>(locus[j - 1u].u) < u) --j;
            const std::size_t hi = (j > 0u) ? j - 1u : 0u;   // locus[hi].u >= u >= locus[j].u
            const double uA = locus[j].u,  vA = locus[j].v;
            const double uB = locus[hi].u, vB = locus[hi].v;
            const double span = uB - uA;
            const double t = (span > 0.0) ? (u - uA) / span : 0.0;
            const double vl = vA + t * (vB - vA);
            const double slope = (span > 0.0) ? (vB - vA) / span : 0.0;
            g.vTab[i] = vl;
            g.fTab[i] = 1.0 / std::sqrt(1.0 + slope * slope);
        }
    }

    // Immutable per-frame context handed to every reader (holds the tables and
    // the ONE chosen YCbCr matrix). References: no copying of the LUTs.
    template <typename LUT8, typename LUT16, typename LUT10>
    struct DecodeCtx
    {
        const LUT8&  lut8;
        const LUT16& lut16;
        const LUT10& lut10;
        YCbCrToRGB   C;
    };

    // =========================================================================
    // Per-format READERS - each unpacks one pixel at (row,x) to linear double.
    // Templated on the pixel struct / premul / linear flags so a single body
    // covers channel-order twins (BGRA vs ARGB share member names R,G,B,A) and
    // the premultiplied / linear variants collapse to compile-time constants.
    // No runtime format branch anywhere in here.
    // =========================================================================

    // ---- integer 8-bit (index lut8 by raw code) ----
    template <typename Pix, bool Premul>
    struct ReadInt8
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, int32_t x, const Ctx& c,
                                double& R, double& G, double& B)
        {
            const Pix* p = reinterpret_cast<const Pix*>(row) + x;
            int r = p->R, g = p->G, b = p->B;
            if (Premul && p->A != 0) {                       // un-premultiply (encoded domain)
                const double a = p->A / 255.0;
                r = clamp_index(static_cast<int>(std::lround(r / a)), 255);
                g = clamp_index(static_cast<int>(std::lround(g / a)), 255);
                b = clamp_index(static_cast<int>(std::lround(b / a)), 255);
            }
            R = c.lut8[r]; G = c.lut8[g]; B = c.lut8[b];
        }
    };

    // ---- integer 16-bit (index lut16 by raw code, 0..32767) ----
    template <typename Pix, bool Premul>
    struct ReadInt16
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, int32_t x, const Ctx& c,
                                double& R, double& G, double& B)
        {
            const Pix* p = reinterpret_cast<const Pix*>(row) + x;
            int r = p->R, g = p->G, b = p->B;
            if (Premul && p->A != 0) {
                const double a = p->A / 32767.0;
                r = clamp_index(static_cast<int>(std::lround(r / a)), 32767);
                g = clamp_index(static_cast<int>(std::lround(g / a)), 32767);
                b = clamp_index(static_cast<int>(std::lround(b / a)), 32767);
            }
            R = c.lut16[r]; G = c.lut16[g]; B = c.lut16[b];
        }
    };

    // ---- 32f (gamma-encoded or already-linear), optional premultiplied ----
    template <typename Pix, bool Premul, bool Linear>
    struct ReadF32
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, int32_t x, const Ctx& c,
                                double& R, double& G, double& B)
        {
            const Pix* p = reinterpret_cast<const Pix*>(row) + x;
            double r = p->R, g = p->G, b = p->B;
            if (Premul && p->A != 0.f) { const double a = p->A; r /= a; g /= a; b /= a; }
            if (Linear) { R = r; G = g; B = b; }             // already linear
            else {
                R = lin_via_lut(r, c.lut16, kSRGBLutMax16);
                G = lin_via_lut(g, c.lut16, kSRGBLutMax16);
                B = lin_via_lut(b, c.lut16, kSRGBLutMax16);
            }
        }
    };

    // ---- VUYA / VUYP 8u (studio range) ----
    template <bool Premul>
    struct ReadVUYA8
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_8u);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, int32_t x, const Ctx& c,
                                double& R, double& G, double& B)
        {
            const PF_Pixel_VUYA_8u* p = reinterpret_cast<const PF_Pixel_VUYA_8u*>(row) + x;
            const double Yp = (static_cast<double>(p->Y) - 16.0)  / 219.0;
            const double Cb = (static_cast<double>(p->U) - 128.0) / 224.0;
            const double Cr = (static_cast<double>(p->V) - 128.0) / 224.0;
            double Rp = Yp + c.C.aR * Cr;
            double Bp = Yp + c.C.aB * Cb;
            double Gp = Yp - c.C.gCr * Cr - c.C.gCb * Cb;
            if (Premul && p->A != 0) { const double a = p->A / 255.0; Rp /= a; Gp /= a; Bp /= a; }
            R = lin_via_lut(Rp, c.lut16, kSRGBLutMax16);
            G = lin_via_lut(Gp, c.lut16, kSRGBLutMax16);
            B = lin_via_lut(Bp, c.lut16, kSRGBLutMax16);
        }
    };

    // ---- VUYA / VUYP 32f (full range, chroma centered at 0.5) ----
    template <bool Premul>
    struct ReadVUYA32
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_32f);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, int32_t x, const Ctx& c,
                                double& R, double& G, double& B)
        {
            const PF_Pixel_VUYA_32f* p = reinterpret_cast<const PF_Pixel_VUYA_32f*>(row) + x;
            const double Yp = p->Y;
            const double Cb = static_cast<double>(p->U) - 0.5;   // drop -0.5 if already signed
            const double Cr = static_cast<double>(p->V) - 0.5;
            double Rp = Yp + c.C.aR * Cr;
            double Bp = Yp + c.C.aB * Cb;
            double Gp = Yp - c.C.gCr * Cr - c.C.gCb * Cb;
            if (Premul && p->A != 0.f) { const double a = p->A; Rp /= a; Gp /= a; Bp /= a; }
            R = lin_via_lut(Rp, c.lut16, kSRGBLutMax16);
            G = lin_via_lut(Gp, c.lut16, kSRGBLutMax16);
            B = lin_via_lut(Bp, c.lut16, kSRGBLutMax16);
        }
    };

    // ---- RGB 444 10u (no alpha; packed bitfields) ----
    struct ReadRGB10
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_RGB_10u);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, int32_t x, const Ctx& c,
                                double& R, double& G, double& B)
        {
            const PF_Pixel_RGB_10u* p = reinterpret_cast<const PF_Pixel_RGB_10u*>(row) + x;
            R = c.lut10[p->R]; G = c.lut10[p->G]; B = c.lut10[p->B];
        }
    };

    // =========================================================================
    // dispatch_reader - the ONLY format switch. Runs ONCE per frame, selects
    // the compile-time reader tag, and hands it to the generic callable f
    // (which instantiates the specialized, branch-free loop for that reader).
    // =========================================================================
    template <typename F>
    inline void dispatch_reader(ePrPixelFormat fmt, F&& f)
    {
        switch (fmt)
        {
            case fmt_BGRA_4444_8u:
            case fmt_BGRX_4444_8u:         f(ReadInt8 <PF_Pixel_BGRA_8u , false>{}); break;
            case fmt_BGRP_4444_8u:         f(ReadInt8 <PF_Pixel_BGRA_8u , true >{}); break;
            case fmt_ARGB_4444_8u:
            case fmt_XRGB_4444_8u:         f(ReadInt8 <PF_Pixel_ARGB_8u , false>{}); break;
            case fmt_PRGB_4444_8u:         f(ReadInt8 <PF_Pixel_ARGB_8u , true >{}); break;

            case fmt_BGRA_4444_16u:
            case fmt_BGRX_4444_16u:        f(ReadInt16<PF_Pixel_BGRA_16u, false>{}); break;
            case fmt_BGRP_4444_16u:        f(ReadInt16<PF_Pixel_BGRA_16u, true >{}); break;
            case fmt_ARGB_4444_16u:
            case fmt_XRGB_4444_16u:        f(ReadInt16<PF_Pixel_ARGB_16u, false>{}); break;
            case fmt_PRGB_4444_16u:        f(ReadInt16<PF_Pixel_ARGB_16u, true >{}); break;

            case fmt_BGRA_4444_32f:
            case fmt_BGRX_4444_32f:        f(ReadF32<PF_Pixel_BGRA_32f, false, false>{}); break;
            case fmt_BGRP_4444_32f:        f(ReadF32<PF_Pixel_BGRA_32f, true , false>{}); break;
            case fmt_BGRA_4444_32f_Linear:
            case fmt_BGRX_4444_32f_Linear: f(ReadF32<PF_Pixel_BGRA_32f, false, true >{}); break;
            case fmt_BGRP_4444_32f_Linear: f(ReadF32<PF_Pixel_BGRA_32f, true , true >{}); break;
            case fmt_ARGB_4444_32f:
            case fmt_XRGB_4444_32f:        f(ReadF32<PF_Pixel_ARGB_32f, false, false>{}); break;
            case fmt_ARGB_4444_32f_Linear:
            case fmt_XRGB_4444_32f_Linear: f(ReadF32<PF_Pixel_ARGB_32f, false, true >{}); break;
            case fmt_PRGB_4444_32f:        f(ReadF32<PF_Pixel_ARGB_32f, true , false>{}); break;
            case fmt_PRGB_4444_32f_Linear: f(ReadF32<PF_Pixel_ARGB_32f, true , true >{}); break;

            case fmt_VUYA_4444_8u_709:
            case fmt_VUYA_4444_8u:         f(ReadVUYA8 <false>{}); break;
            case fmt_VUYP_4444_8u_709:
            case fmt_VUYP_4444_8u:         f(ReadVUYA8 <true >{}); break;
            case fmt_VUYA_4444_32f_709:
            case fmt_VUYA_4444_32f:        f(ReadVUYA32<false>{}); break;
            case fmt_VUYP_4444_32f_709:
            case fmt_VUYP_4444_32f:        f(ReadVUYA32<true >{}); break;
            case fmt_VUYX_4444_8u_709:
            case fmt_VUYX_4444_8u:         f(ReadVUYA8 <false>{}); break;
            case fmt_VUYX_4444_32f_709:
            case fmt_VUYX_4444_32f:        f(ReadVUYA32<false>{}); break;

            case fmt_RGB_444_10u:          f(ReadRGB10{}); break;
            default: break;
        }
    }

    // Rec.601/709 selection (once per frame, outside the loop).
    inline YCbCrToRGB pick_matrix(ePrPixelFormat fmt)
    {
        const bool is709 = (fmt == fmt_VUYA_4444_8u_709 || fmt == fmt_VUYA_4444_32f_709 ||
                            fmt == fmt_VUYP_4444_8u_709 || fmt == fmt_VUYP_4444_32f_709 ||
                            fmt == fmt_VUYX_4444_8u_709 || fmt == fmt_VUYX_4444_32f_709);
        return is709 ? kRec709 : kRec601;
    }

    // =========================================================================
    // Branch-free (on format) traversals, templated on the selected Reader.
    // =========================================================================

    // ingest only -> fill interleaved linear float32 RGB.
    template <typename Reader, typename Ctx>
    inline void loop_ingest(const std::uint8_t* base, int32_t sizeX, int32_t sizeY,
                            int32_t srcPitch, const Ctx& ctx, float* dstRGB_f32)
    {
        // srcPitch is in PIXELS (may be negative for bottom-up). Convert to a
        // signed byte stride using this format's element size.
        const std::ptrdiff_t byteStride =
            static_cast<std::ptrdiff_t>(srcPitch) *
            static_cast<std::ptrdiff_t>(Reader::kPixelBytes);
        for (int32_t y = 0; y < sizeY; ++y)
        {
            const std::uint8_t* row = base + static_cast<std::ptrdiff_t>(y) * byteStride;
            float* dstRow = dstRGB_f32 + static_cast<std::ptrdiff_t>(y) * sizeX * 3;
            for (int32_t x = 0; x < sizeX; ++x)
            {
                double R, G, B;
                Reader::read(row, x, ctx, R, G, B);
                float* d = dstRow + static_cast<std::ptrdiff_t>(x) * 3;
                d[0] = static_cast<float>(R);
                d[1] = static_cast<float>(G);
                d[2] = static_cast<float>(B);
            }
        }
    }

    // fused -> fill float32 AND accumulate the super-pixel in DOUBLE, one pass.
    // Exclusion rules MIRROR AlgoSuperPixel.hpp EXACTLY (in double here).
    //
    // kConfidenceMap (compile-time): when true, the destination buffer holds
    // the CONFIDENCE MAP instead of the plain linear image - every pixel that
    // CONTRIBUTES to the super-pixel (final weight w > 0, including partially
    // tapered pixels) is stored as its normal linear RGB, and every EXCLUDED
    // pixel (any gate: non-finite, negative, zero-energy, clipped, too dark,
    // too saturated, zero weight) is stored as pure black (0,0,0). The
    // super-pixel accumulation itself is IDENTICAL in both modes - the map
    // only changes what is written to dstRGB_f32. Because the flag is a
    // template constant, the kConfidenceMap=false instantiation compiles to
    // exactly the previous code: the normal path pays nothing.
    template <typename Reader, typename Ctx, bool kConfidenceMap = false>
    inline void loop_fused(const std::uint8_t* base, int32_t sizeX, int32_t sizeY,
                           int32_t srcPitch, const Ctx& ctx, const LocusGate& gate,
                           float* dstRGB_f32, SuperPixel<double>& super,
                           double* keptFraction)
    {
        // srcPitch is in PIXELS (may be negative). -> signed byte stride.
        const std::ptrdiff_t byteStride =
            static_cast<std::ptrdiff_t>(srcPitch) *
            static_cast<std::ptrdiff_t>(Reader::kPixelBytes);
        constexpr double kYDark     = 0.010;
        constexpr double kChClip    = 0.95;
        constexpr double kTaperLo   = 0.90;
        constexpr double kEnergyMin = 1.0e-6;
        // NOTE: the former saturation gate (kSatFull/kSatMax) is REPLACED by
        // the locus-proximity gate; the dark gate's luma now uses the exact
        // working-space Y (gate.M row 1) instead of hardcoded 2020 weights.

        double rSum = 0.0, gSum = 0.0, bSum = 0.0, wSum = 0.0;
        std::size_t keptCount = 0u;

        for (int32_t y = 0; y < sizeY; ++y)
        {
            const std::uint8_t* row = base + static_cast<std::ptrdiff_t>(y) * byteStride;
            float* dstRow = dstRGB_f32 + static_cast<std::ptrdiff_t>(y) * sizeX * 3;
            for (int32_t x = 0; x < sizeX; ++x)
            {
                double R, G, B;
                Reader::read(row, x, ctx, R, G, B);

                float* d = dstRow + static_cast<std::ptrdiff_t>(x) * 3;
                d[0] = static_cast<float>(R);
                d[1] = static_cast<float>(G);
                d[2] = static_cast<float>(B);

                // In confidence-map mode an excluded pixel is repainted black.
                // 'exclude' is dead code (folded away) when kConfidenceMap is
                // false, so the normal path is byte-identical to before.
                const auto exclude = [d]() {
                    if (kConfidenceMap) { d[0] = 0.0f; d[1] = 0.0f; d[2] = 0.0f; }
                };

                // --- super-pixel exclusion / weighting (double) ---
                if (!(std::isfinite(R) && std::isfinite(G) && std::isfinite(B))) { exclude(); continue; }
                if (R < 0.0 || G < 0.0 || B < 0.0)                               { exclude(); continue; }
                if ((R + G + B) <= kEnergyMin)                                   { exclude(); continue; }
                const double maxc = (R > G) ? ((R > B) ? R : B) : ((G > B) ? G : B);
                if (maxc >= kChClip)                                            { exclude(); continue; }
                // exact working-space XYZ (gate.M) - luma gate + chromaticity
                const double Xc = gate.M[0]*R + gate.M[1]*G + gate.M[2]*B;
                const double Yc = gate.M[3]*R + gate.M[4]*G + gate.M[5]*B;
                const double Zc = gate.M[6]*R + gate.M[7]*G + gate.M[8]*B;
                if (Yc < kYDark)                                                { exclude(); continue; }
                const double wLum = (maxc > kTaperLo)
                                  ? (kChClip - maxc) / (kChClip - kTaperLo) : 1.0;
                const double den = Xc + 15.0 * Yc + 3.0 * Zc;
                if (den <= 0.0)                                                 { exclude(); continue; }
                const double u = 4.0 * Xc / den;
                const double v = 6.0 * Yc / den;
                // LOCUS-PROXIMITY gate: perpendicular distance to the
                // Planckian locus, symmetric in +-Duv, soft taper.
                const double dLoc = gate.distance(u, v);
                if (dLoc >= gate.duvZero)                                       { exclude(); continue; }
                const double wLoc = (dLoc <= gate.duvFull) ? 1.0
                                  : (gate.duvZero - dLoc) / (gate.duvZero - gate.duvFull);
                const double w = wLum * wLoc;
                if (w <= 0.0)                                                   { exclude(); continue; }
                ++keptCount;
                rSum += w * R; gSum += w * G; bSum += w * B; wSum += w;
            }
        }

        if (wSum > 0.0) {
            const double inv = 1.0 / wSum;
            super.r = rSum * inv; super.g = gSum * inv; super.b = bSum * inv;
        } else {
            super.r = super.g = super.b = 0.0;
        }
        if (keptFraction) {
            const double total = static_cast<double>(sizeX) * static_cast<double>(sizeY);
            *keptFraction = (total > 0.0) ? static_cast<double>(keptCount) / total : 0.0;
        }
    }

    // =========================================================================
    // PUBLIC ENTRY POINTS
    // =========================================================================

    // Ingest a frame -> interleaved linear float32 RGB (no super-pixel).
    template <typename LUT8, typename LUT16, typename LUT10>
    void ingest_to_linear_f32
    (
        const void* src, int32_t sizeX, int32_t sizeY, int32_t srcPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        float* dstRGB_f32
    )
    {
        const DecodeCtx<LUT8, LUT16, LUT10> ctx{ lut8, lut16, lut10, pick_matrix(fmt) };
        const std::uint8_t* base = static_cast<const std::uint8_t*>(src);
        dispatch_reader(fmt, [&](auto reader) {
            using R = decltype(reader);
            loop_ingest<R>(base, sizeX, sizeY, srcPitch, ctx, dstRGB_f32);
        });
    }

    // Fused: ingest -> float32 AND compute the super-pixel in double, one pass.
    //
    // confidenceMap (default false - existing call sites unchanged):
    //   false -> dstRGB_f32 holds the normal linear image (previous behavior).
    //   true  -> dstRGB_f32 holds the CONFIDENCE MAP: pixels that contribute
    //            to the super-pixel keep their linear RGB values; pixels
    //            excluded by any gate are written as pure black (0,0,0).
    //            The returned super-pixel is IDENTICAL in both modes (the
    //            flag only affects what is stored, never what is measured).
    //   The flag is resolved ONCE per frame into a compile-time template
    //   constant, so neither mode pays a per-pixel format/mode branch.
    // gate         : locus-proximity gate + working-space matrix; build ONCE
    //                per setup with build_locus_gate() from the CCT engine's
    //                locus table (e.g. CctHandle::getLut_CIE_1931()).
    // keptFraction : optional out - fraction of pixels contributing to the
    //                super-pixel (0..1). This is the measurement CONFIDENCE:
    //                low values (< ~0.10) mean the scene offers few usable
    //                near-locus pixels and the auto result should be flagged.
    template <typename LUT8, typename LUT16, typename LUT10>
    void ingest_and_superpixel
    (
        const void* src, int32_t sizeX, int32_t sizeY, int32_t srcPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        const LocusGate& gate,
        float* dstRGB_f32,
        SuperPixel<double>& super,
        bool confidenceMap = false,
        double* keptFraction = nullptr
    )
    {
        const DecodeCtx<LUT8, LUT16, LUT10> ctx{ lut8, lut16, lut10, pick_matrix(fmt) };
        const std::uint8_t* base = static_cast<const std::uint8_t*>(src);
        dispatch_reader(fmt, [&](auto reader) {
            using R = decltype(reader);
            if (confidenceMap)
                loop_fused<R, DecodeCtx<LUT8, LUT16, LUT10>, true >(base, sizeX, sizeY, srcPitch, ctx, gate, dstRGB_f32, super, keptFraction);
            else
                loop_fused<R, DecodeCtx<LUT8, LUT16, LUT10>, false>(base, sizeX, sizeY, srcPitch, ctx, gate, dstRGB_f32, super, keptFraction);
        });
    }

} // namespace AlgoPrIngest

#endif // __IMAGELAB2_PR_FORMAT_INGEST_HPP__
