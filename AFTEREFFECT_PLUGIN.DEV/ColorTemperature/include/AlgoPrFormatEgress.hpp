#ifndef __IMAGELAB2_PR_FORMAT_EGRESS_HPP__
#define __IMAGELAB2_PR_FORMAT_EGRESS_HPP__

// =============================================================================
// AlgoPrFormatEgress.hpp - scalar EGRESS: write a linear, interleaved float32
// RGB buffer (as produced by ingest_to_linear_f32 / ingest_and_superpixel)
// back into an Adobe Premiere/After Effects frame buffer, in ANY of the
// formats the ingest supports. Exact structural inverse of the ingest.
//
// ALPHA: the linear buffer carries NO alpha, so the alpha (or X pad) of every
// output pixel is taken from the ORIGINAL incoming Adobe buffer, which the
// caller passes alongside (same pixel format as the destination). For
// premultiplied formats (BGRP/VUYP) that alpha is also re-applied to the
// color, exactly inverting the ingest's un-premultiply.
//
// ENCODING (linear -> display code) is done by INVERTING THE SAME DECODE LUTs
// used at ingest (binary search over the monotonically increasing table):
//   - integer formats: nearest code -> the round trip
//     ingest(decode) -> egress(encode) is BIT-EXACT for every code;
//   - continuous outputs (32f encoded, VUYA): linear interpolation between
//     the two bracketing LUT entries -> smooth encode, transfer-agnostic
//     (whatever transfer the caller's LUT encodes - sRGB, Rec.709, gamma -
//     the egress automatically inverts it; no second table needed).
//
// YCbCr: the forward (R'G'B' -> Y'CbCr) matrix is DERIVED from the same
// YCbCrToRGB reconstruction constants the ingest uses (Kr = 1 - aR/2,
// Kb = 1 - aB/2, Kg = 1 - Kr - Kb; Cr = (R'-Y')/aR, Cb = (B'-Y')/aB), so it
// is the exact inverse of the ingest reconstruction, and the BT.709 (_709
// suffix) versus BT.601 (no suffix) selection is inherited from pick_matrix.
// VUYA 8u is studio range (Y' 16..235, C 16..240), 32f full range with
// chroma stored at +0.5 - mirroring the ingest conventions exactly.
//
// _Linear formats: the float32 values are copied as-is (identity transfer,
// unclamped - HDR and negatives survive); premultiplied _Linear multiplies
// the LINEAR values by alpha, inverting the ingest's linear un-premultiply.
//
// SIZES AND PITCHES ARE IN PIXELS (may be negative for bottom-up frames);
// byte strides are computed internally from the per-format pixel size.
//
// SCALAR, single-threaded, C++14. Host owns concurrency. Branch-free inner
// loop: one dispatch switch per frame selects a compile-time Writer tag.
// =============================================================================

#include <cstdint>
#include <cmath>
#include "AlgoPrFormatIngest.hpp"   // formats, structs, matrices, DecodeCtx
                                    // (adjust name if your tree renamed it)

namespace AlgoPrIngest
{
    // =========================================================================
    // Inverse-LUT encode helpers. 'lut' is the SAME decode table the ingest
    // used: strictly increasing, lut[0] = decode(0), lut[maxCode] = decode(1).
    // =========================================================================

    // Largest-bracket binary search: first index whose value exceeds 'lin'.
    template <typename LUT>
    inline int lut_upper_index(double lin, const LUT& lut, int maxCode)
    {
        int lo = 0, hi = maxCode;              // invariant: lut[lo] <= lin < lut[hi]
        while (hi - lo > 1) {
            const int mid = lo + ((hi - lo) >> 1);
            if (static_cast<double>(lut[mid]) <= lin) lo = mid; else hi = mid;
        }
        return hi;
    }

    // linear -> NEAREST integer code (exact inverse of code -> lut[code]).
    template <typename LUT>
    inline int code_from_linear(double lin, const LUT& lut, int maxCode)
    {
        if (!(lin > static_cast<double>(lut[0])))       return 0;        // <= black (and NaN)
        if (lin >= static_cast<double>(lut[maxCode]))   return maxCode;  // >= white (HDR clamps)
        const int hi = lut_upper_index(lin, lut, maxCode);
        const int lo = hi - 1;
        const double dLo = lin - static_cast<double>(lut[lo]);
        const double dHi = static_cast<double>(lut[hi]) - lin;
        return (dLo <= dHi) ? lo : hi;
    }

    // linear -> CONTINUOUS encoded value in [0,1] (interpolated inverse).
    template <typename LUT>
    inline double encoded_from_linear(double lin, const LUT& lut, int maxCode)
    {
        if (!(lin > static_cast<double>(lut[0])))       return 0.0;
        if (lin >= static_cast<double>(lut[maxCode]))   return 1.0;
        const int hi = lut_upper_index(lin, lut, maxCode);
        const int lo = hi - 1;
        const double vLo = static_cast<double>(lut[lo]);
        const double vHi = static_cast<double>(lut[hi]);
        const double t   = (vHi > vLo) ? (lin - vLo) / (vHi - vLo) : 0.0;
        return (static_cast<double>(lo) + t) / static_cast<double>(maxCode);
    }

    // Forward luma coefficients derived from the reconstruction constants -
    // the exact inverse of the ingest's YCbCr -> R'G'B'.
    struct RGBToYCbCr { double Kr, Kg, Kb, aR, aB; };
    inline RGBToYCbCr forward_matrix(const YCbCrToRGB& C)
    {
        const double Kr = 1.0 - C.aR * 0.5;
        const double Kb = 1.0 - C.aB * 0.5;
        return { Kr, 1.0 - Kr - Kb, Kb, C.aR, C.aB };
    }

    inline int q8 (double v) { return clamp_index(static_cast<int>(std::lround(v)), 255);   }
    inline int q16(double v) { return clamp_index(static_cast<int>(std::lround(v)), 32767); }

    // =========================================================================
    // Per-format WRITERS - each packs one pixel at (dstRow, x) from linear
    // double R,G,B, taking alpha from the SAME-FORMAT incoming pixel at
    // (alphaRow, ax). Mirrors of the ingest readers; no runtime format branch.
    // =========================================================================

    // ---- integer 8-bit ----
    template <typename Pix, bool Premul>
    struct WriteInt8
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, int32_t x,
                                 const std::uint8_t* alphaRow, int32_t ax,
                                 const Ctx& c, double R, double G, double B)
        {
            Pix* p = reinterpret_cast<Pix*>(dstRow) + x;
            const int a = alphaRow
                ? (reinterpret_cast<const Pix*>(alphaRow) + ax)->A : 255;
            int r = code_from_linear(R, c.lut8, 255);
            int g = code_from_linear(G, c.lut8, 255);
            int b = code_from_linear(B, c.lut8, 255);
            if (Premul) {                                   // re-premultiply (encoded domain)
                const double af = a / 255.0;
                r = q8(r * af); g = q8(g * af); b = q8(b * af);
            }
            p->R = static_cast<decltype(p->R)>(r);
            p->G = static_cast<decltype(p->G)>(g);
            p->B = static_cast<decltype(p->B)>(b);
            p->A = static_cast<decltype(p->A)>(a);          // alpha/X passthrough
        }
    };

    // ---- integer 16-bit (0..32767) ----
    template <typename Pix, bool Premul>
    struct WriteInt16
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, int32_t x,
                                 const std::uint8_t* alphaRow, int32_t ax,
                                 const Ctx& c, double R, double G, double B)
        {
            Pix* p = reinterpret_cast<Pix*>(dstRow) + x;
            const int a = alphaRow
                ? (reinterpret_cast<const Pix*>(alphaRow) + ax)->A : 32767;
            int r = code_from_linear(R, c.lut16, kSRGBLutMax16);
            int g = code_from_linear(G, c.lut16, kSRGBLutMax16);
            int b = code_from_linear(B, c.lut16, kSRGBLutMax16);
            if (Premul) {
                const double af = a / 32767.0;
                r = q16(r * af); g = q16(g * af); b = q16(b * af);
            }
            p->R = static_cast<decltype(p->R)>(r);
            p->G = static_cast<decltype(p->G)>(g);
            p->B = static_cast<decltype(p->B)>(b);
            p->A = static_cast<decltype(p->A)>(a);
        }
    };

    // ---- 32f (gamma-encoded or already-linear), optional premultiplied ----
    template <typename Pix, bool Premul, bool Linear>
    struct WriteF32
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, int32_t x,
                                 const std::uint8_t* alphaRow, int32_t ax,
                                 const Ctx& c, double R, double G, double B)
        {
            Pix* p = reinterpret_cast<Pix*>(dstRow) + x;
            const float a = alphaRow
                ? (reinterpret_cast<const Pix*>(alphaRow) + ax)->A : 1.0f;
            double r, g, b;
            if (Linear) { r = R; g = G; b = B; }            // identity, UNCLAMPED (HDR ok)
            else {
                r = encoded_from_linear(R, c.lut16, kSRGBLutMax16);
                g = encoded_from_linear(G, c.lut16, kSRGBLutMax16);
                b = encoded_from_linear(B, c.lut16, kSRGBLutMax16);
            }
            if (Premul) { r *= a; g *= a; b *= a; }         // re-premultiply
            p->R = static_cast<float>(r);
            p->G = static_cast<float>(g);
            p->B = static_cast<float>(b);
            p->A = a;
        }
    };

    // ---- VUYA / VUYP 8u (studio range) ----
    template <bool Premul>
    struct WriteVUYA8
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_8u);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, int32_t x,
                                 const std::uint8_t* alphaRow, int32_t ax,
                                 const Ctx& c, double R, double G, double B)
        {
            PF_Pixel_VUYA_8u* p = reinterpret_cast<PF_Pixel_VUYA_8u*>(dstRow) + x;
            const int a = alphaRow
                ? (reinterpret_cast<const PF_Pixel_VUYA_8u*>(alphaRow) + ax)->A : 255;
            double Rp = encoded_from_linear(R, c.lut16, kSRGBLutMax16);
            double Gp = encoded_from_linear(G, c.lut16, kSRGBLutMax16);
            double Bp = encoded_from_linear(B, c.lut16, kSRGBLutMax16);
            if (Premul) {                                   // premultiply R'G'B' BEFORE the
                const double af = a / 255.0;                // matrix (inverse of the ingest,
                Rp *= af; Gp *= af; Bp *= af;               // which divided AFTER it)
            }
            const RGBToYCbCr M = forward_matrix(c.C);
            const double Yp = M.Kr * Rp + M.Kg * Gp + M.Kb * Bp;
            const double Cr = (Rp - Yp) / M.aR;
            const double Cb = (Bp - Yp) / M.aB;
            p->Y = static_cast<decltype(p->Y)>(q8(16.0  + Yp * 219.0));
            p->U = static_cast<decltype(p->U)>(q8(128.0 + Cb * 224.0));
            p->V = static_cast<decltype(p->V)>(q8(128.0 + Cr * 224.0));
            p->A = static_cast<decltype(p->A)>(a);
        }
    };

    // ---- VUYA / VUYP 32f (full range, chroma stored at +0.5) ----
    template <bool Premul>
    struct WriteVUYA32
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_32f);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, int32_t x,
                                 const std::uint8_t* alphaRow, int32_t ax,
                                 const Ctx& c, double R, double G, double B)
        {
            PF_Pixel_VUYA_32f* p = reinterpret_cast<PF_Pixel_VUYA_32f*>(dstRow) + x;
            const float a = alphaRow
                ? (reinterpret_cast<const PF_Pixel_VUYA_32f*>(alphaRow) + ax)->A : 1.0f;
            double Rp = encoded_from_linear(R, c.lut16, kSRGBLutMax16);
            double Gp = encoded_from_linear(G, c.lut16, kSRGBLutMax16);
            double Bp = encoded_from_linear(B, c.lut16, kSRGBLutMax16);
            if (Premul) { Rp *= a; Gp *= a; Bp *= a; }
            const RGBToYCbCr M = forward_matrix(c.C);
            const double Yp = M.Kr * Rp + M.Kg * Gp + M.Kb * Bp;
            const double Cr = (Rp - Yp) / M.aR;
            const double Cb = (Bp - Yp) / M.aB;
            p->Y = static_cast<float>(Yp);
            p->U = static_cast<float>(Cb + 0.5);            // mirror the ingest's -0.5
            p->V = static_cast<float>(Cr + 0.5);
            p->A = a;
        }
    };

    // ---- RGB 444 10u (no alpha; packed bitfields) ----
    struct WriteRGB10
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_RGB_10u);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, int32_t x,
                                 const std::uint8_t* /*alphaRow*/, int32_t /*ax*/,
                                 const Ctx& c, double R, double G, double B)
        {
            PF_Pixel_RGB_10u* p = reinterpret_cast<PF_Pixel_RGB_10u*>(dstRow) + x;
            p->R = code_from_linear(R, c.lut10, 1023);
            p->G = code_from_linear(G, c.lut10, 1023);
            p->B = code_from_linear(B, c.lut10, 1023);
        }
    };

    // =========================================================================
    // dispatch_writer - the ONLY format switch (once per frame), 1:1 mirror of
    // dispatch_reader.
    // =========================================================================
    template <typename F>
    inline void dispatch_writer(ePrPixelFormat fmt, F&& f)
    {
        switch (fmt)
        {
            case fmt_BGRA_4444_8u:
            case fmt_BGRX_4444_8u:         f(WriteInt8 <PF_Pixel_BGRA_8u , false>{}); break;
            case fmt_BGRP_4444_8u:         f(WriteInt8 <PF_Pixel_BGRA_8u , true >{}); break;
            case fmt_ARGB_4444_8u:
            case fmt_XRGB_4444_8u:         f(WriteInt8 <PF_Pixel_ARGB_8u , false>{}); break;
            case fmt_PRGB_4444_8u:         f(WriteInt8 <PF_Pixel_ARGB_8u , true >{}); break;

            case fmt_BGRA_4444_16u:
            case fmt_BGRX_4444_16u:        f(WriteInt16<PF_Pixel_BGRA_16u, false>{}); break;
            case fmt_BGRP_4444_16u:        f(WriteInt16<PF_Pixel_BGRA_16u, true >{}); break;
            case fmt_ARGB_4444_16u:
            case fmt_XRGB_4444_16u:        f(WriteInt16<PF_Pixel_ARGB_16u, false>{}); break;
            case fmt_PRGB_4444_16u:        f(WriteInt16<PF_Pixel_ARGB_16u, true >{}); break;

            case fmt_BGRA_4444_32f:
            case fmt_BGRX_4444_32f:        f(WriteF32<PF_Pixel_BGRA_32f, false, false>{}); break;
            case fmt_BGRP_4444_32f:        f(WriteF32<PF_Pixel_BGRA_32f, true , false>{}); break;
            case fmt_BGRA_4444_32f_Linear:
            case fmt_BGRX_4444_32f_Linear: f(WriteF32<PF_Pixel_BGRA_32f, false, true >{}); break;
            case fmt_BGRP_4444_32f_Linear: f(WriteF32<PF_Pixel_BGRA_32f, true , true >{}); break;
            case fmt_ARGB_4444_32f:
            case fmt_XRGB_4444_32f:        f(WriteF32<PF_Pixel_ARGB_32f, false, false>{}); break;
            case fmt_ARGB_4444_32f_Linear:
            case fmt_XRGB_4444_32f_Linear: f(WriteF32<PF_Pixel_ARGB_32f, false, true >{}); break;
            case fmt_PRGB_4444_32f:        f(WriteF32<PF_Pixel_ARGB_32f, true , false>{}); break;
            case fmt_PRGB_4444_32f_Linear: f(WriteF32<PF_Pixel_ARGB_32f, true , true >{}); break;

            case fmt_VUYA_4444_8u_709:
            case fmt_VUYA_4444_8u:         f(WriteVUYA8 <false>{}); break;
            case fmt_VUYP_4444_8u_709:
            case fmt_VUYP_4444_8u:         f(WriteVUYA8 <true >{}); break;
            case fmt_VUYA_4444_32f_709:
            case fmt_VUYA_4444_32f:        f(WriteVUYA32<false>{}); break;
            case fmt_VUYP_4444_32f_709:
            case fmt_VUYP_4444_32f:        f(WriteVUYA32<true >{}); break;
            case fmt_VUYX_4444_8u_709:
            case fmt_VUYX_4444_8u:         f(WriteVUYA8 <false>{}); break;
            case fmt_VUYX_4444_32f_709:
            case fmt_VUYX_4444_32f:        f(WriteVUYA32<false>{}); break;

            case fmt_RGB_444_10u:          f(WriteRGB10{}); break;
            default: break;
        }
    }

    // Branch-free traversal, templated on the selected Writer.
    template <typename Writer, typename Ctx>
    inline void loop_egress(const float* srcRGB_f32, int32_t sizeX, int32_t sizeY,
                            std::uint8_t* dstBase, int32_t dstPitch,
                            const std::uint8_t* alphaBase, int32_t alphaSizeX,
                            int32_t alphaPitch, const Ctx& ctx)
    {
        const std::ptrdiff_t dstStride =
            static_cast<std::ptrdiff_t>(dstPitch)   *
            static_cast<std::ptrdiff_t>(Writer::kPixelBytes);
        const std::ptrdiff_t alpStride =
            static_cast<std::ptrdiff_t>(alphaPitch) *
            static_cast<std::ptrdiff_t>(Writer::kPixelBytes);
        const int32_t axMax = (alphaSizeX > 0) ? (alphaSizeX - 1) : 0;

        for (int32_t y = 0; y < sizeY; ++y)
        {
            const float* srcRow = srcRGB_f32 + static_cast<std::ptrdiff_t>(y) * sizeX * 3;
            std::uint8_t* dstRow = dstBase + static_cast<std::ptrdiff_t>(y) * dstStride;
            const std::uint8_t* alpRow = alphaBase
                ? alphaBase + static_cast<std::ptrdiff_t>(y) * alpStride : nullptr;
            for (int32_t x = 0; x < sizeX; ++x)
            {
                const float* s = srcRow + static_cast<std::ptrdiff_t>(x) * 3;
                const int32_t ax = (x < axMax) ? x : axMax;   // clamp into alpha width
                Writer::write(dstRow, x, alpRow, ax, ctx,
                              static_cast<double>(s[0]),
                              static_cast<double>(s[1]),
                              static_cast<double>(s[2]));
            }
        }
    }

    // =========================================================================
    // PUBLIC ENTRY POINT
    //
    // Writes the linear float32 RGB buffer 'srcRGB_f32' (tightly packed,
    // sizeX*sizeY*3, as produced by ingest_to_linear_f32 /
    // ingest_and_superpixel) into the Adobe frame buffer 'dst' of format
    // 'fmt', taking per-pixel ALPHA (and the X pad of the ...X formats) from
    // the ORIGINAL incoming Adobe buffer 'alphaSrc', which must be of the
    // SAME pixel format 'fmt'.
    //
    //   sizeX, sizeY : frame size in PIXELS (linear buffer geometry).
    //   dstPitch     : destination pitch in PIXELS (negative = bottom-up).
    //   lut8/16/10   : the SAME decode LUTs handed to the ingest - the egress
    //                  inverts them, so encode always matches the transfer
    //                  the frame was linearized with.
    //   alphaSrc     : incoming Adobe buffer (alpha source). May be nullptr:
    //                  full opacity is written (255 / 32767 / 1.0f).
    //   alphaSizeX   : incoming buffer width  in PIXELS (x is clamped to it).
    //   alphaPitch   : incoming buffer pitch in PIXELS (negative allowed).
    //
    // Notes: HDR linear values encode-clamp to white for integer / encoded
    // outputs; _Linear outputs are copied unclamped. For premultiplied
    // formats with alpha == 0 the color channels are written as 0 (the
    // color under zero alpha is undefined by convention). fmt_RGB_444_10u
    // has no alpha - the alpha parameters are ignored for it.
    // =========================================================================
    template <typename LUT8, typename LUT16, typename LUT10>
    void egress_from_linear_f32
    (
        const float* srcRGB_f32,
        int32_t sizeX, int32_t sizeY,
        void* dst, int32_t dstPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        const void* alphaSrc, int32_t alphaSizeX, int32_t alphaPitch
    )
    {
        const DecodeCtx<LUT8, LUT16, LUT10> ctx{ lut8, lut16, lut10, pick_matrix(fmt) };
        std::uint8_t* dstBase = static_cast<std::uint8_t*>(dst);
        const std::uint8_t* alpBase = static_cast<const std::uint8_t*>(alphaSrc);
        dispatch_writer(fmt, [&](auto writer) {
            using W = decltype(writer);
            loop_egress<W>(srcRGB_f32, sizeX, sizeY, dstBase, dstPitch,
                           alpBase, alphaSizeX, alphaPitch, ctx);
        });
    }

} // namespace AlgoPrIngest

#endif // __IMAGELAB2_PR_FORMAT_EGRESS_HPP__
