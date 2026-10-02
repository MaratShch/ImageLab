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
#include "AlgoSuperPixel.hpp"     // SuperPixel<> (and the reference two-pass path)

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
        fmt_RGB_444_10u,
        fmt_ARGB_4444_8u,          fmt_ARGB_4444_16u,
        fmt_ARGB_4444_32f,         fmt_ARGB_4444_32f_Linear
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
            case fmt_ARGB_4444_8u:         f(ReadInt8 <PF_Pixel_ARGB_8u , false>{}); break;

            case fmt_BGRA_4444_16u:
            case fmt_BGRX_4444_16u:        f(ReadInt16<PF_Pixel_BGRA_16u, false>{}); break;
            case fmt_BGRP_4444_16u:        f(ReadInt16<PF_Pixel_BGRA_16u, true >{}); break;
            case fmt_ARGB_4444_16u:        f(ReadInt16<PF_Pixel_ARGB_16u, false>{}); break;

            case fmt_BGRA_4444_32f:
            case fmt_BGRX_4444_32f:        f(ReadF32<PF_Pixel_BGRA_32f, false, false>{}); break;
            case fmt_BGRP_4444_32f:        f(ReadF32<PF_Pixel_BGRA_32f, true , false>{}); break;
            case fmt_BGRA_4444_32f_Linear:
            case fmt_BGRX_4444_32f_Linear: f(ReadF32<PF_Pixel_BGRA_32f, false, true >{}); break;
            case fmt_BGRP_4444_32f_Linear: f(ReadF32<PF_Pixel_BGRA_32f, true , true >{}); break;
            case fmt_ARGB_4444_32f:        f(ReadF32<PF_Pixel_ARGB_32f, false, false>{}); break;
            case fmt_ARGB_4444_32f_Linear: f(ReadF32<PF_Pixel_ARGB_32f, false, true >{}); break;

            case fmt_VUYA_4444_8u_709:
            case fmt_VUYA_4444_8u:         f(ReadVUYA8 <false>{}); break;
            case fmt_VUYP_4444_8u_709:
            case fmt_VUYP_4444_8u:         f(ReadVUYA8 <true >{}); break;
            case fmt_VUYA_4444_32f_709:
            case fmt_VUYA_4444_32f:        f(ReadVUYA32<false>{}); break;
            case fmt_VUYP_4444_32f_709:
            case fmt_VUYP_4444_32f:        f(ReadVUYA32<true >{}); break;

            case fmt_RGB_444_10u:          f(ReadRGB10{}); break;
            default: break;
        }
    }

    // Rec.601/709 selection (once per frame, outside the loop).
    inline YCbCrToRGB pick_matrix(ePrPixelFormat fmt)
    {
        const bool is709 = (fmt == fmt_VUYA_4444_8u_709 || fmt == fmt_VUYA_4444_32f_709 ||
                            fmt == fmt_VUYP_4444_8u_709 || fmt == fmt_VUYP_4444_32f_709);
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
    template <typename Reader, typename Ctx>
    inline void loop_fused(const std::uint8_t* base, int32_t sizeX, int32_t sizeY,
                           int32_t srcPitch, const Ctx& ctx, float* dstRGB_f32,
                           SuperPixel<double>& super)
    {
        // srcPitch is in PIXELS (may be negative). -> signed byte stride.
        const std::ptrdiff_t byteStride =
            static_cast<std::ptrdiff_t>(srcPitch) *
            static_cast<std::ptrdiff_t>(Reader::kPixelBytes);
        constexpr double kYDark     = 0.010;
        constexpr double kChClip    = 0.95;
        constexpr double kTaperLo   = 0.90;
        constexpr double kSatFull   = 0.20;
        constexpr double kSatMax    = 0.60;
        constexpr double kEnergyMin = 1.0e-6;
        constexpr double kLumaR     = 0.2627;   // linear Rec.2020 working space
        constexpr double kLumaG     = 0.6780;
        constexpr double kLumaB     = 0.0593;

        double rSum = 0.0, gSum = 0.0, bSum = 0.0, wSum = 0.0;

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

                // --- super-pixel exclusion / weighting (double) ---
                if (!(std::isfinite(R) && std::isfinite(G) && std::isfinite(B))) continue;
                if (R < 0.0 || G < 0.0 || B < 0.0)                               continue;
                if ((R + G + B) <= kEnergyMin)                                   continue;
                const double maxc = (R > G) ? ((R > B) ? R : B) : ((G > B) ? G : B);
                if (maxc >= kChClip)                                            continue;
                const double Y = kLumaR * R + kLumaG * G + kLumaB * B;
                if (Y < kYDark)                                                 continue;
                const double wLum = (maxc > kTaperLo)
                                  ? (kChClip - maxc) / (kChClip - kTaperLo) : 1.0;
                const double minc = (R < G) ? ((R < B) ? R : B) : ((G < B) ? G : B);
                const double sat  = (maxc - minc) / maxc;
                if (sat > kSatMax)                                              continue;
                const double wSat = (sat > kSatFull)
                                  ? (kSatMax - sat) / (kSatMax - kSatFull) : 1.0;
                const double w = wLum * wSat;
                if (w <= 0.0)                                                   continue;
                rSum += w * R; gSum += w * G; bSum += w * B; wSum += w;
            }
        }

        if (wSum > 0.0) {
            const double inv = 1.0 / wSum;
            super.r = rSum * inv; super.g = gSum * inv; super.b = bSum * inv;
        } else {
            super.r = super.g = super.b = 0.0;
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
    template <typename LUT8, typename LUT16, typename LUT10>
    void ingest_and_superpixel
    (
        const void* src, int32_t sizeX, int32_t sizeY, int32_t srcPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        float* dstRGB_f32,
        SuperPixel<double>& super
    )
    {
        const DecodeCtx<LUT8, LUT16, LUT10> ctx{ lut8, lut16, lut10, pick_matrix(fmt) };
        const std::uint8_t* base = static_cast<const std::uint8_t*>(src);
        dispatch_reader(fmt, [&](auto reader) {
            using R = decltype(reader);
            loop_fused<R>(base, sizeX, sizeY, srcPitch, ctx, dstRGB_f32, super);
        });
    }

} // namespace AlgoPrIngest

#endif // __IMAGELAB2_PR_FORMAT_INGEST_HPP__
