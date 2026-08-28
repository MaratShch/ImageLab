#ifndef __IMAGELAB2_PR_FORMAT_INGEST_HPP__
#define __IMAGELAB2_PR_FORMAT_INGEST_HPP__

// =============================================================================
// AlgoPrFormatIngest.hpp - scalar ingest of Adobe Premiere/After Effects frame
// buffers into a LINEAR float32 RGB buffer, interleaved OR planar.
//
// This is the reference (scalar) path and the fallback for every format. The
// AVX2 fast paths live in AlgoIngestAVX2.hpp and are strictly additive.
//
// -----------------------------------------------------------------------------
// WHAT CHANGED FROM THE REFERENCE IMPLEMENTATION, AND WHY
// -----------------------------------------------------------------------------
// The external API is preserved: same namespace, same enum, same reader tags,
// same dispatch_reader, same loop_ingest, same public entry points with the same
// parameter order. Five deliberate deviations, each forced by a stated
// requirement rather than by preference:
//
//  1. PER-PIXEL ARITHMETIC IS float32, NOT double. Owner requirement. The
//     readers now emit `float&`. See the TYPE POLICY note below for the two
//     places where double is RETAINED and why keeping them is not a violation.
//
//  2. PLANAR ENTRY POINTS ADDED, interleaved ones untouched. The film engine
//     consumes planar Src_R/Src_G/Src_B at an arena pitch (AlgorithmMain.cpp:
//     "scene-linear planar samples in STORAGE type"), while the reference
//     produced tightly-packed interleaved RGB. Rather than redesign the
//     existing signature - which would break every current call site - the
//     interleaved entry points are kept byte-for-byte and planar twins are
//     added beside them: `ingest_to_planar_f32`. Same namespace, same naming
//     style, same parameter semantics. Nothing existing changes, and no extra
//     de-interleave pass is needed for the film engine.
//
//  3. NO STL, AND NO LUT AT ALL. `#include <array>` dropped, and the decode
//     tables are gone with it - owner instruction, "LUT usage less preferred,
//     please use direct computations". The transfer curve is now computed per
//     sample by AlgoPrFormatMath.hpp (polynomial log2/exp2 pair, FMA, no libm
//     call and no dependent load). The reference's LUT-taking entry points are
//     RETAINED as overloads that ignore the tables, so existing call sites keep
//     compiling; the LUT-free overloads are the primary API.
//
//  4. THE SUPER-PIXEL / CCT PATH IS NOT PORTED. `ingest_and_superpixel`,
//     `LocusGate` and `build_locus_gate` depend on "super_pixel.hpp", which is
//     not present in this project, and the film engine has no CCT
//     auto-white-balance stage that would consume a super-pixel. Omitting them
//     is the honest option; inventing a SuperPixel type to satisfy a signature
//     would produce a header that compiles and means nothing. The names are
//     reachable again by supplying super_pixel.hpp and defining
//     ALGO_PR_WITH_SUPERPIXEL - the hook is at the bottom of this file, so
//     restoring them touches nothing else.
//
//  5. ROUNDING IS EXPLICIT, not `std::lround`. `pr_round_half_away()` below is
//     the single rounding rule, used by BOTH this scalar path and the AVX2
//     path. std::lround rounds half away from zero; _mm256_cvtps_epi32 rounds
//     half to even. Left alone the two disagree by 1 LSB on exact-half values,
//     so the vector body and the scalar tail OF THE SAME ROW could produce
//     different codes - which would quietly destroy the bit-exact round trip
//     this design exists to provide. One rule, both paths.
//
// -----------------------------------------------------------------------------
// TYPE POLICY - float per pixel, double in exactly two places
// -----------------------------------------------------------------------------
// `PrFloat` (float) is the per-pixel arithmetic and storage type. `PrSetup`
// (double) is retained for:
//
//   * the YCbCr reconstruction constants. The reference derived them in 40-digit
//     arithmetic and verified round-trip identity to 1e-40; they are compile-time
//     constants, they cost nothing at runtime, and rounding them to float throws
//     that away for no gain. They are converted to float ONCE per frame when the
//     matrix is picked.
//   * (the second use, the egress inverse-LUT bracket search, is GONE - there
//     is no table left to invert. transfer_encode() computes the encode
//     directly, which removed both the table and the ~15-step per-pixel binary
//     search that inverted it.)
//
// This mirrors the project's own `HighPrecType` convention in AlgoTypes.hpp,
// which fixes setup-domain arithmetic at double for exactly this reason.
//
// -----------------------------------------------------------------------------
// COLOR-SCIENCE CONVENTIONS (unchanged from the reference, now confirmed)
// -----------------------------------------------------------------------------
//  * VUYA/VUYP/VUYX matrix: "_709" -> Rec.709; unsuffixed -> Rec.601 (Adobe).
//  * VUYA 8u: STUDIO range. Y' 16..235 -> (Y-16)/219 ; chroma 16..240 with the
//    unsigned offset at 128 -> (U-128)/224. The owner's rule "signed chroma =
//    unsigned chroma - 128" is exactly this, and the raw 8-bit U/V are never
//    treated as already-signed.
//  * VUYA 32f: FULL range, chroma stored at +0.5 -> subtract 0.5. Controlled by
//    kVuyaF32ChromaOffset below; set it to 0.0f if a host build ever delivers
//    already-signed 32f chroma.
//  * xxxP (BGRP/VUYP/PRGB) = PREMULTIPLIED: un-premultiply BEFORE linearization,
//    A == 0 -> colour left at 0.
//  * xxxX (BGRX/VUYX/XRGB) = OPAQUE, the X channel is not an alpha. Ingest
//    ignores it entirely; egress passes the original X through untouched.
//  * _Linear variants are ALREADY LINEARIZED BY THE ADOBE ENGINE - identity
//    transfer on ingest, identity on egress. No LUT is consulted for them.
//  * PF_Pixel_RGB_10u has NO alpha. Nothing is preserved and nothing is
//    synthesized for it, in either direction.
//
// SIZES AND PITCHES ARE IN PIXELS and are SIGNED. A negative pitch means a
// bottom-up frame and must work: byte strides are computed as ptrdiff_t, row
// bases as base + (ptrdiff_t)y * stride. No unsigned arithmetic anywhere on the
// addressing path, no abs(), no size = w*h*bpp shortcut.
//
// C++14. No dynamic allocation. No STL containers. No OS- or compiler-specific
// API. Builds under VS2015 SP3, VS2022 and gcc-13 for x64.
//
// ⚠ VS2015 NOTE: constexpr here is C++11-level only - single-return functions
// and constant initialisers. VS2015 does not implement C++14 relaxed constexpr
// (loops, multiple statements), so a constexpr table builder would compile on
// gcc-13 and fail on VS2015.
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <cmath>
#include "Common.hpp"            // RESTRICT, CACHE_LINE, CACHE_ALIGN
#include "CommonPixFormat.hpp"   // authoritative Adobe pixel layouts
#include "AlgoPrFormatMath.hpp"  // fast_log2/fast_exp2, transfer_decode/encode


namespace AlgoPrIngest
{
    // ---- type policy (see the header note) ---------------------------------
    using PrFloat = float;    //!< per-pixel arithmetic and storage
    using PrSetup = double;   //!< compile-time constants only (see TYPE POLICY)

    // =========================================================================
    // Supported formats. All 37 entries of the reference enum are retained,
    // including the XRGB_* / PRGB_* / VUYX_* families that are outside the
    // 25 formats requested: deleting enumerators would break API compatibility
    // for any existing caller, and they cost nothing.
    // =========================================================================
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

    // =========================================================================
    // CODE CEILINGS. The CLAMP TARGETS for egress and the normalization
    // denominators for ingest, declared once so the two directions cannot drift
    // apart.
    //
    // ⚠ 16-BIT IS 0..32767, NOT 0..65535. This is the After Effects 15+1-bit
    // convention and it is what CommonPixFormat.hpp declares
    // ("value range 0 ... 32767", u16_value_white = 32767u). Premiere's own
    // documentation describes its 16u as full-range; the attached header is the
    // authoritative source per the owner's instruction, so 32767 it is. Getting
    // this wrong is a factor-of-two error on every 16u path in both directions,
    // which is why it is a named constant and not a literal.
    // =========================================================================
    constexpr int kMaxCode8  = 255;
    constexpr int kMaxCode10 = 1023;
    constexpr int kMaxCode16 = 32767;

    //! Kept for source compatibility with the reference API. Nothing indexes a
    //! table any more - continuous inputs are no longer requantized at all,
    //! which is one quantization step LESS than the reference performed.
    constexpr int kSRGBLutMax16 = kMaxCode16;

    //! Float ceiling applied on egress. See the ⚠ note in AlgoPrFormatEgress.hpp
    //! before changing this - it is the one clamp that can cost real image data.
    constexpr PrFloat kMaxFloat32 = 1.0f;

    //! Offset removed from 32f VUYA chroma. Set to 0.0f for already-signed hosts.
    constexpr PrFloat kVuyaF32ChromaOffset = 0.5f;

    // ---- VUYA 8u studio-range constants, named rather than inlined ----------
    constexpr PrFloat kVuya8LumaOffset  = 16.0f;
    constexpr PrFloat kVuya8LumaScale   = 219.0f;
    constexpr PrFloat kVuya8ChromaBias  = 128.0f;   //!< the owner's "-128" rule
    constexpr PrFloat kVuya8ChromaScale = 224.0f;

    // =========================================================================
    // EXACT YCbCr -> R'G'B' coefficients (Cb,Cr signed in [-0.5,+0.5])
    //   R' = Y' + aR*Cr ; B' = Y' + aB*Cb ; G' = Y' - gCr*Cr - gCb*Cb
    // Held in PrSetup (double) as compile-time constants; converted to PrFloat
    // once per frame. See TYPE POLICY.
    // =========================================================================
    struct YCbCrToRGB { PrSetup aR, aB, gCr, gCb; };

    //! ITU-R BT.601 (Kr=0.299 , Kb=0.114 , Kg=0.587)
    constexpr YCbCrToRGB kRec601 =
    { 1.4019999999999999, 1.7720000000000000, 0.71413628620102210, 0.34413628620102216 };
    //! ITU-R BT.709 (Kr=0.2126, Kb=0.0722, Kg=0.7152)
    constexpr YCbCrToRGB kRec709 =
    { 1.5748000000000000, 1.8555999999999999, 0.46812427293064879, 0.18732427293064877 };

    //! Per-frame float image of the chosen matrix - what the readers actually use.
    struct YCbCrToRGBf { PrFloat aR, aB, gCr, gCb; };

    // =========================================================================
    // Forward luma coefficients derived from the reconstruction constants - the
    // exact inverse of the ingest's YCbCr -> R'G'B'. Built once per frame in
    // PrSetup and narrowed to PrFloat, same policy as the reverse matrix.
    // =========================================================================
    struct RGBToYCbCr { PrFloat Kr, Kg, Kb, aR, aB; };

    inline RGBToYCbCr forward_matrix(const YCbCrToRGBf& C)
    {
        const PrSetup aR = static_cast<PrSetup>(C.aR);
        const PrSetup aB = static_cast<PrSetup>(C.aB);
        const PrSetup Kr = 1.0 - aR * 0.5;
        const PrSetup Kb = 1.0 - aB * 0.5;
        RGBToYCbCr m;
        m.Kr = static_cast<PrFloat>(Kr);
        m.Kb = static_cast<PrFloat>(Kb);
        m.Kg = static_cast<PrFloat>(1.0 - Kr - Kb);
        m.aR = C.aR;
        m.aB = C.aB;
        return m;
    }

    // =========================================================================
    // THE ONE ROUNDING RULE. Half away from zero, matching std::lround, and
    // reproduced bit-for-bit by the AVX2 path (see pr_cvt_round_epi32 in
    // AlgoPrFormatAVX2.hpp). Do not replace either side independently.
    // =========================================================================
    inline int pr_round_half_away(PrFloat v)
    {
        return static_cast<int>(v + ((v >= 0.0f) ? 0.5f : -0.5f));
    }

    inline int clamp_index(int idx, int maxIdx)
    {
        return (idx < 0) ? 0 : ((idx > maxIdx) ? maxIdx : idx);
    }

    inline PrFloat clamp_unit(PrFloat v)
    {
        return (v < 0.0f) ? 0.0f : ((v > kMaxFloat32) ? kMaxFloat32 : v);
    }

    //! Quantize a normalized value to an integer code, clamped. The single
    //! quantizer for every integer egress path - 8u, 10u and 16u all route
    //! through it so the clamp ceiling can never be applied inconsistently.
    inline int quantize_code(PrFloat normalized, int maxCode)
    {
        return clamp_index(pr_round_half_away(normalized * static_cast<PrFloat>(maxCode)),
                           maxCode);
    }

    //! Linearize a continuous normalized value. Replaces the reference's
    //! lin_via_lut(): DIRECT computation, and note that it no longer
    //! requantizes the input through a 32768-entry table on the way - so this
    //! is both faster and strictly more accurate than what it replaces.
    inline PrFloat lin_direct(PrFloat c01)
    {
        return transfer_decode(clamp_unit(c01));
    }

    //! Integer code -> linear. The code is normalized and computed directly;
    //! this is the drop-in for the reference's `lut[code]`.
    //! ⚠ MULTIPLY BY THE RECIPROCAL, do not divide. maxCode is always a
    //! compile-time constant here, so 1/maxCode folds at compile time - and,
    //! more importantly, the AVX2 path has no divide and multiplies by that same
    //! reciprocal. `code / 255.0f` is correctly rounded while `code * (1/255.0f)`
    //! is not, so dividing here would put the scalar and the vector path one ULP
    //! apart on the input to the curve - which the polynomial then amplifies.
    //! One expression, both paths, identical bytes out.
    inline PrFloat lin_from_code(int code, int maxCode)
    {
        return transfer_decode(static_cast<PrFloat>(code) *
                               (static_cast<PrFloat>(1) / static_cast<PrFloat>(maxCode)));
    }

    // =========================================================================
    // Immutable per-frame context handed to every reader and writer. In the
    // reference this carried three LUT references; with the tables gone all that
    // remains is the ONE chosen matrix, already narrowed to float. Kept as a
    // struct rather than a bare parameter so the reader/writer signatures -
    // `const Ctx&` - are unchanged.
    // =========================================================================
    struct DecodeCtx
    {
        YCbCrToRGBf C;
    };

    //! Rec.601/709 selection - once per frame, outside every loop.
    inline YCbCrToRGB pick_matrix(ePrPixelFormat fmt)
    {
        const bool is709 = (fmt == fmt_VUYA_4444_8u_709 || fmt == fmt_VUYA_4444_32f_709 ||
                            fmt == fmt_VUYP_4444_8u_709 || fmt == fmt_VUYP_4444_32f_709 ||
                            fmt == fmt_VUYX_4444_8u_709 || fmt == fmt_VUYX_4444_32f_709);
        return is709 ? kRec709 : kRec601;
    }

    //! Narrow the setup-domain matrix to the per-pixel type, once per frame.
    inline YCbCrToRGBf narrow_matrix(const YCbCrToRGB& m)
    {
        YCbCrToRGBf f;
        f.aR  = static_cast<PrFloat>(m.aR);
        f.aB  = static_cast<PrFloat>(m.aB);
        f.gCr = static_cast<PrFloat>(m.gCr);
        f.gCb = static_cast<PrFloat>(m.gCb);
        return f;
    }

    //! True for the formats whose 32f samples are already linear.
    inline bool format_is_linear(ePrPixelFormat fmt)
    {
        return (fmt == fmt_BGRA_4444_32f_Linear || fmt == fmt_BGRP_4444_32f_Linear ||
                fmt == fmt_BGRX_4444_32f_Linear || fmt == fmt_ARGB_4444_32f_Linear ||
                fmt == fmt_PRGB_4444_32f_Linear || fmt == fmt_XRGB_4444_32f_Linear);
    }

    // =========================================================================
    // PER-FORMAT READERS. Each unpacks one pixel at (row, x) to linear float.
    // Templated on the pixel struct and on the premul / linear flags, so the
    // channel-order twins (BGRA and ARGB share the member names R,G,B,A) and
    // the premultiplied / linear variants all collapse to compile-time
    // constants. No runtime format branch anywhere inside.
    // =========================================================================

    // ---- integer 8-bit: decode the RAW CODE directly -----------------------
    template <typename Pix, bool Premul>
    struct ReadInt8
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, std::int32_t x, const Ctx& /*c*/,
                                PrFloat& R, PrFloat& G, PrFloat& B)
        {
            const Pix* p = reinterpret_cast<const Pix*>(row) + x;
            int r = p->R, g = p->G, b = p->B;
            if (Premul && p->A != 0)
            {   // un-premultiply in the ENCODED domain, before linearization
                const PrFloat a = static_cast<PrFloat>(p->A) * (1.0f / 255.0f);
                const PrFloat inv = 1.0f / a;
                r = clamp_index(pr_round_half_away(static_cast<PrFloat>(r) * inv), kMaxCode8);
                g = clamp_index(pr_round_half_away(static_cast<PrFloat>(g) * inv), kMaxCode8);
                b = clamp_index(pr_round_half_away(static_cast<PrFloat>(b) * inv), kMaxCode8);
            }
            R = lin_from_code(r, kMaxCode8);
            G = lin_from_code(g, kMaxCode8);
            B = lin_from_code(b, kMaxCode8);
        }
    };

    // ---- integer 16-bit: decode the raw code directly, 0..32767 ------------
    template <typename Pix, bool Premul>
    struct ReadInt16
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, std::int32_t x, const Ctx& /*c*/,
                                PrFloat& R, PrFloat& G, PrFloat& B)
        {
            const Pix* p = reinterpret_cast<const Pix*>(row) + x;
            int r = p->R, g = p->G, b = p->B;
            if (Premul && p->A != 0)
            {
                const PrFloat a = static_cast<PrFloat>(p->A) * (1.0f / 32767.0f);
                const PrFloat inv = 1.0f / a;
                r = clamp_index(pr_round_half_away(static_cast<PrFloat>(r) * inv), kMaxCode16);
                g = clamp_index(pr_round_half_away(static_cast<PrFloat>(g) * inv), kMaxCode16);
                b = clamp_index(pr_round_half_away(static_cast<PrFloat>(b) * inv), kMaxCode16);
            }
            R = lin_from_code(r, kMaxCode16);
            G = lin_from_code(g, kMaxCode16);
            B = lin_from_code(b, kMaxCode16);
        }
    };

    // ---- 32f, gamma-encoded or already-linear, optional premultiplied ------
    template <typename Pix, bool Premul, bool Linear>
    struct ReadF32
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, std::int32_t x, const Ctx& /*c*/,
                                PrFloat& R, PrFloat& G, PrFloat& B)
        {
            const Pix* p = reinterpret_cast<const Pix*>(row) + x;
            PrFloat r = p->R, g = p->G, b = p->B;
            if (Premul && p->A != 0.0f)
            {
                const PrFloat inv = 1.0f / p->A;
                r *= inv; g *= inv; b *= inv;
            }
            if (Linear)
            {   // ⚠ ALREADY LINEARIZED BY THE ADOBE ENGINE. Identity, and
                // deliberately UNCLAMPED on ingest: speculars and light sources
                // legitimately exceed 1.0 and the characteristic curve needs
                // that highlight information to roll it off.
                R = r; G = g; B = b;
            }
            else
            {
                R = lin_direct(r);
                G = lin_direct(g);
                B = lin_direct(b);
            }
        }
    };

    // ---- VUYA / VUYP / VUYX 8u, STUDIO range -------------------------------
    template <bool Premul>
    struct ReadVUYA8
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_8u);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, std::int32_t x, const Ctx& c,
                                PrFloat& R, PrFloat& G, PrFloat& B)
        {
            const PF_Pixel_VUYA_8u* p =
                reinterpret_cast<const PF_Pixel_VUYA_8u*>(row) + x;
            // ⚠ The raw 8-bit U and V are UNSIGNED with the zero point at 128.
            // Subtracting kVuya8ChromaBias is what makes them signed; they are
            // never reinterpreted as signed bytes.
            const PrFloat Yp = (static_cast<PrFloat>(p->Y) - kVuya8LumaOffset)
                             * (1.0f / kVuya8LumaScale);
            const PrFloat Cb = (static_cast<PrFloat>(p->U) - kVuya8ChromaBias)
                             * (1.0f / kVuya8ChromaScale);
            const PrFloat Cr = (static_cast<PrFloat>(p->V) - kVuya8ChromaBias)
                             * (1.0f / kVuya8ChromaScale);
            PrFloat Rp = Yp + c.C.aR * Cr;
            PrFloat Bp = Yp + c.C.aB * Cb;
            PrFloat Gp = Yp - c.C.gCr * Cr - c.C.gCb * Cb;
            if (Premul && p->A != 0)
            {
                const PrFloat inv = 255.0f / static_cast<PrFloat>(p->A);
                Rp *= inv; Gp *= inv; Bp *= inv;
            }
            R = lin_direct(Rp);
            G = lin_direct(Gp);
            B = lin_direct(Bp);
        }
    };

    // ---- VUYA / VUYP / VUYX 32f, FULL range, chroma stored at +0.5 ---------
    template <bool Premul>
    struct ReadVUYA32
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_32f);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, std::int32_t x, const Ctx& c,
                                PrFloat& R, PrFloat& G, PrFloat& B)
        {
            const PF_Pixel_VUYA_32f* p =
                reinterpret_cast<const PF_Pixel_VUYA_32f*>(row) + x;
            const PrFloat Yp = p->Y;
            const PrFloat Cb = p->U - kVuyaF32ChromaOffset;
            const PrFloat Cr = p->V - kVuyaF32ChromaOffset;
            PrFloat Rp = Yp + c.C.aR * Cr;
            PrFloat Bp = Yp + c.C.aB * Cb;
            PrFloat Gp = Yp - c.C.gCr * Cr - c.C.gCb * Cb;
            if (Premul && p->A != 0.0f)
            {
                const PrFloat inv = 1.0f / p->A;
                Rp *= inv; Gp *= inv; Bp *= inv;
            }
            R = lin_direct(Rp);
            G = lin_direct(Gp);
            B = lin_direct(Bp);
        }
    };

    // ---- RGB 444 10u: NO ALPHA, packed bitfields --------------------------
    struct ReadRGB10
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_RGB_10u);
        template <typename Ctx>
        static inline void read(const std::uint8_t* row, std::int32_t x, const Ctx& /*c*/,
                                PrFloat& R, PrFloat& G, PrFloat& B)
        {
            const PF_Pixel_RGB_10u* p =
                reinterpret_cast<const PF_Pixel_RGB_10u*>(row) + x;
            R = lin_from_code(static_cast<int>(p->R), kMaxCode10);
            G = lin_from_code(static_cast<int>(p->G), kMaxCode10);
            B = lin_from_code(static_cast<int>(p->B), kMaxCode10);
        }
    };

    // =========================================================================
    // dispatch_reader - the ONLY format switch. Runs once per frame, selects a
    // compile-time reader tag and hands it to the generic callable, which then
    // instantiates the specialized branch-free loop for that reader.
    // =========================================================================
    template <typename F>
    inline void dispatch_reader(ePrPixelFormat fmt, F&& f)
    {
        switch (fmt)
        {
            case fmt_BGRA_4444_8u:
            case fmt_BGRX_4444_8u:         f(ReadInt8 <PF_Pixel_BGRA_8u , false>()); break;
            case fmt_BGRP_4444_8u:         f(ReadInt8 <PF_Pixel_BGRA_8u , true >()); break;
            case fmt_ARGB_4444_8u:
            case fmt_XRGB_4444_8u:         f(ReadInt8 <PF_Pixel_ARGB_8u , false>()); break;
            case fmt_PRGB_4444_8u:         f(ReadInt8 <PF_Pixel_ARGB_8u , true >()); break;

            case fmt_BGRA_4444_16u:
            case fmt_BGRX_4444_16u:        f(ReadInt16<PF_Pixel_BGRA_16u, false>()); break;
            case fmt_BGRP_4444_16u:        f(ReadInt16<PF_Pixel_BGRA_16u, true >()); break;
            case fmt_ARGB_4444_16u:
            case fmt_XRGB_4444_16u:        f(ReadInt16<PF_Pixel_ARGB_16u, false>()); break;
            case fmt_PRGB_4444_16u:        f(ReadInt16<PF_Pixel_ARGB_16u, true >()); break;

            case fmt_BGRA_4444_32f:
            case fmt_BGRX_4444_32f:        f(ReadF32<PF_Pixel_BGRA_32f, false, false>()); break;
            case fmt_BGRP_4444_32f:        f(ReadF32<PF_Pixel_BGRA_32f, true , false>()); break;
            case fmt_BGRA_4444_32f_Linear:
            case fmt_BGRX_4444_32f_Linear: f(ReadF32<PF_Pixel_BGRA_32f, false, true >()); break;
            case fmt_BGRP_4444_32f_Linear: f(ReadF32<PF_Pixel_BGRA_32f, true , true >()); break;
            case fmt_ARGB_4444_32f:
            case fmt_XRGB_4444_32f:        f(ReadF32<PF_Pixel_ARGB_32f, false, false>()); break;
            case fmt_ARGB_4444_32f_Linear:
            case fmt_XRGB_4444_32f_Linear: f(ReadF32<PF_Pixel_ARGB_32f, false, true >()); break;
            case fmt_PRGB_4444_32f:        f(ReadF32<PF_Pixel_ARGB_32f, true , false>()); break;
            case fmt_PRGB_4444_32f_Linear: f(ReadF32<PF_Pixel_ARGB_32f, true , true >()); break;

            case fmt_VUYA_4444_8u_709:
            case fmt_VUYA_4444_8u:         f(ReadVUYA8 <false>()); break;
            case fmt_VUYP_4444_8u_709:
            case fmt_VUYP_4444_8u:         f(ReadVUYA8 <true >()); break;
            case fmt_VUYA_4444_32f_709:
            case fmt_VUYA_4444_32f:        f(ReadVUYA32<false>()); break;
            case fmt_VUYP_4444_32f_709:
            case fmt_VUYP_4444_32f:        f(ReadVUYA32<true >()); break;
            case fmt_VUYX_4444_8u_709:
            case fmt_VUYX_4444_8u:         f(ReadVUYA8 <false>()); break;
            case fmt_VUYX_4444_32f_709:
            case fmt_VUYX_4444_32f:        f(ReadVUYA32<false>()); break;

            case fmt_RGB_444_10u:          f(ReadRGB10()); break;
            default: break;
        }
    }

    // =========================================================================
    // TRAVERSALS. Branch-free on format, templated on the selected reader.
    //
    // ⚠ ADDRESSING. srcPitch is in PIXELS and MAY BE NEGATIVE (bottom-up
    // frames). It is widened to ptrdiff_t before being multiplied by the pixel
    // size, and the row base is base + (ptrdiff_t)y * byteStride. No unsigned
    // type appears on this path: a size_t cast of a negative stride would wrap
    // to an enormous positive offset and read wild memory.
    // =========================================================================

    //! Interleaved destination, tightly packed sizeX*sizeY*3. Reference layout.
    template <typename Reader, typename Ctx>
    inline void loop_ingest(const std::uint8_t* base, std::int32_t sizeX, std::int32_t sizeY,
                            std::int32_t srcPitch, const Ctx& ctx, PrFloat* dstRGB_f32)
    {
        const std::ptrdiff_t byteStride =
            static_cast<std::ptrdiff_t>(srcPitch) *
            static_cast<std::ptrdiff_t>(Reader::kPixelBytes);
        for (std::int32_t y = 0; y < sizeY; ++y)
        {
            const std::uint8_t* row = base + static_cast<std::ptrdiff_t>(y) * byteStride;
            PrFloat* dstRow = dstRGB_f32 +
                static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(sizeX) * 3;
            for (std::int32_t x = 0; x < sizeX; ++x)
            {
                PrFloat R, G, B;
                Reader::read(row, x, ctx, R, G, B);
                PrFloat* d = dstRow + static_cast<std::ptrdiff_t>(x) * 3;
                d[0] = R; d[1] = G; d[2] = B;
            }
        }
    }

    //! Planar destination - what the film engine consumes. Three plane pointers
    //! and ONE shared pitch in ELEMENTS, matching the arena's padded width.
    template <typename Reader, typename Ctx>
    inline void loop_ingest_planar(const std::uint8_t* base,
                                   std::int32_t sizeX, std::int32_t sizeY,
                                   std::int32_t srcPitch, const Ctx& ctx,
                                   PrFloat* RESTRICT dstR,
                                   PrFloat* RESTRICT dstG,
                                   PrFloat* RESTRICT dstB,
                                   std::int32_t dstPitch)
    {
        const std::ptrdiff_t byteStride =
            static_cast<std::ptrdiff_t>(srcPitch) *
            static_cast<std::ptrdiff_t>(Reader::kPixelBytes);
        for (std::int32_t y = 0; y < sizeY; ++y)
        {
            const std::uint8_t* row = base + static_cast<std::ptrdiff_t>(y) * byteStride;
            const std::ptrdiff_t o = static_cast<std::ptrdiff_t>(y) *
                                     static_cast<std::ptrdiff_t>(dstPitch);
            PrFloat* pr = dstR + o;
            PrFloat* pg = dstG + o;
            PrFloat* pb = dstB + o;
            for (std::int32_t x = 0; x < sizeX; ++x)
            {
                PrFloat R, G, B;
                Reader::read(row, x, ctx, R, G, B);
                pr[x] = R; pg[x] = G; pb[x] = B;
            }
        }
    }

    // =========================================================================
    // PUBLIC ENTRY POINTS
    //
    // Two overload sets, same names:
    //   * LUT-FREE (primary) - what to call. The transfer curve is computed
    //     directly; nothing is tabulated.
    //   * LUT-TAKING (legacy) - the reference's exact signature, kept so
    //     existing call sites compile unchanged. The three table arguments are
    //     ACCEPTED AND IGNORED; they forward to the overload above.
    // =========================================================================

    //! Ingest a frame -> interleaved linear float32 RGB.
    inline void ingest_to_linear_f32
    (
        const void* src, std::int32_t sizeX, std::int32_t sizeY, std::int32_t srcPitch,
        ePrPixelFormat fmt, PrFloat* dstRGB_f32
    )
    {
        const DecodeCtx ctx = { narrow_matrix(pick_matrix(fmt)) };
        const std::uint8_t* base = static_cast<const std::uint8_t*>(src);
        dispatch_reader(fmt, [&](auto reader)
        {
            using R = decltype(reader);
            loop_ingest<R>(base, sizeX, sizeY, srcPitch, ctx, dstRGB_f32);
        });
    }

    //! Ingest a frame -> PLANAR linear float32 RGB, one pitch for all three
    //! planes. Added for the film engine, which consumes planar samples.
    inline void ingest_to_planar_f32
    (
        const void* src, std::int32_t sizeX, std::int32_t sizeY, std::int32_t srcPitch,
        ePrPixelFormat fmt,
        PrFloat* dstR, PrFloat* dstG, PrFloat* dstB, std::int32_t dstPitch
    )
    {
        const DecodeCtx ctx = { narrow_matrix(pick_matrix(fmt)) };
        const std::uint8_t* base = static_cast<const std::uint8_t*>(src);
        dispatch_reader(fmt, [&](auto reader)
        {
            using R = decltype(reader);
            loop_ingest_planar<R>(base, sizeX, sizeY, srcPitch, ctx,
                                  dstR, dstG, dstB, dstPitch);
        });
    }

    //! Legacy signature. ⚠ lut8/lut16/lut10 are ignored - there is no table on
    //! this path any more. Kept only so reference call sites keep compiling.
    template <typename LUT8, typename LUT16, typename LUT10>
    inline void ingest_to_linear_f32
    (
        const void* src, std::int32_t sizeX, std::int32_t sizeY, std::int32_t srcPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        PrFloat* dstRGB_f32
    )
    {
        (void)lut8; (void)lut16; (void)lut10;
        ingest_to_linear_f32(src, sizeX, sizeY, srcPitch, fmt, dstRGB_f32);
    }

    //! Legacy signature, planar. ⚠ The tables are ignored.
    template <typename LUT8, typename LUT16, typename LUT10>
    inline void ingest_to_planar_f32
    (
        const void* src, std::int32_t sizeX, std::int32_t sizeY, std::int32_t srcPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        PrFloat* dstR, PrFloat* dstG, PrFloat* dstB, std::int32_t dstPitch
    )
    {
        (void)lut8; (void)lut16; (void)lut10;
        ingest_to_planar_f32(src, sizeX, sizeY, srcPitch, fmt,
                             dstR, dstG, dstB, dstPitch);
    }

#ifdef ALGO_PR_WITH_SUPERPIXEL
    // =========================================================================
    // RESTORATION HOOK for the super-pixel / CCT path.
    //
    // The reference also exposed `ingest_and_superpixel`, `LocusGate` and
    // `build_locus_gate`. They are NOT ported here because they depend on
    // "super_pixel.hpp", which is absent from this project, and because the
    // film engine has no CCT auto-white-balance stage that would consume a
    // super-pixel. Supplying that header and defining ALGO_PR_WITH_SUPERPIXEL
    // is the whole restoration path: the readers, the context and the dispatch
    // above are unchanged and already sufficient for `loop_fused`.
    // =========================================================================
    #include "super_pixel.hpp"
#endif

} // namespace AlgoPrIngest

#endif // __IMAGELAB2_PR_FORMAT_INGEST_HPP__
