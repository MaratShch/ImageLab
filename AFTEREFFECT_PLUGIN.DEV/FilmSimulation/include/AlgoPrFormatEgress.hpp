#ifndef __IMAGELAB2_PR_FORMAT_EGRESS_HPP__
#define __IMAGELAB2_PR_FORMAT_EGRESS_HPP__

// =============================================================================
// AlgoPrFormatEgress.hpp - scalar EGRESS: write a LINEAR float32 RGB buffer
// (interleaved OR planar, as produced by the ingest / by the film engine) back
// into an Adobe Premiere/After Effects frame buffer, in ANY of the supported
// formats. Exact structural inverse of the ingest.
//
// -----------------------------------------------------------------------------
// ALPHA
// -----------------------------------------------------------------------------
// The linear buffer carries NO alpha. The alpha - or the X pad of the ...X
// formats - of every output pixel is taken from THE SAME PIXEL of the ORIGINAL
// INCOMING Adobe buffer, which the caller passes alongside and which must be of
// the same pixel format as the destination. For the premultiplied formats
// (BGRP / VUYP / PRGB) that alpha is also re-applied to the colour, exactly
// inverting the ingest's un-premultiply.
//
// ⚠ IN-PLACE HAZARD. If the host ever hands the same memory as both `dst` and
// `alphaSrc`, the alpha of a pixel must be read BEFORE that pixel is written.
// Every writer below reads alpha first, into a local, then writes. Do not
// reorder those two statements.
//
// -----------------------------------------------------------------------------
// CLAMPING - added on owner instruction, and one case deserves a warning
// -----------------------------------------------------------------------------
// Every store into the Adobe buffer is clamped to the destination's ceiling:
//
//     8-bit   -> 0 .. 255      (kMaxCode8)   written as unsigned char
//     10-bit  -> 0 .. 1023     (kMaxCode10)
//     16-bit  -> 0 .. 32767    (kMaxCode16)  ⚠ NOT 65535 - see the ingest note
//     float32 -> 0.0 .. 1.0    (kMaxFloat32)
//
// All integer quantization routes through the single `quantize_code()` helper in
// the ingest header, so no path can pick up a different ceiling by accident, and
// the VUYA chroma is clamped AFTER the +128 bias is applied and only then cast
// to unsigned char - never a signed-to-unsigned wrap.
//
// ⚠ THE FLOAT CLAMP IS THE ONE THAT CAN COST REAL DATA, and it is applied to the
// _Linear formats too. Those samples are scene-linear: speculars, light sources
// and the film model's own highlight rolloff legitimately exceed 1.0, and the
// reference implementation left them unclamped for exactly that reason
// ("identity, UNCLAMPED (HDR ok)"). Clamping them to 1.0 discards that
// headroom irreversibly. It is done because it was asked for, it is confined to
// `kClampLinearF32`, and setting that flag false restores HDR passthrough
// without touching anything else. If the composite downstream is 32f, consider
// setting it false.
//
// -----------------------------------------------------------------------------
// ENCODING - DIRECT COMPUTATION, NO TABLE (owner instruction)
// -----------------------------------------------------------------------------
// Linear -> display code is computed by transfer_encode() in
// AlgoPrFormatMath.hpp: the analytic inverse of the curve the ingest applied,
// evaluated with the fast_log2/fast_exp2 pair. Three consequences worth naming:
//
//   * THE ~15-STEP PER-PIXEL BINARY SEARCH IS GONE. The reference inverted a
//     32768-entry table by bracketing it; that was the single most expensive
//     operation on the egress path and it has no replacement cost - the direct
//     encode is ~15 FMA with no memory traffic and no data-dependent branch.
//   * SCALAR AND AVX2 NOW SHARE ONE DEFINITION of the curve, so they agree to
//     within a float ULP instead of being two independent approximations.
//   * The layer is no longer transfer-agnostic: the curve is chosen at compile
//     time by kTransfer (sRGB by default, Rec.709 and pure gamma available).
//     That is the trade the no-LUT instruction implies, stated plainly.
//
// Integer round trip is still exact: measured max error of the encode is 0.021
// of a 16-bit code and 0.0002 of an 8-bit code, both far inside the 0.5
// rounding boundary, and code -> linear -> code is verified exact for all 256
// 8-bit codes, all 1024 10-bit codes and all 32768 16-bit codes.
//
// _Linear formats bypass the curve entirely: the Adobe engine already
// linearized them, so egress is the identity (then the clamp above).
//
// YCbCr: the forward R'G'B' -> Y'CbCr matrix is DERIVED from the same
// reconstruction constants the ingest uses (Kr = 1 - aR/2, Kb = 1 - aB/2,
// Kg = 1 - Kr - Kb; Cr = (R'-Y')/aR, Cb = (B'-Y')/aB), so it is the exact
// inverse of the ingest, and the 709-versus-601 selection is inherited from
// pick_matrix(). VUYA 8u is studio range, 32f full range with chroma at +0.5 -
// mirroring the ingest conventions exactly.
//
// SIZES AND PITCHES ARE IN PIXELS and SIGNED; negative pitch (bottom-up) is
// supported on the destination AND on the alpha source independently.
//
// C++14. No dynamic allocation. No STL containers. Scalar; the AVX2 fast paths
// are in AlgoEgressAVX2.hpp and are strictly additive.
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <cmath>
#include "AlgoPrFormatIngest.hpp"   // formats, structs, matrices, DecodeCtx,
                                    // clamps, quantize_code, pr_round_half_away
                                    // and (via it) AlgoPrFormatMath.hpp

namespace AlgoPrIngest
{
    //! ⚠ Set false to restore unclamped HDR passthrough on the _Linear formats.
    //! See the CLAMPING note in this file's header before changing it.
    constexpr bool kClampLinearF32 = true;

    // =========================================================================
    // DIRECT ENCODE. The reference had three functions here - lut_upper_index(),
    // code_from_linear() and encoded_from_linear() - implementing a bracketing
    // binary search over the decode table. All three are replaced by
    // transfer_encode() plus a quantize, and the table is gone.
    //
    // The names are kept so reference call sites read the same, and the
    // signatures still ACCEPT a LUT argument that is ignored, for the same
    // reason. `PrSetup` (double) parameters are retained on the interface but no
    // longer serve a search - the computation runs in PrFloat.
    // =========================================================================

    //! linear -> NEAREST integer code, clamped to [0, maxCode].
    inline int code_from_linear(PrFloat lin, int maxCode)
    {
        return quantize_code(transfer_encode(lin), maxCode);
    }

    //! linear -> CONTINUOUS encoded value in [0,1].
    inline PrFloat encoded_from_linear(PrFloat lin)
    {
        return clamp_unit(transfer_encode(lin));
    }

    //! Legacy shapes: the table argument is accepted and ignored.
    template <typename LUT>
    inline int code_from_linear(PrFloat lin, const LUT& lut, int maxCode)
    {
        (void)lut;
        return code_from_linear(lin, maxCode);
    }

    template <typename LUT>
    inline PrFloat encoded_from_linear(PrFloat lin, const LUT& lut, int maxCode)
    {
        (void)lut; (void)maxCode;
        return encoded_from_linear(lin);
    }

    // The forward R'G'B' -> Y'CbCr matrix (struct RGBToYCbCr and
    // forward_matrix()) now lives in AlgoPrFormatIngest.hpp, beside the reverse
    // matrix it is derived from. It moved because AlgoPrFormatAVX2.hpp needs it
    // to build the broadcast form, and that header must not have to include the
    // egress to see a type the ingest owns half of.

    //! Clamp then narrow to an unsigned 8-bit code. The cast happens AFTER the
    //! clamp, so a negative chroma can never wrap to a large unsigned value.
    inline A_u_char to_u8(PrFloat v)
    {
        return static_cast<A_u_char>(clamp_index(pr_round_half_away(v), kMaxCode8));
    }

    //! Same for the 0..32767 16-bit convention.
    inline A_u_short to_u16(PrFloat v)
    {
        return static_cast<A_u_short>(clamp_index(pr_round_half_away(v), kMaxCode16));
    }

    // =========================================================================
    // PER-FORMAT WRITERS. Each packs one pixel at (dstRow, x) from linear float
    // R,G,B, taking alpha from the SAME-FORMAT incoming pixel at (alphaRow, ax).
    // Mirrors of the ingest readers; no runtime format branch.
    // =========================================================================

    // ---- integer 8-bit ----------------------------------------------------
    template <typename Pix, bool Premul>
    struct WriteInt8
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, std::int32_t x,
                                 const std::uint8_t* alphaRow, std::int32_t ax,
                                 const Ctx& /*c*/, PrFloat R, PrFloat G, PrFloat B)
        {
            // alpha FIRST - see the in-place hazard note in the file header
            const int a = alphaRow
                ? static_cast<int>((reinterpret_cast<const Pix*>(alphaRow) + ax)->A)
                : kMaxCode8;
            Pix* p = reinterpret_cast<Pix*>(dstRow) + x;
            if (Premul)
            {   // ⚠ PREMULTIPLY IN THE CONTINUOUS DOMAIN, QUANTIZE ONCE.
                // The reference quantized to an integer code first and then
                // multiplied by alpha and re-rounded - two quantizations, and
                // the second one on an already-truncated value. Encoding to a
                // continuous [0,1], scaling, and quantizing once is strictly
                // more accurate and costs nothing.
                const PrFloat af = static_cast<PrFloat>(a) * (1.0f / 255.0f);
                p->R = to_u8(encoded_from_linear(R) * af
                             * static_cast<PrFloat>(kMaxCode8));
                p->G = to_u8(encoded_from_linear(G) * af
                             * static_cast<PrFloat>(kMaxCode8));
                p->B = to_u8(encoded_from_linear(B) * af
                             * static_cast<PrFloat>(kMaxCode8));
            }
            else
            {   // nearest-code inverse -> bit-exact round trip
                p->R = static_cast<A_u_char>(code_from_linear(R, kMaxCode8));
                p->G = static_cast<A_u_char>(code_from_linear(G, kMaxCode8));
                p->B = static_cast<A_u_char>(code_from_linear(B, kMaxCode8));
            }
            p->A = static_cast<A_u_char>(a);   // alpha / X passthrough
        }
    };

    // ---- integer 16-bit, 0..32767 -----------------------------------------
    template <typename Pix, bool Premul>
    struct WriteInt16
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, std::int32_t x,
                                 const std::uint8_t* alphaRow, std::int32_t ax,
                                 const Ctx& /*c*/, PrFloat R, PrFloat G, PrFloat B)
        {
            const int a = alphaRow
                ? static_cast<int>((reinterpret_cast<const Pix*>(alphaRow) + ax)->A)
                : kMaxCode16;
            Pix* p = reinterpret_cast<Pix*>(dstRow) + x;
            if (Premul)
            {
                const PrFloat af = static_cast<PrFloat>(a) * (1.0f / 32767.0f);
                p->R = to_u16(encoded_from_linear(R) * af
                              * static_cast<PrFloat>(kMaxCode16));
                p->G = to_u16(encoded_from_linear(G) * af
                              * static_cast<PrFloat>(kMaxCode16));
                p->B = to_u16(encoded_from_linear(B) * af
                              * static_cast<PrFloat>(kMaxCode16));
            }
            else
            {
                p->R = static_cast<A_u_short>(code_from_linear(R, kMaxCode16));
                p->G = static_cast<A_u_short>(code_from_linear(G, kMaxCode16));
                p->B = static_cast<A_u_short>(code_from_linear(B, kMaxCode16));
            }
            p->A = static_cast<A_u_short>(a);
        }
    };

    // ---- 32f, gamma-encoded or already-linear, optional premultiplied ------
    template <typename Pix, bool Premul, bool Linear>
    struct WriteF32
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, std::int32_t x,
                                 const std::uint8_t* alphaRow, std::int32_t ax,
                                 const Ctx& /*c*/, PrFloat R, PrFloat G, PrFloat B)
        {
            const PrFloat a = alphaRow
                ? (reinterpret_cast<const Pix*>(alphaRow) + ax)->A
                : 1.0f;
            Pix* p = reinterpret_cast<Pix*>(dstRow) + x;
            PrFloat r, g, b;
            if (Linear)
            {   // Adobe already linearized this format; identity back out.
                r = R; g = G; b = B;
            }
            else
            {
                r = encoded_from_linear(R);
                g = encoded_from_linear(G);
                b = encoded_from_linear(B);
            }
            if (Premul) { r *= a; g *= a; b *= a; }
            // ⚠ The clamp. For Linear this is the HDR-destroying one; see the
            // file header. kClampLinearF32 is a compile-time constant so the
            // branch folds away entirely in either configuration.
            if (!Linear || kClampLinearF32)
            {
                r = clamp_unit(r); g = clamp_unit(g); b = clamp_unit(b);
            }
            p->R = r; p->G = g; p->B = b;
            p->A = a;
        }
    };

    // ---- VUYA / VUYP / VUYX 8u, studio range ------------------------------
    template <bool Premul>
    struct WriteVUYA8
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_8u);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, std::int32_t x,
                                 const std::uint8_t* alphaRow, std::int32_t ax,
                                 const Ctx& c, PrFloat R, PrFloat G, PrFloat B)
        {
            const int a = alphaRow
                ? static_cast<int>((reinterpret_cast<const PF_Pixel_VUYA_8u*>(alphaRow) + ax)->A)
                : kMaxCode8;
            PF_Pixel_VUYA_8u* p = reinterpret_cast<PF_Pixel_VUYA_8u*>(dstRow) + x;
            PrFloat Rp = encoded_from_linear(R);
            PrFloat Gp = encoded_from_linear(G);
            PrFloat Bp = encoded_from_linear(B);
            if (Premul)
            {   // premultiply R'G'B' BEFORE the matrix - the inverse of the
                // ingest, which divided AFTER it
                const PrFloat af = static_cast<PrFloat>(a) * (1.0f / 255.0f);
                Rp *= af; Gp *= af; Bp *= af;
            }
            const RGBToYCbCr M = forward_matrix(c.C);
            const PrFloat Yp = M.Kr * Rp + M.Kg * Gp + M.Kb * Bp;
            const PrFloat Cr = (Rp - Yp) / M.aR;
            const PrFloat Cb = (Bp - Yp) / M.aB;
            // ⚠ UNSIGNED RESTORATION. The +128 bias is applied first, the result
            // is clamped to 0..255, and ONLY THEN cast to unsigned char. A
            // signed chroma cast directly to A_u_char would wrap.
            p->Y = to_u8(kVuya8LumaOffset  + Yp * kVuya8LumaScale);
            p->U = to_u8(kVuya8ChromaBias  + Cb * kVuya8ChromaScale);
            p->V = to_u8(kVuya8ChromaBias  + Cr * kVuya8ChromaScale);
            p->A = static_cast<A_u_char>(a);
        }
    };

    // ---- VUYA / VUYP / VUYX 32f, full range, chroma stored at +0.5 --------
    template <bool Premul>
    struct WriteVUYA32
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_32f);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, std::int32_t x,
                                 const std::uint8_t* alphaRow, std::int32_t ax,
                                 const Ctx& c, PrFloat R, PrFloat G, PrFloat B)
        {
            const PrFloat a = alphaRow
                ? (reinterpret_cast<const PF_Pixel_VUYA_32f*>(alphaRow) + ax)->A
                : 1.0f;
            PF_Pixel_VUYA_32f* p = reinterpret_cast<PF_Pixel_VUYA_32f*>(dstRow) + x;
            PrFloat Rp = encoded_from_linear(R);
            PrFloat Gp = encoded_from_linear(G);
            PrFloat Bp = encoded_from_linear(B);
            if (Premul) { Rp *= a; Gp *= a; Bp *= a; }
            const RGBToYCbCr M = forward_matrix(c.C);
            const PrFloat Yp = M.Kr * Rp + M.Kg * Gp + M.Kb * Bp;
            const PrFloat Cr = (Rp - Yp) / M.aR;
            const PrFloat Cb = (Bp - Yp) / M.aB;
            // Y is clamped to [0,1]; chroma is clamped to [0,1] AFTER the +0.5
            // bias, which is the valid encoded range for this format.
            p->Y = clamp_unit(Yp);
            p->U = clamp_unit(Cb + kVuyaF32ChromaOffset);
            p->V = clamp_unit(Cr + kVuyaF32ChromaOffset);
            p->A = a;
        }
    };

    // ---- RGB 444 10u: NO ALPHA, packed bitfields --------------------------
    struct WriteRGB10
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_RGB_10u);
        template <typename Ctx>
        static inline void write(std::uint8_t* dstRow, std::int32_t x,
                                 const std::uint8_t* /*alphaRow*/, std::int32_t /*ax*/,
                                 const Ctx& /*c*/, PrFloat R, PrFloat G, PrFloat B)
        {
            PF_Pixel_RGB_10u* p = reinterpret_cast<PF_Pixel_RGB_10u*>(dstRow) + x;
            // ⚠ THE PAD BITS ARE ZEROED, not left as found. The reference wrote
            // only the three bitfields, so bits 0-1 kept whatever the
            // destination memory happened to hold - which for a fresh or
            // recycled host buffer is indeterminate, and makes the output
            // non-deterministic for identical input. Writing the whole word
            // costs one store either way.
            p->_pad_ = 0u;
            p->R = static_cast<A_u_long>(code_from_linear(R, kMaxCode10));
            p->G = static_cast<A_u_long>(code_from_linear(G, kMaxCode10));
            p->B = static_cast<A_u_long>(code_from_linear(B, kMaxCode10));
            // No alpha exists in this format: nothing is preserved, nothing is
            // synthesized. alphaRow/ax are intentionally unused.
        }
    };

    // =========================================================================
    // dispatch_writer - the ONLY format switch, a 1:1 mirror of dispatch_reader.
    // =========================================================================
    template <typename F>
    inline void dispatch_writer(ePrPixelFormat fmt, F&& f)
    {
        switch (fmt)
        {
            case fmt_BGRA_4444_8u:
            case fmt_BGRX_4444_8u:         f(WriteInt8 <PF_Pixel_BGRA_8u , false>()); break;
            case fmt_BGRP_4444_8u:         f(WriteInt8 <PF_Pixel_BGRA_8u , true >()); break;
            case fmt_ARGB_4444_8u:
            case fmt_XRGB_4444_8u:         f(WriteInt8 <PF_Pixel_ARGB_8u , false>()); break;
            case fmt_PRGB_4444_8u:         f(WriteInt8 <PF_Pixel_ARGB_8u , true >()); break;

            case fmt_BGRA_4444_16u:
            case fmt_BGRX_4444_16u:        f(WriteInt16<PF_Pixel_BGRA_16u, false>()); break;
            case fmt_BGRP_4444_16u:        f(WriteInt16<PF_Pixel_BGRA_16u, true >()); break;
            case fmt_ARGB_4444_16u:
            case fmt_XRGB_4444_16u:        f(WriteInt16<PF_Pixel_ARGB_16u, false>()); break;
            case fmt_PRGB_4444_16u:        f(WriteInt16<PF_Pixel_ARGB_16u, true >()); break;

            case fmt_BGRA_4444_32f:
            case fmt_BGRX_4444_32f:        f(WriteF32<PF_Pixel_BGRA_32f, false, false>()); break;
            case fmt_BGRP_4444_32f:        f(WriteF32<PF_Pixel_BGRA_32f, true , false>()); break;
            case fmt_BGRA_4444_32f_Linear:
            case fmt_BGRX_4444_32f_Linear: f(WriteF32<PF_Pixel_BGRA_32f, false, true >()); break;
            case fmt_BGRP_4444_32f_Linear: f(WriteF32<PF_Pixel_BGRA_32f, true , true >()); break;
            case fmt_ARGB_4444_32f:
            case fmt_XRGB_4444_32f:        f(WriteF32<PF_Pixel_ARGB_32f, false, false>()); break;
            case fmt_ARGB_4444_32f_Linear:
            case fmt_XRGB_4444_32f_Linear: f(WriteF32<PF_Pixel_ARGB_32f, false, true >()); break;
            case fmt_PRGB_4444_32f:        f(WriteF32<PF_Pixel_ARGB_32f, true , false>()); break;
            case fmt_PRGB_4444_32f_Linear: f(WriteF32<PF_Pixel_ARGB_32f, true , true >()); break;

            case fmt_VUYA_4444_8u_709:
            case fmt_VUYA_4444_8u:         f(WriteVUYA8 <false>()); break;
            case fmt_VUYP_4444_8u_709:
            case fmt_VUYP_4444_8u:         f(WriteVUYA8 <true >()); break;
            case fmt_VUYA_4444_32f_709:
            case fmt_VUYA_4444_32f:        f(WriteVUYA32<false>()); break;
            case fmt_VUYP_4444_32f_709:
            case fmt_VUYP_4444_32f:        f(WriteVUYA32<true >()); break;
            case fmt_VUYX_4444_8u_709:
            case fmt_VUYX_4444_8u:         f(WriteVUYA8 <false>()); break;
            case fmt_VUYX_4444_32f_709:
            case fmt_VUYX_4444_32f:        f(WriteVUYA32<false>()); break;

            case fmt_RGB_444_10u:          f(WriteRGB10()); break;
            default: break;
        }
    }

    // =========================================================================
    // TRAVERSALS. Both destination and alpha source carry independent SIGNED
    // pitches; either may be negative.
    // =========================================================================

    //! Interleaved source, tightly packed sizeX*sizeY*3. Reference layout.
    template <typename Writer, typename Ctx>
    inline void loop_egress(const PrFloat* srcRGB_f32,
                            std::int32_t sizeX, std::int32_t sizeY,
                            std::uint8_t* dstBase, std::int32_t dstPitch,
                            const std::uint8_t* alphaBase, std::int32_t alphaSizeX,
                            std::int32_t alphaPitch, const Ctx& ctx)
    {
        const std::ptrdiff_t dstStride =
            static_cast<std::ptrdiff_t>(dstPitch) *
            static_cast<std::ptrdiff_t>(Writer::kPixelBytes);
        const std::ptrdiff_t alpStride =
            static_cast<std::ptrdiff_t>(alphaPitch) *
            static_cast<std::ptrdiff_t>(Writer::kPixelBytes);
        const std::int32_t axMax = (alphaSizeX > 0) ? (alphaSizeX - 1) : 0;

        for (std::int32_t y = 0; y < sizeY; ++y)
        {
            const PrFloat* srcRow = srcRGB_f32 +
                static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(sizeX) * 3;
            std::uint8_t* dstRow = dstBase + static_cast<std::ptrdiff_t>(y) * dstStride;
            const std::uint8_t* alpRow = alphaBase
                ? alphaBase + static_cast<std::ptrdiff_t>(y) * alpStride : nullptr;
            for (std::int32_t x = 0; x < sizeX; ++x)
            {
                const PrFloat* s = srcRow + static_cast<std::ptrdiff_t>(x) * 3;
                const std::int32_t ax = (x < axMax) ? x : axMax;
                Writer::write(dstRow, x, alpRow, ax, ctx, s[0], s[1], s[2]);
            }
        }
    }

    //! Planar source - what the film engine produces.
    template <typename Writer, typename Ctx>
    inline void loop_egress_planar(const PrFloat* RESTRICT srcR,
                                   const PrFloat* RESTRICT srcG,
                                   const PrFloat* RESTRICT srcB,
                                   std::int32_t srcPitch,
                                   std::int32_t sizeX, std::int32_t sizeY,
                                   std::uint8_t* dstBase, std::int32_t dstPitch,
                                   const std::uint8_t* alphaBase, std::int32_t alphaSizeX,
                                   std::int32_t alphaPitch, const Ctx& ctx)
    {
        const std::ptrdiff_t dstStride =
            static_cast<std::ptrdiff_t>(dstPitch) *
            static_cast<std::ptrdiff_t>(Writer::kPixelBytes);
        const std::ptrdiff_t alpStride =
            static_cast<std::ptrdiff_t>(alphaPitch) *
            static_cast<std::ptrdiff_t>(Writer::kPixelBytes);
        const std::int32_t axMax = (alphaSizeX > 0) ? (alphaSizeX - 1) : 0;

        for (std::int32_t y = 0; y < sizeY; ++y)
        {
            const std::ptrdiff_t o = static_cast<std::ptrdiff_t>(y) *
                                     static_cast<std::ptrdiff_t>(srcPitch);
            const PrFloat* pr = srcR + o;
            const PrFloat* pg = srcG + o;
            const PrFloat* pb = srcB + o;
            std::uint8_t* dstRow = dstBase + static_cast<std::ptrdiff_t>(y) * dstStride;
            const std::uint8_t* alpRow = alphaBase
                ? alphaBase + static_cast<std::ptrdiff_t>(y) * alpStride : nullptr;
            for (std::int32_t x = 0; x < sizeX; ++x)
            {
                const std::int32_t ax = (x < axMax) ? x : axMax;
                Writer::write(dstRow, x, alpRow, ax, ctx, pr[x], pg[x], pb[x]);
            }
        }
    }

    // =========================================================================
    // PUBLIC ENTRY POINTS
    //
    //   sizeX, sizeY : frame size in PIXELS (linear buffer geometry).
    //   dstPitch     : destination pitch in PIXELS, negative = bottom-up.
    //   alphaSrc     : the ORIGINAL incoming Adobe buffer, same pixel format as
    //                  the destination. May be nullptr -> full opacity is
    //                  written (255 / 32767 / 1.0f).
    //   alphaSizeX   : incoming buffer width in PIXELS; x is clamped into it.
    //   alphaPitch   : incoming buffer pitch in PIXELS, negative allowed.
    //
    // fmt_RGB_444_10u has no alpha - the three alpha parameters are ignored for
    // it, and nothing is synthesized.
    //
    // Two overload sets, same names: LUT-free (primary), and the reference's
    // LUT-taking signature kept for source compatibility, whose three table
    // arguments are ACCEPTED AND IGNORED.
    // =========================================================================

    //! Interleaved source.
    inline void egress_from_linear_f32
    (
        const PrFloat* srcRGB_f32,
        std::int32_t sizeX, std::int32_t sizeY,
        void* dst, std::int32_t dstPitch,
        ePrPixelFormat fmt,
        const void* alphaSrc, std::int32_t alphaSizeX, std::int32_t alphaPitch
    )
    {
        const DecodeCtx ctx = { narrow_matrix(pick_matrix(fmt)) };
        std::uint8_t* dstBase = static_cast<std::uint8_t*>(dst);
        const std::uint8_t* alpBase = static_cast<const std::uint8_t*>(alphaSrc);
        dispatch_writer(fmt, [&](auto writer)
        {
            using W = decltype(writer);
            loop_egress<W>(srcRGB_f32, sizeX, sizeY, dstBase, dstPitch,
                           alpBase, alphaSizeX, alphaPitch, ctx);
        });
    }

    //! Planar source, one pitch for all three planes - the film engine's layout.
    inline void egress_from_planar_f32
    (
        const PrFloat* srcR, const PrFloat* srcG, const PrFloat* srcB,
        std::int32_t srcPitch,
        std::int32_t sizeX, std::int32_t sizeY,
        void* dst, std::int32_t dstPitch,
        ePrPixelFormat fmt,
        const void* alphaSrc, std::int32_t alphaSizeX, std::int32_t alphaPitch
    )
    {
        const DecodeCtx ctx = { narrow_matrix(pick_matrix(fmt)) };
        std::uint8_t* dstBase = static_cast<std::uint8_t*>(dst);
        const std::uint8_t* alpBase = static_cast<const std::uint8_t*>(alphaSrc);
        dispatch_writer(fmt, [&](auto writer)
        {
            using W = decltype(writer);
            loop_egress_planar<W>(srcR, srcG, srcB, srcPitch, sizeX, sizeY,
                                  dstBase, dstPitch,
                                  alpBase, alphaSizeX, alphaPitch, ctx);
        });
    }

    //! Legacy signature. ⚠ lut8/lut16/lut10 are ignored - the encode is direct.
    template <typename LUT8, typename LUT16, typename LUT10>
    inline void egress_from_linear_f32
    (
        const PrFloat* srcRGB_f32,
        std::int32_t sizeX, std::int32_t sizeY,
        void* dst, std::int32_t dstPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        const void* alphaSrc, std::int32_t alphaSizeX, std::int32_t alphaPitch
    )
    {
        (void)lut8; (void)lut16; (void)lut10;
        egress_from_linear_f32(srcRGB_f32, sizeX, sizeY, dst, dstPitch, fmt,
                               alphaSrc, alphaSizeX, alphaPitch);
    }

    //! Legacy signature, planar. ⚠ The tables are ignored.
    template <typename LUT8, typename LUT16, typename LUT10>
    inline void egress_from_planar_f32
    (
        const PrFloat* srcR, const PrFloat* srcG, const PrFloat* srcB,
        std::int32_t srcPitch,
        std::int32_t sizeX, std::int32_t sizeY,
        void* dst, std::int32_t dstPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        const void* alphaSrc, std::int32_t alphaSizeX, std::int32_t alphaPitch
    )
    {
        (void)lut8; (void)lut16; (void)lut10;
        egress_from_planar_f32(srcR, srcG, srcB, srcPitch, sizeX, sizeY,
                               dst, dstPitch, fmt,
                               alphaSrc, alphaSizeX, alphaPitch);
    }

} // namespace AlgoPrIngest

#endif // __IMAGELAB2_PR_FORMAT_EGRESS_HPP__
