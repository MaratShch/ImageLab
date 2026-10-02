#ifndef __IMAGELAB2_INGEST_AVX2_HPP__
#define __IMAGELAB2_INGEST_AVX2_HPP__

// =============================================================================
// AlgoIngestAVX2.hpp - AVX2 + FMA ingest of Adobe Premiere/After Effects frame
// buffers into a LINEAR float32 RGB buffer, interleaved OR planar.
//
// The scalar path in AlgoPrFormatIngest.hpp stays the reference and the
// fallback; this header is additive, and every entry point here has a scalar
// twin there with the same name plus the `_avx2` suffix removed.
//
// -----------------------------------------------------------------------------
// WHAT CHANGED FROM THE REFERENCE AlgoIngestAVX2.hpp, AND WHY
// -----------------------------------------------------------------------------
//  1. THE REFERENCE COVERED ONE FORMAT FAMILY. It vectorised only 32f encoded
//     BGRA/ARGB (`loop_ingest_f32_srgb`) and explicitly left 8u/10u scalar on
//     the grounds that a LUT gather loses to a scalar load. That finding is
//     correct AND avoidable: computing the curve instead of tabulating it
//     removes the gather entirely, after which the integer formats
//     de-interleave with a shift and a mask - one dword per pixel, element k
//     still pixel k. So all six reader families are vectorised here.
//     `loop_ingest_f32_srgb` is KEPT with its exact reference signature.
//
//  2. THE REFERENCE'S "SIMD" LOOP DE-INTERLEAVED SCALARLY. Its inner body was
//     `for (int k = 0; k < 8; ++k) { R[k]=row[x+k].R; ... }` into a stack array,
//     then a vector decode, then another scalar loop to scatter the results.
//     Only the transcendental was vectorised. Here the AoS<->SoA transposes are
//     real SIMD (see load8_aos4_ps / store8_aos3_ps in AlgoPrFormatAVX2.hpp).
//
//  3. TAILS USE THE SAME CURVE AS THE BODY. The reference did this correctly
//     and it is worth restating: a vector body on one approximation and a
//     scalar tail on another makes the last <8 pixels of every row disagree
//     with their neighbours by a fraction of a code - invisible in a preview,
//     obvious in a difference blend. Since the LUTs were dropped there is only
//     one curve in the layer, so this now holds by construction.
//
// -----------------------------------------------------------------------------
// TRANSFER FUNCTION - NOW IDENTICAL TO THE SCALAR PATH
// -----------------------------------------------------------------------------
// Both paths call ONE definition of the curve (AlgoPrFormatMath.hpp:
// transfer_decode / transfer_decode_ps, same constants, same polynomial pair),
// so there is no longer a scalar-versus-vector semantic difference to manage -
// verified: scalar and AVX2 ingest produce identical floats, and scalar and
// AVX2 egress produce identical bytes. That parity is why lin_from_code()
// multiplies by a reciprocal instead of dividing; see the note there.
//
// The LUT parameters on the legacy entry points are accepted and ignored, in
// both paths, because there is no table left anywhere in this layer.
//
// -----------------------------------------------------------------------------
// GEOMETRY
// -----------------------------------------------------------------------------
//  * sizeX/sizeY/pitches are SIGNED and in PIXELS. Negative pitch (bottom-up
//    frame) is supported; row bases are computed as ptrdiff_t. No unsigned
//    arithmetic on the addressing path.
//  * The vector body runs while 8 whole pixels remain; the SCALAR TAIL handles
//    sizeX % 8. Mandatory, not optional - no frame width is assumed to be a
//    multiple of the vector width.
//  * Every host-buffer access is UNALIGNED (loadu/storeu). Ae/Pr give no
//    alignment guarantee, and on Sandy Bridge and later the unaligned form
//    costs nothing unless it straddles a cache line.
//
// C++14. No dynamic allocation, no STL containers, no `new`, no `std::vector`.
// AVX2 + FMA only - no AVX-512, including the 256-bit AVX-512VL intrinsics.
// Build: GCC/Clang -mavx2 -mfma ; MSVC /arch:AVX2 (FMA implied).
// =============================================================================

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>       // std::memcpy - the bitfield layout self-check only
#include <cmath>
#include "Common.hpp"
#include "AlgoPrFormatIngest.hpp"
#include "AlgoPrFormatAVX2.hpp"

namespace AlgoPrIngest
{
namespace avx2
{
    constexpr std::int32_t kVecWidth = 8;   //!< pixels per AVX2 iteration

    // =========================================================================
    // Per-frame context for the vector paths: the YCbCr matrix in BOTH forms -
    // broadcast for the vector body, narrowed scalar for the tail - so the two
    // cannot pick up different constants. No LUT reference: these paths are
    // analytic.
    // =========================================================================
    struct DecodeCtxV
    {
        YCbCrToRGBf C;    //!< scalar, for the tail
        YCbCrVec    Cv;   //!< broadcast, for the body
    };

    inline DecodeCtxV make_ctx(ePrPixelFormat fmt)
    {
        DecodeCtxV c;
        c.C  = narrow_matrix(pick_matrix(fmt));
        c.Cv = broadcast_matrix(c.C);
        return c;
    }

    // =========================================================================
    // VECTOR READER TAGS. One per format family, mirroring the scalar reader
    // tags one for one. Each provides:
    //     kPixelBytes            - source stride unit, same as the scalar tag
    //     read8(row, x, ctx, ...) - 8 pixels -> three linear float vectors
    //     read1(row, x, ctx, ...) - 1 pixel  -> three linear floats (the tail)
    // Channel order is a template parameter, not a runtime branch: the shift
    // amounts / lane indices below come straight from CommonPixFormat.hpp.
    // =========================================================================

    // ---- integer 8-bit -----------------------------------------------------
    // Shifts are BIT offsets within the pixel's dword, exactly as documented in
    // CommonPixFormat.hpp: BGRA_8u is B@0 G@8 R@16 A@24, ARGB_8u is A@0 R@8
    // G@16 B@24.
    template <typename Pix, int SR, int SG, int SB, int SA, bool Premul>
    struct VReadInt8
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        static constexpr float kInv = 1.0f / static_cast<float>(kMaxCode8);

        static inline void read8(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& /*c*/,
                                 __m256& R, __m256& G, __m256& B)
        {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                row + static_cast<std::ptrdiff_t>(x) * 4));
            const __m256 s = _mm256_set1_ps(kInv);
            __m256 r = _mm256_mul_ps(byte_lane_to_ps<SR>(v), s);
            __m256 g = _mm256_mul_ps(byte_lane_to_ps<SG>(v), s);
            __m256 b = _mm256_mul_ps(byte_lane_to_ps<SB>(v), s);
            if (Premul)
            {   // un-premultiply in the ENCODED domain, before the decode -
                // same order as the scalar reader
                const __m256 a = _mm256_mul_ps(byte_lane_to_ps<SA>(v), s);
                r = unpremul_ps(r, a); g = unpremul_ps(g, a); b = unpremul_ps(b, a);
            }
            R = transfer_decode_ps(clamp_unit_ps(r));
            G = transfer_decode_ps(clamp_unit_ps(g));
            B = transfer_decode_ps(clamp_unit_ps(b));
        }

        static inline void read1(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& /*c*/,
                                 float& R, float& G, float& B)
        {
            const Pix* p = reinterpret_cast<const Pix*>(row) + x;
            float r = static_cast<float>(p->R) * kInv;
            float g = static_cast<float>(p->G) * kInv;
            float b = static_cast<float>(p->B) * kInv;
            if (Premul && p->A != 0)
            {
                const float inv = static_cast<float>(kMaxCode8) / static_cast<float>(p->A);
                r *= inv; g *= inv; b *= inv;
            }
            R = transfer_decode(clamp_unit(r));
            G = transfer_decode(clamp_unit(g));
            B = transfer_decode(clamp_unit(b));
        }
    };

    // ---- integer 16-bit, 0..32767 ------------------------------------------
    // One pixel is TWO dwords, so a channel is the low or the high half of the
    // even or the odd dword. The template indices select among the four:
    //   0 = even.lo   1 = even.hi   2 = odd.lo   3 = odd.hi
    // BGRA_16u -> B=0 G=1 R=2 A=3 ; ARGB_16u -> A=0 R=1 G=2 B=3.
    template <typename Pix, int IR, int IG, int IB, int IA, bool Premul>
    struct VReadInt16
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        static constexpr float kInv = 1.0f / static_cast<float>(kMaxCode16);

        static inline void read8(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& /*c*/,
                                 __m256& R, __m256& G, __m256& B)
        {
            const std::uint8_t* q = row + static_cast<std::ptrdiff_t>(x) * 8;
            const __m256i m0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            const __m256i m1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
            const __m256i ev = even_dwords(m0, m1);
            const __m256i od = odd_dwords (m0, m1);
            const __m256 c0 = half_word_to_ps<false>(ev);
            const __m256 c1 = half_word_to_ps<true >(ev);
            const __m256 c2 = half_word_to_ps<false>(od);
            const __m256 c3 = half_word_to_ps<true >(od);
            const __m256 s = _mm256_set1_ps(kInv);
            __m256 r = _mm256_mul_ps(sel4_ps<IR>(c0, c1, c2, c3), s);
            __m256 g = _mm256_mul_ps(sel4_ps<IG>(c0, c1, c2, c3), s);
            __m256 b = _mm256_mul_ps(sel4_ps<IB>(c0, c1, c2, c3), s);
            if (Premul)
            {
                const __m256 a = _mm256_mul_ps(sel4_ps<IA>(c0, c1, c2, c3), s);
                r = unpremul_ps(r, a); g = unpremul_ps(g, a); b = unpremul_ps(b, a);
            }
            R = transfer_decode_ps(clamp_unit_ps(r));
            G = transfer_decode_ps(clamp_unit_ps(g));
            B = transfer_decode_ps(clamp_unit_ps(b));
        }

        static inline void read1(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& /*c*/,
                                 float& R, float& G, float& B)
        {
            const Pix* p = reinterpret_cast<const Pix*>(row) + x;
            float r = static_cast<float>(p->R) * kInv;
            float g = static_cast<float>(p->G) * kInv;
            float b = static_cast<float>(p->B) * kInv;
            if (Premul && p->A != 0)
            {
                const float inv = static_cast<float>(kMaxCode16) / static_cast<float>(p->A);
                r *= inv; g *= inv; b *= inv;
            }
            R = transfer_decode(clamp_unit(r));
            G = transfer_decode(clamp_unit(g));
            B = transfer_decode(clamp_unit(b));
        }
    };

    // ---- 32f, encoded or already-linear, optional premultiplied ------------
    // Lane indices are member positions in the struct: BGRA_32f is B,G,R,A ->
    // R=2 G=1 B=0 A=3 ; ARGB_32f is A,R,G,B -> R=1 G=2 B=3 A=0.
    template <typename Pix, int IR, int IG, int IB, int IA, bool Premul, bool Linear>
    struct VReadF32
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);

        static inline void read8(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& /*c*/,
                                 __m256& R, __m256& G, __m256& B)
        {
            __m256 c0, c1, c2, c3;
            load8_aos4_ps(reinterpret_cast<const float*>(
                row + static_cast<std::ptrdiff_t>(x) * 16), c0, c1, c2, c3);
            __m256 r = sel4_ps<IR>(c0, c1, c2, c3);
            __m256 g = sel4_ps<IG>(c0, c1, c2, c3);
            __m256 b = sel4_ps<IB>(c0, c1, c2, c3);
            if (Premul)
            {
                const __m256 a = sel4_ps<IA>(c0, c1, c2, c3);
                r = unpremul_ps(r, a); g = unpremul_ps(g, a); b = unpremul_ps(b, a);
            }
            if (Linear)
            {   // ⚠ ALREADY LINEARIZED BY THE ADOBE ENGINE. Identity, and left
                // UNCLAMPED on ingest: speculars above 1.0 are real data and the
                // characteristic curve needs them to roll off.
                R = r; G = g; B = b;
            }
            else
            {
                R = transfer_decode_ps(clamp_unit_ps(r));
                G = transfer_decode_ps(clamp_unit_ps(g));
                B = transfer_decode_ps(clamp_unit_ps(b));
            }
        }

        static inline void read1(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& /*c*/,
                                 float& R, float& G, float& B)
        {
            const Pix* p = reinterpret_cast<const Pix*>(row) + x;
            float r = p->R, g = p->G, b = p->B;
            if (Premul && p->A != 0.0f)
            {
                const float inv = 1.0f / p->A;
                r *= inv; g *= inv; b *= inv;
            }
            if (Linear) { R = r; G = g; B = b; }
            else
            {
                R = transfer_decode(clamp_unit(r));
                G = transfer_decode(clamp_unit(g));
                B = transfer_decode(clamp_unit(b));
            }
        }
    };

    // ---- VUYA / VUYP / VUYX 8u, STUDIO range -------------------------------
    // ⚠ The raw 8-bit U and V are UNSIGNED with zero at 128. Subtracting
    // kVuya8ChromaBias is what makes them signed. They are never reinterpreted
    // as signed bytes - note byte_lane_to_ps uses ZERO extension.
    template <bool Premul>
    struct VReadVUYA8
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_8u);

        static inline void read8(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& c,
                                 __m256& R, __m256& G, __m256& B)
        {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                row + static_cast<std::ptrdiff_t>(x) * 4));
            const __m256 Vv = byte_lane_to_ps< 0>(v);   // V
            const __m256 Uv = byte_lane_to_ps< 8>(v);   // U
            const __m256 Yv = byte_lane_to_ps<16>(v);   // Y
            const __m256 Yp = _mm256_mul_ps(
                _mm256_sub_ps(Yv, _mm256_set1_ps(kVuya8LumaOffset)),
                _mm256_set1_ps(1.0f / kVuya8LumaScale));
            const __m256 Cb = _mm256_mul_ps(
                _mm256_sub_ps(Uv, _mm256_set1_ps(kVuya8ChromaBias)),
                _mm256_set1_ps(1.0f / kVuya8ChromaScale));
            const __m256 Cr = _mm256_mul_ps(
                _mm256_sub_ps(Vv, _mm256_set1_ps(kVuya8ChromaBias)),
                _mm256_set1_ps(1.0f / kVuya8ChromaScale));
            __m256 r, g, b;
            ycbcr_to_rgb_ps(Yp, Cb, Cr, c.Cv, r, g, b);
            if (Premul)
            {   // divide AFTER the matrix, mirroring the scalar reader
                const __m256 a = _mm256_mul_ps(byte_lane_to_ps<24>(v),
                                               _mm256_set1_ps(1.0f / 255.0f));
                r = unpremul_ps(r, a); g = unpremul_ps(g, a); b = unpremul_ps(b, a);
            }
            R = transfer_decode_ps(clamp_unit_ps(r));
            G = transfer_decode_ps(clamp_unit_ps(g));
            B = transfer_decode_ps(clamp_unit_ps(b));
        }

        static inline void read1(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& c,
                                 float& R, float& G, float& B)
        {
            const PF_Pixel_VUYA_8u* p =
                reinterpret_cast<const PF_Pixel_VUYA_8u*>(row) + x;
            const float Yp = (static_cast<float>(p->Y) - kVuya8LumaOffset)
                           * (1.0f / kVuya8LumaScale);
            const float Cb = (static_cast<float>(p->U) - kVuya8ChromaBias)
                           * (1.0f / kVuya8ChromaScale);
            const float Cr = (static_cast<float>(p->V) - kVuya8ChromaBias)
                           * (1.0f / kVuya8ChromaScale);
            float r = Yp + c.C.aR * Cr;
            float b = Yp + c.C.aB * Cb;
            float g = Yp - c.C.gCr * Cr - c.C.gCb * Cb;
            if (Premul && p->A != 0)
            {
                const float inv = 255.0f / static_cast<float>(p->A);
                r *= inv; g *= inv; b *= inv;
            }
            R = transfer_decode(clamp_unit(r));
            G = transfer_decode(clamp_unit(g));
            B = transfer_decode(clamp_unit(b));
        }
    };

    // ---- VUYA / VUYP / VUYX 32f, FULL range, chroma stored at +0.5 ---------
    template <bool Premul>
    struct VReadVUYA32
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_32f);

        static inline void read8(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& c,
                                 __m256& R, __m256& G, __m256& B)
        {
            __m256 c0, c1, c2, c3;   // V, U, Y, A - struct order
            load8_aos4_ps(reinterpret_cast<const float*>(
                row + static_cast<std::ptrdiff_t>(x) * 16), c0, c1, c2, c3);
            const __m256 off = _mm256_set1_ps(kVuyaF32ChromaOffset);
            const __m256 Cr = _mm256_sub_ps(c0, off);
            const __m256 Cb = _mm256_sub_ps(c1, off);
            __m256 r, g, b;
            ycbcr_to_rgb_ps(c2, Cb, Cr, c.Cv, r, g, b);
            if (Premul)
            {
                r = unpremul_ps(r, c3); g = unpremul_ps(g, c3); b = unpremul_ps(b, c3);
            }
            R = transfer_decode_ps(clamp_unit_ps(r));
            G = transfer_decode_ps(clamp_unit_ps(g));
            B = transfer_decode_ps(clamp_unit_ps(b));
        }

        static inline void read1(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& c,
                                 float& R, float& G, float& B)
        {
            const PF_Pixel_VUYA_32f* p =
                reinterpret_cast<const PF_Pixel_VUYA_32f*>(row) + x;
            const float Cb = p->U - kVuyaF32ChromaOffset;
            const float Cr = p->V - kVuyaF32ChromaOffset;
            float r = p->Y + c.C.aR * Cr;
            float b = p->Y + c.C.aB * Cb;
            float g = p->Y - c.C.gCr * Cr - c.C.gCb * Cb;
            if (Premul && p->A != 0.0f)
            {
                const float inv = 1.0f / p->A;
                r *= inv; g *= inv; b *= inv;
            }
            R = transfer_decode(clamp_unit(r));
            G = transfer_decode(clamp_unit(g));
            B = transfer_decode(clamp_unit(b));
        }
    };

    // ---- RGB 444 10u: NO ALPHA, packed bitfields --------------------------
    // Bit layout from CommonPixFormat.hpp: _pad_ 0..1, B 2..11, G 12..21,
    // R 22..31. rgb10_layout_matches() below lets the caller assert that its
    // compiler laid the bitfields out that way.
    struct VReadRGB10
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_RGB_10u);
        static constexpr float kInv = 1.0f / static_cast<float>(kMaxCode10);

        static inline void read8(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& /*c*/,
                                 __m256& R, __m256& G, __m256& B)
        {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                row + static_cast<std::ptrdiff_t>(x) * 4));
            const __m256 s = _mm256_set1_ps(kInv);
            R = transfer_decode_ps(clamp_unit_ps(
                    _mm256_mul_ps(tenbit_lane_to_ps<22>(v), s)));
            G = transfer_decode_ps(clamp_unit_ps(
                    _mm256_mul_ps(tenbit_lane_to_ps<12>(v), s)));
            B = transfer_decode_ps(clamp_unit_ps(
                    _mm256_mul_ps(tenbit_lane_to_ps< 2>(v), s)));
        }

        static inline void read1(const std::uint8_t* row, std::int32_t x,
                                 const DecodeCtxV& /*c*/,
                                 float& R, float& G, float& B)
        {
            const PF_Pixel_RGB_10u* p =
                reinterpret_cast<const PF_Pixel_RGB_10u*>(row) + x;
            R = transfer_decode(clamp_unit(static_cast<float>(p->R) * kInv));
            G = transfer_decode(clamp_unit(static_cast<float>(p->G) * kInv));
            B = transfer_decode(clamp_unit(static_cast<float>(p->B) * kInv));
        }
    };

    //! One-off sanity check the caller may assert at plug-in init: true when the
    //! compiler allocated PF_Pixel_RGB_10u's bitfields LSB-first, which is what
    //! the vector shifts above assume (and what MSVC and GCC both do on x86).
    inline bool rgb10_layout_matches()
    {
        PF_Pixel_RGB_10u p;
        p._pad_ = 0u; p.B = 1u; p.G = 0u; p.R = 0u;
        std::uint32_t w = 0u;
        std::memcpy(&w, &p, sizeof(w));
        return (w == (1u << 2));
    }

    // =========================================================================
    // dispatch_vreader - the ONLY format switch on this path, a 1:1 mirror of
    // the scalar dispatch_reader. Runs once per frame.
    // =========================================================================
    template <typename F>
    inline void dispatch_vreader(ePrPixelFormat fmt, F&& f)
    {
        switch (fmt)
        {
            // BGRA_8u: B@0 G@8 R@16 A@24
            case fmt_BGRA_4444_8u:
            case fmt_BGRX_4444_8u:
                f(VReadInt8<PF_Pixel_BGRA_8u, 16, 8, 0, 24, false>()); break;
            case fmt_BGRP_4444_8u:
                f(VReadInt8<PF_Pixel_BGRA_8u, 16, 8, 0, 24, true >()); break;
            // ARGB_8u: A@0 R@8 G@16 B@24
            case fmt_ARGB_4444_8u:
            case fmt_XRGB_4444_8u:
                f(VReadInt8<PF_Pixel_ARGB_8u, 8, 16, 24, 0, false>()); break;
            case fmt_PRGB_4444_8u:
                f(VReadInt8<PF_Pixel_ARGB_8u, 8, 16, 24, 0, true >()); break;

            // BGRA_16u: B=even.lo G=even.hi R=odd.lo A=odd.hi
            case fmt_BGRA_4444_16u:
            case fmt_BGRX_4444_16u:
                f(VReadInt16<PF_Pixel_BGRA_16u, 2, 1, 0, 3, false>()); break;
            case fmt_BGRP_4444_16u:
                f(VReadInt16<PF_Pixel_BGRA_16u, 2, 1, 0, 3, true >()); break;
            // ARGB_16u: A=even.lo R=even.hi G=odd.lo B=odd.hi
            case fmt_ARGB_4444_16u:
            case fmt_XRGB_4444_16u:
                f(VReadInt16<PF_Pixel_ARGB_16u, 1, 2, 3, 0, false>()); break;
            case fmt_PRGB_4444_16u:
                f(VReadInt16<PF_Pixel_ARGB_16u, 1, 2, 3, 0, true >()); break;

            // BGRA_32f members B,G,R,A -> R=2 G=1 B=0 A=3
            case fmt_BGRA_4444_32f:
            case fmt_BGRX_4444_32f:
                f(VReadF32<PF_Pixel_BGRA_32f, 2, 1, 0, 3, false, false>()); break;
            case fmt_BGRP_4444_32f:
                f(VReadF32<PF_Pixel_BGRA_32f, 2, 1, 0, 3, true , false>()); break;
            case fmt_BGRA_4444_32f_Linear:
            case fmt_BGRX_4444_32f_Linear:
                f(VReadF32<PF_Pixel_BGRA_32f, 2, 1, 0, 3, false, true >()); break;
            case fmt_BGRP_4444_32f_Linear:
                f(VReadF32<PF_Pixel_BGRA_32f, 2, 1, 0, 3, true , true >()); break;
            // ARGB_32f members A,R,G,B -> R=1 G=2 B=3 A=0
            case fmt_ARGB_4444_32f:
            case fmt_XRGB_4444_32f:
                f(VReadF32<PF_Pixel_ARGB_32f, 1, 2, 3, 0, false, false>()); break;
            case fmt_ARGB_4444_32f_Linear:
            case fmt_XRGB_4444_32f_Linear:
                f(VReadF32<PF_Pixel_ARGB_32f, 1, 2, 3, 0, false, true >()); break;
            case fmt_PRGB_4444_32f:
                f(VReadF32<PF_Pixel_ARGB_32f, 1, 2, 3, 0, true , false>()); break;
            case fmt_PRGB_4444_32f_Linear:
                f(VReadF32<PF_Pixel_ARGB_32f, 1, 2, 3, 0, true , true >()); break;

            case fmt_VUYA_4444_8u_709:
            case fmt_VUYA_4444_8u:
            case fmt_VUYX_4444_8u_709:
            case fmt_VUYX_4444_8u:          f(VReadVUYA8 <false>()); break;
            case fmt_VUYP_4444_8u_709:
            case fmt_VUYP_4444_8u:          f(VReadVUYA8 <true >()); break;
            case fmt_VUYA_4444_32f_709:
            case fmt_VUYA_4444_32f:
            case fmt_VUYX_4444_32f_709:
            case fmt_VUYX_4444_32f:         f(VReadVUYA32<false>()); break;
            case fmt_VUYP_4444_32f_709:
            case fmt_VUYP_4444_32f:         f(VReadVUYA32<true >()); break;

            case fmt_RGB_444_10u:           f(VReadRGB10()); break;
            default: break;
        }
    }

    // =========================================================================
    // TRAVERSALS. Branch-free on format; vector body plus MANDATORY scalar tail.
    // =========================================================================

    //! Interleaved destination, tightly packed sizeX*sizeY*3.
    template <typename VReader, typename Ctx>
    inline void loop_ingest_avx2(const std::uint8_t* base,
                                 std::int32_t sizeX, std::int32_t sizeY,
                                 std::int32_t srcPitch, const Ctx& ctx,
                                 float* dstRGB_f32)
    {
        const std::ptrdiff_t byteStride =
            static_cast<std::ptrdiff_t>(srcPitch) *
            static_cast<std::ptrdiff_t>(VReader::kPixelBytes);
        const std::int32_t xVec = (sizeX >= kVecWidth) ? (sizeX - (sizeX % kVecWidth)) : 0;

        for (std::int32_t y = 0; y < sizeY; ++y)
        {
            const std::uint8_t* row = base + static_cast<std::ptrdiff_t>(y) * byteStride;
            float* dstRow = dstRGB_f32 +
                static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(sizeX) * 3;
            std::int32_t x = 0;
            for (; x < xVec; x += kVecWidth)
            {
                __m256 R, G, B;
                VReader::read8(row, x, ctx, R, G, B);
                store8_aos3_ps(dstRow + static_cast<std::ptrdiff_t>(x) * 3, R, G, B);
            }
            for (; x < sizeX; ++x)          // scalar tail - sizeX % 8 pixels
            {
                float* d = dstRow + static_cast<std::ptrdiff_t>(x) * 3;
                VReader::read1(row, x, ctx, d[0], d[1], d[2]);
            }
        }
    }

    //! Planar destination - what the film engine consumes. One shared pitch in
    //! ELEMENTS for all three planes.
    template <typename VReader, typename Ctx>
    inline void loop_ingest_planar_avx2(const std::uint8_t* base,
                                        std::int32_t sizeX, std::int32_t sizeY,
                                        std::int32_t srcPitch, const Ctx& ctx,
                                        float* RESTRICT dstR,
                                        float* RESTRICT dstG,
                                        float* RESTRICT dstB,
                                        std::int32_t dstPitch)
    {
        const std::ptrdiff_t byteStride =
            static_cast<std::ptrdiff_t>(srcPitch) *
            static_cast<std::ptrdiff_t>(VReader::kPixelBytes);
        const std::int32_t xVec = (sizeX >= kVecWidth) ? (sizeX - (sizeX % kVecWidth)) : 0;

        for (std::int32_t y = 0; y < sizeY; ++y)
        {
            const std::uint8_t* row = base + static_cast<std::ptrdiff_t>(y) * byteStride;
            const std::ptrdiff_t o = static_cast<std::ptrdiff_t>(y) *
                                     static_cast<std::ptrdiff_t>(dstPitch);
            float* pr = dstR + o;
            float* pg = dstG + o;
            float* pb = dstB + o;
            std::int32_t x = 0;
            for (; x < xVec; x += kVecWidth)
            {   // planar destination: no re-interleave at all, three stores
                __m256 R, G, B;
                VReader::read8(row, x, ctx, R, G, B);
                _mm256_storeu_ps(pr + x, R);
                _mm256_storeu_ps(pg + x, G);
                _mm256_storeu_ps(pb + x, B);
            }
            for (; x < sizeX; ++x)          // scalar tail
                VReader::read1(row, x, ctx, pr[x], pg[x], pb[x]);
        }
    }

    // =========================================================================
    // PUBLIC ENTRY POINTS. Parameter lists are identical to the scalar entry
    // points in AlgoPrFormatIngest.hpp so a caller can switch paths on a runtime
    // AVX2 check with no other change.
    //
    // ⚠ The LUT parameters are accepted and NOT USED: these paths are analytic
    // sRGB. Kept for signature compatibility - see the TRANSFER FUNCTION note at
    // the top of this file.
    // =========================================================================

    template <typename LUT8, typename LUT16, typename LUT10>
    void ingest_to_linear_f32_avx2
    (
        const void* src, std::int32_t sizeX, std::int32_t sizeY, std::int32_t srcPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        float* dstRGB_f32
    )
    {
        (void)lut8; (void)lut16; (void)lut10;
        const DecodeCtxV ctx = make_ctx(fmt);
        const std::uint8_t* base = static_cast<const std::uint8_t*>(src);
        dispatch_vreader(fmt, [&](auto reader)
        {
            using R = decltype(reader);
            loop_ingest_avx2<R>(base, sizeX, sizeY, srcPitch, ctx, dstRGB_f32);
        });
    }

    template <typename LUT8, typename LUT16, typename LUT10>
    void ingest_to_planar_f32_avx2
    (
        const void* src, std::int32_t sizeX, std::int32_t sizeY, std::int32_t srcPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        float* dstR, float* dstG, float* dstB, std::int32_t dstPitch
    )
    {
        (void)lut8; (void)lut16; (void)lut10;
        const DecodeCtxV ctx = make_ctx(fmt);
        const std::uint8_t* base = static_cast<const std::uint8_t*>(src);
        dispatch_vreader(fmt, [&](auto reader)
        {
            using R = decltype(reader);
            loop_ingest_planar_avx2<R>(base, sizeX, sizeY, srcPitch, ctx,
                                       dstR, dstG, dstB, dstPitch);
        });
    }

    // =========================================================================
    // REFERENCE-COMPATIBLE ENTRY POINT, kept verbatim in name and signature.
    //
    // The reference exposed exactly this one function. It is preserved so
    // existing call sites keep compiling; internally it now uses the vectorised
    // transpose and the shared kernels instead of its own scalar de-interleave.
    // `swapRB` remains unused for the same reason the reference noted: the
    // struct member names already resolve channel order at compile time.
    // =========================================================================
    template <typename Pix>
    inline void loop_ingest_f32_srgb(const std::uint8_t* /*base*/,
                                     std::int32_t /*sizeX*/, std::int32_t /*sizeY*/,
                                     std::int32_t /*srcPitchPx*/, bool /*swapRB*/,
                                     float* /*dstRGB*/)
    {
        // Only the two 32f layouts the reference supported are specialized. Any
        // other Pix is a call-site mistake and is caught here at compile time
        // rather than at link time.
        static_assert(sizeof(Pix) == 0,
            "loop_ingest_f32_srgb supports PF_Pixel_BGRA_32f and PF_Pixel_ARGB_32f "
            "only; use ingest_to_linear_f32_avx2() for the other formats.");
    }

    template <>
    inline void loop_ingest_f32_srgb<PF_Pixel_BGRA_32f>(
        const std::uint8_t* base, std::int32_t sizeX, std::int32_t sizeY,
        std::int32_t srcPitchPx, bool swapRB, float* dstRGB)
    {
        (void)swapRB;
        const DecodeCtxV ctx = make_ctx(fmt_BGRA_4444_32f);
        loop_ingest_avx2<VReadF32<PF_Pixel_BGRA_32f, 2, 1, 0, 3, false, false> >(
            base, sizeX, sizeY, srcPitchPx, ctx, dstRGB);
    }

    template <>
    inline void loop_ingest_f32_srgb<PF_Pixel_ARGB_32f>(
        const std::uint8_t* base, std::int32_t sizeX, std::int32_t sizeY,
        std::int32_t srcPitchPx, bool swapRB, float* dstRGB)
    {
        (void)swapRB;
        const DecodeCtxV ctx = make_ctx(fmt_ARGB_4444_32f);
        loop_ingest_avx2<VReadF32<PF_Pixel_ARGB_32f, 1, 2, 3, 0, false, false> >(
            base, sizeX, sizeY, srcPitchPx, ctx, dstRGB);
    }

} // namespace avx2
} // namespace AlgoPrIngest

#endif // __IMAGELAB2_INGEST_AVX2_HPP__
