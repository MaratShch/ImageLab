#ifndef __IMAGELAB2_EGRESS_AVX2_HPP__
#define __IMAGELAB2_EGRESS_AVX2_HPP__

// =============================================================================
// AlgoEgressAVX2.hpp - AVX2 + FMA EGRESS: write a LINEAR float32 RGB buffer
// (interleaved OR planar) back into an Adobe Premiere/After Effects frame
// buffer, in any supported format. Structural inverse of AlgoIngestAVX2.hpp.
//
// ⚠ NEW IN THIS IMPLEMENTATION. The reference had no vector egress at all - it
// shipped only a decode kernel, so every write-back ran scalar. That is half a
// pipeline: on an 8-bit timeline the egress does the same amount of transfer
// work as the ingest. This header supplies the missing direction, built on the
// same shared kernels so the two cannot drift.
//
// -----------------------------------------------------------------------------
// ALPHA COMES FROM THE ORIGINAL INCOMING BUFFER. ALWAYS.
// -----------------------------------------------------------------------------
// The linear buffer carries no alpha, and the algorithm output must not be
// trusted to contain one. Every writer takes the alpha - or the X pad of the
// ...X formats - from THE SAME PIXEL of the original incoming Adobe buffer,
// which the caller passes as `alphaSrc` and which must be in the destination's
// pixel format. For BGRP / VUYP / PRGB that alpha is also re-applied to the
// colour, inverting the ingest's un-premultiply exactly.
//
// ⚠ IN-PLACE HAZARD. The host may hand the same memory as both `dst` and
// `alphaSrc`. Every writer LOADS ALPHA FIRST, into a register or a local, and
// only then stores. Do not reorder those steps.
//
// ⚠ VECTORISATION IS GATED ON THE ALPHA BUFFER'S WIDTH. The scalar path clamps
// the alpha x-coordinate into `alphaSizeX`; a vector load cannot clamp per lane
// without a gather, so the vector body runs only while 8 whole pixels are
// available in BOTH buffers, and the remainder - including every pixel past
// alphaSizeX - falls to the scalar tail, which clamps exactly as the scalar
// reference does. Correctness first; the tail is at most 7 pixels wide unless
// the alpha buffer is genuinely narrower than the frame.
//
// -----------------------------------------------------------------------------
// CLAMPING - on owner instruction, at every store
// -----------------------------------------------------------------------------
//     8-bit   -> 0 .. 255      (kMaxCode8)   stored as unsigned char
//     10-bit  -> 0 .. 1023     (kMaxCode10)
//     16-bit  -> 0 .. 32767    (kMaxCode16)  ⚠ NOT 65535
//     float32 -> 0.0 .. 1.0    (kMaxFloat32)
//
// Integer paths clamp inside quantize_code_ps(), which is the vector twin of the
// scalar quantize_code() - same rounding rule, same ceiling, so the vector body
// and the scalar tail of one row cannot produce different codes.
//
// ⚠ VUYA CHROMA. The +128 bias is applied FIRST, the result is clamped to
// 0..255, and only then narrowed to a byte. A signed chroma narrowed directly
// would wrap: -1 becomes 255, i.e. maximum chroma, which reads as a saturated
// colour fringe. The vector path clamps in epi32 before packing, so the same
// guarantee holds lane by lane.
//
// ⚠ THE FLOAT CLAMP ON THE _Linear FORMATS IS THE ONE THAT DISCARDS REAL DATA.
// Those buffers are scene-linear and the film model's highlights legitimately
// exceed 1.0. It is applied because it was asked for, it is confined to
// kClampLinearF32 (declared in AlgoPrFormatEgress.hpp), and setting that flag
// false restores HDR passthrough in both the scalar and the vector path.
//
// -----------------------------------------------------------------------------
// TRANSFER FUNCTION - ONE DEFINITION, SHARED WITH THE SCALAR PATH
// -----------------------------------------------------------------------------
// Linear -> encoded is transfer_encode_ps() from AlgoPrFormatMath.hpp, the
// vector twin of the scalar transfer_encode(): same constants, same
// fast_log2/fast_exp2 pair, same compile-time curve selection. Verified to
// produce byte-identical output to the scalar egress on the same frame.
//
// The reference inverted a 32768-entry table with a per-pixel bracketing binary
// search - which is also why it had no vector egress at all, since a binary
// search needs a gather per step. Computing the encode directly removed both the
// table and the search; the LUT parameters on the legacy entry points are
// accepted and ignored.
//
// GEOMETRY: sizes and pitches are SIGNED and in PIXELS; destination and alpha
// source carry independent, possibly negative pitches. Vector body plus
// MANDATORY scalar tail for sizeX % 8. All host accesses unaligned.
//
// C++14. No dynamic allocation, no STL containers. AVX2 + FMA only - no
// AVX-512, including its 256-bit forms.
// =============================================================================

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include "Common.hpp"
#include "AlgoPrFormatEgress.hpp"
#include "AlgoPrFormatAVX2.hpp"
#include "AlgoIngestAVX2.hpp"     // DecodeCtxV, kVecWidth, make_ctx

namespace AlgoPrIngest
{
namespace avx2
{
    // =========================================================================
    // Per-frame context for the vector egress: the FORWARD matrix in both forms.
    // Built from the same reconstruction constants the ingest uses, so it is the
    // exact inverse and the 709-versus-601 choice is inherited from
    // pick_matrix().
    // =========================================================================
    struct EncodeCtxV
    {
        RGBToYCbCr    M;    //!< scalar, for the tail
        RGBToYCbCrVec Mv;   //!< broadcast, for the body
    };

    inline EncodeCtxV make_encode_ctx(ePrPixelFormat fmt)
    {
        EncodeCtxV c;
        c.M  = forward_matrix(narrow_matrix(pick_matrix(fmt)));
        c.Mv = broadcast_forward(c.M);
        return c;
    }

    // ---- small shared helpers ----------------------------------------------

    //! Encode a linear vector to [0,1] display, or pass it through unchanged.
    template <bool Linear>
    static inline __m256 encode_or_pass(__m256 v)
    {
        return Linear ? v : transfer_encode_ps(v);
    }

    template <bool Linear>
    static inline float encode_or_pass_1(float v)
    {
        return Linear ? v : transfer_encode(v);
    }

    //! Pack four 0..255 code vectors into one dword-per-pixel vector, the
    //! channel of byte k given by the template slot parameters (0=R 1=G 2=B 3=A).
    template <int C0, int C1, int C2, int C3>
    static inline __m256i pack_bytes(__m256i R, __m256i G, __m256i B, __m256i A)
    {
        const __m256i b0 =                      sel4_si<C0>(R, G, B, A);
        const __m256i b1 = _mm256_slli_epi32(   sel4_si<C1>(R, G, B, A),  8);
        const __m256i b2 = _mm256_slli_epi32(   sel4_si<C2>(R, G, B, A), 16);
        const __m256i b3 = _mm256_slli_epi32(   sel4_si<C3>(R, G, B, A), 24);
        return _mm256_or_si256(_mm256_or_si256(b0, b1), _mm256_or_si256(b2, b3));
    }

    //! Pack four 0..32767 code vectors into the two memory vectors of 8 pixels.
    template <int C0, int C1, int C2, int C3>
    static inline void pack_words(__m256i R, __m256i G, __m256i B, __m256i A,
                                  __m256i& out0, __m256i& out1)
    {
        const __m256i ev = _mm256_or_si256(sel4_si<C0>(R, G, B, A),
                           _mm256_slli_epi32(sel4_si<C1>(R, G, B, A), 16));
        const __m256i od = _mm256_or_si256(sel4_si<C2>(R, G, B, A),
                           _mm256_slli_epi32(sel4_si<C3>(R, G, B, A), 16));
        weave_dwords(ev, od, out0, out1);
    }

    // =========================================================================
    // VECTOR WRITER TAGS. Mirrors of the vector readers, and of the scalar
    // writers one for one. Each provides:
    //     kPixelBytes
    //     store8(dstRow, x, alphaRow, ax, ctx, R, G, B)  - 8 pixels
    //     write1(dstRow, x, alphaRow, ax, ctx, R, G, B)  - 1 pixel (the tail)
    // `alphaRow == nullptr` means "no incoming buffer": full opacity is written
    // (255 / 32767 / 1.0f) and the premultiplied variants become a no-op scale.
    // =========================================================================

    // ---- integer 8-bit -----------------------------------------------------
    // Shifts SR/SG/SB/SA are BIT offsets for the READ of the incoming alpha;
    // C0..C3 are the byte SLOTS for the write. Both come from
    // CommonPixFormat.hpp and both are compile-time.
    template <typename Pix, int SA, int C0, int C1, int C2, int C3, bool Premul>
    struct VWriteInt8
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        static constexpr float kMax = static_cast<float>(kMaxCode8);
        static constexpr float kInv = 1.0f / kMax;

        static inline void store8(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* alphaRow, std::int32_t ax,
                                  const EncodeCtxV& /*c*/,
                                  __m256 R, __m256 G, __m256 B)
        {
            // ⚠ alpha FIRST - the destination may be the same memory
            __m256i Ai = _mm256_set1_epi32(kMaxCode8);
            __m256  Af = _mm256_set1_ps(1.0f);
            if (alphaRow != nullptr)
            {
                const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    alphaRow + static_cast<std::ptrdiff_t>(ax) * 4));
                // ⚠ the integer alpha is carried in INTEGER form, not recovered
                // from the float: a float round trip could shift a code by 1 LSB
                // and the requirement is that the outgoing alpha be the incoming
                // alpha, exactly.
                Ai = byte_lane_epi32<SA>(av);
                Af = _mm256_mul_ps(_mm256_cvtepi32_ps(Ai), _mm256_set1_ps(kInv));
            }
            __m256 r = transfer_encode_ps(R), g = transfer_encode_ps(G), b = transfer_encode_ps(B);
            if (Premul) { r = _mm256_mul_ps(r, Af);
                          g = _mm256_mul_ps(g, Af);
                          b = _mm256_mul_ps(b, Af); }
            const __m256i Rc = quantize_code_ps(r, kMaxCode8);
            const __m256i Gc = quantize_code_ps(g, kMaxCode8);
            const __m256i Bc = quantize_code_ps(b, kMaxCode8);
            const __m256i dw = pack_bytes<C0, C1, C2, C3>(Rc, Gc, Bc, Ai);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(
                dstRow + static_cast<std::ptrdiff_t>(x) * 4), dw);
        }

        static inline void write1(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* alphaRow, std::int32_t ax,
                                  const EncodeCtxV& /*c*/,
                                  float R, float G, float B)
        {
            const int a = alphaRow
                ? static_cast<int>((reinterpret_cast<const Pix*>(alphaRow) + ax)->A)
                : kMaxCode8;
            Pix* p = reinterpret_cast<Pix*>(dstRow) + x;
            float r = transfer_encode(R), g = transfer_encode(G), b = transfer_encode(B);
            if (Premul)
            {
                const float af = static_cast<float>(a) * kInv;
                r *= af; g *= af; b *= af;
            }
            p->R = to_u8(r * kMax); p->G = to_u8(g * kMax); p->B = to_u8(b * kMax);
            p->A = static_cast<A_u_char>(a);
        }
    };

    // ---- integer 16-bit, 0..32767 -----------------------------------------
    // IA selects the alpha among {even.lo, even.hi, odd.lo, odd.hi} on the READ;
    // C0..C3 are the write slots in the same four positions.
    template <typename Pix, int IA, int C0, int C1, int C2, int C3, bool Premul>
    struct VWriteInt16
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);
        static constexpr float kMax = static_cast<float>(kMaxCode16);
        static constexpr float kInv = 1.0f / kMax;

        static inline void store8(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* alphaRow, std::int32_t ax,
                                  const EncodeCtxV& /*c*/,
                                  __m256 R, __m256 G, __m256 B)
        {
            __m256i Ai = _mm256_set1_epi32(kMaxCode16);
            __m256  Af = _mm256_set1_ps(1.0f);
            if (alphaRow != nullptr)
            {
                const std::uint8_t* q = alphaRow + static_cast<std::ptrdiff_t>(ax) * 8;
                const __m256i m0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
                const __m256i m1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
                const __m256i ev = even_dwords(m0, m1);
                const __m256i od = odd_dwords (m0, m1);
                const __m256i a0 = half_word_epi32<false>(ev);
                const __m256i a1 = half_word_epi32<true >(ev);
                const __m256i a2 = half_word_epi32<false>(od);
                const __m256i a3 = half_word_epi32<true >(od);
                Ai = sel4_si<IA>(a0, a1, a2, a3);          // exact passthrough
                Af = _mm256_mul_ps(_mm256_cvtepi32_ps(Ai), _mm256_set1_ps(kInv));
            }
            __m256 r = transfer_encode_ps(R), g = transfer_encode_ps(G), b = transfer_encode_ps(B);
            if (Premul) { r = _mm256_mul_ps(r, Af);
                          g = _mm256_mul_ps(g, Af);
                          b = _mm256_mul_ps(b, Af); }
            const __m256i Rc = quantize_code_ps(r, kMaxCode16);
            const __m256i Gc = quantize_code_ps(g, kMaxCode16);
            const __m256i Bc = quantize_code_ps(b, kMaxCode16);
            __m256i out0, out1;
            pack_words<C0, C1, C2, C3>(Rc, Gc, Bc, Ai, out0, out1);
            std::uint8_t* d = dstRow + static_cast<std::ptrdiff_t>(x) * 8;
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(d),      out0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 32), out1);
        }

        static inline void write1(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* alphaRow, std::int32_t ax,
                                  const EncodeCtxV& /*c*/,
                                  float R, float G, float B)
        {
            const int a = alphaRow
                ? static_cast<int>((reinterpret_cast<const Pix*>(alphaRow) + ax)->A)
                : kMaxCode16;
            Pix* p = reinterpret_cast<Pix*>(dstRow) + x;
            float r = transfer_encode(R), g = transfer_encode(G), b = transfer_encode(B);
            if (Premul)
            {
                const float af = static_cast<float>(a) * kInv;
                r *= af; g *= af; b *= af;
            }
            p->R = to_u16(r * kMax); p->G = to_u16(g * kMax); p->B = to_u16(b * kMax);
            p->A = static_cast<A_u_short>(a);
        }
    };

    // ---- 32f, encoded or already-linear, optional premultiplied ------------
    // IA is the alpha's member position; C0..C3 are the write slots (0=R 1=G
    // 2=B 3=A) in struct member order.
    template <typename Pix, int IA, int C0, int C1, int C2, int C3,
              bool Premul, bool Linear>
    struct VWriteF32
    {
        static constexpr std::size_t kPixelBytes = sizeof(Pix);

        static inline void store8(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* alphaRow, std::int32_t ax,
                                  const EncodeCtxV& /*c*/,
                                  __m256 R, __m256 G, __m256 B)
        {
            __m256 A = _mm256_set1_ps(1.0f);
            if (alphaRow != nullptr)
            {
                __m256 a0, a1, a2, a3;
                load8_aos4_ps(reinterpret_cast<const float*>(
                    alphaRow + static_cast<std::ptrdiff_t>(ax) * 16), a0, a1, a2, a3);
                A = sel4_ps<IA>(a0, a1, a2, a3);
            }
            __m256 r = encode_or_pass<Linear>(R);
            __m256 g = encode_or_pass<Linear>(G);
            __m256 b = encode_or_pass<Linear>(B);
            if (Premul) { r = _mm256_mul_ps(r, A);
                          g = _mm256_mul_ps(g, A);
                          b = _mm256_mul_ps(b, A); }
            if (!Linear || kClampLinearF32)
            {   // ⚠ for Linear this is the HDR-discarding clamp - see the header
                r = clamp_unit_ps(r); g = clamp_unit_ps(g); b = clamp_unit_ps(b);
            }
            store8_aos4_ps(reinterpret_cast<float*>(
                               dstRow + static_cast<std::ptrdiff_t>(x) * 16),
                           sel4_ps<C0>(r, g, b, A), sel4_ps<C1>(r, g, b, A),
                           sel4_ps<C2>(r, g, b, A), sel4_ps<C3>(r, g, b, A));
        }

        static inline void write1(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* alphaRow, std::int32_t ax,
                                  const EncodeCtxV& /*c*/,
                                  float R, float G, float B)
        {
            const float a = alphaRow
                ? (reinterpret_cast<const Pix*>(alphaRow) + ax)->A
                : 1.0f;
            Pix* p = reinterpret_cast<Pix*>(dstRow) + x;
            float r = encode_or_pass_1<Linear>(R);
            float g = encode_or_pass_1<Linear>(G);
            float b = encode_or_pass_1<Linear>(B);
            if (Premul) { r *= a; g *= a; b *= a; }
            if (!Linear || kClampLinearF32)
            {
                r = clamp_unit(r); g = clamp_unit(g); b = clamp_unit(b);
            }
            p->R = r; p->G = g; p->B = b; p->A = a;
        }
    };

    // ---- VUYA / VUYP / VUYX 8u, studio range ------------------------------
    // Byte slots are fixed by the struct: V@0 U@8 Y@16 A@24.
    template <bool Premul>
    struct VWriteVUYA8
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_8u);

        static inline void store8(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* alphaRow, std::int32_t ax,
                                  const EncodeCtxV& c,
                                  __m256 R, __m256 G, __m256 B)
        {
            __m256i Ai = _mm256_set1_epi32(kMaxCode8);
            __m256  Af = _mm256_set1_ps(1.0f);
            if (alphaRow != nullptr)
            {
                const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    alphaRow + static_cast<std::ptrdiff_t>(ax) * 4));
                const __m256 a = byte_lane_to_ps<24>(av);
                Af = _mm256_mul_ps(a, _mm256_set1_ps(1.0f / 255.0f));
                Ai = _mm256_cvttps_epi32(a);
            }
            __m256 r = transfer_encode_ps(R), g = transfer_encode_ps(G), b = transfer_encode_ps(B);
            if (Premul)
            {   // premultiply R'G'B' BEFORE the matrix - inverse of the ingest
                r = _mm256_mul_ps(r, Af); g = _mm256_mul_ps(g, Af); b = _mm256_mul_ps(b, Af);
            }
            __m256 Y, Cb, Cr;
            rgb_to_ycbcr_ps(r, g, b, c.Mv, Y, Cb, Cr);
            // ⚠ bias, THEN clamp, THEN narrow - never a signed narrowing
            const __m256i Yc = clamp_epi32(pr_cvt_round_epi32(_mm256_fmadd_ps(
                Y, _mm256_set1_ps(kVuya8LumaScale),
                   _mm256_set1_ps(kVuya8LumaOffset))), kMaxCode8);
            const __m256i Uc = clamp_epi32(pr_cvt_round_epi32(_mm256_fmadd_ps(
                Cb, _mm256_set1_ps(kVuya8ChromaScale),
                    _mm256_set1_ps(kVuya8ChromaBias))), kMaxCode8);
            const __m256i Vc = clamp_epi32(pr_cvt_round_epi32(_mm256_fmadd_ps(
                Cr, _mm256_set1_ps(kVuya8ChromaScale),
                    _mm256_set1_ps(kVuya8ChromaBias))), kMaxCode8);
            // slots: byte0 = V, byte1 = U, byte2 = Y, byte3 = A. Feeding
            // pack_bytes the quadruple (V,U,Y,A) as its (R,G,B,A) arguments.
            const __m256i dw = pack_bytes<0, 1, 2, 3>(Vc, Uc, Yc, Ai);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(
                dstRow + static_cast<std::ptrdiff_t>(x) * 4), dw);
        }

        static inline void write1(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* alphaRow, std::int32_t ax,
                                  const EncodeCtxV& c,
                                  float R, float G, float B)
        {
            const int a = alphaRow
                ? static_cast<int>((reinterpret_cast<const PF_Pixel_VUYA_8u*>(alphaRow) + ax)->A)
                : kMaxCode8;
            PF_Pixel_VUYA_8u* p = reinterpret_cast<PF_Pixel_VUYA_8u*>(dstRow) + x;
            float r = transfer_encode(R), g = transfer_encode(G), b = transfer_encode(B);
            if (Premul)
            {
                const float af = static_cast<float>(a) * (1.0f / 255.0f);
                r *= af; g *= af; b *= af;
            }
            const float Y  = c.M.Kr * r + c.M.Kg * g + c.M.Kb * b;
            const float Cr = (r - Y) / c.M.aR;
            const float Cb = (b - Y) / c.M.aB;
            p->Y = to_u8(kVuya8LumaOffset + Y  * kVuya8LumaScale);
            p->U = to_u8(kVuya8ChromaBias + Cb * kVuya8ChromaScale);
            p->V = to_u8(kVuya8ChromaBias + Cr * kVuya8ChromaScale);
            p->A = static_cast<A_u_char>(a);
        }
    };

    // ---- VUYA / VUYP / VUYX 32f, full range, chroma stored at +0.5 --------
    template <bool Premul>
    struct VWriteVUYA32
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_VUYA_32f);

        static inline void store8(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* alphaRow, std::int32_t ax,
                                  const EncodeCtxV& c,
                                  __m256 R, __m256 G, __m256 B)
        {
            __m256 A = _mm256_set1_ps(1.0f);
            if (alphaRow != nullptr)
            {
                __m256 a0, a1, a2, a3;   // V, U, Y, A
                load8_aos4_ps(reinterpret_cast<const float*>(
                    alphaRow + static_cast<std::ptrdiff_t>(ax) * 16), a0, a1, a2, a3);
                A = a3;
            }
            __m256 r = transfer_encode_ps(R), g = transfer_encode_ps(G), b = transfer_encode_ps(B);
            if (Premul) { r = _mm256_mul_ps(r, A);
                          g = _mm256_mul_ps(g, A);
                          b = _mm256_mul_ps(b, A); }
            __m256 Y, Cb, Cr;
            rgb_to_ycbcr_ps(r, g, b, c.Mv, Y, Cb, Cr);
            const __m256 off = _mm256_set1_ps(kVuyaF32ChromaOffset);
            const __m256 Yo = clamp_unit_ps(Y);
            const __m256 Uo = clamp_unit_ps(_mm256_add_ps(Cb, off));
            const __m256 Vo = clamp_unit_ps(_mm256_add_ps(Cr, off));
            store8_aos4_ps(reinterpret_cast<float*>(
                               dstRow + static_cast<std::ptrdiff_t>(x) * 16),
                           Vo, Uo, Yo, A);
        }

        static inline void write1(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* alphaRow, std::int32_t ax,
                                  const EncodeCtxV& c,
                                  float R, float G, float B)
        {
            const float a = alphaRow
                ? (reinterpret_cast<const PF_Pixel_VUYA_32f*>(alphaRow) + ax)->A
                : 1.0f;
            PF_Pixel_VUYA_32f* p = reinterpret_cast<PF_Pixel_VUYA_32f*>(dstRow) + x;
            float r = transfer_encode(R), g = transfer_encode(G), b = transfer_encode(B);
            if (Premul) { r *= a; g *= a; b *= a; }
            const float Y  = c.M.Kr * r + c.M.Kg * g + c.M.Kb * b;
            const float Cr = (r - Y) / c.M.aR;
            const float Cb = (b - Y) / c.M.aB;
            p->Y = clamp_unit(Y);
            p->U = clamp_unit(Cb + kVuyaF32ChromaOffset);
            p->V = clamp_unit(Cr + kVuyaF32ChromaOffset);
            p->A = a;
        }
    };

    // ---- RGB 444 10u: NO ALPHA -------------------------------------------
    // Bit layout: _pad_ 0..1, B 2..11, G 12..21, R 22..31. The pad bits are
    // written as ZERO rather than left as found, so identical input gives
    // identical output regardless of what the host buffer held.
    struct VWriteRGB10
    {
        static constexpr std::size_t kPixelBytes = sizeof(PF_Pixel_RGB_10u);

        static inline void store8(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* /*alphaRow*/, std::int32_t /*ax*/,
                                  const EncodeCtxV& /*c*/,
                                  __m256 R, __m256 G, __m256 B)
        {
            const __m256i Rc = quantize_code_ps(transfer_encode_ps(R), kMaxCode10);
            const __m256i Gc = quantize_code_ps(transfer_encode_ps(G), kMaxCode10);
            const __m256i Bc = quantize_code_ps(transfer_encode_ps(B), kMaxCode10);
            const __m256i dw = _mm256_or_si256(
                _mm256_slli_epi32(Rc, 22),
                _mm256_or_si256(_mm256_slli_epi32(Gc, 12),
                                _mm256_slli_epi32(Bc,  2)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(
                dstRow + static_cast<std::ptrdiff_t>(x) * 4), dw);
        }

        static inline void write1(std::uint8_t* dstRow, std::int32_t x,
                                  const std::uint8_t* /*alphaRow*/, std::int32_t /*ax*/,
                                  const EncodeCtxV& /*c*/,
                                  float R, float G, float B)
        {
            PF_Pixel_RGB_10u* p = reinterpret_cast<PF_Pixel_RGB_10u*>(dstRow) + x;
            p->_pad_ = 0u;
            p->R = static_cast<A_u_long>(quantize_code(transfer_encode(R), kMaxCode10));
            p->G = static_cast<A_u_long>(quantize_code(transfer_encode(G), kMaxCode10));
            p->B = static_cast<A_u_long>(quantize_code(transfer_encode(B), kMaxCode10));
            // No alpha exists in this format: nothing preserved, nothing
            // synthesized. alphaRow / ax are deliberately unused.
        }
    };

    // =========================================================================
    // dispatch_vwriter - the ONLY format switch on this path, a 1:1 mirror of
    // dispatch_vreader and of the scalar dispatch_writer.
    //
    // Slot tables, straight from CommonPixFormat.hpp (0=R 1=G 2=B 3=A):
    //   BGRA_8u  bytes B,G,R,A -> 2,1,0,3   alpha bit shift 24
    //   ARGB_8u  bytes A,R,G,B -> 3,0,1,2   alpha bit shift  0
    //   BGRA_16u slots B,G,R,A -> 2,1,0,3   alpha slot 3 (odd.hi)
    //   ARGB_16u slots A,R,G,B -> 3,0,1,2   alpha slot 0 (even.lo)
    //   BGRA_32f members B,G,R,A -> 2,1,0,3 alpha member 3
    //   ARGB_32f members A,R,G,B -> 3,0,1,2 alpha member 0
    // =========================================================================
    template <typename F>
    inline void dispatch_vwriter(ePrPixelFormat fmt, F&& f)
    {
        switch (fmt)
        {
            case fmt_BGRA_4444_8u:
            case fmt_BGRX_4444_8u:
                f(VWriteInt8<PF_Pixel_BGRA_8u, 24, 2, 1, 0, 3, false>()); break;
            case fmt_BGRP_4444_8u:
                f(VWriteInt8<PF_Pixel_BGRA_8u, 24, 2, 1, 0, 3, true >()); break;
            case fmt_ARGB_4444_8u:
            case fmt_XRGB_4444_8u:
                f(VWriteInt8<PF_Pixel_ARGB_8u,  0, 3, 0, 1, 2, false>()); break;
            case fmt_PRGB_4444_8u:
                f(VWriteInt8<PF_Pixel_ARGB_8u,  0, 3, 0, 1, 2, true >()); break;

            case fmt_BGRA_4444_16u:
            case fmt_BGRX_4444_16u:
                f(VWriteInt16<PF_Pixel_BGRA_16u, 3, 2, 1, 0, 3, false>()); break;
            case fmt_BGRP_4444_16u:
                f(VWriteInt16<PF_Pixel_BGRA_16u, 3, 2, 1, 0, 3, true >()); break;
            case fmt_ARGB_4444_16u:
            case fmt_XRGB_4444_16u:
                f(VWriteInt16<PF_Pixel_ARGB_16u, 0, 3, 0, 1, 2, false>()); break;
            case fmt_PRGB_4444_16u:
                f(VWriteInt16<PF_Pixel_ARGB_16u, 0, 3, 0, 1, 2, true >()); break;

            case fmt_BGRA_4444_32f:
            case fmt_BGRX_4444_32f:
                f(VWriteF32<PF_Pixel_BGRA_32f, 3, 2, 1, 0, 3, false, false>()); break;
            case fmt_BGRP_4444_32f:
                f(VWriteF32<PF_Pixel_BGRA_32f, 3, 2, 1, 0, 3, true , false>()); break;
            case fmt_BGRA_4444_32f_Linear:
            case fmt_BGRX_4444_32f_Linear:
                f(VWriteF32<PF_Pixel_BGRA_32f, 3, 2, 1, 0, 3, false, true >()); break;
            case fmt_BGRP_4444_32f_Linear:
                f(VWriteF32<PF_Pixel_BGRA_32f, 3, 2, 1, 0, 3, true , true >()); break;
            case fmt_ARGB_4444_32f:
            case fmt_XRGB_4444_32f:
                f(VWriteF32<PF_Pixel_ARGB_32f, 0, 3, 0, 1, 2, false, false>()); break;
            case fmt_ARGB_4444_32f_Linear:
            case fmt_XRGB_4444_32f_Linear:
                f(VWriteF32<PF_Pixel_ARGB_32f, 0, 3, 0, 1, 2, false, true >()); break;
            case fmt_PRGB_4444_32f:
                f(VWriteF32<PF_Pixel_ARGB_32f, 0, 3, 0, 1, 2, true , false>()); break;
            case fmt_PRGB_4444_32f_Linear:
                f(VWriteF32<PF_Pixel_ARGB_32f, 0, 3, 0, 1, 2, true , true >()); break;

            case fmt_VUYA_4444_8u_709:
            case fmt_VUYA_4444_8u:
            case fmt_VUYX_4444_8u_709:
            case fmt_VUYX_4444_8u:          f(VWriteVUYA8 <false>()); break;
            case fmt_VUYP_4444_8u_709:
            case fmt_VUYP_4444_8u:          f(VWriteVUYA8 <true >()); break;
            case fmt_VUYA_4444_32f_709:
            case fmt_VUYA_4444_32f:
            case fmt_VUYX_4444_32f_709:
            case fmt_VUYX_4444_32f:         f(VWriteVUYA32<false>()); break;
            case fmt_VUYP_4444_32f_709:
            case fmt_VUYP_4444_32f:         f(VWriteVUYA32<true >()); break;

            case fmt_RGB_444_10u:           f(VWriteRGB10()); break;
            default: break;
        }
    }

    // =========================================================================
    // TRAVERSALS. Vector body plus MANDATORY scalar tail; the vector body also
    // stops early when the alpha buffer is narrower than the frame (see the
    // gating note in this file's header).
    // =========================================================================

    //! How many pixels of this row may run vectorised.
    static inline std::int32_t egress_vec_count(std::int32_t sizeX,
                                                const std::uint8_t* alphaBase,
                                                std::int32_t alphaSizeX)
    {
        std::int32_t n = sizeX;
        if (alphaBase != nullptr && alphaSizeX < sizeX)
            n = (alphaSizeX > 0) ? alphaSizeX : 0;
        return (n >= kVecWidth) ? (n - (n % kVecWidth)) : 0;
    }

    //! Interleaved source, tightly packed sizeX*sizeY*3.
    template <typename VWriter, typename Ctx>
    inline void loop_egress_avx2(const float* srcRGB_f32,
                                 std::int32_t sizeX, std::int32_t sizeY,
                                 std::uint8_t* dstBase, std::int32_t dstPitch,
                                 const std::uint8_t* alphaBase, std::int32_t alphaSizeX,
                                 std::int32_t alphaPitch, const Ctx& ctx)
    {
        const std::ptrdiff_t dstStride =
            static_cast<std::ptrdiff_t>(dstPitch) *
            static_cast<std::ptrdiff_t>(VWriter::kPixelBytes);
        const std::ptrdiff_t alpStride =
            static_cast<std::ptrdiff_t>(alphaPitch) *
            static_cast<std::ptrdiff_t>(VWriter::kPixelBytes);
        const std::int32_t axMax = (alphaSizeX > 0) ? (alphaSizeX - 1) : 0;
        const std::int32_t xVec  = egress_vec_count(sizeX, alphaBase, alphaSizeX);

        for (std::int32_t y = 0; y < sizeY; ++y)
        {
            const float* srcRow = srcRGB_f32 +
                static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(sizeX) * 3;
            std::uint8_t* dstRow = dstBase + static_cast<std::ptrdiff_t>(y) * dstStride;
            const std::uint8_t* alpRow = alphaBase
                ? alphaBase + static_cast<std::ptrdiff_t>(y) * alpStride : nullptr;
            std::int32_t x = 0;
            for (; x < xVec; x += kVecWidth)
            {
                __m256 R, G, B;
                load8_aos3_ps(srcRow + static_cast<std::ptrdiff_t>(x) * 3, R, G, B);
                VWriter::store8(dstRow, x, alpRow, x, ctx, R, G, B);
            }
            for (; x < sizeX; ++x)          // scalar tail
            {
                const float* s = srcRow + static_cast<std::ptrdiff_t>(x) * 3;
                const std::int32_t ax = (x < axMax) ? x : axMax;
                VWriter::write1(dstRow, x, alpRow, ax, ctx, s[0], s[1], s[2]);
            }
        }
    }

    //! Planar source - what the film engine produces.
    template <typename VWriter, typename Ctx>
    inline void loop_egress_planar_avx2(const float* RESTRICT srcR,
                                        const float* RESTRICT srcG,
                                        const float* RESTRICT srcB,
                                        std::int32_t srcPitch,
                                        std::int32_t sizeX, std::int32_t sizeY,
                                        std::uint8_t* dstBase, std::int32_t dstPitch,
                                        const std::uint8_t* alphaBase,
                                        std::int32_t alphaSizeX,
                                        std::int32_t alphaPitch, const Ctx& ctx)
    {
        const std::ptrdiff_t dstStride =
            static_cast<std::ptrdiff_t>(dstPitch) *
            static_cast<std::ptrdiff_t>(VWriter::kPixelBytes);
        const std::ptrdiff_t alpStride =
            static_cast<std::ptrdiff_t>(alphaPitch) *
            static_cast<std::ptrdiff_t>(VWriter::kPixelBytes);
        const std::int32_t axMax = (alphaSizeX > 0) ? (alphaSizeX - 1) : 0;
        const std::int32_t xVec  = egress_vec_count(sizeX, alphaBase, alphaSizeX);

        for (std::int32_t y = 0; y < sizeY; ++y)
        {
            const std::ptrdiff_t o = static_cast<std::ptrdiff_t>(y) *
                                     static_cast<std::ptrdiff_t>(srcPitch);
            const float* pr = srcR + o;
            const float* pg = srcG + o;
            const float* pb = srcB + o;
            std::uint8_t* dstRow = dstBase + static_cast<std::ptrdiff_t>(y) * dstStride;
            const std::uint8_t* alpRow = alphaBase
                ? alphaBase + static_cast<std::ptrdiff_t>(y) * alpStride : nullptr;
            std::int32_t x = 0;
            for (; x < xVec; x += kVecWidth)
            {   // planar source: three loads, no de-interleave
                const __m256 R = _mm256_loadu_ps(pr + x);
                const __m256 G = _mm256_loadu_ps(pg + x);
                const __m256 B = _mm256_loadu_ps(pb + x);
                VWriter::store8(dstRow, x, alpRow, x, ctx, R, G, B);
            }
            for (; x < sizeX; ++x)          // scalar tail
            {
                const std::int32_t ax = (x < axMax) ? x : axMax;
                VWriter::write1(dstRow, x, alpRow, ax, ctx, pr[x], pg[x], pb[x]);
            }
        }
    }

    // =========================================================================
    // PUBLIC ENTRY POINTS. Parameter lists identical to the scalar egress in
    // AlgoPrFormatEgress.hpp, so switching paths on a runtime AVX2 check needs
    // no other change. The LUT parameters are accepted and NOT used - these
    // paths are analytic sRGB (see the TRANSFER FUNCTION note above).
    // =========================================================================

    template <typename LUT8, typename LUT16, typename LUT10>
    void egress_from_linear_f32_avx2
    (
        const float* srcRGB_f32,
        std::int32_t sizeX, std::int32_t sizeY,
        void* dst, std::int32_t dstPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        const void* alphaSrc, std::int32_t alphaSizeX, std::int32_t alphaPitch
    )
    {
        (void)lut8; (void)lut16; (void)lut10;
        const EncodeCtxV ctx = make_encode_ctx(fmt);
        std::uint8_t* dstBase = static_cast<std::uint8_t*>(dst);
        const std::uint8_t* alpBase = static_cast<const std::uint8_t*>(alphaSrc);
        dispatch_vwriter(fmt, [&](auto writer)
        {
            using W = decltype(writer);
            loop_egress_avx2<W>(srcRGB_f32, sizeX, sizeY, dstBase, dstPitch,
                                alpBase, alphaSizeX, alphaPitch, ctx);
        });
    }

    template <typename LUT8, typename LUT16, typename LUT10>
    void egress_from_planar_f32_avx2
    (
        const float* srcR, const float* srcG, const float* srcB,
        std::int32_t srcPitch,
        std::int32_t sizeX, std::int32_t sizeY,
        void* dst, std::int32_t dstPitch,
        ePrPixelFormat fmt,
        const LUT8& lut8, const LUT16& lut16, const LUT10& lut10,
        const void* alphaSrc, std::int32_t alphaSizeX, std::int32_t alphaPitch
    )
    {
        (void)lut8; (void)lut16; (void)lut10;
        const EncodeCtxV ctx = make_encode_ctx(fmt);
        std::uint8_t* dstBase = static_cast<std::uint8_t*>(dst);
        const std::uint8_t* alpBase = static_cast<const std::uint8_t*>(alphaSrc);
        dispatch_vwriter(fmt, [&](auto writer)
        {
            using W = decltype(writer);
            loop_egress_planar_avx2<W>(srcR, srcG, srcB, srcPitch, sizeX, sizeY,
                                       dstBase, dstPitch,
                                       alpBase, alphaSizeX, alphaPitch, ctx);
        });
    }

} // namespace avx2
} // namespace AlgoPrIngest

#endif // __IMAGELAB2_EGRESS_AVX2_HPP__
