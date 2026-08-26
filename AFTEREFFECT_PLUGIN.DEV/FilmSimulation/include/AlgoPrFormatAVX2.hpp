#ifndef __IMAGELAB2_PR_FORMAT_AVX2_HPP__
#define __IMAGELAB2_PR_FORMAT_AVX2_HPP__

// =============================================================================
// AlgoPrFormatAVX2.hpp - shared AVX2 kernels for the ingest and egress fast
// paths: transfer functions, rounding, AoS<->SoA transposes, clamps.
//
// Nothing here reads or writes an Adobe buffer; it is the arithmetic layer that
// AlgoIngestAVX2.hpp and AlgoEgressAVX2.hpp both build on, so the two
// directions cannot drift apart.
//
// -----------------------------------------------------------------------------
// AVX2 AND FMA ONLY - NO AVX-512, AND THAT IS NARROWER THAN IT SOUNDS
// -----------------------------------------------------------------------------
// Several convenient 256-bit intrinsics are NOT AVX2: they are AVX-512VL
// operating on ymm registers, and using them silently raises the CPU
// requirement. Deliberately avoided throughout:
//   _mm256_cvtepi32_epu8 / _mm256_cvtusepi32_epi8   (AVX512VL+BW)
//   _mm256_permutex2var_ps / _mm256_permutexvar_ps  (AVX512VL)
//   every _mm256_mask*_* and _mm256_maskz_*_*       (AVX512VL)
//   _mm256_reduce_*                                 (AVX512)
// The narrowing paths below therefore use _mm256_packus_* plus an explicit lane
// fixup, which is pure AVX2.
//
// -----------------------------------------------------------------------------
// UNALIGNED ACCESS IS THE DEFAULT AND COSTS ESSENTIALLY NOTHING
// -----------------------------------------------------------------------------
// Ae/Pr hand us buffers with arbitrary base addresses and arbitrary - possibly
// negative - pitches, so alignment cannot be assumed and must not be required.
// On Sandy Bridge and later, vmovups/vmovdqu have identical throughput and
// latency to the aligned forms when the access does not cross a 64-byte cache
// line; a line-crossing access costs about one extra cycle and a 4 KB
// page-crossing one costs more. Against a full-frame memory stream that is
// noise. Every host-buffer load and store below is unaligned. Aligned access is
// used ONLY for our own CACHE_ALIGNED stack scratch, where alignment is
// guaranteed by construction.
//
// -----------------------------------------------------------------------------
// ROUNDING PARITY - the one thing that must match the scalar path exactly
// -----------------------------------------------------------------------------
// pr_cvt_round_epi32() reproduces pr_round_half_away() from the scalar header
// bit for bit. _mm256_cvtps_epi32 would round half-to-even instead, so the
// vector body and the scalar tail OF THE SAME ROW would disagree by 1 LSB on
// exact-half values - which is precisely the class of defect that destroys a
// bit-exact round trip while looking like noise. One rule, both paths.
//
// C++14, no allocation, no STL. <immintrin.h> and <cmath> only.
// Build: GCC/Clang -mavx2 -mfma ; MSVC /arch:AVX2 (FMA implied).
// =============================================================================

#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include "AlgoPrFormatMath.hpp"
#include "AlgoPrFormatIngest.hpp"

namespace AlgoPrIngest
{
namespace avx2
{
    // =========================================================================
    // ROUNDING - half away from zero, identical to pr_round_half_away().
    // =========================================================================
    static inline __m256i pr_cvt_round_epi32(__m256 x)
    {
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 signMask =
            _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000u)));
        const __m256 bias = _mm256_or_ps(half, _mm256_and_ps(x, signMask));
        return _mm256_cvttps_epi32(_mm256_add_ps(x, bias));   // truncate after bias
    }

    // =========================================================================
    // CLAMPS. Named per destination ceiling so a path cannot pick up the wrong
    // one; the integer ceilings mirror kMaxCode8/10/16 from the scalar header.
    // =========================================================================
    static inline __m256 clamp_ps(__m256 v, __m256 lo, __m256 hi)
    {
        return _mm256_min_ps(_mm256_max_ps(v, lo), hi);
    }

    static inline __m256 clamp_unit_ps(__m256 v)
    {
        return clamp_ps(v, _mm256_setzero_ps(), _mm256_set1_ps(kMaxFloat32));
    }

    static inline __m256i clamp_epi32(__m256i v, int maxCode)
    {
        const __m256i lo = _mm256_setzero_si256();
        const __m256i hi = _mm256_set1_epi32(maxCode);
        return _mm256_min_epi32(_mm256_max_epi32(v, lo), hi);
    }

    //! Normalized [0,1] float -> clamped integer code. The vector twin of
    //! quantize_code(); same bias, same ceiling, same result.
    static inline __m256i quantize_code_ps(__m256 normalized, int maxCode)
    {
        const __m256 scaled =
            _mm256_mul_ps(normalized, _mm256_set1_ps(static_cast<float>(maxCode)));
        return clamp_epi32(pr_cvt_round_epi32(scaled), maxCode);
    }

    // =========================================================================
    // TRANSCENDENTALS. 8-wide log2 and exp2 built from the exponent/mantissa
    // decomposition, then sRGB decode and its exact inverse.
    //
    // These are carried over from the reference implementation's ingest header,
    // with the encode direction added - the reference had only the decode, so
    // the egress had no vector path at all.
    // =========================================================================

    //! 8-wide log2 for x in (0, +inf): x = 2^e * m, m in [1,2).
    static inline __m256 log2_ps(__m256 x)
    {
        const __m256i xi       = _mm256_castps_si256(x);
        const __m256i expMask  = _mm256_set1_epi32(0x7F800000);
        const __m256i mantMask = _mm256_set1_epi32(0x007FFFFF);
        const __m256i bias     = _mm256_set1_epi32(127);
        const __m256i e = _mm256_sub_epi32(
            _mm256_srli_epi32(_mm256_and_si256(xi, expMask), 23), bias);
        const __m256 ef = _mm256_cvtepi32_ps(e);
        const __m256 m  = _mm256_or_ps(
            _mm256_castsi256_ps(_mm256_and_si256(xi, mantMask)),
            _mm256_set1_ps(1.0f));
        const __m256 t  = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));   // t in [0,1)
        __m256 p = _mm256_set1_ps(1.459860554e-02f);                 // degree-7 Horner
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-7.592089396e-02f));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps( 1.886527228e-01f));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-3.214835301e-01f));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps( 4.717218708e-01f));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-7.202026917e-01f));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps( 1.442633691e+00f));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps( 8.116678600e-07f));
        return _mm256_add_ps(ef, p);
    }

    //! 8-wide exp2: 2^y = 2^floor(y) * 2^frac.
    static inline __m256 exp2_ps(__m256 y)
    {
        const __m256 fl = _mm256_floor_ps(y);
        const __m256 f  = _mm256_sub_ps(y, fl);                      // f in [0,1)
        __m256 p = _mm256_set1_ps(2.187125795e-04f);                 // degree-6 Horner
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.238241248e-03f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.686187232e-03f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.547891155e-02f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.402310971e-01f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.931468376e-01f));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.000000006e+00f));
        const __m256i ei = _mm256_cvtps_epi32(fl);
        const __m256i pw = _mm256_slli_epi32(
            _mm256_add_epi32(ei, _mm256_set1_epi32(127)), 23);
        return _mm256_mul_ps(p, _mm256_castsi256_ps(pw));
    }

    //! x^p for x >= 0, via exp2(p*log2(x)). Guards x <= 0 to 0 so that a
    //! negative sample - legal on a scene-linear buffer - cannot produce a NaN
    //! that then propagates through the blend.
    static inline __m256 pow_ps(__m256 x, float p)
    {
        const __m256 safe = _mm256_max_ps(x, _mm256_set1_ps(1.1754944e-38f));
        const __m256 r = exp2_ps(_mm256_mul_ps(_mm256_set1_ps(p), log2_ps(safe)));
        const __m256 pos = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OQ);
        return _mm256_and_ps(r, pos);
    }

    // =========================================================================
    // TRANSFER CURVES, 8-wide. Branch-free via blend. The constants and the
    // curve selection come from AlgoPrFormatMath.hpp - ONE definition shared
    // with the scalar path, so the two cannot drift. kTransfer is a compile-time
    // constant, so only the selected branch survives.
    // =========================================================================

    //! Display-encoded -> linear, 8 lanes. Vector twin of transfer_decode().
    static inline __m256 transfer_decode_ps(__m256 c)
    {
        if (kTransfer == kTransfer_Gamma)
            return pow_ps(c, kGammaExp);
        if (kTransfer == kTransfer_Rec709)
        {
            const __m256 lin  = _mm256_mul_ps(c, _mm256_set1_ps(k709InvSlope));
            const __m256 base = _mm256_mul_ps(_mm256_add_ps(c, _mm256_set1_ps(k709Offset)),
                                              _mm256_set1_ps(k709InvScale));
            const __m256 pw   = pow_ps(base, k709Gamma);
            const __m256 mask = _mm256_cmp_ps(c, _mm256_set1_ps(k709DecThreshold),
                                              _CMP_LE_OQ);
            return _mm256_blendv_ps(pw, lin, mask);
        }
        const __m256 lin  = _mm256_mul_ps(c, _mm256_set1_ps(kSrgbInvSlope));
        const __m256 base = _mm256_mul_ps(_mm256_add_ps(c, _mm256_set1_ps(kSrgbOffset)),
                                          _mm256_set1_ps(kSrgbInvScale));
        const __m256 pw   = pow_ps(base, kSrgbGamma);
        const __m256 mask = _mm256_cmp_ps(c, _mm256_set1_ps(kSrgbDecThreshold),
                                          _CMP_LE_OQ);
        return _mm256_blendv_ps(pw, lin, mask);
    }

    //! Linear -> display-encoded, 8 lanes. Vector twin of transfer_encode().
    //! ⚠ New in this implementation - the reference had no vector encode at all,
    //! which is why its whole egress ran scalar.
    static inline __m256 transfer_encode_ps(__m256 v)
    {
        if (kTransfer == kTransfer_Gamma)
            return pow_ps(v, kGammaInvExp);
        if (kTransfer == kTransfer_Rec709)
        {
            const __m256 lin  = _mm256_mul_ps(v, _mm256_set1_ps(k709Slope));
            const __m256 pw   = _mm256_fmsub_ps(_mm256_set1_ps(k709Scale),
                                                pow_ps(v, k709InvGamma),
                                                _mm256_set1_ps(k709Offset));
            const __m256 mask = _mm256_cmp_ps(v, _mm256_set1_ps(k709EncThreshold),
                                              _CMP_LE_OQ);
            return _mm256_blendv_ps(pw, lin, mask);
        }
        const __m256 lin  = _mm256_mul_ps(v, _mm256_set1_ps(kSrgbSlope));
        const __m256 pw   = _mm256_fmsub_ps(_mm256_set1_ps(kSrgbScale),
                                            pow_ps(v, kSrgbInvGamma),
                                            _mm256_set1_ps(kSrgbOffset));
        const __m256 mask = _mm256_cmp_ps(v, _mm256_set1_ps(kSrgbEncThreshold),
                                          _CMP_LE_OQ);
        return _mm256_blendv_ps(pw, lin, mask);
    }

    // =========================================================================
    // AoS <-> SoA for 4-channel 32-bit pixels, 8 pixels at a time.
    //
    // ⚠ THIS IS REAL SIMD DE-INTERLEAVE, and it replaces a scalar loop. The
    // reference's loop_ingest_f32_srgb() gathered channels with
    // `for (int k = 0; k < 8; ++k) { R[k]=row[x+k].R; ... }` and scattered the
    // results back the same way, so only the transcendental was vectorised
    // while every load and store stayed scalar. Defensible when the decode
    // dominates, but it is not what "avoid scalar per-pixel processing" asks
    // for. The unpack/shuffle sequence below does the transpose in 8 shuffles
    // for 8 pixels.
    //
    // The interleaved order after unpack+shuffle is p0,p2,p4,p6 | p1,p3,p5,p7,
    // so one _mm256_permutevar8x32_ps per channel restores p0..p7. For paths
    // whose destination is also interleaved the permute could be skipped on both
    // sides, but it is kept for clarity and because the planar destinations
    // require sequential order anyway.
    // =========================================================================
    static inline __m256i deinterleave_index()
    {
        return _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    }

    //! Load 8 four-channel pixels from `p` (unaligned) and split into channels.
    //! c0..c3 come out in the struct's own member order, so for a BGRA layout
    //! c0=B, c1=G, c2=R, c3=A, and for ARGB c0=A, c1=R, c2=G, c3=B.
    static inline void load8_aos4_ps(const float* p,
                                     __m256& c0, __m256& c1, __m256& c2, __m256& c3)
    {
        const __m256 a = _mm256_loadu_ps(p +  0);   // px0, px1
        const __m256 b = _mm256_loadu_ps(p +  8);   // px2, px3
        const __m256 c = _mm256_loadu_ps(p + 16);   // px4, px5
        const __m256 d = _mm256_loadu_ps(p + 24);   // px6, px7

        const __m256 t0 = _mm256_unpacklo_ps(a, b);
        const __m256 t1 = _mm256_unpackhi_ps(a, b);
        const __m256 t2 = _mm256_unpacklo_ps(c, d);
        const __m256 t3 = _mm256_unpackhi_ps(c, d);

        const __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
        const __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));

        const __m256i idx = deinterleave_index();
        c0 = _mm256_permutevar8x32_ps(s0, idx);
        c1 = _mm256_permutevar8x32_ps(s1, idx);
        c2 = _mm256_permutevar8x32_ps(s2, idx);
        c3 = _mm256_permutevar8x32_ps(s3, idx);
    }

    //! Exact inverse of load8_aos4_ps: interleave four channel vectors and store
    //! 8 four-channel pixels to `p` (unaligned).
    static inline void store8_aos4_ps(float* p,
                                      __m256 c0, __m256 c1, __m256 c2, __m256 c3)
    {
        // undo the permute: p0,p1,..,p7 -> p0,p2,p4,p6 | p1,p3,p5,p7
        const __m256i inv = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
        const __m256 s0 = _mm256_permutevar8x32_ps(c0, inv);
        const __m256 s1 = _mm256_permutevar8x32_ps(c1, inv);
        const __m256 s2 = _mm256_permutevar8x32_ps(c2, inv);
        const __m256 s3 = _mm256_permutevar8x32_ps(c3, inv);

        const __m256 t0 = _mm256_unpacklo_ps(s0, s1);   // c0 c1 c0 c1 | ...
        const __m256 t1 = _mm256_unpackhi_ps(s0, s1);
        const __m256 t2 = _mm256_unpacklo_ps(s2, s3);   // c2 c3 c2 c3 | ...
        const __m256 t3 = _mm256_unpackhi_ps(s2, s3);

        const __m256 a = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 b = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
        const __m256 c = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 d = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));

        _mm256_storeu_ps(p +  0, a);
        _mm256_storeu_ps(p +  8, b);
        _mm256_storeu_ps(p + 16, c);
        _mm256_storeu_ps(p + 24, d);
    }

    // =========================================================================
    // AoS <-> SoA for the TIGHTLY PACKED 3-CHANNEL float layout - the
    // interleaved RGB buffer the reference entry points produce and consume.
    //
    // 8 pixels are 24 floats = exactly three vectors, so the transpose is a
    // permute-and-blend per output register: three permutes and two blends each,
    // no memory round trip and no per-lane scalar store. The lane maps below are
    // written out rather than computed because a wrong entry here shuffles
    // channels within a group of eight pixels, which looks like a colour-fringe
    // artefact rather than like a bug.
    //
    //   memory: r0 g0 b0 r1 g1 b1 r2 g2 | b2 r3 g3 b3 r4 g4 b4 r5 | g5 b5 r6 g6 b6 r7 g7 b7
    // =========================================================================

    //! Load 8 tightly packed RGB pixels (unaligned) and split into channels.
    static inline void load8_aos3_ps(const float* p, __m256& R, __m256& G, __m256& B)
    {
        const __m256 i0 = _mm256_loadu_ps(p +  0);
        const __m256 i1 = _mm256_loadu_ps(p +  8);
        const __m256 i2 = _mm256_loadu_ps(p + 16);

        const __m256 r0 = _mm256_permutevar8x32_ps(i0, _mm256_setr_epi32(0,3,6,0,0,0,0,0));
        const __m256 r1 = _mm256_permutevar8x32_ps(i1, _mm256_setr_epi32(0,0,0,1,4,7,0,0));
        const __m256 r2 = _mm256_permutevar8x32_ps(i2, _mm256_setr_epi32(0,0,0,0,0,0,2,5));
        R = _mm256_blend_ps(_mm256_blend_ps(r0, r1, 0x38), r2, 0xC0);

        // G lanes: 0,1,2 <- i0[1,4,7] ; 3,4 <- i1[2,5] ; 5,6,7 <- i2[0,3,6]
        const __m256 g0 = _mm256_permutevar8x32_ps(i0, _mm256_setr_epi32(1,4,7,0,0,0,0,0));
        const __m256 g1 = _mm256_permutevar8x32_ps(i1, _mm256_setr_epi32(0,0,0,2,5,0,0,0));
        const __m256 g2 = _mm256_permutevar8x32_ps(i2, _mm256_setr_epi32(0,0,0,0,0,0,3,6));
        G = _mm256_blend_ps(_mm256_blend_ps(g0, g1, 0x18), g2, 0xE0);

        // B lanes: 0,1 <- i0[2,5] ; 2,3,4 <- i1[0,3,6] ; 5,6,7 <- i2[1,4,7]
        const __m256 b0 = _mm256_permutevar8x32_ps(i0, _mm256_setr_epi32(2,5,0,0,0,0,0,0));
        const __m256 b1 = _mm256_permutevar8x32_ps(i1, _mm256_setr_epi32(0,0,0,3,6,0,0,0));
        const __m256 b2 = _mm256_permutevar8x32_ps(i2, _mm256_setr_epi32(0,0,0,0,0,1,4,7));
        B = _mm256_blend_ps(_mm256_blend_ps(b0, b1, 0x1C), b2, 0xE0);
    }

    //! Exact inverse: interleave three channel vectors into 24 packed floats.
    static inline void store8_aos3_ps(float* p, __m256 R, __m256 G, __m256 B)
    {
        // out0 = r0 g0 b0 r1 g1 b1 r2 g2
        const __m256 pR0 = _mm256_permutevar8x32_ps(R, _mm256_setr_epi32(0,0,0,1,0,0,2,0));
        const __m256 pG0 = _mm256_permutevar8x32_ps(G, _mm256_setr_epi32(0,0,0,0,1,0,0,2));
        const __m256 pB0 = _mm256_permutevar8x32_ps(B, _mm256_setr_epi32(0,0,0,0,0,1,0,0));
        const __m256 o0  = _mm256_blend_ps(_mm256_blend_ps(pR0, pG0, 0x92), pB0, 0x24);

        // out1 = b2 r3 g3 b3 r4 g4 b4 r5
        const __m256 pR1 = _mm256_permutevar8x32_ps(R, _mm256_setr_epi32(0,3,0,0,4,0,0,5));
        const __m256 pG1 = _mm256_permutevar8x32_ps(G, _mm256_setr_epi32(0,0,3,0,0,4,0,0));
        const __m256 pB1 = _mm256_permutevar8x32_ps(B, _mm256_setr_epi32(2,0,0,3,0,0,4,0));
        const __m256 o1  = _mm256_blend_ps(_mm256_blend_ps(pR1, pG1, 0x24), pB1, 0x49);

        // out2 = g5 b5 r6 g6 b6 r7 g7 b7
        const __m256 pR2 = _mm256_permutevar8x32_ps(R, _mm256_setr_epi32(0,0,6,0,0,7,0,0));
        const __m256 pG2 = _mm256_permutevar8x32_ps(G, _mm256_setr_epi32(5,0,0,6,0,0,7,0));
        const __m256 pB2 = _mm256_permutevar8x32_ps(B, _mm256_setr_epi32(0,5,0,0,6,0,0,7));
        const __m256 o2  = _mm256_blend_ps(_mm256_blend_ps(pR2, pG2, 0x49), pB2, 0x92);

        _mm256_storeu_ps(p +  0, o0);
        _mm256_storeu_ps(p +  8, o1);
        _mm256_storeu_ps(p + 16, o2);
    }

    // =========================================================================
    // 8-BIT NARROWING, AVX2-ONLY.
    //
    // ⚠ _mm256_cvtepi32_epu8 is AVX-512VL, so it is off limits. The AVX2 route
    // is packs32->16 then packus16->8, both of which operate PER 128-BIT LANE
    // and therefore scramble the element order; the final
    // _mm256_permutevar8x32_epi32 with {0,4,1,5,2,6,3,7} undoes that. Getting
    // this fixup wrong produces plausible-looking output with the pixels
    // shuffled inside each group of eight, which is exactly the kind of bug
    // that survives a casual glance at a preview window.
    // =========================================================================
    static inline void store8_u8_from_epi32(std::uint8_t* p, __m256i v)
    {
        const __m256i p16 = _mm256_packs_epi32(v, v);        // lanewise 32->16
        const __m256i p8  = _mm256_packus_epi16(p16, p16);   // lanewise 16->8
        const __m256i fix = _mm256_permutevar8x32_epi32(
            p8, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        // the 8 useful bytes now sit in the low 8 bytes of the low lane
        const __m128i lo = _mm256_castsi256_si128(fix);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(p), lo);
    }

    //! Widen 8 unsigned bytes to 8 float lanes.
    static inline __m256 load8_u8_to_ps(const std::uint8_t* p)
    {
        const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
        return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b));
    }

    //! Widen 8 unsigned 16-bit values to 8 float lanes.
    static inline __m256 load8_u16_to_ps(const std::uint16_t* p)
    {
        const __m128i w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(w));
    }

    //! Narrow 8 int32 lanes to 8 unsigned 16-bit values.
    //! ⚠ _mm256_packus_epi32 is lanewise, hence the same permute fixup as the
    //! 8-bit path. The 0..32767 ceiling is inside packus's signed input range,
    //! so no saturation surprise.
    static inline void store8_u16_from_epi32(std::uint16_t* p, __m256i v)
    {
        const __m256i p16 = _mm256_packus_epi32(v, v);
        const __m256i fix = _mm256_permutevar8x32_epi32(
            p16, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p),
                         _mm256_castsi256_si128(fix));
    }

    // =========================================================================
    // SCALAR TWINS OF THE TRANSFER - the scalar tail must agree with the vector
    // body OF THE SAME ROW, so both call the SAME definition:
    // transfer_decode() / transfer_encode() from AlgoPrFormatMath.hpp.
    //
    // ⚠ This is what changed when the LUTs went away. Before, the scalar path
    // interpolated a caller-supplied table while the vector path evaluated a
    // curve, so the two were independent approximations that had to be kept
    // apart per frame. Now there is one curve, one set of constants and one
    // polynomial pair; scalar and AVX2 differ only by FMA association, i.e. a
    // float ULP. The aliases below exist only so the older names still resolve.
    // =========================================================================
    static inline float srgb_decode_1(float c) { return transfer_decode(c); }
    static inline float srgb_encode_1(float c) { return transfer_encode(c); }

    // =========================================================================
    // COMPILE-TIME LANE SELECTION. The channel order of a pixel struct is known
    // at compile time but the four de-interleaved channels live in four separate
    // registers, so the selection has to be by template parameter. The ternary
    // chain folds to a single register name; every operand is a valid vector, so
    // no branch and no dead-code instantiation problem.
    // =========================================================================
    template <int I>
    static inline __m256 sel4_ps(__m256 c0, __m256 c1, __m256 c2, __m256 c3)
    {
        return (I == 0) ? c0 : ((I == 1) ? c1 : ((I == 2) ? c2 : c3));
    }

    template <int I>
    static inline __m256i sel4_si(__m256i c0, __m256i c1, __m256i c2, __m256i c3)
    {
        return (I == 0) ? c0 : ((I == 1) ? c1 : ((I == 2) ? c2 : c3));
    }

    // =========================================================================
    // INTEGER DE-INTERLEAVE / RE-INTERLEAVE, AVX2 ONLY, EXACT.
    //
    // For the 8-bit and 10-bit formats one pixel is one dword, so a channel is
    // isolated with a shift and a mask - no gather, no shuffle, and element k of
    // the vector is still pixel k. That is both the cheapest and the least
    // error-prone route, and it is why these paths vectorise at all despite the
    // reference's (correct) finding that a LUT gather does not pay.
    //
    // For the 16-bit formats one pixel is TWO dwords, so the eight pixels of a
    // vector arrive as two loaded vectors and each channel is either the low or
    // the high half of the even or the odd dword. even_dwords()/odd_dwords()
    // collect those in SEQUENTIAL pixel order; the two unpack+permute sequences
    // below invert them.
    // =========================================================================

    //! Dwords 0,2,4,6 of m0 then of m1, in order: pixels 0..7's first dword.
    static inline __m256i even_dwords(__m256i m0, __m256i m1)
    {
        const __m256i a = _mm256_shuffle_epi32(m0, _MM_SHUFFLE(2, 0, 2, 0));
        const __m256i b = _mm256_shuffle_epi32(m1, _MM_SHUFFLE(2, 0, 2, 0));
        const __m256i u = _mm256_unpacklo_epi64(a, b);
        return _mm256_permutevar8x32_epi32(u, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    }

    //! Dwords 1,3,5,7 of m0 then of m1, in order: pixels 0..7's second dword.
    static inline __m256i odd_dwords(__m256i m0, __m256i m1)
    {
        const __m256i a = _mm256_shuffle_epi32(m0, _MM_SHUFFLE(3, 1, 3, 1));
        const __m256i b = _mm256_shuffle_epi32(m1, _MM_SHUFFLE(3, 1, 3, 1));
        const __m256i u = _mm256_unpacklo_epi64(a, b);
        return _mm256_permutevar8x32_epi32(u, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    }

    //! Inverse of even_dwords/odd_dwords: weave the two dword streams back into
    //! the two memory vectors (pixels 0..3 and pixels 4..7).
    static inline void weave_dwords(__m256i even, __m256i odd,
                                    __m256i& out0, __m256i& out1)
    {
        const __m256i a = _mm256_unpacklo_epi32(even, odd);
        const __m256i b = _mm256_unpackhi_epi32(even, odd);
        out0 = _mm256_permute2x128_si256(a, b, 0x20);
        out1 = _mm256_permute2x128_si256(a, b, 0x31);
    }

    //! Isolate one byte lane of a dword-per-pixel vector as 8 int32 lanes.
    //! ZERO extension - an 8-bit chroma must not arrive already-signed.
    template <int Shift>
    static inline __m256i byte_lane_epi32(__m256i v)
    {
        return (Shift == 24)
            ? _mm256_srli_epi32(v, 24)   // top byte: the shift already masks
            : _mm256_and_si256(_mm256_srli_epi32(v, Shift), _mm256_set1_epi32(0xFF));
    }

    //! Isolate one byte lane of a dword-per-pixel vector as 8 float lanes.
    template <int Shift>
    static inline __m256 byte_lane_to_ps(__m256i v)
    {
        return _mm256_cvtepi32_ps(byte_lane_epi32<Shift>(v));
    }

    //! Isolate one 10-bit field of a dword-per-pixel vector as 8 float lanes.
    template <int Shift>
    static inline __m256 tenbit_lane_to_ps(__m256i v)
    {
        const __m256i b = (Shift == 22)
            ? _mm256_srli_epi32(v, 22)
            : _mm256_and_si256(_mm256_srli_epi32(v, Shift), _mm256_set1_epi32(0x3FF));
        return _mm256_cvtepi32_ps(b);
    }

    //! Isolate the low or high 16-bit half of a dword vector as 8 int32 lanes.
    template <bool High>
    static inline __m256i half_word_epi32(__m256i v)
    {
        return High ? _mm256_srli_epi32(v, 16)
                    : _mm256_and_si256(v, _mm256_set1_epi32(0xFFFF));
    }

    //! Isolate the low or high 16-bit half of a dword vector as 8 float lanes.
    template <bool High>
    static inline __m256 half_word_to_ps(__m256i v)
    {
        return _mm256_cvtepi32_ps(half_word_epi32<High>(v));
    }

    // =========================================================================
    // UN-PREMULTIPLY, branch-free. A == 0 leaves the colour untouched, matching
    // the scalar readers' `if (Premul && p->A != 0)`. _mm256_div_ps is used
    // rather than a reciprocal-plus-Newton pair: the divide is exact, and on a
    // path that already does a pow() it is not the bottleneck.
    // =========================================================================
    static inline __m256 unpremul_ps(__m256 c, __m256 alpha)
    {
        const __m256 nz  = _mm256_cmp_ps(alpha, _mm256_setzero_ps(), _CMP_NEQ_OQ);
        const __m256 safe = _mm256_blendv_ps(_mm256_set1_ps(1.0f), alpha, nz);
        return _mm256_div_ps(c, safe);
    }

    // =========================================================================
    // YCbCr, vectorised. Same reconstruction constants as the scalar path,
    // broadcast once per frame by the caller.
    // =========================================================================
    struct YCbCrVec { __m256 aR, aB, gCr, gCb; };

    static inline YCbCrVec broadcast_matrix(const YCbCrToRGBf& C)
    {
        YCbCrVec v;
        v.aR  = _mm256_set1_ps(C.aR);
        v.aB  = _mm256_set1_ps(C.aB);
        v.gCr = _mm256_set1_ps(C.gCr);
        v.gCb = _mm256_set1_ps(C.gCb);
        return v;
    }

    //! Y'CbCr -> R'G'B', 8 pixels. Cb, Cr already signed.
    static inline void ycbcr_to_rgb_ps(__m256 Y, __m256 Cb, __m256 Cr,
                                       const YCbCrVec& M,
                                       __m256& R, __m256& G, __m256& B)
    {
        R = _mm256_fmadd_ps(M.aR, Cr, Y);
        B = _mm256_fmadd_ps(M.aB, Cb, Y);
        G = _mm256_fnmadd_ps(M.gCb, Cb, _mm256_fnmadd_ps(M.gCr, Cr, Y));
    }

    struct RGBToYCbCrVec { __m256 Kr, Kg, Kb, invAR, invAB; };

    static inline RGBToYCbCrVec broadcast_forward(const RGBToYCbCr& M)
    {
        RGBToYCbCrVec v;
        v.Kr    = _mm256_set1_ps(M.Kr);
        v.Kg    = _mm256_set1_ps(M.Kg);
        v.Kb    = _mm256_set1_ps(M.Kb);
        v.invAR = _mm256_set1_ps(1.0f / M.aR);
        v.invAB = _mm256_set1_ps(1.0f / M.aB);
        return v;
    }

    //! R'G'B' -> Y'CbCr, 8 pixels. Chroma comes out signed.
    static inline void rgb_to_ycbcr_ps(__m256 R, __m256 G, __m256 B,
                                       const RGBToYCbCrVec& M,
                                       __m256& Y, __m256& Cb, __m256& Cr)
    {
        Y  = _mm256_fmadd_ps(M.Kb, B, _mm256_fmadd_ps(M.Kg, G,
                             _mm256_mul_ps(M.Kr, R)));
        Cr = _mm256_mul_ps(_mm256_sub_ps(R, Y), M.invAR);
        Cb = _mm256_mul_ps(_mm256_sub_ps(B, Y), M.invAB);
    }

} // namespace avx2
} // namespace AlgoPrIngest

#endif // __IMAGELAB2_PR_FORMAT_AVX2_HPP__
