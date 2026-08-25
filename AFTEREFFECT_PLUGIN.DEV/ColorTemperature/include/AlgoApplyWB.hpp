#ifndef __IMAGELAB2_APPLY_WB_HPP__
#define __IMAGELAB2_APPLY_WB_HPP__

// =============================================================================
// AlgoApplyWB.hpp - PHASE 2 / Step D : apply the white-balance matrix.
//
//      rgb_out = M_wb * rgb_in        (linear, interleaved, working-space)
//
// Input  : the canonical linear buffer produced by Phase 1 (ingest or the
//          fused AVX2 measure pass) - float32, interleaved RGB, white = 1.0,
//          unclamped.
// Output : the corrected linear buffer, ready for egress_from_linear_f32().
//          May be the SAME pointer as the input (in-place is supported).
//
// M_wb comes from AlgoWhiteBalance.hpp / build_wb_matrix() and is built ONCE
// per frame. This file only applies it - plus the three user-facing modifiers
// that must be folded into the matrix rather than the loop:
//
//   * STRENGTH (0..1)   : blends M_wb toward identity. Applied to the MATRIX
//                         (one lerp of 9 doubles per frame), not per pixel.
//   * HIGHLIGHT-SAFE    : optional scaling so no channel response exceeds 1
//                         (nothing amplified -> no NEW clipping; the image
//                         darkens instead). Off by default - 'strength' is
//                         the primary mitigation for extreme corrections.
//   * CLIP POLICY       : clip_Auto clamps to [0,1] EXCEPT for _Linear target
//                         formats, which legitimately carry HDR (>1) and
//                         small negatives and must pass through untouched.
//                         clip_Always / clip_Never force the behaviour.
//                         Compile with _NOT_CLIP_OUTPUT to force clip_Never
//                         as the default at build time.
//
// PRECISION: the matrix is prepared in double, then broadcast to float for
// the per-pixel work (a 3x3 multiply on values <= a few units - float is
// ample; the accumulation-over-millions problem of the measure pass does not
// arise here).
//
// PORTABILITY: scalar path is plain C++14. AVX2 path uses only
// <immintrin.h>; GCC/Clang -mavx2 -mfma, MSVC /arch:AVX2. Single-threaded
// per call - the host owns concurrency.
// =============================================================================

#include <cstdint>
#include <cmath>
#include "Common.hpp"                 // CACHE_ALIGN
#include "AlgoPrFormatIngest.hpp"     // ePrPixelFormat (for clip_Auto)

#ifdef __AVX2__
  #include <immintrin.h>
#endif

namespace AlgoWB
{
    // ---------------------------------------------------------------- policy
    enum eClipPolicy : int32_t
    {
        clip_Auto   = 0,   // clamp unless the TARGET format is _Linear (HDR)
        clip_Always = 1,
        clip_Never  = 2
    };

    // Does this target format legitimately carry values outside [0,1]?
    inline bool format_is_linear_hdr (AlgoPrIngest::ePrPixelFormat fmt) noexcept
    {
        using namespace AlgoPrIngest;
        return fmt == fmt_BGRA_4444_32f_Linear || fmt == fmt_BGRX_4444_32f_Linear ||
               fmt == fmt_BGRP_4444_32f_Linear || fmt == fmt_ARGB_4444_32f_Linear ||
               fmt == fmt_XRGB_4444_32f_Linear || fmt == fmt_PRGB_4444_32f_Linear;
    }

    // Resolve the effective decision for one frame.
    inline bool resolve_clip (eClipPolicy policy,
                              AlgoPrIngest::ePrPixelFormat targetFmt) noexcept
    {
    #ifdef _NOT_CLIP_OUTPUT
        if (policy == clip_Auto) return false;      // build-time override
    #endif
        switch (policy)
        {
            case clip_Always: return true;
            case clip_Never:  return false;
            case clip_Auto:
            default:          return !format_is_linear_hdr(targetFmt);
        }
    }

    // ------------------------------------------------------- matrix modifiers
    // STRENGTH: M' = (1-s)*I + s*M. s<=0 -> identity, s>=1 -> M unchanged.
    inline void apply_strength (const double M[9], double s, double Mout[9]) noexcept
    {
        if (s >= 1.0) { for (int i = 0; i < 9; ++i) Mout[i] = M[i]; return; }
        if (s <= 0.0) {
            Mout[0]=1;Mout[1]=0;Mout[2]=0;
            Mout[3]=0;Mout[4]=1;Mout[5]=0;
            Mout[6]=0;Mout[7]=0;Mout[8]=1; return;
        }
        const double k = 1.0 - s;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                Mout[r*3+c] = s * M[r*3+c] + ((r == c) ? k : 0.0);
    }

    // HIGHLIGHT-SAFE: scale so max row-sum (the response to neutral white)
    // is <= 1, i.e. no channel amplifies. Returns the scale that was used.
    inline double apply_highlight_safe (double M[9]) noexcept
    {
        double mx = 0.0;
        for (int r = 0; r < 3; ++r) {
            const double resp = M[r*3+0] + M[r*3+1] + M[r*3+2];
            if (resp > mx) mx = resp;
        }
        if (mx <= 1.0 || mx == 0.0) return 1.0;
        const double k = 1.0 / mx;
        for (int i = 0; i < 9; ++i) M[i] *= k;
        return k;
    }

    // Everything a frame needs, prepared once.
    struct ApplyParams
    {
        double        strength;         // 0..1
        bool          highlightSafe;    // default false
        eClipPolicy   clipPolicy;       // default clip_Auto
        AlgoPrIngest::ePrPixelFormat targetFmt;   // for clip_Auto
        ApplyParams()
            : strength(1.0), highlightSafe(false), clipPolicy(clip_Auto)
            , targetFmt(AlgoPrIngest::fmt_BGRA_4444_8u) {}
    };

    // Fold strength + highlight-safe into the final matrix and resolve clipping.
    // Call ONCE per frame, then hand Mfinal to the apply loop.
    inline void prepare_apply (const double M_wb[9], const ApplyParams& p,
                               double Mfinal[9], bool& doClip) noexcept
    {
        apply_strength(M_wb, p.strength, Mfinal);
        if (p.highlightSafe) apply_highlight_safe(Mfinal);
        doClip = resolve_clip(p.clipPolicy, p.targetFmt);
    }

    // ------------------------------------------------------------ scalar apply
    // Reference implementation. src may alias dst (in-place).
    inline void apply_wb_scalar (const float* src, float* dst,
                                 int32_t sizeX, int32_t sizeY,
                                 const double Mfinal[9], bool doClip) noexcept
    {
        const float m0 = (float)Mfinal[0], m1 = (float)Mfinal[1], m2 = (float)Mfinal[2];
        const float m3 = (float)Mfinal[3], m4 = (float)Mfinal[4], m5 = (float)Mfinal[5];
        const float m6 = (float)Mfinal[6], m7 = (float)Mfinal[7], m8 = (float)Mfinal[8];
        const int64_t n = (int64_t)sizeX * (int64_t)sizeY;
        for (int64_t i = 0; i < n; ++i)
        {
            const float r = src[i*3 + 0], g = src[i*3 + 1], b = src[i*3 + 2];
            // NOTE the association order: a*r + (b*g + c*b). It mirrors the
            // AVX2 kernel's fmadd nesting, which makes the two paths
            // BIT-EXACTLY equal (verified) - do not "simplify" the
            // parentheses away, that reintroduces a rounding divergence.
            float R = m0*r + (m1*g + m2*b);
            float G = m3*r + (m4*g + m5*b);
            float B = m6*r + (m7*g + m8*b);
            if (doClip) {
                R = (R < 0.f) ? 0.f : ((R > 1.f) ? 1.f : R);
                G = (G < 0.f) ? 0.f : ((G > 1.f) ? 1.f : G);
                B = (B < 0.f) ? 0.f : ((B > 1.f) ? 1.f : B);
            }
            dst[i*3 + 0] = R; dst[i*3 + 1] = G; dst[i*3 + 2] = B;
        }
    }

#ifdef __AVX2__
    // -------------------------------------------------------------- AVX2 apply
    // Deinterleave 8 pixels -> SoA, 9 FMAs, re-interleave. Same permute/blend
    // pattern as the measure pass's store_rgb8 (verified there).
    namespace detail
    {
        static inline void load_rgb8 (const float* s, __m256& R, __m256& G, __m256& B)
        {
            const __m256 a0 = _mm256_loadu_ps(s +  0);   // r0 g0 b0 r1 g1 b1 r2 g2
            const __m256 a1 = _mm256_loadu_ps(s +  8);   // b2 r3 g3 b3 r4 g4 b4 r5
            const __m256 a2 = _mm256_loadu_ps(s + 16);   // g5 b5 r6 g6 b6 r7 g7 b7
            CACHE_ALIGN float t[24];
            _mm256_store_ps(t +  0, a0);
            _mm256_store_ps(t +  8, a1);
            _mm256_store_ps(t + 16, a2);
            CACHE_ALIGN float rr[8], gg[8], bb[8];
            for (int k = 0; k < 8; ++k) {
                rr[k] = t[k*3 + 0]; gg[k] = t[k*3 + 1]; bb[k] = t[k*3 + 2];
            }
            R = _mm256_load_ps(rr); G = _mm256_load_ps(gg); B = _mm256_load_ps(bb);
        }

        static inline void store_rgb8 (float* dst, __m256 R, __m256 G, __m256 B)
        {
            const __m256i i0 = _mm256_setr_epi32(0,0,0,1,1,1,2,2);
            const __m256i i1 = _mm256_setr_epi32(2,3,3,3,4,4,4,5);
            const __m256i i2 = _mm256_setr_epi32(5,5,6,6,6,7,7,7);
            const __m256 r0 = _mm256_permutevar8x32_ps(R, i0);
            const __m256 g0 = _mm256_permutevar8x32_ps(G, i0);
            const __m256 b0 = _mm256_permutevar8x32_ps(B, i0);
            const __m256 r1 = _mm256_permutevar8x32_ps(R, i1);
            const __m256 g1 = _mm256_permutevar8x32_ps(G, i1);
            const __m256 b1 = _mm256_permutevar8x32_ps(B, i1);
            const __m256 r2 = _mm256_permutevar8x32_ps(R, i2);
            const __m256 g2 = _mm256_permutevar8x32_ps(G, i2);
            const __m256 b2 = _mm256_permutevar8x32_ps(B, i2);
            __m256 o0 = _mm256_blend_ps(r0, g0, 0x92); o0 = _mm256_blend_ps(o0, b0, 0x24);
            __m256 o1 = _mm256_blend_ps(b1, r1, 0x92); o1 = _mm256_blend_ps(o1, g1, 0x24);
            __m256 o2 = _mm256_blend_ps(g2, b2, 0x92); o2 = _mm256_blend_ps(o2, r2, 0x24);
            _mm256_storeu_ps(dst +  0, o0);
            _mm256_storeu_ps(dst +  8, o1);
            _mm256_storeu_ps(dst + 16, o2);
        }
    } // namespace detail

    inline void apply_wb_avx2 (const float* src, float* dst,
                               int32_t sizeX, int32_t sizeY,
                               const double Mfinal[9], bool doClip) noexcept
    {
        const __m256 m0 = _mm256_set1_ps((float)Mfinal[0]);
        const __m256 m1 = _mm256_set1_ps((float)Mfinal[1]);
        const __m256 m2 = _mm256_set1_ps((float)Mfinal[2]);
        const __m256 m3 = _mm256_set1_ps((float)Mfinal[3]);
        const __m256 m4 = _mm256_set1_ps((float)Mfinal[4]);
        const __m256 m5 = _mm256_set1_ps((float)Mfinal[5]);
        const __m256 m6 = _mm256_set1_ps((float)Mfinal[6]);
        const __m256 m7 = _mm256_set1_ps((float)Mfinal[7]);
        const __m256 m8 = _mm256_set1_ps((float)Mfinal[8]);
        const __m256 lo = _mm256_setzero_ps();
        const __m256 hi = _mm256_set1_ps(1.0f);

        const int64_t n = (int64_t)sizeX * (int64_t)sizeY;
        int64_t i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m256 r, g, b;
            detail::load_rgb8(src + i*3, r, g, b);
            __m256 R = _mm256_fmadd_ps(m0, r, _mm256_fmadd_ps(m1, g, _mm256_mul_ps(m2, b)));
            __m256 G = _mm256_fmadd_ps(m3, r, _mm256_fmadd_ps(m4, g, _mm256_mul_ps(m5, b)));
            __m256 B = _mm256_fmadd_ps(m6, r, _mm256_fmadd_ps(m7, g, _mm256_mul_ps(m8, b)));
            if (doClip) {
                R = _mm256_min_ps(_mm256_max_ps(R, lo), hi);
                G = _mm256_min_ps(_mm256_max_ps(G, lo), hi);
                B = _mm256_min_ps(_mm256_max_ps(B, lo), hi);
            }
            detail::store_rgb8(dst + i*3, R, G, B);
        }
        // scalar remainder (< 8 pixels)
        if (i < n) {
            apply_wb_scalar(src + i*3, dst + i*3, (int32_t)(n - i), 1, Mfinal, doClip);
        }
    }
#endif // __AVX2__

    // ------------------------------------------------------- public entry point
    // Prepares the matrix (strength / highlight-safe / clip) and applies it.
    // Uses AVX2 when compiled in; the scalar path is the reference and the
    // fallback. src may alias dst.
    inline void apply_white_balance (const float* src, float* dst,
                                     int32_t sizeX, int32_t sizeY,
                                     const double M_wb[9],
                                     const ApplyParams& params,
                                     bool useAvx2 = true) noexcept
    {
        double Mfinal[9]; bool doClip = true;
        prepare_apply(M_wb, params, Mfinal, doClip);
    #ifdef __AVX2__
        if (useAvx2) { apply_wb_avx2(src, dst, sizeX, sizeY, Mfinal, doClip); return; }
    #else
        (void)useAvx2;
    #endif
        apply_wb_scalar(src, dst, sizeX, sizeY, Mfinal, doClip);
    }

} // namespace AlgoWB

#endif // __IMAGELAB2_APPLY_WB_HPP__
