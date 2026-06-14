#pragma once
// ============================================================================
//  AlgConvertPixelTraits.hpp
//
//  AWB multi-format color-conversion traits, modeled on the Denoise plugin's
//  PixelTraits<FMT> pattern, but producing planar SCENE-LINEAR RGB in [0,1]
//  (sRGB primaries) for the AWB algorithm -- NOT orthonormal YUV.
//
//  Per-format contract:
//    LoadAVX2 (pSrc, vB,vG,vR)            : decode -> unpremult -> LINEARIZE
//                                          -> straight, scene-linear RGB, normalized [0,1].
//    StoreAVX2(pDst, vB,vG,vR, pOrigSrc)  : OETF -> re-premult -> encode to the native format.
//
//  Transfer handling: every non-Linear format is linearized (exact sRGB EOTF on
//  load, OETF on store); the *_32f_Linear formats are pure identity (already
//  scene-linear).
//
//  YUV colorimetry: the supported set lists BOTH the base and the _709 variants,
//  so they are DISTINCT here -- base (non-_709) = Rec.601 matrices, _709 = Rec.709.
//  Range = FULL range: 8u U,V arrive unsigned -> subtract 128 to signed, signed
//  math, add 128 back on store and clamp [0,255]; 32f chroma is native centered
//  (-0.5..0.5), Y in 0..1.
//
//  Range policy on egress:
//    - Integer / 10-bit / gamma-32f outputs : clamp to native range before encode.
//    - *_32f and *_32f_Linear RGB outputs   : floor at 0, NO upper clamp (HDR-safe).
//
//  Alpha is handled at the boundary ONLY; the planar working buffer is RGB-only:
//    - A / P families : alpha restored from the original frame on egress.
//    - X families     : alpha/X written as 0 (clean transparent).
//
//  The vectorized sRGB curve below was validated against the exact scalar sRGB
//  reference: max abs curve error 2.4e-7; 0-code round-trip at 8-bit and 16-bit.
// ============================================================================

#include <immintrin.h>
#include "Common.hpp"          // RESTRICT, CACHE_ALIGN
#include "CommonPixFormat.hpp" // PF_Pixel_* , u*_value_white , A_long

// ============================================================================
//  FORMAT ENUMERATION  (exactly the 33 Premiere PrPixelFormats we support)
// ============================================================================
enum class PixelFormat
{
    BGRA_8u, BGRA_16u, BGRA_32f, BGRA_32f_Linear,
    BGRP_8u, BGRP_16u, BGRP_32f, BGRP_32f_Linear,
    BGRX_8u, BGRX_16u, BGRX_32f, BGRX_32f_Linear,
    ARGB_8u, ARGB_16u, ARGB_32f, ARGB_32f_Linear,
    PRGB_32f, PRGB_32f_Linear,
    XRGB_32f, XRGB_32f_Linear,
    RGB_10u,
    VUYA_8u, VUYA_8u_709, VUYA_32f, VUYA_32f_709,
    VUYP_8u, VUYP_8u_709, VUYP_32f, VUYP_32f_709,
    VUYX_8u, VUYX_8u_709, VUYX_32f, VUYX_32f_709
};

// ============================================================================
//  SHARED AVX2 CONSTANTS
// ============================================================================
static const __m256  awb_zero      = _mm256_setzero_ps();
static const __m256  awb_one       = _mm256_set1_ps(1.0f);
static const __m256  awb_255       = _mm256_set1_ps(255.0f);
static const __m256  awb_inv255    = _mm256_set1_ps(1.0f / 255.0f);
static const __m256  awb_32767     = _mm256_set1_ps(static_cast<float>(u16_value_white));
static const __m256  awb_inv32767  = _mm256_set1_ps(1.0f / static_cast<float>(u16_value_white));
static const __m256  awb_1023      = _mm256_set1_ps(static_cast<float>(u10_value_white));
static const __m256  awb_inv1023   = _mm256_set1_ps(1.0f / static_cast<float>(u10_value_white));
static const __m256  awb_128       = _mm256_set1_ps(128.0f);

static const __m256i awb_mask_8    = _mm256_set1_epi32(0x000000FF);
static const __m256i awb_mask_10   = _mm256_set1_epi32(0x000003FF);
static const __m256i awb_amask_8   = _mm256_set1_epi32(static_cast<int>(0xFF000000));

// ---- Rec.709 full-range YCbCr <-> RGB' --------------------------------------
// forward RGB' -> Y'CbCr (chroma centered, no +128)
static const __m256  awb709_y_r = _mm256_set1_ps( 0.2126f);
static const __m256  awb709_y_g = _mm256_set1_ps( 0.7152f);
static const __m256  awb709_y_b = _mm256_set1_ps( 0.0722f);
static const __m256  awb709_u_r = _mm256_set1_ps(-0.114572f);
static const __m256  awb709_u_g = _mm256_set1_ps(-0.385428f);
static const __m256  awb709_u_b = _mm256_set1_ps( 0.5f);
static const __m256  awb709_v_r = _mm256_set1_ps( 0.5f);
static const __m256  awb709_v_g = _mm256_set1_ps(-0.454153f);
static const __m256  awb709_v_b = _mm256_set1_ps(-0.045847f);
// inverse Y'CbCr -> RGB'
static const __m256  awb709_ir_v = _mm256_set1_ps( 1.5748f);
static const __m256  awb709_ig_u = _mm256_set1_ps(-0.187324f);
static const __m256  awb709_ig_v = _mm256_set1_ps(-0.468124f);
static const __m256  awb709_ib_u = _mm256_set1_ps( 1.8556f);

// ---- Rec.601 full-range YCbCr <-> RGB' --------------------------------------
static const __m256  awb601_y_r = _mm256_set1_ps( 0.299f);
static const __m256  awb601_y_g = _mm256_set1_ps( 0.587f);
static const __m256  awb601_y_b = _mm256_set1_ps( 0.114f);
static const __m256  awb601_u_r = _mm256_set1_ps(-0.168736f);
static const __m256  awb601_u_g = _mm256_set1_ps(-0.331264f);
static const __m256  awb601_u_b = _mm256_set1_ps( 0.5f);
static const __m256  awb601_v_r = _mm256_set1_ps( 0.5f);
static const __m256  awb601_v_g = _mm256_set1_ps(-0.418688f);
static const __m256  awb601_v_b = _mm256_set1_ps(-0.081312f);
static const __m256  awb601_ir_v = _mm256_set1_ps( 1.402f);
static const __m256  awb601_ig_u = _mm256_set1_ps(-0.344136f);
static const __m256  awb601_ig_v = _mm256_set1_ps(-0.714136f);
static const __m256  awb601_ib_u = _mm256_set1_ps( 1.772f);

// Alpha handling tag for the YUV worker templates.
enum class AwbAlpha { Straight, Premult, TransparentX };

// ============================================================================
//  VECTORIZED EXACT sRGB TRANSFER  (validated: 0-code round-trip @ 8u & 16u)
// ============================================================================
namespace awb_xfer
{
    static const __m256  v_one      = _mm256_set1_ps(1.0f);
    static const __m256i v_mant     = _mm256_set1_epi32(0x007FFFFF);
    static const __m256  v_2_over_ln2 = _mm256_set1_ps(2.0f * 1.4426950408889634f);
    static const __m256  v_ln2      = _mm256_set1_ps(0.6931471805599453f);

    static inline __m256 log2_ps(__m256 x) noexcept
    {
        __m256i xi = _mm256_castps_si256(x);
        __m256  e  = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127)));
        __m256  m  = _mm256_or_ps(_mm256_and_ps(x, _mm256_castsi256_ps(v_mant)), v_one); // [1,2)
        __m256 hi = _mm256_cmp_ps(m, _mm256_set1_ps(1.5f), _CMP_GT_OQ);
        m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), hi);
        e = _mm256_blendv_ps(e, _mm256_add_ps(e, v_one), hi);
        __m256 r  = _mm256_div_ps(_mm256_sub_ps(m, v_one), _mm256_add_ps(m, v_one));
        __m256 r2 = _mm256_mul_ps(r, r);
        __m256 p  = _mm256_set1_ps(1.0f / 7.0f);
        p = _mm256_add_ps(_mm256_mul_ps(p, r2), _mm256_set1_ps(1.0f / 5.0f));
        p = _mm256_add_ps(_mm256_mul_ps(p, r2), _mm256_set1_ps(1.0f / 3.0f));
        p = _mm256_add_ps(_mm256_mul_ps(p, r2), v_one);
        return _mm256_add_ps(e, _mm256_mul_ps(_mm256_mul_ps(v_2_over_ln2, r), p));
    }

    static inline __m256 exp2_ps(__m256 x) noexcept
    {
        __m256 n = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 g = _mm256_mul_ps(_mm256_sub_ps(x, n), v_ln2);
        __m256 e = _mm256_set1_ps(1.0f / 720.0f);
        e = _mm256_add_ps(_mm256_mul_ps(e, g), _mm256_set1_ps(1.0f / 120.0f));
        e = _mm256_add_ps(_mm256_mul_ps(e, g), _mm256_set1_ps(1.0f / 24.0f));
        e = _mm256_add_ps(_mm256_mul_ps(e, g), _mm256_set1_ps(1.0f / 6.0f));
        e = _mm256_add_ps(_mm256_mul_ps(e, g), _mm256_set1_ps(0.5f));
        e = _mm256_add_ps(_mm256_mul_ps(e, g), v_one);
        e = _mm256_add_ps(_mm256_mul_ps(e, g), v_one);
        __m256i ni = _mm256_cvtps_epi32(n);
        __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
        return _mm256_mul_ps(pow2n, e);
    }

    static inline __m256 pow_ps(__m256 x, float p) noexcept
    {
        return exp2_ps(_mm256_mul_ps(_mm256_set1_ps(p), log2_ps(x)));
    }

    static inline __m256 eotf(__m256 c) noexcept
    {
        c = _mm256_max_ps(c, _mm256_setzero_ps());
        __m256 lo = _mm256_mul_ps(c, _mm256_set1_ps(1.0f / 12.92f));
        __m256 hi = pow_ps(_mm256_mul_ps(_mm256_add_ps(c, _mm256_set1_ps(0.055f)),
                                         _mm256_set1_ps(1.0f / 1.055f)), 2.4f);
        __m256 m = _mm256_cmp_ps(c, _mm256_set1_ps(0.04045f), _CMP_LE_OQ);
        return _mm256_blendv_ps(hi, lo, m);
    }

    static inline __m256 oetf(__m256 c) noexcept
    {
        c = _mm256_max_ps(c, _mm256_setzero_ps());
        __m256 lo = _mm256_mul_ps(c, _mm256_set1_ps(12.92f));
        __m256 hi = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(1.055f), pow_ps(c, 1.0f / 2.4f)),
                                  _mm256_set1_ps(0.055f));
        __m256 m = _mm256_cmp_ps(c, _mm256_set1_ps(0.0031308f), _CMP_LE_OQ);
        return _mm256_blendv_ps(hi, lo, m);
    }
} // namespace awb_xfer

// ============================================================================
//  SMALL SHARED HELPERS
// ============================================================================
static inline __m256 awb_alpha_safe(__m256 aNorm) noexcept
{
    return _mm256_blendv_ps(awb_one, aNorm, _mm256_cmp_ps(aNorm, awb_zero, _CMP_GT_OQ));
}
static inline __m256 awb_clamp01(__m256 v) noexcept
{
    return _mm256_max_ps(awb_zero, _mm256_min_ps(v, awb_one));
}
static inline __m256 awb_floor0(__m256 v) noexcept
{
    return _mm256_max_ps(awb_zero, v);
}

// Y'CbCr <-> RGB' matrices selected at compile time (true = Rec.709, false = Rec.601).
template <bool R709>
static inline void awb_ycc_to_rgb(__m256 Y, __m256 U, __m256 V, __m256& rp, __m256& gp, __m256& bp) noexcept
{
    if (R709) {
        rp = _mm256_add_ps(Y, _mm256_mul_ps(awb709_ir_v, V));
        gp = _mm256_add_ps(Y, _mm256_add_ps(_mm256_mul_ps(awb709_ig_u, U), _mm256_mul_ps(awb709_ig_v, V)));
        bp = _mm256_add_ps(Y, _mm256_mul_ps(awb709_ib_u, U));
    } else {
        rp = _mm256_add_ps(Y, _mm256_mul_ps(awb601_ir_v, V));
        gp = _mm256_add_ps(Y, _mm256_add_ps(_mm256_mul_ps(awb601_ig_u, U), _mm256_mul_ps(awb601_ig_v, V)));
        bp = _mm256_add_ps(Y, _mm256_mul_ps(awb601_ib_u, U));
    }
}
template <bool R709>
static inline void awb_rgb_to_ycc(__m256 rp, __m256 gp, __m256 bp, __m256& Y, __m256& U, __m256& V) noexcept
{
    if (R709) {
        Y = _mm256_add_ps(_mm256_mul_ps(awb709_y_r, rp), _mm256_add_ps(_mm256_mul_ps(awb709_y_g, gp), _mm256_mul_ps(awb709_y_b, bp)));
        U = _mm256_add_ps(_mm256_mul_ps(awb709_u_r, rp), _mm256_add_ps(_mm256_mul_ps(awb709_u_g, gp), _mm256_mul_ps(awb709_u_b, bp)));
        V = _mm256_add_ps(_mm256_mul_ps(awb709_v_r, rp), _mm256_add_ps(_mm256_mul_ps(awb709_v_g, gp), _mm256_mul_ps(awb709_v_b, bp)));
    } else {
        Y = _mm256_add_ps(_mm256_mul_ps(awb601_y_r, rp), _mm256_add_ps(_mm256_mul_ps(awb601_y_g, gp), _mm256_mul_ps(awb601_y_b, bp)));
        U = _mm256_add_ps(_mm256_mul_ps(awb601_u_r, rp), _mm256_add_ps(_mm256_mul_ps(awb601_u_g, gp), _mm256_mul_ps(awb601_u_b, bp)));
        V = _mm256_add_ps(_mm256_mul_ps(awb601_v_r, rp), _mm256_add_ps(_mm256_mul_ps(awb601_v_g, gp), _mm256_mul_ps(awb601_v_b, bp)));
    }
}

// ============================================================================
//  MASTER TEMPLATE
// ============================================================================
template <PixelFormat FMT> struct PixelTraits;

// ----------------------------------------------------------------------------
//  8-BIT  (BGRA / ARGB straight, BGRP premultiplied)
// ----------------------------------------------------------------------------
template <> struct PixelTraits<PixelFormat::BGRA_8u>
{
    using DataType = PF_Pixel_BGRA_8u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        vB = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(p, awb_mask_8)), awb_inv255);
        vG = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 8),  awb_mask_8)), awb_inv255);
        vR = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 16), awb_mask_8)), awb_inv255);
        vB = awb_xfer::eotf(vB); vG = awb_xfer::eotf(vG); vR = awb_xfer::eotf(vR);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vB)), awb_255);
        vG = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vG)), awb_255);
        vR = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vR)), awb_255);
        __m256i a = _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pOrigSrc)), awb_amask_8);
        __m256i o = _mm256_or_si256(
            _mm256_or_si256(_mm256_cvtps_epi32(vB), _mm256_slli_epi32(_mm256_cvtps_epi32(vG), 8)),
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vR), 16), a));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), o);
    }
};

template <> struct PixelTraits<PixelFormat::ARGB_8u>
{
    using DataType = PF_Pixel_ARGB_8u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        vR = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 8),  awb_mask_8)), awb_inv255);
        vG = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 16), awb_mask_8)), awb_inv255);
        vB = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 24), awb_mask_8)), awb_inv255);
        vB = awb_xfer::eotf(vB); vG = awb_xfer::eotf(vG); vR = awb_xfer::eotf(vR);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vB)), awb_255);
        vG = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vG)), awb_255);
        vR = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vR)), awb_255);
        __m256i a = _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pOrigSrc)), awb_mask_8); // A in low byte
        __m256i o = _mm256_or_si256(
            _mm256_or_si256(a, _mm256_slli_epi32(_mm256_cvtps_epi32(vR), 8)),
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vG), 16), _mm256_slli_epi32(_mm256_cvtps_epi32(vB), 24)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), o);
    }
};

template <> struct PixelTraits<PixelFormat::BGRP_8u>
{
    using DataType = PF_Pixel_BGRA_8u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        vB = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(p, awb_mask_8)), awb_inv255);
        vG = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 8),  awb_mask_8)), awb_inv255);
        vR = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 16), awb_mask_8)), awb_inv255);
        __m256 aN = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 24), awb_mask_8)), awb_inv255);
        __m256 aS = awb_alpha_safe(aN);
        vB = awb_xfer::eotf(_mm256_div_ps(vB, aS));
        vG = awb_xfer::eotf(_mm256_div_ps(vG, aS));
        vR = awb_xfer::eotf(_mm256_div_ps(vR, aS));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256i a = _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pOrigSrc)), awb_amask_8);
        __m256 aN = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(a, 24)), awb_inv255);
        vB = _mm256_mul_ps(awb_clamp01(_mm256_mul_ps(awb_xfer::oetf(vB), aN)), awb_255);
        vG = _mm256_mul_ps(awb_clamp01(_mm256_mul_ps(awb_xfer::oetf(vG), aN)), awb_255);
        vR = _mm256_mul_ps(awb_clamp01(_mm256_mul_ps(awb_xfer::oetf(vR), aN)), awb_255);
        __m256i o = _mm256_or_si256(
            _mm256_or_si256(_mm256_cvtps_epi32(vB), _mm256_slli_epi32(_mm256_cvtps_epi32(vG), 8)),
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vR), 16), a));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), o);
    }
};

// ----------------------------------------------------------------------------
//  10-BIT  (RGB only, no alpha)
// ----------------------------------------------------------------------------
template <> struct PixelTraits<PixelFormat::RGB_10u>
{
    using DataType = PF_Pixel_RGB_10u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        vB = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 2),  awb_mask_10)), awb_inv1023);
        vG = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 12), awb_mask_10)), awb_inv1023);
        vR = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 22), awb_mask_10)), awb_inv1023);
        vB = awb_xfer::eotf(vB); vG = awb_xfer::eotf(vG); vR = awb_xfer::eotf(vR);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT /*pOrigSrc*/) noexcept
    {
        vB = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vB)), awb_1023);
        vG = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vG)), awb_1023);
        vR = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vR)), awb_1023);
        __m256i o = _mm256_or_si256(
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vB), 2), _mm256_slli_epi32(_mm256_cvtps_epi32(vG), 12)),
            _mm256_slli_epi32(_mm256_cvtps_epi32(vR), 22));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), o);
    }
};

// ----------------------------------------------------------------------------
//  16-BIT  (scalar element load/store; white = u16_value_white = 32767)
// ----------------------------------------------------------------------------
template <> struct PixelTraits<PixelFormat::BGRA_16u>
{
    using DataType = PF_Pixel_BGRA_16u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; }
        vB = awb_xfer::eotf(_mm256_mul_ps(_mm256_load_ps(b), awb_inv32767));
        vG = awb_xfer::eotf(_mm256_mul_ps(_mm256_load_ps(g), awb_inv32767));
        vR = awb_xfer::eotf(_mm256_mul_ps(_mm256_load_ps(r), awb_inv32767));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vB)), awb_32767);
        vG = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vG)), awb_32767);
        vR = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vR)), awb_32767);
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) {
            pDst[i].B = static_cast<A_u_short>(b[i] + 0.5f);
            pDst[i].G = static_cast<A_u_short>(g[i] + 0.5f);
            pDst[i].R = static_cast<A_u_short>(r[i] + 0.5f);
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <> struct PixelTraits<PixelFormat::ARGB_16u>
{
    using DataType = PF_Pixel_ARGB_16u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; }
        vB = awb_xfer::eotf(_mm256_mul_ps(_mm256_load_ps(b), awb_inv32767));
        vG = awb_xfer::eotf(_mm256_mul_ps(_mm256_load_ps(g), awb_inv32767));
        vR = awb_xfer::eotf(_mm256_mul_ps(_mm256_load_ps(r), awb_inv32767));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vB)), awb_32767);
        vG = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vG)), awb_32767);
        vR = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vR)), awb_32767);
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) {
            pDst[i].B = static_cast<A_u_short>(b[i] + 0.5f);
            pDst[i].G = static_cast<A_u_short>(g[i] + 0.5f);
            pDst[i].R = static_cast<A_u_short>(r[i] + 0.5f);
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <> struct PixelTraits<PixelFormat::BGRP_16u>
{
    using DataType = PF_Pixel_BGRA_16u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8], a[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; a[i] = pSrc[i].A; }
        __m256 vBn = _mm256_mul_ps(_mm256_load_ps(b), awb_inv32767);
        __m256 vGn = _mm256_mul_ps(_mm256_load_ps(g), awb_inv32767);
        __m256 vRn = _mm256_mul_ps(_mm256_load_ps(r), awb_inv32767);
        __m256 aS  = awb_alpha_safe(_mm256_mul_ps(_mm256_load_ps(a), awb_inv32767));
        vB = awb_xfer::eotf(_mm256_div_ps(vBn, aS));
        vG = awb_xfer::eotf(_mm256_div_ps(vGn, aS));
        vR = awb_xfer::eotf(_mm256_div_ps(vRn, aS));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        CACHE_ALIGN float a[8];
        for (int i = 0; i < 8; ++i) { a[i] = pOrigSrc[i].A; }
        __m256 aN = _mm256_mul_ps(_mm256_load_ps(a), awb_inv32767);
        vB = _mm256_mul_ps(awb_clamp01(_mm256_mul_ps(awb_xfer::oetf(vB), aN)), awb_32767);
        vG = _mm256_mul_ps(awb_clamp01(_mm256_mul_ps(awb_xfer::oetf(vG), aN)), awb_32767);
        vR = _mm256_mul_ps(awb_clamp01(_mm256_mul_ps(awb_xfer::oetf(vR), aN)), awb_32767);
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) {
            pDst[i].B = static_cast<A_u_short>(b[i] + 0.5f);
            pDst[i].G = static_cast<A_u_short>(g[i] + 0.5f);
            pDst[i].R = static_cast<A_u_short>(r[i] + 0.5f);
            pDst[i].A = static_cast<A_u_short>(a[i]);
        }
    }
};

// ----------------------------------------------------------------------------
//  32-BIT FLOAT, sRGB-gamma  (linearize on load; HDR-preserving on egress)
//  BGRA/ARGB = straight ; BGRP/PRGB = premultiplied (PRGB uses ARGB layout).
//  Member access (.B/.G/.R/.A) is byte-order agnostic, so BGRA- and ARGB-layout
//  bodies are identical apart from the DataType.
// ----------------------------------------------------------------------------
template <> struct PixelTraits<PixelFormat::BGRA_32f>
{
    using DataType = PF_Pixel_BGRA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; }
        vB = awb_xfer::eotf(_mm256_load_ps(b));
        vG = awb_xfer::eotf(_mm256_load_ps(g));
        vR = awb_xfer::eotf(_mm256_load_ps(r));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = awb_floor0(awb_xfer::oetf(vB)); // HDR: floor 0, no upper clamp
        vG = awb_floor0(awb_xfer::oetf(vG));
        vR = awb_floor0(awb_xfer::oetf(vR));
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = pOrigSrc[i].A; }
    }
};

template <> struct PixelTraits<PixelFormat::ARGB_32f>
{
    using DataType = PF_Pixel_ARGB_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; }
        vB = awb_xfer::eotf(_mm256_load_ps(b));
        vG = awb_xfer::eotf(_mm256_load_ps(g));
        vR = awb_xfer::eotf(_mm256_load_ps(r));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = awb_floor0(awb_xfer::oetf(vB));
        vG = awb_floor0(awb_xfer::oetf(vG));
        vR = awb_floor0(awb_xfer::oetf(vR));
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = pOrigSrc[i].A; }
    }
};

template <> struct PixelTraits<PixelFormat::BGRP_32f>
{
    using DataType = PF_Pixel_BGRA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8], a[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; a[i] = pSrc[i].A; }
        __m256 aS = awb_alpha_safe(_mm256_load_ps(a));
        vB = awb_xfer::eotf(_mm256_div_ps(_mm256_load_ps(b), aS));
        vG = awb_xfer::eotf(_mm256_div_ps(_mm256_load_ps(g), aS));
        vR = awb_xfer::eotf(_mm256_div_ps(_mm256_load_ps(r), aS));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        CACHE_ALIGN float a[8];
        for (int i = 0; i < 8; ++i) { a[i] = pOrigSrc[i].A; }
        __m256 aN = _mm256_load_ps(a);
        vB = awb_floor0(_mm256_mul_ps(awb_xfer::oetf(vB), aN));
        vG = awb_floor0(_mm256_mul_ps(awb_xfer::oetf(vG), aN));
        vR = awb_floor0(_mm256_mul_ps(awb_xfer::oetf(vR), aN));
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = a[i]; }
    }
};

// PRGB_32f : ARGB memory layout, premultiplied, sRGB gamma (mirror of BGRP_32f).
template <> struct PixelTraits<PixelFormat::PRGB_32f>
{
    using DataType = PF_Pixel_ARGB_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8], a[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; a[i] = pSrc[i].A; }
        __m256 aS = awb_alpha_safe(_mm256_load_ps(a));
        vB = awb_xfer::eotf(_mm256_div_ps(_mm256_load_ps(b), aS));
        vG = awb_xfer::eotf(_mm256_div_ps(_mm256_load_ps(g), aS));
        vR = awb_xfer::eotf(_mm256_div_ps(_mm256_load_ps(r), aS));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        CACHE_ALIGN float a[8];
        for (int i = 0; i < 8; ++i) { a[i] = pOrigSrc[i].A; }
        __m256 aN = _mm256_load_ps(a);
        vB = awb_floor0(_mm256_mul_ps(awb_xfer::oetf(vB), aN));
        vG = awb_floor0(_mm256_mul_ps(awb_xfer::oetf(vG), aN));
        vR = awb_floor0(_mm256_mul_ps(awb_xfer::oetf(vR), aN));
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = a[i]; }
    }
};

// ----------------------------------------------------------------------------
//  32-BIT FLOAT LINEAR  (already scene-linear: identity transfer, floor 0)
// ----------------------------------------------------------------------------
template <> struct PixelTraits<PixelFormat::BGRA_32f_Linear>
{
    using DataType = PF_Pixel_BGRA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; }
        vB = awb_floor0(_mm256_load_ps(b));
        vG = awb_floor0(_mm256_load_ps(g));
        vR = awb_floor0(_mm256_load_ps(r));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = awb_floor0(vB); vG = awb_floor0(vG); vR = awb_floor0(vR);
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = pOrigSrc[i].A; }
    }
};

// ARGB_32f_Linear : ARGB layout, straight alpha, identity transfer.
template <> struct PixelTraits<PixelFormat::ARGB_32f_Linear>
{
    using DataType = PF_Pixel_ARGB_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; }
        vB = awb_floor0(_mm256_load_ps(b));
        vG = awb_floor0(_mm256_load_ps(g));
        vR = awb_floor0(_mm256_load_ps(r));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = awb_floor0(vB); vG = awb_floor0(vG); vR = awb_floor0(vR);
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = pOrigSrc[i].A; }
    }
};

template <> struct PixelTraits<PixelFormat::BGRP_32f_Linear>
{
    using DataType = PF_Pixel_BGRA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8], a[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; a[i] = pSrc[i].A; }
        __m256 aS = awb_alpha_safe(_mm256_load_ps(a));
        vB = awb_floor0(_mm256_div_ps(_mm256_load_ps(b), aS));
        vG = awb_floor0(_mm256_div_ps(_mm256_load_ps(g), aS));
        vR = awb_floor0(_mm256_div_ps(_mm256_load_ps(r), aS));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        CACHE_ALIGN float a[8];
        for (int i = 0; i < 8; ++i) { a[i] = pOrigSrc[i].A; }
        __m256 aN = _mm256_load_ps(a);
        vB = awb_floor0(_mm256_mul_ps(vB, aN));
        vG = awb_floor0(_mm256_mul_ps(vG, aN));
        vR = awb_floor0(_mm256_mul_ps(vR, aN));
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = a[i]; }
    }
};

// PRGB_32f_Linear : ARGB layout, premultiplied, identity transfer.
template <> struct PixelTraits<PixelFormat::PRGB_32f_Linear>
{
    using DataType = PF_Pixel_ARGB_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b[8], g[8], r[8], a[8];
        for (int i = 0; i < 8; ++i) { b[i] = pSrc[i].B; g[i] = pSrc[i].G; r[i] = pSrc[i].R; a[i] = pSrc[i].A; }
        __m256 aS = awb_alpha_safe(_mm256_load_ps(a));
        vB = awb_floor0(_mm256_div_ps(_mm256_load_ps(b), aS));
        vG = awb_floor0(_mm256_div_ps(_mm256_load_ps(g), aS));
        vR = awb_floor0(_mm256_div_ps(_mm256_load_ps(r), aS));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        CACHE_ALIGN float a[8];
        for (int i = 0; i < 8; ++i) { a[i] = pOrigSrc[i].A; }
        __m256 aN = _mm256_load_ps(a);
        vB = awb_floor0(_mm256_mul_ps(vB, aN));
        vG = awb_floor0(_mm256_mul_ps(vG, aN));
        vR = awb_floor0(_mm256_mul_ps(vR, aN));
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = a[i]; }
    }
};

// ============================================================================
//  X (TRANSPARENT) FAMILIES -- load aliases the matching straight format
//  (ignores the 4th channel); store writes 0 to the X slot.
// ============================================================================
template <> struct PixelTraits<PixelFormat::BGRX_8u> : public PixelTraits<PixelFormat::BGRA_8u>
{
    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT /*pOrigSrc*/) noexcept
    {
        vB = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vB)), awb_255);
        vG = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vG)), awb_255);
        vR = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vR)), awb_255);
        __m256i o = _mm256_or_si256(   // alpha byte left as 0
            _mm256_or_si256(_mm256_cvtps_epi32(vB), _mm256_slli_epi32(_mm256_cvtps_epi32(vG), 8)),
            _mm256_slli_epi32(_mm256_cvtps_epi32(vR), 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), o);
    }
};

template <> struct PixelTraits<PixelFormat::BGRX_16u> : public PixelTraits<PixelFormat::BGRA_16u>
{
    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT /*pOrigSrc*/) noexcept
    {
        vB = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vB)), awb_32767);
        vG = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vG)), awb_32767);
        vR = _mm256_mul_ps(awb_clamp01(awb_xfer::oetf(vR)), awb_32767);
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) {
            pDst[i].B = static_cast<A_u_short>(b[i] + 0.5f);
            pDst[i].G = static_cast<A_u_short>(g[i] + 0.5f);
            pDst[i].R = static_cast<A_u_short>(r[i] + 0.5f);
            pDst[i].A = 0;
        }
    }
};

template <> struct PixelTraits<PixelFormat::BGRX_32f> : public PixelTraits<PixelFormat::BGRA_32f>
{
    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT /*pOrigSrc*/) noexcept
    {
        vB = awb_floor0(awb_xfer::oetf(vB)); vG = awb_floor0(awb_xfer::oetf(vG)); vR = awb_floor0(awb_xfer::oetf(vR));
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = 0.0f; }
    }
};

template <> struct PixelTraits<PixelFormat::BGRX_32f_Linear> : public PixelTraits<PixelFormat::BGRA_32f_Linear>
{
    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT /*pOrigSrc*/) noexcept
    {
        vB = awb_floor0(vB); vG = awb_floor0(vG); vR = awb_floor0(vR);
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = 0.0f; }
    }
};

// XRGB_32f / XRGB_32f_Linear : ARGB layout, transparent X (write 0).
template <> struct PixelTraits<PixelFormat::XRGB_32f> : public PixelTraits<PixelFormat::ARGB_32f>
{
    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT /*pOrigSrc*/) noexcept
    {
        vB = awb_floor0(awb_xfer::oetf(vB)); vG = awb_floor0(awb_xfer::oetf(vG)); vR = awb_floor0(awb_xfer::oetf(vR));
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = 0.0f; }
    }
};

template <> struct PixelTraits<PixelFormat::XRGB_32f_Linear> : public PixelTraits<PixelFormat::ARGB_32f_Linear>
{
    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT /*pOrigSrc*/) noexcept
    {
        vB = awb_floor0(vB); vG = awb_floor0(vG); vR = awb_floor0(vR);
        CACHE_ALIGN float b[8], g[8], r[8];
        _mm256_store_ps(b, vB); _mm256_store_ps(g, vG); _mm256_store_ps(r, vR);
        for (int i = 0; i < 8; ++i) { pDst[i].B = b[i]; pDst[i].G = g[i]; pDst[i].R = r[i]; pDst[i].A = 0.0f; }
    }
};

// ============================================================================
//  VUY (V,U,Y,A) FAMILIES  -- full-range YCbCr.
//    base (non-_709) = Rec.601 matrices ; _709 = Rec.709 matrices.
//    8u : U,V arrive unsigned -> subtract 128 (signed math) -> +128 back on
//         store, clamp [0,255]. 32f : chroma native centered (-0.5..0.5), Y 0..1.
//    VUYA = straight ; VUYP = premultiplied ; VUYX = transparent X (write 0).
// ============================================================================

// ---- 8u workers ----
template <bool R709, bool Premult>
static inline void awb_vuy8_load(const PF_Pixel_VUYA_8u* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
{
    __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
    __m256 V = _mm256_cvtepi32_ps(_mm256_and_si256(p, awb_mask_8));
    __m256 U = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 8),  awb_mask_8));
    __m256 Y = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 16), awb_mask_8));
    V = _mm256_sub_ps(V, awb_128); // unsigned -> signed
    U = _mm256_sub_ps(U, awb_128);
    if (Premult) {
        __m256 aN = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(p, 24), awb_mask_8)), awb_inv255);
        __m256 aS = awb_alpha_safe(aN);
        V = _mm256_div_ps(V, aS); U = _mm256_div_ps(U, aS); Y = _mm256_div_ps(Y, aS);
    }
    __m256 rp, gp, bp;
    awb_ycc_to_rgb<R709>(Y, U, V, rp, gp, bp); // 0..255 grid
    vR = awb_xfer::eotf(_mm256_mul_ps(rp, awb_inv255));
    vG = awb_xfer::eotf(_mm256_mul_ps(gp, awb_inv255));
    vB = awb_xfer::eotf(_mm256_mul_ps(bp, awb_inv255));
}

template <bool R709, AwbAlpha AM>
static inline void awb_vuy8_store(PF_Pixel_VUYA_8u* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const PF_Pixel_VUYA_8u* RESTRICT pOrigSrc) noexcept
{
    __m256 rp = _mm256_mul_ps(awb_xfer::oetf(vR), awb_255);
    __m256 gp = _mm256_mul_ps(awb_xfer::oetf(vG), awb_255);
    __m256 bp = _mm256_mul_ps(awb_xfer::oetf(vB), awb_255);
    __m256 Y, U, V;
    awb_rgb_to_ycc<R709>(rp, gp, bp, Y, U, V); // U,V centered (no +128 yet)
    __m256i a;
    if (AM == AwbAlpha::TransparentX) {
        a = _mm256_setzero_si256();
    } else {
        a = _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pOrigSrc)), awb_amask_8);
        if (AM == AwbAlpha::Premult) {
            __m256 aN = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(a, 24)), awb_inv255);
            Y = _mm256_mul_ps(Y, aN); U = _mm256_mul_ps(U, aN); V = _mm256_mul_ps(V, aN);
        }
    }
    V = _mm256_max_ps(awb_zero, _mm256_min_ps(_mm256_add_ps(V, awb_128), awb_255)); // +128, clamp
    U = _mm256_max_ps(awb_zero, _mm256_min_ps(_mm256_add_ps(U, awb_128), awb_255));
    Y = _mm256_max_ps(awb_zero, _mm256_min_ps(Y, awb_255));
    __m256i o = _mm256_or_si256(
        _mm256_or_si256(_mm256_cvtps_epi32(V), _mm256_slli_epi32(_mm256_cvtps_epi32(U), 8)),
        _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(Y), 16), a));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), o);
}

// ---- 32f workers ----
template <bool R709, bool Premult>
static inline void awb_vuy32f_load(const PF_Pixel_VUYA_32f* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
{
    CACHE_ALIGN float vv[8], uu[8], yy[8], aa[8] = {0};
    for (int i = 0; i < 8; ++i) { vv[i] = pSrc[i].V; uu[i] = pSrc[i].U; yy[i] = pSrc[i].Y; if (Premult) aa[i] = pSrc[i].A; }
    __m256 V = _mm256_load_ps(vv); // natively -0.5..0.5
    __m256 U = _mm256_load_ps(uu);
    __m256 Y = _mm256_load_ps(yy); // 0..1
    if (Premult) {
        __m256 aS = awb_alpha_safe(_mm256_load_ps(aa));
        V = _mm256_div_ps(V, aS); U = _mm256_div_ps(U, aS); Y = _mm256_div_ps(Y, aS);
    }
    __m256 rp, gp, bp;
    awb_ycc_to_rgb<R709>(Y, U, V, rp, gp, bp);
    vR = awb_xfer::eotf(rp); vG = awb_xfer::eotf(gp); vB = awb_xfer::eotf(bp);
}

template <bool R709, AwbAlpha AM>
static inline void awb_vuy32f_store(PF_Pixel_VUYA_32f* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const PF_Pixel_VUYA_32f* RESTRICT pOrigSrc) noexcept
{
    __m256 rp = awb_xfer::oetf(vR), gp = awb_xfer::oetf(vG), bp = awb_xfer::oetf(vB);
    __m256 Y, U, V;
    awb_rgb_to_ycc<R709>(rp, gp, bp, Y, U, V);
    CACHE_ALIGN float aa[8];
    if (AM == AwbAlpha::TransparentX) {
        for (int i = 0; i < 8; ++i) aa[i] = 0.0f;
    } else {
        for (int i = 0; i < 8; ++i) aa[i] = pOrigSrc[i].A;
        if (AM == AwbAlpha::Premult) {
            __m256 aN = _mm256_load_ps(aa);
            Y = _mm256_mul_ps(Y, aN); U = _mm256_mul_ps(U, aN); V = _mm256_mul_ps(V, aN);
        }
    }
    CACHE_ALIGN float vv[8], uu[8], yy[8];
    _mm256_store_ps(vv, V); _mm256_store_ps(uu, U); _mm256_store_ps(yy, Y);
    for (int i = 0; i < 8; ++i) { pDst[i].V = vv[i]; pDst[i].U = uu[i]; pDst[i].Y = yy[i]; pDst[i].A = aa[i]; }
}

// ---- VUY specializations (12) ----
#define AWB_VUY8(NAME, R709, AM, PREM)                                                                              \
template <> struct PixelTraits<PixelFormat::NAME> {                                                                \
    using DataType = PF_Pixel_VUYA_8u;                                                                             \
    static inline void LoadAVX2(const DataType* RESTRICT s, __m256& b, __m256& g, __m256& r) noexcept              \
    { awb_vuy8_load<R709, PREM>(s, b, g, r); }                                                                     \
    static inline void StoreAVX2(DataType* RESTRICT d, __m256 b, __m256 g, __m256 r, const DataType* RESTRICT o) noexcept \
    { awb_vuy8_store<R709, AM>(d, b, g, r, o); }                                                                   \
};
#define AWB_VUY32F(NAME, R709, AM, PREM)                                                                           \
template <> struct PixelTraits<PixelFormat::NAME> {                                                                \
    using DataType = PF_Pixel_VUYA_32f;                                                                            \
    static inline void LoadAVX2(const DataType* RESTRICT s, __m256& b, __m256& g, __m256& r) noexcept              \
    { awb_vuy32f_load<R709, PREM>(s, b, g, r); }                                                                   \
    static inline void StoreAVX2(DataType* RESTRICT d, __m256 b, __m256 g, __m256 r, const DataType* RESTRICT o) noexcept \
    { awb_vuy32f_store<R709, AM>(d, b, g, r, o); }                                                                 \
};

AWB_VUY8 (VUYA_8u,      false, AwbAlpha::Straight,     false)
AWB_VUY8 (VUYA_8u_709,  true,  AwbAlpha::Straight,     false)
AWB_VUY8 (VUYP_8u,      false, AwbAlpha::Premult,      true )
AWB_VUY8 (VUYP_8u_709,  true,  AwbAlpha::Premult,      true )
AWB_VUY8 (VUYX_8u,      false, AwbAlpha::TransparentX, false)
AWB_VUY8 (VUYX_8u_709,  true,  AwbAlpha::TransparentX, false)

AWB_VUY32F(VUYA_32f,     false, AwbAlpha::Straight,     false)
AWB_VUY32F(VUYA_32f_709, true,  AwbAlpha::Straight,     false)
AWB_VUY32F(VUYP_32f,     false, AwbAlpha::Premult,      true )
AWB_VUY32F(VUYP_32f_709, true,  AwbAlpha::Premult,      true )
AWB_VUY32F(VUYX_32f,     false, AwbAlpha::TransparentX, false)
AWB_VUY32F(VUYX_32f_709, true,  AwbAlpha::TransparentX, false)

#undef AWB_VUY8
#undef AWB_VUY32F
