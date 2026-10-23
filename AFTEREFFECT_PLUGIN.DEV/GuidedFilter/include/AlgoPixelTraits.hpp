#pragma once
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include "CommonPixFormat.hpp" 
#include "FastAriphmetics.hpp"

// ============================================================================
// FORMAT ENUMERATION 
// ============================================================================
enum class PixelFormat
{
    BGRA_8u, 
    BGRX_8u, 
    BGRP_8u, 
    ARGB_8u,
    BGRA_16u, 
    BGRX_16u, 
    BGRP_16u, 
    ARGB_16u,
    BGRA_32f, 
    BGRX_32f, 
    BGRP_32f, 
    ARGB_32f,
    BGRA_32f_Linear, 
    BGRP_32f_Linear, 
    BGRX_32f_Linear,
    VUYA_8u, 
    VUYP_8u,
    VUYA_16u, 
    VUYP_16u,
    VUYA_32f, 
    VUYP_32f,
    VUYA_8u_709, 
    VUYP_8u_709,
    VUYA_16u_709, 
    VUYP_16u_709,
    VUYA_32f_709, 
    VUYP_32f_709,
    RGB_10u,
    // --- added 2026-10-02 (F4) ---
    VUYX_8u,
    VUYX_8u_709,
    VUYX_32f,
    VUYX_32f_709,
    ARGB_32f_Linear,
    PRGB_8u,
    PRGB_16u,
    PRGB_32f,
    PRGB_32f_Linear,
    XRGB_8u,
    XRGB_16u,
    XRGB_32f,
    XRGB_32f_Linear
};

// ============================================================================
// GLOBAL AVX2 CONSTANTS & COLOR SCIENCE
// ============================================================================
// Up-Scales (Adobe to Engine)
static const __m256 v_scale_10_to_8 = _mm256_set1_ps(255.0f / static_cast<float>(u10_value_white));
static const __m256 v_scale_16_to_8 = _mm256_set1_ps(255.0f / static_cast<float>(u16_value_white));
static const __m256 v_scale_32_to_8 = _mm256_set1_ps(255.0f);

// Down-Scales (Engine back to Adobe)
static const __m256 v_scale_8_to_10 = _mm256_set1_ps(static_cast<float>(u10_value_white) / 255.0f);
static const __m256 v_scale_8_to_16 = _mm256_set1_ps(static_cast<float>(u16_value_white) / 255.0f);
static const __m256 v_scale_8_to_32 = _mm256_set1_ps(1.0f / 255.0f);

// Extraction Masks
static const __m256i v_mask_8bit = _mm256_set1_epi32(0x000000FF);
static const __m256i v_alpha_mask_8bit = _mm256_set1_epi32(static_cast<int>(0xFF000000));
static const __m256i v_mask_10bit = _mm256_set1_epi32(0x000003FF);

// Alpha Normalization
static const __m256 v_alpha_norm_8 = _mm256_set1_ps(1.0f / 255.0f);
static const __m256 v_alpha_norm_16 = _mm256_set1_ps(1.0f / static_cast<float>(u16_value_white));

// Math Constants
static const __m256 v_zero = _mm256_setzero_ps();
static const __m256 v_one = _mm256_set1_ps(1.0f);
static const __m256 v_128 = _mm256_set1_ps(128.0f); 
static const __m256 v_255 = _mm256_set1_ps(255.0f);
static const __m256 v_32767 = _mm256_set1_ps(static_cast<float>(u16_value_white));
static const __m256 v_1023 = _mm256_set1_ps(static_cast<float>(u10_value_white));

// Rec.709 Coefficients (For decoding/encoding Adobe's native YUV formats in the Traits)
static const __m256 v_y_r = _mm256_set1_ps(0.2126f);
static const __m256 v_y_g = _mm256_set1_ps(0.7152f);
static const __m256 v_y_b = _mm256_set1_ps(0.0722f);
static const __m256 v_u_r = _mm256_set1_ps(-0.114572f);
static const __m256 v_u_g = _mm256_set1_ps(-0.385428f);
static const __m256 v_u_b = _mm256_set1_ps(0.5f);
static const __m256 v_v_r = _mm256_set1_ps(0.5f);
static const __m256 v_v_g = _mm256_set1_ps(-0.454153f);
static const __m256 v_v_b = _mm256_set1_ps(-0.045847f);

// Inverse Rec.709
static const __m256 v_inv_r_v = _mm256_set1_ps(1.5748f);
static const __m256 v_inv_g_u = _mm256_set1_ps(-0.187324f);
static const __m256 v_inv_g_v = _mm256_set1_ps(-0.468124f);
static const __m256 v_inv_b_u = _mm256_set1_ps(1.8556f);

// Rec.601 Coefficients (Adobe formats WITHOUT the _709 suffix), full range, chroma centred on 128 (8/16u) or 0 (32f)
static const __m256 v601_y_r = _mm256_set1_ps(0.299f);
static const __m256 v601_y_g = _mm256_set1_ps(0.587f);
static const __m256 v601_y_b = _mm256_set1_ps(0.114f);
static const __m256 v601_u_r = _mm256_set1_ps(-0.168736f);
static const __m256 v601_u_g = _mm256_set1_ps(-0.331264f);
static const __m256 v601_u_b = _mm256_set1_ps(0.5f);
static const __m256 v601_v_r = _mm256_set1_ps(0.5f);
static const __m256 v601_v_g = _mm256_set1_ps(-0.418688f);
static const __m256 v601_v_b = _mm256_set1_ps(-0.081312f);

// Inverse Rec.601
static const __m256 v601_inv_r_v = _mm256_set1_ps(1.402f);
static const __m256 v601_inv_g_u = _mm256_set1_ps(-0.344136f);
static const __m256 v601_inv_g_v = _mm256_set1_ps(-0.714136f);
static const __m256 v601_inv_b_u = _mm256_set1_ps(1.772f);

// YUV matrix selectors used by the VUYx traits templates
struct YuvCoef709
{
    static inline __m256 y_r() noexcept { return v_y_r; }   static inline __m256 y_g() noexcept { return v_y_g; }   static inline __m256 y_b() noexcept { return v_y_b; }
    static inline __m256 u_r() noexcept { return v_u_r; }   static inline __m256 u_g() noexcept { return v_u_g; }   static inline __m256 u_b() noexcept { return v_u_b; }
    static inline __m256 v_r() noexcept { return v_v_r; }   static inline __m256 v_g() noexcept { return v_v_g; }   static inline __m256 v_b() noexcept { return v_v_b; }
    static inline __m256 inv_r_v() noexcept { return v_inv_r_v; } static inline __m256 inv_g_u() noexcept { return v_inv_g_u; }
    static inline __m256 inv_g_v() noexcept { return v_inv_g_v; } static inline __m256 inv_b_u() noexcept { return v_inv_b_u; }
};
struct YuvCoef601
{
    static inline __m256 y_r() noexcept { return v601_y_r; } static inline __m256 y_g() noexcept { return v601_y_g; } static inline __m256 y_b() noexcept { return v601_y_b; }
    static inline __m256 u_r() noexcept { return v601_u_r; } static inline __m256 u_g() noexcept { return v601_u_g; } static inline __m256 u_b() noexcept { return v601_u_b; }
    static inline __m256 v_r() noexcept { return v601_v_r; } static inline __m256 v_g() noexcept { return v601_v_g; } static inline __m256 v_b() noexcept { return v601_v_b; }
    static inline __m256 inv_r_v() noexcept { return v601_inv_r_v; } static inline __m256 inv_g_u() noexcept { return v601_inv_g_u; }
    static inline __m256 inv_g_v() noexcept { return v601_inv_g_v; } static inline __m256 inv_b_u() noexcept { return v601_inv_b_u; }
};

// ============================================================================
// 32f RANGE POLICY
// _KEEP_CLAMP_TO_1 == 0 (default): over-range (HDR / negative) 32f values pass through
//                                  (AE/Pr 32f is "1.0 is white", not clamped).
// _KEEP_CLAMP_TO_1 != 0          : 32f results are clamped to [0 .. 1.0f - FLT_EPSILON] (SDR,
//                                  f32_value_white from CommonPixFormat.hpp).
// Override from the build, e.g. -D_KEEP_CLAMP_TO_1=1 (GCC) or /D_KEEP_CLAMP_TO_1=1 (MSVC).
// ============================================================================
#ifndef _KEEP_CLAMP_TO_1
 #define _KEEP_CLAMP_TO_1 0
#endif

static const __m256 v_32f_white = _mm256_set1_ps(f32_value_white);   // 1.0f - FLT_EPSILON

inline __m256 Clamp01_32f (const __m256 v) noexcept
{
#if (_KEEP_CLAMP_TO_1 != 0)
    return _mm256_max_ps(v_zero, _mm256_min_ps(v, v_32f_white));
#else
    return v;
#endif
}

inline __m256 ClampMax1_32f (const __m256 v) noexcept
{
#if (_KEEP_CLAMP_TO_1 != 0)
    return _mm256_min_ps(v, v_32f_white);
#else
    return v;
#endif
}

// ============================================================================
// INLINE HELPERS
// ============================================================================
// Safe, cross-compiler 256-bit Gamma approximation for Linear Light Sandwiches.
// (Replace with _mm256_pow_ps if linking Intel SVML)
inline __m256 ApplyGammaAVX2(__m256 v, float exponent) noexcept
{
    CACHE_ALIGN float arr[8];
    _mm256_store_ps(arr, v);
    for (int i = 0; i < 8; ++i) {
#if (_KEEP_CLAMP_TO_1 != 0)
        arr[i] = FastCompute::Pow(std::max(0.0f, arr[i]), exponent);
#else
        // pass-through mode: sign-preserving power, so negative values survive the round trip
        const float mag = std::fabs(arr[i]);
        const float p   = (mag > 0.0f) ? FastCompute::Pow(mag, exponent) : 0.0f;
        arr[i] = (arr[i] < 0.0f) ? -p : p;
#endif
    }
    return _mm256_load_ps(arr);
}

// ============================================================================
// THE MASTER TRAIT TEMPLATE
// ============================================================================
// ALL LoadAVX2 functions now return un-premultiplied, perceptual RGB (vB, vG, vR)
// The Core algorithm dispatcher will ONLY deal with RGB -> Orthonormal YUV.
template <PixelFormat FMT>
struct PixelTraits;

// ============================================================================
// 8-BIT SPECIALIZATIONS
// ============================================================================
template <> struct PixelTraits<PixelFormat::BGRA_8u>
{
    using DataType = PF_Pixel_BGRA_8u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i v_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        vB = _mm256_cvtepi32_ps(_mm256_and_si256(v_pixels, v_mask_8bit));
        vG = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 8), v_mask_8bit));
        vR = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 16), v_mask_8bit));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_max_ps(v_zero, _mm256_min_ps(vB, v_255));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(vG, v_255));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(vR, v_255));

        __m256i vOrig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pOrigSrc));
        __m256i vA_int = _mm256_and_si256(vOrig, v_alpha_mask_8bit);

        __m256i vOut = _mm256_or_si256(
            _mm256_or_si256(_mm256_cvtps_epi32(vB), _mm256_slli_epi32(_mm256_cvtps_epi32(vG), 8)),
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vR), 16), vA_int)
        );
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), vOut);
    }
};

template <> struct PixelTraits<PixelFormat::ARGB_8u>
{
    using DataType = PF_Pixel_ARGB_8u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i v_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        vR = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 8), v_mask_8bit));
        vG = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 16), v_mask_8bit));
        vB = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 24), v_mask_8bit));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_max_ps(v_zero, _mm256_min_ps(vB, v_255));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(vG, v_255));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(vR, v_255));

        __m256i vOrig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pOrigSrc));
        __m256i vA_int = _mm256_and_si256(vOrig, v_mask_8bit);

        __m256i vOut = _mm256_or_si256(
            _mm256_or_si256(vA_int, _mm256_slli_epi32(_mm256_cvtps_epi32(vR), 8)),
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vG), 16), _mm256_slli_epi32(_mm256_cvtps_epi32(vB), 24))
        );
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), vOut);
    }
};

template <class C> struct Traits_VUYA_8u
{
    using DataType = PF_Pixel_VUYA_8u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i v_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        __m256 vV = _mm256_cvtepi32_ps(_mm256_and_si256(v_pixels, v_mask_8bit));
        __m256 vU = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 8), v_mask_8bit));
        __m256 vY = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 16), v_mask_8bit));

        vV = _mm256_sub_ps(vV, v_128);
        vU = _mm256_sub_ps(vU, v_128);

        // Decode Rec.709 to Perceptual RGB
        vR = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_r_v(), vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(C::inv_g_u(), vU), _mm256_mul_ps(C::inv_g_v(), vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_b_u(), vU)); 
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        // Encode RGB back to Rec.709 YUV
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(C::y_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::y_g(), vG), _mm256_mul_ps(C::y_b(), vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(C::u_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::u_g(), vG), _mm256_mul_ps(C::u_b(), vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(C::v_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::v_g(), vG), _mm256_mul_ps(C::v_b(), vB)));

        vV = _mm256_add_ps(vV, v_128);
        vU = _mm256_add_ps(vU, v_128);

        vV = _mm256_max_ps(v_zero, _mm256_min_ps(vV, v_255));
        vU = _mm256_max_ps(v_zero, _mm256_min_ps(vU, v_255));
        vY = _mm256_max_ps(v_zero, _mm256_min_ps(vY, v_255));

        __m256i vOrig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pOrigSrc));
        __m256i vA_int = _mm256_and_si256(vOrig, v_alpha_mask_8bit);

        __m256i vOut = _mm256_or_si256(
            _mm256_or_si256(_mm256_cvtps_epi32(vV), _mm256_slli_epi32(_mm256_cvtps_epi32(vU), 8)),
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vY), 16), vA_int)
        );
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), vOut);
    }
};

// --- PREMULTIPLIED 8-BIT ---
template <> struct PixelTraits<PixelFormat::BGRP_8u>
{
    using DataType = PF_Pixel_BGRA_8u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i v_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        vB = _mm256_cvtepi32_ps(_mm256_and_si256(v_pixels, v_mask_8bit));
        vG = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 8), v_mask_8bit));
        vR = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 16), v_mask_8bit));

        __m256 vA = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 24), v_mask_8bit));
        __m256 vA_norm = _mm256_mul_ps(vA, v_alpha_norm_8);
        __m256 vA_safe = _mm256_blendv_ps(v_one, vA_norm, _mm256_cmp_ps(vA_norm, v_zero, _CMP_GT_OQ));

        // Un-premultiply to Perceptual RGB
        vB = _mm256_min_ps(_mm256_div_ps(vB, vA_safe), v_255);
        vG = _mm256_min_ps(_mm256_div_ps(vG, vA_safe), v_255);
        vR = _mm256_min_ps(_mm256_div_ps(vR, vA_safe), v_255);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256i vOrig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pOrigSrc));
        __m256i vA_int = _mm256_and_si256(vOrig, v_alpha_mask_8bit);
        __m256 vA_norm = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(vA_int, 24)), v_alpha_norm_8);

        // Re-premultiply Straight Color * Alpha
        vB = _mm256_mul_ps(vB, vA_norm);
        vG = _mm256_mul_ps(vG, vA_norm);
        vR = _mm256_mul_ps(vR, vA_norm);

        vB = _mm256_max_ps(v_zero, _mm256_min_ps(vB, v_255));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(vG, v_255));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(vR, v_255));

        __m256i vOut = _mm256_or_si256(
            _mm256_or_si256(_mm256_cvtps_epi32(vB), _mm256_slli_epi32(_mm256_cvtps_epi32(vG), 8)),
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vR), 16), vA_int)
        );
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), vOut);
    }
};

template <class C> struct Traits_VUYP_8u
{
    using DataType = PF_Pixel_VUYA_8u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i v_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        __m256 vV = _mm256_cvtepi32_ps(_mm256_and_si256(v_pixels, v_mask_8bit));
        __m256 vU = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 8), v_mask_8bit));
        __m256 vY = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 16), v_mask_8bit));

        __m256 vA = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 24), v_mask_8bit));
        __m256 vA_norm = _mm256_mul_ps(vA, v_alpha_norm_8);
        __m256 vA_safe = _mm256_blendv_ps(v_one, vA_norm, _mm256_cmp_ps(vA_norm, v_zero, _CMP_GT_OQ));

        vV = _mm256_sub_ps(vV, v_128);
        vU = _mm256_sub_ps(vU, v_128);

        vV = _mm256_div_ps(vV, vA_safe);
        vU = _mm256_div_ps(vU, vA_safe);
        vY = _mm256_min_ps(_mm256_div_ps(vY, vA_safe), v_255);

        vR = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_r_v(), vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(C::inv_g_u(), vU), _mm256_mul_ps(C::inv_g_v(), vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_b_u(), vU)); 
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(C::y_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::y_g(), vG), _mm256_mul_ps(C::y_b(), vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(C::u_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::u_g(), vG), _mm256_mul_ps(C::u_b(), vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(C::v_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::v_g(), vG), _mm256_mul_ps(C::v_b(), vB)));

        __m256i vOrig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pOrigSrc));
        __m256i vA_int = _mm256_and_si256(vOrig, v_alpha_mask_8bit);
        __m256 vA_norm = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(vA_int, 24)), v_alpha_norm_8);

        vV = _mm256_mul_ps(vV, vA_norm);
        vU = _mm256_mul_ps(vU, vA_norm);
        vY = _mm256_mul_ps(vY, vA_norm);

        vV = _mm256_add_ps(vV, v_128);
        vU = _mm256_add_ps(vU, v_128);

        vV = _mm256_max_ps(v_zero, _mm256_min_ps(vV, v_255));
        vU = _mm256_max_ps(v_zero, _mm256_min_ps(vU, v_255));
        vY = _mm256_max_ps(v_zero, _mm256_min_ps(vY, v_255));

        __m256i vOut = _mm256_or_si256(
            _mm256_or_si256(_mm256_cvtps_epi32(vV), _mm256_slli_epi32(_mm256_cvtps_epi32(vU), 8)),
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vY), 16), vA_int)
        );
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), vOut);
    }
};

// ============================================================================
// 10-BIT SPECIALIZATION
// ============================================================================
template <> struct PixelTraits<PixelFormat::RGB_10u>
{
    using DataType = PF_Pixel_RGB_10u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i v_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        vB = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 2), v_mask_10bit)), v_scale_10_to_8);
        vG = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 12), v_mask_10bit)), v_scale_10_to_8);
        vR = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 22), v_mask_10bit)), v_scale_10_to_8);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vB, v_scale_8_to_10), v_1023));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vG, v_scale_8_to_10), v_1023));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vR, v_scale_8_to_10), v_1023));

        __m256i vOut = _mm256_or_si256(
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vB), 2), _mm256_slli_epi32(_mm256_cvtps_epi32(vG), 12)),
            _mm256_slli_epi32(_mm256_cvtps_epi32(vR), 22)
        );
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), vOut);
    }
};

// ============================================================================
// 16-BIT SPECIALIZATIONS
// ============================================================================
template <> struct PixelTraits<PixelFormat::BGRA_16u>
{
    using DataType = PF_Pixel_BGRA_16u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; }
        vB = _mm256_mul_ps(_mm256_load_ps(b_arr), v_scale_16_to_8);
        vG = _mm256_mul_ps(_mm256_load_ps(g_arr), v_scale_16_to_8);
        vR = _mm256_mul_ps(_mm256_load_ps(r_arr), v_scale_16_to_8);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vB, v_scale_8_to_16), v_32767));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vG, v_scale_8_to_16), v_32767));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vR, v_scale_8_to_16), v_32767));

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = static_cast<uint16_t>(b_arr[i] + 0.5f);
            pDst[i].G = static_cast<uint16_t>(g_arr[i] + 0.5f);
            pDst[i].R = static_cast<uint16_t>(r_arr[i] + 0.5f);
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <> struct PixelTraits<PixelFormat::ARGB_16u>
{
    using DataType = PF_Pixel_ARGB_16u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; }
        vB = _mm256_mul_ps(_mm256_load_ps(b_arr), v_scale_16_to_8);
        vG = _mm256_mul_ps(_mm256_load_ps(g_arr), v_scale_16_to_8);
        vR = _mm256_mul_ps(_mm256_load_ps(r_arr), v_scale_16_to_8);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vB, v_scale_8_to_16), v_32767));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vG, v_scale_8_to_16), v_32767));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vR, v_scale_8_to_16), v_32767));

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = static_cast<uint16_t>(b_arr[i] + 0.5f);
            pDst[i].G = static_cast<uint16_t>(g_arr[i] + 0.5f);
            pDst[i].R = static_cast<uint16_t>(r_arr[i] + 0.5f);
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <class C> struct Traits_VUYA_16u
{
    using DataType = PF_Pixel_VUYA_16u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8];
        for (int i = 0; i < 8; ++i)
        {
            v_arr[i] = static_cast<float>(pSrc[i].V);
            u_arr[i] = static_cast<float>(pSrc[i].U);
            y_arr[i] = static_cast<float>(pSrc[i].Y);
        }
        __m256 vV = _mm256_mul_ps(_mm256_load_ps(v_arr), v_scale_16_to_8);
        __m256 vU = _mm256_mul_ps(_mm256_load_ps(u_arr), v_scale_16_to_8);
        __m256 vY = _mm256_mul_ps(_mm256_load_ps(y_arr), v_scale_16_to_8);

        vV = _mm256_sub_ps(vV, v_128);
        vU = _mm256_sub_ps(vU, v_128);

        vR = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_r_v(), vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(C::inv_g_u(), vU), _mm256_mul_ps(C::inv_g_v(), vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_b_u(), vU));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(C::y_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::y_g(), vG), _mm256_mul_ps(C::y_b(), vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(C::u_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::u_g(), vG), _mm256_mul_ps(C::u_b(), vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(C::v_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::v_g(), vG), _mm256_mul_ps(C::v_b(), vB)));

        vV = _mm256_add_ps(vV, v_128);
        vU = _mm256_add_ps(vU, v_128);

        vV = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vV, v_scale_8_to_16), v_32767));
        vU = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vU, v_scale_8_to_16), v_32767));
        vY = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vY, v_scale_8_to_16), v_32767));

        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8];
        _mm256_store_ps(v_arr, vV); _mm256_store_ps(u_arr, vU); _mm256_store_ps(y_arr, vY);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].V = static_cast<uint16_t>(v_arr[i] + 0.5f);
            pDst[i].U = static_cast<uint16_t>(u_arr[i] + 0.5f);
            pDst[i].Y = static_cast<uint16_t>(y_arr[i] + 0.5f);
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <typename PixT> struct Traits_P_16u
{
    using DataType = PixT;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8], a_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; a_arr[i] = pSrc[i].A; }

        vB = _mm256_mul_ps(_mm256_load_ps(b_arr), v_scale_16_to_8);
        vG = _mm256_mul_ps(_mm256_load_ps(g_arr), v_scale_16_to_8);
        vR = _mm256_mul_ps(_mm256_load_ps(r_arr), v_scale_16_to_8);

        __m256 vA_norm = _mm256_mul_ps(_mm256_load_ps(a_arr), v_alpha_norm_16);
        __m256 vA_safe = _mm256_blendv_ps(v_one, vA_norm, _mm256_cmp_ps(vA_norm, v_zero, _CMP_GT_OQ));

        vB = _mm256_min_ps(_mm256_div_ps(vB, vA_safe), v_255);
        vG = _mm256_min_ps(_mm256_div_ps(vG, vA_safe), v_255);
        vR = _mm256_min_ps(_mm256_div_ps(vR, vA_safe), v_255);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        CACHE_ALIGN float a_arr[8];
        for (int i = 0; i < 8; ++i) { a_arr[i] = pOrigSrc[i].A; }

        __m256 vA_norm = _mm256_mul_ps(_mm256_load_ps(a_arr), v_alpha_norm_16);
        vB = _mm256_mul_ps(vB, vA_norm);
        vG = _mm256_mul_ps(vG, vA_norm);
        vR = _mm256_mul_ps(vR, vA_norm);

        vB = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vB, v_scale_8_to_16), v_32767));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vG, v_scale_8_to_16), v_32767));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vR, v_scale_8_to_16), v_32767));

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = static_cast<uint16_t>(b_arr[i] + 0.5f);
            pDst[i].G = static_cast<uint16_t>(g_arr[i] + 0.5f);
            pDst[i].R = static_cast<uint16_t>(r_arr[i] + 0.5f);
            pDst[i].A = static_cast<uint16_t>(a_arr[i]);
        }
    }
};

template <class C> struct Traits_VUYP_16u
{
    using DataType = PF_Pixel_VUYA_16u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8], a_arr[8];
        for (int i = 0; i < 8; ++i) { v_arr[i] = pSrc[i].V; u_arr[i] = pSrc[i].U; y_arr[i] = pSrc[i].Y; a_arr[i] = pSrc[i].A; }

        __m256 vV = _mm256_mul_ps(_mm256_load_ps(v_arr), v_scale_16_to_8);
        __m256 vU = _mm256_mul_ps(_mm256_load_ps(u_arr), v_scale_16_to_8);
        __m256 vY = _mm256_mul_ps(_mm256_load_ps(y_arr), v_scale_16_to_8);

        __m256 vA_norm = _mm256_mul_ps(_mm256_load_ps(a_arr), v_alpha_norm_16);
        __m256 vA_safe = _mm256_blendv_ps(v_one, vA_norm, _mm256_cmp_ps(vA_norm, v_zero, _CMP_GT_OQ));

        vV = _mm256_sub_ps(vV, v_128);
        vU = _mm256_sub_ps(vU, v_128);

        vV = _mm256_div_ps(vV, vA_safe);
        vU = _mm256_div_ps(vU, vA_safe);
        vY = _mm256_min_ps(_mm256_div_ps(vY, vA_safe), v_255);

        vR = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_r_v(), vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(C::inv_g_u(), vU), _mm256_mul_ps(C::inv_g_v(), vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_b_u(), vU));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(C::y_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::y_g(), vG), _mm256_mul_ps(C::y_b(), vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(C::u_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::u_g(), vG), _mm256_mul_ps(C::u_b(), vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(C::v_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::v_g(), vG), _mm256_mul_ps(C::v_b(), vB)));

        CACHE_ALIGN float a_arr[8];
        for (int i = 0; i < 8; ++i) { a_arr[i] = pOrigSrc[i].A; }
        __m256 vA_norm = _mm256_mul_ps(_mm256_load_ps(a_arr), v_alpha_norm_16);

        vV = _mm256_mul_ps(vV, vA_norm);
        vU = _mm256_mul_ps(vU, vA_norm);
        vY = _mm256_mul_ps(vY, vA_norm);

        vV = _mm256_add_ps(vV, v_128);
        vU = _mm256_add_ps(vU, v_128);

        vV = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vV, v_scale_8_to_16), v_32767));
        vU = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vU, v_scale_8_to_16), v_32767));
        vY = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vY, v_scale_8_to_16), v_32767));

        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8];
        _mm256_store_ps(v_arr, vV); _mm256_store_ps(u_arr, vU); _mm256_store_ps(y_arr, vY);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].V = static_cast<uint16_t>(v_arr[i] + 0.5f);
            pDst[i].U = static_cast<uint16_t>(u_arr[i] + 0.5f);
            pDst[i].Y = static_cast<uint16_t>(y_arr[i] + 0.5f);
            pDst[i].A = static_cast<uint16_t>(a_arr[i]);
        }
    }
};

// ============================================================================
// 32-BIT FLOAT SPECIALIZATIONS (GAMMA & LINEAR)
// ============================================================================
template <> struct PixelTraits<PixelFormat::BGRA_32f>
{
    using DataType = PF_Pixel_BGRA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; }
        vB = _mm256_mul_ps(_mm256_load_ps(b_arr), v_scale_32_to_8);
        vG = _mm256_mul_ps(_mm256_load_ps(g_arr), v_scale_32_to_8);
        vR = _mm256_mul_ps(_mm256_load_ps(r_arr), v_scale_32_to_8);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = Clamp01_32f(_mm256_mul_ps(vB, v_scale_8_to_32));
        vG = Clamp01_32f(_mm256_mul_ps(vG, v_scale_8_to_32));
        vR = Clamp01_32f(_mm256_mul_ps(vR, v_scale_8_to_32));

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = b_arr[i]; pDst[i].G = g_arr[i]; pDst[i].R = r_arr[i];
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <> struct PixelTraits<PixelFormat::ARGB_32f>
{
    using DataType = PF_Pixel_ARGB_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; }
        vB = _mm256_mul_ps(_mm256_load_ps(b_arr), v_scale_32_to_8);
        vG = _mm256_mul_ps(_mm256_load_ps(g_arr), v_scale_32_to_8);
        vR = _mm256_mul_ps(_mm256_load_ps(r_arr), v_scale_32_to_8);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = Clamp01_32f(_mm256_mul_ps(vB, v_scale_8_to_32));
        vG = Clamp01_32f(_mm256_mul_ps(vG, v_scale_8_to_32));
        vR = Clamp01_32f(_mm256_mul_ps(vR, v_scale_8_to_32));

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = b_arr[i]; pDst[i].G = g_arr[i]; pDst[i].R = r_arr[i];
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

// --- LINEAR GAMMA SANDWICH: THE "SECRET WEAPON" ---
template <typename PixT> struct Traits_RGB_32f_Linear
{
    using DataType = PixT;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; }
        
        // Forward Gamma 1/2.2 Transform (Linear to Perceptual for L2 distance safety)
        vB = _mm256_mul_ps(ApplyGammaAVX2(_mm256_load_ps(b_arr), 1.0f / 2.2f), v_scale_32_to_8);
        vG = _mm256_mul_ps(ApplyGammaAVX2(_mm256_load_ps(g_arr), 1.0f / 2.2f), v_scale_32_to_8);
        vR = _mm256_mul_ps(ApplyGammaAVX2(_mm256_load_ps(r_arr), 1.0f / 2.2f), v_scale_32_to_8);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        // Inverse Gamma 2.2 Transform (Perceptual back to Linear)
        vB = ApplyGammaAVX2(_mm256_mul_ps(vB, v_scale_8_to_32), 2.2f);
        vG = ApplyGammaAVX2(_mm256_mul_ps(vG, v_scale_8_to_32), 2.2f);
        vR = ApplyGammaAVX2(_mm256_mul_ps(vR, v_scale_8_to_32), 2.2f);

        vB = Clamp01_32f(vB);
        vG = Clamp01_32f(vG);
        vR = Clamp01_32f(vR);

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = b_arr[i]; pDst[i].G = g_arr[i]; pDst[i].R = r_arr[i];
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <typename PixT> struct Traits_P_32f
{
    using DataType = PixT;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8], a_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; a_arr[i] = pSrc[i].A; }

        vB = _mm256_load_ps(b_arr); vG = _mm256_load_ps(g_arr); vR = _mm256_load_ps(r_arr);
        __m256 vA_norm = _mm256_load_ps(a_arr);
        __m256 vA_safe = _mm256_blendv_ps(v_one, vA_norm, _mm256_cmp_ps(vA_norm, v_zero, _CMP_GT_OQ));

        vB = _mm256_mul_ps(ClampMax1_32f(_mm256_div_ps(vB, vA_safe)), v_scale_32_to_8);
        vG = _mm256_mul_ps(ClampMax1_32f(_mm256_div_ps(vG, vA_safe)), v_scale_32_to_8);
        vR = _mm256_mul_ps(ClampMax1_32f(_mm256_div_ps(vR, vA_safe)), v_scale_32_to_8);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        CACHE_ALIGN float a_arr[8];
        for (int i = 0; i < 8; ++i) { a_arr[i] = pOrigSrc[i].A; }
        __m256 vA_norm = _mm256_load_ps(a_arr);

        vB = _mm256_mul_ps(_mm256_mul_ps(vB, v_scale_8_to_32), vA_norm);
        vG = _mm256_mul_ps(_mm256_mul_ps(vG, v_scale_8_to_32), vA_norm);
        vR = _mm256_mul_ps(_mm256_mul_ps(vR, v_scale_8_to_32), vA_norm);

        vB = Clamp01_32f(vB);
        vG = Clamp01_32f(vG);
        vR = Clamp01_32f(vR);

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = b_arr[i]; pDst[i].G = g_arr[i]; pDst[i].R = r_arr[i];
            pDst[i].A = a_arr[i];
        }
    }
};

template <typename PixT> struct Traits_P_32f_Linear
{
    using DataType = PixT;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8], a_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; a_arr[i] = pSrc[i].A; }

        vB = _mm256_load_ps(b_arr); vG = _mm256_load_ps(g_arr); vR = _mm256_load_ps(r_arr);
        __m256 vA_norm = _mm256_load_ps(a_arr);
        __m256 vA_safe = _mm256_blendv_ps(v_one, vA_norm, _mm256_cmp_ps(vA_norm, v_zero, _CMP_GT_OQ));

        // 1. Un-premultiply FIRST
        vB = ClampMax1_32f(_mm256_div_ps(vB, vA_safe));
        vG = ClampMax1_32f(_mm256_div_ps(vG, vA_safe));
        vR = ClampMax1_32f(_mm256_div_ps(vR, vA_safe));

        // 2. Apply Forward Gamma Sandwich
        vB = _mm256_mul_ps(ApplyGammaAVX2(vB, 1.0f / 2.2f), v_scale_32_to_8);
        vG = _mm256_mul_ps(ApplyGammaAVX2(vG, 1.0f / 2.2f), v_scale_32_to_8);
        vR = _mm256_mul_ps(ApplyGammaAVX2(vR, 1.0f / 2.2f), v_scale_32_to_8);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        // 1. Inverse Gamma Sandwich
        vB = ApplyGammaAVX2(_mm256_mul_ps(vB, v_scale_8_to_32), 2.2f);
        vG = ApplyGammaAVX2(_mm256_mul_ps(vG, v_scale_8_to_32), 2.2f);
        vR = ApplyGammaAVX2(_mm256_mul_ps(vR, v_scale_8_to_32), 2.2f);

        CACHE_ALIGN float a_arr[8];
        for (int i = 0; i < 8; ++i) { a_arr[i] = pOrigSrc[i].A; }
        __m256 vA_norm = _mm256_load_ps(a_arr);

        // 2. Re-premultiply
        vB = _mm256_mul_ps(vB, vA_norm);
        vG = _mm256_mul_ps(vG, vA_norm);
        vR = _mm256_mul_ps(vR, vA_norm);

        vB = Clamp01_32f(vB);
        vG = Clamp01_32f(vG);
        vR = Clamp01_32f(vR);

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = b_arr[i]; pDst[i].G = g_arr[i]; pDst[i].R = r_arr[i];
            pDst[i].A = a_arr[i];
        }
    }
};

template <class C> struct Traits_VUYA_32f
{
    using DataType = PF_Pixel_VUYA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8];
        for (int i = 0; i < 8; ++i) { v_arr[i] = pSrc[i].V; u_arr[i] = pSrc[i].U; y_arr[i] = pSrc[i].Y; }
        __m256 vV = _mm256_mul_ps(_mm256_load_ps(v_arr), v_scale_32_to_8);
        __m256 vU = _mm256_mul_ps(_mm256_load_ps(u_arr), v_scale_32_to_8);
        __m256 vY = _mm256_mul_ps(_mm256_load_ps(y_arr), v_scale_32_to_8);

        // 32f YUV has no 128 bias: Y 0..1, U/V -0.5..0.5. Scaled x255 to the engine range (F1).
        // Decode directly to RGB
        vR = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_r_v(), vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(C::inv_g_u(), vU), _mm256_mul_ps(C::inv_g_v(), vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_b_u(), vU));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(C::y_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::y_g(), vG), _mm256_mul_ps(C::y_b(), vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(C::u_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::u_g(), vG), _mm256_mul_ps(C::u_b(), vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(C::v_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::v_g(), vG), _mm256_mul_ps(C::v_b(), vB)));

        vV = _mm256_mul_ps(vV, v_scale_8_to_32);
        vU = _mm256_mul_ps(vU, v_scale_8_to_32);
        vY = _mm256_mul_ps(vY, v_scale_8_to_32);

        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8];
        _mm256_store_ps(v_arr, vV); _mm256_store_ps(u_arr, vU); _mm256_store_ps(y_arr, vY);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].V = v_arr[i]; pDst[i].U = u_arr[i]; pDst[i].Y = y_arr[i];
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <class C> struct Traits_VUYP_32f
{
    using DataType = PF_Pixel_VUYA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8], a_arr[8];
        for (int i = 0; i < 8; ++i) { v_arr[i] = pSrc[i].V; u_arr[i] = pSrc[i].U; y_arr[i] = pSrc[i].Y; a_arr[i] = pSrc[i].A; }

        __m256 vV = _mm256_load_ps(v_arr);
        __m256 vU = _mm256_load_ps(u_arr);
        __m256 vY = _mm256_load_ps(y_arr);

        __m256 vA_norm = _mm256_load_ps(a_arr);
        __m256 vA_safe = _mm256_blendv_ps(v_one, vA_norm, _mm256_cmp_ps(vA_norm, v_zero, _CMP_GT_OQ));

        vV = _mm256_div_ps(vV, vA_safe);
        vU = _mm256_div_ps(vU, vA_safe);
        vY = ClampMax1_32f(_mm256_div_ps(vY, vA_safe));

        // engine range (F1)
        vV = _mm256_mul_ps(vV, v_scale_32_to_8);
        vU = _mm256_mul_ps(vU, v_scale_32_to_8);
        vY = _mm256_mul_ps(vY, v_scale_32_to_8);

        vR = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_r_v(), vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(C::inv_g_u(), vU), _mm256_mul_ps(C::inv_g_v(), vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(C::inv_b_u(), vU));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(C::y_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::y_g(), vG), _mm256_mul_ps(C::y_b(), vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(C::u_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::u_g(), vG), _mm256_mul_ps(C::u_b(), vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(C::v_r(), vR), _mm256_add_ps(_mm256_mul_ps(C::v_g(), vG), _mm256_mul_ps(C::v_b(), vB)));

        vV = _mm256_mul_ps(vV, v_scale_8_to_32);
        vU = _mm256_mul_ps(vU, v_scale_8_to_32);
        vY = _mm256_mul_ps(vY, v_scale_8_to_32);

        CACHE_ALIGN float a_arr[8];
        for (int i = 0; i < 8; ++i) { a_arr[i] = pOrigSrc[i].A; }
        __m256 vA_norm = _mm256_load_ps(a_arr);

        vV = _mm256_mul_ps(vV, vA_norm);
        vU = _mm256_mul_ps(vU, vA_norm);
        vY = _mm256_mul_ps(vY, vA_norm);

        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8];
        _mm256_store_ps(v_arr, vV); _mm256_store_ps(u_arr, vU); _mm256_store_ps(y_arr, vY);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].V = v_arr[i]; pDst[i].U = u_arr[i]; pDst[i].Y = y_arr[i];
            pDst[i].A = a_arr[i];
        }
    }
};

// --- PREMULTIPLIED 8-BIT, ARGB MEMORY ORDER (added F4) ---
template <> struct PixelTraits<PixelFormat::PRGB_8u>
{
    using DataType = PF_Pixel_ARGB_8u;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        __m256i v_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrc));
        vR = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 8), v_mask_8bit));
        vG = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 16), v_mask_8bit));
        vB = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v_pixels, 24), v_mask_8bit));

        __m256 vA = _mm256_cvtepi32_ps(_mm256_and_si256(v_pixels, v_mask_8bit));
        __m256 vA_norm = _mm256_mul_ps(vA, v_alpha_norm_8);
        __m256 vA_safe = _mm256_blendv_ps(v_one, vA_norm, _mm256_cmp_ps(vA_norm, v_zero, _CMP_GT_OQ));

        // Un-premultiply to Perceptual RGB
        vB = _mm256_min_ps(_mm256_div_ps(vB, vA_safe), v_255);
        vG = _mm256_min_ps(_mm256_div_ps(vG, vA_safe), v_255);
        vR = _mm256_min_ps(_mm256_div_ps(vR, vA_safe), v_255);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256i vOrig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pOrigSrc));
        __m256i vA_int = _mm256_and_si256(vOrig, v_mask_8bit);
        __m256 vA_norm = _mm256_mul_ps(_mm256_cvtepi32_ps(vA_int), v_alpha_norm_8);

        // Re-premultiply Straight Color * Alpha
        vB = _mm256_mul_ps(vB, vA_norm);
        vG = _mm256_mul_ps(vG, vA_norm);
        vR = _mm256_mul_ps(vR, vA_norm);

        vB = _mm256_max_ps(v_zero, _mm256_min_ps(vB, v_255));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(vG, v_255));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(vR, v_255));

        __m256i vOut = _mm256_or_si256(
            _mm256_or_si256(vA_int, _mm256_slli_epi32(_mm256_cvtps_epi32(vR), 8)),
            _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(vG), 16), _mm256_slli_epi32(_mm256_cvtps_epi32(vB), 24))
        );
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst), vOut);
    }
};

// ============================================================================
// FORMAT ALIASES (Zero-Overhead Inheritance)
// ============================================================================

// 8-Bit & 16-Bit Aliases
template <> struct PixelTraits<PixelFormat::BGRX_8u> : public PixelTraits<PixelFormat::BGRA_8u> {};
template <> struct PixelTraits<PixelFormat::BGRX_16u> : public PixelTraits<PixelFormat::BGRA_16u> {};

// 32-Bit Float Aliases 
template <> struct PixelTraits<PixelFormat::BGRX_32f> : public PixelTraits<PixelFormat::BGRA_32f> {};

// 32-Bit Float templates instantiated per memory layout (F4)
template <> struct PixelTraits<PixelFormat::BGRA_32f_Linear> : public Traits_RGB_32f_Linear<PF_Pixel_BGRA_32f> {};
template <> struct PixelTraits<PixelFormat::ARGB_32f_Linear> : public Traits_RGB_32f_Linear<PF_Pixel_ARGB_32f> {};
template <> struct PixelTraits<PixelFormat::BGRX_32f_Linear> : public PixelTraits<PixelFormat::BGRA_32f_Linear> {};
template <> struct PixelTraits<PixelFormat::BGRP_32f>        : public Traits_P_32f<PF_Pixel_BGRA_32f> {};
template <> struct PixelTraits<PixelFormat::PRGB_32f>        : public Traits_P_32f<PF_Pixel_ARGB_32f> {};
template <> struct PixelTraits<PixelFormat::BGRP_32f_Linear> : public Traits_P_32f_Linear<PF_Pixel_BGRA_32f> {};
template <> struct PixelTraits<PixelFormat::PRGB_32f_Linear> : public Traits_P_32f_Linear<PF_Pixel_ARGB_32f> {};
template <> struct PixelTraits<PixelFormat::BGRP_16u>        : public Traits_P_16u<PF_Pixel_BGRA_16u> {};
template <> struct PixelTraits<PixelFormat::PRGB_16u>        : public Traits_P_16u<PF_Pixel_ARGB_16u> {};

// Premiere Pro YUV: no suffix = Rec.601, _709 suffix = Rec.709 (F2)
template <> struct PixelTraits<PixelFormat::VUYA_8u>      : public Traits_VUYA_8u<YuvCoef601> {};
template <> struct PixelTraits<PixelFormat::VUYA_8u_709>  : public Traits_VUYA_8u<YuvCoef709> {};
template <> struct PixelTraits<PixelFormat::VUYP_8u>      : public Traits_VUYP_8u<YuvCoef601> {};
template <> struct PixelTraits<PixelFormat::VUYP_8u_709>  : public Traits_VUYP_8u<YuvCoef709> {};
template <> struct PixelTraits<PixelFormat::VUYA_16u>     : public Traits_VUYA_16u<YuvCoef601> {};
template <> struct PixelTraits<PixelFormat::VUYA_16u_709> : public Traits_VUYA_16u<YuvCoef709> {};
template <> struct PixelTraits<PixelFormat::VUYP_16u>     : public Traits_VUYP_16u<YuvCoef601> {};
template <> struct PixelTraits<PixelFormat::VUYP_16u_709> : public Traits_VUYP_16u<YuvCoef709> {};
template <> struct PixelTraits<PixelFormat::VUYA_32f>     : public Traits_VUYA_32f<YuvCoef601> {};
template <> struct PixelTraits<PixelFormat::VUYA_32f_709> : public Traits_VUYA_32f<YuvCoef709> {};
template <> struct PixelTraits<PixelFormat::VUYP_32f>     : public Traits_VUYP_32f<YuvCoef601> {};
template <> struct PixelTraits<PixelFormat::VUYP_32f_709> : public Traits_VUYP_32f<YuvCoef709> {};

// "X" (unused 4th channel) formats: same layout as the alpha variant, X is copied unchanged (F4)
template <> struct PixelTraits<PixelFormat::VUYX_8u>      : public PixelTraits<PixelFormat::VUYA_8u> {};
template <> struct PixelTraits<PixelFormat::VUYX_8u_709>  : public PixelTraits<PixelFormat::VUYA_8u_709> {};
template <> struct PixelTraits<PixelFormat::VUYX_32f>     : public PixelTraits<PixelFormat::VUYA_32f> {};
template <> struct PixelTraits<PixelFormat::VUYX_32f_709> : public PixelTraits<PixelFormat::VUYA_32f_709> {};
template <> struct PixelTraits<PixelFormat::XRGB_8u>         : public PixelTraits<PixelFormat::ARGB_8u> {};
template <> struct PixelTraits<PixelFormat::XRGB_16u>        : public PixelTraits<PixelFormat::ARGB_16u> {};
template <> struct PixelTraits<PixelFormat::XRGB_32f>        : public PixelTraits<PixelFormat::ARGB_32f> {};
template <> struct PixelTraits<PixelFormat::XRGB_32f_Linear> : public PixelTraits<PixelFormat::ARGB_32f_Linear> {};