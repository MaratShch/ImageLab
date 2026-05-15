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
    RGB_10u
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
        arr[i] = FastCompute::Pow(std::max(0.0f, arr[i]), exponent);
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

template <> struct PixelTraits<PixelFormat::VUYA_8u>
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
        vR = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_r_v, vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(v_inv_g_u, vU), _mm256_mul_ps(v_inv_g_v, vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_b_u, vU)); 
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        // Encode RGB back to Rec.709 YUV
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(v_y_r, vR), _mm256_add_ps(_mm256_mul_ps(v_y_g, vG), _mm256_mul_ps(v_y_b, vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(v_u_r, vR), _mm256_add_ps(_mm256_mul_ps(v_u_g, vG), _mm256_mul_ps(v_u_b, vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(v_v_r, vR), _mm256_add_ps(_mm256_mul_ps(v_v_g, vG), _mm256_mul_ps(v_v_b, vB)));

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

template <> struct PixelTraits<PixelFormat::VUYP_8u>
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

        vR = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_r_v, vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(v_inv_g_u, vU), _mm256_mul_ps(v_inv_g_v, vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_b_u, vU)); 
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(v_y_r, vR), _mm256_add_ps(_mm256_mul_ps(v_y_g, vG), _mm256_mul_ps(v_y_b, vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(v_u_r, vR), _mm256_add_ps(_mm256_mul_ps(v_u_g, vG), _mm256_mul_ps(v_u_b, vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(v_v_r, vR), _mm256_add_ps(_mm256_mul_ps(v_v_g, vG), _mm256_mul_ps(v_v_b, vB)));

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
            pDst[i].B = static_cast<uint16_t>(b_arr[i]);
            pDst[i].G = static_cast<uint16_t>(g_arr[i]);
            pDst[i].R = static_cast<uint16_t>(r_arr[i]);
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
            pDst[i].B = static_cast<uint16_t>(b_arr[i]);
            pDst[i].G = static_cast<uint16_t>(g_arr[i]);
            pDst[i].R = static_cast<uint16_t>(r_arr[i]);
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <> struct PixelTraits<PixelFormat::VUYA_16u>
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

        vR = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_r_v, vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(v_inv_g_u, vU), _mm256_mul_ps(v_inv_g_v, vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_b_u, vU));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(v_y_r, vR), _mm256_add_ps(_mm256_mul_ps(v_y_g, vG), _mm256_mul_ps(v_y_b, vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(v_u_r, vR), _mm256_add_ps(_mm256_mul_ps(v_u_g, vG), _mm256_mul_ps(v_u_b, vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(v_v_r, vR), _mm256_add_ps(_mm256_mul_ps(v_v_g, vG), _mm256_mul_ps(v_v_b, vB)));

        vV = _mm256_add_ps(vV, v_128);
        vU = _mm256_add_ps(vU, v_128);

        vV = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vV, v_scale_8_to_16), v_32767));
        vU = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vU, v_scale_8_to_16), v_32767));
        vY = _mm256_max_ps(v_zero, _mm256_min_ps(_mm256_mul_ps(vY, v_scale_8_to_16), v_32767));

        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8];
        _mm256_store_ps(v_arr, vV); _mm256_store_ps(u_arr, vU); _mm256_store_ps(y_arr, vY);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].V = static_cast<uint16_t>(v_arr[i]);
            pDst[i].U = static_cast<uint16_t>(u_arr[i]);
            pDst[i].Y = static_cast<uint16_t>(y_arr[i]);
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <> struct PixelTraits<PixelFormat::BGRP_16u>
{
    using DataType = PF_Pixel_BGRA_16u;

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
            pDst[i].B = static_cast<uint16_t>(b_arr[i]);
            pDst[i].G = static_cast<uint16_t>(g_arr[i]);
            pDst[i].R = static_cast<uint16_t>(r_arr[i]);
            pDst[i].A = static_cast<uint16_t>(a_arr[i]);
        }
    }
};

template <> struct PixelTraits<PixelFormat::VUYP_16u>
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

        vR = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_r_v, vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(v_inv_g_u, vU), _mm256_mul_ps(v_inv_g_v, vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_b_u, vU));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(v_y_r, vR), _mm256_add_ps(_mm256_mul_ps(v_y_g, vG), _mm256_mul_ps(v_y_b, vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(v_u_r, vR), _mm256_add_ps(_mm256_mul_ps(v_u_g, vG), _mm256_mul_ps(v_u_b, vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(v_v_r, vR), _mm256_add_ps(_mm256_mul_ps(v_v_g, vG), _mm256_mul_ps(v_v_b, vB)));

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
            pDst[i].V = static_cast<uint16_t>(v_arr[i]);
            pDst[i].U = static_cast<uint16_t>(u_arr[i]);
            pDst[i].Y = static_cast<uint16_t>(y_arr[i]);
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
        vB = _mm256_load_ps(b_arr);
        vG = _mm256_load_ps(g_arr);
        vR = _mm256_load_ps(r_arr);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_max_ps(v_zero, _mm256_min_ps(vB, v_one));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(vG, v_one));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(vR, v_one));

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
        vB = _mm256_load_ps(b_arr);
        vG = _mm256_load_ps(g_arr);
        vR = _mm256_load_ps(r_arr);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        vB = _mm256_max_ps(v_zero, _mm256_min_ps(vB, v_one));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(vG, v_one));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(vR, v_one));

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
template <> struct PixelTraits<PixelFormat::BGRA_32f_Linear>
{
    using DataType = PF_Pixel_BGRA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; }
        
        // Forward Gamma 1/2.2 Transform (Linear to Perceptual for L2 distance safety)
        vB = ApplyGammaAVX2(_mm256_load_ps(b_arr), 1.0f / 2.2f);
        vG = ApplyGammaAVX2(_mm256_load_ps(g_arr), 1.0f / 2.2f);
        vR = ApplyGammaAVX2(_mm256_load_ps(r_arr), 1.0f / 2.2f);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        // Inverse Gamma 2.2 Transform (Perceptual back to Linear)
        vB = ApplyGammaAVX2(vB, 2.2f);
        vG = ApplyGammaAVX2(vG, 2.2f);
        vR = ApplyGammaAVX2(vR, 2.2f);

        vB = _mm256_max_ps(v_zero, _mm256_min_ps(vB, v_one));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(vG, v_one));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(vR, v_one));

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = b_arr[i]; pDst[i].G = g_arr[i]; pDst[i].R = r_arr[i];
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <> struct PixelTraits<PixelFormat::BGRP_32f>
{
    using DataType = PF_Pixel_BGRA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8], a_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; a_arr[i] = pSrc[i].A; }

        vB = _mm256_load_ps(b_arr); vG = _mm256_load_ps(g_arr); vR = _mm256_load_ps(r_arr);
        __m256 vA_norm = _mm256_load_ps(a_arr);
        __m256 vA_safe = _mm256_blendv_ps(v_one, vA_norm, _mm256_cmp_ps(vA_norm, v_zero, _CMP_GT_OQ));

        vB = _mm256_min_ps(_mm256_div_ps(vB, vA_safe), v_one);
        vG = _mm256_min_ps(_mm256_div_ps(vG, vA_safe), v_one);
        vR = _mm256_min_ps(_mm256_div_ps(vR, vA_safe), v_one);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        CACHE_ALIGN float a_arr[8];
        for (int i = 0; i < 8; ++i) { a_arr[i] = pOrigSrc[i].A; }
        __m256 vA_norm = _mm256_load_ps(a_arr);

        vB = _mm256_mul_ps(vB, vA_norm);
        vG = _mm256_mul_ps(vG, vA_norm);
        vR = _mm256_mul_ps(vR, vA_norm);

        vB = _mm256_max_ps(v_zero, _mm256_min_ps(vB, v_one));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(vG, v_one));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(vR, v_one));

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = b_arr[i]; pDst[i].G = g_arr[i]; pDst[i].R = r_arr[i];
            pDst[i].A = a_arr[i];
        }
    }
};

template <> struct PixelTraits<PixelFormat::BGRP_32f_Linear>
{
    using DataType = PF_Pixel_BGRA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8], a_arr[8];
        for (int i = 0; i < 8; ++i) { b_arr[i] = pSrc[i].B; g_arr[i] = pSrc[i].G; r_arr[i] = pSrc[i].R; a_arr[i] = pSrc[i].A; }

        vB = _mm256_load_ps(b_arr); vG = _mm256_load_ps(g_arr); vR = _mm256_load_ps(r_arr);
        __m256 vA_norm = _mm256_load_ps(a_arr);
        __m256 vA_safe = _mm256_blendv_ps(v_one, vA_norm, _mm256_cmp_ps(vA_norm, v_zero, _CMP_GT_OQ));

        // 1. Un-premultiply FIRST
        vB = _mm256_min_ps(_mm256_div_ps(vB, vA_safe), v_one);
        vG = _mm256_min_ps(_mm256_div_ps(vG, vA_safe), v_one);
        vR = _mm256_min_ps(_mm256_div_ps(vR, vA_safe), v_one);

        // 2. Apply Forward Gamma Sandwich
        vB = ApplyGammaAVX2(vB, 1.0f / 2.2f);
        vG = ApplyGammaAVX2(vG, 1.0f / 2.2f);
        vR = ApplyGammaAVX2(vR, 1.0f / 2.2f);
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        // 1. Inverse Gamma Sandwich
        vB = ApplyGammaAVX2(vB, 2.2f);
        vG = ApplyGammaAVX2(vG, 2.2f);
        vR = ApplyGammaAVX2(vR, 2.2f);

        CACHE_ALIGN float a_arr[8];
        for (int i = 0; i < 8; ++i) { a_arr[i] = pOrigSrc[i].A; }
        __m256 vA_norm = _mm256_load_ps(a_arr);

        // 2. Re-premultiply
        vB = _mm256_mul_ps(vB, vA_norm);
        vG = _mm256_mul_ps(vG, vA_norm);
        vR = _mm256_mul_ps(vR, vA_norm);

        vB = _mm256_max_ps(v_zero, _mm256_min_ps(vB, v_one));
        vG = _mm256_max_ps(v_zero, _mm256_min_ps(vG, v_one));
        vR = _mm256_max_ps(v_zero, _mm256_min_ps(vR, v_one));

        CACHE_ALIGN float b_arr[8], g_arr[8], r_arr[8];
        _mm256_store_ps(b_arr, vB); _mm256_store_ps(g_arr, vG); _mm256_store_ps(r_arr, vR);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].B = b_arr[i]; pDst[i].G = g_arr[i]; pDst[i].R = r_arr[i];
            pDst[i].A = a_arr[i];
        }
    }
};

template <> struct PixelTraits<PixelFormat::VUYA_32f>
{
    using DataType = PF_Pixel_VUYA_32f;

    static inline void LoadAVX2(const DataType* RESTRICT pSrc, __m256& vB, __m256& vG, __m256& vR) noexcept
    {
        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8];
        for (int i = 0; i < 8; ++i) { v_arr[i] = pSrc[i].V; u_arr[i] = pSrc[i].U; y_arr[i] = pSrc[i].Y; }
        __m256 vV = _mm256_load_ps(v_arr);
        __m256 vU = _mm256_load_ps(u_arr);
        __m256 vY = _mm256_load_ps(y_arr);

        // 32f YUV doesn't have 128 bias usually in Adobe, it's natively -0.5 to 0.5. 
        // Decode directly to RGB
        vR = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_r_v, vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(v_inv_g_u, vU), _mm256_mul_ps(v_inv_g_v, vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_b_u, vU));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(v_y_r, vR), _mm256_add_ps(_mm256_mul_ps(v_y_g, vG), _mm256_mul_ps(v_y_b, vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(v_u_r, vR), _mm256_add_ps(_mm256_mul_ps(v_u_g, vG), _mm256_mul_ps(v_u_b, vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(v_v_r, vR), _mm256_add_ps(_mm256_mul_ps(v_v_g, vG), _mm256_mul_ps(v_v_b, vB)));

        CACHE_ALIGN float v_arr[8], u_arr[8], y_arr[8];
        _mm256_store_ps(v_arr, vV); _mm256_store_ps(u_arr, vU); _mm256_store_ps(y_arr, vY);

        for (int i = 0; i < 8; ++i)
        {
            pDst[i].V = v_arr[i]; pDst[i].U = u_arr[i]; pDst[i].Y = y_arr[i];
            pDst[i].A = pOrigSrc[i].A;
        }
    }
};

template <> struct PixelTraits<PixelFormat::VUYP_32f>
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
        vY = _mm256_min_ps(_mm256_div_ps(vY, vA_safe), v_one);

        vR = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_r_v, vV)); 
        vG = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(v_inv_g_u, vU), _mm256_mul_ps(v_inv_g_v, vV))); 
        vB = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_b_u, vU));
    }

    static inline void StoreAVX2(DataType* RESTRICT pDst, __m256 vB, __m256 vG, __m256 vR, const DataType* RESTRICT pOrigSrc) noexcept
    {
        __m256 vY = _mm256_add_ps(_mm256_mul_ps(v_y_r, vR), _mm256_add_ps(_mm256_mul_ps(v_y_g, vG), _mm256_mul_ps(v_y_b, vB)));
        __m256 vU = _mm256_add_ps(_mm256_mul_ps(v_u_r, vR), _mm256_add_ps(_mm256_mul_ps(v_u_g, vG), _mm256_mul_ps(v_u_b, vB)));
        __m256 vV = _mm256_add_ps(_mm256_mul_ps(v_v_r, vR), _mm256_add_ps(_mm256_mul_ps(v_v_g, vG), _mm256_mul_ps(v_v_b, vB)));

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

// ============================================================================
// FORMAT ALIASES (Zero-Overhead Inheritance)
// ============================================================================

// 8-Bit & 16-Bit Aliases
template <> struct PixelTraits<PixelFormat::BGRX_8u> : public PixelTraits<PixelFormat::BGRA_8u> {};
template <> struct PixelTraits<PixelFormat::BGRX_16u> : public PixelTraits<PixelFormat::BGRA_16u> {};

// 32-Bit Float Aliases 
template <> struct PixelTraits<PixelFormat::BGRX_32f> : public PixelTraits<PixelFormat::BGRA_32f> {};
template <> struct PixelTraits<PixelFormat::BGRX_32f_Linear> : public PixelTraits<PixelFormat::BGRA_32f_Linear> {};

// Premiere Pro 709 Aliases 
template <> struct PixelTraits<PixelFormat::VUYA_8u_709>  : public PixelTraits<PixelFormat::VUYA_8u> {};
template <> struct PixelTraits<PixelFormat::VUYP_8u_709>  : public PixelTraits<PixelFormat::VUYP_8u> {};
template <> struct PixelTraits<PixelFormat::VUYA_16u_709> : public PixelTraits<PixelFormat::VUYA_16u> {};
template <> struct PixelTraits<PixelFormat::VUYP_16u_709> : public PixelTraits<PixelFormat::VUYP_16u> {};
template <> struct PixelTraits<PixelFormat::VUYA_32f_709> : public PixelTraits<PixelFormat::VUYA_32f> {};
template <> struct PixelTraits<PixelFormat::VUYP_32f_709> : public PixelTraits<PixelFormat::VUYP_32f> {};