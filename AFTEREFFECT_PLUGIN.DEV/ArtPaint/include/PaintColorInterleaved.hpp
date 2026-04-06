#pragma once
#include "CommonPixFormat.hpp"
#include "PaintMemHandler.hpp"
#include "PaintPixelTraits.hpp"
#include <type_traits>

// Inverse Rec.709 Coefficients
static const __m256 v_inv_r_v = _mm256_set1_ps(1.5748f);
static const __m256 v_inv_g_u = _mm256_set1_ps(-0.187324f);
static const __m256 v_inv_g_v = _mm256_set1_ps(-0.468124f);
static const __m256 v_inv_b_u = _mm256_set1_ps(1.8556f);

// Bilinear Interpolation Weights
static const __m256 v_w9 = _mm256_set1_ps(9.0f / 16.0f);
static const __m256 v_w3 = _mm256_set1_ps(3.0f / 16.0f);
static const __m256 v_w1 = _mm256_set1_ps(1.0f / 16.0f);

namespace detail {

// ============================================================================
// NORMAL MODE: 1:1 FULL RESOLUTION UPSCALER (IsHalfSize == false)
// ============================================================================
template <PixelFormat FMT>
inline void convert_to_interleaved_impl
(
    const MemHandler& memHndl,
    const void* RESTRICT origSrcBuf, 
    void* RESTRICT dstBuf,           
    const A_long width,
    const A_long height,
    const A_long srcLinePitch,
    const A_long dstLinePitch,
    std::false_type /* IsHalfSize tag */
) noexcept
{
    using Traits = PixelTraits<FMT>;
    using PixelType = typename Traits::DataType;

    const PixelType* RESTRICT orig_pixels = reinterpret_cast<const PixelType*>(origSrcBuf);
          PixelType* RESTRICT out_pixels  = reinterpret_cast<      PixelType*>(dstBuf);

    const float* RESTRICT in_Y = memHndl.Y_planar;
    const float* RESTRICT in_U = memHndl.U_planar;
    const float* RESTRICT in_V = memHndl.V_planar;

    for (A_long y = 0; y < height; ++y)
    {
        const float* row_Y = in_Y + (y * width);
        const float* row_U = in_U + (y * width);
        const float* row_V = in_V + (y * width);

        const PixelType* orig_row = orig_pixels + (y * srcLinePitch);
              PixelType* out_row  = out_pixels  + (y * dstLinePitch);

        A_long x = 0;
        
        for (; x <= width - 8; x += 8)
        {
            __m256 vY = _mm256_loadu_ps(&row_Y[x]);
            __m256 vU = _mm256_loadu_ps(&row_U[x]);
            __m256 vV = _mm256_loadu_ps(&row_V[x]);

            __m256 out1, out2, out3;

            if (Traits::IsYUV) { out1 = vV; out2 = vU; out3 = vY; } 
            else
            {
                out3 = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_r_v, vV)); 
                out2 = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(v_inv_g_u, vU), _mm256_mul_ps(v_inv_g_v, vV))); 
                out1 = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_b_u, vU)); 
            }

            Traits::StoreAVX2(out_row + x, out1, out2, out3, orig_row + x);
        }

        const A_long remaining = width - x;
        if (remaining > 0)
        {
            CACHE_ALIGN float tail_Y[8] = {0}, tail_U[8] = {0}, tail_V[8] = {0};
            CACHE_ALIGN PixelType tail_orig[8] = {}, tail_out[8]  = {};

            for (A_long i = 0; i < remaining; ++i)
            {
                tail_Y[i] = row_Y[x + i]; tail_U[i] = row_U[x + i]; tail_V[i] = row_V[x + i];
                tail_orig[i] = orig_row[x + i];
            }

            __m256 vY = _mm256_load_ps(tail_Y), vU = _mm256_load_ps(tail_U), vV = _mm256_load_ps(tail_V);
            __m256 out1, out2, out3;

            if (Traits::IsYUV) { out1 = vV; out2 = vU; out3 = vY; } 
            else
            {
                out3 = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_r_v, vV)); 
                out2 = _mm256_add_ps(vY, _mm256_add_ps(_mm256_mul_ps(v_inv_g_u, vU), _mm256_mul_ps(v_inv_g_v, vV))); 
                out1 = _mm256_add_ps(vY, _mm256_mul_ps(v_inv_b_u, vU)); 
            }

            Traits::StoreAVX2(tail_out, out1, out2, out3, tail_orig);
            for (A_long i = 0; i < remaining; ++i) { out_row[x + i] = tail_out[i]; }
        }
    }
    return;
}

// ============================================================================
// FAST MODE: BILINEAR INTERPOLATION UPSCALER (IsHalfSize == true)
// ============================================================================
template <PixelFormat FMT>
inline void convert_to_interleaved_impl
(
    const MemHandler& memHndl,
    const void* RESTRICT origSrcBuf,
    void* RESTRICT dstBuf,
    const A_long width,        
    const A_long height,       
    const A_long srcLinePitch,
    const A_long dstLinePitch,
    std::true_type /* IsHalfSize tag */
) noexcept
{
    using Traits = PixelTraits<FMT>;
    using PixelType = typename Traits::DataType;

    const PixelType* RESTRICT orig_pixels = reinterpret_cast<const PixelType*>(origSrcBuf);
          PixelType* RESTRICT out_pixels  = reinterpret_cast<      PixelType*>(dstBuf);

    const float* RESTRICT in_Y = memHndl.Y_planar;
    const float* RESTRICT in_U = memHndl.U_planar;
    const float* RESTRICT in_V = memHndl.V_planar;

    const A_long planar_width = width / 2;
    const A_long planar_height = height / 2;

    auto calc_bilinear = [&](__m256 v00, __m256 v01, __m256 v10, __m256 v11, 
                             __m256& out_TL, __m256& out_TR, __m256& out_BL, __m256& out_BR) {
        __m256 m00_9 = _mm256_mul_ps(v00, v_w9), m00_3 = _mm256_mul_ps(v00, v_w3), m00_1 = _mm256_mul_ps(v00, v_w1);
        __m256 m01_9 = _mm256_mul_ps(v01, v_w9), m01_3 = _mm256_mul_ps(v01, v_w3), m01_1 = _mm256_mul_ps(v01, v_w1);
        __m256 m10_9 = _mm256_mul_ps(v10, v_w9), m10_3 = _mm256_mul_ps(v10, v_w3), m10_1 = _mm256_mul_ps(v10, v_w1);
        __m256 m11_9 = _mm256_mul_ps(v11, v_w9), m11_3 = _mm256_mul_ps(v11, v_w3), m11_1 = _mm256_mul_ps(v11, v_w1);

        out_TL = _mm256_add_ps(_mm256_add_ps(m00_9, m01_3), _mm256_add_ps(m10_3, m11_1));
        out_TR = _mm256_add_ps(_mm256_add_ps(m00_3, m01_9), _mm256_add_ps(m10_1, m11_3));
        out_BL = _mm256_add_ps(_mm256_add_ps(m00_3, m01_1), _mm256_add_ps(m10_9, m11_3));
        out_BR = _mm256_add_ps(_mm256_add_ps(m00_1, m01_3), _mm256_add_ps(m10_3, m11_9));
    };

    auto interleave_to_rows = [&](__m256 left, __m256 right, __m256& out1, __m256& out2)
    {
        __m256 lo = _mm256_unpacklo_ps(left, right);
        __m256 hi = _mm256_unpackhi_ps(left, right);
        out1 = _mm256_permute2f128_ps(lo, hi, 0x20);
        out2 = _mm256_permute2f128_ps(lo, hi, 0x31);
    };

    auto apply_rec709 = [&](__m256 Y, __m256 U, __m256 V, __m256& o1, __m256& o2, __m256& o3)
    {
        if (Traits::IsYUV) { o1 = V; o2 = U; o3 = Y; } 
        else
        {
            o3 = _mm256_add_ps(Y, _mm256_mul_ps(v_inv_r_v, V)); 
            o2 = _mm256_add_ps(Y, _mm256_add_ps(_mm256_mul_ps(v_inv_g_u, U), _mm256_mul_ps(v_inv_g_v, V))); 
            o1 = _mm256_add_ps(Y, _mm256_mul_ps(v_inv_b_u, U)); 
        }
    };

    for (A_long y = 0; y < planar_height; ++y)
    {
        A_long y_next = (y + 1 < planar_height) ? (y + 1) : y; // Clamp Y to bottom edge

        const float* row_Y0 = in_Y + (y * planar_width);
        const float* row_U0 = in_U + (y * planar_width);
        const float* row_V0 = in_V + (y * planar_width);

        const float* row_Y1 = in_Y + (y_next * planar_width);
        const float* row_U1 = in_U + (y_next * planar_width);
        const float* row_V1 = in_V + (y_next * planar_width);

        const PixelType* orig_row0 = orig_pixels + ((y * 2) * srcLinePitch);
        const PixelType* orig_row1 = orig_pixels + ((y * 2 + 1) * srcLinePitch);
              PixelType* out_row0  = out_pixels  + ((y * 2) * dstLinePitch);
              PixelType* out_row1  = out_pixels  + ((y * 2 + 1) * dstLinePitch);

        A_long x = 0;
        for (; x <= planar_width - 8; x += 8)
        {
            __m256 vY00 = _mm256_loadu_ps(&row_Y0[x]), vY01 = _mm256_loadu_ps(&row_Y0[x + 1]);
            __m256 vY10 = _mm256_loadu_ps(&row_Y1[x]), vY11 = _mm256_loadu_ps(&row_Y1[x + 1]);
            __m256 vU00 = _mm256_loadu_ps(&row_U0[x]), vU01 = _mm256_loadu_ps(&row_U0[x + 1]);
            __m256 vU10 = _mm256_loadu_ps(&row_U1[x]), vU11 = _mm256_loadu_ps(&row_U1[x + 1]);
            __m256 vV00 = _mm256_loadu_ps(&row_V0[x]), vV01 = _mm256_loadu_ps(&row_V0[x + 1]);
            __m256 vV10 = _mm256_loadu_ps(&row_V1[x]), vV11 = _mm256_loadu_ps(&row_V1[x + 1]);

            __m256 vY_TL, vY_TR, vY_BL, vY_BR, vU_TL, vU_TR, vU_BL, vU_BR, vV_TL, vV_TR, vV_BL, vV_BR;
            calc_bilinear(vY00, vY01, vY10, vY11, vY_TL, vY_TR, vY_BL, vY_BR);
            calc_bilinear(vU00, vU01, vU10, vU11, vU_TL, vU_TR, vU_BL, vU_BR);
            calc_bilinear(vV00, vV01, vV10, vV11, vV_TL, vV_TR, vV_BL, vV_BR);

            __m256 vY_R0_1, vY_R0_2, vU_R0_1, vU_R0_2, vV_R0_1, vV_R0_2;
            interleave_to_rows(vY_TL, vY_TR, vY_R0_1, vY_R0_2);
            interleave_to_rows(vU_TL, vU_TR, vU_R0_1, vU_R0_2);
            interleave_to_rows(vV_TL, vV_TR, vV_R0_1, vV_R0_2);

            __m256 vY_R1_1, vY_R1_2, vU_R1_1, vU_R1_2, vV_R1_1, vV_R1_2;
            interleave_to_rows(vY_BL, vY_BR, vY_R1_1, vY_R1_2);
            interleave_to_rows(vU_BL, vU_BR, vU_R1_1, vU_R1_2);
            interleave_to_rows(vV_BL, vV_BR, vV_R1_1, vV_R1_2);

            __m256 r0_1_1, r0_1_2, r0_1_3; apply_rec709(vY_R0_1, vU_R0_1, vV_R0_1, r0_1_1, r0_1_2, r0_1_3);
            __m256 r0_2_1, r0_2_2, r0_2_3; apply_rec709(vY_R0_2, vU_R0_2, vV_R0_2, r0_2_1, r0_2_2, r0_2_3);
            __m256 r1_1_1, r1_1_2, r1_1_3; apply_rec709(vY_R1_1, vU_R1_1, vV_R1_1, r1_1_1, r1_1_2, r1_1_3);
            __m256 r1_2_1, r1_2_2, r1_2_3; apply_rec709(vY_R1_2, vU_R1_2, vV_R1_2, r1_2_1, r1_2_2, r1_2_3);

            A_long dest_x = x * 2;
            Traits::StoreAVX2(out_row0 + dest_x,     r0_1_1, r0_1_2, r0_1_3, orig_row0 + dest_x);
            Traits::StoreAVX2(out_row0 + dest_x + 8, r0_2_1, r0_2_2, r0_2_3, orig_row0 + dest_x + 8);
            Traits::StoreAVX2(out_row1 + dest_x,     r1_1_1, r1_1_2, r1_1_3, orig_row1 + dest_x);
            Traits::StoreAVX2(out_row1 + dest_x + 8, r1_2_1, r1_2_2, r1_2_3, orig_row1 + dest_x + 8);
        }

        const A_long remaining = planar_width - x;
        if (remaining > 0)
        {
            CACHE_ALIGN float tail_Y0[8] = {0}, tail_Y0_s[8] = {0}, tail_Y1[8] = {0}, tail_Y1_s[8] = {0};
            CACHE_ALIGN float tail_U0[8] = {0}, tail_U0_s[8] = {0}, tail_U1[8] = {0}, tail_U1_s[8] = {0};
            CACHE_ALIGN float tail_V0[8] = {0}, tail_V0_s[8] = {0}, tail_V1[8] = {0}, tail_V1_s[8] = {0};

            CACHE_ALIGN PixelType tail_orig0[16] = {}, tail_orig1[16] = {};
            CACHE_ALIGN PixelType tail_out0[16]  = {}, tail_out1[16]  = {};

            for (A_long i = 0; i < remaining; ++i) 
            {
                tail_Y0[i] = row_Y0[x + i]; tail_U0[i] = row_U0[x + i]; tail_V0[i] = row_V0[x + i];
                tail_Y1[i] = row_Y1[x + i]; tail_U1[i] = row_U1[x + i]; tail_V1[i] = row_V1[x + i];
                
                // Edge Clamping
                A_long read_idx = (x + i + 1 < planar_width) ? (x + i + 1) : (x + i);
                tail_Y0_s[i] = row_Y0[read_idx]; tail_U0_s[i] = row_U0[read_idx]; tail_V0_s[i] = row_V0[read_idx];
                tail_Y1_s[i] = row_Y1[read_idx]; tail_U1_s[i] = row_U1[read_idx]; tail_V1_s[i] = row_V1[read_idx];
                
                tail_orig0[i*2]   = orig_row0[x*2 + i*2];
                tail_orig0[i*2+1] = orig_row0[x*2 + i*2 + 1];
                tail_orig1[i*2]   = orig_row1[x*2 + i*2];
                tail_orig1[i*2+1] = orig_row1[x*2 + i*2 + 1];
            }

            __m256 vY00 = _mm256_load_ps(tail_Y0), vY01 = _mm256_load_ps(tail_Y0_s);
            __m256 vY10 = _mm256_load_ps(tail_Y1), vY11 = _mm256_load_ps(tail_Y1_s);
            __m256 vU00 = _mm256_load_ps(tail_U0), vU01 = _mm256_load_ps(tail_U0_s);
            __m256 vU10 = _mm256_load_ps(tail_U1), vU11 = _mm256_load_ps(tail_U1_s);
            __m256 vV00 = _mm256_load_ps(tail_V0), vV01 = _mm256_load_ps(tail_V0_s);
            __m256 vV10 = _mm256_load_ps(tail_V1), vV11 = _mm256_load_ps(tail_V1_s);

            __m256 vY_TL, vY_TR, vY_BL, vY_BR, vU_TL, vU_TR, vU_BL, vU_BR, vV_TL, vV_TR, vV_BL, vV_BR;
            calc_bilinear(vY00, vY01, vY10, vY11, vY_TL, vY_TR, vY_BL, vY_BR);
            calc_bilinear(vU00, vU01, vU10, vU11, vU_TL, vU_TR, vU_BL, vU_BR);
            calc_bilinear(vV00, vV01, vV10, vV11, vV_TL, vV_TR, vV_BL, vV_BR);

            __m256 vY_R0_1, vY_R0_2, vU_R0_1, vU_R0_2, vV_R0_1, vV_R0_2;
            interleave_to_rows(vY_TL, vY_TR, vY_R0_1, vY_R0_2);
            interleave_to_rows(vU_TL, vU_TR, vU_R0_1, vU_R0_2);
            interleave_to_rows(vV_TL, vV_TR, vV_R0_1, vV_R0_2);

            __m256 vY_R1_1, vY_R1_2, vU_R1_1, vU_R1_2, vV_R1_1, vV_R1_2;
            interleave_to_rows(vY_BL, vY_BR, vY_R1_1, vY_R1_2);
            interleave_to_rows(vU_BL, vU_BR, vU_R1_1, vU_R1_2);
            interleave_to_rows(vV_BL, vV_BR, vV_R1_1, vV_R1_2);

            __m256 r0_1_1, r0_1_2, r0_1_3; apply_rec709(vY_R0_1, vU_R0_1, vV_R0_1, r0_1_1, r0_1_2, r0_1_3);
            __m256 r0_2_1, r0_2_2, r0_2_3; apply_rec709(vY_R0_2, vU_R0_2, vV_R0_2, r0_2_1, r0_2_2, r0_2_3);
            __m256 r1_1_1, r1_1_2, r1_1_3; apply_rec709(vY_R1_1, vU_R1_1, vV_R1_1, r1_1_1, r1_1_2, r1_1_3);
            __m256 r1_2_1, r1_2_2, r1_2_3; apply_rec709(vY_R1_2, vU_R1_2, vV_R1_2, r1_2_1, r1_2_2, r1_2_3);

            Traits::StoreAVX2(tail_out0, r0_1_1, r0_1_2, r0_1_3, tail_orig0);
            Traits::StoreAVX2(tail_out0 + 8, r0_2_1, r0_2_2, r0_2_3, tail_orig0 + 8);
            Traits::StoreAVX2(tail_out1, r1_1_1, r1_1_2, r1_1_3, tail_orig1);
            Traits::StoreAVX2(tail_out1 + 8, r1_2_1, r1_2_2, r1_2_3, tail_orig1 + 8);

            for (A_long i = 0; i < remaining * 2; ++i)
            {
                out_row0[x*2 + i] = tail_out0[i];
                out_row1[x*2 + i] = tail_out1[i];
            }
        }
    }
    return;
}

} // end namespace detail

template <PixelFormat FMT, bool IsHalfSize>
void convert_to_interleaved_AVX2
(
    const MemHandler& memHndl,
    const void* RESTRICT origSrcBuf,
    void* RESTRICT dstBuf,
    const A_long width,
    const A_long height,
    const A_long srcLinePitch,
    const A_long dstLinePitch
) noexcept
{
    detail::convert_to_interleaved_impl<FMT>(
        memHndl, origSrcBuf, dstBuf, width, height, srcLinePitch, dstLinePitch, 
        std::integral_constant<bool, IsHalfSize>{}
    );
    return;
}