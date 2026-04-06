#pragma once

#include <type_traits>
#include "CommonPixFormat.hpp"
#include "PaintMemHandler.hpp"
#include "PaintPixelTraits.hpp"


// Rec.709 Coefficients
static const __m256 v_y_r = _mm256_set1_ps(0.2126f);
static const __m256 v_y_g = _mm256_set1_ps(0.7152f);
static const __m256 v_y_b = _mm256_set1_ps(0.0722f);

static const __m256 v_u_r = _mm256_set1_ps(-0.114572f);
static const __m256 v_u_g = _mm256_set1_ps(-0.385428f);
static const __m256 v_u_b = _mm256_set1_ps(0.5f);

static const __m256 v_v_r = _mm256_set1_ps(0.5f);
static const __m256 v_v_g = _mm256_set1_ps(-0.454153f);
static const __m256 v_v_b = _mm256_set1_ps(-0.045847f);

static const __m256 v_quarter = _mm256_set1_ps(0.25f);

namespace detail
{

    // Helper inline function for correct AVX2 horizontal sums across 128-bit boundaries
    inline __m256 pack_2x2_box(__m256 lo, __m256 hi) noexcept
    {
        __m256 h_sum = _mm256_hadd_ps(lo, hi);
        // 0xD8 = _MM_SHUFFLE(3, 1, 2, 0) -> Fixes the 128-bit lane crossings
        return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(h_sum), 0xD8));
    }

    // ============================================================================
    // FAST MODE: 2x2 BOX FILTER DOWNSCALE (IsHalfSize == true)
    // ============================================================================
    template <PixelFormat FMT>
    inline void convert_to_planar_impl
    (
        const void* RESTRICT srcBuf,
        const MemHandler& memHndl,
        const A_long width,
        const A_long height,
        const A_long stride_pixels,
        std::true_type /* IsHalfSize tag */
    ) noexcept
    {
        using Traits = PixelTraits<FMT>;
        using PixelType = typename Traits::DataType;

        const PixelType* RESTRICT in_pixels = reinterpret_cast<const PixelType*>(srcBuf);
        float* RESTRICT out_Y = memHndl.Y_planar;
        float* RESTRICT out_U = memHndl.U_planar;
        float* RESTRICT out_V = memHndl.V_planar;

        const A_long out_width = width / 2;
        const A_long out_height = height / 2;

        for (A_long y = 0; y < out_height; ++y)
        {
            const PixelType* row0 = in_pixels + ((y * 2) * stride_pixels);
            const PixelType* row1 = in_pixels + ((y * 2 + 1) * stride_pixels);

            float* row_Y = out_Y + (y * out_width);
            float* row_U = out_U + (y * out_width);
            float* row_V = out_V + (y * out_width);

            A_long x_out = 0;

            // 1. MAIN AVX2 LOOP
            for (; x_out <= out_width - 8; x_out += 8)
            {
                A_long x_in = x_out * 2;

                __m256 vB0_lo, vG0_lo, vR0_lo, vB0_hi, vG0_hi, vR0_hi;
                __m256 vB1_lo, vG1_lo, vR1_lo, vB1_hi, vG1_hi, vR1_hi;

                Traits::LoadAVX2(row0 + x_in, vB0_lo, vG0_lo, vR0_lo);
                Traits::LoadAVX2(row0 + x_in + 8, vB0_hi, vG0_hi, vR0_hi);
                Traits::LoadAVX2(row1 + x_in, vB1_lo, vG1_lo, vR1_lo);
                Traits::LoadAVX2(row1 + x_in + 8, vB1_hi, vG1_hi, vR1_hi);

                // Vertical Sums
                __m256 vB_vlo = _mm256_add_ps(vB0_lo, vB1_lo), vB_vhi = _mm256_add_ps(vB0_hi, vB1_hi);
                __m256 vG_vlo = _mm256_add_ps(vG0_lo, vG1_lo), vG_vhi = _mm256_add_ps(vG0_hi, vG1_hi);
                __m256 vR_vlo = _mm256_add_ps(vR0_lo, vR1_lo), vR_vhi = _mm256_add_ps(vR0_hi, vR1_hi);

                // Horizontal Sums utilizing the new inline function
                __m256 vB_sum = pack_2x2_box(vB_vlo, vB_vhi);
                __m256 vG_sum = pack_2x2_box(vG_vlo, vG_vhi);
                __m256 vR_sum = pack_2x2_box(vR_vlo, vR_vhi);

                vB_sum = _mm256_mul_ps(vB_sum, v_quarter);
                vG_sum = _mm256_mul_ps(vG_sum, v_quarter);
                vR_sum = _mm256_mul_ps(vR_sum, v_quarter);

                __m256 v_Y_val, v_U_val, v_V_val;

                // Bypass Matrix for native YUV formats
                if (Traits::IsYUV)
                {
                    v_Y_val = vR_sum; // Y goes through R
                    v_U_val = vG_sum; // U goes through G
                    v_V_val = vB_sum; // V goes through B
                }
                else
                {
                    v_Y_val = _mm256_add_ps(_mm256_mul_ps(v_y_r, vR_sum), _mm256_add_ps(_mm256_mul_ps(v_y_g, vG_sum), _mm256_mul_ps(v_y_b, vB_sum)));
                    v_U_val = _mm256_add_ps(_mm256_mul_ps(v_u_r, vR_sum), _mm256_add_ps(_mm256_mul_ps(v_u_g, vG_sum), _mm256_mul_ps(v_u_b, vB_sum)));
                    v_V_val = _mm256_add_ps(_mm256_mul_ps(v_v_r, vR_sum), _mm256_add_ps(_mm256_mul_ps(v_v_g, vG_sum), _mm256_mul_ps(v_v_b, vB_sum)));
                }

                _mm256_storeu_ps(&row_Y[x_out], v_Y_val);
                _mm256_storeu_ps(&row_U[x_out], v_U_val);
                _mm256_storeu_ps(&row_V[x_out], v_V_val);
            }

            // 2. AVX2 PADDED TAIL
            const A_long remaining = out_width - x_out;
            if (remaining > 0)
            {
                CACHE_ALIGN PixelType tail0[16] = {}, tail1[16] = {};
                for (A_long i = 0; i < remaining * 2; ++i)
                {
                    tail0[i] = row0[x_out * 2 + i];
                    tail1[i] = row1[x_out * 2 + i];
                }

                __m256 vB0_lo, vG0_lo, vR0_lo, vB0_hi, vG0_hi, vR0_hi;
                __m256 vB1_lo, vG1_lo, vR1_lo, vB1_hi, vG1_hi, vR1_hi;

                Traits::LoadAVX2(tail0, vB0_lo, vG0_lo, vR0_lo);
                Traits::LoadAVX2(tail0 + 8, vB0_hi, vG0_hi, vR0_hi);
                Traits::LoadAVX2(tail1, vB1_lo, vG1_lo, vR1_lo);
                Traits::LoadAVX2(tail1 + 8, vB1_hi, vG1_hi, vR1_hi);

                __m256 vB_vlo = _mm256_add_ps(vB0_lo, vB1_lo), vB_vhi = _mm256_add_ps(vB0_hi, vB1_hi);
                __m256 vG_vlo = _mm256_add_ps(vG0_lo, vG1_lo), vG_vhi = _mm256_add_ps(vG0_hi, vG1_hi);
                __m256 vR_vlo = _mm256_add_ps(vR0_lo, vR1_lo), vR_vhi = _mm256_add_ps(vR0_hi, vR1_hi);

                __m256 vB_sum = _mm256_mul_ps(pack_2x2_box(vB_vlo, vB_vhi), v_quarter);
                __m256 vG_sum = _mm256_mul_ps(pack_2x2_box(vG_vlo, vG_vhi), v_quarter);
                __m256 vR_sum = _mm256_mul_ps(pack_2x2_box(vR_vlo, vR_vhi), v_quarter);

                __m256 v_Y_val, v_U_val, v_V_val;

                if (Traits::IsYUV)
                {
                    v_Y_val = vR_sum;
                    v_U_val = vG_sum;
                    v_V_val = vB_sum;
                }
                else
                {
                    v_Y_val = _mm256_add_ps(_mm256_mul_ps(v_y_r, vR_sum), _mm256_add_ps(_mm256_mul_ps(v_y_g, vG_sum), _mm256_mul_ps(v_y_b, vB_sum)));
                    v_U_val = _mm256_add_ps(_mm256_mul_ps(v_u_r, vR_sum), _mm256_add_ps(_mm256_mul_ps(v_u_g, vG_sum), _mm256_mul_ps(v_u_b, vB_sum)));
                    v_V_val = _mm256_add_ps(_mm256_mul_ps(v_v_r, vR_sum), _mm256_add_ps(_mm256_mul_ps(v_v_g, vG_sum), _mm256_mul_ps(v_v_b, vB_sum)));
                }

                CACHE_ALIGN float tY[8], tU[8], tV[8];
                _mm256_store_ps(tY, v_Y_val); _mm256_store_ps(tU, v_U_val); _mm256_store_ps(tV, v_V_val);

                for (A_long i = 0; i < remaining; ++i)
                {
                    row_Y[x_out + i] = tY[i]; row_U[x_out + i] = tU[i]; row_V[x_out + i] = tV[i];
                }
            }
        }
    }

    // ============================================================================
    // NORMAL MODE: 1:1 FULL RESOLUTION (IsHalfSize == false)
    // ============================================================================
    template <PixelFormat FMT>
    inline void convert_to_planar_impl
    (
        const void* RESTRICT srcBuf,
        const MemHandler& memHndl,
        const A_long width,
        const A_long height,
        const A_long stride_pixels,
        std::false_type /* IsHalfSize tag */
    ) noexcept
    {
        using Traits = PixelTraits<FMT>;
        using PixelType = typename Traits::DataType;

        const PixelType* RESTRICT in_pixels = reinterpret_cast<const PixelType*>(srcBuf);
        float* RESTRICT out_Y = memHndl.Y_planar;
        float* RESTRICT out_U = memHndl.U_planar;
        float* RESTRICT out_V = memHndl.V_planar;

        for (A_long y = 0; y < height; ++y)
        {
            const PixelType* in_row = in_pixels + (y * stride_pixels);
            float* row_Y = out_Y + (y * width);
            float* row_U = out_U + (y * width);
            float* row_V = out_V + (y * width);

            A_long x = 0;

            // 1. MAIN AVX2 LOOP
            for (; x <= width - 8; x += 8)
            {
                __m256 vB, vG, vR;
                Traits::LoadAVX2(in_row + x, vB, vG, vR);

                __m256 v_Y_val, v_U_val, v_V_val;

                if (Traits::IsYUV)
                {
                    v_Y_val = vR;
                    v_U_val = vG;
                    v_V_val = vB;
                }
                else
                {
                    v_Y_val = _mm256_add_ps(_mm256_mul_ps(v_y_r, vR), _mm256_add_ps(_mm256_mul_ps(v_y_g, vG), _mm256_mul_ps(v_y_b, vB)));
                    v_U_val = _mm256_add_ps(_mm256_mul_ps(v_u_r, vR), _mm256_add_ps(_mm256_mul_ps(v_u_g, vG), _mm256_mul_ps(v_u_b, vB)));
                    v_V_val = _mm256_add_ps(_mm256_mul_ps(v_v_r, vR), _mm256_add_ps(_mm256_mul_ps(v_v_g, vG), _mm256_mul_ps(v_v_b, vB)));
                }

                _mm256_storeu_ps(&row_Y[x], v_Y_val);
                _mm256_storeu_ps(&row_U[x], v_U_val);
                _mm256_storeu_ps(&row_V[x], v_V_val);
            }

            // 2. AVX2 PADDED TAIL
            const A_long remaining = width - x;
            if (remaining > 0)
            {
                CACHE_ALIGN PixelType tail_src[8] = {};
                for (A_long i = 0; i < remaining; ++i) { tail_src[i] = in_row[x + i]; }

                __m256 vB, vG, vR;
                Traits::LoadAVX2(tail_src, vB, vG, vR);

                __m256 v_Y_val, v_U_val, v_V_val;

                if (Traits::IsYUV)
                {
                    v_Y_val = vR;
                    v_U_val = vG;
                    v_V_val = vB;
                }
                else
                {
                    v_Y_val = _mm256_add_ps(_mm256_mul_ps(v_y_r, vR), _mm256_add_ps(_mm256_mul_ps(v_y_g, vG), _mm256_mul_ps(v_y_b, vB)));
                    v_U_val = _mm256_add_ps(_mm256_mul_ps(v_u_r, vR), _mm256_add_ps(_mm256_mul_ps(v_u_g, vG), _mm256_mul_ps(v_u_b, vB)));
                    v_V_val = _mm256_add_ps(_mm256_mul_ps(v_v_r, vR), _mm256_add_ps(_mm256_mul_ps(v_v_g, vG), _mm256_mul_ps(v_v_b, vB)));
                }

                CACHE_ALIGN float tY[8], tU[8], tV[8];
                _mm256_store_ps(tY, v_Y_val); _mm256_store_ps(tU, v_U_val); _mm256_store_ps(tV, v_V_val);

                for (A_long i = 0; i < remaining; ++i)
                {
                    row_Y[x + i] = tY[i]; row_U[x + i] = tU[i]; row_V[x + i] = tV[i];
                }
            }
        }
    }

} // end namespace detail

  // ============================================================================
  // THE PUBLIC TEMPLATE API
  // ============================================================================
template <PixelFormat FMT, bool IsHalfSize>
void convert_to_planar_AVX2
(
    const void* RESTRICT srcBuf,
    const MemHandler& memHndl,
    const A_long width,
    const A_long height,
    const A_long stride_pixels
) noexcept
{
    detail::convert_to_planar_impl<FMT>(
        srcBuf, memHndl, width, height, stride_pixels,
        std::integral_constant<bool, IsHalfSize>{}
    );
}