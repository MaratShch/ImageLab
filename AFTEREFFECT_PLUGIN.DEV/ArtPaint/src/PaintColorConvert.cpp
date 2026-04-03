#include "PaintColorConvert.hpp"


void convert_BGRA8_to_PlanarYUV_AVX2
(
    const PF_Pixel_BGRA_8u* RESTRICT srcBuf,
    const MemHandler& memHandler,
    const A_long width,
    const A_long height,
    const A_long stride_pixels
) noexcept
{
    float* RESTRICT out_Y = memHandler.Y_planar;
    float* RESTRICT out_U = memHandler.U_planar;
    float* RESTRICT out_V = memHandler.V_planar;

    const uint8_t* RESTRICT in_bgra = reinterpret_cast<const uint8_t*>(srcBuf);
    
    // Convert elements to bytes internally for the uint8_t pointer math (1 pixel = 4 bytes)
    const A_long stride_bytes = stride_pixels * 4;

    // Rec.709 coefficients (Scaled for 0-255 float range)
    const __m256 v_y_r = _mm256_set1_ps(0.2126f);
    const __m256 v_y_g = _mm256_set1_ps(0.7152f);
    const __m256 v_y_b = _mm256_set1_ps(0.0722f);

    const __m256 v_u_r = _mm256_set1_ps(-0.114572f);
    const __m256 v_u_g = _mm256_set1_ps(-0.385428f);
    const __m256 v_u_b = _mm256_set1_ps(0.5f);

    const __m256 v_v_r = _mm256_set1_ps(0.5f);
    const __m256 v_v_g = _mm256_set1_ps(-0.454153f);
    const __m256 v_v_b = _mm256_set1_ps(-0.045847f);

    const __m256i v_mask_8bit = _mm256_set1_epi32(0x000000FF);

    for (A_long y = 0; y < height; ++y)
    {
        const uint8_t* row_bgra = in_bgra + (y * stride_bytes);
        float* row_Y = out_Y + (y * width);
        float* row_U = out_U + (y * width);
        float* row_V = out_V + (y * width);

        A_long x = 0;

        for (; x <= width - 8; x += 8)
        {
            __m256i v_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_bgra + (x * 4)));

            __m256i v_b_int = _mm256_and_si256(v_pixels, v_mask_8bit);
            __m256i v_g_int = _mm256_and_si256(_mm256_srli_epi32(v_pixels, 8), v_mask_8bit);
            __m256i v_r_int = _mm256_and_si256(_mm256_srli_epi32(v_pixels, 16), v_mask_8bit);

            __m256 v_b = _mm256_cvtepi32_ps(v_b_int);
            __m256 v_g = _mm256_cvtepi32_ps(v_g_int);
            __m256 v_r = _mm256_cvtepi32_ps(v_r_int);

            __m256 v_Y_val = _mm256_add_ps(_mm256_mul_ps(v_y_r, v_r), _mm256_add_ps(_mm256_mul_ps(v_y_g, v_g), _mm256_mul_ps(v_y_b, v_b)));
            __m256 v_U_val = _mm256_add_ps(_mm256_mul_ps(v_u_r, v_r), _mm256_add_ps(_mm256_mul_ps(v_u_g, v_g), _mm256_mul_ps(v_u_b, v_b)));
            __m256 v_V_val = _mm256_add_ps(_mm256_mul_ps(v_v_r, v_r), _mm256_add_ps(_mm256_mul_ps(v_v_g, v_g), _mm256_mul_ps(v_v_b, v_b)));

            _mm256_storeu_ps(&row_Y[x], v_Y_val);
            _mm256_storeu_ps(&row_U[x], v_U_val);
            _mm256_storeu_ps(&row_V[x], v_V_val);
        }

        // Scalar Tail handling
        for (; x < width; ++x)
        {
            const uint8_t* p = row_bgra + (x * 4);
            float b = static_cast<float>(p[0]);
            float g = static_cast<float>(p[1]);
            float r = static_cast<float>(p[2]);

            row_Y[x] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            row_U[x] = -0.114572f * r - 0.385428f * g + 0.5f * b;
            row_V[x] = 0.5f * r - 0.454153f * g - 0.045847f * b;
        }
    }
}


void convert_PlanarYUV_to_BGRA8_AVX2
(
    const MemHandler& memHandler,
    const PF_Pixel_BGRA_8u* RESTRICT srcBuf,
          PF_Pixel_BGRA_8u* RESTRICT dstBuf,
    const A_long width,
    const A_long height,
    const A_long src_stride_pixels,
    const A_long dst_stride_pixels
) noexcept
{
    const float* RESTRICT in_Y = memHandler.Y_planar;
    const float* RESTRICT in_U = memHandler.U_planar;
    const float* RESTRICT in_V = memHandler.V_planar;

    const uint8_t* RESTRICT in_bgra  = reinterpret_cast<const uint8_t*>(srcBuf);
          uint8_t* RESTRICT out_bgra = reinterpret_cast<      uint8_t*>(dstBuf);
    
    // Convert elements to bytes internally for the uint8_t pointer math (1 pixel = 4 bytes)
    const A_long src_stride_bytes = src_stride_pixels * 4;
    const A_long dst_stride_bytes = dst_stride_pixels * 4;

    // Inverse Rec.709 coefficients
    const __m256 v_r_v = _mm256_set1_ps(1.5748f);
    const __m256 v_g_u = _mm256_set1_ps(-0.187324f);
    const __m256 v_g_v = _mm256_set1_ps(-0.468124f);
    const __m256 v_b_u = _mm256_set1_ps(1.8556f);

    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255  = _mm256_set1_ps(255.0f);
    const __m256i v_alpha_mask = _mm256_set1_epi32(static_cast<int>(0xFF000000)); 

    for (A_long y = 0; y < height; ++y)
    {
        const float* row_Y = in_Y + (y * width);
        const float* row_U = in_U + (y * width);
        const float* row_V = in_V + (y * width);
        const uint8_t* row_in_bgra = in_bgra + (y * src_stride_bytes);
        uint8_t* row_out_bgra = out_bgra + (y * dst_stride_bytes);

        A_long x = 0;

        for (; x <= width - 8; x += 8)
        {
            __m256 v_Y_val = _mm256_loadu_ps(&row_Y[x]);
            __m256 v_U_val = _mm256_loadu_ps(&row_U[x]);
            __m256 v_V_val = _mm256_loadu_ps(&row_V[x]);

            __m256i v_orig_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_in_bgra + (x * 4)));
            __m256i v_alpha = _mm256_and_si256(v_orig_pixels, v_alpha_mask);

            __m256 v_r = _mm256_add_ps(v_Y_val, _mm256_mul_ps(v_r_v, v_V_val));
            __m256 v_g = _mm256_add_ps(v_Y_val, _mm256_add_ps(_mm256_mul_ps(v_g_u, v_U_val), _mm256_mul_ps(v_g_v, v_V_val)));
            __m256 v_b = _mm256_add_ps(v_Y_val, _mm256_mul_ps(v_b_u, v_U_val));

            v_r = _mm256_max_ps(v_zero, _mm256_min_ps(v_r, v_255));
            v_g = _mm256_max_ps(v_zero, _mm256_min_ps(v_g, v_255));
            v_b = _mm256_max_ps(v_zero, _mm256_min_ps(v_b, v_255));

            __m256i v_r_int = _mm256_cvtps_epi32(v_r);
            __m256i v_g_int = _mm256_cvtps_epi32(v_g);
            __m256i v_b_int = _mm256_cvtps_epi32(v_b);

            __m256i v_pixels = _mm256_or_si256(
                _mm256_or_si256(v_b_int, _mm256_slli_epi32(v_g_int, 8)),
                _mm256_or_si256(_mm256_slli_epi32(v_r_int, 16), v_alpha) 
            );

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(row_out_bgra + (x * 4)), v_pixels);
        }

        // Scalar Tail handling
        for (; x < width; ++x)
        {
            float Y_val = row_Y[x];
            float U_val = row_U[x];
            float V_val = row_V[x];

            float r = Y_val + 1.5748f * V_val;
            float g = Y_val - 0.187324f * U_val - 0.468124f * V_val;
            float b = Y_val + 1.8556f * U_val;

            r = std::max(0.0f, std::min(r, 255.0f));
            g = std::max(0.0f, std::min(g, 255.0f));
            b = std::max(0.0f, std::min(b, 255.0f));

            const uint8_t* p_in = row_in_bgra + (x * 4);
            uint8_t* p_out = row_out_bgra + (x * 4);
            
            p_out[0] = static_cast<uint8_t>(b);
            p_out[1] = static_cast<uint8_t>(g);
            p_out[2] = static_cast<uint8_t>(r);
            p_out[3] = p_in[3];
        }
    }
}