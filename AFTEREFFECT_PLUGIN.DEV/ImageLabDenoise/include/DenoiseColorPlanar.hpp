#pragma once

#include <type_traits>
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"
#include "DenoisePixelTraits.hpp"

// ============================================================================
// NOISE CLINIC ORTHONORMAL COEFFICIENTS 
// (Mathematically preserves L2 Euclidean Distances from RGB space)
// ============================================================================
static const __m256 v_ortho_y = _mm256_set1_ps(0.57735027f);
static const __m256 v_ortho_u = _mm256_set1_ps(0.70710678f);
static const __m256 v_ortho_v = _mm256_set1_ps(0.40824829f);
static const __m256 v_ortho_v_g = _mm256_set1_ps(-0.81649658f);

// ============================================================================
// FULL RESOLUTION PLANAR CONVERSION API
// ============================================================================
template <PixelFormat FMT>
void convert_to_planar_AVX2
(
    const void* RESTRICT srcBuf,
    const MemHandler& memHndl,
    const A_long width,
    const A_long height,
    const A_long stride_pixels
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

        // FIX: The internal algorithmic planar buffers have a row stride of padW, not width!
        float* row_Y = out_Y + (y * memHndl.padW);
        float* row_U = out_U + (y * memHndl.padW);
        float* row_V = out_V + (y * memHndl.padW);

        A_long x = 0;

        // 1. MAIN AVX2 LOOP (8 pixels per iteration)
        for (; x <= width - 8; x += 8)
        {
            __m256 vB, vG, vR;
            Traits::LoadAVX2(in_row + x, vB, vG, vR);

            __m256 v_Y_val = _mm256_mul_ps(_mm256_add_ps(vR, _mm256_add_ps(vG, vB)), v_ortho_y);
            __m256 v_U_val = _mm256_mul_ps(_mm256_sub_ps(vR, vB), v_ortho_u);
            __m256 v_V_val = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(vR, vB), v_ortho_v), _mm256_mul_ps(vG, v_ortho_v_g));

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

            __m256 v_Y_val = _mm256_mul_ps(_mm256_add_ps(vR, _mm256_add_ps(vG, vB)), v_ortho_y);
            __m256 v_U_val = _mm256_mul_ps(_mm256_sub_ps(vR, vB), v_ortho_u);
            __m256 v_V_val = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(vR, vB), v_ortho_v), _mm256_mul_ps(vG, v_ortho_v_g));

            CACHE_ALIGN float tY[8], tU[8], tV[8];
            _mm256_store_ps(tY, v_Y_val);
            _mm256_store_ps(tU, v_U_val);
            _mm256_store_ps(tV, v_V_val);

            for (A_long i = 0; i < remaining; ++i)
            {
                row_Y[x + i] = tY[i];
                row_U[x + i] = tU[i];
                row_V[x + i] = tV[i];
            }
        }

        // 3. SAFETY EDGE PADDING (Crucial for Non-Local Bayes)
        // If width is not divisible by 4, fill the remaining padded memory with the last valid pixel.
        // This stops the algorithm from reading uninitialized garbage memory near image edges.
        if (width > 0 && memHndl.padW > width)
        {
            float edge_Y = row_Y[width - 1];
            float edge_U = row_U[width - 1];
            float edge_V = row_V[width - 1];
            for (A_long p = width; p < memHndl.padW; ++p)
            {
                row_Y[p] = edge_Y;
                row_U[p] = edge_U;
                row_V[p] = edge_V;
            }
        }
    }
    return;
}