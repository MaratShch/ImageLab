#pragma once
#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"
#include "DenoisePixelTraits.hpp"

// ============================================================================
// INVERSE ORTHONORMAL COEFFICIENTS
// (Returns exact RGB values from the Orthonormal YUV space)
// ============================================================================
static const __m256 v_inv_y = _mm256_set1_ps(0.57735027f);
static const __m256 v_inv_u_rb = _mm256_set1_ps(0.70710678f);
static const __m256 v_inv_v_rb = _mm256_set1_ps(0.40824829f);
static const __m256 v_inv_v_g = _mm256_set1_ps(-0.81649658f);

// ============================================================================
// FULL RESOLUTION INTERLEAVED OUTPUT API
// ============================================================================
template <PixelFormat FMT>
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
    using Traits = PixelTraits<FMT>;
    using PixelType = typename Traits::DataType;

    const PixelType* RESTRICT orig_pixels = reinterpret_cast<const PixelType*>(origSrcBuf);
    PixelType* RESTRICT out_pixels = reinterpret_cast<      PixelType*>(dstBuf);

    // Using your correct fix: Read from Accumulators!
    const float* RESTRICT in_Y = memHndl.Accum_Y;
    const float* RESTRICT in_U = memHndl.Accum_U;
    const float* RESTRICT in_V = memHndl.Accum_V;

    for (A_long y = 0; y < height; ++y)
    {
        // FIX: The internal algorithmic planar buffers have a row stride of padW, not width!
        const float* row_Y = in_Y + (y * memHndl.padW);
        const float* row_U = in_U + (y * memHndl.padW);
        const float* row_V = in_V + (y * memHndl.padW);

        const PixelType* orig_row = orig_pixels + (y * srcLinePitch);
        PixelType* out_row = out_pixels + (y * dstLinePitch);

        A_long x = 0;

        // 1. MAIN AVX2 LOOP (8 pixels per iteration)
        for (; x <= width - 8; x += 8)
        {
            __m256 vY = _mm256_loadu_ps(&row_Y[x]);
            __m256 vU = _mm256_loadu_ps(&row_U[x]);
            __m256 vV = _mm256_loadu_ps(&row_V[x]);

            __m256 vY_base = _mm256_mul_ps(vY, v_inv_y);
            __m256 vU_rb = _mm256_mul_ps(vU, v_inv_u_rb);
            __m256 vV_rb = _mm256_mul_ps(vV, v_inv_v_rb);

            __m256 vR = _mm256_add_ps(vY_base, _mm256_add_ps(vU_rb, vV_rb));
            __m256 vG = _mm256_add_ps(vY_base, _mm256_mul_ps(vV, v_inv_v_g));
            __m256 vB = _mm256_add_ps(vY_base, _mm256_sub_ps(vV_rb, vU_rb));

            Traits::StoreAVX2(out_row + x, vB, vG, vR, orig_row + x);
        }

        // 2. AVX2 PADDED TAIL
        const A_long remaining = width - x;
        if (remaining > 0)
        {
            CACHE_ALIGN float tail_Y[8] = { 0 }, tail_U[8] = { 0 }, tail_V[8] = { 0 };
            CACHE_ALIGN PixelType tail_orig[8] = {}, tail_out[8] = {};

            for (A_long i = 0; i < remaining; ++i)
            {
                tail_Y[i] = row_Y[x + i];
                tail_U[i] = row_U[x + i];
                tail_V[i] = row_V[x + i];
                tail_orig[i] = orig_row[x + i];
            }

            __m256 vY = _mm256_load_ps(tail_Y);
            __m256 vU = _mm256_load_ps(tail_U);
            __m256 vV = _mm256_load_ps(tail_V);

            __m256 vY_base = _mm256_mul_ps(vY, v_inv_y);
            __m256 vU_rb = _mm256_mul_ps(vU, v_inv_u_rb);
            __m256 vV_rb = _mm256_mul_ps(vV, v_inv_v_rb);

            __m256 vR = _mm256_add_ps(vY_base, _mm256_add_ps(vU_rb, vV_rb));
            __m256 vG = _mm256_add_ps(vY_base, _mm256_mul_ps(vV, v_inv_v_g));
            __m256 vB = _mm256_add_ps(vY_base, _mm256_sub_ps(vV_rb, vU_rb));

            Traits::StoreAVX2(tail_out, vB, vG, vR, tail_orig);
            for (A_long i = 0; i < remaining; ++i)
            {
                out_row[x + i] = tail_out[i];
            }
        }
    }
    return;
}