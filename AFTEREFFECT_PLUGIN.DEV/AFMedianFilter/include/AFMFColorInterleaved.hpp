#pragma once
#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"
#include "AFMFPixelTraits.hpp"

// ============================================================================
// COMPILE-TIME FORMAT EVALUATOR
// Identifies formats that need 0.0f..255.0f scaled down to 0.0f..1.0f
// (This can be shared in AFMFPixelTraits.hpp, placed here for completeness)
// ============================================================================
#ifndef NEEDS_FLOAT_SCALING_DEFINED
#define NEEDS_FLOAT_SCALING_DEFINED
template <PixelFormat FMT>
constexpr bool NeedsFloatScaling() noexcept
{
    return FMT == PixelFormat::BGRA_32f || FMT == PixelFormat::BGRX_32f || 
           FMT == PixelFormat::BGRP_32f || FMT == PixelFormat::ARGB_32f || 
           FMT == PixelFormat::BGRA_32f_Linear || FMT == PixelFormat::BGRX_32f_Linear || 
           FMT == PixelFormat::BGRP_32f_Linear || 
           FMT == PixelFormat::VUYA_32f || FMT == PixelFormat::VUYP_32f || 
           FMT == PixelFormat::VUYA_32f_709 || FMT == PixelFormat::VUYP_32f_709;
}
#endif

// ============================================================================
// FULL RESOLUTION INTERLEAVED OUTPUT API
// ============================================================================
template <PixelFormat FMT>
void convert_to_interleaved_AVX2
(
    const MemHandler& memHndl,
    const void* RESTRICT origSrcBuf, // The original Adobe Input buffer (for Alpha)
    void* RESTRICT dstBuf,           // The final Adobe Output buffer
    const A_long width,
    const A_long height,
    const A_long srcLinePitch,       // origSrcBuf line pitch in elements
    const A_long dstLinePitch        // dstBuf line pitch in elements
) noexcept
{
    using Traits = PixelTraits<FMT>;
    using PixelType = typename Traits::DataType;

    const PixelType* RESTRICT orig_pixels = reinterpret_cast<const PixelType*>(origSrcBuf);
    PixelType* RESTRICT out_pixels        = reinterpret_cast<      PixelType*>(dstBuf);

    // Source Planar Buffers (Strict Routing: Y=R, U=G, V=B)
    const float* RESTRICT in_Y = memHndl.out_Y; // Red
    const float* RESTRICT in_U = memHndl.out_U; // Green
    const float* RESTRICT in_V = memHndl.out_V; // Blue

    const A_long planar_stride = memHndl.strideY_Elements;
    
    // Constant for scaling 32-bit floats back down to 0.0f ... 1.0f
    const __m256 v_scale_down = _mm256_set1_ps(1.0f / 255.0f);

    for (A_long y = 0; y < height; ++y)
    {
        // Safe physical memory bounds for the algorithm Arena
        const float* row_Y = in_Y + (y * planar_stride);
        const float* row_U = in_U + (y * planar_stride);
        const float* row_V = in_V + (y * planar_stride);

        // Safe pointer arithmetic based on Pixel pitch
        const PixelType* orig_row = orig_pixels + (y * srcLinePitch);
        PixelType* out_row        = out_pixels  + (y * dstLinePitch);

        A_long x = 0;

        // 1. MAIN AVX2 LOOP (8 pixels per iteration)
        for (; x <= width - 8; x += 8)
        {
            // Load clean, denoised planar RGB
            __m256 vR = _mm256_loadu_ps(&row_Y[x]);
            __m256 vG = _mm256_loadu_ps(&row_U[x]);
            __m256 vB = _mm256_loadu_ps(&row_V[x]);

            // Compile-time scaling for 32-bit float formats down to [0.0, 1.0]
            if (NeedsFloatScaling<FMT>())
            {
                vR = _mm256_mul_ps(vR, v_scale_down);
                vG = _mm256_mul_ps(vG, v_scale_down);
                vB = _mm256_mul_ps(vB, v_scale_down);
            }

            // Traits safely repack, inject original alpha, and re-premultiply if needed
            Traits::StoreAVX2(out_row + x, vB, vG, vR, orig_row + x);
        }

        // 2. AVX2 PADDED TAIL (The Scalar "Fraction")
        const A_long remaining = width - x;
        if (remaining > 0)
        {
            // Zero-initialized stack arrays (No new/malloc!)
            AVX2_ALIGN float tail_Y[8] = { 0 };
            AVX2_ALIGN float tail_U[8] = { 0 };
            AVX2_ALIGN float tail_V[8] = { 0 };
            AVX2_ALIGN PixelType tail_orig[8] = {};
            AVX2_ALIGN PixelType tail_out[8]  = {};

            // Safely copy fractional pixels into aligned memory
            for (A_long i = 0; i < remaining; ++i)
            {
                tail_Y[i]    = row_Y[x + i];
                tail_U[i]    = row_U[x + i];
                tail_V[i]    = row_V[x + i];
                tail_orig[i] = orig_row[x + i];
            }

            __m256 vR = _mm256_load_ps(tail_Y);
            __m256 vG = _mm256_load_ps(tail_U);
            __m256 vB = _mm256_load_ps(tail_V);

            if (NeedsFloatScaling<FMT>())
            {
                vR = _mm256_mul_ps(vR, v_scale_down);
                vG = _mm256_mul_ps(vG, v_scale_down);
                vB = _mm256_mul_ps(vB, v_scale_down);
            }

            Traits::StoreAVX2(tail_out, vB, vG, vR, tail_orig);

            // Write ONLY the valid remaining pixels back to Adobe's buffer
            for (A_long i = 0; i < remaining; ++i)
            {
                out_row[x + i] = tail_out[i];
            }
        }
    }
}