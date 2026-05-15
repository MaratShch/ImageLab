#pragma once
#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "AlgoMemHandler.hpp"
#include "AFMFPixelTraits.hpp"

// ============================================================================
// COMPILE-TIME FORMAT EVALUATOR
// Identifies formats that need 0.0f..1.0f scaled up to 0.0f..255.0f
// ============================================================================
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

// ============================================================================
// FULL RESOLUTION PLANAR RGB INGEST API
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

    // Target Planar Buffers (Strict Routing: Y=R, U=G, V=B)
    float* RESTRICT out_Y = memHndl.proc_Y; // Red
    float* RESTRICT out_U = memHndl.proc_U; // Green
    float* RESTRICT out_V = memHndl.proc_V; // Blue

    const A_long planar_stride = memHndl.strideY_Elements;

    for (A_long y = 0; y < height; ++y)
    {
        // Calculate row pointers using PIXEL pitches
        const PixelType* in_row = in_pixels + (y * stride_pixels);
        
        float* row_Y = out_Y + (y * planar_stride);
        float* row_U = out_U + (y * planar_stride);
        float* row_V = out_V + (y * planar_stride);

        A_long x = 0;

        // 1. MAIN AVX2 LOOP (8 pixels per iteration)
        for (; x <= width - 8; x += 8)
        {
            __m256 vB, vG, vR;
            
            // Load perceptual RGB from the trait
            Traits::LoadAVX2(in_row + x, vB, vG, vR);

            // Compile-time scaling for 32-bit floats
            if (NeedsFloatScaling<FMT>())
            {
                vB = _mm256_mul_ps(vB, v_255);
                vG = _mm256_mul_ps(vG, v_255);
                vR = _mm256_mul_ps(vR, v_255);
            }

            // Route strictly to Y=R, U=G, V=B
            _mm256_storeu_ps(&row_Y[x], vR);
            _mm256_storeu_ps(&row_U[x], vG);
            _mm256_storeu_ps(&row_V[x], vB);
        }

        // 2. AVX2 PADDED TAIL (The Scalar "Fraction")
        const A_long remaining = width - x;
        if (remaining > 0)
        {
            // Zero-initialized to prevent garbage data causing FP math penalties
            AVX2_ALIGN PixelType tail_in[8] = {};
            
            // CACHE_ALIGN stack arrays replacing malloc/new
            AVX2_ALIGN float tail_Y[8];
            AVX2_ALIGN float tail_U[8];
            AVX2_ALIGN float tail_V[8];

            // Safely copy remaining pixels into the aligned stack buffer
            for (A_long i = 0; i < remaining; ++i)
            {
                tail_in[i] = in_row[x + i];
            }

            __m256 vB, vG, vR;
            Traits::LoadAVX2(tail_in, vB, vG, vR);

            if (NeedsFloatScaling<FMT>())
            {
                vB = _mm256_mul_ps(vB, v_255);
                vG = _mm256_mul_ps(vG, v_255);
                vR = _mm256_mul_ps(vR, v_255);
            }

            // Store AVX2 vectors to our safe aligned stack arrays
            _mm256_store_ps(tail_Y, vR);
            _mm256_store_ps(tail_U, vG);
            _mm256_store_ps(tail_V, vB);

            // Write ONLY the valid remaining pixels back to the main memory Arena
            for (A_long i = 0; i < remaining; ++i)
            {
                row_Y[x + i] = tail_Y[i];
                row_U[x + i] = tail_U[i];
                row_V[x + i] = tail_V[i];
            }
        }
    }
}