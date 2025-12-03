#pragma once

#include "Common.hpp"

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

inline uint64_t RDTSC() noexcept
{
#ifdef _MSC_VER
	return __rdtsc();
#elif defined(__GNUC__) || defined(__clang__)
	uint32_t hi, lo;
	__asm__ volatile("rdtsc" : "=a" (lo), "=d" (hi));
	return ((uint64_t)hi << 32) | lo;
#else
    #error "RDTSC not supported on this compiler/platform"
#endif	
}


// ============================================================================
// HELPER: TILED MATRIX TRANSPOSE (Complex)
// ============================================================================
// Transposes a (Width x Height) matrix into a (Height x Width) matrix.
// Handles interleaved complex data (2 floats per element).
// Uses Tiling (Block Size 32) to maximize CPU Cache hits.
// ----------------------------------------------------------------------------
template <typename T>
inline void utils_transpose_complex_2d (const T* __restrict src, T* __restrict dst, int32_t width, int32_t height) noexcept
{
    constexpr int32_t TILE = 32;

    // Loop over blocks
    for (int32_t y = 0; y < height; y += TILE)
    {
        for (int32_t x = 0; x < width; x += TILE)
        {
            // Handle boundary clipping for non-multiple-of-32 sizes
            const int32_t y_end = (y + TILE < height) ? (y + TILE) : height;
            const int32_t x_end = (x + TILE < width)  ? (x + TILE) : width;

            // Process the block
            for (int32_t i = y; i < y_end; ++i)
            {
                __VECTORIZATION__
                for (int32_t j = x; j < x_end; ++j)
                {
                    const int32_t src_idx = 2 * (i * width + j);
                    const int32_t dst_idx = 2 * (j * height + i);

                    dst[dst_idx] = src[src_idx];     // Real
                    dst[dst_idx + 1] = src[src_idx + 1]; // Imag
                }
            }
        }
    }
    return;
}