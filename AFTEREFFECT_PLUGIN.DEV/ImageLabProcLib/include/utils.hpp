#pragma once

#include "Common.hpp"

// ============================================================================
// HELPER: TILED MATRIX TRANSPOSE (Complex)
// ============================================================================
// Transposes a (Width x Height) matrix into a (Height x Width) matrix.
// Handles interleaved complex data (2 floats per element).
// Uses Tiling (Block Size 32) to maximize CPU Cache hits.
// ----------------------------------------------------------------------------
template <typename T>
struct ComplexPair { T r; T i; };

template <typename T>
inline void utils_transpose_complex_2d
(
    const T* RESTRICT src_raw,
    T* RESTRICT dst_raw,
    ptrdiff_t width,
    ptrdiff_t height
) noexcept
{
    constexpr ptrdiff_t TILE = 32;

    const auto* src = reinterpret_cast<const ComplexPair<T>*>(src_raw);
    auto*       dst = reinterpret_cast<ComplexPair<T>*>(dst_raw);

    for (ptrdiff_t y = 0; y < height; y += TILE)
    {
        for (ptrdiff_t x = 0; x < width; x += TILE)
        {
            const ptrdiff_t block_h = std::min(TILE, height - y);
            const ptrdiff_t block_w = std::min(TILE, width - x);

            const auto* src_block_base = src + y * width + x;
            auto* dst_block_base = dst + x * height + y;

            for (ptrdiff_t i = 0; i < block_h; ++i)
            {
                const auto* src_row = src_block_base + i * width;

                auto* dst_col = dst_block_base + i;

                for (ptrdiff_t j = 0; j < block_w; ++j)
                {
                    dst_col[j * height] = src_row[j];
                }
            }
        }
    }
}