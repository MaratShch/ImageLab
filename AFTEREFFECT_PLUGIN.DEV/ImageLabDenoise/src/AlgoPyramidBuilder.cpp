#include "AlgoPyramidBuilder.hpp"


inline void Mean_Split_2x2
(
    const float* RESTRICT src, 
    float* RESTRICT dst, 
    const int32_t srcW, 
    const int32_t srcH
) noexcept
{
    const int32_t dstW = srcW / 2;
    const int32_t dstH = srcH / 2;
    constexpr float factor = 0.25f;
    
    for (int32_t y = 0; y < dstH; ++y) 
    {
        const int32_t srcY0 = y * 2;
        const int32_t srcY1 = srcY0 + 1;

        for (int32_t x = 0; x < dstW; ++x) 
        {
            const int32_t srcX0 = x * 2;
            const int32_t srcX1 = srcX0 + 1;

            const float sum = src[srcY0 * srcW + srcX0] + 
                              src[srcY0 * srcW + srcX1] +
                              src[srcY1 * srcW + srcX0] + 
                              src[srcY1 * srcW + srcX1];
            
            dst[y * dstW + x] = sum * factor;
        }
    }
}


inline void Mean_Built_2x2
(
    const float* RESTRICT src, 
    float* RESTRICT dst, 
    const int32_t dstW, 
    const int32_t dstH
) noexcept
{
    const int32_t srcW = dstW / 2;
    const int32_t srcH = dstH / 2;
    
    for (int32_t y = 0; y < srcH; ++y) 
    {
        const int32_t dstY0 = y * 2;
        const int32_t dstY1 = dstY0 + 1;

        for (int32_t x = 0; x < srcW; ++x) 
        {
            const float val = src[y * srcW + x];
            const int32_t dstX0 = x * 2;
            const int32_t dstX1 = dstX0 + 1;

            dst[dstY0 * dstW + dstX0] = val;
            dst[dstY0 * dstW + dstX1] = val;
            dst[dstY1 * dstW + dstX0] = val;
            dst[dstY1 * dstW + dstX1] = val;
        }
    }
}


void Build_Laplacian_Level
(
    const float* RESTRICT src_plane, 
    float* RESTRICT dst_downscaled, 
    float* RESTRICT dst_diff, 
    const int32_t srcW, 
    const int32_t srcH
)
{
    // 1. Downscale
    Mean_Split_2x2(src_plane, dst_downscaled, srcW, srcH);
    
    // 2. Upscale directly into diff buffer
    Mean_Built_2x2(dst_downscaled, dst_diff, srcW, srcH);
    
    // 3. Compute High-Frequency Difference: Diff = Original - Upscaled
    const int32_t frameSize = srcW * srcH;
    for (int32_t i = 0; i < frameSize; ++i) 
    {
        dst_diff[i] = src_plane[i] - dst_diff[i];
    }
}


void Build_Laplacian_Pyramid
(
    const MemHandler& mem, 
    const int32_t srcWidth, 
    const int32_t srcHeight
)
{
    // Level 1: Full -> Half
    Build_Laplacian_Level(mem.Y_planar, mem.Y_half, mem.Y_diff_full, srcWidth, srcHeight);
    Build_Laplacian_Level(mem.U_planar, mem.U_half, mem.U_diff_full, srcWidth, srcHeight);
    Build_Laplacian_Level(mem.V_planar, mem.V_half, mem.V_diff_full, srcWidth, srcHeight);

    // Level 2: Half -> Quarter
    const int32_t halfW = srcWidth / 2;
    const int32_t halfH = srcHeight / 2;
    
    Build_Laplacian_Level(mem.Y_half, mem.Y_quart, mem.Y_diff_half, halfW, halfH);
    Build_Laplacian_Level(mem.U_half, mem.U_quart, mem.U_diff_half, halfW, halfH);
    Build_Laplacian_Level(mem.V_half, mem.V_quart, mem.V_diff_half, halfW, halfH);
}


void Reconstruct_Laplacian_Level
(
    const float* RESTRICT src_base,
    const float* RESTRICT src_diff,
    float* RESTRICT dst_reconstructed,
    const int32_t dstW,
    const int32_t dstH
)
{
    // 1. Upscale the base image
    Mean_Built_2x2(src_base, dst_reconstructed, dstW, dstH);
    
    // 2. Add the difference (details) back
    const int32_t frameSize = dstW * dstH;
    for (int32_t i = 0; i < frameSize; ++i)
    {
        dst_reconstructed[i] += src_diff[i];
    }
}