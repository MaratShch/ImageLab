#include "AVX2_AlgoPyramidBuilder.hpp"

// =========================================================
// AVX2 DOWNSCALE (2x2 AVERAGING)
// =========================================================
inline void AVX2_Mean_Split_2x2
(
    const float* RESTRICT src, 
    float* RESTRICT dst, 
    const int32_t srcW, 
    const int32_t srcH
) noexcept
{
    const int32_t dstW = srcW / 2;
    const int32_t dstH = srcH / 2;
    __m128 vFactor = _mm_set1_ps(0.25f);
    
    for (int32_t y = 0; y < dstH; ++y) 
    {
        const int32_t srcY0 = y * 2;
        const int32_t srcY1 = srcY0 + 1;
        
        int32_t x = 0;
        // Process 8 input pixels (2 rows) to produce 4 output pixels
        for (; x <= dstW - 4; x += 4) 
        {
            __m256 r0 = _mm256_loadu_ps(&src[srcY0 * srcW + x * 2]);
            __m256 r1 = _mm256_loadu_ps(&src[srcY1 * srcW + x * 2]);
            __m256 sum256 = _mm256_add_ps(r0, r1);

            __m128 lo = _mm256_castps256_ps128(sum256);
            __m128 hi = _mm256_extractf128_ps(sum256, 1);

            // Horizontal add adjacent pixels: [a0+a1, a2+a3, a0+a1, a2+a3]
            __m128 hsum_lo = _mm_hadd_ps(lo, lo); 
            __m128 hsum_hi = _mm_hadd_ps(hi, hi);

            // Blend high and low correctly: [a0+a1, a2+a3, a4+a5, a6+a7]
            __m128 res = _mm_shuffle_ps(hsum_lo, hsum_hi, _MM_SHUFFLE(1, 0, 1, 0));
            
            res = _mm_mul_ps(res, vFactor);
            _mm_storeu_ps(&dst[y * dstW + x], res);
        }
        
        // Scalar Tail
        for (; x < dstW; ++x) 
        {
            const float sum = src[srcY0 * srcW + x * 2] + src[srcY0 * srcW + x * 2 + 1] +
                              src[srcY1 * srcW + x * 2] + src[srcY1 * srcW + x * 2 + 1];
            dst[y * dstW + x] = sum * 0.25f;
        }
    }
}

// =========================================================
// AVX2 UPSCALE (NEAREST NEIGHBOR DUPLICATION)
// =========================================================
inline void AVX2_Mean_Built_2x2
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
        
        int32_t x = 0;
        // Process 8 input pixels to produce 16 output pixels per row
        for (; x <= srcW - 8; x += 8) 
        {
            __m256 vSrc = _mm256_loadu_ps(&src[y * srcW + x]);

            __m128 lo = _mm256_castps256_ps128(vSrc);
            __m128 hi = _mm256_extractf128_ps(vSrc, 1);

            // Unpack duplicates the pixels inline: [p0, p0, p1, p1]
            __m128 dup_lo0 = _mm_unpacklo_ps(lo, lo);
            __m128 dup_lo1 = _mm_unpackhi_ps(lo, lo);
            __m128 dup_hi0 = _mm_unpacklo_ps(hi, hi);
            __m128 dup_hi1 = _mm_unpackhi_ps(hi, hi);

            __m256 out0 = _mm256_insertf128_ps(_mm256_castps128_ps256(dup_lo0), dup_lo1, 1);
            __m256 out1 = _mm256_insertf128_ps(_mm256_castps128_ps256(dup_hi0), dup_hi1, 1);

            // Blast identical data to both Row 0 and Row 1
            _mm256_storeu_ps(&dst[dstY0 * dstW + x * 2], out0);
            _mm256_storeu_ps(&dst[dstY0 * dstW + x * 2 + 8], out1);
            
            _mm256_storeu_ps(&dst[dstY1 * dstW + x * 2], out0);
            _mm256_storeu_ps(&dst[dstY1 * dstW + x * 2 + 8], out1);
        }
        
        // Scalar Tail
        for (; x < srcW; ++x) 
        {
            const float val = src[y * srcW + x];
            dst[dstY0 * dstW + x * 2] = val;
            dst[dstY0 * dstW + x * 2 + 1] = val;
            dst[dstY1 * dstW + x * 2] = val;
            dst[dstY1 * dstW + x * 2 + 1] = val;
        }
    }
}

// =========================================================
// AVX2 LAPLACIAN DIFFERENCE GENERATOR
// =========================================================
inline void AVX2_Build_Laplacian_Level
(
    const float* RESTRICT src_plane, 
    float* RESTRICT dst_downscaled, 
    float* RESTRICT dst_diff, 
    const int32_t srcW, 
    const int32_t srcH
)
{
    AVX2_Mean_Split_2x2(src_plane, dst_downscaled, srcW, srcH);
    AVX2_Mean_Built_2x2(dst_downscaled, dst_diff, srcW, srcH);
    
    // Vectorized Difference
    const int32_t frameSize = srcW * srcH;
    int32_t i = 0;
    for (; i <= frameSize - 8; i += 8) 
    {
        __m256 vSrc = _mm256_loadu_ps(&src_plane[i]);
        __m256 vDiff = _mm256_loadu_ps(&dst_diff[i]);
        _mm256_storeu_ps(&dst_diff[i], _mm256_sub_ps(vSrc, vDiff));
    }
    for (; i < frameSize; ++i) dst_diff[i] = src_plane[i] - dst_diff[i];
}

void AVX2_Build_Laplacian_Pyramid
(
    const MemHandler& mem, 
    const int32_t srcWidth, 
    const int32_t srcHeight
)
{
    AVX2_Build_Laplacian_Level(mem.Y_planar, mem.Y_half, mem.Y_diff_full, srcWidth, srcHeight);
    AVX2_Build_Laplacian_Level(mem.U_planar, mem.U_half, mem.U_diff_full, srcWidth, srcHeight);
    AVX2_Build_Laplacian_Level(mem.V_planar, mem.V_half, mem.V_diff_full, srcWidth, srcHeight);

    const int32_t halfW = srcWidth / 2;
    const int32_t halfH = srcHeight / 2;
    
    AVX2_Build_Laplacian_Level(mem.Y_half, mem.Y_quart, mem.Y_diff_half, halfW, halfH);
    AVX2_Build_Laplacian_Level(mem.U_half, mem.U_quart, mem.U_diff_half, halfW, halfH);
    AVX2_Build_Laplacian_Level(mem.V_half, mem.V_quart, mem.V_diff_half, halfW, halfH);
}

// =========================================================
// AVX2 LAPLACIAN RECONSTRUCTION
// =========================================================
void AVX2_Reconstruct_Laplacian_Level
(
    const float* RESTRICT src_base,
    const float* RESTRICT src_diff,
    float* RESTRICT dst_reconstructed,
    const int32_t dstW,
    const int32_t dstH
)
{
    AVX2_Mean_Built_2x2(src_base, dst_reconstructed, dstW, dstH);
    
    // Vectorized Addition
    const int32_t frameSize = dstW * dstH;
    int32_t i = 0;
    for (; i <= frameSize - 8; i += 8)
    {
        __m256 vRecon = _mm256_loadu_ps(&dst_reconstructed[i]);
        __m256 vDiff = _mm256_loadu_ps(&src_diff[i]);
        _mm256_storeu_ps(&dst_reconstructed[i], _mm256_add_ps(vRecon, vDiff));
    }
    for (; i < frameSize; ++i) dst_reconstructed[i] += src_diff[i];
}