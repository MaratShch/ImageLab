#include "AssembleFinalImage.hpp"


/**
 * STEP 1: BLENDING & REFORMATTING (AVX2)
 * 
 * Mixes Canvas with Source based on Opacity.
 * Converts Memory Layout: Interleaved (Canvas) -> Semi-Planar (Dest).
 */
void AssembleFinalImage
(
    const float* RESTRICT canvas_lab,   // [Input] Interleaved (L a b ...)
    const float* RESTRICT src_L,        // [Input] Planar L
    const float* RESTRICT src_ab,       // [Input] Interleaved (a b ...)
    float*       RESTRICT dst_L,        // [Output] Planar L
    float*       RESTRICT dst_ab,       // [Output] Interleaved (a b ...)
    int32_t sizeX, 
    int32_t sizeY, 
    const PontillismControls& params
) noexcept
{
    const int32_t num_pixels = sizeX * sizeY;

    // --- SETUP OPACITY ---
    float opacity = (float)params.Opacity;
    if (opacity < 0.0f) opacity = 0.0f;
    if (opacity > 100.0f) opacity = 100.0f;

    const float f_src_ratio = opacity / 100.0f;
    const float f_can_ratio = 1.0f - f_src_ratio;

    const __m256 v_src_ratio = _mm256_set1_ps(f_src_ratio);
    const __m256 v_can_ratio = _mm256_set1_ps(f_can_ratio);

    // Indices for Gathering L, a, b from Canvas (Stride 3)
    const __m256i idx_L = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256i idx_a = _mm256_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22);
    const __m256i idx_b = _mm256_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23);

    int i = 0;
    // --- AVX2 LOOP (8 pixels) ---
    for (; i <= num_pixels - 8; i += 8)
    {
        
        // 1. GATHER CANVAS (Interleaved -> Planar)
        // Replaces "AVX2_Deinterleave_3_Floats"
        const float* p_can = canvas_lab + i * 3;
        
        __m256 c_L = _mm256_i32gather_ps(p_can, idx_L, 4);
        __m256 c_a = _mm256_i32gather_ps(p_can, idx_a, 4);
        __m256 c_b = _mm256_i32gather_ps(p_can, idx_b, 4);

        // 2. LOAD SOURCE (Semi-Planar)
        __m256 s_L = _mm256_loadu_ps(src_L + i);
        
        // Load AB (Interleaved a b a b...) -> Needs de-interleaving
        __m256 s_ab0 = _mm256_loadu_ps(src_ab + i * 2);     // Pixels 0-3
        __m256 s_ab1 = _mm256_loadu_ps(src_ab + i * 2 + 8); // Pixels 4-7

        // De-interleave AB (2-channel shuffle)
        // Group 'a's and 'b's locally in 256-bit lanes
        __m256 s_a_perm0 = _mm256_permutevar8x32_ps(s_ab0, _mm256_setr_epi32(0,2,4,6, 1,3,5,7));
        __m256 s_a_perm1 = _mm256_permutevar8x32_ps(s_ab1, _mm256_setr_epi32(0,2,4,6, 1,3,5,7));
        
        // Merge lanes to get full 'a' vector and 'b' vector
        __m256 s_a = _mm256_permute2f128_ps(s_a_perm0, s_a_perm1, 0x20); // Low 128 of both
        __m256 s_b = _mm256_permute2f128_ps(s_a_perm0, s_a_perm1, 0x31); // High 128 of both

        // 3. BLEND
        // Res = (Canvas * (1-Opacity)) + (Source * Opacity)
        __m256 res_L = _mm256_fmadd_ps(c_L, v_can_ratio, _mm256_mul_ps(s_L, v_src_ratio));
        __m256 res_a = _mm256_fmadd_ps(c_a, v_can_ratio, _mm256_mul_ps(s_a, v_src_ratio));
        __m256 res_b = _mm256_fmadd_ps(c_b, v_can_ratio, _mm256_mul_ps(s_b, v_src_ratio));

        // 4. STORE
        // Store L (Planar) - Direct write
        _mm256_storeu_ps(dst_L + i, res_L);

        // Store AB (Interleaved) - Re-interleave
        // We use unpack to zip 'a' and 'b' back together: a0 b0 a1 b1...
        __m256 out_ab0_lane = _mm256_unpacklo_ps(res_a, res_b);
        __m256 out_ab1_lane = _mm256_unpackhi_ps(res_a, res_b);

        // Fix lane crossing (AVX unpacks stay within 128-bit lanes)
        // Out0: [a0 b0 a1 b1 a2 b2 a3 b3]
        __m256 out_ab0 = _mm256_permute2f128_ps(out_ab0_lane, out_ab1_lane, 0x20);
        // Out1: [a4 b4 a5 b5 a6 b6 a7 b7]
        __m256 out_ab1 = _mm256_permute2f128_ps(out_ab0_lane, out_ab1_lane, 0x31);

        _mm256_storeu_ps(dst_ab + i * 2, out_ab0);
        _mm256_storeu_ps(dst_ab + i * 2 + 8, out_ab1);
    }

    // --- SCALAR FALLBACK ---
    for (; i < num_pixels; ++i)
    {
        float cL = canvas_lab[i*3+0];
        float ca = canvas_lab[i*3+1];
        float cb = canvas_lab[i*3+2];

        float sL = src_L[i];
        float sa = src_ab[i*2+0];
        float sb = src_ab[i*2+1];

        dst_L[i]       = (cL * f_can_ratio) + (sL * f_src_ratio);
        dst_ab[i*2+0]  = (ca * f_can_ratio) + (sa * f_src_ratio);
        dst_ab[i*2+1]  = (cb * f_can_ratio) + (sb * f_src_ratio);
    }
    
    return;
}