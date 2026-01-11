#include <immintrin.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include "Common.hpp"
#include "ArtPointillismEnums.hpp"
#include "AlgoArtisticsRendering.hpp"

/**
 * AVX2 Optimized Canvas Initialization
 * Handles the "Zip" operation (L + ab -> Lab) efficiently.
 */
void Init_Canvas
(
    float* RESTRICT canvas_lab,
    const float* RESTRICT src_L,
    const float* RESTRICT src_ab,
    int32_t width,
	int32_t height, 
    const BackgroundArt bg_mode
)
{
    int32_t num_pixels = width * height;
    int32_t i = 0;

    // Mode 2: SOURCE IMAGE (Complex Interleaving)
    if (BackgroundArt::ART_POINTILLISM_BACKGROUND_SOURCE_IMAGE == bg_mode)
	{ // ART_POINTILLISM_BACKGROUND_SOURCE_IMAGE
        for (; i <= num_pixels - 8; i += 8)
		{
            // 1. Load 8 Luma values (Planar) -> [L0 L1 ... L7]
            __m256 L = _mm256_loadu_ps(src_L + i);

            // 2. Load 8 Chroma pairs (Interleaved) -> [a0 b0 a1 b1 ... a7 b7]
            // We load 16 floats (2 registers)
            __m256 ab_lo = _mm256_loadu_ps(src_ab + i * 2);     // Px 0-3
            __m256 ab_hi = _mm256_loadu_ps(src_ab + i * 2 + 8); // Px 4-7

            // 3. De-interleave 'ab' into 'a' and 'b' planar vectors
            // Shuffle logic:
            // Permute 0x20: Low 128 of A, Low 128 of B
            // Permute 0x31: High 128 of A, High 128 of B
            
            // Step A: Permute within 256-bit lanes to group a's and b's locally
            // Input: a0 b0 a1 b1 | a2 b2 a3 b3
            // Shuff: a0 a1 b0 b1 | a2 a3 b2 b3 (Mask: 0 2 1 3 -> 0xD8) 
            __m256 t0 = _mm256_permute_ps(ab_lo, 0xD8);
            __m256 t1 = _mm256_permute_ps(ab_hi, 0xD8);

            // Step B: Unpack/Permute across lanes
            // We want A: a0 a1 a2 a3 | a4 a5 a6 a7
            // We want B: b0 b1 b2 b3 | b4 b5 b6 b7
            
            // Extract 'a' parts: Low 64 of t0, High 64 of t0... mixed.
            // Let's use the standard "Tricky" sequence for Split->Interleave logic in reverse?
            // Actually, manual shuffle might be clearer.
            // Let's go simple: Gather? No, too slow.
            
            // Revisit De-interleave logic from previous output function:
            __m256 i_a0 = _mm256_permutevar8x32_ps(ab_lo, _mm256_setr_epi32(0,2,4,6, 1,3,5,7));
            __m256 i_a1 = _mm256_permutevar8x32_ps(ab_hi, _mm256_setr_epi32(0,2,4,6, 1,3,5,7));
            __m256 a_plane = _mm256_permute2f128_ps(i_a0, i_a1, 0x20);
            __m256 b_plane = _mm256_permute2f128_ps(i_a0, i_a1, 0x31);

            // 4. Interleave L, a, b into [L a b L a b...]
            // We need to write 24 floats.
            // Since AVX2 doesn't have Scatter, we spill to stack or use scalar writes.
            // Stack spill is fastest L1 op.
            CACHE_ALIGN float tmp[24];
            _mm256_storeu_ps(tmp, L);
            _mm256_storeu_ps(tmp+8, a_plane);
            _mm256_storeu_ps(tmp+16, b_plane);

            float* dst = canvas_lab + i * 3;
            for(int32_t k = 0; k < 8; ++k)
			{
                dst[k*3+0] = tmp[k];
                dst[k*3+1] = tmp[k+8];
                dst[k*3+2] = tmp[k+16];
            }
        }
    } 
    // Mode 0/1: SOLID COLOR (Simple Fill)
    else
	{
        float fill_L = 96.0f, fill_a = 2.0f, fill_b = 8.0f; // Canvas
        if (BackgroundArt::ART_POINTILLISM_BACKGROUND_WHITE == bg_mode)
		{ 
			fill_L = 100.0f; 
			fill_a = 0.0f;
			fill_b = 0.0f;
		} // White
        
        // Create 3 constant vectors
        // But we need to write interleaved.
        // Pattern repeats every 3 floats: L a b L a b L a b ...
        // We can create a permutation of this pattern.
        // It's actually faster to just run scalar loop for fill, or memset if 0.
        // Optimization: Unroll scalar.
        for (; i < num_pixels; ++i)
		{
            canvas_lab[i*3+0] = fill_L;
            canvas_lab[i*3+1] = fill_a;
            canvas_lab[i*3+2] = fill_b;
        }
        return; // Done
    }

    // Scalar Cleanup
    for (; i < num_pixels; ++i)
	{
        canvas_lab[i*3+0] = src_L[i];
        canvas_lab[i*3+1] = src_ab[i*2+0];
        canvas_lab[i*3+2] = src_ab[i*2+1];
    }
	
	return;
}