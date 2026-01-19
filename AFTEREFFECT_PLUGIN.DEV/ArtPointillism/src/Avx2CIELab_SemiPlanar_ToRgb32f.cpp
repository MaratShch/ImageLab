#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "Common.hpp"
#include "Avx2ColorConverts.hpp"


void AVX2_ConvertCIELab_SemiPlanar_ToRgb
(
    const PF_Pixel_BGRA_32f* RESTRICT pSrc, // For Alpha
    const float*             RESTRICT pL,   // Planar L
    const float*             RESTRICT pAB,  // Interleaved AB
    PF_Pixel_BGRA_32f*       RESTRICT pDst, // Output Buffer
    int32_t sizeX, 
    int32_t sizeY,
    int32_t srcPitch, // Pitch in PIXELS
    int32_t dstPitch  // Pitch in PIXELS 
) noexcept
{
    // Pitch in Bytes (4 floats * 4 bytes = 16 bytes per pixel)
    const intptr_t srcStrideBytes = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_BGRA_32f);
    const intptr_t dstStrideBytes = static_cast<intptr_t>(dstPitch) * sizeof(PF_Pixel_BGRA_32f);

    // Clamping Constants
    const __m256 v_zero = _mm256_setzero_ps();
    // Max = 1.0 - epsilon (To avoid hitting exactly 1.0 if strictly required, or just 1.0)
    // Using 1.0f - FLT_EPSILON as requested.
    const __m256 v_max  = _mm256_set1_ps(1.0f - FLT_EPSILON);

    // Alpha Gather Indices for BGRA (A is at float index 3)
    // 3, 7, 11, 15, 19, 23, 27, 31
    const __m256i idx_alpha = _mm256_setr_epi32(3, 7, 11, 15, 19, 23, 27, 31);

    for (int y = 0; y < sizeY; ++y)
    {
        const uint8_t* pRowSrc = reinterpret_cast<const uint8_t*>(pSrc) + (y * srcStrideBytes);
        uint8_t*       pRowDst = reinterpret_cast<uint8_t*>(pDst) + (y * dstStrideBytes);
        
        const float* rowL  = pL + (y * sizeX);
        const float* rowAB = pAB + (y * sizeX * 2);

        int x = 0;
        // Process 8 pixels per loop (8 * 16 bytes = 128 bytes output)
        for (; x <= sizeX - 8; x += 8)
        {
            // 1. LOAD LAB
            __m256 L = _mm256_loadu_ps(rowL + x);
            __m256 ab0 = _mm256_loadu_ps(rowAB + x * 2);
            __m256 ab1 = _mm256_loadu_ps(rowAB + x * 2 + 8);

            // De-interleave AB
            const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256 s0 = _mm256_permutevar8x32_ps(ab0, perm_idx);
            __m256 s1 = _mm256_permutevar8x32_ps(ab1, perm_idx);
            __m256 a = _mm256_permute2f128_ps(s0, s1, 0x20);
            __m256 b = _mm256_permute2f128_ps(s0, s1, 0x31);

            // 2. CONVERT Lab -> Linear RGB
            __m256 R, G, B;
            AVX2_Lab_to_RGB_Linear_Inline(L, a, b, R, G, B);

            // 3. CLAMP (0.0 .. 1.0-EPS)
            R = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, R));
            G = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, G));
            B = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, B));

            // 4. GATHER ALPHA
            const float* src_ptr = reinterpret_cast<const float*>(pRowSrc + x * 16);
            __m256 A = _mm256_i32gather_ps(src_ptr, idx_alpha, 4);

            // 5. TRANSPOSE / PACK (Planar -> Interleaved BGRA)
            // We have Planar: B, G, R, A.
            // Target: B0 G0 R0 A0 | B1 G1 R1 A1 ...
            
            // Step A: Unpack Lo/Hi (B, G) -> BG (Interleaved Pairs)
            __m256 bg_lo = _mm256_unpacklo_ps(B, G); // B0 G0 B1 G1 | B4 G4 B5 G5
            __m256 bg_hi = _mm256_unpackhi_ps(B, G); // B2 G2 B3 G3 | B6 G6 B7 G7

            // Step B: Unpack Lo/Hi (R, A) -> RA (Interleaved Pairs)
            __m256 ra_lo = _mm256_unpacklo_ps(R, A); // R0 A0 R1 A1 | R4 A4 R5 A5
            __m256 ra_hi = _mm256_unpackhi_ps(R, A); // R2 A2 R3 A3 | R6 A6 R7 A7

            // Step C: Cast to Double (64-bit) and Unpack to merge (BG, RA)
            // Result: [B0 G0 R0 A0] (128-bit)
            __m256d row0_d = _mm256_unpacklo_pd(_mm256_castps_pd(bg_lo), _mm256_castps_pd(ra_lo)); // Px 0, 1 (Lo), Px 4, 5 (Hi)
            __m256d row1_d = _mm256_unpackhi_pd(_mm256_castps_pd(bg_lo), _mm256_castps_pd(ra_lo)); // Px 1?? No unpackhi gets the high 64 bits.
            // wait, unpacklo_ps gives [B0 G0 | B1 G1]. 
            // 64-bit cast sees [B0 G0] as one double.
            // unpacklo_pd takes [B0 G0] and [R0 A0]. Result: [B0 G0 R0 A0] | [B4 G4 R4 A4].
            // unpackhi_pd takes [B1 G1] and [R1 A1]. Result: [B1 G1 R1 A1] | [B5 G5 R5 A5].
            
            // Row 0: Px 0, 4. Row 1: Px 1, 5.
            __m256d row1_d_fix = _mm256_unpackhi_pd(_mm256_castps_pd(bg_lo), _mm256_castps_pd(ra_lo));

            __m256d row2_d = _mm256_unpacklo_pd(_mm256_castps_pd(bg_hi), _mm256_castps_pd(ra_hi)); // Px 2, 6
            __m256d row3_d = _mm256_unpackhi_pd(_mm256_castps_pd(bg_hi), _mm256_castps_pd(ra_hi)); // Px 3, 7

            // Cast back to float for permutation
            __m256 r0 = _mm256_castpd_ps(row0_d);
            __m256 r1 = _mm256_castpd_ps(row1_d_fix);
            __m256 r2 = _mm256_castpd_ps(row2_d);
            __m256 r3 = _mm256_castpd_ps(row3_d);

            // 6. STORE (Reordering Lanes)
            // Registers currently hold:
            // r0: Px0 | Px4
            // r1: Px1 | Px5
            // r2: Px2 | Px6
            // r3: Px3 | Px7
            
            // We need to write contiguous Px 0,1,2,3...
            float* dst_ptr = reinterpret_cast<float*>(pRowDst + x * 16);

            // Store Px 0, 1
            _mm_storeu_ps(dst_ptr + 0, _mm256_castps256_ps128(r0));
            _mm_storeu_ps(dst_ptr + 4, _mm256_castps256_ps128(r1));
            // Store Px 2, 3
            _mm_storeu_ps(dst_ptr + 8, _mm256_castps256_ps128(r2));
            _mm_storeu_ps(dst_ptr + 12,_mm256_castps256_ps128(r3));
            
            // Store Px 4, 5 (High lanes)
            _mm_storeu_ps(dst_ptr + 16, _mm256_extractf128_ps(r0, 1));
            _mm_storeu_ps(dst_ptr + 20, _mm256_extractf128_ps(r1, 1));
            // Store Px 6, 7 (High lanes)
            _mm_storeu_ps(dst_ptr + 24, _mm256_extractf128_ps(r2, 1));
            _mm_storeu_ps(dst_ptr + 28, _mm256_extractf128_ps(r3, 1));
        }

        // --- SCALAR FALLBACK ---
        for (; x < sizeX; ++x)
		{
            float l_val = rowL[x];
            float a_val = rowAB[x * 2];
            float b_val = rowAB[x * 2 + 1];

            float R, G, B;
            // Scalar Convert
            float fy = (l_val + 16.0f) / 116.0f;
            float fx = fy + (a_val / 500.0f);
            float fz = fy - (b_val / 200.0f);
            
			auto f_inv = [](float t) noexcept
			{ 
				return (t > 0.206893f) ? (t * t * t) : ((t - 16.0f/116.0f) / 7.787f); 
			};
            
			float X = 0.95047f * f_inv(fx);
            float Y = 1.00000f * ((l_val > 8.0f) ? (fy*fy*fy) : (l_val / 903.3f));
            float Z = 1.08883f * f_inv(fz);
            R =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
            G = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
            B =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;

            // Clamp
            R = std::min(std::max(R, 0.0f), 1.0f - FLT_EPSILON);
            G = std::min(std::max(G, 0.0f), 1.0f - FLT_EPSILON);
            B = std::min(std::max(B, 0.0f), 1.0f - FLT_EPSILON);

            const PF_Pixel_BGRA_32f* s = reinterpret_cast<const PF_Pixel_BGRA_32f*>(reinterpret_cast<const uint8_t*>(pSrc) + x * 16);
            PF_Pixel_BGRA_32f* d = reinterpret_cast<PF_Pixel_BGRA_32f*>(reinterpret_cast<uint8_t*>(pDst) + x * 16);

            d->B = B; d->G = G; d->R = R; d->A = s->A;
        }
    }
	
	return;
}


void AVX2_ConvertCIELab_SemiPlanar_ToRgb_ARGB_32f
(
    const PF_Pixel_ARGB_32f* RESTRICT pSrc, 
    const float*             RESTRICT pL,   
    const float*             RESTRICT pAB,  
    PF_Pixel_ARGB_32f*       RESTRICT pDst, 
    int32_t sizeX,
	int32_t sizeY,
    int32_t srcPitch, // Pitch in PIXELS
    int32_t dstPitch  // Pitch in PIXELS 
) noexcept
{
    const intptr_t srcStrideBytes = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_ARGB_32f);
    const intptr_t dstStrideBytes = static_cast<intptr_t>(dstPitch) * sizeof(PF_Pixel_ARGB_32f);

    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max  = _mm256_set1_ps(1.0f - FLT_EPSILON);

    // Alpha Gather Indices for ARGB (A is at float index 0)
    // 0, 4, 8, 12, 16, 20, 24, 28
    const __m256i idx_alpha = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);

    for (int y = 0; y < sizeY; ++y)
    {
        const uint8_t* pRowSrc = reinterpret_cast<const uint8_t*>(pSrc) + (y * srcStrideBytes);
        uint8_t*       pRowDst = reinterpret_cast<uint8_t*>(pDst) + (y * dstStrideBytes);
        
        const float* rowL  = pL + (y * sizeX);
        const float* rowAB = pAB + (y * sizeX * 2);

        int x = 0;
        for (; x <= sizeX - 8; x += 8)
        {
            // 1. Load Lab & Convert
            __m256 L = _mm256_loadu_ps(rowL + x);
            __m256 ab0 = _mm256_loadu_ps(rowAB + x * 2);
            __m256 ab1 = _mm256_loadu_ps(rowAB + x * 2 + 8);
            
            const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256 s0 = _mm256_permutevar8x32_ps(ab0, perm_idx);
            __m256 s1 = _mm256_permutevar8x32_ps(ab1, perm_idx);
            __m256 a = _mm256_permute2f128_ps(s0, s1, 0x20);
            __m256 b = _mm256_permute2f128_ps(s0, s1, 0x31);

            __m256 R, G, B;
            AVX2_Lab_to_RGB_Linear_Inline(L, a, b, R, G, B);

            R = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, R));
            G = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, G));
            B = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, B));

            // 2. Gather Alpha
            const float* src_ptr = reinterpret_cast<const float*>(pRowSrc + x * 16);
            __m256 A = _mm256_i32gather_ps(src_ptr, idx_alpha, 4);

            // 3. PACK ARGB
            // Pair 1: A, R -> AR_lo, AR_hi
            __m256 ar_lo = _mm256_unpacklo_ps(A, R); // A0 R0 A1 R1...
            __m256 ar_hi = _mm256_unpackhi_ps(A, R);

            // Pair 2: G, B -> GB_lo, GB_hi
            __m256 gb_lo = _mm256_unpacklo_ps(G, B); // G0 B0 G1 B1...
            __m256 gb_hi = _mm256_unpackhi_ps(G, B);

            // Merge (Double cast trick)
            // UnpackLo(AR, GB) -> A0 R0 G0 B0
            __m256d r0_d = _mm256_unpacklo_pd(_mm256_castps_pd(ar_lo), _mm256_castps_pd(gb_lo));
            __m256d r1_d = _mm256_unpackhi_pd(_mm256_castps_pd(ar_lo), _mm256_castps_pd(gb_lo));
            __m256d r2_d = _mm256_unpacklo_pd(_mm256_castps_pd(ar_hi), _mm256_castps_pd(gb_hi));
            __m256d r3_d = _mm256_unpackhi_pd(_mm256_castps_pd(ar_hi), _mm256_castps_pd(gb_hi));

            __m256 r0 = _mm256_castpd_ps(r0_d);
            __m256 r1 = _mm256_castpd_ps(r1_d);
            __m256 r2 = _mm256_castpd_ps(r2_d);
            __m256 r3 = _mm256_castpd_ps(r3_d);

            // 4. Store
            float* dst_ptr = reinterpret_cast<float*>(pRowDst + x * 16);
            _mm_storeu_ps(dst_ptr + 0, _mm256_castps256_ps128(r0));
            _mm_storeu_ps(dst_ptr + 4, _mm256_castps256_ps128(r1));
            _mm_storeu_ps(dst_ptr + 8, _mm256_castps256_ps128(r2));
            _mm_storeu_ps(dst_ptr + 12,_mm256_castps256_ps128(r3));
            _mm_storeu_ps(dst_ptr + 16, _mm256_extractf128_ps(r0, 1));
            _mm_storeu_ps(dst_ptr + 20, _mm256_extractf128_ps(r1, 1));
            _mm_storeu_ps(dst_ptr + 24, _mm256_extractf128_ps(r2, 1));
            _mm_storeu_ps(dst_ptr + 28, _mm256_extractf128_ps(r3, 1));
        }

        // Scalar Fallback
        for (; x < sizeX; ++x)
		{
            float l_val = rowL[x];
            float a_val = rowAB[x * 2];
            float b_val = rowAB[x * 2 + 1];

            // Math ...
            float fy = (l_val + 16.0f) / 116.0f;
            float fx = fy + (a_val / 500.0f);
            float fz = fy - (b_val / 200.0f);
            
			auto f_inv = [](float t) noexcept
			{ 
				return (t > 0.206893f) ? (t * t * t) : ((t - 16.0f/116.0f) / 7.787f);
			};
			
            float X = 0.95047f * f_inv(fx);
            float Y = 1.00000f * ((l_val > 8.0f) ? (fy*fy*fy) : (l_val / 903.3f));
            float Z = 1.08883f * f_inv(fz);
            float R =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
            float G = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
            float B =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;

            R = std::min(std::max(R, 0.0f), 1.0f - FLT_EPSILON);
            G = std::min(std::max(G, 0.0f), 1.0f - FLT_EPSILON);
            B = std::min(std::max(B, 0.0f), 1.0f - FLT_EPSILON);

            const PF_Pixel_ARGB_32f* s = reinterpret_cast<const PF_Pixel_ARGB_32f*>(pRowSrc + x * 16);
            PF_Pixel_ARGB_32f* d = reinterpret_cast<PF_Pixel_ARGB_32f*>(pRowDst + x * 16);

            d->A = s->A;
            d->R = R; d->G = G; d->B = B;
        }
    }
}