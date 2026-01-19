#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "Common.hpp"
#include "Avx2ColorConverts.hpp"

void AVX2_ConvertCIELab_SemiPlanar_ToRgb
(
    const PF_Pixel_BGRA_16u* RESTRICT pSrc, // For Alpha (8 bytes per pixel)
    const float*             RESTRICT pL,   // Planar L
    const float*             RESTRICT pAB,  // Interleaved AB
    PF_Pixel_BGRA_16u*       RESTRICT pDst, // Output Buffer
    int32_t sizeX, 
    int32_t sizeY,
    int32_t           srcPitch,            // pitch of pSrc in PIXELS
    int32_t           dstPitch             // pitch of pDst in PIXELS 
) noexcept
{
    // Convert Pixel Pitch to Byte Stride for arithmetic
    const intptr_t srcPitchBytes = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_BGRA_16u);
    const intptr_t dstPitchBytes = static_cast<intptr_t>(dstPitch) * sizeof(PF_Pixel_BGRA_16u);

    // Scale 0..1 to 0..32767
    const __m256 v_scale = _mm256_set1_ps(32767.0f);
    const __m256 v_half  = _mm256_set1_ps(0.5f);
    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_max   = _mm256_set1_ps(32767.0f);

    // Shuffle mask to extract Alpha (16-bit) into 32-bit integers
    // Source Pixel: [B0 B1 G0 G1 R0 R1 A0 A1] (8 bytes)
    // We want 32-bit Integer: [00 00 A0 A1] (Little Endian: A at low bytes)
    // Indices: 6,7 (Alpha), -1,-1 (Zero)
    const __m256i v_alpha_shuf = _mm256_setr_epi8(
        6,7, -1,-1,  14,15, -1,-1,  22,23, -1,-1,  30,31, -1,-1, // Lane 0 (Px 0-3)
        6,7, -1,-1,  14,15, -1,-1,  22,23, -1,-1,  30,31, -1,-1  // Lane 1 (Px 4-7)
    );

    for (int y = 0; y < sizeY; ++y)
    {
        // Row Pointers
        const uint8_t* rowSrcPtr = reinterpret_cast<const uint8_t*>(pSrc) + (y * srcPitchBytes);
        uint8_t*       rowDstPtr = reinterpret_cast<uint8_t*>(pDst) + (y * dstPitchBytes);
        
        const PF_Pixel_BGRA_16u* rowSrc = reinterpret_cast<const PF_Pixel_BGRA_16u*>(rowSrcPtr);
        PF_Pixel_BGRA_16u*       rowDst = reinterpret_cast<PF_Pixel_BGRA_16u*>(rowDstPtr);

        const float* rowL  = pL + (y * sizeX);
        const float* rowAB = pAB + (y * sizeX * 2);

        int x = 0;
        // Process 8 pixels per loop
        // Input: 8 floats L, 16 floats AB. 
        // Output: 8 pixels * 8 bytes/px = 64 bytes.
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

            // 2. CONVERT Lab -> Linear RGB (0..1)
            __m256 R, G, B;
            AVX2_Lab_to_RGB_Linear_Inline(L, a, b, R, G, B);

            // 3. SCALE & CLAMP (0..32767)
            R = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(R, v_scale, v_half)));
            G = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(G, v_scale, v_half)));
            B = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(B, v_scale, v_half)));

            // 4. CONVERT TO INT32
            __m256i iR = _mm256_cvttps_epi32(R);
            __m256i iG = _mm256_cvttps_epi32(G);
            __m256i iB = _mm256_cvttps_epi32(B);

            // 5. EXTRACT ALPHA (16-bit Source -> 32-bit Ints)
            // Load 8 source pixels (64 bytes) into 2 registers
            const __m256i* src_vec = reinterpret_cast<const __m256i*>(rowSrcPtr + (x * 8));
            __m256i s0_vec = _mm256_loadu_si256(src_vec);     // Px 0-3
            __m256i s1_vec = _mm256_loadu_si256(src_vec + 1); // Px 4-7

            // Shuffle to get 32-bit Integers of Alpha
            __m256i iA_lo = _mm256_shuffle_epi8(s0_vec, v_alpha_shuf);
            __m256i iA_hi = _mm256_shuffle_epi8(s1_vec, v_alpha_shuf);
            // Combine lanes
            __m256i iA = _mm256_permute2f128_si256(iA_lo, iA_hi, 0x20);

            // 6. PACK 32-bit -> 16-bit (BGRA)
            // Target Memory Layout (16-bit words): B0 G0 R0 A0 B1 G1 R1 A1 ...
            
            // Step A: Pack (B, G) -> BG_lo, BG_hi (Interleaved 16-bit pairs)
            // Lane 0: B0 G0 B1 G1 B2 G2 B3 G3
            __m256i bg = _mm256_packus_epi32(iB, iG);

            // Step B: Pack (R, A) -> RA_lo, RA_hi
            // Lane 0: R0 A0 R1 A1 R2 A2 R3 A3
            __m256i ra = _mm256_packus_epi32(iR, iA);

            // Step C: Interleave (Unpack)
            // UnpackLo_Epi32 takes 32-bit chunks.
            // Chunk 0 from BG: [B0 G0] (16-bit words)
            // Chunk 0 from RA: [R0 A0] (16-bit words)
            // Result: [B0 G0 R0 A0] -> Correct Pixel!
            __m256i out_lo = _mm256_unpacklo_epi32(bg, ra); // Pixels 0,1, 4,5
            __m256i out_hi = _mm256_unpackhi_epi32(bg, ra); // Pixels 2,3, 6,7

            // Step D: Reorder Lanes
            // out_lo: [Px0 Px1 | Px4 Px5]
            // out_hi: [Px2 Px3 | Px6 Px7]
            // We want [Px0 Px1 Px2 Px3 | Px4 Px5 Px6 Px7]
            
            // Grab Px0-3 (Low lanes)
            __m256i final_0 = _mm256_permute2f128_si256(out_lo, out_hi, 0x20);
            // Grab Px4-7 (High lanes)
            __m256i final_1 = _mm256_permute2f128_si256(out_lo, out_hi, 0x31);

            // 7. STORE
            __m256i* dst_vec = reinterpret_cast<__m256i*>(rowDstPtr + (x * 8));
            _mm256_storeu_si256(dst_vec,     final_0);
            _mm256_storeu_si256(dst_vec + 1, final_1);
        }

        // --- SCALAR FALLBACK ---
        for (; x < sizeX; ++x)
		{
            float l_val = rowL[x];
            float a_val = rowAB[x * 2];
            float b_val = rowAB[x * 2 + 1];

            float R, G, B;
            // Assumes scalar helper exists, or copy math
            // Convert_CIELab_to_LinearRGB(l_val, a_val, b_val, R, G, B);
            
            // Inline math for standalone correctness:
            float fy = (l_val + 16.0f) / 116.0f;
            float fx = fy + (a_val / 500.0f);
            float fz = fy - (b_val / 200.0f);
            auto f_inv = [](float t) { return (t > 0.206893f) ? (t * t * t) : ((t - 16.0f/116.0f) / 7.787f); };
            float X = 0.95047f * f_inv(fx);
            float Y = 1.00000f * ((l_val > 8.0f) ? (fy*fy*fy) : (l_val / 903.3f));
            float Z = 1.08883f * f_inv(fz);
            R =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
            G = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
            B =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;

            const uint16_t uR = (uint16_t)std::min(std::max(R * 32767.0f + 0.5f, 0.0f), 32767.0f);
            const uint16_t uG = (uint16_t)std::min(std::max(G * 32767.0f + 0.5f, 0.0f), 32767.0f);
            const uint16_t uB = (uint16_t)std::min(std::max(B * 32767.0f + 0.5f, 0.0f), 32767.0f);

            // Pack BGRA
            rowDst[x].A = rowSrc[x].A;
            rowDst[x].R = uR;
            rowDst[x].G = uG;
            rowDst[x].B = uB;
        }
    }
	
	return;
}


void AVX2_ConvertCIELab_SemiPlanar_ToRgb
(
    const PF_Pixel_ARGB_16u* RESTRICT pSrc, 
    const float*             RESTRICT pL,   
    const float*             RESTRICT pAB,  
    PF_Pixel_ARGB_16u*       RESTRICT pDst, 
    int32_t sizeX,
	int32_t sizeY,
    int32_t srcPitch, 
    int32_t dstPitch  
) noexcept
{
    // Convert Pixel Pitch to Byte Stride for arithmetic
    const intptr_t srcPitchBytes = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_BGRA_16u);
    const intptr_t dstPitchBytes = static_cast<intptr_t>(dstPitch) * sizeof(PF_Pixel_BGRA_16u);
	
    const __m256 v_scale = _mm256_set1_ps(32767.0f);
    const __m256 v_half  = _mm256_set1_ps(0.5f);
    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_max   = _mm256_set1_ps(32767.0f);

    // Shuffle mask for Alpha extraction (ARGB)
    // Source: [A0 A1 R0 R1 G0 G1 B0 B1]
    // We want Alpha at bytes 0,1.
    // Shuffle mask to put bytes 0,1 into lower 32-bits.
    const __m256i v_alpha_shuf = _mm256_setr_epi8(
        0,1, -1,-1,  8,9, -1,-1,  16,17, -1,-1,  24,25, -1,-1, // Lane 0
        0,1, -1,-1,  8,9, -1,-1,  16,17, -1,-1,  24,25, -1,-1  // Lane 1
    );

    for (int y = 0; y < sizeY; ++y)
    {
        const uint8_t* rowSrcPtr = reinterpret_cast<const uint8_t*>(pSrc) + (y * srcPitchBytes);
        uint8_t*       rowDstPtr = reinterpret_cast<uint8_t*>(pDst) + (y * dstPitchBytes);

        const PF_Pixel_ARGB_16u* rowSrc = reinterpret_cast<const PF_Pixel_ARGB_16u*>(rowSrcPtr);
        PF_Pixel_ARGB_16u*       rowDst = reinterpret_cast<PF_Pixel_ARGB_16u*>(rowDstPtr);
        
        const float* rowL  = pL + (y * sizeX);
        const float* rowAB = pAB + (y * sizeX * 2);

        int x = 0;
        for (; x <= sizeX - 8; x += 8)
        {
            // 1. Load Lab & Deinterleave
            __m256 L = _mm256_loadu_ps(rowL + x);
            __m256 ab0 = _mm256_loadu_ps(rowAB + x * 2);
            __m256 ab1 = _mm256_loadu_ps(rowAB + x * 2 + 8);
            const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256 s0 = _mm256_permutevar8x32_ps(ab0, perm_idx);
            __m256 s1 = _mm256_permutevar8x32_ps(ab1, perm_idx);
            __m256 a = _mm256_permute2f128_ps(s0, s1, 0x20);
            __m256 b = _mm256_permute2f128_ps(s0, s1, 0x31);

            // 2. Convert & Scale
            __m256 R, G, B;
            AVX2_Lab_to_RGB_Linear_Inline(L, a, b, R, G, B);
            
            R = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(R, v_scale, v_half)));
            G = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(G, v_scale, v_half)));
            B = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(B, v_scale, v_half)));

            __m256i iR = _mm256_cvttps_epi32(R);
            __m256i iG = _mm256_cvttps_epi32(G);
            __m256i iB = _mm256_cvttps_epi32(B);

            // 3. Extract Alpha
            const __m256i* src_vec = reinterpret_cast<const __m256i*>(rowSrcPtr + (x * 8));
            __m256i s0_vec = _mm256_loadu_si256(src_vec);     
            __m256i s1_vec = _mm256_loadu_si256(src_vec + 1); 
            __m256i iA_lo = _mm256_shuffle_epi8(s0_vec, v_alpha_shuf);
            __m256i iA_hi = _mm256_shuffle_epi8(s1_vec, v_alpha_shuf);
            __m256i iA    = _mm256_permute2f128_si256(iA_lo, iA_hi, 0x20);

            // 4. PACK ARGB
            // Target: A R G B
            // Pair 1: (A, R) -> AR
            __m256i ar = _mm256_packus_epi32(iA, iR);
            // Pair 2: (G, B) -> GB
            __m256i gb = _mm256_packus_epi32(iG, iB);

            // Interleave (Unpack)
            // UnpackLo(AR, GB) -> A R G B
            __m256i out_lo = _mm256_unpacklo_epi32(ar, gb);
            __m256i out_hi = _mm256_unpackhi_epi32(ar, gb);

            // Fix Lanes
            __m256i final_0 = _mm256_permute2f128_si256(out_lo, out_hi, 0x20);
            __m256i final_1 = _mm256_permute2f128_si256(out_lo, out_hi, 0x31);

            // Store
            __m256i* dst_vec = reinterpret_cast<__m256i*>(rowDstPtr + (x * 8));
            _mm256_storeu_si256(dst_vec,     final_0);
            _mm256_storeu_si256(dst_vec + 1, final_1);
        }

        // --- SCALAR FALLBACK ---
        for (; x < sizeX; ++x)
		{
            float l_val = rowL[x];
            float a_val = rowAB[x * 2];
            float b_val = rowAB[x * 2 + 1];

            // Same math as BGRA...
            float fy = (l_val + 16.0f) / 116.0f;
            float fx = fy + (a_val / 500.0f);
            float fz = fy - (b_val / 200.0f);
            auto f_inv = [](float t) { return (t > 0.206893f) ? (t * t * t) : ((t - 16.0f/116.0f) / 7.787f); };
            float X = 0.95047f * f_inv(fx);
            float Y = 1.00000f * ((l_val > 8.0f) ? (fy*fy*fy) : (l_val / 903.3f));
            float Z = 1.08883f * f_inv(fz);
            float r_lin =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
            float g_lin = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
            float b_lin =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;

            const uint16_t uR = (uint16_t)std::min(std::max(r_lin * 32767.0f + 0.5f, 0.0f), 32767.0f);
            const uint16_t uG = (uint16_t)std::min(std::max(g_lin * 32767.0f + 0.5f, 0.0f), 32767.0f);
            const uint16_t uB = (uint16_t)std::min(std::max(b_lin * 32767.0f + 0.5f, 0.0f), 32767.0f);

            // Pack ARGB
            rowDst[x].A = rowSrc[x].A;
            rowDst[x].R = uR;
            rowDst[x].G = uG;
            rowDst[x].B = uB;
        }
    }
	
	return;
}