#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "Common.hpp"
#include "Avx2ColorConverts.hpp"


/**
 * AVX2 OUTPUT CONVERTER: Semi-Planar Lab -> BGRA_8u
 * 
 * - Handles 'L' (Planar) and 'AB' (Interleaved) inputs.
 * - Converts Lab -> Linear RGB.
 * - Clamps values to [0, 255].
 * - Takes Alpha from 'pSrc'.
 * - Packs to BGRA (B=0, G=8, R=16, A=24).
 * - Handles Stride/Pitch math for host buffers.
 */
void AVX2_ConvertCIELab_SemiPlanar_ToRgb
(
    const PF_Pixel_BGRA_8u* RESTRICT pSrc, // for take alpha channel only
    const float*            RESTRICT pL,   // source L (planar)
    const float*            RESTRICT pAB,  // source ab (interleaved)
    PF_Pixel_BGRA_8u*       RESTRICT pDst, // destination buffer
    int32_t           sizeX,               // size of image (width)
    int32_t           sizeY,               // number of lines (height)
    int32_t           srcPitch,            // pitch of pSrc in PIXELS
    int32_t           dstPitch             // pitch of pDst in PIXELS 
) noexcept
{
    // Convert Pixel Pitch to Byte Stride for arithmetic
    const intptr_t srcStrideBytes = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_BGRA_8u);
    const intptr_t dstStrideBytes = static_cast<intptr_t>(dstPitch) * sizeof(PF_Pixel_BGRA_8u);

    // Constants for Quantization
    const __m256 v_scale = _mm256_set1_ps(255.0f);
    const __m256 v_half  = _mm256_set1_ps(0.5f);
    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_max   = _mm256_set1_ps(255.0f);

    // Alpha Mask (0xAARRGGBB -> 0xAA000000)
    const __m256i v_alpha_mask = _mm256_set1_epi32(0xFF000000);

    for (int y = 0; y < sizeY; ++y)
    {
        // Calculate Row Pointers
        // Host buffers (Src/Dst) use Stride logic (byte offset)
        const uint8_t* pRowSrcRaw = reinterpret_cast<const uint8_t*>(pSrc) + (y * srcStrideBytes);
        uint8_t*       pRowDstRaw = reinterpret_cast<uint8_t*>(pDst) + (y * dstStrideBytes);
        
        const PF_Pixel_BGRA_8u* rowSrc = reinterpret_cast<const PF_Pixel_BGRA_8u*>(pRowSrcRaw);
        PF_Pixel_BGRA_8u*       rowDst = reinterpret_cast<PF_Pixel_BGRA_8u*>(pRowDstRaw);

        // Lab buffers are tightly packed internally
        const float* rowL  = pL + (y * sizeX);
        const float* rowAB = pAB + (y * sizeX * 2);

        int x = 0;

        // --- AVX2 LOOP (8 pixels per iter) ---
        for (; x <= sizeX - 8; x += 8)
        {
            // 1. LOAD LAB
            // Load L (Planar) -> 8 floats
            __m256 L = _mm256_loadu_ps(rowL + x);

            // Load AB (Interleaved) -> 16 floats
            __m256 ab0 = _mm256_loadu_ps(rowAB + x * 2);
            __m256 ab1 = _mm256_loadu_ps(rowAB + x * 2 + 8);

            // De-interleave AB -> Planar A and Planar B
            // Permute pattern: [0, 2, 4, 6, 1, 3, 5, 7] to group evens (a) and odds (b)
            const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            
            __m256 s0 = _mm256_permutevar8x32_ps(ab0, perm_idx);
            __m256 s1 = _mm256_permutevar8x32_ps(ab1, perm_idx);

            // Extract Low 128 (A's) and High 128 (B's) from shuffled vectors
            __m256 a = _mm256_permute2f128_ps(s0, s1, 0x20); // Low halves -> [a0..a3 | a4..a7]
            __m256 b = _mm256_permute2f128_ps(s0, s1, 0x31); // High halves -> [b0..b3 | b4..b7]

            // 2. CONVERT Lab -> Linear RGB
            __m256 R, G, B;
            AVX2_Lab_to_RGB_Linear_Inline(L, a, b, R, G, B);

            // 3. SCALE & CLAMP
            // R = Clamp(R * 255.0 + 0.5)
            R = _mm256_fmadd_ps(R, v_scale, v_half);
            G = _mm256_fmadd_ps(G, v_scale, v_half);
            B = _mm256_fmadd_ps(B, v_scale, v_half);

            R = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, R));
            G = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, G));
            B = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, B));

            // 4. CONVERT TO INT
            __m256i iR = _mm256_cvttps_epi32(R);
            __m256i iG = _mm256_cvttps_epi32(G);
            __m256i iB = _mm256_cvttps_epi32(B);

            // 5. PACKING
            // Load Alpha from Source
            __m256i src_px = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowSrc + x));
            __m256i alpha  = _mm256_and_si256(src_px, v_alpha_mask);

            // Compose BGRA (Little Endian: B=0-7, G=8-15, R=16-23, A=24-31)
            __m256i out = iB;
            out = _mm256_or_si256(out, _mm256_slli_epi32(iG, 8));
            out = _mm256_or_si256(out, _mm256_slli_epi32(iR, 16));
            out = _mm256_or_si256(out, alpha);

            // 6. STORE
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(rowDst + x), out);
        }

        // --- SCALAR FALLBACK ---
        for (; x < sizeX; ++x)
        {
            float l_val = rowL[x];
            float a_val = rowAB[x * 2 + 0];
            float b_val = rowAB[x * 2 + 1];

            // Manual Scalar Lab->RGB
            // Constants duplicated here or use a scalar inline helper if available
            // Simplified version for brevity, assuming standard D65 math:
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

            float r_lin =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
            float g_lin = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
            float b_lin =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;

            // Clamp & Scale
            const uint8_t uR = (uint8_t)std::min(std::max(r_lin * 255.0f + 0.5f, 0.0f), 255.0f);
            const uint8_t uG = (uint8_t)std::min(std::max(g_lin * 255.0f + 0.5f, 0.0f), 255.0f);
            const uint8_t uB = (uint8_t)std::min(std::max(b_lin * 255.0f + 0.5f, 0.0f), 255.0f);

            // Pack
            rowDst[x].B = uB;
            rowDst[x].G = uG;
            rowDst[x].R = uR;
            rowDst[x].A = rowSrc[x].A;
        }
    }
    
    return;
}


void AVX2_ConvertCIELab_SemiPlanar_ToRgb
(
    const PF_Pixel_ARGB_8u* RESTRICT pSrc, // Original ARGB source (for Alpha)
    const float*            RESTRICT pL,   // source L (planar)
    const float*            RESTRICT pAB,  // source ab (interleaved)
    PF_Pixel_ARGB_8u*       RESTRICT pDst, // destination buffer
    int32_t           sizeX,
    int32_t           sizeY,
    int32_t           srcPitch, // Pitch in PIXELS
    int32_t           dstPitch  // Pitch in PIXELS 
) noexcept
{
    // Convert Pixel Pitch to Byte Stride
    const intptr_t srcStrideBytes = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_ARGB_8u);
    const intptr_t dstStrideBytes = static_cast<intptr_t>(dstPitch) * sizeof(PF_Pixel_ARGB_8u);

    // Constants
    const __m256 v_scale = _mm256_set1_ps(255.0f);
    const __m256 v_half  = _mm256_set1_ps(0.5f);
    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_max   = _mm256_set1_ps(255.0f);

    // Alpha Mask: In ARGB (byte order A,R,G,B), A is at the lowest address (Bits 0-7).
    // Int32 mask: 0x000000FF
    const __m256i v_alpha_mask = _mm256_set1_epi32(0x000000FF);

    for (int y = 0; y < sizeY; ++y)
    {
        // Row Pointers
        const uint8_t* pRowSrcRaw = reinterpret_cast<const uint8_t*>(pSrc) + (y * srcStrideBytes);
        uint8_t*       pRowDstRaw = reinterpret_cast<uint8_t*>(pDst) + (y * dstStrideBytes);
        
        const PF_Pixel_ARGB_8u* rowSrc = reinterpret_cast<const PF_Pixel_ARGB_8u*>(pRowSrcRaw);
        PF_Pixel_ARGB_8u*       rowDst = reinterpret_cast<PF_Pixel_ARGB_8u*>(pRowDstRaw);

        const float* rowL  = pL + (y * sizeX);
        const float* rowAB = pAB + (y * sizeX * 2);

        int x = 0;

        // --- AVX2 LOOP (8 pixels) ---
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

            // 3. SCALE & CLAMP
            R = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(R, v_scale, v_half)));
            G = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(G, v_scale, v_half)));
            B = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, _mm256_fmadd_ps(B, v_scale, v_half)));

            // 4. CONVERT TO INT
            __m256i iR = _mm256_cvttps_epi32(R);
            __m256i iG = _mm256_cvttps_epi32(G);
            __m256i iB = _mm256_cvttps_epi32(B);

            // 5. PACKING (ARGB: 0xBBGGRRAA)
            // Load Alpha
            __m256i src_px = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowSrc + x));
            __m256i alpha  = _mm256_and_si256(src_px, v_alpha_mask);

            // Compose Int32:
            // A: Bits 0-7   (Masked from source)
            // R: Bits 8-15  (Shift 8)
            // G: Bits 16-23 (Shift 16)
            // B: Bits 24-31 (Shift 24)
            
            __m256i out = alpha;
            out = _mm256_or_si256(out, _mm256_slli_epi32(iR, 8));
            out = _mm256_or_si256(out, _mm256_slli_epi32(iG, 16));
            out = _mm256_or_si256(out, _mm256_slli_epi32(iB, 24));

            // 6. STORE
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(rowDst + x), out);
        }

        // --- SCALAR FALLBACK ---
        for (; x < sizeX; ++x)
        {
            float l_val = rowL[x];
            float a_val = rowAB[x * 2 + 0];
            float b_val = rowAB[x * 2 + 1];

            // Scalar Lab->RGB
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

            float r_lin =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
            float g_lin = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
            float b_lin =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;

            // Clamp
            const uint8_t uR = (uint8_t)std::min(std::max(r_lin * 255.0f + 0.5f, 0.0f), 255.0f);
            const uint8_t uG = (uint8_t)std::min(std::max(g_lin * 255.0f + 0.5f, 0.0f), 255.0f);
            const uint8_t uB = (uint8_t)std::min(std::max(b_lin * 255.0f + 0.5f, 0.0f), 255.0f);

            // Pack ARGB
            rowDst[x].A = rowSrc[x].A;
            rowDst[x].R = uR;
            rowDst[x].G = uG;
            rowDst[x].B = uB;
        }
    }
    
    return;
}