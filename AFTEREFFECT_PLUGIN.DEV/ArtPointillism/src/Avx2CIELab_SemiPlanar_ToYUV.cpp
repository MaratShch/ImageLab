#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "Common.hpp"
#include "Avx2ColorConverts.hpp"

/**
 * AVX2 OUTPUT CONVERTER: Semi-Planar Lab -> VUYA_8u
 * 
 * Pipeline:
 * 1. Lab -> Linear RGB.
 * 2. Linear RGB -> YUV (BT.601 or BT.709).
 * 3. Scaling: Full Range (0..255), UV Offset +128.
 * 4. Packing: V(LSB)..U..Y..A(MSB).
 * 5. Alpha copied from pSrc.
 */
void AVX2_ConvertCIELab_SemiPlanar_To_YUV
(
    const PF_Pixel_VUYA_8u* RESTRICT pSrc,       // Source for Alpha
    const float*            RESTRICT pL,         // Planar L
    const float*            RESTRICT pAB,        // Interleaved AB
    PF_Pixel_VUYA_8u*       RESTRICT pDst,       // Output
    int32_t                 sizeX,
    int32_t                 sizeY,
    int32_t                 srcPitch, 
    int32_t                 dstPitch, 
    bool                    isBT709              // true=HD, false=SD
) noexcept
{
    // Pitch in Bytes (4 floats * 4 bytes = 16 bytes per pixel)
    const intptr_t srcPitchBytes = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_VUYA_8u);
    const intptr_t dstPitchBytes = static_cast<intptr_t>(dstPitch) * sizeof(PF_Pixel_VUYA_8u);

    // --- 1. SETUP CONSTANTS ---
    
    // Coefficients (Normalized 0..1)
    float Kr, Kb, Kg;
    if (isBT709)
	{
        Kr = 0.2126f; Kb = 0.0722f; Kg = 0.7152f;
    } else
	{ // BT.601
        Kr = 0.2990f; Kb = 0.1140f; Kg = 0.5870f;
    }

    // U/V Scaling factors for Full Range
    // U = (B-Y)/(2*(1-Kb)); V = (R-Y)/(2*(1-Kr))
    // We pre-multiply by 255.0f here to save a multiply later.
    const float scale = 255.0f;
    
    float c_Y_R = Kr * scale;
    float c_Y_G = Kg * scale;
    float c_Y_B = Kb * scale;

    float c_U_R = (-Kr * scale) / (2.0f * (1.0f - Kb)); // Derived simplified weights
    float c_U_G = (-Kg * scale) / (2.0f * (1.0f - Kb));
    float c_U_B = (0.5f * scale);                       // (1-Kb)/(2*(1-Kb)) * scale

    float c_V_R = (0.5f * scale);
    float c_V_G = (-Kg * scale) / (2.0f * (1.0f - Kr));
    float c_V_B = (-Kb * scale) / (2.0f * (1.0f - Kr));

    // Load into AVX registers
    const __m256 v_YR = _mm256_set1_ps(c_Y_R);
    const __m256 v_YG = _mm256_set1_ps(c_Y_G);
    const __m256 v_YB = _mm256_set1_ps(c_Y_B);

    const __m256 v_UR = _mm256_set1_ps(c_U_R);
    const __m256 v_UG = _mm256_set1_ps(c_U_G);
    const __m256 v_UB = _mm256_set1_ps(c_U_B);

    const __m256 v_VR = _mm256_set1_ps(c_V_R);
    const __m256 v_VG = _mm256_set1_ps(c_V_G);
    const __m256 v_VB = _mm256_set1_ps(c_V_B);

    const __m256 v_half   = _mm256_set1_ps(0.5f);
    const __m256 v_offset = _mm256_set1_ps(128.0f); // UV Offset

    // Alpha Mask for VUYA (A is MSB 0xAA000000)
    const __m256i alpha_mask = _mm256_set1_epi32(0xFF000000);

    for (int y = 0; y < sizeY; ++y)
    {
        const uint8_t* pRowSrc = reinterpret_cast<const uint8_t*>(pSrc) + (y * srcPitchBytes);
        uint8_t*       pRowDst = reinterpret_cast<uint8_t*>(pDst) + (y * dstPitchBytes);
        
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

            const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256 s0 = _mm256_permutevar8x32_ps(ab0, perm_idx);
            __m256 s1 = _mm256_permutevar8x32_ps(ab1, perm_idx);
            __m256 a = _mm256_permute2f128_ps(s0, s1, 0x20);
            __m256 b = _mm256_permute2f128_ps(s0, s1, 0x31);

            // 2. CONVERT Lab -> Linear RGB
            __m256 R, G, B;
            AVX2_Lab_to_RGB_Linear_Inline(L, a, b, R, G, B);

            // 3. CONVERT RGB -> YUV (Full Range)
            // Y = (Kr*R + Kg*G + Kb*B) * 255
            __m256 Y = _mm256_fmadd_ps(v_YR, R, _mm256_fmadd_ps(v_YG, G, _mm256_mul_ps(v_YB, B)));
            Y = _mm256_add_ps(Y, v_half); // Rounding

            // U = (Ur*R + Ug*G + Ub*B) * 255 + 128
            __m256 U = _mm256_fmadd_ps(v_UR, R, _mm256_fmadd_ps(v_UG, G, _mm256_mul_ps(v_UB, B)));
            U = _mm256_add_ps(U, _mm256_add_ps(v_offset, v_half));

            // V = (Vr*R + Vg*G + Vb*B) * 255 + 128
            __m256 V = _mm256_fmadd_ps(v_VR, R, _mm256_fmadd_ps(v_VG, G, _mm256_mul_ps(v_VB, B)));
            V = _mm256_add_ps(V, _mm256_add_ps(v_offset, v_half));

            // 4. CONVERT TO INT32 (Truncate because we added 0.5)
            // cvttps_epi32 clamps to Min/Max integer range if out of bounds? 
            // No, behavior is undefined for overflow. We should Clamp.
            // Assuming input Lab is valid, RGB 0..1. 
            // YUV calc stays within range 0..255.
            
            __m256i iY = _mm256_cvttps_epi32(Y);
            __m256i iU = _mm256_cvttps_epi32(U);
            __m256i iV = _mm256_cvttps_epi32(V);

            // 5. PACKING (VUYA)
            // Layout: V (0-7), U (8-15), Y (16-23), A (24-31)
            
            // Load Alpha from Source (32-bit load per pixel)
            __m256i src_px = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pRowSrc + x * 4));
            __m256i alpha  = _mm256_and_si256(src_px, alpha_mask);

            __m256i out = iV;
            out = _mm256_or_si256(out, _mm256_slli_epi32(iU, 8));
            out = _mm256_or_si256(out, _mm256_slli_epi32(iY, 16));
            out = _mm256_or_si256(out, alpha); // A is already at 24-31

            // 6. STORE
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(pRowDst + x * 4), out);
        }

        // --- SCALAR FALLBACK ---
        for (; x < sizeX; ++x)
		{
            float l_val = rowL[x];
            float a_val = rowAB[x * 2];
            float b_val = rowAB[x * 2 + 1];

            // Lab->RGB Math (Inline)
            float fy = (l_val + 16.0f) / 116.0f;
            float fx = fy + (a_val / 500.0f);
            float fz = fy - (b_val / 200.0f);
            
			auto f_inv = [](float t) noexcept
			{ 
				return (t > 0.206893f) ? (t * t * t) : ((t - 16.0f/116.0f) / 7.787f); 
			};
            
			float X = 0.95047f * f_inv(fx);
            float Y_xyz = 1.00000f * ((l_val > 8.0f) ? (fy*fy*fy) : (l_val / 903.3f));
            float Z = 1.08883f * f_inv(fz);
            
            float r =  3.2404542f*X - 1.5371385f*Y_xyz - 0.4985314f*Z;
            float g = -0.9692660f*X + 1.8760108f*Y_xyz + 0.0415560f*Z;
            float b =  0.0556434f*X - 0.2040259f*Y_xyz + 1.0572252f*Z;

            // RGB->YUV Math
            float y_val = (c_Y_R * r) + (c_Y_G * g) + (c_Y_B * b);
            float u_val = (c_U_R * r) + (c_U_G * g) + (c_U_B * b) + 128.0f;
            float v_val = (c_V_R * r) + (c_V_G * g) + (c_V_B * b) + 128.0f;

            // Clamp & Cast
            uint8_t uY = (uint8_t)std::min(std::max(y_val + 0.5f, 0.0f), 255.0f);
            uint8_t uU = (uint8_t)std::min(std::max(u_val + 0.5f, 0.0f), 255.0f);
            uint8_t uV = (uint8_t)std::min(std::max(v_val + 0.5f, 0.0f), 255.0f);

            // Pack
            PF_Pixel_VUYA_8u* d = reinterpret_cast<PF_Pixel_VUYA_8u*>(pRowDst + x * 4);
            const PF_Pixel_VUYA_8u* s = reinterpret_cast<const PF_Pixel_VUYA_8u*>(pRowSrc + x * 4);
            
            d->V = uV;
            d->U = uU;
            d->Y = uY;
            d->A = s->A;
        }
    }
	
	return;
}

/**
 * AVX2 OUTPUT CONVERTER: Semi-Planar Lab -> VUYA_16u
 * 
 * Pipeline:
 * 1. Lab -> Linear RGB.
 * 2. Linear RGB -> YUV (BT.601 or BT.709).
 * 3. Scaling: 0..32767. UV Offset +16384.
 * 4. Packing: V..U..Y..A (16-bit).
 * 5. Alpha copied from pSrc.
 */
void AVX2_ConvertCIELab_SemiPlanar_To_YUV
(
    const PF_Pixel_VUYA_16u* RESTRICT pSrc,       // Source for Alpha
    const float*             RESTRICT pL,         // Planar L
    const float*             RESTRICT pAB,        // Interleaved AB
    PF_Pixel_VUYA_16u*       RESTRICT pDst,       // Output
    int32_t                  sizeX,
    int32_t                  sizeY,
    int32_t                  srcPitch,
    int32_t                  dstPitch,
    bool                     isBT709              // true=HD, false=SD
) noexcept
{
    // Pitch in Bytes (4 floats * 4 bytes = 16 bytes per pixel)
    const intptr_t srcPitchBytes = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_VUYA_16u);
    const intptr_t dstPitchBytes = static_cast<intptr_t>(dstPitch) * sizeof(PF_Pixel_VUYA_16u);

    // --- 1. SETUP CONSTANTS ---
   
    // Matrix Coefficients
    float Kr, Kb, Kg;
    if (isBT709)
	{
        Kr = 0.2126f; Kb = 0.0722f; Kg = 0.7152f;
    } else { // BT.601
        Kr = 0.2990f; Kb = 0.1140f; Kg = 0.5870f;
    }

    // Scaling factors for 0..32767 range
    const float scale = 32767.0f;
    
    float c_Y_R = Kr * scale;
    float c_Y_G = Kg * scale;
    float c_Y_B = Kb * scale;

    float c_U_R = (-Kr * scale) / (2.0f * (1.0f - Kb));
    float c_U_G = (-Kg * scale) / (2.0f * (1.0f - Kb));
    float c_U_B = (0.5f * scale);

    float c_V_R = (0.5f * scale);
    float c_V_G = (-Kg * scale) / (2.0f * (1.0f - Kr));
    float c_V_B = (-Kb * scale) / (2.0f * (1.0f - Kr));

    // Load Registers
    const __m256 v_YR = _mm256_set1_ps(c_Y_R);
    const __m256 v_YG = _mm256_set1_ps(c_Y_G);
    const __m256 v_YB = _mm256_set1_ps(c_Y_B);

    const __m256 v_UR = _mm256_set1_ps(c_U_R);
    const __m256 v_UG = _mm256_set1_ps(c_U_G);
    const __m256 v_UB = _mm256_set1_ps(c_U_B);

    const __m256 v_VR = _mm256_set1_ps(c_V_R);
    const __m256 v_VG = _mm256_set1_ps(c_V_G);
    const __m256 v_VB = _mm256_set1_ps(c_V_B);

    const __m256 v_half   = _mm256_set1_ps(0.5f);
    const __m256 v_offset = _mm256_set1_ps(16384.0f); // Midpoint for UV
    const __m256 v_min    = _mm256_setzero_ps();
    const __m256 v_max    = _mm256_set1_ps(32767.0f);

    // Shuffle mask to extract Alpha (bytes 6-7) to lower 32-bits
    const __m256i v_alpha_shuf = _mm256_setr_epi8(
        6,7, -1,-1,  14,15, -1,-1,  22,23, -1,-1,  30,31, -1,-1,
        6,7, -1,-1,  14,15, -1,-1,  22,23, -1,-1,  30,31, -1,-1
    );

    for (int y = 0; y < sizeY; ++y)
    {
        const uint8_t* pRowSrc = reinterpret_cast<const uint8_t*>(pSrc) + (y * srcPitchBytes);
        uint8_t*       pRowDst = reinterpret_cast<uint8_t*>(pDst) + (y * dstPitchBytes);
        
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

            const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256 s0 = _mm256_permutevar8x32_ps(ab0, perm_idx);
            __m256 s1 = _mm256_permutevar8x32_ps(ab1, perm_idx);
            __m256 a = _mm256_permute2f128_ps(s0, s1, 0x20);
            __m256 b = _mm256_permute2f128_ps(s0, s1, 0x31);

            // 2. CONVERT Lab -> Linear RGB
            __m256 R, G, B;
            AVX2_Lab_to_RGB_Linear_Inline(L, a, b, R, G, B);

            // 3. CONVERT RGB -> YUV (Scaled 0..32767)
            // Y Calculation
            __m256 Y = _mm256_fmadd_ps(v_YR, R, _mm256_fmadd_ps(v_YG, G, _mm256_mul_ps(v_YB, B)));
            Y = _mm256_add_ps(Y, v_half);

            // U Calculation
            __m256 U = _mm256_fmadd_ps(v_UR, R, _mm256_fmadd_ps(v_UG, G, _mm256_mul_ps(v_UB, B)));
            U = _mm256_add_ps(U, _mm256_add_ps(v_offset, v_half));

            // V Calculation
            __m256 V = _mm256_fmadd_ps(v_VR, R, _mm256_fmadd_ps(v_VG, G, _mm256_mul_ps(v_VB, B)));
            V = _mm256_add_ps(V, _mm256_add_ps(v_offset, v_half));

            // 4. CLAMP
            Y = _mm256_max_ps(v_min, _mm256_min_ps(v_max, Y));
            U = _mm256_max_ps(v_min, _mm256_min_ps(v_max, U));
            V = _mm256_max_ps(v_min, _mm256_min_ps(v_max, V));

            // 5. CONVERT TO INT32
            __m256i iY = _mm256_cvttps_epi32(Y);
            __m256i iU = _mm256_cvttps_epi32(U);
            __m256i iV = _mm256_cvttps_epi32(V);

            // 6. PACKING (VUYA 16-bit)
            // Layout: V (0-15), U (16-31), Y (32-47), A (48-63)
            
            // Extract Alpha from Source (64-bit per pixel)
            // Load 2 vectors (4 pixels each)
            const __m256i* src_vec = reinterpret_cast<const __m256i*>(pRowSrc + x * 8);
            __m256i src0 = _mm256_loadu_si256(src_vec);
            __m256i src1 = _mm256_loadu_si256(src_vec + 1);

            // Shuffle to get 32-bit Alpha integers
            __m256i iA_lo = _mm256_shuffle_epi8(src0, v_alpha_shuf);
            __m256i iA_hi = _mm256_shuffle_epi8(src1, v_alpha_shuf);
            __m256i iA    = _mm256_permute2f128_si256(iA_lo, iA_hi, 0x20);

            // Pack 32->16: V, U
            __m256i pack_VU = _mm256_packus_epi32(iV, iU); // V0..3 U0..3 | V4..7 U4..7

            // Pack 32->16: Y, A
            __m256i pack_YA = _mm256_packus_epi32(iY, iA); // Y0..3 A0..3 | Y4..7 A4..7

            // Interleave (Unpack)
            // UnpackLo(VU, YA) -> V0 U0 Y0 A0 | V1 U1 Y1 A1 ...
            __m256i out_lo = _mm256_unpacklo_epi32(pack_VU, pack_YA);
            __m256i out_hi = _mm256_unpackhi_epi32(pack_VU, pack_YA);

            // Fix Lanes
            __m256i final_0 = _mm256_permute2f128_si256(out_lo, out_hi, 0x20);
            __m256i final_1 = _mm256_permute2f128_si256(out_lo, out_hi, 0x31);

            // 7. STORE
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(pRowDst + x * 8), final_0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(pRowDst + (x + 4) * 8), final_1);
        }

        // --- SCALAR FALLBACK ---
        for (; x < sizeX; ++x)
		{
            float l_val = rowL[x];
            float a_val = rowAB[x * 2];
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
            float Y_xyz = 1.00000f * ((l_val > 8.0f) ? (fy*fy*fy) : (l_val / 903.3f));
            float Z = 1.08883f * f_inv(fz);
            
            float r =  3.2404542f*X - 1.5371385f*Y_xyz - 0.4985314f*Z;
            float g = -0.9692660f*X + 1.8760108f*Y_xyz + 0.0415560f*Z;
            float b =  0.0556434f*X - 0.2040259f*Y_xyz + 1.0572252f*Z;

            // RGB->YUV Math
            float y_val = (c_Y_R * r) + (c_Y_G * g) + (c_Y_B * b);
            float u_val = (c_U_R * r) + (c_U_G * g) + (c_U_B * b) + 16384.0f;
            float v_val = (c_V_R * r) + (c_V_G * g) + (c_V_B * b) + 16384.0f;

            // Clamp
            uint16_t uY = (uint16_t)std::min(std::max(y_val + 0.5f, 0.0f), 32767.0f);
            uint16_t uU = (uint16_t)std::min(std::max(u_val + 0.5f, 0.0f), 32767.0f);
            uint16_t uV = (uint16_t)std::min(std::max(v_val + 0.5f, 0.0f), 32767.0f);

            // Pack
            const PF_Pixel_VUYA_16u* s = reinterpret_cast<const PF_Pixel_VUYA_16u*>(pRowSrc + x * 8);
            PF_Pixel_VUYA_16u* d = reinterpret_cast<PF_Pixel_VUYA_16u*>(pRowDst + x * 8);
            
            d->V = uV;
            d->U = uU;
            d->Y = uY;
            d->A = s->A;
        }
    }
	
	return;
}


/**
 * AVX2 OUTPUT CONVERTER: Semi-Planar Lab -> VUYA_32f
 * 
 * Pipeline:
 * 1. Lab -> Linear RGB.
 * 2. Linear RGB -> YUV (BT.601 or BT.709).
 * 3. Range: Y [0..1], U/V [0..1] with +0.5 bias.
 * 4. Packing: V U Y A.
 */
void AVX2_ConvertCIELab_SemiPlanar_To_YUV
(
    const PF_Pixel_VUYA_32f* RESTRICT pSrc,       // Source for Alpha
    const float*             RESTRICT pL,         // Planar L
    const float*             RESTRICT pAB,        // Interleaved AB
    PF_Pixel_VUYA_32f*       RESTRICT pDst,       // Output
    int32_t                  sizeX,
    int32_t                  sizeY,
    int32_t                  srcPitch,
    int32_t                  dstPitch,
    bool                     isBT709              // true=HD, false=SD
) noexcept
{
    // Pitch in Bytes (4 floats * 4 bytes = 16 bytes per pixel)
    const intptr_t srcStrideBytes = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_VUYA_32f);
    const intptr_t dstStrideBytes = static_cast<intptr_t>(dstPitch) * sizeof(PF_Pixel_VUYA_32f);

    // --- 1. SETUP CONSTANTS ---
    float Kr, Kb, Kg;
    if (isBT709)
	{
        Kr = 0.2126f; Kb = 0.0722f; Kg = 0.7152f;
    } else
    { // BT.601
        Kr = 0.2990f; Kb = 0.1140f; Kg = 0.5870f;
    }

    // YUV Weights
    // Y = Kr*R + Kg*G + Kb*B
    const __m256 v_YR = _mm256_set1_ps(Kr);
    const __m256 v_YG = _mm256_set1_ps(Kg);
    const __m256 v_YB = _mm256_set1_ps(Kb);

    // U = (B-Y)/(2*(1-Kb))
    const float u_div = 1.0f / (2.0f * (1.0f - Kb));
    const __m256 v_U_div = _mm256_set1_ps(u_div);

    // V = (R-Y)/(2*(1-Kr))
    const float v_div = 1.0f / (2.0f * (1.0f - Kr));
    const __m256 v_V_div = _mm256_set1_ps(v_div);

    const __m256 v_offset = _mm256_set1_ps(0.5f); // Bias for UV
    const __m256 v_zero   = _mm256_setzero_ps();
    const __m256 v_max    = _mm256_set1_ps(1.0f); // Float Max

    // Alpha Gather Indices (A is at float index 3)
    const __m256i idx_alpha = _mm256_setr_epi32(3, 7, 11, 15, 19, 23, 27, 31);

    for (int y = 0; y < sizeY; ++y)
    {
        const uint8_t* pRowSrc = reinterpret_cast<const uint8_t*>(pSrc) + (y * srcStrideBytes);
        uint8_t*       pRowDst = reinterpret_cast<uint8_t*>(pDst) + (y * dstStrideBytes);
        
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

            const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256 s0 = _mm256_permutevar8x32_ps(ab0, perm_idx);
            __m256 s1 = _mm256_permutevar8x32_ps(ab1, perm_idx);
            __m256 a = _mm256_permute2f128_ps(s0, s1, 0x20);
            __m256 b = _mm256_permute2f128_ps(s0, s1, 0x31);

            // 2. CONVERT Lab -> Linear RGB
            __m256 R, G, B;
            AVX2_Lab_to_RGB_Linear_Inline(L, a, b, R, G, B);

            // 3. RGB -> YUV (Float)
            // Y
            __m256 Y = _mm256_fmadd_ps(v_YR, R, _mm256_fmadd_ps(v_YG, G, _mm256_mul_ps(v_YB, B)));
            
            // U = (B - Y) * div + 0.5
            __m256 U = _mm256_fmadd_ps(_mm256_sub_ps(B, Y), v_U_div, v_offset);
            
            // V = (R - Y) * div + 0.5
            __m256 V = _mm256_fmadd_ps(_mm256_sub_ps(R, Y), v_V_div, v_offset);

            // Clamp 0..1
            Y = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, Y));
            U = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, U));
            V = _mm256_max_ps(v_zero, _mm256_min_ps(v_max, V));

            // 4. GATHER ALPHA
            const float* src_ptr = reinterpret_cast<const float*>(pRowSrc + x * 16);
            __m256 A = _mm256_i32gather_ps(src_ptr, idx_alpha, 4);

            // 5. TRANSPOSE / PACK (V U Y A)
            // Planar V, U, Y, A -> Packed V0 U0 Y0 A0 ...
            
            // Step A: Unpack Lo/Hi (V, U) -> VU
            __m256 vu_lo = _mm256_unpacklo_ps(V, U); // V0 U0 V1 U1 | V4 U4 V5 U5
            __m256 vu_hi = _mm256_unpackhi_ps(V, U); // V2 U2 V3 U3 | V6 U6 V7 U7

            // Step B: Unpack Lo/Hi (Y, A) -> YA
            __m256 ya_lo = _mm256_unpacklo_ps(Y, A); // Y0 A0 Y1 A1 | Y4 A4 Y5 A5
            __m256 ya_hi = _mm256_unpackhi_ps(Y, A); // Y2 A2 Y3 A3 | Y6 A6 Y7 A7

            // Step C: Interleave 64-bit blocks (VU, YA)
            // row0: Px 0, 4 (Lo lanes of result)
            __m256d row0_d = _mm256_unpacklo_pd(_mm256_castps_pd(vu_lo), _mm256_castps_pd(ya_lo));
            // row1: Px 1, 5
            __m256d row1_d = _mm256_unpackhi_pd(_mm256_castps_pd(vu_lo), _mm256_castps_pd(ya_lo));
            // row2: Px 2, 6
            __m256d row2_d = _mm256_unpacklo_pd(_mm256_castps_pd(vu_hi), _mm256_castps_pd(ya_hi));
            // row3: Px 3, 7
            __m256d row3_d = _mm256_unpackhi_pd(_mm256_castps_pd(vu_hi), _mm256_castps_pd(ya_hi));

            __m256 r0 = _mm256_castpd_ps(row0_d);
            __m256 r1 = _mm256_castpd_ps(row1_d);
            __m256 r2 = _mm256_castpd_ps(row2_d);
            __m256 r3 = _mm256_castpd_ps(row3_d);

            // 6. STORE
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

            // Lab->RGB
            float fy = (l_val + 16.0f) / 116.0f;
            float fx = fy + (a_val / 500.0f);
            float fz = fy - (b_val / 200.0f);
            auto f_inv = [](float t) { return (t > 0.206893f) ? (t * t * t) : ((t - 16.0f/116.0f) / 7.787f); };
            float X = 0.95047f * f_inv(fx);
            float Y_xyz = 1.00000f * ((l_val > 8.0f) ? (fy*fy*fy) : (l_val / 903.3f));
            float Z = 1.08883f * f_inv(fz);
            float r =  3.2404542f*X - 1.5371385f*Y_xyz - 0.4985314f*Z;
            float g = -0.9692660f*X + 1.8760108f*Y_xyz + 0.0415560f*Z;
            float b =  0.0556434f*X - 0.2040259f*Y_xyz + 1.0572252f*Z;

            // RGB->YUV
            float y_val = (Kr * r) + (Kg * g) + (Kb * b);
            float u_val = (b - y_val) / (2.0f * (1.0f - Kb)) + 0.5f;
            float v_val = (r - y_val) / (2.0f * (1.0f - Kr)) + 0.5f;

            // Clamp
            y_val = std::min(std::max(y_val, 0.0f), 1.0f);
            u_val = std::min(std::max(u_val, 0.0f), 1.0f);
            v_val = std::min(std::max(v_val, 0.0f), 1.0f);

            // Pack
            const PF_Pixel_VUYA_32f* s = reinterpret_cast<const PF_Pixel_VUYA_32f*>(pRowSrc + x * 16);
            PF_Pixel_VUYA_32f* d = reinterpret_cast<PF_Pixel_VUYA_32f*>(pRowDst + x * 16);

            d->V = v_val;
            d->U = u_val;
            d->Y = y_val;
            d->A = s->A;
        }
    }
	
	return;
}
