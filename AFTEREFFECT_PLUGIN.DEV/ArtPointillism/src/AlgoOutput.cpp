#include "AlgoOutput.hpp"

const fRGB* AlgoOutput
(
    float* RESTRICT canvas_lab,        // [In/Out] The rendered image
    const float* RESTRICT source_lab,  // [Input] Original image (for blending)
    int32_t width,
	int32_t height,
    const PontillismControls& params
)
{
    const int32_t num_pixels = width * height;

    // --- STEP A: BLENDING (Detail Recovery) ---
    // If Opacity > 0, we mix the source back in.
    if (params.Opacity > 0)
	{
        float blend_factor = static_cast<float>(params.Opacity) / 100.0f;
        
        for (int32_t i = 0; i < num_pixels; ++i)
		{
            Mix_Lab_Pixel(&canvas_lab[i*3], &source_lab[i*3], blend_factor);
        }
    }

    // --- STEP B: COLOR SPACE CONVERSION (Lab -> Linear RGB) ---
    // We reuse the canvas_lab buffer to store Linear RGB to save memory.
    // (Assuming you have a function Convert_Lab_to_LinearRGB_Buffer)
    
    // Note: If you don't have a buffer-wide converter, loop it:
    for (int32_t i = 0; i < num_pixels; ++i)
	{
        float L = canvas_lab[i*3+0];
        float a = canvas_lab[i*3+1];
        float b = canvas_lab[i*3+2];
        
        float R, G, B;
        // Call your single-pixel scalar function
        Convert_CIELab_to_LinearRGB(L, a, b, R, G, B);
        
        canvas_lab[i*3+0] = R;
        canvas_lab[i*3+1] = G;
        canvas_lab[i*3+2] = B;
    }

    // --- STEP C: PACKING (Linear RGB -> sRGB BGRA 8-bit) ---
    // Cast the float buffer to your fRGB struct pointer for type safety
    fRGB* linear_rgb_ptr = reinterpret_cast<fRGB*>(canvas_lab);

	return linear_rgb_ptr;
}


/**
 * PHASE 5: OUTPUT GENERATION
 * 
 * 1. Blends the Rendered Canvas with the Original Source (Detail Recovery).
 * 2. Converts Lab -> Linear RGB.
 * 3. Calls the Packer (Linear RGB -> sRGB BGRA).
 * 
 * Adapted for:
 * - Canvas: Interleaved Lab (float[3])
 * - Source: Split Lab (L: float[1], ab: float[2])
 */
const fRGB* AlgoOutput
(
    float* RESTRICT canvas_lab,      // [In/Out] Rendered Image (Interleaved)
    const float* RESTRICT src_L,     // [Input] Original Luma (Planar)
    const float* RESTRICT src_ab,    // [Input] Original Chroma (Interleaved)
    int width, int height,
    const PontillismControls& params
) 
{
    const int num_pixels = width * height;

    // --- STEP A: BLENDING & CONVERSION ---
    // We combine Blending and Lab->RGB conversion in one loop 
    // to maximize cache efficiency (reading source/canvas once).
    
    // Pre-calculate blending factors
    // params.Opacity is 0..100. 
    // 0 = Effect Only (Default). 100 = Source Only.
    const bool do_blend = (params.Opacity > 0);
    const float blend_src = (float)params.Opacity / 100.0f;
    const float blend_eff = 1.0f - blend_src;

    for (int i = 0; i < num_pixels; ++i)
    {
        // 1. Read Canvas (Interleaved)
        // This is the "Paint" layer
        float L = canvas_lab[i * 3 + 0];
        float a = canvas_lab[i * 3 + 1];
        float b = canvas_lab[i * 3 + 2];

        // 2. Perform Blending (If requested)
        if (do_blend)
        {
            // Read Source (Split Buffers)
            float s_L = src_L[i];
            float s_a = src_ab[i * 2 + 0];
            float s_b = src_ab[i * 2 + 1];

            // Mix
            L = (L * blend_eff) + (s_L * blend_src);
            a = (a * blend_eff) + (s_a * blend_src);
            b = (b * blend_eff) + (s_b * blend_src);
        }

        // 3. Convert CIELab -> Linear RGB (Scalar)
        // We use local variables to avoid reading/writing memory twice
        float R_lin, G_lin, B_lin;
        
        Convert_CIELab_to_LinearRGB(L, a, b, R_lin, G_lin, B_lin);

        // 4. Write back to Canvas Buffer
        // We reuse 'canvas_lab' as the holding tank for Linear RGB data
        // so the Packer can read contiguous memory.
        canvas_lab[i * 3 + 0] = R_lin;
        canvas_lab[i * 3 + 1] = G_lin;
        canvas_lab[i * 3 + 2] = B_lin;
    }

    // --- STEP B: PACKING (Linear RGB -> sRGB BGRA) ---
    // Now 'canvas_lab' contains Linear RGB floats.
    // We cast the pointer to fRGB* (assuming it matches struct layout float{R,G,B})
    const fRGB* linear_rgb_ptr = reinterpret_cast<const fRGB*>(canvas_lab);

	return linear_rgb_ptr;
}

/**
 * PHASE 5 FINAL PACKER (AVX2)
 * Operations:
 * 1. Gather Canvas (Interleaved).
 * 2. Load Source (Split L + AB).
 * 3. Blend (L_out = L_canvas * (1-mix) + L_src * mix).
 * 4. Convert Lab -> Linear RGB.
 * 5. Scale & Quantize (No Gamma).
 * 6. Merge with Original Alpha.
 * 7. Pack BGRA.
 */
void Convert_Result_to_BGRA_AVX2
(
    const PF_Pixel_BGRA_8u*   RESTRICT src1,      // Original (Alpha)
    const float*              RESTRICT canvas_lab,// Canvas (Interleaved)
    const float*              RESTRICT src_L,     // Source L (Planar)
    const float*              RESTRICT src_ab,    // Source AB (Interleaved)
    PF_Pixel_BGRA_8u*         RESTRICT dst,       // Output
    int32_t sizeX, 
    int32_t sizeY, 
    int32_t srcPitch, // in pixels
    int32_t dstPitch, // in pixels
    const PontillismControls& params
)
{
    // --- SETUP BLENDING FACTORS ---
    const bool do_blend = (params.Opacity > 0);
    // Opacity 0 = Result Only (0.0). Opacity 100 = Source Only (1.0).
    const float f_mix = static_cast<float>(params.Opacity) / 100.0f; 
    
    const __m256 mix_src   = _mm256_set1_ps(f_mix);
    const __m256 mix_canvas= _mm256_set1_ps(1.0f - f_mix);

    // --- SETUP GATHER INDICES ---
    // Access L,a,b from Interleaved Canvas [L a b L a b...]
    // Stride 3 floats (12 bytes). 8 pixels.
    const __m256i idx_base = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);

    // --- SETUP PACKING CONSTANTS ---
    const __m256 scale_255 = _mm256_set1_ps(255.0f);
    const __m256 half      = _mm256_set1_ps(0.5f);
    const __m256i alpha_mask = _mm256_set1_epi32(0xFF000000);

    for (int y = 0; y < sizeY; ++y)
    {
        // Pointers for this row
        const PF_Pixel_BGRA_8u* row_src1 = src1 + (y * srcPitch);
        PF_Pixel_BGRA_8u*       row_dst  = dst + (y * dstPitch);
        
        const float* row_canvas = canvas_lab + (y * sizeX * 3);
        const float* row_src_L  = src_L + (y * sizeX);
        const float* row_src_ab = src_ab + (y * sizeX * 2);

        int x = 0;
        
        // --- AVX2 LOOP (8 pixels) ---
        for (; x <= sizeX - 8; x += 8)
        {
            // 1. LOAD CANVAS LAB (Interleaved -> Planar)
            // L_c, a_c, b_c
            __m256 L_c = _mm256_i32gather_ps(row_canvas + x*3 + 0, idx_base, 4);
            __m256 a_c = _mm256_i32gather_ps(row_canvas + x*3 + 1, idx_base, 4);
            __m256 b_c = _mm256_i32gather_ps(row_canvas + x*3 + 2, idx_base, 4);

            // 2. BLEND WITH SOURCE (If Needed)
            if (do_blend)
            {
                // Load Source L (Planar - Contiguous)
                __m256 L_s = _mm256_loadu_ps(row_src_L + x);
                
                // Load Source AB (Interleaved - needs shuffle)
                // Load 16 floats (8 pixels * 2 channels)
                // Layout: [a0 b0 a1 b1 ... a7 b7]
                __m256 ab0 = _mm256_loadu_ps(row_src_ab + x*2);     // Px 0-3
                __m256 ab1 = _mm256_loadu_ps(row_src_ab + x*2 + 8); // Px 4-7
                
                // De-interleave using shuffle/permute
                // This is a bit verbose, but standard technique for complex numbers / paired data
                // Shuffle to get [a0 a1 a2 a3] and [b0 b1 b2 b3]
                __m256 i_a0 = _mm256_permutevar8x32_ps(ab0, _mm256_setr_epi32(0,2,4,6, 1,3,5,7));
                __m256 i_a1 = _mm256_permutevar8x32_ps(ab1, _mm256_setr_epi32(0,2,4,6, 1,3,5,7));
                
                // Extract lower 128 bits (a's) and upper 128 bits (b's)
                __m256 a_s = _mm256_permute2f128_ps(i_a0, i_a1, 0x20); // Lo from a0, Lo from a1
                __m256 b_s = _mm256_permute2f128_ps(i_a0, i_a1, 0x31); // Hi from a0, Hi from a1

                // Perform Mix
                L_c = _mm256_fmadd_ps(L_c, mix_canvas, _mm256_mul_ps(L_s, mix_src));
                a_c = _mm256_fmadd_ps(a_c, mix_canvas, _mm256_mul_ps(a_s, mix_src));
                b_c = _mm256_fmadd_ps(b_c, mix_canvas, _mm256_mul_ps(b_s, mix_src));
            }

            // 3. CONVERT LAB -> LINEAR RGB
            __m256 R_lin, G_lin, B_lin;
            AVX2_Lab_to_LinearRGB(L_c, a_c, b_c, R_lin, G_lin, B_lin);

            // 4. QUANTIZE (Scale 0-255, no Gamma)
            __m256i i_r = _mm256_cvttps_epi32(_mm256_fmadd_ps(R_lin, scale_255, half));
            __m256i i_g = _mm256_cvttps_epi32(_mm256_fmadd_ps(G_lin, scale_255, half));
            __m256i i_b = _mm256_cvttps_epi32(_mm256_fmadd_ps(B_lin, scale_255, half));

            // 5. PACKING (BGRA)
            // Load Original Alpha
            __m256i src_px = _mm256_loadu_si256((const __m256i*)(row_src1 + x));
            __m256i alpha  = _mm256_and_si256(src_px, alpha_mask);

            // Shift and Or
            __m256i out_px = i_b; // B at pos 0
            out_px = _mm256_or_si256(out_px, _mm256_slli_epi32(i_g, 8));  // G at pos 8
            out_px = _mm256_or_si256(out_px, _mm256_slli_epi32(i_r, 16)); // R at pos 16
            out_px = _mm256_or_si256(out_px, alpha);                      // A at pos 24

            // 6. STORE
            _mm256_storeu_si256((__m256i*)(row_dst + x), out_px);
        }

        // --- SCALAR FALLBACK ---
        for (; x < sizeX; ++x)
        {
            float L = row_canvas[x*3+0];
            float a = row_canvas[x*3+1];
            float b = row_canvas[x*3+2];

            if (do_blend)
            {
                float s_L = row_src_L[x];
                float s_a = row_src_ab[x*2+0];
                float s_b = row_src_ab[x*2+1];
                L = L * (1.0f - f_mix) + s_L * f_mix;
                a = a * (1.0f - f_mix) + s_a * f_mix;
                b = b * (1.0f - f_mix) + s_b * f_mix;
            }

            // Scalar Math (Use standard constants or copy inline logic)
            // (Copying shortened logic here for brevity, assume standard Scalar implementation exists)
            float fy = (L + 16.0f) / 116.0f;
            float fx = fy + a / 500.0f;
            float fz = fy - b / 200.0f;
            
            float fx3 = fx*fx*fx; float fz3 = fz*fz*fz;
            float xr = (fx3 > LAB_EPSILON) ? fx3 : (fx - 16.0f/116.0f)/7.787f;
            float yr = (L > 8.0f) ? ((L+16.0f)/116.0f)*((L+16.0f)/116.0f)*((L+16.0f)/116.0f) : L/903.3f;
            float zr = (fz3 > LAB_EPSILON) ? fz3 : (fz - 16.0f/116.0f)/7.787f;

            float X = xr * D65_Xn; float Y = yr * D65_Yn; float Z = zr * D65_Zn;

            float R =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
            float G = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
            float B =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;

            uint8_t ur = (uint8_t)std::min(std::max(R * 255.0f + 0.5f, 0.0f), 255.0f);
            uint8_t ug = (uint8_t)std::min(std::max(G * 255.0f + 0.5f, 0.0f), 255.0f);
            uint8_t ub = (uint8_t)std::min(std::max(B * 255.0f + 0.5f, 0.0f), 255.0f);
            
            PF_Pixel_BGRA_8u out;
            out.B = ub; out.G = ug; out.R = ur;
            out.A = row_src1[x].A;
            row_dst[x] = out;
        }
    }
    
    return;
}


void Convert_Result_to_ARGB_AVX2
(
    const PF_Pixel_ARGB_8u* RESTRICT src1,      // Original Source (for Alpha)
    const float*            RESTRICT canvas_lab,// Rendered Lab (Interleaved)
    const float*            RESTRICT src_L,     // Original Lab L (Planar)
    const float*            RESTRICT src_ab,    // Original Lab AB (Interleaved)
    PF_Pixel_ARGB_8u*       RESTRICT dst,       // Destination Buffer
    int32_t sizeX, 
    int32_t sizeY, 
    int32_t srcPitchBytes, 
    int32_t dstPitchBytes, 
    const PontillismControls& params
)
{
    // --- SETUP BLENDING ---
    const bool do_blend = (params.Opacity > 0);
    const float f_mix_src = (float)params.Opacity / 100.0f;
    const float f_mix_eff = 1.0f - f_mix_src;

    const __m256 mix_src    = _mm256_set1_ps(f_mix_src);
    const __m256 mix_canvas = _mm256_set1_ps(f_mix_eff);

    // --- SETUP PACKING ---
    const __m256 scale_255 = _mm256_set1_ps(255.0f);
    const __m256 half      = _mm256_set1_ps(0.5f);
    
    // Mask for Alpha in ARGB struct (A is at byte 0) -> 0x000000FF
    const __m256i alpha_mask = _mm256_set1_epi32(0x000000FF);
    
    // Gather indices for RGB Stride (3 floats = 12 bytes)
    const __m256i idx_base = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);

    // Cast pointers for byte-wise pitch arithmetic
    const uint8_t* byte_src_ptr = (const uint8_t*)src1;
    uint8_t*       byte_dst_ptr = (uint8_t*)dst;

    for (int y = 0; y < sizeY; ++y)
    {
        
        // Row Pointers
        const PF_Pixel_ARGB_8u* row_src_alpha = (const PF_Pixel_ARGB_8u*)(byte_src_ptr + (y * srcPitchBytes));
        PF_Pixel_ARGB_8u*       row_dst       = (PF_Pixel_ARGB_8u*)(byte_dst_ptr + (y * dstPitchBytes));
        
        const float* row_canvas = canvas_lab + (y * sizeX * 3);
        const float* row_src_L  = src_L + (y * sizeX);
        const float* row_src_ab = src_ab + (y * sizeX * 2);

        int x = 0;
        
        // --- AVX2 LOOP (8 pixels) ---
        for (; x <= sizeX - 8; x += 8)
        {
            
            // 1. GATHER CANVAS LAB
            __m256 L_c = _mm256_i32gather_ps(row_canvas + x*3 + 0, idx_base, 4);
            __m256 a_c = _mm256_i32gather_ps(row_canvas + x*3 + 1, idx_base, 4);
            __m256 b_c = _mm256_i32gather_ps(row_canvas + x*3 + 2, idx_base, 4);

            // 2. BLEND (Optional)
            if (do_blend)
            {
                __m256 L_s = _mm256_loadu_ps(row_src_L + x);
                
                __m256 ab0 = _mm256_loadu_ps(row_src_ab + x*2);
                __m256 ab1 = _mm256_loadu_ps(row_src_ab + x*2 + 8);
                
                __m256 i_a0 = _mm256_permutevar8x32_ps(ab0, _mm256_setr_epi32(0,2,4,6, 1,3,5,7));
                __m256 i_a1 = _mm256_permutevar8x32_ps(ab1, _mm256_setr_epi32(0,2,4,6, 1,3,5,7));
                
                __m256 a_s = _mm256_permute2f128_ps(i_a0, i_a1, 0x20);
                __m256 b_s = _mm256_permute2f128_ps(i_a0, i_a1, 0x31);

                L_c = _mm256_fmadd_ps(L_c, mix_canvas, _mm256_mul_ps(L_s, mix_src));
                a_c = _mm256_fmadd_ps(a_c, mix_canvas, _mm256_mul_ps(a_s, mix_src));
                b_c = _mm256_fmadd_ps(b_c, mix_canvas, _mm256_mul_ps(b_s, mix_src));
            }

            // 3. CONVERT LAB -> LINEAR RGB
            __m256 R_lin, G_lin, B_lin;
            AVX2_Lab_to_LinearRGB (L_c, a_c, b_c, R_lin, G_lin, B_lin);

            // 4. SCALE & QUANTIZE
            __m256i i_r = _mm256_cvttps_epi32(_mm256_fmadd_ps(R_lin, scale_255, half));
            __m256i i_g = _mm256_cvttps_epi32(_mm256_fmadd_ps(G_lin, scale_255, half));
            __m256i i_b = _mm256_cvttps_epi32(_mm256_fmadd_ps(B_lin, scale_255, half));

            // 5. PACKING ARGB
            // Source Load
            __m256i src_argb = _mm256_loadu_si256((const __m256i*)(row_src_alpha + x));
            // Keep Alpha (lowest 8 bits)
            __m256i alpha = _mm256_and_si256(src_argb, alpha_mask);

            // Construct 0xBBGGRRAA (Little Endian int32)
            // A at 0 (No shift)
            // R at 8 (Shift 8)
            // G at 16 (Shift 16)
            // B at 24 (Shift 24)
            
            __m256i out_px = alpha;
            out_px = _mm256_or_si256(out_px, _mm256_slli_epi32(i_r, 8));
            out_px = _mm256_or_si256(out_px, _mm256_slli_epi32(i_g, 16));
            out_px = _mm256_or_si256(out_px, _mm256_slli_epi32(i_b, 24));

            // 6. STORE
            _mm256_storeu_si256((__m256i*)(row_dst + x), out_px);
        }

        // --- SCALAR FALLBACK ---
        for (; x < sizeX; ++x)
        {
            float L = row_canvas[x*3+0];
            float a = row_canvas[x*3+1];
            float b = row_canvas[x*3+2];

            if (do_blend)
            {
                float s_L = row_src_L[x];
                float s_a = row_src_ab[x*2+0];
                float s_b = row_src_ab[x*2+1];
                L = L * f_mix_eff + s_L * f_mix_src;
                a = a * f_mix_eff + s_a * f_mix_src;
                b = b * f_mix_eff + s_b * f_mix_src;
            }

            // Scalar Lab->RGB logic (Standard copy)
            float fy = (L + 16.0f) / 116.0f;
            float fx = fy + a / 500.0f;
            float fz = fy - b / 200.0f;
            
            float fx3 = fx*fx*fx; float fz3 = fz*fz*fz;
            float xr = (fx3 > 0.008856f) ? fx3 : (fx - 16.0f/116.0f)/7.787f;
            float yr = (L > 8.0f) ? ((L+16.0f)/116.0f)*((L+16.0f)/116.0f)*((L+16.0f)/116.0f) : L/903.3f;
            float zr = (fz3 > 0.008856f) ? fz3 : (fz - 16.0f/116.0f)/7.787f;

            float X = xr * 0.95047f; float Y = yr * 1.00000f; float Z = zr * 1.08883f;

            float R =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
            float G = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
            float B =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;

            uint8_t ur = (uint8_t)std::min(std::max(R * 255.0f + 0.5f, 0.0f), 255.0f);
            uint8_t ug = (uint8_t)std::min(std::max(G * 255.0f + 0.5f, 0.0f), 255.0f);
            uint8_t ub = (uint8_t)std::min(std::max(B * 255.0f + 0.5f, 0.0f), 255.0f);
            
            PF_Pixel_ARGB_8u out;
            out.A = row_src_alpha[x].A;
            out.R = ur; 
            out.G = ug; 
            out.B = ub;
            row_dst[x] = out;
        }
    }
    
    return;
}