#ifndef __IMAGE_LAB_ART_POINTILISM_ALGO_OUTPUT__
#define __IMAGE_LAB_ART_POINTILISM_ALGO_OUTPUT__

#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ArtPointillismControl.hpp"


// --- CONSTANTS ---
constexpr float LAB_EPSILON = 0.008856f;
constexpr float LAB_KAPPA   = 903.3f;
constexpr float D65_Xn      = 0.95047f;
constexpr float D65_Yn      = 1.00000f;
constexpr float D65_Zn      = 1.08883f;


const fRGB* AlgoOutput
(
    float* RESTRICT canvas_lab,      // [In/Out] Rendered Image (Interleaved)
    const float* RESTRICT src_L,     // [Input] Original Luma (Planar)
    const float* RESTRICT src_ab,    // [Input] Original Chroma (Interleaved)
    int width, int height,
    const PontillismControls& params
);

const fRGB* AlgoOutput
(
    float* RESTRICT canvas_lab,        // [In/Out] The rendered image
    const float* RESTRICT source_lab,  // [Input] Original image (for blending)
    int32_t width,
	int32_t height,
    const PontillismControls& params
);

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
);

// Converts 8 pixels of Lab to Linear RGB (Planar outputs)
// Note: Input Lab is Planar here (passed as separate registers)
inline void AVX2_Lab_to_LinearRGB
(
    __m256 L, __m256 a, __m256 b,
    __m256& R_out, __m256& G_out, __m256& B_out
) noexcept
{
    // Constants
    const __m256 c_16_116 = _mm256_set1_ps(16.0f / 116.0f);
    const __m256 c_1_116  = _mm256_set1_ps(1.0f / 116.0f);
    const __m256 c_1_500  = _mm256_set1_ps(1.0f / 500.0f);
    const __m256 c_1_200  = _mm256_set1_ps(1.0f / 200.0f);
    const __m256 c_epsilon= _mm256_set1_ps(LAB_EPSILON);
    const __m256 c_7_787  = _mm256_set1_ps(1.0f / 7.787f); // Inverse for multiply
    
    // --- Step 1: Lab -> f(x), f(y), f(z) ---
    // fy = (L + 16) / 116
    __m256 fy = _mm256_mul_ps(_mm256_add_ps(L, _mm256_set1_ps(16.0f)), c_1_116);
    // fx = fy + a / 500
    __m256 fx = _mm256_add_ps(fy, _mm256_mul_ps(a, c_1_500));
    // fz = fy - b / 200
    __m256 fz = _mm256_sub_ps(fy, _mm256_mul_ps(b, c_1_200));

    // Calculate cubes
    __m256 fx3 = _mm256_mul_ps(fx, _mm256_mul_ps(fx, fx));
    __m256 fz3 = _mm256_mul_ps(fz, _mm256_mul_ps(fz, fz));
    
    // Define L threshold: L > (Kappa * Epsilon) -> 903.3 * 0.008856 = 8.0
    __m256 l_mask = _mm256_cmp_ps(L, _mm256_set1_ps(8.0f), _CMP_GT_OQ);
    
    // --- Step 2: Calculate X, Y, Z ---
    
    // Y Calculation
    // if (L > 8) Y = ((L+16)/116)^3  [which is fy*fy*fy]
    // else       Y = L / 903.3
    __m256 fy3 = _mm256_mul_ps(fy, _mm256_mul_ps(fy, fy));
    __m256 y_linear = _mm256_mul_ps(L, _mm256_set1_ps(1.0f / LAB_KAPPA));
    __m256 Y = _mm256_blendv_ps(y_linear, fy3, l_mask);

    // X Calculation
    // if (fx^3 > epsilon) X = fx^3
    // else                X = (fx - 16/116) / 7.787
    __m256 x_mask = _mm256_cmp_ps(fx3, c_epsilon, _CMP_GT_OQ);
    __m256 x_linear = _mm256_mul_ps(_mm256_sub_ps(fx, c_16_116), c_7_787);
    __m256 xr = _mm256_blendv_ps(x_linear, fx3, x_mask);
    __m256 X = _mm256_mul_ps(xr, _mm256_set1_ps(D65_Xn));

    // Z Calculation
    __m256 z_mask = _mm256_cmp_ps(fz3, c_epsilon, _CMP_GT_OQ);
    __m256 z_linear = _mm256_mul_ps(_mm256_sub_ps(fz, c_16_116), c_7_787);
    __m256 zr = _mm256_blendv_ps(z_linear, fz3, z_mask);
    __m256 Z = _mm256_mul_ps(zr, _mm256_set1_ps(D65_Zn));

    // --- Step 3: XYZ -> Linear RGB (D65) ---
    // R =  3.2404542*X - 1.5371385*Y - 0.4985314*Z
    R_out = _mm256_fmadd_ps(_mm256_set1_ps(3.2404542f), X,
            _mm256_fmadd_ps(_mm256_set1_ps(-1.5371385f), Y,
            _mm256_mul_ps(_mm256_set1_ps(-0.4985314f), Z)));

    // G = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
    G_out = _mm256_fmadd_ps(_mm256_set1_ps(-0.9692660f), X,
            _mm256_fmadd_ps(_mm256_set1_ps(1.8760108f), Y,
            _mm256_mul_ps(_mm256_set1_ps(0.0415560f), Z)));

    // B =  0.0556434*X - 0.2040259*Y + 1.0572252*Z
    B_out = _mm256_fmadd_ps(_mm256_set1_ps(0.0556434f), X,
            _mm256_fmadd_ps(_mm256_set1_ps(-0.2040259f), Y,
            _mm256_mul_ps(_mm256_set1_ps(1.0572252f), Z)));

    return;
}

// Inline helper for blending
inline void Mix_Lab_Pixel
(
    float* RESTRICT dest, 
    const float* RESTRICT src, 
    float alpha // 0.0 = Result Only, 1.0 = Source Only
) noexcept
{
    const float inv_alpha = 1.0f - alpha;
    dest[0] = (dest[0] * inv_alpha) + (src[0] * alpha);
    dest[1] = (dest[1] * inv_alpha) + (src[1] * alpha);
    dest[2] = (dest[2] * inv_alpha) + (src[2] * alpha);
}


/**
 * Converts a single CIELab color to Linear RGB (D65 sRGB space).
 * Does NOT apply Gamma correction.
 * 
 * @param L_in  Input Luminance (0.0 - 100.0)
 * @param a_in  Input Chroma A
 * @param b_in  Input Chroma B
 * @param R_out [Output] Linear Red
 * @param G_out [Output] Linear Green
 * @param B_out [Output] Linear Blue
 */
inline void Convert_CIELab_to_LinearRGB
(
    float L_in, float a_in, float b_in,
    float& R_out, float& G_out, float& B_out
)
{
    // --- STEP 1: Lab -> XYZ ---
    
    // Compute f(y), f(x), f(z)
    float fy = (L_in + 16.0f) / 116.0f;
    float fx = (a_in / 500.0f) + fy;
    float fz = fy - (b_in / 200.0f);

    float fx3 = fx * fx * fx;
    float fz3 = fz * fz * fz;

    // Check thresholds (Inverse of the forward logic)
    // If L is very small, we use linear slope logic
    float xr = (fx3 > LAB_EPSILON) ? fx3 : ((116.0f * fx - 16.0f) / 116.0f); // Simplification: actually (fx - 16/116) / 7.787
    // Precise inverse linear: (fx - 16/116) * (108/841) ??? 
    // Let's use the standard form: (val - 16/116) / 7.787
    if (fx3 <= LAB_EPSILON) xr = (fx - 16.0f/116.0f) / 7.787f;
    
    float yr = (L_in > (LAB_KAPPA * LAB_EPSILON)) ? 
               ((L_in + 16.0f) / 116.0f) * ((L_in + 16.0f) / 116.0f) * ((L_in + 16.0f) / 116.0f) : 
               (L_in / LAB_KAPPA);

    float zr = (fz3 > LAB_EPSILON) ? fz3 : ((fz - 16.0f/116.0f) / 7.787f);

    // Scale by Reference White
    float X = xr * D65_Xn;
    float Y = yr * D65_Yn;
    float Z = zr * D65_Zn;

    // --- STEP 2: XYZ -> Linear RGB ---
    // Standard sRGB D65 Matrix multiplication
    
    R_out =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    G_out = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    B_out =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;

    // --- STEP 3: Cleanup ---
    // Values theoretically can slightly exceed [0, 1] due to gamut differences or floating point precision.
    // However, usually we clamp later during packing. 
    // Here we leave them raw Linear Float values for the blending stage.
}




#endif // __IMAGE_LAB_ART_POINTILISM_ALGO_OUTPUT__