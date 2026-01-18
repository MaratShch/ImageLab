#ifndef __IMAGE_LAB_ART_POINTILISM_ALGORITHM_OUTPUT__
#define __IMAGE_LAB_ART_POINTILISM_ALGORITHM_OUTPUT__

#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include "Common.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ArtPointillismControl.hpp"

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
) noexcept;


// --- CONSTANTS ---
constexpr float LAB_EPSILON = 0.008856f;
constexpr float LAB_KAPPA   = 903.3f;
constexpr float D65_Xn      = 0.95047f;
constexpr float D65_Yn      = 1.00000f;
constexpr float D65_Zn      = 1.08883f;


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

#endif // __IMAGE_LAB_ART_POINTILISM_ALGORITHM_OUTPUT__