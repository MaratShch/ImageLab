#ifndef __AVX2_COLOR_CONVERSION_INTERFACES__
#define __AVX2_COLOR_CONVERSION_INTERFACES__

#include <immintrin.h>
#include <cstdint>
#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "FastAriphmetics.hpp"

// --- Stable Vectorized CIELab math functions ---
constexpr float c_Xn_inv    = 1.0f / 0.95047f;
constexpr float c_Zn_inv    = 1.0f / 1.08883f;
constexpr float c_delta_sq3 = 0.0088564516f; 
constexpr float c_lin_slope = 7.787037f;      
constexpr float c_lin_const = 16.0f / 116.0f; 


// Stable Cube Root for AVX2 (Householder's Method)
// Sufficiently accurate for 8-bit imaging and extremely fast
inline __m256 _mm256_safe_cbrt_ps(__m256 x) noexcept
{
    // We start with a safe linear seed for the range [0, 1]
    // y = 0.75x + 0.25 is a decent starting point for x^(1/3)
    __m256 y = _mm256_fmadd_ps(x, _mm256_set1_ps(0.75f), _mm256_set1_ps(0.25f));

    // Householder's iteration (3rd order):
    // y = y * (y^3 + 2x) / (2y^3 + x)
    for(int i = 0; i < 2; ++i)
    {
        __m256 y3 = _mm256_mul_ps(y, _mm256_mul_ps(y, y));
        __m256 num = _mm256_add_ps(y3, _mm256_add_ps(x, x));
        __m256 den = _mm256_add_ps(_mm256_add_ps(y3, y3), x);
        y = _mm256_mul_ps(y, _mm256_div_ps(num, den));
    }
    return y;
}

inline __m256 _mm256_lab_f_ps(__m256 t) noexcept
{
    // t is X/Xn, Y/Yn, or Z/Zn
    __m256 mask = _mm256_cmp_ps(t, _mm256_set1_ps(c_delta_sq3), _CMP_GT_OQ);
    __m256 cube_root = _mm256_safe_cbrt_ps(t);
    __m256 linear = _mm256_fmadd_ps(t, _mm256_set1_ps(c_lin_slope), _mm256_set1_ps(c_lin_const));
    return _mm256_blendv_ps(linear, cube_root, mask);
}

// --- HELPER: Lab -> Linear RGB (AVX2) ---
// Standard D65 conversion.
// Inputs: Planar L, a, b registers.
// Outputs: Planar R, G, B registers.
inline void AVX2_Lab_to_RGB_Linear_Inline
(
    __m256 L,
    __m256 a,
    __m256 b,
    __m256& R_out,
    __m256& G_out,
    __m256& B_out
) noexcept
{
    // Constants
    const __m256 c_16      = _mm256_set1_ps(16.0f);
    const __m256 c_116_inv = _mm256_set1_ps(1.0f / 116.0f);
    const __m256 c_500_inv = _mm256_set1_ps(1.0f / 500.0f);
    const __m256 c_200_inv = _mm256_set1_ps(1.0f / 200.0f);
    
    // Lab thresholds
    const __m256 c_epsilon   = _mm256_set1_ps(0.008856f); // (6/29)^3
    const __m256 c_kappa_inv = _mm256_set1_ps(1.0f / 903.3f); 
    const __m256 c_16_116    = _mm256_set1_ps(16.0f / 116.0f);
    const __m256 c_7787_inv  = _mm256_set1_ps(1.0f / 7.787f);

    // Reference White D65
    const __m256 D65_Xn = _mm256_set1_ps(0.95047f);
    const __m256 D65_Yn = _mm256_set1_ps(1.00000f);
    const __m256 D65_Zn = _mm256_set1_ps(1.08883f);

    // 1. Lab -> f(x,y,z)
    // fy = (L + 16) / 116
    __m256 fy = _mm256_mul_ps(_mm256_add_ps(L, c_16), c_116_inv);
    // fx = fy + a / 500
    __m256 fx = _mm256_add_ps(fy, _mm256_mul_ps(a, c_500_inv));
    // fz = fy - b / 200
    __m256 fz = _mm256_sub_ps(fy, _mm256_mul_ps(b, c_200_inv));

    // Calculate cubes
    __m256 fx3 = _mm256_mul_ps(fx, _mm256_mul_ps(fx, fx));
    __m256 fz3 = _mm256_mul_ps(fz, _mm256_mul_ps(fz, fz));
    
    // 2. XYZ Calculation
    // Y
    __m256 l_mask = _mm256_cmp_ps(L, _mm256_set1_ps(8.0f), _CMP_GT_OQ);
    __m256 fy3    = _mm256_mul_ps(fy, _mm256_mul_ps(fy, fy));
    __m256 y_lin  = _mm256_mul_ps(L, c_kappa_inv);
    __m256 Y      = _mm256_blendv_ps(y_lin, fy3, l_mask);

    // X
    __m256 x_mask = _mm256_cmp_ps(fx3, c_epsilon, _CMP_GT_OQ);
    // Linear approx: (fx - 16/116) / 7.787
    __m256 x_lin  = _mm256_mul_ps(_mm256_sub_ps(fx, c_16_116), c_7787_inv);
    __m256 xr     = _mm256_blendv_ps(x_lin, fx3, x_mask);
    __m256 X      = _mm256_mul_ps(xr, D65_Xn);

    // Z
    __m256 z_mask = _mm256_cmp_ps(fz3, c_epsilon, _CMP_GT_OQ);
    __m256 z_lin  = _mm256_mul_ps(_mm256_sub_ps(fz, c_16_116), c_7787_inv);
    __m256 zr     = _mm256_blendv_ps(z_lin, fz3, z_mask);
    __m256 Z      = _mm256_mul_ps(zr, D65_Zn);

    // 3. XYZ -> Linear RGB
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


// API's for convert between BGRA_8u and CIELab
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_BGRA_8u* RESTRICT pRGB,
    float*                  RESTRICT pL,
    float*                  RESTRICT pAB,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           srcPitch,
    const int32_t           labPitch
) noexcept;

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
) noexcept;


// API's for convert between ARGB_8u and CIELab
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_ARGB_8u* RESTRICT pRGB,
    float*                  RESTRICT pL,
    float*                  RESTRICT pAB,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           srcPitch,
    const int32_t           labPitch
) noexcept;

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
) noexcept;


// API's for convert between BGRA_16u and CIELab
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_BGRA_16u* RESTRICT pRGB,
    float*                   RESTRICT pL,
    float*                   RESTRICT pAB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            srcPitch,
    const int32_t            labPitch
) noexcept;

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
) noexcept;


// API's for convert between ARGB_16u and CIELab
void AVX2_ConvertCIELab_SemiPlanar_ToRgb
(
    const PF_Pixel_ARGB_16u* RESTRICT pSrc, 
    const float*             RESTRICT pL,   
    const float*             RESTRICT pAB,  
    PF_Pixel_ARGB_16u*       RESTRICT pDst, 
    int32_t sizeX,
	int32_t sizeY,
    int32_t           srcPitch,            // pitch of pSrc in PIXELS
    int32_t           dstPitch             // pitch of pDst in PIXELS 
) noexcept;

void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_ARGB_16u* RESTRICT pRGB,
    float*                   RESTRICT pL,
    float*                   RESTRICT pAB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            srcPitch,
    const int32_t            labPitch
) noexcept;


// API's for convert between BGRA_32f and CIELab
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_BGRA_32f* RESTRICT pRGB,
    float*                   RESTRICT pL,
    float*                   RESTRICT pAB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            srcPitch,
    const int32_t            labPitch
) noexcept;

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
) noexcept;


// API's for convert between ARGB_32f and CIELab
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_ARGB_32f* RESTRICT pRGB,
    float*                   RESTRICT pL,
    float*                   RESTRICT pAB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            srcPitch,
    const int32_t            labPitch
) noexcept;

void AVX2_ConvertCIELab_SemiPlanar_ToRgb
(
    const PF_Pixel_ARGB_32f* RESTRICT pSrc, 
    const float*             RESTRICT pL,   
    const float*             RESTRICT pAB,  
    PF_Pixel_ARGB_32f*       RESTRICT pDst, 
    int32_t sizeX,
	int32_t sizeY,
    int32_t srcPitch, 
    int32_t dstPitch
) noexcept;


// API's for convert between RGB_10u and CIELab
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_RGB_10u* RESTRICT pRGB,
    float*                  RESTRICT pL,
    float*                  RESTRICT pAB,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           srcPitch,
    const int32_t           labPitch
) noexcept;

void AVX2_ConvertCIELab_SemiPlanar_To_Rgb
(
    const float*      RESTRICT pL,        // source L (planar)
    const float*      RESTRICT pAB,       // source ab (interleaved)
    PF_Pixel_RGB_10u* RESTRICT pDst,      // destination buffer
    int32_t           sizeX,              // width
    int32_t           sizeY,              // height
    int32_t           dstPitchBytes       // pitch of pDst in BYTES
) noexcept;


// API's for convert from VUYA format to semi-planar CIELab
void AVX2_ConvertVuyaToCIELab_SemiPlanar
(
    const PF_Pixel_VUYA_8u* RESTRICT pVUYA,
    float*                  RESTRICT pL,
    float*                  RESTRICT pAB,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           srcPitch,
    const int32_t           labPitch,
    const bool              isBT709
) noexcept;

void AVX2_ConvertVuyaToCIELab_SemiPlanar
(
    const PF_Pixel_VUYA_16u* RESTRICT pVUYA,
    float*                   RESTRICT pL,
    float*                   RESTRICT pAB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            srcPitch,
    const int32_t            labPitch,
    const bool               isBT709
) noexcept;

void AVX2_ConvertVuyaToCIELab_SemiPlanar
(
    const PF_Pixel_VUYA_32f* RESTRICT pVUYA,
    float*                   RESTRICT pL,
    float*                   RESTRICT pAB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            srcPitch,
    const int32_t            labPitch,
    const bool               isBT709
) noexcept;

// API's for convert semi-planar CIELab to VUYA format
void AVX2_ConvertCIELab_SemiPlanar_To_YUV
(
    const PF_Pixel_VUYA_8u* RESTRICT pSrc,       // Source for Alpha
    const float*            RESTRICT pL,         // Planar L
    const float*            RESTRICT pAB,        // Interleaved AB
    PF_Pixel_VUYA_8u*       RESTRICT pDst,       // Output
    int32_t                 sizeX,
    int32_t                 sizeY,
    int32_t                 srcPitchBytes,       // Pitch in BYTES
    int32_t                 dstPitchBytes,       // Pitch in BYTES
    bool                    isBT709              // true=HD, false=SD
) noexcept;

void AVX2_ConvertCIELab_SemiPlanar_To_YUV
(
    const PF_Pixel_VUYA_16u* RESTRICT pSrc,       // Source for Alpha
    const float*             RESTRICT pL,         // Planar L
    const float*             RESTRICT pAB,        // Interleaved AB
    PF_Pixel_VUYA_16u*       RESTRICT pDst,       // Output
    int32_t                  sizeX,
    int32_t                  sizeY,
    int32_t                  srcPitchBytes,       // Pitch in BYTES
    int32_t                  dstPitchBytes,       // Pitch in BYTES
    bool                     isBT709              // true=HD, false=SD
) noexcept;

void AVX2_ConvertCIELab_SemiPlanar_To_YUV
(
    const PF_Pixel_VUYA_32f* RESTRICT pSrc,       // Source for Alpha
    const float*             RESTRICT pL,         // Planar L
    const float*             RESTRICT pAB,        // Interleaved AB
    PF_Pixel_VUYA_32f*       RESTRICT pDst,       // Output
    int32_t                  sizeX,
    int32_t                  sizeY,
    int32_t                  srcPitch,            // Pitch in PIXELS
    int32_t                  dstPitch,            // Pitch in PIXELS
    bool                     isBT709              // true=HD, false=SD
) noexcept;



#endif // __AVX2_COLOR_CONVERSION_INTERFACES__