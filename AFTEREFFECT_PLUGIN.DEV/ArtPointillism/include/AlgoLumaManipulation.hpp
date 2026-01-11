#ifndef __IMAGE_LAB_ART_POINTILISM_LUMA_MANIPULATIONS__
#define __IMAGE_LAB_ART_POINTILISM_LUMA_MANIPULATIONS__

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include "AefxDevPatch.hpp"
#include "Common.hpp"
#include "CommonAuxPixFormat.hpp"

void CIELab_LumaInvert
(
    const float* RESTRICT pSrc,
    float*       RESTRICT pDst,
    int32_t      sizeX,
    int32_t      sizeY
) noexcept;

void LumaEdgeDetection
(
    const float* RESTRICT pSrc,
          float* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY
) noexcept;

inline float hmax_avx2 (__m256 v) noexcept
{
    __m256 y = _mm256_permute2f128_ps(v, v, 1); // Swap 128-bit lanes
    __m256 m1 = _mm256_max_ps(v, y);            // Max of lanes
    __m256 m2 = _mm256_permute_ps(m1, 0b01001110); // Swap inner floats
    __m256 m3 = _mm256_max_ps(m1, m2);
    __m256 m4 = _mm256_permute_ps(m3, 0b10110001); // Swap neighbors
    __m256 m5 = _mm256_max_ps(m3, m4);
    return _mm_cvtss_f32(_mm256_castps256_ps128(m5));
}

void MixAndNormalizeDensity
(
    const float* RESTRICT luma_src,
    const float* RESTRICT edge_src,
    float* RESTRICT target_dest,
    int pixel_count,
    float sensitivity
) noexcept;

#endif // __IMAGE_LAB_ART_POINTILISM_LUMA_MANIPULATIONS__