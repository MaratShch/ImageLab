#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "FastAriphmetics.hpp"


// -----------------------------------------------------------------------------------------
// CONSTANTS (Scaled 1/255 for direct 8-bit conversion)
// -----------------------------------------------------------------------------------------
// X_coeffs = (Matrix_X / 0.95047) / 255.0
constexpr float K_XR = 0.00170177f;
constexpr float K_XG = 0.00147537f;
constexpr float K_XB = 0.00074442f;

// Y_coeffs = (Matrix_Y / 1.00000) / 255.0
constexpr float K_YR = 0.00083401f;
constexpr float K_YG = 0.00280452f;
constexpr float K_YB = 0.00028304f;

// Z_coeffs = (Matrix_Z / 1.08883) / 255.0
constexpr float K_ZR = 0.00006964f;
constexpr float K_ZG = 0.00042932f;
constexpr float K_ZB = 0.00342261f;

FORCE_INLINE void StorePackedLAB_Fast (float* RESTRICT dst, __m256 L, __m256 a, __m256 b)
{
    // Dump registers to stack (Aligned 32)
    CACHE_ALIGN float buf[24];
    _mm256_store_ps(&buf[0], L);
    _mm256_store_ps(&buf[8], a);
    _mm256_store_ps(&buf[16], b);

    // Unrolled assignment (Compiler optimizes this to vmovups sequences)
    // Pixel 0
    dst[0] = buf[0]; dst[1] = buf[8]; dst[2] = buf[16];
    // Pixel 1
    dst[3] = buf[1]; dst[4] = buf[9]; dst[5] = buf[17];
    // Pixel 2
    dst[6] = buf[2]; dst[7] = buf[10]; dst[8] = buf[18];
    // Pixel 3
    dst[9] = buf[3]; dst[10] = buf[11]; dst[11] = buf[19];
    // Pixel 4
    dst[12] = buf[4]; dst[13] = buf[12]; dst[14] = buf[20];
    // Pixel 5
    dst[15] = buf[5]; dst[16] = buf[13]; dst[17] = buf[21];
    // Pixel 6
    dst[18] = buf[6]; dst[19] = buf[14]; dst[20] = buf[22];
    // Pixel 7
    dst[21] = buf[7]; dst[22] = buf[15]; dst[23] = buf[23];
}

// -----------------------------------------------------------------------------------------
// MAIN KERNEL: BGRA_8u -> CIELab (3 Channel)
// -----------------------------------------------------------------------------------------
inline void ConvertToCIELab_BGRA_8u(
    const PF_Pixel_BGRA_8u* RESTRICT pRGB,
    fCIELabPix*             RESTRICT pLab,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           rgbPitch,
    const int32_t           labPitch
) noexcept
{
    // Constants (Scaled 1/255)
    const __m256 vXR = _mm256_set1_ps(K_XR), vXG = _mm256_set1_ps(K_XG), vXB = _mm256_set1_ps(K_XB);
    const __m256 vYR = _mm256_set1_ps(K_YR), vYG = _mm256_set1_ps(K_YG), vYB = _mm256_set1_ps(K_YB);
    const __m256 vZR = _mm256_set1_ps(K_ZR), vZG = _mm256_set1_ps(K_ZG), vZB = _mm256_set1_ps(K_ZB);

    // Lab
    const __m256 vLabEps = _mm256_set1_ps(0.008856f);
    const __m256 vKappa = _mm256_set1_ps(7.787037f);
    const __m256 vLabAdd = _mm256_set1_ps(16.0f / 116.0f);
    const __m256 v116 = _mm256_set1_ps(116.0f);
    const __m256 vM16 = _mm256_set1_ps(-16.0f);
    const __m256 v500 = _mm256_set1_ps(500.0f);
    const __m256 v200 = _mm256_set1_ps(200.0f);

    // Shuffle Masks
    const __m256i maskB = _mm256_setr_epi8(
        0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1,
        0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1);
    const __m256i maskG = _mm256_setr_epi8(
        1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1,
        1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1);
    const __m256i maskR = _mm256_setr_epi8(
        2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1,
        2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1);

    uint8_t* pRowSrc = (uint8_t*)pRGB;
    uint8_t* pRowDst = (uint8_t*)pLab;

    for (int y = 0; y < sizeY; ++y)
    {
        const __m256i* src = (const __m256i*)pRowSrc;
        float* dst = (float*)pRowDst; // Treat as float array for stride logic
        int x = 0;

        for (; x <= sizeX - 8; x += 8)
        {
            // 1. Load 32 bytes (8 pixels)
            __m256i raw = _mm256_loadu_si256(src);
            src++;

            // 2. Extract and Convert to Float
            __m256 B = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskB));
            __m256 G = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskG));
            __m256 R = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskR));

            // 3. Matrix (Scaled 1/255, No Gamma)
            __m256 X = _mm256_fmadd_ps(R, vXR, _mm256_fmadd_ps(G, vXG, _mm256_mul_ps(B, vXB)));
            __m256 Y = _mm256_fmadd_ps(R, vYR, _mm256_fmadd_ps(G, vYG, _mm256_mul_ps(B, vYB)));
            __m256 Z = _mm256_fmadd_ps(R, vZR, _mm256_fmadd_ps(G, vZG, _mm256_mul_ps(B, vZB)));

            // 4. Lab (Fast Cbrt)
            auto LabF = [&](const __m256 t) noexcept -> __m256
            {
                __m256 isLow = _mm256_cmp_ps(t, vLabEps, _CMP_LE_OQ);
                __m256 resLo = _mm256_fmadd_ps(t, vKappa, vLabAdd);
                __m256 resHi = FastCompute::AVX2::Cbrt(t);
                return _mm256_blendv_ps(resHi, resLo, isLow);
            };

            __m256 fX = LabF(X);
            __m256 fY = LabF(Y);
            __m256 fZ = LabF(Z);

            __m256 L = _mm256_fmadd_ps(v116, fY, vM16);
            __m256 a = _mm256_mul_ps(v500, _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(v200, _mm256_sub_ps(fY, fZ));

            // 5. Store 3-Channel Packed
            StorePackedLAB_Fast(dst + x * 3, L, a, b);
        }

        // Tail
        uint8_t* pTail = (uint8_t*)src;
        for (; x < sizeX; ++x)
        {
            float bS = (float)pTail[0];
            float gS = (float)pTail[1];
            float rS = (float)pTail[2];

            float X = K_XR * rS + K_XG * gS + K_XB * bS;
            float Y = K_YR * rS + K_YG * gS + K_YB * bS;
            float Z = K_ZR * rS + K_ZG * gS + K_ZB * bS;

            auto ToLab = [](const float t) noexcept
            {
                if (t > 0.008856f) return FastCompute::Cbrt(t);
                return (7.787037f * t + 0.137931f);
            };
            float fx = ToLab(X), fy = ToLab(Y), fz = ToLab(Z);

            dst[x * 3 + 0] = 116.0f * fy - 16.0f;
            dst[x * 3 + 1] = 500.0f * (fx - fy);
            dst[x * 3 + 2] = 200.0f * (fy - fz);

            pTail += 4;
        }
        pRowSrc += rgbPitch;
        pRowDst += labPitch;
    }
}

// -----------------------------------------------------------------------------------------
// MAIN KERNEL: ARGB_8u -> CIELab (3 Channel)
// -----------------------------------------------------------------------------------------
inline void ConvertToCIELab_ARGB_8u(
    const PF_Pixel_BGRA_8u* RESTRICT pRGB,
    fCIELabPix*             RESTRICT pLab,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           rgbPitch,
    const int32_t           labPitch
) noexcept
{
    // Constants (Same as above)
    const __m256 vXR = _mm256_set1_ps(K_XR), vXG = _mm256_set1_ps(K_XG), vXB = _mm256_set1_ps(K_XB);
    const __m256 vYR = _mm256_set1_ps(K_YR), vYG = _mm256_set1_ps(K_YG), vYB = _mm256_set1_ps(K_YB);
    const __m256 vZR = _mm256_set1_ps(K_ZR), vZG = _mm256_set1_ps(K_ZG), vZB = _mm256_set1_ps(K_ZB);

    // Lab
    const __m256 vLabEps = _mm256_set1_ps(0.008856f);
    const __m256 vKappa = _mm256_set1_ps(7.787037f); 
    const __m256 vLabAdd = _mm256_set1_ps(0.137931f);
    const __m256 v116 = _mm256_set1_ps(116.0f); 
    const __m256 vM16 = _mm256_set1_ps(-16.0f); 
    const __m256 v500 = _mm256_set1_ps(500.0f); 
    const __m256 v200 = _mm256_set1_ps(200.0f);

    // ARGB Masks (A=0, R=1, G=2, B=3)
    const __m256i maskR = _mm256_setr_epi8(
        1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1,
        1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1);
    const __m256i maskG = _mm256_setr_epi8(
        2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1,
        2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1);
    const __m256i maskB = _mm256_setr_epi8(
        3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1,
        3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1);

    uint8_t* pRowSrc = (uint8_t*)pRGB;
    uint8_t* pRowDst = (uint8_t*)pLab;

    for (int y = 0; y < sizeY; ++y)
    {
        const __m256i* src = (const __m256i*)pRowSrc;
        float* dst = (float*)pRowDst;
        int x = 0;

        for (; x <= sizeX - 8; x += 8)
        {
            __m256i raw = _mm256_loadu_si256(src);
            src++;

            // Shuffle
            __m256 B = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskB));
            __m256 G = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskG));
            __m256 R = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskR));

            // Matrix
            __m256 X = _mm256_fmadd_ps(R, vXR, _mm256_fmadd_ps(G, vXG, _mm256_mul_ps(B, vXB)));
            __m256 Y = _mm256_fmadd_ps(R, vYR, _mm256_fmadd_ps(G, vYG, _mm256_mul_ps(B, vYB)));
            __m256 Z = _mm256_fmadd_ps(R, vZR, _mm256_fmadd_ps(G, vZG, _mm256_mul_ps(B, vZB)));

            auto LabF = [&](const __m256 t) noexcept -> __m256
            {
                __m256 isLow = _mm256_cmp_ps(t, vLabEps, _CMP_LE_OQ);
                __m256 resLo = _mm256_fmadd_ps(t, vKappa, vLabAdd);
                __m256 resHi = FastCompute::AVX2::Cbrt(t);
                return _mm256_blendv_ps(resHi, resLo, isLow);
            };

            __m256 fX = LabF(X); __m256 fY = LabF(Y); __m256 fZ = LabF(Z);
            __m256 L = _mm256_fmadd_ps(v116, fY, vM16);
            __m256 a = _mm256_mul_ps(v500, _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(v200, _mm256_sub_ps(fY, fZ));

            StorePackedLAB_Fast(dst + x * 3, L, a, b);
        }

        uint8_t* pTail = (uint8_t*)src;
        for (; x < sizeX; ++x)
        {
            float rS = (float)pTail[1];
            float gS = (float)pTail[2];
            float bS = (float)pTail[3];

            float X = K_XR * rS + K_XG * gS + K_XB * bS;
            float Y = K_YR * rS + K_YG * gS + K_YB * bS;
            float Z = K_ZR * rS + K_ZG * gS + K_ZB * bS;

            auto ToLab = [](float t) noexcept
            {
                if (t > 0.008856f) return FastCompute::Cbrt(t);
                return (7.787037f * t + 0.137931f);
            };
            float fx = ToLab(X), fy = ToLab(Y), fz = ToLab(Z);

            dst[x * 3 + 0] = 116.0f * fy - 16.0f;
            dst[x * 3 + 1] = 500.0f * (fx - fy);
            dst[x * 3 + 2] = 200.0f * (fy - fz);

            pTail += 4;
        }
        pRowSrc += rgbPitch;
        pRowDst += labPitch;
    }
}