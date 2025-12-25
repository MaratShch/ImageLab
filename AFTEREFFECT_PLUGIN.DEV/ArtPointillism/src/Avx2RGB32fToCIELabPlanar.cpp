#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "FastAriphmetics.hpp"

// -----------------------------------------------------------------------------------------
// CONSTANTS (D65 / sRGB)
// -----------------------------------------------------------------------------------------
// Constants for 32-bit Float Input (Standard)
static const float K_XR = 0.433953f; static const float K_XG = 0.376219f; static const float K_XB = 0.189828f;
static const float K_YR = 0.212673f; static const float K_YG = 0.715152f; static const float K_YB = 0.072175f;
static const float K_ZR = 0.017757f; static const float K_ZG = 0.109477f; static const float K_ZB = 0.872766f;


// -----------------------------------------------------------------------------------------
// FAST MATH KERNELS (Nuclear: No Gamma, Fast Cbrt)
// -----------------------------------------------------------------------------------------

    // Fast Cbrt (Bit Hack + 2 Newton Steps + Reciprocal)
    FORCE_INLINE __m256 FastCbrt_AVX(const __m256 x) noexcept
    {
        __m256i i = _mm256_castps_si256(x);
        __m256i t = _mm256_add_epi32(_mm256_srli_epi32(i, 2), _mm256_srli_epi32(i, 4));
        t = _mm256_add_epi32(t, _mm256_srli_epi32(t, 4));
        t = _mm256_add_epi32(t, _mm256_srli_epi32(t, 8));
        t = _mm256_add_epi32(t, _mm256_srli_epi32(t, 16));
        t = _mm256_add_epi32(t, _mm256_set1_epi32(0x2a5137a0));
        
        __m256 y = _mm256_castsi256_ps(t);
        __m256 two = _mm256_set1_ps(2.0f);
        __m256 third = _mm256_set1_ps(0.33333333f);

        __m256 ryy = _mm256_rcp_ps(_mm256_mul_ps(y, y));
        y = _mm256_mul_ps(third, _mm256_fmadd_ps(two, y, _mm256_mul_ps(x, ryy)));
        
        ryy = _mm256_rcp_ps(_mm256_mul_ps(y, y));
        y = _mm256_mul_ps(third, _mm256_fmadd_ps(two, y, _mm256_mul_ps(x, ryy)));
        return y;
    }



    inline float FastCbrt(const float x) noexcept
    {
        union { int ix; float f; } u;
        u.f = x;
        u.ix = (u.ix / 3) + 0x2a5137a0;
        float y = u.f;
        y = (2.0f * y + x / (y * y)) * 0.33333333f;
        return y;
    }

// -----------------------------------------------------------------------------------------
// HELPER: Store Interleaved AB (a, b, a, b...)
// -----------------------------------------------------------------------------------------
FORCE_INLINE void StoreInterleavedAB(float* RESTRICT dst, __m256 a, __m256 b)
{
    // Input:
    // a: [a0 a1 a2 a3 | a4 a5 a6 a7]
    // b: [b0 b1 b2 b3 | b4 b5 b6 b7]

    // 1. Interleave
    // t0 = [a0 b0 a1 b1 | a4 b4 a5 b5]
    __m256 t0 = _mm256_unpacklo_ps(a, b);
    // t1 = [a2 b2 a3 b3 | a6 b6 a7 b7]
    __m256 t1 = _mm256_unpackhi_ps(a, b);

    // 2. Permute 128-bit lanes to linear order
    // out0 = [a0 b0 a1 b1 | a2 b2 a3 b3] (Pixels 0-3)
    __m256 out0 = _mm256_permute2f128_ps(t0, t1, 0x20);
    // out1 = [a4 b4 a5 b5 | a6 b6 a7 b7] (Pixels 4-7)
    __m256 out1 = _mm256_permute2f128_ps(t0, t1, 0x31);

    // 3. Store (16 floats total, 64 bytes)
    _mm256_storeu_ps(dst, out0);
    _mm256_storeu_ps(dst + 8, out1);
}


// -----------------------------------------------------------------------------------------
// 1. ConvertToCIELab_BGRA_32f (Semi-Planar Output)
// -----------------------------------------------------------------------------------------
void ConvertToCIELab_BGRA_32f
(
    const PF_Pixel_BGRA_32f*   RESTRICT pRGB,
    float*              RESTRICT pL,
    float*              RESTRICT pAB,
    const int32_t sizeX,
    const int32_t sizeY,
    const int32_t rgbPitch,
    const int32_t labPitch
) noexcept
{
    // Constants (Standard Float)
    const __m256 vXR = _mm256_set1_ps(K_XR), vXG = _mm256_set1_ps(K_XG), vXB = _mm256_set1_ps(K_XB);
    const __m256 vYR = _mm256_set1_ps(K_YR), vYG = _mm256_set1_ps(K_YG), vYB = _mm256_set1_ps(K_YB);
    const __m256 vZR = _mm256_set1_ps(K_ZR), vZG = _mm256_set1_ps(K_ZG), vZB = _mm256_set1_ps(K_ZB);
    
    const __m256 vLabEps = _mm256_set1_ps(0.008856f); 
    const __m256 vKappa = _mm256_set1_ps(7.787037f); 
    const __m256 vLabAdd = _mm256_set1_ps(0.137931f);
    const __m256 v116 = _mm256_set1_ps(116.0f); 
    const __m256 vM16 = _mm256_set1_ps(-16.0f); 
    const __m256 v500 = _mm256_set1_ps(500.0f); 
    const __m256 v200 = _mm256_set1_ps(200.0f);
    const __m256 vOne = _mm256_set1_ps(1.0f); 
    const __m256 vZero = _mm256_setzero_ps();

    uint8_t* pRowSrc = (uint8_t*)pRGB;
    uint8_t* pRowL   = (uint8_t*)pL;
    uint8_t* pRowAB  = (uint8_t*)pAB;

    for(int y = 0; y < sizeY; ++y)
    {
        const float* src = (const float*)pRowSrc;
        float* dstL = (float*)pRowL;
        float* dstAB = (float*)pRowAB;
        int x = 0;

        for(; x <= sizeX - 8; x += 8)
        {
            __m256 v0 = _mm256_loadu_ps(src + 0);  
            __m256 v1 = _mm256_loadu_ps(src + 8);
            __m256 v2 = _mm256_loadu_ps(src + 16);
            __m256 v3 = _mm256_loadu_ps(src + 24);

            __m256 t0 = _mm256_unpacklo_ps(v0, v1);
            __m256 t1 = _mm256_unpackhi_ps(v0, v1);
            __m256 t2 = _mm256_unpacklo_ps(v2, v3);
            __m256 t3 = _mm256_unpackhi_ps(v2, v3);
            
            __m256 r0 = _mm256_permute2f128_ps(t0, t2, 0x20); 
            __m256 r2 = _mm256_permute2f128_ps(t0, t2, 0x31);
            __m256 r1 = _mm256_permute2f128_ps(t1, t3, 0x20); 
            __m256 r3 = _mm256_permute2f128_ps(t1, t3, 0x31);

            __m256 B = _mm256_unpacklo_ps(r0, r2);
            __m256 G = _mm256_unpackhi_ps(r0, r2);
            __m256 R = _mm256_unpacklo_ps(r1, r3);

            B = _mm256_min_ps(_mm256_max_ps(B, vZero), vOne);
            G = _mm256_min_ps(_mm256_max_ps(G, vZero), vOne);
            R = _mm256_min_ps(_mm256_max_ps(R, vZero), vOne);

            __m256 X = _mm256_fmadd_ps(R, vXR, _mm256_fmadd_ps(G, vXG, _mm256_mul_ps(B, vXB)));
            __m256 Y = _mm256_fmadd_ps(R, vYR, _mm256_fmadd_ps(G, vYG, _mm256_mul_ps(B, vYB)));
            __m256 Z = _mm256_fmadd_ps(R, vZR, _mm256_fmadd_ps(G, vZG, _mm256_mul_ps(B, vZB)));

            auto LabF = [&](const __m256 t) noexcept -> __m256
            {
                __m256 isLow = _mm256_cmp_ps(t, vLabEps, _CMP_LE_OQ);
                __m256 resLo = _mm256_fmadd_ps(t, vKappa, vLabAdd);
                __m256 resHi = FastCbrt_AVX(t);
                return _mm256_blendv_ps(resHi, resLo, isLow);
            };

            __m256 fX = LabF(X); __m256 fY = LabF(Y); __m256 fZ = LabF(Z);
            __m256 L = _mm256_fmadd_ps(v116, fY, vM16);
            __m256 a = _mm256_mul_ps(v500, _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(v200, _mm256_sub_ps(fY, fZ));

            _mm256_storeu_ps(dstL + x, L);
            StoreInterleavedAB(dstAB + (x * 2), a, b);

            src += 32; 
        }

        for(; x < sizeX; ++x)
        {
            float bS = src[0], gS = src[1], rS = src[2];
            if(bS<0.f) bS=0.f; if(bS>1.f) bS=1.f;
            if(gS<0.f) gS=0.f; if(gS>1.f) gS=1.f;
            if(rS<0.f) rS=0.f; if(rS>1.f) rS=1.f;

            float X = K_XR * rS + K_XG * gS + K_XB * bS;
            float Y = K_YR * rS + K_YG * gS + K_YB * bS;
            float Z = K_ZR * rS + K_ZG * gS + K_ZB * bS;

            auto ToLab = [](const float t) noexcept
            {
                if (t > 0.008856f) return FastCompute::Cbrt(t);
                return (7.787037f * t + 0.137931f);
            };
            float fx = ToLab(X), fy = ToLab(Y), fz = ToLab(Z);

            dstL[x] = 116.0f * fy - 16.0f;
            dstAB[x*2 + 0] = 500.0f * (fx - fy);
            dstAB[x*2 + 1] = 200.0f * (fy - fz);

            src += 4;
        }
        pRowSrc += rgbPitch * sizeof(PF_Pixel_BGRA_32f);
        pRowL   += labPitch * sizeof(float);
        pRowAB  += labPitch * sizeof(float) * 2; // Assuming packed pitch logic
    }
}

// -----------------------------------------------------------------------------------------
// 4. ConvertToCIELab_ARGB_32f (Semi-Planar Output)
// -----------------------------------------------------------------------------------------
void ConvertToCIELab_ARGB_32f
(
    const PF_Pixel_ARGB_32f* RESTRICT pRGB,
    float* RESTRICT pL,
    float* RESTRICT pAB,
    int32_t sizeX,
    int32_t sizeY,
    int32_t rgbPitch,
    int32_t labPitch
) noexcept
{
    // Constants (Same as BGRA_32f)
    const __m256 vXR = _mm256_set1_ps(K_XR), vXG = _mm256_set1_ps(K_XG), vXB = _mm256_set1_ps(K_XB);
    const __m256 vYR = _mm256_set1_ps(K_YR), vYG = _mm256_set1_ps(K_YG), vYB = _mm256_set1_ps(K_YB);
    const __m256 vZR = _mm256_set1_ps(K_ZR), vZG = _mm256_set1_ps(K_ZG), vZB = _mm256_set1_ps(K_ZB);
    
    const __m256 vLabEps = _mm256_set1_ps(0.008856f); 
    const __m256 vKappa = _mm256_set1_ps(7.787037f); 
    const __m256 vLabAdd = _mm256_set1_ps(0.137931f);
    const __m256 v116 = _mm256_set1_ps(116.0f); 
    const __m256 vM16 = _mm256_set1_ps(-16.0f); 
    const __m256 v500 = _mm256_set1_ps(500.0f); 
    const __m256 v200 = _mm256_set1_ps(200.0f);
    const __m256 vOne = _mm256_set1_ps(1.0f); 
    const __m256 vZero = _mm256_setzero_ps();

    uint8_t* pRowSrc = (uint8_t*)pRGB;
    uint8_t* pRowL   = (uint8_t*)pL;
    uint8_t* pRowAB  = (uint8_t*)pAB;

    for(int y = 0; y < sizeY; ++y)
    {
        const float* src = (const float*)pRowSrc;
        float* dstL = (float*)pRowL;
        float* dstAB = (float*)pRowAB;
        int x = 0;

        for(; x <= sizeX - 8; x += 8)
        {
            __m256 v0 = _mm256_loadu_ps(src + 0);
            __m256 v1 = _mm256_loadu_ps(src + 8);
            __m256 v2 = _mm256_loadu_ps(src + 16);
            __m256 v3 = _mm256_loadu_ps(src + 24);

            __m256 t0 = _mm256_unpacklo_ps(v0, v1);
            __m256 t1 = _mm256_unpackhi_ps(v0, v1);
            __m256 t2 = _mm256_unpacklo_ps(v2, v3);
            __m256 t3 = _mm256_unpackhi_ps(v2, v3);
            
            __m256 r0 = _mm256_permute2f128_ps(t0, t2, 0x20); 
            __m256 r2 = _mm256_permute2f128_ps(t0, t2, 0x31);
            __m256 r1 = _mm256_permute2f128_ps(t1, t3, 0x20); 
            __m256 r3 = _mm256_permute2f128_ps(t1, t3, 0x31);

            __m256 R = _mm256_unpackhi_ps(r0, r2); 
            __m256 G = _mm256_unpacklo_ps(r1, r3); 
            __m256 B = _mm256_unpackhi_ps(r1, r3);

            B = _mm256_min_ps(_mm256_max_ps(B, vZero), vOne);
            G = _mm256_min_ps(_mm256_max_ps(G, vZero), vOne);
            R = _mm256_min_ps(_mm256_max_ps(R, vZero), vOne);

            __m256 X = _mm256_fmadd_ps(R, vXR, _mm256_fmadd_ps(G, vXG, _mm256_mul_ps(B, vXB)));
            __m256 Y = _mm256_fmadd_ps(R, vYR, _mm256_fmadd_ps(G, vYG, _mm256_mul_ps(B, vYB)));
            __m256 Z = _mm256_fmadd_ps(R, vZR, _mm256_fmadd_ps(G, vZG, _mm256_mul_ps(B, vZB)));

            auto LabF = [&](const __m256 t) noexcept -> __m256
            {
                __m256 isLow = _mm256_cmp_ps(t, vLabEps, _CMP_LE_OQ);
                __m256 resLo = _mm256_fmadd_ps(t, vKappa, vLabAdd);
                __m256 resHi = FastCbrt_AVX(t);
                return _mm256_blendv_ps(resHi, resLo, isLow);
            };

            __m256 fX = LabF(X); __m256 fY = LabF(Y); __m256 fZ = LabF(Z);
            __m256 L = _mm256_fmadd_ps(v116, fY, vM16);
            __m256 a = _mm256_mul_ps(v500, _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(v200, _mm256_sub_ps(fY, fZ));

            _mm256_storeu_ps(dstL + x, L);
            StoreInterleavedAB(dstAB + (x * 2), a, b);

            src += 32; 
        }

        for(; x < sizeX; ++x)
        {
            float rS = src[1], gS = src[2], bS = src[3];
            
            if(bS<0.f) bS=0.f; if(bS>1.f) bS=1.f;
            if(gS<0.f) gS=0.f; if(gS>1.f) gS=1.f;
            if(rS<0.f) rS=0.f; if(rS>1.f) rS=1.f;

            float X = K_XR * rS + K_XG * gS + K_XB * bS;
            float Y = K_YR * rS + K_YG * gS + K_YB * bS;
            float Z = K_ZR * rS + K_ZG * gS + K_ZB * bS;

            auto ToLab = [](const float t) noexcept
            {
                if (t > 0.008856f) return FastCompute::Cbrt(t);
                return (7.787037f * t + 0.137931f);
            };
            float fx = ToLab(X), fy = ToLab(Y), fz = ToLab(Z);

            dstL[x] = 116.0f * fy - 16.0f;
            dstAB[x*2 + 0] = 500.0f * (fx - fy);
            dstAB[x*2 + 1] = 200.0f * (fy - fz);

            src += 4;
        }
        pRowSrc += rgbPitch * sizeof(PF_Pixel_ARGB_32f);
        pRowL   += labPitch * sizeof(float);
        pRowAB  += labPitch * sizeof(float) * 2; // Assuming packed pitch logic
    }
}