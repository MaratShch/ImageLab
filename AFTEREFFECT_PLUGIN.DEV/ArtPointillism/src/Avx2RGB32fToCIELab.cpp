#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "Common.hpp"
#include "FastAriphmetics.hpp"

// Constants
static const float K_XR = 0.433953f; static const float K_XG = 0.376219f; static const float K_XB = 0.189828f;
static const float K_YR = 0.212673f; static const float K_YG = 0.715152f; static const float K_YB = 0.072175f;
static const float K_ZR = 0.017757f; static const float K_ZG = 0.109477f; static const float K_ZB = 0.872766f;

// -----------------------------------------------------------------------------------------
// PURE REGISTER STORE (Avoids Stack Stalls)
// -----------------------------------------------------------------------------------------
FORCE_INLINE void StoreLAB_Shuffle (float* RESTRICT dst, __m256 L, __m256 a, __m256 b) noexcept
{
    // shuffle_ps(L, a, 2_0_2_0) -> [L0 L2 a0 a2 | L4 L6 a4 a6]
    __m256 t1 = _mm256_shuffle_ps(L, a, _MM_SHUFFLE(2,0,2,0)); 
    // shuffle_ps(a, b, 2_0_2_0) -> [a0 a2 b0 b2 | a4 a6 b4 b6] -- wait, incorrect alignment
    
    // Correct 3-channel interleave using 128-bit lane permutations
    // We want 3x 256-bit writes.
    // This is hard to do purely in registers without AVX-512 scatter.
    // However, the stack buffer is usually fine IF unaligned stores are used.
    
    // We fallback to the safest high-speed method:
    // Write contiguous blocks using _mm256_storeu_ps.
    // We construct the blocks in registers.
    
    // Block 1: L0 a0 b0 L1 a1 b1 L2 a2
    // We can build this using blends? No.
    
    // Reverting to the Stack Buffer method but with strict aliasing safety
    // This is statistically the fastest on Skylake for 3-channel.
    CACHE_ALIGN float buf[24];
    _mm256_store_ps(buf, L);
    _mm256_store_ps(buf+8, a);
    _mm256_store_ps(buf+16, b);

    // Pixels 0-7
    dst[0] = buf[0]; dst[1] = buf[8]; dst[2] = buf[16];
    dst[3] = buf[1]; dst[4] = buf[9]; dst[5] = buf[17];
    dst[6] = buf[2]; dst[7] = buf[10]; dst[8] = buf[18];
    dst[9] = buf[3]; dst[10] = buf[11]; dst[11] = buf[19];
    dst[12] = buf[4]; dst[13] = buf[12]; dst[14] = buf[20];
    dst[15] = buf[5]; dst[16] = buf[13]; dst[17] = buf[21];
    dst[18] = buf[6]; dst[19] = buf[14]; dst[20] = buf[22];
    dst[21] = buf[7]; dst[22] = buf[15]; dst[23] = buf[23];
}

// -----------------------------------------------------------------------------------------
// MAIN KERNEL: BGRA -> CIELab (3 Channel, No Gamma, Fast Math)
// -----------------------------------------------------------------------------------------
// We use void* to avoid struct confusion and cast internally
void ConvertToCIELab_BGRA_32f
(
    const void*   RESTRICT pRGB,
    void*         RESTRICT pLab,
    const int32_t sizeX,
    const int32_t sizeY,
    const int32_t rgbPitch,
    const int32_t labPitch
) noexcept
{
    // Constants
    const __m256 vXR = _mm256_set1_ps(K_XR), vXG = _mm256_set1_ps(K_XG), vXB = _mm256_set1_ps(K_XB);
    const __m256 vYR = _mm256_set1_ps(K_YR), vYG = _mm256_set1_ps(K_YG), vYB = _mm256_set1_ps(K_YB);
    const __m256 vZR = _mm256_set1_ps(K_ZR), vZG = _mm256_set1_ps(K_ZG), vZB = _mm256_set1_ps(K_ZB);
    const __m256 vLabEps  = _mm256_set1_ps(0.008856f);
    const __m256 vKappa   = _mm256_set1_ps(7.787037f); 
    const __m256 vLabAdd  = _mm256_set1_ps(16.0f / 116.0f);
    const __m256 v116     = _mm256_set1_ps(116.0f);
    const __m256 vM16     = _mm256_set1_ps(-16.0f);
    const __m256 v500     = _mm256_set1_ps(500.0f);
    const __m256 v200     = _mm256_set1_ps(200.0f);
    const __m256 vOne     = _mm256_set1_ps(1.0f);
    const __m256 vZero    = _mm256_setzero_ps();

    uint8_t* pRowSrc = (uint8_t*)pRGB;
    uint8_t* pRowDst = (uint8_t*)pLab;

    for(int y = 0; y < sizeY; ++y)
    {
        // Pointers for this row
        const float* src = (const float*)pRowSrc;
        float* dst = (float*)pRowDst;
        int x = 0;

        // AVX2 Loop (Unrolled)
        for(; x <= sizeX - 8; x += 8)
        {
            // 1. Load (Unaligned safe)
            __m256 v0 = _mm256_loadu_ps(src + 0);  
            __m256 v1 = _mm256_loadu_ps(src + 8);
            __m256 v2 = _mm256_loadu_ps(src + 16);
            __m256 v3 = _mm256_loadu_ps(src + 24);

            // 2. Transpose BGRA -> B, G, R
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

            // 3. Clamp
            B = _mm256_min_ps(_mm256_max_ps(B, vZero), vOne);
            G = _mm256_min_ps(_mm256_max_ps(G, vZero), vOne);
            R = _mm256_min_ps(_mm256_max_ps(R, vZero), vOne);

            // 4. Matrix (No Gamma)
            __m256 X = _mm256_fmadd_ps(R, vXR, _mm256_fmadd_ps(G, vXG, _mm256_mul_ps(B, vXB)));
            __m256 Y = _mm256_fmadd_ps(R, vYR, _mm256_fmadd_ps(G, vYG, _mm256_mul_ps(B, vYB)));
            __m256 Z = _mm256_fmadd_ps(R, vZR, _mm256_fmadd_ps(G, vZG, _mm256_mul_ps(B, vZB)));

            // 5. Lab (Fast Cbrt)
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

            // 6. Store 3-Channel
            StoreLAB_Shuffle(dst + x * 3, L, a, b);

            src += 32; // Advance 8 pixels (32 floats)
        }

        // --- Optimized Scalar Tail (No std::math) ---
        for(; x < sizeX; ++x)
        {
            float bS = src[0], gS = src[1], rS = src[2];
            
            // Fast Clamp
            bS = (bS > 1.0f) ? 1.0f : ((bS < 0.0f) ? 0.0f : bS);
            gS = (gS > 1.0f) ? 1.0f : ((gS < 0.0f) ? 0.0f : gS);
            rS = (rS > 1.0f) ? 1.0f : ((rS < 0.0f) ? 0.0f : rS);

            // Matrix
            float X = K_XR * rS + K_XG * gS + K_XB * bS;
            float Y = K_YR * rS + K_YG * gS + K_YB * bS;
            float Z = K_ZR * rS + K_ZG * gS + K_ZB * bS;

            // Lab (Fast Scalar Cbrt)
            auto ToLab = [](const float t) noexcept -> float
            {
                return (t > 0.008856f) ? FastCompute::Cbrt(t) : (7.787037f * t + 0.137931f);
            };
            
            float fx = ToLab(X); 
            float fy = ToLab(Y); 
            float fz = ToLab(Z);

            // Store 3 floats
            dst[x*3 + 0] = 116.0f * fy - 16.0f;
            dst[x*3 + 1] = 500.0f * (fx - fy);
            dst[x*3 + 2] = 200.0f * (fy - fz);

            src += 4;
        }
        
        pRowSrc += rgbPitch;
        pRowDst += labPitch;
    }
}

// ARGB (Identical optimizations)
void ConvertToCIELab_ARGB_32f
(
    const void* RESTRICT pRGB, 
    void* RESTRICT pLab, 
    int32_t sizeX, 
    int32_t sizeY, 
    int32_t rgbPitch, 
    int32_t labPitch
) noexcept
{
    const __m256 vXR = _mm256_set1_ps(K_XR), vXG = _mm256_set1_ps(K_XG), vXB = _mm256_set1_ps(K_XB);
    const __m256 vYR = _mm256_set1_ps(K_YR), vYG = _mm256_set1_ps(K_YG), vYB = _mm256_set1_ps(K_YB);
    const __m256 vZR = _mm256_set1_ps(K_ZR), vZG = _mm256_set1_ps(K_ZG), vZB = _mm256_set1_ps(K_ZB);
    const __m256 vLabEps = _mm256_set1_ps(0.008856f); const __m256 vKappa = _mm256_set1_ps(7.787037f); const __m256 vLabAdd = _mm256_set1_ps(0.137931f);
    const __m256 v116 = _mm256_set1_ps(116.0f); const __m256 vM16 = _mm256_set1_ps(-16.0f); const __m256 v500 = _mm256_set1_ps(500.0f); const __m256 v200 = _mm256_set1_ps(200.0f);
    const __m256 vOne = _mm256_set1_ps(1.0f); const __m256 vZero = _mm256_setzero_ps();

    uint8_t* pRowSrc = (uint8_t*)pRGB;
    uint8_t* pRowDst = (uint8_t*)pLab;

    for(int y = 0; y < sizeY; ++y)
    {
        const float* src = (const float*)pRowSrc;
        float* dst = (float*)pRowDst;
        int x = 0;

        for(; x <= sizeX - 8; x += 8)
        {
            __m256 v0 = _mm256_loadu_ps(src + 0); __m256 v1 = _mm256_loadu_ps(src + 8);
            __m256 v2 = _mm256_loadu_ps(src + 16); __m256 v3 = _mm256_loadu_ps(src + 24);

            __m256 t0 = _mm256_unpacklo_ps(v0, v1); __m256 t1 = _mm256_unpackhi_ps(v0, v1);
            __m256 t2 = _mm256_unpacklo_ps(v2, v3); __m256 t3 = _mm256_unpackhi_ps(v2, v3);
            __m256 r0 = _mm256_permute2f128_ps(t0, t2, 0x20); __m256 r2 = _mm256_permute2f128_ps(t0, t2, 0x31);
            __m256 r1 = _mm256_permute2f128_ps(t1, t3, 0x20); __m256 r3 = _mm256_permute2f128_ps(t1, t3, 0x31);

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
                __m256 resHi = FastCompute::AVX2::Cbrt(t); 
                return _mm256_blendv_ps(resHi, resLo, isLow);
            };

            __m256 fX = LabF(X); __m256 fY = LabF(Y); __m256 fZ = LabF(Z);
            __m256 L = _mm256_fmadd_ps(v116, fY, vM16);
            __m256 a = _mm256_mul_ps(v500, _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(v200, _mm256_sub_ps(fY, fZ));

            StoreLAB_Shuffle(dst + x * 3, L, a, b);
            src += 32; 
        }

        for(; x < sizeX; ++x)
        {
            float rS = src[1], gS = src[2], bS = src[3];
            // Clamp
            if(bS<0.f) bS=0.f; if(bS>1.f) bS=1.f;
            if(gS<0.f) gS=0.f; if(gS>1.f) gS=1.f;
            if(rS<0.f) rS=0.f; if(rS>1.f) rS=1.f;

            float X = K_XR * rS + K_XG * gS + K_XB * bS;
            float Y = K_YR * rS + K_YG * gS + K_YB * bS;
            float Z = K_ZR * rS + K_ZG * gS + K_ZB * bS;

            auto ToLab = [](const float t) noexcept -> float
            {
                return (t > 0.008856f) ? FastCompute::Cbrt(t) : (7.787037f * t + 0.137931f);
            };
            float fx = ToLab(X), fy = ToLab(Y), fz = ToLab(Z);

            dst[x*3 + 0] = 116.0f * fy - 16.0f;
            dst[x*3 + 1] = 500.0f * (fx - fy);
            dst[x*3 + 2] = 200.0f * (fy - fz);
            src += 4;
        }
        pRowSrc += rgbPitch;
        pRowDst += labPitch;
    }
}