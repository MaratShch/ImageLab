#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "Common.hpp"
#include "Avx2ColorConverts.hpp"


#ifdef _WINDOWS
// Use template <int idx> to force compile-time constant for MSVC
template <int idx>
inline __m256 _mm256_gather_vuya_ps(__m256 v0, __m256 v1, __m256 v2, __m256 v3) noexcept
{
    // Now _MM_SHUFFLE receives a template constant, satisfying MSVC C2057
    constexpr int m = _MM_SHUFFLE(idx, idx, idx, idx);

    __m256 r01 = _mm256_shuffle_ps(v0, v1, m);
    __m256 r23 = _mm256_shuffle_ps(v2, v3, m);

    return _mm256_permutevar8x32_ps(
        _mm256_blend_ps(r01, r23, 0xF0),
        _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)
    );
}
#else
// --- Internal Shuffle Helper for 32f De-interleaving ---
// Grabs one component (by index 0..3) from 4 YMM registers (8 pixels total)
// Shuffle Helper: v0..v3 contain 8 VUYA pixels. idx: 0=V, 1=U, 2=Y, 3=A
inline __m256 _mm256_gather_vuya_ps(__m256 v0, __m256 v1, __m256 v2, __m256 v3, const int idx) noexcept
{
    const int m = _MM_SHUFFLE(idx, idx, idx, idx);
    __m256 r01 = _mm256_shuffle_ps(v0, v1, m);
    __m256 r23 = _mm256_shuffle_ps(v2, v3, m);
    return _mm256_permutevar8x32_ps(_mm256_blend_ps(r01, r23, 0xF0),
        _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

#endif // _WINDOWS


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
) noexcept
{
    const float kR = isBT709 ? 1.5748f : 1.4020f;
    const float kGu = isBT709 ? -0.1873f : -0.3441f;
    const float kGv = isBT709 ? -0.4681f : -0.7141f;
    const float kB = isBT709 ? 1.8556f : 1.7720f;
    const float mRX = isBT709 ? 0.4124564f : 0.430574f;
    const float mGX = isBT709 ? 0.3575761f : 0.341550f;
    const float mBX = isBT709 ? 0.1804375f : 0.178325f;
    const float mRY = isBT709 ? 0.2126729f : 0.222015f;
    const float mGY = isBT709 ? 0.7151522f : 0.706655f;
    const float mBY = isBT709 ? 0.0721750f : 0.071330f;
    const float mRZ = isBT709 ? 0.0193339f : 0.020183f;
    const float mGZ = isBT709 ? 0.1191920f : 0.129553f;
    const float mBZ = isBT709 ? 0.9503041f : 0.939180f;

    const __m256 vkR = _mm256_set1_ps(kR);   const __m256 vkGu = _mm256_set1_ps(kGu);
    const __m256 vkGv = _mm256_set1_ps(kGv); const __m256 vkB = _mm256_set1_ps(kB);
    const __m256 vmRX = _mm256_set1_ps(mRX); const __m256 vmGX = _mm256_set1_ps(mGX); const __m256 vmBX = _mm256_set1_ps(mBX);
    const __m256 vmRY = _mm256_set1_ps(mRY); const __m256 vmGY = _mm256_set1_ps(mGY); const __m256 vmBY = _mm256_set1_ps(mBY);
    const __m256 vmRZ = _mm256_set1_ps(mRZ); const __m256 vmGZ = _mm256_set1_ps(mGZ); const __m256 vmBZ = _mm256_set1_ps(mBZ);

    constexpr float inv255 = 1.0f / 255.0f;
    const __m256 vInv255 = _mm256_set1_ps(inv255);
    const __m256 v128 = _mm256_set1_ps(128.0f);
    const __m256i vMask8 = _mm256_set1_epi32(0xFF);

    const intptr_t strideSrc = (intptr_t)srcPitch * sizeof(PF_Pixel_VUYA_8u);
    const intptr_t strideL   = (intptr_t)labPitch * sizeof(float);
    const intptr_t strideAB  = (intptr_t)labPitch * 2 * sizeof(float);

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const PF_Pixel_VUYA_8u* rowSrc = reinterpret_cast<const PF_Pixel_VUYA_8u*>((const uint8_t*)pVUYA + y * strideSrc);
        float* rowL  = reinterpret_cast<float*>((uint8_t*)pL + y * strideL);
        float* rowAB = reinterpret_cast<float*>((uint8_t*)pAB + y * strideAB);

        int32_t x = 0;
        for (; x <= sizeX - 8; x += 8)
        {
            __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowSrc + x));

            __m256 V = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(raw, vMask8)), v128), vInv255);
            __m256 U = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(raw, 8), vMask8)), v128), vInv255);
            __m256 Y = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(raw, 16), vMask8)), vInv255);

            __m256 R = _mm256_fmadd_ps(V, vkR, Y);
            __m256 G = _mm256_add_ps(Y, _mm256_fmadd_ps(U, vkGu, _mm256_mul_ps(V, vkGv)));
            __m256 B = _mm256_fmadd_ps(U, vkB, Y);

            __m256 X = _mm256_fmadd_ps(R, vmRX, _mm256_fmadd_ps(G, vmGX, _mm256_mul_ps(B, vmBX)));
            __m256 XYZ_Y = _mm256_fmadd_ps(R, vmRY, _mm256_fmadd_ps(G, vmGY, _mm256_mul_ps(B, vmBY)));
            __m256 Z = _mm256_fmadd_ps(R, vmRZ, _mm256_fmadd_ps(G, vmGZ, _mm256_mul_ps(B, vmBZ)));

            __m256 fX = _mm256_lab_f_ps(_mm256_mul_ps(X, _mm256_set1_ps(c_Xn_inv)));
            __m256 fY = _mm256_lab_f_ps(XYZ_Y);
            __m256 fZ = _mm256_lab_f_ps(_mm256_mul_ps(Z, _mm256_set1_ps(c_Zn_inv)));

            _mm256_storeu_ps(rowL + x, _mm256_fmsub_ps(_mm256_set1_ps(116.0f), fY, _mm256_set1_ps(16.0f)));
            __m256 a = _mm256_mul_ps(_mm256_set1_ps(500.0f), _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(_mm256_set1_ps(200.0f), _mm256_sub_ps(fY, fZ));

            __m256 ab_lo = _mm256_unpacklo_ps(a, b);
            __m256 ab_hi = _mm256_unpackhi_ps(a, b);
            _mm256_storeu_ps(rowAB + (x * 2),     _mm256_permute2f128_ps(ab_lo, ab_hi, 0x20));
            _mm256_storeu_ps(rowAB + (x * 2) + 8, _mm256_permute2f128_ps(ab_lo, ab_hi, 0x31));
        }

        for (; x < sizeX; ++x)
        {
            float v = (rowSrc[x].V - 128.0f) * inv255;
            float u = (rowSrc[x].U - 128.0f) * inv255;
            float y_ch = (rowSrc[x].Y) * inv255;
            float r = y_ch + kR * v; float g = y_ch + kGu * u + kGv * v; float b = y_ch + kB * u;
            float X = r * mRX + g * mGX + b * mBX; float Y = r * mRY + g * mGY + b * mBY; float Z = r * mRZ + g * mGZ + b * mBZ;
            
            auto ft = [](float t) 
            { 
                return (t > c_delta_sq3) ? FastCompute::Pow(t, 1.0f/3.0f) : (c_lin_slope * t + c_lin_const);
            };
            
            float fx = ft(X * c_Xn_inv); float fy = ft(Y); float fz = ft(Z * c_Zn_inv);
            rowL[x] = 116.0f * fy - 16.0f;
            rowAB[x * 2] = 500.0f * (fx - fy); rowAB[x * 2 + 1] = 200.0f * (fy - fz);
        }
    }
    
    return;
}


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
) noexcept
{
    const float kR = isBT709 ? 1.5748f : 1.4020f;
    const float kGu = isBT709 ? -0.1873f : -0.3441f;
    const float kGv = isBT709 ? -0.4681f : -0.7141f;
    const float kB = isBT709 ? 1.8556f : 1.7720f;
    const float mRX = isBT709 ? 0.4124564f : 0.430574f;
    const float mGX = isBT709 ? 0.3575761f : 0.341550f;
    const float mBX = isBT709 ? 0.1804375f : 0.178325f;
    const float mRY = isBT709 ? 0.2126729f : 0.222015f;
    const float mGY = isBT709 ? 0.7151522f : 0.706655f;
    const float mBY = isBT709 ? 0.0721750f : 0.071330f;
    const float mRZ = isBT709 ? 0.0193339f : 0.020183f;
    const float mGZ = isBT709 ? 0.1191920f : 0.129553f;
    const float mBZ = isBT709 ? 0.9503041f : 0.939180f;

    const __m256 vkR = _mm256_set1_ps(kR); const __m256 vkGu = _mm256_set1_ps(kGu);
    const __m256 vkGv = _mm256_set1_ps(kGv); const __m256 vkB = _mm256_set1_ps(kB);
    const __m256 vmRX = _mm256_set1_ps(mRX); const __m256 vmGX = _mm256_set1_ps(mGX); const __m256 vmBX = _mm256_set1_ps(mBX);
    const __m256 vmRY = _mm256_set1_ps(mRY); const __m256 vmGY = _mm256_set1_ps(mGY); const __m256 vmBY = _mm256_set1_ps(mBY);
    const __m256 vmRZ = _mm256_set1_ps(mRZ); const __m256 vmGZ = _mm256_set1_ps(mGZ); const __m256 vmBZ = _mm256_set1_ps(mBZ);

    constexpr float inv32767 = 1.0f / 32767.0f;
    const __m256 vInv32767 = _mm256_set1_ps(inv32767);
    const __m256 v16384 = _mm256_set1_ps(16384.0f);

    // Masks to extract 16-bit channels to 32-bit float slots (in-lane)
    const __m256i maskV = _mm256_setr_epi8(0,1, -1,-1, 8,9, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 0,1, -1,-1, 8,9, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    const __m256i maskU = _mm256_setr_epi8(2,3, -1,-1, 10,11, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 2,3, -1,-1, 10,11, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    const __m256i maskY = _mm256_setr_epi8(4,5, -1,-1, 12,13, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 4,5, -1,-1, 12,13, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1);

    const intptr_t strideSrc = (intptr_t)srcPitch * sizeof(PF_Pixel_VUYA_16u);
    const intptr_t strideL   = (intptr_t)labPitch * sizeof(float);
    const intptr_t strideAB  = (intptr_t)labPitch * 2 * sizeof(float);

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const PF_Pixel_VUYA_16u* rowSrc = reinterpret_cast<const PF_Pixel_VUYA_16u*>((const uint8_t*)pVUYA + y * strideSrc);
        float* rowL  = reinterpret_cast<float*>((uint8_t*)pL + y * strideL);
        float* rowAB = reinterpret_cast<float*>((uint8_t*)pAB + y * strideAB);

        int32_t x = 0;
        for (; x <= sizeX - 4; x += 4)
        {
            __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowSrc + x));

            auto get_f4 = [&](__m256i mask) noexcept
            {
                __m256i shuf = _mm256_shuffle_epi8(raw, mask);
                // Extract lanes as __m128i correctly
                __m128i lo = _mm256_castsi256_si128(shuf);
                __m128i hi = _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(shuf), 1));
                // Add lanes to get 4 pixels into one 128-bit block
                __m128 f4 = _mm_cvtepi32_ps(_mm_add_epi32(lo, hi));
                return _mm256_insertf128_ps(_mm256_castps128_ps256(f4), _mm_setzero_ps(), 1);
            };

            __m256 V = _mm256_mul_ps(_mm256_sub_ps(get_f4(maskV), v16384), vInv32767);
            __m256 U = _mm256_mul_ps(_mm256_sub_ps(get_f4(maskU), v16384), vInv32767);
            __m256 Y = _mm256_mul_ps(get_f4(maskY), vInv32767);

            __m256 R = _mm256_fmadd_ps(V, vkR, Y);
            __m256 G = _mm256_add_ps(Y, _mm256_fmadd_ps(U, vkGu, _mm256_mul_ps(V, vkGv)));
            __m256 B = _mm256_fmadd_ps(U, vkB, Y);

            __m256 X = _mm256_fmadd_ps(R, vmRX, _mm256_fmadd_ps(G, vmGX, _mm256_mul_ps(B, vmBX)));
            __m256 XYZ_Y = _mm256_fmadd_ps(R, vmRY, _mm256_fmadd_ps(G, vmGY, _mm256_mul_ps(B, vmBY)));
            __m256 Z = _mm256_fmadd_ps(R, vmRZ, _mm256_fmadd_ps(G, vmGZ, _mm256_mul_ps(B, vmBZ)));

            __m256 fX = _mm256_lab_f_ps(_mm256_mul_ps(X, _mm256_set1_ps(c_Xn_inv)));
            __m256 fY = _mm256_lab_f_ps(XYZ_Y);
            __m256 fZ = _mm256_lab_f_ps(_mm256_mul_ps(Z, _mm256_set1_ps(c_Zn_inv)));

            __m128 L4 = _mm256_castps256_ps128(_mm256_fmsub_ps(_mm256_set1_ps(116.0f), fY, _mm256_set1_ps(16.0f)));
            _mm_storeu_ps(rowL + x, L4);

            __m128 a4 = _mm256_castps256_ps128(_mm256_mul_ps(_mm256_set1_ps(500.0f), _mm256_sub_ps(fX, fY)));
            __m128 b4 = _mm256_castps256_ps128(_mm256_mul_ps(_mm256_set1_ps(200.0f), _mm256_sub_ps(fY, fZ)));
            _mm_storeu_ps(rowAB + (x * 2),     _mm_unpacklo_ps(a4, b4));
            _mm_storeu_ps(rowAB + (x * 2) + 4, _mm_unpackhi_ps(a4, b4));
        }

        for (; x < sizeX; ++x)
        {
            float v = (rowSrc[x].V - 16384.0f) * inv32767;
            float u = (rowSrc[x].U - 16384.0f) * inv32767;
            float y_ch = (rowSrc[x].Y) * inv32767;
            float r = y_ch + kR * v; float g = y_ch + kGu * u + kGv * v; float b = y_ch + kB * u;
            float X = r * mRX + g * mGX + b * mBX; float Y = r * mRY + g * mGY + b * mBY; float Z = r * mRZ + g * mGZ + b * mBZ;
            
            auto ft = [](float t)
            {
                return (t > c_delta_sq3) ? FastCompute::Pow(t, 1.0f/3.0f) : (c_lin_slope * t + c_lin_const);
            };
            
            float fx = ft(X * c_Xn_inv); float fy = ft(Y); float fz = ft(Z * c_Zn_inv);
            rowL[x] = 116.0f * fy - 16.0f;
            rowAB[x * 2] = 500.0f * (fx - fy); rowAB[x * 2 + 1] = 200.0f * (fy - fz);
        }
    }
    
    return;
}


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
) noexcept
{
    // 1. Resolve Coefficients (Branch once outside the loop)
    // YUV -> RGB (Inverse Luma)
    const float kR   = isBT709 ? 1.5748f   : 1.4020f;
    const float kGu  = isBT709 ? -0.1873f  : -0.3441f;
    const float kGv  = isBT709 ? -0.4681f  : -0.7141f;
    const float kB   = isBT709 ? 1.8556f   : 1.7720f;

    // RGB -> XYZ (D65 Primaries)
    const float mRX  = isBT709 ? 0.4124564f : 0.430574f;
    const float mGX  = isBT709 ? 0.3575761f : 0.341550f;
    const float mBX  = isBT709 ? 0.1804375f : 0.178325f;
    const float mRY  = isBT709 ? 0.2126729f : 0.222015f;
    const float mGY  = isBT709 ? 0.7151522f : 0.706655f;
    const float mBY  = isBT709 ? 0.0721750f : 0.071330f;
    const float mRZ  = isBT709 ? 0.0193339f : 0.020183f;
    const float mGZ  = isBT709 ? 0.1191920f : 0.129553f;
    const float mBZ  = isBT709 ? 0.9503041f : 0.939180f;

    // 2. Broadcast to AVX2 Registers
    const __m256 vkR = _mm256_set1_ps(kR);  const __m256 vkGu = _mm256_set1_ps(kGu);
    const __m256 vkGv = _mm256_set1_ps(kGv); const __m256 vkB = _mm256_set1_ps(kB);

    const __m256 vmRX = _mm256_set1_ps(mRX); const __m256 vmGX = _mm256_set1_ps(mGX); const __m256 vmBX = _mm256_set1_ps(mBX);
    const __m256 vmRY = _mm256_set1_ps(mRY); const __m256 vmGY = _mm256_set1_ps(mGY); const __m256 vmBY = _mm256_set1_ps(mBY);
    const __m256 vmRZ = _mm256_set1_ps(mRZ); const __m256 vmGZ = _mm256_set1_ps(mGZ); const __m256 vmBZ = _mm256_set1_ps(mBZ);

    const __m256 v05 = _mm256_set1_ps(0.5f);

    const intptr_t strideSrc = (intptr_t)srcPitch * sizeof(PF_Pixel_VUYA_32f);
    const intptr_t strideL   = (intptr_t)labPitch * sizeof(float);
    const intptr_t strideAB  = (intptr_t)labPitch * 2 * sizeof(float);

    for (int32_t y = 0; y < sizeY; ++y)
	{
        const PF_Pixel_VUYA_32f* rowSrc = reinterpret_cast<const PF_Pixel_VUYA_32f*>((const uint8_t*)pVUYA + y * strideSrc);
        float* rowL  = reinterpret_cast<float*>((uint8_t*)pL + y * strideL);
        float* rowAB = reinterpret_cast<float*>((uint8_t*)pAB + y * strideAB);

        int32_t x = 0;
        // Main Loop: Process 8 pixels per iteration
        for (; x <= sizeX - 8; x += 8)
		{
            __m256 v0 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 0));
            __m256 v1 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 2));
            __m256 v2 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 4));
            __m256 v3 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 6));

#ifdef _WINDOWS
            // V = Index 0
            __m256 V = _mm256_sub_ps(_mm256_gather_vuya_ps<0>(v0, v1, v2, v3), v05);
            // U = Index 1
            __m256 U = _mm256_sub_ps(_mm256_gather_vuya_ps<1>(v0, v1, v2, v3), v05);
            // Y = Index 2
            __m256 Y = _mm256_gather_vuya_ps<2>(v0, v1, v2, v3);
#else
            __m256 V = _mm256_sub_ps(_mm256_gather_vuya_ps(v0, v1, v2, v3, 0), v05);
            __m256 U = _mm256_sub_ps(_mm256_gather_vuya_ps(v0, v1, v2, v3, 1), v05);
            __m256 Y = _mm256_gather_vuya_ps(v0, v1, v2, v3, 2);
#endif

            // YUV -> RGB
            __m256 R = _mm256_fmadd_ps(V, vkR, Y);
            __m256 G = _mm256_add_ps(Y, _mm256_fmadd_ps(U, vkGu, _mm256_mul_ps(V, vkGv)));
            __m256 B = _mm256_fmadd_ps(U, vkB, Y);

            // RGB -> XYZ
            __m256 X = _mm256_fmadd_ps(R, vmRX, _mm256_fmadd_ps(G, vmGX, _mm256_mul_ps(B, vmBX)));
            __m256 XYZ_Y = _mm256_fmadd_ps(R, vmRY, _mm256_fmadd_ps(G, vmGY, _mm256_mul_ps(B, vmBY)));
            __m256 Z = _mm256_fmadd_ps(R, vmRZ, _mm256_fmadd_ps(G, vmGZ, _mm256_mul_ps(B, vmBZ)));

            // XYZ -> Lab f(t)
            __m256 fX = _mm256_lab_f_ps(_mm256_mul_ps(X, _mm256_set1_ps(c_Xn_inv)));
            __m256 fY = _mm256_lab_f_ps(XYZ_Y);
            __m256 fZ = _mm256_lab_f_ps(_mm256_mul_ps(Z, _mm256_set1_ps(c_Zn_inv)));

            // L Planar
            _mm256_storeu_ps(rowL + x, _mm256_fmsub_ps(_mm256_set1_ps(116.0f), fY, _mm256_set1_ps(16.0f)));

            // ab Interleaved
            __m256 a = _mm256_mul_ps(_mm256_set1_ps(500.0f), _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(_mm256_set1_ps(200.0f), _mm256_sub_ps(fY, fZ));

            __m256 ab_lo = _mm256_unpacklo_ps(a, b);
            __m256 ab_hi = _mm256_unpackhi_ps(a, b);
            _mm256_storeu_ps(rowAB + (x * 2),     _mm256_permute2f128_ps(ab_lo, ab_hi, 0x20));
            _mm256_storeu_ps(rowAB + (x * 2) + 8, _mm256_permute2f128_ps(ab_lo, ab_hi, 0x31));
        }

        // Tail Logic
        for (; x < sizeX; ++x)
		{
            float v = rowSrc[x].V - 0.5f;
            float u = rowSrc[x].U - 0.5f;
            float y_ch = rowSrc[x].Y;

            float r = y_ch + kR * v;
            float g = y_ch + kGu * u + kGv * v;
            float b = y_ch + kB * u;

            float X = r * mRX + g * mGX + b * mBX;
            float Y = r * mRY + g * mGY + b * mBY;
            float Z = r * mRZ + g * mGZ + b * mBZ;

            auto f = [](float t) noexcept
			{ 
                return (t > c_delta_sq3) ? FastCompute::Pow(t, 1.0f/3.0f) : (c_lin_slope * t + c_lin_const); 
            };

            float fx = f(X * c_Xn_inv); float fy = f(Y); float fz = f(Z * c_Zn_inv);
            
            rowL[x] = 116.0f * fy - 16.0f;
            rowAB[x * 2]     = 500.0f * (fx - fy);
            rowAB[x * 2 + 1] = 200.0f * (fy - fz);
        }
    }
	
	return;
}