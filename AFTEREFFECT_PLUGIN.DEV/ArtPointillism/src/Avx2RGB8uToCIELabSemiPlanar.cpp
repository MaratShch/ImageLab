#include "Avx2ColorConverts.hpp"
#include "FastAriphmetics.hpp"


constexpr float c_inv255    = 1.0f / 255.0f;

// --- BGRA_8u Implementation ---
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_BGRA_8u* RESTRICT pRGB,
    float*                  RESTRICT pL,
    float*                  RESTRICT pAB,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           srcPitch,
    const int32_t           labPitch
) noexcept
{
    const intptr_t strideSrc = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_ARGB_8u);
    const intptr_t strideL   = static_cast<intptr_t>(labPitch) * sizeof(float);
    const intptr_t strideAB  = static_cast<intptr_t>(labPitch  * 2) * sizeof(float);

    // Matrix (Rec.709 D65)
    const __m256 mRX = _mm256_set1_ps(0.4124564f); const __m256 mGX = _mm256_set1_ps(0.3575761f); const __m256 mBX = _mm256_set1_ps(0.1804375f);
    const __m256 mRY = _mm256_set1_ps(0.2126729f); const __m256 mGY = _mm256_set1_ps(0.7151522f); const __m256 mBY = _mm256_set1_ps(0.0721750f);
    const __m256 mRZ = _mm256_set1_ps(0.0193339f); const __m256 mGZ = _mm256_set1_ps(0.1191920f); const __m256 mBZ = _mm256_set1_ps(0.9503041f);

    const __m256 vInv255 = _mm256_set1_ps(c_inv255);

    // BGRA: B=0, G=1, R=2, A=3
    const __m256i maskR = _mm256_setr_epi8(2, -1,-1,-1, 6, -1,-1,-1, 10, -1,-1,-1, 14, -1,-1,-1, 2, -1,-1,-1, 6, -1,-1,-1, 10, -1,-1,-1, 14, -1,-1,-1);
    const __m256i maskG = _mm256_setr_epi8(1, -1,-1,-1, 5, -1,-1,-1, 9,  -1,-1,-1, 13, -1,-1,-1, 1, -1,-1,-1, 5, -1,-1,-1, 9,  -1,-1,-1, 13, -1,-1,-1);
    const __m256i maskB = _mm256_setr_epi8(0, -1,-1,-1, 4, -1,-1,-1, 8,  -1,-1,-1, 12, -1,-1,-1, 0, -1,-1,-1, 4, -1,-1,-1, 8,  -1,-1,-1, 12, -1,-1,-1);

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const PF_Pixel_BGRA_8u* rowSrc = reinterpret_cast<const PF_Pixel_BGRA_8u*>((const uint8_t*)pRGB + y * strideSrc);
        float* rowL  = reinterpret_cast<float*>((uint8_t*)pL + y * strideL);
        float* rowAB = reinterpret_cast<float*>((uint8_t*)pAB + y * strideAB);

        int32_t x = 0;
        for (; x <= sizeX - 8; x += 8)
        {
            __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowSrc + x));

            __m256 R = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskR)), vInv255);
            __m256 G = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskG)), vInv255);
            __m256 B = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskB)), vInv255);

            __m256 X = _mm256_fmadd_ps(R, mRX, _mm256_fmadd_ps(G, mGX, _mm256_mul_ps(B, mBX)));
            __m256 Y = _mm256_fmadd_ps(R, mRY, _mm256_fmadd_ps(G, mGY, _mm256_mul_ps(B, mBY)));
            __m256 Z = _mm256_fmadd_ps(R, mRZ, _mm256_fmadd_ps(G, mGZ, _mm256_mul_ps(B, mBZ)));

            __m256 fX = _mm256_lab_f_ps(_mm256_mul_ps(X, _mm256_set1_ps(c_Xn_inv)));
            __m256 fY = _mm256_lab_f_ps(Y);
            __m256 fZ = _mm256_lab_f_ps(_mm256_mul_ps(Z, _mm256_set1_ps(c_Zn_inv)));

            _mm256_storeu_ps(rowL + x, _mm256_fmsub_ps(_mm256_set1_ps(116.0f), fY, _mm256_set1_ps(16.0f)));

            __m256 a = _mm256_mul_ps(_mm256_set1_ps(500.0f), _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(_mm256_set1_ps(200.0f), _mm256_sub_ps(fY, fZ));

            __m256 ab_lo = _mm256_unpacklo_ps(a, b);
            __m256 ab_hi = _mm256_unpackhi_ps(a, b);
            _mm256_storeu_ps(rowAB + (x * 2),     _mm256_permute2f128_ps(ab_lo, ab_hi, 0x20));
            _mm256_storeu_ps(rowAB + (x * 2) + 8, _mm256_permute2f128_ps(ab_lo, ab_hi, 0x31));
        }

        // Tail handling
        for (; x < sizeX; ++x)
        {
            float r = rowSrc[x].R * c_inv255;
            float g = rowSrc[x].G * c_inv255;
            float b = rowSrc[x].B * c_inv255;
            float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
            float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
            float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

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
}

// --- ARGB_8u Implementation ---
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_ARGB_8u* RESTRICT pRGB,
    float*                  RESTRICT pL,
    float*                  RESTRICT pAB,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           srcPitch,
    const int32_t           labPitch
) noexcept
{
    const intptr_t strideSrc = static_cast<intptr_t>(srcPitch) * sizeof(PF_Pixel_ARGB_8u);
    const intptr_t strideL   = static_cast<intptr_t>(labPitch) * sizeof(float);
    const intptr_t strideAB  = static_cast<intptr_t>(labPitch  * 2) * sizeof(float);

    // Matrix (Rec.709 D65)
    const __m256 mRX = _mm256_set1_ps(0.4124564f); const __m256 mGX = _mm256_set1_ps(0.3575761f); const __m256 mBX = _mm256_set1_ps(0.1804375f);
    const __m256 mRY = _mm256_set1_ps(0.2126729f); const __m256 mGY = _mm256_set1_ps(0.7151522f); const __m256 mBY = _mm256_set1_ps(0.0721750f);
    const __m256 mRZ = _mm256_set1_ps(0.0193339f); const __m256 mGZ = _mm256_set1_ps(0.1191920f); const __m256 mBZ = _mm256_set1_ps(0.9503041f);

    const __m256 vInv255 = _mm256_set1_ps(c_inv255);

    // ARGB: A=0, R=1, G=2, B=3
    const __m256i maskR = _mm256_setr_epi8(1, -1,-1,-1, 5, -1,-1,-1, 9,  -1,-1,-1, 13, -1,-1,-1, 1, -1,-1,-1, 5, -1,-1,-1, 9,  -1,-1,-1, 13, -1,-1,-1);
    const __m256i maskG = _mm256_setr_epi8(2, -1,-1,-1, 6, -1,-1,-1, 10, -1,-1,-1, 14, -1,-1,-1, 2, -1,-1,-1, 6, -1,-1,-1, 10, -1,-1,-1, 14, -1,-1,-1);
    const __m256i maskB = _mm256_setr_epi8(3, -1,-1,-1, 7, -1,-1,-1, 11, -1,-1,-1, 15, -1,-1,-1, 3, -1,-1,-1, 7, -1,-1,-1, 11, -1,-1,-1, 15, -1,-1,-1);

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const PF_Pixel_ARGB_8u* rowSrc = reinterpret_cast<const PF_Pixel_ARGB_8u*>((const uint8_t*)pRGB + y * strideSrc);
        float* rowL  = reinterpret_cast<float*>((uint8_t*)pL  + y * strideL);
        float* rowAB = reinterpret_cast<float*>((uint8_t*)pAB + y * strideAB);

        int32_t x = 0;
        for (; x <= sizeX - 8; x += 8)
        {
            __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowSrc + x));

            __m256 R = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskR)), vInv255);
            __m256 G = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskG)), vInv255);
            __m256 B = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskB)), vInv255);

            __m256 X = _mm256_fmadd_ps(R, mRX, _mm256_fmadd_ps(G, mGX, _mm256_mul_ps(B, mBX)));
            __m256 Y = _mm256_fmadd_ps(R, mRY, _mm256_fmadd_ps(G, mGY, _mm256_mul_ps(B, mBY)));
            __m256 Z = _mm256_fmadd_ps(R, mRZ, _mm256_fmadd_ps(G, mGZ, _mm256_mul_ps(B, mBZ)));

            __m256 fX = _mm256_lab_f_ps(_mm256_mul_ps(X, _mm256_set1_ps(c_Xn_inv)));
            __m256 fY = _mm256_lab_f_ps(Y);
            __m256 fZ = _mm256_lab_f_ps(_mm256_mul_ps(Z, _mm256_set1_ps(c_Zn_inv)));

            _mm256_storeu_ps(rowL + x, _mm256_fmsub_ps(_mm256_set1_ps(116.0f), fY, _mm256_set1_ps(16.0f)));

            __m256 a = _mm256_mul_ps(_mm256_set1_ps(500.0f), _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(_mm256_set1_ps(200.0f), _mm256_sub_ps(fY, fZ));

            __m256 ab_lo = _mm256_unpacklo_ps(a, b);
            __m256 ab_hi = _mm256_unpackhi_ps(a, b);
            _mm256_storeu_ps(rowAB + (x * 2),     _mm256_permute2f128_ps(ab_lo, ab_hi, 0x20));
            _mm256_storeu_ps(rowAB + (x * 2) + 8, _mm256_permute2f128_ps(ab_lo, ab_hi, 0x31));
        }
        // Tail handling
        for (; x < sizeX; ++x)
        {
            float r = rowSrc[x].R * c_inv255;
            float g = rowSrc[x].G * c_inv255;
            float b = rowSrc[x].B * c_inv255;
            float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
            float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
            float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

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