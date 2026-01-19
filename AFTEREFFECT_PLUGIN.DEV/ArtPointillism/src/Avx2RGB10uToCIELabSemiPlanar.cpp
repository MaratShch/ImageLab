#include "Avx2ColorConverts.hpp"
#include "FastAriphmetics.hpp"

constexpr float c_inv1023   = 1.0f / 1023.0f;

// --- RGB_10u Implementation ---
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_RGB_10u* RESTRICT pRGB,
    float*                  RESTRICT pL,
    float*                  RESTRICT pAB,
    const int32_t           sizeX,
    const int32_t           sizeY,
    const int32_t           srcPitch,
    const int32_t           labPitch
) noexcept
{
    const intptr_t strideSrc = (intptr_t)srcPitch * sizeof(PF_Pixel_RGB_10u);
    const intptr_t strideL   = (intptr_t)labPitch * sizeof(float);
    const intptr_t strideAB  = (intptr_t)labPitch * 2 * sizeof(float);

    const __m256 vInv1023 = _mm256_set1_ps(c_inv1023);
    const __m256i vMask10 = _mm256_set1_epi32(0x3FF);

    // Matrix Coefficients (Rec.709 D65)
    const __m256 mRX = _mm256_set1_ps(0.4124564f); const __m256 mGX = _mm256_set1_ps(0.3575761f); const __m256 mBX = _mm256_set1_ps(0.1804375f);
    const __m256 mRY = _mm256_set1_ps(0.2126729f); const __m256 mGY = _mm256_set1_ps(0.7151522f); const __m256 mBY = _mm256_set1_ps(0.0721750f);
    const __m256 mRZ = _mm256_set1_ps(0.0193339f); const __m256 mGZ = _mm256_set1_ps(0.1191920f); const __m256 mBZ = _mm256_set1_ps(0.9503041f);

    for (int32_t y = 0; y < sizeY; ++y)
	{
        // Handle negative pitch by casting to uint8_t for stride math
        const PF_Pixel_RGB_10u* rowSrc = reinterpret_cast<const PF_Pixel_RGB_10u*>((const uint8_t*)pRGB + y * strideSrc);
        float* rowL  = reinterpret_cast<float*>((uint8_t*)pL + y * strideL);
        float* rowAB = reinterpret_cast<float*>((uint8_t*)pAB + y * strideAB);

        int32_t x = 0;
        for (; x <= sizeX - 8; x += 8)
		{
            // Load 8 packed 10-bit pixels
            __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowSrc + x));

            // Extract channels via logical shifts and masks
            // Red:   bits 22-31
            // Green: bits 12-21
            // Blue:  bits 2-11
            __m256 R = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(raw, 22), vMask10));
            __m256 G = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(raw, 12), vMask10));
            __m256 B = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(raw, 2),  vMask10));

            // Normalize to 0.0 ... 1.0
            R = _mm256_mul_ps(R, vInv1023);
            G = _mm256_mul_ps(G, vInv1023);
            B = _mm256_mul_ps(B, vInv1023);

            // Matrix Transform RGB -> XYZ
            __m256 X = _mm256_fmadd_ps(R, mRX, _mm256_fmadd_ps(G, mGX, _mm256_mul_ps(B, mBX)));
            __m256 Y = _mm256_fmadd_ps(R, mRY, _mm256_fmadd_ps(G, mGY, _mm256_mul_ps(B, mBY)));
            __m256 Z = _mm256_fmadd_ps(R, mRZ, _mm256_fmadd_ps(G, mGZ, _mm256_mul_ps(B, mBZ)));

            // XYZ -> f(t)
            __m256 fX = _mm256_lab_f_ps(_mm256_mul_ps(X, _mm256_set1_ps(c_Xn_inv)));
            __m256 fY = _mm256_lab_f_ps(Y);
            __m256 fZ = _mm256_lab_f_ps(_mm256_mul_ps(Z, _mm256_set1_ps(c_Zn_inv)));

            // Store Planar L: L = 116 * f(Y) - 16
            _mm256_storeu_ps(rowL + x, _mm256_fmsub_ps(_mm256_set1_ps(116.0f), fY, _mm256_set1_ps(16.0f)));

            // Calculate a and b
            __m256 a = _mm256_mul_ps(_mm256_set1_ps(500.0f), _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(_mm256_set1_ps(200.0f), _mm256_sub_ps(fY, fZ));

            // Interleave a and b for semi-planar output
            __m256 ab_lo = _mm256_unpacklo_ps(a, b);
            __m256 ab_hi = _mm256_unpackhi_ps(a, b);
            
            // Reorder lanes (AVX2 fix) and store
            _mm256_storeu_ps(rowAB + (x * 2),     _mm256_permute2f128_ps(ab_lo, ab_hi, 0x20));
            _mm256_storeu_ps(rowAB + (x * 2) + 8, _mm256_permute2f128_ps(ab_lo, ab_hi, 0x31));
        }

        // Tail handling for sizeX % 8 != 0
        for (; x < sizeX; ++x)
		{
            uint32_t val;
            // Accessing packed bitfields manually for the tail
            // Standard C++ bitfield access is used for safety
            float r = (float)rowSrc[x].R * c_inv1023;
            float g = (float)rowSrc[x].G * c_inv1023;
            float b = (float)rowSrc[x].B * c_inv1023;

            float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
            float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
            float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

            auto f_tail = [](float t) noexcept
			{ 
                return (t > c_delta_sq3) ? FastCompute::Pow(t, 1.0f/3.0f) : (c_lin_slope * t + c_lin_const); 
            };

            float fx = f_tail(X * c_Xn_inv); float fy = f_tail(Y); float fz = f_tail(Z * c_Zn_inv);
            rowL[x] = 116.0f * fy - 16.0f;
            rowAB[x * 2]     = 500.0f * (fx - fy);
            rowAB[x * 2 + 1] = 200.0f * (fy - fz);
        }
    }
	
	return;
}
