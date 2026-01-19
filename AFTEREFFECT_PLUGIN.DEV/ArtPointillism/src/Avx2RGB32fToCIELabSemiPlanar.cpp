#include "Avx2ColorConverts.hpp"
#include "FastAriphmetics.hpp"

#ifdef _WINDOWS
// Use template <int idx> to force compile-time constant for MSVC
template <int idx>
inline __m256 _mm256_gather_chan_ps(__m256 v0, __m256 v1, __m256 v2, __m256 v3) noexcept
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
inline __m256 _mm256_gather_chan_ps(__m256 v0, __m256 v1, __m256 v2, __m256 v3, const int idx) noexcept
{
    // shuffle_ps extracts component across lanes: [v0_L v0_H v1_L v1_H] -> [comp_0 comp_2 comp_1 comp_3]
    const int m = _MM_SHUFFLE(idx, idx, idx, idx);
    __m256 r01 = _mm256_shuffle_ps(v0, v1, m); // [c0 c0 c2 c2 | c1 c1 c3 c3]
    __m256 r23 = _mm256_shuffle_ps(v2, v3, m); // [c4 c4 c6 c6 | c5 c5 c7 c7]

                                               // Use permutevar to organize indices: 0, 4, 1, 5, 2, 6, 3, 7 -> [c0..c7]
    return _mm256_permutevar8x32_ps(_mm256_blend_ps(r01, r23, 0xF0),
        _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}
#endif // _WINDOWS

// --- BGRA_32f Implementation ---
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_BGRA_32f* RESTRICT pRGB,
    float*                   RESTRICT pL,
    float*                   RESTRICT pAB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            srcPitch,
    const int32_t            labPitch
) noexcept
{
    const intptr_t strideSrc = (intptr_t)srcPitch * sizeof(PF_Pixel_BGRA_32f);
    const intptr_t strideL   = (intptr_t)labPitch * sizeof(float);
    const intptr_t strideAB  = (intptr_t)labPitch * 2 * sizeof(float);

    const __m256 mRX = _mm256_set1_ps(0.4124564f); const __m256 mGX = _mm256_set1_ps(0.3575761f); const __m256 mBX = _mm256_set1_ps(0.1804375f);
    const __m256 mRY = _mm256_set1_ps(0.2126729f); const __m256 mGY = _mm256_set1_ps(0.7151522f); const __m256 mBY = _mm256_set1_ps(0.0721750f);
    const __m256 mRZ = _mm256_set1_ps(0.0193339f); const __m256 mGZ = _mm256_set1_ps(0.1191920f); const __m256 mBZ = _mm256_set1_ps(0.9503041f);

    for (int32_t y = 0; y < sizeY; ++y)
	{
        const PF_Pixel_BGRA_32f* rowSrc = reinterpret_cast<const PF_Pixel_BGRA_32f*>((const uint8_t*)pRGB + y * strideSrc);
        float* rowL  = reinterpret_cast<float*>((uint8_t*)pL + y * strideL);
        float* rowAB = reinterpret_cast<float*>((uint8_t*)pAB + y * strideAB);

        int32_t x = 0;
        for (; x <= sizeX - 8; x += 8)
		{
            // Load 8 pixels (128 bytes)
            __m256 v0 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 0));
            __m256 v1 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 2));
            __m256 v2 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 4));
            __m256 v3 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 6));

#ifdef _WINDOWS
            __m256 R = _mm256_gather_chan_ps<2>(v0, v1, v2, v3); // Called with <2>
            __m256 G = _mm256_gather_chan_ps<1>(v0, v1, v2, v3); // Called with <1>
            __m256 B = _mm256_gather_chan_ps<0>(v0, v1, v2, v3); // Called with <0>
#else
            // De-interleave BGRA (B:0, G:1, R:2, A:3)
            __m256 R = _mm256_gather_chan_ps(v0, v1, v2, v3, 2);
            __m256 G = _mm256_gather_chan_ps(v0, v1, v2, v3, 1);
            __m256 B = _mm256_gather_chan_ps(v0, v1, v2, v3, 0);
#endif

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

        // Tail
        for (; x < sizeX; ++x)
		{
            float r = rowSrc[x].R; float g = rowSrc[x].G; float b = rowSrc[x].B;
            float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
            float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
            float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;
            
			auto f = [](float t) noexcept
			{ 
				return (t > c_delta_sq3) ? FastCompute::Pow(t, 1.0f/3.0f) : (c_lin_slope * t + c_lin_const); 
			};
            
			float fx = f(X * c_Xn_inv); float fy = f(Y); float fz = f(Z * c_Zn_inv);
            rowL[x] = 116.0f * fy - 16.0f;
            rowAB[x * 2] = 500.0f * (fx - fy);
            rowAB[x * 2 + 1] = 200.0f * (fy - fz);
        }
    }
	
	return;
}

// --- ARGB_32f Implementation ---
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_ARGB_32f* RESTRICT pRGB,
    float*                   RESTRICT pL,
    float*                   RESTRICT pAB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            srcPitch,
    const int32_t            labPitch
) noexcept
{
    const intptr_t strideSrc = (intptr_t)srcPitch * sizeof(PF_Pixel_ARGB_32f);
    const intptr_t strideL   = (intptr_t)labPitch * sizeof(float);
    const intptr_t strideAB  = (intptr_t)labPitch * 2 * sizeof(float);

    const __m256 mRX = _mm256_set1_ps(0.4124564f); const __m256 mGX = _mm256_set1_ps(0.3575761f); const __m256 mBX = _mm256_set1_ps(0.1804375f);
    const __m256 mRY = _mm256_set1_ps(0.2126729f); const __m256 mGY = _mm256_set1_ps(0.7151522f); const __m256 mBY = _mm256_set1_ps(0.0721750f);
    const __m256 mRZ = _mm256_set1_ps(0.0193339f); const __m256 mGZ = _mm256_set1_ps(0.1191920f); const __m256 mBZ = _mm256_set1_ps(0.9503041f);

    for (int32_t y = 0; y < sizeY; ++y)
	{
        const PF_Pixel_ARGB_32f* rowSrc = reinterpret_cast<const PF_Pixel_ARGB_32f*>((const uint8_t*)pRGB + y * strideSrc);
        float* rowL  = reinterpret_cast<float*>((uint8_t*)pL + y * strideL);
        float* rowAB = reinterpret_cast<float*>((uint8_t*)pAB + y * strideAB);

        int32_t x = 0;
        for (; x <= sizeX - 8; x += 8)
		{
            __m256 v0 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 0));
            __m256 v1 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 2));
            __m256 v2 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 4));
            __m256 v3 = _mm256_loadu_ps(reinterpret_cast<const float*>(rowSrc + x + 6));

#ifdef _WINDOWS
            __m256 R = _mm256_gather_chan_ps<1>(v0, v1, v2, v3); // Called with <2>
            __m256 G = _mm256_gather_chan_ps<2>(v0, v1, v2, v3); // Called with <1>
            __m256 B = _mm256_gather_chan_ps<3>(v0, v1, v2, v3); // Called with <0>
#else
            // De-interleave ARGB (A:0, R:1, G:2, B:3)
            __m256 R = _mm256_gather_chan_ps(v0, v1, v2, v3, 1);
            __m256 G = _mm256_gather_chan_ps(v0, v1, v2, v3, 2);
            __m256 B = _mm256_gather_chan_ps(v0, v1, v2, v3, 3);
#endif

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

        // Tail
        for (; x < sizeX; ++x)
		{
            float r = rowSrc[x].R; float g = rowSrc[x].G; float b = rowSrc[x].B;
            float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
            float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
            float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;
            
			auto f = [](float t) noexcept
			{ 
				return (t > c_delta_sq3) ? FastCompute::Pow(t, 1.0f/3.0f) : (c_lin_slope * t + c_lin_const); 
			};
            
			float fx = f(X * c_Xn_inv); float fy = f(Y); float fz = f(Z * c_Zn_inv);
            rowL[x] = 116.0f * fy - 16.0f;
            rowAB[x * 2] = 500.0f * (fx - fy);
            rowAB[x * 2 + 1] = 200.0f * (fy - fz);
        }
    }
	
	return;
}