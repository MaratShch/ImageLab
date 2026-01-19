#include "Avx2ColorConverts.hpp"
#include "FastAriphmetics.hpp"

constexpr float c_inv32767   = 1.0f / 32767.0f;

// --- BGRA_16u Implementation ---
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_BGRA_16u* RESTRICT pRGB,
    float*                   RESTRICT pL,
    float*                   RESTRICT pAB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            srcPitch,
    const int32_t            labPitch
) noexcept
{
    const intptr_t strideSrc = (intptr_t)srcPitch * sizeof(PF_Pixel_BGRA_16u);
    const intptr_t strideL   = (intptr_t)labPitch * sizeof(float);
    const intptr_t strideAB  = (intptr_t)labPitch * 2 * sizeof(float);

    const __m256 mRX = _mm256_set1_ps(0.4124564f); const __m256 mGX = _mm256_set1_ps(0.3575761f); const __m256 mBX = _mm256_set1_ps(0.1804375f);
    const __m256 mRY = _mm256_set1_ps(0.2126729f); const __m256 mGY = _mm256_set1_ps(0.7151522f); const __m256 mBY = _mm256_set1_ps(0.0721750f);
    const __m256 mRZ = _mm256_set1_ps(0.0193339f); const __m256 mGZ = _mm256_set1_ps(0.1191920f); const __m256 mBZ = _mm256_set1_ps(0.9503041f);

    const __m256 vInvScale = _mm256_set1_ps(c_inv32767);

    // Shuffle masks to extract 4 pixels (16-bit) to 32-bit float lanes
    // BGRA_16: [B0_lo, B0_hi, G0_lo, G0_hi, R0_lo, R0_hi, A0_lo, A0_hi, ...]
    // Mask selects 2 bytes for the channel and puts them in the low 16 bits of each 32-bit dword
    const __m256i maskR = _mm256_setr_epi8(4,5, -1,-1, 12,13, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 4,5, -1,-1, 12,13, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    const __m256i maskG = _mm256_setr_epi8(2,3, -1,-1, 10,11, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 2,3, -1,-1, 10,11, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    const __m256i maskB = _mm256_setr_epi8(0,1, -1,-1, 8,9,   -1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 0,1, -1,-1, 8,9,   -1,-1, -1,-1,-1,-1, -1,-1,-1,-1);

    for (int32_t y = 0; y < sizeY; ++y)
	{
        const PF_Pixel_BGRA_16u* rowSrc = reinterpret_cast<const PF_Pixel_BGRA_16u*>((const uint8_t*)pRGB + y * strideSrc);
        float* rowL  = reinterpret_cast<float*>((uint8_t*)pL + y * strideL);
        float* rowAB = reinterpret_cast<float*>((uint8_t*)pAB + y * strideAB);

        int32_t x = 0;
        // Process 4 pixels at a time (each pixel is 8 bytes, 4 pixels = 32 bytes = 1 YMM)
        for (; x <= sizeX - 4; x += 4)
		{
            __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowSrc + x));

            // Extract channels and convert to float
            // We use PSHUFB to move bytes, then cvtepi32_ps. 
            // Note: Since we only have 4 pixels in 256 bits, we only use 4 lanes of the resulting __m256
            __m256 R = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskR));
            __m256 G = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskG));
            __m256 B = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskB));
            
            // To get 4 pixels into the low part of a __m128 for easier management
            // We combine the high and low lanes of the shuffles
            __m128 R4 = _mm_add_ps(_mm256_castps256_ps128(R), _mm256_extractf128_ps(R, 1));
            __m128 G4 = _mm_add_ps(_mm256_castps256_ps128(G), _mm256_extractf128_ps(G, 1));
            __m128 B4 = _mm_add_ps(_mm256_castps256_ps128(B), _mm256_extractf128_ps(B, 1));

            // Upgrade to __m256 for the math pipeline (using only 4 lanes)
            __m256 R_f = _mm256_insertf128_ps(_mm256_castps128_ps256(R4), _mm_setzero_ps(), 1);
            __m256 G_f = _mm256_insertf128_ps(_mm256_castps128_ps256(G4), _mm_setzero_ps(), 1);
            __m256 B_f = _mm256_insertf128_ps(_mm256_castps128_ps256(B4), _mm_setzero_ps(), 1);

            R_f = _mm256_mul_ps(R_f, vInvScale);
            G_f = _mm256_mul_ps(G_f, vInvScale);
            B_f = _mm256_mul_ps(B_f, vInvScale);

            __m256 X = _mm256_fmadd_ps(R_f, mRX, _mm256_fmadd_ps(G_f, mGX, _mm256_mul_ps(B_f, mBX)));
            __m256 Y = _mm256_fmadd_ps(R_f, mRY, _mm256_fmadd_ps(G_f, mGY, _mm256_mul_ps(B_f, mBY)));
            __m256 Z = _mm256_fmadd_ps(R_f, mRZ, _mm256_fmadd_ps(G_f, mGZ, _mm256_mul_ps(B_f, mBZ)));

            __m256 fX = _mm256_lab_f_ps(_mm256_mul_ps(X, _mm256_set1_ps(c_Xn_inv)));
            __m256 fY = _mm256_lab_f_ps(Y);
            __m256 fZ = _mm256_lab_f_ps(_mm256_mul_ps(Z, _mm256_set1_ps(c_Zn_inv)));

            __m256 L = _mm256_fmsub_ps(_mm256_set1_ps(116.0f), fY, _mm256_set1_ps(16.0f));
            __m256 a = _mm256_mul_ps(_mm256_set1_ps(500.0f), _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(_mm256_set1_ps(200.0f), _mm256_sub_ps(fY, fZ));

            // Store 4 floats for L
            _mm_storeu_ps(rowL + x, _mm256_castps256_ps128(L));

            // Interleave a and b (4 pixels = 8 floats)
            __m128 a4 = _mm256_castps256_ps128(a);
            __m128 b4 = _mm256_castps256_ps128(b);
            _mm_storeu_ps(rowAB + (x * 2),     _mm_unpacklo_ps(a4, b4));
            _mm_storeu_ps(rowAB + (x * 2) + 4, _mm_unpackhi_ps(a4, b4));
        }

        // Tail
        for (; x < sizeX; ++x)
		{
            float r = rowSrc[x].R * c_inv32767;
            float g = rowSrc[x].G * c_inv32767;
            float b = rowSrc[x].B * c_inv32767;
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
}

// --- ARGB_16u Implementation ---
void AVX2_ConvertRgbToCIELab_SemiPlanar
(
    const PF_Pixel_ARGB_16u* RESTRICT pRGB,
    float*                   RESTRICT pL,
    float*                   RESTRICT pAB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            srcPitch,
    const int32_t            labPitch
) noexcept
{
    const intptr_t strideSrc = (intptr_t)srcPitch * sizeof(PF_Pixel_ARGB_16u);
    const intptr_t strideL   = (intptr_t)labPitch * sizeof(float);
    const intptr_t strideAB  = (intptr_t)labPitch * 2 * sizeof(float);

    const __m256 mRX = _mm256_set1_ps(0.4124564f); const __m256 mGX = _mm256_set1_ps(0.3575761f); const __m256 mBX = _mm256_set1_ps(0.1804375f);
    const __m256 mRY = _mm256_set1_ps(0.2126729f); const __m256 mGY = _mm256_set1_ps(0.7151522f); const __m256 mBY = _mm256_set1_ps(0.0721750f);
    const __m256 mRZ = _mm256_set1_ps(0.0193339f); const __m256 mGZ = _mm256_set1_ps(0.1191920f); const __m256 mBZ = _mm256_set1_ps(0.9503041f);

    const __m256 vInvScale = _mm256_set1_ps(c_inv32767);

    // ARGB_16: 0:A, 1:R, 2:G, 3:B (each 2 bytes)
    const __m256i maskR = _mm256_setr_epi8(2,3, -1,-1, 10,11, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 2,3, -1,-1, 10,11, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    const __m256i maskG = _mm256_setr_epi8(4,5, -1,-1, 12,13, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 4,5, -1,-1, 12,13, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1);
    const __m256i maskB = _mm256_setr_epi8(6,7, -1,-1, 14,15, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 6,7, -1,-1, 14,15, -1,-1, -1,-1,-1,-1, -1,-1,-1,-1);

    for (int32_t y = 0; y < sizeY; ++y)
	{
        const PF_Pixel_ARGB_16u* rowSrc = reinterpret_cast<const PF_Pixel_ARGB_16u*>((const uint8_t*)pRGB + y * strideSrc);
        float* rowL  = reinterpret_cast<float*>((uint8_t*)pL + y * strideL);
        float* rowAB = reinterpret_cast<float*>((uint8_t*)pAB + y * strideAB);

        int32_t x = 0;
        for (; x <= sizeX - 4; x += 4) {
            __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowSrc + x));

            __m256 R = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskR));
            __m256 G = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskG));
            __m256 B = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(raw, maskB));
            
            __m128 R4 = _mm_add_ps(_mm256_castps256_ps128(R), _mm256_extractf128_ps(R, 1));
            __m128 G4 = _mm_add_ps(_mm256_castps256_ps128(G), _mm256_extractf128_ps(G, 1));
            __m128 B4 = _mm_add_ps(_mm256_castps256_ps128(B), _mm256_extractf128_ps(B, 1));

            __m256 R_f = _mm256_insertf128_ps(_mm256_castps128_ps256(R4), _mm_setzero_ps(), 1);
            __m256 G_f = _mm256_insertf128_ps(_mm256_castps128_ps256(G4), _mm_setzero_ps(), 1);
            __m256 B_f = _mm256_insertf128_ps(_mm256_castps128_ps256(B4), _mm_setzero_ps(), 1);

            R_f = _mm256_mul_ps(R_f, vInvScale);
            G_f = _mm256_mul_ps(G_f, vInvScale);
            B_f = _mm256_mul_ps(B_f, vInvScale);

            __m256 X = _mm256_fmadd_ps(R_f, mRX, _mm256_fmadd_ps(G_f, mGX, _mm256_mul_ps(B_f, mBX)));
            __m256 Y = _mm256_fmadd_ps(R_f, mRY, _mm256_fmadd_ps(G_f, mGY, _mm256_mul_ps(B_f, mBY)));
            __m256 Z = _mm256_fmadd_ps(R_f, mRZ, _mm256_fmadd_ps(G_f, mGZ, _mm256_mul_ps(B_f, mBZ)));

            __m256 fX = _mm256_lab_f_ps(_mm256_mul_ps(X, _mm256_set1_ps(c_Xn_inv)));
            __m256 fY = _mm256_lab_f_ps(Y);
            __m256 fZ = _mm256_lab_f_ps(_mm256_mul_ps(Z, _mm256_set1_ps(c_Zn_inv)));

            __m256 L = _mm256_fmsub_ps(_mm256_set1_ps(116.0f), fY, _mm256_set1_ps(16.0f));
            __m256 a = _mm256_mul_ps(_mm256_set1_ps(500.0f), _mm256_sub_ps(fX, fY));
            __m256 b = _mm256_mul_ps(_mm256_set1_ps(200.0f), _mm256_sub_ps(fY, fZ));

            _mm_storeu_ps(rowL + x, _mm256_castps256_ps128(L));
            __m128 a4 = _mm256_castps256_ps128(a);
            __m128 b4 = _mm256_castps256_ps128(b);
            _mm_storeu_ps(rowAB + (x * 2),     _mm_unpacklo_ps(a4, b4));
            _mm_storeu_ps(rowAB + (x * 2) + 4, _mm_unpackhi_ps(a4, b4));
        }
		
        // Tail
        for (; x < sizeX; ++x)
		{
            float r = rowSrc[x].R * c_inv32767;
            float g = rowSrc[x].G * c_inv32767;
            float b = rowSrc[x].B * c_inv32767;
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