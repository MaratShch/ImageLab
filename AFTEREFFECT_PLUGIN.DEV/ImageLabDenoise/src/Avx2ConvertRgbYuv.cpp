#include "ColorConvert.hpp"


void AVX2_Convert_BGRA_8u_YUV
(
    const PF_Pixel_BGRA_8u* RESTRICT pInput,// Input BGRA_8u (Interleaved)
    float* RESTRICT Y_out,                	// Y plane (Luma)
    float* RESTRICT U_out,                	// U plane (Red-Blue axis)
    float* RESTRICT V_out,                	// V plane (Green-Magenta axis)
    int32_t sizeX,                        	// Width in Pixels
    int32_t sizeY,                        	// Height in Pixels
    int32_t linePitch                     	// Row Pitch in Pixels
) noexcept
{
    // 1. Initialize Orthonormal Constants (1/sqrt3, 1/sqrt2, 1/sqrt6)
    const __m256 vSqrt3_inv = _mm256_set1_ps(0.57735027f);
    const __m256 vSqrt2_inv = _mm256_set1_ps(0.70710678f);
    const __m256 vSqrt6_inv = _mm256_set1_ps(0.40824829f);
    const __m256 vTwo       = _mm256_set1_ps(2.0f);

    // 2. Prepare Shuffle Masks
    // AVX2 _mm256_shuffle_epi8 works within 128-bit lanes. 
    // We load 8 pixels (32 bytes). Lane 0: Pix 0-3, Lane 1: Pix 4-7.
    // Masks extract the specific byte and place it at the start of a 32-bit dword.
    const __m256i maskB = _mm256_set_epi8
	(
        -1, -1, -1, 12, -1, -1, -1, 8, -1, -1, -1, 4, -1, -1, -1, 0, // Lane 1 (Pix 4-7)
        -1, -1, -1, 12, -1, -1, -1, 8, -1, -1, -1, 4, -1, -1, -1, 0  // Lane 0 (Pix 0-3)
    );
    const __m256i maskG = _mm256_set_epi8
	(
        -1, -1, -1, 13, -1, -1, -1, 9, -1, -1, -1, 5, -1, -1, -1, 1,
        -1, -1, -1, 13, -1, -1, -1, 9, -1, -1, -1, 5, -1, -1, -1, 1
    );
    const __m256i maskR = _mm256_set_epi8
	(
        -1, -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2,
        -1, -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2
    );

    const int32_t vecSize = 8;
    const int32_t endX = (sizeX / vecSize) * vecSize;

    for (int32_t y = 0; y < sizeY; ++y)
	{
        // Compute row starts based on Pixel Pitches
        const PF_Pixel_BGRA_8u* rowIn = pInput + (y * linePitch);
        float* rowY = Y_out + (y * sizeX); // Planar output has no pitch usually, 
        float* rowU = U_out + (y * sizeX); // but we use sizeX as stride for planar.
        float* rowV = V_out + (y * sizeX);

        int32_t x = 0;

        // AVX2 MAIN KERNEL (8 Pixels per iteration)
        for (; x < endX; x += vecSize)
		{
            // Load 8 pixels (32 bytes = 256 bits)
            __m256i bgra = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowIn + x));

            // Extract Channels and convert to Float
            // Result: fR, fG, fB each contain 8 floats [p0, p1, p2, p3, p4, p5, p6, p7]
            __m256 fB = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(bgra, maskB));
            __m256 fG = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(bgra, maskG));
            __m256 fR = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(bgra, maskR));

            // Perform Orthonormal Math
            // Y = (R + G + B) * (1/sqrt3)
            __m256 resY = _mm256_mul_ps(_mm256_add_ps(fR, _mm256_add_ps(fG, fB)), vSqrt3_inv);
            
            // U = (R - B) * (1/sqrt2)
            __m256 resU = _mm256_mul_ps(_mm256_sub_ps(fR, fB), vSqrt2_inv);
            
            // V = (R + B - 2G) * (1/sqrt6)
            __m256 twoG = _mm256_mul_ps(fG, vTwo);
            __m256 resV = _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(fR, fB), twoG), vSqrt6_inv);

            // Store Planar Results (Unaligned store as we can't guarantee 32-byte alignment on row starts)
            _mm256_storeu_ps(rowY + x, resY);
            _mm256_storeu_ps(rowU + x, resU);
            _mm256_storeu_ps(rowV + x, resV);

        }

        // SCALAR TAIL (Handle remaining pixels < 8)
        for (; x < sizeX; ++x)
		{
            float b = static_cast<float>(rowIn[x].B);
            float g = static_cast<float>(rowIn[x].G);
            float r = static_cast<float>(rowIn[x].R);

            rowY[x] = (r + g + b) * 0.57735027f;
            rowU[x] = (r - b) * 0.70710678f;
            rowV[x] = (r + b - 2.0f * g) * 0.40824829f;
        }
    }
	
	return;
}

