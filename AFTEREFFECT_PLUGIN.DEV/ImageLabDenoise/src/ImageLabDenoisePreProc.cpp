#include "ImageLabDenoiseAlgo.hpp"

void Rgb2YuvSemiplanar_BGRA_8u_AVX2
(
    const PF_Pixel_BGRA_8u* RESTRICT in,
    float* RESTRICT dstY,
    float* RESTRICT dstUV,
    int32_t width,
    int32_t height,
    int32_t linePitch
) noexcept
{
    // 1. Pre-compute Scaled Matrix (Combines Color Matrix + Normalization)
    // --------------------------------------------------------------------
    // We assume input is 0..255 and we want 0..1 range applied to matrix.
    const float normScale = 1.0f / 255.0f;

    // Load base matrix (assuming RGB2YUV is accessible global)
    // Note: User code used RGB2YUV[1]. Adjust index as needed.
    const float* baseM = RGB2YUV[1];

    // Pre-scale coefficients
    const __m256 m0 = _mm256_set1_ps(baseM[0] * normScale);
    const __m256 m1 = _mm256_set1_ps(baseM[1] * normScale);
    const __m256 m2 = _mm256_set1_ps(baseM[2] * normScale);

    const __m256 m3 = _mm256_set1_ps(baseM[3] * normScale);
    const __m256 m4 = _mm256_set1_ps(baseM[4] * normScale);
    const __m256 m5 = _mm256_set1_ps(baseM[5] * normScale);

    const __m256 m6 = _mm256_set1_ps(baseM[6] * normScale);
    const __m256 m7 = _mm256_set1_ps(baseM[7] * normScale);
    const __m256 m8 = _mm256_set1_ps(baseM[8] * normScale);

    // Shuffle Masks for BGRA -> Planar Int32
    const __m256i mask_B = _mm256_setr_epi8(0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1, 0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1);
    const __m256i mask_G = _mm256_setr_epi8(1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1, 1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1);
    const __m256i mask_R = _mm256_setr_epi8(2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1, 2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1);

    for (int32_t j = 0; j < height; j++)
    {
        const uint8_t* RESTRICT pSrcRow = reinterpret_cast<const uint8_t*>(in + j * linePitch);
        float* RESTRICT pDstY  = dstY  + j * width;
        float* RESTRICT pDstUV = dstUV + j * width * 2; // UV is interleaved, so 2x width

        int32_t i = 0;
        const int32_t vecWidth = width - 7;

        for (; i < vecWidth; i += 8)
        {
            // 2. Load and Convert to Planar Float
            // -----------------------------------
            __m256i vPixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrcRow + i * 4));

            __m256 vB = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vPixels, mask_B));
            __m256 vG = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vPixels, mask_G));
            __m256 vR = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vPixels, mask_R));

            // 3. Compute Y (Planar)
            // ---------------------
            // Y = R*m0 + G*m1 + B*m2 (mcoeffs already include 1/255 scale)
            __m256 vY = _mm256_fmadd_ps(vR, m0, _mm256_fmadd_ps(vG, m1, _mm256_mul_ps(vB, m2)));

            // Store Y
            _mm256_storeu_ps(pDstY + i, vY);

            // 4. Compute U and V
            // ------------------
            __m256 vU = _mm256_fmadd_ps(vR, m3, _mm256_fmadd_ps(vG, m4, _mm256_mul_ps(vB, m5)));
            __m256 vV = _mm256_fmadd_ps(vR, m6, _mm256_fmadd_ps(vG, m7, _mm256_mul_ps(vB, m8)));

            // 5. Interleave U and V
            // ---------------------
            // We have: U [0 1 2 3 | 4 5 6 7]
            //          V [0 1 2 3 | 4 5 6 7]
            // We want: UV [0 0 1 1 2 2 3 3] then [4 4 5 5 6 6 7 7]

            // Unpack Low: [U0 V0 U1 V1 | U4 V4 U5 V5] (Cross-lane issue handled below)
            __m256 vUV_a = _mm256_unpacklo_ps(vU, vV);
            // Unpack High: [U2 V2 U3 V3 | U6 V6 U7 V7]
            __m256 vUV_b = _mm256_unpackhi_ps(vU, vV);

            // The unpacking above mixes the lanes correctly for pairs, but we need to store them linearly.
            // vUV_a: U0 V0 U1 V1 (Lane 0) | U4 V4 U5 V5 (Lane 1)
            // vUV_b: U2 V2 U3 V3 (Lane 0) | U6 V6 U7 V7 (Lane 1)

            // We need to write: U0 V0 U1 V1 U2 V2 U3 V3 ...
            // To do this linearly, we need to permute/shuffle.

            // Combine to get first 128 bits (4 pairs): U0 V0 U1 V1 U2 V2 U3 V3
            __m256 vOut1 = _mm256_permute2f128_ps(vUV_a, vUV_b, 0x20);
            // Combine to get second 128 bits: U4 V4 U5 V5 U6 V6 U7 V7
            __m256 vOut2 = _mm256_permute2f128_ps(vUV_a, vUV_b, 0x31);

            // Store UV (16 floats total -> 64 bytes)
            _mm256_storeu_ps(pDstUV + (i * 2), vOut1);
            _mm256_storeu_ps(pDstUV + (i * 2) + 8, vOut2);
        }

        // 6. Tail Handling
        // ----------------
        for (; i < width; ++i)
        {
            const PF_Pixel_BGRA_8u* px = reinterpret_cast<const PF_Pixel_BGRA_8u*>(pSrcRow + i * 4);

            float r = static_cast<float>(px->R) * normScale;
            float g = static_cast<float>(px->G) * normScale;
            float b = static_cast<float>(px->B) * normScale;

            // Notice we use raw base matrix here because inputs are already scaled
            pDstY[i] = r * baseM[0] + g * baseM[1] + b * baseM[2];

            float u = r * baseM[3] + g * baseM[4] + b * baseM[5];
            float v = r * baseM[6] + g * baseM[7] + b * baseM[8];

            pDstUV[i * 2] = u;
            pDstUV[i * 2 + 1] = v;
        }
    }
}
