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
    const float* RESTRICT baseM = RGB2YUV[1];

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
        float* RESTRICT pDstY = dstY + j * width;
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


void Rgb2YuvSemiplanar_ARGB_8u_AVX2
(
    const PF_Pixel_ARGB_8u* RESTRICT in,
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
    const float* RESTRICT baseM = RGB2YUV[1];

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

    // We skip index 0 (Alpha)
    // R is at 1, 5, 9...
    const __m256i mask_R = _mm256_setr_epi8(1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1, 1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1);
    // G is at 2, 6, 10...
    const __m256i mask_G = _mm256_setr_epi8(2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1, 2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1);
    // B is at 3, 7, 11...
    const __m256i mask_B = _mm256_setr_epi8(3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1, 3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1);

    for (int32_t j = 0; j < height; j++)
    {
        const uint8_t* RESTRICT pSrcRow = reinterpret_cast<const uint8_t*>(in + j * linePitch);
        float* RESTRICT pDstY = dstY + j * width;
        float* RESTRICT pDstUV = dstUV + j * width * 2; // UV is interleaved, so 2x width

        int32_t i = 0;
        const int32_t vecWidth = width - 7;

        for (; i < vecWidth; i += 8)
        {
            // 2. Load and Convert to Planar Float
            // -----------------------------------
            __m256i vPixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrcRow + i * 4));

            __m256 vR = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vPixels, mask_R));
            __m256 vG = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vPixels, mask_G));
            __m256 vB = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(vPixels, mask_B));

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


void Rgb2YuvSemiplanar_BGRA_32f_AVX2
(
    const PF_Pixel_BGRA_32f* RESTRICT in,
    float* RESTRICT dstY,
    float* RESTRICT dstUV,
    int32_t width,
    int32_t height,
    int32_t linePitch
) noexcept
{
    // 1. Prepare Constants
    // --------------------------------------------------------------------
    // Load base matrix directly (no 1/255 scaling needed for float pixels)
    // Assuming RGB2YUV is global. Using row [1] based on your previous snippet.
    const float* baseM = RGB2YUV[1];

    const __m256 m0 = _mm256_set1_ps(baseM[0]);
    const __m256 m1 = _mm256_set1_ps(baseM[1]);
    const __m256 m2 = _mm256_set1_ps(baseM[2]);

    const __m256 m3 = _mm256_set1_ps(baseM[3]);
    const __m256 m4 = _mm256_set1_ps(baseM[4]);
    const __m256 m5 = _mm256_set1_ps(baseM[5]);

    const __m256 m6 = _mm256_set1_ps(baseM[6]);
    const __m256 m7 = _mm256_set1_ps(baseM[7]);
    const __m256 m8 = _mm256_set1_ps(baseM[8]);

    // Permutation mask to linearize the transposed data
    // The shuffle step produces: [0, 2, 4, 6] in low lane, [1, 3, 5, 7] in high lane.
    // We want: 0, 1, 2, 3, 4, 5, 6, 7 (indices refer to original pixel order)
    const __m256i vPermMask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (int32_t j = 0; j < height; j++)
    {
        // Pointers
        const float* RESTRICT pSrcRow = reinterpret_cast<const float*>(in + j * linePitch);
        float* RESTRICT pDstY = dstY + j * width;
        float* RESTRICT pDstUV = dstUV + j * width * 2;

        int32_t i = 0;
        const int32_t vecWidth = width - 7;

        for (; i < vecWidth; i += 8)
        {
            // 2. Load 8 Pixels (128 bytes / 32 floats)
            // ------------------------------------------------------------
            // Each BGRA_32f pixel is 16 bytes. 8 pixels = 4 YMM registers.
            // r0: B0 G0 R0 A0 | B1 G1 R1 A1
            // r1: B2 G2 R2 A2 | B3 G3 R3 A3 ...
            __m256 r0 = _mm256_loadu_ps(pSrcRow + i * 4);      // Pixels 0, 1
            __m256 r1 = _mm256_loadu_ps(pSrcRow + i * 4 + 8);  // Pixels 2, 3
            __m256 r2 = _mm256_loadu_ps(pSrcRow + i * 4 + 16); // Pixels 4, 5
            __m256 r3 = _mm256_loadu_ps(pSrcRow + i * 4 + 24); // Pixels 6, 7

                                                               // 3. Transpose AOS (BGRA) -> SOA (Planar)
                                                               // ------------------------------------------------------------
                                                               // Step A: Unpack Low/High (Interleave pairs)
                                                               // t0: B0 B2 G0 G2 | B1 B3 G1 G3
                                                               // t1: R0 R2 A0 A2 | R1 R3 A1 A3
            __m256 t0 = _mm256_unpacklo_ps(r0, r1);
            __m256 t1 = _mm256_unpackhi_ps(r0, r1);
            __m256 t2 = _mm256_unpacklo_ps(r2, r3);
            __m256 t3 = _mm256_unpackhi_ps(r2, r3);

            // Step B: Shuffle to gather Channels (result is still scrambled)
            // vB_mix: B0 B2 B4 B6 | B1 B3 B5 B7
            // vG_mix: G0 G2 G4 G6 | G1 G3 G5 G7
            // vR_mix: R0 R2 R4 R6 | R1 R3 R5 R7
            __m256 vB_mix = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 vG_mix = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 vR_mix = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));

            // Step C: Fix Order (Permute)
            // Result: B0 B1 B2 B3 B4 B5 B6 B7
            __m256 vB = _mm256_permutevar8x32_ps(vB_mix, vPermMask);
            __m256 vG = _mm256_permutevar8x32_ps(vG_mix, vPermMask);
            __m256 vR = _mm256_permutevar8x32_ps(vR_mix, vPermMask);

            // 4. Compute Y (Planar)
            // ------------------------------------------------------------
            __m256 vY = _mm256_fmadd_ps(vR, m0, _mm256_fmadd_ps(vG, m1, _mm256_mul_ps(vB, m2)));
            _mm256_storeu_ps(pDstY + i, vY);

            // 5. Compute U and V
            // ------------------------------------------------------------
            __m256 vU = _mm256_fmadd_ps(vR, m3, _mm256_fmadd_ps(vG, m4, _mm256_mul_ps(vB, m5)));
            __m256 vV = _mm256_fmadd_ps(vR, m6, _mm256_fmadd_ps(vG, m7, _mm256_mul_ps(vB, m8)));

            // 6. Interleave UV
            // ------------------------------------------------------------
            // Same logic as 8-bit version to store U0 V0 U1 V1...
            __m256 vUV_a = _mm256_unpacklo_ps(vU, vV); // U0 V0 U1 V1... (lane interleaved)
            __m256 vUV_b = _mm256_unpackhi_ps(vU, vV);

            __m256 vOut1 = _mm256_permute2f128_ps(vUV_a, vUV_b, 0x20); // Low 128 of a, Low 128 of b
            __m256 vOut2 = _mm256_permute2f128_ps(vUV_a, vUV_b, 0x31); // High 128 of a, High 128 of b

            _mm256_storeu_ps(pDstUV + (i * 2), vOut1);
            _mm256_storeu_ps(pDstUV + (i * 2) + 8, vOut2);
        }

        // 7. Tail Handling
        // ------------------------------------------------------------
        for (; i < width; ++i)
        {
            // Access raw floats (BGRA order)
            // pSrcRow is float*, stride is 4 floats per pixel
            const float* px = pSrcRow + i * 4;

            float b = px[0];
            float g = px[1];
            float r = px[2];
            // alpha (px[3]) ignored

            pDstY[i] = r * baseM[0] + g * baseM[1] + b * baseM[2];

            float u = r * baseM[3] + g * baseM[4] + b * baseM[5];
            float v = r * baseM[6] + g * baseM[7] + b * baseM[8];

            pDstUV[i * 2] = u;
            pDstUV[i * 2 + 1] = v;
        }
    }
}


void Rgb2YuvSemiplanar_ARGB_32f_AVX2
(
    const PF_Pixel_ARGB_32f* RESTRICT in,
    float* RESTRICT dstY,
    float* RESTRICT dstUV,
    int32_t width,
    int32_t height,
    int32_t linePitch
) noexcept
{
    // 1. Prepare Constants
    // --------------------------------------------------------------------
    // Load Color Matrix (Assuming RGB2YUV is global).
    // ARGB vs BGRA doesn't change the Matrix values, only which pixel channel multiplies them.
    const float* baseM = RGB2YUV[1];

    const __m256 m0 = _mm256_set1_ps(baseM[0]);
    const __m256 m1 = _mm256_set1_ps(baseM[1]);
    const __m256 m2 = _mm256_set1_ps(baseM[2]);

    const __m256 m3 = _mm256_set1_ps(baseM[3]);
    const __m256 m4 = _mm256_set1_ps(baseM[4]);
    const __m256 m5 = _mm256_set1_ps(baseM[5]);

    const __m256 m6 = _mm256_set1_ps(baseM[6]);
    const __m256 m7 = _mm256_set1_ps(baseM[7]);
    const __m256 m8 = _mm256_set1_ps(baseM[8]);

    // Permutation mask to linearize the transposed data.
    // The shuffle step produces: [0, 2, 4, 6] in low lane, [1, 3, 5, 7] in high lane.
    // We want: 0, 1, 2, 3, 4, 5, 6, 7 (indices refer to original pixel order).
    const __m256i vPermMask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (int32_t j = 0; j < height; j++)
    {
        // Pointers
        // Note: PF_Pixel_ARGB_32f is usually 4 floats {A, R, G, B}
        const float* RESTRICT pSrcRow = reinterpret_cast<const float*>(in + j * linePitch);
        float* RESTRICT pDstY = dstY + j * width;
        float* RESTRICT pDstUV = dstUV + j * width * 2;

        int32_t i = 0;
        const int32_t vecWidth = width - 7;

        // 2. Vector Loop (8 pixels / 32 floats per iteration)
        for (; i < vecWidth; i += 8)
        {
            // A. Load 8 Pixels (128 bytes)
            // Layout in mem: A0 R0 G0 B0 | A1 R1 G1 B1 ...
            __m256 r0 = _mm256_loadu_ps(pSrcRow + i * 4);      // Pixels 0, 1
            __m256 r1 = _mm256_loadu_ps(pSrcRow + i * 4 + 8);  // Pixels 2, 3
            __m256 r2 = _mm256_loadu_ps(pSrcRow + i * 4 + 16); // Pixels 4, 5
            __m256 r3 = _mm256_loadu_ps(pSrcRow + i * 4 + 24); // Pixels 6, 7

                                                               // B. Transpose ARGB (AOS) -> Planar (SOA)
                                                               // ------------------------------------------------------------

                                                               // Unpack Low/High (Interleave pairs)
                                                               // t0: A0 A2 R0 R2 | A1 A3 R1 R3 (Lane 0 / Lane 1)
                                                               // t1: G0 G2 B0 B2 | G1 G3 B1 B3
            __m256 t0 = _mm256_unpacklo_ps(r0, r1);
            __m256 t1 = _mm256_unpackhi_ps(r0, r1);
            __m256 t2 = _mm256_unpacklo_ps(r2, r3);
            __m256 t3 = _mm256_unpackhi_ps(r2, r3);

            // C. Extract Channels (Adjusted for ARGB)
            // ------------------------------------------------------------
            // We need R (Slot 1), G (Slot 2), B (Slot 3). 
            // Slot 0 (Alpha) is in t0/t2 low parts, we ignore it.

            // R is Slot 1. Located in High 32-bits of t0/t2 pairs.
            // t0 contains [A, A, R, R...]. We want indices 2 and 3.
            // _MM_SHUFFLE(3, 2, 3, 2) selects the High 64-bits of each 128-bit block.
            __m256 vR_mix = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));

            // G is Slot 2. Located in Low 32-bits of t1/t3 pairs.
            // t1 contains [G, G, B, B...]. We want indices 0 and 1.
            // _MM_SHUFFLE(1, 0, 1, 0) selects the Low 64-bits.
            __m256 vG_mix = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));

            // B is Slot 3. Located in High 32-bits of t1/t3 pairs.
            // t1 contains [G, G, B, B...]. We want indices 2 and 3.
            __m256 vB_mix = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));

            // D. Fix Order (Permute)
            // ------------------------------------------------------------
            // Reorders [0, 2, 4, 6, 1, 3, 5, 7] -> [0, 1, 2, 3, 4, 5, 6, 7]
            __m256 vR = _mm256_permutevar8x32_ps(vR_mix, vPermMask);
            __m256 vG = _mm256_permutevar8x32_ps(vG_mix, vPermMask);
            __m256 vB = _mm256_permutevar8x32_ps(vB_mix, vPermMask);

            // E. Compute Y (Planar)
            // ------------------------------------------------------------
            // Y = R*m0 + G*m1 + B*m2
            __m256 vY = _mm256_fmadd_ps(vR, m0, _mm256_fmadd_ps(vG, m1, _mm256_mul_ps(vB, m2)));
            _mm256_storeu_ps(pDstY + i, vY);

            // F. Compute U and V
            // ------------------------------------------------------------
            __m256 vU = _mm256_fmadd_ps(vR, m3, _mm256_fmadd_ps(vG, m4, _mm256_mul_ps(vB, m5)));
            __m256 vV = _mm256_fmadd_ps(vR, m6, _mm256_fmadd_ps(vG, m7, _mm256_mul_ps(vB, m8)));

            // G. Interleave UV
            // ------------------------------------------------------------
            // We want [U0 V0 U1 V1 ... ]
            __m256 vUV_a = _mm256_unpacklo_ps(vU, vV); // Mix Lows
            __m256 vUV_b = _mm256_unpackhi_ps(vU, vV); // Mix Highs

                                                       // Linearize lanes
            __m256 vOut1 = _mm256_permute2f128_ps(vUV_a, vUV_b, 0x20);
            __m256 vOut2 = _mm256_permute2f128_ps(vUV_a, vUV_b, 0x31);

            _mm256_storeu_ps(pDstUV + (i * 2), vOut1);
            _mm256_storeu_ps(pDstUV + (i * 2) + 8, vOut2);
        }

        // 3. Tail Handling (Scalar)
        // ------------------------------------------------------------
        for (; i < width; ++i)
        {
            const float* px = pSrcRow + i * 4;

            // ARGB Layout: [0]=A, [1]=R, [2]=G, [3]=B
            float r = px[1];
            float g = px[2];
            float b = px[3];

            pDstY[i] = r * baseM[0] + g * baseM[1] + b * baseM[2];

            float u = r * baseM[3] + g * baseM[4] + b * baseM[5];
            float v = r * baseM[6] + g * baseM[7] + b * baseM[8];

            pDstUV[i * 2] = u;
            pDstUV[i * 2 + 1] = v;
        }
    }
}


void Rgb2YuvSemiplanar_BGRA_16u_AVX2
(
    const PF_Pixel_BGRA_16u* RESTRICT in,
    float* RESTRICT dstY,
    float* RESTRICT dstUV,
    int32_t width,
    int32_t height,
    int32_t linePitch
) noexcept
{
    // 1. Constants
    const float normScale = 1.0f / 32768.0f;
    const float* baseM = RGB2YUV[1];

    const __m256 m0 = _mm256_set1_ps(baseM[0] * normScale);
    const __m256 m1 = _mm256_set1_ps(baseM[1] * normScale);
    const __m256 m2 = _mm256_set1_ps(baseM[2] * normScale);

    const __m256 m3 = _mm256_set1_ps(baseM[3] * normScale);
    const __m256 m4 = _mm256_set1_ps(baseM[4] * normScale);
    const __m256 m5 = _mm256_set1_ps(baseM[5] * normScale);

    const __m256 m6 = _mm256_set1_ps(baseM[6] * normScale);
    const __m256 m7 = _mm256_set1_ps(baseM[7] * normScale);
    const __m256 m8 = _mm256_set1_ps(baseM[8] * normScale);

    const __m256i vPermMask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (int32_t j = 0; j < height; j++)
    {
        const uint16_t* RESTRICT pSrcRow = reinterpret_cast<const uint16_t*>(in + j * linePitch);
        float* RESTRICT pDstY = dstY + j * width;
        float* RESTRICT pDstUV = dstUV + j * width * 2;

        int32_t i = 0;
        const int32_t vecWidth = width - 7;

        for (; i < vecWidth; i += 8)
        {
            // Load 64 bytes (8 pixels of 4x uint16)
            __m256i vRaw03 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrcRow + i * 4));
            __m256i vRaw47 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrcRow + i * 4 + 16));

            // [FIXED] 2-Step Conversion: uint16 -> int32 -> float
            // -----------------------------------------------------------------------
            // r0: Pixels 0,1 (Low 128 of vRaw03)
            __m256 r0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(vRaw03)));

            // r1: Pixels 2,3 (High 128 of vRaw03)
            __m256 r1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(vRaw03, 1)));

            // r2: Pixels 4,5 (Low 128 of vRaw47)
            __m256 r2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(vRaw47)));

            // r3: Pixels 6,7 (High 128 of vRaw47)
            __m256 r3 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(vRaw47, 1)));
            // -----------------------------------------------------------------------

            // Transpose BGRA
            __m256 t0 = _mm256_unpacklo_ps(r0, r1);
            __m256 t1 = _mm256_unpackhi_ps(r0, r1);
            __m256 t2 = _mm256_unpacklo_ps(r2, r3);
            __m256 t3 = _mm256_unpackhi_ps(r2, r3);

            __m256 vB_mix = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 vG_mix = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 vR_mix = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));

            __m256 vB = _mm256_permutevar8x32_ps(vB_mix, vPermMask);
            __m256 vG = _mm256_permutevar8x32_ps(vG_mix, vPermMask);
            __m256 vR = _mm256_permutevar8x32_ps(vR_mix, vPermMask);

            // Math
            __m256 vY = _mm256_fmadd_ps(vR, m0, _mm256_fmadd_ps(vG, m1, _mm256_mul_ps(vB, m2)));
            _mm256_storeu_ps(pDstY + i, vY);

            __m256 vU = _mm256_fmadd_ps(vR, m3, _mm256_fmadd_ps(vG, m4, _mm256_mul_ps(vB, m5)));
            __m256 vV = _mm256_fmadd_ps(vR, m6, _mm256_fmadd_ps(vG, m7, _mm256_mul_ps(vB, m8)));

            __m256 vUV_a = _mm256_unpacklo_ps(vU, vV);
            __m256 vUV_b = _mm256_unpackhi_ps(vU, vV);
            __m256 vOut1 = _mm256_permute2f128_ps(vUV_a, vUV_b, 0x20);
            __m256 vOut2 = _mm256_permute2f128_ps(vUV_a, vUV_b, 0x31);

            _mm256_storeu_ps(pDstUV + (i * 2), vOut1);
            _mm256_storeu_ps(pDstUV + (i * 2) + 8, vOut2);
        }

        // Tail
        for (; i < width; ++i)
        {
            const uint16_t* px = pSrcRow + i * 4;
            float b = static_cast<float>(px[0]) * normScale;
            float g = static_cast<float>(px[1]) * normScale;
            float r = static_cast<float>(px[2]) * normScale;

            pDstY[i] = r * baseM[0] + g * baseM[1] + b * baseM[2];
            float u = r * baseM[3] + g * baseM[4] + b * baseM[5];
            float v = r * baseM[6] + g * baseM[7] + b * baseM[8];
            pDstUV[i * 2] = u;
            pDstUV[i * 2 + 1] = v;
        }
    }
}


inline void Rgb2YuvSemiplanar_ARGB_16u_AVX2
(
    const PF_Pixel_ARGB_16u* RESTRICT in,
    float* RESTRICT dstY,
    float* RESTRICT dstUV,
    int32_t width,
    int32_t height,
    int32_t linePitch
) noexcept
{
    // 1. Constants
    const float normScale = 1.0f / 32768.0f;
    const float* baseM = RGB2YUV[1];

    const __m256 m0 = _mm256_set1_ps(baseM[0] * normScale);
    const __m256 m1 = _mm256_set1_ps(baseM[1] * normScale);
    const __m256 m2 = _mm256_set1_ps(baseM[2] * normScale);

    const __m256 m3 = _mm256_set1_ps(baseM[3] * normScale);
    const __m256 m4 = _mm256_set1_ps(baseM[4] * normScale);
    const __m256 m5 = _mm256_set1_ps(baseM[5] * normScale);

    const __m256 m6 = _mm256_set1_ps(baseM[6] * normScale);
    const __m256 m7 = _mm256_set1_ps(baseM[7] * normScale);
    const __m256 m8 = _mm256_set1_ps(baseM[8] * normScale);

    const __m256i vPermMask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (int32_t j = 0; j < height; j++)
    {
        const uint16_t* RESTRICT pSrcRow = reinterpret_cast<const uint16_t*>(in + j * linePitch);
        float* RESTRICT pDstY = dstY + j * width;
        float* RESTRICT pDstUV = dstUV + j * width * 2;

        int32_t i = 0;
        const int32_t vecWidth = width - 7;

        for (; i < vecWidth; i += 8)
        {
            // Load
            __m256i vRaw03 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrcRow + i * 4));
            __m256i vRaw47 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pSrcRow + i * 4 + 16));

            // [FIXED] 2-Step Conversion: uint16 -> int32 -> float
            __m256 r0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(vRaw03)));
            __m256 r1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(vRaw03, 1)));
            __m256 r2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(vRaw47)));
            __m256 r3 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(vRaw47, 1)));

            // Transpose ARGB
            __m256 t0 = _mm256_unpacklo_ps(r0, r1);
            __m256 t1 = _mm256_unpackhi_ps(r0, r1);
            __m256 t2 = _mm256_unpacklo_ps(r2, r3);
            __m256 t3 = _mm256_unpackhi_ps(r2, r3);

            __m256 vR_mix = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 vG_mix = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 vB_mix = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));

            __m256 vB = _mm256_permutevar8x32_ps(vB_mix, vPermMask);
            __m256 vG = _mm256_permutevar8x32_ps(vG_mix, vPermMask);
            __m256 vR = _mm256_permutevar8x32_ps(vR_mix, vPermMask);

            // Math
            __m256 vY = _mm256_fmadd_ps(vR, m0, _mm256_fmadd_ps(vG, m1, _mm256_mul_ps(vB, m2)));
            _mm256_storeu_ps(pDstY + i, vY);

            __m256 vU = _mm256_fmadd_ps(vR, m3, _mm256_fmadd_ps(vG, m4, _mm256_mul_ps(vB, m5)));
            __m256 vV = _mm256_fmadd_ps(vR, m6, _mm256_fmadd_ps(vG, m7, _mm256_mul_ps(vB, m8)));

            __m256 vUV_a = _mm256_unpacklo_ps(vU, vV);
            __m256 vUV_b = _mm256_unpackhi_ps(vU, vV);
            __m256 vOut1 = _mm256_permute2f128_ps(vUV_a, vUV_b, 0x20);
            __m256 vOut2 = _mm256_permute2f128_ps(vUV_a, vUV_b, 0x31);

            _mm256_storeu_ps(pDstUV + (i * 2), vOut1);
            _mm256_storeu_ps(pDstUV + (i * 2) + 8, vOut2);
        }

        // Tail
        for (; i < width; ++i)
        {
            const uint16_t* px = pSrcRow + i * 4;
            float r = static_cast<float>(px[1]) * normScale;
            float g = static_cast<float>(px[2]) * normScale;
            float b = static_cast<float>(px[3]) * normScale;

            pDstY[i] = r * baseM[0] + g * baseM[1] + b * baseM[2];
            float u = r * baseM[3] + g * baseM[4] + b * baseM[5];
            float v = r * baseM[6] + g * baseM[7] + b * baseM[8];
            pDstUV[i * 2] = u;
            pDstUV[i * 2 + 1] = v;
        }
    }
}