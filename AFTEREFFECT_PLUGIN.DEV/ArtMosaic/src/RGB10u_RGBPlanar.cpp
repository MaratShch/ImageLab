#include "MosaicColorConvert.hpp"

void rgb2planar
(
    const PF_Pixel_RGB_10u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch
)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;

    // Scale down from [0, 1023] to [0.0f, 255.0f]
    constexpr float scaleDown = 255.0f / 1023.0f;
    const __m256 v_scale = _mm256_set1_ps(scaleDown);

    // Mask to extract exactly 10 bits (0x3FF = 1023)
    const __m256i v_mask_10bit = _mm256_set1_epi32(0x3FF);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_RGB_10u* pLine = pSrc + j * srcPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Load 8 packed 32-bit pixels (32 bytes total)
            __m256i v_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pLine[i]));

            // Shift and mask to isolate the 10-bit channels
            __m256i v_b_int = _mm256_and_si256(_mm256_srli_epi32(v_raw, 2), v_mask_10bit);
            __m256i v_g_int = _mm256_and_si256(_mm256_srli_epi32(v_raw, 12), v_mask_10bit);
            __m256i v_r_int = _mm256_and_si256(_mm256_srli_epi32(v_raw, 22), v_mask_10bit);

            // Convert to floats
            __m256 v_b_f = _mm256_cvtepi32_ps(v_b_int);
            __m256 v_g_f = _mm256_cvtepi32_ps(v_g_int);
            __m256 v_r_f = _mm256_cvtepi32_ps(v_r_int);

            // Scale to [0.0f, 255.0f]
            v_b_f = _mm256_mul_ps(v_b_f, v_scale);
            v_g_f = _mm256_mul_ps(v_g_f, v_scale);
            v_r_f = _mm256_mul_ps(v_r_f, v_scale);

            // Store to planar buffers
            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pB[idx], v_b_f);
            _mm256_storeu_ps(&pG[idx], v_g_f);
            _mm256_storeu_ps(&pR[idx], v_r_f);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;

            // The C++ compiler handles the bitfield extraction here natively
            pR[idx] = static_cast<float>(pLine[i].R) * scaleDown;
            pG[idx] = static_cast<float>(pLine[i].G) * scaleDown;
            pB[idx] = static_cast<float>(pLine[i].B) * scaleDown;
        }
    }

    return;
}


void planar2rgb
(
    const MemHandler& memHndl,
    PF_Pixel_RGB_10u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long dstPitch
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;

    // Scale up from [0.0f, 255.0f] back to [0, 1023]
    constexpr float scaleUp = 1023.0f / 255.0f;
    const __m256 v_scale = _mm256_set1_ps(scaleUp);

    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(1023.0f);

    for (A_long j = 0; j < sizeY; j++)
    {
        PF_Pixel_RGB_10u* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;

            // Load planar floats
            __m256 v_r_f = _mm256_loadu_ps(&pR[idx]);
            __m256 v_g_f = _mm256_loadu_ps(&pG[idx]);
            __m256 v_b_f = _mm256_loadu_ps(&pB[idx]);

            // Scale up to [0 - 1023]
            v_r_f = _mm256_mul_ps(v_r_f, v_scale);
            v_g_f = _mm256_mul_ps(v_g_f, v_scale);
            v_b_f = _mm256_mul_ps(v_b_f, v_scale);

            // Clamp to prevent integer overflow past 1023
            v_r_f = _mm256_min_ps(_mm256_max_ps(v_r_f, v_zero), v_max);
            v_g_f = _mm256_min_ps(_mm256_max_ps(v_g_f, v_zero), v_max);
            v_b_f = _mm256_min_ps(_mm256_max_ps(v_b_f, v_zero), v_max);

            // Convert to 32-bit integers
            __m256i v_r_i = _mm256_cvtps_epi32(v_r_f);
            __m256i v_g_i = _mm256_cvtps_epi32(v_g_f);
            __m256i v_b_i = _mm256_cvtps_epi32(v_b_f);

            // Bit-shift back into correct struct alignment positions
            // B << 2, G << 12, R << 22
            __m256i v_b_shifted = _mm256_slli_epi32(v_b_i, 2);
            __m256i v_g_shifted = _mm256_slli_epi32(v_g_i, 12);
            __m256i v_r_shifted = _mm256_slli_epi32(v_r_i, 22);

            // Pack into a single 32-bit block. (Padding bits 0-1 will naturally be 0)
            __m256i v_packed = _mm256_or_si256(v_b_shifted,
                _mm256_or_si256(v_g_shifted, v_r_shifted));

            // Store exactly 8 pixels (32 bytes) safely to memory
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_packed);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;

            float r = pR[idx] * scaleUp;
            float g = pG[idx] * scaleUp;
            float b = pB[idx] * scaleUp;

                                           // Safe scalar clamping
            r = r < 0.0f ? 0.0f : (r > 1023.0f ? 1023.0f : r);
            g = g < 0.0f ? 0.0f : (g > 1023.0f ? 1023.0f : g);
            b = b < 0.0f ? 0.0f : (b > 1023.0f ? 1023.0f : b);

            // The C++ compiler naturally packs these back into the bitfields
            pOutLine[i]._pad_ = 0;
            pOutLine[i].R = static_cast<A_u_long>(r);
            pOutLine[i].G = static_cast<A_u_long>(g);
            pOutLine[i].B = static_cast<A_u_long>(b);
        }
    }

    return;
}