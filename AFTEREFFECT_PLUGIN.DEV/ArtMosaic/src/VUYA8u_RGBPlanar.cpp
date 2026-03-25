#include "MosaicColorConvert.hpp"


void vuya2planar
(
    const PF_Pixel_VUYA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    bool is709
)
{
    float* RESTRICT pR = memHndl.R_planar;
    float* RESTRICT pG = memHndl.G_planar;
    float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256i mask_FF = _mm256_set1_epi32(0xFF);
    const __m256 v_128 = _mm256_set1_ps(128.0f);

    // Pre-load Matrix Coefficients to avoid branching in the loop
    const float c_RV = is709 ?  1.5748021f  :  1.407500f;
    const float c_GU = is709 ? -0.18732698f : -0.344140f;
    const float c_GV = is709 ? -0.4681240f  : -0.716900f;
    const float c_BU = is709 ?  1.85559927f :  1.779000f;

    const __m256 v_cRV = _mm256_set1_ps(c_RV);
    const __m256 v_cGU = _mm256_set1_ps(c_GU);
    const __m256 v_cGV = _mm256_set1_ps(c_GV);
    const __m256 v_cBU = _mm256_set1_ps(c_BU);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_VUYA_8u* pLine = pSrc + j * srcPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Memory layout: V (Byte 0), U (Byte 1), Y (Byte 2), A (Byte 3)
            __m256i v_vuya = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pLine[i]));

            __m256i v_v_int = _mm256_and_si256(v_vuya, mask_FF);
            __m256i v_u_int = _mm256_and_si256(_mm256_srli_epi32(v_vuya, 8), mask_FF);
            __m256i v_y_int = _mm256_and_si256(_mm256_srli_epi32(v_vuya, 16), mask_FF);

            // Convert to floats
            __m256 v_y_f = _mm256_cvtepi32_ps(v_y_int);
            __m256 v_u_f = _mm256_sub_ps(_mm256_cvtepi32_ps(v_u_int), v_128); // U - 128
            __m256 v_v_f = _mm256_sub_ps(_mm256_cvtepi32_ps(v_v_int), v_128); // V - 128

                                                                              // Calculate RGB using Fused Multiply-Add (a * b + c) for maximum speed
            __m256 v_r = _mm256_fmadd_ps(v_cRV, v_v_f, v_y_f); // R = Y + c_RV * V'

            __m256 v_g = _mm256_fmadd_ps(v_cGU, v_u_f, v_y_f); // G = Y + c_GU * U'
            v_g = _mm256_fmadd_ps(v_cGV, v_v_f, v_g);          // G = G + c_GV * V'

            __m256 v_b = _mm256_fmadd_ps(v_cBU, v_u_f, v_y_f); // B = Y + c_BU * U'

                                                               // Store to planar buffers
            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pR[idx], v_r);
            _mm256_storeu_ps(&pG[idx], v_g);
            _mm256_storeu_ps(&pB[idx], v_b);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            float y = static_cast<float>(pLine[i].Y);
            float u = static_cast<float>(pLine[i].U) - 128.0f;
            float v = static_cast<float>(pLine[i].V) - 128.0f;

            pR[j * sizeX + i] = y + c_RV * v;
            pG[j * sizeX + i] = y + c_GU * u + c_GV * v;
            pB[j * sizeX + i] = y + c_BU * u;
        }
    }

    return;
}


void planar2vuya
(
    const PF_Pixel_VUYA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_VUYA_8u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    bool is709
)
{
    const float* RESTRICT pR = memHndl.R_planar;
    const float* RESTRICT pG = memHndl.G_planar;
    const float* RESTRICT pB = memHndl.B_planar;

    const A_long spanX8 = sizeX & ~7;
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    const __m256 v_128 = _mm256_set1_ps(128.0f);
    const __m256i v_alpha_mask = _mm256_set1_epi32(0xFF000000);

    // Pre-load Forward Matrix Coefficients
    float cY_R, cY_G, cY_B, cU_R, cU_G, cU_B, cV_R, cV_G, cV_B;
    if (true == is709)
    {
        cY_R =  0.212600f;  cY_G = 0.715200f;   cY_B =  0.072200f;
        cU_R = -0.114570f;  cU_G = -0.385430f;  cU_B =  0.500000f;
        cV_R =  0.500000f;  cV_G = -0.454150f;  cV_B = -0.045850f;
    }
    else
    {
        cY_R =  0.299000f;  cY_G =  0.587000f;  cY_B =  0.114000f;
        cU_R = -0.168736f;  cU_G = -0.331264f;  cU_B =  0.500000f;
        cV_R =  0.500000f;  cV_G = -0.418688f;  cV_B = -0.081312f;
    }

    const __m256 v_cY_R = _mm256_set1_ps(cY_R), v_cY_G = _mm256_set1_ps(cY_G), v_cY_B = _mm256_set1_ps(cY_B);
    const __m256 v_cU_R = _mm256_set1_ps(cU_R), v_cU_G = _mm256_set1_ps(cU_G), v_cU_B = _mm256_set1_ps(cU_B);
    const __m256 v_cV_R = _mm256_set1_ps(cV_R), v_cV_G = _mm256_set1_ps(cV_G), v_cV_B = _mm256_set1_ps(cV_B);

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_VUYA_8u* pSrcLine = pSrc + j * srcPitch;
              PF_Pixel_VUYA_8u* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;

            __m256 v_r_f = _mm256_loadu_ps(&pR[idx]);
            __m256 v_g_f = _mm256_loadu_ps(&pG[idx]);
            __m256 v_b_f = _mm256_loadu_ps(&pB[idx]);

            // Y = cY_R*R + cY_G*G + cY_B*B
            __m256 v_y_f = _mm256_mul_ps(v_cY_R, v_r_f);
            v_y_f = _mm256_fmadd_ps(v_cY_G, v_g_f, v_y_f);
            v_y_f = _mm256_fmadd_ps(v_cY_B, v_b_f, v_y_f);

            // U = cU_R*R + cU_G*G + cU_B*B + 128.0f
            __m256 v_u_f = _mm256_fmadd_ps(v_cU_R, v_r_f, v_128);
            v_u_f = _mm256_fmadd_ps(v_cU_G, v_g_f, v_u_f);
            v_u_f = _mm256_fmadd_ps(v_cU_B, v_b_f, v_u_f);

            // V = cV_R*R + cV_G*G + cV_B*B + 128.0f
            __m256 v_v_f = _mm256_fmadd_ps(v_cV_R, v_r_f, v_128);
            v_v_f = _mm256_fmadd_ps(v_cV_G, v_g_f, v_v_f);
            v_v_f = _mm256_fmadd_ps(v_cV_B, v_b_f, v_v_f);

            // Clamp safely between [0.0f, 255.0f]
            v_y_f = _mm256_min_ps(_mm256_max_ps(v_y_f, v_zero), v_255);
            v_u_f = _mm256_min_ps(_mm256_max_ps(v_u_f, v_zero), v_255);
            v_v_f = _mm256_min_ps(_mm256_max_ps(v_v_f, v_zero), v_255);

            // Convert to integers
            __m256i v_y_i = _mm256_cvtps_epi32(v_y_f);
            __m256i v_u_i = _mm256_cvtps_epi32(v_u_f);
            __m256i v_v_i = _mm256_cvtps_epi32(v_v_f);

            // Bit-shift into correct VUYA byte positions (V=0, U=8, Y=16)
            v_u_i = _mm256_slli_epi32(v_u_i, 8);
            v_y_i = _mm256_slli_epi32(v_y_i, 16);

            // Extract original Alpha from pSrc
            __m256i v_src_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pSrcLine[i]));
            __m256i v_src_alpha = _mm256_and_si256(v_src_pixels, v_alpha_mask);

            // Pack V, U, Y, and A into a single block
            __m256i v_vuya = _mm256_or_si256(v_v_i, _mm256_or_si256(v_u_i, _mm256_or_si256(v_y_i, v_src_alpha)));

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOutLine[i]), v_vuya);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            float r = pR[idx];
            float g = pG[idx];
            float b = pB[idx];

            float y = cY_R * r + cY_G * g + cY_B * b;
            float u = cU_R * r + cU_G * g + cU_B * b + 128.0f;
            float v = cV_R * r + cV_G * g + cV_B * b + 128.0f;

            y = y < 0.0f ? 0.0f : (y > 255.0f ? 255.0f : y);
            u = u < 0.0f ? 0.0f : (u > 255.0f ? 255.0f : u);
            v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);

            pOutLine[i].V = static_cast<A_u_char>(v);
            pOutLine[i].U = static_cast<A_u_char>(u);
            pOutLine[i].Y = static_cast<A_u_char>(y);
            pOutLine[i].A = pSrcLine[i].A;
        }
    }

    return;
}