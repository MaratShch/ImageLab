#include "MosaicColorConvert.hpp"

void vuya2planar
(
    const PF_Pixel_VUYA_32f* RESTRICT pSrc,
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
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_scale255 = _mm256_set1_ps(255.0f);

    // Pre-load Matrix Coefficients (BT.709 or BT.601)
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
        const PF_Pixel_VUYA_32f* pLine = pSrc + j * srcPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            // Gather 32-bit floats directly. 
            // Note: _mm256_set_ps takes arguments in reverse order (highest to lowest index)
            __m256 v_v = _mm256_set_ps(pLine[i + 7].V, pLine[i + 6].V, pLine[i + 5].V, pLine[i + 4].V, pLine[i + 3].V, pLine[i + 2].V, pLine[i + 1].V, pLine[i].V);
            __m256 v_u = _mm256_set_ps(pLine[i + 7].U, pLine[i + 6].U, pLine[i + 5].U, pLine[i + 4].U, pLine[i + 3].U, pLine[i + 2].U, pLine[i + 1].U, pLine[i].U);
            __m256 v_y = _mm256_set_ps(pLine[i + 7].Y, pLine[i + 6].Y, pLine[i + 5].Y, pLine[i + 4].Y, pLine[i + 3].Y, pLine[i + 2].Y, pLine[i + 1].Y, pLine[i].Y);

            // Remove chroma bias for 32f color space: U' = U - 0.5, V' = V - 0.5
            v_u = _mm256_sub_ps(v_u, v_half);
            v_v = _mm256_sub_ps(v_v, v_half);

            // Reconstruct RGB [0.0f - 1.0f] via FMA
            __m256 v_r = _mm256_fmadd_ps(v_cRV, v_v, v_y);

            __m256 v_g = _mm256_fmadd_ps(v_cGU, v_u, v_y);
            v_g = _mm256_fmadd_ps(v_cGV, v_v, v_g);

            __m256 v_b = _mm256_fmadd_ps(v_cBU, v_u, v_y);

            // Scale up to your planar buffer range [0.0f - 255.0f]
            v_r = _mm256_mul_ps(v_r, v_scale255);
            v_g = _mm256_mul_ps(v_g, v_scale255);
            v_b = _mm256_mul_ps(v_b, v_scale255);

            const A_long idx = j * sizeX + i;
            _mm256_storeu_ps(&pR[idx], v_r);
            _mm256_storeu_ps(&pG[idx], v_g);
            _mm256_storeu_ps(&pB[idx], v_b);
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            float y = pLine[i].Y;
            float u = pLine[i].U - 0.5f;
            float v = pLine[i].V - 0.5f;

            float r = y + c_RV * v;
            float g = y + c_GU * u + c_GV * v;
            float b = y + c_BU * u;

            pR[j * sizeX + i] = r * 255.0f;
            pG[j * sizeX + i] = g * 255.0f;
            pB[j * sizeX + i] = b * 255.0f;
        }
    }

    return;
}


void planar2vuya
(
    const PF_Pixel_VUYA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_VUYA_32f* RESTRICT pDst,
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
    const __m256 v_half = _mm256_set1_ps(0.5f);

    constexpr float scaleDown = 1.0f / 255.0f;
    const __m256 v_scaleDown = _mm256_set1_ps(scaleDown);

    // Clamp limits: enforce 1.0f - FLT_EPSILON per your safety rule
    const float maxClampVal = 1.0f - std::numeric_limits<float>::epsilon();
    const __m256 v_max = _mm256_set1_ps(maxClampVal);

    // Pre-load Forward Matrix Coefficients
    float cY_R, cY_G, cY_B, cU_R, cU_G, cU_B, cV_R, cV_G, cV_B;
    if (true == is709)
    {
        cY_R =  0.212600f;  cY_G =  0.715200f;  cY_B =  0.072200f;
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

    // Aligned temp arrays for scatter
    CACHE_ALIGN float y_out[8];
    CACHE_ALIGN float u_out[8];
    CACHE_ALIGN float v_out[8];

    for (A_long j = 0; j < sizeY; j++)
    {
        const PF_Pixel_VUYA_32f* pSrcLine = pSrc + j * srcPitch;
        PF_Pixel_VUYA_32f* pOutLine = pDst + j * dstPitch;
        A_long i = 0;

        // --- AVX2 FAST PATH ---
        for (; i < spanX8; i += 8)
        {
            const A_long idx = j * sizeX + i;

            // Load planar and scale down from [0.0 - 255.0] to [0.0 - 1.0]
            __m256 v_r_f = _mm256_mul_ps(_mm256_loadu_ps(&pR[idx]), v_scaleDown);
            __m256 v_g_f = _mm256_mul_ps(_mm256_loadu_ps(&pG[idx]), v_scaleDown);
            __m256 v_b_f = _mm256_mul_ps(_mm256_loadu_ps(&pB[idx]), v_scaleDown);

            // Y = cY_R*R + cY_G*G + cY_B*B
            __m256 v_y_f = _mm256_mul_ps(v_cY_R, v_r_f);
            v_y_f = _mm256_fmadd_ps(v_cY_G, v_g_f, v_y_f);
            v_y_f = _mm256_fmadd_ps(v_cY_B, v_b_f, v_y_f);

            // U = cU_R*R + cU_G*G + cU_B*B + 0.5f (Bias)
            __m256 v_u_f = _mm256_fmadd_ps(v_cU_R, v_r_f, v_half);
            v_u_f = _mm256_fmadd_ps(v_cU_G, v_g_f, v_u_f);
            v_u_f = _mm256_fmadd_ps(v_cU_B, v_b_f, v_u_f);

            // V = cV_R*R + cV_G*G + cV_B*B + 0.5f (Bias)
            __m256 v_v_f = _mm256_fmadd_ps(v_cV_R, v_r_f, v_half);
            v_v_f = _mm256_fmadd_ps(v_cV_G, v_g_f, v_v_f);
            v_v_f = _mm256_fmadd_ps(v_cV_B, v_b_f, v_v_f);

            // Clamp safely between 0.0f and (1.0f - FLT_EPSILON)
            v_y_f = _mm256_min_ps(_mm256_max_ps(v_y_f, v_zero), v_max);
            v_u_f = _mm256_min_ps(_mm256_max_ps(v_u_f, v_zero), v_max);
            v_v_f = _mm256_min_ps(_mm256_max_ps(v_v_f, v_zero), v_max);

            // Store locally
            _mm256_store_ps(y_out, v_y_f);
            _mm256_store_ps(u_out, v_u_f);
            _mm256_store_ps(v_out, v_v_f);

            // Pack struct and copy original Alpha
            for (int k = 0; k < 8; ++k)
            {
                pOutLine[i + k].V = v_out[k];
                pOutLine[i + k].U = u_out[k];
                pOutLine[i + k].Y = y_out[k];
                pOutLine[i + k].A = pSrcLine[i + k].A;
            }
        }

        // --- SCALAR TAIL ---
        for (; i < sizeX; i++)
        {
            const A_long idx = j * sizeX + i;
            float r = pR[idx] * scaleDown;
            float g = pG[idx] * scaleDown;
            float b = pB[idx] * scaleDown;

            float y = cY_R * r + cY_G * g + cY_B * b;
            float u = cU_R * r + cU_G * g + cU_B * b + 0.5f;
            float v = cV_R * r + cV_G * g + cV_B * b + 0.5f;

            y = y < 0.0f ? 0.0f : (y > maxClampVal ? maxClampVal : y);
            u = u < 0.0f ? 0.0f : (u > maxClampVal ? maxClampVal : u);
            v = v < 0.0f ? 0.0f : (v > maxClampVal ? maxClampVal : v);

            pOutLine[i].V = v;
            pOutLine[i].U = u;
            pOutLine[i].Y = y;
            pOutLine[i].A = pSrcLine[i].A;
        }
    }

    return;
}