#include "FastAriphmetics.hpp"
#include "PaintAlgoAVX2.hpp"

void compute_initial_tensors_fused_AVX2
(
    const float* RESTRICT im,
    float* RESTRICT A,
    float* RESTRICT B,
    float* RESTRICT C,
    const A_long width,
    const A_long height
) noexcept
{
    const __m256 v_half = _mm256_set1_ps(0.5f);

    // Process the internal grid (y from 1 to height-2)
    for (A_long y = 1; y < height - 1; y++)
    {
        const float* RESTRICT pSrcLine0 = im + (y - 1) * width;
        const float* RESTRICT pSrcLine1 = im + y * width;
        const float* RESTRICT pSrcLine2 = im + (y + 1) * width;

        float* RESTRICT a_line = A + y * width;
        float* RESTRICT b_line = B + y * width;
        float* RESTRICT c_line = C + y * width;

        A_long x = 1;

        // --- AVX2 INNER KERNEL ---
        for (; x <= width - 9; x += 8)
        {
            // Horizontal gradient: 0.5 * (Src[x+1] - Src[x-1])
            __m256 v_src_right = _mm256_loadu_ps(&pSrcLine1[x + 1]);
            __m256 v_src_left  = _mm256_loadu_ps(&pSrcLine1[x - 1]);
            __m256 v_gx = _mm256_mul_ps(_mm256_sub_ps(v_src_right, v_src_left), v_half);

            // Vertical gradient: 0.5 * (SrcLine2[x] - SrcLine0[x])
            __m256 v_src_down = _mm256_loadu_ps(&pSrcLine2[x]);
            __m256 v_src_up   = _mm256_loadu_ps(&pSrcLine0[x]);
            __m256 v_gy = _mm256_mul_ps(_mm256_sub_ps(v_src_down, v_src_up), v_half);

            // Calculate Tensors: A = gx*gx, B = gy*gy, C = gx*gy
            __m256 v_a = _mm256_mul_ps(v_gx, v_gx);
            __m256 v_b = _mm256_mul_ps(v_gy, v_gy);
            __m256 v_c = _mm256_mul_ps(v_gx, v_gy);

            // Store directly to A, B, C
            _mm256_storeu_ps(&a_line[x], v_a);
            _mm256_storeu_ps(&b_line[x], v_b);
            _mm256_storeu_ps(&c_line[x], v_c);
        }

        // --- SCALAR TAIL (and right border) ---
        for (; x < width - 1; x++)
        {
            float gx = 0.5f * (pSrcLine1[x + 1] - pSrcLine1[x - 1]);
            float gy = 0.5f * (pSrcLine2[x] - pSrcLine0[x]);
            
            a_line[x] = gx * gx;
            b_line[x] = gy * gy;
            c_line[x] = gx * gy;
        }

        // Left border (x = 0)
        float gx_left = pSrcLine1[1] - pSrcLine1[0];
        float gy_left = 0.5f * (pSrcLine2[0] - pSrcLine0[0]);
        a_line[0] = gx_left * gx_left;
        b_line[0] = gy_left * gy_left;
        c_line[0] = gx_left * gy_left;

        // Right border (x = width - 1)
        float gx_right = pSrcLine1[width - 1] - pSrcLine1[width - 2];
        float gy_right = 0.5f * (pSrcLine2[width - 1] - pSrcLine0[width - 1]);
        a_line[width - 1] = gx_right * gx_right;
        b_line[width - 1] = gy_right * gy_right;
        c_line[width - 1] = gx_right * gy_right;
    }

    // --- TOP AND BOTTOM ROW SCALAR HANDLING ---
    // Top Row (y = 0)
    for (A_long x = 0; x < width; x++)
    {
        float gx = (x == 0) ? (im[1] - im[0]) : ((x == width - 1) ? (im[x] - im[x - 1]) : (0.5f * (im[x + 1] - im[x - 1])));
        float gy = im[width + x] - im[x];
        A[x] = gx * gx; B[x] = gy * gy; C[x] = gx * gy;
    }

    // Bottom Row (y = height - 1)
    A_long lastRow = (height - 1) * width;
    A_long prevRow = (height - 2) * width;
    for (A_long x = 0; x < width; x++)
    {
        float gx = (x == 0) ? (im[lastRow + 1] - im[lastRow]) : ((x == width - 1) ? (im[lastRow + x] - im[lastRow + x - 1]) : (0.5f * (im[lastRow + x + 1] - im[lastRow + x - 1])));
        float gy = im[lastRow + x] - im[prevRow + x];
        A[lastRow + x] = gx * gx; B[lastRow + x] = gy * gy; C[lastRow + x] = gx * gy;
    }
}


void linear_gradient_gray_AVX2
(
    const float* RESTRICT im,
    float* RESTRICT pGx,
    float* RESTRICT pGy,
    const A_long sizeX,
    const A_long sizeY
) noexcept
{
    const A_long shortSizeX = sizeX - 1;
    const A_long shortSizeY = sizeY - 1;
    const A_long pitch = sizeX; // Assuming densely packed planar buffer
    
    const __m256 v_half = _mm256_set1_ps(0.5f);

    // --- X GRADIENT ---
    for (A_long j = 0; j < sizeY; j++)
    {
        const float* RESTRICT pSrc = im + j * pitch;
              float* RESTRICT gx   = pGx + j * sizeX;

        gx[0] = pSrc[1] - pSrc[0]; // First pixel

        A_long i = 1;
        const A_long vectorEndX = shortSizeX - ((shortSizeX - 1) % 8);

        for (; i < vectorEndX; i += 8)
        {
            __m256 v_next = _mm256_loadu_ps(&pSrc[i + 1]);
            __m256 v_prev = _mm256_loadu_ps(&pSrc[i - 1]);
            _mm256_storeu_ps(&gx[i], _mm256_mul_ps(v_half, _mm256_sub_ps(v_next, v_prev)));
        }

        for (; i < shortSizeX; i++) { gx[i] = 0.5f * (pSrc[i + 1] - pSrc[i - 1]); }
        gx[shortSizeX] = pSrc[shortSizeX] - pSrc[shortSizeX - 1]; // Last pixel
    }

    // --- Y GRADIENT ---
    // First line
    {
        const float* RESTRICT pSrcLine0 = im;
        const float* RESTRICT pSrcLine1 = im + pitch;
        float* RESTRICT gy = pGy;

        A_long i = 0;
        const A_long vectorEndX = sizeX - (sizeX % 8);
        for (; i < vectorEndX; i += 8) {
            _mm256_storeu_ps(&gy[i], _mm256_sub_ps(_mm256_loadu_ps(&pSrcLine1[i]), _mm256_loadu_ps(&pSrcLine0[i])));
        }
        for (; i < sizeX; i++) { gy[i] = pSrcLine1[i] - pSrcLine0[i]; }
    }

    // Inner lines
    for (A_long j = 1; j < shortSizeY; j++)
    {
        const float* RESTRICT pSrcLine0 = im + (j - 1) * pitch;
        const float* RESTRICT pSrcLine2 = im + (j + 1) * pitch;
        float* RESTRICT gy = pGy + j * sizeX;

        A_long i = 0;
        const A_long vectorEndX = sizeX - (sizeX % 8);

        for (; i < vectorEndX; i += 8)
        {
            __m256 v_s2 = _mm256_loadu_ps(&pSrcLine2[i]);
            __m256 v_s0 = _mm256_loadu_ps(&pSrcLine0[i]);
            _mm256_storeu_ps(&gy[i], _mm256_mul_ps(v_half, _mm256_sub_ps(v_s2, v_s0)));
        }
        for (; i < sizeX; i++) { gy[i] = 0.5f * (pSrcLine2[i] - pSrcLine0[i]); }
    }

    // Last line
    {
        const float* RESTRICT pSrcLine0 = im + (shortSizeY - 1) * pitch;
        const float* RESTRICT pSrcLine1 = im + shortSizeY * pitch;
        float* RESTRICT gy = pGy + shortSizeY * sizeX;

        A_long i = 0;
        const A_long vectorEndX = sizeX - (sizeX % 8);
        for (; i < vectorEndX; i += 8) {
            _mm256_storeu_ps(&gy[i], _mm256_sub_ps(_mm256_loadu_ps(&pSrcLine1[i]), _mm256_loadu_ps(&pSrcLine0[i])));
        }
        for (; i < sizeX; i++) { gy[i] = pSrcLine1[i] - pSrcLine0[i]; }
    }
}

void structure_tensors0_AVX2
(
    const float* RESTRICT gx,
    const float* RESTRICT gy,
    const A_long sizeX, 
    const A_long sizeY,
    float* RESTRICT a,
    float* RESTRICT b,
    float* RESTRICT c
) noexcept
{
    const A_long totalPixels = sizeX * sizeY;
    const A_long vectorEnd = totalPixels - (totalPixels % 8);
    A_long i = 0;
    
    for (; i < vectorEnd; i += 8)
    {
        __m256 v_gx = _mm256_loadu_ps(&gx[i]);
        __m256 v_gy = _mm256_loadu_ps(&gy[i]);

        _mm256_storeu_ps(&a[i], _mm256_mul_ps(v_gx, v_gx));
        _mm256_storeu_ps(&b[i], _mm256_mul_ps(v_gy, v_gy));
        _mm256_storeu_ps(&c[i], _mm256_mul_ps(v_gx, v_gy));
    }

    for (; i < totalPixels; ++i)
    {
        const float gx_i = gx[i];
        const float gy_i = gy[i];
        a[i] = gx_i * gx_i;
        b[i] = gy_i * gy_i;
        c[i] = gx_i * gy_i;
    }
}


// We keep the scalar kernel generator since it only runs once per frame for ~21 values
void generate_gaussian_kernel(float* kernel, const A_long radius, const float sigma) noexcept
{
    for (A_long i = -radius; i <= radius; i++)
    {
        kernel[i + radius] = FastCompute::Exp(-(FastCompute::Pow(static_cast<float>(i) / sigma, 2.f) / 2.f));
    }
}

void convolution_AVX2
(
    const float* RESTRICT imIn,
    const float* RESTRICT kernel,
    float* RESTRICT imOutX, // Temporary horizontal buffer (tmpBlur)
    float* RESTRICT imOut,  // Final smoothed output
    const A_long sizeX,
    const A_long sizeY,
    const A_long radius
) noexcept
{
    // Pre-calculate the total sum of the kernel for the "safe" inner core to perfectly match scalar
    float totalSumK_scalar = 0.f;
    for (A_long i = -radius; i <= radius; i++) {
        totalSumK_scalar += kernel[i + radius];
    }
    const __m256 v_totalSumK = _mm256_set1_ps(totalSumK_scalar);

    // ==========================================
    // 1. HORIZONTAL CONVOLUTION (X-Axis)
    // ==========================================
    for (A_long y = 0; y < sizeY; y++)
    {
        const A_long rowOffset = y * sizeX;
        
        // Left Border (Scalar)
        for (A_long x = 0; x < radius; x++)
        {
            float sumV = 0.f, sumK = 0.f;
            for (A_long i = -radius; i <= radius; i++) {
                if (x + i < 0) continue;
                const float valK = kernel[i + radius];
                sumV += imIn[rowOffset + x + i] * valK;
                sumK += valK;
            }
            imOutX[rowOffset + x] = sumV / sumK;
        }

        // Inner Core (AVX2 - 8 pixels per cycle)
        A_long x = radius;
        const A_long safeEndX = (sizeX - radius);
        const A_long vectorEndX = safeEndX - ((safeEndX - radius) % 8);

        for (; x < vectorEndX; x += 8)
        {
            __m256 v_sumV = _mm256_setzero_ps();
            
            for (A_long i = -radius; i <= radius; i++)
            {
                __m256 v_k = _mm256_set1_ps(kernel[i + radius]);
                __m256 v_px = _mm256_loadu_ps(&imIn[rowOffset + x + i]);
                // Exact scalar parity: multiply then add
                v_sumV = _mm256_add_ps(v_sumV, _mm256_mul_ps(v_px, v_k));
            }
            _mm256_storeu_ps(&imOutX[rowOffset + x], _mm256_div_ps(v_sumV, v_totalSumK));
        }

        // Right Border (Scalar)
        for (; x < sizeX; x++)
        {
            float sumV = 0.f, sumK = 0.f;
            for (A_long i = -radius; i <= radius; i++) {
                if (x + i > sizeX - 1) continue;
                const float valK = kernel[i + radius];
                sumV += imIn[rowOffset + x + i] * valK;
                sumK += valK;
            }
            imOutX[rowOffset + x] = sumV / sumK;
        }
    }

    // ==========================================
    // 2. VERTICAL CONVOLUTION (Y-Axis)
    // ==========================================
    // Top Border (Scalar)
    for (A_long y = 0; y < radius; y++)
    {
        for (A_long x = 0; x < sizeX; x++)
        {
            float sumV = 0.f, sumK = 0.f;
            for (A_long i = -radius; i <= radius; i++) {
                if (y + i < 0) continue;
                const float valK = kernel[i + radius];
                sumV += imOutX[(y + i) * sizeX + x] * valK;
                sumK += valK;
            }
            imOut[y * sizeX + x] = sumV / sumK;
        }
    }

    // Inner Core (AVX2 - We can vectorize across the ENTIRE row width here!)
    const A_long safeEndY = (sizeY - radius);
    for (A_long y = radius; y < safeEndY; y++)
    {
        A_long x = 0;
        const A_long vectorEndX = sizeX - (sizeX % 8);

        for (; x < vectorEndX; x += 8)
        {
            __m256 v_sumV = _mm256_setzero_ps();
            
            for (A_long i = -radius; i <= radius; i++)
            {
                __m256 v_k = _mm256_set1_ps(kernel[i + radius]);
                __m256 v_px = _mm256_loadu_ps(&imOutX[(y + i) * sizeX + x]);
                v_sumV = _mm256_add_ps(v_sumV, _mm256_mul_ps(v_px, v_k));
            }
            _mm256_storeu_ps(&imOut[y * sizeX + x], _mm256_div_ps(v_sumV, v_totalSumK));
        }

        // Row Tail (Scalar)
        for (; x < sizeX; x++)
        {
            float sumV = 0.f;
            for (A_long i = -radius; i <= radius; i++) {
                sumV += imOutX[(y + i) * sizeX + x] * kernel[i + radius];
            }
            imOut[y * sizeX + x] = sumV / totalSumK_scalar;
        }
    }

    // Bottom Border (Scalar)
    for (A_long y = safeEndY; y < sizeY; y++)
    {
        for (A_long x = 0; x < sizeX; x++)
        {
            float sumV = 0.f, sumK = 0.f;
            for (A_long i = -radius; i <= radius; i++) {
                if (y + i > sizeY - 1) continue;
                const float valK = kernel[i + radius];
                sumV += imOutX[(y + i) * sizeX + x] * valK;
                sumK += valK;
            }
            imOut[y * sizeX + x] = sumV / sumK;
        }
    }
}

void smooth_structure_tensors_AVX2
(
    const float* RESTRICT A,
    const float* RESTRICT B,
    const float* RESTRICT C,
    const float sigma, 
    const A_long sizeX,
    const A_long sizeY,
    float* RESTRICT A_reg,
    float* RESTRICT B_reg,
    float* RESTRICT C_reg,
    float* RESTRICT tmpBlur // Passed from MemHandler
) noexcept
{
    CACHE_ALIGN float kernel[128];

    const A_long radius = static_cast<A_long>(std::ceil(2.f * sigma));
    const A_long kernelSize = 2 * radius + 1;
    
    // Tiny stack array for the 1D kernel (Max radius is small, e.g. 10-20)
    generate_gaussian_kernel(kernel, radius, sigma);

    convolution_AVX2(A, kernel, tmpBlur, A_reg, sizeX, sizeY, radius);
    convolution_AVX2(B, kernel, tmpBlur, B_reg, sizeX, sizeY, radius);
    convolution_AVX2(C, kernel, tmpBlur, C_reg, sizeX, sizeY, radius);
}


void diagonalize_structure_tensors_AVX2
(
    const float* RESTRICT A,
    const float* RESTRICT B,
    const float* RESTRICT C,
    const A_long sizeX,
    const A_long sizeY,
    float* RESTRICT Lambda1,
    float* RESTRICT Lambda2,
    float* RESTRICT Eigvect2_x,
    float* RESTRICT Eigvect2_y
) noexcept
{
    const A_long totalPixels = sizeX * sizeY;
    const A_long vectorEnd = totalPixels - (totalPixels % 8);
    
    // Pre-load constants into AVX2 registers
    const __m256 v_half  = _mm256_set1_ps(0.5f);
    const __m256 v_two   = _mm256_set1_ps(2.0f);
    const __m256 v_four  = _mm256_set1_ps(4.0f);
    const __m256 v_zero  = _mm256_setzero_ps();

    A_long i = 0;

    // --- AVX2 MAIN LOOP (8 pixels per cycle) ---
    for (; i < vectorEnd; i += 8)
    {
        // 1. Load Smoothed Tensors A, B, C
        __m256 v_a = _mm256_loadu_ps(&A[i]);
        __m256 v_b = _mm256_loadu_ps(&B[i]);
        __m256 v_c = _mm256_loadu_ps(&C[i]);

        // 2. Compute trace = a + b
        __m256 v_trace = _mm256_add_ps(v_a, v_b);

        // 3. Compute delta = (a - b)^2 + 4 * c^2
        __m256 v_a_minus_b = _mm256_sub_ps(v_a, v_b);
        __m256 v_a_minus_b_sq = _mm256_mul_ps(v_a_minus_b, v_a_minus_b);
        __m256 v_c_sq = _mm256_mul_ps(v_c, v_c);
        __m256 v_four_c_sq = _mm256_mul_ps(v_four, v_c_sq);
        __m256 v_delta = _mm256_add_ps(v_a_minus_b_sq, v_four_c_sq);

        // 4. Compute square root of delta
        __m256 v_sqrtDelta = _mm256_sqrt_ps(v_delta);

        // 5. Compute Lambda 1 and Lambda 2
        __m256 v_L1 = _mm256_mul_ps(v_half, _mm256_add_ps(v_trace, v_sqrtDelta));
        __m256 v_L2 = _mm256_mul_ps(v_half, _mm256_sub_ps(v_trace, v_sqrtDelta));

        // 6. Compute Eigenvectors: x1 = 2*c, x2 = b - a - sqrtDelta
        __m256 v_x1 = _mm256_mul_ps(v_two, v_c);
        __m256 v_x2 = _mm256_sub_ps(_mm256_sub_ps(v_b, v_a), v_sqrtDelta);

        // 7. Compute Norm = sqrt(x1^2 + x2^2)
        __m256 v_x1_sq = _mm256_mul_ps(v_x1, v_x1);
        __m256 v_x2_sq = _mm256_mul_ps(v_x2, v_x2);
        __m256 v_norm_sq = _mm256_add_ps(v_x1_sq, v_x2_sq);
        __m256 v_norm = _mm256_sqrt_ps(v_norm_sq);

        // 8. Branchless Condition: if (norm > 0)
        // Create a mask where (norm > 0) is TRUE (all 1s)
        __m256 v_mask = _mm256_cmp_ps(v_norm, v_zero, _CMP_GT_OQ);

        // Calculate divisions (it's okay if norm is 0 here, the mask will discard the NaNs)
        __m256 v_eig_x_div = _mm256_div_ps(v_x1, v_norm);
        __m256 v_eig_y_div = _mm256_div_ps(v_x2, v_norm);

        // Blend: If mask is true, take division result. If false, take 0.0f
        __m256 v_final_eig_x = _mm256_blendv_ps(v_zero, v_eig_x_div, v_mask);
        __m256 v_final_eig_y = _mm256_blendv_ps(v_zero, v_eig_y_div, v_mask);

        // 9. Store all results
        _mm256_storeu_ps(&Lambda1[i], v_L1);
        _mm256_storeu_ps(&Lambda2[i], v_L2);
        _mm256_storeu_ps(&Eigvect2_x[i], v_final_eig_x);
        _mm256_storeu_ps(&Eigvect2_y[i], v_final_eig_y);
    }

    // --- SCALAR TAIL (Process the remaining pixels) ---
    for (; i < totalPixels; ++i)
    {
        const float a_i = A[i];
        const float b_i = B[i];
        const float c_i = C[i];

        const float delta = (a_i - b_i) * (a_i - b_i) + 4.f * c_i * c_i;
        const float sqrtDelta = std::sqrt(delta);
        const float trace = a_i + b_i;

        Lambda1[i] = 0.5f * (trace + sqrtDelta);
        Lambda2[i] = 0.5f * (trace - sqrtDelta);

        const float x1 = 2.f * c_i;
        const float x2 = b_i - a_i - sqrtDelta;
        const float norm_eig_vect = std::sqrt(x1 * x1 + x2 * x2);

        if (norm_eig_vect > 0.f)
        {
            Eigvect2_x[i] = x1 / norm_eig_vect;
            Eigvect2_y[i] = x2 / norm_eig_vect;
        }
        else
        {
            Eigvect2_x[i] = Eigvect2_y[i] = 0.f;
        }
    }
}