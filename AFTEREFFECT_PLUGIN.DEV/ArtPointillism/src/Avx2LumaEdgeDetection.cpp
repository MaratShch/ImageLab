#include "AlgoLumaManipulation.hpp"

void LumaEdgeDetection
(
    const float* RESTRICT pSrc,
          float* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY
) noexcept
{
    // Safety Check
    if (sizeX < 3 || sizeY < 3)
    {
        std::memcpy(pDst, pSrc, sizeX * sizeY * sizeof(float));
        return;
    }

    const ptrdiff_t stride = sizeX; 
    const size_t row_size_bytes = static_cast<size_t>(sizeX) * sizeof(float);

    // ---------------------------------------------------------
    // 1. TOP BORDER HANDLING
    // ---------------------------------------------------------
    std::memcpy(pDst, pSrc, row_size_bytes);

    // ---------------------------------------------------------
    // 2. CENTRAL PROCESSING (AVX2)
    // ---------------------------------------------------------
    
    // Constant for multiplication by 2.0
    const __m256 c_two = _mm256_set1_ps(2.0f);

    const ptrdiff_t lastLine = static_cast<ptrdiff_t>(sizeY - 1);
    const ptrdiff_t lastPixl = static_cast<ptrdiff_t>(sizeX - 1);

    // We process 8 pixels at a time. 
    // We start at x=1. The vector loop can run as long as (x + 8) < lastPixl.
    // So the limit is sizeX - 1 - 8 = sizeX - 9.
    const ptrdiff_t vecLimit = static_cast<ptrdiff_t>(sizeX) - 9;

    for (ptrdiff_t y = 1; y < lastLine; ++y)
    {
        // Pointers for current sliding window
        const float* row_top = pSrc + (y - 1) * stride;
        const float* row_mid = pSrc + (y    ) * stride;
        const float* row_bot = pSrc + (y + 1) * stride;
        float*       row_dst = pDst + (y    ) * stride;

        // --- [LEFT BORDER] ---
        row_dst[0] = row_mid[0];

        ptrdiff_t x = 1;

        // --- [AVX2 MAIN LOOP] ---
        for (; x <= vecLimit; x += 8)
        {
            // 1. Load Top Row Neighbors (x-1, x, x+1)
            __m256 tl = _mm256_loadu_ps(row_top + x - 1);
            __m256 t  = _mm256_loadu_ps(row_top + x);
            __m256 tr = _mm256_loadu_ps(row_top + x + 1);

            // 2. Load Middle Row Neighbors (x-1, x+1)
            // Center pixel (row_mid[x]) is unused in Sobel convolution
            __m256 l  = _mm256_loadu_ps(row_mid + x - 1);
            __m256 r  = _mm256_loadu_ps(row_mid + x + 1);

            // 3. Load Bottom Row Neighbors (x-1, x, x+1)
            __m256 bl = _mm256_loadu_ps(row_bot + x - 1);
            __m256 b  = _mm256_loadu_ps(row_bot + x);
            __m256 br = _mm256_loadu_ps(row_bot + x + 1);

            // 4. Compute Gx
            // Formula: (tr + 2r + br) - (tl + 2l + bl)
            // Right Side Sum
            __m256 r_sum = _mm256_add_ps(tr, br);
            r_sum        = _mm256_fmadd_ps(r, c_two, r_sum); // r*2 + (tr+br)

            // Left Side Sum
            __m256 l_sum = _mm256_add_ps(tl, bl);
            l_sum        = _mm256_fmadd_ps(l, c_two, l_sum); // l*2 + (tl+bl)

            __m256 Gx    = _mm256_sub_ps(r_sum, l_sum);

            // 5. Compute Gy
            // Formula: (bl + 2b + br) - (tl + 2t + tr)
            // Bottom Side Sum
            __m256 b_sum = _mm256_add_ps(bl, br);
            b_sum        = _mm256_fmadd_ps(b, c_two, b_sum); // b*2 + (bl+br)

            // Top Side Sum
            __m256 t_sum = _mm256_add_ps(tl, tr);
            t_sum        = _mm256_fmadd_ps(t, c_two, t_sum); // t*2 + (tl+tr)

            __m256 Gy    = _mm256_sub_ps(b_sum, t_sum);

            // 6. Compute Magnitude
            // Mag = sqrt(Gx^2 + Gy^2)
            __m256 magSq = _mm256_fmadd_ps(Gx, Gx, _mm256_mul_ps(Gy, Gy));
            __m256 mag   = _mm256_sqrt_ps(magSq);

            // 7. Store Result
            _mm256_storeu_ps(row_dst + x, mag);
        }

        // --- [SCALAR CLEANUP LOOP] ---
        // Handle remaining pixels between vector loop end and right border
        for (; x < lastPixl; ++x)
        {
            float tl = row_top[x - 1];
            float t  = row_top[x    ];
            float tr = row_top[x + 1];

            float l  = row_mid[x - 1];
            float r  = row_mid[x + 1];

            float bl = row_bot[x - 1];
            float b  = row_bot[x    ];
            float br = row_bot[x + 1];

            float Gx = (tr + 2.f * r + br) - (tl + 2.f * l + bl);
            float Gy = (bl + 2.f * b + br) - (tl + 2.f * t + tr);

            row_dst[x] = std::sqrt(Gx * Gx + Gy * Gy);
        }

        // --- [RIGHT BORDER] ---
        row_dst[lastPixl] = row_mid[lastPixl];
    }

    // ---------------------------------------------------------
    // 3. BOTTOM BORDER HANDLING
    // ---------------------------------------------------------
    const float* src_last_row = pSrc + lastLine * stride;
    float*       dst_last_row = pDst + lastLine * stride;

    std::memcpy(dst_last_row, src_last_row, row_size_bytes);

    return;
}