#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include "FastAriphmetics.hpp"
#include "PaintCocircularity_graphs.hpp"

// Fast absolute value mask for AVX2 floats
static const __m256i ABS_MASK = _mm256_set1_epi32(0x7FFFFFFF);

A_long bw_image2cocircularity_graph_AVX2_flat
(
    const float* RESTRICT eigX,
    const float* RESTRICT eigY,
    A_long* RESTRICT pI,
    A_long* RESTRICT pJ,
    float* RESTRICT pLogW,
    size_t max_edges,
    A_long width,
    A_long height,
    float coCirc,
    float coCone,
    A_long radius
)
{
    size_t edgeCount = 0;
    
    const __m256 v_coCone = _mm256_set1_ps(coCone);
    const __m256 v_coCirc = _mm256_set1_ps(coCirc);
    const __m256 v_epsilon = _mm256_set1_ps(1e-6f);

    for (A_long y = 0; y < height; ++y)
    {
        A_long startY = std::max<A_long>(0, y - radius);
        A_long endY   = std::min<A_long>(height - 1, y + radius);

        for (A_long x = 0; x < width; ++x)
        {
            const A_long p0 = y * width + x;
            const float v1x = eigX[p0];
            const float v1y = eigY[p0];
            
            const __m256 v_v1x = _mm256_set1_ps(v1x);
            const __m256 v_v1y = _mm256_set1_ps(v1y);
            const __m256 v_x   = _mm256_set1_ps(static_cast<float>(x));

            for (A_long ny = startY; ny <= endY; ++ny)
            {
                // EXACT SCALAR MATCH: Only search the right-half plane to prevent duplicates
                A_long startX = (ny <= y) ? x + 1 : x;
                startX = std::max<A_long>(0, startX);
                A_long endX = std::min<A_long>(width - 1, x + radius);

                const A_long rowOffset = ny * width;
                const float dy = static_cast<float>(ny - y);
                const __m256 v_dy = _mm256_set1_ps(dy);
                const __m256 v_dy2 = _mm256_mul_ps(v_dy, v_dy);

                A_long nx = startX;
                
                // --- AVX2 INNER KERNEL ---
                for (; nx <= endX - 7; nx += 8)
                {
                    __m256 v_nx = _mm256_add_ps(_mm256_set1_ps(static_cast<float>(nx)), 
                                                _mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f));
                    __m256 v_dx = _mm256_sub_ps(v_nx, v_x);
                    
                    __m256 v_distSq = _mm256_add_ps(_mm256_mul_ps(v_dx, v_dx), v_dy2);
                    __m256 v_dist = _mm256_sqrt_ps(_mm256_add_ps(v_distSq, v_epsilon));

                    __m256 v_dx_n = _mm256_div_ps(v_dx, v_dist);
                    __m256 v_dy_n = _mm256_div_ps(v_dy, v_dist);

                    // Conic constraint
                    __m256 v_v1_dot_d = _mm256_add_ps(_mm256_mul_ps(v_v1x, v_dx_n), _mm256_mul_ps(v_v1y, v_dy_n));
                    __m256 v_abs_v1_dot_d = _mm256_and_ps(v_v1_dot_d, _mm256_castsi256_ps(ABS_MASK));
                    __m256 v_maskCone = _mm256_cmp_ps(v_abs_v1_dot_d, v_coCone, _CMP_GE_OQ);

                    // Co-circularity constraint
                    __m256 v_two = _mm256_set1_ps(2.0f);
                    __m256 v_v0x = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(v_two, v_v1_dot_d), v_dx_n), v_v1x);
                    __m256 v_v0y = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(v_two, v_v1_dot_d), v_dy_n), v_v1y);

                    __m256 v_v2x = _mm256_loadu_ps(&eigX[rowOffset + nx]);
                    __m256 v_v2y = _mm256_loadu_ps(&eigY[rowOffset + nx]);

                    __m256 v_cocirc = _mm256_add_ps(_mm256_mul_ps(v_v0x, v_v2x), _mm256_mul_ps(v_v0y, v_v2y));
                    v_cocirc = _mm256_and_ps(v_cocirc, _mm256_castsi256_ps(ABS_MASK));
                    __m256 v_maskCirc = _mm256_cmp_ps(v_cocirc, v_coCirc, _CMP_GE_OQ);

                    // Combine Masks
                    __m256 v_finalMask = _mm256_and_ps(v_maskCone, v_maskCirc);
                    const int mask = _mm256_movemask_ps(v_finalMask);
                    
                    if (mask != 0) 
                    {
                        for (int b = 0; b < 8; ++b)
                        {
                            if (mask & (1 << b)) 
                            {
                                if (edgeCount < max_edges) 
                                {
                                    pI[edgeCount] = p0;
                                    pJ[edgeCount] = rowOffset + nx + b;
                                    pLogW[edgeCount] = 0.0f; // Directly setting Log(1.0) = 0.0f
                                    edgeCount++;
                                }
                            }
                        }
                    }
                }

                // --- SCALAR TAIL ---
                for (; nx <= endX; ++nx)
                {
                    float dx = static_cast<float>(nx - x);
                    float dist = FastCompute::Sqrt(dx*dx + dy*dy + 1e-6f);

                    float dx_n = dx / dist;
                    float dy_n = dy / dist;

                    float v1_dot_d = v1x * dx_n + v1y * dy_n;
                    
                    if (std::abs(v1_dot_d) >= coCone)
                    {
                        float v0x = 2.0f * v1_dot_d * dx_n - v1x;
                        float v0y = 2.0f * v1_dot_d * dy_n - v1y;

                        float v2x = eigX[rowOffset + nx];
                        float v2y = eigY[rowOffset + nx];

                        float cocirc = std::abs(v0x * v2x + v0y * v2y);
                        if (cocirc >= coCirc)
                        {
                            if (edgeCount < max_edges) 
                            {
                                pI[edgeCount] = p0;
                                pJ[edgeCount] = rowOffset + nx;
                                pLogW[edgeCount] = 0.0f; 
                                edgeCount++;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Cast the size_t back to A_long so it cleanly fits into your nonZeros variable!
    return static_cast<A_long>(edgeCount);
}