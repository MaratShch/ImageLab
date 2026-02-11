#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include "Common.hpp"
#include "FastAriphmetics.hpp"

namespace FourierTransform
{
    template <typename T>
    inline void dct_1D
    (
        const T* RESTRICT in,  // Input: Array of N Real numbers (Pixels)
        T* RESTRICT out,       // Output: Array of N Real numbers (Coefficients)
        ptrdiff_t N
    ) noexcept
    {
        constexpr T PI = static_cast<T>(3.14159265358979323846);
        
        // Normalization factors for Orthonormality (Energy Preservation)
        // If you don't do this, the output values will be huge.
        const T alpha0 = static_cast<T>(std::sqrt(static_cast<T>(1.0) / N));
        const T alphaK = static_cast<T>(std::sqrt(static_cast<T>(2.0) / N));

        for (ptrdiff_t k = 0; k < N; k++)
        {
            T sum{0};

            // Determine the scaling factor for this frequency k
            const T s = (k == 0) ? alpha0 : alphaK;

            for (ptrdiff_t n = 0; n < N; n++)
            {
                // 1. Read Real Input only
                const T xr = in[n];

                // 2. The DCT-II Angle Formula:
                // (n + 0.5) shifts the sample to the "center" of the pixel
                // This is what prevents the "Edge Jump" artifact.
                const T angle = (PI / static_cast<T>(N)) * (static_cast<T>(n) + static_cast<T>(0.5)) * static_cast<T>(k);

                // 3. Compute Cosine
                // Optimization: In production, these Cosines should be a Lookup Table!
                sum += xr * std::cos(angle); 
            }

            // 4. Write Real Output only
            out[k] = s * sum;
        }
    }

    template <typename T>
    inline void idct_1D
    (
        const T* RESTRICT in,  // Input: Array of N Real numbers (Coefficients)
        T* RESTRICT out,       // Output: Array of N Real numbers (Pixels)
        ptrdiff_t N
    ) noexcept
    {
        constexpr T PI = static_cast<T>(3.14159265358979323846);

        // Normalization factors (Identical to Forward)
        const T alpha0 = static_cast<T>(std::sqrt(static_cast<T>(1.0) / N));
        const T alphaK = static_cast<T>(std::sqrt(static_cast<T>(2.0) / N));

        // Outer Loop: Iterate over Output Pixels (n)
        for (ptrdiff_t n = 0; n < N; n++)
        {
            T sum{ 0 };

            // Inner Loop: Sum up all Frequencies (k)
            for (ptrdiff_t k = 0; k < N; k++)
            {
                // 1. Read Coefficient
                const T Xk = in[k];

                // 2. Determine Scale Factor (s)
                // Note: In Inverse DCT, the scale factor is attached to the 
                // coefficient (basis function), so it applies inside the loop.
                const T s = (k == 0) ? alpha0 : alphaK;

                // 3. The Angle Formula (Identical to Forward)
                // We use the exact same angle because the Cosine matrix is symmetric
                // regarding the position (n) and frequency (k) relationship.
                const T angle = (PI / static_cast<T>(N)) * (static_cast<T>(n) + static_cast<T>(0.5)) * static_cast<T>(k);

                // 4. Summation: Pixel = Sum( Coefficient * Scale * Cos(angle) )
                sum += s * Xk * std::cos(angle);
            }

            // 5. Write Pixel Value
            out[n] = sum;
        }
    }

    template <typename T>
    inline void dct_transpose_block_2D
    (
        const T* RESTRICT src,
        T* RESTRICT dst,
        ptrdiff_t src_w,
        ptrdiff_t src_h
    ) noexcept
    {
        constexpr ptrdiff_t TILE_SIZE = 32;

        for (ptrdiff_t y = 0; y < src_h; y += TILE_SIZE)
        {
            for (ptrdiff_t x = 0; x < src_w; x += TILE_SIZE)
            {
                const ptrdiff_t block_h = std::min(TILE_SIZE, src_h - y);
                const ptrdiff_t block_w = std::min(TILE_SIZE, src_w - x);

                const T* p_src_row = src + y * src_w + x;
                T*       p_dst_col = dst + x * src_h + y;

                for (ptrdiff_t by = 0; by < block_h; ++by)
                {
                     for (ptrdiff_t bx = 0; bx < block_w; ++bx)
                    {
                        p_dst_col[bx * src_h] = p_src_row[bx];
                    }

                    p_src_row += src_w;
                    p_dst_col += 1;
                }
            }
        }
    }

    template <typename T>
    inline void GenerateDCTMatrix (const ptrdiff_t N, T* RESTRICT matrix /* array size should be equal or bigger N * N */)
    {
        // Double precision for calculation, cast to float for storage
        constexpr T PI = static_cast<T>(3.14159265358979323846);

        const T invSqrtN = std::sqrt(static_cast<T>(1.0) / N);      // Scale for Row 0
        const T sqrt2N   = std::sqrt(static_cast<T>(2.0) / N);      // Scale for Rows 1..N-1
        const T angleScale = PI / static_cast<T>(N);                // Optimization

        for (int k = 0; k < N; k++) // Rows (Frequency k)
        {
            // 1. Determine Scale Factor C(k)
           const T scale = (0 == k) ? invSqrtN : sqrt2N;

            for (int n = 0; n < N; n++) // Cols (Pixel n)
            {
                // 2. Compute Angle: (PI/N) * k * (n + 0.5)
                // Note: (2n+1)/2 is the same as (n + 0.5)
                T angle = angleScale * static_cast<T>(k) * (static_cast<T>(n) + static_cast<T>(0.5));

                // 3. Compute and Store Value
                matrix[k * N + n] = scale * std::cos(angle);
            }
        }
        return;
    }

    /**
     * @brief High-performance 8-point 1D Forward DCT-II (Loeffler Algorithm).
     * 
     * Performance: 11 Multiplications, 29 Additions.
     * This function handles the butterfly stages and rotations to transform 
     * spatial data into frequency coefficients.
     * 
     * @param src Input array of 8 floats (spatial pixels).
     * @param dst Output array of 8 floats (frequency coefficients).
     */
    template <typename T>
    inline void dct8x8_Loeffler (const T* RESTRICT src, T* RESTRICT dst) noexcept
    {
        // Stage 1: Initial Butterflies
        T s07 = src[0] + src[7];
        T s16 = src[1] + src[6];
        T s25 = src[2] + src[5];
        T s34 = src[3] + src[4];
        
        T d07 = src[0] - src[7];
        T d16 = src[1] - src[6];
        T d25 = src[2] - src[5];
        T d34 = src[3] - src[4];

        // Stage 2: Even/Odd separation
        // Even part (0, 2, 4, 6)
        T s07_34 = s07 + s34;
        T s16_25 = s16 + s25;
        T d07_34 = s07 - s34;
        T d16_25 = s16 - s25;

        // Odd part (1, 3, 5, 7)
        T d07_plus_34 = d07 + d34;
        T d16_plus_25 = d16 + d25;
        T d07_minus_34 = d07 - d34;
        T d16_minus_25 = d16 - d25;

        // Stage 3: Rotations and Butterflies
        // Even Part Processing
        dst[0] = s07_34 + s16_25; // DC Component
        dst[4] = s07_34 - s16_25;

        // Rotation for indices 2 and 6 (cos(6pi/16) and sin(6pi/16))
        constexpr T c6 = 0.382683432; // cos(6pi/16)
        constexpr T s6 = 0.923879532; // sin(6pi/16)
        dst[2] = d16_25 * c6 + d07_34 * s6;
        dst[6] = d07_34 * c6 - d16_25 * s6;

        // Odd Part Processing (Rotations)
        // Rotation for indices 1 and 7
        constexpr T c1 = 0.980785280; // cos(pi/16)
        constexpr T s1 = 0.195090322; // sin(pi/16)
        constexpr T c3 = 0.831469612; // cos(3pi/16)
        constexpr T s3 = 0.555570233; // sin(3pi/16)
        
        // Loeffler uses a specific factorization for the odd part
        T rotation_left_1  = d07_plus_34  + d16_plus_25;
        T rotation_right_1 = d07_plus_34  - d16_plus_25;
        T rotation_left_2  = d16_minus_25 - d07_minus_34;
        T rotation_right_2 = d16_minus_25 + d07_minus_34;

        constexpr T sqrt2 = 1.414213562;
        T odd_stage_1 = rotation_left_1  * static_cast<T>(0.707106781); // (c4)
        T odd_stage_2 = rotation_right_1 * static_cast<T>(0.707106781); // (c4)

        dst[1] = d07_minus_34 * c1 + d16_minus_25 * s1;
        dst[7] = d16_minus_25 * c1 - d07_minus_34 * s1;
        dst[3] = d16_plus_25 * c3 + d07_plus_34 * s3;
        dst[5] = d07_plus_34 * c3 - d16_plus_25 * s3;

        // Final Normalization for Orthonormal DCT-II
        // For TNC, coefficients must be physically meaningful.
        // Normalized by sqrt(1/N) for DC and sqrt(2/N) for AC.
        constexpr T inv_sqrt8 = 0.353553390; // 1/sqrt(8)
        constexpr T inv_2     = 0.500000000; // 1/2 (actually sqrt(2/8))

        dst[0] *= inv_sqrt8;
        for(int i = 1; i < 8; ++i)
        {
            dst[i] *= inv_2;
        }
        return;
    }


        /**
         * @brief Performs a full 2D 8x8 DCT using the 1D Loeffler implementation.
         * Uses a local stack-based transpose to remain cache-friendly.
         */
        template <typename T>
        inline void dct2D_8x8_Loeffler (const T* RESTRICT src, T* RESTRICT dst) noexcept
        {
            CACHE_ALIGN T intermediate[64];
            CACHE_ALIGN T transposed[64];

            // PASS 1: 1D DCT on all 8 Rows
            for (int i = 0; i < 8; ++i)
            {
                FourierTransform::dct8x8_Loeffler<T>(src + (i * 8), intermediate + (i * 8));
            }

            // PASS 2: Transpose the intermediate result
            // Hardcoded 8x8 transpose is very fast for the CPU
            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 8; ++j)
                {
                    transposed[i * 8 + j] = intermediate[j * 8 + i];
                }
            }

            // PASS 3: 1D DCT on the "Columns" (which are now Rows)
            for (int i = 0; i < 8; ++i)
            {
                FourierTransform::dct8x8_Loeffler<T>(transposed + (i * 8), intermediate + (i * 8));
            }

            // PASS 4: Final Transpose back to standard orientation (u, v)
            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 8; ++j)
                {
                    dst[i * 8 + j] = intermediate[j * 8 + i];
                }
            }
            
            return;
        }
        
 
    /**
     * @brief 8-point 1D Inverse DCT-II (Loeffler).
     * Note: Uses the same orthonormal scaling as the forward pass.
     */
    template <typename T>
    inline void idct8x8_Loeffler(const T* RESTRICT src, T* RESTRICT dst) noexcept
    {
        constexpr T PiDiv8 = FastCompute::PI / static_cast<T>(8.0);
        // Standard IDCT-II definition (Orthonormal)
        for (int x = 0; x < 8; ++x)
        {
            T sum = 0.0;
            for (int u = 0; u < 8; ++u)
            {
                // Orthonormal weights: 1/sqrt(8) for DC, sqrt(2/8) for AC
                const T alpha = (u == 0) ? static_cast<T>(0.353553390) : static_cast<T>(0.500000000);
                
                sum += alpha * static_cast<T>(src[u]) * 
                       std::cos(PiDiv8 * u * (x + static_cast<T>(0.5)));
            }
            
            // NO EXTRA MULTIPLIER HERE. The alpha inside the loop handles the scaling.
            dst[x] = static_cast<T>(sum);
        }
    }
 
    // 2D Wrapper (Transpose -> 1D -> Transpose -> 1D)
     template <typename T>
     inline void idct2D_8x8_Loeffler (const T* RESTRICT src, T* RESTRICT dst) noexcept
     {
        CACHE_ALIGN T intermediate[64];
        CACHE_ALIGN T transposed[64];
        CACHE_ALIGN T final_trans[64];
        
        for (int i = 0; i < 8; ++i)
            idct8x8_Loeffler(src + (i * 8), intermediate + (i * 8));
        
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                transposed[i * 8 + j] = intermediate[j * 8 + i];
            
        for (int i = 0; i < 8; ++i)
            idct8x8_Loeffler(transposed + (i * 8), intermediate + (i * 8));
        
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                dst[i * 8 + j] = intermediate[j * 8 + i];
            
        return;    
     }


    void dct_2D (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept;
    void dct_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept;
     
    void idct_2D (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept;
    void idct_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, ptrdiff_t width, ptrdiff_t height) noexcept;

    // special DCT cases 
    void dct_2D_8x8 (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out) noexcept;
    void dct_2D_8x8 (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out) noexcept;

    void idct_2D_8x8(const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out) noexcept;
    void idct_2D_8x8(const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out) noexcept;

    void dct_generate_transform_matrix_f32 (const int N, float*  RESTRICT pMatrix);
    void dct_generate_transform_matrix_f64 (const int N, double* RESTRICT pMatrix);
}


