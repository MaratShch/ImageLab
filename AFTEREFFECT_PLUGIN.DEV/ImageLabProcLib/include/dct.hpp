#include <algorithm>
#include <cstdint>
#include <cmath>
#include "Common.hpp"

namespace FourierTransform
{
    template <typename T>
    inline void dct_1D
    (
        const T* RESTRICT in,  // Input: Array of N Real numbers (Pixels)
        T* RESTRICT out,       // Output: Array of N Real numbers (Coefficients)
        int32_t N
    ) noexcept
    {
        constexpr T PI = static_cast<T>(3.14159265358979323846);
        
        // Normalization factors for Orthonormality (Energy Preservation)
        // If you don't do this, the output values will be huge.
        const T alpha0 = static_cast<T>(std::sqrt(static_cast<T>(1.0) / N));
        const T alphaK = static_cast<T>(std::sqrt(static_cast<T>(2.0) / N));

        for (int32_t k = 0; k < N; k++)
        {
            T sum{0};

            // Determine the scaling factor for this frequency k
            const T s = (k == 0) ? alpha0 : alphaK;

            for (int32_t n = 0; n < N; n++)
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
        int32_t N
    ) noexcept
    {
        constexpr T PI = static_cast<T>(3.14159265358979323846);

        // Normalization factors (Identical to Forward)
        const T alpha0 = static_cast<T>(std::sqrt(static_cast<T>(1.0) / N));
        const T alphaK = static_cast<T>(std::sqrt(static_cast<T>(2.0) / N));

        // Outer Loop: Iterate over Output Pixels (n)
        for (int32_t n = 0; n < N; n++)
        {
            T sum{ 0 };

            // Inner Loop: Sum up all Frequencies (k)
            for (int32_t k = 0; k < N; k++)
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
        int32_t src_w,
        int32_t src_h
    ) noexcept
    {
        constexpr int32_t TILE_SIZE = 32;

        for (int32_t y = 0; y < src_h; y += TILE_SIZE)
        {
            for (int32_t x = 0; x < src_w; x += TILE_SIZE)
            {
                // Handle boundaries
                const int32_t block_h = std::min(TILE_SIZE, src_h - y);
                const int32_t block_w = std::min(TILE_SIZE, src_w - x);

                // Transpose the tile
                for (int32_t by = 0; by < block_h; ++by)
                {
                    for (int32_t bx = 0; bx < block_w; ++bx)
                    {
                        // Input index: (y+by) is row, (x+bx) is col
                        int32_t in_idx = (y + by) * src_w + (x + bx);

                        // Output index: (x+bx) becomes row, (y+by) becomes col
                        // Destination stride is 'src_h' (the new width)
                        int32_t out_idx = (x + bx) * src_h + (y + by);

                        dst[out_idx] = src[in_idx];
                    }
                }
            }
        }
        return;
    }

    template <typename T>
    inline void GenerateDCTMatrix (const int32_t N, T* RESTRICT matrix /* array size should be equal or bigger N * N */)
    {
        // Double precision for calculation, cast to float for storage
        constexpr T PI = static_cast<T>(3.14159265358979323846);

        const T invSqrtN = std::sqrt(static_cast<T>(1.0) / N);      // Scale for Row 0
        const T sqrt2N   = std::sqrt(static_cast<T>(2.0) / N);      // Scale for Rows 1..N-1
        const T angleScale = PI / static_cast<T>(N);                // Optimization

        for (int32_t k = 0; k < N; k++) // Rows (Frequency k)
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

    void dct_2D (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out, int32_t width, int32_t height) noexcept;
    void dct_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, int32_t width, int32_t height) noexcept;
     
    void idct_2D (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out, int32_t width, int32_t height) noexcept;
    void idct_2D (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out, int32_t width, int32_t height) noexcept;

    // special DCT cases 
    void dct_2D_8x8 (const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out) noexcept;
    void dct_2D_8x8 (const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out) noexcept;

    void idct_2D_8x8(const float*  RESTRICT in, float*  RESTRICT scratch, float*  RESTRICT out) noexcept;
    void idct_2D_8x8(const double* RESTRICT in, double* RESTRICT scratch, double* RESTRICT out) noexcept;

    void dct_generate_transform_matrix_f32 (const int N, float*  RESTRICT pMatrix);
    void dct_generate_transform_matrix_f64 (const int N, double* RESTRICT pMatrix);
}


